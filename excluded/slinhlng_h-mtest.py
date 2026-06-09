%%time
import os
import sys
import copy
from datetime import datetime
import gc
import pickle as pkl
import shelve

import pandas as pd
import numpy as np
import cudf

sys.path.append("../input/")
from handmhelpers import io as h_io, sub as h_sub, cv as h_cv, fe as h_fe
from handmhelpers import modeling as h_modeling, candidates as h_can, pairs as h_pairs


from datetime import timedelta
import cudf
import numpy as np

def patched_day_week_numbers(dates: cudf.Series):
    pd_dates = cudf.to_datetime(dates)
    unique_dates = cudf.Series(pd_dates.unique())
    numbered_days = unique_dates - unique_dates.min() + timedelta(1)
    numbered_days = numbered_days.dt.days
    extra_days = numbered_days.max() % 7
    numbered_days -= extra_days
    day_weeks = (numbered_days + 6) // 7  # không dùng applymap
    day_weeks_map = cudf.DataFrame({"day_weeks": day_weeks, "unique_dates": unique_dates}).set_index("unique_dates")["day_weeks"]
    all_day_weeks = pd_dates.map(day_weeks_map).astype("int8")
    return all_day_weeks

import handmhelpers.fe as h_fe
h_fe.day_week_numbers = patched_day_week_numbers


%%time

c, t, a = h_io.load_data(files=['customers.csv', 'transactions_train.csv', 'articles.csv'])        

index_to_id_dict_path = h_fe.reduce_customer_id_memory(c, [t])
t["week_number"] = h_fe.day_week_numbers(t["t_dat"])
t["t_dat"] = h_fe.day_numbers(t["t_dat"])


%%time
# Tạo cặp bài viết cho nhiều tuần lịch sử (94–104) với 15 cặp mỗi bài để nâng recall ứng viên
pairs_per_item = 15

week_number_pairs = {}
for week_number in [94,95,96, 97, 98, 99, 100, 101, 102, 103, 104]:
    print(f"Creating pairs for week number {week_number}")
    week_number_pairs[week_number] = h_pairs.create_pairs(
        t, week_number, pairs_per_item, verbose=False
    )


def create_candidates_with_features_df(t, c, a, customer_batch=None, **kwargs):
    # Tách dữ liệu theo tuần label: features dùng các tuần trước, label là tuần đích
    features_df, label_df = h_cv.feature_label_split(
        t, kwargs["label_week"], kwargs["feature_periods"]
    )
    # Chuẩn hóa thời gian về “cách đây bao nhiêu ngày/tuần”
    features_df["t_dat"] = h_fe.how_many_ago(features_df["t_dat"])
    features_df["week_number"] = h_fe.how_many_ago(features_df["week_number"])

    article_pairs_df = week_number_pairs[kwargs["label_week"] - 1]

    # Xác định tập khách hàng cần xử lý (full/batch/label)
    if len(label_df) > 0:
        customers = label_df["customer_id"].unique()
    elif customer_batch is not None:
        customers = customer_batch
    else:
        customers = None

    # ----- Ứng viên từ nhiều nguồn và đặc trưng rule -----
    features_db = {}
    recent_customer_cand, features_db["customer_article"] = h_can.create_recent_customer_candidates(
        features_df, kwargs["ca_num_weeks"], customers=customers
    )
    (
        cust_last_week_cand,
        cust_last_week_pair_cand,
        features_db["clw"],
        features_db["clw_pairs"],
    ) = h_can.create_last_customer_weeks_and_pairs(
        features_df, article_pairs_df, kwargs["clw_num_weeks"], kwargs["clw_num_pair_weeks"], customers=customers
    )
    popular_can, features_db["popular_articles"] = h_can.create_popular_article_cand(
        features_df, c, a, kwargs["pa_num_weeks"], kwargs["hier_col"],
        num_candidates=kwargs["num_recent_candidates"],
        num_articles=kwargs["num_recent_articles"],
        customers=customers,
    )
    (age_bucket_can, _, age_bucket_pair_features) = h_can.create_age_bucket_candidates(
        features_df, c, kwargs["num_age_buckets"], articles=kwargs["num_recent_articles"], customers=customers
    )
    features_db["age_bucket"] = age_bucket_pair_features

    # NEW: ứng viên random walk
    random_walk_cand, features_db["f_random_walk"] = h_can.create_random_walk_candidates(
        features_df,
        article_pairs_df,
        seed_weeks=kwargs["gd_seed_weeks"],
        seed_articles=kwargs["gd_seed_articles"],
        num_steps=kwargs["gd_steps"],
        restart_prob=kwargs["gd_restart_prob"],
        topk=kwargs["gd_topk"],
        weight_col=kwargs["gd_weight_col"],
        recency_weight=kwargs["gd_recency_weight"],
        exclude_seed_items=kwargs["gd_exclude_seed"],
        customers=customers,
    )

    # Gom rule-score từ từng nguồn
    def build_rule_part(cand_df, feature_tuple, score_col, rule_name):
        feature_df = feature_tuple[1].reset_index()[["customer_id", "article_id", score_col]]
        tmp = cand_df.merge(feature_df, on=["customer_id", "article_id"], how="left")
        tmp = tmp.rename(columns={score_col: "rule_score"})
        tmp["rule_score"] = tmp["rule_score"].fillna(-1)
        tmp["rule"] = rule_name
        return tmp[["customer_id", "article_id", "rule", "rule_score"]]

    rule_parts = [
        build_rule_part(recent_customer_cand, features_db["customer_article"], "ca_purchase_count", "recent"),
        build_rule_part(cust_last_week_cand, features_db["clw"], "ca_count", "last_weeks"),
        build_rule_part(cust_last_week_pair_cand, features_db["clw_pairs"], "pair_lift", "pairs"),
        build_rule_part(age_bucket_can, features_db["age_bucket"], "article_bucket_count", "age_bucket"),
    ]
    # rule popular
    pop_feature_df = features_db["popular_articles"][1].reset_index()[["article_id", "recent_popularity_counts"]]
    pop_rule = popular_can.merge(pop_feature_df, on="article_id", how="left")
    pop_rule = pop_rule.rename(columns={"recent_popularity_counts": "rule_score"})
    pop_rule["rule_score"] = pop_rule["rule_score"].fillna(-1)
    pop_rule["rule"] = "popular"
    rule_parts.append(pop_rule[["customer_id", "article_id", "rule", "rule_score"]])
    # rule random_walk
    rule_parts.append(
        build_rule_part(random_walk_cand, features_db["f_random_walk"], "rw_score", "random_walk")
    )

    rule_df = cudf.concat(rule_parts).sort_values(
        ["rule", "customer_id", "rule_score"], ascending=[True, True, False]
    )
    rule_df["rank_within_rule"] = rule_df.groupby(["rule", "customer_id"]).cumcount()

    rule_features_df = (
        rule_df.groupby(["customer_id", "article_id"])
        .agg({"rule": "nunique", "rule_score": "max", "rank_within_rule": "min"})
        .reset_index()
    )
    rule_features_df.columns = ["customer_id", "article_id", "n_sources", "best_rule_score", "best_rank_within_rule"]

    # Thêm cờ nguồn
    for rule_name in rule_df["rule"].unique().to_pandas():
        flag_df = rule_df[rule_df["rule"] == rule_name][["customer_id", "article_id"]].drop_duplicates()
        flag_df[f"{rule_name}_flag"] = 1
        rule_features_df = rule_features_df.merge(flag_df, how="left", on=["customer_id", "article_id"])
        rule_features_df[f"{rule_name}_flag"] = rule_features_df[f"{rule_name}_flag"].fillna(0).astype("int8")

    # Hợp nhất ứng viên (đã thêm random_walk)
    cand = cudf.concat([
        popular_can,
        recent_customer_cand,
        cust_last_week_cand,
        cust_last_week_pair_cand,
        age_bucket_can,
        random_walk_cand,
    ]).drop_duplicates().sort_values(["customer_id", "article_id"]).reset_index(drop=True)
    del popular_can, recent_customer_cand, cust_last_week_cand, cust_last_week_pair_cand, age_bucket_can, random_walk_cand

    cand = h_can.filter_candidates(cand, t, **kwargs)

    # ----- Sinh thêm đặc trưng -----
    h_fe.create_cust_hier_features(features_df, a, kwargs["hier_cols"], features_db)
    h_fe.create_cust_hier_decay_features(features_df, a, kwargs["hier_cols"], features_db,
                                         decay_gamma=kwargs.get("hier_decay_gamma", 0.3))
    for k, v in list(features_db.items()):
        if k.endswith("_decay_features"):
            hier = k[len("cust_"):-len("_decay_features")]
            cols, df = v
            df = df.rename(columns={"last_seen_category_weeks_ago": f"last_seen_{hier}_weeks_ago"})
            features_db[k] = (cols, df)

    h_fe.create_price_features(features_df, features_db)
    h_fe.create_cust_features(c, features_db)
    h_fe.create_article_cust_features(features_df, c, features_db)
    h_fe.create_lag_features(features_df, a, kwargs["lag_days"], features_db)
    h_fe.create_rebuy_features(features_df, features_db)
    h_fe.create_cust_t_features(features_df, a, features_db)
    try:
        h_fe.create_art_t_features(features_df, a, features_db)
    except TypeError:
        h_fe.create_art_t_features(features_df, features_db)
    del features_df

    if customers is not None:
        cand = cand[cand["customer_id"].isin(customers)]

    if kwargs["cv"]:
        ground_truth_candidates = label_df[["customer_id", "article_id"]].drop_duplicates()
        h_cv.report_candidates(cand, ground_truth_candidates)
        del ground_truth_candidates

    cand_with_f_df = h_can.add_features_to_candidates(cand, features_db, c, a)
    cand_with_f_df = cand_with_f_df.merge(rule_features_df, how="left", on=["customer_id", "article_id"])

    for article_col in kwargs["article_columns"]:
        art_col_map = a.set_index("article_id")[article_col]
        cand_with_f_df[article_col] = cand_with_f_df["article_id"].map(art_col_map)

    # Fill giá trị rule cho ứng viên mới (có random_walk_flag)
    rule_fill = {
        "n_sources": 0,
        "best_rule_score": -1,
        "best_rank_within_rule": 127,
        "recent_flag": 0,
        "last_weeks_flag": 0,
        "pairs_flag": 0,
        "age_bucket_flag": 0,
        "popular_flag": 0,
        "random_walk_flag": 0,
    }
    cand_with_f_df = cand_with_f_df.fillna(rule_fill)

    # Target encode các cột category dựa trên best_rule_score (nếu có)
    target_col = "best_rule_score" if "best_rule_score" in cand_with_f_df.columns else None
    for col in cand_with_f_df.columns:
        if col in ["customer_id", "article_id"]:
            continue
        if str(cand_with_f_df[col].dtype) not in ["int8","int16","int32","int64","float16","float32","float64","bool"]:
            if target_col:
                te = cand_with_f_df.groupby(col)[target_col].mean()
                cand_with_f_df[col] = cand_with_f_df[col].map(te).fillna(0).astype("float32")
            else:
                cand_with_f_df[col] = cand_with_f_df[col].astype("category").cat.codes.astype("float32")

    if kwargs["selected_features"] is not None:
        cand_with_f_df = cand_with_f_df[["customer_id", "article_id"] + kwargs["selected_features"]]

    assert len(cand) == len(cand_with_f_df), "seem to have duplicates in the feature dfs"
    del cand

    return cand_with_f_df, label_df



def calculate_model_score(ids_df, preds, truth_df):
    predictions = h_modeling.create_predictions(ids_df, preds)
    true_labels = h_cv.ground_truth(truth_df).set_index("customer_id")["prediction"]
    score = round(h_cv.comp_average_precision(true_labels, predictions),5)
    
    return score


# Cấu hình chạy CV (đa tuần, tăng ứng viên/estimators + random walk)
cv_params = {
    "cv": True,                     # bật báo cáo recall ứng viên
    "feature_periods": 105,         # dùng 105 tuần lịch sử cho feature
    "label_week": 104,              # tuần label mặc định, sẽ bị override bởi cv_weeks
    "index_to_id_dict_path": index_to_id_dict_path,
    "pairs_file_version": "_v3_5_ex",
    "num_recent_candidates": 80,    # tăng số ứng viên recent để nâng recall
    "num_recent_articles": 20,      # tăng bài phổ biến/age bucket
    "hier_col": "department_no",
    "ca_num_weeks": 3,
    "clw_num_weeks": 12,
    "clw_num_pair_weeks": 2,
    "pa_num_weeks": 2,              # kéo dài phổ biến thêm 2 tuần
    "num_age_buckets": 4,
    "filter_recent_art_weeks": 1,
    "filter_num_articles": None,
    "lag_days": [1, 3, 7, 14, 28],
    "article_columns": ["index_code"],
    "hier_cols": [
        "department_no", "section_no", "index_group_no", "index_code",
        "product_type_no", "product_group_name"
    ],
    "hier_decay_gamma": 0.3,
    "selected_features": None,
    # random walk with restart candidates
    "gd_seed_weeks": 12,
    "gd_seed_articles": 12,
    "gd_steps": 2,
    "gd_restart_prob": "adaptive",
    "gd_topk": 24,
    "gd_weight_col": "customer_count",
    "gd_recency_weight": True,
    "gd_exclude_seed": True,
    "lgbm_params": {                # tăng capacity model cho CV
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    },
    "log_evaluation": 10,
    "early_stopping": 30,
    "eval_at": 12,
    "save_model": True,
    "num_concats": 5,               # ghép 5 tuần train để tăng dữ liệu
}

# Cấu hình train/predict submit (n_estimators cao hơn, ensemble 2 model + random walk)
sub_params = {
    "cv": False,
    "feature_periods": 105,
    "label_week": 105,
    "index_to_id_dict_path": index_to_id_dict_path,
    "pairs_file_version": "_v3_5_ex",
    "num_recent_candidates": 80,
    "num_recent_articles": 20,
    "hier_col": "department_no",
    "ca_num_weeks": 3,
    "clw_num_weeks": 12,
    "clw_num_pair_weeks": 2,
    "pa_num_weeks": 2,
    "num_age_buckets": 4,
    "filter_recent_art_weeks": 1,
    "filter_num_articles": None,
    "lag_days": [1, 3, 7, 14, 28],
    "article_columns": ["index_code"],
    "hier_cols": [
        "department_no", "section_no", "index_group_no", "index_code",
        "product_type_no", "product_group_name"
    ],
    "hier_decay_gamma": 0.3,
    "selected_features": None,
    # random walk with restart candidates
    "gd_seed_weeks": 12,
    "gd_seed_articles": 12,
    "gd_steps": 2,
    "gd_restart_prob": "adaptive",
    "gd_topk": 24,
    "gd_weight_col": "customer_count",
    "gd_recency_weight": True,
    "gd_exclude_seed": True,
    "lgbm_params": {                # nhiều cây hơn cho submit
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    },
    "log_evaluation": 10,
    "eval_at": 12,
    "prediction_models": ["model_104", "model_105"],  # ensemble 2 model
    "save_model": True,
    "num_concats": 5,
}



cand_features_func = create_candidates_with_features_df
scoring_func = calculate_model_score


%%time

# Chạy cross-validation cho các tuần 102–104 (đa tuần để ổn định hơn so với 1 tuần)
cv_weeks = [102, 103, 104]
results = h_modeling.run_all_cvs(
    t, c, a, cand_features_func, scoring_func,
    cv_weeks=cv_weeks, **cv_params
)


from cuml.fil import ForestInference as _FI

_real_load = _FI.load

def _load_compat(*args, output_class=None, is_classifier=None, **kwargs):
    # helper cũ truyền output_class -> map sang is_classifier
    if is_classifier is None and output_class is not None:
        is_classifier = output_class
    return _real_load(*args, is_classifier=is_classifier, **kwargs)

_FI.load = staticmethod(_load_compat)


import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Parameter `output_class` was deprecated.*",
    category=FutureWarning,
)


import math
import handmhelpers.modeling as h_modeling

def full_sub_predict_run_small_batches(t, c, a, cand_features_func, batch_splits=8, **kwargs):
    # Chia khách hàng thành nhiều batch nhỏ để tránh OOM khi tạo feature/predict
    customer_batches = []
    n = len(c)
    for i in range(batch_splits):
        start = i * n // batch_splits
        end = (i + 1) * n // batch_splits
        customer_batches.append(c[start:end]["customer_id"].to_pandas().to_list())

    batch_preds = []
    for idx, customer_batch in enumerate(customer_batches):
        print(f"generating candidates/features for batch #{idx+1} of {len(customer_batches)}")
        sub_ids_df, sub_X = h_modeling.prepare_prediction_dfs(
            t, c, a, cand_features_func, customer_batch=customer_batch, **kwargs
        )
        print(f"candidate/features shape of batch: ({sub_X.shape[0]:,}, {sub_X.shape[1]})")

        # Ensemble các model trong prediction_models, trung bình điểm
        model_paths = kwargs.get("prediction_models")
        model_nums = len(model_paths)
        first_model = h_modeling.ForestInference.load(model_paths[0], model_type="lightgbm", output_class=False)
        sub_pred = h_modeling.pred_in_batches(first_model, sub_X) / model_nums
        del first_model

        for mp in model_paths[1:]:
            m = h_modeling.ForestInference.load(mp, model_type="lightgbm", output_class=False)
            sub_pred += h_modeling.pred_in_batches(m, sub_X) / model_nums
            del m

        batch_preds.append(h_modeling.create_predictions(sub_ids_df, sub_pred))
        del sub_ids_df, sub_X, sub_pred

    return cudf.concat(batch_preds)

# Ghi đè hàm predict gốc để dùng bản chia batch nhỏ
h_modeling.full_sub_predict_run = full_sub_predict_run_small_batches


%%time
gc.collect()
# Train full dữ liệu submit và lưu model theo sub_params
h_modeling.full_sub_train_run(t, c, a, cand_features_func, scoring_func, **sub_params)
# Predict theo batch nhỏ (batch_splits=8) để tránh OOM, dùng ensemble model_104/model_105
predictions = h_modeling.full_sub_predict_run(
    t, c, a, cand_features_func, batch_splits=8, **sub_params
)


sub = h_sub.create_sub(c["customer_id"], predictions, index_to_id_dict_path)
sub.to_csv('dev_submission.csv', index=False)

display(sub.head())
print(sub.shape)

