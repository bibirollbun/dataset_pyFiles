import os
import warnings
import pandas as pd
import numpy as np
import optuna
import holidays
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import sys
sys.path.append("/kaggle/input/russian-car-plates-prices-prediction/")
warnings.filterwarnings('ignore')
from lightgbm import early_stopping, log_evaluation


from supplemental_english import REGION_CODES, GOVERNMENT_CODES


train_raw = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv", dtype={"id": int, "plate": str}, parse_dates=["date"])
test_raw  = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv",  dtype={"id": int, "plate": str}, parse_dates=["date"])
sample_sub = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv")


train_raw


lower_q = train_raw["price"].quantile(0.02)
upper_q = train_raw["price"].quantile(0.98)
mask = (train_raw["price"] >= lower_q) & (train_raw["price"] <= upper_q)
train_raw = train_raw[mask].reset_index(drop=True)


train_raw


def smape_np(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100


REGION_LOOKUP = {
    int(code): region_name
    for region_name, codes in REGION_CODES.items()
    for code in codes
}


def extract_plate_features(df):
    df["plate_series1"] = df["plate"].str[0]
    df["plate_series2"] = df["plate"].str[4:6]
    df["plate_series"] = df["plate_series1"] + df["plate_series2"]
    df["plate_regst_code"] = df["plate"].str[1:4].astype(int)
    df["plate_region_code"] = df["plate"].str[6:].astype(int)
    df.drop(columns=["plate_series1", "plate_series2"], inplace=True)
    return df


def add_region(df):
    df["region"] = df["plate_region_code"].map(REGION_LOOKUP).fillna("Unknown")
    return df


def build_gov_df():
    rows = []
    for (plate_series, (r_start, r_end), region_code), (description, forbidden, advantage, significance) in GOVERNMENT_CODES.items():
        rows.append([
            plate_series,
            int(r_start),
            int(r_end),
            int(region_code),
            description,
            forbidden,
            advantage,
            significance
        ])
    gov = pd.DataFrame(
        rows,
        columns=[
            "plate_series", "range_start", "range_end", "plate_region_code",
            "description", "forbidden_to_buy", "has_advantage", "significance_level"
        ]
    )
    gov["plate_series"] = gov["plate_series"].astype(str)
    gov = gov.loc[gov.index.repeat(gov["range_end"] - gov["range_start"] + 1)].copy()
    gov["plate_regst_code"] = (
        gov.groupby(["plate_series", "plate_region_code"])
           .cumcount()
        + gov["range_start"]
    )
    gov.drop(columns=["range_start", "range_end"], inplace=True)
    gov["govt_vehicle"] = 1
    return gov

gov_df = build_gov_df()


gov_df


def merge_government_features(df):
    df = df.merge(
        gov_df,
        how="left",
        on=["plate_series", "plate_regst_code", "plate_region_code"]
    )
    return df.fillna({
        "description":       "No Description",
        "forbidden_to_buy":  0,
        "has_advantage":     0,
        "significance_level":0,
        "govt_vehicle":      0
    })


ru_holidays = holidays.Russia()

def add_holiday_feature(df):
    df["is_holiday"] = df["date"].apply(lambda d: int(d in ru_holidays))
    return df

def extract_time_features(df):
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["day"]     = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    initial = df.groupby("plate")["date"].transform("min")
    df["days_from_initial_listing"]   = (df["date"] - initial).dt.days
    df["months_from_initial_listing"] = (df["days_from_initial_listing"] // 30).astype(int)
    df["years_from_initial_listing"]  = (df["months_from_initial_listing"] / 12).astype(float)

    df["year_end"]   = (df["month"] == 12).astype(int)
    df["listing_num"] = df.groupby("plate")["date"].rank(method="dense").astype(int)
    df.drop(columns=["date"], inplace=True)
    return df

def additional_plate_features(df):
    df["is_repeating_digits"] = df["plate"].str[1:4].apply(lambda x: int(len(set(x)) == 1))
    seqs = {"123","234","345","456","567","678","789","321","432"}
    df["is_sequential_digits"] = df["plate"].str[1:4].apply(lambda x: int(x in seqs))
    return df

def add_region_avg_price(df, region_avg_dict):
    df["region_avg_price"] = df["region"].map(region_avg_dict).fillna(0)
    df["region_avg_price_log"] = np.log1p(df["region_avg_price"])
    return df

def add_total_listings(df):
    df["total_listings_for_plate"] = df.groupby("plate")["plate"].transform("count")
    return df

def add_target_encoding(train_df, test_df):
    te_series = train_df.groupby("plate_series")["price_log"].mean().to_dict()
    train_df["plate_series_te"] = train_df["plate_series"].map(te_series)
    test_df["plate_series_te"]  = test_df["plate_series"].map(te_series).fillna(train_df["price_log"].mean())

    te_region = train_df.groupby("region")["price_log"].mean().to_dict()
    train_df["region_te"] = train_df["region"].map(te_region)
    test_df["region_te"]  = test_df["region"].map(te_region).fillna(train_df["price_log"].mean())
    return train_df, test_df

def add_frequency_encoding(train_df, test_df):
    freq_series = train_df["plate_series"].value_counts().to_dict()
    train_df["plate_series_freq"] = train_df["plate_series"].map(freq_series)
    test_df["plate_series_freq"]  = test_df["plate_series"].map(freq_series).fillna(0)

    freq_region = train_df["region"].value_counts().to_dict()
    train_df["region_freq"] = train_df["region"].map(freq_region)
    test_df["region_freq"]  = test_df["region"].map(freq_region).fillna(0)
    return train_df, test_df

def preprocess(df, is_train=True, region_avg_dict=None):
    df = extract_plate_features(df)
    df = add_region(df)
    df = merge_government_features(df)
    df = add_holiday_feature(df)
    df = extract_time_features(df)
    df = additional_plate_features(df)
    df = add_region_avg_price(df, region_avg_dict)
    df = add_total_listings(df)
    if is_train:
        df["price_log"] = np.log1p(df["price"])
    df.drop(columns=["plate"], inplace=True, errors="ignore")
    return df

train_raw["region"] = train_raw["plate"].str[6:].astype(int).map(REGION_LOOKUP).fillna("Unknown")
region_avg_dict = train_raw.groupby("region")["price"].mean().to_dict()

train = preprocess(train_raw.copy(), is_train=True, region_avg_dict=region_avg_dict)
test  = preprocess(test_raw.copy(),  is_train=False, region_avg_dict=region_avg_dict)

train, test = add_target_encoding(train, test)
train, test = add_frequency_encoding(train, test)


train


features = [
    "plate_series",
    "plate_regst_code",
    "plate_region_code",
    "region",
    "forbidden_to_buy",
    "has_advantage",
    "significance_level",
    "govt_vehicle",
    "is_repeating_digits",
    "is_sequential_digits",
    "year",
    "month",
    "day",
    "weekday",
    "is_weekend",
    "days_from_initial_listing",
    "months_from_initial_listing",
    "years_from_initial_listing",
    "year_end",
    "listing_num",
    "total_listings_for_plate",
    "region_avg_price_log",
    "is_holiday",
    "plate_series_te",
    "region_te",
    "plate_series_freq",
    "region_freq"
]

categorical_features = [
    features.index("plate_series"),
    features.index("region")
]

X = train[features]
y = train["price_log"]


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 2000, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "MAE",
        "eval_metric":   "SMAPE",
        "random_seed":   42,
        "verbose":       0
    }

    y_binned = pd.qcut(y, 5, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smape_scores = []

    for train_idx, val_idx in skf.split(X, y_binned):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        model_ = CatBoostRegressor(**params)
        model_.fit(
            X_tr, y_tr,
            cat_features=categorical_features,
            eval_set=(X_vl, y_vl),
            early_stopping_rounds=50
        )

        preds_log = model_.predict(X_vl)
        smape_scores.append(smape_np(y_vl, preds_log))

    return np.mean(smape_scores)



study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=80, timeout=3600)

print("Best parameters from Optuna:")
print(study.best_params)
print(f"Best CV SMAPE: {study.best_value:.4f}%")

best_params = study.best_params
best_params.update({
    "loss_function": "MAE",
    "eval_metric":   "SMAPE",
    "random_seed":   42,
    "verbose":       0
})

final_model = CatBoostRegressor(**best_params)
final_model.fit(
    X, y,
    cat_features=categorical_features
)

train_lgb = train.copy()
test_lgb  = test.copy()
for col in ["plate_series", "region"]:
    train_lgb[col] = train_lgb[col].astype("category").cat.codes
    test_lgb[col]  = test_lgb[col].astype("category").cat.codes

X_lgb = train_lgb[features]
y_lgb = train_lgb["price_log"]


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_cat = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
test_pred_cat = np.zeros(len(test))
test_pred_lgb = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    cat_fold = CatBoostRegressor(**best_params)
    cat_fold.fit(
        X_tr, y_tr,
        cat_features=categorical_features,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    oof_cat[va_idx] = cat_fold.predict(X_va)
    test_pred_cat += cat_fold.predict(test[features]) / kf.n_splits


    lgb_fold = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=-1,          
        num_leaves=31,         
        reg_alpha=best_params["l2_leaf_reg"]
    )


    X_tr_lgb = X_lgb.iloc[tr_idx]
    y_tr_lgb = y_lgb.iloc[tr_idx]
    X_va_lgb = X_lgb.iloc[va_idx]
    y_va_lgb = y_lgb.iloc[va_idx]


    lgb_fold.fit(
        X_tr_lgb, y_tr_lgb,
        eval_set=[(X_va_lgb, y_va_lgb)],
        eval_metric="rmse",
        callbacks=[
           early_stopping(stopping_rounds=50),
           log_evaluation(0)
                  ]
    )


    oof_lgb[va_idx] = lgb_fold.predict(X_va_lgb)
    test_pred_lgb += lgb_fold.predict(test_lgb[features]) / kf.n_splits
    
print("OOF SMAPE CatBoost:", smape_np(y, oof_cat))
print("OOF SMAPE LightGBM:", smape_np(y, oof_lgb))


oof_stack = np.vstack([oof_cat, oof_lgb]).T
stack_model = Ridge(alpha=1.0)
stack_model.fit(oof_stack, y)
weights = stack_model.coef_
print("Stacking weights:", weights)

test_stack = np.vstack([test_pred_cat, test_pred_lgb]).T
test_preds_log = stack_model.predict(test_stack)
test_preds = np.expm1(test_preds_log)

submission = pd.DataFrame({
    "id":    test_raw["id"],
    "price": test_preds
})
submission.to_csv("submission_final.csv", index=False)
print("Final stacked submission saved as submission_final.csv")

train_preds_log = stack_model.predict(oof_stack)
print(f"Final OOF SMAPE (stacked): {smape_np(y, train_preds_log):.4f}%")


submission_final = pd.read_csv("/kaggle/working/submission_final.csv")
submission_final["price"] = submission_final["price"].round(0).astype(int)
submission_final.to_csv("submission_final.csv", index=False)


top5 = submission_final.sort_values(by="price", ascending=False).head(5)
print(top5)

