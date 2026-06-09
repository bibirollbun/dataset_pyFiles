!pip install polars
!pip install xgboost


import os
import math
import gc
from typing import List, Dict, Tuple

import numpy as np
import polars as pl
import xgboost as xgb

from sklearn.model_selection import GroupKFold



# -----------------------------
# Config
# -----------------------------
DATA_DIR = "/kaggle/input/aeroclub-recsys-2025"
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "test.parquet")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Outer data split
HOLDOUT_FRAC = 0.05   # validator_ share
N_FOLDS       = 10     # folds on the remaining 95%
USE_FOLD      = 0     # train on fold==0 to start

# Target encoding (m-estimate smoothing)
TE_PRIOR_M = 50.0





# -----------------------------
# Columns / helpers
# -----------------------------
BASIC_CAT_COLS = [
    "nationality", "searchRoute", "corporateTariffCode", "bySelf", "sex", "companyID",
    # leg 0 segments 0-1
    "legs0_segments0_aircraft_code", "legs0_segments0_arrivalTo_airport_city_iata",
    "legs0_segments0_arrivalTo_airport_iata", "legs0_segments0_departureFrom_airport_iata",
    "legs0_segments0_marketingCarrier_code", "legs0_segments0_operatingCarrier_code",
    "legs0_segments0_flightNumber",
    "legs0_segments1_aircraft_code", "legs0_segments1_arrivalTo_airport_city_iata",
    "legs0_segments1_arrivalTo_airport_iata", "legs0_segments1_departureFrom_airport_iata",
    "legs0_segments1_marketingCarrier_code", "legs0_segments1_operatingCarrier_code",
    "legs0_segments1_flightNumber",
    # leg 1 segments 0-1
    "legs1_segments0_aircraft_code", "legs1_segments0_arrivalTo_airport_city_iata",
    "legs1_segments0_arrivalTo_airport_iata", "legs1_segments0_departureFrom_airport_iata",
    "legs1_segments0_marketingCarrier_code", "legs1_segments0_operatingCarrier_code",
    "legs1_segments0_flightNumber",
    "legs1_segments1_aircraft_code", "legs1_segments1_arrivalTo_airport_city_iata",
    "legs1_segments1_arrivalTo_airport_iata", "legs1_segments1_departureFrom_airport_iata",
    "legs1_segments1_marketingCarrier_code", "legs1_segments1_operatingCarrier_code",
    "legs1_segments1_flightNumber",
]
CARRIER_COLS = [
    "legs0_segments0_marketingCarrier_code",
    "legs1_segments0_marketingCarrier_code",
]
EXCLUDE_COLS = set([
    "Id", "ranker_id", "selected", "profileId", "requestDate",
    "legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt",
    "miniRules0_percentage", "miniRules1_percentage",
    "pricingInfo_passengerCount",
])


# -----------------------------
# IO
# -----------------------------
def read_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_parquet(TRAIN_PATH)
    test  = pl.read_parquet(TEST_PATH)
    for c in ["Id", "ranker_id"]:
        if c not in train.columns or c not in test.columns:
            raise ValueError(f"Missing required column `{c}` in train or test.")
    if "selected" not in train.columns:
        raise ValueError("Train must contain `selected`.")
    return train, test

# -----------------------------
# datetime handling for requestDate
# -----------------------------
def _ensure_datetime_column(df: pl.DataFrame, col: str, out_col: str) -> pl.DataFrame:
    if col not in df.columns:
        return df.with_columns(pl.lit(None).alias(out_col))
    dtype = df.schema[col]
    dtype_str = str(dtype)
    if dtype_str.startswith("Utf8"):
        return df.with_columns(pl.col(col).str.to_datetime(strict=False).alias(out_col))
    elif "Datetime" in dtype_str:
        return df.with_columns(pl.col(col).cast(pl.Datetime).alias(out_col))
    elif "Date" in dtype_str:
        return df.with_columns(pl.col(col).cast(pl.Datetime).alias(out_col))
    else:
        # last-resort cast
        return df.with_columns(pl.col(col).cast(pl.Datetime, strict=False).alias(out_col))


# -----------------------------
# Feature engineering (label-free)
# -----------------------------
def dur_to_min(expr: pl.Expr) -> pl.Expr:
    """
    Robustly convert duration-like fields to minutes.
    - If the column is already numeric => return as-is.
    - Else parse strings like "D.HH:MM:SS" or "HH:MM:SS".
    """
    # If numeric, keep it (cast ensures we don't crash on non-numeric)
    num = expr.cast(pl.Float64, strict=False)

    # Parse string forms
    s = expr.cast(pl.Utf8, strict=False)
    # Remove leading "D." (days prefix) when present, but also keep 'days' separately
    t = pl.when(s.str.contains(r"^\d+\.", literal=True)).then(s.str.replace(r"^\d+\.", "")).otherwise(s)
    days    = s.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
    hours   = t.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
    minutes = t.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)

    parsed = (days + hours + minutes).cast(pl.Float64)

    return pl.when(num.is_not_null()).then(num).otherwise(parsed).fill_null(0)


def _ensure_datetime_column(df: pl.DataFrame, col: str, out_col: str) -> pl.DataFrame:
    if col not in df.columns:
        return df.with_columns(pl.lit(None).alias(out_col))
    # robustly cast string/Date/Datetime -> Datetime without schema errors
    return df.with_columns(pl.col(col).cast(pl.Datetime, strict=False).alias(out_col))


def session_time_split(train: pl.DataFrame, holdout_frac=0.05, n_folds=5, seed=42):
    if "ranker_id" not in train.columns:
        raise ValueError("ranker_id is required.")
    if "requestDate" in train.columns:
        df = _ensure_datetime_column(train.select(["ranker_id", "requestDate"]), "requestDate", "requestDate_dt")
        sess_time = (
            df.group_by("ranker_id")
              .agg(pl.col("requestDate_dt").min().alias("session_time"))
              .sort("session_time")
        )
    else:
        sess_time = (
            train.select("ranker_id")
                 .unique()
                 .with_columns(pl.lit(None).alias("session_time"))
                 .sort("ranker_id")
        )

    n_sessions = sess_time.height
    n_holdout = max(1, int(math.ceil(n_sessions * holdout_frac)))

    holdout_ids = set(sess_time.tail(n_holdout)["ranker_id"].to_list())
    train_ids   = set(sess_time.head(n_sessions - n_holdout)["ranker_id"].to_list())

    # GroupKFold over remaining 95% sessions (deterministic shuffle)
    sessions = list(train_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)

    X_dummy = np.zeros(len(sessions))
    y_dummy = np.zeros(len(sessions))
    groups  = np.array(sessions)

    skf = GroupKFold(n_splits=n_folds)
    fold_ids: List[set] = []
    for _, val_idx in skf.split(X_dummy, y_dummy, groups=groups):
        fold_ids.append(set(np.array(sessions)[val_idx].tolist()))
    return train_ids, holdout_ids, fold_ids



def build_features(df: pl.DataFrame) -> pl.DataFrame:
    out = df.clone()

    # --- Durations → minutes (robust to string/numeric) ---
    dur_cols = ["legs0_duration", "legs1_duration"] + \
               [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
    dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in out.columns]
    if dur_exprs:
        out = out.with_columns(dur_exprs)

    # --- Simple label-free features ---
    mc_cols = [f"legs{l}_segments{s}_marketingCarrier_code" for l in (0,1) for s in range(4)]
    mc_exists = [c for c in mc_cols if c in out.columns]

    out = out.with_columns([
        (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
        pl.col("totalPrice").log1p().alias("log_price"),
        (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0)
          .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
          .otherwise(1.0).alias("duration_ratio"),
        (pl.sum_horizontal(pl.col(c).is_not_null().cast(pl.UInt8) for c in mc_exists) if mc_exists else pl.lit(0)).alias("n_marketing_codes_present"),
        (pl.col("frequentFlyer").fill_null("").str.count_matches("/") + (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),
        pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
        (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
    ])

    # --- Segment counts (per leg) ---
    seg_exprs = []
    for leg in (0, 1):
        seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in out.columns]
        if seg_cols:
            seg_exprs.append(pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols).cast(pl.Int32).alias(f"n_segments_leg{leg}"))
        else:
            seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))
    out = out.with_columns(seg_exprs).with_columns([
        (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
        (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
    ])

    # --- One-way flag (needs to exist before is_direct_leg1) ---
    out = out.with_columns([
        pl.when(
            pl.col("legs1_duration").is_null() |
            (pl.col("legs1_duration") == 0) |
            pl.col("legs1_segments0_departureFrom_airport_iata").is_null()
        ).then(1).otherwise(0).cast(pl.Int32).alias("is_one_way"),
    ])

    # --- Direct leg1 depends on is_one_way (create in separate step) ---
    out = out.with_columns([
        pl.when(pl.col("is_one_way") == 1)
          .then(0)
          .otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32))
          .alias("is_direct_leg1"),
    ])

    # --- Use direct flags; create group_size first, then group_size_log in a separate step ---
    out = out.with_columns([
        (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
        ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
        pl.col("Id").count().over("ranker_id").alias("group_size"),
    ])
    
    # IMPORTANT: compute group_size_log in its own call so the column exists
    out = out.with_columns([
        pl.col("group_size").log1p().alias("group_size_log"),
    ])

    # --- Price/duration ranks within session (label-free) ---
    out = out.with_columns([
        pl.col("totalPrice").rank().over("ranker_id").alias("price_rank"),
        (pl.col("totalPrice").rank("average").over("ranker_id") / pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
        (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
        ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
        pl.col("total_duration").rank().over("ranker_id").alias("duration_rank"),
    ])

    # --- Time-of-day/week if timestamps exist (robust cast) ---
    time_feats = []
    for col in ("legs0_departureAt","legs0_arrivalAt","legs1_departureAt","legs1_arrivalAt"):
        if col in out.columns:
            dt = pl.col(col).cast(pl.Datetime, strict=False)
            h  = dt.dt.hour().fill_null(12)
            time_feats += [
                h.alias(f"{col}_hour"),
                dt.dt.weekday().fill_null(0).alias(f"{col}_weekday"),
                (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{col}_business_time"),
            ]
    if time_feats:
        out = out.with_columns(time_feats)

    # --- Fill nulls (numeric→0, strings→"missing") ---
    num_cols = out.select(pl.selectors.numeric()).columns
    str_cols = out.select(pl.selectors.string()).columns
    out = out.with_columns([pl.col(c).fill_null(0) for c in num_cols] + [pl.col(c).fill_null("missing") for c in str_cols])

    return out

# -----------------------------
# Label encoders fit on 95% only (unsupervised) → map everywhere
# -----------------------------
def fit_label_maps(train_df: pl.DataFrame, cat_cols: List[str]) -> Dict[str, pl.DataFrame]:
    maps = {}
    for c in cat_cols:
        if c not in train_df.columns:
            continue
        mapping = (train_df.select(c).unique().drop_nulls().with_row_index(name=f"{c}__code").rename({c: f"{c}"}))
        maps[c] = mapping
    return maps

def apply_label_maps(df: pl.DataFrame, maps: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    out = df
    for c, mapping in maps.items():
        if c not in out.columns:
            continue
        out = (out.join(mapping, on=c, how="left")
                  .with_columns(pl.col(f"{c}__code").fill_null(-1).cast(pl.Int32).alias(f"{c}__le"))
                  .drop(f"{c}__code"))
    return out

# -----------------------------
# Target encoding: fit mappings on TRAIN ONLY (per-fold), apply to val/test
# -----------------------------
def fit_te_mappings(df_train: pl.DataFrame, cols: List[str], target_col: str, m: float) -> Tuple[Dict[str, pl.DataFrame], float]:
    prior = df_train.select(pl.col(target_col).mean()).item()
    maps = {}
    for c in cols:
        if c not in df_train.columns:
            continue
        mp = (
            df_train.group_by(c)
                    .agg([pl.len().alias("cnt"), pl.col(target_col).sum().alias("sum_y")])
                    .with_columns(((pl.col("sum_y") + m * prior) / (pl.col("cnt") + m)).alias(f"{c}_te"))
                    .select([c, f"{c}_te"])
        )
        maps[c] = mp
    return maps, prior

def apply_te_mappings(df: pl.DataFrame, maps: Dict[str, pl.DataFrame], prior: float) -> pl.DataFrame:
    out = df
    for c, mp in maps.items():
        if c not in out.columns:
            continue
        te_col = f"{c}_te"
        out = out.join(mp, on=c, how="left").with_columns(pl.col(te_col).fill_null(prior).alias(te_col))
    return out

# -----------------------------
# Build model matrices (grouped)
# -----------------------------
def select_feature_cols(df: pl.DataFrame) -> List[str]:
    numeric = [c for c in df.select(pl.selectors.numeric()).columns if c not in EXCLUDE_COLS]
    le_cols = [c for c in df.columns if c.endswith("__le")]
    te_cols = [c for c in df.columns if c.endswith("_te")]
    feat = sorted(set(numeric + le_cols + te_cols) - EXCLUDE_COLS)
    return feat

def to_dmatrix(df: pl.DataFrame, feat_cols: List[str]) -> Tuple[xgb.DMatrix, np.ndarray, np.ndarray]:
    ordered = df.sort(["ranker_id"])
    group_sizes = ordered.group_by("ranker_id", maintain_order=True).agg(pl.len())["len"].to_numpy()
    X = ordered.select(feat_cols).to_numpy()
    y = ordered["selected"].to_numpy() if "selected" in ordered.columns else None
    dmat = xgb.DMatrix(X, label=y, group=group_sizes, feature_names=feat_cols)
    return dmat, group_sizes, ordered["Id"].to_numpy()

def hitrate_at_3(y_true: np.ndarray, y_pred: np.ndarray, group_sizes: np.ndarray) -> float:
    start = 0
    hits = 0
    considered = 0
    for g in group_sizes:
        end = start + g
        if g > 10:
            considered += 1
            order = np.argsort(-y_pred[start:end])
            top_true = y_true[start:end][order][:3]
            hits += int(top_true.max() == 1)
        start = end
    return hits / considered if considered > 0 else 0.0

    



# 1) Load
train_raw, test_raw = read_data()

# 2) Split sessions FIRST (fixes your error by handling datetime robustly)
train_ids, holdout_ids, fold_ids = session_time_split(
    train_raw, holdout_frac=HOLDOUT_FRAC, n_folds=N_FOLDS, seed=RANDOM_STATE
)

train_95 = train_raw.filter(pl.col("ranker_id").is_in(list(train_ids)))
validator_ = train_raw.filter(pl.col("ranker_id").is_in(list(holdout_ids)))



train_95.head()



# 3) Build base features (label-free) for 95%, validator_, and test
train_95_f = build_features(train_95)
validator_f = build_features(validator_)
test_f = build_features(test_raw.with_columns(pl.lit(0, dtype=pl.Int8).alias("selected")))  # dummy target



# 4) Fit label encoders on 95% only, apply to all
cat_cols = [c for c in BASIC_CAT_COLS if c in train_95_f.columns]
le_maps = fit_label_maps(train_95_f, cat_cols)
train_95_f = apply_label_maps(train_95_f, le_maps)
validator_f = apply_label_maps(validator_f, le_maps)
test_f = apply_label_maps(test_f, le_maps)





NUM_BOOST_ROUND = 2_000
EARLY_STOPPING_ROUNDS = 200
VERBOSE_EVAL = 50


Identifier = 'SeriesA-R8_Shallow'

XGB_PARAMS = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg@3',
    "learning_rate": 0.0215,
    "max_depth": 12,
    "min_child_weight": 8,
    "subsample": 0.92,
    "colsample_bytree": 0.42,
    "gamma": 3.3084297630544888,
    "lambda": 5.952586917313028,
    "alpha": 0.6395254133055179,
    "seed": RANDOM_STATE,
    "n_jobs": -1,
    # "tree_method": "gpu_hist",  # enable if GPU is available
}



# 6) FULL OOF on 95%: train K models, collect OOF preds and test preds
# Pre-allocate OOF container
train_95_f = train_95_f.with_row_index(name="__row__")
oof_pred = np.zeros(train_95_f.height, dtype=np.float32)
best_iters = []
test_fold_preds = []

for k, val_ids in enumerate(fold_ids):
    if k < 3:
        print(f"\n[OOF] Fold {k}/{N_FOLDS-1}")
        tr_ids_k = set(train_ids) - val_ids
        tr_k = train_95_f.filter(pl.col("ranker_id").is_in(list(tr_ids_k))).drop("__row__")
        va_k = train_95_f.filter(pl.col("ranker_id").is_in(list(val_ids))).drop("__row__")
    
        te_cols = [c for c in CARRIER_COLS if c in tr_k.columns]
    
    
        # Per-fold TE
        te_maps_k, te_prior_k = fit_te_mappings(tr_k.select(["selected"] + te_cols), te_cols, "selected", TE_PRIOR_M)
        tr_k = apply_te_mappings(tr_k, te_maps_k, te_prior_k)
        va_k = apply_te_mappings(va_k, te_maps_k, te_prior_k)
    
        feat_cols_k = select_feature_cols(tr_k)
        dtr_k, gtr_k, _ = to_dmatrix(tr_k, feat_cols_k)
        dva_k, gva_k, ids_va_order = to_dmatrix(va_k, feat_cols_k)
    
        mdl_k = xgb.train(
            XGB_PARAMS, dtr_k,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtr_k, "train"), (dva_k, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=VERBOSE_EVAL
        )
        best_iters.append(mdl_k.best_iteration)
    
        # OOF preds on validation fold
        va_pred_k = mdl_k.predict(dva_k, iteration_range=(0, mdl_k.best_iteration + 1))
    
        # Map back to original rows by Id (keeps safety)
        fold_oof = pl.DataFrame({"Id": ids_va_order, "oof_score": va_pred_k})
        # Align to train_95_f rows
        idx_map = train_95_f.select(["__row__", "Id"]).join(fold_oof, on="Id", how="left")
        mask = idx_map["oof_score"].to_numpy()
        valid_mask = ~np.isnan(mask)
        oof_pred[idx_map["__row__"].to_numpy()[valid_mask]] = mask[valid_mask].astype(np.float32)
    
        # Test preds for this fold (using fold's TE mapping)
        test_k = apply_te_mappings(test_f, te_maps_k, te_prior_k)
        dtest_k, _, _ = to_dmatrix(test_k, feat_cols_k)
        test_fold_preds.append(mdl_k.predict(dtest_k, iteration_range=(0, mdl_k.best_iteration + 1)))
    
        # Clean
        del dtr_k, dva_k, dtest_k
        # del  mdl_k
        gc.collect()


mean_best_iter = int(np.mean(best_iters))
print(f"\n[OOF] Done. Mean best_iteration across folds: {mean_best_iter}")


# Save OOF for 95%

#oof_df = train_95_f.select(["Id","ranker_id"]).with_columns(pl.Series("xgb_ranker_oof", oof_pred))
#oof_path = f"oof_95_xgb_ranker_{Identifier}.csv"
#oof_df.write_csv(oof_path)
#print(f"Saved OOF (95%) to {oof_path}")



RUN_holdout_evaluation = False


if RUN_holdout_evaluation:
    model_full = mdl_k.copy()
    
    te_maps_full, te_prior_full = fit_te_mappings(train_95_f.drop("__row__").select(["selected"] + te_cols), te_cols, "selected", TE_PRIOR_M)
    tr_full = apply_te_mappings(train_95_f.drop("__row__"), te_maps_full, te_prior_full)
    val5 = apply_te_mappings(validator_f, te_maps_full, te_prior_full)
    
    feat_cols_full = select_feature_cols(tr_full)
    dtr_full, gtr_full, _ = to_dmatrix(tr_full, feat_cols_full)
    dval5,  gval5, id_val5_order = to_dmatrix(val5, feat_cols_full)


if RUN_holdout_evaluation:
    
    val5_pred = model_full.predict(dval5, iteration_range=(0, mean_best_iter))
    
    #val5_out = pl.DataFrame({"Id": id_val5_order, "ranker_id": val5.sort(["ranker_id"])["ranker_id"], "xgb_ranker_pred": val5_pred})
    #val5_path = f"pred_5pct_from_95_xgb_ranker_{Identifier}.csv"
    #val5_out.write_csv(val5_path)
    #print(f"Saved 5% validator_ preds to {val5_path}")
    
    
    y_val5 = val5.sort(["ranker_id"])["selected"].to_numpy()
    hr3_holdout = hitrate_at_3(y_val5, val5_pred, gval5)
    print(f"[HOLDOUT 5%] HitRate@3: {hr3_holdout:.4f} (groups>10 only)")



# Submission
test_avg = np.mean(np.vstack(test_fold_preds), axis=0).astype(np.float32)

test_sorted = (
    test_f.sort(["ranker_id"])
          .select(["Id", "ranker_id"])
          .with_columns(pl.Series("selected", test_avg))
)

submission_sorted = (
    test_sorted
    .with_columns(
        pl.col("selected")
          .rank(method="ordinal", descending=True)
          .over("ranker_id")
          .cast(pl.Int32)
          .alias("selected")
    )
    .select(["Id", "ranker_id", "selected"])
)

test_order = test_f.select("Id").with_row_index("ord")
submission = (
    submission_sorted.join(test_order, on="Id", how="inner")
                     .sort("ord")
                     .select(["Id", "ranker_id", "selected"])
)

scores = (
    test_sorted.join(test_order, on="Id", how="inner")
               .sort("ord")
               .select(["Id", "ranker_id", "selected"])
)

# (f) write files
scores_path = f"pred_test_from_95cv_xgb_ranker_{Identifier}.csv"
sub_path    = f"submission.csv"
scores.write_csv(scores_path)
submission.write_csv(sub_path)
print(f"Saved test scores to {scores_path}")
print(f"Saved submission to {sub_path}")





