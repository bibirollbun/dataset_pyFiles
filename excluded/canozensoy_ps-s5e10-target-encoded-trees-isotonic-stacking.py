# ---------------------
# Imports
# ---------------------

import os, gc, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from catboost.utils import get_gpu_device_count

# Silence warnings and library logs
warnings.filterwarnings("ignore")


# ---------------------
# Config
# ---------------------
SEED   = 42
NFOLDS = 5
FORCE_CB_CPU = False  # force CatBoost CPU even if GPU exists
VERBOSE = False       # master switch for any print-outs

def log(_): 
    # silent logger
    return None

np.random.seed(SEED)


# ---------------------
# Data paths
# ---------------------
CANDIDATES = [
    "/kaggle/input/playground-series-s5e10",
    "/kaggle/working",
    "/mnt/data",
    ".",
]
def first_existing_dir(paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return "."
DATA_DIR   = first_existing_dir(CANDIDATES)
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
SUB_PATH   = os.path.join(DATA_DIR, "sample_submission.csv")


# ---------------------
# Load data
# ---------------------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sub   = pd.read_csv(SUB_PATH)

TARGET  = "accident_risk"
train_id = train["id"].copy()
test_id  = test["id"].copy()

# Keep raw copies (CatBoost needs original categoricals)
train_raw = train.copy()
test_raw  = test.copy()

# Drop id for modeling frames
train = train.drop(columns=["id"])
test  = test.drop(columns=["id"])


# ---------------------
# Column taxonomy
# ---------------------
cat_cols  = ["road_type", "lighting", "weather", "time_of_day"]
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
num_cols  = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]


# ---------------------
# Cross features & bins
# ---------------------
def add_cross_feats(df):
    df = df.copy()
    df["rt__tod"]  = df["road_type"].astype(str) + "|" + df["time_of_day"].astype(str)
    df["lgt__wth"] = df["lighting"].astype(str)  + "|" + df["weather"].astype(str)
    return df

def fit_curv_bins(train_curv, q=10):
    qs = np.linspace(0, 1, q+1)
    edges = np.unique(train_curv.quantile(qs).values)
    if len(edges) < (q+1):
        mn, mx = float(train_curv.min()), float(train_curv.max())
        edges = np.linspace(mn - 1e-6, mx + 1e-6, q+1)
    return edges

def add_bins(df, curv_edges):
    df = df.copy()
    labels = [f"curv_q{i}" for i in range(10)]
    df["curv_bin"]  = pd.cut(df["curvature"], bins=curv_edges, labels=labels, include_lowest=True).astype(str)
    df["speed_bin"] = df["speed_limit"].astype(str)
    return df

train_raw = add_cross_feats(train_raw)
test_raw  = add_cross_feats(test_raw)
curv_edges = fit_curv_bins(train["curvature"], q=10)
train_raw = add_bins(train_raw, curv_edges)
test_raw  = add_bins(test_raw,  curv_edges)

engineered_cat = ["rt__tod", "lgt__wth", "curv_bin", "speed_bin"]


# ---------------------
# Frequency encoding (train-only mapping)
# ---------------------
freq_src_cols = cat_cols + engineered_cat + bool_cols
def freq_encode(train_df, test_df, cols):
    out_tr, out_te = {}, {}
    for c in cols:
        s_tr = train_df[c].astype(str)
        s_te = test_df[c].astype(str)
        vc = s_tr.value_counts(normalize=True)
        out_tr[f"freq__{c}"] = s_tr.map(vc).astype("float32").fillna(0.0)
        out_te[f"freq__{c}"] = s_te.map(vc).astype("float32").fillna(0.0)
    return pd.DataFrame(out_tr), pd.DataFrame(out_te)
FE_tr, FE_te = freq_encode(train_raw, test_raw, freq_src_cols)


# ---------------------
# KFold Target Encoding with multiple smoothings (2/5/15)
# ---------------------
te_cols_src = cat_cols + engineered_cat
def kfold_target_encode_multi(tr_df, te_df, y, cols, smoothings=(2.0, 5.0, 15.0), n_splits=NFOLDS, seed=SEED):
    y = y.astype("float32").reset_index(drop=True)
    res_tr, res_te = [], []
    for sm in smoothings:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        global_mean = float(y.mean())
        tr_blk = pd.DataFrame(index=tr_df.index)
        te_blk = pd.DataFrame(index=te_df.index)
        for c in cols:
            oof = np.zeros(len(tr_df), dtype=np.float32)
            s_tr = tr_df[c].astype(str).reset_index(drop=True)
            s_te = te_df[c].astype(str).reset_index(drop=True)
            for tr_idx, va_idx in kf.split(s_tr):
                tmp = pd.DataFrame({"key": s_tr.iloc[tr_idx].values, "y": y.iloc[tr_idx].values})
                agg = tmp.groupby("key")["y"].agg(["mean", "count"])
                smooth = (agg["count"] * agg["mean"] + sm * global_mean) / (agg["count"] + sm)
                mapping = smooth.to_dict()
                mapped = s_tr.iloc[va_idx].map(mapping).astype("float32")
                oof[va_idx] = mapped.fillna(global_mean).values
            # full-train mapping for test
            tmp_full = pd.DataFrame({"key": s_tr.values, "y": y.values})
            agg_full = tmp_full.groupby("key")["y"].agg(["mean", "count"])
            smooth_full = (agg_full["count"] * agg_full["mean"] + sm * global_mean) / (agg_full["count"] + sm)
            full_map = smooth_full.to_dict()
            tr_blk[f"te{int(sm)}__{c}"] = oof
            te_blk[f"te{int(sm)}__{c}"] = s_te.map(full_map).astype("float32").fillna(global_mean).values
        res_tr.append(tr_blk); res_te.append(te_blk)
    return pd.concat(res_tr, axis=1), pd.concat(res_te, axis=1)
TE_tr, TE_te = kfold_target_encode_multi(
    tr_df=train_raw[te_cols_src],
    te_df=test_raw[te_cols_src],
    y=train[TARGET],
    cols=te_cols_src,
    smoothings=(2.0, 5.0, 15.0),
    n_splits=NFOLDS,
    seed=SEED,
)


# ---------------------
# Numeric extras + booleans + meta-risk
# ---------------------
def add_num_extras(df):
    out = pd.DataFrame(index=df.index)
    out["speed_curvature"]     = df["speed_limit"] * df["curvature"]
    out["lanes_speed"]         = df["num_lanes"] * df["speed_limit"]
    out["accident_speed_risk"] = df["num_reported_accidents"] * df["speed_limit"] / 100.0
    out["curvature_squared"]   = (df["curvature"] ** 2.0)
    # a couple more stable interactions
    out["curv_acc"]            = df["curvature"] * df["num_reported_accidents"]
    out["speed2"]              = (df["speed_limit"] ** 2) / 100.0
    return out.astype("float32")
NUMEX_tr = add_num_extras(train)
NUMEX_te = add_num_extras(test)

def bool_to_int(df, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = df[c].astype("int8")
    return out
B_tr = bool_to_int(train, bool_cols)
B_te = bool_to_int(test,  bool_cols)

NUM_tr = train[num_cols].astype("float32")
NUM_te = test[num_cols].astype("float32")

def add_meta_risk(df_raw):
    base = (
        0.30 * df_raw["curvature"].astype(float) +
        0.20 * (df_raw["lighting"].astype(str).eq("night")).astype(int) +
        0.10 * (df_raw["weather"].astype(str).ne("clear")).astype(int) +
        0.20 * (df_raw["speed_limit"].astype(int) >= 60).astype(int) +
        0.10 * (df_raw["num_reported_accidents"].astype(int) > 2).astype(int)
    )
    return base.astype("float32").to_frame("meta_risk")
META_tr = add_meta_risk(train_raw)
META_te = add_meta_risk(test_raw)

# Final numeric matrices
X_num_tr = pd.concat(
    [NUM_tr.reset_index(drop=True),
     B_tr.reset_index(drop=True),
     FE_tr.reset_index(drop=True),
     TE_tr.reset_index(drop=True),
     NUMEX_tr.reset_index(drop=True),
     META_tr.reset_index(drop=True)],
    axis=1
).astype("float32")
X_num_te = pd.concat(
    [NUM_te.reset_index(drop=True),
     B_te.reset_index(drop=True),
     FE_te.reset_index(drop=True),
     TE_te.reset_index(drop=True),
     NUMEX_te.reset_index(drop=True),
     META_te.reset_index(drop=True)],
    axis=1
).astype("float32")
y = train[TARGET].astype("float32")


# ---------------------
# CatBoost matrices
# ---------------------
cb_cat_cols = cat_cols + engineered_cat + bool_cols
for c in engineered_cat:
    train_raw[c] = train_raw[c].astype(str)
    test_raw[c]  = test_raw[c].astype(str)

cb_num_tr = pd.concat([train[num_cols].reset_index(drop=True),
                       NUMEX_tr.reset_index(drop=True),
                       META_tr.reset_index(drop=True)], axis=1)
cb_num_te = pd.concat([test[num_cols].reset_index(drop=True),
                       NUMEX_te.reset_index(drop=True),
                       META_te.reset_index(drop=True)], axis=1)
cb_train = pd.concat([train_raw[cb_cat_cols].reset_index(drop=True),
                      cb_num_tr.reset_index(drop=True)], axis=1)
cb_test  = pd.concat([test_raw[cb_cat_cols].reset_index(drop=True),
                      cb_num_te.reset_index(drop=True)], axis=1)
cb_cat_idx = [cb_train.columns.get_loc(c) for c in cb_cat_cols]


# ---------------------
# Monotonic constraints
# ---------------------
def make_monotone_vector(cols):
    v = [0]*len(cols)
    for f in ["curvature", "speed_limit", "num_reported_accidents"]:
        if f in cols:
            v[list(cols).index(f)] = 1
    return v
mono_vec = make_monotone_vector(list(X_num_tr.columns))
xgb_mono = "(" + ",".join(str(int(x)) for x in mono_vec) + ")"


# ---------------------
# Stratified folds on binned y
# ---------------------
bins = pd.qcut(y, q=min(20, y.nunique()), labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(train), dtype=np.float32)
oof_xgb = np.zeros(len(train), dtype=np.float32)
oof_cb  = np.zeros(len(train), dtype=np.float32)
pred_lgb = np.zeros(len(test), dtype=np.float32)
pred_xgb = np.zeros(len(test), dtype=np.float32)
pred_cb  = np.zeros(len(test), dtype=np.float32)

# CatBoost device
try:
    USE_CB_GPU = (not FORCE_CB_CPU) and (get_gpu_device_count() > 0)
except Exception:
    USE_CB_GPU = False


# ---------------------
# Training
# ---------------------
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num_tr, bins), 1):
    X_trn_num, X_val_num = X_num_tr.iloc[tr_idx], X_num_tr.iloc[va_idx]
    y_trn,     y_val     = y.iloc[tr_idx],     y.iloc[va_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=25000,
        learning_rate=0.03,
        num_leaves=96,
        max_depth=-1,
        min_data_in_leaf=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=1.5,
        reg_alpha=0.0,
        monotone_constraints=mono_vec,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,  # silence LightGBM
    )
    lgb_model.fit(
        X_trn_num, y_trn,
        eval_set=[(X_val_num, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=400, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    best_it = getattr(lgb_model, "best_iteration_", None)
    if best_it is None or best_it <= 0:
        oof_lgb[va_idx] = lgb_model.predict(X_val_num).astype("float32")
        pred_lgb += lgb_model.predict(X_num_te).astype("float32") / NFOLDS
    else:
        oof_lgb[va_idx] = lgb_model.predict(X_val_num, num_iteration=best_it).astype("float32")
        pred_lgb += lgb_model.predict(X_num_te, num_iteration=best_it).astype("float32") / NFOLDS

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=25000,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=5.0,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        reg_alpha=0.1,
        monotone_constraints=xgb_mono,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,  # silence XGB
    )
    xgb_model.fit(
        X_trn_num, y_trn,
        eval_set=[(X_val_num, y_val)],
        early_stopping_rounds=400,
        verbose=False,
    )
    oof_xgb[va_idx] = xgb_model.predict(X_val_num).astype("float32")
    pred_xgb += xgb_model.predict(X_num_te).astype("float32") / NFOLDS

    # CatBoost
    cb_model = cb.CatBoostRegressor(
        loss_function="RMSE",
        iterations=25000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=4.0,
        random_seed=SEED,
        od_type="Iter",
        od_wait=400,
        task_type=("GPU" if USE_CB_GPU else "CPU"),
        verbose=False,  # silence CatBoost
    )
    X_trn_cb = cb_train.iloc[tr_idx]
    X_val_cb = cb_train.iloc[va_idx]
    cb_model.fit(
        X_trn_cb, y_trn,
        eval_set=(X_val_cb, y_val),
        cat_features=cb_cat_idx,
    )
    oof_cb[va_idx] = cb_model.predict(X_val_cb).astype("float32")
    pred_cb += cb_model.predict(cb_test).astype("float32") / NFOLDS

    del X_trn_num, X_val_num, X_trn_cb, X_val_cb
    gc.collect()


# ---------------------
# Level-2 Meta Learner on OOF predictions
# ---------------------
M_tr = pd.DataFrame({
    "p_lgb": oof_lgb,
    "p_xgb": oof_xgb,
    "p_cb":  oof_cb,
    "p_mean": (oof_lgb + oof_xgb + oof_cb)/3.0,
    # a few safe numeric hints
    "curvature": train["curvature"].values.astype(np.float32),
    "speed_limit": train["speed_limit"].values.astype(np.int16),
    "num_reported_accidents": train["num_reported_accidents"].values.astype(np.int8),
    "meta_risk": META_tr["meta_risk"].values.astype(np.float32),
})
M_te = pd.DataFrame({
    "p_lgb": pred_lgb,
    "p_xgb": pred_xgb,
    "p_cb":  pred_cb,
    "p_mean": (pred_lgb + pred_xgb + pred_cb)/3.0,
    "curvature": test["curvature"].values.astype(np.float32),
    "speed_limit": test["speed_limit"].values.astype(np.int16),
    "num_reported_accidents": test["num_reported_accidents"].values.astype(np.int8),
    "meta_risk": META_te["meta_risk"].values.astype(np.float32),
})

stacker = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=20000,
    learning_rate=0.03,
    num_leaves=16,
    max_depth=4,
    min_data_in_leaf=100,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    reg_alpha=0.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,  # silence
)

# use stratified folds on the same bins for the meta
oof_meta = np.zeros(len(train), dtype=np.float32)
pred_meta = np.zeros(len(test), dtype=np.float32)
for fold, (tr_idx, va_idx) in enumerate(skf.split(M_tr, bins), 1):
    X_trn, X_val = M_tr.iloc[tr_idx], M_tr.iloc[va_idx]
    y_trn, y_val = y.iloc[tr_idx],   y.iloc[va_idx]
    stacker.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    best_it = getattr(stacker, "best_iteration_", None)
    if best_it is None or best_it <= 0:
        oof_meta[va_idx] = stacker.predict(X_val).astype("float32")
        pred_meta += stacker.predict(M_te).astype("float32") / NFOLDS
    else:
        oof_meta[va_idx] = stacker.predict(X_val, num_iteration=best_it).astype("float32")
        pred_meta += stacker.predict(M_te, num_iteration=best_it).astype("float32") / NFOLDS


# ---------------------
# Conditional isotonic on meta output
# ---------------------
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_meta, y.values)
oof_iso = iso.transform(oof_meta)
rmse_pre = mean_squared_error(y, oof_meta, squared=False)
rmse_post = mean_squared_error(y, oof_iso, squared=False)
use_iso = rmse_post < rmse_pre

test_pred = iso.transform(pred_meta) if use_iso else pred_meta
test_pred = np.clip(test_pred, 0.0, 1.0).astype("float32")


# ---------------------
# Submission
# ---------------------
submission = pd.DataFrame({"id": test_id, TARGET: test_pred})
submission.to_csv("submission.csv", index=False)


