import os
import sys
import time
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, KBinsDiscretizer

import xgboost as xgb
import optuna  # tuning used only in FULL mode

# -------------------------
# Setup logging early
# -------------------------
BASE_DIR = Path("/kaggle/input/playground-series-s5e11")
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "code_8_1_v4.txt"
SUBMISSION_PATH = OUTPUT_DIR / "submission_4.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
print("Log file initialized at %s", LOG_FILE)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
print("HF_TOKEN present: %s", "yes" if HF_TOKEN else "no")

# -------------------------
# Device selection for XGBoost (purpose: choose CUDA if available)
# -------------------------
def detect_cuda_available() -> bool:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    out = os.popen(f"{exe} -L").read().strip()
    return len(out) > 0

CUDA_AVAILABLE = detect_cuda_available()
XGB_DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
if CUDA_AVAILABLE:
    print("CUDA detected. Using device='%s' with tree_method='hist'.", XGB_DEVICE)
else:
    print("CUDA not detected. Using CPU (device='cpu') with tree_method='hist'.")

# -------------------------
# Competition schema
# -------------------------
TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH = BASE_DIR / "test.csv"
SAMPLE_SUB_PATH = BASE_DIR / "sample_submission.csv"

TARGET_COL = "loan_paid_back"   # binary classification; metric: ROC AUC
ID_COL = "id"
FOLD_COL = "fold"
META_COLS = {TARGET_COL, ID_COL, FOLD_COL}

# Optional numeric columns for special transforms (if present)
C_INCOME = "annual_income"
C_DTI = "debt_to_income_ratio"

# -------------------------
# Load data (purpose: read CSVs; inputs: train/test paths)
# -------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
print("Loaded data. Train shape: %s | Test shape: %s", train.shape, test.shape)
assert TARGET_COL in train.columns, f"Missing target column '{TARGET_COL}' in train.csv"
assert ID_COL in train.columns and ID_COL in test.columns, "Missing id column in train/test."

# -------------------------
# Typing helpers and encoders (purpose: reusable feature builders; inputs: DataFrames/Series)
# -------------------------
def get_cat_num_cols(df: pd.DataFrame, target_col: str, id_col: str, exclude: set):
    cols = [c for c in df.columns if c not in exclude]
    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]
    return cat_cols, num_cols

def pick_top_cats(cat_cols, df, k=6, exclude: set = None):
    exclude = exclude or set()
    cands = []
    for c in cat_cols:
        if c in exclude:
            continue
        n_unique = df[c].nunique(dropna=True)
        if 2 <= n_unique <= 200:
            cands.append((c, n_unique))
    cands.sort(key=lambda t: (-t[1], t[0]))
    sel = [c for c, _ in cands[:k]]
    if len(sel) < min(k, len(cat_cols)):
        rest = [c for c in cat_cols if c not in sel and c not in exclude]
        sel += rest[: (k - len(sel))]
    return sel[:k]

def pick_top_nums(num_cols, df, k=5, exclude: set = None):
    exclude = exclude or set()
    stats = []
    for c in num_cols:
        if c in exclude:
            continue
        series = df[c]
        if series.dtype.kind not in "biufc":
            continue
        nunq = series.nunique(dropna=True)
        if nunq <= 2:
            continue
        var = series.var(skipna=True)
        stats.append((c, 0.0 if pd.isna(var) else float(var)))
    stats.sort(key=lambda t: -t[1])
    return [c for c, _ in stats[:k]]

def add_missing_indicators(df: pd.DataFrame, exclude_cols):
    for c in df.columns:
        if c in exclude_cols:
            continue
        ind_name = f"{c}__isna"
        if ind_name not in df.columns:
            df[ind_name] = df[c].isna().astype(np.int8)
    return df

def frequency_encode(train_pool: pd.DataFrame, series: pd.Series):
    counts = train_pool[series.name].value_counts(dropna=False)
    return counts.to_dict()

def compute_te_map(x: pd.Series, y: pd.Series, m: float = 10.0):
    df = pd.DataFrame({"x": x, "y": y})
    gr = df.groupby("x")["y"].agg(["mean", "count"])
    global_mean = float(y.mean())
    smooth = (gr["mean"] * gr["count"] + global_mean * m) / (gr["count"] + m)
    return smooth.to_dict(), global_mean

def oof_target_encode(train_pool_df, y, col, folds, m=10.0):
    oof = pd.Series(index=train_pool_df.index, dtype="float32")
    for f, (tr_idx, va_idx) in folds.items():
        tr_df = train_pool_df.loc[tr_idx]
        tr_y = y.loc[tr_idx]
        mp, gmean = compute_te_map(tr_df[col], tr_y, m)
        oof.loc[va_idx] = train_pool_df.loc[va_idx, col].map(mp).fillna(gmean).astype("float32")
    full_map, full_gmean = compute_te_map(train_pool_df[col], y, m)
    return oof, full_map, full_gmean

def compute_woe_map(x: pd.Series, y: pd.Series, eps: float = 0.5):
    df = pd.DataFrame({"x": x, "y": y})
    pos = df.groupby("x")["y"].sum(min_count=1)
    cnt = df.groupby("x")["y"].count()
    neg = cnt - pos
    total_pos = float(pos.sum())
    total_neg = float(neg.sum())
    dist_pos = (pos + eps) / (total_pos + eps * len(pos))
    dist_neg = (neg + eps) / (total_neg + eps * len(neg))
    woe = np.log((dist_pos) / (dist_neg))
    mapping = woe.to_dict()
    iv = ((dist_pos - dist_neg) * woe).sum()
    return mapping, float(iv)

def oof_woe_encode(train_pool_df, y, col, folds, eps=0.5):
    oof = pd.Series(index=train_pool_df.index, dtype="float32")
    for f, (tr_idx, va_idx) in folds.items():
        tr_df = train_pool_df.loc[tr_idx]
        tr_y = y.loc[tr_idx]
        mp, _iv = compute_woe_map(tr_df[col], tr_y, eps)
        # Clip WOE values to stabilize
        mp = {k: float(np.clip(v, -3.0, 3.0)) for k, v in mp.items()}
        oof.loc[va_idx] = train_pool_df.loc[va_idx, col].map(mp).fillna(0.0).astype("float32")
    full_map, iv_full = compute_woe_map(train_pool_df[col], y, eps)
    full_map = {k: float(np.clip(v, -3.0, 3.0)) for k, v in full_map.items()}
    return oof, full_map, iv_full

def fit_kbins(train_pool_series, n_bins=10):
    med = float(np.nanmedian(train_pool_series.values))
    tr_vals = train_pool_series.fillna(med).values.reshape(-1, 1)
    enc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    enc.fit(tr_vals)
    return enc, med

def transform_kbins(enc, med, series):
    vals = series.fillna(med).values.reshape(-1, 1)
    b = enc.transform(vals).astype("float32").reshape(-1)
    b = np.where(np.isfinite(b), b, -1.0)
    return pd.Series(b, index=series.index, dtype="float32")

def fit_rank_gaussian(train_pool_series, random_state=2025):
    med = float(np.nanmedian(train_pool_series.values))
    tr_vals = train_pool_series.fillna(med).values.reshape(-1, 1)
    qt = QuantileTransformer(n_quantiles=min(1000, len(tr_vals)), output_distribution="normal", random_state=random_state)
    qt.fit(tr_vals)
    return qt, med

def transform_rank_gaussian(qt, med, series):
    vals = series.fillna(med).values.reshape(-1, 1)
    out = qt.transform(vals).astype("float32").reshape(-1)
    return pd.Series(out, index=series.index, dtype="float32")

def fit_yeojohnson(train_pool_series):
    med = float(np.nanmedian(train_pool_series.values))
    tr_vals = train_pool_series.fillna(med).values.reshape(-1, 1)
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    pt.fit(tr_vals)
    return pt, med

def transform_yeojohnson(pt, med, series):
    vals = series.fillna(med).values.reshape(-1, 1)
    out = pt.transform(vals).astype("float32").reshape(-1)
    return pd.Series(out, index=series.index, dtype="float32")

def group_mean_deviation(train_pool_df, val_df, test_df, cat_cols, num_cols):
    # Fit group means on train_pool and map to val/test; guard meta columns.
    for c in cat_cols:
        if c in META_COLS or c not in train_pool_df.columns:
            continue
        for n in num_cols:
            if n in META_COLS or n not in train_pool_df.columns:
                continue
            gname = f"{n}__gm_{c}"
            devname = f"{n}__dev_{c}"
            grp = train_pool_df.groupby(c, observed=True)[n].mean()
            global_mean = float(train_pool_df[n].mean())
            train_pool_df[gname] = train_pool_df[c].map(grp).fillna(global_mean).astype("float32")
            val_df[gname] = val_df[c].map(grp).fillna(global_mean).astype("float32")
            test_df[gname] = test_df[c].map(grp).fillna(global_mean).astype("float32")
            train_pool_df[devname] = (train_pool_df[n] - train_pool_df[gname]).astype("float32")
            val_df[devname] = (val_df[n] - val_df[gname]).astype("float32")
            test_df[devname] = (test_df[n] - test_df[gname]).astype("float32")
    return train_pool_df, val_df, test_df

def group_percentile_feature(train_pool_df, val_df, test_df, group_col, value_col, feature_name, q=100):
    if group_col not in train_pool_df.columns or value_col not in train_pool_df.columns:
        print("Percentile feature skipped (missing): %s within %s", value_col, group_col)
        return train_pool_df, val_df, test_df
    edges_dict = {}
    for g, sub in train_pool_df[[group_col, value_col]].dropna().groupby(group_col, observed=True):
        vals = sub[value_col].values
        if len(vals) < 2:
            continue
        qs = np.linspace(0.0, 1.0, q + 1)
        try_edges = np.quantile(vals, qs)
        edges = try_edges.copy()
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], float("inf"))
        edges_dict[g] = edges

    def apply_edges(df_in: pd.DataFrame):
        out = pd.Series(index=df_in.index, dtype="float32")
        out.iloc[:] = np.nan
        for g, idx in df_in.groupby(group_col, observed=True).groups.items():
            e = edges_dict.get(g, None)
            if e is None:
                out.loc[idx] = 0.5
                continue
            v = df_in.loc[idx, value_col].fillna(e[0]).values
            bins = np.digitize(v, e[1:-1], right=True)
            denom = max(1, len(e) - 2)
            out.loc[idx] = bins.astype("float32") / float(denom)
        out.fillna(0.5, inplace=True)
        return out

    train_pool_df[feature_name] = apply_edges(train_pool_df[[group_col, value_col]].copy())
    val_df[feature_name] = apply_edges(val_df[[group_col, value_col]].copy())
    test_df[feature_name] = apply_edges(test_df[[group_col, value_col]].copy())
    return train_pool_df, val_df, test_df

# -------------------------
# Global feature selections (purpose: choose candidate categorical and numeric columns)
# -------------------------
exclude_for_typing = {TARGET_COL, ID_COL, FOLD_COL}
all_cat, all_num = get_cat_num_cols(train, TARGET_COL, ID_COL, exclude=exclude_for_typing)
sel_cat = pick_top_cats(all_cat, train, k=6, exclude=META_COLS)
sel_num_for_deviation = pick_top_nums(all_num, train, k=5, exclude=META_COLS)
transform_targets = [c for c in [C_INCOME, C_DTI] if c in train.columns]
all_features_for_te = [c for c in (all_cat + all_num) if c not in META_COLS]
print("Selected categoricals (â‰¤6): %s", sel_cat)
print("Selected numeric for group-mean deviations (â‰¤5): %s", sel_num_for_deviation)
print("Numeric transform targets: %s", transform_targets)
print("All features for target encoding (%d): %s", len(all_features_for_te), all_features_for_te)

# -------------------------
# Preprocess for arbitrary held-out fold (purpose: per-fold encoders; inputs: train_df, test_df, held_out_fold)
# -------------------------
def preprocess_for_outer_fold(train_df, test_df, held_out_fold, sel_cat, sel_num_for_deviation, transform_targets, all_features_for_te):
    """Fit encoders/transforms on train_pool=all folds except held_out_fold; apply to its validation and test."""
    if held_out_fold == -1:
        tr_pool = train_df.copy()
        va = train_df.iloc[0:0].copy()  # empty
    else:
        tr_pool = train_df.loc[train_df[FOLD_COL] != held_out_fold].copy()
        va = train_df.loc[train_df[FOLD_COL] == held_out_fold].copy()
    te = test_df.copy()
    y_pool = tr_pool[TARGET_COL].astype(int)

    # Inner folds based on existing assignment in tr_pool
    inner_fold_ids = sorted(int(f) for f in tr_pool[FOLD_COL].unique().tolist())
    inner_folds = {}
    for f in inner_fold_ids:
        inner_tr_idx = tr_pool.index[tr_pool[FOLD_COL] != f]
        inner_va_idx = tr_pool.index[tr_pool[FOLD_COL] == f]
        inner_folds[int(f)] = (inner_tr_idx, inner_va_idx)

    # Frequency encoding
    for c in sel_cat:
        if c not in tr_pool.columns:
            continue
        mapping = frequency_encode(tr_pool, tr_pool[c])
        tr_pool[f"{c}__freq"] = tr_pool[c].map(mapping).fillna(0).astype("float32")
        if len(va) > 0:
            va[f"{c}__freq"] = va[c].map(mapping).fillna(0).astype("float32")
        te[f"{c}__freq"] = te[c].map(mapping).fillna(0).astype("float32")

    # OOF TE on ALL features (categorical + numerical)
    for c in all_features_for_te:
        if c not in tr_pool.columns:
            continue
        oof_te, te_map, te_g = oof_target_encode(tr_pool, y_pool, c, inner_folds, m=10.0)
        tr_pool[f"{c}__te_m10"] = oof_te.astype("float32")
        if len(va) > 0:
            va[f"{c}__te_m10"] = va[c].map(te_map).fillna(te_g).astype("float32")
        te[f"{c}__te_m10"] = te[c].map(te_map).fillna(te_g).astype("float32")

    # OOF WOE (clip WOE) - only for categorical features
    for c in sel_cat:
        if c not in tr_pool.columns:
            continue
        oof_woe, woe_map, _iv = oof_woe_encode(tr_pool, y_pool, c, inner_folds, eps=0.5)
        tr_pool[f"{c}__woe"] = oof_woe.astype("float32")
        if len(va) > 0:
            va[f"{c}__woe"] = va[c].map(woe_map).fillna(0.0).astype("float32")
        te[f"{c}__woe"] = te[c].map(woe_map).fillna(0.0).astype("float32")

    # Numeric transforms on income & DTI
    for col in transform_targets:
        if col not in tr_pool.columns:
            continue
        enc, med = fit_kbins(tr_pool[col], n_bins=10)
        tr_pool[f"{col}__qbin10"] = transform_kbins(enc, med, tr_pool[col])
        if len(va) > 0:
            va[f"{col}__qbin10"] = transform_kbins(enc, med, va[col])
        te[f"{col}__qbin10"] = transform_kbins(enc, med, te[col])

        qt, med_q = fit_rank_gaussian(tr_pool[col])
        tr_pool[f"{col}__rgauss"] = transform_rank_gaussian(qt, med_q, tr_pool[col])
        if len(va) > 0:
            va[f"{col}__rgauss"] = transform_rank_gaussian(qt, med_q, va[col])
        te[f"{col}__rgauss"] = transform_rank_gaussian(qt, med_q, te[col])

        pt, med_p = fit_yeojohnson(tr_pool[col])
        tr_pool[f"{col}__yeoj"] = transform_yeojohnson(pt, med_p, tr_pool[col])
        if len(va) > 0:
            va[f"{col}__yeoj"] = transform_yeojohnson(pt, med_p, va[col])
        te[f"{col}__yeoj"] = transform_yeojohnson(pt, med_p, te[col])

    # Group mean deviations
    tr_pool, va, te = group_mean_deviation(tr_pool, va, te, sel_cat, sel_num_for_deviation)

    # Percentile features
    if "credit_score" in tr_pool.columns and "grade_subgrade" in tr_pool.columns:
        tr_pool, va, te = group_percentile_feature(tr_pool, va, te, "grade_subgrade", "credit_score", "credit_score__pctl_in_grade")
    if "credit_score" in tr_pool.columns and "education_level" in tr_pool.columns:
        tr_pool, va, te = group_percentile_feature(tr_pool, va, te, "education_level", "credit_score", "credit_score__pctl_in_edu")

    # Missingness indicators
    tr_pool = add_missing_indicators(tr_pool, exclude_cols=META_COLS)
    if len(va) > 0:
        va = add_missing_indicators(va, exclude_cols=META_COLS)
    te = add_missing_indicators(te, exclude_cols={ID_COL})

    # Feature list: original numeric (excluding raw categoricals/meta) + engineered blocks
    excl = {TARGET_COL, ID_COL, FOLD_COL}
    raw_cat, raw_num = get_cat_num_cols(train_df, TARGET_COL, ID_COL, exclude=excl)
    raw_num_cols = [c for c in raw_num if c not in META_COLS]

    eng_cols = [c for c in tr_pool.columns if (
        c not in train_df.columns or
        c.endswith("__freq") or c.endswith("__te_m10") or c.endswith("__woe") or
        "__gm_" in c or "__dev_" in c or
        c.endswith("__qbin10") or c.endswith("__rgauss") or c.endswith("__yeoj") or
        c.endswith("__isna") or
        c.endswith("__pctl_in_grade") or c.endswith("__pctl_in_edu")
    )]
    feature_cols = sorted(set(raw_num_cols + eng_cols))
    feature_cols = [c for c in feature_cols if (c not in META_COLS and not c.endswith("__iv"))]

    X_tr = tr_pool[feature_cols].copy()
    y_tr = tr_pool[TARGET_COL].astype(int).copy()
    if len(va) > 0:
        X_va = va[feature_cols].copy()
        y_va = va[TARGET_COL].astype(int).copy()
    else:
        X_va = va  # empty
        y_va = va  # empty
    X_te = te[feature_cols].copy()
    return X_tr, y_tr, X_va, y_va, X_te, feature_cols

# -------------------------
# XGBoost params and trainers
# -------------------------
def build_xgb_params(base_lr=0.05, n_estimators=1500, early_stopping_rounds=100):
    params = dict(
        booster="gbtree",
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device=XGB_DEVICE,   # 'cuda:0' or 'cpu'
        learning_rate=base_lr,
        max_depth=6,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        gamma=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_bin=256,
        grow_policy="depthwise",
        random_state=2025,
        n_estimators=n_estimators,
        n_jobs=0,
        early_stopping_rounds=early_stopping_rounds,
        verbosity=1,
    )
    return params

def train_xgb_single(X_tr, y_tr, X_va, y_va, params, label="baseline"):
    t0 = time.time()
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    best_it = getattr(clf, "best_iteration", None)
    y_pred_va = clf.predict_proba(X_va, iteration_range=(0, best_it + 1) if best_it is not None else None)[:, 1]
    val_auc = roc_auc_score(y_va, y_pred_va)
    elapsed = time.time() - t0
    print("XGB %s: val AUC=%.6f | best_iteration=%s | time=%.1fs", label, val_auc, str(best_it), elapsed)
    return clf, val_auc, best_it, elapsed

def optuna_tune_xgb(X_tr, y_tr, X_va, y_va, base_params, time_budget_sec=300):
    print("Optuna tuning start (budget=%ds).", time_budget_sec)
    study = optuna.create_study(direction="maximize", study_name="xgb_ps_s5e11_v4")

    def objective(trial: optuna.trial.Trial):
        p = base_params.copy()
        p.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12),
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 12.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
            "n_estimators": trial.suggest_int("n_estimators", 600, 1500),
        })
        # Keep device and tree_method fixed
        p["tree_method"] = base_params["tree_method"]
        p["device"] = base_params["device"]
        p["random_state"] = 2025
        p["early_stopping_rounds"] = base_params["early_stopping_rounds"]

        model = xgb.XGBClassifier(**p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_it = getattr(model, "best_iteration", None)
        y_pred = model.predict_proba(X_va, iteration_range=(0, best_it + 1) if best_it is not None else None)[:, 1]
        auc = roc_auc_score(y_va, y_pred)
        return auc

    study.optimize(objective, n_trials=200, timeout=time_budget_sec, gc_after_trial=True)
    best_params = study.best_params
    best_value = study.best_value
    print("Optuna best AUC=%.6f with params=%s", best_value, json.dumps(best_params))

    tuned_params = base_params.copy()
    tuned_params.update(best_params)
    # Retrain on the same fold-0 split to verify
    model = xgb.XGBClassifier(**tuned_params)
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    best_it = getattr(model, "best_iteration", None)
    y_pred = model.predict_proba(X_va, iteration_range=(0, best_it + 1) if best_it is not None else None)[:, 1]
    auc = roc_auc_score(y_va, y_pred)
    elapsed = time.time() - t0
    print("Tuned retrain: val AUC=%.6f | best_iteration=%s | retrain_time=%.1fs", auc, str(best_it), elapsed)
    return model, auc, best_it, tuned_params

# -------------------------
# CV trainer and final refit
# -------------------------
def assign_outer_folds(df: pd.DataFrame, n_splits=5, seed=2025):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1, dtype=int)
    for i, (_, va_idx) in enumerate(skf.split(df.drop(columns=[TARGET_COL]), df[TARGET_COL].values)):
        folds[va_idx] = i
    out = df.copy()
    out[FOLD_COL] = folds
    return out

def train_xgb_cv_and_predict(train_df, test_df, params, n_splits=5, debug=False):
    print("Starting %d-fold CV training with per-fold encoders.", n_splits)
    train_df = assign_outer_folds(train_df, n_splits=n_splits, seed=2025)
    oof = np.zeros(len(train_df), dtype=np.float32)
    test_preds = []
    fold_aucs = []
    best_its = []

    for f in range(n_splits):
        print("Fold %d/%d: preprocessing (fit on train folds only).", f+1, n_splits)
        X_tr, y_tr, X_va, y_va, X_te, feats = preprocess_for_outer_fold(
            train_df, test_df, held_out_fold=f,
            sel_cat=sel_cat, sel_num_for_deviation=sel_num_for_deviation, transform_targets=transform_targets,
            all_features_for_te=all_features_for_te
        )
        params_use = params.copy()
        if debug:
            # Downsample training to 1000 rows in DEBUG to save time
            if len(X_tr) > 1000:
                X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, test_size=(1.0 - 1000/len(X_tr)), stratify=y_tr, random_state=2025)
            params_use["n_estimators"] = min(200, params_use.get("n_estimators", 1500))
            params_use["early_stopping_rounds"] = min(20, params_use.get("early_stopping_rounds", 100))

        print("Fold %d: training XGBoost.", f+1)
        clf = xgb.XGBClassifier(**params_use)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_it = getattr(clf, "best_iteration", None)
        y_va_pred = clf.predict_proba(X_va, iteration_range=(0, best_it + 1) if best_it is not None else None)[:, 1]
        fold_auc = roc_auc_score(y_va, y_va_pred)
        fold_aucs.append(fold_auc)
        oof[train_df.index[train_df[FOLD_COL] == f]] = y_va_pred
        y_te_pred = clf.predict_proba(X_te, iteration_range=(0, best_it + 1) if best_it is not None else None)[:, 1]
        test_preds.append(y_te_pred)
        best_its.append(best_it if best_it is not None else params_use.get("n_estimators", 1000))
        print("Fold %d: AUC=%.6f | best_iteration=%s", f+1, fold_auc, str(best_it))

    oof_auc = roc_auc_score(train_df[TARGET_COL].values, oof)
    y_test_cv = np.mean(np.vstack(test_preds), axis=0)
    print("CV complete. OOF AUC=%.6f | per-fold AUCs=%s | median best_it=%d",
                 oof_auc, [round(a, 6) for a in fold_aucs], int(np.median(best_its)))
    return y_test_cv, oof_auc, int(np.median(best_its)), feats

def refit_full_and_predict(train_df, test_df, params, debug=False):
    print("Refit on all training data with inner OOF encoders; no held-out validation.")
    # Assign inner folds (for encoders) deterministically
    train_df_full = assign_outer_folds(train_df, n_splits=5, seed=2025)
    X_tr_full, y_tr_full, X_va_dummy, y_va_dummy, X_te_full, feats_full = preprocess_for_outer_fold(
        train_df_full, test_df, held_out_fold=-1,
        sel_cat=sel_cat, sel_num_for_deviation=sel_num_for_deviation, transform_targets=transform_targets,
        all_features_for_te=all_features_for_te
    )
    params_use = params.copy()
    if debug:
        if len(X_tr_full) > 1000:
            X_tr_full, _, y_tr_full, _ = train_test_split(X_tr_full, y_tr_full, test_size=(1.0 - 1000/len(X_tr_full)), stratify=y_tr_full, random_state=2025)
        params_use["n_estimators"] = min(200, params_use.get("n_estimators", 1500))
        params_use["early_stopping_rounds"] = min(20, params_use.get("early_stopping_rounds", 100))

    # For full refit, use training data as eval_set just to track rounds; acceptable since encoders are fixed.
    clf_full = xgb.XGBClassifier(**params_use)
    clf_full.fit(X_tr_full, y_tr_full, eval_set=[(X_tr_full, y_tr_full)], verbose=False)
    best_it_full = getattr(clf_full, "best_iteration", None)
    y_test_full = clf_full.predict_proba(X_te_full, iteration_range=(0, best_it_full + 1) if best_it_full is not None else None)[:, 1]
    print("Full refit complete. best_iteration=%s | n_features=%d", str(best_it_full), X_tr_full.shape[1])
    return y_test_full, best_it_full, feats_full

# -------------------------
# Main pipeline runs twice: DEBUG then FULL
# -------------------------
def run_pipeline(DEBUG: bool):
    mode = "DEBUG" if DEBUG else "FULL"
    print("===== Running in %s mode =====", mode)

    # Create a single 5-fold assignment for baseline/tuning on fold 0
    train_folds = assign_outer_folds(train.copy(), n_splits=5, seed=2025)
    # Preprocess for fold 0 for baseline/tuning
    X_tr0, y_tr0, X_va0, y_va0, X_te0, feats0 = preprocess_for_outer_fold(
        train_folds, test.copy(), held_out_fold=0,
        sel_cat=sel_cat, sel_num_for_deviation=sel_num_for_deviation, transform_targets=transform_targets,
        all_features_for_te=all_features_for_te
    )

    # Baseline params (reduced trees in DEBUG)
    if DEBUG:
        base_params = build_xgb_params(base_lr=0.05, n_estimators=150, early_stopping_rounds=20)
        # Downsample training to 1000 rows for the initial fold-0 baseline
        if len(X_tr0) > 1000:
            X_tr0, _, y_tr0, _ = train_test_split(X_tr0, y_tr0, test_size=(1.0 - 1000/len(X_tr0)), stratify=y_tr0, random_state=2025)
    else:
        base_params = build_xgb_params(base_lr=0.05, n_estimators=1500, early_stopping_rounds=100)

    print("Baseline training on fold 0 (purpose: establish reference AUC).")
    model_base, auc_base, best_it_base, t_base = train_xgb_single(X_tr0, y_tr0, X_va0, y_va0, base_params, label="baseline-fold0")

    # Tuning only in FULL mode
    if not DEBUG:
        model_tuned, auc_tuned, best_it_tuned, tuned_params = optuna_tune_xgb(
            X_tr0, y_tr0, X_va0, y_va0, base_params, time_budget_sec=300
        )
        if auc_tuned >= auc_base:
            final_params = tuned_params
            print("Selected tuned params (AUC=%.6f >= baseline %.6f).", auc_tuned, auc_base)
        else:
            final_params = base_params
            print("Selected baseline params (AUC=%.6f > tuned %.6f).", auc_base, auc_tuned)
    else:
        final_params = base_params
        print("DEBUG mode: tuning skipped; using baseline params.")

    # CV training + predictions
    y_test_cv, oof_auc, median_best_it, feats_cv = train_xgb_cv_and_predict(
        train.copy(), test.copy(), final_params, n_splits=5, debug=DEBUG
    )

    if DEBUG:
        print("DEBUG mode: submission generation skipped per requirements.")
        return

    # Full refit + predictions
    y_test_full, best_it_full, feats_full = refit_full_and_predict(
        train.copy(), test.copy(), final_params, debug=False
    )

    # Blend CV ensemble with full-refit model (simple mean)
    y_test_final = 0.5 * y_test_cv + 0.5 * y_test_full
    y_test_final = np.clip(y_test_final, 1e-9, 1 - 1e-9)

    # Write submission
    submission = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET_COL: y_test_final})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Submission written to %s", SUBMISSION_PATH)

    # Log prediction distribution
    desc = pd.Series(y_test_final).describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
    print("Prediction distribution summary:\n%s", desc.to_string())
    print("Run complete: OOF AUC=%.6f | median_best_it(CV)=%d | device=%s", oof_auc, median_best_it, XGB_DEVICE)

# -------------------------
# Execute: DEBUG then FULL
# -------------------------
run_pipeline(DEBUG=True)   # no submission
run_pipeline(DEBUG=False)




