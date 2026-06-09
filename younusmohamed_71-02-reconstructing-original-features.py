VERSION = "reconstruct_014"
TIME_LIMIT_HOURS = 11.5
N_SPLITS = 5
SEED = 42

# Models to run in HPO 
MODELS_TO_RUN = [
    # "xgboost_gpu",
    "lightgbm_gpu",
    # "catboost_gpu",
    # "logreg",
    # "hist_gbdt",
]

# Competition metadata (kept for consistency)
TARGET = "diagnosed_diabetes"
ID_COL = "id"
SUB_TARGET_COL = "diagnosed_diabetes"

TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/playground-series-s5e12/sample_submission.csv"

ORIG_PATH = "/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv" 

# Reconstructed files to read (no reconstruction done here)
RECON_TRAIN_PATH = "/kaggle/input/s05e12-outputs-diabetes-prediction/train_reconstructed_vreconstruct_007.csv"
RECON_TEST_PATH  = "/kaggle/input/s05e12-outputs-diabetes-prediction/test_reconstructed_vreconstruct_007.csv"

# Output
OUTPUT_DIR = f"model_outputs_v{VERSION}"
RESULTS_CSV = f"{OUTPUT_DIR}/results_v{VERSION}.csv"              # general model results (combos)
HPO_RESULTS_CSV = f"{OUTPUT_DIR}/hpo_trials_v{VERSION}.csv"       # every HPO trial row

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

SKIP_COMBOS = {
    "insulin_level",
    "glucose_postprandial",
    "hba1c",
}
print("Manually skipping combos:", sorted(SKIP_COMBOS))

print("Running version:", VERSION)
print("Models:", MODELS_TO_RUN)
print("Time limit (hours):", TIME_LIMIT_HOURS)


import os, time, gc, warnings, json, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import combinations
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# LightGBM / XGBoost / CatBoost
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

# Extra models (kept to preserve structure)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


# Timing
START_TIME = time.time()
def time_up():
    return (time.time() - START_TIME) >= (TIME_LIMIT_HOURS * 3600)

def seconds_to_str(s):
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)
print("Setup complete.")


train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_SUB_PATH)
orig  = pd.read_csv(ORIG_PATH)

# Features
feature_cols_base = [c for c in train.columns if c not in [TARGET, ID_COL]]

print("Files OK.")
print("Rows: train", len(train), "| test", len(test))
print("Original rows:", len(orig))
print("Train features:", len(feature_cols_base))


# Read reconstructed data only
train_recon = pd.read_csv(RECON_TRAIN_PATH)
test_recon  = pd.read_csv(RECON_TEST_PATH)

print("Loaded reconstructed files.")
print("train_recon shape:", train_recon.shape, "| test_recon shape:", test_recon.shape)

# Feature inventory
feature_cols = [c for c in train_recon.columns if c not in [TARGET, ID_COL]]
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_recon[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]
print(f"FINAL Features: {len(feature_cols)} | Numeric={len(num_cols)} | Categorical={len(cat_cols)}")


# Identify shared columns between Original and comp data (exclude target/id-like)
exclude_cols = {TARGET, ID_COL}
shared_cols = [c for c in feature_cols_base if c in orig.columns and c not in exclude_cols]

# Missing features = columns present in Original but NOT in comp data
orig_feature_cols = [c for c in orig.columns if c not in exclude_cols and c != TARGET]
missing_from_comp = [c for c in orig_feature_cols if c not in train.columns and c not in test.columns]

print(f"Shared cols with Original: {len(shared_cols)}")
print(f"Missing features to reconstruct from Original: {len(missing_from_comp)}")
missing_from_comp[:15]


def write_results_row(row_dict, results_csv=RESULTS_CSV):
    """Append one row to a CSV (create if not exists)."""
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        out = pd.concat([prev, df_row], ignore_index=True)
    else:
        out = df_row
    out.to_csv(results_csv, index=False)

def _safe_write_hpo(row_dict, csv_path=HPO_RESULTS_CSV):
    """Append HPO trial rows (AUC + paths + params)."""
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        out = pd.concat([prev, df_row], ignore_index=True)
    else:
        out = df_row
    out.to_csv(csv_path, index=False)

def save_oof_and_sub(model_name, oof, test_pred, ids_train, ids_test):
    """Save OOF and SUB with a model_name/tag (unique per trial/combo)."""
    oof_path = f"{OUTPUT_DIR}/oof_{model_name}_v{VERSION}.csv"
    sub_path = None
    pd.DataFrame({ID_COL: ids_train, "oof_pred": oof}).to_csv(oof_path, index=False)
    if test_pred is not None and ids_test is not None:
        sub_path = f"{OUTPUT_DIR}/sub_{model_name}_v{VERSION}.csv"
        pd.DataFrame({ID_COL: ids_test, SUB_TARGET_COL: test_pred}).to_csv(sub_path, index=False)
    return oof_path, sub_path


# Reconstruction Preprocessor using shared inputs only
def make_recon_preprocessor(df, shared_cols):
    num_cols = [c for c in shared_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in shared_cols if c not in num_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32))
            ]), cat_cols),
        ],
        remainder="drop"
    )
    return preproc, num_cols, cat_cols

def write_results_row(row_dict, results_csv=RESULTS_CSV):
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        out = pd.concat([prev, df_row], ignore_index=True)
    else:
        out = df_row
    out.to_csv(results_csv, index=False)

def save_oof_and_sub(model_name, oof, test_pred, ids_train, ids_test):
    oof_path = f"{OUTPUT_DIR}/oof_{model_name}.csv"
    sub_path = f"{OUTPUT_DIR}/sub_{model_name}.csv" if (test_pred is not None and ids_test is not None) else None
    pd.DataFrame({ID_COL: ids_train, "oof_pred": oof}).to_csv(oof_path, index=False)
    if sub_path:
        pd.DataFrame({ID_COL: ids_test, SUB_TARGET_COL: test_pred}).to_csv(sub_path, index=False)
    return oof_path, sub_path

def fit_predict_lightgbm(est, Xtr, ytr, Xva, yva, Xte):
    est.set_params(bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED, verbosity=-1)
    est.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(0)]
    )
    va = est.predict_proba(Xva)[:,1]
    te = est.predict_proba(Xte)[:,1] if Xte is not None else None
    return va, te

def fit_predict_xgb(est, Xtr, ytr, Xva, yva, Xte):
    est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    va = est.predict_proba(Xva)[:,1]
    te = est.predict_proba(Xte)[:,1] if Xte is not None else None
    return va, te

def fit_predict_cat(est, Xtr, ytr, Xva, yva, Xte):
    trp = Pool(Xtr, ytr); vap = Pool(Xva, yva)
    est.fit(trp, eval_set=vap, verbose=False, use_best_model=True)
    va = est.predict_proba(Xva)[:,1]
    te = est.predict_proba(Xte)[:,1] if Xte is not None else None
    return va, te


def reconstruct_missing_features(train_df, test_df, orig_df, shared_cols, missing_list, sample_rows=800_000):
    if len(missing_list) == 0:
        return train_df.copy(), test_df.copy(), {}

    # (Optional) subsample Original for speed if huge
    if len(orig_df) > sample_rows:
        orig_fit = orig_df.sample(sample_rows, random_state=SEED).reset_index(drop=True)
    else:
        orig_fit = orig_df.copy()

    # Keep only rows with non-missing shared inputs (if a shared col is missing in orig, skip it)
    available_shared = [c for c in shared_cols if c in orig_fit.columns]
    if len(available_shared) == 0:
        print("[Reconstruct] No shared input columns found in Original; skipping reconstruction.")
        return train_df.copy(), test_df.copy(), {}

    orig_fit = orig_fit.dropna(subset=available_shared, how="any")

    preproc, num_cols, cat_cols = make_recon_preprocessor(orig_fit, available_shared)
    X_all = orig_fit[available_shared]

    # Fit the preprocessor on Original (shared inputs)
    X_all_t = preproc.fit_transform(X_all)

    fitted = {}  # feature_name -> (estimator, label_encoder or None)

    # Transform competition data once
    X_tr_shared = preproc.transform(train_df[available_shared])
    X_te_shared = preproc.transform(test_df[available_shared])

    recon_train = train_df.copy()
    recon_test  = test_df.copy()

    for feat in missing_list:
        if feat not in orig_fit.columns:
            print(f"[Skip] {feat}: not found in Original.")
            continue

        y_raw = orig_fit[feat]
        cat_target = is_categorical_series(y_raw)
        y, le = coerce_target_type(y_raw, cat_target)
        est = build_reconstruction_model(y, cat_target)

        # Small holdout for sanity
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_all_t, y, test_size=0.15, random_state=SEED, stratify=None
        )
        est.fit(X_tr, y_tr)

        if cat_target:
            y_tr_hat = est.predict(X_tr_shared)
            y_te_hat = est.predict(X_te_shared)
            if le is not None:
                inv_tr = le.inverse_transform(y_tr_hat.astype(int))
                inv_te = le.inverse_transform(y_te_hat.astype(int))
                recon_train[feat] = inv_tr
                recon_test[feat]  = inv_te
            else:
                recon_train[feat] = y_tr_hat
                recon_test[feat]  = y_te_hat
        else:
            tr_hat = est.predict(X_tr_shared)
            te_hat = est.predict(X_te_shared)
            try:
                if pd.api.types.is_integer_dtype(orig_df[feat].dtype):
                    tr_hat = np.rint(tr_hat).astype(np.int64)
                    te_hat = np.rint(te_hat).astype(np.int64)
            except Exception:
                pass
            recon_train[feat] = tr_hat
            recon_test[feat]  = te_hat

        fitted[feat] = (est, le)
        print(f"[Reconstructed] {feat} | type={'cat' if cat_target else 'num'}")

    return recon_train, recon_test, fitted


preproc_ohe_sparse = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
        ]), cat_cols),
    ],
    remainder="drop"
)

preproc_ordscale = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler(with_mean=True))]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_cols),
    ],
    remainder="drop"
)

def preprocessor_for_model(model_name):
    if model_name in ["xgboost_gpu", "lightgbm_gpu", "catboost_gpu"]:
        return preproc_ohe_sparse
    if model_name in ["logreg"]:
        return preproc_ordscale
    if model_name in ["hist_gbdt"]:
        return preproc_ohe_sparse
    return preproc_ohe_sparse


def make_xgboost_gpu():
    return XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.0,
        reg_lambda=0.0,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        objective="binary:logistic",
        eval_metric="auc",
        random_state=SEED
    )

def make_lightgbm_gpu():
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=63,
        subsample=0.90,
        colsample_bytree=0.80,
        min_data_in_leaf=25,
        reg_alpha=0.0,
        reg_lambda=0.0,
        device="gpu",
        random_state=SEED,
        verbosity=-1
    )

def make_catboost_gpu():
    return CatBoostClassifier(
        iterations=4000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        random_seed=SEED,
        verbose=False
    )

def make_logreg():
    return LogisticRegression(max_iter=4000, solver="lbfgs", n_jobs=-1)

def make_hist_gbdt():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=None,
        max_bins=255,
        early_stopping=True,
        l2_regularization=0.0,
        random_state=SEED
    )

FACTORIES = {
    "xgboost_gpu": make_xgboost_gpu,
    "lightgbm_gpu": make_lightgbm_gpu,
    "catboost_gpu": make_catboost_gpu,
    "logreg": make_logreg,
    "hist_gbdt": make_hist_gbdt,
}


RECON_FEATURES = [
    "glucose_fasting",
    "glucose_postprandial",
    "insulin_level",
    "hba1c",
    "diabetes_risk_score",
    "diabetes_stage",
]

# Base = all features except TARGET/ID and reconstructed ones
feature_cols_base_no_recon = [
    c for c in train_recon.columns
    if c not in [TARGET, ID_COL] and c not in RECON_FEATURES
]

assert set(RECON_FEATURES).issubset(train_recon.columns)
assert set(RECON_FEATURES).issubset(test_recon.columns)

print(f"Base (no recon) count: {len(feature_cols_base_no_recon)}")
print("Reconstructed:", RECON_FEATURES)

def _combo_name(feats_tuple):
    return "+".join(sorted(feats_tuple)) if feats_tuple else "none"

# Build ALL combos in ascending order of size (1..len)
from itertools import combinations

ALL_COMBOS_FOR_HPO = []
for k in range(1, len(RECON_FEATURES) + 1):  # 1..6 (full set last)
    for comb in combinations(RECON_FEATURES, k):
        ALL_COMBOS_FOR_HPO.append(tuple(sorted(comb)))

print(f"Total planned combos (should be 63): {len(ALL_COMBOS_FOR_HPO)}")
print("First few (k=1):", [ _combo_name(c) for c in ALL_COMBOS_FOR_HPO if len(c)==1 ][:6])
print("Last (k=6):", _combo_name(ALL_COMBOS_FOR_HPO[-1]))


def _build_combo_preprocessors(feature_list, df_ref):
    num_cols_c = [c for c in feature_list if pd.api.types.is_numeric_dtype(df_ref[c])]
    cat_cols_c = [c for c in feature_list if c not in num_cols_c]

    preproc_ohe_sparse_combo = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols_c),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
            ]), cat_cols_c),
        ],
        remainder="drop"
    )

    preproc_ordscale_combo = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler(with_mean=True))]), num_cols_c),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), cat_cols_c),
        ],
        remainder="drop"
    )
    return preproc_ohe_sparse_combo, preproc_ordscale_combo

def _pick_preproc_for_model_combo(model_name, feature_list, df_ref):
    p_ohe, p_ord = _build_combo_preprocessors(feature_list, df_ref)
    if model_name in ["xgboost_gpu", "lightgbm_gpu", "catboost_gpu", "hist_gbdt"]:
        return p_ohe
    if model_name in ["logreg"]:
        return p_ord
    return p_ohe


# Strong LightGBM seeds (3 examples)
LGB_SEEDS = [
    # seed 1
    dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
         n_estimators=4000, subsample=0.90, colsample_bytree=0.80,
         min_child_samples=25, reg_alpha=0.0, reg_lambda=0.0,
         device="gpu", random_state=SEED, verbosity=-1),
    # seed 2
    dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=64,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
         n_estimators=5000, verbosity=-1, random_state=SEED, device="gpu"),
    # seed 3 (tuned-like)
    dict(objective="binary", metric="auc",
         learning_rate=0.0592, num_leaves=26, max_depth=4,
         lambda_l1=1.34, lambda_l2=3.1e-07, min_child_samples=95,
         n_estimators=5000, colsample_bytree=0.565, subsample=0.975,
         random_state=133, verbosity=-1, device="gpu")
]
print(f"LGB seeds: {len(LGB_SEEDS)}")


def _clip(x, lo, hi): 
    return max(lo, min(hi, x))

def _jitter_float(v, rel=0.3, lo=None, lo_b=None, hi=None):
    """Relative Gaussian jitter; optionally clip to [lo_b, hi]."""
    nv = float(v) * (1.0 + np.random.randn()*rel)
    if lo is not None:
        nv = max(lo, nv)
    if hi is not None:
        nv = min(hi, nv)
    return nv

def _jitter_int(v, rel=0.3, lo=1, hi=1000):
    return int(_clip(round(v * (1.0 + np.random.randn()*rel)), lo, hi))

def sample_xgb_around(seed):
    p = deepcopy(seed)
    p["learning_rate"]     = _jitter_float(p.get("learning_rate", 0.02), 0.35, 0.002, 0.2)
    p["max_depth"]         = _jitter_int(p.get("max_depth", 6), 0.5, 2, 12)
    if "min_child_weight" in seed or random.random() < 0.6:
        p["min_child_weight"] = _jitter_float(seed.get("min_child_weight", 5.0), 0.6, 1.0, 20.0)
    p["subsample"]         = _clip(_jitter_float(p.get("subsample", 0.8), 0.25, 0.5, 1.0), 0.5, 1.0)
    p["colsample_bytree"]  = _clip(_jitter_float(p.get("colsample_bytree", 0.7), 0.25, 0.3, 1.0), 0.3, 1.0)
    p["gamma"]             = abs(_jitter_float(p.get("gamma", 0.0), 1.0, 0.0, 5.0))
    p["reg_alpha"]         = abs(_jitter_float(p.get("reg_alpha", 0.0), 1.5, 0.0, 10.0))
    p["reg_lambda"]        = abs(_jitter_float(p.get("reg_lambda", 1.0), 1.0, 0.0, 20.0))
    p["n_estimators"]      = _jitter_int(p.get("n_estimators", 5000), 0.4, 2000, 12000)
    p["tree_method"]       = "gpu_hist"
    p["eval_metric"]       = "auc"
    p["random_state"]      = SEED
    p["n_jobs"]            = -1
    return p

def sample_lgb_around(seed):
    p = deepcopy(seed)
    p["learning_rate"]     = _jitter_float(p.get("learning_rate", 0.02), 0.35, 0.002, 0.2)
    p["num_leaves"]        = _jitter_int(p.get("num_leaves", 63), 0.5, 7, 255)
    if "max_depth" in seed or random.random() < 0.5:
        p["max_depth"]     = _jitter_int(p.get("max_depth", -1 if "max_depth" not in seed else seed["max_depth"]), 0.6, -1, 16)
    p["min_child_samples"] = _jitter_int(p.get("min_child_samples", 20), 0.6, 5, 200)
    p["subsample"]         = _clip(_jitter_float(p.get("subsample", seed.get("bagging_fraction", 0.8)), 0.25, 0.5, 1.0), 0.5, 1.0)
    p["colsample_bytree"]  = _clip(_jitter_float(p.get("colsample_bytree", seed.get("feature_fraction", 0.8)), 0.25, 0.3, 1.0), 0.3, 1.0)
    p["lambda_l1"]         = abs(_jitter_float(p.get("lambda_l1", seed.get("reg_alpha", 0.0)), 1.5, 0.0, 10.0))
    p["lambda_l2"]         = abs(_jitter_float(p.get("lambda_l2", seed.get("reg_lambda", 0.0)), 1.5, 0.0, 10.0))
    p["n_estimators"]      = _jitter_int(p.get("n_estimators", 4000), 0.4, 2000, 12000)
    p["objective"]         = "binary"
    p["metric"]            = "auc"
    p["device"]            = "gpu"
    p["random_state"]      = SEED
    p["verbosity"]         = -1
    # canonicalize
    p.pop("feature_fraction", None)
    p.pop("bagging_fraction", None)
    p.pop("bagging_freq", None)
    p.pop("seed", None)
    return p

def sample_cat_around(seed):
    p = deepcopy(seed)
    p["learning_rate"]     = _jitter_float(p.get("learning_rate", 0.03), 0.35, 0.002, 0.2)
    p["depth"]             = _jitter_int(p.get("depth", 6), 0.5, 3, 10)
    p["l2_leaf_reg"]       = abs(_jitter_float(p.get("l2_leaf_reg", 3.0), 1.0, 0.5, 20.0))
    p["random_strength"]   = abs(_jitter_float(p.get("random_strength", 1.0), 1.0, 0.0, 10.0))
    if p.get("bootstrap_type", "Bayesian") == "Bayesian":
        p["bagging_temperature"] = _clip(_jitter_float(p.get("bagging_temperature", 0.8), 0.5, 0.0, 5.0), 0.0, 5.0)
    p["iterations"]        = _jitter_int(p.get("iterations", 5000), 0.4, 2000, 12000)
    p["loss_function"]     = "Logloss"
    p["eval_metric"]       = "AUC"
    p["task_type"]         = "GPU"
    p["random_seed"]       = SEED
    p["verbose"]           = 0 if "verbose" in p else False
    return p

def _seed_list_for(model_name):
    if model_name == "xgboost_gpu":
        return XGB_SEEDS, sample_xgb_around
    if model_name == "lightgbm_gpu":
        return LGB_SEEDS, sample_lgb_around
    if model_name == "catboost_gpu":
        return CAT_SEEDS, sample_cat_around
    return [], None


def _evaluate_params_cv_with_save(model_name,
                                  params,
                                  train_df,
                                  test_df,
                                  feature_cols_local,
                                  trial_tag,
                                  n_splits=5):
    """Run CV with given params; save OOF/SUB; return AUC mean/std and paths."""
    X = train_df[feature_cols_local].copy()
    y = train_df[TARGET].astype(int).values
    X_te = test_df[feature_cols_local].copy()

    preproc = _pick_preproc_for_model_combo(model_name, feature_cols_local, train_df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train_df), dtype=np.float32)
    test_preds = []
    fold_aucs = []

    def _build_estimator():
        if model_name == "xgboost_gpu":
            return XGBClassifier(**params)
        if model_name == "lightgbm_gpu":
            return lgb.LGBMClassifier(**params)
        if model_name == "catboost_gpu":
            return CatBoostClassifier(**params)
        if model_name == "logreg":
            return LogisticRegression(max_iter=4000, solver="lbfgs", n_jobs=-1)
        return HistGradientBoostingClassifier(random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        if time_up():
            print(f"[{model_name}|{trial_tag}] Time limit reached mid-CV.")
            break

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        fp = preproc.fit(X_tr)
        Xtr_t = fp.transform(X_tr)
        Xva_t = fp.transform(X_va)
        Xte_t = fp.transform(X_te)

        est = _build_estimator()

        if model_name == "lightgbm_gpu":
            est.set_params(bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED, verbosity=-1)
            est.fit(
                Xtr_t, y_tr,
                eval_set=[(Xva_t, y_va)],
                eval_metric="auc",
                callbacks=[early_stopping(200, verbose=False), log_evaluation(0)]
            )
            va_pred = est.predict_proba(Xva_t)[:, 1]
            te_pred = est.predict_proba(Xte_t)[:, 1]
        elif model_name == "xgboost_gpu":
            est.fit(Xtr_t, y_tr, eval_set=[(Xva_t, y_va)], verbose=False)
            va_pred = est.predict_proba(Xva_t)[:, 1]
            te_pred = est.predict_proba(Xte_t)[:, 1]
        elif model_name == "catboost_gpu":
            trp = Pool(Xtr_t, y_tr); vap = Pool(Xva_t, y_va); tep = Pool(Xte_t)
            est.fit(trp, eval_set=vap, verbose=False, use_best_model=True)
            va_pred = est.predict_proba(Xva_t)[:, 1]
            te_pred = est.predict_proba(Xte_t)[:, 1]
        else:
            est_use = est
            if not hasattr(est_use, "predict_proba"):
                est_use = CalibratedClassifierCV(est_use, method="isotonic", cv=3)
            est_use.fit(Xtr_t, y_tr)
            va_pred = est_use.predict_proba(Xva_t)[:, 1]
            te_pred = est_use.predict_proba(Xte_t)[:, 1]

        oof[va_idx] = va_pred
        test_preds.append(te_pred)
        fold_aucs.append(roc_auc_score(y_va, va_pred))

        del Xtr_t, Xva_t, Xte_t
        gc.collect()

    if not fold_aucs:
        return None, None, None, None

    cv_mean, cv_std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    test_pred = np.mean(test_preds, axis=0) if len(test_preds) else None

    tag = f"hpo_{trial_tag}"
    oof_path, sub_path = save_oof_and_sub(
        tag, oof, test_pred,
        train_df[ID_COL].values,
        (test_df[ID_COL].values if test_pred is not None else None)
    )
    return cv_mean, cv_std, oof_path, sub_path


# Optional global prefix to tag per-combo files/rows
HPO_TRIAL_PREFIX = None

def run_hpo_for_model(model_name,
                      train_df,
                      test_df,
                      feature_cols_local,
                      max_trials_per_seed=5,
                      subsample_rows=None,   # set to None to use all rows
                      cv_splits=5):
    seeds, sampler = _seed_list_for(model_name)
    if not seeds or sampler is None:
        print(f"[HPO] {model_name}: no HPO seeds configured; skipping.")
        return None

    # (Optionally) subsample for speed — disabled by default
    train_use = train_df

    best = {"auc": -1.0, "std": None, "params": None, "seed_idx": None, "trial_idx": None}

    for s_idx, seed in enumerate(seeds):
        if time_up(): break

        prefix = (HPO_TRIAL_PREFIX + "_") if HPO_TRIAL_PREFIX else ""
        auc, std, oof_path, sub_path = _evaluate_params_cv_with_save(
            model_name, seed, train_use, test_df, feature_cols_local,
            trial_tag=f"{prefix}{model_name}_seed{s_idx}", n_splits=cv_splits
        )
        _safe_write_hpo({
            "version": VERSION, "model": model_name,
            "kind": "seed", "seed_index": s_idx, "trial_index": -1,
            "cv_auc_mean": (-1 if auc is None else auc),
            "cv_auc_std":  (-1 if std is not None else -1),
            "oof_path": oof_path or "",
            "sub_path": sub_path or "",
            "combo_name": (HPO_TRIAL_PREFIX or "").replace("combo_",""),
            "feature_count": len(feature_cols_local),
            "params_json": json.dumps(seed)
        })
        if (auc is not None) and (auc > best["auc"]):
            best = {"auc": auc, "std": std, "params": deepcopy(seed),
                    "seed_idx": s_idx, "trial_idx": "seed"}

        # Jittered trials around this seed
        for t in range(max_trials_per_seed):
            if time_up(): break
            cand = sampler(seed)
            auc, std, oof_path, sub_path = _evaluate_params_cv_with_save(
                model_name, cand, train_use, test_df, feature_cols_local,
                trial_tag=f"{prefix}{model_name}_seed{s_idx}_trial{t}", n_splits=cv_splits
            )
            _safe_write_hpo({
                "version": VERSION, "model": model_name,
                "kind": "sample", "seed_index": s_idx, "trial_index": t,
                "cv_auc_mean": (-1 if auc is None else auc),
                "cv_auc_std":  (-1 if std is not None else -1),
                "oof_path": oof_path or "",
                "sub_path": sub_path or "",
                "combo_name": (HPO_TRIAL_PREFIX or "").replace("combo_",""),
                "feature_count": len(feature_cols_local),
                "params_json": json.dumps(cand)
            })
            if (auc is not None) and (auc > best["auc"]):
                best = {"auc": auc, "std": std, "params": deepcopy(cand),
                        "seed_idx": s_idx, "trial_idx": t}
                print(f"[HPO|{model_name}|{HPO_TRIAL_PREFIX}] **new best** AUC {auc:.6f} (seed {s_idx}, trial {t})")

    if best["params"] is not None:
        print(f"[HPO|{model_name}|{HPO_TRIAL_PREFIX}] BEST AUC {best['auc']:.6f} (seed {best['seed_idx']}, trial {best['trial_idx']})")
    else:
        print(f"[HPO|{model_name}|{HPO_TRIAL_PREFIX}] No successful trials.")
    return best


# # Which combos to run HPO on
# combos_for_hpo = remaining_combos  # contains k=1..5 plus FULL_SET (k=6)

# print("\n=== HPO on selected combos (saving OOF & SUB for every trial) ===")
# for comb in combos_for_hpo:
#     if time_up():
#         print("\n=== Global time limit reached. Stopping HPO. ===")
#         break

#     combo_name = _combo_name(comb)
#     combo_features = feature_cols_base_no_recon + list(comb)
#     print(f"\n--- HPO for combo={combo_name} | features={len(combo_features)} ---")
#     HPO_TRIAL_PREFIX = f"combo_{combo_name}"

#     for m in MODELS_TO_RUN:
#         print(f"[Run] model={m} | combo={combo_name}")
#         run_hpo_for_model(
#             m,
#             train_recon,
#             test_recon,
#             combo_features,
#             max_trials_per_seed=5,
#             subsample_rows=None,   # set to int if you want faster HPO
#             cv_splits=N_SPLITS
#         )

# # reset tag
# HPO_TRIAL_PREFIX = None
# print("\n=== HPO phase complete ===")


if time_up():
    print("\n=== Global time limit already reached; skipping resume. ===")
else:
    print("\n=== Resuming HPO for pending (model, combo) pairs) in ascending combo-size order ===")
    attempted_pairs = 0
    skipped_pairs = 0

    # Iterate strictly by size: k = 1, 2, ..., len(RECON_FEATURES)
    for k in range(1, len(RECON_FEATURES) + 1):
        print(f"\n--- Combo size k={k} ---")
        combos_k = [c for c in ALL_COMBOS_FOR_HPO if len(c) == k]

        for comb in combos_k:
            if time_up():
                print("\n=== Time limit hit. Stopping resume. ===")
                break

            combo_name = _combo_name(comb)

            # Manual skip only for single-feature combos marked as done
            if k == 1 and combo_name in SKIP_COMBOS:
                print(f"[skip] combo={combo_name} (manually marked complete for ALL models)")
                skipped_pairs += len(MODELS_TO_RUN)
                continue

            combo_features = feature_cols_base_no_recon + list(comb)

            # Tag all trial files/rows with the combo name
            HPO_TRIAL_PREFIX = f"combo_{combo_name}"

            for m in MODELS_TO_RUN:
                if time_up():
                    print("\n=== Time limit hit mid-combo. Stopping. ===")
                    break

                print(f"\n[RUN] HPO -> model={m} | combo={combo_name} | d={len(combo_features)} features")
                _ = run_hpo_for_model(
                    m,
                    train_recon,
                    test_recon,
                    combo_features,
                    max_trials_per_seed=5,    
                    subsample_rows=300_000,   
                    cv_splits=N_SPLITS
                )
                attempted_pairs += 1

            # clear tag to avoid accidental reuse later
            HPO_TRIAL_PREFIX = None

        if time_up():
            break

    print(f"\nDone. Attempted pairs: {attempted_pairs} | Skipped pairs (manual k=1): {skipped_pairs}")


if os.path.exists(HPO_RESULTS_CSV):
    hpo_df = pd.read_csv(HPO_RESULTS_CSV)
    # Clean display
    cols_show = ["model","combo_name","kind","seed_index","trial_index","feature_count",
                 "cv_auc_mean","cv_auc_std","oof_path","sub_path"]
    cols_show = [c for c in cols_show if c in hpo_df.columns]
    print("\nTop trials by AUC:")
    display(hpo_df.sort_values("cv_auc_mean", ascending=False)[cols_show].head(40))

    print("\nBest per (model, combo):")
    best_per_model_combo = (hpo_df.sort_values("cv_auc_mean", ascending=False)
                            .groupby(["model","combo_name"], as_index=False).first())
    display(best_per_model_combo[["model","combo_name","cv_auc_mean","cv_auc_std","oof_path","sub_path"]]
            .sort_values("cv_auc_mean", ascending=False))
else:
    print("No HPO trials CSV found yet.")




