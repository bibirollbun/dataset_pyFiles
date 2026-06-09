VERSION = "FE_014"
TIME_LIMIT_HOURS = 11.5
N_SPLITS = 5
SEED = 42

# Select model(s) to run
MODELS_TO_RUN = [
    # "xgboost_gpu",
    "lightgbm_gpu",
    # "catboost_gpu",
    # "logreg",
    # "hist_gbdt",
]

TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/playground-series-s5e12/sample_submission.csv"

USE_PREBUILT_RECON = True
RECON_TRAIN_PATH = "/kaggle/input/s05e12-outputs-diabetes-prediction/train_reconstructed_vreconstruct_007.csv"
RECON_TEST_PATH  = "/kaggle/input/s05e12-outputs-diabetes-prediction/test_reconstructed_vreconstruct_007.csv"

# Read engineered feature sets created earlier
USE_PREBUILT_ENGINEERED = True
ENGINEERED_DIR = "/kaggle/input/s05e12-outputs-diabetes-prediction"
ENGINEERED_VERSION = "FE_002"   # matches your saved engineered filenames

TARGET = "diagnosed_diabetes"
ID_COL = "id"
SUB_TARGET_COL = "diagnosed_diabetes"

# Paths to 6 feature-importance CSVs (LightGBM GPU)
FI_PATHS = {
    "auto_100": "/kaggle/input/s05e12-outputs-diabetes-prediction/71_03-feat_importance_lightgbm_gpu_auto_100_vFE_008.csv",
    "auto_200": "/kaggle/input/s05e12-outputs-diabetes-prediction/71_03-feat_importance_lightgbm_gpu_auto_200_vFE_008.csv",
    "auto_20":  "/kaggle/input/s05e12-outputs-diabetes-prediction/71_03-feat_importance_lightgbm_gpu_auto_20_vFE_008.csv",
    "auto_50":  "/kaggle/input/s05e12-outputs-diabetes-prediction/71_03-feat_importance_lightgbm_gpu_auto_50_vFE_008.csv",
    "raw_eda":  "/kaggle/input/s05e12-outputs-diabetes-prediction/71_03-feat_importance_lightgbm_gpu_raw_eda_vFE_008.csv",
    "recon_fe": "/kaggle/input/s05e12-outputs-diabetes-prediction/71_03-feat_importance_lightgbm_gpu_recon_fe_vFE_008.csv",
}

TOPK_LIST = [5, 10, 15, 20, 25]
WEIGHT_EPS = 1e-3  # to avoid zeroing features during weighting

# Output
import os
OUTPUT_DIR = f"model_outputs_v{VERSION}"
RESULTS_CSV = f"{OUTPUT_DIR}/results_v{VERSION}.csv"

# NEW: HPO trial log path (fixes NameError later)
HPO_RESULTS_CSV = f"{OUTPUT_DIR}/hpo_trials_v{VERSION}.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Running version:", VERSION)
print("Models:", MODELS_TO_RUN)
print("Time limit (hours):", TIME_LIMIT_HOURS)


import os, time, gc, warnings, json, math, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif

# LightGBM / XGBoost / CatBoost
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

# Extra models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


START_TIME = time.time()
def time_up():
    return (time.time() - START_TIME) >= (TIME_LIMIT_HOURS * 3600)

def seconds_to_str(s):
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def set_seed(seed=SEED):
    np.random.seed(seed); random.seed(seed)
set_seed(SEED)

print("Setup complete.")


train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_SUB_PATH)

feature_cols_base = [c for c in train.columns if c not in [TARGET, ID_COL]]

if USE_PREBUILT_RECON:
    train_recon = pd.read_csv(RECON_TRAIN_PATH)
    test_recon  = pd.read_csv(RECON_TEST_PATH)
    if TARGET in train_recon.columns and TARGET in train.columns:
        try: train_recon[TARGET] = train_recon[TARGET].astype(train[TARGET].dtype)
        except: pass
    for df, ref in [(train_recon, train), (test_recon, test)]:
        if ID_COL in df.columns and ID_COL in ref.columns:
            try: df[ID_COL] = df[ID_COL].astype(ref[ID_COL].dtype)
            except: pass
    print("Loaded reconstructed CSVs:",
          f"train_recon {train_recon.shape} | test_recon {test_recon.shape}")
else:
    train_recon, test_recon = train.copy(), test.copy()

print("Files OK.")
print("Rows: train", len(train), "| test", len(test))
print("Train features:", len(feature_cols_base))


def get_num_cat_cols(df, exclude=None):
    exclude = set(exclude or [])
    num_cols = [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns
                if c not in exclude and c not in num_cols]
    return num_cols, cat_cols

def safe_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(b==0, np.nan, a/b)
    return out

def add_cols(df, new_cols: dict):
    for k, v in new_cols.items():
        df[k] = v
    return df

def cap_outliers(s: pd.Series, q_low=0.01, q_high=0.99):
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    return s.clip(lo, hi)

def log1p_if_positive(s: pd.Series):
    if (s.dropna() >= 0).all():
        return np.log1p(s)
    return s

def quantile_bucket(s: pd.Series, q=5):
    try:
        return pd.qcut(s, q, labels=False, duplicates="drop")
    except Exception:
        return pd.Series(np.nan, index=s.index)

def write_results_row(row_dict, results_csv=RESULTS_CSV):
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        out = pd.concat([prev, df_row], ignore_index=True)
    else:
        out = df_row
    out.to_csv(results_csv, index=False)

def save_oof_and_sub(set_name, model_name, oof, test_pred, ids_train, ids_test):
    oof_path = f"{OUTPUT_DIR}/oof_{model_name}_{set_name}_v{VERSION}.csv"
    sub_path = f"{OUTPUT_DIR}/sub_{model_name}_{set_name}_v{VERSION}.csv"
    pd.DataFrame({ID_COL: ids_train, "oof_pred": oof}).to_csv(oof_path, index=False)
    if test_pred is not None and ids_test is not None:
        pd.DataFrame({ID_COL: ids_test, SUB_TARGET_COL: test_pred}).to_csv(sub_path, index=False)
    else:
        sub_path = None
    return oof_path, sub_path

def save_importance_csv(set_name, model_name, feat_names, fold_importances):
    imp = (pd.DataFrame({"feature": feat_names, "importance": fold_importances})
             .groupby("feature", as_index=False)["importance"].sum()
             .sort_values("importance", ascending=False))
    imp_path = f"{OUTPUT_DIR}/feat_importance_{model_name}_{set_name}_v{VERSION}.csv"
    imp.to_csv(imp_path, index=False)
    return imp_path


from scipy import sparse

def make_preprocessors(train_df, target_col=TARGET, id_col=ID_COL):
    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

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
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=True))
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    return feature_cols, num_cols, cat_cols, preproc_ohe_sparse, preproc_ordscale

def preprocessor_for_model(model_name, train_df):
    feature_cols, num_cols, cat_cols, preproc_ohe_sparse, preproc_ordscale = make_preprocessors(train_df)
    if model_name in ["xgboost_gpu", "lightgbm_gpu", "catboost_gpu", "hist_gbdt"]:
        return feature_cols, preproc_ohe_sparse
    elif model_name in ["logreg"]:
        return feature_cols, preproc_ordscale
    else:
        return feature_cols, preproc_ohe_sparse


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
        n_estimators=3500,
        learning_rate=0.025,
        num_leaves=63,
        subsample=0.90,
        colsample_bytree=0.80,
        min_child_samples=25,
        reg_alpha=0.0,
        reg_lambda=0.0,
        device="gpu",
        random_state=SEED,
        verbosity=-1
    )

def make_catboost_gpu():
    return CatBoostClassifier(
        iterations=3500,
        learning_rate=0.025,
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


def fe_raw_basic(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr = train_df.copy(); te = test_df.copy()
    exclude = [TARGET, ID_COL]
    num_cols = [c for c in tr.columns if c not in exclude and pd.api.types.is_numeric_dtype(tr[c])]
    cat_cols = [c for c in tr.columns if c not in exclude and c not in num_cols]

    for c in num_cols:
        lo, hi = tr[c].quantile(0.01), tr[c].quantile(0.99)
        tr[f"{c}_cap"] = tr[c].clip(lo, hi); te[f"{c}_cap"] = te[c].clip(lo, hi)
        if (tr[c].dropna() >= 0).all():
            tr[f"{c}_log1p"] = np.log1p(tr[c]); te[f"{c}_log1p"] = np.log1p(te[c])

    tr["num_nan_ratio"] = tr[num_cols].isna().mean(axis=1) if num_cols else 0.0
    te["num_nan_ratio"] = te[num_cols].isna().mean(axis=1) if num_cols else 0.0

    if len(num_cols) >= 4:
        var_rank = tr[num_cols].var().sort_values(ascending=False)
        topk = list(var_rank.index[:6])
        for i in range(len(topk)):
            for j in range(i+1, len(topk)):
                a, b = topk[i], topk[j]
                tr[f"{a}_plus_{b}"]  = tr[a] + tr[b];  te[f"{a}_plus_{b}"]  = te[a] + te[b]
                tr[f"{a}_minus_{b}"] = tr[a] - tr[b];  te[f"{a}_minus_{b}"] = te[a] - te[b]
                tr[f"{a}_ratio_{b}"] = np.where(tr[b]==0, np.nan, tr[a]/tr[b])
                te[f"{a}_ratio_{b}"] = np.where(te[b]==0, np.nan, te[a]/te[b])

    for c in num_cols:
        if tr[c].dropna().skew() > 1.0:
            try:
                tr[f"{c}_q5"] = pd.qcut(tr[c], 5, labels=False, duplicates="drop")
                te[f"{c}_q5"] = pd.qcut(te[c], 5, labels=False, duplicates="drop")
            except Exception:
                pass

    return tr, te

def _consistent_stage_codes(tr_stage: pd.Series, te_stage: pd.Series):
    both = pd.concat([tr_stage, te_stage], axis=0)
    codes = pd.Categorical(both).codes
    tr_codes = pd.Series(codes[:len(tr_stage)], index=tr_stage.index).astype(float)
    te_codes = pd.Series(codes[len(tr_stage):], index=te_stage.index).astype(float)
    return tr_codes, te_codes

def fe_recon_basic(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr = train_df.copy(); te = test_df.copy()
    cols = set(train_df.columns)

    gf  = "glucose_fasting"
    gpp = "glucose_postprandial"
    ins = "insulin_level"
    a1c = "hba1c"
    rs  = "diabetes_risk_score"
    stg = "diabetes_stage"

    if gf in cols and gpp in cols:
        tr["glucose_delta"] = tr[gpp] - tr[gf]; te["glucose_delta"] = te[gpp] - te[gf]
        tr["glucose_ratio"] = safe_div(tr[gpp], tr[gf]); te["glucose_ratio"] = safe_div(te[gpp], te[gf])

    if gf in cols and ins in cols:
        tr["insulin_resistance_proxy"] = safe_div(tr[gf], (tr[ins] + 1e-3))
        te["insulin_resistance_proxy"] = safe_div(te[gf], (te[ins] + 1e-3))

    if a1c in cols:
        tr["a1c_eag"] = 28.7 * tr[a1c] - 46.7; te["a1c_eag"] = 28.7 * te[a1c] - 46.7

    if rs in cols:
        tr["risk_bucket_q5"] = quantile_bucket(tr[rs], q=5); te["risk_bucket_q5"] = quantile_bucket(te[rs], q=5)

    if stg in cols:
        if pd.api.types.is_numeric_dtype(tr[stg]):
            tr["stage_code"] = tr[stg].astype(float); te["stage_code"] = te[stg].astype(float)
        else:
            tr_codes, te_codes = _consistent_stage_codes(tr[stg], te[stg])
            tr["stage_code"] = tr_codes; te["stage_code"] = te_codes

    for c in [x for x in [gf, gpp, ins, a1c, rs] if x in cols]:
        tr[f"{c}_cap"] = cap_outliers(tr[c]); te[f"{c}_cap"] = cap_outliers(te[c])
        tr[f"{c}_log1p"] = log1p_if_positive(tr[c]); te[f"{c}_log1p"] = log1p_if_positive(te[c])

    return tr, te


def auto_generate_features(train_df: pd.DataFrame, test_df: pd.DataFrame, budget=20):
    tr = train_df.copy(); te = test_df.copy()
    exclude = [TARGET, ID_COL]
    num_cols = [c for c in tr.columns if c not in exclude and pd.api.types.is_numeric_dtype(tr[c])]
    cat_cols = [c for c in tr.columns if c not in exclude and c not in num_cols]

    cand = {}
    var_rank = tr[num_cols].var().sort_values(ascending=False) if len(num_cols) else pd.Series(dtype=float)
    topn = list(var_rank.index[:min(12, len(var_rank))])
    for c in topn:
        mu, sd = tr[c].mean(), tr[c].std(ddof=0) or 1.0
        cand[f"{c}_z"] = (tr[c] - mu) / sd
        cand[f"{c}_2"] = tr[c] * tr[c]
    for i in range(len(topn)):
        for j in range(i+1, len(topn)):
            a, b = topn[i], topn[j]
            cand[f"{a}_x_{b}"] = tr[a] * tr[b]
            cand[f"{a}_r_{b}"] = np.where(tr[b]==0, np.nan, tr[a]/tr[b])
    for c in cat_cols:
        if tr[c].nunique(dropna=True) <= 20:
            freq = tr[c].value_counts(dropna=False) / len(tr)
            cand[f"{c}_freq"] = tr[c].map(freq)
    skewed = [c for c in num_cols if tr[c].dropna().skew() > 1.0]
    for c in skewed[:10]:
        try:
            cand[f"{c}_q5auto"] = pd.qcut(tr[c], 5, labels=False, duplicates="drop")
        except Exception:
            pass

    C = pd.DataFrame(index=tr.index, data=cand)
    MI_X = C.copy()
    for col in MI_X.columns:
        if not pd.api.types.is_numeric_dtype(MI_X[col]):
            MI_X[col] = pd.Categorical(MI_X[col]).codes
    mi = mutual_info_classif(MI_X.fillna(-999), tr[TARGET].astype(int), random_state=SEED)
    mi_series = pd.Series(mi, index=MI_X.columns).sort_values(ascending=False)
    keep = list(mi_series.head(min(budget, len(mi_series))).index)

    for k in keep:
        if k.endswith("_z"):
            base = k[:-2]; mu, sd = tr[base].mean(), tr[base].std(ddof=0) or 1.0
            tr[k] = (tr[base] - mu) / sd; te[k] = (te[base] - mu) / sd
        elif k.endswith("_2"):
            base = k[:-2]; tr[k] = tr[base] * tr[base]; te[k] = te[base] * te[base]
        elif "_x_" in k:
            a, b = k.split("_x_"); tr[k] = tr[a] * tr[b]; te[k] = te[a] * te[b]
        elif "_r_" in k:
            a, b = k.split("_r_")
            tr[k] = np.where(tr[b]==0, np.nan, tr[a]/tr[b])
            te[k] = np.where(te[b]==0, np.nan, te[a]/te[b])
        elif k.endswith("_freq"):
            base = k[:-5]; freq = tr[base].value_counts(dropna=False) / len(tr)
            tr[k] = tr[base].map(freq); te[k] = te[base].map(freq)
        elif k.endswith("_q5auto"):
            base = k.replace("_q5auto", "")
            try:
                tr[k] = pd.qcut(tr[base], 5, labels=False, duplicates="drop")
                te[k] = pd.qcut(te[base], 5, labels=False, duplicates="drop")
            except Exception:
                tr[k] = np.nan; te[k] = np.nan
        else:
            tr[k] = C[k]; te[k] = np.nan

    return tr, te, keep, mi_series


def _load_engineered_pair(name: str, must_have_target: bool = True):
    tr_path = f"{ENGINEERED_DIR}/train_{name}_v{ENGINEERED_VERSION}.csv"
    te_path = f"{ENGINEERED_DIR}/test_{name}_v{ENGINEERED_VERSION}.csv"
    if not os.path.exists(tr_path): raise FileNotFoundError(f"Missing engineered TRAIN file: {tr_path}")
    if not os.path.exists(te_path): raise FileNotFoundError(f"Missing engineered TEST file: {te_path}")
    tr_df = pd.read_csv(tr_path); te_df = pd.read_csv(te_path)
    if ID_COL in tr_df.columns:
        try: tr_df[ID_COL] = tr_df[ID_COL].astype(train[ID_COL].dtype)
        except: pass
    if ID_COL in te_df.columns:
        try: te_df[ID_COL] = te_df[ID_COL].astype(test[ID_COL].dtype)
        except: pass
    if must_have_target and TARGET not in tr_df.columns:
        tr_df = tr_df.merge(train[[ID_COL, TARGET]], on=ID_COL, how="left")
    tr_df = tr_df.loc[:, ~tr_df.columns.duplicated()]
    te_df = te_df.loc[:, ~te_df.columns.duplicated()]
    return tr_df, te_df

if USE_PREBUILT_ENGINEERED:
    print("Reading prebuilt engineered feature sets from:", ENGINEERED_DIR)
    train_raw,     test_raw      = _load_engineered_pair("raw_eda",    must_have_target=True)
    train_recon_fe, test_recon_fe= _load_engineered_pair("recon_fe",   must_have_target=True)
    train_auto20,  test_auto20   = _load_engineered_pair("auto20",     must_have_target=True)
    train_auto50,  test_auto50   = _load_engineered_pair("auto50",     must_have_target=True)
    train_auto100, test_auto100  = _load_engineered_pair("auto100",    must_have_target=True)
    train_auto200, test_auto200  = _load_engineered_pair("auto200",    must_have_target=True)

    print("Engineered feature sets loaded:")
    for name, df in [("raw_eda", train_raw), ("recon_fe", train_recon_fe),
                     ("auto20", train_auto20), ("auto50", train_auto50),
                     ("auto100", train_auto100), ("auto200", train_auto200)]:
        print(f" - {name:10s} -> shape {df.shape}")
else:
    raise RuntimeError("USE_PREBUILT_ENGINEERED is False. Set it True to read prebuilt datasets.")


def _load_engineered_pair(name: str, must_have_target: bool = True):
    tr_path = f"{ENGINEERED_DIR}/train_{name}_v{ENGINEERED_VERSION}.csv"
    te_path = f"{ENGINEERED_DIR}/test_{name}_v{ENGINEERED_VERSION}.csv"
    if not os.path.exists(tr_path): raise FileNotFoundError(f"Missing engineered TRAIN file: {tr_path}")
    if not os.path.exists(te_path): raise FileNotFoundError(f"Missing engineered TEST file: {te_path}")
    tr_df = pd.read_csv(tr_path); te_df = pd.read_csv(te_path)
    if ID_COL in tr_df.columns:
        try: tr_df[ID_COL] = tr_df[ID_COL].astype(train[ID_COL].dtype)
        except: pass
    if ID_COL in te_df.columns:
        try: te_df[ID_COL] = te_df[ID_COL].astype(test[ID_COL].dtype)
        except: pass
    if must_have_target and TARGET not in tr_df.columns:
        tr_df = tr_df.merge(train[[ID_COL, TARGET]], on=ID_COL, how="left")
    tr_df = tr_df.loc[:, ~tr_df.columns.duplicated()]
    te_df = te_df.loc[:, ~te_df.columns.duplicated()]
    return tr_df, te_df

if USE_PREBUILT_ENGINEERED:
    print("Reading prebuilt engineered feature sets from:", ENGINEERED_DIR)
    train_raw,     test_raw      = _load_engineered_pair("raw_eda",    must_have_target=True)
    train_recon_fe, test_recon_fe= _load_engineered_pair("recon_fe",   must_have_target=True)
    train_auto20,  test_auto20   = _load_engineered_pair("auto20",     must_have_target=True)
    train_auto50,  test_auto50   = _load_engineered_pair("auto50",     must_have_target=True)
    train_auto100, test_auto100  = _load_engineered_pair("auto100",    must_have_target=True)
    train_auto200, test_auto200  = _load_engineered_pair("auto200",    must_have_target=True)

    print("Engineered feature sets loaded:")
    for name, df in [("raw_eda", train_raw), ("recon_fe", train_recon_fe),
                     ("auto20", train_auto20), ("auto50", train_auto50),
                     ("auto100", train_auto100), ("auto200", train_auto200)]:
        print(f" - {name:10s} -> shape {df.shape}")
else:
    raise RuntimeError("USE_PREBUILT_ENGINEERED is False. Set it True to read prebuilt datasets.")


# # Ensure reconstructed frames exist (reuse prebuilt CSVs if configured earlier)
# if 'train_recon' not in globals() or 'test_recon' not in globals():
#     if USE_PREBUILT_RECON:
#         train_recon = pd.read_csv(RECON_TRAIN_PATH)
#         test_recon  = pd.read_csv(RECON_TEST_PATH)
#         if TARGET in train_recon.columns and TARGET in train.columns:
#             try: train_recon[TARGET] = train_recon[TARGET].astype(train[TARGET].dtype)
#             except: pass
#         for df, ref in [(train_recon, train), (test_recon, test)]:
#             if ID_COL in df.columns and ID_COL in ref.columns:
#                 try: df[ID_COL] = df[ID_COL].astype(ref[ID_COL].dtype)
#                 except: pass
#         print("Loaded reconstructed CSVs:",
#               f"train_recon {train_recon.shape} | test_recon {test_recon.shape}")
#     else:
#         train_recon, test_recon = train.copy(), test.copy()
#         print("Using original train/test as fallback for reconstruction-dependent FE.")

# # B. Raw-EDA features on competition columns (ENGINEERED)
# train_raw, test_raw = fe_raw_basic(train, test)

# # C. Recon-aware features (ENGINEERED on reconstructed data)
# train_recon_fe, test_recon_fe = fe_recon_basic(train_recon, test_recon)

# # D. Auto-generated sets on reconstructed space (ENGINEERED; MI-selected)
# train_auto20,  test_auto20,  keep20,  mi20  = auto_generate_features(train_recon_fe, test_recon_fe, budget=20)
# train_auto50,  test_auto50,  keep50,  mi50  = auto_generate_features(train_recon_fe, test_recon_fe, budget=50)
# train_auto100, test_auto100, keep100, mi100 = auto_generate_features(train_recon_fe, test_recon_fe, budget=100)
# train_auto200, test_auto200, keep200, mi200 = auto_generate_features(train_recon_fe, test_recon_fe, budget=200)

# print("Engineered feature sets ready:")
# for name, df in [
#     ("raw_eda",   train_raw),
#     ("recon_fe",  train_recon_fe),
#     ("auto_20",   train_auto20),
#     ("auto_50",   train_auto50),
#     ("auto_100",  train_auto100),
#     ("auto_200",  train_auto200),
# ]:
#     print(f" - {name:10s} -> shape {df.shape}")


# to_save = {
#     f"{OUTPUT_DIR}/train_raw_eda_v{VERSION}.csv":  train_raw,
#     f"{OUTPUT_DIR}/test_raw_eda_v{VERSION}.csv":   test_raw,
#     f"{OUTPUT_DIR}/train_recon_fe_v{VERSION}.csv": train_recon_fe,
#     f"{OUTPUT_DIR}/test_recon_fe_v{VERSION}.csv":  test_recon_fe,
#     f"{OUTPUT_DIR}/train_auto20_v{VERSION}.csv":   train_auto20,
#     f"{OUTPUT_DIR}/test_auto20_v{VERSION}.csv":    test_auto20,
#     f"{OUTPUT_DIR}/train_auto50_v{VERSION}.csv":   train_auto50,
#     f"{OUTPUT_DIR}/test_auto50_v{VERSION}.csv":    test_auto50,
#     f"{OUTPUT_DIR}/train_auto100_v{VERSION}.csv":  train_auto100,
#     f"{OUTPUT_DIR}/test_auto100_v{VERSION}.csv":   test_auto100,
#     f"{OUTPUT_DIR}/train_auto200_v{VERSION}.csv":  train_auto200,
#     f"{OUTPUT_DIR}/test_auto200_v{VERSION}.csv":   test_auto200,
# }
# for path, df in to_save.items():
#     df.to_csv(path, index=False)
# print("Saved engineered datasets to:", OUTPUT_DIR)


# def _load_engineered_pair(name: str, must_have_target: bool = True):
#     """
#     Loads train_<name>_v{VERSION}.csv and test_<name>_v{VERSION}.csv
#     from ENGINEERED_DIR, aligns dtypes for ID and TARGET, and returns (train_df, test_df).
#     """
#     tr_path = f"{ENGINEERED_DIR}/train_{name}_v{ENGINEERED_VERSION}.csv"
#     te_path = f"{ENGINEERED_DIR}/test_{name}_v{ENGINEERED_VERSION}.csv"

#     if not os.path.exists(tr_path):
#         raise FileNotFoundError(f"Missing engineered TRAIN file: {tr_path}")
#     if not os.path.exists(te_path):
#         raise FileNotFoundError(f"Missing engineered TEST file: {te_path}")

#     tr_df = pd.read_csv(tr_path)
#     te_df = pd.read_csv(te_path)

#     # Align ID dtype
#     if ID_COL in tr_df.columns:
#         try: tr_df[ID_COL] = tr_df[ID_COL].astype(train[ID_COL].dtype)
#         except: pass
#     if ID_COL in te_df.columns:
#         try: te_df[ID_COL] = te_df[ID_COL].astype(test[ID_COL].dtype)
#         except: pass

#     # Ensure target presence for train if required
#     if must_have_target and TARGET not in tr_df.columns:
#         # If the saved engineered file didn't include TARGET, merge it from original train
#         tr_df = tr_df.merge(train[[ID_COL, TARGET]], on=ID_COL, how="left")

#     # Safety: ensure no duplicate columns sneaked in
#     tr_df = tr_df.loc[:, ~tr_df.columns.duplicated()]
#     te_df = te_df.loc[:, ~te_df.columns.duplicated()]

#     return tr_df, te_df


# # Only read engineered sets we need to evaluate now
# if USE_PREBUILT_ENGINEERED:
#     print("Reading prebuilt engineered feature sets from:", ENGINEERED_DIR)

#     # B. Raw-EDA features on competition columns (ENGINEERED)
#     train_raw,    test_raw    = _load_engineered_pair("raw_eda",    must_have_target=True)

#     # C. Recon-aware features (ENGINEERED)
#     train_recon_fe, test_recon_fe = _load_engineered_pair("recon_fe", must_have_target=True)

#     # D. Auto-generated sets (ENGINEERED; MI-selected)
#     train_auto20,  test_auto20  = _load_engineered_pair("auto20",   must_have_target=True)
#     train_auto50,  test_auto50  = _load_engineered_pair("auto50",   must_have_target=True)
#     train_auto100, test_auto100 = _load_engineered_pair("auto100",  must_have_target=True)
#     train_auto200, test_auto200 = _load_engineered_pair("auto200",  must_have_target=True)

#     print("Engineered feature sets loaded:")
#     for name, df in [
#         ("raw_eda",   train_raw),
#         ("recon_fe",  train_recon_fe),
#         ("auto20",    train_auto20),
#         ("auto50",    train_auto50),
#         ("auto100",   train_auto100),
#         ("auto200",   train_auto200),
#     ]:
#         print(f" - {name:10s} -> shape {df.shape}")
# else:
#     raise RuntimeError("USE_PREBUILT_ENGINEERED is False. Set it True to read prebuilt datasets.")


if "fi_tables" not in globals():
    # define _read_fi_csv if needed
    if "_read_fi_csv" not in globals():
        def _read_fi_csv(path: str) -> pd.DataFrame:
            df = pd.read_csv(path)
            if "feature" not in df.columns:
                maybe = [c for c in df.columns if c.lower().startswith("feat")]
                if maybe: df = df.rename(columns={maybe[0]: "feature"})
            if "importance" not in df.columns:
                maybe = [c for c in df.columns if c.lower().startswith("import")]
                if maybe: df = df.rename(columns={maybe[0]: "importance"})
            df = df[["feature","importance"]].copy()
            df["importance"] = df["importance"].fillna(0).astype(float).clip(lower=0)
            df = (df.groupby("feature", as_index=False)["importance"]
                    .sum().sort_values("importance", ascending=False))
            return df
    fi_tables = {k: _read_fi_csv(v) for k, v in FI_PATHS.items()}

# Build the bundle dictionary
feature_sets = {
    "raw_eda":   (train_raw,     test_raw),
    "recon_fe":  (train_recon_fe, test_recon_fe),
    "auto_20":   (train_auto20,  test_auto20),
    "auto_50":   (train_auto50,  test_auto50),
    "auto_100":  (train_auto100, test_auto100),
    "auto_200":  (train_auto200, test_auto200),
}

# Ensure TARGET in train frames
for name, (tr_df, te_df) in list(feature_sets.items()):
    if TARGET not in tr_df.columns:
        tr_df = tr_df.merge(train[[ID_COL, TARGET]], on=ID_COL, how="left")
        feature_sets[name] = (tr_df, te_df)

# Helper maps and functions (redeclare if running this cell standalone)
if "_weights_from_fi" not in globals():
    def _weights_from_fi(fi_df: pd.DataFrame) -> dict:
        if fi_df.empty: return {}
        imp = fi_df["importance"].astype(float).values
        m = imp.max() if imp.size else 0.0
        w = (imp / m) if m > 0 else np.zeros_like(imp)
        w = np.maximum(w, 0.0)
        return dict(zip(fi_df["feature"], w))

if "_apply_feature_weights" not in globals():
    def _apply_feature_weights(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
        if not weights: return df
        out = df.copy()
        for col, w in weights.items():
            if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col] * (float(w) + WEIGHT_EPS)
        return out

if "_subset_by_topk" not in globals():
    def _subset_by_topk(train_df: pd.DataFrame, test_df: pd.DataFrame, fi_df: pd.DataFrame, k: int):
        top_feats = list(fi_df.head(k)["feature"])
        keep_feats = [c for c in top_feats if c in train_df.columns]
        cols = [ID_COL] + ([TARGET] if TARGET in train_df.columns else []) + keep_feats
        tr_k = train_df[cols].copy()
        te_cols = [c for c in cols if c != TARGET]
        te_k = test_df[te_cols].copy()
        return tr_k, te_k, keep_feats

if "_signature_from_cols" not in globals():
    def _signature_from_cols(cols: list) -> str:
        return "|".join(sorted(cols))

# Prepare run bundles (full + weighted + top-K)
run_bundles = []

# Full sets (unweighted + weighted)
for set_name, (tr_df, te_df) in feature_sets.items():
    run_bundles.append((f"{set_name}_full_unweighted", tr_df, te_df))
    fi_df = fi_tables.get(set_name, pd.DataFrame(columns=["feature","importance"]))
    w = _weights_from_fi(fi_df)
    tr_w = _apply_feature_weights(tr_df.drop(columns=[TARGET], errors="ignore"), w)
    if TARGET in tr_df.columns:
        tr_w = tr_w.join(tr_df[[ID_COL, TARGET]].set_index(ID_COL), on=ID_COL)
    te_w = _apply_feature_weights(te_df, w)
    run_bundles.append((f"{set_name}_full_weighted", tr_w, te_w))

# Top-K subsets across sets, dedupe by feature signature
seen_sigs = set()
topk_pairs = []

fi_map = {
    "raw_eda": "raw_eda",
    "recon_fe": "recon_fe",
    "auto_20": "auto_20",
    "auto_50": "auto_50",
    "auto_100": "auto_100",
    "auto_200": "auto_200",
}

for set_name, (tr_df, te_df) in feature_sets.items():
    fi_df = fi_tables[fi_map[set_name]]
    for k in TOPK_LIST:
        tr_k, te_k, keep_feats = _subset_by_topk(tr_df, te_df, fi_df, k)
        if len(keep_feats) == 0:
            continue
        sig = _signature_from_cols(keep_feats)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            topk_pairs.append((f"{set_name}_top{k}_unweighted", tr_k, te_k))

        w = _weights_from_fi(fi_df)
        tr_kw = _apply_feature_weights(tr_k.drop(columns=[TARGET], errors="ignore"), w)
        if TARGET in tr_k.columns:
            tr_kw = tr_kw.join(tr_k[[ID_COL, TARGET]].set_index(ID_COL), on=ID_COL)
        te_kw = _apply_feature_weights(te_k, w)
        sig_w = sig + "_w"
        if sig_w not in seen_sigs:
            seen_sigs.add(sig_w)
            topk_pairs.append((f"{set_name}_top{k}_weighted", tr_kw, te_kw))

run_bundles.extend(topk_pairs)

print(f"Total run bundles prepared: {len(run_bundles)}")
for name, _, _ in run_bundles[:8]:
    print("  ->", name)


LGB_SEEDS = [
    dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63,
         n_estimators=4000, subsample=0.90, colsample_bytree=0.80,
         min_child_samples=25, reg_alpha=0.0, reg_lambda=0.0,
         device="gpu", random_state=SEED, verbosity=-1),
    dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=64,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
         n_estimators=5000, verbosity=-1, random_state=SEED, device="gpu"),
    dict(objective="binary", metric="auc",
         learning_rate=0.0592, num_leaves=26, max_depth=4,
         lambda_l1=1.34, lambda_l2=3.1e-07, min_child_samples=95,
         n_estimators=5000, colsample_bytree=0.565, subsample=0.975,
         random_state=133, verbosity=-1, device="gpu")
]

def _clip(x, lo, hi): return max(lo, min(hi, x))
def _jitter_float(x, rel=0.3, lo=None, hi=None):
    mu = float(x)
    span = abs(mu) * rel + 1e-12
    val = mu + (random.random()*2 - 1) * span
    if lo is not None: val = max(lo, val)
    if hi is not None: val = min(hi, val)
    return float(val)

def _jitter_int(x, rel=0.3, lo=None, hi=None):
    v = int(round(_jitter_float(x, rel, lo, hi)))
    if lo is not None: v = max(int(lo), v)
    if hi is not None: v = min(int(hi), v)
    return int(v)

def sample_lgb_around(seed):
    p = dict(seed)  # shallow copy
    p["learning_rate"]     = _jitter_float(p.get("learning_rate", 0.02), 0.4, 0.002, 0.2)
    p["num_leaves"]        = _jitter_int(p.get("num_leaves", 63), 0.6, 7, 255)
    if "max_depth" in p or random.random() < 0.5:
        p["max_depth"]     = _jitter_int(p.get("max_depth", -1), 0.7, -1, 16)
    p["min_child_samples"] = _jitter_int(p.get("min_child_samples", 20), 0.6, 5, 200)
    p["subsample"]         = _clip(_jitter_float(p.get("subsample", p.get("bagging_fraction", 0.8)), 0.35, 0.5, 1.0), 0.5, 1.0)
    p["colsample_bytree"]  = _clip(_jitter_float(p.get("colsample_bytree", p.get("feature_fraction", 0.8)), 0.35, 0.3, 1.0), 0.3, 1.0)
    p["lambda_l1"]         = abs(_jitter_float(p.get("lambda_l1", p.get("reg_alpha", 0.0)), 1.2, 0.0, 20.0))
    p["lambda_l2"]         = abs(_jitter_float(p.get("lambda_l2", p.get("reg_lambda", 0.0)), 1.2, 0.0, 20.0))
    p["n_estimators"]      = _jitter_int(p.get("n_estimators", 4000), 0.4, 2000, 12000)
    p["objective"]         = "binary"
    p["metric"]            = "auc"
    p["device"]            = "gpu"
    p["verbosity"]         = -1
    # cleanup aliases
    p.pop("feature_fraction", None)
    p.pop("bagging_fraction", None)
    p.pop("bagging_freq", None)
    p.pop("seed", None)
    return p

def _seed_list_for(model_name):
    if model_name == "lightgbm_gpu":
        return LGB_SEEDS, sample_lgb_around
    # elif model_name == "xgboost_gpu": return XGB_SEEDS, sample_xgb_around
    # elif model_name == "catboost_gpu": return CAT_SEEDS, sample_cat_around
    return [], None


# Robust CSV-append logger for HPO trials
def _safe_write_hpo(row: dict, path: str = HPO_RESULTS_CSV):
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        try:
            prev = pd.read_csv(path)
            out = pd.concat([prev, df_row], ignore_index=True)
        except Exception:
            # fallback if file is malformed
            out = df_row
    else:
        out = df_row
    out.to_csv(path, index=False)


def fit_predict_lightgbm(est, Xtr, ytr, Xva, yva, Xte):
    est.set_params(bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED, verbosity=-1)
    est.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        callbacks=[early_stopping(200, verbose=False), log_evaluation(0)]
    )
    va = est.predict_proba(Xva)[:, 1]
    te = est.predict_proba(Xte)[:, 1] if Xte is not None else None
    return va, te, getattr(est, "feature_importances_", None)

def _evaluate_params_cv_with_save(model_name,
                                  params,
                                  set_name,
                                  train_df,
                                  test_df,
                                  n_splits=5):
    feature_cols, preproc = preprocessor_for_model(model_name, train_df)

    X = train_df[feature_cols].copy()
    if X.shape[1] == 0:
        print(f"[{set_name} | {model_name}] 0 usable features; skipping.")
        return None, None, None, None

    y = train_df[TARGET].astype(int).values
    X_te = test_df[feature_cols].copy() if test_df is not None else None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_df), dtype=np.float32)
    test_preds, fold_aucs = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        if time_up():
            print(f"[{set_name} | {model_name}] Time limit reached mid-CV.")
            break

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        fp = preproc.fit(X_tr)
        Xtr_t = fp.transform(X_tr)
        Xva_t = fp.transform(X_va)
        Xte_t = fp.transform(X_te) if X_te is not None else None

        if model_name == "lightgbm_gpu":
            est = lgb.LGBMClassifier(**params)
            va_pred, te_pred, _ = fit_predict_lightgbm(est, Xtr_t, y_tr, Xva_t, y_va, Xte_t)
        else:
            est = FACTORIES.get(model_name, make_lightgbm_gpu)()
            est_use = est if hasattr(est, "predict_proba") else CalibratedClassifierCV(est, method="isotonic", cv=3)
            est_use.fit(Xtr_t, y_tr)
            va_pred = est_use.predict_proba(Xva_t)[:, 1]
            te_pred = est_use.predict_proba(Xte_t)[:, 1] if Xte_t is not None else None

        oof[va_idx] = va_pred
        if te_pred is not None: test_preds.append(te_pred)
        fold_aucs.append(roc_auc_score(y_va, va_pred))

        del Xtr_t, Xva_t, Xte_t
        gc.collect()

    if not fold_aucs:
        return None, None, None, None

    cv_mean, cv_std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    test_pred = np.mean(test_preds, axis=0) if test_preds else None

    oof_path, sub_path = save_oof_and_sub(set_name, model_name, oof, test_pred,
                                          train_df[ID_COL].values,
                                          (test_df[ID_COL].values if (test_df is not None and test_pred is not None) else None))
    return cv_mean, cv_std, oof_path, sub_path


HPO_TRIAL_PREFIX = None  # global tag in filenames/rows

def run_hpo_for_model(model_name,
                      base_set_name,
                      train_df,
                      test_df,
                      max_trials_per_seed=5,
                      cv_splits=5):
    seeds, sampler = _seed_list_for(model_name)
    if not seeds or sampler is None:
        print(f"[HPO] {model_name}: no HPO seeds configured; skipping.")
        return None

    best = {"auc": -1.0, "std": None, "params": None, "seed_idx": None, "trial_idx": None}

    for s_idx, seed in enumerate(seeds):
        if time_up(): break

        trial_set = f"hpo_{HPO_TRIAL_PREFIX}_{model_name}_seed{s_idx}"
        auc, std, oof_path, sub_path = _evaluate_params_cv_with_save(
            model_name=model_name,
            params=seed,
            set_name=trial_set,
            train_df=train_df,
            test_df=test_df,
            n_splits=cv_splits
        )
        _safe_write_hpo({
            "version": VERSION, "feature_set": HPO_TRIAL_PREFIX, "model": model_name,
            "kind": "seed", "seed_index": s_idx, "trial_index": -1,
            "cv_auc_mean": (-1 if auc is None else auc),
            "cv_auc_std":  (std if std is not None else -1),
            "oof_path": oof_path or "", "sub_path": sub_path or "",
            "params_json": json.dumps(seed)
        })
        if (auc is not None) and (auc > best["auc"]):
            best = {"auc": auc, "std": std, "params": dict(seed),
                    "seed_idx": s_idx, "trial_idx": "seed"}

        for t in range(max_trials_per_seed):
            if time_up(): break
            cand = sampler(seed)
            trial_set = f"hpo_{HPO_TRIAL_PREFIX}_{model_name}_seed{s_idx}_trial{t}"
            auc, std, oof_path, sub_path = _evaluate_params_cv_with_save(
                model_name=model_name,
                params=cand,
                set_name=trial_set,
                train_df=train_df,
                test_df=test_df,
                n_splits=cv_splits
            )
            _safe_write_hpo({
                "version": VERSION, "feature_set": HPO_TRIAL_PREFIX, "model": model_name,
                "kind": "sample", "seed_index": s_idx, "trial_index": t,
                "cv_auc_mean": (-1 if auc is None else auc),
                "cv_auc_std":  (std if std is not None else -1),
                "oof_path": oof_path or "", "sub_path": sub_path or "",
                "params_json": json.dumps(cand)
            })
            if (auc is not None) and (auc > best["auc"]):
                best = {"auc": auc, "std": std, "params": dict(cand),
                        "seed_idx": s_idx, "trial_idx": t}
                print(f"[HPO|{model_name}|{HPO_TRIAL_PREFIX}] **new best** AUC {auc:.6f} (seed {s_idx}, trial {t})")

    if best["params"] is not None:
        print(f"[HPO|{model_name}|{HPO_TRIAL_PREFIX}] BEST AUC {best['auc']:.6f} (seed {best['seed_idx']}, trial {best['trial_idx']})")
    else:
        print(f"[HPO|{model_name}|{HPO_TRIAL_PREFIX}] No successful trials.")
    return best


results = []

if time_up():
    print("\n=== Global time limit already reached; stopping. ===")
else:
    print("\n=== HPO across engineered feature bundles ===")
    for set_name, tr_df, te_df in run_bundles:
        # guard against degenerate sets
        feat_cols = [c for c in tr_df.columns if c not in [TARGET, ID_COL]]
        if len(feat_cols) == 0:
            print(f"[skip] {set_name}: no usable features.")
            continue

        HPO_TRIAL_PREFIX = set_name
        for model_name in MODELS_TO_RUN:
            if time_up():
                print("\n=== Global time limit reached; stopping. ===")
                break
            print(f"\n[RUN HPO] set={set_name} | model={model_name} | d={len(feat_cols)}")
            best = run_hpo_for_model(
                model_name=model_name,
                base_set_name=set_name,
                train_df=tr_df,
                test_df=te_df,
                max_trials_per_seed=5,      # adjust if needed
                cv_splits=N_SPLITS
            )
            if best and best["params"] is not None:
                write_results_row({
                    "version": VERSION,
                    "feature_set": set_name,
                    "model": model_name,
                    "cv_auc_mean": best["auc"],
                    "cv_auc_std": best["std"] if best["std"] is not None else -1,
                    "folds_completed": N_SPLITS,
                    "train_time_sec": None,
                    "train_time_hms": None,
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "oof_path": "", "sub_path": "", "importance_path": ""
                })
    HPO_TRIAL_PREFIX = None


if os.path.exists(HPO_RESULTS_CSV):
    hpo_df = pd.read_csv(HPO_RESULTS_CSV)
    cols_show = ["model","feature_set","kind","seed_index","trial_index",
                 "cv_auc_mean","cv_auc_std","oof_path","sub_path","params_json"]
    cols_show = [c for c in cols_show if c in hpo_df.columns]
    print("\nTop trials by AUC:")
    display(hpo_df.sort_values("cv_auc_mean", ascending=False)[cols_show].head(40))

    print("\nBest per (model, feature_set):")
    best_per = (hpo_df.sort_values("cv_auc_mean", ascending=False)
                      .groupby(["model","feature_set"], as_index=False).first())
    cols2 = [c for c in ["model","feature_set","cv_auc_mean","cv_auc_std","oof_path","sub_path"] if c in best_per.columns]
    display(best_per[cols2].sort_values("cv_auc_mean", ascending=False))
else:
    print("No HPO trials CSV found yet.")




