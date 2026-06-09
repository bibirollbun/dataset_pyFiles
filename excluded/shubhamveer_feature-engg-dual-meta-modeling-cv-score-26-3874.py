# Models
!pip install lightgbm xgboost catboost shap



# Config & Imports
import gc
import numpy as np
import pandas as pd
import itertools
import os
import psutil
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

import xgboost as xgb
from catboost import CatBoostRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import shap

# Config constants (these will be used by the FE pipeline)
DATA_DIR = '/kaggle/input/playground-series-s5e9/'
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

TARGET_COL = "BeatsPerMinute"
ID_COL = "id"

RANDOM_STATE = 2005
N_SPLITS = 5
TARGET_LOG = False  # if you train on log(target) set True to invert preds
SEED = RANDOM_STATE

# Memory-safety controls (tweakable)
MEMORY_SAFETY = False         # True = conservative defaults; False = allow heavy expansion (risky)
MAX_BASE_NUMERIC = 10 
MAX_INTERACTION_PAIRS = 100 
USE_POLY_SKLEARN = True 
POLY_DEGREE = 3
POLY_INTERACTION_ONLY = False

DO_ROW_STATS = False
DO_LOG_SQRT = False

# NEW: Trigonometric transforms
DO_TRIG=False

DO_RANKS = False
DO_ROW_ZSCORES = False
DO_PAIRWISE_ARITH = True
PAIRWISE_OPS = ["plus", "minus", "mul", "div"]
Drop_high_unique = True
# Toggle for clipping
DO_CLIP = False   # set to False to skip clipping extremes

CLIP_LOWER_Q = 0.001
CLIP_UPPER_Q = 0.999

USE_GPU = True  # ğŸ”€ change this to False if you want CPU


def augment_with_multi_noise(df, noise_levels=[0.01, 0.05, 0.1], numeric_only=True, random_state=None):
    """
    Create training data with multiple noisy versions of the dataset.
    Handles NaNs safely.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    augmented_df = [df.copy()]  # keep original
    
    for nl in noise_levels:
        noisy_df = df.copy()
        cols = noisy_df.select_dtypes(include=np.number).columns if numeric_only else noisy_df.columns
        
        for col in cols:
            std = noisy_df[col].std(skipna=True)  # safe std
            if pd.isna(std) or std == 0:  
                continue  # skip constant or invalid cols
            
            noise = np.random.normal(0, nl * std, noisy_df[col].shape[0])
            
            # Add noise only where values are not NaN
            noisy_df[col] = noisy_df[col].where(noisy_df[col].isna(), noisy_df[col] + noise)
        
        noisy_df["noise_level"] = nl
        augmented_df.append(noisy_df)
    
    return pd.concat(augmented_df, ignore_index=True)

# Data loading
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
orig_train = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")
orig_test = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Test.csv")
print(f"Train: {train.shape}, Test: {test.shape}")
print("Adding orignal data")
print(orig_test.columns)
orig_test.drop("BeatsPerMinute", axis=1, inplace=True)
train = pd.concat([orig_train, train], axis=0)
print(f"Train: {train.shape}, Test: {test.shape}")
print(display(train.info()))
print("Adding Noise to the data only (Training)")
# train = augment_with_multi_noise(train, noise_levels=[0.05], numeric_only=True, random_state=2004)
# print(f"Train: {train.shape}, Test: {test.shape}")
# print(display(train.info()))
# train['noise_level'] = train['noise_level'].fillna(0)


# Helpers / diagnostics
import gc
def memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def print_header(title):
    print("\n" + "="*40)
    print(title)
    print("="*40)

def reduce_mem_usage(df: pd.DataFrame, silent=False):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(col_type) and col != TARGET_COL:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    end_mem = df.memory_usage().sum() / 1024**2
    if not silent:
        print(f"[mem] reduced {start_mem:.2f} MB -> {end_mem:.2f} MB")
    return df

def safe_div(a: pd.Series, b: pd.Series, eps=1e-6):
    b_safe = b.copy().astype(np.float32)
    near_zero = b_safe.abs() < eps
    b_safe[near_zero] = eps * np.sign(b_safe[near_zero].replace(0, 1))
    res = a.astype(np.float32) / b_safe
    res = res.replace([np.inf, -np.inf], np.nan)
    return res

def check_numeric_issues(df: pd.DataFrame, name="df", top=10):
    print_header(f"{name} summary")
    print("shape:", df.shape)
    num_df = df.select_dtypes(include=[np.number])
    print("numeric cols:", num_df.shape[1])
    print("NaN per col (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(top))
    print("Inf count:", np.isinf(df.values).sum())
    if num_df.shape[1] > 0:
        max_abs = num_df.abs().max().sort_values(ascending=False).head(top)
        print("Top 10 columns by abs(max):")
        print(max_abs)
    print("[mem] process usage MB:", memory_mb())

# Load raw and separate target
y = train[TARGET_COL].copy()
X = train.drop(columns=[TARGET_COL], errors='ignore').copy()
X_test = test.copy()

# drop common id cols
for idc in [ID_COL, "ID", "Id", "index"]:
    if idc in X.columns: X.drop(columns=[idc], inplace=True)
    if idc in X_test.columns: X_test.drop(columns=[idc], inplace=True)

print_header("Initial shapes")
print("Initial shapes:", X.shape, X_test.shape)
print("[mem] after load MB:", memory_mb())

# Pick numeric base columns
def pick_numeric_columns(X, max_cols=MAX_BASE_NUMERIC):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if "id" in num_cols:
        num_cols.remove("id")
    if len(num_cols) <= max_cols:
        return num_cols
    variances = X[num_cols].var().sort_values(ascending=False)
    selected = variances.index[:max_cols].tolist()
    return selected

base_numeric = pick_numeric_columns(X, MAX_BASE_NUMERIC)
print("Selected base numeric columns (count):", len(base_numeric))
if len(base_numeric) < 1:
    raise ValueError("No numeric columns found to engineer on.")

# Keep minimal copies / downcast early
X_work      = X[base_numeric].astype(np.float32).copy()
X_test_work = X_test.reindex(columns=base_numeric).astype(np.float32).copy().fillna(0.0)

reduce_mem_usage(X_work, silent=True)
reduce_mem_usage(X_test_work, silent=True)

print_header("Step: row-level statistics")
if DO_ROW_STATS:
    for df,label in [(X_work, "train"), (X_test_work, "test")]:
        df["row_sum"]  = df[base_numeric].sum(axis=1).astype(np.float32)
        df["row_mean"] = df[base_numeric].mean(axis=1).astype(np.float32)
        df["row_std"]  = df[base_numeric].std(axis=1).fillna(0).astype(np.float32)
    print("Added row_sum, row_mean, row_std on base numeric columns.")

print_header("Step: log/sqrt transforms")
if DO_LOG_SQRT:
    for c in base_numeric:
        if (X_work[c] > 0).all() and (X_test_work[c] > 0).all():
            X_work[f"log1p_{c}"] = np.log1p(X_work[c]).astype(np.float32)
            X_test_work[f"log1p_{c}"] = np.log1p(X_test_work[c]).astype(np.float32)
        X_work[f"sqrt_{c}"] = (np.sign(X_work[c]) * np.sqrt(np.abs(X_work[c]))).astype(np.float32)
        X_test_work[f"sqrt_{c}"] = (np.sign(X_test_work[c]) * np.sqrt(np.abs(X_test_work[c]))).astype(np.float32)
    print("Added log1p_ (where safe) and sqrt_ transforms for base numeric cols.")

print_header("Step: row-relative ratios")
for df,label in [(X_work, "train"), (X_test_work, "test")]:
    row_sum = df[base_numeric].sum(axis=1).replace(0, 1).astype(np.float32)
    for c in base_numeric:
        df[f"{c}_row_ratio"] = safe_div(df[c], row_sum)

print("Added _row_ratio features.")

print_header("Step: z-score per row (optional)")
if DO_ROW_ZSCORES:
    for df in [X_work, X_test_work]:
        row_mean = df[base_numeric].mean(axis=1)
        row_std  = df[base_numeric].std(axis=1).replace(0, 1)
        for c in base_numeric:
            df[f"{c}_zscore"] = ((df[c] - row_mean) / row_std).astype(np.float32)
    print("Added per-row z-scores (heavy).")

print_header("Step: Polynomial / interaction features")
if USE_POLY_SKLEARN:
    numeric_cols_for_poly = X_work.select_dtypes(include=[np.number]).columns.tolist()
    print("Using sklearn PolynomialFeatures on", len(numeric_cols_for_poly), "columns.")
    poly = PolynomialFeatures(degree=POLY_DEGREE, interaction_only=POLY_INTERACTION_ONLY, include_bias=False)
    X_poly = poly.fit_transform(X_work[numeric_cols_for_poly].astype(np.float32))
    X_test_poly = poly.transform(X_test_work[numeric_cols_for_poly].astype(np.float32))
    try:
        poly_names = poly.get_feature_names_out(numeric_cols_for_poly)
    except Exception:
        poly_names = [f"poly_{i}" for i in range(X_poly.shape[1])]
    X_poly_df = pd.DataFrame(X_poly, columns=poly_names, index=X_work.index).astype(np.float32)
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_names, index=X_test_work.index).astype(np.float32)
    interaction_cols = [c for c in poly_names if ('*' in c) or (' ' in c)]
    X_poly_inter = X_poly_df[interaction_cols]
    X_test_poly_inter = X_test_poly_df[interaction_cols]
    X_work = pd.concat([X_work, X_poly_inter], axis=1)
    X_test_work = pd.concat([X_test_work, X_test_poly_inter], axis=1)
    del X_poly, X_test_poly, X_poly_df, X_test_poly_df, X_poly_inter, X_test_poly_inter
    gc.collect()
    print("Finished sklearn PolynomialFeatures (interaction subset kept).")
else:
    numeric_cols_for_int = X_work.select_dtypes(include=[np.number]).columns.tolist()
    pair_candidates = list(itertools.combinations(numeric_cols_for_int, 2))
    pair_candidates = pair_candidates[:MAX_INTERACTION_PAIRS]
    created = 0
    for a,b in pair_candidates:
        X_work[f"{a}*{b}"] = (X_work[a] * X_work[b]).astype(np.float32)
        X_test_work[f"{a}*{b}"] = (X_test_work[a] * X_test_work[b]).astype(np.float32)
        created += 1
    print(f"Created {created} manual product interaction features (capped).")

# NEW: Trigonometric transforms
if DO_TRIG:
    print_header("Step: Trigonometric transforms")
    for c in base_numeric:
        X_work[f"sin_{c}"] = np.sin(X_work[c]).astype(np.float32)
        X_work[f"cos_{c}"] = np.cos(X_work[c]).astype(np.float32)
        X_work[f"tan_{c}"] = np.tan(X_work[c]).replace([np.inf, -np.inf], np.nan).astype(np.float32)

        X_test_work[f"sin_{c}"] = np.sin(X_test_work[c]).astype(np.float32)
        X_test_work[f"cos_{c}"] = np.cos(X_test_work[c]).astype(np.float32)
        X_test_work[f"tan_{c}"] = np.tan(X_test_work[c]).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # pairwise trig interactions
    trig_cols = []
    for c in base_numeric:
        trig_cols.extend([f"sin_{c}", f"cos_{c}", f"tan_{c}"])

    pair_candidates = list(itertools.combinations(trig_cols, 2))
    pair_candidates = pair_candidates[:MAX_INTERACTION_PAIRS]
    for a, b in pair_candidates:
        X_work[f"{a}_mul_{b}"] = (X_work[a] * X_work[b]).astype(np.float32)
        X_test_work[f"{a}_mul_{b}"] = (X_test_work[a] * X_test_work[b]).astype(np.float32)

    print("Added sin, cos, tan and their pairwise interaction features.")

print_header("Step: Pairwise arithmetic features (plus, minus, mul, div)")
if DO_PAIRWISE_ARITH:
    pairs = list(itertools.combinations(base_numeric, 2))
    pairs = pairs[:(MAX_INTERACTION_PAIRS // max(1, len(PAIRWISE_OPS)))]
    cnt = 0
    for a,b in pairs:
        if "plus" in PAIRWISE_OPS:
            X_work[f"{a}_plus_{b}"] = (X_work[a] + X_work[b]).astype(np.float32)
            X_test_work[f"{a}_plus_{b}"] = (X_test_work[a] + X_test_work[b]).astype(np.float32)
        if "minus" in PAIRWISE_OPS:
            X_work[f"{a}_minus_{b}"] = (X_work[a] - X_work[b]).astype(np.float32)
            X_test_work[f"{a}_minus_{b}"] = (X_test_work[a] - X_test_work[b]).astype(np.float32)
        if "mul" in PAIRWISE_OPS:
            X_work[f"{a}_mul_{b}"] = (X_work[a] * X_work[b]).astype(np.float32)
            X_test_work[f"{a}_mul_{b}"] = (X_test_work[a] * X_test_work[b]).astype(np.float32)
        if "div" in PAIRWISE_OPS:
            X_work[f"{a}_div_{b}"] = safe_div(X_work[a], X_work[b]).astype(np.float32)
            X_test_work[f"{a}_div_{b}"] = safe_div(X_test_work[a], X_test_work[b]).astype(np.float32)
        cnt += 1
    print(f"Pairwise arithmetic features created for {cnt} base pairs (ops: {PAIRWISE_OPS}).")

# Align columns between train & test (fill with zeros where missing)
X_work, X_test_work = X_work.align(X_test_work, join="outer", axis=1, fill_value=0.0)
X_test_work = X_test_work.reindex(columns=X_work.columns, fill_value=0.0)

print_header("Diagnostics after feature engineering")
print("Train shape:", X_work.shape)
print("Test  shape:", X_test_work.shape)
print("[mem] current MB:", memory_mb())
print("Top NaN counts (train):")
print(X_work.isna().sum().sort_values(ascending=False).head(10))

# Clip extremes (quantile-based, optional)
if DO_CLIP:
    print_header("Clipping extremes by quantiles")
    numeric_train = X_work.select_dtypes(include=[np.number])
    low = numeric_train.quantile(CLIP_LOWER_Q)
    high = numeric_train.quantile(CLIP_UPPER_Q)
    X_clip = numeric_train.clip(lower=low, upper=high, axis=1)
    X_test_clip = X_test_work.select_dtypes(include=[np.number]).clip(lower=low, upper=high, axis=1)
    X_work.update(X_clip)
    X_test_work.update(X_test_clip)
    del X_clip, X_test_clip
    gc.collect()
    print("Clipping done.")

    check_numeric_issues(X_work, "Train features after clipping", top=10)
    check_numeric_issues(X_test_work, "Test features after clipping", top=10)
else:
    print_header("Skipping clipping step (DO_CLIP=False)")



check_numeric_issues(X_work, "Train features after clipping", top=10)
check_numeric_issues(X_test_work, "Test features after clipping", top=10)

# Drop very high unique columns (likely IDs/leakage)
def drop_high_unique(df_train, df_test, frac_threshold=0.95):
    n = len(df_train)
    uniq_frac = df_train.nunique() / n
    drop_cols = uniq_frac[uniq_frac > frac_threshold].index.tolist()
    if drop_cols:
        print("Dropping high-unique cols (likely ids/leakage):", drop_cols[:30])
        df_train = df_train.drop(columns=drop_cols, errors='ignore')
        df_test = df_test.drop(columns=drop_cols, errors='ignore')
    return df_train, df_test

X_work, X_test_work = drop_high_unique(X_work, X_test_work, frac_threshold=0.95)
X_test_work = X_test_work.reindex(columns=X_work.columns, fill_value=0.0)

print_header("Final feature set ready for modeling")
print("Final train shape:", X_work.shape)
print("Final test  shape:", X_test_work.shape)
print("[mem] before CV MB:", memory_mb())


import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb

def hybrid_feature_selection(X, y, task="regression", top_n=300,
                             random_state=42, use_gpu=False, corr_abs=True):
    """
    Hybrid feature selection using Variance filter + LightGBM + (MI or Correlation).
    
    If use_gpu=True, replaces slow Mutual Information with fast Correlation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series / np.array
        Target
    task : str
        "regression" or "classification"
    top_n : int
        Number of features to keep
    random_state : int
        Random seed
    use_gpu : bool
        If True -> GPU LightGBM + Correlation
        If False -> CPU LightGBM + Mutual Information
    corr_abs : bool
        Use absolute correlation values (default True)
    
    Returns
    -------
    X_selected : pd.DataFrame
        Reduced feature set
    selected_features : list
        Names of selected features
    """

    print("\n" + "="*50)
    print("Step 1: Variance Threshold Filter")
    print("="*50)

    vt = VarianceThreshold(threshold=0.0)
    X_vt = vt.fit_transform(X)
    kept_vt = X.columns[vt.get_support()]
    X_vt = pd.DataFrame(X_vt, columns=kept_vt)

    print(f"Removed {X.shape[1] - X_vt.shape[1]} constant features.")
    print(f"Remaining after variance filter: {X_vt.shape[1]} features")

    # Step 2: LightGBM importance
    print("\n" + "="*50)
    print("Step 2: LightGBM Feature Importance")
    print("="*50)

    params = {
        "objective": "regression" if task == "regression" else "binary",
        "learning_rate": 0.05,
        "num_leaves": 200,
        "n_estimators": 1000,     # slightly reduced for speed
        "verbosity": -1,
        "random_state": random_state
    }
    if use_gpu:
        params["device"] = "gpu"

    dtrain = lgb.Dataset(X_vt, label=y)
    lgb_model = lgb.train(params, dtrain)
    imp_lgb = lgb_model.feature_importance()
    rank_lgb = pd.Series(imp_lgb, index=kept_vt).rank(ascending=True)
    print(rank_lgb)

    # Step 3: Correlation (GPU-friendly) or MI
    print("\n" + "="*50)
    print("Step 3: Secondary Ranking (MI or Correlation)")
    print("="*50)

    if use_gpu:
        # Correlation as proxy for MI (much faster)
        corrs = X_vt.corrwith(pd.Series(y)).fillna(0.0)
        if corr_abs:
            corrs = corrs.abs()
        rank_secondary = corrs.rank(ascending=False)
        print("Used Correlation (GPU-friendly) instead of MI")
    else:
        # CPU Mutual Information
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        if task == "regression":
            mi = mutual_info_regression(X_vt, y, random_state=random_state, n_neighbors=3, n_jobs=-1)
        else:
            mi = mutual_info_classif(X_vt, y, random_state=random_state, n_jobs=-1)
        rank_secondary = pd.Series(mi, index=kept_vt).rank(ascending=False)
        print("Used Mutual Information (CPU)")

    # Step 4: Combine ranks
    scores = 0.6 * rank_lgb + 0.4 * rank_secondary
    selected_features = scores.sort_values().head(top_n).index.tolist()
    X_selected = X[selected_features]

    print("\n" + "="*50)
    print(f"Original features: {X.shape[1]} â†’ Selected features: {X_selected.shape[1]}")
    print("="*50)

    return X_selected, selected_features



X_work, selected_features = hybrid_feature_selection(
    X_work, y,
    task="regression",
    top_n=200,
    use_gpu=True   # GPU LightGBM + Correlation
)



X_test_work = X_test_work[selected_features]


# CV setup
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# =============================
# Base Models Setup (Cleaned)
# =============================
import gc
import numpy as np
from copy import deepcopy
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb
from catboost import CatBoostRegressor

RANDOM_STATE = 3014
FEATURES = X_work.columns.tolist()

# -----------------------------
# Model toggles
# -----------------------------
USE_LGB = True
USE_XGB = True
USE_CAT = True

USE_LASSO = True
USE_RIDGE = True
USE_LINEAR = True

# -----------------------------
# LightGBM params
# -----------------------------
lgb_params = {
    "max_depth": 6,
    "num_leaves": 60,
    "colsample_bytree": 0.9,
    "subsample": 0.9,
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "reg_alpha": 0.8,
    "reg_lambda": 4.0,
    "metric": "rmse",
    "verbose": 0,
}
if USE_GPU:
    lgb_params.update({"device_type": "gpu", "gpu_use_dp": False})
else:
    lgb_params.update({"device_type": "cpu"})

# -----------------------------
# XGBoost params
# -----------------------------
xgb_params = {
    "max_depth": 6,
    "colsample_bytree": 0.9,
    "subsample": 0.9,
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "gamma": 10.0,
    "max_delta_step": 2,
    "eval_metric": "rmse",
    "enable_categorical": True,
}

if USE_GPU:
    xgb_params.update({"tree_method": "gpu_hist"})
else:
    xgb_params.update({"tree_method": "hist"})

# -----------------------------
# CatBoost params
# -----------------------------
cat_params = {
    "max_depth": 6,
    "colsample_bylevel": 0.9,
    "n_estimators": 2000,
    "learning_rate": 0.08,
    "random_strength": 0.1,
    "loss_function": "RMSE",
    "verbose": 100,
}

if USE_GPU:
    cat_params.update({"task_type": "GPU", "devices": "0"})
    cat_params.pop("colsample_bylevel", None)  # GPU restriction
else:
    cat_params.update({"task_type": "CPU", "colsample_bylevel": 0.884050283064001})

# -----------------------------
# Sklearn model params
# -----------------------------
lasso_params = {"alpha": 0.01, "random_state": RANDOM_STATE}
ridge_params = {
    "max_iter": 969,
    "alpha": 996.2276353844511,
    "tol": 4.41828902390748e-05,
    "random_state": RANDOM_STATE
}
linear_params = {}  # defaults

sklearn_models = {
    "lasso": (USE_LASSO, Lasso(**lasso_params)),
    "ridge": (USE_RIDGE, Ridge(**ridge_params)),
    "linear": (USE_LINEAR, LinearRegression(**linear_params))
}

# -----------------------------
# Storage dicts
# -----------------------------
oof = {}
preds = {}
models = {}

if USE_LGB: oof["lgb"], preds["lgb"] = np.zeros(len(y)), np.zeros(len(X_test_work))
if USE_XGB: oof["xgb"], preds["xgb"] = np.zeros(len(y)), np.zeros(len(X_test_work))
if USE_CAT: oof["cat"], preds["cat"] = np.zeros(len(y)), np.zeros(len(X_test_work))

for name, (flag, _) in sklearn_models.items():
    if flag:
        oof[name] = np.zeros(len(y))
        preds[name] = np.zeros(len(X_test_work))

# =============================
# CV Loop
# =============================
print(f"\n==== Starting {N_SPLITS}-Fold CV ====")

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_work), 1):
    print(f"\n==== Fold {fold} ====")
    X_tr, y_tr = X_work.iloc[trn_idx][FEATURES], y.iloc[trn_idx]
    X_val, y_val = X_work.iloc[val_idx][FEATURES], y.iloc[val_idx]

    # ---- LightGBM ----
    if USE_LGB:
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params=lgb_params,
            train_set=dtrain,
            valid_sets=[dval],
            num_boost_round=2000,
            callbacks=[early_stopping(stopping_rounds=200)]
        )
        oof["lgb"][val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        preds["lgb"] += model.predict(X_test_work[FEATURES], num_iteration=model.best_iteration) / N_SPLITS
        print("LGB RMSE:", mean_squared_error(y_val, oof["lgb"][val_idx], squared=False))
        models.setdefault("lgb", []).append(model)

    # ---- XGBoost ----
    if USE_XGB:
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        oof["xgb"][val_idx] = model.predict(X_val)
        preds["xgb"] += model.predict(X_test_work[FEATURES]) / N_SPLITS
        print("XGB RMSE:", mean_squared_error(y_val, oof["xgb"][val_idx], squared=False))
        models.setdefault("xgb", []).append(model)

    # ---- CatBoost ----
    if USE_CAT:
        model = CatBoostRegressor(**cat_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof["cat"][val_idx] = model.predict(X_val)
        preds["cat"] += model.predict(X_test_work[FEATURES]) / N_SPLITS
        print("CAT RMSE:", mean_squared_error(y_val, oof["cat"][val_idx], squared=False))
        models.setdefault("cat", []).append(model)

    # ---- Sklearn models ----
    for name, (flag, base_model) in sklearn_models.items():
        if not flag:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test_work[FEATURES].copy())

        model = deepcopy(base_model)  # reset per fold
        model.fit(X_tr_s, y_tr)

        oof[name][val_idx] = model.predict(X_val_s)
        preds[name] += model.predict(X_test_s) / N_SPLITS
        print(f"{name.upper()} RMSE:", mean_squared_error(y_val, oof[name][val_idx], squared=False))
        models.setdefault(name, []).append(model)

    gc.collect()

# =============================
# Overall Scores
# =============================
print("\n==== Overall OOF ====")
for name in oof:
    score = mean_squared_error(y, oof[name], squared=False)
    print(f"{name.upper()} OOF RMSE: {score:.5f}")



import numpy as np

def add_noise_to_meta(X, noise_level=0.01, random_state=None):
    """
    Add Gaussian noise to OOF predictions for meta-model regression.
    
    Parameters
    ----------
    X : np.ndarray
        OOF predictions (shape: n_samples, n_models).
    noise_level : float
        Noise fraction relative to feature std.
    random_state : int or None
        Random seed.
    
    Returns
    -------
    X_noisy : np.ndarray
        Noisy training features for meta-model.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X_noisy = X.copy()
    for i in range(X.shape[1]):
        std = X[:, i].std()
        if std == 0 or np.isnan(std):
            continue
        noise = np.random.normal(0, noise_level * std, X.shape[0])
        X_noisy[:, i] += noise
    
    return X_noisy


# ===============================
# Meta-Model Stacking
# ===============================
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# -------------------------------------------------
# Step 1: Prepare meta features
# -------------------------------------------------
USE_BOOSTERS = True   # ğŸ”€ Set False if you only want sklearn-level stacking

if USE_BOOSTERS:
    meta_train = np.vstack([
        oof['lgb'], oof['xgb'], oof['cat'],
        oof['lasso'], oof['ridge'], oof['linear']
    ]).T
    meta_test = np.vstack([
        preds['lgb'], preds['xgb'], preds['cat'],
        preds['lasso'], preds['ridge'], preds['linear']
    ]).T
else:
    meta_train = np.vstack([
        oof['lasso'], oof['ridge'], oof['linear']
    ]).T
    meta_test = np.vstack([
        preds['lasso'], preds['ridge'], preds['linear']
    ]).T

# -------------------------------------------------
# Step 2: Scale meta features
# -------------------------------------------------
scaler = StandardScaler()
meta_train = add_noise_to_meta(meta_train, noise_level=0.25, random_state=2005)
meta_train_scaled = scaler.fit_transform(meta_train)
meta_test_scaled = scaler.transform(meta_test)

# -------------------------------------------------
# Step 3: Define meta-models
# -------------------------------------------------
meta_models = {
    "Ridge": Ridge(alpha=1.0, tol=0.1, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.8, tol=0.1, random_state=RANDOM_STATE),
    "Linear": LinearRegression()
}

# -------------------------------------------------
# Step 4: Cross-validation training & evaluation
# -------------------------------------------------
results = {}

for name, model in meta_models.items():
    print(f"\n### Meta-Model: {name}")
    meta_oof = np.zeros(len(meta_train))   # Out-of-fold preds
    meta_preds = np.zeros(len(meta_test))  # Test preds (avg across folds)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(meta_train_scaled), start=1):
        X_tr, y_tr = meta_train_scaled[trn_idx], y.iloc[trn_idx]
        X_val, y_val = meta_train_scaled[val_idx], y.iloc[val_idx]

        # Fit model
        model.fit(X_tr, y_tr)

        # Predict
        meta_oof[val_idx] = model.predict(X_val)
        meta_preds += model.predict(meta_test_scaled) / N_SPLITS

        # Fold RMSE
        fold_rmse = mean_squared_error(y_val, meta_oof[val_idx], squared=False)
        print(f"  Fold {fold} RMSE: {fold_rmse:.5f}")

    # OOF RMSE
    oof_rmse = mean_squared_error(y, meta_oof, squared=False)
    print(f"  Overall OOF RMSE: {oof_rmse:.5f}")

    # Store results
    results[name] = {
        "oof_rmse": oof_rmse,
        "oof_preds": meta_oof,
        "test_preds": meta_preds,
        "model": model
    }

# -------------------------------------------------
# Step 5: Select best meta-model
# -------------------------------------------------
best_model_name = min(results, key=lambda x: results[x]["oof_rmse"])
print(f"\nBest Meta-Model: {best_model_name} "
      f"(OOF RMSE = {results[best_model_name]['oof_rmse']:.5f})")

final_preds = results[best_model_name]["test_preds"]



import numpy as np
import pandas as pd
import gc, os, sys, joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.optimize import differential_evolution

# =====================================
# Reproducibility
# =====================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =====================================
# Prepare Level-2 meta features (replace with your oofs/preds dicts)
# =====================================
final_train = np.vstack([
    oof['lgb'], oof['xgb'], oof['cat'],
    results['Ridge']['oof_preds'],
    results['Lasso']['oof_preds'],
    results['Linear']['oof_preds']
]).T

final_test = np.vstack([
    preds['lgb'], preds['xgb'], preds['cat'],
    results['Ridge']['test_preds'],
    results['Lasso']['test_preds'],
    results['Linear']['test_preds']
]).T

# Handle NaNs just in case
final_train = np.nan_to_num(final_train, nan=np.nanmean(final_train))
final_test = np.nan_to_num(final_test, nan=np.nanmean(final_test))

# =====================================
# Ridge Meta-Model (CV Stacking)
# =====================================
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

ridge_oof = np.zeros(len(final_train))
ridge_preds = np.zeros(len(final_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(final_train), start=1):
    X_tr, y_tr = final_train[trn_idx], y.iloc[trn_idx]
    X_val, y_val = final_train[val_idx], y.iloc[val_idx]

    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_tr, y_tr)

    ridge_oof[val_idx] = ridge.predict(X_val)
    ridge_preds += ridge.predict(final_test) / N_SPLITS

    fold_rmse = mean_squared_error(y_val, ridge_oof[val_idx], squared=False)
    print(f"Ridge Fold {fold} RMSE: {fold_rmse:.5f}")

ridge_rmse = mean_squared_error(y, ridge_oof, squared=False)
print(f"\n### Ridge Stacking CV RMSE: {ridge_rmse:.5f}")

# =====================================
# Hill Climbing Ensemble
# =====================================
def hill_climb(X, y, max_iter=5000, step=0.05, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n_models = X.shape[1]
    weights = np.ones(n_models) / n_models
    best_rmse = mean_squared_error(y, X @ weights, squared=False)

    for _ in range(max_iter):
        i, j = rng.choice(n_models, 2, replace=False)
        new_weights = weights.copy()
        delta = step * (1 if rng.random() < 0.5 else -1)
        new_weights[i] += delta
        new_weights[j] -= delta
        if np.any(new_weights < 0):
            continue
        new_weights /= new_weights.sum()
        new_rmse = mean_squared_error(y, X @ new_weights, squared=False)
        if new_rmse < best_rmse:
            weights, best_rmse = new_weights, new_rmse
    return weights, best_rmse

hill_w, hill_rmse = hill_climb(final_train, y)
hill_w = np.clip(hill_w, 0, None)
hill_w /= hill_w.sum()
hill_oof = final_train @ hill_w
hill_preds = final_test @ hill_w

# =====================================
# Differential Evolution Ensemble
# =====================================
def rmse_loss(w, X, y):
    w = np.clip(np.array(w), 0, None)
    if w.sum() == 0:
        w = np.ones_like(w)
    w /= w.sum()
    preds = X @ w
    return mean_squared_error(y, preds, squared=False)

bounds = [(0.0, 1.0)] * final_train.shape[1]
result = differential_evolution(
    rmse_loss,
    bounds,
    args=(final_train, y),
    maxiter=500,
    popsize=15,
    tol=1e-6,
    seed=RANDOM_STATE,
    disp=False
)
de_w = np.clip(result.x, 0, None)
de_w /= de_w.sum()
de_oof = final_train @ de_w
de_preds = final_test @ de_w
de_rmse = mean_squared_error(y, de_oof, squared=False)

# =====================================
# Compare Strategies
# =====================================
leaderboard = {
    "Ridge": {"rmse": ridge_rmse, "preds": ridge_preds},
    "HillClimb": {"rmse": hill_rmse, "preds": hill_preds, "weights": hill_w},
    "DifferentialEvolution": {"rmse": de_rmse, "preds": de_preds, "weights": de_w}
}

print("\n### Ensemble Strategy RMSEs (lower is better):")
for k, v in leaderboard.items():
    print(f" {k:22s} : {v['rmse']:.6f}")

# =====================================
# Super-Ensemble (Ridge over 3 strategies)
# =====================================
meta_train = np.vstack([ridge_oof, hill_oof, de_oof]).T
meta_test = np.vstack([ridge_preds, hill_preds, de_preds]).T

final_oof = np.zeros(len(meta_train))
final_preds = np.zeros(len(meta_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(meta_train), start=1):
    X_tr, y_tr = meta_train[trn_idx], y.iloc[trn_idx]
    X_val, y_val = meta_train[val_idx], y.iloc[val_idx]

    meta_ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    meta_ridge.fit(X_tr, y_tr)

    final_oof[val_idx] = meta_ridge.predict(X_val)
    final_preds += meta_ridge.predict(meta_test) / N_SPLITS

meta_rmse = mean_squared_error(y, final_oof, squared=False)
print(f"\n### Super-Ensemble Ridge (on top of Ridge/Hill/DE) RMSE: {meta_rmse:.5f}")

# =====================================
# Save Submission
# =====================================
sample_sub = pd.read_csv(
    "/kaggle/input/playground-series-s5e9/sample_submission.csv",
    index_col="id"
)
sample_sub["BeatsPerMinute"] = final_preds
sample_sub.to_csv("submission.csv")
print("\nâœ… Final submission saved as submission.csv")



# import shap
# import matplotlib.pyplot as plt

# # Use Ridge as example meta-model
# ridge_meta_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)

# # Fit on all meta features
# ridge_meta_model.fit(final_train, y)

# # Initialize SHAP Explainer
# explainer = shap.Explainer(ridge_meta_model, final_train)
# shap_values = explainer(final_train)

# # Feature names (base + meta model predictions)
# meta_feature_names = [
#     "oof_lgb", "oof_xgb", "oof_cat",
#     "Ridge_oof", "Lasso_oof", "Linear_oof"
# ]

# # Summary plot
# shap.summary_plot(shap_values.values, features=final_train, feature_names=meta_feature_names, plot_type="bar")

# # Detailed beeswarm plot
# shap.summary_plot(shap_values.values, features=final_train, feature_names=meta_feature_names)

# # Optional: individual prediction explanation
# idx = 10  # example row
# shap.plots.waterfall(shap_values[idx])


