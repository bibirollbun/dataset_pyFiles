# --- Core imports & global config
import os, time, warnings, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy.optimize import nnls
from scipy.stats import pearsonr

from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# XGBoost is optional (not all Kaggle runtimes have it / sometimes no GPU)
try:
    import xgboost as xgb
except Exception:
    xgb = None
    print("XGBoost not available in this environment. Stage 2 will be skipped.")


# Silence minor warnings to keep notebook clean
warnings.filterwarnings("ignore")

# Display / plotting prefs
pd.options.display.max_columns = 200
pd.options.display.width = 180

plt.style.use("default")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

def rmse(a, b) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(a, b)))

# --- Reproducibility / runtime knobs ---

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

FAST_MODE   = False      # True => faster dev run (fewer trees / fewer seeds)
CV_SPLITS   = 5          # K folds
BINS_FOR_CV = 25         # quantile bins for stratified regression CV
N_JOBS      = os.cpu_count() or -1

ID_COL        = "id"
TARGET        = "accident_risk"
TARGET_BOUNDS = (0.0, 1.0)   # clip final predictions to this range

USE_RESIDUAL_MODEL = True   # Stage 2: XGBoost prior-corrected residual learner
USE_CALIBRATION    = True   # final clipping + OOF sanity checks

if FAST_MODE:
    N_SEEDS_LGB = 2
    LGB_TREES   = 600
    XGB_TREES   = 800
else:
    N_SEEDS_LGB = 2   # you can raise to 4 for extra stability (longer runtime)
    LGB_TREES   = 1100
    XGB_TREES   = 1200

SEED_SCHEDULE = [SEED + 1337 * i for i in range(N_SEEDS_LGB)]

# Decide best XGBoost tree_method depending on runtime (GPU or not)
_GPU_VISIBLE   = os.environ.get("CUDA_VISIBLE_DEVICES")
GPU_AVAILABLE  = (_GPU_VISIBLE not in (None, "", "-1"))
XGB_TREE_METHOD = "gpu_hist" if GPU_AVAILABLE else "hist"

# Working dir (for submission file)
WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

print(f"CV={CV_SPLITS} folds | stratified bins={BINS_FOR_CV}")
print(f"LGB_TREES={LGB_TREES} | XGB_TREES={XGB_TREES}")
print(f"Residual stage enabled? {USE_RESIDUAL_MODEL}")
print(f"Blending+clipping enabled? {USE_CALIBRATION}")
print(f"XGBoost tree_method: {XGB_TREE_METHOD} | GPU_AVAILABLE={GPU_AVAILABLE}")




TRAIN_FILE   = "/kaggle/input/playground-series-s5e10/train.csv"
TEST_FILE    = "/kaggle/input/playground-series-s5e10/test.csv"
SAMPLE_FILE  = "/kaggle/input/playground-series-s5e10/sample_submission.csv"

# Load CSVs
train_raw  = pd.read_csv(TRAIN_FILE)
test_raw   = pd.read_csv(TEST_FILE)
sample_sub = pd.read_csv(SAMPLE_FILE)


# Preview basic shapes / sanity check

print(f"Train shape: {train_raw.shape} | Test shape: {test_raw.shape}")
display(train_raw.head())

# --- Integrity checks / no-leakage policy ---------------------------------

# 1. Train must contain all test columns + the target
expected_cols = set(test_raw.columns.tolist() + [TARGET])
missing_cols  = [c for c in expected_cols if c not in train_raw.columns]
assert len(missing_cols) == 0, f"Train is missing expected columns: {missing_cols}"

# 2. Target must only exist in train, never in test
assert TARGET in train_raw.columns, f"Target column '{TARGET}' not found in train."
assert TARGET not in test_raw.columns, "Test should not contain the target."

# 3. ID column must exist in both
assert ID_COL in train_raw.columns, f"ID column '{ID_COL}' not found in train."
assert ID_COL in test_raw.columns,  f"ID column '{ID_COL}' not found in test."

# We'll work on local copies
train = train_raw.copy()
test  = test_raw.copy()

print("Data integrity checks passed.")
print("Using official competition data (train/test from /kaggle/input).")
print("No external data. No test-target leakage.")



# --- Basic dataset audit ---------------------------------------------------
# - dtypes / structure
# - missingness
# - numeric summary
# - correlations / simple relationships

print("Dataset Overview:")
train.info()

# --- Missing values --------------------------------------------------------
missing = (
    train.isnull()
         .mean()
         .sort_values(ascending=False)
)
missing = missing[missing > 0]

print("\nMissing Values (ratio):")
if len(missing) > 0:
    display(missing.to_frame("missing_ratio"))
else:
    print("No missing values detected.")

# --- Numerical summary -----------------------------------------------------
print("\nNumerical Summary:")
try:
    display(
        train.describe()
             .T
             .style.background_gradient(cmap="Blues")
    )
except Exception:
    # fallback if .style isn't available 
    display(train.describe().T)

# --- Correlation with target ----------------------------------------------
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
corr = train[num_cols].corr()

# Sort features by absolute correlation with the target
target_corr = (
    corr[TARGET]
    .drop(TARGET)
    .sort_values(ascending=False)
)

top_features = target_corr.abs().head(12).index.tolist()

# Heatmap on the most relevant numerical features (for readability)
mask = np.triu(np.ones((len(top_features), len(top_features))), k=1)

plt.figure(figsize=(10,7))
sns.heatmap(
    corr.loc[top_features, top_features],
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.6,
    linecolor="gray",
    mask=mask,
    cbar_kws={"shrink": 0.8},
)
plt.title("Top correlated numerical features", fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- Example relationship: curvature vs accident_risk ---------------------
if "curvature" in train.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x=train["curvature"],
        y=train[TARGET],
        alpha=0.35,
        edgecolor=None,
        s=18
    )
    plt.title("curvature vs accident_risk", fontsize=11)
    plt.xlabel("curvature")
    plt.ylabel("accident_risk")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# --- Risk by weather category  ------------------------------
if "weather" in train.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(
        x=train["weather"],
        y=train[TARGET]
    )
    plt.title("accident_risk by weather", fontsize=11)
    plt.xlabel("weather")
    plt.ylabel("accident_risk")
    plt.xticks(rotation=25)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# --- Speed limit distribution ------------------------------
if "speed_limit" in train.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(
        train["speed_limit"],
        bins=30,
        kde=True
    )
    plt.title("Distribution of speed_limit", fontsize=11)
    plt.xlabel("speed_limit")
    plt.ylabel("count")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()





required_cols = ["curvature", "speed_limit", "num_reported_accidents"]
missing_required = [c for c in required_cols if c not in train.columns]
assert len(missing_required) == 0, f"Missing required columns: {missing_required}"

def _col(df, name, dtype=float, fill=0.0):
    """Safe column accessor with fallback default."""
    if name in df.columns:
        return df[name].astype(dtype).values
    return np.full(len(df), fill, dtype=dtype)

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize basic categoricals if present
    for c in ("lighting", "weather"):
        if c in df.columns:
            s = df[c].astype(str).str.lower().str.strip()
            df[c] = s.replace({"nan": "unknown"}).fillna("unknown")

    curvature   = _col(df, "curvature", float, 0.0)
    speed_limit = _col(df, "speed_limit", float, 0.0)
    num_acc     = _col(df, "num_reported_accidents", float, 0.0)
    num_lanes   = _col(df, "num_lanes", float, 1.0)

    # interpretable engineered features
    df["curv_speed"]    = curvature * speed_limit
    df["curv_ratio"]    = curvature / np.where(speed_limit == 0, 1.0, speed_limit)
    df["acc_log"]       = np.log1p(np.clip(num_acc, 0, None))
    df["acc_per_lane"]  = num_acc / np.where(num_lanes == 0, 1.0, num_lanes)
    df["critical_zone"] = ((curvature > 0.6) & (speed_limit > 80)).astype(int)

    return df

train = engineer(train)
test  = engineer(test)

print("Feature engineering applied.")
print(f"Train shape after FE: {train.shape} | Test shape after FE: {test.shape}")
display(train[["curvature", "speed_limit", "curv_speed", "acc_log", "critical_zone"]].head())

# --- Interpretable prior ---------------------------------------------------

def risk_prior_fn(df: pd.DataFrame) -> np.ndarray:
    """Hand-designed baseline risk score in [0,1]."""
    n = len(df)

    curvature = _col(df, "curvature", float, 0.0)
    speed_lim = _col(df, "speed_limit", float, 0.0)

    lighting = df.get("lighting", pd.Series(["unknown"] * n)).astype(str).str.lower()
    weather  = df.get("weather",  pd.Series(["clear"]   * n)).astype(str).str.lower()

    night_flag = (lighting == "night").astype(int).values
    badwx_flag = (weather  != "clear").astype(int).values  # anything not "clear" = worse condition

    # normalize curvature/speed to [0,1] scale
    curv_norm  = curvature / (curvature.max() + 1e-9)
    speed_norm = speed_lim / (speed_lim.max() + 1e-9)

    score = (
        0.30 * curv_norm +
        0.20 * speed_norm +
        0.30 * night_flag +
        0.20 * badwx_flag
    )

    return np.clip(score, 0, 1).astype(float)

train["risk_prior"] = risk_prior_fn(train)
test["risk_prior"]  = risk_prior_fn(test)

print("risk_prior added.")
print(train["risk_prior"].describe()[["min", "max", "mean"]])
display(train[["risk_prior", TARGET]].head())

# --- risk_prior vs true target --------------------------------------------

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=train["risk_prior"],
    y=train[TARGET],
    alpha=0.4,
    edgecolor=None,
    s=18
)
plt.title("risk_prior vs accident_risk", fontsize=11)
plt.xlabel("risk_prior (~[0,1])")
plt.ylabel("accident_risk")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()



# --- Categorical encoding --------------------------------------------------

cat_cols = [
    c for c in train.columns
    if train[c].dtype == "object" and c not in [ID_COL, TARGET]
]

encoders = {}

for c in cat_cols:
    le = LabelEncoder()
    both = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0)
    le.fit(both)
    train[c] = le.transform(train[c].astype(str)).astype("int32")
    test[c]  = le.transform(test[c].astype(str)).astype("int32")
    encoders[c] = le

print(f"Categorical columns encoded: {cat_cols}")

# --- Build final feature matrices -----------------------------------------

feat_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

X      = train[feat_cols].copy()
y      = train[TARGET].astype(float).copy()
X_test = test[feat_cols].copy()

num_cols_model = [c for c in feat_cols if np.issubdtype(X[c].dtype, np.number)]
cat_cols_model = [c for c in cat_cols if c in feat_cols]

assert list(X.columns) == list(X_test.columns), "Feature mismatch between train and test."
assert y.notnull().all(), "Target contains NaNs."

print("Feature matrix ready.")
print(f"Total features: {len(feat_cols)}")
print(f"{len(num_cols_model)} numeric | {len(cat_cols_model)} categorical")
print(f"Target range: {y.min():.4f} â†’ {y.max():.4f} | mean={y.mean():.4f}")



def make_folds_stratified_reg(y, n_splits=5, seed=42, bins=25):
    """
    Build stratified folds for a continuous target by:
    1. binning y into quantile buckets,
    2. running StratifiedKFold on those bins.

    Returns:
        folds   : list of (train_idx, valid_idx) pairs
        y_bins  : bin index per sample (for diagnostics / debugging)
    """
    assert bins <= len(y), "bins must be <= number of samples"

    y_bins = pd.qcut(y, q=bins, duplicates="drop").cat.codes

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    fold_pairs = list(skf.split(np.zeros(len(y)), y_bins))

    return fold_pairs, y_bins


folds, y_bins = make_folds_stratified_reg(
    y,
    n_splits=CV_SPLITS,
    seed=SEED,
    bins=BINS_FOR_CV
)

print(
    f"CV folds: {len(folds)} | "
    f"stratified bins: {y_bins.nunique()}"
)



# - Train directly on accident_risk
# - Bag across multiple random seeds 
# - Collect OOF predictions and per-fold RMSE

base_params = dict(
    n_estimators=LGB_TREES,
    learning_rate=0.042 if not FAST_MODE else 0.05,
    num_leaves=55,
    subsample=0.9,
    colsample_bytree=0.82,
    min_child_samples=40,
    reg_lambda=0.05,
    n_jobs=N_JOBS,
)

oof_lgb        = np.zeros(len(X), dtype=float)
pred_lgb       = np.zeros(len(X_test), dtype=float)
fold_rmses_lgb = []
best_iters_lgb = []

t_lgb = time.perf_counter()

for seed_idx, seed in enumerate(SEED_SCHEDULE, start=1):
    params = {**base_params, "random_state": seed}

    oof_tmp  = np.zeros(len(X), dtype=float)
    pred_tmp = np.zeros(len(X_test), dtype=float)

    print(f"\n[Stage 1 / LightGBM] seed {seed} ({seed_idx}/{len(SEED_SCHEDULE)})")

    for f, (tr_idx, va_idx) in enumerate(folds, start=1):
        model_lgb = LGBMRegressor(**params)

        model_lgb.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[early_stopping(100), log_evaluation(0)],
        )

        best_iters_lgb.append(int(model_lgb.best_iteration_))

        # OOF preds for this fold
        oof_tmp[va_idx] = model_lgb.predict(
            X.iloc[va_idx],
            num_iteration=model_lgb.best_iteration_
        )

        # Test preds (fold-avg within this seed)
        pred_tmp += model_lgb.predict(
            X_test,
            num_iteration=model_lgb.best_iteration_
        ) / len(folds)

        fold_rmse = rmse(y.iloc[va_idx], oof_tmp[va_idx])
        print(f"  Fold {f} RMSE (this seed) = {fold_rmse:.5f} (best_iter={model_lgb.best_iteration_})")


    # Average this seed into global accumulators
    oof_lgb  += oof_tmp  / len(SEED_SCHEDULE)
    pred_lgb += pred_tmp / len(SEED_SCHEDULE)

    seed_rmse = rmse(y, oof_tmp)
    print(f"Seed OOF RMSE = {seed_rmse:.5f}")

# After averaging across seeds, compute per-fold RMSE using final OOF
for _, va_idx in folds:
    fold_rmses_lgb.append(rmse(y.iloc[va_idx], oof_lgb[va_idx]))

elapsed_lgb = time.perf_counter() - t_lgb

rmse_lgb = rmse(y, oof_lgb)

print("\n[Stage 1 â€” LightGBM summary]")
print(f"OOF RMSE              = {rmse_lgb:.5f}")
print(f"Per-fold RMSE         = {[f'{v:.5f}' for v in fold_rmses_lgb]}")
print(f"Fold RMSE std         = {np.std(fold_rmses_lgb):.5f}")
print(f"Runtime (s)           = {elapsed_lgb:.1f}")
print(f"Mean(best_iteration_) = {np.mean(best_iters_lgb):.1f}")

# Optional diagnostics (can be skipped to keep the notebook lighter)
if False:
    lgb_residual = y.values - oof_lgb

    plt.figure(figsize=(6,4))
    sns.histplot(lgb_residual, bins=40, kde=True)
    plt.title("Residuals After LightGBM (Stage 1)", fontsize=11)
    plt.xlabel("true - lgb_pred")
    plt.ylabel("count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.scatter(
        oof_lgb,
        lgb_residual,
        s=10,
        alpha=0.4
    )
    plt.axhline(0, color="#555555", linestyle="--", linewidth=1)
    plt.title("Residuals vs LightGBM OOF Prediction", fontsize=11)
    plt.xlabel("LightGBM OOF prediction")
    plt.ylabel("true - pred")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Feature importance from the last fitted model
try:
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model_lgb.feature_importances_.astype(float),
    }).sort_values("importance", ascending=False)

    top_imp = importances.head(20)

    print("\nTop LightGBM features:")
    display(top_imp)

    plt.figure(figsize=(6,5))
    plt.barh(top_imp["feature"][::-1], top_imp["importance"][::-1])
    plt.title("Top 20 Features (Stage 1 â€” LightGBM)", fontsize=11)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Feature importance skipped:", e)



# Train on (accident_risk - risk_prior), then add risk_prior back at inference.

if (xgb is None) or (not USE_RESIDUAL_MODEL):
    print("Stage 2 skipped (XGBoost not available or residual stage disabled).")
    oof_xgb        = np.zeros(len(X), dtype=float)
    pred_xgb       = np.zeros(len(X_test), dtype=float)
    fold_rmses_xgb = []
    best_iters_xgb = []
    rmse_xgb       = rmse(y, oof_xgb)

else:
    # residual target: what the simple prior did not explain
    xgb_target = (y.values - train["risk_prior"].values).astype(float)

    oof_xgb        = np.zeros(len(X), dtype=float)
    pred_xgb       = np.zeros(len(X_test), dtype=float)
    fold_rmses_xgb = []
    best_iters_xgb = []

    t_xgb = time.perf_counter()

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.035 if not FAST_MODE else 0.05,
        "max_depth": 6,
        "subsample": 0.92,
        "colsample_bytree": 0.86,
        "lambda": 1.0,
        "alpha": 0.6,
        "tree_method": XGB_TREE_METHOD, 
        "seed": SEED,
        "nthread": -1,
    }

    print(f"[Stage 2 / XGBoost] tree_method = {XGB_TREE_METHOD}")

    for f, (tr_idx, va_idx) in enumerate(folds, start=1):
        dtrain = xgb.DMatrix(X.iloc[tr_idx],  label=xgb_target[tr_idx])
        dvalid = xgb.DMatrix(X.iloc[va_idx],  label=xgb_target[va_idx])
        dtest  = xgb.DMatrix(X_test)

        booster = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=XGB_TREES,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        best_iter        = getattr(booster, "best_iteration", None)
        best_ntree_limit = getattr(booster, "best_ntree_limit", None)
        best_iters_xgb.append(int(best_iter) if best_iter is not None else None)

        # predict residuals on valid and test
        if best_ntree_limit is not None:
            valid_residual_pred = booster.predict(dvalid, ntree_limit=int(best_ntree_limit))
            test_residual_pred  = booster.predict(dtest,  ntree_limit=int(best_ntree_limit))
        elif best_iter is not None:
            valid_residual_pred = booster.predict(dvalid, iteration_range=(0, int(best_iter)+1))
            test_residual_pred  = booster.predict(dtest,  iteration_range=(0, int(best_iter)+1))
            best_ntree_limit = int(best_iter) + 1
        else:
            valid_residual_pred = booster.predict(dvalid)
            test_residual_pred  = booster.predict(dtest)

        # map residuals back to full accident_risk space
        oof_xgb[va_idx] = valid_residual_pred + train["risk_prior"].values[va_idx]
        pred_xgb       += (test_residual_pred + test["risk_prior"].values) / len(folds)

        fold_rmse = rmse(y.iloc[va_idx], oof_xgb[va_idx])
        fold_rmses_xgb.append(fold_rmse)
        print(f"  Fold {f} RMSE = {fold_rmse:.5f} | best_iter={best_iter} | best_ntree_limit={best_ntree_limit}")

    elapsed_xgb = time.perf_counter() - t_xgb
    rmse_xgb    = rmse(y, oof_xgb)

    print("\n[Stage 2 â€” XGBoost summary]")
    print(f"OOF RMSE                      = {rmse_xgb:.5f}")
    print(f"Per-fold RMSE                 = {[f'{v:.5f}' for v in fold_rmses_xgb]}")
    print(f"Fold RMSE std                 = {np.std(fold_rmses_xgb):.5f}")
    print(f"Runtime (s)                   = {elapsed_xgb:.1f}")
    print(f"Best iterations (first folds) = {best_iters_xgb[:5]}")



# Stage-wise OOF RMSE comparison (Stage 1 vs Stage 2)

rmse_lgb = rmse(y, oof_lgb)
rmse_xgb = rmse(y, oof_xgb)

comp_df = pd.DataFrame({
    "Stage": [
        "Stage 1 â€” LightGBM (main learner)",
        "Stage 2 â€” XGBoost (prior-corrected residual learner)",
    ],
    "OOF_RMSE": [
        rmse_lgb,
        rmse_xgb,
    ],
    "Î” vs Stage 1": [
        0.0,
        rmse_lgb - rmse_xgb,
    ],
    "Notes": [
        "Fits accident_risk directly",
        "Trains on (accident_risk - risk_prior), then adds risk_prior back",
    ],
})

print("Stage RMSE comparison:")
try:
    display(
        comp_df.style.format({
            "OOF_RMSE": "{:.5f}",
            "Î” vs Stage 1": "{:+.5f}",
        }).background_gradient(
            cmap="Blues",
            subset=["OOF_RMSE", "Î” vs Stage 1"]
        )
    )
except Exception:
    print(comp_df.to_string(index=False))



# Combine Stage 1 (LightGBM) and Stage 2 (XGBoost residual) using non-negative least squares.
# NNLS enforces weights >= 0, then we normalize so they sum to 1.

oof_stack  = np.vstack([oof_lgb,  oof_xgb]).T   # [n_train, 2]
test_stack = np.vstack([pred_lgb, pred_xgb]).T  # [n_test, 2]

w_raw, _ = nnls(oof_stack, y.values.astype(float))
w = w_raw / (w_raw.sum() + 1e-12)

print("\n[Stage 3 / NNLS Blend] Weights:")
print(f"  Stage 1 â€” LightGBM (main learner)                        : {w[0]:.4f}")
print(f"  Stage 2 â€” XGBoost (prior-corrected residual learner)     : {w[1]:.4f}")

blend_oof  = oof_stack  @ w
blend_test = test_stack @ w

rmse_blend = rmse(y, blend_oof)

print("\n[Stage 3 â€” NNLS Blend summary]")
print(f"OOF RMSE (Stage 1 â€” LightGBM)                              = {rmse_lgb:.5f}")
print(f"OOF RMSE (Stage 2 â€” XGBoost, prior-corrected residual)     = {rmse_xgb:.5f}")
print(f"OOF RMSE (Stage 3 â€” NNLS Blend, non-negative convex mix)   = {rmse_blend:.5f}")
print(f"Î” vs Stage 1                                               = {rmse_lgb - rmse_blend:+.5f}")
print(f"Î” vs Stage 2                                               = {rmse_xgb - rmse_blend:+.5f}")

# visualize the learned NNLS weights
plt.figure(figsize=(4,3))
plt.bar(
    ["Stage 1\nLightGBM", "Stage 2\nXGBoost residual"],
    w
)
plt.title("NNLS blend weights (Stage 3)", fontsize=11)
plt.ylabel("Weight")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Build final comparison table including Stage 3
blend_comp_df = pd.DataFrame({
    "Stage": [
        "Stage 1 â€” LightGBM (main learner)",
        "Stage 2 â€” XGBoost (prior-corrected residual learner)",
        "Stage 3 â€” NNLS Blend (non-negative convex combiner)",
    ],
    "OOF_RMSE": [
        rmse_lgb,
        rmse_xgb,
        rmse_blend,
    ],
    "Î” vs Stage 1": [
        0.0,
        rmse_lgb - rmse_xgb,
        rmse_lgb - rmse_blend,
    ],
    "Notes": [
        "Fits accident_risk directly",
        "Learns (accident_risk - risk_prior), then adds risk_prior back",
        "Non-negative convex mix of Stage 1 and Stage 2",
    ],
})

print("\nStage 1 / Stage 2 / Stage 3 comparison:")
try:
    display(
        blend_comp_df.style.format({
            "OOF_RMSE": "{:.5f}",
            "Î” vs Stage 1": "{:+.5f}",
        }).background_gradient(
            cmap="Blues",
            subset=["OOF_RMSE", "Î” vs Stage 1"]
        )
    )
except Exception:
    print(blend_comp_df.to_string(index=False))



# Final calibration / clipping to [0,1]
lo, hi = TARGET_BOUNDS
final_oof  = np.clip(blend_oof,  lo, hi)
final_test = np.clip(blend_test, lo, hi)

final_rmse = rmse(y, final_oof)
final_r    = pearsonr(y, final_oof)[0]

print("[Final blended model diagnostics]")
print(f"OOF RMSE (clipped to [{lo},{hi}])     = {final_rmse:.5f}")
print(f"Pearson r (OOF)                       = {final_r:.4f}")
print(f"Prediction range (OOF)                = {final_oof.min():.4f} â†’ {final_oof.max():.4f}")
print(f"Prediction range (TEST)               = {final_test.min():.4f} â†’ {final_test.max():.4f}")

# True vs predicted: calibration / monotonicity check
plt.figure(figsize=(6,5))
hb = plt.hexbin(
    y,
    final_oof,
    gridsize=30,
    cmap="Blues",
    mincnt=1,
)
plt.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1)
plt.colorbar(hb, label="density")
plt.title(f"OOF true vs predicted | r = {final_r:.3f}", fontsize=12)
plt.xlabel("true accident_risk")
plt.ylabel("predicted accident_risk (OOF)")
plt.xlim(lo, hi)
plt.ylim(lo, hi)
plt.grid(alpha=0.25, linestyle="--", linewidth=0.6)
plt.tight_layout()
plt.show()

# Per-fold RMSE comparison for all stages
fold_rmses_blend = [rmse(y.iloc[va], blend_oof[va]) for _, va in folds]

idx   = np.arange(1, len(folds) + 1)
width = 0.27

plt.figure(figsize=(10,4.6))

bars_lgb   = plt.bar(idx - width, fold_rmses_lgb,   width, label="Stage 1 â€” LightGBM")
bars_xgb   = plt.bar(idx,         fold_rmses_xgb,   width, label="Stage 2 â€” XGBoost residual")
bars_blend = plt.bar(idx + width, fold_rmses_blend, width, label="Stage 3 â€” NNLS blend")

for rects in [bars_lgb, bars_xgb, bars_blend]:
    for bar in rects:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h,
            f"{h:.5f}",
            ha='center',
            va='bottom',
            fontsize=8,
        )

plt.xticks(idx, [f"Fold {i}" for i in idx], fontsize=9)
plt.ylabel("RMSE", fontsize=10)
plt.title("Per-fold OOF RMSE by stage", fontsize=12)
plt.grid(alpha=0.25, axis='y')
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=True
)
plt.tight_layout()
plt.show()

# Blend weight stability via bootstrap
def nnls_bootstrap(oof_preds, y_true, iters=20, seed=SEED):
    rng = np.random.default_rng(seed)
    weights = []
    n = len(y_true)

    for _ in range(iters):
        idx_bs = rng.integers(0, n, n)
        w_bs, _ = nnls(oof_preds[idx_bs], y_true[idx_bs].astype(float))
        w_bs = w_bs / (w_bs.sum() + 1e-12)
        weights.append(w_bs)

    return np.vstack(weights)

W = nnls_bootstrap(oof_stack, y.values, iters=20, seed=SEED)
mean_lgb, std_lgb = W[:, 0].mean(), W[:, 0].std()
mean_xgb, std_xgb = W[:, 1].mean(), W[:, 1].std()

print("[NNLS weight stability]")
print(f"Stage 1 â€” LightGBM weight:     mean={mean_lgb:.4f}  std={std_lgb:.4f}")
print(f"Stage 2 â€” XGBoost weight:      mean={mean_xgb:.4f}  std={std_xgb:.4f}")

plt.figure(figsize=(5.5,3.4))
plt.boxplot(
    W,
    labels=["Stage 1 â€” LightGBM", "Stage 2 â€” XGBoost residual"],
    patch_artist=True,
    boxprops=dict(facecolor="#e0ecf8"),
    medianprops=dict(color="#d62728", linewidth=1.5),
)
plt.title("NNLS blend weight distribution (bootstrap)", fontsize=11)
plt.ylabel("blend weight")
plt.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
plt.tight_layout()
plt.show()



# Build submission dataframe
submission = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    TARGET: final_test,
})

submit_path = WORK_DIR / "submission.csv"
submission.to_csv(submit_path, index=False)

print("[Submission file]")
print(f"path        : {submit_path}")
print(f"rows        : {submission.shape[0]}")
print(f"columns     : {list(submission.columns)}")
print(f"prediction range (test) : {final_test.min():.4f} â†’ {final_test.max():.4f}")
print(f"prediction mean (test)  : {final_test.mean():.4f}")

display(submission.head())


