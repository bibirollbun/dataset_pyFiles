# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns


train_path = "/kaggle/input/playground-series-s5e9/train.csv"
test_path = "/kaggle/input/playground-series-s5e9/test.csv"
# Load data
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Shapes:")
print({"train": train.shape, "test": test.shape})

print("\nTrain columns:")
print(train.columns.tolist())

print("\nTest columns:")
print(test.columns.tolist())

print("\nData types (train):")
print(train.dtypes)

print("\nData types (test):")
print(test.dtypes)



# Quick peeks
from IPython.display import display

print("Train head:")
display(train.head())

print("\nTest head:")
display(test.head())

print("\nTrain describe (numeric):")
display(train.describe())

print("\nMissing values (train):")
display(train.isna().sum().sort_values(ascending=False))

print("\nMissing values (test):")
display(test.isna().sum().sort_values(ascending=False))



# Target distribution and correlations (if target exists)
TARGET = "BeatsPerMinute"

if TARGET in train.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(train[TARGET].dropna(), kde=True, ax=axes[0], bins=40)
    axes[0].set_title("Target distribution: BeatsPerMinute")
    sns.boxplot(x=train[TARGET], ax=axes[1])
    axes[1].set_title("Target boxplot")
    plt.show()

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    corr = train[numeric_cols].corr(numeric_only=True)
    if TARGET in corr.columns:
        top_corr = (
            corr[TARGET]
            .drop(labels=[TARGET])
            .sort_values(ascending=False)
            .head(15)
        )
        print("Top correlations with target:")
        display(top_corr)
else:
    print(f"Target column '{TARGET}' not found in train.")



# Persist lightweight data docs and artifacts
from pathlib import Path
import os
cur_dir = os.getcwd()

DOCS_DIR = Path( cur_dir + "/data_docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Columns and dtypes
cols_dtypes = {
    "train": {c: str(train[c].dtype) for c in train.columns},
    "test": {c: str(test[c].dtype) for c in test.columns},
}
(pd.Series(cols_dtypes["train"]).to_frame("dtype")
   .assign(split="train")
   .to_csv(DOCS_DIR / "columns_dtypes_train.csv"))
(pd.Series(cols_dtypes["test"]).to_frame("dtype")
   .assign(split="test")
   .to_csv(DOCS_DIR / "columns_dtypes_test.csv"))

# Shapes
pd.DataFrame({
    "split": ["train", "test"],
    "rows": [len(train), len(test)],
    "cols": [train.shape[1], test.shape[1]],
}).to_csv(DOCS_DIR / "shapes.csv", index=False)

# Missingness
missing_train = (train.isna().sum().to_frame("na_count")
                 .assign(na_ratio=lambda df: df["na_count"]/len(train)))
missing_test = (test.isna().sum().to_frame("na_count")
                .assign(na_ratio=lambda df: df["na_count"]/len(test)))
missing_train.to_csv(DOCS_DIR / "missingness_train.csv")
missing_test.to_csv(DOCS_DIR / "missingness_test.csv")

# Describe numeric
train.describe().to_csv(DOCS_DIR / "describe_numeric_train.csv")
test.describe().to_csv(DOCS_DIR / "describe_numeric_test.csv")

# Target correlations (if present)
TARGET = "BeatsPerMinute"
if TARGET in train.columns:
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    corr = train[numeric_cols].corr(numeric_only=True)
    if TARGET in corr.columns:
        corr[TARGET].drop(labels=[TARGET]).sort_values(ascending=False).to_csv(
            DOCS_DIR / "corr_with_target.csv"
        )

print(f"Wrote artifacts to: {DOCS_DIR}")



# Lightweight validations / invariants
TARGET = "BeatsPerMinute"

# 1) Target presence and no missing
if TARGET in train.columns:
    assert train[TARGET].notna().all(), "Target column has missing values"

# 2) No duplicate IDs in each split
if "id" in train.columns:
    assert train["id"].is_unique, "Duplicate ids in train"
if "id" in test.columns:
    assert test["id"].is_unique, "Duplicate ids in test"

# 3) Feature parity: test features should be train features minus target
train_features = [c for c in train.columns if c != TARGET]
if TARGET in train.columns:
    assert set(test.columns) == set(train_features), "Train/Test feature mismatch"

# 4) Basic range sanity checks for known numeric columns (customize as needed)
num_cols = [c for c in test.columns if pd.api.types.is_numeric_dtype(test[c])]
for col in num_cols:
    # example guards against infinities
    assert np.isfinite(test[col]).all(), f"Non-finite values in test column {col}"
    if col in train.columns:
        assert np.isfinite(train[col]).all(), f"Non-finite values in train column {col}"

print("All lightweight validations passed.")



shapes = pd.read_csv(DOCS_DIR / "shapes.csv")
miss_train = pd.read_csv(DOCS_DIR / "missingness_train.csv", index_col=0)
miss_test = pd.read_csv(DOCS_DIR / "missingness_test.csv", index_col=0)
cols_train = pd.read_csv(DOCS_DIR / "columns_dtypes_train.csv")
cols_test = pd.read_csv(DOCS_DIR / "columns_dtypes_test.csv")

corr_target = None
corr_path = DOCS_DIR / "corr_with_target.csv"
if corr_path.exists():
    corr_target = pd.read_csv(corr_path, index_col=0).squeeze()

print("Loaded artifacts:")
print({
    "shapes": shapes.shape,
    "miss_train": miss_train.shape,
    "miss_test": miss_test.shape,
    "cols_train": cols_train.shape,
    "cols_test": cols_test.shape,
    "corr_target": None if corr_target is None else int(corr_target.shape[0])
})


# Feature engineering helpers
from dataclasses import dataclass

@dataclass
class FEStats:
    means: pd.Series
    stds: pd.Series


def compute_stats(df: pd.DataFrame, exclude_cols: list[str]) -> FEStats:
    num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    means = df[num_cols].mean()
    stds = df[num_cols].std(ddof=1).replace(0, np.nan)
    return FEStats(means=means, stds=stds)


def feature_engineering(df: pd.DataFrame, stats: FEStats, is_train: bool, target_col: str = "BeatsPerMinute") -> pd.DataFrame:
    out = df.copy()

    # Common numeric columns excluding identifiers and target
    exclude = {"id", target_col}
    numeric_cols = [c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])]

    # 1) Monotonic transforms (log1p) for positively-skewed features
    if "TrackDurationMs" in out.columns:
        out["TrackDurationSec"] = out["TrackDurationMs"] / 1000.0
        out["log_TrackDurationMs"] = np.log1p(out["TrackDurationMs"].clip(lower=0))
    # clamp to (0,1] before logit for bounded vars
    bounded_cols = [c for c in numeric_cols if (out[c].min() >= 0) and (out[c].max() <= 1)]
    for c in bounded_cols:
        eps = 1e-6
        out[f"logit_{c}"] = np.log((out[c].clip(eps, 1-eps)) / (1 - out[c].clip(eps, 1-eps)))

    # 2) Interaction features capturing plausible audio relationships
    # Energy vs Loudness (expect positive relation)
    if set(["Energy", "AudioLoudness"]).issubset(out.columns):
        out["Energy_x_Loudness"] = out["Energy"] * (-out["AudioLoudness"])  # louder (more negative dB) -> higher magnitude
        out["Energy_div_LoudnessMag"] = out["Energy"] / (1e-6 + (-out["AudioLoudness"]).abs())

    # Rhythm x Energy synergy
    if set(["RhythmScore", "Energy"]).issubset(out.columns):
        out["Rhythm_x_Energy"] = out["RhythmScore"] * out["Energy"]

    # Vocal vs Instrumental contrast
    if set(["VocalContent", "InstrumentalScore"]).issubset(out.columns):
        out["Vocal_minus_Instrumental"] = out["VocalContent"] - out["InstrumentalScore"]
        out["Vocal_div_Instrumental"] = out["VocalContent"] / (1e-6 + out["InstrumentalScore"]) 

    # Acoustic quality interactions
    if "AcousticQuality" in out.columns:
        for partner in ["Energy", "RhythmScore", "VocalContent", "InstrumentalScore", "MoodScore"]:
            if partner in out.columns:
                out[f"AcousticQuality_x_{partner}"] = out["AcousticQuality"] * out[partner]

    # Live performance likelihood interactions (may relate to tempo variance)
    if "LivePerformanceLikelihood" in out.columns:
        for partner in ["Energy", "RhythmScore", "MoodScore"]:
            if partner in out.columns:
                out[f"LivePerf_x_{partner}"] = out["LivePerformanceLikelihood"] * out[partner]

    # 3) Centering and scaling (z-scores) using train stats
    for c in numeric_cols:
        if c in stats.means.index:
            mean_c = stats.means[c]
            std_c = stats.stds[c]
            if pd.notna(std_c) and std_c > 0:
                out[f"z_{c}"] = (out[c] - mean_c) / std_c

    # 4) Simple polynomial terms for top correlated features (from artifacts if available)
    top_feats = []
    if corr_target is not None:
        top_feats = corr_target.abs().sort_values(ascending=False).head(5).index.tolist()
    for c in top_feats:
        if c in out.columns and c != target_col:
            out[f"sq_{c}"] = out[c] ** 2
            out[f"sqrt_{c}"] = np.sqrt(out[c].clip(lower=0))

    # 5) Keep id and target columns as-is
    keep_cols = ["id"] + [target_col] if (is_train and target_col in out.columns) else ["id"]
    # Return with engineered columns appended
    return out




# Apply feature engineering and persist
EXCLUDE = ["id", "BeatsPerMinute"]
stats = compute_stats(train, exclude_cols=EXCLUDE)

train_fe = feature_engineering(train, stats, is_train=True)
test_fe = feature_engineering(test, stats, is_train=False)

# Preview columns and sizes
print({
    "train_fe_shape": train_fe.shape,
    "test_fe_shape": test_fe.shape,
    "new_cols_count": len([c for c in train_fe.columns if c not in train.columns])
})

# Persist to disk (robust to pyarrow issues)
FE_DIR = Path(cur_dir + "/data_fe")
FE_DIR.mkdir(parents=True, exist_ok=True)

# Coerce problematic dtypes

def _coerce_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        dt = out[c].dtype
        if pd.api.types.is_object_dtype(dt) or isinstance(dt, pd.CategoricalDtype):
            out[c] = out[c].astype("string")
        elif pd.api.types.is_bool_dtype(dt):
            out[c] = out[c].astype("bool")
        elif pd.api.types.is_integer_dtype(dt):
            # keep as int64
            out[c] = out[c].astype("int64", copy=False)
        elif pd.api.types.is_float_dtype(dt):
            out[c] = out[c].astype("float64", copy=False)
    return out

train_fe_out = _coerce_for_parquet(train_fe)
test_fe_out = _coerce_for_parquet(test_fe)

saved_parquet = False
try:
    train_fe_out.to_parquet(FE_DIR / "train_fe.parquet", engine="pyarrow", index=False)
    test_fe_out.to_parquet(FE_DIR / "test_fe.parquet", engine="pyarrow", index=False)
    saved_parquet = True
except Exception as e1:
    print("PyArrow write failed, trying fastparquet...", repr(e1))
    try:
        train_fe_out.to_parquet(FE_DIR / "train_fe.parquet", engine="fastparquet", index=False)
        test_fe_out.to_parquet(FE_DIR / "test_fe.parquet", engine="fastparquet", index=False)
        saved_parquet = True
    except Exception as e2:
        print("Fastparquet write failed, falling back to CSV...", repr(e2))
        train_fe_out.to_csv(FE_DIR / "train_fe.csv", index=False)
        test_fe_out.to_csv(FE_DIR / "test_fe.csv", index=False)

# Also save full CSV copies for maximum portability
train_fe_out.to_csv(FE_DIR / "train_fe.csv", index=False)
test_fe_out.to_csv(FE_DIR / "test_fe.csv", index=False)

# Also save a compact CSV with only id and new features (for inspection)
new_cols = [c for c in train_fe.columns if c not in train.columns and c != "BeatsPerMinute"]
train_new = train_fe[["id"] + new_cols]
train_new.to_csv(FE_DIR / "train_new_features_preview.csv", index=False)

print(f"Saved engineered datasets and preview to {FE_DIR}. Parquet_ok={saved_parquet}")



# Preview engineered columns and quick sanity checks
from IPython.display import display

# Show a sample of new columns
new_cols = [c for c in train_fe.columns if c not in train.columns]
print(f"Number of new features: {len(new_cols)}")
print("First 20 new features:")
print(new_cols[:20])

# Null checks on engineered features
na_counts = train_fe[new_cols].isna().sum().sort_values(ascending=False).head(10)
print("Top NA counts among new features:")
print(na_counts)

# Basic correlation of new features to target (top 10 by abs)
if "BeatsPerMinute" in train_fe.columns:
    corr_new = train_fe[new_cols + ["BeatsPerMinute"]].corr(numeric_only=True)["BeatsPerMinute"].drop("BeatsPerMinute").abs().sort_values(ascending=False)
    print("Top 10 |corr| of new features with target:")
    display(corr_new.head(10))

# Ensure id preserved and shapes match
assert "id" in train_fe.columns and "id" in test_fe.columns
assert set(test_fe.columns) == set(train_fe.columns) - {"BeatsPerMinute"}

print("Feature engineering preview checks passed.")




# Modeling: load engineered datasets with fallback
from pathlib import Path

FE_DIR = Path(cur_dir + "/data_fe")
train_fe_path_parquet = FE_DIR / "train_fe.parquet"
test_fe_path_parquet = FE_DIR / "test_fe.parquet"
train_fe_path_csv = FE_DIR / "train_fe.csv"
test_fe_path_csv = FE_DIR / "test_fe.csv"

def _read_fe(path_parquet: Path, path_csv: Path) -> pd.DataFrame:
    # Prefer CSV to avoid pyarrow extension issues
    if path_csv.exists():
        try:
            return pd.read_csv(path_csv)
        except Exception as e:
            print("CSV read failed, trying Parquet...", repr(e))
    # Try fastparquet then pyarrow
    if path_parquet.exists():
        try:
            return pd.read_parquet(path_parquet, engine="fastparquet")
        except Exception as e1:
            print("fastparquet read failed, trying pyarrow...", repr(e1))
            return pd.read_parquet(path_parquet, engine="pyarrow")
    raise FileNotFoundError(f"Neither {path_csv} nor {path_parquet} found")

train_fe = _read_fe(train_fe_path_parquet, train_fe_path_csv)
test_fe  = _read_fe(test_fe_path_parquet,  test_fe_path_csv)

print({
    "train_fe": train_fe.shape,
    "test_fe": test_fe.shape,
})

TARGET = "BeatsPerMinute"
features = [c for c in train_fe.columns if c not in ["id", TARGET]]
X = train_fe[features]
y = train_fe[TARGET]
X_test = test_fe[features]



# Modular training utilities
from dataclasses import dataclass
from typing import Dict, Tuple, List
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

@dataclass
class CVResult:
    oof: np.ndarray
    test: np.ndarray
    fold_scores: List[float]


def stratify_bins(y: pd.Series, n_bins: int = 20, random_state: int = 42) -> np.ndarray:
    # Quantile binning for approximate stratification of continuous target
    ranks = pd.qcut(y.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    return ranks.values


def train_catboost(X, y, X_test, n_splits=5, seed=42, params=None, use_gpu: bool = True) -> CVResult:
    from catboost import CatBoostRegressor, Pool
    if params is None:
        params = {
            'depth': 8,
            'learning_rate': 0.05,
            'l2_leaf_reg': 6.0,
            'loss_function': 'RMSE',
            'random_seed': seed,
            'allow_writing_files': False,
            'thread_count': -1,
        }
    # Inject GPU settings if requested and not already set
    if use_gpu and params.get('task_type') is None:
        params['task_type'] = 'GPU'
        params.setdefault('devices', '0')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    bins = stratify_bins(y)

    oof = np.zeros(len(X))
    test = np.zeros(len(X_test))
    scores = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, bins), 1):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val), verbose=False, use_best_model=True, early_stopping_rounds=200)
        val_pred = model.predict(X_val)
        oof[val_idx] = val_pred
        scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))
        test += model.predict(X_test) / n_splits
    return CVResult(oof=oof, test=test, fold_scores=scores)


def train_lightgbm(X, y, X_test, n_splits=5, seed=42, params=None, use_gpu: bool = True) -> CVResult:
    import lightgbm as lgb
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 1,
            'min_data_in_leaf': 500,
            'verbose': -1,
            'seed': seed,
        }
    # Inject GPU setting if requested and not already set
    if use_gpu and params.get('device') is None:
        params['device'] = 'gpu'
    
    # Handle DART mode (no early stopping)
    is_dart = params.get('boosting', 'gbdt') == 'dart'
    if is_dart:
        # DART doesn't support early stopping, use fixed rounds
        num_rounds = 5000  # Conservative for DART
        callbacks = [lgb.log_evaluation(500)]
    else:
        num_rounds = 20000
        callbacks = [lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(200)]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    bins = stratify_bins(y)

    oof = np.zeros(len(X))
    test = np.zeros(len(X_test))
    scores = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, bins), 1):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=num_rounds,
                          valid_sets=[dtrain, dvalid], valid_names=['train','valid'],
                          callbacks=callbacks)
        val_pred = model.predict(X_val, num_iteration=model.best_iteration if not is_dart else None)
        oof[val_idx] = val_pred
        scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))
        test += model.predict(X_test, num_iteration=model.best_iteration if not is_dart else None) / n_splits
    return CVResult(oof=oof, test=test, fold_scores=scores)


def train_xgboost(X, y, X_test, n_splits=5, seed=42, params=None, use_gpu: bool = True) -> CVResult:
    import xgboost as xgb
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 0,
            'max_leaves': 256,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'seed': seed,
        }
    # Inject GPU settings if requested
    if use_gpu:
        # XGBoost >=2.0: use device='cuda' with hist
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    bins = stratify_bins(y)

    oof = np.zeros(len(X))
    test = np.zeros(len(X_test))
    scores = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, bins), 1):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dvl = xgb.DMatrix(X_val, label=y_val)
        dte = xgb.DMatrix(X_test)
        model = xgb.train(params, dtr, num_boost_round=20000,
                          evals=[(dtr,'train'), (dvl,'valid')],
                          early_stopping_rounds=200,
                          verbose_eval=200)
        val_pred = model.predict(dvl, iteration_range=(0, model.best_iteration + 1))
        oof[val_idx] = val_pred
        scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))
        test += model.predict(dte, iteration_range=(0, model.best_iteration + 1)) / n_splits
    return CVResult(oof=oof, test=test, fold_scores=scores)


def optimize_blend_weights(preds: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
    # Non-negative weights that sum to 1 (simple constrained least squares via projection)
    names = list(preds.keys())
    P = np.column_stack([preds[n] for n in names])
    # closed-form unconstrained
    w = np.linalg.lstsq(P, y_true, rcond=None)[0]
    # project to simplex
    w = np.clip(w, 0, None)
    s = w.sum()
    if s == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return {n: float(wi) for n, wi in zip(names, w)}


def local_linear_correction(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    # Fit y_true ≈ a * y_pred + b
    A = np.column_stack([y_pred, np.ones_like(y_pred)])
    a, b = np.linalg.lstsq(A, y_true, rcond=None)[0]
    return float(a), float(b)


def apply_corrections(pred: np.ndarray, a: float, b: float, clip_range=(40.0, 220.0)) -> np.ndarray:
    out = a * pred + b
    if clip_range is not None:
        out = np.clip(out, clip_range[0], clip_range[1])
    return out



# Orchestrate training, optimized blending, correction, and submission
SEED = 220296
N_SPLITS = 5

cat_res = train_catboost(X, y, X_test, n_splits=N_SPLITS, seed=SEED)
lgb_res = train_lightgbm(X, y, X_test, n_splits=N_SPLITS, seed=SEED)
xgb_res = train_xgboost(X, y, X_test, n_splits=N_SPLITS, seed=SEED)

print({
    "cat_rmse_mean": float(np.mean(cat_res.fold_scores)),
    "lgb_rmse_mean": float(np.mean(lgb_res.fold_scores)),
    "xgb_rmse_mean": float(np.mean(xgb_res.fold_scores)),
})

# Optimize blend weights on OOF
preds_oof = {"cat": cat_res.oof, "lgb": lgb_res.oof, "xgb": xgb_res.oof}
weights = optimize_blend_weights(preds_oof, y.values)
print({"blend_weights": weights})

blend_oof = sum(weights[k] * preds_oof[k] for k in preds_oof)
blend_rmse = np.sqrt(mean_squared_error(y, blend_oof))
print({"blend_oof_rmse": float(blend_rmse)})

# Local linear correction on OOF and apply to test
blend_test = sum(weights[k] * v for k, v in {"cat": cat_res.test, "lgb": lgb_res.test, "xgb": xgb_res.test}.items())
a, b = local_linear_correction(y.values, blend_oof)
print({"linear_correction": {"a": a, "b": b}})

blend_test_corr = apply_corrections(blend_test, a, b, clip_range=(40.0, 220.0))

# Build submission
submission = pd.DataFrame({
    "id": test_fe["id"],
    "BeatsPerMinute": blend_test_corr
})
SUB_DIR = Path(cur_dir + "/submissions")
SUB_DIR.mkdir(parents=True, exist_ok=True)
sub_path = SUB_DIR / "submission_optimized.csv"
submission.to_csv(sub_path, index=False)
print(f"Wrote submission: {sub_path}")



# Training hygiene: early stopping, feature parity, and dataset caching
# Early stopping and round caps are already set: patience=200, num_boost_round=20000

# 1) Ensure validation features are identical to test features (no leakage)
val_features = features  # by construction above
assert set(val_features) == set(test_fe.columns) - {"id"}, "Validation features must match test features exactly"

# 2) Cache datasets in binary format to speed re-runs
from pathlib import Path
import os

CACHE_DIR = Path(cur_dir + "/model_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# LightGBM cache
try:
    import lightgbm as lgb
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)
    lgb_train.construct()
    lgb_train.save_binary(str(CACHE_DIR / "lgb_train.bin"))
    lgb_test = lgb.Dataset(X_test, reference=lgb_train, free_raw_data=False)
    lgb_test.construct()
    lgb_test.save_binary(str(CACHE_DIR / "lgb_test.bin"))
    print("Saved LightGBM binaries to", CACHE_DIR)
except Exception as e:
    print("LightGBM cache skipped:", repr(e))

# XGBoost cache
try:
    import xgboost as xgb
    dtrain_full = xgb.DMatrix(X, label=y)
    dtrain_full.save_binary(str(CACHE_DIR / "xgb_train.buffer"))
    dtest_full = xgb.DMatrix(X_test)
    dtest_full.save_binary(str(CACHE_DIR / "xgb_test.buffer"))
    print("Saved XGBoost binaries to", CACHE_DIR)
except Exception as e:
    print("XGBoost cache skipped:", repr(e))



# Optuna tuning for CatBoost, LightGBM, XGBoost
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

SEED = 220296
skf_bins = stratify_bins(y)

# 1) CatBoost objective
def objective_cat(trial: optuna.Trial) -> float:
    from catboost import CatBoostRegressor, Pool
    params = {
        'depth': trial.suggest_int('depth', 6, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.08),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 12.0),
        'loss_function': 'RMSE',
        'random_seed': SEED,
        'allow_writing_files': False,
        'thread_count': -1,
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'grow_policy': 'Lossguide',
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for trn_idx, val_idx in skf.split(X, skf_bins):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val), verbose=False, use_best_model=True, early_stopping_rounds=200)
        oof[val_idx] = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y, oof))
    return rmse

# 2) LightGBM objective (gbdt)
def objective_lgb_gbdt(trial: optuna.Trial) -> float:
    import lightgbm as lgb
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': trial.suggest_int('num_leaves', 63, 511),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 3000),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1,
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'verbose': -1,
        'seed': SEED,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for trn_idx, val_idx in skf.split(X, skf_bins):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=20000,
                          valid_sets=[dtrain, dvalid],
                          callbacks=[lgb.early_stopping(stopping_rounds=200)])
        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y, oof))
    return rmse

# 3) LightGBM objective (dart)
def objective_lgb_dart(trial: optuna.Trial) -> float:
    import lightgbm as lgb
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'dart',
        'learning_rate': 0.05,
        'num_leaves': trial.suggest_int('num_leaves', 63, 511),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 3000),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1,
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'drop_rate': trial.suggest_float('drop_rate', 0.05, 0.2),
        'verbose': -1,
        'seed': SEED,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for trn_idx, val_idx in skf.split(X, skf_bins):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=20000,
                          valid_sets=[dtrain, dvalid],
                          callbacks=[lgb.early_stopping(stopping_rounds=200)])
        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y, oof))
    return rmse

# 4) XGBoost objective
def objective_xgb(trial: optuna.Trial) -> float:
    import xgboost as xgb
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 0,
        'max_leaves': trial.suggest_int('max_leaves', 64, 512),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 20.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'tree_method': 'hist',
        'sampling_method': trial.suggest_categorical('sampling_method', ['uniform', 'gradient_based']),
        'seed': SEED,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for trn_idx, val_idx in skf.split(X, skf_bins):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dvl = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(params, dtr, num_boost_round=20000,
                          evals=[(dtr,'train'), (dvl,'valid')],
                          early_stopping_rounds=200,
                          verbose_eval=False)
        oof[val_idx] = model.predict(dvl, iteration_range=(0, model.best_iteration + 1))
    rmse = np.sqrt(mean_squared_error(y, oof))
    return rmse

print("Optuna objectives ready. Use small n_trials (e.g., 20-50) per model.")



# Run small Optuna sweeps and retrain best models
import optuna

N_TRIALS = 20

# XGBoost tuning
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=False)
print("Best XGB:", study_xgb.best_value, study_xgb.best_params)

best_xgb_params = study_xgb.best_params | {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'learning_rate': 0.05, 'tree_method': 'hist', 'seed': 220296, 'max_depth': 0}

xgb_res = train_xgboost(X, y, X_test, n_splits=5, seed=220296, params=best_xgb_params)

print({
    "xgb_rmse_mean": float(np.mean(xgb_res.fold_scores)),
})

# Blending and correction
preds_oof = {"xgb": xgb_res.oof}
weights = optimize_blend_weights(preds_oof, y.values)
blend_oof = sum(weights[k] * preds_oof[k] for k in preds_oof)
a, b = local_linear_correction(y.values, blend_oof)
blend_test = sum(weights[k] * v for k, v in {"xgb": xgb_res.test}.items())
blend_test_corr = apply_corrections(blend_test, a, b, clip_range=(40.0, 220.0))

# Submissions: 1) per-model raw, 2) blended, 3) blended + local linear correction
SUB_DIR = Path(cur_dir + "/submissions")
SUB_DIR.mkdir(parents=True, exist_ok=True)


pd.DataFrame({"id": test_fe["id"], "BeatsPerMinute": xgb_res.test}).to_csv(SUB_DIR / "submission_xgb.csv", index=False)

pd.DataFrame({"id": test_fe["id"], "BeatsPerMinute": blend_test}).to_csv(SUB_DIR / "submission_blend.csv", index=False)
pd.DataFrame({"id": test_fe["id"], "BeatsPerMinute": blend_test_corr}).to_csv(SUB_DIR / "submission_blend_corrected.csv", index=False)

print("Wrote submissions:")
print({
    "cat": str(SUB_DIR / "submission_cat.csv"),
    "lgb": str(SUB_DIR / "submission_lgb.csv"),
    "xgb": str(SUB_DIR / "submission_xgb.csv"),
    "blend": str(SUB_DIR / "submission_blend.csv"),
    "blend_corrected": str(SUB_DIR / "submission_blend_corrected.csv"),





