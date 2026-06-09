# =========================================================
# Starter Imports
# =========================================================

# Standard libraries
import os
import gc
import sys
import warnings
from pathlib import Path

# Utilities
import numpy as np
import pandas as pd
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# Hyperparameter optimization
import optuna

# Warnings & plotting style
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# Data paths
DATA_DIR = Path("/kaggle/input/playground-series-s5e9")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

# Quick data load check
if TRAIN_PATH.exists():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Sample submission shape:", sample_sub.shape)
else:
    print("âš ï¸� Data files not found in", DATA_DIR)

train0 = train.copy()
test0 = test.copy()


train.head()


# features distribution
train.hist(bins=50, figsize=(10, 10))
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
sns.histplot(train["BeatsPerMinute"], bins=50, kde=True, color="steelblue")
plt.title("Target distribution â€“ BeatsPerMinute", fontsize=14)
plt.xlabel("BPM")
plt.ylabel("Count")
plt.show()


corr = train.corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr[["BeatsPerMinute"]].sort_values(by="BeatsPerMinute", ascending=False),
            annot=True, cmap="coolwarm", cbar=False, vmin=-1, vmax=1)
plt.title("Correlation of features with BeatsPerMinute", fontsize=14)
plt.show()



num_features = [
    "RhythmScore", "AudioLoudness", "VocalContent",
    "AcousticQuality", "InstrumentalScore",
    "LivePerformanceLikelihood", "MoodScore",
    "TrackDurationMs", "Energy"
]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,12))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.scatterplot(x=train[col], y=train["BeatsPerMinute"], alpha=0.3, ax=axes[i], s=20)
    axes[i].set_title(f"{col} vs BPM")

plt.tight_layout()
plt.show()



# Create tempo bins
bins = [0, 100, 130, 200]
labels = ["Slow (<100)", "Medium (100-130)", "Fast (>130)"]
train["BPM_bin"] = pd.cut(train["BeatsPerMinute"], bins=bins, labels=labels)

plt.figure(figsize=(12,6))
sns.boxplot(x="BPM_bin", y="Energy", data=train, palette="Set2")
plt.title("Energy distribution across BPM bins", fontsize=14)
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x="BPM_bin", y="RhythmScore", data=train, palette="Set3")
plt.title("RhythmScore distribution across BPM bins", fontsize=14)
plt.show()



# correlation heatmap between all numeric features
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation Heatmap of Numeric Features", fontsize=16)
plt.show()


from pandas .plotting import scatter_matrix
attributes = ["BeatsPerMinute", "RhythmScore", "Energy", "AudioLoudness", "VocalContent"]
scatter_matrix(train[attributes], figsize=(12, 8), diagonal='kde')
plt.suptitle("Scatter Matrix of Key Features", fontsize=16)
plt.show()


# Copy dataset
df = train.copy()

# Candidate engineered features
df["Vocals_Ratio"] = df["VocalContent"] / (df["VocalContent"] + df["InstrumentalScore"] + 1e-6)
df["Loudness_per_Energy"] = df["AudioLoudness"] / (df["Energy"] + 1e-6)
df["RhythmEnergy"] = df["RhythmScore"] * df["Energy"]
df["Mood_minus_Energy"] = df["MoodScore"] - df["Energy"]
df["Acoustic_minus_Instrumental"] = df["AcousticQuality"] - df["InstrumentalScore"]
df["LogDuration"] = np.log1p(df["TrackDurationMs"])

# Recompute correlations
corr_new = df.corr(numeric_only=True)["BeatsPerMinute"].sort_values(ascending=False)

print(corr_new.head(15))
print(corr_new.tail(10))



# =========================================================
# Fast prototyping: RF + XGB + LGBM, K=3, sampling opzionale
# =========================================================

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ---------- Config ----------
TARGET = "BeatsPerMinute"
ID_COL = "id"
USE_SAMPLE = True          # <-- metti False per usare tutto
N_SAMPLES = 120_000        # <-- regola per velocitÃ /accuratezza
N_SPLITS = 3               # <-- 3-fold CV

# ---------- Data ----------
df = train.copy()
if USE_SAMPLE:
    df = df.sample(N_SAMPLES, random_state=42)

y = df[TARGET].astype("float32").values
X = df.drop(columns=[TARGET, ID_COL, "BPM_bin"])

# downcast numerici per risparmiare RAM
for c in X.columns:
    if pd.api.types.is_float_dtype(X[c]) or pd.api.types.is_integer_dtype(X[c]):
        X[c] = pd.to_numeric(X[c], downcast="float")
FEATURES = X.columns.tolist()

# ---------- Feature engineering ----------
def add_engineered_features(X_df: pd.DataFrame) -> pd.DataFrame:
    X_df = X_df.copy()
    eps = 1e-6
    X_df["LogDuration"] = np.log1p(X_df["TrackDurationMs"])
    X_df["Vocals_Ratio"] = X_df["VocalContent"] / (X_df["VocalContent"] + X_df["InstrumentalScore"] + eps)
    X_df["Loudness_per_Energy"] = X_df["AudioLoudness"] / (X_df["Energy"] + eps)
    X_df["Mood_minus_Energy"] = X_df["MoodScore"] - X_df["Energy"]
    X_df["RhythmEnergy"] = X_df["RhythmScore"] * X_df["Energy"]
    return X_df

feat_eng = FunctionTransformer(add_engineered_features, validate=False)

# ---------- Preprocessor ----------
# quando serve scalare (qui solo per completezza: trees non lo richiedono)
def make_preprocessor(apply_scaling: bool = False) -> ColumnTransformer:
    if apply_scaling:
        return ColumnTransformer(
            transformers=[("scale", StandardScaler(), FEATURES + [
                "LogDuration","Vocals_Ratio","Loudness_per_Energy","Mood_minus_Energy","RhythmEnergy"
            ])],
            remainder="drop",
            verbose_feature_names_out=False
        )
    else:
        # passa tutto senza elencare le colonne (nuove feature incluse)
        return ColumnTransformer(
            transformers=[],
            remainder="passthrough",
            verbose_feature_names_out=False
        )

def make_tree_pipeline(model):
    return Pipeline([
        ("feat_eng", feat_eng),
        ("prep", make_preprocessor(apply_scaling=False)),  # no scaling for trees
        ("model", model),
    ])

# ---------- Modelli (parametri ridotti, single-thread) ----------
pipelines = {
    "RandomForest": make_tree_pipeline(
        RandomForestRegressor(
            n_estimators=150,       # ridotto
            max_depth=None,
            n_jobs=1,               # single-thread
            random_state=42
        )
    ),
    "XGBoost": make_tree_pipeline(
        XGBRegressor(
            n_estimators=200,       # ridotto
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            max_bin=256,
            nthread=1,              # single-thread
            random_state=42
        )
    ),
    "LightGBM": make_tree_pipeline(
        LGBMRegressor(
            n_estimators=300,       # ridotto
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            max_bin=255,
            n_jobs=1,               # single-thread
            random_state=42
        )
    ),
}

# ---------- CV ----------
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}

results = {}
for name, pipe in pipelines.items():
    scores = cross_validate(
        pipe, X, y, cv=cv, scoring=scoring,
        n_jobs=1, return_train_score=False
    )
    results[name] = {
        "RMSE": -scores["test_rmse"].mean(),
        "MAE": -scores["test_mae"].mean(),
        "R2": scores["test_r2"].mean(),
    }
    del scores; gc.collect()

res_df = pd.DataFrame(results).T.sort_values(by="RMSE")
print("\nCV Results (3-fold, sample={}):".format(len(df)))
print(res_df)

best_name = res_df.index[0]
print(f"\nBest pipeline: {best_name}")



from sklearn.dummy import DummyRegressor

# -----------------------------
# Baseline Dummy Regressor
# -----------------------------
dummy = DummyRegressor(strategy="mean")
cv = KFold(n_splits=3, shuffle=True, random_state=42)

scores = cross_validate(
    dummy, X, y, cv=cv,
    scoring=scoring, n_jobs=1, return_train_score=False
)

dummy_results = {
    "RMSE": -scores["test_rmse"].mean(),
    "MAE": -scores["test_mae"].mean(),
    "R2": scores["test_r2"].mean(),
}

print("\nDummy Regressor baseline (predict mean BPM):")
print(dummy_results)



# Paths
OUTPUT_DIR = Path("/kaggle/working/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Baseline prediction = mean of train target
mean_bpm = train0["BeatsPerMinute"].mean()
print("Mean BPM from train:", mean_bpm)

# Create submission DataFrame
submission = pd.DataFrame({
    "id": test0["id"],
    "BeatsPerMinute": mean_bpm
})

# Save submission
submission_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"âœ… Submission saved to {submission_path} with shape {submission.shape}")


