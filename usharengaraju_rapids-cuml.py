
# ========================
# 0. Imports (RAPIDS + utils)
# ========================
# If you get import errors for cudf / cuml, install RAPIDS wheels appropriate
# for your CUDA version, e.g. (template only, may need changes):
# !pip install cudf-cu12 cuml-cu12 --extra-index-url https://pypi.nvidia.com -q

import os

import cudf
import cupy as cp  # used mostly for type checks and conversions
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from cuml.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="matplotlib.colors"
)



# ========================
# 1. Config: sampling size & CV
# ========================

# Use only a subset of the training data for faster experimentation
USE_SAMPLE = True

# Number of rows to sample from train (if USE_SAMPLE is True).
N_SAMPLE = 50_000

# Cross-validation setup
N_SPLITS = 3
RANDOM_STATE = 42




# ========================
# 2. Load data with cuDF
# ========================

DATA_DIR = "/kaggle/input/playground-series-s5e12"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = cudf.read_csv(train_path)
test = cudf.read_csv(test_path)
sample_sub = cudf.read_csv(sample_path)

print("Original train shape:", train.shape)
print("Test shape:", test.shape)

ID_COL = "id"
TARGET_COL = "diagnosed_diabetes"

assert TARGET_COL in train.columns, "Target column not found in train!"
assert ID_COL in train.columns, "ID column not found in train!"

# ---- Sample a subset of training data for faster experimentation ----
if USE_SAMPLE and len(train) > N_SAMPLE:
    train = train.sample(n=N_SAMPLE, random_state=RANDOM_STATE).reset_index(drop=True)
    print("\nUsing a sampled training set:")
else:
    print("\nUsing full training set (sampling disabled or train smaller than N_SAMPLE).")

print("Train shape actually used:", train.shape)



# ========================
# EDA & Visualizations
# ========================

# Convert GPU DataFrames → CPU pandas for plotting
train_cpu = train.to_pandas()

print("\n=== Basic Info ===")
print(train_cpu.info())



print("\n=== Missing Values ===")
print(train_cpu.isnull().sum())



print("\n=== Descriptive Statistics ===")
train_cpu.describe().T



plt.figure(figsize=(6,5))

# Better visual theme
sns.set_style("whitegrid")

# Modern color palette for binary classification
palette = ["#3b8eea", "#e64b3c"]

sns.countplot(
    x=train_cpu[TARGET_COL],
    palette=palette,
    edgecolor="black",
    linewidth=1.2
)

plt.title("Target Distribution: diagnosed_diabetes", fontsize=14, weight="bold")
plt.xlabel("Diagnosed Diabetes (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Count", fontsize=12)

# Annotate the bars with counts
for p in plt.gca().patches:
    plt.gca().annotate(
        format(p.get_height(), ','),
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center',
        xytext=(0, 10),
        textcoords='offset points',
        fontsize=11
    )

plt.tight_layout()
plt.show()



# -------------------------
# Correlation Heatmap
# -------------------------
# Nice minimal style
sns.set(style="white")

# 1. Keep only numeric columns
num_df = train_cpu.select_dtypes(include=[np.number]).copy()

# 2. Drop ID column if present
num_df = num_df.drop(columns=[ID_COL], errors="ignore")

# 3. Compute correlation and clean it
corr = num_df.corr()
corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0)
corr = corr.clip(-1, 1)   # just in case of float noise

# 4. Plot full matrix (no mask → no NaN → no warning)
plt.figure(figsize=(14, 12))

sns.heatmap(
    corr,
    cmap="Spectral",          # nice diverging palette
    vmin=-1, vmax=1, center=0,
    square=True,
    linewidths=0.4,
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Correlation"},
)

plt.title("Feature Correlation Heatmap", fontsize=16, weight="bold")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()



# -------------------------
# Numeric Feature Distributions (Histograms)
# -------------------------
import matplotlib.pyplot as plt
import numpy as np

# Identify numeric columns
num_cols = [
    c for c in train_cpu.columns
    if train_cpu[c].dtype != "object" 
    and c not in [ID_COL, TARGET_COL]
]

# Arrange grid size automatically
n_cols = 4
n_rows = int(np.ceil(len(num_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    ax.hist(train_cpu[col], bins=30, edgecolor="white", linewidth=0.7)
    
    # Title formatting
    ax.set_title(col, fontsize=12, fontweight="bold")
    
    # Clean style
    ax.grid(alpha=0.2)
    ax.set_facecolor("#f7f7f7")

# Turn off unused subplot axes
for ax in axes[len(num_cols):]:
    ax.axis("off")

plt.suptitle("Numeric Feature Distributions", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()




# ========================
# 3. Basic preprocessing & encoding
# ========================

# Separate target
y = train[TARGET_COL].astype("int32")
X = train.drop(columns=[TARGET_COL])
X_test = test.copy()

# Feature columns (everything except ID)
feature_cols = [c for c in X.columns if c != ID_COL]

print("Number of features:", len(feature_cols))

# Work only on feature subset for encoding
X_feat = X[feature_cols].copy()
X_test_feat = X_test[feature_cols].copy()

# Joint label-encoding for each feature (train + test together)
for col in feature_cols:
    combined = cudf.concat([X_feat[col], X_test_feat[col]], ignore_index=True)
    codes, uniques = combined.factorize()
    X_feat[col] = codes[:len(X_feat)]
    X_test_feat[col] = codes[len(X_feat):]

# Rebuild encoded frames with ID column preserved
X_encoded = cudf.concat([X[[ID_COL]], X_feat], axis=1)
X_test_encoded = cudf.concat([X_test[[ID_COL]], X_test_feat], axis=1)

# Explicit fill (factorize already sets NaN to -1, but we ensure)
for col in feature_cols:
    X_encoded[col] = X_encoded[col].fillna(-1)
    X_test_encoded[col] = X_test_encoded[col].fillna(-1)

X_train_gpu = X_encoded.drop(columns=[ID_COL])
X_test_gpu = X_test_encoded.drop(columns=[ID_COL])

print("Final train shape:", X_train_gpu.shape, " Final test shape:", X_test_gpu.shape)

# For sklearn split (CPU-side)
y_np = y.to_numpy()




# ========================
# 4. Cross-validation with cuML RandomForest (sampled data)
# ========================
# Store predictions as NumPy arrays to avoid CuPy indexing quirks.
# Handle predict_proba output as either 1D or 2D.

def to_pos_class_proba(pred_raw):
    """Convert cuML predict_proba output to a 1D NumPy array of positive-class probabilities.

    Handles:
      - CuPy arrays (1D or 2D)
      - cuDF Series / DataFrame
      - NumPy arrays
    """
    # CuPy array
    if isinstance(pred_raw, cp.ndarray):
        arr = cp.asnumpy(pred_raw)
    else:
        # cuDF or NumPy-like
        try:
            # cuDF Series/DataFrame usually has .to_numpy()
            arr = pred_raw.to_numpy()
        except AttributeError:
            arr = np.asarray(pred_raw)

    if arr.ndim == 1:
        return arr
    elif arr.ndim == 2:
        # assume last column is positive class
        return arr[:, -1]
    else:
        raise ValueError(f"Unexpected predict_proba output shape: {arr.shape}")


skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_pred = np.zeros(len(X_train_gpu), dtype=np.float32)
test_pred = np.zeros(len(X_test_gpu), dtype=np.float32)

fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_gpu.to_pandas(), y_np), 1):
    print(f"\n=== Fold {fold} / {N_SPLITS} ===")
    
    X_tr = X_train_gpu.iloc[tr_idx]
    X_val = X_train_gpu.iloc[val_idx]
    y_tr = y.iloc[tr_idx]
    y_val = y.iloc[val_idx]

    model = RandomForestClassifier(
        n_estimators=300,       # fewer trees for speed
        max_depth=12,
        max_features="sqrt",   # 'auto' is deprecated; 'sqrt' is standard RF behavior
        random_state=RANDOM_STATE,
        n_streams=1,
    )

    model.fit(X_tr, y_tr)

    # Validation probabilities
    val_proba_raw = model.predict_proba(X_val)
    val_proba_np = to_pos_class_proba(val_proba_raw).astype(np.float32)

    # Save OOF predictions
    oof_pred[val_idx] = val_proba_np

    val_auc = roc_auc_score(y_val.to_numpy(), val_proba_np)
    fold_scores.append(val_auc)
    print(f"Fold {fold} ROC AUC: {val_auc:.5f}")

    # Test predictions (average over folds)
    test_proba_raw = model.predict_proba(X_test_gpu)
    test_proba_np = to_pos_class_proba(test_proba_raw).astype(np.float32)
    test_pred += test_proba_np / N_SPLITS

# Overall OOF ROC AUC (on sampled train)
oof_auc = roc_auc_score(y_np, oof_pred)
print(f"\nOOF ROC AUC (sampled train): {oof_auc:.5f}")
print("Fold scores:", fold_scores, "Mean:", np.mean(fold_scores), "Std:", np.std(fold_scores))




# ========================
# 5. (Optional) Final model on sampled data
# ========================

train_final_model = False  # keep False for quickest runs

if train_final_model:
    final_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_streams=1,
    )

    final_model.fit(X_train_gpu, y)
    final_test_proba_raw = final_model.predict_proba(X_test_gpu)
    final_test_pred = to_pos_class_proba(final_test_proba_raw).astype(np.float32)
else:
    final_model = None
    final_test_pred = None




# ========================
# 6. Build submission
# ========================

# Choose which predictions to use:
# - `test_pred` -> CV-averaged predictions
# - `final_test_pred` -> full-data model predictions (if you set train_final_model=True)

pred_to_use = test_pred
if final_test_pred is not None:
    # Example blend: 50/50 between CV and final model
    pred_to_use = 0.5 * test_pred + 0.5 * final_test_pred

submission = sample_sub.copy()

print("Submission columns:", submission.columns)

# Ensure we write the probabilities into the target column
submission[TARGET_COL] = pred_to_use

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
submission.head()


