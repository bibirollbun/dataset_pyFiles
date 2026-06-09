# ===============================================================
# STEP 1 â€” Setup, Configuration, and Data Loading
# ===============================================================

# Standard libraries
import os
import gc
import sys
from datetime import datetime

# Data & ML libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV


# ===============================================================
# CONFIGURATION
# ===============================================================
CONFIG = {
    "seed": 42,
    "n_folds": 5,
    "target": "accident_risk",
    "id_col": "id",
    "train_path": "/kaggle/input/playground-series-s5e10/train.csv",
    "test_path":  "/kaggle/input/playground-series-s5e10/test.csv",
    "sub_path":   "/kaggle/input/playground-series-s5e10/sample_submission.csv",
    "use_stratified": False,  # toggle if target is categorical
}

# ===============================================================
# UTILITY FUNCTIONS
# ===============================================================
def log(msg: str):
    """Print timestamped log messages for cleaner outputs."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def set_seed(seed: int = 42):
    """Ensure reproducibility across NumPy, Python, and LightGBM."""
    np.random.seed(seed)
    log(f"Random seed set to {seed}")

def load_data(train_path: str, test_path: str, sub_path: str):
    """Load train, test, and submission CSV files."""
    log("Loading datasets ...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sub_path)

    log(f"âœ… Train shape: {train.shape}")
    log(f"âœ… Test shape : {test.shape}")
    log(f"âœ… Columns: {list(train.columns[:10])} ...")

    return train, test, sub

def show_versions():
    """Display key library versions for reproducibility."""
    log("Library versions:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"LightGBM: {lgb.__version__}")

# ===============================================================
# MAIN EXECUTION
# ===============================================================
set_seed(CONFIG["seed"])
show_versions()

# Load data
train, test, sub = load_data(
    CONFIG["train_path"],
    CONFIG["test_path"],
    CONFIG["sub_path"]
)

# Basic cleanup to free memory early
gc.collect()


# ===============================================================
# STEP 2 â€” Exploratory Data Analysis (EDA)
# ===============================================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)
TARGET = CONFIG["target"]
ID_COL = CONFIG["id_col"]


log("Starting EDA ...")

# ---------------------------------------------------------------
# 2.1: Basic dataset overview
# ---------------------------------------------------------------
log("Basic dataset info:")
display(train.describe(include="all").T.head(10))

log("Checking missing values ...")
missing = train.isna().mean().sort_values(ascending=False)
display(missing[missing > 0].head(10))

# ---------------------------------------------------------------
# 2ï¸�.2: Target distribution
# ---------------------------------------------------------------
log("Target variable distribution:")
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(train[TARGET], kde=True, bins=30, color="skyblue", ax=ax)
ax.set_title("Distribution of Accident Risk", fontsize=13)
plt.show()

print(f"Target mean: {train[TARGET].mean():.4f}")
print(f"Target median: {train[TARGET].median():.4f}")
print(f"Target std: {train[TARGET].std():.4f}")

# ---------------------------------------------------------------
# 2.3ï¸�: Feature overview
# ---------------------------------------------------------------
features = [c for c in train.columns if c not in [ID_COL, TARGET]]
cat_cols = [c for c in features if train[c].dtype == "object" or train[c].nunique() < 50]
num_cols = [c for c in features if c not in cat_cols]

log(f"Detected {len(cat_cols)} categorical and {len(num_cols)} numerical features.")
log(f"Example categorical features: {cat_cols[:5]}")
log(f"Example numerical features: {num_cols[:5]}")

# ---------------------------------------------------------------
# 2.4ï¸�: Categorical feature summary (top few)
# ---------------------------------------------------------------
for c in cat_cols[:3]:  # limit to first 3 for quick inspection
    plt.figure(figsize=(6,3))
    sns.countplot(data=train, x=c, hue=None)
    plt.title(f"{c} distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# 2.5ï¸�: Numerical feature distributions (top few)
# ---------------------------------------------------------------
for c in num_cols[:3]:
    plt.figure(figsize=(6,3))
    sns.histplot(train[c], bins=30, color="teal", kde=True)
    plt.title(f"{c} distribution")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# 2.6: Smart Correlation Heatmap (handles categorical + numeric)
# ---------------------------------------------------------------

# --- Safety check: ensure `encoded` exists ---
if 'encoded' not in locals():
    # If not yet created, use training data + target
    encoded = train.copy()
    # Encode categoricals numerically for correlation
    for c in encoded.select_dtypes(include='object').columns:
        encoded[c] = encoded[c].astype('category').cat.codes
    log("âš™ï¸� Created fallback 'encoded' DataFrame for correlation heatmap.")

# --- Compute correlation matrix including target ---
corr = encoded.corr()

# --- Clean invalid and constant data ---
corr = corr.replace([np.inf, -np.inf], np.nan)  # remove inf
corr = corr.fillna(0)  # replace NaN with 0

# Drop constant columns (no variance)
constant_cols = corr.columns[corr.nunique() <= 1]
if len(constant_cols) > 0:
    log(f"Dropping constant columns from correlation matrix: {list(constant_cols)}")
    corr = corr.drop(columns=constant_cols, errors="ignore")
    corr = corr.drop(index=constant_cols, errors="ignore")

# --- Ensure we have at least one valid column to plot ---
if corr.shape[0] > 1:
    plt.figure(figsize=(12, 10))

    # Safe normalization: compute vmin/vmax explicitly
    vmin, vmax = np.nanmin(corr.values), np.nanmax(corr.values)
    if vmin == vmax:  # avoid division by zero in colormap
        vmin, vmax = -1, 1

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap (Numeric + Encoded Categorical)", fontsize=14)
    plt.tight_layout()
    plt.show()

    log(f"âœ… Correlation heatmap plotted successfully. Shape: {corr.shape}")
else:
    log("âš ï¸� Skipping heatmap: not enough valid columns after cleaning.")



# ---------------------------------------------------------------
# 2.7: Target Relationship Analysis
# ---------------------------------------------------------------
import scipy.stats as ss
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

log("Analyzing featureâ€“target relationships ...")

TARGET = CONFIG["target"]

# --- 1. Numeric feature relationships ---
num_features = train.select_dtypes(include=[np.number]).columns.drop(TARGET, errors="ignore")

if len(num_features) > 0:
    correlations = {}
    for col in num_features:
        if train[col].nunique() > 1:
            corr_val = train[[col, TARGET]].corr().iloc[0, 1]
            correlations[col] = corr_val

    corr_df = (
        pd.Series(correlations)
        .dropna()
        .sort_values(key=lambda x: abs(x), ascending=False)
        .head(10)
    )

    log("Top 10 numeric features correlated with target:")
    display(corr_df)

    # Plot the top correlated numeric features
    for col in corr_df.index:
        plt.figure(figsize=(6, 3))
        sns.scatterplot(x=train[col], y=train[TARGET], alpha=0.4)
        plt.title(f"{col} vs {TARGET} (corr={corr_df[col]:.2f})")
        plt.tight_layout()
        plt.show()
else:
    log("No numeric features found for target correlation analysis.")

# --- 2. Categorical feature relationships ---
cat_features = train.select_dtypes(include=["object", "category"]).columns

if len(cat_features) > 0:
    for col in cat_features[:5]:  # limit to first 5 for clarity
        plt.figure(figsize=(6, 3))
        sns.barplot(
            data=train,
            x=col,
            y=TARGET,
            estimator=np.mean,
            errorbar=None,
        )
        plt.title(f"Mean {TARGET} by {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
else:
    log("No categorical features found for target mean analysis.")



# ===============================================================
# STEP 3 â€” FEATURE ENGINEERING & PREPROCESSING
# ===============================================================

log("Feature Engineering & Preprocessing...")

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 3.1 Separate target and features
# ---------------------------------------------------------------
TARGET = CONFIG["target"]

X = train.drop(columns=[TARGET])
y = train[TARGET]
X_test = test.copy()

# ---------------------------------------------------------------
# 3.2 Identify numeric and categorical columns
# ---------------------------------------------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

log(f"Identified {len(num_cols)} numeric and {len(cat_cols)} categorical columns.")

# ---------------------------------------------------------------
# 3ï¸�.3: Rare category grouping (reduces noise & improves generalization)
# ---------------------------------------------------------------
def group_rare_categories(df, cols, threshold=0.01):
    """
    Replace rare categories (below threshold proportion) with 'Other'.
    """
    for c in cols:
        freqs = df[c].value_counts(normalize=True)
        rare_labels = freqs[freqs < threshold].index
        df[c] = df[c].replace(rare_labels, "Other")
    return df

X = group_rare_categories(X, cat_cols)
X_test = group_rare_categories(X_test, cat_cols)

# ---------------------------------------------------------------
# 3.4: Temporal / cyclical feature extraction (if applicable)
# ---------------------------------------------------------------
def extract_time_features(df, time_col):
    """
    Extract time-based features from a datetime column.
    Automatically adds cyclical encodings for hour and month.
    """
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    
    # Cyclical encoding
    for col, max_val in [("hour", 24), ("month", 12)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df

# Example usage (uncomment if time column exists)
# X = extract_time_features(X, "date_time")
# X_test = extract_time_features(X_test, "date_time")

# ---------------------------------------------------------------
# 3.5ï¸�: Handle missing values + scaling + encoding via pipeline
# ---------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# ---------------------------------------------------------------
# 3.6ï¸�: Optional: create domain-specific interaction features
# ---------------------------------------------------------------
if "vehicle_speed" in X.columns and "visibility" in X.columns:
    X["speed_visibility_ratio"] = X["vehicle_speed"] / (X["visibility"] + 1)
    X_test["speed_visibility_ratio"] = X_test["vehicle_speed"] / (X_test["visibility"] + 1)
    num_cols.append("speed_visibility_ratio")

log("Feature engineering complete. Proceeding to preprocessing...")

# ---------------------------------------------------------------
# 3.7ï¸�: Fit-transform the preprocessor (train â†’ fit, test â†’ transform)
# ---------------------------------------------------------------
X_preprocessed = preprocessor.fit_transform(X)
X_test_preprocessed = preprocessor.transform(X_test)

log(f"âœ… Preprocessing complete. "
    f"Train shape: {X_preprocessed.shape}, Test shape: {X_test_preprocessed.shape}")

# ---------------------------------------------------------------
# 3.8ï¸�: Optional: get feature names (useful for model interpretation)
# ---------------------------------------------------------------
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names = [f"feature_{i}" for i in range(X_preprocessed.shape[1])]

log(f"Total engineered features: {len(feature_names)}")

# ---------------------------------------------------------------
# 3.9ï¸�: Store for next steps
# ---------------------------------------------------------------
train_processed = pd.DataFrame(X_preprocessed, columns=feature_names)
test_processed = pd.DataFrame(X_test_preprocessed, columns=feature_names)
train_processed[TARGET] = y.reset_index(drop=True)

log("Data ready for model training.")



# ===============================================================
# VISUALIZATION for Step-3 (Feature Engineering)
# ===============================================================

log("Visualizing engineered feature distributions...")

# Numeric feature distributions
sampled_train = train_processed.sample(10000, random_state=42) if len(train_processed) > 10000 else train_processed
numeric_features = [col for col in sampled_train.columns if np.issubdtype(sampled_train[col].dtype, np.number)][:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_features):
    sns.histplot(sampled_train[col], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f"Distribution: {col}")
plt.tight_layout()
plt.show()

# Correlation heatmap
log("Plotting correlation heatmap...")
corr = sampled_train[numeric_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Sampled Features)")
plt.show()

# Categorical feature balance (if any)
cat_features = [c for c in sampled_train.columns if sampled_train[c].dtype == 'object'][:4]
if cat_features:
    fig, axes = plt.subplots(1, len(cat_features), figsize=(5 * len(cat_features), 4))
    axes = np.atleast_1d(axes)
    for i, col in enumerate(cat_features):
        sns.countplot(y=col, data=sampled_train, ax=axes[i])
        axes[i].set_title(f"Category Counts: {col}")
    plt.tight_layout()
    plt.show()



# ===============================================================
# STEP 4 â€” MODEL TRAINING & VALIDATION (REGRESSION VERSION)
# ===============================================================

log("Starting Model Training & Validation (Regression)...")

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------
# 4.1: Configuration
# ---------------------------------------------------------------
N_SPLITS = CONFIG.get("cv_folds", 5)
SEED = CONFIG.get("seed", 42)
TARGET = CONFIG["target"]

# Extract preprocessed data
X = train_processed.drop(columns=[TARGET])
y = train_processed[TARGET]
X_test = test_processed.copy()

# ---------------------------------------------------------------
# 4.2: Cross-validation setup
# ---------------------------------------------------------------
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_metrics = []

# ---------------------------------------------------------------
# 4.3: LightGBM parameters for regression
# ---------------------------------------------------------------
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": SEED,
}

# ---------------------------------------------------------------
# 4.4: Training loop
# ---------------------------------------------------------------
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    log(f"ğŸš€ Training Fold {fold}/{N_SPLITS} ...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=2000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200)
    ],
)

    # Validation predictions
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred

    # Test predictions (averaged)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS

    # Evaluation metrics
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    r2 = r2_score(y_val, val_pred)

    fold_metrics.append({
        "fold": fold,
        "RMSE": rmse,
        "R2": r2,
    })
    log(f"Fold {fold} â€” RMSE: {rmse:.4f}, RÂ²: {r2:.4f}")

# ---------------------------------------------------------------
# 4.5: Cross-validation summary
# ---------------------------------------------------------------
metrics_df = pd.DataFrame(fold_metrics)
log("Cross-validation results:")
display(metrics_df)

mean_rmse = metrics_df["RMSE"].mean()
mean_r2 = metrics_df["R2"].mean()

log(f"âœ… CV Summary â€” Mean RMSE: {mean_rmse:.4f}, Mean RÂ²: {mean_r2:.4f}")

# ---------------------------------------------------------------
# 4.6: OOF performance evaluation
# ---------------------------------------------------------------
oof_rmse = mean_squared_error(y, oof_preds, squared=False)
oof_r2 = r2_score(y, oof_preds)
log(f"Overall OOF RMSE: {oof_rmse:.4f}, Overall OOF RÂ²: {oof_r2:.4f}")

# ---------------------------------------------------------------
# 4.7: Feature importance visualization
# ---------------------------------------------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importance(),
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(25), x="Importance", y="Feature")
plt.title("Top 25 Feature Importances (LightGBM Regression)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 4.8: Save predictions for submission
# ---------------------------------------------------------------
submission = pd.DataFrame({
    "id": test.index,
    TARGET: test_preds
})
submission.to_csv("submission.csv", index=False)
log("Saved submission.csv successfully!")

log("âœ… Regression model training and evaluation finished.")



# ===============================================================
# MODEL DIAGNOSTICS & VISUALIZATION (REGRESSION)
# ===============================================================

log("Visualizing Regression Performance...")

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Overall performance summary
oof_rmse = mean_squared_error(y, oof_preds, squared=False)
oof_r2 = r2_score(y, oof_preds)
log(f"Overall OOF RMSE: {oof_rmse:.4f}, Overall OOF RÂ²: {oof_r2:.4f}")

# ---------------------------------------------------------------
# CV metric variance across folds
# ---------------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.barplot(x="fold", y="RMSE", data=metrics_df)
plt.title("Cross-Validation RMSE per Fold")
plt.ylabel("RMSE")
plt.xlabel("Fold")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.barplot(x="fold", y="R2", data=metrics_df)
plt.title("Cross-Validation RÂ² per Fold")
plt.ylabel("RÂ²")
plt.xlabel("Fold")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# True vs Predicted scatter (OOF)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y, y=oof_preds, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title("True vs Predicted (OOF)")
plt.xlabel("True Target")
plt.ylabel("Predicted Target")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------
residuals = y - oof_preds

# Residual distribution
plt.figure(figsize=(7, 4))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution (OOF)")
plt.xlabel("Residual (True - Predicted)")
plt.tight_layout()
plt.show()

# Residuals vs Predicted (to check heteroscedasticity)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=oof_preds, y=residuals, alpha=0.4)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Target")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

log("âœ… Regression performance visualizations generated.")


# ===============================================================
# MODEL INTERPRETABILITY WITH SHAP (Shapely Additive Explanation)
# ===============================================================

log("Interpreting model with SHAP values...")

# Install and import SHAP
try:
    import shap
except ImportError:
    !pip install shap -q
    import shap

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Prepare explainer on a sample
# ---------------------------------------------------------------
# For speed, use a representative subset
SAMPLE_SIZE = min(2000, len(X))
X_sample = X.sample(SAMPLE_SIZE, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

log(f"Computed SHAP values for {SAMPLE_SIZE} samples and {X_sample.shape[1]} features.")

# ---------------------------------------------------------------
# Global Feature Importance (SHAP summary)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("Global Feature Importance (mean |SHAP|)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Detailed Feature Impact (SHAP beeswarm)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("Feature Impact and Direction on Predictions")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Feature Dependence Plots (top features)
# ---------------------------------------------------------------
top_feats = feature_importance["Feature"].head(3).tolist()
log(f"Generating SHAP dependence plots for top features: {top_feats}")

for feat in top_feats:
    shap.dependence_plot(feat, shap_values, X_sample, show=False)
    plt.title(f"SHAP Dependence: {feat}")
    plt.tight_layout()
    plt.show()

log("âœ… SHAP interpretability visualizations generated.")



# ===============================================================
# STEP 5 â€” MODEL TRAINING & VALIDATION (XGBOOST REGRESSION)
# ===============================================================

log("Model Training & Validation (XGBoost Regression)...")

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from packaging import version

log(f"Using XGBoost version: {xgb.__version__}")

# ---------------------------------------------------------------
# 5.1: Configuration
# ---------------------------------------------------------------
N_SPLITS = CONFIG.get("cv_folds", 5)
SEED = CONFIG.get("seed", 42)
TARGET = CONFIG["target"]

X = train_processed.drop(columns=[TARGET])
y = train_processed[TARGET]
X_test = test_processed.copy()

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds_xgb = np.zeros(len(X))
test_preds_xgb = np.zeros(len(X_test))
fold_metrics_xgb = []

# ---------------------------------------------------------------
# 5.2: XGBoost parameters (regression)
# ---------------------------------------------------------------
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": SEED,
    "tree_method": "hist",   # Faster on CPUs
}

# ---------------------------------------------------------------
# 5.3: Training loop with CV
# ---------------------------------------------------------------
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    log(f"ğŸš€ Training Fold {fold}/{N_SPLITS} ...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    evals = [(dtrain, "train"), (dval, "valid")]

    model_xgb = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=200
    )

    # Predictions
    val_pred = model_xgb.predict(dval, iteration_range=(0, model_xgb.best_iteration + 1))
    oof_preds_xgb[val_idx] = val_pred
    test_preds_xgb += model_xgb.predict(dtest, iteration_range=(0, model_xgb.best_iteration + 1)) / N_SPLITS

    # Metrics
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    r2 = r2_score(y_val, val_pred)
    fold_metrics_xgb.append({"fold": fold, "RMSE": rmse, "R2": r2})

    log(f"Fold {fold} â€” RMSE: {rmse:.4f}, RÂ²: {r2:.4f}")

# ---------------------------------------------------------------
# 5.4: Cross-validation summary
# ---------------------------------------------------------------
metrics_df_xgb = pd.DataFrame(fold_metrics_xgb)
log("Cross-validation results (XGBoost):")
display(metrics_df_xgb)

mean_rmse_xgb = metrics_df_xgb["RMSE"].mean()
mean_r2_xgb = metrics_df_xgb["R2"].mean()
log(f"âœ… CV Summary â€” Mean RMSE: {mean_rmse_xgb:.4f}, Mean RÂ²: {mean_r2_xgb:.4f}")

# ---------------------------------------------------------------
# 5.5: Feature importance
# ---------------------------------------------------------------
xgb_importance = model_xgb.get_score(importance_type="weight")
xgb_importance_df = (
    pd.DataFrame.from_dict(xgb_importance, orient="index", columns=["Importance"])
    .reset_index()
    .rename(columns={"index": "Feature"})
    .sort_values(by="Importance", ascending=False)
)

plt.figure(figsize=(10, 8))
sns.barplot(data=xgb_importance_df.head(25), x="Importance", y="Feature", palette="mako")
plt.title("Top 25 Feature Importances (XGBoost Regression)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 5.6: Save submission
# ---------------------------------------------------------------
submission_xgb = pd.DataFrame({
    "id": test.index,
    TARGET: test_preds_xgb
})
submission_xgb.to_csv("submission_xgb.csv", index=False)
log("Saved submission_xgb.csv successfully!")

log("âœ… XGBoost regression training and evaluation finished.")



# ===============================================================
# MODEL DIAGNOSTICS & VISUALIZATION (XGBOOST)
# ===============================================================

log("Visualizing XGBoost Regression Performance...")

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Overall OOF performance
oof_rmse_xgb = mean_squared_error(y, oof_preds_xgb, squared=False)
oof_r2_xgb = r2_score(y, oof_preds_xgb)
log(f"Overall XGBoost OOF RMSE: {oof_rmse_xgb:.4f}, Overall RÂ²: {oof_r2_xgb:.4f}")

# ---------------------------------------------------------------
# CV metric variance across folds
# ---------------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.barplot(x="fold", y="RMSE", data=metrics_df_xgb)
plt.title("XGBoost Cross-Validation RMSE per Fold")
plt.ylabel("RMSE")
plt.xlabel("Fold")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.barplot(x="fold", y="R2", data=metrics_df_xgb)
plt.title("XGBoost Cross-Validation RÂ² per Fold")
plt.ylabel("RÂ²")
plt.xlabel("Fold")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# True vs Predicted (OOF)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y, y=oof_preds_xgb, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title("True vs Predicted (OOF) â€” XGBoost")
plt.xlabel("True Target")
plt.ylabel("Predicted Target")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------
residuals_xgb = y - oof_preds_xgb

plt.figure(figsize=(7, 4))
sns.histplot(residuals_xgb, kde=True)
plt.title("Residual Distribution (OOF) â€” XGBoost")
plt.xlabel("Residual (True - Predicted)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(x=oof_preds_xgb, y=residuals_xgb, alpha=0.4)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Predicted Values â€” XGBoost")
plt.xlabel("Predicted Target")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Feature importance visualization (Top 20)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 7))
sns.barplot(data=xgb_importance_df.head(20), x="Importance", y="Feature", palette="crest")
plt.title("Top 20 Feature Importances (XGBoost Regression)")
plt.tight_layout()
plt.show()

log("âœ… XGBoost regression visualizations generated.")


print("          MODEL COMPARISON        ")
print(f"LightGBM  | Mean RMSE: {mean_rmse:.4f} | RÂ²: {mean_r2:.4f}")
print(f"XGBoost   | Mean RMSE: {mean_rmse_xgb:.4f} | RÂ²: {mean_r2_xgb:.4f}")
print(f"LightGBM  | OOF RMSE:  {oof_rmse:.4f} | OOF RÂ²: {oof_r2:.4f}")
print(f"XGBoost   | OOF RMSE:  {oof_rmse_xgb:.4f} | OOF RÂ²: {oof_r2_xgb:.4f}")


# ===============================================================
# MODEL INTERPRETABILITY WITH SHAP (XGBOOST)
# ===============================================================

log("Interpreting XGBoost Regression Model with SHAP...")

# 1ï¸�âƒ£ Import / install SHAP
try:
    import shap
except ImportError:
    !pip install shap -q
    import shap

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------
# 2ï¸�âƒ£ Prepare a sample for explanation
# ---------------------------------------------------------------
SAMPLE_SIZE = min(2000, len(X))
X_sample = X.sample(SAMPLE_SIZE, random_state=42)

# Create DMatrix for XGBoost sample
dX_sample = xgb.DMatrix(X_sample)

# ---------------------------------------------------------------
# 3ï¸�âƒ£ Initialize SHAP explainer
# ---------------------------------------------------------------
explainer_xgb = shap.TreeExplainer(model_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_sample)

log(f"Computed SHAP values for {SAMPLE_SIZE} samples and {X_sample.shape[1]} features.")

# ---------------------------------------------------------------
# 4ï¸�âƒ£ Global importance (mean |SHAP|)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb, X_sample, plot_type="bar", show=False)
plt.title("XGBoost â€” Global Feature Importance (mean |SHAP|)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 5ï¸�âƒ£ Beeswarm plot (feature impact and direction)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_xgb, X_sample, show=False)
plt.title("XGBoost â€” Feature Impact and Direction on Predictions")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 6ï¸�âƒ£ Dependence plots (top correlated features)
# ---------------------------------------------------------------
top_feats_xgb = xgb_importance_df["Feature"].head(3).tolist()
log(f"Generating SHAP dependence plots for top features: {top_feats_xgb}")

for feat in top_feats_xgb:
    shap.dependence_plot(feat, shap_values_xgb, X_sample, show=False)
    plt.title(f"XGBoost â€” SHAP Dependence: {feat}")
    plt.tight_layout()
    plt.show()

log("âœ… SHAP interpretability visualizations generated for XGBoost.")



# ===============================================================
# STEP 6 â€” MODEL COMPARISON & ENSEMBLE
# ===============================================================

log("Comparing models and building ensemble...")

# ---------------------------------------------------------------
# 6.1: Model performance summary table
# ---------------------------------------------------------------
comparison_df = pd.DataFrame({
    "Model": ["LightGBM", "XGBoost"],
    "Mean RMSE": [mean_rmse, mean_rmse_xgb],
    "OOF RMSE": [oof_rmse, oof_rmse_xgb],
    "Mean RÂ²": [mean_r2, mean_r2_xgb],
    "OOF RÂ²": [oof_r2, oof_r2_xgb]
}).sort_values("OOF RMSE")

display(comparison_df)

plt.figure(figsize=(6, 4))
sns.barplot(data=comparison_df.melt(id_vars="Model", value_vars=["OOF RMSE", "OOF RÂ²"]),
            x="Model", y="value", hue="variable", palette="viridis")
plt.title("Model Comparison: LightGBM vs XGBoost")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 6.2: Ensemble prediction (simple average)
# ---------------------------------------------------------------
log("Creating simple averaged ensemble...")

ensemble_oof = (oof_preds + oof_preds_xgb) / 2
ensemble_test = (test_preds + test_preds_xgb) / 2

ensemble_rmse = mean_squared_error(y, ensemble_oof, squared=False)
ensemble_r2 = r2_score(y, ensemble_oof)

log(f"âœ… Ensemble OOF RMSE: {ensemble_rmse:.4f}, RÂ²: {ensemble_r2:.4f}")

# ---------------------------------------------------------------
# 6.3: Weighted ensemble (optional tuning)
# ---------------------------------------------------------------
# You can manually adjust these weights if one model performs better
w_lgb = 0.6
w_xgb = 0.4

weighted_oof = w_lgb * oof_preds + w_xgb * oof_preds_xgb
weighted_test = w_lgb * test_preds + w_xgb * test_preds_xgb

weighted_rmse = mean_squared_error(y, weighted_oof, squared=False)
weighted_r2 = r2_score(y, weighted_oof)

log(f"âœ… Weighted Ensemble OOF RMSE: {weighted_rmse:.4f}, RÂ²: {weighted_r2:.4f}")

# ---------------------------------------------------------------
# 6.4: True vs Predicted (Ensemble)
# ---------------------------------------------------------------
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y, y=ensemble_oof, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title("True vs Predicted (OOF) â€” Ensemble Model")
plt.xlabel("True Target")
plt.ylabel("Predicted Target")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 6.5: Save ensemble submission
# ---------------------------------------------------------------
submission_ensemble = pd.DataFrame({
    "id": test.index,
    TARGET: ensemble_test
})
submission_ensemble.to_csv("submission_ensemble.csv", index=False)

log("âœ… Saved submission_ensemble.csv successfully!")
log("ğŸ�¯ Ensemble built and evaluated.")



# ===============================================================
# FINAL: Submission file (submission.csv)
# ===============================================================
import numpy as np
import pandas as pd
import zipfile
import datetime
from pathlib import Path

# Config / safety defaults (adjust if your notebook uses different names)
ID_COL = CONFIG.get("id_col", "id") if "CONFIG" in globals() else "id"
TARGET = CONFIG.get("target", "accident_risk") if "CONFIG" in globals() else "accident_risk"

# Output folder and timestamp
output_dir = Path("submissions")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Validate that 'test' DataFrame exists ---
assert "test" in globals(), "ERROR: 'test' DataFrame not found in the environment."

# --- Ensure prediction arrays exist; try common names ---
# Adjust these names if you used different variables earlier
pred_vars = {
    "lgb": globals().get("test_preds"),
    "xgb": globals().get("test_preds_xgb"),
    "ensemble": globals().get("ensemble_test")
}

# Quick check: at least one prediction array must exist
if all(v is None for v in pred_vars.values()):
    raise RuntimeError("No prediction arrays found (test_preds / test_preds_xgb / ensemble_test).")

# If any prediction array is None, skip saving that file
# Ensure predictions are numpy arrays and same length as test
n_test = len(test)
for k, v in list(pred_vars.items()):
    if v is None:
        pred_vars.pop(k)
        continue
    pred_arr = np.asarray(v).reshape(-1)
    if pred_arr.shape[0] != n_test:
        raise ValueError(f"Length mismatch for {k} predictions: {pred_arr.shape[0]} vs test rows {n_test}")
    pred_vars[k] = pred_arr  # store normalized array

# --- Clip predictions into [0,1] as required by the competition ---
for k in list(pred_vars.keys()):
    pred_vars[k] = np.clip(pred_vars[k], 0.0, 1.0)

# --- Build DataFrames using the real test ID column (not test.index) ---
lgb_path = xgb_path = ensemble_path = None

if "lgb" in pred_vars:
    df_lgb = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: pred_vars["lgb"]})
    lgb_path = output_dir / f"submission_lgb_{timestamp}.csv"
    df_lgb.to_csv(lgb_path, index=False)

if "xgb" in pred_vars:
    df_xgb = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: pred_vars["xgb"]})
    xgb_path = output_dir / f"submission_xgb_{timestamp}.csv"
    df_xgb.to_csv(xgb_path, index=False)

if "ensemble" in pred_vars:
    df_ens = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: pred_vars["ensemble"]})
    ensemble_path = output_dir / f"submission_ensemble_{timestamp}.csv"
    df_ens.to_csv(ensemble_path, index=False)

# --- Choose which file to save as final 'submission.csv' ---
# Prefer ensemble if available, else pick the best by OOF RMSE if variables exist, otherwise pick any existing prediction.
final_submission_path = output_dir / "submission.csv"

chosen_path = None
if ensemble_path is not None:
    chosen_path = ensemble_path
else:
    # fallback: try to choose best model by OOF RMSE if those variables exist
    try:
        scores = {
            "lgb": float(oof_rmse) if "oof_rmse" in globals() else None,
            "xgb": float(oof_rmse_xgb) if "oof_rmse_xgb" in globals() else None
        }
    except Exception:
        scores = {}
    # pick available preds with lowest score
    available = [k for k in ("lgb", "xgb") if k in pred_vars]
    if scores and any(scores.get(k) is not None for k in available):
        best = min((k for k in available if scores.get(k) is not None), key=lambda x: scores[x])
        chosen_path = lgb_path if best == "lgb" else xgb_path
    else:
        # final fallback: pick the first produced file
        for p in (ensemble_path, lgb_path, xgb_path):
            if p is not None:
                chosen_path = p
                break

if chosen_path is None:
    raise RuntimeError("No submission file was produced. Check prediction variables.")

# --- Final validation before saving ---
best_sub = pd.read_csv(chosen_path)

# 1) IDs must match test IDs
if set(best_sub[ID_COL]) != set(test[ID_COL]):
    missing = set(test[ID_COL]) - set(best_sub[ID_COL])
    extra = set(best_sub[ID_COL]) - set(test[ID_COL])
    raise ValueError(
        "ID mismatch detected.\n"
        f"Missing sample (first 10): {list(missing)[:10]}\n"
        f"Extra sample (first 10): {list(extra)[:10]}"
    )

# 2) Predictions must be floats in [0,1]
if not np.issubdtype(best_sub[TARGET].dtype, np.number):
    raise TypeError(f"{TARGET} column must be numeric.")
if not best_sub[TARGET].between(0, 1).all():
    # Clip and warn
    best_sub[TARGET] = best_sub[TARGET].clip(0.0, 1.0)

# Save final submission.csv
best_sub.to_csv(final_submission_path, index=False)
print(f"Saved final submission to: {final_submission_path}")

# Optional: zip everything for record keeping
zip_path = output_dir / f"all_submissions_{timestamp}.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for p in (lgb_path, xgb_path, ensemble_path, final_submission_path):
        if p is not None:
            zipf.write(p, arcname=p.name)
print(f"Archived submissions to: {zip_path}")

# Quick printout for sanity
print("\nSample of final submission:")
display(best_sub.head())
print("\nSubmission shape:", best_sub.shape)


