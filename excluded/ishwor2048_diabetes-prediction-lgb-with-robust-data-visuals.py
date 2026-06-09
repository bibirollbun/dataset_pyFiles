# --------------------------
# Basic Libraries
# --------------------------
import numpy as np
import pandas as pd
import random
import os

# ------------------------
# Data visualization libraries
# ------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------
# Scikit-learn for machine learning models, CV, evaluation and preprocessing
# --------------------------------------
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

import lightgbm as lgb


# Setting the  seed value
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

set_seed()


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv") # Training dataset to train the model
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv") # Test dataset in order to test model's performance on unseen data


# Check rows and columns of my dataset
print(f"There are {train.shape[0]} number of rows and {train.shape[1]} number of columns in the training dataset")


# Check rows and columns of the test dataset
test.shape


# Making sure that all the rows and column and cell values are visible without any truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Looking at a first 5 rows of the training dataset
train.head()


# Looking at random few rows of the training and test dataset
train.sample(10)


test.sample(10) # this will shuffle the dataset at row level, and will pull random 10 rows


train.describe(exclude="object") # checking statistical description of the training dataset, numerical columns


train.describe(include="object") # checking statistical description of object columns in the training data


test.describe().T # Checking statistical description of the numerical columns of the test dataset, in transpose format (rows to columns and vis-a-versa)


test.describe(include="object").T # Statistical description of the categorical columns of the test datasets


train.info() # Looking at the data information, rows, columns, data types, missing rows, etc.


train.ethnicity.value_counts() # counting number of records for each category of ethnicity column


train.employment_status.value_counts() # counting number of rows for each category of employment status


test.info() # checking data information for the test set, including rows, columns, data types of each columns, missing rows, etc.


# visualizing the target data distribution
plt.figure(figsize=(5,4))
sns.countplot(data=train, x=train.diagnosed_diabetes)
plt.title("Target Distribution (Diagnosed Diabetes)")
plt.ylabel("Count")
plt.grid(axis="y", alpha=0.3)
plt.show()

print(train["diagnosed_diabetes"].value_counts(normalize=True))


num_features = [
    "age",
    "bmi",
    "physical_activity_minutes_per_week",
    "cholesterol_total",
    "triglycerides"
]

for col in num_features:
    plt.figure(figsize=(6,4))
    sns.histplot(
        data=train,
        x=col,
        hue=train.diagnosed_diabetes,
        bins=40,
        kde=True,
        stat="density",
        common_norm=False
    )
    plt.title(f"Distribution of {col} by Diabetes Status")
    plt.grid(alpha=0.3)
    plt.show()


box_cols = [
    "age",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp"
]

for col in box_cols:
    plt.figure(figsize=(5,4))
    sns.boxplot(data=train, x=train.diagnosed_diabetes, y=col)
    plt.title(f"{col} vs Diabetes")
    plt.grid(alpha=0.3)
    plt.show()



plt.figure(figsize=(6,5))
sns.scatterplot(
    data=train,
    x="bmi",
    y="waist_to_hip_ratio",
    hue="diagnosed_diabetes",
    alpha=0.4
)
plt.title("BMI vs Waist-to-Hip Ratio")
plt.grid(alpha=0.3)
plt.show()


corr_features = [
    "age", "bmi", "waist_to_hip_ratio",
    "systolic_bp", "diastolic_bp",
    "cholesterol_total", "hdl_cholesterol",
    "ldl_cholesterol", "triglycerides",
    "diagnosed_diabetes"
]

plt.figure(figsize=(9,7))
sns.heatmap(
    train[corr_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap (Key Features)")
plt.show()



TARGET = "diagnosed_diabetes" # based on the problem given, choosing diagnosed_diabetes as target variable
ID_COL = "id" # ID column would not have much value on the training and prediction quality, flagging it to exclude

# training features
features = [c for c in train.columns if c not in [TARGET, ID_COL]]

X = train[features].copy()
y = train[TARGET].copy()

X_test = test[features].copy()


print(f"Train: {X.shape}, Test: {X_test.shape}")
print(f"Target Rate: {y.mean()}")


cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()


print(f"Categorical Columns: {cat_cols}")
print("---------------------------------------------------------")
print(f"Numerical Columns: {num_cols}")


# Function to create domain specific features that add value to the ROC
def add_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bp_ratio"] = df["systolic_bp"] / (df["diastolic_bp"] + 1.0)
    df["chol_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1.0)
    return df


# Add new features to training (X) and test (X_test) features
X = add_safe_features(X)
X_test = add_safe_features(X_test)


# updating categorical and numerical columns after new features assignment
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()


for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")


def drift_auc_for_feature(train_col: pd.Series, test_col: pd.Series, seed=SEED) -> float:
    # Build a dataset to classify source: 0=train, 1=test
    tr = train_col.copy()
    te = test_col.copy()

    df = pd.concat([tr, te], axis=0).reset_index(drop=True)
    src = np.array([0]*len(tr) + [1]*len(te))

    # If categorical, use codes (safe drift proxy)
    if str(df.dtype) == "category" or df.dtype == "object":
        df = df.astype("category").cat.codes.replace(-1, np.nan).astype(float)
    else:
        df = pd.to_numeric(df, errors="coerce")

    # Simple impute
    vals = df.values.reshape(-1, 1)
    imp = SimpleImputer(strategy="median")
    vals = imp.fit_transform(vals)

    # Very simple LGBM classifier to detect drift on this single feature
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(src))

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 8,
        "min_data_in_leaf": 200,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "verbose": -1,
        "seed": seed,
    }

    for tr_idx, va_idx in skf.split(vals, src):
        dtr = lgb.Dataset(vals[tr_idx], label=src[tr_idx])
        dva = lgb.Dataset(vals[va_idx], label=src[va_idx])
        m = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va_idx] = m.predict(vals[va_idx])

    return roc_auc_score(src, oof)


drift_scores = []
for col in X.columns:
    score = drift_auc_for_feature(X[col], X_test[col], seed=SEED)
    drift_scores.append((col, score))


drift_df = pd.DataFrame(drift_scores, columns=["feature", "drift_auc"]).sort_values("drift_auc", ascending=False)
drift_df.head(15)


DROP_TOP_K = 4  # try 2, 4, 6 (start with 4)
to_drop = drift_df["feature"].head(DROP_TOP_K).tolist()

print("Dropping drifting features:", to_drop)

X = X.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)

# Refresh cat columns after drop
cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]


plt.figure(figsize=(7,4))
plt.hist(drift_df["drift_auc"], bins=20)
plt.title("Drift AUC distribution (train vs test)")
plt.xlabel("Drift AUC (higher = more drift)")
plt.ylabel("Count")
plt.grid(alpha=0.2)
plt.show()


# ============================================================
# LightGBM Training Function (OOF + Test Prediction)
# ============================================================
def train_lgb_oof_test(X, y, X_test, cat_cols, seed=42):
    """
    Train a regularized LightGBM model using Repeated Stratified K-Fold CV.

    This function:
    - Generates out-of-fold (OOF) predictions for unbiased CV evaluation
    - Produces averaged test predictions across all folds
    - Uses strong regularization to improve generalization on public LB
    - Supports native categorical handling in LightGBM

    Parameters
    ----------
    X : pd.DataFrame
        Training feature matrix
    y : pd.Series
        Binary target labels
    X_test : pd.DataFrame
        Test feature matrix
    cat_cols : list
        List of categorical column names (category dtype)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    oof : np.ndarray
        Out-of-fold predicted probabilities for training data
    test_pred : np.ndarray
        Averaged predicted probabilities for test data
    cv_auc : float
        ROC-AUC score computed on OOF predictions
    """

    # --------------------------------------------------------
    # Cross-validation strategy
    # --------------------------------------------------------
    # Repeated Stratified K-Fold:
    # - Maintains class balance in each fold
    # - Repetition reduces dependence on a single lucky split
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=seed
    )

    # Initialize prediction arrays
    # OOF predictions align with training indices
    # Test predictions are accumulated and averaged across folds
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    # --------------------------------------------------------
    # LightGBM hyperparameters (generalization-focused)
    # --------------------------------------------------------
    params = {
        "objective": "binary",          # Binary classification
        "metric": "auc",                # Evaluation metric
        "learning_rate": 0.01,          # Low LR for smoother learning
        "num_leaves": 20,               # Small tree complexity
        "max_depth": 5,                 # Limit depth to reduce overfitting
        "min_data_in_leaf": 140,        # Strong regularization
        "feature_fraction": 0.65,       # Column subsampling
        "bagging_fraction": 0.65,       # Row subsampling
        "bagging_freq": 1,
        "lambda_l1": 10,                # L1 regularization
        "lambda_l2": 25,                # L2 regularization
        "max_cat_to_onehot": 4,         # Handle small categorical features safely
        "verbose": -1,
        "seed": seed,
    }

    # --------------------------------------------------------
    # Training loop across CV folds
    # --------------------------------------------------------
    total_folds = cv.get_n_splits()

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        # Split data into training and validation sets
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Create LightGBM datasets
        # free_raw_data=False allows reuse of data across folds
        dtr = lgb.Dataset(
            X_tr,
            label=y_tr,
            categorical_feature=cat_cols,
            free_raw_data=False
        )
        dva = lgb.Dataset(
            X_va,
            label=y_va,
            categorical_feature=cat_cols,
            free_raw_data=False
        )

        # Train LightGBM model with early stopping
        model = lgb.train(
            params,
            dtr,
            num_boost_round=8000,            # High cap; early stopping decides
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(
                stopping_rounds=300,
                verbose=False
            )]
        )

        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------
        # OOF predictions for validation fold
        oof[va_idx] = model.predict(
            X_va,
            num_iteration=model.best_iteration
        )

        # Accumulate test predictions (averaged later)
        test_pred += model.predict(
            X_test,
            num_iteration=model.best_iteration
        ) / total_folds

    # --------------------------------------------------------
    # Final CV evaluation using OOF predictions
    # --------------------------------------------------------
    cv_auc = roc_auc_score(y, oof)

    return oof, test_pred, cv_auc


SEEDS = [42, 1337, 2025]  # seed averaging helps LB stability


oof_lgb_all = []
test_lgb_all = []
auc_lgb_all = []

for s in SEEDS:
    oof_s, test_s, auc_s = train_lgb_oof_test(X, y, X_test, cat_cols, seed=s)
    oof_lgb_all.append(oof_s)
    test_lgb_all.append(test_s)
    auc_lgb_all.append(auc_s)
    print(f"LGB seed {s} CV AUC: {auc_s:.5f}")


# ============================================================
# LightGBM: Seed-Averaged Predictions (Variance Reduction)
# ============================================================
# Motivation:
# - Individual random seeds can produce slightly different models
# - Averaging across multiple seeds improves robustness
# - Reduces sensitivity to lucky/unlucky initializations
# - Typically narrows the CV–Public LB gap in Playground datasets

# Average out-of-fold predictions across all trained seeds
oof_lgb = np.mean(oof_lgb_all, axis=0)

# Average test predictions across all trained seeds
test_lgb = np.mean(test_lgb_all, axis=0)

# Evaluate LightGBM using seed-averaged out-of-fold predictions
print("LightGBM Seed-Averaged CV AUC:",
      roc_auc_score(y, oof_lgb))


# ============================================================
# Logistic Regression: Preprocessing Pipeline Construction
# ============================================================
# Goal:
# - Prepare numerical and categorical features appropriately
# - Ensure Logistic Regression receives scaled, well-imputed inputs
# - Build a reusable and leakage-safe preprocessing pipeline

# ----------------------------
# Numerical feature pipeline
# ----------------------------
# Steps:
# 1. Impute missing values using median (robust to outliers)
# 2. Standardize features (required for linear models)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# ----------------------------
# Categorical feature pipeline
# ----------------------------
# Steps:
# 1. Impute missing values using most frequent category
# 2. One-hot encode categories
#    - handle_unknown='ignore' prevents inference-time errors
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", __import__("sklearn").preprocessing.OneHotEncoder(
        handle_unknown="ignore"
    )),
])

# ----------------------------
# Column-wise preprocessing
# ----------------------------
# Applies:
# - Numerical pipeline to numerical columns
# - Categorical pipeline to categorical columns
# Drops any columns not explicitly listed (safety measure)
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# ----------------------------
# Final Logistic Regression pipeline
# ----------------------------
# Combines preprocessing + model into a single object
# class_weight='balanced' compensates for target imbalance
lr_model = Pipeline(steps=[
    ("prep", preprocess),
    ("lr", LogisticRegression(
        max_iter=4000,
        class_weight="balanced"
    )),
])


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_lr = np.zeros(len(X))
test_lr = np.zeros(len(X_test))


# ============================================================
# Logistic Regression: Stratified K-Fold Training & Evaluation
# ============================================================
# Purpose:
# - Acts as a stable baseline model
# - Provides robust predictions under distribution shift
# - Serves as a regularizing component in the final ensemble

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    # Train Logistic Regression on the current training fold
    lr_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])

    # Generate out-of-fold predictions for validation data
    # These predictions are used for unbiased CV AUC estimation
    oof_lr[va_idx] = lr_model.predict_proba(X.iloc[va_idx])[:, 1]

    # Generate predictions for the test set
    # Average predictions across folds to reduce variance
    test_lr += lr_model.predict_proba(X_test)[:, 1] / skf.n_splits

# Evaluate Logistic Regression using full out-of-fold predictions
print("Logistic Regression CV AUC:", roc_auc_score(y, oof_lr))


# ============================================================
# Final Conservative Ensemble (LightGBM + Logistic Regression)
# ============================================================
# Motivation:
# - LightGBM captures nonlinear patterns and interactions
# - Logistic Regression provides stability under distribution shift
# - A conservative weighted blend improves generalization on Public LB

# Blend out-of-fold predictions (used for unbiased CV evaluation)
oof_blend = (
    0.85 * oof_lgb   # Primary model: regularized LightGBM (strong signal)
    + 0.15 * oof_lr  # Stabilizer: Logistic Regression (robust baseline)
)

# Blend test predictions using the same weights
test_blend = (
    0.85 * test_lgb  # Majority contribution from LightGBM
    + 0.15 * test_lr # Smaller contribution to reduce overfitting risk
)

# Evaluate blended model using out-of-fold predictions
print("Blended CV AUC (LightGBM + Logistic Regression):",
      roc_auc_score(y, oof_blend))


fpr, tpr, _ = roc_curve(y, oof_blend)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"OOF ROC (AUC={roc_auc:.4f})")
plt.plot([0,1], [0,1], "--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (OOF) - Safe Blend")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
sub[TARGET] = test_blend

out_path = "/kaggle/working/submission_lgb_no_catboost.csv"
sub.to_csv(out_path, index=False)

print("Saved:", out_path)
sub.head()




