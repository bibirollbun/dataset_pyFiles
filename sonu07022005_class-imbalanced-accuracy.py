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


# ==============================================
# Detailed EDA for Playground Series - S5E12
# ==============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid", font_scale=1.05)

# -----------------------
# 1. Load the datasets
# -----------------------
DATA_PATH = Path("/kaggle/input/playground-series-s5e12")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

print("=== Shapes ===")
print(f"train: {train.shape}")
print(f"test:  {test.shape}")
print(f"sample_submission: {sample_submission.shape}\n")

print("=== Train head ===")
display(train.head())

print("\n=== Test head ===")
display(test.head())

print("\n=== Sample submission head ===")
display(sample_submission.head())

# --------------------------------------
# 2. Identify target and ID columns
# --------------------------------------
# Try to infer target from sample_submission (last column typically)
possible_targets = [c for c in sample_submission.columns if c in train.columns]

if len(possible_targets) == 1:
    target_col = possible_targets[0]
else:
    # fallback: assume last column in train that is not in test is target
    diff_cols = [c for c in train.columns if c not in test.columns]
    if len(diff_cols) == 1:
        target_col = diff_cols[0]
    else:
        target_col = train.columns[-1]  # very generic fallback

# ID column: typically the other column in sample_submission
id_candidates = [c for c in sample_submission.columns if c != target_col]
id_col = id_candidates[0] if len(id_candidates) >= 1 else None

print("\n=== Inferred columns ===")
print(f"Target column: {target_col}")
print(f"ID column:     {id_col}")

# --------------------------------------
# 3. Basic info & column types
# --------------------------------------
print("\n=== Train info ===")
print(train.info())

print("\n=== Test info ===")
print(test.info())

# Separate features & target
if target_col in train.columns:
    y = train[target_col]
    X = train.drop(columns=[target_col])
else:
    y = None
    X = train.copy()

# Feature types
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\n=== Feature type summary ===")
print(f"Numeric features:     {len(num_cols)}")
print(f"Categorical features: {len(cat_cols)}")
print("Numeric columns:", num_cols[:10], "..." if len(num_cols) > 10 else "")
print("Categorical columns:", cat_cols[:10], "..." if len(cat_cols) > 10 else "")

# --------------------------------------
# 4. Missing values
# --------------------------------------
def missing_report(df, name):
    print(f"\n=== Missing values report: {name} ===")
    mis_cnt = df.isna().sum()
    mis_pct = df.isna().mean() * 100
    mr = pd.DataFrame({"missing_count": mis_cnt, "missing_pct": mis_pct})
    mr = mr[mr["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    if mr.empty:
        print("No missing values.")
    else:
        display(mr)

missing_report(train, "train")
missing_report(test, "test")

# --------------------------------------
# 5. Target analysis
# --------------------------------------
if y is not None:
    print("\n=== Target basic statistics ===")
    print("dtype:", y.dtype)
    print("nunique:", y.nunique())
    print(y.describe())

    # Heuristics: treat as classification if few unique values
    n_unique = y.nunique()
    is_classification = (y.dtype == "object") or (n_unique <= 20)

    if is_classification:
        print("\nTarget distribution (counts):")
        display(y.value_counts().to_frame("count"))
        print("\nTarget distribution (relative):")
        display((y.value_counts(normalize=True) * 100).to_frame("percentage (%)"))

        plt.figure(figsize=(6, 4))
        sns.countplot(x=y)
        plt.title("Target distribution (classification)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    else:
        # Regression-style
        plt.figure(figsize=(6, 4))
        sns.histplot(y, kde=True)
        plt.title("Target distribution (regression)")
        plt.tight_layout()
        plt.show()
else:
    print("\nNo target column found or target not in train.")

# --------------------------------------
# 6. Numeric features: basic stats
# --------------------------------------
if num_cols:
    print("\n=== Numeric features: describe() ===")
    display(train[num_cols].describe().T)

    # Histograms of numeric features (train)
    print("\n=== Numeric features: distributions (histograms) ===")
    n_cols_plot = 3
    n_plots = len(num_cols)
    n_rows_plot = int(np.ceil(n_plots / n_cols_plot))

    plt.figure(figsize=(5 * n_cols_plot, 3.5 * n_rows_plot))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows_plot, n_cols_plot, i)
        sns.histplot(train[col], kde=True, bins=30)
        plt.title(col)
    plt.tight_layout()
    plt.show()

# --------------------------------------
# 7. Categorical features: basic stats
# --------------------------------------
if cat_cols:
    print("\n=== Categorical features: cardinality ===")
    cat_card = train[cat_cols].nunique().sort_values(ascending=False)
    display(cat_card.to_frame("n_unique"))

    # Show value counts for top few categorical columns
    max_example_cats = 5  # how many categorical features to display distributions for
    print(f"\n=== Example categorical distributions (up to {max_example_cats} features) ===")
    for col in cat_cols[:max_example_cats]:
        print(f"\nColumn: {col}")
        display(train[col].value_counts(dropna=False).head(20).to_frame("count"))

        plt.figure(figsize=(6, 4))
        vc = train[col].value_counts().head(20)
        sns.barplot(x=vc.index.astype(str), y=vc.values)
        plt.title(f"{col} (top 20 categories)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

# --------------------------------------
# 8. Correlations (numeric)
# --------------------------------------
if num_cols:
    print("\n=== Correlation matrix (numeric features) ===")
    corr = train[num_cols].corr()
    # Show a small heatmap for top 15 features with largest overall variance
    # (to keep plot readable)
    var = train[num_cols].var().sort_values(ascending=False)
    top_corr_cols = var.head(min(15, len(var))).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(train[top_corr_cols].corr(), annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (top numeric features)")
    plt.tight_layout()
    plt.show()

    # Correlation with target if target is numeric
    if y is not None and np.issubdtype(y.dtype, np.number):
        print("\n=== Correlation with numeric target ===")
        corr_target = train[num_cols + [target_col]].corr()[target_col].drop(target_col)
        corr_target = corr_target.sort_values(ascending=False)
        display(corr_target.to_frame("corr_with_target"))

        plt.figure(figsize=(6, max(4, 0.3 * len(corr_target))))
        sns.barplot(x=corr_target.values, y=corr_target.index)
        plt.title("Correlation with target")
        plt.tight_layout()
        plt.show()

# --------------------------------------
# 9. Relationship between top features and target
# --------------------------------------
if y is not None and num_cols:
    # Take top few numeric features with highest variance
    top_num = train[num_cols].var().sort_values(ascending=False).head(6).index.tolist()
    print("\n=== Feature vs Target (top numeric features) ===")
    print("Top numeric features:", top_num)

    n_cols_plot = 3
    n_plots = len(top_num)
    n_rows_plot = int(np.ceil(n_plots / n_cols_plot))

    plt.figure(figsize=(5 * n_cols_plot, 4 * n_rows_plot))
    for i, col in enumerate(top_num, 1):
        plt.subplot(n_rows_plot, n_cols_plot, i)
        if (y.dtype == "object") or (y.nunique() <= 20):
            # classification: plot boxplot by class
            sns.boxplot(x=y, y=train[col])
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{col} vs {target_col}")
        else:
            # regression: scatter with a small alpha
            sns.scatterplot(x=train[col], y=y, alpha=0.3)
            plt.title(f"{col} vs {target_col}")
    plt.tight_layout()
    plt.show()

# --------------------------------------
# 10. Train vs Test comparison (feature distributions)
# --------------------------------------
common_feature_cols = [c for c in train.columns if c in test.columns and c != target_col]
common_num_cols = [c for c in common_feature_cols if c in num_cols]

print("\n=== Train vs Test: basic comparison ===")
print(f"Common feature columns: {len(common_feature_cols)}")
print(f"Common numeric feature columns: {len(common_num_cols)}")

# Describe numeric features in train & test
print("\nTrain numeric describe():")
display(train[common_num_cols].describe().T)

print("\nTest numeric describe():")
display(test[common_num_cols].describe().T)

# Plot a few numeric distributions in train vs test
max_compare = min(6, len(common_num_cols))
print(f"\n=== Train vs Test distributions (first {max_compare} numeric features) ===")
for col in common_num_cols[:max_compare]:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(train[col], label="train", fill=True, alpha=0.4)
    sns.kdeplot(test[col], label="test", fill=True, alpha=0.4)
    plt.title(f"Train vs Test: {col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\nEDA complete.")


# ===========================================
# XGBoost Pipeline for Playground S5E12
# ===========================================
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)

from xgboost import XGBClassifier, XGBRegressor

# -------------------------
# 1. Load the data
# -------------------------
DATA_PATH = Path("/kaggle/input/playground-series-s5e12")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

print("Shapes:")
print("  train:", train.shape)
print("  test: ", test.shape)
print("  sample_submission:", sample_submission.shape)

# -------------------------
# 2. Infer ID and target
# -------------------------
# Target is usually the column in sample_submission that also exists in train
possible_targets = [c for c in sample_submission.columns if c in train.columns]
if len(possible_targets) == 1:
    target_col = possible_targets[0]
else:
    # Fallback: any column in train not in test
    diff_cols = [c for c in train.columns if c not in test.columns]
    if len(diff_cols) == 1:
        target_col = diff_cols[0]
    else:
        target_col = train.columns[-1]  # very generic fallback

id_candidates = [c for c in sample_submission.columns if c != target_col]
id_col = id_candidates[0] if len(id_candidates) >= 1 else None

print("\nInferred columns:")
print("  ID column:     ", id_col)
print("  Target column: ", target_col)

# -------------------------
# 3. Split features & target
# -------------------------
y = train[target_col].copy()
X = train.drop(columns=[target_col])

# Remove ID from features if present
if id_col is not None and id_col in X.columns:
    X = X.drop(columns=[id_col])
if id_col is not None and id_col in test.columns:
    test_features = test.drop(columns=[id_col])
else:
    test_features = test.copy()

# Detect numeric & categorical features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nFeatures:")
print("  Numeric:     ", len(num_cols))
print("  Categorical: ", len(cat_cols))

# -------------------------
# 4. Decide classification or regression
# -------------------------
n_unique = y.nunique()
is_classification = (y.dtype == "object") or (n_unique <= 20)

print("\nProblem type:")
print("  unique target values:", n_unique)
print("  classification?:     ", is_classification)

# For classification, encode labels if needed
label_encoder = None
if is_classification:
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
else:
    y_enc = y.values  # numeric regression target

# -------------------------
# 5. Train/validation split
# -------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc if is_classification else None
)

print("\nTrain/Valid shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

# -------------------------
# 6. Preprocessing pipelines
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# -------------------------
# 7. XGBoost model
# -------------------------
if is_classification:
    n_classes = len(np.unique(y_enc))
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
else:
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_model),
])

# -------------------------
# 8. Train and validate
# -------------------------
model_pipeline.fit(X_train, y_train)

# Validation predictions
if is_classification:
    n_classes = len(np.unique(y_enc))
    if n_classes == 2:
        # Probabilities for positive class
        y_valid_proba = model_pipeline.predict_proba(X_valid)[:, 1]
        y_valid_pred = (y_valid_proba >= 0.5).astype(int)
        acc = accuracy_score(y_valid, y_valid_pred)
        try:
            auc = roc_auc_score(y_valid, y_valid_proba)
        except ValueError:
            auc = None
        print("\nValidation metrics (binary classification):")
        print("  Accuracy:", round(acc, 4))
        if auc is not None:
            print("  ROC AUC: ", round(auc, 4))
    else:
        y_valid_pred = model_pipeline.predict(X_valid)
        acc = accuracy_score(y_valid, y_valid_pred)
        print("\nValidation metrics (multi-class classification):")
        print("  Accuracy:", round(acc, 4))
else:
    y_valid_pred = model_pipeline.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_valid_pred, squared=False)
    r2 = r2_score(y_valid, y_valid_pred)
    print("\nValidation metrics (regression):")
    print("  RMSE:", round(rmse, 4))
    print("  R2:  ", round(r2, 4))

# -------------------------
# 9. Train on full train data
# -------------------------
print("\nFitting model on full training data...")
model_pipeline.fit(X, y_enc)

# -------------------------
# 10. Predict on test and create submission
# -------------------------
if is_classification:
    n_classes = len(np.unique(y_enc))
    if n_classes == 2:
        # Most Kaggle binary tasks expect probabilities for class 1
        test_pred = model_pipeline.predict_proba(test_features)[:, 1]
    else:
        # For multi-class, predicted class index
        test_pred = model_pipeline.predict(test_features)
        # Map back to original labels if needed
        if label_encoder is not None:
            test_pred = label_encoder.inverse_transform(test_pred.astype(int))
else:
    test_pred = model_pipeline.predict(test_features)

submission = sample_submission.copy()
submission[target_col] = test_pred
submission.to_csv("submission.csv", index=False)

print("\nSaved submission.csv with shape:", submission.shape)
submission.head()


# ===========================================
# XGBoost Pipeline + Confusion Matrix + ROC
# ===========================================
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    classification_report,
)

from xgboost import XGBClassifier, XGBRegressor

sns.set(style="whitegrid", font_scale=1.05)

# -------------------------
# 1. Load the data
# -------------------------
DATA_PATH = Path("/kaggle/input/playground-series-s5e12")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

print("Shapes:")
print("  train:", train.shape)
print("  test: ", test.shape)
print("  sample_submission:", sample_submission.shape)

# -------------------------
# 2. Infer ID and target
# -------------------------
# Target is usually the column in sample_submission that also exists in train
possible_targets = [c for c in sample_submission.columns if c in train.columns]
if len(possible_targets) == 1:
    target_col = possible_targets[0]
else:
    # Fallback: any column in train not in test
    diff_cols = [c for c in train.columns if c not in test.columns]
    if len(diff_cols) == 1:
        target_col = diff_cols[0]
    else:
        target_col = train.columns[-1]  # very generic fallback

id_candidates = [c for c in sample_submission.columns if c != target_col]
id_col = id_candidates[0] if len(id_candidates) >= 1 else None

print("\nInferred columns:")
print("  ID column:     ", id_col)
print("  Target column: ", target_col)

# -------------------------
# 3. Split features & target
# -------------------------
y = train[target_col].copy()
X = train.drop(columns=[target_col])

# Remove ID from features if present
if id_col is not None and id_col in X.columns:
    X = X.drop(columns=[id_col])
if id_col is not None and id_col in test.columns:
    test_features = test.drop(columns=[id_col])
else:
    test_features = test.copy()

# Detect numeric & categorical features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nFeatures:")
print("  Numeric:     ", len(num_cols))
print("  Categorical: ", len(cat_cols))

# -------------------------
# 4. Decide classification or regression
# -------------------------
n_unique = y.nunique()
is_classification = (y.dtype == "object") or (n_unique <= 20)

print("\nProblem type:")
print("  unique target values:", n_unique)
print("  classification?:     ", is_classification)

# For classification, encode labels if needed
label_encoder = None
if is_classification:
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
else:
    y_enc = y.values  # numeric regression target

# -------------------------
# 5. Train/validation split
# -------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc if is_classification else None
)

print("\nTrain/Valid shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

# -------------------------
# 6. Preprocessing pipelines
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# -------------------------
# 7. XGBoost model
# -------------------------
if is_classification:
    n_classes = len(np.unique(y_enc))
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
else:
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_model),
])

# -------------------------
# 8. Train and validate
# -------------------------
model_pipeline.fit(X_train, y_train)

if is_classification:
    n_classes = len(np.unique(y_enc))
    y_valid_proba = model_pipeline.predict_proba(X_valid)

    if n_classes == 2:
        # Binary classification: use probability of positive class
        y_valid_score = y_valid_proba[:, 1]
        y_valid_pred = (y_valid_score >= 0.5).astype(int)

        acc = accuracy_score(y_valid, y_valid_pred)
        try:
            auc = roc_auc_score(y_valid, y_valid_score)
        except ValueError:
            auc = None

        print("\nValidation metrics (binary classification):")
        print("  Accuracy:", round(acc, 4))
        if auc is not None:
            print("  ROC AUC: ", round(auc, 4))
        print("\nClassification report:")
        # For readability, convert back to original labels
        if label_encoder is not None:
            y_valid_true_labels = label_encoder.inverse_transform(y_valid)
            y_valid_pred_labels = label_encoder.inverse_transform(y_valid_pred)
            print(classification_report(y_valid_true_labels, y_valid_pred_labels))
        else:
            print(classification_report(y_valid, y_valid_pred))

        # -------------------------
        # Confusion matrix (binary)
        # -------------------------
        if label_encoder is not None:
            display_labels = label_encoder.classes_
        else:
            display_labels = np.unique(y)

        cm = confusion_matrix(y_valid, y_valid_pred)
        plt.figure(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix (Validation)")
        plt.tight_layout()
        plt.show()

        # -------------------------
        # ROC curve (binary)
        # -------------------------
        fpr, tpr, thresholds = roc_curve(y_valid, y_valid_score)
        plt.figure(figsize=(5, 4))
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.title("ROC Curve (Validation)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        # Multi-class classification
        y_valid_pred = model_pipeline.predict(X_valid)
        acc = accuracy_score(y_valid, y_valid_pred)
        print("\nValidation metrics (multi-class classification):")
        print("  Accuracy:", round(acc, 4))

        if label_encoder is not None:
            y_valid_true_labels = label_encoder.inverse_transform(y_valid)
            y_valid_pred_labels = label_encoder.inverse_transform(y_valid_pred)
            print("\nClassification report:")
            print(classification_report(y_valid_true_labels, y_valid_pred_labels))

            cm = confusion_matrix(y_valid_true_labels, y_valid_pred_labels, labels=label_encoder.classes_)
            plt.figure(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
            disp.plot(cmap="Blues", values_format="d")
            plt.title("Confusion Matrix (Validation, Multi-class)")
            plt.tight_layout()
            plt.show()
        else:
            print("\nClassification report:")
            print(classification_report(y_valid, y_valid_pred))

else:
    # Regression metrics
    y_valid_pred = model_pipeline.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_valid_pred, squared=False)
    r2 = r2_score(y_valid, y_valid_pred)
    print("\nValidation metrics (regression):")
    print("  RMSE:", round(rmse, 4))
    print("  R2:  ", round(r2, 4))

# -------------------------
# 9. Train on full train data
# -------------------------
print("\nFitting model on full training data...")
model_pipeline.fit(X, y_enc)

# -------------------------
# 10. Predict on test and create submission
# -------------------------
if is_classification:
    n_classes = len(np.unique(y_enc))
    if n_classes == 2:
        # For binary competitions: often need probability of positive class
        test_pred = model_pipeline.predict_proba(test_features)[:, 1]
    else:
        test_pred = model_pipeline.predict(test_features)
        if label_encoder is not None:
            test_pred = label_encoder.inverse_transform(test_pred.astype(int))
else:
    test_pred = model_pipeline.predict(test_features)

submission = sample_submission.copy()
submission[target_col] = test_pred
submission.to_csv("submission.csv", index=False)

print("\nSaved submission.csv with shape:", submission.shape)
submission.head()


# ======================================================
# XGBoost with selected features removed (S5E12)
# Drops: id, sleep_hours_per_day, screen_time_hours_per_day
# ======================================================
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    classification_report,
)

from xgboost import XGBClassifier, XGBRegressor

sns.set(style="whitegrid", font_scale=1.05)

# -------------------------
# 1. Load data
# -------------------------
DATA_PATH = Path("/kaggle/input/playground-series-s5e12")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

print("Shapes:")
print("  train:", train.shape)
print("  test: ", test.shape)
print("  sample_submission:", sample_submission.shape)

# -------------------------
# 2. Infer target and ID
# -------------------------
possible_targets = [c for c in sample_submission.columns if c in train.columns]
if len(possible_targets) == 1:
    target_col = possible_targets[0]
else:
    diff_cols = [c for c in train.columns if c not in test.columns]
    if len(diff_cols) == 1:
        target_col = diff_cols[0]
    else:
        target_col = train.columns[-1]

id_candidates = [c for c in sample_submission.columns if c != target_col]
id_col = id_candidates[0] if len(id_candidates) >= 1 else None

print("\nInferred columns:")
print("  ID column:     ", id_col)
print("  Target column: ", target_col)

# -------------------------
# 3. Drop chosen features
# -------------------------
drop_features = ["id", "sleep_hours_per_day", "screen_time_hours_per_day"]

# Ensure we only drop columns that exist
drop_features_train = [c for c in drop_features if c in train.columns]
drop_features_test = [c for c in drop_features if c in test.columns]

y = train[target_col].copy()
X = train.drop(columns=[target_col] + drop_features_train)

test_features = test.drop(columns=drop_features_test)

print("\nDropped features:", drop_features_train)
print("X shape after drop:", X.shape)
print("test_features shape after drop:", test_features.shape)

# -------------------------
# 4. Detect numeric & categorical
# -------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nFeatures:")
print("  Numeric:     ", len(num_cols))
print("  Categorical: ", len(cat_cols))

# -------------------------
# 5. Decide classification vs regression
# -------------------------
n_unique = y.nunique()
is_classification = (y.dtype == "object") or (n_unique <= 20)

print("\nProblem type:")
print("  unique target values:", n_unique)
print("  classification?:     ", is_classification)

label_encoder = None
if is_classification:
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
else:
    y_enc = y.values

# -------------------------
# 6. Train/validation split
# -------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc if is_classification else None
)

print("\nTrain/Valid shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

# -------------------------
# 7. Preprocessing
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# -------------------------
# 8. XGBoost model
# -------------------------
if is_classification:
    n_classes = len(np.unique(y_enc))
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
else:
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_model),
])

# -------------------------
# 9. Train and validate
# -------------------------
model_pipeline.fit(X_train, y_train)

if is_classification:
    n_classes = len(np.unique(y_enc))
    y_valid_proba = model_pipeline.predict_proba(X_valid)

    if n_classes == 2:
        # Binary classification
        y_valid_score = y_valid_proba[:, 1]
        y_valid_pred = (y_valid_score >= 0.5).astype(int)

        acc = accuracy_score(y_valid, y_valid_pred)
        try:
            auc = roc_auc_score(y_valid, y_valid_score)
        except ValueError:
            auc = None

        print("\nValidation metrics (binary classification, XGBoost):")
        print("  Accuracy:", round(acc, 4))
        if auc is not None:
            print("  ROC AUC: ", round(auc, 4))

        # Convert to original labels for readability
        if label_encoder is not None:
            y_valid_true_labels = label_encoder.inverse_transform(y_valid)
            y_valid_pred_labels = label_encoder.inverse_transform(y_valid_pred)
        else:
            y_valid_true_labels = y_valid
            y_valid_pred_labels = y_valid_pred

        print("\nClassification report:")
        print(classification_report(y_valid_true_labels, y_valid_pred_labels))

        # Confusion matrix
        cm = confusion_matrix(y_valid_true_labels, y_valid_pred_labels)
        labels = np.unique(y_valid_true_labels)

        plt.figure(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix (Validation, XGBoost)")
        plt.tight_layout()
        plt.show()

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_valid, y_valid_score)
        plt.figure(figsize=(5, 4))
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.title("ROC Curve (Validation, XGBoost)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        # Multi-class
        y_valid_pred = model_pipeline.predict(X_valid)
        acc = accuracy_score(y_valid, y_valid_pred)
        print("\nValidation metrics (multi-class classification, XGBoost):")
        print("  Accuracy:", round(acc, 4))

        if label_encoder is not None:
            y_valid_true_labels = label_encoder.inverse_transform(y_valid)
            y_valid_pred_labels = label_encoder.inverse_transform(y_valid_pred)
            labels = label_encoder.classes_
        else:
            y_valid_true_labels = y_valid
            y_valid_pred_labels = y_valid_pred
            labels = np.unique(y_valid_true_labels)

        print("\nClassification report:")
        print(classification_report(y_valid_true_labels, y_valid_pred_labels))

        cm = confusion_matrix(y_valid_true_labels, y_valid_pred_labels, labels=labels)
        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix (Validation, Multi-class, XGBoost)")
        plt.tight_layout()
        plt.show()

else:
    # Regression
    y_valid_pred = model_pipeline.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_valid_pred, squared=False)
    r2 = r2_score(y_valid, y_valid_pred)
    print("\nValidation metrics (regression, XGBoost):")
    print("  RMSE:", round(rmse, 4))
    print("  R2:  ", round(r2, 4))

# -------------------------
# 10. Train on full data & predict test
# -------------------------
print("\nFitting XGBoost on full training data...")
model_pipeline.fit(X, y_enc)

if is_classification:
    n_classes = len(np.unique(y_enc))
    if n_classes == 2:
        test_pred = model_pipeline.predict_proba(test_features)[:, 1]
    else:
        test_pred = model_pipeline.predict(test_features)
        if label_encoder is not None:
            test_pred = label_encoder.inverse_transform(test_pred.astype(int))
else:
    test_pred = model_pipeline.predict(test_features)

submission = sample_submission.copy()
submission[target_col] = test_pred
submission.to_csv("submission.csv", index=False)

print("\nSaved submission.csv with shape:", submission.shape)
submission.head()


# ===========================================
# Random Forest Pipeline + CM + ROC
# ===========================================
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    classification_report,
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sns.set(style="whitegrid", font_scale=1.05)

# -------------------------
# 1. Load the data
# -------------------------
DATA_PATH = Path("/kaggle/input/playground-series-s5e12")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

print("Shapes:")
print("  train:", train.shape)
print("  test: ", test.shape)
print("  sample_submission:", sample_submission.shape)

# -------------------------
# 2. Infer ID and target
# -------------------------
possible_targets = [c for c in sample_submission.columns if c in train.columns]
if len(possible_targets) == 1:
    target_col = possible_targets[0]
else:
    diff_cols = [c for c in train.columns if c not in test.columns]
    if len(diff_cols) == 1:
        target_col = diff_cols[0]
    else:
        target_col = train.columns[-1]

id_candidates = [c for c in sample_submission.columns if c != target_col]
id_col = id_candidates[0] if len(id_candidates) >= 1 else None

print("\nInferred columns:")
print("  ID column:     ", id_col)
print("  Target column: ", target_col)

# -------------------------
# 3. Split features & target
# -------------------------
y = train[target_col].copy()
X = train.drop(columns=[target_col])

# Remove ID from features if present
if id_col is not None and id_col in X.columns:
    X = X.drop(columns=[id_col])
if id_col is not None and id_col in test.columns:
    test_features = test.drop(columns=[id_col])
else:
    test_features = test.copy()

# Detect numeric & categorical features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nFeatures:")
print("  Numeric:     ", len(num_cols))
print("  Categorical: ", len(cat_cols))

# -------------------------
# 4. Decide classification or regression
# -------------------------
n_unique = y.nunique()
is_classification = (y.dtype == "object") or (n_unique <= 20)

print("\nProblem type:")
print("  unique target values:", n_unique)
print("  classification?:     ", is_classification)

# Encode classification labels if needed
label_encoder = None
if is_classification:
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
else:
    y_enc = y.values

# -------------------------
# 5. Train/validation split
# -------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc if is_classification else None
)

print("\nTrain/Valid shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

# -------------------------
# 6. Preprocessing pipelines
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# -------------------------
# 7. Random Forest model
# -------------------------
if is_classification:
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
else:
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model),
])

# -------------------------
# 8. Train and validate
# -------------------------
model_pipeline.fit(X_train, y_train)

if is_classification:
    n_classes = len(np.unique(y_enc))
    y_valid_proba = model_pipeline.predict_proba(X_valid)

    if n_classes == 2:
        # Binary classification: probability of positive class
        y_valid_score = y_valid_proba[:, 1]
        y_valid_pred = (y_valid_score >= 0.5).astype(int)

        acc = accuracy_score(y_valid, y_valid_pred)
        try:
            auc = roc_auc_score(y_valid, y_valid_score)
        except ValueError:
            auc = None

        print("\nValidation metrics (binary classification, Random Forest):")
        print("  Accuracy:", round(acc, 4))
        if auc is not None:
            print("  ROC AUC: ", round(auc, 4))

        # Convert back to original labels for readable report
        if label_encoder is not None:
            y_valid_true_labels = label_encoder.inverse_transform(y_valid)
            y_valid_pred_labels = label_encoder.inverse_transform(y_valid_pred)
        else:
            y_valid_true_labels = y_valid
            y_valid_pred_labels = y_valid_pred

        print("\nClassification report:")
        print(classification_report(y_valid_true_labels, y_valid_pred_labels))

        # Confusion matrix
        cm = confusion_matrix(y_valid_true_labels, y_valid_pred_labels)
        labels = np.unique(y_valid_true_labels)

        plt.figure(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix (Validation, Random Forest)")
        plt.tight_layout()
        plt.show()

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_valid, y_valid_score)
        plt.figure(figsize=(5, 4))
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.title("ROC Curve (Validation, Random Forest)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        # Multi-class classification
        y_valid_pred = model_pipeline.predict(X_valid)
        acc = accuracy_score(y_valid, y_valid_pred)
        print("\nValidation metrics (multi-class classification, Random Forest):")
        print("  Accuracy:", round(acc, 4))

        if label_encoder is not None:
            y_valid_true_labels = label_encoder.inverse_transform(y_valid)
            y_valid_pred_labels = label_encoder.inverse_transform(y_valid_pred)
            labels = label_encoder.classes_
        else:
            y_valid_true_labels = y_valid
            y_valid_pred_labels = y_valid_pred
            labels = np.unique(y_valid_true_labels)

        print("\nClassification report:")
        print(classification_report(y_valid_true_labels, y_valid_pred_labels))

        cm = confusion_matrix(y_valid_true_labels, y_valid_pred_labels, labels=labels)
        plt.figure(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix (Validation, Multi-class, Random Forest)")
        plt.tight_layout()
        plt.show()

else:
    # Regression metrics
    y_valid_pred = model_pipeline.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_valid_pred, squared=False)
    r2 = r2_score(y_valid, y_valid_pred)
    print("\nValidation metrics (regression, Random Forest):")
    print("  RMSE:", round(rmse, 4))
    print("  R2:  ", round(r2, 4))

# -------------------------
# 9. Train on full train data
# -------------------------
print("\nFitting Random Forest on full training data...")
model_pipeline.fit(X, y_enc)

# -------------------------
# 10. Predict on test and create submission
# -------------------------
if is_classification:
    n_classes = len(np.unique(y_enc))
    if n_classes == 2:
        # Usually competitions want probability of positive class
        test_pred = model_pipeline.predict_proba(test_features)[:, 1]
    else:
        test_pred = model_pipeline.predict(test_features)
        if label_encoder is not None:
            test_pred = label_encoder.inverse_transform(test_pred.astype(int))
else:
    test_pred = model_pipeline.predict(test_features)

submission = sample_submission.copy()
submission[target_col] = test_pred
submission.to_csv("submission_rf.csv", index=False)

print("\nSaved submission_rf.csv with shape:", submission.shape)
submission.head()


from collections import Counter

# y is your original target; y_enc is label-encoded target
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

print("Label mapping (encoded -> original):")
for i, c in enumerate(label_encoder.classes_):
    print(f"  {i} -> {c}")

# ---- 1) Check class distribution ----
class_counts = Counter(y_enc)
total = len(y_enc)

print("\nClass counts and ratios:")
for cls_id, count in class_counts.items():
    ratio = count / total
    print(f"  class {cls_id} ({label_encoder.classes_[cls_id]}): {count} ({ratio:.2%})")

minority_ratio = min(count / total for count in class_counts.values())
majority_ratio = max(count / total for count in class_counts.values())
imbalance_ratio = majority_ratio / minority_ratio
print(f"\nImbalance ratio (majority / minority): {imbalance_ratio:.2f}")

# ---- 2) Compute scale_pos_weight for binary classification ----
scale_pos_weight = 1.0  # default (no reweighting)

if len(class_counts) == 2:
    # Decide which encoded class is the "positive" class.
    # Often the minority class is the positive one.
    # Here we choose the minority class automatically:
    pos_class = min(class_counts, key=class_counts.get)  # minority class index
    n_pos = class_counts[pos_class]
    n_neg = total - n_pos
    scale_pos_weight = n_neg / n_pos

    print(f"\nAssuming encoded class {pos_class} "
          f"({label_encoder.classes_[pos_class]}) is the POSITIVE class.")
    print(f"n_neg = {n_neg}, n_pos = {n_pos}")
    print(f"Suggested scale_pos_weight: {scale_pos_weight:.3f}")
else:
    print("\nMore than two classes: scale_pos_weight is not directly used. "
          "Consider class_weight or sampling instead.")


# ======================================================
# XGBoost classification with class-imbalance handling
# - Uses class weights (sample_weight) on training data
# - Calculates all main classification metrics
# ======================================================
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

# -------------------------
# 1. Load data
# -------------------------
DATA_PATH = Path("/kaggle/input/playground-series-s5e12")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

print("Shapes:")
print("  train:", train.shape)
print("  test: ", test.shape)
print("  sample_submission:", sample_submission.shape)

# -------------------------
# 2. Infer target and ID
# -------------------------
possible_targets = [c for c in sample_submission.columns if c in train.columns]
if len(possible_targets) == 1:
    target_col = possible_targets[0]
else:
    diff_cols = [c for c in train.columns if c not in test.columns]
    if len(diff_cols) == 1:
        target_col = diff_cols[0]
    else:
        target_col = train.columns[-1]

id_candidates = [c for c in sample_submission.columns if c != target_col]
id_col = id_candidates[0] if len(id_candidates) >= 1 else None

print("\nTarget column:", target_col)
print("ID column:    ", id_col)

# -------------------------
# 3. Optionally drop some features
# -------------------------
# You asked earlier to drop: id, sleep_hours_per_day, screen_time_hours_per_day
drop_features = ["id", "sleep_hours_per_day", "screen_time_hours_per_day"]

drop_features_train = [c for c in drop_features if c in train.columns]
drop_features_test = [c for c in drop_features if c in test.columns]

train = train.drop(columns=drop_features_train)
test_features = test.drop(columns=drop_features_test)

print("\nDropped from train:", drop_features_train)
print("Train columns after drop:", len(train.columns))

# -------------------------
# 4. Split features & target
# -------------------------
y = train[target_col].astype(int)  # ensure 0/1 ints
X = train.drop(columns=[target_col])

# -------------------------
# 5. Train/validation split (stratified)
# -------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTrain/Valid shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

# -------------------------
# 6. Check class imbalance and build class weights
# -------------------------
print("\nClass distribution in full data:")
print(y.value_counts(normalize=False).to_frame("count"))
print("\nClass distribution (percentage):")
print((y.value_counts(normalize=True) * 100).to_frame("percent"))

# Compute class weights on TRAIN ONLY
unique, counts = np.unique(y_train, return_counts=True)
class_counts = dict(zip(unique, counts))
n_samples = len(y_train)
n_classes = len(unique)

class_weights = {
    cls: n_samples / (n_classes * count)
    for cls, count in class_counts.items()
}

print("\nClass counts in TRAIN:", class_counts)
print("Computed class weights:", class_weights)

# Create sample_weight array for training samples
sample_weight_train = np.array([class_weights[cls] for cls in y_train])

# -------------------------
# 7. Preprocessing
# -------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nNumeric features:", len(num_cols))
print("Categorical features:", len(cat_cols))

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# -------------------------
# 8. XGBoost classifier (binary)
# -------------------------
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_model),
])

# -------------------------
# 9. Fit model with sample weights (to handle imbalance)
# -------------------------
model_pipeline.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weight_train
)

# -------------------------
# 10. Evaluate on validation set
# -------------------------
y_valid_proba = model_pipeline.predict_proba(X_valid)[:, 1]  # prob of class 1
y_valid_pred = (y_valid_proba >= 0.5).astype(int)

acc = accuracy_score(y_valid, y_valid_pred)
auc = roc_auc_score(y_valid, y_valid_proba)

print("\n=== Validation metrics (with class weights) ===")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC:  {auc:.4f}\n")

print("Classification report:")
print(classification_report(y_valid, y_valid_pred))

cm = confusion_matrix(y_valid, y_valid_pred, labels=[0, 1])
print("Confusion matrix (rows=true, cols=pred):\n", cm)

# Optional: plot confusion matrix
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Validation)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1])
plt.yticks(tick_marks, [0, 1])
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# Optional: ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_valid, y_valid_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 11. Train on full data & predict test for Kaggle
# -------------------------
print("\nFitting on FULL training data with class weights...")

# Recompute class weights on full data for final model (optional but consistent)
unique_full, counts_full = np.unique(y, return_counts=True)
class_counts_full = dict(zip(unique_full, counts_full))
n_samples_full = len(y)
n_classes_full = len(unique_full)
class_weights_full = {
    cls: n_samples_full / (n_classes_full * count)
    for cls, count in class_counts_full.items()
}
sample_weight_full = np.array([class_weights_full[cls] for cls in y])

model_pipeline.fit(
    X,
    y,
    model__sample_weight=sample_weight_full
)

# Predict probability of class 1 for test data
test_pred_proba = model_pipeline.predict_proba(test_features)[:, 1]

submission = sample_submission.copy()
submission[target_col] = test_pred_proba
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv with shape:", submission.shape)
submission.head()


# ======================================================
# XGBoost classification + CLASS IMBALANCE CHECK + HANDLING
# - Drops: id, sleep_hours_per_day, screen_time_hours_per_day
# - Uses class weights (sample_weight) to handle imbalance
# ======================================================
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from xgboost import XGBClassifier

# -------------------------
# 1. Load data
# -------------------------
DATA_PATH = Path("/kaggle/input/playground-series-s5e12")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

print("Shapes:")
print("  train:", train.shape)
print("  test: ", test.shape)
print("  sample_submission:", sample_submission.shape)

# -------------------------
# 2. Infer target and ID
# -------------------------
possible_targets = [c for c in sample_submission.columns if c in train.columns]
if len(possible_targets) == 1:
    target_col = possible_targets[0]
else:
    diff_cols = [c for c in train.columns if c not in test.columns]
    if len(diff_cols) == 1:
        target_col = diff_cols[0]
    else:
        target_col = train.columns[-1]

id_candidates = [c for c in sample_submission.columns if c != target_col]
id_col = id_candidates[0] if len(id_candidates) >= 1 else None

print("\nTarget column:", target_col)
print("ID column:    ", id_col)

# -------------------------
# 3. Drop chosen features
# -------------------------
drop_features = ["id", "sleep_hours_per_day", "screen_time_hours_per_day"]

drop_features_train = [c for c in drop_features if c in train.columns]
drop_features_test = [c for c in drop_features if c in test.columns]

train = train.drop(columns=drop_features_train)
test_features = test.drop(columns=drop_features_test)

print("\nDropped from train:", drop_features_train)
print("Train columns after drop:", len(train.columns))

# -------------------------
# 4. Split features & target
# -------------------------
y = train[target_col].astype(int)  # ensure 0/1 ints
X = train.drop(columns=[target_col])

# -------------------------
# 5. CHECK CLASS IMBALANCE
# -------------------------
print("\n=== CLASS IMBALANCE CHECK (full data) ===")
class_counts = y.value_counts().sort_index()
class_perc = (class_counts / len(y) * 100).round(2)

print("Counts:")
print(class_counts.to_frame("count"))
print("\nPercentages:")
print(class_perc.to_frame("percent"))

minority = class_counts.min()
majority = class_counts.max()
imbalance_ratio = majority / minority
print(f"\nImbalance ratio (majority / minority): {imbalance_ratio:.2f}")

# Simple bar plot of class distribution
plt.figure(figsize=(4, 3))
plt.bar(class_counts.index.astype(str), class_counts.values, color=["tab:blue", "tab:orange"])
plt.title("Class Distribution (Full Data)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -------------------------
# 6. Train/validation split (stratified)
# -------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTrain/Valid shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

# -------------------------
# 7. Compute CLASS WEIGHTS on TRAIN (for imbalance handling)
# -------------------------
print("\n=== CLASS WEIGHTS (on training set) ===")
unique, counts = np.unique(y_train, return_counts=True)
class_counts_train = dict(zip(unique, counts))
n_samples = len(y_train)
n_classes = len(unique)

# Inverse-frequency weights: w_c = N / (K * n_c)
class_weights = {
    cls: n_samples / (n_classes * count)
    for cls, count in class_counts_train.items()
}

print("Class counts train:", class_counts_train)
print("Class weights:", {k: round(v, 3) for k, v in class_weights.items()})

# Build sample_weight array for each training sample
sample_weight_train = np.array([class_weights[cls] for cls in y_train])

# -------------------------
# 8. Preprocessing
# -------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nNumeric features:", len(num_cols))
print("Categorical features:", len(cat_cols))

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# -------------------------
# 9. XGBoost classifier (binary)
# -------------------------
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_model),
])

# -------------------------
# 10. Fit model WITH CLASS WEIGHTS
# -------------------------
model_pipeline.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weight_train
)

# -------------------------
# 11. Evaluate on validation set
# -------------------------
y_valid_proba = model_pipeline.predict_proba(X_valid)[:, 1]  # prob of class 1
y_valid_pred = (y_valid_proba >= 0.5).astype(int)

acc = accuracy_score(y_valid, y_valid_pred)
auc = roc_auc_score(y_valid, y_valid_proba)

print("\n=== VALIDATION METRICS (with class weights) ===")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC:  {auc:.4f}\n")

print("Classification report:")
print(classification_report(y_valid, y_valid_pred))

cm = confusion_matrix(y_valid, y_valid_pred, labels=[0, 1])
print("Confusion matrix (rows=true, cols=pred):\n", cm)

# Confusion matrix plot
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Validation)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1])
plt.yticks(tick_marks, [0, 1])
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_valid, y_valid_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# 12. Train on FULL data (with class weights) & predict test
# -------------------------
print("\nFitting on FULL training data with class weights...")

unique_full, counts_full = np.unique(y, return_counts=True)
class_counts_full = dict(zip(unique_full, counts_full))
n_samples_full = len(y)
n_classes_full = len(unique_full)

class_weights_full = {
    cls: n_samples_full / (n_classes_full * count)
    for cls, count in class_counts_full.items()
}
sample_weight_full = np.array([class_weights_full[cls] for cls in y])

print("Full-data class counts:", class_counts_full)
print("Full-data class weights:", {k: round(v, 3) for k, v in class_weights_full.items()})

model_pipeline.fit(
    X,
    y,
    model__sample_weight=sample_weight_full
)

# Predict probability of class 1 for test data
test_pred_proba = model_pipeline.predict_proba(test_features)[:, 1]

submission = sample_submission.copy()
submission[target_col] = test_pred_proba
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv with shape:", submission.shape)
submission.head()


# ======================================================
# Basic hyperparameter tuning for XGBoost (classification)
# using RandomizedSearchCV
# ======================================================
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from xgboost import XGBClassifier

# -------------------------
# 1. Load data
# -------------------------
DATA_PATH = Path("/kaggle/input/playground-series-s5e12")

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

# -------------------------
# 2. Infer target and ID
# -------------------------
possible_targets = [c for c in sample_submission.columns if c in train.columns]
if len(possible_targets) == 1:
    target_col = possible_targets[0]
else:
    diff_cols = [c for c in train.columns if c not in test.columns]
    if len(diff_cols) == 1:
        target_col = diff_cols[0]
    else:
        target_col = train.columns[-1]

id_candidates = [c for c in sample_submission.columns if c != target_col]
id_col = id_candidates[0] if len(id_candidates) >= 1 else None

print("Target column:", target_col)
print("ID column:    ", id_col)

# -------------------------
# 3. Optionally drop some features
# -------------------------
drop_features = ["id", "sleep_hours_per_day", "screen_time_hours_per_day"]

drop_features_train = [c for c in drop_features if c in train.columns]
drop_features_test = [c for c in drop_features if c in test.columns]

train = train.drop(columns=drop_features_train)
test_features = test.drop(columns=drop_features_test)

print("Dropped from train:", drop_features_train)

# -------------------------
# 4. Split features & target
# -------------------------
y = train[target_col].astype(int)   # ensure 0/1 ints
X = train.drop(columns=[target_col])

# -------------------------
# 5. Train/validation split
# -------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTrain/Valid shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

# -------------------------
# 6. Preprocessing
# -------------------------
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\nNumeric features:", len(num_cols))
print("Categorical features:", len(cat_cols))

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# -------------------------
# 7. Base XGBoost classifier (no heavy tuning yet)
# -------------------------
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb),
])

# -------------------------
# 8. Hyperparameter search space
# -------------------------
param_distributions = {
    "model__n_estimators": [300, 600, 900, 1200],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__max_depth": [3, 4, 5, 6],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__gamma": [0, 0.1, 0.2, 0.3],
}

# -------------------------
# 9. RandomizedSearchCV (tunes hyperparams on training set)
# -------------------------
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_distributions,
    n_iter=20,                # reduce to 10 if it's too slow
    scoring="accuracy",       # change to "roc_auc" if you care about AUC
    cv=3,                     # 3-fold CV on X_train
    verbose=1,
    n_jobs=-1,
    random_state=42,
)

print("\nStarting RandomizedSearchCV...")
search.fit(X_train, y_train)

print("\nBest CV accuracy:", search.best_score_)
print("Best hyperparameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

best_model = search.best_estimator_

# -------------------------
# 10. Evaluate best model on validation set
# -------------------------
y_valid_proba = best_model.predict_proba(X_valid)[:, 1]
y_valid_pred = (y_valid_proba >= 0.5).astype(int)

acc = accuracy_score(y_valid, y_valid_pred)
auc = roc_auc_score(y_valid, y_valid_proba)

print("\nValidation performance of best model:")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC:  {auc:.4f}\n")

print("Classification report:")
print(classification_report(y_valid, y_valid_pred))

# -------------------------
# 11. Fit best model on FULL data & create submission
# -------------------------
print("\nFitting best model on FULL training data and creating submission...")
best_model.fit(X, y)

test_pred_proba = best_model.predict_proba(test_features)[:, 1]

submission = sample_submission.copy()
submission[target_col] = test_pred_proba
submission.to_csv("submission_xgb_tuned.csv", index=False)

print("Saved submission_xgb_tuned.csv with shape:", submission.shape)
submission.head()




