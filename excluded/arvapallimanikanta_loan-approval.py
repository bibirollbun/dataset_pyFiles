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


# ============================================================
# Loan Approval Prediction — Kaggle Playground S4E10
# Full pipeline: EDA (safe) + CV model selection + submission
# Includes RuntimeWarning silencer for clean output.
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ----------------------------
# 1) Paths and loading
# ----------------------------
DATA_DIR = "/kaggle/input/playground-series-s4e10"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("Reading data from:", DATA_DIR)
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)
print("Shapes -> train:", train.shape, " test:", test.shape)

# ----------------------------------------
# 2) Infer ID and target from sample_submission
# ----------------------------------------
id_col = sample_sub.columns[0]
target_col = sample_sub.columns[1]

if target_col not in train.columns:
    raise ValueError(
        f"Target '{target_col}' (from sample_submission) not found in train.csv.\nTrain cols: {list(train.columns)}"
    )
if id_col not in test.columns:
    raise ValueError(
        f"ID '{id_col}' (from sample_submission) not found in test.csv.\nTest cols: {list(test.columns)}"
    )

print(f"ID column: {id_col} | Target column: {target_col}")

# ----------------------------------------
# 3) EDA (safe/guarded)
# ----------------------------------------
eda_train = train.copy()

# a) Missing values
missing_counts = eda_train.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
if len(missing_counts) > 0:
    plt.figure(figsize=(10, max(3, 0.3 * len(missing_counts))))
    sns.barplot(x=missing_counts.values, y=missing_counts.index)
    plt.title("Missing Values per Column (train)")
    plt.xlabel("Count"); plt.ylabel("Column")
    plt.tight_layout(); plt.show()
else:
    print("No missing values found in train.csv ✅")

# b) Target balance
if eda_train[target_col].nunique() <= 100:
    plt.figure(figsize=(5,4))
    eda_train[target_col].value_counts().plot(kind="bar")
    plt.title("Target Balance (train)")
    plt.xlabel(target_col); plt.ylabel("Count")
    plt.tight_layout(); plt.show()
else:
    print("Target has many unique values; skipping simple bar plot.")

# c) Numeric distributions (top 8)
num_cols_all = eda_train.select_dtypes(include=[np.number]).columns.tolist()
if target_col in num_cols_all:
    try: num_cols_all.remove(target_col)
    except ValueError: pass
top_num = num_cols_all[:8]
if len(top_num) > 0:
    eda_train[top_num].hist(figsize=(14, 8), bins=30)
    plt.suptitle("Numeric Feature Distributions (Top 8)", y=1.02)
    plt.tight_layout(); plt.show()
else:
    print("No numeric features to plot.")

# d) Categorical counts (top few)
cat_cols_all = eda_train.select_dtypes(include=["object", "category"]).columns.tolist()
if id_col in cat_cols_all:
    cat_cols_all.remove(id_col)
if len(cat_cols_all) > 0:
    top_cat = sorted(cat_cols_all, key=lambda c: eda_train[c].nunique())[:6]
    for c in top_cat:
        plt.figure(figsize=(7,4))
        eda_train[c].value_counts().head(10).plot(kind="bar")
        plt.title(f"Top 10 Values in {c}")
        plt.xlabel(c); plt.ylabel("Count")
        plt.tight_layout(); plt.show()
else:
    print("No categorical features to plot.")

# e) Correlation heatmap (numeric)
num_for_corr = eda_train.select_dtypes(include=[np.number])
if num_for_corr.shape[1] >= 2:
    corr = num_for_corr.corr()
    # Replace potential NaNs (constant columns) for clean plotting
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Columns)")
    plt.tight_layout(); plt.show()
else:
    print("Not enough numeric columns for correlation heatmap.")

# ----------------------------------------
# 4) Prepare data
# ----------------------------------------
y = train[target_col]
X = train.drop(columns=[target_col])

# Optional: drop leakage columns if any (none by default)
leak_like_cols = []
X = X.drop(columns=[c for c in leak_like_cols if c in X.columns], errors="ignore")

# Column types
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Remove ID from features if present
if id_col in cat_cols: cat_cols.remove(id_col)
if id_col in num_cols: num_cols.remove(id_col)

print("Numeric cols:", len(num_cols))
print("Categorical cols:", len(cat_cols))

# ----------------------------------------
# 5) Preprocessing
# ----------------------------------------
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

# Metric
is_binary = (pd.Series(y).nunique() == 2)
scoring = "roc_auc" if is_binary else "accuracy"
print("Scoring metric:", scoring)

# ----------------------------------------
# 6) Models
# ----------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        objective="binary:logistic" if is_binary else "multi:softprob",
        eval_metric="auc" if is_binary else "mlogloss",
        tree_method="hist"
    )
}

# ----------------------------------------
# 7) Cross-validation model selection
# ----------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}

print("Running 5-fold CV...")
for name, est in models.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", est)])
    try:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_scores[name] = (scores.mean(), scores.std())
        print(f"{name}: {scoring} = {scores.mean():.5f} ± {scores.std():.5f}")
    except Exception as e:
        print(f"{name} CV failed: {e}")

if not cv_scores:
    raise RuntimeError("All CV runs failed. Check data or simplify models.")

best_model_name = max(cv_scores, key=lambda k: cv_scores[k][0])
print("\nBest by CV:", best_model_name, "->", scoring, f"{cv_scores[best_model_name][0]:.5f}")

# ----------------------------------------
# 8) Fit best model on full training
# ----------------------------------------
best_est = models[best_model_name]
final_pipe = Pipeline(steps=[("prep", preprocessor), ("model", best_est)])
print("Fitting best model on full train...")
final_pipe.fit(X, y)

# ----------------------------------------
# 9) Holdout evaluation + plots
# ----------------------------------------
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
final_pipe.fit(X_tr, y_tr)
va_pred = final_pipe.predict(X_va)

# Confusion matrix
try:
    cm = confusion_matrix(y_va, va_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()
except Exception as e:
    print("Confusion matrix skipped:", e)

# ROC (binary only)
if is_binary:
    try:
        va_proba = final_pipe.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, va_proba)
        print(f"Holdout ROC-AUC: {auc:.5f}")
        RocCurveDisplay.from_predictions(y_va, va_proba)
        plt.title(f"ROC Curve - {best_model_name}")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("ROC curve skipped:", e)
else:
    try:
        acc = accuracy_score(y_va, va_pred)
        print(f"Holdout Accuracy: {acc:.5f}")
    except Exception as e:
        print("Holdout metric skipped:", e)

# ----------------------------------------
# 10) Feature importance (tree models)
# ----------------------------------------
def plot_feature_importance_from_pipeline(pipe, top_n=20, title="Feature Importance"):
    try:
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]

        # numeric feature names
        num_names = []
        if "num" in prep.named_transformers_:
            num_names = list(prep.named_transformers_["num"].feature_names_in_) \
                        if hasattr(prep.named_transformers_["num"], "feature_names_in_") else list(prep.transformers_[0][2])

        # categorical ohe names
        cat_names = []
        if "cat" in prep.named_transformers_:
            cat_pipe = prep.named_transformers_["cat"]
            if hasattr(cat_pipe.named_steps["ohe"], "get_feature_names_out"):
                cat_cols_local = prep.transformers_[1][2]
                cat_names = cat_pipe.named_steps["ohe"].get_feature_names_out(cat_cols_local).tolist()

        feature_names = num_names + cat_names

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "get_booster"):
            importances = model.feature_importances_
        else:
            print("Model does not provide feature_importances_.")
            return

        n = min(len(feature_names), len(importances))
        feature_names = feature_names[:n]; importances = importances[:n]

        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False).head(top_n)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=fi, x="importance", y="feature")
        plt.title(title + f" (Top {top_n})")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("Feature importance skipped:", e)

if best_model_name in ["RandomForest", "XGBoost"]:
    print("Plotting feature importance…")
    final_pipe.fit(X, y)  # ensure fitted on full data for names
    plot_feature_importance_from_pipeline(final_pipe, top_n=20, title=f"{best_model_name} Feature Importance")
else:
    print("Best model is not tree-based; skipping feature importance.")

# ----------------------------------------
# 11) Predict test & write submission
# ----------------------------------------
print("Predicting test labels...")
X_test = test.copy()
X_test = X_test.drop(columns=[c for c in leak_like_cols if c in X_test.columns], errors="ignore")

test_preds = final_pipe.predict(X_test)
submission = sample_sub.copy()
submission[target_col] = test_preds

# Match dtype if needed
try:
    submission[target_col] = submission[target_col].astype(sample_sub[target_col].dtype)
except: pass

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(f"\n✅ Saved: {out_path}")
display(submission.head())

print("\nDone. Submit 'submission.csv' on the competition page.")



!ls /kaggle/input


