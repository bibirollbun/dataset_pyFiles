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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


print(df_train.head())
print("Shape:", df_train.shape)


df_train.describe()


df_train.duplicated().sum()


df_train.isna().sum()


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# =========================
# 1. Load data
# =========================
df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

# =========================
# 2. Simple feature engineering
# =========================
# Ratios and interaction features (optional but simple)
df_train["loan_to_income"] = df_train["loan_amount"] / df_train["annual_income"]
df_test["loan_to_income"] = df_test["loan_amount"] / df_test["annual_income"]

df_train["monthly_payment"] = df_train["loan_amount"] * df_train["interest_rate"] / 12
df_test["monthly_payment"] = df_test["loan_amount"] * df_test["interest_rate"] / 12

df_train["payment_to_income"] = df_train["monthly_payment"] / (df_train["annual_income"] / 12)
df_test["payment_to_income"] = df_test["monthly_payment"] / (df_test["annual_income"] / 12)

# =========================
# 3. Prepare train/valid splits
# =========================
X = df_train.drop(["loan_paid_back", "id"], axis=1)
y = df_train["loan_paid_back"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Drop id from test features too
X_test = df_test.drop("id", axis=1)

# =========================
# 4. Preprocessing
# =========================
cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

# =========================
# 5. Model (simple RandomForest)
# =========================
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", clf)
])

# =========================
# 6. Train
# =========================
print("Training model...")
model.fit(X_train, y_train)

# =========================
# 7. Evaluate
# =========================
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
y_valid_proba = model.predict_proba(X_valid)[:, 1]

print("\n" + "="*50)
print(f"Train accuracy:      {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Validation accuracy: {accuracy_score(y_valid, y_valid_pred):.4f}")
print(f"ROC-AUC:             {roc_auc_score(y_valid, y_valid_proba):.4f}")
print("="*50)
print("\nClassification report:\n")
print(classification_report(y_valid, y_valid_pred))

# =========================
# 8. Predict on test & create submission
# =========================



# If you want to submit probabilities (recommended for this comp):
y_test_proba = model.predict_proba(df_test_features)[:, 1]

submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
submission["loan_paid_back"] = y_test_proba

# Save to /kaggle/working/ (default output dir)
submission.to_csv("submission.csv", index=False)

print("submission.csv saved!")



import os
print(os.listdir("/kaggle/working"))





