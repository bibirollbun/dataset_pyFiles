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


# ------------------------------
# Load datasets
# ------------------------------
train_path = "/kaggle/input/playground-series-s5e11/train.csv"
test_path = "/kaggle/input/playground-series-s5e11/test.csv"
sample_path = "/kaggle/input/playground-series-s5e11/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

# ------------------------------
# Quick inspection
# ------------------------------
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_sub.shape)

train.head()



# ------------------------------
# Missing value summary for train
# ------------------------------
print("ğŸ”� Missing Values in TRAIN:")
missing_train = (train.isnull().mean() * 100).sort_values(ascending=False)
print(missing_train[missing_train > 0].round(2))

# ------------------------------
# Missing value summary for test
# ------------------------------
print("\nğŸ”� Missing Values in TEST:")
missing_test = (test.isnull().mean() * 100).sort_values(ascending=False)
print(missing_test[missing_test > 0].round(2))



# -----------------------------
# Numerical & Categorical columns
# -----------------------------

# TRAIN
num_cols_train = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols_train = train.select_dtypes(include=["object", "category"]).columns.tolist()

print("TRAIN Numerical Columns:", num_cols_train)
print("TRAIN Categorical Columns:", cat_cols_train)


# TEST
num_cols_test = test.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols_test = test.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nTEST Numerical Columns:", num_cols_test)
print("TEST Categorical Columns:", cat_cols_test)



# Drop ID column safely (only if it exists)

if "id" in train.columns:
    train = train.drop(columns=["id"])

print("ID column removed successfully.")



!pip install cupy-cuda12x cuml-cu12 --quiet


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

# ---------------------------
# 1. DV and Features
# ---------------------------
y = train["loan_paid_back"]
X = train.drop(columns=["loan_paid_back"])

# ---------------------------
# 2. Identify Categorical Columns (by index)
# ---------------------------
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
cat_indices = [X.columns.get_loc(col) for col in cat_cols]

# ---------------------------
# 3. Train/Validation Split
# ---------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. CatBoost Model (GPU)
# ---------------------------
cat_model = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    task_type="GPU",   # enable GPU
    verbose=50
)

# ---------------------------
# 5. Fit Model
# ---------------------------
cat_model.fit(
    X_train, y_train,
    cat_features=cat_indices,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

# ---------------------------
# 6. Predict Probabilities
# ---------------------------
y_pred_proba = cat_model.predict_proba(X_valid)[:, 1]

# ---------------------------
# 7. Evaluate AUC
# ---------------------------
auc = roc_auc_score(y_valid, y_pred_proba)
print(f"GPU CatBoost AUC: {auc:.5f}")



# Make sure 'id' column is preserved for submission
test_ids = test["id"]
X_test = test.drop(columns=["id"])

# ---------------------------
# 2. Predict probabilities
# ---------------------------
# Reuse the trained pipeline gpu_logreg
test_pred_proba = cat_model.predict_proba(X_test)[:, 1]

# ---------------------------
# 3. Create submission DataFrame
# ---------------------------
submission = pd.DataFrame({
    "id": test_ids,
    "loan_paid_back": test_pred_proba
})



# ---------------------------
# 4. Save to CSV
# ---------------------------
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

