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


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df.columns


df.head()


# ================================================================
# xgb_final_submission_with_fe_and_es_fixed.py
# Clean XGBoost (v3.x compatible) + early stopping + feature engineering + submission
# ================================================================

import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from xgboost.callback import EarlyStopping

# ==============================
# 0) Diagnostics
# ==============================
print(f"{xgb.__version__=}")

# ==============================
# 1) Load data
# ==============================
base = Path("/kaggle") if Path("/kaggle").exists() else Path("kaggle")
train_path = base / "input/playground-series-s5e11/train.csv"
test_path = base / "input/playground-series-s5e11/test.csv"

df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
df["loan_paid_back"] = df["loan_paid_back"].astype(int)

# ==============================
# 2) Define base features
# ==============================
drop_cols = ["id"]
num_base = [
    "annual_income", "debt_to_income_ratio", "credit_score",
    "loan_amount", "interest_rate"
]
cat_cols = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade",
]
target = "loan_paid_back"

X = df.drop(columns=[target] + drop_cols)
y = df[target]
X = X[num_base + cat_cols]
X_test_raw = test_df[num_base + cat_cols]

# ==============================
# 3) Feature engineering
# ==============================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        income = X["annual_income"].replace(0, np.nan)
        X["loan_to_income_ratio"] = X["loan_amount"] / income
        X["loan_amount_log"] = np.log1p(X["loan_amount"].clip(lower=0))
        X["income_log"] = np.log1p(X["annual_income"].clip(lower=0))
        X["dti_x_interest"] = X["debt_to_income_ratio"] * X["interest_rate"]
        for c in ["loan_to_income_ratio", "loan_amount_log", "income_log", "dti_x_interest"]:
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)

        numeric_grade_value = [f"{g}{s}" for g in "ABCDEFG" for s in "12345"]
        rank = {k:i for i,k in enumerate(numeric_grade_value)}
        X["grade_numeric"] = X["grade_subgrade"].map(rank)

        X["credit_to_income"] = X["credit_score"] / (X["annual_income"] / 1000)

        X["grade_is_top"] = (X["grade_numeric"] < 10).astype(int)
        X["grade_bucket"] = (X["grade_numeric"] // 5)

        return X

num_all = num_base + [
    "loan_to_income_ratio", "loan_amount_log", "income_log", "dti_x_interest", "grade_numeric",
    "credit_to_income", "grade_is_top", "grade_bucket"
]

# ==============================
# 4) Preprocessing
# ==============================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_all),
    ("cat", categorical_transformer, cat_cols),
])

# ==============================
# 5) Split for early stopping
# ==============================
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# Apply FE + preprocessing manually
fe = FeatureEngineer()
X_tr_fe = fe.fit_transform(X_tr)
X_val_fe = fe.transform(X_val)

prep = preprocessor.fit(X_tr_fe, y_tr)
X_tr_prep = prep.transform(X_tr_fe)
X_val_prep = prep.transform(X_val_fe)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_tr_prep, label=y_tr)
dval = xgb.DMatrix(X_val_prep, label=y_val)

# ==============================
# 6) XGBoost parameters and training with early stopping
# ==============================
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",  # "gpu_hist" if GPU available
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "gamma": 0.1,
    "random_state": 42,
}

print("Training with early stopping (XGBoost 3.x style)...")
evals = [(dtrain, "train"), (dval, "validation")]
es = EarlyStopping(rounds=100, save_best=True, maximize=True)

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=5000,
    evals=evals,
    callbacks=[es],
    verbose_eval=100,
)

print(f"Best iteration: {bst.best_iteration}")
print(f"Best AUC: {bst.best_score}")

# ==============================
# 7) Retrain on full data with best iteration
# ==============================
print("Retraining on full data with best iteration...")
X_all_fe = fe.transform(X)
# Use the same preprocessor already fit on training data to avoid leakage
X_all_prep = prep.transform(X_all_fe)
dall = xgb.DMatrix(X_all_prep, label=y)

bst_final = xgb.train(
    params=params,
    dtrain=dall,
    num_boost_round=bst.best_iteration + 1
)

# ==============================
# 8) Predict test set and save submission
# ==============================
print("Predicting on test data...")
X_test_fe = fe.transform(X_test_raw)
X_test_prep = preprocessor.transform(X_test_fe)
dtest = xgb.DMatrix(X_test_prep)

test_proba = bst_final.predict(dtest)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
sub = pd.DataFrame({
    "id": test_df["id"],
    "loan_paid_back": test_proba
})
out_path = f"submission_{timestamp}.csv"
sub.to_csv(out_path, index=False)
print(f"Submission file saved as {out_path=}")
print(sub.head())

# Also save on normal filename.
print(f"Submission file also saved as submission.csv")
sub.to_csv('submission.csv', index=False)


sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
sub = pd.read_csv("submission.csv")

print(sample.shape, sub.shape)
print(sample.columns, sub.columns)




