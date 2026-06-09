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


train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train.head()


train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------------------
# 1ï¸�âƒ£ Load data
# ---------------------------
train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

# ---------------------------
# 2ï¸�âƒ£ Split features and target
# ---------------------------
X = train.drop("loan_paid_back", axis=1)
y = train["loan_paid_back"]

# ---------------------------
# 3ï¸�âƒ£ Encode categorical features
# ---------------------------
cat_cols = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade"
]

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = test[col].astype(str).map(lambda x: x if x in le.classes_ else "Unknown")
    le_classes = np.append(le.classes_, "Unknown")
    le.classes_ = le_classes
    test[col] = le.transform(test[col])

# ---------------------------
# 4ï¸�âƒ£ Feature Engineering
# ---------------------------
for df in [X, test]:
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1e-5)
    df["credit_to_debt_ratio"] = df["credit_score"] / (df["debt_to_income_ratio"] + 1e-5)
    df["interest_to_income_ratio"] = df["interest_rate"] / (df["annual_income"] + 1e-5)
    df["monthly_installment"] = df["loan_amount"] * (df["interest_rate"] / 12)
    df["income_to_installment_ratio"] = df["annual_income"] / (df["monthly_installment"] + 1e-5)
    df["credit_income_ratio"] = df["credit_score"] / (df["annual_income"] + 1e-5)
    df["risk_factor"] = (
        df["loan_to_income_ratio"] * 0.5 +
        df["credit_to_debt_ratio"] * 0.3 +
        df["interest_to_income_ratio"] * 0.2
    )

# ---------------------------
# 5ï¸�âƒ£ Split for validation
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X.drop("id", axis=1),
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# 6ï¸�âƒ£ Train individual models
# ---------------------------

# XGBoost
xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.06,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.3,
    min_child_weight=3,
    reg_lambda=1.2,
    reg_alpha=0.6,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

# CatBoost
cat = CatBoostClassifier(
    iterations=600,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=2,
    random_seed=42,
    verbose=False,
    cat_features=[]
)
cat.fit(X_train, y_train)

# LightGBM
lgb = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
lgb.fit(X_train, y_train)

# ---------------------------
# 7ï¸�âƒ£ Stacking (Meta Model)
# ---------------------------
val_preds = pd.DataFrame({
    "xgb": xgb.predict_proba(X_val)[:, 1],
    "cat": cat.predict_proba(X_val)[:, 1],
    "lgb": lgb.predict_proba(X_val)[:, 1],
})
meta = LogisticRegression(max_iter=1000)
meta.fit(val_preds, y_val)

# ---------------------------
# 8ï¸�âƒ£ Evaluate
# ---------------------------
val_meta_pred = meta.predict_proba(val_preds)[:, 1]
print("\nâœ… Ensemble ROC-AUC:", round(roc_auc_score(y_val, val_meta_pred), 5))

# ---------------------------
# 9ï¸�âƒ£ Predict on test
# ---------------------------
test_preds = pd.DataFrame({
    "xgb": xgb.predict_proba(test.drop("id", axis=1))[:, 1],
    "cat": cat.predict_proba(test.drop("id", axis=1))[:, 1],
    "lgb": lgb.predict_proba(test.drop("id", axis=1))[:, 1],
})

final_pred = meta.predict_proba(test_preds)[:, 1]
final_pred = final_pred.round(1)

# ---------------------------
# ğŸ”Ÿ Save submission
# ---------------------------
submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": [f"{p:.1f}" for p in final_pred]
})
submission.to_csv("submission.csv", index=False)

print("\nâœ… submission.csv created successfully with ensemble predictions!")
print(submission.head())





