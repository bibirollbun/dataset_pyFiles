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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Load cleaned data
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X = train.drop("diagnosed_diabetes", axis=1)
y = train["diagnosed_diabetes"]

# Convert categorical columns to category codes (fast)
categorical_cols = X.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype("category").cat.codes
    test[col] = test[col].astype("category").cat.codes

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="auc",
    use_label_encoder=False
)
xgb_model.fit(X_tr, y_tr)
xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(X_tr, y_tr)
lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]

# Ensemble
val_pred_ensemble = (xgb_val_pred + lgb_val_pred) / 2
auc_score = roc_auc_score(y_val, val_pred_ensemble)
print("Validation AUC (Ensemble, Fast):", auc_score)

# Predict on test
xgb_test_pred = xgb_model.predict_proba(test)[:, 1]
lgb_test_pred = lgb_model.predict_proba(test)[:, 1]
test_pred_ensemble = (xgb_test_pred + lgb_test_pred) / 2

submission = pd.DataFrame({"id": test["id"], "diagnosed_diabetes": test_pred_ensemble})
submission.to_csv("submission_fast_ensemble.csv", index=False)
print("Submission saved as submission_fast_ensemble.csv")



submission = pd.read_csv("submission_fast_ensemble.csv")
print(submission.head())
print(submission.columns)
print(submission.info())



submission["diagnosed_diabetes"] = submission["diagnosed_diabetes"].clip(0, 1)



submission.to_csv("submission.csv", index=False)





