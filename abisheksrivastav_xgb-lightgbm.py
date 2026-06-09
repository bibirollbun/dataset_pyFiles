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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

TARGET = "y"
ID_COL = "id"

# Feature engineering
def feature_engineering(df):
    df["balance_to_duration"] = df["balance"] / (df["duration"] + 1)
    df["campaign_per_previous"] = df["campaign"] / (df["previous"] + 1)
    df["pdays_missing"] = (df["pdays"] == -1).astype(int)
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# Encode categorical features
cat_cols = train.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Features
features = [col for col in train.columns if col not in [ID_COL, TARGET]]
X = train[features]
y = train[TARGET]
X_test = test[features]

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold predictions
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\nðŸŸ© Fold {fold + 1}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        
        n_estimators=20000,
        learning_rate=0.06,
        num_leaves=100,
        max_depth=10,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.79,
        reg_lambda=3.0,
        max_bin=4523,
        random_state=42,
        verbosity=-1,
        
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=100)
        ]
    )
    lgb_oof = lgb_model.predict_proba(X_valid)[:, 1]
    lgb_test = lgb_model.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_estimators=1000,
        learning_rate=0.03,
        tree_method="gpu_hist",
        use_label_encoder=False,
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=100
    )
    xgb_oof = xgb_model.predict_proba(X_valid)[:, 1]
    xgb_test = xgb_model.predict_proba(X_test)[:, 1]

    # Average predictions
    oof_preds[valid_idx] = (lgb_oof + xgb_oof) / 2
    test_preds += (lgb_test + xgb_test) / 2 / skf.n_splits

# Evaluate
auc = roc_auc_score(y, oof_preds)
print(f"\nâœ… CV ROC AUC Score: {auc:.5f}")

# Submission
submission[TARGET] = test_preds
submission.to_csv("submission_xgb_lgb.csv", index=False)


