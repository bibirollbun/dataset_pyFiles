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
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

X = train.drop(columns=['id', 'diagnosed_diabetes'])
y = train['diagnosed_diabetes']
X_test = test.drop(columns=['id'])

# Encode categoricals
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# Feature Engineering (Interactions)
X['smoke_income'] = X['smoking_status'] * X['income_level']
X['edu_income'] = X['education_level'] * X['income_level']

X_test['smoke_income'] = X_test['smoking_status'] * X_test['income_level']
X_test['edu_income'] = X_test['education_level'] * X_test['income_level']

# CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))

# LightGBM
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.02,
    'num_leaves': 90,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 4,
    'lambda_l1': 0.3,
    'lambda_l2': 0.3,
    'min_data_in_leaf': 25,
    'verbosity': -1,
    'random_state': 42
}

# XGBoost
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.02,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'seed': 42
}

for fold, (tr, va) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1} ---")

    # LightGBM Train
    dtr = lgb.Dataset(X.iloc[tr], y.iloc[tr], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[va], y.iloc[va], reference=dtr, categorical_feature=cat_cols)
    model_lgb = lgb.train(
        lgb_params, dtr, num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(250)]
    )
    oof_lgb[va] = model_lgb.predict(X.iloc[va])
    pred_lgb += model_lgb.predict(X_test) / n_splits
    print("LGB Fold AUC:", roc_auc_score(y.iloc[va], oof_lgb[va]))

    # XGBoost Train
    dtrain_x = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
    dvalid_x = xgb.DMatrix(X.iloc[va], label=y.iloc[va])
    model_xgb = xgb.train(
        xgb_params, dtrain_x, num_boost_round=3000,
        evals=[(dvalid_x, "valid")],
        early_stopping_rounds=250, verbose_eval=False
    )
    oof_xgb[va] = model_xgb.predict(dvalid_x)
    pred_xgb += model_xgb.predict(xgb.DMatrix(X_test)) / n_splits
    print("XGB Fold AUC:", roc_auc_score(y.iloc[va], oof_xgb[va]))

# Overall CV
print("\nLGB CV AUC:", roc_auc_score(y, oof_lgb))
print("XGB CV AUC:", roc_auc_score(y, oof_xgb))

# Ensemble
final_preds = (pred_lgb + pred_xgb) / 2
submission = pd.DataFrame({'id': test['id'], 'diagnosed_diabetes': final_preds})
submission.to_csv('submission.csv', index=False)
print("\nNew submission saved!")
print(submission.head())

