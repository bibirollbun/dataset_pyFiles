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
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
import optuna
import warnings
warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Feature engineering
def preprocess(df):
    df = df.copy()
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    df['Duration_Heart'] = df['Duration'] * df['Heart_Rate']
    df['Duration_BMI'] = df['Duration'] * df['BMI']
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    return df

train = preprocess(train)
test = preprocess(test)

features = [col for col in train.columns if col not in ['id', 'Calories']]
X = train[features]
y = np.log1p(train['Calories'])
X_test = test[features]

# Stacking arrays
oof_lgb = np.zeros(len(X))
preds_lgb = np.zeros(len(X_test))

oof_xgb = np.zeros(len(X))
preds_xgb = np.zeros(len(X_test))

oof_cat = np.zeros(len(X))
preds_cat = np.zeros(len(X_test))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])

    oof_lgb[val_idx] = model.predict(X_val)
    preds_lgb += model.predict(X_test) / kf.n_splits

# XGBoost
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    oof_xgb[val_idx] = model.predict(X_val)
    preds_xgb += model.predict(X_test) / kf.n_splits

# CatBoost
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(n_estimators=1000, learning_rate=0.03, depth=6, random_seed=42, verbose=0, task_type="GPU")
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    oof_cat[val_idx] = model.predict(X_val)
    preds_cat += model.predict(X_test) / kf.n_splits

# Meta model (Ridge)
meta_train = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
meta_test = np.vstack([preds_lgb, preds_xgb, preds_cat]).T

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_train, y)
final_preds_log = meta_model.predict(meta_test)
final_preds = np.expm1(final_preds_log)
final_preds = np.clip(final_preds, 0, None)  # Remove negatives if any

# Save submission
sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub['Calories'] = final_preds
sub.to_csv('submission_stacked.csv', index=False)

# Print CV RMSLE
oof_pred = meta_model.predict(meta_train)
oof_rmsle = mean_squared_log_error(np.expm1(y), np.expm1(oof_pred)) ** 0.5
print(f"OOF RMSLE (stacked): {oof_rmsle:.5f}")


