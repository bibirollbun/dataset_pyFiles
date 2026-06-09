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
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

TARGET = 'Calories'

# Drop ID if exists
if 'ID' in train.columns:
    train.drop('ID', axis=1, inplace=True)
    test.drop('ID', axis=1, inplace=True)

# Label encode categoricals
cat_cols = train.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Log1p target transform
train[TARGET] = np.log1p(train[TARGET])

# Quantile normalization
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X = train.drop(TARGET, axis=1)
y = train[TARGET]
X_norm = pd.DataFrame(qt.fit_transform(X), columns=X.columns)
X_test_norm = pd.DataFrame(qt.transform(test), columns=X.columns)

# Prepare stacking containers
meta_features = np.zeros((len(train), 3))
meta_test = np.zeros((len(test), 3))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop over folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X_norm)):
    print(f"Fold {fold+1}")
    X_train, y_train = X_norm.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X_norm.iloc[val_idx], y.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6,
                                   subsample=0.8, colsample_bytree=0.8, device='gpu')
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100)])
    meta_features[val_idx, 0] = lgb_model.predict(X_val)
    meta_test[:, 0] += lgb_model.predict(X_test_norm) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8,
                                  tree_method='gpu_hist', predictor='gpu_predictor')
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100, verbose=False)
    meta_features[val_idx, 1] = xgb_model.predict(X_val)
    meta_test[:, 1] += xgb_model.predict(X_test_norm) / kf.n_splits

    # CatBoost
    cb_model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6,
                                     task_type='GPU', verbose=0)
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    meta_features[val_idx, 2] = cb_model.predict(X_val)
    meta_test[:, 2] += cb_model.predict(X_test_norm) / kf.n_splits

# Meta-model
meta_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=3)
meta_model.fit(meta_features, y)
stacked_preds = meta_model.predict(meta_test)

# Inverse transform
final_preds = np.expm1(stacked_preds)
submission[TARGET] = final_preds
submission.to_csv('submission_stacked.csv', index=False)

# OOF RMSLE
oof_preds = meta_model.predict(meta_features)
oof_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
print(f"OOF RMSLE: {oof_rmsle:.5f}")


