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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor



train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print("Train:", train.shape)
print("Test:", test.shape)



# Drop ID column temporarily
train_id = train['id']
test_id = test['id']
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

# Identify categorical columns
cat_cols = train.select_dtypes(include='object').columns

# Label encode categorical columns (for all 3 models)
le = LabelEncoder()
for col in cat_cols:
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Fill any missing values
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)



X = train.drop('accident_risk', axis=1)
y = train['accident_risk']



kf = KFold(n_splits=5, shuffle=True, random_state=42)



xgb_preds = np.zeros(len(test))
lgb_preds = np.zeros(len(test))
cat_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )
    xgb_model.fit(X_train, y_train)
    xgb_oof = xgb_model.predict(X_val)
    xgb_rmse = mean_squared_error(y_val, xgb_oof, squared=False)
    print("XGBoost RMSE:", xgb_rmse)
    xgb_preds += xgb_model.predict(test) / kf.n_splits

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    lgb_oof = lgb_model.predict(X_val)
    lgb_rmse = mean_squared_error(y_val, lgb_oof, squared=False)
    print("LightGBM RMSE:", lgb_rmse)
    lgb_preds += lgb_model.predict(test) / kf.n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        verbose=0,
        random_seed=42
    )
    cat_model.fit(X_train, y_train)
    cat_oof = cat_model.predict(X_val)
    cat_rmse = mean_squared_error(y_val, cat_oof, squared=False)
    print("CatBoost RMSE:", cat_rmse)
    cat_preds += cat_model.predict(test) / kf.n_splits



final_pred = (0.4 * xgb_preds) + (0.35 * lgb_preds) + (0.25 * cat_preds)



submission = pd.DataFrame({
    'id': test_id,
    'accident_risk': np.clip(final_pred, 0, 1)  # ensure values between 0 and 1
})
submission.to_csv('submission.csv', index=False)
print("submission.csv ready for upload!")





