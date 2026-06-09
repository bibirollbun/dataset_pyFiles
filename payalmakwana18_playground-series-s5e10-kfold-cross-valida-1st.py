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
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train['curv_speed'] = train['curvature'] * train['speed_limit']
test['curv_speed'] = test['curvature'] * test['speed_limit']
train['lane_speed'] = train['num_lanes'] * train['speed_limit']
test['lane_speed'] = test['num_lanes'] * test['speed_limit']
for col in ['curvature', 'num_reported_accidents']:
    train[col + '_log'] = np.log1p(train[col])
    test[col + '_log'] = np.log1p(test[col])


def target_encode(train_df, test_df, col, target='accident_risk'):
    means = train_df.groupby(col)[target].mean()
    train_df[col + '_te'] = train_df[col].map(means)
    test_df[col + '_te'] = test_df[col].map(means).fillna(train_df[target].mean())
    return train_df, test_df

for cat in ['road_type', 'weather', 'lighting', 'time_of_day']:
    train, test = target_encode(train, test, cat)


drop_cols = ['id', 'accident_risk', 'road_type', 'weather', 'lighting', 'time_of_day']
X = train.drop(columns=drop_cols)
y = train['accident_risk']
X_test = test[X.columns]


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


# XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.77,
    'colsample_bytree': 0.78,
    'min_child_weight': 5,
    'random_state': 42,
    'tree_method': 'hist',
    'enable_categorical': True,
    'early_stopping_rounds': 200
}


# LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.015,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 2.0,
    'reg_alpha': 1.0,
    'random_state': 42,
    'verbose': -1
}


cat_params = {
    'loss_function': 'RMSE',
    'iterations': 2000,
    'learning_rate': 0.019,
    'depth': 7,
    'random_seed': 42,
    'verbose': False
}


# Out-of-fold and prediction arrays
xgb_oof, xgb_preds = np.zeros(len(X)), np.zeros(len(X_test))
lgb_oof, lgb_preds = np.zeros(len(X)), np.zeros(len(X_test))
cat_oof, cat_preds = np.zeros(len(X)), np.zeros(len(X_test))


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}")

    # Split data
    X_train_, X_val_ = X.iloc[train_idx], X.iloc[val_idx]
    y_train_, y_val_ = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_, y_train_,
                  eval_set=[(X_val_, y_val_)],
                  verbose=0)
    xgb_oof[val_idx] = xgb_model.predict(X_val_)
    xgb_preds += xgb_model.predict(X_test) / n_splits

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_, y_train_,
                  eval_set=[(X_val_, y_val_)],
                  callbacks=[lgb.early_stopping(stopping_rounds=200)])
    lgb_oof[val_idx] = lgb_model.predict(X_val_)
    lgb_preds += lgb_model.predict(X_test) / n_splits

    # CatBoost
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train_, y_train_,
                  eval_set=(X_val_, y_val_),
                  use_best_model=True)
    cat_oof[val_idx] = cat_model.predict(X_val_)
    cat_preds += cat_model.predict(X_test) / n_splits


print("XGBoost CV RMSE:", np.sqrt(mean_squared_error(y, xgb_oof)))
print("LightGBM CV RMSE:", np.sqrt(mean_squared_error(y, lgb_oof)))
print("CatBoost CV RMSE:", np.sqrt(mean_squared_error(y, cat_oof)))


# Stacking (meta-model)
meta_features_train = np.vstack([xgb_oof, lgb_oof, cat_oof]).T
meta_features_test = np.vstack([xgb_preds, lgb_preds, cat_preds]).T
meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
meta_model.fit(meta_features_train, y)
stacked_preds = meta_model.predict(meta_features_test)


print(f"RidgeCV Stacking RMSE (train): {np.sqrt(mean_squared_error(y, meta_model.predict(meta_features_train)))}")


# Submission (stacked)
submission['accident_risk'] = np.clip(stacked_preds, 0, 1)
submission.to_csv('submission.csv', index=False)
print(submission.head())


# submission['accident_risk'] = np.clip(0.45*xgb_preds + 0.35*lgb_preds + 0.20*cat_preds, 0, 1)
# submission.to_csv('submission_blend.csv', index=False)

