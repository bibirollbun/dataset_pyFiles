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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from skopt import gp_minimize
from skopt.space import Real


def get_rmsle(y_pred, y_actual):
    """RMSLEを計算"""
    y_pred = np.maximum(y_pred, 0)
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)


df = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
df.columns = map(str.lower, df.columns)


for col in ['casual', 'registered', 'count']:
    df[f'{col}_log'] = np.log(df[col] + 1)


dt = pd.DatetimeIndex(df['datetime'])
df.set_index(dt, inplace=True)
df['date'] = dt.date
df['day'] = dt.day
df['month'] = dt.month
df['year'] = dt.year
df['hour'] = dt.hour
df['dow'] = dt.dayofweek
df['woy'] = df['woy'] = dt.isocalendar().week


# peak特徴量
df['peak'] = df[['hour', 'workingday']].apply(
    lambda x: (0, 1)[(x['workingday'] == 1 and (x['hour'] == 8 or 17 <= x['hour'] <= 18 or 12 <= x['hour'] <= 13)) or
                     (x['workingday'] == 0 and 10 <= x['hour'] <= 19)], axis=1)



# 祝日の調整
df['holiday'] = df[['month', 'day', 'holiday', 'year']].apply(
    lambda x: (x['holiday'], 1)[x['year'] == 2012 and x['month'] == 10 and x['day'] == 30], axis=1)
df['holiday'] = df[['month', 'day', 'holiday']].apply(
    lambda x: (x['holiday'], 1)[x['month'] == 12 and x['day'] in [24, 26, 31]], axis=1)
df['workingday'] = df[['month', 'day', 'workingday']].apply(
    lambda x: (x['workingday'], 0)[x['month'] == 12 and x['day'] in [24, 31]], axis=1)



# sticky特徴量
df['sticky'] = df[['humidity', 'workingday']].apply(
    lambda x: (0, 1)[x['workingday'] == 1 and x['humidity'] >= 60], axis=1)



# ideal特徴量の作成
train_temp_values = df['temp'].values
train_windspeed_values = df['windspeed'].values
train_count_values = df['count'].values
season_values = df['season'].values
unique_seasons = np.unique(season_values)

season_params = {}

def eval_func(df1, df2, params):
    a, b = params
    return ((df1 >= a) & (df2 <= b)).astype(float)

for season in unique_seasons:
    mask = season_values == season
    season_temps = train_temp_values[mask]
    season_windspeeds = train_windspeed_values[mask]
    season_counts = train_count_values[mask]

    def calc_corr(params):
        combined = eval_func(season_temps, season_windspeeds, params)
        valid = ~np.isnan(combined) & ~np.isnan(season_counts)
        if np.sum(valid) == 0 or np.std(combined[valid]) == 0:
            return 1.0
        corr = np.corrcoef(combined[valid], season_counts[valid])[0, 1]
        return 1 - np.abs(corr)

    space = [Real(0, season_temps.max(), name='a'), Real(0, season_windspeeds.max(), name='b')]
    res = gp_minimize(calc_corr, space, n_calls=50, random_state=42 + season, verbose=False)
    season_params[season] = res.x

df['ideal'] = df.apply(
    lambda row: eval_func(np.array([row['temp']]), np.array([row['windspeed']]), 
                         season_params[row['season']])[0], axis=1)



feature_cols = ['weather', 'temp', 'atemp', 'humidity', 'windspeed',
                'holiday', 'workingday', 'season', 'hour', 'dow', 
                'woy', 'year', 'peak', 'sticky', 'ideal']


X_train = df[feature_cols]
y_train_casual = df['casual_log']
y_train_registered = df['registered_log']


# CatBoost (Casual)
print("CatBoost (Casual) 学習")
catboost_casual_params = {
    'iterations': 1707, 'depth': 8, 'learning_rate': 0.026001381228830025, 
    'l2_leaf_reg': 4.210335727372638, 'random_state': 42, 'verbose': 0
}
model_cat_casual = CatBoostRegressor(**catboost_casual_params)
model_cat_casual.fit(X_train, y_train_casual)


# CatBoost (Registered)
print("CatBoost (Registered) 学習")
catboost_registered_params = {
    'iterations': 1686, 'depth': 8, 'learning_rate': 0.035297191286179294, 
    'l2_leaf_reg': 8.337847821597938, 'random_state': 42, 'verbose': 0
}
model_cat_registered = CatBoostRegressor(**catboost_registered_params)
model_cat_registered.fit(X_train, y_train_registered)


# XGBoost (Casual)
print("XGBoost (Casual) 学習")
xgboost_casual_params = {
    'n_estimators': 857, 'max_depth': 6, 'learning_rate': 0.019008480228079223, 
    'subsample': 0.7604603292118235, 'colsample_bytree': 0.8583425762812965, 
    'random_state': 42, 'n_jobs': -1
}
model_xgb_casual = XGBRegressor(**xgboost_casual_params)
model_xgb_casual.fit(X_train, y_train_casual)


# XGBoost (Registered)
print("XGBoost (Registered) 学習")
xgboost_registered_params = {
    'n_estimators': 248, 'max_depth': 7, 'learning_rate': 0.059721049157371794, 
    'subsample': 0.9758905175875565, 'colsample_bytree': 0.8783547221751052, 
    'random_state': 42, 'n_jobs': -1
}
model_xgb_registered = XGBRegressor(**xgboost_registered_params)
model_xgb_registered.fit(X_train, y_train_registered)


# LightGBM (Casual)
print("LightGBM (Casual) 学習")
lightgbm_casual_params = {
    'n_estimators': 677, 'learning_rate': 0.016117781696726268, 
    'num_leaves': 83, 'subsample': 0.7206395419234519, 
    'colsample_bytree': 0.803738583052787, 'random_state': 42, 
    'n_jobs': -1, 'verbose': -1
}
model_lgb_casual = LGBMRegressor(**lightgbm_casual_params)
model_lgb_casual.fit(X_train, y_train_casual)


# テストデータの読み込みと前処理
df_test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
df_test.columns = map(str.lower, df_test.columns)


# 日時特徴量の追加
dt_test = pd.DatetimeIndex(df_test['datetime'])
df_test.set_index(dt_test, inplace=True)
df_test['date'] = dt_test.date
df_test['day'] = dt_test.day
df_test['month'] = dt_test.month
df_test['year'] = dt_test.year
df_test['hour'] = dt_test.hour
df_test['dow'] = dt_test.dayofweek
df_test['woy'] = dt_test.isocalendar().week


# peak特徴量
df_test['peak'] = df_test[['hour', 'workingday']].apply(
    lambda x: (0, 1)[(x['workingday'] == 1 and (x['hour'] == 8 or 17 <= x['hour'] <= 18 or 12 <= x['hour'] <= 13)) or
                     (x['workingday'] == 0 and 10 <= x['hour'] <= 19)], axis=1)

# 祝日の調整
df_test['holiday'] = df_test[['month', 'day', 'holiday', 'year']].apply(
    lambda x: (x['holiday'], 1)[x['year'] == 2012 and x['month'] == 10 and x['day'] == 30], axis=1)
df_test['holiday'] = df_test[['month', 'day', 'holiday']].apply(
    lambda x: (x['holiday'], 1)[x['month'] == 12 and x['day'] in [24, 26, 31]], axis=1)
df_test['workingday'] = df_test[['month', 'day', 'workingday']].apply(
    lambda x: (x['workingday'], 0)[x['month'] == 12 and x['day'] in [24, 31]], axis=1)

# sticky特徴量
df_test['sticky'] = df_test[['humidity', 'workingday']].apply(
    lambda x: (0, 1)[x['workingday'] == 1 and x['humidity'] >= 60], axis=1)

# ideal特徴量
df_test['ideal'] = df_test.apply(
    lambda row: eval_func(np.array([row['temp']]), np.array([row['windspeed']]), 
                         season_params[row['season']])[0], axis=1)

X_test = df_test[feature_cols]


# Casual予測
pred_cat_casual_log = model_cat_casual.predict(X_test)
pred_xgb_casual_log = model_xgb_casual.predict(X_test)
pred_lgb_casual_log = model_lgb_casual.predict(X_test)

pred_cat_casual = np.exp(pred_cat_casual_log) - 1
pred_xgb_casual = np.exp(pred_xgb_casual_log) - 1
pred_lgb_casual = np.exp(pred_lgb_casual_log) - 1

# Casual アンサンブル
pred_casual = (pred_xgb_casual + pred_lgb_casual + pred_cat_casual) / 3


# Registered予測
pred_cat_registered_log = model_cat_registered.predict(X_test)
pred_xgb_registered_log = model_xgb_registered.predict(X_test)

pred_cat_registered = np.exp(pred_cat_registered_log) - 1
pred_xgb_registered = np.exp(pred_xgb_registered_log) - 1

# Registered アンサンブル
pred_registered =  (0.4 * pred_xgb_registered) + (0.6 * pred_cat_registered)


# 最終予測 (Count)
pred_count = pred_casual + pred_registered
pred_count = np.maximum(pred_count, 0)


submission = pd.DataFrame({
    'datetime': df_test['datetime'].reset_index(drop=True),
    'count': pred_count
})
submission.to_csv('submission.csv', index=False)




