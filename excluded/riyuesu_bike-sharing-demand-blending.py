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
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import os

# --- 0. データ準備 (前回の最適な状態を再現) ---
dfs = {}
BASE_PATH = '/kaggle/input/bike-sharing-demand/'

for name in ['train', 'test']:
    file_path = os.path.join(BASE_PATH, f'{name}.csv')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: 指定されたパス '{file_path}' でファイルを読み込めませんでした。")
        raise
    df['_data'] = name
    dfs[name] = df
df_all = pd.concat([dfs['train'], dfs['test']], ignore_index=True)
df_all['datetime'] = pd.to_datetime(df_all['datetime'])

# 特徴量エンジニアリング
df_all['year'] = df_all['datetime'].dt.year
df_all['month'] = df_all['datetime'].dt.month
df_all['hour'] = df_all['datetime'].dt.hour
df_all['weekday'] = df_all['datetime'].dt.weekday
df_all['day_hour'] = df_all['weekday'].astype(str) + '_' + df_all['hour'].astype(str)
df_all['is_rush_hour'] = df_all['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
df_all['is_night'] = df_all['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)

# 周期的な特徴量 (sin/cos変換)
df_all['hour_sin'] = np.sin(2 * np.pi * df_all['hour'] / 24)
df_all['hour_cos'] = np.cos(2 * np.pi * df_all['hour'] / 24)

# windspeed / humidity の欠損値（0）を補完
is_train = df_all['_data'] == 'train'
drop_indices = df_all[is_train & (df_all['humidity'] <= 2)].index
df_all = df_all.drop(drop_indices)

df_wind_not_zero = df_all[df_all['windspeed'] > 0].copy()
df_wind_zero = df_all[df_all['windspeed'] == 0].copy()
wind_cols = ['temp', 'atemp', 'season', 'weather', 'humidity', 'month', 'year', 'hour', 'weekday']

if not df_wind_zero.empty and not df_wind_not_zero.empty:
    rf_wind = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf_wind.fit(df_wind_not_zero[wind_cols], df_wind_not_zero['windspeed'])
    windspeed_pred = rf_wind.predict(df_wind_zero[wind_cols])
    df_all.loc[df_all['windspeed'] == 0, 'windspeed'] = windspeed_pred

# カスタムイベント (holiday/workingday) の修正 
dates_to_fix = [
    (pd.Timestamp(2011, 4, 15), 1, 0),
    (pd.Timestamp(2012, 4, 16), 1, 0),
    (pd.Timestamp(2011, 11, 25), 0, 1),
    (pd.Timestamp(2012, 11, 23), 0, 1),
    (pd.Timestamp(2011, 12, 24), 0, 1),(pd.Timestamp(2012, 12, 24), 0, 1),
    (pd.Timestamp(2011, 12, 26), 0, 1),(pd.Timestamp(2012, 12, 26), 0, 1),
    (pd.Timestamp(2011, 12, 31), 0, 1),(pd.Timestamp(2012, 12, 31), 0, 1),
    (pd.Timestamp(2012, 5, 21), 0, 1),
    (pd.Timestamp(2012, 6, 1), 0, 1),
    (pd.Timestamp(2012, 10, 30), 0, 1),
]

for date, workingday_value, holiday_value in dates_to_fix:
    df_all.loc[df_all['datetime'].dt.date == date.date(), 'workingday'] = workingday_value
    df_all.loc[df_all['datetime'].dt.date == date.date(), 'holiday'] = holiday_value

# 快適フラグ
df_all['ideal'] = df_all[['temp', 'windspeed']].apply(
    lambda x: 1 if (x['temp'] > 27 and x['windspeed'] < 30) else 0, axis = 1
)
# 不適フラグ
df_all['sticky'] = df_all[['humidity', 'workingday']].apply(
    lambda x: 1 if (x['workingday'] == 1 and x['humidity'] >= 60) else 0, axis = 1
)

df_all = df_all.drop(['atemp', 'datetime'], axis=1)

categorical_features_to_convert = ['year', 'month', 'hour', 'weekday', 'season', 'weather', 'holiday', 'workingday', 'ideal', 'sticky']
for col in categorical_features_to_convert:
    df_all[col] = df_all[col].astype('category')

categorical_features_for_dummies = ['year', 'month', 'hour', 'weekday', 'season', 'weather', 'holiday', 'workingday', 'day_hour', 'ideal', 'sticky']
df_all = pd.get_dummies(df_all, columns=categorical_features_for_dummies, drop_first=True)

df_targets = df_all[['count', 'casual', 'registered']].copy()
df_data_flag = df_all['_data'].copy()
df_all = df_all.drop(['count', 'casual', 'registered', '_data'], axis=1)

X_train = df_all[df_data_flag == 'train']
X_test = df_all[df_data_flag == 'test']
y_train_casual_log = np.log1p(df_targets['casual'][df_data_flag == 'train'])
y_train_registered_log = np.log1p(df_targets['registered'][df_data_flag == 'train'])

# モデルのアンサンブル
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Registered 
LGBM_PARAMS_REG_BASE = {
    'objective':'regression','metric': 'rmse','n_estimators': 5000,
    'learning_rate': 0.015, 'num_leaves': 40, 'min_child_samples': 25,
    'lambda_l1': 0.1, 'lambda_l2': 0.1,
    'bagging_fraction': 0.8, 'bagging_freq': 1,
    'feature_fraction': 0.8, 'n_jobs': -1, 'verbose': -1
}
# Casual
LGBM_PARAMS_CASUAL_BASE = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 5000,
    'learning_rate': 0.05, 'num_leaves': 65, 'min_child_samples': 5,
    'lambda_l2': 0.01, 'bagging_fraction': 0.8, 'bagging_freq': 1,
    'feature_fraction': 0.8, 'n_jobs': -1, 'verbose': -1
}

test_preds_casual_lgbm = np.zeros(X_test.shape[0])
test_preds_registered_lgbm = np.zeros(X_test.shape[0])

# LightGBM
print("--- 2.1. LightGBM K-Fold Prediction ---")
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    
    current_seed = 42 + fold
    
    # Registered
    y_train_reg, y_val_reg = y_train_registered_log.iloc[train_index], y_train_registered_log.iloc[val_index]
    params_reg = LGBM_PARAMS_REG_BASE.copy()
    params_reg['seed'] = params_reg['bagging_seed'] = params_reg['feature_fraction_seed'] = current_seed
    model_reg = lgb.LGBMRegressor(**params_reg)
    model_reg.fit(X_train_fold, y_train_reg, eval_set=[(X_val_fold, y_val_reg)], eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)])
    test_preds_registered_lgbm += model_reg.predict(X_test) / NFOLDS
    
    # Casual
    y_train_cas, y_val_cas = y_train_casual_log.iloc[train_index], y_train_casual_log.iloc[val_index]
    params_cas = LGBM_PARAMS_CASUAL_BASE.copy()
    params_cas['seed'] = params_cas['bagging_seed'] = params_cas['feature_fraction_seed'] = current_seed
    model_cas = lgb.LGBMRegressor(**params_cas)
    model_cas.fit(X_train_fold, y_train_cas, eval_set=[(X_val_fold, y_val_cas)], eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)])
    test_preds_casual_lgbm += model_cas.predict(X_test) / NFOLDS

# Ridge回帰 
print("--- 2.2. Ridge Regression Prediction ---")
RIDGE_ALPHA = 20

# Registered
ridge_reg = Ridge(alpha=RIDGE_ALPHA, random_state=42)
ridge_reg.fit(X_train, y_train_registered_log)
test_preds_registered_ridge = ridge_reg.predict(X_test)

# Casual
ridge_cas = Ridge(alpha=RIDGE_ALPHA, random_state=42)
ridge_cas.fit(X_train, y_train_casual_log)
test_preds_casual_ridge = ridge_cas.predict(X_test)

# ブレンド 
print("--- 2.3. Blending LGBM and Ridge (Alpha=0.9) ---")
BLENDING_ALPHA = 0.9

# Registered Blending
test_preds_registered_blended = (BLENDING_ALPHA * test_preds_registered_lgbm) + ((1 - BLENDING_ALPHA) * test_preds_registered_ridge)

# Casual Blending
test_preds_casual_blended = (BLENDING_ALPHA * test_preds_casual_lgbm) + ((1 - BLENDING_ALPHA) * test_preds_casual_ridge)


# 逆変換と合計 
predictions_casual = np.expm1(test_preds_casual_blended)
predictions_registered = np.expm1(test_preds_registered_blended)

predictions_casual = np.maximum(0, predictions_casual)
predictions_registered = np.maximum(0, predictions_registered)

predictions_count = (predictions_casual + predictions_registered).astype(int)

# 提出ファイルの作成
submission = pd.DataFrame({
    'datetime': dfs['test']['datetime'],
    'count': predictions_count
})
submission.to_csv('submission_lgbm_ridge_blending_improved.csv', index=False)

print("\n✅ モデルのブレンド (LGBM + Ridge) による予測完了。")
print("ファイル名: submission_lgbm_ridge_blending.csv として保存されました。")

