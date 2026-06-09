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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import holidays


train = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/train_df.csv')
test = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/test_df.csv')


train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])
train = train.set_index('datetime')
test = test.set_index('datetime')


train.info()


test.info()


plt.figure(figsize=(12, 6))
plt.plot(train['e_users'])
plt.grid()
plt.show()


'''
# STL分解
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(train['e_users'], model='additive', period=24*7)
fig, ax = plt.subplots(4, 1, figsize=(12, 10))
decomposition.trend.plot(ax=ax[0])
ax[0].set_title('Trend')
decomposition.seasonal.plot(ax=ax[1])
ax[1].set_title('Seasonal')
decomposition.resid.plot(ax=ax[2])
ax[2].set_title('Residual')
decomposition.observed.plot(ax=ax[3])
ax[3].set_title('Observed')
plt.tight_layout()
plt.show()
'''


# 自己相関と偏自己相関
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(train['e_users'], ax=ax[0], lags=24*7)
plot_pacf(train['e_users'], ax=ax[1], lags=24*7)
plt.show()


from statsmodels.tsa.stattools import acf

acf_values = acf(train['e_users'], nlags=100)
best_lag = (acf_values > 0.2).argmax()  # 相関が 0.2 を下回る最小のラグ
print(f"推奨ラグ: {best_lag}")



plt.figure(figsize=(12, 6))
plt.plot(train['e_users'])
plt.xlim(pd.to_datetime('2022-03-01'), pd.to_datetime('2022-05-01'))
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(train['e_users'])
plt.xlim(pd.to_datetime('2023-03-01'), pd.to_datetime('2023-05-01'))
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(train['e_users'])
plt.xlim(pd.to_datetime('2024-03-01'), pd.to_datetime('2024-04-01'))
plt.grid()
plt.show()


df = pd.concat([train, test])
df


plt.figure(figsize=(12, 6))
plt.plot(df['promotion_1'], label='Promotion1')
plt.plot(df['promotion_2'], label='Promotion2')
plt.plot(df['promotion_3'], label='Promotion3')
plt.legend()
plt.grid()
plt.show()


train


def engineer(df_old: pd.DataFrame, is_lstm: bool = False) -> pd.DataFrame:

  df = df_old.copy()

  df['p2/p1'] = df['promotion_2'] / df['promotion_1']
  df['p3/p1'] = df['promotion_3'] / df['promotion_1']
  df['p3/p2'] = df['promotion_3'] / df['promotion_2']
  df['p1mulp2'] = df['promotion_1'] * df['promotion_2']
  df['p3mulp1'] = df['promotion_1'] * df['promotion_3']
  df['p2mulp3'] = df['promotion_2'] * df['promotion_3']

  if is_lstm==False:

    for i in range(24):
      df[f'p1_shift{i+1}h'] = df['promotion_1'].shift(i+1)
      df[f'p2_shift{i+1}h'] = df['promotion_2'].shift(i+1)
      df[f'p3_shift{i+1}h'] = df['promotion_3'].shift(i+1)
      df[f'p3mulp1_shift{i+1}h'] = df['p3mulp1'].shift(i+1)

    for i in range(7):
      df[f'p1_shift{i+1}d'] = df['promotion_1'].shift(24*(i+1))
      df[f'p2_shift{i+1}d'] = df['promotion_2'].shift(24*(i+1))
      df[f'p3_shift{i+1}d'] = df['promotion_3'].shift(24*(i+1))
      df[f'p3mulp1_shift{i+1}d'] = df['p3mulp1'].shift(24*(i+1))

    for i in range(4):
      df[f'p1_shift{i+1}w'] = df['promotion_1'].shift(24*7*(i+1))
      df[f'p2_shift{i+1}w'] = df['promotion_2'].shift(24*7*(i+1))
      df[f'p3_shift{i+1}w'] = df['promotion_3'].shift(24*7*(i+1))
      df[f'p3mulp1_shift{i+1}w'] = df['p3mulp1'].shift(24*7*(i+1))


  df['sin_dayofyear'] = np.sin(2 * np.pi * df.index.dayofyear / 366)
  df['cos_dayofyear'] = np.cos(2 * np.pi * df.index.dayofyear / 366)
  df['sin_dayofmonth'] = np.sin(2 * np.pi * df.index.day / 31)
  df['cos_dayofmonth'] = np.cos(2 * np.pi * df.index.day / 31)
  df['sin_dayofweek'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
  df['cos_dayofweek'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
  df['sin_hour'] = np.sin(2 * np.pi * df.index.hour / 24)
  df['cos_hour'] = np.cos(2 * np.pi * df.index.hour / 24)

  #df['march_april'] = df.index.month.isin([3, 4]).astype(int)

  jp_holidays = holidays.Japan()
  #df['is_jp_holiday'] = df.index.to_series().apply(lambda x: x in jp_holidays).astype(int)

  usa_holidays = holidays.US()
  df['is_usa_holiday'] = df.index.to_series().apply(lambda x: x in usa_holidays).astype(int)

  eu_holidays = holidays.EuropeanCentralBank()
  df['is_eu_holiday'] = df.index.to_series().apply(lambda x: x in eu_holidays).astype(int)

  china_holidays = holidays.China()
  df['is_china_holiday'] = df.index.to_series().apply(lambda x: x in china_holidays).astype(int)

  india_holidays = holidays.India()
  df['is_india_holiday'] = df.index.to_series().apply(lambda x: x in india_holidays).astype(int)

  df['log_e_users'] = np.log(df['e_users'])
  df = df.drop('e_users', axis=1)

  # log_e_users以外のカラムの欠損値を削除
  temp = df.dropna(subset=df.columns.drop('log_e_users'))
  df = temp.copy()


  return df


engineered_df = engineer(df)
engineered_df


temp = engineered_df.copy()

train_engineered = temp[:'2023-12-31']
valid_engineered = temp['2024-01-01':'2024-08-31']
test_engineered = temp['2024-09-01':]

engineered_df_time = engineer(df_old=df, is_lstm=True)
train_time = engineered_df_time[:'2023-12-31']
valid_time = engineered_df_time['2024-01-01':'2024-08-31']
test_time = engineered_df_time['2024-09-01':]

train_valid_time = pd.concat([train_time, valid_time])


# p1mulp3とp2mulp3の関係をlog_e_usersの値に応じて色を変えてプロット
plt.figure(figsize=(12, 6))
scatter = plt.scatter(temp['p3mulp1'], temp['p2mulp3'], c=temp['log_e_users'], cmap='viridis')
plt.colorbar(scatter, label='log_e_users')  # カラーバーを追加
plt.grid()
plt.xlabel('p1mulp3')
plt.ylabel('p2mulp3')
plt.title('Scatter plot of p1mulp3 vs p2mulp3 colored by log_e_users')
plt.show()



'''
temp_columns = train.columns.drop('log_e_users')

for i in range(len(temp_columns)):
  for j in range(i+1, len(temp_columns)):
    plt.figure(figsize=(12, 6))
    plt.scatter(train[temp_columns[i]], train[temp_columns[j]], c=train['log_e_users'], cmap='viridis')
    plt.colorbar()
    plt.xlabel(temp_columns[i])
    plt.ylabel(temp_columns[j])
    plt.title(f'Scatter plot of {temp_columns[i]} vs {temp_columns[j]} colored by log_e_users')
    plt.show()
'''


# p1mulp3とe_usersの関係
plt.figure(figsize=(12, 6))
plt.scatter(train_engineered['p3mulp1'], train_engineered['log_e_users'])
plt.grid()
plt.show()


X_train = train_engineered.drop('log_e_users', axis=1)
y_train = train_engineered['log_e_users']
X_valid = valid_engineered.drop('log_e_users', axis=1)
y_valid = valid_engineered['log_e_users']
X_test = test_engineered.drop('log_e_users', axis=1)


from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_valid_scaled = X_scaler.transform(X_valid)
X_test_scaled = X_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_valid_scaled = y_scaler.transform(y_valid.values.reshape(-1, 1))


X_train_time = train_time.drop('log_e_users', axis=1)
y_train_time = train_time['log_e_users']
X_valid_time = valid_time.drop('log_e_users', axis=1)
y_valid_time = valid_time['log_e_users']
X_test_time = test_time.drop('log_e_users', axis=1)

X_train_valid_time = train_valid_time.drop('log_e_users', axis=1)
y_train_valid_time = train_valid_time['log_e_users']


from sklearn.preprocessing import StandardScaler

# ラグ特徴量なし、検証あり
X_scaler_time = StandardScaler()
X_train_scaled_time = X_scaler_time.fit_transform(X_train_time)
X_valid_scaled_time = X_scaler_time.transform(X_valid_time)
X_test_scaled_time = X_scaler_time.transform(X_test_time)

y_scaler_time = StandardScaler()
y_train_scaled_time = y_scaler_time.fit_transform(y_train.values.reshape(-1, 1))
y_valid_scaled_time = y_scaler_time.transform(y_valid.values.reshape(-1, 1))

# ラグ特徴量なし、検証なし
X_scaler_train_valid_time = StandardScaler()
X_train_valid_scaled_time = X_scaler_train_valid_time.fit_transform(X_train_valid_time)
X_test_scaled_time_novalid = X_scaler_train_valid_time.transform(X_test_time)

y_scaler_time_novalid = StandardScaler()
y_train_valid_scaled_time = y_scaler_time_novalid.fit_transform(y_train_valid_time.values.reshape(-1, 1))


'''
from statsmodels.tsa.statespace.sarimax import SARIMAX

y_sarimax = y_train_valid_time
exog = X_train_valid_scaled_time

sarimax_model = SARIMAX(y_sarimax,
                order=(1, 1, 1),        # (p, d, q)
                seasonal_order=(1, 1, 1, 24),  # (P, D, Q, s)
                exog=exog,
                trend='c',
                enforce_stationarity=False,
                enforce_invertibility=False)
result = sarimax_model.fit(maxiter=20, disp=1)
print(result.summary())
'''


'''
steps = len(test_time)
sarimax_test_pred = result.forecast(steps=steps, exog=X_test_scaled_time_novalid)

# sarimax_test_predのプロット
plt.figure(figsize=(12, 6))
plt.plot(index=test_time.index, value=sarimax_test_pred, label='SARIMAX Forecast')
plt.legend()
plt.grid()
plt.show()
'''


train


train_engineered


from prophet import Prophet
from sklearn.metrics import mean_squared_error

# Initialize Prophet model
prophet_model = Prophet(
    changepoint_prior_scale=0.5,
    yearly_seasonality=True,  # 年単位の季節性は無効化（短期予測では不要）
    weekly_seasonality=True,  # 週単位の季節性を追加
    daily_seasonality=True    # 日単位の季節性を追加
)

# 外生変数（追加回帰変数）をモデルに追加
prophet_model.add_regressor('promotion_1')
prophet_model.add_regressor('promotion_2')
prophet_model.add_regressor('promotion_3')
# 1時間ごとの周期性を追加（24時間周期で変動するものを考慮）
prophet_model.add_seasonality(name='hourly', period=24, fourier_order=8)

prophet_train = pd.DataFrame()
prophet_train['ds'] = train_engineered.index
prophet_train['y'] = train_engineered['log_e_users'].values
prophet_train['promotion_1'] = train_engineered['promotion_1'].values
prophet_train['promotion_2'] = train_engineered['promotion_2'].values
prophet_train['promotion_3'] = train_engineered['promotion_3'].values
prophet_train

# Train the model
prophet_model.fit(prophet_train)


# Generate future dataframe for validation set
future_valid_test = prophet_model.make_future_dataframe(periods=len(pd.concat([valid_engineered, test_engineered])), freq='H')

# 未来のデータに外生変数を追加（予測時にも必要）
# future_valid_test に valid_engineered, test_engineered の temperature をマージ
train_engineered = train_engineered.reset_index().rename(columns={'datetime': 'ds'})
valid_engineered = valid_engineered.reset_index().rename(columns={'datetime': 'ds'})
test_engineered = test_engineered.reset_index().rename(columns={'datetime': 'ds'})

future_valid_test = future_valid_test.merge(
    pd.concat([train_engineered[['ds', 'promotion_1', 'promotion_2', 'promotion_3']], valid_engineered[['ds', 'promotion_1', 'promotion_2', 'promotion_3']], test_engineered[['ds', 'promotion_1', 'promotion_2', 'promotion_3']]]),
    on='ds',
    how='left'
)

train_engineered = train_engineered.rename(columns={'ds': 'datetime'}).set_index('datetime')
valid_engineered = valid_engineered.rename(columns={'ds': 'datetime'}).set_index('datetime')
test_engineered = test_engineered.rename(columns={'ds': 'datetime'}).set_index('datetime')

forecast = prophet_model.predict(future_valid_test)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


future_valid_test


# valid_engineeredのlog_e_usersとRMSEを計算
print(f'Valid RMSE: {np.sqrt(mean_squared_error(valid_engineered["log_e_users"], forecast["yhat"][-len(valid_engineered):]))}')


fig1 = prophet_model.plot(forecast)


fig2 = prophet_model.plot_components(forecast)



import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


params = {'objective': 'regression',
          'metric': 'rmse',
          'random_state': 42,
          'n_estimators': 100,
          'learning_rate': 0.01
          }
lgbm_model = LGBMRegressor(**params)
lgbm_model.fit(X_train, y_train,
               eval_metric='rmse',
               eval_set=[(X_valid, y_valid)],
               callbacks = [lgb.early_stopping(stopping_rounds=50)]
               )
lgbm_train_pred = lgbm_model.predict(X_train)
lgbm_valid_pred = lgbm_model.predict(X_valid)
lgbm_test_pred = lgbm_model.predict(X_test)

print(f'LGBM Train RMSE: {np.sqrt(mean_squared_error(y_train, lgbm_train_pred))}')
print(f'LGBM Valid RMSE: {np.sqrt(mean_squared_error(y_valid, lgbm_valid_pred))}')

plt.figure(figsize=(12, 6))

sns.lineplot(x=X_train.index, y=y_train, label='Train Actual')
sns.lineplot(x=X_valid.index, y=y_valid, label='Valid Actual')
sns.lineplot(x=X_train.index, y=lgbm_train_pred, label='Train Predicted')
sns.lineplot(x=X_valid.index, y=lgbm_valid_pred, label='Valid Predicted')

plt.grid()
plt.legend()
plt.show()


import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


params = {'metric': 'rmse',
          'random_state': 42,
          'n_estimators': 100,
          'learning_rate': 0.01}
xgb_model = XGBRegressor(**params, callbacks=[xgb.callback.EarlyStopping(rounds=50)])
xgb_model.fit(X_train, y_train,
              eval_set = [(X_valid, y_valid)]
              )
xgb_train_pred = xgb_model.predict(X_train)
xgb_valid_pred = xgb_model.predict(X_valid)
xgb_test_pred = xgb_model.predict(X_test)

print(f'Train_RMSE: {np.sqrt(mean_squared_error(y_train, xgb_train_pred))}')
print(f'Valid_RMSE: {np.sqrt(mean_squared_error(y_valid, xgb_valid_pred))}')

plt.figure(figsize=(12, 6))

sns.lineplot(x=X_train.index, y=y_train, label='Train Actual')
sns.lineplot(x=X_valid.index, y=y_valid, label='Valid Actual')
sns.lineplot(x=X_train.index, y=xgb_train_pred, label='Train Predicted')
sns.lineplot(x=X_valid.index, y=xgb_valid_pred, label='Valid Predicted')

plt.grid()
plt.legend()
plt.show()


'''
!pip uninstall -y numpy
!pip install numpy==1.26.4 --no-cache-dir
!pip install catboost --no-cache-dir --no-deps

import numpy as np
print(np.__version__)
'''


'''
import catboost

print(np.__version__)  # numpyのバージョン確認
print(catboost.__version__)  # catboostのバージョン確認
'''


!pip install catboost


# catboost
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

params = {'random_state': 42,
          'iterations': 100,
          'learning_rate': 0.01,
          'eval_metric': 'RMSE',
          'early_stopping_rounds': 50}
cat_model = CatBoostRegressor(**params)
cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
cat_train_pred = cat_model.predict(X_train)
cat_valid_pred = cat_model.predict(X_valid)
cat_test_pred = cat_model.predict(X_test)

print(f'Train_RMSE: {np.sqrt(mean_squared_error(y_train, cat_train_pred))}')
print(f'Valid_RMSE: {np.sqrt(mean_squared_error(y_valid, cat_valid_pred))}')

plt.figure(figsize=(12, 6))

sns.lineplot(x=X_train.index, y=y_train, label='Train Actual')
sns.lineplot(x=X_valid.index, y=y_valid, label='Valid Actual')
sns.lineplot(x=X_train.index, y=cat_train_pred, label='Train Predicted')
sns.lineplot(x=X_valid.index, y=cat_valid_pred, label='Valid Predicted')

plt.grid()
plt.legend()
plt.show()


'''
# MLP
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

params = {'hidden_layer_sizes': (1024, 512, 256, 128, 64, 32),
          'activation': 'tanh',
          'solver': 'adam',
          'random_state': 42
}
mlp_model = MLPRegressor(**params)
mlp_model.fit(X_train_scaled, y_train_scaled)
mlp_train_pred = y_scaler.inverse_transform(mlp_model.predict(X_train_scaled).reshape(-1, 1))
mlp_valid_pred = y_scaler.inverse_transform(mlp_model.predict(X_valid_scaled).reshape(-1, 1))
mlp_test_pred = y_scaler.inverse_transform(mlp_model.predict(X_test_scaled).reshape(-1, 1))

print(f'Train_RMSE: {np.sqrt(mean_squared_error(y_train_scaled, mlp_train_pred))}')
print(f'Valid_RMSE: {np.sqrt(mean_squared_error(y_valid_scaled, mlp_valid_pred))}')

plt.figure(figsize=(12, 6))

sns.lineplot(x=X_train.index, y=y_train, label='Train Actual')
sns.lineplot(x=X_valid.index, y=y_valid, label='Valid Actual')
sns.lineplot(x=X_train.index, y=mlp_train_pred.flatten(), label='Train Predicted')
sns.lineplot(x=X_valid.index, y=mlp_valid_pred.flatten(), label='Valid Predicted')

plt.grid()
plt.legend()
plt.show()
'''


# アンサンブル
ensemble_list = ['lgbm', 'xgb', 'cat']
ensemble_train_pred = (lgbm_train_pred + xgb_train_pred + cat_train_pred) / len(ensemble_list)
ensemble_valid_pred = (lgbm_valid_pred + xgb_valid_pred + cat_valid_pred) / len(ensemble_list)
ensemble_test_pred = (lgbm_test_pred + xgb_test_pred + cat_test_pred) / len(ensemble_list)

print(f'Train_RMSE: {np.sqrt(mean_squared_error(y_train, ensemble_train_pred))}')
print(f'Valid_RMSE: {np.sqrt(mean_squared_error(y_valid, ensemble_valid_pred))}')


# 特徴量重要度
plt.figure(figsize=(10, 40))
feature_importance = pd.DataFrame(lgbm_model.feature_importances_, index=X_train.columns, columns=['importance'])
feature_importance.sort_values('importance', ascending=False)
plt.barh(feature_importance.index, feature_importance['importance'])
plt.show()


submission = pd.read_csv('/kaggle/input/prediction-of-e-commerce-users/submission.csv')
submission


ensemble_test_pred


pred_log_value = ensemble_test_pred
submission['e_users'] = np.exp(pred_log_value)


submission


plt.plot(submission['e_users'])
plt.grid()
plt.show()


submission.to_csv('/kaggle/working/my_submission.csv', index=False)




