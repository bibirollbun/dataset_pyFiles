# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error


import pandas as pd
data_path = '/kaggle/input/bike-sharing-demand/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sampleSubmission.csv')


# train，test を区別する
train['is_train'] = 1
test['is_train'] = 0
test['casual'] = np.nan
test['registered'] = np.nan
test['count'] = np.nan
all_df = pd.concat([train, test], ignore_index=True)


# dataframeの分解
all_df['datetime'] = pd.to_datetime(all_df['datetime'])
all_df['year']  = all_df['datetime'].dt.year
all_df['month'] = all_df['datetime'].dt.month
all_df['day']   = all_df['datetime'].dt.day
all_df['hour']  = all_df['datetime'].dt.hour
all_df['dow']   = all_df['datetime'].dt.dayofweek  # 0=月, 6=日


# 週末
all_df['is_weekend'] = (all_df['dow'] >= 5).astype(int)

# 時間 × 曜日(0〜167 の整数）
all_df['hour_dow'] = all_df['hour'] + 24 * all_df['dow']

# 温度・湿度のビン化
all_df['temp_bin'] = pd.cut(
    all_df['temp'],
    bins=[-np.inf, 5, 10, 15, 20, 25, 30, np.inf],
    labels=False
)

all_df['humidity_bin'] = pd.cut(
    all_df['humidity'],
    bins=[-np.inf, 20, 40, 60, 80, 100],
    labels=False
)

# 日内周期を sin，　cos で表現
all_df['hour_sin'] = np.sin(2 * np.pi * all_df['hour'] / 24)
all_df['hour_cos'] = np.cos(2 * np.pi * all_df['hour'] / 24)

all_df


# windspeed = 0 の補完
wind_features = ['season', 'weather', 'humidity', 'month', 'temp', 'year', 'atemp']

wind_not0 = all_df[all_df['windspeed'] != 0].copy()
wind_0 = all_df[all_df['windspeed'] == 0].copy()

rf_wind = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_wind.fit(wind_not0[wind_features], wind_not0['windspeed'])

wind_0.loc[:, 'windspeed'] = rf_wind.predict(wind_0[wind_features])

# 補完後に結合し直す
all_df = pd.concat([wind_not0, wind_0], ignore_index=True)
all_df = all_df.sort_values('datetime').reset_index(drop=True)


train_df = all_df[all_df['is_train'] == 1].copy()
test_df  = all_df[all_df['is_train'] == 0].copy()


# casual, registered の両方をlog1p変換
train_df['casual_log'] = np.log1p(train_df['casual'])
train_df['registered_log'] = np.log1p(train_df['registered'])
train_df['count_log'] = np.log1p(train_df['count'])


feature_cols = [
    'season', 'holiday', 'workingday', 'weather',
    'temp', 'atemp', 'humidity', 'windspeed',
    'year', 'month', 'day', 'hour', 'dow',
    'is_weekend', 'hour_dow',
    'temp_bin', 'humidity_bin',
    'hour_sin', 'hour_cos'
]


# trainのデータ分割
train_mask  = train_df['day'] <= 15
valid_mask  = train_df['day'] >= 16  

train_tr = train_df[train_mask]
valid_tr = train_df[valid_mask]

X_train = train_tr[feature_cols]
X_valid = valid_tr[feature_cols]


# casual，　registeredに分かれて学習
y_train_casual_log     = train_tr['casual_log']
y_train_registered_log = train_tr['registered_log']

y_valid_casual_log     = valid_tr['casual_log']
y_valid_registered_log = valid_tr['registered_log']
y_valid_count          = valid_tr['count']  

rf_params = dict(n_estimators=300,max_depth=None,random_state=42,n_jobs=-1)

model_casual = RandomForestRegressor(**rf_params)
model_registered = RandomForestRegressor(**rf_params)


model_casual.fit(X_train, y_train_casual_log)
model_registered.fit(X_train, y_train_registered_log)


# 検証データで予測して RMSLE を計算
valid_pred_casual_log     = model_casual.predict(X_valid)
valid_pred_registered_log = model_registered.predict(X_valid)

valid_pred_casual     = np.expm1(valid_pred_casual_log)
valid_pred_registered = np.expm1(valid_pred_registered_log)

# 合計レンタル数 = casual + registered
valid_pred_count = valid_pred_casual + valid_pred_registered

# RMSLE を計算（負の値が出ないように 0 でクリップ）
valid_pred_count = np.maximum(0, valid_pred_count)
y_valid_count_clipped = np.maximum(0, y_valid_count.values)

rmsle_score = np.sqrt(mean_squared_log_error(y_valid_count_clipped, valid_pred_count))
print(f"Validation RMSLE (casual+registered RF): {rmsle_score:.4f}")


# test データでの予測とsubmission 作成
X_test = test_df[feature_cols]

X_full = train_df[feature_cols]
y_full_casual_log     = train_df['casual_log']
y_full_registered_log = train_df['registered_log']

model_casual_full     = RandomForestRegressor(**rf_params)
model_registered_full = RandomForestRegressor(**rf_params)

model_casual_full.fit(X_full, y_full_casual_log)
model_registered_full.fit(X_full, y_full_registered_log)

test_pred_casual_log     = model_casual_full.predict(X_test)
test_pred_registered_log = model_registered_full.predict(X_test)

test_pred_casual     = np.expm1(test_pred_casual_log)
test_pred_registered = np.expm1(test_pred_registered_log)


# 目的関数(count)
test_pred_count = test_pred_casual + test_pred_registered
test_pred_count = np.maximum(0, test_pred_count)

submission = pd.DataFrame({
    'datetime': test_df['datetime'],
    'count': test_pred_count
})

submission.to_csv('submission.csv', index=False)

