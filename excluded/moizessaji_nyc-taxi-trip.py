# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import math
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


zf = zipfile.ZipFile('/kaggle/input/nyc-taxi-trip-duration/train.zip') 
train = pd.read_csv(zf.open('train.csv'),parse_dates=['pickup_datetime','dropoff_datetime'])
zf = zipfile.ZipFile('/kaggle/input/nyc-taxi-trip-duration/test.zip')
test = pd.read_csv(zf.open('test.csv'),parse_dates=['pickup_datetime'])


print(f'train shape\n{train.shape}')
print(f'train describe\n{train.describe()}')
print(f'train info\n{train.isna().any()}')
print(f'\ntest shape\n{test.shape}')
print(f'\ntest describe\n{test.describe()}')
print(f'test info\n{test.isna().any()}')


def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0  
    
    # Convert latitude and longitude from degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in kilometers
    distance = R * c
    return distance


def distance(df):
    coords  = (df['pickup_latitude'],df['pickup_longitude'],df['dropoff_latitude'],df['dropoff_longitude'])
    df['distance_km'] = haversine(*coords)
    # df['speed_m_s'] = df['distance_km']*1000/df['trip_duration']
    # df['diff_seconds'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()
    return df


%pip install meteostat
from meteostat import Point, Daily, Hourly
from datetime import datetime

# New York City coordinates
location = Point(40.7128, -74.0060)

# Set date range
start = datetime(2015, 12, 31)
end = datetime(2016, 7, 31)

# Fetch daily and hourly weather data
data_daily = Daily(location, start, end).fetch()
data_hourly = Hourly(location, start, end).fetch()

# print(data_daily.head())
# print(data_hourly.head())


data_hourly = data_hourly.reset_index().rename(columns={'DatetimeIndex': 'weather_datetime'})

data_hourly['time'] = data_hourly['time'].dt.tz_localize('UTC')

data_hourly['time_ny'] = data_hourly['time'].dt.tz_convert('America/New_York')

data_hourly['time_ny'] = data_hourly['time_ny'].dt.tz_localize(None)


def time_features(df):
    df['pickup_day'] = df['pickup_datetime'].dt.day
    # df['dropoff_day'] = df['dropoff_datetime'].dt.day
    df['pickup_month'] = df['pickup_datetime'].dt.month
    # df['dropoff_month'] = df['dropoff_datetime'].dt.month
    df['pickup_year'] = df['pickup_datetime'].dt.year
    # df['dropoff_year'] = df['dropoff_datetime'].dt.year
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    # df['dropoff_hour'] = df['dropoff_datetime'].dt.hour
    df['pickup_week'] = df['pickup_datetime'].dt.isocalendar()['week']
    # df['dropoff_week'] = df['dropoff_datetime'].dt.isocalendar()['week']
    df['pickup_datetime_hour_trunc'] = df['pickup_datetime'].dt.floor('h')
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    return df


def weather_merge(df):
    merged_df = df.merge(data_hourly,how='left',left_on='pickup_datetime_hour_trunc',right_on='time_ny')
    return merged_df


# merged_train.isna().any()


# merged_train = merged_train.sort_values(['pickup_datetime','dropoff_datetime'])


# merged_train[['temp', 'prcp', 'snow', 'coco']] = merged_train[['temp', 'prcp', 'snow', 'coco']].ffill()


# def eval_train_df(df):
#     train_df = df[df['pickup_month']<=5]
#     X_train = train_df[features]
#     y_train = train_df['trip_duration']
#     test_df = df[~df['pickup_month']<=5]
#     return train_df,test_df


# merged_train.columns
# Index(['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime',
#        'passenger_count', 'pickup_longitude', 'pickup_latitude',
#        'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
#        'trip_duration', 'distance_km', 'speed_m_s', 'diff_seconds',
#        'pickup_day', 'dropoff_day', 'pickup_month', 'dropoff_month',
#        'pickup_hour', 'dropoff_hour', 'pickup_year', 'dropoff_year',
#        'pickup_week', 'dropoff_week', 'pickup_datetime_hour_trunc', 'time',
#        'temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres',
#        'tsun', 'coco', 'time_ny'],
#       dtype='object')


# test.columns


features = [
       'passenger_count',
    'pickup_longitude', 
    'pickup_latitude',
    'dropoff_longitude',
       'dropoff_latitude',
       'distance_km', 
       'pickup_day',  
    # 'pickup_month', 
       'pickup_hour',
    # 'pickup_year', 
       # 'pickup_week', 
    'pickup_dayofweek',
       'temp','prcp'
]


def train_split(df):
    X_train = df[features]
    y_train = df['trip_duration']
    return X_train, y_train


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


# model = RandomForestRegressor(
#                 n_estimators=100, 
#                 max_depth=5, 
#                 random_state=42, 
#                 n_jobs=-1,
#     min_samples_split=2,min_samples_leaf=4,max_features = 'sqrt'
#             # 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 5
#             )


# model.fit(X_train, y_train)


# y_pred = model.predict(X_test)


# y_pred = np.exp(y_pred)


# np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test)) ** 2))


from sklearn.metrics import mean_squared_error, r2_score


train_df = distance(train)
train_df = time_features(train_df)
train_merged = weather_merge(train_df)
X_train, y_train = train_split(train_merged)

test_df = distance(test)
test_df = time_features(test_df)
test_merged = weather_merge(test_df)



import lightgbm as lgb
from lightgbm import LGBMRegressor

lgbm = lgb.LGBMRegressor()
lgbm.fit(X_train, y_train)
# print(lgbm.score(X_train, y_train), lgbm.score(X_test, y_test))
# print(np.sqrt(mean_squared_error (y_test, lgbm.predict(X_test))))


y_pred = lgbm.predict(test_merged[features])


submission = pd.DataFrame({'id': test.id, 'trip_duration': (y_pred)})


submission.to_csv("submission.csv", index=False)


# without weather data - non merged data
# [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.018620 seconds.
# You can set `force_row_wise=true` to remove the overhead.
# And if memory is not enough, you can set `force_col_wise=true`.
# [LightGBM] [Info] Total Bins 1338
# [LightGBM] [Info] Number of data points in the train set: 1224328, number of used features: 8
# [LightGBM] [Info] Start training from score 6.454869
# 0.7138562185599941 0.7123454878265641
# 0.4288593168582211



# with weather data merged df
# [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.020480 seconds.
# You can set `force_row_wise=true` to remove the overhead.
# And if memory is not enough, you can set `force_col_wise=true`.
# [LightGBM] [Info] Total Bins 1497
# [LightGBM] [Info] Number of data points in the train set: 1224328, number of used features: 10
# [LightGBM] [Info] Start training from score 6.454869
# 0.7149781430716527 0.7134874068726587
# 0.42800723634402765



# [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.070708 seconds.
# You can set `force_row_wise=true` to remove the overhead.
# And if memory is not enough, you can set `force_col_wise=true`.
# [LightGBM] [Info] Total Bins 1505
# [LightGBM] [Info] Number of data points in the train set: 1224328, number of used features: 11
# [LightGBM] [Info] Start training from score 6.454869
# 0.7153013134246655 0.7138704288279119
# 0.4277210517254694


