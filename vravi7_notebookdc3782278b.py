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
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import haversine

train = pd.read_csv('//kaggle/input/ny-taxi-competition/train.csv')
test = pd.read_csv('/kaggle/input/ny-taxi-competition/test.csv')


def feature_engineering(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    
    df['trip_distance'] = df.apply(
        lambda row: haversine_distance(
            row['pickup_latitude'], row['pickup_longitude'], 
            row['dropoff_latitude'], row['dropoff_longitude']
        ), 
        axis=1
    )
    
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    return haversine.haversine((lat1, lon1), (lat2, lon2), unit='km')

train_processed = feature_engineering(train.copy())
test_processed = feature_engineering(test.copy())


features = [
    'pickup_longitude', 'pickup_latitude', 
    'dropoff_longitude', 'dropoff_latitude', 
    'passenger_count', 'vendor_id', 
    'hour', 'day_of_week', 'month', 
    'is_weekend', 'trip_distance',
    'hour_sin', 'hour_cos'
]

X = train_processed[features]
y_train = train_processed['trip_duration']
X_test = test_processed[features]


scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


def train_lightgbm(X_train, y_train):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []
    
    for train_index, val_index in tscv.split(X_train):
        X_train_split, X_val_split = X_train[train_index], X_train[val_index]
        y_train_split, y_val_split = y_train.iloc[train_index], y_train.iloc[val_index]
        
        train_data = lgb.Dataset(X_train_split, label=y_train_split)
        val_data = lgb.Dataset(X_val_split, label=y_val_split)
        
        model = lgb.train(
            params, 
            train_data, 
            valid_sets=[val_data],  # Validation set for early stopping
            valid_names=['val'],   
            num_boost_round=200, 
        )
        
        y_pred = model.predict(X_val_split)
        rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
        rmse_scores.append(rmse)
    
    print(f'Cross-validation RMSE: {np.mean(rmse_scores)}')
    return model

lgb_model = train_lightgbm(X_train, y_train)


 test_predictions = lgb_model.predict(X_test)


 submission = pd.DataFrame({
        'id': test['id'],
        'trip_duration': test_predictions
    })
submission.to_csv('submission.csv', index=False)
 print('Submission file generated successfully!')

