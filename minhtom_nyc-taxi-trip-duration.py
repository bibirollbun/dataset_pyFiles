 import pandas as pd 


train_df = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/train.zip",compression = "zip")
train_df.head() 


train_df['id'] = train_df['id'].str.replace('id', '').astype(int)



train_df.info()


train_df.isnull().sum()


train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime']) 
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])


train_df['hour'] = train_df['pickup_datetime'].dt.hour
train_df['day'] = train_df['pickup_datetime'].dt.day
train_df['weekday'] = train_df['pickup_datetime'].dt.weekday 
train_df['month'] = train_df['pickup_datetime'].dt.month
train_df['year'] = train_df['pickup_datetime'].dt.year
train_df['weekend'] = train_df['weekday'].isin([5,6]).astype(int)
train_df['is_daytime'] = train_df['hour'].between(6,18).astype(int)
train_df.info()


hour_duration = train_df.groupby('hour')['trip_duration'].mean()


import numpy as np 


def haversine(latitude1, longitude1, latitude2, longitude2) :
    phi1 = np.radians(latitude1)
    phi2 = np.radians(latitude2) 
    delta_phi = np.radians(latitude2 - latitude1) 
    delta_lamda = np.radians(longitude2 - longitude1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lamda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c


def manhattan(latitude1, longitude1, latitude2, longitude2) :
    dlat = np.abs(latitude2 - latitude1)
    dlon = np.abs(longitude2 - longitude1) 
    lat_km = np.radians(dlat) * 6371
    lon_km = np.radians(dlon) * 6371 * np.cos(np.radians((latitude1 + latitude2)/2)) 
    return lat_km + lon_km


def bearing(lat1, lon1, lat2, lon2):
    delta_lon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    
    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    
    return (bearing + 360) % 360 



train_df['plat'] = train_df['pickup_latitude']
train_df['plon'] = train_df['pickup_longitude']
train_df['dlat'] = train_df['dropoff_latitude']
train_df['dlon'] = train_df['dropoff_longitude']


train_df['distance_km'] = haversine(train_df['plat'],train_df['plon'],train_df['dlat'],train_df['dlon'])


train_df['manhattan_km'] = manhattan(train_df['plat'],train_df['plon'],train_df['dlat'],train_df['dlon'])


train_df['bearing'] = bearing(train_df['plat'],train_df['plon'],train_df['dlat'],train_df['dlon'])


train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].map({'N': 0,'Y' : 1})
train_df['store_and_fwd_flag'].value_counts()


train_df.info()


def is_rush_hr(hour) :
    return (hour >= 0) & (hour <= 12) | (hour >= 16) & (hour <= 19)
train_df['is_rush_hour'] = train_df['hour'].apply(is_rush_hr).astype(int)


train_df['lat_rounded'] = train_df['pickup_latitude'].round(2)
train_df['long_rounded'] = train_df['pickup_longitude'].round(2)


train_df['same_zone'] = (
    (train_df['pickup_latitude'].round(2) == train_df['dropoff_latitude'].round(2)) &
    (train_df['pickup_longitude'].round(2) == train_df['dropoff_longitude'].round(2))
).astype(int)



train_df['hour_avg_duration'] = train_df['hour'].map(hour_duration)



feature_col = [
    'hour',
    'day',
    'weekday',
    'weekend',
    'month',
    'year',
    'plat',
    'plon',
    'dlat',
    'dlon',
    'is_rush_hour',
    'lat_rounded',
    'long_rounded',
    'is_daytime',
    'passenger_count',
    'store_and_fwd_flag',
    'distance_km',
    'manhattan_km',
    'bearing',
    'same_zone'
    'hour_avg_duration'
]
x_train = train_df[feature_col]
y_train = train_df['trip_duration'] 


y_train_log = np.log1p(y_train)


from sklearn.model_selection import train_test_split 
x_tr, x_val, y_tr, y_val = train_test_split(x_train,y_train_log, test_size = 0.2, random_state = 42)


import xgboost as xgb 
model = xgb.XGBRegressor (
    tree_method = "gpu_hist",
    predictor = "gpu_predictor",
    n_estimators = 500,
    learning_rate = 0.1,
    max_depth = 7,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42,
    n_jobs = -1,
)
model.fit(x_tr,y_tr)


from sklearn.metrics import mean_squared_log_error

y_val_pred_log = model.predict(x_val)          
y_val_pred = np.expm1(y_val_pred_log)         
y_val_true = np.expm1(y_val)                  

rmsle = np.sqrt(mean_squared_log_error(y_val_true, y_val_pred))
print(rmsle)


test_df = pd.read_csv('/kaggle/input/nyc-taxi-trip-duration/test.zip',compression = 'zip')
test_df.head()


test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])
test_df['hour'] = test_df['pickup_datetime'].dt.hour
test_df['day'] = test_df['pickup_datetime'].dt.day
test_df['weekday'] = test_df['pickup_datetime'].dt.weekday
test_df['month'] = test_df['pickup_datetime'].dt.month
test_df['year'] = test_df['pickup_datetime'].dt.year
test_df['weekend'] = test_df['weekday'].isin([5,6]).astype(int)
test_df['is_daytime'] = test_df['hour'].between(6,18).astype(int)



def haversine(latitude1, longitude1, latitude2, longitude2) :
    phi1 = np.radians(latitude1)
    phi2 = np.radians(latitude2)
    delta_phi = np.radians(latitude2 - latitude1) 
    delta_lamda = np.radians(longitude2 - longitude1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lamda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c


def manhattan(latitude1, longitude1, latitude2, longitude2) :
    dlat = np.abs(latitude2 - latitude1)
    dlon = np.abs(longitude2 - longitude1) 
    lat_km = np.radians(dlat) * 6371
    lon_km = np.radians(dlon) * 6371 * np.cos(np.radians((latitude1 + latitude2)/2)) 
    return lat_km + lon_km


def bearing(lat1, lon1, lat2, lon2):
    delta_lon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    
    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    
    return (bearing + 360) % 360 



test_df['plat'] = test_df['pickup_latitude']
test_df['plon'] = test_df['pickup_longitude']
test_df['dlat'] = test_df['dropoff_latitude'] 
test_df['dlon'] = test_df['dropoff_longitude']


test_df['distance_km'] = haversine(test_df['plat'],test_df['plon'],test_df['dlat'],test_df['dlon'])
test_df['manhattan_km'] = manhattan(test_df['plat'],test_df['plon'],test_df['dlat'],test_df['dlon'])


test_df['bearing'] = bearing(test_df['plat'],test_df['plon'],test_df['dlat'],test_df['dlon'])


test_df['store_and_fwd_flag'] = test_df['store_and_fwd_flag'].map({'N': 0,'Y' : 1})   


def is_rush_hr(hour) :
    return (hour >= 0) & (hour <= 12) | (hour >= 16) & (hour <= 19)
test_df['is_rush_hour'] = test_df['hour'].apply(is_rush_hr).astype(int)


test_df['lat_rounded'] = test_df['pickup_latitude'].round(2)
test_df['long_rounded'] = test_df['pickup_longitude'].round(2)


test_df['same_zone'] = (
    (test_df['pickup_latitude'].round(2) == test_df['dropoff_latitude'].round(2)) &
    (test_df['pickup_longitude'].round(2) == test_df['dropoff_longitude'].round(2))
).astype(int)


test_df['hour_avg_duration'] = test_df['hour'].map(hour_duration)



feature_col = [
    'hour',
    'day',
    'weekday',
    'weekend',
    'month',
    'year',
    'plat',
    'plon',
    'dlat',
    'dlon',
    'is_rush_hour',
    'lat_rounded',
    'long_rounded',
    'is_daytime',
    'passenger_count',
    'store_and_fwd_flag',
    'distance_km',
    'manhattan_km',
    'bearing',
    'same_zone',
    'hour_avg_duration'
]
x_test = test_df[feature_col]


y_test_pred_log = model.predict(x_test)
y_test_pred = np.expm1(y_test_pred_log)


submission = pd.DataFrame ({
    'id' : test_df['id'],
    'trip_duration' : y_test_pred 
})
submission.to_csv('submission.csv',index = False) 

