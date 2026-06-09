# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        paths.append(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# unpack data to working folder 
import zipfile
for file in paths:
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall('/kaggle/working')


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from haversine import haversine
from sklearn.model_selection import train_test_split


df = pd.read_csv("/kaggle/working/train.csv")

train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

X_train = train.drop(columns=['trip_duration'])
y_train = train['trip_duration']
X_test = test.drop(columns=['trip_duration'])
y_test = test['trip_duration']


df.head()


# 1. anomalies
df['pickup_date'] = pd.to_datetime(df['pickup_datetime']).dt.date
daily_counts = df.groupby('pickup_date').size().reset_index(name='trip_count')
anomalous_dates = ['2016-01-21', '2016-01-22', '2016-01-23', '2016-05-27', '2016-05-28']
df['is_anomaly'] = df['pickup_date'].astype(str).isin(anomalous_dates).astype(int)


# 2. temporal features
df['pickup_day_of_year'] = pd.to_datetime(df['pickup_datetime']).dt.dayofyear
df['pickup_day_of_week'] = pd.to_datetime(df['pickup_datetime']).dt.dayofweek
df['pickup_hour_of_day'] = pd.to_datetime(df['pickup_datetime']).dt.hour


#3. Traffic congestion and 5. trip_distance
def vec_haversine(row):
    return haversine((row['pickup_latitude'], row['pickup_longitude']),(row['dropoff_latitude'], row['dropoff_longitude']))

df['trip_distance'] = df.apply(vec_haversine, axis=1)

def avg_speed(row):
    return row['trip_distance'] / row['trip_duration']

df['avg_speed'] = df.apply(avg_speed, axis=1)



fig, ax = plt.subplots(figsize=(10, 10))
df_grb = df.groupby(['pickup_day_of_week', 'pickup_hour_of_day'])[['avg_speed']].mean().reset_index()
df_temp = df_grb.pivot(index='pickup_hour_of_day', columns='pickup_day_of_week', values='avg_speed')
plt.figure(figsize=(20,20))
sns.heatmap(df_temp, annot=True, ax=ax)
plt.show()


df['traffic'] = 0
df.loc[(((8 <= df['pickup_hour_of_day']) & (df['pickup_hour_of_day'] <= 18)) & ((0 <= df['pickup_day_of_week']) & (df['pickup_day_of_week'] <= 4))), 'traffic'] = 1


# 4. Passenger count
df['log_trip_duration'] = df['trip_duration'].apply(np.log1p)
plt.figure(figsize=(10,10))
sns.boxplot(x='passenger_count', y="log_trip_duration", data=df)


print(df['passenger_count'].value_counts()[[0, 7]])
#most of the trips with "0" passenger count are very short, so it might have been a fake trip.
#let's drop those
df = df[~((df['passenger_count'] == 0) & (df['trip_duration'] < 120))]


#6. airport proximity
jfk_coords = (40.645730, -73.784467)
ee_coords = (40.773130, -73.873494)
def airport_dropoff(row):
    if (haversine((row['dropoff_latitude'], row['dropoff_longitude']), jfk_coords) < 1) or (haversine((row['dropoff_latitude'], row['dropoff_longitude']), ee_coords) < 0.5):
        return True
    else:
        return False

def airport_pickup(row):
    if ((haversine((row['pickup_latitude'], row['pickup_longitude']), jfk_coords)) < 1) or ((haversine((row['pickup_latitude'], row['pickup_longitude']), ee_coords)) < 0.5):
        return True
    else:
        return False
df['airport_dropoff'] = df.apply(airport_dropoff, axis=1)
df['airport_pickup'] = df.apply(airport_pickup, axis=1)


# 7. Start/end NYC
#created the bounds by estimating on google maps
nyc_bounds = {
    'lat_min': 40.5,   
    'lat_max': 40.92,
    'lon_min': -74.3,
    'lon_max': -73.91
}
df['nyc_pickup'] = (df['pickup_latitude'] >= nyc_bounds['lat_min']) & (df['pickup_latitude'] <= nyc_bounds['lat_max']) & (df['pickup_longitude'] >= nyc_bounds['lon_min']) & (df['pickup_longitude'] <= nyc_bounds['lon_max'])
df['nyc_dropoff'] = (df['dropoff_latitude'] >= nyc_bounds['lat_min']) & (df['dropoff_latitude'] <= nyc_bounds['lat_max']) & (df['dropoff_longitude'] >= nyc_bounds['lon_min']) & (df['dropoff_longitude'] <= nyc_bounds['lon_max'])


#8. Improving distance calculation 
# if we're in Manhattan it makes more sense for us to use the M-distance
man_bounds = {
    'lat_min': 40.701,   
    'lat_max': 40.88,
    'lon_min': -74.01,
    'lon_max': -73.91
}

def is_in_manhattan(lat, lon):
    return (
        (lat >= man_bounds['lat_min']) & 
        (lat <= man_bounds['lat_max']) &
        (lon >= man_bounds['lon_min']) & 
        (lon <= man_bounds['lon_max'])
    )
    
lat_km_per_degree = 111
lon_km_per_degree = 111 * np.cos(np.radians(df['pickup_latitude']))

manhattan_distance = (
    abs(df['pickup_latitude'] - df['dropoff_latitude']) * lat_km_per_degree + 
    abs(df['pickup_longitude'] - df['dropoff_longitude']) * lon_km_per_degree
)

both_in_manhattan = (
    is_in_manhattan(df['pickup_latitude'], df['pickup_longitude']) & 
    is_in_manhattan(df['dropoff_latitude'], df['dropoff_longitude'])
)

#we update if the whole trip is in manhattan
df['trip_distance'] = np.where(
    both_in_manhattan,
    manhattan_distance,
    df['trip_distance']
)

#makes sense to recalculate average speed
df = df.drop('avg_speed', axis=1)
df['avg_speed'] = df.apply(avg_speed, axis=1)


# 9. Geospatial outliers
print(df[df['pickup_longitude'] < -75])


for col in ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']:
    print(f"\n{col}:")
    print(f"  Min: {df[col].min()}")
    print(f"  Max: {df[col].max()}")
    print(f"  Mean: {df[col].mean():.4f}")
    print(f"  Std: {df[col].std():.4f}")
    
    # Show extreme values
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]


df = df[~((df['pickup_latitude'] > 42.7) | (df['pickup_latitude'] < 38))]
df = df[~((df['pickup_longitude'] > -68) | (df['pickup_longitude'] < -80))]
df = df[~((df['dropoff_latitude'] > 42.7) | (df['dropoff_latitude'] < 38))]
df = df[~((df['dropoff_longitude'] > -68) | (df['dropoff_longitude'] < -80))]

#I made the bounds very generous


plt.hist(df[df['trip_duration'] < df['trip_duration'].quantile(.99)]['trip_duration'], bins=100)
plt.title('Taxi trip duration')
plt.show()


df = df[~((df['trip_duration'] < 60 ))]
df = df[~((df['trip_duration'] > 60 * 60 * 6 ))]


df['trip_duration'].max()





df.head()


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


numeric = ['passenger_count', 'avg_speed', 'trip_distance', 'pickup_hour_of_day', 'pickup_day_of_week']
categorical = ['airport_dropoff', 'airport_pickup', 'traffic']

column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('scaling', StandardScaler(), numeric)
])



from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# Linear Model

def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

train, test = train_test_split(df, test_size=0.3, random_state=42)

x_train = train.drop(columns=['trip_duration', 'log_trip_duration'])
y_train = train[['log_trip_duration']]
x_test = test.drop(columns=['trip_duration', 'log_trip_duration'])
y_test = test[['log_trip_duration']]

dummy_pipeline = Pipeline([
    ('preprocessing', column_transformer),
    ('regressor', DummyRegressor(strategy='mean'))
])

dummy_pipeline.fit(x_train, y_train)
y_pred_dummy = dummy_pipeline.predict(x_test)

print(f"Dummy Train RMSLE: {rmsle(y_train, dummy_pipeline.predict(x_train)):.4f}")
print(f"Dummy Test RMSLE:  {rmsle(y_test, y_pred_dummy):.4f}\n")


#LinearRegression
column_transformer.fit(x_train)

feature_names = []

ohe_features = column_transformer.named_transformers_['ohe'].get_feature_names_out(categorical)
feature_names.extend(ohe_features)

feature_names.extend(numeric)

print(feature_names)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42)
}

results = {}
feat_importance = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessing', column_transformer),
        ('regressor', model)
    ])


    pipeline.fit(x_train, y_train)
    y_pred_train = pipeline.predict(x_train)
    y_pred_test = pipeline.predict(x_test)
    
    train_rmsle = rmsle(y_train, y_pred_train)
    test_rmsle = rmsle(y_test, y_pred_test)
    
    results[name] = {
        'model': pipeline,
        'train_rmsle': train_rmsle,
        'test_rmsle': test_rmsle
    }

    feat_importance[name] = {
        'model': pipeline, 
        'feature_importance': pipeline.named_steps['regressor'].coef_
    }
    
    print(f"\n{name}:")
    print(f"  Train RMSLE: {train_rmsle:.4f}")
    print(f"  Test RMSLE:  {test_rmsle:.4f}")



#Feature importance
for model_name in feat_importance:
    print("Model:", model_name)
    importance = feat_importance[model_name]['feature_importance']
    importance = importance.flatten()
    
    for i, v in enumerate(importance):
        print('Feature: %-40s, Score: %.5f' % (feature_names[i], v))
    print()


from sklearn.ensemble import RandomForestRegressor

x_train = train.drop(columns=['trip_duration', 'log_trip_duration'])
y_train = train['log_trip_duration']
x_test = test.drop(columns=['trip_duration', 'log_trip_duration'])
y_test = test['log_trip_duration']

rf_pipeline = Pipeline([
    ('preprocessing', column_transformer),
    ('regressor', RandomForestRegressor(random_state=42, verbose=1, n_jobs=-1))
])

rf_pipeline.fit(x_train, y_train)

y_pred_train_rf = rf_pipeline.predict(x_train)
y_pred_test_rf = rf_pipeline.predict(x_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))

feature_importance_rf = rf_pipeline.named_steps['regressor'].feature_importances_


print(f" Train RMSE: {train_rmse:.4f}")
print(f" Test RMSE:  {test_rmse:.4f}")


submission = pd.DataFrame({
    'id': test['id'],
    'trip_duration': y_pred_test_rf
})

submission.to_csv('submission.csv', index=False)

print("Submission file created!")
print(submission.head())

