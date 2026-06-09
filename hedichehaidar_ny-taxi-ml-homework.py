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
for file in paths[:3]:
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall('/kaggle/working')


import numpy as np 
import pandas as pd 
from haversine import haversine, Unit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("/kaggle/working/train.csv")
df_test = pd.read_csv('/kaggle/input/test-real/test_real.csv')
df_test.shape




# Datetime processing
for _df in [df, df_test]:
    _df['pickup_datetime'] = pd.to_datetime(_df['pickup_datetime'])

    _df['pickup_date'] = _df['pickup_datetime'].dt.date

    _df['pickup_day_of_year'] = _df['pickup_datetime'].dt.dayofyear
    _df['pickup_day_of_week'] = _df['pickup_datetime'].dt.dayofweek  
    _df['pickup_hour_of_day'] = _df['pickup_datetime'].dt.hour


# Daily trip count & anomaly detection
daily_counts = df.groupby('pickup_date').size().rename('trip_count')

# Quick look at daily demand
plt.figure()
daily_counts.plot()
plt.title("Daily Trip Counts")
plt.xlabel("Date")
plt.ylabel("Number of trips")
plt.show()

# Identify two anomalies: the two with min demands

low_anomaly_dates = daily_counts.nsmallest(2).index.tolist()

for _df in [df, df_test]:
    _df['is_anomaly'] = _df['pickup_date'].isin(low_anomaly_dates).astype(int)


# Haversine distance & improved distance feature
def compute_haversine_distance(row):
    return haversine(
        (row['pickup_latitude'], row['pickup_longitude']),
        (row['dropoff_latitude'], row['dropoff_longitude']),
        unit=Unit.KILOMETERS
    )

def compute_manhattan_distance(row):
    lat1 = row['pickup_latitude']
    lon1 = row['pickup_longitude']
    lat2 = row['dropoff_latitude']
    lon2 = row['dropoff_longitude']

    d1 = haversine((lat1, lon1), (lat1, lon2), unit=Unit.KILOMETERS)
    d2 = haversine((lat1, lon2), (lat2, lon2), unit=Unit.KILOMETERS)
    return d1 + d2

for _df in [df, df_test]:
    _df['trip_distance'] = _df.apply(compute_haversine_distance, axis=1)
    _df['manhattan_distance'] = _df.apply(compute_manhattan_distance, axis=1)


#Drop trips with 0 or >6 as outliers.
df = df[df['passenger_count'].between(1, 6)].copy()
df_test = df_test[df_test['passenger_count'].between(1, 6)].copy()

print(df.head())
print(df_test.head())


# Traffic congestion: average speed + congestion bin

df = df[df['trip_duration'] > 0].copy()
df['speed_kmh'] = df['trip_distance'] / (df['trip_duration'] / 3600)

# Inspect speed distribution
plt.figure()
df['speed_kmh'].hist(bins=100)
plt.xlim(0, 120)
plt.xlabel("Speed (km/h)")
plt.ylabel("Frequency")
plt.title("Speed distribution")
plt.show()

# Average speed per (day_of_week, hour_of_day)
speed_by_time = (
    df.groupby(['pickup_day_of_week', 'pickup_hour_of_day'])['speed_kmh']
      .mean()
      .rename('avg_speed_kmh')
      .reset_index()
)

# Define traffic congestion levels:
# heavy traffic (1) if avg speed <= 25th percentile of avg_speed_kmh
# light traffic (0) otherwise
speed_threshold = speed_by_time['avg_speed_kmh'].quantile(0.25)
print("Speed threshold for heavy traffic (km/h):", speed_threshold)

speed_by_time['traffic_congestion'] = (
    (speed_by_time['avg_speed_kmh'] <= speed_threshold).astype(int)
)

# Map back to each row
congestion_map = speed_by_time.set_index(
    ['pickup_day_of_week', 'pickup_hour_of_day']
)['traffic_congestion']

df['traffic_congestion'] = df.set_index(
    ['pickup_day_of_week', 'pickup_hour_of_day']
).index.map(congestion_map)
df['traffic_congestion'] = df['traffic_congestion'].fillna(0).astype(int)

# same for df_test
df_test['speed_kmh'] = df_test['trip_distance'] / (60.0 / 3600)
df_test['speed_kmh'] = df_test['speed_kmh'].fillna(df['speed_kmh'].median())
df_test['traffic_congestion'] = df_test.set_index(
    ['pickup_day_of_week', 'pickup_hour_of_day']
).index.map(congestion_map)
df_test['traffic_congestion'] = df_test['traffic_congestion'].fillna(0).astype(int)


# NYC boundaries and geospatial filter
NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92
NYC_LON_MIN, NYC_LON_MAX = -74.27, -73.68

def within_nyc(lat, lon):
    return (
        (lat >= NYC_LAT_MIN) & (lat <= NYC_LAT_MAX) &
        (lon >= NYC_LON_MIN) & (lon <= NYC_LON_MAX)
    )


for _df in [df, df_test]:
    _df['pickup_in_nyc'] = within_nyc(_df['pickup_latitude'], _df['pickup_longitude']).astype(int)
    _df['dropoff_in_nyc'] = within_nyc(_df['dropoff_latitude'], _df['dropoff_longitude']).astype(int)

# Geospatial outliers: coordinates far outside NYC
def geo_valid_mask(_df):
    return (
        within_nyc(_df['pickup_latitude'], _df['pickup_longitude']) &
        within_nyc(_df['dropoff_latitude'], _df['dropoff_longitude']) &
        (_df['trip_distance'] > 0) &
        (_df['trip_distance'] < 100) 
    )

geo_mask_train = geo_valid_mask(df)
print("Dropping geospatial outliers (train):", (~geo_mask_train).sum())
df = df[geo_mask_train].copy()

# same for df_test
geo_mask_test = geo_valid_mask(df_test)
print("Dropping geospatial outliers (test):", (~geo_mask_test).sum())
df_test = df_test[geo_mask_test].copy()


# Airport proximity features

JFK_COORDS = (40.6413, -73.7781)
LGA_COORDS = (40.7769, -73.8740)
EWR_COORDS = (40.6895, -74.1745)

def is_near_airport(lat, lon, airport_coords, radius_km=2.0):
    return haversine((lat, lon), airport_coords, unit=Unit.KILOMETERS) <= radius_km

def add_airport_features(_df):
    _df['pickup_is_jfk'] = _df.apply(
        lambda r: is_near_airport(r['pickup_latitude'], r['pickup_longitude'], JFK_COORDS), axis=1
    ).astype(int)
    _df['dropoff_is_jfk'] = _df.apply(
        lambda r: is_near_airport(r['dropoff_latitude'], r['dropoff_longitude'], JFK_COORDS), axis=1
    ).astype(int)

    _df['pickup_is_lga'] = _df.apply(
        lambda r: is_near_airport(r['pickup_latitude'], r['pickup_longitude'], LGA_COORDS), axis=1
    ).astype(int)
    _df['dropoff_is_lga'] = _df.apply(
        lambda r: is_near_airport(r['dropoff_latitude'], r['dropoff_longitude'], LGA_COORDS), axis=1
    ).astype(int)

    _df['pickup_is_ewr'] = _df.apply(
        lambda r: is_near_airport(r['pickup_latitude'], r['pickup_longitude'], EWR_COORDS), axis=1
    ).astype(int)
    _df['dropoff_is_ewr'] = _df.apply(
        lambda r: is_near_airport(r['dropoff_latitude'], r['dropoff_longitude'], EWR_COORDS), axis=1
    ).astype(int)
    return _df

df = add_airport_features(df)
df_test = add_airport_features(df_test)


# Trip duration distribution & cleaning
plt.figure()
df['trip_duration'].hist(bins=200)
plt.xlabel("Trip duration (seconds)")
plt.ylabel("Frequency")
plt.title("Trip duration distribution (raw)")
plt.show()

plt.figure()
np.log1p(df['trip_duration']).hist(bins=200)
plt.xlabel("log(1 + trip_duration)")
plt.ylabel("Frequency")
plt.title("Trip duration distribution (log-scale)")
plt.show()


# remove trips < 60s (too short, often errors) 
# remove trips > 3 hours (10800s) as extreme outliers
duration_mask = (df['trip_duration'] >= 60) & (df['trip_duration'] <= 10800)
print("Dropping duration outliers:", (~duration_mask).sum())
df = df[duration_mask].copy()


# Recreate train/test split for modeling

drop_cols = ['pickup_datetime', 'pickup_date', 'id']

X = df.drop(columns=['trip_duration'] + drop_cols)
y = df['trip_duration']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("Final train shape:", X_train.shape)
print("Final valid shape:", X_valid.shape)

X_test_kaggle = df_test.drop(columns=drop_cols, errors='ignore')
print("Kaggle test shape after feature engineering & cleaning:", X_test_kaggle.shape)



# Drop target + any non-feature helper columns
feature_cols = X_train.columns

# Manually group features
numeric_features = [
    'trip_distance',
    'manhattan_distance',
    'speed_kmh',
    'pickup_day_of_year',
    'pickup_latitude', 'pickup_longitude',
    'dropoff_latitude', 'dropoff_longitude'
]

categorical_features = [
    'pickup_day_of_week',      
    'pickup_hour_of_day',      
    'passenger_count',        
    'is_anomaly',
    'traffic_congestion',
    'pickup_in_nyc', 'dropoff_in_nyc',
    'pickup_is_jfk', 'dropoff_is_jfk',
    'pickup_is_lga', 'dropoff_is_lga',
    'pickup_is_ewr', 'dropoff_is_ewr'
]




from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)



from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import Pipeline
import numpy as np

def rmsle_score(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)

    return np.sqrt(
        np.mean(
            (np.log1p(y_pred) - np.log1p(y_true)) ** 2
        )
    )

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)




baseline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', DummyRegressor(strategy='median'))
])

baseline.fit(X_train, y_train_log)

y_pred_log = baseline.predict(X_valid)
y_pred = np.expm1(y_pred_log)
rmsle_dummy = rmsle_score(y_valid, y_pred)

print("Dummy Baseline RMSLE:", rmsle_dummy)



lin_reg = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', LinearRegression())
])

lin_reg.fit(X_train, y_train_log)
y_pred_log = lin_reg.predict(X_valid)
y_pred = np.expm1(y_pred_log)
rmsle_lr = rmsle_score(y_valid, y_pred)
print("Linear Regression RMSLE:", rmsle_lr)



ridge = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', Ridge(alpha=1.0))
])

ridge.fit(X_train, y_train_log)
y_pred_log = ridge.predict(X_valid)
y_pred = np.expm1(y_pred_log)
rmsle_ridge = rmsle_score(y_valid, y_pred)
print("Ridge Regression RMSLE:", rmsle_ridge)



lasso = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', Lasso(alpha=0.01, max_iter=5000))
])

lasso.fit(X_train, y_train_log)
y_pred_log = lasso.predict(X_valid)
y_pred = np.expm1(y_pred_log)
rmsle_lasso = rmsle_score(y_valid, y_pred)
print("Lasso Regression RMSLE:", rmsle_lasso)



models = ["Dummy", "LinearReg", "Ridge", "Lasso"]
scores = [rmsle_dummy, rmsle_lr, rmsle_ridge, rmsle_lasso]

for m, s in zip(models, scores):
    print(f"{m:10s} → RMSLE: {s:.5f}")




y_test_log_pred = lasso.predict(X_test_kaggle)

y_test_pred = np.expm1(y_test_log_pred)

y_test_pred = np.clip(y_test_pred, a_min=1, a_max=None).astype(int)

submission = pd.DataFrame({
    "id": df_test["id"],
    "trip_duration": y_test_pred
})

submission.to_csv("submission.csv", index=False)
submission.head()


