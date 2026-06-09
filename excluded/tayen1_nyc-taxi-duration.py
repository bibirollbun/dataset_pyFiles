# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

output_dir = '/kaggle/working/'

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        data_path_name = os.path.join(dirname, filename)
        print(data_path_name)
        with zipfile.ZipFile(data_path_name, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

print("Finish unzip")

for dirname, _, filenames in os.walk(output_dir):
    for filename in filenames:
        print(os.path.join(output_dir, filename))


import matplotlib.pyplot as plt
import seaborn as sns


test_df = pd.read_csv("/kaggle/working/test.csv")
train_df = pd.read_csv("/kaggle/working/train.csv")
sample_submission_df = pd.read_csv("/kaggle/working/sample_submission.csv")


train_df.shape


test_df.shape


train_df.head()


test_df.head()


train_df.info()


# add a column convert trip_duration sec to minutes
train_df['trip_duration_min'] = train_df['trip_duration'] / 60


train_df['trip_duration_hour'] = train_df['trip_duration_min'] / 60


train_df.describe()


trip_duration_min = train_df['trip_duration_min']
# Create 15-minute bins (0–480 mins = 8 hours)
bins = np.arange(0, 481, 15)
labels = [f"{i}-{i+15}" for i in bins[:-1]]

# Bin the data and count trips per interval
trip_counts = pd.cut(trip_duration_min, bins=bins, labels=labels, right=False).value_counts().sort_index()

# Create a summary DataFrame
trip_stats = pd.DataFrame({
    'Duration Interval (min)': trip_counts.index,
    'Trip Count': trip_counts.values
})

trip_stats



plt.figure(figsize=(10, 5))
sns.histplot(train_df['trip_duration_hour'], bins = 100,binwidth = 0.05, color='orange')
plt.title('Trip Duration (hr)')
plt.xlabel('Trip Duration (hr)')
plt.ylabel('Frequency')
plt.xlim(0,1)
#plt.ylim(0, 20000)
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(np.log1p(train_df['trip_duration']), bins=100, kde=True, color='orange')
plt.title('Log-Transformed Trip Duration')
plt.xlabel('log(Trip Duration + 1)')
plt.ylabel('Frequency')
plt.show()


# convert pickup_datetime to datetime
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['pickup_hour'] = train_df['pickup_datetime'].dt.hour
train_df['pickup_day'] = train_df['pickup_datetime'].dt.dayofweek


plt.figure(figsize=(12,5))
sns.countplot(x='pickup_hour', data=train_df)
plt.title('Trips by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Trip Count')
plt.show()


weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
train_df['pickup_day_name'] = train_df['pickup_day'].map({
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
    3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
})
plt.figure(figsize=(10,5))
sns.countplot(x='pickup_day_name', data=train_df, order=weekday_labels)
plt.title('Trips by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Trip Count')
plt.show()


hourly_avg = train_df.groupby('pickup_hour')['trip_duration'].mean()
plt.figure(figsize=(12, 6))
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker='o')

plt.xticks(range(0, 24))  # X-axis from 0 to 23
plt.title('Average Trip Duration by Hour of Day')
plt.xlabel('Hour of Day (0 = midnight)')
plt.ylabel('Average Trip Duration (seconds)')
plt.grid(True)
plt.tight_layout()
plt.show()


avg_by_day = train_df.groupby('pickup_day_name')['trip_duration'].mean()

avg_by_day = avg_by_day.reindex(weekday_labels)
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_by_day.index, y=avg_by_day.values, marker='o')

plt.title('Average Trip Duration by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Trip Duration (seconds)')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,6))
sns.scatterplot(x=train_df['pickup_longitude'], y=train_df['pickup_latitude'], s=1, alpha=0.3)
plt.title('Pickup Locations (Zoomed)')
plt.xlim(-74.05, -73.75)
plt.ylim(40.63, 40.85)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=train_df[train_df['trip_duration'] < 3600],  # remove extreme outliers
    x='pickup_longitude',
    y='pickup_latitude',
    hue=np.log1p(train_df[train_df['trip_duration'] < 3600]['trip_duration']),
    palette='viridis',
    alpha=0.5,
    legend=False
)
plt.title('Pickup Locations Colored by Trip Duration (log scale)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-74.05, -73.75)
plt.ylim(40.63, 40.85)
plt.show()


trip_counts = train_df['passenger_count'].value_counts().sort_index()
trip_counts


plt.figure(figsize=(8,5))
sns.countplot(x='passenger_count', data=train_df)
plt.title('Trip Count by Passenger Count')
plt.show()


passenger_avg = train_df.groupby('passenger_count')['trip_duration'].mean()
plt.figure(figsize=(10, 5))
sns.lineplot(x=passenger_avg.index, y=passenger_avg.values, marker='o')

plt.title('Average Trip Duration by Passenger Count')
plt.xlabel('Passenger Count')
plt.ylabel('Average Trip Duration (seconds)')
plt.xticks(passenger_avg.index)  # Ensure all categories show
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 5))
sns.countplot(x='vendor_id', data=train_df, palette='Set2')

plt.title('Trip Count by Vendor ID')
plt.xlabel('Vendor ID')
plt.ylabel('Number of Trips')
plt.xticks([0, 1], labels=['Vendor 1', 'Vendor 2'])  # optional custom labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


vendor_avg_duration = train_df.groupby('vendor_id')['trip_duration'].mean()

plt.figure(figsize=(6, 5))
sns.barplot(x=vendor_avg_duration.index, y=vendor_avg_duration.values, palette='Set2')

plt.title('Average Trip Duration by Vendor ID')
plt.xlabel('Vendor ID')
plt.ylabel('Avg Trip Duration (seconds)')
plt.xticks([0, 1], labels=['Vendor 1', 'Vendor 2'])  # Optional: label mapping
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 5))
sns.countplot(x='store_and_fwd_flag', data=train_df, palette='Set3')

plt.title('Trip Count by Store-and-Forward Flag')
plt.xlabel('store_and_fwd_flag (N = Realtime, Y = Stored)')
plt.ylabel('Number of Trips')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


avg_duration = train_df.groupby('store_and_fwd_flag')['trip_duration'].mean()

plt.figure(figsize=(6, 5))
sns.barplot(x=avg_duration.index, y=avg_duration.values, palette='Set3')

plt.title('Average Trip Duration by Store-and-Forward Flag')
plt.xlabel('store_and_fwd_flag (N = Realtime, Y = Stored)')
plt.ylabel('Avg Trip Duration (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Encode store_and_fwd_flag 
train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].map({'N': 0, 'Y': 1})


train_df.head()


# only choose the features, avoid the feature that for EDA
features = [
    'vendor_id', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'passenger_count', 'store_and_fwd_flag',
    'pickup_hour', 'pickup_day'
]


# target use raw and log
X = train_df[features]
y_raw = train_df['trip_duration']
y_log = np.log1p(train_df['trip_duration'])


#split train and test
X_train, X_valid, y_train_raw, y_valid_raw = train_test_split(X, y_raw, test_size=0.2, random_state=42)
_, _, y_train_log, y_valid_log = train_test_split(X, y_log, test_size=0.2, random_state=42)


# bulid the model
model_raw = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
model_log = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)


# train model
model_raw.fit(X_train, y_train_raw)
model_log.fit(X_train, y_train_log)


# predict
pred_raw = model_raw.predict(X_valid)
pred_log = model_log.predict(X_valid) 


from sklearn.metrics import mean_squared_error
# Clip predictions to avoid negatives
pred_raw_clipped = np.maximum(pred_raw, 0)

# Now calculate RMSLE safely
rmsle＿raw = mean_squared_error(np.log1p(y_valid_raw), np.log1p(pred_raw_clipped), squared=False)



rmsle_log = mean_squared_error(y_valid_log, pred_log, squared=False)

print(f"RMSLE: {rmsle_raw:.4f}")
print(f"RMSLE from log-transformed model: {rmsle_log:.4f}")


# harversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    return km

train_df['haversine_distance'] = haversine_distance(
    train_df['pickup_latitude'],
    train_df['pickup_longitude'],
    train_df['dropoff_latitude'],
    train_df['dropoff_longitude']
)


train_df['manhattan_distance'] = (
    np.abs(train_df['dropoff_longitude'] - train_df['pickup_longitude']) +
    np.abs(train_df['dropoff_latitude'] - train_df['pickup_latitude'])
)


# rush hour : 7-9am and 4-7pm
train_df['rush_hour'] = train_df['pickup_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)


train_df['weekend'] = train_df['pickup_day'].apply(lambda x: 1 if x >= 5 else 0)


features = [
    'vendor_id', 'passenger_count', 'store_and_fwd_flag',
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
    'pickup_hour', 'pickup_day',
    'haversine_distance', 'manhattan_distance',
    'rush_hour', 'weekend'
]


X = train_df[features]
y = np.log1p(train_df['trip_duration'])


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)



model.fit(X_train, y_train)


y_pred = model.predict(X_valid)


rmsle = mean_squared_error(y_valid, y_pred, squared=False)

print(f"RMSLE after Feature Engineering: {rmsle:.4f}")


train_df['trip_duration'].describe()


# Remove trip duration lower than 10 sec and also higher than 3600 sec 
lower_bound = 10
upper_bound = 3600

train_df_cleaned = train_df[(train_df['trip_duration'] >= lower_bound) & (train_df['trip_duration'] <= upper_bound)]

print(f"Before handling outliers, there are {train_df.shape[0]} rows. After handling outliers, there are {train_df_cleaned.shape[0]} rows.")


features = [
    'vendor_id', 'passenger_count', 'store_and_fwd_flag',
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
    'pickup_hour', 'pickup_day',
    'haversine_distance', 'manhattan_distance',
    'rush_hour', 'weekend'
]


X = train_df_cleaned[features]
y = np.log1p(train_df_cleaned['trip_duration'])


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


model.fit(X_train, y_train)


y_pred = model.predict(X_valid)


rmsle = mean_squared_error(y_valid, y_pred, squared=False)

print(f"RMSLE after removing outliers: {rmsle:.4f}")


# 1. Data Preprocessing

# store_and_fwd_flag: N=0, Y=1
test_df['store_and_fwd_flag'] = test_df['store_and_fwd_flag'].map({'N':0, 'Y':1})

# pickup_datetime ➔  pickup_hour, pickup_day
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])
test_df['pickup_hour'] = test_df['pickup_datetime'].dt.hour
test_df['pickup_day'] = test_df['pickup_datetime'].dt.dayofweek

# haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

test_df['haversine_distance'] = haversine_distance(
    test_df['pickup_latitude'],
    test_df['pickup_longitude'],
    test_df['dropoff_latitude'],
    test_df['dropoff_longitude']
)

# manhattan distance
test_df['manhattan_distance'] = (
    np.abs(test_df['dropoff_longitude'] - test_df['pickup_longitude']) +
    np.abs(test_df['dropoff_latitude'] - test_df['pickup_latitude'])
)

# rush_hour, weekend
test_df['rush_hour'] = test_df['pickup_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
test_df['weekend'] = test_df['pickup_day'].apply(lambda x: 1 if x >= 5 else 0)

# 2. choose features same as train data
features = [
    'vendor_id', 'passenger_count', 'store_and_fwd_flag',
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
    'pickup_hour', 'pickup_day',
    'haversine_distance', 'manhattan_distance',
    'rush_hour', 'weekend'
]

X_test = test_df[features]

# 3. use the final model to predict
test_preds_log = model.predict(X_test)

# log convert back 
test_preds = np.expm1(test_preds_log)

# in case there is negative value
test_preds = np.clip(test_preds, 1, None)

# 4. submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'], 
    'trip_duration': test_preds
})

# 5. csv file
submission.to_csv('submission.csv', index=False)

print("✅ Successfully created submission.csv!")





