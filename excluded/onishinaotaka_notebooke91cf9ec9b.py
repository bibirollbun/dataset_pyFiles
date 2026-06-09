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


df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", parse_dates=["pickup_datetime"], nrows=500000)
df = df[df['pickup_datetime'] >= '2009-01-01']
df['year'] = df['pickup_datetime'].dt.year
yearly_avg_fare = df.groupby('year')['fare_amount'].mean()
print(yearly_avg_fare)


yearly_stats = df.groupby('year')['fare_amount'].agg(['mean', 'sum', 'count'])
print(yearly_stats)


df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 200)]


yearly_stats = df.groupby('year')['fare_amount'].agg(['mean', 'sum', 'count'])
print(yearly_stats)


import matplotlib.pyplot as plt

yearly_avg_fare.plot(kind='bar', title='Yearly Average Taxi Fare (from 2009)')
plt.xlabel('Year')
plt.ylabel('Average Fare ($)')
plt.grid(True)
plt.tight_layout()
plt.show()


df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
fare_by_passenger = df.groupby('passenger_count')['fare_amount'].mean()
print(fare_by_passenger)


fare_stats = df.groupby('passenger_count')['fare_amount'].agg(['mean', 'median', 'std', 'count'])
print(fare_stats)


fare_by_passenger.plot(kind='bar', title='Average Fare by Passenger Count', color='skyblue')
plt.xlabel('Passenger Count')
plt.ylabel('Average Fare ($)')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()


df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
df['weekday'] = df['pickup_datetime'].dt.dayofweek
avg_fare_by_weekday = df.groupby('weekday')['fare_amount'].mean()
weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
avg_fare_by_weekday.index = [weekday_labels[i] for i in avg_fare_by_weekday.index]
print(avg_fare_by_weekday)


avg_fare_by_weekday.plot(kind='bar', title='Average Taxi Fare by Weekday')
plt.ylabel('Average Fare ($)')
plt.xlabel('Weekday')
plt.show()


df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", parse_dates=["pickup_datetime"], nrows=500000)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)

df = df[
    (df["distance_km"] > 0) & (df["distance_km"] <= 100) &
    (df["fare_amount"] > 0) & (df["fare_amount"] <= 200)
]

df["distance_bin"] = df["distance_km"].round().astype(int)

fare_by_distance = df.groupby("distance_bin")["fare_amount"].mean().reset_index()

print(fare_by_distance)


plt.figure(figsize=(10,6))
plt.plot(fare_by_distance["distance_bin"], fare_by_distance["fare_amount"], marker='o')
plt.title("Average Fare by Distance (0-35 km)")
plt.xlabel("Distance (km)")
plt.ylabel("Average Fare ($)")
plt.grid(True)
plt.show()


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(
    df["pickup_latitude"], df["pickup_longitude"],
    df["dropoff_latitude"], df["dropoff_longitude"]
)

df = df[
    (df["distance_km"] > 0) & (df["distance_km"] <= 27) &
    (df["fare_amount"] > 0) & (df["fare_amount"] <= 200)
]

df["distance_bin"] = df["distance_km"].round().astype(int)

fare_by_distance = df.groupby("distance_bin")["fare_amount"].mean().reset_index()

print(fare_by_distance)


plt.figure(figsize=(10,6))
plt.plot(fare_by_distance["distance_bin"], fare_by_distance["fare_amount"], marker='o')
plt.title("Average Fare by Distance (0-27 km)")
plt.xlabel("Distance (km)")
plt.ylabel("Average Fare ($)")
plt.grid(True)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000)  
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")


train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["day"] = train_df["pickup_datetime"].dt.dayofweek

train_df = train_df.dropna()
train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] < 200)]

features = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count", "hour", "day"]
X = train_df[features]
y = train_df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.2f}")


test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["day"] = test_df["pickup_datetime"].dt.dayofweek
X_test = test_df[features]

test_df["fare_amount"] = model.predict(X_test)
submission = test_df[["key", "fare_amount"]]
submission.to_csv("submission.csv", index=False)




