# Wargings
import warnings
warnings.filterwarnings("ignore")


# Import Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from statsmodels.stats.outliers_influence import variance_inflation_factor
import zipfile


# Import ML Libraries
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout


# Load Train & Test Datasets
with zipfile.ZipFile("/kaggle/input/nyc-taxi-trip-duration/train.zip") as z:
    with z.open(z.namelist()[0]) as f:
        train_df = pd.read_csv(f)

with zipfile.ZipFile("/kaggle/input/nyc-taxi-trip-duration/test.zip") as z:
    with z.open(z.namelist()[0]) as f:
        test_df = pd.read_csv(f)


# Train Dataset: Sample
train_df.sample(3)


# Datasets: Shape
print(f"\033[1mTrain Dataset - Shape\033[0m")
print(train_df.shape)
print("*" * 40)
print(f"\033[1mTest Dataset - Shape\033[0m")
print(test_df.shape)
print("*" * 40)


# Dataset: Columns 
print(f"\033[1mTrain Dataset - Columns\033[0m")
print(train_df.columns)
print("*" * 40)
print(f"\033[1mTest Dataset - Columns\033[0m")
print(test_df.columns)
print("*" * 40)


# Dataset: Missing Values
print(f"\033[1mTrain Dataset - Missing Values %\033[0m")
print(train_df.isnull().sum().sort_values(ascending=False)/len(train_df) *100)
print("*" * 40)
print(f"\033[1mTest Dataset - Missing Values %\033[0m")
print(test_df.isnull().sum().sort_values(ascending=False)/len(test_df) *100)
print("*" * 40)


# Dataset: Info
print(f"\033[1mTrain Dataset - Info\033[0m")
print(train_df.info())
print("*" * 40)
print(f"\033[1mTest Dataset - Info\033[0m")
print(test_df.info())
print("*" * 40)


# Special: Dtype
target_y = 'trip_duration'
y = ['trip_duration']


# Distance in Miles Function
def haversine_np(lat1, lon1, lat2, lon2):
    # Earth radius in miles
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# Apply Train Distance
train_df['distance'] = haversine_np(
    train_df['pickup_latitude'], train_df['pickup_longitude'],
    train_df['dropoff_latitude'], train_df['dropoff_longitude']
)

test_df['distance'] = haversine_np(
    test_df['pickup_latitude'], test_df['pickup_longitude'],
    test_df['dropoff_latitude'], test_df['dropoff_longitude']
)


# Visualize Trip Duration & Distance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x="distance", y="trip_duration", alpha=0.3)

plt.title("Trip Duration vs Distance", fontsize=16)
plt.xlabel("Distance (miles)", fontsize=12)
plt.ylabel("Trip Duration (seconds)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


# Anomaly Detection
clf = IsolationForest(random_state = 42, contamination = 0.01)
train_df['Anomaly'] = clf.fit_predict(train_df[['distance', 'trip_duration']])


# Visualize Anomaly
plt.figure(figsize=(10, 6))

plt.scatter(train_df.loc[train_df.Anomaly == -1, ['distance']], 
                 train_df.loc[train_df.Anomaly == -1, ['trip_duration']], c='red')

plt.scatter(train_df.loc[train_df.Anomaly == 1, ['distance']], 
                 train_df.loc[train_df.Anomaly == 1, ['trip_duration']], c='green')

plt.title("Outlier vs Normal Trips", fontsize=16)
plt.xlabel("Distance (miles)", fontsize=12)
plt.ylabel("Trip Duration (seconds)", fontsize=12)
plt.legend(title='Trip Type')
plt.grid(True)
plt.tight_layout()
plt.show()


# Remove Anomalies
train_df = train_df.loc[train_df['Anomaly'] == 1].copy()


# Visualize Trip Duration & Distance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x="distance", y="trip_duration", alpha=0.3)

plt.title("Trip Duration vs Distance - Anomaly Removed", fontsize=16)
plt.xlabel("Distance (miles)", fontsize=12)
plt.ylabel("Trip Duration (seconds)", fontsize=12)
plt.grid(True)
plt.tight_layout()


# Convert Datetime
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

test_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
test_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

train_df['date'] = train_df['pickup_datetime'].dt.date
test_df['date'] = test_df['pickup_datetime'].dt.date



# Visualize Distance by Date
plt.figure(figsize=(12, 6))
sns.lineplot(x="date", y="distance", data=train_df, ci=None)
plt.title("Average Distance by Date", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Distance (miles)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



# Visualize Trip Duration by Date
plt.figure(figsize=(12, 6))
sns.lineplot(x="date", y="trip_duration", data=train_df, ci=None)
plt.title("Average Trip Duration by Date", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Trip Duration (seconds)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Adapt Data for Prophet: Distance
data = train_df.groupby(['date'])['distance'].agg('sum')
data = pd.DataFrame({'date':data.index, 'distance':data.values})
data['date'] = pd.to_datetime(data['date'])
data.rename(columns = {'distance': 'y', 'date': 'ds'}, inplace = True)


# Load Prophet
m = Prophet(seasonality_mode='additive').fit(data)


# Future Dataset
future = m.make_future_dataframe(periods = 30)


# Forecast
forecast = m.predict(future)


# Visualize Prophet Forecast: Distance
fig = m.plot(forecast)


# Adapt Data for Prophet: Trip Duration
data = train_df.groupby(['date'])['trip_duration'].agg('sum')
data = pd.DataFrame({'date':data.index, 'trip_duration':data.values})
data['date'] = pd.to_datetime(data['date'])
data.rename(columns = {'trip_duration': 'y', 'date': 'ds'}, inplace = True)


# Load Prophet
m = Prophet(seasonality_mode='additive').fit(data)


# Future Dataset
future = m.make_future_dataframe(periods = 30)


# Forecast
forecast = m.predict(future)


# Visualize Prophet Forecast: Trip Duration
fig = m.plot(forecast)


# Load/Fit KNN Pickup
kmeans = KMeans(n_clusters=5, random_state=42).fit(train_df[['pickup_longitude','pickup_latitude']])


# Cluster Pickup
pickup_clusters = kmeans.predict(train_df[['pickup_longitude','pickup_latitude']])
pickup_clusters_test = kmeans.predict(test_df[['pickup_longitude','pickup_latitude']])


# Load/Fit KNN DropOff
kmeans = KMeans(n_clusters=5, random_state=42).fit(train_df[['dropoff_longitude','dropoff_latitude']])


# Cluster DropOff
dropoff_clusters = kmeans.predict(train_df[['dropoff_longitude','dropoff_latitude']])
dropoff_clusters_test = kmeans.predict(test_df[['dropoff_longitude','dropoff_latitude']])


# Add Clusters to Dataset
train_df['pickup_clusters'] = pickup_clusters
test_df['pickup_clusters'] = pickup_clusters_test

train_df['dropoff_clusters'] = dropoff_clusters
test_df['dropoff_clusters'] = dropoff_clusters_test


# Dataset Backup
train_backup = train_df.copy()
test_backup = test_df.copy()


# Encode Dataset
pickup_clusters_encoded = pd.get_dummies(train_df['pickup_clusters'], prefix='pickup_cluster')
dropoff_clusters_encoded = pd.get_dummies(train_df['dropoff_clusters'], prefix='dropoff_cluster')
store_and_fwd_flag_encoded = pd.get_dummies(train_df['store_and_fwd_flag'], prefix='store_and_fwd_flag')
passenger_count_encoded = pd.get_dummies(train_df['passenger_count'], prefix='passenger_count')
vendor_id_encoded = pd.get_dummies(train_df['vendor_id'], prefix='vendor_id')

test_pickup_clusters_encoded = pd.get_dummies(test_df['pickup_clusters'], prefix='pickup_cluster')
test_dropoff_clusters_encoded = pd.get_dummies(test_df['dropoff_clusters'], prefix='dropoff_cluster')
test_store_and_fwd_flag_encoded = pd.get_dummies(test_df['store_and_fwd_flag'], prefix='store_and_fwd_flag')
test_passenger_count_encoded = pd.get_dummies(test_df['passenger_count'], prefix='passenger_count')
test_vendor_id_encoded = pd.get_dummies(test_df['vendor_id'], prefix='vendor_id')


# Drop Unwanted Columns
train_df.drop('pickup_clusters', axis = 1, inplace = True)
train_df.drop('dropoff_clusters', axis = 1, inplace = True)
train_df.drop('store_and_fwd_flag', axis = 1, inplace = True)
train_df.drop('passenger_count', axis = 1, inplace = True)
train_df.drop('vendor_id', axis = 1, inplace = True)

test_df.drop('pickup_clusters', axis = 1, inplace = True)
test_df.drop('dropoff_clusters', axis = 1, inplace = True)
test_df.drop('store_and_fwd_flag', axis = 1, inplace = True)
test_df.drop('passenger_count', axis = 1, inplace = True)
test_df.drop('vendor_id', axis = 1, inplace = True)


# Join Dataset
train_df = train_df.join(pickup_clusters_encoded)
train_df = train_df.join(dropoff_clusters_encoded)
train_df = train_df.join(store_and_fwd_flag_encoded)
train_df = train_df.join(passenger_count_encoded)
train_df = train_df.join(vendor_id_encoded)

test_df = test_df.join(test_pickup_clusters_encoded)
test_df = test_df.join(test_dropoff_clusters_encoded)
test_df = test_df.join(test_store_and_fwd_flag_encoded)
test_df = test_df.join(test_passenger_count_encoded)
test_df = test_df.join(test_vendor_id_encoded)


# Check Columns Match
train_cols = train_df.columns
test_cols = test_df.columns
print([x for x in train_cols if x not in test_cols])


# Add UnMatch Columns
test_df['dropoff_cluster_4'] = 0
test_df['passenger_count_7'] = 0
test_df['passenger_count_8'] = 0


# Set X/y
X = train_df.drop(['id', 'pickup_datetime', 'dropoff_datetime', 
              'pickup_longitude', 'pickup_latitude', 
              'dropoff_longitude', 'dropoff_latitude', 
              'date', 'trip_duration', 'Anomaly'], axis = 1).copy()

y = train_df['trip_duration']


# Load mMdel
reg = xgb.XGBRegressor()


# Model Fit
reg.fit(X.values, y.values)


# Load Test
X_test = test_df.drop(['id', 'pickup_datetime', 'dropoff_datetime', 
              'pickup_longitude', 'pickup_latitude', 
              'dropoff_longitude', 'dropoff_latitude', 
              'date'], axis = 1).copy()


# Adjust X_test
X_test = X_test[['distance', 'pickup_cluster_0', 'pickup_cluster_1', 'pickup_cluster_2',
               'pickup_cluster_3', 'pickup_cluster_4', 'dropoff_cluster_0',
               'dropoff_cluster_1', 'dropoff_cluster_2', 'dropoff_cluster_3',
               'dropoff_cluster_4', 'store_and_fwd_flag_N', 'store_and_fwd_flag_Y',
               'passenger_count_0', 'passenger_count_1', 'passenger_count_2',
               'passenger_count_3', 'passenger_count_4', 'passenger_count_5',
               'passenger_count_6', 'passenger_count_7', 'passenger_count_8',
               'passenger_count_9', 'vendor_id_1', 'vendor_id_2']]


# Model Predicitions
pred = reg.predict(X_test.values)


# Submit Predictions
submission = test_df['id']
submission = {"id":test_df["id"],"trip_duration":pred}
submission = pd.DataFrame(submission)


# Save Submission
submission.to_csv("submission.csv",index=False)

