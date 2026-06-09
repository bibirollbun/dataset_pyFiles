#Basic Setup
#Core
import pandas as pd
import numpy as np
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#Settings
import warnings
warnings.filterwarnings('ignore')
#Aesthetics
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', 50)
#Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Basic Setup Complete")


#Loading a manageable subset for exploration
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
N_ROWS = 100000
df = pd.read_csv(TRAIN_PATH, nrows=N_ROWS)
print(df.shape)
df.head()


# Loose NYC bounding box constants
#We will use these in cleaning to cut obvious geo outliers
NYC_LAT_MIN, NYC_LAT_MAX = 40.3, 41.2
NYC_LON_MIN, NYC_LON_MAX = -74.5, -72.8

print("NYC bounds ready.")


# View structure and data types
df.info()


#Quick peek at data
df.head(3)


#Checking missing values
missing = df.isna().mean().sort_values(ascending=False)
missing


#Descriptive statistics
df.describe(include='all')



#Check column data types
df.dtypes


#Convert all columns to float/int
num_cols = [
    'fare_amount',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count'
]

# Convert to numeric, turn bad values into NaN
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')



df[num_cols].dtypes


#Handle missing and invalid data
critical_cols = [
    'fare_amount',
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'passenger_count', 'pickup_datetime'
]
df = df.dropna(subset=critical_cols).copy()

#Remove absurd fare values
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 200)]

#Keep reasonable passenger counts
df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <=6)]




#Filter unrealistic coordinates
df = df[
    df['pickup_latitude'].between(40.3, 41.2) &
    df['dropoff_latitude'].between(40.3, 41.2) &
    df['pickup_longitude'].between(-74.5, -72.8) &
    df['dropoff_longitude'].between(-74.5, -72.8)
].copy()


#Compute trip distance using Haversine Formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 #earth radius(km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

#Apply distance function
df['distance_km'] = haversine(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

#Filter impossible or huge distances
df = df[(df['distance_km'] > 0.05) & (df['distance_km'] < 100)].copy()

    



#Extract datetime features
#Convert pickup_datetime to pandas datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors = 'coerce')
df = df.dropna(subset=['pickup_datetime']).copy()

#Extract key temporal features
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['year'] = df['pickup_datetime'].dt.year
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

df[['fare_amount', 'distance_km', 'passenger_count', 'hour', 'day', 'month', 'year']].describe().T




#Independent features
features = ['distance_km', 'passenger_count', 'hour', 'day', 'month', 'year']

#Target variable
target = 'fare_amount'

X = df[features]
y = df[target]

print(X.shape, y.shape)


#Split into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.2, random_state = RANDOM_STATE
)

print("Train size: ", X_train.shape)
print("Test size: ", X_test.shape)


#Apply Standard Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#Fit only training data, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print("Feature Scaling complete.")


#Train the model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)


#Predict on test set

y_pred_lr = lr.predict(X_test_scaled)


#Evaluate performance

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 

mae = mean_absolute_error(y_test, y_pred_lr)
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_lr)

print("Linear Regression Performance: ")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R squared score: {r2:.3f}")


#Visualize Predictions
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lr, alpha=0.3, color='royalblue')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Linear Regression: Actual vs Predicted Fares')
plt.show()


#Train the model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X_train, y_train)



#Predict and evaluate
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Performance: ")
print(f"Mean Absolute Error: {mae_rf:.2f}")
print(f"Root Mean Squared Error: {rmse_rf:.2f}")
print(f"R squared score: {r2_rf:.3f}")


#Visualize performance
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, alpha=0.3, color='forestgreen')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Random Forest: Actual vs. Predicted Fares')
plt.show()


#Feature importance
imp = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=imp, palette='Greens_r')
plt.title('Random Forest Feature Importance')
plt.show()


#Load test data

test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_df.head()


#Preprocess test data (same as training)

#Compute trip distance using Haversine Formula

#Apply distance function
test_df['distance_km'] = haversine(
    test_df['pickup_latitude'], test_df['pickup_longitude'],
    test_df['dropoff_latitude'], test_df['dropoff_longitude']
)

# Extract datetime features
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'], errors='coerce')
test_df['hour'] = test_df['pickup_datetime'].dt.hour
test_df['day'] = test_df['pickup_datetime'].dt.day
test_df['month'] = test_df['pickup_datetime'].dt.month
test_df['year'] = test_df['pickup_datetime'].dt.year


#Select features
X_final = test_df[['distance_km', 'passenger_count', 'hour', 'day', 'month', 'year']]


#Make predictions
test_df['predicted_fare'] = rf.predict(X_final)
test_df[['key', 'predicted_fare']].head()


#Save prediction as csv
submission = pd.DataFrame({
    'key': test_df['key'],
    'fare_amount': test_df['predicted_fare']
})

submission.to_csv("submission.csv", index=False)
print("submission.csv file created successfully!")


