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


import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (12, 8)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


train_df = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/train.zip")
train_df.head()


test_df = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/test.zip")
test_df.head()


submission_df = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/sample_submission.zip")
submission_df.head()


train_df.shape, test_df.shape


train_df = train_df.copy()
test_df = test_df.copy()


from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate the great circle distance between two points on Earth"""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km # Earth radius in kilometers


def calculate_distance(df):
    df['distance_to_be_covered']= haversine_distance(df['pickup_latitude'],df['pickup_longitude'],df['dropoff_latitude'],df['dropoff_longitude'])
    return df

train_df = calculate_distance(train_df)
test_df = calculate_distance(test_df)


def create_date_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df['IsWeekend'] = df['pickup_day_of_week'].isin([5, 6]).astype(int)
    
    # Seasonal features
    df['pickup_season'] = (df['pickup_month'] % 12 + 3) // 3

    # Part of the day (categorical feature)
    def get_part_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df["pickup_part_of_day"] = df["pickup_hour"].apply(get_part_of_day)

    return df

train_df = create_date_features(train_df)
test_df = create_date_features(test_df)


def directional_movement(df):
    # Direction features
    df['direction_ns'] = df['dropoff_latitude'] - df['pickup_latitude']
    df['direction_ew'] = df['dropoff_longitude'] - df['pickup_longitude']

    return df

train_df = directional_movement(train_df)
test_df = directional_movement(test_df)


import numpy as np

def add_bearing_feature(df):
    """
    Compute the bearing (direction of travel) between pickup and dropoff points.

    Bearing tells you the direction from the pickup point to the dropoff point, measured clockwise from North:

    0° → due North
    
    90° → due East
    
    180° → due South
    
    270° → due West
    """
    # Convert lat/long from degrees to radians
    pickup_lat = np.radians(df['pickup_latitude'])
    pickup_lng = np.radians(df['pickup_longitude'])
    dropoff_lat = np.radians(df['dropoff_latitude'])
    dropoff_lng = np.radians(df['dropoff_longitude'])
    
    # Compute the difference in longitude
    d_lng = dropoff_lng - pickup_lng

    # Compute the bearing using trigonometry
    x = np.sin(d_lng) * np.cos(dropoff_lat)
    y = np.cos(pickup_lat) * np.sin(dropoff_lat) - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(d_lng)
    bearing = np.degrees(np.arctan2(x, y))

    # Normalize to 0–360 degrees
    df['bearing'] = round((bearing + 360) % 360,ndigits=2 )

    return df

train_df = add_bearing_feature(train_df)
test_df = add_bearing_feature(test_df)


train_df["log_trip_duration"] = np.log1p(train_df["trip_duration"])


train_df.hist(bins = 25,figsize=(30,20))


numeric_df = train_df.select_dtypes(include=['int64', 'float64'])
numeric_df.corr()['log_trip_duration'].sort_values(ascending=False)


from pandas.plotting import scatter_matrix


sample = ['distance_to_be_covered', 'pickup_longitude','bearing', 'log_trip_duration']
scatter_matrix(train_df[sample][:100000], figsize=(20,10))


corr_matrix = train_df.corr(numeric_only=True)

plt.figure(figsize=(30,20))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
)


from sklearn.model_selection import train_test_split


train_df,val_df = train_test_split(train_df, test_size=0.2, random_state=42)


train_df.shape, val_df.shape, test_df.shape


train_df.columns


input_cols = [
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
        'distance_to_be_covered', 'pickup_year','pickup_season',
       'pickup_month', 'pickup_day', 'pickup_hour', 'pickup_day_of_week',
       'IsWeekend', 'pickup_part_of_day', 'direction_ns', 'direction_ew',
       'bearing']

input_cols


target_col = 'log_trip_duration'
target_col


# Training dataset inputs and target cols
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

# Validation dataset inputs and target
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

# Testing dataset inputs
test_inputs = test_df[input_cols].copy()


numeric_cols = list(var for var in train_inputs.columns if train_inputs[var].dtype != "O")

numeric_cols


categorical_cols = list(var for var in train_inputs.columns if train_inputs[var].dtype == 'O')

categorical_cols


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))


train_inputs.loc[:, encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs.loc[:, encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs.loc[:, encoded_cols] = encoder.transform(test_inputs[categorical_cols])


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


scaler.fit(train_inputs[numeric_cols])


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


X_train.head()


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import XGBRegressor


%%time
model_0 = XGBRegressor(
    objective='reg:squarederror',
    n_jobs=-1,
    n_estimators=6000,              # Reduce from 6000
    max_depth=12,                    # Shallower
    random_state=42,
    learning_rate=0.1,             # Keep same
    min_child_weight=5,            # More conservative
    subsample=0.9,                  # Keep same
    colsample_bytree=0.9,           # More conservative
    early_stopping_rounds=50,      # More patience
    reg_alpha=7.0,                  # More regularization
    reg_lambda=15.0,                # More regularization
)

model_0.fit(X_train,train_targets,eval_set=[(X_val, val_targets)], verbose=False)


val_pred = model_0.predict(X_val)
val_pred[:5]


from sklearn.metrics import mean_squared_log_error


rmsle = np.sqrt(mean_squared_log_error(np.expm1(val_targets), np.expm1(val_pred)))
print("✅ Validation RMSLE:", round(rmsle, 4))


importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_0.feature_importances_
}).sort_values('importance', ascending=False)

plt.title("Feature Importance")
sns.barplot(data=importance_df.head(10),x='importance', y='feature',saturation=0.75)


test_preds = model_0.predict(X_test)
test_preds


submission_df['trip_duration'] = np.expm1(test_preds)

# Verify the update
print("Updated submission preview:")
print(submission_df.head())
print(f"\nSubmission shape: {submission_df.shape}")

# Save the updated submission
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




