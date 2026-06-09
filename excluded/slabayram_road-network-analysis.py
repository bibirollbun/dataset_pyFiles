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


import pandas as pd
import numpy as np
import datetime
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. DATA LOADING 

possible_paths = [
    '/kaggle/input/nyc-taxi-trip-duration/train.csv',       # Standard Kaggle Path
    '/kaggle/input/nyc-taxi-trip-duration/train.zip',       # Zipped Kaggle Path
    'train.csv'                                             # Local Path
]

df = None

for path in possible_paths:
    if os.path.exists(path):
        print(f"File found at: {path}")
        try:
            if path.endswith('.zip'):
                with zipfile.ZipFile(path, 'r') as z:
                    with z.open('train.csv') as f:
                      
                        df = pd.read_csv(f).sample(frac=0.5, random_state=42)
            else:
                df = pd.read_csv(path).sample(frac=0.5, random_state=42)
            break
        except Exception as e:
            print(f"Error loading {path}: {e}")

print(f"Dataset loaded successfully. Total Rows: {len(df)}")



df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  # Monday=0, Sunday=6
df['month'] = df['pickup_datetime'].dt.month

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

df['distance_km'] = haversine_distance(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

#2.Data Cleaning

df = df[(df['trip_duration'] < 10800) & (df['trip_duration'] > 60)]
df = df[(df['distance_km'] > 0.1) & (df['distance_km'] < 100)]

print(f"Data after cleaning: {len(df)} rows")



# 3. Model Training
features = ['pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude',
            'distance_km', 'hour', 'day_of_week']

X = df[features]
y = df['trip_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Random Forest Model")

rf_model = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=42, n_jobs=-1)

rf_model.fit(X_train, y_train)


# 4. Evaluation
print("\nEvaluating Model Performance...")
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Success Metrics:")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} seconds (~{rmse/60:.1f} minutes)")
print(f"R2 Score: {r2:.3f}")

# 5. Saving the Model
model_filename = 'nyc_taxi_model.pkl'
joblib.dump(rf_model, model_filename)


