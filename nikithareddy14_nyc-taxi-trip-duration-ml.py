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


# ================================================================
# ğŸ—½ NYC Taxi Trip Duration - Complete Kaggle Notebook
# Author: [Your Name]
# Description: Unzip â†’ EDA â†’ Feature Engineering â†’ XGBoost Model â†’ Submission
# ================================================================

# ========================
# 1ï¸�âƒ£ Import Libraries
# ========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import zipfile
import os
import warnings

# Ignore FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# ========================
# 2ï¸�âƒ£ Unzip Dataset Files
# ========================
input_path = "/kaggle/input/nyc-taxi-trip-duration"

with zipfile.ZipFile(f"{input_path}/train.zip", "r") as zip_ref:
    zip_ref.extractall("/kaggle/working")

with zipfile.ZipFile(f"{input_path}/test.zip", "r") as zip_ref:
    zip_ref.extractall("/kaggle/working")

with zipfile.ZipFile(f"{input_path}/sample_submission.zip", "r") as zip_ref:
    zip_ref.extractall("/kaggle/working")

# ========================
# 3ï¸�âƒ£ Load Data
# ========================
train = pd.read_csv("/kaggle/working/train.csv")
test = pd.read_csv("/kaggle/working/test.csv")
sample_sub = pd.read_csv("/kaggle/working/sample_submission.csv")

print("âœ… Data Loaded Successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()

# ========================
# 4ï¸�âƒ£ Data Cleaning & Feature Engineering
# ========================
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])

# Extract datetime features
for df in [train, test]:
    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["pickup_datetime"].dt.dayofweek

# Calculate distance using Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

train["distance_km"] = haversine(train["pickup_longitude"], train["pickup_latitude"],
                                 train["dropoff_longitude"], train["dropoff_latitude"])
test["distance_km"] = haversine(test["pickup_longitude"], test["pickup_latitude"],
                                test["dropoff_longitude"], test["dropoff_latitude"])

# Remove invalid or extreme values
train = train[(train["trip_duration"] < 50000) & (train["distance_km"] > 0)]
train["log_trip_duration"] = np.log1p(train["trip_duration"])

# Replace infinite or NaN values
train = train.replace([np.inf, -np.inf], np.nan)
train = train.dropna()

# ========================
# 5ï¸�âƒ£ Exploratory Data Analysis (Visualizations)
# ========================
plt.figure(figsize=(8, 4))
sns.histplot(train["trip_duration"], bins=100, color="steelblue")
plt.title("Distribution of Trip Duration")
plt.xlabel("Trip Duration (seconds)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="distance_km", y="trip_duration", data=train, alpha=0.3)
plt.title("Trip Duration vs Distance (in km)")
plt.xlabel("Distance (km)")
plt.ylabel("Trip Duration (seconds)")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x="pickup_hour", y="trip_duration", data=train)
plt.title("Trip Duration by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Trip Duration (seconds)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(train[["trip_duration", "pickup_hour", "pickup_day_of_week", "distance_km"]].corr(), 
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ========================
# 6ï¸�âƒ£ Prepare Data for Model
# ========================
features = ["vendor_id", "pickup_hour", "pickup_day_of_week",
            "pickup_month", "pickup_day", "distance_km"]

X = train[features]
y = train["log_trip_duration"]
X_test = test[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================
# 7ï¸�âƒ£ Train Model (XGBoost)
# ========================
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred)))
print("âœ… Validation RMSLE:", round(rmsle, 4))

# ========================
# 8ï¸�âƒ£ Generate Submission
# ========================
test_pred = model.predict(X_test)
submission = pd.DataFrame({
    "id": test["id"],
    "trip_duration": np.expm1(test_pred)
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… submission.csv created successfully in /kaggle/working")

# ========================
# 9ï¸�âƒ£ Feature Importance Visualization
# ========================
plt.figure(figsize=(8, 5))
sns.barplot(x=model.feature_importances_, y=features, palette="viridis")
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

print("âœ… Notebook completed successfully! Ready to submit.")


