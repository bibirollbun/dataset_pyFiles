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


#Load and Prepare Data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test_df = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

# Quick check
print(f"Training shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print("\nTraining head:")
print(train_df.head(2))


#Feature Engineering
def create_features(df):
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Basic time features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.dayofweek
    
    # Advanced features
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 16) & (df['hour'] <= 18))
    df['is_weekend'] = df['weekday'] >= 5
    df['seasonal_temp'] = df['temp'] * (df['month'] % 12 / 3)  # Interaction feature
    
    # Binned features
    df['hour_type'] = pd.cut(df['hour'], 
                            bins=[0, 6, 10, 15, 19, 24],
                            labels=['night', 'morning', 'midday', 'evening', 'late_night'])
    
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

print("Created features:")
print(train_df[['datetime', 'hour', 'is_rush_hour', 'hour_type']].head(3))


#Prepare Training Data
# Selected features
features = [
    'hour', 'weekday', 'month', 'year',
    'is_rush_hour', 'is_weekend', 'seasonal_temp',
    'temp', 'atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 'weather'
]

X = train_df[features]
y = train_df['count']

# Log-transform target (helps with RMSLE)
y_log = np.log1p(y)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, 
    test_size=0.2, 
    random_state=42
)

print(f"\nTraining on {X_train.shape[0]} samples")
print(f"Validating on {X_val.shape[0]} samples")


#Train Optimized Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    max_samples=0.8,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)
print("Training complete!")


#Evaluate Model
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Validation predictions
val_preds = model.predict(X_val)

# Convert back from log scale
val_preds_exp = np.expm1(val_preds)
y_val_exp = np.expm1(y_val)

# Calculate RMSLE
score = rmsle(y_val_exp, val_preds_exp)
print(f"Validation RMSLE: {score:.4f}")

# Feature importance
importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop features:")
print(importances.head(10))


#Generate Test Predictions
# Prepare test features
X_test = test_df[features]

# Predict (remember to use log scale)
test_preds_log = model.predict(X_test)
test_preds = np.expm1(test_preds_log)  # Convert back

# Ensure no negative predictions
test_preds = np.clip(test_preds, 0, None)

# Create submission
submission = pd.DataFrame({
    'datetime': test_df['datetime'],
    'count': test_preds
})

submission.to_csv("submission_improved.csv", index=False)
print("Submission saved!")


#Visual Analysis
plt.figure(figsize=(12, 6))

# Actual vs Predicted
plt.scatter(y_val_exp, val_preds_exp, alpha=0.3)
plt.plot([0, max(y_val_exp)], [0, max(y_val_exp)], '--r')
plt.xlabel('Actual Count')
plt.ylabel('Predicted Count')
plt.title('Actual vs Predicted Bike Counts')
plt.grid(alpha=0.2)
plt.show()

# Hourly pattern
hourly_avg = train_df.groupby('hour')['count'].mean()
plt.figure(figsize=(12, 4))
hourly_avg.plot(kind='bar')
plt.title('Average Rentals by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Average Count')
plt.show()

