import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print("=== AVAILABLE DATASETS ===")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Load Dataset
import numpy as np
import pandas as pd
# Use the exact path shown in Step 1 output
df = pd.read_csv('/kaggle/input/heat-code-by-fodse/chennai_weather_ml_competition_dataset.csv')
print(df.info())
print(df.head())
print(df.tail())


# Basic info
print("Missing values:")
print(df.isnull().sum())
print(f"\nDuplicates: {df.duplicated().sum()}")
df.describe()


# Detect outliers
numerical_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
for col in numerical_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    print(f"{col}: {outliers} outliers")


# Clean data
df_clean = df.copy()
# Handle outliers
for col in ['humidity', 'pressure', 'wind_speed']:
    df_clean[col] = df_clean[col].clip(df_clean[col].quantile(0.01), df_clean[col].quantile(0.99))


# Split data
split_idx = int(len(df_clean) * 0.8)
train_data = df_clean.iloc[:split_idx]
test_data = df_clean.iloc[split_idx:]
print(f"Train: {len(train_data)}, Test: {len(test_data)}")


print("=== FEATURE ENGINEERING ===")

# Create working copy
df_features = df_clean.copy()

# Convert date/time to datetime
df_features['datetime'] = pd.to_datetime(df_features['date'] + ' ' + df_features['time'])

# Extract time features
df_features['hour'] = df_features['datetime'].dt.hour
df_features['day_of_week'] = df_features['datetime'].dt.dayofweek
df_features['month'] = df_features['datetime'].dt.month

# Cyclical encoding (important for ML models)
df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)

# Weather interaction features
if 'humidity' in df_features.columns:
    df_features['heat_index'] = df_features['temperature'] + 0.5 * (df_features['humidity'] - 10)
if 'wind_speed' in df_features.columns:
    df_features['wind_chill'] = df_features['temperature'] - (df_features['wind_speed'] * 0.7)

# Temperature change features
if 'temp_lag_1h' in df_features.columns:
    df_features['temp_change_1h'] = df_features['temperature'] - df_features['temp_lag_1h']
if 'temp_lag_24h' in df_features.columns:
    df_features['temp_change_24h'] = df_features['temperature'] - df_features['temp_lag_24h']

print(f"Features created! New shape: {df_features.shape}")
print(f"New features: {[col for col in df_features.columns if col not in df_clean.columns]}")


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

print("=== MODEL TRAINING ===")

# Prepare features and target
target = 'temperature'
exclude_cols = ['date', 'time', 'datetime', target]
feature_cols = [col for col in df_features.columns if col not in exclude_cols]

X = df_features[feature_cols].fillna(df_features[feature_cols].median())
y = df_features[target]

# Train-test split (80/20)
split_idx = int(len(df_features) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training features: {len(feature_cols)}")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Test multiple models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
print("\nModel Performance:")
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'model': model, 'rmse': rmse, 'r2': r2}
    print(f"{name}: RMSE={rmse:.3f}, R²={r2:.3f}")

# Select best model
best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")


print("=== GENERATING COMPETITION SUBMISSION ===")

# Competition requires predictions for 10 AM to 9 PM (12 time slots)
prediction_hours = list(range(10, 22))  # 10, 11, 12, ..., 21

# Create prediction data for each hour
prediction_data = []
for hour in prediction_hours:
    # Use median values for most features, set specific hour
    pred_row = X.median()
    
    # Set time-specific features
    pred_row['hour'] = hour
    pred_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    pred_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Use current month (you can adjust based on prediction date)
    current_month = df_features['month'].iloc[-1]
    pred_row['month'] = current_month
    pred_row['month_sin'] = np.sin(2 * np.pi * current_month / 12)
    pred_row['month_cos'] = np.cos(2 * np.pi * current_month / 12)
    
    prediction_data.append(pred_row)

# Convert to DataFrame and make predictions
X_submit = pd.DataFrame(prediction_data)
X_submit = X_submit[feature_cols]  # Ensure same feature order

# Generate predictions
competition_predictions = best_model.predict(X_submit)

# Create submission file in required format
submission_df = pd.DataFrame({
    'ID': list(range(1, 13)),  # IDs 1-12 for 10 AM to 9 PM
    'temperature_prediction': competition_predictions
})

# Display predictions
print("Competition Predictions:")
print("ID → Time → Temperature")
for i, (id_val, temp) in enumerate(zip(submission_df['ID'], submission_df['temperature_prediction'])):
    hour = prediction_hours[i]
    print(f"{id_val:2d} → {hour:02d}:00 → {temp:.1f}°C")

# Save submission file
submission_df.to_csv('submission.csv', index=False)
print(f"\n✅ submission.csv created!")
print(f"✅ {len(submission_df)} predictions generated")
print("\nSubmission file format:")
print(submission_df)

