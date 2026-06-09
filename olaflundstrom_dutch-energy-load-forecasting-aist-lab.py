import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading data...")
train = pd.read_csv('/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv')
test = pd.read_csv('/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv')
sample_sub = pd.read_csv('/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/sample_submission_new.csv')

# Parse timestamps
train['timestamp_utc'] = pd.to_datetime(train['timestamp_utc'])
test['timestamp_utc'] = pd.to_datetime(test['timestamp_utc'])

train = train.sort_values('timestamp_utc').reset_index(drop=True)
test = test.sort_values('timestamp_utc').reset_index(drop=True)

# ============================================================================
# 2. FETCH WEATHER DATA FROM OPEN-METEO
# ============================================================================
print("Fetching weather data...")

import requests
from datetime import datetime

# Netherlands coordinates (approximate center)
lat, lon = 52.1326, 5.2913

# Date ranges
train_start = train['timestamp_utc'].min().strftime('%Y-%m-%d')
train_end = train['timestamp_utc'].max().strftime('%Y-%m-%d')
test_start = test['timestamp_utc'].min().strftime('%Y-%m-%d')
test_end = test['timestamp_utc'].max().strftime('%Y-%m-%d')

# Fetch hourly weather data
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": train_start,
    "end_date": test_end,
    "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,rain,snowfall,cloud_cover,wind_speed_10m,wind_direction_10m,wind_speed_100m,shortwave_radiation,direct_radiation,diffuse_radiation",
    "timezone": "UTC"
}

response = requests.get(url, params=params)
weather_data = response.json()

# Convert to DataFrame
weather_df = pd.DataFrame({
    'timestamp_utc': pd.to_datetime(weather_data['hourly']['time']),
    'temperature': weather_data['hourly']['temperature_2m'],
    'humidity': weather_data['hourly']['relative_humidity_2m'],
    'dew_point': weather_data['hourly']['dew_point_2m'],
    'precipitation': weather_data['hourly']['precipitation'],
    'rain': weather_data['hourly']['rain'],
    'snowfall': weather_data['hourly']['snowfall'],
    'cloud_cover': weather_data['hourly']['cloud_cover'],
    'wind_speed_10m': weather_data['hourly']['wind_speed_10m'],
    'wind_direction': weather_data['hourly']['wind_direction_10m'],
    'wind_speed_100m': weather_data['hourly']['wind_speed_100m'],
    'shortwave_radiation': weather_data['hourly']['shortwave_radiation'],
    'direct_radiation': weather_data['hourly']['direct_radiation'],
    'diffuse_radiation': weather_data['hourly']['diffuse_radiation']
})

# Resample to 15-minute intervals using forward fill
weather_df = weather_df.set_index('timestamp_utc')
weather_df = weather_df.resample('15min').ffill().reset_index()

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("Engineering features...")

def create_features(df, weather_df, is_test=False):
    # Merge weather
    df = df.merge(weather_df, on='timestamp_utc', how='left')
    
    # Time features
    df['hour'] = df['timestamp_utc'].dt.hour
    df['dayofweek'] = df['timestamp_utc'].dt.dayofweek
    df['quarter'] = df['timestamp_utc'].dt.quarter
    df['month'] = df['timestamp_utc'].dt.month
    df['day'] = df['timestamp_utc'].dt.day
    df['dayofyear'] = df['timestamp_utc'].dt.dayofyear
    df['weekofyear'] = df['timestamp_utc'].dt.isocalendar().week
    df['quarter_hour'] = (df['timestamp_utc'].dt.hour * 4 + df['timestamp_utc'].dt.minute // 15)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['quarter_hour_sin'] = np.sin(2 * np.pi * df['quarter_hour'] / 96)
    df['quarter_hour_cos'] = np.cos(2 * np.pi * df['quarter_hour'] / 96)
    
    # Weekend/weekday
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_workday'] = ((df['dayofweek'] < 5) & (df['hour'] >= 8) & (df['hour'] < 18)).astype(int)
    
    # Weather interactions
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['wind_temp'] = df['wind_speed_10m'] * df['temperature']
    df['radiation_cloud'] = df['shortwave_radiation'] * (100 - df['cloud_cover'])
    df['feels_like'] = df['temperature'] - (df['wind_speed_10m'] * 0.7)
    
    # Lagged weather features (respecting 3-day latency = 288 steps)
    if not is_test:
        lag_steps = [288, 288 + 96, 288 + 192]  # 3 days, 3.5 days, 4 days
        for col in ['temperature', 'humidity', 'wind_speed_10m', 'shortwave_radiation']:
            for lag in lag_steps:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Rolling weather stats (must respect latency)
    if not is_test:
        for col in ['temperature', 'humidity', 'wind_speed_10m']:
            df[f'{col}_roll24h'] = df[col].shift(288).rolling(96).mean()
            df[f'{col}_roll48h'] = df[col].shift(288).rolling(192).mean()
    
    return df

# Create features
train_fe = create_features(train.copy(), weather_df, is_test=False)
test_fe = create_features(test.copy(), weather_df, is_test=True)

# For test, we need to add lagged features from train+test combined
# Combine train and test for proper lagging
combined = pd.concat([train_fe[['timestamp_utc'] + [c for c in train_fe.columns if c in ['temperature', 'humidity', 'wind_speed_10m', 'shortwave_radiation']]], 
                      test_fe[['timestamp_utc'] + [c for c in test_fe.columns if c in ['temperature', 'humidity', 'wind_speed_10m', 'shortwave_radiation']]]], 
                     ignore_index=True)

for col in ['temperature', 'humidity', 'wind_speed_10m', 'shortwave_radiation']:
    for lag in [288, 288 + 96, 288 + 192]:
        combined[f'{col}_lag{lag}'] = combined[col].shift(lag)

for col in ['temperature', 'humidity', 'wind_speed_10m']:
    combined[f'{col}_roll24h'] = combined[col].shift(288).rolling(96).mean()
    combined[f'{col}_roll48h'] = combined[col].shift(288).rolling(192).mean()

# Extract test portion
test_start_idx = len(train_fe)
test_lag_features = combined.iloc[test_start_idx:test_start_idx + len(test_fe)]

for col in ['temperature', 'humidity', 'wind_speed_10m', 'shortwave_radiation']:
    for lag in [288, 288 + 96, 288 + 192]:
        test_fe[f'{col}_lag{lag}'] = test_lag_features[f'{col}_lag{lag}'].values

for col in ['temperature', 'humidity', 'wind_speed_10m']:
    test_fe[f'{col}_roll24h'] = test_lag_features[f'{col}_roll24h'].values
    test_fe[f'{col}_roll48h'] = test_lag_features[f'{col}_roll48h'].values

# ============================================================================
# 4. PREPARE TRAINING DATA
# ============================================================================
print("Preparing training data...")

# Drop rows with NaN (due to lagging)
train_clean = train_fe.dropna()

# Features to use
feature_cols = [col for col in train_clean.columns if col not in ['timestamp_utc', 'net_load_kwh']]
X_train = train_clean[feature_cols]
y_train = train_clean['net_load_kwh']

X_test = test_fe[feature_cols].fillna(method='ffill').fillna(0)

# ============================================================================
# 5. TRAIN LIGHTGBM MODEL
# ============================================================================
print("Training LightGBM model...")

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1
}

# Train with early stopping
tss = TimeSeriesSplit(n_splits=3)
models = []

for fold, (train_idx, val_idx) in enumerate(tss.split(X_train)):
    print(f"Fold {fold + 1}/3")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    models.append(model)
    
    # Validate
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    mae = np.mean(np.abs(y_val - val_pred))
    nrmse = rmse / (y_val.max() - y_val.min()) * 100
    nmae = mae / (y_val.max() - y_val.min()) * 100
    
    print(f"Fold {fold + 1} - NRMSE: {nrmse:.2f}%, NMAE: {nmae:.2f}%")

# ============================================================================
# 6. GENERATE PREDICTIONS
# ============================================================================
print("Generating predictions...")

# Ensemble predictions
predictions = np.mean([model.predict(X_test, num_iteration=model.best_iteration) for model in models], axis=0)

# Create submission
submission = sample_sub.copy()
submission['net_load_kwh'] = predictions

submission.to_csv('submission.csv', index=False)
print("Submission file created!")

# Calculate metrics on a held-out validation set from train
val_size = int(len(train_clean) * 0.2)
X_val_final = X_train.iloc[-val_size:]
y_val_final = y_train.iloc[-val_size:]

val_pred = np.mean([model.predict(X_val_final, num_iteration=model.best_iteration) for model in models], axis=0)
rmse = np.sqrt(np.mean((y_val_final - val_pred) ** 2))
mae = np.mean(np.abs(y_val_final - val_pred))
nrmse = rmse / (y_val_final.max() - y_val_final.min()) * 100
nmae = mae / (y_val_final.max() - y_val_final.min()) * 100

print(f"\nFinal Validation Metrics:")
print(f"NRMSE: {nrmse:.2f}%")
print(f"NMAE: {nmae:.2f}%")
print(f"\nTarget: NRMSE < 5% and NMAE < 5%")

