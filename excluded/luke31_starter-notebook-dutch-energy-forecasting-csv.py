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


"""
Dutch Energy Supplier Load Forecasting Challenge
Competition-Compliant Solution
=================================================
Predicts 48 hours ahead using weather data and respecting 72-hour data latency
Target: NRMSE < 5% and NMAE < 5%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb

print("=" * 80)
print("DUTCH ENERGY SUPPLIER LOAD FORECASTING CHALLENGE")
print("Competition-Compliant Solution")
print("=" * 80)

# ==========================================
# 1. LOAD DATA
# ==========================================
print("\n1. LOADING DATA")
print("-" * 40)

# File paths
train_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv'
test_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv'

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"✓ Train data loaded: {train_df.shape}")
print(f"✓ Test data loaded: {test_df.shape}")
print(f"  Train columns: {train_df.columns.tolist()}")
print(f"  Test columns: {test_df.columns.tolist()}")

# Standardize column names
if 'Datetime' in train_df.columns:
    train_df = train_df.rename(columns={'Datetime': 'timestamp_utc', 'Actual Net': 'net_load_kwh'})
if 'Datetime' in test_df.columns:
    test_df = test_df.rename(columns={'Datetime': 'timestamp_utc'})
    if 'Actual Net' in test_df.columns:
        test_df = test_df.rename(columns={'Actual Net': 'net_load_kwh'})

# Convert to datetime
train_df['timestamp_utc'] = pd.to_datetime(train_df['timestamp_utc'])
test_df['timestamp_utc'] = pd.to_datetime(test_df['timestamp_utc'])

# Sort by timestamp
train_df = train_df.sort_values('timestamp_utc').reset_index(drop=True)
test_df = test_df.sort_values('timestamp_utc').reset_index(drop=True)

# Add row_id to test if not present
if 'row_id' not in test_df.columns:
    test_df['row_id'] = range(len(test_df))

print(f"\nData period:")
print(f"  Train: {train_df['timestamp_utc'].min()} to {train_df['timestamp_utc'].max()}")
print(f"  Test: {test_df['timestamp_utc'].min()} to {test_df['timestamp_utc'].max()}")
print(f"  Frequency: 15-minute intervals")

# ==========================================
# 2. QUICK EDA
# ==========================================
print("\n2. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

print("\nTarget statistics (net_load_kwh):")
print(train_df['net_load_kwh'].describe())
print(f"\nNegative values (generation > consumption): {(train_df['net_load_kwh'] < 0).sum()} ({(train_df['net_load_kwh'] < 0).mean()*100:.1f}%)")

# Simple visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time series sample
train_df_plot = train_df.copy()
axes[0, 0].plot(train_df['timestamp_utc'][:672], train_df_plot['net_load_kwh'][:672], alpha=0.7)
axes[0, 0].set_title('Load Time Series (1 week sample)')
axes[0, 0].set_ylabel('Net Load (kWh/15min)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Distribution
axes[0, 1].hist(train_df_plot['net_load_kwh'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Zero')
axes[0, 1].set_title('Load Distribution')
axes[0, 1].set_xlabel('Net Load (kWh/15min)')
axes[0, 1].legend()

# Daily pattern
train_df_plot['hour'] = train_df_plot['timestamp_utc'].dt.hour + train_df_plot['timestamp_utc'].dt.minute/60
daily = train_df_plot.groupby('hour')['net_load_kwh'].mean()
axes[1, 0].plot(daily.index, daily.values, marker='o', markersize=3)
axes[1, 0].set_title('Average Daily Pattern')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Avg Net Load')
axes[1, 0].grid(True, alpha=0.3)

# Weekly pattern  
train_df_plot['dow'] = train_df_plot['timestamp_utc'].dt.dayofweek
weekly = train_df_plot.groupby('dow')['net_load_kwh'].mean()
axes[1, 1].bar(range(7), weekly.values, color='skyblue', edgecolor='navy')
axes[1, 1].set_title('Average Weekly Pattern')
axes[1, 1].set_xticks(range(7))
axes[1, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
axes[1, 1].set_ylabel('Avg Net Load')

plt.tight_layout()
plt.show()

# ==========================================
# 3. FETCH WEATHER DATA (OPEN-METEO)
# ==========================================
print("\n3. FETCHING WEATHER DATA")
print("-" * 40)

def fetch_weather_data(start_date, end_date):
    """
    Fetch historical weather data from Open-Meteo API
    for major Dutch cities
    """
    # Dutch cities
    locations = [
        (52.3676, 4.9041, 'Amsterdam'),
        (51.9244, 4.4777, 'Rotterdam'),
        (52.0907, 5.1214, 'Utrecht'),
        (51.4416, 5.4697, 'Eindhoven'),
        (53.2194, 6.5665, 'Groningen')
    ]
    
    # Weather features to fetch
    weather_features = [
        'temperature_2m',
        'relative_humidity_2m', 
        'dew_point_2m',
        'apparent_temperature',
        'precipitation',
        'rain',
        'pressure_msl',
        'surface_pressure',
        'cloud_cover',
        'wind_speed_10m',
        'wind_direction_10m',
        'wind_gusts_10m',
        'direct_radiation',
        'diffuse_radiation',
        'global_tilted_irradiance'
    ]
    
    all_weather = []
    
    for lat, lon, city in locations:
        print(f"  Fetching weather for {city}...")
        
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join(weather_features),
            'timezone': 'UTC'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            weather_df = pd.DataFrame(data['hourly'])
            weather_df['timestamp_utc'] = pd.to_datetime(weather_df['time'])
            weather_df = weather_df.drop('time', axis=1)
            weather_df = weather_df.set_index('timestamp_utc')
            
            # Resample to 15-minute intervals
            weather_df = weather_df.resample('15min').interpolate(method='linear')
            
            # Add city prefix
            weather_df = weather_df.add_prefix(f'{city}_')
            all_weather.append(weather_df)
            
        except Exception as e:
            print(f"    Error: {e}. Using synthetic weather.")
            
            # Generate synthetic weather as fallback
            dates = pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC')
            n = len(dates)
            np.random.seed(42 + locations.index((lat, lon, city)))
            
            weather_df = pd.DataFrame(index=dates)
            weather_df.index.name = 'timestamp_utc'
            
            # Synthetic patterns
            hour_of_year = (np.arange(n) % (365*24*4)) / 4
            hour_of_day = (np.arange(n) % (24*4)) / 4
            
            # Temperature with seasonal and daily variation
            weather_df[f'{city}_temperature_2m'] = (
                10 + 10*np.cos(2*np.pi*hour_of_year/(365*24)) +
                3*np.sin(2*np.pi*hour_of_day/24 - np.pi/2) +
                np.random.normal(0, 2, n)
            )
            
            weather_df[f'{city}_relative_humidity_2m'] = 70 + 15*np.sin(hour_of_year*np.pi/180) + np.random.normal(0, 5, n)
            weather_df[f'{city}_wind_speed_10m'] = np.abs(5 + 3*np.sin(hour_of_year*np.pi/90) + np.random.normal(0, 2, n))
            weather_df[f'{city}_pressure_msl'] = 1013 + 10*np.sin(hour_of_year*np.pi/720) + np.random.normal(0, 3, n)
            weather_df[f'{city}_cloud_cover'] = np.clip(50 + 30*np.sin(hour_of_day*np.pi/12) + np.random.normal(0, 20, n), 0, 100)
            
            # Solar radiation (peaks at noon, varies by season)
            solar_intensity = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12))
            seasonal_factor = 0.5 + 0.5 * np.cos(2 * np.pi * hour_of_year / (365*24))
            weather_df[f'{city}_direct_radiation'] = np.maximum(0,
                600 * solar_intensity * seasonal_factor * (hour_of_day >= 6) * (hour_of_day <= 18) +
                np.random.normal(0, 30, n)
            )
            
            all_weather.append(weather_df)
    
    if all_weather:
        # Combine all cities
        combined = pd.concat(all_weather, axis=1)
        
        # Add averages across cities
        for feature in weather_features:
            city_cols = [col for col in combined.columns if feature in col]
            if city_cols:
                combined[f'avg_{feature}'] = combined[city_cols].mean(axis=1)
                combined[f'max_{feature}'] = combined[city_cols].max(axis=1)
                combined[f'min_{feature}'] = combined[city_cols].min(axis=1)
        
        print(f"  Weather data shape: {combined.shape}")
        return combined
    
    return pd.DataFrame()

# Fetch weather for the entire period
start = (train_df['timestamp_utc'].min() - timedelta(days=4)).strftime('%Y-%m-%d')  # Extra days for lags
end = test_df['timestamp_utc'].max().strftime('%Y-%m-%d')
weather_df = fetch_weather_data(start, end)

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
print("\n4. FEATURE ENGINEERING")
print("-" * 40)

def create_features(df, weather_df):
    """
    Create features for modeling
    Respects 72-hour (288 steps) data latency
    """
    features = df.copy()
    features = features.set_index('timestamp_utc')
    
    # Merge weather data
    if not weather_df.empty:
        weather_aligned = weather_df.reindex(features.index, method='nearest')
        weather_aligned = weather_aligned.fillna(method='ffill').fillna(method='bfill')
        features = pd.concat([features, weather_aligned], axis=1)
    
    # Time-based features
    features['hour'] = features.index.hour
    features['minute'] = features.index.minute
    features['day_of_week'] = features.index.dayofweek
    features['month'] = features.index.month
    features['quarter'] = features.index.quarter
    features['day_of_month'] = features.index.day
    features['day_of_year'] = features.index.dayofyear
    features['week_of_year'] = features.index.isocalendar().week.astype(int)
    
    # Binary time features
    features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
    features['is_weekday'] = (features.index.dayofweek < 5).astype(int)
    features['is_monday'] = (features.index.dayofweek == 0).astype(int)
    features['is_friday'] = (features.index.dayofweek == 4).astype(int)
    
    # Time of day categories
    features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 5)).astype(int)
    features['is_morning'] = ((features['hour'] >= 6) & (features['hour'] <= 11)).astype(int)
    features['is_afternoon'] = ((features['hour'] >= 12) & (features['hour'] <= 17)).astype(int)
    features['is_evening'] = ((features['hour'] >= 18) & (features['hour'] <= 21)).astype(int)
    features['is_business_hour'] = ((features['hour'] >= 9) & (features['hour'] <= 17) & (features['is_weekday'] == 1)).astype(int)
    
    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    features['doy_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
    features['doy_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
    
    # Weather-based features
    if 'avg_temperature_2m' in features.columns:
        # Temperature features
        features['temp_squared'] = features['avg_temperature_2m'] ** 2
        features['temp_cubed'] = features['avg_temperature_2m'] ** 3
        
        # Heating and cooling degree (base temperatures for Netherlands)
        features['heating_degree'] = np.maximum(0, 18 - features['avg_temperature_2m'])
        features['cooling_degree'] = np.maximum(0, features['avg_temperature_2m'] - 24)
        
        # Temperature bins
        features['temp_very_cold'] = (features['avg_temperature_2m'] < 0).astype(int)
        features['temp_cold'] = ((features['avg_temperature_2m'] >= 0) & (features['avg_temperature_2m'] < 10)).astype(int)
        features['temp_mild'] = ((features['avg_temperature_2m'] >= 10) & (features['avg_temperature_2m'] < 20)).astype(int)
        features['temp_warm'] = ((features['avg_temperature_2m'] >= 20) & (features['avg_temperature_2m'] < 25)).astype(int)
        features['temp_hot'] = (features['avg_temperature_2m'] >= 25).astype(int)
        
        # Interactions
        features['temp_x_hour'] = features['avg_temperature_2m'] * features['hour']
        features['temp_x_weekend'] = features['avg_temperature_2m'] * features['is_weekend']
        features['temp_x_business'] = features['avg_temperature_2m'] * features['is_business_hour']
    
    if 'avg_wind_speed_10m' in features.columns:
        features['wind_squared'] = features['avg_wind_speed_10m'] ** 2
        features['high_wind'] = (features['avg_wind_speed_10m'] > 10).astype(int)
        
        if 'avg_temperature_2m' in features.columns:
            # Wind chill
            T = features['avg_temperature_2m']
            V = features['avg_wind_speed_10m']
            features['wind_chill'] = np.where(
                (T < 10) & (V > 4.8),
                13.12 + 0.6215*T - 11.37*V**0.16 + 0.3965*T*V**0.16,
                T
            )
    
    if 'avg_relative_humidity_2m' in features.columns and 'avg_temperature_2m' in features.columns:
        # Heat index
        T = features['avg_temperature_2m']
        RH = features['avg_relative_humidity_2m']
        features['heat_index'] = np.where(
            T >= 27,
            -8.785 + 1.611*T + 2.339*RH - 0.146*T*RH,
            T
        )
        
        # Comfort index
        features['discomfort_index'] = 0.5 * (T + 14.5 * (1 + 0.01 * RH))
    
    if 'avg_direct_radiation' in features.columns:
        features['radiation_sqrt'] = np.sqrt(features['avg_direct_radiation'])
        features['high_solar'] = (features['avg_direct_radiation'] > 500).astype(int)
        features['solar_x_temp'] = features.get('avg_temperature_2m', 0) * features['avg_direct_radiation'] / 100
    
    # Fourier features for seasonality
    for period_hours in [24, 24*7, 24*30]:
        period_steps = period_hours * 4  # Convert to 15-min steps
        for k in range(1, 3):
            features[f'fourier_{period_hours}h_{k}_sin'] = np.sin(2*np.pi*k*features.index.hour/period_hours)
            features[f'fourier_{period_hours}h_{k}_cos'] = np.cos(2*np.pi*k*features.index.hour/period_hours)
    
    # Dutch holidays
    holidays = []
    for year in features.index.year.unique():
        holidays.extend([
            pd.Timestamp(f'{year}-01-01', tz='UTC'),  # New Year
            pd.Timestamp(f'{year}-04-27', tz='UTC'),  # King's Day
            pd.Timestamp(f'{year}-05-05', tz='UTC'),  # Liberation Day
            pd.Timestamp(f'{year}-12-25', tz='UTC'),  # Christmas
            pd.Timestamp(f'{year}-12-26', tz='UTC'),  # Boxing Day
        ])
    
    features['is_holiday'] = features.index.normalize().isin(holidays).astype(int)
    
    return features

# Create features for train and test
print("Creating features for training data...")
train_features = create_features(train_df, weather_df)
print(f"  Train features shape: {train_features.shape}")

print("Creating features for test data...")
test_features = create_features(test_df, weather_df)
print(f"  Test features shape: {test_features.shape}")

# Fill missing values
train_features = train_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
test_features = test_features.fillna(method='ffill').fillna(method='bfill').fillna(0)

# Columns only in train/val
train_only = set(train_features.columns) - set(test_features.columns)
# Columns only in test
test_only = set(test_features.columns) - set(train_features.columns)

print("Columns only in train/val:", train_only)
print("Columns only in test:", test_only)


# ==========================================
# 5. PREPARE TRAINING DATA (48-HOUR AHEAD)
# ==========================================
print("\n5. PREPARING TRAINING DATA FOR 48-HOUR AHEAD PREDICTION")
print("-" * 40)

# Define prediction horizon
HORIZON_HOURS = 48
HORIZON_STEPS = HORIZON_HOURS * 4  # 192 steps at 15-minute intervals

# Get feature columns (exclude target and metadata)
feature_cols = [col for col in train_features.columns 
                if col not in ['net_load_kwh', 'row_id']]

print(f"Number of features: {len(feature_cols)}")

# Create training samples
# For each timestamp, we use features from 48 hours before to predict the value at that timestamp
X_train = []
y_train = []

for i in range(HORIZON_STEPS, len(train_features)):
    # Features from 48 hours ago (192 steps back)
    X_train.append(train_features[feature_cols].iloc[i - HORIZON_STEPS].values)
    # Target is the load at current timestamp
    y_train.append(train_features['net_load_kwh'].iloc[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Train/validation split (80/20)
val_size = 0.2
split_idx = int(len(X_train) * (1 - val_size))

X_tr = X_train[:split_idx]
X_val = X_train[split_idx:]
y_tr = y_train[:split_idx]
y_val = y_train[split_idx:]

print(f"Training samples: {X_tr.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

# ==========================================
# 6. MODEL TRAINING
# ==========================================
print("\n6. TRAINING MODELS")
print("-" * 40)

# Scale features
scaler_X = RobustScaler()
X_tr_scaled = scaler_X.fit_transform(X_tr)
X_val_scaled = scaler_X.transform(X_val)

# Initialize models
models = {}

# 1. LightGBM (Primary model)
print("Training LightGBM...")
models['lgb'] = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=100,
    max_depth=15,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
models['lgb'].fit(X_tr_scaled, y_tr)

# 2. XGBoost
print("Training XGBoost...")
models['xgb'] = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=12,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)
models['xgb'].fit(X_tr_scaled, y_tr)

# 3. Random Forest
print("Training Random Forest...")
models['rf'] = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
models['rf'].fit(X_tr_scaled, y_tr)

# ==========================================
# 7. MODEL EVALUATION
# ==========================================
print("\n7. EVALUATING MODELS")
print("-" * 40)

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate NRMSE and NMAE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    mean_load = np.mean(np.abs(y_true))
    nrmse = (rmse / mean_load) * 100
    nmae = (mae / mean_load) * 100
    
    print(f"\n{model_name}:")
    print(f"  NRMSE: {nrmse:.2f}%")
    print(f"  NMAE: {nmae:.2f}%")
    
    if nrmse < 5 and nmae < 5:
        print(f"  ✓ MEETS competition targets!")
    
    return {'nrmse': nrmse, 'nmae': nmae, 'rmse': rmse, 'mae': mae}

# Evaluate each model
val_metrics = {}
val_predictions = {}

for name, model in models.items():
    pred = model.predict(X_val_scaled)
    val_predictions[name] = pred
    val_metrics[name] = evaluate_model(y_val, pred, name.upper())

# Create ensemble
print("\nENSEMBLE (Average):")
ensemble_pred = np.mean([val_predictions[name] for name in models.keys()], axis=0)
ensemble_metrics = evaluate_model(y_val, ensemble_pred, "ENSEMBLE")

# Select best model
best_model_name = min(val_metrics.keys(), key=lambda x: val_metrics[x]['nrmse'])
print(f"\nBest individual model: {best_model_name.upper()}")

# ==========================================
# 8. RETRAIN ON FULL DATA
# ==========================================
print("\n8. RETRAINING BEST MODEL ON FULL TRAINING DATA")
print("-" * 40)

# Retrain best model on all training data
X_train_scaled = scaler_X.fit_transform(X_train)

if best_model_name == 'lgb':
    final_model = lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=120,
        max_depth=15,
        min_child_samples=15,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.08,
        reg_lambda=0.08,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
elif best_model_name == 'xgb':
    final_model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.025,
        max_depth=12,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.08,
        reg_lambda=0.08,
        random_state=42,
        n_jobs=-1
    )
else:
    final_model = models[best_model_name]

final_model.fit(X_train_scaled, y_train)
print("✓ Model retrained on full dataset")

# ==========================================
# 9. GENERATE TEST PREDICTIONS
# ==========================================
print("\n9. GENERATING TEST PREDICTIONS (48 HOURS AHEAD)")
print("-" * 40)

# For test predictions, we need the features from 48 hours BEFORE each test timestamp
# Combine train and test to get historical data for test predictions
all_features = pd.concat([train_features, test_features])
all_features = all_features.sort_index()

# Generate predictions for test set
test_predictions = []

for test_time in test_features.index:
    # Get the timestamp 48 hours before
    feature_time = test_time - timedelta(hours=48)
    
    if feature_time in all_features.index:
        # Use features from 48 hours ago
        X_test = all_features[feature_cols].loc[feature_time].values.reshape(1, -1)
    else:
        # If we don't have data from exactly 48 hours ago, use nearest available
        # This handles edge cases at the beginning of test period
        available_times = all_features.index[all_features.index < test_time]
        if len(available_times) > 0:
            # Use the closest available time that's at least 48 hours before
            valid_times = available_times[available_times <= feature_time]
            if len(valid_times) > 0:
                nearest_time = valid_times[-1]
            else:
                # Use the earliest available time
                nearest_time = available_times[0]
            X_test = all_features[feature_cols].loc[nearest_time].values.reshape(1, -1)
        else:
            # Fallback: use mean features
            X_test = np.mean(X_train, axis=0).reshape(1, -1)
    
    # Make prediction
    X_test_scaled = scaler_X.transform(X_test)
    pred = final_model.predict(X_test_scaled)[0]
    test_predictions.append(pred)

test_predictions = np.array(test_predictions)
print(f"Generated {len(test_predictions)} predictions for test set")

# ==========================================
# 10. CREATE SUBMISSION
# ==========================================
print("\n10. CREATING SUBMISSION FILE")
print("-" * 40)

# Create submission DataFrame
submission = pd.DataFrame({
    'row_id': test_df['row_id'].values,
    'predicted_net_load_kwh': test_predictions
})

# Verify submission
print(f"Submission shape: {submission.shape}")
print(f"Columns: {submission.columns.tolist()}")
print(f"Prediction range: [{submission['predicted_net_load_kwh'].min():.2f}, {submission['predicted_net_load_kwh'].max():.2f}]")
print(f"Prediction mean: {submission['predicted_net_load_kwh'].mean():.2f}")
print(f"Prediction std: {submission['predicted_net_load_kwh'].std():.2f}")

# Check for any NaN values
if submission.isnull().any().any():
    print("WARNING: Found NaN values in submission, filling with mean")
    submission = submission.fillna(submission['predicted_net_load_kwh'].mean())

# Save submission
submission.to_csv('submission.csv', index=False)
print("\n✓ Submission saved to 'submission.csv'")

# Display first and last rows
print("\nFirst 10 rows of submission:")
print(submission.head(10))
print("\nLast 10 rows of submission:")
print(submission.tail(10))

# ==========================================
# 11. FINAL SUMMARY
# ==========================================
print("\n" + "=" * 80)
print("COMPETITION SUMMARY")
print("=" * 80)

print(f"\nModel: {best_model_name.upper()}")
print(f"Validation Performance:")
print(f"  NRMSE: {val_metrics[best_model_name]['nrmse']:.2f}%")
print(f"  NMAE: {val_metrics[best_model_name]['nmae']:.2f}%")
print(f"Competition Target: NRMSE < 5%, NMAE < 5%")

print(f"\nPrediction Setup:")
print(f"  Forecast horizon: {HORIZON_HOURS} hours ({HORIZON_STEPS} steps)")
print(f"  Data frequency: 15-minute intervals")
print(f"  Data latency: 72 hours (respects 3-day lag requirement)")
print(f"  Weather sources: 5 Dutch cities from Open-Meteo")

print(f"\nFeatures Used: {len(feature_cols)}")
print(f"  Time features: Cyclical encoding, holidays, peak hours")
print(f"  Weather features: Temperature, humidity, wind, radiation")
print(f"  Derived features: Heating/cooling degree, wind chill, heat index")

print("\n" + "=" * 80)
print("SUBMISSION READY FOR KAGGLE!")
print("File: submission.csv")
print("Format: row_id, predicted_net_load_kwh")
print("=" * 80)




