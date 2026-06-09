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
Dutch Energy Forecasting - CORRECTED VERSION
=============================================
Properly implements 72-hour latency + 48-hour horizon constraint
Target: NRMSE < 5%, NMAE < 5%
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, HuberRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import iqr

print("=" * 80)
print("DUTCH ENERGY FORECASTING - CORRECTED VERSION")
print("Properly Implements 72h Latency + 48h Horizon")
print("=" * 80)

# ==========================================
# CRITICAL CONSTANTS
# ==========================================
STEPS_PER_HOUR = 4  # 15-minute intervals

# Competition constraints
MIN_LAG_HOURS = 72      # Data latency: can only use data from >=72h ago
HORIZON_HOURS = 48      # Prediction horizon: predict 48h ahead

MIN_LAG_STEPS = MIN_LAG_HOURS * STEPS_PER_HOUR   # 288 steps
HORIZON_STEPS = HORIZON_HOURS * STEPS_PER_HOUR   # 192 steps
TOTAL_LOOKBACK_STEPS = MIN_LAG_STEPS + HORIZON_STEPS  # 480 steps (120h)

print(f"\nCONSTRAINTS:")
print(f"  Minimum lag: {MIN_LAG_HOURS}h ({MIN_LAG_STEPS} steps)")
print(f"  Prediction horizon: {HORIZON_HOURS}h ({HORIZON_STEPS} steps)")
print(f"  Total lookback needed: {MIN_LAG_HOURS + HORIZON_HOURS}h ({TOTAL_LOOKBACK_STEPS} steps)")

# ==========================================
# 1. LOAD DATA
# ==========================================
print("\n1. LOADING DATA")
print("-" * 40)

train_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv'
test_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"✓ Train: {train_df.shape}")
print(f"✓ Test: {test_df.shape}")

# Standardize column names
if 'Datetime' in train_df.columns:
    train_df = train_df.rename(columns={'Datetime': 'timestamp_utc', 'Actual Net': 'net_load_kwh'})
if 'Datetime' in test_df.columns:
    test_df = test_df.rename(columns={'Datetime': 'timestamp_utc'})
    if 'Actual Net' in test_df.columns:
        test_df = test_df.rename(columns={'Actual Net': 'net_load_kwh'})

train_df['timestamp_utc'] = pd.to_datetime(train_df['timestamp_utc'])
test_df['timestamp_utc'] = pd.to_datetime(test_df['timestamp_utc'])

train_df = train_df.sort_values('timestamp_utc').reset_index(drop=True)
test_df = test_df.sort_values('timestamp_utc').reset_index(drop=True)

if 'row_id' not in test_df.columns:
    test_df['row_id'] = range(len(test_df))

print(f"Train: {train_df['timestamp_utc'].min()} to {train_df['timestamp_utc'].max()}")
print(f"Test: {test_df['timestamp_utc'].min()} to {test_df['timestamp_utc'].max()}")

# ==========================================
# 2. FETCH WEATHER DATA
# ==========================================
print("\n2. FETCHING WEATHER DATA")
print("-" * 40)

def fetch_weather_data(start_date, end_date):
    """Fetch weather data from Open-Meteo API"""
    locations = [
        (52.3676, 4.9041, 'Amsterdam'),
        (51.9244, 4.4777, 'Rotterdam'),
        (52.0907, 5.1214, 'Utrecht'),
        (51.4416, 5.4697, 'Eindhoven'),
        (53.2194, 6.5665, 'Groningen')
    ]
    
    weather_features = [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'precipitation', 'rain',
        'pressure_msl', 'surface_pressure', 'cloud_cover',
        'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
        'direct_radiation', 'diffuse_radiation', 'shortwave_radiation'
    ]
    
    all_weather = []
    
    for lat, lon, city in locations:
        print(f"  Fetching {city}...")
        try:
            response = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    'latitude': lat, 'longitude': lon,
                    'start_date': start_date, 'end_date': end_date,
                    'hourly': ','.join(weather_features),
                    'timezone': 'UTC'
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            weather_df = pd.DataFrame(data['hourly'])
            weather_df['timestamp_utc'] = pd.to_datetime(weather_df['time'])
            weather_df = weather_df.drop('time', axis=1).set_index('timestamp_utc')
            weather_df = weather_df.resample('15min').interpolate(method='linear')
            weather_df = weather_df.add_prefix(f'{city}_')
            all_weather.append(weather_df)
            
        except Exception as e:
            print(f"    Using synthetic data for {city}")
            dates = pd.date_range(start=start_date, end=end_date, freq='15min')
            n = len(dates)
            np.random.seed(42 + locations.index((lat, lon, city)))
            
            weather_df = pd.DataFrame(index=dates)
            weather_df.index.name = 'timestamp_utc'
            
            hour_of_year = (np.arange(n) % (365*24*4)) / 4
            hour_of_day = (np.arange(n) % (24*4)) / 4
            
            weather_df[f'{city}_temperature_2m'] = (
                10 + 10*np.cos(2*np.pi*hour_of_year/(365*24)) +
                3*np.sin(2*np.pi*hour_of_day/24 - np.pi/2) +
                np.random.normal(0, 2, n)
            )
            weather_df[f'{city}_relative_humidity_2m'] = np.clip(
                70 + 15*np.sin(hour_of_year*np.pi/180) + np.random.normal(0, 5, n), 0, 100
            )
            weather_df[f'{city}_wind_speed_10m'] = np.abs(
                5 + 3*np.sin(hour_of_year*np.pi/90) + np.random.normal(0, 2, n)
            )
            weather_df[f'{city}_pressure_msl'] = (
                1013 + 10*np.sin(hour_of_year*np.pi/720) + np.random.normal(0, 3, n)
            )
            weather_df[f'{city}_direct_radiation'] = np.maximum(0,
                600 * np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12)) * 
                (0.5 + 0.5 * np.cos(2 * np.pi * hour_of_year / (365*24))) *
                (hour_of_day >= 6) * (hour_of_day <= 18)
            )
            
            all_weather.append(weather_df)
    
    if all_weather:
        combined = pd.concat(all_weather, axis=1)
        
        # Aggregate features
        for feature in weather_features:
            city_cols = [col for col in combined.columns if feature in col]
            if city_cols:
                combined[f'avg_{feature}'] = combined[city_cols].mean(axis=1)
                combined[f'max_{feature}'] = combined[city_cols].max(axis=1)
                combined[f'min_{feature}'] = combined[city_cols].min(axis=1)
                combined[f'std_{feature}'] = combined[city_cols].std(axis=1)
        
        print(f"  Weather shape: {combined.shape}")
        return combined
    
    return pd.DataFrame()

start = (train_df['timestamp_utc'].min() - timedelta(days=30)).strftime('%Y-%m-%d')
end = test_df['timestamp_utc'].max().strftime('%Y-%m-%d')
weather_df = fetch_weather_data(start, end)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("\n3. FEATURE ENGINEERING")
print("-" * 40)

def create_features(df, weather_df, include_target_lags=True):
    """
    Create features respecting the 72-hour latency constraint.
    
    CRITICAL: When include_target_lags=True, all lag features must be >=72 hours.
    Weather features can be from current time (available in real-time).
    """
    features = df.copy()
    features = features.set_index('timestamp_utc')
    
    # === WEATHER FEATURES (current time - these are allowed) ===
    if not weather_df.empty:
        weather_aligned = weather_df.reindex(features.index, method='nearest')
        weather_aligned = weather_aligned.ffill().bfill()
        features = pd.concat([features, weather_aligned], axis=1)
    
    # === TIME FEATURES (always available) ===
    features['hour'] = features.index.hour
    features['minute'] = features.index.minute
    features['day_of_week'] = features.index.dayofweek
    features['month'] = features.index.month
    features['quarter'] = features.index.quarter
    features['day_of_year'] = features.index.dayofyear
    features['week_of_year'] = features.index.isocalendar().week.astype(int)
    
    # Binary indicators
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    features['is_weekday'] = (features['day_of_week'] < 5).astype(int)
    features['is_monday'] = (features['day_of_week'] == 0).astype(int)
    features['is_friday'] = (features['day_of_week'] == 4).astype(int)
    features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 5)).astype(int)
    features['is_morning'] = ((features['hour'] >= 6) & (features['hour'] <= 11)).astype(int)
    features['is_afternoon'] = ((features['hour'] >= 12) & (features['hour'] <= 17)).astype(int)
    features['is_evening'] = ((features['hour'] >= 18) & (features['hour'] <= 21)).astype(int)
    features['is_business_hour'] = ((features['hour'] >= 9) & (features['hour'] <= 17) & 
                                    (features['is_weekday'] == 1)).astype(int)
    features['is_peak_morning'] = ((features['hour'] >= 7) & (features['hour'] <= 9)).astype(int)
    features['is_peak_evening'] = ((features['hour'] >= 17) & (features['hour'] <= 20)).astype(int)
    
    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    features['doy_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
    features['doy_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
    
    # === WEATHER-DERIVED FEATURES ===
    if 'avg_temperature_2m' in features.columns:
        T = features['avg_temperature_2m']
        
        # Polynomial
        features['temp_squared'] = T ** 2
        features['temp_cubed'] = T ** 3
        
        # Degree days (energy demand indicators)
        features['heating_degree'] = np.maximum(0, 18 - T)
        features['cooling_degree'] = np.maximum(0, T - 24)
        features['heating_degree_sq'] = features['heating_degree'] ** 2
        features['cooling_degree_sq'] = features['cooling_degree'] ** 2
        
        # Temperature bins
        features['temp_very_cold'] = (T < 0).astype(int)
        features['temp_cold'] = ((T >= 0) & (T < 10)).astype(int)
        features['temp_mild'] = ((T >= 10) & (T < 20)).astype(int)
        features['temp_warm'] = ((T >= 20) & (T < 25)).astype(int)
        features['temp_hot'] = (T >= 25).astype(int)
        
        # Interactions with time
        features['temp_x_hour'] = T * features['hour']
        features['temp_x_weekend'] = T * features['is_weekend']
        features['temp_x_business'] = T * features['is_business_hour']
        features['temp_x_dow_sin'] = T * features['dow_sin']
        
        # Rolling statistics on temperature (recent trends)
        for window in [4, 8, 12, 24, 48, 96]:
            features[f'temp_roll_mean_{window}'] = T.rolling(window, min_periods=1).mean()
            features[f'temp_roll_std_{window}'] = T.rolling(window, min_periods=1).std()
        
        # EMA
        for span in [4, 12, 24, 96]:
            features[f'temp_ema_{span}'] = T.ewm(span=span, min_periods=1).mean()
        
        # Temperature momentum
        features['temp_momentum_4'] = T - T.shift(4)
        features['temp_momentum_12'] = T - T.shift(12)
        features['temp_momentum_24'] = T - T.shift(24)
    
    # Wind features
    if 'avg_wind_speed_10m' in features.columns:
        W = features['avg_wind_speed_10m']
        features['wind_squared'] = W ** 2
        features['wind_cubed'] = W ** 3  # Wind power
        features['high_wind'] = (W > 10).astype(int)
        features['wind_ema_12'] = W.ewm(span=12, min_periods=1).mean()
        
        if 'avg_temperature_2m' in features.columns:
            T = features['avg_temperature_2m']
            features['wind_chill'] = np.where(
                (T < 10) & (W > 4.8),
                13.12 + 0.6215*T - 11.37*W**0.16 + 0.3965*T*W**0.16,
                T
            )
    
    # Solar radiation features
    if 'avg_direct_radiation' in features.columns:
        R = features['avg_direct_radiation']
        features['radiation_sqrt'] = np.sqrt(R)
        features['radiation_log'] = np.log1p(R)
        features['high_solar'] = (R > 500).astype(int)
        
        if 'avg_temperature_2m' in features.columns:
            features['solar_x_temp'] = R * features['avg_temperature_2m'] / 100
        
        for window in [12, 24, 96]:
            features[f'radiation_ema_{window}'] = R.ewm(span=window, min_periods=1).mean()
    
    # Humidity features
    if 'avg_relative_humidity_2m' in features.columns and 'avg_temperature_2m' in features.columns:
        T = features['avg_temperature_2m']
        RH = features['avg_relative_humidity_2m']
        features['heat_index'] = np.where(
            T >= 27,
            -8.785 + 1.611*T + 2.339*RH - 0.146*T*RH,
            T
        )
    
    # === FOURIER FEATURES (Multi-scale seasonality) ===
    for period_hours in [6, 12, 24, 24*7, 24*14, 24*30]:
        for k in range(1, 4):
            features[f'fourier_{period_hours}h_{k}_sin'] = np.sin(
                2*np.pi*k*features.index.hour/period_hours
            )
            features[f'fourier_{period_hours}h_{k}_cos'] = np.cos(
                2*np.pi*k*features.index.hour/period_hours
            )
    
    # === HOLIDAYS ===
    holidays = []
    for year in features.index.year.unique():
        holidays.extend([
            pd.Timestamp(f'{year}-01-01'),
            pd.Timestamp(f'{year}-04-27'),
            pd.Timestamp(f'{year}-05-05'),
            pd.Timestamp(f'{year}-12-25'),
            pd.Timestamp(f'{year}-12-26'),
        ])
    
    features['is_holiday'] = features.index.normalize().isin(
        [pd.Timestamp(h.date()) for h in holidays]
    ).astype(int)
    
    days_to_holiday = []
    for idx in features.index:
        date_only = pd.Timestamp(idx.date())
        distances = [abs((h - date_only).days) for h in 
                    [pd.Timestamp(holiday.date()) for holiday in holidays]]
        days_to_holiday.append(min(distances) if distances else 365)
    features['days_to_holiday'] = days_to_holiday
    features['days_to_holiday_sin'] = np.sin(2 * np.pi * features['days_to_holiday'] / 365)
    
    # === TARGET LAG FEATURES (CRITICAL: >=72h only) ===
    if include_target_lags and 'net_load_kwh' in features.columns:
        # Direct lags (all >=72h)
        for lag_hours in [73, 84, 96, 120, 144, 168, 336, 504]:
            lag_steps = lag_hours * STEPS_PER_HOUR
            features[f'lag_{lag_hours}h'] = features['net_load_kwh'].shift(lag_steps)
        
        # Rolling statistics on 72h+ lagged data
        lagged_series = features['net_load_kwh'].shift(MIN_LAG_STEPS)
        
        for window_hours in [24, 48, 72, 168]:
            window_steps = window_hours * STEPS_PER_HOUR
            features[f'lag_roll_mean_{window_hours}h'] = lagged_series.rolling(
                window_steps, min_periods=1
            ).mean()
            features[f'lag_roll_std_{window_hours}h'] = lagged_series.rolling(
                window_steps, min_periods=1
            ).std()
            features[f'lag_roll_min_{window_hours}h'] = lagged_series.rolling(
                window_steps, min_periods=1
            ).min()
            features[f'lag_roll_max_{window_hours}h'] = lagged_series.rolling(
                window_steps, min_periods=1
            ).max()
        
        # EMA on lagged data
        for span in [12, 24, 48, 96]:
            features[f'lag_ema_{span}'] = lagged_series.ewm(span=span, min_periods=1).mean()
        
        # Same time different days (from 72h+ ago)
        for days_back in [4, 7, 14, 28]:
            if days_back * 96 > MIN_LAG_STEPS:
                features[f'same_time_{days_back}d_ago'] = features['net_load_kwh'].shift(
                    days_back * 96
                )
        
        # Diff features
        if 'lag_73h' in features.columns and 'lag_96h' in features.columns:
            features['lag_diff_73_96'] = features['lag_73h'] - features['lag_96h']
        if 'lag_96h' in features.columns and 'lag_168h' in features.columns:
            features['lag_diff_96_168'] = features['lag_96h'] - features['lag_168h']
    
    return features

print("Creating features for training...")
train_features = create_features(train_df, weather_df, include_target_lags=True)
print(f"  Train features: {train_features.shape}")

print("Creating features for test...")
test_features = create_features(test_df, weather_df, include_target_lags=False)
print(f"  Test features: {test_features.shape}")

# Fill missing values
train_features = train_features.ffill().bfill().fillna(0)
test_features = test_features.ffill().bfill().fillna(0)

# ==========================================
# 4. FEATURE SELECTION
# ==========================================
print("\n4. FEATURE SELECTION")
print("-" * 40)

# Get common features
feature_cols = [col for col in train_features.columns 
                if col not in ['net_load_kwh', 'row_id'] 
                and col in test_features.columns]

print(f"Initial features: {len(feature_cols)}")

# Remove highly correlated features
if len(train_features) > TOTAL_LOOKBACK_STEPS + 1000:
    X_sample = train_features[feature_cols].iloc[
        TOTAL_LOOKBACK_STEPS:TOTAL_LOOKBACK_STEPS+1000
    ].copy()
    corr_matrix = X_sample.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    feature_cols = [f for f in feature_cols if f not in to_drop]
    print(f"After removing high correlation (>0.95): {len(feature_cols)}")

# ==========================================
# 5. PREPARE TRAINING DATA - **CORRECTED**
# ==========================================
print("\n5. PREPARING TRAINING DATA (CORRECTED)")
print("-" * 40)

X_train = []
y_train = []

# CRITICAL FIX: Proper temporal alignment
# We need data from i-480 (120h ago) to predict i-288 (72h ago), which is 48h ahead
for i in range(TOTAL_LOOKBACK_STEPS, len(train_features)):
    # Features from 120 hours ago
    feature_idx = i - TOTAL_LOOKBACK_STEPS
    
    # Target is 48h ahead from feature time (= 72h ago from current time i)
    target_idx = i - MIN_LAG_STEPS
    
    X_train.append(train_features[feature_cols].iloc[feature_idx].values)
    y_train.append(train_features['net_load_kwh'].iloc[target_idx])

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"✓ X_train: {X_train.shape}")
print(f"✓ y_train: {y_train.shape}")
print(f"✓ Effective training samples: {len(X_train)}")

# Validation split (time-based)
val_size = 0.15
split_idx = int(len(X_train) * (1 - val_size))

X_tr = X_train[:split_idx]
X_val = X_train[split_idx:]
y_tr = y_train[:split_idx]
y_val = y_train[split_idx:]

print(f"✓ Training: {X_tr.shape[0]}, Validation: {X_val.shape[0]}")

# ==========================================
# 6. TRAIN MODELS
# ==========================================
print("\n6. TRAINING MODELS")
print("-" * 40)

# Scale features
scaler_X = RobustScaler()
X_tr_scaled = scaler_X.fit_transform(X_tr)
X_val_scaled = scaler_X.transform(X_val)

models = {}

# 1. LightGBM
print("Training LightGBM...")
models['lgb'] = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.015,
    num_leaves=200,
    max_depth=25,
    min_child_samples=10,
    subsample=0.92,
    colsample_bytree=0.92,
    reg_alpha=0.03,
    reg_lambda=0.03,
    min_split_gain=0.008,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
models['lgb'].fit(
    X_tr_scaled, y_tr,
    eval_set=[(X_val_scaled, y_val)],
    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]
)

# 2. XGBoost
print("Training XGBoost...")
models['xgb'] = xgb.XGBRegressor(
    n_estimators=1200,
    learning_rate=0.018,
    max_depth=18,
    min_child_weight=2,
    subsample=0.92,
    colsample_bytree=0.92,
    reg_alpha=0.03,
    reg_lambda=0.03,
    gamma=0.008,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
models['xgb'].fit(X_tr_scaled, y_tr, eval_set=[(X_val_scaled, y_val)], verbose=False)

# 3. CatBoost
print("Training CatBoost...")
models['cat'] = cb.CatBoostRegressor(
    iterations=1000,
    learning_rate=0.02,
    depth=12,
    l2_leaf_reg=0.5,
    random_seed=42,
    verbose=0
)
models['cat'].fit(X_tr_scaled, y_tr, eval_set=(X_val_scaled, y_val), 
                  early_stopping_rounds=100, verbose=False)

# 4. Gradient Boosting
print("Training Gradient Boosting...")
models['gb'] = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.025,
    max_depth=14,
    min_samples_split=12,
    min_samples_leaf=8,
    subsample=0.92,
    random_state=42
)
models['gb'].fit(X_tr_scaled, y_tr)

# 5. Extra Trees
print("Training Extra Trees...")
models['et'] = ExtraTreesRegressor(
    n_estimators=400,
    max_depth=30,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
models['et'].fit(X_tr_scaled, y_tr)

# 6. Huber
print("Training Huber...")
models['huber'] = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=200)
models['huber'].fit(X_tr_scaled, y_tr)

# ==========================================
# 7. EVALUATE & ENSEMBLE
# ==========================================
print("\n7. EVALUATION & ENSEMBLE")
print("-" * 40)

def evaluate_model(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mean_load = np.mean(np.abs(y_true))
    nrmse = (rmse / mean_load) * 100
    nmae = (mae / mean_load) * 100
    
    print(f"\n{name}:")
    print(f"  NRMSE: {nrmse:.2f}%")
    print(f"  NMAE: {nmae:.2f}%")
    
    if nrmse < 5 and nmae < 5:
        print(f"  ✓ MEETS TARGET!")
    
    return {'nrmse': nrmse, 'nmae': nmae}

# Evaluate individual models
val_metrics = {}
for name, model in models.items():
    pred = model.predict(X_val_scaled)
    val_metrics[name] = evaluate_model(y_val, pred, name.upper())

# Stacking ensemble
print("\nStacking Ensemble...")
stack_X_tr = np.column_stack([model.predict(X_tr_scaled) for model in models.values()])
stack_X_val = np.column_stack([model.predict(X_val_scaled) for model in models.values()])

meta_models = {
    'ridge': Ridge(alpha=0.5),
    'huber': HuberRegressor(epsilon=1.5),
    'lgb': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, verbose=-1)
}

meta_preds = {}
for name, meta_model in meta_models.items():
    meta_model.fit(stack_X_tr, y_tr)
    pred = meta_model.predict(stack_X_val)
    meta_preds[name] = pred
    evaluate_model(y_val, pred, f"STACKED-{name.upper()}")

# Select best meta-learner
best_meta = min(meta_preds.keys(), 
                key=lambda x: np.sqrt(mean_squared_error(y_val, meta_preds[x])))
print(f"\n✓ Best meta-learner: {best_meta.upper()}")
final_meta = meta_models[best_meta]

# ==========================================
# 8. RETRAIN ON FULL DATA
# ==========================================
print("\n8. RETRAINING ON FULL DATA")
print("-" * 40)

X_train_scaled = scaler_X.fit_transform(X_train)

for name in list(models.keys()):
    print(f"Retraining {name}...")
    if name == 'lgb':
        model = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.015, num_leaves=200,
                                   max_depth=25, min_child_samples=10, subsample=0.92,
                                   colsample_bytree=0.92, reg_alpha=0.03, reg_lambda=0.03,
                                   min_split_gain=0.008, random_state=42, verbose=-1, n_jobs=-1)
    elif name == 'xgb':
        model = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.018, max_depth=18,
                                  min_child_weight=2, subsample=0.92, colsample_bytree=0.92,
                                  reg_alpha=0.03, reg_lambda=0.03, gamma=0.008,
                                  random_state=42, n_jobs=-1, verbosity=0)
    elif name == 'cat':
        model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.02, depth=12,
                                       l2_leaf_reg=0.5, random_seed=42, verbose=0)
    else:
        model = models[name]
    
    model.fit(X_train_scaled, y_train)
    models[name] = model

# Retrain meta-learner
stack_X_full = np.column_stack([model.predict(X_train_scaled) for model in models.values()])
final_meta.fit(stack_X_full, y_train)

print("✓ All models retrained")

# ==========================================
# 9. GENERATE TEST PREDICTIONS - **CORRECTED**
# ==========================================
print("\n9. GENERATING TEST PREDICTIONS (CORRECTED)")
print("-" * 40)

# Combine train and test features for lookup
all_features = pd.concat([train_features, test_features])
all_features = all_features.sort_index()

test_predictions = []

for test_time in test_features.index:
    # CRITICAL FIX: Use features from 120 hours ago
    feature_time = test_time - timedelta(hours=MIN_LAG_HOURS + HORIZON_HOURS)
    
    if feature_time in all_features.index:
        X_test = all_features[feature_cols].loc[feature_time].values.reshape(1, -1)
    else:
        # Fallback: nearest available time
        available = all_features.index[all_features.index <= feature_time]
        if len(available) > 0:
            nearest = available[-1]
            X_test = all_features[feature_cols].loc[nearest].values.reshape(1, -1)
        else:
            X_test = np.zeros((1, len(feature_cols)))
    
    X_test_scaled = scaler_X.transform(X_test)
    
    # Ensemble prediction
    stack_input = np.column_stack([model.predict(X_test_scaled) for model in models.values()])
    pred = final_meta.predict(stack_input)[0]
    
    test_predictions.append(pred)

test_predictions = np.array(test_predictions)
print(f"✓ Generated {len(test_predictions)} predictions")

# ==========================================
# 10. POST-PROCESSING
# ==========================================
print("\n10. POST-PROCESSING")
print("-" * 40)

# Smoothing
try:
    smoothed = savgol_filter(test_predictions, window_length=11, polyorder=3, mode='nearest')
except:
    smoothed = gaussian_filter1d(test_predictions, sigma=1.2, mode='nearest')

# Outlier correction
median = np.median(smoothed)
mad = np.median(np.abs(smoothed - median))
threshold = 5 * mad
outliers = np.abs(smoothed - median) > threshold

if outliers.any():
    print(f"Correcting {outliers.sum()} outliers...")
    for i in np.where(outliers)[0]:
        if i > 0 and i < len(smoothed) - 1:
            smoothed[i] = (smoothed[i-1] + smoothed[i+1]) / 2
        elif i == 0:
            smoothed[i] = smoothed[i+1]
        else:
            smoothed[i] = smoothed[i-1]

# Blend
final_predictions = 0.8 * smoothed + 0.2 * test_predictions

# ==========================================
# 11. CREATE SUBMISSION
# ==========================================
print("\n11. CREATING SUBMISSION")
print("-" * 40)

submission = pd.DataFrame({
    'row_id': test_df['row_id'].values,
    'predicted_net_load_kwh': final_predictions
})

if submission.isnull().any().any():
    print("WARNING: NaN found, filling...")
    submission = submission.fillna(submission['predicted_net_load_kwh'].mean())

submission.to_csv('submission.csv', index=False)

print(f"✓ Submission saved")
print(f"  Shape: {submission.shape}")
print(f"  Range: [{submission['predicted_net_load_kwh'].min():.2f}, "
      f"{submission['predicted_net_load_kwh'].max():.2f}]")
print(f"  Mean: {submission['predicted_net_load_kwh'].mean():.2f}")
print(f"  Std: {submission['predicted_net_load_kwh'].std():.2f}")

print("\nFirst 10 rows:")
print(submission.head(10))

# ==========================================
# 12. SUMMARY
# ==========================================
print("\n" + "=" * 80)
print("CORRECTED MODEL SUMMARY")
print("=" * 80)

print(f"\nBest Meta-Learner: {best_meta.upper()}")
best_val_metrics = evaluate_model(y_val, meta_preds[best_meta], "FINAL VALIDATION")

print(f"\nCRITICAL FIXES APPLIED:")
print(f"  ✓ Training uses features from 120h ago (not 48h)")
print(f"  ✓ Target is 48h ahead from feature time (72h ago from 'now')")
print(f"  ✓ Test predictions use 120h lookback (not 48h)")
print(f"  ✓ All lag features respect 72h minimum")

print(f"\nTechniques Applied:")
print(f"  ✓ {len(feature_cols)} engineered features")
print(f"  ✓ Weather data from Open-Meteo API")
print(f"  ✓ 6 diverse models (LGB, XGB, Cat, GB, ET, Huber)")
print(f"  ✓ Multi-level stacking ensemble")
print(f"  ✓ Savitzky-Golay smoothing")
print(f"  ✓ MAD-based outlier correction")

print("\n" + "=" * 80)
print("SUBMISSION READY!")
print("=" * 80)

