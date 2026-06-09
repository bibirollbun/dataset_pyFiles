import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
import requests
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading data...")
train = pd.read_csv('/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv')
test = pd.read_csv('/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv')
sample_sub = pd.read_csv('/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/sample_submission_new.csv')

train['timestamp_utc'] = pd.to_datetime(train['timestamp_utc'])
test['timestamp_utc'] = pd.to_datetime(test['timestamp_utc'])

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Train date range: {train['timestamp_utc'].min()} to {train['timestamp_utc'].max()}")
print(f"Test date range: {test['timestamp_utc'].min()} to {test['timestamp_utc'].max()}")

# ============================================================================
# 2. FETCH WEATHER DATA
# ============================================================================
def fetch_weather_data(start_date, end_date, latitude=52.37, longitude=4.89):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "snow_depth", "weather_code", "pressure_msl", "surface_pressure",
            "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "et0_fao_evapotranspiration", "vapour_pressure_deficit",
            "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
            "wind_gusts_10m", "soil_temperature_0_to_7cm",
            "shortwave_radiation", "direct_radiation", "diffuse_radiation",
            "direct_normal_irradiance", "terrestrial_radiation"
        ],
        "timezone": "UTC"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    weather_df = pd.DataFrame({
        'timestamp_utc': pd.to_datetime(data['hourly']['time']),
        'temperature': data['hourly']['temperature_2m'],
        'apparent_temp': data['hourly']['apparent_temperature'],
        'humidity': data['hourly']['relative_humidity_2m'],
        'dew_point': data['hourly']['dew_point_2m'],
        'precipitation': data['hourly']['precipitation'],
        'rain': data['hourly']['rain'],
        'snowfall': data['hourly']['snowfall'],
        'snow_depth': data['hourly']['snow_depth'],
        'weather_code': data['hourly']['weather_code'],
        'pressure_msl': data['hourly']['pressure_msl'],
        'surface_pressure': data['hourly']['surface_pressure'],
        'cloud_cover': data['hourly']['cloud_cover'],
        'cloud_cover_low': data['hourly']['cloud_cover_low'],
        'cloud_cover_mid': data['hourly']['cloud_cover_mid'],
        'cloud_cover_high': data['hourly']['cloud_cover_high'],
        'evapotranspiration': data['hourly']['et0_fao_evapotranspiration'],
        'vapour_pressure_deficit': data['hourly']['vapour_pressure_deficit'],
        'wind_speed': data['hourly']['wind_speed_10m'],
        'wind_speed_100m': data['hourly']['wind_speed_100m'],
        'wind_direction': data['hourly']['wind_direction_10m'],
        'wind_gusts': data['hourly']['wind_gusts_10m'],
        'soil_temp': data['hourly']['soil_temperature_0_to_7cm'],
        'shortwave_radiation': data['hourly']['shortwave_radiation'],
        'direct_radiation': data['hourly']['direct_radiation'],
        'diffuse_radiation': data['hourly']['diffuse_radiation'],
        'direct_normal_irradiance': data['hourly']['direct_normal_irradiance'],
        'terrestrial_radiation': data['hourly']['terrestrial_radiation']
    })
    
    return weather_df

print("Fetching comprehensive weather data...")
train_start = (train['timestamp_utc'].min() - timedelta(days=14)).date()
train_end = train['timestamp_utc'].max().date()
test_end = (test['timestamp_utc'].max() + timedelta(days=1)).date()

weather = fetch_weather_data(str(train_start), str(test_end))
weather = weather.set_index('timestamp_utc').resample('15T').ffill().reset_index()

print(f"Weather data shape: {weather.shape}")

# ============================================================================
# 3. ADVANCED FEATURE ENGINEERING
# ============================================================================
def create_features(df, weather_df, full_data=None):
    df = df.copy()
    df = df.merge(weather_df, on='timestamp_utc', how='left')
    
    # Time features
    df['hour'] = df['timestamp_utc'].dt.hour
    df['dayofweek'] = df['timestamp_utc'].dt.dayofweek
    df['quarter_hour'] = df['timestamp_utc'].dt.minute // 15
    df['day'] = df['timestamp_utc'].dt.day
    df['month'] = df['timestamp_utc'].dt.month
    df['year'] = df['timestamp_utc'].dt.year
    df['dayofyear'] = df['timestamp_utc'].dt.dayofyear
    df['weekofyear'] = df['timestamp_utc'].dt.isocalendar().week
    df['quarter'] = df['timestamp_utc'].dt.quarter
    
    # Advanced time features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['is_weekend'] == 0)).astype(int)
    
    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter_hour'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter_hour'] / 4)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Weather interactions
    df['temp_squared'] = df['temperature'] ** 2
    df['temp_cubed'] = df['temperature'] ** 3
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['temp_wind'] = df['temperature'] * df['wind_speed']
    df['temp_pressure'] = df['temperature'] * df['pressure_msl']
    df['humidity_wind'] = df['humidity'] * df['wind_speed']
    df['total_radiation'] = df['direct_radiation'] + df['diffuse_radiation']
    df['radiation_cloud'] = df['shortwave_radiation'] * (100 - df['cloud_cover'])
    df['heating_degree_days'] = np.maximum(18 - df['temperature'], 0)
    df['cooling_degree_days'] = np.maximum(df['temperature'] - 18, 0)
    
    # Weather conditions
    df['is_raining'] = (df['precipitation'] > 0).astype(int)
    df['is_snowing'] = (df['snowfall'] > 0).astype(int)
    df['is_cloudy'] = (df['cloud_cover'] > 50).astype(int)
    df['is_windy'] = (df['wind_speed'] > 15).astype(int)
    
    # Weather rolling features (short-term trends)
    for col in ['temperature', 'humidity', 'wind_speed', 'pressure_msl', 'cloud_cover']:
        df[f'{col}_roll_mean_4'] = df[col].rolling(4, min_periods=1).mean()
        df[f'{col}_roll_std_4'] = df[col].rolling(4, min_periods=1).std()
        df[f'{col}_roll_mean_12'] = df[col].rolling(12, min_periods=1).mean()
        df[f'{col}_roll_diff_1'] = df[col].diff(1)
        df[f'{col}_roll_diff_4'] = df[col].diff(4)
    
    # Radiation features
    df['radiation_efficiency'] = df['diffuse_radiation'] / (df['shortwave_radiation'] + 1)
    df['direct_ratio'] = df['direct_radiation'] / (df['total_radiation'] + 1)
    
    # Lag features respecting 3-day latency (288 steps)
    if full_data is not None and 'net_load_kwh' in full_data.columns:
        # Sort by timestamp
        full_data = full_data.sort_values('timestamp_utc').reset_index(drop=True)
        
        # Create lag features
        lag_steps = [288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300,
                     336, 384, 432, 480, 576, 672]  # 3-7 days
        
        for lag in lag_steps:
            lag_col = f'load_lag_{lag}'
            full_data[lag_col] = full_data['net_load_kwh'].shift(lag)
        
        # Rolling statistics on lagged data
        base_lag = 288
        for window in [4, 8, 12, 24, 48, 96, 192]:
            full_data[f'load_roll_mean_{window}'] = full_data['net_load_kwh'].shift(base_lag).rolling(window, min_periods=1).mean()
            full_data[f'load_roll_std_{window}'] = full_data['net_load_kwh'].shift(base_lag).rolling(window, min_periods=1).std()
            full_data[f'load_roll_min_{window}'] = full_data['net_load_kwh'].shift(base_lag).rolling(window, min_periods=1).min()
            full_data[f'load_roll_max_{window}'] = full_data['net_load_kwh'].shift(base_lag).rolling(window, min_periods=1).max()
            full_data[f'load_roll_median_{window}'] = full_data['net_load_kwh'].shift(base_lag).rolling(window, min_periods=1).median()
        
        # Weekly patterns
        for i in [1, 2, 3, 4]:
            full_data[f'load_week_{i}'] = full_data['net_load_kwh'].shift(i * 7 * 96)
        
        # Hourly patterns (same hour different days)
        for i in [3, 4, 5, 6, 7]:
            full_data[f'load_same_hour_day_{i}'] = full_data['net_load_kwh'].shift(i * 96)
        
        # Merge lag features back
        lag_cols = [col for col in full_data.columns if col.startswith('load_')]
        df = df.merge(full_data[['timestamp_utc'] + lag_cols], on='timestamp_utc', how='left')
    
    # Exponential moving averages on weather
    for col in ['temperature', 'wind_speed', 'humidity']:
        df[f'{col}_ema_4'] = df[col].ewm(span=4, adjust=False).mean()
        df[f'{col}_ema_12'] = df[col].ewm(span=12, adjust=False).mean()
    
    return df

# Combine train and test for lag feature creation
print("Creating advanced features...")
full_data = pd.concat([train, test], ignore_index=True).sort_values('timestamp_utc').reset_index(drop=True)

train_features = create_features(train, weather, full_data)
test_features = create_features(test, weather, full_data)

# ============================================================================
# 4. PREPARE DATA
# ============================================================================
train_clean = train_features.dropna(subset=['net_load_kwh']).reset_index(drop=True)

exclude_cols = ['timestamp_utc', 'net_load_kwh']
feature_cols = [col for col in train_clean.columns if col not in exclude_cols and not train_clean[col].isna().all()]

X = train_clean[feature_cols].fillna(0)
y = train_clean['net_load_kwh']

print(f"Training samples: {len(X)}, Features: {len(feature_cols)}")

# Time-based split
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx].copy(), X[split_idx:].copy()
y_train, y_val = y[:split_idx].copy(), y[split_idx:].copy()

# ============================================================================
# 5. TRAIN ENSEMBLE MODELS
# ============================================================================
print("\n" + "="*60)
print("TRAINING ENSEMBLE MODELS")
print("="*60)

# Model 1: LightGBM
print("\n[1/3] Training LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 255,
    'learning_rate': 0.03,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_child_samples': 10,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'verbose': -1,
    'n_jobs': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

lgb_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    valid_names=['val'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

# Model 2: XGBoost
print("\n[2/3] Training XGBoost...")
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'min_child_weight': 5,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'verbosity': 0
}

xgb_model = xgb.train(
    xgb_params,
    xgb.DMatrix(X_train, label=y_train),
    num_boost_round=2000,
    evals=[(xgb.DMatrix(X_val, label=y_val), 'val')],
    early_stopping_rounds=50,
    verbose_eval=100
)

val_pred_xgb = xgb_model.predict(xgb.DMatrix(X_val))

# Model 3: CatBoost
print("\n[3/3] Training CatBoost...")
cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    verbose=100
)

val_pred_cat = cat_model.predict(X_val)

# ============================================================================
# 6. OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================================
print("\n" + "="*60)
print("OPTIMIZING ENSEMBLE WEIGHTS")
print("="*60)

from scipy.optimize import minimize

def ensemble_loss(weights, preds, y_true):
    ensemble_pred = sum(w * p for w, p in zip(weights, preds))
    rmse = np.sqrt(np.mean((y_true - ensemble_pred) ** 2))
    return rmse

predictions = [val_pred_lgb, val_pred_xgb, val_pred_cat]
initial_weights = [1/3, 1/3, 1/3]

result = minimize(
    lambda w: ensemble_loss(w, predictions, y_val),
    initial_weights,
    method='SLSQP',
    bounds=[(0, 1)] * 3,
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)

optimal_weights = result.x
print(f"Optimal weights: LGB={optimal_weights[0]:.3f}, XGB={optimal_weights[1]:.3f}, CAT={optimal_weights[2]:.3f}")

val_pred_ensemble = sum(w * p for w, p in zip(optimal_weights, predictions))

# ============================================================================
# 7. EVALUATE
# ============================================================================
rmse = np.sqrt(np.mean((y_val - val_pred_ensemble) ** 2))
mae = np.mean(np.abs(y_val - val_pred_ensemble))
nrmse = rmse / (y_val.max() - y_val.min()) * 100
nmae = mae / (y_val.max() - y_val.min()) * 100

print(f"\n{'='*60}")
print(f"ENSEMBLE VALIDATION RESULTS:")
print(f"{'='*60}")
print(f"RMSE:  {rmse:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"NRMSE: {nrmse:.2f}%")
print(f"NMAE:  {nmae:.2f}%")
print(f"{'='*60}")

# ============================================================================
# 8. PREDICT ON TEST SET
# ============================================================================
X_test_with_nan = test_features[feature_cols]

print("\nNaN values on X_test before forward fill:")
# Check number of NaN values before forward fill
nan_counts = X_test_with_nan.isna().sum()
with pd.option_context('display.max_rows', None):
    print(nan_counts[nan_counts > 0])

# MAIN CHANGE: Change from replacing of 0, to using forward fill to just use past lag-value as new lag value
# even better would be to use auto-regression to fill NaN lag-features before predicting next time-step
X_test = X_test_with_nan.ffill() 

# Check number of NaN values after forward fill
print("\nNaN values on X_test after forward fill:")
nan_counts = X_test.isna().sum()
with pd.option_context('display.max_rows', None):
    print(nan_counts[nan_counts > 0])

print("\nGenerating test predictions...")
test_pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
test_pred_xgb = xgb_model.predict(xgb.DMatrix(X_test))
test_pred_cat = cat_model.predict(X_test)

test_pred_ensemble = sum(w * p for w, p in zip(optimal_weights, [test_pred_lgb, test_pred_xgb, test_pred_cat]))

# ============================================================================
# 9. CREATE SUBMISSION
# ============================================================================
submission = sample_sub.copy()
submission['net_load_kwh'] = test_pred_ensemble
submission.to_csv('submission.csv', index=False)

print("\nâœ“ Submission created: submission.csv")
print(f"\nPrediction stats: Min={test_pred_ensemble.min():.2f}, Max={test_pred_ensemble.max():.2f}, Mean={test_pred_ensemble.mean():.2f}")
print("\nDone! ðŸŽ¯")




