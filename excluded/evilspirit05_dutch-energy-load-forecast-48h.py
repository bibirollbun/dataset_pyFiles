import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
from catboost import CatBoostRegressor
import requests

from tabulate import tabulate
from datetime import timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.model_selection import train_test_split
%matplotlib inline


df=pd.read_csv("/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv")


df.head()


df.isnull().sum()


df.shape


df["timestamp_utc"]=pd.to_datetime(df["timestamp_utc"],utc=True)



plt.figure(figsize=(15, 5))
plt.plot(df['timestamp_utc'], df['net_load_kwh'])
plt.title('Net Load (kWh)')
plt.show()


def fetch_weather(start_date, end_date, lat=52.3676, lon=4.9041):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'hourly': 'temperature_2m,apparent_temperature,precipitation,cloud_cover,wind_speed_10m,shortwave_radiation',
        'timezone': 'UTC'
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Error handling
    if response.status_code != 200 or 'hourly' not in data:
        raise ValueError("API request failed or invalid response")
    
    # Create hourly DF with timestamp_utc and weather features
    weather_df = pd.DataFrame({
        'timestamp_utc': pd.to_datetime(data['hourly']['time'], utc=True),
        'temperature_2m': data['hourly']['temperature_2m'],
        'apparent_temperature': data['hourly']['apparent_temperature'],
        'precipitation': data['hourly']['precipitation'],
        'cloud_cover': data['hourly']['cloud_cover'],
        'wind_speed_10m': data['hourly']['wind_speed_10m'],
        'shortwave_radiation': data['hourly']['shortwave_radiation']
    })
    
    # Interpolate to 15-min intervals
    weather_df = weather_df.set_index('timestamp_utc').resample('15T').interpolate(method='linear').ffill().reset_index()
    
    return weather_df


# def create_features(df):
#     df['year'] = df['timestamp_utc'].dt.year
#     df['month'] = df['timestamp_utc'].dt.month
#     df['day'] = df['timestamp_utc'].dt.day
#     df['hour'] = df['timestamp_utc'].dt.hour
#     df['minute'] = df['timestamp_utc'].dt.minute
#     df['dayofweek'] = df['timestamp_utc'].dt.dayofweek
#     df['dayofyear'] = df['timestamp_utc'].dt.dayofyear
#     df['weekofyear'] = df['timestamp_utc'].dt.isocalendar().week.astype(int)
#     df['quarter'] = df['timestamp_utc'].dt.quarter
#     df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
#     df['is_month_start'] = df['timestamp_utc'].dt.is_month_start.astype(int)
#     df['is_month_end'] = df['timestamp_utc'].dt.is_month_end.astype(int)
#     df['is_quarter_start'] = df['timestamp_utc'].dt.is_quarter_start.astype(int)
#     df['is_quarter_end'] = df['timestamp_utc'].dt.is_quarter_end.astype(int)
#     df['is_year_start'] = df['timestamp_utc'].dt.is_year_start.astype(int)
#     df['is_year_end'] = df['timestamp_utc'].dt.is_year_end.astype(int)

#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
#     df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
#     df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
#     df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
#     df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
#     df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
#     df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
#     df['weekofyear_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
#     df['weekofyear_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)

#     weather_cols = ['temperature_2m','apparent_temperature','precipitation','cloud_cover','wind_speed_10m','shortwave_radiation']
#     for col in weather_cols:
#         if col in df.columns:
#             df[f'{col}_lag0'] = df[col]
#             df[f'{col}_lag4'] = df[col].shift(4)

#     num_cols = df.select_dtypes(include=[np.number]).columns
#     for col in num_cols:
#         df[col].fillna(df[col].mean(), inplace=True)

#     return df



import pandas as pd
import numpy as np
import holidays

def create_features(df):
    df = df.copy()
    
    # Time features from timestamp_utc
    df['hour'] = df['timestamp_utc'].dt.hour
    df['dayofweek'] = df['timestamp_utc'].dt.dayofweek
    df['month'] = df['timestamp_utc'].dt.month
    df['dayofyear'] = df['timestamp_utc'].dt.dayofyear
    df['weekofyear'] = df['timestamp_utc'].dt.isocalendar().week
    df['quarter'] = df['timestamp_utc'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)  # 8 AM to 6 PM
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)  # 10 PM to 5 AM
    
    # Dutch holidays
    nl_holidays = holidays.Netherlands(years=df['timestamp_utc'].dt.year.unique())
    df['is_holiday'] = df['timestamp_utc'].dt.date.isin(nl_holidays).astype(int)
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # Weather features
    weather_cols = ['temperature_2m', 'apparent_temperature', 'precipitation', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation']
    for col in weather_cols:
        if col in df.columns:
            # Rolling statistics (1h, 2h)
            df[f'{col}_roll_mean_4'] = df[col].rolling(window=4, min_periods=1).mean()  # 1h
            df[f'{col}_roll_std_8'] = df[col].rolling(window=8, min_periods=1).std()   # 2h
            # Differences (1h, 3h)
            df[f'{col}_diff_4'] = df[col].diff(4)   # 1h
            df[f'{col}_diff_12'] = df[col].diff(12)  # 3h
            # Polynomial terms for key variables
            if col in ['temperature_2m', 'shortwave_radiation']:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_sqrt'] = np.sqrt(np.maximum(df[col], 0))  # Avoid negative sqrt
    
    # Weather interactions
    if all(col in df.columns for col in ['temperature_2m', 'shortwave_radiation']):
        df['temp_solar_interaction'] = df['temperature_2m'] * df['shortwave_radiation']
        df['temp_solar_ratio'] = df['temperature_2m'] / (df['shortwave_radiation'] + 1e-5)
    if all(col in df.columns for col in ['precipitation', 'cloud_cover']):
        df['precip_cloud_ratio'] = df['precipitation'] / (df['cloud_cover'] + 1e-5)
    if all(col in df.columns for col in ['temperature_2m', 'apparent_temperature']):
        df['temp_ratio'] = df['apparent_temperature'] / (df['temperature_2m'] + 1e-5)
    
    # Time-weather interactions
    if 'temperature_2m' in df.columns:
        df['hour_x_temp'] = df['hour'] * df['temperature_2m']
        df['dayofweek_x_temp'] = df['dayofweek'] * df['temperature_2m']
        df['is_business_hour_x_temp'] = df['is_business_hour'] * df['temperature_2m']
    if 'shortwave_radiation' in df.columns:
        df['hour_x_solar'] = df['hour'] * df['shortwave_radiation']
        df['is_night_x_solar'] = df['is_night'] * df['shortwave_radiation']
    
    # Seasonal category
    df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], labels=['winter', 'spring', 'summer', 'autumn'])
    df = pd.get_dummies(df, columns=['season'], prefix='season', drop_first=True)
    
    # Fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if not df[col].mode().empty:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


start_date = df['timestamp_utc'].min().floor('D')
end_date = df['timestamp_utc'].max().ceil('D')
weather_full = fetch_weather(start_date, end_date)
df = df.merge(weather_full, on='timestamp_utc', how='left')
df = create_features(df)
df.drop(columns=['timestamp_utc'], inplace=True)


df.head()


df.shape


X = df.drop(columns=['net_load_kwh'])
y = df['net_load_kwh']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


X_train.shape,X_test.shape


model = CatBoostRegressor(iterations=1000,learning_rate=0.5,depth=6,loss_function='RMSE',random_seed=42,verbose=200)
model.fit(X_train, y_train)


def compute_metrics(y_true, y_pred, mean_target):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    nmae = (mae / mean_target) * 100
    nrmse = (rmse / mean_target) * 100
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Explained Variance': evs,
        'MAPE (%)': mape,
        'NMAE (%)': nmae,
        'NRMSE (%)': nrmse
    }

mean_target = y_test.mean()
y_pred_test = model.predict(X_test)
metrics = compute_metrics(y_test, y_pred_test, mean_target)
print(tabulate([[m, f"{v:.4f}"] for m, v in metrics.items()], headers=["Metric", "Value"], tablefmt="grid"))



test_df=pd.read_csv("/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv")


test_df.head()


test_df.shape


test_df.isnull().sum()


Id=test_df.row_id


test_df['timestamp_utc'] = pd.to_datetime(test_df['timestamp_utc'], utc=True)


start_date = test_df['timestamp_utc'].min().floor('D')
end_date = test_df['timestamp_utc'].max().ceil('D')
weather_full = fetch_weather(start_date, end_date)
test_df = test_df.merge(weather_full, on='timestamp_utc', how='left')
test_df = create_features(test_df)
test_df.drop(columns=['timestamp_utc',"row_id"], inplace=True)


y_pred=model.predict(test_df)
submission = pd.DataFrame({'row_id': Id, 'net_load_kwh': y_pred})
submission.to_csv('submission.csv', index=False)


df=pd.read_csv("/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv")


df.head()


df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)


def fetch_weather(start_date, end_date, lat=52.3676, lon=4.9041):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'hourly': 'temperature_2m,apparent_temperature,precipitation,cloud_cover,wind_speed_10m,shortwave_radiation',
        'timezone': 'UTC'
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    weather_df = pd.DataFrame({
        'timestamp_utc': pd.to_datetime(data['hourly']['time'], utc=True),
        'temperature_2m': data['hourly']['temperature_2m'],
        'apparent_temperature': data['hourly']['apparent_temperature'],
        'precipitation': data['hourly']['precipitation'],
        'cloud_cover': data['hourly']['cloud_cover'],
        'wind_speed_10m': data['hourly']['wind_speed_10m'],
        'shortwave_radiation': data['hourly']['shortwave_radiation']
    })
    
    weather_df.set_index('timestamp_utc', inplace=True)
    weather_15min = weather_df.resample('15T').interpolate(method='linear').ffill().reset_index()
    
    return weather_15min

weather_df = fetch_weather(df['timestamp_utc'].min(), df['timestamp_utc'].max())
df = df.merge(weather_df, on='timestamp_utc', how='left')


def add_time_features(new_df, timestamp_col='timestamp_utc'):
    new_df = new_df.copy()
    new_df['hour'] = new_df[timestamp_col].dt.hour
    new_df['minute'] = new_df[timestamp_col].dt.minute
    new_df['dayofweek'] = new_df[timestamp_col].dt.dayofweek
    new_df['is_weekend'] = new_df['dayofweek'].isin([5, 6]).astype(int)
    new_df['month'] = new_df[timestamp_col].dt.month
    new_df['quarter'] = new_df[timestamp_col].dt.quarter
    new_df['hour_sin'] = np.sin(2 * np.pi * new_df['hour'] / 24)
    new_df['hour_cos'] = np.cos(2 * np.pi * new_df['hour'] / 24)
    new_df['dow_sin'] = np.sin(2 * np.pi * new_df['dayofweek'] / 7)
    new_df['dow_cos'] = np.cos(2 * np.pi * new_df['dayofweek'] / 7)
    return new_df

df=add_time_features(df,"timestamp_utc")


def add_weather_features(new_df, weather_cols=['temperature_2m','apparent_temperature','precipitation','cloud_cover','wind_speed_10m','shortwave_radiation']):
    new_df = new_df.copy()
    windows = [12, 24, 48, 96]
    lags = [1, 4, 12]
    
    for col in weather_cols:
        for w in windows:
            new_df[f'{col}_rollmean_{w}'] = new_df[col].rolling(w, min_periods=1).mean()
            new_df[f'{col}_rollstd_{w}'] = new_df[col].rolling(w, min_periods=1).std()
        for lag in lags:
            new_df[f'{col}_lag_{lag}'] = new_df[col].shift(lag)
    
    new_df['temp_x_cloud'] = new_df['temperature_2m'] * new_df['cloud_cover']
    new_df['temp_x_wind'] = new_df['temperature_2m'] * new_df['wind_speed_10m']
    
    return new_df

df = add_weather_features(df)
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


df.drop(columns=["timestamp_utc"],axis=1,inplace=True)


df.head()


df.shape


target_col = 'net_load_kwh'
feature_cols = [col for col in df.columns if col not in ['timestamp_utc', target_col]]
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


X_train.shape


model = CatBoostRegressor(iterations=2000,learning_rate=0.1,depth=6,loss_function='MAE',eval_metric='MAE',random_seed=42,verbose=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


def nrmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (y_true.max() - y_true.min())

def nmae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / (y_true.max() - y_true.min())

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

print(f"NRMSE: {nrmse(y_test, y_pred):.4f}")
print(f"NMAE: {nmae(y_test, y_pred):.4f}")
print(f"MAPE: {mape(y_test, y_pred):.2f}%")
print(f"RMSE: {rmse(y_test, y_pred):.4f}")
print(f"MAE: {mae(y_test, y_pred):.4f}")
print(f"R²: {r2(y_test, y_pred):.4f}")


test_df=pd.read_csv("/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv")


Id=test_df.row_id


test_df['timestamp_utc'] = pd.to_datetime(test_df['timestamp_utc'], utc=True)

weather_df = fetch_weather(test_df['timestamp_utc'].min(), test_df['timestamp_utc'].max())
test_df = test_df.merge(weather_df, on='timestamp_utc', how='left')
test_df=add_time_features(test_df,"timestamp_utc")

test_df = add_weather_features(test_df)
numeric_cols = test_df.select_dtypes(include=np.number).columns
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].mean())


test_df.head()


test_df.drop(columns=["timestamp_utc","row_id"],axis=1,inplace=True)
test_df.isnull().sum()


y_pred=model.predict(test_df)
submission = pd.DataFrame({'row_id': Id, 'net_load_kwh': y_pred})
submission.to_csv('new_submission.csv', index=False)
submission.head()


import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.003,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)

def nrmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (y_true.max() - y_true.min())

def nmae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / (y_true.max() - y_true.min())

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

print(f"NRMSE: {nrmse(y_test, y_pred):.4f}")
print(f"NMAE: {nmae(y_test, y_pred):.4f}")
print(f"MAPE: {mape(y_test, y_pred):.2f}%")
print(f"RMSE: {rmse(y_test, y_pred):.4f}")
print(f"MAE: {mae(y_test, y_pred):.4f}")
print(f"R²: {r2(y_test, y_pred):.4f}")


y_pred=lgb_model.predict(test_df)
submission = pd.DataFrame({'row_id': Id, 'net_load_kwh': y_pred})
submission.to_csv('lgb_submission.csv', index=False)
submission.head()


train = pd.read_csv('/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv')
train['timestamp_utc'] = pd.to_datetime(train['timestamp_utc'])


import pandas as pd
import requests
from scipy import interpolate

def fetch_weather(start_date, end_date, lat=52.37, lon=4.89):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_str}&end_date={end_str}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,"
        f"shortwave_radiation,wind_speed_10m,cloud_cover"
        f"&timezone=UTC"
    )

    response = requests.get(url)
    data = response.json()
    if 'hourly' not in data:
        return pd.DataFrame()

    hourly_times = pd.to_datetime(data['hourly']['time'])
    hourly_df = pd.DataFrame({
        'timestamp_utc': hourly_times,
        'temp_2m': data['hourly']['temperature_2m'],
        'humidity_2m': data['hourly']['relative_humidity_2m'],
        'precip': data['hourly']['precipitation'],
        'solar_rad': data['hourly']['shortwave_radiation'],
        'wind_speed_10m': data['hourly']['wind_speed_10m'],
        'cloud_cover': data['hourly']['cloud_cover']
    })

    full_times = pd.date_range(start=hourly_times[0], end=hourly_times[-1], freq='15min')
    weather_15min = pd.DataFrame({'timestamp_utc': full_times})

    for col in hourly_df.columns[1:]:
        interp_func = interpolate.interp1d(
            hourly_times.astype('int64'), 
            hourly_df[col], 
            kind='linear', 
            fill_value='extrapolate'
        )
        weather_15min[col] = interp_func(full_times.astype('int64'))

    return weather_15min



split_idx = int(0.8 * len(train))
train_data = train.iloc[:split_idx].copy()
val_data = train.iloc[split_idx:].copy()
print(f"Train: {len(train_data)}, Val: {len(val_data)}")


# Fetch for train+val
weather = fetch_weather(train['timestamp_utc'].min(), train['timestamp_utc'].max())
train_data = train_data.merge(weather, on='timestamp_utc', how='left')
val_data = val_data.merge(weather, on='timestamp_utc', how='left')
val_data.fillna(method='ffill', inplace=True)


train_data.head()


val_data.isnull().sum()


import numpy as np

# Define weather columns
weather_cols = ['temp_2m', 'humidity_2m', 'precip', 'solar_rad', 'wind_speed_10m', 'cloud_cover']

# Add weather lags and rolling means
def add_weather_features(df):
    for col in weather_cols:
        df[f'{col}_lag1'] = df[col].shift(1)  # 15-min lag
        df[f'{col}_rolling_mean_4'] = df[col].rolling(window=4).mean()  # 1-hour rolling mean
    return df

# Add calendar features
def add_calendar_features(df):
    df['hour'] = df['timestamp_utc'].dt.hour
    df['dayofweek'] = df['timestamp_utc'].dt.dayofweek
    df['month'] = df['timestamp_utc'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

# Apply to all DataFrames
train_data = add_weather_features(train_data)
val_data = add_weather_features(val_data)


train_data = add_calendar_features(train_data)
val_data = add_calendar_features(val_data)


# Handle NaNs from lags/rolling
train_data.fillna(method='ffill', inplace=True)
val_data.fillna(method='ffill', inplace=True)


# Verify features
print("Train features:", train_data.columns.tolist())
print("Val features:", val_data.columns.tolist())



# Define feature columns (exclude timestamp and target)
feature_cols = [col for col in train_data.columns if col not in ['timestamp_utc', 'net_load_kwh']]
# Prepare train and validation data
X_train = train_data[feature_cols].values
y_train = train_data['net_load_kwh'].values
X_val = val_data[feature_cols].values
y_val = val_data['net_load_kwh'].values


import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Train XGBoost (single-step for simplicity)
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Predict on validation (single-step for now)
val_preds = model.predict(X_val)

# Evaluate
def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = rmse / np.std(y_true)
    nmae = mae / np.mean(np.abs(y_true))
    return nrmse, nmae

nrmse, nmae = calc_metrics(y_val, val_preds)
print(f"NRMSE: {nrmse*100:.2f}%, NMAE: {nmae*100:.2f}%")


test_df=pd.read_csv("/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv")
test_df.head()


Id=test_df.row_id
test_df.drop(columns=["row_id"],axis=1,inplace=True)


test_df['timestamp_utc'] = pd.to_datetime(test_df['timestamp_utc'])
test_weather = fetch_weather(test_df['timestamp_utc'].min(), test_df['timestamp_utc'].max())
test_df = test_df.merge(test_weather, on='timestamp_utc', how='left')
test_df = add_weather_features(test_df)
test_df = add_calendar_features(test_df)
test_df.fillna(method='ffill', inplace=True)
test_df=test_df[feature_cols].values
# Predict on test
test_preds = model.predict(test_df)
# Create submission
sub = pd.DataFrame({'row_id': Id, 'net_load_kwh': test_preds})
sub.to_csv('xgb_submission.csv', index=False)
print("Submission saved as submission.csv")


submission.head()




