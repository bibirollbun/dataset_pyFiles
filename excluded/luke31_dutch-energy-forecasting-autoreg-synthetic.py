import numpy as np
import pandas as pd
import polars as pl
from datetime import timedelta
import requests
from datetime import datetime, date
import seaborn as sns
import matplotlib.pyplot as plt

# File paths
train_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/train_expanded.csv'
test_path = '/kaggle/input/dutch-energy-supplier-load-forecasting-challenge/test_new.csv'

# Load data
train_df = pl.read_csv(train_path, schema_overrides={"timestamp_utc": pl.Datetime("us")} )
test_df = pl.read_csv(test_path, schema_overrides={"timestamp_utc": pl.Datetime("us")} )

print(f"✓ Train data loaded: {train_df.shape}")
print(f"✓ Test data loaded: {test_df.shape}")
print(f"  Train columns: {train_df.columns}")
print(f"  Test columns: {test_df.columns}")


# combine dfs
test_df_no_rowid = test_df.select("timestamp_utc").with_columns(
    pl.lit(None).alias("net_load_kwh")
)
df_combined = pl.concat([train_df, test_df_no_rowid])

df_combined.to_pandas().plot(x="timestamp_utc", y="net_load_kwh")
plt.show()


# Netherlands coordinates (approximate center)
def fetch_weather_data(start_date, end_date, latitude=52.1326, longitude=5.2913):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        # "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,rain,snowfall,cloud_cover,wind_speed_10m,wind_direction_10m,wind_speed_100m,shortwave_radiation,direct_radiation,diffuse_radiation",
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "snow_depth", "weather_code", "pressure_msl", "surface_pressure",
            "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "et0_fao_evapotranspiration", "vapour_pressure_deficit",
            "wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
            "wind_gusts_10m", "soil_temperature_0_to_7cm",
            "shortwave_radiation", "direct_radiation", "diffuse_radiation", "global_tilted_irradiance",
            "direct_normal_irradiance", "terrestrial_radiation"
        ],
        "timezone": "UTC"
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    weather_df = pl.DataFrame({
        "timestamp_utc": pl.Series(data["hourly"]["time"]).str.strptime(pl.Datetime),
        "temperature": data["hourly"]["temperature_2m"],
        "apparent_temp": data["hourly"]["apparent_temperature"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "dew_point": data["hourly"]["dew_point_2m"],
        "precipitation": data["hourly"]["precipitation"],
        "rain": data["hourly"]["rain"],
        "snowfall": data["hourly"]["snowfall"],
        "snow_depth": data["hourly"]["snow_depth"],
        "weather_code": data["hourly"]["weather_code"],
        "pressure_msl": data["hourly"]["pressure_msl"],
        "surface_pressure": data["hourly"]["surface_pressure"],
        "cloud_cover": data["hourly"]["cloud_cover"],
        "cloud_cover_low": data["hourly"]["cloud_cover_low"],
        "cloud_cover_mid": data["hourly"]["cloud_cover_mid"],
        "cloud_cover_high": data["hourly"]["cloud_cover_high"],
        "evapotranspiration": data["hourly"]["et0_fao_evapotranspiration"],
        "vapour_pressure_deficit": data["hourly"]["vapour_pressure_deficit"],
        "wind_speed_10m": data["hourly"]["wind_speed_10m"],
        "wind_direction": data["hourly"]["wind_direction_10m"],
        "wind_speed_100m": data["hourly"]["wind_speed_100m"],
        "wind_gusts": data["hourly"]["wind_gusts_10m"],
        "soil_temp": data["hourly"]["soil_temperature_0_to_7cm"],
        "shortwave_radiation": data["hourly"]["shortwave_radiation"],
        "direct_radiation": data["hourly"]["direct_radiation"],
        "diffuse_radiation": data["hourly"]["diffuse_radiation"],
        "direct_normal_irradiance": data["hourly"]["direct_normal_irradiance"],
        "global_tilted_irradiance": data["hourly"]["global_tilted_irradiance"],
        "terrestrial_radiation": data["hourly"]["terrestrial_radiation"],
    })

    weather_df = (
        weather_df.upsample(time_column="timestamp_utc", every="15m")
            .interpolate()
            .fill_null(strategy="forward")
    )
    
    return weather_df

# CHANGED: Use full 2024 year from 2024-01-01 until 2025-09-25
# start = (df_combined['timestamp_utc'].min() - timedelta(days=14)).date()
# end = (df_combined['timestamp_utc'].max() + timedelta(days=1)).date()
start = date(2024,1,1)
end = (df_combined['timestamp_utc'].max() + timedelta(days=1)).date()

weather_df = fetch_weather_data(str(start), str(end))
weather_df


# Calculate trend to be used for features
train_trend_df = train_df.with_columns(
    (pl.col("timestamp_utc").cast(pl.Int64) / 1e9).alias("timestamp_numeric")
)
a, b = np.polyfit(train_trend_df["timestamp_numeric"], train_trend_df["net_load_kwh"], 1)
# coeffs = np.polyfit(train_trend_df["timestamp_numeric"], train_trend_df["net_load_kwh"], 2)
# poly=np.poly1d(coeffs)
train_trend_df = train_trend_df.with_columns(
    (a * pl.col("timestamp_numeric") + b).alias("net_load_kwh_trend")
    # pl.Series("trend_poly2", poly(train_trend_df["timestamp_numeric"].to_numpy())).alias("net_load_kwh_trend_poly")
)
train_trend_df.to_pandas().plot(x="timestamp_utc", y=["net_load_kwh","net_load_kwh_trend"])
plt.show()


# 288 = 3 days in 15-min intervals
# max_lag_steps 672
def make_features(df, weather_df, add_load_lag=True):
    # weather
    df = df.join(weather_df, on="timestamp_utc", how="left")
    
    # Time features
    df = df.with_columns([
        (pl.col("timestamp_utc").cast(pl.Int64) / 1e9).alias("timestamp_numeric"),
        pl.col("timestamp_utc").dt.hour().alias("hour"),
        (pl.col("timestamp_utc").dt.weekday() - 1).alias("dayofweek"), # polars weekday is iso format 1-7, remove 1 to get 0-6
        pl.col("timestamp_utc").dt.quarter().alias("quarter"),
        pl.col("timestamp_utc").dt.month().alias("month"),
        pl.col("timestamp_utc").dt.day().alias("day"),
        pl.col("timestamp_utc").dt.ordinal_day().alias("dayofyear"),
        pl.col("timestamp_utc").dt.week().alias("weekofyear"),
        (pl.col("timestamp_utc").dt.hour() * 4 + (pl.col("timestamp_utc").dt.minute() // 15)).alias("quarter_hour"),
    ])
    
    # Cyclical encoding
    df = df.with_columns([
        (2 * np.pi * pl.col("hour") / 24).sin().alias("hour_sin"),
        (2 * np.pi * pl.col("hour") / 24).cos().alias("hour_cos"),
        (2 * np.pi * pl.col("dayofweek") / 7).sin().alias("dayofweek_sin"),
        (2 * np.pi * pl.col("dayofweek") / 7).cos().alias("dayofweek_cos"),
        (2 * np.pi * pl.col("dayofyear") / 365).sin().alias("dayofyear_sin"),
        (2 * np.pi * pl.col("dayofyear") / 365).cos().alias("dayofyear_cos"),
        (2 * np.pi * pl.col("quarter_hour") / 96).sin().alias("quarter_hour_sin"),
        (2 * np.pi * pl.col("quarter_hour") / 96).cos().alias("quarter_hour_cos"),
    ])
    
    # Weekend / workday flags
    df = df.with_columns([
        # Basic day-based flags
        (pl.col("dayofweek") >= 5).cast(pl.Int8).alias("is_weekend"),
        (pl.col("dayofweek") < 5).cast(pl.Int8).alias("is_weekday"),
        (pl.col("dayofweek") == 0).cast(pl.Int8).alias("is_monday"),
        (pl.col("dayofweek") == 4).cast(pl.Int8).alias("is_friday"),
    
        # Time-based flags
        ((pl.col("hour") >= 22) | (pl.col("hour") <= 5)).cast(pl.Int8).alias("is_night"),
        ((pl.col("hour") >= 6) & (pl.col("hour") <= 11)).cast(pl.Int8).alias("is_morning"),
        ((pl.col("hour") >= 12) & (pl.col("hour") <= 17)).cast(pl.Int8).alias("is_afternoon"),
        ((pl.col("hour") >= 18) & (pl.col("hour") <= 21)).cast(pl.Int8).alias("is_evening"),
    
        # Business and peak hours
        ((pl.col("hour") >= 9) & (pl.col("hour") <= 17) & (pl.col("dayofweek") < 5))
            .cast(pl.Int8)
            .alias("is_business_hour"),
        ((pl.col("hour") >= 7) & (pl.col("hour") <= 9)).cast(pl.Int8).alias("is_peak_morning"),
        ((pl.col("hour") >= 17) & (pl.col("hour") <= 20)).cast(pl.Int8).alias("is_peak_evening"),
    
        # Optional: combined flag similar to your 'is_workday' idea
        ((pl.col("dayofweek") < 5) & (pl.col("hour") >= 8) & (pl.col("hour") < 18))
            .cast(pl.Int8)
            .alias("is_workday"),
    ])

    # trend
    if add_load_lag:
        df = df.with_columns(
            (a * (pl.col("timestamp_utc").cast(pl.Int64) / 1e9) + b).alias("net_load_kwh_trend")
        )
    
        # lag-features
        day_step = 96
        lag_load_steps = [
            int(day_step * 3),       # 3 days
            int(day_step * 3.25),    # 3.25 days
            int(day_step * 3.5),     # 3.5 days
            int(day_step * 4),       # 4 days
            int(day_step * 5),       # 5 days
            int(day_step * 6),       # 6 days
            int(day_step * 7),       # 7 days
        ]
    
        df = df.with_columns([
                # raw lag values
                pl.col("net_load_kwh").shift(lag).alias(f"load_lag_{lag}")
                for lag in lag_load_steps
            ] + [
                # rolling means over the past 1 day before each lag point
                pl.col("net_load_kwh").shift(lag).rolling_mean(day_step).alias(f"load_lag_{lag}_mean1d")
                for lag in lag_load_steps
            ]
        )

    # Weather
    df = df.with_columns([
        (pl.col("temperature") * pl.col("temperature")).alias("temperature_squared"),
        (pl.col("shortwave_radiation") * pl.col("shortwave_radiation")).alias("shortwave_radiation_squared"),
        (pl.col("global_tilted_irradiance") * pl.col("global_tilted_irradiance")).alias("global_tilted_irradiance_squared"),
    ])
    
    # Weather interactions
    df = df.with_columns([
        (pl.col("temperature") * pl.col("humidity")).alias("temp_humidity"),
        (pl.col("wind_speed_10m") * pl.col("temperature")).alias("wind_temp"),
        (pl.col("shortwave_radiation") * (100 - pl.col("cloud_cover"))).alias("radiation_cloud"),
        (pl.col("temperature") - (pl.col("wind_speed_10m") * 0.7)).alias("feels_like"),
    ])

    # Lagged waether features (respecting 3-day latency = 288 steps)
    lag_weather_steps = [288, 288 + 96, 288 + 192]  # 3 days, 3.5 days, 4 days
    lag_weather_cols = ["temperature", "humidity", "wind_speed_10m", "shortwave_radiation"]
    df = df.with_columns([
        pl.col(col).shift(lag).alias(f"{col}_lag{lag}")
        for col in lag_weather_cols
        for lag in lag_weather_steps
    ])

    # Rolling weather stats (must respect latency)
    rolling_weather_cols = ["temperature", "humidity", "wind_speed_10m"]
    df = df.with_columns([
        pl.col(col).shift(288).rolling_mean(window_size=96).alias(f"{col}_roll24h")
        for col in rolling_weather_cols
    ] + [
        pl.col(col).shift(288).rolling_mean(window_size=192).alias(f"{col}_roll48h")
        for col in rolling_weather_cols
    ])

    
    # fill_null backwards for lag-features
    lag_cols = []
    if add_load_lag:
        lag_cols += (
            [f"load_lag_{lag}" for lag in lag_load_steps] + 
            [f"load_lag_{lag}_mean1d" for lag in lag_load_steps] +
            [f"{col}_lag{lag}" for col in lag_weather_cols for lag in lag_weather_steps]
        )
    lag_cols += (
        [f"{col}_roll24h" for col in rolling_weather_cols] +
        [f"{col}_roll48h" for col in rolling_weather_cols]
    )
    
    df = df.with_columns([
        pl.col(col)
        # .fill_null(strategy="forward") # will be handled by autoregression
        .fill_null(strategy="backward")
        .alias(col)
        for col in lag_cols
    ])
    
    return df


synthetic_input_df = make_features(train_df, weather_df, add_load_lag=False)
synthetic_input_df


import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

df = synthetic_input_df
synt_train_df = df.filter(pl.col("timestamp_utc") < pl.datetime(2025, 8, 15))
synt_val_df = df.filter(pl.col("timestamp_utc") >= pl.datetime(2025, 8, 15))

feature_cols = [c for c in df.columns if c not in ["timestamp_utc", "net_load_kwh"]]
X_train = synt_train_df.select(feature_cols).to_pandas()
y_train = synt_train_df["net_load_kwh"].to_pandas()

X_val = synt_val_df.select(feature_cols).to_pandas()
y_val = synt_val_df["net_load_kwh"].to_pandas()

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=48,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print("MAE:", mean_absolute_error(y_val, y_pred))

residuals = y_val - y_pred
print(residuals)


synthetic_data_feature_cols = [
    "shortwave_radiation", 
    "global_tilted_irradiance",
    "quarter_hour_sin",
    "feels_like",
    "hour_sin", 
    "hour_cos",
    "quarter_hour",
    "apparent_temp",
    "evapotranspiration",
    "shortwave_radiation_squared",
    "wind_speed_100m",
    "wind_speed_10m",
    "soil_temp",
    "direct_radiation",
    "temperature", 
    "humidity"
]
residual_features = X_val[synthetic_data_feature_cols]
Z = np.column_stack([residuals, residual_features])
mu = Z.mean(axis=0)
Sigma = np.cov(Z, rowvar=False)
print(f"{mu.shape=}, {Sigma.shape=}")


# --- 1. Select the target weather period ---
weather_future = weather_df.filter(
    (pl.col("timestamp_utc") >= pl.datetime(2024,1,1)) &
    (pl.col("timestamp_utc") < pl.datetime(2025,5,8))
)

# --- 2. Add the same time-based features as in training ---
weather_future = make_features(
    pl.DataFrame({"timestamp_utc": weather_future["timestamp_utc"]}),
    weather_df, 
    add_load_lag=False
)

# --- 3. Prepare features for model prediction ---
X_future = weather_future.select(feature_cols).to_pandas()

# --- 4. Get base model prediction (deterministic component) ---
y_future_pred = model.predict(X_future)

# --- 5. Conditional residual sampling ---
# Assuming you have mean vector (mu) and covariance matrix (Sigma)
# with order: [residual, temperature, humidity, wind_speed_10m, hour_sin, hour_cos]

mu_r = mu[0]
mu_x = mu[1:]
Sigma_rr = Sigma[0, 0]
Sigma_rx = Sigma[0, 1:]
Sigma_xx = Sigma[1:, 1:]

# Invert Sigma_xx safely (regularize if necessary)
Sigma_xx_inv = np.linalg.inv(Sigma_xx + 1e-6 * np.eye(Sigma_xx.shape[0]))

# --- 6. Compute conditional residual distribution ---
Xx = X_future[synthetic_data_feature_cols].to_numpy()

# (Xx - mu_x) should have shape [N, D]; multiply with Sigma_rx @ Sigma_xx_inv @ (x - mu_x)
conditional_mean = mu_r + (Xx - mu_x) @ (Sigma_xx_inv @ Sigma_rx.T)
conditional_var = float(Sigma_rr - Sigma_rx @ Sigma_xx_inv @ Sigma_rx.T)
conditional_std = np.sqrt(max(conditional_var, 1e-8))

# --- 7. Sample residuals per observation ---
residual_samples = np.random.normal(conditional_mean, conditional_std)

# --- 8. Combine deterministic + stochastic components ---
synthetic_load = y_future_pred + residual_samples

# --- 9. Wrap into Polars DataFrame ---
synthetic_df = weather_future.with_columns(
    pl.Series("synthetic_net_load_kwh", synthetic_load)
)
synthetic_df = synthetic_df.with_columns(
    pl.col("synthetic_net_load_kwh")
    .rolling_mean(window_size=3)
    .fill_null(strategy="backward")
    .fill_null(strategy="forward")
    .alias("synthetic_net_load_kwh_rolling_mean_3_steps")
)
synthetic_df.to_pandas().plot(x="timestamp_utc", y=["synthetic_net_load_kwh","synthetic_net_load_kwh_rolling_mean_3_steps"])
synthetic_df = synthetic_df.with_columns(
 pl.col("synthetic_net_load_kwh").alias("net_load_kwh")
).select(["timestamp_utc", "net_load_kwh"])
print(synthetic_df.describe())
print(synthetic_df)


train_combined_df = pl.concat([synthetic_df, train_df])

train_combined_df.to_pandas().plot(x="timestamp_utc", y=["net_load_kwh"])
plt.axvline(pd.Timestamp("2025-05-08"), color="red", linestyle="--", label="2025-05-08")
plt.legend()
plt.show()

train_features = make_features(train_combined_df, weather_df)
train_features


Xy_train = train_features

X_train = Xy_train.select(pl.exclude("net_load_kwh","timestamp_utc"))
y_train = Xy_train.select(pl.col("net_load_kwh"))
print(X_train)
print(y_train)


# Train
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from typing import Protocol, Any, List
from abc import abstractmethod


class ModelTrainerProtocol(Protocol):
    @abstractmethod
    def train_model(self, X_tr, y_tr, X_val=None, y_val=None) -> Any:
        ...

    @abstractmethod
    def predict(self, X) -> Any:
        ...

    @abstractmethod
    def plot_feature_importance(self, models: List["ModelTrainerProtocol"]) -> None:
        ...
        
class GBMTrainer:
    def __init__(self) -> None:
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 24,
            'max_depth': 5,
            'learning_rate': 0.05,
            'num_boost_round': 200, 
            'feature_fraction': 0.6,
            'bagging_fraction': 0.65,
            'bagging_freq': 1,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'min_child_samples': 200,
            'min_gain_to_split': 0.2,
            'verbose': -1,
            'n_jobs': -1
        }
        self.model = None
        self.trend_model = None
        self.X_scaler = RobustScaler()
        self.y_scaler = RobustScaler()
    
    def train_model(self, X_tr, y_tr, X_val=None, y_val=None, plot=True, log=True) -> Any:
        X_tr_scaled = pd.DataFrame(self.X_scaler.fit_transform(X_tr), columns=X_tr.columns)
        y_tr_scaled = self.y_scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
        
        train_data = lgb.Dataset(X_tr_scaled, label=y_tr_scaled)
        evals_result = {}  # to record eval results
        if X_val is not None and y_val is not None:
            X_val_scaled = pd.DataFrame(self.X_scaler.transform(X_val), columns=X_val.columns)
            y_val_scaled = self.y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
            
            val_data = lgb.Dataset(X_val_scaled, label=y_val_scaled, reference=train_data)
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'validation'],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=log), lgb.log_evaluation(100 if log else 2000), lgb.record_evaluation(evals_result)]
            )
        else:
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(100), lgb.record_evaluation(evals_result)]
            )
        
        # --- Plot learning curves ---
        if plot:
            metric = list(evals_result['train'].keys())[0]  # e.g. 'l2', 'rmse', 'mae'
            plt.figure(figsize=(8, 5))
            plt.plot(evals_result['train'][metric], label=f'Train {metric}')
            if 'validation' in evals_result:
                plt.plot(evals_result['validation'][metric], label=f'Validation {metric}')
            plt.xlabel('Iteration')
            plt.ylabel(metric)
            plt.title('LightGBM Learning Curve')
            plt.legend()
            plt.grid(True)
            plt.show()
        return evals_result

    def predict(self, X) -> Any:
        if not self.model:
            raise RuntimeError("model not trained")
        X_scaled = pd.DataFrame(self.X_scaler.transform(X), columns=X.columns)
        pred_scaled = self.model.predict(X_scaled, num_iteration=self.model.best_iteration)
        pred = self.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        return pred

    def plot_feature_importance(self, models: List[ModelTrainerProtocol]) -> None:
        all_importances = np.array([m.model.feature_importance(importance_type='gain') for m in models]) # gain/split
        avg_importances = all_importances.mean(axis=0)
        feature_importances = pl.DataFrame({
            "name": models[0].model.feature_name(),
            "importance": avg_importances
        })
        TOP = 20  # number of top features to plot
        
        # Normalize or sort importances
        importance_data = (
            feature_importances
            .sort("importance", descending=True)
            .head(TOP)
        )
        
        # Convert to pandas for seaborn plotting
        importance_pd = importance_data.to_pandas()
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=importance_pd,
            x="importance",
            y="name",
            ax=ax,
            orient="h",
            palette="viridis"
        )
        
        # Add text labels for percentages
        for i, patch in enumerate(ax.patches):
            width = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            perc = 100 * importance_pd["importance"].iloc[i] / importance_pd["importance"].sum()
            ax.text(width, y, f"{perc:.1f}%", va="center", ha="left", fontsize=9)
        
        plt.title(f"Top {TOP} features sorted by importance", fontsize=12)
        plt.tight_layout()
        plt.show()
        
def get_model_trainer():
    return GBMTrainer()



from typing import Dict, Tuple
from itertools import product

def grid_search_gbm(X_tr, y_tr, X_val, y_val, param_grid: Dict[str, List[Any]]) -> Tuple[GBMTrainer, Dict[str, Any]]:
    """
    Perform a simple grid search over LightGBM parameters.
    Returns the best GBMTrainer instance and its params.
    """
    best_rmse = float("inf")
    best_params = None
    best_model = None
    metric = "rmse"

    keys, values = zip(*param_grid.items())

    for combination in product(*values):
        params = dict(zip(keys, combination))
        trainer: ModelTrainerProtocol = get_model_trainer()
        trainer.params.update(params)
        evals = trainer.train_model(X_tr, y_tr, X_val, y_val, plot=False, log=False)
        final_validation_rmse = evals['validation'][metric][-1]
        if final_validation_rmse < best_rmse:
            best_rmse = final_validation_rmse
            best_params = params
            best_model = trainer
            print(f"✅ New best RMSE: {best_rmse:.4f}, params: {params}")

    print("\n=== Best parameters ===")
    print(best_params)
    print(f"Best RMSE: {best_rmse:.4f}, params: {params}")

    return best_model, best_params

param_grid = {
    # --- Core Learning Parameters ---
    "num_leaves": [16, 24, 48],       # 31 controls model complexity
    "max_depth": [5, 9],           # 7 deeper = more complex trees; -1 = no limit
    "learning_rate": [0.05],  # 0.01,, 0.1 lower = slower learning but often better generalization
    "num_boost_round": [200, 500],      # 500 number of boosting rounds

    # --- Sampling Parameters (regularization via randomness) ---
    "feature_fraction": [0.4, 0.6],  # 0.75, 0.9,  fraction of features per iteration
    "bagging_fraction": [0.4, 0.6],        # 0.8, fraction of data sampled
    "bagging_freq": [1],                  # , 5 frequency of bagging

    # --- Regularization ---
    "reg_alpha": [1.0, 2.0],          # 0.0, 0.5,  L1 penalty
    "reg_lambda": [1.0, 2.0],         # 0.0, 0.5,  L2 penalty

    # --- Tree Split / Leaf Control ---
    "min_child_samples": [50, 100],   # 100, 500 min data points in a leaf
    "min_gain_to_split": [0.2],  #  0.0, 0.1, min loss reduction to make a split
}

# get last fold only
tss = TimeSeriesSplit(n_splits=3)
for fold, (train_idx, val_idx) in enumerate(tss.split(X_train)):
    if fold == (tss.n_splits - 1):
        print(f"Using final fold: {fold + 1}/{tss.n_splits}")
        X_tr, X_val = X_train[train_idx].to_pandas(), X_train[val_idx].to_pandas()
        y_tr, y_val = y_train[train_idx].to_pandas().squeeze(), y_train[val_idx].to_pandas().squeeze()    
        break

best_params = {
    'num_leaves': 16, 
    'max_depth': 5, 
    'learning_rate': 0.05, 
    'num_boost_round': 200, 
    'feature_fraction': 0.4, 
    'bagging_fraction': 0.6, 
    'bagging_freq': 1, 
    'reg_alpha': 1.0, 
    'reg_lambda': 2.0, 
    'min_child_samples': 50, 
    'min_gain_to_split': 0.2
}
should_find_best_params = False
if should_find_best_params:
    print("Performing grid-search LightGBM model...")
    _, best_params = grid_search_gbm(X_tr, y_tr, X_val, y_val, param_grid)


print("Training LightGBM model...")
tss = TimeSeriesSplit(n_splits=3)

models = []
for fold, (train_idx, val_idx) in enumerate(tss.split(X_train)):
    print(f"Fold {fold + 1}/3 train:{len(train_idx)}, val:{len(val_idx)}")
    
    X_tr, X_val = X_train[train_idx].to_pandas(), X_train[val_idx].to_pandas()
    y_tr, y_val = y_train[train_idx].to_pandas().squeeze(), y_train[val_idx].to_pandas().squeeze()    
    
    model_trainer: ModelTrainerProtocol = get_model_trainer()
    model_trainer.train_model(X_tr, y_tr, X_val, y_val)
    model_trainer.params.update(best_params)
    models.append(model_trainer)

    # See metric discussion https://www.kaggle.com/competitions/dutch-energy-supplier-load-forecasting-challenge/discussion/610821
    # You should normalise both by the mean of the absolute target values, not the raw mean or the range.
    val_pred = model_trainer.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    mae = np.mean(np.abs(y_val - val_pred))
    mean_load = np.mean(np.abs(y_val)) # changed
    nrmse = rmse / mean_load * 100
    nmae = mae / mean_load * 100
    
    print(f"Fold {fold + 1} - NRMSE: {nrmse:.2f}%, NMAE: {nmae:.2f}%")
 

def last_pred(X, models):
    # predict only using most recent model
    model = models[-1]
    return model.predict(X)

def ensemble_pred(X, models):
    # Assign increasing weights, so later models get higher weight
    weights = np.arange(1, len(models) + 1)
    preds = np.array([model.predict(X) for model in models])
    
    # Weighted average
    weighted_avg = np.average(preds, axis=0, weights=weights) # scaling issue, inverse scaler is applied per model
    return weighted_avg

def predict(X, models):
    return last_pred(X, models) # most recent model performs best on leaderboard


models[0].plot_feature_importance(models)


# Plot evaluate

from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 7. MODEL EVALUATION
# ==========================================
print("\n7. EVALUATING MODELS")
print("-" * 40)


def plot_predictions(features, test_predictions, actual_values=None, title='Load Time Series (1 week sample)', feature_cols='feature'):
    one_week_interval = 672
    # Plot predictions with 1 week earlier value
    num_weeks = 3
    fig, axes = plt.subplots(num_weeks,1, figsize=(10,10))
    # Time series sample
    for i in range(0,num_weeks):
        ax = axes[i]
        range_start = one_week_interval * i
        range_end = range_start + one_week_interval
        ax.plot(features[range_start:range_end], alpha=0.6, label=feature_cols, linestyle=":")
        if actual_values is not None:
            ax.plot(actual_values[range_start:range_end], alpha=0.8, linewidth=2, markersize=3, color="blue", label="actual")
        ax.plot(test_predictions[range_start:range_end], alpha=0.8, linewidth=2, markersize=3, linestyle="--", color="red", label="predictions")
        ax.set_title(title)
        ax.set_ylabel('Net Load (kWh/15min)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-650, 650)
        ax.tick_params(axis='x', rotation=45)

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
val_pred = predict(X_val, models) # last_pred(X_val, models)
pred_tr = predict(X_tr, models)

plot_feature_cols =["load_lag_288", "diffuse_radiation", "net_load_kwh_trend"]
plot_feature_cols_index = [X_train.columns.index(name) for name in plot_feature_cols]
plot_predictions(X_tr.to_numpy()[:, plot_feature_cols_index], pred_tr, y_tr.to_numpy(), title=f'Load Time Series (1 week sample) train',feature_cols=plot_feature_cols)
plot_predictions(X_val.to_numpy()[:, plot_feature_cols_index], val_pred, y_val.to_numpy(), title=f'Load Time Series (1 week sample) val',feature_cols=plot_feature_cols)
evaluate_model(y_val, val_pred, "LGBM")


# Use ALL data for final model in list
model_trainer_final: ModelTrainerProtocol = get_model_trainer()
model_trainer_final.train_model(X_train.to_pandas(), y_train.to_pandas())
model_trainer_final.params.update(best_params)
models.append(model_trainer_final)

# Evaluate each model
pred_train_final = predict(X_train, models)

plot_feature_cols =["load_lag_288", "diffuse_radiation", "net_load_kwh_trend"]
plot_feature_cols_index = [X_train.columns.index(name) for name in plot_feature_cols]
val_size = int(len(X_train) * 0.2)
plot_predictions(X_train.to_numpy()[:-val_size, plot_feature_cols_index], pred_train_final[:-val_size], y_train.to_numpy()[:-val_size], title=f'Load Time Series (1 week sample) train',feature_cols=plot_feature_cols)
evaluate_model(y_train.to_numpy()[:-val_size], pred_train_final[:-val_size], "LGBM")


print("Generating predictions...")

# Vectorized predictions (multi-output in one step)
# test_predictions = model.predict(X_test, num_iteration=model.best_iteration)

max_lag = 96 * 7 # 672

df_combined_indexed = df_combined.with_row_index()

# Auto-regressive prediction
test_preds = []
X_test_list = []
for x_test_timestamp_utc in test_df["timestamp_utc"].to_list():
    # Find the row number corresponding to the target timestamp
    match_row_nr = df_combined_indexed.filter(pl.col("timestamp_utc") == x_test_timestamp_utc)["index"][0]
    
    # Get max_lag rows before it (include the match itself)
    start = max(match_row_nr - max_lag, 0)
    df_hist = df_combined_indexed.slice(start, match_row_nr - start + 1).drop("index")
    
    combined_features = make_features(df_hist, weather_df)
    
    x_test = combined_features.filter(pl.col("timestamp_utc") == x_test_timestamp_utc).select(
        pl.exclude("net_load_kwh","timestamp_utc")).to_pandas()
    X_test_list.append(x_test)

    pred = predict(x_test, models) # .reshape(1,-1)
    test_preds.append(pred)

    # update net_load_kwh with prediction, it will be used to build new lag-features
    df_combined_indexed = df_combined_indexed.with_columns(
        pl.when(
            (pl.col("timestamp_utc") == x_test_timestamp_utc) &
            (pl.col('net_load_kwh').is_null()))
        .then(pl.lit(pred))
        .otherwise(pl.col('net_load_kwh')).alias("net_load_kwh")
    )

X_test = np.concatenate(X_test_list)
test_predictions = np.concatenate(test_preds)


submission_preparation = pl.DataFrame({
    'row_id': test_df['row_id'],
    'timestamp_utc': test_df['timestamp_utc'],
    'timestamp_numeric': test_df['timestamp_utc'].cast(pl.Int64) / 1e9,
    'original_predicted_net_load_kwh': test_predictions
})
timestamp_numeric_baseline = submission_preparation.select("timestamp_numeric").min().item()
# submission_preparation = submission_preparation.with_columns(
#     (a * (pl.col("timestamp_numeric") - timestamp_numeric_baseline)).alias("trend_adjustment")
# )
# submission_preparation = submission_preparation.with_columns(
#     (pl.col("original_predicted_net_load_kwh") + pl.col("trend_adjustment") - 10).alias("predicted_net_load_kwh") ## TODO Remove hard-coded fix
# )
submission_preparation = submission_preparation.with_columns(
    (pl.col("original_predicted_net_load_kwh")).alias("predicted_net_load_kwh") ## Just use original prediction which includes trend
)
submission_preparation


# Create submission
submission = submission_preparation.select(["row_id","predicted_net_load_kwh"])

submission.write_csv('submission.csv')
print("Submission saved to 'submission.csv'")

plot_predictions(X_test[:, plot_feature_cols_index], test_predictions, feature_cols=plot_feature_cols)


# plot final trend combined with training
submission_df_no_rowid = submission_preparation.select("timestamp_utc","predicted_net_load_kwh").with_columns(
    pl.col("predicted_net_load_kwh").alias("net_load_kwh")
).drop("predicted_net_load_kwh")
df_combined_submission = pl.concat([train_combined_df, submission_df_no_rowid])

df_combined_submission_trend = df_combined_submission.with_columns(
    (pl.col("timestamp_utc").cast(pl.Int64) / 1e9).alias("timestamp_numeric")
)
a_train_submission, b_train_submission = np.polyfit(df_combined_submission_trend["timestamp_numeric"], df_combined_submission_trend["net_load_kwh"], 1)
df_combined_submission_trend = df_combined_submission_trend.with_columns(
    (a * pl.col("timestamp_numeric") + b).alias("net_load_kwh_trend_train"),
    (a_train_submission * pl.col("timestamp_numeric") + b_train_submission).alias("net_load_kwh_trend_train_submission")
)
df_combined_submission_trend.to_pandas().plot(x="timestamp_utc", y=["net_load_kwh","net_load_kwh_trend_train","net_load_kwh_trend_train_submission"])
plt.show()

submission.describe()


# Calculate metrics on a held-out validation set from train
val_size = int(len(X_train) * 0.2)
X_val_final = X_train.slice(-val_size, val_size).to_pandas()
y_val_final = y_train.slice(-val_size, val_size).to_pandas().squeeze()

val_pred = predict(X_val_final, models)
rmse = np.sqrt(np.mean((y_val_final - val_pred) ** 2))
mae = np.mean(np.abs(y_val_final - val_pred))
mean_load = np.mean(np.abs(y_val_final)) # changed
nrmse = rmse / mean_load * 100
nmae = mae / mean_load * 100

print(f"\nFinal Validation Metrics:")
print(f"NRMSE: {nrmse:.2f}%")
print(f"NMAE: {nmae:.2f}%")
print(f"\nTarget: NRMSE < 5% and NMAE < 5%")




