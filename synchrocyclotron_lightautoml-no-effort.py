!pip install lightautoml


# Standard python libraries
import os
import requests

# Essential DS libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

# LightAutoML presets, task and report generation
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from lightautoml.report.report_deco import ReportDeco, ReportDecoUtilized
from lightautoml.addons.tabular_interpretation import SSWARM


N_THREADS = 4
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 3600
TARGET_NAME = 'rainfall'


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


test.shape


train.head()


train.rename(columns={'temparature': 'temperature'}, inplace=True)
test.rename(columns={'temparature': 'temperature'}, inplace=True)


def create_features(df):
    """
    Generates advanced weather features from a DataFrame containing weather data.

    This function creates a variety of features based on time, temperature, humidity, wind, pressure, clouds, 
    sunshine, and their interactions. It includes cyclic encodings, seasonal anomalies, and new features like 
    lag variables, pressure tendency, and moisture flux to enhance weather analysis or modeling.

    Parameters:
    df (pd.DataFrame): Input DataFrame with columns including 'day', 'temperature', 'maxtemp', 'mintemp', 
                       'dewpoint', 'humidity', 'windspeed', 'winddirection', 'pressure', 'cloud', 'sunshine'.

    Returns:
    pd.DataFrame: The input DataFrame with additional advanced weather features.
    """

    # =============== TIME-BASED FEATURES ===============
    # Calculate day of year (1-365) from a day counter
    df['day_of_year'] = ((df['day'] - 1) % 365) + 1

    # Calculate month (1-12) using cumulative days in a non-leap year
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    df['month'] = df['day_of_year'].apply(
        lambda x: next((m for m, days in enumerate(np.cumsum(month_lengths), 1) if x <= days), 12)
    )

    # Define seasons for Northern Hemisphere: 1=Winter, 2=Spring, 3=Summer, 4=Fall
    season_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}
    df['season'] = df['month'].map(season_map)

    # Cyclic encoding for month (captures circular nature of time)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Cyclic encoding for day of year
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # =============== TEMPERATURE-BASED FEATURES ===============
    # Daily temperature range (diurnal variation)
    df['temp_range'] = df['maxtemp'] - df['mintemp']

    # Deviation of temperature from daily average
    df['temp_from_avg'] = df['temperature'] - ((df['maxtemp'] + df['mintemp']) / 2)

    # Temperature anomaly relative to seasonal average
    season_avg_temp = df.groupby('season')['temperature'].transform('mean')
    df['temp_season_anomaly'] = df['temperature'] - season_avg_temp

    # =============== HUMIDITY AND MOISTURE FEATURES ===============
    # Dew point depression (indicates moisture availability)
    df['dew_depression'] = df['temperature'] - df['dewpoint']

    # Vapor pressure (actual, based on dew point)
    df['vapor_pressure'] = 6.11 * 10.0 ** (7.5 * df['dewpoint'] / (237.3 + df['dewpoint']))

    # Saturation vapor pressure (based on temperature)
    df['saturation_vapor_pressure'] = 6.11 * 10.0 ** (7.5 * df['temperature'] / (237.3 + df['temperature']))

    # Vapor pressure deficit (potential for evaporation)
    df['vapor_pressure_deficit'] = df['saturation_vapor_pressure'] - df['vapor_pressure']

    # Calculated relative humidity (%)
    df['calculated_humidity'] = 100 * (df['vapor_pressure'] / df['saturation_vapor_pressure'])

    # Difference between observed and calculated humidity
    df['humidity_gradient'] = df['humidity'] - df['calculated_humidity']

    # Humidity anomaly relative to seasonal average
    season_avg_humidity = df.groupby('season')['humidity'].transform('mean')
    df['humidity_season_anomaly'] = df['humidity'] - season_avg_humidity

    # =============== WIND FEATURES ===============
    # Wind components (u: east-west, v: north-south, meteorological convention)
    df['wind_u'] = -df['windspeed'] * np.sin(np.radians(df['winddirection']))
    df['wind_v'] = -df['windspeed'] * np.cos(np.radians(df['winddirection']))

    # Cyclical encoding of wind direction
    df['wind_dir_sin'] = np.sin(np.radians(df['winddirection']))
    df['wind_dir_cos'] = np.cos(np.radians(df['winddirection']))

    # Wind speed anomaly relative to seasonal average
    season_avg_wind = df.groupby('season')['windspeed'].transform('mean')
    df['wind_season_anomaly'] = df['windspeed'] - season_avg_wind

    # =============== PRESSURE FEATURES ===============
    # Normalize pressure using its standard deviation (reference: 1013 hPa)
    pressure_std = df['pressure'].std()
    df['pressure_normalized'] = (df['pressure'] - 1013) / pressure_std

    # Pressure anomaly relative to seasonal average
    season_avg_pressure = df.groupby('season')['pressure'].transform('mean')
    df['pressure_season_anomaly'] = df['pressure'] - season_avg_pressure

    # =============== CLOUD AND SUNSHINE FEATURES ===============
    # Ratio of cloud cover to sunshine, with small epsilon to avoid division by zero
    epsilon = 1e-6
    df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'] + epsilon)

    # Effective sky condition (cloud impact adjusted by sunshine)
    df['effective_sky'] = df['cloud'] * (1 - df['sunshine'] / 24)

    # Cloud cover anomaly relative to seasonal average
    season_avg_cloud = df.groupby('season')['cloud'].transform('mean')
    df['cloud_season_anomaly'] = df['cloud'] - season_avg_cloud

    # =============== COMBINED/INTERACTION FEATURES ===============
    # Temperature-humidity index (simplified heat index)
    df['temp_humidity_index'] = 0.8 * df['temperature'] + (df['humidity'] / 100) * (df['temperature'] - 14.4) + 46.4

    # Moist static energy proxy (combines temperature and moisture)
    df['moist_energy'] = df['temperature'] + 2.5 * df['vapor_pressure']

    # Potential for warm rain (temperature-humidity interaction)
    df['warm_rain_potential'] = df['humidity'] * df['temperature'] / 100

    # Atmospheric instability index
    df['instability_index'] = df['temp_range'] * df['humidity'] / 100

    # Moisture advection (wind-driven moisture transport)
    df['moisture_advection'] = df['windspeed'] * df['humidity'] / 100

    # Pressure-temperature interaction (scaled for balance)
    df['pressure_temp_factor'] = df['pressure_normalized'] * df['temperature'] / 20

    # Season-specific interactions
    df['winter_rain_factor'] = np.where(df['season'] == 1, df['humidity'] * df['pressure_normalized'] * -1, 0)
    df['summer_convection'] = np.where(df['season'] == 3, df['temp_range'] * df['humidity'] / 100, 0)

    # =============== CATEGORICAL FEATURES ===============
    # Wind direction in 8 categories (N, NE, E, SE, S, SW, W, NW)
    df['wind_dir_8cat'] = pd.cut(
        df['winddirection'],
        bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
        labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
        include_lowest=True
    )

    # Temperature categories
    df['temp_category'] = pd.cut(
        df['temperature'],
        bins=[-np.inf, 5, 10, 15, 20, 25, 30, np.inf],
        labels=['Very Cold', 'Cold', 'Cool', 'Mild', 'Warm', 'Hot', 'Very Hot']
    )

    # Humidity categories
    df['humidity_category'] = pd.cut(
        df['humidity'],
        bins=[0, 30, 60, 75, 85, 95, 100],
        labels=['Very Dry', 'Dry', 'Moderate', 'Humid', 'Very Humid', 'Saturated']
    )

    # =============== POTENTIAL RAINFALL INDICATORS ===============
    # K-Index proxy (stability indicator for precipitation)
    df['k_index_proxy'] = df['temperature'] - df['dew_depression'] + df['humidity'] / 100 * 20

    # Precipitation potential (combines humidity, dew depression, and cloud cover)
    df['precip_potential'] = (df['humidity'] / 100) * (100 - df['dew_depression']) * (df['cloud'] / 100)

    # Cloud-moisture interaction
    df['cloud_moisture'] = df['cloud'] * df['humidity'] / 100

    # Seasonal adjustment to rainfall probability
    df['seasonal_rain_prob'] = df['precip_potential'] * df.apply(
        lambda x: 1.3 if x['season'] in [1, 4] else  # Higher in Winter and Fall
                 0.9 if x['season'] == 3 else  # Lower in Summer
                 1.1,  # Moderate in Spring
        axis=1
    )

    # =============== NEW FEATURES ===============
    # Lag features (previous day's values)
    df['temp_lag1'] = df['temperature'].shift(1)
    df['humidity_lag1'] = df['humidity'].shift(1)

    # Pressure tendency (change from previous day)
    df['pressure_tendency'] = df['pressure'].diff()

    # Moisture flux components (directional moisture transport)
    df['moisture_flux_u'] = df['wind_u'] * df['humidity'] / 100
    df['moisture_flux_v'] = df['wind_v'] * df['humidity'] / 100

    return df


train = create_features(train)
test = create_features(test)


train.head()


train_data, test_data = train_test_split(
    train,
    test_size=TEST_SIZE,
    stratify=train[TARGET_NAME],
    random_state=RANDOM_STATE
)

print(f'Data is splitted. Parts sizes: train_data = {train_data.shape}, test_data = {test_data.shape}')


roles = {
    'target': TARGET_NAME,
    'drop': ['day', 'id']
}


task = Task('binary',
            metric='auc')


automl = TabularAutoML(
    task = task,
    timeout = TIMEOUT,
    cpu_limit = N_THREADS,
    reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
    general_params = {'use_algos': [[
        'linear_l2', 
        'lgb', 'lgb_tuned',
        'catboost', 'catboost_tuned',
        'nn', 'nn_tuned'
    ]]},
)


out_of_fold_predictions = automl.fit_predict(train_data, roles = roles, verbose = 1)


print(automl.create_model_str_desc())


print(f'OOF score: {roc_auc_score(train_data[TARGET_NAME].values, out_of_fold_predictions.data[:, 0])}')


%%time
test_pred = automl.predict(test)
print(f'Prediction for te_data:\n{test_pred[:10]}\nShape = {test_pred.shape}')


submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': 0
})


submission['rainfall'] = test_pred.data[:, 0]


submission


submission.to_csv('submission.csv', index = False)


print(f"Submission file created: {'submission.csv' in os.listdir()}")

