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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Visualization
import matplotlib.pyplot as plt # Visualization
from colorama import Fore

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

import warnings # Supress warnings 
warnings.filterwarnings('ignore')

np.random.seed(7)


df = pd.read_csv("../input/acea-water-prediction/Aquifer_Petrignano.csv")
df.head()


df.columns


# Remove old rows
df = df[df.Rainfall_Bastia_Umbra.notna()].reset_index(drop=True)
# Remove not usefull columns
df = df.drop(['Depth_to_Groundwater_P24', 'Temperature_Petrignano'], axis=1)


# Simplify column names
df.columns = ['date', 'rainfall', 'depth_to_groundwater', 'temperature', 'drainage_volume', 'river_hydrometry']

targets = ['depth_to_groundwater']
features = [feature for feature in df.columns if feature not in targets]
df.head()


df.shape


from datetime import datetime, date 

df['date'] = pd.to_datetime(df['date'], format = '%d/%m/%Y')
df.head().style.set_properties(subset=['date'], **{'background-color': 'dodgerblue'})


# To compelte the data, as naive method, we will use ffill
f, ax = plt.subplots(nrows=5, ncols=1, figsize=(15, 25))

for i, column in enumerate(df.drop('date', axis=1).columns):
    sns.lineplot(x=df['date'], y=df[column].fillna(method='ffill'), ax=ax[i], color='dodgerblue')
    ax[i].set_title('Feature: {}'.format(column), fontsize=14)
    ax[i].set_ylabel(ylabel=column, fontsize=14)
                      
    ax[i].set_xlim([date(2009, 1, 1), date(2020, 6, 30)]) 


df = df.sort_values(by='date')

# Check time intervals
df['delta'] = df['date'] - df['date'].shift(1)

df[['date', 'delta']].head()


df['delta'].sum(), df['delta'].count()


df = df.drop('delta', axis=1)
df.isna().sum()


# Checking missing values and plotting with replacing with NaN value

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))

old_hydrometry = df['river_hydrometry'].copy()
df['river_hydrometry'] = df['river_hydrometry'].replace(0, np.nan)

sns.lineplot(x=df['date'], y=old_hydrometry, ax=ax[0], color='darkorange', label='original')
sns.lineplot(x=df['date'], y=df['river_hydrometry'].fillna(np.inf), ax=ax[0], color='dodgerblue', label='modified')
ax[0].set_title('Feature: Hydrometry', fontsize=14)
ax[0].set_ylabel(ylabel='Hydrometry', fontsize=14)
ax[0].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])

old_drainage = df['drainage_volume'].copy()
df['drainage_volume'] = df['drainage_volume'].replace(0, np.nan)

sns.lineplot(x=df['date'], y=old_drainage, ax=ax[1], color='darkorange', label='original')
sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(np.inf), ax=ax[1], color='dodgerblue', label='modified')
ax[1].set_title('Feature: Drainage', fontsize=14)
ax[1].set_ylabel(ylabel='Drainage', fontsize=14)
ax[1].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])



f, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,5))

sns.heatmap(df.T.isna(), cmap='Blues')
ax.set_title('Missing Values', fontsize=16)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
plt.show()


f, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))

sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(0), ax=ax[0], color='darkorange', label = 'modified')
sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(np.inf), ax=ax[0], color='dodgerblue', label = 'original')
ax[0].set_title('Fill NaN with 0', fontsize=14)
ax[0].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

mean_drainage = df['drainage_volume'].mean()
sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(mean_drainage), ax=ax[1], color='darkorange', label = 'modified')
sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(np.inf), ax=ax[1], color='dodgerblue', label = 'original')
ax[1].set_title(f'Fill NaN with Mean Value ({mean_drainage:.0f})', fontsize=14)
ax[1].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

sns.lineplot(x=df['date'], y=df['drainage_volume'].ffill(), ax=ax[2], color='darkorange', label = 'modified')
sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(np.inf), ax=ax[2], color='dodgerblue', label = 'original')
ax[2].set_title(f'FFill', fontsize=14)
ax[2].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

sns.lineplot(x=df['date'], y=df['drainage_volume'].interpolate(), ax=ax[3], color='darkorange', label = 'modified')
sns.lineplot(x=df['date'], y=df['drainage_volume'].fillna(np.inf), ax=ax[3], color='dodgerblue', label = 'original')
ax[3].set_title(f'Interpolate', fontsize=14)
ax[3].set_ylabel(ylabel='Volume C10 Petrignano', fontsize=14)

for i in range(4):
    ax[i].set_xlim([date(2019, 5, 1), date(2019, 10, 1)])
    
plt.tight_layout()
plt.show()


df['drainage_volume'] = df['drainage_volume'].interpolate()
df['river_hydrometry'] = df['river_hydrometry'].interpolate()
df['depth_to_groundwater'] = df['depth_to_groundwater'].interpolate()


fig, ax = plt.subplots(ncols=2, nrows=3, sharex=True, figsize=(16,12))

# Drainage Volume (Daily)
sns.lineplot(data=df, x='date', y='drainage_volume', color='dodgerblue', ax=ax[0, 0])
ax[0, 0].set_title('Drainage Volume', fontsize=14)

# Weekly Drainage Volume
resampled_df = df.set_index('date').resample('7D').sum().reset_index()
sns.lineplot(data=resampled_df, x='date', y='drainage_volume', color='dodgerblue', ax=ax[1, 0])
ax[1, 0].set_title('Weekly Drainage Volume', fontsize=14)

# Monthly Drainage Volume
resampled_df = df.set_index('date').resample('M').sum().reset_index()
sns.lineplot(data=resampled_df, x='date', y='drainage_volume', color='dodgerblue', ax=ax[2, 0])
ax[2, 0].set_title('Monthly Drainage Volume', fontsize=14)

# Setting x-axis limits
for i in range(3):
    ax[i, 0].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])

# Temperature (Daily)
sns.lineplot(data=df, x='date', y='temperature', color='dodgerblue', ax=ax[0, 1])
ax[0, 1].set_title('Daily Temperature (Acc.)', fontsize=14)

# Weekly Temperature
resampled_df = df.set_index('date').resample('7D').mean().reset_index()
sns.lineplot(data=resampled_df, x='date', y='temperature', color='dodgerblue', ax=ax[1, 1])
ax[1, 1].set_title('Weekly Temperature (Acc.)', fontsize=14)

# Monthly Temperature
resampled_df = df.set_index('date').resample('M').mean().reset_index()
sns.lineplot(data=resampled_df, x='date', y='temperature', color='dodgerblue', ax=ax[2, 1])
ax[2, 1].set_title('Monthly Temperature (Acc.)', fontsize=14)

for i in range(3):
    ax[i, 1].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
plt.show()


# As we can see, downsample to weekly could smooth the data and help with analysis
downsample = df[['date',
                 'depth_to_groundwater', 
                 'temperature',
                 'drainage_volume', 
                 'river_hydrometry',
                 'rainfall'
                ]].resample('7D', on='date').mean().reset_index(drop=False)

df = downsample.copy()


# Adding columns
import pandas as pd

df['date'] = pd.to_datetime(df['date'])  

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week  
df['quarter'] = df['date'].dt.quarter
df['season'] = df['month'] % 12 // 3 + 1  # Assigns seasons based on month

#df[['date', 'year', 'month', 'day', 'day_of_year', 'week_of_year', 'quarter', 'season']].head()

df.head()



f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))

sns.lineplot(x=df['date'], y=df['month'], color='dodgerblue')
ax.set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
plt.show()


month_in_year = 12
df['month_sin'] = np.sin(2*np.pi*df['month']/month_in_year)
df['month_cos'] = np.cos(2*np.pi*df['month']/month_in_year)

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

sns.scatterplot(x=df.month_sin, y=df.month_cos, color='dodgerblue')
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose

core_columns =  [
    'rainfall', 'temperature', 'drainage_volume', 
    'river_hydrometry', 'depth_to_groundwater'
]

for column in core_columns:
    decomp = seasonal_decompose(df[column], period=52, model='additive', extrapolate_trend='freq')
    df[f"{column}_trend"] = decomp.trend
    df[f"{column}_seasonal"] = decomp.seasonal


# plotting level, trend, seasonality, noise
from statsmodels.tsa.seasonal import seasonal_decompose

fig, ax = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(13,6))

for i, column in enumerate(['temperature', 'depth_to_groundwater']):
    res = seasonal_decompose(df[column], period=52, model='additive', extrapolate_trend='freq')

    ax[0, i].set_title(f'Decomposition of {column}', fontsize=16)
    res.observed.plot(ax=ax[0, i], legend=False, color='dodgerblue')
    ax[0, i].set_ylabel('Observed', fontsize=14)

    res.trend.plot(ax=ax[1, i], legend=False, color='dodgerblue')
    ax[1, i].set_ylabel('Trend', fontsize=14)

    res.seasonal.plot(ax=ax[2, i], legend=False, color='dodgerblue')
    ax[2, i].set_ylabel('Seasonal', fontsize=14)

    res.resid.plot(ax=ax[3, i], legend=False, color='dodgerblue')
    ax[3, i].set_ylabel('Residual', fontsize=14)

plt.show()



weeks_in_month = 4

for column in core_columns:
    df[f'{column}_seasonal_shift_b_2m'] = df[f'{column}_seasonal'].shift(-2 * weeks_in_month)
    df[f'{column}_seasonal_shift_b_1m'] = df[f'{column}_seasonal'].shift(-1 * weeks_in_month)
    df[f'{column}_seasonal_shift_1m'] = df[f'{column}_seasonal'].shift(1 * weeks_in_month)
    df[f'{column}_seasonal_shift_2m'] = df[f'{column}_seasonal'].shift(2 * weeks_in_month)
    df[f'{column}_seasonal_shift_3m'] = df[f'{column}_seasonal'].shift(3 * weeks_in_month)


f, ax = plt.subplots(nrows=5, ncols=1, figsize=(11, 11))
f.suptitle('Seasonal Components of Features', fontsize=16)

for i, column in enumerate(core_columns):
    sns.lineplot(x=df['date'], y=df[column + '_seasonal'], ax=ax[i], color='dodgerblue', label='P25')
    ax[i].set_ylabel(ylabel=column, fontsize=14)
    ax[i].set_xlim([date(2017, 9, 30), date(2020, 6, 30)])
    
plt.tight_layout()
plt.show()


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

corrmat = df[core_columns].corr()

sns.heatmap(corrmat, annot=True, vmin=-1, vmax=1, cmap='coolwarm_r', ax=ax[0])
ax[0].set_title('Correlation Matrix of Core Features', fontsize=16)

shifted_cols = [
    'depth_to_groundwater_seasonal',         
    'temperature_seasonal_shift_b_2m',
    'drainage_volume_seasonal_shift_2m', 
    'river_hydrometry_seasonal_shift_3m'
]
corrmat = df[shifted_cols].corr()

sns.heatmap(corrmat, annot=True, vmin=-1, vmax=1, cmap='coolwarm_r', ax=ax[1])
ax[1].set_title('Correlation Matrix of Lagged Features', fontsize=16)

plt.tight_layout()
plt.show()


df.info()


df.tail()


# df = df.dropna()


df.isna().sum()


df_ml = df.copy()
df_ml = df_ml.dropna()


from sklearn.model_selection import train_test_split

X = df_ml.drop(columns=["depth_to_groundwater",'date'])
y = df_ml['depth_to_groundwater']



X_ml_train, X_ml_test, y_ml_train, y_ml_test = train_test_split(X, y, random_state=42,test_size=0.2)

X_ml_train.shape, y_ml_train.shape



X_ml_train.shape,y_ml_train.shape


X_ml_train.shape


X_ml_train.isna().sum()


y_ml_train.head()


X_ml_train.shape, y_ml_train.shape


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.naive_bayes import GaussianNB

scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_ml_train)
X_scaled_test = scaler.transform(X_ml_test)


ml_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'SVR': SVR(),
    'LightGBM': LGBMRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42)
}

results = {}
for name, model in ml_models.items():
    model.fit(X_scaled_train, y_ml_train)
    preds = model.predict(X_scaled_test)
    mse = mean_squared_error(y_ml_test, preds)
    r2 = r2_score(y_ml_test, preds)
    results[name] = {'MSE': mse, 'R2': r2}

results


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error

# Define a function to calculate all common regression errors
def calculate_errors(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'MedianAE': median_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

# Compute all errors for each model
detailed_results = {}
for name, model in ml_models.items():
    preds = model.predict(X_scaled_test)
    detailed_results[name] = calculate_errors(y_ml_test, preds)

detailed_results


from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score




# Split data
split_index = int(len(df) * 0.8)
train_df = df_ml.iloc[:split_index]
test_df = df_ml.iloc[split_index:]

# Separate features and target
features = [col for col in df.columns if col not in ['date', 'depth_to_groundwater']]
X_train = train_df[features]
y_train = train_df['depth_to_groundwater']
X_test = test_df[features]
y_test = test_df['depth_to_groundwater']

# XGBoost
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

# AdaBoost
ada_model = AdaBoostRegressor()
ada_model.fit(X_train, y_train)
ada_predictions = ada_model.predict(X_test)
ada_mae = mean_absolute_error(y_test, ada_predictions)
ada_rmse = np.sqrt(mean_squared_error(y_test, ada_predictions))

# For XGBoost
xgb_metrics = {
    'MAE': mean_absolute_error(y_test, xgb_predictions),
    'MSE': mean_squared_error(y_test, xgb_predictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_predictions)),
    'MAPE': mean_absolute_percentage_error(y_test, xgb_predictions),
    'MedianAE': median_absolute_error(y_test, xgb_predictions),
    'R2': r2_score(y_test, xgb_predictions)
}

# Print XGBoost metrics
print('XGBoost Metrics:')
for metric, value in xgb_metrics.items():
    print(f'{metric}: {value}')

print("__________________________________________________")
# For AdaBoost
ada_metrics = {
    'MAE': mean_absolute_error(y_test, ada_predictions),
    'MSE': mean_squared_error(y_test, ada_predictions),
    'RMSE': np.sqrt(mean_squared_error(y_test, ada_predictions)),
    'MAPE': mean_absolute_percentage_error(y_test, ada_predictions),
    'MedianAE': median_absolute_error(y_test, ada_predictions),
    'R2': r2_score(y_test, ada_predictions)
}

# Print AdaBoost metrics
print('AdaBoost Metrics:')
for metric, value in ada_metrics.items():
    print(f'{metric}: {value}')


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Assuming X_train, y_train, X_test, y_test are already defined from the dataset

# Scale features
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)

# Scale target for neural network
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()



# Define and train the neural network
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='sigmoid'))  # Output layer for regression
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train_scaled, y_train_scaled, epochs=80, batch_size=32, validation_split=0.2, verbose=1,)
y_pred_scaled = model.predict(X_test_scaled)
y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'MedianAE': median_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


# Calculate and print metrics for Neural Network
nn_metrics = calculate_metrics(y_test, y_pred)
print('Neural Network Metrics:')
for metric, value in nn_metrics.items():
    print(f'{metric}: {value}')


from sklearn.model_selection import TimeSeriesSplit

N_SPLITS = 3

X = df['date']
y = df['depth_to_groundwater']

folds = TimeSeriesSplit(n_splits=N_SPLITS)


X.info()


f, ax = plt.subplots(nrows=N_SPLITS, ncols=2, figsize=(16, 9))

for i, (train_index, valid_index) in enumerate(folds.split(X)):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    sns.lineplot(
        x=X_train, 
        y=y_train, 
        ax=ax[i,0], 
        color='dodgerblue', 
        label='train'
    )
    sns.lineplot(
        x=X_train[len(X_train) - len(X_valid):(len(X_train) - len(X_valid) + len(X_valid))], 
        y=y_train[len(X_train) - len(X_valid):(len(X_train) - len(X_valid) + len(X_valid))], 
        ax=ax[i,1], 
        color='dodgerblue', 
        label='train'
    )

    for j in range(2):
        sns.lineplot(x= X_valid, y= y_valid, ax=ax[i, j], color='darkorange', label='validation')
    ax[i, 0].set_title(f"Rolling Window with Adjusting Training Size (Split {i+1})", fontsize=16)
    ax[i, 1].set_title(f"Rolling Window with Constant Training Size (Split {i+1})", fontsize=16)

for i in range(N_SPLITS):
    ax[i, 0].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
    ax[i, 1].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
    
plt.tight_layout()
plt.show()


train_size = int(0.85 * len(df))
test_size = len(df) - train_size

univariate_df = df[['date', 'depth_to_groundwater']].copy()
univariate_df.columns = ['ds', 'y']

train = univariate_df.iloc[:train_size, :]

x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])

print(len(train), len(x_valid))


from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score
)
import numpy as np
from prophet import Prophet
from colorama import Fore  # Assuming colorama is already imported

# Train the model
model = Prophet()
model.fit(train)

# Predict on valid set
y_pred = model.predict(x_valid)

# Extract predictions for validation period
preds = y_pred.tail(test_size)['yhat']

# Calculate all metrics
metrics = {
    'MAE': mean_absolute_error(y_valid, preds),
    'MSE': mean_squared_error(y_valid, preds),
    'RMSE': np.sqrt(mean_squared_error(y_valid, preds)),
    'MAPE': mean_absolute_percentage_error(y_valid, preds),
    'MedianAE': median_absolute_error(y_valid, preds),
    'R2': r2_score(y_valid, preds)
}

# Print formatted results
print(Fore.GREEN + 'Model Evaluation Metrics:')
for metric, value in metrics.items():
    print(f"{Fore.GREEN}{metric}: {value:.4f}" if metric != 'R2' else f"{Fore.GREEN}{metric}: {value:.4f}")




# Plot the forecast
import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(1)
f.set_figheight(6)
f.set_figwidth(15)

# Plot Prophet's prediction
model.plot(y_pred, ax=ax)

# Add ground truth overlay
sns.lineplot(
    x=x_valid['ds'].tail(test_size),  # Ensure we only plot validation period
    y=y_valid['y'].tail(test_size),   # Match test period length
    ax=ax, 
    color='orange', 
    label='Ground truth'
)

# Add metrics to title using f-string
ax.set_title(
    f'Prediction vs Ground Truth\nMAE: {metrics["MAE"]:.2f}, RMSE: {metrics["RMSE"]:.2f}', 
    fontsize=14
)

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Depth to Groundwater', fontsize=14)
ax.legend()

plt.show()



from statsmodels.tsa.arima.model import ARIMA

# Fit model
model = ARIMA(y_train, order=(1,1,1))
model_fit = model.fit()

# Prediction with ARIMA
forecast_obj = model_fit.get_forecast(steps=90)
y_pred = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int()  

# Calcuate metrics
score_mae = mean_absolute_error(y_valid, y_pred)
score_rmse = math.sqrt(mean_squared_error(y_valid, y_pred))

print(Fore.GREEN + 'MAE: {}'.format(score_mae))
print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))


import matplotlib.pyplot as plt
import seaborn as sns

# Forecast
forecast_obj = model_fit.get_forecast(steps=90)
y_pred = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int()

# Plotting
f, ax = plt.subplots(1)
f.set_figheight(6)
f.set_figwidth(15)

# Plot forecasted values
sns.lineplot(x=y_valid.index, y=y_pred, ax=ax, label='Forecast', color='blue')

# Plot actual values
sns.lineplot(x=y_valid.index, y=y_valid['y'], ax=ax, label='Ground truth', color='orange')

# Plot confidence interval
ax.fill_between(y_valid.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                color='skyblue', alpha=0.3, label='Confidence Interval')

# Annotations
ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=14)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Depth to Groundwater', fontsize=14)

ax.set_ylim(-35, -18)
ax.legend()
plt.tight_layout()
plt.show()



!pip -q install pmdarima


from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

model = pm.auto_arima(y_train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


model.plot_diagnostics(figsize=(16,8))
plt.show()


from sklearn.preprocessing import MinMaxScaler

data = univariate_df.filter(['y'])
#Convert the dataframe to a numpy array
dataset = data.values

scaler = MinMaxScaler(feature_range=(-1, 0))
scaled_data = scaler.fit_transform(dataset)

scaled_data[:10]


# Defines the rolling window
look_back = 52
# Split into train and test sets
train, test = scaled_data[:train_size-look_back,:], scaled_data[train_size-look_back:,:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        a = dataset[i-look_back:i, 0]
        X.append(a)
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)

x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

print(len(x_train), len(x_test))


from keras.models import Sequential
from keras.layers import Dense, LSTM

#Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5, validation_data=(x_test, y_test))

model.summary()


# Lets predict with the model
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])

test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Get the root mean squared error (RMSE) and MAE
score_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
score_mae = mean_absolute_error(y_test[0], test_predict[:,0])

print(Fore.GREEN + 'MAE: {}'.format(score_mae))
print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))


feature_columns = [
    'rainfall',
    'temperature',
    'drainage_volume',
    'river_hydrometry',
]
target_column = ['depth_to_groundwater']

train_size = int(0.85 * len(df))

multivariate_df = df[['date'] + target_column + feature_columns].copy()
multivariate_df.columns = ['ds', 'y'] + feature_columns

train = multivariate_df.iloc[:train_size, :]
x_train, y_train = pd.DataFrame(multivariate_df.iloc[:train_size, [0,2,3,4,5]]), pd.DataFrame(multivariate_df.iloc[:train_size, 1])
x_valid, y_valid = pd.DataFrame(multivariate_df.iloc[train_size:, [0,2,3,4,5]]), pd.DataFrame(multivariate_df.iloc[train_size:, 1])

train.head()


from prophet import Prophet


# Train the model
model = Prophet()
model.add_regressor('rainfall')
model.add_regressor('temperature')
model.add_regressor('drainage_volume')
model.add_regressor('river_hydrometry')

# Fit the model with train set
model.fit(train)

# Predict on valid set
y_pred = model.predict(x_valid)

# Calcuate metrics
score_mae = mean_absolute_error(y_valid, y_pred['yhat'])
score_rmse = math.sqrt(mean_squared_error(y_valid, y_pred['yhat']))

print(Fore.GREEN + 'MAE: {}'.format(score_mae))

print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))


# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(6)
f.set_figwidth(15)

model.plot(y_pred, ax=ax)
sns.lineplot(x=x_valid['ds'], y=y_valid['y'], ax=ax, color='orange', label='Ground truth') #navajowhite

ax.set_title(f'Prediction \n MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}', fontsize=14)
ax.set_xlabel(xlabel='Date', fontsize=14)
ax.set_ylabel(ylabel='Depth to Groundwater', fontsize=14)

plt.show()


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler
import numpy as np

# Convert to numpy arrays
X_train = x_train.drop(columns=['ds']).values
X_valid = x_valid.drop(columns=['ds']).values
y_train = y_train.values
y_valid = y_valid.values

# Feature Scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_valid_scaled = scaler_X.transform(X_valid)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_valid_scaled = scaler_y.transform(y_valid)

# Reshape to 3D [samples, timesteps, features]
def create_sequences(data, seq_length=30):
    xs = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        xs.append(x)
    return np.array(xs)

seq_length = 30  # Lookback window

# Create sequences
X_train_seq = create_sequences(X_train_scaled, seq_length)
X_valid_seq = create_sequences(X_valid_scaled, seq_length)

# Align target values with sequences
y_train_seq = y_train_scaled[seq_length:]
y_valid_seq = y_valid_scaled[seq_length:]

# LSTM Model
model = Sequential([
    LSTM(70, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.2),
    LSTM(28, return_sequences=True),
    Dropout(0.2),
    LSTM(8, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_valid_seq, y_valid_seq),
    epochs=25,
    batch_size=32,
    verbose=1
)

# Make predictions
train_predict = model.predict(X_train_seq)
valid_predict = model.predict(X_valid_seq)

# Inverse transform predictions
train_predict = scaler_y.inverse_transform(train_predict)
valid_predict = scaler_y.inverse_transform(valid_predict)

# For evaluation, use corresponding y values
y_train_actual = y_train[seq_length:]
y_valid_actual = y_valid[seq_length:]

# Calculate metrics (using your previous metrics code)
metrics_train = {
    'MAE': mean_absolute_error(y_train_actual, train_predict),
    'MSE': mean_squared_error(y_train_actual, train_predict),
    'RMSE': np.sqrt(mean_squared_error(y_train_actual, train_predict)),
    'MAPE': mean_absolute_percentage_error(y_train_actual, train_predict),
    'MedianAE': median_absolute_error(y_train_actual, train_predict),
    'R2': r2_score(y_train_actual, train_predict)
}

metrics_valid = {
    'MAE': mean_absolute_error(y_valid_actual, valid_predict),
    'MSE': mean_squared_error(y_valid_actual, valid_predict),
    'RMSE': np.sqrt(mean_squared_error(y_valid_actual, valid_predict)),
    'MAPE': mean_absolute_percentage_error(y_valid_actual, valid_predict),
    'MedianAE': median_absolute_error(y_valid_actual, valid_predict),
    'R2': r2_score(y_valid_actual, valid_predict)
}

print("Training Metrics:", metrics_train)
print("Validation Metrics:", metrics_valid)

# Plotting
plt.figure(figsize=(15,6))
plt.plot(x_valid['ds'].iloc[seq_length:], y_valid_actual, label='Ground Truth', color='orange')
plt.plot(x_valid['ds'].iloc[seq_length:], valid_predict, label='Predictions', color='blue')
plt.title(f'LSTM Predictions vs Actual\nValidation MAE: {metrics_valid["MAE"]:.2f}, RMSE: {metrics_valid["RMSE"]:.2f}')
plt.xlabel('Date')
plt.ylabel('Depth to Groundwater')
plt.legend()
plt.show()








