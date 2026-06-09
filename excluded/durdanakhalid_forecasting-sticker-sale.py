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


# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from itertools import product
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import holidays


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv',parse_dates=['date'])
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv',parse_dates=['date'])


train.head()


test.head()


train.isnull().sum()


# Find the pattern of missingness
sns.heatmap(train.isna(),cbar=False)
plt.title("Missing data Heatmap")
plt.show()


train.set_index('date',inplace=True)
test.set_index('date',inplace=True)


train.head()


def feature_engineering(df):
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    df['year'] = df.index.year
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['day_of_year'] = df.index.dayofyear
    # Define countries and their holidays
    countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
    
    # Create a set to store all holiday dates across the specified countries
    holiday_dates = set()

    # Loop through each country and add its holidays to the set
    for country in countries:
        try:
            # Add holidays for the years present in your dataset (adjust years as needed)
            country_holidays = holidays.CountryHoliday(country, years=[2021, 2022, 2023])
            holiday_dates.update(country_holidays.keys())
        except KeyError:
            print(f"Holidays for {country} are not available in the holidays library.")
    
    # Add 'is_holiday' column: 1 if the date is a holiday in any country, 0 otherwise
    df['is_holiday'] = df.index.isin(holiday_dates).astype(int)
    
    return df


train = feature_engineering(train)
test = feature_engineering(test)


# Seaparate missing and non missing data
missing_data = train[train['num_sold'].isna()]
non_missing_data = train.dropna()


categorical_columns = ['country','store','product']
nummerical_columns = ['year','month','day','weekday','is_weekend','day_of_year']
drop_columns = ['id']
preprocessor = ColumnTransformer(transformers=[
    ('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical_columns),
    ('num','passthrough',nummerical_columns)],
    remainder='drop'
)
# Separate the features and target
X_train = non_missing_data.drop(columns=drop_columns + ['num_sold'])
y_train = non_missing_data['num_sold']

X_missing = missing_data.drop(columns=drop_columns + ['num_sold'])

# Apply the ColumnTransformer
X_train_transformed = preprocessor.fit_transform(X_train)
X_missing_transformed = preprocessor.transform(X_missing)


# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(X_train_transformed,y_train)

# Predict missing Values
missing_data['num_sold'] = rf_model.predict(X_missing_transformed)

# Combine the datasets back
train = pd.concat([non_missing_data,missing_data]).sort_index()


# Plot to visualize
plt.plot(train.index,train['num_sold'],label='Imputed Data')
plt.title("Data After Filling Missing Values")
plt.legend()
plt.show()


# Plot sales over time
plt.figure(figsize=(12, 5))
plt.plot(train['num_sold'], label='Sticker Sales')
plt.title('Sticker Sales Over Time')
plt.legend()
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(train['num_sold'], model='additive', period=365)
decomposition.plot()
plt.show()


from statsmodels.tsa.seasonal import STL
stl = STL(train['num_sold'], period=365)
result = stl.fit()
result.plot()
plt.show()


# Perform seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(train['num_sold'], model='additive', period=30)
decomp.plot()
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series (assuming monthly data)
decomposition = seasonal_decompose(train['num_sold'], model='additive', period=12)  

# Plot the decomposed components
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(train['num_sold'], label='Original')
plt.legend()

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='green')
plt.legend()

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality', color='red')
plt.legend()

plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals', color='purple')
plt.legend()

plt.tight_layout()
plt.show()


import seaborn as sns

train['year'] = train.index.year
train['month'] = train.index.month

plt.figure(figsize=(12, 6))
sns.lineplot(data=train, x='month', y='num_sold', hue='year', palette='tab10')
plt.title('Seasonal Pattern Across Different Years')
plt.show()


from statsmodels.tsa.stattools import adfuller
result = adfuller(train['num_sold'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] < 0.05:
    print("The series is stationary")
else:
    print("The series is not stationary")


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ACF plot
plot_acf(train['num_sold'], lags=50)  # Adjust lags as necessary
plt.show()

# PACF plot
plot_pacf(train['num_sold'], lags=50)  # Adjust lags as necessary
plt.show()


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(train['num_sold'])


train['rolling_mean_7'] = train['num_sold'].rolling(window=7).mean()
train['rolling_std_7'] = train['num_sold'].rolling(window=7).std()
train['lag_1'] = train['num_sold'].shift(1)  # Previous day's sales
train['lag_7'] = train['num_sold'].shift(7)  # Sales from 7 days ago
train['lag_30'] = train['num_sold'].shift(30)  # Sales from 30 days ago


train[['num_sold', 'lag_1', 'lag_7', 'lag_30']].corr()


# Select only numeric columns
numeric_columns = train.select_dtypes(include=['number'])
corr_matrix = numeric_columns.corr()

# Plot heatmap
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


train.fillna(0,inplace=True)


from statsmodels.stats.outliers_influence import variance_inflation_factor

features = train[['lag_1', 'lag_7', 'rolling_mean_7']]
vif = pd.DataFrame()
vif['Features'] = features.columns
vif['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
print(vif)


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train['num_sold'], order=(2, 0, 0))  # AR(2) model
result = model.fit()
print(result.summary())


# Define function for SARIMA Hyperparameter Tuning
def sarima_hyperparameter_tuning(train_series, param_grid):
    best_score, best_params = float('inf'), None
    for params in product(*param_grid.values()):
        try:
            model = SARIMAX(train_series, order=params[:3], seasonal_order=params[3:], enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False)
            score = result.aic
            if score < best_score:
                best_score, best_params = score, params
        except:
            continue
    return best_params


# Define parameter grid
param_grid = {
    'p': [0, 1],
    'd': [0, 1],
    'q': [0, 1],
    'P': [0],
    'D': [0, 1],
    'Q': [0],
    's': [7]  # Weekly seasonality
}

# Find best SARIMA parameters
best_params = sarima_hyperparameter_tuning(train['num_sold'], param_grid)
print("Best SARIMAX Parameters:", best_params)


# Train final SARIMAX model with best parameters
model = SARIMAX(train['num_sold'], order=best_params[:3], seasonal_order=best_params[3:], enforce_stationarity=False, enforce_invertibility=False)
result = model.fit(disp=False)
print(result.summary())


# Forecast for test dataset
forecast = result.get_forecast(steps=len(test))


print(forecast.predicted_mean.values)


# print(forecast.predicted_mean.values)
test['num_sold'] = forecast.predicted_mean.values


test.head(9)








# Save submission file
submission = test[['num_sold']]
submission.reset_index(inplace=True)
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")


df1=pd.read_csv("submission.csv")


df1.head()




