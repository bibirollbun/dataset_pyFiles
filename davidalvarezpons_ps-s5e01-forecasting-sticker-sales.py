# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


trainDs = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', index_col="id")
testDs = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', index_col="id")


trainDs.head() # 2010 - 2016


testDs.head() # 2017 - 2019


date_ranges = {
    'train_start': trainDs.iloc[0,:]['date'],
    'train_end': trainDs.iloc[-1,:]['date'],
    'forecast_start': testDs.iloc[0,:]['date'], 
    'forecast_end': testDs.iloc[-1,:]['date'], 
}
date_ranges


trainDs.describe()


trainDs.dtypes


# Count values for each category
def countFeatureValues(df):
    keys = ['country', 'store', 'product']
    counts = {}
    for k in keys: 
        counts[k] = {
            'values': df[k].value_counts(), 
            'unique': df[k].nunique(),
        }
    return counts


countFeatureValues(trainDs)


trainDs['date'].value_counts()


print("Number of NA values in the target column: ", sum(trainDs['num_sold'].isna()))
print("Total number of rows in the training dataset: ", len(trainDs))


# Number of elements for each triplet where target value is not NaN
trainDs[~trainDs['num_sold'].isna()][['country', 'store', 'product']].value_counts()


# Number of elements for each triplet where target value *is* NaN
trainDs[trainDs['num_sold'].isna()][['country', 'store', 'product']].value_counts()


def plot_na_count(ax, data): 
    naSold = data.copy()
    naSold['date'] = pd.to_datetime(naSold['date'])
    naSold['num_sold_is_na'] = naSold['num_sold'].isna()
    daily_data = naSold.groupby('date')['num_sold_is_na'].sum().reset_index()
    ax.scatter(daily_data['date'], daily_data['num_sold_is_na'], label='Number of NA Sold')
    ax.set_title("Number of NA Sold", fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.grid(alpha=0.5)

def plot_na_count_by_col(ax, data, column): 
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['num_sold_is_na'] = df['num_sold'].isna()
    daily_data = df.groupby(['date', column])['num_sold_is_na'].sum().reset_index()
    pivoted_data = daily_data.pivot(index='date', columns=column, values='num_sold_is_na')
    for colName in pivoted_data.columns:
        if sum(pivoted_data[colName]) > 0:
            data_to_plot = pivoted_data[ pivoted_data[colName] > 0 ]
            ax.scatter(data_to_plot.index, data_to_plot[colName], label=colName, marker='.')
    ax.set_title('Number NA Sold Over Time by ' + column, fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Number NA Sold', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.5)

_, ax = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
plot_na_count(ax[0], trainDs)
plot_na_count_by_col(ax[1], trainDs, 'country')
plot_na_count_by_col(ax[2], trainDs, 'store')
plot_na_count_by_col(ax[3], trainDs, 'product')
plt.tight_layout()
plt.show()


from sklearn.base import BaseEstimator, TransformerMixin
import holidays

class AddHolidayTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self): 
        self.col = 'date'
        self.featOut = set()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        col = self.col
        # Source: https://www.kaggle.com/competitions/playground-series-s5e1/discussion/554680
        country2Code = dict(zip(np.sort(df.country.unique()), ['CA', 'FI', 'IT', 'KE', 'NO', 'SG']))
        h = {ct: holidays.country_holidays(cd, years=range(2010, 2020)) for ct, cd in country2Code.items()}
        df[f"{col}_is_holiday"] = 0
        for c in country2Code:
            df.loc[df['country'] == c, f"{col}_is_holiday"] = df['date'].isin(h[c]).astype(int)
        self.featOut.add(f"{col}_is_holiday")
        return df

    def get_feature_names_out(self, input_features=None): 
        return list(self.featOut)

import random
startIndex = random.randint(0, trainDs.shape[0])
holidayTrf = AddHolidayTransformer()
holidayDs = holidayTrf.transform(trainDs)
holidayDs.iloc[startIndex:].head(n=20)


from sklearn.base import BaseEstimator, TransformerMixin

class DateColumnTransformer(BaseEstimator, TransformerMixin):    
    def __init__(self, dropCol=True): 
        self.featOut = []
        self.dropCol = dropCol
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        col = df.columns[0]
        df[col] = pd.to_datetime(df[col])
        df[f"{col}_year"] = df[col].dt.year
        # Years are in range 2010 - 2019, let's consider this to build a year ratio
        minYear = 2010 
        maxYear = 2019
        df[f"{col}_year_ratio"] = (df[col].dt.year - minYear) / (maxYear - minYear)
    
        df[f"{col}_quarter"] = df[col].dt.quarter
        df[f"{col}_quarter_sin"] = np.sin(2 * np.pi * df[f"{col}_quarter"] / 4)
        df[f"{col}_quarter_cos"] = np.cos(2 * np.pi * df[f"{col}_quarter"] / 4)
        
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12)
        df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12)
    
        df[f"{col}_biannual_month"] = (12 * (df[f"{col}_year"] % 2)) + df[f"{col}_month"]
        df[f"{col}_biannual_month_sin"] = np.sin(2 * np.pi * df[f"{col}_biannual_month"] / 24)
        df[f"{col}_biannual_month_cos"] = np.cos(2 * np.pi * df[f"{col}_biannual_month"] / 24)
        
        df[f"{col}_dayofmonth"] = df[col].dt.day
        df[f"{col}_dayofmonth_sin"] = np.sin(2 * np.pi * df[f"{col}_dayofmonth"] / df[col].dt.days_in_month )
        df[f"{col}_dayofmonth_cos"] = np.cos(2 * np.pi * df[f"{col}_dayofmonth"] / df[col].dt.days_in_month )
        
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * df[f"{col}_dayofweek"] / 7)
        df[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * df[f"{col}_dayofweek"] / 7)
        
        df[f"{col}_weekend"] = pd.Series(df[f"{col}_dayofweek"] >= 5).astype('float')
        df.loc[df[f"{col}_dayofweek"] == 4, f"{col}_weekend"] = 0.5 # Fridays are half weekend
    
        df[f"{col}_xmas_season"] = pd.Series(
            (df[f"{col}_month"] == 12) & (df[f"{col}_dayofmonth"] >= 22) |
            (df[f"{col}_month"] == 1) & (df[f"{col}_dayofmonth"] <= 6)
        ).astype('float')
        
        df[f"{col}_dayofyear"] = df[col].dt.dayofyear
        df[f"{col}_dayofyear_sin"] = np.sin(2 * np.pi * df[f"{col}_dayofyear"] / (365 + df[col].dt.is_leap_year) )
        df[f"{col}_dayofyear_cos"] = np.cos(2 * np.pi * df[f"{col}_dayofyear"] / (365 + df[col].dt.is_leap_year) )
        
        # df[f"{col}_epoch_secs"] = df[col].astype("int64") // 10**9

        # Clean up auxiliary columns
        columnsToDrop = [f"{col}_year", f"{col}_month", f"{col}_quarter", f"{col}_dayofmonth", f"{col}_dayofweek", f"{col}_biannual_month"]
        if self.dropCol: 
            columnsToDrop.append(col)
        df = df.drop(columnsToDrop, axis=1)
        newCols = [c for c in df.columns if c not in X.columns or (not self.dropCol and c == col) ]
        self.featOut = newCols
        return df

    def get_feature_names_out(self, input_features=None):
        return self.featOut

import random
startIndex = random.randint(0, trainDs.shape[0])
dateTrf = DateColumnTransformer(dropCol=False)
splitDateDs = dateTrf.transform(trainDs)
print('Columns: ', dateTrf.get_feature_names_out())
splitDateDs.iloc[startIndex:].head(n=50)


def plot_date_transformations(ax, data, col='date'): 
    df = data.copy()
    df[col] = pd.to_datetime(df[col])
    daily_data = df.groupby(col).first().reset_index()
    new_cols = [name for name in daily_data.columns if name.startswith(f"{col}_") and name.endswith(('_sin', '_cos')) ]
    for c in new_cols: 
        ax.plot(daily_data[col], daily_data[c], label=c)
    ax.set_title(f'Date transformation columns extracted from "{col}"', fontsize=12)
    ax.set_xlabel('Date', fontsize=8)
    ax.legend(fontsize=10, bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.5)
    return ax

_, ax = plt.subplots(3, 1, figsize=(12, 15))
plot_date_transformations(ax[0], splitDateDs.head(9000))
plot_date_transformations(ax[1], splitDateDs.head(90000))
plot_date_transformations(ax[2], splitDateDs)
plt.tight_layout()
plt.show()


from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import requests

class AddGdpTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']
        self.gdpData = defaultdict(dict)
    
    def fit(self, X, y=None):
        return self

    def getGdpData(self, country, year): 
        if country not in self.gdpData or year not in self.gdpData[country]: 
            url='https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json'
            response = requests.get(url.format(country,year)).json()
            self.gdpData[country][year] = response[1][0]['value']
        return self.gdpData[country][year]
            
    def transform(self, X, y=None):
        df = X.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['alpha3'] = df['country'].map(dict(zip(
            np.sort(df['country'].unique()), self.alpha3s)))
        years = np.sort(df['date'].dt.year.unique())
        gdp = np.array([
            [self.getGdpData(alpha3, year) for year in years]
            for alpha3 in self.alpha3s
        ])
        gdp = pd.DataFrame(gdp, index=self.alpha3s, columns=years)
        df['gdp'] = df.apply(lambda s: gdp.loc[s['alpha3'], s['year']], axis=1)
        return df.drop(['year', 'alpha3'], axis=1)

    def get_feature_names_out(self, input_features=None): 
        return ['gdp']


startIndex = random.randint(0, trainDs.shape[0])
gdpTrf = AddGdpTransformer()
gdpDs = gdpTrf.transform(trainDs)
gdpDs.iloc[startIndex:].head(n=20)


sns.heatmap(gdpDs.loc[:, ['gdp', 'num_sold']].corr(), annot=True)


from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import requests

class AddInflationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.alpha3s = ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP']
        self.cpiData = defaultdict(dict)
    
    def fit(self, X, y=None):
        return self

    def getInflationData(self, country, year): 
        if country not in self.cpiData or year not in self.cpiData[country]: 
            url='https://api.worldbank.org/v2/country/{0}/indicator/FP.CPI.TOTL.ZG?date={1}&format=json'
            response = requests.get(url.format(country,year)).json()
            self.cpiData[country][year] = response[1][0]['value']
        return self.cpiData[country][year]
            
    def transform(self, X, y=None):
        df = X.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['alpha3'] = df['country'].map(dict(zip(
            np.sort(df['country'].unique()), self.alpha3s)))
        years = np.sort(df['date'].dt.year.unique())
        cpi = np.array([
            [self.getInflationData(alpha3, year) for year in years]
            for alpha3 in self.alpha3s
        ])
        cpi = pd.DataFrame(cpi, index=self.alpha3s, columns=years)
        df['cpi'] = df.apply(lambda s: cpi.loc[s['alpha3'], s['year']], axis=1)
        return df.drop(['year', 'alpha3'], axis=1)

    def get_feature_names_out(self, input_features=None): 
        return ['cpi']

startIndex = random.randint(0, trainDs.shape[0])
cpiTrf = AddInflationTransformer()
inflationDs = cpiTrf.transform(trainDs)
inflationDs.iloc[startIndex:].head(n=20)


sns.heatmap(inflationDs.loc[:, ['cpi', 'num_sold']].corr(), annot=True)


_, ax = plt.subplots(2, figsize=(12, 10))
sns.histplot(trainDs['num_sold'], kde=True, ax=ax[0])
sns.histplot(np.log1p(trainDs['num_sold']), kde=True, ax=ax[1])


def plot_timeseries(ax, data): 
    df = data.copy()
    df['date'] = pd.to_datetime(data['date'])
    daily_sales = df.groupby('date')['num_sold'].sum().reset_index()
    ax.plot(daily_sales['date'], daily_sales['num_sold'], label='Number Sold', color='blue')
    ax.set_title('Number Sold Over Time', fontsize=12)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Number Sold', fontsize=10)
    ax.grid(alpha=0.5)
    ax.legend(fontsize=8, bbox_to_anchor=(1, 1))

def plot_timeseries_by_column(ax, data, column): 
    df = data.copy()
    df['date'] = pd.to_datetime(data['date'])
    daily_sales = df.groupby(['date', column])['num_sold'].sum().reset_index()
    pivoted_sales = daily_sales.pivot(index='date', columns=column, values='num_sold')
    for colName in pivoted_sales.columns:
        ax.plot(pivoted_sales.index, pivoted_sales[colName], label=colName)
    ax.set_title('Number Sold Over Time by ' + column, fontsize=12)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Number Sold', fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.5)
    
_, ax = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
plot_timeseries(ax[0], trainDs)
plot_timeseries_by_column(ax[1], trainDs, 'country')
plot_timeseries_by_column(ax[2], trainDs, 'store')
plot_timeseries_by_column(ax[3], trainDs, 'product')
plt.tight_layout()
plt.show()


trainDs.head()





from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

def makePreprocessingPipeline(ds): 

    column_transformer = ColumnTransformer(transformers=[
        ('numeric_scaler', MinMaxScaler(feature_range=(-1, 1)), ['gdp', 'cpi']), 
        ('date_extension', DateColumnTransformer(), ['date']),
        # TODO: Skip for now since we will be grouping the datasets by these categories
        ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'), ['country', 'store', 'product']), 
    ],  remainder="passthrough")

    preprocessor = Pipeline([
        ("gdp_addition", AddGdpTransformer()),
        ("cpi_addition", AddInflationTransformer()),
        ("holiday_addition", AddHolidayTransformer()),
        ("column_transformer", column_transformer), 
    ])
    return preprocessor, column_transformer

prepTestDs = trainDs.loc[:100, ~trainDs.columns.isin(['num_sold'])]
featEngPipeline, colTr = makePreprocessingPipeline(prepTestDs)
aux = featEngPipeline.fit_transform(prepTestDs)
columns = [c.split('__')[-1] for c in colTr.get_feature_names_out()]
pd.DataFrame(aux, columns=columns)


from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

def makeExogeneousPipeline(ds): 

    column_transformer = ColumnTransformer(transformers=[
        # ('numeric_scaler', MinMaxScaler(feature_range=(-1, 1)), ['gdp', 'cpi']), 
        ('date_extension', DateColumnTransformer(dropCol=False), ['date']),
    ],  remainder="passthrough")

    preprocessor = Pipeline([
        ("gdp_addition", AddGdpTransformer()),
        ("cpi_addition", AddInflationTransformer()),
        ("holiday_addition", AddHolidayTransformer()),
        ("column_transformer", column_transformer), 
    ])
    return preprocessor, column_transformer

X_train = trainDs.copy()
y_train = trainDs['num_sold']
X_train = X_train.drop('num_sold', axis=1)
preprocessor, colTr = makeExogeneousPipeline(X_train)
prepData = preprocessor.fit_transform(X_train)
columns = [c.split('__')[-1] for c in colTr.get_feature_names_out()]
X_train = pd.DataFrame(prepData, columns=columns)
X_train.head()


X_test = testDs.copy()
prepData = preprocessor.transform(X_test)
columns = [c.split('__')[-1] for c in colTr.get_feature_names_out()]
X_test = pd.DataFrame(prepData, columns=columns)
X_test.head()


import itertools

def split_data_by_features(X, y=None, columns=[]): 
    nCols = len(columns)
    nRows = len(X)
    values = []
    for c in columns: 
        values.append(X[c].unique())
    groups = list(itertools.product(*values))
    splits = {}
    for group in groups: 
        mask = np.full(nRows, True)
        for i in range(nCols): 
            col = columns[i]
            val = group[i]
            mask &= X[col] == val
        splits[group] = X[mask].drop(columns, axis=1)
        if y is not None: 
            splits[group][y.name] = y 
    return splits

splits = split_data_by_features(X_train, y=y_train, columns=['country', 'store', 'product'])
{k: len(v) for k, v in splits.items()}


splits[('Italy', 'Discount Stickers', 'Holographic Goose')]


date_ranges


!pip install skforecast


df = splits[('Italy', 'Discount Stickers', 'Holographic Goose')].copy()

# Re-format the date column and set it as index
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.set_index('date', inplace = True)
df.sort_index(inplace = True)

# Set the dataset frequency to be (D)aily data
df = df.asfreq('D', method = 'bfill')

# Change data type to float. Also, we only use data since Jan 2020 until Nov 2023
exog_df = df.loc['2010-01-01':'2016-12-31', df.columns != 'num_sold']
for c in exog_df.columns:
    exog_df[c] = pd.to_numeric(exog_df[c], errors = 'coerce')
df = pd.to_numeric(df.loc['2010-01-01':'2016-12-31', 'num_sold'], errors = 'coerce') 
# Fill missing value with the latest available data
df.ffill(inplace = True)

df


exog_df


train_start = '2010-01-01'
train_end = '2015-12-31'

test_start = '2016-01-01'
test_end = '2016-12-31'

forecast_start = '2017-01-01'
forecast_end = '2019-12-31'


fig, ax = plt.subplots(figsize=(7, 3))
df.loc[train_start:train_end].plot(ax=ax, label = "Train")
df.loc[test_start:test_end].plot(ax=ax, label = "Test")
ax.legend()


from skforecast.recursive import ForecasterRecursive
from sklearn.tree import DecisionTreeRegressor

# Define the forecaster
forecaster = ForecasterRecursive(
    # Add the sklearn regressor and lags
    regressor = DecisionTreeRegressor(random_state = 123),
    lags = 30
)

# Fit the model using train data
forecaster.fit(y = df.loc[train_start:train_end], exog=exog_df.loc[train_start:train_end])

# Predict the test period
predicted_test = forecaster.predict(steps = len(df.loc[test_start:test_end]), exog=exog_df.loc[test_start:test_end])


fig, ax = plt.subplots(figsize=(7, 3))
df.loc[test_start:test_end].plot(ax=ax, label = "Test")
predicted_test.plot(ax=ax, label = 'Predicted DT')
ax.legend()


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

rmse_train = np.sqrt(np.mean(np.square(forecaster.in_sample_residuals_)))
rmse_test = np.sqrt(mean_squared_error(df.loc[test_start:test_end], predicted_test))
mape_test = mean_absolute_percentage_error(df.loc[test_start:test_end], predicted_test)
print('RMSE Train:', rmse_train,'\nRMSE Test:', rmse_test, '\nMAPE Test:', mape_test)


from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from sklearn.tree import DecisionTreeRegressor

forecaster_dt = ForecasterRecursive(
    regressor = DecisionTreeRegressor(random_state = 123),
    lags = 30
)

# With Refit and Increasing Train Size
cv = TimeSeriesFold(
     steps = 30,
     initial_train_size = len(df.loc[train_start:train_end]),
     refit = True, # Change this to False to disable refit
     fixed_train_size = False, # Set this to true for fixed train size
     allow_incomplete_fold = True,
 )
metric, predicted_cv_dt = backtesting_forecaster(
    forecaster = forecaster_dt,
    y = df.loc[train_start:test_end],
    cv = cv, 
    metric = 'mean_absolute_percentage_error',
)

print('[DT] CV Train MAPE:', metric)

# Predict the test period

# Fit the model using train data
forecaster_dt.fit(y = df.loc[train_start:train_end])

# Predict the test period
predicted_test_dt = forecaster_dt.predict(steps = len(df.loc[test_start:test_end]))
test_mape_dt = mean_absolute_percentage_error(df.loc[test_start:test_end], predicted_test_dt)
print('[DT] Test MAPE:', test_mape_dt)


from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from xgboost import XGBRegressor

forecaster_xgb = ForecasterRecursive(
    regressor = XGBRegressor(eval_metric='mape', n_jobs=None, random_state = 123),
    lags = 50
)

# With Refit and Increasing Train Size
cv = TimeSeriesFold(
     steps = 30,
     initial_train_size = len(df.loc[train_start:train_end]),
     refit = True, # Change this to False to disable refit
     fixed_train_size = False, # Set this to true for fixed train size
     allow_incomplete_fold = True,
 )
metric, predicted_cv_xgb = backtesting_forecaster(
    forecaster = forecaster_xgb,
    y = df.loc[train_start:test_end],
    cv = cv, 
    metric = 'mean_absolute_percentage_error',
)

print('[XGB] CV Train MAPE:', metric)

# Predict the test period

# Fit the model using train data
forecaster_xgb.fit(y = df.loc[train_start:train_end])

# Predict the test period
predicted_test_xgb = forecaster_xgb.predict(steps = len(df.loc[test_start:test_end]))
test_mape_xgb = mean_absolute_percentage_error(df.loc[test_start:test_end], predicted_test_xgb)
print('[XGB] Test MAPE:', test_mape_xgb)


from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from xgboost import XGBRegressor

forecaster_xgb_exog = ForecasterRecursive(
    regressor = XGBRegressor(eval_metric='mape', n_jobs=None, random_state = 123),
    lags = 20
)

# With Refit and Increasing Train Size
cv = TimeSeriesFold(
     steps = 30,
     initial_train_size = len(df.loc[train_start:train_end]),
     refit = True, # Change this to False to disable refit
     fixed_train_size = False, # Set this to true for fixed train size
     allow_incomplete_fold = True,
 )
metric, predicted_cv_xgb_exog = backtesting_forecaster(
    forecaster = forecaster_xgb_exog,
    y = df.loc[train_start:test_end],
    exog = exog_df.loc[train_start:test_end],
    cv = cv, 
    metric = 'mean_absolute_percentage_error',
)

print('[XGB+exog] CV Train MAPE:', metric)

# Predict the test period

# Fit the model using train data
forecaster_xgb_exog.fit(y = df.loc[train_start:train_end], exog=exog_df.loc[train_start:train_end])

# Predict the test period
predicted_test_xgb_exog = forecaster_xgb_exog.predict(
    steps = len(df.loc[test_start:test_end]), 
    exog=exog_df.loc[test_start:test_end])
test_mape_xgb_exog = mean_absolute_percentage_error(df.loc[test_start:test_end], predicted_test_xgb_exog)
print('[XGB+exog] Test MAPE:', test_mape_xgb_exog)


# GRID SEARCH + EXOG

from skforecast.model_selection import TimeSeriesFold, grid_search_forecaster
from xgboost import XGBRegressor

param_grid = {
    'n_estimators': [1000], 
    'learning_rate': [0.1],
    'reg_alpha': [0.005],
    'max_depth': [5],
}
lags_grid = [5]

forecaster_xgb_exog_grid = ForecasterRecursive(
    regressor = XGBRegressor(eval_metric='mape', random_state = 123),
    lags = 20
)

# With Refit and Increasing Train Size
cv = TimeSeriesFold(
     steps = 10,
     initial_train_size = len(df.loc[train_start:train_end]),
     refit = True, # Change this to False to disable refit
     fixed_train_size = False, # Set this to true for fixed train size
     allow_incomplete_fold = True,
 )
predicted_cv_xgb_exog_grid = grid_search_forecaster(
    forecaster = forecaster_xgb_exog_grid,
    y = df.loc[train_start:test_end],
    exog = exog_df.loc[train_start:test_end],
    cv = cv, 
    metric = 'mean_absolute_percentage_error',
    param_grid = param_grid, 
    lags_grid = lags_grid, 
    return_best = True,
    verbose = False,
    show_progress = True,
    n_jobs = -1
)


# Fit the model using train data
forecaster_xgb_exog_best = ForecasterRecursive(
    regressor = XGBRegressor(
        eval_metric='mape', 
        learning_rate=0.1, 
        reg_alpha=0.005,
        max_depth=5, 
        n_estimators=1000,
        random_state = 123, 
        n_jobs=-1
    ),
    lags = 5
)
forecaster_xgb_exog_best.fit(
    y = df.loc[train_start:train_end], 
    exog=exog_df.loc[train_start:train_end]
)
predicted_test_xgb_exog_best = forecaster_xgb_exog_best.predict(
    steps = len(df.loc[test_start:test_end]), 
    exog=exog_df.loc[test_start:test_end]
)
test_mape_xgb_exog_best = mean_absolute_percentage_error(
    df.loc[test_start:test_end], 
    predicted_test_xgb_exog_best
)
print('[XGB+exog+grid_best] Test MAPE:', test_mape_xgb_exog_best)

# [XGB+exog+grid_best] Test MAPE: 0.04717454180572926


fig, ax = plt.subplots(figsize=(7, 3))
df.loc[test_start:test_end].plot(ax=ax, label = "Test")
# predicted_test.plot(ax=ax, label = 'Predicted DT')
# arima_test.plot(ax=ax, label = 'Predicted ARIMA')
# predicted_test_dt.plot(ax=ax, label = 'Predicted DT+CV')
# predicted_test_xgb.plot(ax=ax, label = 'Predicted XGB+CV')
# predicted_test_xgb_exog.plot(ax=ax, label = 'Predicted XGB+exog+CV')
predicted_test_xgb_exog_best.plot(ax=ax, label = 'Predicted XGB+exog+grid_best')
ax.legend()


fig, ax = plt.subplots(figsize=(7, 3))
df.loc[train_start:train_end].plot(ax=ax, label = "Train")
df.loc[test_start:test_end].plot(ax=ax, label = "Test")
predicted_test_xgb.plot(ax=ax, label = 'Predicted CV')
predicted_test_xgb_exog.plot(ax=ax, label = 'Predicted XGB+exog+CV')
predicted_test_xgb_exog_best.plot(ax=ax, label = 'Predicted XGB+exog+grid_best')
ax.legend()


train_splits = split_data_by_features(X_train, y=y_train, columns=['country', 'store', 'product'])
train_splits[('Italy', 'Discount Stickers', 'Holographic Goose')].head()


train_start = '2010-01-01'
train_end = '2015-12-31'

test_start = '2016-01-01'
test_end = '2016-12-31'

forecast_start = '2017-01-01'
forecast_end = '2019-12-31'


%%time

from datetime import datetime
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor

def setupIndexAndNumericTypes(data): 
    df = data.copy()
    # Re-format the date column and set it as index
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace = True)
    df.sort_index(inplace = True)
    # Set the dataset frequency to be (D)aily data
    df = df.asfreq('D', method = 'bfill')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors = 'coerce')
    return df

def trainSplitModel(splitDs, train_start, train_end, test_start, test_end): 
    df = setupIndexAndNumericTypes(splitDs)
    # Change data type to float. Also, we only use data since Jan 2020 until Nov 2023
    exog_df = df.loc['2010-01-01':'2016-12-31', df.columns != 'num_sold']
    y = pd.to_numeric(df.loc['2010-01-01':'2016-12-31', 'num_sold'], errors = 'coerce') 
    # Fill missing value with the latest available data
    y.ffill(inplace = True)
    y.bfill(inplace=True)
    # Last chance filling
    y.fillna(value=0, inplace=True)
    
    y_train = y.loc[train_start:train_end]
    exog_train = exog_df.loc[train_start:train_end]

    y_test = y.loc[test_start:test_end]
    exog_test = exog_df.loc[test_start:test_end]
    
    model = ForecasterRecursive(
        regressor = XGBRegressor(
            eval_metric='mape', 
            learning_rate=0.1, 
            reg_alpha=0.005,
            max_depth=5, 
            n_estimators=1000,
            # n_estimators=100,
            random_state = 123, 
            n_jobs=-1
        ),
        lags = 5
    )
    model.fit(y = y_train, exog=exog_train)
    y_pred_test = model.predict(steps = len(y_test), exog=exog_test)
    metrics = {
        'rmse_train': np.sqrt(np.mean(np.square(model.in_sample_residuals_))),
        'mape_test': mean_absolute_percentage_error(y_test, y_pred_test),
    }
    model.fit(y = y, exog=exog_df)
    return model, metrics

models = {}
train_errors = []
test_errors = []
for splitKey, splitDs in train_splits.items(): 
    print(f'Key: {splitKey}')
    t0 = datetime.now()
    model, metrics = trainSplitModel(splitDs, train_start, train_end, test_start, test_end)
    models[splitKey] = model
    t1 = datetime.now()
    seconds = (t1 - t0).total_seconds()
    print(f'{splitKey} metrics: {metrics} ({seconds}s)')
    train_errors.append(metrics['rmse_train'])
    test_errors.append(metrics['mape_test'])

print('Average RMSE Train: ', np.mean(train_errors))
print('Average MAPE Test: ', np.mean(test_errors))


test_splits = split_data_by_features(X_test, columns=['country', 'store', 'product'])


test_splits[('Italy', 'Discount Stickers', 'Holographic Goose')]


%%time
preds = {}
for key, splitDs in test_splits.items():
    if key in models:
        preds[key] = models[key].predict(
            steps = len(splitDs), 
            exog = setupIndexAndNumericTypes(splitDs)
        )
    else: 
        print('Key not in models:', key)


preds[('Italy', 'Discount Stickers', 'Holographic Goose')]['2017-01-01']


df = {'id': [], 'num_sold': []}
for index, row in testDs.iterrows():
    key = (row['country'], row['store'], row['product'])
    date = row['date']
    df['id'].append(index)
    df['num_sold'].append(preds[key][date])

output = pd.DataFrame(df)
output.head()


output.to_csv('submission_skforecast.csv', index=False)
pd.read_csv('submission_skforecast.csv')


testAll = testDs.copy()
testAll = testAll.reset_index()
testAll['num_sold'] = output['num_sold']

_, ax = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
plot_timeseries(ax[0], testAll)
plot_timeseries_by_column(ax[1], testAll, 'country')
plot_timeseries_by_column(ax[2], testAll, 'store')
plot_timeseries_by_column(ax[3], testAll, 'product')
plt.tight_layout()
plt.show()


trainDs.head()


df = trainDs.pivot(index=['date', 'country'], columns=['store', 'product'], values='num_sold')
df.columns  = [ '_'.join(col) for col in df.columns.values]
df = df.reset_index()
countries = df['country'].unique()
countrySets = {}
for c in countries: 
    tmp = df[ df['country'] == c ]
    tmp = tmp.drop('country', axis=1)
    tmp = setupIndexAndNumericTypes(tmp)
    for col in tmp.columns: 
        # Fill missing value with the latest available data
        tmp[col].ffill(inplace = True)
        tmp[col].bfill(inplace=True)
        # Last chance filling
        tmp[col].fillna(value=0, inplace=True)
    countrySets[c] = tmp
countrySets['Canada'].head()


%%time 

def getExogeneousDataByCountry(dates, countries): 
    res = {}
    for c in set(countries): 
        tmp = pd.DataFrame({'date': pd.to_datetime(dates), 'country': [c] * len(dates)})
        p, colTr = makeExogeneousPipeline(tmp)
        rawExog = p.fit_transform(tmp)
        cols = [c.split('__')[-1] for c in colTr.get_feature_names_out()]
        exogDf = pd.DataFrame(rawExog, columns=columns)
        exogDf = setupIndexAndNumericTypes(exogDf.drop('country', axis=1))
        res[c] = exogDf
    return res

trainExogs = getExogeneousDataByCountry(trainDs['date'].unique(), trainDs['country'].unique())
testExogs = getExogeneousDataByCountry(testDs['date'].unique(), testDs['country'].unique())


trainExogs['Italy'].head()


from skforecast.recursive import ForecasterRecursiveMultiSeries
from xgboost import XGBRegressor

# Define the forecaster
forecaster = ForecasterRecursiveMultiSeries(
    # Add the sklearn regressor and lags
    regressor = XGBRegressor(random_state = 123),
    lags = 30
)

# Fit the model using train data
forecaster.fit(series = aux.loc[train_start:train_end])

# Predict the test period
predicted_test = forecaster.predict(steps = len(aux.loc[test_start:test_end]))


predicted_test


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

rmse_train = { k: np.sqrt(np.mean(np.square(v))) for k, v in forecaster.in_sample_residuals_.items() }
rmse_test = np.sqrt(mean_squared_error(aux.loc[test_start:test_end], predicted_test))
mape_test = mean_absolute_percentage_error(aux.loc[test_start:test_end], predicted_test)
print('RMSE Train:', rmse_train,'\nRMSE Test:', rmse_test, '\nMAPE Test:', mape_test)


fig, ax = plt.subplots(figsize=(16, 6))
aux.loc[test_start:test_end].plot(ax=ax, linestyle='dotted', label = "Test")
predicted_test.plot(ax=ax, label = 'Predicted DT')
ax.legend().remove()


%%time
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold, OneStepAheadFold, bayesian_search_forecaster_multiseries
from skforecast.exceptions import OneStepAheadValidationWarning
from xgboost import XGBRegressor
from datetime import datetime

def search_space(trial):
    search_space  = {
        'lags'          : trial.suggest_categorical('lags', [5]),
        'n_estimators'  : trial.suggest_int('n_estimators', 1000, 5000),
        'max_depth'     : trial.suggest_int('max_depth', 5, 10),
        'learning_rate' : trial.suggest_float('learning_rate', 0.001, 0.1),
        'reg_alpha'     : trial.suggest_float('reg_alpha', 0.001, 0.01),
    } 
    return search_space

def trainMultiSeriesModel(data, exog):
    model = ForecasterRecursiveMultiSeries(
        regressor = XGBRegressor(eval_metric='mape', random_state = 123),
        lags = 5
    )

    with warnings.catch_warnings():
        # warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.simplefilter('ignore', category=OneStepAheadValidationWarning)
        # With Refit and Increasing Train Size
        cv = TimeSeriesFold(
             steps = 10,
             initial_train_size = len(data.loc[train_start:train_end]),
             refit = True, # Change this to False to disable refit
             fixed_train_size = False, # Set this to true for fixed train size
             allow_incomplete_fold = True,
        )
        cv_search = OneStepAheadFold(initial_train_size = len(data.loc[train_start:train_end]),)
        train_result = bayesian_search_forecaster_multiseries(
            forecaster = model,
            series = data.loc[train_start:test_end],
            exog = exog.loc[train_start:test_end],
            cv = cv_search, 
            metric = 'mean_absolute_percentage_error',
            search_space = search_space,
            return_best = True,
            verbose = False,
            show_progress = True,
            n_jobs = -1
        )

        predicted_test = model.predict(
            steps=len(data.loc[test_start:test_end]), 
            exog=exog.loc[test_start:test_end])
        mape_test = mean_absolute_percentage_error(data.loc[test_start:test_end], predicted_test)
        print('\nMAPE Test:', mape_test)
        
    return model, train_result, mape_test

# Italy
# Without Exog Backtesting metric: 0.14282024416741462
# With Exog Backtesting metric: 0.0538118478429706
trainings = {}
for c in countrySets: 
    t0 = datetime.now()
    print(f"Starting Training for {c}...")
    model, train_result, mape_test = trainMultiSeriesModel(countrySets[c], trainExogs[c])
    trainings[c] = {
        'model': model, 
        'train_result': train_result, 
        'mape_test': mape_test,
    }
    t1 = datetime.now()
    print(f"Training for {c} finished in {(t1-t0).total_seconds()}, with MAPE: {mape_test}")





warnings.simplefilter('ignore', category=DeprecationWarning)
predicted_test = forecaster_multi_xgb.predict(steps=len(aux.loc[test_start:test_end]), exog=auxExog.loc[test_start:test_end])
mape_test = mean_absolute_percentage_error(aux.loc[test_start:test_end], predicted_test)
print('\nMAPE Test:', mape_test)









from skforecast.deep_learning.utils import create_and_compile_model
from keras.losses import MeanAbsolutePercentageError
from keras.optimizers import Adam

model = create_and_compile_model(
    series=aux.loc[train_start:train_end],
    levels=None, 
    lags=32,
    steps=1,
    recurrent_layer="LSTM",
    recurrent_units=4,
    dense_units=16,
    optimizer=Adam(learning_rate=0.01), 
    loss=MeanAbsolutePercentageError()
)
model.summary()


from skforecast.deep_learning import ForecasterRnn
from keras.callbacks import EarlyStopping

forecasterRnn = ForecasterRnn(
    regressor=model,
    levels=list(aux.columns),
    fit_kwargs={
        "epochs": 10,      # Number of epochs to train the model.
        "batch_size": 32,  # Batch size to train the model.
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=5)
        ],  # Callback to stop training when it is no longer learning.
        "series_val": aux.loc[test_start:test_end],  # Validation data for model training.
    },
)  


from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
cv = TimeSeriesFold(
     steps = 1,
     initial_train_size = len(aux.loc[train_start:train_end]),
     refit = True, # Change this to False to disable refit
     fixed_train_size = False, # Set this to true for fixed train size
     allow_incomplete_fold = True,
)
with warnings.catch_warnings():
    # warnings.simplefilter("ignore")
    '''
    metrics, predictions = backtesting_forecaster_multiseries(
        forecaster=forecasterRnn,
        series = aux.loc[train_start:test_end],
        exog = auxExog.loc[train_start:test_end],
        levels=forecasterRnn.levels,
        cv=cv,
        metric="mean_absolute_error",
        verbose=False, # Set to True for detailed information
    )
    '''





from sklearn.model_selection import train_test_split

targetColumn = 'num_sold'
trainWithoutTargetNA = trainDs.dropna(subset=[targetColumn], inplace=False)
y_all = trainWithoutTargetNA[targetColumn]
X_all = trainWithoutTargetNA.drop(targetColumn, axis=1)
X_train, X_eval, y_train, y_eval = train_test_split(X_all, y_all, test_size=.20)


preprocessor, colTr = makePreprocessingPipeline(X_train)


from xgboost import XGBRegressor

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('model', XGBRegressor(
            eval_metric='mape', 
            learning_rate=0.1, 
            reg_alpha=0.005,
            max_depth=5, 
            n_estimators=1000,
            # n_estimators=100,
            random_state = 123, 
            n_jobs=-1)
    ),
])
pipeline.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_percentage_error

y_eval_pred = pipeline.predict(X_eval)

print("Evaluation MAPE:", mean_absolute_percentage_error(y_eval, y_eval_pred))


y_eval_pred 


y_test = pipeline.predict(X_test)


output = pd.DataFrame({'id': testDs.index,
                       'num_sold': y_test})
output.to_csv('submission_xgb.csv', index=False)
pd.read_csv('submission_xgb.csv')


testAll = testDs.copy()
testAll['num_sold'] = y_test

_, ax = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
plot_timeseries(ax[0], testAll)
plot_timeseries_by_column(ax[1], testAll, 'country')
plot_timeseries_by_column(ax[2], testAll, 'store')
plot_timeseries_by_column(ax[3], testAll, 'product')
plt.tight_layout()
plt.show()




