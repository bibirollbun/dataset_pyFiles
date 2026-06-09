!pip install -U autogluon > /dev/null


# show versions
from autogluon.core.utils import show_versions
show_versions()


# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# ML
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


# configs

# show more/all columns:
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 100)

# random seed
my_random_seed = 180

# colors
default_color_1 = 'darkblue'
default_color_2 = 'darkgreen'
default_color_3 = 'darkred'


# load data
df_train = pd.read_csv('../input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('../input/playground-series-s5e1/test.csv')
df_sub = pd.read_csv('../input/playground-series-s5e1/sample_submission.csv')


# preview
df_train.head(10)


# convert dates
df_train.date = pd.to_datetime(df_train.date)
df_test.date = pd.to_datetime(df_test.date)


# structure training data
df_train.info()


# structure test data
df_test.info()


# additional features
df_train['year'] = df_train.date.dt.year
df_train['month'] = df_train.date.dt.month
df_train['day'] = df_train.date.dt.day

df_test['year'] = df_test.date.dt.year
df_test['month'] = df_test.date.dt.month
df_test['day'] = df_test.date.dt.day


# preview
df_train.head()


# basic stats - training
df_train.describe(include='all')


# basic stats - test
df_test.describe(include='all')


# define features
features_cat = ['country', 'store', 'product']
features_time = ['year', 'month', 'day']


# list of countries
countries = df_train['country'].value_counts().index.to_list()
print(countries)


# list of stores
stores = df_train['store'].value_counts().index.to_list()
print(stores)


# list of products
products = df_train['product'].value_counts().index.to_list()
print(products)


target = 'num_sold'


# missing values?
df_train[target].isna().sum()


# fill missings with 0
df_train[target] = df_train[target].fillna(0)


# plot histogram
plt.figure(figsize=(10,3))
plt.hist(df_train[target], bins=50, color=default_color_3)
plt.title(target)
plt.grid()
plt.show()


# full time series
plt.figure(figsize=(15,3))
plt.scatter(df_train.date, df_train[target], color=default_color_3, s=1)
plt.grid()
plt.show()


# full time series - colored by country
plt.figure(figsize=(15,6))
sns.scatterplot(data=df_train, x='date', y=target, 
                hue=df_train['country'], s=1)
plt.grid()
plt.show()


# list of years (sorted)
years_train = df_train.year.value_counts().sort_index().index.to_list()
print(years_train)


# create combined key
df_train['combo_key'] = df_train['country'] + '|' + df_train['store'] + '|' + df_train['product']
df_test['combo_key'] = df_test['country'] + '|' + df_test['store'] + '|' + df_test['product']


# overall mean
df_train[target].mean()


# mean by year
stats_year = df_train.groupby(by=['year'])[target].mean()
stats_year


# plot target mean by year
plt.figure(figsize=(8,3))
plt.scatter(stats_year.index, stats_year, color=default_color_3)
plt.title('Mean of target by year')
plt.grid()
plt.show()


# mean by month
stats_month = df_train.groupby(by=['month'])[target].mean()
stats_month


# plot target mean by month
plt.figure(figsize=(8,3))
plt.scatter(stats_month.index, stats_month, color=default_color_3)
plt.title('Mean of target by month')
plt.grid()
plt.show()


# mean by country
df_train.groupby(by=['country'])[target].mean()


# mean by store
df_train.groupby(by=['store'])[target].mean()


# mean by product
df_train.groupby(by=['product'])[target].mean()


# mean by combination country|store|product
stats_combo = df_train.groupby(by=['combo_key'])[target].mean()
stats_combo


# create table of static features
static_df = df_train[['combo_key', 'country', 'store', 'product']].drop_duplicates()
static_df.head()

# prepare training data for AutoGluon
train_data = TimeSeriesDataFrame.from_data_frame(
    df_train[['combo_key', 'date', 'month', 'day', target]], # just using relevant columns
    id_column = 'combo_key',
    timestamp_column = 'date',
    static_features_df = static_df # include static features
)


# define predictor and fit model
time_limit_in_secs = 2 * 60 * 60

predictor = TimeSeriesPredictor(
    prediction_length = 3*365, # three years
    eval_metric = 'MAPE',
    target = target,
    quantile_levels = [],
    path = 'autogluon_ts_models')

models = predictor.fit(train_data,                       
                       presets = 'best_quality',
                       time_limit = time_limit_in_secs,
                       enable_ensemble = True)


# show leaderboard
models.leaderboard()


# calc predictions (as autogluon.timeseries.dataset.ts_dataframe.TimeSeriesDataFrame)
predictions = predictor.predict(train_data)
predictions


# convert result including translation of multi-index to columns & renaming columns
df_pred = predictions.to_data_frame().reset_index()
df_pred = df_pred.rename(columns = { 'item_id' : 'combo_key',
                                     'timestamp' : 'date',
                                     'mean' : 'prediction'})
df_pred.head()


# map predictions to correct rows
df_test = pd.merge(df_test, df_pred, on=['combo_key', 'date'], how='left')
df_test.head()


# post processing
df_test['prediction'] = df_test['prediction'].clip(0,6000)


# plot predictions
plt.figure(figsize=(10,3))
plt.hist(df_test['prediction'], bins=25, color=default_color_3)
plt.title('Predictions')
plt.grid()
plt.show()


# as time series
plt.figure(figsize=(15,3))
plt.scatter(df_test['date'], df_test['prediction'], s=1, color=default_color_3)
plt.title('Predictions as time series')
plt.grid()
plt.show()


# prepare submission
df_sub[target] = df_test['prediction']
df_sub.head(10)


# and save submission file
df_sub.to_csv('submission.csv', index=False)

