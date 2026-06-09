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


df_train = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/train.csv', parse_dates=['date'])
df_test = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/test.csv', parse_dates=['date'])
df_sample = pd.read_csv('/kaggle/input/tabular-playground-series-jan-2022/sample_submission.csv')
print('='*50)
print('■Training data information\n')
df_train.info()
print('='*50)
print('■Test data information\n')
df_test.info()
print('='*50)
print('■Sample submission data information\n')
df_sample.info()
print('='*50)


print('='*50)
print('■The training data\n')
display(df_train)
print('='*50)
print('■The test data\n')
display(df_test)
print('='*50)
print('■The sample submission data\n')
display(df_sample)
print('='*50)


import matplotlib.pyplot as plt
!pip install --quiet japanize-matplotlib
import japanize_matplotlib
plt_font_family = plt.rcParams['font.family']
import seaborn as sns
from decimal import Decimal, ROUND_HALF_UP
import math
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import MSTL
!pip install -q pystan
!pip install -q prophet
from prophet import Prophet
!pip install -q optuna
import optuna
from contextlib import redirect_stdout


sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = [12, 9]
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = plt_font_family
init_notebook_mode(connected = True)
pd.set_option('display.max_columns', None)


print('='*50)
print('■Training Data\n')
display(df_train.isnull().sum())
print('='*50)
print('■Test Data\n')
display(df_test.isnull().sum())
print('='*50)


print('='*50)
print('■Training Data\n')
display(df_train['date'].unique())
print('='*50)
print('■Test Data\n')
display(df_test['date'].unique())
print('='*50)


check_continuity_train = all(pd.date_range(df_train['date'].min(), df_train['date'].max(), freq='d').values == df_train['date'].unique())
check_continuity_test = all(pd.date_range(df_test['date'].min(), df_test['date'].max(), freq='d').values == df_test['date'].unique())
print("Are the 'date' in the training data continuous at equal intervals?: {}".format('Yes' if check_continuity_train else 'No'))
print("Are the 'date' in the test data continuous at equal intervals?: {}".format('Yes' if check_continuity_test else 'No'))


features = ['country', 'store', 'product']
check_features_match = (df_train.drop_duplicates(features).sort_values(by=features).loc[:, features].values == df_test.drop_duplicates(features).sort_values(by=features).loc[:, features].values).all()
print("Are unique combinations of 'country', 'store', and 'product' matched between training and test data?: {}".format('Yes' if check_features_match else 'No'))


check_id = df_train.loc[:, 'row_id'].is_unique and df_test.loc[:, 'row_id'].is_unique
print("Is 'row_id' a unique variable?: {}".format('Yes' if check_id else 'No'))


objective_variable = 'num_sold'


df_train_original = df_train.copy()
df_test_original = df_test.copy()
df_test[objective_variable] = np.nan
df_train = df_train.pivot(index='date', columns=['country', 'store', 'product'], values=objective_variable)
df_test = df_test.pivot(index='date', columns=['country', 'store', 'product'], values=objective_variable)
print('='*50)
print('■Training Data\n')
display(df_train)
print('='*50)
print('■Test Data\n')
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    display(df_test)
print('='*50)


country = 'Finland'
store = 'KaggleMart'
product = 'Kaggle Mug'
print("Bin count based on Sturges' rule: {}".format(np.ceil(1 + math.log2(len(df_train.loc[:, (country, store, product)])))))


n_bins = 12
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    print('='*50)
    for country in np.unique([x[0] for x in df_train.columns]):
      for store in np.unique([x[1] for x in df_train.columns]):
        for product in np.unique([x[2] for x in df_train.columns]):
          mean = np.ceil(df_train.loc[:, (country, store, product)].mean()*pow(10, 5))/pow(10, 5)
          low_bound = np.ceil((mean - 3*df_train.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
          high_bound = np.ceil((mean + 3*df_train.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
          fig, ax = plt.subplots(1, 1, figsize=(20, 6))
          df_product = df_train.loc[:, (country, store, product)]
          ax.plot(df_product, 'b-.', lw=1)
          ax.axhline(mean, color='r', linestyle='-.', lw=2)
          ax.axhline(low_bound, color='r', linestyle='-.', lw=1.5)
          ax.axhline(high_bound, color='r', linestyle='-.', lw=1.5)
          x_min, x_max = ax.get_xticks().min(), ax.get_xticks().max()
          y_min, y_max = ax.get_yticks().min(), ax.get_yticks().max()
          ax.set_xlim(x_min, x_max)
          ax.set_ylim([min(low_bound, y_min)-(y_max-y_min)*0.05, max(high_bound, y_max)+(y_max-y_min)*0.05])
          ax.text(x=x_max+(x_max-x_min)*0.01, y=mean+(y_max-y_min)*0.01, s='mean: {}'.format(mean))
          ax.text(x=x_max+(x_max-x_min)*0.01, y=high_bound+(y_max-y_min)*0.01, s='mean + std * 3')
          ax.text(x=x_max+(x_max-x_min)*0.01, y=low_bound+(y_max-y_min)*0.01, s='mean - std * 3')
          ax.set_title('Time-series data of sales quantity (country: {}, store:{}, product:{})'.format(country, store, product), fontsize=16)
          ax.set_xlabel('date', fontsize=14)
          ax.set_ylabel(objective_variable, fontsize=14)
          plt.show()
          fig, ax = plt.subplots(1, 1, figsize = (20, 6))
          sns.histplot(data=df_product, bins=n_bins, ax=ax, kde=True, color='b', edgecolor='k')
          x_min = ax.get_xticks().min()
          x_max = ax.get_xticks().max()
          y_min = ax.get_yticks().min()
          y_max = ax.get_yticks().max()
          ax.set_ylim([y_min, y_max+(y_max-y_min)*0.05])
          ax.axvline(mean, color='r', linestyle='-.', lw=2)
          ax.text(x=mean+(x_max-x_min)*0.01, y=y_max, s='mean: {}'.format(mean))
          ax.axvline(low_bound, color='r', linestyle='-.', lw=1.5)
          ax.text(x=low_bound+(x_max-x_min)*0.01, y=y_max, s='mean - std * 3')
          ax.axvline(high_bound, color='r', linestyle='-.', lw=1.5)
          ax.text(x=high_bound-(x_max-x_min)*0.01, y=y_max, s='mean + std * 3')
          ax.set_title('Histogram of sales quantity (country: {}, store:{}, product:{})'.format(country, store, product), fontsize=16)
          ax.set_xlabel(objective_variable, fontsize=14)
          ax.set_ylabel('count', fontsize=14)
          plt.show()
          print('='*50)


df_skew_kurt = pd.DataFrame(index=df_train.columns, columns=['skew', 'kurt'])
df_skew_kurt['skew'] = df_train.skew()
df_skew_kurt['kurt'] = df_train.kurt()
display(df_skew_kurt)


df_train_before_log = df_train.copy()
df_train = np.log(df_train)


print('='*50)
for country in np.unique([x[0] for x in df_train.columns]):
  for store in np.unique([x[1] for x in df_train.columns]):
    for product in np.unique([x[2] for x in df_train.columns]):
      mean = np.ceil(df_train_before_log.loc[:, (country, store, product)].mean()*pow(10, 5))/pow(10, 5)
      low_bound = np.ceil((mean - 3*df_train_before_log.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
      high_bound = np.ceil((mean + 3*df_train_before_log.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
      fig, ax = plt.subplots(1, 1, figsize=(20, 6))
      df_product = df_train_before_log.loc[:, (country, store, product)]
      ax.plot(df_product, 'b-.', lw=1)
      ax.axhline(mean, color='r', linestyle='-.', lw=2)
      ax.axhline(low_bound, color='r', linestyle='-.', lw=1.5)
      ax.axhline(high_bound, color='r', linestyle='-.', lw=1.5)
      x_min, x_max = ax.get_xticks().min(), ax.get_xticks().max()
      y_min, y_max = ax.get_yticks().min(), ax.get_yticks().max()
      ax.set_xlim(x_min, x_max)
      ax.set_ylim([min(low_bound, y_min)-(y_max-y_min)*0.05, max(high_bound, y_max)+(y_max-y_min)*0.05])
      ax.text(x=x_max+(x_max-x_min)*0.01, y=mean+(y_max-y_min)*0.01, s='mean: {}'.format(mean))
      ax.text(x=x_max+(x_max-x_min)*0.01, y=high_bound+(y_max-y_min)*0.01, s='mean + std * 3')
      ax.text(x=x_max+(x_max-x_min)*0.01, y=low_bound+(y_max-y_min)*0.01, s='mean - std * 3')
      ax.set_title('Time-series data of sales quantity before logarithmic transformation (country: {}, store:{}, product:{})'.format(country, store, product), fontsize=16)
      ax.set_xlabel('date', fontsize=14)
      ax.set_ylabel(objective_variable, fontsize=14)
      plt.show()
      mean = np.ceil(df_train.loc[:, (country, store, product)].mean()*pow(10, 5))/pow(10, 5)
      low_bound = np.ceil((mean - 3*df_train.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
      high_bound = np.ceil((mean + 3*df_train.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
      fig, ax = plt.subplots(1, 1, figsize=(20, 6))
      df_product = df_train.loc[:, (country, store, product)]
      ax.plot(df_product, 'b-.', lw=1)
      ax.axhline(mean, color='r', linestyle='-.', lw=2)
      ax.axhline(low_bound, color='r', linestyle='-.', lw=1.5)
      ax.axhline(high_bound, color='r', linestyle='-.', lw=1.5)
      x_min, x_max = ax.get_xticks().min(), ax.get_xticks().max()
      y_min, y_max = ax.get_yticks().min(), ax.get_yticks().max()
      ax.set_xlim(x_min, x_max)
      ax.set_ylim([min(low_bound, y_min)-(y_max-y_min)*0.05, max(high_bound, y_max)+(y_max-y_min)*0.05])
      ax.text(x=x_max+(x_max-x_min)*0.01, y=mean+(y_max-y_min)*0.01, s='mean: {}'.format(mean))
      ax.text(x=x_max+(x_max-x_min)*0.01, y=high_bound+(y_max-y_min)*0.01, s='mean + std * 3')
      ax.text(x=x_max+(x_max-x_min)*0.01, y=low_bound+(y_max-y_min)*0.01, s='mean - std * 3')
      ax.set_title('Time-series data of sales quantity after logarithmic transformation (country: {}, store:{}, product:{})'.format(country, store, product), fontsize=16)
      ax.set_xlabel('date', fontsize=14)
      ax.set_ylabel(objective_variable, fontsize=14)
      plt.show()
      print('='*50)


n_bins = 12
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    print('='*50)
    for country in np.unique([x[0] for x in df_train.columns]):
      for store in np.unique([x[1] for x in df_train.columns]):
        for product in np.unique([x[2] for x in df_train.columns]):
          mean = np.ceil(df_train_before_log.loc[:, (country, store, product)].mean()*pow(10, 5))/pow(10, 5)
          low_bound = np.ceil((mean - 3*df_train_before_log.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
          high_bound = np.ceil((mean + 3*df_train_before_log.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
          fig, ax = plt.subplots(1, 1, figsize = (20, 6))
          df_product = df_train_before_log.loc[:, (country, store, product)]
          sns.histplot(data=df_product, bins=n_bins, ax=ax, kde=True, color='b', edgecolor='k')
          x_min = ax.get_xticks().min()
          x_max = ax.get_xticks().max()
          y_min = ax.get_yticks().min()
          y_max = ax.get_yticks().max()
          ax.set_ylim([y_min, y_max+(y_max-y_min)*0.05])
          ax.axvline(mean, color='r', linestyle='-.', lw=2)
          ax.text(x=mean+(x_max-x_min)*0.01, y=y_max, s='mean: {}'.format(mean))
          ax.axvline(low_bound, color='r', linestyle='-.', lw=1.5)
          ax.text(x=low_bound+(x_max-x_min)*0.01, y=y_max, s='mean - std * 3')
          ax.axvline(high_bound, color='r', linestyle='-.', lw=1.5)
          ax.text(x=high_bound-(x_max-x_min)*0.01, y=y_max, s='mean + std * 3')
          ax.set_title('Histogram of sales quantity before logarithmic transformation (country: {}, store:{}, product:{})'.format(country, store, product), fontsize=16)
          ax.set_xlabel(objective_variable, fontsize=14)
          ax.set_ylabel('count', fontsize=14)
          plt.show()
          mean = np.ceil(df_train.loc[:, (country, store, product)].mean()*pow(10, 5))/pow(10, 5)
          low_bound = np.ceil((mean - 3*df_train.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
          high_bound = np.ceil((mean + 3*df_train.loc[:, (country, store, product)].std())*pow(10, 5))/pow(10, 5)
          fig, ax = plt.subplots(1, 1, figsize = (20, 6))
          df_product = df_train.loc[:, (country, store, product)]
          sns.histplot(data=df_product, bins=n_bins, ax=ax, kde=True, color='b', edgecolor='k')
          x_min = ax.get_xticks().min()
          x_max = ax.get_xticks().max()
          y_min = ax.get_yticks().min()
          y_max = ax.get_yticks().max()
          ax.set_ylim([y_min, y_max+(y_max-y_min)*0.05])
          ax.axvline(mean, color='r', linestyle='-.', lw=2)
          ax.text(x=mean+(x_max-x_min)*0.01, y=y_max, s='mean: {}'.format(mean))
          ax.axvline(low_bound, color='r', linestyle='-.', lw=1.5)
          ax.text(x=low_bound+(x_max-x_min)*0.01, y=y_max, s='mean - std * 3')
          ax.axvline(high_bound, color='r', linestyle='-.', lw=1.5)
          ax.text(x=high_bound-(x_max-x_min)*0.01, y=y_max, s='mean + std * 3')
          ax.set_title('Histogram of sales quantity after logarithmic transformation (country: {}, store:{}, product:{})'.format(country, store, product), fontsize=16)
          ax.set_xlabel(objective_variable, fontsize=14)
          ax.set_ylabel('count', fontsize=14)
          plt.show()
          print('='*50)


df_skew_kurt_log = pd.DataFrame(index=df_train.columns, columns=['skew', 'kurt'])
df_skew_kurt_log['skew'] = df_train.skew()
df_skew_kurt_log['kurt'] = df_train.kurt()
df_skew_kurt_abs_diff = (np.abs(df_skew_kurt_log) - np.abs(df_skew_kurt)).rename(columns={'skew':'skew_abs_diff', 'kurt':'kurt_abs_diff'})
df_skew_kurt_log.join(df_skew_kurt, lsuffix='_After_log', rsuffix='_Before_log').join(df_skew_kurt_abs_diff)


df_count = pd.DataFrame(np.zeros(len(df_train)), index=df_train.index, columns=['count'])
for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      df_product = df_train.loc[:, (country, store, product)]
      mean = df_product.mean()
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        df_count.loc[:, 'count'] = df_count.loc[:, 'count'].values + (df_product >= (mean + 3 * df_product.std())).astype(float).values


fig = go.Figure()
fig.add_trace(go.Scatter(x=df_count.index,
                         y=df_count.loc[:, 'count'],
                         mode='lines',
                         opacity=0.5,
                         name='count'))
fig.update_layout(title=dict(text='The number of times the objective variable showed a peak equal to or higher than the mean value + standard deviation x 3',
                             font_size=16),
                  xaxis=dict(title=dict(text='Date', font_size=14),
                             rangeslider=dict(visible=True),
                             type='date'),
                  yaxis=dict(title=dict(text='Count', font_size=14)),
                  dragmode='pan')
fig.show()


list_year_end = [x for x in pd.concat([df_train, df_test], axis=0).index if x.month == 12 and x.day in range(27, 32)]
df_year_end = pd.DataFrame({'holiday': 'year_end', 'ds': list_year_end})
df_year_end


def adfuller_test(df, feature, level):
  """Performing the extended Dickey-Fuller test and verifying the results"""
  print('='*50)
  print(feature)
  print('='*50)
  dftest1 = adfuller(df.loc[:, feature])
  print('ADF Statistics: {}'.format(dftest1[0]))
  print('P-Value: {}'.format(dftest1[1]))
  print('Critical value:')
  for k, v in dftest1[4].items():
    if float(k[:-1])*0.01 == level:
      print('\t', k, v)
      result = 'Reject the null hypothesis (series data is non-stationary) and accept the alternative hypothesis (series data is stationary).' if dftest1[0] < v else 'The null hypothesis (series data is non-stationary) is not rejected.'
      print('Result:{}'.format(result))


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      adfuller_test(df_train_before_log, (country, store, product), 0.05)


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      adfuller_test(df_train_before_log.diff(1).dropna(), (country, store, product), 0.05)


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      adfuller_test(df_train, (country, store, product), 0.05)


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      adfuller_test(df_train.diff(1).dropna(), (country, store, product), 0.05)


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      print('='*50)
      print("Original series of '{}'({}, {}, {})".format(objective_variable, country, store, product))
      print('='*50)
      fig, axes = plt.subplots(1, 2, figsize=(20, 6))
      plot_acf(df_train_before_log.loc[:, (country, store, product)], lags=31, ax=axes[0])
      plot_pacf(df_train_before_log.loc[:, (country, store, product)], lags=31, ax=axes[1])
      plt.show()


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      print('='*50)
      print("Difference series (order 1) of '{}'({}, {}, {})".format(objective_variable, country, store, product))
      print('='*50)
      fig, axes = plt.subplots(1, 2, figsize=(20, 6))
      plot_acf(df_train_before_log.loc[:, (country, store, product)].diff(1).dropna(), lags=31, ax=axes[0])
      plot_pacf(df_train_before_log.loc[:, (country, store, product)].diff(1).dropna(), lags=31, ax=axes[1])
      plt.show()


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      print('='*50)
      print("Logarithmic series of '{}'({}, {}, {})".format(objective_variable, country, store, product))
      print('='*50)
      fig, axes = plt.subplots(1, 2, figsize=(20, 6))
      plot_acf(df_train.loc[:, (country, store, product)], lags=31, ax=axes[0])
      plot_pacf(df_train.loc[:, (country, store, product)], lags=31, ax=axes[1])
      plt.show()


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      print('='*50)
      print("Difference series (order 1) of logarithmic series of '{}'({}, {}, {})".format(objective_variable, country, store, product))
      print('='*50)
      fig, axes = plt.subplots(1, 2, figsize=(20, 6))
      plot_acf(df_train.loc[:, (country, store, product)].diff(1).dropna(), lags=31, ax=axes[0])
      plot_pacf(df_train.loc[:, (country, store, product)].diff(1).dropna(), lags=31, ax=axes[1])
      plt.show()


def test_seasonality(data, periods, feature=None):
  """Seasonality verification"""
  if isinstance(data, pd.core.frame.DataFrame):
    data = data.loc[:, feature]
#  mstl = MSTL(data, periods=periods, stl_kwargs=dict(trend=365*2+1, robust=True))
  mstl = MSTL(data, periods=periods, stl_kwargs=dict(trend=365*2+1))
  result = mstl.fit()
  result.plot()
  plt.show()
  resid_std = result.resid.std()
  print('-'*50)
  print('Strength of seasonal variation (standard deviation of seasonal variation / standard deviation of residuals)')
  print('-'*50)
  if isinstance(result.seasonal, pd.core.frame.DataFrame):
    for col in result.seasonal.columns:
      seasonal_std =result.seasonal[col].std()
      print('{}\t: {:.2f}'.format(col, seasonal_std/resid_std))
  else:
    seasonal_std =result.seasonal.std()
    print('seasonal\t: {:.2f}'.format(seasonal_std/resid_std))


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      print('='*50)
      print("Trend and Seasonal variation of '{}'({}, {}, {})".format(objective_variable, country, store, product))
      print('='*50)
      test_seasonality(df_train, [7, 365], (country, store, product))


df_param_tuning_train = df_train.loc[:'2017-12-31']
df_param_tuning_val = df_train.loc['2018-01-01':]


val_length = len(df_param_tuning_val)


def objective(trial):
  params = {'changepoint_prior_scale': trial.suggest_uniform('changepoint_prior_scale', 0.001, 0.5),
            'seasonality_prior_scale': trial.suggest_uniform('seasonality_prior_scale', 0.01, 10),
            'holidays_prior_scale': trial.suggest_uniform('holidays_prior_scale', 0.01, 10),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'changepoint_range': trial.suggest_discrete_uniform('changepoint_range', 0.8, 0.95, 0.001)}
  m = Prophet(holidays=df_year_end, **params)
  country_name = country
  if country_name == 'Finland':
    country_name = 'FI'
  elif country_name == 'Sweden':
    country_name = 'SE'
  elif country_name == 'Norway':
    country_name = 'NO'
  else:
    raise ValueError('Invalid country name:{}'.format(country_name))
  m.add_country_holidays(country_name=country_name)
  m.fit(df_temp_train)
  df_future = m.make_future_dataframe(periods=val_length, freq='d')
  df_pred = m.predict(df_future)
  preds = df_pred.tail(val_length)
#  val_rmse = root_mean_squared_error(df_temp_val['y'], preds['yhat'])
  val_rmse = np.sqrt(mean_squared_error(df_temp_val['y'], preds['yhat']))
  return val_rmse


dict_params = {}
for country in df_train_original.loc[:, 'country'].unique():
  dict_params_store = {}
  for store in df_train_original.loc[:, 'store'].unique():
    dict_params_product = {}
    for product in df_train_original.loc[:, 'product'].unique():
      df_temp_train = df_param_tuning_train.loc[:, (country, store, product)].rename('y').reset_index().rename(columns={'date': 'ds'})
      df_temp_val = df_param_tuning_val.loc[:, (country, store, product)].rename('y').reset_index().rename(columns={'date': 'ds'})
      study = optuna.create_study(direction='minimize')
      with redirect_stdout(open(os.devnull, 'w')):
        study.optimize(objective, n_trials=50)
      dict_params_product[product] = study.best_params.copy()
    dict_params_store[store] = dict_params_product
  dict_params[country] = dict_params_store


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      print('='*50)
      print("Best parameters ({}, {}, {}):".format(country, store, product))
      print(dict_params[country][store][product])
      print('='*50)


dict_model = {}
for country in df_train_original.loc[:, 'country'].unique():
  dict_model_store = {}
  for store in df_train_original.loc[:, 'store'].unique():
    dict_model_product = {}
    for product in df_train_original.loc[:, 'product'].unique():
      m = Prophet(holidays=df_year_end, **dict_params[country][store][product])
      country_name = country
      if country_name == 'Finland':
        country_name = 'FI'
      elif country_name == 'Sweden':
        country_name = 'SE'
      elif country_name == 'Norway':
        country_name = 'NO'
      else:
        raise ValueError('Invalid country name:{}'.format(country_name))
      m.add_country_holidays(country_name=country_name)
      dict_model_product[product] = m
    dict_model_store[store] = dict_model_product
  dict_model[country] = dict_model_store


test_length = len(df_test)
for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      m = dict_model[country][store][product]
      df_temp = df_train.loc[:, (country, store, product)].rename('y').reset_index().rename(columns={'date': 'ds'})
      m.fit(df_temp)
      df_future = m.make_future_dataframe(periods=test_length, freq='D')
      df_pred = m.predict(df_future)
      df_pred = df_pred.set_index('ds')
      df_pred.index.name = 'date'
      df_test.loc[:, (country, store, product)] = df_pred['yhat']


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      fig, ax = plt.subplots(1, 1, figsize=(20, 6))
      df_product = np.exp(df_train.loc[:, (country, store, product)])
      df_product_pred = np.exp(df_test.loc[:, (country, store, product)])
      ax.plot(df_product, 'b-.', lw=1, label='Training data(actual)')
      ax.plot(df_product_pred, 'g-.', lw=1, label='Test data(predicted)')
      x_min, x_max = ax.get_xticks().min(), ax.get_xticks().max()
      y_min, y_max = ax.get_yticks().min(), ax.get_yticks().max()
      ax.set_ylim([min(low_bound, y_min)-(y_max-y_min)*0.05, max(high_bound, y_max)+(y_max-y_min)*0.05])
      ax.set_title('Time-series data of sales quantity (country: {}, store:{}, product:{})'.format(country, store, product), fontsize=16)
      ax.set_xlabel('date', fontsize=14)
      ax.set_ylabel(objective_variable, fontsize=14)
      ax.legend(loc='best')
      plt.show()


df_train_holdout_train = df_train.loc[:'2017-12-31']
df_train_holdout_val = df_train.loc['2018-01-01':]


with warnings.catch_warnings():
  warnings.simplefilter('ignore', category=FutureWarning)
  df_temp = df_train_holdout_val.copy()
  df_temp.loc[:, :] = np.nan
  df_train_holdout_val = df_train_holdout_val.join(df_temp, rsuffix='_pred')


dict_model_for_eval = {}
for country in df_train_original.loc[:, 'country'].unique():
  dict_model_store_for_eval = {}
  for store in df_train_original.loc[:, 'store'].unique():
    dict_model_product_for_eval = {}
    for product in df_train_original.loc[:, 'product'].unique():
      m = Prophet(holidays=df_year_end, **dict_params[country][store][product])
      country_name = country
      if country_name == 'Finland':
        country_name = 'FI'
      elif country_name == 'Sweden':
        country_name = 'SE'
      elif country_name == 'Norway':
        country_name = 'NO'
      else:
        raise ValueError('Invalid country name:{}'.format(country_name))
      m.add_country_holidays(country_name=country_name)
      dict_model_product_for_eval[product] = m
    dict_model_store_for_eval[store] = dict_model_product_for_eval
  dict_model_for_eval[country] = dict_model_store_for_eval


test_length_for_eval = len(df_train_holdout_val)
for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      m = dict_model_for_eval[country][store][product]
      df_temp = df_train_holdout_train.loc[:, (country, store, product)].rename('y').reset_index().rename(columns={'date': 'ds'})
      m.fit(df_temp)
      df_future = m.make_future_dataframe(periods=test_length_for_eval, freq='D')
      df_pred = m.predict(df_future)
      df_pred = df_pred.set_index('ds')
      df_pred.index.name = 'date'
      df_train_holdout_val.loc[:, ('{}_pred'.format(country), store, product)] = df_pred['yhat']


def smape(y_true, y_pred):
  """Symmetric mean absolute percentage error"""
  return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


dict_metrics_for_eval = {}
list_metrics = ['SMAPE', 'RMSE', 'MAE', 'MAPE', 'R2']
for country in df_train_original.loc[:, 'country'].unique():
  dict_metrics_store = {}
  for store in df_train_original.loc[:, 'store'].unique():
    dict_metrics_product = {}
    for product in df_train_original.loc[:, 'product'].unique():
      dict_metrics = {}
      dict_metrics['SMAPE'] = smape(np.exp(df_train_holdout_val.loc[:, (country, store, product)]), np.exp(df_train_holdout_val.loc[:, ('{}_pred'.format(country), store, product)]))
#      dict_metrics['RMSE'] = root_mean_squared_error(np.exp(df_train_holdout_val.loc[:, (country, store, product)]), np.exp(df_train_holdout_val.loc[:, ('{}_pred'.format(country), store, product)]))
      dict_metrics['RMSE'] = np.sqrt(mean_squared_error(np.exp(df_train_holdout_val.loc[:, (country, store, product)]), np.exp(df_train_holdout_val.loc[:, ('{}_pred'.format(country), store, product)])))
      dict_metrics['MAE'] = mean_absolute_error(np.exp(df_train_holdout_val.loc[:, (country, store, product)]), np.exp(df_train_holdout_val.loc[:, ('{}_pred'.format(country), store, product)]))
      dict_metrics['MAPE'] = mean_absolute_percentage_error(np.exp(df_train_holdout_val.loc[:, (country, store, product)]), np.exp(df_train_holdout_val.loc[:, ('{}_pred'.format(country), store, product)]))*100
      dict_metrics['R2'] = r2_score(np.exp(df_train_holdout_val.loc[:, (country, store, product)]), np.exp(df_train_holdout_val.loc[:, ('{}_pred'.format(country), store, product)]))
      dict_metrics_product[product] = dict_metrics
    dict_metrics_store[store] = dict_metrics_product
  dict_metrics_for_eval[country] = dict_metrics_store


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      print('='*50)
      print('Evaluation index of the evaluation model of num_sold({}, {}, {})'.format(country, store, product))
      print('='*50)
      for metric in list_metrics:
        print('{} =\t{}'.format(metric, np.ceil(dict_metrics_for_eval[country][store][product][metric]*pow(10, 5))/pow(10, 5)))


for country in df_train_original.loc[:, 'country'].unique():
  for store in df_train_original.loc[:, 'store'].unique():
    for product in df_train_original.loc[:, 'product'].unique():
      fig, ax = plt.subplots(1, 1, figsize=(20, 6))
      df_product = np.exp(df_train_holdout_train.loc[:, (country, store, product)])
      df_product_pred = np.exp(df_train_holdout_val.loc[:, [(country, store, product), ('{}_pred'.format(country), store, product)]])
      ax.plot(df_product, 'b-.', lw=1, label='Training data(actual)')
      ax.plot(df_product_pred.loc[:, (country, store, product)], 'b-.', lw=1, label='Validation data(actual)')
      ax.plot(df_product_pred.loc[:, ('{}_pred'.format(country), store, product)], 'g-.', lw=1, label='Validation data(predicted)')
      x_min, x_max = ax.get_xticks().min(), ax.get_xticks().max()
      y_min, y_max = ax.get_yticks().min(), ax.get_yticks().max()
      ax.set_ylim([min(low_bound, y_min)-(y_max-y_min)*0.05, max(high_bound, y_max)+(y_max-y_min)*0.05])
      ax.set_title('Time-series data of sales quantity (country: {}, store:{}, product:{})'.format(country, store, product), fontsize=16)
      ax.set_xlabel('date', fontsize=14)
      ax.set_ylabel(objective_variable, fontsize=14)
      ax.legend(loc='upper left')
      plt.show()


df_output = pd.merge(df_test_original, df_test.stack(['country', 'store', 'product']).rename('num_sold').reset_index(), on=['date', 'country', 'store', 'product'])
df_output = df_output.loc[:, ['row_id', 'num_sold']]
df_output = df_output.set_index('row_id').sort_index()
df_output = np.exp(df_output)


df_output.to_csv('submission.csv')

