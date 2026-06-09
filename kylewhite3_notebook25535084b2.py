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


import pandas as pd

base = '/kaggle/input/m5-forecasting-accuracy/'

# sales by item/store/day
sales = pd.read_csv(base + 'sales_train_validation.csv')

# calendar features
calendar = pd.read_csv(base + 'calendar.csv')

# sell prices by item/store/week
prices = pd.read_csv(base + 'sell_prices.csv')

# glimpse
print(sales.shape, calendar.shape, prices.shape)
sales.head()


import pandas as pd
import plotly.express as px

# 1) Load the data (adjust folder name if needed)
base = '/kaggle/input/m5-forecasting-accuracy/'
sales    = pd.read_csv(base + 'sales_train_validation.csv')
calendar = pd.read_csv(base + 'calendar.csv')
prices   = pd.read_csv(base + 'sell_prices.csv')

# 2) Melt sales to “long” format and join calendar
sales_long = sales.melt(
    id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'],
    var_name='d', value_name='sales'
)
sales_long = sales_long.merge(
    calendar[['d','date','wm_yr_wk']], on='d', how='left'
)
sales_long['date'] = pd.to_datetime(sales_long['date'])

# ──────────────────────────────────────────────────────────────────────────
# A) Total daily sales trend
daily = sales_long.groupby('date')['sales'].sum().reset_index()
fig = px.line(daily, x='date', y='sales',
              title='Total Daily Sales Over Time')
fig.show()


# B) Daily sales by product category
cat_ts = sales_long.groupby(['date','cat_id'])['sales']\
                   .sum().reset_index()
fig = px.line(cat_ts, x='date', y='sales', color='cat_id',
              title='Daily Sales by Category')
fig.show()


# C) Distribution of cumulative sales per item
item_tot = sales_long.groupby('id')['sales'].sum().reset_index()
fig = px.histogram(item_tot, x='sales', nbins=50,
                   title='Total Sales per Item Distribution')
fig.update_layout(xaxis_title='Total Sales', yaxis_title='Count of Items')
fig.show()


# D) Sell-price distribution
fig = px.histogram(prices, x='sell_price', nbins=50,
                   title='Distribution of Sell Prices')
fig.update_layout(xaxis_title='Price', yaxis_title='Count')
fig.show()


# E) Price vs. sales for one sample SKU+store
#   (merge on store_id, item_id, wm_yr_wk)
merged = sales_long.merge(prices, on=['store_id','item_id','wm_yr_wk'], how='left')
sample = merged[(merged['item_id']=='HOBBIES_1_001') & (merged['store_id']=='CA_1')]
fig = px.scatter(sample, x='sell_price', y='sales',
                 hover_data=['date'], title='Price vs Sales: HOBBIES_1_001 @ CA_1')
fig.update_layout(xaxis_title='Sell Price', yaxis_title='Units Sold')
fig.show()


sales_long.head(10)


# 0) install AutoGluon if you haven’t already
!pip install autogluon.timeseries

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# 1) Load the M5 sales data (adjust path if needed)
base = '/kaggle/input/m5-forecasting-accuracy/'
sales = pd.read_csv(base + 'sales_train_validation.csv')
calendar = pd.read_csv(base + 'calendar.csv')

# Melt to long format and merge dates
sales_long = sales.melt(
    id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'],
    var_name='d', value_name='sales'
)
sales_long = sales_long.merge(
    calendar[['d','date']], on='d', how='left'
)
sales_long['date'] = pd.to_datetime(sales_long['date'])

# non-destructive: returns a new DataFrame
sales_long = sales_long.rename(columns={'id': 'item_id'})

# 2) Create a TimeSeriesDataFrame
ts_df = TimeSeriesDataFrame.from_data_frame(
    sales_long,
    timestamp_col='date',
    target_col='sales'
)

# 3) Initialize the predictor for a 13-day horizon
predictor = TimeSeriesPredictor(
    prediction_length=13,
    frequency='D',                # daily data
    eval_metric='MAE',            # feel free to choose MAPE, RMSE, etc.
    path='ag_ts_models/'          # where to save models
)

# 4) Fit with 3-fold time-series cross-validation
predictor.fit(
    train_data=ts_df,
    presets="fast_training",     # or 'high_quality', 'best_quality'
    time_limit=3600,              # in seconds, or omit for no limit
    num_validation_folds=3        # performs 3-fold rolling-origin CV
)

# 5) View cross-validation results
cv_results = predictor.fit_summary()   # includes CV scores per fold
print(cv_results)

# 6) Generate forecasts for the next 13 days
#    (predict() will produce a TSDF with dates beyond the last training date)
forecast_df = predictor.predict(ts_df)
print(forecast_df.head(20))

