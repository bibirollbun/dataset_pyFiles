import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns

import warnings 
warnings.filterwarnings("ignore")


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
print(sample_submission.shape)
sample_submission.head(2)


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
print(test.shape)
test.head(2)
# (98550, 5)


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
print(train.shape)
train.head(2)
# (230130, 6)


# missing data ratio
train_missing_ratio = train.isna().sum() / train.shape[0]
test_missing_ratio = test.isna().sum() / test.shape[0]

missing_data = pd.DataFrame({'Missing_Ratio_Train': train_missing_ratio, 
                             'Missing_Ratio_Test': test_missing_ratio})

print(missing_data.to_string(formatters={'Missing_Ratio_Train': '{:.6f}'.format, 
                                         'Missing_Ratio_Test': '{:.6f}'.format}))


# split datetime data into year, month, day, day_of_week
train['year'] = pd.to_datetime(train['date']).dt.year
train['month'] = pd.to_datetime(train['date']).dt.month
train['day'] = pd.to_datetime(train['date']).dt.day
train['day_of_week'] = pd.to_datetime(train['date']).dt.day_name()

test['year'] = pd.to_datetime(test['date']).dt.year
test['month'] = pd.to_datetime(test['date']).dt.month
test['day'] = pd.to_datetime(test['date']).dt.day
test['day_of_week'] = pd.to_datetime(test['date']).dt.day_name()

print(train.shape, test.shape)
train.head(2)


# data describe
train.describe()
# 2010 ~ 2016yr data


test.describe()
# 2017-01-01 ~ 2019-12-31 3yr 1095day (365*3)
# 98550 = 365*3*90(country*store*product)


# missing data ratio
train_missing_ratio = train.isna().sum() / train.shape[0]
test_missing_ratio = test.isna().sum() / test.shape[0]

missing_data = pd.DataFrame({'Missing_Ratio_Train': train_missing_ratio, 
                             'Missing_Ratio_Test': test_missing_ratio})

print(missing_data.to_string(formatters={'Missing_Ratio_Train': '{:.6f}'.format, 
                                         'Missing_Ratio_Test': '{:.6f}'.format}))


# Replacing Spaces with Underscores in Column Values
train["store"] = train["store"].str.replace(' ', '_')
train["product"] = train["product"].str.replace(' ', '_')

test["store"] = test["store"].str.replace(' ', '_')
test["product"] = test["product"].str.replace(' ', '_')

train.head(2)


# Table of num_Sold by Country and Year
grouped_data = train.pivot_table(index='year', columns='country', 
                                 values='num_sold', aggfunc='sum').astype(int)
grouped_data.head(2)


# num_sold Percentage by country
grouped_data_percentage = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100
print(grouped_data_percentage)


# plotting num_Sold by Country and Year

fig, ax = plt.subplots(figsize=(8, 3))
grouped_data.plot(kind='line', marker='o', ax=ax, title='num_Sold by Country and Year')
ax.set_xlabel('year', fontsize=10)
ax.set_ylabel('num_sold', fontsize=10)
ax.legend(title='country', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
plt.tight_layout()
plt.show()


# corrmat
'''corrmat = grouped_data.corr()
corrmat'''
# Canada is similar with Norway and Kenya is similar with Singapore.
# So let's fill Canada and Kenya's NaN based on Norway and Singapore data.


# num_sold ratio of the both countries 
'''ratio_C_N  = (grouped_data['Canada'] / grouped_data['Norway']).mean()
print(f"the ratio of Canada to Norway : {ratio_C_N:.6f}")

ratio_K_S  = (grouped_data['Kenya'] / grouped_data['Singapore']).mean()
print(f"the ratio of Kenya to Singapore : {ratio_K_S:.6f}")'''
# Let's fill NaN using this ratio.


def plot_num_sold_by_country(data, date_variable):    
    grouped_data = data.pivot_table(index=date_variable, columns='country', values='num_sold', 
                                     aggfunc='sum').astype(int)

    fig, ax = plt.subplots(figsize=(8, 3))
    grouped_data.plot(kind='line', marker='o', ax=ax, title=f'num_Sold by Country and {date_variable}')
    ax.set_xlabel(date_variable, fontsize=10)
    ax.set_ylabel('num_sold', fontsize=10)
    ax.legend(title='country', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.tight_layout()
    plt.show()

# plotting num_Sold by Country and Month
plot_num_sold_by_country(train, 'month')


# plotting num_Sold by Country and Day
plot_num_sold_by_country(train, 'day')


# plotting num_Sold by Country and Day_of_Week
plot_num_sold_by_country(train, 'day_of_week')
# Holiday effect/trand


# Table of num_Sold by Country and Date
grouped_data = train.pivot_table(index='date', columns='country', 
                                 values='num_sold', aggfunc='sum').astype(int)
grouped_data.head(2)


# plotting num_Sold by Country and Date
grouped_data = train.pivot_table(index='date', columns='country', values='num_sold', 
                                 aggfunc='sum').astype(int)

fig, ax = plt.subplots(figsize=(16, 6))
grouped_data.plot(kind='line', marker='o', ax=ax, title='num_Sold by Country and Date')
ax.set_xlabel('date', fontsize=10)
ax.set_ylabel('num_sold', fontsize=10)
ax.legend(title='country', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
plt.tight_layout()
plt.show()


# change to time serial data with country_store_product
train['country_store_product'] = train['country'] + '_' + train['store'] + '_' + train['product']
train_group = train.groupby(['date', 'country_store_product'], 
                            dropna=False).agg({'num_sold': lambda x: x.sum(skipna=False)}).reset_index()
train_series = train_group.pivot(index='date', columns='country_store_product', values='num_sold')

train_series.reset_index(inplace=True)
print(train_series.shape)
train_series.head(2)


train_series_nan = train_series.drop(['date'], axis=1)
fully_missing = train_series_nan.columns[train_series_nan.isna().all()].tolist()
print("fully_missing_column:", fully_missing)


# finding similar column
from scipy.stats import pearsonr
def find_similar_series(target_column, data, n_similar=2):

    correlations = {}
    for column in data.columns:
        if column != target_column and not data[column].isna().all():
            
            valid_data = data[column].dropna()
            if len(valid_data) >= 1:  
                correlation, _ = pearsonr(valid_data, valid_data)
                correlations[column] = correlation
    return sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:n_similar]


# Canada_Discount_Stickers_Holographic_Goose
target_column = 'Canada_Discount_Stickers_Holographic_Goose'
similar_series = find_similar_series(target_column, train_series_nan)
print("similar_comumn:", similar_series)


# Canada_Discount_Stickers_Holographic_Goose - NaN Value Estimation
finland_discount_columns = train_series.filter(regex='^Finland_Discount_Stickers_')
finland_discount_2016_sum = finland_discount_columns.tail(366).sum()
print(finland_discount_2016_sum)
print(finland_discount_2016_sum['Finland_Discount_Stickers_Holographic_Goose'] / 
      finland_discount_2016_sum['Finland_Discount_Stickers_Kerneler'])


train_series['Canada_Discount_Stickers_Holographic_Goose'] = (
    train_series['Canada_Discount_Stickers_Holographic_Goose'].fillna(
        train_series['Canada_Discount_Stickers_Kerneler'] * 0.3069283815817829))


# Kenya_Discount_Stickers_Holographic_Goose
target_column = 'Kenya_Discount_Stickers_Holographic_Goose'
similar_series = find_similar_series(target_column, train_series_nan)
print("similar_comumn:", similar_series)


train_series['Kenya_Discount_Stickers_Holographic_Goose'] = (
    train_series['Kenya_Discount_Stickers_Holographic_Goose'].fillna(
        train_series['Canada_Discount_Stickers_Kerneler'] * 0.3069283815817829 * 0.561924/16.685395 ))

train_series['Kenya_Discount_Stickers_Holographic_Goose']


# columns with NaN
train_missing_ratio = train_series.isna().sum() / train_series.shape[0]
train_missing_ratio_nonzero = train_missing_ratio[train_missing_ratio > 0]
missing_data = pd.DataFrame({'Missing_Ratio_Train': train_missing_ratio_nonzero})
missing_data.round(12)


# filling NaN by simple assumption 
'''train_series['Canada_Discount_Stickers_Holographic_Goose'] = (
    train_series['Canada_Discount_Stickers_Holographic_Goose'].fillna(
        train_series['Norway_Discount_Stickers_Holographic_Goose'] * 0.529217))

train_series['Canada_Stickers_for_Less_Holographic_Goose'] = (
    train_series['Canada_Stickers_for_Less_Holographic_Goose'].fillna(
        train_series['Norway_Stickers_for_Less_Holographic_Goose'] * 0.529217))

train_series['Kenya_Discount_Stickers_Holographic_Goose'] = (
    train_series['Kenya_Discount_Stickers_Holographic_Goose'].fillna(
        train_series['Singapore_Discount_Stickers_Holographic_Goose'] * 0.020726))

train_series['Kenya_Premium_Sticker_Mart_Holographic_Goose'] = (
    train_series['Kenya_Premium_Sticker_Mart_Holographic_Goose'].fillna(
        train_series['Singapore_Premium_Sticker_Mart_Holographic_Goose'] * 0.020726))

train_series['Kenya_Stickers_for_Less_Holographic_Goose'] = (
    train_series['Kenya_Stickers_for_Less_Holographic_Goose'].fillna(
        train_series['Singapore_Stickers_for_Less_Holographic_Goose'] * 0.020726))'''


# columns with NaN
train_missing_ratio = train_series.isna().sum() / train_series.shape[0]
train_missing_ratio_nonzero = train_missing_ratio[train_missing_ratio > 0]
missing_data = pd.DataFrame({'Missing_Ratio_Train': train_missing_ratio_nonzero})
missing_data.round(12)


# filling NaN by interpolate
train_series['Canada_Discount_Stickers_Holographic_Goose'] = (
    train_series['Canada_Discount_Stickers_Holographic_Goose'].interpolate())	

train_series['Canada_Discount_Stickers_Kerneler'] = (
    train_series['Canada_Discount_Stickers_Kerneler'].interpolate())

train_series['Canada_Premium_Sticker_Mart_Holographic_Goose'] = (
    train_series['Canada_Premium_Sticker_Mart_Holographic_Goose'].interpolate())	

train_series['Canada_Stickers_for_Less_Holographic_Goose'] = (
    train_series['Canada_Stickers_for_Less_Holographic_Goose'].interpolate())	

train_series['Kenya_Discount_Stickers_Holographic_Goose'] = (
    train_series['Kenya_Discount_Stickers_Holographic_Goose'].interpolate())

train_series['Kenya_Discount_Stickers_Kerneler'] = (
    train_series['Kenya_Discount_Stickers_Kerneler'].interpolate())

train_series['Kenya_Discount_Stickers_Kerneler_Dark_Mode'] = (
    train_series['Kenya_Discount_Stickers_Kerneler_Dark_Mode'].interpolate())

train_series['Kenya_Premium_Sticker_Mart_Holographic_Goose'] = (
    train_series['Kenya_Premium_Sticker_Mart_Holographic_Goose'].interpolate())

train_series['Kenya_Stickers_for_Less_Holographic_Goose'] = (
    train_series['Kenya_Stickers_for_Less_Holographic_Goose'].interpolate())

train_series.isna().sum().sum()


train_series.head(2)


train_series_2016 = train_series[train_series['date'] >= '2016-01-01']
print(train_series_2016.shape)
train_series_2016.head(2)


# select second column
train_series_2 = train_series[['date', 'Canada_Discount_Stickers_Kaggle']]
train_series_2.columns = ['ds', 'y']
train_series_2.head(2)


# splitting into train and validation data
train_2 = train_series_2[:-366] # training data 2010-01-01 ~ 2015-12-31
val_2 = train_series_2[-366:] # validation data 2016-01-01 ~ 2016-12-31
print(train_2.shape, val_2.shape)
val_2.head(2)


# call prophet 
from prophet import Prophet
prophet_model = Prophet()
prophet_model


# training
prophet_model.fit(train_2)


# Prophet Prediction
future = prophet_model.make_future_dataframe(periods=366)
forecast = prophet_model.predict(future)
print(forecast.shape)
forecast.head(2)


# Prophet Prediction
val_2[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]
print(val_2.shape)
val_2.head(2)


# Prophet Model's MAPE
from sklearn.metrics import mean_absolute_percentage_error
prophet_mape = mean_absolute_percentage_error(val_2['y'], val_2['yhat'])
print(f"Prophet_MAPE: {prophet_mape:.2%}")
# Prophet_MAPE: 5.69%


# compare with simple baseline prediction
val_2['baseline'] = train_2['y'][-366:].values
baseline_mape = mean_absolute_percentage_error(val_2['y'], val_2['baseline'])
print(f"Baseline_MAPE: {baseline_mape:.2%}")
# Baseline_MAPE: 14.77%


# Prediction Plot of Prophet & baseline
def plot_predictions(val_2):
    plt.figure(figsize=(12, 3))

    # Real Value(num_sold) Plot
    plt.plot(val_2['ds'], val_2['y'], label='num_Sold', color='black', linewidth=0.6)

    # Prophet Prediction Plot
    plt.plot(val_2['ds'], val_2['yhat'], 
             label='Prophet_Predict', color='blue', linewidth=0.6)
    
    plt.fill_between(val_2['ds'], val_2['yhat_lower'], val_2['yhat_upper'], 
                     color='blue', alpha=0.1, label='Prophet Uncertainty Intervals')

    # Baseline Prediction Plot
    plt.plot(val_2['ds'], val_2['baseline'], label='Baseline_Predict', 
             color='red', linewidth=0.6)

    plt.xlabel('Date')
    plt.ylabel('num_Sold')
    plt.title('Prophet vs Baseline Prediction')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(axis='y')

    plt.tight_layout()
    plt.show()

plot_predictions(val_2)


# Prophet_model plotting
fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)
prophet_model.plot(forecast, ax=ax)
plt.show()


# trend components
fig = prophet_model.plot_components(forecast)
fig.set_size_inches(8, 4)
plt.tight_layout() 
plt.show()


# trend changing point
from prophet.plot import add_changepoints_to_plot

fig = plt.figure(figsize=(16, 3))

ax = fig.add_subplot(111)
prophet_model.plot(forecast, ax=ax)

a = add_changepoints_to_plot(ax, prophet_model, forecast)

plt.show()


# Hyperparameter and CV for Prophet 
from itertools import product
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import logging
logging.getLogger("cmdstanpy").disabled = True

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0], 
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

MAPE = []

for params in all_params:
    m = Prophet(**params).fit(train_2)
    df_cv = cross_validation(m, initial = '730 days' , period = "180 days", 
                             horizon = '366 days', parallel = 'processes')
    df_p = performance_metrics(df_cv, rolling_window=1)
    MAPE.append(df_p['mape'].values[0])

tuning_results = pd.DataFrame(all_params)
tuning_results['mape'] = MAPE

best_params = all_params[np.argmin(MAPE)]
best_params
# 'changepoint_prior_scale': 0.5, 'holidays_prior_scale': 0.01, 'seasonality_prior_scale': 0.01


# training with optimized prophet model
prophet_tuned = Prophet(changepoint_prior_scale=0.5, 
                        holidays_prior_scale=0.01, 
                        seasonality_prior_scale=0.01)
prophet_tuned.fit(train_2)


# Prophet Prediction
future = prophet_tuned.make_future_dataframe(periods=366)
forecast = prophet_tuned.predict(future)
print(forecast.shape)
forecast.head(2)


# Prophet Prediction
val_2[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']]
print(val_2.shape)
val_2.head(2)


# Optimized Prophet Model's MAPE
tuned_prophet_mape = mean_absolute_percentage_error(val_2['y'], val_2['yhat'])
print(f"Tuned_Prophet_MAPE: {tuned_prophet_mape:.2%}")
# Tuned_Prophet_MAPE: 5.34% better result
# but do not apply to all time series because of overfitting


# all time series MAPE 

from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

def apply_prophet_model(train_series, column_name):
    train_data = train_series[['date', column_name]].rename(columns={'date': 'ds', column_name: 'y'})
    train = train_data[:-366]
    val = train_data[-366:]
    
    model = Prophet()
    model.fit(train)
    
    future = model.make_future_dataframe(periods=366)
    forecast = model.predict(future)
    
    val['yhat'] = forecast['yhat'][-366:].values
    
    mape = mean_absolute_percentage_error(val['y'], val['yhat'])
    
    return mape

results = {}

for column in train_series.columns[1:]:  
    mape = apply_prophet_model(train_series, column)
    results[column] = mape
    print(f"{column} - MAPE: {mape:.2%}")


# Predicted_num_sold df
from datetime import datetime, timedelta

def apply_prophet_model(train_series, column_name):
    train_data = train_series[['date', column_name]].rename(columns={'date': 'ds', column_name: 'y'})
    train = train_data[:-366]
    
    model = Prophet()
    model.fit(train)
    
    future = model.make_future_dataframe(periods=366)
    forecast = model.predict(future)
    
    return forecast['yhat']

start_date = datetime(2016, 1, 1)
num_days = 366  # 2016yr val data
results = {}
for column in train_series.columns[1:]:
    results[column] = apply_prophet_model(train_series, column)

forecast_df = pd.DataFrame()
for day in range(num_days):
    current_date = start_date + timedelta(days=day)
    daily_forecast = {'date': current_date}
    
    for column in train_series.columns[1:]:
        daily_forecast[column] = results[column][day]
    
    forecast_df = pd.concat([forecast_df, pd.DataFrame([daily_forecast])], ignore_index=True)

forecast_df = forecast_df.melt(id_vars=['date'], var_name='product', value_name='predicted_num_sold')
forecast_df = forecast_df.sort_values(['date', 'product']).reset_index(drop=True)

print(forecast_df.shape)
forecast_df.head(2)


# val_df
product_columns = train_series_2016.columns[1:]

val_df = pd.DataFrame(columns=['date', 'product', 'num_sold'])

for date in train_series_2016['date']:
    day_data = train_series_2016[train_series_2016['date'] == date]
    
    for product in product_columns:
        new_row = pd.DataFrame({
            'date': [date],
            'product': [product],
            'num_sold': [day_data[product].values[0]]
        })
        val_df = pd.concat([val_df, new_row], ignore_index=True)

print(val_df.shape)
val_df.head(2)


mape = mean_absolute_percentage_error(val_df['num_sold'], forecast_df['predicted_num_sold'])
print(f"Val_MAPE: {mape:.2%}")


test.describe()
# 2017-01-01 ~ 2019-12-31 3yr 1095day (365*3)
# 98550 = 365*3*90(country*store*product)


# Predicted_num_sold df
from datetime import datetime, timedelta

def apply_prophet_model(train_series, column_name):
    train_data = train_series[['date', column_name]].rename(columns={'date': 'ds', column_name: 'y'})
    train = train_data[:]
    
    model = Prophet()
    model.fit(train)
    
    future = model.make_future_dataframe(periods=1095)
    forecast = model.predict(future)
    
    return forecast['yhat']

start_date = datetime(2017, 1, 1)
num_days = 1095  # 2016yr val data
results = {}
for column in train_series.columns[1:]:
    results[column] = apply_prophet_model(train_series, column)

forecast_df = pd.DataFrame()
for day in range(num_days):
    current_date = start_date + timedelta(days=day)
    daily_forecast = {'date': current_date} 
    
    for column in train_series.columns[1:]:
        daily_forecast[column] = round(results[column][day])
    
    forecast_df = pd.concat([forecast_df, pd.DataFrame([daily_forecast])], ignore_index=True)

forecast_df = forecast_df.melt(id_vars=['date'], var_name='product', value_name='predicted_num_sold')
forecast_df = forecast_df.sort_values(['date', 'product']).reset_index(drop=True)

print(forecast_df.shape)
forecast_df.head(2)


submission = pd.DataFrame({'id': test.id, 'num_sold': forecast_df.predicted_num_sold})
print(submission.shape)
submission.head()


submission.to_csv('submission.csv', index=False)


submission = pd.read_csv('/kaggle/working/submission.csv')
print(submission.shape)
submission.tail(2)

