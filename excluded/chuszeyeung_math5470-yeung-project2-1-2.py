!pip install prophet


from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score


data = pd.read_csv("/kaggle/input/sales-train-evaluation/sales_train_evaluation.csv")


data.head(3)


# Melt the dataset to a long format
data_long = data.melt(id_vars=['store_id', 'dept_id', 'item_id','id','cat_id','state_id'],
                      var_name='day',
                      value_name='sales')

# Convert `day` column to numerical format
data_long['day'] = data_long['day'].str.extract('(\d+)').astype(int)

# Sort by store, department, item, and day
data_long = data_long.sort_values(['store_id', 'dept_id', 'item_id','id','cat_id','state_id', 'day'])


data_long.head(3)


data_long['dept_id'].unique()


data_long.head(3)


# As checked from spreadsheet "calendar", day 1 is 2011-01-29, then add this columne accordingly
start_date = pd.Timestamp('2011-01-29')
data_long['date'] = start_date + pd.to_timedelta(data_long['day'] - 1, unit='d')


data_long.head(3)


# To sum up the 'sale' for each dept_id
df = data_long.groupby(['dept_id','day','date'])['sales'].sum().reset_index()



df.head(3)


FOODS_1 = df[df['dept_id'] == 'FOODS_1'].copy()
FOODS_2 = df[df['dept_id'] == 'FOODS_2'].copy()
FOODS_3 = df[df['dept_id'] == 'FOODS_3'].copy()
HOBBIES_1 = df[df['dept_id'] == 'HOBBIES_1'].copy()
HOBBIES_2 = df[df['dept_id'] == 'HOBBIES_2'].copy()
HOUSEHOLD_1 = df[df['dept_id'] == 'HOUSEHOLD_1'].copy()
HOUSEHOLD_2 = df[df['dept_id'] == 'HOUSEHOLD_2'].copy()



FOODS_1.head(3)


FOODS_1['date'] = pd.to_datetime(FOODS_1['date'])
FOODS_2['date'] = pd.to_datetime(FOODS_2['date'])
FOODS_3['date'] = pd.to_datetime(FOODS_3['date'])
HOBBIES_1['date'] = pd.to_datetime(HOBBIES_1['date'])
HOBBIES_2['date'] = pd.to_datetime(HOBBIES_2['date'])
HOUSEHOLD_1['date'] = pd.to_datetime(HOUSEHOLD_1['date'])
HOUSEHOLD_2['date'] = pd.to_datetime(HOUSEHOLD_2['date'])


FOODS_2.head(3)


FOODS_1.set_index('date', inplace=True)
FOODS_2.set_index('date', inplace=True)
FOODS_3.set_index('date', inplace=True)
HOBBIES_1.set_index('date', inplace=True)
HOBBIES_2.set_index('date', inplace=True)
HOUSEHOLD_1.set_index('date', inplace=True)
HOUSEHOLD_2.set_index('date', inplace=True)


FOODS_2.head(3)


# set the date as index
FOODS_1['Dates'] = FOODS_1.index
FOODS_2['Dates'] = FOODS_2.index
FOODS_3['Dates'] = FOODS_3.index
HOBBIES_1['Dates'] = HOBBIES_1.index
HOBBIES_2['Dates'] = HOBBIES_2.index
HOUSEHOLD_1['Dates'] = HOUSEHOLD_1.index
HOUSEHOLD_2['Dates'] = HOUSEHOLD_2.index


FOODS_3.head(3)


df_FOODS_1 = FOODS_1[['sales', 'Dates']].copy()
df_FOODS_2 = FOODS_2[['sales', 'Dates']].copy()
df_FOODS_3 = FOODS_3[['sales', 'Dates']].copy()
df_HOBBIES_1 = HOBBIES_1[['sales', 'Dates']].copy()
df_HOBBIES_2 = HOBBIES_2[['sales', 'Dates']].copy()
df_HOUSEHOLD_1 = HOUSEHOLD_1[['sales', 'Dates']].copy()
df_HOUSEHOLD_2 = HOUSEHOLD_2[['sales', 'Dates']].copy()


df_HOUSEHOLD_1.head(3)


# make the dataframes to fit the format of prophet model, 'y' is sales, 'ds' is date
df_FOODS_1.columns = ['y', 'ds']
df_FOODS_2.columns = ['y', 'ds']
df_FOODS_3.columns = ['y', 'ds']
df_HOBBIES_1.columns = ['y', 'ds']
df_HOBBIES_2.columns = ['y', 'ds']
df_HOUSEHOLD_1.columns = ['y', 'ds']
df_HOUSEHOLD_2.columns = ['y', 'ds']


df_HOUSEHOLD_1.head(3)


df_FOODS_1.head(3)


df_FOODS_1["y"].plot(figsize=(15, 5));
plt.xlabel('Time/Date', fontsize=12)  # X-axis label
plt.ylabel('Sales', fontsize=12)      # Y-axis label
plt.title('FOODS_1 Sales Over Time', fontsize=14)  # Title


df_FOODS_2["y"].plot(figsize=(15, 5));
plt.xlabel('Time/Date', fontsize=12)  # X-axis label
plt.ylabel('Sales', fontsize=12)      # Y-axis label
plt.title('FOODS_2 Sales Over Time', fontsize=14)  # Title


df_FOODS_3["y"].plot(figsize=(15, 5));
plt.xlabel('Time/Date', fontsize=12)  # X-axis label
plt.ylabel('Sales', fontsize=12)      # Y-axis label
plt.title('FOODS_3 Sales Over Time', fontsize=14)  # Title


df_HOBBIES_1["y"].plot(figsize=(15, 5));
plt.xlabel('Time/Date', fontsize=12)  # X-axis label
plt.ylabel('Sales', fontsize=12)      # Y-axis label
plt.title('HOBBIES_1 Sales Over Time', fontsize=14)  # Title


df_HOBBIES_2["y"].plot(figsize=(15, 5));
plt.xlabel('Time/Date', fontsize=12)  # X-axis label
plt.ylabel('Sales', fontsize=12)      # Y-axis label
plt.title('HOBBIES_2 Sales Over Time', fontsize=14)  # Title


df_HOUSEHOLD_1["y"].plot(figsize=(15, 5));
plt.xlabel('Time/Date', fontsize=12)  # X-axis label
plt.ylabel('Sales', fontsize=12)      # Y-axis label
plt.title('HOUSEHOLD_1 Sales Over Time', fontsize=14)  # Title


df_HOUSEHOLD_2["y"].plot(figsize=(15, 5));
plt.xlabel('Time/Date', fontsize=12)  # X-axis label
plt.ylabel('Sales', fontsize=12)      # Y-axis label
plt.title('HOUSEHOLD_2 Sales Over Time', fontsize=14)  # Title


# split the training set and testing set for the forecast result comparasion
split_date = '2016-04-24'
train_FOODS_1 = df_FOODS_1.loc[df_FOODS_1.index <= split_date].copy()
test_FOODS_1 = df_FOODS_1.loc[df_FOODS_1.index > split_date].copy()

train_FOODS_2 = df_FOODS_2.loc[df_FOODS_2.index <= split_date].copy()
test_FOODS_2 = df_FOODS_2.loc[df_FOODS_2.index > split_date].copy()

train_FOODS_3 = df_FOODS_3.loc[df_FOODS_3.index <= split_date].copy()
test_FOODS_3 = df_FOODS_3.loc[df_FOODS_3.index > split_date].copy()

train_HOBBIES_1 = df_HOBBIES_1.loc[df_HOBBIES_1.index <= split_date].copy()
test_HOBBIES_1 = df_HOBBIES_1.loc[df_HOBBIES_1.index > split_date].copy()

train_HOBBIES_2 = df_HOBBIES_2.loc[df_HOBBIES_2.index <= split_date].copy()
test_HOBBIES_2 = df_HOBBIES_2.loc[df_HOBBIES_2.index > split_date].copy()

train_HOUSEHOLD_1 = df_HOUSEHOLD_1.loc[df_HOUSEHOLD_1.index <= split_date].copy()
test_HOUSEHOLD_1 = df_HOUSEHOLD_1.loc[df_HOUSEHOLD_1.index > split_date].copy()

train_HOUSEHOLD_2 = df_HOUSEHOLD_2.loc[df_HOUSEHOLD_2.index <= split_date].copy()
test_HOUSEHOLD_2 = df_HOUSEHOLD_2.loc[df_HOUSEHOLD_2.index > split_date].copy()



m_FOODS_1 = Prophet(growth='linear')
m_FOODS_1.add_country_holidays(country_name='US')
m_FOODS_1.fit(train_FOODS_1)

m_FOODS_2 = Prophet(growth='linear', seasonality_mode='multiplicative')
m_FOODS_2.add_country_holidays(country_name='US')
m_FOODS_2.fit(train_FOODS_2)

m_FOODS_3 = Prophet(growth='linear', seasonality_mode='multiplicative')
m_FOODS_3.add_country_holidays(country_name='US')
m_FOODS_3.fit(train_FOODS_3)

m_HOBBIES_1 = Prophet(growth='linear', seasonality_mode='multiplicative')
m_HOBBIES_1.add_country_holidays(country_name='US')
m_HOBBIES_1.fit(train_HOBBIES_1)

m_HOBBIES_2 = Prophet(growth='linear', seasonality_mode='multiplicative')
m_HOBBIES_2.add_country_holidays(country_name='US')
m_HOBBIES_2.fit(train_HOBBIES_2)

m_HOUSEHOLD_1 = Prophet(growth='linear', seasonality_mode='multiplicative')
m_HOUSEHOLD_1.add_country_holidays(country_name='US')
m_HOUSEHOLD_1.fit(train_HOUSEHOLD_1)

m_HOUSEHOLD_2 = Prophet(growth='linear', seasonality_mode='multiplicative')
m_HOUSEHOLD_2.add_country_holidays(country_name='US')
m_HOUSEHOLD_2.fit(train_HOUSEHOLD_2)


future_FOODS_1 = m_FOODS_1.make_future_dataframe(periods=28, freq='D')
future_FOODS_2 = m_FOODS_2.make_future_dataframe(periods=28)
future_FOODS_3 = m_FOODS_3.make_future_dataframe(periods=28)
future_HOBBIES_1 = m_HOBBIES_1.make_future_dataframe(periods=28)
future_HOBBIES_2 = m_HOBBIES_2.make_future_dataframe(periods=28)
future_HOUSEHOLD_1 = m_HOUSEHOLD_1.make_future_dataframe(periods=28)
future_HOUSEHOLD_2 = m_HOUSEHOLD_2.make_future_dataframe(periods=28)




future_FOODS_1.tail()


forecast_FOODS_1 = m_FOODS_1.predict(future_FOODS_1)
forecast_FOODS_2= m_FOODS_2.predict(future_FOODS_2)
forecast_FOODS_3 = m_FOODS_3.predict(future_FOODS_3)
forecast_HOBBIES_1 = m_HOBBIES_1.predict(future_HOBBIES_1)
forecast_HOBBIES_2  = m_HOBBIES_2.predict(future_HOBBIES_2 )
forecast_HOUSEHOLD_1 = m_HOUSEHOLD_1.predict(future_HOUSEHOLD_1)
forecast_HOUSEHOLD_2 = m_HOUSEHOLD_2.predict(future_HOUSEHOLD_2)


forecast_FOODS_1.tail(28)


fig = m_FOODS_1.plot(forecast_FOODS_1);
a = add_changepoints_to_plot(fig.gca(), m_FOODS_1, forecast_FOODS_1)

# Add actual test data points
plt.scatter(test_FOODS_1['ds'], test_FOODS_1['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for FOODS_1', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for FOODS_1', fontsize=14)

# Add legend
plt.legend(['Actual', 'Forecast', 'Uncertainty Interval', 'Actual Data'])

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


m_FOODS_1.plot_components(forecast_FOODS_1);


fig = m_FOODS_2.plot(forecast_FOODS_2);
a = add_changepoints_to_plot(fig.gca(), m_FOODS_2, forecast_FOODS_2)
# Add actual test data points
plt.scatter(test_FOODS_2['ds'], test_FOODS_2['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for FOODS_2', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for FOODS_2', fontsize=14)

# Add legend
plt.legend(['Actual', 'Forecast', 'Uncertainty Interval', 'Actual Data'])

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m_FOODS_3.plot(forecast_FOODS_3);
a = add_changepoints_to_plot(fig.gca(), m_FOODS_3, forecast_FOODS_3)
# Add actual test data points
plt.scatter(test_FOODS_3['ds'], test_FOODS_3['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for FOODS_1', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for FOODS_3', fontsize=14)

# Add legend
plt.legend(['Actual', 'Forecast', 'Uncertainty Interval', 'Actual Data'])

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m_HOBBIES_1.plot(forecast_HOBBIES_1);
a = add_changepoints_to_plot(fig.gca(), m_HOBBIES_1, forecast_HOBBIES_1)
# Add actual test data points
plt.scatter(test_HOBBIES_1['ds'], test_HOBBIES_1['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for HOBBIES_1', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for HOBBIES_1', fontsize=14)

# Add legend
plt.legend(['Actual', 'Forecast', 'Uncertainty Interval', 'Actual Data'])

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m_HOBBIES_2.plot(forecast_HOBBIES_2);
a = add_changepoints_to_plot(fig.gca(), m_HOBBIES_2, forecast_HOBBIES_2)
# Add actual test data points
plt.scatter(test_HOBBIES_2['ds'], test_HOBBIES_2['y'], color='red', s=10, label='Actual Test Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for HOBBIES_2', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for HOBBIES_2', fontsize=14)

# Add legend
plt.legend(['Actual', 'Forecast', 'Uncertainty Interval', 'Actual Data'])

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m_HOUSEHOLD_1.plot(forecast_HOUSEHOLD_1);
a = add_changepoints_to_plot(fig.gca(),m_HOUSEHOLD_1, forecast_HOUSEHOLD_1)
# Add actual test data points
plt.scatter(test_HOUSEHOLD_1['ds'], test_HOUSEHOLD_1['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for HOUSEHOLD_1', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for HOUSEHOLD_1', fontsize=14)

# Add legend
plt.legend(['Actual', 'Forecast', 'Uncertainty Interval', 'Actual Data'])

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m_HOUSEHOLD_2.plot(forecast_HOUSEHOLD_2);
a = add_changepoints_to_plot(fig.gca(),m_HOUSEHOLD_2, forecast_HOUSEHOLD_2)
# Add actual test data points
plt.scatter(test_HOUSEHOLD_2['ds'], test_HOUSEHOLD_2['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for HOUSEHOLD_1', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for HOUSEHOLD_2', fontsize=14)

# Add legend
plt.legend(['Actual', 'Forecast', 'Uncertainty Interval', 'Actual Data'])

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


from sklearn.metrics import mean_absolute_percentage_error, r2_score


# take the forecating values in next 28 days for each sales

testing_FOODS_1 = forecast_FOODS_1[(forecast_FOODS_1['ds'] > '2016-04-24')][['yhat']].copy()
testing_FOODS_2 = forecast_FOODS_2[(forecast_FOODS_2['ds'] > '2016-04-24')][['yhat']].copy()
testing_FOODS_3 = forecast_FOODS_3[(forecast_FOODS_3['ds'] > '2016-04-24')][['yhat']].copy()
testing_HOBBIES_1 = forecast_HOBBIES_1[(forecast_HOBBIES_1['ds'] > '2016-04-24')][['yhat']].copy()
testing_HOBBIES_2 = forecast_HOBBIES_2[(forecast_HOBBIES_2['ds'] > '2016-04-24')][['yhat']].copy()
testing_HOUSEHOLD_1 = forecast_HOUSEHOLD_1[(forecast_HOUSEHOLD_1['ds'] > '2016-04-24')][['yhat']].copy()
testing_HOUSEHOLD_2 = forecast_HOUSEHOLD_2[(forecast_HOUSEHOLD_2['ds'] > '2016-04-24')][['yhat']].copy()

testing_FOODS_1


training_FOODS_1 = test_FOODS_1[['y']].copy()
training_FOODS_2 = test_FOODS_2[['y']].copy()
training_FOODS_3 = test_FOODS_3[['y']].copy()
training_HOBBIES_1 = test_HOBBIES_1[['y']].copy()
training_HOBBIES_2 = test_HOBBIES_2[['y']].copy()
training_HOUSEHOLD_1 = test_HOUSEHOLD_1[['y']].copy()
training_HOUSEHOLD_2 = test_HOUSEHOLD_2[['y']].copy()


print("Shape of testing1_FOODS_1:", training_FOODS_1.shape)
print("Shape of testing_FOODS_1:", testing_FOODS_1.shape)


# Check the mape for forecasting values for each sale

mape_FOODS_1 = mean_absolute_percentage_error(training_FOODS_1, testing_FOODS_1)
mape_FOODS_2 = mean_absolute_percentage_error(training_FOODS_2, testing_FOODS_2)
mape_FOODS_3 = mean_absolute_percentage_error(training_FOODS_3, testing_FOODS_3)
mape_HOBBIES_1 = mean_absolute_percentage_error(training_HOBBIES_1, testing_HOBBIES_1)
mape_HOBBIES_2 = mean_absolute_percentage_error(training_HOBBIES_2, testing_HOBBIES_2)
mape_HOUSEHOLD_1 = mean_absolute_percentage_error(training_HOUSEHOLD_1, testing_HOUSEHOLD_1)
mape_HOUSEHOLD_2 = mean_absolute_percentage_error(training_HOUSEHOLD_2, testing_HOUSEHOLD_2)
print("the mape_FOODS_1", mape_FOODS_1)
print("the mape_FOODS_2", mape_FOODS_2)
print("the mape_FOODS_3", mape_FOODS_3)
print("the mape_HOBBIES_1", mape_HOBBIES_1)
print("the mape_HOBBIES_2", mape_HOBBIES_2)
print("the mape_HOUSEHOLD_1", mape_HOUSEHOLD_1)
print("the mape_HOUSEHOLD_2", mape_HOUSEHOLD_2)


data_long.head(3)


data_calendar = pd.read_csv("/kaggle/input/calendar/calendar.csv")


data_calendar = data_calendar.iloc[:1913]
data_calendar.head(3)

data_calendar['date'] = pd.to_datetime(data_calendar['date'])
data_calendar.set_index('date', inplace=True)
data_calendar.head(3)


train_FOODS_1['snap_CA'] = data_calendar['snap_CA']
train_FOODS_1['snap_TX'] = data_calendar['snap_TX']
train_FOODS_1['snap_WI'] = data_calendar['snap_WI']

train_FOODS_2['snap_CA'] = data_calendar['snap_CA']
train_FOODS_2['snap_TX'] = data_calendar['snap_TX']
train_FOODS_2['snap_WI'] = data_calendar['snap_WI']

train_FOODS_3['snap_CA'] = data_calendar['snap_CA']
train_FOODS_3['snap_TX'] = data_calendar['snap_TX']
train_FOODS_3['snap_WI'] = data_calendar['snap_WI']

train_HOBBIES_1['snap_CA'] = data_calendar['snap_CA']
train_HOBBIES_1['snap_TX'] = data_calendar['snap_TX']
train_HOBBIES_1['snap_WI'] = data_calendar['snap_WI']

train_HOBBIES_2['snap_CA'] = data_calendar['snap_CA']
train_HOBBIES_2['snap_TX'] = data_calendar['snap_TX']
train_HOBBIES_2['snap_WI'] = data_calendar['snap_WI']

train_HOUSEHOLD_1['snap_CA'] = data_calendar['snap_CA']
train_HOUSEHOLD_1['snap_TX'] = data_calendar['snap_TX']
train_HOUSEHOLD_1['snap_WI'] = data_calendar['snap_WI']

train_HOUSEHOLD_2['snap_CA'] = data_calendar['snap_CA']
train_HOUSEHOLD_2['snap_TX'] = data_calendar['snap_TX']
train_HOUSEHOLD_2['snap_WI'] = data_calendar['snap_WI']


m1 = Prophet(n_changepoints=70)
m1.add_country_holidays(country_name='US')
m1.add_regressor('snap_CA', mode='additive')
m1.add_regressor('snap_TX', mode='additive')
m1.add_regressor('snap_WI', mode='additive')
m1.fit(train_FOODS_1)

m2 = Prophet()
m2.add_country_holidays(country_name='US')
m2.add_regressor('snap_CA', mode='additive')
m2.add_regressor('snap_TX', mode='additive')
m2.add_regressor('snap_WI', mode='additive')
m2.fit(train_FOODS_2)

m3 = Prophet(seasonality_mode='multiplicative')
m3.add_country_holidays(country_name='US')
m3.add_regressor('snap_CA', mode='additive')
m3.add_regressor('snap_TX', mode='additive')
m3.add_regressor('snap_WI', mode='additive')
m3.fit(train_FOODS_3)

m4 = Prophet(seasonality_mode='multiplicative')
m4.add_country_holidays(country_name='US')
m4.add_regressor('snap_CA', mode='additive')
m4.add_regressor('snap_TX', mode='additive')
m4.add_regressor('snap_WI', mode='additive')
m4.fit(train_HOBBIES_1)

m5 = Prophet(n_changepoints=70, seasonality_mode='multiplicative')
m5.add_country_holidays(country_name='US')
m5.add_regressor('snap_CA', mode='additive')
m5.add_regressor('snap_TX', mode='additive')
m5.add_regressor('snap_WI', mode='additive')
m5.fit(train_HOBBIES_2)

m6 = Prophet(seasonality_mode='multiplicative')
m6.add_country_holidays(country_name='US')
m6.add_regressor('snap_CA', mode='additive')
m6.add_regressor('snap_TX', mode='additive')
m6.add_regressor('snap_WI', mode='additive')
m6.fit(train_HOUSEHOLD_1)

m7 = Prophet(seasonality_mode='multiplicative')
m7.add_country_holidays(country_name='US')
m7.add_regressor('snap_CA', mode='additive')
m7.add_regressor('snap_TX', mode='additive')
m7.add_regressor('snap_WI', mode='additive')
m7.fit(train_HOUSEHOLD_2)


future_FOODS_1 = m1.make_future_dataframe(periods=28)
future_FOODS_2 = m2.make_future_dataframe(periods=28)
future_FOODS_3 = m3.make_future_dataframe(periods=28)
future_HOBBIES_1 = m4.make_future_dataframe(periods=28)
future_HOBBIES_2 = m5.make_future_dataframe(periods=28)
future_HOUSEHOLD_1 = m6.make_future_dataframe(periods=28)
future_HOUSEHOLD_2 = m7.make_future_dataframe(periods=28)
future_FOODS_1.head(2)


train_FOODS_1_idx = future_FOODS_1['ds'].isin(train_FOODS_1.index)
test_FOODS_1_idx = ~ train_FOODS_1_idx

train_FOODS_2_idx = future_FOODS_2['ds'].isin(train_FOODS_2.index)
test_FOODS_2_idx = ~ train_FOODS_2_idx

train_FOODS_3_idx = future_FOODS_3['ds'].isin(train_FOODS_3.index)
test_FOODS_3_idx = ~ train_FOODS_3_idx

train_HOBBIES_1_idx = future_HOBBIES_1['ds'].isin(train_HOBBIES_1.index)
test_HOBBIES_1_idx = ~ train_FOODS_1_idx

train_HOBBIES_2_idx = future_HOBBIES_2['ds'].isin(train_HOBBIES_2.index)
test_HOBBIES_2_idx = ~ train_HOBBIES_2_idx

train_HOUSEHOLD_1_idx = future_HOUSEHOLD_1['ds'].isin(train_HOUSEHOLD_1.index)
test_HOUSEHOLD_1_idx = ~ train_HOUSEHOLD_1_idx

train_HOUSEHOLD_2_idx = future_HOUSEHOLD_2['ds'].isin(train_HOUSEHOLD_2.index)
test_HOUSEHOLD_2_idx = ~ train_HOUSEHOLD_2_idx




regressors = ['snap_CA','snap_TX','snap_WI']
for r in regressors:
  future_FOODS_1.loc[train_FOODS_1_idx,r]=train_FOODS_1[r].to_list()
  future_FOODS_2.loc[train_FOODS_2_idx,r]=train_FOODS_2[r].to_list()
  future_FOODS_3.loc[train_FOODS_3_idx,r]=train_FOODS_3[r].to_list()
  future_HOBBIES_1.loc[train_HOBBIES_1_idx,r]=train_HOBBIES_1[r].to_list()
  future_HOBBIES_2.loc[train_HOBBIES_2_idx,r]=train_HOBBIES_2[r].to_list()
  future_HOUSEHOLD_1.loc[train_HOUSEHOLD_1_idx,r]=train_HOUSEHOLD_1[r].to_list()
  future_HOUSEHOLD_2.loc[train_HOUSEHOLD_2_idx,r]=train_HOUSEHOLD_2[r].to_list()



for r in regressors[:]:
  future_FOODS_1.loc[test_FOODS_1_idx ,r]=\
    train_FOODS_1.iloc[-28:][r].to_list()
  future_FOODS_2.loc[test_FOODS_2_idx ,r]=\
    train_FOODS_2.iloc[-28:][r].to_list()
  future_FOODS_3.loc[test_FOODS_3_idx ,r]=\
    train_FOODS_3.iloc[-28:][r].to_list()
  future_HOBBIES_1.loc[test_HOBBIES_1_idx ,r]=\
    train_HOBBIES_1.iloc[-28:][r].to_list()
  future_HOBBIES_2.loc[test_HOBBIES_2_idx ,r]=\
    train_HOBBIES_2.iloc[-28:][r].to_list()
  future_HOUSEHOLD_1.loc[test_HOUSEHOLD_1_idx ,r]=\
    train_HOUSEHOLD_1.iloc[-28:][r].to_list()
  future_HOUSEHOLD_2.loc[test_HOUSEHOLD_2_idx ,r]=\
    train_HOUSEHOLD_2.iloc[-28:][r].to_list()


future_HOUSEHOLD_2.tail(30)


forecast_FOODS_1=m1.predict(future_FOODS_1)
forecast_FOODS_2=m2.predict(future_FOODS_2)
forecast_FOODS_3=m3.predict(future_FOODS_3)
forecast_HOBBIES_1=m4.predict(future_HOBBIES_1)
forecast_HOBBIES_2=m5.predict(future_HOBBIES_2)
forecast_HOUSEHOLD_1=m6.predict(future_HOUSEHOLD_1)
forecast_HOUSEHOLD_2=m7.predict(future_HOUSEHOLD_2)


fig = m1.plot(forecast_FOODS_1);
a = add_changepoints_to_plot(fig.gca(), m_FOODS_1, forecast_FOODS_1)

# Add actual test data points
plt.scatter(test_FOODS_1['ds'], test_FOODS_1['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for FOODS_1', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed

plt.title('Sales Forecast for FOODS_1', fontsize=14)

# Create custom legend with marker
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', label='Trend'),
    Line2D([0], [0], color='dodgerblue', linestyle='-', label='Forecast'),
    plt.Rectangle((0,0), 1, 1, fc='lightblue', alpha=0.5, label='Uncertainty Interval'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='Actual sales'),
    Line2D([0], [0], marker='o', color='black',linestyle='None',
           markersize=7, label='Actual sales')
]
plt.legend(handles=legend_elements)


# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()



plt.show()



fig = m2.plot(forecast_FOODS_2);
a = add_changepoints_to_plot(fig.gca(), m_FOODS_2, forecast_FOODS_2)

# Add actual test data points
plt.scatter(test_FOODS_2['ds'], test_FOODS_2['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for FOODS_2', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for FOODS_2', fontsize=14)


# Create custom legend with marker
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', label='Trend'),
    Line2D([0], [0], color='dodgerblue', linestyle='-', label='Forecast'),
    plt.Rectangle((0,0), 1, 1, fc='lightblue', alpha=0.5, label='Uncertainty Interval'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='Actual sales'),
    Line2D([0], [0], marker='o', color='black',linestyle='None',
           markersize=7, label='Actual sales')
]
plt.legend(handles=legend_elements)

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m3.plot(forecast_FOODS_3);
a = add_changepoints_to_plot(fig.gca(), m_FOODS_3, forecast_FOODS_3)

# Add actual test data points
plt.scatter(test_FOODS_3['ds'], test_FOODS_3['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for FOODS_3', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for FOODS_3', fontsize=14)

# Create custom legend with marker
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', label='Trend'),
    Line2D([0], [0], color='dodgerblue', linestyle='-', label='Forecast'),
    plt.Rectangle((0,0), 1, 1, fc='lightblue', alpha=0.5, label='Uncertainty Interval'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='Actual sales'),
    Line2D([0], [0], marker='o', color='black',linestyle='None',
           markersize=7, label='Actual sales')
]
plt.legend(handles=legend_elements)
# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m4.plot(forecast_HOBBIES_1);
a = add_changepoints_to_plot(fig.gca(), m_HOBBIES_1, forecast_HOBBIES_1)

# Add actual test data points
plt.scatter(test_HOBBIES_1['ds'], test_HOBBIES_1['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for HOBBIES_1', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for HOBBIES_1', fontsize=14)

# Create custom legend with marker
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', label='Trend'),
    Line2D([0], [0], color='dodgerblue', linestyle='-', label='Forecast'),
    plt.Rectangle((0,0), 1, 1, fc='lightblue', alpha=0.5, label='Uncertainty Interval'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='Actual sales'),
    Line2D([0], [0], marker='o', color='black',linestyle='None',
           markersize=7, label='Actual sales')
]
plt.legend(handles=legend_elements)

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m5.plot(forecast_HOBBIES_2);
a = add_changepoints_to_plot(fig.gca(), m_HOBBIES_2, forecast_HOBBIES_2)

# Add actual test data points
plt.scatter(test_HOBBIES_2['ds'], test_HOBBIES_2['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for HOBBIES_2', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for HOBBIES_2', fontsize=14)

# Create custom legend with marker
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', label='Trend'),
    Line2D([0], [0], color='dodgerblue', linestyle='-', label='Forecast'),
    plt.Rectangle((0,0), 1, 1, fc='lightblue', alpha=0.5, label='Uncertainty Interval'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='Actual sales'),
    Line2D([0], [0], marker='o', color='black',linestyle='None',
           markersize=7, label='Actual sales')
]
plt.legend(handles=legend_elements)

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m6.plot(forecast_HOUSEHOLD_1);
a = add_changepoints_to_plot(fig.gca(), m_HOUSEHOLD_1, forecast_HOUSEHOLD_1)

# Add actual test data points
plt.scatter(test_HOUSEHOLD_1['ds'], test_HOUSEHOLD_1['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for HOUSEHOLD_1', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for HOUSEHOLD_1', fontsize=14)

# Create custom legend with marker
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', label='Trend'),
    Line2D([0], [0], color='dodgerblue', linestyle='-', label='Forecast'),
    plt.Rectangle((0,0), 1, 1, fc='lightblue', alpha=0.5, label='Uncertainty Interval'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='Actual sales'),
    Line2D([0], [0], marker='o', color='black',linestyle='None',
           markersize=7, label='Actual sales')
]
plt.legend(handles=legend_elements)

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


fig = m7.plot(forecast_HOUSEHOLD_2);
a = add_changepoints_to_plot(fig.gca(), m_HOUSEHOLD_2, forecast_HOUSEHOLD_2)

# Add actual test data points
plt.scatter(test_HOUSEHOLD_2['ds'], test_HOUSEHOLD_2['y'], color='red', s=10, label='Actual Data')
#Add labels
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.title('Sales Forecast for HOUSEHOLD_2', fontsize=14)

# Focus on recent data and forecast
plt.xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2016-06-01'))  # Adjust dates as needed
# Optionally, you can also add a title
plt.title('Sales Forecast for HOUSEHOLD_2', fontsize=14)

# Create custom legend with marker
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', label='Trend'),
    Line2D([0], [0], color='dodgerblue', linestyle='-', label='Forecast'),
    plt.Rectangle((0,0), 1, 1, fc='lightblue', alpha=0.5, label='Uncertainty Interval'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='Actual sales'),
    Line2D([0], [0], marker='o', color='black',linestyle='None',
           markersize=7, label='Actual sales')
]
plt.legend(handles=legend_elements)

# Adjust layout
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


testing_FOODS_1 = forecast_FOODS_1[(forecast_FOODS_1['ds'] > '2016-04-24')][['yhat']].copy()
testing_FOODS_2 = forecast_FOODS_2[(forecast_FOODS_2['ds'] > '2016-04-24')][['yhat']].copy()
testing_FOODS_3 = forecast_FOODS_3[(forecast_FOODS_3['ds'] > '2016-04-24')][['yhat']].copy()
testing_HOBBIES_1  = forecast_HOBBIES_1[(forecast_HOBBIES_1['ds'] > '2016-04-24')][['yhat']].copy()
testing_HOBBIES_2 = forecast_HOBBIES_2[(forecast_HOBBIES_2['ds'] > '2016-04-24')][['yhat']].copy()
testing_HOUSEHOLD_1 = forecast_HOUSEHOLD_1[(forecast_HOUSEHOLD_1['ds'] > '2016-04-24')][['yhat']].copy()
testing_HOUSEHOLD_2 = forecast_HOUSEHOLD_2[(forecast_HOUSEHOLD_2['ds'] > '2016-04-24')][['yhat']].copy()


training_FOODS_1 = test_FOODS_1[['y']].copy()
training_FOODS_2 = test_FOODS_2[['y']].copy()
training_FOODS_3 = test_FOODS_3[['y']].copy()
training_HOBBIES_1 = test_HOBBIES_1[['y']].copy()
training_HOBBIES_2 = test_HOBBIES_2[['y']].copy()
training_HOUSEHOLD_1 = test_HOUSEHOLD_1[['y']].copy()
training_HOUSEHOLD_2 = test_HOUSEHOLD_2[['y']].copy()


print("Shape of testing1_FOODS_1:", training_FOODS_1.shape)
print("Shape of testing_FOODS_1:", testing_FOODS_1.shape)


mape_FOODS_1 = mean_absolute_percentage_error(training_FOODS_1, testing_FOODS_1)
mape_FOODS_2 = mean_absolute_percentage_error(training_FOODS_2, testing_FOODS_2)
mape_FOODS_3 = mean_absolute_percentage_error(training_FOODS_3, testing_FOODS_3)
mape_HOBBIES_1 = mean_absolute_percentage_error(training_HOBBIES_1, testing_HOBBIES_1)
mape_HOBBIES_2 = mean_absolute_percentage_error(training_HOBBIES_2, testing_HOBBIES_2)
mape_HOUSEHOLD_1 = mean_absolute_percentage_error(training_HOUSEHOLD_1, testing_HOUSEHOLD_1)
mape_HOUSEHOLD_2 = mean_absolute_percentage_error(training_HOUSEHOLD_2, testing_HOUSEHOLD_2)
print("the mape_FOODS_1", mape_FOODS_1)
print("the mape_FOODS_2", mape_FOODS_2)
print("the mape_FOODS_3", mape_FOODS_3)
print("the mape_HOBBIES_1", mape_HOBBIES_1)
print("the mape_HOBBIES_2", mape_HOBBIES_2)
print("the mape_HOUSEHOLD_1", mape_HOUSEHOLD_1)
print("the mape_HOUSEHOLD_2", mape_HOUSEHOLD_2)


training_FOODS_1.head(3)


testing_FOODS_1.head(3)


test_FOODS_1.head(3)


# to make the index aligned
testing_FOODS_1['date']=test_FOODS_1.index.copy()
testing_FOODS_2['date']=test_FOODS_2.index.copy()
testing_FOODS_3['date']=test_FOODS_3.index.copy()
testing_HOBBIES_1['date']=test_HOBBIES_1.index.copy()
testing_HOBBIES_2['date']=test_HOBBIES_2.index.copy()
testing_HOUSEHOLD_1['date']=test_HOUSEHOLD_1.index.copy()
testing_HOUSEHOLD_2['date']=test_HOUSEHOLD_2.index.copy()


testing_FOODS_1['date'] = pd.to_datetime(testing_FOODS_1['date'])
testing_FOODS_2['date'] = pd.to_datetime(testing_FOODS_2['date'])
testing_FOODS_3['date'] = pd.to_datetime(testing_FOODS_3['date'])
testing_HOBBIES_1['date'] = pd.to_datetime(testing_HOBBIES_1['date'])
testing_HOBBIES_2['date'] = pd.to_datetime(testing_HOBBIES_2['date'])
testing_HOUSEHOLD_1['date'] = pd.to_datetime(testing_HOUSEHOLD_1['date'])
testing_HOUSEHOLD_2['date'] = pd.to_datetime(testing_HOUSEHOLD_2['date'])

testing_FOODS_1.set_index('date', inplace=True)
testing_FOODS_2.set_index('date', inplace=True)
testing_FOODS_3.set_index('date', inplace=True)
testing_HOBBIES_1.set_index('date', inplace=True)
testing_HOBBIES_2.set_index('date', inplace=True)
testing_HOUSEHOLD_1.set_index('date', inplace=True)
testing_HOUSEHOLD_2.set_index('date', inplace=True)

testing_FOODS_1.head(3)




# Create new DataFrames with 'actual' and 'forecast' values
comparison_FOODS_1 = pd.DataFrame({
    'actual': training_FOODS_1['y'],
    'forecast': testing_FOODS_1['yhat']
})

comparison_FOODS_2 = pd.DataFrame({
    'actual': training_FOODS_2['y'],
    'forecast': testing_FOODS_2['yhat']
})

comparison_FOODS_3 = pd.DataFrame({
    'actual': training_FOODS_3['y'],
    'forecast': testing_FOODS_3['yhat']
})

comparison_HOBBIES_1 = pd.DataFrame({
    'actual': training_HOBBIES_1['y'],
    'forecast': testing_HOBBIES_1['yhat']
})

comparison_HOBBIES_2 = pd.DataFrame({
    'actual': training_HOBBIES_2['y'],
    'forecast': testing_HOBBIES_2['yhat']
})

comparison_HOUSEHOLD_1 = pd.DataFrame({
    'actual': training_HOUSEHOLD_1['y'],
    'forecast': testing_HOUSEHOLD_1['yhat']
})

comparison_HOUSEHOLD_2 = pd.DataFrame({
    'actual': training_HOUSEHOLD_2['y'],
    'forecast': testing_HOUSEHOLD_2['yhat']
})


# add mape column to each dataframe
comparison_FOODS_1['MAPE'] = abs((comparison_FOODS_1['actual'] - comparison_FOODS_1['forecast']) / comparison_FOODS_1['actual']) * 100
comparison_FOODS_2['MAPE'] = abs((comparison_FOODS_2['actual'] - comparison_FOODS_2['forecast']) / comparison_FOODS_2['actual']) * 100
comparison_FOODS_3['MAPE'] = abs((comparison_FOODS_3['actual'] - comparison_FOODS_3['forecast']) / comparison_FOODS_3['actual']) * 100
comparison_HOBBIES_1['MAPE'] = abs((comparison_HOBBIES_1['actual'] - comparison_HOBBIES_1['forecast']) / comparison_HOBBIES_1['actual']) * 100
comparison_HOBBIES_2['MAPE'] = abs((comparison_HOBBIES_2['actual'] - comparison_HOBBIES_2['forecast']) / comparison_HOBBIES_2['actual']) * 100
comparison_HOUSEHOLD_1['MAPE'] = abs((comparison_HOUSEHOLD_1['actual'] - comparison_HOUSEHOLD_1['forecast']) / comparison_HOUSEHOLD_1['actual']) * 100
comparison_HOUSEHOLD_2['MAPE'] = abs((comparison_HOUSEHOLD_2['actual'] - comparison_HOUSEHOLD_2['forecast']) / comparison_HOUSEHOLD_2['actual']) * 100



comparison_FOODS_1.to_csv('comparison_FOODS_1.csv', index=True)
comparison_FOODS_2.to_csv('comparison_FOODS_2.csv', index=True)
comparison_FOODS_3.to_csv('comparison_FOODS_3.csv', index=True)
comparison_HOBBIES_1.to_csv('comparison_HOBBIES_1.csv', index=True)
comparison_HOBBIES_2.to_csv('comparison_HOBBIES_2.csv', index=True)
comparison_HOUSEHOLD_1.to_csv('comparison_HOUSEHOLD_1.csv', index=True)
comparison_HOUSEHOLD_2.to_csv('comparison_HOUSEHOLD_2.csv', index=True)

from google.colab import files
files.download('comparison_FOODS_1.csv')
files.download('comparison_FOODS_2.csv')
files.download('comparison_FOODS_3.csv')
files.download('comparison_HOBBIES_1.csv')
files.download('comparison_HOBBIES_2.csv')
files.download('comparison_HOUSEHOLD_1.csv')
files.download('comparison_HOUSEHOLD_2.csv')


files.download('comparison_FOODS_2.csv')


files.download('comparison_FOODS_3.csv')


files.download('comparison_HOBBIES_2.csv')


files.download('comparison_HOUSEHOLD_1.csv')


files.download('comparison_HOUSEHOLD_2.csv')


from prophet.diagnostics import cross_validation


df_cv_FOODS_1 = cross_validation(
    m1,
    initial='365 days',
    period='56 days',
    horizon='28 days')

df_cv_FOODS_2 = cross_validation(
    m2,
    initial='365 days',
    period='56 days',
    horizon='28 days')

df_cv_FOODS_3 = cross_validation(
    m3,
    initial='365 days',
    period='56 days',
    horizon='28 days')

df_cv_HOBBIES_1 = cross_validation(
    m4,
    initial='365 days',
    period='56 days',
    horizon='28 days')

df_cv_HOBBIES_2 = cross_validation(
    m5,
    initial='365 days',
    period='56 days',
    horizon='28 days')

df_cv_HOUSEHOLD_1 = cross_validation(
    m6,
    initial='365 days',
    period='56 days',
    horizon='28 days')

df_cv_HOUSEHOLD_2 = cross_validation(
    m7,
    initial='365 days',
    period='56 days',
    horizon='28 days')


df_cv_FOODS_2


from prophet.diagnostics import performance_metrics
pm1 = performance_metrics(df_cv_FOODS_1)
pm1.tail(28)
print('For FOODS_1')
print("the mean: ",pm1['smape'].mean())
print("the standard deviation: ",pm1['smape'].std())
print("the max: ",pm1['smape'].max())
print("the min: ",pm1['smape'].min())



pm2 = performance_metrics(df_cv_FOODS_2)
pm2.tail(28)
print('For FOODS_2')
print("the mean: ",pm2['smape'].mean())
print("the standard deviation: ",pm2['smape'].std())
print("the max: ",pm2['smape'].max())
print("the min: ",pm2['smape'].min())


pm3 = performance_metrics(df_cv_FOODS_3)
pm3.tail(28)
print('For FOODS_3')
print("the mean: ",pm3['smape'].mean())
print("the standard deviation: ",pm3['smape'].std())
print("the max: ",pm3['smape'].max())
print("the min: ",pm3['smape'].min())


pm4 = performance_metrics(df_cv_HOBBIES_1)
pm4.tail(28)
print('For HOBBIES_1')
print("the mean: ",pm4['smape'].mean())
print("the standard deviation: ",pm4['smape'].std())
print("the max: ",pm4['smape'].max())
print("the min: ",pm4['smape'].min())


pm5 = performance_metrics(df_cv_HOBBIES_2)
pm5.tail(28)
print('For HOBBIES_2')
print("the mean: ",pm5['smape'].mean())
print("the standard deviation: ",pm5['smape'].std())
print("the max: ",pm5['smape'].max())
print("the min: ",pm5['smape'].min())


pm6 = performance_metrics(df_cv_HOUSEHOLD_1)
pm6.tail(28)
print('For HOUSEHOLD_1')
print("the mean: ",pm6['smape'].mean())
print("the standard deviation: ",pm6['smape'].std())
print("the max: ",pm6['smape'].max())
print("the min: ",pm6['smape'].min())


pm7 = performance_metrics(df_cv_HOUSEHOLD_2)
pm7.tail(28)
print('For HOUSEHOLD_2')
print("the mean: ",pm7['smape'].mean())
print("the standard deviation: ",pm7['smape'].std())
print("the max: ",pm7['smape'].max())
print("the min: ",pm7['smape'].min())


from prophet.plot import plot_cross_validation_metric
plot_cross_validation_metric(df_cv_FOODS_1, metric='smape');
# Add title
plt.title('Cross Validation SMAPE Metrics for FOODS_1',
          fontsize=14,
          pad=15)

plot_cross_validation_metric(df_cv_FOODS_2, metric='smape');
# Add title
plt.title('Cross Validation SMAPE Metrics for FOODS_2',
          fontsize=14,
          pad=15)
plot_cross_validation_metric(df_cv_FOODS_3, metric='smape');
# Add title
plt.title('Cross Validation SMAPE Metrics for FOODS_3',
          fontsize=14,
          pad=15)
plot_cross_validation_metric(df_cv_HOBBIES_1, metric='smape');
# Add title
plt.title('Cross Validation SMAPE Metrics for HOBBIES_1',
          fontsize=14,
          pad=15)
plot_cross_validation_metric(df_cv_HOBBIES_2, metric='smape');
# Add title
plt.title('Cross Validation SMAPE Metrics for HOBBIES_2',
          fontsize=14,
          pad=15)
plot_cross_validation_metric(df_cv_HOUSEHOLD_1, metric='smape');
# Add title
plt.title('Cross Validation SMAPE Metrics for HOUSEHOLD_1',
          fontsize=14,
          pad=15)
plot_cross_validation_metric(df_cv_HOUSEHOLD_2, metric='smape');
# Add title
plt.title('Cross Validation SMAPE Metrics for HOUSEHOLD_2',
          fontsize=14,
          pad=15)

