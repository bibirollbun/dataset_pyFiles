import pandas as pd
from prophet import Prophet

# Load train data
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet').reset_index()

# Format data for Prophet
# Prophet expects: 'ds' = datetime, 'y' = target
df_prophet = train_df[['timestamp', 'label']].rename(columns={'timestamp': 'ds', 'label': 'y'})

# Convert to datetime if needed
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], unit='s')  # or unit='ms' depending on format

# Fit model
model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.fit(df_prophet)





train_df.head()


model


# Prepare test timestamps
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet').reset_index()

test_df.head()


# Create dummy datetime index for test: assume 1-min intervals
test_df = test_df.reset_index(drop=True)
start_time = train_df['timestamp'].max() + pd.Timedelta(minutes=1)
test_df['timestamp'] = pd.date_range(start=start_time, periods=len(test_df), freq='T')

X_test = TimeSeries.from_dataframe(test, time_col='timestamp', value_cols=features)



future = test_df[['ID']].rename(columns={'ID': 'ds'})
future['ds'] = pd.to_datetime(future['ds'], unit='s')  # or unit='ms'




# Forecast
forecast = model.predict(future)

# Prepare submission
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submission['label'] = forecast['yhat']
submission.to_csv('submission_prophet.csv', index=False)

