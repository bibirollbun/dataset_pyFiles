import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

# Merge store info
train = train.merge(store, on='Store', how='left')
test = test.merge(store, on='Store', how='left')

# Convert dates
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

# Basic info
print(train.head())
print(train.info())


# EDA Example: Sales trend
plt.figure(figsize=(12,5))
train.groupby('Date')['Sales'].sum().plot()
plt.title('Total Sales Over Time')
plt.show()


# Feature Engineering: Add date parts
for df in [train, test]:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

# Drop closed stores (Sales=0)
train = train[train['Open']==1]

# Log-transform Sales
train['Sales_log'] = np.log1p(train['Sales'])

features = ['Store','DayOfWeek','Promo','Year','Month','Day']
X = train[features]
y = train['Sales_log']

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# Evaluate
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Validation RMSE: {rmse}')


# Advanced Feature Engineering

# Competition Open Since
for df in [train, test]:
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)

    df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                               (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)

    df['Promo2Open'] = 12 * (df['Year'] - df['Promo2SinceYear']) + \
                          (df['Month'] - (df['Promo2SinceWeek'] // 4))
    df['Promo2Open'] = df['Promo2Open'].apply(lambda x: x if x > 0 else 0)

# Lag Features (Sales)
train = train.sort_values(['Store','Date'])
train['Sales_lag1'] = train.groupby('Store')['Sales'].shift(1)
train['Sales_lag7'] = train.groupby('Store')['Sales'].shift(7)
train['Sales_roll7'] = train.groupby('Store')['Sales'].shift(1).rolling(7).mean()
train['Sales_roll30'] = train.groupby('Store')['Sales'].shift(1).rolling(30).mean()

# Fill NA values from lag features
train = train.fillna(0)

# Update features
features = ['Store','DayOfWeek','Promo','Year','Month','Day','CompetitionOpen','Promo2Open',
            'Sales_lag1','Sales_lag7','Sales_roll7','Sales_roll30']
X = train[features]
y = train['Sales_log']

# Re-train with advanced features
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

# FIXED: Use callback API for early stopping
model_adv = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=200,
    callbacks=[lgb.early_stopping(stopping_rounds=20)]
)

# Evaluate advanced model
y_pred = model_adv.predict(X_val)
rmse_adv = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Advanced Validation RMSE: {rmse_adv}')

# Feature importance
lgb.plot_importance(model_adv, max_num_features=15)
plt.show()



import shap

explainer = shap.TreeExplainer(model_adv)
shap_values = explainer.shap_values(X_val)

shap.summary_plot(shap_values, X_val, plot_type="bar")



# =============================
# Prophet Benchmark (Store 1 Example, Fixed)
# =============================
from prophet import Prophet

# Store 1 daily sales
store1 = train[train['Store'] == 1][['Date', 'Sales']].copy()
store1 = store1.rename(columns={'Date': 'ds', 'Sales': 'y'})

# Fit Prophet model
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prophet_model.fit(store1)

# Validation start date from train (based on split point)
val_start_date = train['Date'].iloc[int(len(train) * 0.8)]  # since you did 80/20 split

# Forecast length = number of days after val_start_date
val_len_store1 = (store1['ds'] >= val_start_date).sum()

future = prophet_model.make_future_dataframe(periods=val_len_store1, freq='D')
forecast = prophet_model.predict(future)

# Extract actual vs predicted
val_dates = store1[store1['ds'] >= val_start_date]['ds'].reset_index(drop=True)
val_sales = store1[store1['ds'] >= val_start_date]['y'].reset_index(drop=True)
pred_sales = forecast[['ds', 'yhat']].tail(len(val_dates))['yhat'].values

# Compute RMSE
rmse_prophet = np.sqrt(mean_squared_error(val_sales, pred_sales))
print(f'Prophet Validation RMSE (Store 1): {rmse_prophet}')

# Plot results
plt.figure(figsize=(12,5))
plt.plot(val_dates, val_sales, label='Actual')
plt.plot(val_dates, pred_sales, label='Prophet Forecast')
plt.legend()
plt.title('Prophet vs Actual Sales - Store 1 Validation')
plt.show()



# =============================
# LSTM Forecasting (Store 1 Example)
# =============================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Focus on one store (Store 1)
store1 = train[train['Store'] == 1][['Date','Sales']].copy()
store1 = store1.set_index('Date').asfreq('D').fillna(0)

# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_sales = scaler.fit_transform(store1[['Sales']])

# Create sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(scaled_sales, seq_length)

# Train/validation split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Build LSTM model
model_lstm = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')

# Train
history = model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, 
                         validation_data=(X_val, y_val), verbose=1)

# Predict
preds = model_lstm.predict(X_val)
preds_rescaled = scaler.inverse_transform(preds)

# Compare
actual = scaler.inverse_transform(y_val)
plt.figure(figsize=(12,5))
plt.plot(actual, label='Actual')
plt.plot(preds_rescaled, label='LSTM Forecast')
plt.legend()
plt.title('LSTM Forecast vs Actual (Store 1)')
plt.show()



!pip install neuralforecast --quiet

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.utils import AirPassengersDF

# Format Rossmann store data for NeuralForecast
df = train[['Date','Store','Sales']].rename(columns={'Date':'ds','Store':'unique_id','Sales':'y'})

# Use only 1 store for demo
df = df[df['unique_id']==1]

# Train TFT model
models = [TFT(input_size=30, h=14, max_steps=200)]
nf = NeuralForecast(models=models, freq='D')
nf.fit(df)

# Forecast 14 days ahead
Y_hat_df = nf.predict()
print(Y_hat_df.head())



# =============================
# Final Prediction for Submission
# =============================

# âœ… Use only features available in test (no lag features, since test has no Sales history)
features_no_lag = ['Store','DayOfWeek','Promo','Year','Month','Day','CompetitionOpen','Promo2Open']

# Re-train model on FULL training set with no-lag features
X_train_full = train[features_no_lag]
y_train_full = train['Sales_log']

train_data_full = lgb.Dataset(X_train_full, label=y_train_full)

model_final = lgb.train(
    params,
    train_data_full,
    num_boost_round=200
)

# Predict on test
X_test = test[features_no_lag]
test['Sales_pred'] = model_final.predict(X_test)
test['Sales'] = np.expm1(test['Sales_pred'])  # reverse log1p

# Save submission file
submission = pd.DataFrame({'Id': test['Id'], 'Sales': test['Sales']})
submission.to_csv('submission.csv', index=False)

submission.head()


