import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
warnings.simplefilter(action='ignore', category=FutureWarning)


DATA_PATH = '/kaggle/input/playground-series-s3e20'

train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'),index_col='ID_LAT_LON_YEAR_WEEK')
test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'),index_col='ID_LAT_LON_YEAR_WEEK')
sample_submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'),index_col='ID_LAT_LON_YEAR_WEEK')


# Print some general train/test info
print(f'\nShape of training dataset: {train_df.shape}')
print(f'\nShape of test dataset: {test_df.shape}')

# Print some general target info
target = list(set(train_df.columns)-set(test_df.columns)).pop()
target_type = 'continuous' if train_df[target].nunique() >= 10 else 'categorical'
print(f"\nTarget variable: \'{target}\'")
print(f'\nType of target variable: {target_type}')


train_df.head()


train_df.info()


train_df_filtered = train_df[(train_df['latitude'] == -0.51) & (train_df['longitude'] == 29.29)]

train_df = train_df_filtered


print(train_df['SulphurDioxide_SO2_column_number_density'].describe())



missing_values = train_df['SulphurDioxide_SO2_column_number_density'].isnull().sum()
print(f'Missing values: {missing_values}')

train_df['SulphurDioxide_SO2_column_number_density'].fillna(train_df['SulphurDioxide_SO2_column_number_density'].mean(), inplace=True)



import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.hist(train_df['SulphurDioxide_SO2_column_number_density'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Sulphur Dioxide Column Density')
plt.xlabel('Sulphur Dioxide SO2 Column Density')
plt.ylabel('Frequency')
plt.show()



# boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(train_df['SulphurDioxide_SO2_column_number_density'], vert=False)
plt.title('Boxplot of Sulphur Dioxide Column Density')
plt.xlabel('Sulphur Dioxide SO2 Column Density')
plt.show()



from statsmodels.graphics.tsaplots import plot_acf

# ACF
plt.figure(figsize=(10, 6))
plot_acf(train_df['SulphurDioxide_SO2_column_number_density'].dropna(), lags=50)
plt.title('Autocorrelation of Sulphur Dioxide SO2 Column Density')
plt.show()



# MA
train_df['Ozone_rolling_mean'] = train_df['Ozone_O3_column_number_density'].rolling(window=12).mean()

plt.figure(figsize=(12, 6))
plt.plot(train_df['year'], train_df['Ozone_O3_column_number_density'], label='Original Data', color='blue', alpha=0.5)
plt.plot(train_df['year'], train_df['Ozone_rolling_mean'], label='12-Point Moving Average', color='red', linewidth=2)
plt.title('Ozone Column Density with 12-Point Moving Average')
plt.xlabel('Year')
plt.ylabel('Ozone Column Density')
plt.legend()
plt.grid(True)
plt.show()



from statsmodels.graphics.tsaplots import plot_acf

# ACF
plt.figure(figsize=(10, 6))
plot_acf(train_df['Ozone_O3_column_number_density'].dropna(), lags=50)
plt.title('Autocorrelation of Ozone Column Density')
plt.show()



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
data = train_df[['year', 'Ozone_O3_column_number_density']].dropna()

# MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Ozone_O3_column_number_density'].values.reshape(-1, 1))

plt.plot(data['year'], data_scaled)
plt.title('Scaled Ozone Column Density')
plt.xlabel('Year')
plt.ylabel('Scaled Ozone Column Density')
plt.show()



# def create_dataset(data, n_steps):
#     X, y = [], []
#     for i in range(len(data)):
#         end_ix = i + n_steps
#         if end_ix > len(data)-1:
#             break
#         seq_X, seq_y = data[i:end_ix], data[end_ix]
#         X.append(seq_X)
#         y.append(seq_y)
#     return np.array(X), np.array(y)

# n_steps = 12  
# X, y = create_dataset(data_scaled, n_steps)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# model = Sequential()
# model.add(LSTM(units=50, return_sequences=False, input_shape=(n_steps, 1)))
# model.add(Dense(units=1)) 

# model.compile(optimizer='adam', loss='mean_squared_error')
# model.summary()
# history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test))


def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 12  
X, y = create_dataset(data_scaled, n_steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(n_steps, 1)))
lstm_model.add(Dense(units=1)) 
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test))

y_pred = lstm_model.predict(X_test)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_scaled = scaler.inverse_transform(y_pred)

plt.plot(y_test_scaled, label='Actual')
plt.plot(y_pred_scaled, label='Predicted')
plt.title('Ozone Column Density - LSTM Predictions')
plt.xlabel('Time')
plt.ylabel('Ozone Column Density')
plt.legend()
plt.show()



# y_pred = model.predict(X_test)

# y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
# y_pred_scaled = scaler.inverse_transform(y_pred)

# plt.plot(y_test_scaled, label='Actual')
# plt.plot(y_pred_scaled, label='Predicted')
# plt.title('Ozone Column Density - Actual vs Predicted')
# plt.xlabel('Time')
# plt.ylabel('Ozone Column Density')
# plt.legend()
# plt.show()



from sklearn.metrics import mean_squared_error
import math

mse = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse = math.sqrt(mse)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')



import warnings  
from statsmodels.tools.sm_exceptions import ConvergenceWarning  
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(train_df[['Ozone_O3_column_number_density']].dropna())

train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[:train_size], data_scaled[train_size:]

# train ARIMA model
p, d, q = 3, 1, 2  
model = ARIMA(train, order=(p, d, q))
arima_result = model.fit()

# predict
history = list(train)
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit()
    pred = model_fit.forecast()[0] 
    predictions.append(pred)
    history.append(test[t])  

test_original = scaler.inverse_transform(test)
predictions_original = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


# plt.figure(figsize=(12, 6))
plt.plot(test_original, label='Actual', color='blue')
plt.plot(predictions_original, label='Predicted', linestyle='dashed', color='orange')
plt.title('Ozone Column Density - ARIMA Model Predictions')
plt.xlabel('Time')
plt.ylabel('Ozone Column Density')
plt.legend()
plt.show()



from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test_original, predictions_original)
rmse = np.sqrt(mean_squared_error(test_original, predictions_original))

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')


# Residual Analysis
residuals = y_test_scaled.flatten() - y_pred_scaled.flatten()
plt.hist(residuals, bins=50, color='purple', edgecolor='black', alpha=0.7)
plt.title('Residual Distribution of LSTM Predictions')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Model Comparison
rmse_lstm = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
mae_lstm = mean_absolute_error(y_test_scaled, y_pred_scaled)
rmse_arima = np.sqrt(mean_squared_error(test_original, predictions_original))
mae_arima = mean_absolute_error(test_original, predictions_original)

plt.bar(['LSTM', 'ARIMA'], [rmse_lstm, rmse_arima], color=['blue', 'orange'])
plt.title('RMSE Comparison Between LSTM and ARIMA')
plt.ylabel('RMSE')
plt.show()

# Future Prediction



import matplotlib.pyplot as plt

# Create a figure with two subplots: 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# First subplot: LSTM model predictions
axes[0].plot(y_test_scaled, label='Actual')  # Plot actual values
axes[0].plot(y_pred_scaled, label='Predicted')  # Plot predicted values
axes[0].set_title('Ozone Column Density - LSTM Predictions')  # Set title for the LSTM plot
axes[0].set_xlabel('Time')  # Set x-axis label
axes[0].set_ylabel('Ozone Column Density')  # Set y-axis label
axes[0].legend()  # Display legend

# Second subplot: ARIMA model predictions
axes[1].plot(test_original, label='Actual', color='blue')  # Plot actual values for ARIMA
axes[1].plot(predictions_original, label='Predicted', linestyle='dashed', color='orange')  # Plot predicted values for ARIMA
axes[1].set_title('Ozone Column Density - ARIMA Model Predictions')  # Set title for the ARIMA plot
axes[1].set_xlabel('Time')  # Set x-axis label
axes[1].set_ylabel('Ozone Column Density')  # Set y-axis label
axes[1].legend()  # Display legend

# Adjust layout to prevent overlap between subplots
plt.tight_layout()

# Show the figure
plt.show()




# Future Prediction
future_steps = 50
X_input = X_test[-1].reshape(1, n_steps, 1)
future_predictions = []
for _ in range(future_steps):
    pred = lstm_model.predict(X_input)  # Ensure using LSTM model
    future_predictions.append(pred[0, 0])
    X_input = np.append(X_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
future_predictions_scaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

plt.plot(range(len(y_test_scaled)), y_test_scaled, label='Actual', color='blue')
plt.plot(range(len(y_test_scaled), len(y_test_scaled) + future_steps), future_predictions_scaled, label='LSTM Future Predictions', color='green', linestyle='dashed')
plt.title('Future Prediction of Ozone Column Density Using LSTM')
plt.xlabel('Time')
plt.ylabel('Ozone Column Density')
plt.legend()
plt.show()

