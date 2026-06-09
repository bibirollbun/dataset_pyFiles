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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Activation


df=pd.read_csv('/kaggle/input/directional-forecasting-cryptocurrencies/train.csv')
df_test=pd.read_csv('/kaggle/input/directional-forecasting-cryptocurrencies/test.csv')


# Check Dataset
df.head(10)


df_test.head(10)


# Convert timestamp to datetime (if needed, adjust the unit if it's not in milliseconds)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # or 's' for seconds
# Set timestamp as index
df = df.set_index('timestamp')

df_test['timestamp'] = pd.to_datetime(df_test['timestamp'], unit='ms')  # or 's' for seconds
# Set timestamp as index
df_test = df_test.set_index('timestamp')


df.head(10)


df_test.head(10)


!pip install ta


# Feature Engineering (Example: creating lagged features)
def create_features(df, lags=7):
    # Lagged Features (as before)
    for i in range(1, lags + 1):
        df[f'close_lag_{i}'] = df['close'].shift(i)

    # Moving Averages
    df['close_ma_5'] = df['close'].rolling(window=5).mean()  # 5-day moving average
    df['close_ma_10'] = df['close'].rolling(window=10).mean() # 10-day moving average
    df['close_ma_20'] = df['close'].rolling(window=20).mean() # 20-day moving average

    # Volatility (using rolling standard deviation)
    df['close_volatility_5'] = df['close'].rolling(window=5).std()
    df['close_volatility_10'] = df['close'].rolling(window=10).std()
    df['close_volatility_20'] = df['close'].rolling(window=20).std()


    # Example: RSI (Relative Strength Index) - Requires ta library (pip install ta)
    try:
      from ta.momentum import RSIIndicator
      rsi_indicator = RSIIndicator(close=df['close'], window=14) # 14-day RSI is common
      df['rsi_14'] = rsi_indicator.rsi()
    except ImportError:
      print("ta library not found. Install it with: pip install ta")


    return df

df = create_features(df, lags=7)  # Create features (adjust lags as needed)
df_test = create_features(df_test, lags=7)


# Handle missing values (after creating lags)
df = df.dropna()  # Or use imputation methods if you prefer
df_test = df_test.dropna()
row_id=df_test['row_id']
df_test = df_test.drop('row_id', axis=1)


# Feature Scaling (MinMaxScaler is common for neural networks)
scaler = MinMaxScaler()
numerical_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume'] + [col for col in df.columns if 'lag' in col] # Include lagged features
df.loc[:, numerical_cols] = scaler.fit_transform(df[numerical_cols])  # Use .loc

df_test.loc[:, numerical_cols] = scaler.fit_transform(df_test[numerical_cols])  # Use .loc


# Prepare data for the model
X = df.drop('target', axis=1)
y = df['target']


# 2. Data Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False) # Important: shuffle=False for time series


# 3. Model Building (LSTM Example - you can experiment with other architectures)

# Get the number of features (columns in X_train)
n_features = X_train.shape[1]

model = keras.Sequential([
    Input(shape=(n_features, 1)),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1),                       # Output layer with 1 neuron
    Activation('sigmoid')           # Sigmoid activation for binary classification
])


# Reshape input data for LSTM (samples, timesteps, features)
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)    
df_test = df_test.values.reshape(df_test.shape[0], df_test.shape[1], 1) # If df_test is also a DataFrame, keep .values, otherwise remove it as well


# 4. Model Compilation and Training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Use binary_crossentropy
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))


# Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Prediction
probabilities = model.predict(X_test) # Get probabilities
predictions = (probabilities > 0.5).astype(int)  # Convert probabilities to 0 or 1 (threshold at 0.5)

# Save the model
model.save("crypto_forecasting_model.h5")

# To load the model later:
# loaded_model = keras.models.load_model("crypto_forecasting_model.h5"))


probabilities = model.predict(df_test)


predictions = (probabilities > 0.5).astype(int)


print(len(row_id))
print(len(predictions))


# Make sure row_id and predictions are 1D:

if isinstance(row_id, pd.DataFrame) or isinstance(row_id, pd.Series):
    row_id = row_id.values.ravel()  # Convert pandas Series/DataFrame to 1D NumPy array
elif isinstance(row_id, np.ndarray) and row_id.ndim > 1:
    row_id = row_id.ravel()  # Flatten NumPy array if it's multi-dimensional

if isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
    predictions = predictions.values.ravel()
elif isinstance(predictions, np.ndarray) and predictions.ndim > 1:
    predictions = predictions.ravel()

submission_df = pd.DataFrame({'row_id': row_id, 'target': predictions})


# 3. Save to CSV
submission_df.to_csv('sample.csv', index=False)  # index=False to avoid saving the DataFrame index

print("sample.csv created successfully!")


model.save("my_trained_model.h5")


#num_predictions = len(df_test)
#predictions_original_scale = np.random.rand(num_predictions)

