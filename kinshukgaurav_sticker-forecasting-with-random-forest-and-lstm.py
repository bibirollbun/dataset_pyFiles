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
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Improved Preprocessing
def preprocess_improved(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['daysinmonth'] = df['date'].dt.daysinmonth
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df = df.drop('date', axis=1)

    for col in ['country', 'store', 'product']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Load Data
train_df = pd.read_csv("train_sales_sticker.csv")
test_df = pd.read_csv("test_sales_sticker.csv")

# Preprocess Data
train_df = preprocess_improved(train_df)
test_df = preprocess_improved(test_df)

# Separate features and target
X = train_df.drop('num_sold', axis=1)
y = train_df['num_sold']

# Handle NaN values explicitly
nan_rows = y.isna()
if nan_rows.any():
    print(f"Number of NaN values in target variable: {nan_rows.sum()}")
    X = X.loc[~nan_rows]
    y = y.loc[~nan_rows]
    print("Rows with NaN values removed.")
else:
    print("No NaN values found in target variable.")


# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5)
model.fit(X_train, y_train)

# Validation
y_pred_val = model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, y_pred_val)
print(f"Improved Validation MAPE: {mape}")

# Prediction on test data
predictions = model.predict(test_df)
predictions[predictions < 0] = 0

# Create Submission File
submission = pd.DataFrame({'id': test_df['id'], 'num_sold': predictions.astype(int)})
submission.to_csv('submission_improved.csv', index=False)
print("Improved submission file created successfully!")




import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Improved Preprocessing (modified for LSTM)
def preprocess_improved(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['daysinmonth'] = df['date'].dt.daysinmonth
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df = df.drop('date', axis=1)

    for col in ['country', 'store', 'product']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Load Data
train_df = pd.read_csv("train_sales_sticker.csv")
test_df = pd.read_csv("test_sales_sticker.csv")

# Preprocess Data
train_df = preprocess_improved(train_df)
test_df = preprocess_improved(test_df)

# Separate features and target
X = train_df.drop('num_sold', axis=1)
y = train_df['num_sold']

# Handle NaN values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
if y.isna().any():
    X = X[~y.isna()]
    y = y.dropna()

# Scaling (MinMaxScaler is often preferred for LSTMs)
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.values.reshape(-1, 1)) # Reshape y for scaling
test_df = scaler_X.transform(imputer.transform(test_df))

# Reshape data for LSTM (samples, timesteps, features)
n_timesteps = 1  # Predict based on the previous single point
n_features = X.shape[1]
X = X.reshape(X.shape[0], n_timesteps, n_features)
test_df = test_df.reshape(test_df.shape[0], n_timesteps, n_features)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]


# LSTM Model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1)) # Output layer
model.compile(optimizer='adam', loss='mse')

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Prediction
predictions = model.predict(test_df)

# Inverse scaling
predictions = scaler_y.inverse_transform(predictions)
predictions[predictions < 0] = 0

# Create Submission File
submission = pd.DataFrame({'id': test_df[:,0,0].astype(int), 'num_sold': predictions.flatten().astype(int)}) #Corrected the id part
submission.to_csv('submission_lstm.csv', index=False)
print("LSTM submission file created successfully!")

