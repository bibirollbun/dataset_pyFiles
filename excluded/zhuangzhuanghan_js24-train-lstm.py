import pandas as pd
import numpy as np
###################################
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
###################################
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
###################################
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
###################
import warnings
warnings.filterwarnings("ignore")
#################
import polars as pl


# Configuration

class CFG: 
    # Note: this is convenient for 
    # updating data/versioning due to different input
    # which is a very common use in Kaggle community
    seed = 42
    target_col = "responder_6"
    feature_cols = ["symbol_id", "time_id"] \
        + [f"feature_{idx:02d}" for idx in range(79)]
    ##########
    categorical_cols = []


# Data loading

DT_GT = 1550 
# Note: Kaggle doesn't have enough RAM for full-size data
# Full-size can be done using chunk-wise run in Colab Pro+

train = pl.scan_parquet(
    "/kaggle/input/js24-preprocessing-create-lags/training.parquet"
).filter(pl.col("date_id") > DT_GT).collect().to_pandas()

valid = pl.scan_parquet(
    "/kaggle/input/js24-preprocessing-create-lags/validation.parquet"
).filter(pl.col("date_id") > DT_GT).collect().to_pandas()

# train.shape, valid.shape
train = pd.concat([train, valid]).reset_index(drop=True)
train = train.fillna(method = 'ffill').fillna(0)
valid = valid.fillna(method = 'ffill').fillna(0)

# Train vs Valid (We do One-fold in this demo)

X_train = train[ CFG.feature_cols ]
y_train = train[ CFG.target_col ]
w_train = train[ "weight" ]
X_valid = valid[ CFG.feature_cols ]
y_valid = valid[ CFG.target_col ]
w_valid = valid[ "weight" ]

(X_train.shape, y_train.shape, w_train.shape, X_valid.shape, y_valid.shape, w_valid.shape)


# make room

import gc
del train
del valid
gc.collect()


# Scale the features

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.values)
X_valid = scaler.fit_transform(X_valid.values)
y_train = y_train.values
y_valid = y_valid.values

# ReShape the X & y
def create_sequences(input_x, input_y, time_steps):
    X, y = [], [] # Note this will create large datasets
    for i in range(len(input_x) - time_steps):
        X.append(input_x[i:(i + time_steps)])
        y.append(input_y[i + time_steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(X_train, y_train, 3)
X_valid, y_valid = create_sequences(X_valid, y_valid, 3)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], X_valid.shape[2]))

# Build the LSTM model

model = Sequential()
model.add(LSTM(
    50, activation='relu', return_sequences=True, 
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer for regression


# Compile and fit the model
# NOTE: It takes quite a long time

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Model prediction & evaluation

y_pred_train = model.predict(X_train)
train_score = r2_score(y_train, y_pred_train, sample_weight=w_train)

y_pred_valid = model.predict(X_valid)
valid_score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid)

print(f"Train R2: {train_score}, Validation R2: {valid_score}")

