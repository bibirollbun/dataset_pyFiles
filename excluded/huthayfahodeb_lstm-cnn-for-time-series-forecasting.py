import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout, LSTM, Concatenate
from tensorflow.keras.losses import MeanAbsolutePercentageError
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=["date"])


train_df.head()


train_df.info()


train_df.drop(columns = ['id'], inplace = True)


train_df.isnull().sum()


train_df[train_df['num_sold'].isnull()].index


train_df.dropna(inplace = True)


train_df.isnull().sum()


train_df['country'].value_counts()


train_df['store'].value_counts()


train_df['product'].value_counts()


train_df['date'] = pd.to_datetime(train_df['date'])

train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['quarter'] = train_df['date'].dt.quarter

train_df.drop(columns = ['date'], inplace = True)


train_df.head()


obj_col = list(train_df.select_dtypes(include = 'object').columns)


obj_col


for col in obj_col:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])


train_df.head()


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    
    for i in range(len(sequence)):

        end_ix = i + n_steps

        if end_ix > len(sequence)-1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)


sequence = train_df['num_sold'].values[:50000]

X, y = split_sequence(sequence, 14)

for i in range(5):
    print(X[i], y[i])
    print()


X.shape, y.shape


n_features = 1
N = len(X)

X = X.reshape(X.shape[0], X.shape[1], n_features)

X.shape


input_shape = (X.shape[1], n_features)

inputs = Input(shape=input_shape)

cnn_x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
cnn_x = MaxPooling1D(pool_size=2)(cnn_x)
cnn_x = Dropout(0.2)(cnn_x)
cnn_x = Flatten()(cnn_x) 

lstm_x = LSTM(64, activation='tanh', return_sequences=False)(inputs)

concat = Concatenate()([cnn_x, lstm_x])

x = Dense(64, activation='relu')(concat)
x = Dropout(0.2)(x)

outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)


model.compile(optimizer='adam', loss='mape')


tf.keras.utils.plot_model(model, show_shapes=True, to_file='model_uni.png', dpi=70,)


model.fit(X[: -N//2], y[: -N//2], epochs = 50, validation_data = (X[-N//2:], y[-N//2:]))


y_val_pred = model.predict(X[-N//2:])

mape = mean_absolute_percentage_error(y[-N//2:], y_val_pred)
print(f"MAPE on validation data: {mape:.2f}%")


cols = [col for col in train_df.columns if col != 'num_sold'] + ['num_sold']
sequences = train_df[cols].values


sequences[:2]


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    
    for i in range(len(sequences)):

        end_ix = i + n_steps

        if end_ix > len(sequences)-1:
            break

        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
        
    return np.array(X), np.array(y)


n_steps = 10

X, y = split_sequences(sequences, n_steps)
print(X.shape, y.shape)

n_features = X.shape[2]

for i in range(4):
    print(X[i], y[i])
    print()


n_features = X.shape[2]
N = len(X)

X = X.reshape(X.shape[0], X.shape[1], n_features)

X.shape


n_features


input_shape = (X.shape[1], n_features)

inputs = Input(shape=input_shape)

cnn_x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
cnn_x = MaxPooling1D(pool_size=2)(cnn_x)
cnn_x = Dropout(0.2)(cnn_x)
cnn_x = Flatten()(cnn_x) 

lstm_x = LSTM(64, activation='tanh', return_sequences=False)(inputs)

concat = Concatenate()([cnn_x, lstm_x])

x = Dense(64, activation='relu')(concat)
x = Dropout(0.2)(x)

outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)


model.compile(optimizer='adam', loss='mape')


tf.keras.utils.plot_model(model, show_shapes=True, to_file='model_uni.png', dpi=70,)


model.fit(X[: -N//2], y[: -N//2], epochs = 25, validation_data = (X[-N//2:], y[-N//2:]))


y_val_pred = model.predict(X[-N//2:])

mape = mean_absolute_percentage_error(y[-N//2:], y_val_pred)
print(f"MAPE on validation data: {mape:.2f}%")




