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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
train_data['date'] = pd.to_datetime(train_data['date'])
train_data.sort_values(by=['country', 'store', 'product','date'], inplace= True)
#train_data.set_index('date', inplace= True)
train_data.describe()


train_data.query('num_sold.isna()').groupby(['country', 'store', 'product'], as_index= False).count()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_data.fillna(0, inplace= True)
train_data['num_sold'] = scaler.fit_transform(train_data[['num_sold']])


train_data = pd.get_dummies(train_data, columns=['country', 'store', 'product'], prefix=['C', 'S', 'P'])


import matplotlib.pyplot as plt

idx = 8

fig = plt.Figure(figsize= (20, 9))
ax = fig.add_subplot(111)
ax.plot(train_data['num_sold'][idx * 2557: idx * 2557 + 14])

from IPython.display import display
display(fig)


def windowing_data(df, window_size= 7, is_y=True):
    days = df['date'].max() - df['date'].min()
    days = days.days
    df_np = df.drop(columns=['id', 'date']).to_numpy()
    X = []
    y = []
    for row in range(0, len(df_np), days + 1):
        for i in range(days + 1 - window_size):
            idx = i + row
            row_t = [ row_i for row_i in df_np[idx : idx + window_size, 0:] ]
            X.append(row_t)
            if is_y: y.append(df_np[idx + window_size , 0])

    if is_y: return np.array(X).astype('float32') , np.array(y).astype('float32')
    else: return np.array(X).astype('float32')


WINDOW_SIZE = 7
X, y= windowing_data(train_data, WINDOW_SIZE)
X.shape , y.shape


split_percentage = 0.8
split_test_val_percentage = 0.5
split_idx = round(split_percentage * len(y))

X_train, y_train = X[:split_idx]  , y[:split_idx]
X_valt , y_valt = X[split_idx:] , y[split_idx:]

split_test_val_idx = round(split_test_val_percentage * len(y_valt))
X_val, y_val, = X_valt[:split_test_val_idx]  , y_valt[:split_test_val_idx]
X_test , y_test = X_valt[split_test_val_idx:] , y_valt[split_test_val_idx:]

X_train.shape , X_val.shape , X_test.shape


X_valt


from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError
from tensorflow.keras.layers import LSTM, Dense, InputLayer , GRU , Input , concatenate 
from tensorflow.keras.metrics import MAPE
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2


model = Sequential([
    InputLayer((WINDOW_SIZE, 15)),
    GRU(units= 32,
        activation= 'tanh',
        dropout= 0.2,
        kernel_regularizer= l2(0.001)), #return_sequences= False
    Dense(1)
])

model.summary()


model.compile(loss= MeanSquaredError(),
             optimizer= Adam(learning_rate= 1e-5, clipvalue=1.0),  #clipvalue=0.001
             metrics= [MeanAbsolutePercentageError()])


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
cp = ModelCheckpoint('model.keras', save_best_only= True)

model.fit(X_train, y_train, validation_data= (X_val , y_val), callbacks=[early_stopping, cp], epochs= 35)


from sklearn.metrics import mean_absolute_percentage_error

trained_model = load_model('model.keras')
y_test_hat = trained_model.predict(X_test)

mape= mean_absolute_percentage_error(y_test, y_test_hat)
print(f"Mape: {mape:.2f}%")


fig = plt.Figure(figsize= (20, 9))
ax = fig.add_subplot(111)
ax.plot(y_test)
ax.plot(y_test_hat)

from IPython.display import display
display(fig)

