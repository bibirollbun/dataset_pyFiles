!pip install -q opendatasets


from google.colab import files
import shutil
import os

uploaded = files.upload()
os.makedirs('/root/.config/kaggle', exist_ok=True)
shutil.copy('kaggle.json', '/root/.config/kaggle/kaggle.json')


import opendatasets as od

url = 'https://www.kaggle.com/competitions/playground-series-s5e9/data'
od.download(url)


os.listdir('./playground-series-s5e9')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv('./playground-series-s5e9/train.csv')
test = pd.read_csv('./playground-series-s5e9/test.csv')


train.head()


train.isna().sum()


train.describe()


for column in train.columns:
    Q1 = train[column].quantile(0.25)
    Q3 = train[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train = train[(train[column] >= lower_bound) & (train[column] <= upper_bound)]


X = train.drop(columns=['BeatsPerMinute', 'id'])
X_test = test.drop(columns=['id'])
y = train['BeatsPerMinute']


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint


class FCNRegressor():
    def __init__(self):
        self.model = Sequential()

    def build(self, input_shape):
        self.model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X, y):
        checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
        self.model.fit(X, y, epochs=2, batch_size=32, validation_split=0.1, callbacks=[checkpoint])

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)


import tensorflow as tf

devices = tf.config.list_physical_devices('GPU')
if devices:
    tf.config.experimental.set_memory_growth(devices[0], True)


fcn_regressor = FCNRegressor()
fcn_regressor.build(input_shape=X_train.shape[1])
fcn_regressor.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error


predict = fcn_regressor.predict(X_valid)
print(np.sqrt(mean_squared_error(predict, y_valid)))


from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)


predict = linear_regressor.predict(X_valid)
print(np.sqrt(mean_squared_error(predict, y_valid)))


!pip install -q catboost


from catboost import CatBoostRegressor

catboost_regressor = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=10)
catboost_regressor.fit(X_train, y_train)


predict = catboost_regressor.predict(X_valid)
print(np.sqrt(mean_squared_error(predict, y_valid)))


predict = linear_regressor.predict(X_test)


results = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': predict
})

results.to_csv('submission.csv', index=False)
results.head()


!kaggle competitions submit playground-series-s5e9 -f submission.csv -m "My submission"

