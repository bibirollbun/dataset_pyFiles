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


import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head()


test.head()


train.info()


train.isnull().sum()


train.shape


train.describe()


train['age'].value_counts()


train['job'].value_counts()


sns.countplot(x=train['job']);
plt.xticks(rotation=45);


train['marital'].value_counts()


sns.countplot(x=train['marital']);


train['education'].value_counts()


sns.countplot(x=train['education']);


train['default'].value_counts()


train['balance'].value_counts()


train['housing'].value_counts()


train['loan'].value_counts()


train['contact'].value_counts()


sns.countplot(x=train['contact']);


train['day'].value_counts()


train['month'].value_counts()


sns.countplot(x=train['month']);


train['duration'].value_counts()


train['campaign'].value_counts()


train['pdays'].value_counts()


train['previous'].value_counts()


train['poutcome'].value_counts()


sns.countplot(x=train['poutcome']);


train['y'].value_counts()


sns.countplot(x=train['y']);


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout


from sklearn.preprocessing import normalize, scale
from tensorflow.keras.callbacks import EarlyStopping

import keras
from keras import layers
from keras import ops

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


test_ids = test['id'].copy()


df = pd.concat([train, test], sort=False)


df['pdays'] = df['pdays'].replace(-1, 0)
df['balance'] = df['balance'].replace(-1, 0)


numeric_cols = ['id', 'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype('float')


df=pd.get_dummies(df,drop_first=True)


df.head()


df.isnull().sum()


del df['id']


train_processed = df[:len(train)].copy()
test_processed = df[len(train):].copy()


del test_processed['y']


x=train_processed.drop('y', axis=1)
y=train_processed[['y']]


x=scale(x)


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=42)


model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # ✓ Binary classification için sigmoid
])

# Binary classification için
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


model.summary()


early_stop=EarlyStopping(monitor='val_loss', patience=10)


history=model.fit(x,y, epochs=5, batch_size=3, validation_data=(x_test,y_test), callbacks=[early_stop], verbose=1)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


tahmin=model.predict(x_test)


tahmin


sonuc=pd.DataFrame()


tahmin = tahmin.ravel()


sonuc['y']=tahmin


sonuc['id']=test['id']


sonuc


sonuc.to_csv('sonuc.csv', index=False)




