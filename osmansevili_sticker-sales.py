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
pd.set_option('display.max_columns',100)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


train.head()


test.head()


submission.head()


df=pd.concat([train,test])


df.head()


df.shape


df.info()


df.isnull().sum()


df['date']=pd.to_datetime(df['date'])


df['day']=(df['date']).dt.day
df['month']=(df['date']).dt.month
df['year']=(df['date']).dt.year


df=df.drop(['date'],axis=1)


df.head()


df['country'].value_counts()


df['store'].value_counts()


df['product'].value_counts()


df['num_sold'].value_counts()


df['num_sold']=df['num_sold'].fillna('0')
df['num_sold']=df['num_sold'].astype(int)


df.head()


df.info()


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(x='num_sold', y='product', data=df, palette='viridis')

plt.title('Number of Products Sold by Product', fontsize=16)
plt.xlabel('Number Sold', fontsize=14)
plt.ylabel('Product', fontsize=14)

plt.show()


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(x='num_sold', y='country', data=df, palette='viridis')

plt.title('Number of Products Sold by Product', fontsize=16)
plt.xlabel('Number Sold', fontsize=14)
plt.ylabel('country', fontsize=14)

plt.show()


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(x='year', y='num_sold', data=df, palette='viridis')

plt.title('Number of Products Sold by year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number Sold', fontsize=14)

plt.show()


df=pd.get_dummies(df,drop_first=True)


df.head()


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

from sklearn.metrics import r2_score, mean_squared_error


del df['id']


train_processed = df[:len(train)].copy()
test_processed = df[len(train):].copy()


del test_processed['num_sold']


x=train_processed.drop('num_sold', axis=1)
y=train_processed[['num_sold']]


x=scale(x)


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=42)


model=Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',)


model.summary()


early_stop=EarlyStopping(monitor='val_loss', patience=10)


history=model.fit(x,y, epochs=10, batch_size=10, validation_data=(x_test,y_test), callbacks=[early_stop], verbose=1)


tahmin=model.predict(x_test)


r2_score(y_test, tahmin)


tahmin


sonuc=pd.DataFrame()


tahmin = tahmin.ravel()


sonuc['num_sold']=tahmin


sonuc


sonuc['id']=test['id']


sonuc


sonuc.to_csv('sonuc.csv', index=False)




