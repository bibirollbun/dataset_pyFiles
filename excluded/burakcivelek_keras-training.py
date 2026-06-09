# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_df.head(3)


le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
test_df['Sex'] = le.fit_transform(test_df['Sex'])


train_df = train_df.drop(columns=['id','Height','Weight'])
test_df = test_df.drop(columns=['Height','Weight'])


train_df.head(3)


cols = [i for i in train_df.columns if i != 'Calories' and i != 'Sex']

mms = MinMaxScaler()
mms.fit(train_df[cols])

train_df[cols] = mms.transform(train_df[cols])
test_df[cols] = mms.transform(test_df[cols]) 


test_df


y_train = train_df['Calories']
x_train = train_df.drop('Calories',axis=1)

x_train = x_train.values.astype('float64')
y_train = y_train.values.astype('float64')

x_test = test_df.drop('id',axis=1)

x_test = x_test.values.astype('float64')


model = Sequential()

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=4, batch_size=50)


sub = pd.DataFrame()


sub['id'] = test_df['id']


y_pred = model.predict(x_test)
sub['Calories'] = y_pred


sub[sub['Calories'] < 0] = 0


sub.to_csv('/kaggle/working/submission.csv', index=False)

