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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD



df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df.head()



encoder = LabelEncoder()
df['Target'] = encoder.fit_transform(df['Target'])
df



df.drop(columns=['id'], inplace=True)



X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



model = Sequential([
    Dense(8, activation='relu', input_dim=36),
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),

    Dense(3, activation='softmax')
])

model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.fit(X_train, y_train, epochs=100)



model.evaluate(X_train, y_train)



model.evaluate(X_test, y_test)





