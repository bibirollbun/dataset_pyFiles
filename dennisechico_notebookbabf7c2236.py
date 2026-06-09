# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# load dataset
df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df.columns


df.head()


df = df.drop(['person_emp_length', 'id', 'cb_person_default_on_file'], axis=1 )


df.head()


df = pd.get_dummies(df)


df.head()


y = df['loan_status']
X = df.drop(['loan_status'], axis=1)


X.head()


y.head()


# Dividir nuestro conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


len(X_train), len(X_test)


model = models.Sequential([
    layers.Input(shape=(23,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

model.summary()


model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])



model.fit(X, y, epochs=20, validation_data=(X_test, y_test))


model.predict(X_test)


df_test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")


ids = df_test['id']


df_test = df_test.drop(['person_emp_length', 'id', 'cb_person_default_on_file'], axis=1 )


df_test = pd.get_dummies(df_test)


preds = model.predict(df_test)


pd.DataFrame({'id':ids, 'loan_status':preds.flatten()}).to_csv('outout.csv',index=None)




