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


data = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
data.head(10)


data.describe()


output=pd.DataFrame(columns = ["id", 'BeatsPerMinute'])


x = data.drop(columns = ["id", "BeatsPerMinute"])


y = data['BeatsPerMinute']


pip install keras


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(x.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='linear')
])



model.compile(
    optimizer = 'adam',
    loss = 'mean_squared_error',
    metrics = ['mae']
)


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    restore_best_weights = True,
)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 5,
    callbacks = [early_stopping],
    verbose =1
)


test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


test_data = test.drop(columns = ['id'])


output['id'] = test['id']
output['BeatsPerMinute'] = 0


pred = model.predict(test_data)


output['BeatsPerMinute'] = pred


output.to_csv("/kaggle/working/submission.csv", index = False)




