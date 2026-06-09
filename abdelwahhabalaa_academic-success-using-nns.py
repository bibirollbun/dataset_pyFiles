import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import SGD, Adamax
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df.head()


df.info()


df.drop(columns = ['id'], inplace = True)


x = df.drop(columns = ['Target'])
y = df['Target']


x


y


Encoder = OneHotEncoder(sparse_output = False)
y = Encoder.fit_transform(y.values.reshape(-1, 1))


y


x_train, x_dummy, y_train, y_dummy = train_test_split(x, y, test_size=0.2, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_dummy, y_dummy, test_size=0.5, random_state=42)


model = Sequential([
    Dense(1024, activation = 'relu', input_dim = x_train.shape[1]),
    Dense(512, activation = 'relu'),
    Dense(256, activation = 'relu'),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(32, activation = 'relu'),
    Dense(16, activation = 'relu'),
    Dense(8, activation = 'relu'),
    Dense(3, activation = 'softmax'),
    ])
model.compile(
    optimizer=Adamax(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit(x_train,y_train, validation_data=(x_valid,y_valid), epochs=100, batch_size=32, callbacks=EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))


print('Loss:',model.evaluate(x_train,y_train)[0])
print('Accuracy:',model.evaluate(x_train,y_train)[1])

