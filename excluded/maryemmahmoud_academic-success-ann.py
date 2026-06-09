import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder , MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
df.head()


df.info()


df.isna().sum()


df.duplicated().sum()


df.describe()


df.drop(['id'],axis=1,inplace=True)


le=LabelEncoder()


df['Target'] = le.fit_transform(df['Target'])


x=df.drop(['Target'],axis=1)
y=df['Target']


sc=StandardScaler()


x=sc.fit_transform(x)


x.shape[1]


x_train, x_dummy, y_train, y_dummy = train_test_split(x, y, test_size=0.2, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_dummy, y_dummy, test_size=0.5, random_state=42)



model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(36,)),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])





optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))



early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)


history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test),
                    callbacks=[early_stopping, lr_scheduler])



test_accuracy = model.evaluate(x_test, y_test)
test_accuracy


plt.figure(figsize= (30, 8))
plt.style.use('fivethirtyeight')

plt.plot(history.history['loss'], 'r', label= 'Training loss')
plt.plot(history.history['val_loss'], 'g', label= 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout
plt.show()




