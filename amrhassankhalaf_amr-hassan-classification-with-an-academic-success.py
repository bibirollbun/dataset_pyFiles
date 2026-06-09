import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")


df.head()


df.info()


df.drop('id', axis=1, inplace=True)


df.isna().sum()


label_encoder=LabelEncoder()


df['Target'] = label_encoder.fit_transform(df['Target'])


X = df.drop('Target', axis=1)
y = df['Target']


scaler = MinMaxScaler()
X = scaler.fit_transform(X)



y


y = scaler.fit_transform(y.values.reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


X_train.shape


y_train.shape


model = Sequential([
    Dense(128, activation = 'relu', input_dim=36),
    Dense(64, activation = 'relu'),
    Dense(32, activation = 'relu'),
    Dense(16, activation = 'relu'),

    Dense(3, activation = 'softmax')
])


# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.summary()


model.fit(X_train, y_train, epochs=50)


model.evaluate(X_train, y_train)

