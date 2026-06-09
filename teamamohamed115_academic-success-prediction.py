import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


X_train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv').drop(columns=['Target'])
X_test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
y_train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')['Target']


X_train.drop(columns=['id'], inplace=True)
X_test.drop(columns=['id'], inplace=True)


X_train.info()


X_train.head()


encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)


model = Sequential([
    Dense(63, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(33, activation='relu'),
    Dense(18, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, epochs=50, batch_size=12)


y_pred = model.predict(X_test)


y_pred = np.argmax(y_pred, axis=1)
y_pred = encoder.inverse_transform(y_pred)


y_pred


output = pd.DataFrame({'id': range(y_pred.shape[0]), 'Target': y_pred})


output.head()


output.to_csv('submission.csv', index=False)




