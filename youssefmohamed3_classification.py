import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


df = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df.head()


X = df.drop(columns=['id', 'Target'])
Y = df['Target']


encoder = LabelEncoder()
Y = encoder.fit_transform(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(128, activation='relu', input_dim=36),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=5)

