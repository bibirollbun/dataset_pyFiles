import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from tensorflow.keras.optimizers import SGD 


df = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")


df.head()


X= df.drop(columns=['id', 'Target'])
y= df['Target']


encoder = LabelEncoder()
y = encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


model = LinearRegression()

model.fit(X_train , y_train)


model = Sequential([
    Dense(256, activation='relu', input_dim=36),
    Dense(128, activation='relu'),
    Dense(56, activation='relu'),
    
    Dense(3, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.fit(X_train, y_train, epochs=15)


model.fit(X_test, y_test, epochs=15)


model.evaluate(X_train, y_train)


model.evaluate(X_test, y_test)

