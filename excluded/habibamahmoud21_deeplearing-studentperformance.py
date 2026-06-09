import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df.head()


df.info()


X= df.drop(columns=['id', 'Target'])
y= df['Target']


encoder=LabelEncoder()
df['Target']=encoder.fit_transform(df['Target'])
df





X = df.iloc[:,:-1]
y = df.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



model = Sequential([
    Dense(64, activation='relu', input_dim=37),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
   
    Dense(3, activation='softmax')
])

model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


model.fit(X_train, y_train, epochs=100)


model.evaluate(X_test, y_test)


model.evaluate(X_train, y_train)

