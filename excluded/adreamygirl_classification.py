import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD



df = pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')


df.info()


df.head()





 df.drop(columns=['id'], inplace=True)


X = df.iloc[:, :-1]
y = df.iloc[:, -1]



 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)


 model = Sequential([
    Dense(8, activation='relu', input_dim=5),
    Dense(4, activation='relu'),
    Dense(4, activation='relu'),

    Dense(1, activation='')
])

model.compile(optimizer=SGD(), loss='mean_absolute_error')


model.evaluate(X_train, y_train)


model.evaluate(X_test, y_test)

