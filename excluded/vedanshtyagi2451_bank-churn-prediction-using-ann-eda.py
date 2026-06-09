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


data_train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')

data_train.head()


data_train.shape


data_train['CustomerId'].nunique()


data_train[data_train['CustomerId'] == 15674932]


data_train['Surname'].nunique()


165034/2797


X_train = data_train.drop(['id', 'CustomerId', 'Exited','Surname'], axis=1)
Y_train = data_train['Exited']
X_test = data_test.drop(['id', 'CustomerId','Surname'], axis=1)


X_train


X_train.columns


X_train = pd.get_dummies(X_train, columns=['Geography', 'Gender'])
X_test = pd.get_dummies(X_test, columns=['Geography', 'Gender'])


X_train


X_test


X_train.dtypes


X_train['HasCrCard'] = X_train['HasCrCard'].astype(bool)
X_train['IsActiveMember'] = X_train['IsActiveMember'].astype(bool)

X_test['HasCrCard'] = X_test['HasCrCard'].astype(bool)
X_test['IsActiveMember'] = X_test['IsActiveMember'].astype(bool)


from sklearn.preprocessing import StandardScaler

# Select numeric columns only
numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns

scaler = StandardScaler()

# Fit on train, transform both train and test
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


X_train.dtypes


X_train


X_train.shape


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple ANN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.1))

model.add(Dense(48, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_split=0.2)


import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


y_pred_prob = model.predict(X_test)


y_pred = (y_pred_prob > 0.5).astype(int).flatten()


submission = pd.DataFrame({
    'id': data_test['id'],       # from original test set
    'Exited': y_pred             # your predictions
})


submission.to_csv('submission.csv', index=False)




