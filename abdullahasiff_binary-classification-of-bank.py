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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

train_data = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv', index_col=False)
test_data = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv', index_col=False)

test_id = test_data['id']


print(f"Shape of the training data is: {train_data.shape}")
print(f"Shape of the testing data is : {test_data.shape}")


train_data.columns


train_data.drop(['id'], axis=1, inplace=True)
train_data.head()


test_data.drop(['id'], axis=1, inplace=True)
test_data.head()


print(train_data.isnull().sum())
print(test_data.isnull().sum())


train_data.info()


cat_train = train_data.select_dtypes(['object']).columns
cat_train


for col in cat_train:
    print(f"Column : {col} == {train_data[col].unique()}")
    print()


# Training Data
from sklearn.model_selection import train_test_split

X = train_data.drop(['y'], axis=1)
Y = train_data['y']



# testing Data
from sklearn.model_selection import train_test_split

X1 = test_data.copy()
# Y1 = test_data['y']


# Preparaing Test data

ohe_cols = ['job', 'marital', 'contact', 'month', 'poutcome']
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False)
test_encoded = ohe.fit_transform(X1[ohe_cols])
test_encoded = pd.DataFrame(test_encoded, columns=ohe.get_feature_names_out(ohe_cols))
X1 = pd.concat([X1.drop(columns=ohe_cols), test_encoded], axis=1)  # Merge with original X, dropping ohe_cols
X1.head()


labels = ['default','housing','loan','education']
from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder()

ordinal_mappings = {
    'education': {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3},  # Add all categories
    'housing': {'no': 0, 'yes': 1},
    'loan': {'no': 0, 'yes': 1},
    'default':{'no':0,'yes':1} }

for col in labels:
    if col in ordinal_mappings:
        X1[col] = X1[col].map(ordinal_mappings[col])
        X1[col] = X1[col].fillna(-1).astype(int)




ohe_cols = ['job', 'marital', 'contact', 'month', 'poutcome']
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False)
train_encoded = ohe.fit_transform(X[ohe_cols])
train_encoded = pd.DataFrame(train_encoded, columns=ohe.get_feature_names_out(ohe_cols))
X = pd.concat([X.drop(columns=ohe_cols), train_encoded], axis=1)  # Merge with original X, dropping ohe_cols
X.head()


labels = ['default','housing','loan','education']
from sklearn.preprocessing import OrdinalEncoder
ordinal = OrdinalEncoder()

ordinal_mappings = {
    'education': {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3},  # Add all categories
    'housing': {'no': 0, 'yes': 1},
    'loan': {'no': 0, 'yes': 1},
    'default':{'no':0,'yes':1} }

for col in labels:
    if col in ordinal_mappings:
        X[col] = X[col].map(ordinal_mappings[col])
        X[col] = X[col].fillna(-1).astype(int)

X
        


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)


from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout


model = keras.Sequential()
model.add(Dense(128, activation='relu',input_dim=40))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
callback = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode="auto",
)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),callbacks = callback, epochs=20)



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()





from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Generate predictions and probabilities for the test set (using X1, Y1 for evaluation)
y_pred = (model.predict(X1) > 0.5).astype(int).flatten()  # Binary predictions (threshold 0.5)
y_pred_proba = model.predict(X1).flatten()  # Probabilities for positive class

# Create submission DataFrame using test_id and test_pred
submission = pd.DataFrame({'id': test_id, 'y': y_pred})

# Save submission to CSV
submission.to_csv('submission.csv', index=False)

# Display first few rows of submission
submission.head()




