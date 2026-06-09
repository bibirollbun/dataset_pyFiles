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


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


df_train.sample(5)


df_train.info()


df_train.describe()


df_train['rainfall'].value_counts()


plt.figure(figsize=(12,6))
sns.heatmap(df_train.corr(),annot=True)


X = df_train.drop(columns = ['rainfall'])
y = df_train['rainfall']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#class weights

# classes = np.unique(y_train)
# class_to_index = {cls: idx for idx, cls in enumerate(classes)}
# y_train_numeric = np.array([class_to_index[cls] for cls in y_train])
# class_counts = np.bincount(y_train_numeric)
# total_samples = len(y_train_numeric)
# class_weights = total_samples / (len(classes) * class_counts)
# class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
# class_weights_dict


from keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)


df_test['winddirection'].fillna(df_test['winddirection'].median(), inplace = True)
df_test_scaled = scaler.transform(df_test)


model = Sequential([
    Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(16, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.summary()


optimizer = Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])


history = model.fit(X_train, y_train, batch_size = 32, epochs = 200, callbacks = [earlystopping], validation_split = 0.2)


y_pred = model.predict(X_test)


y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
accuracy_score(y_test, y_pred)


y_submission = model.predict(df_test_scaled).flatten()
df_submission['rainfall'] = y_submission
df_submission.head()


df_submission.to_csv('Rainfall_prediction.csv', index = False)




