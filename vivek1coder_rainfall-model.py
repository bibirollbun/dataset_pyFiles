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


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


df.sample(10)


df.shape


df.isnull().sum()


df.dtypes


X = df.drop(columns=['rainfall','id'])


X.shape


y = df.iloc[:,-1:]


y


X


col = X.columns
col


from sklearn.model_selection import train_test_split as tts


X_train,X_test,y_train,y_test = tts(X,y,test_size=0.2,random_state=51)





from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled = pd.DataFrame(X_train_scaled,columns=col)


X_train_scaled


X_test_scaled = pd.DataFrame(X_test_scaled,columns=col)


X_test_scaled


import matplotlib.pyplot as plt


X_train_scaled.shape


y_test.value_counts()


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(11,activation='relu',input_dim=11))
model.add(Dense(44,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.summary()


model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])


history=model.fit(X_train_scaled,y_train,epochs=100,validation_split=0.2)


import matplotlib.pyplot as plt


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


prediction = model.predict(X_test_scaled)



y_pred = np.where(prediction>0.5,1,0)


from sklearn.metrics import accuracy_score


accuracy_score(y_test,y_pred)




