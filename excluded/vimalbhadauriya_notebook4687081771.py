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


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df


df = df.drop(['id','day',],axis =1)
df


from sklearn.model_selection import train_test_split



df.head()


X = df.drop(['rainfall'],axis =1)
X
y = df[['rainfall']]
y


from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train


import pickle
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import datetime


#build model

model = Sequential([
    Dense(512,activation='relu',input_shape=(X_train.shape[1],)),
    Dense(256,activation='relu'),
    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])


model.summary()


opt = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.BinaryCrossentropy()


model.compile(optimizer = opt,loss=loss,metrics=['accuracy'])


early_stopping_callbacks = EarlyStopping(monitor='val_loss',patience = 10)



history = model.fit(
    X_train,y_train,validation_data = (X_test,y_test),epochs =100,
    callbacks=[early_stopping_callbacks]
)


model.save('model.h5')


df1 = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


df1 = df1.drop(['day','id'],axis =1)


df1


x_test = scaler.transform(df1)
x_test


y_pred = model.predict(x_test)


y_pred


y_pred_binary = (y_pred >= 0.5).astype(int)


y_pred_binary


y_pred_binary = pd.DataFrame(y_pred_binary)


df2 = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


df2


df2 = df2[['id']]


y_pred_binary


df = pd.concat([df2,y_pred_binary],axis =1)


df


df.rename(columns = {0:'rainfall'},inplace = True)


df


df.to_csv('model.csv',index = False)


y_pred_binary.info()


y_pred_binary


df


df.rename()

