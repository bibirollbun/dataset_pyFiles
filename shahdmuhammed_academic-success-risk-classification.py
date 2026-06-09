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
import sklearn.metrics 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense , Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler


df = pd.read_csv(r'/kaggle/input/playground-series-s4e6/train.csv')


df.head()


df.info()


df.isna().sum()


df.describe()


df['Target'].value_counts()


plt.figure(figsize=(6,4))
sns.countplot(x=df['Target'])
plt.title("Target Variable Distribution")
plt.show()


df.drop(['id'] , axis = 1 , inplace = True)


df.head()


X = df.drop(['Target'], axis = 1)
y = df['Target']


X.shape


y.shape


from sklearn.preprocessing import MinMaxScaler
features = [
    'Daytime/evening attendance', 'Displaced', 'Educational special needs','Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Target'
]
features_to_scale = [col for col in df.columns if col not in features]

scaler = MinMaxScaler()


encoder = LabelEncoder()
Y = encoder.fit_transform(y)


x_train , x_test , y_train , y_test = train_test_split(X,Y,train_size=0.8,random_state=0)


x_train[features_to_scale] = scaler.fit_transform(x_train[features_to_scale])
x_test[features_to_scale] = scaler.transform(x_test[features_to_scale])


x_train.shape


x_test.shape


y_train.shape


y_test.shape


input_dim = x_train.shape[1]
model = Sequential()
model.add(Dense(128,activation='relu',input_dim=input_dim))
model.add(Dropout(0.25))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.summary ()


history = model.fit(x_train,y_train,epochs=20,validation_split=0.2 )


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='best')  
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


y_pred = model.predict(x_test)


model.evaluate(x_test,y_test)




