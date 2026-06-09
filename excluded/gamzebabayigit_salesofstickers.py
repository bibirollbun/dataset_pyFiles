import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install protobuf==3.20.3


import pandas as pd
pd.set_option('display.max_columns',100)

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 


from sklearn.preprocessing import normalize, scale
from tensorflow.keras.callbacks import EarlyStopping

import keras
from keras import layers
from keras import ops

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


train.head()


test.head()



submission.head()


df=pd.concat([train,test])


df.head()


df.shape


df.info()


df.isnull().sum()


df['date']=pd.to_datetime(df['date'])


df['day']=(df['date']).dt.day
df['month']=(df['date']).dt.month
df['year']=(df['date']).dt.year


df=df.drop(['date'],axis=1)


df.head()


df['country'].value_counts()


df['store'].value_counts()


df['product'].value_counts()


df['num_sold'].value_counts()


df['num_sold']=df['num_sold'].fillna('0')
df['num_sold']=df['num_sold'].astype(int)


df.head()


df.info()


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(x='num_sold', y='product', data=df, palette='viridis')

plt.title('Number of Products Sold by Product', fontsize=16)
plt.xlabel('Number Sold', fontsize=14)
plt.ylabel('Product', fontsize=14)

plt.show()


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(x='num_sold', y='country', data=df, palette='viridis')

plt.title('Number of Products Sold by Product', fontsize=16)
plt.xlabel('Number Sold', fontsize=14)
plt.ylabel('country', fontsize=14)

plt.show()


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(x='year', y='num_sold', data=df, palette='viridis')

plt.title('Number of Products Sold by year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number Sold', fontsize=14)

plt.show()


df=pd.get_dummies(df,drop_first=True)


df.head()


del df['id']


train_processed = df[:len(train)].copy()
test_processed = df[len(train):].copy()


del test_processed['num_sold']


x=train_processed.drop('num_sold', axis=1)
y=train_processed[['num_sold']]


x=scale(x)


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=42)


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))   
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')



model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)



history = model.fit(x_train, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_data=(x_test, y_test), 
                    callbacks=[early_stop], 
                    verbose=1)


x_final_test = scale(test_processed)
 
tahmin_final = model.predict(x_final_test)
 
tahmin_final = tahmin_final.ravel()

print(f"Tahmin edilen satır sayısı: {len(tahmin_final)}")  


tahmin_final


submission = pd.DataFrame({
    'id': test['id'],         
    'num_sold': tahmin_final  
})
 
submission.to_csv('submission_fixed.csv', index=False)

print("Dosya oluşturuldu. Satır sayısı:", len(submission))
print(submission.head())




