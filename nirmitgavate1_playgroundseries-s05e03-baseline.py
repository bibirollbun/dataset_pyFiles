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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score
from lightgbm import LGBMClassifier

import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout,LSTM,Input,Reshape,LeakyReLU,Conv1D,Flatten
from tensorflow.keras.regularizers import L1,L2,l1,l2
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


train_df=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_df.head()


test_df.head()


train_df.info()


test_df.info()


cols=train_df.columns


fig,axes=plt.subplots(2,6,figsize=(13,6))
axes=axes.flatten()
for i in range(12):
    axes[i].scatter(data=train_df,x=cols[i],y='rainfall')
    axes[i].set_title(f'{cols[i]} vs rainfall',fontsize=8)
    # print(i)
plt.suptitle('features vs target')
plt.tight_layout()


fig,axes=plt.subplots(2,6,figsize=(13,6))
axes=axes.flatten()
for i in range(12):
    axes[i].hist(data=train_df,x=cols[i])
    axes[i].set_title(f'{cols[i]}',fontsize=8)
plt.suptitle('hist plot of features')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
plt.title("Correlation of columns")
sns.heatmap(train_df.corr(),annot=True,cmap='viridis')
plt.show()


plt.title('correlation wrt target')
train_df.corr()['rainfall'].sort_values(ascending=False).plot(kind='bar')
plt.show()


train_df.corr()['rainfall'].sort_values(ascending=False)


wd_mean=train_df['winddirection'].mean()
test_df['winddirection']=test_df['winddirection'].fillna(wd_mean)


train_df.head()


pressure_mean=train_df['pressure'].mean()
maxtemp_mean=train_df['maxtemp'].mean()
mintemp_mean=train_df['mintemp'].mean()
temparature_mean=train_df['temparature'].mean()
dewpoint_mean=train_df['dewpoint'].mean()
humidity_mean=train_df['humidity'].mean()
cloud_mean=train_df['cloud'].mean()
sunshine_mean=train_df['sunshine'].mean()
winddirection_mean=train_df['winddirection'].mean()
windspeed_mean=train_df['windspeed'].mean()


train_df[['humidity','cloud']].describe()


train_df.corr()['rainfall'].sort_values(ascending=False)


def FE(train,test):
    train['relative_humidity']=100-train['humidity']
    test['relative_humidity']=100-test['humidity']

    train['relative_cloud']=100-train['cloud']
    test['relative_cloud']=100-test['cloud']
    
    train['windspeed_humidity']=train['windspeed']*train['humidity']
    test['windspeed_humidity']=test['windspeed']*test['humidity'] 
    
    train['windspeed_by_humidity']=train['windspeed']/train['humidity']
    test['windspeed_by_humidity']=test['windspeed']/test['humidity'] 
         
    test['cloud_humidity']=test['cloud']*test['humidity']
    train['cloud_humidity']=train['cloud']*train['humidity']
    
    test['cloud_by_humidity']=test['cloud']/test['humidity']
    train['cloud_by_humidity']=train['cloud']/train['humidity']
    
    train['day_of_week']=train['day']%7
    test['day_of_week']=test['day']%7
    return train,test
train_df,test_df=FE(train_df,test_df)



train_df.head()


X=train_df.drop('rainfall',axis=1)
y=train_df['rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)
scaled_test_df=scaler.transform(test_df)


D=scaled_X_train.shape[1]


i = Input(shape=(D,))
x = Dense(256, activation='relu')(i)
x = Dropout(0.2)(x)
x=Reshape((1,-1))(x)
x=Conv1D(32,strides=1,kernel_size=1)(x)
x=Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1, activation='sigmoid')(x)


model=Model(i,x)
model.compile(
    optimizer=SGD(0.001),
    loss='binary_crossentropy',
    metrics=['AUC']
)


# early_stopping=EarlyStopping(monitor='val_loss',restore_best_weights=True)


model.summary()


r=model.fit(scaled_X_train,y_train,validation_data=(scaled_X_test,y_test),epochs=100)


plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()


plt.plot(r.history['AUC'],label='AUC')
plt.plot(r.history['val_AUC'],label='val_AUC')
plt.legend()


output=model.predict(scaled_test_df)


preds=output[:,0]


sub_df=pd.DataFrame()
sub_df['id']=test_df['id']
sub_df['rainfall']=preds


sub_df.to_csv("submission.csv",index=False)


sub_df.head()




