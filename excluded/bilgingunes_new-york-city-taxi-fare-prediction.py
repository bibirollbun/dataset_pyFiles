import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 1000000)


test=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv', nrows = 1000000)


sample=pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv', nrows = 1000000)


train.head()


train.shape


test.head()


train.isnull().sum() 


train[train["dropoff_longitude"].isnull()]


train = train.dropna(subset=["dropoff_longitude"])


train.isnull().sum()


train.info() #object verimiz var birisi id ama key ismini almış diğeri de datetime


train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"]) #datetime veri tipine çevir
train["Day"] = train["pickup_datetime"].dt.day #gün verisini çek ve gün sütununa ekle
train["Month"] = train["pickup_datetime"].dt.month #ay
train["Years"] = train["pickup_datetime"].dt.year #yıl
train.drop("pickup_datetime",axis=1,inplace=True) #ayırdığımız için artık orijinal sütuna ihtiyacaımız yok


test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"]) #datetime veri tipine çevir
test["Day"] = test["pickup_datetime"].dt.day #gün verisini çek ve gün sütununa ekle
test["Month"] = test["pickup_datetime"].dt.month #ay
test["Years"] = test["pickup_datetime"].dt.year #yıl
test.drop("pickup_datetime",axis=1,inplace=True) #ayırdığımız için artık orijinal sütuna ihtiyacaımız yok


train.head()


train['fare_amount'].describe()#min maks değerlerine bakıyorduk ki negatif ücret değerleri olduğunu gördük bunları atalım


train.shape


train = train.drop(train[train['fare_amount']<0].index, axis=0)
train.shape


train['passenger_count'].describe() #208 yolculu bir veri var bunu atalım


train = train.drop(train[train['passenger_count']==208].index, axis = 0)


train["pickup_latitude"].describe() # -90/90 toplam 180 enlem var bu min maks değerleri yanlış o satırları atalım


train = train.drop(((train[train['pickup_latitude']<-90])).index, axis=0)
train = train.drop(((train[train['pickup_latitude']>90])).index, axis=0)


train["pickup_longitude"].describe() # -180/180 toplam 360 var dışındakileri atalım


train = train.drop(((train[train['pickup_longitude']<-180])).index, axis=0)
train = train.drop(((train[train['pickup_longitude']>180])).index, axis=0)


train.shape


train = train.drop(((train[train['dropoff_latitude']<-90])).index, axis=0)
train = train.drop(((train[train['dropoff_latitude']>90])).index, axis=0)


train = train.drop(((train[train['dropoff_longitude']<-180])).index, axis=0)
train = train.drop(((train[train['dropoff_longitude']>180])).index, axis=0)


train.shape




