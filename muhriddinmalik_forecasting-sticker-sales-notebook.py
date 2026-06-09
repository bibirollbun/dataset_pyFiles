import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from math import *
import tensorflow as tf

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split as tts ,GridSearchCV
from sklearn.linear_model import LinearRegression , Lasso , Ridge
from sklearn.ensemble import RandomForestRegressor
from keras.utils import to_categorical

import keras
import warnings
warnings.filterwarnings("ignore")



df= pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df


df.info()


df.isna().sum()


df.num_sold.mean()


plt.plot(df['num_sold'],'*')


most_frequent = df['num_sold'].value_counts().idxmax()
df ['num_sold'] = df['num_sold'].fillna(most_frequent)
df.num_sold.isna().sum()


df.isna().sum()


df_train  =df.copy()


df_train


def split_date(df_train):
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_train['day'],df_train['month'],df_train['year'] = df_train['date'].dt.day, df_train['date'].dt.month , df_train['date'].dt.year
    df_train.drop('date',axis=1,inplace=True)
    return df_train
df_train = split_date(df_train)
df_train


df_train.info()


plt.plot(df_train['num_sold'],'*')


output = df_train['num_sold']
df_train = df_train.drop(['id','num_sold'],axis=1)


df_train.isna().sum()


ohe = OneHotEncoder(sparse=False, drop='first')  # drop='first' for avoiding dummy variable trap
oe = OrdinalEncoder()
def encoder(df_train):
    categ = df_train.drop(['day','month','year'],axis=1)
    cols = categ.columns
    data_arr =ohe.fit_transform(df_train[cols])

    data_frame = pd.DataFrame(
        data_arr, 
        columns=ohe.get_feature_names_out(cols)
    )
    # Combine the encoded data with the original numeric columns
    data_train = pd.concat([df_train.drop(columns=cols),data_frame], axis=1)
    data_train[['day','month','year']] = oe.fit_transform(data_train[['day','month','year']])
    df_train ['day'] = df_train['day'].fillna(0)
    df_train ['month'] = df_train['day'].fillna(0)
    df_train ['year'] = df_train['year'].fillna(0)
    return data_train
data_train = encoder(df_train)
data_train


x_train, x_val , y_train , y_val = tts(data_train,output,test_size=0.3,random_state = 42)
y_train 


model = LinearRegression()
model.fit(x_train,y_train)


model.score(x_val,y_val)


model2 = Lasso(alpha=0.1, random_state=42)
model2.fit(x_train,y_train)


model2.score(x_val,y_val)


model3 = Ridge(alpha=0.1, random_state=42)
model3.fit(x_train,y_train)


model3.score(x_val,y_val)


model4 = RandomForestRegressor()
model4.fit(x_train,y_train)


model4.score(x_val,y_val)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
df_test


df_test.info()


df_test = split_date(df_test)
df_test


ids = df_test['id'].values
df_test.drop('id',axis=1,inplace=True)


data_test = encoder(df_test)
data_test


predictions = model4.predict(data_test)
#predictions_2 = model2.predict(data_test)
#predictions_3 = model3.predict(data_test)


submission = pd.DataFrame({ 
    'id' : ids,
    'num_sold' : predictions
})
submission


submission.to_csv("submission.csv",index=False)

