import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',None)


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import normalize, scale

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


train_df=pd.read_csv('/kaggle/input/playground-series-s3e5/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s3e5/test.csv')


train_df.head()


train_df=train_df.drop('Id', axis=1)


train_df.info()


train_df.describe()


train_df.corr()


train_df.shape


train_df.isnull().sum()


train_df['quality'].value_counts()


# Distribution of quality
plt.figure(figsize=(10, 6))
sns.histplot(train_df['quality'], bins=30, kde=True)
plt.title('Distribution of quality', fontsize=16)
plt.xlabel('quality', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid()
plt.show()


# Relationship between alcohol and quality
plt.figure(figsize=(10, 6))
sns.scatterplot(x='alcohol', y='quality', data=train_df, alpha=0.6)
plt.title('alcohol vs quality', fontsize=16)
plt.xlabel('alcohol', fontsize=12)
plt.ylabel('quality', fontsize=12)
plt.grid()
plt.show()


# Box plot of quality by sulphates
plt.figure(figsize=(10, 6))
sns.boxplot(x='sulphates', y='quality', data=train_df)
plt.title('quality distribution by sulphates', fontsize=16)
plt.xlabel('sulphates', fontsize=12)
plt.ylabel('quality', fontsize=12)
plt.grid()
plt.show()


# Average quality by density
plt.figure(figsize=(12, 6))
avg_quality_by_density = train_df.groupby('density')['quality'].mean().sort_values(ascending=False)
sns.barplot(x=avg_quality_by_density.index, y=avg_quality_by_density.values, palette='viridis')
plt.title('Average Quality by Density', fontsize=16)
plt.xlabel('Density', fontsize=12)
plt.ylabel('Average Quality', fontsize=12)
plt.xticks(rotation=45)
plt.grid()
plt.show()




# Feature Engineering
train_df['acidity_sugar'] = train_df['fixed acidity'] * train_df['residual sugar']
train_df['density_alcohol'] = train_df['density'] * train_df['alcohol']


# Log Transformation
train_df['fixed acidity'] = np.log1p(train_df['fixed acidity'])
train_df['residual sugar'] = np.log1p(train_df['residual sugar'])


# Scaling
scaler = StandardScaler()
features = train_df.drop(['quality'], axis=1)
features_scaled = scaler.fit_transform(features)


# Splitting the data
X = features_scaled
y = train_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def algo_test(x,y):
        #Bütün modelleri tanımlıyorum
        L=LinearRegression()
        R=Ridge()
        Lass=Lasso()
        E=ElasticNet()
        sgd=SGDRegressor()
        ETR=ExtraTreeRegressor()
        GBR=GradientBoostingRegressor()
        kn=KNeighborsRegressor()
        rkn=RadiusNeighborsRegressor(radius=1.0)
        ada=AdaBoostRegressor()
        dt=DecisionTreeRegressor()
        xgb=XGBRegressor()
        svr=SVR()
        mlp_regressor = MLPRegressor()
        rf=RandomForestRegressor()

       
        
        algos=[L,R,Lass,E,sgd,ETR,GBR,ada,kn,dt,xgb,svr,mlp_regressor,rf]
        algo_names=['Linear','Ridge','Lasso','ElasticNet','SGD','Extra Tree','Gradient Boosting',
                    'KNeighborsRegressor','AdaBoost','Decision Tree','XGBRegressor','SVR','mlp_regressor','RandomForestRegressor']
        
        x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=.20,random_state=42)
        
        r_squared= []
        rmse= []
        mae= []
        
        #Hata ve doğruluk oranlarını bir tablo haline getirmek için bir dataframe oluşturuyorum
        result=pd.DataFrame(columns=['R_Squared','RMSE','MAE'],index=algo_names)
        
        
        for algo in algos:
            p=algo.fit(x_train,y_train).predict(x_test)
            r_squared.append(r2_score(y_test,p))
            rmse.append(mean_squared_error(y_test,p)**.5)
            mae.append(mean_absolute_error(y_test,p))
        
            

        #result adlı tabloya doğruluk ve hata oranlarımı yerleştiriyorum
        result.R_Squared=r_squared
        result.RMSE=rmse
        result.MAE=mae
        
       #oluşturduğum result tablosunu doğruluk oranına (r2_score) göre sıralayıp dönüyor
        rtable=result.sort_values('R_Squared',ascending=False)
        return rtable


algo_test(X,y)




