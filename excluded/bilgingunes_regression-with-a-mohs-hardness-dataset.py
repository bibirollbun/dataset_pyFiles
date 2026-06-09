import seaborn as sns
import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train=pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')


test=pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')


sample=pd.read_csv('/kaggle/input/playground-series-s3e25/sample_submission.csv')


train.shape


train.head()


test.shape


sample.shape


sample.head()


train.info()


train.isnull().sum()


train.corr()


import matplotlib.pyplot as plt

sns.set(font_scale=1)
plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(numeric_only=True), annot=True, fmt=".2f")

plt.show()


sns.histplot(train,x='Hardness')


sns.histplot(train,x='density_Average')


sns.boxplot(train,x='density_Total')


percentile_95 = train['density_Total'].quantile(0.95)
train = train[train['density_Total'] <= percentile_95]


train.shape


sns.boxplot(train,x='density_Total')





x = train.drop(['id', 'Hardness','atomicweight_Average'], axis=1)
y = train['Hardness']


x.shape,y.shape


train.head()


x.head()


y.head()


from sklearn.metrics import r2_score, mean_squared_error


from sklearn.linear_model import LinearRegression


from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


def regression_algo(x,y):
    
    L = LinearRegression()
    E = ElasticNet()
    R = Ridge()
    Lass = Lasso()
    ETR=ExtraTreeRegressor()
    GBR=GradientBoostingRegressor()
    XGBC= XGBRegressor()
    RFR=RandomForestRegressor()
          
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=13)
    
    algos = [L,E,R,Lass,ETR,GBR,XGBC,RFR]
    algo_names = ['Linear','ElasticNet','Ridge','Lasso','Extra Tree','Gradient Boosting','XGradientBooting','Random Forest']
    r_squared = []
    rmse = []
    mae = []
        
    result = pd.DataFrame(columns = ['R_Squared','RMSE','MAE'],index = algo_names)
       
    for algo in algos:
        algo.fit(x_train,y_train)
             
        r_squared.append(r2_score(y_test,algo.predict(x_test)))
        rmse.append(mean_squared_error(y_test, algo.predict(x_test))**.5)
        mae.append(mean_absolute_error(y_test, algo.predict(x_test)))

    result.R_Squared = r_squared
    result.RMSE = rmse
    result.MAE= mae
    
    return result.sort_values('R_Squared', ascending=False)


regression_algo(x,y)


GBR=GradientBoostingRegressor()


GBRmodel=GBR.fit(x,y)


test = test.drop(['id', 'atomicweight_Average'], axis=1)


pred=GBR.predict(test)


pred


submission_ids= sample['id']
submission = pd.DataFrame({'id': submission_ids, 'Hardness': pred})
submission.to_csv('submission.csv', index=False)
submission




