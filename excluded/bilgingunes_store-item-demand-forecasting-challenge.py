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


train=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')
test=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')
sample=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')


train.head()


train.isnull().sum().sum()


test.drop("id",axis=1,inplace=True)


train.shape, test.shape


train.info()


train["date"] = pd.to_datetime(train["date"]) #datetime veri tipine çevir
train["Day"] = train["date"].dt.day #gün verisini çek ve gün sütununa ekle
train["Month"] = train["date"].dt.month #ay
train["Years"] = train["date"].dt.year #yıl
train.drop("date",axis=1,inplace=True) #ayırdığımız için artık orijinal sütuna ihtiyacaımız yok


test["date"] = pd.to_datetime(test["date"]) #datetime veri tipine çevir
test["Day"] = test["date"].dt.day #gün verisini çek ve gün sütununa ekle
test["Month"] = test["date"].dt.month #ay
test["Years"] = test["date"].dt.year #yıl
test.drop("date",axis=1,inplace=True) #ayırdığımız için artık orijinal sütuna ihtiyacaımız yok


train.head()


train["store"].unique()


train["item"].unique()


train["sales"].unique()


x_train = train.drop(['sales'], axis=1)
y_train = train['sales']


x_test = train.drop(['sales'], axis=1)
y_test = train['sales']


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
    
    result = result.sort_values('R_Squared', ascending=False).round(2)
    return result.applymap(lambda x: f"{x:.2f}")



regression_algo(x_train,y_train) 


GBR=GradientBoostingRegressor()#en iyi algoritmamız bu olduğu için tahminini bunun üzerinden alacağız
GBRmodel=GBR.fit(x_train,y_train)


pred=GBR.predict(test)


pred


sample.head()


submission_ids= sample['id']
submission = pd.DataFrame({'id': submission_ids, 'sales': pred})
submission.to_csv('submission.csv', index=False)
submission
#id sütnunu silmiştik kaggle sistem formatına uygun olması için tekrar sample dosyasından çekerek geri ekledik
#yükleyebilmek için tahmini csv dosyası olarak dışarı aktardık




