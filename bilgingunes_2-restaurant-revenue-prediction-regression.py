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


train=pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')


test=pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')


sample=pd.read_csv('/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv')


train.head()


train.shape


test.shape


train.isnull().sum().sum()


train["Open Date"] = pd.to_datetime(train["Open Date"]) #datetime veri tipine çevir
train["Day"] = train["Open Date"].dt.day #gün verisini çek ve gün sütununa ekle
train["Day_Name"] = train["Open Date"].dt.day_name() #gün ismi sütunu
train["Month"] = train["Open Date"].dt.month #ay
train["Years"] = train["Open Date"].dt.year #yıl
train.drop("Open Date",axis=1,inplace=True) #ayırdığımız için artık orijinal sütuna ihtiyacaımız yok


#aynı işlemleri test için yaptık
test["Open Date"] = pd.to_datetime(test["Open Date"])
test["Day"] = test["Open Date"].dt.day
test["Day_Name"] = test["Open Date"].dt.day_name()
test["Month"] = test["Open Date"].dt.month
test["Years"] = test["Open Date"].dt.year
test.drop("Open Date",axis=1,inplace=True)


train["Years"] = 2025-train["Years"] #yıl kolonundaki verileri bugünden çıkardık
test["Years"] = 2025-test["Years"] 


train.head()


train=train.drop(['Day_Name'], axis=1)


test=test.drop(['Day_Name'], axis=1)


cityPerc = train[["City Group", "revenue"]].groupby(['City Group'],as_index=False).mean()

citygroupDummy = pd.get_dummies(train['City Group'])
train = train.join(citygroupDummy)

citygroupDummyTest = pd.get_dummies(test['City Group'])
test = test.join(citygroupDummyTest)

train = train.drop('City Group', axis=1)
test = test.drop('City Group', axis=1)


train.head()


train.info()


train.Type.value_counts() #type kolonundaki veri tipleri sayısı


import seaborn as sns
sns.countplot(train,x='Type')


sns.boxplot(train,x='revenue')


percentile_95 = train['revenue'].quantile(0.95)
train = train[train['revenue'] <= percentile_95]


train.shape


test.shape


import matplotlib.pyplot as plt
sns.countplot(train,x='City')
plt.xticks(rotation=90)
plt.show()


df = pd.concat(objs=[train, test], axis=0) #getdummies uygulamadan önce train ve test birleştirdik. 
#ayrı ayrı yapsaydık sütunlar karışabilirdi


df.shape


cleandf=pd.get_dummies(df) #sözel değerleri T/F olarak düzenledi


cleandf.head()


cleandf.shape


cleandf.drop(['Id'],axis=1,inplace=True)


train=cleandf[:130]
test=cleandf[130:] #tekrar train ve test olarak böldük


train.shape


correlations=train.corr(numeric_only=True).revenue.sort_values(ascending=False)
#en yüksek korelasypndan en düşüğe sırala


correlations=pd.DataFrame(correlations) #dataframe haline getir


correlations


nan_corr_columns = correlations.index[correlations['revenue'].isna()].tolist()
#y ile korelasyonları nan olan sütunları bul ve listele


nan_corr_columns


# augment the train data because it is so smaller according to test data 
import numpy as np

num_bootstrap_samples = 1000


bootstrap_samples = []
for _ in range(num_bootstrap_samples):
    
    bootstrap_sample = train.sample(n=len(train), replace=True)
    bootstrap_samples.append(bootstrap_sample)


train = pd.concat(bootstrap_samples)


train.shape


x_train = train.drop(['revenue'], axis=1)
y_train = train['revenue']


x_test = train.drop(['revenue'], axis=1)
y_test = train['revenue']


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



regression_algo(x_train,y_train) #ilk 3 algoritme overfitting sebebiyle %100 başarılı çıktı bunları kullanmayacağız


GBR=GradientBoostingRegressor()#en iyi algoritmamız bu olduğu için tahminini bunun üzerinden alacağız


GBRmodel=GBR.fit(x_train,y_train)


test = test.drop(['revenue'], axis=1)


pred=GBR.predict(test)


pred


pred.max()  # En büyük değer


pred.min()  # En küçük değer


submission_ids= sample['Id']
submission = pd.DataFrame({'Id': submission_ids, 'Prediction': pred})
submission.to_csv('submission5.csv', index=False)
submission
#id sütnunu silmiştik kaggle sistem formatına uygun olması için tekrar sample dosyasından çekerek geri ekledik
#yükleyebilmek için tahmini csv dosyası olarak dışarı aktardık




