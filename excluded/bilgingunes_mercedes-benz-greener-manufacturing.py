import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings("ignore")


train=pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')


test=pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip')


sample=pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip')


train.head()


train.shape


test.head()


sample.head()


test.shape


test.isnull().sum()


train.info()


train.isnull().sum()


train.isnull().sum().sum()


train.describe()


train.corr(numeric_only=True) #bir değişkenin değeri hep sabitse korelasyon NaN çıkabilir


import seaborn as sns
sns.histplot(train,x='y')


sns.countplot(train,x='X350')


sns.countplot(train,x='X150')


sns.boxplot(train,x='y')


percentile_95 = train['y'].quantile(0.95)
train = train[train['y'] <= percentile_95]


train.shape#4209'dan 3998 satıra düştü


test.shape


df = pd.concat(objs=[train, test], axis=0) #getdummies uygulamadan önce train ve test birleştirdik. 
#ayrı ayrı yapsaydık sütunlar karışabilirdi


df.shape


cleandf=pd.get_dummies(df) #sözel değerleri T/F olarak düzenledi


cleandf.head()


cleandf.shape


cleandf.drop(['ID'],axis=1,inplace=True)


train=cleandf[:3998]
test=cleandf[3998:] #tekrar train ve test olarak böldük


train.shape


test.shape #get dummies yaptığım için sütnl sayısı arttı


correlations=train.corr(numeric_only=True).y.sort_values(ascending=False)
#en yüksek korelasypndan en düşüğe sırala


correlations #en çok etkileyen faktör x314müş ve bazı sütunlarla arasında bağlantı yokmuş


correlations=pd.DataFrame(correlations) #dataframe haline getir


correlations


nan_corr_columns = correlations.index[correlations['y'].isna()].tolist()
#y ile korelasyonları nan olan sütunları bul ve listele


nan_corr_columns


train = train.drop(columns = nan_corr_columns)
test = test.drop(columns = nan_corr_columns)
#bu sütunları düşür


train.shape #576'den 540 sütuna düştü


test.shape


x_train = train.drop(['y'], axis=1)
y_train = train['y']


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


test = test.drop(['y'], axis=1)


pred=GBR.predict(test)


pred


pred.max()  # En büyük değer


pred.min()  # En küçük değer


sample.head()


submission_ids= sample['ID']
submission = pd.DataFrame({'ID': submission_ids, 'y': pred})
submission.to_csv('submission1.csv', index=False)
submission
#id sütnunu silmiştik kaggle sistem formatına uygun olması için tekrar sample dosyasından çekerek geri ekledik
#yükleyebilmek için tahmini csv dosyası olarak dışarı aktardık




