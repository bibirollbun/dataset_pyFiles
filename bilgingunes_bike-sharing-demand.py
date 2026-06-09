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


train=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
sample=pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')


train.head()


test.head()


sample.head()


train.isnull().sum().sum()


train.shape, test.shape


train["datetime"] = pd.to_datetime(train["datetime"]) #datetime veri tipine çevir
train["Day"] = train["datetime"].dt.day #gün verisini çek ve gün sütununa ekle
train["Month"] = train["datetime"].dt.month #ay
train["Years"] = train["datetime"].dt.year #yıl
train.drop("datetime",axis=1,inplace=True) #ayırdığımız için artık orijinal sütuna ihtiyacaımız yok


#aynısı test için
test["datetime"] = pd.to_datetime(test["datetime"]) #datetime veri tipine çevir
test["Day"] = test["datetime"].dt.day #gün verisini çek ve gün sütununa ekle
test["Month"] = test["datetime"].dt.month #ay
test["Years"] = test["datetime"].dt.year #yıl
test.drop("datetime",axis=1,inplace=True) #ayırdığımız için artık orijinal sütuna ihtiyacaımız yok


train.head()


train = train.drop(['casual', 'registered'], axis=1)#toplam sayım yeterli olur





import seaborn as sns
sns.countplot(train,x='holiday')


sns.countplot(train,x='weather')





sns.countplot(train,x='season')


sns.boxplot(train,x='count')


import matplotlib.pyplot as plt

sns.set(font_scale=1)
plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(numeric_only=True), annot=True, fmt=".2f")

plt.show()


train=pd.get_dummies(train, columns=['weather', 'season'], drop_first=True)
test=pd.get_dummies(test, columns=['weather', 'season'], drop_first=True)


x_train = train.drop(['count'], axis=1)
y_train = train['count']


x_test = train.drop(['count'], axis=1)
y_test = train['count']


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor 
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error
import numpy as np

def regression_algo(x, y):
    L = LinearRegression()
    E = ElasticNet()
    R = Ridge()
    Lass = Lasso()
    ETR = ExtraTreeRegressor()
    GBR = GradientBoostingRegressor()
    XGBC = XGBRegressor(objective='reg:squarederror')
    RFR = RandomForestRegressor()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)

    algos = [L, E, R, Lass, ETR, GBR, XGBC, RFR]
    algo_names = ['Linear', 'ElasticNet', 'Ridge', 'Lasso', 'Extra Tree', 'Gradient Boosting', 'XGradientBooting', 'Random Forest']

    r_squared = []
    rmse = []
    mae = []
    rmsle = []

    result = pd.DataFrame(columns=['R_Squared', 'RMSE', 'MAE', 'RMSLE'], index=algo_names)

    for algo in algos:
        algo.fit(x_train, y_train)
        predictions = algo.predict(x_test)

        predictions[predictions < 0] = 0
        y_test_non_negative = y_test.copy()
        y_test_non_negative[y_test_non_negative < 0] = 0

        r_squared.append(r2_score(y_test, predictions))
        rmse.append(mean_squared_error(y_test, predictions)**.5)
        mae.append(mean_absolute_error(y_test, predictions))
        rmsle.append(np.sqrt(mean_squared_log_error(y_test_non_negative, predictions)))

    result.R_Squared = r_squared
    result.RMSE = rmse
    result.MAE = mae
    result.RMSLE = rmsle

    result = result.sort_values('RMSLE', ascending=True).round(2)
    return result.applymap(lambda x: f"{x:.2f}")


regression_algo(x_train,y_train) #bu yarışmada RMSLE önemli ona göre sıraladık


from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()


RFRmodel=RFR.fit(x_train,y_train)


pred=RFR.predict(test)


pred


sample.head()


submission_ids= sample['datetime']
submission = pd.DataFrame({'datetime': submission_ids, 'count': pred})
submission.to_csv('submission1.csv', index=False)
submission




