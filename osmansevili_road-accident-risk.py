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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape) 
print(train.head())


train.info()


train.describe()


train.isnull().sum()


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='road_type')
plt.title('Yol tipi', fontsize=14, fontweight='bold')
plt.xlabel('road_type')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='weather')
plt.title('Hava Durumu', fontsize=14, fontweight='bold')
plt.xlabel('weather')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='lighting')
plt.title('Aydınlatma', fontsize=14, fontweight='bold')
plt.xlabel('lighting')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='time_of_day')
plt.title('Zaman Dilimi', fontsize=14, fontweight='bold')
plt.xlabel('time_of_day')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=train, x='num_lanes')
plt.title('Şerit Sayısı:', fontsize=14, fontweight='bold')
plt.xlabel('num_lanes')
plt.ylabel('Count')
plt.show()


    plt.figure(figsize=(8, 5))
    train['accident_risk'].value_counts().plot(kind='bar')
    plt.title('Accident Risk Dağılımı')
    plt.xlabel('Risk Seviyesi')
    plt.ylabel('Frekans')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


train=pd.get_dummies(train,drop_first=True)
test=pd.get_dummies(test,drop_first=True)


train.head()


test.head()


x=train.drop(['accident_risk'],axis=1)
y=train[['accident_risk']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


lr=LinearRegression()


lr.fit(x,y)


tahmin=lr.predict(x_test)


r2_score(y_test,tahmin)


tahmin


sonuc=pd.DataFrame()
if len(tahmin.shape) > 1:
    tahmin = tahmin.flatten()
sonuc['accident_risk']=tahmin


sonuc


sonuc['id']=test['id']


sonuc


sonuc['accident_risk']=sonuc['accident_risk'].astype('int32')


sonuc.to_csv('sonuc.csv',index=False)




