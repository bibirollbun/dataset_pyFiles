# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


df.head(5)


df.isnull().sum()


sns.heatmap(df.isnull())


df.info()


df.drop(columns=['id'],inplace=True)



for i in df.columns:
    if df[i].dtype=='object':
        df[i].fillna(df[i].mode()[0],inplace=True)
    else:
        df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean(),inplace=True)


df.isnull().sum()


df.columns


plt.plot(pd.DataFrame(df.groupby('Brand')['Price'].mean()))



sns.countplot(x=df['Color'])
plt.xticks(rotation=30)


plt.plot(pd.DataFrame(df.groupby('Color')['Price'].mean()))



scale=StandardScaler()
label=LabelEncoder()


for i in df.columns:
    if df[i].dtype=='object':
        df[i]=label.fit_transform(df[i])



X_train,X_test,y_train,y_test=train_test_split(df.drop(columns=['Price']),df['Price'])


X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.fit(X_test)


sgd=SGDRegressor(max_iter=100,alpha=0.1,penalty='l1')


sgd.fit(X_train,y_train)


y_pred=sgd.predict(X_test)


r2_score(y_test,y_pred)


rf=RandomForestRegressor(n_estimators=10,bootstrap=True)


rf.fit(X_train,y_train)


y_pred=rf.predict(X_test)


r2_score(y_test,y_pred)




