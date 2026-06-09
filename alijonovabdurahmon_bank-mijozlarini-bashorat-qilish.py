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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


df_train = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
df_test = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')
df_train.head()


df_train.info()


df_train.isnull().sum()


id=df_test['id']
df_train.drop(['id','CustomerId','Surname'],axis=1,inplace=True)
df_test.drop(['id','CustomerId','Surname'],axis=1,inplace=True)
labelencode=LabelEncoder()
df_train['Gender']=labelencode.fit_transform(df_train['Gender'])
df_train['Geography']=labelencode.fit_transform(df_train['Geography'])
df_test['Gender']=labelencode.fit_transform(df_test['Gender'])
df_test['Geography']=labelencode.fit_transform(df_test['Geography'])
df_train.head()


df_test.isnull().sum()


df_train.corrwith(df_train['Exited']).abs().sort_values(ascending=False)


X=df_train.drop('Exited',axis=1)
y=df_train['Exited']


scaler = StandardScaler()
X = scaler.fit_transform(X)
df_test=scaler.fit_transform(df_test)


xgb_model = XGBClassifier()
xgb_model.fit(X, y)


y_pred = xgb_model.predict(df_test)


submission=pd.DataFrame({'Id':id, 'Exited':y_pred})
submission.sample(10)


submission.to_csv('Submission.csv', index=False)


