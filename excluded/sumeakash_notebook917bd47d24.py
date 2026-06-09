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
train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
# 5 random data print form train 
print(train.sample(5))
print(train.shape)


train.describe()


train.isnull().sum()


test.isnull().sum()


train.info()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in train.columns:
    if train[col].dtype == 'object' or train[col].dtype=='bool':
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])      # fit on train
for col in test.columns: 
    if test[col].dtype=='object' or train[col].dtype=='bool':
        le=LabelEncoder()
        test[col]=le.fit_transform(test[col])


train.sample(3)


train.corr()


X=train.drop(columns='accident_risk')
y=train['accident_risk']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


X1=train[['id','curvature','speed_limit','lighting','weather']]
X_train1,X_test1,y_train1,y_test1=train_test_split(X1,y,test_size=0.2,random_state=2)


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
dt=DecisionTreeRegressor(max_depth=7)
dt.fit(X_train1,y_train1)
y_pred2=dt.predict(X_test1)
r2_score(y_test1,y_pred2)


y_test1.iloc[:5]


predict=pd.DataFrame(y_pred2)


test=test[['id','curvature','speed_limit','lighting','weather']]


test['final_predict']=dt.predict(test)
test
submission=test[['id','final_predict']]


submission.to_csv('/kaggle/working/submission.csv', index=False)


