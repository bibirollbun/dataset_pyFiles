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


d=pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


d.head()


d.info()


d.describe()


d.dtypes


d.index


d.columns


d.shape


d.isnull().sum()


import matplotlib.pyplot as plt 
import seaborn as sns
plt.figure(figsize=(12,6))
sns.boxplot(x="road_type", y="accident_risk", data=d)
plt.title("Accident Risk Distribution by Road Type")
plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(x="lighting", y="accident_risk", data=d)
plt.title("Accident Risk by Lighting Conditions")
plt.show()


p= d.pivot_table(values="accident_risk", 
                             index="time_of_day", 
                             columns="weather", 
                             aggfunc="mean")
sns.heatmap(p,annot=True,cmap='coolwarm')


d1=pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


d1.head()


d1.info()


d1.describe()


d1.isnull().sum()


d1.shape


d1.columns


d.drop(columns=['id'], inplace=True)
d1.drop(columns=['id'], inplace=True)


X = d.drop(columns=['accident_risk'])


X.shape


y=d['accident_risk']


X_val= d1


X


cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols= X.select_dtypes(include=[np.number]).columns.tolist()


from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


imp= SimpleImputer(strategy='mean')
X[num_cols] = imp.fit_transform(X[num_cols])
X_val[num_cols] = imp.transform(X_val[num_cols])

imp1= SimpleImputer(strategy='most_frequent')
X[cat_cols] = imp1.fit_transform(X[cat_cols])
X_val[cat_cols] = imp1.transform(X_val[cat_cols])


l= {}
for i in cat_cols:
    le = LabelEncoder()
    X[i] = le.fit_transform(X[i].astype(str))
    X_val[i] = le.transform(X_val[i].astype(str))
    l[i] = le


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


X_train.shape


X_test.shape


import lightgbm as lgb

lgb_params = {
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'regression',
    'metric': 'rmse',
    'random_state': 42
}


l= lgb.LGBMRegressor(**lgb_params)
l.fit(X, y, eval_set=[(X_test, y_test)], eval_metric='rmse', callbacks=[lgb.early_stopping(100)])


l_pred= l.predict(X_val)


from sklearn.metrics import mean_squared_error


l_pred1= l.predict(X_test)


np.sqrt(mean_squared_error(y_test, l_pred1))


sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
sub['accident_risk'] = l_pred
sub.to_csv("submission.csv",index=False)
sub.head()


sub['accident_risk'].hist()


sub.shape

