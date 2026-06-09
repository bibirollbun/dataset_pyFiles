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


df_train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')


df_train.head(5)


df_train.info()


df_train.describe()


df_train.drop(['id','CustomerId','Surname'], axis = 1, inplace = True)



df_test.drop(['id','CustomerId','Surname'], axis = 1, inplace = True)


df_train['Geography'].unique()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


df_train['Geography']= le.fit_transform(df_train['Geography'])
df_train['Gender']= le.fit_transform(df_train['Gender'])


df_test['Geography']= le.fit_transform(df_test['Geography'])
df_test['Gender']= le.fit_transform(df_test['Gender'])


df_train.head()


df_test.head()


X_train = df_train.drop('Exited', axis=1)
Y_train = df_train['Exited']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X_train_sc= scaler.fit_transform(X_train)


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()


from sklearn.model_selection import GridSearchCV
parameters={'penalty':['l1','l2', 'elasticnet'], 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter':[90, 92,96, 95, 100, 1000, 2500, 5000]}


log_cv = GridSearchCV(log, parameters, scoring='accuracy', cv=5)


log_cv.fit(X_train, Y_train)


print(log_cv.best_params_)
print(log_cv.best_score_)


y_pred = log_cv.predict(df_test)
sample_sub=pd.read_csv("/kaggle/input/playground-series-s4e1/sample_submission.csv")


sample_sub['Exited']=y_pred
results = sample_sub[['id', 'Exited']]
results.to_csv('/kaggle/working/submission.csv', index=False)
results.head()

