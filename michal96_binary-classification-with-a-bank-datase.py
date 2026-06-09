
import numpy as np 
import pandas as pd 
import sklearn.preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


train.head()




categorical_cols = train.select_dtypes(include=['object', 'category']).columns

encoders = {col: sklearn.preprocessing.LabelEncoder() for col in categorical_cols}

for col in categorical_cols:
  encoders[col].fit(train[col])
  train[col] = encoders[col].transform(train[col])


train.head()


train


train.drop("id", axis=1, inplace=True)
y = train['y']
train.drop("y", axis=1, inplace=True)



lr = LogisticRegression()
lr.fit(X = train, y = y)
lr.score(X = train, y = y)


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


for col in categorical_cols:
  test[col] = encoders[col].transform(test[col])


id = test['id']
test.drop("id", axis=1, inplace=True)


pred = lr.predict_proba(test)
pred = pred[:, 1]
sub = pd.DataFrame({'id': id, 'y': pred})
sub.to_csv('submission_log.csv', index = False)


from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(verbose = True, n_estimators = 1000)
rf.fit(X = train, y = y)
rf.score(X = train, y = y)


pred = rf.predict_proba(test)
pred = pred[:, 1]
sub = pd.DataFrame({'id': id, 'y': pred})
sub.to_csv('submission_RF_1000.csv', index = False)


import xgboost as xgb




xb = xgb.XGBClassifier(learning_rate =  0.1, max_depth =10).fit(X = train, y = y)
xb.score(X = train, y = y)


pred = xb.predict_proba(test)
pred = pred[:, 1]
sub = pd.DataFrame({'id': id, 'y': pred})
sub.to_csv('submission_xgb.csv', index = False)






param_grid = {
    'max_depth': [10],
    'learning_rate': [0,18, 0.16,0.14,0.12, 0.1, 0.08, 0.06, 0.04, 0.02],
         }  

grid = GridSearchCV(xgb.XGBRegressor(silent=True)
                   ,param_grid
                   ,n_jobs=1
                   ,cv=3
                   ,scoring='r2'
                   ,verbose=1
                   ,refit=True)


grid.fit(train, y = y)


grid.best_params_







