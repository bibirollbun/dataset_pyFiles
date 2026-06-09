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


sample = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


train.head()


def encode(data):
    data["Sex"] = (data["Sex"] == 'male').astype(int)
    return data
train = encode(train)
test = encode(test)


from sklearn.preprocessing import StandardScaler,QuantileTransformer,MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV,train_test_split


pipe1 = Pipeline([
    ('scale',StandardScaler()),
    ('model',LinearRegression())
])
pipe2 = Pipeline([
    ('scale',QuantileTransformer(random_state = 42)),
    ('model',LinearRegression())
])
pipe3 = Pipeline([
    ('scale',MinMaxScaler()),
    ('model',LinearRegression())
])
#pipes 1-3 have LinearRegression()

pipe4 = Pipeline([
    ('scale',StandardScaler()),
    ('model',KNeighborsRegressor())
])
pipe5 = Pipeline([
    ('scale',QuantileTransformer(random_state = 42)),
    ('model',KNeighborsRegressor())
])
pipe6 = Pipeline([
    ('scale',MinMaxScaler()),
    ('model',KNeighborsRegressor())
])
#pipes 4-6 have KNeighborsRegressor()

pipe7 = Pipeline([
    ('scale',StandardScaler()),
    ('model',RandomForestRegressor(n_jobs = -1,random_state = 42,))
])
pipe8 = Pipeline([
    ('scale',QuantileTransformer(random_state = 42)),
    ('model',RandomForestRegressor(random_state = 42))
])
pipe9 = Pipeline([
    ('scale',MinMaxScaler()),
    ('model',RandomForestRegressor(random_state = 42))
])
#pipes 7-9 have RandomForestRegressor()


param1 = {}
param2 = {'scale__n_quantiles':[1000,1500]}
param3 = {}

param4 = {'model__n_neighbors':[3,5]}
param5 = {'scale__n_quantiles':[1000,1500],'model__n_neighbors':[3,5]}
param6 = {'model__n_neighbors':[3,5]}

param7 = {'model__max_depth':[3,4]}
param8 = {'scale__n_quantiles':[1000,1500],'model__max_depth':[3,4]}
param9 = {'model__max_depth':[3,4]}


names = ['SS_LR','QT_LR','MMS_LR','SS_KNN','QT_KNN','MMS_KNN','SS_RFR','QT_RFR','MMS_RFR']
pipes = [pipe1,pipe2,pipe3,pipe4,pipe5,pipe6,pipe7,pipe8,pipe9]
params = [param1,param2,param3,param4,param5,param6,param7,param8,param9]


X = train.drop('Calories',axis = 1)
X = X.drop('id',axis = 1)
y = train["Calories"]
X_train,X_test,y_train,y_test = train_test_split(X,y)


#Making a RMSLE function to return the error.
def rmsle_function(y_true,y_pred):
    y_true = abs(y_true)
    y_pred = abs(y_pred)
    return np.sqrt(metrics.mean_squared_log_error(y_true,y_pred))

rmsle_scorer = metrics.make_scorer(rmsle_function,greater_is_better = False)


'''for name,param,pipe in zip(names,params,pipes):
    grid = GridSearchCV(estimator = pipe,param_grid = param,n_jobs = -1,cv = 4,scoring = rmsle_scorer)
    grid.fit(X_train,y_train)
    y_pred = grid.predict(X_test)
    error = rmsle_function(y_test,y_pred)
    print(f"{name}:{error}")
'''


df = test.drop(['id'],axis = 1)
paitient_id = test['id']


pipe1 = Pipeline([
    ('scale',MinMaxScaler()),
    ('model',KNeighborsRegressor(n_neighbors = 7))
])

pipe2 = Pipeline([
    ('scale',MinMaxScaler()),
    ('model',KNeighborsRegressor(n_neighbors = 9))
])


pipe1.fit(X,y)
pipe2.fit(X,y)
y_pred1 = pipe1.predict(df)
y_pred2 = pipe2.predict(df)


submission3 = pd.DataFrame({
    'id':paitient_id,
    'Calories':y_pred1
})
submission4 = pd.DataFrame({
    'id':paitient_id,
    'Calories':y_pred2
})


submission3.to_csv('submission3.csv',index = False)
submission4.to_csv('submission4.csv',index = False)

