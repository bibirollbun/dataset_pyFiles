# lazy predict
!pip install lazypredict -q


#
# Libraries
#

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, string, re, random, gc, pickle, math,warnings
import json
from datetime import date
from tqdm.keras import TqdmCallback
from tqdm import tqdm

# Sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.metrics import pairwise
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.ensemble import *
import sklearn

# Stats
import scipy
from scipy.stats import *
from scipy.sparse import csr_matrix

# Lazy Predict
from lazypredict.Supervised import LazyRegressor

# Setting
pd.set_option('max_colwidth',None)
seed = random.randint(1,100)
warnings.simplefilter('ignore')


#
# Data
#

# path
train = '/kaggle/input/molecular-machine-learning/train.csv'
test = '/kaggle/input/molecular-machine-learning/test.csv'

# load
df_train = pd.read_csv(train,index_col='Batch_ID')
df_test = pd.read_csv(test,index_col='Batch_ID')

# stats
print(f"Training data-shape: {df_train.shape} | Test data-shape: {df_test.shape}\n")

# view
df_train.head()


#
# Pre-process
#

# drop columns
train_ds = df_train.drop(['Smiles'], axis=1, inplace=False)
test_ds = df_test.drop(['Smiles','T80'], axis=1, inplace=False)

# view
train_ds.head()


#
# Lazy Prediction
#

x = train_ds.loc[:, train_ds.columns != 'T80']
y = train_ds[['T80']]

x, y = shuffle(x, y, random_state=seed)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)

#print(models)


#
# Pipeline - LarsCV Regressor (finding best params)
#

# pipeline
pipe_lcv = Pipeline([
    ('scaler', MinMaxScaler()),
    ('lcv', LarsCV())
])

# param-grid
param_grid = {
    'lcv__max_iter': [500, 800, 1000],
    'lcv__cv': [5, 8, 10],
    'lcv__max_n_alphas': [1000,1500,2000],
}

# Search CV
search_lcv = GridSearchCV(pipe_lcv, param_grid, cv=5)
search_lcv.fit(x_train, y_train)
print(f"Best params for LarsCV Regressor: {search_lcv.best_params_}")


#
# Pipeline - LassoLarsCV Regressor (finding best params)
#

# pipeline
pipe_llcv = Pipeline([
    ('scaler', MinMaxScaler()),
    ('llcv', LassoLarsCV())
])

# param-grid
param_grid = {
    'llcv__max_iter': [500, 800, 1000],
    'llcv__cv': [5, 8, 10],
    'llcv__max_n_alphas': [1000,1500,2000],
}

# Search CV
search_llcv = GridSearchCV(pipe_llcv, param_grid, cv=5)
search_llcv.fit(x_train, y_train)
print(f"Best params for LassoLarsCV Regressor: {search_llcv.best_params_}")


#
# Pipeline - LassoCV Regressor (finding best params)
#

# pipeline
pipe_lacv = Pipeline([
    ('scaler', MinMaxScaler()),
    ('lacv', LassoCV(random_state=seed))
])

# param-grid
param_grid = {
    'lacv__eps': [1e-3, 1e-2, 1e-1],
    'lacv__n_alphas': [100, 200, 300],
    'lacv__max_iter': [1000,1500,2000],
    'lacv__cv': [5, 8, 10],
    'lacv__tol':[1e-4,1e-3,1e-2]
}

# Search CV
search_lacv = GridSearchCV(pipe_lacv, param_grid, cv=5)
search_lacv.fit(x_train, y_train)
print(f"Best params for LassoCV Regressor: {search_lacv.best_params_}")


#
# Pipeline - ElasticNetCV Regressor (finding best params)
#

# pipeline
pipe_encv = Pipeline([
    ('scaler', MinMaxScaler()),
    ('encv', ElasticNetCV(random_state=seed))
])

# param-grid
param_grid = {
    'encv__l1_ratio': [0.5,0.1,0.7],
    'encv__eps': [1e-3, 1e-2, 1e-1],
    'encv__n_alphas': [100, 200, 300],
    'encv__max_iter': [1000,1500,2000],
    'encv__cv': [5, 8, 10],
    'encv__tol':[1e-4,1e-3,1e-2]
}

# Search CV
search_encv = GridSearchCV(pipe_encv, param_grid, cv=5)
search_encv.fit(x_train, y_train)
print(f"Best params for ElasticNetCV Regressor: {search_encv.best_params_}")


#
# Pipeline - KNN Regressor (finding best params)
#

# pipeline
pipe_k = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knnr', KNeighborsRegressor())
])

# param-grid
param_grid = {
    'knnr__n_neighbors': [5,10,15],
    'knnr__leaf_size': [30, 60, 90]
}

# Search CV
search_kr = GridSearchCV(pipe_k, param_grid, cv=5)
search_kr.fit(x_train, y_train)
print(f"Best params for KNN Regressor: {search_kr.best_params_}")


#
# Pipeline - Voting Regressor
#

# LarsCV regressor
#r1 = LarsCV(**search_lcv.best_params_)
r1 = search_lcv.best_estimator_

# LassoLarsCV regressor
#r2 = LassoLarsCV(**search_llcv.best_params_)
r2 = search_llcv.best_estimator_

# LassoCV regressor
#r3 = LassoCV(**search_lacv.best_params_)
r3 = search_lacv.best_estimator_

# ElasticNet regressor
#r4 = ElasticNetCV(**search_encv.best_params_)
r4 = search_encv.best_estimator_

# KNN regressor
#r5 = KNeighborsRegressor(**search_kr.best_params_)
r5 = search_kr.best_estimator_

# pipeline
pipe_v = Pipeline([
    ('scaler', StandardScaler()),
    ('votingR', VotingRegressor([('LarsCV', r1),
                                 ('LassoLarsCV', r2),
                                 ('LassoCV', r3),
                                 ('ElasticNetCV', r4),
                                 ('KNN', r5)]))
])

# fit
pipe_v.fit(x_train, y_train)

# predict
y_pred = pipe_v.predict(x_test)

# rmse
mse = mean_squared_error(y_test,y_pred)
rmse = (mse)**(1/2)
print(f"Root Mean Squared Error: {rmse}")


#
# Prediction on Test-Data
#

ss = StandardScaler()
test_ds_scaled = ss.fit_transform(test_ds)

batch_col = test_ds.index.tolist()
T80_pred = list(pipe_v.predict(test_ds))


#
# Submission
# 

T80_pred_rnd = [np.round(i,2) for i in T80_pred]

submission = {
                'Batch_ID':batch_col,
                'T80':T80_pred_rnd
             }

df_sub = pd.DataFrame(submission)

# export 
df_sub.to_csv('submission.csv',index=False)

