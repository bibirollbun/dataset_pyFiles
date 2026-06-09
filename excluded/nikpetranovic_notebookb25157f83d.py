# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
predict_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.info()


train.describe().T


train.isnull().sum()


predict_data.describe().T


predict_data.isnull().sum()


# histogram stuff for train
plt.figure(figsize=(24,9))
for j in range(13):
    if j == 0 or j == 12: #id and rainfall respectively
        continue
    plt.subplot(2,6,j)
    plt.hist(train.iloc[:,j], bins = 50)
    plt.xlabel(train.columns[j])
    plt.ylabel('Count')


# histogram stuff for prediction data
plt.figure(figsize=(24,9))
for j in range(13):
    if j == 0 or j == 12: #id and rainfall respectively
        continue
    plt.subplot(2,6,j)
    plt.hist(predict_data.iloc[:,j], bins = 50)
    plt.xlabel(predict_data.columns[j])
    plt.ylabel('Count')


imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(predict_data)

predict_data_numpy = imp.transform(predict_data)

predict_data = pd.DataFrame(predict_data_numpy, columns = predict_data.columns)

predict_data.isnull().sum()


plt.figure(figsize=(24,24))
for j in range(13):
    for k in range (13):
        if j == 0 or j == 12: #id and rainfall respectively
            continue
        if k == 0 or k == 12: #id and rainfall respectively
            continue
        plt.subplot(11,11,11*(j-1)+k)
        plt.scatter(train.iloc[:,j],train.iloc[:,k])
        #plt.xlabel(train.columns[j])
        #plt.ylabel(train.columns[k])


print(train.columns)
print(train.loc[(train['winddirection'] > 150) & (train['maxtemp'] < 20)])
#df.loc[df['column_name'] == some_value]


print(train.loc[(train['pressure'] > 1030)])


plt.figure(figsize=(24,24))
for j in range(13):
    for k in range (13):
        if j == 0 or j == 12: #id and rainfall respectively
            continue
        if k == 0 or k == 12: #id and rainfall respectively
            continue
        plt.subplot(11,11,11*(j-1)+k)
        plt.scatter(train.iloc[:,j],train.iloc[:,k])
        plt.scatter(train.iloc[409,j],train.iloc[409,k], color = 'r')
        plt.scatter(train.iloc[1122,j],train.iloc[1122,k], color = 'm')
        plt.scatter(train.iloc[[17,383,1828,1840],j],train.iloc[[17,383,1828,1840],k], color = 'c')
        #plt.xlabel(train.columns[j])
        #plt.ylabel(train.columns[k])


train = train.drop([409, 1122,17, 383,1828, 1840])
train.describe().T


X = train.drop('rainfall', axis=1)
y = train['rainfall']


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=68)
score = cross_val_score(lr,X,y,cv=5,scoring='roc_auc')
print(score)
print(score.mean())

# code to run to go through the param grid below, output given in markdown comment below

'''lr = LogisticRegression(random_state=68)

param_grid = {"penalty": ["l1", "l2", "elasticnet", None],
             "class_weight": [None, "balanced"],
             "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
             "max_iter": [1000, 1500, 2000]}

from sklearn.model_selection import RandomizedSearchCV

h = RandomizedSearchCV(lr, param_distributions = param_grid, n_iter = 100, cv=5, verbose = True, n_jobs=-1)
hh = h.fit(X,y)
hh.best_params_'''


model_lr = LogisticRegression(penalty = 'l2', max_iter = 1000, class_weight = None, C = 0.01)
model_lr.fit(X, y)

#if using this to predict
#y_pred = model_lr.predict_proba(predict_data)[:,1]


from sklearn.ensemble import RandomForestClassifier

rfr = RandomForestClassifier(random_state = 68, n_estimators = 100, min_samples_split = 10, min_samples_leaf = 1)
score = cross_val_score(rfr,X,y,cv=5)
print(score)
print(score.mean())

# seems to do worse than logistic regression, but we shall persist anyway
# code to run to go through the param grid below, output given in markdown comment below

'''
rfr = RandomForestClassifier(random_state = 68)
param_grid = {'n_estimators': [100,200,500],
             'min_samples_leaf': [1,2,5],
             'min_samples_split': [2,5,10]}

h = RandomizedSearchCV(rfr, param_distributions = param_grid, n_iter = 100, cv=5, verbose = True, n_jobs=-1)
hh = h.fit(X,y)
hh.best_params_'''




model_rfr = RandomForestClassifier(random_state = 68, n_estimators = 100, min_samples_split = 10, min_samples_leaf = 2)


from catboost import CatBoostClassifier

cat = CatBoostClassifier(iterations = 100, depth = 4, learning_rate= 0.1, random_state = 68)
cat.fit(X,y)
print(score)
print(score[1:].mean())

# code to run to go through the param grid below, output given in markdown comment below

'''
cat = CatBoostClassifier(random_state = 68, verbose = 0)
param_grid = {'iterations': [75,100,200,500],
             'depth': [2,4,6,8],
             'learning_rate': [0.01,0.03,0.05,0.1,0.15,0.2]}

h = RandomizedSearchCV(cat, param_distributions = param_grid, n_iter = 100, cv=5, verbose = True, n_jobs=-1)
hh = h.fit(X,y)
hh.best_params_'''


model_cat = CatBoostClassifier(iterations = 75, depth = 6, learning_rate= 0.05, random_state = 68)


from sklearn.ensemble import VotingClassifier

my_voting_thing = VotingClassifier(estimators = [('lr', model_lr), ('rf', model_rfr), ('cat', model_cat)], voting = 'soft')

my_voting_thing.fit(X, y)
y_pred = my_voting_thing.predict_proba(predict_data)[:, 1]

output = pd.DataFrame({'id': predict_data.id, 'rainfall': y_pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

