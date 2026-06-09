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


import matplotlib.pyplot as plt


# loading the data set
X_train = pd.read_csv('../input/playground-series-s5e11/train.csv')
X_test = pd.read_csv('../input/playground-series-s5e11/test.csv')


X_train['new_rate'] = X_train['interest_rate'].round()
X_test['new_rate'] = X_test['interest_rate'].round()


# basic information, tells about the data types and non-null count
X_train.info()


X_train.describe()


# Creating a heat map using correlation matrix
train_num = X_train.select_dtypes(exclude = object)
label = list(train_num.columns)

plt.figure(figsize=(16, 6))
plt.imshow(train_num.corr(), cmap = 'RdYlGn')
plt.xticks(ticks=range(len(label)), labels=label, rotation=90)
plt.yticks(ticks=range(len(label)), labels=label)

plt.colorbar()

plt.show()


# boc plots for all numerical features
i=1
plt.figure(figsize=(16,14))
for col in train_num.drop(['id', 'loan_paid_back'], axis = 1).columns:
    plt.subplot(2,3,i)
    i+=1
    plt.boxplot(train_num[col], patch_artist = True)
    plt.xticks(ticks=[1], labels = [col])
plt.show()


def cat_distribute(col):
    print(f"Plotting over {col}")
    plt.figure(figsize=(16, 4))
    plt.subplot(1,3,1)
    index = X_train.groupby(col)[col].count().index
    plt.bar(index, X_train.groupby(col)[col].count())
    plt.title('Overall Distribution')
    plt.xticks(rotation=40)

    plt.subplot(1,3,2)
    plt.bar(index, X_train[X_train.loan_paid_back==1].groupby(col)[col].count())
    plt.title('Those who paid back their loans')
    plt.xticks(rotation=40)
    
    plt.subplot(1,3,3)
    plt.bar(index, X_train[X_train.loan_paid_back==0].groupby(col)[col].count())
    plt.title('Loans Unpaid')
    plt.xticks(rotation=40)
    plt.show()

cat_distribute('gender')
cat_distribute('marital_status')
cat_distribute('education_level')
cat_distribute('loan_purpose')
cat_distribute('grade_subgrade')


def scatter(col1, col2):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.scatter(X_train[col1][loan_paid], X_train[col2][loan_paid], color = 'b', label = 'loan paid')
    plt.scatter(X_train[col1][loan_unpaid], X_train[col2][loan_unpaid], color = 'r', label = 'loan not paid')
    plt.xlabel(f'{col1}')
    plt.ylabel(f'{col2}')

    plt.title(f"{col1} VS {col2}")
    plt.legend()

    plt.subplot(1,2,2)
    plt.scatter(X_test[col1], X_test[col2], color = 'c', label = col1)
    plt.xlabel(f'{col1}')
    plt.ylabel(f'{col2}')

    plt.title(f"{col1} VS {col2}")
    plt.legend()
    
    plt.show()

loan_paid = X_train['loan_paid_back']==1
loan_unpaid = X_train['loan_paid_back']==0


scatter('loan_amount', 'annual_income')


scatter('debt_to_income_ratio', 'credit_score')


scatter('interest_rate', 'credit_score')


scatter('debt_to_income_ratio', 'annual_income')


# importing useful libraries

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# Extracting categorical data
X_cat = X_train.drop(['id', 'annual_income', 'loan_amount', 'loan_paid_back'], axis = 1)
test_cat = X_test.drop(['id', 'annual_income', 'loan_amount'], axis = 1)
X_cat.head()


# One Hot Encoding

oh = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
oh_X = pd.DataFrame(oh.fit_transform(X_cat))
oh_X.columns = oh_X.columns.astype('category')
oh_X = oh_X.astype('category')

oh_t = pd.DataFrame(oh.transform(test_cat))
oh_t.columns = oh_t.columns.astype('category')
oh_t = oh_t.astype('category')

oh_X.head()


# Extracting Numerical Features
X = X_train.select_dtypes(include = np.number)
test = X_test.select_dtypes(include = np.number)

X.drop('id', axis = 1, inplace = True)
X.head()

# Merging
X = pd.concat([X, oh_X], axis = 1)
X_test = pd.concat([test, oh_t], axis = 1)


y = X.pop('loan_paid_back')
testID = X_test.pop('id')


model = XGBClassifier(n_estimators = 10000,
                      early_stopping_rounds = 2000,
                      tree_method = 'hist',
                      device = 'cuda',
                      eval_metric = 'logloss',
                      objective = 'binary:logistic',
                      enable_categorical = True,
                      min_child_weight = 89,
                      max_leaves = 4, 
                      reg_alpha = 3.2,
                      reg_lambda = 5,
                      eta = 0.1, 
                      random_state=42)
"""from sklearn.linear_model import LogisticRegression
model = LogisticRegression()"""


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


#cross_val_score(model, X, y, cv = 5, scoring = 'roc_auc').mean()


model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=500)


roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


final = model.predict_proba(X_test)[:,1]

final = pd.DataFrame({'id': testID, 'loan_paid_back' : final})

final.head()


final.to_csv('submission.csv', index = False)

