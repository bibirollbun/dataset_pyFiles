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


# importin useful libraries

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# loading the data

X_full = pd.read_csv('../input/playground-series-s5e11/train.csv')
X_test = pd.read_csv('../input/playground-series-s5e11/test.csv')
X_full.head()


# test feature

X_full['monthly_income'] = X_full['annual_income'] / 12
X_full['debt'] = X_full['debt_to_income_ratio'] * X_full['monthly_income']

X_test['monthly_income'] = X_test['annual_income'] / 12
X_test['debt'] = X_test['debt_to_income_ratio'] * X_test['monthly_income']


# Extracting categorical data
X_cat = X_full.select_dtypes(exclude = np.number)
test_cat = X_test.select_dtypes(exclude = np.number)
X_cat.head()


# One Hot Encoding

oh = OneHotEncoder(sparse_output = False)
oh_X = pd.DataFrame(oh.fit_transform(X_cat))
oh_X.columns = oh_X.columns.astype(str)

oh_t = pd.DataFrame(oh.transform(test_cat))
oh_t.columns = oh_t.columns.astype(str)

oh_X.head()


# Extracting Numerical Features
X = X_full.select_dtypes(include = np.number)
test = X_test.select_dtypes(include = np.number)

X.drop('id', axis = 1, inplace = True)
X.head()

# Merging
X = pd.concat([X, oh_X], axis = 1)
X_test = pd.concat([test, oh_t], axis = 1)


y = X.pop('loan_paid_back')
testID = X_test.pop('id')


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, random_state = 6)
X['Cluster'] = kmeans.fit_predict(X)
X_test['Cluster'] = kmeans.predict(X_test)


# Scaling the data
scale = StandardScaler()
cols = X.columns
X = pd.DataFrame(scale.fit_transform(X))
X_test = pd.DataFrame(scale.transform(X_test))

X.columns = cols
X_test.columns = cols

X.head(2)


import seaborn as sns
sns.relplot(
    x="annual_income", y="loan_amount", hue="Cluster", data=X, height=6,
)


model = LogisticRegression()


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 6)


model.fit(X_train, y_train)


pred = model.predict(X_val)


roc_auc_score(pred, y_val)


cross_val_score(model, X, y, cv=5, scoring = 'roc_auc').mean()


model.fit(X, y)


final = model.predict_proba(X_test)[:,1]

final = pd.DataFrame({'id': testID, 'loan_paid_back' : final})

final.head()


final.to_csv('submission.csv', index = False)

