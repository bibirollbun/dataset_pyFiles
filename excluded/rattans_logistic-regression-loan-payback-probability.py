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


X_full.nunique()


# Extracting categorical data
X_cat = X_full.drop(['id', 'annual_income', 'loan_amount', 'loan_paid_back', 'interest_rate'], axis = 1)
test_cat = X_test.drop(['id', 'annual_income', 'loan_amount', 'interest_rate'], axis = 1)
X_cat.head()


# Extracting Numerical Features
X = X_full.select_dtypes(include = np.number)
test = X_test.select_dtypes(include = np.number)

X.drop('id', axis = 1, inplace = True)
X.head()


y = X.pop('loan_paid_back')
testID = test.pop('id')


cols = list(test.columns)
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        X[f'{cols[i]}/{cols[j]}'] = X[cols[i]] / X[cols[j]]
        X[f'{cols[i]}*{cols[j]}'] = X[cols[i]] * X[cols[j]]
        X[f'{cols[i]}+{cols[j]}'] = X[cols[i]] + X[cols[j]]
        X[f'{cols[i]}-{cols[j]}'] = X[cols[i]] - X[cols[j]]

        X[f'{cols[j]}/{cols[i]}'] = X[cols[j]] / X[cols[i]]

        test[f'{cols[i]}/{cols[j]}'] = test[cols[i]] / test[cols[j]]
        test[f'{cols[i]}*{cols[j]}'] = test[cols[i]] * test[cols[j]]
        test[f'{cols[i]}+{cols[j]}'] = test[cols[i]] + test[cols[j]]
        test[f'{cols[i]}-{cols[j]}'] = test[cols[i]] - test[cols[j]]

        test[f'{cols[j]}/{cols[i]}'] = test[cols[j]] / test[cols[i]]

    X[f'log{cols[i]}'] = X[cols[i]].apply(np.log)
    X[f'1/{cols[i]}'] = 1 / X[cols[i]]
    test[f'log{cols[i]}'] = test[cols[i]].apply(np.log)
    test[f'1/{cols[i]}'] = 1 / test[cols[i]]

X.head()    


# One Hot Encoding

oh = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
oh_X = pd.DataFrame(oh.fit_transform(X_cat))
oh_X.columns = oh_X.columns.astype(str)

oh_t = pd.DataFrame(oh.transform(test_cat))
oh_t.columns = oh_t.columns.astype(str)

oh_X.head()


X = X.join(oh_X)
test = test.join(oh_t)


# Scaling the data
scale = StandardScaler()
cols = X.columns
X = pd.DataFrame(scale.fit_transform(X))
test = pd.DataFrame(scale.transform(test))

X.columns = cols
test.columns = cols

X.head(2)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, random_state = 6)
X['Cluster'] = kmeans.fit_predict(X)
test['Cluster'] = kmeans.predict(test)


"""import seaborn as sns
sns.relplot(
    x="annual_income", y="loan_amount", hue="Cluster", data=X, height=6,
)"""


model = LogisticRegression(C=1e-3, max_iter = 1000)


#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 6)


#model.fit(X_train, y_train)


#pred = model.predict(X_val)


#roc_auc_score(pred, y_val)


cross_val_score(model, X, y, cv=5, scoring = 'roc_auc').mean()


model.fit(X, y)


final = model.predict_proba(test)[:,1]

final = pd.DataFrame({'id': testID, 'loan_paid_back' : final})

final.head()


final.to_csv('submission.csv', index = False)

