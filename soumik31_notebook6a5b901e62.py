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
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train_df.head()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_df.head()


sample_df = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sample_df.head()


train_df.nunique()


train_df.isna().sum()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_df.drop(['id','rainfall'],inplace=False, axis='columns'),train_df.rainfall,test_size=0.2,random_state=10)


import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# model_params = {
#     # 'svm': {
#     #     'model': SVC(gamma='auto'),
#     #     'params': {
#     #         'C': [0.1, 1, 10, 100],
#     #         'kernel': ['rbf', 'linear']
#     #     }
#     # },
#     # 'random_forest': {
#     #     'model': RandomForestClassifier(),
#     #     'params': {
#     #         'n_estimators': [10, 50, 100, 200],
#     #         'max_depth': [None, 10, 20, 30],
#     #         'min_samples_split': [2, 5, 10],
#     #         'min_samples_leaf': [1, 2, 4]
#     #     }
#     # },
#     # 'logistic_regression': {
#     #     'model': LogisticRegression(solver='liblinear'), #added solver to avoid warnings
#     #     'params': {
#     #         'C': [0.1, 1, 5, 10, 100],
#     #         'penalty': ['l1', 'l2']
#     #     }
#     # },
#     # 'gaussian_naive_bayes': {
#     #     'model': GaussianNB(),
#     #     'params': {}  # Gaussian Naive Bayes has few tunable parameters
#     # },
#     # 'knn': {
#     #     'model': KNeighborsClassifier(),
#     #     'params': {
#     #         'n_neighbors': [3, 5, 7, 11],
#     #         'weights': ['uniform', 'distance'],
#     #         'p': [1, 2]  # 1: Manhattan, 2: Euclidean
#     #     }
#     # },
#     # 'decision_tree': {
#     #     'model': DecisionTreeClassifier(),
#     #     'params': {
#     #         'max_depth': [None, 10, 20, 30],
#     #         'min_samples_split': [2, 5, 10],
#     #         'min_samples_leaf': [1, 2, 4],
#     #         'criterion': ['gini', 'entropy']
#     #     }
#     # },
#     # 'xgboost': {
#     #     'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'), # added eval_metric and use_label_encoder to avoid warnings.
#     #     'params': {
#     #         'n_estimators': [50, 100, 200],
#     #         'learning_rate': [0.01, 0.1, 0.2, 0.3],
#     #         'max_depth': [3, 4, 5, 6],
#     #         'subsample': [0.7, 0.8, 0.9, 1],
#     #         'colsample_bytree': [0.7, 0.8, 0.9, 1]
#     #     }
#     # }
#     'xgboost': {
#         'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'), # added eval_metric and use_label_encoder to avoid warnings.
#         'params': {
#             'colsample_bytree': [0.6],
#             'learning_rate': [0.1],
#             'max_depth': [3],
#             'n_estimators': [35, 40, 45],
#             'subsample': [0.87],
#         }
#     }
# }

# scores=[]
# for model_name, mp in model_params.items():
#     clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
#     clf.fit(X_train,y_train)
#     scores.append({
#         'model': model_name,
#         'best_score': clf.best_score_,
#         'best_params': clf.best_params_
#     })
# scores


test_df.head()


X = train_df.drop(['id','rainfall'],inplace=False, axis='columns')
y = train_df.rainfall
model = XGBClassifier(colsample_bytree=0.6, learning_rate=0.1, max_depth=3, n_estimators=40, subsample=0.87)
model.fit(X,y)


results = model.predict(test_df.drop(['id'],inplace=False, axis='columns'))


results


final_df = pd.DataFrame({'id': test_df.id.to_numpy(), 'rainfall': results})


final_df.to_csv('submission.csv',index=False)


final_df.head()

