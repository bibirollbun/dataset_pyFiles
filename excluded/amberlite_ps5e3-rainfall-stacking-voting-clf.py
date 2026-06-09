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


# import libraries

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score, recall_score


# download data, get features and targets

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train_feature = train.drop(columns=['id', 'day', 'rainfall'])
y = train['rainfall'].values


# standardization of features

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_feature)
train_scaled.shape


# train-test slit of fetures and targets

X_train, X_test, y_train, y_test = train_test_split(train_feature, y, test_size=0.2, stratify=y, random_state=0)


# fit base models

models = [LogisticRegression(),
          KNeighborsClassifier(),
          GaussianNB(),
          SVC(probability=True),
          DecisionTreeClassifier(random_state=0),
          RandomForestClassifier(random_state=0),
          GradientBoostingClassifier(random_state=0),
          XGBClassifier(),
          LGBMClassifier(verbose=0),
          CatBoostClassifier(verbose=False)]

accuracy = []
precision = []
recall = []
f1 = []
roc_auc = []

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))
    roc_auc.append(roc_auc_score(y_test, y_pred_prob))
results = pd.DataFrame({
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'f1':f1,
    'roc_auc':roc_auc
}, index=['LogisticRegression',
          'KNeighborsClassifier',
          'GaussianNB',
          'SVC',
          'DecisionTreeClassifier',
          'RandomForestClassifier',
          'GradientBoostingClassifier',
          'XGBClassifier',
          'LGBMClassifier',
          'CatBoostClassifier'])


# see results of base models

results


# choose models with the best metrics

results.loc[['LogisticRegression', 'SVC', 'GradientBoostingClassifier', 'CatBoostClassifier']]


# fit LogisticRegression model

log_r = LogisticRegression()
log_r.fit(X_train, y_train)
y_pred = log_r.predict(X_test)
y_pred_prob = log_r.predict_proba(X_test)[:, 1]

print(f'accuracy {accuracy_score(y_test, y_pred):.3f}')
print(f'f1 {f1_score(y_test, y_pred):.3f}')
print(f'precision {precision_score(y_test, y_pred):.3f}')
print(f'recall {recall_score(y_test, y_pred):.3f}')
print(f'roc_auc {roc_auc_score(y_test, y_pred_prob):.3f}')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predict targets')
plt.ylabel('True targets')
plt.title('Confusion matrix')
plt.show()


# optimization of svm model

svm = SVC(kernel='rbf', probability=True)
param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = [{'C': param_range,
               'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100, 1000],
               'kernel': ['rbf']}]
gs = GridSearchCV(estimator=svm,
                  param_grid=param_grid,
                  scoring='roc_auc',
                  cv=5,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(f'best precision score = {gs.best_score_:.3f}')
print(f'best params: {gs.best_params_}')


# fit the optimized svm model

svm = SVC(probability=True, **{'C': 1000, 'gamma': 'scale', 'kernel': 'rbf'})
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
y_pred_prob = svm.predict_proba(X_test)[:, 1]

print(f'accuracy {accuracy_score(y_test, y_pred):.3f}')
print(f'f1 {f1_score(y_test, y_pred):.3f}')
print(f'precision {precision_score(y_test, y_pred):.3f}')
print(f'recall {recall_score(y_test, y_pred):.3f}')
print(f'roc_auc {roc_auc_score(y_test, y_pred_prob):.3f}')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predict targets')
plt.ylabel('True targets')
plt.title('Confusion matrix')
plt.show()


# fit the GradientBoostingClassifier model

gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
y_pred_prob = gbc.predict_proba(X_test)[:, 1]

print(f'accuracy {accuracy_score(y_test, y_pred):.3f}')
print(f'f1 {f1_score(y_test, y_pred):.3f}')
print(f'precision {precision_score(y_test, y_pred):.3f}')
print(f'recall {recall_score(y_test, y_pred):.3f}')
print(f'roc_auc {roc_auc_score(y_test, y_pred_prob):.3f}')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predict targets')
plt.ylabel('True targets')
plt.title('Confusion matrix')
plt.show()


cb = CatBoostClassifier(verbose=False)
cb.fit(X_train, y_train)
y_pred = cb.predict(X_test)
y_pred_prob = cb.predict_proba(X_test)[:, 1]

print(f'accuracy {accuracy_score(y_test, y_pred):.3f}')
print(f'f1 {f1_score(y_test, y_pred):.3f}')
print(f'precision {precision_score(y_test, y_pred):.3f}')
print(f'recall {recall_score(y_test, y_pred):.3f}')
print(f'roc_auc {roc_auc_score(y_test, y_pred_prob):.3f}')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predict targets')
plt.ylabel('True targets')
plt.title('Confusion matrix')
plt.show()


# make the VotingClassifier on best models

voting_model = VotingClassifier(
    estimators=[
        ('log_r', log_r),
        ('gbc', gbc),
        ('svm', svm),
        ('cb', cb)
    ], voting='soft', verbose=False
)
voting_model.fit(X_train, y_train)
y_pred = voting_model.predict(X_test)
y_pred_prob = voting_model.predict_proba(X_test)[:, 1]
print(f'accuracy {accuracy_score(y_test, y_pred):.3f}')
print(f'f1 {f1_score(y_test, y_pred):.3f}')
print(f'precision {precision_score(y_test, y_pred):.3f}')
print(f'recall {recall_score(y_test, y_pred):.3f}')
print(f'roc_auc {roc_auc_score(y_test, y_pred_prob):.3f}')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predict targets')
plt.ylabel('True targets')
plt.title('Confusion matrix')
plt.show()


# make the StackingClassifier on the SVM and GradientBoostingClassifier models

base_models = [
    ('gbc', gbc),
    ('svm', svm),
    ('cb', cb)
]

meta_model = LogisticRegression()

stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
y_pred_prob = stacking_model.predict_proba(X_test)[:, 1]
print(f'accuracy {accuracy_score(y_test, y_pred):.3f}')
print(f'f1 {f1_score(y_test, y_pred):.3f}')
print(f'precision {precision_score(y_test, y_pred):.3f}')
print(f'recall {recall_score(y_test, y_pred):.3f}')
print(f'roc_auc {roc_auc_score(y_test, y_pred_prob):.3f}')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predict targets')
plt.ylabel('True targets')
plt.title('Confusion matrix')
plt.show()


# fit ensemble models on all train data
stacking_model.fit(train_scaled, y)
voting_model.fit(train_scaled, y)


# download test data and standardization

test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


test.head().T


test.isnull().sum()


test.fillna(test.mean(), inplace=True)


test_feature = test.drop(columns=['id', 'day'])

scaler = StandardScaler()
test_scaled = scaler.fit_transform(test_feature)
test_scaled.shape


# make predictions

predict_1 = stacking_model.predict_proba(test_scaled)[:, 1]
predict_2 = voting_model.predict_proba(test_scaled)[:, 1]


# make submission for stacking_model

sample_submission_1 = pd.DataFrame({
    'id':test['id'].values,
    'rainfall':predict_1
})
sample_submission_1.to_csv('sample_submission.csv', index=False)




