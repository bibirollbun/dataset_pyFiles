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
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, RocCurveDisplay


train_set = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
test_set = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')


train_set.shape


train_set.info() # NaN qiymatlar mavjud emas


train_set.describe()


train_set.head()


df = train_set.drop(['id', 'CustomerId', 'Surname'], axis = 1)
df.head()


print(df['Gender'].value_counts(), '\n')
print(df['Geography'].value_counts())


# Gender va Geography ustuning LabelEncoder yordamida sonli qiymatga almashtirib olyapmiz
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Geography'] = encoder.fit_transform(df['Geography'])


df.head()


df.corr().abs()
df.corr().abs().style.background_gradient(cmap = "coolwarm")


df.corrwith(df['Exited']).abs().sort_values(ascending=False)


exited_values = df['Exited'].value_counts()/len(df)

plt.figure(figsize=(5, 6))
plt.pie(exited_values, labels=['Qarzlarini qaytargan', "Qarzlarini qaytarmagan"], autopct="%1.1f%%")
plt.title("Mijozlarning bashorat qilinishi kerak bo'lgan qiymatlarining datasetdagi ulushi")
plt.show()


# CreditScore, Age, EstimatedSalary, Tenure

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

sns.histplot(data = df, x = 'CreditScore', hue = 'Exited', ax=ax[0][0])

sns.histplot(data = df, x = 'Age', hue = 'Exited', ax=ax[0][1])

sns.histplot(data = df, x = 'EstimatedSalary', hue = 'Exited', ax=ax[1][0])

sns.histplot(data = df, x = 'Tenure', hue = 'Exited', ax=ax[1][1])

plt.show()


# Exited
# Geography, Gender, HasCard, IsActiveMember

fig, ax = plt.subplots(2, 2, figsize=(10, 7))

sns.countplot(data=df, x = 'Exited', hue = 'Geography', ax = ax[0][0])
ax[0][0].legend(title = 'Geography', labels = ['France', 'Spain', 'Germany'])

sns.countplot(data=df, x = 'Exited', hue = 'Gender', ax = ax[0][1])
ax[0][1].legend(title = 'Gender', labels=['Male', 'Female'])

sns.countplot(data=df, x = 'Exited', hue = 'HasCrCard', ax = ax[1][0])
ax[1][0].legend(title = 'HasCrCard', labels=['Yes', 'No'])

sns.countplot(data=df, x = 'Exited', hue = 'IsActiveMember', ax = ax[1][1])
ax[1][1].legend(title = 'IsAciveMember', labels=['Yes', 'No'])

plt.show()


X = df.drop('Exited', axis=1)
y = df['Exited']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=32)


LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)

y_predict = LR_model.predict(X_test)
print(classification_report(y_test, y_predict))

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Confusion Matrix")

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Roc curve').plot(ax=ax[1])
ax[1].set_title("ROC Curve")

plt.show()



RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)

y_predict = RF_model.predict(X_test)
print(classification_report(y_test, y_predict))

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Confusion Matrix")

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Roc curve').plot(ax=ax[1])
ax[1].set_title("ROC Curve")

plt.show()


SVC_model = SVC()
SVC_model.fit(X_train, y_train)

y_predict = SVC_model.predict(X_test)
print(classification_report(y_test, y_predict))

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Confusion Matrix")

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Roc curve').plot(ax=ax[1])
ax[1].set_title("ROC Curve")

plt.show()


param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200]
    }
xgb_model = XGBClassifier()
grid_search = GridSearchCV(xgb_model, param_grid, cv=5)
grid_search.fit(X, y)


grid_search.cv_results_['rank_test_score']


grid_search.best_estimator_


XGB_model = XGBClassifier(max_depth = 5, n_estimators = 100, learning_rate = 0.1)
XGB_model.fit(X_train, y_train)

y_predict = XGB_model.predict(X_test)
print(classification_report(y_test, y_predict))

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Confusion Matrix")

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Roc curve').plot(ax=ax[1])
ax[1].set_title("ROC Curve")

plt.show()


tree_model = DecisionTreeClassifier()
params = {'max_depth':np.arange(30)}
tree_gs_model = GridSearchCV(estimator=tree_model, param_grid=params, cv = 5)
tree_gs_model.fit(X, y)


scores = tree_gs_model.cv_results_['rank_test_score']
scores


tree_gs_model.best_params_


tree_gs_model.best_score_


plt.figure(figsize = (15, 8))
plt.plot(params['max_depth'], scores)
plt.xticks(params['max_depth'])
plt.yticks(scores)
plt.grid(alpha = 0.5)
plt.show()


tree_model = DecisionTreeClassifier(max_depth=7)
tree_model.fit(X_train, y_train)

y_predict = tree_model.predict(X_test)
print(classification_report(y_test, y_predict))

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("Confusion Matrix")

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Roc curve').plot(ax=ax[1])
ax[1].set_title("ROC Curve")

plt.show()


df_test = test_set.drop(['id', 'CustomerId', 'Surname'], axis = 1)
df_test.head()


encoder = LabelEncoder()
df_test['Gender'] = encoder.fit_transform(df_test['Gender'])
df_test['Geography'] = encoder.fit_transform(df_test['Geography'])

df_test.head()


scaler = StandardScaler()
X_test = scaler.fit_transform(df_test)
X_test[0, :]


prediction = XGB_model.predict_proba(X_test)[:,1]
prediction


final = pd.DataFrame({'id':test_set.index, 'prediction':prediction})


final.to_csv('submission.csv', index = False)




