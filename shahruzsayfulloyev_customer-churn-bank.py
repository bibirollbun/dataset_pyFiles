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
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import  confusion_matrix, classification_report, accuracy_score
%matplotlib inline
df = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv')
df_test = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/test.csv')
sample_submission = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/sample_submission.csv')
sample_submission


test_ids = df_test['id']


encoder = OneHotEncoder( sparse_output=False)
df_encoded = pd.DataFrame(encoder.fit_transform(df[['Gender', 'Geography']]), columns=encoder.get_feature_names_out(['Gender', 'Geography']))
df = pd.concat([df, df_encoded], axis=1)
df.drop(['Gender', 'Geography', 'id', 'CustomerId','Surname','Geography_Spain', 'Gender_Female'], axis=1, inplace=True)
df


df['TenureByAge'] = df['Tenure']/(df['Age']+1)
df['IsActiveMemberByAge'] = df['IsActiveMember']/(df['Age']+1)
df['CreditScoreByAge'] = df['CreditScore']/(df['Age']+1)
df['CreditScoreByNumOfProducts'] = df['CreditScore']/(df['NumOfProducts']+1)
df['CreditScoreByBalance'] = df['CreditScore']/(df['Balance']+1)
df['AgeByHasCrCard'] = df['Age']/(df['HasCrCard']+1)
df['AgeByNumOfProducts'] = df['Age']/(df['NumOfProducts']+1)
df['AgeByIsActiveMember'] = df['Age']/(df['IsActiveMember']+1)
df['AgeByCreditScore'] = df['Age']/(df['CreditScore']+1)
df['AgeByHasCrCard'] = df['Age']/(df['HasCrCard']+1)


df


df.describe()


df.corrwith(df['Exited'], numeric_only=True).abs().sort_values(ascending=False)


X = df.drop('Exited', axis=1)
y = df['Exited']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


LR_model = LogisticRegression(class_weight='balanced', random_state=42)
LR_model.fit(X_train_scaled, y_train)
LR_pred = LR_model.predict(X_test_scaled)

print(metrics.classification_report(y_test,LR_pred))
print("Model-aniqligi:", metrics.accuracy_score(y_test, LR_pred))

conf_mat = metrics.confusion_matrix(y_test, LR_pred)
sns.heatmap(conf_mat, annot = True, fmt='g')
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(y_test, LR_pred)
roc_auc = metrics.auc(fpr, tpr)
lr_roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
lr_roc_display.plot()
plt.show()


svm_model = SVC(probability=True, class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)

print(metrics.classification_report(y_test,svm_pred))
print("Model-aniqligi:", metrics.accuracy_score(y_test, svm_pred))

conf_mat = metrics.confusion_matrix(y_test, svm_pred)
print(conf_mat)
sns.heatmap(conf_mat, annot = True, fmt='g')
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(y_test, svm_pred)
roc_auc = metrics.auc(fpr, tpr)
svm_roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
svm_roc_display.plot()
plt.show()


tree_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
tree_model.fit(X_train_scaled, y_train)
tree_pred = tree_model.predict(X_test_scaled)

print(metrics.classification_report(y_test,tree_pred))
print("Model-aniqligi:", metrics.accuracy_score(y_test, tree_pred))

conf_mat = metrics.confusion_matrix(y_test, tree_pred)
sns.heatmap(conf_mat, annot = True, fmt='g')
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(y_test, tree_pred)
roc_auc = metrics.auc(fpr, tpr)
tree_roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
tree_roc_display.plot()
plt.show()


RF_model = RandomForestClassifier(class_weight='balanced', random_state=42)
RF_model.fit(X_train_scaled, y_train)
RF_pred = RF_model.predict(X_test_scaled)

print(metrics.classification_report(y_test,RF_pred))
print("Model-aniqligi:", metrics.accuracy_score(y_test, RF_pred))

conf_mat = metrics.confusion_matrix(y_test, RF_pred)
sns.heatmap(conf_mat, annot = True, fmt='g')
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(y_test, RF_pred)
roc_auc = metrics.auc(fpr, tpr)
rf_roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
rf_roc_display.plot()
plt.show()


xgb_model = XGBClassifier(class_weight='balanced', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

xgb_pred = xgb_model.predict(X_test_scaled)

print(metrics.classification_report(y_test,xgb_pred))
print("Model-aniqligi:", metrics.accuracy_score(y_test, xgb_pred))

conf_mat = metrics.confusion_matrix(y_test, xgb_pred)
print(conf_mat)
sns.heatmap(conf_mat, annot = True, fmt='g')
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(y_test, xgb_pred)
roc_auc = metrics.auc(fpr, tpr)
xgb_roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
xgb_roc_display.plot()
plt.show()


df_test['TenureByAge'] = df_test['Tenure']/(df_test['Age']+1)
df_test['IsActiveMemberByAge'] = df_test['IsActiveMember']/(df_test['Age']+1)
df_test['CreditScoreByAge'] = df_test['CreditScore']/(df_test['Age']+1)
df_test['CreditScoreByNumOfProducts'] = df_test['CreditScore']/(df_test['NumOfProducts']+1)
df_test['CreditScoreByBalance'] = df_test['CreditScore']/(df_test['Balance']+1)
df_test['AgeByHasCrCard'] = df_test['Age']/(df_test['HasCrCard']+1)
df_test['AgeByNumOfProducts'] = df_test['Age']/(df_test['NumOfProducts']+1)
df_test['AgeByIsActiveMember'] = df_test['Age']/(df_test['IsActiveMember']+1)
df_test['AgeByCreditScore'] = df_test['Age']/(df_test['CreditScore']+1)
df_test['AgeByHasCrCard'] = df_test['Age']/(df_test['HasCrCard']+1)
df_test


encoder = OneHotEncoder( sparse_output=False)
df_test_encoded = pd.DataFrame(encoder.fit_transform(df_test[['Gender', 'Geography']]), columns=encoder.get_feature_names_out(['Gender', 'Geography']))
df_test = pd.concat([df_test, df_test_encoded], axis=1)
df_test.drop(['Gender', 'Geography', 'id', 'CustomerId','Surname','Geography_Spain', 'Gender_Female'], axis=1, inplace=True)
df_test


df_test = df_test.reindex(columns=X_train.columns, fill_value=0)

df_test_scaled = scaler.transform(df_test)

predict_test = svm_model.predict_proba(df_test_scaled)[:,1]


submit_test = pd.DataFrame({'id': test_ids, 'Exited': predict_test})
submit_test


submit_test.to_csv('submission.csv', index=False)

