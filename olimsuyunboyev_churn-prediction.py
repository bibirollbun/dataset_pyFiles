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


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


train = pd.read_csv("/kaggle/input/churn-prediction-2024/train.csv", sep=';')
test = pd.read_csv("/kaggle/input/churn-prediction-2024/test.csv", sep=';')


train.head()


train.info()


train['Sex'].value_counts()


train['Sex'] = train['Sex'].replace({'Male': 1, 'Female': 0})


train.head()


age_mean = train['Age'].mean()
age_mean


train['Age'] = train['Age'].fillna(age_mean)


train.info()


label = LabelEncoder()
train['State'] = label.fit_transform(train['State'].values)
train['Phone number'] = label.fit_transform(train['Phone number'].values)
train['Plans'] = label.fit_transform(train['Plans'].values)
train['Total day minutes'] = label.fit_transform(train['Total day minutes'].values)
train['Total day calls'] = label.fit_transform(train['Total day calls'].values)
train['Total night minutes'] = label.fit_transform(train['Total night minutes'].values)
train['Total intl minutes'] = label.fit_transform(train['Total intl minutes'].values)
train['Total charge'] = label.fit_transform(train['Total charge'].values)
train['Customer service calls'] = label.fit_transform(train['Customer service calls'].values)


X = train.drop("Churn", axis=1)
y = train['Churn']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Model yaratamiz
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Modelni baholaymiz
y_predict = log_model.predict(X_test)
log_acc = metrics.accuracy_score(y_test, y_predict)
log_acc
log_precision = precision_score(y_test, y_predict)
log_recall = recall_score(y_test, y_predict)
print(metrics.classification_report(y_test, y_predict))
print("Model aniqligi", metrics.accuracy_score(y_test, y_predict))

# Confusion matrix
conf_mat = metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(conf_mat, annot=True, fmt='g')
plt.show()

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
log_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=log_auc, estimator_name='ROC curve')
display.plot()
plt.show()


# Model yaratamiz
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Modelni baholaymiz
y_predict = svm_model.predict(X_test)
svm_acc = metrics.accuracy_score(y_test, y_predict)
svm_acc
svm_precision = precision_score(y_test, y_predict)
svm_recall = recall_score(y_test, y_predict)
print(metrics.classification_report(y_test, y_predict))
print("Model aniqligi", metrics.accuracy_score(y_test, y_predict))

# Confusion matrix
conf_mat = metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(conf_mat, annot=True, fmt='g')
plt.show()

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
svm_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=svm_auc, estimator_name='ROC curve')
display.plot()
plt.show()


# Model yaratamiz
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# Modelni baholaymiz
y_predict = tree_model.predict(X_test)
tree_acc = metrics.accuracy_score(y_test, y_predict)
tree_acc
tree_precision = precision_score(y_test, y_predict)
tree_recall = recall_score(y_test, y_predict)
print(metrics.classification_report(y_test, y_predict))
print("Model aniqligi", metrics.accuracy_score(y_test, y_predict))

# Confusion matrix
conf_mat = metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(conf_mat, annot=True, fmt='g')
plt.show()

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
tree_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=tree_auc, estimator_name='ROC curve')
display.plot()
plt.show()


# Model yaratamiz
RF_model = RandomForestClassifier(n_estimators=100)
RF_model.fit(X_train, y_train)

# Modelni baholaymiz
y_predict = RF_model.predict(X_test)
rf_acc = metrics.accuracy_score(y_test, y_predict)
rf_acc
rf_precision = precision_score(y_test, y_predict)
rf_recall = recall_score(y_test, y_predict)
print(metrics.classification_report(y_test, y_predict))
print("Model aniqligi", metrics.accuracy_score(y_test, y_predict))

# Confusion matrix
conf_mat = metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(conf_mat, annot=True, fmt='g')
plt.show()

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
rf_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=rf_auc, estimator_name='ROC curve')
display.plot()
plt.show()


# Model yaratamiz
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Modelni baholaymiz
y_predict = xgb_model.predict(X_test)
xgb_acc = metrics.accuracy_score(y_test, y_predict)
xgb_acc
xgb_precision = precision_score(y_test, y_predict)
xgb_recall = recall_score(y_test, y_predict)
print(metrics.classification_report(y_test, y_predict))
print("Model aniqligi", metrics.accuracy_score(y_test, y_predict))

# Confusion matrix
conf_mat = metrics.confusion_matrix(y_test, y_predict)
sns.heatmap(conf_mat, annot=True, fmt='g')
plt.show()

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
xgb_auc = metrics.auc(fpr, tpr)
display_roc = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=xgb_auc, estimator_name='ROC curve')
display_roc.plot()
plt.show()


models = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines',  
              'Decision Tree', 'Random Forest', 'XGBoost'],
    'Accuracy Score': [log_acc, svm_acc, tree_acc, rf_acc, xgb_acc],
    'ROC-AUC': [log_auc, svm_auc, tree_auc, rf_auc, xgb_auc],
    'Precision': [log_precision, svm_precision, tree_precision, rf_precision, xgb_precision],
    'Recall': [log_recall, svm_recall, tree_recall, rf_recall, xgb_recall]
})
models.sort_values(by='Accuracy Score', ascending=False)







