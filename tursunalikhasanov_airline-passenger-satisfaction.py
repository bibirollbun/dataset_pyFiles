# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, jaccard_score, roc_curve, confusion_matrix, classification_report, auc, RocCurveDisplay, f1_score

from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


sample_sub = pd.read_csv("/kaggle/input/aviakompaniya/sample_submission.csv")
train = pd.read_csv("/kaggle/input/aviakompaniya/train_dataset.csv")
test = pd.read_csv("/kaggle/input/aviakompaniya/test_dataset.csv")
print(sample_sub.shape, train.shape, test.shape)


train.head()


train.isnull().sum()


test.isnull().sum()


train = train.dropna(subset='Arrival Delay in Minutes')
test = test.dropna(subset='Arrival Delay in Minutes')


train.info()


sns.countplot(data=train, x='Customer Type', hue='Gender')
plt.show()


sns.countplot(data=train, x='Type of Travel', hue='Class')
plt.show()


train.head(1)


print(train.Gender.unique())
print(train['Customer Type'].unique())
print(train['Type of Travel'].unique())
print(train['Class'].unique())


def prepared(df):
    
    gender = {'Female': 0, 'Male': 1}
    df['Gender'] = df['Gender'].map(gender)
    
    customer_type = {'disloyal Customer': 0, 'Loyal Customer': 1}
    df['Customer Type'] = df['Customer Type'].map(customer_type)
    
    type_travel =  {'Business travel': 0, 'Personal Travel': 1}
    df['Type of Travel'] = df['Type of Travel'].map(type_travel)
    
    type_class =  {'Eco': 0, 'Business': 1, 'Eco Plus': 2}
    df['Class'] = df['Class'].map(type_class)
    
    df.drop('id', axis=1, inplace=True)
    
    std_scaler = StandardScaler()
    df = std_scaler.fit_transform(df)
    
    return df


X = train.drop('satisfaction', axis=1)
y = train[['satisfaction']]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)


y_train.head()


X_train = prepared(X_train)


X_train


X_test = prepared(X_test)


X_test


LOG_model = LogisticRegressionCV()
LOG_model.fit(X_train, y_train.values.ravel())

predict = LOG_model.predict(X_test)

print(f"ACCURACY: {accuracy_score(y_test, predict)}")
print(f"JACCARD: {jaccard_score(y_test, predict)}")
print(f"ALL REPORTS: {classification_report(y_test, predict)}")

sns.heatmap(confusion_matrix(y_test, predict), annot=True, fmt='g')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, predict)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


KNN_model = KNeighborsClassifier(n_neighbors=9)
KNN_model.fit(X_train, y_train.values.ravel())

predict = KNN_model.predict(X_test)

print(f"ACCURACY: {accuracy_score(y_test, predict)}")
print(f"JACCARD: {jaccard_score(y_test, predict)}")
print(f"ALL REPORTS: {classification_report(y_test, predict)}")

sns.heatmap(confusion_matrix(y_test, predict), annot=True, fmt='g')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, predict)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


f1 = []
for k in range(1,25):
    knn = KNeighborsClassifier(n_neighbors=k) # k-ni qiymati
    knn.fit(X_train, y_train.values.ravel())
    y_predict = knn.predict(X_test)
    f1.append(f1_score(y_test, y_predict))

plt.figure(figsize=(10,6))
plt.plot(range(1,25),f1)
plt.xticks(range(1,25))
plt.grid()
plt.show()


SVM_model = SVC()
SVM_model.fit(X_train, y_train.values.ravel())

predict = SVM_model.predict(X_test)

print(f"ACCURACY: {accuracy_score(y_test, predict)}")
print(f"JACCARD: {jaccard_score(y_test, predict)}")
print(f"ALL REPORTS: {classification_report(y_test, predict)}")

sns.heatmap(confusion_matrix(y_test, predict), annot=True, fmt='g')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, predict)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


Tree_model = DecisionTreeClassifier(max_depth=7)
Tree_model.fit(X_train, y_train.values.ravel())

predict = Tree_model.predict(X_test)

print(f"ACCURACY: {accuracy_score(y_test, predict)}")
print(f"JACCARD: {jaccard_score(y_test, predict)}")
print(f"ALL REPORTS: {classification_report(y_test, predict)}")

sns.heatmap(confusion_matrix(y_test, predict), annot=True, fmt='g')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, predict)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()

#cols = df.drop('Drug', axis=1).columns
#classes = df['Drug'].unique()

plt.figure(figsize=(15,10))
tree.plot_tree(Tree_model, filled=True)
plt.show()


RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train.values.ravel())

predict = RF_model.predict(X_test)

print(f"ACCURACY: {accuracy_score(y_test, predict)}")
print(f"JACCARD: {jaccard_score(y_test, predict)}")
print(f"ALL REPORTS: {classification_report(y_test, predict)}")

sns.heatmap(confusion_matrix(y_test, predict), annot=True, fmt='g')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, predict)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="ROC curve")
display.plot()
plt.show()


test = prepared(test)


test


prediction = SVM_model.predict(test)


test = pd.read_csv("/kaggle/input/aviakompaniya/test_dataset.csv")
test.isnull().sum()


test = test.fillna(np.mean(test['Arrival Delay in Minutes']))


test.shape


test = prepared(test)


yhat = SVM_model.predict(test)


sample_sub['satisfaction'] = yhat


sample_sub.to_csv('submission.csv', index=False)




