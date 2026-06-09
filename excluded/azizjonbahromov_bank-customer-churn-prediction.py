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


# Datasetni yuklab olish
train = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv", index_col="id")
submit = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/sample_submission.csv")

train.head()


sum(train.duplicated()) #Train datasetdagi Duplicated qiymatlarni tekshirish


sum(test.duplicated()) #Test datasetdagi Duplicated qiymatlarni tekshirish


train.info()


test.info()


test.head()


train.describe()


train.drop(['Surname'], axis=1, inplace=True)
train


train['Geography'].value_counts() #Geography ustunini qiymatlarini tekshirish


# Datasetning 'Geography ustuni qiymatlarini categorydan songa o'tkazish
train['Geography'] = train['Geography'].map({'France':0, 'Spain':1, 'Germany':2})
train


train['Gender'].value_counts() #Gender ustunini qiymatlarini tekshirish


# Datasetning 'Geography ustuni qiymatlarini categorydan songa o'tkazish
train['Gender'] = train['Gender'].map({'Male':0, 'Female':1})
train


train.info()


train.corrwith(train['Exited']).sort_values(ascending=False) #Exited ustuni bilan boshqa ustunlar korrelatsiyasini tekshirish


# Korrelatsiyasi past bo'lgan ustunlarni tashlab yuboramiz
train.drop(['EstimatedSalary', 'CustomerId', 'HasCrCard', 'Tenure', 'CreditScore'], axis=1, inplace=True)
train


# Datasetning 'Geography ustuni qiymatlarini categorydan songa o'tkazish
test['Gender'] = test['Gender'].map({'Male':0, 'Female':1})
test


test['Geography'] = test['Geography'].map({'France':0, 'Spain':1, 'Germany':2})
test


test.drop(['Surname', 'EstimatedSalary', 'CustomerId', 'HasCrCard', 'Tenure', 'CreditScore'], axis=1, inplace=True)
test


train


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train = pd.DataFrame(scaler.fit_transform(train), columns = train.columns)
test = pd.DataFrame(scaler.fit_transform(test), columns = test.columns)


train


test


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score


X_train = train.drop(['Exited'], axis=1).copy()
y_train = train['Exited'].copy()
X_test = test


X_train


y_train


import matplotlib.pyplot as plt
acc = []
for i in range(1,25):
    model_KNN = KNeighborsClassifier(n_neighbors=i)
    model_KNN.fit(X_train, y_train)
    y_pred = model_KNN.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    acc.append(accuracy)
plt.plot(acc)


model_KNN = KNeighborsClassifier(n_neighbors=1)
model_KNN.fit(X_train, y_train)
y_pred = model_KNN.predict(X_train)

print(f"Model: {model} \n")
print(f"Accuracy: {accuracy_score(y_train, y_pred)} \n")
print(f"Classification Report: {classification_report(y_train, y_pred)}")


acc = []
for i in range(1,7):
    model_RF = RandomForestClassifier(n_estimators=100, max_depth=i, random_state=42)
    model_RF.fit(X_train, y_train)
    y_pred = model_RF.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    acc.append(accuracy)
plt.plot(acc)


model_RF = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_RF.fit(X_train, y_train)
y_pred = model_RF.predict(X_train)

print(f"Model: {model} \n")
print(f"Accuracy: {accuracy_score(y_train, y_pred)} \n")
print(f"Classification Report: {classification_report(y_train, y_pred)}")


acc = []
for i in range(1,7):
    model_GB = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=i, random_state=42)
    model_GB.fit(X_train, y_train)
    y_pred = model_GB.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    acc.append(accuracy)
plt.plot(acc)


model_GB = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=42)
model_GB.fit(X_train, y_train)
y_pred = model_GB.predict(X_train)

print(f"Model: {model} \n")
print(f"Accuracy: {accuracy_score(y_train, y_pred)} \n")
print(f"Classification Report: {classification_report(y_train, y_pred)}")


model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
y_pred = model_LR.predict(X_train)

print(f"Model: {model} \n")
print(f"Accuracy: {accuracy_score(y_train, y_pred)} \n")
print(f"Classification Report: {classification_report(y_train, y_pred)}")


model_GB.fit(X_train, y_train)

#train data uchun test
y_pred = model_GB.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"GradientBoostingClassifier model accuracy: {accuracy:.4f}")


#Train uchun ROC AUC Score
y_proba_train = model_GB.predict_proba(X_train)[:, 1]
roc_auc = roc_auc_score(y_train, y_proba_train)
print(f"ROC AUC Score: {roc_auc:.4f}")


#test data uchun test
y_stack_pred = model_GB.predict_proba(X_test)[:, 1]


#kaggle uchun csv ga aylantirildi
y_test = pd.DataFrame(y_stack_pred)
# ans.to_csv('answers.csv')
y_test.columns = ['Exited']
y_test['id'] = np.arange(15000, 25000 )
y_test = y_test[['id', 'Exited']]

#kaggle ushun csv. Mana shu kod bilan Kaggle ~93% aniqlikga erishildi
y_test.to_csv('predict_4.csv', index=False)
y_test




