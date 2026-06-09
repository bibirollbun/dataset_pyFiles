# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/aviakompaniya/train_dataset.csv', index_col=0)
test = pd.read_csv('/kaggle/input/aviakompaniya/test_dataset.csv')
sample = pd.read_csv('/kaggle/input/aviakompaniya/sample_submission.csv')


train.head()


test.head()


train['Arrival Delay in Minutes'].value_counts()


train.shape


train.select_dtypes(include='object').columns


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


train['Gender'] = labelencoder.fit_transform(train['Gender'].values)
train['Customer Type'] = labelencoder.fit_transform(train['Customer Type'].values)
train['Type of Travel'] = labelencoder.fit_transform(train['Type of Travel'].values)
train['Class'] = labelencoder.fit_transform(train['Class'].values)


train.isnull().sum()


train = train.fillna(0)


train.info()


train['satisfaction'].value_counts()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = train.drop('satisfaction', axis=1)
y = train['satisfaction']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC

# LogisticRegression
lr_model = LogisticRegression(penalty='l2', solver='liblinear')
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_valid)

# DecisionTreeClassifier (DecistionTreeRegression to'g'irlandi)
tr_model = DecisionTreeClassifier()
tr_model.fit(X_train, y_train)
tr_y_pred = tr_model.predict(X_valid)

# RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_valid)

# Support Vector Machine (uzoq vaqt talab qilgani uchun sinab ko'rolmadim)
# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train, y_train)
# svm_y_pred = svm_model.predict(X_valid)


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Umumiy holda baholovchi funksiyamizni yozib olamiz
def score(y_true, y_pred, model_name):
    acc = accuracy_score(y_valid, y_pred)
    auc_score = roc_auc_score(y_valid, y_pred)
    print(model_name)
    print(f"-- AUC: {auc_score:.2f}")
    print(f"-- Accuracy: {acc:.2f}")
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True)
    plt.show()



preds = [lr_y_pred, tr_y_pred, rf_y_pred]
models_name = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier']
for pred, model_name in zip(preds, models_name):
    score(y_valid, pred, model_name)


from sklearn.metrics import classification_report

print(classification_report(y_valid, rf_y_pred))


from sklearn.metrics import roc_curve, auc

# ROC chizig'ini chizamiz
y_pred_prob = rf_model.predict_proba(X_valid)[:, 1]
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_prob)
roc_auc = auc(fpr, tpr)

# ROC curve ni chizish
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



# Yuqoridagi kabi RandomForest algoritmidan foydalanamiz
test['Gender'] = labelencoder.fit_transform(test['Gender'].values)
test['Customer Type'] = labelencoder.fit_transform(test['Customer Type'].values)
test['Type of Travel'] = labelencoder.fit_transform(test['Type of Travel'].values)
test['Class'] = labelencoder.fit_transform(test['Class'].values)


test.set_index(test.columns[0], inplace=True)
test = test.fillna(0)


test_y = rf_model.predict(test)
sample['satisfaction'] = test_y
sample.head()


sample.to_csv('aviakom_new.csv', index=False)




