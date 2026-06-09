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


!pip install catboost


from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/kaggle/input/binaryclassificationwithabankchurndataset/train.csv', index_col=0)
df.head()


df.shape


df.info()


df.describe()


df.corr(numeric_only=True)


df = df.drop(columns=['CustomerId', 'Surname'])
df.head()


clss_dis = df['Exited'].value_counts()/len(df)*100

plt.figure(figsize=(7,7))
plt.pie(clss_dis, labels=clss_dis.index, autopct='%1.2f%%')
plt.show()

sns.countplot(x='Exited', data=df)
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(20, 10))
sns.histplot(x='CreditScore', bins=30, data=df, hue='Exited', ax=ax[0, 0])
sns.histplot(x='Balance', data=df, bins=30, hue='Exited', ax=ax[0, 1])
sns.histplot(x='EstimatedSalary', bins=60, data=df, hue='Exited', ax=ax[1, 0])

sns.countplot(x='Tenure', data=df, hue='Exited', ax=ax[1, 1])

plt.tight_layout()
plt.show()


sns.histplot(x='Age', bins=50, data=df, hue='Exited')
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.countplot(x='Gender', hue='Exited', data=df, ax=ax[0])
sns.countplot(x='Geography', hue='Exited', data=df, ax=ax[1])
plt.show()


categorical_columns = ['Gender', 'Geography'] 
numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'] 
target = 'Exited' 


X = df[categorical_columns + numerical_columns]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = CatBoostClassifier(learning_rate=0.05, l2_leaf_reg=3, cat_features=categorical_columns, early_stopping_rounds=50, verbose=100, random_seed=42)


model.fit(X_train, y_train,eval_set=(X_test, y_test),use_best_model=True)


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


print("AUC", roc_auc_score(y_test, y_pred_proba))
print("Aniqlik (Accuracy):", accuracy_score(y_test, y_pred))
print("\nBaholash hisoboti:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()


feature_importance = pd.DataFrame({'Feature': X.columns,'Importance': model.get_feature_importance()}).sort_values(by='Importance', ascending=False)
print("\nXususiyatlarning ahamiyati:\n", feature_importance)


df1 = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv")
df1.head()


t_ids = df1['id']
t_ids.head()


df1 = df1.drop(columns=['id', 'CustomerId', 'Surname'])
df1.head()


categorical_columns = ['Gender', 'Geography'] 
numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']


X = df1[categorical_columns + numerical_columns]
y_prediction = model.predict(X)  
y_prediction_proba = model.predict_proba(X)[:, 1]
y_prediction_proba


submission = pd.DataFrame({'id': t_ids, 'Exited': y_prediction_proba})

# CSV faylga saqlash
submission.to_csv('tayyor.csv', index=False)
print("Fayl tayyor: tayyor.csv")

