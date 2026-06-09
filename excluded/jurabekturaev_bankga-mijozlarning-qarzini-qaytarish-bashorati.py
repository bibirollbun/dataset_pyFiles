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


sub_sample = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/sample_submission.csv")
df_test = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv", index_col=0)
df_train = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv", index_col=0)


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from xgboost import XGBClassifier


df_train.shape


df_train.info()


df_train.describe()


df_train.corrwith(df_train['Exited'], numeric_only=True).abs().sort_values(ascending=False)


rate = df_train['Exited'].value_counts()/len(df_train)*100
rate


plt.pie(rate, labels=['Berilmaydi','Beriladi'])
plt.show()


df_train.drop(df_train[['Surname','CustomerId']], axis=1, inplace=True)


df_train.head()


df_train['Geography'].value_counts()


df_train['Geography'] = df_train['Geography'].map({'France': 0, 'Spain': 1, 'Germany': 2})
df_train['Gender'] = df_train['Gender'].map({'Female': 0, 'Male': 1})


corr = df_train.corr().abs()
corr.style.background_gradient(cmap='coolwarm')


X = df_train.drop('Exited', axis=1)
Y = df_train['Exited']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, stratify=Y, test_size=0.2, random_state=42)


xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)

# Modelni baholash
y_pred = xgb_model.predict(x_test)
print(metrics.classification_report(y_test, y_pred))
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)*100:.1f}%")

# confusion matrix
conf_matx = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matx, annot=True, fmt='g')
plt.show()

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC curve')
display.plot()
plt.show()


df_test.head()


df_test.drop(df_test[['CustomerId','Surname']], axis=1, inplace=True)


df_test['Geography'] = df_test['Geography'].map({'France': 0, 'Spain': 1, 'Germany': 2})
df_test['Gender'] = df_test['Gender'].map({'Female': 0, 'Male': 1})


df_test.head()


df_test_scaled = scaler.transform(df_test)


df_test_proba = xgb_model.predict_proba(df_test_scaled)[:, 1]


submission = pd.DataFrame({'id': df_test.index, 'Exited': df_test_proba})
submission.head()


submission.to_csv('Submission.csv', index=False)




