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


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


df.head()


df.info()


df.columns


df.isnull().sum()


df.describe()


df.dtypes


df.index


df.columns


df.tail()


df['y'].value_counts()


df2 = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


df2.head()


df2.columns


df2.isnull().sum()


df.shape


df2.shape


df.describe()


df2.describe()


df2.columns


df2['poutcome'].value_counts()


X = df.drop(['id','y'], axis = 1)
y = df['y']
X_val = df2.drop(['id'],axis =1)


df.dtypes


cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']


from sklearn.preprocessing import LabelEncoder,StandardScaler
l= {}
for i in cat_cols:
    le = LabelEncoder()
    le1 = pd.concat([X[i].astype(str),X_val[i].astype(str)]).unique()
    le.fit(le1)
    X[i] = le.transform(X[i].astype(str))
    X_val[i] = le.transform(X_val[i].astype(str))
    l[i] = le


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


X_train.shape


X_test.shape


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train,y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


from sklearn.metrics import roc_auc_score
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)


print("AUC-ROC Score",auc_score)


test_pred_proba = model.predict_proba(X_val)[:,1]
test_pred_proba = np.clip(test_pred_proba,0,1)


submission = pd.DataFrame({
    'id':df2['id'],
    'y': test_pred_proba
})
submission.to_csv('submission.csv',index=False)


submission.head()


submission.shape




