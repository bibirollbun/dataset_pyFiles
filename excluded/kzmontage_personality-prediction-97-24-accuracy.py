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


import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import statsmodels.api as sma
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train.head()


null = {'Total Values' : train.shape[0],
        'Total Unique Value': train.nunique(),
        'Missing Values': train.isnull().sum(),
        'Missing Values %': np.round((train.isnull().sum()/train.shape[0])*100,2),
        'Dtype': train.dtypes
}


pd.DataFrame(null)


# Replacing Missing Values with mode of the column as all the columns are categorical
for col in train.columns:
    if train[col].isnull().sum() > 0:
        train[col].fillna(train[col].mode()[0], inplace=True)


for col in test.columns:
    if test[col].isnull().sum() > 0:
        test[col].fillna(test[col].mode()[0], inplace=True)



for col in train.columns:
    if col != 'id':
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=train)
        plt.title(f'{col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()


for col in train.columns:
    if col not in ['id', 'Personality']:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, hue='Personality', data=train)
        plt.title(f'{col} vs Personality')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()


train['Stage_fear'] = train['Stage_fear'].map({'Yes':1, 'No':0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes':1, 'No':0})
train['Personality'] = train['Personality'].map({'Extrovert':1, 'Introvert':0})


plt.figure(figsize=(12,8))
sns.heatmap(train.corr(), annot=True,cmap='YlGnBu')
plt.show()


test['Stage_fear'] = test['Stage_fear'].map({'Yes':1, 'No':0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes':1, 'No':0})


x= train.iloc[:,1:8]
y =train.iloc[:,-1]


x_train,x_val,y_train,y_val=train_test_split(x,y,stratify=y,test_size=0.2,random_state=1)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, recall_score, precision_score, classification_report


lr = LogisticRegression()


lr.fit(x_train,y_train)


lr.predict(x_val)


print(classification_report(y_val, lr.predict(x_val)))


dt = DecisionTreeClassifier()


dt.fit(x_train,y_train)


print(classification_report(y_val, dt.predict(x_val)))


rf=RandomForestClassifier()


rf.fit(x_train,y_train)


print(classification_report(y_val, rf.predict(x_val)))


kn = KNeighborsClassifier()


kn.fit(x_train,y_train)


print(classification_report(y_val, kn.predict(x_val)))


xgb = XGBClassifier()


xgb.fit(x_train,y_train)


print(classification_report(y_val, xgb.predict(x_val)))


lg = LGBMClassifier()


lg.fit(x_train,y_train)


print(classification_report(y_val, lg.predict(x_val)))


ct = CatBoostClassifier(learning_rate=0.05)


ct.fit(x_train,y_train)


print(classification_report(y_val, ct.predict(x_val)))


# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', ct),
        ('lgbm', lg),
        ('lr', lr),
        ('rf',rf),
        ('knn',kn)
    ],
    voting='soft'
)


ensemble.fit(x_train,y_train)


val_probs = ensemble.predict_proba(x_val)[:, 1]
best_threshold = 0.5
best_acc = 0

# for threshold in np.arange(0.4, 0.6, 0.01):
#     preds = (val_probs >= threshold).astype(int)


for threshold in np.arange(0.4, 0.6, 0.01):
    preds = (val_probs >= threshold).astype(int)


print(classification_report(y_val, preds))


accuracy_score(y_val, preds)


val_probs


test_probs = ensemble.predict_proba(test.iloc[:,1:])[:, 1]


test_preds = (test_probs >= best_threshold).astype(int)


len(test_preds)


df_submission = pd.DataFrame(test.iloc[:,0])


df_submission.insert(1, 'Personality', test_preds)


df_submission['Personality'] = df_submission['Personality'].map({1:'Extrovert', 0:'Introvert'})


df_submission.to_csv('submission.csv', index=False)




