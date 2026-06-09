import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

train.head()


train.info()


train.shape


train.describe()


train.isnull().sum()


train = train.drop(columns='id', axis=1)
train['marital'].value_counts()


plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
sns.histplot(train['age'], kde=True, bins=20)
plt.title('Distribution of age')

plt.subplot(1, 2, 2)
sns.histplot(train['balance'], kde=True)
plt.ylim(0, 20000)
plt.title('Distribution of balance')

plt.tight_layout()
plt.show()


cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
plt.figure(figsize=(18,12))
for i, c in enumerate(cols):
    plt.subplot(3, 3, i+1)
    sns.countplot(x=c, data=train)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


sns.countplot(x='y', data=train)


plt.figure(figsize=(6,4))
sns.barplot(x=train['job'], y=train['y'])
plt.xticks(rotation=75)


plt.figure(figsize=(6,4))
sns.histplot(x=train['duration'], kde=True)
plt.xlim(0, 2000)
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x='y', y='duration', data=train)
plt.show()


def feature_eng(df):
    df = df.replace({'no':0, 'yes':1})
    df['marital'] = df['marital'].replace({'married':0, 'single':1, 'divorced':2})
    df['education'] = df['education'].replace({'primary':0, 'secondary':1, 'tertiary':2, 'unknown':np.nan})
    df['education'] = df['education'].fillna(round(df['education'].mean()))
    df['contact'] = df['contact'].replace({'cellular':0, 'telephone':1, 'unknown':np.nan})
    df['contact'] = df['contact'].fillna(round(df['contact'].mean()))
    df['month'] = df['month'].replace({'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12})
    df['contacted_before'] = (df['pdays'] != -1).astype(int)
    df = df.drop(columns='poutcome', axis=1)
    df['job'] = df['job'].replace({'management':0, 'blue-collar':1, 'technician':2, 'admin.':3, 'services':4, 'retired':5, 'self-employed':6,
                                  'entrepreneur':7, 'unemployed':8, 'housemaid':9, 'student':10, 'unknown':np.nan})
    df['job'] = df['job'].fillna(round(df['job'].mean()))
    return df

train = feature_eng(train)
test = feature_eng(test)


train.head()


train = train.drop(columns='pdays', axis=1)
test = test.drop(columns = 'pdays', axis=1)


train.describe()


X = train.drop(columns='y', axis=1)
Y = train['y']

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


X_train, X_val, Y_train, Y_val = train_test_split(scaled_X, Y, test_size = 0.1, random_state=42)
print(X_train.shape, X_val.shape, X.shape)


log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
train_pred = log_reg.predict(X_train)
ras = roc_auc_score(train_pred, Y_train)
print(f"Accuracy for training data: {ras}")
val_pred = log_reg.predict(X_val)
ras = roc_auc_score(val_pred, Y_val)
print(f"Accuracy for validation data: {ras}")


dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
train_pred = dtc.predict(X_train)
ras = roc_auc_score(train_pred, Y_train)
print(f"Accuracy for training data: {ras}")
val_pred = dtc.predict(X_val)
ras = roc_auc_score(val_pred, Y_val)
print(f"Accuracy for validation data: {ras}")


gnb = GaussianNB()
gnb.fit(X_train, Y_train)
train_pred = gnb.predict(X_train)
ras = roc_auc_score(train_pred, Y_train)
print(f"Accuracy for training data: {ras}")
val_pred = gnb.predict(X_val)
ras = roc_auc_score(val_pred, Y_val)
print(f"Accuracy for validation data: {ras}")


X_test = test.drop(columns='id', axis=1)
scaled_X_test = scaler.transform(X_test)

preds = dtc.predict(scaled_X_test)
submission = pd.DataFrame({'id': test['id'], 'y': preds})
submission


submission.to_csv('submission.csv', index=False)

