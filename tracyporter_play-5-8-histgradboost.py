import numpy as np
import pandas as pd
import os

from scipy.stats import ks_2samp #nonparametric test

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


pd.set_option('display.max_columns', None)


train


train.info()


for col in train:
    if train[col].dtype == 'object':
        print(col, train[col].unique())


train.isna().sum().sum()


test


test.isna().sum().sum()


for col in test:
    if test[col].dtype == 'object':
        print(col, test[col].unique())


submission


del_cols = []

for col in test:
    stat, pv = ks_2samp(train[col], test[col])
    if pv < 0.05:
        del_cols.append(col)

print(del_cols)

train = train.drop(del_cols, axis = 1)
test = test.drop(del_cols, axis = 1)

train.shape, test.shape


#analyse target
plt.bar(train['y'].unique(), train['y'].value_counts(), width=0.1)
plt.show()


#analyse age
plt.hist(train['age'], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.show()


#analyse job
plt.barh(train['job'].unique(), train['job'].value_counts())
plt.show()


#analyse marital status
plt.pie(train['marital'].value_counts(), labels=train['marital'].unique(), autopct='%1.1f%%', startangle=140)
plt.show()


#analyse education
plt.pie(train['education'].value_counts(), labels=train['education'].unique(), autopct='%1.1f%%', startangle=140)
plt.show()


#analyse default
plt.bar(train['default'].unique(), train['default'].value_counts(), width=0.1)
plt.show()


#analyse housing
plt.bar(train['housing'].unique(), train['housing'].value_counts(), width=0.1)
plt.show()


#analyse loan
plt.bar(train['loan'].unique(), train['loan'].value_counts(), width=0.1)
plt.show()


#analyse contact
plt.pie(train['contact'].value_counts(), labels=train['contact'].unique(), autopct='%1.1f%%', startangle=140)
plt.show()


#analyse day
plt.hist(train['day'], bins=31, color='blue', edgecolor='black', alpha=0.7)
plt.show()


#analyse month
plt.barh(train['month'].unique(), train['month'].value_counts())
plt.show()


#analyse poutcome
plt.pie(train['poutcome'].value_counts(), labels=train['poutcome'].unique(), autopct='%1.1f%%', startangle=140)
plt.show()


train_num = train.select_dtypes(exclude = ['object'])
corr = train_num.corr()
sns.heatmap(corr,cmap='crest')


enc = OrdinalEncoder()

for col in test:
    if test[col].dtype =='object':
        train[col] = enc.fit_transform(train[col].values.reshape(-1,1))
        test[col] = enc.transform(test[col].values.reshape(-1,1))


train.info()


test.info()


y = train.pop('y')
X = train
X_test = test


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape


model = HistGradientBoostingClassifier(class_weight='balanced', max_iter=1000, random_state=42).fit(X_train,y_train)
model.score(X_train, y_train)


y_pred = model.predict(X_val)
y_pred


acc = accuracy_score(y_val, y_pred)
acc


#analyse y_pred
unique_values, counts = np.unique(y_pred, return_counts=True)
plt.bar(unique_values, counts, width=0.1)
plt.show()


pred = model.predict_proba(X_test)
pred[:,1]


submission['y'] = pred[:,1]
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

