import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

import pylab 
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train


train.info()


train.isna().sum()


train.dropna(inplace=True)
train.shape


for col in train:
    print(col, train[col].nunique())


for col in train:
    if train[col].nunique()<7:
        print(col, train[col].unique())


plt.hist(train['country'])
plt.show()


plt.hist(train['store'])
plt.show()


plt.hist(train['product'])
plt.show()


test


for col in test:
    print(col, test[col].nunique())


for col in test:
    if test[col].nunique()<7:
        print(col, test[col].unique())


plt.hist(test['country'])
plt.show()


plt.hist(test['store'])
plt.show()


plt.hist(test['product'])
plt.show()


test.isna().sum()


submission


target = train.pop('num_sold')

plt.hist(target,bins=50)


train.drop('id', axis=1, inplace=True)
test.drop('id',axis=1, inplace=True)

train.shape, test.shape


enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

for col in train:
    if train[col].dtype == 'object':
        train[col] = enc.fit_transform(train[col].values.reshape(-1,1))
        test[col] = enc.transform(test[col].values.reshape(-1,1))


y = target
X = train
X_test = test


X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.1, shuffle=True, random_state=42)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape


model = ExtraTreesRegressor(n_estimators=1000, random_state=42).fit(X_train, y_train)
model.score(X_train, y_train)


y_pred = model.predict(X_val)
y_pred


mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
rmse


stats.probplot(y_pred, dist="norm", plot=pylab)
pylab.show()


fig, ax = plt.subplots()
ax.scatter(y_val, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


pred = model.predict(X_test)
pred


submission['num_sold'] = pred
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

