import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

import pylab
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e2'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.info()


train.isna().sum()


train.shape


train.dropna(inplace=True)
train.shape


test


test.isna().sum()


test.shape


submission


train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

train.shape, test.shape


for col in train:
    if train[col].dtype == 'object':
        print(col, train[col].unique())


plt.hist(train['Brand'])
plt.show()


plt.hist(train['Material'])
plt.show()


plt.hist(train['Size'])
plt.show()


plt.hist(train['Laptop Compartment'])
plt.show()


plt.hist(train['Waterproof'])
plt.show()


plt.hist(train['Color'], bins=6, width=0.8)
plt.show()


sns.displot(train['Price'], kde=True)


for col in test:
    if test[col].dtype == 'object':
        test[col] = test[col].fillna('not listed')
    elif test[col].dtype in ['int64', 'float64']:
        test[col] = test[col].fillna(-1)

print(test.isna().sum().sum())


enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

for col in train:
    if train[col].dtype == 'object':
        train[col] = enc.fit_transform(train[col].values.reshape(-1, 1))
        test[col] = enc.transform(test[col].values.reshape(-1, 1))


train.info()


y = train.pop('Price')
X = train
X_test = test


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape


model = ElasticNet(random_state=42).fit(X_train, y_train)
model.score(X_train, y_train)


y_pred = model.predict(X_val)
y_pred


mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
rmse


df = pd.DataFrame({'y_test':y_val, 'y_pred':y_pred})
df


stats.probplot(y_pred, dist="norm", plot=pylab)
pylab.show()


fig, ax = plt.subplots()
ax.scatter(y_val, y_pred, edgecolors=(0,0,0))
ax.plot([y.min(), y.max()],[y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


pred =model.predict(X_test)
pred


if len(pred) != len(test):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X_train = train.drop(columns=['Price'])
    y_train = train['Price']
    model.fit(X_train, y_train)
    pred = model.predict(test)


submission['Price'] = pred
submission.to_csv('bagpack_submission_v1.csv', index=False)
submission = pd.read_csv('bagpack_submission_v1.csv')
submission

