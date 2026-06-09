
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

import pylab
from scipy.stats import ks_2samp
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train


train.info()


train.isna().sum()


train.dropna(inplace=True)
train.shape


test


test.isna().sum()


submission


train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

train.shape, test.shape


drop_col = []
alpha = 0.05

for col in test:
    stat, p_value = ks_2samp(train[col], test[col])
    if p_value < alpha:
        drop_col.append(col)

print(drop_col)

train = train.drop(drop_col, axis=1)
test = test.drop(drop_col, axis = 1)

train.shape, test.shape


train_numeric = train.select_dtypes(include=['float'], exclude=['object'])
test_numeric = test.select_dtypes(include=['float'], exclude=['object'])
train_numeric.shape, test_numeric.shape


for col in train:
    if train[col].dtype == 'object':
        print(col,train[col].unique() )


for col in test:
    if test[col].dtype == 'object':
        print(col,train[col].unique())


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


plt.hist(train['Style'])
plt.show()


plt.hist(train['Color'], bins=6, rwidth=0.8)
plt.show()


sns.displot(train['Price'], kde=True)


for col in test_numeric:
    if test_numeric[col].dtype == 'object':
        test_numeric[col] = test_numeric[col].fillna('not listed')
    if test_numeric[col].dtype == 'int' or test_numeric[col].dtype == 'float':
        test_numeric[col] = test_numeric[col].fillna(-1)

test_numeric.isna().sum().sum()


enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

for col in train:
    if train[col].dtype == 'object':
        train[col] = enc.fit_transform(train[col].values.reshape(-1,1))
        test[col] = enc.transform(test[col].values.reshape(-1,1))



train.info()


test.info()


y = train_numeric.pop('Price')
X = train_numeric
X_test = test_numeric


X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.1, shuffle=True, random_state=42)
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape


model = ElasticNet(random_state=42).fit(X_train, y_train)
model.score(X_train, y_train)


y_pred = model.predict(X_val)
y_pred


mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
rmse


df = pd.DataFrame({'y_val':y_val, 'y_pred':y_pred})
df


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



submission['Price'] = pred
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

