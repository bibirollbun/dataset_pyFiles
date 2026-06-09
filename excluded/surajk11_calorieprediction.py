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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


df1 = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df2 = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df1.head()


df2.head()


df1.info(), df2.info()


df1.drop(columns = ['id'], inplace = True)


df1.describe()


df2.describe()


# Checking distribution of features
for col in df1.columns:
    if col == 'Sex':
        continue
    sns.displot(df1[col], kind = 'kde')
    plt.show()


# Column Transformer
tnf1 = ColumnTransformer(transformers = [
    ('ohe_sex', OneHotEncoder(sparse_output = False, dtype = 'int'), [0])
], remainder = 'passthrough') 

tnf2 = ColumnTransformer(transformers = [
    ('scale', StandardScaler(), [0, 1, 2, 3, 4, 5, 6])
], remainder = 'passthrough')


pipe = Pipeline([
    ('tnf1', tnf1),
    ('tnf2', tnf2)
])


X = df1.drop(columns = ['Calories'])
y = df1['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


X_train_transformed = pipe.fit_transform(X_train)
X_test_transformed = pipe.transform(X_test)


# Linear regression
lr = LinearRegression()
lr.fit(X_train_transformed, y_train)
y_pred = lr.predict(X_test_transformed)

print(f'r2 score : {r2_score(y_pred, y_test):.4f}')

y_pred = np.maximum(0, y_pred)
print(f'RMSLE : {np.sqrt(mean_squared_log_error(y_pred, y_test)):.4f}')


# Ridge Regression
r = Ridge(alpha = 0.1, max_iter = 1000, tol = 1e-6)
r.fit(X_train_transformed, y_train)
y_pred_ridge = r.predict(X_test_transformed)

print(f'r2 score : {r2_score(y_pred_ridge, y_test):.4f}')

y_pred_ridge = np.maximum(0, y_pred_ridge)
print(f'RMSLE : {np.sqrt(mean_squared_log_error(y_pred_ridge, y_test)):.4f}')


# Lasso Regression
ls = Lasso(alpha = 0.1, max_iter = 1000, tol = 1e-6)
ls.fit(X_train_transformed, y_train)
y_pred_lasso = ls.predict(X_test_transformed)

print(f'r2 score : {r2_score(y_pred_lasso, y_test):.4f}')

y_pred_lasso = np.maximum(0, y_pred_lasso)
print(f'RMSLE : {np.sqrt(mean_squared_log_error(y_pred_lasso, y_test)):.4f}')


# ElasticNet Regression
en = ElasticNet(alpha = 0.1, max_iter = 1000, tol = 1e-6)
en.fit(X_train_transformed, y_train)
y_pred_en = en.predict(X_test_transformed)

print(f'r2 score : {r2_score(y_pred_en, y_test):.4f}')

y_pred_en = np.maximum(0, y_pred_en)
print(f'RMSLE : {np.sqrt(mean_squared_log_error(y_pred_en, y_test)):.4f}')


# Polynomial regression
poly = PolynomialFeatures(degree = 3, include_bias = False)
X_train_poly = poly.fit_transform(X_train_transformed)
X_test_poly = poly.transform(X_test_transformed)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)

print(f'r2 score : {r2_score(y_pred_poly, y_test):.4f}')

y_pred_poly = np.maximum(0, y_pred_poly)
print(f'RMSLE : {np.sqrt(mean_squared_log_error(y_pred_poly, y_test)):.4f}')


df = df2.copy()
df.head()


X = df.drop(columns = 'id')


X_transformed = pipe.transform(X)
X_poly = poly.fit_transform(X_transformed)
prediction = lr_poly.predict(X_poly)


prediction = np.maximum(0, prediction)


df = pd.DataFrame(prediction, columns=['Calories'])
df.index = df2['id']
df.to_csv('submission.csv')

