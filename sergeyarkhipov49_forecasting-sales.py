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


# load data
df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


df.head()


df.info()


df.isna().sum()


df['date'] = pd.to_datetime(df['date'])
df['country'] = df['country'].astype('string')
df['store'] = df['store'].astype('string')
df['product'] = df['product'].astype('string')


df['date'].max() - df['date'].min()


df['country'].value_counts()


df['store'].value_counts()


df['product'].value_counts()


df.loc[df['num_sold'].isna(), ['country', 'product']].value_counts()


indexes = df.loc[df['num_sold'].isna(), ['country', 'product']].value_counts().index.to_list()


for country, product in indexes:
    cond = (df['country'] == country) & (df['product'] == product)
    df.loc[((df['num_sold'].isna()) & cond), 'num_sold'] = df.loc[cond, 'num_sold'].median()


df['num_sold'].hist(bins=32)


df['log_num_sold'] = np.log(df['num_sold'])


df['log_num_sold'].hist(bins=32)


df.boxplot(column=['num_sold'])


df.boxplot(column=['log_num_sold'])


(df.loc[df['log_num_sold'] < 2.5].shape[0] / df.shape[0]) * 100


df_filter = df[df['log_num_sold'] > 2.5]


df_filter


df_filter['year'] = df_filter['date'].dt.year
df_filter['month'] = df_filter['date'].dt.month
df_filter['day'] = df_filter['date'].dt.day
df_filter['quarter'] = df_filter['date'].dt.quarter
df_filter['dow'] = df_filter['date'].dt.dayofweek
df_filter['is_weekend'] = df_filter['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)


df_filter.head()


df_ohe = pd.get_dummies(df_filter, columns=['country', 'store', 'product'], dtype='int')


from sklearn.model_selection import train_test_split


target = df_ohe['num_sold']
df_ohe.drop(columns=['date', 'num_sold', 'log_num_sold'], inplace=True)


X_train, X_test, y_train, y_test = train_test_split(df_ohe, target, test_size=0.3, shuffle=True)


from sklearn.metrics import mean_absolute_percentage_error as mape


# Проба линейной регрессии

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'MAPE: {mape(y_test, y_pred)}')


# Проба дерева

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'MAPE: {mape(y_test, y_pred)}')


# Проба леса

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'MAPE: {mape(y_test, y_pred)}')


# Проба градиента

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'MAPE: {mape(y_test, y_pred)}')


model = RandomForestRegressor(n_estimators=100, max_depth=20)
model.fit(X_train, y_train)


!pip install lightautoml


train = df_ohe[:round(df_ohe.shape[0] * 0.7)]
test = df_ohe[round(df_ohe.shape[0] * 0.7):]


train.drop(columns=['log_num_sold', 'date'], inplace=True)


from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

automl = TabularAutoML(task = Task(name='reg', loss='mae', metric=mape))
oof_preds = automl.fit_predict(train, roles = {'target': 'num_sold'}, verbose=1).data
test_preds = automl.predict(test).data


mape(test['num_sold'], test_preds)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

df_test['date'] = pd.to_datetime(df_test['date'])
df_test['country'] = df_test['country'].astype('string')
df_test['store'] = df_test['store'].astype('string')
df_test['product'] = df_test['product'].astype('string')

df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['quarter'] = df_test['date'].dt.quarter
df_test['dow'] = df_test['date'].dt.dayofweek
df_test['is_weekend'] = df_test['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

df_test_ohe = pd.get_dummies(df_test, columns=['country', 'store', 'product'], dtype='int')

test_predict = automl.predict(df_test_ohe).data


pd.Series(data=test_predict, index=df_for_model['id']).to_csv('submission.csv')

