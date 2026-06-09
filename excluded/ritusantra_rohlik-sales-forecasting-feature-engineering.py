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


import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split


df_train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
df_test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')


df_train.info()


df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])


df_train.info()


df_test.info()


X = df_train.drop(columns=['sales','availability'])
y = df_train['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


X_train['warehouse'].nunique()


X_train.isnull().sum()


X_test.isnull().sum()


df_test.isnull().sum()


fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(15,4))

sns.boxplot(X_train['total_orders'].values, ax=ax1,orient='h')
ax1.set_title('X_train total_orders')

sns.boxplot(X_test['total_orders'].values, ax=ax2,orient='h')
ax2.set_title('X_test total_orders')

plt.show()


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


transformer = ColumnTransformer(transformers=[
                                    ('ohe',OneHotEncoder(drop='first', sparse_output=False),['warehouse']),
                                    ('si',SimpleImputer(strategy='median'),['total_orders'])    
                                ], 
                                remainder='passthrough')

transformer2 = ColumnTransformer(transformers=[
                                    ('ohe',OneHotEncoder(drop='first', sparse_output=False),['warehouse'])    
                                ], 
                                remainder='passthrough')


X_train_trf = transformer.fit_transform(X_train)
X_test_trf = transformer.transform(X_test)

df_test_trf = transformer2.fit_transform(df_test)


X_train_trf = pd.DataFrame(X_train_trf, columns=transformer.get_feature_names_out())
X_test_trf = pd.DataFrame(X_test_trf, columns=transformer.get_feature_names_out())

df_test_trf = pd.DataFrame(df_test_trf, columns=transformer2.get_feature_names_out())


X_train_trf.shape, X_test_trf.shape


df_test_trf.shape


df_test_trf.isnull().sum()


X_train_trf['remainder__date']


X_train_trf['dt_year'] = X_train_trf['remainder__date'].dt.year
X_train_trf['dt_month'] = X_train_trf['remainder__date'].dt.month
X_train_trf['dt_day'] = X_train_trf['remainder__date'].dt.day

X_train_trf.drop(columns=['remainder__date'],inplace=True)

X_test_trf['dt_year'] = X_test_trf['remainder__date'].dt.year
X_test_trf['dt_month'] = X_test_trf['remainder__date'].dt.month
X_test_trf['dt_day'] = X_test_trf['remainder__date'].dt.day

X_test_trf.drop(columns=['remainder__date'],inplace=True)

df_test_trf['dt_year'] = df_test_trf['remainder__date'].dt.year
df_test_trf['dt_month'] = df_test_trf['remainder__date'].dt.month
df_test_trf['dt_day'] = df_test_trf['remainder__date'].dt.day

df_test_trf.drop(columns=['remainder__date'],inplace=True)


X_test_trf = X_test_trf.astype('int')
X_train_trf = X_train_trf.astype('int')

df_test_trf = df_test_trf.astype('int')


df_test_trf.columns = ['warehouse_Budapest_1', 'warehouse_Frankfurt_1',
                       'warehouse_Munich_1', 'warehouse_Prague_1', 'warehouse_Prague_2', 'warehouse_Prague_3',
                       'unique_id', 'total_orders', 'sell_price_main', 'type_0_discount', 'type_1_discount', 'type_2_discount', 
                       'type_3_discount', 'type_4_discount', 'type_5_discount', 'type_6_discount',
                       'dt_year', 'dt_month', 'dt_day']


X_train_trf.columns


columns_to_plot = ['si__total_orders','remainder__sell_price_main','remainder__type_0_discount',
       'remainder__type_1_discount', 'remainder__type_2_discount',
       'remainder__type_3_discount', 'remainder__type_4_discount',
       'remainder__type_5_discount', 'remainder__type_6_discount']


for col in columns_to_plot :
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_train_trf[col])
    plt.title(col)
    
    plt.subplot(122)
    stats.probplot(X_train_trf[col], dist='norm',plot=plt)
    plt.title(col)
    
    plt.show()


for col in columns_to_plot :
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_test_trf[col])
    plt.title(col)
    
    plt.subplot(122)
    stats.probplot(X_test_trf[col], dist='norm',plot=plt)
    plt.title(col)
    
    plt.show()


print(X_train_trf['si__total_orders'].skew())
print(X_train_trf['remainder__sell_price_main'].skew())

print(X_test_trf['si__total_orders'].skew())
print(X_test_trf['remainder__sell_price_main'].skew())


from sklearn.preprocessing import FunctionTransformer

trf = ColumnTransformer(transformers=[
    ('lt1',FunctionTransformer(func=np.log1p),['si__total_orders']),
   ('lt2',FunctionTransformer(func=np.log1p),['remainder__sell_price_main']),
    ], remainder='passthrough')

X_train_log_trf = trf.fit_transform(X_train_trf)
X_test_log_trf = trf.transform(X_test_trf)


X_train_log_trf = pd.DataFrame(X_train_log_trf,
columns = ['total_orders', 'sell_price_main','warehouse_Budapest_1', 'warehouse_Frankfurt_1',
       'warehouse_Munich_1', 'warehouse_Prague_1',
       'warehouse_Prague_2', 'warehouse_Prague_3',  'unique_id', 'type_0_discount',
       'type_1_discount', 'type_2_discount',
       'type_3_discount', 'type_4_discount',
       'type_5_discount', 'type_6_discount', 'dt_year',
       'dt_month', 'dt_day'])



X_test_log_trf = pd.DataFrame(X_test_log_trf,
columns = ['total_orders', 'sell_price_main','warehouse_Budapest_1', 'warehouse_Frankfurt_1',
       'warehouse_Munich_1', 'warehouse_Prague_1',
       'warehouse_Prague_2', 'warehouse_Prague_3',  'unique_id', 'type_0_discount',
       'type_1_discount', 'type_2_discount',
       'type_3_discount', 'type_4_discount',
       'type_5_discount', 'type_6_discount', 'dt_year',
       'dt_month', 'dt_day'])


X_train_log_trf.shape, X_test_log_trf.shape


print(y_train.isnull().sum())
print(y_test.isnull().sum())


print(y_train.skew())
print(y_test.skew())


y_train_trf = y_train.fillna(y_train.median())
y_test_trf = y_test.fillna(y_test.median())


print(y_train_trf.isnull().sum())
print(y_test_trf.isnull().sum())


y_train_trf.shape, y_test_trf.shape


X_train_log_trf = X_train_log_trf.astype('int')
X_test_log_trf = X_test_log_trf.astype('int')


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_log_trf,y_train_trf)


y_pred = lr.predict(X_test_log_trf)

from sklearn.metrics import r2_score
r2 = r2_score(y_test_trf, y_pred)
print("R² score for Linear Regression:", r2)


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train_log_trf,y_train_trf)


y_pred2 = dt.predict(X_test_log_trf)

from sklearn.metrics import r2_score
r2 = r2_score(y_test_trf, y_pred2)
print("R² score for Decision Tree Regressor:", r2)


new_order_df_test_trf = ['total_orders', 'sell_price_main', 'warehouse_Budapest_1',
       'warehouse_Frankfurt_1', 'warehouse_Munich_1', 'warehouse_Prague_1',
       'warehouse_Prague_2', 'warehouse_Prague_3', 'unique_id',
       'type_0_discount', 'type_1_discount', 'type_2_discount',
       'type_3_discount', 'type_4_discount', 'type_5_discount',
       'type_6_discount', 'dt_year', 'dt_month', 'dt_day']


df_test_trf = df_test_trf[new_order_df_test_trf]


df_test_trf.columns


y_pred3 = dt.predict(df_test_trf)

df_test['sales_hat'] = y_pred3
df_test.head()


df_test[['unique_id','date','sales_hat']].head()


df_test['unique_id'] = df_test['unique_id'].astype(str)
df_test['date_str'] = df_test['date'].dt.strftime('%Y-%m-%d')


df_test[['unique_id','date_str','sales_hat']].info()


df_test['id'] = df_test['unique_id'] + '_' + df_test['date_str']


soution = df_test[['id','sales_hat']]


soution.to_csv('solution.csv', index=False)




