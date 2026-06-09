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


import numpy as np
import pandas as pd


df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_train.head()


df_train.info()


# df_train['store'].unique() # 3 products
# df_train['country'].unique() # 6 products
df_train['product'].unique() # 5 products


df_train.drop('id', axis=1, inplace=True)


# We need to cast the date object column into datetime type
df_train['date_'] = pd.to_datetime(df_train['date'], format="%Y-%m-%d")

df_train['year'] = df_train['date_'].dt.year
df_train['month'] = df_train['date_'].dt.month
df_train['date_only'] = df_train['date_'].dt.day
df_train.drop('date', axis=1, inplace=True)


# for col in df_train.columns:
#     print(df_train[col].value_counts())

for col in df_train.columns:
    print(df_train[col].value_counts(normalize=True) * 100)


df_train[df_train.columns].isnull().sum() # the missing value are from num_sold
df_train.dropna(inplace=True)


df_train[df_train.columns].isnull().sum()


df_train.shape[0]


import seaborn as sns
import matplotlib.pyplot as plt

# categorical_columns = df_train.columns
for col in df_train.columns:
    if col != 'num_sold':
        sns.countplot(x=col, data=df_train)
        plt.show()
    else:
        continue


for col in df_train.columns:
    if col not in ['num_sold', 'date_']:
        sns.boxplot(x=col, y='num_sold', data=df_train)
        plt.show()
    else:
        continue


nominal_col = ['country', 'store', 'product', 'num_sold']
df_train = df_train[nominal_col]


# df_train.drop('date_', axis=1, inplace=True)
df_train = pd.get_dummies(data=df_train, columns=['country', 'store', 'product'], drop_first=True, dtype=float)


df_train.dropna(inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


X = df_train.drop(columns=['num_sold'])
y = df_train['num_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))


df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
df_test.head()

df_id = df_test[['id']]
df_test.drop('date', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)

df_test = pd.get_dummies(data=df_test, columns=['country', 'store', 'product'], drop_first=True, dtype=float)


y_test = model.predict(df_test)
df_id['num_sold'] = y_test


df_id.to_csv("submission.csv", index=False)

