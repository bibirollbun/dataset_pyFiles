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


import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate


df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv', index_col=0).fillna(0)
df = df.assign(**{
    'year': df['date'].str.split('-').str[0].astype(int),
    'month': df['date'].str.split('-').str[1].astype(int),
    'day': df['date'].str.split('-').str[2].astype(int),
})
df = df.drop(columns=['date'])
# df['date'] = pd.to_datetime(df['date'])
df.head()


sns.boxplot(x='country', y='num_sold', showfliers=True, data=df, showmeans=True)


sns.catplot(x='country', y='num_sold', data=df, kind='box', showmeans=True, col='store', hue='country')


sns.catplot(x='product', y='num_sold', data=df, kind='box', showmeans=True, col='country', hue='product', col_wrap=3)


labelencoder = LabelEncoder()
dit = {}
for col in ['product', 'store', 'country']:
    df[col] = labelencoder.fit_transform(df[col])
    dit[col] = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))


train, test, y_train, y_test = train_test_split(df.drop(columns='num_sold'), df['num_sold'], test_size=0.25, random_state=999)


cls = HistGradientBoostingRegressor()
cls.fit(train, y_train)
cls.score(test, y_test)


cross_validate(cls, train, y_train, cv=5, scoring='r2')





test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv', index_col=0).fillna(0)
test = test.assign(**{
    'year': test['date'].str.split('-').str[0].astype(int),
    'month': test['date'].str.split('-').str[1].astype(int),
    'day': test['date'].str.split('-').str[2].astype(int),
})
test = test.drop(columns=['date'])
test.head()


for col in ['product', 'store', 'country']:
    test[col] = labelencoder.fit_transform(test[col])
    dit[f'test_{col}'] = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))


dit.keys()


test.head()


test = test.assign(num_sold=cls.predict(test).astype(int))
test.head()


test['num_sold'].reset_index().to_csv('submission.csv', index=False)




