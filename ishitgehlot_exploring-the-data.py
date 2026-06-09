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



train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
ext_train = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train.head()


ext_train.head()


print(f'train shape :{train.shape}   ext_train shape: {ext_train.shape} ')


train.info()


ext_train.info()


ext_train.isnull().sum()


train.isnull().sum()


train.isnull().sum().sum()


57199/300000


train.describe()


train.head()


# sns.histplot(data = train, x = 'Weight Capacity (kg)', kde = True)
import matplotlib.style as style
style.use('fivethirtyeight')
train['Weight Capacity (kg)'].dropna().plot(kind = 'hist')


train['Weight Capacity (kg)'].dropna().plot(kind = 'kde')


train['Price'].dropna().plot(kind = 'kde')


sns.countplot(x = train['Brand'].dropna())


sns.countplot(x = train['Material'].dropna())


sns.countplot(x = train['Style'].dropna())


sns.countplot(x = train['Size'].dropna())


sns.countplot(x = train['Color'].dropna())


sns.barplot(x = train['Color'].dropna(), y = train['Price'])


train.head()


pd.crosstab(train['Style'], train['Color'], dropna = True).plot(kind = 'bar')


sns.barplot(x = train['Material'].dropna(), y = train['Price'])


sns.barplot(x = train['Brand'].dropna(), y = train['Price'])


sns.barplot(x = train['Waterproof'].dropna(), y = train['Price'])


sns.barplot(x = train['Compartments'].dropna(), y = train['Price'])


sns.boxplot(x = train['Price'])


q1 = np.quantile(train['Price'], 0.25)
q3 = np.quantile(train['Price'], 0.75)
iqr = q3-q1
train[~((train['Price'] > q1 - 1.5 * iqr) & (train['Price'] < q3 + 1.5 * iqr))]




