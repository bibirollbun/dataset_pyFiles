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


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


train.head()


train.isnull().sum()


train.duplicated().sum()


train.dtypes


train.describe()


import matplotlib.pyplot as plt
import seaborn as sns


#sns.countplot(x=train.Sex)


sns.countplot(data=train, x='Sex')


sns.histplot(x=train.Height,kde=True)


sns.histplot(x=train['Weight'],kde=True)


numerical_cols = train.select_dtypes(include = {'number'}).columns.drop('id')


correlation = train.corr(numeric_only=True)
sns.heatmap(correlation,cmap='Blues')


#train.replace({'Sex':{'male':0, 'female':1}},inplace=True)


train['Sex'] = train['Sex'].replace({'male': 0, 'female': 1}).astype(int)


train.head()


numerical_cols


plt.figure(figsize=(15, 8))
sns.boxplot(data = train[numerical_cols])


for col in numerical_cols:
    sns.boxplot(data = train[col])
    plt.title(col)
    plt.show()




