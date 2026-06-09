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


import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


print(f'Train Set Shape: {train.shape}')
print(f'Test Set Shape: {test.shape}')


train.info()


test.info()


train.head()


test.head()


print(train['country'].nunique())
train['country'].value_counts()


print(train['store'].nunique())
train['store'].value_counts()


print(train['product'].nunique())
train['product'].value_counts()


train.duplicated().sum(), test.duplicated().sum()


train.isna().sum()


test.isna().sum()


train[train['num_sold'].isna()].groupby('store').size()


train[train['num_sold'].isna()].groupby('product').size()


train[train['num_sold'].isna()].groupby('country').size()


train[train['num_sold'].isna()].groupby(['country', 'store', 'product']).size()


train[train['num_sold'].isna()].groupby(['store', 'product']).size()


#Date column
train['day_of_week'] = pd.to_datetime(train['date']).dt.day_name()
test['day_of_week'] = pd.to_datetime(test['date']).dt.day_name()

train['is_weekend'] = train['day_of_week'].isin(['Saturday', 'Sunday'])
test['is_weekend'] = test['day_of_week'].isin(['Saturday', 'Sunday'])

train['month'] = pd.to_datetime(train['date']).dt.month
test['month'] = pd.to_datetime(test['date']).dt.month

train['year'] = pd.to_datetime(train['date']).dt.year
test['year'] = pd.to_datetime(test['date']).dt.year

train['quarter'] = pd.to_datetime(train['date']).dt.quarter
test['quarter'] = pd.to_datetime(test['date']).dt.quarter

train['day_of_month'] = pd.to_datetime(train['date']).dt.day
test['day_of_month'] = pd.to_datetime(test['date']).dt.day

train['week_of_year'] = pd.to_datetime(train['date']).dt.isocalendar().week
test['week_of_year'] = pd.to_datetime(test['date']).dt.isocalendar().week


# #Check each country has own specific holidays
# import holidays
# us_holidays = holidays.US()

# train['is_holiday'] = pd.to_datetime(train['date']).isin(us_holidays)
# test['is_holiday'] = pd.to_datetime(test['date']).isin(us_holidays)


print(train.shape)
train.head()


train.info()


train.describe().T


train.describe(include='O').T


print(test.shape)
test.describe().T


test.info()


test.describe().T


test.describe(include='O').T


import matplotlib.pyplot as plt
import seaborn as sns


train_date = train['date']
test_date = test['date']

test_id = test['id']

train = train.drop(columns=['id', 'date'], axis=1)
test = test.drop(columns=['id', 'date'], axis=1)


#Change the type
train['week_of_year'] = train['week_of_year'].astype('int32')
test['week_of_year'] = test['week_of_year'].astype('int32')


cat_cols = [col for col in train.select_dtypes('O').columns]
num_cols = [col for col in train.select_dtypes('number').columns]
print(len(cat_cols), '\n', cat_cols)
print(len(num_cols), '\n', num_cols)


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(10,6))
index = 0
axs = axs.flatten()

for col in num_cols:
    sns.histplot(train[col], bins=50, ax=axs[index])
    index += 1
    
plt.tight_layout();


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(10,6))
index = 0
axs = axs.flatten()

for col in num_cols:
    sns.boxplot(y=col,data=train, ax=axs[index])
    index += 1
    
plt.tight_layout();


fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
index = 0
axs = axs.flatten()
for col in cat_cols:
    sns.countplot(train,x=col, ax=axs[index])
    index += 1

plt.tight_layout();


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(10,6))
index = 0
axs = axs.flatten()

for col in test.select_dtypes('number').columns:
    sns.histplot(test[col], bins=50, ax=axs[index])
    index += 1

for ax in axs[index:]:
    ax.axis('off')
    
plt.tight_layout();


plt.figure(dpi=100)

fig, axs = plt.subplots(nrows=2,ncols=3,figsize=(10,6))
index = 0
axs = axs.flatten()

for col in test.select_dtypes('number').columns:
    sns.boxplot(y=col,data=test, ax=axs[index])
    index += 1

for ax in axs[index:]:
    ax.axis('off')
    
    
plt.tight_layout();


fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
index = 0
axs = axs.flatten()
for col in cat_cols:
    sns.countplot(test,x=col, ax=axs[index])
    index += 1

plt.tight_layout();


plt.figure(figsize=(12,6), dpi=100)
plt.subplot(1,2,1)
sns.countplot(data=train, x='is_weekend');

plt.subplot(1,2,2)
sns.countplot(data=test, x='is_weekend');

plt.tight_layout()


sns.countplot(data=test, x='country', hue='product');


plt.figure(dpi=200)
sns.histplot(data=train, x='num_sold', hue='product');


plt.figure(dpi=200)
sns.histplot(data=train, x='num_sold', hue='store');


plt.figure(dpi=200)
sns.histplot(data=train, x='num_sold', hue='country');

