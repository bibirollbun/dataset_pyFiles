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
import glob
import os
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv")
test = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/test.csv")


train.head()


train.shape


train.describe()


train.info()


train['date'] = pd.to_datetime(train['date'])


train.isnull().sum()


train.duplicated().sum()


test.info()


test['date'] = pd.to_datetime(train['date'])


test.isnull().sum()


test.duplicated().sum()


train.shape, test.shape


min(train['date']), max(train['date'])


max(train['date']) - min(train['date'])


train['date'].nunique()


temp = train[['date', 'sales']].groupby(["date"]).sum()

temp.plot(figsize=(20, 5), title="Total sales vs date", grid=True, ylabel="Sales", xlabel="Date")
plt.show()


temp = train[["date", "store", "sales"]].groupby(['date', "store"], as_index=False).sum()

fig = plt.figure()
ax1 = fig.add_subplot(111)

for store in temp['store'].unique():
    temp2 = temp[temp['store'] == store]
    temp2['sales'].plot(figsize=(20, 5), ax=ax1, grid=True, title='Total salse by store vs date', xlabel="Date", ylabel="Sales")

ax1.legend(['store1', 'store2', 'store3', 'store4', 'store5', 'store6', 'store7', 'store8', 'store9', 'store10'])

plt.show()


temp = train[['date', "item", "sales"]].groupby(['date', 'item'], as_index=False).sum()

fig = plt.figure()
ax1 = fig.add_subplot(111)

for item in temp['item'].unique():
    temp2 = temp[temp['item'] == item]
    temp2["sales"].plot(figsize=(20, 5), ax=ax1, grid=True ,title="Total sales by Item vs date", xlabel="Date", ylabel="Salse")

plt.show()


train2 = train[train['date'] >= "2017-01-01"]


## reshape data to apply shift method
train3 = train2.sort_values("date").groupby(['item', 'store', 'date'], as_index=False)
train4 = train3.agg({"sales": ['mean']})
train4.columns = ['item', 'store', 'date', 'sales']


train4.head()


def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()

    # Input sequences --> t-n,.......,t-1
    for i in reange(window, 0, 1):
        cols.append(data.shift(i))
        names += [("%s(t-%d)" % (col, i)) for col in data.columns]
    
    # currnet fimestamp
    cols.append(data)
    name += [("%s(t+%d)" % (col, lag)) for col in data.columns]
    
    # putting all togather
    agg = pd.concat(col, axis=1)
    agg.columns = names
    
    # drop rows with nan values
    if dropna:
        agg.dropna(inplace=True)
    return agg







