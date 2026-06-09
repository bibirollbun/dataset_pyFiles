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
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor,DMatrix
from sklearn.metrics import mean_absolute_percentage_error as MAPE


#load and check the data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
print('df_train: {} df_test: {}'.format(df_train.shape, df_test.shape))
# df_train: (230130, 6) df_test: (98550, 5)
df_train.head(2)


TARGET = 'num_sold'
CATS = [x for x in df_train.columns if df_train[x].dtype == 'object' and x != 'date']
CATS


# print('train date range',df_train.loc[:,'date'][0],' to ',df_train.loc[:,'date'][len(df_train)-1])              
# print('test date range ',df_test.loc[:,'date'][0],' to ',df_test.loc[:,'date'][len(df_test)-1])
# # train date range 2010-01-01  to  2016-12-31
# # test date range  2017-01-01  to  2019-12-31


# there will be some missing sales data to deal with
df_train.isnull().sum()


# for now we will just drop the observations with null sales
df_train = df_train.dropna(subset=TARGET)
df_train.shape


# create date-related columns
def date_transform(df):
    df['date'] =  pd.to_datetime(df['date'])
    df['month'] =  df['date'].dt.strftime('%B')
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['day number'] =df['date'].dt.dayofweek
    df['day of week'] =df['date'].dt.strftime('%A')

date_transform(df_train)
date_transform(df_test)
df_train.head(1)


from holidays import CountryHoliday
import datetime as dt

dd_holidays = {}       # for reference and review
dd_holiday_dates = {}  # this used in setting the holiday_flag
for c in df_train['country'].unique():  
    data = []
    holiday_list = []
    for h in CountryHoliday(c, years = np.arange(df_train['year'].min(), df_test['year'].max() + 1,1)).items():
        d = h[0].strftime('%Y-%m-%d')
        data.append({'holiday_date': d, 'holiday':h[1]})
        holiday_list.append(d)
    dd_holidays[c] = data    
    dd_holiday_dates[c] = holiday_list


def set_holiday(df_row):
    c = df_row['country']
    date = df_row['date'].strftime('%Y-%m-%d')
    holidays = dd_holiday_dates[c]
    if date in holidays:
        return 1
    else:
        return 0
    
df_train['holiday_flag'] = df_train.apply(lambda x: set_holiday(x), axis=1)
df_test['holiday_flag'] = df_test.apply(lambda x: set_holiday(x), axis=1)

print('train holiday sum:', df_train['holiday_flag'].sum(), 'test holiday sum:', df_test['holiday_flag'].sum())
# train holiday sum: 6948 test holiday sum: 3045

