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


import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore') 
import statistics
import datetime
from datetime import timedelta

# Data loading
stores_data = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
test_data = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
train_data = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
features_data = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
sample_submission = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')

# Data merging and converting dates to datetime
feature_store = features_data.merge(stores_data, how='inner', on="Store")
feature_store['Date'] = pd.to_datetime(feature_store['Date'])
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])

# 修复这里：使用 isocalendar().week 替代 .week
feature_store['Week'] = feature_store['Date'].dt.isocalendar().week
feature_store['Year'] = feature_store['Date'].dt.year
feature_store['Day'] = feature_store['Date'].dt.day

# useable dataframe merging
train_df = train_data.merge(feature_store, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
test_df = test_data.merge(feature_store, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)

# Easter marking
train_df.loc[(train_df.Year==2010) & (train_df.Week==13), 'IsHoliday'] = True
train_df.loc[(train_df.Year==2011) & (train_df.Week==16), 'IsHoliday'] = True
train_df.loc[(train_df.Year==2012) & (train_df.Week==14), 'IsHoliday'] = True
test_df.loc[(test_df.Year==2013) & (test_df.Week==13), 'IsHoliday'] = True

# Cinco De Mayo / Mother's Day
train_df.loc[(train_df.Year==2010) & (train_df.Week==18), 'IsHoliday'] = True
train_df.loc[(train_df.Year==2011) & (train_df.Week==18), 'IsHoliday'] = True
train_df.loc[(train_df.Year==2012) & (train_df.Week==18), 'IsHoliday'] = True
test_df.loc[(test_df.Year==2013) & (test_df.Week==18), 'IsHoliday'] = True

# July 4th
train_df.loc[(train_df.Year==2010) & (train_df.Week==26), 'IsHoliday'] = True
train_df.loc[(train_df.Year==2011) & (train_df.Week==26), 'IsHoliday'] = True
train_df.loc[(train_df.Year==2012) & (train_df.Week==27), 'IsHoliday'] = True
test_df.loc[(test_df.Year==2013) & (test_df.Week==27), 'IsHoliday'] = True

def type_conversion_full(final_data):
    final_data.Type = final_data.Type.apply(lambda x: 3 if x == 'A' else (2 if x == 'B' else 1))
    return final_data

train_df = type_conversion_full(train_df)
test_df = type_conversion_full(test_df)

train_min = train_df[['Store', 'Dept', 'IsHoliday', 'Size', 'Type', 'Week', 'Year', 'Day']].copy()
y = train_df[['Weekly_Sales']].copy()
X_train, X_test, y_train, y_test = train_test_split(train_min, y, random_state=0, test_size=0.1)

RF = RandomForestRegressor()
RF.fit(X_train, y_train.values.ravel())  # 添加 .values.ravel() 解决形状警告

test = test_df[['Store', 'Dept', 'IsHoliday', 'Size', 'Type', 'Week', 'Year', 'Day']].copy()
predict_rf = RF.predict(test)

ETR = ensemble.ExtraTreesRegressor(bootstrap=True, random_state=0)
ETR.fit(X_train, y_train.values.ravel())  # 添加 .values.ravel() 解决形状警告
predict_etr = ETR.predict(test)

avg_preds = (predict_rf + predict_etr) / 2
test_strip = test_df[['Store', 'Dept', 'Date', 'Week', 'Year']]
test_strip['Weekly_Sales'] = avg_preds

def week_51_adj(row):
    compareval = test_strip[(test_strip['Store'] == row.Store) & 
                           (test_strip['Dept'] == row.Dept) & 
                           (test_strip['Week'] == 52)]
    if compareval.empty:
        return row.Weekly_Sales
    elif row.Weekly_Sales > 1.5 * compareval.Weekly_Sales.median():
        return row.Weekly_Sales * 0.85
    else:
        return row.Weekly_Sales

def week_52_adj(row):
    compareval = test_strip[(test_strip['Store'] == row.Store) & 
                           (test_strip['Dept'] == row.Dept) & 
                           (test_strip['Week'] == 51)]
    if compareval.empty:
        return row.Weekly_Sales
    elif row.Weekly_Sales * 1.275 < compareval.Weekly_Sales.median():
        return row.Weekly_Sales * 1.2
    else:
        return row.Weekly_Sales

test_strip['Weekly_Sales'] = test_strip.apply(lambda row: week_51_adj(row) if row.Week == 51 else row.Weekly_Sales, axis=1)
test_strip['Weekly_Sales'] = test_strip.apply(lambda row: week_52_adj(row) if row.Week == 52 else row.Weekly_Sales, axis=1)

sample_submission['Weekly_Sales'] = test_strip['Weekly_Sales']
sample_submission.to_csv('submission.csv', index=False)

