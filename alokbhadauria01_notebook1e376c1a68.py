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


train = pd.read_csv("/kaggle/input/used-car-price-prediction-during-inflation/train.csv")
train.head()


train = train.drop(['id'], axis=1)


train.info()


train['brand'].unique()


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


train['brand'] = encoder.fit_transform(train['brand'])
train['model'] = encoder.fit_transform(train['model'])
train['fuel_type'] = encoder.fit_transform(train['fuel_type'])
train['engine'] = encoder.fit_transform(train['engine'])
train['transmission'] = encoder.fit_transform(train['transmission'])
train['ext_col'] = encoder.fit_transform(train['ext_col'])
train['int_col'] = encoder.fit_transform(train['int_col'])
train['accident'] = encoder.fit_transform(train['accident'])
train['clean_title'] = encoder.fit_transform(train['clean_title'])


train = pd.DataFrame(train)
train.corr()


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'most_frequent')
train = imputer.fit_transform(train)


test = pd.read_csv("/kaggle/input/used-car-price-prediction-during-inflation/test.csv")
test.head()


test = test.drop(['id'], axis=1)
test.info()


test['brand'] = encoder.fit_transform(test['brand'])
test['model'] = encoder.fit_transform(test['model'])
test['fuel_type'] = encoder.fit_transform(test['fuel_type'])
test['engine'] = encoder.fit_transform(test['engine'])
test['transmission'] = encoder.fit_transform(test['transmission'])
test['ext_col'] = encoder.fit_transform(test['ext_col'])
test['int_col'] = encoder.fit_transform(test['int_col'])
test['accident'] = encoder.fit_transform(test['accident'])
test['clean_title'] = encoder.fit_transform(test['clean_title'])


test = pd.DataFrame(test)
test.corr()

