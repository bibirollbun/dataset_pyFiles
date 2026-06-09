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


col = train.columns


train.info()


train.columns


train['accident'].unique()


train['brand'].unique()


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
train['brand'] = encoder.fit_transform(train['brand'])


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'most_frequent')

train = pd.DataFrame(train)
train = imputer.fit_transform(train)


train=pd.DataFrame(train,columns=col)
train


train.head()


train.isna().sum()


train.info()


test_csv = pd.read_csv("/kaggle/input/used-car-price-prediction-during-inflation/test.csv")
test_csv


test_csv.head()


col = test_csv.columns
col


test_csv.info()


test_csv.describe()


test_csv.isnull().sum()


test_csv['model'].unique()


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

test_csv['model']=encoder.fit_transform(test_csv['model'])


test_csv.head()


test_csv.info()


test_csv['brand'].unique()


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

test_csv['brand']=encoder.fit_transform(test_csv['brand'])


test_csv.info()


test_csv['fuel_type'].unique()


test_csv['fuel_type']=encoder.fit_transform(test_csv['fuel_type'])


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="most_frequent")


test_csv.head()


test_csv['brand'] = encoder.fit_transform(test_csv['brand'])
test_csv['model'] = encoder.fit_transform(test_csv['model'])
test_csv['fuel_type'] = encoder.fit_transform(test_csv['fuel_type'])
test_csv['engine'] = encoder.fit_transform(test_csv['engine'])
test_csv['transmission'] = encoder.fit_transform(test_csv['transmission'])
test_csv['ext_col'] = encoder.fit_transform(test_csv['ext_col'])
test_csv['int_col'] = encoder.fit_transform(test_csv['int_col'])
test_csv['accident'] = encoder.fit_transform(test_csv['accident'])
test_csv['clean_title'] = encoder.fit_transform(test_csv['clean_title'])


test_csv.isnull().sum()


test_csv.head()

