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


train.info()


train.isnull().sum()


from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy='most_frequent')


train[['fuel_type']]=imputer.fit_transform(train[['fuel_type']])


train[['accident']]=imputer.fit_transform(train[['accident']])



train[['clean_title']]=imputer.fit_transform(train[['clean_title']])



train.isnull().sum()



from sklearn.preprocessing import LabelEncoder


encoder=LabelEncoder()


train['fuel_type']=encoder.fit_transform(train['fuel_type'])


train['accident']=encoder.fit_transform(train['accident'])



train['clean_title']=encoder.fit_transform(train['clean_title'])



train['engine']=encoder.fit_transform(train['engine'])



train['transmission']=encoder.fit_transform(train['transmission'])


train['ext_col']=encoder.fit_transform(train['ext_col'])



train['int_col']=encoder.fit_transform(train['int_col'])



train['brand']=encoder.fit_transform(train['brand'])



train['model']=encoder.fit_transform(train['model'])



train.info()


train.corr()




