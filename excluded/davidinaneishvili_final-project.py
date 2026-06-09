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


train = pd.read_csv('/kaggle/input/playground-series-s3e1/train.csv')


train


train.info()


test = pd.read_csv('/kaggle/input/playground-series-s3e1/test.csv')


test


train.isnull().sum()


trainset = train.sample(frac = 0.8)
valset = train.drop(trainset.index)


def prepare_data(data, status = "Train", info_dct = None):
    if status == "Train":
        info_dct = {}

    unnecessary_columns = ["id"]

    if status == "Train":
        data = data.drop(unnecessary_columns, axis = 1)
        info_dct["unnecessary_columns"] = unnecessary_columns
    elif status == "Test":
        unnecessary_columns = info_dct.get("unnecessary_columns")
        data = data.drop(unnecessary_columns, axis = 1)
        
    if status == "Train":

        missing_dct = {}
        columns = data.columns
        for i in columns:
            col = data.loc[:, [i]]
            median = col.median()
            missing_dct[i] = median
            data.loc[:, [i]].fillna(median)
        info_dct["Missing"] = missing_dct
    elif status == "Test":
        
        missing_dct = info_dct.get("Missing")
        columns = data.columns
        for i in columns:
            median = missing_dct.get(i)

            data.loc[:, [i]].fillna(median)
        
        
    return data, info_dct


clean_trainset, info_dct = prepare_data(trainset, 'Train')

clean_valset, info_dct = prepare_data(valset, 'Test', info_dct)


clean_trainset


clean_valset

