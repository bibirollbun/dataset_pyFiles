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

!python -m pip install --upgrade pip
!python -m pip install autogluon

from autogluon.tabular import TabularDataset, TabularPredictor


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train.info()


train['Brand_numeric'] = pd.factorize(train['Brand'])[0]
train['Material_numeric'] = pd.factorize(train['Material'])[0]
train['Size_numeric'] = pd.factorize(train['Size'])[0]
train['Laptop Compartment_numeric'] = pd.factorize(train['Laptop Compartment'])[0]
train['Waterproof_numeric'] = pd.factorize(train['Waterproof'])[0]
train['Style_numeric'] = pd.factorize(train['Style'])[0]
train['Color_numeric'] = pd.factorize(train['Color'])[0]


label = 'Price'

predictor = TabularPredictor(label=label).fit(train, time_limit = 200)


predictor.evaluate(train, silent=True)


predictor.leaderboard(train)


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test = test.fillna(value = 0)

test['Brand_numeric'] = pd.factorize(test['Brand'])[0]
test['Material_numeric'] = pd.factorize(test['Material'])[0]
test['Size_numeric'] = pd.factorize(test['Size'])[0]
test['Laptop Compartment_numeric'] = pd.factorize(test['Laptop Compartment'])[0]
test['Waterproof_numeric'] = pd.factorize(test['Waterproof'])[0]
test['Style_numeric'] = pd.factorize(test['Style'])[0]
test['Color_numeric'] = pd.factorize(test['Color'])[0]
y_pred = predictor.predict(test)



output = pd.DataFrame({'id': test.id,
                       'Price': y_pred})
output.to_csv('submission.csv', index=False)


output


