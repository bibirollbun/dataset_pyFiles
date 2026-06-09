# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_df.info()


train_df.isnull().sum()


train_df.head(3)


train_df = train_df.drop('id', axis=1)


train_df = pd.get_dummies(train_df, columns=['Sex'], drop_first=True)


test_df.info()


train_df.head(3)


y_train = train_df['Calories']
x_train = train_df.drop('Calories', axis=1)


rf = RandomForestRegressor()
model = rf.fit(x_train,y_train)


test_df


test_df  = pd.get_dummies(test_df, columns=['Sex'], drop_first=True)


x_test = test_df.drop('id', axis=1)


y_pred = model.predict(x_test)


result = pd.DataFrame()


result['id'] = test_df['id']


result['Calories'] = y_pred


result.head(3)


result.to_csv('result.csv', index=False)




