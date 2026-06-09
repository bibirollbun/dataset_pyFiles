import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


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


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train.head()


train.shape


train.isnull().sum()


train.duplicated().sum()


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test.head()


test.shape


test.isnull().sum()


test.duplicated().sum()


combined = pd.concat([train, test], axis = 0)
combined.shape


combined["Duration_HeartRate"] = combined["Duration"] * combined["Heart_Rate"]


combined["Duration_BodyTemp"] = combined["Duration"] * combined["Body_Temp"]


sex_mapping = {'male':'0','female':'1'}
combined['Sex'] = (combined['Sex'].replace(sex_mapping)).astype(float)


combined = combined.drop('id', axis = 1)


newtrain = combined.iloc[0:750000, :]
newtest = combined.iloc[750000: , :]


newtest = newtest.drop('Calories', axis = 1)


newtrain.head()


newtest.head()


x = newtrain.drop('Calories', axis = 1)
y = newtrain['Calories']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


xgb_model = XGBRegressor(random_state=21, tree_method='hist', device='cuda', n_jobs=-1)
y_pred = xgb_model.fit(x_train, y_train).predict(x_test)
y_pred = np.maximum(y_pred, 0)
np.sqrt(mean_squared_log_error(y_test, y_pred))


x_train = newtrain.drop('Calories', axis = 1)
y_train = newtrain['Calories']
x_test = newtest


xgb_model = XGBRegressor(random_state=21, tree_method='hist', device='cuda', n_jobs=-1)
y_pred = xgb_model.fit(x_train, y_train).predict(x_test)
y_pred = np.maximum(y_pred, 0)


solution = pd.DataFrame({'id' : test['id'], 'Calories' : np.abs(y_pred)})
solution.head()


solution.to_csv('Solution.csv', index = False)




