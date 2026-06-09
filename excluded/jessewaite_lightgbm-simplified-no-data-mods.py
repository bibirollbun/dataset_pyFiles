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


import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submit_df = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

# remove id column from the test dataset
test_df.drop(columns=['id'], inplace=True)


train_df.head()


train_df.describe()


train_df.info()


X = train_df.drop(columns=['id','BeatsPerMinute'])
y = train_df['BeatsPerMinute']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


#LightGBM model begin
lgb_params = {'feature_fraction':1,
              'learning_rate':0.01,
              'max_depth':10,
              'metric':'rmse',
              'n_estimators':315,
              'num_leaves':30,
              'objective':'regression',
              'random_state':42}
lgb_model = LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train)


lgb_pred = lgb_model.predict(X_test)
lgb_rmse = mean_squared_error(y_test, lgb_pred)
print(f'Root Mean Squared Error: {np.sqrt(lgb_rmse):.6f}')


fig, (ax1, ax2) = plt.subplots(1,2)
plt.suptitle('BPM Histograms')
ax1.hist(lgb_pred, bins=50)
ax1.set_xlabel('LightGBM Predicted')
ax2.hist(y_test, bins=50)
ax2.set_xlabel('Actual')
plt.tight_layout()
plt.show()


#LightGBM
pred = lgb_model.predict(test_df)

data = {'id': submit_df['id'],
        'BeatsPerMinute':pred}

df = pd.DataFrame(data)
df.to_csv('submission.csv', index=False)
df[:5]

