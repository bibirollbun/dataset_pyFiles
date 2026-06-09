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


train=pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")


train.head()


train.info()


import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split


X = train.drop(['id', 'sale_price', 'sale_date'], axis=1)
y = train['sale_price']



categoricals = ['sale_warning', 'join_status', 'city', 'zoning', 'subdivision', 'submarket']
for col in categoricals:
    X[col] = X[col].astype('category')


params = {
    'objective': 'quantile',
    'alpha': 0.5,  # for median
    'metric': 'quantile',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}


# Train models for different quantiles
lower_model = lgb.LGBMRegressor(objective='quantile', alpha=0.05)  # 5th percentile
median_model = lgb.LGBMRegressor(objective='quantile', alpha=0.5)  # median
upper_model = lgb.LGBMRegressor(objective='quantile', alpha=0.95)  # 95th percentile


# Fit models
lower_model.fit(X, y)
median_model.fit(X, y)
upper_model.fit(X, y)


test_data = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
X_test = test_data.drop(['id', 'sale_date'], axis=1)
for col in categoricals:
    X_test[col] = X_test[col].astype('category')


lower_pred = lower_model.predict(X_test)
upper_pred = upper_model.predict(X_test)


submission = pd.DataFrame({
    'id': test_data['id'],
    'pi_lower': lower_pred,
    'pi_upper': upper_pred
})
submission.to_csv('submission.csv', index=False)




