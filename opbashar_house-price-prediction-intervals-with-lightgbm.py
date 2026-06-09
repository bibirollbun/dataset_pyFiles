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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
test.head()


train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
train.head()


train['sale_date'] = pd.to_datetime(train['sale_date'])
test['sale_date'] = pd.to_datetime(test['sale_date'])



train['year'] = train['sale_date'].dt.year
test['year'] = test['sale_date'].dt.year

train['month'] = train['sale_date'].dt.month
test['month'] = train['sale_date'].dt.month

train['day'] = train['sale_date'].dt.day
test['day'] = train['sale_date'].dt.day







latest_sale_date = train['sale_date'].max()
time_differnces = latest_sale_date - train['sale_date']
train['age_of_sale_days'] = time_differnces.dt.days
train['age_of_sale_years'] = (time_differnces / np.timedelta64(1, 'D')) / 365.25


train.head()


latest_sale_date = train['sale_date'].max()
time_differnces = latest_sale_date - train['sale_date']
test['age_of_sale_days'] = time_differnces.dt.days
test['age_of_sale_years'] = (time_differnces / np.timedelta64(1, 'D')) / 365.25


train.head()


train.info()
train.describe()


train['val_ratio'] = train['land_val']/train['imp_val']
test['val_ratio'] = test['land_val']/test['imp_val']


train['total_baths'] = train['bath_full'] + 0.75 * train['bath_3qtr'] + 0.5 * train['bath_half']
test['total_baths'] = test['bath_full'] + 0.75 * test['bath_3qtr'] + 0.5 * test['bath_half']

train['bath_to_beds'] = train['total_baths']/train['beds']
test['bath_to_beds'] = test['total_baths']/test['beds']



test.drop(columns = ['sale_nbr'],inplace = True)


train.drop(columns=['sale_nbr'], inplace=True)



train.isnull().sum()



train['bath_to_beds'].dtype


train.head()


train['sale_price_log'] = np.log1p(train['sale_price'])


categorical_features = ['submarket', 'city', 'zoning', 'subdivision', 'join_status', 'sale_warning']
for col in categorical_features:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


drop_cols = ['id', 'sale_date', 'sale_price', 'sale_price_log']
X = train.drop(columns=drop_cols)
y = train['sale_price_log']
X_test = test.drop(columns=['id', 'sale_date'])



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



model_lower = LGBMRegressor(
    objective='quantile',
    alpha=0.05,
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=20,
    verbose=-1
)

model_lower.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='quantile',
    callbacks=[early_stopping(100), log_evaluation(100)]
)



model_upper = LGBMRegressor(
    objective='quantile',
    alpha=0.95,
    n_estimators=2000,
    learning_rate=0.01,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=20,
    verbose=-1
)

model_upper.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='quantile',
    callbacks=[early_stopping(100), log_evaluation(100)]
)



y_val_lower = np.expm1(model_lower.predict(X_val))
y_val_upper = np.expm1(model_upper.predict(X_val))
y_val_true = np.expm1(y_val)

y_val_lower = np.minimum(y_val_lower, y_val_upper)

def winkler_score(y_true, upper, lower, alpha=0.1):
    score = upper - lower
    below = y_true < lower
    above = y_true > upper
    score += (2 / alpha) * (lower - y_true) * below
    score += (2 / alpha) * (y_true - upper) * above
    return np.mean(score)

print(f"Local Winkler Score: {winkler_score(y_val_true, y_val_upper, y_val_lower):.2f}")



print(X_train.shape[1])
print(X_test.shape[1])


pi_lower_test = np.expm1(model_lower.predict(X_test))
pi_upper_test = np.expm1(model_upper.predict(X_test))
pi_lower_test = np.minimum(pi_lower_test, pi_upper_test)



submission = pd.DataFrame({
    'id': test['id'],
    'pi_lower': pi_lower_test,
    'pi_upper': pi_upper_test
})

submission.to_csv('submission.csv', index=False)
print("✅ Submission file created successfully.")





