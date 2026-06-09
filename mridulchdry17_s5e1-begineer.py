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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_df


print(f"train data :{train_df.shape} \ntest data :{test_df.shape}")


train_df.describe()


train_df.info()


train_df.isnull().sum()


# since no of null rows are only 8k so we are taking an assumption that we should drop these columns  


train_df.dropna(inplace=True)


train_df.nunique()


train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


train_df = pd.get_dummies(train_df, columns=['country', 'store', 'product'], drop_first=False)
test_df = pd.get_dummies(test_df, columns=['country', 'store', 'product'], drop_first=False)


train_df


train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['is_weekend'] = train_df['day_of_week'].isin([5, 6]).astype(int)


test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.dayofweek
test_df['is_weekend'] = test_df['day_of_week'].isin([5, 6]).astype(int)


train_df


train_df.nunique()


train_df = train_df.sort_values(by='date')


test_df


train_df


X = train_df.drop(columns=['date','num_sold'])
y = train_df['num_sold']


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, shuffle=False)  # Keep shuffle=False for time series


X_val


y


from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.fit(X_train, y_train)



y_pred = model.predict(X_val)


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)


mse,r2


predictions = model.predict(test_df)


submission = test_df[['id']]
submission['num_sold'] = predictions
submission.to_csv('submission_CAT_1.csv', index=False)
































































