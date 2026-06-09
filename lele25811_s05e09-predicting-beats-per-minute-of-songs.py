import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train_df


train_df.isnull().sum()


test_df.isnull().sum()


train_df.dtypes


test_df.dtypes


col = ['id', 'BeatsPerMinute']

y = train_df['BeatsPerMinute']
train_df = train_df.drop(columns=col, axis=0)


X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size=0.2)


xgbr = XGBRegressor()
xgbr.fit(X_train, y_train)


prediction = xgbr.predict(X_val)


mean_squared_error(y_val, prediction)


test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)


submission_prediction = xgbr.predict(test_df)


submission = pd.DataFrame({'id': test_ids.values,
                          'BeatsPerMinute': submission_prediction 
                          })
submission.head(5)


submission.to_csv('/kaggle/working/calories_submission.csv', index=False)

