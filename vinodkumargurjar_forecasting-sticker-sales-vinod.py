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


df_train=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


df_train.head(5)


df_test.head(5)


sample_submission.head(5)


# Convert 'date' to datetime format
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])


# Create time-based features
df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_train['dayofweek'] = df_train['date'].dt.dayofweek


df_train.head(5)


# Preprocess test set
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['dayofweek'] = df_test['date'].dt.dayofweek


df_test.head(5)


df_train.info()


df_test.info()


df_train.isnull().sum()


df_test.isnull().sum()


len(df_train["id"]),len(df_test["id"])


((df_train.isnull().sum())/(len(df_train["id"])))*100


((df_test.isnull().sum())/(len(df_test["id"])))*100


df_train.dtypes


train_cat_columns=[]
for i in df_train.columns:
    if df_train[i].dtypes=='O':
        train_cat_columns.append(i)
train_cat_columns


test_cat_columns=[]
for i in df_test.columns:
    if df_test[i].dtypes=='O':
        test_cat_columns.append(i)
test_cat_columns


df_train[train_cat_columns].nunique()


df_test[test_cat_columns].nunique()


for column in df_train.columns:
    if df_train[column].dtype == 'object':
        # Fill with mode for object columns
        mode_value = df_train[column].mode()[0]  # Get the mode and take the first one if there are multiple
        df_train[column].fillna(mode_value, inplace=True)
    elif df_train[column].dtype in ['int64', 'float64']:
        # Fill with mean for numeric columns
        mean_value = df_train[column].mean()
        df_train[column].fillna(mean_value, inplace=True)


df_train.head(5)


df_train.isnull().sum()


df_test.isnull().sum()


df_train.drop(columns="id", axis=1, inplace=True)
df_test.drop(columns="id", axis=1, inplace=True)


df_train.head(5)


# Encode categorical features
from sklearn.preprocessing import LabelEncoder
for col in train_cat_columns:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])


df_train.head(5)


for col in test_cat_columns:
    df_test[col] = le.fit_transform(df_test[col])


df_test.head(5)


# Split train-test data
from sklearn.model_selection import train_test_split
X = df_train.drop(columns=['num_sold', 'date'])  # Features
y = df_train['num_sold']  # Target variable

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xgb
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error

# Train the model
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1,
                         max_depth=6, objective="reg:absoluteerror", random_state=42)
model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, y_pred))

mse=mean_squared_error(y_val,y_pred)
rmse=mse**0.5
print(f"Root Mean Squared Error (RMSE) on Validation Set: {rmse}")

mean_absolute_percentage_error=mean_absolute_percentage_error(y_val,y_pred)
print(f"mean_absolute_percentage_error on Validation Set: {mean_absolute_percentage_error}")




df_test = df_test.drop(columns=['date'])


forecasting = model.predict(df_test)


forecasting


sample_submission.head(5)


sample_submission["num_sold"] = forecasting

sample_submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")


sample_submission.head(5)




