import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv('/kaggle/input/au-1131-house-prices-prediction/train1121.csv')
df


if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)


y = df['SalePrice']
X = df.drop(columns=['SalePrice'])


categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns


imputer_num = SimpleImputer(strategy='median')  # Fill missing values with median for numerical data
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')  # Fill missing values with most common category
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])


X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_val.shape


model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5

(mae, rmse)



df_test=pd.read_csv('/kaggle/input/au-1131-house-prices-prediction/test1121.csv')
df_test


if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)
else:
    test_ids = None


df_test[numerical_cols] = imputer_num.transform(df_test[numerical_cols])
df_test[categorical_cols] = imputer_cat.transform(df_test[categorical_cols])


df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)


missing_cols = set(X.columns) - set(df_test.columns)
missing_cols


for col in missing_cols:
    df_test[col] = 0


df_test = df_test[X.columns]
df_test


predictions = model.predict(df_test)
predictions


submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

