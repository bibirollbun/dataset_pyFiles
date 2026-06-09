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


df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df


missing_values = df.isnull().sum()
print(missing_values)


# For numerical columns, fill missing values with median
numerical_cols = ['Weight Capacity (kg)']  # Add other numerical columns if any
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# For categorical columns, fill missing values with mode
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']  # Add other categorical columns if needed
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Verify the changes
print(df.isnull().sum())


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


df


categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
# Label Encoding for categorical columns (you can also use OneHotEncoding for a larger dataset)
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
X = df[['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']]
y = df['Price']


X


# Step 4: Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
df_test


# For numerical columns, fill missing values with median
numerical_cols = ['Weight Capacity (kg)']  # Add other numerical columns if any
df_test[numerical_cols] = df_test[numerical_cols].fillna(df[numerical_cols].median())

# For categorical columns, fill missing values with mode
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']  # Add other categorical columns if needed
for col in categorical_cols:
    df_test[col] = df_test[col].fillna(df_test[col].mode()[0])

# Verify the changes
print(df_test.isnull().sum())


le = LabelEncoder()
for col in categorical_cols:
    df_test[col] = le.fit_transform(df_test[col])
X_test = df_test[['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)']]
X_test


y_pred = model.predict(X_test)
y_pred


sumbission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sumbission


sumbission["Price"] = y_pred
sumbission


sumbission.to_csv("submission.csv", index = False)




