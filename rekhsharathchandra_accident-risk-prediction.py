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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


df_train.shape


df_test.shape


df_train.head()


df_train=df_train.drop("id", axis=1)
df_test=df_test.drop("id", axis=1)


df_train.head()


df_train.isna().sum()


num_cols =  df_train.select_dtypes(include="number").columns.tolist()
cat_cols = df_train.select_dtypes(exclude="number").columns.tolist()
num_cols.remove("accident_risk")

print(cat_cols)
print(num_cols)


for c in cat_cols :
   print(f" {c} : {df_train[c].unique()}")


bool_cols = ["road_signs_present", "public_road","holiday", "school_season"]
for col in bool_cols :
    df_train[col]= df_train[col].astype(int)
    df_test[col]=df_test[col].astype(int)


le = LabelEncoder()

cate_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in cate_cols :
    df_train[col]= le.fit_transform(df_train[col])
    df_test[col]=le.transform(df_test[col])


X_train = df_train.drop('accident_risk', axis =1)
y_train = df_train['accident_risk']
X_test  = df_test


from xgboost import XGBRegressor

xgb = XGBRegressor()
print(" Training ....")
xgb.fit(X_train, y_train)
print("Predicting ....")
y_pred = xgb.predict(X_test)
print("Done! :", y_pred.shape)


from sklearn.metrics import mean_squared_error
train_pred = xgb.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
print("Train RMSE:", train_rmse)


df_sub['accident_risk'] = y_pred


df_sub.head()


df_sub.to_csv('submission.csv', index=False)




