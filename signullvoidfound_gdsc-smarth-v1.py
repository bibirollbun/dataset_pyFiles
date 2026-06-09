# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train=pd.read_csv("../input/recruitment-task-for-gdsc-ml/MiNDAT.csv")
df_test=pd.read_csv("../input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv")



df_train.describe()


df_test.describe()


df_train.head()


df_test.head()


df_train.isnull().sum()


df_test.isnull().sum()


df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

numeric_col = df_train.select_dtypes(include=['int64', 'float64']).columns
categorical_col = df_train.select_dtypes(include=['object', 'category']).columns

numeric_mean = df_train[numeric_col].mean()
categorical_mode = df_train[categorical_col].mode().iloc[0]

df_train[numeric_col] = df_train[numeric_col].fillna(numeric_mean)
df_train[categorical_col] = df_train[categorical_col].fillna(categorical_mode)

common_numeric_col = [col for col in numeric_col if col in df_test.columns]
common_categorical_col = [col for col in categorical_col if col in df_test.columns]

df_test[common_numeric_col] = df_test[common_numeric_col].fillna(numeric_mean[common_numeric_col])
df_test[common_categorical_col] = df_test[common_categorical_col].fillna(categorical_mode[common_categorical_col])

print("Remaining NaNs in training data is :", df_train.isnull().sum())
print("Remaining NaNs in testing data is :", df_test.isnull().sum())



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

x = df_train.drop('CORRUCYSTIC_DENSITY', axis=1)
y = df_train['CORRUCYSTIC_DENSITY']
x = pd.get_dummies(x, drop_first=True)
x_test = pd.get_dummies(df_test.drop('CORRUCYSTIC_DENSITY', axis=1, errors='ignore'), drop_first=True)
x_test = x_test.reindex(columns=x.columns, fill_value=0)
x_train,x_val,y_train,y_val=train_test_split(x, y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_val)

rmse = mean_squared_error(y_val, y_pred, squared=False)
print("The value RMSE:", rmse)


test_pred = model.predict(x_test)  
spec=pd.read_csv("../input/recruitment-task-for-gdsc-ml/SPECIMEN.csv")
spec['CORRUCYSTIC_DENSITY'] = test_pred
spec




    spec.to_csv('submission.csv', index=False)

