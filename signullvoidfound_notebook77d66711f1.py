# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import sklearn.preprocessing
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
 # data processing, CSV file I/O (e.g. pd.read_csv)

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


df_train


df_train.describe()


if "LOCAL_IDENTIFIER" in df_train.columns:
    df_train = df_train.drop(columns=["LOCAL_IDENTIFIER"])

for i in df_train.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(6,4))
    
    sns.histplot(df_train[i].dropna(), bins=30, kde=True)
    plt.title(f"Distribution of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")
    
    skewness = df_train[i].skew()
    print(f"Skewness of {i}: {skewness:.3f}")
    
    plt.show()



constant_cols = [col for col in df_train.columns if df_train[col].nunique() == 1]
df_train = df_train.drop(columns=constant_cols)
df_test = df_test.drop(columns=constant_cols)

print("Dropped:", constant_cols)



df_train.head()


df_test.head()


df_test.isnull().sum()


skewness = df_train.skew(numeric_only=True)

print(skewness.sort_values())



for col in df_train.select_dtypes(include=["float64","int64"]).columns:
    if abs(df_train[col].skew()) < 1:
        df_train[col] = df_train[col].fillna(df_train[col].mean())
    else:
        df_train[col] = df_train[col].fillna(df_train[col].median())
df_train.isnull().sum()



for col in df_test.select_dtypes(include=["float64","int64"]).columns:
    if abs(df_test[col].skew()) < 1:
        df_test[col] = df_test[col].fillna(df_test[col].mean())
    else:
        df_test[col] = df_train[col].fillna(df_test[col].median())
df_test.isnull().sum()



cat_fill_values = {}

for j in df_train.select_dtypes(include=["object", "category"]).columns:
    mode_val = df_train[j].mode()[0]    
    cat_fill_values[j] = mode_val
    df_train[j] = df_train[j].fillna(mode_val)
    
for k, value in cat_fill_values.items():
    if k in df_test.columns:  
        df_test[k] = df_test[k].fillna(value)



from sklearn.preprocessing import OneHotEncoder

cat_cols = df_train.select_dtypes(include=["object", "category"]).columns

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

train_ohe = ohe.fit_transform(df_train[cat_cols])
test_ohe  = ohe.transform(df_test[cat_cols])

import pandas as pd
train_ohe_df = pd.DataFrame(train_ohe, columns=ohe.get_feature_names_out(cat_cols), index=df_train.index)
test_ohe_df  = pd.DataFrame(test_ohe,  columns=ohe.get_feature_names_out(cat_cols), index=df_test.index)

df_train_final = df_train.drop(columns=cat_cols).join(train_ohe_df)
df_test_final  = df_test.drop(columns=cat_cols).join(test_ohe_df)



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

