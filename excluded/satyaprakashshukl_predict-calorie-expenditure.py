import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


# To check first few rows in the dataset 
df_train.head()


# To check last row in the dataset
df_train.tail()


# To check for null values in the data
df_train.isnull().sum()


# to check columns present in the dataset
df_train.columns


# to check datatypes in dataframe
df_train.dtypes


# to check for more in detail, info about dataframe
df_train.info()


# if any null found in dataframe use 

df_train.dropna()


# to fill any missing value using simple mean, median or mode

# df_train =df_train.fillna(df_train.mean()) This is time consuming process better to go with specific cplumn approach if any
# columns has nan or missing values 


# Here it check for variables which are null, create a boolean using df_train.isnull()
for i in df_train.columns[df_train.isnull().any(axis=0)]: #axis= 0 for row othersie for columns
    df_train[i].fillna(df_train[i].mean(),inplace=True)

# Best practise usally one can tryfirst split into train and test, then replace NA by mean on train and then apply this stateful preprocessing model to test
# Posible leakage can be the reason for the above to work out
# this process take less time to execute in comparison to approach above shown
# refrenced form here https://stackoverflow.com/questions/18689823/pandas-dataframe-replace-nan-values-with-average-of-columns


# to check statistics of dataframe

df_train.describe()


df_train.drop_duplicates()


df_train['Sex'].value_counts() # to check what is the distribution of values for any columns


df_train.shape,df_test.shape,df_sub.shape


# How to drop uncessary columns like id

df_train = df_train.drop('id',axis=1)
df_train.head()


# how to check for numerical and categroical column in the dataset

object_col = df_train.select_dtypes(include=['object']).dtypes
object_col


numerical_col = df_train.select_dtypes(include=['int','float']).dtypes
numerical_col


df_train.select_dtypes(include=['object'])


df_train.select_dtypes(include=['int','float'])


# Some filtering operation to check on data 
df_great_70 = df_train[(df_train['Age']>70) & (df_train['Age']>75)]


df_great_70


df_train.columns


df_train['Weight'].value_counts()


df_train[(df_train['Weight']>80) & (df_train['Weight']>90)]


df_train[(df_train['Height']>185) & (df_train['Height']>195)]


df_train[(df_train['Heart_Rate']>95) & (df_train['Heart_Rate']>105)]


df_mod = df_train.copy()


df_mod.head()


df_mod  = df_mod.sort_values(by='Age',ascending=True)
df_mod.head(100)


df_mod = df_mod.sort_values(by='Heart_Rate',ascending=False)
df_mod.head(100)


df_train.corr()




