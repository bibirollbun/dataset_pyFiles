import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

import optuna

import warnings
warnings.filterwarnings("ignore")



df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
display(df_train.head())
print("Rows: ",df_train.shape[0], "Columns: ", df_train.shape[1])


df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
display(df_test.head())
print("Rows: ",df_test.shape[0], "Columns: ", df_train.shape[1])


df_submission = pd.read_csv("//kaggle/input/playground-series-s5e6/sample_submission.csv")
display(df_submission.head())
print("Rows: ",df_submission.shape[0], "Columns: ", df_submission.shape[1])


df_train.info()


df_train.info()


df_train.describe()


df_test.describe()


df_train.isnull().sum()


df_test.isnull().sum()


df_train.dtypes


df_test.dtypes


df_train.duplicated().sum()


df_test.duplicated().sum()


df_train.columns


df_test.columns


df_train['Soil Type'].unique()


df_train['Crop Type'].unique()


df_test['Soil Type'].unique()


df_test['Crop Type'].unique()


df_train['Soil Type'].value_counts()


df_train['Crop Type'].value_counts()


df_test['Soil Type'].value_counts()


df_test['Crop Type'].value_counts()


df_train_encoded = pd.get_dummies(df_train, columns = ['Soil Type', 'Crop Type'])
df_test_encoded = pd.get_dummies(df_test, columns = ['Soil Type', 'Crop Type'])


display(df_train_encoded.head())
print("Rows: ", df_train_encoded.shape[0], "\nColumns: ", df_train_encoded.shape[1])


display(df_test_encoded.head())
print("Rows: ", df_test_encoded.shape[0], "\nColumns: ", df_test_encoded.shape[1])


le = LabelEncoder()


df_train_encoded.columns


df_train_encoded['Fertilizer Name'].value_counts()


df_train_encoded['Fertilizer Name'].unique()


df_train_encoded['Fertilizer_Name'] = le.fit_transform(df_train['Fertilizer Name'])


df_train_encoded = df_train_encoded.drop(['Fertilizer Name'], axis = 1)


display(df_train_encoded.head())
print("Rows: ", df_train_encoded.shape[0], "\nColumns: ", df_train_encoded.shape[1])

