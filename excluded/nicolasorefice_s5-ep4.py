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


import matplotlib.pyplot as plt


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df_train.head()


df_train = df_train.drop(["Podcast_Name","Episode_Title", "id"], axis = 1)
df_train.head()


df_train.info()


df_train = df_train.ffill()
df_train = df_train.bfill()


df_train.isna().sum()


name_columns = df_train.columns
name_columns


df_numeric = df_train.select_dtypes(exclude=['object', 'string'])
df_numeric.columns


import seaborn as sns

matrice_corr = df_numeric.corr()

sns.heatmap(matrice_corr)


df_train["Genre"].unique()
df_train["pub_day_time"] = df_train["Publication_Day"] +" - " +  df_train["Publication_Time"]
df_train.head()


df_object = df_train.select_dtypes(exclude=["float64"])
df_object.columns


def encoding_value(df, name_column):
    unique_value = df[name_column].unique()
    value_int = 0
    for value in unique_value:
        df[name_column] = df[name_column].replace(value,value_int)
        value_int += 1
    return df

for column in df_object:
    df_train = encoding_value(df_train, column)

df_train.head()


df_train["ads_per_minute"] = df_train["Episode_Length_minutes"]/df_train["Number_of_Ads"]
matrice_corr = df_train.corr()
sns.heatmap(matrice_corr)

