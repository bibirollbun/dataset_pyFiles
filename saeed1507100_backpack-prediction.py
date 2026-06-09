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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


url = "/kaggle/input/playground-series-s5e2/train.csv"
data = pd.read_csv(url)
data.head()


data.info()


data.describe().T


df = data.copy()
categorical_columns = df.select_dtypes(include=["object"]).columns
numeric_columns = df.select_dtypes(include=["number"]).drop(["id"], axis=1).columns

print(f"Categorical: {categorical_columns}\nNumeric: {numeric_columns}")


# Check for duplicates (No duplicates)
df.duplicated().sum()


missing_value_pct = df.isnull().sum() / df.shape[0] * 100
missing_value_pct = missing_value_pct[missing_value_pct>0]

cols_with_missing_values = missing_value_pct.index
missing_value_pct


plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(col)
plt.suptitle("Distribution of Numeric columns")
plt.tight_layout()
plt.show()


plt.figure(figsize = (15, 10))
for i, col in enumerate(numeric_columns):
    plt.subplot(3,3, i+1)
    sns.boxplot(x=df[col])
    plt.title(col)

plt.tight_layout()


df[df.isnull().any(axis=1)]


for col in cols_with_missing_values:
    if col in categorical_columns:
        df[col] = df[col].fillna("unknown")
        
    elif col in numeric_columns:
        print(f"{col}: {df[col].median()}")
        df[col] = df[col].fillna(df[col].median())


df_temp = df[list(categorical_columns)+list(numeric_columns)].copy()
for col in categorical_columns:
    df_temp[col] = LabelEncoder().fit_transform(df_temp[col])

correlation_matrix = df_temp.corr()

plt.figure(figsize=(16,12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.show()




