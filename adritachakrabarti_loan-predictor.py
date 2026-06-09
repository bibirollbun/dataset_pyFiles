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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_train.head(10)


val = df_train[df_train["gender"].duplicated()]
print(val)


count = df_train["gender"].value_counts()
print(count)
count2 = df_train["marital_status"].value_counts()
print(count2)
count3 = df_train["education_level"].value_counts()
print(count3)
count4 = df_train["loan_purpose"].value_counts()
print(count4)
count5 = df_train["grade_subgrade"].value_counts()
print(count5)
print(df_train.columns)


mappings = {}
for col in df_train.select_dtypes(include=['object']).columns:
    df_train[col] = df_train[col].astype('category')
    df_train[col + '_encoded'] = df_train[col].cat.codes
    mappings[col] = dict(enumerate(df_train[col].cat.categories))
print("Encoded DataFrame:")
print(df_train)
print("\nEncoding Index per Column:")
for col, mapping in mappings.items():
    print(f"\nColumn: {col}")
    for idx, val in mapping.items():
        print(f"  {idx} → {val}")


df_train.head(10)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_test.head(10)


corr_matrix=df_train.corr()
print(corr_matrix)




