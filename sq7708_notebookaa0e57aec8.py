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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


print(df.head())


print(f"{df.shape[0]}行, {df.shape[1]}列")


print(df.info())


print(df.isnull().sum())


drop=["veil-type","spore-print-color","stem-root","veil-color","stem-surface","gill-spacing"]



print(df.columns)


print(df.isnull().sum())


df['habitat'].value_counts()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')


print(df.head())


print(f"{df.shape[0]}行, {df.shape[1]}列")


print(df.info())


print(df.isnull().sum())





# 数値データの基本統計量
print(df.describe())


# 文字データの基本統計量
print(df.describe(include='object'))


#欠損値が多いの特徴量の削除
drop = ['veil-type','spore-print-color','stem-root','veil-color','stem-surface','gill-spacing']
df = df.drop(drop, axis=1)


print(df.columns)


print(df.isnull().sum())


df['habitat'].value_counts()


df['habitat'].unique()


valid_habitats = [
    'd', 'g', 'l', 'm', 'h', 'w', 'p', 'u', 'e', 's', 
    'n', 't', 'r', 'y', 'a', 'k', 'c', 'b', 'o', 'f', 
    'i', 'x', 'z'
]
df = df[df['habitat'].isin(valid_habitats)]


df['habitat'].value_counts()



df['cap-surface'].value_counts()


# 文字列で、かつ長さが1より大きい行を特定
# 文字列にすべて変更して1文字
condition = df['cap-surface'].astype(str).str.len() > 1
df = df[~condition]
df['cap-surface'].value_counts()


print(df.isnull().sum())


df['cap-shape'].value_counts()


# 文字列で、かつ長さが1より大きい行を特定
# 文字列にすべて変更して1文字
condition = df['cap-shape'].astype(str).str.len() > 1
df = df[~condition]
df['cap-shape'].value_counts()


df['does-bruise-or-bleed'].value_counts()


condition = df['does-bruise-or-bleed'].astype(str).str.len() > 1
df = df[~condition]
df['does-bruise-or-bleed'].value_counts()


df['gill-attachment'].value_counts()


condition = df['gill-attachment'].astype(str).str.len() > 1
df = df[~condition]
df['gill-attachment'].value_counts()


condition = df['gill-attachment'].astype(str).str.len() > 1
df = df[~condition]
df['gill-attachment'].value_counts()


condition = df['gill-color'].astype(str).str.len() > 1
df = df[~condition]
df['gill-color'].value_counts()


values_to_delete = ['4', '5']
condition = df['gill-color'].isin(values_to_delete)
df = df[~condition]
df['gill-color'].value_counts()


df['stem-color'].value_counts()


condition = df['stem-color'].astype(str).str.len() > 1
df = df[~condition]
df['stem-color'].value_counts()


df['has-ring'].value_counts()


df['ring-type'].value_counts()


values_to_delete = ['4', '1']
condition = df['ring-type'].isin(values_to_delete)
df = df[~condition]
df['ring-type'].value_counts()


condition = df['ring-type'].astype(str).str.len() > 1
df = df[~condition]
df['ring-type'].value_counts()


df['season'].value_counts()


print(df.isnull().sum())




