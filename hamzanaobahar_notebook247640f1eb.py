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


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.head()


df.info()


df.shape


import seaborn as sns
import matplotlib.pyplot as plt


categorical_cols = [
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status'
]


fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(10, 18))

for ax, col in zip(axes, categorical_cols):
    counts = df[col].value_counts()
    ax.bar(counts.index, counts.values)
    ax.set_title(col)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


df.describe()


from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
)


ohe = OneHotEncoder()
oe = OrdinalEncoder()


from sklearn.compose import ColumnTransformer
categorical_transformer = ColumnTransformer(
    transformers=[
        ('ord', oe, ['education_level', 'income_level']),
        ('ohe', ohe, [
            'gender',
            'ethnicity',
            'smoking_status',
            'employment_status'
        ])
    ],
    remainder='drop'
)


