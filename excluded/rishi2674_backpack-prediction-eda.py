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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_df.head()


train_df.shape


train_df.info()


train_df.isnull().sum()


train_df.describe().T


train_df.dtypes


train_df.columns


train_df.drop(["id"],axis=1,inplace=True)
train_df.head()


categorical_cols = ['Brand', 'Material', 'Size','Laptop Compartment',
       'Waterproof', 'Style', 'Color']
for col in categorical_cols:
    # print(col)
    print(train_df[col].value_counts())


numerical_cols = [col for col in train_df.columns if col not in categorical_cols]

# Plot histograms for all numerical columns
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, len(numerical_cols) // 2 + 1, i)
    sns.histplot(train_df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


# print(train_df.isnull())
sns.heatmap(train_df.isnull(),cmap='flare')


for cols in categorical_cols:
    for col in numerical_cols:
        print("Grouped by:",cols," finding mean of :",col)
        print(train_df.groupby(cols)[col].mean())


numerical_cols = numerical_cols[:2]
for col in numerical_cols:
    train_df[col].fillna(train_df[col].median(),inplace=True)


for col in categorical_cols:
    # Get unique non-null values
    unique_values = train_df[col].dropna().unique()
    
    # Fill missing values with random choices from existing values
    train_df[col] = train_df[col].fillna(np.random.choice(unique_values))


train_df.isnull().sum()




