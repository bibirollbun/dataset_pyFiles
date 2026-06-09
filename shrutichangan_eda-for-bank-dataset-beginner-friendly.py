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


plt.style.use('ggplot')


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df.head()



print("Shape of dataset:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic statistics:\n", df.describe())


target_col = "y"  #  your actual target column

print("\nTarget distribution:\n", df[target_col].value_counts())

plt.figure(figsize=(6,4))
df[target_col].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Target Variable Distribution")
plt.xlabel("Target Classes")
plt.ylabel("Count")
plt.show()


num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols.remove(target_col)  # remove target if numerical

for col in num_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


cat_cols = df.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    plt.figure(figsize=(6,4))
    df[col].value_counts().head(10).plot(kind='bar', color='orange')
    plt.title(f"Top Categories in {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


plt.figure(figsize=(10,6))
corr = df[num_cols + [target_col]].corr()
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()


