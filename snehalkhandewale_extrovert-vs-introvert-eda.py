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


data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


data.head()


data.info()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


# Check unique classes in target
print(data["Personality"].value_counts())





# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include='float64').columns
categorical_cols = data.select_dtypes(include='object').drop("Personality", axis=1).columns



# 1. Numerical features vs Target (ANOVA + boxplots)
print("\n--- ANOVA Test (Numerical columns vs Personality) ---")
for col in numerical_cols:
    groups = [group[col].dropna() for name, group in data.groupby("Personality")]
    f_stat, p_val = f_oneway(*groups)
    print(f"{col}: p-value = {p_val:.4f}")
    sns.boxplot(data=data, x="Personality", y=col)
    plt.title(f"{col} vs Personality")
    plt.show()




# 2. Categorical features vs Target (countplots)
print("\n--- Countplot (Categorical columns vs Personality) ---")
for col in categorical_cols:
    sns.countplot(data=data, x=col, hue="Personality")
    plt.title(f"{col} vs Personality")
    plt.xticks(rotation=45)
    plt.show()



numerical_cols = data.select_dtypes(include='float64').columns

# Histogram for each numerical column
data[numerical_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Distributions of Numerical Features")
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 6))
sns.heatmap(data[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Numerical Features)")
plt.show()


