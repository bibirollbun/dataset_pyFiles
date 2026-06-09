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


# =====================================
# 1. Setup
# =====================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

#test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")  
print("Shape of dataset:", df.shape)

# Quick look
df.head()


# =====================================
# 2. Basic Info & Data Quality
# =====================================
df.info()
df.describe(include='all')

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar')
    plt.title("Missing Values by Feature")
    plt.show()
else:
    print("âœ… No missing values found.")

# Duplicates
print("Duplicate rows:", df.duplicated().sum())


# =====================================
# 3. Target Distribution
# =====================================
target_col = "y"  
if target_col in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title("Target Class Distribution")
    plt.show()
    print(df[target_col].value_counts(normalize=True))



# =====================================
# 4. Numerical Feature Analysis
# =====================================
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target_col, errors='ignore')

# Histograms
df[num_cols].hist(figsize=(15, 12), bins=30, edgecolor='black')
plt.suptitle("Numerical Feature Distributions")
plt.show()

# Boxplots vs Target
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target_col, y=col, data=df)
    plt.title(f"{col} vs {target_col}")
    plt.show()


# =====================================
# 5. Categorical Feature Analysis
# =====================================
cat_cols = df.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.show()
    
    if target_col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=target_col, hue=col, data=df)
        plt.title(f"{col} vs {target_col}")
        plt.show()


# =====================================
# 6. Correlation Analysis (fixed)
# =====================================
corr = df[num_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", annot=False, cbar=True, linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.show()



# =====================================
# 7. Outlier Detection
# =====================================
from scipy import stats
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color="skyblue")
    plt.title(f"Outliers in {col}", fontsize=13)
    plt.show()

# Z-score based Outlier Count
outliers = {}
for col in num_cols:
    z_scores = np.abs(stats.zscore(df[col]))
    outliers[col] = (z_scores > 3).sum()

outliers = pd.Series(outliers).sort_values(ascending=False)
print("Outlier counts per feature:\n", outliers)



# =====================================
# 8. Class Imbalance
# =====================================
if target_col in df.columns:
    class_counts = df[target_col].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    print("Class Counts:\n", class_counts)
    print("Imbalance Ratio:", imbalance_ratio)

    class_counts.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title="Class Distribution")
    plt.ylabel("")
    plt.show()


