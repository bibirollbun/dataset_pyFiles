import pandas as pd 
import numpy as np 
import seaborn as sns

df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df.head()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_test


import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)


# understanding the data structure
df.shape


# checking all feature names
df.columns


# basic information
df.info()


# numerical vs categorical columns
df.dtypes.value_counts()


# checking unique values for each columns
df.nunique()


# dropping unnecessary id column
df = df.drop('id', axis=1)
df


# checking missing or null values
df.isnull().sum()


# Basic Statistics
df.describe().T


# EDA based on target Variable y
target_counts = df['y'].value_counts()
print("Class distribution: ", target_counts)
# class imbalance metrics
print("Class ratio: ", target_counts[1]/ target_counts[0])


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize  = (10, 6))
sns.countplot(x=df['y'], data  = df)
plt.show()


# Visualization of data Distribution
num_cols = df.select_dtypes(include = 'number').columns

for col in num_cols:
    plt.figure(figsize = (10, 6))
    sns.histplot(df[col], kde = True)
    plt.title(f'{col} distribution with skewness of {df[col].skew()}')
    plt.show()


# countplot for categorical columns
cat_cols = df.select_dtypes(include = 'object').columns
for col in cat_cols:
    plt.figure(figsize = (16, 8))
    sns.countplot(x = df[col], data = df)
    plt.figure(figsize = (16, 8))
    sns.countplot(x = df[col], data = df, hue = 'y')
    plt.show()
    


# Categorical Feature impace on target
# Target mean plot
for col in cat_cols:
    target_rate = df.groupby(col)['y'].mean().sort_values(ascending=False)
    plt.figure(figsize = (10, 4))
    target_rate.plot(kind = 'bar')
    plt.title(f"Mean Target By {col}")
    plt.show()

# it shows which category levels are most associated with calss 1
    


# Visualization of outliers in dataFrame

for col in num_cols:
    if col!='y':
        plt.figure(figsize = (10, 6))
        sns.boxplot(x = df[col], data = df)
        plt.show()


# Visualization of outliers in dataFrame for target var

for col in num_cols:
    if col!='y':
        plt.figure(figsize = (10, 6))
        sns.boxplot(x = 'y', y = col, data = df)
        plt.show()


# Numerical feature impact on target
from scipy.stats import ttest_ind

for col in num_cols:
    if col!='y':
        class0 = df[df['y']==0][col] 
        class1 = df[df['y']==1][col]
        stat, p = ttest_ind(class0, class1)
        print(f"{col}: p-value = {p}")

# to see if differences between classes are significant


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


for col in num_cols:
    if col != 'y':
        sns.kdeplot(data=df, x=col, hue='y', common_norm=False)
        plt.title(f"{col} by Class")
        plt.show()



# Multicollinearity check using VIF to spot redundant feature

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

X = df[num_cols].drop('y', axis=1)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# High VIF (>10) means the feature is redundant and can cause instability in models like logistic regression


# Variables Corelation
df[num_cols].corr()


# Visualization of Correlation
plt.Figure(figsize = (10, 6))
sns.heatmap(df[num_cols].corr(), annot = True, linewidth = 0.5, fmt = '.2f', cmap = 'coolwarm')


# Interaction effects
# sometimes 2 features together tell more than alone

sns.scatterplot(data  = df, x = 'age', y = 'balance', hue = 'y')


# Pairplot Visualization

selected = ['duration', 'balance', 'y']
sns.pairplot(df[selected], palette='coolwarm')





