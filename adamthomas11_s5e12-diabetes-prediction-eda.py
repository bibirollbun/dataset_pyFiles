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


# Statistical Tests
from scipy.stats import ttest_ind

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.head()


print(f"Rows = {df.shape[0]:,}")
print(f"Features = {df.shape[1] - 1}")
print(f"Null values = {df.isnull().sum().sum()}")
print("\n")
print(f"Attempting to predict: {df.columns[-1]}")


df.drop('id', axis=1).describe(include="all")


df.dtypes


df['diagnosed_diabetes'].value_counts()


num_cols_non_binary = [
    col for col in df.select_dtypes(include=['int', 'float']).columns
    if col != 'id' and df[col].nunique() > 2
]

for col in num_cols_non_binary:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1]})
        
    # overall distribution
    sns.histplot(data=df, x=col, kde=True, ax=axes[0], bins=50)
    axes[0].set_title(f"Distribution of {col}", fontsize=14)
    axes[0].set_ylabel("Count")

    # normalised per group (percentage)
    sns.kdeplot(
        data=df,
        x=col,
        hue='diagnosed_diabetes',
        common_norm=False,  
        fill=True,        
        alpha=0.5,       
        ax=axes[1]
    )
    axes[1].set_title(f"{col} by diagnosed_diabetes (Normalised Density)", fontsize=14)
    axes[1].set_ylabel("Density")
    axes[1].set_xlabel(col)

    plt.tight_layout()
    plt.show()


for i in num_cols_non_binary:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.boxplot(
        data=df,
        x='diagnosed_diabetes',
        y=i,
        palette='Set1',
        ax=axes[0]
    )
    axes[0].set_title(f"{i} by diagnosed_diabetes")
    axes[0].set_xlabel("Diabetes")
    axes[0].set_ylabel(i)

    # Descriptive text
     # compute stats
    mean0 = df.loc[df.diagnosed_diabetes == 0, i].mean()
    mean1 = df.loc[df.diagnosed_diabetes == 1, i].mean()
    diff = mean1 - mean0
    
    group0 = df.loc[df['diagnosed_diabetes'] == 0, i]
    group1 = df.loc[df['diagnosed_diabetes'] == 1, i]
    
    tstat, pval = ttest_ind(group0, group1, equal_var=False)
    
    # Cohen's d
    pooled_sd = ((group0.std()**2 + group1.std()**2) / 2)**0.5
    d = diff / pooled_sd
    
    # effect size interpretation
    if abs(d) < 0.2:
        effect_text = "Very small effect"
    elif abs(d) < 0.5:
        effect_text = "Small effect"
    elif abs(d) < 0.8:
        effect_text = "Medium effect"
    else:
        effect_text = "Large effect"
    
    interpretation = f"{effect_text} (d={d:.2f})."
    
    # final text block
    text = (
        f"Mean (No): {mean0:.2f}\n"
        f"Mean (Yes): {mean1:.2f}\n"
        f"Difference: {diff:.2f}\n"
        f"T-test: t={tstat:.2f}, p={pval:.2g}\n"
        f"Cohen's d: {d:.2f}\n\n"
        f"Interpretation: {interpretation}"
    )

    axes[1].text(0.5, 0.5, text, fontsize=14, ha='center', va='center')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()



corr = df[num_cols_non_binary].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,          
    fmt=".2f",           
    cmap="coolwarm",      
    cbar=False,            
    square=True,        
    linewidths=0.5
)
plt.title("Correlation Heatmap of Numerical Features", fontsize=16, loc='left')
plt.show()


cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
binary_num_cols = [
    col for col in df.columns
    if df[col].dtype in ['int64', 'float64']
    and df[col].nunique() == 2
    and col != 'diagnosed_diabetes'
]

df[binary_num_cols] = df[binary_num_cols].astype('category')

categorical_cols = cat_cols + binary_num_cols

for col in df[categorical_cols]:
    print('\n')
    print(df[col].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))


for col in categorical_cols:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1]})

    # overall distribution
    order = df[col].value_counts().index 
    sns.countplot(data=df, x=col, order=order, ax=axes[0], palette="Set1")
    axes[0].set_title(f"Distribution of {col}")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)

    # distribution by diagnosed_diabetes (normalised per class)
    sns.histplot(
        data=df,
        x=col,
        hue='diagnosed_diabetes',
        multiple='fill',      
        stat='percent', 
        shrink=0.8,
        palette="Set2",
        ax=axes[1]
    )
    axes[1].set_title(f"{col} by diagnosed_diabetes (Percentage)")
    axes[1].set_ylabel("Percentage")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

