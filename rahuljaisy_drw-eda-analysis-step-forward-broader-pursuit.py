import sys  
import random  
import numpy as np  
import pandas as pd  
from pathlib import Path  

# Version check for reproducibility  
assert sys.version_info >= (3, 8), "Python>=3.8 required"  
print(f"Python {sys.version_info.major}.{sys.version_info.minor}")  

# Seed for reproducibility  
SEED = 42  
random.seed(SEED)  
np.random.seed(SEED)  

# File paths  
data_dir = Path('/kaggle/input/drw-crypto-market-prediction')  
train_path = data_dir / 'train.parquet'  
test_path = data_dir / 'test.parquet'  

# Load a sample of the dataset to save memory  
df_full = pd.read_parquet(train_path)  
df = df_full.sample(n=5000, random_state=SEED).reset_index(drop=True)  # Sample 5000 rows  
df_test = pd.read_parquet(test_path)  
print(f"Original Train shape: {df_full.shape}, Sampled Train shape: {df.shape}, Test shape: {df_test.shape}")  

# Convert float64 columns to float32 for memory efficiency  
float_cols = df.select_dtypes(include=['float64']).columns  
df[float_cols] = df[float_cols].astype('float32')  

df.head(3)


# Basic data quality check  
def check_df(d):  
    print("-- Shape --", d.shape)  
    print("-- Missing values --", d.isna().sum().sum())  
    infs = np.isinf(d.select_dtypes('number')).sum().sum()  
    print("-- Infinities --", infs)  
    consts = [c for c in d if d[c].nunique() == 1]  
    print(f"-- Constant features ({len(consts)}) --", consts[:5])  
  
check_df(df)


# Handle missing values  
for col in df.columns:  
    if df[col].dtype == 'object':  
        df[col].fillna(df[col].mode()[0], inplace=True)  
    else:  
        df[col].fillna(df[col].median(), inplace=True)  
  
# Handle infinite values  
df.replace([np.inf, -np.inf], np.nan, inplace=True)  
df.fillna(df.max() * 1.1, inplace=True)  
  
print("Missing values after cleaning:", df.isna().sum().sum())  
print("Infinite values after cleaning:", np.isinf(df.select_dtypes('number')).sum().sum())


cat_cols = df.select_dtypes(include=['object']).columns  
for col in cat_cols:  
    print(f"Frequency table for {col}:")  
    print(df[col].value_counts(normalize=True))  
    print()


quant_cols = df.select_dtypes(include=['number']).columns  
summary_stats = df[quant_cols].describe().T  
summary_stats['skew'] = df[quant_cols].skew()  
summary_stats['kurtosis'] = df[quant_cols].kurtosis()  
summary_stats


import matplotlib.pyplot as plt  
import scipy.stats as st  

# Histograms (limit to first 6 numeric columns to save memory)  
for col in quant_cols[:6]:  
    plt.figure(figsize=(6, 4))  
    plt.hist(df[col], bins=20, edgecolor='black')  
    plt.title(f'Histogram of {col}')  
    plt.xlabel(col)  
    plt.ylabel('Frequency')  
    plt.show()


# Boxplots (limit to first 6 numeric columns)  
for col in quant_cols[:6]:  
    plt.figure(figsize=(6, 4))  
    plt.boxplot(df[col].values, vert=False)  
    plt.title(f'Boxplot of {col}')  
    plt.show()


# QQ-plots (limit to first 6 numeric columns)  
for col in quant_cols[:6]:  
    plt.figure(figsize=(6, 4))  
    st.probplot(df[col], dist="norm", plot=plt)  
    plt.title(f'QQ-plot of {col}')  
    plt.show()


# Categorical-Categorical relationships  
for col1 in cat_cols:  
    for col2 in cat_cols:  
        if col1 != col2:  
            print(f"Cross-tabulation for {col1} and {col2}:")  
            print(pd.crosstab(df[col1], df[col2], normalize='index'))  
            print()


# Categorical-Quantitative pairs  
for cat_col in cat_cols:  
    for quant_col in quant_cols[:6]:  # limit to first 6 for memory  
        print(f"Group summary for {cat_col} and {quant_col}:")  
        print(df.groupby(cat_col)[quant_col].agg(['mean', 'std']))  
        print()


# Correlation matrix (only numeric)  
corr_matrix = df[quant_cols].corr()  
print("Correlation matrix:")  
print(corr_matrix)  

# Strong correlations  
strong_corrs = corr_matrix[(corr_matrix > 0.7) | (corr_matrix < -0.7)]  
print("Strong correlations (|r| > 0.7):")  
print(strong_corrs)


# Scatterplots (limit to first 5 numeric columns)  
for i, col1 in enumerate(quant_cols[:5]):  
    for col2 in quant_cols[:5]:  
        if col1 != col2:  
            plt.figure(figsize=(6, 4))  
            plt.scatter(df[col1], df[col2], s=4, alpha=0.3)  
            plt.title(f'{col1} vs {col2}')  
            plt.xlabel(col1)  
            plt.ylabel(col2)  
            plt.show()


# Side-by-side boxplots (limit to first 5 numeric columns)  
for cat_col in cat_cols:  
    for quant_col in quant_cols[:5]:  
        plt.figure(figsize=(6, 4))  
        df.boxplot(column=quant_col, by=cat_col)  
        plt.title(f'{quant_col} by {cat_col}')  
        plt.suptitle('')  
        plt.show()


# Pairplot (very limited columns to save memory)  
import seaborn as sns  
sns.pairplot(df[quant_cols[:4]], diag_kind='kde')  
plt.show()


# Define imbalance (assuming these columns exist)  
df['imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-6)

# Histogram of imbalance  
plt.figure(figsize=(6, 4))  
plt.hist(df['imbalance'], bins=100, edgecolor='black')  
plt.title('Order Book Imbalance')  
plt.xlabel('Imbalance')  
plt.ylabel('Frequency')  
plt.show()

# Scatter with target  
plt.figure(figsize=(6, 4))  
plt.scatter(df['imbalance'], df['label'], s=4, alpha=0.3)  
plt.title('Imbalance vs Label')  
plt.xlabel('Imbalance')  
plt.ylabel('Label')  
plt.show()


# Outlier removal (using 99th percentile)  
df_no_outliers = df[(df['bid_qty'] < df['bid_qty'].quantile(0.99)) & (df['ask_qty'] < df['ask_qty'].quantile(0.99))]  
summary_stats_no_outliers = df_no_outliers[quant_cols].describe().T  
summary_stats_no_outliers


# Missingness visualization  
missingness = df.isnull().sum()  
plt.figure(figsize=(8, 4))  
missingness.plot(kind='bar')  
plt.title('Missing Values by Feature')  
plt.ylabel('Count')  
plt.show()

