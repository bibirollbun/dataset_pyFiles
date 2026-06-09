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


# ============================================================
# COMPLETE EDA - PLAYGROUND SERIES S5E10
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

print("="*70)
print("       EXPLORATORY DATA ANALYSIS - PLAYGROUND SERIES S5E10")
print("="*70)

# ============================================================
# 1. LOAD ALL DATA
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“� 1. LOADING DATA")
print("ğŸ”·"*35)

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"\nğŸ“Š Dataset Shapes:")
print(f"   Train: {train.shape[0]:,} rows Ã— {train.shape[1]} columns")
print(f"   Test: {test.shape[0]:,} rows Ã— {test.shape[1]} columns")
print(f"   Sample Submission: {sample_sub.shape[0]:,} rows Ã— {sample_sub.shape[1]} columns")

# ============================================================
# 2. BASIC INFORMATION - TRAIN
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 2. TRAIN DATASET - BASIC INFO")
print("ğŸ”·"*35)

print(f"\nğŸ“‹ Columns ({len(train.columns)}):")
print(train.columns.tolist())

print(f"\nğŸ“‹ Data Types:")
print(train.dtypes)

print(f"\nğŸ“‹ First 10 Rows:")
print(train.head(10))

print(f"\nğŸ“‹ Last 5 Rows:")
print(train.tail(5))

print(f"\nğŸ“‹ Info:")
train.info()

# ============================================================
# 3. BASIC INFORMATION - TEST
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 3. TEST DATASET - BASIC INFO")
print("ğŸ”·"*35)

print(f"\nğŸ“‹ Columns ({len(test.columns)}):")
print(test.columns.tolist())

print(f"\nğŸ“‹ Data Types:")
print(test.dtypes)

print(f"\nğŸ“‹ First 10 Rows:")
print(test.head(10))

print(f"\nğŸ“‹ Info:")
test.info()

# ============================================================
# 4. SAMPLE SUBMISSION
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 4. SAMPLE SUBMISSION")
print("ğŸ”·"*35)

print(f"\nğŸ“‹ Columns:")
print(sample_sub.columns.tolist())

print(f"\nğŸ“‹ First 10 Rows:")
print(sample_sub.head(10))

print(f"\nğŸ“‹ Data Types:")
print(sample_sub.dtypes)

print(f"\nğŸ“‹ Target Column Stats:")
target_col = [c for c in sample_sub.columns if c != 'id'][0] if len(sample_sub.columns) > 1 else None
if target_col:
    print(f"   Target column: '{target_col}'")
    print(f"   Unique values: {sample_sub[target_col].nunique()}")
    print(sample_sub[target_col].describe())

# ============================================================
# 5. IDENTIFY TARGET VARIABLE
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ�¯ 5. IDENTIFYING TARGET VARIABLE")
print("ğŸ”·"*35)

# Find columns in train but not in test (excluding 'id')
train_cols = set(train.columns)
test_cols = set(test.columns)
target_candidates = train_cols - test_cols - {'id'}

print(f"\nColumns only in train (potential targets): {target_candidates}")

if target_candidates:
    target = list(target_candidates)[0]
    print(f"\nğŸ�¯ Target Variable: '{target}'")
    print(f"   Data type: {train[target].dtype}")
    print(f"   Unique values: {train[target].nunique()}")
    print(f"   Missing values: {train[target].isnull().sum()}")
    
    if train[target].dtype in ['int64', 'float64']:
        print(f"\n   Statistics:")
        print(train[target].describe())
    else:
        print(f"\n   Value Counts:")
        print(train[target].value_counts())
else:
    target = None
    print("Could not identify target variable")

# ============================================================
# 6. STATISTICAL SUMMARY
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“ˆ 6. STATISTICAL SUMMARY")
print("ğŸ”·"*35)

print("\nğŸ“Š TRAIN - Numerical Summary:")
print(train.describe())

print("\nğŸ“Š TRAIN - Categorical Summary:")
print(train.describe(include=['object', 'category']))

print("\nğŸ“Š TEST - Numerical Summary:")
print(test.describe())

# ============================================================
# 7. MISSING VALUES ANALYSIS
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ”� 7. MISSING VALUES ANALYSIS")
print("ğŸ”·"*35)

def analyze_missing(df, name):
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    print(f"\nğŸ“‹ {name} Missing Values:")
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("   âœ… No missing values!")
    
    return missing_df

train_missing = analyze_missing(train, "TRAIN")
test_missing = analyze_missing(test, "TEST")

# Visualize missing values
if len(train_missing) > 0 or len(test_missing) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if len(train_missing) > 0:
        train_missing['Missing %'].head(20).plot(kind='barh', ax=axes[0], color='coral')
        axes[0].set_title('Train - Missing Values %')
        axes[0].set_xlabel('Missing %')
    else:
        axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
        axes[0].set_title('Train - Missing Values')
    
    if len(test_missing) > 0:
        test_missing['Missing %'].head(20).plot(kind='barh', ax=axes[1], color='steelblue')
        axes[1].set_title('Test - Missing Values %')
        axes[1].set_xlabel('Missing %')
    else:
        axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
        axes[1].set_title('Test - Missing Values')
    
    plt.tight_layout()
    plt.show()

# ============================================================
# 8. COLUMN-WISE ANALYSIS
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ”� 8. COLUMN-WISE ANALYSIS")
print("ğŸ”·"*35)

# Separate numeric and categorical columns
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nğŸ“Š Numeric Columns ({len(numeric_cols)}):")
print(numeric_cols)

print(f"\nğŸ“Š Categorical Columns ({len(categorical_cols)}):")
print(categorical_cols)

# Detailed analysis of each column
print("\n" + "-"*60)
print("DETAILED COLUMN ANALYSIS")
print("-"*60)

for col in train.columns:
    print(f"\n{'='*50}")
    print(f"Column: '{col}'")
    print(f"{'='*50}")
    print(f"  Data Type: {train[col].dtype}")
    print(f"  Unique Values: {train[col].nunique()}")
    print(f"  Missing: {train[col].isnull().sum()} ({train[col].isnull().sum()/len(train)*100:.2f}%)")
    
    if train[col].dtype in ['int64', 'float64']:
        print(f"  Min: {train[col].min()}")
        print(f"  Max: {train[col].max()}")
        print(f"  Mean: {train[col].mean():.4f}")
        print(f"  Median: {train[col].median()}")
        print(f"  Std: {train[col].std():.4f}")
        print(f"  Skewness: {train[col].skew():.4f}")
        print(f"  Kurtosis: {train[col].kurtosis():.4f}")
        
        # Check for zeros and negatives
        print(f"  Zeros: {(train[col] == 0).sum()}")
        print(f"  Negatives: {(train[col] < 0).sum()}")
    else:
        print(f"\n  Value Counts (Top 10):")
        print(train[col].value_counts().head(10))

# ============================================================
# 9. TARGET VARIABLE ANALYSIS
# ============================================================
if target:
    print("\n" + "ğŸ”·"*35)
    print(f"ğŸ�¯ 9. TARGET VARIABLE ANALYSIS: '{target}'")
    print("ğŸ”·"*35)
    
    if train[target].dtype in ['int64', 'float64']:
        # Numerical target
        print("\nğŸ“Š Target Statistics:")
        print(train[target].describe())
        
        print(f"\nğŸ“Š Additional Stats:")
        print(f"   Skewness: {train[target].skew():.4f}")
        print(f"   Kurtosis: {train[target].kurtosis():.4f}")
        
        # Visualize target distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram
        axes[0, 0].hist(train[target], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(train[target].mean(), color='red', linestyle='--', label=f'Mean: {train[target].mean():.2f}')
        axes[0, 0].axvline(train[target].median(), color='green', linestyle='--', label=f'Median: {train[target].median():.2f}')
        axes[0, 0].set_xlabel(target)
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Distribution of {target}')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(train[target].dropna())
        axes[0, 1].set_ylabel(target)
        axes[0, 1].set_title(f'Box Plot of {target}')
        
        # Log distribution (if positive)
        if train[target].min() > 0:
            axes[1, 0].hist(np.log1p(train[target]), bins=50, edgecolor='black', alpha=0.7, color='coral')
            axes[1, 0].set_xlabel(f'log1p({target})')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title(f'Log Distribution of {target}')
        else:
            axes[1, 0].hist(train[target], bins=50, edgecolor='black', alpha=0.7, color='coral')
            axes[1, 0].set_title(f'Distribution (no log - has non-positive values)')
        
        # QQ plot
        stats.probplot(train[target].dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'Q-Q Plot of {target}')
        
        plt.tight_layout()
        plt.show()
        
        # Check if classification or regression
        unique_vals = train[target].nunique()
        if unique_vals <= 30:
            print(f"\nğŸ“Š Value Counts (looks like classification with {unique_vals} classes):")
            print(train[target].value_counts())
            
            # Bar plot for classes
            plt.figure(figsize=(10, 5))
            train[target].value_counts().sort_index().plot(kind='bar', color='steelblue', edgecolor='black')
            plt.xlabel(target)
            plt.ylabel('Count')
            plt.title(f'Target Class Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    else:
        # Categorical target
        print("\nğŸ“Š Target Value Counts:")
        print(train[target].value_counts())
        
        plt.figure(figsize=(10, 5))
        train[target].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
        plt.xlabel(target)
        plt.ylabel('Count')
        plt.title(f'Target Distribution: {target}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ============================================================
# 10. NUMERIC FEATURES DISTRIBUTION
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 10. NUMERIC FEATURES DISTRIBUTION")
print("ğŸ”·"*35)

# Remove id and target from numeric columns for visualization
numeric_features = [c for c in numeric_cols if c not in ['id', target]]

if len(numeric_features) > 0:
    # Calculate grid size
    n_features = len(numeric_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_features):
        ax = axes[idx]
        ax.hist(train[col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col}\nSkew: {train[col].skew():.2f}')
    
    # Hide unused subplots
    for idx in range(len(numeric_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Box plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for idx, col in enumerate(numeric_features):
        ax = axes[idx]
        ax.boxplot(train[col].dropna())
        ax.set_ylabel(col)
        ax.set_title(f'{col}')
    
    for idx in range(len(numeric_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Box Plots of Numeric Features', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# ============================================================
# 11. CATEGORICAL FEATURES ANALYSIS
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 11. CATEGORICAL FEATURES ANALYSIS")
print("ğŸ”·"*35)

cat_features = [c for c in categorical_cols if c not in ['id', target]]

if len(cat_features) > 0:
    for col in cat_features:
        print(f"\n{'='*50}")
        print(f"Column: '{col}'")
        print(f"{'='*50}")
        print(f"  Unique values: {train[col].nunique()}")
        print(f"\n  Value Counts:")
        print(train[col].value_counts().head(15))
    
    # Visualize categorical features
    n_features = len(cat_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(cat_features):
        ax = axes[idx]
        value_counts = train[col].value_counts().head(15)
        value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_title(f'{col} (Top 15)')
        ax.tick_params(axis='x', rotation=45)
    
    for idx in range(len(cat_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
else:
    print("No categorical features found.")

# ============================================================
# 12. CORRELATION ANALYSIS
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 12. CORRELATION ANALYSIS")
print("ğŸ”·"*35)

if len(numeric_features) > 1:
    # Correlation matrix
    corr_cols = numeric_features + ([target] if target and target in numeric_cols else [])
    corr_matrix = train[corr_cols].corr()
    
    print("\nğŸ“Š Correlation Matrix:")
    print(corr_matrix)
    
    # Heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', linewidths=0.5, annot_kws={'size': 8})
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Top correlations with target
    if target and target in corr_matrix.columns:
        target_corr = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)
        print(f"\nğŸ“Š Correlations with Target '{target}':")
        print(target_corr)
        
        # Visualize
        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in target_corr.values]
        target_corr.plot(kind='barh', color=colors)
        plt.xlabel('Correlation')
        plt.title(f'Feature Correlations with {target}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.show()

# ============================================================
# 13. FEATURE VS TARGET ANALYSIS
# ============================================================
if target and target in train.columns:
    print("\n" + "ğŸ”·"*35)
    print(f"ğŸ“Š 13. FEATURE VS TARGET ANALYSIS")
    print("ğŸ”·"*35)
    
    # Numeric features vs target
    if len(numeric_features) > 0:
        n_features = min(len(numeric_features), 12)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_features[:n_features]):
            ax = axes[idx]
            ax.scatter(train[col], train[target], alpha=0.3, s=5)
            ax.set_xlabel(col)
            ax.set_ylabel(target)
            ax.set_title(f'{col} vs {target}')
        
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    # Categorical features vs target (if classification)
    if len(cat_features) > 0 and train[target].nunique() <= 20:
        for col in cat_features[:5]:
            plt.figure(figsize=(10, 5))
            if train[target].dtype in ['int64', 'float64']:
                train.groupby(col)[target].mean().sort_values().plot(kind='bar', color='steelblue')
                plt.ylabel(f'Mean {target}')
            else:
                pd.crosstab(train[col], train[target]).plot(kind='bar', stacked=True)
                plt.ylabel('Count')
            plt.xlabel(col)
            plt.title(f'{col} vs {target}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# ============================================================
# 14. TRAIN VS TEST COMPARISON
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 14. TRAIN VS TEST COMPARISON")
print("ğŸ”·"*35)

common_cols = [c for c in train.columns if c in test.columns and c != 'id']
print(f"\nCommon columns: {len(common_cols)}")

# Compare distributions
numeric_common = [c for c in common_cols if train[c].dtype in ['int64', 'float64']]

if len(numeric_common) > 0:
    comparison_stats = []
    for col in numeric_common:
        comparison_stats.append({
            'Column': col,
            'Train Mean': train[col].mean(),
            'Test Mean': test[col].mean(),
            'Train Std': train[col].std(),
            'Test Std': test[col].std(),
            'Train Min': train[col].min(),
            'Test Min': test[col].min(),
            'Train Max': train[col].max(),
            'Test Max': test[col].max(),
        })
    
    comparison_df = pd.DataFrame(comparison_stats)
    print("\nğŸ“Š Train vs Test Statistics:")
    print(comparison_df.to_string())
    
    # Visualize distribution comparison
    n_features = min(len(numeric_common), 8)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_common[:n_features]):
        ax = axes[idx]
        ax.hist(train[col].dropna(), bins=50, alpha=0.5, label='Train', color='blue', density=True)
        ax.hist(test[col].dropna(), bins=50, alpha=0.5, label='Test', color='red', density=True)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.set_title(f'{col}')
        ax.legend()
    
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Train vs Test Distribution Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# ============================================================
# 15. OUTLIER DETECTION
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 15. OUTLIER DETECTION")
print("ğŸ”·"*35)

if len(numeric_features) > 0:
    outlier_stats = []
    
    for col in numeric_features:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((train[col] < lower_bound) | (train[col] > upper_bound)).sum()
        outlier_pct = outliers / len(train) * 100
        
        outlier_stats.append({
            'Column': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outliers': outliers,
            'Outlier %': outlier_pct
        })
    
    outlier_df = pd.DataFrame(outlier_stats)
    outlier_df = outlier_df.sort_values('Outlier %', ascending=False)
    print("\nğŸ“Š Outlier Analysis (IQR Method):")
    print(outlier_df.to_string())
    
    # Visualize outliers
    if outlier_df['Outlier %'].max() > 0:
        plt.figure(figsize=(10, 6))
        plt.barh(outlier_df['Column'], outlier_df['Outlier %'], color='coral')
        plt.xlabel('Outlier %')
        plt.title('Outlier Percentage by Feature')
        plt.tight_layout()
        plt.show()

# ============================================================
# 16. DUPLICATE ANALYSIS
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ”„ 16. DUPLICATE ANALYSIS")
print("ğŸ”·"*35)

train_dupes = train.duplicated().sum()
test_dupes = test.duplicated().sum()

print(f"\nTrain duplicate rows: {train_dupes} ({train_dupes/len(train)*100:.2f}%)")
print(f"Test duplicate rows: {test_dupes} ({test_dupes/len(test)*100:.2f}%)")

# Check ID uniqueness
if 'id' in train.columns:
    print(f"\nTrain ID uniqueness: {train['id'].nunique()} / {len(train)}")
if 'id' in test.columns:
    print(f"Test ID uniqueness: {test['id'].nunique()} / {len(test)}")

# ============================================================
# 17. MEMORY USAGE
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ’¾ 17. MEMORY USAGE")
print("ğŸ”·"*35)

print("\nğŸ“Š Train Memory Usage:")
print(train.memory_usage(deep=True))
print(f"\nTotal: {train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nğŸ“Š Test Memory Usage:")
print(test.memory_usage(deep=True))
print(f"\nTotal: {test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================
# 18. FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("ğŸ“‹ FINAL SUMMARY REPORT")
print("="*70)

# Problem type detection
if target:
    if train[target].dtype in ['int64', 'float64']:
        if train[target].nunique() <= 20:
            problem_type = "Classification"
            n_classes = train[target].nunique()
        else:
            problem_type = "Regression"
            n_classes = "N/A"
    else:
        problem_type = "Classification"
        n_classes = train[target].nunique()
else:
    problem_type = "Unknown"
    n_classes = "Unknown"

print(f"""
â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
â•‘                        DATASET OVERVIEW                               â•‘
â• â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•£
â•‘                                                                       â•‘
â•‘  ğŸ“Š Dataset Sizes:                                                    â•‘
â•‘     â€¢ Train: {train.shape[0]:>10,} rows Ã— {train.shape[1]:>3} columns                      â•‘
â•‘     â€¢ Test: {test.shape[0]:>11,} rows Ã— {test.shape[1]:>3} columns                      â•‘
â•‘     â€¢ Submission: {sample_sub.shape[0]:>6,} rows Ã— {sample_sub.shape[1]:>3} columns                      â•‘
â•‘                                                                       â•‘
â•‘  ğŸ�¯ Problem Type: {problem_type:<15}                                  â•‘
â•‘     â€¢ Target Variable: {target if target else 'Unknown':<20}                       â•‘
â•‘     â€¢ Number of Classes: {str(n_classes):<15}                               â•‘
â•‘                                                                       â•‘
â•‘  ğŸ“Š Features:                                                         â•‘
â•‘     â€¢ Numeric: {len(numeric_cols):>5}                                               â•‘
â•‘     â€¢ Categorical: {len(categorical_cols):>5}                                           â•‘
â•‘                                                                       â•‘
â•‘  ğŸ”� Data Quality:                                                     â•‘
â•‘     â€¢ Train Missing Values: {train.isnull().sum().sum():>8}                             â•‘
â•‘     â€¢ Test Missing Values: {test.isnull().sum().sum():>9}                             â•‘
â•‘     â€¢ Train Duplicates: {train_dupes:>11}                                   â•‘
â•‘     â€¢ Test Duplicates: {test_dupes:>12}                                   â•‘
â•‘                                                                       â•‘
â•‘  ğŸ’¾ Memory:                                                           â•‘
â•‘     â€¢ Train: {train.memory_usage(deep=True).sum()/1024**2:>10.2f} MB                               â•‘
â•‘     â€¢ Test: {test.memory_usage(deep=True).sum()/1024**2:>11.2f} MB                               â•‘
â•‘                                                                       â•‘
â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
""")

# Feature summary table
print("\nğŸ“Š FEATURE SUMMARY:")
print("-"*70)
feature_summary = []
for col in train.columns:
    if col != 'id':
        feature_summary.append({
            'Feature': col,
            'Type': str(train[col].dtype),
            'Unique': train[col].nunique(),
            'Missing': train[col].isnull().sum(),
            'Missing%': f"{train[col].isnull().sum()/len(train)*100:.1f}%"
        })

feature_df = pd.DataFrame(feature_summary)
print(feature_df.to_string(index=False))

print("\n" + "="*70)
print("                    EDA COMPLETED SUCCESSFULLY! âœ…")
print("="*70)


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency

# =============================================================================
# COMPLETE MULTICOLLINEARITY CHECK - PLAYGROUND SERIES S5E10
# =============================================================================

print("="*70)
print("       ğŸ”� COMPLETE MULTICOLLINEARITY ANALYSIS")
print("="*70)

# -----------------------------------------------------------------------------
# 1. VIF ANALYSIS (Numerical Features)
# -----------------------------------------------------------------------------
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 1. VIF ANALYSIS (Numerical Features)")
print("ğŸ”·"*35)

num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
X_num = train[num_cols].copy()

vif_data = pd.DataFrame()
vif_data["Feature"] = X_num.columns
vif_data["VIF"] = [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
vif_data["Status"] = vif_data["VIF"].apply(lambda x: "â�Œ HIGH" if x > 5 else ("âš ï¸� MODERATE" if x > 2 else "âœ… OK"))
vif_data = vif_data.sort_values("VIF", ascending=False)

print(vif_data.to_string(index=False))

# -----------------------------------------------------------------------------
# 2. CORRELATION MATRIX (All Numerical)
# -----------------------------------------------------------------------------
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 2. CORRELATION MATRIX (Numerical Features)")
print("ğŸ”·"*35)

corr_cols = num_cols + ['accident_risk']
corr_matrix = train[corr_cols].corr().round(4)
print(corr_matrix)

print("\nğŸ“ˆ Correlations with Target:")
target_corr = corr_matrix['accident_risk'].drop('accident_risk').sort_values(key=abs, ascending=False)
print(target_corr.to_string())

# -----------------------------------------------------------------------------
# 3. CRAMER'S V (Categorical-Categorical)
# -----------------------------------------------------------------------------
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 3. CRAMER'S V (Categorical Features)")
print("ğŸ”·"*35)

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
cramers_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)

for col1 in cat_cols:
    for col2 in cat_cols:
        cramers_matrix.loc[col1, col2] = cramers_v(train[col1], train[col2])

print("\nCramer's V Matrix (values > 0.3 indicate concern):")
print(cramers_matrix.round(4))

max_cramers = cramers_matrix.where(~np.eye(len(cat_cols), dtype=bool)).max().max()
print(f"\nâš ï¸�  Max Cramer's V (excluding diagonal): {max_cramers:.4f}")
print(f"   Status: {'â�Œ MULTICOLLINEARITY DETECTED' if max_cramers > 0.3 else 'âœ… No multicollinearity'}")

# -----------------------------------------------------------------------------
# 4. BOOLEAN FEATURE CORRELATIONS
# -----------------------------------------------------------------------------
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 4. BOOLEAN FEATURE CORRELATIONS")
print("ğŸ”·"*35)

bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
bool_corr = train[bool_cols].astype(int).corr().round(4)
print(bool_corr)

max_bool_corr = bool_corr.where(~np.eye(len(bool_cols), dtype=bool)).abs().max().max()
print(f"\nâš ï¸�  Max absolute correlation: {max_bool_corr:.4f}")
print(f"   Status: {'â�Œ MULTICOLLINEARITY DETECTED' if max_bool_corr > 0.7 else 'âœ… No multicollinearity'}")

# -----------------------------------------------------------------------------
# 5. CROSS-TYPE ANALYSIS (Numerical vs Categorical encoded)
# -----------------------------------------------------------------------------
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 5. CROSS-TYPE ANALYSIS (Num vs Cat)")
print("ğŸ”·"*35)

train_encoded = pd.get_dummies(train, columns=cat_cols, drop_first=False)
encoded_cat_cols = [col for col in train_encoded.columns if col.startswith(tuple(cat_cols))]

cross_corr_max = 0
high_corr_pairs = []

for num_col in num_cols:
    for cat_col in encoded_cat_cols:
        corr_val = abs(train_encoded[num_col].corr(train_encoded[cat_col]))
        if corr_val > 0.5:
            high_corr_pairs.append((num_col, cat_col, corr_val))
        cross_corr_max = max(cross_corr_max, corr_val)

if high_corr_pairs:
    print("\nâš ï¸�  High cross-correlations detected (> 0.5):")
    for pair in high_corr_pairs:
        print(f"   {pair[0]} â†” {pair[1]}: {pair[2]:.4f}")
else:
    print(f"\nâœ… No high cross-correlations detected.")
    print(f"   Max correlation: {cross_corr_max:.4f}")

# -----------------------------------------------------------------------------
# 6. CONDITION NUMBER
# -----------------------------------------------------------------------------
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 6. CONDITION NUMBER CHECK")
print("ğŸ”·"*35)

X_full = train_encoded.drop(['id', 'accident_risk'], axis=1).astype(float)
condition_number = np.linalg.cond(X_full)

print(f"Condition Number: {condition_number:.2f}")
if condition_number < 30:
    print("Status: âœ… No multicollinearity")
elif condition_number < 100:
    print("Status: âš ï¸� Moderate multicollinearity")
elif condition_number < 1000:
    print("Status: â�Œ Strong multicollinearity")
else:
    print("Status: ğŸ”´ Severe multicollinearity")

# -----------------------------------------------------------------------------
# 7. FINAL SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("                    ğŸ�¯ FINAL MULTICOLLINEARITY SUMMARY")
print("="*70)

summary_data = {
    "Check": [
        "Numerical VIF",
        "Numerical Correlations", 
        "Categorical (Cramer's V)",
        "Boolean Correlations",
        "Cross-Type Correlations",
        "Condition Number"
    ],
    "Max Value": [
        f"{vif_data['VIF'].max():.2f}",
        f"{corr_matrix.where(~np.eye(len(corr_cols), dtype=bool)).abs().max().max():.4f}",
        f"{max_cramers:.4f}",
        f"{max_bool_corr:.4f}",
        f"{cross_corr_max:.4f}",
        f"{condition_number:.0f}"
    ],
    "Status": [
        "âœ… OK" if vif_data['VIF'].max() < 5 else "â�Œ HIGH",
        "âœ… OK" if corr_matrix.where(~np.eye(len(corr_cols), dtype=bool)).abs().max().max() < 0.7 else "â�Œ HIGH",
        "âœ… OK" if max_cramers < 0.3 else "â�Œ HIGH", 
        "âœ… OK" if max_bool_corr < 0.7 else "â�Œ HIGH",
        "âœ… OK" if cross_corr_max < 0.5 else "â�Œ HIGH",
        "âœ… OK" if condition_number < 30 else "â�Œ HIGH"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Overall verdict
all_ok = all([
    vif_data['VIF'].max() < 5,
    corr_matrix.where(~np.eye(len(corr_cols), dtype=bool)).abs().max().max() < 0.7,
    max_cramers < 0.3,
    max_bool_corr < 0.7,
    cross_corr_max < 0.5,
    condition_number < 30
])

print("\n" + "="*70)
if all_ok:
    print("ğŸ�‰ OVERALL VERDICT: NO MULTICOLLINEARITY ISSUES DETECTED!")
    print("   âœ… All features can be used as-is in your model.")
else:
    print("âš ï¸�  OVERALL VERDICT: MULTICOLLINEARITY DETECTED!")
    print("   â�Œ Consider removing or combining highly correlated features.")
print("="*70)

# -----------------------------------------------------------------------------
# 8. FEATURE RECOMMENDATIONS
# -----------------------------------------------------------------------------
print("\n" + "ğŸ”·"*35)
print("ğŸ“‹ FEATURE RECOMMENDATIONS")
print("ğŸ”·"*35)

print("""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚  ğŸ“Œ Based on correlation with target (accident_risk):          â”‚
â”‚                                                                 â”‚
â”‚  Strongest Predictors:                                          â”‚
â”‚     â€¢ curvature (0.54)     â†’ Keep, very important              â”‚
â”‚     â€¢ speed_limit (0.43)   â†’ Keep, very important              â”‚
â”‚     â€¢ num_reported_accidents (0.21) â†’ Keep                     â”‚
â”‚     â€¢ num_lanes (-0.006)   â†’ Weak predictor, but no harm       â”‚
â”‚                                                                 â”‚
â”‚  ğŸ“Œ No features need removal due to multicollinearity          â”‚
â”‚  ğŸ“Œ All categorical features are independent of each other     â”‚
â”‚  ğŸ“Œ All boolean features are independent of each other         â”‚
â”‚                                                                 â”‚
â”‚  ğŸ’¡ Suggestion: Consider feature engineering:                  â”‚
â”‚     - curvature Ã— speed_limit interaction                      â”‚
â”‚     - Group rare categories if any appear after train/test mergeâ”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

# Return key metrics for programmatic use
results = {
    'vif_data': vif_data,
    'corr_matrix': corr_matrix,
    'cramers_matrix': cramers_matrix,
    'bool_corr': bool_corr,
    'condition_number': condition_number,
    'no_multicollinearity': all_ok,
    'target_correlations': target_corr
}

print("\nâœ… Multicollinearity analysis complete!")
results



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("       LIGHTGBM MODEL - PLAYGROUND SERIES S5E10")
print("="*70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“� 1. LOADING DATA")
print("ğŸ”·"*35)

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")
print(f"âœ… Sample submission shape: {sample_submission.shape}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ”§ 2. FEATURE ENGINEERING")
print("ğŸ”·"*35)

def feature_engineering(df):
    """Create new features from existing ones"""
    df = df.copy()
    
    # ---- Interaction Features ----
    # Curvature and speed interaction (dangerous combo)
    df['curvature_speed'] = df['curvature'] * df['speed_limit']
    
    # Curvature per lane (higher = more dangerous)
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1)
    
    # Speed per lane
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)
    
    # Accidents per lane
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    
    # ---- Risk Score Features ----
    # High curvature indicator
    df['high_curvature'] = (df['curvature'] > 0.7).astype(int)
    
    # High speed indicator
    df['high_speed'] = (df['speed_limit'] >= 60).astype(int)
    
    # Dangerous combination: high curvature + high speed
    df['dangerous_combo'] = df['high_curvature'] * df['high_speed']
    
    # ---- Binned Features ----
    df['curvature_bin'] = pd.cut(df['curvature'], bins=5, labels=False)
    df['speed_bin'] = pd.cut(df['speed_limit'], bins=5, labels=False)
    
    # ---- Boolean aggregations ----
    bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    df['safety_score'] = df[bool_cols].sum(axis=1)
    
    # No signs on public road (risky)
    df['public_no_signs'] = ((df['public_road'] == True) & 
                              (df['road_signs_present'] == False)).astype(int)
    
    # Holiday and school season overlap
    df['holiday_school'] = ((df['holiday'] == True) & 
                            (df['school_season'] == True)).astype(int)
    
    # ---- Curvature transformations ----
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_sqrt'] = np.sqrt(df['curvature'])
    df['curvature_log'] = np.log1p(df['curvature'])
    
    # ---- Speed transformations ----
    df['speed_squared'] = df['speed_limit'] ** 2
    df['speed_normalized'] = (df['speed_limit'] - 25) / (70 - 25)
    
    # ---- Accident transformations ----
    df['accidents_log'] = np.log1p(df['num_reported_accidents'])
    df['has_accidents'] = (df['num_reported_accidents'] > 0).astype(int)
    df['multiple_accidents'] = (df['num_reported_accidents'] > 2).astype(int)
    
    return df

# Apply feature engineering
train = feature_engineering(train)
test = feature_engineering(test)

print(f"âœ… Features after engineering:")
print(f"   Train: {train.shape[1]} columns")
print(f"   Test: {test.shape[1]} columns")

# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ�·ï¸� 3. ENCODING CATEGORICAL FEATURES")
print("ğŸ”·"*35)

# Identify categorical columns
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']

# Label encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le
    print(f"   âœ… Encoded '{col}': {le.classes_}")

# Convert boolean to int
for col in boolean_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

print(f"\nâœ… Converted {len(boolean_cols)} boolean columns to int")

# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 4. PREPARING FEATURES AND TARGET")
print("ğŸ”·"*35)

# Target variable
target = 'accident_risk'
y = train[target]

# Features to exclude
exclude_cols = ['id', target]

# Feature columns
feature_cols = [col for col in train.columns if col not in exclude_cols]

X = train[feature_cols]
X_test = test[feature_cols]

print(f"âœ… Number of features: {len(feature_cols)}")
print(f"âœ… Training samples: {len(X)}")
print(f"âœ… Test samples: {len(X_test)}")
print(f"\nğŸ“‹ Features used ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col}")

# ============================================================
# 5. LIGHTGBM MODEL WITH CROSS-VALIDATION
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸš€ 5. TRAINING LIGHTGBM MODEL")
print("ğŸ”·"*35)

# LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

print("\nğŸ“‹ LightGBM Parameters:")
for key, value in lgb_params.items():
    print(f"   â€¢ {key}: {value}")

# Cross-validation setup
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for predictions and scores
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
feature_importance = pd.DataFrame()
fold_scores = []

print(f"\nğŸ”„ Starting {n_folds}-Fold Cross-Validation...")
print("-" * 60)

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\nğŸ“Š Fold {fold}/{n_folds}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Predictions
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Clip predictions to valid range [0, 1]
    val_pred = np.clip(val_pred, 0, 1)
    test_pred = np.clip(test_pred, 0, 1)
    
    # Store predictions
    oof_predictions[val_idx] = val_pred
    test_predictions += test_pred / n_folds
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)
    fold_scores.append(rmse)
    
    print(f"   âœ… RMSE: {rmse:.6f} | MAE: {mae:.6f} | RÂ²: {r2:.6f}")
    print(f"   âœ… Best iteration: {model.best_iteration}")
    
    # Feature importance
    fold_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain'),
        'fold': fold
    })
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

# ============================================================
# 6. CROSS-VALIDATION RESULTS
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 6. CROSS-VALIDATION RESULTS")
print("ğŸ”·"*35)

# Overall OOF metrics
oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
oof_mae = mean_absolute_error(y, oof_predictions)
oof_r2 = r2_score(y, oof_predictions)

print(f"\n{'='*60}")
print(f"ğŸ“ˆ FOLD-WISE RMSE SCORES:")
print(f"{'='*60}")
for i, score in enumerate(fold_scores, 1):
    print(f"   Fold {i}: {score:.6f}")
print(f"{'='*60}")
print(f"   Mean:  {np.mean(fold_scores):.6f}")
print(f"   Std:   {np.std(fold_scores):.6f}")
print(f"{'='*60}")

print(f"\nğŸ“Š OVERALL OOF METRICS:")
print(f"   â€¢ RMSE: {oof_rmse:.6f}")
print(f"   â€¢ MAE:  {oof_mae:.6f}")
print(f"   â€¢ RÂ²:   {oof_r2:.6f}")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 7. FEATURE IMPORTANCE (Top 20)")
print("ğŸ”·"*35)

# Aggregate feature importance
mean_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)

print(f"\n{'Rank':<6}{'Feature':<30}{'Importance':<15}")
print("-" * 55)
for i, (feat, imp) in enumerate(mean_importance.head(20).items(), 1):
    print(f"{i:<6}{feat:<30}{imp:<15.2f}")

# ============================================================
# 8. PREDICTIONS ANALYSIS
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“Š 8. PREDICTIONS ANALYSIS")
print("ğŸ”·"*35)

print("\nğŸ“‹ OOF Predictions Statistics:")
print(f"   â€¢ Min:    {oof_predictions.min():.4f}")
print(f"   â€¢ Max:    {oof_predictions.max():.4f}")
print(f"   â€¢ Mean:   {oof_predictions.mean():.4f}")
print(f"   â€¢ Median: {np.median(oof_predictions):.4f}")
print(f"   â€¢ Std:    {oof_predictions.std():.4f}")

print("\nğŸ“‹ Test Predictions Statistics:")
print(f"   â€¢ Min:    {test_predictions.min():.4f}")
print(f"   â€¢ Max:    {test_predictions.max():.4f}")
print(f"   â€¢ Mean:   {test_predictions.mean():.4f}")
print(f"   â€¢ Median: {np.median(test_predictions):.4f}")
print(f"   â€¢ Std:    {test_predictions.std():.4f}")

print("\nğŸ“‹ Actual Target Statistics:")
print(f"   â€¢ Min:    {y.min():.4f}")
print(f"   â€¢ Max:    {y.max():.4f}")
print(f"   â€¢ Mean:   {y.mean():.4f}")
print(f"   â€¢ Median: {y.median():.4f}")
print(f"   â€¢ Std:    {y.std():.4f}")

# ============================================================
# 9. CREATE SUBMISSION
# ============================================================
print("\n" + "ğŸ”·"*35)
print("ğŸ“¤ 9. CREATING SUBMISSION FILE")
print("ğŸ”·"*35)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': test_predictions
})

# Ensure predictions are within valid range
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)

# Save submission
submission.to_csv('submission_lightgbm.csv', index=False)

print(f"\nâœ… Submission file saved: 'submission_lightgbm.csv'")
print(f"   â€¢ Shape: {submission.shape}")
print(f"\nğŸ“‹ Submission Preview:")
print(submission.head(10))

print(f"\nğŸ“Š Submission Statistics:")
print(submission['accident_risk'].describe())

# ============================================================
# 10. FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("ğŸ“‹ FINAL SUMMARY")
print("="*70)

print(f"""
â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
â•‘                        MODEL SUMMARY                                  â•‘
â• â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•£
â•‘                                                                       â•‘
â•‘  ğŸ�¯ Model: LightGBM Regressor                                        â•‘
â•‘                                                                       â•‘
â•‘  ğŸ“Š Cross-Validation Results ({n_folds} Folds):                             â•‘
â•‘     â€¢ Mean RMSE:  {np.mean(fold_scores):.6f}                                    â•‘
â•‘     â€¢ Std RMSE:   {np.std(fold_scores):.6f}                                    â•‘
â•‘     â€¢ OOF RMSE:   {oof_rmse:.6f}                                    â•‘
â•‘     â€¢ OOF MAE:    {oof_mae:.6f}                                    â•‘
â•‘     â€¢ OOF RÂ²:     {oof_r2:.6f}                                    â•‘
â•‘                                                                       â•‘
â•‘  ğŸ“Š Features:                                                         â•‘
â•‘     â€¢ Total Features: {len(feature_cols):3d}                                       â•‘
â•‘     â€¢ Top Feature: {mean_importance.index[0]:<25}             â•‘
â•‘                                                                       â•‘
â•‘  ğŸ“¤ Submission:                                                       â•‘
â•‘     â€¢ File: submission_lightgbm.csv                                   â•‘
â•‘     â€¢ Rows: {len(submission):,}                                              â•‘
â•‘                                                                       â•‘
â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
""")

print("="*70)
print("                    TRAINING COMPLETED SUCCESSFULLY! âœ…")
print("="*70)




