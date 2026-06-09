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


# ============================================
# COMPREHENSIVE EDA - PLAYGROUND SERIES S5E11
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================
# 1. LOAD DATA
# ============================================
print("="*70)
print("ğŸ“� LOADING DATA")
print("="*70)

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")
print(f"âœ… Sample submission shape: {sample_sub.shape}")

# ============================================
# 2. BASIC DATA OVERVIEW
# ============================================
print("\n" + "="*70)
print("ğŸ“Š BASIC DATA OVERVIEW")
print("="*70)

print("\nğŸ“Œ TRAIN DATA - First 5 Rows:")
display(train.head())

print("\nğŸ“Œ TEST DATA - First 5 Rows:")
display(test.head())

print("\nğŸ“Œ SAMPLE SUBMISSION:")
display(sample_sub.head())

print("\nğŸ“Œ TRAIN COLUMNS:")
print(train.columns.tolist())

print("\nğŸ“Œ TEST COLUMNS:")
print(test.columns.tolist())

# Identify target column
target_col = [col for col in train.columns if col not in test.columns and col != 'id']
print(f"\nğŸ�¯ TARGET COLUMN(S): {target_col}")

# ============================================
# 3. DATA TYPES & INFO
# ============================================
print("\n" + "="*70)
print("ğŸ“‹ DATA TYPES & INFO")
print("="*70)

print("\nğŸ“Œ TRAIN DATA INFO:")
print(train.info())

print("\nğŸ“Œ DATA TYPES SUMMARY:")
dtype_df = pd.DataFrame({
    'Column': train.columns,
    'Dtype': train.dtypes.values,
    'Non-Null Count': train.count().values,
    'Null Count': train.isnull().sum().values,
    'Null %': (train.isnull().sum().values / len(train) * 100).round(2),
    'Unique Values': train.nunique().values
})
display(dtype_df)

# Separate numerical and categorical columns
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove id and target from feature lists if present
if 'id' in numerical_cols:
    numerical_cols.remove('id')
if target_col and target_col[0] in numerical_cols:
    numerical_features = [col for col in numerical_cols if col != target_col[0]]
else:
    numerical_features = numerical_cols

print(f"\nğŸ“Š Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"ğŸ“Š Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# ============================================
# 4. MISSING VALUES ANALYSIS
# ============================================
print("\n" + "="*70)
print("â�“ MISSING VALUES ANALYSIS")
print("="*70)

# Train missing values
train_missing = train.isnull().sum()
train_missing_pct = (train_missing / len(train) * 100).round(2)

# Test missing values
test_missing = test.isnull().sum()
test_missing_pct = (test_missing / len(test) * 100).round(2)

missing_df = pd.DataFrame({
    'Train Missing': train_missing,
    'Train Missing %': train_missing_pct,
    'Test Missing': test_missing,
    'Test Missing %': test_missing_pct
})

print("\nğŸ“Œ Missing Values Summary:")
display(missing_df[missing_df['Train Missing'] > 0].sort_values('Train Missing %', ascending=False))

if train_missing.sum() == 0 and test_missing.sum() == 0:
    print("âœ… No missing values in the dataset!")
else:
    # Visualize missing values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if train_missing.sum() > 0:
        train_missing[train_missing > 0].sort_values(ascending=True).plot(
            kind='barh', ax=axes[0], color='coral')
        axes[0].set_title('Train - Missing Values Count')
        axes[0].set_xlabel('Count')
    else:
        axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
        axes[0].set_title('Train - Missing Values')
    
    if test_missing.sum() > 0:
        test_missing[test_missing > 0].sort_values(ascending=True).plot(
            kind='barh', ax=axes[1], color='steelblue')
        axes[1].set_title('Test - Missing Values Count')
        axes[1].set_xlabel('Count')
    else:
        axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
        axes[1].set_title('Test - Missing Values')
    
    plt.tight_layout()
    plt.show()

# ============================================
# 5. STATISTICAL SUMMARY
# ============================================
print("\n" + "="*70)
print("ğŸ“ˆ STATISTICAL SUMMARY")
print("="*70)

print("\nğŸ“Œ TRAIN - Numerical Statistics:")
display(train.describe().T.round(3))

print("\nğŸ“Œ TEST - Numerical Statistics:")
display(test.describe().T.round(3))

if categorical_cols:
    print("\nğŸ“Œ TRAIN - Categorical Statistics:")
    display(train[categorical_cols].describe().T)

# ============================================
# 6. TARGET VARIABLE ANALYSIS
# ============================================
if target_col:
    print("\n" + "="*70)
    print("ğŸ�¯ TARGET VARIABLE ANALYSIS")
    print("="*70)
    
    target = target_col[0]
    
    print(f"\nğŸ“Œ Target: {target}")
    print(f"ğŸ“Œ Dtype: {train[target].dtype}")
    print(f"ğŸ“Œ Unique Values: {train[target].nunique()}")
    
    if train[target].dtype in ['int64', 'float64']:
        print(f"\nğŸ“Œ Target Statistics:")
        print(train[target].describe())
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        
        # Distribution
        axes[0].hist(train[target], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(train[target].mean(), color='red', linestyle='--', label=f'Mean: {train[target].mean():.2f}')
        axes[0].axvline(train[target].median(), color='green', linestyle='--', label=f'Median: {train[target].median():.2f}')
        axes[0].set_title(f'Distribution of {target}')
        axes[0].set_xlabel(target)
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(train[target].dropna())
        axes[1].set_title(f'Box Plot of {target}')
        axes[1].set_ylabel(target)
        
        # Log distribution if positive values
        if train[target].min() > 0:
            axes[2].hist(np.log1p(train[target]), bins=50, color='coral', edgecolor='black', alpha=0.7)
            axes[2].set_title(f'Log Distribution of {target}')
            axes[2].set_xlabel(f'log({target})')
            axes[2].set_ylabel('Frequency')
        else:
            sns.kdeplot(data=train[target], ax=axes[2], fill=True, color='coral')
            axes[2].set_title(f'KDE Plot of {target}')
        
        plt.tight_layout()
        plt.show()
        
        # Skewness and Kurtosis
        print(f"\nğŸ“Œ Skewness: {train[target].skew():.4f}")
        print(f"ğŸ“Œ Kurtosis: {train[target].kurtosis():.4f}")
        
    else:
        # Categorical target
        print(f"\nğŸ“Œ Value Counts:")
        value_counts = train[target].value_counts()
        print(value_counts)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        value_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
        axes[0].set_title(f'Distribution of {target}')
        axes[0].set_xlabel(target)
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        axes[1].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', 
                   colors=plt.cm.Pastel1.colors)
        axes[1].set_title(f'{target} Distribution')
        
        plt.tight_layout()
        plt.show()

# ============================================
# 7. NUMERICAL FEATURES ANALYSIS
# ============================================
print("\n" + "="*70)
print("ğŸ“Š NUMERICAL FEATURES ANALYSIS")
print("="*70)

num_features = [col for col in numerical_cols if col not in ['id'] + target_col]

if num_features:
    print(f"\nğŸ“Œ Number of numerical features: {len(num_features)}")
    
    # Statistics comparison
    stats_comparison = pd.DataFrame({
        'Feature': num_features,
        'Train Mean': [train[col].mean() for col in num_features],
        'Test Mean': [test[col].mean() for col in num_features],
        'Train Std': [train[col].std() for col in num_features],
        'Test Std': [test[col].std() for col in num_features],
        'Train Skew': [train[col].skew() for col in num_features],
        'Train Kurt': [train[col].kurtosis() for col in num_features]
    })
    display(stats_comparison.round(3))
    
    # Distribution plots
    n_cols = 4
    n_rows = (len(num_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(num_features) == 1 else axes.flatten()
    
    for idx, col in enumerate(num_features):
        ax = axes[idx]
        ax.hist(train[col].dropna(), bins=30, alpha=0.5, label='Train', color='blue', density=True)
        ax.hist(test[col].dropna(), bins=30, alpha=0.5, label='Test', color='red', density=True)
        ax.set_title(f'{col}')
        ax.legend()
        ax.set_xlabel('')
    
    # Hide empty subplots
    for idx in range(len(num_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Numerical Features Distribution (Train vs Test)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Box plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(num_features) == 1 else axes.flatten()
    
    for idx, col in enumerate(num_features):
        ax = axes[idx]
        train[[col]].boxplot(ax=ax)
        ax.set_title(f'{col}')
    
    for idx in range(len(num_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Numerical Features Box Plots', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# ============================================
# 8. CATEGORICAL FEATURES ANALYSIS
# ============================================
print("\n" + "="*70)
print("ğŸ“Š CATEGORICAL FEATURES ANALYSIS")
print("="*70)

cat_features = [col for col in categorical_cols if col not in target_col]

if cat_features:
    print(f"\nğŸ“Œ Number of categorical features: {len(cat_features)}")
    
    for col in cat_features:
        print(f"\nğŸ“Œ {col}:")
        print(f"   Unique values: {train[col].nunique()}")
        print(f"   Top 5 values:\n{train[col].value_counts().head()}")
    
    # Plot categorical features
    n_cols = min(3, len(cat_features))
    n_rows = (len(cat_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axes = np.array(axes).flatten() if len(cat_features) > 1 else [axes]
    
    for idx, col in enumerate(cat_features):
        ax = axes[idx]
        value_counts = train[col].value_counts()
        if len(value_counts) > 15:
            value_counts = value_counts.head(15)
        value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'{col}')
        ax.tick_params(axis='x', rotation=45)
    
    for idx in range(len(cat_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Categorical Features Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("âœ… No categorical features found.")

# ============================================
# 9. CORRELATION ANALYSIS
# ============================================
print("\n" + "="*70)
print("ğŸ”— CORRELATION ANALYSIS")
print("="*70)

# Compute correlation matrix for numerical features
corr_features = [col for col in numerical_cols if col != 'id']

if len(corr_features) > 1:
    corr_matrix = train[corr_features].corr()
    
    # Full correlation heatmap
    plt.figure(figsize=(min(16, len(corr_features)), min(14, len(corr_features))))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=len(corr_features) <= 15, 
                fmt='.2f', cmap='coolwarm', center=0, 
                linewidths=0.5, annot_kws={'size': 8})
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Correlation with target
    if target_col:
        target = target_col[0]
        if target in corr_features:
            target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
            
            print(f"\nğŸ“Œ Correlation with {target}:")
            print(target_corr.round(4))
            
            plt.figure(figsize=(10, max(6, len(target_corr) * 0.3)))
            colors = ['green' if x > 0 else 'red' for x in target_corr.values]
            target_corr.plot(kind='barh', color=colors)
            plt.title(f'Feature Correlation with {target}')
            plt.xlabel('Correlation')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            plt.show()
    
    # Highly correlated features
    print("\nğŸ“Œ Highly Correlated Feature Pairs (|corr| > 0.8):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        display(pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False))
    else:
        print("âœ… No highly correlated feature pairs found.")

# ============================================
# 10. FEATURE vs TARGET ANALYSIS
# ============================================
if target_col and num_features:
    print("\n" + "="*70)
    print("ğŸ�¯ FEATURE vs TARGET ANALYSIS")
    print("="*70)
    
    target = target_col[0]
    
    # Select top features by correlation
    if target in train.select_dtypes(include=['int64', 'float64']).columns:
        target_corr = train[num_features + [target]].corr()[target].drop(target)
        top_features = target_corr.abs().sort_values(ascending=False).head(6).index.tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(top_features[:6]):
            ax = axes[idx]
            ax.scatter(train[col], train[target], alpha=0.3, s=10)
            ax.set_xlabel(col)
            ax.set_ylabel(target)
            ax.set_title(f'{col} vs {target}\n(corr: {target_corr[col]:.3f})')
        
        plt.suptitle('Top Features vs Target', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

# ============================================
# 11. OUTLIER ANALYSIS
# ============================================
print("\n" + "="*70)
print("ğŸ”� OUTLIER ANALYSIS (IQR Method)")
print("="*70)

outlier_summary = []

for col in num_features:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)][col]
    outlier_pct = len(outliers) / len(train) * 100
    
    outlier_summary.append({
        'Feature': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outlier Count': len(outliers),
        'Outlier %': outlier_pct
    })

outlier_df = pd.DataFrame(outlier_summary)
display(outlier_df.sort_values('Outlier %', ascending=False).round(3))

# Visualize outlier percentages
if len(num_features) > 0:
    plt.figure(figsize=(12, max(4, len(num_features) * 0.4)))
    outlier_df_sorted = outlier_df.sort_values('Outlier %', ascending=True)
    plt.barh(outlier_df_sorted['Feature'], outlier_df_sorted['Outlier %'], color='coral')
    plt.xlabel('Outlier Percentage (%)')
    plt.title('Outlier Percentage by Feature')
    plt.tight_layout()
    plt.show()

# ============================================
# 12. TRAIN vs TEST DISTRIBUTION COMPARISON
# ============================================
print("\n" + "="*70)
print("ğŸ“Š TRAIN vs TEST DISTRIBUTION COMPARISON")
print("="*70)

common_num_cols = [col for col in num_features if col in test.columns]

if common_num_cols:
    ks_results = []
    
    for col in common_num_cols:
        stat, p_value = stats.ks_2samp(train[col].dropna(), test[col].dropna())
        ks_results.append({
            'Feature': col,
            'KS Statistic': stat,
            'P-Value': p_value,
            'Different Distribution': 'Yes' if p_value < 0.05 else 'No'
        })
    
    ks_df = pd.DataFrame(ks_results).sort_values('P-Value')
    print("\nğŸ“Œ Kolmogorov-Smirnov Test (Train vs Test):")
    display(ks_df.round(4))
    
    # Visualize
    different_dist = ks_df[ks_df['Different Distribution'] == 'Yes']
    if len(different_dist) > 0:
        print(f"\nâš ï¸� Features with significantly different distributions: {len(different_dist)}")
        print(different_dist['Feature'].tolist())

# ============================================
# 13. DATA SUMMARY REPORT
# ============================================
print("\n" + "="*70)
print("ğŸ“‹ FINAL DATA SUMMARY REPORT")
print("="*70)

print(f"""
ğŸ“Œ DATASET OVERVIEW:
   â€¢ Train samples: {len(train):,}
   â€¢ Test samples: {len(test):,}
   â€¢ Total features: {len(train.columns) - 1 - len(target_col)}
   â€¢ Numerical features: {len(num_features)}
   â€¢ Categorical features: {len(cat_features)}
   â€¢ Target column: {target_col[0] if target_col else 'Not identified'}

ğŸ“Œ MISSING VALUES:
   â€¢ Train missing: {train.isnull().sum().sum():,} ({(train.isnull().sum().sum() / train.size * 100):.2f}%)
   â€¢ Test missing: {test.isnull().sum().sum():,} ({(test.isnull().sum().sum() / test.size * 100):.2f}%)

ğŸ“Œ DUPLICATE ROWS:
   â€¢ Train duplicates: {train.duplicated().sum():,}
   â€¢ Test duplicates: {test.duplicated().sum():,}
""")

if target_col:
    target = target_col[0]
    if train[target].dtype in ['int64', 'float64']:
        print(f"""ğŸ“Œ TARGET STATISTICS ({target}):
   â€¢ Mean: {train[target].mean():.4f}
   â€¢ Median: {train[target].median():.4f}
   â€¢ Std: {train[target].std():.4f}
   â€¢ Min: {train[target].min():.4f}
   â€¢ Max: {train[target].max():.4f}
   â€¢ Skewness: {train[target].skew():.4f}
""")
    else:
        print(f"""ğŸ“Œ TARGET DISTRIBUTION ({target}):
{train[target].value_counts().to_string()}
""")

print("="*70)
print("âœ… EDA COMPLETE!")
print("="*70)


import os

# For Kaggle
if os.path.exists('/kaggle/input/'):
    print("ğŸ“‚ Kaggle Input Directory:")
    for folder in os.listdir('/kaggle/input/'):
        folder_path = f'/kaggle/input/{folder}'
        print(f"\n  ğŸ“� {folder}/")
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path)[:10]:
                print(f"      - {file}")

# For Google Colab
if os.path.exists('/content/'):
    print("\nğŸ“‚ Colab Content Directory:")
    for item in os.listdir('/content/'):
        print(f"  - {item}")


# UPDATE THESE PATHS BASED ON YOUR ENVIRONMENT:
train_file = '/kaggle/input/YOUR-COMPETITION-NAME/train.csv'
test_file = '/kaggle/input/YOUR-COMPETITION-NAME/test.csv'
submission_file = '/kaggle/input/YOUR-COMPETITION-NAME/sample_submission.csv'


import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DETECT FILE PATHS
# ============================================================
print("=" * 70)
print("ğŸ“� DETECTING FILE PATHS")
print("=" * 70)

# Common file path patterns
possible_paths = [
    # Kaggle paths
    '/kaggle/input/',
    '/kaggle/input/playground-series-s5e6/',
    '/kaggle/input/loan-prediction/',
    # Google Colab paths
    '/content/',
    '/content/drive/MyDrive/',
    # Local paths
    './',
    './data/',
    '../input/',
]

# Try to find the correct path
train_file = None
test_file = None
submission_file = None

# First, let's see what's in common directories
print("\nğŸ”� Searching for data files...")

for base_path in possible_paths:
    if os.path.exists(base_path):
        # List directories in this path
        try:
            items = os.listdir(base_path)
            # Check for CSV files directly
            if 'train.csv' in items:
                train_file = os.path.join(base_path, 'train.csv')
                test_file = os.path.join(base_path, 'test.csv')
                submission_file = os.path.join(base_path, 'sample_submission.csv')
                print(f"âœ… Found files in: {base_path}")
                break
            # Check subdirectories (for Kaggle)
            for item in items:
                sub_path = os.path.join(base_path, item)
                if os.path.isdir(sub_path):
                    sub_items = os.listdir(sub_path)
                    if 'train.csv' in sub_items:
                        train_file = os.path.join(sub_path, 'train.csv')
                        test_file = os.path.join(sub_path, 'test.csv')
                        submission_file = os.path.join(sub_path, 'sample_submission.csv')
                        print(f"âœ… Found files in: {sub_path}")
                        break
        except PermissionError:
            continue
    if train_file:
        break

# If still not found, list what's available
if not train_file:
    print("\nâš ï¸� Could not auto-detect files. Listing available directories:")
    for base_path in ['/kaggle/input/', '/content/', './']:
        if os.path.exists(base_path):
            print(f"\nğŸ“‚ {base_path}:")
            try:
                for item in os.listdir(base_path):
                    print(f"   - {item}")
            except:
                print("   (Cannot list)")
    
    # Manual path input - UPDATE THIS IF NEEDED
    print("\n" + "=" * 70)
    print("âš ï¸� PLEASE UPDATE THE FILE PATHS BELOW:")
    print("=" * 70)
    
    # Try Kaggle competition format
    train_file = '/kaggle/input/playground-series-s5e6/train.csv'
    test_file = '/kaggle/input/playground-series-s5e6/test.csv'
    submission_file = '/kaggle/input/playground-series-s5e6/sample_submission.csv'

print(f"\nğŸ“„ Train file: {train_file}")
print(f"ğŸ“„ Test file: {test_file}")
print(f"ğŸ“„ Submission file: {submission_file}")

# ============================================================
# 2. LOAD DATA
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“� LOADING DATA")
print("=" * 70)

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
submission = pd.read_csv(submission_file)

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")
print(f"âœ… Submission shape: {submission.shape}")

# Separate features and target
X = train.drop(columns=['id', 'loan_paid_back'])
y = train['loan_paid_back']
X_test = test.drop(columns=['id'])

print(f"âœ… Features shape: {X.shape}")
print(f"âœ… Target distribution: {y.value_counts(normalize=True).to_dict()}")

# ============================================================
# 3. CUSTOM FEATURE ENGINEERING TRANSFORMER
# ============================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # ----- Grade Features -----
        X['grade'] = X['grade_subgrade'].str[0]
        X['subgrade_num'] = X['grade_subgrade'].str[1:].astype(int)
        
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        X['grade_ordinal'] = X['grade'].map(grade_map)
        X['grade_score'] = (X['grade_ordinal'] - 1) * 5 + X['subgrade_num']
        
        # ----- Financial Ratios -----
        X['income_to_loan'] = X['annual_income'] / (X['loan_amount'] + 1)
        X['loan_to_income'] = X['loan_amount'] / (X['annual_income'] + 1)
        X['monthly_income'] = X['annual_income'] / 12
        X['monthly_payment_est'] = (X['loan_amount'] * X['interest_rate'] / 100) / 12 + X['loan_amount'] / 36
        X['payment_to_income'] = X['monthly_payment_est'] / (X['monthly_income'] + 1)
        
        # ----- Debt Metrics -----
        X['total_debt'] = X['annual_income'] * X['debt_to_income_ratio']
        X['remaining_income'] = X['annual_income'] - X['total_debt']
        X['new_dti'] = (X['total_debt'] + X['monthly_payment_est'] * 12) / (X['annual_income'] + 1)
        
        # ----- Risk Indicators -----
        X['high_interest'] = (X['interest_rate'] > 15).astype(int)
        X['low_credit'] = (X['credit_score'] < 600).astype(int)
        X['high_dti'] = (X['debt_to_income_ratio'] > 0.2).astype(int)
        X['risk_score'] = X['high_interest'] + X['low_credit'] + X['high_dti']
        
        # ----- Credit Score Features -----
        X['credit_deviation'] = X['credit_score'] - 680
        X['credit_bin'] = pd.cut(
            X['credit_score'], 
            bins=[0, 580, 670, 740, 800, 900],
            labels=[1, 2, 3, 4, 5]
        ).astype(float).fillna(1)
        
        # ----- Interaction Features -----
        X['credit_x_grade'] = X['credit_score'] / (X['grade_ordinal'] + 1)
        X['income_x_credit'] = X['annual_income'] * X['credit_score'] / 1000000
        X['dti_x_rate'] = X['debt_to_income_ratio'] * X['interest_rate']
        X['credit_div_dti'] = X['credit_score'] / (X['debt_to_income_ratio'] * 100 + 1)
        X['loan_x_rate'] = X['loan_amount'] * X['interest_rate'] / 10000
        
        # ----- Log Transformations -----
        X['loan_log'] = np.log1p(X['loan_amount'])
        X['income_log'] = np.log1p(X['annual_income'])
        
        return X

# ============================================================
# 4. DEFINE FEATURE COLUMNS
# ============================================================
# Categorical columns
cat_cols = ['gender', 'marital_status', 'education_level', 
            'employment_status', 'loan_purpose', 'grade_subgrade', 'grade']

# Numerical columns (original + engineered)
num_cols = [
    'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
    'subgrade_num', 'grade_ordinal', 'grade_score',
    'income_to_loan', 'loan_to_income', 'monthly_income', 'monthly_payment_est', 'payment_to_income',
    'total_debt', 'remaining_income', 'new_dti',
    'high_interest', 'low_credit', 'high_dti', 'risk_score',
    'credit_deviation', 'credit_bin',
    'credit_x_grade', 'income_x_credit', 'dti_x_rate', 'credit_div_dti', 'loan_x_rate',
    'loan_log', 'income_log'
]

# ============================================================
# 5. BUILD PREPROCESSING PIPELINE
# ============================================================
print("\n" + "=" * 70)
print("ğŸ”§ BUILDING PIPELINE")
print("=" * 70)

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='drop'
)

print("âœ… Preprocessing pipeline created!")

# ============================================================
# 6. CROSS-VALIDATION WITH XGBOOST
# ============================================================
print("\n" + "=" * 70)
print("ğŸ”„ CROSS-VALIDATION (5-Fold Stratified)")
print("=" * 70)

# XGBoost parameters
xgb_params = {
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.03,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'early_stopping_rounds': 100
}

# Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store results
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
fold_scores = []
feature_importance_list = []

# Feature engineering instance
fe = FeatureEngineer()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'â”€' * 50}")
    print(f"ğŸ“Š Fold {fold + 1}/{n_splits}")
    print(f"{'â”€' * 50}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test_fold = X_test.copy()
    
    # Apply feature engineering
    X_train_fe = fe.fit_transform(X_train)
    X_val_fe = fe.transform(X_val)
    X_test_fe = fe.transform(X_test_fold)
    
    # Fit preprocessor and transform
    X_train_processed = preprocessor.fit_transform(X_train_fe)
    X_val_processed = preprocessor.transform(X_val_fe)
    X_test_processed = preprocessor.transform(X_test_fe)
    
    # Train XGBoost
    model = XGBClassifier(**xgb_params)
    model.fit(
        X_train_processed, 
        y_train,
        eval_set=[(X_val_processed, y_val)],
        verbose=100
    )
    
    # Predictions
    val_pred = model.predict_proba(X_val_processed)[:, 1]
    oof_predictions[val_idx] = val_pred
    test_predictions += model.predict_proba(X_test_processed)[:, 1] / n_splits
    
    # Store feature importance
    feature_importance_list.append(model.feature_importances_)
    
    # Calculate fold score
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(fold_auc)
    
    print(f"\nâœ… Fold {fold + 1} AUC: {fold_auc:.5f}")
    print(f"   Best iteration: {model.best_iteration}")

# ============================================================
# 7. RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“ˆ CROSS-VALIDATION RESULTS")
print("=" * 70)

overall_auc = roc_auc_score(y, oof_predictions)

print(f"\n{'Fold':<10} {'AUC Score':<15}")
print("-" * 25)
for i, score in enumerate(fold_scores):
    print(f"Fold {i+1:<5} {score:.5f}")
print("-" * 25)
print(f"{'Mean':<10} {np.mean(fold_scores):.5f}")
print(f"{'Std':<10} {np.std(fold_scores):.5f}")
print(f"{'Overall':<10} {overall_auc:.5f}")

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š FEATURE IMPORTANCE (Top 20)")
print("=" * 70)

feature_names = num_cols + cat_cols
avg_importance = np.mean(feature_importance_list, axis=0)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': avg_importance
}).sort_values('importance', ascending=False)

print(f"\n{'Rank':<6} {'Feature':<30} {'Importance':<15}")
print("-" * 55)
for idx, (_, row) in enumerate(importance_df.head(20).iterrows()):
    print(f"{idx+1:<6} {row['feature']:<30} {row['importance']:.4f}")

# ============================================================
# 9. CREATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("ğŸ’¾ CREATING SUBMISSION")
print("=" * 70)

submission['loan_paid_back'] = test_predictions
submission.to_csv('submission.csv', index=False)

print("\nâœ… Submission saved to 'submission.csv'")
print(f"\nğŸ“Œ Submission Preview:")
print(submission.head(10))

print(f"\nğŸ“Œ Prediction Statistics:")
print(f"   Mean:  {test_predictions.mean():.4f}")
print(f"   Std:   {test_predictions.std():.4f}")
print(f"   Min:   {test_predictions.min():.4f}")
print(f"   Max:   {test_predictions.max():.4f}")

# ============================================================
# 10. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("âœ… PIPELINE EXECUTION COMPLETE!")
print("=" * 70)
print(f"""
    Model: XGBoost Classifier
    CV Score (AUC): {np.mean(fold_scores):.5f} Â± {np.std(fold_scores):.5f}
    Overall OOF AUC: {overall_auc:.5f}
    Features Used: {len(feature_names)}
    Folds: {n_splits}
""")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DETECT AND LOAD DATA
# ============================================================
print("=" * 70)
print("ğŸ“� LOADING DATA")
print("=" * 70)

# Auto-detect file paths
possible_paths = [
    '/kaggle/input/playground-series-s5e6/',
    '/kaggle/input/',
    '/content/',
    './',
    './data/',
]

train_file = None
for base_path in possible_paths:
    if os.path.exists(base_path):
        try:
            items = os.listdir(base_path)
            if 'train.csv' in items:
                train_file = os.path.join(base_path, 'train.csv')
                break
            for item in items:
                sub_path = os.path.join(base_path, item)
                if os.path.isdir(sub_path):
                    if 'train.csv' in os.listdir(sub_path):
                        train_file = os.path.join(sub_path, 'train.csv')
                        break
        except:
            continue
    if train_file:
        break

if train_file is None:
    train_file = '/kaggle/input/playground-series-s5e6/train.csv'

print(f"ğŸ“„ Loading from: {train_file}")
train = pd.read_csv(train_file)
print(f"âœ… Train shape: {train.shape}")

# Target variable
target = 'loan_paid_back'
y = train[target]

# ============================================================
# 2. CLASS DISTRIBUTION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š CLASS DISTRIBUTION ANALYSIS")
print("=" * 70)

# Value counts
class_counts = y.value_counts()
class_percentages = y.value_counts(normalize=True) * 100

print("\nğŸ“Œ Target Variable: loan_paid_back")
print("-" * 40)
print(f"{'Class':<15} {'Count':<15} {'Percentage':<15}")
print("-" * 40)
for cls in class_counts.index:
    label = "Paid Back" if cls == 1 else "Not Paid"
    print(f"{label:<15} {class_counts[cls]:<15,} {class_percentages[cls]:.2f}%")
print("-" * 40)
print(f"{'Total':<15} {len(y):<15,} {'100.00%':<15}")

# ============================================================
# 3. IMBALANCE METRICS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“ˆ IMBALANCE METRICS")
print("=" * 70)

# Calculate various imbalance metrics
majority_class = class_counts.max()
minority_class = class_counts.min()
imbalance_ratio = majority_class / minority_class
majority_percentage = (majority_class / len(y)) * 100
minority_percentage = (minority_class / len(y)) * 100

print(f"""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚                    IMBALANCE METRICS                        â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Majority Class (Paid Back = 1):     {majority_class:>15,}      â”‚
â”‚  Minority Class (Not Paid = 0):      {minority_class:>15,}      â”‚
â”‚                                                             â”‚
â”‚  Majority Percentage:                {majority_percentage:>15.2f}%     â”‚
â”‚  Minority Percentage:                {minority_percentage:>15.2f}%     â”‚
â”‚                                                             â”‚
â”‚  Imbalance Ratio:                    {imbalance_ratio:>15.2f}:1     â”‚
â”‚  (Majority / Minority)                                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

# ============================================================
# 4. IMBALANCE SEVERITY ASSESSMENT
# ============================================================
print("=" * 70)
print("âš ï¸� IMBALANCE SEVERITY ASSESSMENT")
print("=" * 70)

if imbalance_ratio < 1.5:
    severity = "BALANCED"
    color = "ğŸŸ¢"
    recommendation = "No special handling needed."
elif imbalance_ratio < 3:
    severity = "MILD IMBALANCE"
    color = "ğŸŸ¡"
    recommendation = "Consider class weights or stratified sampling."
elif imbalance_ratio < 10:
    severity = "MODERATE IMBALANCE"
    color = "ğŸŸ "
    recommendation = "Use class weights, SMOTE, or undersampling."
else:
    severity = "SEVERE IMBALANCE"
    color = "ğŸ”´"
    recommendation = "Definitely use class weights, SMOTE, or ensemble methods."

print(f"""
{color} Severity Level: {severity}

ğŸ“‹ Assessment:
   â€¢ Imbalance Ratio: {imbalance_ratio:.2f}:1
   â€¢ Minority Class: {minority_percentage:.2f}%
   â€¢ Majority Class: {majority_percentage:.2f}%

ğŸ’¡ Recommendation: {recommendation}
""")

# ============================================================
# 5. VISUALIZATION
# ============================================================
print("=" * 70)
print("ğŸ“Š VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Bar Chart
ax1 = axes[0]
colors = ['#ff6b6b', '#4ecdc4']
bars = ax1.bar(['Not Paid (0)', 'Paid Back (1)'], 
               [class_counts[0], class_counts[1]], 
               color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Class', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')

# Add count labels on bars
for bar, count in zip(bars, [class_counts[0], class_counts[1]]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
             f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Pie Chart
ax2 = axes[1]
explode = (0.05, 0)
ax2.pie([class_counts[0], class_counts[1]], 
        explode=explode,
        labels=['Not Paid (0)', 'Paid Back (1)'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
        textprops={'fontsize': 11})
ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

# Plot 3: Imbalance Ratio Visual
ax3 = axes[2]
categories = ['Minority\n(Not Paid)', 'Majority\n(Paid Back)']
values = [1, imbalance_ratio]
bars = ax3.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Ratio', fontsize=12)
ax3.set_title(f'Imbalance Ratio ({imbalance_ratio:.2f}:1)', fontsize=14, fontweight='bold')
ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7)

# Add ratio labels
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('class_imbalance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nâœ… Visualization saved as 'class_imbalance.png'")

# ============================================================
# 6. CLASS DISTRIBUTION BY FEATURES
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š CLASS DISTRIBUTION BY CATEGORICAL FEATURES")
print("=" * 70)

cat_features = ['gender', 'marital_status', 'education_level', 
                'employment_status', 'loan_purpose', 'grade_subgrade']

# Calculate repayment rate by category
for feature in cat_features[:5]:  # Exclude grade_subgrade (too many categories)
    print(f"\nğŸ“Œ {feature.upper()}")
    print("-" * 50)
    
    grouped = train.groupby(feature)[target].agg(['count', 'sum', 'mean'])
    grouped.columns = ['Total', 'Paid Back', 'Repayment Rate']
    grouped['Not Paid'] = grouped['Total'] - grouped['Paid Back']
    grouped = grouped[['Total', 'Paid Back', 'Not Paid', 'Repayment Rate']]
    grouped['Repayment Rate'] = grouped['Repayment Rate'].apply(lambda x: f"{x*100:.2f}%")
    grouped = grouped.sort_values('Total', ascending=False)
    
    print(grouped.to_string())

# ============================================================
# 7. GRADE ANALYSIS (Important for loan prediction)
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š REPAYMENT RATE BY LOAN GRADE")
print("=" * 70)

train['grade'] = train['grade_subgrade'].str[0]
grade_analysis = train.groupby('grade')[target].agg(['count', 'mean']).round(4)
grade_analysis.columns = ['Count', 'Repayment Rate']
grade_analysis = grade_analysis.sort_index()

print("\n" + grade_analysis.to_string())

# Visualize repayment rate by grade
fig, ax = plt.subplots(figsize=(10, 5))
grades = grade_analysis.index.tolist()
rates = grade_analysis['Repayment Rate'].values * 100

colors_gradient = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(grades)))
bars = ax.bar(grades, rates, color=colors_gradient, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Loan Grade', fontsize=12)
ax.set_ylabel('Repayment Rate (%)', fontsize=12)
ax.set_title('Repayment Rate by Loan Grade', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)

# Add percentage labels
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('repayment_by_grade.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. NUMERICAL FEATURES DISTRIBUTION BY CLASS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š NUMERICAL FEATURES BY CLASS")
print("=" * 70)

num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                'loan_amount', 'interest_rate']

print(f"\n{'Feature':<25} {'Not Paid (Mean)':<18} {'Paid Back (Mean)':<18} {'Difference':<15}")
print("-" * 80)

for feature in num_features:
    mean_0 = train[train[target] == 0][feature].mean()
    mean_1 = train[train[target] == 1][feature].mean()
    diff = mean_1 - mean_0
    diff_pct = (diff / mean_0) * 100 if mean_0 != 0 else 0
    
    print(f"{feature:<25} {mean_0:<18.2f} {mean_1:<18.2f} {diff:+.2f} ({diff_pct:+.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(num_features):
    ax = axes[idx]
    
    # Plot distributions for each class
    train[train[target] == 0][feature].hist(ax=ax, bins=50, alpha=0.6, 
                                             label='Not Paid', color='#ff6b6b', density=True)
    train[train[target] == 1][feature].hist(ax=ax, bins=50, alpha=0.6, 
                                             label='Paid Back', color='#4ecdc4', density=True)
    
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{feature} Distribution by Class', fontsize=11, fontweight='bold')
    ax.legend()

# Remove empty subplot
axes[5].axis('off')

plt.tight_layout()
plt.savefig('numerical_by_class.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 9. HANDLING IMBALANCE - RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ’¡ RECOMMENDATIONS FOR HANDLING IMBALANCE")
print("=" * 70)

print(f"""
Based on the analysis:
â€¢ Imbalance Ratio: {imbalance_ratio:.2f}:1
â€¢ Minority Class (Not Paid): {minority_percentage:.2f}%
â€¢ Majority Class (Paid Back): {majority_percentage:.2f}%

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚                  RECOMMENDED STRATEGIES                             â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                     â”‚
â”‚  1ï¸�âƒ£  CLASS WEIGHTS (Recommended âœ“)                                  â”‚
â”‚      â€¢ Set scale_pos_weight in XGBoost                              â”‚
â”‚      â€¢ scale_pos_weight = {imbalance_ratio:.2f}                              â”‚
â”‚                                                                     â”‚
â”‚  2ï¸�âƒ£  STRATIFIED SAMPLING                                            â”‚
â”‚      â€¢ Already using StratifiedKFold âœ“                              â”‚
â”‚      â€¢ Maintains class distribution in each fold                    â”‚
â”‚                                                                     â”‚
â”‚  3ï¸�âƒ£  EVALUATION METRIC                                              â”‚
â”‚      â€¢ Use AUC-ROC (robust to imbalance) âœ“                          â”‚
â”‚      â€¢ Avoid accuracy (misleading with imbalance)                   â”‚
â”‚                                                                     â”‚
â”‚  4ï¸�âƒ£  THRESHOLD TUNING                                               â”‚
â”‚      â€¢ Default 0.5 may not be optimal                               â”‚
â”‚      â€¢ Tune based on precision-recall trade-off                     â”‚
â”‚                                                                     â”‚
â”‚  5ï¸�âƒ£  OPTIONAL: SMOTE/Undersampling                                  â”‚
â”‚      â€¢ May help but not always necessary                            â”‚
â”‚      â€¢ With ~20% minority, class weights often sufficient           â”‚
â”‚                                                                     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

# ============================================================
# 10. CALCULATE OPTIMAL CLASS WEIGHT
# ============================================================
print("=" * 70)
print("âš–ï¸� OPTIMAL CLASS WEIGHT CALCULATION")
print("=" * 70)

# Method 1: Inverse frequency
weight_0 = len(y) / (2 * (y == 0).sum())
weight_1 = len(y) / (2 * (y == 1).sum())

# Method 2: Scale pos weight for XGBoost
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

print(f"""
ğŸ“Š Class Weight Options:

Method 1: Inverse Frequency (for class_weight parameter)
   â€¢ Class 0 (Not Paid) weight:  {weight_0:.4f}
   â€¢ Class 1 (Paid Back) weight: {weight_1:.4f}
   
   Usage: class_weight={{0: {weight_0:.4f}, 1: {weight_1:.4f}}}

Method 2: Scale Pos Weight (for XGBoost)
   â€¢ scale_pos_weight = {scale_pos_weight:.4f}
   
   Note: This gives more weight to the MINORITY class (Not Paid)
   since we're predicting probability of paying back.
   
   For XGBoost: scale_pos_weight = {(y == 0).sum() / (y == 1).sum():.4f}
   (ratio of negative to positive class)
""")

# ============================================================
# 11. SUMMARY
# ============================================================
print("=" * 70)
print("ğŸ“‹ FINAL SUMMARY")
print("=" * 70)

print(f"""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚                     DATA IMBALANCE SUMMARY                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                     â”‚
â”‚  Target Variable: loan_paid_back                                    â”‚
â”‚                                                                     â”‚
â”‚  Class Distribution:                                                â”‚
â”‚    â€¢ Class 1 (Paid Back):  {class_counts[1]:>10,} ({majority_percentage:.2f}%)              â”‚
â”‚    â€¢ Class 0 (Not Paid):   {class_counts[0]:>10,} ({minority_percentage:.2f}%)              â”‚
â”‚                                                                     â”‚
â”‚  Imbalance Ratio: {imbalance_ratio:.2f}:1                                          â”‚
â”‚  Severity: {severity:<15}                                       â”‚
â”‚                                                                     â”‚
â”‚  Key Findings:                                                      â”‚
â”‚    â€¢ Moderate imbalance (~80/20 split)                              â”‚
â”‚    â€¢ Lower grades (E, F, G) have lower repayment rates              â”‚
â”‚    â€¢ Higher credit scores correlate with repayment                  â”‚
â”‚    â€¢ Higher interest rates correlate with non-payment               â”‚
â”‚                                                                     â”‚
â”‚  Recommended Approach:                                              â”‚
â”‚    âœ“ Use class weights in XGBoost                                   â”‚
â”‚    âœ“ Use AUC-ROC as evaluation metric                               â”‚
â”‚    âœ“ Use StratifiedKFold for cross-validation                       â”‚
â”‚                                                                     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

print("\nâœ… IMBALANCE ANALYSIS COMPLETE!")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("ğŸ“� LOADING DATA")
print("=" * 70)

# Auto-detect file paths
possible_paths = [
    '/kaggle/input/playground-series-s5e6/',
    '/kaggle/input/',
    '/content/',
    './',
    './data/',
]

train_file = None
for base_path in possible_paths:
    if os.path.exists(base_path):
        try:
            items = os.listdir(base_path)
            if 'train.csv' in items:
                train_file = os.path.join(base_path, 'train.csv')
                test_file = os.path.join(base_path, 'test.csv')
                submission_file = os.path.join(base_path, 'sample_submission.csv')
                break
            for item in items:
                sub_path = os.path.join(base_path, item)
                if os.path.isdir(sub_path):
                    if 'train.csv' in os.listdir(sub_path):
                        train_file = os.path.join(sub_path, 'train.csv')
                        test_file = os.path.join(sub_path, 'test.csv')
                        submission_file = os.path.join(sub_path, 'sample_submission.csv')
                        break
        except:
            continue
    if train_file:
        break

if train_file is None:
    train_file = '/kaggle/input/playground-series-s5e6/train.csv'
    test_file = '/kaggle/input/playground-series-s5e6/test.csv'
    submission_file = '/kaggle/input/playground-series-s5e6/sample_submission.csv'

print(f"ğŸ“„ Loading from: {train_file}")
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
submission = pd.read_csv(submission_file)

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")

# Separate features and target
X = train.drop(columns=['id', 'loan_paid_back'])
y = train['loan_paid_back']
X_test = test.drop(columns=['id'])

print(f"âœ… Target distribution:\n{y.value_counts()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Grade Features
        X['grade'] = X['grade_subgrade'].str[0]
        X['subgrade_num'] = X['grade_subgrade'].str[1:].astype(int)
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        X['grade_ordinal'] = X['grade'].map(grade_map)
        X['grade_score'] = (X['grade_ordinal'] - 1) * 5 + X['subgrade_num']
        
        # Financial Ratios
        X['income_to_loan'] = X['annual_income'] / (X['loan_amount'] + 1)
        X['loan_to_income'] = X['loan_amount'] / (X['annual_income'] + 1)
        X['monthly_income'] = X['annual_income'] / 12
        X['monthly_payment_est'] = (X['loan_amount'] * X['interest_rate'] / 100) / 12 + X['loan_amount'] / 36
        X['payment_to_income'] = X['monthly_payment_est'] / (X['monthly_income'] + 1)
        
        # Debt Metrics
        X['total_debt'] = X['annual_income'] * X['debt_to_income_ratio']
        X['remaining_income'] = X['annual_income'] - X['total_debt']
        X['new_dti'] = (X['total_debt'] + X['monthly_payment_est'] * 12) / (X['annual_income'] + 1)
        
        # Risk Indicators
        X['high_interest'] = (X['interest_rate'] > 15).astype(int)
        X['low_credit'] = (X['credit_score'] < 600).astype(int)
        X['high_dti'] = (X['debt_to_income_ratio'] > 0.2).astype(int)
        X['risk_score'] = X['high_interest'] + X['low_credit'] + X['high_dti']
        
        # Credit Features
        X['credit_deviation'] = X['credit_score'] - 680
        X['credit_bin'] = pd.cut(X['credit_score'], bins=[0, 580, 670, 740, 800, 900],
                                  labels=[1, 2, 3, 4, 5]).astype(float).fillna(1)
        
        # Interactions
        X['credit_x_grade'] = X['credit_score'] / (X['grade_ordinal'] + 1)
        X['income_x_credit'] = X['annual_income'] * X['credit_score'] / 1000000
        X['dti_x_rate'] = X['debt_to_income_ratio'] * X['interest_rate']
        X['credit_div_dti'] = X['credit_score'] / (X['debt_to_income_ratio'] * 100 + 1)
        X['loan_x_rate'] = X['loan_amount'] * X['interest_rate'] / 10000
        
        # Log Transformations
        X['loan_log'] = np.log1p(X['loan_amount'])
        X['income_log'] = np.log1p(X['annual_income'])
        
        return X

# ============================================================
# 3. DEFINE COLUMNS
# ============================================================
cat_cols = ['gender', 'marital_status', 'education_level', 
            'employment_status', 'loan_purpose', 'grade_subgrade', 'grade']

num_cols = [
    'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
    'subgrade_num', 'grade_ordinal', 'grade_score',
    'income_to_loan', 'loan_to_income', 'monthly_income', 'monthly_payment_est', 'payment_to_income',
    'total_debt', 'remaining_income', 'new_dti',
    'high_interest', 'low_credit', 'high_dti', 'risk_score',
    'credit_deviation', 'credit_bin',
    'credit_x_grade', 'income_x_credit', 'dti_x_rate', 'credit_div_dti', 'loan_x_rate',
    'loan_log', 'income_log'
]

# ============================================================
# 4. BUILD PREPROCESSING PIPELINE
# ============================================================
print("\n" + "=" * 70)
print("ğŸ”§ BUILDING PIPELINE")
print("=" * 70)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='drop'
)

print("âœ… Pipeline created!")

# ============================================================
# 5. CROSS-VALIDATION WITH COMPREHENSIVE METRICS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ”„ CROSS-VALIDATION WITH METRICS CALCULATION")
print("=" * 70)

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"ğŸ“Š Scale pos weight: {scale_pos_weight:.4f}")

# XGBoost parameters
xgb_params = {
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.03,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'early_stopping_rounds': 100,
    'scale_pos_weight': scale_pos_weight
}

# Cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Storage for predictions and metrics
oof_predictions_proba = np.zeros(len(X))  # Probability predictions
oof_predictions_class = np.zeros(len(X))  # Class predictions
test_predictions = np.zeros(len(X_test))

# Storage for fold-wise metrics
fold_metrics = {
    'fold': [],
    'accuracy': [],
    'balanced_accuracy': [],
    'precision_0': [],
    'precision_1': [],
    'recall_0': [],
    'recall_1': [],
    'f1_0': [],
    'f1_1': [],
    'f1_macro': [],
    'f1_weighted': [],
    'roc_auc': [],
    'pr_auc': [],
    'mcc': [],
    'kappa': []
}

# Feature engineering instance
fe = FeatureEngineer()

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'â”€' * 50}")
    print(f"ğŸ“Š Fold {fold + 1}/{n_splits}")
    print(f"{'â”€' * 50}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test_fold = X_test.copy()
    
    # Apply feature engineering
    X_train_fe = fe.fit_transform(X_train)
    X_val_fe = fe.transform(X_val)
    X_test_fe = fe.transform(X_test_fold)
    
    # Fit preprocessor and transform
    X_train_processed = preprocessor.fit_transform(X_train_fe)
    X_val_processed = preprocessor.transform(X_val_fe)
    X_test_processed = preprocessor.transform(X_test_fe)
    
    # Train XGBoost
    model = XGBClassifier(**xgb_params)
    model.fit(
        X_train_processed, y_train,
        eval_set=[(X_val_processed, y_val)],
        verbose=100
    )
    
    # Predictions
    val_pred_proba = model.predict_proba(X_val_processed)[:, 1]
    val_pred_class = model.predict(X_val_processed)
    
    oof_predictions_proba[val_idx] = val_pred_proba
    oof_predictions_class[val_idx] = val_pred_class
    test_predictions += model.predict_proba(X_test_processed)[:, 1] / n_splits
    
    # Calculate fold metrics
    fold_metrics['fold'].append(fold + 1)
    fold_metrics['accuracy'].append(accuracy_score(y_val, val_pred_class))
    fold_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_val, val_pred_class))
    fold_metrics['precision_0'].append(precision_score(y_val, val_pred_class, pos_label=0))
    fold_metrics['precision_1'].append(precision_score(y_val, val_pred_class, pos_label=1))
    fold_metrics['recall_0'].append(recall_score(y_val, val_pred_class, pos_label=0))
    fold_metrics['recall_1'].append(recall_score(y_val, val_pred_class, pos_label=1))
    fold_metrics['f1_0'].append(f1_score(y_val, val_pred_class, pos_label=0))
    fold_metrics['f1_1'].append(f1_score(y_val, val_pred_class, pos_label=1))
    fold_metrics['f1_macro'].append(f1_score(y_val, val_pred_class, average='macro'))
    fold_metrics['f1_weighted'].append(f1_score(y_val, val_pred_class, average='weighted'))
    fold_metrics['roc_auc'].append(roc_auc_score(y_val, val_pred_proba))
    fold_metrics['pr_auc'].append(average_precision_score(y_val, val_pred_proba))
    fold_metrics['mcc'].append(matthews_corrcoef(y_val, val_pred_class))
    fold_metrics['kappa'].append(cohen_kappa_score(y_val, val_pred_class))
    
    print(f"âœ… Fold {fold + 1} ROC-AUC: {fold_metrics['roc_auc'][-1]:.5f}")
    print(f"   Best iteration: {model.best_iteration}")

# ============================================================
# 6. OVERALL METRICS CALCULATION
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“ˆ OVERALL METRICS (Out-of-Fold Predictions)")
print("=" * 70)

# Calculate overall metrics
overall_metrics = {
    'Accuracy': accuracy_score(y, oof_predictions_class),
    'Balanced Accuracy': balanced_accuracy_score(y, oof_predictions_class),
    'Precision (Class 0 - Not Paid)': precision_score(y, oof_predictions_class, pos_label=0),
    'Precision (Class 1 - Paid)': precision_score(y, oof_predictions_class, pos_label=1),
    'Recall (Class 0 - Not Paid)': recall_score(y, oof_predictions_class, pos_label=0),
    'Recall (Class 1 - Paid)': recall_score(y, oof_predictions_class, pos_label=1),
    'F1-Score (Class 0 - Not Paid)': f1_score(y, oof_predictions_class, pos_label=0),
    'F1-Score (Class 1 - Paid)': f1_score(y, oof_predictions_class, pos_label=1),
    'F1-Score (Macro)': f1_score(y, oof_predictions_class, average='macro'),
    'F1-Score (Weighted)': f1_score(y, oof_predictions_class, average='weighted'),
    'ROC-AUC Score': roc_auc_score(y, oof_predictions_proba),
    'PR-AUC Score': average_precision_score(y, oof_predictions_proba),
    'Matthews Correlation Coefficient': matthews_corrcoef(y, oof_predictions_class),
    'Cohen Kappa Score': cohen_kappa_score(y, oof_predictions_class)
}

print(f"\n{'Metric':<40} {'Value':<15}")
print("=" * 55)
for metric, value in overall_metrics.items():
    print(f"{metric:<40} {value:.5f}")

# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š DETAILED CLASSIFICATION REPORT")
print("=" * 70)

print("\n" + classification_report(y, oof_predictions_class, 
                                   target_names=['Not Paid (0)', 'Paid Back (1)'],
                                   digits=4))

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y, oof_predictions_class)
tn, fp, fn, tp = cm.ravel()

print(f"""
                      Predicted
                   Negative  Positive
                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
Actual Negative  â”‚ TN={tn:>6} â”‚ FP={fp:>6} â”‚
                 â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
Actual Positive  â”‚ FN={fn:>6} â”‚ TP={tp:>6} â”‚
                 â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

ğŸ“Œ Interpretation:
   â€¢ True Negatives (TN):  {tn:>8,} - Correctly predicted as Not Paid
   â€¢ False Positives (FP): {fp:>8,} - Incorrectly predicted as Paid (Type I Error)
   â€¢ False Negatives (FN): {fn:>8,} - Incorrectly predicted as Not Paid (Type II Error)
   â€¢ True Positives (TP):  {tp:>8,} - Correctly predicted as Paid Back
""")

# Additional derived metrics from confusion matrix
specificity = tn / (tn + fp)
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"""
ğŸ“Œ Derived Metrics:
   â€¢ Specificity (True Negative Rate): {specificity:.4f}
   â€¢ Negative Predictive Value (NPV):  {npv:.4f}
   â€¢ False Positive Rate (FPR):        {fpr:.4f}
   â€¢ False Negative Rate (FNR):        {fnr:.4f}
""")

# ============================================================
# 9. FOLD-WISE METRICS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š FOLD-WISE METRICS SUMMARY")
print("=" * 70)

fold_df = pd.DataFrame(fold_metrics)

# Display key metrics by fold
print("\nğŸ“Œ Key Metrics by Fold:")
display_cols = ['fold', 'accuracy', 'precision_1', 'recall_1', 'f1_1', 'roc_auc', 'pr_auc']
print(fold_df[display_cols].to_string(index=False, float_format='%.4f'))

# Mean and Std
print(f"\nğŸ“Œ Mean Â± Std across folds:")
print("-" * 60)
for col in display_cols[1:]:
    mean_val = fold_df[col].mean()
    std_val = fold_df[col].std()
    print(f"   {col:<20}: {mean_val:.4f} Â± {std_val:.4f}")

# ============================================================
# 10. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š GENERATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 15))

# ----- Plot 1: Confusion Matrix Heatmap -----
ax1 = fig.add_subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Paid (0)', 'Paid Back (1)'],
            yticklabels=['Not Paid (0)', 'Paid Back (1)'],
            annot_kws={'size': 14}, ax=ax1)
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# ----- Plot 2: Normalized Confusion Matrix -----
ax2 = fig.add_subplot(2, 3, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Not Paid (0)', 'Paid Back (1)'],
            yticklabels=['Not Paid (0)', 'Paid Back (1)'],
            annot_kws={'size': 14}, ax=ax2)
ax2.set_xlabel('Predicted Label', fontsize=12)
ax2.set_ylabel('True Label', fontsize=12)
ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')

# ----- Plot 3: ROC Curve -----
ax3 = fig.add_subplot(2, 3, 3)
fpr_curve, tpr_curve, thresholds_roc = roc_curve(y, oof_predictions_proba)
roc_auc = auc(fpr_curve, tpr_curve)

ax3.plot(fpr_curve, tpr_curve, color='#2ecc71', lw=2, 
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax3.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
ax3.fill_between(fpr_curve, tpr_curve, alpha=0.3, color='#2ecc71')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate', fontsize=12)
ax3.set_ylabel('True Positive Rate', fontsize=12)
ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right', fontsize=10)
ax3.grid(True, alpha=0.3)

# ----- Plot 4: Precision-Recall Curve -----
ax4 = fig.add_subplot(2, 3, 4)
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y, oof_predictions_proba)
pr_auc = average_precision_score(y, oof_predictions_proba)

ax4.plot(recall_curve, precision_curve, color='#e74c3c', lw=2,
         label=f'PR Curve (AUC = {pr_auc:.4f})')
ax4.fill_between(recall_curve, precision_curve, alpha=0.3, color='#e74c3c')

# Baseline (proportion of positive class)
baseline = y.mean()
ax4.axhline(y=baseline, color='gray', linestyle='--', lw=2, 
            label=f'Baseline (Positive Rate = {baseline:.2f})')

ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('Recall', fontsize=12)
ax4.set_ylabel('Precision', fontsize=12)
ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax4.legend(loc='lower left', fontsize=10)
ax4.grid(True, alpha=0.3)

# ----- Plot 5: Metrics Comparison Bar Chart -----
ax5 = fig.add_subplot(2, 3, 5)
metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
class_0_values = [overall_metrics['Precision (Class 0 - Not Paid)'],
                  overall_metrics['Recall (Class 0 - Not Paid)'],
                  overall_metrics['F1-Score (Class 0 - Not Paid)']]
class_1_values = [overall_metrics['Precision (Class 1 - Paid)'],
                  overall_metrics['Recall (Class 1 - Paid)'],
                  overall_metrics['F1-Score (Class 1 - Paid)']]

x = np.arange(len(metrics_to_plot))
width = 0.35

bars1 = ax5.bar(x - width/2, class_0_values, width, label='Not Paid (0)', color='#ff6b6b')
bars2 = ax5.bar(x + width/2, class_1_values, width, label='Paid Back (1)', color='#4ecdc4')

ax5.set_ylabel('Score', fontsize=12)
ax5.set_title('Precision, Recall, F1-Score by Class', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_to_plot)
ax5.legend()
ax5.set_ylim(0, 1.1)
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

# ----- Plot 6: Threshold Analysis -----
ax6 = fig.add_subplot(2, 3, 6)

# Calculate metrics at different thresholds
thresholds = np.arange(0.1, 0.95, 0.05)
precision_list = []
recall_list = []
f1_list = []

for thresh in thresholds:
    pred_thresh = (oof_predictions_proba >= thresh).astype(int)
    precision_list.append(precision_score(y, pred_thresh, zero_division=0))
    recall_list.append(recall_score(y, pred_thresh))
    f1_list.append(f1_score(y, pred_thresh))

ax6.plot(thresholds, precision_list, 'b-', lw=2, label='Precision', marker='o', markersize=4)
ax6.plot(thresholds, recall_list, 'g-', lw=2, label='Recall', marker='s', markersize=4)
ax6.plot(thresholds, f1_list, 'r-', lw=2, label='F1-Score', marker='^', markersize=4)
ax6.axvline(x=0.5, color='gray', linestyle='--', lw=1, label='Default Threshold (0.5)')

ax6.set_xlabel('Threshold', fontsize=12)
ax6.set_ylabel('Score', fontsize=12)
ax6.set_title('Metrics vs. Classification Threshold', fontsize=14, fontweight='bold')
ax6.legend(loc='best', fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.set_xlim([0.1, 0.9])
ax6.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('classification_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print("âœ… Visualizations saved as 'classification_metrics.png'")

# ============================================================
# 11. OPTIMAL THRESHOLD ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ�¯ OPTIMAL THRESHOLD ANALYSIS")
print("=" * 70)

# Find optimal threshold based on different criteria
thresholds_fine = np.arange(0.1, 0.9, 0.01)
results = []

for thresh in thresholds_fine:
    pred_thresh = (oof_predictions_proba >= thresh).astype(int)
    results.append({
        'threshold': thresh,
        'precision': precision_score(y, pred_thresh, zero_division=0),
        'recall': recall_score(y, pred_thresh),
        'f1': f1_score(y, pred_thresh),
        'balanced_acc': balanced_accuracy_score(y, pred_thresh)
    })

threshold_df = pd.DataFrame(results)

# Find optimal thresholds
optimal_f1_idx = threshold_df['f1'].idxmax()
optimal_balanced_idx = threshold_df['balanced_acc'].idxmax()

# Youden's J statistic (optimal point on ROC curve)
j_scores = tpr_curve - fpr_curve
optimal_j_idx = np.argmax(j_scores)
optimal_j_threshold = thresholds_roc[optimal_j_idx]

print(f"""
ğŸ“Œ Optimal Thresholds:

   Method                          Threshold    F1-Score    Balanced Acc
   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
   Maximum F1-Score                {threshold_df.loc[optimal_f1_idx, 'threshold']:.2f}         {threshold_df.loc[optimal_f1_idx, 'f1']:.4f}      {threshold_df.loc[optimal_f1_idx, 'balanced_acc']:.4f}
   Maximum Balanced Accuracy       {threshold_df.loc[optimal_balanced_idx, 'threshold']:.2f}         {threshold_df.loc[optimal_balanced_idx, 'f1']:.4f}      {threshold_df.loc[optimal_balanced_idx, 'balanced_acc']:.4f}
   Youden's J Statistic (ROC)      {optimal_j_threshold:.2f}         -           -
   Default (0.5)                   0.50         {threshold_df[threshold_df['threshold']==0.5]['f1'].values[0]:.4f}      {threshold_df[threshold_df['threshold']==0.5]['balanced_acc'].values[0]:.4f}
""")

# ============================================================
# 12. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“‹ COMPLETE METRICS SUMMARY")
print("=" * 70)

summary_table = f"""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚                        CLASSIFICATION METRICS SUMMARY                    â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                          â”‚
â”‚  ğŸ“Š OVERALL PERFORMANCE                                                  â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€                                                   â”‚
â”‚  Accuracy:                    {overall_metrics['Accuracy']:.4f}                                â”‚
â”‚  Balanced Accuracy:           {overall_metrics['Balanced Accuracy']:.4f}                                â”‚
â”‚  ROC-AUC Score:               {overall_metrics['ROC-AUC Score']:.4f}                                â”‚
â”‚  PR-AUC Score:                {overall_metrics['PR-AUC Score']:.4f}                                â”‚
â”‚  Matthews Corr. Coefficient:  {overall_metrics['Matthews Correlation Coefficient']:.4f}                                â”‚
â”‚  Cohen's Kappa:               {overall_metrics['Cohen Kappa Score']:.4f}                                â”‚
â”‚                                                                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                          â”‚
â”‚  ğŸ“Š CLASS 0 (NOT PAID) METRICS                                          â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€                                          â”‚
â”‚  Precision:                   {overall_metrics['Precision (Class 0 - Not Paid)']:.4f}                                â”‚
â”‚  Recall:                      {overall_metrics['Recall (Class 0 - Not Paid)']:.4f}                                â”‚
â”‚  F1-Score:                    {overall_metrics['F1-Score (Class 0 - Not Paid)']:.4f}                                â”‚
â”‚  Support:                     {int((y == 0).sum()):,}                              â”‚
â”‚                                                                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                          â”‚
â”‚  ğŸ“Š CLASS 1 (PAID BACK) METRICS                                         â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€                                         â”‚
â”‚  Precision:                   {overall_metrics['Precision (Class 1 - Paid)']:.4f}                                â”‚
â”‚  Recall:                      {overall_metrics['Recall (Class 1 - Paid)']:.4f}                                â”‚
â”‚  F1-Score:                    {overall_metrics['F1-Score (Class 1 - Paid)']:.4f}                                â”‚
â”‚  Support:                     {int((y == 1).sum()):,}                              â”‚
â”‚                                                                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                          â”‚
â”‚  ğŸ“Š AGGREGATE METRICS                                                    â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€                                                   â”‚
â”‚  F1-Score (Macro):            {overall_metrics['F1-Score (Macro)']:.4f}                                â”‚
â”‚  F1-Score (Weighted):         {overall_metrics['F1-Score (Weighted)']:.4f}                                â”‚
â”‚                                                                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                          â”‚
â”‚  ğŸ“Š CONFUSION MATRIX                                                     â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€                                                    â”‚
â”‚                          Predicted 0    Predicted 1                      â”‚
â”‚  Actual 0 (Not Paid)     TN={tn:>7,}    FP={fp:>7,}                      â”‚
â”‚  Actual 1 (Paid Back)    FN={fn:>7,}    TP={tp:>7,}                      â”‚
â”‚                                                                          â”‚
â”‚  Specificity:                 {specificity:.4f}                                â”‚
â”‚  False Positive Rate:         {fpr:.4f}                                â”‚
â”‚  False Negative Rate:         {fnr:.4f}                                â”‚
â”‚                                                                          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
"""

print(summary_table)

# ============================================================
# 13. CREATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("ğŸ’¾ CREATING SUBMISSION")
print("=" * 70)

submission['loan_paid_back'] = test_predictions
submission.to_csv('submission.csv', index=False)

print("âœ… Submission saved to 'submission.csv'")
print(f"\nğŸ“Œ Test Predictions Statistics:")
print(f"   Mean:  {test_predictions.mean():.4f}")
print(f"   Std:   {test_predictions.std():.4f}")
print(f"   Min:   {test_predictions.min():.4f}")
print(f"   Max:   {test_predictions.max():.4f}")

print("\n" + "=" * 70)
print("âœ… COMPLETE METRICS ANALYSIS FINISHED!")
print("=" * 70)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("ğŸ“� LOADING DATA")
print("=" * 70)

# Auto-detect file paths
possible_paths = [
    '/kaggle/input/playground-series-s5e6/',
    '/kaggle/input/',
    '/content/',
    './',
    './data/',
]

train_file = None
for base_path in possible_paths:
    if os.path.exists(base_path):
        try:
            items = os.listdir(base_path)
            if 'train.csv' in items:
                train_file = os.path.join(base_path, 'train.csv')
                test_file = os.path.join(base_path, 'test.csv')
                submission_file = os.path.join(base_path, 'sample_submission.csv')
                break
            for item in items:
                sub_path = os.path.join(base_path, item)
                if os.path.isdir(sub_path):
                    if 'train.csv' in os.listdir(sub_path):
                        train_file = os.path.join(sub_path, 'train.csv')
                        test_file = os.path.join(sub_path, 'test.csv')
                        submission_file = os.path.join(sub_path, 'sample_submission.csv')
                        break
        except:
            continue
    if train_file:
        break

if train_file is None:
    train_file = '/kaggle/input/playground-series-s5e6/train.csv'
    test_file = '/kaggle/input/playground-series-s5e6/test.csv'
    submission_file = '/kaggle/input/playground-series-s5e6/sample_submission.csv'

print(f"ğŸ“„ Loading from: {train_file}")
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
submission = pd.read_csv(submission_file)

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")

# Separate features and target
X = train.drop(columns=['id', 'loan_paid_back'])
y = train['loan_paid_back']
X_test = test.drop(columns=['id'])

# ============================================================
# 2. ANALYZE CLASS IMBALANCE
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š CLASS IMBALANCE ANALYSIS")
print("=" * 70)

class_counts = y.value_counts()
class_percentages = y.value_counts(normalize=True) * 100

n_class_0 = (y == 0).sum()  # Minority class (Not Paid)
n_class_1 = (y == 1).sum()  # Majority class (Paid Back)
total = len(y)

print(f"""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚                      CLASS DISTRIBUTION                              â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                      â”‚
â”‚  Class 0 (Not Paid - MINORITY):    {n_class_0:>10,} ({n_class_0/total*100:.2f}%)         â”‚
â”‚  Class 1 (Paid Back - MAJORITY):   {n_class_1:>10,} ({n_class_1/total*100:.2f}%)         â”‚
â”‚  Total Samples:                    {total:>10,}                      â”‚
â”‚                                                                      â”‚
â”‚  Imbalance Ratio:                  {n_class_1/n_class_0:.2f}:1                           â”‚
â”‚                                                                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

# ============================================================
# 3. CALCULATE CLASS WEIGHTS
# ============================================================
print("=" * 70)
print("âš–ï¸� CLASS WEIGHT CALCULATION METHODS")
print("=" * 70)

# Method 1: Simple ratio (for scale_pos_weight)
# scale_pos_weight = negative_cases / positive_cases
# In XGBoost, this gives more weight to the POSITIVE class (class 1)
# But we want to give more weight to MINORITY class (class 0 - Not Paid)

# For binary classification where:
# - Class 1 = Positive (Paid Back) - Majority
# - Class 0 = Negative (Not Paid) - Minority

# scale_pos_weight increases the weight of positive class predictions
# Since our positive class (1) is the majority, we need scale_pos_weight < 1
# OR we can flip our perspective

# Method 1: Standard scale_pos_weight
scale_pos_weight_standard = n_class_0 / n_class_1  # < 1, gives less weight to positive class

# Method 2: Inverse (if we want to penalize false negatives on minority class)
scale_pos_weight_inverse = n_class_1 / n_class_0  # > 1, gives more weight to positive class

# Method 3: Balanced weights
weight_class_0 = total / (2 * n_class_0)
weight_class_1 = total / (2 * n_class_1)

# Method 4: Square root balancing (less aggressive)
scale_pos_weight_sqrt = np.sqrt(n_class_0 / n_class_1)

print(f"""
ğŸ“Œ Different Class Weight Methods:

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚  METHOD                           â”‚  VALUE      â”‚  EFFECT               â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  1. Standard scale_pos_weight     â”‚  {scale_pos_weight_standard:.4f}     â”‚  Balances classes      â”‚
â”‚     (minority/majority)           â”‚             â”‚                       â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  2. Inverse scale_pos_weight      â”‚  {scale_pos_weight_inverse:.4f}     â”‚  Emphasizes majority   â”‚
â”‚     (majority/minority)           â”‚             â”‚                       â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  3. Balanced Class Weights        â”‚             â”‚                       â”‚
â”‚     - Weight for Class 0          â”‚  {weight_class_0:.4f}     â”‚  More weight to        â”‚
â”‚     - Weight for Class 1          â”‚  {weight_class_1:.4f}     â”‚  minority class        â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  4. Square Root Balancing         â”‚  {scale_pos_weight_sqrt:.4f}     â”‚  Moderate balancing    â”‚
â”‚     (sqrt of standard)            â”‚             â”‚                       â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  5. No Weighting (Baseline)       â”‚  1.0000     â”‚  No adjustment         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

ğŸ’¡ For XGBoost scale_pos_weight:
   - Values < 1: Give LESS weight to positive class (Class 1)
   - Values > 1: Give MORE weight to positive class (Class 1)
   - Value = 1: No adjustment (default)

ğŸ�¯ Our Goal: Better detect Class 0 (Not Paid - Minority)
   We'll use scale_pos_weight = {scale_pos_weight_standard:.4f} to balance the classes
""")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Grade Features
        X['grade'] = X['grade_subgrade'].str[0]
        X['subgrade_num'] = X['grade_subgrade'].str[1:].astype(int)
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        X['grade_ordinal'] = X['grade'].map(grade_map)
        X['grade_score'] = (X['grade_ordinal'] - 1) * 5 + X['subgrade_num']
        
        # Financial Ratios
        X['income_to_loan'] = X['annual_income'] / (X['loan_amount'] + 1)
        X['loan_to_income'] = X['loan_amount'] / (X['annual_income'] + 1)
        X['monthly_income'] = X['annual_income'] / 12
        X['monthly_payment_est'] = (X['loan_amount'] * X['interest_rate'] / 100) / 12 + X['loan_amount'] / 36
        X['payment_to_income'] = X['monthly_payment_est'] / (X['monthly_income'] + 1)
        
        # Debt Metrics
        X['total_debt'] = X['annual_income'] * X['debt_to_income_ratio']
        X['remaining_income'] = X['annual_income'] - X['total_debt']
        X['new_dti'] = (X['total_debt'] + X['monthly_payment_est'] * 12) / (X['annual_income'] + 1)
        
        # Risk Indicators
        X['high_interest'] = (X['interest_rate'] > 15).astype(int)
        X['low_credit'] = (X['credit_score'] < 600).astype(int)
        X['high_dti'] = (X['debt_to_income_ratio'] > 0.2).astype(int)
        X['risk_score'] = X['high_interest'] + X['low_credit'] + X['high_dti']
        
        # Credit Features
        X['credit_deviation'] = X['credit_score'] - 680
        X['credit_bin'] = pd.cut(X['credit_score'], bins=[0, 580, 670, 740, 800, 900],
                                  labels=[1, 2, 3, 4, 5]).astype(float).fillna(1)
        
        # Interactions
        X['credit_x_grade'] = X['credit_score'] / (X['grade_ordinal'] + 1)
        X['income_x_credit'] = X['annual_income'] * X['credit_score'] / 1000000
        X['dti_x_rate'] = X['debt_to_income_ratio'] * X['interest_rate']
        X['credit_div_dti'] = X['credit_score'] / (X['debt_to_income_ratio'] * 100 + 1)
        X['loan_x_rate'] = X['loan_amount'] * X['interest_rate'] / 10000
        
        # Log Transformations
        X['loan_log'] = np.log1p(X['loan_amount'])
        X['income_log'] = np.log1p(X['annual_income'])
        
        return X

# ============================================================
# 5. DEFINE COLUMNS
# ============================================================
cat_cols = ['gender', 'marital_status', 'education_level', 
            'employment_status', 'loan_purpose', 'grade_subgrade', 'grade']

num_cols = [
    'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
    'subgrade_num', 'grade_ordinal', 'grade_score',
    'income_to_loan', 'loan_to_income', 'monthly_income', 'monthly_payment_est', 'payment_to_income',
    'total_debt', 'remaining_income', 'new_dti',
    'high_interest', 'low_credit', 'high_dti', 'risk_score',
    'credit_deviation', 'credit_bin',
    'credit_x_grade', 'income_x_credit', 'dti_x_rate', 'credit_div_dti', 'loan_x_rate',
    'loan_log', 'income_log'
]

# ============================================================
# 6. BUILD PREPROCESSING PIPELINE
# ============================================================
print("\n" + "=" * 70)
print("ğŸ”§ BUILDING PREPROCESSING PIPELINE")
print("=" * 70)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='drop'
)

print("âœ… Preprocessing pipeline created!")

# ============================================================
# 7. COMPARE MODELS: WITHOUT vs WITH CLASS WEIGHTS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ”„ COMPARING MODELS: WITHOUT vs WITH CLASS WEIGHTS")
print("=" * 70)

# Feature engineering instance
fe = FeatureEngineer()

# Cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define different weight configurations to test
weight_configs = {
    'No Weights (Baseline)': 1.0,
    'Balanced (ratio)': scale_pos_weight_standard,
    'Square Root': scale_pos_weight_sqrt,
    'Custom (0.5)': 0.5,
    'Custom (0.3)': 0.3,
}

# Store results for comparison
all_results = {}

for config_name, weight_value in weight_configs.items():
    print(f"\n{'â”€' * 70}")
    print(f"ğŸ”§ Testing: {config_name} (scale_pos_weight = {weight_value:.4f})")
    print(f"{'â”€' * 70}")
    
    # XGBoost parameters with current weight
    xgb_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.03,
        'min_child_weight': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50,
        'scale_pos_weight': weight_value  # âš¡ CLASS WEIGHT ADJUSTMENT
    }
    
    # Storage for this configuration
    oof_proba = np.zeros(len(X))
    oof_class = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Feature engineering
        X_train_fe = fe.fit_transform(X_train)
        X_val_fe = fe.transform(X_val)
        X_test_fe = fe.transform(X_test.copy())
        
        # Preprocessing
        X_train_proc = preprocessor.fit_transform(X_train_fe)
        X_val_proc = preprocessor.transform(X_val_fe)
        X_test_proc = preprocessor.transform(X_test_fe)
        
        # Train model
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train_proc, y_train,
            eval_set=[(X_val_proc, y_val)],
            verbose=False
        )
        
        # Predictions
        val_proba = model.predict_proba(X_val_proc)[:, 1]
        val_class = model.predict(X_val_proc)
        
        oof_proba[val_idx] = val_proba
        oof_class[val_idx] = val_class
        test_proba += model.predict_proba(X_test_proc)[:, 1] / n_splits
        
        fold_auc = roc_auc_score(y_val, val_proba)
        fold_aucs.append(fold_auc)
    
    # Calculate all metrics for this configuration
    cm = confusion_matrix(y, oof_class)
    tn, fp, fn, tp = cm.ravel()
    
    results = {
        'scale_pos_weight': weight_value,
        'oof_proba': oof_proba,
        'oof_class': oof_class,
        'test_proba': test_proba,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': accuracy_score(y, oof_class),
        'balanced_accuracy': balanced_accuracy_score(y, oof_class),
        'precision_0': precision_score(y, oof_class, pos_label=0),
        'precision_1': precision_score(y, oof_class, pos_label=1),
        'recall_0': recall_score(y, oof_class, pos_label=0),
        'recall_1': recall_score(y, oof_class, pos_label=1),
        'f1_0': f1_score(y, oof_class, pos_label=0),
        'f1_1': f1_score(y, oof_class, pos_label=1),
        'f1_macro': f1_score(y, oof_class, average='macro'),
        'f1_weighted': f1_score(y, oof_class, average='weighted'),
        'roc_auc': roc_auc_score(y, oof_proba),
        'pr_auc': average_precision_score(y, oof_proba),
        'mcc': matthews_corrcoef(y, oof_class),
        'kappa': cohen_kappa_score(y, oof_class),
        'fold_aucs': fold_aucs
    }
    
    all_results[config_name] = results
    
    print(f"   ROC-AUC: {results['roc_auc']:.5f}")
    print(f"   F1 (Macro): {results['f1_macro']:.5f}")
    print(f"   Recall (Class 0): {results['recall_0']:.5f}")

# ============================================================
# 8. COMPARISON RESULTS TABLE
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š DETAILED COMPARISON OF CLASS WEIGHT CONFIGURATIONS")
print("=" * 70)

# Create comparison DataFrame
comparison_data = []
for config_name, results in all_results.items():
    comparison_data.append({
        'Configuration': config_name,
        'Weight': results['scale_pos_weight'],
        'ROC-AUC': results['roc_auc'],
        'PR-AUC': results['pr_auc'],
        'Accuracy': results['accuracy'],
        'Balanced Acc': results['balanced_accuracy'],
        'Precision (0)': results['precision_0'],
        'Recall (0)': results['recall_0'],
        'F1 (0)': results['f1_0'],
        'Precision (1)': results['precision_1'],
        'Recall (1)': results['recall_1'],
        'F1 (1)': results['f1_1'],
        'F1 Macro': results['f1_macro'],
        'MCC': results['mcc']
    })

comparison_df = pd.DataFrame(comparison_data)

print("\nğŸ“Œ Performance Metrics Comparison:")
print("-" * 120)
print(comparison_df.to_string(index=False, float_format='%.4f'))

# ============================================================
# 9. DETAILED ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š DETAILED ANALYSIS BY CONFIGURATION")
print("=" * 70)

for config_name, results in all_results.items():
    print(f"\n{'â•�' * 70}")
    print(f"ğŸ“Œ {config_name} (scale_pos_weight = {results['scale_pos_weight']:.4f})")
    print(f"{'â•�' * 70}")
    
    print(f"""
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
    â”‚  CONFUSION MATRIX                                               â”‚
    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    â”‚                          Predicted                              â”‚
    â”‚                     Negative    Positive                        â”‚
    â”‚  Actual Negative   TN={results['tn']:>7,}   FP={results['fp']:>7,}                    â”‚
    â”‚  Actual Positive   FN={results['fn']:>7,}   TP={results['tp']:>7,}                    â”‚
    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    â”‚  CLASS 0 (Not Paid - Minority)                                  â”‚
    â”‚    Precision: {results['precision_0']:.4f}  Recall: {results['recall_0']:.4f}  F1: {results['f1_0']:.4f}         â”‚
    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    â”‚  CLASS 1 (Paid Back - Majority)                                 â”‚
    â”‚    Precision: {results['precision_1']:.4f}  Recall: {results['recall_1']:.4f}  F1: {results['f1_1']:.4f}         â”‚
    â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    â”‚  OVERALL METRICS                                                â”‚
    â”‚    ROC-AUC: {results['roc_auc']:.4f}   PR-AUC: {results['pr_auc']:.4f}   MCC: {results['mcc']:.4f}          â”‚
    â”‚    Balanced Accuracy: {results['balanced_accuracy']:.4f}   F1 Macro: {results['f1_macro']:.4f}            â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    """)

# ============================================================
# 10. FIND BEST CONFIGURATION
# ============================================================
print("\n" + "=" * 70)
print("ğŸ�† BEST CONFIGURATION SELECTION")
print("=" * 70)

# Find best by different metrics
best_by_roc_auc = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Configuration']
best_by_f1_macro = comparison_df.loc[comparison_df['F1 Macro'].idxmax(), 'Configuration']
best_by_recall_0 = comparison_df.loc[comparison_df['Recall (0)'].idxmax(), 'Configuration']
best_by_balanced_acc = comparison_df.loc[comparison_df['Balanced Acc'].idxmax(), 'Configuration']
best_by_mcc = comparison_df.loc[comparison_df['MCC'].idxmax(), 'Configuration']

print(f"""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚                    BEST CONFIGURATION BY METRIC                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                          â”‚
â”‚  Best by ROC-AUC:           {best_by_roc_auc:<30}           â”‚
â”‚  Best by F1-Score (Macro):  {best_by_f1_macro:<30}           â”‚
â”‚  Best by Recall (Class 0):  {best_by_recall_0:<30}           â”‚
â”‚  Best by Balanced Accuracy: {best_by_balanced_acc:<30}           â”‚
â”‚  Best by MCC:               {best_by_mcc:<30}           â”‚
â”‚                                                                          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

# ============================================================
# 11. VISUALIZATION
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š GENERATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 16))

# ----- Plot 1: Metrics Comparison Bar Chart -----
ax1 = fig.add_subplot(2, 3, 1)
metrics_to_compare = ['ROC-AUC', 'Balanced Acc', 'F1 Macro', 'MCC']
x = np.arange(len(weight_configs))
width = 0.2

for i, metric in enumerate(metrics_to_compare):
    values = comparison_df[metric].values
    ax1.bar(x + i * width, values, width, label=metric)

ax1.set_xlabel('Configuration', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Overall Metrics Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels([c.split('(')[0].strip() for c in weight_configs.keys()], rotation=45, ha='right')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3, axis='y')

# ----- Plot 2: Class-wise Recall Comparison -----
ax2 = fig.add_subplot(2, 3, 2)
x = np.arange(len(weight_configs))
width = 0.35

bars1 = ax2.bar(x - width/2, comparison_df['Recall (0)'].values, width, 
                label='Recall (Class 0 - Minority)', color='#ff6b6b')
bars2 = ax2.bar(x + width/2, comparison_df['Recall (1)'].values, width, 
                label='Recall (Class 1 - Majority)', color='#4ecdc4')

ax2.set_xlabel('Configuration', fontsize=12)
ax2.set_ylabel('Recall', fontsize=12)
ax2.set_title('Recall by Class', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([c.split('(')[0].strip() for c in weight_configs.keys()], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# ----- Plot 3: F1-Score Comparison -----
ax3 = fig.add_subplot(2, 3, 3)
x = np.arange(len(weight_configs))
width = 0.35

bars1 = ax3.bar(x - width/2, comparison_df['F1 (0)'].values, width, 
                label='F1 (Class 0 - Minority)', color='#ff6b6b')
bars2 = ax3.bar(x + width/2, comparison_df['F1 (1)'].values, width, 
                label='F1 (Class 1 - Majority)', color='#4ecdc4')

ax3.set_xlabel('Configuration', fontsize=12)
ax3.set_ylabel('F1-Score', fontsize=12)
ax3.set_title('F1-Score by Class', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([c.split('(')[0].strip() for c in weight_configs.keys()], rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# ----- Plot 4: ROC Curves Comparison -----
ax4 = fig.add_subplot(2, 3, 4)
colors = plt.cm.Set1(np.linspace(0, 1, len(weight_configs)))

for (config_name, results), color in zip(all_results.items(), colors):
    fpr, tpr, _ = roc_curve(y, results['oof_proba'])
    auc_score = results['roc_auc']
    ax4.plot(fpr, tpr, color=color, lw=2, 
             label=f"{config_name.split('(')[0].strip()} (AUC={auc_score:.4f})")

ax4.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax4.set_xlabel('False Positive Rate', fontsize=12)
ax4.set_ylabel('True Positive Rate', fontsize=12)
ax4.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax4.legend(loc='lower right', fontsize=8)
ax4.grid(True, alpha=0.3)

# ----- Plot 5: Precision-Recall Curves Comparison -----
ax5 = fig.add_subplot(2, 3, 5)

for (config_name, results), color in zip(all_results.items(), colors):
    precision, recall, _ = precision_recall_curve(y, results['oof_proba'])
    pr_auc = results['pr_auc']
    ax5.plot(recall, precision, color=color, lw=2,
             label=f"{config_name.split('(')[0].strip()} (AUC={pr_auc:.4f})")

ax5.axhline(y=y.mean(), color='gray', linestyle='--', lw=1, label=f'Baseline ({y.mean():.2f})')
ax5.set_xlabel('Recall', fontsize=12)
ax5.set_ylabel('Precision', fontsize=12)
ax5.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
ax5.legend(loc='lower left', fontsize=8)
ax5.grid(True, alpha=0.3)

# ----- Plot 6: Confusion Matrices -----
ax6 = fig.add_subplot(2, 3, 6)

# Show confusion matrix for the balanced configuration
balanced_results = all_results['Balanced (ratio)']
cm = balanced_results['confusion_matrix']
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Not Paid (0)', 'Paid Back (1)'],
            yticklabels=['Not Paid (0)', 'Paid Back (1)'],
            annot_kws={'size': 12}, ax=ax6)
ax6.set_xlabel('Predicted', fontsize=12)
ax6.set_ylabel('Actual', fontsize=12)
ax6.set_title('Confusion Matrix (Balanced Weights)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_weight_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("âœ… Visualization saved as 'class_weight_comparison.png'")

# ============================================================
# 12. FINAL MODEL WITH BEST WEIGHTS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ�¯ TRAINING FINAL MODEL WITH BALANCED WEIGHTS")
print("=" * 70)

# Use balanced weights for final model
best_config = 'Balanced (ratio)'
best_weight = all_results[best_config]['scale_pos_weight']

print(f"âœ… Using: {best_config}")
print(f"âœ… scale_pos_weight = {best_weight:.4f}")

# Get predictions from best configuration
final_oof_proba = all_results[best_config]['oof_proba']
final_oof_class = all_results[best_config]['oof_class']
final_test_proba = all_results[best_config]['test_proba']

# ============================================================
# 13. FINAL CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š FINAL MODEL CLASSIFICATION REPORT")
print("=" * 70)

print("\n" + classification_report(y, final_oof_class,
                                   target_names=['Not Paid (0)', 'Paid Back (1)'],
                                   digits=4))

# ============================================================
# 14. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“‹ FINAL SUMMARY")
print("=" * 70)

baseline_results = all_results['No Weights (Baseline)']
balanced_results = all_results['Balanced (ratio)']

print(f"""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚              IMPACT OF CLASS WEIGHTS ON XGBOOST                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                          â”‚
â”‚  ğŸ“Š BASELINE (No Weights)                                                â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€                                              â”‚
â”‚    ROC-AUC:         {baseline_results['roc_auc']:.4f}                                         â”‚
â”‚    F1 Macro:        {baseline_results['f1_macro']:.4f}                                         â”‚
â”‚    Recall (0):      {baseline_results['recall_0']:.4f}  â†� Minority class detection              â”‚
â”‚    Recall (1):      {baseline_results['recall_1']:.4f}                                         â”‚
â”‚    Balanced Acc:    {baseline_results['balanced_accuracy']:.4f}                                         â”‚
â”‚                                                                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                          â”‚
â”‚  âš–ï¸� BALANCED WEIGHTS (scale_pos_weight = {best_weight:.4f})                      â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€                                â”‚
â”‚    ROC-AUC:         {balanced_results['roc_auc']:.4f}  (Î” {balanced_results['roc_auc'] - baseline_results['roc_auc']:+.4f})                          â”‚
â”‚    F1 Macro:        {balanced_results['f1_macro']:.4f}  (Î” {balanced_results['f1_macro'] - baseline_results['f1_macro']:+.4f})                          â”‚
â”‚    Recall (0):      {balanced_results['recall_0']:.4f}  (Î” {balanced_results['recall_0'] - baseline_results['recall_0']:+.4f}) â†� IMPROVED!              â”‚
â”‚    Recall (1):      {balanced_results['recall_1']:.4f}  (Î” {balanced_results['recall_1'] - baseline_results['recall_1']:+.4f})                          â”‚
â”‚    Balanced Acc:    {balanced_results['balanced_accuracy']:.4f}  (Î” {balanced_results['balanced_accuracy'] - baseline_results['balanced_accuracy']:+.4f})                          â”‚
â”‚                                                                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                                          â”‚
â”‚  ğŸ’¡ KEY INSIGHTS:                                                        â”‚
â”‚    â€¢ Class weights help balance recall between classes                   â”‚
â”‚    â€¢ Minority class (Not Paid) detection is improved                     â”‚
â”‚    â€¢ Trade-off: Slightly lower majority class performance                â”‚
â”‚    â€¢ Overall balanced accuracy improves with proper weighting            â”‚
â”‚                                                                          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

# ============================================================
# 15. CREATE SUBMISSION
# ============================================================
print("\n" + "=" * 70)
print("ğŸ’¾ CREATING SUBMISSION")
print("=" * 70)

submission['loan_paid_back'] = final_test_proba
submission.to_csv('submission.csv', index=False)

print("âœ… Submission saved to 'submission.csv'")
print(f"\nğŸ“Œ Submission Preview:")
print(submission.head(10))

print(f"\nğŸ“Œ Prediction Statistics:")
print(f"   Mean:  {final_test_proba.mean():.4f}")
print(f"   Std:   {final_test_proba.std():.4f}")
print(f"   Min:   {final_test_proba.min():.4f}")
print(f"   Max:   {final_test_proba.max():.4f}")

print("\n" + "=" * 70)
print("âœ… CLASS WEIGHT ANALYSIS COMPLETE!")
print("=" * 70)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ğŸ“¦ USING MANUAL SMOTE IMPLEMENTATION (No imblearn required)")
print("=" * 70)

# ============================================================
# MANUAL SAMPLING TECHNIQUES IMPLEMENTATION
# ============================================================

class ManualRandomOverSampler:
    """Random Oversampling - duplicates minority samples"""
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_count = np.max(counts)
        
        # Get indices of minority class
        minority_indices = np.where(y == minority_class)[0]
        n_minority = len(minority_indices)
        n_to_add = majority_count - n_minority
        
        # Randomly duplicate minority samples
        duplicate_indices = np.random.choice(minority_indices, size=n_to_add, replace=True)
        
        X_resampled = np.vstack([X, X[duplicate_indices]])
        y_resampled = np.hstack([y, y[duplicate_indices]])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


class ManualSMOTE:
    """SMOTE - Synthetic Minority Oversampling Technique"""
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_count = np.max(counts)
        
        # Get minority samples
        minority_mask = y == minority_class
        X_minority = X[minority_mask]
        n_minority = len(X_minority)
        n_to_generate = majority_count - n_minority
        
        if n_to_generate <= 0:
            return X, y
        
        # Fit nearest neighbors
        k = min(self.k_neighbors, n_minority - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_minority)
        
        # Generate synthetic samples
        synthetic_samples = []
        
        for _ in range(n_to_generate):
            idx = np.random.randint(0, n_minority)
            sample = X_minority[idx]
            
            _, indices = nn.kneighbors([sample])
            neighbor_idx = np.random.choice(indices[0][1:])
            neighbor = X_minority[neighbor_idx]
            
            # Interpolate
            gap = np.random.random()
            synthetic = sample + gap * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(len(synthetic_samples), minority_class)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.hstack([y, y_synthetic])
        
        shuffle_idx = np.random.permutation(len(y_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


class ManualBorderlineSMOTE:
    """Borderline-SMOTE - focuses on samples near decision boundary"""
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        majority_count = np.max(counts)
        
        minority_mask = y == minority_class
        X_minority = X[minority_mask]
        n_minority = len(X_minority)
        n_to_generate = majority_count - n_minority
        
        if n_to_generate <= 0:
            return X, y
        
        # Find borderline samples
        k = min(self.k_neighbors, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        
        borderline_indices = []
        for i, sample in enumerate(X_minority):
            _, indices = nn.kneighbors([sample])
            neighbors_y = y[indices[0][1:]]
            n_majority_neighbors = np.sum(neighbors_y == majority_class)
            
            # Borderline if half or more neighbors are majority
            if k // 2 <= n_majority_neighbors < k:
                borderline_indices.append(i)
        
        if len(borderline_indices) == 0:
            borderline_indices = list(range(n_minority))
        
        X_borderline = X_minority[borderline_indices]
        
        # Fit NN on minority class only
        k_minority = min(self.k_neighbors, n_minority - 1)
        nn_minority = NearestNeighbors(n_neighbors=k_minority + 1)
        nn_minority.fit(X_minority)
        
        # Generate synthetic samples from borderline
        synthetic_samples = []
        
        for _ in range(n_to_generate):
            idx = np.random.randint(0, len(X_borderline))
            sample = X_borderline[idx]
            
            _, indices = nn_minority.kneighbors([sample])
            neighbor_idx = np.random.choice(indices[0][1:])
            neighbor = X_minority[neighbor_idx]
            
            gap = np.random.random()
            synthetic = sample + gap * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(len(synthetic_samples), minority_class)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.hstack([y, y_synthetic])
        
        shuffle_idx = np.random.permutation(len(y_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


class ManualADASYN:
    """ADASYN - Adaptive Synthetic Sampling"""
    def __init__(self, n_neighbors=5, random_state=42):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        majority_count = np.max(counts)
        
        minority_mask = y == minority_class
        X_minority = X[minority_mask]
        n_minority = len(X_minority)
        n_to_generate = majority_count - n_minority
        
        if n_to_generate <= 0:
            return X, y
        
        # Calculate density ratio for each minority sample
        k = min(self.n_neighbors, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        
        ratios = []
        for sample in X_minority:
            _, indices = nn.kneighbors([sample])
            neighbors_y = y[indices[0][1:]]
            ratio = np.sum(neighbors_y == majority_class) / k
            ratios.append(ratio)
        
        ratios = np.array(ratios)
        if ratios.sum() == 0:
            ratios = np.ones(n_minority) / n_minority
        else:
            ratios = ratios / ratios.sum()
        
        # Generate samples based on ratios
        n_samples_per_point = np.round(ratios * n_to_generate).astype(int)
        
        # Fit NN on minority only
        k_minority = min(self.n_neighbors, n_minority - 1)
        nn_minority = NearestNeighbors(n_neighbors=k_minority + 1)
        nn_minority.fit(X_minority)
        
        synthetic_samples = []
        
        for i, n_samples in enumerate(n_samples_per_point):
            sample = X_minority[i]
            _, indices = nn_minority.kneighbors([sample])
            
            for _ in range(n_samples):
                neighbor_idx = np.random.choice(indices[0][1:])
                neighbor = X_minority[neighbor_idx]
                
                gap = np.random.random()
                synthetic = sample + gap * (neighbor - sample)
                synthetic_samples.append(synthetic)
        
        if len(synthetic_samples) == 0:
            return X, y
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(len(synthetic_samples), minority_class)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.hstack([y, y_synthetic])
        
        shuffle_idx = np.random.permutation(len(y_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


class ManualRandomUnderSampler:
    """Random Undersampling - removes majority samples"""
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        minority_count = np.min(counts)
        
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Undersample majority
        undersampled_majority = np.random.choice(majority_indices, size=minority_count, replace=False)
        
        selected_indices = np.hstack([minority_indices, undersampled_majority])
        
        shuffle_idx = np.random.permutation(len(selected_indices))
        selected_indices = selected_indices[shuffle_idx]
        
        return X[selected_indices], y[selected_indices]


class ManualSMOTETomek:
    """SMOTE + Tomek Links cleaning"""
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        # First apply SMOTE
        smote = ManualSMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Then remove Tomek links
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_resampled)
        
        tomek_links = []
        for i in range(len(X_resampled)):
            _, indices = nn.kneighbors([X_resampled[i]])
            neighbor_idx = indices[0][1]
            
            # Check if they form a Tomek link (mutual nearest neighbors with different classes)
            _, neighbor_indices = nn.kneighbors([X_resampled[neighbor_idx]])
            if neighbor_indices[0][1] == i and y_resampled[i] != y_resampled[neighbor_idx]:
                tomek_links.append(i)
        
        # Keep non-Tomek samples
        keep_mask = np.ones(len(X_resampled), dtype=bool)
        keep_mask[tomek_links] = False
        
        return X_resampled[keep_mask], y_resampled[keep_mask]


print("âœ… Manual sampling classes defined!")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“� LOADING DATA")
print("=" * 70)

possible_paths = [
    '/kaggle/input/playground-series-s5e6/',
    '/kaggle/input/',
    '/content/',
    './',
]

train_file = None
for base_path in possible_paths:
    if os.path.exists(base_path):
        try:
            items = os.listdir(base_path)
            if 'train.csv' in items:
                train_file = os.path.join(base_path, 'train.csv')
                test_file = os.path.join(base_path, 'test.csv')
                submission_file = os.path.join(base_path, 'sample_submission.csv')
                break
            for item in items:
                sub_path = os.path.join(base_path, item)
                if os.path.isdir(sub_path):
                    if 'train.csv' in os.listdir(sub_path):
                        train_file = os.path.join(sub_path, 'train.csv')
                        test_file = os.path.join(sub_path, 'test.csv')
                        submission_file = os.path.join(sub_path, 'sample_submission.csv')
                        break
        except:
            continue
    if train_file:
        break

if train_file is None:
    train_file = '/kaggle/input/playground-series-s5e6/train.csv'
    test_file = '/kaggle/input/playground-series-s5e6/test.csv'
    submission_file = '/kaggle/input/playground-series-s5e6/sample_submission.csv'

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
submission = pd.read_csv(submission_file)

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")

X = train.drop(columns=['id', 'loan_paid_back'])
y = train['loan_paid_back'].values
X_test = test.drop(columns=['id'])

# ============================================================
# 2. CLASS DISTRIBUTION
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š ORIGINAL CLASS DISTRIBUTION")
print("=" * 70)

n_class_0 = (y == 0).sum()
n_class_1 = (y == 1).sum()
total = len(y)

print(f"""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚  Class 0 (Not Paid - MINORITY):    {n_class_0:>10,} ({n_class_0/total*100:.2f}%)         â”‚
â”‚  Class 1 (Paid Back - MAJORITY):   {n_class_1:>10,} ({n_class_1/total*100:.2f}%)         â”‚
â”‚  Imbalance Ratio:                  {n_class_1/n_class_0:.2f}:1                           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("=" * 70)
print("ğŸ”§ FEATURE ENGINEERING")
print("=" * 70)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['grade'] = X['grade_subgrade'].str[0]
        X['subgrade_num'] = X['grade_subgrade'].str[1:].astype(int)
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        X['grade_ordinal'] = X['grade'].map(grade_map)
        X['grade_score'] = (X['grade_ordinal'] - 1) * 5 + X['subgrade_num']
        
        X['income_to_loan'] = X['annual_income'] / (X['loan_amount'] + 1)
        X['loan_to_income'] = X['loan_amount'] / (X['annual_income'] + 1)
        X['monthly_income'] = X['annual_income'] / 12
        X['monthly_payment_est'] = (X['loan_amount'] * X['interest_rate'] / 100) / 12 + X['loan_amount'] / 36
        X['payment_to_income'] = X['monthly_payment_est'] / (X['monthly_income'] + 1)
        
        X['total_debt'] = X['annual_income'] * X['debt_to_income_ratio']
        X['remaining_income'] = X['annual_income'] - X['total_debt']
        X['new_dti'] = (X['total_debt'] + X['monthly_payment_est'] * 12) / (X['annual_income'] + 1)
        
        X['high_interest'] = (X['interest_rate'] > 15).astype(int)
        X['low_credit'] = (X['credit_score'] < 600).astype(int)
        X['high_dti'] = (X['debt_to_income_ratio'] > 0.2).astype(int)
        X['risk_score'] = X['high_interest'] + X['low_credit'] + X['high_dti']
        
        X['credit_deviation'] = X['credit_score'] - 680
        X['credit_bin'] = pd.cut(X['credit_score'], bins=[0, 580, 670, 740, 800, 900],
                                  labels=[1, 2, 3, 4, 5]).astype(float).fillna(1)
        
        X['credit_x_grade'] = X['credit_score'] / (X['grade_ordinal'] + 1)
        X['income_x_credit'] = X['annual_income'] * X['credit_score'] / 1000000
        X['dti_x_rate'] = X['debt_to_income_ratio'] * X['interest_rate']
        X['credit_div_dti'] = X['credit_score'] / (X['debt_to_income_ratio'] * 100 + 1)
        X['loan_x_rate'] = X['loan_amount'] * X['interest_rate'] / 10000
        
        X['loan_log'] = np.log1p(X['loan_amount'])
        X['income_log'] = np.log1p(X['annual_income'])
        
        return X

fe = FeatureEngineer()
X_fe = fe.fit_transform(X)
X_test_fe = fe.transform(X_test)
print("âœ… Feature engineering complete!")

# ============================================================
# 4. PREPROCESSING
# ============================================================
original_cat_cols = ['gender', 'marital_status', 'education_level', 
                     'employment_status', 'loan_purpose', 'grade_subgrade', 'grade']

num_cols = [
    'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
    'subgrade_num', 'grade_ordinal', 'grade_score',
    'income_to_loan', 'loan_to_income', 'monthly_income', 'monthly_payment_est', 'payment_to_income',
    'total_debt', 'remaining_income', 'new_dti',
    'high_interest', 'low_credit', 'high_dti', 'risk_score',
    'credit_deviation', 'credit_bin',
    'credit_x_grade', 'income_x_credit', 'dti_x_rate', 'credit_div_dti', 'loan_x_rate',
    'loan_log', 'income_log'
]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, original_cat_cols)
    ],
    remainder='drop'
)

print("âœ… Preprocessing pipeline created!")

# ============================================================
# 5. DEFINE SAMPLING STRATEGIES
# ============================================================
print("\n" + "=" * 70)
print("ğŸ”§ SAMPLING STRATEGIES")
print("=" * 70)

sampling_strategies = {
    'No Sampling (Baseline)': None,
    'Random Oversampling': ManualRandomOverSampler(random_state=42),
    'SMOTE': ManualSMOTE(k_neighbors=5, random_state=42),
    'Borderline-SMOTE': ManualBorderlineSMOTE(k_neighbors=5, random_state=42),
    'ADASYN': ManualADASYN(n_neighbors=5, random_state=42),
    'SMOTE + Tomek': ManualSMOTETomek(k_neighbors=5, random_state=42),
    'Random Undersampling': ManualRandomUnderSampler(random_state=42),
}

print(f"âœ… Defined {len(sampling_strategies)} strategies (Manual Implementation)")

# ============================================================
# 6. COMPARE ALL TECHNIQUES
# ============================================================
print("\n" + "=" * 70)
print("ğŸ”„ COMPARING SAMPLING TECHNIQUES")
print("=" * 70)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

xgb_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.03,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'early_stopping_rounds': 50
}

all_results = {}

for strategy_name, sampler in sampling_strategies.items():
    print(f"\n{'â”€' * 70}")
    print(f"ğŸ”§ {strategy_name}")
    print(f"{'â”€' * 70}")
    
    oof_proba = np.zeros(len(X))
    oof_class = np.zeros(len(X))
    test_proba = np.zeros(len(X_test))
    fold_aucs = []
    samples_info = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_fe, y)):
        X_train_fold = X_fe.iloc[train_idx].copy()
        X_val_fold = X_fe.iloc[val_idx].copy()
        y_train_fold = y[train_idx].copy()
        y_val_fold = y[val_idx]
        
        X_train_proc = preprocessor.fit_transform(X_train_fold)
        X_val_proc = preprocessor.transform(X_val_fold)
        X_test_proc = preprocessor.transform(X_test_fe)
        
        original_count = len(y_train_fold)
        
        if sampler is not None:
            try:
                X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_proc, y_train_fold)
                resampled_count = len(y_train_resampled)
                
                if fold == 0:
                    print(f"   Fold 1: {original_count:,} â†’ {resampled_count:,} samples")
                    print(f"   Class dist: {dict(Counter(y_train_resampled.astype(int)))}")
            except Exception as e:
                print(f"   âš ï¸� Fold {fold+1} error: {str(e)[:40]}")
                X_train_resampled, y_train_resampled = X_train_proc, y_train_fold
                resampled_count = original_count
        else:
            X_train_resampled, y_train_resampled = X_train_proc, y_train_fold
            resampled_count = original_count
        
        samples_info.append(resampled_count)
        
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_val_proc, y_val_fold)],
            verbose=False
        )
        
        val_proba = model.predict_proba(X_val_proc)[:, 1]
        val_class = model.predict(X_val_proc)
        
        oof_proba[val_idx] = val_proba
        oof_class[val_idx] = val_class
        test_proba += model.predict_proba(X_test_proc)[:, 1] / n_splits
        
        fold_aucs.append(roc_auc_score(y_val_fold, val_proba))
    
    cm = confusion_matrix(y, oof_class)
    tn, fp, fn, tp = cm.ravel()
    
    results = {
        'oof_proba': oof_proba,
        'oof_class': oof_class,
        'test_proba': test_proba,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'avg_samples': np.mean(samples_info),
        'accuracy': accuracy_score(y, oof_class),
        'balanced_accuracy': balanced_accuracy_score(y, oof_class),
        'precision_0': precision_score(y, oof_class, pos_label=0),
        'precision_1': precision_score(y, oof_class, pos_label=1),
        'recall_0': recall_score(y, oof_class, pos_label=0),
        'recall_1': recall_score(y, oof_class, pos_label=1),
        'f1_0': f1_score(y, oof_class, pos_label=0),
        'f1_1': f1_score(y, oof_class, pos_label=1),
        'f1_macro': f1_score(y, oof_class, average='macro'),
        'f1_weighted': f1_score(y, oof_class, average='weighted'),
        'roc_auc': roc_auc_score(y, oof_proba),
        'pr_auc': average_precision_score(y, oof_proba),
        'mcc': matthews_corrcoef(y, oof_class),
        'kappa': cohen_kappa_score(y, oof_class),
    }
    
    all_results[strategy_name] = results
    
    print(f"   ğŸ“Š ROC-AUC: {results['roc_auc']:.5f} | F1 Macro: {results['f1_macro']:.5f}")
    print(f"   ğŸ“Š Recall(0): {results['recall_0']:.5f} | Recall(1): {results['recall_1']:.5f}")

# ============================================================
# 7. RESULTS COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š RESULTS COMPARISON")
print("=" * 70)

comparison_data = []
for name, res in all_results.items():
    comparison_data.append({
        'Strategy': name,
        'Samples': f"{res['avg_samples']:,.0f}",
        'ROC-AUC': res['roc_auc'],
        'PR-AUC': res['pr_auc'],
        'Bal Acc': res['balanced_accuracy'],
        'Prec(0)': res['precision_0'],
        'Rec(0)': res['recall_0'],
        'F1(0)': res['f1_0'],
        'Prec(1)': res['precision_1'],
        'Rec(1)': res['recall_1'],
        'F1(1)': res['f1_1'],
        'F1 Macro': res['f1_macro'],
        'MCC': res['mcc']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_sorted = comparison_df.sort_values('F1 Macro', ascending=False)

print("\nğŸ“Œ Sorted by F1 Macro:")
print("-" * 120)
print(comparison_sorted.to_string(index=False))

# ============================================================
# 8. BEST STRATEGY
# ============================================================
print("\n" + "=" * 70)
print("ğŸ�† BEST STRATEGY BY METRIC")
print("=" * 70)

for metric in ['ROC-AUC', 'F1 Macro', 'Rec(0)', 'Bal Acc', 'MCC']:
    best_idx = comparison_df[metric].idxmax()
    best = comparison_df.loc[best_idx]
    print(f"   {metric:<12}: {best['Strategy']:<25} ({best[metric]:.5f})")

# ============================================================
# 9. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“Š GENERATING VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 15))

# Plot 1: ROC-AUC Comparison
ax1 = fig.add_subplot(2, 3, 1)
strategies = list(all_results.keys())
roc_aucs = [all_results[s]['roc_auc'] for s in strategies]
colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

bars = ax1.barh(range(len(strategies)), roc_aucs, color=colors)
ax1.set_yticks(range(len(strategies)))
ax1.set_yticklabels([s.replace('(Baseline)', '\n(Baseline)') for s in strategies], fontsize=9)
ax1.set_xlabel('ROC-AUC', fontsize=12)
ax1.set_title('ROC-AUC by Strategy', fontsize=14, fontweight='bold')
ax1.set_xlim([min(roc_aucs) - 0.01, max(roc_aucs) + 0.01])

for bar, val in zip(bars, roc_aucs):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)

# Plot 2: Recall Comparison
ax2 = fig.add_subplot(2, 3, 2)
recall_0 = [all_results[s]['recall_0'] for s in strategies]
recall_1 = [all_results[s]['recall_1'] for s in strategies]

x = np.arange(len(strategies))
width = 0.35

ax2.bar(x - width/2, recall_0, width, label='Recall (Class 0 - Minority)', color='#ff6b6b')
ax2.bar(x + width/2, recall_1, width, label='Recall (Class 1 - Majority)', color='#4ecdc4')
ax2.set_xticks(x)
ax2.set_xticklabels([s.split()[0] for s in strategies], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Recall')
ax2.set_title('Recall by Class', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=8)
ax2.set_ylim([0, 1.1])

# Plot 3: F1 Comparison
ax3 = fig.add_subplot(2, 3, 3)
f1_0 = [all_results[s]['f1_0'] for s in strategies]
f1_1 = [all_results[s]['f1_1'] for s in strategies]
f1_macro = [all_results[s]['f1_macro'] for s in strategies]

width = 0.25
ax3.bar(x - width, f1_0, width, label='F1 (Class 0)', color='#ff6b6b')
ax3.bar(x, f1_1, width, label='F1 (Class 1)', color='#4ecdc4')
ax3.bar(x + width, f1_macro, width, label='F1 (Macro)', color='#9b59b6')
ax3.set_xticks(x)
ax3.set_xticklabels([s.split()[0] for s in strategies], rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('F1-Score')
ax3.set_title('F1-Score by Strategy', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right', fontsize=8)

# Plot 4: ROC Curves
ax4 = fig.add_subplot(2, 3, 4)
colors_line = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

for (name, res), color in zip(all_results.items(), colors_line):
    fpr, tpr, _ = roc_curve(y, res['oof_proba'])
    ax4.plot(fpr, tpr, color=color, lw=2, label=f"{name.split()[0]} ({res['roc_auc']:.4f})")

ax4.plot([0, 1], [0, 1], 'k--', lw=1)
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax4.legend(loc='lower right', fontsize=7)

# Plot 5: PR Curves
ax5 = fig.add_subplot(2, 3, 5)

for (name, res), color in zip(all_results.items(), colors_line):
    prec, rec, _ = precision_recall_curve(y, res['oof_proba'])
    ax5.plot(rec, prec, color=color, lw=2, label=f"{name.split()[0]} ({res['pr_auc']:.4f})")

ax5.axhline(y=y.mean(), color='gray', linestyle='--', label=f'Baseline ({y.mean():.2f})')
ax5.set_xlabel('Recall')
ax5.set_ylabel('Precision')
ax5.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax5.legend(loc='lower left', fontsize=7)

# Plot 6: Summary Metrics
ax6 = fig.add_subplot(2, 3, 6)
metrics = ['ROC-AUC', 'Bal Acc', 'F1 Macro', 'MCC']
baseline_vals = [all_results['No Sampling (Baseline)'][m.lower().replace(' ', '_').replace('-', '_')] 
                 if m.lower().replace(' ', '_').replace('-', '_') in all_results['No Sampling (Baseline)']
                 else all_results['No Sampling (Baseline)']['roc_auc'] for m in metrics]

# Get actual values
baseline_vals = [
    all_results['No Sampling (Baseline)']['roc_auc'],
    all_results['No Sampling (Baseline)']['balanced_accuracy'],
    all_results['No Sampling (Baseline)']['f1_macro'],
    all_results['No Sampling (Baseline)']['mcc']
]

smote_vals = [
    all_results['SMOTE']['roc_auc'],
    all_results['SMOTE']['balanced_accuracy'],
    all_results['SMOTE']['f1_macro'],
    all_results['SMOTE']['mcc']
]

x = np.arange(len(metrics))
width = 0.35

ax6.bar(x - width/2, baseline_vals, width, label='Baseline', color='#3498db')
ax6.bar(x + width/2, smote_vals, width, label='SMOTE', color='#e74c3c')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics)
ax6.set_ylabel('Score')
ax6.set_title('Baseline vs SMOTE', fontsize=14, fontweight='bold')
ax6.legend()
ax6.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('smote_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("âœ… Saved: smote_comparison.png")

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ğŸ“‹ FINAL SUMMARY")
print("=" * 70)

baseline = all_results['No Sampling (Baseline)']
best_f1_strategy = comparison_df.loc[comparison_df['F1 Macro'].idxmax(), 'Strategy']
best = all_results[best_f1_strategy]

print(f"""
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
â”‚                     SMOTE ANALYSIS SUMMARY                               â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  ğŸ“Š BASELINE                        â”‚  ğŸ�† BEST: {best_f1_strategy:<20}  â”‚
â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€                     â”‚  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€   â”‚
â”‚  ROC-AUC:    {baseline['roc_auc']:.4f}                â”‚  ROC-AUC:    {best['roc_auc']:.4f} ({best['roc_auc']-baseline['roc_auc']:+.4f})      â”‚
â”‚  F1 Macro:   {baseline['f1_macro']:.4f}                â”‚  F1 Macro:   {best['f1_macro']:.4f} ({best['f1_macro']-baseline['f1_macro']:+.4f})      â”‚
â”‚  Recall(0):  {baseline['recall_0']:.4f}                â”‚  Recall(0):  {best['recall_0']:.4f} ({best['recall_0']-baseline['recall_0']:+.4f})      â”‚
â”‚  Recall(1):  {baseline['recall_1']:.4f}                â”‚  Recall(1):  {best['recall_1']:.4f} ({best['recall_1']-baseline['recall_1']:+.4f})      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

# ============================================================
# 11. CREATE SUBMISSION
# ============================================================
print("=" * 70)
print("ğŸ’¾ CREATING SUBMISSION")
print("=" * 70)

submission['loan_paid_back'] = best['test_proba']
submission.to_csv('submission.csv', index=False)

print(f"âœ… Saved using: {best_f1_strategy}")
print(f"\nğŸ“Œ Stats: Mean={best['test_proba'].mean():.4f}, Std={best['test_proba'].std():.4f}")
print(submission.head())

print("\n" + "=" * 70)
print("âœ… COMPLETE!")
print("=" * 70)


pip install threadpoolctl==3.1.0




