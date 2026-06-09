"""
=============================================================================
COMPREHENSIVE EDA: Road Accident Risk Prediction
Kaggle Playground Series - October 2025
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("="*80)
print("ROAD ACCIDENT RISK PREDICTION - EXPLORATORY DATA ANALYSIS")
print("="*80)


# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "="*80)
print("1. LOADING DATA")
print("="*80)

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"\nâœ“ Train shape: {train.shape}")
print(f"âœ“ Test shape: {test.shape}")
print(f"âœ“ Sample submission shape: {sample_sub.shape}")


# =============================================================================
# 2. BASIC DATA OVERVIEW
# =============================================================================
print("\n" + "="*80)
print("2. BASIC DATA OVERVIEW")
print("="*80)

print("\n--- First 5 rows of Training Data ---")
print(train.head())

print("\n--- Data Types ---")
print(train.dtypes.value_counts())
print("\nColumn Types:")
print(train.dtypes)

print("\n--- Basic Info ---")
train.info()


# =============================================================================
# 3. TARGET VARIABLE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("3. TARGET VARIABLE ANALYSIS (accident_risk)")
print("="*80)

target_stats = train['accident_risk'].describe()
print("\n--- Target Statistics ---")
print(target_stats)

print(f"\n--- Target Distribution Metrics ---")
print(f"Skewness: {skew(train['accident_risk']):.4f}")
print(f"Kurtosis: {kurtosis(train['accident_risk']):.4f}")
print(f"Range: [{train['accident_risk'].min():.4f}, {train['accident_risk'].max():.4f}]")
print(f"IQR: {train['accident_risk'].quantile(0.75) - train['accident_risk'].quantile(0.25):.4f}")

# Visualize target
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Target Variable (accident_risk) Analysis', fontsize=16, fontweight='bold')

# Histogram
axes[0, 0].hist(train['accident_risk'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(train['accident_risk'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {train["accident_risk"].mean():.3f}')
axes[0, 0].axvline(train['accident_risk'].median(), color='green', linestyle='--', 
                    label=f'Median: {train["accident_risk"].median():.3f}')
axes[0, 0].set_xlabel('Accident Risk')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Accident Risk')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(train['accident_risk'], vert=True)
axes[0, 1].set_ylabel('Accident Risk')
axes[0, 1].set_title('Box Plot of Accident Risk')
axes[0, 1].grid(True, alpha=0.3)

# KDE plot
train['accident_risk'].plot(kind='kde', ax=axes[1, 0], linewidth=2)
axes[1, 0].set_xlabel('Accident Risk')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Kernel Density Estimation')
axes[1, 0].grid(True, alpha=0.3)

# QQ plot
stats.probplot(train['accident_risk'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# 4. MISSING VALUES ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("4. MISSING VALUES ANALYSIS")
print("="*80)

train_missing = train.isnull().sum()
test_missing = test.isnull().sum()

print("\n--- Training Data Missing Values ---")
if train_missing.sum() == 0:
    print("âœ“ No missing values in training data!")
else:
    missing_df = pd.DataFrame({
        'Column': train_missing[train_missing > 0].index,
        'Missing_Count': train_missing[train_missing > 0].values,
        'Percentage': (train_missing[train_missing > 0].values / len(train) * 100)
    }).sort_values('Missing_Count', ascending=False)
    print(missing_df)

print("\n--- Test Data Missing Values ---")
if test_missing.sum() == 0:
    print("âœ“ No missing values in test data!")
else:
    missing_test_df = pd.DataFrame({
        'Column': test_missing[test_missing > 0].index,
        'Missing_Count': test_missing[test_missing > 0].values,
        'Percentage': (test_missing[test_missing > 0].values / len(test) * 100)
    }).sort_values('Missing_Count', ascending=False)
    print(missing_test_df)



# =============================================================================
# 5. FEATURE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("5. FEATURE ANALYSIS")
print("="*80)

# Identify feature types
numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numeric_features:
    numeric_features.remove('id')
if 'accident_risk' in numeric_features:
    numeric_features.remove('accident_risk')

categorical_features = train.select_dtypes(include=['object']).columns.tolist()

print(f"\nâœ“ Numeric Features ({len(numeric_features)}): {numeric_features}")
print(f"âœ“ Categorical Features ({len(categorical_features)}): {categorical_features}")

# Numeric features statistics
if numeric_features:
    print("\n--- Numeric Features Statistics ---")
    print(train[numeric_features].describe().T)
    
    # Check for constant features
    print("\n--- Checking for Low Variance Features ---")
    for col in numeric_features:
        unique_ratio = train[col].nunique() / len(train)
        if unique_ratio < 0.01:
            print(f"âš  {col}: {train[col].nunique()} unique values ({unique_ratio*100:.2f}% of data)")
    
    # Visualize numeric features distributions
    n_cols = 3
    n_rows = (len(numeric_features) + n_cols - 1) // n_cols
    
    if len(numeric_features) > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if len(numeric_features) > 1 else [axes]
        
        for idx, col in enumerate(numeric_features):
            train[col].hist(bins=50, ax=axes[idx], edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col}\nSkew: {skew(train[col]):.2f}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_features), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Numeric Features Distributions', fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.show()

# Categorical features analysis
if categorical_features:
    print("\n--- Categorical Features Analysis ---")
    for col in categorical_features:
        print(f"\n{col}:")
        print(f"  Unique values: {train[col].nunique()}")
        print(f"  Top 10 categories:\n{train[col].value_counts().head(10)}")
        
        # Check if same categories in train and test
        train_cats = set(train[col].unique())
        test_cats = set(test[col].unique())
        
        only_in_train = train_cats - test_cats
        only_in_test = test_cats - train_cats
        
        if only_in_train:
            print(f"  âš  Categories only in train: {only_in_train}")
        if only_in_test:
            print(f"  âš  Categories only in test: {only_in_test}")
    
    # Visualize categorical features
    n_cols = 2
    n_rows = (len(categorical_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if len(categorical_features) > 1 else [axes]
    
    for idx, col in enumerate(categorical_features):
        value_counts = train[col].value_counts().head(15)
        value_counts.plot(kind='bar', ax=axes[idx], color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{col} (Top 15 categories)')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(categorical_features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Categorical Features Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()



# =============================================================================
# 6. CORRELATION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("6. CORRELATION ANALYSIS")
print("="*80)

if numeric_features:
    # Correlation with target
    correlations = train[numeric_features + ['accident_risk']].corr()['accident_risk'].drop('accident_risk').sort_values(ascending=False)
    
    print("\n--- Correlation with Target (accident_risk) ---")
    print(correlations)
    
    # Visualize correlations
    plt.figure(figsize=(10, 6))
    correlations.plot(kind='barh', color='coral', edgecolor='black')
    plt.xlabel('Correlation with accident_risk')
    plt.title('Feature Correlation with Target Variable', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    # Full correlation matrix
    if len(numeric_features) > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = train[numeric_features + ['accident_risk']].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Identify highly correlated features
        print("\n--- Highly Correlated Feature Pairs (|corr| > 0.8) ---")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], 
                                           corr_matrix.columns[j], 
                                           corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
        else:
            print("  âœ“ No highly correlated feature pairs found")



# =============================================================================
# 7. CATEGORICAL vs TARGET ANALYSIS
# =============================================================================
if categorical_features:
    print("\n" + "="*80)
    print("7. CATEGORICAL FEATURES vs TARGET ANALYSIS")
    print("="*80)
    
    for col in categorical_features:
        print(f"\n--- {col} vs accident_risk ---")
        grouped = train.groupby(col)['accident_risk'].agg(['mean', 'median', 'std', 'count']).sort_values('mean', ascending=False)
        print(grouped.head(15))
    
    # Visualize relationship
    n_cols = 2
    n_rows = (len(categorical_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if len(categorical_features) > 1 else [axes]
    
    for idx, col in enumerate(categorical_features):
        grouped = train.groupby(col)['accident_risk'].mean().sort_values(ascending=False).head(15)
        grouped.plot(kind='bar', ax=axes[idx], color='teal', edgecolor='black')
        axes[idx].set_title(f'Mean Accident Risk by {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Mean Accident Risk')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(categorical_features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Categorical Features Impact on Target', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 8. OUTLIER DETECTION
# =============================================================================
if numeric_features:
    print("\n" + "="*80)
    print("8. OUTLIER DETECTION (IQR Method)")
    print("="*80)
    
    for col in numeric_features:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
        outlier_pct = len(outliers) / len(train) * 100
        
        print(f"\n{col}:")
        print(f"  Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")
        print(f"  Outliers: {len(outliers)} ({outlier_pct:.2f}%)")


# =============================================================================
# 9. TRAIN vs TEST DISTRIBUTION COMPARISON
# =============================================================================
print("\n" + "="*80)
print("9. TRAIN vs TEST DISTRIBUTION COMPARISON")
print("="*80)

common_features = [col for col in train.columns if col in test.columns and col != 'accident_risk']

if numeric_features:
    print("\n--- Numeric Features Distribution Comparison ---")
    comparison_stats = []
    
    for col in numeric_features:
        if col in test.columns:
            train_mean = train[col].mean()
            test_mean = test[col].mean()
            train_std = train[col].std()
            test_std = test[col].std()
            
            comparison_stats.append({
                'Feature': col,
                'Train_Mean': train_mean,
                'Test_Mean': test_mean,
                'Mean_Diff_%': abs(train_mean - test_mean) / train_mean * 100 if train_mean != 0 else 0,
                'Train_Std': train_std,
                'Test_Std': test_std
            })
    
    comp_df = pd.DataFrame(comparison_stats)
    print(comp_df.to_string(index=False))
    
    # Visualize train vs test
    n_plot = min(6, len([f for f in numeric_features if f in test.columns]))
    if n_plot > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for col in numeric_features:
            if col in test.columns and plot_idx < 6:
                axes[plot_idx].hist(train[col], bins=50, alpha=0.6, label='Train', edgecolor='black')
                axes[plot_idx].hist(test[col], bins=50, alpha=0.6, label='Test', edgecolor='black')
                axes[plot_idx].set_title(f'{col}')
                axes[plot_idx].set_xlabel(col)
                axes[plot_idx].set_ylabel('Frequency')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, 6):
            axes[idx].axis('off')
        
        plt.suptitle('Train vs Test Distribution Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


# =============================================================================
# 10. FINAL SUMMARY & KEY INSIGHTS
# =============================================================================
print("\n" + "="*80)
print("10. COMPREHENSIVE DATA SUMMARY & KEY INSIGHTS")
print("="*80)

summary = {
    'Dataset Info': {
        'Train samples': len(train),
        'Test samples': len(test),
        'Total features': len(train.columns) - 2,  # excluding id and target
        'Numeric features': len(numeric_features),
        'Categorical features': len(categorical_features)
    },
    'Target Variable (accident_risk)': {
        'Mean': train['accident_risk'].mean(),
        'Median': train['accident_risk'].median(),
        'Std': train['accident_risk'].std(),
        'Min': train['accident_risk'].min(),
        'Max': train['accident_risk'].max(),
        'Skewness': skew(train['accident_risk']),
        'Kurtosis': kurtosis(train['accident_risk'])
    },
    'Data Quality': {
        'Train missing values': train.isnull().sum().sum(),
        'Test missing values': test.isnull().sum().sum(),
        'Train duplicates': train.duplicated().sum(),
        'Test duplicates': test.duplicated().sum()
    }
}

print("\n" + "="*50)
print("DATASET INFORMATION")
print("="*50)
for key, value in summary['Dataset Info'].items():
    print(f"{key:.<30} {value}")

print("\n" + "="*50)
print("TARGET VARIABLE STATISTICS")
print("="*50)
for key, value in summary['Target Variable (accident_risk)'].items():
    if isinstance(value, float):
        print(f"{key:.<30} {value:.6f}")
    else:
        print(f"{key:.<30} {value}")

print("\n" + "="*50)
print("DATA QUALITY")
print("="*50)
for key, value in summary['Data Quality'].items():
    print(f"{key:.<30} {value}")

print("\n" + "="*50)
print("FEATURE LISTS")
print("="*50)
print(f"\nNumeric Features ({len(numeric_features)}):")
for i, feat in enumerate(numeric_features, 1):
    print(f"  {i}. {feat}")

if categorical_features:
    print(f"\nCategorical Features ({len(categorical_features)}):")
    for i, feat in enumerate(categorical_features, 1):
        print(f"  {i}. {feat}")

if numeric_features:
    print("\n" + "="*50)
    print("TOP CORRELATIONS WITH TARGET")
    print("="*50)
    top_corr = train[numeric_features + ['accident_risk']].corr()['accident_risk'].drop('accident_risk').abs().sort_values(ascending=False).head(10)
    for feat, corr in top_corr.items():
        print(f"{feat:.<30} {corr:.4f}")

print("\n" + "="*50)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*50)

insights = []

# Check target distribution
if abs(skew(train['accident_risk'])) < 0.5:
    insights.append("âœ“ Target is relatively symmetric - good for most regression algorithms")
else:
    insights.append("âš  Target shows skewness - consider transformation or robust models")

# Check missing values
if train.isnull().sum().sum() == 0 and test.isnull().sum().sum() == 0:
    insights.append("âœ“ No missing values - data is clean")
else:
    insights.append("âš  Missing values detected - implement imputation strategy")

# Check duplicates
if train.duplicated().sum() > 0:
    insights.append(f"âš  {train.duplicated().sum()} duplicate rows in training data")

# Feature insights
if numeric_features:
    insights.append(f"âœ“ {len(numeric_features)} numeric features available for modeling")
if categorical_features:
    insights.append(f"âœ“ {len(categorical_features)} categorical features - consider encoding strategies")

# Correlation insights
if numeric_features and len(numeric_features) > 1:
    max_corr = train[numeric_features].corr().abs().values[np.triu_indices_from(train[numeric_features].corr().abs().values, k=1)].max()
    if max_corr > 0.9:
        insights.append("âš  Very high feature correlation detected - consider dimensionality reduction")

insights.append("\nðŸ“Š Suggested Next Steps:")
insights.append("  1. Feature Engineering: Create interaction terms, polynomial features")
insights.append("  2. Encoding: Handle categorical variables (Target/Label/One-Hot encoding)")
insights.append("  3. Scaling: Normalize/Standardize numeric features")
insights.append("  4. Baseline Model: Start with simple models (Linear Regression, Ridge)")
insights.append("  5. Advanced Models: Gradient Boosting (XGBoost, LightGBM, CatBoost)")
insights.append("  6. Cross-Validation: Implement robust CV strategy (5-10 folds)")
insights.append("  7. Ensemble: Combine multiple models for better performance")

for insight in insights:
    print(insight)

print("\n" + "="*80)
print("EDA COMPLETE! Ready for modeling strategy discussion.")
print("="*80)




