# Core data manipulation and analysis
import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List, Dict, Any

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr, spearmanr, normaltest, jarque_bera
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
warnings.filterwarnings('ignore')

# Plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("ğŸ“¦ All libraries imported successfully!")


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sample_sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")



print("=== BASIC DATA OVERVIEW ===")
print(f"Training columns: {train.shape[1]}")
print(f"Test columns: {test.shape[1]}")

# Check common columns
train_cols = set(train.columns)
test_cols = set(test.columns)
common_cols = train_cols.intersection(test_cols)
train_only = train_cols - test_cols
test_only = test_cols - train_cols

print(f"Common columns: {len(common_cols)}")
print(f"Train only: {list(train_only)}")
print(f"Test only: {list(test_only)}")

# Data types
print(f"\nData types:")
print(train.dtypes.value_counts())

# Memory usage
train_memory = train.memory_usage(deep=True).sum() / 1024**2
test_memory = test.memory_usage(deep=True).sum() / 1024**2
print(f"\nMemory usage:")
print(f"Train: {train_memory:.1f} MB")
print(f"Test: {test_memory:.1f} MB")


target = train['label']

print("=== TARGET VARIABLE ANALYSIS ===")
print(f"Count: {len(target):,}")
print(f"Mean: {target.mean():.6f}")
print(f"Median: {target.median():.6f}")
print(f"Std: {target.std():.6f}")
print(f"Min: {target.min():.6f}")
print(f"Max: {target.max():.6f}")
print(f"Skewness: {target.skew():.6f}")
print(f"Kurtosis: {target.kurtosis():.6f}")

# Normality test
try:
    stat, p_value = normaltest(target.dropna())
    print(f"Normality test p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("Target is NOT normally distributed")
    else:
        print("Target appears normally distributed")
except:
    print("Could not perform normality test")

# Outliers using IQR
Q1, Q3 = target.quantile(0.25), target.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = target[(target < lower_bound) | (target > upper_bound)]
print(f"Outliers (IQR method): {len(outliers)} ({len(outliers)/len(target)*100:.2f}%)")



fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
fig.suptitle('Target Variable Analysis', fontsize=18, y=1.02, fontweight='bold')

# Histogram with KDE 
sns.histplot(target, bins=50, kde=True, ax=axes[0,0], color='skyblue', edgecolor='navy', alpha=0.7)
axes[0,0].axvline(target.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {target.mean():.4f}')
axes[0,0].axvline(target.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {target.median():.4f}')
axes[0,0].set_title('Distribution Histogram', fontsize=14, pad=10)
axes[0,0].set_xlabel('Target Value', fontsize=12)
axes[0,0].set_ylabel('Frequency', fontsize=12)
axes[0,0].legend(fontsize=10, framealpha=0.9)
axes[0,0].grid(True, alpha=0.3)
# Box plot with annotations
box = axes[0,1].boxplot(target, patch_artist=True, 
                        boxprops=dict(facecolor='lightgreen', alpha=0.7),
                        whiskerprops=dict(color='green', linewidth=1.5),
                        capprops=dict(color='green', linewidth=1.5),
                        medianprops=dict(color='darkred', linewidth=2))

# summary statistics
stats_text = f"""
Min: {np.min(target):.2f}
Q1: {np.percentile(target, 25):.2f}
Median: {np.median(target):.2f}
Q3: {np.percentile(target, 75):.2f}
Max: {np.max(target):.2f}
IQR: {np.percentile(target, 75) - np.percentile(target, 25):.2f}
"""
axes[0,1].text(1.2, 0.5, stats_text, transform=axes[0,1].transAxes, 
              bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
axes[0,1].set_title('Distribution Box Plot', fontsize=14, pad=10)
axes[0,1].set_ylabel('Target Value', fontsize=12)
axes[0,1].grid(True, alpha=0.3)

# Time series with rolling average
sample_size = min(1000, len(target))
rolling_window = sample_size // 20  # 5% of sample size

axes[1,0].plot(target[:sample_size], alpha=0.5, color='purple', label='Raw data')
axes[1,0].plot(pd.Series(target[:sample_size]).rolling(rolling_window).mean(), 
              color='darkorange', linewidth=2, label=f'Rolling mean (window={rolling_window})')
axes[1,0].set_title(f'Time Series (First {sample_size} points)', fontsize=14, pad=10)
axes[1,0].set_xlabel('Index', fontsize=12)
axes[1,0].set_ylabel('Target Value', fontsize=12)
axes[1,0].legend(fontsize=10)
axes[1,0].grid(True, alpha=0.3)

# Q-Q plot with RÂ² value
stats.probplot(target, dist="norm", plot=axes[1,1])
axes[1,1].lines[0].set_markerfacecolor('blue')
axes[1,1].lines[0].set_markersize(4.0)
axes[1,1].lines[1].set_color('red')
axes[1,1].lines[1].set_linewidth(2.0)

# Calculate RÂ² for the Q-Q plot
(osm, osr), (slope, intercept, r) = stats.probplot(target, dist="norm")
axes[1,1].text(0.05, 0.9, f'RÂ² = {r**2:.3f}', transform=axes[1,1].transAxes,
              bbox=dict(facecolor='white', alpha=0.8))
axes[1,1].set_title('Normality Q-Q Plot', fontsize=14, pad=10)
axes[1,1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print("=== MISSING VALUES ANALYSIS ===")

# Training data
train_missing = train.isnull().sum()
train_missing_pct = (train_missing / len(train)) * 100
train_missing_df = pd.DataFrame({
    'Missing_Count': train_missing,
    'Missing_Percent': train_missing_pct
})
train_missing_df = train_missing_df[train_missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print(f"Training data:")
print(f"Columns with missing values: {len(train_missing_df)}")
if len(train_missing_df) > 0:
    print("Top 10 columns with missing values:")
    print(train_missing_df.head(10))
else:
    print("âœ… No missing values in training data!")

# Test data
test_missing = test.isnull().sum()
test_missing_pct = (test_missing / len(test)) * 100
test_missing_df = pd.DataFrame({
    'Missing_Count': test_missing,
    'Missing_Percent': test_missing_pct
})
test_missing_df = test_missing_df[test_missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print(f"\nTest data:")
print(f"Columns with missing values: {len(test_missing_df)}")
if len(test_missing_df) > 0:
    print("Top 10 columns with missing values:")
    print(test_missing_df.head(10))
else:
    print("âœ… No missing values in test data!")


print("=== FEATURE CORRELATION ANALYSIS ===")

# Get feature columns (exclude target)
feature_cols = [col for col in train.columns if col != 'label']
print(f"Analyzing {len(feature_cols)} features")

# Calculate correlations with target
correlations = []
target = train['label']

for col in feature_cols:
    try:
        corr, p_value = pearsonr(train[col].fillna(0), target)
        if not np.isnan(corr):
            correlations.append({
                'Feature': col,
                'Correlation': corr,
                'Abs_Correlation': abs(corr),
                'P_Value': p_value
            })
    except:
        continue

# Create correlation dataframe
corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)

print(f"Successfully calculated correlations for {len(corr_df)} features")
print(f"\nTop 20 features by absolute correlation:")
print("-" * 60)
for i, row in corr_df.head(20).iterrows():
    significance = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
    print(f"{row['Feature']:>15}: {row['Correlation']:>8.6f} {significance}")

# Correlation strength distribution
strong_corr = corr_df[corr_df['Abs_Correlation'] > 0.1]
moderate_corr = corr_df[(corr_df['Abs_Correlation'] > 0.05) & (corr_df['Abs_Correlation'] <= 0.1)]
weak_corr = corr_df[corr_df['Abs_Correlation'] <= 0.05]

print(f"\nCorrelation strength distribution:")
print(f"Strong (|r| > 0.1): {len(strong_corr)} features")
print(f"Moderate (0.05 < |r| â‰¤ 0.1): {len(moderate_corr)} features")
print(f"Weak (|r| â‰¤ 0.05): {len(weak_corr)} features")


# Top correlations bar plot
top_20 = corr_df.head(20)

plt.figure(figsize=(12, 8))
colors = ['red' if x < 0 else '#0010d9' for x in top_20['Correlation']]
plt.barh(range(len(top_20)), top_20['Correlation'], color=colors, alpha=0.7)
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Correlation with Target')
plt.title('Top 20 Features - Correlation with Target')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Correlation distribution
plt.figure(figsize=(10, 8))
plt.hist(corr_df['Correlation'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(corr_df['Correlation'].mean(), color='red', linestyle='--', 
           label=f'Mean: {corr_df["Correlation"].mean():.4f}')
plt.xlabel('Correlation with Target')
plt.ylabel('Frequency')
plt.title('Distribution of Feature Correlations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Analyze top 12 features distributions
top_features = corr_df.head(12)['Feature'].tolist()

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Top 12 Features - Distribution Analysis', fontsize=16)
axes = axes.flatten()

for i, feature in enumerate(top_features):
    data = train[feature].dropna()
    
    # Histogram
    axes[i].hist(data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    
    # Add mean and median lines
    mean_val = data.mean()
    median_val = data.median()
    axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
    axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')
    
    axes[i].set_title(f'{feature}\nSkew: {data.skew():.3f}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("=== FEATURE STATISTICS SUMMARY ===")

# Get top 20 features for detailed analysis
top_20_features = corr_df.head(20)['Feature'].tolist()

# Create summary statistics
summary_stats = []
for feature in top_20_features:
    data = train[feature].dropna()
    stats_dict = {
        'Feature': feature,
        'Count': len(data),
        'Mean': data.mean(),
        'Std': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        'Skewness': data.skew(),
        'Kurtosis': data.kurtosis(),
        'Correlation': corr_df[corr_df['Feature'] == feature]['Correlation'].iloc[0]
    }
    summary_stats.append(stats_dict)

summary_df = pd.DataFrame(summary_stats)
print("Top 20 Features Summary Statistics:")
summary_df.round(4)


# Create correlation heatmap for top features
top_15_features = corr_df.head(15)['Feature'].tolist() + ['label']

# Sample data if too large
if len(train) > 5000:
    sample_data = train[top_15_features].sample(n=5000, random_state=42)
else:
    sample_data = train[top_15_features]

correlation_matrix = sample_data.corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='crest', 
            center=0,
            square=True, 
            fmt='.3f',
            cbar_kws={"shrink": .8})

plt.title('Correlation Heatmap - Top 15 Features + Target', fontsize=14)
plt.tight_layout()
plt.show()


print("=== FEATURE VALUE RANGES ANALYSIS ===")

# Analyze value ranges for top features
top_10_features = corr_df.head(10)['Feature'].tolist()

range_analysis = []
for feature in top_10_features:
    data = train[feature].dropna()
    range_dict = {
        'Feature': feature,
        'Min': data.min(),
        'Max': data.max(),
        'Range': data.max() - data.min(),
        '1st_Quartile': data.quantile(0.25),
        '3rd_Quartile': data.quantile(0.75),
        'IQR': data.quantile(0.75) - data.quantile(0.25),
        'Correlation': corr_df[corr_df['Feature'] == feature]['Correlation'].iloc[0]
    }
    range_analysis.append(range_dict)

range_df = pd.DataFrame(range_analysis)
print("Top 10 Features - Value Ranges:")
print(range_df.round(6))

# Visualize ranges
plt.figure(figsize=(12, 8))
features = range_df['Feature']
ranges = range_df['Range']
colors = ['red' if x < 0 else '#0010d9' for x in range_df['Correlation']]

plt.barh(range(len(features)), ranges, color=colors, alpha=0.7)
plt.yticks(range(len(features)), features)
plt.xlabel('Value Range')
plt.title('Top 10 Features - Value Ranges (Color = Correlation Sign)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


 print("=== DATA QUALITY ASSESSMENT ===")

# Check for constant features
constant_features = []
for col in train.columns:
    if col != 'label':
        if train[col].nunique() <= 1:
            constant_features.append(col)

print(f"Constant features (nunique <= 1): {len(constant_features)}")
if constant_features:
    print("Constant features:", constant_features[:10])

# Check for highly skewed features
highly_skewed = []
for feature in corr_df.head(20)['Feature']:
    skew_val = train[feature].skew()
    if abs(skew_val) > 2:
        highly_skewed.append((feature, skew_val))

print(f"\nHighly skewed features (|skew| > 2): {len(highly_skewed)}")
for feature, skew_val in highly_skewed[:10]:
    print(f"  {feature}: {skew_val:.3f}")

# Check feature correlations between themselves
print(f"\nInter-feature correlation analysis:")
top_features_for_corr = corr_df.head(10)['Feature'].tolist()
feature_corr_matrix = train[top_features_for_corr].corr()

# Find highly correlated feature pairs
high_corr_pairs = []
for i in range(len(feature_corr_matrix.columns)):
    for j in range(i+1, len(feature_corr_matrix.columns)):
        corr_val = feature_corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:  # High correlation threshold
            high_corr_pairs.append((
                feature_corr_matrix.columns[i], 
                feature_corr_matrix.columns[j], 
                corr_val
            ))

print(f"Highly correlated feature pairs (|r| > 0.8): {len(high_corr_pairs)}")
for feat1, feat2, corr_val in high_corr_pairs:
    print(f"  {feat1} <-> {feat2}: {corr_val:.4f}")


print("=" * 60)
print("ğŸ�¯ FINAL EDA SUMMARY AND INSIGHTS")
print("=" * 60)

print(f"ğŸ“Š Dataset Overview:")
print(f"  â€¢ Training samples: {len(train):,}")
print(f"  â€¢ Test samples: {len(test):,}")
print(f"  â€¢ Total features: {len([col for col in train.columns if col != 'label'])}")
print(f"  â€¢ Target variable: 'label'")

print(f"\nğŸ�¯ Target Variable:")
print(f"  â€¢ Mean: {target.mean():.6f}")
print(f"  â€¢ Std: {target.std():.6f}")
print(f"  â€¢ Skewness: {target.skew():.3f}")
print(f"  â€¢ Distribution: {'Normal' if abs(target.skew()) < 0.5 else 'Skewed'}")

print(f"\nğŸ”� Data Quality:")
missing_cols = len([col for col in train.columns if train[col].isnull().sum() > 0])
print(f"  â€¢ Columns with missing values: {missing_cols}")
print(f"  â€¢ Constant features: {len(constant_features)}")
print(f"  â€¢ Highly skewed features: {len(highly_skewed)}")

print(f"\nğŸ“ˆ Feature Correlations:")
print(f"  â€¢ Features with strong correlation (|r| > 0.1): {len(strong_corr)}")
print(f"  â€¢ Features with moderate correlation (0.05 < |r| â‰¤ 0.1): {len(moderate_corr)}")
print(f"  â€¢ Features with weak correlation (|r| â‰¤ 0.05): {len(weak_corr)}")

print(f"\nğŸ�† Top 5 Most Important Features:")
for i, row in corr_df.head(5).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Correlation']:.6f}")

print(f"\nğŸ’¡ Key Insights:")
print(f"  â€¢ Target shows {'low' if target.std() < 0.01 else 'moderate' if target.std() < 0.1 else 'high'} variability")
print(f"  â€¢ {'Few' if len(strong_corr) < 10 else 'Many'} features show strong correlation with target")
print(f"  â€¢ Data appears {'clean' if missing_cols == 0 else 'to have missing values'}")
print(f"  â€¢ Feature engineering {'may' if len(highly_skewed) > 5 else 'might not'} be needed for skewed features")

print("\nâœ… EDA Complete! feel free to create a copy and share your insights in the comments !")




