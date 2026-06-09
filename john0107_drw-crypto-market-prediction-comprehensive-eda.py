# Core libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ML & Statistics
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Utilities
import os
import gc
import time

# Configure plotting
plt.style.use('seaborn-whitegrid')
sns.set_theme(style="whitegrid", palette="muted", color_codes=True)
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.max_columns', 100)

print("âœ… Libraries imported")


# Load datasets
sample_submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
train_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")

print(f"ğŸ“Š Training Data: {train_df.shape}")
print(f"ğŸ“Š Test Data: {test_df.shape}")
print(f"ğŸ’¾ Train Memory: {train_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")


# Dataset shape and structure overview
print("ğŸ“Š DATASET OVERVIEW")
print("="*50)
print(f"ğŸ�¯ Training Data Shape: {train_df.shape}")
print(f"ğŸ§ª Test Data Shape: {test_df.shape}")
print(f"ğŸ“� Sample Submission Shape: {sample_submission.shape}")

print(f"\nğŸ“� Dataset Dimensions:")
print(f"   ğŸ“ˆ Total training samples: {train_df.shape[0]:,}")
print(f"   ğŸ”¢ Total features: {train_df.shape[1]:,}")
print(f"   ğŸ“Š Feature to sample ratio: 1:{train_df.shape[0]//train_df.shape[1]:,}")

# Check if datasets have datetime index
print(f"\nğŸ•� Index Information:")
print(f"   ğŸ“… Train index type: {type(train_df.index)}")
print(f"   ğŸ“… Test index type: {type(test_df.index)}")

# Quick peek at the data structure
print(f"\nğŸ‘€ First few column names:")
print(f"   ğŸ�·ï¸� {list(train_df.columns[:10])}")
if len(train_df.columns) > 10:
    print(f"   ... and {len(train_df.columns)-10} more columns")

# Check for target variable
if 'label' in train_df.columns:
    print(f"\nğŸ�¯ Target variable found: 'label'")
    print(f"   ğŸ“Š Target stats: min={train_df['label'].min():.4f}, max={train_df['label'].max():.4f}")
    print(f"   ğŸ“Š Target mean: {train_df['label'].mean():.6f}")


# Check for missing values
missing_counts = train_df.isnull().sum()
total_missing = missing_counts.sum()

if total_missing > 0:
    missing_features = missing_counts[missing_counts > 0].sort_values(ascending=False)
    print(f"â�Œ Missing values found: {len(missing_features)} features")
    display(missing_features.head(10))
else:
    print("âœ… No missing values found")

# Check test set
test_missing = test_df.isnull().sum().sum()
print(f"Test set missing values: {test_missing}")

del missing_counts
if 'missing_features' in locals():
    del missing_features
gc.collect()


# Check for infinite values
inf_counts = np.isinf(train_df.select_dtypes(include=[np.number])).sum()
inf_features = inf_counts[inf_counts > 0]

if len(inf_features) > 0:
    print(f"â�Œ Found {len(inf_features)} features with infinite values")
    # Remove features with infinite values
    train_df = train_df.drop(inf_features.index.tolist(), axis=1)
    test_features_to_drop = [col for col in inf_features.index if col in test_df.columns]
    if test_features_to_drop:
        test_df = test_df.drop(test_features_to_drop, axis=1)
    print(f"âœ… Removed infinite value features. New shape: {train_df.shape}")
else:
    print("âœ… No infinite values found")

del inf_counts
if 'inf_features' in locals():
    del inf_features
gc.collect()


# Find and remove constant features (zero variance)
constant_features = []
for col in train_df.select_dtypes(include=[np.number]).columns:
    if train_df[col].var() == 0:
        constant_features.append(col)

if constant_features:
    print(f"â�Œ Found {len(constant_features)} constant features")
    train_df = train_df.drop(constant_features, axis=1)
    test_constant_features = [col for col in constant_features if col in test_df.columns]
    if test_constant_features:
        test_df = test_df.drop(test_constant_features, axis=1)
    print(f"âœ… Removed constant features. New shape: {train_df.shape}")
else:
    print("âœ… No constant features found")

del constant_features
gc.collect()


# Final data summary
print(f"ğŸ“Š Final Dataset Summary:")
print(f"   Training samples: {train_df.shape[0]:,}")
print(f"   Features: {train_df.shape[1]:,}")
print(f"   Memory usage: {train_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

if 'label' in train_df.columns:
    feature_cols = [col for col in train_df.columns if col != 'label']
    print(f"   Predictive features: {len(feature_cols)}")
    print(f"   Target: label")


# Let's take a quick peek at our cleaned data
print("ğŸ‘€ QUICK PEEK AT CLEANED DATA")
print("="*40)

# Display basic info
print(f"ğŸ“Š Dataset shape: {train_df.shape}")
print(f"ğŸ“Š Index range: {train_df.index.min()} to {train_df.index.max()}")

# Show first few rows
print("\nğŸ”� First 3 rows of cleaned data:")
display(train_df.head(3))

# Show column names grouped by type
if 'label' in train_df.columns:
    feature_cols = [col for col in train_df.columns if col != 'label']
    print(f"\nğŸ�·ï¸� Features ({len(feature_cols)}): {feature_cols[:10]} {'...' if len(feature_cols) > 10 else ''}")
    print(f"ğŸ�¯ Target: label")


# Target variable statistical analysis
target = train_df['label']

print(f"ğŸ“Š Target Statistics:")
print(f"   Count: {target.count():,}")
print(f"   Mean: {target.mean():.8f}")
print(f"   Std: {target.std():.8f}")
print(f"   Min: {target.min():.8f}")
print(f"   Max: {target.max():.8f}")
print(f"   Skewness: {target.skew():.4f}")
print(f"   Kurtosis: {target.kurtosis():.4f}")

# Check for unique values
unique_count = target.nunique()
print(f"   Unique values: {unique_count:,}")
print(f"   Unique ratio: {unique_count / len(target):.4f}")


# ====================================================================
# ğŸ“ˆ TARGET VARIABLE VISUALIZATIONS
# ====================================================================

# Comprehensive target variable visualization
if 'label' in train_df.columns:
    target = train_df['label']
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution histogram
    sns.histplot(target, bins=50, color='lightblue', ax=axes[0, 0], kde=True)
    axes[0, 0].axvline(target.mean(), color='red', linestyle='--', label=f'Mean: {target.mean():.4f}')
    axes[0, 0].axvline(target.median(), color='green', linestyle='--', label=f'Median: {target.median():.4f}')
    axes[0, 0].set_title('Target Distribution')
    axes[0, 0].legend()
    
    # 2. Box plot
    sns.boxplot(y=target, color='lightcoral', ax=axes[0, 1])
    axes[0, 1].set_title('Target Boxplot')
    
    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(target.sample(min(5000, len(target))), plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot vs Normal')
    
    # 4. Time series sample
    sample_size = min(10000, len(target))
    sample_target = target.iloc[-sample_size:]
    axes[1, 1].plot(sample_target.index[-sample_size:], sample_target.values, color='purple', linewidth=1)
    axes[1, 1].set_title('Time Series (Recent Sample)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Additional statistical plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Distribution with statistical annotations
    sns.histplot(target, bins=50, kde=True, color='skyblue', ax=axes[0])
    axes[0].axvline(target.mean(), color='red', linestyle='--', label=f'Mean: {target.mean():.6f}')
    axes[0].axvline(target.median(), color='green', linestyle='--', label=f'Median: {target.median():.6f}')
    axes[0].set_title('Target Distribution with Central Tendencies')
    axes[0].legend()
    
    # 2. Cumulative distribution
    sorted_values = np.sort(target)
    cum_prob = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    axes[1].plot(sorted_values, cum_prob, color='orange', linewidth=2)
    axes[1].set_title('Cumulative Distribution Function')
    axes[1].set_xlabel('Target Value')
    axes[1].set_ylabel('Cumulative Probability')
    
    # 3. Rolling statistics (if enough data)
    if len(target) > 1000:
        window = len(target) // 100  # Dynamic window size
        rolling_mean = target.rolling(window=window).mean()
        rolling_std = target.rolling(window=window).std()
        
        axes[2].plot(rolling_mean.iloc[window:].index[::window], 
                     rolling_mean.iloc[window:].values[::window], 
                     label='Rolling Mean', color='blue')
        
        axes[2].fill_between(rolling_mean.iloc[window:].index[::window],
                           (rolling_mean - rolling_std).iloc[window:].values[::window],
                           (rolling_mean + rolling_std).iloc[window:].values[::window],
                           alpha=0.3, label='Â±1 Std Dev')
        axes[2].set_title('Rolling Statistics')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'Insufficient data for rolling statistics',
                   ha='center', va='center')
        axes[2].set_title('Rolling Statistics (N/A)')
    
    plt.tight_layout()
    plt.show()
else:
    print("Target variable 'label' not found!")


# ====================================================================
# ğŸ”� OUTLIER DETECTION AND ANALYSIS
# ====================================================================

print("\nğŸ”� OUTLIER DETECTION ANALYSIS")
print("="*60)

target = train_df['label']

# IQR Method
Q1 = target.quantile(0.25)
Q3 = target.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_outliers = target[(target < lower_bound) | (target > upper_bound)]

print(f"ğŸ“Š IQR METHOD:")
print(f"   ğŸ“‰ Q1 (25th percentile): {Q1:.8f}")
print(f"   ğŸ“‰ Q3 (75th percentile): {Q3:.8f}")
print(f"   ğŸ“‰ IQR: {IQR:.8f}")
print(f"   ğŸ“‰ Lower bound: {lower_bound:.8f}")
print(f"   ğŸ“‰ Upper bound: {upper_bound:.8f}")
print(f"   ğŸš¨ Outliers found: {len(iqr_outliers):,} ({len(iqr_outliers)/len(target)*100:.2f}%)")

# Z-Score Method (using 3 standard deviations)
z_scores = np.abs(stats.zscore(target))
z_outliers = target[z_scores > 3]

print(f"\nğŸ“Š Z-SCORE METHOD (|z| > 3):")
print(f"   ğŸš¨ Outliers found: {len(z_outliers):,} ({len(z_outliers)/len(target)*100:.2f}%)")

# Modified Z-Score Method (more robust)
median = target.median()
mad = np.median(np.abs(target - median))  # Median Absolute Deviation
modified_z_scores = 0.6745 * (target - median) / mad
modified_z_outliers = target[np.abs(modified_z_scores) > 3.5]

print(f"\nğŸ“Š MODIFIED Z-SCORE METHOD (|mz| > 3.5):")
print(f"   ğŸš¨ Outliers found: {len(modified_z_outliers):,} ({len(modified_z_outliers)/len(target)*100:.2f}%)")

# Show extreme values
print(f"\nğŸ”� EXTREME VALUES:")
print(f"   ğŸ”´ Top 5 highest: {target.nlargest(5).values}")
print(f"   ğŸ”µ Top 5 lowest: {target.nsmallest(5).values}")

# Outlier impact assessment
target_no_iqr_outliers = target[(target >= lower_bound) & (target <= upper_bound)]

print(f"\nğŸ“ˆ IMPACT OF OUTLIERS (IQR method):")
print(f"   ğŸ“‹ Original mean: {target.mean():.8f}")
print(f"   ğŸ“‹ Mean without outliers: {target_no_iqr_outliers.mean():.8f}")
print(f"   ğŸ“‹ Original std: {target.std():.8f}")
print(f"   ğŸ“‹ Std without outliers: {target_no_iqr_outliers.std():.8f}")


# Create sample for intensive analyses (5% of data)
sample_size = max(10000, len(train_df) // 20)  # At least 10k samples
sample_df = train_df.iloc[-sample_size:].copy()

print(f"ğŸ“Š Analysis sample: {len(sample_df):,} rows ({len(sample_df)/len(train_df)*100:.1f}% of data)")


# Feature statistics overview
feature_cols = [col for col in sample_df.columns if col != 'label'] if 'label' in sample_df.columns else list(sample_df.columns)

print(f"ğŸ“Š Feature Analysis Summary:")
print(f"   Total features: {len(feature_cols)}")

# Basic feature statistics
feature_stats = {}
for col in feature_cols:
    feature_stats[col] = {
        'mean': sample_df[col].mean(),
        'std': sample_df[col].std(),
        'skewness': sample_df[col].skew(),
        'zeros_pct': (sample_df[col] == 0).sum() / len(sample_df) * 100
    }

feature_stats_df = pd.DataFrame(feature_stats).T

print(f"   High skew features (|skew| > 2): {(abs(feature_stats_df['skewness']) > 2).sum()}")
print(f"   Low variance features (std < 0.01): {(feature_stats_df['std'] < 0.01).sum()}")
print(f"   High zero features (>50% zeros): {(feature_stats_df['zeros_pct'] > 50).sum()}")


# Mutual Information Analysis
X_sample = sample_df.drop('label', axis=1)
y_sample = sample_df['label']

# Handle infinite values
X_sample = X_sample.replace([np.inf, -np.inf], np.nan).fillna(0)

# Compute mutual information scores
mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42, n_neighbors=3)
mi_scores_series = pd.Series(mi_scores, index=X_sample.columns, name='MI_Score').sort_values(ascending=False)

print(f"ğŸ“Š Mutual Information Results:")
print(f"   Mean MI score: {mi_scores_series.mean():.6f}")
print(f"   Max MI score: {mi_scores_series.max():.6f}")
print(f"   Features with MI > 0.01: {(mi_scores_series > 0.01).sum()}")
print(f"   Features with MI > 0.05: {(mi_scores_series > 0.05).sum()}")

print(f"\nğŸ�† Top 10 Most Important Features:")
for i, (feature, score) in enumerate(mi_scores_series.head(10).items(), 1):
    print(f"   {i:2d}. {feature}: {score:.6f}")


# ====================================================================
# ğŸ“ˆ MUTUAL INFORMATION VISUALIZATIONS
# ====================================================================

# Mutual Information Visualizations
if 'label' in sample_df.columns and 'mi_scores_series' in locals():
    # Create comprehensive MI visualizations using seaborn and matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. MI Scores Distribution
    sns.histplot(mi_scores_series.values, bins=50, kde=True, color='lightblue', ax=axes[0, 0])
    axes[0, 0].axvline(mi_scores_series.mean(), color='red', linestyle='--', 
                      label=f'Mean: {mi_scores_series.mean():.4f}')
    axes[0, 0].set_title('MI Scores Distribution')
    axes[0, 0].set_xlabel('MI Score')
    axes[0, 0].legend()
    
    # 2. Top 20 features
    top_20_mi = mi_scores_series.head(20)
    sns.barplot(y=top_20_mi.index, x=top_20_mi.values, orient='h', color='lightcoral', ax=axes[0, 1])
    axes[0, 1].set_title('Top 20 Features by MI Score')
    axes[0, 1].set_xlabel('MI Score')
    
    # 3. MI scores vs feature index (sorted)
    axes[1, 0].plot(range(len(mi_scores_series)), mi_scores_series.values, color='green')
    axes[1, 0].set_title('MI Score vs Feature Index')
    axes[1, 0].set_xlabel('Feature Index (sorted by importance)')
    axes[1, 0].set_ylabel('MI Score')
    
    # 4. Cumulative importance
    cumulative_mi = np.cumsum(mi_scores_series.values)
    cumulative_mi_pct = cumulative_mi / cumulative_mi[-1] * 100
    axes[1, 1].plot(range(len(cumulative_mi_pct)), cumulative_mi_pct, color='purple')
    axes[1, 1].axhline(50, color='red', linestyle='--', alpha=0.7, label='50%')
    axes[1, 1].axhline(80, color='orange', linestyle='--', alpha=0.7, label='80%')
    axes[1, 1].axhline(95, color='green', linestyle='--', alpha=0.7, label='95%')
    axes[1, 1].set_title('Cumulative Feature Importance')
    axes[1, 1].set_xlabel('Number of Features')
    axes[1, 1].set_ylabel('Cumulative Importance %')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Additional visualization focused on top features
    plt.figure(figsize=(12, 6))
    
    # Bar chart of top 30 features
    top_30_mi = mi_scores_series.head(30)
    sns.barplot(y=top_30_mi.index, x=top_30_mi.values, orient='h', palette='viridis')
    plt.title('Top 30 Features by Mutual Information')
    plt.xlabel('MI Score')
    plt.tight_layout()
    plt.show()
    
    print(f"âœ… MI analysis: {len(mi_scores_series)} features, top score: {mi_scores_series.iloc[0]:.6f}")
else:
    print("Cannot create MI visualizations - missing required data!")


# Feature correlation analysis
corr_sample_size = min(len(sample_df), 5000)
corr_sample = sample_df.iloc[-corr_sample_size:]

# Select top features for correlation analysis
if 'mi_scores_series' in locals():
    top_features = mi_scores_series.head(50).index.tolist()
    analysis_cols = top_features + (['label'] if 'label' in corr_sample.columns else [])
else:
    analysis_cols = corr_sample.columns[:50].tolist()

# Compute correlation matrix
corr_data = corr_sample[analysis_cols]
corr_matrix = corr_data.corr()

# Analyze correlation patterns
if 'label' in corr_matrix.columns:
    target_correlations = corr_matrix['label'].drop('label').abs().sort_values(ascending=False)
    
    print(f"Target correlation - Max: {target_correlations.iloc[0]:.4f}, Strong correlations (>0.1): {(target_correlations > 0.1).sum()}")

# Find highly correlated feature pairs
high_correlation_threshold = 0.9
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > high_correlation_threshold:
            high_corr_pairs.append((
                corr_matrix.columns[i], 
                corr_matrix.columns[j], 
                corr_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    print(f"Found {len(high_corr_pairs)} highly correlated pairs (|corr| > {high_correlation_threshold})")
else:
    print("No highly correlated feature pairs found")


# Create correlation heatmap for top features
plt.figure(figsize=(12, 8))

# Select subset for visualization
if 'mi_scores_series' in locals() and 'label' in corr_matrix.columns:
    viz_features = mi_scores_series.head(25).index.tolist() + ['label']
    viz_corr_matrix = corr_matrix.loc[viz_features, viz_features]
    title_suffix = "(Top 25 Features + Target)"
else:
    viz_corr_matrix = corr_matrix.iloc[:30, :30]
    title_suffix = "(Top 30 Features)"

# Create heatmap
mask = np.triu(np.ones_like(viz_corr_matrix, dtype=bool))
plt.figure(figsize=(12, 12))

sns.heatmap(
    viz_corr_matrix,
    mask=mask,
    annot=True,
    annot_kws={"size": 8},     # Smaller font size for annotations
    cmap='RdYlBu_r',
    center=0,
    square=True,
    fmt='.2f',
    cbar_kws={"shrink": 0.8},
    linewidths=0.5,             # Optional: adds separation lines
    linecolor='lightgrey'      # Optional: cleaner grid look
)

plt.title(f'Feature Correlation Matrix {title_suffix}', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.yticks(rotation=0)               # Keep y-axis labels horizontal
plt.tight_layout()
plt.show()

# Target correlation analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
target_correlations.head(15).plot(kind='barh', color='steelblue', alpha=0.7)
plt.title('Top 15 Target Correlations')
plt.xlabel('Absolute Correlation')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
all_target_corrs = corr_matrix['label'].drop('label')
plt.hist(all_target_corrs, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axvline(all_target_corrs.mean(), color='red', linestyle='--', 
            label=f'Mean: {all_target_corrs.mean():.4f}')
plt.title('Target Correlation Distribution')
plt.xlabel('Correlation with Target')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"âœ… Correlation analysis complete - {len(analysis_cols)} features, {len(high_corr_pairs)} high-corr pairs")


# Temporal patterns analysis
temporal_sample_size = min(50000, len(train_df))
temporal_sample = train_df.iloc[-temporal_sample_size:].copy()

print(f"Analyzing temporal patterns on {len(temporal_sample):,} recent observations")


# ====================================================================
# ğŸ“Š TEMPORAL VISUALIZATIONS
# ====================================================================

# Temporal Visualizations with Seaborn
target_temporal = temporal_sample['label']

# Create comprehensive temporal visualizations using matplotlib/seaborn
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Target time series (sample for visualization)
sample_size = min(5000, len(target_temporal))
sample_step = max(1, len(target_temporal) // sample_size)
sample_target = target_temporal.iloc[::sample_step]

axes[0, 0].plot(range(len(sample_target)), sample_target.values, color='blue', linewidth=1)
axes[0, 0].set_title('Target Time Series (Recent Data)')
axes[0, 0].set_xlabel('Time Index')
axes[0, 0].set_ylabel('Target Value')

# 2. Rolling statistics
if len(target_temporal) > 1000:
    window = len(target_temporal) // 100
    rolling_mean = target_temporal.rolling(window=window).mean()
    rolling_std = target_temporal.rolling(window=window).std()
    
    # Sample for visualization
    r_sample_step = max(1, len(rolling_mean) // 1000)
    x_values = range(window, len(rolling_mean), r_sample_step)
    
    axes[0, 1].plot(x_values, rolling_mean.iloc[window::r_sample_step], label='Rolling Mean', color='green')
    axes[0, 1].fill_between(
        x_values,
        rolling_mean.iloc[window::r_sample_step] - rolling_std.iloc[window::r_sample_step],
        rolling_mean.iloc[window::r_sample_step] + rolling_std.iloc[window::r_sample_step],
        alpha=0.3, label='Â±1 Std Dev'
    )
    axes[0, 1].set_title('Rolling Mean & Volatility')
    axes[0, 1].legend()

# 3. Autocorrelation function
max_lags = min(30, len(target_temporal) // 10)
if max_lags > 0:
    # Using statsmodels plot_acf function
    plot_pacf(target_temporal, lags=max_lags, ax=axes[1, 0], alpha=0.05, title='Autocorrelation Function')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Correlation')
    # Store autocorrelation values for later reference
    #acf_result = plot_pacf(target_temporal, lags=max_lags, alpha=0.05)
    #autocorrs = acf_result.correlations[1:]  # Skip lag 0 which is always 1.0

# 4. Target distribution over time periods
n_periods = 5
period_size = len(target_temporal) // n_periods
period_data = []
period_names = []

for i in range(n_periods):
    start_idx = i * period_size
    end_idx = (i + 1) * period_size if i < n_periods - 1 else len(target_temporal)
    period_data.append(target_temporal.iloc[start_idx:end_idx].values)
    period_names.append(f'P{i+1}')

sns.boxplot(data=period_data, ax=axes[1, 1], palette='viridis')
axes[1, 1].set_xticklabels(period_names)
axes[1, 1].set_title('Target Distribution Over Time Periods')
axes[1, 1].set_xlabel('Time Period')
axes[1, 1].set_ylabel('Target Value')

plt.tight_layout()
plt.show()

# Additional time-based visualizations
if len(target_temporal) > 1000:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Volatility pattern
    vol_window = max(50, len(target_temporal) // 200)
    volatility = target_temporal.rolling(window=vol_window).std()
    vol_sample_step = max(1, len(volatility) // 1000)
    
    # Skip NaN values at the beginning due to rolling window
    x_vol = range(vol_window, len(volatility), vol_sample_step)
    axes[0].plot(x_vol, volatility.iloc[vol_window::vol_sample_step], color='purple')
    axes[0].set_title('Volatility Pattern')
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('Target Volatility')
    
    # 2. Cumulative sum (trend analysis)
    cum_target = np.cumsum(target_temporal)
    cum_sample_step = max(1, len(cum_target) // 1000)
    axes[1].plot(range(0, len(cum_target), cum_sample_step), 
                 cum_target.iloc[::cum_sample_step], 
                 color='darkgreen')
    axes[1].set_title('Cumulative Sum')
    axes[1].set_xlabel('Time Index')
    axes[1].set_ylabel('Cumulative Sum')
    
    plt.tight_layout()
    plt.show()


# Principal Component Analysis
# Use sample data for computational efficiency
X_pca = sample_df.drop('label', axis=1)
y_pca = sample_df['label']

# Clean and standardize data
X_pca = X_pca.replace([np.inf, -np.inf], np.nan).fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

print(f"PCA input: {X_pca.shape[0]:,} samples, {X_pca.shape[1]:,} features")


# Fit PCA directly with variance threshold of 95%
target_variance = 0.95
pca = PCA(n_components=target_variance, random_state=42)
pca.fit(X_scaled)

# Get the number of components selected to achieve 95% variance
optimal_components = pca.n_components_
total_explained_variance = np.sum(pca.explained_variance_ratio_)

# Store results in a dictionary for visualization purposes
pca_results = {
    optimal_components: {
        'pca_model': pca,
        'total_explained_variance': total_explained_variance,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'transformed_data': pca.transform(X_scaled)
    }
}

print(f"PCA results: {optimal_components} components for {total_explained_variance:.3f} variance")
print(f"\nğŸ�¯ OPTIMAL COMPONENTS: {optimal_components} (for {target_variance:.1%} variance explained)")


# Additional visualizations with seaborn and matplotlib
plt.figure(figsize=(15, 10))

# Plot 1: Scree plot (elbow method)
plt.subplot(2, 3, 1)
if optimal_components in pca_results:
    explained_var = pca_results[optimal_components]['explained_variance_ratio']
    sns.lineplot(x=range(1, min(21, len(explained_var)+1)), y=explained_var[:20], 
                marker='o', color='blue', linewidth=2)
    plt.title('Scree Plot (First 20 Components)')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True, alpha=0.3)

# Plot 2: Cumulative variance
plt.subplot(2, 3, 2)
cumulative_var = np.cumsum(pca_results[optimal_components]['explained_variance_ratio'])
sns.lineplot(x=range(1, len(cumulative_var)+1), y=cumulative_var, 
         marker='o', color='darkgreen', linewidth=2)
plt.axhline(y=target_variance, color='red', linestyle='--', alpha=0.7, 
           label=f'{target_variance:.1%} target')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Component importance visualization
plt.subplot(2, 3, 3)
# With just one PCA result, show bars for different variance thresholds
variance_thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
components_needed = []
for threshold in variance_thresholds:
    # Find how many components needed for this threshold
    cumsum = np.cumsum(pca_results[optimal_components]['explained_variance_ratio'])
    n_needed = np.argmax(cumsum >= threshold) + 1
    components_needed.append(n_needed)

sns.barplot(x=[f"{int(t*100)}%" for t in variance_thresholds], y=components_needed, palette='viridis')
plt.title('Components for Variance Thresholds')
plt.xlabel('Variance Threshold')
plt.ylabel('Components Required')
plt.grid(True, alpha=0.3)

# Plot 4: Top component loadings with seaborn
plt.subplot(2, 3, 4)
if optimal_components in pca_results:
    pca_model = pca_results[optimal_components]['pca_model']
    pc1_loadings = pca_model.components_[0]
    
    # Get top 15 features by absolute loading
    top_indices = np.argsort(np.abs(pc1_loadings))[-15:]
    top_loadings = pc1_loadings[top_indices]
    top_features = [X_pca.columns[i] for i in top_indices]
    
    # Create DataFrame for seaborn
    loading_df = pd.DataFrame({
        'Feature': [f"...{feat[-10:]}" for feat in top_features],
        'Loading': top_loadings
    })
    
    # Use seaborn barplot with color mapped to loading value
    sns.barplot(y='Feature', x='Loading', data=loading_df, 
                palette='RdYlGn', orient='h')
    plt.title('Top PC1 Feature Loadings')
    plt.xlabel('Loading Value')
    plt.grid(True, alpha=0.3)

# Plot 5: 2D PCA projection with seaborn
plt.subplot(2, 3, 5)
if optimal_components in pca_results:
    transformed_data = pca_results[optimal_components]['transformed_data']
    
    # Sample for visualization
    sample_size = min(2000, transformed_data.shape[0])
    sample_indices = np.random.choice(transformed_data.shape[0], sample_size, replace=False)
    
    pc1_sample = transformed_data[sample_indices, 0]
    pc2_sample = transformed_data[sample_indices, 1] if transformed_data.shape[1] > 1 else np.zeros(len(sample_indices))
    
    if y_pca is not None:
        # Create DataFrame for seaborn
        vis_df = pd.DataFrame({
            'PC1': pc1_sample,
            'PC2': pc2_sample,
            'Target': y_pca.iloc[sample_indices].values
        })
        
        # Use seaborn scatterplot
        sns.scatterplot(data=vis_df, x='PC1', y='PC2', hue='Target', 
                       palette='viridis', alpha=0.6, s=10, legend=False)
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
                    ax=plt.gca(), label='Target Value')
    else:
        # Simple scatter without target coloring
        sns.scatterplot(x=pc1_sample, y=pc2_sample, color='blue', alpha=0.6, s=10)
    
    plt.title('First Two Principal Components')
    plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.3f} variance)')
    plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.3f} variance)' if len(pca_model.explained_variance_ratio_) > 1 else 'PC2')
    plt.grid(True, alpha=0.3)

# Plot 6: Component importance distribution
plt.subplot(2, 3, 6)
if optimal_components in pca_results:
    # Show distribution of variance explained by components
    explained_importance = pca_results[optimal_components]['explained_variance_ratio']
    component_df = pd.DataFrame({
        'Component': range(1, len(explained_importance)+1),
        'Variance Explained': explained_importance
    })
    
    # Calculate components needed for different variance levels
    cumsum = np.cumsum(explained_importance)
    key_thresholds = [50, 75, 90, 95, 99]
    components_needed = []
    
    for threshold in key_thresholds:
        n_needed = np.argmax(cumsum >= threshold/100) + 1
        components_needed.append(n_needed)
    
    # Show annotations for key variance thresholds
    for i, (threshold, n_comp) in enumerate(zip(key_thresholds, components_needed)):
        plt.axvline(x=n_comp, color=f'C{i}', linestyle='--', alpha=0.7, 
                  label=f'{threshold}% variance: {n_comp} components')
    
    # Plot variance distribution
    sns.histplot(component_df['Variance Explained'], bins=20, kde=True, 
               color='lightgreen', alpha=0.7)
    plt.title('Component Importance Distribution')
    plt.xlabel('Variance Explained per Component')
    plt.ylabel('Count')
    plt.legend(fontsize='x-small')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nâœ… PCA analysis completed!")
if optimal_components:
    print(f"   ğŸ�¯ Recommended components: {optimal_components}")
    print(f"   ğŸ“Š Variance explained: {pca_results[optimal_components]['total_explained_variance']:.4f}")
    print(f"   ğŸ“Š Dimensionality reduction: {X_pca.shape[1]} â†’ {optimal_components} ({(1-optimal_components/X_pca.shape[1])*100:.1f}% reduction)")

