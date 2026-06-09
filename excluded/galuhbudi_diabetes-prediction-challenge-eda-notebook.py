# =========================================================
# ğŸ“Š Diabetes Prediction: Advanced EDA + Drift Detection
# Playground Series S5E12
# Including: Mutual Info, KS-Test & Feature Engineering
# =========================================================

# ğŸ”¥ Cell 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
%matplotlib inline

# Diabetes-themed colors
DIABETES_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# ğŸ“¥ Cell 3: Data Loading & Overview
# <a id="data-loading"></a>

train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print("ğŸ“Š DATASET OVERVIEW")
print("="*50)
print(f"âœ“ Train shape: {train_df.shape}")
print(f"âœ“ Test shape: {test_df.shape}")
print(f"âœ“ Columns: {train_df.shape[1]} features")

train_df.info()
display(train_df.head())


# ğŸ�¯ Cell 4: Target Variable Analysis
# <a id="target-analysis"></a>

target_counts = train_df['diagnosed_diabetes'].value_counts()
target_ratio = train_df['diagnosed_diabetes'].value_counts(normalize=True)

print("ğŸ�¯ TARGET DISTRIBUTION")
print("="*50)
print(f"âœ“ No Diabetes: {target_counts[0]:,} ({target_ratio[0]:.1%})")
print(f"âœ“ Diabetes: {target_counts[1]:,} ({target_ratio[1]:.1%})")
print(f"âœ“ Imbalance Ratio: {target_ratio[0]/target_ratio[1]:.2f}:1")
print(f"âœ“ Baseline ROC AUC: {target_ratio[0]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(x=target_counts.index, y=target_counts.values, 
            palette=DIABETES_COLORS[:2], ax=axes[0])
axes[0].set_title('Target Count', fontweight='bold')
axes[0].set_xticklabels(['No Diabetes', 'Diabetes'])

axes[1].pie(target_counts.values, labels=['No Diabetes', 'Diabetes'], 
            autopct='%1.1f%%', colors=DIABETES_COLORS[:2])
axes[1].set_title('Target Percentage', fontweight='bold')

plt.tight_layout()
plt.show()


# ğŸ”� Cell 5: Missing Values Analysis
# <a id="missing-values"></a>

def missing_table(df, name="Dataset"):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    mis_table = pd.concat([mis_val, mis_val_percent], axis=1).rename(
        columns={0: 'Missing Count', 1: '% of Total'})
    
    mis_table = mis_table[mis_table.iloc[:,1] != 0].sort_values(
        '% of Total', ascending=False).round(1)
    
    print(f"ğŸ“‹ {name}: {mis_table.shape[0]} columns with missing values\n")
    return mis_table

train_missing = missing_table(train_df, "Train Data")
test_missing = missing_table(test_df, "Test Data")

if not train_missing.empty:
    display(train_missing.head())


# ğŸ“ˆ Cell 6: Feature Statistics
# <a id="feature-stats"></a>

numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('diagnosed_diabetes')

print("ğŸ“Š NUMERIC FEATURES SUMMARY")
print("="*50)
print(f"âœ“ Total numeric features: {len(numeric_features)}")

numeric_summary = train_df[numeric_features].describe().T
numeric_summary['skewness'] = train_df[numeric_features].skew()
numeric_summary['kurtosis'] = train_df[numeric_features].kurtosis()
display(numeric_summary.head(10).round(2))


# ğŸ”— Cell 7: Correlation Analysis
# <a id="correlation"></a>

correlations = train_df[numeric_features + ['diagnosed_diabetes']].corr()['diagnosed_diabetes'].drop('diagnosed_diabetes')
correlations = correlations.sort_values()

print("ğŸ”— TOP CORRELATIONS WITH TARGET")
print("="*50)
print("ğŸ“ˆ Strongest Positive:")
display(correlations.head(5))
print("\nğŸ“‰ Strongest Negative:")
display(correlations.tail(5))

# Visualize
fig, ax = plt.subplots(figsize=(12, 10))
top_features = correlations.abs().sort_values(ascending=False).head(12)
corr_matrix = train_df[top_features.index.tolist() + ['diagnosed_diabetes']].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
ax.set_title('Top 12 Features Correlation Matrix', fontweight='bold')
plt.show()


# ğŸ“Š Cell 8: Feature Distributions
# <a id="distributions"></a>

top_6_features = correlations.abs().sort_values(ascending=False).head(6).index

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_6_features):
    sns.histplot(data=train_df, x=feature, hue='diagnosed_diabetes', 
                 kde=True, stat="density", common_norm=False, 
                 palette=DIABETES_COLORS[:2], ax=axes[idx], alpha=0.7)
    axes[idx].set_title(f'{feature}', fontweight='bold')
    if idx == 0:
        axes[idx].legend(labels=['No', 'Yes'], title='Diabetes')

plt.suptitle('Feature Distributions by Diabetes Status', y=0.98, fontweight='bold')
plt.tight_layout()
plt.show()


# ğŸš¨ Cell 9: Outliers Detection
# <a id="outliers"></a>

def detect_outliers_iqr(df, features):
    outliers_dict = {}
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[feature] < Q1 - 1.5*IQR) | (df[feature] > Q3 + 1.5*IQR)]
        outliers_dict[feature] = outliers.shape[0]
    
    return pd.DataFrame.from_dict(outliers_dict, orient='index', columns=['Outlier Count'])

outliers_info = detect_outliers_iqr(train_df, top_6_features)
outliers_info['Outlier %'] = (outliers_info['Outlier Count'] / len(train_df) * 100).round(2)
outliers_info = outliers_info.sort_values('Outlier %', ascending=False)

print("ğŸš¨ OUTLIERS SUMMARY (IQR Method)")
print("="*50)
display(outliers_info)

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_6_features):
    sns.boxplot(data=train_df, x='diagnosed_diabetes', y=feature, 
                palette=DIABETES_COLORS[:2], ax=axes[idx])
    axes[idx].set_title(f'{feature}', fontweight='bold')
    axes[idx].set_xticklabels(['No', 'Yes'])

plt.tight_layout()
plt.show()


# ğŸ”¬ Cell 10: Mutual Information Analysis â­�
# <a id="mutual-info"></a>

def calc_mutual_info(series, target_series):
    digitized = pd.cut(series, bins=15, labels=False, duplicates='drop')
    return mutual_info_score(digitized, target_series)

print("ğŸ”¬ MUTUAL INFORMATION (Non-Linear Relationships)")
print("="*60)
print("ğŸ’¡ This reveals hidden patterns that linear correlation misses!")

mi_scores = {}
for feature in top_6_features:
    mi_scores[feature] = calc_mutual_info(train_df[feature], train_df['diagnosed_diabetes'])

mi_scores = pd.Series(mi_scores).sort_values(ascending=False)
display(mi_scores.round(4))

fig, ax = plt.subplots(figsize=(10, 6))
mi_scores.plot(kind='barh', color=DIABETES_COLORS[3], ax=ax)
ax.set_title('Mutual Information Scores with Target\nâ­� Highlights Non-Linear Dependencies', 
             fontweight='bold')
ax.set_xlabel('MI Score')
plt.tight_layout()
plt.show()


# ğŸ”„ Cell 11: Train-Test Drift Detection â­�
# <a id="drift-check"></a>

print("ğŸ”„ TRAIN-TEST DISTRIBUTION DRIFT (KS Test)")
print("="*60)
print("ğŸ’¡ Are train and test from the same distribution? Let's verify!")

comparison_stats = pd.DataFrame({
    'Train_Mean': train_df[numeric_features].mean(),
    'Test_Mean': test_df[numeric_features].mean(),
    'Mean_Diff_%': ((test_df[numeric_features].mean() - train_df[numeric_features].mean()) / train_df[numeric_features].mean() * 100).abs().round(2)
})

comparison_stats = comparison_stats.sort_values('Mean_Diff_%', ascending=False)
print("ğŸ“Š Features with highest mean shift:")
display(comparison_stats.head(8))

# KS Test
print("\nğŸ”� Kolmogorov-Smirnov Test Results:")
ks_results = []
for col in numeric_features[:12]:
    ks_stat, p_value = stats.ks_2samp(train_df[col], test_df[col])
    ks_results.append({'Feature': col, 'KS_Statistic': ks_stat, 'P_Value': p_value})

ks_df = pd.DataFrame(ks_results).sort_values('KS_Statistic', ascending=False)
display(ks_df)

# Visualization for top drift feature
top_drift_feature = comparison_stats.index[0]
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Histogram
sns.histplot(train_df[top_drift_feature], bins=30, color=DIABETES_COLORS[0], 
             label='Train', stat='density', alpha=0.7, ax=axes[0])
sns.histplot(test_df[top_drift_feature], bins=30, color=DIABETES_COLORS[1], 
             label='Test', stat='density', alpha=0.7, ax=axes[0])
axes[0].legend()
axes[0].set_title(f'{top_drift_feature} Distribution', fontweight='bold')

# CDF
axes[1].plot(np.sort(train_df[top_drift_feature]), 
             np.linspace(0, 1, len(train_df), endpoint=False), 
             label='Train CDF', color=DIABETES_COLORS[0], linewidth=2)
axes[1].plot(np.sort(test_df[top_drift_feature]), 
             np.linspace(0, 1, len(test_df), endpoint=False), 
             label='Test CDF', color=DIABETES_COLORS[1], linewidth=2)
axes[1].legend()
axes[1].set_title('Cumulative Distribution Function (CDF)', fontweight='bold')

plt.tight_layout()
plt.show()


# 9ï¸�âƒ£ TRAIN-TEST DISTRIBUTION COMPARISON (KS Test)

print("ğŸ”„ TRAIN vs TEST DISTRIBUTION COMPARISON")
print("="*60)

comparison_stats = pd.DataFrame({
    'Train_Mean': train_df[numeric_features].mean(),
    'Test_Mean': test_df[numeric_features].mean(),
    'Mean_Diff_%': ((test_df[numeric_features].mean() - train_df[numeric_features].mean()) / train_df[numeric_features].mean() * 100).abs().round(2),
    'Train_Std': train_df[numeric_features].std(),
    'Test_Std': test_df[numeric_features].std()
})

comparison_stats = comparison_stats.sort_values('Mean_Diff_%', ascending=False)
print("ğŸ“Š TOP 10 FEATURES WITH HIGHEST MEAN DIFFERENCE:")
display(comparison_stats.head(10))

print("\nğŸ”� KOLMOGOROV-SMIRNOV TEST RESULTS:")
print("-" * 40)
ks_results = []
for col in numeric_features[:15]:  # Test top 15
    ks_stat, p_value = stats.ks_2samp(train_df[col], test_df[col])
    ks_results.append({'Feature': col, 'KS_Statistic': ks_stat, 'P_Value': p_value})

ks_df = pd.DataFrame(ks_results).sort_values('KS_Statistic', ascending=False)
display(ks_df.head(8))




