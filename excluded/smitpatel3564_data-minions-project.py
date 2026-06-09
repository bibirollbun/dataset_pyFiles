import numpy as np
import pandas as pd
import os


!pip install -q scikit-learn==1.5.2


import sklearn
sklearn.__version__


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import torch
from sklearn.pipeline import Pipeline


train_df=pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


print(f"Dataset contains {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
train_df.head()



train_df.info()


test_ids = test_df['id']  # saving ids in another variable for submission

target_column = 'Premium Amount'

categorical_columns = train_df.select_dtypes(include=['object']).columns
numerical_columns = train_df.select_dtypes(exclude=['object']).columns

print("Target Column:", target_column)
print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


train_df.describe().round(2)


for column in categorical_columns:
    num_unique = train_df[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")



for column in categorical_columns:
    print(f"\nTop value counts in '{column}':\n{train_df[column].value_counts().head(10)}")


print("The mean of columns:")
print(train_df[numerical_columns].mean())

print("\nThe std dev of columns:")
print(train_df[numerical_columns].std())

print("\nThe skewness of columns:")
print(train_df[numerical_columns].skew())


print("Missing Values in Each column")
print(train_df.isna().sum())


# ğŸ�¨ Enhanced Missing Values Analysis
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('ğŸ”� Comprehensive Missing Values Analysis', fontsize=16, fontweight='bold')

# 1. Missing values heatmap
sns.heatmap(train_df.isnull(), cbar=True, cmap="Reds", ax=axes[0,0], 
            cbar_kws={'label': 'Missing Values'})
axes[0,0].set_title("Training Data Missing Values Pattern", fontweight='bold')
axes[0,0].set_xlabel("Features")
axes[0,0].set_ylabel("Samples")

# 2. Missing values percentage
missing_percent = (train_df.isnull().sum() / len(train_df)) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=True)

if len(missing_percent) > 0:
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(missing_percent)))
    bars = axes[0,1].barh(range(len(missing_percent)), missing_percent.values, color=colors)
    axes[0,1].set_yticks(range(len(missing_percent)))
    axes[0,1].set_yticklabels(missing_percent.index)
    axes[0,1].set_xlabel("Missing Percentage (%)")
    axes[0,1].set_title("Missing Values by Feature", fontweight='bold')
    axes[0,1].grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (bar, value) in enumerate(zip(bars, missing_percent.values)):
        axes[0,1].text(value + 0.1, i, f'{value:.1f}%', va='center', ha='left', fontweight='bold')
else:
    axes[0,1].text(0.5, 0.5, 'âœ… No Missing Values\nin Training Data', 
                   ha='center', va='center', transform=axes[0,1].transAxes,
                   fontsize=14, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    axes[0,1].set_title("Training Data Missing Values", fontweight='bold')

# 3. Test data missing values heatmap
sns.heatmap(test_df.isnull(), cbar=True, cmap="Blues", ax=axes[1,0],
            cbar_kws={'label': 'Missing Values'})
axes[1,0].set_title("Test Data Missing Values Pattern", fontweight='bold')
axes[1,0].set_xlabel("Features")
axes[1,0].set_ylabel("Samples")

# 4. Test data missing values percentage
test_missing_percent = (test_df.isnull().sum() / len(test_df)) * 100
test_missing_percent = test_missing_percent[test_missing_percent > 0].sort_values(ascending=True)

if len(test_missing_percent) > 0:
    colors_test = plt.cm.Blues(np.linspace(0.3, 1, len(test_missing_percent)))
    bars_test = axes[1,1].barh(range(len(test_missing_percent)), test_missing_percent.values, color=colors_test)
    axes[1,1].set_yticks(range(len(test_missing_percent)))
    axes[1,1].set_yticklabels(test_missing_percent.index)
    axes[1,1].set_xlabel("Missing Percentage (%)")
    axes[1,1].set_title("Test Data Missing Values by Feature", fontweight='bold')
    axes[1,1].grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (bar, value) in enumerate(zip(bars_test, test_missing_percent.values)):
        axes[1,1].text(value + 0.1, i, f'{value:.1f}%', va='center', ha='left', fontweight='bold')
else:
    axes[1,1].text(0.5, 0.5, 'âœ… No Missing Values\nin Test Data', 
                   ha='center', va='center', transform=axes[1,1].transAxes,
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    axes[1,1].set_title("Test Data Missing Values", fontweight='bold')

plt.tight_layout()
plt.show()

# Detailed missing values analysis
print("ğŸ”� MISSING VALUES DETAILED ANALYSIS")
print("=" * 60)

print("ğŸ“Š TRAINING DATA:")
train_total_missing = train_df.isnull().sum().sum()
train_total_cells = len(train_df) * len(train_df.columns)
print(f"   Total Missing Values: {train_total_missing:,}")
print(f"   Total Data Points: {train_total_cells:,}")
print(f"   Missing Percentage: {(train_total_missing/train_total_cells)*100:.2f}%")

if train_total_missing > 0:
    print("   Features with Missing Values:")
    for col, missing_count in train_df.isnull().sum().items():
        if missing_count > 0:
            print(f"     â€¢ {col:<20}: {missing_count:,} ({(missing_count/len(train_df))*100:.1f}%)")

print("\nğŸ“Š TEST DATA:")
test_total_missing = test_df.isnull().sum().sum()
test_total_cells = len(test_df) * len(test_df.columns)
print(f"   Total Missing Values: {test_total_missing:,}")
print(f"   Total Data Points: {test_total_cells:,}")
print(f"   Missing Percentage: {(test_total_missing/test_total_cells)*100:.2f}%")

if test_total_missing > 0:
    print("   Features with Missing Values:")
    for col, missing_count in test_df.isnull().sum().items():
        if missing_count > 0:
            print(f"     â€¢ {col:<20}: {missing_count:,} ({(missing_count/len(test_df))*100:.1f}%)")


print("Test Missing Values per Column")
print(test_df.isnull().sum())


# Droping Duplicates If Any
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()
print(f"Shape of Train before droping duplicates {train_df.shape}")

print("No Duplicates Were Found ")


print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

display(train_df.head())
train_df.info()
train_df.describe().T



# ğŸ�¨ Enhanced Categorical Features Analysis
cat_cols = ['Gender','Marital Status','Education Level','Occupation','Location',
            'Policy Type','Customer Feedback','Smoking Status','Exercise Frequency','Property Type']

# Create a more comprehensive visualization
fig, axes = plt.subplots(5, 2, figsize=(18, 25))
fig.suptitle('ğŸ“Š Comprehensive Categorical Features Analysis', fontsize=16, fontweight='bold')

# Color palettes for better visual appeal
color_palettes = ['Set2', 'Set3', 'pastel', 'dark', 'colorblind', 'bright', 'muted', 'deep', 'Set1', 'Paired']

for idx, col in enumerate(cat_cols):
    row = idx // 2
    col_idx = idx % 2
    
    # Get value counts
    value_counts = train_df[col].value_counts()
    
    # Create enhanced bar plot
    sns.countplot(data=train_df, x=col, ax=axes[row, col_idx], 
                  palette=color_palettes[idx % len(color_palettes)], 
                  order=value_counts.index)
    
    # Rotate labels for better readability
    axes[row, col_idx].tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for i, v in enumerate(value_counts.values):
        axes[row, col_idx].text(i, v + max(value_counts.values) * 0.01, 
                               f'{v:,}\n({v/len(train_df)*100:.1f}%)', 
                               ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Styling
    axes[row, col_idx].set_title(f"{col} Distribution", fontsize=12, fontweight='bold')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel("Count")
    axes[row, col_idx].grid(True, alpha=0.3, axis='y')
    
    # Add diversity index
    diversity = 1 - sum((value_counts / len(train_df)) ** 2)
    axes[row, col_idx].text(0.02, 0.95, f'Diversity: {diversity:.3f}', 
                           transform=axes[row, col_idx].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

plt.tight_layout()
plt.show()

# Enhanced categorical analysis
print("ğŸ“Š CATEGORICAL FEATURES DETAILED ANALYSIS")
print("=" * 70)

for col in cat_cols:
    print(f"\nğŸ”� {col.upper()}")
    print("-" * 50)
    
    value_counts = train_df[col].value_counts()
    total_count = len(train_df)
    
    print(f"Unique Categories: {train_df[col].nunique()}")
    print(f"Most Common: {value_counts.index[0]} ({value_counts.iloc[0]:,} - {value_counts.iloc[0]/total_count*100:.1f}%)")
    print(f"Least Common: {value_counts.index[-1]} ({value_counts.iloc[-1]:,} - {value_counts.iloc[-1]/total_count*100:.1f}%)")
    
    # Diversity index (Simpson's diversity index)
    diversity = 1 - sum((value_counts / total_count) ** 2)
    print(f"Diversity Index: {diversity:.3f} (Higher = More Diverse)")
    
    # Check for potential data quality issues
    if train_df[col].nunique() > total_count * 0.8:
        print("âš ï¸�  High cardinality - consider grouping rare categories")
    elif train_df[col].nunique() < 3:
        print("â„¹ï¸�  Low cardinality - binary or few categories")


# ğŸ�¨ Enhanced Numerical Features Analysis
num_cols = ['Age','Annual Income','Number of Dependents','Health Score',
            'Previous Claims','Vehicle Age','Credit Score','Insurance Duration']

# Create subplots for better visualization
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle('ğŸ“ˆ Comprehensive Numerical Features Analysis', fontsize=16, fontweight='bold')

for idx, col in enumerate(num_cols):
    row = idx // 2
    col_idx = idx % 2
    
    # Enhanced histogram with statistics
    sns.histplot(train_df[col], kde=True, stat='density', alpha=0.7, ax=axes[row, col_idx])
    
    # Add mean and median lines
    mean_val = train_df[col].mean()
    median_val = train_df[col].median()
    
    axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.1f}')
    axes[row, col_idx].axvline(median_val, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'Median: {median_val:.1f}')
    
    # Styling
    axes[row, col_idx].set_title(f"{col} Distribution", fontsize=12, fontweight='bold')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel("Density")
    axes[row, col_idx].legend()
    axes[row, col_idx].grid(True, alpha=0.3)
    
    # Add skewness annotation
    skewness = train_df[col].skew()
    axes[row, col_idx].text(0.02, 0.95, f'Skew: {skewness:.2f}', transform=axes[row, col_idx].transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.show()

# Summary statistics table
print("ğŸ“Š NUMERICAL FEATURES SUMMARY STATISTICS")
print("=" * 80)
summary_stats = train_df[num_cols].describe().round(2)
print(summary_stats.to_string())

print("\nğŸ”� SKEWNESS ANALYSIS")
print("=" * 40)
for col in num_cols:
    skew_val = train_df[col].skew()
    if abs(skew_val) < 0.5:
        interpretation = "Nearly Normal"
    elif abs(skew_val) < 1:
        interpretation = "Moderately Skewed"
    else:
        interpretation = "Highly Skewed"
    print(f"{col:<20}: {skew_val:>6.3f} ({interpretation})")


# ğŸ�¨ Enhanced Correlation Analysis
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('ğŸ”— Comprehensive Correlation Analysis', fontsize=16, fontweight='bold')

# 1. Full correlation heatmap
correlation_matrix = train_df[num_cols+['Premium Amount']].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

im1 = sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                  center=0, square=True, ax=axes[0], cbar_kws={"shrink": .8})
axes[0].set_title('Correlation Matrix (Numerical Features)', fontweight='bold')

# 2. Premium correlation focus
premium_corr = correlation_matrix['Premium Amount'].sort_values(key=abs, ascending=False)[1:]  # Exclude self-correlation
colors = ['red' if x < 0 else 'green' for x in premium_corr.values]
bars = axes[1].barh(range(len(premium_corr)), premium_corr.values, color=colors, alpha=0.7)
axes[1].set_yticks(range(len(premium_corr)))
axes[1].set_yticklabels(premium_corr.index)
axes[1].set_xlabel('Correlation with Premium Amount')
axes[1].set_title('Premium Amount Correlations', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.5)

# Add correlation values on bars
for i, (bar, value) in enumerate(zip(bars, premium_corr.values)):
    axes[1].text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                 va='center', ha='left' if value >= 0 else 'right', fontweight='bold')

# 3. Feature importance based on correlation
feature_importance = premium_corr.abs().sort_values(ascending=True)
colors_imp = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))

bars_imp = axes[2].barh(range(len(feature_importance)), feature_importance.values, color=colors_imp)
axes[2].set_yticks(range(len(feature_importance)))
axes[2].set_yticklabels(feature_importance.index)
axes[2].set_xlabel('Absolute Correlation with Premium')
axes[2].set_title('Feature Importance (Correlation-based)', fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')

# Add importance values on bars
for i, (bar, value) in enumerate(zip(bars_imp, feature_importance.values)):
    axes[2].text(value + 0.005, i, f'{value:.3f}', va='center', ha='left', fontweight='bold')

# Detailed correlation analysis
print("ğŸ”— CORRELATION ANALYSIS INSIGHTS")
print("=" * 60)

print("ğŸ“ˆ STRONGEST POSITIVE CORRELATIONS WITH PREMIUM:")
positive_corr = premium_corr[premium_corr > 0].sort_values(ascending=False)
for feature, corr_val in positive_corr.head(3).items():
    print(f"   â€¢ {feature:<20}: +{corr_val:.3f}")

print("\nğŸ“‰ STRONGEST NEGATIVE CORRELATIONS WITH PREMIUM:")
negative_corr = premium_corr[premium_corr < 0].sort_values(ascending=True)
for feature, corr_val in negative_corr.head(3).items():
    print(f"   â€¢ {feature:<20}: {corr_val:.3f}")

print(f"\nğŸ�¯ CORRELATION STRENGTH INTERPRETATION:")
print(f"   â€¢ Strong (|r| > 0.7): {sum(abs(premium_corr) > 0.7)} features")
print(f"   â€¢ Moderate (0.3 < |r| â‰¤ 0.7): {sum((abs(premium_corr) > 0.3) & (abs(premium_corr) <= 0.7))} features")
print(f"   â€¢ Weak (|r| â‰¤ 0.3): {sum(abs(premium_corr) <= 0.3)} features")

# Multicollinearity check
print(f"\nâš ï¸�  MULTICOLLINEARITY CHECK:")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

if high_corr_pairs:
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"   â€¢ {feat1} â†” {feat2}: {corr_val:.3f}")
else:
    print("   âœ… No high multicollinearity detected (|r| > 0.8)")



# Calculate the correlation matrix
correlation_matrix = train_df[numerical_columns].corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Variables", fontsize=16)
plt.show()



# Optional : Premium Amount Distribution Analysis with Multiple Transformations
from scipy.stats import yeojohnson, skew, boxcox
import numpy as np

# Apply different transformations for comparison
original_data = train_df['Premium Amount']

# 1. Original (no transformation)
original_skew = original_data.skew()

# 2. Log transformation (log1p to handle zeros)
log_transformed = np.log1p(original_data)
log_skew = skew(log_transformed)

# 3. Square root transformation
sqrt_transformed = np.sqrt(original_data)
sqrt_skew = skew(sqrt_transformed)

# 4. Cube root transformation
cube_root_transformed = np.cbrt(original_data)
cube_root_skew = skew(cube_root_transformed)

# 5. Box-Cox transformation (only if all values are positive)
if (original_data > 0).all():
    boxcox_transformed, boxcox_lambda = boxcox(original_data)
    boxcox_skew = skew(boxcox_transformed)
else:
    boxcox_transformed = None
    boxcox_skew = None

# 6. Yeo-Johnson transformation
yj_transformed, yj_lambda = yeojohnson(original_data)
yj_skew = skew(yj_transformed)

# Create comparison visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('ğŸ”� Transformation Comparison for Premium Amount', fontsize=16, fontweight='bold')

# Plot 1: Original Distribution
sns.histplot(original_data, kde=True, stat='density', alpha=0.7, color='red', ax=axes[0,0])
axes[0,0].axvline(original_data.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: ${original_data.mean():,.0f}')
axes[0,0].axvline(original_data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: ${original_data.median():,.0f}')
axes[0,0].set_title(f"Original Distribution\nSkew: {original_skew:.3f}")
axes[0,0].set_xlabel("Premium Amount ($)")
axes[0,0].set_ylabel("Density")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Log Transformation
sns.histplot(log_transformed, kde=True, stat='density', alpha=0.7, color='blue', ax=axes[0,1])
axes[0,1].axvline(log_transformed.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {log_transformed.mean():.2f}')
axes[0,1].axvline(log_transformed.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {log_transformed.median():.2f}')
axes[0,1].set_title(f"Log Transformation\nSkew: {log_skew:.3f}")
axes[0,1].set_xlabel("Log(Premium Amount + 1)")
axes[0,1].set_ylabel("Density")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Square Root Transformation
sns.histplot(sqrt_transformed, kde=True, stat='density', alpha=0.7, color='purple', ax=axes[1,0])
axes[1,0].axvline(sqrt_transformed.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {sqrt_transformed.mean():.2f}')
axes[1,0].axvline(sqrt_transformed.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {sqrt_transformed.median():.2f}')
axes[1,0].set_title(f"Square Root Transformation\nSkew: {sqrt_skew:.3f}")
axes[1,0].set_xlabel("âˆš(Premium Amount)")
axes[1,0].set_ylabel("Density")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Cube Root Transformation
sns.histplot(cube_root_transformed, kde=True, stat='density', alpha=0.7, color='brown', ax=axes[1,1])
axes[1,1].axvline(cube_root_transformed.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {cube_root_transformed.mean():.2f}')
axes[1,1].axvline(cube_root_transformed.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {cube_root_transformed.median():.2f}')
axes[1,1].set_title(f"Cube Root Transformation\nSkew: {cube_root_skew:.3f}")
axes[1,1].set_xlabel("âˆ›(Premium Amount)")
axes[1,1].set_ylabel("Density")
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

# Plot 5: Box-Cox Transformation (if applicable)
if boxcox_transformed is not None:
    sns.histplot(boxcox_transformed, kde=True, stat='density', alpha=0.7, color='orange', ax=axes[2,0])
    axes[2,0].axvline(boxcox_transformed.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {boxcox_transformed.mean():.2f}')
    axes[2,0].axvline(np.median(boxcox_transformed), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(boxcox_transformed):.2f}')
    axes[2,0].set_title(f"Box-Cox Transformation (Î»={boxcox_lambda:.3f})\nSkew: {boxcox_skew:.3f}")
    axes[2,0].set_xlabel("Box-Cox Transformed Premium")
    axes[2,0].set_ylabel("Density")
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)
else:
    axes[2,0].text(0.5, 0.5, 'Box-Cox Not Applicable\n(Contains non-positive values)', 
                   ha='center', va='center', transform=axes[2,0].transAxes,
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[2,0].set_title("Box-Cox Transformation")

# Plot 6: Yeo-Johnson Transformation
sns.histplot(yj_transformed, kde=True, stat='density', alpha=0.7, color='green', ax=axes[2,1])
axes[2,1].axvline(yj_transformed.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {yj_transformed.mean():.2f}')
axes[2,1].axvline(np.median(yj_transformed), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(yj_transformed):.2f}')
axes[2,1].set_title(f"Yeo-Johnson Transformation (Î»={yj_lambda:.3f})\nSkew: {yj_skew:.3f}")
axes[2,1].set_xlabel("Yeo-Johnson Transformed Premium")
axes[2,1].set_ylabel("Density")
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create skewness comparison chart
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle('ğŸ“Š Transformation Skewness Comparison', fontsize=16, fontweight='bold')

# Prepare data for comparison
transformations = ['Original', 'Log', 'Square Root', 'Cube Root', 'Yeo-Johnson']
skewness_values = [abs(original_skew), abs(log_skew), abs(sqrt_skew), abs(cube_root_skew), abs(yj_skew)]
colors = ['red', 'blue', 'purple', 'brown', 'green']

# Add Box-Cox if applicable
if boxcox_skew is not None:
    transformations.insert(-1, 'Box-Cox')
    skewness_values.insert(-1, abs(boxcox_skew))
    colors.insert(-1, 'orange')

# Create bar plot
bars = ax.bar(transformations, skewness_values, color=colors, alpha=0.7, edgecolor='black')

# Add value labels on bars
for bar, value in zip(bars, skewness_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Highlight the best transformation
best_idx = skewness_values.index(min(skewness_values))
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

ax.set_title('Absolute Skewness Comparison Across Transformations')
ax.set_ylabel('Absolute Skewness')
ax.set_xlabel('Transformation Method')
ax.grid(True, alpha=0.3, axis='y')

# Add annotation for best transformation
ax.annotate(f'BEST: {transformations[best_idx]}', 
            xy=(best_idx, skewness_values[best_idx]), xytext=(best_idx, skewness_values[best_idx] + 0.05),
            arrowprops=dict(arrowstyle='->', color='gold', lw=2),
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.7))

plt.tight_layout()
plt.show()

# Summary statistics
print("ğŸ“Š TRANSFORMATION SUMMARY STATISTICS")
print("=" * 60)
for i, (name, skew_val) in enumerate(zip(transformations, skewness_values)):
    status = "ğŸ¥‡ OPTIMAL" if i == best_idx else "ğŸ“Š"
    print(f"{status} {name}: |Skewness| = {skew_val:.4f}")

print(f"\nğŸ�¯ RECOMMENDATION: {transformations[best_idx]} transformation shows the lowest skewness.")
print(f"ğŸ“ˆ IMPROVEMENT: {abs(original_skew) - min(skewness_values):.4f} reduction in absolute skewness")

# Show original vs best transformation comparison
best_transformation_name = transformations[best_idx]
if best_transformation_name == 'Yeo-Johnson':
    best_transformed_data = yj_transformed
elif best_transformation_name == 'Box-Cox':
    best_transformed_data = boxcox_transformed
elif best_transformation_name == 'Log':
    best_transformed_data = log_transformed
elif best_transformation_name == 'Square Root':
    best_transformed_data = sqrt_transformed
elif best_transformation_name == 'Cube Root':
    best_transformed_data = cube_root_transformed
else:
    best_transformed_data = original_data

print(f"\nğŸ”� DETAILED COMPARISON: Original vs {best_transformation_name}")
print("-" * 50)
print(f"Original Skewness: {original_skew:.4f}")
print(f"{best_transformation_name} Skewness: {min(skewness_values):.4f}")
print(f"Skewness Reduction: {((abs(original_skew) - min(skewness_values)) / abs(original_skew) * 100):.1f}%")


def date(df):

    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Year'] = df['Policy Start Date'].dt.year
    df['Day'] = df['Policy Start Date'].dt.day
    df['Month'] = df['Policy Start Date'].dt.month
    df['Month_name'] = df['Policy Start Date'].dt.month_name()
    df['Day_of_week'] = df['Policy Start Date'].dt.day_name()
    df['Week'] = df['Policy Start Date'].dt.isocalendar().week
    df['Year_sin'] = np.sin(2 * np.pi * df['Year'])
    df['Year_cos'] = np.cos(2 * np.pi * df['Year'])
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    df['Year_sin'] = np.sin(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Year_cos'] = np.cos(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['Group']=(df['Year']-2020)*48+df['Month']*4+df['Day']//7

    df.drop('Policy Start Date', axis=1, inplace=True)

    return df

# Apply the date function to both datasets
train_df = date(train_df)
test_df = date(test_df)


# Define features and target
numerical_features = [
    'Age', 'Annual Income', 'Number of Dependents', 'Health Score',
    'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration',
    'Year_sin', 'Year_cos', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos'
]
categorical_features = [
    'Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
    'Policy Type', 'Customer Feedback', 'Smoking Status', 'Exercise Frequency',
    'Property Type', 'Month_name', 'Day_of_week'
]
target_column = 'Premium Amount'


# Split train data into features and target
X = train_df.drop(columns=[target_column, 'id', 'Group', 'Year', 'Month', 'Day', 'Week'])
y = train_df[target_column]


# Simple preprocessing pipeline
# Numerical features pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features pipeline  
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ],
    remainder='drop'
)

# Preprocess train and test data
print("ğŸ”„ APPLYING PREPROCESSING...")
X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test_df.drop(columns=['id', 'Group', 'Year', 'Month', 'Day', 'Week']))

# Get feature names
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features))

# Create processed DataFrames
X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
test_processed_df = pd.DataFrame(test_processed, columns=feature_names, index=test_df.index)

print("âœ… Multi-encoding preprocessing completed!")
print(f"ğŸ“Š Processed training data shape: {X_processed_df.shape}")
print(f"ğŸ“Š Processed test data shape: {test_processed_df.shape}")
print(f"ğŸ�¯ Target variable shape: {y.shape}")

# Display first few rows of processed data
print("\nğŸ”� Sample of processed features:")
X_processed_df.head(3)

# Check for any remaining missing values
print(f"\nğŸ”§ Missing values in processed data: {X_processed_df.isnull().sum().sum()}")
print(f"ğŸ”§ Missing values in test data: {test_processed_df.isnull().sum().sum()}")

# Display encoding summary
print(f"\nğŸ�—ï¸� ENCODING SUMMARY:")
print(f"   â€¢ Numerical features: {len(numerical_features)} columns")
print(f"   â€¢ Binary encoded features: {len([col for col in feature_names if 'binary' in col])} columns")
print(f"   â€¢ Target encoded features: {len([col for col in feature_names if 'target' in col])} columns")
print(f"   â€¢ Total features: {X_processed_df.shape[1]} columns")


# ğŸ�¯ SKLEARN PIPELINE VISUALIZATION
from sklearn import set_config

# Enable sklearn pipeline diagram display
set_config(display='diagram')

print("ğŸ“Š PREPROCESSING PIPELINE DIAGRAM")
print("=" * 50)

# Display the preprocessing pipeline
preprocessor


# ğŸ”§ PIPELINE STRUCTURE & SUMMARY
print("ğŸ�—ï¸� COMPLETE PREPROCESSING PIPELINE STRUCTURE")
print("=" * 60)

print("ğŸ“‹ Pipeline Components:")
print(f"   â€¢ Numerical Pipeline: {len(numerical_features)} features")
print(f"     - Imputation: SimpleImputer (strategy='median')")
print(f"     - Features: {numerical_features}")

print(f"\n   â€¢ Categorical Pipeline: {len(categorical_features)} features")  
print(f"     - Imputation: SimpleImputer (strategy='constant', fill_value='Unknown')")
print(f"     - Encoding: OneHotEncoder (handle_unknown='ignore')")
print(f"     - Features: {categorical_features}")

print(f"\nğŸ“Š Pipeline Output:")
print(f"   â€¢ Input Shape: {X.shape}")
print(f"   â€¢ Output Shape: {X_processed.shape}")
print(f"   â€¢ Feature Expansion: {X_processed.shape[1] / X.shape[1]:.2f}x")

print(f"\nâœ… Pipeline Status: READY FOR MODEL TRAINING")

# Reset display config to default after showing the diagram
set_config(display='text')


PERFORMANCE_SETTINGS = {
    'USE_SAMPLING': True,
    'SAMPLE_SIZE': 0.10,
    'CV_FOLDS': 3,
    'N_ESTIMATORS_BASELINE': 30,
    'N_ESTIMATORS_TUNED': [50, 100],
    'RANDOM_SEARCH_ITER': 8,
    'N_JOBS': -1,
    'MAX_DEPTH': 10
}

print("âš¡ ULTRA-FAST PERFORMANCE MODE FOR LARGE DATASETS:")
print("=" * 60)
for key, value in PERFORMANCE_SETTINGS.items():
    print(f"{key:25s}: {value}")
print("=" * 60)
print("\nğŸ’¡ CURRENT OPTIMIZATIONS (FASTEST):")
print("  âœ“ Using only 10% of data (150K rows)")
print("  âœ“ 3-fold CV (instead of 5)")
print("  âœ“ Only 30 trees for baseline models")
print("  âœ“ Max depth limited to 10")
print("  âœ“ Reduced hyperparameter search (8 iterations)")
print("  âœ“ All CPU cores utilized (n_jobs=-1)")
print("\nâ�±ï¸�  EXPECTED RUNTIME:")
print("  â€¢ Baseline models: ~1-3 minutes (was 15-30 min)")
print("  â€¢ Hyperparameter tuning: ~2-5 minutes (was 30-60 min)")
print("  â€¢ Total: ~3-8 minutes (was 1-2 hours)")
print("\nğŸ’¡ FOR BETTER ACCURACY (when you have more time):")
print("  â€¢ Change SAMPLE_SIZE to 0.20 or 0.30")
print("  â€¢ Increase N_ESTIMATORS_BASELINE to 100")
print("  â€¢ Increase RANDOM_SEARCH_ITER to 20")
print("  â€¢ For final submission, set USE_SAMPLING = False")
print("=" * 60)


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time

RANDOM_STATE = 42
SAMPLE_SIZE = 0.10
USE_SAMPLING = True

print("=" * 70)
if USE_SAMPLING:
    print(f"âš¡ FAST MODE ACTIVATED: Using {SAMPLE_SIZE*100:.0f}% sample")
    print(f"   Original dataset: {X_processed_df.shape[0]:,} rows")
    print(f"   Training sample: {int(X_processed_df.shape[0] * SAMPLE_SIZE):,} rows")
    print(f"   Speed gain: ~{1/SAMPLE_SIZE:.0f}x faster")
    
    X_train, _, y_train, _ = train_test_split(
        X_processed_df, y, 
        train_size=SAMPLE_SIZE, 
        random_state=RANDOM_STATE,
        stratify=pd.qcut(y, q=10, labels=False, duplicates='drop')
    )
    print(f"\nâœ“ Sample created: {X_train.shape[0]:,} rows Ã— {X_train.shape[1]} features")
    print("âœ“ Stratified sampling ensures representative distribution")
else:
    X_train = X_processed_df
    y_train = y
    print(f"ğŸ�Œ FULL MODE: Using complete dataset")
    print(f"   This will take significantly longer!")
    print(f"   Dataset: {X_train.shape[0]:,} rows")

print("=" * 70)


models_baseline = {
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=500),
    "ElasticNet": ElasticNet(random_state=RANDOM_STATE, max_iter=500),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=8),
    "RandomForest": RandomForestRegressor(
        n_estimators=30, max_depth=10, min_samples_split=10,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=30, max_depth=10, min_samples_split=10,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=30, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=30, max_depth=8, num_leaves=31,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=30, depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, verbose=0
    ),
}

baseline_results = []
kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

print("\nâš¡ Training baseline models (3-fold CV, reduced complexity)...")
print("=" * 70)
total_start = time.time()

for name, model in models_baseline.items():
    start = time.time()
    
    cv_mae = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1).mean()
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1).mean()
    
    elapsed = time.time() - start
    
    baseline_results.append({
        'Model': name,
        'CV_MAE': round(cv_mae, 2),
        'CV_RMSE': round(cv_rmse, 2),
        'CV_R2': round(cv_r2, 4),
        'Time_sec': round(elapsed, 2)
    })
    print(f"âœ“ {name:18s} | RMSE: {cv_rmse:7.2f} | RÂ²: {cv_r2:6.4f} | Time: {elapsed:5.1f}s")

total_baseline_time = time.time() - total_start
baseline_df = pd.DataFrame(baseline_results).sort_values('CV_RMSE')

print("=" * 70)
print(f"âš¡ Total baseline training: {total_baseline_time:.1f}s ({total_baseline_time/60:.1f} min)")
print(f"âœ“ Best model: {baseline_df.iloc[0]['Model']} (RMSE: {baseline_df.iloc[0]['CV_RMSE']})")
print("=" * 70)

baseline_df


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
baseline_df_sorted = baseline_df.sort_values('CV_R2', ascending=False)
axes[0].barh(baseline_df_sorted['Model'], baseline_df_sorted['CV_R2'], color='steelblue')
axes[0].set_xlabel('RÂ² Score')
axes[0].set_title('Model Performance: RÂ² Score')
axes[0].grid(axis='x', alpha=0.3)

baseline_df_sorted_rmse = baseline_df.sort_values('CV_RMSE')
axes[1].barh(baseline_df_sorted_rmse['Model'], baseline_df_sorted_rmse['CV_RMSE'], color='salmon')
axes[1].set_xlabel('RMSE')
axes[1].set_title('Model Performance: RMSE')
axes[1].grid(axis='x', alpha=0.3)

axes[2].barh(baseline_df['Model'], baseline_df['Time_sec'], color='gold')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_title('Training Time')
axes[2].grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


print("=" * 80)
print("BASELINE ANALYSIS - KEY FINDINGS")
print("=" * 80)
print(f"Best Model: {baseline_df.iloc[0]['Model']}")
print(f"Best CV RMSE: {baseline_df.iloc[0]['CV_RMSE']}")
print(f"Best CV R2: {baseline_df.iloc[0]['CV_R2']}")
print(f"\nWorst Model: {baseline_df.iloc[-1]['Model']}")
print(f"Worst CV RMSE: {baseline_df.iloc[-1]['CV_RMSE']}")
print(f"Worst CV R2: {baseline_df.iloc[-1]['CV_R2']}")
print("\n" + "=" * 80)
print("IDENTIFIED ISSUES:")
print("=" * 80)
print("1. Low R2 scores across all models - weak predictive power")
print("2. High RMSE values - large prediction errors")
print("3. Linear models underperforming - non-linear relationships present")
print("4. Tree-based models show better performance but still room for improvement")
print("5. Gradient boosting methods (XGBoost, LightGBM, CatBoost) are top performers")
print("\nUNDERLYING CAUSES:")
print("- Default hyperparameters not optimized for this dataset")
print("- Potential feature interactions not captured")
print("- Model complexity may be insufficient")
print("- Need for hyperparameter tuning and feature engineering")
print("=" * 80)


print("Generating residual plots for top 3 models...")
top3_models = baseline_df.head(3)['Model'].tolist()
fig, axes = plt.subplots(len(top3_models), 2, figsize=(14, 4*len(top3_models)))

sample_size_plot = min(50000, len(X_train))
indices = np.random.choice(len(X_train), sample_size_plot, replace=False)
X_sample = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
y_sample = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]

for idx, model_name in enumerate(top3_models):
    model = models_baseline[model_name]
    model.fit(X_sample, y_sample)
    y_pred = model.predict(X_sample)
    residuals = y_sample - y_pred
    
    axes[idx, 0].scatter(y_pred, residuals, alpha=0.3, s=1)
    axes[idx, 0].axhline(y=0, color='r', linestyle='--')
    axes[idx, 0].set_xlabel('Predicted Values')
    axes[idx, 0].set_ylabel('Residuals')
    axes[idx, 0].set_title(f'{model_name} - Residual Plot (n={sample_size_plot:,})')
    axes[idx, 0].grid(alpha=0.3)
    
    axes[idx, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[idx, 1].set_xlabel('Residuals')
    axes[idx, 1].set_ylabel('Frequency')
    axes[idx, 1].set_title(f'{model_name} - Residual Distribution')
    axes[idx, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
print("âœ“ Residual analysis complete")


param_grids = {
    "RandomForest": {
        'n_estimators': [50, 100],
        'max_depth': [10, 15],
        'min_samples_split': [5, 10],
        'max_features': ['sqrt']
    },
    "XGBoost": {
        'n_estimators': [50, 100],
        'max_depth': [6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8]
    },
    "LightGBM": {
        'n_estimators': [50, 100],
        'max_depth': [8, 12],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31]
    },
    "CatBoost": {
        'iterations': [50, 100],
        'depth': [6, 8],
        'learning_rate': [0.05, 0.1]
    }
}

tuned_models = {}
tuned_results = []
total_start = time.time()

top_models = baseline_df.head(3)['Model'].tolist()
models_to_tune = [m for m in top_models if m in param_grids]

print("\nâš¡ HYPERPARAMETER TUNING (Reduced search space for speed)")
print("=" * 70)
print(f"Tuning {len(models_to_tune)} best models: {', '.join(models_to_tune)}")
print(f"Strategy: RandomizedSearchCV with 8 iterations per model")
print("=" * 70)

for model_name in models_to_tune:
    print(f"\nğŸ”§ Tuning {model_name}...")
    start = time.time()
    
    base_model = models_baseline[model_name]
    
    random_search = RandomizedSearchCV(
        base_model,
        param_grids[model_name],
        n_iter=8,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    tuned_models[model_name] = best_model
    
    cv_rmse = -random_search.best_score_
    cv_r2 = cross_val_score(best_model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1).mean()
    
    elapsed = time.time() - start
    
    tuned_results.append({
        'Model': model_name,
        'Best_Params': str(random_search.best_params_),
        'CV_RMSE': round(cv_rmse, 2),
        'CV_R2': round(cv_r2, 4),
        'Time_sec': round(elapsed, 2)
    })
    
    print(f"   Best params: {random_search.best_params_}")
    print(f"   âœ“ CV RMSE: {cv_rmse:.2f} | RÂ²: {cv_r2:.4f} | Time: {elapsed:.1f}s")

total_time = time.time() - total_start
print("\n" + "=" * 70)
print(f"âš¡ Total tuning time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"âœ“ Best tuned model: {tuned_results[0]['Model']}")
print("=" * 70)

tuned_df = pd.DataFrame(tuned_results).sort_values('CV_RMSE')
tuned_df


comparison_data = []

for model_name in tuned_df['Model'].tolist():
    baseline_row = baseline_df[baseline_df['Model'] == model_name].iloc[0]
    tuned_row = tuned_df[tuned_df['Model'] == model_name].iloc[0]
    
    comparison_data.append({
        'Model': model_name,
        'Baseline_RMSE': baseline_row['CV_RMSE'],
        'Tuned_RMSE': tuned_row['CV_RMSE'],
        'RMSE_Improvement': round(baseline_row['CV_RMSE'] - tuned_row['CV_RMSE'], 2),
        'Baseline_R2': baseline_row['CV_R2'],
        'Tuned_R2': tuned_row['CV_R2'],
        'R2_Improvement': round(tuned_row['CV_R2'] - baseline_row['CV_R2'], 4)
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nğŸ“Š BASELINE vs TUNED COMPARISON:")
comparison_df


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

models_list = comparison_df['Model']
x = np.arange(len(models_list))
width = 0.35

axes[0].bar(x - width/2, comparison_df['Baseline_RMSE'], width, label='Baseline', alpha=0.8)
axes[0].bar(x + width/2, comparison_df['Tuned_RMSE'], width, label='Tuned', alpha=0.8)
axes[0].set_xlabel('Model')
axes[0].set_ylabel('RMSE')
axes[0].set_title('RMSE Comparison: Baseline vs Tuned')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_list, rotation=45)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(x - width/2, comparison_df['Baseline_R2'], width, label='Baseline', alpha=0.8)
axes[1].bar(x + width/2, comparison_df['Tuned_R2'], width, label='Tuned', alpha=0.8)
axes[1].set_xlabel('Model')
axes[1].set_ylabel('RÂ² Score')
axes[1].set_title('RÂ² Comparison: Baseline vs Tuned')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models_list, rotation=45)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


best_model_name = tuned_df.iloc[0]['Model']
best_model = tuned_models[best_model_name]
best_model.fit(X_train, y_train)

if best_model_name in ['RandomForest', 'ExtraTrees']:
    importances = best_model.feature_importances_
elif best_model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting']:
    importances = best_model.feature_importances_
else:
    importances = None

if importances is not None:
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


print("=" * 80)
print("IMPROVEMENT SUMMARY")
print("=" * 80)
for idx, row in comparison_df.iterrows():
    print(f"\n{row['Model']}:")
    print(f"  RMSE: {row['Baseline_RMSE']} â†’ {row['Tuned_RMSE']} (Î” {row['RMSE_Improvement']})")
    print(f"  RÂ²: {row['Baseline_R2']} â†’ {row['Tuned_R2']} (Î” +{row['R2_Improvement']})")
    
avg_rmse_improvement = comparison_df['RMSE_Improvement'].mean()
avg_r2_improvement = comparison_df['R2_Improvement'].mean()

print("\n" + "=" * 80)
print(f"Average RMSE Improvement: {avg_rmse_improvement:.2f}")
print(f"Average RÂ² Improvement: +{avg_r2_improvement:.4f}")
print(f"\nBest Model After Tuning: {best_model_name}")
print(f"Best CV RMSE: {tuned_df.iloc[0]['CV_RMSE']}")
print(f"Best CV RÂ²: {tuned_df.iloc[0]['CV_R2']}")
print("=" * 80)


print("=" * 80)
print("FUTURE PLAN FOR PROJECT COMPLETION")
print("=" * 80)
print("\n1. ADVANCED FEATURE ENGINEERING:")
print("   - Create polynomial features for key numerical variables")
print("   - Engineer interaction features between highly correlated variables")
print("   - Apply domain-specific transformations for insurance data")
print("   - Test feature selection methods (RFE, SelectKBest)")
print("\n2. ENSEMBLE METHODS:")
print("   - Implement stacking with multiple base learners")
print("   - Create voting regressors combining best models")
print("   - Test blending strategies for final predictions")
print("\n3. HYPERPARAMETER OPTIMIZATION:")
print("   - Use Bayesian optimization (Optuna) for more efficient search")
print("   - Implement early stopping for gradient boosting models")
print("   - Fine-tune top 2-3 models with extensive grid search")
print("\n4. MODEL VALIDATION:")
print("   - Implement stratified K-fold for better cross-validation")
print("   - Analyze learning curves to detect overfitting/underfitting")
print("   - Perform residual analysis for all final models")
print("\n5. FINAL DELIVERABLES:")
print("   - Generate predictions on test set with best model")
print("   - Create comprehensive model documentation")
print("   - Develop business insights from feature importances")
print("   - Prepare final presentation with actionable recommendations")
print("\n6. ADDITIONAL METHODS TO EXPLORE:")
print("   - Neural networks for non-linear pattern capture")
print("   - AutoML frameworks (TPOT, Auto-sklearn) for automated optimization")
print("   - Time-based validation if temporal patterns exist")
print("   - Quantile regression for prediction intervals")
print("\n7. PERFORMANCE MONITORING:")
print("   - Track model drift over time")
print("   - Implement A/B testing framework for model comparison")
print("   - Set up automated retraining pipeline")
print("=" * 80)


print("Generating final predictions on test set...")
best_model_name = tuned_df.iloc[0]['Model']
best_model = tuned_models[best_model_name]

if USE_SAMPLING:
    print(f"âš¡ Retraining {best_model_name} on FULL training dataset for final predictions...")
    final_start = time.time()
    best_model.fit(X_processed_df, y)
    print(f"âœ“ Training complete in {time.time() - final_start:.1f}s")
else:
    best_model.fit(X_train, y_train)

test_predictions = best_model.predict(test_processed_df)
submission = pd.DataFrame({
    'id': test_ids,
    'Premium Amount': test_predictions
})


