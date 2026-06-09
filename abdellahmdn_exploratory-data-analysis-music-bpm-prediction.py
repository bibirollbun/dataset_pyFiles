import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")



train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


print("="*60)
print("ğŸ“Š DATASET BASIC INFORMATION")
print("="*60)

print(f"Training data shape: {train_df.shape}")
print(f"Number of features: {len(train_df.columns) - 1}")
print(f"Number of samples: {len(train_df)}")

features = [col for col in train_df.columns if col != 'BeatsPerMinute']
target = 'BeatsPerMinute'

print(f"\nFeatures ({len(features)}):")
for i, feature in enumerate(features, 1):
    print(f"{i:2d}. {feature}")

print(f"\nTarget variable: {target}")


print("\n" + "="*40)
print("DATA TYPES")
print("="*40)
print(train_df.dtypes)

print("\n" + "="*40)
print("MISSING VALUES")
print("="*40)
train_df.isnull().sum()




print("\n" + "="*60)
print("ğŸ“ˆ DESCRIPTIVE STATISTICS")
print("="*60)

desc_stats = train_df.describe()
print("Training Data Summary:")
print(desc_stats.round(4))

# Additional statistics
print("\n" + "="*40)
print("ADDITIONAL STATISTICS")
print("="*40)

additional_stats = pd.DataFrame({
    'Skewness': train_df.select_dtypes(include=[np.number]).skew(),
    'Kurtosis': train_df.select_dtypes(include=[np.number]).kurtosis(),
    'IQR': train_df.select_dtypes(include=[np.number]).quantile(0.75) - 
           train_df.select_dtypes(include=[np.number]).quantile(0.25)
})
print(additional_stats.round(4))


print("\n" + "="*60)
print("ğŸ�¯ TARGET VARIABLE ANALYSIS")
print("="*60)

target_data = train_df[target]

print(f"Target: {target}")
print(f"Mean: {target_data.mean():.2f}")
print(f"Median: {target_data.median():.2f}")
print(f"Std: {target_data.std():.2f}")
print(f"Min: {target_data.min():.2f}")
print(f"Max: {target_data.max():.2f}")
print(f"Range: {target_data.max() - target_data.min():.2f}")
print(f"Skewness: {target_data.skew():.2f}")
print(f"Kurtosis: {target_data.kurtosis():.2f}")


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Target Variable (BeatsPerMinute) Distribution Analysis', fontsize=16, y=0.98)

# Histogram
axes[0,0].hist(target_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(target_data.mean(), color='red', linestyle='--', label=f'Mean: {target_data.mean():.1f}')
axes[0,0].axvline(target_data.median(), color='green', linestyle='--', label=f'Median: {target_data.median():.1f}')
axes[0,0].set_title('Distribution of BeatsPerMinute')
axes[0,0].set_xlabel('BeatsPerMinute')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Box plot
axes[0,1].boxplot(target_data, patch_artist=True, 
                 boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[0,1].set_title('BeatsPerMinute Box Plot')
axes[0,1].set_ylabel('BeatsPerMinute')
axes[0,1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(target_data, dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot (Normal Distribution)')
axes[1,0].grid(True, alpha=0.3)

# Density plot
target_data.plot.density(ax=axes[1,1], color='purple', alpha=0.7)
axes[1,1].set_title('Density Plot')
axes[1,1].set_xlabel('BeatsPerMinute')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("ğŸ“Š FEATURE DISTRIBUTIONS")
print("="*60)

n_features = len(features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
fig.suptitle('Feature Distributions', fontsize=16, y=0.98)

if n_features == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for i, feature in enumerate(features):
    if i < len(axes):
        data = train_df[feature]
        axes[i].hist(data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[i].set_title(f'{feature}\nSkew: {data.skew():.2f}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

# Hide empty subplots
for i in range(len(features), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# Print skewness analysis
print("\nSkewness Analysis:")
print("-" * 40)
for feature in features:
    skew_val = train_df[feature].skew()
    if abs(skew_val) > 1:
        skew_level = "Highly skewed"
    elif abs(skew_val) > 0.5:
        skew_level = "Moderately skewed"
    else:
        skew_level = "Normal"
    
    print(f"{feature:25s}: {skew_val:7.2f} ({skew_level})")


print("\n" + "="*60)
print("ğŸ”— CORRELATION ANALYSIS")
print("="*60)

# Calculate correlation matrix
corr_matrix = train_df.corr()

# Correlation with target
target_corr = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)
print("Correlation with Target (BeatsPerMinute):")
print("-" * 45)
for feature, corr in target_corr.items():
    strength = ""
    if abs(corr) > 0.7:
        strength = " (Strong)"
    elif abs(corr) > 0.3:
        strength = " (Moderate)"
    elif abs(corr) > 0.1:
        strength = " (Weak)"
    else:
        strength = " (Very weak)"
    
    print(f"{feature:25s}: {corr:7.4f}{strength}")


plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
colors = ['red' if x < 0 else 'green' for x in target_corr.values]
bars = target_corr.plot(kind='barh', color=colors, alpha=0.7)
plt.title('Feature Correlations with BeatsPerMinute', fontsize=14)
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')

# Add correlation values on bars
for i, (feature, corr) in enumerate(target_corr.items()):
    plt.text(corr + (0.01 if corr > 0 else -0.01), i, f'{corr:.3f}', 
             ha='left' if corr > 0 else 'right', va='center', fontsize=9)

plt.tight_layout()
plt.show()



print("\n" + "="*40)
print("CORRELATION SIGNIFICANCE TESTING")
print("="*40)

correlations = []
for feature in features:
    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(train_df[feature], train_df[target])
    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(train_df[feature], train_df[target])
    
    correlations.append({
        'Feature': feature,
        'Pearson_Corr': pearson_corr,
        'Pearson_P_Value': pearson_p,
        'Spearman_Corr': spearman_corr,
        'Spearman_P_Value': spearman_p,
        'Significant': 'Yes' if pearson_p < 0.05 else 'No'
    })

corr_df = pd.DataFrame(correlations).sort_values('Pearson_Corr', key=abs, ascending=False)
print("Detailed Correlation Analysis:")
print(corr_df.round(4))


top_features = corr_df.head(6)['Feature'].tolist()

n_cols = 3
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
fig.suptitle('Scatter Plots: Top Features vs BeatsPerMinute', fontsize=16, y=0.98)

axes = axes.flatten()

for i, feature in enumerate(top_features):
    if i < len(axes):
        # Scatter plot
        axes[i].scatter(train_df[feature], train_df[target], 
                      alpha=0.6, s=30, color='blue')
        
        # Add trend line
        z = np.polyfit(train_df[feature], train_df[target], 1)
        p = np.poly1d(z)
        axes[i].plot(train_df[feature], p(train_df[feature]), "r--", alpha=0.8, linewidth=2)
        
        corr = train_df[feature].corr(train_df[target])
        axes[i].set_title(f'{feature}\nCorrelation: {corr:.3f}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('BeatsPerMinute')
        axes[i].grid(True, alpha=0.3)


print("\n" + "="*60)
print("ğŸš¨ OUTLIER ANALYSIS")
print("="*60)

# IQR method for outlier detection
outlier_summary = []

for column in train_df.select_dtypes(include=[np.number]).columns:
    Q1 = train_df[column].quantile(0.25)
    Q3 = train_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train_df[(train_df[column] < lower_bound) | 
                       (train_df[column] > upper_bound)]
    
    outlier_summary.append({
        'Feature': column,
        'Total_Outliers': len(outliers),
        'Outlier_Percentage': (len(outliers) / len(train_df)) * 100,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Percentage', ascending=False)
print("Outlier Summary (IQR Method):")
print(outlier_df.round(4))


# Get features with outliers for visualization
features_with_outliers = outlier_df[outlier_df['Outlier_Percentage'] > 0]['Feature'].head(6).tolist()

if features_with_outliers:
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
    fig.suptitle('Box Plots: Features with Outliers', fontsize=16, y=0.98)
    
    axes = axes.flatten()
    
    for i, feature in enumerate(features_with_outliers):
        if i < len(axes):
            bp = train_df.boxplot(column=feature, ax=axes[i], patch_artist=True,
                                boxprops=dict(facecolor='lightgreen', alpha=0.7))
            axes[i].set_title(f'{feature}')
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(features_with_outliers), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
else:
    print("No significant outliers detected in the dataset.")


print("\n" + "="*60)
print("ğŸ“� FEATURE SCALING ANALYSIS")
print("="*60)

scale_analysis = pd.DataFrame({
    'Feature': features + [target],
    'Mean': [train_df[f].mean() for f in features + [target]],
    'Std': [train_df[f].std() for f in features + [target]],
    'Min': [train_df[f].min() for f in features + [target]],
    'Max': [train_df[f].max() for f in features + [target]],
    'Range': [train_df[f].max() - train_df[f].min() for f in features + [target]]
})

print("Feature Scaling Requirements:")
print(scale_analysis.round(4))

print("\nScaling Recommendations:")
print("-" * 30)
for _, row in scale_analysis.iterrows():
    feature = row['Feature']
    if feature != target:  # Don't scale target
        if row['Range'] > 1000:
            print(f"ğŸ”´ {feature}: Large range ({row['Range']:.0f}) - SCALING REQUIRED")
        elif row['Std'] > 100:
            print(f"ğŸŸ¡ {feature}: High variance (std: {row['Std']:.2f}) - Scaling recommended")
        elif row['Range'] > 100:
            print(f"ğŸŸ¡ {feature}: Moderate range ({row['Range']:.2f}) - Consider scaling")
        else:
            print(f"ğŸŸ¢ {feature}: Good scale - No scaling needed")

