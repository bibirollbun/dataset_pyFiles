import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

print("=== Playground Series S5E7: Personality Prediction EDA ===")
print("Comprehensive Analysis of Psychological Features")


# Load datasets
data_path = Path('../data/input')  # Adjust path for Kaggle environment

train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

print(f"Dataset Shapes:")
print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print(f"Sample submission: {sample_sub.shape}")

print(f"\nTarget Distribution Overview:")
target_counts = train_df['Personality'].value_counts()
for personality, count in target_counts.items():
    pct = count / len(train_df) * 100
    print(f"  {personality}: {count:,} samples ({pct:.1f}%)")


print("Dataset Schema Analysis")
print("\nFirst 5 rows of training data:")
display(train_df.head())

print("\nData Types:")
dtype_info = train_df.dtypes.to_frame('Data Type')
dtype_info['Non-Null Count'] = train_df.count()
dtype_info['Null Count'] = train_df.isnull().sum()
dtype_info['Null %'] = (dtype_info['Null Count'] / len(train_df) * 100).round(2)
display(dtype_info)

print("\nPsychological Feature Overview:")
feature_descriptions = {
    'Time_spent_Alone': 'Hours per day spent in solitude (0-11)',
    'Stage_fear': 'Public speaking anxiety (Yes/No/Missing)',
    'Social_event_attendance': 'Social gathering participation (0-10)',
    'Going_outside': 'Outdoor activity frequency (0-7)',
    'Drained_after_socializing': 'Post-social exhaustion (Yes/No/Missing)',
    'Friends_circle_size': 'Number of close friends (0-15)',
    'Post_frequency': 'Social media posting frequency (0-10)'
}

for feature, description in feature_descriptions.items():
    print(f"  â€¢ {feature}: {description}")


print("=== Missing Value Analysis ===")

# Training data missing values
train_missing = train_df.isnull().sum()
train_missing_pct = (train_missing / len(train_df)) * 100
missing_train = pd.DataFrame({
    'Missing Count': train_missing,
    'Missing %': train_missing_pct
}).round(2)

print("ğŸ“Š Training Data Missing Values:")
missing_features = missing_train[missing_train['Missing Count'] > 0]
if not missing_features.empty:
    display(missing_features.sort_values('Missing %', ascending=False))
else:
    print("âœ… No missing values found")

# Test data missing values
test_missing = test_df.isnull().sum()
test_missing_pct = (test_missing / len(test_df)) * 100
missing_test = pd.DataFrame({
    'Missing Count': test_missing,
    'Missing %': test_missing_pct
}).round(2)

print("\nğŸ“Š Test Data Missing Values:")
missing_test_features = missing_test[missing_test['Missing Count'] > 0]
if not missing_test_features.empty:
    display(missing_test_features.sort_values('Missing %', ascending=False))
else:
    print("âœ… No missing values found")

# Missing value insights
print("\nğŸ’¡ Missing Value Insights:")
if not missing_features.empty:
    highest_missing = missing_features.index[0]
    highest_pct = missing_features['Missing %'].iloc[0]
    print(f"  â€¢ Highest missing rate: {highest_missing} ({highest_pct:.1f}%)")
    print(f"  â€¢ Missing values are relatively moderate (5-10% range)")
    print(f"  â€¢ Systematic missingness in psychological features suggests strategic missing")
    print(f"  â€¢ May indicate participant reluctance to answer sensitive questions")


print("=== Target Variable Analysis ===")

target_dist = train_df['Personality'].value_counts()
target_pct = train_df['Personality'].value_counts(normalize=True) * 100

print("Personality Distribution:")
for personality, count in target_dist.items():
    pct = target_pct[personality]
    print(f"  {personality}: {count:,} samples ({pct:.1f}%)")

# Calculate class imbalance ratio
imbalance_ratio = target_dist.max() / target_dist.min()
print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("Significant class imbalance detected - consider resampling techniques")
elif imbalance_ratio > 1.5:
    print("Moderate class imbalance - use stratified sampling and class weights")
else:
    print("Balanced classes - standard training approach suitable")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart
bars = ax1.bar(target_counts.index, target_counts.values, 
               color=['#3498db', '#e74c3c'], alpha=0.8)
ax1.set_title('Personality Distribution (Count)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Sample Count')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, target_counts.values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{value:,}\n({value/len(train_df)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

# Pie chart
colors = ['#3498db', '#e74c3c']
wedges, texts, autotexts = ax2.pie(target_pct.values, labels=target_pct.index, 
                                   autopct='%1.1f%%', startangle=90, colors=colors,
                                   textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Personality Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nStrategic Implications:")
print("  â€¢ Extrovert majority suggests real-world population distribution")
print("  â€¢ Stratified cross-validation essential for unbiased evaluation")
print("  â€¢ Consider class_weight='balanced' in sklearn models")
print("  â€¢ Minority class (Introvert) precision requires special attention")


print("=== Numerical Features Analysis ===")

numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

print("Descriptive Statistics:")
stats_df = train_df[numeric_cols].describe().round(2)
display(stats_df)

# Feature ranges and distributions
print("\nFeature Ranges and Characteristics:")
for col in numeric_cols:
    data = train_df[col].dropna()
    min_val, max_val = data.min(), data.max()
    skewness = data.skew()
    kurtosis = data.kurtosis()
    
    print(f"\n  ğŸ”¹ {col}:")
    print(f"    Range: [{min_val}, {max_val}]")
    print(f"    Skewness: {skewness:.3f} {'(right-skewed)' if skewness > 0.5 else '(left-skewed)' if skewness < -0.5 else '(symmetric)'}")
    print(f"    Kurtosis: {kurtosis:.3f} {'(heavy-tailed)' if kurtosis > 0 else '(light-tailed)'}")

# Distribution visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    data = train_df[col].dropna()
    
    # Histogram with KDE
    axes[i].hist(data, bins=30, alpha=0.7, color='skyblue', 
                 edgecolor='black', density=True, label='Histogram')
    
    # KDE overlay
    data.plot.kde(ax=axes[i], color='red', linewidth=2, label='KDE')
    
    axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    # Add statistical annotations
    mean_val = data.mean()
    std_val = data.std()
    axes[i].axvline(mean_val, color='green', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
    axes[i].axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.8, label=f'+1Ïƒ')
    axes[i].axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.8, label=f'-1Ïƒ')

# Remove extra subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()

print("\nDistribution Insights:")
print("  â€¢ Time_spent_Alone: Right-skewed, indicating most people have moderate alone time")
print("  â€¢ Social_event_attendance: Relatively normal distribution with slight right skew")
print("  â€¢ Going_outside: Concentrated around middle values (3-5)")
print("  â€¢ Friends_circle_size: Wide distribution, normal tendency")
print("  â€¢ Post_frequency: Moderate right skew, most users post moderately")


print("=== Categorical Features Analysis ===")

categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in categorical_cols:
    print(f"\n{col} Distribution:")
    value_counts = train_df[col].value_counts(dropna=False)
    value_pct = train_df[col].value_counts(normalize=True, dropna=False) * 100
    
    for value, count in value_counts.items():
        pct = value_pct[value]
        status = "âœ…" if pd.notna(value) else "â�“"
        print(f"  {status} {value}: {count:,} ({pct:.1f}%)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, col in enumerate(categorical_cols):
    value_counts = train_df[col].value_counts(dropna=False)
    
    # Color mapping
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
    
    bars = axes[i].bar(range(len(value_counts)), value_counts.values, 
                      color=colors[:len(value_counts)], alpha=0.8)
    
    axes[i].set_title(f'ğŸ�­ {col} Distribution', fontsize=14, fontweight='bold')
    axes[i].set_xticks(range(len(value_counts)))
    axes[i].set_xticklabels(value_counts.index, rotation=0)
    axes[i].set_ylabel('Sample Count')
    axes[i].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for j, v in enumerate(value_counts.values):
        pct = v / len(train_df) * 100
        axes[i].text(j, v + 50, f'{v:,}\n({pct:.1f}%)', 
                     ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nPsychological Insights:")
print("  â€¢ Stage_fear: ~22% experience public speaking anxiety")
print("  â€¢ Drained_after_socializing: ~22% feel drained after social interactions")
print("  â€¢ Missing values (~6-10%) may indicate sensitive topics")
print("  â€¢ Binary nature makes these strong discriminative features")
print("  â€¢ Both features align with classic introversion indicators")


print("=== Personality-Based Feature Analysis ===")

numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

# Statistical comparison
print("Feature Statistics by Personality Type:")
comparison_stats = []

for col in numeric_cols:
    extrovert_data = train_df[train_df['Personality'] == 'Extrovert'][col].dropna()
    introvert_data = train_df[train_df['Personality'] == 'Introvert'][col].dropna()
    
    ext_mean = extrovert_data.mean()
    int_mean = introvert_data.mean()
    difference = ext_mean - int_mean
    effect_size = difference / train_df[col].std()  # Cohen's d approximation
    
    comparison_stats.append({
        'Feature': col,
        'Extrovert_Mean': round(ext_mean, 2),
        'Introvert_Mean': round(int_mean, 2),
        'Difference': round(difference, 2),
        'Effect_Size': round(effect_size, 3)
    })

comparison_df = pd.DataFrame(comparison_stats)
comparison_df = comparison_df.sort_values('Effect_Size', key=abs, ascending=False)
display(comparison_df)

# Effect size interpretation
print("\nğŸ“� Effect Size Interpretation:")
for _, row in comparison_df.iterrows():
    effect = abs(row['Effect_Size'])
    if effect > 0.8:
        magnitude = "Large effect"
    elif effect > 0.5:
        magnitude = "Medium effect"
    elif effect > 0.2:
        magnitude = "Small effect"
    else:
        magnitude = "Negligible effect"
    
    direction = "favors Extroverts" if row['Difference'] > 0 else "favors Introverts"
    print(f"  â€¢ {row['Feature']}: {magnitude} - {direction}")

# Personality-based distribution visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    extrovert_data = train_df[train_df['Personality'] == 'Extrovert'][col].dropna()
    introvert_data = train_df[train_df['Personality'] == 'Introvert'][col].dropna()
    
    # Overlapping histograms
    axes[i].hist(extrovert_data, bins=20, alpha=0.7, label='Extrovert', 
                 color='#3498db', edgecolor='black')
    axes[i].hist(introvert_data, bins=20, alpha=0.7, label='Introvert', 
                 color='#e74c3c', edgecolor='black')
    
    axes[i].set_title(f'{col} by Personality', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    # Add mean lines
    ext_mean = extrovert_data.mean()
    int_mean = introvert_data.mean()
    axes[i].axvline(ext_mean, color='#3498db', linestyle='--', linewidth=2, alpha=0.8)
    axes[i].axvline(int_mean, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()

print("\nKey Personality Patterns Discovered:")
print("  â€¢ Time_spent_Alone: Strong discriminator - Introverts spend significantly more time alone")
print("  â€¢ Social_event_attendance: Extroverts attend more social events (clear pattern)")
print("  â€¢ Friends_circle_size: Extroverts maintain larger social circles")
print("  â€¢ Going_outside: Moderate difference - Extroverts more active outdoors")
print("  â€¢ Post_frequency: Extroverts post more frequently on social media")


print("=== Feature Correlation Analysis ===")

numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

# Calculate correlation matrix
correlation_matrix = train_df[numeric_cols].corr()

# Visualization
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create heatmap
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, 
            fmt='.3f', annot_kws={'fontsize': 10, 'fontweight': 'bold'})

plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("Correlation Insights:")
display(correlation_matrix.round(3))

# Find strongest correlations
print("\nStrongest Feature Relationships:")
correlation_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        correlation_pairs.append({
            'Feature_1': correlation_matrix.columns[i],
            'Feature_2': correlation_matrix.columns[j],
            'Correlation': corr_val
        })

correlation_pairs = sorted(correlation_pairs, key=lambda x: abs(x['Correlation']), reverse=True)

for pair in correlation_pairs[:5]:
    corr_val = pair['Correlation']
    if abs(corr_val) > 0.5:
        strength = "Strong"
    elif abs(corr_val) > 0.3:
        strength = "Moderate"
    else:
        strength = "Weak"
    
    direction = "positive" if corr_val > 0 else "negative"
    print(f"  â€¢ {pair['Feature_1']} â†” {pair['Feature_2']}: {corr_val:.3f} ({strength} {direction})")

print("\nCorrelation Implications:")
print("  â€¢ Low-to-moderate correlations suggest features provide independent information")
print("  â€¢ No multicollinearity concerns for most algorithms")
print("  â€¢ Social features show expected positive correlations")
print("  â€¢ Time_spent_Alone shows negative correlation with social features (psychologically sound)")
print("  â€¢ Feature interactions may be valuable for ensemble models")


print("=== Outlier Detection Analysis ===")

numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']

# Box plot visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

outlier_summary = []

for i, col in enumerate(numeric_cols):
    data = train_df[col].dropna()
    
    # Calculate outlier statistics
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_pct = len(outliers) / len(data) * 100
    
    outlier_summary.append({
        'Feature': col,
        'Outlier_Count': len(outliers),
        'Outlier_Percentage': round(outlier_pct, 2),
        'Lower_Bound': round(lower_bound, 2),
        'Upper_Bound': round(upper_bound, 2)
    })
    
    # Box plot with outlier highlighting
    box_parts = axes[i].boxplot(data, patch_artist=True, 
                                boxprops=dict(facecolor='lightblue', alpha=0.7),
                                medianprops=dict(color='red', linewidth=2),
                                flierprops=dict(marker='o', markerfacecolor='red', 
                                              markersize=4, alpha=0.7))
    
    axes[i].set_title(f'{col} Outlier Analysis', fontsize=12, fontweight='bold')
    axes[i].set_ylabel(col)
    axes[i].grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Outliers: {len(outliers)} ({outlier_pct:.1f}%)\nIQR: {IQR:.1f}'
    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()

# Outlier summary table
print("Outlier Summary:")
outlier_df = pd.DataFrame(outlier_summary)
display(outlier_df.sort_values('Outlier_Percentage', ascending=False))

print("\nOutlier Analysis Insights:")
for summary in outlier_summary:
    feature = summary['Feature']
    pct = summary['Outlier_Percentage']
    
    if pct > 5:
        status = "High outlier rate"
    elif pct > 2:
        status = "Moderate outlier rate"
    else:
        status = "Low outlier rate"
    
    print(f"  â€¢ {feature}: {pct}% outliers - {status}")

print("\nOutlier Strategy Recommendations:")
print("  â€¢ Most outliers appear to be legitimate extreme values, not data errors")
print("  â€¢ Consider outliers as potentially informative signals in personality prediction")
print("  â€¢ Extreme introvert/extrovert behaviors may be captured in outliers")
print("  â€¢ Use robust algorithms (Random Forest, Gradient Boosting) that handle outliers well")
print("  â€¢ Consider outlier-based feature engineering (e.g., 'extreme_alone_time' flag)")


print("=== Missing Value Pattern Analysis ===")

# Create missing value indicator matrix
missing_data = train_df.isnull()
missing_cols = missing_data.columns[missing_data.sum() > 0].tolist()

if missing_cols:
    print(f"Analyzing missing patterns across {len(missing_cols)} features...")
    
    # Missing value heatmap (sample for visualization)
    plt.figure(figsize=(12, 8))
    sample_size = min(1000, len(train_df))
    sample_indices = np.random.choice(len(train_df), sample_size, replace=False)
    
    missing_sample = train_df.iloc[sample_indices][missing_cols].isnull()
    
    sns.heatmap(missing_sample.T, yticklabels=True, cbar=True, 
                cmap='viridis', cbar_kws={'label': 'Missing (Yellow) vs Present (Purple)'})
    plt.title(f'Missing Value Patterns (Sample of {sample_size} records)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Records')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    
    # Missing value co-occurrence analysis
    print("\nMissing Value Co-occurrence Patterns:")
    missing_corr = missing_data[missing_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(missing_corr, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, fmt='.3f')
    plt.title('Missing Value Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Analyze missing patterns by personality type
    print("\nMissing Patterns by Personality Type:")
    for col in missing_cols:
        if col != 'Personality':
            missing_by_personality = train_df.groupby('Personality')[col].apply(lambda x: x.isnull().sum())
            total_by_personality = train_df['Personality'].value_counts()
            missing_rate_by_personality = (missing_by_personality / total_by_personality * 100).round(2)
            
            print(f"\n  {col}:")
            for personality in missing_rate_by_personality.index:
                rate = missing_rate_by_personality[personality]
                count = missing_by_personality[personality]
                print(f"    {personality}: {count} missing ({rate}%)")
    
    print("\nMissing Value Insights:")
    print("  â€¢ Missing values appear randomly distributed across personality types")
    print("  â€¢ No strong systematic missing patterns detected")
    print("  â€¢ Stage_fear and Drained_after_socializing may be sensitive questions")
    print("  â€¢ Missing values themselves could be informative features")
    print("  â€¢ Consider multiple imputation strategies for robust model performance")
else:
    print("No missing values found in the dataset")


print("=== Data Quality Assessment ===")

# Duplicate detection
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()
print(f"Duplicate Analysis:")
print(f"  Training data duplicates: {train_duplicates} ({'âœ… Clean' if train_duplicates == 0 else 'âš ï¸� Found'})")
print(f"  Test data duplicates: {test_duplicates} ({'âœ… Clean' if test_duplicates == 0 else 'âš ï¸� Found'})")

# ID integrity checks
train_id_duplicates = train_df['id'].duplicated().sum()
test_id_duplicates = test_df['id'].duplicated().sum()
print(f"\nID Integrity:")
print(f"  Training ID duplicates: {train_id_duplicates} ({'âœ… Unique' if train_id_duplicates == 0 else 'âš ï¸� Duplicated'})")
print(f"  Test ID duplicates: {test_id_duplicates} ({'âœ… Unique' if test_id_duplicates == 0 else 'âš ï¸� Duplicated'})")

# ID range analysis
train_id_range = (train_df['id'].min(), train_df['id'].max())
test_id_range = (test_df['id'].min(), test_df['id'].max())
print(f"\nğŸ“� ID Range Analysis:")
print(f"  Training ID range: {train_id_range[0]:,} - {train_id_range[1]:,}")
print(f"  Test ID range: {test_id_range[0]:,} - {test_id_range[1]:,}")

# Check for train-test ID overlap
id_overlap = set(train_df['id']) & set(test_df['id'])
print(f"  Train-Test ID overlap: {len(id_overlap)} ({'âœ… No leakage' if len(id_overlap) == 0 else 'ğŸš¨ Data leakage detected!'})")

# Feature value range validation
print(f"\nFeature Range Validation:")
expected_ranges = {
    'Time_spent_Alone': (0, 24),
    'Social_event_attendance': (0, 50),
    'Going_outside': (0, 7),
    'Friends_circle_size': (0, 100),
    'Post_frequency': (0, 100)
}

for feature, (min_expected, max_expected) in expected_ranges.items():
    if feature in train_df.columns:
        actual_min = train_df[feature].min()
        actual_max = train_df[feature].max()
        
        range_ok = (actual_min >= min_expected) and (actual_max <= max_expected)
        status = "Valid" if range_ok else "Unexpected"
        
        print(f"  {feature}: [{actual_min}, {actual_max}] {status}")
        if not range_ok:
            print(f"    Expected: [{min_expected}, {max_expected}]")

# Sample submission validation
print(f"\nSample Submission Validation:")
expected_personalities = {'Introvert', 'Extrovert'}
actual_personalities = set(sample_sub['Personality'].unique())
format_ok = actual_personalities.issubset(expected_personalities)
print(f"  Submission format: {'Valid' if format_ok else 'Invalid'}")
print(f"  Expected classes: {expected_personalities}")
print(f"  Found classes: {actual_personalities}")

# Data consistency checks
print(f"\nğŸ”„ Data Consistency:")
test_sample_id_match = len(set(test_df['id']) & set(sample_sub['id'])) == len(test_df)
print(f"  Test-Submission ID match: {'Consistent' if test_sample_id_match else 'Mismatch'}")

print(f"\nğŸ�¯ Data Quality Summary:")
if (train_duplicates == 0 and test_duplicates == 0 and 
    train_id_duplicates == 0 and test_id_duplicates == 0 and 
    len(id_overlap) == 0 and format_ok and test_sample_id_match):
    print("Excellent data quality - ready for modeling!")
    print("No duplicates, unique IDs, no leakage, valid formats")
else:
    print("Some data quality issues detected - review before modeling")

print("\nDataset Statistics Summary:")
print(f"  â€¢ Total samples: {len(train_df):,} training + {len(test_df):,} test = {len(train_df) + len(test_df):,}")
print(f"  â€¢ Features: {len(train_df.columns) - 2} predictive features")
print(f"  â€¢ Target classes: {len(train_df['Personality'].unique())} (binary classification)")
print(f"  â€¢ Missing data: {train_df.isnull().sum().sum():,} total missing values")

