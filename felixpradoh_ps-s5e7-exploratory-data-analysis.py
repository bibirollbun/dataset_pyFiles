# Enhanced library imports for comprehensive analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis and hypothesis testing
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Machine learning and preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Configuration
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Random seed for reproducibility
np.random.seed(513)

print("âœ… Enhanced libraries imported successfully!")
print(f"ğŸ“Š Pandas version: {pd.__version__}")
print(f"ğŸ“ˆ NumPy version: {np.__version__}")
print(f"ğŸ�¨ Matplotlib version: {plt.matplotlib.__version__}")
print(f"ğŸ“Š Seaborn version: {sns.__version__}")
print("ğŸ”§ All required packages are available!")


# Load the training and test datasets. Display the first few rows of the training set to get an initial overview of the data structure.

input_path = "/kaggle/input/playground-series-s5e7"

df_train = pd.read_csv(f'{input_path}/train.csv')
df_test = pd.read_csv(f'{input_path}/test.csv')

# Show the first rows of the training set
df_train.head()


# Consistent color configuration for features
# Define a unique color palette for each feature
import itertools
from matplotlib import cm

# Get all numerical and categorical columns (excluding the target)
all_features = [col for col in df_train.columns if col != 'Personality']

# Use a matplotlib color palette
color_palette = sns.color_palette('tab20', n_colors=len(all_features))
feature_colors = dict(zip(all_features, color_palette))

def get_feature_colors(features):
    """Returns a list of colors for a list of features."""
    return [feature_colors.get(f, '#333333') for f in features]


# Drop the 'id' column if present
df_train = df_train.drop(columns=['id'], errors='ignore')
df_test = df_test.drop(columns=['id'], errors='ignore')


# Display a statistical summary of the training data to understand the distribution and range of each feature.
df_train.describe()


# Missing Values Visualization
# Visualize missing values in the training dataset to identify potential data quality issues.

plt.figure(figsize=(10, 6))
sns.heatmap(df_train.isnull(), cbar=False, cmap='viridis')
plt.title('NaNs in the train dataset')
plt.show()



# Create DataFrames with MultiIndex where the first level is 'feature' and the second is 'set'
train_info = pd.DataFrame({
    'feature': df_train.columns,
    'dtype': df_train.dtypes.values,
    'nunique': df_train.nunique().values,
    'set': 'train'
})

test_info = pd.DataFrame({
    'feature': df_test.columns,
    'dtype': df_test.dtypes.values,
    'nunique': df_test.nunique().values,
    'set': 'test'
})
# Combine both into a single DataFrame
combined_info = pd.concat([train_info, test_info], ignore_index=True)

# Create a pivot table to show dtype and nunique by set and feature
pivot_table = combined_info.pivot_table(
    index='feature',
    columns='set',
    values=['dtype', 'nunique'],
    aggfunc='first'
)

display(pivot_table)


TARGET = 'Personality'

if TARGET in df_train.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_train[TARGET], kde=False, bins=30)
    plt.title('Target Variable Distribution')
    plt.xlabel(TARGET)
    plt.ylabel('Count')
    plt.show()
else:
    print('No target column found in training set.')


# Correlation heatmap for numerical features with colorbar normalized between -1 and 1
plt.figure(figsize=(12, 8))
corr = df_train.corr(numeric_only=True)
sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    square=True,
    vmin=-1,
    vmax=1
)
plt.title('Correlation Heatmap (Training Set)')
plt.show()


# Plot distributions for all numerical features in the training set
num_cols = df_train.select_dtypes(include=['float64', 'int64']).columns

n_cols = 3
n_rows = len(num_cols) // n_cols + int(len(num_cols) % n_cols > 0)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    color = feature_colors.get(col, '#333333')
    axes[i].hist(df_train[col].dropna(), bins=30, color=color, alpha=0.85)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Numerical Feature Distributions (Training Set)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# Show value counts for categorical features in the training set in a pivot_table style
cat_cols = df_train.select_dtypes(include=['object', 'category']).columns

if len(cat_cols) > 0:
    value_counts_dict = {}
    for col in cat_cols:
        value_counts_dict[col] = df_train[col].value_counts()
    # Combine into a DataFrame and fill NaN with 0
    value_counts_df = pd.DataFrame(value_counts_dict).fillna(0).astype(int)
    display(value_counts_df)
    # Plot value counts for up to 4 categorical features in a single figure with 2x2 subplots
    n = min(4, len(cat_cols))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols[:n]):
        bar_colors = get_feature_colors([col]*value_counts_df[col].shape[0])
        sns.barplot(x=value_counts_df.index, y=value_counts_df[col], color=feature_colors.get(col, '#333333'), ax=axes[i])
        axes[i].set_title(f'Value Counts for {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
    # Hide unused subplots
    for j in range(n, 4):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print('No categorical features found in the training set.')


# Show missing values by variable type in a summary DataFrame

def missing_summary(df, name):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_missing = df[num_cols].isnull().sum()
    cat_missing = df[cat_cols].isnull().sum()
    summary = pd.DataFrame({
        'Feature': list(num_missing.index) + list(cat_missing.index),
        'Type': ['Numerical'] * len(num_missing) + ['Categorical'] * len(cat_missing),
        'NaNs_Count': list(num_missing.values) + list(cat_missing.values),
        'NaNs_Pct': list(num_missing.values / len(df) * 100) + list(cat_missing.values / len(df) * 100)
    })
    summary = summary[summary['NaNs_Count'] > 0].sort_values('NaNs_Count', ascending=False)
    summary.insert(0, 'Set', name)
    return summary

train_missing = missing_summary(df_train, 'Train')
test_missing = missing_summary(df_test, 'Test')
missing_table = pd.concat([train_missing, test_missing], ignore_index=True)

# display(missing_table)

# Show as a pivot table: index=Feature, columns=Set, values=NaNs_Count and NaNs_Pct
pivot = missing_table.pivot_table(
    index='Feature',
    columns='Set',
    values=['NaNs_Count', 'NaNs_Pct'],
    aggfunc='first'
)

# Format NaNs_Pct to 2 decimals
pivot.loc[:, ('NaNs_Pct', slice(None))] = pivot.loc[:, ('NaNs_Pct', slice(None))].round(2)

display(pivot)


# Detect outliers in numerical features using the IQR method
def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] < lower) | (df[col] > upper)]

num_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
outlier_counts = {}
for col in num_cols:
    outliers = detect_outliers(df_train, col)
    outlier_counts[col] = len(outliers)

print('Outlier counts per numerical feature:')
for col, count in outlier_counts.items():
    print(f'{col}: {count}')

# Visualize outliers for the first five numerical features using subplots (1 row, 5 columns, vertical boxplots)
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

first_five = list(num_cols[:5])
fig, axes = plt.subplots(1, 5, figsize=(20, 10))
for i, col in enumerate(first_five):
    color = feature_colors.get(col, '#333333')
    sns.boxplot(y=df_train[col], ax=axes[i], orient='v', color=color)
    axes[i].set_xlabel('')
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.show()

print('Outliers may indicate data entry errors, rare events, or valid extreme values. Their treatment depends on domain knowledge and modeling goals.')


# Impute missing values: numerical features with median, categorical features with mode
df_train_imputed = df_train.copy()
df_test_imputed = df_test.copy()

# Numerical features: fill NaNs with median
num_cols = df_train_imputed.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    median_value = df_train_imputed[col].median()
    df_train_imputed[col].fillna(median_value, inplace=True)
    if col in df_test_imputed.columns:
        df_test_imputed[col].fillna(median_value, inplace=True)  # Use train median for test set

# Categorical features: fill NaNs with mode
cat_cols = df_train_imputed.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    mode_value = df_train_imputed[col].mode()[0]
    df_train_imputed[col].fillna(mode_value, inplace=True)
    if col in df_test_imputed.columns:
        df_test_imputed[col].fillna(mode_value, inplace=True)  # Use train mode for test set


# Calculate and plot mutual information between features and the target using imputed data

target_col = TARGET
if target_col in df_train_imputed.columns:
    X = df_train_imputed.drop(columns=[target_col])
    y = df_train_imputed[target_col]
    # Encode categorical variables
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include=['object', 'category']).columns:
        X_enc[col] = X_enc[col].astype('category').cat.codes
    # Choose MI function based on target type
    if y.nunique() < 20 and y.dtype in ['int64', 'int32', 'category', 'object']:
        mi = mutual_info_classif(X_enc, y, discrete_features='auto', random_state=513)
    else:
        mi = mutual_info_regression(X_enc, y, discrete_features='auto', random_state=513)
    mi_series = pd.Series(mi, index=X_enc.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    # Usar colores consistentes
    bar_colors = get_feature_colors(mi_series.index)
    sns.barplot(x=mi_series.values, y=mi_series.index, palette=bar_colors)
    plt.title('Mutual Information with Target (Imputed Data)')
    plt.xlabel('Mutual Information')
    plt.ylabel('Feature')
    plt.show()
else:
    print('No target column found in imputed training set.')



# Advanced Statistical Analysis
print("ğŸ”¬ ADVANCED STATISTICAL ANALYSIS")
print("="*60)

# 1. Normality Testing for Numerical Features
print("\nğŸ“Š 1. NORMALITY TESTING (Shapiro-Wilk Test)")
print("-" * 45)

normality_results = []
for col in num_cols:
    if col != TARGET:
        # Sample 5000 random observations for large datasets
        sample_size = min(5000, len(df_train_imputed[col]))
        sample_data = df_train_imputed[col].sample(sample_size, random_state=42)
        
        # Shapiro-Wilk test
        statistic, p_value = stats.shapiro(sample_data)
        is_normal = p_value > 0.05
        
        normality_results.append({
            'Feature': col,
            'Statistic': statistic,
            'P-value': p_value,
            'Is_Normal': is_normal,
            'Interpretation': 'Normal' if is_normal else 'Non-Normal'
        })

normality_df = pd.DataFrame(normality_results)
display(normality_df.style.hide(axis='index'))

# 2. Feature-Target Correlation Significance Testing
print("\n\nğŸ“ˆ 2. FEATURE-TARGET CORRELATION SIGNIFICANCE")
print("-" * 50)

if TARGET in df_train_imputed.columns:
    correlation_results = []
    target_data = df_train_imputed[TARGET]
    
    # Encode target if categorical
    if target_data.dtype == 'object':
        le = LabelEncoder()
        target_encoded = le.fit_transform(target_data)
    else:
        target_encoded = target_data
    
    for col in num_cols:
        if col != TARGET:
            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(df_train_imputed[col], target_encoded)
            
            # Spearman correlation (non-parametric)
            spearman_corr, spearman_p = spearmanr(df_train_imputed[col], target_encoded)
            
            correlation_results.append({
                'Feature': col,
                'Pearson_Corr': pearson_corr,
                'Pearson_P': pearson_p,
                'Spearman_Corr': spearman_corr,
                'Spearman_P': spearman_p,
                'Significant': 'Yes' if min(pearson_p, spearman_p) < 0.05 else 'No'
            })
    
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values('Pearson_P')
    display(corr_df.round(4).style.hide(axis='index'))

# 3. Statistical Summary by Target Class
print("\n\nğŸ“Š 3. STATISTICAL SUMMARY BY TARGET CLASS")
print("-" * 45)

if TARGET in df_train_imputed.columns:
    summary_by_target = df_train_imputed.groupby(TARGET)[num_cols].agg(['mean', 'std', 'median'])
    display(summary_by_target.round(3))


# Detect columns containing only 'yes' and 'no' values
yes_no_cols = [
    col for col in df_train_imputed.columns 
    if df_train_imputed[col].dropna().isin(['Yes', 'No']).all()
]

# Convert these columns to 1/0 (float type)
for col in yes_no_cols:
    df_train_imputed[col] = df_train_imputed[col].map({'Yes': 1, 'No': 0}).astype(float)

print(f"âœ… Converted 'yes'/'no' columns to 1/0: {yes_no_cols}")



# Advanced Visualization Suite
print("ğŸ�¨ ADVANCED VISUALIZATION ANALYSIS")
print("="*60)

# 1. Violin Plots for Feature Distribution by Target Class
print("\nğŸ“Š 1. DISTRIBUTION ANALYSIS BY TARGET CLASS")
print("-" * 45)

if TARGET in df_train_imputed.columns and len(num_cols) > 0:
    # Selecciona solo las columnas numÃ©ricas de top_features
    top_features = mi_series.head(6).index.tolist()
    numeric_top_features = [f for f in top_features if pd.api.types.is_numeric_dtype(df_train_imputed[f])]
    
    # Crea los violin plots solo para las columnas numÃ©ricas
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(numeric_top_features):
        color = feature_colors.get(feature, '#333333')
        sns.violinplot(data=df_train_imputed, x=TARGET, y=feature, ax=axes[i], palette=[color, color])
        axes[i].set_title(f'Distribution of {feature} by {TARGET}')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel("")  # Remove x-axis label
    
    # Oculta los ejes no usados si hay menos de 6 features numÃ©ricas
    for j in range(len(numeric_top_features), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# 2. Advanced Correlation Analysis with Clustering
print("\n\nğŸ”— 2. HIERARCHICAL CORRELATION CLUSTERING")
print("-" * 45)

# Create a clustered correlation heatmap
plt.figure(figsize=(14, 10))
corr_matrix = df_train_imputed.corr(numeric_only=True)

# Use hierarchical clustering to reorder features
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Convert correlation to distance matrix
distance_matrix = 1 - np.abs(corr_matrix)
condensed_distances = squareform(distance_matrix)
linkage_matrix = linkage(condensed_distances, method='ward')

# Create dendrogram to get optimal ordering
dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
optimal_order = dendro['leaves']

# Reorder correlation matrix
reordered_corr = corr_matrix.iloc[optimal_order, optimal_order]

# Create clustered heatmap
sns.heatmap(reordered_corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, vmin=-1, vmax=1)
plt.title('Hierarchically Clustered Correlation Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Feature Interaction Analysis
print("\n\nğŸ”„ 3. FEATURE INTERACTION ANALYSIS")
print("-" * 40)

if TARGET in df_train_imputed.columns and len(top_features) >= 2:
    # Create interaction plots for top features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(min(4, len(top_features)-1)):
        feature1 = top_features[i]
        feature2 = top_features[i+1]
        color1 = feature_colors.get(feature1, '#333333')
        color2 = feature_colors.get(feature2, '#333333')
        # Scatter plot colored by target
        for idx, personality in enumerate(df_train_imputed[TARGET].unique()):
            mask = df_train_imputed[TARGET] == personality
            axes[i].scatter(df_train_imputed.loc[mask, feature1], 
                          df_train_imputed.loc[mask, feature2], 
                          alpha=0.6, label=personality, color=[color1, color2][idx % 2])
        axes[i].set_xlabel(feature1)
        axes[i].set_ylabel(feature2)
        axes[i].set_title(f'{feature1} vs {feature2} by {TARGET}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 4. Distribution Comparison Analysis
print("\n\nğŸ“ˆ 4. DISTRIBUTION COMPARISON BY TARGET")
print("-" * 45)

if TARGET in df_train_imputed.columns:
    # Create comprehensive distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Box plots
    top_4_features = top_features[:4]
    for i, feature in enumerate(top_4_features):
        row, col = i // 2, i % 2
        color = feature_colors.get(feature, '#333333')
        sns.boxplot(data=df_train_imputed, x=TARGET, y=feature, ax=axes[row, col], palette=[color, color])
        axes[row, col].set_title(f'{feature} Distribution by {TARGET}')
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].set_xlabel("")
    
    plt.tight_layout()
    plt.show()





