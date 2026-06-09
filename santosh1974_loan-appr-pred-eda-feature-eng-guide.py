# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind

# Machine learning libraries for advanced analysis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

# Color palette for consistent visualizations
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"\nTraining set memory usage: {train_df.memory_usage().sum() / 1024**2:.2f} MB")


# Display first few rows
print("First 5 rows of training data:")
display(train_df.head())

# Data types and basic info
print("\nData Types and Non-null Counts:")
train_df.info()

# Statistical summary
print("\nStatistical Summary:")
display(train_df.describe(include='all'))

# Identify numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

# Remove 'id' and 'loan_status' from feature lists
if 'id' in numerical_cols:
    numerical_cols.remove('id')
if 'loan_status' in numerical_cols:
    numerical_cols.remove('loan_status')

print(f"\nNumerical Features ({len(numerical_cols)}): {numerical_cols}")
print(f"\nCategorical Features ({len(categorical_cols)}): {categorical_cols}")


# Target distribution
target_distribution = train_df['loan_status'].value_counts()
target_percentage = train_df['loan_status'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
axes[0].bar(target_distribution.index, target_distribution.values, color=['#2ca02c', '#d62728'])
axes[0].set_xlabel('Loan Status')
axes[0].set_ylabel('Count')
axes[0].set_title('Target Variable Distribution (Count)')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Rejected (0)', 'Approved (1)'])

# Add count labels
for i, v in enumerate(target_distribution.values):
    axes[0].text(i, v + 100, str(v), ha='center')

# Pie chart
axes[1].pie(target_percentage.values, labels=['Rejected', 'Approved'], 
            autopct='%1.2f%%', colors=['#d62728', '#2ca02c'], startangle=90)
axes[1].set_title('Target Variable Distribution (Percentage)')

plt.tight_layout()
plt.show()

print(f"Target Variable Distribution:")
print(f"Rejected (0): {target_distribution.get(0, 0)} ({target_percentage.get(0, 0):.2f}%)")
print(f"Approved (1): {target_distribution.get(1, 0)} ({target_percentage.get(1, 0):.2f}%)")
print(f"\nClass Imbalance Ratio: {target_distribution.get(0, 0) / target_distribution.get(1, 1):.2f}")


def missing_value_analysis(df, dataset_name="Dataset"):
    """Comprehensive missing value analysis"""
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(missing_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        axes[0].barh(missing_df['Column'], missing_df['Missing_Percentage'], color='#ff7f0e')
        axes[0].set_xlabel('Missing Percentage (%)')
        axes[0].set_title(f'Missing Values in {dataset_name}')
        axes[0].invert_yaxis()
        
        # Heatmap
        sns.heatmap(df[missing_df['Column']].isnull(), cbar=True, yticklabels=False, 
                    cmap='RdYlBu', ax=axes[1])
        axes[1].set_title('Missing Value Pattern')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nMissing Values Summary for {dataset_name}:")
        display(missing_df)
    else:
        print(f"\nNo missing values found in {dataset_name}!")
    
    return missing_df

# Analyze missing values
train_missing = missing_value_analysis(train_df, "Training Set")
test_missing = missing_value_analysis(test_df, "Test Set")

# Missing value patterns
if len(train_missing) > 0:
    print("\nMissing Value Correlation Matrix:")
    missing_corr = train_df.isnull().corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(missing_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation of Missing Values')
    plt.show()


# Check for duplicate rows
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()

print(f"Duplicate rows in training set: {train_duplicates}")
print(f"Duplicate rows in test set: {test_duplicates}")

# Check for duplicate IDs
if 'id' in train_df.columns:
    train_id_duplicates = train_df['id'].duplicated().sum()
    test_id_duplicates = test_df['id'].duplicated().sum()
    print(f"\nDuplicate IDs in training set: {train_id_duplicates}")
    print(f"Duplicate IDs in test set: {test_id_duplicates}")


# Check for potential data type issues
def check_data_types(df):
    """Check for potential data type mismatches"""
    issues = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if it could be numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                issues.append(f"{col}: Could be converted to numeric")
            except:
                pass
            
            # Check if it could be datetime
            try:
                pd.to_datetime(df[col], errors='raise')
                issues.append(f"{col}: Could be converted to datetime")
            except:
                pass
    
    if issues:
        print("Potential data type conversions:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No obvious data type issues detected")

check_data_types(train_df)


# Analyze numerical features
if len(numerical_cols) > 0:
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numerical_cols):
        if idx < len(axes):
            # Histogram with KDE
            axes[idx].hist(train_df[col].dropna(), bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
            axes[idx].set_title(f'{col} Distribution')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            
            # Add statistics
            mean = train_df[col].mean()
            median = train_df[col].median()
            axes[idx].axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
            axes[idx].axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
            axes[idx].legend()
    
    # Remove empty subplots
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


def detect_outliers(df, columns, method='IQR'):
    """Detect outliers using IQR or Z-score method"""
    outlier_summary = []
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for col in columns:
        if col in df.columns:
            if method == 'IQR':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            else:  # Z-score method
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = df[col][z_scores > 3]
            
            outlier_percentage = (len(outliers) / len(df)) * 100
            outlier_summary.append({
                'Feature': col,
                'Outlier_Count': len(outliers),
                'Outlier_Percentage': outlier_percentage,
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Mean': df[col].mean(),
                'Std': df[col].std()
            })
    
    outlier_df = pd.DataFrame(outlier_summary)
    
    # Visualize outlier percentages
    if len(outlier_df) > 0:
        axes[0].barh(outlier_df['Feature'], outlier_df['Outlier_Percentage'], color='#ff7f0e')
        axes[0].set_xlabel('Outlier Percentage (%)')
        axes[0].set_title(f'Outliers Detection ({method} Method)')
        axes[0].invert_yaxis()
        
        # Box plots
        if len(numerical_cols) <= 10:
            train_df[numerical_cols].boxplot(ax=axes[1], rot=45)
            axes[1].set_title('Box Plots for Numerical Features')
        else:
            # Select top 10 features with most outliers
            top_outlier_cols = outlier_df.nlargest(10, 'Outlier_Percentage')['Feature'].tolist()
            train_df[top_outlier_cols].boxplot(ax=axes[1], rot=45)
            axes[1].set_title('Box Plots for Top 10 Features with Outliers')
        
        plt.tight_layout()
        plt.show()
        
        print("\nOutlier Summary:")
        display(outlier_df.sort_values('Outlier_Percentage', ascending=False))
    
    return outlier_df

# Detect outliers
outlier_summary = detect_outliers(train_df, numerical_cols, method='IQR')


# Calculate skewness and kurtosis
skewness_kurtosis = []

for col in numerical_cols:
    if col in train_df.columns:
        skewness_kurtosis.append({
            'Feature': col,
            'Skewness': train_df[col].skew(),
            'Kurtosis': train_df[col].kurtosis(),
            'Transformation_Needed': abs(train_df[col].skew()) > 1
        })

sk_df = pd.DataFrame(skewness_kurtosis).sort_values('Skewness', key=abs, ascending=False)

print("\nSkewness and Kurtosis Analysis:")
display(sk_df)

# Visualize skewness
plt.figure(figsize=(12, 6))
colors_sk = ['red' if x else 'green' for x in sk_df['Transformation_Needed']]
plt.barh(sk_df['Feature'], sk_df['Skewness'], color=colors_sk)
plt.xlabel('Skewness')
plt.title('Feature Skewness (Red: Transformation Recommended)')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=1, color='orange', linestyle='--', alpha=0.5, label='Threshold')
plt.axvline(x=-1, color='orange', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


if len(categorical_cols) > 0:
    categorical_summary = []
    
    for col in categorical_cols:
        categorical_summary.append({
            'Feature': col,
            'Unique_Values': train_df[col].nunique(),
            'Missing_Count': train_df[col].isnull().sum(),
            'Missing_Percentage': (train_df[col].isnull().sum() / len(train_df)) * 100,
            'Most_Frequent': train_df[col].mode()[0] if not train_df[col].mode().empty else None,
            'Most_Frequent_Count': train_df[col].value_counts().iloc[0] if len(train_df[col].value_counts()) > 0 else 0
        })
    
    cat_summary_df = pd.DataFrame(categorical_summary).sort_values('Unique_Values', ascending=False)
    
    print("\nCategorical Features Summary:")
    display(cat_summary_df)
    
    # Visualize categorical distributions
    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(categorical_cols):
        if idx < len(axes):
            value_counts = train_df[col].value_counts().head(10)
            
            if len(value_counts) <= 10:
                axes[idx].bar(range(len(value_counts)), value_counts.values, color='#2ca02c')
                axes[idx].set_xticks(range(len(value_counts)))
                axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            else:
                axes[idx].bar(range(10), value_counts.values[:10], color='#2ca02c')
                axes[idx].set_xticks(range(10))
                axes[idx].set_xticklabels(value_counts.index[:10], rotation=45, ha='right')
            
            axes[idx].set_title(f'{col} Distribution (Top 10)')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Count')
    
    # Remove empty subplots
    for idx in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


def analyze_categorical_target_relationship(df, cat_cols, target_col='loan_status'):
    """Analyze relationship between categorical features and target"""
    
    significant_features = []
    
    for col in cat_cols:
        if col in df.columns:
            # Create contingency table
            contingency = pd.crosstab(df[col], df[target_col])
            
            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            # Calculate CramÃ©r's V
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            
            significant_features.append({
                'Feature': col,
                'Chi2_Statistic': chi2,
                'P_Value': p_value,
                'Cramers_V': cramers_v,
                'Significant': p_value < 0.05
            })
    
    significance_df = pd.DataFrame(significant_features).sort_values('Cramers_V', ascending=False)
    
    print("\nCategorical Features - Target Relationship:")
    display(significance_df)
    
    # Visualize top features
    top_features = significance_df.head(6)
    
    if len(top_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_features.iterrows()):
            if idx < 6:
                col = row['Feature']
                
                # Calculate approval rates
                approval_rates = df.groupby(col)[target_col].mean().sort_values(ascending=False)
                
                axes[idx].bar(range(len(approval_rates)), approval_rates.values, 
                             color=['#2ca02c' if x > 0.5 else '#d62728' for x in approval_rates.values])
                axes[idx].set_xticks(range(len(approval_rates)))
                axes[idx].set_xticklabels(approval_rates.index, rotation=45, ha='right')
                axes[idx].set_ylabel('Approval Rate')
                axes[idx].set_title(f'{col} - Approval Rate')
                axes[idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    return significance_df

if len(categorical_cols) > 0:
    cat_significance = analyze_categorical_target_relationship(train_df, categorical_cols)


def analyze_numerical_target_relationship(df, num_cols, target_col='loan_status'):
    """Analyze relationship between numerical features and target"""
    
    # Statistical tests
    statistical_tests = []
    
    for col in num_cols:
        if col in df.columns:
            approved = df[df[target_col] == 1][col].dropna()
            rejected = df[df[target_col] == 0][col].dropna()
            
            # T-test
            t_stat, p_value = ttest_ind(approved, rejected)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((approved.std()**2 + rejected.std()**2) / 2)
            cohens_d = (approved.mean() - rejected.mean()) / pooled_std if pooled_std != 0 else 0
            
            statistical_tests.append({
                'Feature': col,
                'Approved_Mean': approved.mean(),
                'Rejected_Mean': rejected.mean(),
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'Cohens_D': cohens_d,
                'Significant': p_value < 0.05
            })
    
    stats_df = pd.DataFrame(statistical_tests).sort_values('P_Value')
    
    print("\nNumerical Features - Target Relationship:")
    display(stats_df)
    
    # Visualize distributions for top features
    top_features = stats_df[stats_df['Significant']].head(6)
    
    if len(top_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_features.iterrows()):
            if idx < 6:
                col = row['Feature']
                
                # Create violin plot
                df_plot = df[[col, target_col]].dropna()
                
                approved = df_plot[df_plot[target_col] == 1][col]
                rejected = df_plot[df_plot[target_col] == 0][col]
                
                axes[idx].violinplot([rejected, approved], positions=[0, 1], 
                                    showmeans=True, showmedians=True)
                axes[idx].set_xticks([0, 1])
                axes[idx].set_xticklabels(['Rejected', 'Approved'])
                axes[idx].set_ylabel(col)
                axes[idx].set_title(f'{col} by Loan Status')
        
        plt.tight_layout()
        plt.show()
    
    return stats_df

if len(numerical_cols) > 0:
    num_significance = analyze_numerical_target_relationship(train_df, numerical_cols)


# Correlation matrix for numerical features
if len(numerical_cols) > 1:
    correlation_matrix = train_df[numerical_cols + ['loan_status']].corr()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'Feature_1': correlation_matrix.columns[i],
                    'Feature_2': correlation_matrix.columns[j],
                    'Correlation': correlation_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print("\nHighly Correlated Feature Pairs (|correlation| > 0.7):")
        display(pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False))
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', mask=mask, 
                cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()
    
    # Target correlations
    target_corr = correlation_matrix['loan_status'].drop('loan_status').sort_values(key=abs, ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors_corr = ['#2ca02c' if x > 0 else '#d62728' for x in target_corr.values]
    plt.barh(target_corr.index, target_corr.values, color=colors_corr)
    plt.xlabel('Correlation with Target')
    plt.title('Feature Correlations with Loan Status')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Features Correlated with Target:")
    display(pd.DataFrame({'Feature': target_corr.head(10).index, 
                          'Correlation': target_corr.head(10).values}))


# Calculate feature importance
from sklearn.preprocessing import LabelEncoder

# Prepare data for modeling
X = train_df.drop(['loan_status', 'id'], axis=1, errors='ignore')
y = train_df['loan_status']

# Encode categorical variables
le_dict = {}
X_encoded = X.copy()

for col in categorical_cols:
    if col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = X_encoded[col].fillna('missing')
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        le_dict[col] = le

# Fill missing values for numerical features
for col in numerical_cols:
    if col in X_encoded.columns:
        X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())

# Train Random Forest for feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_encoded, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(top_features['Feature'], top_features['Importance'], color='#9467bd')
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 20 Most Important Features:")
display(top_features)


# Analyze interactions between top features
top_5_features = feature_importance.head(5)['Feature'].tolist()

if len(top_5_features) >= 2:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for i in range(len(top_5_features)):
        for j in range(i+1, len(top_5_features)):
            if plot_idx < 4:
                feat1 = top_5_features[i]
                feat2 = top_5_features[j]
                
                if feat1 in numerical_cols and feat2 in numerical_cols:
                    # Scatter plot for numerical features
                    scatter = axes[plot_idx].scatter(X_encoded[feat1], X_encoded[feat2], 
                                                    c=y, cmap='RdYlGn', alpha=0.5)
                    axes[plot_idx].set_xlabel(feat1)
                    axes[plot_idx].set_ylabel(feat2)
                    axes[plot_idx].set_title(f'{feat1} vs {feat2}')
                    plt.colorbar(scatter, ax=axes[plot_idx])
                else:
                    # Heatmap for categorical or mixed features
                    pivot_table = pd.crosstab(X_encoded[feat1], X_encoded[feat2], 
                                             values=y, aggfunc='mean')
                    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', 
                              ax=axes[plot_idx], vmin=0, vmax=1)
                    axes[plot_idx].set_xlabel(feat2)
                    axes[plot_idx].set_ylabel(feat1)
                    axes[plot_idx].set_title(f'Approval Rate: {feat1} vs {feat2}')
                
                plot_idx += 1
    
    plt.tight_layout()
    plt.show()


# Calculate data quality scores
quality_metrics = {
    'Completeness': (1 - train_df.isnull().sum().sum() / (len(train_df) * len(train_df.columns))) * 100,
    'Uniqueness': (1 - train_df.duplicated().sum() / len(train_df)) * 100,
    'Target_Balance': min(train_df['loan_status'].value_counts(normalize=True)) * 100,
    'Feature_Count': len(train_df.columns) - 2,  # Excluding id and target
    'Sample_Size': len(train_df),
    'Feature_to_Sample_Ratio': (len(train_df.columns) - 2) / len(train_df) * 1000
}

quality_df = pd.DataFrame(list(quality_metrics.items()), columns=['Metric', 'Value'])

print("\nData Quality Metrics:")
display(quality_df)

# Visualize quality metrics
fig, ax = plt.subplots(figsize=(10, 6))
colors_quality = ['green' if x > 80 else 'orange' if x > 50 else 'red' 
                  for x in quality_df[quality_df['Metric'].isin(['Completeness', 'Uniqueness', 'Target_Balance'])]['Value']]
ax.bar(quality_df['Metric'][:3], quality_df['Value'][:3], color=colors_quality)
ax.set_ylabel('Percentage')
ax.set_title('Data Quality Scores')
ax.set_ylim(0, 105)

for i, v in enumerate(quality_df['Value'][:3]):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.show()


feature_engineering_suggestions = {
    'Binning': [],
    'Polynomial_Features': [],
    'Log_Transform': [],
    'Scaling_Required': [],
    'Interaction_Terms': []
}

# Identify features for different transformations
for col in numerical_cols:
    if col in train_df.columns:
        skewness = abs(train_df[col].skew())
        
        # Log transformation for highly skewed features
        if skewness > 2:
            feature_engineering_suggestions['Log_Transform'].append(col)
        
        # Binning for features with outliers
        if col in outlier_summary['Feature'].values:
            outlier_pct = outlier_summary[outlier_summary['Feature'] == col]['Outlier_Percentage'].values[0]
            if outlier_pct > 5:
                feature_engineering_suggestions['Binning'].append(col)
        
        # All numerical features need scaling
        feature_engineering_suggestions['Scaling_Required'].append(col)

# Identify potential interaction terms
if len(top_5_features) >= 2:
    for i in range(len(top_5_features)):
        for j in range(i+1, len(top_5_features)):
            feature_engineering_suggestions['Interaction_Terms'].append(
                f"{top_5_features[i]} * {top_5_features[j]}"
            )

print("\n=== FEATURE ENGINEERING RECOMMENDATIONS ===\n")
for strategy, features in feature_engineering_suggestions.items():
    if features:
        print(f"{strategy}:")
        for feat in features[:5]:  # Show top 5 for each category
            print(f"  - {feat}")
        if len(features) > 5:
            print(f"  ... and {len(features)-5} more")
        print()


categorical_engineering = {
    'Target_Encoding_Recommended': [],
    'One_Hot_Encoding_Recommended': [],
    'Frequency_Encoding_Recommended': [],
    'Embedding_Recommended': []
}

for col in categorical_cols:
    if col in train_df.columns:
        cardinality = train_df[col].nunique()
        
        if cardinality <= 5:
            categorical_engineering['One_Hot_Encoding_Recommended'].append(col)
        elif cardinality <= 20:
            categorical_engineering['Target_Encoding_Recommended'].append(col)
        elif cardinality <= 100:
            categorical_engineering['Frequency_Encoding_Recommended'].append(col)
        else:
            categorical_engineering['Embedding_Recommended'].append(col)

print("=== CATEGORICAL ENCODING RECOMMENDATIONS ===\n")
for strategy, features in categorical_engineering.items():
    if features:
        print(f"{strategy}:")
        for feat in features:
            print(f"  - {feat} (cardinality: {train_df[feat].nunique()})")
        print()


print("=" * 60)
print("EXPLORATORY DATA ANALYSIS SUMMARY")
print("=" * 60)

summary_insights = f"""
ğŸ“Š **Dataset Overview:**
- Training samples: {len(train_df):,}
- Test samples: {len(test_df):,}
- Total features: {len(train_df.columns) - 2}
- Numerical features: {len(numerical_cols)}
- Categorical features: {len(categorical_cols)}

ğŸ�¯ **Target Variable:**
- Class distribution: {train_df['loan_status'].value_counts().to_dict()}
- Imbalance ratio: {target_distribution.get(0, 0) / target_distribution.get(1, 1):.2f}
- Recommendation: {'Consider class balancing techniques' if target_distribution.get(0, 0) / target_distribution.get(1, 1) > 1.5 else 'Classes are relatively balanced'}

âš ï¸� **Data Quality:**
- Missing values: {len(train_missing)} features with missing data
- Completeness: {quality_metrics['Completeness']:.2f}%
- Duplicates: {train_duplicates} duplicate rows

ğŸ“ˆ **Top Predictive Features:**
1. {feature_importance.iloc[0]['Feature']} (Importance: {feature_importance.iloc[0]['Importance']:.4f})
2. {feature_importance.iloc[1]['Feature']} (Importance: {feature_importance.iloc[1]['Importance']:.4f})
3. {feature_importance.iloc[2]['Feature']} (Importance: {feature_importance.iloc[2]['Importance']:.4f})

ğŸ”§ **Preprocessing Requirements:**
- Features requiring log transformation: {len(feature_engineering_suggestions['Log_Transform'])}
- Features with significant outliers: {len(feature_engineering_suggestions['Binning'])}
- High cardinality categorical features: {len(categorical_engineering['Embedding_Recommended'])}
"""

print(summary_insights)


modeling_recommendations = """
ğŸš€ **MODELING RECOMMENDATIONS**

1. **Preprocessing Pipeline:**
   - Handle missing values (median/mode imputation or advanced techniques)
   - Apply log transformation to highly skewed features
   - Scale numerical features (StandardScaler or RobustScaler)
   - Encode categorical variables appropriately based on cardinality

2. **Feature Selection:**
   - Use top 20 features from Random Forest importance
   - Consider removing highly correlated features (correlation > 0.95)
   - Create interaction terms for top features

3. **Model Selection:**
   - Start with: LightGBM, XGBoost, CatBoost (handle categoricals well)
   - Consider ensemble methods for better performance
   - Use cross-validation (StratifiedKFold with 5 folds)

4. **Class Imbalance Handling:**
   - Use class_weight='balanced' parameter
   - Try SMOTE or ADASYN for oversampling
   - Consider focal loss for neural networks

5. **Hyperparameter Tuning:**
   - Use Optuna or Bayesian Optimization
   - Focus on: max_depth, learning_rate, n_estimators, subsample
   - Monitor for overfitting with validation curves

6. **Evaluation Strategy:**
   - Primary metric: ROC-AUC (as specified)
   - Also monitor: Precision-Recall AUC, F1-score
   - Use probability calibration if needed

7. **Post-processing:**
   - Optimize probability threshold based on business requirements
   - Consider probability calibration (Platt scaling or Isotonic regression)
"""

print(modeling_recommendations)


next_steps = """
ğŸ“‹ **IMMEDIATE NEXT STEPS:**

1. âœ… Complete data preprocessing based on EDA findings
2. âœ… Create feature engineering pipeline
3. âœ… Split data for validation (80-20 or cross-validation)
4. âœ… Implement baseline model (Logistic Regression)
5. âœ… Train advanced models (Gradient Boosting)
6. âœ… Perform hyperparameter optimization
7. âœ… Create ensemble model
8. âœ… Generate submission file with predicted probabilities

Remember: The competition metric is ROC-AUC, so focus on:
- Probability calibration
- Proper handling of class imbalance
- Feature interactions that improve separation
"""

print(next_steps)

# Save preprocessed data for modeling
print("\nğŸ’¾ Saving preprocessed data...")
# train_df.to_csv('train_preprocessed.csv', index=False)
# test_df.to_csv('test_preprocessed.csv', index=False)
print("âœ… EDA Complete! Ready for modeling phase.")




