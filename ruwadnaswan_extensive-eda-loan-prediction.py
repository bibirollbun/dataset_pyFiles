


# ============================================================================
# COMPREHENSIVE EDA FOR LOAN REPAYMENT PREDICTION
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, kstest, shapiro
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, f_classif
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)






# ============================================================================
# 1. DATA LOADING AND BASIC EXPLORATION
# ============================================================================

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {train.columns.tolist()}")

# Basic info
print("\n" + "="*80)
print("DATA TYPES AND MEMORY USAGE")
print("="*80)
print(train.info())

# First few rows
print("\n" + "="*80)
print("FIRST 5 ROWS")
print("="*80)
print(train.head())

# Statistical summary
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print(train.describe())

# Target distribution
print("\n" + "="*80)
print("TARGET VARIABLE DISTRIBUTION")
print("="*80)
print(train['loan_paid_back'].value_counts())
print(f"\nClass Balance: {train['loan_paid_back'].value_counts(normalize=True)}")



# ============================================================================
# 2. MISSING VALUE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("MISSING VALUES ANALYSIS")
print("="*80)

missing_train = pd.DataFrame({
    'Column': train.columns,
    'Missing_Count': train.isnull().sum(),
    'Missing_Percentage': (train.isnull().sum() / len(train)) * 100,
    'Dtype': train.dtypes
})
missing_train = missing_train[missing_train['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
print(missing_train)

# Visualize missing values
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Missing value heatmap
sns.heatmap(train.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=axes[0])
axes[0].set_title('Missing Values Heatmap (Train)', fontsize=14, fontweight='bold')

# Missing value bar plot
if len(missing_train) > 0:
    missing_train.plot(x='Column', y='Missing_Percentage', kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Missing Value Percentages', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Missing %')
    axes[1].set_xlabel('Columns')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
else:
    axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=16)

plt.tight_layout()
plt.show()



# ============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TARGET VARIABLE DEEP DIVE")
print("="*80)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Target Distribution', 'Target Percentage'),
    specs=[[{'type': 'bar'}, {'type': 'pie'}]]
)

target_counts = train['loan_paid_back'].value_counts()
fig.add_trace(
    go.Bar(x=['Defaulted (0)', 'Paid Back (1)'], y=target_counts.values, 
           marker_color=['#FF6B6B', '#4ECDC4'],
           text=target_counts.values, textposition='auto'),
    row=1, col=1
)

fig.add_trace(
    go.Pie(labels=['Defaulted (0)', 'Paid Back (1)'], values=target_counts.values,
           marker_colors=['#FF6B6B', '#4ECDC4']),
    row=1, col=2
)

fig.update_layout(height=400, showlegend=False, title_text="Target Variable Analysis")
fig.show()



# ============================================================================
# 4. NUMERICAL FEATURES ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("NUMERICAL FEATURES ANALYSIS")
print("="*80)

# Identify numerical columns
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in numerical_cols:
    numerical_cols.remove('id')
if 'loan_paid_back' in numerical_cols:
    numerical_cols.remove('loan_paid_back')

print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Distribution plots for numerical features
n_cols = 4
n_rows = int(np.ceil(len(numerical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    sns.histplot(train[col], kde=True, ax=axes[idx], color='steelblue', bins=50)
    axes[idx].set_title(f'{col}\nSkew: {train[col].skew():.2f} | Kurt: {train[col].kurtosis():.2f}', 
                        fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('')

# Hide empty subplots
for idx in range(len(numerical_cols), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold', y=1.001)
plt.tight_layout()
plt.show()

# Skewness and Kurtosis analysis
print("\nSkewness and Kurtosis:")
skew_kurt = pd.DataFrame({
    'Feature': numerical_cols,
    'Skewness': [train[col].skew() for col in numerical_cols],
    'Kurtosis': [train[col].kurtosis() for col in numerical_cols],
    'Mean': [train[col].mean() for col in numerical_cols],
    'Median': [train[col].median() for col in numerical_cols],
    'Std': [train[col].std() for col in numerical_cols]
})
print(skew_kurt.sort_values('Skewness', ascending=False))



# ============================================================================
# 5. CATEGORICAL FEATURES ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CATEGORICAL FEATURES ANALYSIS")
print("="*80)

# Identify categorical columns
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Unique values count
for col in categorical_cols:
    print(f"\n{col}: {train[col].nunique()} unique values")
    print(train[col].value_counts().head(10))

# Visualize categorical features
if len(categorical_cols) > 0:
    n_cols = 3
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*5))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        value_counts = train[col].value_counts()
        axes[idx].bar(range(len(value_counts)), value_counts.values, color='coral')
        axes[idx].set_title(f'{col} Distribution\n({train[col].nunique()} categories)', 
                           fontsize=11, fontweight='bold')
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[idx].set_ylabel('Count')
    
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Distribution of Categorical Features', fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.show()



# ============================================================================
# 6. BIVARIATE ANALYSIS - NUMERICAL VS TARGET
# ============================================================================

print("\n" + "="*80)
print("BIVARIATE ANALYSIS: NUMERICAL FEATURES VS TARGET")
print("="*80)

# Box plots
n_cols = 4
n_rows = int(np.ceil(len(numerical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    sns.boxplot(data=train, x='loan_paid_back', y=col, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'{col} vs Target', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Loan Paid Back')

for idx in range(len(numerical_cols), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Numerical Features vs Target (Box Plots)', fontsize=16, fontweight='bold', y=1.001)
plt.tight_layout()
plt.show()

# Violin plots for key features
key_features = numerical_cols[:8] if len(numerical_cols) > 8 else numerical_cols
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, col in enumerate(key_features):
    sns.violinplot(data=train, x='loan_paid_back', y=col, ax=axes[idx], palette='muted')
    axes[idx].set_title(f'{col} vs Target', fontsize=10, fontweight='bold')

plt.suptitle('Key Numerical Features vs Target (Violin Plots)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Statistical tests (T-test) for numerical features
print("\nT-Test Results (Numerical Features vs Target):")
ttest_results = []
for col in numerical_cols:
    group0 = train[train['loan_paid_back'] == 0][col].dropna()
    group1 = train[train['loan_paid_back'] == 1][col].dropna()
    
    if len(group0) > 0 and len(group1) > 0:
        t_stat, p_value = ttest_ind(group0, group1)
        ttest_results.append({
            'Feature': col,
            'T-Statistic': t_stat,
            'P-Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No',
            'Mean_Default': group0.mean(),
            'Mean_PaidBack': group1.mean(),
            'Difference': group1.mean() - group0.mean()
        })

ttest_df = pd.DataFrame(ttest_results).sort_values('P-Value')
print(ttest_df)




# ============================================================================
# 7. BIVARIATE ANALYSIS - CATEGORICAL VS TARGET
# ============================================================================

print("\n" + "="*80)
print("BIVARIATE ANALYSIS: CATEGORICAL FEATURES VS TARGET")
print("="*80)

if len(categorical_cols) > 0:
    n_cols = 3
    n_rows = int(np.ceil(len(categorical_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*5))
    axes = axes.flatten()
    
    for idx, col in enumerate(categorical_cols):
        cross_tab = pd.crosstab(train[col], train['loan_paid_back'], normalize='index') * 100
        cross_tab.plot(kind='bar', stacked=False, ax=axes[idx], color=['#FF6B6B', '#4ECDC4'])
        axes[idx].set_title(f'{col} vs Loan Paid Back %', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Percentage (%)')
        axes[idx].legend(['Defaulted', 'Paid Back'], loc='best')
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Categorical Features vs Target (Percentage)', fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.show()
    
    # Chi-square tests
    print("\nChi-Square Test Results (Categorical Features vs Target):")
    chi2_results = []
    for col in categorical_cols:
        contingency_table = pd.crosstab(train[col], train['loan_paid_back'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        chi2_results.append({
            'Feature': col,
            'Chi2-Statistic': chi2,
            'P-Value': p_value,
            'DOF': dof,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    chi2_df = pd.DataFrame(chi2_results).sort_values('P-Value')
    print(chi2_df)



# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Correlation matrix
corr_matrix = train[numerical_cols + ['loan_paid_back']].corr()

# Heatmap
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Correlation with target
target_corr = corr_matrix['loan_paid_back'].drop('loan_paid_back').sort_values(ascending=False)
print("\nCorrelation with Target Variable:")
print(target_corr)

# Visualize correlation with target
fig, ax = plt.subplots(figsize=(10, 12))
colors = ['green' if x > 0 else 'red' for x in target_corr.values]
ax.barh(range(len(target_corr)), target_corr.values, color=colors)
ax.set_yticks(range(len(target_corr)))
ax.set_yticklabels(target_corr.index)
ax.set_xlabel('Correlation with Target', fontsize=12)
ax.set_title('Feature Correlation with Loan Paid Back', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.show()

# Highly correlated features (multicollinearity check)
print("\nHighly Correlated Feature Pairs (|correlation| > 0.7):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Feature1': corr_matrix.columns[i],
                'Feature2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
    print(high_corr_df)
else:
    print("No highly correlated pairs found")




# ============================================================================
# 9. OUTLIER DETECTION
# ============================================================================

print("\n" + "="*80)
print("OUTLIER DETECTION")
print("="*80)

# IQR method
outlier_summary = []
for col in numerical_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
    outlier_percentage = (len(outliers) / len(train)) * 100
    
    outlier_summary.append({
        'Feature': col,
        'Outlier_Count': len(outliers),
        'Outlier_Percentage': outlier_percentage,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'Min': train[col].min(),
        'Max': train[col].max()
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Percentage', ascending=False)
print(outlier_df)

# Box plots for outlier visualization
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    sns.boxplot(y=train[col], ax=axes[idx], color='lightblue')
    axes[idx].set_title(f'{col}\nOutliers: {outlier_df[outlier_df["Feature"]==col]["Outlier_Percentage"].values[0]:.2f}%', 
                       fontsize=10, fontweight='bold')

for idx in range(len(numerical_cols), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Outlier Detection (Box Plots)', fontsize=16, fontweight='bold', y=1.001)
plt.tight_layout()
plt.show()



# ============================================================================
# 10. FEATURE IMPORTANCE - MUTUAL INFORMATION
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE - MUTUAL INFORMATION")
print("="*80)

# Prepare data for mutual information
X = train[numerical_cols].fillna(train[numerical_cols].median())
y = train['loan_paid_back']

# Calculate mutual information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=numerical_cols).sort_values(ascending=False)

print("Mutual Information Scores:")
print(mi_scores)

# Visualize
fig, ax = plt.subplots(figsize=(10, 12))
mi_scores.plot(kind='barh', ax=ax, color='teal')
ax.set_xlabel('Mutual Information Score', fontsize=12)
ax.set_title('Feature Importance - Mutual Information', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()



# ============================================================================
# 11. FEATURE INTERACTIONS & ENGINEERING IDEAS
# ============================================================================

print("\n" + "="*80)
print("FEATURE INTERACTIONS & ENGINEERING")
print("="*80)

# Create some feature interactions
if 'monthly_income' in train.columns and 'loan_amount' in train.columns:
    train['loan_to_income_ratio'] = train['loan_amount'] / (train['monthly_income'] * 12 + 1)
    print("\nCreated: loan_to_income_ratio")

if 'credit_score' in train.columns and 'debt_to_income_ratio' in train.columns:
    train['credit_debt_interaction'] = train['credit_score'] * (1 / (train['debt_to_income_ratio'] + 1))
    print("Created: credit_debt_interaction")

if 'loan_amount' in train.columns and 'interest_rate' in train.columns:
    train['total_interest_paid'] = train['loan_amount'] * (train['interest_rate'] / 100)
    print("Created: total_interest_paid")

if 'current_balance' in train.columns and 'total_credit_limit' in train.columns:
    train['credit_utilization'] = train['current_balance'] / (train['total_credit_limit'] + 1)
    print("Created: credit_utilization")

if 'num_of_open_accounts' in train.columns and 'age' in train.columns:
    train['accounts_per_age'] = train['num_of_open_accounts'] / (train['age'] + 1)
    print("Created: accounts_per_age")

if 'installment' in train.columns and 'monthly_income' in train.columns:
    train['installment_to_income'] = train['installment'] / (train['monthly_income'] + 1)
    print("Created: installment_to_income")

# Analyze new features
new_features = ['loan_to_income_ratio', 'credit_debt_interaction', 'total_interest_paid', 
                'credit_utilization', 'accounts_per_age', 'installment_to_income']
available_new_features = [f for f in new_features if f in train.columns]

if available_new_features:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(available_new_features):
        if idx < len(axes):
            sns.boxplot(data=train, x='loan_paid_back', y=col, ax=axes[idx], palette='Set3')
            axes[idx].set_title(f'{col} vs Target', fontsize=10, fontweight='bold')
    
    for idx in range(len(available_new_features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Engineered Features vs Target', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Correlation of new features with target
    if available_new_features:
        new_corr = train[available_new_features + ['loan_paid_back']].corr()['loan_paid_back'].drop('loan_paid_back')
        print("\nCorrelation of Engineered Features with Target:")
        print(new_corr.sort_values(ascending=False))


# ============================================================================
# 12. ADVANCED VISUALIZATIONS - PAIR PLOTS
# ============================================================================

print("\n" + "="*80)
print("PAIR PLOTS FOR KEY FEATURES")
print("="*80)

# Select top features based on correlation with target
top_features = abs(target_corr).nlargest(5).index.tolist()
print(f"Top 5 features for pair plot: {top_features}")

if len(top_features) >= 3:
    sample_data = train[top_features[:5] + ['loan_paid_back']].sample(min(5000, len(train)), random_state=42)
    
    pairplot = sns.pairplot(sample_data, hue='loan_paid_back', palette='Set1', 
                            diag_kind='kde', plot_kws={'alpha': 0.6}, height=3)
    pairplot.fig.suptitle('Pair Plot - Top Features', y=1.001, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()




# ============================================================================
# 13. DISTRIBUTION COMPARISON: TRAIN VS TEST
# ============================================================================

print("\n" + "="*80)
print("TRAIN VS TEST DISTRIBUTION COMPARISON")
print("="*80)

# Compare distributions
n_cols = 4
n_rows = int(np.ceil(len(numerical_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    if col in test.columns:
        axes[idx].hist(train[col].dropna(), bins=50, alpha=0.6, label='Train', color='blue', density=True)
        axes[idx].hist(test[col].dropna(), bins=50, alpha=0.6, label='Test', color='red', density=True)
        axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[idx].legend()
        axes[idx].set_xlabel('')

for idx in range(len(numerical_cols), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Train vs Test Distribution Comparison', fontsize=16, fontweight='bold', y=1.001)
plt.tight_layout()
plt.show()

# KS test for train vs test
print("\nKolmogorov-Smirnov Test (Train vs Test):")
ks_results = []
for col in numerical_cols:
    if col in test.columns:
        train_data = train[col].dropna()
        test_data = test[col].dropna()
        
        if len(train_data) > 0 and len(test_data) > 0:
            ks_stat, p_value = kstest(train_data, test_data.values)
            ks_results.append({
                'Feature': col,
                'KS-Statistic': ks_stat,
                'P-Value': p_value,
                'Same_Distribution': 'Yes' if p_value > 0.05 else 'No'
            })

if ks_results:
    ks_df = pd.DataFrame(ks_results).sort_values('P-Value')
    print(ks_df)




# ============================================================================
# 14. PCA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*80)

# Prepare data
X_pca = train[numerical_cols].fillna(train[numerical_cols].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# Fit PCA
pca = PCA()
pca.fit(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[0].set_title('Scree Plot - Explained Variance by Component', fontsize=14, fontweight='bold')

axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='-', color='darkorange')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
axes[1].set_xlabel('Number of Components', fontsize=12)
axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nNumber of components for 95% variance: {np.argmax(cumulative_variance >= 0.95) + 1}")
print(f"Number of components for 99% variance: {np.argmax(cumulative_variance >= 0.99) + 1}")

# PCA with 2 components for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=train['loan_paid_back'], 
                     cmap='coolwarm', alpha=0.6, s=20)
plt.colorbar(scatter, label='Loan Paid Back')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
plt.title('PCA - First Two Principal Components', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# ============================================================================
# 15. LOAN PURPOSE ANALYSIS
# ============================================================================

if 'loan_purpose' in train.columns:
    print("\n" + "="*80)
    print("LOAN PURPOSE ANALYSIS")
    print("="*80)
    
    # Default rate by loan purpose
    purpose_analysis = train.groupby('loan_purpose').agg({
        'loan_paid_back': ['count', 'mean', 'sum']
    }).round(4)
    purpose_analysis.columns = ['Count', 'PaybackRate', 'TotalPaidBack']
    purpose_analysis = purpose_analysis.sort_values('PaybackRate', ascending=False)
    print(purpose_analysis)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    purpose_analysis['PaybackRate'].plot(kind='barh', ax=axes[0], color='teal')
    axes[0].set_xlabel('Payback Rate', fontsize=12)
    axes[0].set_title('Loan Payback Rate by Purpose', fontsize=14, fontweight='bold')
    
    purpose_analysis['Count'].plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_xlabel('Count', fontsize=12)
    axes[1].set_title('Loan Count by Purpose', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()



# ============================================================================
# 16. CREDIT SCORE ANALYSIS
# ============================================================================

if 'credit_score' in train.columns:
    print("\n" + "="*80)
    print("CREDIT SCORE ANALYSIS")
    print("="*80)
    
    # Create credit score bins
    train['credit_score_bin'] = pd.cut(train['credit_score'], 
                                        bins=[0, 580, 670, 740, 800, 850],
                                        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    credit_analysis = train.groupby('credit_score_bin').agg({
        'loan_paid_back': ['count', 'mean']
    }).round(4)
    credit_analysis.columns = ['Count', 'PaybackRate']
    print(credit_analysis)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.violinplot(data=train, x='credit_score_bin', y='credit_score', ax=axes[0], palette='Set2')
    axes[0].set_title('Credit Score Distribution by Bin', fontsize=14, fontweight='bold')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    
    credit_analysis['PaybackRate'].plot(kind='bar', ax=axes[1], color='green', alpha=0.7)
    axes[1].set_title('Loan Payback Rate by Credit Score Category', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Payback Rate')
    axes[1].set_xlabel('Credit Score Category')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 17. AGE GROUP ANALYSIS
# ============================================================================

if 'age' in train.columns:
    print("\n" + "="*80)
    print("AGE GROUP ANALYSIS")
    print("="*80)
    
    # Create age bins
    train['age_group'] = pd.cut(train['age'], 
                                 bins=[0, 25, 35, 45, 55, 65, 100],
                                 labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    age_analysis = train.groupby('age_group').agg({
        'loan_paid_back': ['count', 'mean'],
        'loan_amount': 'mean',
        'credit_score': 'mean'
    }).round(2)
    age_analysis.columns = ['Count', 'PaybackRate', 'AvgLoanAmount', 'AvgCreditScore']
    print(age_analysis)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    age_analysis['PaybackRate'].plot(kind='bar', ax=axes[0, 0], color='purple', alpha=0.7)
    axes[0, 0].set_title('Payback Rate by Age Group', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Payback Rate')
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
    
    age_analysis['Count'].plot(kind='bar', ax=axes[0, 1], color='orange', alpha=0.7)
    axes[0, 1].set_title('Loan Count by Age Group', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
    
    age_analysis['AvgLoanAmount'].plot(kind='bar', ax=axes[1, 0], color='blue', alpha=0.7)
    axes[1, 0].set_title('Average Loan Amount by Age Group', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Avg Loan Amount')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    age_analysis['AvgCreditScore'].plot(kind='bar', ax=axes[1, 1], color='green', alpha=0.7)
    axes[1, 1].set_title('Average Credit Score by Age Group', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Avg Credit Score')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()



# ============================================================================
# 18. SUMMARY STATISTICS BY TARGET
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS BY TARGET")
print("="*80)

for col in numerical_cols[:10]:  # Top 10 numerical features
    print(f"\n{col}:")
    print(train.groupby('loan_paid_back')[col].describe().round(2))

