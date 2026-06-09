# =========================================================
# SECTION 1: DATA LOADING & INITIAL EXPLORATION
# =========================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from lightgbm import LGBMRegressor

# Visualization setup
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1.1 LOAD DATA
# =========================================================
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print("ğŸ“¦ Dataset successfully loaded!\n")
print(f"Train shape: {train.shape}")
print(f"Test shape : {test.shape}")

# =========================================================
# 1.2 TARGET & BASIC STATISTICS
# =========================================================
target_col = 'loan_paid_back'

print("\nğŸ�¯ Target variable distribution:")
print(train[target_col].value_counts(normalize=True).round(3))

print("\nğŸ“Š Target descriptive statistics:")
print(train[target_col].describe())

# =========================================================
# 1.3 MISSING VALUES & DATA TYPES
# =========================================================
print("\nğŸ”� Missing Values Overview:")
missing = train.isnull().sum()
if missing.sum() == 0:
    print("âœ… No missing values detected.")
else:
    display(missing[missing > 0])

print("\nğŸ§© Data Types Summary:")
print(train.dtypes.value_counts())

# =========================================================
# 1.4 FEATURE GROUPING
# =========================================================
# Detect ID columns (exact 'id' or ending in '_id')
id_cols = [col for col in train.columns if col.lower() == 'id' or col.lower().endswith('_id')]

# Detect categorical and numeric variables
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude target and IDs from numeric group
numeric_cols = [col for col in numeric_cols if col not in id_cols + [target_col]]

# Master feature dictionary
FEATURE_GROUPS = {
    'id': id_cols,
    'numeric': numeric_cols,
    'categorical': categorical_cols,
    'target': [target_col]
}

# Display summary
print("\nğŸ—‚ï¸� FEATURE GROUPS DETECTED")
for group, cols in FEATURE_GROUPS.items():
    print(f"â€¢ {group.upper():<10} ({len(cols):>3}) â†’ {cols if len(cols) < 10 else str(cols[:8]) + ' ...'}")

print(f"\nTotal usable features (excluding ID & target): {len(numeric_cols) + len(categorical_cols)}")


display(train.head(5))


display(test.head(5))


# =========================================================
# SECTION 2 - TARGET DISTRIBUTION & CLASS IMBALANCE
# =========================================================

target_col = FEATURE_GROUPS['target'][0]

# --- 2.1 Basic counts and proportions
target_counts = train[target_col].value_counts().sort_index()
target_pct = train[target_col].value_counts(normalize=True).sort_index() * 100

target_summary = pd.DataFrame({
    'Count': target_counts,
    'Percentage (%)': target_pct.round(2)
})
target_summary['Ratio_vs_majority'] = (target_summary['Count'] / target_summary['Count'].max()).round(3)

display(target_summary)

# --- 2.2 Plot class distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot
sns.barplot(x=target_counts.index, y=target_counts.values, palette='viridis', ax=ax[0])
ax[0].set_title('Target Class Distribution')
ax[0].set_xlabel('Loan Paid Back (1 = Yes, 0 = No)')
ax[0].set_ylabel('Count')
for i, val in enumerate(target_counts.values):
    ax[0].text(i, val + len(train) * 0.002, f"{val:,}", ha='center')

# Pie chart
ax[1].pie(target_counts.values, labels=[f'{k}: {v:.2f}%' for k, v in target_pct.items()],
          autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', 2))
ax[1].set_title('Class Balance (Pie Chart)')

plt.tight_layout()
plt.show()

# --- 2.3 Imbalance metrics
majority = target_counts.max()
minority = target_counts.min()
imbalance_ratio = round(majority / minority, 2)
minority_pct = round(minority / len(train) * 100, 2)

print("ğŸ“Š CLASS IMBALANCE SUMMARY")
print("=" * 35)
print(f"ğŸ”¹ Majority class proportion: {majority / len(train):.2%}")
print(f"ğŸ”¹ Minority class proportion: {minority / len(train):.2%}")
print(f"ğŸ”¹ Imbalance ratio (maj/min): {imbalance_ratio}")
print(f"ğŸ”¹ Minority represents {minority_pct}% of total")

# --- 2.4 Recommendation
if imbalance_ratio < 1.5:
    note = "âœ… The dataset is fairly balanced. Standard CV is fine."
elif imbalance_ratio < 3:
    note = "âš ï¸� Moderate imbalance detected. Prefer StratifiedKFold and ROC-AUC metric."
else:
    note = "ğŸš¨ Strong imbalance detected. Consider class weights or SMOTE balancing."
print("\nğŸ’¡ Recommendation:", note)



# =========================================================
# Section 3: Categorical Variable Analysis
# =========================================================

categorical_features = FEATURE_GROUPS['categorical'].copy()

# ---------------------------------------------------------
# 3.1 Simplify 'grade_subgrade' -> 'grade_simple'
# ---------------------------------------------------------
if 'grade_subgrade' in categorical_features:
    train['grade_simple'] = train['grade_subgrade'].astype(str).str[0]
    test['grade_simple'] = test['grade_subgrade'].astype(str).str[0]
    categorical_features.remove('grade_subgrade')
    categorical_features.append('grade_simple')
    FEATURE_GROUPS['categorical'] = categorical_features
    print("âœ… Created simplified feature 'grade_simple' from 'grade_subgrade'.")

print(f"ğŸ”� Analysing {len(categorical_features)} categorical variables: {categorical_features}")


# ---------------------------------------------------------
# 3.2 Summary table per categorical variable
# ---------------------------------------------------------
print("\nğŸ“‹ Summary Statistics by Category:")
global_mean = train[target_col].mean()
for col in categorical_features:
    summary = train.groupby(col)[target_col].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)
    summary['diff_vs_global'] = (summary['mean'] - global_mean).round(3)
    print(f"\nğŸ”¹ {col.upper()}")
    display(summary.head())

# ---------------------------------------------------------
# 3.3 ANOVA (Significance) + Tukey HSD (Post-hoc)
# ---------------------------------------------------------
print("\nğŸ”¬ Statistical Significance Tests (ANOVA):")
for col in categorical_features:
    groups = [train.loc[train[col] == val, target_col].values for val in train[col].unique()]
    if len(groups) > 1:
        f_stat, p_value = f_oneway(*groups)
        print(f"   {col:25s} - F-stat: {f_stat:8.2f}, p-value: {p_value:.2e}")

print("\nğŸ“Š Post-Hoc Analysis: Tukey HSD Results")
for col in categorical_features:
    if train[col].nunique() > 2:
        print(f"\nğŸ”¹ {col.upper()} ({train[col].nunique()} categories):")
        tukey = pairwise_tukeyhsd(endog=train[target_col], groups=train[col], alpha=0.05)
        print(tukey.summary())

# ---------------------------------------------------------
# 3.4 Association Strength - Cramer's V
# ---------------------------------------------------------
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

print("\nğŸ“ˆ Cramer's V (Association Strength):")
cramer_results = {}
for col in categorical_features:
    val = cramers_v(train[col], train[target_col])
    cramer_results[col] = val
    print(f"   {col:25s}: {val:.3f}")

# ---------------------------------------------------------
# 3.5 Predictive Strength - Information Value (IV)
# ---------------------------------------------------------
def calculate_woe_iv(df, feature, target):
    lst = []
    for val in df[feature].dropna().unique():
        sub = df[df[feature] == val]
        good = len(sub[sub[target] == 1])
        bad = len(sub[sub[target] == 0])
        lst.append([val, good, bad])
    dset = pd.DataFrame(lst, columns=['Value', 'Good', 'Bad'])
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log((dset['Distr_Good'] + 1e-6) / (dset['Distr_Bad'] + 1e-6))
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    return dset['IV'].sum()

print("\nğŸ“Š Information Value (Predictive Strength):")
iv_results = {}
for col in categorical_features:
    iv = calculate_woe_iv(train, col, target_col)
    iv_results[col] = iv
    print(f"   {col:25s}: {iv:.3f}")

# ---------------------------------------------------------
# 3.6 Summary ranking of categorical variables
# ---------------------------------------------------------
df_summary = pd.DataFrame({
    'Feature': categorical_features,
    'CramersV': [cramer_results[c] for c in categorical_features],
    'IV': [iv_results[c] for c in categorical_features]
}).sort_values('IV', ascending=False)

print("\nğŸ�� Summary Ranking (sorted by IV):")
display(df_summary)



# =========================================================
# SECTION 4.1 STATISTICS + PCA EXPLAINABILITY
# =========================================================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

numerical_features = FEATURE_GROUPS['numeric']
target_col = FEATURE_GROUPS['target'][0]

# ---------------------------------------------------------
#  Descriptive Statistics Summary
# ---------------------------------------------------------
print("ğŸ“Š DESCRIPTIVE STATISTICS SUMMARY (with RMSE & RMSE/Mean Ratio)")

means = train[numerical_features].mean()
variances = train[numerical_features].var()
rmses = np.sqrt(variances)

desc_table = pd.DataFrame({
    'Mean': means,
    'Median': train[numerical_features].median(),
    'Min': train[numerical_features].min(),
    'Max': train[numerical_features].max(),
    'RMSE': rmses,
    'RMSE/Mean': (rmses / means).round(3),
    'Skewness': train[numerical_features].skew(),
    'Kurtosis': train[numerical_features].kurtosis()
}).round(3)

display(desc_table)

# ---------------------------------------------------------
# Visual Exploration (Histograms + Boxplots)
# ---------------------------------------------------------
palette = sns.blend_palette(["#5DADE2", "#58D68D"], as_cmap=False, n_colors=6)
n_cols = 2
n_rows = len(numerical_features)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))

for i, col in enumerate(numerical_features):
    # Histogram with mean & median
    sns.histplot(train[col], bins=40, color=palette[2], edgecolor='black', alpha=0.7, ax=axes[i,0])
    axes[i,0].axvline(train[col].mean(), color='red', linestyle='--', label='Mean')
    axes[i,0].axvline(train[col].median(), color='orange', linestyle='-', label='Median')
    axes[i,0].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[i,0].legend()
    axes[i,0].grid(alpha=0.3)
    
    # Boxplot
    sns.boxplot(x=train[col], ax=axes[i,1], color=palette[3])
    axes[i,1].set_title(f'{col} Boxplot (Outliers & Quartiles)', fontsize=12, fontweight='bold')
    axes[i,1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# PCA â€“ Explained Variance and Loadings (Explainability)
# ---------------------------------------------------------
print("\nğŸ”� PCA ANALYSIS ON NUMERICAL FEATURES")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train[numerical_features])

pca = PCA()
pca.fit(X_scaled)

# Variance explained table
explained_variance = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Explained Variance (%)': np.round(pca.explained_variance_ratio_ * 100, 2),
    'Cumulative Variance (%)': np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2)
})
display(explained_variance)

# Component loadings (weights per variable)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=numerical_features
).round(3)

print("\nğŸ“ˆ PCA COMPONENT LOADINGS (by variable):")
display(loadings)

# Scree plot
plt.figure(figsize=(8,5))
sns.lineplot(x=range(1, len(pca.explained_variance_ratio_)+1),
             y=np.cumsum(pca.explained_variance_ratio_)*100,
             marker='o', color='#58D68D')
plt.title('Cumulative Explained Variance by Principal Components', fontsize=12, fontweight='bold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained (%)')
plt.grid(alpha=0.4)
plt.show()



# =========================================================
# SECTION 4.2 - CROSS-FEATURE INTERACTIONS
# =========================================================

target_col = FEATURE_GROUPS['target'][0]

print(f"\nğŸ”— Key Feature Interactions:")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

# 1ï¸�âƒ£ Credit Score Ã— Interest Rate
interaction = train.copy()
interaction['credit_bin'] = pd.qcut(interaction['credit_score'], q=5, labels=[f'Q{i+1}' for i in range(5)])
data = interaction.groupby(['credit_bin', 'grade_simple'])[target_col].mean().unstack()
data.plot(kind='bar', ax=axes[0], width=0.8, cmap='PuBuGn')
axes[0].set_title('Credit Score Ã— Grade', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Avg Loan Payback Rate')
axes[0].legend(title='Grade')
axes[0].tick_params(axis='x', rotation=0)

# 2ï¸�âƒ£ Employment Ã— Education
data = train.groupby(['employment_status', 'education_level'])[target_col].mean().unstack()
data.plot(kind='bar', ax=axes[1], cmap='PuBuGn')
axes[1].set_title('Employment Ã— Education', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Avg Loan Payback Rate')
axes[1].legend(title='Education Level')
axes[1].tick_params(axis='x', rotation=30)

# 3ï¸�âƒ£ Loan Purpose Ã— Grade
data = train.groupby(['loan_purpose', 'grade_simple'])[target_col].mean().unstack()
data.plot(kind='bar', ax=axes[2], cmap='PuBuGn')
axes[2].set_title('Loan Purpose Ã— Grade', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Avg Loan Payback Rate')
axes[2].legend(title='Grade')
axes[2].tick_params(axis='x', rotation=45)

# 4ï¸�âƒ£ Income Ã— Debt Ratio (binned)
train['income_bin'] = pd.qcut(train['annual_income'], q=4, labels=['Low','Mid-Low','Mid-High','High'])
train['dti_bin'] = pd.qcut(train['debt_to_income_ratio'], q=4, labels=['Low','Mid-Low','Mid-High','High'])
data = train.groupby(['income_bin', 'dti_bin'])[target_col].mean().unstack()
sns.heatmap(data, annot=True, fmt=".2f", cmap='PuBuGn', ax=axes[3])
axes[3].set_title('Income Ã— Debt-to-Income Ratio', fontsize=12, fontweight='bold')
axes[3].set_ylabel('Income Bin')
axes[3].set_xlabel('Debt-to-Income Bin')

# 5ï¸�âƒ£ Loan Amount Ã— Interest Rate
train['loan_bin'] = pd.qcut(train['loan_amount'], q=4, labels=['Low','Mid-Low','Mid-High','High'])
train['rate_bin'] = pd.qcut(train['interest_rate'], q=4, labels=['Low','Mid-Low','Mid-High','High'])
data = train.groupby(['loan_bin', 'rate_bin'])[target_col].mean().unstack()
sns.heatmap(data, annot=True, fmt=".2f", cmap='YlGnBu', ax=axes[4])
axes[4].set_title('Loan Amount Ã— Interest Rate', fontsize=12, fontweight='bold')
axes[4].set_ylabel('Loan Size')
axes[4].set_xlabel('Interest Rate')

# 6ï¸�âƒ£ Employment Ã— Loan Purpose
data = train.groupby(['employment_status', 'loan_purpose'])[target_col].mean().unstack()
data.plot(kind='bar', ax=axes[5], cmap='YlGnBu')
axes[5].set_title('Employment Ã— Loan Purpose', fontsize=12, fontweight='bold')
axes[5].set_ylabel('Avg Loan Payback Rate')
axes[5].legend(title='Purpose')
axes[5].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('feature_interactions.png', dpi=300, bbox_inches='tight')
plt.show()



# =========================================================
# SECTION 4.3 - NUMERIC-NUMERIC RELATIONSHIPS
# =========================================================


numerical_features = FEATURE_GROUPS['numeric']

print("\nğŸ”— Pairwise Relationships Between Numerical Variables")

# ---------------------------------------------------------
# 4.3.1 - Scatter Matrix (overview)
# ---------------------------------------------------------
sns.pairplot(
    data=train,
    vars=numerical_features,
    corner=True,
    diag_kind='kde',
    plot_kws={'alpha':0.3, 's':10, 'color':'#1f77b4'}
)
plt.suptitle("Pairwise Numeric Relationships", y=1.02, fontsize=13, fontweight='bold')
plt.show()

# ---------------------------------------------------------
# 4.3.2 - Highlight Key Relationships
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Credit Score vs Interest Rate
sns.regplot(data=train, x='credit_score', y='interest_rate',
            scatter_kws={'alpha':0.3, 's':8}, line_kws={'color':'red'},
            ax=axes[0,0])
axes[0,0].set_title('Credit Score vs Interest Rate', fontsize=12, fontweight='bold')

# Credit Score vs Loan Amount
sns.scatterplot(data=train, x='credit_score', y='loan_amount',
                alpha=0.3, s=10, ax=axes[0,1])
axes[0,1].set_title('Credit Score vs Loan Amount', fontsize=12, fontweight='bold')

# Annual Income vs Loan Amount
sns.scatterplot(data=train, x='annual_income', y='loan_amount',
                alpha=0.3, s=10, ax=axes[0,2])
axes[0,2].set_title('Annual Income vs Loan Amount', fontsize=12, fontweight='bold')

# Annual Income vs Debt-to-Income Ratio
sns.scatterplot(data=train, x='annual_income', y='debt_to_income_ratio',
                alpha=0.3, s=10, ax=axes[1,0])
axes[1,0].set_title('Annual Income vs Debt-to-Income Ratio', fontsize=12, fontweight='bold')

# Loan Amount vs Interest Rate
sns.scatterplot(data=train, x='loan_amount', y='interest_rate',
                alpha=0.3, s=10, ax=axes[1,1])
axes[1,1].set_title('Loan Amount vs Interest Rate', fontsize=12, fontweight='bold')

# Debt-to-Income vs Interest Rate
sns.scatterplot(data=train, x='debt_to_income_ratio', y='interest_rate',
                alpha=0.3, s=10, ax=axes[1,2])
axes[1,2].set_title('Debt-to-Income vs Interest Rate', fontsize=12, fontweight='bold')

for ax in axes.ravel():
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pairwise_numeric_relationships.png', dpi=300, bbox_inches='tight')
plt.show()



# =========================================================
# SECTION 4.4 - CATEGORICAL Ã— NUMERICAL INTERACTIONS
# =========================================================
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import pandas as pd

target_col = FEATURE_GROUPS['target'][0]
categorical_features = FEATURE_GROUPS['categorical']
numerical_features = FEATURE_GROUPS['numeric']

print("\nğŸ”� Exploring interactions between categorical and numerical features...")

# ---------------------------------------------------------
# 1ï¸�âƒ£ BOXPLOTS â€” numeric distributions by category
# ---------------------------------------------------------
pairs = [
    ('employment_status', 'annual_income'),
    ('loan_purpose', 'loan_amount'),
    ('grade_simple', 'credit_score'),
    ('education_level', 'interest_rate')
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.ravel()

for (cat, num), ax in zip(pairs, axes):
    sns.boxplot(data=train, x=cat, y=num, palette="crest", ax=ax)
    ax.set_title(f'{num} by {cat}', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2ï¸�âƒ£ HEATMAP â€” mean numeric value by two categorical vars
# ---------------------------------------------------------
pivot = train.pivot_table(
    values='loan_amount',
    index='loan_purpose',
    columns='grade_simple',
    aggfunc='mean'
)
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title("Average Loan Amount by Loan Purpose and Grade", fontsize=12, fontweight='bold')
plt.ylabel("Loan Purpose")
plt.xlabel("Grade")
plt.show()

# ---------------------------------------------------------
# 3ï¸�âƒ£ ANOVA â€” statistical test for mean differences
# ---------------------------------------------------------
print("\nğŸ“Š ANOVA Significance Tests (feature by numeric variable):")
for cat, num in pairs:
    groups = [train.loc[train[cat] == c, num] for c in train[cat].unique()]
    f, p = f_oneway(*groups)
    print(f"   {num:20s} by {cat:20s} â†’ F={f:8.2f}, p={p:.2e}")



# =========================================================
# SECTION 5.1 - Correlation Analysis 
# =========================================================

target_col = FEATURE_GROUPS['target'][0]
numerical_features = FEATURE_GROUPS['numeric']

# ---------------------------------------------------------
# 5.1 - Compute Correlations
# ---------------------------------------------------------
print("ğŸ“ˆ Calculating Pearson, Spearman and Kendall correlations...\n")

corr_pearson = train[numerical_features + [target_col]].corr(method='pearson')
corr_spearman = train[numerical_features + [target_col]].corr(method='spearman')
corr_kendall = train[numerical_features + [target_col]].corr(method='kendall')

# Extract target correlations
target_corrs = pd.DataFrame({
    'Feature': numerical_features,
    'Pearson': corr_pearson[target_col].drop(target_col),
    'Spearman': corr_spearman[target_col].drop(target_col),
    'Kendall': corr_kendall[target_col].drop(target_col)
}).sort_values(by='Spearman', ascending=False).round(3)

display(target_corrs)

# ---------------------------------------------------------
# 5.2 - Heatmaps
# ---------------------------------------------------------
plt.figure(figsize=(7,5))
sns.heatmap(corr_pearson, cmap='PuBuGn', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Pearson Correlation Heatmap', fontsize=12, fontweight='bold')
plt.show()

plt.figure(figsize=(7,5))
sns.heatmap(corr_spearman, cmap='PuBuGn', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Spearman Correlation Heatmap', fontsize=12, fontweight='bold')
plt.show()

# ---------------------------------------------------------
# 5.3 - Correlation Comparison Barplot
# ---------------------------------------------------------
plt.figure(figsize=(8,5))
target_corrs_melt = target_corrs.melt(id_vars='Feature', var_name='Method', value_name='Correlation')
sns.barplot(data=target_corrs_melt, x='Feature', y='Correlation', hue='Method', palette='PuBuGn')
plt.title('Feature Correlation with Loan Payback (by Method)', fontsize=12, fontweight='bold')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.show()


# ---------------------------------------------------------
# 5.2 - Optional: Pairwise Relationships (Scatter + Density)
# ---------------------------------------------------------

train_sample = train.sample(n=10000, random_state=42)
sns.pairplot(train_sample, vars=numerical_features, 
             hue=target_col, diag_kind='kde',
             plot_kws={'alpha':0.5, 's':10}, 
             palette='Greens')
plt.suptitle("Pairwise Feature Relationships (10k sample)", y=1.02)
plt.show()


# ===============================================================
# ğŸ§  SECTION 6 â€” BASELINE MODELING (LightGBM)
# ===============================================================

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    roc_curve, ConfusionMatrixDisplay
)

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

# ---------------------------------------------------------------
# 6.1 TRAIN / VALIDATION SPLIT
# ---------------------------------------------------------------
TARGET = "loan_paid_back"

X = train.drop(columns=[TARGET, "id"])
y = train[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"âœ… Train set: {X_train.shape}, Validation set: {X_val.shape}")
print(f"ğŸ�¯ Target mean â€” Train: {y_train.mean():.5f}, Val: {y_val.mean():.5f}")

# ---------------------------------------------------------------
# 6.2 ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"ğŸ“¦ Categorical features detected: {cat_cols}")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)

    X_train[col] = le.transform(X_train[col].astype(str))
    X_val[col] = le.transform(X_val[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

    le_dict[col] = le

print("âœ… All categorical columns encoded successfully.")

# ---------------------------------------------------------------
# 6.3 BASELINE LIGHTGBM MODEL
# ---------------------------------------------------------------

lgb_params = {
    "objective": "binary",                # Cross-entropy loss
    "metric": ["auc", "binary_logloss"],  # AUC + log-loss
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.88,
    "colsample_bytree": 0.55,
    "lambda_l1": 0.24,
    "lambda_l2": 0.28,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

print("\nğŸš€ Training LightGBM baseline model...")
model = lgb.LGBMClassifier(**lgb_params)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_names=["train", "valid"],
    eval_metric="auc",
    callbacks=[log_evaluation(200), early_stopping(100)]
)

# ---------------------------------------------------------------
# 6.4 VALIDATION EVALUATION
# ---------------------------------------------------------------
y_pred_proba = model.predict_proba(X_val)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_val, y_pred_proba)
f1 = f1_score(y_val, y_pred)

print(f"\nğŸ“ˆ Validation AUC: {auc:.4f}")
print(f"ğŸ�¯ Validation F1:  {f1:.4f}")
print("\nğŸ“Š Classification Report:")
print(classification_report(y_val, y_pred))

# ---------------------------------------------------------------
# 6.5 VISUAL EVALUATION
# ---------------------------------------------------------------

# Confusion Matrix
plt.figure(figsize=(3, 3))
ConfusionMatrixDisplay.from_estimator(model, X_val, y_val, cmap="Blues")
plt.title("Confusion Matrix â€” LGBM Baseline")
plt.show()

# ---------------------------------------------------------------
# 6.6 FEATURE IMPORTANCE
# ---------------------------------------------------------------
importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=importances.head(20), x="Importance", y="Feature", palette="crest")
plt.title("Top 20 Feature Importances â€” LGBM Baseline")
plt.tight_layout()
plt.show()

print("\nğŸ�� Baseline LightGBM training complete.")



# ===============================================================
# 6.7 TRAINING DYNAMICS â€” AUC & LOGLOSS BY ITERATION
# ===============================================================

evals_result = model.evals_result_

# Extract metrics
train_auc = evals_result["train"]["auc"]
valid_auc = evals_result["valid"]["auc"]
train_logloss = evals_result["train"]["binary_logloss"]
valid_logloss = evals_result["valid"]["binary_logloss"]

iterations = range(1, len(train_auc) + 1)

fig, ax1 = plt.subplots(figsize=(8,5))

# Plot AUC (left axis)
ax1.plot(iterations, train_auc, label="Train AUC", color="#1f77b4", lw=2)
ax1.plot(iterations, valid_auc, label="Valid AUC", color="#2ca02c", lw=2)
ax1.set_xlabel("Iterations")
ax1.set_ylabel("AUC", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.grid(alpha=0.3)

# Secondary axis for Logloss
ax2 = ax1.twinx()
ax2.plot(iterations, train_logloss, label="Train Logloss", color="#ff7f0e", lw=1.8, linestyle="--")
ax2.plot(iterations, valid_logloss, label="Valid Logloss", color="#d62728", lw=1.8, linestyle="--")
ax2.set_ylabel("Logloss", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

# Titles and legend
plt.title("Training Dynamics â€” AUC and Logloss per Iteration", fontsize=12, fontweight="bold")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="center right", frameon=False)
plt.tight_layout()
plt.show()



# ===============================================================
# ğŸ§® SECTION 6.8 â€” PREDICTION DISTRIBUTION (TRAIN vs TEST)
# ===============================================================
test_features = test.drop(columns=["id"], errors="ignore")

# ---------------------------------------------------------------
# 1ï¸�âƒ£ Generate validation & test predictions (probabilities)
# ---------------------------------------------------------------
print("ğŸš€ Generating predictions on validation and test sets...")

val_preds = model.predict_proba(X_val)[:, 1]
test_preds = model.predict_proba(test_features)[:, 1]

print(f"âœ… Validation predictions: {len(val_preds)} | Test predictions: {len(test_preds)}")
print(f"ğŸ“Š Mean probability â€” Val: {val_preds.mean():.4f} | Test: {test_preds.mean():.4f}")

# ---------------------------------------------------------------
# 2ï¸�âƒ£ Summary statistics
# ---------------------------------------------------------------
summary = pd.DataFrame({
    "Set": ["Validation", "Test"],
    "Mean": [val_preds.mean(), test_preds.mean()],
    "Std": [val_preds.std(), test_preds.std()],
    "Min": [val_preds.min(), test_preds.min()],
    "25%": [np.percentile(val_preds, 25), np.percentile(test_preds, 25)],
    "50%": [np.percentile(val_preds, 50), np.percentile(test_preds, 50)],
    "75%": [np.percentile(val_preds, 75), np.percentile(test_preds, 75)],
    "Max": [val_preds.max(), test_preds.max()]
})
print("\nğŸ“ˆ PREDICTION SUMMARY STATISTICS (Validation vs Test)")
display(summary.round(4))

# ---------------------------------------------------------------
# 3ï¸�âƒ£ Histogram + KDE â€” comparison
# ---------------------------------------------------------------
plt.figure(figsize=(12, 5))
sns.kdeplot(val_preds, label="Validation", fill=True, color="#3498db", alpha=0.45)
sns.kdeplot(test_preds, label="Test", fill=True, color="#1abc9c", alpha=0.45)
plt.title("Distribution of Predicted Probabilities (Validation vs Test)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Probability (loan_paid_back)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 4ï¸�âƒ£ Cumulative Distribution (ECDF)
# ---------------------------------------------------------------
plt.figure(figsize=(12, 5))
sns.ecdfplot(val_preds, label="Validation", color="#2980b9", lw=2)
sns.ecdfplot(test_preds, label="Test", color="#16a085", lw=2)
plt.title("Cumulative Distribution Comparison (ECDF)", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Probability")
plt.ylabel("Cumulative Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 5ï¸�âƒ£ Shift diagnostics
# ---------------------------------------------------------------
val_mean, test_mean = np.mean(val_preds), np.mean(test_preds)
mean_shift = (test_mean - val_mean) / val_mean * 100
print(f"ğŸ“‰ Mean shift (Test vs Val): {mean_shift:+.2f}%")

if abs(mean_shift) < 5:
    print("âœ… No significant drift detected â€” prediction calibration is stable across datasets.")
elif mean_shift > 5:
    print("âš ï¸� Model predicts slightly higher probabilities on Test â€” potential optimism bias.")
else:
    print("âš ï¸� Model predicts lower probabilities on Test â€” potential underestimation of repayment likelihood.")

# ---------------------------------------------------------------
# 6ï¸�âƒ£ Skewness & Kurtosis
# ---------------------------------------------------------------
from scipy.stats import skew, kurtosis

val_skew, val_kurt = skew(val_preds), kurtosis(val_preds)
test_skew, test_kurt = skew(test_preds), kurtosis(test_preds)

print(f"\nğŸ“Š DISTRIBUTION SHAPE METRICS")
print(f"Validation â€” Skew: {val_skew:.3f}, Kurtosis: {val_kurt:.3f}")
print(f"Test        â€” Skew: {test_skew:.3f}, Kurtosis: {test_kurt:.3f}")

if abs(test_skew - val_skew) > 0.5:
    print("âš ï¸� Noticeable asymmetry shift between Validation and Test distributions.")
else:
    print("âœ… Similar symmetry between Validation and Test â€” consistent model calibration.")

if abs(test_kurt - val_kurt) > 1:
    print("âš ï¸� Kurtosis difference > 1 â€” Test predictions more/less extreme than Validation.")
else:
    print("âœ… Comparable kurtosis â€” no excessive overconfidence or flattening detected.")



# ===============================================================
# âš¡ SECTION 7 â€” EXPLAINABILITY ANALYSIS (SHAP INTERPRETATION)
# ===============================================================

import shap
import multiprocessing

# ---------------------------------------------------------------
# 7.1 SAFETY CHECK
# ---------------------------------------------------------------
if not hasattr(model, "booster_"):
    raise ValueError("â�Œ The model is not fitted yet. Please train it before running SHAP analysis.")

# Detect available CPU cores
n_cores = multiprocessing.cpu_count()
print(f"ğŸ’» Detected {n_cores} CPU cores â€” using all for SHAP computation.")

# ---------------------------------------------------------------
# 7.2 CREATE SHAP EXPLAINER
# ---------------------------------------------------------------
explainer = shap.TreeExplainer(
    model,
    feature_perturbation="tree_path_dependent",
    model_output="raw"
)

# ---------------------------------------------------------------
# 7.3 SAMPLE VALIDATION DATA FOR ANALYSIS
# ---------------------------------------------------------------
N = 3000
if len(X_val) > N:
    print(f"ğŸ“‰ Sampling {N} rows from validation set for SHAP analysis (out of {len(X_val)}).")
    X_shap = X_val.sample(N, random_state=42)
else:
    X_shap = X_val.copy()

# ---------------------------------------------------------------
# 7.4 COMPUTE SHAP VALUES
# ---------------------------------------------------------------
print("âš™ï¸� Computing SHAP values (this may take a moment)...")
shap_values = explainer.shap_values(X_shap, check_additivity=False)
print("âœ… SHAP values computed successfully.")

# Handle LightGBM binary output (list of arrays)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# ---------------------------------------------------------------
# 7.5 FEATURE IMPORTANCE BY MEAN ABSOLUTE SHAP VALUE
# ---------------------------------------------------------------
shap_df = pd.DataFrame(shap_values, columns=X_shap.columns)
shap_importance = (
    np.abs(shap_df)
    .mean()
    .sort_values(ascending=False)
    .rename("mean_abs_shap")
    .reset_index()
    .rename(columns={"index": "feature"})
)

print("\nğŸ�† Top 15 Most Influential Features (by mean |SHAP|):")
display(shap_importance.head(15))

# ---------------------------------------------------------------
# 7.6 SHAP VISUALIZATIONS
# ---------------------------------------------------------------

# Summary plot (beeswarm)
plt.figure(figsize=(20, 8))
shap.summary_plot(shap_values, X_shap, max_display=30, show=False)
plt.title("SHAP Summary Plot â€” Top 30 Features", fontsize=14, weight="bold")
plt.show()

# Bar plot of mean |SHAP| values
plt.figure(figsize=(12, 5))
shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=30, show=False)
plt.title("Mean Absolute SHAP Values â€” Top 30 Features", fontsize=14, weight="bold")
plt.show()


# ===============================================================
# SECTION 8 â€” BASELINE SUBMISSION (LightGBM)
# ===============================================================

# ---------------------------------------------------------------
# 1ï¸�âƒ£ Sanity Checks
# ---------------------------------------------------------------
print("ğŸ“¦ Preparing baseline submission file...")

print(f"Test set shape: {test.shape}")
print(f"Predictions available: {len(test_preds) if 'test_preds' in locals() else 0}")

# Generate predictions if missing
if "test_preds" not in locals():
    print("âš™ï¸� Generating test predictions using trained baseline model...")
    test_preds = model.predict_proba(test.drop(columns=["id"]))[:, 1]
else:
    print("âœ… Using existing predictions from previous step.")

# ---------------------------------------------------------------
# 2ï¸�âƒ£ Build Submission DataFrame
# ---------------------------------------------------------------
submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": np.clip(test_preds, 0, 1)   # ensures numeric stability
})

# Preview
print("\nğŸ“‹ Submission preview:")
display(submission.head())

# ---------------------------------------------------------------
# 3ï¸�âƒ£ Save File
# ---------------------------------------------------------------
output_path = "submission.csv"
submission.to_csv(output_path, index=False)

print(f"\nâœ… Submission file saved successfully â†’ {output_path}")
print(f"ğŸ“� File ready for Kaggle upload â€” {submission.shape[0]} rows Ã— {submission.shape[1]} columns")


