# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')


print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nMissing values:\n{train.isnull().sum().sum()}")


train.info()


train.head()


train.columns.tolist()


train.dtypes.value_counts()


train_missing = train.isnull().sum()
test_missing = test.isnull().sum()


print(train_missing)


print(test_missing)


null_variants = [
    'null', 'NULL', 'Null',
    'na', 'NA', 'N/A', 'n/a',
    'none', 'None', 'NONE',
    'nan', 'NaN', 'NAN',
    'missing', 'Missing', 'MISSING',
    'unknown', 'Unknown', 'UNKNOWN',
    '', ' ', '  ',  # Empty strings and whitespace
    '-', '--', '---',
    '?', '??',
    'not available', 'Not Available',
    'not specified', 'Not Specified'
]


train.isna().sum()


train.isnull()


pd.set_option('display.max_row',None)
pd.set_option('display.max_column',None)


train.isnull().sum()


train['loan_paid_back'].value_counts(normalize=True)


train.describe()


cat_features = ['gender', 'marital_status', 'education_level', 
                'employment_status', 'loan_purpose', 'grade_subgrade']
for col in cat_features:
    print(f"\n{col}: {train[col].nunique()} unique values")
    print(train[col].value_counts().head(10))


numeric_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                    'loan_amount', 'interest_rate']
print("\nCorrelation with target:")
print(train[numeric_features + ['loan_paid_back']].corr()['loan_paid_back'].sort_values(ascending=False))


train['loan_paid_back'].value_counts().plot(kind='bar', title='Target Distribution')
plt.show()


numeric_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                    'loan_amount', 'interest_rate']
categorical_features = ['gender', 'marital_status', 'education_level', 
                       'employment_status', 'loan_purpose', 'grade_subgrade']
target = 'loan_paid_back'


plt.figure(figsize=(12, 10))
correlation_matrix = train[numeric_features + [target]].corr()
sns.heatmap(correlation_matrix, 
            mask=None,
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=2,
            cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix\n(Strong correlations highlighted)', 
          fontsize=16, fontweight='bold', pad=20)
plt.show()


sample_size = min(10000, len(train))
train_sample = train.sample(n=sample_size, random_state=42)

pairplot = sns.pairplot(
    train_sample[numeric_features + [target]], 
    hue=target,
    palette={0: '#e74c3c', 1: '#2ecc71'},
    diag_kind='kde',
    corner=True,
    plot_kws={'alpha': 0.6, 's': 20}
)
pairplot.fig.suptitle('Pairwise Feature Relationships (Colored by Target)', 
                      fontsize=16, fontweight='bold', y=1.01)
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    ax = axes[idx]
    
    # KDE plots for each target class
    train[train[target] == 0][feature].plot(kind='kde', ax=ax, 
                                             label='Not Paid Back (0)', 
                                             color='#e74c3c', linewidth=2)
    train[train[target] == 1][feature].plot(kind='kde', ax=ax, 
                                             label='Paid Back (1)', 
                                             color='#2ecc71', linewidth=2)
    
    ax.set_title(f'{feature} Distribution by Target', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

# Hide extra subplot
axes[-1].axis('off')

plt.suptitle('Feature Distributions: Paid Back vs Not Paid Back', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    ax = axes[idx]
    
    train.boxplot(column=feature, by=target, ax=ax, 
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.6),
                  medianprops=dict(color='red', linewidth=2))
    
    ax.set_title(f'{feature} by Target', fontsize=12, fontweight='bold')
    ax.set_xlabel('Loan Paid Back')
    ax.set_ylabel(feature)
    ax.get_figure().suptitle('')  # Remove automatic title

axes[-1].axis('off')

plt.suptitle('Box Plots: Feature Ranges by Target', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    ax = axes[idx]
    
    sns.violinplot(data=train, x=target, y=feature, ax=ax,
                   palette={0: '#e74c3c', 1: '#2ecc71'},
                   inner='quartile')
    
    ax.set_title(f'{feature} Distribution Shape by Target', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Loan Paid Back')
    ax.set_ylabel(feature)

axes[-1].axis('off')

plt.suptitle('Violin Plots: Distribution Shapes by Target', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()



print("\nCreating Scatter Plots for Key Feature Interactions...")

# Top correlated pairs
top_pairs = [
    ('debt_to_income_ratio', 'credit_score'),
    ('credit_score', 'interest_rate'),
    ('debt_to_income_ratio', 'interest_rate'),
    ('annual_income', 'loan_amount')
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (feat1, feat2) in enumerate(top_pairs):
    ax = axes[idx]
    
    # Sample for faster plotting
    sample = train.sample(n=5000, random_state=42)
    
    scatter = ax.scatter(sample[feat1], sample[feat2], 
                        c=sample[target], 
                        cmap='RdYlGn',
                        alpha=0.6, 
                        s=30,
                        edgecolors='black',
                        linewidth=0.5)
    
    ax.set_xlabel(feat1, fontsize=11, fontweight='bold')
    ax.set_ylabel(feat2, fontsize=11, fontweight='bold')
    ax.set_title(f'{feat1} vs {feat2}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Loan Paid Back', rotation=270, labelpad=20)

plt.suptitle('Feature Interactions: Scatter Plots (Colored by Target)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
    ax = axes[idx]
    
    # Calculate target rate for each category
    target_rate = train.groupby(feature)[target].agg(['mean', 'count']).reset_index()
    target_rate = target_rate.sort_values('mean', ascending=False)
    
    # Create bar plot
    bars = ax.barh(target_rate[feature].astype(str), 
                   target_rate['mean'],
                   color='steelblue',
                   edgecolor='black')
    
    # Color bars by payback rate
    for i, bar in enumerate(bars):
        rate = target_rate.iloc[i]['mean']
        if rate > 0.85:
            bar.set_color('#2ecc71')  # Green for high payback
        elif rate < 0.70:
            bar.set_color('#e74c3c')  # Red for low payback
        else:
            bar.set_color('#3498db')  # Blue for medium
    
    ax.set_xlabel('Loan Payback Rate', fontsize=10, fontweight='bold')
    ax.set_title(f'{feature} - Payback Rate by Category', 
                 fontsize=11, fontweight='bold')
    ax.axvline(train[target].mean(), color='red', linestyle='--', 
               linewidth=2, label='Overall Average')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (val, count) in enumerate(zip(target_rate['mean'], target_rate['count'])):
        ax.text(val + 0.01, i, f'{val:.2%} (n={count:,})', 
                va='center', fontsize=9)

plt.suptitle('Categorical Features: Payback Rate Analysis\n(Green=High, Blue=Medium, Red=Low)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()




fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
    ax = axes[idx]
    
    # Create crosstab
    ct = pd.crosstab(train[feature], train[target], normalize='index') * 100
    
    ct.plot(kind='barh', stacked=True, ax=ax,
            color=['#e74c3c', '#2ecc71'],
            edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax.set_title(f'{feature} - Target Distribution', 
                 fontsize=11, fontweight='bold')
    ax.legend(['Not Paid Back', 'Paid Back'], loc='best')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Categorical Features: Stacked Proportions by Target', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()






# Top 2 most important features
top_features = [
    ('debt_to_income_ratio', 'credit_score'),
    ('interest_rate', 'credit_score')
]

for feat1, feat2 in top_features:
    # Sample for faster plotting
    sample = train.sample(n=5000, random_state=42)
    
    g = sns.jointplot(data=sample, x=feat1, y=feat2, 
                     hue=target, 
                     palette={0: '#e74c3c', 1: '#2ecc71'},
                     alpha=0.5,
                     height=10)
    
    g.fig.suptitle(f'Joint Distribution: {feat1} vs {feat2}', 
                   fontsize=14, fontweight='bold', y=1.01)
    plt.show()



# ============================================================================
# PREPROCESSING, FEATURE ENGINEERING & SELECTION
# ============================================================================

from sklearn.feature_selection import mutual_info_classif

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

#  CRITICAL: Save target FIRST before any processing
target = train['loan_paid_back'].copy()
train = train.drop(columns=['loan_paid_back'])  # Remove to avoid leakage

# Save IDs
train_ids = train['id'].copy()
test_ids = test['id'].copy()

print(f"âœ“ Train shape: {train.shape}")
print(f"âœ“ Test shape: {test.shape}")
print(f"âœ“ Target shape: {target.shape}")
print(f"âœ“ Target distribution:\n{target.value_counts()}")

# ============================================================================
# STEP 1: PREPROCESSING - HANDLE CATEGORICALS
# ============================================================================

print("\n" + "="*70)
print("STEP 1: CATEGORICAL ENCODING")
print("="*70)

def encode_categoricals(train_df, test_df):
    """
    Encode categorical features with appropriate strategies
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # -------------------------------------------------------------------
    # 1.1 ORDINAL ENCODING for grade_subgrade (preserves natural order)
    # -------------------------------------------------------------------
    print("\n1. Encoding grade_subgrade (ordinal)...")
    
    # Define grade order: A1 (best) to G5 (worst)
    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    subgrades = [f"{g}{i}" for g in grades for i in range(1, 6)]
    grade_map = {grade: idx for idx, grade in enumerate(subgrades, start=1)}
    
    train_encoded['grade_ordinal'] = train_encoded['grade_subgrade'].map(grade_map)
    test_encoded['grade_ordinal'] = test_encoded['grade_subgrade'].map(grade_map)
    
    print(f"   âœ“ Created grade_ordinal (1-35 scale)")
    print(f"   Sample mapping: A1={grade_map.get('A1', 'N/A')}, C3={grade_map.get('C3', 'N/A')}, G5={grade_map.get('G5', 'N/A')}")
    
    # -------------------------------------------------------------------
    # 1.2 ONE-HOT ENCODING for employment_status (moderate importance)
    # -------------------------------------------------------------------
    print("\n2. Encoding employment_status (one-hot)...")
    
    # One-hot encode
    emp_train = pd.get_dummies(train_encoded['employment_status'], prefix='emp', drop_first=True)
    emp_test = pd.get_dummies(test_encoded['employment_status'], prefix='emp', drop_first=True)
    
    # Ensure same columns
    for col in emp_train.columns:
        if col not in emp_test.columns:
            emp_test[col] = 0
    
    train_encoded = pd.concat([train_encoded, emp_train], axis=1)
    test_encoded = pd.concat([test_encoded, emp_test], axis=1)
    
    print(f"   âœ“ Created {len(emp_train.columns)} employment dummy variables")
    
    # -------------------------------------------------------------------
    # 1.3 LABEL ENCODING for low-importance categoricals
    # -------------------------------------------------------------------
    print("\n3. Encoding weak categorical features (label encoding)...")
    
    weak_categoricals = ['gender', 'marital_status', 'education_level', 'loan_purpose']
    
    encoders = {}
    for col in weak_categoricals:
        le = LabelEncoder()
        train_encoded[f'{col}_encoded'] = le.fit_transform(train_encoded[col].astype(str))
        test_encoded[f'{col}_encoded'] = le.transform(test_encoded[col].astype(str))
        encoders[col] = le
        print(f"   âœ“ Encoded {col} ({train_encoded[col].nunique()} categories)")
    
    # Drop original categorical columns
    cols_to_drop = ['grade_subgrade', 'employment_status'] + weak_categoricals
    train_encoded = train_encoded.drop(columns=cols_to_drop)
    test_encoded = test_encoded.drop(columns=cols_to_drop)
    
    return train_encoded, test_encoded

train_processed, test_processed = encode_categoricals(train, test)

print(f"\nâœ“ Encoding complete!")
print(f"   Features after encoding: {train_processed.shape[1]}")

# ============================================================================
# STEP 2: LOG TRANSFORMS FOR SKEWED FEATURES
# ============================================================================

print("\n" + "="*70)
print("STEP 2: LOG TRANSFORMS FOR SKEWED FEATURES")
print("="*70)

def apply_log_transforms(train_df, test_df):
    """
    Apply log transforms to right-skewed features
    """
    train_log = train_df.copy()
    test_log = test_df.copy()
    
    # Features identified as right-skewed from EDA
    skewed_features = ['debt_to_income_ratio', 'annual_income', 'loan_amount']
    
    for feature in skewed_features:
        # Log1p (handles zeros)
        train_log[f'{feature}_log'] = np.log1p(train_log[feature])
        test_log[f'{feature}_log'] = np.log1p(test_log[feature])
        
        # Calculate skewness reduction
        original_skew = train_df[feature].skew()
        new_skew = train_log[f'{feature}_log'].skew()
        
        print(f"   {feature}:")
        print(f"      Original skewness: {original_skew:.3f}")
        print(f"      After log transform: {new_skew:.3f}")
        print(f"      âœ“ Reduction: {abs(original_skew - new_skew):.3f}")
    
    return train_log, test_log

train_processed, test_processed = apply_log_transforms(train_processed, test_processed)

print(f"\nâœ“ Log transforms complete!")
print(f"   Features after transforms: {train_processed.shape[1]}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING (Based on EDA Insights)
# ============================================================================

print("\n" + "="*70)
print("STEP 3: FEATURE ENGINEERING")
print("="*70)

def engineer_features(train_df, test_df):
    """
    Create interaction and engineered features based on EDA insights
    """
    train_fe = train_df.copy()
    test_fe = test_df.copy()
    
    print("\n3.1 CRITICAL INTERACTIONS (from scatter plot analysis)...")
    
    # -------------------------------------------------------------------
    # INTERACTION 1: debt_to_income Ã— credit_score (strongest interaction)
    # -------------------------------------------------------------------
    train_fe['debt_credit_ratio'] = train_fe['debt_to_income_ratio'] / (train_fe['credit_score'] + 1)
    test_fe['debt_credit_ratio'] = test_fe['debt_to_income_ratio'] / (test_fe['credit_score'] + 1)
    
    train_fe['debt_credit_product'] = train_fe['debt_to_income_ratio'] * (1 / (train_fe['credit_score'] + 1))
    test_fe['debt_credit_product'] = test_fe['debt_to_income_ratio'] * (1 / (test_fe['credit_score'] + 1))
    
    print("   âœ“ Created debt_credit_ratio and debt_credit_product")
    
    # -------------------------------------------------------------------
    # INTERACTION 2: credit_score Ã— interest_rate (strong correlation -0.538)
    # -------------------------------------------------------------------
    train_fe['credit_interest_ratio'] = train_fe['credit_score'] / (train_fe['interest_rate'] + 1)
    test_fe['credit_interest_ratio'] = test_fe['credit_score'] / (test_fe['interest_rate'] + 1)
    
    train_fe['credit_interest_product'] = train_fe['credit_score'] * train_fe['interest_rate']
    test_fe['credit_interest_product'] = test_fe['credit_score'] * test_fe['interest_rate']
    
    print("   âœ“ Created credit_interest_ratio and credit_interest_product")
    
    # -------------------------------------------------------------------
    # INTERACTION 3: debt_to_income Ã— interest_rate (financial burden)
    # -------------------------------------------------------------------
    train_fe['financial_burden'] = train_fe['debt_to_income_ratio'] * train_fe['interest_rate']
    test_fe['financial_burden'] = test_fe['debt_to_income_ratio'] * test_fe['interest_rate']
    
    print("   âœ“ Created financial_burden (debt Ã— interest)")
    
    # -------------------------------------------------------------------
    # INTERACTION 4: Three-way mega interaction (ultimate risk score)
    # -------------------------------------------------------------------
    train_fe['ultimate_risk_score'] = (
        train_fe['debt_to_income_ratio'] * train_fe['interest_rate'] / (train_fe['credit_score'] + 1)
    )
    test_fe['ultimate_risk_score'] = (
        test_fe['debt_to_income_ratio'] * test_fe['interest_rate'] / (test_fe['credit_score'] + 1)
    )
    
    train_fe['creditworthiness_index'] = (
        train_fe['credit_score'] * (1 - train_fe['debt_to_income_ratio']) / (train_fe['interest_rate'] + 1)
    )
    test_fe['creditworthiness_index'] = (
        test_fe['credit_score'] * (1 - test_fe['debt_to_income_ratio']) / (test_fe['interest_rate'] + 1)
    )
    
    print("   âœ“ Created ultimate_risk_score and creditworthiness_index")
    
    print("\n3.2 POLYNOMIAL FEATURES (for top predictors)...")
    
    # -------------------------------------------------------------------
    # POLYNOMIAL: debt_to_income_ratio (strongest predictor -0.336)
    # -------------------------------------------------------------------
    train_fe['debt_to_income_squared'] = train_fe['debt_to_income_ratio'] ** 2
    test_fe['debt_to_income_squared'] = test_fe['debt_to_income_ratio'] ** 2
    
    train_fe['debt_to_income_cubed'] = train_fe['debt_to_income_ratio'] ** 3
    test_fe['debt_to_income_cubed'] = test_fe['debt_to_income_ratio'] ** 3
    
    train_fe['debt_to_income_sqrt'] = np.sqrt(train_fe['debt_to_income_ratio'])
    test_fe['debt_to_income_sqrt'] = np.sqrt(test_fe['debt_to_income_ratio'])
    
    print("   âœ“ Created polynomial features for debt_to_income_ratio")
    
    # -------------------------------------------------------------------
    # POLYNOMIAL: credit_score (second strongest +0.235)
    # -------------------------------------------------------------------
    train_fe['credit_score_squared'] = train_fe['credit_score'] ** 2
    test_fe['credit_score_squared'] = test_fe['credit_score'] ** 2
    
    print("   âœ“ Created polynomial features for credit_score")
    
    # -------------------------------------------------------------------
    # POLYNOMIAL: interest_rate (third strongest -0.131)
    # -------------------------------------------------------------------
    train_fe['interest_rate_squared'] = train_fe['interest_rate'] ** 2
    test_fe['interest_rate_squared'] = test_fe['interest_rate'] ** 2
    
    print("   âœ“ Created polynomial features for interest_rate")
    
    print("\n3.3 THRESHOLD/QUADRANT FEATURES (from visual analysis)...")
    
    # -------------------------------------------------------------------
    # THRESHOLD: High debt flag
    # -------------------------------------------------------------------
    train_fe['high_debt_flag'] = (train_fe['debt_to_income_ratio'] > 0.2).astype(int)
    test_fe['high_debt_flag'] = (test_fe['debt_to_income_ratio'] > 0.2).astype(int)
    
    train_fe['extreme_debt_flag'] = (train_fe['debt_to_income_ratio'] > 0.3).astype(int)
    test_fe['extreme_debt_flag'] = (test_fe['debt_to_income_ratio'] > 0.3).astype(int)
    
    print("   âœ“ Created debt threshold flags")
    
    # -------------------------------------------------------------------
    # THRESHOLD: Credit score tiers
    # -------------------------------------------------------------------
    train_fe['poor_credit_flag'] = (train_fe['credit_score'] < 600).astype(int)
    test_fe['poor_credit_flag'] = (test_fe['credit_score'] < 600).astype(int)
    
    train_fe['excellent_credit_flag'] = (train_fe['credit_score'] > 750).astype(int)
    test_fe['excellent_credit_flag'] = (test_fe['credit_score'] > 750).astype(int)
    
    print("   âœ“ Created credit score tier flags")
    
    # -------------------------------------------------------------------
    # THRESHOLD: High interest rate flag
    # -------------------------------------------------------------------
    train_fe['high_interest_flag'] = (train_fe['interest_rate'] > 15).astype(int)
    test_fe['high_interest_flag'] = (test_fe['interest_rate'] > 15).astype(int)
    
    print("   âœ“ Created interest rate flags")
    
    # -------------------------------------------------------------------
    # QUADRANT: Risk zones (from scatter plot analysis)
    # -------------------------------------------------------------------
    train_fe['danger_zone'] = (
        (train_fe['debt_to_income_ratio'] > 0.2) & (train_fe['credit_score'] < 650)
    ).astype(int)
    test_fe['danger_zone'] = (
        (test_fe['debt_to_income_ratio'] > 0.2) & (test_fe['credit_score'] < 650)
    ).astype(int)
    
    train_fe['safe_zone'] = (
        (train_fe['debt_to_income_ratio'] < 0.15) & (train_fe['credit_score'] > 700)
    ).astype(int)
    test_fe['safe_zone'] = (
        (test_fe['debt_to_income_ratio'] < 0.15) & (test_fe['credit_score'] > 700)
    ).astype(int)
    
    print("   âœ“ Created risk zone quadrant flags")
    
    print("\n3.4 AGGREGATION FEATURES...")
    
    # -------------------------------------------------------------------
    # AGG: Numeric feature statistics
    # -------------------------------------------------------------------
    numeric_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                    'loan_amount', 'interest_rate']
    
    train_fe['numeric_mean'] = train_fe[numeric_cols].mean(axis=1)
    test_fe['numeric_mean'] = test_fe[numeric_cols].mean(axis=1)
    
    train_fe['numeric_std'] = train_fe[numeric_cols].std(axis=1)
    test_fe['numeric_std'] = test_fe[numeric_cols].std(axis=1)
    
    train_fe['numeric_max'] = train_fe[numeric_cols].max(axis=1)
    test_fe['numeric_max'] = test_fe[numeric_cols].max(axis=1)
    
    train_fe['numeric_min'] = train_fe[numeric_cols].min(axis=1)
    test_fe['numeric_min'] = test_fe[numeric_cols].min(axis=1)
    
    print("   âœ“ Created aggregation features (mean, std, max, min)")
    
    print("\n3.5 DOMAIN KNOWLEDGE FEATURES...")
    
    # -------------------------------------------------------------------
    # DOMAIN: Income ratios
    # -------------------------------------------------------------------
    train_fe['loan_to_income_ratio'] = train_fe['loan_amount'] / (train_fe['annual_income'] + 1)
    test_fe['loan_to_income_ratio'] = test_fe['loan_amount'] / (test_fe['annual_income'] + 1)
    
    train_fe['income_to_debt_ratio'] = train_fe['annual_income'] / (train_fe['debt_to_income_ratio'] + 0.01)
    test_fe['income_to_debt_ratio'] = test_fe['annual_income'] / (test_fe['debt_to_income_ratio'] + 0.01)
    
    print("   âœ“ Created income ratio features")
    
    # -------------------------------------------------------------------
    # DOMAIN: Payment burden
    # -------------------------------------------------------------------
    train_fe['monthly_payment_estimate'] = (
        train_fe['loan_amount'] * (train_fe['interest_rate'] / 100) / 12
    )
    test_fe['monthly_payment_estimate'] = (
        test_fe['loan_amount'] * (test_fe['interest_rate'] / 100) / 12
    )
    
    train_fe['payment_to_income_ratio'] = (
        train_fe['monthly_payment_estimate'] / (train_fe['annual_income'] / 12 + 1)
    )
    test_fe['payment_to_income_ratio'] = (
        test_fe['monthly_payment_estimate'] / (test_fe['annual_income'] / 12 + 1)
    )
    
    print("   âœ“ Created payment burden features")
    
    return train_fe, test_fe

train_processed, test_processed = engineer_features(train_processed, test_processed)

print(f"\nâœ“ Feature engineering complete!")
print(f"   Total features after engineering: {train_processed.shape[1]}")

# ============================================================================
# STEP 4: FEATURE SELECTION
# ============================================================================

print("\n" + "="*70)
print("STEP 4: FEATURE SELECTION")
print("="*70)

# Prepare data for feature selection
X_train = train_processed.drop(['id'], axis=1, errors='ignore')
y_train = target

print(f"\n4.1 Initial feature count: {X_train.shape[1]}")
print(f"   Target shape: {y_train.shape}")

# -------------------------------------------------------------------
# 4.1 Remove constant features
# -------------------------------------------------------------------
print("\n4.1 Removing constant features...")
constant_features = [col for col in X_train.columns if X_train[col].nunique() <= 1]
if constant_features:
    print(f"   Found {len(constant_features)} constant features: {constant_features}")
    X_train = X_train.drop(columns=constant_features)
    test_processed = test_processed.drop(columns=constant_features, errors='ignore')
else:
    print("   âœ“ No constant features found")

# -------------------------------------------------------------------
# 4.2 Remove highly correlated features
# -------------------------------------------------------------------
print("\n4.2 Removing highly correlated features (>0.95)...")
correlation_matrix = X_train.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

high_corr_features = [
    column for column in upper_triangle.columns 
    if any(upper_triangle[column] > 0.95)
]

if high_corr_features:
    print(f"   Found {len(high_corr_features)} highly correlated features")
    print(f"   Dropping: {high_corr_features}")
    X_train = X_train.drop(columns=high_corr_features)
    test_processed = test_processed.drop(columns=high_corr_features, errors='ignore')
else:
    print("   âœ“ No highly correlated features found")

# -------------------------------------------------------------------
# 4.3 Calculate mutual information scores
# -------------------------------------------------------------------
print("\n4.3 Calculating mutual information scores...")
mi_scores = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=3)
mi_scores_df = pd.DataFrame({
    'feature': X_train.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\n   Top 20 features by Mutual Information:")
print(mi_scores_df.head(20).to_string(index=False))

# -------------------------------------------------------------------
# 4.4 Keep top features + ensure important ones are included
# -------------------------------------------------------------------
print("\n4.4 Selecting final feature set...")

# Always keep these critical features (from EDA)
must_keep = [
    'debt_to_income_ratio', 'credit_score', 'interest_rate',
    'grade_ordinal', 'ultimate_risk_score', 'creditworthiness_index',
    'debt_credit_ratio', 'financial_burden'
]

# Get top N features by MI score
n_features_to_keep = 50  # Adjust based on your preference
top_mi_features = mi_scores_df.head(n_features_to_keep)['feature'].tolist()

# Combine must_keep + top MI features
selected_features = list(set(must_keep + top_mi_features))
selected_features = [f for f in selected_features if f in X_train.columns]

print(f"   Selected {len(selected_features)} features")

# Apply selection
X_train_selected = X_train[selected_features]
X_test_selected = test_processed[selected_features]

print(f"\nâœ“ Feature selection complete!")
print(f"   Final feature count: {X_train_selected.shape[1]}")

# ============================================================================
# STEP 5: FINAL DATA PREPARATION
# ============================================================================

print("\n" + "="*70)
print("STEP 5: FINAL DATA PREPARATION")
print("="*70)

# Check for any remaining issues
print("\nData quality check:")
print(f"   Train shape: {X_train_selected.shape}")
print(f"   Test shape: {X_test_selected.shape}")
print(f"   Train nulls: {X_train_selected.isnull().sum().sum()}")
print(f"   Test nulls: {X_test_selected.isnull().sum().sum()}")
print(f"   Train infinite: {np.isinf(X_train_selected.select_dtypes(include=[np.number])).sum().sum()}")
print(f"   Test infinite: {np.isinf(X_test_selected.select_dtypes(include=[np.number])).sum().sum()}")

# Replace any inf values
X_train_selected = X_train_selected.replace([np.inf, -np.inf], np.nan)
X_test_selected = X_test_selected.replace([np.inf, -np.inf], np.nan)

# Fill any NaNs with 0
if X_train_selected.isnull().sum().sum() > 0:
    print(f"\n   Filling {X_train_selected.isnull().sum().sum()} NaN values with 0")
    X_train_selected = X_train_selected.fillna(0)
    X_test_selected = X_test_selected.fillna(0)

print("\nâœ“ Data is clean and ready for modeling!")

# ============================================================================
# STEP 6: SAVE PROCESSED DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 6: SAVING PROCESSED DATA")
print("="*70)

# Save processed data
X_train_selected.to_csv('X_train_final.csv', index=False)
y_train.to_csv('y_train_final.csv', index=False)
X_test_selected.to_csv('X_test_final.csv', index=False)
test_ids.to_csv('test_ids.csv', index=False)

print("\nâœ“ Files saved:")
print("   - X_train_final.csv")
print("   - y_train_final.csv")
print("   - X_test_final.csv")
print("   - test_ids.csv")

# Feature list for reference
pd.DataFrame({'feature': X_train_selected.columns}).to_csv('feature_list.csv', index=False)
print("   - feature_list.csv")

print("\n" + "="*70)
print("âœ… PREPROCESSING, FEATURE ENGINEERING & SELECTION COMPLETE!")
print("="*70)
print(f"\nğŸ“Š Summary:")
print(f"   Original features: 12")
print(f"   After encoding: ~20")
print(f"   After engineering: ~{train_processed.shape[1]}")
print(f"   After selection: {X_train_selected.shape[1]}")
print(f"\nğŸ�¯ Ready for modeling!")
print(f"   Train samples: {X_train_selected.shape[0]}")
print(f"   Test samples: {X_test_selected.shape[0]}")
print(f"   Features: {X_train_selected.shape[1]}")
print(f"   Class balance: {y_train.value_counts().to_dict()}")


# ============================================================================
# MULTI-MODEL GPU TRAINING PIPELINE - Playground S5E11
# XGBoost GPU + LightGBM GPU + CatBoost GPU + Ensemble
# ============================================================================

from sklearn.metrics import f1_scor

print("="*70)
print(" GPU-ACCELERATED MODEL TRAINING PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: LOAD PROCESSED DATA
# ============================================================================

print("\n Loading processed data...")

X_train = pd.read_csv('X_train_final.csv')
y_train = pd.read_csv('y_train_final.csv').values.ravel()
X_test = pd.read_csv('X_test_final.csv')
test_ids = pd.read_csv('test_ids.csv').values.ravel()

#  FIX: Convert y_train to integer
y_train = y_train.astype(int)

print(f"âœ“ X_train: {X_train.shape}")
print(f"âœ“ y_train: {y_train.shape}")
print(f"âœ“ X_test: {X_test.shape}")
print(f"âœ“ y_train dtype: {y_train.dtype}")  # Should be int now
print(f"âœ“ Class distribution: {np.bincount(y_train)}")

# Calculate class weights for imbalance
class_0_count = np.sum(y_train == 0)
class_1_count = np.sum(y_train == 1)
scale_pos_weight = class_1_count / class_0_count
print(f"âœ“ Scale pos weight: {scale_pos_weight:.4f}")

# ============================================================================
# STEP 2: CROSS-VALIDATION SETUP
# ============================================================================

print("\n" + "="*70)
print(" CROSS-VALIDATION SETUP")
print("="*70)

N_SPLITS = 5
RANDOM_STATE = 42

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print(f"âœ“ Using {N_SPLITS}-Fold Stratified CV")
print(f"âœ“ Random state: {RANDOM_STATE}")

# Storage for predictions
oof_predictions = {}  # Out-of-fold predictions for each model
test_predictions = {}  # Test predictions for each model
cv_scores = {}  # CV scores for each model

# ============================================================================
# MODEL 1: XGBOOST GPU
# ============================================================================

print("\n" + "="*70)
print(" MODEL 1: XGBOOST GPU")
print("="*70)

import xgboost as xgb

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',  # GPU accelerated
    'device': 'cuda',  # Use GPU
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1/scale_pos_weight,  # Handle imbalance
    'random_state': RANDOM_STATE,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'verbosity': 1
}

print("Parameters:")
for key, val in xgb_params.items():
    print(f"   {key}: {val}")

# Cross-validation
xgb_oof = np.zeros(len(X_train))
xgb_test_preds = np.zeros(len(X_test))
xgb_scores = []

print("\nğŸ”„ Training XGBoost with CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n   Fold {fold}/{N_SPLITS}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Train
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    xgb_oof[val_idx] = val_preds
    
    # Test predictions (average across folds)
    xgb_test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # Score
    fold_auc = roc_auc_score(y_val, val_preds)
    xgb_scores.append(fold_auc)
    print(f"      Fold {fold} AUC: {fold_auc:.6f}")

# Overall CV score
xgb_cv_score = roc_auc_score(y_train, xgb_oof)
print(f"\n XGBoost CV AUC: {xgb_cv_score:.6f} (+/- {np.std(xgb_scores):.6f})")

oof_predictions['xgboost'] = xgb_oof
test_predictions['xgboost'] = xgb_test_preds
cv_scores['xgboost'] = xgb_cv_score

# ============================================================================
# MODEL 2: LIGHTGBM GPU
# ============================================================================

print("\n" + "="*70)
print("âš¡ MODEL 2: LIGHTGBM GPU")
print("="*70)

import lightgbm as lgb

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',  # Use GPU
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'boosting_type': 'gbdt',
    'max_depth': 7,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1/scale_pos_weight,
    'random_state': RANDOM_STATE,
    'n_estimators': 1000,
    'verbosity': -1
}

print("Parameters:")
for key, val in lgb_params.items():
    print(f"   {key}: {val}")

# Cross-validation
lgb_oof = np.zeros(len(X_train))
lgb_test_preds = np.zeros(len(X_test))
lgb_scores = []

print("\n Training LightGBM with CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n   Fold {fold}/{N_SPLITS}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Train
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    lgb_oof[val_idx] = val_preds
    
    # Test predictions
    lgb_test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # Score
    fold_auc = roc_auc_score(y_val, val_preds)
    lgb_scores.append(fold_auc)
    print(f"      Fold {fold} AUC: {fold_auc:.6f}")

# Overall CV score
lgb_cv_score = roc_auc_score(y_train, lgb_oof)
print(f"\n LightGBM CV AUC: {lgb_cv_score:.6f} (+/- {np.std(lgb_scores):.6f})")

oof_predictions['lightgbm'] = lgb_oof
test_predictions['lightgbm'] = lgb_test_preds
cv_scores['lightgbm'] = lgb_cv_score



# ============================================================================
# MODEL 3: CATBOOST GPU
# ============================================================================

print("\n" + "="*70)
print(" MODEL 3: CATBOOST GPU")
print("="*70)

from catboost import CatBoostClassifier, Pool

cat_params = {

    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': 'GPU',  # Use GPU
    'devices': '0',
    'depth': 7,
    'learning_rate': 0.05,
    'iterations': 1000,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'min_data_in_leaf': 20,
    'l2_leaf_reg': 3,
    'random_strength': 0.5,
    'scale_pos_weight': 1/scale_pos_weight,
    'random_seed': RANDOM_STATE,
    'early_stopping_rounds': 50,
    'verbose': 100
}

print("Parameters:")
for key, val in cat_params.items():
    print(f"   {key}: {val}")

# Cross-validation
cat_oof = np.zeros(len(X_train))
cat_test_preds = np.zeros(len(X_test))
cat_scores = []

print("\n Training CatBoost with CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n   Fold {fold}/{N_SPLITS}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Create pools
    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val, y_val)
    
    # Train
    model = CatBoostClassifier(**cat_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=False
    )
    
    # Predict
    val_preds = model.predict_proba(X_val)[:, 1]
    cat_oof[val_idx] = val_preds
    
    # Test predictions
    cat_test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # Score
    fold_auc = roc_auc_score(y_val, val_preds)
    cat_scores.append(fold_auc)
    print(f"      Fold {fold} AUC: {fold_auc:.6f}")

# Overall CV score
cat_cv_score = roc_auc_score(y_train, cat_oof)
print(f"\n CatBoost CV AUC: {cat_cv_score:.6f} (+/- {np.std(cat_scores):.6f})")

oof_predictions['catboost'] = cat_oof
test_predictions['catboost'] = cat_test_preds
cv_scores['catboost'] = cat_cv_score

# ============================================================================
# STEP 3: MODEL COMPARISON
# ============================================================================

print("\n" + "="*70)
print(" MODEL COMPARISON")
print("="*70)

print("\nCross-Validation AUC Scores:")
print("-" * 40)
for model_name, score in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"   {model_name:15s}: {score:.6f}")

# Visualize
plt.figure(figsize=(10, 6))
models = list(cv_scores.keys())
scores = list(cv_scores.values())
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = plt.bar(models, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
plt.ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
plt.ylim(min(scores) - 0.01, max(scores) + 0.01)

# Add value labels
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{score:.6f}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nâœ“ Comparison plot saved: model_comparison.png")

# ============================================================================
# STEP 4: ENSEMBLE PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print(" ENSEMBLE PREDICTIONS")
print("="*70)

# Simple average ensemble
ensemble_oof = np.mean([oof_predictions['xgboost'], 
                        oof_predictions['lightgbm'],
                        oof_predictions['catboost']], axis=0)

ensemble_test = np.mean([test_predictions['xgboost'],
                         test_predictions['lightgbm'],
                         test_predictions['catboost']], axis=0)

ensemble_cv_score = roc_auc_score(y_train, ensemble_oof)

print(f"\n Simple Average Ensemble CV AUC: {ensemble_cv_score:.6f}")

# Weighted ensemble (weights based on CV performance)
weights = np.array([cv_scores['xgboost'], cv_scores['lightgbm'], cv_scores['catboost']])
weights = weights / weights.sum()

print(f"\nWeights: XGB={weights[0]:.3f}, LGB={weights[1]:.3f}, CAT={weights[2]:.3f}")

weighted_oof = (weights[0] * oof_predictions['xgboost'] +
                weights[1] * oof_predictions['lightgbm'] +
                weights[2] * oof_predictions['catboost'])

weighted_test = (weights[0] * test_predictions['xgboost'] +
                 weights[1] * test_predictions['lightgbm'] +
                 weights[2] * test_predictions['catboost'])

weighted_cv_score = roc_auc_score(y_train, weighted_oof)

print(f" Weighted Ensemble CV AUC: {weighted_cv_score:.6f}")

# ============================================================================
# STEP 5: GENERATE SUBMISSIONS
# ============================================================================

print("\n" + "="*70)
print(" GENERATING SUBMISSION FILES")
print("="*70)

# Submission 1: Best single model
best_model = max(cv_scores, key=cv_scores.get)
best_preds = test_predictions[best_model]

submission_best = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': best_preds
})
submission_best.to_csv('submission_best_single.csv', index=False)
print(f"\nâœ“ Best single model ({best_model}): submission_best_single.csv")
print(f"   CV AUC: {cv_scores[best_model]:.6f}")

# Submission 2: Simple average ensemble
submission_avg = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': ensemble_test
})
submission_avg.to_csv('submission_ensemble_avg.csv', index=False)
print(f"\nâœ“ Average ensemble: submission_ensemble_avg.csv")
print(f"   CV AUC: {ensemble_cv_score:.6f}")

# Submission 3: Weighted ensemble
submission_weighted = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': weighted_test
})
submission_weighted.to_csv('submission_ensemble_weighted.csv', index=False)
print(f"\nâœ“ Weighted ensemble: submission_ensemble_weighted.csv")
print(f"   CV AUC: {weighted_cv_score:.6f}")

# ============================================================================
# STEP 6: FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" TRAINING COMPLETE - FINAL SUMMARY")
print("="*70)

print("\n Model Performance:")
print("-" * 50)
print(f"   XGBoost GPU:          {cv_scores['xgboost']:.6f}")
print(f"   LightGBM GPU:         {cv_scores['lightgbm']:.6f}")
print(f"   CatBoost GPU:         {cv_scores['catboost']:.6f}")
print(f"   Average Ensemble:     {ensemble_cv_score:.6f}")
print(f"   Weighted Ensemble:    {weighted_cv_score:.6f}")

print("\n Files Generated:")
print("-" * 50)
print("   âœ“ submission_best_single.csv")
print("   âœ“ submission_ensemble_avg.csv")
print("   âœ“ submission_ensemble_weighted.csv")
print("   âœ“ model_comparison.png")

print("\nğŸ�¯ Recommended Submission:")
if weighted_cv_score >= ensemble_cv_score and weighted_cv_score >= max(cv_scores.values()):
    print("    submission_ensemble_weighted.csv (BEST!)")
elif ensemble_cv_score >= max(cv_scores.values()):
    print("    submission_ensemble_avg.csv (BEST!)")
else:
    print(f"  submission_best_single.csv ({best_model}) (BEST!)")




# ============================================================================
# POST-PROCESSING: RANK ENSEMBLE (LGBM + XGB ONLY)
# ============================================================================

import pandas as pd
import numpy as np
from scipy.stats import rankdata

print("="*70)
print(" GENERATING RANK ENSEMBLE (Dropping CatBoost)")
print("="*70)

# 1. Load the OOF predictions (from memory)
# We trust LightGBM most, XGBoost second.
oof_lgb = oof_predictions['lightgbm']
oof_xgb = oof_predictions['xgboost']

# 2. Load the Test predictions (from memory)
test_lgb = test_predictions['lightgbm']
test_xgb = test_predictions['xgboost']

# 3. Perform Rank Averaging on Test Set
# Formula: (Rank(Model A) * Weight A) + (Rank(Model B) * Weight B)
# We give slightly more weight to LGBM because it had higher CV
w_lgb = 0.60
w_xgb = 0.40

print(f"Blending: {w_lgb}*LightGBM + {w_xgb}*XGBoost")

# Rank test predictions (returns 1 to N)
rank_lgb = rankdata(test_lgb)
rank_xgb = rankdata(test_xgb)

# Combine ranks (normalized between 0 and 1)
final_rank_prediction = (rank_lgb * w_lgb + rank_xgb * w_xgb) / len(test_lgb)

# 4. Create Submission
submission_rank = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': final_rank_prediction
})

#5. # ============================================================================
# FINAL ULTRA-STABLE MASTER BLEND (Optimized Weights)
# ============================================================================
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import os

print("="*70)
print("ğŸ‘‘ EXECUTING FINAL ULTRA-STABLE BLEND (Target: 0.924+)")
print("="*70)

# 1. Define the Arsenal and FINAL Weights
submission_weights = {
    # Pillar 1: Optimized LightGBM
    'submission_lgbm_optuna.csv': 0.40,     
    
    # Pillar 2: Optimized XGBoost
    'submission_xgb_optuna.csv':  0.30,     
    
    # Pillar 3: TF-DF (Crucial Non-GBM Diversity)
    'submission_tfdf.csv':        0.15,     
    
    # Base Ensemble (Robustness/Previous Best Blend)
    'submission_rank_ensemble.csv': 0.15,   
    
    # Note: NN, Pseudo, and TabNet files are omitted for stability.
}

final_pred = np.zeros(len(pd.read_csv('test_ids.csv')))
total_weight = 0
found_models = []

print("\nBlending Models with Weights:")
print("-" * 50)

for filename, weight in submission_weights.items():
    if os.path.exists(filename):
        print(f"  âœ… Blended {filename} (Weight: {weight:.2f})")
        p = pd.read_csv(filename)['loan_paid_back'].values
        
        # Rank Transform (Essential for combining different models)
        ranked_p = rankdata(p) / len(p)
        
        # Apply weighted average
        final_pred += ranked_p * weight
        total_weight += weight
        found_models.append(filename)
    else:
        # We should not be missing any of these pillars, but we check.
        print(f"â�Œ WARNING: Missing {filename}. Cannot achieve 100% target weight.")

# 2. Normalize and Save
if total_weight > 0.99: # Ensure all files were found (total weight should be 1.00)
    final_pred /= total_weight
    
    submission_name = 'submission_FINAL_TOP_10.csv'
    pd.DataFrame({
        'id': pd.read_csv('test_ids.csv').iloc[:,0],
        'loan_paid_back': final_pred
    }).to_csv(submission_name, index=False)
    
    print("-" * 50)
    print(f"ğŸš€ GENERATED: {submission_name}")
    print("   Submit this file immediately. It represents the optimized peak performance.")
else:
    print("\nâš ï¸� ERROR: Missing critical submission files. Please verify that Optuna and TF-DF ran successfully.")ali

submission_rank.to_csv('submission_rank_ensemble.csv', index=False)
print("\nâœ“ Generated: submission_rank_ensemble.csv")
print("  Insight: Rank averaging is robust against calibration differences.")
print("  Action: Submit this file to Kaggle!")


!pip install tensorflow_decision_forests -q


# ============================================================================
# STAGE 1 (CORRECTED & VERIFIED): DEEP LEARNING ARSENAL
# ============================================================================

import pandas as pd
import numpy as np
import tensorflow_decision_forests as tfdf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("="*70)
print(" LAUNCHING DEEP LEARNING PIPELINE (INDEX FIX APPLIED)")
print("="*70)

# 1. Load Data
X_train_full = pd.read_csv('X_train_final.csv')
y_train_full = pd.read_csv('y_train_final.csv').values.ravel()
X_test = pd.read_csv('X_test_final.csv')
test_ids = pd.read_csv('test_ids.csv').iloc[:,0]

# 2. Create Validation Split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"Data Split: Train {X_tr.shape}, Val {X_val.shape}")

# 3. Fix Data Types for TF-DF (Convert Boolean to Int)
# (Must do this on copies to avoid SettingWithCopy warnings)
X_tr = X_tr.copy()
X_val = X_val.copy()
X_test = X_test.copy()

bool_cols = X_tr.select_dtypes(include=['bool']).columns
if len(bool_cols) > 0:
    print(f"Converting {len(bool_cols)} boolean cols to int...")
    for col in bool_cols:
        X_tr[col] = X_tr[col].astype(int)
        X_val[col] = X_val[col].astype(int)
        X_test[col] = X_test[col].astype(int)

# ============================================================================
# MODEL A: TENSORFLOW DECISION FORESTS (TF-DF)
# ============================================================================
print("\n[1/2] Training TF-DF (Gradient Boosted Trees)...")

# ğŸ› ï¸� FIX: Assign target directly to copy instead of concat
# This ensures alignment by position, ignoring index mismatches
train_df_tf = X_tr.copy()
train_df_tf['target'] = y_tr  # Assign numpy array directly

val_df_tf = X_val.copy()
val_df_tf['target'] = y_val

# Convert to TF Datasets
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df_tf, label="target")
val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(val_df_tf, label="target")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test)

# Train
model_tfdf = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.CLASSIFICATION, verbose=0)
model_tfdf.fit(train_ds)

# SCORE
val_pred_tfdf = model_tfdf.predict(val_ds).flatten()
score_tfdf = roc_auc_score(y_val, val_pred_tfdf)
print(f" TF-DF Validation AUC: {score_tfdf:.5f}")

# PREDICT TEST
pred_tfdf = model_tfdf.predict(test_ds).flatten()
pd.DataFrame({'id': test_ids, 'loan_paid_back': pred_tfdf}).to_csv('submission_tfdf.csv', index=False)
print("âœ“ Saved submission_tfdf.csv")


# # ============================================================================
# # OPTUNA HYPERPARAMETER OPTIMIZATION (LIGHTGBM + GPU)
# # ============================================================================

# import lightgbm as lgb
# import optuna


# print("="*70)
# print(" LAUNCHING OPTUNA (LGBM on GPU)")
# print("="*70)

# # 1. Load Final Data
# X_train = pd.read_csv('X_train_final.csv')
# y_train = pd.read_csv('y_train_final.csv').values.ravel()
# X_test = pd.read_csv('X_test_final.csv')
# test_ids = pd.read_csv('test_ids.csv').iloc[:,0]

# # Fix Dtypes (if any booleans slipped through)
# for df in [X_train, X_test]:
#     bool_cols = df.select_dtypes(include=['bool']).columns
#     for col in bool_cols:
#         df[col] = df[col].astype(int)

# # 2. Define the Objective Function (3-Fold CV)
# def objective_lgbm(trial):
#     # Parameters to be optimized
#     params = {
#         'objective': 'binary',
#         'metric': 'auc',
#         'device': 'gpu', # CRITICAL: GPU ACCELERATION
#         'boosting_type': 'gbdt',
#         'n_estimators': 1500, # Use more trees, early stopping will find the best
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.008, 0.05),
#         'num_leaves': trial.suggest_int('num_leaves', 16, 64),
#         'max_depth': trial.suggest_int('max_depth', 5, 10),
#         'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
#         'subsample': trial.suggest_float('subsample', 0.6, 0.95),
#         'scale_pos_weight': 4.0, # Handle Imbalance
#         'random_state': 42,
#         'n_jobs': -1,
#         'verbose': -1,
#     }

#     # Internal 3-Fold CV for robustness
#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     oof_preds = np.zeros(len(X_train))
    
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
#         X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_tr, y_val = y_train[train_idx], y_train[val_idx]

#         model = lgb.LGBMClassifier(**params)
#         model.fit(
#             X_tr, y_tr,
#             eval_set=[(X_val, y_val)],
#             callbacks=[lgb.early_stopping(50, verbose=False)],
#         )
#         oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

#     # Return the global OOF AUC score for Optuna to maximize
#     return roc_auc_score(y_train, oof_preds)

# # 3. Run Optimization
# # We use TPE sampler (Tree-structured Parzen Estimator) as it's efficient for GBMs
# study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
# N_TRIALS = 20 # Set this higher (e.g., 200) if you have more time
# print(f"\nOptimization started: {N_TRIALS} trials.")
# study.optimize(objective_lgbm, n_trials=N_TRIALS, show_progress_bar=True)

# # 4. Final Training and Submission
# print("\n" + "="*70)
# print(" FINAL TRAINING WITH OPTIMIZED PARAMETERS")
# print("="*70)

# # Final model parameters
# best_params = study.best_params
# best_params.update({
#     'objective': 'binary',
#     'metric': 'auc',
#     'device': 'gpu',
#     'n_estimators': 3000, # More trees for the final run
#     'scale_pos_weight': 4.0,
#     'random_state': 42,
#     'n_jobs': -1,
#     'verbose': -1,
# })

# # Train final model on FULL dataset
# final_model_lgbm = lgb.LGBMClassifier(**best_params)
# final_model_lgbm.fit(X_train, y_train)

# # Predict and Save
# preds_optuna = final_model_lgbm.predict_proba(X_test)[:, 1]
# submission_optuna = pd.DataFrame({
#     'id': test_ids,
#     'loan_paid_back': preds_optuna
# })
# submission_optuna.to_csv('submission_lgbm_optuna.csv', index=False)

# print(f"\n BEST CV AUC Found: {study.best_value:.6f}")
# print("âœ“ Saved 'submission_lgbm_optuna.csv'")
# print("---")
# print("NEXT: Blend this file with your TF-DF and XGBoost submissions!")


import pandas as pd
import lightgbm as lgb
import numpy as np
import os

# Final model parameters
best_params = study.best_params
best_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',
    'n_estimators': 3000, # Use maximum trees for final run
    'scale_pos_weight': 4.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
})

print("="*70)
print(" FINAL TRAINING WITH OPTIMIZED PARAMETERS")
print("="*70)
print(f"Parameters Used: {best_params}")

# 1. Load Data (Full Set)
X_train = pd.read_csv('X_train_final.csv')
y_train = pd.read_csv('y_train_final.csv').values.ravel()
X_test = pd.read_csv('X_test_final.csv')
test_ids = pd.read_csv('test_ids.csv').iloc[:,0]

# Fix Dtypes
for df in [X_train, X_test]:
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

# 2. Train final model on FULL dataset
final_model_lgbm = lgb.LGBMClassifier('learning_rate': 0.01589201562116427, 'num_leaves': 62, 'max_depth': 9, 'min_child_samples': 68, 'reg_alpha': 2.5361081166471375e-07, 'reg_lambda': 2.5348407664333426e-07, 'colsample_bytree': 0.6203292642588698, 'subsample': 0.9031616510212273, 'objective': 'binary', 'metric': 'auc', 'device': 'gpu', 'n_estimators': 3000, 'scale_pos_weight': 4.0, 'random_state': 42, 'n_jobs': -1, 'verbose': -1)
print("\nTraining final model on 100% of data...")
final_model_lgbm.fit(X_train, y_train)

# 3. Predict and Save
preds_optuna = final_model_lgbm.predict_proba(X_test)[:, 1]
submission_optuna = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': preds_optuna
})

submission_optuna.to_csv('submission_lgbm_optuna.csv', index=False)

print("\n Final Optimized LGBM Submission Saved: 'submission_lgbm_optuna.csv'")


# ============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION (XGBOOST + GPU)
# ============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print("="*70)
print(" LAUNCHING OPTUNA (XGBoost on GPU)")
print("="*70)

# 1. Load Final Data (Assumed already loaded, but included for robustness)
X_train = pd.read_csv('X_train_final.csv')
y_train = pd.read_csv('y_train_final.csv').values.ravel()

# Fix Dtypes (if any booleans slipped through)
for df in [X_train]:
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

# 2. Define the Objective Function (3-Fold CV)
def objective_xgb(trial):
    # Parameters to be optimized
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist', # CRITICAL: Enables GPU-accelerated histogram tree method
        'device': 'cuda',      # CRITICAL: Forces GPU usage
        'n_estimators': 1500,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.008, 0.05),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0), # Complexity control
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'scale_pos_weight': 4.0, # Handle Imbalance
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    # Internal 3-Fold CV for robustness
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Return the global OOF AUC score
    return roc_auc_score(y_train, oof_preds)

# 3. Run Optimization
study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
N_TRIALS = 100
print(f"\nOptimization started: {N_TRIALS} trials.")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)

# 4. Final Training and Submission
print("\n" + "="*70)
print(" FINAL TRAINING WITH OPTIMIZED PARAMETERS")
print("="*70)

# Final model parameters
best_params = study_xgb.best_params
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'n_estimators': 3000, 
    'scale_pos_weight': 4.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
})

# Train final model on FULL dataset
final_model_xgb = xgb.XGBClassifier(**best_params)
final_model_xgb.fit(X_train, y_train)

# Predict and Save
preds_optuna_xgb = final_model_xgb.predict_proba(X_test)[:, 1]
submission_optuna_xgb = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': preds_optuna_xgb
})
submission_optuna_xgb.to_csv('submission_xgb_optuna.csv', index=False)

print(f"\n XGBoost Best CV AUC Found: {study_xgb.best_value:.6f}")
print("âœ“ Saved 'submission_xgb_optuna.csv'")


X_test = pd.read_csv('X_test_final.csv')
test_ids = pd.read_csv('test_ids.csv').iloc[:,0]
print("\n" + "="*70)
print(" FINAL TRAINING WITH OPTIMIZED PARAMETERS")
print("="*70)

# Final model parameters
best_params = study_xgb.best_params
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'n_estimators': 3000, 
    'scale_pos_weight': 4.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
})

# Train final model on FULL dataset
final_model_xgb = xgb.XGBClassifier(**best_params)
final_model_xgb.fit(X_train, y_train)

# Predict and Save
preds_optuna_xgb = final_model_xgb.predict_proba(X_test)[:, 1]
submission_optuna_xgb = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': preds_optuna_xgb
})
submission_optuna_xgb.to_csv('submission_xgb_optuna.csv', index=False)

print(f"\n XGBoost Best CV AUC Found: {study_xgb.best_value:.6f}")
print("âœ“ Saved 'submission_xgb_optuna.csv'")


# ============================================================================
# FINAL ULTRA-STABLE MASTER BLEND (Optimized Weights)
# ============================================================================
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import os

print("="*70)
print(" EXECUTING FINAL ULTRA-STABLE BLEND (Target: 0.924+)")
print("="*70)

# 1. Define the Arsenal and FINAL Weights
submission_weights = {
    # Pillar 1: Optimized LightGBM
    'submission_lgbm_optuna.csv': 0.40,     
    
    # Pillar 2: Optimized XGBoost
    'submission_xgb_optuna.csv':  0.40,     
    
    # Pillar 3: TF-DF (Crucial Non-GBM Diversity)
    'submission_tfdf.csv':        0.15,     
    
    # Base Ensemble (Robustness/Previous Best Blend)
    'submission_rank_ensemble.csv': 0.15,   
}

final_pred = np.zeros(len(pd.read_csv('test_ids.csv')))
total_weight = 0
found_models = []

print("\nBlending Models with Weights:")
print("-" * 50)

for filename, weight in submission_weights.items():
    if os.path.exists(filename):
        print(f"  âœ… Blended {filename} (Weight: {weight:.2f})")
        p = pd.read_csv(filename)['loan_paid_back'].values
        
        # Rank Transform (Essential for combining different models)
        ranked_p = rankdata(p) / len(p)
        
        # Apply weighted average
        final_pred += ranked_p * weight
        total_weight += weight
        found_models.append(filename)
    else:
        # We should not be missing any of these pillars, but we check.
        print(f" WARNING: Missing {filename}. Cannot achieve 100% target weight.")

# 2. Normalize and Save
if total_weight > 0.99: # Ensure all files were found (total weight should be 1.00)
    final_pred /= total_weight
    
    submission_name = 'submission_FINAL_TOP_10.csv'
    pd.DataFrame({
        'id': pd.read_csv('test_ids.csv').iloc[:,0],
        'loan_paid_back': final_pred
    }).to_csv(submission_name, index=False)
    
    print("-" * 50)
    print(f" GENERATED: {submission_name}")
    print("   Submit this file immediately. It represents the optimized peak performance.")
else:
    print("\n ERROR: Missing critical submission files. Please verify that Optuna and TF-DF ran successfully.")


from scipy.stats import rankdata

print("="*70)
print("âˆ› FINAL CUBE ROOT BLEND")
print("="*70)

# 1. Define Arsenal (Using the previous weights)
files_to_blend = {
    'submission_lgbm_optuna.csv': 0.40,
    'submission_xgb_optuna.csv':  0.40,
    'submission_tfdf.csv':        0.20,
}

final_pred = np.zeros(len(pd.read_csv('test_ids.csv')))
total_weight = 0

for filename, weight in files_to_blend.items():
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        p = df['loan_paid_back'].values
        
        # ğŸ§ª TRICK: Apply the cube root (power of 1/3)
        # We must clip values slightly away from 0 and 1 before taking the cube root
        p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
        p_powered = np.power(p_clipped, 1/3)
        
        # We apply linear weighted average on the transformed values
        final_pred += p_powered * weight
        total_weight += weight
    else:
        print(f"â�Œ Missing {filename} - Cannot complete blend.")
        break

# Normalize and Re-exponentiate (Cube the final result)
if total_weight > 0.99:
    final_pred /= total_weight
    final_pred = np.power(final_pred, 3) # Cube root blend requires final cubing
    
    submission_name = 'submission_CUBE_BLEND_FINAL.csv'
    pd.DataFrame({
        'id': pd.read_csv('test_ids.csv').iloc[:,0],
        'loan_paid_back': final_pred
    }).to_csv(submission_name, index=False)
    
    print("-" * 50)
    print(f"ğŸš€ GENERATED: {submission_name}")




