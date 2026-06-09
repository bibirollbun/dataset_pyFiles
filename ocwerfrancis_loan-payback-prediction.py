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


import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


import warnings
warnings.filterwarnings('ignore')


loan_df = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
loan_df.head()



train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
train_df.head()


train_df.shape


test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
test_df.head()


test_df.shape


loan_df.shape, train_df.shape, test_df.shape


train_df=train_df.drop(['id'],axis=1)
test_df=test_df.drop(['id'],axis=1)


loan_df=loan_df[train_df.columns.tolist()]
loan_df.head()


train_df = pd.concat([train_df, loan_df], ignore_index=True)
train_df=train_df.sample(frac=1)
train_df= train_df.reset_index(drop=True)
train_df.head()


print("\nMissing Values:")
print(train_df.isnull().sum())


train_df.info()


train_df.describe().T


round(train_df.loan_paid_back.value_counts(normalize=True) *100)


categorical_cols = ['gender', 'marital_status','education_level','employment_status', 'loan_purpose']

for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    
    # Countplot vs loan_paid_back
    plt.figure(figsize=(10, 4))
    ax = sns.countplot(x=train_df[col], hue=train_df['loan_paid_back'], palette='Set1', order=train_df[col].value_counts().index)
    plt.title(f'{col} vs Loan Paid Back' ,fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)

    # Add percentage annotations on bars
    total = len(train_df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + total * 0.01
        ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=9)
    
    # Improve legend
    plt.legend(title='Loan Paid Back', labels=['Defaulted (0)', 'Paid Back (1)'])
    
    plt.tight_layout()
    plt.show()


# Target distribution
plt.figure(figsize=(8,5))
sns.countplot(x='loan_paid_back', data=train_df)
plt.title('loan_paid_back Distribution')
plt.show()


train_df.hist(bins=60,figsize=(30,20))


numeric_df = train_df.select_dtypes(include=['int64', 'float64'])
numeric_df.corr()['loan_paid_back'].sort_values(ascending=False)


corr_matrix = train_df.corr(numeric_only=True)

plt.figure(figsize=(20,10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
)


from pandas.plotting import scatter_matrix


train_df.dtypes


numeric_df = [
    'annual_income','credit_score',
    'debt_to_income_ratio','interest_rate',
    'loan_amount']

sns.pairplot(train_df[numeric_df],diag_kind='kde',corner=True)
plt.suptitle('Scatter Matrix (Seaborn Pairplot)', fontsize=22, y=1.02)
# Save as PNG (high-resolution)
plt.savefig("loan_scatter_matrix.png", dpi=300, bbox_inches='tight')
plt.show()


categorical_cols = [col for col in loan_df.columns if loan_df[col].dtype == 'O']
numerical_cols = [col for col in loan_df.columns.drop('loan_paid_back') if loan_df[col].dtype != 'O']
print("FEATURE TYPE SUMMARY")
print("="*80)
print(f"\n Numerical features ({len(numerical_cols)}):")
for i, col in enumerate(numerical_cols, 1):
    print(f"   {i}. {col}")

print(f"\n Categorical features ({len(categorical_cols)}):")
for i, col in enumerate(categorical_cols, 1):
    print(f"   {i}. {col}")

print(f"\n Total predictive features: {len(numerical_cols) + len(categorical_cols)}")



# Statistical summary of numerical features
print("NUMERICAL FEATURES - STATISTICAL SUMMARY")
print("="*80)

numerical_stats = train_df[numerical_cols].describe().T
numerical_stats['missing'] = train_df[numerical_cols].isnull().sum().values
numerical_stats['skewness'] = train_df[numerical_cols].skew().values
numerical_stats['kurtosis'] = train_df[numerical_cols].kurtosis().values

display(numerical_stats.style.background_gradient(cmap='coolwarm', subset=['mean', 'std', 'skewness', 'kurtosis']))


# Distribution of numerical features
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    # Plot distribution with KDE
    axes[idx].hist(train_df[col], bins=50, alpha=0.6, color='steelblue', edgecolor='black', density=True, label='Histogram')
    
    # Add KDE
    train_df[col].plot(kind='kde', ax=axes[idx], color='red', linewidth=2, label='KDE')
    
    axes[idx].set_title(f'{col} Distribution', fontsize=13, fontweight='bold', pad=10)
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Density', fontsize=11)
    axes[idx].grid(alpha=0.3, linestyle='--')
    
    # Add statistics box
    mean_val = train_df[col].mean()
    median_val = train_df[col].median()
    std_val = train_df[col].std()
    
    axes[idx].axvline(mean_val, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.2f}')
    axes[idx].axvline(median_val, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Median: {median_val:.2f}')
    
    axes[idx].legend(fontsize=9, loc='upper right')

plt.suptitle('Numerical Features Distribution Analysis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()


# Categorical features summary
print("CATEGORICAL FEATURES - DETAILED ANALYSIS")
print("="*80)

for col in categorical_cols:
    print(f"Feature: {col.upper()}")
    print(f"{'='*80}")
    print(f"Unique values: {train_df[col].nunique()}")
    print(f"Most common: {train_df[col].mode()[0]}")
    print(f"\nValue Counts:")
    
    value_counts_df = pd.DataFrame({
        'Value': train_df[col].value_counts().index,
        'Count': train_df[col].value_counts().values,
        'Percentage': (train_df[col].value_counts(normalize=True) * 100).values
    })
    display(value_counts_df.head(10).style.background_gradient(cmap='Blues', subset=['Count', 'Percentage']))


# Outlier detection using IQR method
print("OUTLIER DETECTION (IQR METHOD)")
print("="*80)

outlier_summary = []

for col in numerical_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)][col]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(train_df)) * 100
    
    outlier_summary.append({
        'Feature': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'Outlier_Count': outlier_count,
        'Outlier_Percentage': outlier_pct
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Percentage', ascending=False)
display(outlier_df.style.background_gradient(cmap='Reds', subset=['Outlier_Count', 'Outlier_Percentage']))

print(f"\n Outlier Summary:")
print(f"   - Total features with outliers: {(outlier_df['Outlier_Count'] > 0).sum()}")
print(f"   - Average outlier percentage: {outlier_df['Outlier_Percentage'].mean():.2f}%")


train_df = train_df.copy()
test_df = test_df.copy()
loan_df = loan_df.copy()


def advanced_feature_engineering(df, is_train=True):
    """
    Comprehensive feature engineering for loan prediction
    
    This function creates multiple types of features:
    - Financial ratios and metrics
    - Risk scores and composite metrics
    - Interaction features
    - Binned/categorical versions of numerical features
    - Statistical aggregations
    - Domain-specific features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (train or test)
    is_train : bool
        Whether this is training data
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with engineered features
    """
    
    df = df.copy()
    
    print("FEATURE ENGINEERING PIPELINE")
    print("="*80)
    print(f"Starting features: {df.shape[1]}")
    
    # ========================================
    # 1. FINANCIAL RATIO FEATURES
    # ========================================
    print("\n[1/11]  Creating financial ratio features...")
    
    # Core financial ratios
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_payment_estimate'] = (df['loan_amount'] * df['interest_rate']) / 1200
    df['payment_to_income_ratio'] = df['monthly_payment_estimate'] / (df['monthly_income'] + 1)
    
    # Debt calculations
    df['current_debt_amount'] = df['debt_to_income_ratio'] * df['annual_income']
    df['total_debt_with_loan'] = df['current_debt_amount'] + df['loan_amount']
    df['new_debt_to_income'] = df['total_debt_with_loan'] / (df['annual_income'] + 1)
    df['debt_increase_ratio'] = df['new_debt_to_income'] / (df['debt_to_income_ratio'] + 0.01)
    df['debt_increase_amount'] = df['loan_amount']
    
    # Disposable income
    df['disposable_income'] = df['annual_income'] - df['current_debt_amount']
    df['disposable_income_ratio'] = df['disposable_income'] / (df['annual_income'] + 1)
    df['loan_to_disposable_income'] = df['loan_amount'] / (df['disposable_income'] + 1)
    df['monthly_disposable_income'] = df['disposable_income'] / 12
    
    # Payment burden
    df['payment_to_disposable_ratio'] = df['monthly_payment_estimate'] / (df['monthly_disposable_income'] + 1)
    df['annual_payment_burden'] = df['monthly_payment_estimate'] * 12
    df['payment_burden_ratio'] = df['annual_payment_burden'] / (df['annual_income'] + 1)
    
    print(f"âœ“ Created 16 financial ratio features")
    
    # ========================================
    # 2. CREDIT SCORE FEATURES
    # ========================================
    print("\n[2/11]  Creating credit score features...")
    
    # Normalize and transform credit score
    df['credit_score_normalized'] = df['credit_score'] / 850
    df['credit_risk_score'] = 1 - df['credit_score_normalized']
    df['credit_score_squared'] = df['credit_score'] ** 2
    df['credit_score_log'] = np.log1p(df['credit_score'])
    
    # Credit categories
    df['credit_category'] = pd.cut(df['credit_score'], 
                                     bins=[0, 580, 670, 740, 800, 850],
                                     labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
    
    # Credit score bins
    df['credit_bin'] = pd.cut(df['credit_score'], bins=10, labels=False)
    
    # Interactions with other features
    df['credit_income_interaction'] = df['credit_score'] * df['annual_income']
    df['credit_times_dti'] = df['credit_score'] * df['debt_to_income_ratio']
    df['credit_loan_interaction'] = df['credit_score'] * df['loan_amount']
    
    print(f"âœ“ Created 9 credit score features")
    
    # ========================================
    # 3. INTEREST RATE FEATURES
    # ========================================
    print("\n[3/11]  Creating interest rate features...")
    
    # Interest rate flags and categories
    df['high_interest_flag'] = (df['interest_rate'] > df['interest_rate'].median()).astype(int)
    df['very_high_interest'] = (df['interest_rate'] > df['interest_rate'].quantile(0.75)).astype(int)
    df['low_interest_flag'] = (df['interest_rate'] < df['interest_rate'].quantile(0.25)).astype(int)
    
    # Interest cost calculations
    df['total_interest_cost'] = df['loan_amount'] * df['interest_rate'] / 100
    df['interest_burden'] = df['total_interest_cost'] / (df['annual_income'] + 1)
    df['monthly_interest_cost'] = df['total_interest_cost'] / 12
    
    # Interest rate vs credit score (should be inversely related)
    df['interest_credit_mismatch'] = df['interest_rate'] * (1 - df['credit_score_normalized'])
    df['interest_credit_ratio'] = df['interest_rate'] / (df['credit_score'] / 100)
    
    # Interest rate transformations
    df['interest_rate_squared'] = df['interest_rate'] ** 2
    df['interest_rate_log'] = np.log1p(df['interest_rate'])
    
    print(f"âœ“ Created 10 interest rate features")
    
    # ========================================
    # 4. COMPOSITE RISK SCORES
    # ========================================
    print("\n[4/11]   Creating composite risk scores...")
    
    # Multi-factor risk scores (weighted combinations)
    df['risk_score_v1'] = (
        df['debt_to_income_ratio'] * 0.25 +
        df['loan_to_income_ratio'] * 0.25 +
        df['credit_risk_score'] * 0.30 +
        (df['interest_rate'] / 100) * 0.20
    )
    
    df['risk_score_v2'] = (
        df['payment_to_income_ratio'] * 0.40 +
        df['new_debt_to_income'] * 0.35 +
        df['interest_burden'] * 0.25
    )
    
    df['risk_score_v3'] = (
        df['debt_to_income_ratio'] * 0.30 +
        df['payment_burden_ratio'] * 0.30 +
        df['credit_risk_score'] * 0.40
    )
    
    # Affordability score (higher is better)
    df['affordability_score'] = (
        df['credit_score_normalized'] * 0.40 +
        (1 - df['debt_to_income_ratio']) * 0.30 +
        df['disposable_income_ratio'] * 0.30
    )
    
    # Financial health score
    df['financial_health_score'] = (
        df['affordability_score'] * 0.60 -
        df['risk_score_v1'] * 0.40
    )
    
    print(f"âœ“ Created 5 composite risk scores")
    
    # Continue in next cell...
    return df


def complete_feature_engineering(df):
    """
    Comprehensive feature engineering pipeline for loan prediction
    """
    df = df.copy()
    
    print(" FEATURE ENGINEERING PIPELINE")
    print("="*80)
    print(f"Starting features: {df.shape[1]}")
    
    # 1. FINANCIAL RATIOS
    print("\n[1/11]  Financial ratio features...")
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_payment_estimate'] = (df['loan_amount'] * df['interest_rate']) / 1200
    df['payment_to_income_ratio'] = df['monthly_payment_estimate'] / (df['monthly_income'] + 1)
    df['current_debt_amount'] = df['debt_to_income_ratio'] * df['annual_income']
    df['total_debt_with_loan'] = df['current_debt_amount'] + df['loan_amount']
    df['new_debt_to_income'] = df['total_debt_with_loan'] / (df['annual_income'] + 1)
    df['debt_increase_ratio'] = df['new_debt_to_income'] / (df['debt_to_income_ratio'] + 0.01)
    df['disposable_income'] = df['annual_income'] - df['current_debt_amount']
    df['disposable_income_ratio'] = df['disposable_income'] / (df['annual_income'] + 1)
    df['loan_to_disposable_income'] = df['loan_amount'] / (df['disposable_income'] + 1)
    df['monthly_disposable_income'] = df['disposable_income'] / 12
    df['payment_to_disposable_ratio'] = df['monthly_payment_estimate'] / (df['monthly_disposable_income'] + 1)
    df['annual_payment_burden'] = df['monthly_payment_estimate'] * 12
    df['payment_burden_ratio'] = df['annual_payment_burden'] / (df['annual_income'] + 1)
    print(f"âœ“ Created 15 features")
    
    # 2. CREDIT SCORE FEATURES
    print("[2/11]  Credit score features...")
    df['credit_score_normalized'] = df['credit_score'] / 850
    df['credit_risk_score'] = 1 - df['credit_score_normalized']
    df['credit_score_squared'] = df['credit_score'] ** 2
    df['credit_score_log'] = np.log1p(df['credit_score'])
    df['credit_category'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 800, 850],
                                     labels=['poor', 'fair', 'good', 'very_good', 'excellent']).astype(object)
    df['credit_income_interaction'] = df['credit_score'] * df['annual_income']
    df['credit_times_dti'] = df['credit_score'] * df['debt_to_income_ratio']
    df['credit_loan_interaction'] = df['credit_score'] * df['loan_amount']
    print(f"âœ“ Created 8 features")
    
    # 3. INTEREST RATE FEATURES
    print("[3/11]  Interest rate features...")
    df['high_interest_flag'] = (df['interest_rate'] > df['interest_rate'].median()).astype(int)
    df['very_high_interest'] = (df['interest_rate'] > df['interest_rate'].quantile(0.75)).astype(int)
    df['low_interest_flag'] = (df['interest_rate'] < df['interest_rate'].quantile(0.25)).astype(int)
    df['total_interest_cost'] = df['loan_amount'] * df['interest_rate'] / 100
    df['interest_burden'] = df['total_interest_cost'] / (df['annual_income'] + 1)
    df['interest_credit_mismatch'] = df['interest_rate'] * (1 - df['credit_score_normalized'])
    df['interest_credit_ratio'] = df['interest_rate'] / (df['credit_score'] / 100)
    df['interest_rate_squared'] = df['interest_rate'] ** 2
    print(f"âœ“ Created 8 features")
    
    # 4. RISK SCORES
    print("[4/11]   Composite risk scores...")
    df['risk_score_v1'] = (df['debt_to_income_ratio'] * 0.25 + df['loan_to_income_ratio'] * 0.25 +
                           df['credit_risk_score'] * 0.30 + (df['interest_rate'] / 100) * 0.20)
    df['risk_score_v2'] = (df['payment_to_income_ratio'] * 0.40 + df['new_debt_to_income'] * 0.35 +
                           df['interest_burden'] * 0.25)
    df['affordability_score'] = (df['credit_score_normalized'] * 0.40 + 
                                 (1 - df['debt_to_income_ratio']) * 0.30 +
                                 df['disposable_income_ratio'] * 0.30)
    df['financial_health_score'] = df['affordability_score'] * 0.60 - df['risk_score_v1'] * 0.40
    print(f"   âœ“ Created 4 features")
    
    # 5. LOAN AMOUNT FEATURES
    print("[5/11]  Loan amount features...")
    df['loan_size'] = pd.cut(df['loan_amount'], bins=[0, 10000, 20000, 30000, np.inf],
                              labels=['small', 'medium', 'large', 'very_large']).astype(object)
    df['loan_amount_squared'] = df['loan_amount'] ** 2
    df['loan_amount_log'] = np.log1p(df['loan_amount'])
    df['annual_income_log'] = np.log1p(df['annual_income'])
    df['loan_amount_sqrt'] = np.sqrt(df['loan_amount'])
    print(f"âœ“ Created 5 features")
    
    # 6. BINNING FEATURES
    print("[6/11]  Binned features...")
    df['income_decile'] = pd.qcut(df['annual_income'], q=10, labels=False, duplicates='drop')
    df['credit_decile'] = pd.qcut(df['credit_score'], q=10, labels=False, duplicates='drop')
    df['loan_decile'] = pd.qcut(df['loan_amount'], q=10, labels=False, duplicates='drop')
    df['dti_decile'] = pd.qcut(df['debt_to_income_ratio'], q=10, labels=False, duplicates='drop')
    df['interest_decile'] = pd.qcut(df['interest_rate'], q=10, labels=False, duplicates='drop')
    print(f"âœ“ Created 5 features")
    
    # 7. INTERACTION FEATURES
    print("[7/11]  Interaction features...")
    df['income_x_credit'] = df['annual_income'] * df['credit_score']
    df['dti_x_interest'] = df['debt_to_income_ratio'] * df['interest_rate']
    df['loan_x_interest'] = df['loan_amount'] * df['interest_rate']
    df['income_x_dti'] = df['annual_income'] * df['debt_to_income_ratio']
    df['income_credit_loan'] = df['annual_income'] * df['credit_score'] * df['loan_amount']
    df['dti_interest_credit'] = df['debt_to_income_ratio'] * df['interest_rate'] * df['credit_score']
    print(f"âœ“ Created 6 features")
    
    # 8. GRADE FEATURES
    print("[8/11]  Grade/subgrade features...")
    df['grade'] = df['grade_subgrade'].str[0]
    df['subgrade_num'] = df['grade_subgrade'].str[1:].astype(int)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_numeric'] = df['grade'].map(grade_map)
    df['full_grade_score'] = df['grade_numeric'] * 10 + df['subgrade_num']
    df['grade_credit_ratio'] = df['full_grade_score'] / (df['credit_score'] / 100)
    print(f"âœ“ Created 5 features")
    
    # 9. STATISTICAL AGGREGATIONS
    print("[9/11]  Statistical aggregations...")
    df['mean_financial_metrics'] = df[['debt_to_income_ratio', 'loan_to_income_ratio', 
                                        'payment_to_income_ratio']].mean(axis=1)
    df['max_financial_burden'] = df[['debt_to_income_ratio', 'loan_to_income_ratio', 
                                      'payment_to_income_ratio']].max(axis=1)
    df['min_financial_burden'] = df[['debt_to_income_ratio', 'loan_to_income_ratio', 
                                      'payment_to_income_ratio']].min(axis=1)
    df['std_financial_metrics'] = df[['debt_to_income_ratio', 'loan_to_income_ratio', 
                                       'payment_to_income_ratio']].std(axis=1)
    print(f"âœ“ Created 4 features")
    
    # 10. CATEGORICAL COMBINATIONS
    print("[10/11]  Categorical combinations...")
    df['gender_marital'] = df['gender'] + '_' + df['marital_status']
    df['education_employment'] = df['education_level'] + '_' + df['employment_status']
    df['gender_education'] = df['gender'] + '_' + df['education_level']
    df['marital_employment'] = df['marital_status'] + '_' + df['employment_status']
    df['purpose_grade'] = df['loan_purpose'] + '_' + df['grade']
    df['employment_purpose'] = df['employment_status'] + '_' + df['loan_purpose']
    print(f"âœ“ Created 6 features")
    
    # 11. ANOMALY FLAGS
    print("[11/11]  Anomaly detection flags...")
    df['extreme_dti'] = (df['debt_to_income_ratio'] > df['debt_to_income_ratio'].quantile(0.90)).astype(int)
    df['low_income'] = (df['annual_income'] < df['annual_income'].quantile(0.25)).astype(int)
    df['large_loan'] = (df['loan_amount'] > df['loan_amount'].quantile(0.75)).astype(int)
    df['risky_combo_1'] = ((df['debt_to_income_ratio'] > 0.4) & (df['credit_score'] < 650)).astype(int)
    df['risky_combo_2'] = ((df['loan_to_income_ratio'] > 0.5) & (df['interest_rate'] > 15)).astype(int)
    df['safe_combo'] = ((df['credit_score'] > 750) & (df['debt_to_income_ratio'] < 0.3)).astype(int)
    df['high_risk_all'] = (df['extreme_dti'] & df['risky_combo_1']).astype(int)
    print(f"âœ“ Created 7 features")
    
    print("\n" + "="*80)
    print(f" Feature Engineering Complete!")
    print(f"   Final features: {df.shape[1]}")
    print(f"   New features: {df.shape[1] - 13}")
    print("="*80)
    
    return df

# Apply feature engineering
train_df = complete_feature_engineering(train_df)
test_df = complete_feature_engineering(test_df)


from sklearn.model_selection import train_test_split



train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


train_df.columns


input_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount',
       'interest_rate', 'gender', 'marital_status', 'education_level',
       'employment_status', 'loan_purpose', 'grade_subgrade',
       'loan_to_income_ratio', 'monthly_income', 'monthly_payment_estimate',
       'payment_to_income_ratio', 'current_debt_amount',
       'total_debt_with_loan', 'new_debt_to_income', 'debt_increase_ratio',
       'disposable_income', 'disposable_income_ratio',
       'loan_to_disposable_income', 'monthly_disposable_income',
       'payment_to_disposable_ratio', 'annual_payment_burden',
       'payment_burden_ratio', 'credit_score_normalized', 'credit_risk_score',
       'credit_score_squared', 'credit_score_log', 'credit_category',
       'credit_income_interaction', 'credit_times_dti',
       'credit_loan_interaction', 'high_interest_flag', 'very_high_interest',
       'low_interest_flag', 'total_interest_cost', 'interest_burden',
       'interest_credit_mismatch', 'interest_credit_ratio',
       'interest_rate_squared', 'risk_score_v1', 'risk_score_v2',
       'affordability_score', 'financial_health_score', 'loan_size',
       'loan_amount_squared', 'loan_amount_log', 'annual_income_log',
       'loan_amount_sqrt', 'income_decile', 'credit_decile', 'loan_decile',
       'dti_decile', 'interest_decile', 'income_x_credit', 'dti_x_interest',
       'loan_x_interest', 'income_x_dti', 'income_credit_loan',
       'dti_interest_credit', 'grade', 'subgrade_num', 'grade_numeric',
       'full_grade_score', 'grade_credit_ratio', 'mean_financial_metrics',
       'max_financial_burden', 'min_financial_burden', 'std_financial_metrics',
       'gender_marital', 'education_employment', 'gender_education',
       'marital_employment', 'purpose_grade', 'employment_purpose',
       'extreme_dti', 'low_income', 'large_loan', 'risky_combo_1',
       'risky_combo_2', 'safe_combo', 'high_risk_all']
input_cols


target_col = 'loan_paid_back'
target_col


# Training dataset inputs and target

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy() 

# Validation dataset inputs and target

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy() 

# Testing dataset inputs and target

test_inputs = test_df[input_cols].copy()


numerical_cols = [var for var in train_inputs.columns if train_inputs[var].dtype != 'O']

numerical_cols


category_cols = [var for var in train_inputs.columns if train_inputs[var].dtype == 'O']


category_cols


train_inputs[category_cols].isnull().sum()


train_inputs[numerical_cols].isnull().sum()


train_inputs.head()


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[category_cols])
encoded_cols = list(encoder.get_feature_names_out(category_cols))


train_inputs[encoded_cols] = encoder.transform(train_inputs[category_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[category_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[category_cols])


train_inputs[encoded_cols]


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaler.fit(train_inputs[numerical_cols])


train_inputs[numerical_cols] = scaler.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = scaler.transform(val_inputs[numerical_cols])
test_inputs[numerical_cols] = scaler.transform(test_inputs[numerical_cols])


X_train = train_inputs[numerical_cols + encoded_cols]
X_val = val_inputs[numerical_cols + encoded_cols]
X_test = test_inputs[numerical_cols + encoded_cols]


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report


import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

xgb_params = {
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_estimators': 10000,
    'learning_rate': 0.05,
    'max_depth': 3,
    'max_leaves': 1000,
    'min_child_weight': 11,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.8,
    'colsample_bynode': 0.8,
    'gamma': 0.1,
    'reg_alpha': 5.0,
    'reg_lambda': 10.0,
    'scale_pos_weight': 0.55,
}

print("Training XGBoost with K-Fold Cross Validation")
print("Parameters:")
for key, value in xgb_params.items():
    print(f"  {key}: {value}")

# Initialize K-Fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
models = []

# K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, train_targets)):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1}")
    print(f"{'='*50}")
    
    # Split data for this fold
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = train_targets.iloc[train_idx], train_targets.iloc[val_idx]
    
    # Initialize and train model
    model = xgb.XGBClassifier(**xgb_params)
    
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=1000,
        early_stopping_rounds=200
    )
    
    # Predictions
    val_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_fold_val, val_pred_proba)
    auc_scores.append(auc)
    models.append(model)
    
    print(f"Fold {fold + 1} AUC: {auc:.4f}")

# Print overall results
print(f"\n{'='*50}")
print("K-Fold Cross Validation Results")
print(f"{'='*50}")
print(f"Mean AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
print(f"Individual Fold AUCs: {[f'{score:.4f}' for score in auc_scores]}")


# Test the model on validation/Train data
train_accuracy = model.score(X_train, train_targets)
val_accuracy = model.score(X_val, val_targets)  

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Overfitting Gap: {train_accuracy - val_accuracy:.4f}")


train_pred = model.predict(X_train)
train_pred[:5]


val_pred = model.predict(X_val)
val_pred[:5]


val_targets[:5]


# view accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(val_pred, val_targets)
print('XGBClassifier Model accuracy score: {0:0.4f}'.format(accuracy_score(val_targets, val_pred)))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Get predictions
val_predictions = model.predict(X_val)
val_probabilities = model.predict_proba(X_val)[:, 1]

print("=== BASIC PERFORMANCE METRICS ===")
print(f"Accuracy:  {accuracy_score(val_targets, val_predictions):.4f}")
print(f"Precision: {precision_score(val_targets, val_predictions):.4f}")
print(f"Recall:    {recall_score(val_targets, val_predictions):.4f}")
print(f"F1-Score:  {f1_score(val_targets, val_predictions):.4f}")


# Get probability predictions instead of class predictions
val_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1

# Calculate ROC-AUC
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(val_targets, val_pred_proba)
print(f"XGBClassifier Model ROC-AUC score: {auc_score:.4f}")


# Check class distribution
print("\nðŸ“Š Class Distribution Analysis:")
print(f"Training set - Class 0: {np.sum(train_targets == 0):,} | Class 1: {np.sum(train_targets == 1):,}")
print(f"Validation set - Class 0: {np.sum(val_targets == 0):,} | Class 1: {np.sum(val_targets == 1):,}")

# Check if we have imbalance
class_ratio = np.sum(val_targets == 0) / np.sum(val_targets == 1)
print(f"Class ratio (0:1): {class_ratio:.2f}:1")


# Let's see the prediction distribution to understand the model better
plt.figure(figsize=(12, 4))

# Plot 1: Prediction distribution
plt.subplot(1, 2, 1)
plt.hist(val_pred_proba, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predictions')

# Plot 2: ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(val_targets, val_pred_proba)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {auc_score:.4f})')
plt.tight_layout()
plt.show()


from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(val_targets, val_pred_proba, n_bins=10)
plt.figure(figsize=(12, 4))
plt.plot(prob_pred, prob_true, marker='o')
plt.xlabel("Mean predicted probability")
plt.ylabel("True probability")
plt.title("Calibration Curve")
plt.show()


importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.title("Feature Importance")
sns.barplot(data=importance_df.head(10),x='importance', y='feature',saturation=0.75)


test_preds = model.predict(X_test)


submission_df = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
submission_df


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

lgb_params = {
    'device': 'gpu',  # Use GPU
    'metric': 'auc',
    'objective': 'binary',
    'random_state': 42,
    'n_estimators': 10000,
    'learning_rate': 0.05,
    'max_depth': 3,           # Equivalent to XGBoost's max_depth
    'num_leaves': 27,       # Equivalent to XGBoost's max_leaves
    'min_child_weight': 11,   # Similar to XGBoost's min_child_weight
    'subsample': 0.9,         # Row sampling
    'colsample_bytree': 0.9,  # Feature sampling per tree
    'subsample_freq': 1,      # Frequency for subsample
    'reg_alpha': 5.0,         # L1 regularization
    'reg_lambda': 10.0,       # L2 regularization
    'scale_pos_weight': 0.65,
    'verbosity': -1,          # Less verbose output
}

print("Training LightGBM with K-Fold Cross Validation")
print("Parameters:")
for key, value in lgb_params.items():
    print(f"  {key}: {value}")

# Initialize K-Fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
models = []
best_iterations = []

# K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, train_targets)):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1}")
    print(f"{'='*50}")
    
    # Split data for this fold
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = train_targets.iloc[train_idx], train_targets.iloc[val_idx]
    
    # Initialize and train model
    model = lgb.LGBMClassifier(**lgb_params)
    
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=True),
            lgb.log_evaluation(500)  # Print every 500 iterations
        ]
    )
    
    # Predictions
    val_pred_proba = model.predict_proba(X_fold_val)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_fold_val, val_pred_proba)
    auc_scores.append(auc)
    models.append(model)
    best_iterations.append(model.best_iteration_)
    
    print(f"Fold {fold + 1} AUC: {auc:.4f}")
    print(f"Fold {fold + 1} Best Iteration: {model.best_iteration_}")

# Print overall results
print(f"\n{'='*50}")
print("K-Fold Cross Validation Results")
print(f"{'='*50}")
print(f"Mean AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
print(f"Individual Fold AUCs: {[f'{score:.4f}' for score in auc_scores]}")
print(f"Best Iterations: {best_iterations}")
print(f"Average Best Iteration: {np.mean(best_iterations):.0f}")



# Get probability predictions instead of class predictions
val_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1

# Calculate ROC-AUC
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(val_targets, val_pred_proba)
print(f"LGBMClassifier Model ROC-AUC score: {auc_score:.4f}")


# Let's see the prediction distribution to understand the model better
plt.figure(figsize=(12, 4))

# Plot 1: Prediction distribution
plt.subplot(1, 2, 1)
plt.hist(val_pred_proba, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predictions')

# Plot 2: ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(val_targets, val_pred_proba)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {auc_score:.4f})')
plt.tight_layout()
plt.show()


test_preds = model.predict(X_test)


submission_df['loan_paid_back'] = test_preds

# Verify the update
print("Updated submission preview:")
print(submission_df.head())
print(f"\nSubmission shape: {submission_df.shape}")

# Save the updated submission
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




