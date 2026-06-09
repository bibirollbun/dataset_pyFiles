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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
train_df.info()


train_df.head(5)


train_df.describe()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 120

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print('Train shape:', train.shape)
print(train.dtypes)
print("\nTarget distribution:")
print(train['loan_paid_back'].value_counts(normalize=True).rename('proportion'))

# --- Helper functions ---
def create_income_bins(df, col='annual_income'):
    maxv = df[col].max()
    bins = [0, 25000, 50000, 75000, 100000, 150000, 200000, maxv + 1]
    labels = ['<25k', '25-50k', '50-75k', '75-100k', '100-150k', '150-200k', '>200k']
    return pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)

def create_numeric_bins(series, bins, labels=None):
    bins = np.unique(np.sort(bins))
    if len(bins) < 2:
        bins = [series.min(), series.max() + 0.01]
    return pd.cut(series, bins=bins, labels=labels[:len(bins)-1] if labels else None, include_lowest=True)

# --- Define bins safely ---
credit_bins = [0, 580, 670, 740, 800, 850, 900]
credit_labels = ['Very low','Low','Fair','Good','Very good','Excellent']

dti_max = train['debt_to_income_ratio'].max()
dti_bins = np.linspace(0, dti_max + 0.001, 7)
dti_labels = ['<20%','20-35%','35-50%','50-75%','75-100%','>100%']

loan_bins = train['loan_amount'].quantile([0, 0.25, 0.5, 0.75, 0.9, 1.0]).values
loan_labels = ['Q1','Q2','Q3','Q4','Top10%']

ir_min, ir_max = train['interest_rate'].min(), train['interest_rate'].max()
ir_bins = [ir_min-0.01, 5, 10, 15, 20, 30, ir_max+0.01]
ir_bins = np.unique(np.sort(ir_bins))
ir_labels = ['<=5','5-10','10-15','15-20','20-30','>30']

# --- Create categorical bin columns ---
train['income_group'] = create_income_bins(train, 'annual_income')
train['credit_group'] = create_numeric_bins(train['credit_score'], credit_bins, credit_labels)
train['dti_group'] = create_numeric_bins(train['debt_to_income_ratio'], dti_bins, dti_labels)
train['loan_amount_group'] = create_numeric_bins(train['loan_amount'], loan_bins, loan_labels)
train['interest_rate_group'] = create_numeric_bins(train['interest_rate'], ir_bins, ir_labels)



# --- Analysis Section ---
def plot_avg_payback(df, feature, figsize=(7,4)):
    grouped = df.groupby(feature)['loan_paid_back'].mean().sort_index()
    plt.figure(figsize=figsize)
    grouped.plot(kind='bar', color='teal', edgecolor='black')
    plt.title(f'Average Loan Payback by {feature}')
    plt.ylabel('Average Payback Probability')
    plt.xlabel(feature)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()
    return grouped

# 1ï¸�âƒ£ Income
print("\nPayback by Income Group:")
print(plot_avg_payback(train, 'income_group'))

# 2ï¸�âƒ£ Debt-to-Income Ratio
print("\nPayback by DTI Group:")
print(plot_avg_payback(train, 'dti_group'))

# 3ï¸�âƒ£ Credit Score
print("\nPayback by Credit Score Group:")
print(plot_avg_payback(train, 'credit_group'))

# 4ï¸�âƒ£ Loan Amount
print("\nPayback by Loan Amount Group:")
print(plot_avg_payback(train, 'loan_amount_group'))

# 5ï¸�âƒ£ Interest Rate
print("\nPayback by Interest Rate Group:")
print(plot_avg_payback(train, 'interest_rate_group'))

# 6ï¸�âƒ£ Education
print("\nPayback by Education Level:")
print(plot_avg_payback(train, 'education_level'))

# 7ï¸�âƒ£ Employment Status
print("\nPayback by Employment Status:")
print(plot_avg_payback(train, 'employment_status'))

# --- Bonus Deep Combinations ---
pivot_income_credit = train.pivot_table(values='loan_paid_back', index='income_group', columns='credit_group', aggfunc='mean')
plt.figure(figsize=(10,6))
sns.heatmap(pivot_income_credit, annot=True, fmt='.2f', cmap='YlGnBu')
plt.title('Payback Probability: Income vs Credit Score')
plt.show()

pivot_edu_emp = train.pivot_table(values='loan_paid_back', index='education_level', columns='employment_status', aggfunc='mean')
plt.figure(figsize=(10,6))
sns.heatmap(pivot_edu_emp, annot=True, fmt='.2f', cmap='magma')
plt.title('Payback Probability: Education vs Employment')
plt.show()

pivot_dti_ir = train.pivot_table(values='loan_paid_back', index='dti_group', columns='interest_rate_group', aggfunc='mean')
plt.figure(figsize=(10,6))
sns.heatmap(pivot_dti_ir, annot=True, fmt='.2f', cmap='viridis')
plt.title('Payback Probability: DTI vs Interest Rate')
plt.show()

print("\nâœ… Deep EDA complete! You can now visually see how each feature and combination affects loan repayment.")



import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')

# Quick check
print(train_df[['gender', 'marital_status', 'loan_paid_back']].head())

# 1ï¸�âƒ£ Average payback rate by gender
gender_payback = train_df.groupby('gender')['loan_paid_back'].mean().sort_values(ascending=False)
print("\nAverage Payback Rate by Gender:")
print(gender_payback)

# 2ï¸�âƒ£ Average payback rate by marital status
marital_payback = train_df.groupby('marital_status')['loan_paid_back'].mean().sort_values(ascending=False)
print("\nAverage Payback Rate by Marital Status:")
print(marital_payback)

# 3ï¸�âƒ£ Combined analysis: gender + marital status
combined_payback = train_df.groupby(['gender', 'marital_status'])['loan_paid_back'].mean().unstack()
print("\nAverage Payback Rate by Gender and Marital Status:")
print(combined_payback)

# 4ï¸�âƒ£ Visualization: Gender-wise comparison
plt.figure(figsize=(6,4))
gender_payback.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Average Loan Payback Rate by Gender')
plt.ylabel('Average Payback Probability')
plt.xticks(rotation=0)
plt.show()

# 5ï¸�âƒ£ Visualization: Marital status comparison
plt.figure(figsize=(6,4))
marital_payback.plot(kind='bar', color=['lightgreen', 'orange'])
plt.title('Average Loan Payback Rate by Marital Status')
plt.ylabel('Average Payback Probability')
plt.xticks(rotation=0)
plt.show()

# 6ï¸�âƒ£ Visualization: Combined comparison
plt.figure(figsize=(6,4))
combined_payback.plot(kind='bar')
plt.title('Average Loan Payback Rate by Gender and Marital Status')
plt.ylabel('Average Payback Probability')
plt.xticks(rotation=0)
plt.legend(title='Marital Status')
plt.show()



# ===============================================
# ğŸ”� Phase 2: Model-Driven EDA + Feature Engineering
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# --- Load data ---
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print("âœ… Data Loaded")
print(train.shape)
print(train['loan_paid_back'].value_counts(normalize=True))

# --- Base numeric columns ---
num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

# ===============================================
# 1ï¸�âƒ£ Correlation Heatmap
# ===============================================
plt.figure(figsize=(8,6))
corr = train[num_cols + ['loan_paid_back']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# ===============================================
# 2ï¸�âƒ£ Ratio Feature Engineering
# ===============================================
train['loan_to_income_ratio'] = train['loan_amount'] / (train['annual_income'] + 1e-6)
train['credit_to_loan_ratio'] = train['credit_score'] / (train['loan_amount'] + 1e-6)
train['dti_x_interest'] = train['debt_to_income_ratio'] * train['interest_rate']

# Distribution
sns.pairplot(train[['loan_to_income_ratio', 'credit_to_loan_ratio', 'dti_x_interest', 'loan_paid_back']], 
             diag_kind='kde', hue='loan_paid_back', plot_kws={'alpha':0.3})
plt.suptitle("Derived Ratio Features vs Payback", y=1.02)
plt.show()

# ===============================================
# 3ï¸�âƒ£ Grade/Subgrade Analysis
# ===============================================
plt.figure(figsize=(12,4))
sns.barplot(data=train, x='grade_subgrade', y='loan_paid_back', 
            order=sorted(train['grade_subgrade'].unique()))
plt.xticks(rotation=90)
plt.title("Average Payback Rate by Grade/Subgrade")
plt.show()

# ===============================================
# 4ï¸�âƒ£ Loan Purpose Analysis
# ===============================================
plt.figure(figsize=(10,4))
sns.barplot(data=train, x='loan_purpose', y='loan_paid_back')
plt.xticks(rotation=45)
plt.title("Payback Rate by Loan Purpose")
plt.show()

# ===============================================
# 5ï¸�âƒ£ Outlier & Distribution Checks
# ===============================================
train[num_cols].hist(bins=40, figsize=(10,8))
plt.suptitle("Distribution of Numeric Features", y=1.02)
plt.show()

for col in num_cols:
    sns.boxplot(x=train[col])
    plt.title(f"Outlier Check: {col}")
    plt.show()

# ===============================================
# 6ï¸�âƒ£ Log Transform
# ===============================================
for col in ['annual_income','loan_amount']:
    train[f'log_{col}'] = np.log1p(train[col])

# ===============================================
# 7ï¸�âƒ£ Target Encoding
# ===============================================
for col in ['loan_purpose','grade_subgrade']:
    mapping = train.groupby(col)['loan_paid_back'].mean()
    train[col + '_enc'] = train[col].map(mapping)

# ===============================================
# 8ï¸�âƒ£ Encode categorical columns
# ===============================================
cat_cols = ['gender','marital_status','education_level','employment_status']
le = LabelEncoder()
for c in cat_cols:
    train[c] = le.fit_transform(train[c])

# ===============================================
# 9ï¸�âƒ£ Correlation Heatmap (with new features)
# ===============================================
new_cols = num_cols + ['loan_to_income_ratio','credit_to_loan_ratio','dti_x_interest','log_annual_income','log_loan_amount']
plt.figure(figsize=(10,6))
sns.heatmap(train[new_cols + ['loan_paid_back']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation with New Derived Features")
plt.show()

# ===============================================
# ğŸ”Ÿ Baseline XGBoost Model
# ===============================================
features = num_cols + ['loan_to_income_ratio','credit_to_loan_ratio','dti_x_interest',
                       'log_annual_income','log_loan_amount','loan_purpose_enc','grade_subgrade_enc'] + cat_cols

X = train[features]
y = train['loan_paid_back']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc'
)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:,1]

auc = roc_auc_score(y_val, y_pred)
print(f"\nğŸŒŸ Baseline XGBoost AUC: {auc:.4f}")

# ===============================================
# 11ï¸�âƒ£ Feature Importance
# ===============================================
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=feat_imp.head(15), x='Importance', y='Feature')
plt.title("Top 15 Feature Importances (XGBoost)")
plt.show()

# ===============================================
# 12ï¸�âƒ£ SHAP Summary
# ===============================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val, max_display=15)



# ============================================================================
# EDA-OPTIMIZED ULTIMATE SOLUTION - TARGETING 0.95+ ROC AUC
# Based on Deep EDA Insights: Employment Status + DTI + Credit Score Dominance
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("ğŸš€ EDA-OPTIMIZED ULTIMATE PIPELINE")
print("="*80)
print("Key Insights Applied:")
print("  âœ… Employment Status = Most Important (0.83)")
print("  âœ… DTI Interactions = Critical")
print("  âœ… Credit Score Groups = Strong Signal")
print("  âœ… Income Sweet Spot = 75-100k")
print("  âœ… Risk-Based Interest Rate Encoding")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATASETS
# ============================================================================
print("\nğŸ“‚ Loading datasets...")

train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

# Try to load original dataset for additional training data
try:
    original = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
    print(f"âœ… Original dataset loaded: {original.shape}")
    
    # Align column names if needed
    if 'loan_paid_back' in original.columns:
        # Combine datasets
        common_cols = [c for c in train.columns if c in original.columns]
        train = pd.concat([train[common_cols], original[common_cols]], ignore_index=True)
        print(f"âœ… Combined training size: {train.shape}")
except:
    print("âš ï¸� Original dataset not found - using competition data only")

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================================
# STEP 2: EDA-DRIVEN FEATURE ENGINEERING
# ============================================================================
print("\nğŸ”§ Creating EDA-optimized features...")

def create_eda_features(df):
    """Feature engineering based on deep EDA insights"""
    df = df.copy()
    
    # ==================================================
    # 1. EMPLOYMENT STATUS FEATURES (Most Important!)
    # ==================================================
    # Based on EDA: Retired=0.997, Employed=0.894, Student=0.264, Unemployed=0.078
    df['is_retired'] = (df['employment_status'] == 'Retired').astype(int)
    df['is_student_or_unemployed'] = df['employment_status'].isin(['Student', 'Unemployed']).astype(int)
    df['is_employed_stable'] = df['employment_status'].isin(['Employed', 'Self-employed']).astype(int)
    
    # Employment risk score
    employment_risk = {
        'Retired': 1,
        'Employed': 2,
        'Self-employed': 2,
        'Student': 5,
        'Unemployed': 5
    }
    df['employment_risk_score'] = df['employment_status'].map(employment_risk)
    
    # ==================================================
    # 2. DTI FEATURES (Strong -0.34 correlation)
    # ==================================================
    # Based on EDA: <20% = 88%, >100% = 11%
    df['dti_very_low'] = (df['debt_to_income_ratio'] < 0.20).astype(int)
    df['dti_low'] = ((df['debt_to_income_ratio'] >= 0.20) & (df['debt_to_income_ratio'] < 0.35)).astype(int)
    df['dti_high'] = (df['debt_to_income_ratio'] >= 0.50).astype(int)
    df['dti_critical'] = (df['debt_to_income_ratio'] >= 0.75).astype(int)
    
    # DTI risk bands
    df['dti_risk_band'] = pd.cut(df['debt_to_income_ratio'],
                                  bins=[0, 0.20, 0.35, 0.50, 0.75, 1.0],
                                  labels=[1, 2, 3, 4, 5]).astype(float)
    
    # ==================================================
    # 3. CREDIT SCORE FEATURES (Strong +0.23 correlation)
    # ==================================================
    # Based on EDA: Very Good=95%, Very Low=62%
    df['credit_excellent'] = (df['credit_score'] >= 750).astype(int)
    df['credit_good'] = ((df['credit_score'] >= 680) & (df['credit_score'] < 750)).astype(int)
    df['credit_poor'] = (df['credit_score'] < 600).astype(int)
    
    # Credit score groups (matching EDA bins)
    df['credit_group'] = pd.cut(df['credit_score'],
                                bins=[0, 580, 630, 680, 730, 900],
                                labels=['very_low', 'low', 'fair', 'good', 'very_good'])
    
    # Credit percentile
    df['credit_percentile'] = df['credit_score'].rank(pct=True)
    
    # ==================================================
    # 4. INCOME SWEET SPOT FEATURES (75-100k = 82.4%)
    # ==================================================
    df['income_sweet_spot'] = ((df['annual_income'] >= 75000) & 
                               (df['annual_income'] <= 100000)).astype(int)
    df['income_very_high_risk'] = (df['annual_income'] > 200000).astype(int)
    
    # Income groups (matching EDA)
    df['income_group'] = pd.cut(df['annual_income'],
                                bins=[0, 25000, 50000, 75000, 100000, 150000, 200000, 500000],
                                labels=['<25k', '25-50k', '50-75k', '75-100k', '100-150k', '150-200k', '>200k'])
    
    # ==================================================
    # 5. INTEREST RATE AS RISK PROXY (Strong -0.13 correlation)
    # ==================================================
    # Based on EDA: <=5% = 94%, 20-30% = 55%
    df['interest_low_risk'] = (df['interest_rate'] <= 10).astype(int)
    df['interest_high_risk'] = (df['interest_rate'] > 15).astype(int)
    df['interest_critical'] = (df['interest_rate'] > 20).astype(int)
    
    # Interest rate groups
    df['interest_group'] = pd.cut(df['interest_rate'],
                                  bins=[0, 5, 10, 15, 20, 30],
                                  labels=['<=5', '5-10', '10-15', '15-20', '20-30'])
    
    # ==================================================
    # 6. CRITICAL INTERACTION FEATURES (From EDA)
    # ==================================================
    # DTI Ã— Interest (shown in heatmaps as critical)
    df['dti_x_interest'] = df['debt_to_income_ratio'] * df['interest_rate']
    df['dti_x_interest_squared'] = df['dti_x_interest'] ** 2
    
    # Credit Ã— Income interaction (high credit + low income still risky)
    df['credit_income_interaction'] = df['credit_score'] * df['annual_income'] / 100000
    
    # Employment Ã— DTI (critical combination)
    df['employment_dti_risk'] = df['employment_risk_score'] * df['dti_risk_band']
    
    # Credit Ã— DTI (inverse relationship)
    df['credit_debt_stress'] = df['debt_to_income_ratio'] * (850 - df['credit_score']) / 100
    
    # ==================================================
    # 7. LOAN BURDEN CALCULATIONS
    # ==================================================
    df['monthly_income'] = df['annual_income'] / 12
    df['debt_amount'] = df['annual_income'] * df['debt_to_income_ratio']
    df['total_debt_with_loan'] = df['debt_amount'] + df['loan_amount']
    df['new_dti'] = df['total_debt_with_loan'] / (df['annual_income'] + 1)
    
    # Payment calculations
    df['estimated_monthly_payment'] = df['loan_amount'] / 36  # Assume 3-year term
    df['payment_to_income'] = df['estimated_monthly_payment'] / (df['monthly_income'] + 1)
    
    # Interest burden
    df['total_interest'] = df['loan_amount'] * df['interest_rate'] / 100
    df['total_loan_cost'] = df['loan_amount'] + df['total_interest']
    
    # Loan size relative to income
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    df['loan_to_income_pct'] = df['loan_to_income_ratio'] * 100
    
    # ==================================================
    # 8. ADVANCED RATIOS
    # ==================================================
    df['income_to_loan'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_to_loan'] = df['credit_score'] / (df['loan_amount'] / 1000 + 1)
    
    # ==================================================
    # 9. POLYNOMIAL & LOG FEATURES
    # ==================================================
    df['credit_squared'] = df['credit_score'] ** 2
    df['dti_squared'] = df['debt_to_income_ratio'] ** 2
    df['interest_squared'] = df['interest_rate'] ** 2
    
    df['log_income'] = np.log1p(df['annual_income'])
    df['log_loan'] = np.log1p(df['loan_amount'])
    df['sqrt_credit'] = np.sqrt(df['credit_score'])
    
    # ==================================================
    # 10. COMPOSITE RISK SCORES
    # ==================================================
    # Triple risk score (Employment + DTI + Credit)
    df['triple_risk_score'] = (
        df['employment_risk_score'] * 3 +
        df['dti_risk_band'] * 2 +
        (5 - df['credit_score'] / 170)  # Normalize credit to 1-5 scale
    )
    
    # Financial health score
    df['financial_health'] = (
        df['credit_percentile'] * 0.4 +
        (1 - df['debt_to_income_ratio']) * 0.3 +
        (df['income_sweet_spot'] * 0.3)
    )
    
    return df

# Apply feature engineering
train = create_eda_features(train)
test = create_eda_features(test)

print(f"âœ… Features created. Train shape: {train.shape}")

# ==================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# ==================================================
print("\nğŸ�·ï¸� Encoding categorical features...")

categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status',
                   'loan_purpose', 'grade_subgrade', 'credit_group', 'income_group',
                   'interest_group']

label_encoders = {}
for col in categorical_cols:
    if col in train.columns:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        label_encoders[col] = le

print(f"âœ… Encoded {len(categorical_cols)} categorical features")

# ==================================================
# STEP 4: PREPARE DATA
# ==================================================
print("\nğŸ“Š Preparing training data...")

target = 'loan_paid_back'
drop_cols = ['id', target]
features = [c for c in train.columns if c not in drop_cols]

X = train[features].fillna(0)
y = train[target]
X_test = test[features].fillna(0)

print(f"âœ… Training on {len(features)} features")
print(f"âœ… Training samples: {len(X):,}")

# ==================================================
# STEP 5: MULTI-MODEL ENSEMBLE WITH OPTIMIZED PARAMETERS
# ==================================================
print("\nğŸ¤– Training 3-Model Ensemble with 5-Fold CV...")
print("="*80)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Storage for out-of-fold predictions
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

# Storage for test predictions
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

# Model parameters (tuned for this specific problem)
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 9,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_gain_to_split': 0.01,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 9,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'gamma': 0.1,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

cat_params = {
    'iterations': 3000,
    'learning_rate': 0.02,
    'depth': 9,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'eval_metric': 'AUC',
    'verbose': False,
    'task_type': 'CPU',
    'border_count': 254
}

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n{'='*80}")
    print(f"ğŸ“Œ FOLD {fold}/{n_splits}")
    print(f"{'='*80}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # ==================
    # MODEL 1: LightGBM
    # ==================
    print("\nğŸ”¹ Training LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits
    lgb_score = roc_auc_score(y_val, oof_lgb[val_idx])
    print(f"   âœ… LightGBM AUC: {lgb_score:.6f}")
    
    # ==================
    # MODEL 2: XGBoost
    # ==================
    print("\nğŸ”¹ Training XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(X_val))
    test_xgb += xgb_model.predict(xgb.DMatrix(X_test)) / n_splits
    xgb_score = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"   âœ… XGBoost AUC: {xgb_score:.6f}")
    
    # ==================
    # MODEL 3: CatBoost
    # ==================
    print("\nğŸ”¹ Training CatBoost...")
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )
    
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(X_test)[:, 1] / n_splits
    cat_score = roc_auc_score(y_val, oof_cat[val_idx])
    print(f"   âœ… CatBoost AUC: {cat_score:.6f}")
    
    # Fold summary
    fold_avg = (lgb_score + xgb_score + cat_score) / 3
    fold_scores.append([lgb_score, xgb_score, cat_score, fold_avg])
    print(f"\n   ğŸ“Š Fold {fold} Average: {fold_avg:.6f}")

# ==================================================
# STEP 6: STACKING META-MODEL
# ==================================================
print("\n" + "="*80)
print("ğŸ�¯ LEVEL 2: STACKING META-MODEL")
print("="*80)

# Create meta features
meta_train = np.column_stack([oof_lgb, oof_xgb, oof_cat])
meta_test = np.column_stack([test_lgb, test_xgb, test_cat])

# Train meta-model (Logistic Regression with regularization)
meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
meta_model.fit(meta_train, y)

# Final stacked predictions
stacked_pred = meta_model.predict_proba(meta_test)[:, 1]

# Calculate all scores
lgb_cv = roc_auc_score(y, oof_lgb)
xgb_cv = roc_auc_score(y, oof_xgb)
cat_cv = roc_auc_score(y, oof_cat)
avg_cv = roc_auc_score(y, (oof_lgb + oof_xgb + oof_cat) / 3)
stacked_oof = meta_model.predict_proba(meta_train)[:, 1]
stacked_cv = roc_auc_score(y, stacked_oof)

print("\nğŸ“Š FINAL CROSS-VALIDATION SCORES:")
print("="*80)
print(f"LightGBM (solo):     {lgb_cv:.6f}")
print(f"XGBoost (solo):      {xgb_cv:.6f}")
print(f"CatBoost (solo):     {cat_cv:.6f}")
print(f"Simple Average:      {avg_cv:.6f}")
print(f"STACKED (Level 2):   {stacked_cv:.6f} â­�")
print("="*80)

print("\nğŸ“ˆ Per-Fold Breakdown:")
print("-" * 80)
print(f"{'Fold':<6} {'LightGBM':<12} {'XGBoost':<12} {'CatBoost':<12} {'Average':<12}")
print("-" * 80)
for i, scores in enumerate(fold_scores, 1):
    print(f"{i:<6} {scores[0]:<12.6f} {scores[1]:<12.6f} {scores[2]:<12.6f} {scores[3]:<12.6f}")
print("-" * 80)

# ==================================================
# STEP 7: CREATE SUBMISSION
# ==================================================
print("\nğŸ“¤ Creating submission file...")

final_submission = pd.DataFrame({
    'id': test['id'],
    'loan_paid_back': stacked_pred
})

final_submission.to_csv('submission.csv', index=False)

print("âœ… Submission saved: submission.csv")
print(f"\nPrediction statistics:")
print(final_submission['loan_paid_back'].describe())




