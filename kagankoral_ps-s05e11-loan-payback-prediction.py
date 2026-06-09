import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print('Train Data Shape:',{train_df.shape})
print('Test Data Shape:',{test_df.shape})

train_df.head()


# Data Types and Null Values
print(train_df.info())

# Null Values Check
print(train_df.isnull().sum())

# Statistics
train_df.describe()


numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()

numerical_columns.remove('id')
numerical_columns.remove('loan_paid_back')

print(f"Numerical Columns ({len(numerical_columns)}):")
print(numerical_columns)
print(f"\nCategoric Columns ({len(categorical_columns)}):")
print(categorical_columns)


# Target Variable Analysis
plt.figure(figsize=(5, 4))
train_df['loan_paid_back'].value_counts().plot(kind='bar', color=['#e74c3c', '#2ecc71'])
plt.title('Loan Payback Distribution (Count)', fontsize=14, fontweight='bold')
plt.xlabel('Loan Paid Back (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()


# Numerical Features Distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_columns):
    axes[idx].hist(train_df[col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Categorical Features Distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(categorical_columns):
    value_counts = train_df[col].value_counts()
    axes[idx].bar(range(len(value_counts)), value_counts.values, color='coral', edgecolor='black')
    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
    axes[idx].set_xticks(range(len(value_counts)))
    axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# Correlation
correlation_data = train_df[numerical_columns + ['loan_paid_back']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# Target vs Numerical Feaatures
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_columns):
    sns.boxplot(data=train_df, x='loan_paid_back', y=col, ax=axes[idx], palette=['#e74c3c', '#2ecc71'])
    axes[idx].set_title(f'{col} vs Loan Payback', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Loan Paid Back (0=No, 1=Yes)')
    axes[idx].set_ylabel(col)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Feature Engineering
def create_features(df):
    df = df.copy()
    
    # 1. Grade/Subgrade Features
    df['grade'] = df['grade_subgrade'].str[0]  # A, B, C, D, E, F, G
    df['subgrade'] = df['grade_subgrade'].str[1].astype(int)  # 1, 2, 3, 4, 5
    
    # Ordinal encoding for grade (A=1, G=7)
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_ordinal'] = df['grade'].map(grade_map)
    
    # Combined risk score (A1=1, G5=35)
    df['grade_subgrade_ordinal'] = (df['grade_ordinal'] - 1) * 5 + df['subgrade']
    
    # 2. Financial Ratio Features
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
    
    # Monthly income and debt calculations
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_debt'] = df['annual_income'] * df['debt_to_income_ratio'] / 12
    df['available_monthly_income'] = df['monthly_income'] - df['monthly_debt']
    
    # Estimated monthly payment (simple interest approximation)
    df['estimated_monthly_payment'] = (df['loan_amount'] * (1 + df['interest_rate']/100)) / 36  # assuming 3-year term
    df['payment_to_income_ratio'] = df['estimated_monthly_payment'] / (df['monthly_income'] + 1)
    
    # Available income after loan payment
    df['residual_income'] = df['available_monthly_income'] - df['estimated_monthly_payment']
    df['total_debt_burden'] = df['debt_to_income_ratio'] + df['payment_to_income_ratio']
    
    # 3. Credit Risk Features
    df['credit_utilization_proxy'] = df['loan_amount'] / (df['credit_score'] + 1)
    df['risk_adjusted_amount'] = df['loan_amount'] * df['interest_rate'] / (df['credit_score'] + 1)
    df['credit_to_interest_ratio'] = df['credit_score'] / (df['interest_rate'] + 1)
    
    # 4. Interaction Features
    df['dti_x_interest'] = df['debt_to_income_ratio'] * df['interest_rate']
    df['credit_x_income'] = df['credit_score'] * df['annual_income'] / 100000
    df['loan_x_interest'] = df['loan_amount'] * df['interest_rate']
    
    # 5. Binning features (risk categories)
    df['income_bin'] = pd.cut(df['annual_income'], bins=[0, 30000, 50000, 70000, 100000, 500000], 
                               labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    df['credit_score_bin'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 800, 900],
                                     labels=['poor', 'fair', 'good', 'very_good', 'excellent'])
    df['dti_bin'] = pd.cut(df['debt_to_income_ratio'], bins=[0, 0.1, 0.2, 0.35, 1.0],
                            labels=['low', 'moderate', 'high', 'very_high'])
    
    # 6. Polynomial features (squared terms for important features)
    df['credit_score_squared'] = df['credit_score'] ** 2
    df['interest_rate_squared'] = df['interest_rate'] ** 2
    df['dti_squared'] = df['debt_to_income_ratio'] ** 2
    
    return df

# Apply feature engineering
train_df = create_features(train_df)
test_df = create_features(test_df)

print(f"Train shape after feature engineering: {train_df.shape}")
print(f"Test shape after feature engineering: {test_df.shape}")


from sklearn.model_selection import StratifiedKFold

def target_encode_features(train, test, cat_cols, target_col='loan_paid_back', n_splits=5):
    train = train.copy()
    test = test.copy()
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for col in cat_cols:
        print(f"Encoding {col}...")
        
        # Convert to string to handle categorical types
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        
        train[f'{col}_target_enc'] = 0.0
        for train_idx, val_idx in skf.split(train, train[target_col]):
            target_mean = train.iloc[train_idx].groupby(col)[target_col].mean()
            train.loc[val_idx, f'{col}_target_enc'] = train.loc[val_idx, col].map(target_mean)
        
        target_mean = train.groupby(col)[target_col].mean()
        test[f'{col}_target_enc'] = test[col].map(target_mean)
        
        # NaN handling (global mean)
        global_mean = train[target_col].mean()
        train[f'{col}_target_enc'] = train[f'{col}_target_enc'].fillna(global_mean)
        test[f'{col}_target_enc'] = test[f'{col}_target_enc'].fillna(global_mean)
    
    return train, test

# Apply target encoding
cat_features = ['gender', 'marital_status', 'education_level', 
                'employment_status', 'loan_purpose', 'grade', 'income_bin', 
                'credit_score_bin', 'dti_bin']

train_df, test_df = target_encode_features(train_df, test_df, cat_features)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Prepare data
X = train_df.drop(['id', 'loan_paid_back', 'grade_subgrade'], axis=1)
y = train_df['loan_paid_back']
X_test = test_df.drop(['id', 'grade_subgrade'], axis=1)

# Encode remaining categorical columns
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 
            'loan_purpose', 'grade', 'income_bin', 'credit_score_bin', 'dti_bin']

from sklearn.preprocessing import LabelEncoder
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

print(f"Training data shape: {X.shape}")
print(f"Test data shape: {X_test.shape}")


def train_lightgbm(X, y, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_splits
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
    
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nLightGBM Overall AUC: {overall_auc:.6f}")
    
    return oof_preds, test_preds

lgb_oof, lgb_test = train_lightgbm(X, y, X_test)


def train_xgboost(X, y, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'early_stopping_rounds': 100
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params, n_estimators=2000)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
    
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nXGBoost Overall AUC: {overall_auc:.6f}")
    
    return oof_preds, test_preds

xgb_oof, xgb_test = train_xgboost(X, y, X_test)


def train_catboost(X, y, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    params = {
        'iterations': 2000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 100,
        'task_type': 'CPU'
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=100
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
    
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nCatBoost Overall AUC: {overall_auc:.6f}")
    
    return oof_preds, test_preds

catb_oof, catb_test = train_catboost(X, y, X_test)


print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

lgb_auc = roc_auc_score(y, lgb_oof)
xgb_auc = roc_auc_score(y, xgb_oof)
catb_auc = roc_auc_score(y, catb_oof)

print(f"LightGBM CV AUC:  {lgb_auc:.6f}")
print(f"XGBoost CV AUC:   {xgb_auc:.6f}")
print(f"CatBoost CV AUC:  {catb_auc:.6f}")
print("="*60)

best_model = max([('LightGBM', lgb_auc), ('XGBoost', xgb_auc), ('CatBoost', catb_auc)], 
                 key=lambda x: x[1])
print(f"Best Single Model: {best_model[0]} with AUC {best_model[1]:.6f}")
print("="*60 + "\n")



weights = {
    'lgb': 0.30,
    'xgb': 0.25,
    'catb': 0.45 
}

ensemble_oof = (lgb_oof * weights['lgb'] + 
                xgb_oof * weights['xgb'] + 
                catb_oof * weights['catb'])

ensemble_test = (lgb_test * weights['lgb'] + 
                 xgb_test * weights['xgb'] + 
                 catb_test * weights['catb'])

ensemble_auc = roc_auc_score(y, ensemble_oof)
print(f"\n{'='*60}")
print(f"WEIGHTED ENSEMBLE AUC: {ensemble_auc:.6f}")
print(f"{'='*60}")

simple_ensemble_oof = (lgb_oof + xgb_oof + catb_oof) / 3
simple_ensemble_auc = roc_auc_score(y, simple_ensemble_oof)
print(f"Simple Average AUC: {simple_ensemble_auc:.6f}")
print(f"Weighted Average AUC: {ensemble_auc:.6f}")
print(f"Improvement: {(ensemble_auc - simple_ensemble_auc)*100:.4f}%")


# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': ensemble_test
})

submission.to_csv('submission.csv', index=False)
print("\n✅ Submission file created: ../submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nPrediction distribution:")
print(submission['loan_paid_back'].describe())





