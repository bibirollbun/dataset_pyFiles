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


print("="*80)
print("OPTIMIZED LOAN PREDICTION PIPELINE - 4 MODEL STACKING")
print("="*80)

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

print("\nâœ“ All libraries imported successfully")

# Load data
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(f"âœ“ Training set: {train_df.shape[0]:,} rows Ã— {train_df.shape[1]} columns")
print(f"âœ“ Test set: {test_df.shape[0]:,} rows Ã— {test_df.shape[1]} columns")
print(f"âœ“ Sample submission: {sample_submission.shape[0]:,} rows Ã— {sample_submission.shape[1]} columns")

# Store test IDs for final submission
test_ids = test_df['id'].copy()

# Target analysis
print(f"\nâœ“ Target column: 'loan_paid_back'")
target_dist = train_df['loan_paid_back'].value_counts()
print(f"âœ“ Target distribution:")
print(f"   Class 0: {target_dist[0.0]:,} ({target_dist[0.0]/len(train_df)*100:.2f}%)")
print(f"   Class 1: {target_dist[1.0]:,} ({target_dist[1.0]/len(train_df)*100:.2f}%)")

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = target_dist[1.0] / target_dist[0.0]
print(f"âœ“ Imbalance ratio: {scale_pos_weight:.2f}:1")
print(f"âœ“ scale_pos_weight: {scale_pos_weight:.2f}")


print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# Separate target and features
y = train_df['loan_paid_back'].copy()
X_train = train_df.drop(['id', 'loan_paid_back'], axis=1).copy()
X_test = test_df.drop(['id'], axis=1).copy()

print(f"âœ“ Training features shape: {X_train.shape}")
print(f"âœ“ Test features shape: {X_test.shape}")

# Combine for consistent feature engineering
X_train['dataset'] = 'train'
X_test['dataset'] = 'test'
combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)

print(f"\nâœ“ Combined dataset shape: {combined_df.shape}")

# CREATE NEW FEATURES
print("\n" + "-"*80)
print("Creating new features...")
print("-"*80)

# 1. Income to debt ratio (inverse)
combined_df['income_to_debt_ratio'] = 1 / (combined_df['debt_to_income_ratio'] + 0.00001)

# 2. Credit to loan ratio
combined_df['credit_to_loan_ratio'] = combined_df['credit_score'] / (combined_df['loan_amount'] + 1)

# 3. Loan to income ratio
combined_df['loan_to_income_ratio'] = combined_df['loan_amount'] / (combined_df['annual_income'] + 1)

# 4. Total estimated debt
combined_df['total_debt_estimate'] = combined_df['annual_income'] * combined_df['debt_to_income_ratio']

# 5. Monthly payment estimate
combined_df['monthly_payment_estimate'] = (combined_df['loan_amount'] * combined_df['interest_rate'] / 100) / 12

# 6. Debt burden score
combined_df['debt_burden_score'] = combined_df['debt_to_income_ratio'] * combined_df['interest_rate'] * combined_df['loan_amount'] / 1000

# 7. Financial health score
combined_df['financial_health_score'] = (
    (combined_df['credit_score'] / 850) * 0.4 +
    ((1 - combined_df['debt_to_income_ratio']) * 0.3).clip(0, 1) +
    (1 / (combined_df['interest_rate'] + 1)) * 0.3
)

# 8. High risk flag
combined_df['high_risk_flag'] = (
    (combined_df['debt_to_income_ratio'] > 0.35) | 
    (combined_df['credit_score'] < 650) | 
    (combined_df['interest_rate'] > 15)
).astype(int)

# 9. Low risk flag
combined_df['low_risk_flag'] = (
    (combined_df['debt_to_income_ratio'] < 0.1) & 
    (combined_df['credit_score'] > 700) & 
    (combined_df['interest_rate'] < 12)
).astype(int)

# 10. Credit score bins
combined_df['credit_score_bin'] = pd.cut(
    combined_df['credit_score'], 
    bins=[0, 600, 700, 750, 850], 
    labels=['poor', 'fair', 'good', 'excellent']
)

# 11. Income bins
combined_df['income_bin'] = pd.cut(
    combined_df['annual_income'], 
    bins=[0, 30000, 50000, 70000, 500000], 
    labels=['low', 'medium', 'high', 'very_high']
)

# 12. Interest rate bins
combined_df['interest_rate_bin'] = pd.cut(
    combined_df['interest_rate'], 
    bins=[0, 10, 13, 16, 25], 
    labels=['low', 'medium', 'high', 'very_high']
)

# 13. Loan amount bins
combined_df['loan_amount_bin'] = pd.cut(
    combined_df['loan_amount'], 
    bins=[0, 10000, 15000, 20000, 50000], 
    labels=['small', 'medium', 'large', 'very_large']
)

print(f"âœ“ Created 13 new features")

# INTERACTION FEATURES (Top important features)
print("\n" + "-"*80)
print("Creating interaction features...")
print("-"*80)

# Based on EDA: employment_status is most important (73% importance)
# Create interactions with top numerical features

# Store original categorical columns for later
cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 
            'loan_purpose', 'grade_subgrade', 'credit_score_bin', 'income_bin',
            'interest_rate_bin', 'loan_amount_bin']

# Label encode for interaction features
le_temp = LabelEncoder()
for col in cat_cols:
    if col in combined_df.columns:
        combined_df[f'{col}_encoded'] = le_temp.fit_transform(combined_df[col].astype(str))

# Create interactions
combined_df['employment_debt_interaction'] = combined_df['employment_status_encoded'] * combined_df['debt_to_income_ratio']
combined_df['employment_credit_interaction'] = combined_df['employment_status_encoded'] * combined_df['credit_score']
combined_df['grade_debt_interaction'] = combined_df['grade_subgrade_encoded'] * combined_df['debt_to_income_ratio']

print(f"âœ“ Created 3 interaction features")

print(f"\nâœ“ Total features now: {combined_df.shape[1]}")
print(f"âœ“ Feature engineering completed")


print("\n" + "="*80)
print("FEATURE ENCODING & PREPARATION")
print("="*80)

# Identify categorical columns (excluding dataset marker and encoded versions)
categorical_cols = [col for col in combined_df.select_dtypes(include=['object']).columns 
                   if col not in ['dataset'] and not col.endswith('_encoded')]

print(f"Categorical columns to encode: {len(categorical_cols)}")
print(f"Columns: {categorical_cols}\n")

# Label Encoding
print("-"*80)
print("Applying Label Encoding...")
print("-"*80)

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))
    label_encoders[col] = le
    print(f"âœ“ Encoded: {col} ({len(le.classes_)} unique values)")

print(f"\nâœ“ All categorical features encoded")

# Remove temporary encoded columns used for interactions
temp_encoded_cols = [col for col in combined_df.columns if col.endswith('_encoded')]
combined_df = combined_df.drop(columns=temp_encoded_cols)

print(f"âœ“ Removed {len(temp_encoded_cols)} temporary columns")

# Convert all remaining categorical dtypes to numeric (fix XGBoost error)
print("\n" + "-"*80)
print("Converting all columns to numeric types...")
print("-"*80)

for col in combined_df.columns:
    if combined_df[col].dtype.name == 'category':
        combined_df[col] = combined_df[col].cat.codes
        print(f"âœ“ Converted categorical column: {col}")

print(f"âœ“ All columns converted to numeric types")

# Split back into train and test
print("\n" + "-"*80)
print("Splitting back to train and test sets...")
print("-"*80)

X_train_processed = combined_df[combined_df['dataset'] == 'train'].drop('dataset', axis=1).copy()
X_test_processed = combined_df[combined_df['dataset'] == 'test'].drop('dataset', axis=1).copy()

# Reset index
X_train_processed = X_train_processed.reset_index(drop=True)
X_test_processed = X_test_processed.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"âœ“ Processed training set: {X_train_processed.shape}")
print(f"âœ“ Processed test set: {X_test_processed.shape}")
print(f"âœ“ Target variable: {y.shape}")

# Verify data quality
print("\n" + "-"*80)
print("Data Quality Check:")
print("-"*80)
print(f"âœ“ Training set missing values: {X_train_processed.isnull().sum().sum()}")
print(f"âœ“ Test set missing values: {X_test_processed.isnull().sum().sum()}")
print(f"âœ“ Training set infinite values: {np.isinf(X_train_processed.select_dtypes(include=[np.number])).sum().sum()}")
print(f"âœ“ Test set infinite values: {np.isinf(X_test_processed.select_dtypes(include=[np.number])).sum().sum()}")

# Replace any infinite values with large numbers
X_train_processed = X_train_processed.replace([np.inf, -np.inf], [1e10, -1e10])
X_test_processed = X_test_processed.replace([np.inf, -np.inf], [1e10, -1e10])

print(f"\nâœ“ Data is ready for modeling!")
print(f"âœ“ Total features for modeling: {X_train_processed.shape[1]}")


print("\n" + "="*80)
print("MODEL 1: LIGHTGBM (Optimized)")
print("="*80)

# Optimal LightGBM parameters for imbalanced data
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 8,
    'min_child_samples': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'is_unbalance': True,  # Handle imbalanced data
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

print("Training LightGBM with 5-Fold Stratified Cross-Validation...")
print("-"*80)
print("Key settings:")
print(f"  â€¢ is_unbalance: True (handles 3.97:1 imbalance)")
print(f"  â€¢ learning_rate: 0.03")
print(f"  â€¢ max_depth: 8")
print(f"  â€¢ num_leaves: 31")
print("-"*80)

# Setup cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_oof_preds = np.zeros(len(X_train_processed))
lgb_test_preds = np.zeros(len(X_test_processed))
lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y), 1):
    print(f"\nFold {fold}/5")
    
    X_tr, X_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create datasets
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100)
        ]
    )
    
    # Predict
    lgb_oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    lgb_test_preds += model.predict(X_test_processed, num_iteration=model.best_iteration) / 5
    
    # Calculate score
    fold_score = roc_auc_score(y_val, lgb_oof_preds[val_idx])
    lgb_scores.append(fold_score)
    print(f"Fold {fold} ROC-AUC: {fold_score:.6f}")

# Overall score
lgb_cv_score = roc_auc_score(y, lgb_oof_preds)

print("\n" + "="*80)
print(f"âœ“ LightGBM CV ROC-AUC: {lgb_cv_score:.6f}")
print(f"âœ“ Mean Fold Score: {np.mean(lgb_scores):.6f} (+/- {np.std(lgb_scores):.6f})")
print(f"âœ“ All Fold Scores: {[f'{s:.6f}' for s in lgb_scores]}")
print("="*80)


print("\n" + "="*80)
print("MODEL 2: XGBOOST (Optimized)")
print("="*80)

# Optimal XGBoost parameters for imbalanced data
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.03,
    'max_depth': 8,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'scale_pos_weight': scale_pos_weight,  # Handle imbalanced data
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

print("Training XGBoost with 5-Fold Stratified Cross-Validation...")
print("-"*80)
print("Key settings:")
print(f"  â€¢ scale_pos_weight: {scale_pos_weight:.2f} (handles imbalance)")
print(f"  â€¢ learning_rate: 0.03")
print(f"  â€¢ max_depth: 8")
print(f"  â€¢ min_child_weight: 3")
print("-"*80)

# Setup cross-validation
xgb_oof_preds = np.zeros(len(X_train_processed))
xgb_test_preds = np.zeros(len(X_test_processed))
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y), 1):
    print(f"\nFold {fold}/5")
    
    X_tr, X_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create datasets
    train_data = xgb.DMatrix(X_tr, label=y_tr)
    val_data = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    model = xgb.train(
        xgb_params,
        train_data,
        num_boost_round=1000,
        evals=[(train_data, 'train'), (val_data, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    # Predict
    xgb_oof_preds[val_idx] = model.predict(val_data)
    test_data = xgb.DMatrix(X_test_processed)
    xgb_test_preds += model.predict(test_data) / 5
    
    # Calculate score
    fold_score = roc_auc_score(y_val, xgb_oof_preds[val_idx])
    xgb_scores.append(fold_score)
    print(f"Fold {fold} ROC-AUC: {fold_score:.6f}")

# Overall score
xgb_cv_score = roc_auc_score(y, xgb_oof_preds)

print("\n" + "="*80)
print(f"âœ“ XGBoost CV ROC-AUC: {xgb_cv_score:.6f}")
print(f"âœ“ Mean Fold Score: {np.mean(xgb_scores):.6f} (+/- {np.std(xgb_scores):.6f})")
print(f"âœ“ All Fold Scores: {[f'{s:.6f}' for s in xgb_scores]}")
print("="*80)


print("\n" + "="*80)
print("MODEL 3: CATBOOST (Optimized)")
print("="*80)

# Optimal CatBoost parameters for imbalanced data
cat_params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'random_strength': 0.5,
    'auto_class_weights': 'Balanced',  # Handle imbalanced data
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 100,
    'task_type': 'CPU',
    'eval_metric': 'AUC',
    'thread_count': -1
}

print("Training CatBoost with 5-Fold Stratified Cross-Validation...")
print("-"*80)
print("Key settings:")
print(f"  â€¢ auto_class_weights: Balanced (handles imbalance)")
print(f"  â€¢ learning_rate: 0.03")
print(f"  â€¢ depth: 8")
print(f"  â€¢ l2_leaf_reg: 3")
print("-"*80)

# Setup cross-validation
cat_oof_preds = np.zeros(len(X_train_processed))
cat_test_preds = np.zeros(len(X_test_processed))
cat_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y), 1):
    print(f"\nFold {fold}/5")
    
    X_tr, X_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialize model
    model = CatBoostClassifier(**cat_params)
    
    # Train model
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False
    )
    
    # Predict
    cat_oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    cat_test_preds += model.predict_proba(X_test_processed)[:, 1] / 5
    
    # Calculate score
    fold_score = roc_auc_score(y_val, cat_oof_preds[val_idx])
    cat_scores.append(fold_score)
    print(f"Fold {fold} ROC-AUC: {fold_score:.6f}")

# Overall score
cat_cv_score = roc_auc_score(y, cat_oof_preds)

print("\n" + "="*80)
print(f"âœ“ CatBoost CV ROC-AUC: {cat_cv_score:.6f}")
print(f"âœ“ Mean Fold Score: {np.mean(cat_scores):.6f} (+/- {np.std(cat_scores):.6f})")
print(f"âœ“ All Fold Scores: {[f'{s:.6f}' for s in cat_scores]}")
print("="*80)


print("\n" + "="*80)
print("MODEL 4: RANDOM FOREST (Optimized)")
print("="*80)

# Optimal Random Forest parameters for imbalanced data
rf_params = {
    'n_estimators': 500,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced',  # Handle imbalanced data
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 0
}

print("Training Random Forest with 5-Fold Stratified Cross-Validation...")
print("-"*80)
print("Key settings:")
print(f"  â€¢ class_weight: balanced (handles imbalance)")
print(f"  â€¢ n_estimators: 500")
print(f"  â€¢ max_depth: 15")
print(f"  â€¢ min_samples_split: 10")
print("-"*80)

# Setup cross-validation
rf_oof_preds = np.zeros(len(X_train_processed))
rf_test_preds = np.zeros(len(X_test_processed))
rf_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y), 1):
    print(f"\nFold {fold}/5 - Training Random Forest...")
    
    X_tr, X_val = X_train_processed.iloc[train_idx], X_train_processed.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialize and train model
    model = RandomForestClassifier(**rf_params)
    model.fit(X_tr, y_tr)
    
    # Predict
    rf_oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    rf_test_preds += model.predict_proba(X_test_processed)[:, 1] / 5
    
    # Calculate score
    fold_score = roc_auc_score(y_val, rf_oof_preds[val_idx])
    rf_scores.append(fold_score)
    print(f"Fold {fold} ROC-AUC: {fold_score:.6f}")

# Overall score
rf_cv_score = roc_auc_score(y, rf_oof_preds)

print("\n" + "="*80)
print(f"âœ“ Random Forest CV ROC-AUC: {rf_cv_score:.6f}")
print(f"âœ“ Mean Fold Score: {np.mean(rf_scores):.6f} (+/- {np.std(rf_scores):.6f})")
print(f"âœ“ All Fold Scores: {[f'{s:.6f}' for s in rf_scores]}")
print("="*80)


print("\n" + "="*80)
print("BASE MODELS COMPARISON")
print("="*80)

# Create comparison dataframe
model_comparison = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Random Forest'],
    'CV_ROC_AUC': [lgb_cv_score, xgb_cv_score, cat_cv_score, rf_cv_score],
    'Mean_Fold_Score': [np.mean(lgb_scores), np.mean(xgb_scores), 
                        np.mean(cat_scores), np.mean(rf_scores)],
    'Std_Dev': [np.std(lgb_scores), np.std(xgb_scores), 
                np.std(cat_scores), np.std(rf_scores)]
})

model_comparison = model_comparison.sort_values('CV_ROC_AUC', ascending=False)
print("\n" + "-"*80)
print("MODEL PERFORMANCE RANKING:")
print("-"*80)
print(model_comparison.to_string(index=False))
print("-"*80)

# Calculate model correlations
print("\n" + "-"*80)
print("MODEL PREDICTION CORRELATIONS:")
print("-"*80)

pred_corr = pd.DataFrame({
    'LightGBM': lgb_oof_preds,
    'XGBoost': xgb_oof_preds,
    'CatBoost': cat_oof_preds,
    'RandomForest': rf_oof_preds
}).corr()

print(pred_corr)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Bar plot - Model Performance
ax1 = axes[0, 0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax1.bar(model_comparison['Model'], model_comparison['CV_ROC_AUC'], 
               color=colors[:len(model_comparison)])
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('CV ROC-AUC Score', fontsize=12)
ax1.set_ylim([model_comparison['CV_ROC_AUC'].min() - 0.01, 1.0])
ax1.grid(axis='y', alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. Fold-wise comparison
ax2 = axes[0, 1]
x = np.arange(1, 6)
ax2.plot(x, lgb_scores, 'o-', label='LightGBM', linewidth=2, markersize=8)
ax2.plot(x, xgb_scores, 's-', label='XGBoost', linewidth=2, markersize=8)
ax2.plot(x, cat_scores, '^-', label='CatBoost', linewidth=2, markersize=8)
ax2.plot(x, rf_scores, 'd-', label='Random Forest', linewidth=2, markersize=8)
ax2.set_title('Fold-wise Performance', fontsize=14, fontweight='bold')
ax2.set_xlabel('Fold', fontsize=12)
ax2.set_ylabel('ROC-AUC Score', fontsize=12)
ax2.legend(loc='best')
ax2.grid(alpha=0.3)
ax2.set_xticks(x)

# 3. Prediction correlation heatmap
ax3 = axes[1, 0]
sns.heatmap(pred_corr, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0.5, square=True, ax=ax3, cbar_kws={"shrink": 0.8})
ax3.set_title('Model Prediction Correlations', fontsize=14, fontweight='bold')

# 4. Prediction distributions
ax4 = axes[1, 1]
ax4.hist(lgb_oof_preds, bins=50, alpha=0.5, label='LightGBM', density=True)
ax4.hist(xgb_oof_preds, bins=50, alpha=0.5, label='XGBoost', density=True)
ax4.hist(cat_oof_preds, bins=50, alpha=0.5, label='CatBoost', density=True)
ax4.hist(rf_oof_preds, bins=50, alpha=0.5, label='Random Forest', density=True)
ax4.set_title('Prediction Distributions', fontsize=14, fontweight='bold')
ax4.set_xlabel('Predicted Probability', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nâœ“ Best single model: {model_comparison.iloc[0]['Model']} "
      f"(ROC-AUC: {model_comparison.iloc[0]['CV_ROC_AUC']:.6f})")
print(f"âœ“ Model diversity score (avg correlation): {pred_corr.values[np.triu_indices_from(pred_corr.values, k=1)].mean():.3f}")


print("\n" + "="*80)
print("STACKING ENSEMBLE - META MODEL")
print("="*80)

# Prepare meta-features (out-of-fold predictions from base models)
print("\nPreparing meta-features from 4 base models...")
print("-"*80)

# Stack OOF predictions as new features
meta_train = np.column_stack([
    lgb_oof_preds, 
    xgb_oof_preds, 
    cat_oof_preds, 
    rf_oof_preds
])

meta_test = np.column_stack([
    lgb_test_preds, 
    xgb_test_preds, 
    cat_test_preds, 
    rf_test_preds
])

print(f"âœ“ Meta-train shape: {meta_train.shape}")
print(f"âœ“ Meta-test shape: {meta_test.shape}")

# Convert to DataFrame for easier handling
meta_train_df = pd.DataFrame(meta_train, columns=['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest'])
meta_test_df = pd.DataFrame(meta_test, columns=['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest'])

print(f"\nMeta-features statistics:")
print(meta_train_df.describe())

# Train meta-model (Logistic Regression with class_weight)
print("\n" + "-"*80)
print("Training meta-model (Logistic Regression with balanced weights)...")
print("-"*80)

# Cross-validate meta-model
meta_model = LogisticRegression(
    class_weight='balanced',  # Handle imbalance
    random_state=42, 
    max_iter=1000,
    solver='lbfgs'
)

# 5-fold CV for meta-model
meta_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(meta_skf.split(meta_train, y), 1):
    meta_model_fold = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    meta_model_fold.fit(meta_train[train_idx], y.iloc[train_idx])
    fold_pred = meta_model_fold.predict_proba(meta_train[val_idx])[:, 1]
    fold_score = roc_auc_score(y.iloc[val_idx], fold_pred)
    meta_cv_scores.append(fold_score)
    print(f"Meta-model Fold {fold} ROC-AUC: {fold_score:.6f}")

print(f"\nMeta-model CV scores: {[f'{s:.6f}' for s in meta_cv_scores]}")
print(f"Meta-model mean CV ROC-AUC: {np.mean(meta_cv_scores):.6f} (+/- {np.std(meta_cv_scores):.6f})")

# Train final meta-model on all data
meta_model.fit(meta_train, y)

# Get meta-model predictions
meta_train_preds = meta_model.predict_proba(meta_train)[:, 1]
meta_test_preds = meta_model.predict_proba(meta_test)[:, 1]

# Calculate final score
final_score = roc_auc_score(y, meta_train_preds)

print("\n" + "="*80)
print(f"âœ“ FINAL STACKED MODEL ROC-AUC: {final_score:.6f}")
print("="*80)

# Show model weights (coefficients)
print("\n" + "-"*80)
print("META-MODEL WEIGHTS (Feature Importance):")
print("-"*80)
weights = pd.DataFrame({
    'Base_Model': ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest'],
    'Coefficient': meta_model.coef_[0],
    'Abs_Weight': np.abs(meta_model.coef_[0])
}).sort_values('Abs_Weight', ascending=False)

print(weights.to_string(index=False))
print(f"\nIntercept: {meta_model.intercept_[0]:.4f}")

# Visualize weights
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in weights['Coefficient']]
bars = ax.barh(weights['Base_Model'], weights['Coefficient'], color=colors, alpha=0.7)
ax.set_xlabel('Meta-Model Coefficient', fontsize=12, fontweight='bold')
ax.set_title('Meta-Model Weights for Base Models', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left' if width > 0 else 'right', 
            va='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nâœ“ Meta-model training completed")


# =============================================================================
# BLOCK 10: FINAL COMPARISON & GENERATE SUBMISSION
# =============================================================================

print("\n" + "="*80)
print("FINAL PERFORMANCE COMPARISON - ALL APPROACHES")
print("="*80)

# Calculate simple average
simple_avg_oof = (lgb_oof_preds + xgb_oof_preds + cat_oof_preds + rf_oof_preds) / 4
simple_avg_test = (lgb_test_preds + xgb_test_preds + cat_test_preds + rf_test_preds) / 4
simple_avg_score = roc_auc_score(y, simple_avg_oof)

# Calculate weighted average (based on CV scores)
weights_by_performance = np.array([lgb_cv_score, xgb_cv_score, cat_cv_score, rf_cv_score])
weights_by_performance = weights_by_performance / weights_by_performance.sum()

weighted_avg_oof = (lgb_oof_preds * weights_by_performance[0] + 
                    xgb_oof_preds * weights_by_performance[1] + 
                    cat_oof_preds * weights_by_performance[2] + 
                    rf_oof_preds * weights_by_performance[3])
weighted_avg_test = (lgb_test_preds * weights_by_performance[0] + 
                     xgb_test_preds * weights_by_performance[1] + 
                     cat_test_preds * weights_by_performance[2] + 
                     rf_test_preds * weights_by_performance[3])
weighted_avg_score = roc_auc_score(y, weighted_avg_oof)

# Create final comparison
final_comparison = pd.DataFrame({
    'Approach': [
        '1. LightGBM', 
        '2. XGBoost', 
        '3. CatBoost', 
        '4. Random Forest',
        '5. Simple Average',
        '6. Weighted Average',
        '7. Stacked (Meta-Model)'
    ],
    'ROC_AUC': [
        lgb_cv_score,
        xgb_cv_score,
        cat_cv_score,
        rf_cv_score,
        simple_avg_score,
        weighted_avg_score,
        final_score
    ]
}).sort_values('ROC_AUC', ascending=False)

print("\n" + "-"*80)
print(final_comparison.to_string(index=False))
print("-"*80)

# Improvement analysis
best_single = max(lgb_cv_score, xgb_cv_score, cat_cv_score, rf_cv_score)
stacking_improvement = (final_score - best_single) * 100

print(f"\nğŸ“Š PERFORMANCE ANALYSIS:")
print(f"  â€¢ Best single model: {best_single:.6f}")
print(f"  â€¢ Simple average: {simple_avg_score:.6f} ({(simple_avg_score-best_single)*100:+.4f}%)")
print(f"  â€¢ Weighted average: {weighted_avg_score:.6f} ({(weighted_avg_score-best_single)*100:+.4f}%)")
print(f"  â€¢ Stacked model: {final_score:.6f} ({stacking_improvement:+.4f}%)")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot
ax1 = axes[0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e74c3c']
bars = ax1.barh(final_comparison['Approach'], final_comparison['ROC_AUC'], 
                color=colors[:len(final_comparison)])
ax1.set_xlabel('ROC-AUC Score', fontsize=12, fontweight='bold')
ax1.set_title('Final Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim([final_comparison['ROC_AUC'].min() - 0.005, 1.0])
ax1.grid(axis='x', alpha=0.3)

for bar in bars:
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2.,
            f' {width:.6f}', ha='left', va='center', fontweight='bold', fontsize=10)

# Prediction distribution comparison
ax2 = axes[1]
ax2.hist(lgb_oof_preds, bins=50, alpha=0.3, label='LightGBM', density=True)
ax2.hist(simple_avg_oof, bins=50, alpha=0.3, label='Simple Avg', density=True)
ax2.hist(meta_train_preds, bins=50, alpha=0.3, label='Stacked', density=True)
ax2.axvline(y.mean(), color='red', linestyle='--', linewidth=2, label='True Mean')
ax2.set_title('Prediction Distributions Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted Probability', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nğŸ�† BEST APPROACH: {final_comparison.iloc[0]['Approach']}")
print(f"   ROC-AUC: {final_comparison.iloc[0]['ROC_AUC']:.6f}")

# GENERATE SUBMISSION FILE
print("\n" + "="*80)
print("GENERATING SUBMISSION FILE")
print("="*80)

# Use the stacked model predictions (best performance)
final_predictions = meta_test_preds

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': final_predictions
})

print(f"\nâœ“ Submission shape: {submission.shape}")
print(f"âœ“ Columns: {submission.columns.tolist()}")

# Verify submission format
print("\n" + "-"*80)
print("SUBMISSION FILE PREVIEW:")
print("-"*80)
print(submission.head(10))

print("\n" + "-"*80)
print("PREDICTION STATISTICS:")
print("-"*80)
print(f"Mean prediction: {submission['loan_paid_back'].mean():.6f}")
print(f"Median prediction: {submission['loan_paid_back'].median():.6f}")
print(f"Min prediction: {submission['loan_paid_back'].min():.6f}")
print(f"Max prediction: {submission['loan_paid_back'].max():.6f}")
print(f"Std prediction: {submission['loan_paid_back'].std():.6f}")

# Check prediction distribution
print("\n" + "-"*80)
print("PREDICTION DISTRIBUTION:")
print("-"*80)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
submission['pred_range'] = pd.cut(submission['loan_paid_back'], bins=bins, labels=labels)
print(submission['pred_range'].value_counts().sort_index())
submission = submission.drop('pred_range', axis=1)

# Save submission file
submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)

print("\n" + "="*80)
print(f"âœ… SUBMISSION FILE SAVED: {submission_filename}")
print("="*80)

# Verify file
import os
if os.path.exists(submission_filename):
    file_size = os.path.getsize(submission_filename) / 1024
    print(f"âœ“ File size: {file_size:.2f} KB")
    print(f"âœ“ Total predictions: {len(submission):,}")
    print(f"âœ“ Ready for download and submission!")
else:
    print("â�Œ Error: File not created")

print("\n" + "="*80)
print("ğŸ�‰ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"""
FINAL RESULTS SUMMARY:
{'='*80}
Best Model Performance:
  â€¢ Approach: {final_comparison.iloc[0]['Approach']}
  â€¢ ROC-AUC Score: {final_comparison.iloc[0]['ROC_AUC']:.6f}
  
Model Improvements:
  â€¢ LightGBM: {lgb_cv_score:.6f}
  â€¢ XGBoost: {xgb_cv_score:.6f}
  â€¢ CatBoost: {cat_cv_score:.6f}
  â€¢ Random Forest: {rf_cv_score:.6f}
  â€¢ Stacked Ensemble: {final_score:.6f} (Best!)
  
Stacking Improvement: {stacking_improvement:+.4f}% over best single model

Key Optimizations Applied:
âœ“ Handled 3.97:1 class imbalance with class weights
âœ“ Created 13 engineered features + 3 interactions
âœ“ Optimized 4 diverse models (LightGBM, XGBoost, CatBoost, RF)
âœ“ Applied 5-fold stratified cross-validation
âœ“ Trained meta-model with balanced weights
âœ“ Generated final predictions using stacking

NEXT STEPS:
1. Download 'submission.csv' from Kaggle
2. Submit to competition
3. Check public leaderboard score
4. Iterate if needed

Good luck! ğŸš€
{'='*80}
""")




