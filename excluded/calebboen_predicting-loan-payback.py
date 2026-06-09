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
import plotly.express as px
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
import xgboost as xgb
from sklearn.ensemble import VotingClassifier  
import joblib

import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)
import random
random.seed(42)


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

train.head()


print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample sub shape:", sample_sub.shape)


train_id = train['id']
test_id = test['id']
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


print("\nTrain info:")
print(train.info())
print("\nMissing values in train:\n", train.isnull().sum())

print("\nTest info:")
print(test.info())
print("\nMissing values in test:\n", test.isnull().sum())


target_col = 'loan_paid_back'
print(f"\nTarget distribution:\n{train[target_col].value_counts(normalize=True)}")


numeric_cols = train.select_dtypes(include=[np.number]).columns.drop(target_col)
print("\nNumeric features summary:\n")
print(train[numeric_cols].describe())


cat_cols = train.select_dtypes(include=['object']).columns
print(f"\nCategorical columns: {list(cat_cols)}")
for col in cat_cols:
    print(f"\n{col} unique values:\n{train[col].value_counts().head(10)}")  # Top 10 to avoid spam


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

train[target_col].value_counts().plot(kind='bar', ax=axes[0,0])
axes[0,0].set_title('Target Distribution')

train['annual_income'].hist(bins=50, ax=axes[0,1])
axes[0,1].set_title('Annual Income Distribution')

train['credit_score'].hist(bins=50, ax=axes[1,0])
axes[1,0].set_title('Credit Score Distribution')

train['debt_to_income_ratio'].hist(bins=50, ax=axes[1,1])
axes[1,1].set_title('DTI Ratio Distribution')
plt.tight_layout()
plt.show()


if 'gender' in cat_cols:
    pd.crosstab(train['gender'], train[target_col], normalize='index').plot(kind='bar', ax=plt.gca())
    plt.title('Gender vs Target (Normalized)')
    plt.show()


full_train = train.copy()
full_train[target_col] = full_train[target_col]  


numeric_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
corr_matrix = full_train[numeric_cols + [target_col]].corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
plt.title('Spearman Correlation Matrix (Numerics + Target)')
plt.show()

print("\nSpearman correlations with target:")
for col in numeric_cols:
    corr = corr_matrix.loc[col, target_col]
    print(f"{col}: {corr:.4f}")


print("\nKS tests (numeric features by target):")
for col in numeric_cols:
    stat, pval = ks_2samp(full_train[full_train[target_col]==1][col], 
                          full_train[full_train[target_col]==0][col])
    print(f"{col}: KS-stat={stat:.4f}, p-value={pval:.2e} (sig diff: {pval<0.05})")


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(numeric_cols):
    row, col_idx = i // 3, i % 3
    sns.boxplot(data=full_train, x=target_col, y=col, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'{col} by Target')
    axes[row, col_idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for i, col in enumerate(cat_cols):
    row, col_idx = i // 3, i % 3
    crosstab = pd.crosstab(full_train[col], full_train[target_col], normalize='index')
    crosstab.plot(kind='bar', stacked=True, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'{col} vs Target (Stacked Prop)')
    axes[row, col_idx].legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[row, col_idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


# Log-transform skeweds (annual_income, loan_amount)
full_train['log_annual_income'] = np.log1p(full_train['annual_income'])
full_train['log_loan_amount'] = np.log1p(full_train['loan_amount'])
full_train['dti_outlier'] = (full_train['debt_to_income_ratio'] > 0.4).astype(int)


# Ordinal grade: Extract numeric (C=3, D=4) + subgrade digit
full_train['grade_num'] = full_train['grade_subgrade'].str[0].map({'C': 3, 'D': 4})
full_train['subgrade_num'] = full_train['grade_subgrade'].str[1].astype(int)
full_train['grade_interaction'] = full_train['grade_num'] * full_train['subgrade_num']


# Rare cat grouping (threshold <1% freq)
def rare_group(col, threshold=0.01):
    freq = full_train[col].value_counts(normalize=True)
    rare = freq[freq < threshold].index
    full_train[col + '_grouped'] = full_train[col].replace(rare, 'Rare')
    return full_train[col + '_grouped'].value_counts()

print("\nRare-grouped categories (post <1%):")
for col in ['employment_status', 'loan_purpose']:
    print(f"\n{col}:")
    print(rare_group(col))


# Apply to test
test['log_annual_income'] = np.log1p(test['annual_income'])
test['log_loan_amount'] = np.log1p(test['loan_amount'])
test['dti_outlier'] = (test['debt_to_income_ratio'] > 0.4).astype(int)
test['grade_num'] = test['grade_subgrade'].str[0].map({'C': 3, 'D': 4})
test['subgrade_num'] = test['grade_subgrade'].str[1].astype(int)
test['grade_interaction'] = test['grade_num'] * test['subgrade_num']
for col in ['employment_status', 'loan_purpose']:
    freq = full_train[col].value_counts(normalize=True)  # Use train freq for consistency
    rare = freq[freq < 0.01].index
    test[col + '_grouped'] = test[col].replace(rare, 'Rare')


new_feats = ['log_annual_income', 'log_loan_amount', 'dti_outlier', 'grade_num', 'subgrade_num', 'grade_interaction']
print(f"\nNew engineered features: {new_feats}")
print("Train new feats summary:\n", full_train[new_feats].describe())


new_feats = ['log_annual_income', 'log_loan_amount', 'dti_outlier', 'grade_num', 'subgrade_num', 'grade_interaction']
print(f"\nNew engineered features: {new_feats}")
print("Train new feats summary:\n", full_train[new_feats].describe())


feature_cols = numeric_cols + cat_cols + new_feats
X = full_train[feature_cols]
y = full_train[target_col]


label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le  
    test[col] = le.transform(test[col].astype(str))  # Consistent encoding


X = X.fillna(0)  
test_eng = test[feature_cols].fillna(0)  

print(f"Final feature set: {len(feature_cols)} features")
print(f"X shape: {X.shape}, y positives: {y.mean():.4f}")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
scores = []


pos_weight = (1 - y.mean()) / y.mean()  # ~4.0 for 20% i.e. target=1 is majority (0.8), so for default=0 minority, weight=0.8/0.2=4
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': pos_weight,  # Boost minority (default=0)
    'random_state': 42,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Datasets for LGBM
    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_ds)
    
    # Train with early stopping
    model = lgb.train(
        params,
        train_ds,
        valid_sets=[val_ds],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_preds
    
    score = roc_auc_score(y_val, val_preds)
    scores.append(score)
    print(f"Fold {fold+1}: AUC = {score:.6f}")


oof_auc = roc_auc_score(y, oof_preds)
print(f"\nMean CV AUC: {np.mean(scores):.6f} (+/- {np.std(scores):.4f}) | OOF AUC: {oof_auc:.6f}")


importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain'),
    'folder': fold+1  # Last fold for viz
}).sort_values('importance', ascending=False)

print("\nTop 15 Feature Importances (Gain):\n", importances.head(15))


plt.figure(figsize=(10, 8))
sns.barplot(data=importances.head(15), x='importance', y='feature')
plt.title('Top 15 LightGBM Feature Importances (Gain)')
plt.tight_layout()
plt.show()


# Prune low-importance feats (threshold 1000 gain from baseline)
keep_feats = importances[importances['importance'] > 1000]['feature'].tolist() + cat_cols  # Ensure cats
keep_feats = list(set(keep_feats))  # Dedupe
X_pruned = X[keep_feats]
test_eng_pruned = test_eng[keep_feats]
print(f"Pruned to {len(keep_feats)} features: {keep_feats}")


cat_feats = [c for c in cat_cols if c in keep_feats]
print(f"cat_feats: {cat_feats} (matches cat_cols? {cat_feats == cat_cols})")


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'scale_pos_weight': pos_weight,
        'random_state': 42,
        'verbose': -1
    }
    
    cv_scores = []
    for train_idx, val_idx in skf.split(X_pruned, y):  # Reuse skf
        X_train, X_val = X_pruned.iloc[train_idx], X_pruned.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=[c for c in cat_cols if c in keep_feats])
        val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=[c for c in cat_cols if c in keep_feats], reference=train_ds)
        
        model = lgb.train(params, train_ds, valid_sets=[val_ds], num_boost_round=500,
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        cv_scores.append(roc_auc_score(y_val, val_preds))
    
    return np.mean(cv_scores)


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)
best_params = study.best_params
print(f"Best LGBM params: {best_params}")
print(f"Best CV AUC from tuning: {study.best_value:.6f}")


tuned_oof_lgb = np.zeros(len(X_pruned))
tuned_scores_lgb = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_pruned, y)):
    X_train, X_val = X_pruned.iloc[train_idx], X_pruned.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=[c for c in cat_cols if c in keep_feats])
    val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=[c for c in cat_cols if c in keep_feats], reference=train_ds)
    
    full_params = {**params, **best_params}  # Merge with fixed
    model = lgb.train(full_params, train_ds, valid_sets=[val_ds], num_boost_round=1000,
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    tuned_oof_lgb[val_idx] = val_preds
    tuned_scores_lgb.append(roc_auc_score(y_val, val_preds))
    print(f"Tuned Fold {fold+1}: AUC = {tuned_scores_lgb[-1]:.6f}")

tuned_auc_lgb = roc_auc_score(y, tuned_oof_lgb)
print(f"\nTuned LGBM Mean CV AUC: {np.mean(tuned_scores_lgb):.6f} (+/- {np.std(tuned_scores_lgb):.4f}) | OOF: {tuned_auc_lgb:.6f}")


xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': pos_weight,
    'random_state': 42,
    'verbosity': 0
}
xgb_oof = np.zeros(len(X_pruned))
xgb_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_pruned, y)):
    X_train, X_val = X_pruned.iloc[train_idx], X_pruned.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=[(dval, 'val')], 
                      early_stopping_rounds=50, verbose_eval=False)
    
    best_iter = getattr(model, 'best_iteration', 500)
    val_preds = model.predict(dval, iteration_range=(0, best_iter))
    xgb_oof[val_idx] = val_preds
    score = roc_auc_score(y_val, val_preds)
    xgb_scores.append(score)
    print(f"XGB Fold {fold+1}: AUC = {score:.6f}")

xgb_auc = roc_auc_score(y, xgb_oof)
print(f"XGB Mean CV AUC: {np.mean(xgb_scores):.6f} (+/- {np.std(xgb_scores):.4f}) | OOF: {xgb_auc:.6f}")


ensemble_oof = (tuned_oof_lgb + xgb_oof) / 2
ensemble_auc = roc_auc_score(y, ensemble_oof)
print(f"\nEnsemble OOF AUC: {ensemble_auc:.6f}")


tuned_full_lgb = lgb.train(full_params, lgb.Dataset(X_pruned, label=y, categorical_feature=cat_feats), num_boost_round=500)
lgb_test_preds = tuned_full_lgb.predict(test_eng_pruned, num_iteration=tuned_full_lgb.best_iteration)

full_dtrain = xgb.DMatrix(X_pruned, label=y)
full_dtest = xgb.DMatrix(test_eng_pruned)
full_xgb = xgb.train(xgb_params, full_dtrain, num_boost_round=500)  # No evals, full rounds
xgb_test_preds = full_xgb.predict(full_dtest, iteration_range=(0, 500))  # Full 500

ensemble_test_preds = (lgb_test_preds + xgb_test_preds) / 2
sub_ensemble = pd.DataFrame({'id': test_id, 'loan_paid_back': ensemble_test_preds})
print("\nEnsemble sub head:\n", sub_ensemble.head())
sub_ensemble.to_csv('submission.csv', index=False)  
print("Submission saved as 'submission.csv'")

