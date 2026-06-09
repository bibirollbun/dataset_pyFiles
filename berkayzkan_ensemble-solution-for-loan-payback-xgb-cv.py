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
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')

print('Train Shape:', train_df.shape)
print('Test Shape:', test_df.shape)

train = train_df.copy()
test = test_df.copy()


# Define target and categorical columns
TARGET = 'loan_paid_back'  # boolean
CATEGORICAL_COLS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
BASE = [col for col in train_df.columns if col not in ['id', TARGET]]



# Create ORIG Features
ORIG = []

for col in BASE:
    # MEAN
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # COUNT
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)

print(f'âœ… {len(ORIG)} ORIG Features created!')

# Create FEATURES list
FEATURES = BASE + ORIG
print(f'âœ… Total {len(FEATURES)} features (BASE: {len(BASE)} + ORIG: {len(ORIG)})')


# Add numerical_cols to the FEATURES list and remove duplicates
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                  'loan_amount', 'interest_rate']

# Limit outliers
for col in numerical_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)
    test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)

# Add numerical_cols to the FEATURES list
FEATURES = list(set(FEATURES + numerical_cols))  # Use set to remove duplicates
print(f'âœ… FEATURES list updated: {len(FEATURES)} features')


# Display the list of all features
print("List of all features:")
for i, feature in enumerate(FEATURES, 1):
    print(f"{i}. {feature}")


# Preprocessing function (for reuse)
def preprocess_features(df, features, cat_cols, numeric_cols):
    """Prepare categorical and numeric columns"""
    df_processed = df[features].copy()
    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('NA').astype('category')
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(0)
    return df_processed

print('âœ… Preprocessing function defined')


# ====== Split train data ======
from sklearn.model_selection import train_test_split

X = train.drop(columns=[TARGET])
y = train[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
print('âœ… Train-test split completed')


# Prepare data for optimization
numeric_cols_opt = [col for col in FEATURES if col not in CATEGORICAL_COLS]
X_train_split_opt = preprocess_features(X_train, FEATURES, CATEGORICAL_COLS, numeric_cols_opt)
X_val_split_opt = preprocess_features(X_valid, FEATURES, CATEGORICAL_COLS, numeric_cols_opt)
X_test_opt = preprocess_features(test, FEATURES, CATEGORICAL_COLS, numeric_cols_opt)

print('âœ… Data prepared for optimization')


# 2ï¸�âƒ£ HYPERPARAMETER TUNING (Optuna)
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

print('='*70)
print('2ï¸�âƒ£  HYPERPARAMETER TUNING (Optuna)')
print('='*70)

def objective(trial):
    """Optuna objective function"""
    params = {
        'n_estimators': 10000,
        'max_depth': trial.suggest_int('max_depth', 4, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.0095, 0.0105, log=True),
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'random_state': 42,
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'enable_categorical': True,
        'early_stopping_rounds': 50,
    }

    model_trial = XGBClassifier(**params)
    model_trial.fit(
        X_train_split_opt, y_train,
        eval_set=[(X_val_split_opt, y_valid)],
        verbose=False
    )
    
    y_pred = model_trial.predict_proba(X_val_split_opt)[:, 1]
    auc = roc_auc_score(y_valid, y_pred)
    
    return auc

# Optuna study
print('\nğŸ”� Optuna optimization starting...')

study = optuna.create_study(
    direction='maximize', 
    study_name='xgboost_optimization',
    sampler=TPESampler(seed=42)
)

study.optimize(objective, n_trials=5, timeout=3600, show_progress_bar=True)

print(f'\nâœ… Optimization completed!')
print(f'   Best trial: {study.best_trial.number}')
print(f'   Best AUC: {study.best_value:.4f}')
print(f'\nğŸ“Š Best parameters:')
for key, value in study.best_params.items():
    print(f'   {key}: {value}')

# Final model with best parameters
best_params = study.best_params.copy()
best_params.update({
    'n_estimators': 10000,
    'tree_method': 'hist',
    'device': 'cuda',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'enable_categorical': True,
    'early_stopping_rounds': 50,
})

print('\nğŸš€ Training final model with best parameters...')
final_model = XGBClassifier(**best_params)
final_model.fit(
    X_train_split_opt, y_train,
    eval_set=[(X_train_split_opt, y_train), (X_val_split_opt, y_valid)],
    verbose=1000
)

# Predict on test set with tuned model
pred_tuned = final_model.predict_proba(X_test_opt)[:, 1]

submission_tuned = pd.DataFrame({
    "id": test["id"],
    TARGET: pred_tuned
})

submission_tuned.to_csv("submission_xgboost_tuned.csv", index=False)
print(f'\nğŸ“� Tuned submission saved: submission_xgboost_tuned.csv')

# Validation AUC
y_val_pred_tuned = final_model.predict_proba(X_val_split_opt)[:, 1]
val_auc_tuned = roc_auc_score(y_valid, y_val_pred_tuned)
print(f'   Validation AUC: {val_auc_tuned:.4f}')


# 3ï¸�âƒ£ FEATURE SELECTION (Remove low-importance features)
import matplotlib.pyplot as plt

print('='*70)
print('3ï¸�âƒ£  FEATURE SELECTION')
print('='*70)

# Calculate feature importance from the best model (use tuned or early stopping model)
# Here, we use the final_model (tuned)
feature_importance_sel = pd.DataFrame({
    'feature': X_train_split_opt.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nğŸ“Š Feature Importance Statistics:')
print(feature_importance_sel['importance'].describe())

# Identify low-importance features
# Threshold: features below the median or with very low importance
threshold = feature_importance_sel['importance'].quantile(0.25)  # Bottom 25%
low_importance_features = feature_importance_sel[feature_importance_sel['importance'] < threshold]['feature'].tolist()

print(f'\nğŸ—‘ï¸�  {len(low_importance_features)} low-importance features found (importance < {threshold:.2f})')
print(f'   Total number of features: {len(FEATURES)}')
print(f'   Remaining features: {len(FEATURES) - len(low_importance_features)}')

# Display the 10 least important features
print(f'\n   10 least important features:')
for i, row in feature_importance_sel.tail(10).iterrows():
    print(f'     {row["feature"]}: {row["importance"]:.2f}')

# Create new feature set
FEATURES_SELECTED = [f for f in FEATURES if f not in low_importance_features]
numeric_cols_selected = [c for c in FEATURES_SELECTED if c not in CATEGORICAL_COLS]

print(f'\nâœ… New feature set prepared: {len(FEATURES_SELECTED)} features')

# Retrain the model
X_train_selected = preprocess_features(train, FEATURES_SELECTED, CATEGORICAL_COLS, numeric_cols_selected)
X_test_selected = preprocess_features(test, FEATURES_SELECTED, CATEGORICAL_COLS, numeric_cols_selected)

X_train_split_selected = preprocess_features(X_train, FEATURES_SELECTED, CATEGORICAL_COLS, numeric_cols_selected)
X_val_split_selected = preprocess_features(X_valid, FEATURES_SELECTED, CATEGORICAL_COLS, numeric_cols_selected)

print('\nğŸš€ Training model with selected features...')
# Override early_stopping_rounds to 200 for longer training
params_selected = best_params.copy()
params_selected['early_stopping_rounds'] = 200
params_selected['n_estimators'] = 10000

model_selected = XGBClassifier(**params_selected)


model_selected.fit(
    X_train_split_selected, y_train,
    eval_set=[(X_train_split_selected, y_train), (X_val_split_selected, y_valid)],
    verbose=100
)

print(f'\nâœ… Feature selection model training completed!')

# Predict
pred_selected = model_selected.predict_proba(X_test_selected)[:, 1]

submission_selected = pd.DataFrame({
    "id": test["id"],
    TARGET: pred_selected
})

submission_selected.to_csv("submission_xgboost_selected.csv", index=False)
print(f'ğŸ“� Submission saved: submission_xgboost_selected.csv')

# Validation AUC
y_val_pred_selected = model_selected.predict_proba(X_val_split_selected)[:, 1]
val_auc_selected = roc_auc_score(y_valid, y_val_pred_selected)
print(f'   Validation AUC: {val_auc_selected:.4f}')

# Feature reduction comparison
print(f'\nğŸ“‰ Feature reduction results:')
print(f'   Before: {len(FEATURES)} features â†’ After: {len(FEATURES_SELECTED)} features')
print(f'   Reduction: {len(low_importance_features)} features ({100*len(low_importance_features)/len(FEATURES):.1f}%)')


# 4ï¸�âƒ£ CROSS-VALIDATION for Robust Predictions
from sklearn.model_selection import StratifiedKFold
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

print('='*70)
print('4ï¸�âƒ£  K-FOLD CROSS-VALIDATION (5-Fold)')
print('='*70)

# Use best_params if available, otherwise use baseline parameters
try:
    params_to_use = best_params.copy()
    print('âœ… Using Optuna best_params')
except NameError:
    print('âš ï¸�  best_params not found, using baseline parameters')
    params_to_use = {
        'n_estimators': 10000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'random_state': 42,
        'min_child_weight': 89,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'gamma': 0.11,
        'reg_alpha': 1.8,
        'reg_lambda': 5.2,
        'enable_categorical': True,
        'early_stopping_rounds': 100,
    }

n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Array for out-of-fold predictions
oof_predictions = np.zeros(len(train))
test_predictions = np.zeros(len(test))

fold_scores = []

# Select feature set: use FEATURES_SELECTED if available, otherwise use FEATURES
try:
    features_to_use = FEATURES_SELECTED
    numeric_cols_to_use = numeric_cols_selected
    print(f'âœ… Using FEATURES_SELECTED: {len(features_to_use)} features')
except NameError:
    features_to_use = FEATURES
    numeric_cols_to_use = [col for col in FEATURES if col not in CATEGORICAL_COLS]
    print(f'âš ï¸�  FEATURES_SELECTED not found, using all FEATURES: {len(features_to_use)} features')

# Prepare CV data
X_train_cv = preprocess_features(train, features_to_use, CATEGORICAL_COLS, numeric_cols_to_use)
y_train_cv = train[TARGET].astype(int)
X_test_cv = preprocess_features(test, features_to_use, CATEGORICAL_COLS, numeric_cols_to_use)

print(f'âœ… CV data prepared: {X_train_cv.shape}')

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_cv, y_train_cv), 1):
    print(f'\n{"="*50}')
    print(f'ğŸ“‚ Fold {fold}/{n_folds}')
    print(f'{"="*50}')
    
    X_fold_train = X_train_cv.iloc[train_idx]
    y_fold_train = y_train_cv.iloc[train_idx]
    X_fold_val = X_train_cv.iloc[val_idx]
    y_fold_val = y_train_cv.iloc[val_idx]
    
    print(f'   Train: {X_fold_train.shape}, Val: {X_fold_val.shape}')
    
    # Create model
    fold_model = XGBClassifier(**params_to_use)
    
    print(f'   ğŸš€ Training model...')
    fold_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=1000
    )
    
    # Validation predictions
    oof_predictions[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]
    
    # Test predictions (average across folds)
    test_predictions += fold_model.predict_proba(X_test_cv)[:, 1] / n_folds
    
    fold_auc = roc_auc_score(y_fold_val, oof_predictions[val_idx])
    fold_scores.append(fold_auc)
    print(f'   âœ… Fold {fold} AUC: {fold_auc:.4f}')

# Overall CV score
cv_auc = roc_auc_score(y_train_cv, oof_predictions)
cv_std = np.std(fold_scores)

print(f'\n{"="*70}')
print(f'ğŸ“Š CROSS-VALIDATION RESULTS')
print(f'{"="*70}')
print(f'   Overall CV AUC: {cv_auc:.4f}')
print(f'   Std Dev: {cv_std:.4f}')
print(f'   Min Fold AUC: {min(fold_scores):.4f}')
print(f'   Max Fold AUC: {max(fold_scores):.4f}')
print(f'\n   Fold AUC Details:')
for i, score in enumerate(fold_scores, 1):
    print(f'     Fold {i}: {score:.4f}')

# CV submission
submission_cv = pd.DataFrame({
    "id": test["id"],
    TARGET: test_predictions
})

submission_cv.to_csv("submission_xgboost_cv.csv", index=False)
print(f'\nğŸ“� CV submission saved: submission_xgboost_cv.csv')
print(f'   This is the most robust prediction! (5-fold average)')

# Visualize Fold AUCs
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(1, n_folds+1), fold_scores, color='steelblue', alpha=0.7, edgecolor='black')
plt.axhline(y=cv_auc, color='red', linestyle='--', linewidth=2, label=f'Mean AUC: {cv_auc:.4f}')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('AUC Score', fontsize=12)
plt.title('Cross-Validation: Fold-wise AUC Scores', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Load the CSV files
xgboost_csv = pd.read_csv('submission_xgboost_cv.csv')
xgboost_tuned = pd.read_csv('submission_xgboost_tuned.csv')
xgboost_selected = pd.read_csv('submission_xgboost_selected.csv')

# Assuming the CSVs have predictions columns
# Extract prediction columns (adjust column names as needed)
predictions = pd.DataFrame({
    'xgb_cv': xgboost_csv.iloc[:, -1],  # Last column assumed to be predictions
    'xgb_tuned': xgboost_tuned.iloc[:, -1],
    'xgb_selected': xgboost_selected.iloc[:, -1]
})

# Create H-blend using Ridge regression
# If you have true labels, replace y_true with actual target variable
# For demonstration, using simple averaging if no labels available
weights = [0.32, 0.38, 0.30]  # Equal weights for 3 models

# Simple weighted blend
h_blend = (predictions['xgb_cv'] * weights[0] + 
           predictions['xgb_tuned'] * weights[1] + 
           predictions['xgb_selected'] * weights[2])

# Create final submission with h-blend
submission_hblend = pd.DataFrame({
    'id': xgboost_csv['id'],
    'loan_paid_back': h_blend
})

# Save to CSV file
submission_hblend.to_csv('submission.csv', index=False)

print("Ensemble Solution file created successfully!")
print(f"Blend shape: {h_blend.shape}")
print(f"\nModel weights:")
print(f"  XGBoost CV: {weights[0]:.3f}")
print(f"  XGBoost Tuned: {weights[1]:.3f}")
print(f"  XGBoost Selected: {weights[2]:.3f}")
print(f"\nFirst 5 predictions:")
print(submission_hblend.head())

