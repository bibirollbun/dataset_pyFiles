%load_ext autoreload
%autoreload 2
    
import os
import sys
import math
import random
import warnings
from pathlib import Path
from typing import Iterable
from IPython.display import display, Markdown, IFrame

# --- Third-party
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler # Import this
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import seaborn as sns

from diabetes_preprocessing import FeatureFactory
from model_visualizer import ModelVisualizer
from experiment_setup import ExperimentSetup

# --- Notebook settings
warnings.filterwarnings('ignore')

%matplotlib inline


helper = ExperimentSetup()

# Get the seed, and apply it to all the internals like pandas and numpy
seed = helper.set_seeds()

helper.configure_pandas()
helper.suppress_warnings()

# The target feature
TARGET = 'diagnosed_diabetes'

# Show GPU be utilized in training
USE_GPU = False

# Recursive Feature Elimination should be performed in the pipeline.
PERFORM_RFE = False

# Optuna tuning should be performed in the pipeline
PERFORM_OPTUNA_TUNING = False


training_df = helper.read_training_dataset()


test_df = helper.read_test_dataset()


# Define which strategies to use for the XGBoost model here:
fe_strategies = [
    'drop_id',
    'ordinal_encoding',
    'medical_metrics',
    'clinical_indices',
    'interactions',
    'ratios'
]


print('Performing initial feature engineering.')

# Get the target before we start messing with features
y = training_df[TARGET]

# Initialize and apply the feature factory
feature_engineer = FeatureFactory(strategies=fe_strategies, target=TARGET, seed=seed)

# Fit_transform on training, transform on test
X_full = feature_engineer.fit_transform(training_df)
X_test = feature_engineer.transform(test_df)

X = X_full.drop(TARGET, axis=1, errors='ignore')

print(f'Old Feature Count: {training_df.shape[1] - 1}')
print(f'New Feature Count: {X.shape[1]}')
print(X.head())


n_jobs = 1 if USE_GPU else -1

# Convert object columns to 'category' dtype
X_rfe = X.copy()

# Identify string/object columns
cat_cols = X_rfe.select_dtypes(include=['object', 'category']).columns.tolist()

if cat_cols:
    # Use OrdinalEncoder to turn "Female" -> 0, "Male" -> 1
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_rfe[cat_cols] = oe.fit_transform(X_rfe[cat_cols])
    
    # Ensure they are cast to integer type for XGBoost
    # We cast to 'category' type so XGBoost knows to treat them as discrete
    for col in cat_cols:
        X_rfe[col] = X_rfe[col].astype('category')
        
if PERFORM_RFE:    
    # Suppress the specific XGBoost warning to clean up the output
    warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
    
    # Use a "stfe" set of parameters
    rfe_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,          # Standard starting depth
        'colsample_bytree': 0.8, # Allow it to see most features
        'reg_alpha': 0.1,        # Light regularization only
        'device': 'cuda',
        'tree_method': 'hist',
        'enable_categorical': True
    }
    
    # 'gpu_hist' is removed. Now we use 'hist' + device='cuda'
    if USE_GPU:
        rfe_params['device'] = 'cuda'
    else:
        rfe_params['device'] = 'cpu'
    
    # Setup Model with Best Params
    # We use the exact same params that found the signal in the noise
    clf = xgb.XGBClassifier(**rfe_params)

    # Initialize RFECV
    # step=1: remove 1 feature at a time (most precise)
    # min_features_to_select=10: Don't go below 10 features
    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        scoring='roc_auc',
        min_features_to_select=15,
        n_jobs=n_jobs,
        verbose=0
    )
    
    print('Running Recursive Feature Elimination (this will take a few minutes)...')
    rfecv.fit(X_rfe, y)
    
    # Results
    print(f'Optimal number of features: {rfecv.n_features_}')
    
    # Plot Performance vs Number of Features
    # Note: usage of cv_results_ depends on sklearn version, this is for recent versions
    n_scores = len(rfecv.cv_results_['mean_test_score'])
    plt.figure(figsize=(10, 6))
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross validation score (AUC)')
    plt.plot(
        range(10, 10 + n_scores),
        rfecv.cv_results_['mean_test_score']
    )
    plt.show()

    # Save the selected features for the next step
    optimal_cols = X_rfe.columns[rfecv.support_]
else:
    print('Using all columns as optimal features.')
    optimal_cols = X_rfe.columns
    
print('\nOptimal Features:')
print(optimal_cols.tolist())


# A split in the data has been described here: https://www.kaggle.com/code/masayakawamata/s5e12-xgb-bridging-the-cv-lb-gap
# This is the index identified above; we'll use it to split the data
ORIGINAL_START_INDEX = 678260

# Determine the ratio of true/false in the target for use in scale_pos_weight below
neg = (y == 0).sum()
pos = (y == 1).sum()
base_ratio = neg / pos
print(f'Ratio of neg/pos in the target is {base_ratio}')

def objective(trial):
    
    # Suggest Parameters
    param = {
        # Core Booster Params
        'n_estimators': 10000,              # Give it room to run, let early stopping cut it
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 5.0, 50.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        
        # Regularization (Expanded ranges)
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),  # L1 Reg
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2 Reg
        'gamma': trial.suggest_float('gamma', 0.1, 5.0),
        
        # Tree Structure & Imbalance
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        # Helps with class imbalance (approx ratio of negatives/positives)
        'scale_pos_weight': trial.suggest_float(
            'scale_pos_weight',
            0.5 * base_ratio,
            1.5 * base_ratio
        ),  
        
        # Boilerplate
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist' if USE_GPU else 'hist',
        'enable_categorical': True,
        'early_stopping_rounds': 50,
        'random_state': seed,
        'n_jobs': n_jobs
    }

    # Ensure optimal_cols is a standard list for filtering
    # (Handling cases where it might be a pandas Index or numpy array)
    cols_to_keep = list(optimal_cols) if not isinstance(optimal_cols, list) else optimal_cols
    
    # Separate Synthetic and Original Indices
    synthetic_mask = training_df.index < ORIGINAL_START_INDEX
    original_mask = training_df.index >= ORIGINAL_START_INDEX
    
    # Indices for synthetic data (Always used in Training)
    synth_indices = training_df[synthetic_mask].index.to_numpy()
    
    # Indices for original data (Used for CV splitting)
    orig_indices = training_df[original_mask].index.to_numpy()
    orig_targets = y[original_mask] # Needed for stratification
    
    # Initialize CV on Original Data ONLY
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(y)) # Keep full size for compatibility, but only fill original
    test_preds = np.zeros(len(test_df))

    # Custom Loop
    # We split ONLY the original indices
    for fold, (train_idx_orig_relative, val_idx_orig_relative) in enumerate(skf.split(orig_indices, orig_targets)):
        
        # Map relative indices back to global dataframe indices
        global_train_orig_idx = orig_indices[train_idx_orig_relative]
        global_val_orig_idx = orig_indices[val_idx_orig_relative]
        
        # CONSTRUCT TRAINING SET: All Synthetic + (K-1) Original Folds
        final_train_idx = np.concatenate([synth_indices, global_train_orig_idx])
        
        # CONSTRUCT VALIDATION SET: Just the K-th Original Fold
        final_val_idx = global_val_orig_idx
        
        # Fill fold for training and validation using the indices we just constructed.
        X_train_fold = training_df.iloc[final_train_idx]
        y_train_fold = y.iloc[final_train_idx]
        
        X_val_fold = training_df.iloc[final_val_idx]
        y_val_fold = y.iloc[final_val_idx]
        
        # Feature Engineering (Per fold to prevent leakage)
        fe_fold = FeatureFactory(strategies=fe_strategies, target=TARGET, seed=seed)
        
        X_train_trans = fe_fold.fit_transform(X_train_fold)
        X_val_trans = fe_fold.transform(X_val_fold)

        # Create Sample Weights. Initialize weights as 1.0
        train_weights = np.ones(len(X_train_fold))
        
        # Identify indices that belong to the Original Dataset within the current training fold
        num_synth = len(synth_indices)
        
        # Everything after num_synth is Original data. Assign higher weight (e.g., 2.0 or 3.0)
        train_weights[num_synth:] = 2.0

        # Restrict to optimal column set
        if cols_to_keep is not None:
            final_cols = [c for c in cols_to_keep if c in X_train_trans.columns]
            if not final_cols:
                # If this ever happens in tuning, you'd rather fail fast
                raise RuntimeError("No overlap between optimal_cols and fold feature columns.")
            X_train_trans = X_train_trans[final_cols]
            X_val_trans   = X_val_trans[final_cols]
        
        # Identify categorical columns for CatBoost
        cat_cols = X_train_trans.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Ensure categorical columns are type 'category' (XGBoost-friendly)
        if cat_cols:
            for c in cat_cols:
                X_train_trans[c] = X_train_trans[c].astype('category')
                X_val_trans[c]   = X_val_trans[c].astype('category')
            
        # Create model
        model = xgb.XGBClassifier(**param)

        # Train the model
        model.fit(
            X_train_trans, y_train_fold,
            sample_weight=train_weights,
            eval_set=[(X_val_trans, y_val_fold), (X_train_trans, y_train_fold)],
            verbose=False
        )
        
        # Store OOF predictions (Only for the validation rows)
        val_probs = model.predict_proba(
            X_val_trans,
            iteration_range=(0, model.best_iteration + 1)
        )[:, 1]
        oof_preds[final_val_idx] = val_probs
        
        # Score this fold
        fold_auc = roc_auc_score(y_val_fold, val_probs)

    # Final Metric Calculation
    # ONLY calculate score on the original data indices
    final_valid_mask = oof_preds != 0 
    original_data_auc = roc_auc_score(y[original_mask], oof_preds[original_mask])
    
    return np.mean(original_data_auc)


if PERFORM_OPTUNA_TUNING:
    # Run Optimization
    sampler = TPESampler(seed=seed)
    
    study = optuna.create_study(
        study_name='xgb_diabetes_prediction_optuna', 
        direction='maximize',
        sampler=sampler
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print('Best Params:', study.best_params)
    print('Best AUC:', study.best_value)

    best_params = study.best_params
else:
    # Values from earlier tuning
    best_params = {
        'learning_rate': 0.015850445694934434,
        'max_depth': 3,
        'min_child_weight': 16.797147341558652,
        'subsample': 0.7750686010058031,
        'colsample_bytree': 0.5812599321332268,
        'reg_alpha': 8.702466449360665,
        'reg_lambda': 2.17242599189163,
        'gamma': 2.714193077761068,
        'grow_policy': 'lossguide',
        'scale_pos_weight': 0.8639196242743161
    }


# Cross-Validation Training Loop

xgb_tuned_params = {
    **best_params,
    'n_estimators': 12000,    # High iterations have been observed; giving it room
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': n_jobs,
    'random_state': seed,
    'early_stopping_rounds': 50,
    'tree_method': 'gpu_hist' if USE_GPU else None,
    'enable_categorical': True,
}

print('Final XGBoost CV Training with OOF / Test predictions')
print('======================================================')
print('Parameters used for training:')
print(xgb_tuned_params)

# Ensure optimal_cols is a standard list for filtering
# (Handling cases where it might be a pandas Index or numpy array)
cols_to_keep = list(optimal_cols) if not isinstance(optimal_cols, list) else optimal_cols

# Separate Synthetic and Original Indices
synthetic_mask = training_df.index < ORIGINAL_START_INDEX
original_mask = training_df.index >= ORIGINAL_START_INDEX

# Indices for synthetic data (Always used in Training)
synth_indices = training_df[synthetic_mask].index.to_numpy()

# Indices for original data (Used for CV splitting)
orig_indices = training_df[original_mask].index.to_numpy()
orig_targets = y[original_mask] # Needed for stratification

# Initialize CV on Original Data ONLY
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

print(f"Training on {len(synth_indices)} synthetic rows + Original Data Folds")
print(f"Validating strictly on {len(orig_indices)} original rows")
print(' ')

# Storage for plotting later
eval_results = []
models = []

# Arrays to store results
oof_preds = np.zeros(len(y)) # Keep full size for compatibility, but only fill original
test_preds = np.zeros(len(test_df))

# Custom Loop
# We split ONLY the original indices
for fold, (train_idx_orig_relative, val_idx_orig_relative) in enumerate(skf.split(orig_indices, orig_targets)):

    print(f"Starting processing of fold {fold+1}")

    # Map relative indices back to global dataframe indices
    global_train_orig_idx = orig_indices[train_idx_orig_relative]
    global_val_orig_idx = orig_indices[val_idx_orig_relative]
    
    # CONSTRUCT TRAINING SET: All Synthetic + (K-1) Original Folds
    final_train_idx = np.concatenate([synth_indices, global_train_orig_idx])
    
    # CONSTRUCT VALIDATION SET: Just the K-th Original Fold
    final_val_idx = global_val_orig_idx
    
    # Fill fold for training and validation using the indices we just constructed.
    X_train_fold = training_df.iloc[final_train_idx]
    y_train_fold = y.iloc[final_train_idx]
    
    X_val_fold = training_df.iloc[final_val_idx]
    y_val_fold = y.iloc[final_val_idx]
    
    # Feature Engineering (Per fold to prevent leakage)
    fe_fold = FeatureFactory(strategies=fe_strategies, target=TARGET, seed=seed)
    
    X_train_trans = fe_fold.fit_transform(X_train_fold)
    X_val_trans = fe_fold.transform(X_val_fold)
    X_test_trans = fe_fold.transform(test_df)
    
    # Create Sample Weights. Initialize weights as 1.0
    train_weights = np.ones(len(X_train_fold))
    
    # Identify indices that belong to the Original Dataset within the current training fold
    num_synth = len(synth_indices)
    
    # Everything after num_synth is Original data. Assign higher weight (e.g., 2.0 or 3.0)
    train_weights[num_synth:] = 2.0

    # Restrict to optimal column set
    if cols_to_keep is not None:
        final_cols = [c for c in cols_to_keep if c in X_train_trans.columns]
        if not final_cols:
            raise RuntimeError("No overlap between optimal_cols and fold feature columns.")
        X_train_trans = X_train_trans[final_cols]
        X_val_trans   = X_val_trans[final_cols]
        X_test_trans  = X_test_trans[final_cols]

    cat_cols = X_train_trans.select_dtypes(include=['object', 'category']).columns.tolist()

    if cat_cols:
        # Use OrdinalEncoder to turn "Female" -> 0, "Male" -> 1
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_trans[cat_cols] = oe.fit_transform(X_train_trans[cat_cols])
        X_val_trans[cat_cols] = oe.transform(X_val_trans[cat_cols])
        X_test_trans[cat_cols] = oe.transform(X_test_trans[cat_cols])

        # Ensure they are cast to integer type for XGBoost
        # We cast to 'category' type so XGBoost knows to treat them as discrete
        for col in cat_cols:
            X_train_trans[col] = X_train_trans[col].astype('category')
            X_val_trans[col] = X_val_trans[col].astype('category')
            X_test_trans[col] = X_test_trans[col].astype('category')
        
    # Create model
    # Note: We are fitting the preprocessor every fold. 
    model = xgb.XGBClassifier(**xgb_tuned_params)
    
    # Train Model
    # The pipeline will call feature_factory.fit_transform(X_train)
    # and then pass that result to xgb.fit()
    model.fit(
        X_train_trans, y_train_fold,
        sample_weight=train_weights,
        eval_set=[(X_val_trans, y_val_fold), (X_train_trans, y_train_fold)],
        verbose=100
    )
    
    # Store history for this fold
    eval_results.append(model.evals_result())
    models.append(model)
    
    # Store OOF predictions (Only for the validation rows)
    val_probs = model.predict_proba(
        X_val_trans,
        iteration_range=(0, model.best_iteration + 1)
    )[:, 1]
    oof_preds[final_val_idx] = val_probs
    
    # Score this fold
    fold_auc = roc_auc_score(y_val_fold, val_probs)
    print(f"Fold {fold+1} Original-Data AUC: {fold_auc:.5f} (best_iteration={model.best_iteration})")
    
    # Accumulate Test Preds
    test_probs = model.predict_proba(
        X_test_trans,
        iteration_range=(0, model.best_iteration + 1)
    )[:, 1]
    test_preds += test_probs / 5 # n_splits
    
# Final Metric Calculation
# ONLY calculate score on the original data indices
final_valid_mask = oof_preds != 0 # 
original_data_auc = roc_auc_score(y[original_mask], oof_preds[original_mask])
print(f"\nOverall CV AUC (Original Data Only): {original_data_auc:.5f}")    


xgb_tuned_params


mviz = ModelVisualizer(model_name='XGBoost')

mviz.plot_learning_curves(eval_results)


mviz.plot_feature_importance(models, show_values=True)


mviz.plot_roc_curve(y[original_mask], oof_preds[original_mask])


submission_df = helper.read_sample_submission_dataset()
submission_df[TARGET] = test_preds

print('SUBMISSION')
print('==========')

print(submission_df.head(10))


submission_df.to_csv('submission.csv', index=False)
print('Saved: submission.csv')


# Ensure directory exists
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True) 

# Save OOF predictions and raw Test predictions.
# We will load these files in a separate "Blending Notebook".
oof_df = pd.DataFrame({'id': training_df.index, 'pred_xgb': oof_preds, 'target': y})
oof_df.to_csv(f'{output_dir}/xgb_oof_preds.csv', index=False)

test_pred_df = pd.DataFrame({'id': test_df.index, 'pred_xgb': test_preds})
test_pred_df.to_csv(f'{output_dir}/xgb_test_preds.csv', index=False)

print(f'Saved for Blending: {output_dir}/xgb_oof_preds.csv, {output_dir}/xgb_test_preds.csv')




