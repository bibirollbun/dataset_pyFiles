%load_ext autoreload
%autoreload 2
    
%pip install --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON lightgbm


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
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler # Import this
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import seaborn as sns

from diabetes_preprocessing import FeatureFactory
from model_visualizer import ModelVisualizer
from experiment_setup import ExperimentSetup

# --- Notebook settings
warnings.filterwarnings('ignore')

%matplotlib inline

import ipywidgets as widgets
widgets.IntProgress()


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


# Define which strategies to use for the LightGBM model here:
fe_strategies = [
    'drop_id',
    'ordinal_encoding',
    'medical_metrics',
    'clinical_indices',
    'interactions',
    'ratios',
    'log',
    'binning',
    'cohort_deviations',
    'clustering'
]


print('Performing initial feature engineering.')

# Get the target before we start messing with features
y = training_df[TARGET]

# Initialize and apply the feature factory
feature_engineer = FeatureFactory(strategies=fe_strategies, target=TARGET)

# Fit_transform on training, transform on test
X_full = feature_engineer.fit_transform(training_df)
X_test = feature_engineer.transform(test_df)

X = X_full.drop(TARGET, axis=1, errors='ignore')

print(f'Old Feature Count: {training_df.shape[1] - 1}')
print(f'New Feature Count: {X.shape[1]}')
print(X.head())


# LightGBM handles CPU parallelism very well, but use 1 job if using GPU to avoid contention
n_jobs = 1 if USE_GPU else -1

X_rfe = X.copy()

if PERFORM_RFE:    
    # Suppress a specific warning to clean up the output
    warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
    
    # Sensible defaults for Feature Selection (RFECV)
    rfe_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        
        # Speed/Capacity Balance
        'n_estimators': 1000, 
        'learning_rate': 0.05,      # Standard robust baseline
        
        # Tree Structure (Standard Defaults)
        'num_leaves': 63,           # Approx depth 6 (2^6 = 64)
        'max_depth': -1,            # Let num_leaves handle complexity
        'min_child_samples': 30,    # Standard guard against leaf overfitting
        
        # Stochastic Components (Good for feature diversity)
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        
        # Regularization (Light, just for stability)
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_split_gain': 0.0,
        
        # Hardware / System
        'random_state': seed,
        'n_jobs': n_jobs,           # 1 if GPU, -1 if CPU
        'verbosity': -1,
        'device': 'gpu' if USE_GPU else 'cpu'
    }
    
    # Set additional parameters if on GPU
    if rfe_params['device'] == 'gpu':
        rfe_params['gpu_platform_id'] =  0
        rfe_params['gpu_device_id'] = 0

    # Setup Model with Best Params
    # We use the exact same params that found the signal in the noise
    clf = lgb.LGBMClassifier(**rfe_params)

    # Convert object columns to 'category' dtype
    cat_cols = X_rfe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        # Use OrdinalEncoder to turn "Female" -> 0, "Male" -> 1
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_rfe[cat_cols] = oe.fit_transform(X_rfe[cat_cols])
        
        # We cast to 'category' type so LightGBM knows to treat them as discrete
        for col in cat_cols:
            X_rfe[col] = X_rfe[col].astype('category')
 
    # Initialize RFECV
    # step=1: remove 1 feature at a time (most precise)
    # min_features_to_select=10: Don't go below 10 features
    rfecv = RFECV(
        estimator=clf,
        step=0.05,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        scoring='roc_auc',
        min_features_to_select=15,
        n_jobs=n_jobs,
        verbose=1
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
    optimal_cols = X.columns[rfecv.support_]
else:
    print('Using columns from earlier RFE as optimal features.')
    optimal_cols = pd.Index([
        'age',
        'alcohol_consumption_per_week',
        'physical_activity_minutes_per_week',
        'diet_score',
        'sleep_hours_per_day',
        'screen_time_hours_per_day',
        'bmi',
        'waist_to_hip_ratio',
        'systolic_bp',
        'diastolic_bp',
        'heart_rate',
        'cholesterol_total',
        'hdl_cholesterol',
        'ldl_cholesterol',
        'triglycerides',
        'gender',
        'ethnicity',
        'education_level',
        'income_level',
        'smoking_status',
        'employment_status',
        'family_history_diabetes',
        'hypertension_history',
        'cardiovascular_history',
        'education_level_ord',
        'income_level_ord',
        'pulse_pressure',
        'mean_arterial_pressure',
        'non_hdl_cholesterol',
        'vai_proxy',
        'lap_proxy',
        'age_bmi_interaction',
        'sedentary_ratio',
        'cholesterol_risk_ratio',
        'bp_ratio',
        'whr_bmi_product',
        'triglycerides_log',
        'screen_time_log',
        'cluster_label',
        'age_decile',
        'bmi_cohort_mean',
        'bmi_dev_from_cohort'
    ])
    
print('\nOptimal Features:')
print(optimal_cols.tolist())


# A split in the data has been described here: https://www.kaggle.com/code/masayakawamata/s5e12-xgb-bridging-the-cv-lb-gap
# This is the index identified above; we'll use it to split the data
ORIGINAL_START_INDEX = 678260

def objective(trial):
    
    # Suggest Parameters
    params = {
        'n_estimators': 2000,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
        
        # Key LightGBM Parameters
        'num_leaves': trial.suggest_int('num_leaves', 30, 127),  # Main complexity control
        'max_depth': -1,  # Let the tree grow or not based on num_leaves and min_child_samples
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),

        # Randomly choose between standard GBDT splits (False) and Extra Trees splits (True)
        # An 'extra_trees' trial never wins, so stop wasting trials on it
        # 'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
        
        # Class Imbalance
        # This helps significantly with AUC on imbalanced data
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 3.0),

        # Sampling & Regularization
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        
        # Regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),   # Equiv to XGB 'gamma'
        'path_smooth': trial.suggest_float('path_smooth', 0.0, 5.0),    
        'max_bin': 127,
        
        # Boilerplate
        'objective': 'binary',
        'metric': 'auc',
        'device': 'gpu' if USE_GPU else 'cpu',
        'random_state': seed,
        'n_jobs': -1,
        'subsample_freq': 5, # Resample less frequently
        'path_smooth': 0.1,  # Helps smoothing out leaves with few samples
        'verbosity': -1
    }

    # Set additional parameters if on GPU
    if params['device'] == 'gpu':
        params['gpu_platform_id'] =  0
        params['gpu_device_id'] = 0

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
                raise RuntimeError('No overlap between optimal_cols and fold feature columns.')
            X_train_trans = X_train_trans[final_cols]
            X_val_trans   = X_val_trans[final_cols]

        cat_cols = X_train_trans.select_dtypes(include=['object', 'category']).columns.tolist()
    
        if cat_cols:
            # Use OrdinalEncoder to turn "Female" -> 0, "Male" -> 1
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_trans[cat_cols] = oe.fit_transform(X_train_trans[cat_cols])
            X_val_trans[cat_cols] = oe.transform(X_val_trans[cat_cols])
            
            # We cast to 'category' type so LightGBM knows to treat them as discrete
            for col in cat_cols:
                X_train_trans[col] = X_train_trans[col].astype('category')
                X_val_trans[col] = X_val_trans[col].astype('category')

        # Encoding and Model Fitting
        model = lgb.LGBMClassifier(**params)
        
        model.fit(
            X_train_trans, y_train_fold,
            eval_set=[(X_val_trans, y_val_fold)],
            eval_metric='auc',
            sample_weight=train_weights,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0) # Set to 0 to reduce noise
            ]
        )
        
        # Store OOF predictions (Only for the validation rows)
        val_probs = model.predict_proba(
            X_val_trans,
            iteration_range=(0, model.best_iteration_ + 1)
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
        study_name='lgb_diabetes_prediction_optuna', 
        direction='maximize',
        sampler=sampler
    )
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    print('Best Params:', study.best_params)
    print('Best AUC:', study.best_value)

    best_params = study.best_params
else:
    # Values from earlier tuning
    best_params = {
        'learning_rate': 0.03434731425229975,
        'num_leaves': 39,
        'min_child_samples': 46,
        'scale_pos_weight': 1.3277022491204855,
        'subsample': 0.8078229214647135,
        'colsample_bytree': 0.5398282081958882,
        'reg_alpha': 0.0010237834621212593,
        'reg_lambda': 0.021165236937808807,
        'min_split_gain': 0.34917008259188265,
        'path_smooth': 0.6210959321120694
    }


# Build LGB parameters from best_params
lgb_tuned_params = {
    **best_params,
    'n_estimators': 8000,
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_jobs': n_jobs,
    'random_state': seed,
    'device_type': 'gpu' if USE_GPU else 'cpu',
    'verbosity': -1,
}

print('Final LightGBM CV Training with OOF / Test predictions')
print('======================================================')
print('Parameters used for training:')
print(lgb_tuned_params)
print(' ')

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

print(f'Training on {len(synth_indices)} synthetic rows + Original Data Folds')
print(f'Validating strictly on {len(orig_indices)} original rows')
print(' ')

# Prepare storage
eval_results = []
models = []

oof_preds = np.zeros(len(y)) # Keep full size for compatibility, but only fill original
test_preds = np.zeros(len(test_df))

# Custom Loop
# We split ONLY the original indices
for fold, (train_idx_orig_relative, val_idx_orig_relative) in enumerate(skf.split(orig_indices, orig_targets)):

    print(f'Starting processing of fold {fold+1}')

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
            raise RuntimeError('No overlap between optimal_cols and fold feature columns.')
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

        # We cast to 'category' type so LightGBM knows to treat them as discrete
        for col in cat_cols:
            X_train_trans[col] = X_train_trans[col].astype('category')
            X_val_trans[col] = X_val_trans[col].astype('category')
            X_test_trans[col] = X_test_trans[col].astype('category')

    # Encoding and Model Fitting
    model = lgb.LGBMClassifier(**lgb_tuned_params)
    
    model.fit(
        X_train_trans, y_train_fold,
        eval_set=[(X_val_trans, y_val_fold), (X_train_trans, y_train_fold)],
        eval_metric='auc',
        sample_weight=train_weights,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    eval_results.append(model.evals_result_)
    models.append(model)
    
    # Store OOF predictions (Only for the validation rows)
    val_probs = model.predict_proba(
        X_val_trans,
        iteration_range=(0, model.best_iteration_ + 1)
    )[:, 1]
    oof_preds[final_val_idx] = val_probs
    
    # Score this fold
    fold_auc = roc_auc_score(y_val_fold, val_probs)
    print(f'Fold {fold+1} Original-Data AUC: {fold_auc:.5f} (best_iteration={model.best_iteration_})')
    
    # Accumulate Test Preds
    test_probs = model.predict_proba(
        X_test_trans,
        iteration_range=(0, model.best_iteration_ + 1)
    )[:, 1]
    test_preds += test_probs / 5 # n_splits
    
# Final Metric Calculation
# ONLY calculate score on the original data indices
final_valid_mask = oof_preds != 0 # 
original_data_auc = roc_auc_score(y[original_mask], oof_preds[original_mask])
print(f'\nOverall CV AUC (Original Data Only): {original_data_auc:.5f}')


mviz = ModelVisualizer(model_name='LightGBM')

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
oof_df = pd.DataFrame({'id': training_df.index, 'pred_lgb': oof_preds, 'target': y})
oof_df.to_csv(f'{output_dir}/lgb_oof_preds.csv', index=False)

test_pred_df = pd.DataFrame({'id': test_df.index, 'pred_lgb': test_preds})
test_pred_df.to_csv(f'{output_dir}/lgb_test_preds.csv', index=False)

print(f'Saved for Blending: {output_dir}/lgb_oof_preds.csv, {output_dir}/lgb_test_preds.csv')




