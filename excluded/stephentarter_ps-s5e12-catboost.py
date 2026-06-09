%load_ext autoreload
%autoreload 2
    
import os
import gc
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
import catboost as cb
from catboost import Pool
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


# Define which strategies to use for the CatBoost model here:
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
feature_engineer = FeatureFactory(strategies=fe_strategies, target=TARGET, seed=seed)

# Fit_transform on training, transform on test
X_full = feature_engineer.fit_transform(training_df)
X_test = feature_engineer.transform(test_df)

X = X_full.drop(TARGET, axis=1, errors='ignore')

print(f'Old Feature Count: {training_df.shape[1] - 1}')
print(f'New Feature Count: {X.shape[1]}')
print(X.head())


def get_cat_features(df, cardinality_threshold=10):
    # Start with explicit string/category columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Add integer columns that act as categories
    # Iterate through numerical columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in num_cols:
        # If a number has very few unique values, treat it as categorical
        if df[col].nunique() <= cardinality_threshold:
            cat_cols.append(col)
            
    return cat_cols

cat_features = get_cat_features(X)
print(f'Detected Categoricals: {cat_features}')

print('Sanitizing categorical features...')

# Fill NaNs with a generic placeholder (optional, but safer)
# and convert to string to ensure no floats remain.
for col in cat_features:
    # Ensure the column is in the dataframe (sanity check)
    if col in X.columns:
        # Check if it's float or object with mixed types
        # We force it to string. This converts 2.0 -> "2.0" and NaN -> "nan"
        X[col] = X[col].astype(str)
        
        # We must apply the exact same transformation to X_test
        X_test[col] = X_test[col].astype(str)
            
print('Categoricals converted to strings. Floats eliminated.')


from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.model_selection import train_test_split

n_jobs = 1 if USE_GPU else -1

if PERFORM_RFE:    
    print('Running Recursive Feature Elimination (this will take a few minutes)...')
    
    # Define the model
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='AUC',
        cat_features=cat_features, # Pass your list of names or indices
        verbose=200,
        random_state=seed,
        early_stopping_rounds=50
    )
    
    # CREATE A SPLIT FOR FEATURE SELECTION
    # We need a validation set so CatBoost knows which features generalize well
    X_fs_train, X_fs_val, y_fs_train, y_fs_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Create CatBoost Pools (required for feature selection)
    train_pool = Pool(X_fs_train, y_fs_train, cat_features=cat_features)
    val_pool = Pool(X_fs_val, y_fs_val, cat_features=cat_features)   
    
    # Run Native Feature Selection
    summary = model.select_features(
        train_pool,
        eval_set=val_pool,
        features_for_select=train_pool.get_feature_names(), # Pass the list of column names
        num_features_to_select=25,                          # How many features you want to keep
        steps=1,                                            # How many elimination rounds
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=False,                            # We just want the list, we'll retrain later
        plot=False
    )
    
    # View Selected Features
    # Save the selected features for the next step
    optimal_cols = summary['selected_features_names']
    print('Selected Features:', optimal_cols)
else:
    print('Using columns from an earlier RFE as optimal features.')
    optimal_cols = pd.Index([
        'age',
        'physical_activity_minutes_per_week',
        'diet_score',
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
        'family_history_diabetes',
        'cardiovascular_history', 
        'non_hdl_cholesterol',
        'vai_proxy',
        'lap_proxy',
        'bmi_class',
        'age_bmi_interaction',
        'sedentary_ratio',
        'cholesterol_risk_ratio', 
        'triglycerides_log',
        'screen_time_log',
        'bmi_cohort_mean'
    ])
    
print('\nOptimal Features:')
print(optimal_cols)


# A split in the data has been described here: https://www.kaggle.com/code/masayakawamata/s5e12-xgb-bridging-the-cv-lb-gap
# This is the index identified above; we'll use it to split the data
ORIGINAL_START_INDEX = 678260

# Filter the datasets to contain just the optimal features
X_rfe = X[optimal_cols]

# Determine the cat_features for the new dataset
cat_features = get_cat_features(X_rfe)

# Ensure optimal_cols is a standard list for filtering
# (Handling cases where it might be a pandas Index or numpy array)
cols_to_keep = list(optimal_cols) if not isinstance(optimal_cols, list) else optimal_cols
    
def objective(trial):
    # Explicit garbage collection at start of trial
    gc.collect()
    
    # Suggest critical categorical parameters first to control logic flow
    bootstrap_type = 'Bernoulli'   # trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'MVS'])
    grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
    
    param = {
        # Core Structure
        'iterations': 8000,              # Equiv to n_estimators
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        
        # STRUCTURAL PARAMS
        'grow_policy': grow_policy,
        'depth': trial.suggest_int('depth', 4, 10),
        
        # REGULARIZATION
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-1, 10.0, log=True),

        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 3.0),
        
        # BOOTSTRAPPING
        'bootstrap_type': bootstrap_type,
    
        # CATEGORICALS
        # Important: Dictates when to switch to One-Hot vs Target Encoding
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 10),
        
        # GENERAL
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        
        # Boilerplate
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'GPU' if USE_GPU else 'CPU',
        'random_seed': seed,
        'early_stopping_rounds': 200,
        'thread_count': n_jobs,
        'verbose': False,
        'allow_writing_files': False,        # Stop it from spamming drive with log folders
    }

    if grow_policy == 'Lossguide':
        param['max_leaves'] = trial.suggest_int('max_leaves', 16, 64)
        
    if param['task_type'] == 'GPU':
        param['metric_period'] = 100

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
        
        # Ensure categorical columns are strings (CatBoost-friendly)
        if cat_cols:
            for c in cat_cols:
                X_train_trans[c] = X_train_trans[c].astype(str)
                X_val_trans[c]   = X_val_trans[c].astype(str)

        # Create CatBoost Pools and immediately delete the pandas DataFrames to free RAM
        train_pool = Pool(X_train_trans, y_train_fold, cat_features=cat_cols, weight=train_weights)
        val_pool = Pool(X_val_trans, y_val_fold, cat_features=cat_cols)

        # Model with the current trial's params
        model = cb.CatBoostClassifier(**param, cat_features=cat_cols)
        
        model.fit(
            train_pool,
            eval_set=[val_pool],
            verbose=False,
            use_best_model=True,
        )
        
        # Store OOF predictions (Only for the validation rows)
        val_probs = model.predict_proba(X_val_trans)[:, 1]
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
        study_name='cb_diabetes_prediction_optuna', 
        direction='maximize',
        sampler=sampler
    )
    study.optimize(objective, n_trials=40, n_jobs=1, gc_after_trial=True, show_progress_bar=True)

    print('Best Params:', study.best_params)
    print('Best AUC:', study.best_value)

    best_params = study.best_params
else:
    # Values from earlier tuning
    best_params = {
        'grow_policy': 'Lossguide',
        'learning_rate': 0.021548495022782603,
        'depth': 8,
        'l2_leaf_reg': 5.457296004538218,
        'random_strength': 0.6497444068426594,
        'scale_pos_weight': 1.0017466334645033,
        'one_hot_max_size': 5,
        'subsample': 0.7616606176479994,
        'min_data_in_leaf': 1,
        'max_leaves': 23
    }


# Cross-Validation Training Loop

cb_tuned_params = {
    **best_params,
    'iterations': 8000,
    'eval_metric': 'AUC',
    'task_type': 'CPU',  # We must use CPU if we want both training and validation numbers
    'random_state': seed,
    'verbose': 200,
    'bootstrap_type': 'Bernoulli',
    'allow_writing_files': False,        # Stop it from spamming drive with log folders
    'early_stopping_rounds': 200,
}

print('Final CatBoost CV Training with OOF / Test predictions')
print('======================================================')
print('Parameters used for training:')
print(cb_tuned_params)

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

    # Ensure categorical columns are strings (CatBoost-friendly)
    if cat_cols:
        for c in cat_cols:
            X_train_trans[c] = X_train_trans[c].astype(str)
            X_val_trans[c]   = X_val_trans[c].astype(str)
            X_test_trans[c]   = X_test_trans[c].astype(str)

    # Create CatBoost Pools and immediately delete the pandas DataFrames to free RAM
    train_pool = Pool(X_train_trans, y_train_fold, cat_features=cat_cols, weight=train_weights)
    val_pool = Pool(X_val_trans, y_val_fold, cat_features=cat_cols)
    test_pool = Pool(X_test_trans, cat_features=cat_cols)

    # Transform Datakage)
    # Note: We are fitting the preprocessor every fold. 
    model = cb.CatBoostClassifier(**cb_tuned_params, cat_features=cat_cols)
    
    # Train Model
    model.fit(
        train_pool,
        # CatBoost automatically tracks metrics on the training set (learn)
        # We pass validation set to track generalization
        eval_set=[train_pool, val_pool],
        verbose=100,
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    # Store Results
    eval_results.append(model.get_evals_result()) # Different method name than XGB!
    models.append(model)    

    # Store OOF predictions (Only for the validation rows)
    val_probs = model.predict_proba(val_pool)[:, 1]
    oof_preds[final_val_idx] = val_probs
    
    # Score this fold
    fold_auc = roc_auc_score(val_pool.get_label(), val_probs)
    print(f"Fold {fold+1} Original-Data AUC: {fold_auc:.5f} (best_iteration={model.get_best_iteration()})")
    
    # Predict Test Set (Accumulate for averaging)
    test_probs = model.predict_proba(test_pool)[:, 1]
    test_preds += test_probs / n_folds

# Final Metric Calculation
# ONLY calculate score on the original data indices
final_valid_mask = oof_preds != 0 # 
original_data_auc = roc_auc_score(y[original_mask], oof_preds[original_mask])
print(f"\nOverall CV AUC (Original Data Only): {original_data_auc:.5f}")


mviz = ModelVisualizer(model_name='CatBoost')

mviz.plot_learning_curves(eval_results, metric='Logloss')


mviz.plot_feature_importance(models, show_values=True)


mviz.plot_roc_curve(y[original_mask], oof_preds[original_mask])


submission_df = helper.read_sample_submission_dataset()
submission_df[TARGET] = test_preds

print('SAMPLE SUBMISSION')
print('=================')

print(submission_df.head(10))


submission_df.to_csv('submission.csv', index=False)
print('Saved: submission.csv')


# Ensure directory exists
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True) 

# Save OOF predictions and raw Test predictions.
# We will load these files in a separate "Blending Notebook".
oof_df = pd.DataFrame({'id': training_df.index, 'pred_cb': oof_preds, 'target': y})
oof_df.to_csv(f'{output_dir}/cb_oof_preds.csv', index=False)

test_pred_df = pd.DataFrame({'id': test_df.index, 'pred_cb': test_preds})
test_pred_df.to_csv(f'{output_dir}/cb_test_preds.csv', index=False)

print(f'Saved for Blending: {output_dir}/cb_oof_preds.csv, {output_dir}/cb_test_preds.csv')




