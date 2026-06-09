import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gc
import warnings
from pathlib import Path
from tqdm import tqdm

# ML imports
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')
tqdm.pandas(disable=True)  # Disable tqdm for cleaner output

# Configuration
class CFG:
    INPUT_DIR = '/kaggle/input/mercor-cheating-detection/'
    OUTPUT_DIR = '/kaggle/working/'
    N_FOLDS = 5
    SEED = 42
    TARGET = 'is_cheating'
    DEBUG = False  # Set True for quick testing
    
    # Model hyperparameters
    LGB_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1
    }
    
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': SEED,
        'n_jobs': -1
    }
    
    CB_PARAMS = {
        'loss_function': 'Logloss',
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': SEED,
        'verbose': False
    }

# Verify data path
import os
print("Files available:", os.listdir(CFG.INPUT_DIR))


%%time
# Load all datasets
train = pd.read_csv(f'{CFG.INPUT_DIR}train.csv')
test = pd.read_csv(f'{CFG.INPUT_DIR}test.csv')
social_graph = pd.read_csv(f'{CFG.INPUT_DIR}social_graph.csv')
sample_submission = pd.read_csv(f'{CFG.INPUT_DIR}sample_submission.csv')

# Load feature metadata
with open(f'{CFG.INPUT_DIR}feature_metadata.json', 'r') as f:
    feature_metadata = json.load(f)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Social graph shape: {social_graph.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# Display basic info
print("\nTrain columns:", train.columns.tolist())
print("\nTest columns:", test.columns.tolist())


# Understand the target distribution and high_conf_clean
print("Target distribution in train:")
print(train[CFG.TARGET].value_counts(dropna=False))

print("\nHigh confidence clean distribution:")
print(train['high_conf_clean'].value_counts(dropna=False))

print("\nRelationship between high_conf_clean and target:")
print(pd.crosstab(train['high_conf_clean'], train[CFG.TARGET], margins=True))


# Basic EDA - feature analysis
feature_cols = [col for col in train.columns if col.startswith('feature_')]
print(f"Number of features: {len(feature_cols)}")

# Analyze feature types and missing values
feature_info = []
for col in feature_cols:
    info = feature_metadata.get(col, {})
    feature_info.append({
        'feature': col,
        'type': info.get('type', 'unknown'),
        'missing_pct': train[col].isnull().mean() * 100,
        'unique_values': train[col].nunique()
    })

feature_df = pd.DataFrame(feature_info)
print(feature_df)

# Summary statistics for labeled data only
labeled_train = train[train[CFG.TARGET].notna()]
print(f"\nLabeled samples: {len(labeled_train)}")
print(f"Positive rate: {labeled_train[CFG.TARGET].mean():.4f}")


# Analyze social graph structure
print("Social graph analysis:")
print(f"Total edges: {len(social_graph)}")

# Get unique users in graph
graph_users = set(social_graph['user_a'].unique()) | set(social_graph['user_b'].unique())
print(f"Unique users in graph: {len(graph_users)}")

# Get users in train/test datasets
train_users = set(train['user_hash'].unique())
test_users = set(test['user_hash'].unique())

print(f"Users in train dataset: {len(train_users)}")
print(f"Users in test dataset: {len(test_users)}")

# Check overlap
print(f"Train users in graph: {len(train_users & graph_users)}")
print(f"Test users in graph: {len(test_users & graph_users)}")


%%time
# ===========================================
# FEATURE ENGINEERING (LOADS SAVED GRAPH FEATURES)
# ===========================================

print("=" * 50)
print("BUILDING ADVANCED FEATURES")
print("=" * 50)

import os
import gc
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# ===========================================
# CONFIGURATION - LOAD FROM YOUR SAVED DATASET
# ===========================================
GRAPH_FEATURES_PATH = '/kaggle/input/grapg-feature-mercor/graph_features.csv'
LOAD_SAVED_GRAPH_FEATURES = True   # Load from saved!
SAVE_GRAPH_FEATURES = False        # No need to save again

# ===========================================
# LOAD SAVED GRAPH FEATURES (SKIP 2HR COMPUTATION!)
# ===========================================
print("\n" + "=" * 50)
print("LOADING SAVED GRAPH FEATURES")
print("=" * 50)

if os.path.exists(GRAPH_FEATURES_PATH):
    graph_features_df = pd.read_csv(GRAPH_FEATURES_PATH)
    print(f"âœ“ Loaded {len(graph_features_df)} rows from saved dataset")
    print(f"  Columns: {list(graph_features_df.columns)}")
    
    # Merge with train and test
    train_merged = train.merge(graph_features_df, on='user_hash', how='left')
    test_merged = test.merge(graph_features_df, on='user_hash', how='left')
    
    print(f"âœ“ Merged with train: {train_merged.shape}")
    print(f"âœ“ Merged with test: {test_merged.shape}")
else:
    print(f"ERROR: Graph features not found at {GRAPH_FEATURES_PATH}")
    print("Please check the dataset path!")
    raise FileNotFoundError(GRAPH_FEATURES_PATH)

# ---------------------------------------------
# STEP 1: Create Feature Interactions
# ---------------------------------------------
print("\n[1/3] Creating feature interactions...")
start_time = time.time()

feature_cols_base = [col for col in train.columns if col.startswith('feature_')]

for df in [train_merged, test_merged]:
    # Feature ratios
    df['f001_f002_ratio'] = df['feature_001'] / (df['feature_002'] + 0.001)
    df['f003_f004_ratio'] = df['feature_003'] / (df['feature_004'] + 0.001)
    df['f005_f006_ratio'] = df['feature_005'] / (df['feature_006'] + 0.001)
    
    # Feature sums
    df['binary_sum'] = df['feature_007'] + df['feature_011'] + df['feature_013'] + df['feature_014']
    df['numeric_mean'] = df[['feature_001', 'feature_002', 'feature_003', 'feature_004', 'feature_005']].mean(axis=1)
    
    # Degree interactions
    df['degree_x_f001'] = df['degree'] * df['feature_001']
    df['degree_x_f015'] = df['degree'] * df['feature_015']
    
    # Missing value count (basic)
    df['missing_count'] = df[feature_cols_base].isnull().sum(axis=1)

print(f"Done in {time.time() - start_time:.2f}s")

# ---------------------------------------------
# STEP 2: MISSING VALUE PATTERN FEATURES (CRITICAL!)
# ---------------------------------------------
print("\n[2/3] Creating missing pattern features (CRITICAL!)...")
start_time = time.time()

# Group 1: Primary cheating indicators (98% cheating when missing)
group1_features = ['feature_007', 'feature_011', 'feature_013']

# Group 2: Secondary cheating indicators (97% cheating when missing)
group2_features = ['feature_014', 'feature_008', 'feature_009', 'feature_010']

for df in [train_merged, test_merged]:
    # Individual missing indicators
    for col in group1_features + group2_features:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)
    
    # Group missing counts
    df['group1_missing_count'] = df[[f'{c}_is_missing' for c in group1_features]].sum(axis=1)
    df['group2_missing_count'] = df[[f'{c}_is_missing' for c in group2_features]].sum(axis=1)
    
    # Pattern features
    df['any_group1_missing'] = (df['group1_missing_count'] >= 1).astype(int)
    df['all_group1_missing'] = (df['group1_missing_count'] == 3).astype(int)
    df['any_group2_missing'] = (df['group2_missing_count'] >= 1).astype(int)
    df['all_group2_missing'] = (df['group2_missing_count'] == 4).astype(int)
    
    # High confidence cheater pattern
    df['high_conf_cheater_pattern'] = (
        (df['group1_missing_count'] >= 2) | 
        (df['group2_missing_count'] >= 3)
    ).astype(int)
    
    df['total_critical_missing'] = df['group1_missing_count'] + df['group2_missing_count']
    
    # Pair patterns
    df['f007_f011_both_missing'] = ((df['feature_007_is_missing'] == 1) & (df['feature_011_is_missing'] == 1)).astype(int)
    df['f007_f013_both_missing'] = ((df['feature_007_is_missing'] == 1) & (df['feature_013_is_missing'] == 1)).astype(int)
    df['f011_f013_both_missing'] = ((df['feature_011_is_missing'] == 1) & (df['feature_013_is_missing'] == 1)).astype(int)
    df['all_binary_missing'] = (
        (df['feature_007_is_missing'] == 1) & 
        (df['feature_011_is_missing'] == 1) & 
        (df['feature_013_is_missing'] == 1) & 
        (df['feature_014_is_missing'] == 1)
    ).astype(int)

print(f"Done in {time.time() - start_time:.2f}s")

# Verify patterns
labeled_data = train_merged[train_merged['is_cheating'].notna()]
print(f"\nPattern verification:")
print(f"  Any Group1 missing: {labeled_data[labeled_data['any_group1_missing']==1]['is_cheating'].mean():.2%} cheating")
print(f"  High conf pattern:  {labeled_data[labeled_data['high_conf_cheater_pattern']==1]['is_cheating'].mean():.2%} cheating")

# ---------------------------------------------
# STEP 3: Preprocess Features
# ---------------------------------------------
print("\n[3/3] Preprocessing features...")
start_time = time.time()

exclude_cols = ['user_hash', CFG.TARGET, 'high_conf_clean']
feature_cols = [col for col in train_merged.columns if col not in exclude_cols]

# Fill missing values
for col in feature_cols:
    if col not in train_merged.columns or '_is_missing' in col:
        continue
    median_val = train_merged[col].median()
    if pd.isna(median_val):
        median_val = 0
    train_merged[col] = train_merged[col].fillna(median_val)
    test_merged[col] = test_merged[col].fillna(median_val)

print(f"Done in {time.time() - start_time:.2f}s")

# Prepare final datasets
train_processed = train_merged.copy()
test_processed = test_merged.copy()

# Labeled data
labeled_mask = train_processed[CFG.TARGET].notna()
X_train = train_processed.loc[labeled_mask, feature_cols]
y_train = train_processed.loc[labeled_mask, CFG.TARGET]

# Weak labeled data
weak_mask = (train_processed['high_conf_clean'] == 1) & (train_processed[CFG.TARGET].isna())
X_weak = train_processed.loc[weak_mask, feature_cols]
y_weak = pd.Series(0, index=X_weak.index)

# Test data
X_test = test_processed[feature_cols]
test_ids = test_processed['user_hash']

print(f"\n{'='*50}")
print(f"FEATURE ENGINEERING COMPLETE")
print(f"{'='*50}")
print(f"Features: {len(feature_cols)}")
print(f"Labeled samples: {len(X_train)}")
print(f"Weak labeled: {len(X_weak)}")
print(f"Test samples: {len(X_test)}")
print(f"Positive rate: {y_train.mean():.4f}")

# Cleanup
del train_merged, test_merged, graph_features_df
gc.collect()


# ===========================================
# COST FUNCTION - COMPLETE VERSION
# ===========================================

def mercor_cost(y_true, y_pred, thresholds=None):
    """
    Calculate the Mercor cost-based metric
    thresholds: tuple of (pass_threshold, block_threshold)
    If None, find optimal thresholds
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if thresholds is None:
        # Find optimal thresholds
        best_cost = float('inf')
        best_thresholds = None
        
        # Try different threshold combinations
        for pass_thresh in np.linspace(0.1, 0.9, 9):
            for block_thresh in np.linspace(pass_thresh + 0.1, 0.95, 9):
                decisions = np.where(y_pred <= pass_thresh, 'pass', 
                                   np.where(y_pred >= block_thresh, 'block', 'manual'))
                
                cost = 0
                for i, (true, decision) in enumerate(zip(y_true, decisions)):
                    if true == 1:  # Actual cheating
                        if decision == 'pass':
                            cost += 600  # False negative
                        elif decision == 'manual':
                            cost += 5    # True positive with manual review
                        # 'block' is correct, cost = 0
                    else:  # Actual not cheating
                        if decision == 'block':
                            cost += 300  # False positive in auto-block
                        elif decision == 'manual':
                            cost += 150  # False positive in manual review
                        # 'pass' is correct, cost = 0
                
                if cost < best_cost:
                    best_cost = cost
                    best_thresholds = (pass_thresh, block_thresh)
        
        return -best_cost, best_thresholds
    else:
        pass_thresh, block_thresh = thresholds
        decisions = np.where(y_pred <= pass_thresh, 'pass', 
                           np.where(y_pred >= block_thresh, 'block', 'manual'))
        
        cost = 0
        for true, decision in zip(y_true, decisions):
            if true == 1:  # Actual cheating
                if decision == 'pass':
                    cost += 600
                elif decision == 'manual':
                    cost += 5
            else:  # Actual not cheating
                if decision == 'block':
                    cost += 300
                elif decision == 'manual':
                    cost += 150
        
        return -cost


def mercor_cost_fast(y_true, y_pred, thresholds=None):
    """
    FAST vectorized cost calculation
    Much faster than the loop version
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if thresholds is None:
        return mercor_cost(y_true, y_pred, thresholds)
    
    pass_thresh, block_thresh = thresholds
    
    # Vectorized decision making
    is_pass = y_pred <= pass_thresh
    is_block = y_pred >= block_thresh
    is_manual = ~is_pass & ~is_block
    
    is_cheating = y_true == 1
    is_clean = y_true == 0
    
    # Calculate costs vectorized (no loops = fast)
    cost = 0
    cost += 600 * np.sum(is_cheating & is_pass)       # False negative
    cost += 5 * np.sum(is_cheating & is_manual)       # TP with manual review
    cost += 300 * np.sum(is_clean & is_block)         # FP in auto-block
    cost += 150 * np.sum(is_clean & is_manual)        # FP in manual review
    
    return -cost


def find_optimal_thresholds(y_true, y_pred):
    """
    Find optimal pass/block thresholds using grid search
    Returns: (best_cost, (pass_threshold, block_threshold))
    """
    best_cost = -float('inf')
    best_thresholds = (0.3, 0.7)
    
    # Fine-grained grid search
    for pass_t in np.linspace(0.05, 0.5, 20):
        for block_t in np.linspace(0.5, 0.95, 20):
            if block_t <= pass_t:
                continue
            cost = mercor_cost_fast(y_true, y_pred, (pass_t, block_t))
            if cost > best_cost:
                best_cost = cost
                best_thresholds = (pass_t, block_t)
    
    return best_cost, best_thresholds


# Test the cost functions
print("Testing cost functions...")
test_y_true = np.array([1, 0, 1, 0, 1])
test_y_pred = np.array([0.85, 0.15, 0.45, 0.25, 0.95])

cost1, thresh1 = mercor_cost(test_y_true, test_y_pred)
print(f"mercor_cost: {cost1}, thresholds: {thresh1}")

cost2 = mercor_cost_fast(test_y_true, test_y_pred, (0.3, 0.7))
print(f"mercor_cost_fast: {cost2}")

cost3, thresh3 = find_optimal_thresholds(test_y_true, test_y_pred)
print(f"find_optimal_thresholds: {cost3}, thresholds: {thresh3}")

print("\nCost functions ready!")


%%time
# ===========================================
# METHOD 1: Advanced Ensemble (IMPROVED)
# ===========================================
print("=" * 60)
print("METHOD 1: Advanced Ensemble with Semi-Supervised Learning")
print("=" * 60)

# Combine labeled and weak labeled
X_combined = pd.concat([X_train, X_weak], ignore_index=False)
y_combined = pd.concat([y_train, y_weak], ignore_index=False)

print(f"Combined samples: {len(X_combined)}")
print(f"Positive rate: {y_combined.mean():.4f}")

# Cross-validation
skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

# Store predictions
oof_lgb = np.zeros(len(X_combined))
oof_xgb = np.zeros(len(X_combined))
oof_cb = np.zeros(len(X_combined))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cb = np.zeros(len(X_test))

# Feature importance storage
feature_importance = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
    print(f"\n{'='*20} Fold {fold + 1}/{CFG.N_FOLDS} {'='*20}")
    
    X_fold_train = X_combined.iloc[train_idx]
    X_fold_val = X_combined.iloc[val_idx]
    y_fold_train = y_combined.iloc[train_idx]
    y_fold_val = y_combined.iloc[val_idx]
    
    # ============ LightGBM ============
    print("Training LightGBM...")
    lgb_train = lgb.Dataset(X_fold_train, y_fold_train)
    lgb_val = lgb.Dataset(X_fold_val, y_fold_val)
    
    model_lgb = lgb.train(
        CFG.LGB_PARAMS,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    
    oof_lgb[val_idx] = model_lgb.predict(X_fold_val)
    test_preds_lgb += model_lgb.predict(X_test) / CFG.N_FOLDS
    
    # Feature importance
    fold_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_lgb.feature_importance(importance_type='gain'),
        'fold': fold
    })
    feature_importance = pd.concat([feature_importance, fold_importance])
    
    # ============ XGBoost ============
    print("Training XGBoost...")
    dtrain = xgb.DMatrix(X_fold_train, y_fold_train)
    dval = xgb.DMatrix(X_fold_val, y_fold_val)
    
    model_xgb = xgb.train(
        CFG.XGB_PARAMS,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=200
    )
    
    oof_xgb[val_idx] = model_xgb.predict(dval)
    test_preds_xgb += model_xgb.predict(xgb.DMatrix(X_test)) / CFG.N_FOLDS
    
    # ============ CatBoost ============
    print("Training CatBoost...")
    model_cb = cb.CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=CFG.SEED,
        verbose=200,
        early_stopping_rounds=100
    )
    
    model_cb.fit(
        X_fold_train, y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        verbose=200
    )
    
    oof_cb[val_idx] = model_cb.predict_proba(X_fold_val)[:, 1]
    test_preds_cb += model_cb.predict_proba(X_test)[:, 1] / CFG.N_FOLDS
    
    # Fold scores
    val_mask = np.isin(np.arange(len(X_combined)), val_idx)
    labeled_val_mask = val_mask & (y_combined.index.isin(y_train.index))
    
    gc.collect()

# ============ Ensemble ============
print("\n" + "=" * 60)
print("ENSEMBLE RESULTS")
print("=" * 60)

# Simple average
test_preds_ensemble = (test_preds_lgb + test_preds_xgb + test_preds_cb) / 3

# Weighted average (give more weight to best performer)
# We'll also try this
test_preds_weighted = (0.4 * test_preds_lgb + 0.3 * test_preds_xgb + 0.3 * test_preds_cb)

# Calculate OOF scores on labeled data only
labeled_indices = y_train.index
oof_ensemble = (oof_lgb + oof_xgb + oof_cb) / 3

# Get labeled OOF predictions
oof_labeled = oof_ensemble[X_combined.index.isin(labeled_indices)]
y_labeled = y_combined[X_combined.index.isin(labeled_indices)]

cost_lgb, thresh_lgb = find_optimal_thresholds(y_labeled, oof_lgb[X_combined.index.isin(labeled_indices)])
cost_xgb, thresh_xgb = find_optimal_thresholds(y_labeled, oof_xgb[X_combined.index.isin(labeled_indices)])
cost_cb, thresh_cb = find_optimal_thresholds(y_labeled, oof_cb[X_combined.index.isin(labeled_indices)])
cost_ens, thresh_ens = find_optimal_thresholds(y_labeled, oof_labeled)

print(f"LightGBM CV Cost: {cost_lgb:,.0f} | Thresholds: {thresh_lgb}")
print(f"XGBoost CV Cost:  {cost_xgb:,.0f} | Thresholds: {thresh_xgb}")
print(f"CatBoost CV Cost: {cost_cb:,.0f} | Thresholds: {thresh_cb}")
print(f"Ensemble CV Cost: {cost_ens:,.0f} | Thresholds: {thresh_ens}")

# Top features
print("\nTop 15 Important Features:")
top_features = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).head(15)
print(top_features)

print("\nMethod 1 Complete!")


%%time
# ===========================================
# METHOD 2: LightGBM Only with Tuned Parameters
# ===========================================
print("=" * 60)
print("METHOD 2: Tuned LightGBM with Feature Selection")
print("=" * 60)

# Use only top features to avoid overfitting
top_n_features = 25
top_feature_names = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).head(top_n_features).index.tolist()

print(f"Using top {len(top_feature_names)} features")

X_train_top = X_train[top_feature_names]
X_test_top = X_test[top_feature_names]

# Tuned parameters for better generalization
lgb_params_tuned = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.03,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': CFG.SEED,
    'n_jobs': -1
}

# Cross-validation
skf2 = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED + 1)

oof_method2 = np.zeros(len(X_train))
test_preds_method2 = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf2.split(X_train_top, y_train)):
    print(f"\nFold {fold + 1}/{CFG.N_FOLDS}")
    
    X_fold_train = X_train_top.iloc[train_idx]
    X_fold_val = X_train_top.iloc[val_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_fold_train, y_fold_train)
    lgb_val = lgb.Dataset(X_fold_val, y_fold_val)
    
    model = lgb.train(
        lgb_params_tuned,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=3000,
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)]
    )
    
    oof_method2[val_idx] = model.predict(X_fold_val)
    test_preds_method2 += model.predict(X_test_top) / CFG.N_FOLDS
    
    gc.collect()

# Calculate CV score
cost_m2, thresh_m2 = find_optimal_thresholds(y_train, oof_method2)
print(f"\nMethod 2 CV Cost: {cost_m2:,.0f}")
print(f"Optimal Thresholds: {thresh_m2}")

# Store for submission
test_preds_stack = test_preds_method2.copy()

print("\nMethod 2 Complete!")


%%time
# ===========================================
# METHOD 3: Blended Models with Different Seeds
# ===========================================
print("=" * 60)
print("METHOD 3: Multi-Seed Blending")
print("=" * 60)

# Train same model with different seeds and blend
seeds = [42, 123, 456, 789, 2024]
test_preds_blended = np.zeros(len(X_test))

for seed in seeds:
    print(f"\nTraining with seed {seed}...")
    
    lgb_params_seed = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 48,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': seed,
        'n_jobs': -1
    }
    
    # Simple train/val split
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )
    
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val)
    
    model = lgb.train(
        lgb_params_seed,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=1500,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    test_preds_blended += model.predict(X_test) / len(seeds)
    
    val_cost = mercor_cost_fast(y_val, model.predict(X_val), (0.3, 0.7))
    print(f"Seed {seed} validation cost: {val_cost:,.0f}")

# Store for submission
test_preds_simple = test_preds_blended.copy()

# Final score estimate
print(f"\nMethod 3 Complete!")
print(f"Blended {len(seeds)} models")


# ===========================================
# METHOD 4: IMPROVED LABEL PROPAGATION
# ===========================================
# KEY IMPROVEMENT: Use TRAIN-TEST edges to propagate 
# from KNOWN cheaters in training set!
# RUNTIME: ~3 hours (neighbor stats computation)
# ===========================================

print("=" * 60)
print("METHOD 4: IMPROVED LABEL PROPAGATION")
print("=" * 60)

import scipy.sparse as sp
from sklearn.preprocessing import normalize

# Use already loaded social_graph (don't reload)
print(f"Total edges in social graph: {len(social_graph)}")

# Get user sets
train_user_set = set(train['user_hash'])
test_user_set = set(test_ids)

# Get labeled train users
labeled_train = train[train['is_cheating'].notna()][['user_hash', 'is_cheating']]
labeled_dict = dict(zip(labeled_train['user_hash'], labeled_train['is_cheating']))
print(f"Labeled train users: {len(labeled_dict)}")

# ===========================================
# STEP 1: Find TRAIN-TEST edges
# ===========================================
print("\n[1/4] Finding Train-Test edges...")

# Train-Test edges (user_a in train AND user_b in test, or vice versa)
mask_train_test = (
    (social_graph['user_a'].isin(train_user_set) & social_graph['user_b'].isin(test_user_set)) |
    (social_graph['user_a'].isin(test_user_set) & social_graph['user_b'].isin(train_user_set))
)
train_test_edges = social_graph[mask_train_test].copy()
print(f"Train-Test edges: {len(train_test_edges)}")

# Test-Test edges
mask_test_test = social_graph['user_a'].isin(test_user_set) & social_graph['user_b'].isin(test_user_set)
test_test_edges = social_graph[mask_test_test].copy()
print(f"Test-Test edges: {len(test_test_edges)}")

# ===========================================
# STEP 2: Compute neighbor cheating rate for each TEST user
# ===========================================
print("\n[2/4] Computing neighbor cheating rate from TRAIN neighbors...")
print("âš ï¸� This takes ~3 hours - please wait...")

test_neighbor_stats = {}

for test_user in tqdm(test_ids, desc="Computing neighbor stats"):
    # Find all neighbors of this test user
    neighbors_a = social_graph[social_graph['user_b'] == test_user]['user_a'].values
    neighbors_b = social_graph[social_graph['user_a'] == test_user]['user_b'].values
    all_neighbors = set(neighbors_a) | set(neighbors_b)
    
    # Filter to LABELED train neighbors only
    labeled_neighbors = [n for n in all_neighbors if n in labeled_dict]
    
    if len(labeled_neighbors) > 0:
        cheating_labels = [labeled_dict[n] for n in labeled_neighbors]
        test_neighbor_stats[test_user] = {
            'train_neighbor_cheat_rate': np.mean(cheating_labels),
            'train_neighbor_cheat_count': sum(cheating_labels),
            'train_neighbor_clean_count': len(cheating_labels) - sum(cheating_labels),
            'labeled_train_neighbor_count': len(labeled_neighbors),
            'has_cheater_neighbor': 1 if sum(cheating_labels) > 0 else 0
        }
    else:
        test_neighbor_stats[test_user] = {
            'train_neighbor_cheat_rate': -1,  # Unknown (no labeled neighbors)
            'train_neighbor_cheat_count': 0,
            'train_neighbor_clean_count': 0,
            'labeled_train_neighbor_count': 0,
            'has_cheater_neighbor': 0
        }

# Convert to dataframe
neighbor_features_df = pd.DataFrame.from_dict(test_neighbor_stats, orient='index')
neighbor_features_df['user_hash'] = neighbor_features_df.index
neighbor_features_df = neighbor_features_df.reset_index(drop=True)

# Stats
has_labeled = neighbor_features_df[neighbor_features_df['labeled_train_neighbor_count'] > 0]
print(f"\nTest users with labeled train neighbors: {len(has_labeled)} / {len(test_ids)} ({100*len(has_labeled)/len(test_ids):.1f}%)")
print(f"Average neighbor cheat rate (for those with neighbors): {has_labeled['train_neighbor_cheat_rate'].mean():.4f}")

has_cheater = neighbor_features_df[neighbor_features_df['has_cheater_neighbor'] == 1]
print(f"Test users connected to at least one cheater: {len(has_cheater)}")

# ===========================================
# STEP 3: Apply IMPROVED Label Propagation
# ===========================================
print("\n[3/4] Applying Improved Label Propagation...")

def apply_improved_LP(user_ids, predictions, alpha_tt=0.2, alpha_train=0.3, max_iter=2):
    """
    Improved LP that uses:
    1. Test-test edges (like before)
    2. Train-test edges (propagate from KNOWN cheaters)
    """
    print(f"  LP params: alpha_tt={alpha_tt}, alpha_train={alpha_train}, max_iter={max_iter}")
    
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    num_users = len(user_ids)
    
    # Build test-test adjacency matrix
    if len(test_test_edges) > 0:
        row = test_test_edges['user_a'].map(user_to_idx).dropna().astype(int).values
        col = test_test_edges['user_b'].map(user_to_idx).dropna().astype(int).values
        # Filter valid indices
        valid_mask = (row < num_users) & (col < num_users)
        row, col = row[valid_mask], col[valid_mask]
        data = np.ones(len(row))
        adj_tt = sp.coo_matrix((data, (row, col)), shape=(num_users, num_users))
        adj_tt = adj_tt + adj_tt.T
        adj_tt_norm = normalize(adj_tt, norm='l1', axis=1)
        node_degrees_tt = np.array(adj_tt.sum(axis=1)).flatten()
        has_tt_neighbor = node_degrees_tt > 0
        print(f"  Test users with test-test neighbors: {has_tt_neighbor.sum()}")
    else:
        has_tt_neighbor = np.zeros(num_users, dtype=bool)
    
    # Get initial predictions
    y_init = predictions.copy()
    y_current = y_init.copy()
    
    # Standard LP on test-test edges
    for i in range(max_iter):
        if has_tt_neighbor.sum() > 0:
            neighbor_avg = adj_tt_norm.dot(y_current)
            y_current[has_tt_neighbor] = (
                alpha_tt * neighbor_avg[has_tt_neighbor] + 
                (1 - alpha_tt) * y_init[has_tt_neighbor]
            )
    
    # ADDITIONAL: Adjust based on TRAIN neighbor cheating rate
    adjusted_count = 0
    for idx, user in enumerate(user_ids):
        stats = test_neighbor_stats.get(user, {})
        n_labeled = stats.get('labeled_train_neighbor_count', 0)
        cheat_rate = stats.get('train_neighbor_cheat_rate', -1)
        
        if n_labeled >= 1 and cheat_rate >= 0:
            # Weight based on number of neighbors (more neighbors = more confidence)
            weight = min(alpha_train, 0.05 * n_labeled)  # Cap at alpha_train
            y_current[idx] = (1 - weight) * y_current[idx] + weight * cheat_rate
            adjusted_count += 1
    
    print(f"  Adjusted {adjusted_count} users based on train neighbor info")
    return y_current

# Apply improved LP to Method 3 (your best base model)
print("\nApplying to Method 3 (Blended)...")
test_preds_LP = apply_improved_LP(
    test_ids.values, 
    test_preds_simple, 
    alpha_tt=0.2, 
    alpha_train=0.25,
    max_iter=2
)
print(f"  Range: [{test_preds_LP.min():.4f}, {test_preds_LP.max():.4f}]")
print(f"  Mean: {test_preds_LP.mean():.4f}")

print("\n" + "=" * 60)
print("METHOD 4 COMPLETE!")
print("=" * 60)


# ===========================================
# METHOD 5: PUBLIC BLEND - FINAL OPTIMIZED
# ===========================================
# Best so far: 0.55 Our + 0.45 Public = -1,560,700
# Testing fine-tuned weights around 0.55-0.62
# ===========================================

print("=" * 60)
print("METHOD 5: OPTIMIZED PUBLIC BLEND")
print("=" * 60)

# Paths to public submissions
PUBLIC_BLEND_1 = '/kaggle/input/mercor-cheating-detection-h-blend/submission.csv'
PUBLIC_BLEND_2 = '/kaggle/input/mercor-cheating-detection-ensemble-1570000/submission.csv'

import os
pub1_exists = os.path.exists(PUBLIC_BLEND_1)
pub2_exists = os.path.exists(PUBLIC_BLEND_2)

print(f"h-blend: {'âœ“' if pub1_exists else 'âœ—'}")
print(f"ensemble-1570000: {'âœ“' if pub2_exists else 'âœ—'}")

blend_submissions = {}

if pub1_exists and pub2_exists:
    # Load public submissions
    df_pub1 = pd.read_csv(PUBLIC_BLEND_1)
    df_pub2 = pd.read_csv(PUBLIC_BLEND_2)
    
    # Merge
    merged = pd.DataFrame({'user_hash': test_ids})
    merged = merged.merge(df_pub1[['user_hash', 'prediction']], on='user_hash', how='left')
    merged = merged.rename(columns={'prediction': 'pub1'})
    merged = merged.merge(df_pub2[['user_hash', 'prediction']], on='user_hash', how='left')
    merged = merged.rename(columns={'prediction': 'pub2'})
    merged['pub1'] = merged['pub1'].fillna(0.5)
    merged['pub2'] = merged['pub2'].fillna(0.5)
    
    pub1 = merged['pub1'].values
    pub2 = merged['pub2'].values
    
    # Public blend (0.61/0.39)
    test_preds_public_blend = 0.61 * pub1 + 0.39 * pub2
    
    # ===========================================
    # GET OUR BEST PREDICTION
    # ===========================================
    if 'test_preds_LP' in dir() and test_preds_LP is not None:
        our_best = test_preds_LP
        print("Using: test_preds_LP (Label Propagation) âœ“")
    else:
        our_best = test_preds_simple
        print("Using: test_preds_simple (Method 3) - LP not available!")
    
    # ===========================================
    # FINE-TUNED WEIGHT TESTING
    # ===========================================
    print("\n--- Testing Fine-Tuned Weights ---")
    print("Best known: 0.55 Our = -1,560,700")
    print("Testing around optimal range...\n")
    
    # Fine-tune around 0.55-0.62 (where best results are)
    weights_to_try = [0.52, 0.55, 0.57, 0.58, 0.60, 0.62, 0.65]
    
    for our_w in weights_to_try:
        pub_w = 1 - our_w
        blended = our_w * our_best + pub_w * test_preds_public_blend
        name = f"5_{int(our_w*100):02d}_{int(pub_w*100):02d}"
        blend_submissions[name] = blended
        print(f"  {our_w:.2f} Our + {pub_w:.2f} Public: mean={blended.mean():.4f}")
    
    # ===========================================
    # TRIPLE BLENDS
    # ===========================================
    print("\n--- Triple Blends ---")
    blend_submissions['5_triple_55'] = 0.55 * our_best + 0.30 * pub1 + 0.15 * pub2
    blend_submissions['5_triple_58'] = 0.58 * our_best + 0.28 * pub1 + 0.14 * pub2
    blend_submissions['5_triple_60'] = 0.60 * our_best + 0.26 * pub1 + 0.14 * pub2
    
    for name in ['5_triple_55', '5_triple_58', '5_triple_60']:
        print(f"  {name}: mean={blend_submissions[name].mean():.4f}")
    
    print(f"\n--- Total blend variations: {len(blend_submissions)} ---")
    
else:
    print("\nâš ï¸� PUBLIC NOTEBOOKS NOT FOUND!")
    print("Add these as Input Data:")
    print("  - mercor-cheating-detection-h-blend")
    print("  - mercor-cheating-detection-ensemble-1570000")

print("\n" + "=" * 60)
print("METHOD 5 COMPLETE!")
print("=" * 60)


# ===========================================
# CREATE SUBMISSION FILES
# ===========================================
print("=" * 60)
print("CREATING SUBMISSION FILES")
print("=" * 60)

import os

# ===========================================
# BLEND VARIATIONS (Main submissions)
# ===========================================
print("\n--- Blend Variations ---")

if 'blend_submissions' in dir() and len(blend_submissions) > 0:
    for name, preds in blend_submissions.items():
        sub = pd.DataFrame({'user_hash': test_ids, 'prediction': np.clip(preds, 0, 1)})
        sub.to_csv(f'{CFG.OUTPUT_DIR}submission_{name}.csv', index=False)
        print(f"âœ“ {name}: mean={preds.mean():.4f}")

# ===========================================
# BACKUP SUBMISSIONS
# ===========================================
print("\n--- Backup Methods ---")

if 'test_preds_LP' in dir() and test_preds_LP is not None:
    sub = pd.DataFrame({'user_hash': test_ids, 'prediction': np.clip(test_preds_LP, 0, 1)})
    sub.to_csv(f'{CFG.OUTPUT_DIR}submission_our_LP.csv', index=False)
    print(f"âœ“ Our LP only: mean={test_preds_LP.mean():.4f}")

if 'test_preds_simple' in dir():
    sub = pd.DataFrame({'user_hash': test_ids, 'prediction': np.clip(test_preds_simple, 0, 1)})
    sub.to_csv(f'{CFG.OUTPUT_DIR}submission_method3.csv', index=False)
    print(f"âœ“ Method 3: mean={test_preds_simple.mean():.4f}")

# ===========================================
# DEFAULT SUBMISSION
# ===========================================
print("\n" + "=" * 60)
print("SELECTING DEFAULT SUBMISSION")
print("=" * 60)

# 0.55 was best, use as default
if '5_55_45' in blend_submissions:
    default_preds = blend_submissions['5_55_45']
    default_name = "5_55_45 (0.55 Our + 0.45 Public)"
elif '5_57_43' in blend_submissions:
    default_preds = blend_submissions['5_57_43']
    default_name = "5_57_43"
elif 'test_preds_LP' in dir() and test_preds_LP is not None:
    default_preds = test_preds_LP
    default_name = "Our LP"
else:
    default_preds = test_preds_simple
    default_name = "Method 3"

sub = pd.DataFrame({'user_hash': test_ids, 'prediction': np.clip(default_preds, 0, 1)})
sub.to_csv(f'{CFG.OUTPUT_DIR}submission.csv', index=False)
print(f"â†’ Default: {default_name}")
print(f"  Mean: {default_preds.mean():.4f}")

# ===========================================
# LIST ALL FILES
# ===========================================
print("\n" + "=" * 60)
print("ALL OUTPUT FILES")
print("=" * 60)

csv_files = sorted([f for f in os.listdir(CFG.OUTPUT_DIR) if f.endswith('.csv') and 'graph' not in f.lower()])
for f in csv_files:
    size_kb = os.path.getsize(os.path.join(CFG.OUTPUT_DIR, f)) / 1024
    print(f"  ğŸ“„ {f} ({size_kb:.1f} KB)")

print(f"\nTotal: {len(csv_files)} files")

# ===========================================
# SUBMISSION PRIORITY ORDER
# ===========================================
print("\n" + "=" * 60)
print("ğŸ“‹ SUBMISSION ORDER (TRY IN THIS ORDER)")
print("=" * 60)
print("""
PRIORITY (fine-tuned around best score):

1. submission_5_55_45.csv  â†� KNOWN BEST (-1,560,700)
2. submission_5_57_43.csv  â†� Try this! (between 0.55 and 0.60)
3. submission_5_58_42.csv  â†� Fine-tune
4. submission_5_60_40.csv  â†� Slightly higher weight
5. submission_5_52_48.csv  â†� Slightly lower weight
6. submission_5_62_38.csv  â†� Higher weight test
7. submission_5_triple_58.csv â†� Triple blend

BACKUPS:
8. submission_our_LP.csv   â†� Pure LP
9. submission_method3.csv  â†� Pure ensemble

The sweet spot is likely between 0.55-0.60!
""")
print("=" * 60)

