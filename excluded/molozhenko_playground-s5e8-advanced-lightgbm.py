"""
Advanced LightGBM Model for Bank Term Deposit Prediction
Enhanced with improved feature engineering, optimized hyperparameters, and better validation
"""

import pandas as pd
import numpy as np

from typing import Tuple
import random
import time
import os
import gc

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from tqdm.auto import tqdm

import warnings
warnings.simplefilter('ignore')

tqdm.pandas()

# Configuration - enhanced parameters
class CFG:
    mode = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'localhost')
    path = "../input/playground-series-s5e8/"
    original = "../input/bank-marketing-dataset-full/"

    n_splits = 10  # Increased for more stable validation
    seed = 42
    
    # Optimized hyperparameters
    learning_rate = 0.025  # Reduced for better accuracy
    num_boost_round = 80000  # Increased
    early_stopping_rounds = 400  # Increased for more patient training
    verbose_eval = False if mode=='Batch' else 1000

    target = "y"
    plot_importance = False
    use_pseudo_labeling = True
    use_target_encoding = True
    use_clustering = True
    
    # New parameters for improvement
    pseudo_threshold_high = 0.99  # Stricter threshold
    pseudo_threshold_low = 0.01
    target_encoding_smoothing = 5.0  # Increased smoothing

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.seed)

# Load data
train = pd.read_csv(CFG.path + "train.csv").drop(columns=['id'])
test = pd.read_csv(CFG.path + "test.csv").drop(columns=['id'])

original = pd.read_csv(CFG.original + "bank-full.csv", sep=";")
original[CFG.target] = (original[CFG.target]=="yes").astype(int)

features = test.columns.to_list()

# Improved Target Encoder with cross-validation
class ImprovedTargetEncoder:
    def __init__(self, smooth=5.0, cv_folds=5):
        self.smooth = smooth
        self.cv_folds = cv_folds
        self.global_mean = None
        self.category_means = {}
        
    def fit_transform(self, X, y):
        self.global_mean = y.mean()
        self.category_means = {}
        
        # Use cross-validation for target encoding
        result = np.zeros(len(X))
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=CFG.seed)
        
        for train_idx, val_idx in cv.split(X, y):
            # Calculate statistics on train part
            temp_means = {}
            for category in X.iloc[train_idx].unique():
                mask = X.iloc[train_idx] == category
                if mask.sum() > 0:
                    category_sum = y.iloc[train_idx][mask].sum()
                    category_count = mask.sum()
                    smoothed_mean = (category_sum + self.smooth * self.global_mean) / (category_count + self.smooth)
                    temp_means[category] = smoothed_mean
            
            # Apply to validation part
            for category in X.iloc[val_idx].unique():
                mask_val = X.iloc[val_idx] == category
                if category in temp_means:
                    result[val_idx[mask_val]] = temp_means[category]
                else:
                    result[val_idx[mask_val]] = self.global_mean
        
        # Save final statistics for transform
        for category in X.unique():
            mask = X == category
            category_sum = y[mask].sum()
            category_count = mask.sum()
            smoothed_mean = (category_sum + self.smooth * self.global_mean) / (category_count + self.smooth)
            self.category_means[category] = smoothed_mean
        
        return result
    
    def transform(self, X):
        result = np.full(len(X), self.global_mean)
        for category, mean_val in self.category_means.items():
            mask = X == category
            result[mask] = mean_val
        return result

# Global variables for storing clustering models
cluster_models = {}
cluster_stats = {}

# Extended feature engineering function
def create_advanced_features(df, target_encoders=None, fit_encoders=False, kmeans_model=None, scaler=None):
    global cluster_models, cluster_stats
    df = df.copy()
    
    # Basic binary features
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['balance_negative'] = (df['balance'] < 0).astype(int)
    df['balance_zero'] = (df['balance'] == 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['duration_short'] = (df['duration'] < 100).astype(int)
    df['duration_medium'] = ((df['duration'] >= 100) & (df['duration'] <= 300)).astype(int)
    df['duration_long'] = (df['duration'] > 300).astype(int)
    df['duration_very_long'] = (df['duration'] > 600).astype(int)
    df['campaign_single'] = (df['campaign'] == 1).astype(int)
    df['campaign_multiple'] = (df['campaign'] > 2).astype(int)
    df['campaign_high'] = (df['campaign'] > 5).astype(int)
    df['campaign_very_high'] = (df['campaign'] > 10).astype(int)
    
    # Enhanced numerical transformations
    df['log_duration'] = np.log1p(df['duration'])
    df['log_campaign'] = np.log1p(df['campaign'])
    df['log_previous'] = np.log1p(df['previous'])
    df['sqrt_age'] = np.sqrt(df['age'])
    df['sqrt_duration'] = np.sqrt(df['duration'])
    df['log_balance'] = np.sign(df['balance']) * np.log1p(np.abs(df['balance']))
    df['balance_abs'] = np.abs(df['balance'])
    df['balance_abs_log'] = np.log1p(df['balance_abs'])
    
    # Power features
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3
    df['duration_squared'] = df['duration'] ** 2
    df['duration_cubed'] = df['duration'] ** 3
    df['campaign_squared'] = df['campaign'] ** 2
    df['balance_squared'] = df['balance'] ** 2
    
    # More detailed binning
    df['age_bin_5'] = pd.cut(df['age'], bins=5, labels=False)
    df['age_bin_10'] = pd.cut(df['age'], bins=10, labels=False)
    df['age_bin_detailed'] = pd.cut(df['age'], bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75, 100], labels=False)
    
    # Quantile binning
    df['duration_qbin_5'] = pd.qcut(df['duration'], q=5, labels=False, duplicates='drop')
    df['duration_qbin_10'] = pd.qcut(df['duration'], q=10, labels=False, duplicates='drop')
    df['duration_qbin_20'] = pd.qcut(df['duration'], q=20, labels=False, duplicates='drop')
    
    df['balance_qbin_5'] = pd.qcut(df['balance'], q=5, labels=False, duplicates='drop')
    df['balance_qbin_10'] = pd.qcut(df['balance'], q=10, labels=False, duplicates='drop')
    df['balance_qbin_20'] = pd.qcut(df['balance'], q=20, labels=False, duplicates='drop')
    
    df['campaign_bin'] = pd.cut(df['campaign'], bins=[-1, 1, 2, 3, 5, 10, 100], labels=False)
    
    # Extended feature interactions
    df['age_duration'] = df['age'] * df['duration']
    df['age_campaign'] = df['age'] * df['campaign']
    df['age_balance'] = df['age'] * df['balance']
    df['age_previous'] = df['age'] * df['previous']
    df['duration_campaign'] = df['duration'] * df['campaign']
    df['duration_balance'] = df['duration'] * df['balance']
    df['duration_previous'] = df['duration'] * df['previous']
    df['campaign_balance'] = df['campaign'] * df['balance']
    df['campaign_previous'] = df['campaign'] * df['previous']
    df['balance_previous'] = df['balance'] * df['previous']
    
    # Three-way interactions
    df['age_duration_campaign'] = df['age'] * df['duration'] * df['campaign']
    df['age_balance_duration'] = df['age'] * df['balance'] * df['duration']
    df['duration_campaign_balance'] = df['duration'] * df['campaign'] * df['balance']
    
    # Enhanced ratios
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['duration_per_age'] = df['duration'] / (df['age'] + 1)
    df['campaign_per_age'] = df['campaign'] / (df['age'] + 1)
    df['balance_per_age'] = df['balance'] / (df['age'] + 1)
    df['balance_per_campaign'] = df['balance'] / (df['campaign'] + 1)
    df['previous_per_campaign'] = df['previous'] / (df['campaign'] + 1)
    df['age_per_duration'] = df['age'] / (df['duration'] + 1)
    df['campaign_per_duration'] = df['campaign'] / (df['duration'] + 1)
    
    # Group statistics
    df['balance_rank'] = df['balance'].rank(pct=True)
    df['duration_rank'] = df['duration'].rank(pct=True)
    df['age_rank'] = df['age'].rank(pct=True)
    df['campaign_rank'] = df['campaign'].rank(pct=True)
    
    # Contact-related features
    df['contact_success_rate'] = df['previous'] / (df['campaign'] + df['previous'] + 1)
    df['pdays_binned'] = pd.cut(df['pdays'], bins=[-2, -1, 0, 30, 60, 120, 200, 400, 1000], labels=False)
    df['has_pdays'] = (df['pdays'] != -1).astype(int)
    df['pdays_recent'] = ((df['pdays'] > 0) & (df['pdays'] <= 30)).astype(int)
    df['pdays_old'] = (df['pdays'] > 180).astype(int)
    
    # Seasonal features
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_map)
    df['quarter'] = ((df['month_num'] - 1) // 3) + 1
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df['is_spring'] = df['month'].isin(['mar', 'apr', 'may']).astype(int)
    df['is_summer'] = df['month'].isin(['jun', 'jul', 'aug']).astype(int)
    df['is_autumn'] = df['month'].isin(['sep', 'oct', 'nov']).astype(int)
    df['is_winter'] = df['month'].isin(['dec', 'jan', 'feb']).astype(int)
    df['is_peak_season'] = df['month'].isin(['may', 'jun', 'jul', 'aug', 'nov']).astype(int)
    
    # Day of month features
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['is_month_start'] = (df['day'] <= 5).astype(int)
    df['is_month_end'] = (df['day'] >= 25).astype(int)
    df['is_month_middle'] = ((df['day'] > 10) & (df['day'] <= 20)).astype(int)
    
    # Target encoding for categorical features
    if CFG.use_target_encoding and target_encoders is not None:
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        
        if fit_encoders:
            for feature in categorical_features:
                if feature in df.columns:
                    df[f'{feature}_target_encoded'] = target_encoders[feature].fit_transform(df[feature], df[CFG.target] if CFG.target in df.columns else None)
        else:
            for feature in categorical_features:
                if feature in df.columns and feature in target_encoders:
                    df[f'{feature}_target_encoded'] = target_encoders[feature].transform(df[feature])
    
    # Simplified clustering
    if CFG.use_clustering:
        numerical_features = ['age', 'balance', 'duration', 'campaign', 'previous']
        if all(feat in df.columns for feat in numerical_features):
            if fit_encoders:
                # Normalization for clustering
                scaler = RobustScaler()
                scaled_features = scaler.fit_transform(df[numerical_features])
                
                # Single clustering with 12 clusters
                kmeans_model = KMeans(n_clusters=12, random_state=CFG.seed, n_init=10, max_iter=300)
                df['cluster'] = kmeans_model.fit_predict(scaled_features)
                
                # Cluster statistics
                for feature in numerical_features:
                    df[f'{feature}_cluster_mean'] = df.groupby('cluster')[feature].transform('mean')
                    df[f'{feature}_cluster_std'] = df.groupby('cluster')[feature].transform('std')
                    df[f'{feature}_cluster_rank'] = df.groupby('cluster')[feature].rank(pct=True)
                
                return df, kmeans_model, scaler
            else:
                if kmeans_model is not None and scaler is not None:
                    scaled_features = scaler.transform(df[numerical_features])
                    df['cluster'] = kmeans_model.predict(scaled_features)
                    
                    # For test data create simple statistics
                    for feature in numerical_features:
                        df[f'{feature}_cluster_mean'] = df.groupby('cluster')[feature].transform('mean')
                        df[f'{feature}_cluster_std'] = df.groupby('cluster')[feature].transform('std')
                        df[f'{feature}_cluster_rank'] = df.groupby('cluster')[feature].rank(pct=True)
    
    return df

# Enhanced pseudo-labeling function
def pseudo_labeling(train_df, test_df, threshold_high=CFG.pseudo_threshold_high, threshold_low=CFG.pseudo_threshold_low):
    """Add high-confidence predictions as pseudo-labels"""
    if not CFG.use_pseudo_labeling:
        return train_df
        
    # More accurate model for pseudo-label generation
    quick_params = {
        'objective': "binary",
        'metric': 'binary_logloss',
        'verbosity': -1,
        'random_state': CFG.seed,
        'learning_rate': 0.05,
        'num_leaves': 120,
        'max_depth': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'n_jobs': -1
    }
    
    X_train = train_df[features]
    y_train = train_df[CFG.target]
    X_test = test_df[features]
    
    # Categorical features
    X_train[features] = X_train[features].astype("category")
    X_test[features] = X_test[features].astype("category")
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(quick_params, dtrain, num_boost_round=3000)
    
    test_preds = model.predict(X_test)
    
    # Select only very confident predictions
    confident_mask = (test_preds > threshold_high) | (test_preds < threshold_low)
    pseudo_labels = (test_preds > 0.5).astype(int)
    
    if confident_mask.sum() > 0:
        pseudo_df = test_df[confident_mask].copy()
        pseudo_df[CFG.target] = pseudo_labels[confident_mask]
        
        print(f"Added {confident_mask.sum()} pseudo-labels ({confident_mask.mean():.2%} of test data)")
        return pd.concat([train_df, pseudo_df], ignore_index=True)
    
    return train_df

# Optimized LightGBM parameters
params = {
    'objective': "binary",
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': "gbdt",
    'random_state': CFG.seed,
    'learning_rate': CFG.learning_rate,
    'max_depth': 15,  # Increased depth
    'num_leaves': 200,  # Increased leaves
    'max_bin': 400,  # Increased bins
    'subsample': 0.85,  # Slightly increased
    'colsample_bytree': 0.7,  # Slightly decreased for regularization
    'subsample_freq': 1,
    'reg_alpha': 6.0,  # Increased L1 regularization
    'reg_lambda': 4.0,  # Increased L2 regularization
    'min_child_samples': 25,  # Increased
    'min_split_gain': 0.001,  # Decreased for more sensitivity
    'n_jobs': -1,
    'extra_trees': True,
    'bagging_seed': CFG.seed,
    'feature_fraction_seed': CFG.seed,
    'path_smooth': 0.2,
    'min_data_in_bin': 5,
    'feature_pre_filter': False,
    'lambda_l1': 0.5,  # Additional L1 regularization
    'lambda_l2': 0.3,  # Additional L2 regularization
    'cat_smooth': 20,  # Smoothing for categorical features
    'max_cat_to_onehot': 8,  # Limit on one-hot encoding
}

# Initialize improved target encoders
target_encoders = {}
if CFG.use_target_encoding:
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for feature in categorical_features:
        target_encoders[feature] = ImprovedTargetEncoder(smooth=CFG.target_encoding_smoothing, cv_folds=5)

# Apply extended feature engineering
print("Creating advanced features with improvements...")
train_enhanced, kmeans_model, scaler = create_advanced_features(train, target_encoders, fit_encoders=True)
test_enhanced = create_advanced_features(test, target_encoders, fit_encoders=False, kmeans_model=kmeans_model, scaler=scaler)

# Get list of all features
all_features = [col for col in train_enhanced.columns if col != CFG.target]
print(f"Total number of features: {len(all_features)}")

# Apply improved pseudo-labeling
print("Applying improved pseudo-labeling...")
train_with_pseudo = pseudo_labeling(train_enhanced, test_enhanced)

# Initialize arrays for predictions
oof = np.zeros(train.shape[0])
pred = np.zeros(test.shape[0])

# Cross-validation setup
cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
splitter = cv.split(train_enhanced, train_enhanced[CFG.target])

# Feature importance tracking
feature_importance = pd.DataFrame()

print(f"Starting improved {CFG.n_splits}-fold cross-validation...")

for fold, (trn_idx, val_idx) in enumerate(splitter):
    start_time = time.time()

    # Apply same transformations to original data
    original_enhanced = create_advanced_features(original, target_encoders, fit_encoders=False, 
                                               kmeans_model=kmeans_model, scaler=scaler)
    
    # Prepare training data
    train_fold = train_with_pseudo.iloc[trn_idx] if len(train_with_pseudo) > len(train_enhanced) else train_enhanced.iloc[trn_idx]
    M_train = pd.concat([train_fold, original_enhanced])
    M_train = M_train.drop_duplicates(subset=features, keep="first", ignore_index=True)
    
    X_train = M_train[all_features]
    y_train = M_train[CFG.target]

    X_valid = train_enhanced.loc[val_idx, all_features]
    y_valid = train_enhanced.loc[val_idx, CFG.target]
    X_test = test_enhanced[all_features].copy()
    
    # Process categorical features
    cat_features = [col for col in features if col in X_train.columns]
    X_train[cat_features] = X_train[cat_features].astype("category")
    X_valid[cat_features] = X_valid[cat_features].astype("category")
    X_test[cat_features] = X_test[cat_features].astype("category")

    # Create datasets for LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    # LightGBM callbacks
    ES = lgb.callback.early_stopping(stopping_rounds=CFG.early_stopping_rounds, verbose=False)
    LE = lgb.log_evaluation(period=CFG.verbose_eval, show_stdv=True)

    # Train the model
    model = lgb.train(params, 
                      train_set=dtrain,
                      valid_sets=[dtrain, dvalid],
                      valid_names=["train", "valid"],
                      num_boost_round=CFG.num_boost_round,
                      callbacks=[ES, LE])

    # Generate predictions
    oof[val_idx] = model.predict(X_valid)
    pred += model.predict(X_test) / CFG.n_splits

    # Track feature importance
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = model.feature_name()
    fold_importance["importance"] = model.feature_importance(importance_type='gain')
    fold_importance["fold"] = fold
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    # Calculate fold score
    score = roc_auc_score(y_valid, oof[val_idx])

    end_time = time.time()
    print("----------------------------------------------------------------")
    print(f"Fold: {fold:02d}, AUC: {score:.6f}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score['valid']['binary_logloss']:.6f}")
    print(f"Time: {end_time-start_time:.2f} sec.\n")

    # Memory cleanup
    del model, dtrain, dvalid
    gc.collect()

# Calculate final cross-validation score
score = roc_auc_score(train[CFG.target], oof)
print("================================================================")
print(f"Final Improved CV AUC: {score:.6f}")
print("================================================================")

# Feature importance analysis
feature_importance_agg = feature_importance.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
feature_importance_agg = feature_importance_agg.sort_values('mean', ascending=False)

print("\nTop 25 most important features:")
print("-" * 60)
for i, (_, row) in enumerate(feature_importance_agg.head(25).iterrows()):
    print(f"{i+1:2d}. {row['feature']:35s} {row['mean']:8.1f} ± {row['std']:6.1f}")

# Save results
np.save("improved_oof.npy", oof)
np.save("improved_pred.npy", pred)

# Create submission
submission = pd.read_csv(CFG.path + "sample_submission.csv")
submission[CFG.target] = pred

submission.to_csv("submission.csv", index=False)
print(f"\nImproved submission saved: improved_submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"Prediction stats - Min: {pred.min():.6f}, Max: {pred.max():.6f}, Mean: {pred.mean():.6f}")
print(f"Prediction std: {pred.std():.6f}")

# Additional statistics
print(f"\nModel performance analysis:")
print(f"OOF AUC: {score:.6f}")
print(f"OOF predictions - Min: {oof.min():.6f}, Max: {oof.max():.6f}, Mean: {oof.mean():.6f}")
print(f"Feature importance top 5: {feature_importance_agg.head(5)['feature'].tolist()}")

