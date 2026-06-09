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


#!/usr/bin/env python
# coding: utf-8

"""
DRW Crypto Market Prediction Competition
Complete Solution with Advanced Feature Engineering
"""

# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages if not already installed"""
    packages = ['pandas', 'numpy', 'lightgbm', 'scikit-learn', 'scipy']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])

# Run installation
install_packages()

# Import libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler, Nystroem
import gc
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
class Config:
    """Configuration settings for the model"""
    # Paths - Updated for Kaggle environment
    train_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
    test_path = '/kaggle/input/drw-crypto-market-prediction/test.parquet'
    sample_submission_path = '/kaggle/input/drw-crypto-market-prediction/sample_submission.csv'
    submission_path = 'submission.csv'
    
    # Model parameters
    seed = 42
    n_folds = 5
    
    # Memory-efficient LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'seed': seed,
        'n_jobs': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        # Memory optimization parameters
        'max_bin': 63,
        'min_data_in_bin': 5,
        'feature_pre_filter': False,
        'force_col_wise': True,
        'histogram_pool_size': -1,
    }
    
    # Training parameters
    early_stopping_rounds = 100
    n_estimators = 1000
    verbose_eval = 100
    
    # Feature engineering
    use_recent_months_only = True
    recent_months = 3
    
    # Feature selection parameters
    use_feature_selection = True
    feature_selection_threshold = 0.7
    min_features_to_keep = 100
    
    # Feature engineering parameters
    feature_engineering_config = {
        'max_new_features_per_type': 5,
        'feature_generation_log': [],
        'round_1': {
            'polynomial': {
                'enabled': True,
                'degree': 2,
                'interaction_only': True,
                'include_bias': False,
                'max_features': 10,
                'feature_selection': 'importance',
                'max_new_features': 5
            },
            'kernel': {
                'enabled': True,
                'kernels': ['rbf'],
                'n_components': 5,
                'gamma': 0.1,
                'subset_size': 0.3,
                'max_new_features': 5
            },
            'statistical': {
                'enabled': True,
                'transforms': ['log1p', 'sqrt'],
                'threshold': 0.01,
                'max_new_features': 5
            },
            'custom': {
                'enabled': True,
                'ratios': True,
                'differences': False,
                'products': True,
                'max_new_features': 5
            }
        },
        'round_2': {
            'polynomial': {
                'enabled': True,
                'degree': 2,
                'interaction_only': True,
                'include_bias': False,
                'max_features': 8,
                'feature_selection': 'importance',
                'max_new_features': 5
            },
            'kernel': {
                'enabled': False,
                'kernels': ['rbf'],
                'n_components': 5,
                'gamma': 0.05,
                'subset_size': 0.2,
                'max_new_features': 5
            },
            'statistical': {
                'enabled': False,
                'transforms': ['log1p'],
                'threshold': 0.01,
                'max_new_features': 5
            },
            'custom': {
                'enabled': True,
                'ratios': True,
                'differences': False,
                'products': True,
                'max_new_features': 5
            }
        }
    }

def reduce_mem_usage(dataframe, dataset_name):
    """Reduce memory usage by optimizing data types"""
    print(f'Reducing memory usage for: {dataset_name}')
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2

    for col in dataframe.columns:
        col_type = dataframe[col].dtype
        
        if col_type != 'object' and col != 'timestamp':
            c_min = dataframe[col].min()
            c_max = dataframe[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dataframe[col] = dataframe[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dataframe[col] = dataframe[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dataframe[col] = dataframe[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dataframe[col] = dataframe[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dataframe[col] = dataframe[col].astype(np.float32)
                else:
                    dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
    print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
    print(f'--- Decreased memory usage by {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%\n')

    return dataframe

def log_feature_generation(round_num, transform_type, feature_names, original_count):
    """Log information about generated features"""
    log_entry = {
        'round': round_num,
        'transform_type': transform_type,
        'features_generated': feature_names,
        'num_generated': len(feature_names),
        'original_candidates': original_count,
        'timestamp': pd.Timestamp.now()
    }
    Config.feature_engineering_config['feature_generation_log'].append(log_entry)
    
    print(f"Generated {len(feature_names)} {transform_type} features from {original_count} candidates")

def apply_statistical_transforms(X, config, round_num):
    """Apply statistical transformations to features with controlled generation"""
    if not config['enabled']:
        return X
    
    X_transformed = X.copy()
    potential_features = []
    
    for col in X.columns:
        if X[col].std() > config['threshold']:
            for transform in config['transforms']:
                if transform == 'log1p':
                    if X[col].min() >= 0:
                        new_col = np.log1p(X[col])
                        potential_features.append((f'{col}_log1p', new_col))
                elif transform == 'sqrt':
                    if X[col].min() >= 0:
                        new_col = np.sqrt(X[col])
                        potential_features.append((f'{col}_sqrt', new_col))
    
    original_count = len(potential_features)
    if len(potential_features) > config['max_new_features']:
        np.random.seed(42 + round_num)
        indices = np.random.choice(len(potential_features), config['max_new_features'], replace=False)
        selected_features = [potential_features[i] for i in indices]
    else:
        selected_features = potential_features
    
    feature_names = []
    for name, values in selected_features:
        X_transformed[name] = values.astype(np.float32)
        feature_names.append(name)
    
    log_feature_generation(round_num, 'statistical', feature_names, original_count)
    
    return X_transformed

def apply_custom_features(X, config, round_num):
    """Apply custom feature engineering with controlled generation"""
    if not config['enabled']:
        return X
    
    X_transformed = X.copy()
    potential_features = []
    
    feature_cols = X.columns.tolist()
    n_features = min(10, len(feature_cols))
    np.random.seed(42 + round_num)
    selected_features = np.random.choice(feature_cols, n_features, replace=False)
    
    for i, col1 in enumerate(selected_features):
        for col2 in selected_features[i+1:]:
            if config['ratios']:
                denominator = X[col2] + 1e-8
                ratio = X[col1] / denominator
                if ratio.std() > 0.01:
                    potential_features.append((f'{col1}_ratio_{col2}', ratio, 'ratio'))
            
            if config['products']:
                prod = X[col1] * X[col2]
                if prod.std() > 0.01:
                    potential_features.append((f'{col1}_prod_{col2}', prod, 'product'))
    
    original_count = len(potential_features)
    if len(potential_features) > config['max_new_features']:
        np.random.seed(43 + round_num)
        indices = np.random.choice(len(potential_features), config['max_new_features'], replace=False)
        selected_features = [potential_features[i] for i in indices]
    else:
        selected_features = potential_features
    
    feature_names = []
    feature_types = {'ratio': 0, 'difference': 0, 'product': 0}
    for name, values, feat_type in selected_features:
        X_transformed[name] = values.astype(np.float32)
        feature_names.append(name)
        feature_types[feat_type] += 1
    
    log_entry_detail = f"Generated: {feature_types['ratio']} ratios, {feature_types['product']} products"
    print(f"  {log_entry_detail}")
    log_feature_generation(round_num, 'custom', feature_names, original_count)
    
    return X_transformed

def apply_polynomial_features(X, config, feature_importance, round_num):
    """Apply polynomial feature transformations with controlled generation"""
    if not config['enabled']:
        return X
    
    if config['feature_selection'] == 'importance' and feature_importance is not None:
        top_features = feature_importance.head(config['max_features'])['feature'].tolist()
        top_features = [f for f in top_features if f in X.columns]
    else:
        np.random.seed(44 + round_num)
        top_features = np.random.choice(X.columns, 
                                      min(config['max_features'], len(X.columns)), 
                                      replace=False).tolist()
    
    print(f"Creating polynomial features from {len(top_features)} selected features")
    
    poly = PolynomialFeatures(
        degree=config['degree'],
        interaction_only=config['interaction_only'],
        include_bias=config['include_bias']
    )
    
    X_poly = poly.fit_transform(X[top_features])
    feature_names = poly.get_feature_names_out(top_features)
    
    poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    poly_df = poly_df.drop(columns=top_features, errors='ignore')
    
    potential_features = [(col, poly_df[col]) for col in poly_df.columns]
    original_count = len(potential_features)
    
    if len(potential_features) > config['max_new_features']:
        np.random.seed(45 + round_num)
        indices = np.random.choice(len(potential_features), config['max_new_features'], replace=False)
        selected_features = [potential_features[i] for i in indices]
    else:
        selected_features = potential_features
    
    X_transformed = X.copy()
    feature_names = []
    for name, values in selected_features:
        X_transformed[name] = values.astype(np.float32)
        feature_names.append(name)
    
    log_feature_generation(round_num, 'polynomial', feature_names, original_count)
    
    del poly_df, X_poly
    gc.collect()
    
    return X_transformed

def apply_kernel_features(X, config, round_num):
    """Apply kernel transformations with controlled generation"""
    if not config['enabled']:
        return X
    
    X_transformed = X.copy()
    
    n_samples = int(len(X) * config['subset_size'])
    sample_indices = np.random.choice(len(X), n_samples, replace=False)
    
    for kernel_idx, kernel_type in enumerate(config['kernels']):
        print(f"Applying {kernel_type} kernel transformation...")
        
        max_components_per_kernel = config['max_new_features'] // len(config['kernels'])
        n_components = min(config['n_components'], max_components_per_kernel)
        
        feature_names = []
        
        if kernel_type == 'rbf':
            rbf_sampler = RBFSampler(
                gamma=config['gamma'],
                n_components=n_components,
                random_state=42 + round_num + kernel_idx
            )
            
            rbf_sampler.fit(X.iloc[sample_indices])
            X_kernel = rbf_sampler.transform(X)
            
            for i in range(X_kernel.shape[1]):
                feat_name = f'rbf_component_{i}'
                X_transformed[feat_name] = X_kernel[:, i].astype(np.float32)
                feature_names.append(feat_name)
        
        log_feature_generation(round_num, f'{kernel_type}_kernel', feature_names, n_components)
        
        del X_kernel
        gc.collect()
    
    return X_transformed

def apply_feature_engineering_round(X, config_round, round_num, feature_importance=None):
    """Apply a complete round of feature engineering with controlled generation"""
    print(f"\n=== Feature Engineering Round {round_num} ===")
    initial_features = X.shape[1]
    
    X_transformed = X
    
    if 'statistical' in config_round:
        X_transformed = apply_statistical_transforms(X_transformed, config_round['statistical'], round_num)
        gc.collect()
    
    if 'custom' in config_round:
        X_transformed = apply_custom_features(X_transformed, config_round['custom'], round_num)
        gc.collect()
    
    if 'polynomial' in config_round:
        X_transformed = apply_polynomial_features(X_transformed, config_round['polynomial'], feature_importance, round_num)
        gc.collect()
    
    if 'kernel' in config_round:
        X_transformed = apply_kernel_features(X_transformed, config_round['kernel'], round_num)
        gc.collect()
    
    print(f"Features increased from {initial_features} to {X_transformed.shape[1]}")
    print(f"Net new features added: {X_transformed.shape[1] - initial_features}")
    
    variance_threshold = 1e-8
    low_variance_cols = X_transformed.columns[X_transformed.var() < variance_threshold]
    if len(low_variance_cols) > 0:
        X_transformed = X_transformed.drop(columns=low_variance_cols)
        print(f"Removed {len(low_variance_cols)} low variance features")
    
    return X_transformed

def remove_highly_correlated_features(X, threshold=0.99):
    """Remove features with correlation above threshold"""
    print(f"\nRemoving features with correlation > {threshold}")
    
    n_features = X.shape[1]
    chunk_size = 100
    to_remove = set()
    
    for i in range(0, n_features, chunk_size):
        for j in range(i, n_features, chunk_size):
            chunk_i = X.iloc[:, i:min(i+chunk_size, n_features)]
            chunk_j = X.iloc[:, j:min(j+chunk_size, n_features)]
            
            corr_chunk = chunk_i.corrwith(chunk_j)
            
            for idx, corr_val in corr_chunk.items():
                if abs(corr_val) > threshold and chunk_i.columns[0] != idx:
                    if chunk_i.columns[0] < idx:
                        to_remove.add(idx)
                    else:
                        to_remove.add(chunk_i.columns[0])
    
    X_reduced = X.drop(columns=list(to_remove))
    print(f"Removed {len(to_remove)} highly correlated features")
    
    return X_reduced

def save_feature_generation_report(log_data, filename='feature_generation_report.txt'):
    """Save a detailed report of all feature generation steps"""
    with open(filename, 'w') as f:
        f.write("Feature Generation Report\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in log_data:
            f.write(f"Round {entry['round']} - {entry['transform_type']}\n")
            f.write(f"Timestamp: {entry['timestamp']}\n")
            f.write(f"Generated {entry['num_generated']} features from {entry['original_candidates']} candidates\n")
            f.write(f"Selected ratio: {entry['num_generated']/max(entry['original_candidates'], 1):.2%}\n")
            f.write("Generated features:\n")
            for i, feat in enumerate(entry['features_generated']):
                f.write(f"  {i+1}. {feat}\n")
            f.write("\n" + "-" * 80 + "\n\n")
        
        f.write("Summary Statistics\n")
        f.write("=" * 80 + "\n")
        total_generated = sum(entry['num_generated'] for entry in log_data)
        total_candidates = sum(entry['original_candidates'] for entry in log_data)
        f.write(f"Total features generated: {total_generated}\n")
        f.write(f"Total candidate features considered: {total_candidates}\n")
        f.write(f"Overall selection ratio: {total_generated/max(total_candidates, 1):.2%}\n")
    
    print(f"\nFeature generation report saved to {filename}")

def load_data():
    """Load train and test data from parquet files with memory optimization"""
    print("Loading data...")
    
    train_df = pd.read_parquet(Config.train_path)
    test_df = pd.read_parquet(Config.test_path)
    
    print(f"Initial train shape: {train_df.shape}")
    print(f"Initial test shape: {test_df.shape}")
    
    train_df = reduce_mem_usage(train_df, 'train')
    test_df = reduce_mem_usage(test_df, 'test')
    
    gc.collect()
    
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Preprocess the data with memory efficiency in mind"""
    print("\nPreprocessing data...")
    
    if Config.use_recent_months_only and 'timestamp' in train_df.columns:
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        
        cutoff_date = train_df['timestamp'].max() - pd.DateOffset(months=Config.recent_months)
        train_df = train_df[train_df['timestamp'] >= cutoff_date].reset_index(drop=True)
        print(f"Using data from {cutoff_date} onwards. New train shape: {train_df.shape}")
        
        gc.collect()
    
    if 'timestamp' in train_df.columns:
        train_df = train_df.drop('timestamp', axis=1)
    if 'timestamp' in test_df.columns:
        test_df = test_df.drop('timestamp', axis=1)
    
    feature_cols = [col for col in train_df.columns if col != 'label']
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['label'].copy()
    X_test = test_df[feature_cols].copy()
    
    del train_df
    gc.collect()
    
    if X_train.isnull().sum().sum() > 0:
        print("Handling missing values...")
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    
    print("Adding engineered features...")
    
    X_train['bid_ask_imbalance'] = (X_train['bid_qty'] - X_train['ask_qty']) / (X_train['bid_qty'] + X_train['ask_qty'] + 1e-8)
    X_train['buy_sell_imbalance'] = (X_train['buy_qty'] - X_train['sell_qty']) / (X_train['buy_qty'] + X_train['sell_qty'] + 1e-8)
    
    X_test['bid_ask_imbalance'] = (X_test['bid_qty'] - X_test['ask_qty']) / (X_test['bid_qty'] + X_test['ask_qty'] + 1e-8)
    X_test['buy_sell_imbalance'] = (X_test['buy_qty'] - X_test['sell_qty']) / (X_test['buy_qty'] + X_test['sell_qty'] + 1e-8)
    
    for col in ['bid_ask_imbalance', 'buy_sell_imbalance']:
        X_train[col] = X_train[col].astype(np.float32)
        X_test[col] = X_test[col].astype(np.float32)
    
    print(f"Final train features: {X_train.shape[1]}")
    
    gc.collect()
    
    return X_train, y_train, X_test

def pearson_correlation(y_true, y_pred):
    """Calculate Pearson correlation coefficient"""
    return pearsonr(y_true, y_pred)[0]

def select_features_by_importance(X_train, y_train, X_test, threshold=0.7, min_features=100):
    """Perform feature selection based on LightGBM feature importance"""
    print("\nPerforming feature selection...")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'verbosity': -1,
        'seed': 42,
        'n_jobs': -1,
        'force_col_wise': True,
        'max_bin': 63,
    }
    
    sample_size = min(50000, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train.iloc[sample_idx]
    y_sample = y_train.iloc[sample_idx]
    
    train_set = lgb.Dataset(X_sample, y_sample, free_raw_data=True)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=100,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    n_features = max(int(len(X_train.columns) * threshold), min_features)
    n_features = min(n_features, len(X_train.columns))
    
    selected_features = importance_df.head(n_features)['feature'].tolist()
    
    print(f"Selected {len(selected_features)} features out of {len(X_train.columns)}")
    print(f"Top 10 features: {selected_features[:10]}")
    
    del model, train_set, X_sample, y_sample
    gc.collect()
    
    return X_train[selected_features], X_test[selected_features], selected_features

def train_lightgbm_single_fold(X_train, y_train, X_test, fold_data):
    """Train a single fold of LightGBM with memory optimization"""
    fold, train_idx, valid_idx = fold_data
    print(f"\nFold {fold + 1}/{Config.n_folds}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    train_set = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
    valid_set = lgb.Dataset(X_val, y_val, free_raw_data=True, reference=train_set)
    
    model = lgb.train(
        Config.lgb_params,
        train_set,
        num_boost_round=Config.n_estimators,
        valid_sets=[valid_set],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(Config.early_stopping_rounds),
            lgb.log_evaluation(Config.verbose_eval)
        ]
    )
    
    val_predictions = model.predict(X_val, num_iteration=model.best_iteration)
    test_predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    fold_score = pearson_correlation(y_val, val_predictions)
    print(f"Fold {fold + 1} Pearson correlation: {fold_score:.6f}")
    
    importance = model.feature_importance(importance_type='gain')
    
    del train_set, valid_set, X_tr, X_val, y_tr, y_val
    gc.collect()
    
    return val_predictions, test_predictions, valid_idx, importance, fold_score

def train_lightgbm_cv(X_train, y_train, X_test):
    """Train LightGBM with cross-validation using memory-efficient approach"""
    print("\nTraining LightGBM with cross-validation...")
    
    oof_predictions = np.zeros(len(X_train), dtype=np.float32)
    test_predictions = np.zeros(len(X_test), dtype=np.float32)
    
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = X_train.columns
    
    kf = KFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    
    scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        fold_data = (fold, train_idx, valid_idx)
        val_preds, test_preds, valid_idx, importance, fold_score = train_lightgbm_single_fold(
            X_train, y_train, X_test, fold_data
        )
        
        oof_predictions[valid_idx] = val_preds
        test_predictions += test_preds / Config.n_folds
        
        feature_importance[f'fold_{fold + 1}'] = importance
        scores.append(fold_score)
        
        gc.collect()
    
    oof_score = pearson_correlation(y_train, oof_predictions)
    print(f"\nOverall OOF Pearson correlation: {oof_score:.6f}")
    print(f"Mean CV score: {np.mean(scores):.6f} (+/- {np.std(scores):.6f})")
    
    feature_importance['importance'] = feature_importance[[f'fold_{i+1}' for i in range(Config.n_folds)]].mean(axis=1)
    feature_importance = feature_importance[['feature', 'importance']].sort_values('importance', ascending=False)
    
    print("\nTop 20 most important features:")
    print(feature_importance.head(20))
    
    return test_predictions, oof_predictions, feature_importance

def create_submission(test_predictions, test_df):
    """Create submission file"""
    print("\nCreating submission file...")
    
    # Load sample submission to get the correct format
    sample_sub = pd.read_csv(Config.sample_submission_path)
    
    # Check what columns are available in sample submission
    print(f"Sample submission columns: {sample_sub.columns.tolist()}")
    print(f"Sample submission shape: {sample_sub.shape}")
    
    # Create submission dataframe based on available columns
    if 'id' in sample_sub.columns:
        submission = pd.DataFrame({
            'id': sample_sub['id'],
            'prediction': test_predictions
        })
    elif 'ID' in sample_sub.columns:
        submission = pd.DataFrame({
            'ID': sample_sub['ID'],
            'prediction': test_predictions
        })
    elif len(sample_sub.columns) == 1:
        # If sample submission only has label column, use index as ID
        submission = pd.DataFrame({
            'prediction': test_predictions
        })
        submission.index = sample_sub.index
    else:
        # Use the first column as ID column (whatever it's named)
        id_column = sample_sub.columns[0]
        submission = pd.DataFrame({
            id_column: sample_sub[id_column],
            'prediction': test_predictions
        })
    
    # Ensure we have predictions for all test samples
    assert len(submission) == len(sample_sub), f"Submission length mismatch: {len(submission)} vs {len(sample_sub)}"
    
    # Save submission with the same index structure as sample
    submission.to_csv(Config.submission_path, index=('id' not in sample_sub.columns and 'ID' not in sample_sub.columns))
    print(f"Submission saved to {Config.submission_path}")
    
    # Display statistics
    print(f"\nSubmission statistics:")
    print(f"Mean: {test_predictions.mean():.6f}")
    print(f"Std: {test_predictions.std():.6f}")
    print(f"Min: {test_predictions.min():.6f}")
    print(f"Max: {test_predictions.max():.6f}")
    
    return submission

def main():
    """Main execution function with two-stage feature engineering"""
    print("DRW Crypto Price Movement Prediction - Advanced Feature Engineering")
    print("=" * 65)
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set random seed for reproducibility
    np.random.seed(Config.seed)
    
    # Load data with memory optimization
    train_df, test_df = load_data()
    
    # Store test_df reference for submission
    test_df_copy = test_df.copy()
    
    # Initial preprocessing
    X_train, y_train, X_test = preprocess_data(train_df, test_df)
    
    # Clean up
    del train_df, test_df
    gc.collect()
    
    print("\n=== Stage 1: Initial Feature Reduction ===")
    
    # Remove zero-variance and duplicate columns
    initial_features = X_train.shape[1]
    X_train = X_train.loc[:, X_train.nunique() > 1]
    X_test = X_test[X_train.columns]
    
    # Remove duplicate columns
    duplicate_cols = []
    for i in range(len(X_train.columns)):
        for j in range(i + 1, len(X_train.columns)):
            if X_train.iloc[:, i].equals(X_train.iloc[:, j]):
                duplicate_cols.append(X_train.columns[j])
    
    X_train = X_train.drop(columns=duplicate_cols)
    X_test = X_test.drop(columns=duplicate_cols)
    print(f"Removed {initial_features - X_train.shape[1]} zero-variance/duplicate features")
    
    # Remove highly correlated features
    X_train = remove_highly_correlated_features(X_train, threshold=0.99)
    X_test = X_test[X_train.columns]
    
    gc.collect()
    
    # First LightGBM feature selection
    if Config.use_feature_selection:
        print("\n=== First LightGBM Feature Selection ===")
        X_train, X_test, selected_features = select_features_by_importance(
            X_train, y_train, X_test, 
            threshold=0.5,
            min_features=200
        )
        gc.collect()
    
    # First round of feature engineering
    if 'round_1' in Config.feature_engineering_config:
        X_train = apply_feature_engineering_round(
            X_train, 
            Config.feature_engineering_config['round_1'], 
            round_num=1
        )
        X_test = apply_feature_engineering_round(
            X_test, 
            Config.feature_engineering_config['round_1'], 
            round_num=1
        )
        
        # Ensure same columns in train and test
        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        
        gc.collect()
    
    # Second LightGBM feature selection
    print("\n=== Second LightGBM Feature Selection ===")
    X_train, X_test, selected_features = select_features_by_importance(
        X_train, y_train, X_test, 
        threshold=Config.feature_selection_threshold,
        min_features=Config.min_features_to_keep
    )
    
    # Get feature importance for second round
    temp_importance = pd.DataFrame({
        'feature': selected_features[:30],
        'importance': range(30, 0, -1)
    })
    
    gc.collect()
    
    # Second round of feature engineering
    if 'round_2' in Config.feature_engineering_config:
        X_train = apply_feature_engineering_round(
            X_train, 
            Config.feature_engineering_config['round_2'], 
            round_num=2,
            feature_importance=temp_importance
        )
        X_test = apply_feature_engineering_round(
            X_test, 
            Config.feature_engineering_config['round_2'], 
            round_num=2,
            feature_importance=temp_importance
        )
        
        # Ensure same columns
        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
        
        gc.collect()
    
    # Final feature selection
    print("\n=== Final Feature Selection ===")
    X_train, X_test, selected_features = select_features_by_importance(
        X_train, y_train, X_test, 
        threshold=0.8,
        min_features=100
    )
    
    print(f"\nFinal feature count: {X_train.shape[1]}")
    gc.collect()
    
    # Train final model with cross-validation
    print("\n=== Training Final Model ===")
    cv_predictions, oof_predictions, feature_importance = train_lightgbm_cv(X_train, y_train, X_test)
    
    # Use CV predictions
    test_predictions = cv_predictions
    
    # Create submission
    submission = create_submission(test_predictions, test_df_copy)
    
    # Final cleanup
    del X_train, y_train, X_test, test_df_copy
    gc.collect()
    
    # Save feature generation report
    if len(Config.feature_engineering_config['feature_generation_log']) > 0:
        save_feature_generation_report(
            Config.feature_engineering_config['feature_generation_log'],
            'feature_generation_report.txt'
        )
    
    print("\nTraining complete!")
    print(f"Final model used {len(selected_features)} features after two rounds of engineering")
    print(f"Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return submission, feature_importance

if __name__ == "__main__":
    submission, feature_importance = main()
    print("\nSubmission file created successfully!")
    print("Please submit the 'submission.csv' file to the competition.")

