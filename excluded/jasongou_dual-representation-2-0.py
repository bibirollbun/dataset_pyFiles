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


#!/usr/bin/env python3
"""
DRW Crypto Market Prediction - Complete Solution with AutoML Options
===================================================================
Includes Optuna, LightAutoML, and FLAML for comprehensive optimization
"""

import gc
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Try to import optimization libraries
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available - will use default parameters")

try:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
    LIGHTAUTOML_AVAILABLE = True
except ImportError:
    LIGHTAUTOML_AVAILABLE = False
    print("LightAutoML not available")

try:
    from flaml import AutoML as FLAMLAutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    print("FLAML not available")

# Set random seed
np.random.seed(42)

# Check environment
KAGGLE_INPUT_PATH = '/kaggle/input/drw-crypto-market-prediction'
DATA_PATH = KAGGLE_INPUT_PATH if os.path.exists(KAGGLE_INPUT_PATH) else '.'

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = os.path.join(DATA_PATH, "train.parquet")
    TEST_PATH = os.path.join(DATA_PATH, "test.parquet")
    SUBMISSION_PATH = os.path.join(DATA_PATH, "sample_submission.csv")
    
    # Core features - proven to work well
    CORE_FEATURES = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", 
        "X860", "X674", "X415", "X345", "X137", "X855", "X174", "X302", 
        "X178", "X532", "X168", "X612", "X888", "X421", "X333"
    ]
    
    MARKET_FEATURES = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    
    # Features for dual representation
    DUAL_FEATURES = ["X612", "X860", "X168", "X174", "X333", "X345"]
    
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    
    # Memory optimization
    USE_FLOAT32 = True
    
    # Sampling strategy
    SELECTION_SAMPLES = [100000, 200000, 300000]
    SELECTION_RATIOS = [0.8, 0.7, 0.6]
    MAX_ADDITIONAL_FEATURES = 50
    
    # Tuning parameters
    USE_AUTOML = True  # Use AutoML if available
    N_TUNING_TRIALS = 20 if OPTUNA_AVAILABLE else 0
    TUNING_SAMPLE_SIZE = 100000
    
    # AutoML settings
    LIGHTAUTOML_TIMEOUT = 300  # 5 minutes
    FLAML_TIME_BUDGET = 300    # 5 minutes

# Default parameters if optimization is not available
DEFAULT_XGB_PARAMS = {
    'tree_method': 'hist',
    'device': 'cpu',
    'n_estimators': 500,
    'max_depth': 10,
    'learning_rate': 0.03,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 1.5,
    'reg_alpha': 10,
    'reg_lambda': 10,
    'random_state': Config.RANDOM_STATE,
    'n_jobs': 2,
    'verbosity': 0
}

DEFAULT_LGB_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 500,
    'num_leaves': 31,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 50,
    'random_state': Config.RANDOM_STATE,
    'n_jobs': 2,
    'verbose': -1
}

# =========================
# Memory Management
# =========================
def reduce_mem_usage(dataframe, verbose=True):
    """Optimized memory reduction function"""
    if verbose:
        print('Reducing memory usage...')
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype
        if col_type != 'object':
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
                if Config.USE_FLOAT32 and c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dataframe[col] = dataframe[col].astype(np.float32)
                elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dataframe[col] = dataframe[col].astype(np.float16)
                else:
                    dataframe[col] = dataframe[col].astype(np.float32)
    
    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    if verbose:
        print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
        print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
        print(f'--- Decreased by {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%\n')
    
    return dataframe

def print_memory_usage():
    """Print current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"Current memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    except:
        pass

def clean_memory():
    """Force garbage collection"""
    gc.collect()

# =========================
# Feature Engineering
# =========================
class MemoryEfficientFeatureEngineer:
    """Memory-efficient feature engineering"""
    
    def __init__(self):
        self.selected_features = None
        self.feature_stats = {}
    
    def create_rank_features_batch(self, df, features, batch_size=10):
        """Create rank features in batches to save memory"""
        rank_features = []
        
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size]
            batch_df = pd.DataFrame(index=df.index)
            
            for feature in batch_features:
                if feature in df.columns:
                    batch_df[f"{feature}_rank"] = df[feature].rank(method='dense', pct=True).astype(np.float32)
            
            rank_features.append(batch_df)
            clean_memory()
        
        if rank_features:
            return pd.concat(rank_features, axis=1)
        else:
            return pd.DataFrame(index=df.index)
    
    def create_essential_features(self, df):
        """Create only the most essential engineered features"""
        features = pd.DataFrame(index=df.index)
        
        # Core market microstructure features
        if all(col in df.columns for col in ['bid_qty', 'ask_qty']):
            total = df['bid_qty'] + df['ask_qty'] + 1e-8
            features['bid_ask_imbalance'] = ((df['bid_qty'] - df['ask_qty']) / total).astype(np.float32)
            features['bid_ratio'] = (df['bid_qty'] / total).astype(np.float32)
        
        if all(col in df.columns for col in ['buy_qty', 'sell_qty']):
            total = df['buy_qty'] + df['sell_qty'] + 1e-8
            features['buy_sell_pressure'] = ((df['buy_qty'] - df['sell_qty']) / total).astype(np.float32)
        
        if 'volume' in df.columns and all(col in df.columns for col in ['bid_qty', 'ask_qty']):
            features['liquidity'] = ((df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-8)).astype(np.float32)
        
        return features
    
    def create_top_interactions(self, df, top_features, n_interactions=10):
        """Create interactions only between top features"""
        interactions = pd.DataFrame(index=df.index)
        
        # Only use top 5 features for interactions
        interaction_features = [f for f in top_features if f in df.columns][:5]
        
        count = 0
        for i in range(len(interaction_features)):
            for j in range(i+1, len(interaction_features)):
                if count >= n_interactions:
                    break
                f1, f2 = interaction_features[i], interaction_features[j]
                interactions[f'{f1}_x_{f2}'] = (df[f1] * df[f2]).astype(np.float32)
                count += 1
        
        return interactions

# =========================
# Progressive Feature Selection
# =========================
class OptimizedFeatureSelector:
    """Memory-efficient feature selection"""
    
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
    
    def evaluate_features_chunked(self, train_df, candidate_features, label_col, chunk_size=50):
        """Evaluate features in chunks to save memory"""
        scores = {}
        
        for i in range(0, len(candidate_features), chunk_size):
            chunk_features = candidate_features[i:i+chunk_size]
            
            # Calculate correlations
            for feat in chunk_features:
                if feat in train_df.columns:
                    corr = abs(train_df[feat].corr(train_df[label_col]))
                    if not np.isnan(corr):
                        scores[feat] = corr
            
            clean_memory()
        
        return scores
    
    def progressive_selection(self, train_df, baseline_features, max_features=50):
        """Progressive feature selection with larger samples"""
        print("\n=== Progressive Feature Selection ===")
        
        # Get candidate features
        all_features = [col for col in train_df.columns if col.startswith('X')]
        candidates = [f for f in all_features if f not in baseline_features]
        
        if not candidates:
            return []
        
        print(f"Evaluating {len(candidates)} candidate features...")
        current_features = candidates
        
        # Progressive selection with increasing samples
        for stage, (sample_size, keep_ratio) in enumerate(zip(Config.SELECTION_SAMPLES, Config.SELECTION_RATIOS), 1):
            if len(current_features) <= max_features:
                break
            
            # Use minimum of sample size and available data
            actual_sample_size = min(sample_size, len(train_df))
            print(f"\nStage {stage}: Using {actual_sample_size:,} samples (keeping {keep_ratio:.0%})")
            
            # Sample recent data
            start_idx = max(0, len(train_df) - actual_sample_size)
            sample_df = train_df.iloc[start_idx:start_idx + actual_sample_size]
            
            # Evaluate features
            scores = self.evaluate_features_chunked(sample_df, current_features, Config.LABEL_COLUMN)
            
            # Sort by score
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Keep top features
            n_keep = int(len(sorted_features) * keep_ratio)
            n_keep = min(n_keep, max_features)
            current_features = [feat for feat, _ in sorted_features[:n_keep]]
            
            print(f"  Kept {len(current_features)} features (top score: {sorted_features[0][1]:.4f})")
            
            clean_memory()
        
        # Final selection
        self.selected_features = current_features[:max_features]
        self.feature_scores = {f: s for f, s in scores.items() if f in self.selected_features}
        
        print(f"\nFinal selection: {len(self.selected_features)} features")
        return self.selected_features

# =========================
# Hyperparameter Tuning with Multiple Options
# =========================
class ModelTuner:
    """Hyperparameter tuning with Optuna"""
    
    def __init__(self, model_type='xgb'):
        self.model_type = model_type
        self.best_params = None
    
    def objective(self, trial, X_train, y_train, X_valid, y_valid):
        """Objective function for Optuna"""
        
        if self.model_type == 'xgb':
            params = {
                'tree_method': 'hist',
                'device': 'cpu',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 100),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 100),
                'random_state': Config.RANDOM_STATE,
                'n_jobs': 2,
                'verbosity': 0
            }
            
            model = XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 
                     verbose=False, early_stopping_rounds=50)
            
        else:  # lgb
            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'random_state': Config.RANDOM_STATE,
                'n_jobs': 2,
                'verbose': -1
            }
            
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        
        # Predict and calculate score
        y_pred = model.predict(X_valid)
        score = pearsonr(y_valid, y_pred)[0]
        
        return score
    
    def tune(self, train_df, features, n_trials=20):
        """Run hyperparameter tuning"""
        if not OPTUNA_AVAILABLE:
            print(f"  Optuna not available - using default {self.model_type.upper()} parameters")
            if self.model_type == 'xgb':
                return DEFAULT_XGB_PARAMS
            else:
                return DEFAULT_LGB_PARAMS
        
        print(f"\nTuning {self.model_type.upper()} hyperparameters with Optuna...")
        
        # Use a sample for tuning
        sample_size = min(Config.TUNING_SAMPLE_SIZE, len(train_df))
        sample_df = train_df.tail(sample_size)
        
        # Split data
        X = sample_df[features].values
        y = sample_df[Config.LABEL_COLUMN].values
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=Config.RANDOM_STATE
        )
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=Config.RANDOM_STATE)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_valid, y_valid),
            n_trials=n_trials
        )
        
        self.best_params = study.best_params
        print(f"Best score: {study.best_value:.4f}")
        
        # Add fixed parameters
        if self.model_type == 'xgb':
            self.best_params.update({
                'tree_method': 'hist',
                'device': 'cpu',
                'random_state': Config.RANDOM_STATE,
                'n_jobs': 2,
                'verbosity': 0
            })
        else:
            self.best_params.update({
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': Config.RANDOM_STATE,
                'n_jobs': 2,
                'verbose': -1
            })
        
        return self.best_params

# =========================
# AutoML Training Functions
# =========================
def train_with_lightautoml(train_df, test_df, features):
    """Train using LightAutoML - Fixed version"""
    if not LIGHTAUTOML_AVAILABLE:
        return None, 0
    
    print("\nTraining with LightAutoML...")
    
    # Prepare data - LightAutoML expects a single DataFrame with target column
    train_data = train_df[features + [Config.LABEL_COLUMN]].copy()
    test_data = test_df[features].copy()
    
    # Create AutoML task
    task = Task('reg', metric='r2')
    
    # Initialize AutoML
    automl = TabularAutoML(
        task=task,
        timeout=Config.LIGHTAUTOML_TIMEOUT,
        cpu_limit=2,
        reader_params={'n_jobs': 2}
    )
    
    # Train - pass the full DataFrame and specify roles
    oof_pred = automl.fit_predict(
        train_data,
        roles={'target': Config.LABEL_COLUMN}
    )
    
    # Score
    score = pearsonr(train_df[Config.LABEL_COLUMN], oof_pred.data[:, 0])[0]
    print(f"LightAutoML CV Score: {score:.4f}")
    
    # Test predictions
    test_pred = automl.predict(test_data).data[:, 0]
    
    return test_pred, score

def train_with_flaml(train_df, test_df, features):
    """Train using FLAML AutoML"""
    if not FLAML_AVAILABLE:
        return None, 0
    
    print("\nTraining with FLAML AutoML...")
    
    # Prepare data
    X_train = train_df[features].values
    y_train = train_df[Config.LABEL_COLUMN].values
    X_test = test_df[features].values
    
    # Initialize FLAML
    automl = FLAMLAutoML()
    
    # Configure settings
    settings = {
        "time_budget": Config.FLAML_TIME_BUDGET,
        "metric": 'r2',
        "task": 'regression',
        "n_jobs": 2,
        "estimator_list": ['xgboost', 'lgbm', 'rf', 'extra_tree'],
        "seed": Config.RANDOM_STATE
    }
    
    # Train
    automl.fit(X_train, y_train, **settings)
    
    # Get CV score - FLAML uses negative R2, so we need to negate it
    score = -automl.best_score if automl.best_score < 0 else automl.best_score
    print(f"FLAML Best CV Score: {score:.4f}")
    print(f"FLAML Best Estimator: {automl.best_estimator}")
    
    # Test predictions
    test_pred = automl.predict(X_test)
    
    return test_pred, score

# =========================
# Model Training with Time Decay
# =========================
def train_optimized_model(train_df, test_df, features, params, model_type='xgb', use_time_decay=True):
    """Train model with optimized parameters"""
    print(f"\nTraining {model_type.upper()} with optimized parameters...")
    
    n_samples = len(train_df)
    oof_predictions = np.zeros(n_samples)
    test_predictions = np.zeros(len(test_df))
    
    # Time decay weights
    if use_time_decay:
        positions = np.arange(n_samples)
        weights = 0.95 ** (1.0 - positions / (n_samples - 1))
        weights = weights * n_samples / weights.sum()
    else:
        weights = np.ones(n_samples)
    
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), 1):
        print(f"  Fold {fold}/{Config.N_FOLDS}", end=' ')
        
        X_train = train_df.iloc[train_idx][features].values
        y_train = train_df.iloc[train_idx][Config.LABEL_COLUMN].values
        X_valid = train_df.iloc[valid_idx][features].values
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN].values
        w_train = weights[train_idx]
        
        # Train model
        if model_type == 'xgb':
            model = XGBRegressor(**params)
            model.fit(X_train, y_train, sample_weight=w_train,
                     eval_set=[(X_valid, y_valid)], verbose=False)
        else:  # lgb
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train, sample_weight=w_train,
                     eval_set=[(X_valid, y_valid)])
        
        # Predictions
        oof_predictions[valid_idx] = model.predict(X_valid)
        test_predictions += model.predict(test_df[features].values) / Config.N_FOLDS
        
        # Score
        fold_score = pearsonr(y_valid, oof_predictions[valid_idx])[0]
        fold_scores.append(fold_score)
        print(f"Score: {fold_score:.4f}")
        
        # Clean up
        del model, X_train, y_train, X_valid, y_valid
        clean_memory()
    
    # Overall score
    cv_score = pearsonr(train_df[Config.LABEL_COLUMN].values, oof_predictions)[0]
    print(f"  CV Score: {cv_score:.4f} (std: {np.std(fold_scores):.4f})")
    
    return test_predictions, cv_score

# =========================
# Main Pipeline
# =========================
def main():
    """Main execution pipeline"""
    print("=" * 80)
    print("DRW CRYPTO - OPTIMIZED SOLUTION WITH AUTOML")
    print(f"Started at: {datetime.now()}")
    print("=" * 80)
    
    # Show available libraries
    print("\nAvailable optimization libraries:")
    print(f"  - Optuna: {'Yes' if OPTUNA_AVAILABLE else 'No'}")
    print(f"  - LightAutoML: {'Yes' if LIGHTAUTOML_AVAILABLE else 'No'}")
    print(f"  - FLAML: {'Yes' if FLAML_AVAILABLE else 'No'}")
    
    # Initial memory status
    print_memory_usage()
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Optimize memory immediately
    print("\n2. Optimizing memory...")
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    print_memory_usage()
    
    # Initialize components
    feature_engineer = MemoryEfficientFeatureEngineer()
    feature_selector = OptimizedFeatureSelector()
    
    # Feature selection
    print("\n3. Feature engineering...")
    
    # Select additional features using progressive selection
    additional_features = feature_selector.progressive_selection(
        train_df, Config.CORE_FEATURES, Config.MAX_ADDITIONAL_FEATURES
    )
    
    # Prepare features efficiently
    print("\nPreparing final feature set...")
    
    # Base features
    base_features = [f for f in Config.CORE_FEATURES + Config.MARKET_FEATURES if f in train_df.columns]
    
    # Create engineered features
    train_eng = feature_engineer.create_essential_features(train_df)
    test_eng = feature_engineer.create_essential_features(test_df)
    eng_feature_names = list(train_eng.columns)
    
    # Create rank features for dual representation (in batches)
    dual_features = [f for f in Config.DUAL_FEATURES if f in train_df.columns]
    train_dual_rank = feature_engineer.create_rank_features_batch(train_df, dual_features)
    test_dual_rank = feature_engineer.create_rank_features_batch(test_df, dual_features)
    
    # Add selected additional features as ranks
    if additional_features:
        train_add_rank = feature_engineer.create_rank_features_batch(train_df, additional_features[:20])
        test_add_rank = feature_engineer.create_rank_features_batch(test_df, additional_features[:20])
    else:
        train_add_rank = pd.DataFrame(index=train_df.index)
        test_add_rank = pd.DataFrame(index=test_df.index)
    
    # Create minimal interactions
    train_interactions = feature_engineer.create_top_interactions(train_df, Config.CORE_FEATURES, n_interactions=5)
    test_interactions = feature_engineer.create_top_interactions(test_df, Config.CORE_FEATURES, n_interactions=5)
    
    # Combine all features
    train_final = pd.concat([
        train_df[base_features],
        train_eng,
        train_dual_rank,
        train_add_rank,
        train_interactions
    ], axis=1)
    train_final[Config.LABEL_COLUMN] = train_df[Config.LABEL_COLUMN]
    
    test_final = pd.concat([
        test_df[base_features],
        test_eng,
        test_dual_rank,
        test_add_rank,
        test_interactions
    ], axis=1)
    
    # Clean up intermediate dataframes
    del train_eng, test_eng, train_dual_rank, test_dual_rank, train_add_rank, test_add_rank, train_interactions, test_interactions
    clean_memory()
    
    # Get feature columns
    feature_cols = [col for col in train_final.columns if col != Config.LABEL_COLUMN]
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"  - Base features: {len(base_features)}")
    print(f"  - Engineered features: {len(eng_feature_names)}")
    print(f"  - Dual representation: {len(dual_features)}")
    print(f"  - Additional features: {min(20, len(additional_features))}")
    print_memory_usage()
    
    # Store all predictions
    all_predictions = {}
    all_scores = {}
    
    # 4. Traditional models with hyperparameter tuning
    print("\n4. Traditional model training...")
    
    # Tune XGBoost
    xgb_tuner = ModelTuner(model_type='xgb')
    xgb_params = xgb_tuner.tune(train_final, feature_cols, n_trials=Config.N_TUNING_TRIALS)
    clean_memory()
    
    # Train XGBoost
    xgb_pred, xgb_score = train_optimized_model(
        train_final, test_final, feature_cols, xgb_params, 
        model_type='xgb', use_time_decay=True
    )
    all_predictions['xgb'] = xgb_pred
    all_scores['xgb'] = xgb_score
    clean_memory()
    
    # Tune LightGBM
    lgb_tuner = ModelTuner(model_type='lgb')
    lgb_params = lgb_tuner.tune(train_final, feature_cols, n_trials=Config.N_TUNING_TRIALS)
    clean_memory()
    
    # Train LightGBM
    lgb_pred, lgb_score = train_optimized_model(
        train_final, test_final, feature_cols, lgb_params,
        model_type='lgb', use_time_decay=True
    )
    all_predictions['lgb'] = lgb_pred
    all_scores['lgb'] = lgb_score
    clean_memory()
    
    # 5. AutoML models (if available and enabled)
    if Config.USE_AUTOML:
        print("\n5. AutoML model training...")
        
        # LightAutoML
        try:
            lama_pred, lama_score = train_with_lightautoml(train_final, test_final, feature_cols)
            if lama_pred is not None:
                all_predictions['lightautoml'] = lama_pred
                all_scores['lightautoml'] = lama_score
                clean_memory()
        except Exception as e:
            print(f"LightAutoML failed: {e}")
        
        # FLAML
        try:
            flaml_pred, flaml_score = train_with_flaml(train_final, test_final, feature_cols)
            if flaml_pred is not None:
                all_predictions['flaml'] = flaml_pred
                all_scores['flaml'] = flaml_score
                clean_memory()
        except Exception as e:
            print(f"FLAML failed: {e}")
    
    # 6. Create ensemble
    print("\n6. Creating ensemble...")
    
    # Calculate weights based on scores
    total_score = sum(all_scores.values())
    weights = {name: score/total_score for name, score in all_scores.items()}
    
    # Create weighted ensemble
    ensemble_pred = np.zeros_like(list(all_predictions.values())[0])
    for name, pred in all_predictions.items():
        ensemble_pred += weights[name] * pred
    
    print("\nEnsemble weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f} (score: {all_scores[name]:.4f})")
    
    # 7. Save predictions
    print("\n7. Saving predictions...")
    
    # Save individual predictions
    for name, pred in all_predictions.items():
        submission = submission_df.copy()
        submission['prediction'] = pred
        submission.to_csv(f'submission_{name}.csv', index=False)
        print(f"Saved: submission_{name}.csv")
    
    # Save ensemble
    submission_ensemble = submission_df.copy()
    submission_ensemble['prediction'] = ensemble_pred
    submission_ensemble.to_csv('submission_ensemble.csv', index=False)
    print("Saved: submission_ensemble.csv")
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Features used: {len(feature_cols)}")
    print(f"\nModel scores:")
    for name, score in all_scores.items():
        print(f"  {name}: {score:.4f}")
    print(f"\nExpected ensemble performance: >{max(all_scores.values()):.4f}")
    
    if OPTUNA_AVAILABLE and Config.N_TUNING_TRIALS > 0:
        print(f"\nBest XGBoost params: {xgb_params}")
        print(f"\nBest LightGBM params: {lgb_params}")
    
    print(f"\nCompleted at: {datetime.now()}")
    print_memory_usage()
    
    return ensemble_pred, feature_cols

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    predictions, features = main()

