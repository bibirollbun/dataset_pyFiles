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
# -*- coding: utf-8 -*-

"""
DRW Crypto Market Prediction - Extended Runtime Pipeline with Comprehensive Checkpointing
This implementation extends runtime significantly and provides full pipeline state persistence
allowing resumption from any stage of execution.
"""

# Install required packages
!pip install flaml -q
!pip install lightgbm -q
!pip install xgboost -q
!pip install catboost -q
!pip install psutil -q

import pandas as pd
import numpy as np
from flaml import AutoML
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import warnings
import json
from datetime import datetime, timedelta
import gc
import itertools
import os
import psutil
import pickle
import time
import hashlib

warnings.filterwarnings('ignore')

class CFG:
    """Configuration settings for extended runtime pipeline"""
    # File paths
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Extended checkpoint paths
    checkpoint_dir = "./checkpoints"
    checkpoint_version = "v2"
    checkpoint_manifest = "checkpoint_manifest.json"
    checkpoint_features = "checkpoint_features.pkl"
    checkpoint_history = "checkpoint_history.json"
    checkpoint_data = "checkpoint_data.pkl"
    checkpoint_model = "checkpoint_model.pkl"
    checkpoint_pipeline = "checkpoint_pipeline.json"
    checkpoint_results = "checkpoint_results.pkl"
    
    # Model parameters
    n_folds = 10  # Increased from 5
    random_seed = 42
    
    # Significantly extended time budgets
    feature_evaluation_time = 1000  # 10 minutes per feature evaluation (was 300)
    final_model_time = 14400*.5*.5  # 4 hours for final model (was 3600)
    max_runtime_hours = 11.5  # Maximum total runtime in hours
    
    # Expanded feature engineering parameters
    new_feature_depth = 7  # Increased from 6
    feature_width = 60  # Increased from 40
    min_improvement = 0.00001  # Reduced from 0.00005
    max_base_features = 150  # Increased from 100
    
    # Enhanced transformations
    transformations = ['log', 'sqrt', 'square', 'reciprocal', 'exp', 'abs', 'tanh', 'sigmoid']
    transformation_weight = 0.35
    
    # Additional operations for combinations
    advanced_operations = ['max', 'min', 'mean', 'std', 'weighted_mean']
    
    # Performance optimization
    parallel_evaluation = True
    memory_optimization_interval = 50  # Run memory optimization every N features
    
    # Checkpoint intervals
    checkpoint_interval_minutes = 30  # Save checkpoint every 30 minutes
    feature_checkpoint_interval = 1  # Save after every 5 features added

class PipelineState:
    """Manages pipeline execution state"""
    def __init__(self):
        self.start_time = None
        self.stage = None
        self.substage = None
        self.progress = {}
        self.results = {}
        self.errors = []
        self.runtime_stats = {}
        
    def to_dict(self):
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stage': self.stage,
            'substage': self.substage,
            'progress': self.progress,
            'results': self.results,
            'errors': self.errors,
            'runtime_stats': self.runtime_stats
        }
    
    @classmethod
    def from_dict(cls, data):
        state = cls()
        state.start_time = datetime.fromisoformat(data['start_time']) if data.get('start_time') else None
        state.stage = data.get('stage')
        state.substage = data.get('substage')
        state.progress = data.get('progress', {})
        state.results = data.get('results', {})
        state.errors = data.get('errors', [])
        state.runtime_stats = data.get('runtime_stats', {})
        return state

def ensure_checkpoint_dir():
    """Ensure checkpoint directory exists"""
    if not os.path.exists(CFG.checkpoint_dir):
        os.makedirs(CFG.checkpoint_dir)
        print(f"Created checkpoint directory: {CFG.checkpoint_dir}")

def get_checkpoint_path(filename):
    """Get full checkpoint path with version"""
    return os.path.join(CFG.checkpoint_dir, f"{CFG.checkpoint_version}_{filename}")

def save_checkpoint(checkpoint_type, data, use_versioning=True):
    """Save checkpoint data with optional versioning"""
    ensure_checkpoint_dir()
    
    if use_versioning:
        filepath = get_checkpoint_path(checkpoint_type)
    else:
        filepath = os.path.join(CFG.checkpoint_dir, checkpoint_type)
    
    try:
        if checkpoint_type.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        print(f"Checkpoint saved: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving checkpoint {checkpoint_type}: {e}")
        return False

def load_checkpoint(checkpoint_type, use_versioning=True):
    """Load checkpoint data if exists"""
    if use_versioning:
        filepath = get_checkpoint_path(checkpoint_type)
    else:
        filepath = os.path.join(CFG.checkpoint_dir, checkpoint_type)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        if checkpoint_type.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        print(f"Checkpoint loaded: {filepath}")
        return data
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_type}: {e}")
        return None

def save_pipeline_state(state):
    """Save complete pipeline state"""
    manifest = {
        'version': CFG.checkpoint_version,
        'timestamp': datetime.now().isoformat(),
        'state': state.to_dict(),
        'config': {
            'feature_depth': CFG.new_feature_depth,
            'feature_width': CFG.feature_width,
            'min_improvement': CFG.min_improvement,
            'max_runtime_hours': CFG.max_runtime_hours
        }
    }
    return save_checkpoint(CFG.checkpoint_manifest, manifest, use_versioning=False)

def load_pipeline_state():
    """Load complete pipeline state"""
    manifest = load_checkpoint(CFG.checkpoint_manifest, use_versioning=False)
    if manifest:
        return PipelineState.from_dict(manifest['state'])
    return PipelineState()

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def check_runtime_limit(start_time):
    """Check if runtime limit has been reached"""
    elapsed = (datetime.now() - start_time).total_seconds() / 3600
    remaining = CFG.max_runtime_hours - elapsed
    
    if remaining <= 0:
        print(f"\nRuntime limit reached ({CFG.max_runtime_hours} hours)")
        return True, 0
    
    print(f"Runtime: {elapsed:.1f}/{CFG.max_runtime_hours} hours ({remaining:.1f} hours remaining)")
    return False, remaining

def safe_divide(numerator, denominator, fill_value=0):
    """Safe division handling zero denominators"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            np.abs(denominator) > 1e-10,
            numerator / denominator,
            fill_value
        )
    return np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)

def reduce_memory_usage(df, name):
    """Optimize dataframe memory usage"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of {name}: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    gc.collect()
    return df

def create_base_features(df):
    """Create market microstructure features with additional complexity"""
    features_created = []
    
    if 'bid_qty' in df.columns and 'ask_qty' in df.columns:
        df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
        df['bid_ask_imbalance'] = safe_divide(
            df['bid_qty'] - df['ask_qty'],
            df['bid_qty'] + df['ask_qty']
        )
        df['total_depth'] = df['bid_qty'] + df['ask_qty']
        df['bid_ratio'] = safe_divide(df['bid_qty'], df['total_depth'])
        df['depth_imbalance_squared'] = df['bid_ask_imbalance'] ** 2
        features_created.extend(['bid_ask_spread', 'bid_ask_imbalance', 'total_depth', 'bid_ratio', 'depth_imbalance_squared'])
    
    if 'buy_qty' in df.columns and 'sell_qty' in df.columns:
        df['order_imbalance'] = safe_divide(
            df['buy_qty'] - df['sell_qty'],
            df['buy_qty'] + df['sell_qty']
        )
        df['order_flow'] = df['buy_qty'] - df['sell_qty']
        df['trade_intensity'] = df['buy_qty'] + df['sell_qty']
        df['buy_ratio'] = safe_divide(df['buy_qty'], df['trade_intensity'])
        df['order_flow_squared'] = df['order_flow'] ** 2
        df['trade_intensity_log'] = np.log1p(df['trade_intensity'])
        features_created.extend(['order_imbalance', 'order_flow', 'trade_intensity', 'buy_ratio', 'order_flow_squared', 'trade_intensity_log'])
    
    if 'volume' in df.columns:
        df['log_volume'] = np.log1p(df['volume'])
        df['sqrt_volume'] = np.sqrt(df['volume'])
        df['volume_squared'] = df['volume'] ** 2
        features_created.extend(['log_volume', 'sqrt_volume', 'volume_squared'])
    
    # Cross features between market data
    if all(col in df.columns for col in ['bid_qty', 'buy_qty']):
        df['bid_buy_ratio'] = safe_divide(df['bid_qty'], df['buy_qty'])
        df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
        features_created.extend(['bid_buy_ratio', 'bid_buy_interaction'])
    
    if all(col in df.columns for col in ['ask_qty', 'sell_qty']):
        df['ask_sell_ratio'] = safe_divide(df['ask_qty'], df['sell_qty'])
        df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
        features_created.extend(['ask_sell_ratio', 'ask_sell_interaction'])
    
    # Complex microstructure features
    if all(col in df.columns for col in ['bid_qty', 'ask_qty', 'volume']):
        df['volume_to_depth_ratio'] = safe_divide(df['volume'], df['total_depth'])
        df['spread_volume_interaction'] = df['bid_ask_spread'] * df['volume']
        features_created.extend(['volume_to_depth_ratio', 'spread_volume_interaction'])
    
    return features_created

def clean_features(df, features):
    """Clean feature data with enhanced handling"""
    df_clean = df[features].copy()
    
    # Replace infinities
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values with median
    for col in features:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                median_val = 0
            df_clean[col] = df_clean[col].fillna(median_val)
            
            # Clip extreme values
            percentile_99 = df_clean[col].quantile(0.99)
            percentile_1 = df_clean[col].quantile(0.01)
            df_clean[col] = df_clean[col].clip(lower=percentile_1, upper=percentile_99)
    
    return df_clean

def calculate_feature_importance(train_data, features):
    """Calculate feature importance using enhanced LightGBM"""
    print("\nCalculating feature importance...")
    
    # Sample data for speed
    sample_size = min(100000, len(train_data))
    sample_data = train_data.sample(n=sample_size, random_state=CFG.random_seed)
    
    X = clean_features(sample_data, features)
    y = sample_data['label']
    
    # Enhanced LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': CFG.random_seed,
        'n_estimators': 200,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X, y)
    
    # Get importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Top 15 important features:")
    for idx, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")
    
    return importance_df

def apply_transformation(series, transformation):
    """Apply mathematical transformation with extended options"""
    if transformation == 'log':
        return np.log1p(series - series.min() + 1)
    elif transformation == 'sqrt':
        return np.sqrt(series - series.min())
    elif transformation == 'square':
        return series ** 2
    elif transformation == 'reciprocal':
        return safe_divide(1, series)
    elif transformation == 'exp':
        clipped = np.clip(series, -10, 10)
        return np.exp(clipped)
    elif transformation == 'abs':
        return np.abs(series)
    elif transformation == 'tanh':
        return np.tanh(series)
    elif transformation == 'sigmoid':
        return 1 / (1 + np.exp(-np.clip(series, -10, 10)))
    return series

def evaluate_features_parallel(train_data, val_data, features_list):
    """Evaluate multiple feature sets in parallel manner"""
    results = []
    
    for features in features_list:
        try:
            score = evaluate_features(train_data, val_data, features)
            results.append(score)
        except Exception as e:
            print(f"Error evaluating features: {e}")
            results.append(-1)
    
    return results

def evaluate_features(train_data, val_data, features):
    """Evaluate feature set performance with enhanced model"""
    # Prepare data
    X_train = clean_features(train_data, features)
    y_train = train_data['label']
    X_val = clean_features(val_data, features)
    y_val = val_data['label']
    
    # Enhanced model parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'verbose': -1,
        'seed': CFG.random_seed,
        'n_estimators': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20)])
    
    # Evaluate
    val_pred = model.predict(X_val)
    score = pearsonr(y_val, val_pred)[0]
    
    return score

def generate_feature_candidates(features, importance_dict, width, existing_features, depth):
    """Generate feature candidates with progressive complexity"""
    candidates = []
    
    # Sort by importance
    sorted_features = sorted(features, key=lambda x: importance_dict.get(x, 0), reverse=True)
    
    # Adjust selection based on depth
    top_n = min(50 + depth * 5, len(sorted_features))
    top_features = sorted_features[:top_n]
    
    # Transformations
    n_transformations = int(width * CFG.transformation_weight)
    transform_features = top_features[:min(25 + depth * 2, len(top_features))]
    
    for i, feat in enumerate(transform_features):
        for transform in CFG.transformations:
            if len(candidates) >= n_transformations:
                break
            feature_name = f"{feat}_{transform}"
            if feature_name not in existing_features:
                candidates.append({
                    'type': 'transformation',
                    'feature': feat,
                    'transformation': transform
                })
    
    # Binary combinations
    operations = ['multiply', 'divide', 'add', 'subtract']
    combination_features = top_features[:min(30 + depth * 2, len(top_features))]
    
    for feat1, feat2 in itertools.combinations(combination_features, 2):
        if len(candidates) >= width * 0.7:
            break
        for op in operations:
            if len(candidates) >= width * 0.7:
                break
            feature_name = f"{feat1}_{op[:3]}_{feat2}"
            if feature_name not in existing_features:
                candidates.append({
                    'type': 'combination',
                    'feat1': feat1,
                    'feat2': feat2,
                    'operation': op
                })
    
    # Advanced combinations
    if len(candidates) < width:
        for op in CFG.advanced_operations:
            for feat1, feat2 in itertools.combinations(top_features[:20], 2):
                if len(candidates) >= width:
                    break
                feature_name = f"{feat1}_{op}_{feat2}"
                if feature_name not in existing_features:
                    candidates.append({
                        'type': 'advanced',
                        'feat1': feat1,
                        'feat2': feat2,
                        'operation': op
                    })
    
    # Three-way interactions for later depths
    if depth >= 5 and len(candidates) < width:
        for feat1, feat2, feat3 in itertools.combinations(top_features[:10], 3):
            if len(candidates) >= width:
                break
            feature_name = f"{feat1}_x_{feat2}_x_{feat3}"
            if feature_name not in existing_features:
                candidates.append({
                    'type': 'triple',
                    'feat1': feat1,
                    'feat2': feat2,
                    'feat3': feat3
                })
    
    return candidates[:width]

def create_new_feature(df, candidate):
    """Create feature from candidate with extended operations"""
    if candidate['type'] == 'transformation':
        feature_name = f"{candidate['feature']}_{candidate['transformation']}"
        values = apply_transformation(df[candidate['feature']], candidate['transformation'])
        
    elif candidate['type'] == 'combination':
        feat1 = candidate['feat1']
        feat2 = candidate['feat2']
        op = candidate['operation']
        
        if op == 'multiply':
            feature_name = f"{feat1}_mul_{feat2}"
            values = df[feat1] * df[feat2]
        elif op == 'divide':
            feature_name = f"{feat1}_div_{feat2}"
            values = safe_divide(df[feat1], df[feat2])
        elif op == 'add':
            feature_name = f"{feat1}_add_{feat2}"
            values = df[feat1] + df[feat2]
        else:  # subtract
            feature_name = f"{feat1}_sub_{feat2}"
            values = df[feat1] - df[feat2]
    
    elif candidate['type'] == 'advanced':
        feat1 = candidate['feat1']
        feat2 = candidate['feat2']
        op = candidate['operation']
        
        if op == 'max':
            feature_name = f"{feat1}_max_{feat2}"
            values = np.maximum(df[feat1], df[feat2])
        elif op == 'min':
            feature_name = f"{feat1}_min_{feat2}"
            values = np.minimum(df[feat1], df[feat2])
        elif op == 'mean':
            feature_name = f"{feat1}_mean_{feat2}"
            values = (df[feat1] + df[feat2]) / 2
        elif op == 'std':
            feature_name = f"{feat1}_std_{feat2}"
            values = np.sqrt(((df[feat1] - df[feat1].mean())**2 + (df[feat2] - df[feat2].mean())**2) / 2)
        else:  # weighted_mean
            feature_name = f"{feat1}_wmean_{feat2}"
            weight1 = np.abs(df[feat1]).sum()
            weight2 = np.abs(df[feat2]).sum()
            total_weight = weight1 + weight2
            if total_weight > 0:
                values = (df[feat1] * weight1 + df[feat2] * weight2) / total_weight
            else:
                values = (df[feat1] + df[feat2]) / 2
    
    elif candidate['type'] == 'triple':
        feat1 = candidate['feat1']
        feat2 = candidate['feat2']
        feat3 = candidate['feat3']
        feature_name = f"{feat1}_x_{feat2}_x_{feat3}"
        values = df[feat1] * df[feat2] * df[feat3]
    
    return feature_name, values

def greedy_feature_engineering(train_data, initial_features, importance_df, pipeline_state):
    """Perform greedy feature engineering with comprehensive checkpointing"""
    print("\nStarting extended feature engineering...")
    
    # Check for existing checkpoint
    checkpoint_state = load_checkpoint(CFG.checkpoint_features)
    checkpoint_history = load_checkpoint(CFG.checkpoint_history)
    
    if checkpoint_state and checkpoint_history:
        print(f"Resuming from checkpoint with {len(checkpoint_state['features'])} features")
        current_features = checkpoint_state['features']
        best_score = checkpoint_state['best_score']
        feature_history = checkpoint_history
        start_depth = checkpoint_state.get('last_depth', 0) + 1
        last_checkpoint_time = datetime.fromisoformat(checkpoint_state.get('last_checkpoint_time', datetime.now().isoformat()))
        
        # Recreate engineered features
        print("Recreating engineered features...")
        for i, feat_info in enumerate(feature_history):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(feature_history)} features recreated")
            try:
                candidate = feat_info['details']
                feature_name, train_values = create_new_feature(train_data, candidate)
                if feature_name not in train_data.columns:
                    train_data[feature_name] = train_values
            except Exception as e:
                print(f"Error recreating feature {feat_info.get('feature', 'unknown')}: {e}")
    else:
        current_features = initial_features.copy()
        feature_history = []
        start_depth = 0
        best_score = None
        last_checkpoint_time = datetime.now()
    
    # Create validation split
    val_size = int(len(train_data) * 0.2)
    val_indices = np.random.choice(len(train_data), size=val_size, replace=False)
    val_data = train_data.iloc[val_indices].copy()
    train_fe = train_data.drop(train_data.index[val_indices]).copy()
    
    # Calculate baseline if needed
    if best_score is None:
        best_score = evaluate_features(train_fe, val_data, current_features)
        print(f"Baseline score: {best_score:.6f}")
    
    # Get importance dict
    importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
    
    # Track all created features to avoid duplicates
    all_created_features = set(current_features)
    
    # Feature engineering loop
    features_added_since_checkpoint = 0
    
    for depth in range(start_depth, CFG.new_feature_depth):
        # Check runtime limit
        exceeded, remaining_hours = check_runtime_limit(pipeline_state.start_time)
        if exceeded:
            print("Runtime limit exceeded, saving progress and exiting...")
            break
        
        print(f"\nDepth {depth + 1}/{CFG.new_feature_depth}")
        print(f"Current features: {len(current_features)}")
        print(f"Memory usage: {get_memory_usage():.2f} GB")
        
        # Update pipeline state
        pipeline_state.substage = f"feature_engineering_depth_{depth + 1}"
        pipeline_state.progress['current_depth'] = depth + 1
        pipeline_state.progress['features_count'] = len(current_features)
        
        # Generate candidates with depth-aware strategy
        candidates = generate_feature_candidates(
            current_features, importance_dict, CFG.feature_width, all_created_features, depth
        )
        print(f"Evaluating {len(candidates)} candidates...")
        
        best_candidate = None
        best_new_score = best_score
        improvement_found = False
        candidates_evaluated = 0
        
        # Batch evaluation for efficiency
        batch_size = 10
        
        for batch_start in range(0, len(candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]
            
            if batch_start % 20 == 0:
                print(f"  Progress: {batch_start}/{len(candidates)} candidates evaluated")
            
            for candidate in batch_candidates:
                try:
                    # Create feature
                    feature_name, train_values = create_new_feature(train_fe, candidate)
                    
                    if feature_name in train_fe.columns:
                        continue
                    
                    # Add temporarily
                    train_fe[feature_name] = train_values
                    val_data[feature_name] = create_new_feature(val_data, candidate)[1]
                    
                    # Evaluate
                    temp_features = current_features + [feature_name]
                    score = evaluate_features(train_fe, val_data, temp_features)
                    
                    if score > best_new_score:
                        best_candidate = candidate
                        best_candidate['name'] = feature_name
                        best_new_score = score
                        improvement_found = True
                    
                    # Remove
                    train_fe.drop(columns=[feature_name], inplace=True)
                    val_data.drop(columns=[feature_name], inplace=True)
                    
                    candidates_evaluated += 1
                    
                except Exception as e:
                    print(f"  Error evaluating candidate: {e}")
                    continue
            
            # Periodic memory optimization
            if candidates_evaluated % CFG.memory_optimization_interval == 0:
                gc.collect()
        
        # Add best feature
        if best_candidate and (best_new_score - best_score) > CFG.min_improvement:
            feature_name = best_candidate['name']
            train_fe[feature_name] = create_new_feature(train_fe, best_candidate)[1]
            val_data[feature_name] = create_new_feature(val_data, best_candidate)[1]
            
            current_features.append(feature_name)
            all_created_features.add(feature_name)
            improvement = best_new_score - best_score
            best_score = best_new_score
            
            print(f"Added {feature_name}: {best_score:.6f} (+{improvement:.6f})")
            
            feature_history.append({
                'feature': feature_name,
                'type': best_candidate['type'],
                'details': best_candidate,
                'score': best_score,
                'improvement': improvement,
                'depth': depth + 1,
                'timestamp': datetime.now().isoformat()
            })
            
            features_added_since_checkpoint += 1
            
            # Save checkpoint after interval or time threshold
            time_since_checkpoint = (datetime.now() - last_checkpoint_time).total_seconds() / 60
            
            if (features_added_since_checkpoint >= CFG.feature_checkpoint_interval or 
                time_since_checkpoint >= CFG.checkpoint_interval_minutes):
                
                checkpoint_state = {
                    'features': current_features,
                    'best_score': best_score,
                    'last_depth': depth,
                    'last_checkpoint_time': datetime.now().isoformat(),
                    'total_features_engineered': len(feature_history)
                }
                save_checkpoint(CFG.checkpoint_features, checkpoint_state)
                save_checkpoint(CFG.checkpoint_history, feature_history)
                save_pipeline_state(pipeline_state)
                
                features_added_since_checkpoint = 0
                last_checkpoint_time = datetime.now()
                print(f"Checkpoint saved at depth {depth + 1}")
            
        else:
            print("No improvement found at this depth")
            if not improvement_found:
                print("Early stopping - no candidates showed improvement")
                # Save final state before stopping
                checkpoint_state = {
                    'features': current_features,
                    'best_score': best_score,
                    'last_depth': depth,
                    'last_checkpoint_time': datetime.now().isoformat(),
                    'total_features_engineered': len(feature_history)
                }
                save_checkpoint(CFG.checkpoint_features, checkpoint_state)
                save_checkpoint(CFG.checkpoint_history, feature_history)
                break
    
    return current_features, feature_history

def train_final_model(train_data, test_data, features, pipeline_state):
    """Train final model with extended time budget and checkpointing"""
    print("\nTraining final model with extended parameters...")
    print(f"Time budget: {CFG.final_model_time/3600:.1f} hours")
    
    # Check for existing model checkpoint
    model_checkpoint = load_checkpoint(CFG.checkpoint_model)
    if model_checkpoint:
        print("Found existing model checkpoint")
        test_predictions = model_checkpoint.get('predictions')
        cv_scores = model_checkpoint.get('cv_scores', [])
        if test_predictions is not None and len(cv_scores) > 0:
            print(f"Using saved predictions with CV score: {np.mean(cv_scores):.6f}")
            return test_predictions, cv_scores
    
    # Prepare data
    X_train = clean_features(train_data, features)
    y_train = train_data['label']
    X_test = clean_features(test_data, features)
    
    # Update pipeline state
    pipeline_state.stage = "model_training"
    
    # Train with FLAML
    try:
        automl = AutoML()
        automl_settings = {
            "time_budget": CFG.final_model_time,
            "metric": 'r2',
            "estimator_list": ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree'],  # Added more estimators
            "task": "regression",
            "seed": CFG.random_seed,
            "verbose": 2,  # Increased verbosity
            "eval_method": "cv",
            "n_splits": 5,
            "early_stop": True,
            "retrain_full": True,
            "ensemble": True,  # Enable ensemble
            "max_iter": 10000,  # Increased iterations
            "mem_thres": 128 * 1024 * 1024 * 1024  # 128GB memory threshold
        }
        
        print("Starting extended AutoML training...")
        automl.fit(X_train, y_train, **automl_settings)
        test_predictions = automl.predict(X_test)
        
        print(f"Best estimator: {automl.best_estimator}")
        print(f"Best config: {automl.best_config}")
        print(f"Best validation score: {automl.best_loss}")
        
    except Exception as e:
        # Enhanced fallback
        print(f"AutoML error: {e}")
        print("Using enhanced LightGBM ensemble fallback...")
        
        # Train multiple models with different parameters
        models = []
        model_configs = [
            {'num_leaves': 127, 'learning_rate': 0.03, 'n_estimators': 2000},
            {'num_leaves': 63, 'learning_rate': 0.05, 'n_estimators': 1500},
            {'num_leaves': 255, 'learning_rate': 0.02, 'n_estimators': 1000},
        ]
        
        for config in model_configs:
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': CFG.random_seed,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                **config
            }
            
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_train, y_train)
            models.append(model)
        
        # Ensemble predictions
        predictions = []
        for model in models:
            predictions.append(model.predict(X_test))
        test_predictions = np.mean(predictions, axis=0)
    
    # Enhanced cross-validation
    kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.random_seed)
    cv_scores = []
    
    print("Calculating detailed cross-validation scores...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Processing fold {fold + 1}/{CFG.n_folds}")
        
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Use best parameters from AutoML if available
        if 'automl' in locals() and hasattr(automl, 'best_config'):
            fold_model = lgb.LGBMRegressor(**automl.best_config)
        else:
            fold_model = lgb.LGBMRegressor(
                n_estimators=500, 
                num_leaves=63,
                learning_rate=0.05,
                random_state=CFG.random_seed, 
                verbose=-1
            )
        
        fold_model.fit(X_fold_train, y_fold_train)
        fold_pred = fold_model.predict(X_fold_val)
        
        fold_score = pearsonr(y_fold_val, fold_pred)[0]
        cv_scores.append(fold_score)
        print(f"    Fold {fold + 1} score: {fold_score:.6f}")
    
    print(f"CV Score: {np.mean(cv_scores):.6f} (±{np.std(cv_scores):.6f})")
    
    # Save model checkpoint
    model_checkpoint = {
        'predictions': test_predictions,
        'cv_scores': cv_scores,
        'timestamp': datetime.now().isoformat(),
        'features_used': len(features)
    }
    save_checkpoint(CFG.checkpoint_model, model_checkpoint)
    
    return test_predictions, cv_scores

def main():
    """Main execution with comprehensive checkpoint support"""
    # Load or create pipeline state
    pipeline_state = load_pipeline_state()
    
    if pipeline_state.start_time is None:
        pipeline_state.start_time = datetime.now()
        timestamp = pipeline_state.start_time.strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = pipeline_state.start_time.strftime("%Y%m%d_%H%M%S")
        print(f"Resuming pipeline from {pipeline_state.stage or 'beginning'}")
    
    print("="*80)
    print("DRW CRYPTO MARKET PREDICTION - EXTENDED RUNTIME PIPELINE")
    print(f"Timestamp: {timestamp}")
    print(f"Maximum runtime: {CFG.max_runtime_hours} hours")
    print("="*80)
    
    # Configuration summary
    print("\nExtended Configuration:")
    print(f"  Feature engineering depth: {CFG.new_feature_depth}")
    print(f"  Candidates per iteration: {CFG.feature_width}")
    print(f"  Minimum improvement threshold: {CFG.min_improvement}")
    print(f"  Feature evaluation time: {CFG.feature_evaluation_time}s")
    print(f"  Final model time budget: {CFG.final_model_time/3600:.1f} hours")
    print(f"  Checkpoint interval: {CFG.checkpoint_interval_minutes} minutes")
    
    # Stage 1: Data Loading
    if pipeline_state.stage in [None, 'data_loading']:
        pipeline_state.stage = 'data_loading'
        print("\nStage 1: Loading data...")
        
        data_checkpoint = load_checkpoint(CFG.checkpoint_data)
        if data_checkpoint:
            print("Loading data from checkpoint...")
            train_data = data_checkpoint['train']
            test_data = data_checkpoint['test']
            sample_submission = data_checkpoint['submission']
        else:
            train_data = pd.read_parquet(CFG.train_path)
            test_data = pd.read_parquet(CFG.test_path)
            sample_submission = pd.read_csv(CFG.sample_sub_path)
            
            # Save data checkpoint
            data_checkpoint = {
                'train': train_data,
                'test': test_data,
                'submission': sample_submission
            }
            save_checkpoint(CFG.checkpoint_data, data_checkpoint)
        
        print(f"Train shape: {train_data.shape}")
        print(f"Test shape: {test_data.shape}")
        
        # Optimize memory
        train_data = reduce_memory_usage(train_data, "train")
        test_data = reduce_memory_usage(test_data, "test")
        
        pipeline_state.results['data_loaded'] = True
        save_pipeline_state(pipeline_state)
    else:
        # Load from checkpoint
        data_checkpoint = load_checkpoint(CFG.checkpoint_data)
        train_data = data_checkpoint['train']
        test_data = data_checkpoint['test']
        sample_submission = data_checkpoint['submission']
    
    # Stage 2: Feature Preparation
    if pipeline_state.stage in [None, 'data_loading', 'feature_preparation']:
        pipeline_state.stage = 'feature_preparation'
        print("\nStage 2: Feature preparation...")
        
        # Get features
        x_features = [col for col in train_data.columns if col.startswith('X')]
        market_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
        
        # Create base features
        base_features = create_base_features(train_data)
        test_base = create_base_features(test_data)
        
        # All features
        all_features = x_features + market_features + base_features
        all_features = [f for f in all_features if f in train_data.columns and f in test_data.columns]
        
        print(f"\nTotal features: {len(all_features)}")
        
        # Feature importance
        importance_df = calculate_feature_importance(train_data, all_features)
        
        # Select top features
        initial_features = importance_df.head(CFG.max_base_features)['feature'].tolist()
        
        pipeline_state.results['initial_features'] = initial_features
        pipeline_state.results['importance_df'] = importance_df
        save_pipeline_state(pipeline_state)
    else:
        initial_features = pipeline_state.results['initial_features']
        importance_df = pipeline_state.results['importance_df']
    
    # Stage 3: Feature Engineering
    if pipeline_state.stage in [None, 'data_loading', 'feature_preparation', 'feature_engineering']:
        pipeline_state.stage = 'feature_engineering'
        
        # Check runtime before starting expensive feature engineering
        exceeded, remaining_hours = check_runtime_limit(pipeline_state.start_time)
        if not exceeded and remaining_hours > 0.5:  # Need at least 30 minutes
            final_features, feature_history = greedy_feature_engineering(
                train_data, initial_features, importance_df, pipeline_state
            )
            
            pipeline_state.results['final_features'] = final_features
            pipeline_state.results['feature_history'] = feature_history
            save_pipeline_state(pipeline_state)
        else:
            print("Insufficient time for feature engineering, using existing features")
            final_features = pipeline_state.results.get('final_features', initial_features)
            feature_history = pipeline_state.results.get('feature_history', [])
    else:
        final_features = pipeline_state.results['final_features']
        feature_history = pipeline_state.results['feature_history']
    
    # Apply engineered features to full dataset
    print("\nApplying engineered features to full dataset...")
    features_applied = 0
    for feat_info in feature_history:
        try:
            candidate = feat_info['details']
            feature_name, train_values = create_new_feature(train_data, candidate)
            _, test_values = create_new_feature(test_data, candidate)
            
            if feature_name not in train_data.columns:
                train_data[feature_name] = train_values
            if feature_name not in test_data.columns:
                test_data[feature_name] = test_values
            
            features_applied += 1
            if features_applied % 20 == 0:
                print(f"  Applied {features_applied}/{len(feature_history)} features")
                
        except Exception as e:
            print(f"Error applying feature {feat_info['feature']}: {e}")
            final_features = [f for f in final_features if f != feat_info['feature']]
    
    # Ensure all features exist
    final_features = [f for f in final_features if f in train_data.columns and f in test_data.columns]
    
    print(f"\nFinal features: {len(final_features)}")
    print(f"Engineered features added: {len(feature_history)}")
    
    # Stage 4: Model Training
    exceeded, remaining_hours = check_runtime_limit(pipeline_state.start_time)
    if not exceeded and remaining_hours > 0.25:  # Need at least 15 minutes
        test_predictions, cv_scores = train_final_model(train_data, test_data, final_features, pipeline_state)
    else:
        print("Insufficient time for model training, using saved predictions if available")
        model_checkpoint = load_checkpoint(CFG.checkpoint_model)
        if model_checkpoint:
            test_predictions = model_checkpoint['predictions']
            cv_scores = model_checkpoint.get('cv_scores', [])
        else:
            print("No saved predictions found, using simple model")
            # Quick fallback model
            X_train = clean_features(train_data, final_features)
            y_train = train_data['label']
            X_test = clean_features(test_data, final_features)
            
            model = lgb.LGBMRegressor(n_estimators=100, random_state=CFG.random_seed)
            model.fit(X_train, y_train)
            test_predictions = model.predict(X_test)
            cv_scores = [0.0]  # Placeholder
    
    # Save predictions
    submission = sample_submission.copy()
    submission['prediction'] = test_predictions
    submission_filename = f"submission_{timestamp}.csv"
    submission.to_csv(submission_filename, index=False)
    
    print(f"\nSaved predictions to {submission_filename}")
    
    # Save comprehensive summary
    runtime_hours = (datetime.now() - pipeline_state.start_time).total_seconds() / 3600
    
    summary = {
        'timestamp': timestamp,
        'runtime_hours': runtime_hours,
        'configuration': {
            'feature_depth': CFG.new_feature_depth,
            'feature_width': CFG.feature_width,
            'min_improvement': CFG.min_improvement,
            'feature_evaluation_time': CFG.feature_evaluation_time,
            'final_model_time': CFG.final_model_time,
            'max_runtime_hours': CFG.max_runtime_hours
        },
        'features': {
            'initial': len(initial_features),
            'engineered': len(feature_history),
            'final': len(final_features)
        },
        'cv_score': {
            'mean': np.mean(cv_scores) if cv_scores else 0,
            'std': np.std(cv_scores) if cv_scores else 0,
            'scores': cv_scores
        },
        'feature_history': feature_history,
        'top_engineered_features': [f['feature'] for f in sorted(
            feature_history, key=lambda x: x.get('improvement', 0), reverse=True
        )[:20]],
        'pipeline_state': pipeline_state.to_dict()
    }
    
    with open(f'summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\nPipeline completed successfully!")
    print(f"Total runtime: {runtime_hours:.2f} hours")
    print(f"Final CV Score: {np.mean(cv_scores):.6f}")

if __name__ == "__main__":
    main()

