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
Enhanced ML Pipeline using FLAML AutoML with Intelligent Meta-Modeling
Combines automated hyperparameter optimization with sophisticated ensemble techniques
"""

# -----------------------------
# Installation Requirements (run first)
# -----------------------------
!pip install flaml[automl] optuna scikit-optimize

# -----------------------------
# Imports
# -----------------------------
import os
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pickle
import joblib

# Core ML
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor

# FLAML AutoML
try:
    from flaml import AutoML
    from flaml.automl.data import get_output_from_log
except ImportError:
    print("Installing FLAML...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'flaml[automl]'])
    from flaml import AutoML
    from flaml.automl.data import get_output_from_log

# Traditional models (as backup)
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

# -----------------------------
# Configuration
# -----------------------------
CONFIG = {
    'seed': 42,
    'n_folds': 5,
    'data_dir': "/kaggle/input/playground-series-s5e9",
    'id_col': "id",
    'target_col': "BeatsPerMinute",
    'base_features': [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ],
    'flaml': {
        'time_budget': 300,  # seconds per model
        'max_iter': 1000,
        'metric': 'rmse',
        'task': 'regression',
        'n_jobs': -1,
        'verbose': 1,
        'ensemble': True,
        'max_mem_in_mb': 8192,
        'early_stop': True,
        'eval_method': 'cv',
        'n_splits': 3,
    },
    'meta_models': {
        'use_neural_net': True,
        'use_stacking': True,
        'use_blending': True,
    }
}

# -----------------------------
# Utilities
# -----------------------------
def rmse(y_true, y_pred):
    """Calculate RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

@dataclass
class ModelResult:
    """Container for model results"""
    oof_pred: np.ndarray
    test_pred: np.ndarray
    cv_score: float
    model_name: str
    feature_importance: Optional[Dict] = None
    model_params: Optional[Dict] = None

# -----------------------------
# Feature Engineering
# -----------------------------
class AdvancedFeatureEngineer:
    """Advanced feature engineering with multiple strategies"""
    
    def __init__(self, config):
        self.config = config
        self.transformers = {}
        
    def fit_transform(self, train_df, test_df):
        """Create comprehensive feature set"""
        train = train_df.copy()
        test = test_df.copy()
        
        base_cols = self.config['base_features']
        new_features = []
        
        # 1. Statistical aggregations
        for col in base_cols:
            # Polynomial features
            train[f'{col}_sq'] = train[col] ** 2
            test[f'{col}_sq'] = test[col] ** 2
            train[f'{col}_sqrt'] = np.sqrt(np.abs(train[col]))
            test[f'{col}_sqrt'] = np.sqrt(np.abs(test[col]))
            new_features.extend([f'{col}_sq', f'{col}_sqrt'])
            
            # Log transformations for positive features
            if train[col].min() > 0:
                train[f'{col}_log'] = np.log1p(train[col])
                test[f'{col}_log'] = np.log1p(test[col])
                new_features.append(f'{col}_log')
        
        # 2. Interaction features (selective to avoid explosion)
        important_interactions = [
            ('Energy', 'MoodScore'),
            ('AudioLoudness', 'Energy'),
            ('VocalContent', 'InstrumentalScore'),
            ('RhythmScore', 'TrackDurationMs'),
            ('AcousticQuality', 'LivePerformanceLikelihood')
        ]
        
        for col1, col2 in important_interactions:
            # Products
            train[f'{col1}_x_{col2}'] = train[col1] * train[col2]
            test[f'{col1}_x_{col2}'] = test[col1] * test[col2]
            
            # Ratios (safe division)
            eps = 1e-8
            train[f'{col1}_div_{col2}'] = train[col1] / (train[col2] + eps)
            test[f'{col1}_div_{col2}'] = test[col1] / (test[col2] + eps)
            
            new_features.extend([f'{col1}_x_{col2}', f'{col1}_div_{col2}'])
        
        # 3. Clustering-based features
        from sklearn.cluster import KMeans
        
        # Create clusters on normalized features
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train[base_cols])
        test_scaled = scaler.transform(test[base_cols])
        
        for n_clusters in [5, 10, 15]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            train[f'cluster_{n_clusters}'] = kmeans.fit_predict(train_scaled)
            test[f'cluster_{n_clusters}'] = kmeans.predict(test_scaled)
            
            # Distance to nearest cluster center
            train_distances = kmeans.transform(train_scaled)
            test_distances = kmeans.transform(test_scaled)
            train[f'min_cluster_dist_{n_clusters}'] = train_distances.min(axis=1)
            test[f'min_cluster_dist_{n_clusters}'] = test_distances.min(axis=1)
            
            new_features.extend([f'cluster_{n_clusters}', f'min_cluster_dist_{n_clusters}'])
        
        # 4. Frequency encoding for binned features
        for col in base_cols:
            # Create bins
            bins = pd.qcut(train[col], q=10, duplicates='drop')
            train[f'{col}_bin'] = bins.cat.codes
            test[f'{col}_bin'] = pd.cut(test[col], bins=bins.cat.categories, labels=False)
            
            # Frequency encoding
            freq_map = train[f'{col}_bin'].value_counts(normalize=True).to_dict()
            train[f'{col}_freq'] = train[f'{col}_bin'].map(freq_map)
            test[f'{col}_freq'] = test[f'{col}_bin'].map(freq_map).fillna(0)
            
            new_features.extend([f'{col}_bin', f'{col}_freq'])
        
        # 5. Target encoding (using cross-validation to avoid leakage)
        if self.config['target_col'] in train.columns:
            target = train[self.config['target_col']]
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for col in base_cols[:3]:  # Limit to avoid overfitting
                train[f'{col}_target_enc'] = 0
                
                for train_idx, val_idx in kf.split(train):
                    # Calculate mean target per category on train fold
                    temp_df = train.iloc[train_idx][[col, self.config['target_col']]]
                    encoding = temp_df.groupby(col)[self.config['target_col']].mean()
                    
                    # Apply to validation fold
                    train.loc[val_idx, f'{col}_target_enc'] = train.iloc[val_idx][col].map(encoding)
                
                # For test, use full training data
                encoding = train.groupby(col)[self.config['target_col']].mean()
                test[f'{col}_target_enc'] = test[col].map(encoding).fillna(target.mean())
                
                new_features.append(f'{col}_target_enc')
        
        # Combine all features
        feature_cols = base_cols + new_features
        
        return train, test, feature_cols

# -----------------------------
# FLAML AutoML Models
# -----------------------------
class FLAMLModelTrainer:
    """Train multiple FLAML models with different configurations"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        
    def train_flaml_model(self, X_train, y_train, X_test, 
                          model_type='auto', time_budget=300, 
                          custom_hp=None):
        """Train a FLAML AutoML model"""
        
        automl = AutoML()
        
        settings = {
            "time_budget": time_budget,
            "metric": self.config['flaml']['metric'],
            "task": self.config['flaml']['task'],
            "n_jobs": self.config['flaml']['n_jobs'],
            "verbose": 0,
            "seed": self.config['seed'],
            "eval_method": "cv",
            "n_splits": 3,
        }
        
        # Customize estimator list based on model_type
        if model_type == 'tree':
            settings["estimator_list"] = ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree']
        elif model_type == 'linear':
            settings["estimator_list"] = ['ridge', 'sgd']
        elif model_type == 'fast':
            settings["estimator_list"] = ['lgbm', 'xgboost']
        # else 'auto' uses all available
        
        if custom_hp:
            settings.update(custom_hp)
        
        # Train
        automl.fit(X_train, y_train, **settings)
        
        # Get predictions
        oof_pred = automl.predict(X_train)
        test_pred = automl.predict(X_test)
        
        return {
            'model': automl,
            'oof_pred': oof_pred,
            'test_pred': test_pred,
            'best_estimator': automl.best_estimator,
            'best_config': automl.best_config,
            'best_loss': automl.best_loss,
        }
    
    def train_diverse_models(self, X, y, X_test):
        """Train multiple diverse FLAML models"""
        results = []
        
        # 1. Standard FLAML with all estimators
        print("Training FLAML Model 1: Full AutoML")
        result1 = self.train_flaml_model(X, y, X_test, 
                                         model_type='auto', 
                                         time_budget=self.config['flaml']['time_budget'])
        results.append(ModelResult(
            oof_pred=result1['oof_pred'],
            test_pred=result1['test_pred'],
            cv_score=result1['best_loss'],
            model_name='FLAML_Full',
            model_params=result1['best_config']
        ))
        
        # 2. Tree-based only
        print("Training FLAML Model 2: Tree-based models")
        result2 = self.train_flaml_model(X, y, X_test, 
                                         model_type='tree', 
                                         time_budget=self.config['flaml']['time_budget']//2)
        results.append(ModelResult(
            oof_pred=result2['oof_pred'],
            test_pred=result2['test_pred'],
            cv_score=result2['best_loss'],
            model_name='FLAML_Trees',
            model_params=result2['best_config']
        ))
        
        # 3. Fast models with ensemble
        print("Training FLAML Model 3: Fast ensemble")
        result3 = self.train_flaml_model(
            X, y, X_test, 
            model_type='fast', 
            time_budget=self.config['flaml']['time_budget']//2,
            custom_hp={'ensemble': True, 'max_iter': 500}
        )
        results.append(ModelResult(
            oof_pred=result3['oof_pred'],
            test_pred=result3['test_pred'],
            cv_score=result3['best_loss'],
            model_name='FLAML_Fast',
            model_params=result3['best_config']
        ))
        
        # 4. FLAML with custom search space
        print("Training FLAML Model 4: Custom hyperparameters")
        custom_hp = {
            'custom_hp': {
                'lgbm': {
                    'n_estimators': {'domain': [100, 2000], 'init_value': 500},
                    'num_leaves': {'domain': [31, 255], 'init_value': 127},
                    'learning_rate': {'domain': [0.01, 0.3], 'init_value': 0.1},
                },
                'xgboost': {
                    'n_estimators': {'domain': [100, 2000], 'init_value': 500},
                    'max_depth': {'domain': [3, 15], 'init_value': 8},
                    'learning_rate': {'domain': [0.01, 0.3], 'init_value': 0.1},
                }
            }
        }
        result4 = self.train_flaml_model(X, y, X_test, 
                                         model_type='auto', 
                                         time_budget=self.config['flaml']['time_budget'],
                                         custom_hp=custom_hp)
        results.append(ModelResult(
            oof_pred=result4['oof_pred'],
            test_pred=result4['test_pred'],
            cv_score=result4['best_loss'],
            model_name='FLAML_Custom',
            model_params=result4['best_config']
        ))
        
        return results

# -----------------------------
# Intelligent Meta-Modeling
# -----------------------------
class IntelligentMetaModel:
    """Advanced meta-modeling techniques"""
    
    def __init__(self, config):
        self.config = config
        self.meta_models = {}
        
    def create_meta_features(self, predictions_list):
        """Create sophisticated meta-features from base predictions"""
        preds = np.column_stack(predictions_list)
        meta_features = []
        
        # Basic predictions
        meta_features.append(preds)
        
        # Statistical aggregations
        meta_features.append(np.mean(preds, axis=1, keepdims=True))
        meta_features.append(np.median(preds, axis=1, keepdims=True))
        meta_features.append(np.std(preds, axis=1, keepdims=True))
        meta_features.append(np.min(preds, axis=1, keepdims=True))
        meta_features.append(np.max(preds, axis=1, keepdims=True))
        
        # Pairwise differences (for diversity measurement)
        n_models = preds.shape[1]
        for i in range(n_models):
            for j in range(i+1, n_models):
                diff = np.abs(preds[:, i] - preds[:, j]).reshape(-1, 1)
                meta_features.append(diff)
        
        # Rank features
        ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, preds)
        meta_features.append(ranks)
        
        # Variance-based features
        high_variance_mask = (np.std(preds, axis=1) > np.percentile(np.std(preds, axis=1), 75)).astype(float)
        meta_features.append(high_variance_mask.reshape(-1, 1))
        
        return np.hstack(meta_features)
    
    def train_neural_meta_model(self, X_meta_train, y_train, X_meta_test):
        """Train a neural network meta-model"""
        
        # Normalize features
        scaler = StandardScaler()
        X_meta_scaled = scaler.fit_transform(X_meta_train)
        X_test_scaled = scaler.transform(X_meta_test)
        
        # Define neural network
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=512,
            learning_rate='adaptive',
            learning_rate_init=0.01,
            max_iter=1000,
            shuffle=True,
            random_state=self.config['seed'],
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            verbose=False
        )
        
        # Train
        nn_model.fit(X_meta_scaled, y_train)
        
        # Predictions
        nn_oof = nn_model.predict(X_meta_scaled)
        nn_test = nn_model.predict(X_test_scaled)
        
        self.meta_models['neural_net'] = {
            'model': nn_model,
            'scaler': scaler
        }
        
        return nn_oof, nn_test
    
    def train_stacking_meta_model(self, base_predictions_train, y_train, base_predictions_test):
        """Train multiple stacking models and blend them"""
        
        stacking_models = {
            'ridge': Ridge(alpha=1.0, random_state=self.config['seed']),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.config['seed']),
        }
        
        stack_oof = {}
        stack_test = {}
        
        for name, model in stacking_models.items():
            # K-fold for stacking
            kf = KFold(n_splits=5, shuffle=True, random_state=self.config['seed'])
            oof = np.zeros(len(base_predictions_train))
            test_preds = []
            
            for train_idx, val_idx in kf.split(base_predictions_train):
                X_train_fold = base_predictions_train[train_idx]
                y_train_fold = y_train[train_idx]
                X_val_fold = base_predictions_train[val_idx]
                
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train_fold, y_train_fold)
                
                oof[val_idx] = model_clone.predict(X_val_fold)
                test_preds.append(model_clone.predict(base_predictions_test))
            
            stack_oof[name] = oof
            stack_test[name] = np.mean(test_preds, axis=0)
        
        # Blend stacking models
        final_oof = np.mean(list(stack_oof.values()), axis=0)
        final_test = np.mean(list(stack_test.values()), axis=0)
        
        return final_oof, final_test
    
    def optimize_blend_weights(self, predictions_train, y_train):
        """Optimize blend weights using Bayesian optimization"""
        from scipy.optimize import differential_evolution
        
        n_models = predictions_train.shape[1]
        
        def objective(weights):
            weights = weights / weights.sum()
            blend = (predictions_train * weights).sum(axis=1)
            return rmse(y_train, blend)
        
        # Differential evolution for global optimization
        bounds = [(0, 1)] * n_models
        result = differential_evolution(
            objective, 
            bounds, 
            seed=self.config['seed'],
            maxiter=1000,
            popsize=15,
            tol=1e-6
        )
        
        optimal_weights = result.x / result.x.sum()
        return optimal_weights

# -----------------------------
# Main Pipeline
# -----------------------------
def main_flaml_pipeline():
    """Main execution pipeline"""
    
    print("="*60)
    print("FLAML AutoML with Intelligent Meta-Modeling Pipeline")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(os.path.join(CONFIG['data_dir'], "train.csv"))
    test_df = pd.read_csv(os.path.join(CONFIG['data_dir'], "test.csv"))
    
    # Feature engineering
    print("\n1. Advanced Feature Engineering...")
    fe = AdvancedFeatureEngineer(CONFIG)
    train_fe, test_fe, feature_cols = fe.fit_transform(train_df, test_df)
    
    # Prepare data
    X = train_fe[feature_cols]
    y = train_fe[CONFIG['target_col']]
    X_test = test_fe[feature_cols]
    
    print(f"   Features created: {len(feature_cols)}")
    print(f"   Train shape: {X.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Target transformation
    print("\n2. Target Transformation...")
    pt = PowerTransformer(method='yeo-johnson')
    y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Train FLAML models
    print("\n3. Training FLAML AutoML Models...")
    flaml_trainer = FLAMLModelTrainer(CONFIG)
    flaml_results = flaml_trainer.train_diverse_models(X, y_transformed, X_test)
    
    print("\n   FLAML Model Results:")
    for result in flaml_results:
        # Inverse transform predictions
        result.oof_pred = pt.inverse_transform(result.oof_pred.reshape(-1, 1)).ravel()
        result.test_pred = pt.inverse_transform(result.test_pred.reshape(-1, 1)).ravel()
        cv_score = rmse(y, result.oof_pred)
        print(f"   {result.model_name}: CV RMSE = {cv_score:.5f}")
    
    # Train traditional models with FLAML-discovered hyperparameters
    print("\n4. Training Traditional Models with Optimized Hyperparameters...")
    
    # Extract best hyperparameters from FLAML for traditional models
    traditional_results = []
    
    # Use FLAML's best configurations to train more robust versions
    for flaml_result in flaml_results[:2]:  # Use top 2 FLAML configs
        if flaml_result.model_params and 'learner' in flaml_result.model_params:
            learner = flaml_result.model_params['learner']
            params = flaml_result.model_params.get('ml', {})
            
            if 'lgbm' in learner.lower():
                # Train LightGBM with CV
                lgb_params = {
                    'n_estimators': params.get('n_estimators', 1000),
                    'learning_rate': params.get('learning_rate', 0.05),
                    'num_leaves': params.get('num_leaves', 127),
                    'subsample': params.get('subsample', 0.9),
                    'colsample_bytree': params.get('colsample_bytree', 0.9),
                    'random_state': CONFIG['seed']
                }
                
                kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
                oof = np.zeros(len(X))
                test_preds = []
                
                for train_idx, val_idx in kf.split(X):
                    model = lgb.LGBMRegressor(**lgb_params)
                    model.fit(
                        X.iloc[train_idx], y_transformed[train_idx],
                        eval_set=[(X.iloc[val_idx], y_transformed[val_idx])],
                        callbacks=[lgb.early_stopping(100, verbose=False)]
                    )
                    oof[val_idx] = model.predict(X.iloc[val_idx])
                    test_preds.append(model.predict(X_test))
                
                oof = pt.inverse_transform(oof.reshape(-1, 1)).ravel()
                test_pred = pt.inverse_transform(np.mean(test_preds, axis=0).reshape(-1, 1)).ravel()
                
                traditional_results.append(ModelResult(
                    oof_pred=oof,
                    test_pred=test_pred,
                    cv_score=rmse(y, oof),
                    model_name=f'LGB_FLAML_HP_{len(traditional_results)}'
                ))
    
    # Combine all results
    all_results = flaml_results + traditional_results
    
    print("\n5. Intelligent Meta-Modeling...")
    meta_model = IntelligentMetaModel(CONFIG)
    
    # Prepare base predictions
    base_oof = np.column_stack([r.oof_pred for r in all_results])
    base_test = np.column_stack([r.test_pred for r in all_results])
    
    # Create meta-features
    meta_features_train = meta_model.create_meta_features([r.oof_pred for r in all_results])
    meta_features_test = meta_model.create_meta_features([r.test_pred for r in all_results])
    
    # Train different meta-models
    meta_predictions = {}
    
    if CONFIG['meta_models']['use_neural_net']:
        print("   Training Neural Network Meta-Model...")
        nn_oof, nn_test = meta_model.train_neural_meta_model(
            meta_features_train, y.values, meta_features_test
        )
        meta_predictions['neural_net'] = (nn_oof, nn_test)
        print(f"   Neural Net CV RMSE: {rmse(y, nn_oof):.5f}")
    
    if CONFIG['meta_models']['use_stacking']:
        print("   Training Stacking Meta-Model...")
        stack_oof, stack_test = meta_model.train_stacking_meta_model(
            base_oof, y.values, base_test
        )
        meta_predictions['stacking'] = (stack_oof, stack_test)
        print(f"   Stacking CV RMSE: {rmse(y, stack_oof):.5f}")
    
    if CONFIG['meta_models']['use_blending']:
        print("   Optimizing Blend Weights...")
        optimal_weights = meta_model.optimize_blend_weights(base_oof, y.values)
        blend_oof = (base_oof * optimal_weights).sum(axis=1)
        blend_test = (base_test * optimal_weights).sum(axis=1)
        meta_predictions['optimized_blend'] = (blend_oof, blend_test)
        print(f"   Optimized Blend CV RMSE: {rmse(y, blend_oof):.5f}")
        print(f"   Optimal weights: {optimal_weights}")
    
    # Final ensemble of meta-models
    print("\n6. Final Ensemble...")
    if len(meta_predictions) > 0:
        final_oof_list = [pred[0] for pred in meta_predictions.values()]
        final_test_list = [pred[1] for pred in meta_predictions.values()]
        
        # Weight meta-models by their CV performance
        meta_scores = [rmse(y, oof) for oof in final_oof_list]
        meta_weights = 1 / np.array(meta_scores)
        meta_weights = meta_weights / meta_weights.sum()
        
        final_oof = sum(w * oof for w, oof in zip(meta_weights, final_oof_list))
        final_test = sum(w * test for w, test in zip(meta_weights, final_test_list))
    else:
        # Fallback to simple average
        final_oof = base_oof.mean(axis=1)
        final_test = base_test.mean(axis=1)
    
    # Isotonic calibration
    print("   Applying isotonic calibration...")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(final_oof, y)
    final_oof_calibrated = iso.predict(final_oof)
    final_test_calibrated = iso.predict(final_test)
    
    # Clipping
    clip_low = max(y.quantile(0.001), 40)
    clip_high = min(y.quantile(0.999), 240)
    final_test_clipped = np.clip(final_test_calibrated, clip_low, clip_high)
    
    print(f"\n   Final CV RMSE: {rmse(y, final_oof_calibrated):.5f}")
    
    # Save submission
    submission = pd.DataFrame({
        CONFIG['id_col']: test_df[CONFIG['id_col']],
        CONFIG['target_col']: final_test_clipped
    })
    submission.to_csv("submission_flaml_meta.csv", index=False)
    print("\n7. Submission saved to 'submission_flaml_meta.csv'")
    
    # Save model artifacts
    print("\n8. Saving model artifacts...")
    artifacts = {
        'feature_engineer': fe,
        'power_transformer': pt,
        'meta_model': meta_model,
        'isotonic': iso,
        'config': CONFIG,
        'results': all_results,
        'meta_predictions': meta_predictions
    }
    
    joblib.dump(artifacts, 'flaml_meta_artifacts.pkl')
    print("   Model artifacts saved to 'flaml_meta_artifacts.pkl'")
    
    return submission, all_results, meta_predictions

if __name__ == "__main__":
    submission, results, meta_preds = main_flaml_pipeline()

