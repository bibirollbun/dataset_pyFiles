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

# # Comprehensive Medoid Clustering Pipeline
# This notebook implements a complete medoid clustering approach with multiple methods

import os
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
import gc
from typing import List, Tuple, Dict, Optional
from itertools import product

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import mode
from sklearn.metrics.pairwise import cosine_similarity

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

# Configuration
SEED = 42
N_FOLDS = 5
DATA_DIR = "/kaggle/input/playground-series-s5e9"
ID_COL = "id"
TARGET_COL = "BeatsPerMinute"

BASE_COLS = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy'
]

# Clustering configuration - reduced for faster execution
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
CLUSTERING_METHODS = ['kmeans', 'kmedoids', 'hierarchical']

# Utils
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

@dataclass
class ModelResult:
    oof: np.ndarray
    test_pred: np.ndarray
    cv_score: float
    n_clusters: int
    cluster_name: str
    model_type: str
    method: str

# Custom K-Medoids Implementation
class KMedoids:
    """K-Medoids clustering without external dependencies"""
    def __init__(self, n_clusters=5, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_ = None
        self.labels_ = None
        self.cluster_centers_ = None
        
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = len(X)
        actual_clusters = min(self.n_clusters, n_samples)
        
        # Initialize medoids randomly
        self.medoid_indices_ = np.random.choice(n_samples, actual_clusters, replace=False)
        
        for iteration in range(self.max_iter):
            # Assign to nearest medoid
            distances = cdist(X, X[self.medoid_indices_])
            labels = np.argmin(distances, axis=1)
            
            # Update medoids
            new_medoid_indices = []
            for k in range(actual_clusters):
                cluster_mask = labels == k
                if np.sum(cluster_mask) > 0:
                    cluster_points = X[cluster_mask]
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    if len(cluster_points) == 1:
                        new_medoid_indices.append(cluster_indices[0])
                    else:
                        within_cluster_distances = cdist(cluster_points, cluster_points)
                        total_distances = np.sum(within_cluster_distances, axis=1)
                        best_medoid_idx = np.argmin(total_distances)
                        new_medoid_indices.append(cluster_indices[best_medoid_idx])
                else:
                    new_medoid_indices.append(self.medoid_indices_[k])
            
            new_medoid_indices = np.array(new_medoid_indices)
            
            # Check convergence
            if np.array_equal(sorted(new_medoid_indices), sorted(self.medoid_indices_)):
                break
                
            self.medoid_indices_ = new_medoid_indices
        
        # Final assignment
        distances = cdist(X, X[self.medoid_indices_])
        self.labels_ = np.argmin(distances, axis=1)
        self.cluster_centers_ = X[self.medoid_indices_]
        return self
    
    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet")
        distances = cdist(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

# Clustering Feature Generator
class ClusteringFeatureGenerator:
    """Generate features based on clustering method"""
    
    def __init__(self, n_clusters, method='kmeans', random_state=42):
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clusterer = None
        
    def fit(self, X):
        X_array = X.values if hasattr(X, 'values') else X
        X_scaled = self.scaler.fit_transform(X_array)
        
        if self.n_clusters == 0:
            return self
        
        max_clusters = max(2, len(X) // 100)
        actual_clusters = min(self.n_clusters, max_clusters)
        
        if self.method == 'kmeans':
            self.clusterer = KMeans(n_clusters=actual_clusters, random_state=self.random_state, n_init=10)
            self.clusterer.fit(X_scaled)
        elif self.method == 'kmedoids':
            self.clusterer = KMedoids(n_clusters=actual_clusters, random_state=self.random_state)
            self.clusterer.fit(X_scaled)
        elif self.method == 'hierarchical':
            self.clusterer = AgglomerativeClustering(n_clusters=actual_clusters)
            self.clusterer.fit(X_scaled)
        
        return self
    
    def transform(self, X):
        X_array = X.values if hasattr(X, 'values') else X
        
        if self.n_clusters == 0 or self.clusterer is None:
            return X_array
        
        X_scaled = self.scaler.transform(X_array)
        features = [X_array]
        
        # Get cluster assignments
        if hasattr(self.clusterer, 'predict'):
            cluster_labels = self.clusterer.predict(X_scaled)
            features.append(cluster_labels.reshape(-1, 1))
            
            # Add distance features for KMeans
            if self.method == 'kmeans' and hasattr(self.clusterer, 'transform'):
                distances = self.clusterer.transform(X_scaled)
                features.append(distances.min(axis=1).reshape(-1, 1))
                features.append(distances.mean(axis=1).reshape(-1, 1))
        elif hasattr(self.clusterer, 'labels_'):
            # For hierarchical clustering
            features.append(self.clusterer.labels_.reshape(-1, 1))
        
        return np.hstack(features)

# Feature Engineering
def build_features(train, test, base_cols):
    """Create basic engineered features"""
    tr = train.copy()
    te = test.copy()
    
    for c in base_cols:
        tr[c] = tr[c].astype(np.float32)
        te[c] = te[c].astype(np.float32)
    
    # Statistical features
    tr['mean_all'] = tr[base_cols].mean(axis=1)
    te['mean_all'] = te[base_cols].mean(axis=1)
    
    tr['std_all'] = tr[base_cols].std(axis=1)
    te['std_all'] = te[base_cols].std(axis=1)
    
    # PCA features
    pca = PCA(n_components=3, random_state=42)
    pca_tr = pca.fit_transform(tr[base_cols])
    pca_te = pca.transform(te[base_cols])
    
    for i in range(3):
        tr[f'pca_{i}'] = pca_tr[:, i]
        te[f'pca_{i}'] = pca_te[:, i]
    
    # Simple polynomial features
    for c in base_cols[:3]:
        tr[f"{c}_sq"] = tr[c] ** 2
        te[f"{c}_sq"] = te[c] ** 2
    
    feat_cols = [c for c in tr.columns if c not in [ID_COL, TARGET_COL]]
    return tr, te, feat_cols

# Model Training
def train_model(X, y, X_test, model_type='lgb', n_folds=5):
    """Train a single model with cross-validation"""
    
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    # Model parameters
    if model_type == 'lgb':
        params = {
            'n_estimators': 1500,
            'learning_rate': 0.025,
            'num_leaves': 95,
            'max_depth': -1,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_lambda': 1.2,
            'min_child_samples': 25,
            'random_state': SEED,
            'n_jobs': -1,
            'verbose': -1
        }
    elif model_type == 'xgb':
        params = {
            'n_estimators': 1500,
            'learning_rate': 0.025,
            'max_depth': 7,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_lambda': 1.2,
            'random_state': SEED,
            'n_jobs': -1
        }
    elif model_type == 'cat':
        params = {
            'iterations': 1500,
            'learning_rate': 0.035,
            'depth': 7,
            'l2_leaf_reg': 3.5,
            'subsample': 0.85,
            'random_seed': SEED,
            'verbose': False
        }
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(75, verbose=False)]
            )
        elif model_type == 'xgb':
            model = XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=75,
                verbose=False
            )
        elif model_type == 'cat':
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=75,
                use_best_model=True
            )
        
        oof[val_idx] = model.predict(X_val)
        test_preds.append(model.predict(X_test))
    
    test_pred = np.mean(test_preds, axis=0)
    cv_score = rmse(y, oof)
    
    return oof, test_pred, cv_score

# Ensemble
class WeightedEnsemble:
    """Simple weighted ensemble"""
    
    def __init__(self):
        self.weights = None
        
    def fit(self, results, y_true, max_models=15):
        results_sorted = sorted(results, key=lambda x: x.cv_score)
        top_results = results_sorted[:min(max_models, len(results))]
        
        scores = np.array([r.cv_score for r in top_results])
        self.weights = 1 / scores
        self.weights = self.weights / self.weights.sum()
        
        pred_matrix = np.column_stack([r.oof for r in top_results])
        
        best_score = rmse(y_true, (pred_matrix * self.weights).sum(axis=1))
        best_weights = self.weights.copy()
        
        for _ in range(50):
            test_weights = self.weights + np.random.randn(len(self.weights)) * 0.04
            test_weights = np.maximum(test_weights, 0)
            if test_weights.sum() > 0:
                test_weights = test_weights / test_weights.sum()
                score = rmse(y_true, (pred_matrix * test_weights).sum(axis=1))
                if score < best_score:
                    best_score = score
                    best_weights = test_weights
        
        self.weights = best_weights
        return top_results
    
    def predict(self, results):
        oof_matrix = np.column_stack([r.oof for r in results])
        test_matrix = np.column_stack([r.test_pred for r in results])
        
        ensemble_oof = (oof_matrix * self.weights).sum(axis=1)
        ensemble_test = (test_matrix * self.weights).sum(axis=1)
        
        return ensemble_oof, ensemble_test

# Main Pipeline
def main():
    print("="*60)
    print("MEDOID CLUSTERING PIPELINE FOR KAGGLE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"   Train shape: {train.shape}")
    print(f"   Test shape: {test.shape}")
    
    # Feature engineering
    print("\n2. Creating features...")
    train_fe, test_fe, feature_cols = build_features(train, test, BASE_COLS)
    X = train_fe[feature_cols]
    y = train_fe[TARGET_COL].values
    X_test = test_fe[feature_cols]
    print(f"   Number of features: {len(feature_cols)}")
    
    # Target transformation
    print("\n3. Transforming target...")
    pt = PowerTransformer(method='yeo-johnson')
    y_transformed = pt.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Generate configurations
    configs = []
    configs.append({'n_clusters': 0, 'method': 'none', 'name': 'baseline'})
    
    for method in CLUSTERING_METHODS:
        for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            configs.append({
                'n_clusters': n_clusters,
                'method': method,
                'name': f'{method}_{n_clusters}'
            })
    
    print(f"\n4. Training models with {len(configs)} configurations...")
    
    # Train models
    all_results = []
    
    for i, config in enumerate(configs):
        n_clusters = config['n_clusters']
        method = config['method']
        cluster_name = config['name']
        
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{len(configs)}")
            if all_results:
                best = min([r.cv_score for r in all_results])
                print(f"   Best CV so far: {best:.5f}")
        
        try:
            # Apply clustering
            if method != 'none' and n_clusters > 0:
                generator = ClusteringFeatureGenerator(n_clusters, method=method)
                generator.fit(X)
                X_augmented = generator.transform(X)
                X_test_augmented = generator.transform(X_test)
            else:
                X_augmented = X.values
                X_test_augmented = X_test.values
            
            # Train models
            for model_type in ['lgb', 'cat']:  # Skip XGB for speed
                oof, test_pred, cv_score = train_model(
                    X_augmented, y_transformed, X_test_augmented, 
                    model_type, n_folds=4
                )
                
                # Inverse transform
                oof = pt.inverse_transform(oof.reshape(-1, 1)).ravel()
                test_pred = pt.inverse_transform(test_pred.reshape(-1, 1)).ravel()
                cv_score = rmse(y, oof)
                
                result = ModelResult(
                    oof=oof,
                    test_pred=test_pred,
                    cv_score=cv_score,
                    n_clusters=n_clusters,
                    cluster_name=cluster_name,
                    model_type=model_type,
                    method=method
                )
                
                all_results.append(result)
                
                if cv_score < 26.465:
                    print(f"   ✓ {cluster_name} ({model_type}): {cv_score:.5f}")
        
        except Exception as e:
            print(f"   ✗ {cluster_name} failed: {str(e)[:50]}")
        
        if (i + 1) % 20 == 0:
            gc.collect()
    
    print(f"\n5. Trained {len(all_results)} models successfully")
    
    # Show best models
    print("\n6. Top 5 models:")
    top_models = sorted(all_results, key=lambda x: x.cv_score)[:5]
    for i, r in enumerate(top_models, 1):
        print(f"   {i}. {r.method}_{r.n_clusters} ({r.model_type}): {r.cv_score:.5f}")
    
    # Create ensemble
    print("\n7. Creating ensemble...")
    ensemble = WeightedEnsemble()
    selected = ensemble.fit(all_results, y, max_models=15)
    ensemble_oof, ensemble_test = ensemble.predict(selected)
    
    ensemble_score = rmse(y, ensemble_oof)
    print(f"   Ensemble CV: {ensemble_score:.5f}")
    print(f"   Models in ensemble: {len(selected)}")
    
    # Post-processing
    print("\n8. Applying calibration...")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(ensemble_oof, y)
    
    final_oof = iso.predict(ensemble_oof)
    final_test = iso.predict(ensemble_test)
    
    final_score = rmse(y, final_oof)
    print(f"   Final CV: {final_score:.5f}")
    
    # Prepare submission
    clip_low = np.percentile(y, 0.5)
    clip_high = np.percentile(y, 99.5)
    final_test_clipped = np.clip(final_test, clip_low, clip_high)
    
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET_COL: final_test_clipped
    })
    
    submission.to_csv("submission.csv", index=False)
    print("\n9. Submission saved to submission.csv")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Models trained: {len(all_results)}")
    print(f"Best single model: {min([r.cv_score for r in all_results]):.5f}")
    print(f"Final ensemble: {final_score:.5f}")
    print(f"Improvement: {min([r.cv_score for r in all_results]) - final_score:.5f}")
    
    return submission

# Execute pipeline
if __name__ == "__main__":
    submission = main()
    print("\nPipeline completed successfully!")

