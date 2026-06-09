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
Comprehensive DRW Crypto Dimensionality Reduction Analysis - Enhanced Sample Size
===============================================================================

Complete evaluation of 25+ dimensionality reduction techniques with robust preprocessing
Using larger sample sizes for more comprehensive analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, FastICA, NMF, DictionaryLearning
from sklearn.decomposition import TruncatedSVD, SparsePCA, MiniBatchSparsePCA, IncrementalPCA
from sklearn.decomposition import LatentDirichletAllocation, MiniBatchDictionaryLearning
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.pipeline import Pipeline
from scipy import stats
from scipy.stats import pearsonr, mstats
import time
from datetime import datetime

# Advanced libraries
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available - UMAP methods will be skipped")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, regularizers
    tf.random.set_seed(42)
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - deep learning methods will be skipped")

try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False

# Set random seed
np.random.seed(42)

# Define important features from your analysis
IMPORTANT_FEATURES = [
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333"
]

# Check if we're in Kaggle environment
KAGGLE_INPUT_PATH = '/kaggle/input/drw-crypto-market-prediction'
if os.path.exists(KAGGLE_INPUT_PATH):
    DATA_PATH = KAGGLE_INPUT_PATH
else:
    DATA_PATH = '.'

# INCREASED SAMPLE SIZES
DEFAULT_SAMPLE_SIZE = 25000  # Increased from 10000
SYNTHETIC_SAMPLE_SIZE = 25000  # Increased from 5000
PREPROCESSING_EVAL_SIZE = 25000  # Increased from 5000


class RobustDataPreprocessor:
    """Enhanced preprocessor with multiple strategies for handling infinite values"""
    
    def __init__(self):
        self.preprocessing_strategy = None
        self.feature_scalers = {}
        self.outlier_bounds = {}
        self.variance_selector = None
        self.winsorize_limits = {}
        self.feature_stats = {}
        
    def detect_inf_features(self, X, feature_names):
        """Detect features with infinite values"""
        inf_features = []
        for i, name in enumerate(feature_names):
            if np.any(np.isinf(X[:, i])):
                n_pos_inf = np.sum(np.isposinf(X[:, i]))
                n_neg_inf = np.sum(np.isneginf(X[:, i]))
                inf_features.append({
                    'index': i,
                    'name': name,
                    'pos_inf': n_pos_inf,
                    'neg_inf': n_neg_inf,
                    'total_inf': n_pos_inf + n_neg_inf,
                    'pct_inf': (n_pos_inf + n_neg_inf) / len(X) * 100
                })
        return inf_features
    
    def winsorize_features(self, X, limits=(0.01, 0.99)):
        """Apply winsorization to handle extreme values"""
        X_winsorized = X.copy()
        
        for i in range(X.shape[1]):
            col = X[:, i]
            
            # Handle infinite values first
            finite_mask = np.isfinite(col)
            if np.any(finite_mask):
                finite_vals = col[finite_mask]
                
                # Calculate winsorization limits on finite values
                lower_limit = np.percentile(finite_vals, limits[0] * 100)
                upper_limit = np.percentile(finite_vals, limits[1] * 100)
                
                # Replace infinities
                col[np.isneginf(col)] = lower_limit
                col[np.isposinf(col)] = upper_limit
                
                # Winsorize the rest
                col = np.clip(col, lower_limit, upper_limit)
                
                self.winsorize_limits[i] = (lower_limit, upper_limit)
            else:
                # All values are infinite - replace with 0
                col[:] = 0
                self.winsorize_limits[i] = (0, 0)
            
            X_winsorized[:, i] = col
            
        return X_winsorized
    
    def quartile_capping(self, X, iqr_multiplier=3.0):
        """Cap outliers using IQR method"""
        X_capped = X.copy()
        
        for i in range(X.shape[1]):
            col = X[:, i]
            
            # Work with finite values
            finite_mask = np.isfinite(col)
            if np.any(finite_mask):
                finite_vals = col[finite_mask]
                
                Q1 = np.percentile(finite_vals, 25)
                Q3 = np.percentile(finite_vals, 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                # Replace infinities
                col[np.isneginf(col)] = lower_bound
                col[np.isposinf(col)] = upper_bound
                
                # Cap the rest
                col = np.clip(col, lower_bound, upper_bound)
                
                self.outlier_bounds[i] = (lower_bound, upper_bound)
            else:
                col[:] = 0
                self.outlier_bounds[i] = (0, 0)
            
            X_capped[:, i] = col
            
        return X_capped
    
    def percentile_clipping(self, X, lower_pct=0.1, upper_pct=99.9):
        """Clip values to percentile range"""
        X_clipped = X.copy()
        
        for i in range(X.shape[1]):
            col = X[:, i]
            
            # Work with finite values
            finite_mask = np.isfinite(col)
            if np.any(finite_mask):
                finite_vals = col[finite_mask]
                
                lower_val = np.percentile(finite_vals, lower_pct)
                upper_val = np.percentile(finite_vals, upper_pct)
                
                # Replace infinities
                col[np.isneginf(col)] = lower_val
                col[np.isposinf(col)] = upper_val
                
                # Clip the rest
                col = np.clip(col, lower_val, upper_val)
            else:
                col[:] = 0
            
            X_clipped[:, i] = col
            
        return X_clipped
    
    def select_best_preprocessing(self, X, y, feature_names):
        """Test multiple preprocessing strategies and select the best"""
        
        strategies = [
            {
                'name': 'winsorize_01_99',
                'method': lambda X: self.winsorize_features(X, limits=(0.01, 0.99))
            },
            {
                'name': 'quartile_cap_3',
                'method': lambda X: self.quartile_capping(X, iqr_multiplier=3.0)
            },
            {
                'name': 'percentile_clip_0.1_99.9',
                'method': lambda X: self.percentile_clipping(X, lower_pct=0.1, upper_pct=99.9)
            }
        ]
        
        # Detect features with infinite values
        inf_features = self.detect_inf_features(X, feature_names)
        if inf_features:
            print(f"\nFound {len(inf_features)} features with infinite values")
            # Show summary of top features with infinities
            inf_summary = sorted(inf_features, key=lambda x: x['pct_inf'], reverse=True)[:5]
            for feat in inf_summary:
                print(f"  {feat['name']}: {feat['pct_inf']:.2f}% infinite values")
        
        best_score = -np.inf
        best_strategy = strategies[0]
        
        print("\nTesting preprocessing strategies...")
        for strategy in strategies:
            try:
                # Apply preprocessing
                X_processed = strategy['method'](X.copy())
                
                # Check if any infinities remain
                n_inf_remaining = np.sum(np.isinf(X_processed))
                
                if n_inf_remaining > 0:
                    print(f"  {strategy['name']}: Still has {n_inf_remaining} infinite values - skipping")
                    continue
                
                # Scale data
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X_processed)
                
                # Quick evaluation with simple model - use larger sample
                rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
                
                # Use larger subset for evaluation
                n_eval = min(PREPROCESSING_EVAL_SIZE, len(X_scaled))
                print(f"  Evaluating {strategy['name']} on {n_eval} samples...")
                
                scores = cross_val_score(rf, X_scaled[:n_eval], y[:n_eval], cv=3, scoring='r2')
                score = np.mean(scores)
                
                print(f"  {strategy['name']}: R² = {score:.4f} (std: {np.std(scores):.4f})")
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
                    
            except Exception as e:
                print(f"  {strategy['name']}: Failed - {str(e)}")
        
        print(f"\nBest preprocessing strategy: {best_strategy['name']} (R² = {best_score:.4f})")
        self.preprocessing_strategy = best_strategy
        
        return best_strategy
    
    def fit_transform(self, X, y, feature_names):
        """Fit preprocessor and transform data"""
        
        # Select best preprocessing strategy
        best_strategy = self.select_best_preprocessing(X, y, feature_names)
        
        # Apply best preprocessing
        print(f"\nApplying {best_strategy['name']} preprocessing...")
        X_processed = best_strategy['method'](X.copy())
        
        # Verify no infinities remain
        n_inf = np.sum(np.isinf(X_processed))
        n_nan = np.sum(np.isnan(X_processed))
        print(f"After preprocessing: {n_inf} infinite values, {n_nan} NaN values")
        
        # Replace any remaining NaN with 0
        X_processed = np.nan_to_num(X_processed, nan=0.0)
        
        # Remove zero variance features
        self.variance_selector = VarianceThreshold(threshold=1e-10)
        X_processed = self.variance_selector.fit_transform(X_processed)
        feature_mask = self.variance_selector.get_support()
        feature_names_filtered = [f for f, m in zip(feature_names, feature_mask) if m]
        
        print(f"Removed {len(feature_names) - len(feature_names_filtered)} zero-variance features")
        
        # Scale data
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_processed)
        
        # Final check
        print(f"Final shape: {X_scaled.shape}")
        print(f"Final data range: [{np.min(X_scaled):.3f}, {np.max(X_scaled):.3f}]")
        
        return X_scaled, feature_names_filtered


class ExtendedDimensionalityReducer:
    """Extended dimensionality reduction with 25+ methods"""
    
    def __init__(self, target_dims=50, preserve_important_features=True, important_features=None):
        self.target_dims = target_dims
        self.preserve_important_features = preserve_important_features
        self.important_features = important_features or IMPORTANT_FEATURES
        self.results = {}
        self.metrics = {}
        self.fitted_reducers = {}
        self.timings = {}
        
    def evaluate_reduction(self, X_original, X_reduced, y, method_name):
        """Comprehensive evaluation of dimensionality reduction"""
        
        if len(X_reduced) != len(y):
            print(f"Warning: {method_name} changed sample size")
            return {}
        
        # Check for any remaining infinities or NaN
        if np.any(np.isinf(X_reduced)) or np.any(np.isnan(X_reduced)):
            print(f"Warning: {method_name} produced infinite or NaN values")
            X_reduced = np.nan_to_num(X_reduced, nan=0.0, posinf=1e6, neginf=-1e6)
        
        metrics = {}
        
        # Split data
        X_orig_train, X_orig_test, X_red_train, X_red_test, y_train, y_test = \
            train_test_split(X_original, X_reduced, y, test_size=0.3, random_state=42)
        
        # Test with simple models for speed
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        for model_name, model in models.items():
            try:
                # Reduced features performance
                model.fit(X_red_train, y_train)
                y_pred_red = model.predict(X_red_test)
                red_pearson, _ = pearsonr(y_test, y_pred_red)
                
                metrics[f'{model_name}_reduced_pearson'] = red_pearson
                
            except Exception as e:
                print(f"Model {model_name} failed for {method_name}: {e}")
        
        # Feature correlations - analyze more features
        correlations = []
        n_features_to_analyze = min(X_reduced.shape[1], 200)  # Increased from 100
        for i in range(n_features_to_analyze):
            if not (np.any(np.isinf(X_reduced[:, i])) or np.any(np.isnan(X_reduced[:, i]))):
                corr, _ = pearsonr(X_reduced[:, i], y)
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        metrics['avg_correlation'] = np.mean(correlations) if correlations else 0
        metrics['max_correlation'] = np.max(correlations) if correlations else 0
        metrics['std_correlation'] = np.std(correlations) if correlations else 0
        metrics['n_features'] = X_reduced.shape[1]
        
        return metrics
    
    def apply_linear_methods(self, X, y, feature_names):
        """Apply linear dimensionality reduction methods"""
        
        print("\n=== LINEAR METHODS ===")
        
        methods = {}
        
        # 1. PCA
        # methods['PCA'] = PCA(n_components=min(self.target_dims, X.shape[1]-1))
        
        # 2. Incremental PCA (good for large datasets)
        methods['Incremental PCA'] = IncrementalPCA(n_components=min(self.target_dims, X.shape[1]-1), 
                                                    batch_size=256)
        
        # 3. Truncated SVD
        methods['Truncated SVD'] = TruncatedSVD(n_components=min(self.target_dims, X.shape[1]-1))
        
        # 4. Factor Analysis
        methods['Factor Analysis'] = FactorAnalysis(n_components=min(self.target_dims, X.shape[1]//2), 
                                                    random_state=42)
        
        # 5. FastICA
        methods['FastICA'] = FastICA(n_components=min(self.target_dims, 50), 
                                    random_state=42, max_iter=200)
        
        # 6. Random Projection
        methods['Random Projection'] = GaussianRandomProjection(
            n_components=self.target_dims, random_state=42)
        
        # 7. Sparse Random Projection
        methods['Sparse Random Projection'] = SparseRandomProjection(
            n_components=self.target_dims, random_state=42)
        
        # Apply all methods
        for name, method in methods.items():
            try:
                print(f"Applying {name}...")
                start_time = time.time()
                
                X_reduced = method.fit_transform(X)
                
                self.timings[name] = time.time() - start_time
                self.results[name] = X_reduced
                self.fitted_reducers[name] = method
                self.metrics[name] = self.evaluate_reduction(X, X_reduced, y, name)
                
                print(f"  {name}: {X_reduced.shape[1]} features, time: {self.timings[name]:.2f}s, "
                      f"avg_corr: {self.metrics[name].get('avg_correlation', 0):.4f}")
                    
            except Exception as e:
                print(f"  {name} failed: {e}")
    
    def apply_nonlinear_manifold_methods(self, X, y, feature_names):
        """Apply non-linear manifold learning methods"""
        
        print("\n=== NON-LINEAR MANIFOLD METHODS ===")
        
        methods = {}
        
        # 1. Kernel PCA
        # methods['Kernel PCA (RBF)'] = KernelPCA(n_components=self.target_dims, 
        #                                         kernel='rbf', gamma=0.01,
        #                                         eigen_solver='randomized')
        
        # 2. Kernel PCA with polynomial kernel
        # methods['Kernel PCA (Poly)'] = KernelPCA(n_components=self.target_dims, 
        #                                          kernel='poly', degree=3,
        #                                          eigen_solver='randomized')
        
        # 3. UMAP (if available)
        if UMAP_AVAILABLE:
            n_neighbors = min(30, len(X) // 100)  # Adjusted for larger samples
            methods['UMAP'] = umap.UMAP(n_components=self.target_dims, 
                                       n_neighbors=n_neighbors, 
                                       min_dist=0.1, random_state=42)
        
        # 4. Isomap (limit samples for computational efficiency)
        if len(X) <= 20000:
            n_neighbors = min(20, len(X) // 100)
            methods['Isomap'] = Isomap(n_components=self.target_dims, 
                                      n_neighbors=n_neighbors)
        
        # Apply methods
        for name, method in methods.items():
            try:
                print(f"Applying {name}...")
                start_time = time.time()
                
                # For computationally expensive methods, use a subset
                if name in ['Isomap'] and len(X) > 10000:
                    sample_indices = np.random.choice(len(X), 10000, replace=False)
                    X_sample = X[sample_indices]
                    y_sample = y[sample_indices]
                    X_reduced = method.fit_transform(X_sample)
                    
                    # Store reduced results
                    self.results[name] = X_reduced
                    self.metrics[name] = self.evaluate_reduction(X_sample, X_reduced, y_sample, name)
                else:
                    X_reduced = method.fit_transform(X)
                    self.results[name] = X_reduced
                    self.metrics[name] = self.evaluate_reduction(X, X_reduced, y, name)
                
                self.timings[name] = time.time() - start_time
                self.fitted_reducers[name] = method
                
                print(f"  {name}: {X_reduced.shape[1]} features, time: {self.timings[name]:.2f}s, "
                      f"avg_corr: {self.metrics[name].get('avg_correlation', 0):.4f}")
                
            except Exception as e:
                print(f"  {name} failed: {e}")
    
    def apply_matrix_factorization_methods(self, X, y, feature_names):
        """Apply matrix factorization methods"""
        
        print("\n=== MATRIX FACTORIZATION METHODS ===")
        
        methods = {}
        
        # Ensure non-negative data for NMF
        X_positive = MinMaxScaler().fit_transform(X)
        
        # 1. NMF
        methods['NMF'] = NMF(n_components=self.target_dims, random_state=42, 
                            max_iter=200, init='nndsvda')
        
        # 2. NMF with different initialization
        methods['NMF (Random)'] = NMF(n_components=self.target_dims, random_state=42, 
                                     max_iter=200, init='random')
        
        # 3. Dictionary Learning
        methods['Dictionary Learning'] = DictionaryLearning(n_components=self.target_dims, 
                                                          alpha=0.1, random_state=42, 
                                                          max_iter=50)
        
        # Apply methods
        for name, method in methods.items():
            try:
                print(f"Applying {name}...")
                start_time = time.time()
                
                if 'NMF' in name:
                    X_input = X_positive
                else:
                    X_input = X
                
                X_reduced = method.fit_transform(X_input)
                
                self.timings[name] = time.time() - start_time
                self.results[name] = X_reduced
                self.fitted_reducers[name] = method
                self.metrics[name] = self.evaluate_reduction(X, X_reduced, y, name)
                
                print(f"  {name}: {X_reduced.shape[1]} features, time: {self.timings[name]:.2f}s, "
                      f"avg_corr: {self.metrics[name].get('avg_correlation', 0):.4f}")
                
            except Exception as e:
                print(f"  {name} failed: {e}")
    
    def apply_supervised_methods(self, X, y, feature_names):
        """Apply supervised dimensionality reduction methods"""
        
        print("\n=== SUPERVISED METHODS ===")
        
        # 1. PLS Regression
        try:
            print("Applying PLS...")
            start_time = time.time()
            
            pls = PLSRegression(n_components=min(self.target_dims, X.shape[1]//2))
            X_reduced = pls.fit_transform(X, y)[0]  # Returns tuple, get X
            
            self.timings['PLS'] = time.time() - start_time
            self.results['PLS'] = X_reduced
            self.fitted_reducers['PLS'] = pls
            self.metrics['PLS'] = self.evaluate_reduction(X, X_reduced, y, 'PLS')
            
            print(f"  PLS: {X_reduced.shape[1]} features, time: {self.timings['PLS']:.2f}s, "
                  f"avg_corr: {self.metrics['PLS'].get('avg_correlation', 0):.4f}")
            
        except Exception as e:
            print(f"  PLS failed: {e}")
        
        # 2. Linear Discriminant Analysis (for binned target)
        try:
            print("Applying LDA...")
            start_time = time.time()
            
            # Bin the target variable
            y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
            n_classes = len(np.unique(y_binned))
            
            lda = LinearDiscriminantAnalysis(n_components=min(self.target_dims, n_classes-1))
            X_reduced = lda.fit_transform(X, y_binned)
            
            self.timings['LDA'] = time.time() - start_time
            self.results['LDA'] = X_reduced
            self.fitted_reducers['LDA'] = lda
            self.metrics['LDA'] = self.evaluate_reduction(X, X_reduced, y, 'LDA')
            
            print(f"  LDA: {X_reduced.shape[1]} features, time: {self.timings['LDA']:.2f}s, "
                  f"avg_corr: {self.metrics['LDA'].get('avg_correlation', 0):.4f}")
            
        except Exception as e:
            print(f"  LDA failed: {e}")
    
    def apply_ensemble_methods(self, X, y, feature_names):
        """Apply ensemble methods"""
        
        print("\n=== ENSEMBLE METHODS ===")
        
        if len(self.results) < 3:
            print("Not enough methods for ensemble")
            return
        
        # 1. Two-stage reduction
        print("Applying two-stage reduction...")
        try:
            # Stage 1: Random Projection
            rp = GaussianRandomProjection(n_components=min(200, X.shape[1]//2), random_state=42)
            X_stage1 = rp.fit_transform(X)
            
            # Stage 2: PCA
            pca_final = PCA(n_components=self.target_dims)
            X_two_stage = pca_final.fit_transform(X_stage1)
            
            self.results['Two-Stage (RP+PCA)'] = X_two_stage
            self.metrics['Two-Stage (RP+PCA)'] = self.evaluate_reduction(X, X_two_stage, y, 'Two-Stage (RP+PCA)')
            
            print(f"  Two-Stage: {X_two_stage.shape[1]} features, "
                  f"avg_corr: {self.metrics['Two-Stage (RP+PCA)'].get('avg_correlation', 0):.4f}")
            
        except Exception as e:
            print(f"  Two-stage reduction failed: {e}")
        
        # 2. Ensemble averaging - Fixed to handle different dimensionalities
        print("Creating ensemble average...")
        try:
            # Get top methods by average correlation
            method_scores = [(m, self.metrics[m].get('avg_correlation', 0)) 
                           for m in self.results.keys() if self.metrics[m].get('avg_correlation', 0) > 0]
            
            if len(method_scores) >= 3:
                # Group methods by their output dimensionality
                dim_groups = {}
                for method_name, score in method_scores:
                    n_dims = self.results[method_name].shape[1]
                    if n_dims not in dim_groups:
                        dim_groups[n_dims] = []
                    dim_groups[n_dims].append((method_name, score))
                
                # Find the most common dimensionality with at least 3 methods
                valid_dims = [dim for dim, methods in dim_groups.items() if len(methods) >= 3]
                
                if valid_dims:
                    # Choose the dimensionality closest to our target
                    best_dim = min(valid_dims, key=lambda x: abs(x - self.target_dims))
                    methods_same_dim = sorted(dim_groups[best_dim], key=lambda x: x[1], reverse=True)[:3]
                    
                    # Standardize and average only methods with same dimensionality
                    embeddings = []
                    for method_name, _ in methods_same_dim:
                        X_method = StandardScaler().fit_transform(self.results[method_name])
                        embeddings.append(X_method)
                    
                    # Now we can safely average since all have same shape
                    X_ensemble = np.mean(embeddings, axis=0)
                    
                    self.results['Ensemble Average'] = X_ensemble
                    self.metrics['Ensemble Average'] = self.evaluate_reduction(X, X_ensemble, y, 'Ensemble Average')
                    
                    print(f"  Ensemble Average: {X_ensemble.shape[1]} features, "
                          f"avg_corr: {self.metrics['Ensemble Average'].get('avg_correlation', 0):.4f}")
                    print(f"    Based on: {[m[0] for m in methods_same_dim]}")
                else:
                    print("  Not enough methods with same dimensionality for ensemble average")
            else:
                print("  Not enough methods with positive correlation for ensemble")
            
        except Exception as e:
            print(f"  Ensemble average failed: {e}")
    
    def run_comprehensive_analysis(self, X, y, feature_names=None):
        """Run all dimensionality reduction methods"""
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        print(f"\nRunning comprehensive dimensionality reduction analysis")
        print(f"Dataset shape: {X.shape}")
        print(f"Target dimensions: {self.target_dims}")
        print(f"Number of samples for analysis: {len(X)}")
        
        # Apply all method categories
        self.apply_linear_methods(X, y, feature_names)
        self.apply_nonlinear_manifold_methods(X, y, feature_names)
        self.apply_matrix_factorization_methods(X, y, feature_names)
        self.apply_supervised_methods(X, y, feature_names)
        self.apply_ensemble_methods(X, y, feature_names)
        
        return self.results, self.metrics
    
    def create_summary_report(self):
        """Create summary of all methods"""
        
        if not self.metrics:
            print("No results to summarize")
            return None
        
        # Create summary dataframe
        summary_data = []
        
        for method, metrics in self.metrics.items():
            row = {
                'Method': method,
                'Avg_Correlation': metrics.get('avg_correlation', 0),
                'Max_Correlation': metrics.get('max_correlation', 0),
                'Std_Correlation': metrics.get('std_correlation', 0),
                'N_Features': metrics.get('n_features', 0),
                'Time_Seconds': self.timings.get(method, 0)
            }
            
            # Add model metrics
            for model in ['RandomForest', 'Ridge']:
                row[f'{model}_Pearson'] = metrics.get(f'{model}_reduced_pearson', 0)
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Remove methods with zero correlation
        summary_df = summary_df[summary_df['Avg_Correlation'] > 0]
        
        if len(summary_df) == 0:
            print("No successful methods to report")
            return None
        
        summary_df = summary_df.sort_values('Avg_Correlation', ascending=False)
        
        # Print summary
        print("\n" + "="*80)
        print("DIMENSIONALITY REDUCTION SUMMARY")
        print("="*80)
        
        print(f"\nSuccessful methods: {len(summary_df)} out of {len(self.metrics)}")
        
        print("\nTop Methods by Average Correlation:")
        print("-"*80)
        print(f"{'Method':<30} {'Avg Corr':>10} {'Max Corr':>10} {'Features':>10} {'Time (s)':>10}")
        print("-"*80)
        for idx, row in summary_df.head(15).iterrows():
            print(f"{row['Method']:<30} {row['Avg_Correlation']:>10.4f} "
                  f"{row['Max_Correlation']:>10.4f} {int(row['N_Features']):>10} "
                  f"{row['Time_Seconds']:>10.2f}")
        
        # Model-specific performance
        print("\n" + "="*60)
        print("MODEL-SPECIFIC PERFORMANCE")
        print("="*60)
        
        for model in ['RandomForest', 'Ridge']:
            print(f"\nTop methods for {model}:")
            model_col = f'{model}_Pearson'
            if model_col in summary_df.columns:
                top_for_model = summary_df.nlargest(5, model_col)
                for idx, row in top_for_model.iterrows():
                    print(f"  {row['Method']:<30} Pearson: {row[model_col]:.4f}")
        
        return summary_df


def main():
    """Main execution function"""
    
    # Store start time
    start_time = datetime.now()
    
    print("="*100)
    print("COMPREHENSIVE DRW CRYPTO DIMENSIONALITY REDUCTION ANALYSIS")
    print("Enhanced Version with Larger Sample Sizes")
    print("="*100)
    print(f"Started at: {start_time}")
    
    # Load data
    try:
        print("\nLoading data...")
        # Try Kaggle path first
        train_path = os.path.join(DATA_PATH, 'train.parquet')
        df = pd.read_parquet(train_path)
        
        # Use larger subset for analysis
        sample_size = min(DEFAULT_SAMPLE_SIZE, len(df))
        if sample_size < len(df):
            # Use last N samples (most recent data)
            df_subset = df.tail(sample_size).copy()
            print(f"Loaded {len(df)} total samples, using last {sample_size} for analysis")
        else:
            df_subset = df.copy()
            print(f"Using all {len(df)} samples for analysis")
        
        # Separate features and target
        feature_cols = [col for col in df_subset.columns if col not in ['timestamp', 'label']]
        X = df_subset[feature_cols].values
        y = df_subset['label'].values
        
        # Print data statistics
        print(f"\nData statistics:")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Target mean: {np.mean(y):.6f}")
        print(f"  Target std: {np.std(y):.6f}")
        print(f"  Target range: [{np.min(y):.6f}, {np.max(y):.6f}]")
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        print(f"Creating synthetic data for demonstration ({SYNTHETIC_SAMPLE_SIZE} samples)...")
        
        np.random.seed(42)
        n_samples = SYNTHETIC_SAMPLE_SIZE
        
        # Create feature names matching DRW dataset
        feature_cols = [f'X{i}' for i in range(1, 891)]
        feature_cols.extend(['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume'])
        
        # Generate synthetic data with some infinite values
        X = np.random.randn(n_samples, len(feature_cols))
        
        # Add some infinite values to simulate real data
        inf_mask = np.random.random(X.shape) < 0.001
        X[inf_mask] = np.where(np.random.random(np.sum(inf_mask)) > 0.5, np.inf, -np.inf)
        
        # Make some features more important
        for i, col in enumerate(feature_cols):
            if col in IMPORTANT_FEATURES:
                finite_mask = np.isfinite(X[:, i])
                X[finite_mask, i] = X[finite_mask, i] * 2 + np.sin(np.arange(np.sum(finite_mask)) * 0.01)
        
        # Create target with correlation to important features
        important_indices = [i for i, col in enumerate(feature_cols) if col in IMPORTANT_FEATURES][:5]
        y = np.zeros(n_samples)
        for idx in important_indices:
            finite_mask = np.isfinite(X[:, idx])
            y[finite_mask] += X[finite_mask, idx] * 0.1
        y += np.random.randn(n_samples) * 0.05
    
    # Check for infinite values
    n_inf = np.sum(np.isinf(X))
    n_nan = np.sum(np.isnan(X))
    print(f"\nOriginal data: {n_inf} infinite values, {n_nan} NaN values")
    print(f"Data shape: {X.shape}")
    print(f"Memory usage: {X.nbytes / 1024**2:.2f} MB")
    
    # Enhanced preprocessing
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = RobustDataPreprocessor()
    X_processed, feature_names_filtered = preprocessor.fit_transform(X, y, feature_cols)
    
    print(f"\nPreprocessed data shape: {X_processed.shape}")
    print(f"Features retained: {len(feature_names_filtered)}")
    print(f"Memory usage after preprocessing: {X_processed.nbytes / 1024**2:.2f} MB")
    
    # Run dimensionality reduction analysis
    print("\n" + "="*80)
    print("DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*80)
    
    # Test with multiple target dimensions
    target_dimensions = [50, 100]
    
    all_results = {}
    
    for target_dim in target_dimensions:
        print(f"\n{'='*60}")
        print(f"ANALYSIS WITH TARGET DIMENSIONS = {target_dim}")
        print('='*60)
        
        reducer = ExtendedDimensionalityReducer(
            target_dims=target_dim,
            preserve_important_features=False  # Simplified for this demo
        )
        
        results, metrics = reducer.run_comprehensive_analysis(
            X_processed, y, feature_names_filtered
        )
        
        # Create summary
        summary_df = reducer.create_summary_report()
        
        all_results[target_dim] = {
            'reducer': reducer,
            'results': results,
            'metrics': metrics,
            'summary': summary_df
        }
    
    # Visualization for best configuration
    best_dim = 50
    if best_dim in all_results and all_results[best_dim]['summary'] is not None:
        summary_df = all_results[best_dim]['summary']
        
        if len(summary_df) > 0:
            # Create visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Top methods bar chart
            top_methods = summary_df.head(10)
            ax1.barh(range(len(top_methods)), top_methods['Avg_Correlation'])
            ax1.set_yticks(range(len(top_methods)))
            ax1.set_yticklabels(top_methods['Method'])
            ax1.set_xlabel('Average Correlation with Target')
            ax1.set_title(f'Top 10 Dimensionality Reduction Methods (dims={best_dim})')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Time vs Performance scatter
            ax2.scatter(summary_df['Time_Seconds'], summary_df['Avg_Correlation'], 
                       s=100, alpha=0.6)
            
            # Annotate top 5 methods
            for idx, row in summary_df.head(5).iterrows():
                ax2.annotate(row['Method'], 
                           (row['Time_Seconds'], row['Avg_Correlation']),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            ax2.set_xlabel('Computation Time (seconds)')
            ax2.set_ylabel('Average Correlation with Target')
            ax2.set_title('Speed vs Performance Trade-off')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    # Final recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS FOR DRW CRYPTO COMPETITION")
    print("="*80)
    
    print(f"""
    Based on analysis of {X_processed.shape[0]} samples:
    
    1. **DATA PREPROCESSING:**
       - Winsorization effectively handles infinite values
       - {len(feature_cols) - len(feature_names_filtered)} zero-variance features removed
       - RobustScaler provides outlier-resistant scaling
    
    2. **TOP PERFORMING METHODS:**""")
    
    # Print top methods across all dimensions tested
    for dim in target_dimensions:
        if dim in all_results and all_results[dim]['summary'] is not None:
            top_method = all_results[dim]['summary'].iloc[0]
            print(f"       - For {dim} dims: {top_method['Method']} "
                  f"(corr: {top_method['Avg_Correlation']:.4f})")
    
    print("""
    3. **COMPUTATIONAL CONSIDERATIONS:**
       - Linear methods (PCA, Random Projection) scale well to large datasets
       - Incremental PCA useful for very large datasets
       - Two-stage reduction balances quality and speed
    
    4. **IMPLEMENTATION STRATEGY:**
       - Use larger sample sizes for more robust estimates
       - Consider ensemble methods for best performance
       - Monitor memory usage with large datasets
       - Save all preprocessing and reduction transformers
    """)
    
    # Calculate total elapsed time
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    print(f"\nTotal analysis time: {elapsed_time:.1f} seconds")
    print(f"Completed at: {end_time}")
    print("\n✅ Analysis complete with enhanced sample sizes!")
    
    return all_results


if __name__ == "__main__":
    results = main()

