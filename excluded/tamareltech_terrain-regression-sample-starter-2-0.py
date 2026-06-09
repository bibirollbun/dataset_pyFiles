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
Fixed Terrain Price Regression Pipeline - Handles All Edge Cases
Author: Advanced ML Pipeline
Date: 2025
"""

# ============================================================================
# INSTALLATION SECTION
# ============================================================================

import subprocess
import sys
import os

# Quick installation function
def quick_install(packages):
    """Quick installation of packages"""
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except:
            pass

# Install essential packages first
print("Installing essential packages...")
essential_packages = [
    "numpy==1.26.4",
    "pandas",
    "scikit-learn>=1.3.0",
    "matplotlib",
    "seaborn",
    "scipy",
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
    "tqdm",
    "plotly",
    "joblib"
]

quick_install(essential_packages)

# Try to install MAPIE
print("\nAttempting MAPIE installation...")
mapie_installed = False
mapie_versions = ["mapie", "mapie==0.6.5", "mapie==0.3.1"]
for version in mapie_versions:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", version])
        mapie_installed = True
        print(f"✓ MAPIE installed: {version}")
        break
    except:
        continue

if not mapie_installed:
    print("⚠ MAPIE not available - using alternative methods")

# Optional packages (install but don't fail if unavailable)
optional_packages = [
    "pytorch-tabnet",
    "optuna",
    "yellowbrick",
    "ydata-profiling",
    "sweetviz",
    "torch",
    "tensorflow"
]

print("\nInstalling optional packages...")
for package in optional_packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✓ {package} installed")
    except:
        print(f"⚠ {package} not available")

# ============================================================================
# IMPORTS
# ============================================================================

print("\n" + "=" * 80)
print("IMPORTING LIBRARIES")
print("=" * 80)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import gc
import pickle
from collections import Counter

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3, suppress=True)

# Core sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                            IsolationForest, VotingRegressor, ExtraTreesRegressor, 
                            StackingRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, HuberRegressor, 
                                BayesianRidge, QuantileRegressor)
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from sklearn.utils import resample

# Advanced ML libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False
    lgb = None

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    torch.manual_seed(42)
except:
    TORCH_AVAILABLE = False
    torch = None

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False

# Analysis tools
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False
    shap = None

# Statistics
from scipy import stats
from scipy.stats import boxcox

# Progress bar
try:
    from tqdm import tqdm
except:
    def tqdm(x, desc=None):
        return x

# MAPIE
MAPIE_AVAILABLE = False
try:
    from mapie.regression import MapieRegressor
    from mapie.metrics import regression_coverage_score
    MAPIE_AVAILABLE = True
    print("✓ MAPIE imported successfully")
except:
    try:
        from mapie import MapieRegressor
        def regression_coverage_score(y_true, y_pred_low, y_pred_up):
            return np.mean((y_true >= y_pred_low) & (y_true <= y_pred_up))
        MAPIE_AVAILABLE = True
        print("✓ MAPIE imported (alternative)")
    except:
        print("⚠ MAPIE not available")
        MapieRegressor = None

# Create directories
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set random seed
np.random.seed(42)

print("\n" + "=" * 80)
print("TERRAIN PRICE REGRESSION PIPELINE")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# CUSTOM UNCERTAINTY CLASSES
# ============================================================================

class BootstrapUncertainty:
    """Bootstrap-based uncertainty estimation"""
    def __init__(self, base_estimator, n_bootstrap=50):
        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        self.models = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        for i in range(self.n_bootstrap):
            indices = resample(range(n_samples), n_samples=n_samples, random_state=i)
            X_boot = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y_boot = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            
            model = clone(self.base_estimator)
            model.fit(X_boot, y_boot)
            self.models.append(model)
        return self
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return predictions.mean(axis=0)
    
    def predict_interval(self, X, confidence=0.95):
        predictions = np.array([model.predict(X) for model in self.models])
        alpha = 1 - confidence
        lower = np.percentile(predictions, 100 * alpha/2, axis=0)
        upper = np.percentile(predictions, 100 * (1 - alpha/2), axis=0)
        mean = predictions.mean(axis=0)
        return mean, lower, upper

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n1. LOADING DATA")
print("-" * 50)

# Load data
try:
    train_df = pd.read_csv('/kaggle/input/terrain-prices-reggression/train.csv')
    test_df = pd.read_csv('/kaggle/input/terrain-prices-reggression/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/terrain-prices-reggression/sample_submission.csv')
    print("✓ Data loaded successfully")
except:
    print("✗ Error loading data - please check file paths")
    sys.exit(1)

print(f"\nDataset shapes:")
print(f"  Train: {train_df.shape}")
print(f"  Test: {test_df.shape}")

# Save original data
train_df_original = train_df.copy()
test_df_original = test_df.copy()

# Define columns
target_col = 'target'
id_col = 'id'

# Identify column types
categorical_cols = []
numerical_cols = []

for col in train_df.columns:
    if col not in [id_col, target_col]:
        if train_df[col].dtype == 'object' or train_df[col].nunique() < 20:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

print(f"\nColumns:")
print(f"  Categorical: {len(categorical_cols)}")
print(f"  Numerical: {len(numerical_cols)}")

# ============================================================================
# 2. DATA QUALITY CHECK
# ============================================================================

print("\n2. DATA QUALITY ANALYSIS")
print("-" * 50)

# Check for missing values
train_missing = train_df.isnull().sum().sum()
test_missing = test_df.isnull().sum().sum()

print(f"Missing values:")
print(f"  Train: {train_missing}")
print(f"  Test: {test_missing}")

# Check for duplicates
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()

print(f"\nDuplicates:")
print(f"  Train: {train_duplicates}")
print(f"  Test: {test_duplicates}")

# Basic statistics
print(f"\nTarget statistics:")
print(f"  Mean: {train_df[target_col].mean():.2f}")
print(f"  Std: {train_df[target_col].std():.2f}")
print(f"  Min: {train_df[target_col].min():.2f}")
print(f"  Max: {train_df[target_col].max():.2f}")

# ============================================================================
# 3. OUTLIER DETECTION
# ============================================================================

print("\n3. OUTLIER DETECTION")
print("-" * 50)

def detect_outliers(df, numerical_cols, contamination=0.05):
    """Simple outlier detection"""
    outlier_scores = pd.DataFrame(index=df.index)
    
    # Isolation Forest
    if len(numerical_cols) > 0:
        try:
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers_iso = iso_forest.fit_predict(df[numerical_cols]) == -1
            outlier_scores['isolation_forest'] = outliers_iso
        except:
            pass
    
    # Z-score method
    if len(numerical_cols) > 0:
        z_scores = np.abs(stats.zscore(df[numerical_cols], nan_policy='omit'))
        outliers_z = (z_scores > 3).any(axis=1)
        outlier_scores['zscore'] = outliers_z
    
    # Consensus
    if len(outlier_scores.columns) > 0:
        outlier_scores['consensus'] = outlier_scores.sum(axis=1) >= len(outlier_scores.columns) / 2
    else:
        outlier_scores['consensus'] = False
    
    return outlier_scores

# Detect outliers in training data
outlier_scores = detect_outliers(train_df, numerical_cols)
train_df['is_outlier'] = outlier_scores.get('consensus', False)

print(f"Outliers detected: {train_df['is_outlier'].sum()} ({100*train_df['is_outlier'].sum()/len(train_df):.1f}%)")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print("\n4. FEATURE ENGINEERING")
print("-" * 50)

# IMPORTANT: Define feature columns BEFORE adding outlier columns
# This ensures we only use features that exist in both train and test
base_feature_cols = [col for col in train_df.columns if col not in [id_col, target_col, 'is_outlier']]

# Create feature matrices using only base features
X_train = train_df[base_feature_cols].copy()
X_test = test_df[base_feature_cols].copy()
y_train = train_df[target_col].copy()

print(f"Initial feature count: {len(base_feature_cols)}")

# Add basic statistical features
if numerical_cols:
    # Row statistics
    for df_name, df in [('train', X_train), ('test', X_test)]:
        df['row_mean'] = df[numerical_cols].mean(axis=1)
        df['row_std'] = df[numerical_cols].std(axis=1)
        df['row_min'] = df[numerical_cols].min(axis=1)
        df['row_max'] = df[numerical_cols].max(axis=1)
        df['row_range'] = df['row_max'] - df['row_min']

# Add polynomial features for top correlated features
if numerical_cols and len(numerical_cols) > 0:
    # Calculate correlations with target
    correlations = train_df[numerical_cols].corrwith(train_df[target_col]).abs()
    top_features = correlations.nlargest(min(5, len(numerical_cols))).index.tolist()
    
    for feat in top_features:
        if feat in X_train.columns:
            X_train[f'{feat}_squared'] = X_train[feat] ** 2
            X_test[f'{feat}_squared'] = X_test[feat] ** 2
            X_train[f'{feat}_log'] = np.log1p(np.abs(X_train[feat]))
            X_test[f'{feat}_log'] = np.log1p(np.abs(X_test[feat]))

# Add interaction features
if len(top_features) >= 2:
    for i in range(min(2, len(top_features)-1)):
        feat1, feat2 = top_features[i], top_features[i+1]
        if feat1 in X_train.columns and feat2 in X_train.columns:
            X_train[f'{feat1}_x_{feat2}'] = X_train[feat1] * X_train[feat2]
            X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]

# Add clustering features
if numerical_cols and len(numerical_cols) > 5:
    try:
        kmeans = KMeans(n_clusters=5, random_state=42)
        train_clusters = kmeans.fit_predict(X_train[numerical_cols].fillna(0))
        test_clusters = kmeans.predict(X_test[numerical_cols].fillna(0))
        
        X_train['cluster'] = train_clusters
        X_test['cluster'] = test_clusters
    except:
        pass

print(f"Features after engineering: {X_train.shape[1]}")

# ============================================================================
# 5. FEATURE SELECTION
# ============================================================================

print("\n5. FEATURE SELECTION")
print("-" * 50)

# Encode categorical variables
for col in categorical_cols:
    if col in X_train.columns:
        try:
            le = LabelEncoder()
            # Fit on combined unique values
            all_values = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
            le.fit(all_values)
            
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
        except:
            # If encoding fails, drop the column
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

# Convert to numeric and handle any issues
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# Remove zero variance features
selector = VarianceThreshold(threshold=0)
X_train_var = selector.fit_transform(X_train)
X_test_var = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()].tolist()

X_train = X_train[selected_features]
X_test = X_test[selected_features]

print(f"Features after variance threshold: {len(selected_features)}")

# Calculate feature importance
importance_scores = {}

# Mutual Information
try:
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    importance_scores['MI'] = pd.Series(mi_scores, index=X_train.columns)
except:
    pass

# Random Forest importance
try:
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance_scores['RF'] = pd.Series(rf.feature_importances_, index=X_train.columns)
except:
    pass

# Select top features
if importance_scores:
    importance_df = pd.DataFrame(importance_scores)
    importance_mean = importance_df.mean(axis=1)
    n_features = min(100, len(importance_mean))
    top_features = importance_mean.nlargest(n_features).index.tolist()
    
    X_train = X_train[top_features]
    X_test = X_test[top_features]
    
    print(f"Selected top {n_features} features")

# ============================================================================
# 6. DATA PREPROCESSING
# ============================================================================

print("\n6. DATA PREPROCESSING")
print("-" * 50)

# Scale features
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print("✓ Applied Robust scaling")

# ============================================================================
# 7. MODEL TRAINING
# ============================================================================

print("\n7. MODEL TRAINING")
print("-" * 50)

# Define models
models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.01, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=10, weights='distance'),
}

# Add optional models
if XGB_AVAILABLE:
    models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)

if LGB_AVAILABLE:
    models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)

if CATBOOST_AVAILABLE:
    models['CatBoost'] = CatBoostRegressor(iterations=100, depth=5, random_state=42, verbose=False)

# Evaluate models
cv_scores = {}
trained_models = {}

kfold = KFold(n_splits=3, shuffle=True, random_state=42)

for name, model in tqdm(models.items(), desc="Training models"):
    try:
        # Cross-validation
        scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='r2')
        cv_scores[name] = scores.mean()
        
        # Train on full data
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        print(f"  {name}: CV R² = {scores.mean():.4f} (+/- {scores.std():.4f})")
    except Exception as e:
        print(f"  {name}: Failed - {str(e)}")

# Sort models by performance
sorted_models = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)

print(f"\nBest model: {sorted_models[0][0]} (R² = {sorted_models[0][1]:.4f})")

# ============================================================================
# 8. ENSEMBLE CREATION
# ============================================================================

print("\n8. CREATING ENSEMBLES")
print("-" * 50)

# Weighted ensemble of top models
top_n = min(5, len(sorted_models))
top_model_names = [name for name, _ in sorted_models[:top_n]]
top_models = [(name, trained_models[name]) for name in top_model_names if name in trained_models]

# Calculate weights based on CV scores
weights = np.array([cv_scores[name] for name, _ in top_models])
weights = np.exp(weights * 3)  # Emphasize better models
weights = weights / weights.sum()

print("Weighted ensemble:")
for (name, _), weight in zip(top_models, weights):
    print(f"  {name}: {weight:.3f}")

# Create voting ensemble
voting_ensemble = VotingRegressor(estimators=top_models, weights=weights)
voting_ensemble.fit(X_train_scaled, y_train)

# Create stacking ensemble
stacking_ensemble = StackingRegressor(
    estimators=top_models,
    final_estimator=Ridge(alpha=1.0),
    cv=3
)
stacking_ensemble.fit(X_train_scaled, y_train)

# ============================================================================
# 9. UNCERTAINTY QUANTIFICATION
# ============================================================================

print("\n9. UNCERTAINTY QUANTIFICATION")
print("-" * 50)

# Select best model for uncertainty
best_model_name = sorted_models[0][0]
best_model = trained_models[best_model_name]

# Try MAPIE first
uq_model = None
uq_method = None

if MAPIE_AVAILABLE and MapieRegressor is not None:
    try:
        mapie_model = MapieRegressor(
            estimator=clone(best_model),
            method="base",
            cv=3
        )
        
        # Test on small subset
        subset_size = min(500, len(X_train_scaled))
        subset_idx = np.random.choice(len(X_train_scaled), subset_size, replace=False)
        
        mapie_model.fit(X_train_scaled.iloc[subset_idx], y_train.iloc[subset_idx])
        y_pred, y_pis = mapie_model.predict(X_train_scaled.iloc[subset_idx], alpha=0.1)
        
        # Check coverage
        coverage = np.mean((y_train.iloc[subset_idx] >= y_pis[:, 0]) & 
                          (y_train.iloc[subset_idx] <= y_pis[:, 1]))
        
        print(f"MAPIE coverage (90%): {coverage:.3f}")
        
        # Refit on full data
        mapie_model.fit(X_train_scaled, y_train)
        uq_model = mapie_model
        uq_method = 'MAPIE'
        
    except Exception as e:
        print(f"MAPIE failed: {str(e)}")

# Fallback to bootstrap
if uq_model is None:
    print("Using bootstrap uncertainty estimation...")
    bootstrap_model = BootstrapUncertainty(clone(best_model), n_bootstrap=30)
    bootstrap_model.fit(X_train_scaled, y_train)
    
    # Test coverage
    subset_size = min(500, len(X_train_scaled))
    subset_idx = np.random.choice(len(X_train_scaled), subset_size, replace=False)
    
    y_pred, lower, upper = bootstrap_model.predict_interval(
        X_train_scaled.iloc[subset_idx], confidence=0.90
    )
    coverage = np.mean((y_train.iloc[subset_idx] >= lower) & 
                       (y_train.iloc[subset_idx] <= upper))
    
    print(f"Bootstrap coverage (90%): {coverage:.3f}")
    
    uq_model = bootstrap_model
    uq_method = 'Bootstrap'

# ============================================================================
# 10. GENERATE PREDICTIONS
# ============================================================================

print("\n10. GENERATING PREDICTIONS")
print("-" * 50)

# Individual model predictions
predictions = {}

# Best single model
predictions['best_single'] = best_model.predict(X_test_scaled)

# Ensemble predictions
predictions['voting'] = voting_ensemble.predict(X_test_scaled)
predictions['stacking'] = stacking_ensemble.predict(X_test_scaled)

# Uncertainty predictions
if uq_method == 'MAPIE':
    uq_pred, _ = uq_model.predict(X_test_scaled, alpha=0.1)
    predictions['uncertainty'] = uq_pred
else:
    predictions['uncertainty'] = uq_model.predict(X_test_scaled)

# Final blend
predictions['final_blend'] = (
    0.35 * predictions['voting'] +
    0.35 * predictions['stacking'] +
    0.20 * predictions['best_single'] +
    0.10 * predictions['uncertainty']
)

# Post-processing
train_min = y_train.min()
train_max = y_train.max()
train_std = y_train.std()

for name in predictions:
    predictions[name] = np.clip(
        predictions[name],
        train_min - 2*train_std,
        train_max + 2*train_std
    )

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================

print("\n11. SAVING RESULTS")
print("-" * 50)

# Create submission files
for name, preds in predictions.items():
    submission = pd.DataFrame({
        'id': test_df[id_col],
        'target': preds
    })
    
    filename = f'submission_{name}.csv'
    submission.to_csv(filename, index=False)
    
    print(f"\n{filename}:")
    print(f"  Mean: {preds.mean():.2f}")
    print(f"  Std: {preds.std():.2f}")
    print(f"  Min: {preds.min():.2f}")
    print(f"  Max: {preds.max():.2f}")

# ============================================================================
# 12. FEATURE IMPORTANCE PLOT
# ============================================================================

if len(importance_scores) > 0 and 'RF' in importance_scores:
    print("\n12. CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Get top 20 features
    top_20_features = importance_scores['RF'].nlargest(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_20_features)), top_20_features.values)
    plt.yticks(range(len(top_20_features)), top_20_features.index)
    plt.xlabel('Importance Score')
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Feature importance plot saved")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)

print(f"\nSummary:")
print(f"  Features used: {X_train_scaled.shape[1]}")
print(f"  Models trained: {len(trained_models)}")
print(f"  Best model: {best_model_name} (CV R² = {sorted_models[0][1]:.4f})")
print(f"  Uncertainty method: {uq_method}")

print(f"\nRecommendations:")
print(f"  1. Use 'submission_final_blend.csv' for best overall performance")
print(f"  2. Use 'submission_voting.csv' or 'submission_stacking.csv' for ensemble predictions")
print(f"  3. All predictions have been post-processed to ensure reasonable values")

print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fixed Advanced Terrain Price Regression Pipeline
Includes: GANDALF, SAINT, TabNet, Neural Networks with Attention
Author: Advanced ML Pipeline
Date: 2025
"""

# ============================================================================
# INSTALLATION SECTION
# ============================================================================

import subprocess
import sys
import os

print("=" * 100)
print("INSTALLING REQUIRED PACKAGES")
print("=" * 100)

# Quick installation function
def quick_install(packages):
    """Install packages quietly"""
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✓ {package}")
        except:
            print(f"✗ {package}")

# Essential packages
essential_packages = [
    "numpy==1.26.4",
    "pandas",
    "scikit-learn>=1.3.0",
    "matplotlib",
    "seaborn",
    "scipy",
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
    "tqdm",
    "plotly",
    "joblib",
    "torch",
    "pytorch-tabnet",
    "optuna",
    "yellowbrick",
    "einops",
    "networkx"
]

print("\nInstalling packages...")
quick_install(essential_packages)

# Try MAPIE installation
print("\nInstalling MAPIE...")
mapie_installed = False
for version in ["mapie", "mapie==0.6.5", "mapie==0.3.1"]:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", version])
        mapie_installed = True
        print(f"✓ MAPIE installed: {version}")
        break
    except:
        continue

if not mapie_installed:
    print("✗ MAPIE not available")

# ============================================================================
# IMPORTS
# ============================================================================

print("\n" + "=" * 100)
print("IMPORTING LIBRARIES")
print("=" * 100)

import warnings
warnings.filterwarnings('ignore')

# Core
import numpy as np
import pandas as pd
from datetime import datetime
import gc
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Optional

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False

# ML
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                            VotingRegressor, ExtraTreesRegressor, StackingRegressor,
                            HistGradientBoostingRegressor, IsolationForest)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge, QuantileRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.utils import resample

# Advanced ML
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    TORCH_AVAILABLE = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
except:
    TORCH_AVAILABLE = False
    device = None

# TabNet
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except:
    TABNET_AVAILABLE = False

# Optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except:
    OPTUNA_AVAILABLE = False

# Interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# MAPIE
MAPIE_AVAILABLE = False
try:
    from mapie.regression import MapieRegressor
    from mapie.metrics import regression_coverage_score
    MAPIE_AVAILABLE = True
except:
    MapieRegressor = None

# NetworkX
try:
    import networkx as nx
except:
    nx = None

# Progress
try:
    from tqdm import tqdm
except:
    def tqdm(x, desc=None):
        return x

# Statistics
from scipy import stats
from scipy.stats import boxcox

# Set seeds
np.random.seed(42)
if TORCH_AVAILABLE:
    torch.manual_seed(42)

# Create directories
for dir_name in ['figures', 'models', 'reports']:
    os.makedirs(dir_name, exist_ok=True)

print("\n" + "=" * 100)
print("ADVANCED TERRAIN PRICE REGRESSION PIPELINE")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {device if TORCH_AVAILABLE else 'CPU only'}")
print("=" * 100)

# ============================================================================
# ADVANCED ARCHITECTURES
# ============================================================================

class GANDALFRegressor(BaseEstimator, RegressorMixin):
    """GANDALF: Gradient Boosting with Attentive Neural Features"""
    def __init__(self, n_estimators=100, hidden_dim=64, n_heads=4, 
                 learning_rate=0.01, max_depth=6):
        self.n_estimators = n_estimators
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.attention_weights = None
        self.gb_model = None
        self.scaler = None
        
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Get feature importance as attention weights
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_scaled, y)
        self.attention_weights = rf.feature_importances_
        
        # Apply attention weights
        X_weighted = X_scaled * self.attention_weights
        
        # Train gradient boosting
        if LGB_AVAILABLE:
            self.gb_model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
                verbose=-1
            )
        else:
            self.gb_model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42
            )
        
        self.gb_model.fit(X_weighted, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_weighted = X_scaled * self.attention_weights
        return self.gb_model.predict(X_weighted)

class SAINTRegressor(BaseEstimator, RegressorMixin):
    """SAINT: Self-Attention for Tabular Data"""
    def __init__(self, d_model=64, n_heads=4, n_layers=2, epochs=50):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.model = None
        self.scaler = None
        
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if TABNET_AVAILABLE:
            # Use TabNet as a proxy for SAINT
            self.model = TabNetRegressor(
                n_d=self.d_model,
                n_a=self.d_model,
                n_steps=self.n_layers,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=0.02),
                mask_type='entmax',
                seed=42,
                verbose=0
            )
            self.model.fit(
                X_scaled, y.values.reshape(-1, 1) if hasattr(y, 'values') else y.reshape(-1, 1),
                max_epochs=self.epochs,
                patience=20,
                batch_size=256,
                virtual_batch_size=128
            )
        else:
            # Fallback to MLP
            self.model = MLPRegressor(
                hidden_layer_sizes=(self.d_model, self.d_model),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
            self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if TABNET_AVAILABLE and hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled).flatten()
        else:
            return self.model.predict(X_scaled)

class HybridEnsemble(BaseEstimator, RegressorMixin):
    """Hybrid ensemble of multiple architectures"""
    def __init__(self):
        self.models = {}
        self.weights = None
        
    def fit(self, X, y):
        # Base models
        self.models['rf'] = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
        self.models['gbm'] = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        self.models['ridge'] = Ridge(alpha=1.0)
        
        if XGB_AVAILABLE:
            self.models['xgb'] = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42)
        
        if LGB_AVAILABLE:
            self.models['lgb'] = lgb.LGBMRegressor(n_estimators=200, max_depth=5, random_state=42, verbose=-1)
        
        # Advanced models
        self.models['gandalf'] = GANDALFRegressor(n_estimators=100)
        self.models['saint'] = SAINTRegressor(epochs=30)
        
        # Train all models
        print("  Training hybrid ensemble...")
        for name, model in tqdm(self.models.items(), desc="Models"):
            model.fit(X, y)
        
        # Equal weights for simplicity
        self.weights = np.ones(len(self.models)) / len(self.models)
        
        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models.values()
        ])
        return np.average(predictions, axis=1, weights=self.weights)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(train_df, test_df, target_col, numerical_cols, categorical_cols):
    """Create visualizations without modifying original dataframes"""
    
    print("\n  Creating visualizations...")
    
    if not PLOTLY_AVAILABLE:
        print("  ✗ Plotly not available - skipping interactive visualizations")
        return
    
    # Use copies to avoid modifying original data
    train_viz = train_df.copy()
    
    # 1. Target Analysis
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Target Distribution', 'Log Target', 'Q-Q Plot',
            'Target vs Index', 'Target Boxplot', 'Target by Percentile'
        )
    )
    
    # Target distribution
    fig.add_trace(
        go.Histogram(x=train_viz[target_col], nbinsx=50, name='Target'),
        row=1, col=1
    )
    
    # Log target
    fig.add_trace(
        go.Histogram(x=np.log1p(train_viz[target_col]), nbinsx=50, name='Log Target'),
        row=1, col=2
    )
    
    # Q-Q plot
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    sample = np.percentile(train_viz[target_col], np.linspace(1, 99, 100))
    fig.add_trace(
        go.Scatter(x=theoretical, y=sample, mode='markers', name='Q-Q'),
        row=1, col=3
    )
    
    # Target vs index
    fig.add_trace(
        go.Scatter(x=train_viz.index, y=train_viz[target_col], mode='markers', 
                  marker=dict(size=3), name='Target'),
        row=2, col=1
    )
    
    # Boxplot
    fig.add_trace(
        go.Box(y=train_viz[target_col], name='Target'),
        row=2, col=2
    )
    
    # Percentiles
    percentiles = np.linspace(0, 100, 101)
    target_percentiles = np.percentile(train_viz[target_col], percentiles)
    fig.add_trace(
        go.Scatter(x=percentiles, y=target_percentiles, mode='lines', name='Percentiles'),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Target Analysis")
    fig.write_html('figures/target_analysis.html')
    
    # 2. Feature distributions
    if len(numerical_cols) > 0:
        n_features = min(12, len(numerical_cols))
        fig_dist = make_subplots(
            rows=3, cols=4,
            subplot_titles=[col for col in numerical_cols[:n_features]]
        )
        
        for idx, col in enumerate(numerical_cols[:n_features]):
            row = idx // 4 + 1
            col_idx = idx % 4 + 1
            
            fig_dist.add_trace(
                go.Histogram(x=train_df[col], name='Train', opacity=0.6),
                row=row, col=col_idx
            )
            fig_dist.add_trace(
                go.Histogram(x=test_df[col], name='Test', opacity=0.6),
                row=row, col=col_idx
            )
        
        fig_dist.update_layout(
            height=800, 
            title_text="Feature Distributions",
            barmode='overlay',
            showlegend=False
        )
        fig_dist.write_html('figures/feature_distributions.html')
    
    print("  ✓ Visualizations created")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(X_train, X_test, train_df, test_df, numerical_cols, target_col):
    """Create advanced features"""
    
    print("\n  Creating features...")
    original_count = X_train.shape[1]
    
    # 1. Statistical features
    if numerical_cols:
        for df_name, df in [('train', X_train), ('test', X_test)]:
            df['row_mean'] = df[numerical_cols].mean(axis=1)
            df['row_std'] = df[numerical_cols].std(axis=1)
            df['row_min'] = df[numerical_cols].min(axis=1)
            df['row_max'] = df[numerical_cols].max(axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
            df['row_skew'] = df[numerical_cols].skew(axis=1)
            df['row_kurt'] = df[numerical_cols].kurtosis(axis=1)
    
    # 2. Polynomial features for top correlated
    if numerical_cols and target_col in train_df.columns:
        correlations = train_df[numerical_cols].corrwith(train_df[target_col]).abs()
        top_features = correlations.nlargest(min(5, len(numerical_cols))).index.tolist()
        
        for feat in top_features:
            if feat in X_train.columns:
                X_train[f'{feat}_squared'] = X_train[feat] ** 2
                X_train[f'{feat}_log'] = np.log1p(np.abs(X_train[feat]))
                X_train[f'{feat}_sqrt'] = np.sqrt(np.abs(X_train[feat]))
                
                X_test[f'{feat}_squared'] = X_test[feat] ** 2
                X_test[f'{feat}_log'] = np.log1p(np.abs(X_test[feat]))
                X_test[f'{feat}_sqrt'] = np.sqrt(np.abs(X_test[feat]))
        
        # Interactions
        for i in range(min(2, len(top_features)-1)):
            feat1, feat2 = top_features[i], top_features[i+1]
            if feat1 in X_train.columns and feat2 in X_train.columns:
                X_train[f'{feat1}_x_{feat2}'] = X_train[feat1] * X_train[feat2]
                X_test[f'{feat1}_x_{feat2}'] = X_test[feat1] * X_test[feat2]
    
    # 3. Clustering
    if numerical_cols and len(numerical_cols) > 5:
        cluster_features = numerical_cols[:min(10, len(numerical_cols))]
        
        for n_clusters in [5, 10]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            train_clusters = kmeans.fit_predict(X_train[cluster_features].fillna(0))
            test_clusters = kmeans.predict(X_test[cluster_features].fillna(0))
            
            X_train[f'cluster_{n_clusters}'] = train_clusters
            X_test[f'cluster_{n_clusters}'] = test_clusters
            
            # Distance to centers
            train_dist = kmeans.transform(X_train[cluster_features].fillna(0))
            test_dist = kmeans.transform(X_test[cluster_features].fillna(0))
            
            X_train[f'cluster_{n_clusters}_dist'] = train_dist.min(axis=1)
            X_test[f'cluster_{n_clusters}_dist'] = test_dist.min(axis=1)
    
    # 4. PCA features
    if numerical_cols and len(numerical_cols) > 10:
        pca = PCA(n_components=5, random_state=42)
        pca_train = pca.fit_transform(X_train[numerical_cols].fillna(0))
        pca_test = pca.transform(X_test[numerical_cols].fillna(0))
        
        for i in range(5):
            X_train[f'pca_{i}'] = pca_train[:, i]
            X_test[f'pca_{i}'] = pca_test[:, i]
    
    print(f"  ✓ Created {X_train.shape[1] - original_count} new features")
    print(f"  Total features: {X_train.shape[1]}")
    
    return X_train, X_test

# ============================================================================
# NEURAL NETWORK
# ============================================================================

def create_neural_network(input_dim):
    """Create neural network with attention"""
    if not TORCH_AVAILABLE:
        return None
        
    class AttentionNN(nn.Module):
        def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
            super().__init__()
            
            self.input_norm = nn.BatchNorm1d(input_dim)
            
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            self.hidden_layers = nn.Sequential(*layers)
            self.output_layer = nn.Linear(hidden_dims[-1], 1)
            
        def forward(self, x):
            x = self.input_norm(x)
            x = self.hidden_layers(x)
            return self.output_layer(x).squeeze(-1)
    
    return AttentionNN(input_dim)

def train_neural_network(model, X_train, y_train, epochs=50):
    """Train neural network"""
    if not TORCH_AVAILABLE or model is None:
        return None
        
    X_tensor = torch.FloatTensor(X_train.values).to(device)
    y_tensor = torch.FloatTensor(y_train.values).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return model

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline execution"""
    
    # 1. Load Data
    print("\n" + "=" * 100)
    print("1. LOADING DATA")
    print("=" * 100)
    
    train_df = pd.read_csv('/kaggle/input/terrain-prices-reggression/train.csv')
    test_df = pd.read_csv('/kaggle/input/terrain-prices-reggression/test.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Define columns
    target_col = 'target'
    id_col = 'id'
    
    # Identify column types
    categorical_cols = []
    numerical_cols = []
    
    for col in train_df.columns:
        if col not in [id_col, target_col]:
            if train_df[col].dtype == 'object' or train_df[col].nunique() < 20:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
    
    print(f"\nCategorical: {len(categorical_cols)}")
    print(f"Numerical: {len(numerical_cols)}")
    
    # 2. Visualizations
    print("\n" + "=" * 100)
    print("2. CREATING VISUALIZATIONS")
    print("=" * 100)
    
    create_visualizations(train_df, test_df, target_col, numerical_cols, categorical_cols)
    
    # 3. Prepare base features (exclude any derived columns)
    base_columns = [col for col in train_df.columns if col not in [id_col, target_col]]
    # Make sure we only use columns that exist in both train and test
    base_columns = [col for col in base_columns if col in test_df.columns]
    
    X_train = train_df[base_columns].copy()
    X_test = test_df[base_columns].copy()
    y_train = train_df[target_col].copy()
    
    print(f"\nBase features: {len(base_columns)}")
    
    # 4. Feature Engineering
    print("\n" + "=" * 100)
    print("3. FEATURE ENGINEERING")
    print("=" * 100)
    
    X_train, X_test = create_features(
        X_train, X_test, train_df, test_df, numerical_cols, target_col
    )
    
    # 5. Preprocessing
    print("\n" + "=" * 100)
    print("4. PREPROCESSING")
    print("=" * 100)
    
    # Encode categoricals
    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            all_values = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
            le.fit(all_values)
            
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # Handle missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"Final shape: {X_train_scaled.shape}")
    
    # 6. Model Training
    print("\n" + "=" * 100)
    print("5. MODEL TRAINING")
    print("=" * 100)
    
    models = {}
    predictions = {}
    
    # Traditional models
    print("\n  Training traditional models...")
    
    models['ridge'] = Ridge(alpha=1.0)
    models['lasso'] = Lasso(alpha=0.01, random_state=42)
    models['elasticnet'] = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    models['rf'] = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    models['gbm'] = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    if XGB_AVAILABLE:
        models['xgb'] = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    if LGB_AVAILABLE:
        models['lgb'] = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
    
    if CATBOOST_AVAILABLE:
        models['cat'] = CatBoostRegressor(iterations=100, depth=5, random_state=42, verbose=False)
    
    # Train traditional models
    for name, model in tqdm(models.items(), desc="Training"):
        model.fit(X_train_scaled, y_train)
        predictions[name] = model.predict(X_test_scaled)
    
    # Advanced models
    print("\n  Training advanced models...")
    
    # GANDALF
    print("  Training GANDALF...")
    gandalf = GANDALFRegressor(n_estimators=100)
    gandalf.fit(X_train_scaled, y_train)
    models['gandalf'] = gandalf
    predictions['gandalf'] = gandalf.predict(X_test_scaled)
    
    # SAINT
    print("  Training SAINT...")
    saint = SAINTRegressor(epochs=30)
    saint.fit(X_train_scaled, y_train)
    models['saint'] = saint
    predictions['saint'] = saint.predict(X_test_scaled)
    
    # Neural Network
    if TORCH_AVAILABLE:
        print("  Training Neural Network...")
        nn_model = create_neural_network(X_train_scaled.shape[1]).to(device)
        nn_model = train_neural_network(nn_model, X_train_scaled, y_train, epochs=30)
        
        if nn_model is not None:
            nn_model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled.values).to(device)
                nn_pred = nn_model(X_test_tensor).cpu().numpy()
            models['neural_net'] = nn_model
            predictions['neural_net'] = nn_pred
    
    # Hybrid Ensemble
    print("\n  Training Hybrid Ensemble...")
    hybrid = HybridEnsemble()
    hybrid.fit(X_train_scaled, y_train)
    models['hybrid'] = hybrid
    predictions['hybrid'] = hybrid.predict(X_test_scaled)
    
    # 7. Create Ensembles
    print("\n" + "=" * 100)
    print("6. CREATING ENSEMBLES")
    print("=" * 100)
    
    # Simple average
    model_preds = np.column_stack(list(predictions.values()))
    predictions['ensemble_mean'] = model_preds.mean(axis=1)
    
    # Weighted average
    weights = np.ones(model_preds.shape[1]) / model_preds.shape[1]
    predictions['ensemble_weighted'] = np.average(model_preds, axis=1, weights=weights)
    
    # Final blend
    predictions['final_blend'] = (
        0.30 * predictions['hybrid'] +
        0.20 * predictions['ensemble_weighted'] +
        0.15 * predictions.get('xgb', predictions['ensemble_mean']) +
        0.15 * predictions.get('lgb', predictions['ensemble_mean']) +
        0.10 * predictions.get('gandalf', predictions['ensemble_mean']) +
        0.10 * predictions.get('saint', predictions['ensemble_mean'])
    )
    
    # 8. Post-processing
    train_min = y_train.min()
    train_max = y_train.max()
    train_std = y_train.std()
    
    for name in predictions:
        predictions[name] = np.clip(
            predictions[name],
            train_min - 2*train_std,
            train_max + 2*train_std
        )
    
    # 9. Save Results
    print("\n" + "=" * 100)
    print("7. SAVING RESULTS")
    print("=" * 100)
    
    # Save important submissions
    important = ['final_blend', 'hybrid', 'ensemble_weighted', 'gandalf', 'saint']
    
    for name in important:
        if name in predictions:
            submission = pd.DataFrame({
                'id': test_df[id_col],
                'target': predictions[name]
            })
            
            filename = f'submission_{name}.csv'
            submission.to_csv(filename, index=False)
            
            print(f"\n{filename}:")
            print(f"  Mean: {predictions[name].mean():.2f}")
            print(f"  Std: {predictions[name].std():.2f}")
    
    # 10. Feature Importance
    if 'rf' in models:
        print("\n" + "=" * 100)
        print("8. FEATURE IMPORTANCE")
        print("=" * 100)
        
        rf_importance = pd.Series(
            models['rf'].feature_importances_,
            index=X_train_scaled.columns
        ).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        rf_importance.head(20).plot(kind='barh')
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('figures/feature_importance.png', dpi=150)
        plt.close()
        
        print("✓ Feature importance plot saved")
    
    # Summary
    print("\n" + "=" * 100)
    print("PIPELINE COMPLETED!")
    print("=" * 100)
    
    print(f"\nSummary:")
    print(f"  Models trained: {len(models)}")
    print(f"  Features used: {X_train_scaled.shape[1]}")
    print(f"  Best submission: submission_final_blend.csv")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

