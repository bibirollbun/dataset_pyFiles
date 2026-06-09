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


# ============================================
# INSTALLATIONS
# ============================================
!pip install -q prophet
!pip install -q koolbox
!pip install -q scikit-learn==1.5.2
!pip install -q autogluon
!pip install -q flaml[automl]
!pip install -q mljar-supervised
!pip install -q h2o
!pip install -q optuna
!pip install -q lightgbm
!pip install -q xgboost
!pip install -q catboost
!pip install -q ngboost
!pip install -q shap
!pip install -q tsfresh

# ============================================
# IMPORTS
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.base import clone, BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.stats import pearsonr, spearmanr
from scipy.signal import savgol_filter
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal
import joblib
import gc
import os
from typing import Dict, List, Tuple, Optional
import shap
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# ============================================
# CONFIGURATION
# ============================================
class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    target = "label"
    n_folds = 5
    seed = 42
    
    # Feature list
    X_FEATURES = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779',
                  'X25', 'X532', 'X520', 'X329', 'X383', 'X751', 'X535', 'X639', 'X596', 'X761',
                  "X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51",
                  "X598", "X385", "X603", "X674", "X415", "X345", "X174", "X178", "X168", "X612",
                  "bid_qty", "ask_qty", "buy_qty", "sell_qty"]
    
    # Advanced configuration
    use_recent_months = 6  # Only use recent months of data
    noise_percentile = 80  # Percentile for noise detection
    min_feature_importance = 0.001  # Minimum feature importance threshold

# ============================================
# UTILITY FUNCTIONS
# ============================================
def _pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def reduce_mem_usage(dataframe, dataset):    
    print(f'Reducing memory usage for: {dataset}')
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype
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
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
    print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
    print(f'--- Decreased memory usage by {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%\n')

    return dataframe

# ============================================
# ADVANCED FEATURE ENGINEERING
# ============================================
def feature_engineering(df):
    # Original features
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])
    
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    # New microstructure features
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    return df

def add_lag_features(df, features, lags=[1, 5, 15, 30, 60]):
    """Add lag features for time series"""
    new_features = []
    for feature in features:
        if feature in df.columns:
            for lag in lags:
                lag_col = f'{feature}_lag_{lag}'
                df[lag_col] = df[feature].shift(lag)
                new_features.append(lag_col)
    return df, new_features

def add_rolling_features(df, features, windows=[5, 15, 30, 60]):
    """Add rolling statistics"""
    new_features = []
    for feature in features:
        if feature in df.columns:
            for window in windows:
                # Rolling mean
                mean_col = f'{feature}_rolling_mean_{window}'
                df[mean_col] = df[feature].rolling(window, min_periods=1).mean()
                new_features.append(mean_col)
                
                # Rolling std
                std_col = f'{feature}_rolling_std_{window}'
                df[std_col] = df[feature].rolling(window, min_periods=1).std()
                new_features.append(std_col)
                
                # Rolling min/max spread
                spread_col = f'{feature}_rolling_spread_{window}'
                rolling_max = df[feature].rolling(window, min_periods=1).max()
                rolling_min = df[feature].rolling(window, min_periods=1).min()
                df[spread_col] = rolling_max - rolling_min
                new_features.append(spread_col)
    
    return df, new_features

# ============================================
# ADVANCED NOISE REDUCTION & FEATURE COMPRESSION
# ============================================
class HierarchicalNoiseReducer(BaseEstimator, TransformerMixin):
    """Hierarchical noise reduction with multiple stages"""
    
    def __init__(self, n_components=50, noise_threshold=0.1, n_clusters=5):
        self.n_components = n_components
        self.noise_threshold = noise_threshold
        self.n_clusters = n_clusters
        self.scalers = {}
        self.reducers = {}
        self.noise_filters = {}
        self.feature_clusters = None
        
    def fit(self, X, y=None):
        X_array = X.values if hasattr(X, 'values') else X
        
        # Stage 1: Identify feature clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        feature_corr = np.corrcoef(X_array.T)
        self.feature_clusters = self.kmeans.fit_predict(feature_corr)
        
        # Stage 2: Apply cluster-specific noise reduction
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.feature_clusters == cluster_id
            X_cluster = X_array[:, cluster_mask]
            
            # Robust scaling per cluster
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            self.scalers[cluster_id] = scaler
            
            # Noise estimation using rolling variance
            noise_levels = []
            for col in range(X_scaled.shape[1]):
                # Use multiple methods to estimate noise
                diff_noise = np.std(np.diff(X_scaled[:, col])) / np.sqrt(2)
                mad_noise = np.median(np.abs(X_scaled[:, col] - np.median(X_scaled[:, col]))) * 1.4826
                combined_noise = (diff_noise + mad_noise) / 2
                noise_levels.append(combined_noise)
            
            noise_levels = np.array(noise_levels)
            noise_threshold = np.percentile(noise_levels, CFG.noise_percentile)
            self.noise_filters[cluster_id] = noise_levels < noise_threshold
            
            # Apply dimensionality reduction to clean features
            clean_features = X_scaled[:, self.noise_filters[cluster_id]]
            if clean_features.shape[1] > 0:
                n_comp = min(self.n_components // self.n_clusters, clean_features.shape[1])
                reducer = PCA(n_components=n_comp, random_state=42)
                reducer.fit(clean_features)
                self.reducers[cluster_id] = reducer
            
        return self
    
    def transform(self, X):
        X_array = X.values if hasattr(X, 'values') else X
        transformed_features = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.feature_clusters == cluster_id
            X_cluster = X_array[:, cluster_mask]
            
            # Scale
            X_scaled = self.scalers[cluster_id].transform(X_cluster)
            
            # Apply noise filter and reduce
            if cluster_id in self.reducers:
                clean_features = X_scaled[:, self.noise_filters[cluster_id]]
                reduced_features = self.reducers[cluster_id].transform(clean_features)
                transformed_features.append(reduced_features)
            
            # Keep some noisy features with smoothing
            noisy_features = X_scaled[:, ~self.noise_filters[cluster_id]]
            if noisy_features.shape[1] > 0:
                # Apply Savitzky-Golay filter for smoothing
                smoothed = np.zeros_like(noisy_features)
                for col in range(noisy_features.shape[1]):
                    try:
                        smoothed[:, col] = savgol_filter(noisy_features[:, col], 
                                                        window_length=min(11, noisy_features.shape[0]//2*2-1), 
                                                        polyorder=3)
                    except:
                        smoothed[:, col] = noisy_features[:, col]
                
                # Take top principal components of smoothed noisy features
                if smoothed.shape[1] > 5:
                    reducer = TruncatedSVD(n_components=5, random_state=42)
                    smoothed = reducer.fit_transform(smoothed)
                transformed_features.append(smoothed)
        
        return np.hstack(transformed_features)

class AdaptiveFeatureSelector(BaseEstimator, TransformerMixin):
    """Adaptive feature selection based on stability and importance"""
    
    def __init__(self, n_features=100, stability_threshold=0.7):
        self.n_features = n_features
        self.stability_threshold = stability_threshold
        self.selected_features = None
        self.feature_scores = None
        
    def fit(self, X, y):
        X_array = X.values if hasattr(X, 'values') else X
        n_samples, n_features = X_array.shape
        
        # Calculate multiple importance metrics
        # 1. Mutual information
        mi_scores = mutual_info_regression(X_array, y, random_state=42)
        
        # 2. Correlation with target
        corr_scores = np.abs([pearsonr(X_array[:, i], y)[0] for i in range(n_features)])
        
        # 3. Feature stability (using bootstrap)
        stability_scores = np.zeros(n_features)
        n_bootstrap = 10
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_array[idx]
            y_boot = y[idx]
            
            # Use simple model for speed
            model = LGBMRegressor(n_estimators=50, num_leaves=31, random_state=42, verbose=-1)
            model.fit(X_boot, y_boot)
            
            importances = model.feature_importances_
            top_features = np.argsort(importances)[-self.n_features:]
            stability_scores[top_features] += 1
        
        stability_scores /= n_bootstrap
        
        # Combine scores
        mi_scores_norm = mi_scores / (mi_scores.max() + 1e-10)
        corr_scores_norm = corr_scores / (corr_scores.max() + 1e-10)
        
        self.feature_scores = (mi_scores_norm + corr_scores_norm + stability_scores) / 3
        
        # Select features above stability threshold and top n_features
        stable_mask = stability_scores >= self.stability_threshold
        stable_indices = np.where(stable_mask)[0]
        
        if len(stable_indices) < self.n_features:
            # Add more features based on combined score
            remaining_indices = np.where(~stable_mask)[0]
            remaining_scores = self.feature_scores[remaining_indices]
            additional_indices = remaining_indices[np.argsort(remaining_scores)[-( self.n_features - len(stable_indices)):]]
            self.selected_features = np.concatenate([stable_indices, additional_indices])
        else:
            # Select top n_features from stable features
            stable_scores = self.feature_scores[stable_indices]
            top_stable_indices = stable_indices[np.argsort(stable_scores)[-self.n_features:]]
            self.selected_features = top_stable_indices
        
        return self
    
    def transform(self, X):
        X_array = X.values if hasattr(X, 'values') else X
        return X_array[:, self.selected_features]

# ============================================
# NGBOOST WITH UNCERTAINTY
# ============================================
class UncertaintyAwareNGBoost(BaseEstimator, RegressorMixin):
    """NGBoost with uncertainty-based sample weighting"""
    
    def __init__(self, n_estimators=500, learning_rate=0.01, minibatch_frac=0.5, 
                 uncertainty_weight=True):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.uncertainty_weight = uncertainty_weight
        self.model = None
        self.uncertainty_threshold = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        # Initial fit to get uncertainty estimates
        self.model = NGBRegressor(
            Dist=Normal,
            n_estimators=100,  # Quick initial fit
            learning_rate=self.learning_rate * 2,
            minibatch_frac=self.minibatch_frac,
            verbose=False,
            random_state=42
        )
        
        self.model.fit(X, y)
        
        if self.uncertainty_weight:
            # Get uncertainty estimates
            y_dist = self.model.pred_dist(X)
            uncertainties = y_dist.std()
            
            # Create sample weights (lower weight for high uncertainty)
            self.uncertainty_threshold = np.percentile(uncertainties, 75)
            sample_weights = 1.0 / (1.0 + uncertainties / self.uncertainty_threshold)
            sample_weights = sample_weights / sample_weights.mean()  # Normalize
        else:
            sample_weights = None
        
        # Final fit with all estimators
        self.model = NGBRegressor(
            Dist=Normal,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            verbose=False,
            random_state=42
        )
        
        if X_val is not None and y_val is not None:
            self.model.fit(X, y, X_val=X_val, Y_val=y_val, sample_weight=sample_weights)
        else:
            self.model.fit(X, y, sample_weight=sample_weights)
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X):
        y_dist = self.model.pred_dist(X)
        return y_dist.mean(), y_dist.std()

# ============================================
# HIERARCHICAL ENSEMBLE
# ============================================
class HierarchicalEnsemble(BaseEstimator, RegressorMixin):
    """Multi-level ensemble with uncertainty weighting"""
    
    def __init__(self, base_models, meta_model=None, use_uncertainty=True):
        self.base_models = base_models
        self.meta_model = meta_model or RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
        self.use_uncertainty = use_uncertainty
        self.level1_models = {}
        self.level2_model = None
        
    def fit(self, X, y, cv=None):
        if cv is None:
            cv = TimeSeriesSplit(n_splits=5)
        
        # Level 1: Train base models with cross-validation
        level1_predictions = {}
        level1_uncertainties = {}
        
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            oof_preds = np.zeros(len(X))
            oof_uncertainty = np.ones(len(X))
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Clone and train model
                model_clone = clone(model)
                
                if hasattr(model_clone, 'predict_with_uncertainty'):
                    model_clone.fit(X_train, y_train, X_val, y_val)
                    preds, uncertainty = model_clone.predict_with_uncertainty(X_val)
                    oof_preds[val_idx] = preds
                    oof_uncertainty[val_idx] = uncertainty
                else:
                    model_clone.fit(X_train, y_train)
                    oof_preds[val_idx] = model_clone.predict(X_val)
                
                self.level1_models[f"{name}_fold{fold}"] = model_clone
            
            level1_predictions[name] = oof_preds
            level1_uncertainties[name] = oof_uncertainty
            
            score = _pearsonr(y, oof_preds)
            print(f"{name} OOF Score: {score:.6f}")
        
        # Level 2: Train meta-model
        X_meta = pd.DataFrame(level1_predictions)
        
        if self.use_uncertainty:
            # Add uncertainty-weighted features
            for name in level1_predictions:
                uncertainty = level1_uncertainties[name]
                weight = 1.0 / (1.0 + uncertainty / uncertainty.mean())
                X_meta[f"{name}_weighted"] = level1_predictions[name] * weight
        
        self.level2_model = clone(self.meta_model)
        self.level2_model.fit(X_meta, y)
        
        # Print meta-model weights
        if hasattr(self.level2_model, 'coef_'):
            print("\nMeta-model weights:")
            for feature, weight in zip(X_meta.columns, self.level2_model.coef_):
                print(f"{feature}: {weight:.4f}")
        
        return self
    
    def predict(self, X):
        level1_predictions = {}
        level1_uncertainties = {}
        
        for name, model in self.base_models.items():
            preds = []
            uncertainties = []
            
            # Average predictions from all folds
            for fold in range(5):  # Assuming 5 folds
                model_key = f"{name}_fold{fold}"
                if model_key in self.level1_models:
                    fold_model = self.level1_models[model_key]
                    
                    if hasattr(fold_model, 'predict_with_uncertainty'):
                        pred, unc = fold_model.predict_with_uncertainty(X)
                        preds.append(pred)
                        uncertainties.append(unc)
                    else:
                        preds.append(fold_model.predict(X))
                        uncertainties.append(np.ones(len(X)))
            
            level1_predictions[name] = np.mean(preds, axis=0)
            level1_uncertainties[name] = np.mean(uncertainties, axis=0)
        
        X_meta = pd.DataFrame(level1_predictions)
        
        if self.use_uncertainty:
            for name in level1_predictions:
                uncertainty = level1_uncertainties[name]
                weight = 1.0 / (1.0 + uncertainty / uncertainty.mean())
                X_meta[f"{name}_weighted"] = level1_predictions[name] * weight
        
        return self.level2_model.predict(X_meta)

# ============================================
# MAIN PIPELINE
# ============================================
def main():
    print("Starting Advanced DRW Crypto Market Prediction Pipeline...")
    print("Focus: Minimizing train-CV gap through noise reduction and hierarchical learning")
    
    # Load data
    print("\n1. Loading data...")
    train = pd.read_parquet(CFG.train_path).reset_index(drop=True)
    test = pd.read_parquet(CFG.test_path).reset_index(drop=True)
    
    # Use only recent data as suggested
    if CFG.use_recent_months > 0:
        total_minutes = CFG.use_recent_months * 30 * 24 * 60  # Approximate
        train = train.tail(total_minutes).reset_index(drop=True)
        print(f"Using only last {CFG.use_recent_months} months of data: {len(train)} samples")
    
    # Select features
    selected_columns = CFG.X_FEATURES + ["volume"]
    train = train[selected_columns + [CFG.target]]
    test = test[selected_columns]
    
    # Add timestamp if missing
    if '__index_level_0__' not in train.columns:
        train['__index_level_0__'] = pd.date_range('2023-03-01', periods=len(train), freq='T')
    if '__index_level_0__' not in test.columns:
        test['__index_level_0__'] = pd.date_range('2024-03-01', periods=len(test), freq='T')
    
    # Apply feature engineering
    print("\n2. Advanced Feature Engineering...")
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    # Add time series features
    important_features = ['volume', 'order_flow_imbalance', 'bid_ask_imbalance', 'liquidity_ratio']
    train, lag_features = add_lag_features(train, important_features)
    test, _ = add_lag_features(test, important_features)
    
    train, rolling_features = add_rolling_features(train, important_features)
    test, _ = add_rolling_features(test, important_features)
    
    # Fill NaN from lag/rolling features
    train = train.fillna(method='ffill').fillna(0)
    test = test.fillna(method='ffill').fillna(0)
    
    # Remove base features and timestamp
    to_remove = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "__index_level_0__"]
    train = train.drop(columns=[col for col in to_remove if col in train.columns])
    test = test.drop(columns=[col for col in to_remove if col in test.columns])
    
    # Reduce memory
    train = reduce_mem_usage(train, "train")
    test = reduce_mem_usage(test, "test")
    
    # Prepare data
    X = train.drop(CFG.target, axis=1)
    y = train[CFG.target]
    X_test = test
    
    print(f"Shape after feature engineering: {X.shape}")
    
    # ============================================
    # HIERARCHICAL NOISE REDUCTION
    # ============================================
    print("\n3. Applying Hierarchical Noise Reduction...")
    noise_reducer = HierarchicalNoiseReducer(n_components=100, noise_threshold=0.15, n_clusters=8)
    X_reduced = pd.DataFrame(noise_reducer.fit_transform(X))
    X_test_reduced = pd.DataFrame(noise_reducer.transform(X_test))
    
    print(f"Shape after noise reduction: {X_reduced.shape}")
    
    # ============================================
    # ADAPTIVE FEATURE SELECTION
    # ============================================
    print("\n4. Adaptive Feature Selection...")
    feature_selector = AdaptiveFeatureSelector(n_features=150, stability_threshold=0.6)
    X_selected = pd.DataFrame(feature_selector.fit_transform(X_reduced, y))
    X_test_selected = pd.DataFrame(feature_selector.transform(X_test_reduced))
    
    print(f"Shape after feature selection: {X_selected.shape}")
    
    # ============================================
    # PREPARE MODELS
    # ============================================
    print("\n5. Preparing Advanced Models...")
    
    # Base models with focus on regularization
    base_models = {
        'NGBoost': UncertaintyAwareNGBoost(
            n_estimators=500,
            learning_rate=0.01,
            minibatch_frac=0.5,
            uncertainty_weight=True
        ),
        
        'LightGBM_Regularized': LGBMRegressor(
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=31,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=50,
            reg_lambda=50,
            min_child_samples=100,
            min_split_gain=0.1,
            random_state=42,
            verbose=-1
        ),
        
        'XGBoost_Regularized': XGBRegressor(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=50,
            reg_lambda=50,
            min_child_weight=100,
            gamma=5,
            random_state=42,
            verbosity=0
        ),
        
        'CatBoost_Regularized': CatBoostRegressor(
            iterations=300,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=50,
            subsample=0.6,
            colsample_bylevel=0.6,
            min_data_in_leaf=100,
            random_seed=42,
            verbose=False
        ),
        
        'Ridge_Strong': Ridge(alpha=100.0, solver='saga', max_iter=10000),
        
        'ElasticNet_Strong': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
    }
    
    # ============================================
    # HIERARCHICAL ENSEMBLE TRAINING
    # ============================================
    print("\n6. Training Hierarchical Ensemble...")
    
    # Use TimeSeriesSplit for better time series validation
    cv = TimeSeriesSplit(n_splits=5, test_size=len(X_selected)//10)
    
    ensemble = HierarchicalEnsemble(
        base_models=base_models,
        meta_model=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5),
        use_uncertainty=True
    )
    
    ensemble.fit(X_selected, y, cv=cv)
    
    # Get predictions
    final_predictions = ensemble.predict(X_test_selected)
    
    # ============================================
    # ANALYZE TRAIN-CV GAP
    # ============================================
    print("\n7. Analyzing Train-CV Gap...")
    
    # Calculate in-sample predictions for gap analysis
    train_predictions = ensemble.predict(X_selected)
    train_score = _pearsonr(y, train_predictions)
    
    # Cross-validation scores
    cv_scores = []
    for train_idx, val_idx in cv.split(X_selected, y):
        X_val = X_selected.iloc[val_idx]
        y_val = y.iloc[val_idx]
        val_pred = ensemble.predict(X_val)
        cv_scores.append(_pearsonr(y_val, val_pred))
    
    cv_score = np.mean(cv_scores)
    train_cv_gap = train_score - cv_score
    
    print(f"\nTrain Score: {train_score:.6f}")
    print(f"CV Score: {cv_score:.6f}")
    print(f"Train-CV Gap: {train_cv_gap:.6f}")
    
    # ============================================
    # POST-PROCESSING FOR STABILITY
    # ============================================
    print("\n8. Post-processing for Stability...")
    
    # Apply slight smoothing to predictions
    window_size = 5
    final_predictions_smoothed = pd.Series(final_predictions).rolling(
        window=window_size, center=True, min_periods=1
    ).mean().values
    
    # Clip extreme predictions based on training distribution
    train_percentiles = np.percentile(y, [1, 99])
    final_predictions_clipped = np.clip(
        final_predictions_smoothed,
        train_percentiles[0],
        train_percentiles[1]
    )
    
    # ============================================
    # SAVE RESULTS
    # ============================================
    print("\n9. Saving Results...")
    
    # Save submission
    sub = pd.read_csv(CFG.sample_sub_path)
    sub["prediction"] = final_predictions_clipped
    sub.to_csv("submission_advanced.csv", index=False)
    print("Submission saved to submission_advanced.csv")
    
    # Save models and preprocessors
    joblib.dump(noise_reducer, "noise_reducer.pkl")
    joblib.dump(feature_selector, "feature_selector.pkl")
    joblib.dump(ensemble, "ensemble_model.pkl")
    
    # Diagnostic plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Feature importance from adaptive selection
    plt.subplot(2, 2, 1)
    feature_scores = feature_selector.feature_scores[feature_selector.selected_features]
    plt.hist(feature_scores, bins=30, alpha=0.7, color='blue')
    plt.title('Selected Feature Scores Distribution')
    plt.xlabel('Feature Score')
    plt.ylabel('Count')
    
    # Plot 2: Prediction distribution
    plt.subplot(2, 2, 2)
    plt.hist(y, bins=50, alpha=0.5, label='Training', color='blue')
    plt.hist(final_predictions_clipped, bins=50, alpha=0.5, label='Test Predictions', color='red')
    plt.title('Prediction Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot 3: Cross-validation scores
    plt.subplot(2, 2, 3)
    plt.plot(cv_scores, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=cv_score, color='r', linestyle='--', label=f'Mean: {cv_score:.6f}')
    plt.title('Cross-Validation Scores by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Train vs CV predictions scatter
    plt.subplot(2, 2, 4)
    sample_size = min(5000, len(y))
    sample_idx = np.random.choice(len(y), sample_size, replace=False)
    plt.scatter(y.iloc[sample_idx], train_predictions[sample_idx], alpha=0.5, s=1)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title(f'Train Predictions (Sample, r={train_score:.6f})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig('advanced_model_diagnostics.png', dpi=300)
    plt.show()
    
    print("\nPipeline completed successfully!")
    print(f"Final Train-CV Gap: {train_cv_gap:.6f}")
    
    return final_predictions_clipped, train_score, cv_score, train_cv_gap

# ============================================
# RUN MAIN PIPELINE
# ============================================
if __name__ == "__main__":
    final_predictions, train_score, cv_score, train_cv_gap = main()

