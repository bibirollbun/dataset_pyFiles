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


import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, RANSACRegressor, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import pearsonr, rankdata
import warnings
warnings.filterwarnings('ignore')
import gc

# ===== Feature Engineering =====
def feature_engineering(df):
    """Enhanced feature engineering with market microstructure focus"""
    # Original features
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    
    # Enhanced market microstructure features
    df['log_volume'] = np.log1p(df['volume'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-8)
    
    # Additional time-aware features
    df['volume_intensity'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['trade_aggressiveness'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
    
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    return df

# ===== Configuration =====
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Baseline features that should NOT be rank transformed (they already work well)
    BASELINE_FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333",
        "bid_qty", "ask_qty"
    ]
    
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    ENHANCED_FOLDS = 5
    RANDOM_STATE = 42

# ===== Selective Rank Transformation =====
class SelectiveRankTransformer:
    """
    Revolutionary approach: Apply rank transformation ONLY to non-baseline features
    while preserving the strong baseline features unchanged.
    """
    
    def __init__(self, baseline_features):
        self.baseline_features = set(baseline_features)
        self.rank_mappings = {}
        self.fitted = False
    
    def _rank_transform_feature(self, values, feature_name, fit=False):
        """Apply rank transformation to a single feature"""
        finite_mask = np.isfinite(values)
        
        if not finite_mask.any():
            return np.full_like(values, 0.5)
        
        finite_vals = values[finite_mask]
        
        if fit:
            # Store unique sorted values for consistent transformation
            unique_vals = np.sort(np.unique(finite_vals))
            self.rank_mappings[feature_name] = unique_vals
        
        if feature_name in self.rank_mappings:
            # Transform using stored mapping
            mapping_vals = self.rank_mappings[feature_name]
            ranks = np.searchsorted(mapping_vals, finite_vals, side='left')
            percentiles = ranks / len(mapping_vals)
            percentiles = np.clip(percentiles, 0, 1)
        else:
            # Fallback to direct ranking
            ranks = rankdata(finite_vals, method='average')
            percentiles = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.array([0.5])
        
        result = np.full_like(values, 0.5)
        result[finite_mask] = percentiles
        
        return result
    
    def fit_transform(self, df, feature_cols):
        """Fit and transform: rank transform only non-baseline features"""
        print(f"   ğŸ�¯ Selective Rank Transform:")
        
        baseline_count = sum(1 for f in feature_cols if f in self.baseline_features)
        rank_count = len(feature_cols) - baseline_count
        
        print(f"      â€¢ Preserving {baseline_count} baseline features (no transform)")
        print(f"      â€¢ Rank transforming {rank_count} additional features")
        
        result_df = df.copy()
        
        for col in feature_cols:
            if col not in self.baseline_features and col in df.columns:
                # Apply rank transformation
                transformed_vals = self._rank_transform_feature(
                    df[col].values, col, fit=True
                )
                result_df[col] = transformed_vals
        
        self.fitted = True
        return result_df
    
    def transform(self, df, feature_cols):
        """Transform new data using fitted parameters"""
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        result_df = df.copy()
        
        for col in feature_cols:
            if col not in self.baseline_features and col in df.columns:
                # Apply rank transformation using fitted mapping
                transformed_vals = self._rank_transform_feature(
                    df[col].values, col, fit=False
                )
                result_df[col] = transformed_vals
        
        return result_df

# ===== Smart Feature Selector =====
class SmartFeatureSelector:
    """Select top features for analysis beyond baseline"""
    
    def __init__(self, baseline_features, max_additional=50):
        self.baseline_features = set(baseline_features)
        self.max_additional = max_additional
        self.selected_additional = []
        self.all_features = []
    
    def fit(self, df, target):
        """Select best additional features beyond baseline"""
        print(f"   ğŸ”� Smart Feature Selection:")
        
        # Get all possible features (engineered features)
        all_features = [col for col in df.columns 
                       if col not in ['label', 'timestamp'] 
                       and col not in self.baseline_features]
        
        print(f"      â€¢ Baseline features: {len(self.baseline_features)}")
        print(f"      â€¢ Additional candidates: {len(all_features)}")
        
        # Calculate correlations for additional features
        correlations = []
        for col in all_features:
            if col in df.columns:
                corr = abs(df[col].corr(target))
                if not np.isnan(corr):
                    correlations.append((col, corr))
        
        # Sort and select top features
        correlations.sort(key=lambda x: x[1], reverse=True)
        self.selected_additional = [col for col, _ in correlations[:self.max_additional]]
        
        # Combine baseline + selected additional
        self.all_features = list(self.baseline_features) + self.selected_additional
        
        print(f"      â€¢ Selected additional: {len(self.selected_additional)}")
        print(f"      â€¢ Total features: {len(self.all_features)}")
        
        return self.all_features
    
    def get_features(self):
        """Get selected features"""
        return self.all_features

# ===== Intelligent Ensemble Manager =====
class IntelligentEnsembleManager:
    """Improved ensemble manager with better model selection"""
    
    def __init__(self):
        self.models = {}
        self.baseline_score = None
    
    def add_model(self, name, oof_preds, test_preds, score, model_type="standard"):
        """Add a model with intelligent tracking"""
        self.models[name] = {
            'oof': oof_preds,
            'test': test_preds,
            'score': score,
            'type': model_type
        }
        
        if self.baseline_score is None:
            self.baseline_score = score
            print(f"   ğŸ“Š Baseline set: {name} (score: {score:.4f})")
        else:
            improvement = (score - self.baseline_score) / self.baseline_score * 100
            status = "âœ…" if score > self.baseline_score else "ğŸ“ˆ" if score > self.baseline_score * 0.9 else "âš ï¸�"
            print(f"   ğŸ“Š {status} {name}: {score:.4f} ({improvement:+.1f}%)")
    
    def get_tree_ensemble(self, train_labels):
        """Get the strong tree ensemble"""
        tree_models = {k: v for k, v in self.models.items() 
                      if any(x in k.lower() for x in ['xgb', 'lgbm', 'tree'])}
        
        if len(tree_models) >= 2:
            # Equal weight ensemble of tree models
            tree_oof = np.mean([model['oof'] for model in tree_models.values()], axis=0)
            tree_test = np.mean([model['test'] for model in tree_models.values()], axis=0)
            tree_score = pearsonr(train_labels, tree_oof)[0]
            return tree_oof, tree_test, tree_score
        elif tree_models:
            # Single tree model
            model = list(tree_models.values())[0]
            return model['oof'], model['test'], model['score']
        else:
            return None, None, 0.0
    
    def get_weighted_ensemble(self, train_labels, min_score_threshold=0.01):
        """Get performance-weighted ensemble"""
        valid_models = {k: v for k, v in self.models.items() 
                       if v['score'] > min_score_threshold and not np.isnan(v['score'])}
        
        if not valid_models:
            return None, None, 0.0, {}
        
        # Calculate weights based on performance
        total_score = sum(model['score'] for model in valid_models.values())
        weights = {k: v['score'] / total_score for k, v in valid_models.items()}
        
        # Create weighted ensemble
        weighted_oof = sum(weights[k] * valid_models[k]['oof'] for k in weights)
        weighted_test = sum(weights[k] * valid_models[k]['test'] for k in weights)
        weighted_score = pearsonr(train_labels, weighted_oof)[0]
        
        return weighted_oof, weighted_test, weighted_score, weights
    
    def get_rank_ensemble(self, train_labels):
        """Get ensemble of rank-transformed models only"""
        rank_models = {k: v for k, v in self.models.items() if 'rank' in k.lower()}
        
        if not rank_models:
            return None, None, 0.0
        
        # Simple average of rank models
        rank_oof = np.mean([model['oof'] for model in rank_models.values()], axis=0)
        rank_test = np.mean([model['test'] for model in rank_models.values()], axis=0)
        rank_score = pearsonr(train_labels, rank_oof)[0]
        
        return rank_oof, rank_test, rank_score

# ===== Model Parameters (CPU Only) =====
def get_baseline_xgb_params():
    """Baseline XGBoost parameters (CPU)"""
    return {
        "tree_method": "hist",
        "device": "cpu",
        "colsample_bylevel": 0.4778,
        "colsample_bynode": 0.3628,
        "colsample_bytree": 0.7107,
        "gamma": 1.7095,
        "learning_rate": 0.02213,
        "max_depth": 20,
        "max_leaves": 12,
        "min_child_weight": 16,
        "n_estimators": 1667,
        "subsample": 0.06567,
        "reg_alpha": 39.3524,
        "reg_lambda": 75.4484,
        "verbosity": 0,
        "random_state": Config.RANDOM_STATE,
        "n_jobs": -1
    }

def get_baseline_lgbm_params():
    """Baseline LightGBM parameters (CPU)"""
    return {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 10,
        "reg_lambda": 10,
        "random_state": Config.RANDOM_STATE,
        "device": "cpu",
        "verbosity": -1,
        "n_jobs": -1
    }

def get_conservative_xgb_params():
    """Conservative XGBoost for rank features"""
    base = get_baseline_xgb_params()
    return {
        **base,
        "learning_rate": 0.015,
        "max_depth": 15,
        "min_child_weight": 25,
        "subsample": 0.08,
        "colsample_bytree": 0.6,
        "reg_alpha": 50,
        "reg_lambda": 100,
        "n_estimators": 1200,
    }

# ===== Utility Functions =====
def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    """Create time decay weights for more recent data importance"""
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def get_model_slices(n_samples: int):
    """Define different data slices for training"""
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples)},
    ]

def train_single_model(X_train, y_train, X_valid, y_valid, X_test, model_type, params, sample_weights=None):
    """Train a single model"""
    if model_type == "xgb":
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, 
                 sample_weight=sample_weights,
                 eval_set=[(X_valid, y_valid)], 
                 verbose=False)
    elif model_type == "lgbm":
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train,
                 sample_weight=sample_weights,
                 eval_set=[(X_valid, y_valid)],
                 callbacks=[])
    elif model_type == "huber":
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)
        
        model = HuberRegressor(**params)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        valid_pred = model.predict(X_valid_scaled)
        test_pred = model.predict(X_test_scaled)
        return valid_pred, test_pred
    elif model_type == "lasso":
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)
        
        model = Lasso(**params)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        valid_pred = model.predict(X_valid_scaled)
        test_pred = model.predict(X_test_scaled)
        return valid_pred, test_pred
    elif model_type == "sklearn_mlp":
        # Feature selection for MLP
        selector = SelectKBest(score_func=f_regression, k=min(30, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_valid_selected = selector.transform(X_valid)
        X_test_selected = selector.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_valid_scaled = scaler.transform(X_valid_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Train MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        
        try:
            mlp.fit(X_train_scaled, y_train)
            valid_pred = mlp.predict(X_valid_scaled)
            test_pred = mlp.predict(X_test_scaled)
            return valid_pred, test_pred
        except:
            return np.zeros(len(y_valid)), np.zeros(X_test.shape[0])
    
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)
    return valid_pred, test_pred

# ===== Data Loading =====
def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    
    # Load with baseline features first
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.BASELINE_FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.BASELINE_FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    print(f"Raw data - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Apply feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Smart feature selection for additional features
    feature_selector = SmartFeatureSelector(Config.BASELINE_FEATURES, max_additional=50)
    selected_features = feature_selector.fit(train_df, train_df[Config.LABEL_COLUMN])
    
    print(f"Enhanced data - Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Selected features: {len(selected_features)}")
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df, selected_features

# ===== Model Training Functions =====
def train_baseline_models(train_df, test_df, features, ensemble_manager):
    """Train baseline models (no rank transformation)"""
    print("\nğŸš€ Training Baseline Models...")
    
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)
    
    # Working models configuration
    models_config = [
        {"name": "xgb_baseline", "type": "xgb", "params": get_baseline_xgb_params()},
        {"name": "lgbm_baseline", "type": "lgbm", "params": get_baseline_lgbm_params()},
        {"name": "huber_baseline", "type": "huber", "params": {"epsilon": 1.5, "alpha": 0.01, "max_iter": 500}},
        {"name": "lasso_baseline", "type": "lasso", "params": {"alpha": 0.001, "max_iter": 1000}},
    ]
    
    for model_config in models_config:
        print(f"\n   Training {model_config['name']}...")
        
        # Initialize prediction storage
        oof_preds = {s["name"]: np.zeros(n_samples) for s in model_slices}
        test_preds = {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        
        full_weights = create_time_decay_weights(n_samples)
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
            X_valid = train_df.iloc[valid_idx][features].values
            y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN].values
            X_test = test_df[features].values
            
            for s in model_slices:
                cutoff = s["cutoff"]
                slice_name = s["name"]
                subset = train_df.iloc[cutoff:].reset_index(drop=True)
                rel_idx = train_idx[train_idx >= cutoff] - cutoff
                
                if len(rel_idx) == 0:
                    continue
                    
                X_train = subset.iloc[rel_idx][features].values
                y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN].values
                sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]
                
                try:
                    valid_pred, test_pred = train_single_model(
                        X_train, y_train, X_valid, y_valid, X_test, 
                        model_config['type'], model_config['params'], sw
                    )
                    
                    # Store OOF predictions
                    mask = valid_idx >= cutoff
                    if mask.any():
                        oof_preds[slice_name][valid_idx[mask]] = valid_pred[mask]
                    
                    if cutoff > 0 and (~mask).any():
                        oof_preds[slice_name][valid_idx[~mask]] = oof_preds["full_data"][valid_idx[~mask]]
                    
                    test_preds[slice_name] += test_pred
                    
                except Exception as e:
                    print(f"      Error in {slice_name}: {str(e)}")
                    continue
        
        # Normalize test predictions
        for slice_name in test_preds:
            test_preds[slice_name] /= Config.N_FOLDS
        
        # Calculate ensemble of slices
        final_oof = np.mean(list(oof_preds.values()), axis=0)
        final_test = np.mean(list(test_preds.values()), axis=0)
        final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]
        
        # Add to ensemble manager
        ensemble_manager.add_model(model_config['name'], final_oof, final_test, final_score, "baseline")

def train_rank_models(train_df, test_df, features, ensemble_manager):
    """Train models with selective rank transformation"""
    print("\nğŸ�¯ Training Rank-Transformed Models...")
    
    # Apply selective rank transformation
    rank_transformer = SelectiveRankTransformer(Config.BASELINE_FEATURES)
    train_rank = rank_transformer.fit_transform(train_df, features)
    test_rank = rank_transformer.transform(test_df, features)
    
    # Rank-specific models (more conservative)
    rank_models_config = [
        {"name": "xgb_rank", "type": "xgb", "params": get_conservative_xgb_params()},
        {"name": "lgbm_rank", "type": "lgbm", "params": get_baseline_lgbm_params()},
        {"name": "mlp_rank", "type": "sklearn_mlp", "params": {}},
    ]
    
    kf = KFold(n_splits=Config.ENHANCED_FOLDS, shuffle=False)
    n_samples = len(train_df)
    
    for model_config in rank_models_config:
        print(f"\n   Training {model_config['name']} with rank features...")
        
        model_oof = np.zeros(n_samples)
        model_test = np.zeros(len(test_df))
        fold_scores = []
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
            X_train = train_rank.iloc[train_idx][features].values
            y_train = train_rank.iloc[train_idx][Config.LABEL_COLUMN].values
            X_valid = train_rank.iloc[valid_idx][features].values
            y_valid = train_rank.iloc[valid_idx][Config.LABEL_COLUMN].values
            X_test = test_rank[features].values
            
            # Time decay weights
            sw = create_time_decay_weights(len(train_idx))
            
            try:
                valid_pred, test_pred = train_single_model(
                    X_train, y_train, X_valid, y_valid, X_test, 
                    model_config['type'], model_config['params'], sw
                )
                
                model_oof[valid_idx] = valid_pred
                model_test += test_pred
                
                fold_score = pearsonr(y_valid, valid_pred)[0] if len(np.unique(valid_pred)) > 1 else 0
                fold_scores.append(fold_score)
                
            except Exception as e:
                print(f"      Error in fold {fold}: {str(e)}")
                fold_scores.append(0.0)
        
        # Normalize test predictions
        model_test /= Config.ENHANCED_FOLDS
        
        # Calculate final score
        final_score = pearsonr(train_df[Config.LABEL_COLUMN], model_oof)[0]
        
        print(f"      CV scores: {fold_scores}")
        print(f"      Final score: {final_score:.4f}")
        
        # Add to ensemble manager
        ensemble_manager.add_model(model_config['name'], model_oof, model_test, final_score, "rank")

def train_combined_model(train_df, test_df, features, ensemble_manager):
    """Train combined model: Baseline + Top rank-transformed features"""
    print("\nğŸš€ Training Combined Model (Baseline + Top Rank Features)...")
    
    # Apply rank transformation
    rank_transformer = SelectiveRankTransformer(Config.BASELINE_FEATURES)
    train_rank = rank_transformer.fit_transform(train_df, features)
    test_rank = rank_transformer.transform(test_df, features)
    
    # Select top additional features (beyond baseline)
    baseline_features = [f for f in Config.BASELINE_FEATURES if f in train_df.columns]
    additional_features = [f for f in features if f not in Config.BASELINE_FEATURES]
    
    # Calculate correlations for additional features and select top 20
    correlations = []
    for col in additional_features:
        if col in train_rank.columns:
            corr = abs(train_rank[col].corr(train_df[Config.LABEL_COLUMN]))
            if not np.isnan(corr):
                correlations.append((col, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_additional = [col for col, _ in correlations[:20]]
    
    print(f"   Combined features: {len(baseline_features)} baseline + {len(top_additional)} rank-transformed")
    
    # Train combined XGBoost model
    kf = KFold(n_splits=Config.ENHANCED_FOLDS, shuffle=False)
    n_samples = len(train_df)
    
    model_oof = np.zeros(n_samples)
    model_test = np.zeros(len(test_df))
    fold_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        # Combine baseline (original) + rank-transformed additional features
        X_train_baseline = train_df.iloc[train_idx][baseline_features].values
        X_train_rank = train_rank.iloc[train_idx][top_additional].values
        X_train = np.hstack([X_train_baseline, X_train_rank]) if top_additional else X_train_baseline
        
        X_valid_baseline = train_df.iloc[valid_idx][baseline_features].values
        X_valid_rank = train_rank.iloc[valid_idx][top_additional].values
        X_valid = np.hstack([X_valid_baseline, X_valid_rank]) if top_additional else X_valid_baseline
        
        X_test_baseline = test_df[baseline_features].values
        X_test_rank = test_rank[top_additional].values
        X_test = np.hstack([X_test_baseline, X_test_rank]) if top_additional else X_test_baseline
        
        y_train = train_df.iloc[train_idx][Config.LABEL_COLUMN].values
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN].values
        
        # Time decay weights
        sw = create_time_decay_weights(len(train_idx))
        
        try:
            valid_pred, test_pred = train_single_model(
                X_train, y_train, X_valid, y_valid, X_test,
                "xgb", get_conservative_xgb_params(), sw
            )
            
            model_oof[valid_idx] = valid_pred
            model_test += test_pred
            
            fold_score = pearsonr(y_valid, valid_pred)[0] if len(np.unique(valid_pred)) > 1 else 0
            fold_scores.append(fold_score)
            
            print(f"      Fold {fold} score: {fold_score:.4f}")
            
        except Exception as e:
            print(f"      Error in fold {fold}: {str(e)}")
            fold_scores.append(0.0)
    
    # Normalize test predictions
    model_test /= Config.ENHANCED_FOLDS
    
    # Calculate final score
    final_score = pearsonr(train_df[Config.LABEL_COLUMN], model_oof)[0]
    
    print(f"   ğŸ“Š Combined model CV scores: {fold_scores}")
    print(f"   ğŸ“Š Combined model final score: {final_score:.4f}")
    
    # Add to ensemble manager
    ensemble_manager.add_model("combined_rank", model_oof, model_test, final_score, "combined")
    
    return final_score

# ===== Main Execution =====
def main():
    """Main execution pipeline"""
    print("ğŸ�¯ Clean Ensemble with Selective Rank Transformation")
    print("=" * 60)
    print("ğŸ“‹ Approach:")
    print("   â€¢ Fixed all device and component issues")
    print("   â€¢ Added selective rank transformation")
    print("   â€¢ Intelligent ensemble management")
    print("   â€¢ Multiple submission strategies")
    print("=" * 60)
    
    # Load data
    train_df, test_df, submission_df, features = load_data()
    
    # Target analysis
    target = train_df[Config.LABEL_COLUMN]
    print(f"\nğŸ“Š Target Analysis:")
    print(f"   Mean: {target.mean():.4f}")
    print(f"   Std:  {target.std():.4f}")
    print(f"   Range: [{target.min():.2f}, {target.max():.2f}]")
    
    # Initialize ensemble manager
    ensemble_manager = IntelligentEnsembleManager()
    
    # Stage 1: Train baseline models
    train_baseline_models(train_df, test_df, features, ensemble_manager)
    
    # Stage 2: Train rank-transformed models
    train_rank_models(train_df, test_df, features, ensemble_manager)
    
    # Stage 3: Train combined model
    combined_score = train_combined_model(train_df, test_df, features, ensemble_manager)
    
    # Create multiple submissions
    print("\nğŸ“� Creating Multiple Submission Strategies...")
    
    train_labels = train_df[Config.LABEL_COLUMN]
    
    # 1. Tree Ensemble (XGBoost + LightGBM)
    tree_oof, tree_test, tree_score = ensemble_manager.get_tree_ensemble(train_labels)
    if tree_test is not None:
        submission_tree = submission_df.copy()
        submission_tree["prediction"] = tree_test
        submission_tree.to_csv("submission_tree_ensemble.csv", index=False)
        print(f"   ğŸ“Š Tree Ensemble: {tree_score:.4f}")
    
    # 2. Weighted Performance Ensemble
    weighted_result = ensemble_manager.get_weighted_ensemble(train_labels)
    if weighted_result[0] is not None:
        weighted_oof, weighted_test, weighted_score, weights = weighted_result
        submission_weighted = submission_df.copy()
        submission_weighted["prediction"] = weighted_test
        submission_weighted.to_csv("submission_weighted_ensemble.csv", index=False)
        print(f"   ğŸ“Š Weighted Ensemble: {weighted_score:.4f}")
        print(f"      Top weights: {', '.join([f'{k}: {v:.2f}' for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]])}")
    
    # 3. Rank Models Only
    rank_oof, rank_test, rank_score = ensemble_manager.get_rank_ensemble(train_labels)
    if rank_test is not None:
        submission_rank = submission_df.copy()
        submission_rank["prediction"] = rank_test
        submission_rank.to_csv("submission_rank_only.csv", index=False)
        print(f"   ğŸ“Š Rank Ensemble: {rank_score:.4f}")
    
    # 4. Combined Model Only (Expected Best)
    if "combined_rank" in ensemble_manager.models:
        submission_combined = submission_df.copy()
        submission_combined["prediction"] = ensemble_manager.models["combined_rank"]["test"]
        submission_combined.to_csv("submission_combined_rank.csv", index=False)
        print(f"   ğŸ“Š Combined Model: {combined_score:.4f}")
    
    # 5. XGBoost Baseline Only
    if "xgb_baseline" in ensemble_manager.models:
        submission_baseline = submission_df.copy()
        submission_baseline["prediction"] = ensemble_manager.models["xgb_baseline"]["test"]
        submission_baseline.to_csv("submission_xgb_baseline.csv", index=False)
        baseline_score = ensemble_manager.models["xgb_baseline"]["score"]
        print(f"   ğŸ“Š XGBoost Baseline: {baseline_score:.4f}")
    
    # Summary and Analysis
    print("\nğŸ�† Final Results Summary:")
    print("=" * 50)
    
    print("\nModel Performance:")
    for name, model in sorted(ensemble_manager.models.items(), key=lambda x: x[1]['score'], reverse=True):
        print(f"   {name:20s}: {model['score']:.4f}")
    
    print(f"\nFiles Created:")
    print(f"   â€¢ submission_tree_ensemble.csv")
    print(f"   â€¢ submission_weighted_ensemble.csv") 
    print(f"   â€¢ submission_rank_only.csv")
    print(f"   â€¢ submission_combined_rank.csv (Expected Best)")
    print(f"   â€¢ submission_xgb_baseline.csv")
    
    # Expected improvement analysis
    if "combined_rank" in ensemble_manager.models and "xgb_baseline" in ensemble_manager.models:
        baseline = ensemble_manager.models["xgb_baseline"]["score"]
        combined = ensemble_manager.models["combined_rank"]["score"]
        improvement = (combined - baseline) / baseline * 100
        
        print(f"\nğŸ“ˆ Selective Rank Transformation Impact:")
        print(f"   Baseline Score:    {baseline:.4f}")
        print(f"   Combined Score:    {combined:.4f}")
        print(f"   Improvement:       {improvement:+.1f}%")
        print(f"   Expected (research): +6.2%")
        
        if improvement >= 5.0:
            print(f"   ğŸ�‰ SUCCESS! Achieved target improvement!")
        elif improvement >= 2.0:
            print(f"   âœ… GOOD! Meaningful improvement achieved!")
        elif improvement > 0:
            print(f"   ğŸ“ˆ PROGRESS! Some improvement achieved!")
        else:
            print(f"   âš ï¸�  Room for optimization...")
    
    print(f"\nğŸ”¬ Recommendation: Try submission_combined_rank.csv first!")
    
    # Clean up memory
    gc.collect()

if __name__ == "__main__":
    main()

