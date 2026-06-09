import os
import sys
import time
import gc
import warnings
import re
import subprocess
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from scipy.optimize import nnls
import itertools

# Core ML libraries
import lightgbm as lgb
import xgboost as xgb

# Optional libraries with fallback handling
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Using fallback models.")

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise RuntimeError("Please install optuna: pip install optuna")

try:
    from cir_model import CenteredIsotonicRegression
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cir_model", "--quiet"])
        from cir_model import CenteredIsotonicRegression
    except Exception:
        raise RuntimeError("Please install cir_model: pip install cir_model")


class Config:
    """Central configuration for the entire pipeline."""
    
    # Paths
    TRAIN_PATH = '/kaggle/input/playground-series-s5e9/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s5e9/test.csv'
    WORKDIR = '/kaggle/working/'
    
    # Target variable
    TARGET = 'BeatsPerMinute'
    
    # Feature engineering
    TOP_K_FOR_PAIRWISE = 12
    PAIRWISE_CAP = 66
    TOP_FINAL_FEATURES = 200
    
    # Model training
    RANDOM_SEED = 42
    N_MODELS_SELECTION = 3  # For feature selection phase
    N_FOLDS = 10
    NUM_REPEATS = 3  # For final ensemble
    
    # Optimization
    OPTUNA_TRIALS = 500
    MIN_WEIGHT = 0.01
    L2_REG = 1e-3
    
    # Analysis
    CORR_THRESHOLD = 0.95
    CUMULATIVE_THRESHOLDS = [0.90, 0.95, 0.99]
    TOP_N_FOR_BAR = 60
    
    # Performance
    VERBOSE = True
    FAST_MODE = False  # Set True for debugging with smaller trees
    USE_GPU = False  # Auto-detected if False
    N_JOBS = 6


def print_versions():
    """Print versions of ML libraries for debugging."""
    libraries = {
        'lightgbm': lgb,
        'xgboost': xgb,
        'catboost': cb if CATBOOST_AVAILABLE else None
    }
    
    for name, lib in libraries.items():
        if lib:
            try:
                print(f"{name}.__version__: {lib.__version__}")
            except AttributeError:
                print(f"{name}: installed (version unknown)")
        else:
            print(f"{name}: not installed")


def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage by downcasting numeric types."""
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col].dtype):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(df[col].dtype):
            df[col] = df[col].astype('float32')
    return df


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))



class GPUDetector:
    """Detect GPU availability for different ML libraries."""
    
    @staticmethod
    def detect_all(prefer_gpu: bool = False) -> Dict[str, bool]:
        """Detect GPU support for all libraries."""
        gpu_support = {
            'lgb': GPUDetector._detect_lgb(),
            'xgb': GPUDetector._detect_xgb(),
            'cb': GPUDetector._detect_cb() if CATBOOST_AVAILABLE else False
        }
        
        if not prefer_gpu:
            gpu_support = {k: False for k in gpu_support}
        
        return gpu_support
    
    @staticmethod
    def _detect_lgb() -> bool:
        """Detect LightGBM GPU support."""
        try:
            X_test = np.random.rand(20, 4).astype('float32')
            y_test = np.random.rand(20).astype('float32')
            
            model = lgb.LGBMRegressor(device='gpu', n_estimators=1, verbosity=-1)
            model.fit(X_test, y_test, 
                     eval_set=[(X_test, y_test)],
                     callbacks=[lgb.early_stopping(1), lgb.log_evaluation(0)])
            return True
        except Exception:
            return False
    
    @staticmethod
    def _detect_xgb() -> bool:
        """Detect XGBoost GPU support."""
        try:
            X_test = np.random.rand(20, 4).astype('float32')
            y_test = np.random.rand(20).astype('float32')
            
            model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=1, verbosity=0)
            model.fit(X_test, y_test, 
                     eval_set=[(X_test, y_test)],
                     early_stopping_rounds=1, verbose=False)
            return True
        except Exception:
            return False
    
    @staticmethod
    def _detect_cb() -> bool:
        """Detect CatBoost GPU support."""
        if not CATBOOST_AVAILABLE:
            return False
        try:
            X_test = np.random.rand(20, 4).astype('float32')
            y_test = np.random.rand(20).astype('float32')
            
            model = cb.CatBoostRegressor(task_type='GPU', iterations=1, verbose=False)
            model.fit(X_test, y_test)
            return True
        except Exception:
            return False



class FeatureEngineer:
    """Advanced feature engineering for BPM prediction."""
    
    @staticmethod
    def create_advanced_features(train_df: pd.DataFrame, 
                                test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create advanced features for both train and test sets."""
        tr = train_df.copy()
        te = test_df.copy()
        
        for df in (tr, te):
            # Duration features
            if 'TrackDurationMs' in df.columns:
                df['duration_minutes'] = df['TrackDurationMs'] / 60000.0
                df['duration_seconds'] = df['TrackDurationMs'] / 1000.0
                df['is_short_track'] = (df['duration_minutes'] < 2.5).astype(int)
            
            # Loudness features
            if 'AudioLoudness' in df.columns:
                df['loudness_normalized'] = (df['AudioLoudness'] + 30) / 25
                df['loudness_positive'] = np.maximum(df['AudioLoudness'] + 60, 0)
                df['loudness_category'] = pd.cut(
                    df['AudioLoudness'],
                    bins=[-np.inf, -20, -10, -5, np.inf],
                    labels=[0, 1, 2, 3]
                ).astype(int)
            
            # Electronic score
            if 'AcousticQuality' in df.columns:
                df['electronic_score'] = 1 - df['AcousticQuality']
            
            # Core feature transformations
            core_features = ['Energy', 'RhythmScore', 'MoodScore', 'VocalContent',
                           'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood']
            core_features = [c for c in core_features if c in df.columns]
            
            for col in core_features:
                clipped = np.clip(df[col].fillna(0), 0, 10)
                df[f'{col}_log'] = np.log1p(np.clip(df[col].fillna(0), 0, None))
                df[f'{col}_sqrt'] = np.sqrt(np.clip(df[col].fillna(0), 0, None))
                df[f'{col}_reciprocal'] = 1.0 / (df[col].fillna(0) + 1e-6)
                df[f'{col}_sq'] = clipped ** 2
                df[f'{col}_cu'] = clipped ** 3
            
            # Interaction features
            if set(['Energy', 'RhythmScore']).issubset(df.columns):
                df['energy_rhythm_product'] = df['Energy'] * df['RhythmScore']
                df['energy_minus_rhythm'] = df['Energy'] - df['RhythmScore']
                df['energy_divided_by_rhythm'] = df['Energy'] / (df['RhythmScore'] + 1e-6)
                df['energy_rhythm_centroid'] = (df['Energy'] + df['RhythmScore']) / 2.0
                df['audio_bandwidth'] = np.abs(df['Energy'] - df['RhythmScore'])
            
            if 'MoodScore' in df.columns and 'Energy' in df.columns:
                df['mood_energy_product'] = df['MoodScore'] * df['Energy']
            
            if 'VocalContent' in df.columns and 'AcousticQuality' in df.columns:
                df['vocal_acoustic_product'] = df['VocalContent'] * df['AcousticQuality']
                df['vocal_divided_by_acoustic'] = df['VocalContent'] / (df['AcousticQuality'] + 1e-6)
            
            # Tempo score
            if 'Energy' in df.columns and 'RhythmScore' in df.columns:
                df['fast_tempo_score'] = ((df['Energy'] > 0.7).astype(int) + 
                                         (df['RhythmScore'] > 0.7).astype(int))
            
            # Statistical features
            if len(core_features) >= 2:
                arr = np.vstack([df[c].fillna(0).values for c in core_features]).T
                df['feature_mean'] = arr.mean(axis=1)
                df['feature_std'] = arr.std(axis=1)
        
        return tr, te
    
    @staticmethod
    def generate_pairwise_features(X: pd.DataFrame, 
                                  top_numeric: List[str],
                                  pairwise_cap: int = 66) -> pd.DataFrame:
        """Generate pairwise features and binning."""
        X_new = X.copy()
        
        # Polynomial features for top numeric columns
        for col in top_numeric:
            a = X_new[col].astype('float32').fillna(0.0)
            X_new[f'{col}_sq'] = (a * a).astype('float32')
            X_new[f'{col}_sqrt'] = np.sqrt(np.abs(a)).astype('float32')
            
            if (a >= 0).all():
                X_new[f'{col}_log1p'] = np.log1p(a).astype('float32')
            else:
                X_new[f'{col}_log1p_shift'] = np.log1p(a - a.min() + 1e-6).astype('float32')
            
            X_new[f'{col}_recip'] = (1.0 / (a + 1e-6)).astype('float32')
        
        # Pairwise interactions
        pairs = list(itertools.combinations(top_numeric, 2))[:pairwise_cap]
        for (c1, c2) in pairs:
            a = X_new[c1].astype('float32').fillna(0.0)
            b = X_new[c2].astype('float32').fillna(0.0)
            X_new[f'{c1}_x_{c2}'] = (a * b).astype('float32')
            X_new[f'{c1}_div_{c2}'] = (a / (b + 1e-6)).astype('float32')
            X_new[f'{c1}_minus_{c2}'] = (a - b).astype('float32')
        
        # Binning features
        for col in top_numeric:
            try:
                X_new[f'{col}_quartile'] = pd.cut(X_new[col], bins=4, 
                                                  labels=False, 
                                                  include_lowest=True).fillna(0).astype('int8')
                X_new[f'{col}_decile'] = pd.cut(X_new[col], bins=10, 
                                               labels=False, 
                                               include_lowest=True).fillna(0).astype('int8')
            except Exception:
                # Use quantile-based cutting if regular cutting fails
                X_new[f'{col}_quartile'] = pd.qcut(X_new[col].rank(method='first'), 
                                                   q=4, labels=False, 
                                                   duplicates='drop').astype('float').fillna(0).astype('int8')
                X_new[f'{col}_decile'] = pd.qcut(X_new[col].rank(method='first'), 
                                                q=10, labels=False, 
                                                duplicates='drop').astype('float').fillna(0).astype('int8')
        
        return X_new


class FeatureSelector:
    """Feature importance calculation and selection."""
    
    def __init__(self, config: Config, gpu_support: Dict[str, bool]):
        self.config = config
        self.gpu_support = gpu_support
    
    def quick_robust_rank(self, X: pd.DataFrame, y: pd.Series, 
                         seed: int, rounds: int = 400, 
                         early: int = 40) -> Tuple[pd.DataFrame, Any]:
        """
        Train multiple models for robust feature importance ranking.
        Returns importance DataFrame and a representative model.
        """
        Xf = X.copy().fillna(0)
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xf, y, test_size=0.15, random_state=seed
        )
        
        feature_names = Xf.columns.tolist()
        imp_accum = pd.DataFrame({'feature': feature_names, 'sum_gain': 0.0})
        model_count = 0
        
        # LightGBM
        lgb_params = self._get_lgb_params(seed, rounds)
        try:
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_tr, y_tr, 
                         eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(early), lgb.log_evaluation(0)])
            
            gains = lgb_model.booster_.feature_importance(importance_type='gain')
            total = gains.sum() if gains.sum() > 0 else 1.0
            imp_accum['sum_gain'] += gains / float(total)
            model_count += 1
        except Exception as e:
            print(f"LightGBM training failed: {e}")
        
        # XGBoost
        if self._should_use_xgb():
            xgb_params = self._get_xgb_params(seed, rounds)
            try:
                xgb_model = xgb.XGBRegressor(**xgb_params)
                xgb_model.fit(X_tr, y_tr, 
                             eval_set=[(X_val, y_val)],
                             early_stopping_rounds=early, verbose=False)
                
                gains_xgb = self._extract_xgb_gains(xgb_model, feature_names)
                total = gains_xgb.sum() if gains_xgb.sum() > 0 else 1.0
                imp_accum['sum_gain'] += gains_xgb / float(total)
                model_count += 1
            except Exception as e:
                print(f"XGBoost training failed: {e}")
        
        # CatBoost
        if CATBOOST_AVAILABLE and self.gpu_support.get('cb', False):
            cb_params = self._get_cb_params(seed, rounds)
            try:
                cb_model = cb.CatBoostRegressor(**cb_params)
                cb_model.fit(X_tr, y_tr, 
                           eval_set=(X_val, y_val),
                           use_best_model=True, verbose=False)
                
                gains_cb = np.array(
                    cb_model.get_feature_importance(type='PredictionValuesChange'), 
                    dtype=float
                )
                if gains_cb.sum() > 0 and len(gains_cb) == len(feature_names):
                    imp_accum['sum_gain'] += gains_cb / float(gains_cb.sum())
                    model_count += 1
            except Exception as e:
                print(f"CatBoost training failed: {e}")
        
        if model_count == 0:
            raise RuntimeError("No models trained successfully")
        
        imp_accum['avg_gain'] = imp_accum['sum_gain'] / float(model_count)
        imp_df = (imp_accum[['feature', 'avg_gain']]
                 .rename(columns={'avg_gain': 'gain'})
                 .sort_values('gain', ascending=False)
                 .reset_index(drop=True))
        
        # Train representative model on full data
        rep_model = lgb.LGBMRegressor(**lgb_params)
        rep_model.fit(pd.concat([X_tr, X_val], axis=0), 
                     pd.concat([y_tr, y_val], axis=0),
                     callbacks=[lgb.log_evaluation(0)])
        
        return imp_df, rep_model
    
    def _get_lgb_params(self, seed: int, rounds: int) -> Dict:
        """Get LightGBM parameters."""
        params = {
            'n_estimators': rounds,
            'learning_rate': 0.03,
            'num_leaves': 128,
            'max_depth': -1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.5,
            'min_child_samples': 20,
            'n_jobs': self.config.N_JOBS,
            'verbosity': -1,
            'random_state': seed
        }
        if self.gpu_support.get('lgb', False):
            params['device'] = 'gpu'
        return params
    
    def _get_xgb_params(self, seed: int, rounds: int) -> Dict:
        """Get XGBoost parameters."""
        params = {
            'n_estimators': rounds,
            'learning_rate': 0.03,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbosity': 0,
            'n_jobs': self.config.N_JOBS,
            'random_state': seed
        }
        if self.gpu_support.get('xgb', False):
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'
        return params
    
    def _get_cb_params(self, seed: int, rounds: int) -> Dict:
        """Get CatBoost parameters."""
        params = {
            'iterations': rounds,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'verbose': False,
            'random_seed': seed
        }
        if self.gpu_support.get('cb', False):
            params['task_type'] = 'GPU'
        return params
    
    def _should_use_xgb(self) -> bool:
        """Check if XGBoost should be used."""
        return True  # Always try XGBoost
    
    def _extract_xgb_gains(self, model: xgb.XGBRegressor, 
                          feature_names: List[str]) -> np.ndarray:
        """Extract feature gains from XGBoost model."""
        booster = model.get_booster()
        score_dict = booster.get_score(importance_type='gain')
        gains = np.zeros(len(feature_names), dtype=float)
        
        for key, value in score_dict.items():
            idx = None
            
            # Try to parse feature index
            if isinstance(key, str) and re.fullmatch(r'f\d+', key):
                try:
                    idx = int(key[1:])
                except Exception:
                    digits = re.findall(r'\d+', key)
                    idx = int(digits[0]) if digits else None
            elif key in feature_names:
                idx = feature_names.index(key)
            else:
                digits = re.findall(r'\d+', key)
                idx = int(digits[0]) if digits else None
            
            if idx is not None and 0 <= idx < len(gains):
                gains[idx] = value
        
        return gains



class ImportanceAnalyzer:
    """Analyze and visualize feature importances."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def analyze(self, imp_df: pd.DataFrame, X_train: pd.DataFrame = None) -> Dict:
        """
        Comprehensive importance analysis.
        Returns dictionary with analysis results.
        """
        results = {}
        
        # Basic statistics
        results['total_features'] = len(imp_df)
        results['zero_importance'] = (imp_df['importance'] == 0.0).sum()
        results['positive_importance'] = (imp_df['importance'] > 0.0).sum()
        
        # Cumulative importance analysis
        results['cumulative_thresholds'] = self._analyze_cumulative(imp_df)
        
        # Recommended feature count
        results['recommended_k'] = results['cumulative_thresholds'].get(0.95, 
                                   results['cumulative_thresholds'].get(0.90, 200))
        
        # Correlation-based pruning if data available
        if X_train is not None:
            results['pruned_features'] = self._greedy_corr_prune(
                X_train, imp_df, self.config.CORR_THRESHOLD
            )
        else:
            results['pruned_features'] = imp_df['feature'].head(
                results['recommended_k']
            ).tolist()
        
        return results
    
    def visualize(self, imp_df: pd.DataFrame, output_prefix: str):
        """Create visualization plots."""
        self._plot_top_bar(imp_df, output_prefix + "_topbar.png")
        self._plot_cumulative(imp_df, output_prefix + "_cumulative.png")
        self._plot_histogram(imp_df, output_prefix + "_hist.png")
    
    def _analyze_cumulative(self, imp_df: pd.DataFrame) -> Dict[float, int]:
        """Analyze cumulative importance thresholds."""
        vals = imp_df['importance'].values
        total = vals.sum() if vals.sum() > 0 else 1.0
        cum = np.cumsum(vals) / total
        
        results = {}
        for threshold in self.config.CUMULATIVE_THRESHOLDS:
            idx = np.searchsorted(cum, threshold) + 1
            results[threshold] = idx
        
        return results
    
    def _plot_top_bar(self, imp_df: pd.DataFrame, outpath: str):
        """Plot top N features by importance."""
        top = imp_df.head(self.config.TOP_N_FOR_BAR).iloc[::-1]
        
        plt.figure(figsize=(10, max(3, self.config.TOP_N_FOR_BAR * 0.12)))
        plt.barh(top['feature'], top['importance'])
        plt.xlabel("Aggregated importance (mean gain)")
        plt.title(f"Top {self.config.TOP_N_FOR_BAR} features by importance")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    
    def _plot_cumulative(self, imp_df: pd.DataFrame, outpath: str):
        """Plot cumulative importance curve."""
        vals = imp_df['importance'].values
        total = vals.sum() if vals.sum() > 0 else 1.0
        cum = np.cumsum(vals) / total
        ranks = np.arange(1, len(vals) + 1)
        
        plt.figure(figsize=(8, 4))
        plt.plot(ranks, cum, linewidth=2)
        
        for threshold in self.config.CUMULATIVE_THRESHOLDS:
            idx = np.searchsorted(cum, threshold) + 1
            plt.axvline(idx, color='gray', linestyle='--', linewidth=1)
            plt.text(idx + 1, threshold - 0.02, 
                    f"{idx} features → {int(threshold*100)}%", 
                    va='center', fontsize=9)
        
        plt.xlabel("Feature rank")
        plt.ylabel("Cumulative share of importance")
        plt.title("Cumulative importance by feature rank")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    
    def _plot_histogram(self, imp_df: pd.DataFrame, outpath: str):
        """Plot importance histogram."""
        plt.figure(figsize=(6, 3))
        plt.hist(imp_df['importance'].values, bins=80)
        plt.xlabel("Importance (gain)")
        plt.title("Histogram of feature importances")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    
    def _greedy_corr_prune(self, X: pd.DataFrame, imp_df: pd.DataFrame, 
                          threshold: float) -> List[str]:
        """Greedy correlation-based feature pruning."""
        cols_ordered = imp_df['feature'].tolist()
        kept = []
        
        for feature in cols_ordered:
            if feature not in X.columns:
                continue
            
            drop_flag = False
            for kept_feature in kept:
                corr = X[feature].corr(X[kept_feature])
                if pd.isna(corr):
                    corr = 0.0
                if abs(corr) >= threshold:
                    drop_flag = True
                    break
            
            if not drop_flag:
                kept.append(feature)
        
        return kept


class ModelFactory:
    """Factory for creating diverse ensemble models."""
    
    def __init__(self, config: Config, gpu_support: Dict[str, bool]):
        self.config = config
        self.gpu_support = gpu_support
    
    def create_ensemble_models(self, seed: int = None) -> Dict[str, Any]:
        """Create 20 diverse models for ensemble."""
        if seed is None:
            seed = self.config.RANDOM_SEED
        
        models = {}
        
        # Define model configurations
        small = 300 if self.config.FAST_MODE else 800
        med = 600 if self.config.FAST_MODE else 1000
        large = 900 if self.config.FAST_MODE else 1400
        
        # XGBoost variants (5 models)
        models.update(self._create_xgb_models(seed, small, med, large))
        
        # LightGBM variants (7 models)
        models.update(self._create_lgb_models(seed, small, med, large))
        
        # CatBoost variants (5 models)
        if CATBOOST_AVAILABLE:
            models.update(self._create_cb_models(seed, small, med, large))
        else:
            models.update(self._create_fallback_models(seed))
        
        # HistGradientBoosting (1 model)
        models['HGB'] = HistGradientBoostingRegressor(
            max_iter=600 if self.config.FAST_MODE else 1000,
            learning_rate=0.03,
            max_depth=12,
            random_state=seed,
            early_stopping=True
        )
        
        # Additional models to reach 20
        models['XGB_extra'] = xgb.XGBRegressor(
            n_estimators=med, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            verbosity=0, random_state=seed,
            **self._get_xgb_gpu_params()
        )
        
        models['LGB_alt2'] = lgb.LGBMRegressor(
            n_estimators=med, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.75, num_leaves=48,
            n_jobs=-1, random_state=seed, verbosity=-1,
            device='gpu' if self.gpu_support.get('lgb', False) else 'cpu'
        )
        
        return models
    
    def _create_xgb_models(self, seed: int, small: int, med: int, 
                          large: int) -> Dict[str, Any]:
        """Create XGBoost model variants."""
        xgb_common = {
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbosity': 0,
            'random_state': seed,
            **self._get_xgb_gpu_params()
        }
        
        return {
            'XGB_shallow': xgb.XGBRegressor(
                n_estimators=small, max_depth=3, learning_rate=0.06, **xgb_common
            ),
            'XGB_mid': xgb.XGBRegressor(
                n_estimators=med, max_depth=6, learning_rate=0.04, **xgb_common
            ),
            'XGB_deep': xgb.XGBRegressor(
                n_estimators=med, max_depth=9, learning_rate=0.03, **xgb_common
            ),
            'XGB_regularized': xgb.XGBRegressor(
                n_estimators=large, max_depth=6, learning_rate=0.025,
                reg_alpha=2.0, reg_lambda=3.0,
                subsample=0.85, colsample_bytree=0.85,
                verbosity=0, random_state=seed,
                **self._get_xgb_gpu_params()
            )
        }
    
    def _create_lgb_models(self, seed: int, small: int, med: int, 
                          large: int) -> Dict[str, Any]:
        """Create LightGBM model variants."""
        lgb_common = {
            'n_jobs': -1,
            'random_state': seed,
            'verbosity': -1,
            'device': 'gpu' if self.gpu_support.get('lgb', False) else 'cpu'
        }
        
        return {
            'LGB_fast': lgb.LGBMRegressor(
                n_estimators=small, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, num_leaves=31,
                **lgb_common
            ),
            'LGB_complex': lgb.LGBMRegressor(
                n_estimators=med, max_depth=8, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8, num_leaves=100,
                **lgb_common
            ),
            'LGB_boosting': lgb.LGBMRegressor(
                n_estimators=large, max_depth=6, learning_rate=0.02,
                subsample=0.85, colsample_bytree=0.85, num_leaves=60,
                **lgb_common
            ),
            'LGB_large_leaves': lgb.LGBMRegressor(
                n_estimators=med, max_depth=-1, learning_rate=0.03,
                num_leaves=256, feature_fraction=0.6,
                bagging_fraction=0.8, bagging_freq=5,
                **lgb_common
            ),
            'LGB_rf': lgb.LGBMRegressor(
                boosting_type='rf', n_estimators=small,
                subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, num_leaves=64,
                learning_rate=0.1, **lgb_common
            ),
            'LGB_goss': lgb.LGBMRegressor(
                boosting_type='goss', n_estimators=med,
                learning_rate=0.03, num_leaves=80,
                **lgb_common
            )
        }
    
    def _create_cb_models(self, seed: int, small: int, med: int, 
                         large: int) -> Dict[str, Any]:
        """Create CatBoost model variants."""
        cb_common = {
            'verbose': False,
            'random_seed': seed
        }
        if self.gpu_support.get('cb', False):
            cb_common['task_type'] = 'GPU'
        
        return {
            'CB_cons': cb.CatBoostRegressor(
                iterations=800 if not self.config.FAST_MODE else small,
                depth=4, learning_rate=0.05, l2_leaf_reg=3,
                **cb_common
            ),
            'CB_mid': cb.CatBoostRegressor(
                iterations=900 if not self.config.FAST_MODE else small,
                depth=6, learning_rate=0.045, l2_leaf_reg=2,
                **cb_common
            ),
            'CB_aggr': cb.CatBoostRegressor(
                iterations=700 if not self.config.FAST_MODE else small,
                depth=5, learning_rate=0.06, l2_leaf_reg=1,
                **cb_common
            ),
            'CB_bal': cb.CatBoostRegressor(
                iterations=1000 if not self.config.FAST_MODE else small,
                depth=5, learning_rate=0.03, l2_leaf_reg=2.5,
                **cb_common
            ),
            'CB_alt': cb.CatBoostRegressor(
                iterations=med, depth=5, learning_rate=0.04,
                l2_leaf_reg=1.5, **cb_common
            )
        }
    
    def _create_fallback_models(self, seed: int) -> Dict[str, Any]:
        """Create fallback models when CatBoost is not available."""
        return {
            'CB_cons': ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
            'CB_mid': ExtraTreesRegressor(n_estimators=120, random_state=seed+1, n_jobs=-1),
            'CB_aggr': ExtraTreesRegressor(n_estimators=100, random_state=seed+2, n_jobs=-1),
            'CB_bal': ExtraTreesRegressor(n_estimators=120, random_state=seed+3, n_jobs=-1),
            'CB_alt': ExtraTreesRegressor(n_estimators=110, random_state=seed+4, n_jobs=-1)
        }
    
    def _get_xgb_gpu_params(self) -> Dict:
        """Get XGBoost GPU parameters if available."""
        if self.gpu_support.get('xgb', False):
            return {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
        return {}


class EnsembleTrainer:
    """Train and optimize ensemble models."""
    
    def __init__(self, config: Config, model_factory: ModelFactory):
        self.config = config
        self.model_factory = model_factory
    
    def train_with_cv(self, X: pd.DataFrame, y: pd.Series, 
                     X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train ensemble with repeated k-fold cross-validation.
        Returns OOF and test predictions.
        """
        n_rows = X.shape[0]
        models = self.model_factory.create_ensemble_models()
        model_names = list(models.keys())
        n_models = len(model_names)
        
        oof_accum = np.zeros((n_rows, n_models), dtype='float64')
        test_accum = np.zeros((X_test.shape[0], n_models), dtype='float64')
        
        for repeat in range(self.config.NUM_REPEATS):
            rep_seed = self.config.RANDOM_SEED + repeat * 1001
            print(f"\n=== REPEAT {repeat+1}/{self.config.NUM_REPEATS} (seed={rep_seed}) ===")
            
            kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=rep_seed)
            oof_rep = np.zeros((n_rows, n_models), dtype='float64')
            test_rep = np.zeros((X_test.shape[0], n_models), dtype='float64')
            
            for mi, (name, base_model) in enumerate(models.items()):
                print(f"\nTraining model [{mi+1}/{n_models}]: {name}")
                oof_preds = np.zeros(n_rows, dtype='float64')
                test_preds_fold = np.zeros((X_test.shape[0], self.config.N_FOLDS), dtype='float64')
                
                for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
                    print(f"  Fold {fold+1}/{self.config.N_FOLDS}...", end=" ")
                    
                    X_tr = X.iloc[tr_idx].values
                    y_tr = y.iloc[tr_idx].values
                    X_val = X.iloc[val_idx].values
                    y_val = y.iloc[val_idx].values
                    
                    # Clone model for this fold
                    model = self._clone_model(base_model)
                    
                    # Fit model
                    self._fit_model(model, X_tr, y_tr, X_val, y_val, X.columns)
                    
                    # Predictions
                    oof_preds[val_idx] = self._predict(model, X_val, X.columns)
                    test_preds_fold[:, fold] = self._predict(model, X_test.values, X_test.columns)
                    
                    del model
                    gc.collect()
                    print("done")
                
                test_rep[:, mi] = test_preds_fold.mean(axis=1)
                oof_rep[:, mi] = oof_preds
                print(f"  {name} OOF RMSE: {rmse(y.values, oof_preds):.6f}")
            
            oof_accum += oof_rep
            test_accum += test_rep
        
        # Average across repeats
        oof_matrix = oof_accum / float(self.config.NUM_REPEATS)
        test_matrix = test_accum / float(self.config.NUM_REPEATS)
        
        return oof_matrix, test_matrix, model_names
    
    def optimize_weights(self, oof_matrix: np.ndarray, y: np.ndarray) -> Dict:
        """
        Optimize ensemble weights using multiple methods.
        Returns dictionary with different weight sets.
        """
        weights = {}
        
        # Optuna optimization
        print("\nOptimizing weights with Optuna...")
        weights['optuna'], optuna_score = self._optuna_optimize(oof_matrix, y)
        print(f"Optuna best score: {optuna_score:.6f}")
        
        # NNLS optimization
        weights['nnls'], _ = nnls(oof_matrix, y)
        weights['nnls'] = weights['nnls'] / (weights['nnls'].sum() + 1e-12)
        print(f"NNLS OOF RMSE: {rmse(y, oof_matrix.dot(weights['nnls'])):.6f}")
        
        # BayesianRidge meta-learning
        print("\nTraining BayesianRidge meta-model...")
        meta = BayesianRidge()
        meta.fit(oof_matrix, y)
        weights['meta_model'] = meta
        
        return weights
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model for training."""
        try:
            return model.__class__(**model.get_params())
        except Exception:
            from sklearn.base import clone
            try:
                return clone(model)
            except Exception:
                return model
    
    def _fit_model(self, model: Any, X_tr: np.ndarray, y_tr: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray, columns: List[str]):
        """Fit a model with appropriate API."""
        try:
            if isinstance(model, lgb.LGBMRegressor):
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
            elif CATBOOST_AVAILABLE and isinstance(model, cb.CatBoostRegressor):
                model.fit(pd.DataFrame(X_tr, columns=columns), y_tr,
                         eval_set=(pd.DataFrame(X_val, columns=columns), y_val),
                         early_stopping_rounds=80, verbose=False)
            elif isinstance(model, xgb.XGBRegressor):
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         early_stopping_rounds=80, verbose=False)
            else:
                model.fit(X_tr, y_tr)
        except Exception:
            model.fit(X_tr, y_tr)
    
    def _predict(self, model: Any, X: np.ndarray, columns: List[str]) -> np.ndarray:
        """Make predictions with a model."""
        try:
            return model.predict(X)
        except Exception:
            if CATBOOST_AVAILABLE and isinstance(model, cb.CatBoostRegressor):
                return model.predict(pd.DataFrame(X, columns=columns))
            return np.zeros(X.shape[0])
    
    def _optuna_optimize(self, oof_matrix: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize weights using Optuna."""
        n = oof_matrix.shape[1]
        
        def objective(trial):
            raw = np.array([trial.suggest_float(f"w_{i}", self.config.MIN_WEIGHT, 1.0) 
                          for i in range(n)])
            w = raw / raw.sum()
            pred = oof_matrix.dot(w)
            return rmse(y, pred) + self.config.L2_REG * np.sum(w**2)
        
        study = optuna.create_study(direction="minimize", 
                                   sampler=TPESampler(seed=self.config.RANDOM_SEED))
        study.optimize(objective, n_trials=self.config.OPTUNA_TRIALS, 
                      show_progress_bar=True)
        
        best_raw = np.array([study.best_trial.params[f"w_{i}"] for i in range(n)])
        best_w = best_raw / best_raw.sum()
        
        return best_w, study.best_value


class BPMPipeline:
    """Complete pipeline for BPM prediction."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.gpu_support = None
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = None
        self.importance_analyzer = ImportanceAnalyzer(self.config)
        self.model_factory = None
        self.ensemble_trainer = None
    
    def run(self):
        """Execute the complete pipeline."""
        start_time = time.time()
        
        print("="*80)
        print("BPM PREDICTION PIPELINE")
        print("="*80)
        
        # Initialize
        self._initialize()
        
        # Stage 1: Feature Engineering & Selection
        print("\n" + "="*80)
        print("STAGE 1: FEATURE ENGINEERING & SELECTION")
        print("="*80)
        
        X_final, X_test_final, y, feature_importances = self._run_feature_engineering()
        
        # Stage 2: Importance Analysis & Pruning
        print("\n" + "="*80)
        print("STAGE 2: IMPORTANCE ANALYSIS & PRUNING")
        print("="*80)
        
        X_pruned, X_test_pruned, selected_features = self._run_importance_analysis(
            X_final, X_test_final, feature_importances
        )
        
        # Stage 3: Ensemble Training
        print("\n" + "="*80)
        print("STAGE 3: ENSEMBLE TRAINING")
        print("="*80)
        
        oof_matrix, test_matrix, model_names = self._run_ensemble_training(
            X_pruned, y, X_test_pruned
        )
        
        # Stage 4: Weight Optimization & Calibration
        print("\n" + "="*80)
        print("STAGE 4: WEIGHT OPTIMIZATION & CALIBRATION")
        print("="*80)
        
        submissions = self._run_optimization_and_calibration(
            oof_matrix, test_matrix, y, model_names
        )
        
        # Save submissions
        self._save_submissions(submissions)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETED IN {elapsed/60:.2f} MINUTES")
        print("="*80)
    
    def _initialize(self):
        """Initialize components."""
        print("\nInitializing components...")
        print_versions()
        
        # Detect GPU support
        self.gpu_support = GPUDetector.detect_all(self.config.USE_GPU)
        print(f"GPU Support: {self.gpu_support}")
        
        # Initialize components
        self.feature_selector = FeatureSelector(self.config, self.gpu_support)
        self.model_factory = ModelFactory(self.config, self.gpu_support)
        self.ensemble_trainer = EnsembleTrainer(self.config, self.model_factory)
    
    def _run_feature_engineering(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
        """Run feature engineering and selection."""
        # Load data
        print("\nLoading data...")
        train = pd.read_csv(self.config.TRAIN_PATH).reset_index(drop=False)
        test = pd.read_csv(self.config.TEST_PATH).reset_index(drop=False)
        
        if 'id' not in train.columns:
            train.rename(columns={'index': 'id'}, inplace=True)
        if 'id' not in test.columns:
            test.rename(columns={'index': 'id'}, inplace=True)
        
        # Advanced feature engineering
        print("Creating advanced features...")
        train_fe, test_fe = self.feature_engineer.create_advanced_features(train, test)
        
        # Prepare features
        exclude = ['id', self.config.TARGET]
        feature_cols = [c for c in train_fe.columns 
                       if c not in exclude and c in test_fe.columns]
        
        X_raw = train_fe[feature_cols].copy()
        X_test_raw = test_fe[feature_cols].copy()
        y = train_fe[self.config.TARGET].copy().reset_index(drop=True)
        
        # Clean and downcast
        for df in [X_raw, X_test_raw]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            for col in df.columns:
                if df[col].isnull().any():
                    med = X_raw[col].median()
                    df[col].fillna(med, inplace=True)
        
        X_raw = downcast_df(X_raw)
        X_test_raw = downcast_df(X_test_raw)
        
        # Feature importance ranking
        print("Computing feature importances...")
        imp_base, _ = self.feature_selector.quick_robust_rank(
            X_raw, y, seed=self.config.RANDOM_SEED
        )
        
        # Generate pairwise features
        numeric_feats = X_raw.select_dtypes(include=[np.number]).columns.tolist()
        top_k_candidates = [f for f in imp_base['feature'].tolist() 
                          if f in numeric_feats][:self.config.TOP_K_FOR_PAIRWISE]
        
        print(f"Generating pairwise features from top {len(top_k_candidates)} features...")
        X_aug = self.feature_engineer.generate_pairwise_features(
            X_raw, top_k_candidates, self.config.PAIRWISE_CAP
        )
        X_test_aug = self.feature_engineer.generate_pairwise_features(
            X_test_raw, top_k_candidates, self.config.PAIRWISE_CAP
        )
        
        # Scale features
        print("Scaling features...")
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_aug.values.astype('float32')),
            columns=X_aug.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_aug.values.astype('float32')),
            columns=X_test_aug.columns
        )
        
        # Train LGB models for final feature selection
        print("Running final feature selection...")
        all_importances = []
        
        for m in range(self.config.N_MODELS_SELECTION):
            seed_val = self.config.RANDOM_SEED + m * 101
            kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=seed_val)
            
            for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_tr = X_scaled.iloc[tr_idx]
                X_val = X_scaled.iloc[val_idx]
                y_tr = y.iloc[tr_idx]
                y_val = y.iloc[val_idx]
                
                model = lgb.LGBMRegressor(
                    n_estimators=4000,
                    learning_rate=0.03,
                    num_leaves=128,
                    feature_fraction=0.8,
                    bagging_fraction=0.8,
                    random_state=seed_val,
                    n_jobs=self.config.N_JOBS,
                    verbosity=-1,
                    device='gpu' if self.gpu_support.get('lgb', False) else 'cpu'
                )
                
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
                
                gains = model.booster_.feature_importance(importance_type='gain')
                imp_df = pd.DataFrame({
                    'feature': X_scaled.columns,
                    'importance': gains,
                    'fold': fold,
                    'model_seed': seed_val
                })
                all_importances.append(imp_df)
                
                del model
                gc.collect()
        
        # Aggregate importances
        imp_all = pd.concat(all_importances, axis=0)
        imp_mean = (imp_all.groupby('feature')['importance']
                   .mean()
                   .sort_values(ascending=False)
                   .reset_index())
        
        # Select top features
        final_features = imp_mean['feature'].head(self.config.TOP_FINAL_FEATURES).tolist()
        X_final = X_scaled[final_features].copy()
        X_test_final = X_test_scaled[final_features].copy()
        
        # Save intermediate results
        os.makedirs(self.config.WORKDIR, exist_ok=True)
        X_final.to_csv(os.path.join(self.config.WORKDIR, 'X_final_scaled.csv'), index=False)
        X_test_final.to_csv(os.path.join(self.config.WORKDIR, 'X_test_final_scaled.csv'), index=False)
        imp_mean.to_csv(os.path.join(self.config.WORKDIR, 'aggregated_feature_importances.csv'), index=False)
        
        return X_final, X_test_final, y, imp_mean
    
    def _run_importance_analysis(self, X: pd.DataFrame, X_test: pd.DataFrame, 
                                imp_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Run importance analysis and pruning."""
        # Analyze importances
        analysis_results = self.importance_analyzer.analyze(imp_df, X)
        
        print(f"Total features: {analysis_results['total_features']}")
        print(f"Recommended features (95% cumulative): {analysis_results['recommended_k']}")
        print(f"Features after correlation pruning: {len(analysis_results['pruned_features'])}")
        
        # Visualize
        self.importance_analyzer.visualize(
            imp_df, 
            os.path.join(self.config.WORKDIR, 'importance')
        )
        
        # Create pruned datasets
        selected_features = analysis_results['pruned_features']
        X_pruned = X[selected_features].copy()
        X_test_pruned = X_test[selected_features].copy()
        
        # Save pruned data
        X_pruned.to_csv(os.path.join(self.config.WORKDIR, 'pruned_X_train.csv'), index=False)
        X_test_pruned.to_csv(os.path.join(self.config.WORKDIR, 'pruned_X_test.csv'), index=False)
        pd.Series(selected_features).to_csv(
            os.path.join(self.config.WORKDIR, 'pruned_feature_list.csv'), 
            index=False, header=False
        )
        
        return X_pruned, X_test_pruned, selected_features
    
    def _run_ensemble_training(self, X: pd.DataFrame, y: pd.Series, 
                              X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Train ensemble models."""
        oof_matrix, test_matrix, model_names = self.ensemble_trainer.train_with_cv(
            X, y, X_test
        )
        
        # Save matrices
        pd.DataFrame(oof_matrix, columns=model_names).to_csv(
            os.path.join(self.config.WORKDIR, 'oof_matrix.csv'), index=False
        )
        pd.DataFrame(test_matrix, columns=model_names).to_csv(
            os.path.join(self.config.WORKDIR, 'test_matrix.csv'), index=False
        )
        
        return oof_matrix, test_matrix, model_names
    
    def _run_optimization_and_calibration(self, oof_matrix: np.ndarray, 
                                         test_matrix: np.ndarray,
                                         y: np.ndarray, 
                                         model_names: List[str]) -> Dict:
        """Optimize weights and calibrate predictions."""
        # Optimize weights
        weights = self.ensemble_trainer.optimize_weights(oof_matrix, y.values)
        
        # Generate predictions
        submissions = {}
        
        # Optuna weighted
        oof_optuna = oof_matrix.dot(weights['optuna'])
        test_optuna = test_matrix.dot(weights['optuna'])
        
        # Calibrate with CIR
        cir_optuna = CenteredIsotonicRegression().fit(oof_optuna, y.values)
        test_optuna_cal = cir_optuna.transform(test_optuna)
        submissions['optuna'] = test_optuna_cal
        
        print(f"Optuna OOF RMSE (calibrated): {rmse(y.values, cir_optuna.transform(oof_optuna)):.6f}")
        
        # BayesianRidge meta
        meta = weights['meta_model']
        oof_meta = meta.predict(oof_matrix)
        test_meta = meta.predict(test_matrix)
        
        cir_meta = CenteredIsotonicRegression().fit(oof_meta, y.values)
        test_meta_cal = cir_meta.transform(test_meta)
        submissions['bayesianridge'] = test_meta_cal
        
        print(f"BayesianRidge OOF RMSE (calibrated): {rmse(y.values, cir_meta.transform(oof_meta)):.6f}")
        
        # NNLS (uncalibrated)
        test_nnls = test_matrix.dot(weights['nnls'])
        submissions['nnls'] = test_nnls
        
        return submissions
    
    def _save_submissions(self, submissions: Dict):
        """Save submission files."""
        # Load test IDs from the original test file
        test_id = None
        
        # First try to load from the original test path
        if os.path.exists(self.config.TEST_PATH):
            try:
                test_df = pd.read_csv(self.config.TEST_PATH)
                if 'id' in test_df.columns:
                    test_id = test_df['id'].values
                    print(f"Loaded test IDs from {self.config.TEST_PATH}")
                    print(f"Test ID range: {test_id[0]} to {test_id[-1]}")
            except Exception as e:
                print(f"Warning: Could not load test IDs from {self.config.TEST_PATH}: {e}")
        
        # Fallback to test.csv in working directory
        if test_id is None:
            test_path_workdir = os.path.join(self.config.WORKDIR, 'test.csv')
            if os.path.exists(test_path_workdir):
                try:
                    test_df = pd.read_csv(test_path_workdir)
                    if 'id' in test_df.columns:
                        test_id = test_df['id'].values
                        print(f"Loaded test IDs from {test_path_workdir}")
                        print(f"Test ID range: {test_id[0]} to {test_id[-1]}")
                except Exception as e:
                    print(f"Warning: Could not load test IDs from {test_path_workdir}: {e}")
        
        # Last resort - generate IDs starting from where train ends
        if test_id is None:
            print("Warning: Could not load test IDs from file. Generating IDs...")
            # Try to get the last train ID to continue from there
            if os.path.exists(self.config.TRAIN_PATH):
                try:
                    train_df = pd.read_csv(self.config.TRAIN_PATH, usecols=['id'])
                    last_train_id = train_df['id'].max()
                    test_id = np.arange(last_train_id + 1, 
                                       last_train_id + 1 + len(submissions['optuna']))
                    print(f"Generated test IDs starting from {test_id[0]}")
                except Exception:
                    # If all else fails, start from 0
                    test_id = np.arange(len(submissions['optuna']))
                    print("Generated test IDs starting from 0")
            else:
                test_id = np.arange(len(submissions['optuna']))
        
        # Save submissions
        for name, predictions in submissions.items():
            sub = pd.DataFrame({
                'id': test_id,
                self.config.TARGET: predictions
            })
            filename = f'submission_{name}.csv'
            filepath = os.path.join(self.config.WORKDIR, filename)
            sub.to_csv(filepath, index=False)
            print(f"Saved: {filename} (shape: {sub.shape})")


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = BPMPipeline()
    pipeline.run()




