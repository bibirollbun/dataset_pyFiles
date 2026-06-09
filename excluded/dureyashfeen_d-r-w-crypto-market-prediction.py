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


#%% [markdown]
### ğŸ”� 1. Problem Understanding: Crypto Market Prediction
# The challenge: Predict crypto market movements using noisy, volatile market data
# Key obstacles: 
#   - Extreme volatility (flash crashes, pumps)
#   - 24/7 market with changing regimes
#   - Whale manipulation causing abnormal patterns
#
# My solution: Combine Prophet's temporal anomaly detection with XGBoost's predictive power
#%% [code]
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Install Prophet if needed
try:
    import prophet
except ImportError:
    print("Prophet not found. Installing...")
    !pip install prophet --quiet
    print("Prophet installed.")

# Core imports
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import gc
from collections import defaultdict
import logging

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import joblib

# Prophet
from prophet import Prophet
logging.getLogger('prophet').setLevel(logging.WARNING)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Initializing Temporal Anomaly Adjustment Pipeline...")

#%% [markdown]
### âš™ï¸� 2. Configuration & Data Setup
# Why these features? I selected market microstructure signals (bid/ask quantities) 
# combined with technical indicators showing highest predictive power in EDA
#%% [code]
# File paths (Kaggle environment)
TRAIN_DATA_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
TEST_DATA_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
SUBMISSION_TEMPLATE_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

# Feature selection - market microstructure + key technical indicators
SELECTED_PREDICTORS = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779',
                       'X25', 'X532', 'X520', 'X329', 'X383', 'X751', 'X535', 'X639', 'X596', 'X761',
                       "X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51",
                       "X598", "X385", "X603", "X674", "X415", "X345", "X174", "X178", "X168", "X612",
                       "bid_qty", "ask_qty", "buy_qty", "sell_qty"]

INPUT_COLUMNS = SELECTED_PREDICTORS + ["volume"]
PREDICTION_TARGET = "label"

#%% [markdown]
### ğŸ•µï¸� 3. Temporal Anomaly Detection Engine
# I use Prophet to find time-based anomalies because:
#   - Automatically handles multiple seasonality (intraday, weekly)
#   - Robust to missing data common in crypto
#   - Provides uncertainty intervals for anomaly scoring
#%% [code]
@dataclass
class AnomalyProfile:
    """Stores patterns for feature's temporal anomalies"""
    feature_name: str
    value_segments: List[Tuple[float, float]]  # Anomalous value ranges
    anomaly_scores: Dict[float, float]  # Value â†’ anomaly score
    correction_intensity: Dict[float, float]  # Value â†’ correction strength
    temporal_trends: Dict[str, Any]  # Time patterns

class TemporalAnomalyDetector:
    """Identifies temporal anomalies using Prophet and creates correction rules"""
    
    def __init__(self):
        self.profiles: Dict[str, AnomalyProfile] = {}
    
    def impute_temporal_extremes(self, df: pd.DataFrame, feature: str,
                                 time_index_col: str = '__index_level_0__') -> pd.Series:
        """Replace extreme values using Prophet's forecasts"""
        imputed = df[feature].copy()
        finite_vals = imputed[np.isfinite(imputed)]
        
        if len(finite_vals) < 50: 
            logger.warning(f"  Insufficient data for {feature} imputation. Using median.")
            return imputed.fillna(finite_vals.median() if finite_vals.any() else 0)
        
        # Identify extremes (0.1% and 99.9% tails)
        lower_bound = np.percentile(finite_vals, 0.1)
        upper_bound = np.percentile(finite_vals, 99.9)
        extreme_mask = (imputed < lower_bound) | (imputed > upper_bound) | ~np.isfinite(imputed)
        
        if not extreme_mask.any():
            return imputed  # No extremes found
        
        try:
            # Prepare Prophet data
            ts_df = pd.DataFrame({'ds': df[time_index_col], 'y': imputed})
            ts_df.loc[extreme_mask, 'y'] = np.nan
            ts_df_clean = ts_df.dropna(subset=['y'])
            
            if len(ts_df_clean) < 100:
                return imputed.fillna(finite_vals.median())
            
            # Fit Prophet (optimized for crypto volatility)
            model = Prophet(
                changepoint_prior_scale=0.15, 
                interval_width=0.98,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            model.fit(ts_df_clean)
            
            # Predict and replace extremes
            future = pd.DataFrame({'ds': df[time_index_col]})
            forecast = model.predict(future)
            imputed.loc[extreme_mask] = forecast.loc[extreme_mask, 'yhat'].values
            logger.info(f"  Imputed {extreme_mask.sum()} extremes for {feature}")
            
        except Exception as e:
            logger.error(f"  Prophet imputation failed: {e}. Using median.")
            imputed[extreme_mask] = np.median(finite_vals)
            
        return imputed
        
    def detect_temporal_anomalies(self, df: pd.DataFrame, feature: str,
                                  time_index_col: str = '__index_level_0__') -> Optional[AnomalyProfile]:
        """Identify temporal outliers and create correction profile"""
        logger.info(f"Analyzing {feature} for anomalies...")
        
        ts_df = pd.DataFrame({'ds': df[time_index_col], 'y': df[feature]}).dropna(subset=['y'])
        
        if len(ts_df) < 200:
            logger.warning(f"Insufficient data ({len(ts_df)} points) for {feature}")
            return None
            
        try:
            # Hourly resampling for efficiency
            hourly_ts_df = ts_df.set_index('ds').resample('1H').mean().reset_index().dropna()
            
            if len(hourly_ts_df) < 50:
                return None
            
            # Fit Prophet (tuned for anomaly detection)
            model = Prophet(
                changepoint_prior_scale=0.08,
                interval_width=0.99,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            model.fit(hourly_ts_df)
            forecast = model.predict(hourly_ts_df)
            
            # Detect anomalies using prediction intervals
            anomaly_values = []
            anomaly_scores = {}
            
            for idx, row in ts_df.iterrows():
                nearest = forecast.iloc[np.argmin(np.abs(forecast['ds'] - row['ds']))]
                y_val = row['y']
                lower, upper = nearest['yhat_lower'], nearest['yhat_upper']
                
                score = 0
                if y_val < lower:
                    score = min(1.0, (lower - y_val) / (nearest['yhat'] - lower + 1e-6))
                elif y_val > upper:
                    score = min(1.0, (y_val - upper) / (upper - nearest['yhat'] + 1e-6))
                
                if score > 0.05:
                    anomaly_values.append((y_val, score))
                anomaly_scores[y_val] = score
            
            # Group anomaly values into contiguous ranges
            value_segments = self._group_anomalous_ranges(anomaly_values)
            
            # Create correction mapping
            correction_map = self._derive_adjustment_factors(
                ts_df['y'].values, anomaly_scores, value_segments
            )
            
            # Capture time patterns
            temporal_patterns = self._capture_time_patterns(ts_df, anomaly_scores)
            
            return AnomalyProfile(
                feature_name=feature,
                value_segments=value_segments,
                anomaly_scores=anomaly_scores,
                correction_intensity=correction_map,
                temporal_trends=temporal_patterns
            )
            
        except Exception as e:
            logger.error(f"Prophet analysis failed for {feature}: {e}")
            return None
            
    def _group_anomalous_ranges(self, anomalies: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Group contiguous anomaly values into ranges"""
        if not anomalies: return []
        
        anomalies.sort(key=lambda x: x[0])
        ranges = []
        current_min = current_max = anomalies[0][0]
        
        for val, _ in anomalies[1:]:
            if val <= current_max * 1.005 + 1e-6:
                current_max = val
            else:
                ranges.append((current_min, current_max))
                current_min = current_max = val
                
        ranges.append((current_min, current_max))
        return ranges
        
    def _derive_adjustment_factors(self, values: np.ndarray,
                                  scores: Dict[float, float],
                                  segments: List[Tuple[float, float]]) -> Dict[float, float]:
        """Map values to correction strengths"""
        finite_vals = values[np.isfinite(values)]
        if not finite_vals.any(): return {0.0: 0.0}
        
        # Sample across value range
        value_points = np.linspace(finite_vals.min(), finite_vals.max(), 1000)
        strengths = []
        
        for val in value_points:
            base_strength = 0.0
            
            # Check if in anomaly segment
            for seg_min, seg_max in segments:
                if seg_min <= val <= seg_max:
                    base_strength = max(base_strength, 0.5)
                    break
            
            # Incorporate nearest anomaly score
            closest_score = 0.0
            min_dist = float('inf')
            for known_val, score in scores.items():
                dist = abs(known_val - val)
                if dist < min_dist:
                    min_dist = dist
                    closest_score = score
                    
            if min_dist < (finite_vals.max() - finite_vals.min()) * 0.005:
                base_strength = max(base_strength, closest_score * 0.8)
                
            strengths.append(base_strength)
        
        return dict(zip(value_points, strengths))
        
    def _capture_time_patterns(self, ts_df: pd.DataFrame,
                               scores: Dict[float, float]) -> Dict[str, Any]:
        """Extract hourly/daily anomaly patterns"""
        patterns = {
            'hourly_anomaly_rate': defaultdict(list),
            'daily_anomaly_rate': defaultdict(list),
            'value_percentiles': {}
        }
        
        ts_df['hour'] = ts_df['ds'].dt.hour
        ts_df['day'] = ts_df['ds'].dt.dayofweek
        
        for _, row in ts_df.iterrows():
            score = scores.get(row['y'], 0)
            patterns['hourly_anomaly_rate'][row['hour']].append(score)
            patterns['daily_anomaly_rate'][row['day']].append(score)
            
        # Average scores
        for hour in patterns['hourly_anomaly_rate']:
            patterns['hourly_anomaly_rate'][hour] = np.mean(patterns['hourly_anomaly_rate'][hour])
        for day in patterns['daily_anomaly_rate']:
            patterns['daily_anomaly_rate'][day] = np.mean(patterns['daily_anomaly_rate'][day])
            
        # Value distribution
        patterns['value_percentiles'] = {
            p: np.percentile(ts_df['y'], p) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        
        return dict(patterns)

#%% [markdown]
### ğŸ› ï¸� 4. Data Refinement System
# This is where I combine Prophet's insights with statistical methods:
#   - Uses Isotonic Regression for smooth, monotonic correction curves
#   - Blends temporal and statistical anomaly detection
#   - Memory optimized for large crypto datasets
#%% [code]
class DataRefiner:
    """Applies anomaly corrections using learned patterns"""
    
    def __init__(self):
        self.anomaly_detector = TemporalAnomalyDetector()
        self.anomaly_profiles: Dict[str, AnomalyProfile] = {}
        self.static_bounds: Dict[str, Tuple[float, float]] = {}
        self.correction_regressors: Dict[str, IsotonicRegression] = {}
        
    def fit(self, features: pd.DataFrame, timestamps: Optional[pd.Series] = None):
        """Learn correction rules from data"""
        logger.info("Fitting Data Refiner...")
        
        if timestamps is not None:
            df_with_time = features.copy()
            df_with_time['__index_level_0__'] = timestamps
            
            # Priority features for Prophet processing
            priority_features = ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']
            sample_x = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756']
            
            # Impute extremes
            for feat in priority_features:
                if feat in features.columns:
                    features[feat] = self.anomaly_detector.impute_temporal_extremes(df_with_time, feat)
            
            # Detect anomalies
            for feat in priority_features + sample_x:
                if feat in features.columns:
                    profile = self.anomaly_detector.detect_temporal_anomalies(df_with_time, feat)
                    if profile:
                        self.anomaly_profiles[feat] = profile
        
        # Statistical bounds for all features
        for feat in INPUT_COLUMNS:
            if feat not in features.columns:
                continue
                
            values = features[feat].values
            finite_vals = values[np.isfinite(values)]
            
            if not finite_vals.any():
                self.static_bounds[feat] = (0.0, 0.0)
                continue
                
            # IQR-based bounds
            q1, q3 = np.percentile(finite_vals, [25, 75])
            iqr = q3 - q1
            lower = q1 - 2.5 * iqr
            upper = q3 + 2.5 * iqr
            self.static_bounds[feat] = (lower, upper)
            
            # Build correction mapping
            self._build_combined_correction_map(feat, finite_vals)
            
        return self
        
    def _build_combined_correction_map(self, feature: str, clean_values: np.ndarray):
        """Create smooth correction curve blending Prophet and stats"""
        if not clean_values.any():
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit([0, 1], [0, 0])
            self.correction_regressors[feature] = iso_reg
            return

        # Sample value range
        value_points = np.linspace(clean_values.min(), clean_values.max(), 500)
        strengths = []
        
        for val in value_points:
            # Statistical component
            lower_stat, upper_stat = self.static_bounds[feature]
            if val < lower_stat:
                stat_strength = min(1.0, (lower_stat - val) / (upper_stat - lower_stat + 1e-6))
            elif val > upper_stat:
                stat_strength = min(1.0, (val - upper_stat) / (upper_stat - lower_stat + 1e-6))
            else:
                stat_strength = 0.0
                
            # Prophet component
            prophet_strength = 0.0
            if feature in self.anomaly_profiles:
                profile = self.anomaly_profiles[feature]
                # Check anomaly segments
                for seg_min, seg_max in profile.value_segments:
                    if seg_min <= val <= seg_max:
                        prophet_strength = max(prophet_strength, 0.7)
                        break
                
                # Nearest anomaly score
                closest_score = 0.0
                min_dist = float('inf')
                if profile.anomaly_scores:
                    for known_val, score in profile.anomaly_scores.items():
                        dist = abs(known_val - val)
                        if dist < min_dist:
                            min_dist = dist
                            closest_score = score
                    
                    if min_dist < (clean_values.max() - clean_values.min()) * 0.002:
                        prophet_strength = max(prophet_strength, closest_score * 0.9)
            
            # Combine strengths
            if prophet_strength > 0:
                final_strength = (prophet_strength * 0.7) + (stat_strength * 0.3)
            else:
                final_strength = stat_strength * 0.6
                
            strengths.append(final_strength)
        
        # Fit Isotonic Regression (ensures monotonic correction)
        if value_points.any() and strengths:
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(value_points, strengths)
            self.correction_regressors[feature] = iso_reg
        else:
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit([0, 1], [0, 0])
            self.correction_regressors[feature] = iso_reg
        
    def transform(self, features: pd.DataFrame, intensity: float = 0.5) -> pd.DataFrame:
        """Apply corrections with specified intensity"""
        adjusted = features.copy()
        
        for feature in INPUT_COLUMNS:
            if feature not in features or feature not in self.correction_regressors:
                continue
                
            values = features[feature].values
            finite_mask = np.isfinite(values)
            
            if np.any(finite_mask):
                # Get correction strengths
                strengths = self.correction_regressors[feature].predict(values[finite_mask])
                median_val = np.median(values[finite_mask])
                
                # Apply correction: blend towards median
                adjusted_vals = values[finite_mask] * (1 - strengths * intensity) + \
                                median_val * (strengths * intensity)
                
                adjusted.loc[finite_mask, feature] = adjusted_vals
                
        return adjusted

#%% [markdown]
### ğŸ§¹ 5. Data Cleaning Utilities
# Crypto data is messy! These ensure:
#   - No NaNs/Infs break the pipeline
#   - Efficient memory usage (critical for Kaggle)
#%% [code]
def ensure_numerical_integrity(df: pd.DataFrame, fill_method='median') -> pd.DataFrame:
    """Remove NaNs/Infs using robust methods"""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # Handle infinities
            inf_count = np.sum(np.isinf(df_clean[col]))
            if inf_count > 0:
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaNs
            finite_vals = df_clean[col][np.isfinite(df_clean[col])]
            if finite_vals.empty:
                df_clean[col] = 0
            else:
                fill_val = np.median(finite_vals) if fill_method == 'median' else np.mean(finite_vals)
                df_clean[col] = df_clean[col].fillna(fill_val)
    
    # Final check
    if not df_clean.apply(np.isfinite).all().all():
        df_clean = df_clean.replace([np.inf, -np.inf], 0).fillna(0)
        
    return df_clean

def optimize_memory_footprint(df: pd.DataFrame, name: str):
    """Downcast numeric types to reduce memory usage"""
    logger.info(f"Optimizing memory for {name}...")
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != 'object':
            c_min, c_max = df[col].min(), df[col].max()
            
            if not np.isfinite(c_min) or not np.isfinite(c_max):
                continue
                
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"  Memory reduced: {start_mem:.2f}MB â†’ {end_mem:.2f}MB")
    return df

#%% [markdown]
### ğŸ¤– 6. Crypto Prediction Pipeline
# The core system that ties everything together:
#   - Calibrates anomaly adjustment intensity using time-series CV
#   - Trains XGBoost on refined data
#   - Generates predictions with explainable visualizations
#%% [code]
class CryptoPredictiveFlow:
    """End-to-end prediction system with anomaly adjustment"""
    
    def __init__(self):
        self.data_refiner = DataRefiner()
        self.feature_scaler = RobustScaler()
        self.prediction_model = None
        self.calibrated_intensity = 0.6
        
    def calibrate_adjustment_intensity(self, features: pd.DataFrame, targets: pd.Series,
                                      timestamps: Optional[pd.Series] = None) -> float:
        """Find optimal correction strength using time-series CV"""
        logger.info("Calibrating adjustment intensity...")
        
        self.data_refiner.fit(features, timestamps)
        tscv = TimeSeriesSplit(n_splits=3)
        intensity_levels = [0.0, 0.3, 0.6, 0.8, 1.0]
        results = {}
        
        for intensity in intensity_levels:
            logger.info(f"Testing intensity = {intensity:.1f}")
            
            # Apply correction
            adjusted = self.data_refiner.transform(features, intensity)
            adjusted = ensure_numerical_integrity(adjusted)
            
            # Scale features
            scaled = pd.DataFrame(
                self.feature_scaler.fit_transform(adjusted),
                columns=adjusted.columns
            )
            scaled = ensure_numerical_integrity(scaled)
            
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(scaled)):
                # Split data
                X_train, y_train = scaled.iloc[train_idx], targets.iloc[train_idx]
                X_val, y_val = scaled.iloc[val_idx], targets.iloc[val_idx]
                
                if X_train.empty or X_val.empty:
                    continue
                    
                # Handle NaN targets
                y_train = y_train.fillna(y_train.median())
                y_val = y_val.fillna(y_val.median())
                
                # Train temporary model
                model = XGBRegressor(
                    n_estimators=120,
                    max_depth=7,
                    learning_rate=0.06,
                    subsample=0.75,
                    colsample_bytree=0.6,
                    random_state=42,
                    n_jobs=-1,
                    missing=np.nan
                )
                model.fit(X_train, y_train)
                
                # Evaluate
                preds = model.predict(X_val)
                score = pearsonr(y_val, preds)[0]
                fold_scores.append(0.0 if np.isnan(score) else score)
                logger.info(f"  Fold {fold}: Pearson = {score:.4f}")
            
            if fold_scores:
                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)
                results[intensity] = (mean_score, std_score)
                logger.info(f"  Intensity {intensity:.1f}: {mean_score:.4f} Â± {std_score:.4f}")
        
        # Select best intensity (penalize high variance)
        best_intensity = max(results.keys(), 
                            key=lambda k: results[k][0] - 0.2 * results[k][1]) 
        self.calibrated_intensity = best_intensity
        logger.info(f"Optimal intensity: {best_intensity:.2f}")
        return best_intensity
        
    def fit_pipeline(self, features: pd.DataFrame, targets: pd.Series, 
                     timestamps: Optional[pd.Series] = None):
        """Train full prediction pipeline"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING PREDICTION PIPELINE")
        logger.info("="*80)
        
        # Calibrate and apply correction
        self.calibrate_adjustment_intensity(features, targets, timestamps)
        adjusted = self.data_refiner.transform(features, self.calibrated_intensity)
        adjusted = ensure_numerical_integrity(adjusted)
        
        # Scale features
        scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(adjusted),
            columns=adjusted.columns
        )
        scaled = ensure_numerical_integrity(scaled)
        
        # Train final model
        logger.info("Training final XGBoost model...")
        self.prediction_model = XGBRegressor(
            n_estimators=400,
            max_depth=9,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.7,
            gamma=0.15,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            missing=np.nan
        )
        
        # Handle target NaNs
        targets = targets.fillna(targets.median())
        self.prediction_model.fit(scaled, targets)
        
        # Training performance
        train_preds = self.prediction_model.predict(scaled)
        train_score = pearsonr(targets, train_preds)[0]
        logger.info(f"Training Pearson correlation: {train_score:.4f}")
        
        return self
        
    def generate_predictions(self, test_data: pd.DataFrame) -> np.ndarray:
        """Generate predictions on new data"""
        logger.info("Generating test predictions...")
        
        # Apply learned corrections
        adjusted_test = self.data_refiner.transform(test_data, self.calibrated_intensity)
        adjusted_test = ensure_numerical_integrity(adjusted_test)
        
        # Scale using training scaler
        scaled_test = pd.DataFrame(
            self.feature_scaler.transform(adjusted_test),
            columns=adjusted_test.columns
        )
        scaled_test = ensure_numerical_integrity(scaled_test)
        
        return self.prediction_model.predict(scaled_test)
        
    def visualize_correction_patterns(self, output_path: str = 'anomaly_insights.png'):
        """Visualize anomaly correction profiles"""
        features_to_plot = [f for f in ['volume', 'bid_qty', 'ask_qty', 'X363', 'X321', 'X405'] 
                            if f in self.data_refiner.anomaly_profiles]
        
        if not features_to_plot:
            logger.warning("No anomaly profiles to visualize")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(features_to_plot[:6]):
            profile = self.data_refiner.anomaly_profiles[feature]
            ax = axes[i]
            
            # Plot correction curve
            values = list(profile.correction_intensity.keys())
            strengths = list(profile.correction_intensity.values())
            sorted_idx = np.argsort(values)
            ax.plot(np.array(values)[sorted_idx], np.array(strengths)[sorted_idx], 
                    'b-', alpha=0.7, label='Correction Strength')
            
            # Highlight anomaly segments
            for seg_min, seg_max in profile.value_segments:
                ax.axvspan(seg_min, seg_max, alpha=0.15, color='red',
                           label='Anomaly Zone' if i == 0 else '')
            
            # Add distribution percentiles
            percentiles = profile.temporal_trends['value_percentiles']
            for p, val in percentiles.items():
                if p in [10, 50, 90]:
                    ax.axvline(val, color='gray', ls=':', alpha=0.4,
                               label=f'{p}th %ile' if i == 0 else '')
            
            ax.set_title(f"{feature} Correction Profile")
            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Correction Intensity")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        plt.suptitle("Temporal Anomaly Correction Insights", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        logger.info(f"Saved visualization: {output_path}")

#%% [markdown]
### ğŸš€ 7. Execution Workflow
# This ties everything together for end-to-end execution:
#   1. Load and preprocess data
#   2. Train pipeline
#   3. Generate predictions
#   4. Create insights visualization
#%% [code]
def execute_prediction_flow():
    """End-to-end prediction workflow"""
    logger.info("Loading data...")
    try:
        train_data = pd.read_parquet(TRAIN_DATA_PATH)
        test_data = pd.read_parquet(TEST_DATA_PATH)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return None
    
    # Extract features/targets
    X_train = train_data[INPUT_COLUMNS].copy()
    y_train = train_data[PREDICTION_TARGET].copy()
    X_test = test_data[INPUT_COLUMNS].copy()
    
    # Get timestamps if available
    train_timestamps = None
    if '__index_level_0__' in train_data.columns:
        train_timestamps = train_data['__index_level_0__']
        if not pd.api.types.is_datetime64_any_dtype(train_timestamps):
            train_timestamps = pd.to_datetime(train_timestamps, unit='s')
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Clean data
    X_train = ensure_numerical_integrity(X_train)
    X_test = ensure_numerical_integrity(X_test)
    
    # Optimize memory
    X_train = optimize_memory_footprint(X_train, "Train Features")
    X_test = optimize_memory_footprint(X_test, "Test Features")
    
    # Train pipeline
    pipeline = CryptoPredictiveFlow()
    pipeline.fit_pipeline(X_train, y_train, train_timestamps)
    
    # Visualize insights
    pipeline.visualize_correction_patterns()
    
    # Generate predictions
    predictions = pipeline.generate_predictions(X_test)
    
    # Save submission
    submission = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
    submission['prediction'] = predictions
    submission.to_csv('crypto_predictions.csv', index=False)
    logger.info("Saved predictions to crypto_predictions.csv")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE EXECUTION COMPLETE!")
    logger.info(f"Calibrated intensity: {pipeline.calibrated_intensity:.2f}")
    logger.info("="*80)
    
    return pipeline

#%% [markdown]
### â–¶ï¸� 8. Run the Pipeline!
# Execute the full workflow and show sample predictions
#%% [code]
if __name__ == "__main__":
    crypto_pipeline = execute_prediction_flow()
    
    # Display sample results
    if crypto_pipeline:
        sample_preds = pd.read_csv('crypto_predictions.csv').head()
        print("\nSample predictions:")
        display(sample_preds)

