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
from scipy.stats import pearsonr, rankdata, boxcox
from scipy.special import lambertw
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import warnings
warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    FEATURES = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333"
    ]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42

XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
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

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS}
]

# =========================
# Helper Functions for Infinity Handling
# =========================
def clip_extreme_values(x, percentile_range=(0.01, 99.99)):
    """Clip extreme values to specified percentile range"""
    finite_mask = np.isfinite(x)
    if not finite_mask.any():
        return x
    
    finite_values = x[finite_mask]
    lower = np.percentile(finite_values, percentile_range[0])
    upper = np.percentile(finite_values, percentile_range[1])
    
    x_clipped = x.copy()
    x_clipped = np.clip(x_clipped, lower, upper)
    return x_clipped

def handle_infinities(x, method='clip'):
    """Replace infinite values with finite extremes"""
    if not np.any(~np.isfinite(x)):
        return x
    
    x_clean = x.copy()
    finite_mask = np.isfinite(x)
    
    if finite_mask.any():
        finite_values = x[finite_mask]
        
        if method == 'clip':
            # Replace with extreme percentiles
            min_val = np.percentile(finite_values, 0.1)
            max_val = np.percentile(finite_values, 99.9)
            range_val = max_val - min_val
            
            x_clean[x == float('-inf')] = min_val - 0.01 * range_val
            x_clean[x == float('inf')] = max_val + 0.01 * range_val
            x_clean[np.isnan(x)] = np.median(finite_values)
        
        elif method == 'median':
            # Replace with median
            median_val = np.median(finite_values)
            x_clean[~finite_mask] = median_val
    
    return x_clean

def reduce_mem_usage(dataframe, dataset, verbose=True):    
    """Reduces memory usage by converting columns to more efficient data types"""
    if verbose:
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
    if verbose:
        print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
        print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
        print(f'--- Decreased memory usage by {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%\n')

    return dataframe

# =========================
# Transformation Functions with Robust Infinity Handling
# =========================
def robust_rank_transform(x, method='average'):
    """Rank transformation - extremely robust to outliers and noise"""
    # Handle infinities first
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    ranks = np.zeros_like(x, dtype=float)
    
    if finite_mask.any():
        finite_ranks = rankdata(x[finite_mask], method=method)
        ranks[finite_mask] = finite_ranks
        
        if (~finite_mask).any():
            max_rank = np.max(finite_ranks)
            ranks[x == float('-inf')] = 0
            ranks[x == float('inf')] = max_rank + 1
    
    if ranks.max() > ranks.min():
        ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    
    return ranks

def robust_quantile_transform(x, n_quantiles=1000, subsample=100000):
    """Robust quantile transformation with subsampling for large datasets"""
    # Handle infinities first
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    if len(x_finite) > subsample:
        np.random.seed(42)
        indices = np.random.choice(len(x_finite), subsample, replace=False)
        fit_data = x_finite[indices]
    else:
        fit_data = x_finite
    
    qt = QuantileTransformer(n_quantiles=min(n_quantiles, len(fit_data)), 
                            output_distribution='normal')
    qt.fit(fit_data.reshape(-1, 1))
    
    x_transformed = x.copy()
    x_transformed[finite_mask] = qt.transform(x_finite.reshape(-1, 1)).ravel()
    
    # Post-transformation clip to prevent extreme values
    x_transformed = clip_extreme_values(x_transformed, percentile_range=(0.1, 99.9))
    
    return x_transformed

def tapered_sigmoid_transform(x, k=0.1):
    """Apply a tapered sigmoid transformation"""
    # Handle infinities
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    median = np.median(x_finite)
    mad = np.median(np.abs(x_finite - median))
    
    if mad == 0:
        mad = np.std(x_finite)
    if mad == 0:
        return x
    
    x_standardized = (x - median) / (mad * 1.4826)
    x_standardized = np.clip(x_standardized, -10, 10)  # Prevent extreme values
    
    x_transformed = np.tanh(k * x_standardized)
    
    scale = np.percentile(x_finite, 95) - np.percentile(x_finite, 5)
    x_final = x_transformed * scale / 2 + median
    
    return x_final

def sinh_arcsinh_transform(x, epsilon=0.1, delta=1.0):
    """Sinh-arcsinh transformation with infinity handling"""
    # Handle infinities and clip extreme values
    x = handle_infinities(x, method='clip')
    x = clip_extreme_values(x, percentile_range=(0.1, 99.9))
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    mean = np.mean(x_finite)
    std = np.std(x_finite)
    if std == 0:
        return x
    
    # Limit standardized values to prevent overflow
    x_standardized = (x - mean) / std
    x_standardized = np.clip(x_standardized, -10, 10)
    
    # Apply transformation with overflow protection
    with np.errstate(over='ignore', invalid='ignore'):
        x_transformed = np.sinh(delta * np.arcsinh(x_standardized) - epsilon)
    
    # Replace any new infinities
    x_transformed = handle_infinities(x_transformed, method='clip')
    
    # Scale back
    x_final = x_transformed * std + mean
    
    # Final clip to ensure no extreme values
    x_final = clip_extreme_values(x_final, percentile_range=(0.05, 99.95))
    
    return x_final

def adaptive_scaling_transform(x, n_bins=20):
    """Adaptive scaling based on local density"""
    # Handle infinities
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    # Create bins based on value distribution
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(x_finite, percentiles)
    bin_edges[0] -= 1e-10
    bin_edges[-1] += 1e-10
    
    # Calculate scale for each bin
    scales = np.ones_like(x_finite)
    for i in range(n_bins):
        mask = (x_finite >= bin_edges[i]) & (x_finite < bin_edges[i + 1])
        if np.sum(mask) > 1:
            bin_values = x_finite[mask]
            median = np.median(bin_values)
            mad = np.median(np.abs(bin_values - median))
            scale = mad * 1.4826 if mad > 0 else np.std(bin_values)
            scales[mask] = scale if scale > 0 else 1.0
    
    # Apply transformation
    x_transformed = x.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        x_transformed[finite_mask] = x_finite / scales
    
    # Handle any new infinities
    x_transformed = handle_infinities(x_transformed, method='clip')
    
    # Clip extreme values
    x_transformed = clip_extreme_values(x_transformed)
    
    return x_transformed

def regime_detection_transform(x, n_regimes=5):
    """Regime detection based on distribution clustering"""
    # Handle infinities
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    # Use percentile-based regime detection
    percentiles = np.linspace(0, 100, n_regimes + 1)
    regime_boundaries = np.percentile(x_finite, percentiles)
    
    # Encode values based on regime
    x_transformed = x.copy()
    finite_values = x_transformed[finite_mask]
    
    for i in range(len(finite_values)):
        val = finite_values[i]
        regime = np.searchsorted(regime_boundaries[1:-1], val)
        finite_values[i] = regime / (n_regimes - 1)
    
    x_transformed[finite_mask] = finite_values
    
    # This transformation is bounded [0, 1] so no additional handling needed
    return x_transformed

def modified_zscore_transform(x, threshold=3.5):
    """Modified Z-score using median and MAD for robustness"""
    # Handle infinities
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    median = np.median(x_finite)
    mad = np.median(np.abs(x_finite - median))
    
    if mad == 0:
        mad = np.std(x_finite)
    if mad == 0:
        return x
    
    # Modified z-score
    with np.errstate(divide='ignore', invalid='ignore'):
        modified_z = 0.6745 * (x - median) / mad
    
    # Clip extreme values
    x_transformed = np.clip(modified_z, -threshold, threshold)
    
    # Handle any infinities
    x_transformed = handle_infinities(x_transformed, method='clip')
    
    return x_transformed

def asymmetric_winsorize_transform(x, lower_pct=1, upper_pct=5):
    """Asymmetric winsorization adapted for crypto (positive skew)"""
    # Handle infinities
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    lower_bound = np.percentile(x_finite, lower_pct)
    upper_bound = np.percentile(x_finite, 100 - upper_pct)
    
    x_transformed = x.copy()
    x_transformed[x_transformed < lower_bound] = lower_bound
    x_transformed[x_transformed > upper_bound] = upper_bound
    
    return x_transformed

def log_transform_robust(x, shift_method='auto'):
    """Robust log transformation with automatic shift for negative values"""
    # Handle infinities
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    # Determine shift
    min_val = np.min(x_finite)
    if shift_method == 'auto':
        if min_val <= 0:
            shift = abs(min_val) + 1
        else:
            shift = 0
    else:
        shift = shift_method
    
    # Apply log transformation
    x_transformed = x.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        x_transformed[finite_mask] = np.log1p(x_finite + shift)
    
    # Handle any new infinities
    x_transformed = handle_infinities(x_transformed, method='clip')
    
    return x_transformed

def huber_transform(x, epsilon=1.345):
    """Huber transformation - robust to outliers"""
    # Handle infinities
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    # Standardize
    median = np.median(x_finite)
    mad = np.median(np.abs(x_finite - median))
    
    if mad == 0:
        mad = np.std(x_finite)
    if mad == 0:
        return x
    
    x_standardized = (x - median) / (mad * 1.4826)
    x_standardized = np.clip(x_standardized, -20, 20)  # Prevent extreme values
    
    # Huber transformation
    x_transformed = x.copy()
    mask_linear = np.abs(x_standardized) <= epsilon
    mask_log = ~mask_linear & finite_mask
    
    x_transformed[mask_linear] = x_standardized[mask_linear]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        x_transformed[mask_log] = epsilon * np.sign(x_standardized[mask_log]) * \
                                  (1 + np.log(np.abs(x_standardized[mask_log]) / epsilon))
    
    # Handle any infinities
    x_transformed = handle_infinities(x_transformed, method='clip')
    
    # Scale back
    scale = np.percentile(x_finite, 95) - np.percentile(x_finite, 5)
    x_transformed = x_transformed * scale / (2 * epsilon) + median
    
    return x_transformed

def box_cox_robust(x):
    """Box-Cox transformation with robust handling"""
    # Handle infinities and clip extreme values
    x = handle_infinities(x, method='clip')
    x = clip_extreme_values(x, percentile_range=(0.1, 99.9))
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    # Ensure positive values
    min_val = np.min(x_finite)
    if min_val <= 0:
        shift = abs(min_val) + 1
        x_shifted = x_finite + shift
    else:
        shift = 0
        x_shifted = x_finite
    
    try:
        # Clip to reasonable range before Box-Cox
        x_shifted = np.clip(x_shifted, 1e-10, 1e10)
        
        x_bc, lambda_param = boxcox(x_shifted)
        
        # Check for infinities in result
        if not np.isfinite(x_bc).all():
            raise ValueError("Box-Cox produced non-finite values")
        
        x_transformed = x.copy()
        x_transformed[finite_mask] = x_bc
        
        # Normalize to original scale
        bc_mean = np.mean(x_bc)
        bc_std = np.std(x_bc)
        if bc_std > 0:
            x_transformed[finite_mask] = (x_transformed[finite_mask] - bc_mean) * \
                                        np.std(x_finite) / bc_std + np.mean(x_finite)
        
        # Final clip
        x_transformed = clip_extreme_values(x_transformed)
        
    except:
        # Fallback to log transform if Box-Cox fails
        x_transformed = log_transform_robust(x)
    
    return x_transformed

def percentile_rank_transform(x):
    """Percentile rank transformation - converts values to their percentile ranks"""
    # Handle infinities
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    from scipy.stats import percentileofscore
    
    x_transformed = x.copy()
    percentiles = np.zeros_like(x_finite)
    
    # Use vectorized approach for efficiency
    sorted_indices = np.argsort(x_finite)
    percentiles[sorted_indices] = np.arange(1, len(x_finite) + 1) / len(x_finite)
    
    x_transformed[finite_mask] = percentiles
    
    # This transformation is bounded [0, 1] so no additional handling needed
    return x_transformed

def double_sigmoid_transform(x, k1=0.1, k2=0.05):
    """Double sigmoid - different transformations for positive and negative values"""
    # Handle infinities
    x = handle_infinities(x, method='clip')
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    median = np.median(x_finite)
    mad = np.median(np.abs(x_finite - median))
    
    if mad == 0:
        mad = np.std(x_finite)
    if mad == 0:
        return x
    
    x_standardized = (x - median) / (mad * 1.4826)
    x_standardized = np.clip(x_standardized, -10, 10)  # Prevent extreme values
    
    # Apply different sigmoid for positive and negative values
    x_transformed = x.copy()
    pos_mask = (x_standardized >= 0) & finite_mask
    neg_mask = (x_standardized < 0) & finite_mask
    
    x_transformed[pos_mask] = np.tanh(k1 * x_standardized[pos_mask])
    x_transformed[neg_mask] = np.tanh(k2 * x_standardized[neg_mask])
    
    # Scale back
    scale = np.percentile(x_finite, 95) - np.percentile(x_finite, 5)
    x_transformed = x_transformed * scale / 2 + median
    
    return x_transformed

def lambert_w_transform(x):
    """Lambert W transformation with robust handling"""
    # Handle infinities and clip extreme values
    x = handle_infinities(x, method='clip')
    x = clip_extreme_values(x, percentile_range=(0.5, 99.5))
    
    finite_mask = np.isfinite(x)
    x_finite = x[finite_mask]
    
    if len(x_finite) == 0:
        return x
    
    # Standardize
    mean = np.mean(x_finite)
    std = np.std(x_finite)
    if std == 0:
        return x
    
    x_standardized = (x - mean) / std
    x_standardized = np.clip(x_standardized, -5, 5)  # Limit range
    
    # Estimate delta parameter using skewness
    from scipy.stats import skew
    skewness = skew(x_finite)
    delta = np.sign(skewness) * min(abs(skewness) / 3, 0.3)  # Reduced delta
    
    # Apply Lambert W transformation
    x_transformed = x.copy()
    u = delta * x_standardized
    
    # For numerical stability
    mask_small = np.abs(u) < 0.01
    x_transformed[mask_small & finite_mask] = x_standardized[mask_small & finite_mask]
    
    mask_large = ~mask_small & finite_mask
    if mask_large.any():
        u_large = u[mask_large & finite_mask]
        # Limit the argument to lambertw
        arg = np.clip(u_large * np.exp(u_large), -700, 700)
        
        with np.errstate(invalid='ignore', over='ignore'):
            w_values = np.real(lambertw(arg))
        
        # Replace any infinities
        w_values = handle_infinities(w_values, method='clip')
        
        if abs(delta) > 1e-10:
            x_transformed[mask_large] = w_values / delta
        else:
            x_transformed[mask_large] = x_standardized[mask_large]
    
    # Scale back
    x_transformed = x_transformed * std + mean
    
    # Final clip
    x_transformed = clip_extreme_values(x_transformed)
    
    return x_transformed

def apply_transformation(df, method, columns=None, **kwargs):
    """Apply transformation to specified columns with robust infinity handling"""
    transformed_df = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = [col for col in numeric_cols if col not in ['timestamp', 'label']]
    
    for col in columns:
        original_values = df[col].values.copy()
        
        # Pre-process: handle existing infinities
        original_values = handle_infinities(original_values, method='clip')
        
        # Apply transformation
        if method == 'sigmoid':
            k = kwargs.get('k', 0.1)
            transformed_values = tapered_sigmoid_transform(original_values, k=k)
            
        elif method == 'rank':
            rank_method = kwargs.get('rank_method', 'average')
            transformed_values = robust_rank_transform(original_values, method=rank_method)
            
        elif method == 'robust_quantile':
            n_quantiles = kwargs.get('n_quantiles', 1000)
            subsample = kwargs.get('subsample', 100000)
            transformed_values = robust_quantile_transform(original_values, 
                                                         n_quantiles=n_quantiles,
                                                         subsample=subsample)
            
        elif method == 'sinh_arcsinh':
            epsilon = kwargs.get('epsilon', 0.1)
            delta = kwargs.get('delta', 1.0)
            transformed_values = sinh_arcsinh_transform(original_values, epsilon=epsilon, delta=delta)
            
        elif method == 'yeo_johnson':
            try:
                # Clip extreme values before transformation
                clipped_values = clip_extreme_values(original_values, percentile_range=(0.1, 99.9))
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                transformed_values = pt.fit_transform(clipped_values.reshape(-1, 1)).ravel()
                # Clip again after transformation
                transformed_values = clip_extreme_values(transformed_values)
            except:
                # Fallback to rank transform
                transformed_values = robust_rank_transform(original_values)
            
        elif method == 'quantile':
            try:
                output_dist = kwargs.get('output_distribution', 'normal')
                n_quantiles = kwargs.get('n_quantiles', 10000)
                
                # Use robust quantile transform instead
                transformed_values = robust_quantile_transform(original_values, 
                                                             n_quantiles=n_quantiles,
                                                             subsample=100000)
                
                if output_dist == 'normal':
                    finite_mask = np.isfinite(original_values)
                    if finite_mask.any():
                        scale = np.std(original_values[finite_mask])
                        center = np.median(original_values[finite_mask])
                        transformed_values = transformed_values * scale + center
                        transformed_values = clip_extreme_values(transformed_values)
            except:
                # Fallback
                transformed_values = robust_rank_transform(original_values)
                
        elif method == 'adaptive_scaling':
            n_bins = kwargs.get('n_bins', 20)
            transformed_values = adaptive_scaling_transform(original_values, n_bins=n_bins)
            
        elif method == 'regime_detection':
            n_regimes = kwargs.get('n_regimes', 5)
            transformed_values = regime_detection_transform(original_values, n_regimes=n_regimes)
            
        elif method == 'modified_zscore':
            threshold = kwargs.get('threshold', 3.5)
            transformed_values = modified_zscore_transform(original_values, threshold=threshold)
            
        elif method == 'asymmetric_winsorize':
            lower_pct = kwargs.get('lower_pct', 1)
            upper_pct = kwargs.get('upper_pct', 5)
            transformed_values = asymmetric_winsorize_transform(original_values, 
                                                              lower_pct=lower_pct, 
                                                              upper_pct=upper_pct)
            
        elif method == 'log_robust':
            shift_method = kwargs.get('shift_method', 'auto')
            transformed_values = log_transform_robust(original_values, shift_method=shift_method)
            
        elif method == 'huber':
            epsilon = kwargs.get('epsilon', 1.345)
            transformed_values = huber_transform(original_values, epsilon=epsilon)
            
        elif method == 'box_cox':
            transformed_values = box_cox_robust(original_values)
            
        elif method == 'lambert_w':
            transformed_values = lambert_w_transform(original_values)
            
        elif method == 'percentile_rank':
            transformed_values = percentile_rank_transform(original_values)
            
        elif method == 'double_sigmoid':
            k1 = kwargs.get('k1', 0.1)
            k2 = kwargs.get('k2', 0.05)
            transformed_values = double_sigmoid_transform(original_values, k1=k1, k2=k2)
        
        else:
            raise ValueError(f"Unknown transformation method: {method}")
        
        # Post-process: final check for infinities
        transformed_values = handle_infinities(transformed_values, method='clip')
        
        # Ensure no extreme values remain
        if method not in ['rank', 'regime_detection', 'percentile_rank']:  # These are already bounded
            transformed_values = clip_extreme_values(transformed_values, percentile_range=(0.01, 99.99))
        
        transformed_df[col] = transformed_values
    
    return transformed_df

# =========================
# Training Functions
# =========================
def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples)}
    ]

def train_and_evaluate(train_df, test_df):
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)

    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        for learner in LEARNERS
    }

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"  Fold {fold}/{Config.N_FOLDS}")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff

            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]

            for learner in LEARNERS:
                model = learner["Estimator"](**learner["params"])
                model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_valid, y_valid)], verbose=False)

                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                if cutoff > 0 and (~mask).any():
                    oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]

                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES])

    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS

    return oof_preds, test_preds

def create_submission(train_df, oof_preds, test_preds, submission_df, method_name):
    learner_ensembles = {}
    for learner_name in oof_preds:
        scores = {s: pearsonr(train_df[Config.LABEL_COLUMN], oof_preds[learner_name][s])[0]
                  for s in oof_preds[learner_name]}
        total_score = sum(scores.values())

        oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
        test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
        score_simple = pearsonr(train_df[Config.LABEL_COLUMN], oof_simple)[0]

        oof_weighted = sum(scores[s] / total_score * oof_preds[learner_name][s] for s in scores)
        test_weighted = sum(scores[s] / total_score * test_preds[learner_name][s] for s in scores)
        score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]

        print(f"  {learner_name.upper()} Simple Ensemble Pearson:   {score_simple:.4f}")
        print(f"  {learner_name.upper()} Weighted Ensemble Pearson: {score_weighted:.4f}")

        learner_ensembles[learner_name] = {
            "oof_simple": oof_simple,
            "test_simple": test_simple,
            "oof_weighted": oof_weighted,
            "test_weighted": test_weighted
        }

    final_oof = np.mean([le["oof_weighted"] for le in learner_ensembles.values()], axis=0)
    final_test = np.mean([le["test_weighted"] for le in learner_ensembles.values()], axis=0)
    final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]

    print(f"  FINAL ensemble Pearson: {final_score:.4f}")

    submission_df["prediction"] = final_test
    filename = f"submission_{method_name}.csv"
    submission_df.to_csv(filename, index=False)
    print(f"  Saved: {filename}")
    
    return final_score

# =========================
# Main Execution
# =========================
def main():
    # Define all transformation methods with their parameters
    transformation_methods = [
        # Original methods (with safer parameters)
        {"name": "robust_quantile", "params": {"n_quantiles": 1000, "subsample": 100000}},
        {"name": "rank", "params": {"rank_method": "average"}},
        {"name": "sigmoid", "params": {"k": 0.05}},
        {"name": "quantile", "params": {"output_distribution": "normal", "n_quantiles": 10000}},
        {"name": "sinh_arcsinh", "params": {"epsilon": 0.1, "delta": 0.5}},  # Reduced delta
        {"name": "yeo_johnson", "params": {}},
        
        # New distribution-based methods
        {"name": "adaptive_scaling", "params": {"n_bins": 20}},
        {"name": "regime_detection", "params": {"n_regimes": 5}},
        {"name": "modified_zscore", "params": {"threshold": 3.5}},
        {"name": "asymmetric_winsorize", "params": {"lower_pct": 1, "upper_pct": 5}},
        {"name": "log_robust", "params": {"shift_method": "auto"}},
        {"name": "huber", "params": {"epsilon": 1.345}},
        {"name": "box_cox", "params": {}},
        {"name": "lambert_w", "params": {}},
        {"name": "percentile_rank", "params": {}},
        {"name": "double_sigmoid", "params": {"k1": 0.1, "k2": 0.05}}
    ]
    
    # Load raw data once
    print("Loading raw data...")
    train_df_raw = pd.read_parquet(Config.TRAIN_PATH)
    test_df_raw = pd.read_parquet(Config.TEST_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    print(f"Raw data - Train: {train_df_raw.shape}, Test: {test_df_raw.shape}")
    
    # Check for infinities in raw data
    print("\nChecking for infinities in raw data...")
    for col in Config.FEATURES:
        n_inf_train = (~np.isfinite(train_df_raw[col])).sum()
        n_inf_test = (~np.isfinite(test_df_raw[col])).sum()
        if n_inf_train > 0 or n_inf_test > 0:
            print(f"  {col}: Train infinities = {n_inf_train}, Test infinities = {n_inf_test}")
    
    # Reduce memory usage
    train_df_raw = reduce_mem_usage(train_df_raw, 'train', verbose=False)
    test_df_raw = reduce_mem_usage(test_df_raw, 'test', verbose=False)
    
    results = []
    
    # Process each transformation method
    for method_info in transformation_methods:
        method_name = method_info["name"]
        method_params = method_info["params"]
        
        print(f"\n{'='*60}")
        print(f"Processing transformation method: {method_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Apply transformation only to FEATURES columns
            print(f"Applying {method_name} transformation...")
            train_df = train_df_raw.copy()
            test_df = test_df_raw.copy()
            
            # Transform only the feature columns
            train_df[Config.FEATURES] = apply_transformation(
                train_df[Config.FEATURES], 
                method_name, 
                columns=Config.FEATURES,
                **method_params
            )[Config.FEATURES]
            
            test_df[Config.FEATURES] = apply_transformation(
                test_df[Config.FEATURES], 
                method_name, 
                columns=Config.FEATURES,
                **method_params
            )[Config.FEATURES]
            
            # Verify no infinities remain
            has_inf = False
            for col in Config.FEATURES:
                if (~np.isfinite(train_df[col])).any() or (~np.isfinite(test_df[col])).any():
                    print(f"  WARNING: Column {col} still has infinities after transformation!")
                    has_inf = True
            
            if has_inf:
                print(f"  Skipping {method_name} due to remaining infinities")
                continue
            
            # Reset index
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            
            # Train and evaluate
            print(f"Training models with {method_name} transformation...")
            oof_preds, test_preds = train_and_evaluate(train_df, test_df)
            
            # Create submission
            print(f"Creating submission for {method_name}...")
            final_score = create_submission(train_df, oof_preds, test_preds, 
                                          submission_df.copy(), method_name)
            
            results.append({"method": method_name, "score": final_score})
            
        except Exception as e:
            print(f"  ERROR with {method_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL METHODS")
    print(f"{'='*60}")
    for result in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"{result['method']:<25} Pearson: {result['score']:.4f}")
    
    if results:
        best_method = max(results, key=lambda x: x['score'])
        print(f"\nBest method: {best_method['method']} (Pearson: {best_method['score']:.4f})")
        print(f"Total submission files created: {len(results)}")

if __name__ == "__main__":
    main()


# import sys
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import KFold
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from scipy.stats import pearsonr, rankdata, boxcox
# from scipy.special import lambertw
# from sklearn.preprocessing import QuantileTransformer, PowerTransformer
# import warnings
# warnings.filterwarnings('ignore')

# # =========================
# # Configuration
# # =========================
# class Config:
#     TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
#     TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
#     SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

#     FEATURES = [
#         "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
#         "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
#         "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333"
#     ]

#     LABEL_COLUMN = "label"
#     N_FOLDS = 3
#     RANDOM_STATE = 42

# XGB_PARAMS = {
#     "tree_method": "hist",
#     "device": "gpu",
#     "colsample_bylevel": 0.4778,
#     "colsample_bynode": 0.3628,
#     "colsample_bytree": 0.7107,
#     "gamma": 1.7095,
#     "learning_rate": 0.02213,
#     "max_depth": 20,
#     "max_leaves": 12,
#     "min_child_weight": 16,
#     "n_estimators": 1667,
#     "subsample": 0.06567,
#     "reg_alpha": 39.3524,
#     "reg_lambda": 75.4484,
#     "verbosity": 0,
#     "random_state": Config.RANDOM_STATE,
#     "n_jobs": -1
# }

# LEARNERS = [
#     {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS}
# ]

# # =========================
# # Helper Functions
# # =========================
# def clip_extreme_values(x, percentile_range=(0.01, 99.99)):
#     """Clip extreme values to specified percentile range"""
#     finite_mask = np.isfinite(x)
#     if not finite_mask.any():
#         return x
    
#     finite_values = x[finite_mask]
#     lower = np.percentile(finite_values, percentile_range[0])
#     upper = np.percentile(finite_values, percentile_range[1])
    
#     x_clipped = x.copy()
#     x_clipped = np.clip(x_clipped, lower, upper)
#     return x_clipped

# def handle_infinities(x, method='clip'):
#     """Replace infinite values with finite extremes"""
#     if not np.any(~np.isfinite(x)):
#         return x
    
#     x_clean = x.copy()
#     finite_mask = np.isfinite(x)
    
#     if finite_mask.any():
#         finite_values = x[finite_mask]
        
#         if method == 'clip':
#             # Replace with extreme percentiles
#             min_val = np.percentile(finite_values, 0.1)
#             max_val = np.percentile(finite_values, 99.9)
#             range_val = max_val - min_val
            
#             x_clean[x == float('-inf')] = min_val - 0.01 * range_val
#             x_clean[x == float('inf')] = max_val + 0.01 * range_val
#             x_clean[np.isnan(x)] = np.median(finite_values)
        
#         elif method == 'median':
#             # Replace with median
#             median_val = np.median(finite_values)
#             x_clean[~finite_mask] = median_val
    
#     return x_clean

# def reduce_mem_usage(dataframe, dataset, verbose=True):    
#     """Reduces memory usage by converting columns to more efficient data types"""
#     if verbose:
#         print(f'Reducing memory usage for: {dataset}')
#     initial_mem_usage = dataframe.memory_usage().sum() / 1024**2

#     for col in dataframe.columns:
#         col_type = dataframe[col].dtype

#         c_min = dataframe[col].min()
#         c_max = dataframe[col].max()
#         if str(col_type)[:3] == 'int':
#             if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                 dataframe[col] = dataframe[col].astype(np.int8)
#             elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                 dataframe[col] = dataframe[col].astype(np.int16)
#             elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                 dataframe[col] = dataframe[col].astype(np.int32)
#             elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                 dataframe[col] = dataframe[col].astype(np.int64)
#         else:
#             if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                 dataframe[col] = dataframe[col].astype(np.float16)
#             elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                 dataframe[col] = dataframe[col].astype(np.float32)
#             else:
#                 dataframe[col] = dataframe[col].astype(np.float64)

#     final_mem_usage = dataframe.memory_usage().sum() / 1024**2
#     if verbose:
#         print(f'--- Memory usage before: {initial_mem_usage:.2f} MB')
#         print(f'--- Memory usage after: {final_mem_usage:.2f} MB')
#         print(f'--- Decreased memory usage by {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%\n')

#     return dataframe

# # =========================
# # Transformation Functions (with infinity handling)
# # =========================
# def robust_rank_transform(x, method='average'):
#     """Rank transformation - extremely robust to outliers and noise"""
#     # First handle infinities
#     x = handle_infinities(x, method='clip')
    
#     finite_mask = np.isfinite(x)
#     ranks = np.zeros_like(x, dtype=float)
    
#     if finite_mask.any():
#         finite_ranks = rankdata(x[finite_mask], method=method)
#         ranks[finite_mask] = finite_ranks
        
#         if (~finite_mask).any():
#             max_rank = np.max(finite_ranks)
#             ranks[x == float('-inf')] = 0
#             ranks[x == float('inf')] = max_rank + 1
    
#     if ranks.max() > ranks.min():
#         ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    
#     return ranks

# def robust_quantile_transform(x, n_quantiles=1000, subsample=100000):
#     """Robust quantile transformation with subsampling for large datasets"""
#     # Handle infinities first
#     x = handle_infinities(x, method='clip')
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     if len(x_finite) > subsample:
#         np.random.seed(42)
#         indices = np.random.choice(len(x_finite), subsample, replace=False)
#         fit_data = x_finite[indices]
#     else:
#         fit_data = x_finite
    
#     qt = QuantileTransformer(n_quantiles=min(n_quantiles, len(fit_data)), 
#                             output_distribution='normal')
#     qt.fit(fit_data.reshape(-1, 1))
    
#     x_transformed = x.copy()
#     x_transformed[finite_mask] = qt.transform(x_finite.reshape(-1, 1)).ravel()
    
#     # Post-transformation clip to prevent extreme values
#     x_transformed = clip_extreme_values(x_transformed, percentile_range=(0.1, 99.9))
    
#     return x_transformed

# def tapered_sigmoid_transform(x, k=0.1):
#     """Apply a tapered sigmoid transformation"""
#     # Handle infinities
#     x = handle_infinities(x, method='clip')
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     median = np.median(x_finite)
#     mad = np.median(np.abs(x_finite - median))
    
#     if mad == 0:
#         mad = np.std(x_finite)
#     if mad == 0:
#         return x
    
#     x_standardized = (x - median) / (mad * 1.4826)
#     x_transformed = np.tanh(k * x_standardized)
    
#     scale = np.percentile(x_finite, 95) - np.percentile(x_finite, 5)
#     x_final = x_transformed * scale / 2 + median
    
#     return x_final

# def sinh_arcsinh_transform(x, epsilon=0.1, delta=1.0):
#     """Sinh-arcsinh transformation with infinity handling"""
#     # Handle infinities and clip extreme values
#     x = handle_infinities(x, method='clip')
#     x = clip_extreme_values(x, percentile_range=(0.1, 99.9))
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     mean = np.mean(x_finite)
#     std = np.std(x_finite)
#     if std == 0:
#         return x
    
#     # Limit standardized values to prevent overflow
#     x_standardized = (x - mean) / std
#     x_standardized = np.clip(x_standardized, -10, 10)
    
#     # Apply transformation with overflow protection
#     with np.errstate(over='ignore', invalid='ignore'):
#         x_transformed = np.sinh(delta * np.arcsinh(x_standardized) - epsilon)
    
#     # Replace any new infinities
#     x_transformed = handle_infinities(x_transformed, method='clip')
    
#     # Scale back
#     x_final = x_transformed * std + mean
    
#     # Final clip to ensure no extreme values
#     x_final = clip_extreme_values(x_final, percentile_range=(0.05, 99.95))
    
#     return x_final

# def novas_transform(x, window_size=100):
#     """NoVaS transformation with infinity handling"""
#     x = handle_infinities(x, method='clip')
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     sorted_indices = np.argsort(x_finite)
#     sorted_x = x_finite[sorted_indices]
    
#     scales = np.zeros_like(sorted_x)
#     for i in range(len(sorted_x)):
#         start = max(0, i - window_size // 2)
#         end = min(len(sorted_x), i + window_size // 2)
#         window = sorted_x[start:end]
        
#         median = np.median(window)
#         mad = np.median(np.abs(window - median))
#         scales[i] = mad * 1.4826 if mad > 0 else 1.0
    
#     scale_map = np.zeros_like(x_finite)
#     scale_map[sorted_indices] = scales
    
#     x_transformed = x.copy()
#     x_transformed[finite_mask] = x_finite / scale_map
    
#     # Clip extreme values
#     x_transformed = clip_extreme_values(x_transformed)
    
#     return x_transformed

# def directional_change_transform(x, threshold=0.01):
#     """Directional Change framework adapted for feature transformation"""
#     x = handle_infinities(x, method='clip')
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     sorted_indices = np.argsort(x_finite)
#     sorted_x = x_finite[sorted_indices]
    
#     dc_levels = [sorted_x[0]]
#     last_extreme = sorted_x[0]
    
#     for val in sorted_x[1:]:
#         if abs(val - last_extreme) / (abs(last_extreme) + 1e-10) > threshold:
#             dc_levels.append(val)
#             last_extreme = val
    
#     dc_levels = np.array(dc_levels)
    
#     x_transformed = x.copy()
#     finite_values = x_transformed[finite_mask]
    
#     for i, val in enumerate(finite_values):
#         distances = np.abs(dc_levels - val)
#         nearest_idx = np.argmin(distances)
#         finite_values[i] = nearest_idx / (len(dc_levels) - 1) if len(dc_levels) > 1 else 0.5
    
#     x_transformed[finite_mask] = finite_values
    
#     if (x == float('-inf')).any():
#         x_transformed[x == float('-inf')] = 0
#     if (x == float('inf')).any():
#         x_transformed[x == float('inf')] = 1
    
#     return x_transformed

# def modified_zscore_transform(x, threshold=3.5):
#     """Modified Z-score using median and MAD for robustness"""
#     x = handle_infinities(x, method='clip')
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     median = np.median(x_finite)
#     mad = np.median(np.abs(x_finite - median))
    
#     if mad == 0:
#         mad = np.std(x_finite)
#     if mad == 0:
#         return x
    
#     modified_z = 0.6745 * (x - median) / mad
#     x_transformed = np.clip(modified_z, -threshold, threshold)
    
#     return x_transformed

# def asymmetric_winsorize_transform(x, lower_pct=1, upper_pct=5):
#     """Asymmetric winsorization adapted for crypto (positive skew)"""
#     x = handle_infinities(x, method='clip')
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     lower_bound = np.percentile(x_finite, lower_pct)
#     upper_bound = np.percentile(x_finite, 100 - upper_pct)
    
#     x_transformed = x.copy()
#     x_transformed[x_transformed < lower_bound] = lower_bound
#     x_transformed[x_transformed > upper_bound] = upper_bound
    
#     return x_transformed

# def log_transform_robust(x, shift_method='auto'):
#     """Robust log transformation with automatic shift for negative values"""
#     x = handle_infinities(x, method='clip')
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     min_val = np.min(x_finite)
#     if shift_method == 'auto':
#         if min_val <= 0:
#             shift = abs(min_val) + 1
#         else:
#             shift = 0
#     else:
#         shift = shift_method
    
#     x_transformed = x.copy()
#     with np.errstate(divide='ignore', invalid='ignore'):
#         x_transformed[finite_mask] = np.log1p(x_finite + shift)
    
#     # Handle any new infinities
#     x_transformed = handle_infinities(x_transformed, method='clip')
    
#     return x_transformed

# def huber_transform(x, epsilon=1.345):
#     """Huber transformation - robust to outliers"""
#     x = handle_infinities(x, method='clip')
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     median = np.median(x_finite)
#     mad = np.median(np.abs(x_finite - median))
    
#     if mad == 0:
#         mad = np.std(x_finite)
#     if mad == 0:
#         return x
    
#     x_standardized = (x - median) / (mad * 1.4826)
    
#     x_transformed = x.copy()
#     mask_linear = np.abs(x_standardized) <= epsilon
#     mask_log = ~mask_linear & finite_mask
    
#     x_transformed[mask_linear] = x_standardized[mask_linear]
    
#     with np.errstate(divide='ignore', invalid='ignore'):
#         x_transformed[mask_log] = epsilon * np.sign(x_standardized[mask_log]) * \
#                                   (1 + np.log(np.abs(x_standardized[mask_log]) / epsilon))
    
#     # Handle any infinities
#     x_transformed = handle_infinities(x_transformed, method='clip')
    
#     scale = np.percentile(x_finite, 95) - np.percentile(x_finite, 5)
#     x_transformed = x_transformed * scale / (2 * epsilon) + median
    
#     return x_transformed

# def box_cox_robust(x):
#     """Box-Cox transformation with robust handling"""
#     x = handle_infinities(x, method='clip')
#     x = clip_extreme_values(x, percentile_range=(0.1, 99.9))
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     # Ensure positive values
#     min_val = np.min(x_finite)
#     if min_val <= 0:
#         shift = abs(min_val) + 1
#         x_shifted = x_finite + shift
#     else:
#         shift = 0
#         x_shifted = x_finite
    
#     try:
#         # Clip to reasonable range before Box-Cox
#         x_shifted = np.clip(x_shifted, 1e-10, 1e10)
        
#         x_bc, lambda_param = boxcox(x_shifted)
        
#         # Check for infinities in result
#         if not np.isfinite(x_bc).all():
#             raise ValueError("Box-Cox produced non-finite values")
        
#         x_transformed = x.copy()
#         x_transformed[finite_mask] = x_bc
        
#         # Normalize to original scale
#         bc_mean = np.mean(x_bc)
#         bc_std = np.std(x_bc)
#         if bc_std > 0:
#             x_transformed[finite_mask] = (x_transformed[finite_mask] - bc_mean) * \
#                                         np.std(x_finite) / bc_std + np.mean(x_finite)
        
#         # Final clip
#         x_transformed = clip_extreme_values(x_transformed)
        
#     except:
#         # Fallback to log transform if Box-Cox fails
#         x_transformed = log_transform_robust(x)
    
#     return x_transformed

# def lambert_w_transform(x):
#     """Lambert W transformation with robust handling"""
#     x = handle_infinities(x, method='clip')
#     x = clip_extreme_values(x, percentile_range=(0.5, 99.5))
    
#     finite_mask = np.isfinite(x)
#     x_finite = x[finite_mask]
    
#     if len(x_finite) == 0:
#         return x
    
#     mean = np.mean(x_finite)
#     std = np.std(x_finite)
#     if std == 0:
#         return x
    
#     x_standardized = (x - mean) / std
#     x_standardized = np.clip(x_standardized, -5, 5)  # Limit range
    
#     from scipy.stats import skew
#     skewness = skew(x_finite)
#     delta = np.sign(skewness) * min(abs(skewness) / 3, 0.3)  # Reduced delta
    
#     x_transformed = x.copy()
#     u = delta * x_standardized
    
#     mask_small = np.abs(u) < 0.01
#     x_transformed[mask_small & finite_mask] = x_standardized[mask_small & finite_mask]
    
#     mask_large = ~mask_small & finite_mask
#     if mask_large.any():
#         u_large = u[mask_large & finite_mask]
#         # Limit the argument to lambertw
#         arg = np.clip(u_large * np.exp(u_large), -700, 700)
        
#         with np.errstate(invalid='ignore', over='ignore'):
#             w_values = np.real(lambertw(arg))
        
#         # Replace any infinities
#         w_values = handle_infinities(w_values, method='clip')
        
#         if abs(delta) > 1e-10:
#             x_transformed[mask_large] = w_values / delta
#         else:
#             x_transformed[mask_large] = x_standardized[mask_large]
    
#     # Scale back
#     x_transformed = x_transformed * std + mean
    
#     # Final clip
#     x_transformed = clip_extreme_values(x_transformed)
    
#     return x_transformed

# def apply_transformation(df, method, columns=None, **kwargs):
#     """Apply transformation to specified columns with robust infinity handling"""
#     transformed_df = df.copy()
    
#     if columns is None:
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         columns = [col for col in numeric_cols if col not in ['timestamp', 'label']]
    
#     for col in columns:
#         original_values = df[col].values.copy()
        
#         # Pre-process: handle existing infinities
#         original_values = handle_infinities(original_values, method='clip')
        
#         # Apply transformation
#         if method == 'sigmoid':
#             k = kwargs.get('k', 0.1)
#             transformed_values = tapered_sigmoid_transform(original_values, k=k)
            
#         elif method == 'rank':
#             rank_method = kwargs.get('rank_method', 'average')
#             transformed_values = robust_rank_transform(original_values, method=rank_method)
            
#         elif method == 'robust_quantile':
#             n_quantiles = kwargs.get('n_quantiles', 1000)
#             subsample = kwargs.get('subsample', 100000)
#             transformed_values = robust_quantile_transform(original_values, 
#                                                          n_quantiles=n_quantiles,
#                                                          subsample=subsample)
            
#         elif method == 'sinh_arcsinh':
#             epsilon = kwargs.get('epsilon', 0.1)
#             delta = kwargs.get('delta', 1.0)
#             transformed_values = sinh_arcsinh_transform(original_values, epsilon=epsilon, delta=delta)
            
#         elif method == 'yeo_johnson':
#             try:
#                 # Clip extreme values before transformation
#                 clipped_values = clip_extreme_values(original_values, percentile_range=(0.1, 99.9))
#                 pt = PowerTransformer(method='yeo-johnson', standardize=False)
#                 transformed_values = pt.fit_transform(clipped_values.reshape(-1, 1)).ravel()
#                 # Clip again after transformation
#                 transformed_values = clip_extreme_values(transformed_values)
#             except:
#                 # Fallback to rank transform
#                 transformed_values = robust_rank_transform(original_values)
            
#         elif method == 'quantile':
#             try:
#                 output_dist = kwargs.get('output_distribution', 'normal')
#                 n_quantiles = kwargs.get('n_quantiles', 10000)
                
#                 # Use robust quantile transform instead
#                 transformed_values = robust_quantile_transform(original_values, 
#                                                              n_quantiles=n_quantiles,
#                                                              subsample=100000)
                
#                 if output_dist == 'normal':
#                     finite_mask = np.isfinite(original_values)
#                     if finite_mask.any():
#                         scale = np.std(original_values[finite_mask])
#                         center = np.median(original_values[finite_mask])
#                         transformed_values = transformed_values * scale + center
#                         transformed_values = clip_extreme_values(transformed_values)
#             except:
#                 # Fallback
#                 transformed_values = robust_rank_transform(original_values)
                
#         elif method == 'novas':
#             window_size = kwargs.get('window_size', 100)
#             transformed_values = novas_transform(original_values, window_size=window_size)
            
#         elif method == 'directional_change':
#             threshold = kwargs.get('threshold', 0.01)
#             transformed_values = directional_change_transform(original_values, threshold=threshold)
            
#         elif method == 'modified_zscore':
#             threshold = kwargs.get('threshold', 3.5)
#             transformed_values = modified_zscore_transform(original_values, threshold=threshold)
            
#         elif method == 'asymmetric_winsorize':
#             lower_pct = kwargs.get('lower_pct', 1)
#             upper_pct = kwargs.get('upper_pct', 5)
#             transformed_values = asymmetric_winsorize_transform(original_values, 
#                                                               lower_pct=lower_pct, 
#                                                               upper_pct=upper_pct)
            
#         elif method == 'log_robust':
#             shift_method = kwargs.get('shift_method', 'auto')
#             transformed_values = log_transform_robust(original_values, shift_method=shift_method)
            
#         elif method == 'huber':
#             epsilon = kwargs.get('epsilon', 1.345)
#             transformed_values = huber_transform(original_values, epsilon=epsilon)
            
#         elif method == 'box_cox':
#             transformed_values = box_cox_robust(original_values)
            
#         elif method == 'lambert_w':
#             transformed_values = lambert_w_transform(original_values)
        
#         else:
#             raise ValueError(f"Unknown transformation method: {method}")
        
#         # Post-process: final check for infinities
#         transformed_values = handle_infinities(transformed_values, method='clip')
        
#         # Ensure no extreme values remain
#         if method not in ['rank', 'directional_change']:  # These are already bounded
#             transformed_values = clip_extreme_values(transformed_values, percentile_range=(0.01, 99.99))
        
#         transformed_df[col] = transformed_values
    
#     return transformed_df

# # =========================
# # Training Functions
# # =========================
# def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
#     positions = np.arange(n)
#     normalized = positions / (n - 1)
#     weights = decay ** (1.0 - normalized)
#     return weights * n / weights.sum()

# def get_model_slices(n_samples: int):
#     return [
#         {"name": "full_data", "cutoff": 0},
#         {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
#         {"name": "last_50pct", "cutoff": int(0.50 * n_samples)}
#     ]

# def train_and_evaluate(train_df, test_df):
#     n_samples = len(train_df)
#     model_slices = get_model_slices(n_samples)

#     oof_preds = {
#         learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
#         for learner in LEARNERS
#     }
#     test_preds = {
#         learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
#         for learner in LEARNERS
#     }

#     full_weights = create_time_decay_weights(n_samples)
#     kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

#     for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
#         print(f"  Fold {fold}/{Config.N_FOLDS}")
#         X_valid = train_df.iloc[valid_idx][Config.FEATURES]
#         y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

#         for s in model_slices:
#             cutoff = s["cutoff"]
#             slice_name = s["name"]
#             subset = train_df.iloc[cutoff:].reset_index(drop=True)
#             rel_idx = train_idx[train_idx >= cutoff] - cutoff

#             X_train = subset.iloc[rel_idx][Config.FEATURES]
#             y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
#             sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]

#             for learner in LEARNERS:
#                 model = learner["Estimator"](**learner["params"])
#                 model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_valid, y_valid)], verbose=False)

#                 mask = valid_idx >= cutoff
#                 if mask.any():
#                     idxs = valid_idx[mask]
#                     oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
#                 if cutoff > 0 and (~mask).any():
#                     oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]

#                 test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES])

#     # Normalize test predictions
#     for learner_name in test_preds:
#         for slice_name in test_preds[learner_name]:
#             test_preds[learner_name][slice_name] /= Config.N_FOLDS

#     return oof_preds, test_preds

# def create_submission(train_df, oof_preds, test_preds, submission_df, method_name):
#     learner_ensembles = {}
#     for learner_name in oof_preds:
#         scores = {s: pearsonr(train_df[Config.LABEL_COLUMN], oof_preds[learner_name][s])[0]
#                   for s in oof_preds[learner_name]}
#         total_score = sum(scores.values())

#         oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
#         test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
#         score_simple = pearsonr(train_df[Config.LABEL_COLUMN], oof_simple)[0]

#         oof_weighted = sum(scores[s] / total_score * oof_preds[learner_name][s] for s in scores)
#         test_weighted = sum(scores[s] / total_score * test_preds[learner_name][s] for s in scores)
#         score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]

#         print(f"  {learner_name.upper()} Simple Ensemble Pearson:   {score_simple:.4f}")
#         print(f"  {learner_name.upper()} Weighted Ensemble Pearson: {score_weighted:.4f}")

#         learner_ensembles[learner_name] = {
#             "oof_simple": oof_simple,
#             "test_simple": test_simple,
#             "oof_weighted": oof_weighted,
#             "test_weighted": test_weighted
#         }

#     final_oof = np.mean([le["oof_weighted"] for le in learner_ensembles.values()], axis=0)
#     final_test = np.mean([le["test_weighted"] for le in learner_ensembles.values()], axis=0)
#     final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]

#     print(f"  FINAL ensemble Pearson: {final_score:.4f}")

#     submission_df["prediction"] = final_test
#     filename = f"submission_{method_name}.csv"
#     submission_df.to_csv(filename, index=False)
#     print(f"  Saved: {filename}")
    
#     return final_score

# # =========================
# # Main Execution
# # =========================
# def main():
#     # Define all transformation methods with their parameters
#     transformation_methods = [
#         # Original methods (with safer parameters)
#         {"name": "robust_quantile", "params": {"n_quantiles": 1000, "subsample": 100000}},
#         {"name": "rank", "params": {"rank_method": "average"}},
#         {"name": "sigmoid", "params": {"k": 0.05}},
#         {"name": "quantile", "params": {"output_distribution": "normal", "n_quantiles": 10000}},
#         {"name": "sinh_arcsinh", "params": {"epsilon": 0.1, "delta": 0.5}},  # Reduced delta
#         {"name": "yeo_johnson", "params": {}},
        
#         # New advanced methods
#         {"name": "novas", "params": {"window_size": 100}},
#         {"name": "directional_change", "params": {"threshold": 0.01}},
#         {"name": "modified_zscore", "params": {"threshold": 3.5}},
#         {"name": "asymmetric_winsorize", "params": {"lower_pct": 1, "upper_pct": 5}},
#         {"name": "log_robust", "params": {"shift_method": "auto"}},
#         {"name": "huber", "params": {"epsilon": 1.345}},
#         {"name": "box_cox", "params": {}},
#         {"name": "lambert_w", "params": {}}
#     ]
    
#     # Load raw data once
#     print("Loading raw data...")
#     train_df_raw = pd.read_parquet(Config.TRAIN_PATH)
#     test_df_raw = pd.read_parquet(Config.TEST_PATH)
#     submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
#     print(f"Raw data - Train: {train_df_raw.shape}, Test: {test_df_raw.shape}")
    
#     # Check for infinities in raw data
#     print("\nChecking for infinities in raw data...")
#     for col in Config.FEATURES:
#         n_inf_train = (~np.isfinite(train_df_raw[col])).sum()
#         n_inf_test = (~np.isfinite(test_df_raw[col])).sum()
#         if n_inf_train > 0 or n_inf_test > 0:
#             print(f"  {col}: Train infinities = {n_inf_train}, Test infinities = {n_inf_test}")
    
#     # Reduce memory usage
#     train_df_raw = reduce_mem_usage(train_df_raw, 'train', verbose=False)
#     test_df_raw = reduce_mem_usage(test_df_raw, 'test', verbose=False)
    
#     results = []
    
#     # Process each transformation method
#     for method_info in transformation_methods:
#         method_name = method_info["name"]
#         method_params = method_info["params"]
        
#         print(f"\n{'='*60}")
#         print(f"Processing transformation method: {method_name.upper()}")
#         print(f"{'='*60}")
        
#         try:
#             # Apply transformation only to FEATURES columns
#             print(f"Applying {method_name} transformation...")
#             train_df = train_df_raw.copy()
#             test_df = test_df_raw.copy()
            
#             # Transform only the feature columns
#             train_df[Config.FEATURES] = apply_transformation(
#                 train_df[Config.FEATURES], 
#                 method_name, 
#                 columns=Config.FEATURES,
#                 **method_params
#             )[Config.FEATURES]
            
#             test_df[Config.FEATURES] = apply_transformation(
#                 test_df[Config.FEATURES], 
#                 method_name, 
#                 columns=Config.FEATURES,
#                 **method_params
#             )[Config.FEATURES]
            
#             # Verify no infinities remain
#             has_inf = False
#             for col in Config.FEATURES:
#                 if (~np.isfinite(train_df[col])).any() or (~np.isfinite(test_df[col])).any():
#                     print(f"  WARNING: Column {col} still has infinities after transformation!")
#                     has_inf = True
            
#             if has_inf:
#                 print(f"  Skipping {method_name} due to remaining infinities")
#                 continue
            
#             # Reset index
#             train_df = train_df.reset_index(drop=True)
#             test_df = test_df.reset_index(drop=True)
            
#             # Train and evaluate
#             print(f"Training models with {method_name} transformation...")
#             oof_preds, test_preds = train_and_evaluate(train_df, test_df)
            
#             # Create submission
#             print(f"Creating submission for {method_name}...")
#             final_score = create_submission(train_df, oof_preds, test_preds, 
#                                           submission_df.copy(), method_name)
            
#             results.append({"method": method_name, "score": final_score})
            
#         except Exception as e:
#             print(f"  ERROR with {method_name}: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     # Summary
#     print(f"\n{'='*60}")
#     print("SUMMARY OF ALL METHODS")
#     print(f"{'='*60}")
#     for result in sorted(results, key=lambda x: x['score'], reverse=True):
#         print(f"{result['method']:<25} Pearson: {result['score']:.4f}")
    
#     if results:
#         best_method = max(results, key=lambda x: x['score'])
#         print(f"\nBest method: {best_method['method']} (Pearson: {best_method['score']:.4f})")
#         print(f"Total submission files created: {len(results)}")

# if __name__ == "__main__":
#     main()

