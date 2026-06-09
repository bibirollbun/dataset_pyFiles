import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import random
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import warnings
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor

# Ignore warnings for clarity
warnings.filterwarnings("ignore")

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Configuration
NUM_REALIZATIONS = 10  # Number of realizations to predict

# Parameters for evaluation metric calculation
LOG_SLOPES = [1.0406028049510443, 0.0, 7.835345062351012]
LOG_OFFSETS = [-6.430669850650689, -2.1617411566043896, -45.24876794412965]


def load_data(train_path, test_path=None):
    """
    Load and prepare the dataset
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    
    # Extract IDs
    train_ids = train_df['geology_id'].values
    
    # Extract input columns (-49 to 0)
    input_cols = [str(i) for i in range(-49, 1)]
    
    # Extract output columns (1 to 300)
    output_cols = [str(i) for i in range(1, 301)]
    
    # Handle test data if provided
    if test_path:
        test_df = pd.read_csv(test_path)
        test_ids = test_df['geology_id'].values
        
        return train_df, train_ids, input_cols, output_cols, test_df, test_ids
    
    return train_df, train_ids, input_cols, output_cols


def create_derived_features(X):
    """
    Create enhanced derived features from input data with robust error handling
    
    New features include:
    1. Linear trend (slope)
    2. Moving average of last 5 points
    3. Standard deviation of last 15 points
    4. Second-order coefficient (curvature)
    5. Rate of change in last 10 points
    6. Ratio between recent and prior mean
    7. Spectral power in different frequency bands
    8. Entropy measure of signal variability
    9. Adaptative trend strength indicator
    10. Pattern change detection feature
    """
    n_samples, n_features = X.shape
    n_derived = 10  # Increased from 6 to 10 features
    derived_features = np.zeros((n_samples, n_derived))
    
    # Calculate features for each sample
    for i in range(n_samples):
        # Get the most recent points
        recent_points = X[i, -15:]  # Last 15 points
        x_vals = np.arange(-14, 1)  # Corresponding positions
        
        # Full series for spectral features
        full_series = X[i, :]
        
        # Check for NaN or inf values
        valid_mask = ~np.isnan(recent_points) & ~np.isinf(recent_points)
        valid_points = recent_points[valid_mask]
        valid_x = x_vals[valid_mask]
        
        # Clean full series for spectral analysis
        full_valid_mask = ~np.isnan(full_series) & ~np.isinf(full_series)
        full_valid = full_series[full_valid_mask]
        
        # Default values for safety
        derived_features[i, :] = np.zeros(n_derived)
        
        # 1. Linear trend (slope) - Same as original
        try:
            if len(valid_points) >= 2:
                slope, intercept = np.polyfit(valid_x, valid_points, 1)
                derived_features[i, 0] = slope
            else:
                derived_features[i, 0] = 0.0
        except:
            derived_features[i, 0] = 0.0
        
        # 2. Moving average of last 5 points - Same as original
        try:
            last_5_valid = X[i, -5:]
            last_5_valid = last_5_valid[~np.isnan(last_5_valid) & ~np.isinf(last_5_valid)]
            if len(last_5_valid) > 0:
                derived_features[i, 1] = np.mean(last_5_valid)
            else:
                derived_features[i, 1] = 0.0
        except:
            derived_features[i, 1] = 0.0
        
        # 3. Standard deviation of last 15 points - Same as original
        try:
            if len(valid_points) > 1:
                derived_features[i, 2] = np.std(valid_points)
            else:
                derived_features[i, 2] = 0.01
        except:
            derived_features[i, 2] = 0.01
        
        # 4. Second-order coefficient (curvature) - Same as original
        try:
            if len(valid_points) >= 3:
                coeffs = np.polyfit(valid_x, valid_points, 2)
                derived_features[i, 3] = coeffs[0]  # Quadratic coefficient
            else:
                derived_features[i, 3] = 0.0
        except:
            derived_features[i, 3] = 0.0
        
        # 5. Rate of change in last 10 points - Enhanced with relative change
        try:
            if len(valid_points) >= 10:
                first_valid = valid_points[0]
                last_valid = valid_points[-1]
                absolute_change = last_valid - first_valid
                # Calculate relative change to better capture magnitude-independent rates
                if abs(first_valid) > 1e-6:
                    derived_features[i, 4] = absolute_change / abs(first_valid)
                else:
                    derived_features[i, 4] = absolute_change
            else:
                derived_features[i, 4] = 0.0
        except:
            derived_features[i, 4] = 0.0
        
        # 6. Ratio between recent and prior mean - Enhanced with robust calculation
        try:
            recent_valid = X[i, -5:]
            recent_valid = recent_valid[~np.isnan(recent_valid) & ~np.isinf(recent_valid)]
            
            prior_valid = X[i, -15:-5]
            prior_valid = prior_valid[~np.isnan(prior_valid) & ~np.isinf(prior_valid)]
            
            if len(recent_valid) > 0 and len(prior_valid) > 0:
                recent_mean = np.mean(recent_valid)
                prior_mean = np.mean(prior_valid)
                
                # More robust ratio calculation with sigmoid normalization
                if abs(prior_mean) > 1e-6:
                    ratio = recent_mean / prior_mean
                    # Sigmoid-like normalization to handle extreme values
                    derived_features[i, 5] = 2 / (1 + np.exp(-ratio)) - 1
                else:
                    derived_features[i, 5] = 0.0
            else:
                derived_features[i, 5] = 0.0
        except:
            derived_features[i, 5] = 0.0
        
        # 7. Spectral power in different frequency bands
        try:
            if len(full_valid) >= 10:
                # Simple FFT-based spectral analysis
                from scipy.fft import fft
                
                # Apply windowing to reduce spectral leakage
                window = np.hanning(len(full_valid))
                windowed_data = full_valid * window
                
                # Compute FFT
                fft_values = fft(windowed_data)
                fft_magnitude = np.abs(fft_values[:len(fft_values)//2])
                
                # Calculate power in different frequency bands
                if len(fft_magnitude) >= 3:
                    low_freq_power = np.sum(fft_magnitude[:len(fft_magnitude)//3])
                    mid_freq_power = np.sum(fft_magnitude[len(fft_magnitude)//3:2*len(fft_magnitude)//3])
                    high_freq_power = np.sum(fft_magnitude[2*len(fft_magnitude)//3:])
                    
                    # Calculate ratio of high to low frequency power (captures noise vs trend)
                    total_power = low_freq_power + mid_freq_power + high_freq_power
                    if total_power > 1e-6:
                        derived_features[i, 6] = (high_freq_power - low_freq_power) / total_power
                    else:
                        derived_features[i, 6] = 0.0
                else:
                    derived_features[i, 6] = 0.0
            else:
                derived_features[i, 6] = 0.0
        except:
            derived_features[i, 6] = 0.0
        
        # 8. Entropy measure of signal variability (approximate entropy)
        try:
            if len(valid_points) >= 4:
                # Simple approximation of entropy using differences between consecutive points
                # Higher values indicate more randomness, lower values indicate more structure
                diffs = np.diff(valid_points)
                if len(diffs) > 0:
                    # Calculate normalized entropy-like measure using absolute differences
                    abs_diffs = np.abs(diffs)
                    if np.sum(abs_diffs) > 1e-6:
                        probs = abs_diffs / np.sum(abs_diffs)
                        # Higher values for more uniform differences (predictable), lower for more variable (chaotic)
                        non_zero_probs = probs[probs > 0]
                        if len(non_zero_probs) > 0:
                            entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
                            # Normalize by maximum possible entropy for this number of points
                            max_entropy = np.log(len(non_zero_probs))
                            if max_entropy > 0:
                                derived_features[i, 7] = entropy / max_entropy
                            else:
                                derived_features[i, 7] = 0.0
                        else:
                            derived_features[i, 7] = 0.0
                    else:
                        derived_features[i, 7] = 0.0  # No variation
                else:
                    derived_features[i, 7] = 0.0
            else:
                derived_features[i, 7] = 0.5  # Default value
        except:
            derived_features[i, 7] = 0.5  # Default value
        
        # 9. Adaptive trend strength indicator
        try:
            if len(valid_points) >= 5:
                # Calculate linear trend
                slope, intercept = np.polyfit(valid_x, valid_points, 1)
                
                # Generate trend line
                trend_line = slope * valid_x + intercept
                
                # Calculate residuals
                residuals = valid_points - trend_line
                
                # Trend strength is ratio of trend variance to total variance
                residual_var = np.var(residuals) if len(residuals) > 1 else 0
                total_var = np.var(valid_points) if len(valid_points) > 1 else 0
                
                if total_var > 1e-6:
                    # 1 means perfect trend, 0 means no trend
                    trend_strength = 1 - (residual_var / total_var)
                    derived_features[i, 8] = trend_strength
                else:
                    derived_features[i, 8] = 0.0
            else:
                derived_features[i, 8] = 0.0
        except:
            derived_features[i, 8] = 0.0
        
        # 10. Pattern change detection feature
        try:
            if len(valid_points) >= 8:
                # Split the recent points into first and second half
                mid_point = len(valid_points) // 2
                first_half = valid_points[:mid_point]
                second_half = valid_points[mid_point:]
                
                if len(first_half) > 0 and len(second_half) > 0:
                    # Calculate statistics for each half
                    first_mean = np.mean(first_half)
                    second_mean = np.mean(second_half)
                    first_std = np.std(first_half) if len(first_half) > 1 else 0.01
                    second_std = np.std(second_half) if len(second_half) > 1 else 0.01
                    
                    # Calculate various change metrics
                    mean_change = abs(second_mean - first_mean)
                    std_change = abs(second_std - first_std)
                    
                    # Normalize by first half statistics to get relative change
                    mean_base = max(abs(first_mean), 0.01)
                    std_base = max(first_std, 0.01)
                    
                    # Combine into unified change detection metric
                    rel_mean_change = mean_change / mean_base
                    rel_std_change = std_change / std_base
                    
                    # Average of relative changes in mean and std (higher = more change)
                    derived_features[i, 9] = (rel_mean_change + rel_std_change) / 2
                else:
                    derived_features[i, 9] = 0.0
            else:
                derived_features[i, 9] = 0.0
        except:
            derived_features[i, 9] = 0.0
    
    # Check for invalid final values
    if np.isnan(derived_features).any() or np.isinf(derived_features).any():
        derived_features = np.nan_to_num(derived_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return derived_features


def prepare_data(train_df, input_cols, output_cols):
    """
    Prepare data with improved imputation and derived features
    """
    print("Preparing data...")
    
    # Extract matrices
    X = train_df[input_cols].values
    y = train_df[output_cols].values
    
    # Use KNNImputer to better preserve spatial structure
    print("Handling missing values with KNNImputer...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_imputed = imputer.fit_transform(X)
    
    # Check if NaNs or infinites still exist
    if np.isnan(X_imputed).any() or np.isinf(X_imputed).any():
        print("WARNING: NaN or infinite values still exist after imputation!")
        X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create derived features
    print("Creating derived features...")
    derived_features = create_derived_features(X_imputed)
    
    # Combine with original features
    X_enhanced = np.hstack((X_imputed, derived_features))
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_enhanced, y, test_size=0.2, random_state=SEED)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val, derived_features.shape[1]


# Adicionar estas importações ao início do seu notebook
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
from concurrent.futures import ThreadPoolExecutor
import functools

def optimize_hyperparameters(X, y, model_type, param_space, n_iter=20, cv=3):
    """
    Optimize hyperparameters using Bayesian optimization
    
    Args:
        X: Features
        y: Target
        model_type: Type of model ('xgb', 'lgb', 'gbr', 'rf')
        param_space: Dictionary of parameter spaces
        n_iter: Number of iterations for optimization
        cv: Number of cross-validation folds
    
    Returns:
        Best parameters
    """
    if model_type == 'xgb':
        model = xgb.XGBRegressor(random_state=SEED)
    elif model_type == 'lgb':
        model = lgb.LGBMRegressor(random_state=SEED)
    elif model_type == 'gbr':
        model = GradientBoostingRegressor(random_state=SEED)
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=SEED)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create BayesSearchCV object
    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=SEED,
        n_jobs=-1 if model_type != 'xgb' else 1,  # XGBoost with n_jobs can cause issues in some environments
        verbose=0
    )
    
    # Fit the optimizer
    opt.fit(X, y)
    
    # Return best parameters
    return opt.best_params_

class StackingRegressor(BaseEstimator, RegressorMixin):
    """
    Custom stacking regressor with more advanced features than sklearn's StackingRegressor.
    This implementation uses out-of-fold predictions for the meta-model training.
    """
    def __init__(self, base_models, meta_model, n_folds=5, use_features_in_meta=False):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_features_in_meta = use_features_in_meta
        self.base_models_ = None
        self.meta_model_ = None
        self.kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        
    def fit(self, X, y):
        self.base_models_ = [clone(model) for model in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        
        # Generate out-of-fold predictions for training meta-model
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        out_of_fold_preds = np.zeros((n_samples, n_models))
        
        # For each base model, generate out-of-fold predictions
        for i, model in enumerate(self.base_models_):
            oof_pred = np.zeros(n_samples)
            
            # For each fold
            for train_idx, valid_idx in self.kf.split(X):
                # Split data
                X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
                y_train_fold = y[train_idx]
                
                # Train model
                model.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                oof_pred[valid_idx] = model.predict(X_valid_fold)
            
            # Store out-of-fold predictions
            out_of_fold_preds[:, i] = oof_pred
            
            # Retrain on all data
            model.fit(X, y)
        
        # Prepare meta features
        if self.use_features_in_meta:
            meta_features = np.hstack((out_of_fold_preds, X))
        else:
            meta_features = out_of_fold_preds
        
        # Train meta model
        self.meta_model_.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        # Make predictions with base models
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models_
        ])
        
        # Add original features if specified
        if self.use_features_in_meta:
            meta_features = np.hstack((meta_features, X))
        
        # Make final prediction
        return self.meta_model_.predict(meta_features)

def train_model_for_position(pos_idx, X_train_processed, y_train, n_outputs, positions_params=None):
    """
    Train models for a specific position with hyperparameter optimization
    
    Args:
        pos_idx: Position index
        X_train_processed: Processed training features
        y_train: Training targets
        n_outputs: Total number of outputs
        positions_params: Pre-optimized parameters for each position type
    
    Returns:
        Dictionary with trained models and metadata
    """
    pos = pos_idx + 1  # Convert to 1-indexed
    print(f"Training models for position {pos} of {n_outputs}...")
    
    # Extract target for this position
    y_pos = y_train[:, pos_idx]
    
    # Dictionary to store model performance
    model_performances = {}
    
    # Position groups to handle optimization efficiently
    if pos <= 20:
        pos_group = 'early'  # First 20 positions
    elif pos <= 100:
        pos_group = 'mid_early'  # Positions 21-100
    elif pos <= 200:
        pos_group = 'mid_late'  # Positions 101-200
    else:
        pos_group = 'late'  # Positions 201-300
    
    # Use pre-optimized parameters if available
    if positions_params and pos_group in positions_params:
        print(f"  Using pre-optimized parameters for position group: {pos_group}")
        param_set = positions_params[pos_group]
        xgb_params = param_set['xgb']
        lgb_params = param_set['lgb']
        gbr_params = param_set['gbr']
        rf_params = param_set['rf']
    else:
        # Default parameters if optimization wasn't done
        print("  Using default parameters (no optimization data available)")
        
        # XGBoost default params
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': SEED
        }
        
        # LightGBM default params
        lgb_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'random_state': SEED
        }
        
        # GBR default params
        gbr_params = {
            'n_estimators': 150,
            'max_depth': 4,
            'learning_rate': 0.05,
            'loss': 'squared_error',
            'subsample': 0.8,
            'random_state': SEED
        }
        
        # RF default params
        rf_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 5,
            'random_state': SEED
        }
    
    # Ridge and ElasticNet params (less sensitive to tuning)
    ridge_params = {
        'alpha': 1.0,
        'solver': 'auto',
        'random_state': SEED
    }
    
    elastic_params = {
        'alpha': 0.5,
        'l1_ratio': 0.5,
        'random_state': SEED,
        'max_iter': 1000
    }
    
    # Train XGBoost
    try:
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_processed, y_pos)
        xgb_pred = xgb_model.predict(X_train_processed)
        xgb_mse = mean_squared_error(y_pos, xgb_pred)
        model_performances['xgb'] = {'model': xgb_model, 'mse': xgb_mse}
    except Exception as e:
        print(f"  Error training XGBoost: {e}")
    
    # Train LightGBM
    try:
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train_processed, y_pos)
        lgb_pred = lgb_model.predict(X_train_processed)
        lgb_mse = mean_squared_error(y_pos, lgb_pred)
        model_performances['lgb'] = {'model': lgb_model, 'mse': lgb_mse}
    except Exception as e:
        print(f"  Error training LightGBM: {e}")
    
    # Train GBR
    try:
        gbr_model = GradientBoostingRegressor(**gbr_params)
        gbr_model.fit(X_train_processed, y_pos)
        gbr_pred = gbr_model.predict(X_train_processed)
        gbr_mse = mean_squared_error(y_pos, gbr_pred)
        model_performances['gbr'] = {'model': gbr_model, 'mse': gbr_mse}
    except Exception as e:
        print(f"  Error training GBR: {e}")
    
    # Train RF
    try:
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train_processed, y_pos)
        rf_pred = rf_model.predict(X_train_processed)
        rf_mse = mean_squared_error(y_pos, rf_pred)
        model_performances['rf'] = {'model': rf_model, 'mse': rf_mse}
    except Exception as e:
        print(f"  Error training RF: {e}")
    
    # Train Ridge Regression
    try:
        ridge_model = Ridge(**ridge_params)
        ridge_model.fit(X_train_processed, y_pos)
        ridge_pred = ridge_model.predict(X_train_processed)
        ridge_mse = mean_squared_error(y_pos, ridge_pred)
        model_performances['ridge'] = {'model': ridge_model, 'mse': ridge_mse}
    except Exception as e:
        print(f"  Error training Ridge: {e}")
    
    # Train ElasticNet
    try:
        elastic_model = ElasticNet(**elastic_params)
        elastic_model.fit(X_train_processed, y_pos)
        elastic_pred = elastic_model.predict(X_train_processed)
        elastic_mse = mean_squared_error(y_pos, elastic_pred)
        model_performances['elastic'] = {'model': elastic_model, 'mse': elastic_mse}
    except Exception as e:
        print(f"  Error training ElasticNet: {e}")
    
    # Train Stacking model if we have enough base models
    if len(model_performances) >= 3:
        try:
            # Get available models for stacking
            base_models = [info['model'] for info in model_performances.values()]
            
            # Use RidgeCV as meta-learner (more stable than OLS)
            meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
            
            # Create stacking regressor
            stack = StackingRegressor(
                base_models=base_models[:3],  # Use top 3 models to avoid overfitting
                meta_model=meta_learner,
                n_folds=3,
                use_features_in_meta=False
            )
            
            # Train stacking model
            stack.fit(X_train_processed, y_pos)
            stack_pred = stack.predict(X_train_processed)
            stack_mse = mean_squared_error(y_pos, stack_pred)
            model_performances['stack'] = {'model': stack, 'mse': stack_mse}
        except Exception as e:
            print(f"  Error training Stacking model: {e}")
    
    # Calculate weights using more sophisticated weighting scheme
    weights = {}
    # Base weight using inverse MSE
    total_inv_mse = 0
    inv_mse_weights = {}
    
    for model_name, info in model_performances.items():
        inv_mse = 1.0 / (info['mse'] + 1e-6)
        inv_mse_weights[model_name] = inv_mse
        total_inv_mse += inv_mse
    
    # Normalize inverse MSE weights
    for model_name in inv_mse_weights:
        inv_mse_weights[model_name] /= total_inv_mse
    
    # Apply weight adjustments based on model characteristics
    for model_name in model_performances:
        # Start with inverse MSE weight
        weights[model_name] = inv_mse_weights[model_name]
        
        # Boost stacking model weight if it's good
        if model_name == 'stack' and model_performances['stack']['mse'] < min(
            info['mse'] for name, info in model_performances.items() if name != 'stack'
        ):
            weights[model_name] *= 1.5
        
        # Boost tree-based models at the beginning (positions 1-60)
        # Tree models better capture the initial trajectory
        if pos <= 60 and model_name in ['xgb', 'lgb', 'gbr', 'rf']:
            weights[model_name] *= 1.2
        
        # Boost linear models for far positions (positions 200-300)
        # Linear models tend to extrapolate better for far positions
        if pos >= 200 and model_name in ['ridge', 'elastic']:
            weights[model_name] *= 1.2
    
    # Renormalize weights
    total_weight = sum(weights.values())
    for model_name in weights:
        weights[model_name] /= total_weight
    
    # Store models and their weights
    model_data = {
        'models': {name: info['model'] for name, info in model_performances.items()},
        'weights': weights,
        'performances': {name: info['mse'] for name, info in model_performances.items()}
    }
    
    # Print performance summary
    print(f"  Model performances (MSE) and weights:")
    for model_name, info in model_performances.items():
        print(f"    {model_name}: MSE={info['mse']:.4f}, Weight={weights[model_name]:.2f}")
    
    return pos, model_data

def train_advanced_ensemble_models(X_train, y_train, n_derived_features, n_jobs=-1):
    """
    Train an advanced ensemble of models including XGBoost, LightGBM, and stacking
    with parallel training and hyperparameter optimization
    """
    print("Training advanced ensemble with XGBoost, LightGBM, and stacking...")
    
    # Determine number of output models (one for each future position)
    n_outputs = y_train.shape[1]
    
    # Create models for key positions and interpolate intermediates
    key_positions = []
    
    # More granularity at the beginning (where the metric is more sensitive)
    key_positions.extend(list(range(0, 60, 10)))
    key_positions.extend(list(range(60, 240, 30)))
    key_positions.extend(list(range(240, n_outputs, 20)))
    
    if key_positions[-1] != n_outputs - 1:
        key_positions.append(n_outputs - 1)  # Ensure we have the last point
    
    # Dictionary to store models
    models = {}
    
    # Scale the data
    scaler = StandardScaler()
    X_original = X_train[:, :-n_derived_features]  # Separate original features
    X_derived = X_train[:, -n_derived_features:]   # Separate derived features
    
    # Scale only the original features
    X_original_scaled = scaler.fit_transform(X_original)
    
    # Recombine
    X_train_processed = np.hstack((X_original_scaled, X_derived))
    
    # Perform hyperparameter optimization for representative positions
    print("Performing hyperparameter optimization for representative positions...")
    positions_params = {}
    
    # Define parameter spaces for Bayesian optimization
    xgb_param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(3, 6),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'min_child_weight': Integer(1, 5),
        'gamma': Real(0.0, 5.0),
        'reg_alpha': Real(0.0, 1.0),
        'reg_lambda': Real(0.0, 1.0)
    }
    
    lgb_param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(3, 6),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'min_child_samples': Integer(5, 30),
        'reg_alpha': Real(0.0, 1.0),
        'reg_lambda': Real(0.0, 1.0)
    }
    
    gbr_param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(3, 6),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'subsample': Real(0.6, 1.0)
    }
    
    rf_param_space = {
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(3, 8),
        'min_samples_split': Integer(2, 10)
    }
    
    # Representative positions for different parts of the sequence
    rep_positions = [10, 60, 150, 250]  # Early, mid-early, mid-late, late
    position_groups = ['early', 'mid_early', 'mid_late', 'late']
    
    # For each representative position, find optimal hyperparameters
    for rep_pos, pos_group in zip(rep_positions, position_groups):
        print(f"Optimizing hyperparameters for position group: {pos_group} (position {rep_pos})")
        # Get target for this position
        y_pos = y_train[:, rep_pos]
        
        # Optimize XGBoost
        print("  Optimizing XGBoost...")
        xgb_best_params = optimize_hyperparameters(X_train_processed, y_pos, 'xgb', xgb_param_space, n_iter=15)
        
        # Optimize LightGBM
        print("  Optimizing LightGBM...")
        lgb_best_params = optimize_hyperparameters(X_train_processed, y_pos, 'lgb', lgb_param_space, n_iter=15)
        
        # Optimize GradientBoosting
        print("  Optimizing GradientBoosting...")
        gbr_best_params = optimize_hyperparameters(X_train_processed, y_pos, 'gbr', gbr_param_space, n_iter=15)
        
        # Optimize RandomForest
        print("  Optimizing RandomForest...")
        rf_best_params = optimize_hyperparameters(X_train_processed, y_pos, 'rf', rf_param_space, n_iter=15)
        
        # Add random_state to all params
        xgb_best_params['random_state'] = SEED
        lgb_best_params['random_state'] = SEED
        gbr_best_params['random_state'] = SEED
        rf_best_params['random_state'] = SEED
        
        # Store optimized params
        positions_params[pos_group] = {
            'xgb': xgb_best_params,
            'lgb': lgb_best_params,
            'gbr': gbr_best_params,
            'rf': rf_best_params
        }
        
        print(f"  Optimization completed for position {rep_pos}")
    
    # Save optimized parameters
    joblib.dump(positions_params, 'position_params.pkl')
    print("Hyperparameter optimization completed and saved")
    
    # Train models for all key positions (potentially in parallel)
    train_func = functools.partial(
        train_model_for_position,
        X_train_processed=X_train_processed,
        y_train=y_train,
        n_outputs=n_outputs,
        positions_params=positions_params
    )
    
    # Determine parallelization approach
    if n_jobs != 1:
        print(f"Training models in parallel with {n_jobs} workers...")
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            results = list(executor.map(train_func, key_positions))
        
        # Process results
        for pos, model_data in results:
            models[pos] = model_data
    else:
        print("Training models sequentially...")
        for pos_idx in key_positions:
            pos, model_data = train_func(pos_idx)
            models[pos] = model_data
    
    return models, scaler, key_positions, n_derived_features

def predict_with_advanced_ensemble(X, models, scaler, key_positions, n_derived_features):
    """
    Generate predictions using the advanced ensemble and interpolation
    """
    n_samples = X.shape[0]
    n_outputs = 300  # Total number of output positions
    
    # Separate and scale features
    X_original = X[:, :-n_derived_features]
    X_derived = X[:, -n_derived_features:]
    X_original_scaled = scaler.transform(X_original)
    X_processed = np.hstack((X_original_scaled, X_derived))
    
    # Initialize prediction matrix
    predictions = np.zeros((n_samples, n_outputs))
    
    # Generate predictions for key positions
    for pos in key_positions:
        model_pos = pos + 1  # Convert to 1-indexed
        model_info = models[model_pos]
        pos_idx = pos  # Convert to 0-indexed
        
        # Initialize blended predictions for this position
        blended_preds = np.zeros(n_samples)
        
        # Get all models and their weights
        model_dict = model_info['models']
        weight_dict = model_info['weights']
        
        # Generate and blend predictions from all models
        for model_name, model in model_dict.items():
            try:
                # Get predictions from this model
                preds = model.predict(X_processed)
                
                # Add weighted predictions to the blend
                weight = weight_dict[model_name]
                blended_preds += weight * preds
            except Exception as e:
                print(f"Error predicting with {model_name} at position {pos}: {e}")
        
        predictions[:, pos_idx] = blended_preds
    
    # Improved interpolation strategy - adaptive smoothing
    for i in range(n_samples):
        # Known positions (0-indexed)
        x_known = np.array(key_positions)
        # Known values for these positions
        y_known = predictions[i, key_positions]
        
        # All positions (0-indexed)
        x_all = np.arange(n_outputs)
        
        # Analyze the rate of change to determine appropriate smoothing
        if len(y_known) > 2:
            # Calculate approximate derivatives
            diffs = np.abs(np.diff(y_known))
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            
            # Determine smoothing parameter based on variability
            if max_diff > 5 * mean_diff:
                # High variability - use less smoothing to preserve features
                smoothing = 0.05
            elif max_diff > 2 * mean_diff:
                # Moderate variability
                smoothing = 0.1
            else:
                # Low variability - use more smoothing
                smoothing = 0.2
        else:
            # Default smoothing
            smoothing = 0.1
        
        # Use spline for smooth interpolation with adaptive smoothing
        try:
            if len(x_known) > 3:
                # If we have at least 4 points, use cubic spline with adaptive smoothing
                spline = UnivariateSpline(x_known, y_known, k=3, s=smoothing)
                predictions[i, :] = spline(x_all)
            else:
                # Fallback to linear interpolation
                predictions[i, :] = np.interp(x_all, x_known, y_known)
        except:
            # In case of error, use linear interpolation
            predictions[i, :] = np.interp(x_all, x_known, y_known)
    
    return predictions


def calculate_inverse_covariance_vector():
    """
    Calculate the D_T^{-1}(x) vector for all positions from 1 to 300
    """
    inverse_cov_vector = np.zeros(300)
    for x in range(1, 301):
        if 1 <= x <= 60:
            k = 0  # First region (1-60)
        elif 61 <= x <= 244:
            k = 1  # Second region (61-244)
        else:
            k = 2  # Third region (245-300)
        
        # Apply formula: D_T^{-1}(x) = exp(log(x) * a_k + b_k)
        inverse_cov_vector[x-1] = np.exp(np.log(x) * LOG_SLOPES[k] + LOG_OFFSETS[k])
    
    return inverse_cov_vector


def analyze_variance_structure(X_train, y_train, n_derived_features):
    """
    Analyze variance structure in the data to calibrate realization generation
    """
    print("Analyzing variance structure in the data...")
    
    # Use only original features for variance analysis
    X_train_original = X_train[:, :-n_derived_features]
    
    # Number of samples and points
    n_samples = X_train_original.shape[0]
    n_input_points = X_train_original.shape[1]
    n_output_points = y_train.shape[1]
    
    # Calculate variance at each position of input data
    input_variance = np.var(X_train_original, axis=0)
    
    # Calculate variance at each position of output data
    output_variance = np.var(y_train, axis=0)
    
    # Calculate average autocorrelation of known data
    autocorr_dists = []
    
    for sample_idx in range(min(500, n_samples)):  # Limit to avoid being too slow
        # Get data for this sample
        sample = X_train_original[sample_idx]
        
        # Check for invalid values
        if np.isnan(sample).any() or np.isinf(sample).any():
            continue
            
        try:
            # Calculate autocorrelation
            corr = np.correlate(sample, sample, mode='full')
            
            # Normalize for correlation
            corr = corr / np.max(corr)
            
            # Extract positive part (excluding zero)
            half_len = len(corr) // 2
            corr = corr[half_len+1:]
            
            # Store distance where correlation drops to 0.5
            half_corr_dist = np.argmax(corr < 0.5)
            if half_corr_dist > 0:
                autocorr_dists.append(half_corr_dist)
        except:
            continue
    
    # Calculate average autocorrelation distance
    if autocorr_dists:
        mean_autocorr_dist = np.mean(autocorr_dists)
    else:
        mean_autocorr_dist = 5.0  # Default value if we can't calculate
    
    # Calculate variance scale factors based on evaluation metric
    inverse_cov_vector = calculate_inverse_covariance_vector()
    variance_scale_factors = 1.0 / np.sqrt(inverse_cov_vector)
    
    # Normalize to a reasonable range
    variance_scale_factors = variance_scale_factors / np.mean(variance_scale_factors)
    
    # Visualize analysis results
    plt.figure(figsize=(12, 10))
    
    # Plot variance
    plt.subplot(2, 1, 1)
    plt.plot(range(-n_input_points, 0), input_variance, 'b-', label='Input')
    plt.plot(range(1, n_output_points + 1), output_variance, 'r-', label='Output')
    plt.title('Variance by Position')
    plt.xlabel('Position')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    
    # Plot variance scale factors
    plt.subplot(2, 1, 2)
    plt.plot(range(1, 301), variance_scale_factors)
    plt.title('Variance Scale Factors (based on NLL metric)')
    plt.xlabel('Position')
    plt.ylabel('Scale Factor')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('variance_analysis_refined.png')
    print("Refined variance analysis visualization saved as 'variance_analysis_refined.png'")
    
    # Return estimated parameters
    variance_params = {
        'input_variance': input_variance,
        'output_variance': output_variance,
        'mean_autocorr_dist': mean_autocorr_dist,
        'variance_scale_factors': variance_scale_factors
    }
    
    return variance_params


def generate_calibrated_realizations(X, base_predictions, variance_params, n_derived_features, num_realizations=10):
    """
    Generate calibrated realizations based on statistical analysis and base predictions
    """
    # Use only original features for continuity
    X_original = X[:, :-n_derived_features]
    
    n_samples = X_original.shape[0]
    n_outputs = base_predictions.shape[1]  # Should be 300
    all_realizations = np.zeros((n_samples, num_realizations, n_outputs))
    
    # Extract variance parameters
    mean_autocorr_dist = variance_params['mean_autocorr_dist']
    variance_scale_factors = variance_params['variance_scale_factors']
    output_variance = variance_params['output_variance']
    
    # Calculate global factor to adjust NLL closer to zero
    global_variance_factor = 0.8  # Factor adjusted to bring NLL closer to zero
    
    # First realization: will be the base prediction
    print("Generating calibrated realizations...")
    all_realizations[:, 0, :] = base_predictions
    
    # Define more diverse strategies to explore solution space
    strategies = [
        # Name, base scale, correlation length, trend, trend amplitude
        ("high_fidelity", 0.8, mean_autocorr_dist * 0.8, None, 0.0),
        ("low_variance", 0.6, mean_autocorr_dist * 1.2, None, 0.0),
        ("high_variance", 1.0, mean_autocorr_dist * 0.6, None, 0.0),
        ("upward_trend", 0.8, mean_autocorr_dist, "upward", 1.5),
        ("downward_trend", 0.8, mean_autocorr_dist, "downward", 1.5),
        ("late_upward_trend", 0.8, mean_autocorr_dist, "late_upward", 2.0),
        ("late_downward_trend", 0.8, mean_autocorr_dist, "late_downward", 2.0),
        ("short_oscillation", 0.7, mean_autocorr_dist * 0.5, "oscillatory", 1.0),
        ("long_oscillation", 0.7, mean_autocorr_dist * 1.5, "oscillatory", 1.2),
    ]
    
    # For each additional realization
    for r in range(1, num_realizations):
        # Select strategy
        strategy_idx = min(r-1, len(strategies)-1)
        strategy_name, scale_base, corr_length, trend_type, trend_amplitude = strategies[strategy_idx]
        
        print(f"Generating realization {r} with strategy: {strategy_name}")
        
        for i in range(n_samples):
            # Use base prediction
            realization = base_predictions[i].copy()
            
            # Estimate variability based on input data and first positions of output
            try:
                # Get last valid points
                last_points = X_original[i, -10:]
                valid_mask = ~np.isnan(last_points) & ~np.isinf(last_points)
                valid_last_points = last_points[valid_mask]
                
                if len(valid_last_points) > 0:
                    input_std = np.std(valid_last_points)
                else:
                    input_std = 0.1  # Default value if no valid points
            except:
                input_std = 0.1
            
            # Use average output variance of initial output (more reliable)
            output_std_init = np.sqrt(np.mean(output_variance[:20]))
            
            # Use a blend of both
            base_std = (input_std + output_std_init) / 2
            
            # If base_std is too small, use a minimum value
            base_std = max(base_std, 0.01)
            
            # Calculate noise scale
            noise_scale = base_std * scale_base * global_variance_factor
            
            # Generate base noise
            noise = np.random.normal(0, noise_scale, n_outputs)
            
            # Apply scale factors to adjust to NLL metric
            scaled_noise = noise * variance_scale_factors
            
            # Smooth noise to introduce spatial autocorrelation
            smooth_noise = gaussian_filter1d(scaled_noise, sigma=corr_length)
            
            # Apply modifications according to strategy
            if trend_type == "upward":
                # Add upward trend (more pronounced)
                trend_magnitude = base_std * trend_amplitude
                trend = np.linspace(0, trend_magnitude, n_outputs)
                smooth_noise += trend
                
            elif trend_type == "downward":
                # Add downward trend (more pronounced)
                trend_magnitude = base_std * trend_amplitude
                trend = np.linspace(trend_magnitude, 0, n_outputs)
                smooth_noise += trend
                
            elif trend_type == "late_upward":
                # Trend that starts flat and then goes up
                trend_magnitude = base_std * trend_amplitude
                x = np.linspace(0, 1, n_outputs)
                trend = trend_magnitude * (np.exp(3*x) - 1) / (np.exp(3) - 1)
                smooth_noise += trend
                
            elif trend_type == "late_downward":
                # Trend that starts flat and then goes down
                trend_magnitude = base_std * trend_amplitude
                x = np.linspace(0, 1, n_outputs)
                trend = trend_magnitude * (1 - (np.exp(3*x) - 1) / (np.exp(3) - 1))
                smooth_noise += trend
                
            elif trend_type == "oscillatory":
                # Add oscillatory pattern
                n_cycles = 2 if "long" in strategy_name else 4
                cycle_magnitude = base_std * trend_amplitude
                cycles = np.sin(np.linspace(0, n_cycles*np.pi, n_outputs)) * cycle_magnitude
                smooth_noise += cycles
            
            # Add processed noise
            realization += smooth_noise
            
            # Ensure continuity with input data (last point)
            try:
                # Get last valid point from input
                last_valid_idx = -1
                while last_valid_idx >= -X_original.shape[1]:
                    last_point = X_original[i, last_valid_idx]
                    if not np.isnan(last_point) and not np.isinf(last_point):
                        break
                    last_valid_idx -= 1
                
                if last_valid_idx >= -X_original.shape[1]:
                    offset = realization[0] - X_original[i, last_valid_idx]
                    realization -= offset
            except:
                # In case of error, don't adjust offset
                pass
            
            # For some strategies, apply additional smoothing
            if strategy_name in ["high_fidelity", "long_oscillation"]:
                try:
                    # Use Savitzky-Golay filter for more smoothing
                    window_size = min(15, int(corr_length*2) + 1)
                    window_size = window_size + 1 if window_size % 2 == 0 else window_size  # Ensure it's odd
                    realization = savgol_filter(realization, window_length=window_size, polyorder=2)
                except:
                    # In case of error, don't apply additional smoothing
                    pass
            
            # Check if realization contains invalid values
            if np.isnan(realization).any() or np.isinf(realization).any():
                # Replace with base prediction
                realization = base_predictions[i].copy()
            
            # Store realization
            all_realizations[i, r, :] = realization
    
    return all_realizations


def calculate_nll_loss(y_true, predictions):
    """
    Calculate Negative Log Likelihood Loss as defined in the evaluation
    
    Args:
        y_true: Array of true values with shape (n_samples, 300)
        predictions: Array of realizations with shape (n_samples, num_realizations, 300)
    
    Returns:
        Average NLL loss for all samples
    """
    n_samples = y_true.shape[0]
    num_realizations = predictions.shape[1]
    
    # Calculate idealized inverse covariance vector
    inverse_cov_vector = calculate_inverse_covariance_vector()
    
    # Calculate NLL for each sample
    losses = np.zeros(n_samples)
    
    # Probability of each realization (equiprobable)
    p_i = 1.0 / num_realizations
    
    for i in range(n_samples):
        # Array to store gaussian misfit for each realization
        gaussian_misfits = np.zeros(num_realizations)
        
        for r in range(num_realizations):
            # Calculate error vector (residual)
            error_vector = y_true[i] - predictions[i, r]
            
            # Check for invalid values in error
            if np.isnan(error_vector).any() or np.isinf(error_vector).any():
                error_vector = np.nan_to_num(error_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # According to metric documentation:
            # E_i,G = exp(e_i(x) ⋅ D_T^{-1}(x) ⋅ e_i(x))
            weighted_error_sum = np.sum(error_vector**2 * inverse_cov_vector)
            
            # Limit to avoid numerical problems
            weighted_error_sum = np.clip(weighted_error_sum, -700, 700)
            
            # Calculate gaussian misfit
            gaussian_misfits[r] = np.exp(weighted_error_sum)
        
        # Calculate weighted sum of misfits (NLL)
        sum_weighted_misfits = np.sum(p_i * gaussian_misfits)
        
        # Avoid numerical problems with logarithm
        if sum_weighted_misfits > 1e-300:
            losses[i] = -np.log(sum_weighted_misfits)
        else:
            # Maximum value for extreme cases
            losses[i] = 700.0
    
    # Calculate mean and display statistics
    mean_loss = np.mean(losses)
    median_loss = np.median(losses)
    high_loss_count = np.sum(losses > 100)
    
    print(f"NLL Statistics: mean={mean_loss:.2f}, median={median_loss:.2f}, high NLL samples: {high_loss_count}/{n_samples}")
    
    return mean_loss


def calibrate_nll_target(val_realizations, y_val, initial_value=0.8, target_nll=-69):
    """
    Adjust variance scale of realizations to bring NLL closer to target
    """
    print("Calibrating variance for target NLL...")
    
    # Clone original realizations
    scaled_realizations = val_realizations.copy()
    
    # First realization is not altered
    base_predictions = val_realizations[:, 0, :]
    
    # Function to scale alternative realizations
    def scale_realizations(scale_factor):
        for r in range(1, scaled_realizations.shape[1]):
            # Get base and alternative realizations
            base = base_predictions
            alt = val_realizations[:, r, :]
            
            # Calculate difference
            diff = alt - base
            
            # Apply scale factor to difference
            scaled_diff = diff * scale_factor
            
            # Update scaled realization
            scaled_realizations[:, r, :] = base + scaled_diff
        
        # Calculate NLL with scaled realizations
        nll = calculate_nll_loss(y_val, scaled_realizations)
        return nll
    
    # Test different scales and choose the best
    best_scale = initial_value
    best_nll = scale_realizations(initial_value)
    print(f"Initial NLL with scale {initial_value}: {best_nll}")
    
    # Fine-tuning if we're far from target
    scales_to_try = []
    if best_nll < target_nll:  # NLL too negative, need to reduce variance
        scales_to_try = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    else:  # NLL too positive, need to increase variance
        scales_to_try = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for scale in scales_to_try:
        nll = scale_realizations(scale)
        print(f"NLL with scale {scale}: {nll}")
        
        # Check if we're closer to target
        if abs(nll - target_nll) < abs(best_nll - target_nll):
            best_scale = scale
            best_nll = nll
    
    print(f"Best scale: {best_scale}, Resulting NLL: {best_nll}")
    return best_scale, best_nll


def prepare_submission(test_ids, all_realizations, output_path):
    """
    Prepare submission file
    """
    print("Preparing submission file...")
    
    # Create dictionary for DataFrame
    submission_dict = {'geology_id': test_ids}
    
    # Add columns for first realization (1 to 300)
    for i in range(300):
        submission_dict[str(i+1)] = all_realizations[:, 0, i]
    
    # Add columns for other realizations
    for r in range(1, all_realizations.shape[1]):
        for i in range(300):
            submission_dict[f'r_{r}_pos_{i+1}'] = all_realizations[:, r, i]
    
    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_dict)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")


def visualize_examples(X_val, y_val, val_realizations, n_derived_features, num_examples=3):
    """
    Visualize some examples
    """
    print("Visualizing examples...")
    
    # Use only original features
    X_val_original = X_val[:, :-n_derived_features]
    
    # Choose some random examples
    indices = np.random.choice(X_val_original.shape[0], num_examples, replace=False)
    
    plt.figure(figsize=(15, 12))
    for i, idx in enumerate(indices):
        plt.subplot(num_examples, 1, i+1)
        
        # Known data (input)
        plt.plot(range(-49, 1), X_val_original[idx], 'b-', label='Known Data')
        
        # True value (target)
        plt.plot(range(1, 301), y_val[idx], 'g-', label='Ground Truth')
        
        # Multiple realizations (predictions)
        for r in range(val_realizations.shape[1]):
            if r == 0:
                plt.plot(range(1, 301), val_realizations[idx, r, :], 'r-', alpha=0.7, label='Base Prediction')
            else:
                plt.plot(range(1, 301), val_realizations[idx, r, :], 'r-', alpha=0.3)
        
        plt.axvline(x=0, color='k', linestyle='--')
        plt.title(f'Example #{idx}')
        plt.xlabel('Position X')
        plt.ylabel('Coordinate Z')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('validation_predictions_corrected.png')
    print("Visualization saved as 'validation_predictions_corrected.png'")


def main():
    # File paths (adjust as needed)
    train_path = "/kaggle/input/geology-forecast-challenge-open/data/train.csv"
    test_path = "/kaggle/input/geology-forecast-challenge-open/data/test.csv"
    
    # Load data
    train_df, train_ids, input_cols, output_cols, test_df, test_ids = load_data(train_path, test_path)
    
    # Prepare data with derived features
    X_train, X_val, y_train, y_val, n_derived_features = prepare_data(train_df, input_cols, output_cols)
    
    # Analyze data statistics
    variance_params = analyze_variance_structure(X_train, y_train, n_derived_features)
    
    # Train advanced ensemble with XGBoost, LightGBM and stacking
    models, scaler, key_positions, n_derived_features = train_advanced_ensemble_models(X_train, y_train, n_derived_features)
    
    # Generate base predictions for validation using advanced ensemble
    val_base_predictions = predict_with_advanced_ensemble(X_val, models, scaler, key_positions, n_derived_features)
    
    # Generate calibrated realizations for validation
    val_realizations = generate_calibrated_realizations(X_val, val_base_predictions, variance_params, n_derived_features, NUM_REALIZATIONS)
    
    # Calculate initial NLL
    nll_loss = calculate_nll_loss(y_val, val_realizations)
    print(f"Initial Negative Log Likelihood (NLL): {nll_loss}")
    
    # Calibrate realizations for a target NLL
    best_scale, best_nll = calibrate_nll_target(val_realizations, y_val, initial_value=0.8, target_nll=-69)
    
    # Visualize some examples
    visualize_examples(X_val, y_val, val_realizations, n_derived_features)
    
    # Prepare test data
    print("Preparing test data...")
    X_test = test_df[input_cols].values
    
    # Handle missing values in test set
    print("Handling missing values in test set...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_test_imputed = imputer.fit_transform(X_test)
    
    # Check and fix residual NaN or infinite values
    if np.isnan(X_test_imputed).any() or np.isinf(X_test_imputed).any():
        print("Fixing residual NaN/Inf values...")
        X_test_imputed = np.nan_to_num(X_test_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create derived features for test set robustly
    print("Creating derived features for test set...")
    test_derived_features = create_derived_features(X_test_imputed)
    X_test_enhanced = np.hstack((X_test_imputed, test_derived_features))
    
    # Generate base predictions for test using advanced ensemble
    test_base_predictions = predict_with_advanced_ensemble(X_test_enhanced, models, scaler, key_positions, n_derived_features)
    
    # Generate realizations for test set
    test_realizations = generate_calibrated_realizations(X_test_enhanced, test_base_predictions, variance_params, n_derived_features, NUM_REALIZATIONS)
    
    # Apply best scale found during calibration
    print(f"Applying calibrated variance scale: {best_scale}")
    for r in range(1, NUM_REALIZATIONS):
        # Get base and alternative realizations
        base = test_realizations[:, 0, :]
        alt = test_realizations[:, r, :]
        
        # Calculate difference
        diff = alt - base
        
        # Apply scale factor to difference
        scaled_diff = diff * best_scale
        
        # Update scaled realization
        test_realizations[:, r, :] = base + scaled_diff
    
    # Prepare and save submission file
    prepare_submission(test_ids, test_realizations, "submission_advanced_ensemble.csv")
    
    print(f"Process completed successfully! Calibrated NLL: {best_nll}")

if __name__ == "__main__":
    # Start timer to measure total execution time
    import time
    start_time = time.time()
    
    try:
        # Install required packages if needed
        try:
            import xgboost
            import lightgbm
            import skopt
        except ImportError:
            print("Installing required packages...")
            import subprocess
            
            # Install packages using pip
            packages = ["xgboost", "lightgbm", "scikit-optimize"]
            for package in packages:
                print(f"Installing {package}...")
                subprocess.check_call(["pip", "install", package, "--quiet"])
            
            # Import after installation
            import xgboost
            import lightgbm
            import skopt
            print("All required packages installed successfully!")
        
        # Run the main function
        main()
        
        # Calculate and display execution time
        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        
        # Try to give helpful error information
        import traceback
        traceback.print_exc()
        
        # Calculate execution time even if there was an error
        execution_time = time.time() - start_time
        print(f"Execution terminated after {execution_time:.2f} seconds")

