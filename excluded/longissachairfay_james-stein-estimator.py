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


# Install required packages
!pip install feature-engine category_encoders catboost shap optuna scikit-learn xgboost lightgbm -q

# Memory-Optimized DRW Crypto Market Prediction Pipeline with James-Stein Encoding
# Optimized for Kaggle's memory constraints while maintaining prediction quality

import numpy as np
import pandas as pd
import os
import sys
import warnings
import gc
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union

# Core ML libraries
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

# Boosting models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Statistical and feature engineering
from scipy.stats import pearsonr, spearmanr
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import YeoJohnsonTransformer
from category_encoders import JamesSteinEncoder

# Optimization and analysis
import optuna
from optuna.samplers import TPESampler
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Configure environment
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)

# Force garbage collection
gc.collect()

# ============================================================================================
# Memory-Optimized Configuration
# ============================================================================================

class Config:
    """Centralized configuration optimized for memory efficiency"""
    
    # File paths
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Feature groups
    MARKET_FEATURES = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    
    # Core anonymous features identified through analysis (reduced set)
    KEY_X_FEATURES = [
        'X287', 'X446', 'X66', 'X123', 'X385', 'X594', 'X25', 'X3',
        'X37', 'X174', 'X298', 'X168', 'X1', 'X76', 'X21', 'X19'
    ]
    
    # Features to discretize for James-Stein encoding
    FEATURES_TO_DISCRETIZE = ['volume', 'buy_qty', 'sell_qty']  # Reduced set
    N_BINS = 8  # Reduced bins for memory efficiency
    
    # Model parameters
    LABEL_COLUMN = "label"
    RANDOM_STATE = 42
    N_FOLDS = 3  # Reduced folds for memory
    VALIDATION_SIZE = 0.15
    GAP_SIZE = 0.02
    
    # Memory-optimized parameters
    INITIAL_FEATURE_REDUCTION = 300  # Aggressively reduce features early
    VARIANCE_THRESHOLD = 0.01  # Higher threshold to remove more features
    N_SHAP_FEATURES = 50  # Reduced final features
    SHAP_SAMPLE_SIZE = 2000  # Smaller sample for SHAP
    USE_FLOAT32 = True  # Use float32 instead of float64
    CHUNK_SIZE = 50000  # Process data in chunks
    
    # Simplified ensemble
    ENSEMBLE_WEIGHTS = {
        'xgboost': 1.0  # Use only XGBoost to save memory
    }

# ============================================================================================
# Memory-Efficient Data Type Optimization
# ============================================================================================

def optimize_dtypes(df):
    """Convert data types to save memory"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if Config.USE_FLOAT32:
                    df[col] = df[col].astype(np.float32)
                    
    return df

# ============================================================================================
# Early Feature Reduction
# ============================================================================================

def reduce_features_early(df, target=None, n_features=300):
    """Aggressively reduce features early in the pipeline"""
    
    # First, remove constant features
    constant_filter = VarianceThreshold(threshold=0)
    df_filtered = constant_filter.fit_transform(df)
    kept_cols = df.columns[constant_filter.get_support()].tolist()
    df = df[kept_cols]
    
    logging.info(f"Removed {len(df.columns) - len(kept_cols)} constant features")
    
    # If we still have too many features and have a target, use correlation
    if len(df.columns) > n_features and target is not None:
        correlations = df.corrwith(target).abs()
        top_features = correlations.nlargest(n_features).index.tolist()
        df = df[top_features]
        logging.info(f"Reduced to top {n_features} correlated features")
    
    return df

# ============================================================================================
# Memory-Efficient James-Stein Encoder
# ============================================================================================

class MemoryEfficientJamesSteinEncoder:
    """Memory-optimized James-Stein encoder"""
    
    def __init__(self, categorical_features, min_samples_leaf=20, smoothing=1.0):
        self.categorical_features = categorical_features
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.global_mean = None
        self.encoders = {}
        
    def fit(self, X, y):
        """Fit the encoder on training data"""
        if not self.categorical_features:
            return self
        
        self.global_mean = float(y.mean())
        
        # Convert y to Series if needed
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
        
        # Process each categorical feature
        for col in self.categorical_features:
            if col in X.columns:
                try:
                    encoder = JamesSteinEncoder(
                        cols=[col],
                        return_df=True,
                        handle_missing='value',
                        handle_unknown='value',
                        model='independent',
                        randomized=True,
                        sigma=self.smoothing
                    )
                    
                    encoder.fit(X[[col]], y)
                    self.encoders[col] = encoder
                    
                    logging.info(f"Fitted encoder for {col}")
                except Exception as e:
                    logging.warning(f"Failed to fit encoder for {col}: {str(e)}")
        
        return self
    
    def transform(self, X):
        """Transform features using fitted encoders"""
        X_transformed = X.copy()
        
        for col in self.categorical_features:
            if col in X.columns and col in self.encoders:
                try:
                    encoded_values = self.encoders[col].transform(X[[col]])
                    encoded_col_name = f'{col}_encoded'
                    X_transformed[encoded_col_name] = encoded_values[col].astype(np.float32)
                    X_transformed = X_transformed.drop(columns=[col])
                except Exception as e:
                    logging.warning(f"Failed to transform {col}: {str(e)}")
        
        # Fill missing values
        if self.global_mean is not None:
            X_transformed = X_transformed.fillna(self.global_mean)
        
        return X_transformed

# ============================================================================================
# Simplified Feature Engineering
# ============================================================================================

def create_essential_features(df):
    """Create only the most essential engineered features"""
    
    # Order flow imbalance
    if all(col in df.columns for col in ['buy_qty', 'sell_qty', 'volume']):
        df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
        df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market depth
    if all(col in df.columns for col in ['bid_qty', 'ask_qty']):
        df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    # Log volume
    if 'volume' in df.columns:
        df['log_volume'] = np.log1p(df['volume'])
    
    # Convert to float32
    if Config.USE_FLOAT32:
        float_cols = df.select_dtypes(include=[np.float64]).columns
        df[float_cols] = df[float_cols].astype(np.float32)
    
    return df

def create_categorical_features_memory_efficient(df, features_to_discretize, n_bins=8):
    """Memory-efficient categorical feature creation"""
    categorical_features = []
    
    for feature in features_to_discretize:
        if feature in df.columns:
            try:
                # Use pandas qcut for memory efficiency
                cat_feature_name = f'{feature}_cat'
                df[cat_feature_name] = pd.qcut(df[feature], q=n_bins, labels=False, duplicates='drop')
                df[cat_feature_name] = df[cat_feature_name].fillna(-1).astype(np.int8)
                categorical_features.append(cat_feature_name)
                
            except Exception as e:
                logging.warning(f"Failed to discretize {feature}: {str(e)}")
    
    return df, categorical_features

# ============================================================================================
# Lightweight Feature Selection
# ============================================================================================

class LightweightFeatureSelector:
    """Memory-efficient feature selection using simple methods"""
    
    def __init__(self, n_features=50):
        self.n_features = n_features
        self.selected_features = None
        
    def fit(self, X, y):
        """Select features using correlation and variance"""
        
        # Remove low variance features
        variance_filter = VarianceThreshold(threshold=Config.VARIANCE_THRESHOLD)
        X_filtered = variance_filter.fit_transform(X)
        kept_features = X.columns[variance_filter.get_support()].tolist()
        
        # Use SelectKBest for efficiency
        selector = SelectKBest(f_regression, k=min(self.n_features, len(kept_features)))
        selector.fit(X_filtered, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.selected_features = [kept_features[i] for i in range(len(kept_features)) if selected_mask[i]]
        
        logging.info(f"Selected {len(self.selected_features)} features")
        
        return self
    
    def transform(self, X):
        """Transform to selected features"""
        return X[self.selected_features]

# ============================================================================================
# Memory-Efficient Model Training
# ============================================================================================

def train_model_memory_efficient(X_train, y_train, X_val, y_val, X_test, categorical_features):
    """Memory-efficient model training pipeline"""
    
    # Apply encoding
    encoder = MemoryEfficientJamesSteinEncoder(categorical_features=categorical_features)
    X_train_encoded = encoder.fit(X_train, y_train).transform(X_train)
    X_val_encoded = encoder.transform(X_val)
    X_test_encoded = encoder.transform(X_test)
    
    # Clear original data
    del X_train, X_val, X_test
    gc.collect()
    
    # Feature selection
    selector = LightweightFeatureSelector(n_features=Config.N_SHAP_FEATURES)
    X_train_selected = selector.fit(X_train_encoded, y_train).transform(X_train_encoded)
    X_val_selected = selector.transform(X_val_encoded)
    X_test_selected = selector.transform(X_test_encoded)
    
    # Clear encoded data
    del X_train_encoded, X_val_encoded, X_test_encoded
    gc.collect()
    
    # Simple preprocessing
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train model with conservative parameters
    model = XGBRegressor(
        n_estimators=500,  # Reduced trees
        max_depth=4,  # Shallower trees
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=10,
        reg_lambda=10,
        random_state=Config.RANDOM_STATE,
        n_jobs=1,  # Single thread to save memory
        verbosity=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Generate predictions
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate score
    val_score = pearsonr(y_val, val_pred)[0]
    
    return val_pred, test_pred, val_score

# ============================================================================================
# Main Training Function with Memory Management
# ============================================================================================

def train_and_predict_memory_efficient(train_df, test_df):
    """Memory-efficient training pipeline"""
    
    logging.info("Starting memory-optimized pipeline")
    logging.info(f"Initial train shape: {train_df.shape}, test shape: {test_df.shape}")
    
    # Optimize data types
    train_df = optimize_dtypes(train_df)
    test_df = optimize_dtypes(test_df)
    gc.collect()
    
    # Extract target
    y_train = train_df[Config.LABEL_COLUMN].values
    
    # Early feature reduction for X features
    x_cols = [col for col in train_df.columns if col.startswith('X')]
    other_cols = [col for col in train_df.columns if not col.startswith('X') and col != Config.LABEL_COLUMN]
    
    # Reduce X features
    if len(x_cols) > Config.INITIAL_FEATURE_REDUCTION:
        x_train = train_df[x_cols]
        x_train = reduce_features_early(x_train, pd.Series(y_train), Config.INITIAL_FEATURE_REDUCTION)
        reduced_x_cols = x_train.columns.tolist()
        
        # Update dataframes
        train_df = pd.concat([train_df[other_cols + [Config.LABEL_COLUMN]], x_train], axis=1)
        test_df = pd.concat([test_df[other_cols], test_df[reduced_x_cols]], axis=1)
        
        del x_train
        gc.collect()
    
    # Create essential features
    train_df = create_essential_features(train_df)
    test_df = create_essential_features(test_df)
    
    # Create categorical features
    train_df, categorical_features = create_categorical_features_memory_efficient(
        train_df, Config.FEATURES_TO_DISCRETIZE, Config.N_BINS
    )
    test_df, _ = create_categorical_features_memory_efficient(
        test_df, Config.FEATURES_TO_DISCRETIZE, Config.N_BINS
    )
    
    # Prepare features
    feature_cols = [col for col in train_df.columns 
                   if col not in ['timestamp', 'ID', Config.LABEL_COLUMN]]
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    logging.info(f"Features after engineering: {len(feature_cols)}")
    
    # Simple time series split
    n_samples = len(X_train)
    val_size = int(Config.VALIDATION_SIZE * n_samples)
    
    # Use only one validation split to save memory
    train_end = n_samples - val_size
    
    X_train_fold = X_train.iloc[:train_end]
    y_train_fold = y_train[:train_end]
    X_val_fold = X_train.iloc[train_end:]
    y_val_fold = y_train[train_end:]
    
    # Train model
    logging.info("Training model...")
    val_pred, test_pred, val_score = train_model_memory_efficient(
        X_train_fold, y_train_fold,
        X_val_fold, y_val_fold,
        X_test,
        categorical_features
    )
    
    logging.info(f"Validation score: {val_score:.4f}")
    
    # Clean up
    del X_train, X_train_fold, X_val_fold
    gc.collect()
    
    return test_pred, val_score

# ============================================================================================
# Main Execution
# ============================================================================================

def main():
    """Main execution function with memory management"""
    try:
        # Load data
        logging.info("Loading data...")
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)
        submission_df = pd.read_csv(Config.SUBMISSION_PATH)
        
        # Validate data
        assert Config.LABEL_COLUMN in train_df.columns, f"Label column {Config.LABEL_COLUMN} not found"
        
        # Remove missing labels
        train_df = train_df.dropna(subset=[Config.LABEL_COLUMN])
        
        # Train and predict
        test_predictions, val_score = train_and_predict_memory_efficient(train_df, test_df)
        
        # Create submission
        submission_df['prediction'] = test_predictions
        submission_df.to_csv('submission_memory_optimized.csv', index=False)
        
        logging.info(f"\nValidation Score: {val_score:.4f}")
        logging.info("Submission saved as submission_memory_optimized.csv")
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        # Final cleanup
        gc.collect()

if __name__ == "__main__":
    main()

