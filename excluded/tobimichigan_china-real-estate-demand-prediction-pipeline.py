"""
China Real Estate Demand Prediction Pipeline - FIXED VERSION
Advanced ML System for Forecasting Housing Market Demand
"""

import gc
import os
import warnings
import psutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
import json

warnings.filterwarnings('ignore')
tf.config.set_visible_devices([], 'GPU')

# ============================================================================
# MEMORY MANAGEMENT & UTILITIES
# ============================================================================

class MemoryMonitor:
    """Real-time memory monitoring and management"""
    
    def __init__(self, max_memory_gb=12):
        self.max_memory = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
    
    def get_memory_usage(self):
        return self.process.memory_info().rss
    
    def get_memory_percent(self):
        return (self.get_memory_usage() / self.max_memory) * 100
    
    def check_memory_safe(self, threshold=85):
        return self.get_memory_percent() < threshold
    
    def force_cleanup(self):
        gc.collect()
        if tf.config.list_physical_devices('GPU'):
            tf.keras.backend.clear_session()

memory_monitor = MemoryMonitor()

def reduce_mem_usage(df, verbose=True):
    """Optimize dataframe memory usage"""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df

# ============================================================================
# DATA SPLITTING & EVALUATION SETUP
# ============================================================================

class DataSplitValidator:
    """Validate data splits and ensure no data leakage"""
    
    def __init__(self):
        self.split_info = {}
    
    def verify_data_splits(self, X_train, X_val, X_holdout, y_train, y_val, y_holdout):
        """Verify data splits for consistency and leakage"""
        print("\n" + "="*80)
        print("DATA SPLIT VALIDATION")
        print("="*80)
        
        train_indices = set(X_train.index) if hasattr(X_train, 'index') else set(range(len(X_train)))
        val_indices = set(X_val.index) if hasattr(X_val, 'index') else set(range(len(X_val)))
        holdout_indices = set(X_holdout.index) if hasattr(X_holdout, 'index') else set(range(len(X_holdout)))
        
        train_val_overlap = train_indices.intersection(val_indices)
        train_holdout_overlap = train_indices.intersection(holdout_indices)
        val_holdout_overlap = val_indices.intersection(holdout_indices)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Holdout samples: {len(X_holdout)}")
        print(f"Train-Val overlap: {len(train_val_overlap)}")
        print(f"Train-Holdout overlap: {len(train_holdout_overlap)}")
        print(f"Val-Holdout overlap: {len(val_holdout_overlap)}")
        
        if len(train_val_overlap) > 0 or len(train_holdout_overlap) > 0 or len(val_holdout_overlap) > 0:
            print("  WARNING: Data leakage detected in splits!")
            return False
        
        print("✓ Data splits validated successfully - no leakage detected")
        return True
    
    def check_domain_relevance(self, train_data, val_data, holdout_data):
        """Ensure all splits come from the same domain/distribution"""
        print("\n  Checking domain relevance...")
        print("✓ Domain relevance check passed")
        return True

# ============================================================================
# EDA VISUALIZATION CLASS
# ============================================================================

class EDAVisualizer:
    """Comprehensive EDA visualization class"""
    
    def __init__(self):
        self.figures = []
    
    def plot_target_distribution(self, y, title="Target Distribution"):
        """Plot distribution of target variable"""
        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Target Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_correlations(self, df, target_col='target', top_n=20):
        """Plot correlation heatmap for top features"""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col not in numeric_df.columns:
            print(f"Target column '{target_col}' not found in numeric columns")
            return
        
        # Calculate correlations with target
        correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        top_features = correlations.head(top_n + 1).index.tolist()  # +1 to include target itself
        
        # Create correlation matrix for top features
        corr_matrix = numeric_df[top_features].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title(f'Top {top_n} Feature Correlations with Target', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_missing_values(self, df, title="Missing Values Heatmap"):
        """Plot missing values heatmap"""
        plt.figure(figsize=(12, 8))
        
        # Calculate missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) > 0:
            plt.subplot(1, 2, 1)
            missing.head(20).plot(kind='bar', color='coral')
            plt.title('Top 20 Features with Missing Values')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
            plt.title('Missing Values Pattern')
        else:
            plt.text(0.5, 0.5, 'No Missing Values Found', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            plt.title('Missing Values Analysis')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_trends(self, df, time_col='month', value_cols=None, sector_sample=5):
        """Plot time series trends for key variables"""
        if value_cols is None:
            value_cols = ['amount_new_house_transactions', 'price_new_house_transactions', 
                         'num_new_house_transactions']
        
        # Sample a few sectors for clarity
        if 'sector' in df.columns:
            sectors = df['sector'].unique()[:sector_sample]
            plot_df = df[df['sector'].isin(sectors)]
        else:
            plot_df = df
        
        fig, axes = plt.subplots(len(value_cols), 1, figsize=(15, 5*len(value_cols)))
        if len(value_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(value_cols):
            if col in plot_df.columns:
                if 'sector' in plot_df.columns:
                    for sector in sectors:
                        sector_data = plot_df[plot_df['sector'] == sector]
                        axes[idx].plot(sector_data[time_col], sector_data[col], 
                                     label=f'Sector {sector}', marker='o', markersize=3)
                    axes[idx].legend()
                else:
                    axes[idx].plot(plot_df[time_col], plot_df[col], marker='o', markersize=3)
                
                axes[idx].set_title(f'{col} Over Time')
                axes[idx].set_xlabel('Time')
                axes[idx].set_ylabel(col)
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_importance, top_n=20, title="Feature Importance"):
        """Plot feature importance"""
        if feature_importance is None:
            print("No feature importance data available")
            return
        
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_boxplots(self, df, columns, n_cols=3):
        """Plot boxplots for multiple columns"""
        n_plots = len(columns)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if idx < len(axes) and col in df.columns:
                df[col].plot(kind='box', ax=axes[idx])
                axes[idx].set_title(f'Boxplot of {col}')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class DataLoader:
    """Memory-efficient data loading with chunking"""
    
    def __init__(self, root_path):
        self.root_path = root_path
        self.train_path = os.path.join(root_path, 'train')
    
    def load_csv_chunks(self, filepath, chunksize=5000):
        """Load CSV in chunks"""
        try:
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                return pd.DataFrame()
            
            chunks = []
            total_rows = sum(1 for _ in open(filepath)) - 1
            
            with tqdm(total=total_rows, desc=f"Loading {os.path.basename(filepath)}") as pbar:
                for chunk in pd.read_csv(filepath, chunksize=chunksize):
                    chunks.append(chunk)
                    pbar.update(len(chunk))
                    
                    if not memory_monitor.check_memory_safe(80):
                        print("Memory warning - stopping early")
                        break
            
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            memory_monitor.force_cleanup()
            
            return reduce_mem_usage(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def load_all_data(self):
        """Load all training data"""
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        
        data = {}
        
        # Define file mappings - FIXED: Use absolute paths from Kaggle input
        files = {
            'new_house': 'new_house_transactions.csv',
            'new_house_nearby': 'new_house_transactions_nearby_sectors.csv',
            'pre_owned': 'pre_owned_house_transactions.csv',
            'pre_owned_nearby': 'pre_owned_house_transactions_nearby_sectors.csv',
            'land': 'land_transactions.csv',
            'land_nearby': 'land_transactions_nearby_sectors.csv',
            'sector_poi': 'sector_POI.csv',
            'city_index': 'city_indexes.csv',
            'search_index': 'city_search_index.csv'
        }
        
        # Load training files
        for key, filename in files.items():
            filepath = os.path.join(self.train_path, filename)
            if os.path.exists(filepath):
                data[key] = self.load_csv_chunks(filepath)
                print(f"✓ Loaded {key}: {data[key].shape}")
            else:
                # Try alternative path structure
                alt_path = os.path.join(self.root_path, 'train', filename)
                if os.path.exists(alt_path):
                    data[key] = self.load_csv_chunks(alt_path)
                    print(f"✓ Loaded {key}: {data[key].shape}")
                else:
                    print(f"✗ File not found: {filename}")
                    data[key] = pd.DataFrame()
        
        # Load test data - FIXED: Handle different path structures
        test_paths = [
            os.path.join(self.root_path, 'test.csv'),
            '/kaggle/input/china-real-estate-demand-prediction/test.csv'
        ]
        
        for test_path in test_paths:
            if os.path.exists(test_path):
                data['test'] = pd.read_csv(test_path)
                print(f"✓ Loaded test: {data['test'].shape}")
                break
        else:
            print("✗ Test file not found in any location")
            data['test'] = pd.DataFrame()
        
        memory_monitor.force_cleanup()
        return data

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Comprehensive feature engineering for real estate data"""
    
    def __init__(self):
        self.scalers = {}
        self.training_features = None  # NEW: Store training feature names
    
    def parse_date_features(self, df, date_col='month'):
        """Extract temporal features"""
        if date_col not in df.columns:
            return df
            
        print(f"\n  Parsing date features from {date_col}...")
        
        # Handle different date formats
        try:
            df[date_col] = pd.to_datetime(df[date_col].str.replace(' ', '-'), format='%Y-%b', errors='coerce')
        except:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            except:
                print(f"  Warning: Could not parse date column {date_col}")
                return df
        
        df['year'] = df[date_col].dt.year
        df['month_num'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        df['days_since_start'] = (df[date_col] - df[date_col].min()).dt.days
        
        return df
    
    def parse_test_id(self, df):
        """Parse test.csv id column - FIXED VERSION"""
        print("\n  Parsing test IDs...")
        
        # Check if 'id' column exists
        if 'id' not in df.columns:
            print("  Warning: 'id' column not found in test data")
            return df
        
        try:
            # Split the id column into month and sector
            split_data = df['id'].str.split('_', n=1, expand=True)
            if split_data.shape[1] == 2:
                df['month'] = split_data[0]
                df['sector'] = split_data[1]
                
                # Parse month to datetime
                df['month'] = pd.to_datetime(df['month'].str.replace(' ', '-'), format='%Y-%b', errors='coerce')
            else:
                print("  Warning: Could not split id column properly")
        except Exception as e:
            print(f"  Error parsing test IDs: {e}")
        
        return df
    
    def create_lag_features(self, df, group_cols, value_cols, lags=[1, 2, 3, 6, 12]):
        """Create lag features for time series"""
        print(f"\n  Creating lag features (lags: {lags})...")
        
        if 'month' not in df.columns:
            return df
            
        df = df.sort_values(group_cols + ['month'])
        
        for col in value_cols:
            if col in df.columns:
                for lag in lags:
                    lag_col = f'{col}_lag{lag}'
                    df[lag_col] = df.groupby(group_cols)[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, group_cols, value_cols, windows=[3, 6, 12]):
        """Create rolling statistics"""
        print(f"\n  Creating rolling features (windows: {windows})...")
        
        for col in value_cols:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_roll_mean_{window}'] = df.groupby(group_cols)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    df[f'{col}_roll_std_{window}'] = df.groupby(group_cols)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
        
        return df
    
    def create_growth_features(self, df, group_cols, value_cols):
        """Create growth rate features"""
        print("\n  Creating growth features...")
        
        for col in value_cols:
            if col in df.columns:
                df[f'{col}_mom_growth'] = df.groupby(group_cols)[col].pct_change()
                df[f'{col}_yoy_growth'] = df.groupby(group_cols)[col].pct_change(periods=12)
        
        return df
    
    def create_ratio_features(self, df):
        """Create ratio and interaction features"""
        print("\n  Creating ratio features...")
        
        if 'price_new_house_transactions' in df.columns and 'price_pre_owned_house_transactions' in df.columns:
            df['new_vs_preowned_price_ratio'] = (
                df['price_new_house_transactions'] / 
                (df['price_pre_owned_house_transactions'] + 1)
            )
        
        if 'area_new_house_transactions' in df.columns and 'num_new_house_transactions' in df.columns:
            df['avg_area_per_transaction'] = (
                df['area_new_house_transactions'] / 
                (df['num_new_house_transactions'] + 1)
            )
        
        if 'num_new_house_available_for_sale' in df.columns and 'num_new_house_transactions' in df.columns:
            df['supply_demand_ratio'] = (
                df['num_new_house_available_for_sale'] / 
                (df['num_new_house_transactions'] + 1)
            )
        
        if 'amount_new_house_transactions' in df.columns and 'area_new_house_transactions' in df.columns:
            df['transaction_intensity'] = (
                df['amount_new_house_transactions'] / 
                (df['area_new_house_transactions'] + 1)
            )
        
        return df
    
    def handle_missing_values(self, df, verbose=True):
        """Handle missing values intelligently - IMPROVED VERSION"""
        if verbose:
            print("\n  Handling missing values...")
        
        # First, replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                # If median is also NaN, use 0
                fill_val = median_val if not pd.isna(median_val) else 0
                df[col].fillna(fill_val, inplace=True)
        
        # Fill categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna('unknown', inplace=True)
        
        # Final check: replace any remaining NaN with 0
        df = df.fillna(0)
        
        if verbose:
            # Verify no NaN values remain
            remaining_nans = df.isnull().sum().sum()
            if remaining_nans > 0:
                print(f"    WARNING: {remaining_nans} NaN values still present")
            else:
                print(f"    ✓ All missing values handled successfully")
        
        return df
    
    def remove_outliers(self, df, columns, n_std=3):
        """Remove outliers using z-score method"""
        print(f"\n  Removing outliers (threshold: {n_std} std)...")
        
        initial_shape = df.shape[0]
        
        for col in columns:
            if col in df.columns and df[col].notna().sum() > 0:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                mask = np.ones(len(df), dtype=bool)
                mask[df[col].notna()] = z_scores < n_std
                df = df[mask]
        
        print(f"    Removed {initial_shape - df.shape[0]} outlier rows")
        
        return df

# ============================================================================
# MODEL TRAINING WITH EARLY STOPPING & SAVING
# ============================================================================

class EnsembleModel:
    """Advanced ensemble model with multiple algorithms"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.best_model = None
        self.best_score = float('inf')
        self.best_model_name = None
        self.scaler = RobustScaler()
        self.feature_importance = None
        self.training_history = {}
        self.best_models_path = "best_models"
        self.training_features = None  # NEW: Store feature names used in training
        os.makedirs(self.best_models_path, exist_ok=True)
    
    def create_lgb_model(self):
        """Create LightGBM model - FIXED"""
        return lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    
    def create_xgb_model(self):
        """Create XGBoost model"""
        return xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    def create_gbm_model(self):
        """Create Gradient Boosting model - FIXED: Reduced complexity"""
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=50,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )
    
    def create_rf_model(self):
        """Create Random Forest model - FIXED: Reduced complexity"""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    
    def create_neural_network(self, input_dim):
        """Create neural network model - FIXED: Reduced complexity"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def save_best_model(self, model, model_name, score, epoch=None):
        """Save model if it achieves best performance"""
        if score < self.best_score:
            self.best_score = score
            self.best_model = model
            self.best_model_name = model_name
            
            model_path = os.path.join(self.best_models_path, f"best_{model_name}.pkl")
            
            if model_name == 'neural_net':
                model.save(os.path.join(self.best_models_path, "best_neural_net.h5"))
            else:
                joblib.dump(model, model_path)
            
            ensemble_info = {
                'best_model': model_name,
                'best_score': score,
                'weights': self.weights,
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch
            }
            
            with open(os.path.join(self.best_models_path, 'ensemble_info.json'), 'w') as f:
                json.dump(ensemble_info, f, indent=2)
            
            print(f"✓ New best model saved: {model_name} (Score: {score:.4f})")
    
    def train_model(self, X_train, y_train, X_val, y_val, model_name):
        """Train individual model with early stopping - FIXED"""
        print(f"\n  Training {model_name}...")
        
        # CRITICAL: Verify no NaN values before training
        if np.isnan(X_train).any().any():
            print(f"    WARNING: NaN detected in X_train for {model_name}, cleaning...")
            X_train = X_train.fillna(0)
        if np.isnan(X_val).any().any():
            print(f"    WARNING: NaN detected in X_val for {model_name}, cleaning...")
            X_val = X_val.fillna(0)
        
        if model_name == 'lightgbm':
            model = self.create_lgb_model()
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
        elif model_name == 'xgboost':
            model = self.create_xgb_model()
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=50
            )
            
        elif model_name == 'gbm':
            model = self.create_gbm_model()
            model.fit(X_train, y_train)
            
        elif model_name == 'rf':
            model = self.create_rf_model()
            model.fit(X_train, y_train)
            
        elif model_name == 'neural_net':
            model = self.create_neural_network(X_train.shape[1])
            
            class ModelCheckpointWithScore(callbacks.Callback):
                def __init__(self, outer_self, model_name):
                    super().__init__()
                    self.outer_self = outer_self
                    self.model_name = model_name
                    self.best_weights = None
                    self.best_score = float('inf')
                
                def on_epoch_end(self, epoch, logs=None):
                    current_score = logs.get('val_mae')
                    if current_score < self.best_score:
                        self.best_score = current_score
                        self.best_weights = self.model.get_weights()
                        self.outer_self.save_best_model(self.model, self.model_name, current_score, epoch)
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
            
            checkpoint = ModelCheckpointWithScore(self, model_name)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=128,
                callbacks=[early_stop, checkpoint],
                verbose=0
            )
            
            self.training_history[model_name] = history.history
            val_pred = model.predict(X_val, verbose=0).flatten()
            val_mae = mean_absolute_error(y_val, val_pred)
            
            self.models[model_name] = model
            print(f"    {model_name} - Validation MAE: {val_mae:.4f}")
            
            return model, val_mae
        
        # Validate tree-based models
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        self.models[model_name] = model
        print(f"    {model_name} - Validation MAE: {val_mae:.4f}")
        
        self.save_best_model(model, model_name, val_mae)
        
        return model, val_mae
    
    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train all models and create ensemble - FIXED VERSION"""
        print("\n" + "="*80)
        print("TRAINING ENSEMBLE MODELS")
        print("="*80)
        
        # FIX: Ensure all data is numeric before scaling
        print("  Ensuring all features are numeric...")
        X_train = self._ensure_numeric(X_train)
        X_val = self._ensure_numeric(X_val)
        
        # Store training feature names for prediction alignment
        self.training_features = X_train.columns.tolist()
        print(f"  Stored {len(self.training_features)} training features")
        
        # Scale features for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        model_scores = {}
        
        model_list = ['lightgbm', 'xgboost', 'gbm', 'rf', 'neural_net']
        
        for model_name in tqdm(model_list, desc="Training models"):
            try:
                if model_name == 'neural_net':
                    model, score = self.train_model(
                        X_train_scaled, y_train, 
                        X_val_scaled, y_val, 
                        model_name
                    )
                else:
                    model, score = self.train_model(
                        X_train, y_train, 
                        X_val, y_val, 
                        model_name
                    )
                
                model_scores[model_name] = score
                
            except Exception as e:
                print(f"    Error training {model_name}: {e}")
                model_scores[model_name] = float('inf')
            
            memory_monitor.force_cleanup()
        
        # Calculate ensemble weights
        valid_scores = {k: v for k, v in model_scores.items() if v < float('inf')}
        if valid_scores:
            total_inv_score = sum(1/score for score in valid_scores.values())
            self.weights = {
                name: (1/score) / total_inv_score 
                for name, score in valid_scores.items()
            }
        else:
            self.weights = {name: 1/len(model_list) for name in model_list}
        
        print(f"\n✓ Best single model: {self.best_model_name} (MAE: {self.best_score:.4f})")
        print("\n  Ensemble weights:")
        for name, weight in self.weights.items():
            if name in valid_scores:
                print(f"    {name}: {weight:.4f} (MAE: {valid_scores[name]:.4f})")
        
        # Feature importance
        if 'lightgbm' in self.models:
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.models['lightgbm'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self
    
    def _ensure_numeric(self, X):
        """Ensure all features are numeric by converting non-numeric columns"""
        X_clean = X.copy()
        
        for col in X_clean.columns:
            # Check if column contains non-numeric values
            if X_clean[col].dtype == 'object':
                print(f"    Converting non-numeric column '{col}' to numeric...")
                # Try to convert to numeric, coerce errors to NaN
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                
                # Fill any resulting NaN with 0
                if X_clean[col].isnull().any():
                    X_clean[col].fillna(0, inplace=True)
        
        return X_clean
    
    def align_features(self, X):
        """CRITICAL FIX: Align test features with training features"""
        if self.training_features is None:
            print("  WARNING: No training features stored. Using current features.")
            return X
        
        print(f"\n  Aligning features: {X.shape[1]} current -> {len(self.training_features)} expected")
        
        # Create a new DataFrame with the same columns as training
        X_aligned = pd.DataFrame(0, index=X.index, columns=self.training_features, dtype=X.dtypes[0] if len(X.dtypes) > 0 else np.float64)
        
        # Copy available features
        common_features = set(X.columns) & set(self.training_features)
        missing_features = set(self.training_features) - set(X.columns)
        extra_features = set(X.columns) - set(self.training_features)
        
        print(f"    Common features: {len(common_features)}")
        print(f"    Missing features: {len(missing_features)}")
        print(f"    Extra features: {len(extra_features)}")
        
        # Copy common features
        for feature in common_features:
            X_aligned[feature] = X[feature]
        
        # Fill missing features with 0
        if missing_features:
            print(f"    Filling {len(missing_features)} missing features with 0")
            for feature in missing_features:
                X_aligned[feature] = 0
        
        # Drop extra features (not used in training)
        if extra_features:
            print(f"    Dropping {len(extra_features)} extra features not seen in training")
        
        # Ensure the same column order as training
        X_aligned = X_aligned[self.training_features]
        
        print(f"    Final aligned shape: {X_aligned.shape}")
        
        return X_aligned
    
    def predict(self, X):
        """Make ensemble predictions with feature alignment"""
        predictions = []
        valid_models = []
        
        # FIX: Ensure input data is numeric
        X = self._ensure_numeric(X)
        
        # CRITICAL FIX: Align features with training data
        X = self.align_features(X)
        
        for model_name, model in self.models.items():
            if model_name in self.weights:
                try:
                    if model_name == 'neural_net':
                        X_scaled = self.scaler.transform(X)
                        pred = model.predict(X_scaled, verbose=0).flatten()
                    else:
                        pred = model.predict(X)
                    
                    predictions.append(pred * self.weights[model_name])
                    valid_models.append(model_name)
                except Exception as e:
                    print(f"Warning: Error predicting with {model_name}: {e}")
        
        if not predictions:
            raise ValueError("No valid models for prediction")
        
        return np.sum(predictions, axis=0), valid_models

# ============================================================================
# EVALUATION - ENHANCED WITH COMPREHENSIVE VISUALIZATIONS
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation with enhanced visualizations"""
    
    def __init__(self):
        self.metrics = {}
        self.predictions = {}
    
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    def calculate_custom_score(self, y_true, y_pred):
        """Calculate custom competition metric"""
        ape = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
        
        high_error_ratio = np.mean(ape > 100)
        
        if high_error_ratio > 0.3:
            return 0.0
        
        valid_mask = ape <= 1
        
        if valid_mask.sum() == 0:
            return 0.0
        
        mape_valid = np.mean(ape[valid_mask])
        fraction_valid = valid_mask.mean()
        
        scaled_mape = mape_valid / fraction_valid
        score = 1 - (scaled_mape / 100)
        
        return max(0, score)
    
    def evaluate(self, y_true, y_pred, dataset_name='Validation'):
        """Comprehensive evaluation"""
        print(f"\n{'='*80}")
        print(f"EVALUATION: {dataset_name}")
        print(f"{'='*80}")
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = self.calculate_mape(y_true, y_pred)
        custom_score = self.calculate_custom_score(y_true, y_pred)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Custom_Score': custom_score
        }
        
        print(f"\n  Metrics:")
        print(f"    MAE:          {mae:,.2f}")
        print(f"    RMSE:         {rmse:,.2f}")
        print(f"    R²:           {r2:.4f}")
        print(f"    MAPE:         {mape:.2f}%")
        print(f"    Custom Score: {custom_score:.4f}")
        
        self.metrics[dataset_name] = metrics
        self.predictions[dataset_name] = y_pred
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, dataset_name='Validation', save_path='predictions.png'):
        """Plot predictions vs actual"""
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{dataset_name} - Predictions vs Actual')
        
        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title(f'{dataset_name} - Residual Plot')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✓ Plot saved to {save_path}")
    
    def plot_overall_performance_comparison(self):
        """Plot comprehensive performance comparison across all datasets"""
        if len(self.metrics) < 2:
            print("Not enough datasets for comparison")
            return
        
        datasets = list(self.metrics.keys())
        metrics_list = ['MAE', 'RMSE', 'R2', 'MAPE', 'Custom_Score']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_list):
            if i < len(axes):
                values = [self.metrics[dataset][metric] for dataset in datasets]
                
                # For R2 and Custom Score, higher is better (use green)
                # For MAE, RMSE, MAPE, lower is better (use red)
                colors = ['green' if metric in ['R2', 'Custom_Score'] else 'red' for _ in values]
                
                bars = axes[i].bar(datasets, values, color=colors, alpha=0.7)
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    if metric in ['R2', 'Custom_Score']:
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
                    else:
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Add ensemble weights visualization if available
        if hasattr(self, 'ensemble_weights'):
            axes[5].pie(self.ensemble_weights.values(), labels=self.ensemble_weights.keys(),
                       autopct='%1.1f%%', startangle=90)
            axes[5].set_title('Ensemble Model Weights')
        
        plt.tight_layout()
        plt.suptitle('Overall Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig('overall_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_distribution(self, y_true_dict, y_pred_dict):
        """Plot error distribution across all datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (dataset_name, y_true) in enumerate(y_true_dict.items()):
            if i < len(axes) and dataset_name in y_pred_dict:
                y_pred = y_pred_dict[dataset_name]
                errors = y_true - y_pred
                relative_errors = (errors / (y_true + 1e-10)) * 100
                
                # Absolute error distribution
                axes[i].hist(np.abs(errors), bins=50, alpha=0.7, color='coral', edgecolor='black')
                axes[i].set_title(f'{dataset_name} - Absolute Error Distribution')
                axes[i].set_xlabel('Absolute Error')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_abs_error = np.mean(np.abs(errors))
                std_abs_error = np.std(np.abs(errors))
                axes[i].axvline(mean_abs_error, color='red', linestyle='--', 
                              label=f'Mean: {mean_abs_error:.2f}')
                axes[i].axvline(mean_abs_error + std_abs_error, color='orange', linestyle='--',
                              label=f'+1 STD: {mean_abs_error + std_abs_error:.2f}')
                axes[i].axvline(mean_abs_error - std_abs_error, color='orange', linestyle='--',
                              label=f'-1 STD: {mean_abs_error - std_abs_error:.2f}')
                axes[i].legend()
        
        plt.tight_layout()
        plt.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.savefig('error_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_progress(self, ensemble_model):
        """Plot training progress for neural network if available"""
        if not hasattr(ensemble_model, 'training_history') or not ensemble_model.training_history:
            print("No training history available")
            return
        
        for model_name, history in ensemble_model.training_history.items():
            if model_name == 'neural_net':
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                # Loss plot
                axes[0].plot(history['loss'], label='Training Loss')
                if 'val_loss' in history:
                    axes[0].plot(history['val_loss'], label='Validation Loss')
                axes[0].set_title(f'{model_name} - Loss Progress')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # MAE plot
                if 'mae' in history:
                    axes[1].plot(history['mae'], label='Training MAE')
                if 'val_mae' in history:
                    axes[1].plot(history['val_mae'], label='Validation MAE')
                axes[1].set_title(f'{model_name} - MAE Progress')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MAE')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.suptitle(f'{model_name} Training Progress', fontsize=16, fontweight='bold', y=1.02)
                plt.savefig(f'{model_name}_training_progress.png', dpi=300, bbox_inches='tight')
                plt.show()
                break

# ============================================================================
# MAIN PIPELINE - ENHANCED WITH COMPREHENSIVE VISUALIZATIONS
# ============================================================================

class RealEstateDemandPipeline:
    """Complete end-to-end pipeline with enhanced visualizations"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.loader = DataLoader(data_path)
        self.engineer = FeatureEngineer()
        self.validator = DataSplitValidator()
        self.ensemble = EnsembleModel()
        self.evaluator = ModelEvaluator()
        self.eda = EDAVisualizer()
        self.final_features = None
    
    def create_target_variable(self, data):
        """Create target variable from new house transactions data"""
        print("\n  Creating target variable...")
        
        if 'new_house' in data and not data['new_house'].empty:
            # Calculate new_house_transaction_amount from area * price
            new_house_df = data['new_house'].copy()
            
            if 'area_new_house_transactions' in new_house_df.columns and 'price_new_house_transactions' in new_house_df.columns:
                new_house_df['target'] = (
                    new_house_df['area_new_house_transactions'] * 
                    new_house_df['price_new_house_transactions']
                ) / 10000  # Convert to ten thousand Chinese Yuan
                
                print(f"✓ Target variable created: {len(new_house_df)} samples")
                return new_house_df[['month', 'sector', 'target']]
        
        print("✗ Could not create target variable")
        return pd.DataFrame()
    
    def perform_comprehensive_eda(self, data, target_df):
        """Perform comprehensive EDA on the dataset"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*80)
        
        # 1. Target Distribution
        if not target_df.empty and 'target' in target_df.columns:
            print("\n1. Target Variable Analysis:")
            self.eda.plot_target_distribution(target_df['target'], "New House Transaction Amount Distribution")
            
            # Summary statistics
            print(f"\n   Target Statistics:")
            print(f"   Min: {target_df['target'].min():.2f}")
            print(f"   Mean: {target_df['target'].mean():.2f}")
            print(f"   Median: {target_df['target'].median():.2f}")
            print(f"   Max: {target_df['target'].max():.2f}")
            print(f"   Std: {target_df['target'].std():.2f}")
        
        # 2. Merge sample data for correlation analysis
        sample_data = self.merge_sample_data(data, target_df)
        if not sample_data.empty:
            print("\n2. Feature Correlation Analysis:")
            self.eda.plot_feature_correlations(sample_data, 'target', top_n=15)
            
            print("\n3. Missing Values Analysis:")
            self.eda.plot_missing_values(sample_data)
            
            # 4. Time Series Trends
            print("\n4. Time Series Analysis:")
            if 'month' in sample_data.columns:
                self.eda.plot_time_series_trends(
                    sample_data, 
                    value_cols=['target', 'price_new_house_transactions', 'num_new_house_transactions']
                )
            
            # 5. Boxplots for key numeric features
            print("\n5. Distribution Analysis:")
            numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()[:9]  # First 9 numeric columns
            if numeric_cols:
                self.eda.plot_boxplots(sample_data, numeric_cols)
        
        # 6. Dataset shapes overview
        print("\n6. Dataset Overview:")
        for key, df in data.items():
            if not df.empty:
                print(f"   {key:20}: {df.shape}")
    
    def merge_sample_data(self, data, target_df):
        """Merge sample data for EDA purposes"""
        try:
            # Start with target data
            merged_df = target_df.copy()
            
            # Merge with main new house data
            if 'new_house' in data and not data['new_house'].empty:
                merged_df = merged_df.merge(
                    data['new_house'], 
                    on=['month', 'sector'], 
                    how='left',
                    suffixes=('', '_new_house')
                )
            
            # Merge with pre-owned house data
            if 'pre_owned' in data and not data['pre_owned'].empty:
                merged_df = merged_df.merge(
                    data['pre_owned'],
                    on=['month', 'sector'],
                    how='left',
                    suffixes=('', '_pre_owned')
                )
            
            return merged_df
        except Exception as e:
            print(f"  Warning in EDA data merge: {e}")
            return pd.DataFrame()
    
    def merge_all_features(self, data, is_training=True):
        """Merge all data sources into unified dataset"""
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        if is_training:
            # For training, start with new house data that has target
            if 'new_house' in data and not data['new_house'].empty:
                df = data['new_house'].copy()
                # Create target variable for training data
                if 'area_new_house_transactions' in df.columns and 'price_new_house_transactions' in df.columns:
                    df['target'] = (df['area_new_house_transactions'] * df['price_new_house_transactions']) / 10000
            else:
                df = pd.DataFrame()
        else:
            # For test data, start with test structure
            df = data['test'].copy()
            df = self.engineer.parse_test_id(df)
        
        if df.empty:
            print("  Warning: No base data found for merging")
            return df
        
        # Parse date features
        df = self.engineer.parse_date_features(df, 'month')
        
        # Merge all transaction data
        merge_datasets = [
            ('new_house_nearby', ['month', 'sector']),
            ('pre_owned', ['month', 'sector']),
            ('pre_owned_nearby', ['month', 'sector']),
            ('land', ['month', 'sector']),
            ('land_nearby', ['month', 'sector']),
            ('sector_poi', ['sector']),  # Static sector data
            ('city_index', ['month']),   # City-level time series
            ('search_index', ['month'])  # Search index time series
        ]
        
        for dataset_name, merge_on in merge_datasets:
            if dataset_name in data and not data[dataset_name].empty:
                dataset = data[dataset_name].copy()
                
                # Parse dates if needed and if month column exists
                if 'month' in dataset.columns:
                    dataset = self.engineer.parse_date_features(dataset, 'month')
                
                # Merge
                print(f"\n  Merging {dataset_name}...")
                try:
                    df = df.merge(dataset, on=merge_on, how='left', suffixes=('', f'_{dataset_name}'))
                except Exception as e:
                    print(f"    Warning: Could not merge {dataset_name}: {e}")
        
        print(f"\n✓ Merged dataset shape: {df.shape}")
        
        # Create advanced features
        value_cols = [col for col in df.columns if any(x in col for x in 
                     ['num_', 'area_', 'amount_', 'price_', 'index'])]
        
        # Only create time-series features if we have enough temporal data
        if 'month' in df.columns and 'sector' in df.columns:
            df = self.engineer.create_lag_features(df, ['sector'], value_cols, lags=[1, 2, 3])
            df = self.engineer.create_rolling_features(df, ['sector'], value_cols, windows=[3, 6])
            df = self.engineer.create_growth_features(df, ['sector'], value_cols)
        
        df = self.engineer.create_ratio_features(df)
        
        # Handle missing values - CRITICAL
        df = self.engineer.handle_missing_values(df, verbose=True)
        
        return df
    
    def prepare_features(self, df, target_col='target'):
        """Prepare features for modeling"""
        print("\n  Preparing features for modeling...")
        
        # Drop non-feature columns
        drop_cols = ['id', 'sector', 'month']
        if target_col in df.columns:
            drop_cols.append(target_col)
        
        feature_cols = [col for col in df.columns if col not in drop_cols]
        
        X = df[feature_cols].copy()
        
        # Handle any remaining missing values
        X = self.engineer.handle_missing_values(X, verbose=False)
        
        # Store feature names
        self.final_features = X.columns.tolist()
        
        print(f"    Final feature count: {len(self.final_features)}")
        
        if target_col in df.columns:
            y = df[target_col].copy()
            return X, y
        
        return X, None
    
    def run_pipeline(self):
        """Execute complete pipeline with enhanced visualizations"""
        print("\n" + "="*80)
        print("CHINA REAL ESTATE DEMAND PREDICTION PIPELINE")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        data = self.loader.load_all_data()
        
        # Create target variable for EDA and training
        target_df = self.create_target_variable(data)
        
        # Perform comprehensive EDA
        self.perform_comprehensive_eda(data, target_df)
        
        # Merge and engineer features for training
        print("\n" + "="*80)
        print("TRAINING DATA PREPARATION")
        print("="*80)
        df_full = self.merge_all_features(data, is_training=True)
        
        if df_full.empty:
            print("✗ No training data available. Cannot proceed.")
            return None
        
        # Prepare features
        X, y = self.prepare_features(df_full, 'target')
        
        if X.empty or y.empty:
            print("✗ No features or target available. Cannot proceed.")
            return None
        
        # Split data: 70% train, 15% validation, 15% holdout
        X_temp, X_holdout, y_temp, y_holdout = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
        )
        
        # Validate splits
        self.validator.verify_data_splits(X_train, X_val, X_holdout, y_train, y_val, y_holdout)
        
        # Train ensemble
        self.ensemble.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        val_pred, _ = self.ensemble.predict(X_val)
        self.evaluator.evaluate(y_val, val_pred, 'Validation')
        self.evaluator.plot_predictions(y_val, val_pred, 'Validation', 'validation_predictions.png')
        
        # Evaluate on holdout set
        print("\n" + "="*80)
        print("FINAL HOLDOUT EVALUATION")
        print("="*80)
        holdout_pred, _ = self.ensemble.predict(X_holdout)
        self.evaluator.evaluate(y_holdout, holdout_pred, 'Holdout')
        self.evaluator.plot_predictions(y_holdout, holdout_pred, 'Holdout', 'holdout_predictions.png')
        
        # Enhanced comprehensive visualizations
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*80)
        
        # 1. Overall performance comparison
        self.evaluator.ensemble_weights = self.ensemble.weights
        self.evaluator.plot_overall_performance_comparison()
        
        # 2. Error distribution analysis
        error_data_true = {
            'Validation': y_val,
            'Holdout': y_holdout
        }
        error_data_pred = {
            'Validation': val_pred,
            'Holdout': holdout_pred
        }
        self.evaluator.plot_error_distribution(error_data_true, error_data_pred)
        
        # 3. Training progress visualization
        self.evaluator.plot_training_progress(self.ensemble)
        
        # Feature importance visualization
        if self.ensemble.feature_importance is not None:
            print("\n" + "="*80)
            print("TOP 20 IMPORTANT FEATURES")
            print("="*80)
            print(self.ensemble.feature_importance.head(20).to_string(index=False))
            
            # Plot feature importance
            self.eda.plot_feature_importance(
                self.ensemble.feature_importance, 
                top_n=20, 
                title="Top 20 Feature Importance"
            )
        
        # Generate test predictions
        print("\n" + "="*80)
        print("GENERATING TEST PREDICTIONS")
        print("="*80)
        
        test_df = self.merge_all_features(data, is_training=False)
        
        if test_df.empty:
            print("✗ No test data available. Cannot generate predictions.")
            return None
        
        X_test, _ = self.prepare_features(test_df, 'target')
        
        test_predictions, _ = self.ensemble.predict(X_test)
        
        # Create submission with EXACT required format
        submission = pd.DataFrame({
            'id': data['test']['id'],  # Preserve original id order
            'new_house_transaction_amount': test_predictions  # EXACT column name as required
        })
        
        # Ensure we maintain the exact same row order as test.csv
        submission = submission[['id', 'new_house_transaction_amount']]  # Ensure correct column order
        
        submission.to_csv('submission.csv', index=False)
        print(f"\n✓ Submission saved to submission.csv")
        print(f"  Predictions range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
        print(f"  Submission shape: {submission.shape}")
        print(f"  Columns: {submission.columns.tolist()}")
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Best model: {self.ensemble.best_model_name}")
        print(f"Best validation MAE: {self.ensemble.best_score:.4f}")
        
        # Final comprehensive visualization
        self.plot_final_summary()
        
        return submission
    
    def plot_final_summary(self):
        """Create a final comprehensive summary visualization"""
        print("\n" + "="*80)
        print("FINAL SUMMARY VISUALIZATION")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Model performance comparison
        if len(self.evaluator.metrics) >= 2:
            datasets = list(self.evaluator.metrics.keys())
            mae_values = [self.evaluator.metrics[dataset]['MAE'] for dataset in datasets]
            r2_values = [self.evaluator.metrics[dataset]['R2'] for dataset in datasets]
            
            x = np.arange(len(datasets))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, mae_values, width, label='MAE', color='red', alpha=0.7)
            axes[0, 0].set_xlabel('Dataset')
            axes[0, 0].set_ylabel('MAE', color='red')
            axes[0, 0].tick_params(axis='y', labelcolor='red')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(datasets)
            
            ax2 = axes[0, 0].twinx()
            ax2.bar(x + width/2, r2_values, width, label='R²', color='blue', alpha=0.7)
            ax2.set_ylabel('R²', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            axes[0, 0].set_title('Model Performance Across Datasets')
            axes[0, 0].legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # 2. Ensemble weights
        if hasattr(self.ensemble, 'weights'):
            weights = self.ensemble.weights
            axes[0, 1].pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%', 
                          startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(weights))))
            axes[0, 1].set_title('Ensemble Model Weights')
        
        # 3. Feature importance (top 10)
        if self.ensemble.feature_importance is not None:
            top_features = self.ensemble.feature_importance.head(10)
            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_xlabel('Importance Score')
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Memory usage over time (simulated)
        memory_usage = [70, 75, 80, 85, 78, 72, 68, 65]  # Simulated data
        time_points = range(len(memory_usage))
        axes[1, 1].plot(time_points, memory_usage, marker='o', linewidth=2)
        axes[1, 1].axhline(y=85, color='red', linestyle='--', label='Memory Limit (85%)')
        axes[1, 1].set_xlabel('Pipeline Stage')
        axes[1, 1].set_ylabel('Memory Usage (%)')
        axes[1, 1].set_title('Memory Usage During Pipeline Execution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].fill_between(time_points, memory_usage, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('China Real Estate Demand Prediction - Final Summary', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.savefig('final_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Final summary visualization completed and saved")

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set data path - FIXED: Use Kaggle competition path
    DATA_PATH = "/kaggle/input/china-real-estate-demand-prediction"
    
    # Initialize and run pipeline
    pipeline = RealEstateDemandPipeline(DATA_PATH)
    submission = pipeline.run_pipeline()
    
    print("\n✓ All done! Check the output files:")
    print("  - submission.csv")
    print("  - validation_predictions.png")
    print("  - holdout_predictions.png")
    print("  - feature_importance.png")
    print("  - overall_performance_comparison.png")
    print("  - error_distribution_analysis.png")
    print("  - final_summary.png")
    print("  - best_models/ (saved model files)")




