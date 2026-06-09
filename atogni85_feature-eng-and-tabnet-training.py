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


! pip install pytorch_tabnet -qqqq


# Suppress warnings first
import warnings
warnings.filterwarnings('ignore')

# Standard library imports
import os
import json
import joblib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List, Dict

# Third-party numerical libraries
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn preprocessing and model selection
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
)
from sklearn.base import BaseEstimator, TransformerMixin

# Scikit-learn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)

# Deep learning and specialized ML libraries
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

# Gradient boosting libraries
import xgboost as xgb
from xgboost import plot_importance


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')

test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


class f_eng(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature engineering class for banking classification datasets.
    Handles encoding, temporal features, derived features, and scaling.
    """
    
    def __init__(self, 
                 scale_numerical: bool = True,
                 create_interactions: bool = True,
                 create_temporal_features: bool = True,
                 binning_strategy: str = 'quantile',  # 'quantile' or 'uniform'
                 balance_outlier_threshold: float = 3.0):
        
        self.scale_numerical = scale_numerical
        self.create_interactions = create_interactions
        self.create_temporal_features = create_temporal_features
        self.binning_strategy = binning_strategy
        self.balance_outlier_threshold = balance_outlier_threshold
        
        # Initialize scalers and encoders
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        
        # Define column mappings
        self.month_to_season = {
            'jan': 'winter', 'feb': 'winter', 'dec': 'winter',
            'mar': 'spring', 'apr': 'spring', 'may': 'spring',
            'jun': 'summer', 'jul': 'summer', 'aug': 'summer',
            'sep': 'autumn', 'oct': 'autumn', 'nov': 'autumn'
        }
        
        self.month_to_quarter = {
            'jan': 'Q1', 'feb': 'Q1', 'mar': 'Q1',
            'apr': 'Q2', 'may': 'Q2', 'jun': 'Q2',
            'jul': 'Q3', 'aug': 'Q3', 'sep': 'Q3',
            'oct': 'Q4', 'nov': 'Q4', 'dec': 'Q4'
        }
        
        self.education_order = {
            'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3
        }
        
        # Store fitted parameters
        self.fitted_params = {}
        self.categorical_columns = []
        self.numerical_columns = []
        
    def _encode_ordinal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode ordinal features with meaningful order."""
        df = df.copy()
        
        # Education (ordinal)
        if 'education' in df.columns:
            df['education_encoded'] = df['education'].map(self.education_order)
            df['education_encoded'] = df['education_encoded'].fillna(0)  # Unknown as 0
        
        return df
    
    def _encode_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode binary yes/no features."""
        df = df.copy()
        
        binary_cols = ['default', 'housing', 'loan']
        for col in binary_cols:
            if col in df.columns:
                df[f'{col}_binary'] = (df[col] == 'yes').astype(int)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal and seasonal features."""
        if not self.create_temporal_features:
            return df
            
        df = df.copy()
        
        if 'month' in df.columns:
            # Season mapping
            df['season'] = df['month'].map(self.month_to_season)
            
            # Quarter mapping
            df['quarter'] = df['month'].map(self.month_to_quarter)
            
            # Month as cyclic feature
            month_num = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            df['month_num'] = df['month'].map(month_num)
            df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived financial and behavioral features."""
        df = df.copy()
        
        # Financial profile features
        if all(col in df.columns for col in ['housing', 'loan', 'default']):
            df['debt_burden'] = (
                (df['housing'] == 'yes').astype(int) + 
                (df['loan'] == 'yes').astype(int)
            )
            df['financial_risk'] = (
                (df['default'] == 'yes').astype(int) + df['debt_burden']
            )
        
        # Balance-related features
        if 'balance' in df.columns:
            df['negative_balance'] = (df['balance'] < 0).astype(int)
            df['zero_balance'] = (df['balance'] == 0).astype(int)
            
            if 'age' in df.columns:
                # Avoid division by zero
                df['balance_per_age'] = df['balance'] / (df['age'] + 1e-6)
        
        # Contact experience features
        if 'previous' in df.columns:
            df['is_first_contact'] = (df['previous'] == 0).astype(int)
            
            if 'campaign' in df.columns:
                df['contact_intensity'] = df['campaign'] + df['previous']
                # Avoid division by zero
                df['success_rate_proxy'] = df['previous'] / (df['campaign'] + 1)
        
        # pdays handling
        if 'pdays' in df.columns:
            df['never_contacted'] = (df['pdays'] == -1).astype(int)
            df['days_since_last'] = df['pdays'].replace(-1, 999)  # High value for never contacted
            df['recent_contact'] = (df['pdays'].between(0, 30)).astype(int)
        
        # Duration features
        if 'duration' in df.columns:
            df['short_call'] = (df['duration'] < 120).astype(int)  # Less than 2 minutes
            df['long_call'] = (df['duration'] > 600).astype(int)   # More than 10 minutes
        
        return df
    
    def _create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age group binning."""
        if 'age' not in df.columns:
            return df
            
        df = df.copy()
        
        # Define age bins based on banking customer segments
        age_bins = [0, 30, 45, 60, 100]
        age_labels = ['young', 'middle', 'mature', 'senior']
        
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)
        
        return df
    
    def _create_balance_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create balance tier binning."""
        if 'balance' not in df.columns:
            return df
            
        df = df.copy()
        
        if self.binning_strategy == 'quantile':
            df['balance_tier'] = pd.qcut(df['balance'], q=5, 
                                       labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                                       duplicates='drop')
        else:  # uniform
            df['balance_tier'] = pd.cut(df['balance'], bins=5,
                                      labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        if not self.create_interactions:
            return df
            
        df = df.copy()
        
        # Young professional
        if all(col in df.columns for col in ['age', 'job']):
            professional_jobs = ['management', 'technician', 'admin.', 'services']
            df['young_professional'] = (
                (df['age'] < 35) & 
                (df['job'].isin(professional_jobs))
            ).astype(int)
        
        # High-value customer (age + balance interaction)
        if all(col in df.columns for col in ['age', 'balance']):
            df['mature_high_balance'] = (
                (df['age'] >= 45) & (df['balance'] > df['balance'].quantile(0.75))
            ).astype(int)
        
        # Experienced positive outcome
        if all(col in df.columns for col in ['previous', 'poutcome']):
            df['positive_experience'] = (
                (df['previous'] > 0) & (df['poutcome'] == 'success')
            ).astype(int)
        
        # Married with loans (family responsibility)
        if all(col in df.columns for col in ['marital', 'housing', 'loan']):
            df['family_financial_burden'] = (
                (df['marital'] == 'married') & 
                ((df['housing'] == 'yes') | (df['loan'] == 'yes'))
            ).astype(int)
        
        return df
    
    def _perform_one_hot_encoding(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Perform one-hot encoding for nominal categorical variables."""
        df = df.copy()
        
        # Define nominal categorical columns to encode
        nominal_cols = ['job', 'marital', 'contact', 'poutcome']
        
        if self.create_temporal_features:
            nominal_cols.extend(['season', 'quarter'])
        
        if 'age_group' in df.columns:
            nominal_cols.append('age_group')
        if 'balance_tier' in df.columns:
            nominal_cols.append('balance_tier')
        
        # Filter existing columns
        nominal_cols = [col for col in nominal_cols if col in df.columns]
        
        if fit:
            self.categorical_columns = nominal_cols
        
        # Perform one-hot encoding
        for col in nominal_cols:
            if col in df.columns:
                # Get dummies
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features with appropriate scalers."""
        if not self.scale_numerical:
            return df
            
        df = df.copy()
        
        # Define numerical columns for different scaling strategies
        standard_scale_cols = ['age', 'duration', 'campaign']
        robust_scale_cols = ['balance', 'days_since_last']  # Potentially with outliers
        minmax_scale_cols = ['balance_per_age', 'success_rate_proxy']  # Derived features
        
        # Filter existing columns
        standard_scale_cols = [col for col in standard_scale_cols if col in df.columns]
        robust_scale_cols = [col for col in robust_scale_cols if col in df.columns]
        minmax_scale_cols = [col for col in minmax_scale_cols if col in df.columns]
        
        if fit:
            # Fit scalers
            if standard_scale_cols:
                self.standard_scaler.fit(df[standard_scale_cols])
            if robust_scale_cols:
                self.robust_scaler.fit(df[robust_scale_cols])
            if minmax_scale_cols:
                self.minmax_scaler.fit(df[minmax_scale_cols])
        
        # Transform
        if standard_scale_cols:
            df[standard_scale_cols] = self.standard_scaler.transform(df[standard_scale_cols])
        if robust_scale_cols:
            df[robust_scale_cols] = self.robust_scaler.transform(df[robust_scale_cols])
        if minmax_scale_cols:
            df[minmax_scale_cols] = self.minmax_scaler.transform(df[minmax_scale_cols])
        
        return df
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature engineering pipeline."""
        X = X.copy()
        
        # Apply transformations in order
        X = self._encode_ordinal_features(X)
        X = self._encode_binary_features(X)
        X = self._create_temporal_features(X)
        X = self._create_derived_features(X)
        X = self._create_age_groups(X)
        X = self._create_balance_tiers(X)
        X = self._create_interaction_features(X)
        X = self._perform_one_hot_encoding(X, fit=True)
        X = self._scale_numerical_features(X, fit=True)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted parameters."""
        X = X.copy()
        
        # Apply transformations in the same order as fit
        X = self._encode_ordinal_features(X)
        X = self._encode_binary_features(X)
        X = self._create_temporal_features(X)
        X = self._create_derived_features(X)
        X = self._create_age_groups(X)
        X = self._create_balance_tiers(X)
        X = self._create_interaction_features(X)
        X = self._perform_one_hot_encoding(X, fit=False)
        X = self._scale_numerical_features(X, fit=False)
        
        # Drop original categorical columns that were encoded
        cols_to_drop = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'poutcome']
        
        if self.create_temporal_features:
            cols_to_drop.extend(['season', 'quarter', 'month_num'])
        
        cols_to_drop.extend(['age_group', 'balance_tier'])
        
        # Only drop existing columns
        cols_to_drop = [col for col in cols_to_drop if col in X.columns]
        X = X.drop(columns=cols_to_drop)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get the names of output features after transformation."""
        X_transformed = self.transform(X.head(1))  # Transform just one row to get column names
        return X_transformed.columns.tolist()





fe = f_eng(
        scale_numerical=True,
        create_interactions=True,
        create_temporal_features=True
    )
    
# Fit and transform
df_transformed = fe.fit_transform(train_df)

print("Original shape:", train_df.shape)
print("Transformed shape:", df_transformed.shape)
print("\nNew features created:")
print(df_transformed.columns.tolist())


class TabNetBinaryClassifier:
    """
    Class for training a TabNet binary classifier with GPU support
    """
    
    def __init__(self, 
                 n_d=32, 
                 n_a=32, 
                 n_steps=5, 
                 gamma=1.3,
                 n_independent=2,
                 n_shared=2,
                 lambda_sparse=1e-3,
                 optimizer_fn=torch.optim.Adam,
                 optimizer_params=dict(lr=1e-2),
                 mask_type='entmax',
                 scheduler_params=dict(step_size=50, gamma=0.9),
                 scheduler_fn=torch.optim.lr_scheduler.StepLR,
                 epsilon=1e-15,
                 device_name='auto'):
        """
        Initialize the TabNet classifier
        
        Parameters:
        -----------
        n_d : int
            Dimension of learned representations
        n_a : int 
            Dimension of attention
        n_steps : int
            Number of steps in feature selection
        gamma : float
            Coefficient for aggregated attention
        lambda_sparse : float
            Regularization coefficient for sparsity
        device_name : str
            'auto', 'cuda', 'cpu' or specific device ('cuda:0')
        """
        
        # Device configuration
        self.device = self._setup_device(device_name)
        print(f"Device used: {self.device}")
        
        self.tabnet_params = {
            'n_d': n_d,
            'n_a': n_a, 
            'n_steps': n_steps,
            'gamma': gamma,
            'n_independent': n_independent,
            'n_shared': n_shared,
            'lambda_sparse': lambda_sparse,
            'optimizer_fn': optimizer_fn,
            'optimizer_params': optimizer_params,
            'mask_type': mask_type,
            'scheduler_params': scheduler_params,
            'scheduler_fn': scheduler_fn,
            'epsilon': epsilon,
            'device_name': self.device
        }
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def _setup_device(self, device_name):
        """
        Configure the computing device (CPU/GPU)
        """
        if device_name == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"GPU available: {torch.cuda.get_device_name()}")
                print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = 'cpu'
                print("GPU not available, using CPU")
        else:
            device = device_name
            if device.startswith('cuda') and not torch.cuda.is_available():
                print("WARNING: GPU requested but not available, using CPU")
                device = 'cpu'
        
        return device
    
    def get_gpu_memory_info(self):
        """
        Returns GPU memory information
        """
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            device_idx = 0 if self.device == 'cuda' else int(self.device.split(':')[1])
            allocated = torch.cuda.memory_allocated(device_idx) / 1e9
            reserved = torch.cuda.memory_reserved(device_idx) / 1e9
            total = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
            
            print(f"GPU Memory:")
            print(f"  - Allocated: {allocated:.2f} GB")
            print(f"  - Reserved: {reserved:.2f} GB") 
            print(f"  - Total: {total:.2f} GB")
            print(f"  - Free: {total - reserved:.2f} GB")
            
            return {
                'allocated': allocated,
                'reserved': reserved,
                'total': total,
                'free': total - reserved
            }
        else:
            print("GPU memory not available")
            return None
    
    def clear_gpu_memory(self):
        """
        Clear GPU memory
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
    
    def prepare_data(self, df, target_col='y', test_size=0.2, random_state=42, scale_features=True):
        """
        Prepare data for training
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with data
        target_col : str
            Name of target column
        test_size : float
            Proportion of test set
        random_state : int
            Seed for reproducibility
        scale_features : bool
            Whether to apply scaling to features
        """
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables if present
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            print(f"Found {len(categorical_columns)} categorical variables: {categorical_columns.tolist()}")
            # Encode categorical variables
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Convert to float32 to optimize GPU memory
        X = X.astype(np.float32)
        
        # Encode target if necessary
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = self.label_encoder.fit_transform(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scaling if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            X_train = X_train.values
            X_test = X_test.values
        
        # Convert to float32 for GPU
        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_train = y_train.values.astype(np.int64)
        self.y_test = y_test.values.astype(np.int64)
        
        print(f"Data prepared:")
        print(f"  - Training set: {self.X_train.shape}")
        print(f"  - Test set: {self.X_test.shape}")
        print(f"  - Feature data type: {self.X_train.dtype}")
        print(f"  - Target data type: {self.y_train.dtype}")
        print(f"  - Train class distribution: {np.bincount(self.y_train)}")
        print(f"  - Test class distribution: {np.bincount(self.y_test)}")
        
        # Show memory usage if on GPU
        if self.device.startswith('cuda'):
            data_size_mb = (self.X_train.nbytes + self.X_test.nbytes + 
                           self.y_train.nbytes + self.y_test.nbytes) / 1e6
            print(f"  - Data size in memory: {data_size_mb:.1f} MB")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self, 
              max_epochs=200, 
              patience=15, 
              batch_size=1024,
              virtual_batch_size=128,
              num_workers=0,
              drop_last=False):
        """
        Train the TabNet model
        
        Parameters:
        -----------
        max_epochs : int
            Maximum number of epochs
        patience : int
            Patience for early stopping
        batch_size : int
            Batch size
        virtual_batch_size : int
            Virtual batch size
        num_workers : int
            Number of workers for DataLoader (0 for GPU)
        """
        
        if not hasattr(self, 'X_train'):
            raise ValueError("You must first prepare data with prepare_data()")
        
        # Adapt batch_size for GPU
        if self.device.startswith('cuda'):
            gpu_memory = self.get_gpu_memory_info()
            if gpu_memory and gpu_memory['free'] < 2.0:  # Less than 2GB free
                suggested_batch_size = min(batch_size, 512)
                print(f"Limited GPU memory, reducing batch_size to {suggested_batch_size}")
                batch_size = suggested_batch_size
            
            # Optimize num_workers for GPU
            if num_workers == 0:
                num_workers = min(4, torch.cuda.device_count() * 2)
                
        print(f"Training configuration:")
        print(f"  - Device: {self.device}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Virtual batch size: {virtual_batch_size}")
        print(f"  - Num workers: {num_workers}")
        
        # Initialize model
        self.model = TabNetClassifier(**self.tabnet_params)
        
        # Check memory before training
        if self.device.startswith('cuda'):
            self.clear_gpu_memory()
            print("GPU memory before training:")
            self.get_gpu_memory_info()
        
        # Training
        print("\nStarting TabNet training...")
        
        try:
            self.model.fit(
                X_train=self.X_train,
                y_train=self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                eval_name=['test'],
                eval_metric=['accuracy', 'auc'],
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=num_workers,
                drop_last=drop_last,
            )
            
            self.is_fitted = True
            print("Training completed!")
            
            # Check memory after training
            if self.device.startswith('cuda'):
                print("\nGPU memory after training:")
                self.get_gpu_memory_info()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nERROR: Insufficient GPU memory!")
                print(f"Try to:")
                print(f"  - Reduce batch_size (current: {batch_size})")
                print(f"  - Reduce n_d and n_a (current: {self.tabnet_params['n_d']}, {self.tabnet_params['n_a']})")
                print(f"  - Use device_name='cpu'")
                self.clear_gpu_memory()
            raise e
        
        return self.model
    
    def predict(self, X=None):
        """
        Make predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet!")
        
        if X is None:
            X = self.X_test
            
        # Convert to float32 for consistency
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        else:
            X = X.astype(np.float32)
            
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X=None):
        """
        Return prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet!")
        
        if X is None:
            X = self.X_test
            
        # Convert to float32 for consistency
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)
        elif not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        else:
            X = X.astype(np.float32)
            
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def evaluate(self, X=None, y=None, plot_results=True):
        """
        Evaluate model performance
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet!")
        
        if X is None:
            X = self.X_test
            y = self.y_test
        
        # Predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        auc_score = roc_auc_score(y, y_pred_proba[:, 1])
        
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred))
        
        if plot_results:
            self.plot_results(y, y_pred, y_pred_proba[:, 1])
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_results(self, y_true, y_pred, y_pred_proba):
        """
        Visualize results
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True)
        
        # Distribution of Probabilities
        axes[2].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Class 0', color='red')
        axes[2].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Class 1', color='blue')
        axes[2].set_xlabel('Predicted Probability')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Predicted Probabilities')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=20):
        """
        Visualize feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet!")
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        if self.feature_names:
            feature_names = self.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances - TabNet')
        plt.xlabel('Feature Importance')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def save_model(self, filepath):
        """
        Save the model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet!")
        
        self.model.save_model(filepath)
        print(f"Model saved at: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        """
        self.model = TabNetClassifier(device_name=self.device)
        self.model.load_model(filepath)
        self.is_fitted = True
        print(f"Model loaded from: {filepath}")
    
    def get_model_summary(self):
        """
        Return model and hardware summary
        """
        if not self.is_fitted:
            print("Model not yet trained")
            return
        
        print(f"\n=== MODEL SUMMARY ===")
        print(f"Device: {self.device}")
        print(f"TabNet Parameters:")
        for key, value in self.tabnet_params.items():
            if key != 'device_name':
                print(f"  - {key}: {value}")
        
        if hasattr(self.model, 'network'):
            total_params = sum(p.numel() for p in self.model.network.parameters())
            trainable_params = sum(p.numel() for p in self.model.network.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        if self.device.startswith('cuda'):
            self.get_gpu_memory_info()


# GPU-optimized usage example
if __name__ == "__main__":
    # Initialize classifier with GPU
    classifier = TabNetBinaryClassifier(
        n_d=64,
        n_a=64, 
        n_steps=7,
        gamma=1.5,
        lambda_sparse=1e-3,
        device_name='auto'  # Automatically detect GPU
    )
    
    # Show GPU info
    classifier.get_gpu_memory_info()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(df_transformed, target_col='y')
    
    # Train model (batch_size optimized for GPU)
    model = classifier.train(
        max_epochs=100, 
        patience=20, 
        batch_size=2048,  # Larger batch size for GPU
        virtual_batch_size=256,
        num_workers=4  # Parallel data loading
    )
    
    # Evaluate performance
    results = classifier.evaluate()
    
    # Show model summary
    classifier.get_model_summary()
    
    # Show feature importance
    importance_df = classifier.plot_feature_importance(top_n=15)
    
    # Save model
    classifier.save_model('tabnet_binary_classifier_gpu.zip')
    
    # Clear GPU memory
    classifier.clear_gpu_memory()


# Fit and transform
df_transformed = fe.fit_transform(train_df)


test_df_transformed = fe.fit_transform(test_df)


submissions = classifier.predict(test_df_transformed)


pd.DataFrame(submissions).to_csv('submissions.csv')




