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


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Input data files are available in the read-only "../input/" directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Install required packages
print("Installing required packages...")
!pip install koolbox scikit-learn==1.5.2 prophet optuna --quiet

# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, kurtosis, skew, entropy
from xgboost import XGBRegressor
from sklearn.base import clone
from koolbox import Trainer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import optuna
import joblib
import gc
from scipy import stats
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

warnings.filterwarnings("ignore")

class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    target = "label"
    n_folds = 10  # Increased for better stability analysis
    n_stability_folds = 5  # Additional folds for stability testing
    seed = 42

    run_optuna = True
    n_optuna_trials = 250

@dataclass
class CompressionProfile:
    """Stores compression parameters for each feature"""
    method: str
    strength: float
    median: float
    mad: float
    noise_level: float
    outlier_score: float
    optimal_compression: float
    gradient_sensitivity: float
    mutual_info_score: float

@dataclass
class StabilityMetrics:
    """Stores stability metrics for a strategy"""
    mean_cv_score: float
    std_cv_score: float
    stability_score: float  # mean - 2*std (higher is better)
    fold_scores: List[float]
    inter_fold_correlation: float

class IntelligentAdaptiveCompressor:
    """Advanced compressor with gradient-based optimization and mutual information analysis"""
    
    def __init__(self):
        self.compression_profiles: Dict[str, CompressionProfile] = {}
        self.label_profile: CompressionProfile = None
        self.feature_importance: Dict[str, float] = {}
        self.gradient_map: Dict[str, float] = {}
        
    def analyze_feature_advanced(self, data: np.ndarray, target: np.ndarray, 
                                feature_name: str) -> Dict[str, float]:
        """Advanced feature analysis including gradient sensitivity and mutual information"""
        # Remove NaN values for analysis
        mask = ~(np.isnan(data) | np.isnan(target))
        clean_data = data[mask]
        clean_target = target[mask]
        
        if len(clean_data) < 10:
            return {
                'noise_level': 0,
                'outlier_score': 0,
                'distribution_score': 0,
                'gradient_sensitivity': 0,
                'mutual_info_score': 0,
                'optimal_compression': 0
            }
        
        # Basic statistics
        median = np.median(clean_data)
        mad = np.median(np.abs(clean_data - median))
        
        # Noise level estimation
        if len(clean_data) > 100:
            # FFT-based noise estimation
            fft = np.fft.fft(clean_data)
            frequencies = np.abs(fft)
            noise_level = np.sum(frequencies[len(frequencies)//2:]) / np.sum(frequencies)
        else:
            diffs = np.diff(np.sort(clean_data))
            noise_level = np.std(diffs[diffs != 0]) / (mad + 1e-8)
            
        # Outlier score with robust estimation
        if mad > 0:
            z_scores = np.abs((clean_data - median) / mad)
            outlier_score = np.mean(z_scores > 3)
            extreme_outlier_score = np.mean(z_scores > 6)
        else:
            outlier_score = 0
            extreme_outlier_score = 0
            
        # Distribution characteristics
        try:
            kurt = kurtosis(clean_data)
            skewness = abs(skew(clean_data))
            # Normalize distribution score
            distribution_score = 1 / (1 + np.exp(-(kurt / 10 + skewness / 5)))
        except:
            distribution_score = 0.5
            
        # Gradient sensitivity - how much does target change with feature
        try:
            # Sort by feature value
            sorted_idx = np.argsort(clean_data)
            sorted_feature = clean_data[sorted_idx]
            sorted_target = clean_target[sorted_idx]
            
            # Calculate local gradients
            feature_diffs = np.diff(sorted_feature)
            target_diffs = np.diff(sorted_target)
            
            # Avoid division by zero
            mask = feature_diffs != 0
            if np.any(mask):
                gradients = np.abs(target_diffs[mask] / feature_diffs[mask])
                gradient_sensitivity = np.percentile(gradients, 90)  # 90th percentile gradient
            else:
                gradient_sensitivity = 0
        except:
            gradient_sensitivity = 0
            
        # Mutual information score
        try:
            # Discretize for mutual information calculation
            n_bins = min(20, len(clean_data) // 10)
            feature_bins = pd.qcut(clean_data, n_bins, labels=False, duplicates='drop')
            target_bins = pd.qcut(clean_target, n_bins, labels=False, duplicates='drop')
            
            # Calculate mutual information
            from sklearn.metrics import mutual_info_score
            mutual_info = mutual_info_score(feature_bins, target_bins)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(n_bins)
            mutual_info_score_norm = mutual_info / max_entropy if max_entropy > 0 else 0
        except:
            mutual_info_score_norm = 0
            
        # Intelligent compression determination
        # Features with high mutual information should be compressed less
        # Features with high noise should be compressed more
        # Features with extreme outliers need careful handling
        
        compression_factors = {
            'noise': noise_level * 0.3,
            'outliers': outlier_score * 0.2 + extreme_outlier_score * 0.1,
            'distribution': distribution_score * 0.2,
            'gradient': (1 - np.tanh(gradient_sensitivity / 10)) * 0.1,  # Less compression for sensitive features
            'mutual_info': (1 - mutual_info_score_norm) * 0.3  # Less compression for informative features
        }
        
        optimal_compression = np.clip(sum(compression_factors.values()), 0, 0.8)
        
        return {
            'median': median,
            'mad': mad,
            'noise_level': noise_level,
            'outlier_score': outlier_score,
            'distribution_score': distribution_score,
            'gradient_sensitivity': gradient_sensitivity,
            'mutual_info_score': mutual_info_score_norm,
            'optimal_compression': optimal_compression,
            'compression_factors': compression_factors
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            feature_compression_override: Dict[str, float] = None,
            label_compression: float = 0.0,
            use_gradient_optimization: bool = True):
        """Learn compression parameters with gradient optimization"""
        
        print("Learning intelligent adaptive compression parameters...")
        
        # Calculate feature importance using random forest
        if use_gradient_optimization:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            self.feature_importance = dict(zip(X.columns, rf.feature_importances_))
        
        # Analyze each feature
        for col in X.columns:
            analysis = self.analyze_feature_advanced(X[col].values, y.values, col)
            
            # Use override if provided, otherwise use adaptive compression
            if feature_compression_override and col in feature_compression_override:
                compression = feature_compression_override[col]
            else:
                # Adjust compression based on feature importance
                base_compression = analysis['optimal_compression']
                if use_gradient_optimization and col in self.feature_importance:
                    importance_factor = 1 - self.feature_importance[col] / max(self.feature_importance.values())
                    compression = base_compression * (0.5 + 0.5 * importance_factor)
                else:
                    compression = base_compression
            
            # Select method based on comprehensive analysis
            if analysis['outlier_score'] > 0.15 or analysis['distribution_score'] > 0.7:
                method = 'robust_tanh'  # Custom robust version
            elif abs(analysis.get('skewness', 0)) > 2:
                method = 'log'
            elif analysis['gradient_sensitivity'] > 5:
                method = 'soft_clip'  # Gentle clipping for sensitive features
            else:
                method = 'sigmoid'
            
            self.compression_profiles[col] = CompressionProfile(
                method=method,
                strength=compression,
                median=analysis['median'],
                mad=analysis['mad'],
                noise_level=analysis['noise_level'],
                outlier_score=analysis['outlier_score'],
                optimal_compression=analysis['optimal_compression'],
                gradient_sensitivity=analysis['gradient_sensitivity'],
                mutual_info_score=analysis['mutual_info_score']
            )
            
            print(f"  {col}: {method} compression at {compression:.3f} "
                  f"(noise={analysis['noise_level']:.3f}, MI={analysis['mutual_info_score']:.3f}, "
                  f"gradient={analysis['gradient_sensitivity']:.3f})")
        
        # Analyze label if provided
        if y is not None and label_compression > 0:
            # For labels, use very conservative compression
            label_analysis = self.analyze_feature_advanced(y.values, y.values, 'label')
            self.label_profile = CompressionProfile(
                method='soft_clip',  # Always use soft clipping for labels
                strength=label_compression * 0.5,  # Reduce strength for safety
                median=label_analysis['median'],
                mad=label_analysis['mad'],
                noise_level=label_analysis['noise_level'],
                outlier_score=label_analysis['outlier_score'],
                optimal_compression=label_compression,
                gradient_sensitivity=0,
                mutual_info_score=1
            )
            print(f"  Label: soft_clip compression at {label_compression * 0.5:.3f}")
    
    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply learned compression with advanced methods"""
        X_compressed = X.copy()
        
        # Compress each feature using its profile
        for col in X.columns:
            if col in self.compression_profiles:
                profile = self.compression_profiles[col]
                X_compressed[col] = self._compress_data_advanced(
                    X[col].values,
                    profile
                )
        
        # Compress labels if profile exists
        y_compressed = y
        if y is not None and self.label_profile is not None:
            y_compressed = pd.Series(
                self._compress_data_advanced(y.values, self.label_profile),
                index=y.index
            )
        
        return X_compressed, y_compressed
    
    def _compress_data_advanced(self, data: np.ndarray, profile: CompressionProfile) -> np.ndarray:
        """Apply compression using advanced methods"""
        # Handle NaN values
        nan_mask = np.isnan(data)
        data_clean = data[~nan_mask]
        
        if len(data_clean) == 0 or profile.strength == 0:
            return data
        
        # Normalize using stored statistics
        if profile.mad > 0:
            normalized = (data - profile.median) / (profile.mad * 6)
        else:
            return data
        
        # Apply compression method
        if profile.method == 'robust_tanh':
            # Robust tanh with outlier handling
            alpha = 1 - profile.strength
            compressed = np.where(
                np.abs(normalized) < 3,
                np.tanh(normalized * alpha),
                np.sign(normalized) * (0.995 + 0.005 * np.tanh((np.abs(normalized) - 3) * alpha))
            )
        elif profile.method == 'soft_clip':
            # Soft clipping - gentle compression at extremes
            threshold = 2 * (1 - profile.strength + 0.1)
            compressed = np.where(
                np.abs(normalized) < threshold,
                normalized,
                np.sign(normalized) * (threshold + np.log1p(np.abs(normalized) - threshold) * 0.5)
            )
        elif profile.method == 'tanh':
            compressed = np.tanh(normalized * (1 - profile.strength))
        elif profile.method == 'sigmoid':
            compressed = 2 / (1 + np.exp(-normalized * (1 - profile.strength) * 2)) - 1
        elif profile.method == 'arctan':
            compressed = (2/np.pi) * np.arctan(normalized * (1 - profile.strength) * np.pi/2)
        elif profile.method == 'log':
            sign = np.sign(normalized)
            abs_norm = np.abs(normalized)
            compressed = sign * np.log1p(abs_norm * (1 - profile.strength)) / np.log1p(1 - profile.strength)
        
        # Scale back
        result = compressed * (profile.mad * 6) + profile.median
        
        # Preserve NaN values
        if nan_mask.any():
            full_result = np.full_like(data, np.nan, dtype=np.float64)
            full_result[~nan_mask] = result[~nan_mask]
            return full_result
        
        return result
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform predictions if labels were compressed"""
        if self.label_profile is None:
            return predictions
        
        # Approximate inverse transformation
        # This is a simplified version - in practice, you might want to fit an inverse function
        profile = self.label_profile
        
        # Normalize predictions
        normalized = (predictions - profile.median) / (profile.mad * 6)
        
        # Apply approximate inverse based on method
        if profile.method == 'soft_clip':
            # Inverse soft clip
            threshold = 2 * (1 - profile.strength + 0.1)
            inverse = np.where(
                np.abs(normalized) < threshold,
                normalized,
                np.sign(normalized) * (threshold + np.expm1(np.abs(normalized) - threshold) * 2)
            )
        else:
            # For other methods, use simple scaling
            inverse = normalized / (1 - profile.strength * 0.5)
        
        # Scale back
        return inverse * (profile.mad * 6) + profile.median
    
    def save(self, filepath: str):
        """Save compression profiles"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'compression_profiles': self.compression_profiles,
                'label_profile': self.label_profile,
                'feature_importance': self.feature_importance
            }, f)
    
    def load(self, filepath: str):
        """Load compression profiles"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.compression_profiles = data['compression_profiles']
            self.label_profile = data['label_profile']
            self.feature_importance = data.get('feature_importance', {})

def calculate_stability_metrics(scores_dict: Dict[str, List[float]]) -> StabilityMetrics:
    """Calculate stability metrics across folds"""
    all_scores = []
    for model_scores in scores_dict.values():
        all_scores.extend(model_scores)
    
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    
    # Calculate inter-fold correlation
    fold_predictions = defaultdict(list)
    for model, scores in scores_dict.items():
        for i, score in enumerate(scores):
            fold_predictions[i].append(score)
    
    # Average correlation between fold results
    correlations = []
    folds = list(fold_predictions.keys())
    for i in range(len(folds)):
        for j in range(i+1, len(folds)):
            corr = np.corrcoef(fold_predictions[folds[i]], fold_predictions[folds[j]])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    inter_fold_corr = np.mean(correlations) if correlations else 0
    
    return StabilityMetrics(
        mean_cv_score=mean_score,
        std_cv_score=std_score,
        stability_score=mean_score - 2 * std_score,  # Conservative stability metric
        fold_scores=all_scores,
        inter_fold_correlation=inter_fold_corr
    )

def reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
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
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe

def _pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

# Define model parameters with stability-focused tuning
lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.5625888953382505,
    "learning_rate": 0.029312951475451557,
    "min_child_samples": 63,
    "min_child_weight": 0.11456572852335424,
    "n_estimators": 126,
    "n_jobs": -1,
    "num_leaves": 37,
    "random_state": 42,
    "reg_alpha": 85.2476527854083,
    "reg_lambda": 99.38305361388907,
    "subsample": 0.450669817684892,
    "verbose": -1,
    "feature_fraction_seed": 42,  # For reproducibility
    "bagging_seed": 42
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.34695458228489784,
    "learning_rate": 0.031023014900595287,
    "min_child_samples": 30,
    "min_child_weight": 0.4727729225033618,
    "n_estimators": 220,
    "n_jobs": -1,
    "num_leaves": 58,
    "random_state": 42,
    "reg_alpha": 38.665994901468224,
    "reg_lambda": 92.76991677464294,
    "subsample": 0.4810891284493255,
    "verbose": -1,
    "feature_fraction_seed": 42,
    "bagging_seed": 42
}

xgb_params = {
    "colsample_bylevel": 0.4778015829774066,
    "colsample_bynode": 0.362764358742407,
    "colsample_bytree": 0.7107423488010493,
    "gamma": 1.7094857725240398,
    "learning_rate": 0.02213323588455387,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 39.352415706891264,
    "reg_lambda": 75.44843704068275,
    "subsample": 0.06566669853471274,
    "verbosity": 0,
    "seed": 42
}

# GANDALF Model Implementation
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class GANDALF(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, learning_rate=0.01, max_depth=5, 
                 feature_fraction=0.8, bagging_fraction=0.8, lambda_reg=1.0,
                 min_data_in_leaf=20, num_iterations=100, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.lambda_reg = lambda_reg
        self.min_data_in_leaf = min_data_in_leaf
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(random_state)  # For reproducibility
        
    def _build_network(self, input_dim):
        """Build the neural network architecture for GANDALF"""
        class GANDALFNet(nn.Module):
            def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
                super(GANDALFNet, self).__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers)
                
            def forward(self, x):
                return self.network(x)
        
        return GANDALFNet(input_dim).to(self.device)
    
    def fit(self, X, y):
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).reshape(-1, 1).to(self.device)
        
        # Build network
        self.model = self._build_network(X_tensor.shape[1])
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lambda_reg)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.num_iterations):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{self.num_iterations}], Loss: {total_loss/len(dataloader):.4f}')
        
        return self
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions

# AutoEncoder MLP
from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
import numpy as np

class AutoEncoderMLP(BaseEstimator, RegressorMixin):
    def __init__(self, num_columns, hidden_units, dropout_rates, lr=1e-3, seed=42):
        self.num_columns = num_columns
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.lr = lr
        self.seed = seed
        tf.random.set_seed(seed)
        self.model = self._build_model()
    
    def _build_model(self):
        inp = tf.keras.layers.Input(shape=(self.num_columns,))
        x0 = tf.keras.layers.BatchNormalization()(inp)

        encoder = tf.keras.layers.GaussianNoise(self.dropout_rates[0])(x0)
        encoder = tf.keras.layers.Dense(self.hidden_units[0])(encoder)
        encoder = tf.keras.layers.BatchNormalization()(encoder)
        encoder = tf.keras.layers.Activation('swish')(encoder)

        decoder = tf.keras.layers.Dropout(self.dropout_rates[1])(encoder)
        decoder = tf.keras.layers.Dense(self.num_columns, name='decoder')(decoder)

        x_reg = tf.keras.layers.Dense(self.hidden_units[1])(encoder)
        x_reg = tf.keras.layers.BatchNormalization()(x_reg)
        x_reg = tf.keras.layers.Activation('swish')(x_reg)
        x_reg = tf.keras.layers.Dropout(self.dropout_rates[2])(x_reg)

        out_reg = tf.keras.layers.Dense(1, activation='linear', name='target')(x_reg)

        model = tf.keras.models.Model(inputs=inp, outputs=[decoder, out_reg])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss={"decoder": tf.keras.losses.MeanSquaredError(),
                  "target": tf.keras.losses.MeanSquaredError()},
            loss_weights={"decoder": 0.3, "target": 1.0}
        )
        return model

    def fit(self, X, y):
        self.model.fit(
            X, {"decoder": X, "target": y},
            epochs=50,
            batch_size=8192,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5)
            ],
            verbose=0
        )
        return self

    def predict(self, X):
        _, y_pred = self.model.predict(X, verbose=0)
        return y_pred.flatten()

def run_pipeline_with_stability_analysis(feature_compression: float, 
                                       label_compression: float,
                                       use_adaptive: bool,
                                       strategy_name: str,
                                       n_stability_folds: int = 5):
    """
    Run pipeline with stability analysis using multiple fold configurations
    """
    print(f"\n{'='*80}")
    print(f"Running pipeline with stability analysis: {strategy_name}")
    print(f"Feature compression: {feature_compression:.2f}, Label compression: {label_compression:.2f}")
    print(f"Adaptive: {use_adaptive}, Stability folds: {n_stability_folds}")
    print(f"{'='*80}\n")
    
    # Load data
    train = pd.read_parquet(CFG.train_path).reset_index(drop=True)
    test = pd.read_parquet(CFG.test_path).reset_index(drop=True)
    
    # Define features
    X_FEATURES = ['X363', 'X321', 'X405', 'X730', 'X523', 'X756', 'X589', 'X462', 'X779',
                    'X25', 'X532', 'X520', 'X329', 'X383', 'X751', 'X535', 'X639', 'X596', 'X761',
                "X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51",
                "X598", "X385", "X603", "X674", "X415", "X345", "X174", "X178", "X168", "X612",
                "bid_qty", "ask_qty", "buy_qty", "sell_qty"]
    
    selected_columns = X_FEATURES + ["volume"]
    
    # Select columns
    train = train[selected_columns + [CFG.target]]
    test = test[selected_columns]
    
    # Prepare data
    X_train = train.drop(CFG.target, axis=1)
    y_train = train[CFG.target]
    X_test = test
    
    # Initialize and fit compressor
    compressor = IntelligentAdaptiveCompressor()
    
    if use_adaptive:
        compressor.fit(X_train, y_train, label_compression=label_compression, use_gradient_optimization=True)
    else:
        feature_overrides = {col: feature_compression for col in X_train.columns}
        compressor.fit(X_train, y_train, 
                      feature_compression_override=feature_overrides,
                      label_compression=label_compression,
                      use_gradient_optimization=False)
    
    # Apply compression
    X_train_compressed, y_train_compressed = compressor.transform(X_train, y_train)
    X_test_compressed, _ = compressor.transform(X_test, None)
    
    # Save compressor
    compressor.save(f'compressor_{strategy_name}.pkl')
    
    # Reduce memory
    X_train_compressed = reduce_mem_usage(X_train_compressed, "train")
    X_test_compressed = reduce_mem_usage(X_test_compressed, "test")
    
    # Run stability analysis with different fold configurations
    stability_results = {}
    all_test_predictions = []
    
    for fold_config in range(n_stability_folds):
        print(f"\nStability fold configuration {fold_config + 1}/{n_stability_folds}")
        
        # Create different fold splits
        if fold_config == 0:
            cv = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
        elif fold_config == 1:
            cv = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed + 1)
        elif fold_config == 2:
            cv = KFold(n_splits=8, shuffle=True, random_state=CFG.seed + 2)
        elif fold_config == 3:
            # Time series split for temporal stability
            cv = TimeSeriesSplit(n_splits=min(5, CFG.n_folds))
        else:
            cv = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed + fold_config)
        
        # Initialize storage for this fold configuration
        scores = {}
        test_preds = {}
        
        # Train models
        # LightGBM (gbdt)
        lgbm_trainer = Trainer(
            LGBMRegressor(**lgbm_params),
            cv=cv,
            metric=_pearsonr,
            task="regression",
            metric_precision=6
        )
        lgbm_trainer.fit(X_train_compressed, y_train_compressed)
        scores["LightGBM (gbdt)"] = lgbm_trainer.fold_scores
        test_preds["LightGBM (gbdt)"] = lgbm_trainer.predict(X_test_compressed)
        
        # LightGBM (goss)
        lgbm_goss_trainer = Trainer(
            LGBMRegressor(**lgbm_goss_params),
            cv=cv,
            metric=_pearsonr,
            task="regression",
            metric_precision=6
        )
        lgbm_goss_trainer.fit(X_train_compressed, y_train_compressed)
        scores["LightGBM (goss)"] = lgbm_goss_trainer.fold_scores
        test_preds["LightGBM (goss)"] = lgbm_goss_trainer.predict(X_test_compressed)
        
        # XGBoost
        xgb_trainer = Trainer(
            XGBRegressor(**xgb_params),
            cv=cv,
            metric=_pearsonr,
            task="regression",
            metric_precision=6
        )
        xgb_trainer.fit(X_train_compressed, y_train_compressed)
        scores["XGBoost"] = xgb_trainer.fold_scores
        test_preds["XGBoost"] = xgb_trainer.predict(X_test_compressed)
        
        # Skip GANDALF and AutoEncoder for stability analysis (too slow)
        # but calculate ensemble from the three main models
        
        # Calculate weighted predictions for this fold
        ensemble_weights = {
            "LightGBM (gbdt)": 0.4,
            "LightGBM (goss)": 0.3,
            "XGBoost": 0.3
        }
        
        weighted_test_pred = np.zeros(len(X_test))
        for model_name, weight in ensemble_weights.items():
            weighted_test_pred += weight * test_preds[model_name]
        
        all_test_predictions.append(weighted_test_pred)
        stability_results[f'fold_{fold_config}'] = scores
    
    # Calculate overall stability metrics
    all_scores_flat = {}
    for fold_results in stability_results.values():
        for model, scores in fold_results.items():
            if model not in all_scores_flat:
                all_scores_flat[model] = []
            all_scores_flat[model].extend(scores)
    
    stability_metrics = calculate_stability_metrics(all_scores_flat)
    
    # Average predictions across stability folds
    final_predictions = np.mean(all_test_predictions, axis=0)
    
    # If labels were compressed, apply inverse transformation
    if label_compression > 0 and compressor.label_profile is not None:
        final_predictions = compressor.inverse_transform_predictions(final_predictions)
    
    # Clean up memory
    del X_train, y_train, X_test, train, test
    gc.collect()
    
    return final_predictions, stability_metrics, compressor, all_test_predictions

# Dynamic weight optimization based on CV performance
def optimize_ensemble_weights(predictions_dict: Dict[str, np.ndarray], 
                            cv_scores: Dict[str, float]) -> Dict[str, float]:
    """Optimize ensemble weights based on CV performance and prediction diversity"""
    
    # Normalize CV scores
    total_score = sum(cv_scores.values())
    if total_score == 0:
        return {k: 1/len(cv_scores) for k in cv_scores.keys()}
    
    base_weights = {k: v/total_score for k, v in cv_scores.items()}
    
    # Calculate prediction diversity (correlation between models)
    correlations = {}
    models = list(predictions_dict.keys())
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:
                corr = np.corrcoef(predictions_dict[model1], predictions_dict[model2])[0, 1]
                correlations[f"{model1}_{model2}"] = corr
    
    # Adjust weights based on diversity (less correlated models get higher weights)
    diversity_bonus = {}
    for model in models:
        model_correlations = []
        for corr_key, corr_val in correlations.items():
            if model in corr_key:
                model_correlations.append(corr_val)
        
        avg_correlation = np.mean(model_correlations) if model_correlations else 0.5
        diversity_bonus[model] = 1 - avg_correlation  # Higher bonus for less correlated models
    
    # Combine base weights with diversity bonus
    final_weights = {}
    for model in models:
        performance_weight = base_weights.get(model, 1/len(models))
        diversity_weight = diversity_bonus.get(model, 0.5)
        final_weights[model] = performance_weight * 0.7 + diversity_weight * 0.3
    
    # Normalize final weights
    total_weight = sum(final_weights.values())
    final_weights = {k: v/total_weight for k, v in final_weights.items()}
    
    return final_weights

# Main execution
if __name__ == "__main__":
    # Extended compression grid with more intelligent strategies
    compression_grid = [
        # (feature_compression, label_compression, use_adaptive, name)
        (0.0, 0.0, False, "baseline_no_compression"),
        (0.2, 0.0, False, "mild_features_20pct"),
        (0.4, 0.0, False, "moderate_features_40pct"),
        (0.6, 0.0, False, "strong_features_60pct"),
        (0.3, 0.05, False, "balanced_30_5pct"),
        (0.5, 0.1, False, "balanced_50_10pct"),
        (0.0, 0.0, True, "adaptive_auto"),
        (0.0, 0.05, True, "adaptive_label_5pct"),
        (0.0, 0.1, True, "adaptive_label_10pct"),
        (0.0, 0.15, True, "adaptive_label_15pct"),
    ]
    
    # Store results
    all_results = {}
    
    # Run grid search with stability analysis
    for feature_comp, label_comp, use_adaptive, name in compression_grid:
        try:
            predictions, stability_metrics, compressor, fold_predictions = \
                run_pipeline_with_stability_analysis(
                    feature_comp, label_comp, use_adaptive, name, 
                    n_stability_folds=CFG.n_stability_folds
                )
            
            all_results[name] = {
                'predictions': predictions,
                'stability_metrics': stability_metrics,
                'compressor': compressor,
                'fold_predictions': fold_predictions
            }
            
            # Save individual submission
            sub = pd.read_csv(CFG.sample_sub_path)
            sub["prediction"] = predictions
            sub.to_csv(f"submission_{name}.csv", index=False)
            print(f"\nSaved submission_{name}.csv")
            print(f"Stability score: {stability_metrics.stability_score:.4f}")
            print(f"Mean CV: {stability_metrics.mean_cv_score:.4f} ± {stability_metrics.std_cv_score:.4f}")
            
        except Exception as e:
            print(f"\nError in strategy {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Select top 3 most stable strategies
    print("\n" + "="*80)
    print("SELECTING TOP 3 MOST STABLE STRATEGIES")
    print("="*80)
    
    stability_scores = {
        name: results['stability_metrics'].stability_score 
        for name, results in all_results.items()
    }
    
    top_3_strategies = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("\nTop 3 most stable strategies:")
    for i, (strategy, score) in enumerate(top_3_strategies, 1):
        metrics = all_results[strategy]['stability_metrics']
        print(f"{i}. {strategy}: stability={score:.4f}, "
              f"mean={metrics.mean_cv_score:.4f}, std={metrics.std_cv_score:.4f}")
    
    # Create final ensemble from top 3 strategies
    print("\nCreating final ensemble from top 3 strategies...")
    
    # Calculate dynamic weights based on stability and performance
    top_predictions = {}
    top_cv_scores = {}
    
    for strategy, _ in top_3_strategies:
        top_predictions[strategy] = all_results[strategy]['predictions']
        top_cv_scores[strategy] = all_results[strategy]['stability_metrics'].mean_cv_score
    
    # Optimize weights
    optimal_weights = optimize_ensemble_weights(top_predictions, top_cv_scores)
    
    print("\nOptimal ensemble weights:")
    for strategy, weight in optimal_weights.items():
        print(f"  {strategy}: {weight:.3f}")
    
    # Create final ensemble
    final_ensemble = np.zeros_like(list(top_predictions.values())[0])
    for strategy, weight in optimal_weights.items():
        final_ensemble += weight * top_predictions[strategy]
    
    # Save final ensemble
    sub = pd.read_csv(CFG.sample_sub_path)
    sub["prediction"] = final_ensemble
    sub.to_csv("submission_final_stable_ensemble.csv", index=False)
    print("\nSaved submission_final_stable_ensemble.csv")
    
    # Create comprehensive visualization
    # 1. Stability comparison plot
    plt.figure(figsize=(12, 8))
    strategies = list(stability_scores.keys())
    scores = list(stability_scores.values())
    means = [all_results[s]['stability_metrics'].mean_cv_score for s in strategies]
    stds = [all_results[s]['stability_metrics'].std_cv_score for s in strategies]
    
    x = np.arange(len(strategies))
    plt.bar(x, scores, alpha=0.7, label='Stability Score')
    plt.errorbar(x, means, yerr=stds, fmt='o', color='red', label='Mean ± Std')
    
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Strategy Stability Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Prediction correlation heatmap for top strategies
    plt.figure(figsize=(8, 6))
    corr_matrix = np.corrcoef([top_predictions[s] for s, _ in top_3_strategies])
    sns.heatmap(corr_matrix, 
                xticklabels=[s for s, _ in top_3_strategies],
                yticklabels=[s for s, _ in top_3_strategies],
                annot=True, fmt='.3f', cmap='coolwarm', center=0.5)
    plt.title('Top 3 Strategies Prediction Correlation')
    plt.tight_layout()
    plt.savefig('top_strategies_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Feature compression analysis for adaptive strategies
    adaptive_strategies = [name for _, _, use_adaptive, name in compression_grid if use_adaptive]
    if adaptive_strategies and any(s in all_results for s in adaptive_strategies):
        # Get one adaptive compressor for analysis
        adaptive_name = next(s for s in adaptive_strategies if s in all_results)
        compressor = all_results[adaptive_name]['compressor']
        
        # Extract compression strengths
        feature_data = []
        for feat, profile in list(compressor.compression_profiles.items())[:30]:  # Top 30 features
            feature_data.append({
                'Feature': feat,
                'Compression': profile.strength,
                'Noise': profile.noise_level,
                'MI Score': profile.mutual_info_score,
                'Gradient': profile.gradient_sensitivity
            })
        
        feature_df = pd.DataFrame(feature_data)
        feature_df = feature_df.sort_values('Compression', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Compression strength by feature
        ax1.barh(feature_df['Feature'], feature_df['Compression'])
        ax1.set_xlabel('Compression Strength')
        ax1.set_title('Adaptive Compression by Feature')
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot: Noise vs MI Score
        scatter = ax2.scatter(feature_df['Noise'], feature_df['MI Score'], 
                            c=feature_df['Compression'], cmap='viridis', s=100)
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Mutual Information Score')
        ax2.set_title('Feature Analysis: Noise vs Information')
        plt.colorbar(scatter, ax=ax2, label='Compression')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('adaptive_compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save comprehensive summary
    summary_data = []
    for name, results in all_results.items():
        metrics = results['stability_metrics']
        summary_data.append({
            'Strategy': name,
            'Mean_CV': metrics.mean_cv_score,
            'Std_CV': metrics.std_cv_score,
            'Stability_Score': metrics.stability_score,
            'Inter_Fold_Correlation': metrics.inter_fold_correlation,
            'Is_Top_3': name in [s for s, _ in top_3_strategies],
            'Final_Weight': optimal_weights.get(name, 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Stability_Score', ascending=False)
    summary_df.to_csv('intelligent_compression_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("INTELLIGENT COMPRESSION PIPELINE COMPLETED!")
    print("="*80)
    print("Generated files:")
    print("- Individual submissions for each strategy")
    print("- submission_final_stable_ensemble.csv (TOP 3 ENSEMBLE)")
    print("- Compression profiles (.pkl files)")
    print("- stability_analysis.png")
    print("- top_strategies_correlation.png")
    print("- adaptive_compression_analysis.png")
    print("- intelligent_compression_summary.csv")
    print("="*80)

