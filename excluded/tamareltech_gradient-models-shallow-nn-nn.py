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


# ==================== INSTALLATIONS ====================
# !pip install -q xgboost lightgbm tensorflow scikit-learn pandas numpy

# ==================== IMPORTS ====================
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
import gc
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# ==================== CONFIGURATION ====================
class CFG:
    # Data paths
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Feature selection parameters
    feature_selection_sample_size = 100000  # Sample size for feature selection
    top_features_percentage = 0.3  # Select top 30% features
    
    # Model parameters
    use_gpu = True
    random_state = 42
    
    # Neural network parameters
    nn_epochs = 80
    nn_batch_size = 2048
    nn_learning_rate = 0.001
    nn_patience = 10
    
    # Cross-validation
    n_folds = 5

# ==================== UTILITY FUNCTIONS ====================
def reduce_mem_usage(df, verbose=True):
    """Reduce memory usage of dataframe"""
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
                df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by handling inf, -inf, and NaN values"""
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if any(keyword in col.lower() for keyword in ['qty', 'volume']):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())
    
    return df

def create_time_weights(n_samples, decay_factor=0.95):
    """Create time-based weights with exponential decay"""
    positions = np.arange(n_samples)
    weights = decay_factor ** (n_samples - positions - 1)
    return weights / weights.sum() * len(weights)

# ==================== GRADIENT BOOSTING FEATURE SELECTOR ====================
class GradientBoostingFeatureSelector:
    """Feature selector using XGBoost and LightGBM"""
    
    def __init__(self):
        self.feature_importance = {}
        self.selected_features = []
        
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate feature importance using gradient boosting models"""
        print("\nCalculating feature importance...")
        
        # Split data for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # XGBoost model
        print("Training XGBoost...")
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': CFG.random_state,
            'tree_method': 'gpu_hist' if CFG.use_gpu else 'hist',
            'verbosity': 0
        }
        
        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        xgb_importance = dict(zip(X.columns, xgb_model.feature_importances_))
        
        # LightGBM model
        print("Training LightGBM...")
        lgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_leaves': 50,
            'random_state': CFG.random_state,
            'device': 'gpu' if CFG.use_gpu else 'cpu',
            'verbosity': -1
        }
        
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        lgb_importance = dict(zip(X.columns, lgb_model.feature_importances_))
        
        # Combine importances (average of XGBoost and LightGBM)
        combined_importance = {}
        for feature in X.columns:
            combined_importance[feature] = (
                xgb_importance[feature] + lgb_importance[feature]
            ) / 2
        
        # Normalize
        max_importance = max(combined_importance.values())
        for feature in combined_importance:
            combined_importance[feature] /= max_importance
        
        self.feature_importance = combined_importance
        
        # Clean up
        del xgb_model, lgb_model
        gc.collect()
        
        return combined_importance
    
    def select_top_features(self, top_percentage: float = 0.3) -> List[str]:
        """Select top features based on importance"""
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        n_features = int(len(sorted_features) * top_percentage)
        self.selected_features = [feature for feature, _ in sorted_features[:n_features]]
        
        print(f"\nSelected {len(self.selected_features)} features out of {len(sorted_features)}")
        print(f"Top 10 features: {self.selected_features[:10]}")
        
        return self.selected_features

# ==================== NEURAL NETWORK MODEL ====================
class CryptoNeuralNetwork:
    """Deep neural network for crypto prediction"""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.model = None
        self.scaler = RobustScaler()
        
    def build_model(self) -> models.Model:
        """Build neural network architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.n_features,)),
            
            # First block
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth block
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fifth block
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            
            # Output layer
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=CFG.nn_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series,
              sample_weights: Optional[np.ndarray] = None) -> None:
        """Train the neural network"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build model
        self.model = self.build_model()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CFG.nn_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=CFG.nn_epochs,
            batch_size=CFG.nn_batch_size,
            sample_weight=sample_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, batch_size=CFG.nn_batch_size).flatten()

# ==================== MAIN PIPELINE ====================
def main():
    print("="*80)
    print("SIMPLIFIED CRYPTO PREDICTION PIPELINE")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train = pd.read_parquet(CFG.train_path)
    test = pd.read_parquet(CFG.test_path)
    sample_submission = pd.read_csv(CFG.sample_sub_path)
    
    # Reduce memory usage
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    
    # Clean data
    train = clean_data(train)
    test = clean_data(test)
    
    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Separate features and target
    feature_cols = [col for col in train.columns if col not in ['label', 'timestamp']]
    X_train_full = train[feature_cols]
    y_train_full = train['label']
    X_test = test[feature_cols]
    
    # Ensure critical features are always included
    critical_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    
    # ==================== FEATURE SELECTION ====================
    print("\n" + "="*60)
    print("FEATURE SELECTION PHASE")
    print("="*60)
    
    # Use recent data for feature selection
    sample_size = min(CFG.feature_selection_sample_size, len(X_train_full))
    X_sample = X_train_full.tail(sample_size).reset_index(drop=True)
    y_sample = y_train_full.tail(sample_size).reset_index(drop=True)
    
    # Initialize feature selector
    selector = GradientBoostingFeatureSelector()
    
    # Calculate feature importance
    feature_importance = selector.calculate_feature_importance(X_sample, y_sample)
    
    # Select top features
    selected_features = selector.select_top_features(CFG.top_features_percentage)
    
    # Ensure critical features are included
    for feature in critical_features:
        if feature in feature_cols and feature not in selected_features:
            selected_features.append(feature)
    
    print(f"\nFinal number of selected features: {len(selected_features)}")
    
    # Apply feature selection
    X_train_selected = X_train_full[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Save feature importance
    feature_report = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values()),
        'selected': [f in selected_features for f in feature_importance.keys()]
    }).sort_values('importance', ascending=False)
    feature_report.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to feature_importance.csv")
    
    # ==================== NEURAL NETWORK TRAINING ====================
    print("\n" + "="*60)
    print("NEURAL NETWORK TRAINING PHASE")
    print("="*60)
    
    # Create time-based weights
    sample_weights = create_time_weights(len(X_train_selected))
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=CFG.n_folds)
    oof_predictions = np.zeros(len(X_train_selected))
    test_predictions = np.zeros(len(X_test_selected))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_selected)):
        print(f"\n--- Fold {fold + 1}/{CFG.n_folds} ---")
        
        X_fold_train = X_train_selected.iloc[train_idx]
        X_fold_val = X_train_selected.iloc[val_idx]
        y_fold_train = y_train_full.iloc[train_idx]
        y_fold_val = y_train_full.iloc[val_idx]
        fold_weights = sample_weights[train_idx]
        
        # Train neural network
        nn_model = CryptoNeuralNetwork(n_features=len(selected_features))
        nn_model.train(
            X_fold_train, y_fold_train,
            X_fold_val, y_fold_val,
            sample_weights=fold_weights
        )
        
        # Make predictions
        val_pred = nn_model.predict(X_fold_val)
        test_pred = nn_model.predict(X_test_selected)
        
        # Store predictions
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / CFG.n_folds
        
        # Calculate fold score
        fold_score = pearsonr(y_fold_val, val_pred)[0]
        fold_scores.append(fold_score)
        print(f"Fold {fold + 1} Pearson correlation: {fold_score:.4f}")
        
        # Clean up
        del nn_model
        gc.collect()
        tf.keras.backend.clear_session()
    
    # ==================== FINAL EVALUATION ====================
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # Calculate overall performance
    overall_score = pearsonr(y_train_full, oof_predictions)[0]
    print(f"\nOverall OOF Pearson correlation: {overall_score:.4f}")
    print(f"Average fold score: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    
    # Create submission
    submission = sample_submission.copy()
    submission['prediction'] = test_predictions
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved to submission.csv")
    
    # Save results summary
    results = {
        'n_original_features': len(feature_cols),
        'n_selected_features': len(selected_features),
        'overall_oof_score': float(overall_score),
        'fold_scores': [float(s) for s in fold_scores],
        'mean_fold_score': float(np.mean(fold_scores)),
        'std_fold_score': float(np.std(fold_scores))
    }
    
    with open('results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults summary saved to results_summary.json")
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()

