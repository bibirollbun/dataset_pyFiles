import pandas as pd
import numpy as np
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Gradient Boosting Models
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set style and random seed
sns.set_style('whitegrid')
np.random.seed(42)


class HybridCryptoPredictor:
    """
    Hybrid prediction system that combines autoregressive features from labels
    during training with market microstructure features for robust predictions.
    """
    
    def __init__(self, lookback_windows=None):
        """
        Initialize the hybrid predictor.
        
        Parameters:
        -----------
        lookback_windows : list of int
            Window sizes for creating lagged features
        """
        self.lookback_windows = lookback_windows or [1, 5, 10, 20, 30, 60]
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        self.validation_scores = []
        
    def create_basic_features(self, df):
        """
        Create basic features from market data that are available in both train and test.
        These features ensure we can make predictions on the test set.
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic market features - these are available in both train and test
        market_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        
        # Copy original features
        for col in market_cols:
            if col in df.columns:
                features[col] = df[col].fillna(0)
        
        # Order book imbalance
        if 'bid_qty' in df.columns and 'ask_qty' in df.columns:
            features['order_imbalance'] = (
                (df['bid_qty'] - df['ask_qty']) / 
                (df['bid_qty'] + df['ask_qty'] + 1e-8)
            )
        
        # Trade flow imbalance
        if 'buy_qty' in df.columns and 'sell_qty' in df.columns:
            features['trade_imbalance'] = (
                (df['buy_qty'] - df['sell_qty']) / 
                (df['buy_qty'] + df['sell_qty'] + 1e-8)
            )
        
        # Volume ratios
        if 'volume' in df.columns:
            for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']:
                if col in df.columns:
                    features[f'{col}_volume_ratio'] = df[col] / (df['volume'] + 1e-8)
        
        # Log transforms for volume features
        for col in market_cols:
            if col in features.columns:
                features[f'log_{col}'] = np.log1p(features[col])
        
        # Replace any infinities or NaN values
        features = features.replace([np.inf, -np.inf], 0).fillna(0)
        
        return features
    
    def create_autoregressive_features(self, df, target_col='label'):
        """
        Create autoregressive features from the target variable.
        Only used during training when we have access to historical labels.
        """
        ar_features = pd.DataFrame(index=df.index)
        
        if target_col in df.columns:
            target = df[target_col]
            
            # Create lagged features
            for window in self.lookback_windows:
                ar_features[f'target_lag_{window}'] = target.shift(window)
                
                # Rolling statistics
                ar_features[f'target_ma_{window}'] = target.rolling(
                    window=window, min_periods=1
                ).mean()
                
                ar_features[f'target_std_{window}'] = target.rolling(
                    window=window, min_periods=1
                ).std()
                
                # Momentum features
                ar_features[f'target_momentum_{window}'] = target - target.shift(window)
        
        # Fill NaN values
        ar_features = ar_features.fillna(0)
        
        return ar_features
    
    def prepare_features(self, df, is_training=True):
        """
        Prepare all features for modeling.
        """
        # Get basic market features (available for both train and test)
        basic_features = self.create_basic_features(df)
        
        if is_training:
            # Add autoregressive features during training
            ar_features = self.create_autoregressive_features(df)
            features = pd.concat([basic_features, ar_features], axis=1)
        else:
            # Test set only has basic features
            features = basic_features
        
        # Add a subset of the anonymized features (X_1 to X_50 to keep it simple)
        anon_cols = [col for col in df.columns if col.startswith('X_')][:50]
        for col in anon_cols:
            if col in df.columns:
                features[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train an ensemble of models.
        """
        print("Training ensemble models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # XGBoost model
        self.models['xgb'] = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=800,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.7,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM model
        self.models['lgb'] = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=5,
            lambda_l1=0.05,
            lambda_l2=1.0,
            min_data_in_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        
        # Train models
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train XGBoost
            self.models['xgb'].fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Train LightGBM
            self.models['lgb'].fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Calculate validation scores
            val_scores = {}
            for name, model in self.models.items():
                val_pred = model.predict(X_val_scaled)
                val_corr, _ = pearsonr(y_val, val_pred)
                val_scores[name] = val_corr
                print(f"{name} validation correlation: {val_corr:.4f}")
            
            # Ensemble prediction
            ensemble_pred = np.mean([
                self.models['xgb'].predict(X_val_scaled),
                self.models['lgb'].predict(X_val_scaled)
            ], axis=0)
            ensemble_corr, _ = pearsonr(y_val, ensemble_pred)
            print(f"Ensemble validation correlation: {ensemble_corr:.4f}")
            
            return ensemble_corr
        else:
            # Train without validation
            self.models['xgb'].fit(X_train_scaled, y_train)
            self.models['lgb'].fit(X_train_scaled, y_train)
            return None
    
    def predict(self, features):
        """
        Generate ensemble predictions.
        """
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(features_scaled)
            predictions.append(pred)
        
        # Simple average ensemble
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def time_series_validation(self, features, labels, n_splits=5):
        """
        Perform time series cross-validation.
        """
        print(f"\nPerforming {n_splits}-fold time series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
            print(f"\nFold {fold + 1}/{n_splits}")
            print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
            
            X_train_fold = features.iloc[train_idx]
            y_train_fold = labels.iloc[train_idx]
            X_val_fold = features.iloc[val_idx]
            y_val_fold = labels.iloc[val_idx]
            
            fold_score = self.train_model(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold
            )
            
            if fold_score is not None:
                cv_scores.append(fold_score)
        
        self.validation_scores = cv_scores
        print(f"\nCV scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        return cv_scores


def main():
    """
    Main execution pipeline.
    """
    print("DRW Crypto Market Prediction - Hybrid Baseline Solution")
    print("=" * 60)
    
    # Load training data
    print("\nLoading training data...")
    train_df = pd.read_parquet(
        '/kaggle/input/drw-crypto-market-prediction/train.parquet',
        engine='pyarrow'
    )
    print(f"Training data shape: {train_df.shape}")
    
    # Inspect columns
    print("\nInspecting data columns...")
    print(f"Total columns: {len(train_df.columns)}")
    
    # Check for timestamp column
    if 'timestamp' in train_df.columns:
        print("Timestamp column found")
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        cutoff_date = train_df['timestamp'].max() - pd.Timedelta(days=240)
        train_df = train_df[train_df['timestamp'] >= cutoff_date].reset_index(drop=True)
        print(f"Using data from {cutoff_date} onwards ({len(train_df)} rows)")
    else:
        print("No timestamp column found - using all data")
        # Use last portion of data assuming temporal ordering
        cutoff_idx = max(0, len(train_df) - 300000)  # Use approximately last 300k rows
        train_df = train_df.iloc[cutoff_idx:].reset_index(drop=True)
        print(f"Using last {len(train_df)} rows of data")
    
    # Display available columns
    market_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'label']
    available_market_cols = [col for col in market_cols if col in train_df.columns]
    print(f"\nAvailable market columns: {available_market_cols}")
    
    anonymized_cols = [col for col in train_df.columns if col.startswith('X_')]
    print(f"Number of anonymized features: {len(anonymized_cols)}")
    
    # Basic statistics on label
    if 'label' in train_df.columns:
        print(f"\nLabel statistics:")
        print(f"  Mean: {train_df['label'].mean():.6f}")
        print(f"  Std: {train_df['label'].std():.6f}")
        print(f"  Min: {train_df['label'].min():.6f}")
        print(f"  Max: {train_df['label'].max():.6f}")
    
    # Initialize predictor
    predictor = HybridCryptoPredictor()
    
    # Prepare features
    print("\nPreparing features...")
    train_features = predictor.prepare_features(train_df, is_training=True)
    train_labels = train_df['label']
    
    print(f"\nFeature matrix shape: {train_features.shape}")
    print(f"Features: {len(train_features.columns)} total")
    print(f"  - Market features: {len([c for c in train_features.columns if not c.startswith('X_') and not c.startswith('target_')])}")
    print(f"  - Autoregressive features: {len([c for c in train_features.columns if c.startswith('target_')])}")
    print(f"  - Anonymized features: {len([c for c in train_features.columns if c.startswith('X_')])}")
    
    # Time series validation
    cv_scores = predictor.time_series_validation(train_features, train_labels, n_splits=5)
    
    # Train final model
    print("\nTraining final model on all data...")
    predictor.train_model(train_features, train_labels)
    
    # Feature importance
    print("\nTop 20 most important features:")
    for name, model in predictor.models.items():
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': train_features.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            print(f"\n{name.upper()} top features:")
            for idx, row in importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Generate test predictions
    print("\nLoading test data...")
    test_df = pd.read_parquet(
        '/kaggle/input/drw-crypto-market-prediction/test.parquet',
        engine='pyarrow'
    )
    print(f"Test data shape: {test_df.shape}")
    
    # Prepare test features (no autoregressive features available)
    print("\nPreparing test features...")
    test_features = predictor.prepare_features(test_df, is_training=False)
    
    # Ensure test features match training features
    # Add missing columns with zeros
    for col in train_features.columns:
        if col not in test_features.columns:
            test_features[col] = 0
    
    # Reorder columns to match training
    test_features = test_features[train_features.columns]
    
    print(f"Test feature matrix shape: {test_features.shape}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predictor.predict(test_features)
    
    # Create submission
    submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
    submission['prediction'] = predictions
    submission.to_csv('submission.csv', index=False)
    
    print("\nSubmission created successfully!")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution of predictions
    axes[0, 0].hist(predictions, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Test Predictions')
    
    # Sample predictions
    axes[0, 1].plot(predictions[:500], alpha=0.7, linewidth=0.8)
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Predicted Value')
    axes[0, 1].set_title('First 500 Test Predictions')
    
    # Training label distribution vs predictions
    axes[1, 0].hist(train_labels, bins=50, alpha=0.5, label='Training Labels', density=True)
    axes[1, 0].hist(predictions, bins=50, alpha=0.5, label='Test Predictions', density=True)
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    
    # CV scores
    if cv_scores:
        axes[1, 1].plot(range(1, len(cv_scores) + 1), cv_scores, 'o-', markersize=8)
        axes[1, 1].axhline(np.mean(cv_scores), color='red', linestyle='--', label=f'Mean: {np.mean(cv_scores):.4f}')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Pearson Correlation')
        axes[1, 1].set_title('Cross-Validation Performance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return predictor, submission


if __name__ == "__main__":
    predictor, submission = main()







