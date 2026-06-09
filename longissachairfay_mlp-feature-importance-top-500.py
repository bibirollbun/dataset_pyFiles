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


"""
DRW Crypto Market Prediction - Complete Working Pipeline
========================================================
Memory-optimized neural network solution for crypto prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gc
import warnings
from pathlib import Path
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import logging
import joblib

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================
class Config:
    # Paths
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    OUTPUT_DIR = Path("/kaggle/working")
    
    # Data settings
    N_FOLDS = 3
    TOP_N_FEATURES = 150
    SAMPLE_SIZE = 50000
    RANDOM_STATE = 42
    
    # Neural Network
    USE_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_GPU else "cpu")
    BATCH_SIZE = 2048
    HIDDEN_DIMS = [512, 256, 128]
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 10

logger.info(f"Using device: {Config.DEVICE}")

# =========================
# Memory Optimization
# =========================
def optimize_memory(df):
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        if col != 'timestamp':
            df[col] = df[col].astype(np.float32)
    return df

# =========================
# Feature Engineering
# =========================
class FeatureEngineer:
    """Simple but effective feature engineering"""
    
    def __init__(self):
        self.feature_cols = None
        
    def create_features(self, df):
        """Create market microstructure features"""
        logger.info("Creating features...")
        
        # Basic market features
        if all(col in df.columns for col in ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]):
            eps = 1e-10
            
            # Core features
            df['book_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + eps)
            df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
            df['order_flow_ratio'] = df['net_order_flow'] / (df['volume'] + eps)
            df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + eps)
            df['trade_intensity'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + eps)
            df['total_depth'] = df['bid_qty'] + df['ask_qty']
            df['depth_ratio'] = df['bid_qty'] / (df['ask_qty'] + eps)
            df['vpin_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
            
            # Convert to float32
            for col in ['book_imbalance', 'net_order_flow', 'order_flow_ratio', 'kyle_lambda',
                       'trade_intensity', 'total_depth', 'depth_ratio', 'vpin_proxy']:
                df[col] = df[col].astype(np.float32)
        
        # Simple X features statistics
        x_cols = [col for col in df.columns if col.startswith('X_')]
        if len(x_cols) > 0:
            # Process in chunks
            chunk_size = 100
            for i in range(0, len(x_cols), chunk_size):
                chunk_cols = x_cols[i:i+chunk_size]
                df[f'X_mean_{i//chunk_size}'] = df[chunk_cols].mean(axis=1).astype(np.float32)
                df[f'X_std_{i//chunk_size}'] = df[chunk_cols].std(axis=1).astype(np.float32)
        
        # Clean up
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Store feature columns
        self.feature_cols = [col for col in df.columns if col not in ['label', 'timestamp']]
        
        logger.info(f"Created {len(self.feature_cols)} features")
        return df

# =========================
# Feature Selection
# =========================
class FeatureSelector:
    """Select top features using LightGBM importance"""
    
    def __init__(self, n_features=150):
        self.n_features = n_features
        self.selected_features = None
        
    def fit(self, X, y):
        """Fit the feature selector"""
        logger.info(f"Selecting top {self.n_features} features...")
        
        # Sample data for speed
        n_samples = min(Config.SAMPLE_SIZE, len(X))
        idx = np.random.choice(len(X), n_samples, replace=False)
        
        # Train LightGBM
        lgb_data = lgb.Dataset(X.iloc[idx], label=y[idx])
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'verbose': -1,
            'force_col_wise': True
        }
        
        model = lgb.train(
            params,
            lgb_data,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Select top features
        self.selected_features = feature_importance.head(self.n_features)['feature'].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        # Clean up
        del model, lgb_data
        gc.collect()
        
        return self
    
    def transform(self, X):
        """Transform data to selected features"""
        return X[self.selected_features]
    
    def fit_transform(self, X, y):
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)

# =========================
# Neural Network Model
# =========================
class CryptoMLP(nn.Module):
    """Multi-layer perceptron for crypto prediction"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Skip connection
        self.skip = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Main path
        out = self.model(x)
        
        # Add skip connection
        skip = self.skip(x)
        out = out + 0.1 * skip
        
        return out

# =========================
# Model Trainer
# =========================
class ModelTrainer:
    """Train and evaluate neural network models"""
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.scalers = []
        self.feature_selector = None
        self.feature_engineer = None
        self.selected_features = None
        
    def train(self, train_df):
        """Train models with cross-validation"""
        logger.info("Starting model training...")
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        train_df = self.feature_engineer.create_features(train_df)
        
        # Prepare data
        X = train_df[self.feature_engineer.feature_cols]
        y = train_df['label'].values.astype(np.float32)
        
        # Clean up
        del train_df
        gc.collect()
        
        # Feature selection
        self.feature_selector = FeatureSelector(n_features=self.config.TOP_N_FEATURES)
        X = self.feature_selector.fit_transform(X, y)
        self.selected_features = X.columns.tolist()
        
        # Cross-validation
        kf = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"\nTraining fold {fold + 1}/{self.config.N_FOLDS}")
            
            # Split data
            X_train = X.iloc[train_idx].values
            X_val = X.iloc[val_idx].values
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_val = scaler.transform(X_val).astype(np.float32)
            
            # Train model
            model, score = self._train_fold(X_train, y_train, X_val, y_val, fold)
            
            # Store
            self.models.append(model)
            self.scalers.append(scaler)
            cv_scores.append(score)
            
            # Clean up
            del X_train, X_val, y_train, y_val
            gc.collect()
            
        logger.info(f"\nCV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
    def _train_fold(self, X_train, y_train, X_val, y_val, fold):
        """Train a single fold"""
        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE * 2,
            shuffle=False
        )
        
        # Create model
        model = CryptoMLP(
            input_dim=X_train.shape[1],
            hidden_dims=self.config.HIDDEN_DIMS,
            dropout_rate=self.config.DROPOUT_RATE
        ).to(self.config.DEVICE)
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        # Training
        best_score = -np.inf
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            # Train
            model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.config.DEVICE)
                batch_y = batch_y.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_preds = []
            val_targets = []
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.config.DEVICE)
                    batch_y = batch_y.to(self.config.DEVICE)
                    
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    
                    val_loss += loss.item()
                    val_preds.extend(output.cpu().numpy().flatten())
                    val_targets.extend(batch_y.cpu().numpy().flatten())
            
            # Metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_score = pearsonr(val_targets, val_preds)[0]
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if val_score > best_score:
                best_score = val_score
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                          f"Val Loss = {avg_val_loss:.4f}, Val Score = {val_score:.4f}")
            
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        logger.info(f"Fold {fold+1} best score: {best_score:.4f}")
        
        return model, best_score
    
    def predict(self, test_df):
        """Make predictions on test data"""
        logger.info("Making predictions...")
        
        # Apply feature engineering
        test_df = self.feature_engineer.create_features(test_df)
        
        # Select features
        X_test = test_df[self.selected_features].values.astype(np.float32)
        
        # Clean up
        del test_df
        gc.collect()
        
        # Get predictions from each fold
        all_predictions = []
        
        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            logger.info(f"Predicting with fold {i+1}/{len(self.models)}")
            
            # Scale
            X_scaled = scaler.transform(X_test).astype(np.float32)
            
            # Predict
            model.eval()
            predictions = []
            
            with torch.no_grad():
                dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
                loader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE * 4, shuffle=False)
                
                for batch in loader:
                    batch_x = batch[0].to(self.config.DEVICE)
                    output = model(batch_x)
                    predictions.extend(output.cpu().numpy().flatten())
            
            all_predictions.append(np.array(predictions))
            
            # Clean up
            del X_scaled
            gc.collect()
        
        # Average predictions
        final_predictions = np.mean(all_predictions, axis=0)
        
        return final_predictions

# =========================
# Main Pipeline
# =========================
def main():
    """Main execution pipeline"""
    logger.info("Starting DRW Crypto Prediction Pipeline")
    logger.info("="*60)
    
    # Load training data
    logger.info("Loading training data...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    train_df = optimize_memory(train_df)
    logger.info(f"Training data shape: {train_df.shape}")
    
    # Use recent data if too large
    if len(train_df) > 600000:
        logger.info("Using recent 600k rows...")
        train_df = train_df.tail(600000).reset_index(drop=True)
    
    # Initialize trainer
    trainer = ModelTrainer(Config)
    
    # Train models
    trainer.train(train_df)
    
    # Clean up
    del train_df
    gc.collect()
    
    # Load test data
    logger.info("\nLoading test data...")
    test_df = pd.read_parquet(Config.TEST_PATH)
    test_df = optimize_memory(test_df)
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Generate predictions
    predictions = trainer.predict(test_df)
    
    # Create submission
    submission = pd.DataFrame({
        'row_id': range(len(predictions)),
        'label': predictions
    })
    
    submission.to_csv(Config.OUTPUT_DIR / 'submission.csv', index=False)
    logger.info(f"\nSubmission saved to: {Config.OUTPUT_DIR / 'submission.csv'}")
    
    # Clean up
    del test_df, predictions
    gc.collect()
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)

if __name__ == "__main__":
    main()

