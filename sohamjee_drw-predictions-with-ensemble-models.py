import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Using tree-based models only.")

# Set random seeds for reproducibility
np.random.seed(42)
if PYTORCH_AVAILABLE:
    torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=== DRW Crypto Price Prediction Challenge ===")
print("Comprehensive ML Solution with Ensemble Methods")
print("="*50)


import gc
def clear_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_and_explore_data():
    """Load data and perform initial exploration"""
    print("\n1. Loading and Exploring Data...")
    
    # Load training data
    TRAINING_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TESTING_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    train_df = pd.read_parquet(TRAINING_PATH)
    test_df = pd.read_parquet(TESTING_PATH)

    start_date = "2023-11-01 00:00:00"
    end_date = "2024-02-29 23:59:00"

    # Filter the DataFrame and update train_df with the subset
    train_df = train_df.loc[start_date:end_date]
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Training period: {train_df.index.min()} to {train_df.index.max()}")

  # Basic statistics
    print("\nBasic Statistics:")
    print(f"Target variable (label) statistics:")
    print(train_df['label'].describe())

# Check for missing values
    print(f"\nMissing values in training data: {train_df.isnull().sum().sum()}")
    print(f"Missing values in test data: {test_df.isnull().sum().sum()}")

# Feature columns
    feature_cols = [col for col in train_df.columns if col.startswith('X')]
    public_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    
    print(f"\nNumber of proprietary features (X_*): {len(feature_cols)}")
    print(f"Number of public market features: {len(public_cols)}")
    
    return train_df, test_df, feature_cols, public_cols


  def create_market_microstructure_features(df):
    """Create advanced market microstructure features"""
    print("\n2. Creating Market Microstructure Features...")
    
    df = df.copy()
    
    # Order book imbalance features
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    
    # Trading intensity features
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['buy_pressure'] = df['buy_qty'] / (df['volume'] + 1e-8)
    df['sell_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['net_flow'] = df['buy_qty'] - df['sell_qty']
    
    # Volume-weighted features
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_15'] = df['volume'].rolling(window=15).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_15'] + 1e-8)
    
    # Momentum and volatility proxies
    for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
        df[f'{col}_pct_change'] = df[col].pct_change()
        df[f'{col}_rolling_std'] = df[col].rolling(window=10).std()
        df[f'{col}_z_score'] = (df[col] - df[col].rolling(window=20).mean()) / (df[col].rolling(window=20).std() + 1e-8)
    
    return df
    


def create_technical_indicators(df):
    """Create technical analysis indicators"""
    print("Creating Technical Indicators...")
    
    df = df.copy()
    
    # Price proxy using volume-weighted average
    df['price_proxy'] = (df['bid_qty'] + df['ask_qty']) / 2
    
    # Moving averages
    for window in [5, 10, 15, 30]:
        df[f'price_ma_{window}'] = df['price_proxy'].rolling(window=window).mean()
        df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
    
    # RSI-like indicator
    delta = df['price_proxy'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD-like indicator
    ema_12 = df['price_proxy'].ewm(span=12).mean()
    ema_26 = df['price_proxy'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['price_proxy'].rolling(window=20).mean()
    bb_std = df['price_proxy'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['price_proxy'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df
    


def create_time_features(df):
    """Create time-based features"""
    print("Creating Time-based Features...")
    
    df = df.copy()
    df['timestamp'] = df.index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['minute'] = df['timestamp'].dt.minute
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_lag_features(df, feature_cols, lags=[1, 2, 3, 5, 10]):
    """Create lagged features for key variables"""
    print("Creating Lag Features...")
    
    df = df.copy()
    
    # Key features to lag
    key_features = ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'net_flow', 'buy_sell_ratio']
    
    for feature in key_features:
        if feature in df.columns:
            for lag in lags:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    # Lag some proprietary features (sample a few to avoid overfitting)
    sample_x_features = feature_cols[:20]  # Use first 20 X features
    for feature in sample_x_features:
        for lag in [1, 2, 3]:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    return df


def create_interaction_features(df):
    """Create interaction features between key variables"""
    print("Creating Interaction Features...")
    
    df = df.copy()
    
    # Volume interactions
    df['volume_bid_interaction'] = df['volume'] * df['bid_qty']
    df['volume_ask_interaction'] = df['volume'] * df['ask_qty']
    df['volume_net_flow_interaction'] = df['volume'] * df['net_flow']
    
    # Ratio interactions
    df['bid_ask_volume_ratio'] = df['bid_ask_ratio'] * df['volume_ratio']
    df['buy_sell_volume_ratio'] = df['buy_sell_ratio'] * df['volume_ratio']
    
    return df


def perform_feature_selection(X, y, method='mutual_info', k=200):
    """Perform feature selection using various methods"""
    print(f"\n3. Performing Feature Selection (method: {method}, k: {k})...")
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    elif method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    else:
        raise ValueError("Method must be 'mutual_info' or 'f_regression'")
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    
    return X_selected, selected_features, selector


def preprocess_data(train_df, test_df , feature_cols, public_cols):
    """Comprehensive data preprocessing pipeline"""
    print("\n4. Training Data Preprocessing Pipeline...")
    
    # Feature engineering
    train_df = create_market_microstructure_features(train_df)
    train_df = create_technical_indicators(train_df)
    train_df = create_time_features(train_df)
    train_df = create_lag_features(train_df, feature_cols)
    train_df = create_interaction_features(train_df)
    
    test_df = create_market_microstructure_features(test_df)
    test_df = create_technical_indicators(test_df)
    test_df = create_time_features(test_df)
    test_df = create_lag_features(test_df, feature_cols)
    test_df = create_interaction_features(test_df)
    
    # Get all feature columns
    all_feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'label']]
    
    # Handle missing values
    train_df[all_feature_cols] = train_df[all_feature_cols].fillna(method='ffill').fillna(0)
    test_df[all_feature_cols] = test_df[all_feature_cols].fillna(method='ffill').fillna(0)
    
    # Remove infinite values
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Total features after engineering: {len(all_feature_cols)}")

    try:
        train_df.to_csv('/kaggle/working/processed_train_data.csv')
        test_df.to_csv('/kaggle/working/processed_test_data.csv')
        print("Processed data saved successfully to 'processed_train_data.csv'")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
        
    return train_df, test_df , all_feature_cols


class LSTMModel(nn.Module):
    """LSTM model for sequence prediction"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def create_sequences(data, seq_length=30):
    """Create sequences for LSTM training"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def train_lstm_model(X_train, y_train, X_val, y_val, seq_length=30, epochs=50):
    """Train LSTM model"""
    print("Training LSTM Model...")
    
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping LSTM training.")
        return None, None
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, seq_length)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(device)
    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
    y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = LSTMModel(input_size=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return model, (train_losses, val_losses)


class EnsembleModel:
    """Ensemble of multiple models with different strengths"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scalers = {}
        self.is_fitted = False
    
    def _initialize_models(self):
        """Initialize all models in the ensemble"""
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            # 'rf': RandomForestRegressor(
            #     n_estimators=300,
            #     max_depth=10,
            #     min_samples_split=10,
            #     min_samples_leaf=5,
            #     random_state=42,
            #     n_jobs=-1
            # ),
            'gbr': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            if model_name in ['ridge', 'elastic']:
                self.scalers[model_name] = StandardScaler()
            else:
                self.scalers[model_name] = RobustScaler()

    def fit(self, X, y, validation_split=0.2):
        """Train all models in the ensemble"""
        print("\n5. Training Ensemble Models...")
        
        self._initialize_models()
        
        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train each model
        val_scores = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Scale features
            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
            X_val_scaled = self.scalers[model_name].transform(X_val)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Validate
            val_pred = model.predict(X_val_scaled)
            val_score = pearsonr(y_val, val_pred)[0]
            val_scores[model_name] = val_score
            
            print(f"{model_name} validation correlation: {val_score:.4f}")
        
        # Calculate ensemble weights based on validation performance
        total_score = sum(max(0, score) for score in val_scores.values())
        if total_score > 0:
            self.weights = {name: max(0, score) / total_score for name, score in val_scores.items()}
        else:
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        print(f"\nEnsemble weights: {self.weights}")
        
        # Train LSTM if available
        # if PYTORCH_AVAILABLE:
        #     try:
        #         lstm_model, _ = train_lstm_model(X_train, y_train, X_val, y_val)
        #         if lstm_model is not None:
        #             self.models['lstm'] = lstm_model
        #             # Add LSTM to weights with moderate weight
        #             self.weights['lstm'] = 0.1
        #             # Renormalize weights
        #             total_weight = sum(self.weights.values())
        #             self.weights = {k: v / total_weight for k, v in self.weights.items()}
        #     except Exception as e:
        #         print(f"LSTM training failed: {e}")
        
        self.is_fitted = True
        return self

    def predict(self, X):
        """Make predictions using the ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            # if model_name == 'lstm':
            #     # Handle LSTM prediction separately
            #     if PYTORCH_AVAILABLE:
            #         try:
            #             X_scaled = self.scalers['xgboost'].transform(X)  # Use XGBoost scaler
            #             X_seq, _ = create_sequences(X_scaled, seq_length=30)
            #             if len(X_seq) > 0:
            #                 X_tensor = torch.FloatTensor(X_seq)
            #                 model.eval()
            #                 with torch.no_grad():
            #                     lstm_pred = model(X_tensor).numpy().flatten()
            #                 # Pad predictions to match input length
            #                 padded_pred = np.full(len(X), lstm_pred[-1])
            #                 padded_pred[-len(lstm_pred):] = lstm_pred
            #                 predictions[model_name] = padded_pred
            #             else:
            #                 predictions[model_name] = np.zeros(len(X))
            #         except Exception as e:
            #             print(f"LSTM prediction failed: {e}")
            #             predictions[model_name] = np.zeros(len(X))
            # else:
            X_scaled = self.scalers[model_name].transform(X)
            predictions[model_name] = model.predict(X_scaled)
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            ensemble_pred += self.weights[model_name] * pred
        
        return ensemble_pred 

    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
        
        return importance_dict


def evaluate_model(model, X, y, cv_folds=5):
    """Evaluate model using time series cross-validation"""
    print("\n6. Model Evaluation...")
    
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    correlations = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create and train model for this fold
        fold_model = EnsembleModel()
        fold_model.fit(X_train_fold, y_train_fold, validation_split=0.2)
        
        # Predict
        y_pred = fold_model.predict(X_val_fold)
        
        # Calculate correlation
        correlation = pearsonr(y_val_fold, y_pred)[0]
        correlations.append(correlation)
        
        print(f"Fold {fold + 1} correlation: {correlation:.4f}")
    
    mean_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)
    
    print(f"\nCross-validation results:")
    print(f"Mean correlation: {mean_correlation:.4f} ± {std_correlation:.4f}")
    
    return mean_correlation, std_correlation


def main():
    """Main execution pipeline"""
    print("Starting Crypto Price Prediction Pipeline...")
    
    # Load data
    train_df, test_df, feature_cols, public_cols = load_and_explore_data()
    
    # Preprocess data
    train_df, test_df, all_feature_cols = preprocess_data(train_df, test_df, feature_cols, public_cols)
    
    # Prepare features and target
    X = train_df[all_feature_cols].values
    y = train_df['label'].values
    X_test = test_df[all_feature_cols].values
    
    # Feature selection
    X_selected, selected_features, selector = perform_feature_selection(
        pd.DataFrame(X, columns=all_feature_cols), y, method='mutual_info', k=300
    )
    
    # Apply same selection to test data
    X_test_selected = selector.transform(pd.DataFrame(X_test, columns=all_feature_cols))
    
    print(f"\nUsing {len(selected_features)} selected features")
    
    # Focus on recent data as suggested in the tips
    recent_months = 2  # Use last 6 months
    cutoff_date = train_df.index.max() - pd.DateOffset(months=recent_months)
    recent_mask = train_df.index >= cutoff_date
    
    X_recent = X_selected[recent_mask]
    y_recent = y[recent_mask]
    
    print(f"Using recent {recent_months} months of data: {len(X_recent)} samples")
    
    # Train final ensemble model
    final_model = EnsembleModel()
    final_model.fit(X_recent, y_recent, validation_split=0.2)
    
    # Make predictions on test set
    test_predictions = final_model.predict(X_test_selected)
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_df['timestamp'] if 'timestamp' in test_df.columns else range(len(test_df)),
        'label': test_predictions
    })
    
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print(f"\nSubmission file created: submission.csv")
    print(f"Test predictions statistics:")
    print(f"Mean: {test_predictions.mean():.6f}")
    print(f"Std: {test_predictions.std():.6f}")
    print(f"Min: {test_predictions.min():.6f}")
    print(f"Max: {test_predictions.max():.6f}")
    
    # Feature importance analysis
    importance_dict = final_model.get_feature_importance()
    if importance_dict:
        print(f"\nTop 10 most important features:")
        for model_name, importance in importance_dict.items():
            if len(importance) > 0:
                top_indices = np.argsort(importance)[-10:][::-1]
                print(f"\n{model_name}:")
                for i, idx in enumerate(top_indices):
                    feature_name = selected_features[idx]
                    print(f"  {i+1}. {feature_name}: {importance[idx]:.4f}")
    
    return final_model, submission



if __name__ == '__main__':
    model, submission = main()
    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print("="*50)

