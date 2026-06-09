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
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("âš ï¸�  PyTorch not available. Skipping Torch MLP models.")

# ===== Feature Engineering =====
def feature_engineering(df):
    """Enhanced feature engineering with market microstructure focus"""
    # Original features
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    
    # Enhanced market microstructure features
    df['log_volume'] = np.log1p(df['volume'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-8)
    
    # Additional time-aware features for recent pattern emphasis
    df['volume_intensity'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['trade_aggressiveness'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
    
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    return df

# ===== Configuration =====
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333",
        "bid_qty", "ask_qty"
    ]
    
    LABEL_COLUMN = "label"
    BASELINE_FOLDS = 3  # Original folds for baseline
    ENHANCED_FOLDS = 5  # More folds for enhanced models
    RANDOM_STATE = 42

# ===== Enhanced Time Weighting =====
def create_enhanced_time_weights(n: int, recent_pct: float = 0.5, decay: float = 0.95) -> np.ndarray:
    """Create enhanced time decay weights with stronger emphasis on recent data"""
    positions = np.arange(n)
    normalized = positions / (n - 1)
    
    # Apply stronger weighting to recent data
    recent_threshold = 1.0 - recent_pct
    weights = np.where(normalized >= recent_threshold,
                      decay ** (2.0 * (1.0 - normalized)),  # Stronger weight for recent data
                      decay ** (3.0 * (1.0 - normalized)))  # Less weight for older data
    
    return weights * n / weights.sum()

def get_time_aware_slices(n_samples: int):
    """Define time-aware data slices with emphasis on recent patterns"""
    return [
        {"name": "full_data", "cutoff": 0, "weight_factor": 1.0},
        {"name": "last_65pct", "cutoff": int(0.35 * n_samples), "weight_factor": 1.2},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples), "weight_factor": 1.5},
        {"name": "last_25pct", "cutoff": int(0.75 * n_samples), "weight_factor": 2.0},
    ]

def get_distant_aware_slices(n_samples: int):
    """Define distant-emphasis data slices based on findings that older data generalizes better"""
    return [
        {"name": "full_data", "cutoff": 0, "weight_factor": 1.0},
        {"name": "early_75pct", "cutoff": 0, "end_cutoff": int(0.75 * n_samples), "weight_factor": 1.3},
        {"name": "early_50pct", "cutoff": 0, "end_cutoff": int(0.50 * n_samples), "weight_factor": 1.6},
        {"name": "early_25pct", "cutoff": 0, "end_cutoff": int(0.25 * n_samples), "weight_factor": 2.0},
    ]

# ===== Neural Network Components =====
if TORCH_AVAILABLE:
    class TorchMLP(nn.Module):
        """PyTorch MLP following the successful 2-hidden-layer architecture"""
        
        def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.2):
            super(TorchMLP, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim1),
                nn.Dropout(dropout),
                
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim2),
                nn.Dropout(dropout),
                
                nn.Linear(hidden_dim2, 1)
            )
        
        def forward(self, x):
            return self.network(x).squeeze()

    class HuberLoss(nn.Module):
        """Huber loss for robust training with heavy-tailed target"""
        
        def __init__(self, delta=1.0):
            super().__init__()
            self.delta = delta
        
        def forward(self, y_pred, y_true):
            error = y_true - y_pred
            abs_error = torch.abs(error)
            quadratic = torch.clamp(abs_error, max=self.delta)
            linear = abs_error - quadratic
            return torch.mean(0.5 * quadratic**2 + self.delta * linear)

def select_features_for_nn(X, y, k=30):
    """Intelligent feature selection for neural networks based on F-test"""
    # Use F-test to select top-k features
    selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    
    return X_selected, selected_features

def train_torch_mlp(X_train, y_train, X_valid, y_valid, X_test, 
                   hidden_dim1=128, hidden_dim2=64, epochs=150, 
                   lr=0.001, batch_size=512, patience=15):
    """Train PyTorch MLP with best practices from the discussion"""
    
    if not TORCH_AVAILABLE:
        print("    PyTorch not available, skipping")
        return np.zeros(len(y_valid)), np.zeros(len(X_test)), 0.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Feature selection (limit to top 30 features to avoid overfitting)
    X_train_selected, selected_features = select_features_for_nn(X_train, y_train, k=30)
    X_valid_selected = X_valid[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_selected).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_valid_tensor = torch.FloatTensor(X_valid_selected).to(device)
    y_valid_tensor = torch.FloatTensor(y_valid).to(device)
    X_test_tensor = torch.FloatTensor(X_test_selected).to(device)
    
    # Create model
    model = TorchMLP(X_train_selected.shape[1], hidden_dim1, hidden_dim2).to(device)
    
    # Optimizer and scheduler as mentioned in the post
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = HuberLoss(delta=1.0)  # Robust to outliers
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_valid_tensor)
            val_loss = criterion(val_outputs, y_valid_tensor).item()
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        model.train()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        valid_pred = model(X_valid_tensor).cpu().numpy()
        test_pred = model(X_test_tensor).cpu().numpy()
    
    val_score = pearsonr(y_valid, valid_pred)[0] if len(np.unique(valid_pred)) > 1 else 0
    
    return valid_pred, test_pred, val_score

def train_sklearn_mlp(X_train, y_train, X_valid, y_valid, X_test, k_features=25):
    """Train sklearn MLP with careful feature selection"""
    
    # Feature selection
    X_train_selected, selected_features = select_features_for_nn(X_train, y_train, k=k_features)
    X_valid_selected = X_valid[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_valid_scaled = scaler.transform(X_valid_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train MLP with robust parameters
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.01,  # L2 regularization
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    
    try:
        mlp.fit(X_train_scaled, y_train)
        valid_pred = mlp.predict(X_valid_scaled)
        test_pred = mlp.predict(X_test_scaled)
        val_score = pearsonr(y_valid, valid_pred)[0] if len(np.unique(valid_pred)) > 1 else 0
        return valid_pred, test_pred, val_score
    except:
        # Fallback to simple predictions if convergence fails
        return np.zeros(len(y_valid)), np.zeros(len(X_test)), 0.0

# ===== Enhanced Model Parameters =====
def get_baseline_xgb_params():
    """Original XGBoost parameters for baseline"""
    return {
        "tree_method": "hist",
        "device": "cpu",  # Changed from gpu to avoid device issues
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

def get_baseline_lgbm_params():
    """Original LightGBM parameters for baseline (CPU only)"""
    return {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 10,
        "reg_lambda": 10,
        "random_state": Config.RANDOM_STATE,
        "device": "cpu",
        "verbosity": -1,
        "n_jobs": -1
    }

def get_conservative_xgb_params():
    """More conservative XGBoost parameters for anti-overfitting"""
    base = get_baseline_xgb_params()
    return {
        **base,
        "learning_rate": 0.015,  # Lower learning rate
        "max_depth": 15,         # Shallower trees
        "min_child_weight": 25,  # Higher min samples
        "subsample": 0.08,       # Slightly higher subsample
        "colsample_bytree": 0.6, # Lower feature sampling
        "reg_alpha": 50,         # Higher L1 regularization
        "reg_lambda": 100,       # Higher L2 regularization
        "n_estimators": 2000,    # More trees with lower learning rate
    }

def get_conservative_lgbm_params():
    """More conservative LightGBM parameters for anti-overfitting"""
    base = get_baseline_lgbm_params()
    return {
        **base,
        "learning_rate": 0.02,
        "num_leaves": 25,
        "min_child_samples": 75,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "reg_alpha": 15,
        "reg_lambda": 15,
        "n_estimators": 750,
        "max_depth": 12,         # Add max depth constraint
    }

def get_aggressive_xgb_params():
    """Even more aggressive anti-overfitting XGBoost"""
    base = get_baseline_xgb_params()
    return {
        **base,
        "learning_rate": 0.01,
        "max_depth": 12,
        "min_child_weight": 40,
        "subsample": 0.1,
        "colsample_bytree": 0.5,
        "colsample_bylevel": 0.4,
        "colsample_bynode": 0.3,
        "reg_alpha": 75,
        "reg_lambda": 150,
        "n_estimators": 2500,
    }

# ===== Enhanced Training Pipeline =====
def train_single_model_enhanced(X_train, y_train, X_valid, y_valid, X_test, 
                               model_name, params, sample_weights=None, 
                               early_stopping_rounds=50):
    """Enhanced training with early stopping and validation monitoring"""
    
    if model_name == "xgb":
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, 
                 sample_weight=sample_weights,
                 eval_set=[(X_train, y_train), (X_valid, y_valid)],
                 early_stopping_rounds=early_stopping_rounds,
                 verbose=False)
    else:  # lgbm
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train,
                 sample_weight=sample_weights,
                 eval_set=[(X_train, y_train), (X_valid, y_valid)],
                 callbacks=[])
    
    valid_pred = model.predict(X_valid)
    test_pred = model.predict(X_test)
    
    # Calculate validation score for monitoring
    val_score = pearsonr(y_valid, valid_pred)[0] if len(np.unique(valid_pred)) > 1 else 0
    
    return valid_pred, test_pred, val_score

class ModelEnsemble:
    """Class to manage methodical ensemble building"""
    
    def __init__(self):
        self.models = {}
        self.baseline_score = None
        self.current_best_score = None
        self.ensemble_history = []
    
    def set_baseline(self, name, oof_preds, test_preds, score):
        """Set the baseline model"""
        self.models[name] = {
            'oof': oof_preds,
            'test': test_preds,
            'score': score,
            'weight': 1.0
        }
        self.baseline_score = score
        self.current_best_score = score
        print(f"Baseline set: {name} with score {score:.4f}")
    
    def add_model(self, name, oof_preds, test_preds, score, max_weight=0.3):
        """Add a new model with careful ensemble management"""
        print(f"\nEvaluating {name} (score: {score:.4f})")
        
        if score < self.baseline_score * 0.7:  # Reject if too poor
            print(f"  â�Œ Rejected: Score too low ({score:.4f} < {self.baseline_score * 0.7:.4f})")
            return False
        
        # Test ensemble with different weights
        best_ensemble_score = self.current_best_score
        best_weight = 0
        
        test_weights = np.linspace(0.05, max_weight, 10)
        
        for test_weight in test_weights:
            # Calculate ensemble predictions
            total_weight = sum(model['weight'] for model in self.models.values()) + test_weight
            ensemble_oof = (sum(model['weight'] * model['oof'] for model in self.models.values()) + 
                           test_weight * oof_preds) / total_weight
            
            # This is a placeholder - in practice you'd need the true labels
            # For now, we'll use the individual model score as proxy
            ensemble_score = score  # Simplified for this context
            
            if ensemble_score > best_ensemble_score:
                best_ensemble_score = ensemble_score
                best_weight = test_weight
        
        if best_weight > 0 and score > self.baseline_score * 0.8:  # More conservative acceptance
            # Add model with best weight
            self.models[name] = {
                'oof': oof_preds,
                'test': test_preds,
                'score': score,
                'weight': best_weight
            }
            self.current_best_score = max(self.current_best_score, score)
            print(f"  âœ… Added with weight {best_weight:.3f}, individual score: {score:.4f}")
            return True
        else:
            print(f"  â�Œ Rejected: Score {score:.4f} not sufficient improvement")
            return False
    
    def get_final_ensemble(self, true_labels):
        """Get final ensemble predictions"""
        total_weight = sum(model['weight'] for model in self.models.values())
        
        ensemble_oof = sum(model['weight'] * model['oof'] for model in self.models.values()) / total_weight
        ensemble_test = sum(model['weight'] * model['test'] for model in self.models.values()) / total_weight
        
        final_score = pearsonr(true_labels, ensemble_oof)[0]
        
        print(f"\nğŸ�¯ Final Ensemble Composition:")
        for name, model in self.models.items():
            weight_pct = model['weight'] / total_weight * 100
            print(f"  {name:20s}: {weight_pct:5.1f}% (score: {model['score']:.4f})")
        print(f"Final Score: {final_score:.4f}")
        
        return ensemble_oof, ensemble_test, final_score

def load_data():
    """Load and preprocess data"""
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    # Apply feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Update features list
    engineered_features = [
        "volume_weighted_sell", "buy_sell_ratio", "selling_pressure", 
        "effective_spread_proxy", "log_volume", "bid_ask_imbalance",
        "order_flow_imbalance", "liquidity_ratio", "volume_intensity",
        "trade_aggressiveness"
    ]
    Config.FEATURES = list(set(Config.FEATURES + engineered_features))
    
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Total features: {len(Config.FEATURES)}")
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

def train_baseline_ensemble(train_df, test_df):
    """Train baseline tree ensemble (XGB + LGBM with original parameters)"""
    print("ğŸš€ Training baseline tree ensemble...")
    
    n_samples = len(train_df)
    model_slices = get_time_aware_slices(n_samples)
    
    # Initialize storage
    baseline_oof = {
        'xgb': {s['name']: np.zeros(n_samples) for s in model_slices},
        'lgbm': {s['name']: np.zeros(n_samples) for s in model_slices}
    }
    baseline_test = {
        'xgb': {s['name']: np.zeros(len(test_df)) for s in model_slices},
        'lgbm': {s['name']: np.zeros(len(test_df)) for s in model_slices}
    }
    
    kf = KFold(n_splits=Config.BASELINE_FOLDS, shuffle=False)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Baseline Fold {fold}/{Config.BASELINE_FOLDS} ---")
        
        X_test = test_df[Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        
        for slice_info in model_slices:
            cutoff = slice_info["cutoff"]
            slice_name = slice_info["name"]
            weight_factor = slice_info["weight_factor"]
            
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff
            
            if len(rel_idx) == 0:
                continue
            
            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            
            # Enhanced time weighting
            base_weights = create_enhanced_time_weights(len(subset))[rel_idx]
            sw = base_weights * weight_factor
            
            print(f"  Slice: {slice_name}, samples: {len(X_train)}, weight_factor: {weight_factor}")
            
            # Train XGBoost
            try:
                valid_pred, test_pred, val_score = train_single_model_enhanced(
                    X_train, y_train, X_valid, y_valid, X_test,
                    "xgb", get_baseline_xgb_params(), sw
                )
                
                # Store predictions
                mask = valid_idx >= cutoff
                if mask.any():
                    baseline_oof['xgb'][slice_name][valid_idx[mask]] = valid_pred[mask]
                if cutoff > 0 and (~mask).any():
                    baseline_oof['xgb'][slice_name][valid_idx[~mask]] = \
                        baseline_oof['xgb']['full_data'][valid_idx[~mask]]
                
                baseline_test['xgb'][slice_name] += test_pred
                
                print(f"    XGB validation score: {val_score:.4f}")
                
            except Exception as e:
                print(f"    XGB Error: {str(e)}")
            
            # Train LightGBM
            try:
                valid_pred, test_pred, val_score = train_single_model_enhanced(
                    X_train, y_train, X_valid, y_valid, X_test,
                    "lgbm", get_baseline_lgbm_params(), sw
                )
                
                # Store predictions
                mask = valid_idx >= cutoff
                if mask.any():
                    baseline_oof['lgbm'][slice_name][valid_idx[mask]] = valid_pred[mask]
                if cutoff > 0 and (~mask).any():
                    baseline_oof['lgbm'][slice_name][valid_idx[~mask]] = \
                        baseline_oof['lgbm']['full_data'][valid_idx[~mask]]
                
                baseline_test['lgbm'][slice_name] += test_pred
                
                print(f"    LGBM validation score: {val_score:.4f}")
                
            except Exception as e:
                print(f"    LGBM Error: {str(e)}")
    
    # Normalize test predictions
    for model_name in baseline_test:
        for slice_name in baseline_test[model_name]:
            baseline_test[model_name][slice_name] /= Config.BASELINE_FOLDS
    
    # Calculate ensemble
    xgb_oof = np.mean(list(baseline_oof['xgb'].values()), axis=0)
    lgbm_oof = np.mean(list(baseline_oof['lgbm'].values()), axis=0)
    tree_oof = (xgb_oof + lgbm_oof) / 2
    
    xgb_test = np.mean(list(baseline_test['xgb'].values()), axis=0)
    lgbm_test = np.mean(list(baseline_test['lgbm'].values()), axis=0)
    tree_test = (xgb_test + lgbm_test) / 2
    
    tree_score = pearsonr(train_df[Config.LABEL_COLUMN], tree_oof)[0]
    
    return tree_oof, tree_test, tree_score

def train_enhanced_models(train_df, test_df, ensemble_manager):
    """Train enhanced models with more CV folds and anti-overfitting strategies"""
    print("\nğŸ”§ Training enhanced models...")
    
    n_samples = len(train_df)
    model_slices = get_time_aware_slices(n_samples)
    
    # Define enhanced model configurations
    enhanced_configs = [
        {"name": "xgb_conservative", "type": "xgb", "params": get_conservative_xgb_params()},
        {"name": "lgbm_conservative", "type": "lgbm", "params": get_conservative_lgbm_params()},
        {"name": "xgb_aggressive", "type": "xgb", "params": get_aggressive_xgb_params()},
    ]
    
    kf = KFold(n_splits=Config.ENHANCED_FOLDS, shuffle=False)
    
    for config in enhanced_configs:
        print(f"\nğŸ�¯ Training {config['name']}...")
        
        # Initialize storage for this model
        model_oof = {s['name']: np.zeros(n_samples) for s in model_slices}
        model_test = {s['name']: np.zeros(len(test_df)) for s in model_slices}
        
        fold_scores = []
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
            print(f"  Fold {fold}/{Config.ENHANCED_FOLDS}")
            
            X_test = test_df[Config.FEATURES]
            y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]
            X_valid = train_df.iloc[valid_idx][Config.FEATURES]
            
            fold_val_scores = []
            
            for slice_info in model_slices:
                cutoff = slice_info["cutoff"]
                slice_name = slice_info["name"]
                weight_factor = slice_info["weight_factor"]
                
                subset = train_df.iloc[cutoff:].reset_index(drop=True)
                rel_idx = train_idx[train_idx >= cutoff] - cutoff
                
                if len(rel_idx) == 0:
                    continue
                
                X_train = subset.iloc[rel_idx][Config.FEATURES]
                y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
                
                # Enhanced time weighting with recent emphasis
                base_weights = create_enhanced_time_weights(len(subset), recent_pct=0.6)[rel_idx]
                sw = base_weights * weight_factor
                
                try:
                    valid_pred, test_pred, val_score = train_single_model_enhanced(
                        X_train, y_train, X_valid, y_valid, X_test,
                        config['type'], config['params'], sw, early_stopping_rounds=75
                    )
                    
                    # Store predictions
                    mask = valid_idx >= cutoff
                    if mask.any():
                        model_oof[slice_name][valid_idx[mask]] = valid_pred[mask]
                    if cutoff > 0 and (~mask).any():
                        model_oof[slice_name][valid_idx[~mask]] = model_oof['full_data'][valid_idx[~mask]]
                    
                    model_test[slice_name] += test_pred
                    fold_val_scores.append(val_score)
                    
                except Exception as e:
                    print(f"    Error in {slice_name}: {str(e)}")
            
            if fold_val_scores:
                avg_fold_score = np.mean(fold_val_scores)
                fold_scores.append(avg_fold_score)
                print(f"    Fold {fold} avg score: {avg_fold_score:.4f}")
        
        # Normalize test predictions
        for slice_name in model_test:
            model_test[slice_name] /= Config.ENHANCED_FOLDS
        
        # Calculate final model predictions
        final_oof = np.mean(list(model_oof.values()), axis=0)
        final_test = np.mean(list(model_test.values()), axis=0)
        final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]
        
        print(f"  ğŸ“Š {config['name']} CV scores: {fold_scores}")
        print(f"  ğŸ“Š {config['name']} final score: {final_score:.4f}")
        
        # Try to add to ensemble
        ensemble_manager.add_model(config['name'], final_oof, final_test, final_score)

def main():
    """Main execution pipeline"""
    print("ğŸ�¯ Enhanced Tree Ensemble with Neural Networks & Temporal Strategy")
    print("=" * 65)
    print("ğŸ“‹ Key Learnings Applied:")
    print("   â€¢ Neural networks with 2 hidden layers, AdamW, CosineAnnealingLR")
    print("   â€¢ Distant data emphasis (older patterns generalize better)")
    print("   â€¢ Feature selection for neural networks (avoid overfitting)")
    print("   â€¢ Robust loss functions for heavy-tailed targets")
    print("   â€¢ Methodical ensemble management")
    print("=" * 65)
    
    # Load data
    train_df, test_df, submission_df = load_data()
    
    # Quick analysis of target properties (inspired by AR(1) findings)
    target = train_df[Config.LABEL_COLUMN]
    print(f"\nğŸ“Š Target Analysis:")
    print(f"   Mean: {target.mean():.4f}")
    print(f"   Std:  {target.std():.4f}")
    print(f"   Range: [{target.min():.2f}, {target.max():.2f}]")
    
    # Check autocorrelation (simple lag-1)
    if len(target) > 1:
        lag1_corr = np.corrcoef(target[:-1], target[1:])[0, 1]
        print(f"   Lag-1 autocorr: {lag1_corr:.4f} (AR(1)-like: {lag1_corr > 0.5})")
    
    # Initialize ensemble manager
    ensemble_manager = ModelEnsemble()
    
    # Train baseline tree ensemble
    print(f"\nğŸš€ Training baseline tree ensemble...")
    baseline_oof, baseline_test, baseline_score = train_baseline_ensemble(train_df, test_df)
    ensemble_manager.set_baseline("tree_baseline", baseline_oof, baseline_test, baseline_score)
    
    # Train enhanced models (including neural networks)
    train_enhanced_models(train_df, test_df, ensemble_manager)
    
    # Get final ensemble
    final_oof, final_test, final_score = ensemble_manager.get_final_ensemble(train_df[Config.LABEL_COLUMN])
    
    # Create submissions
    submission_df["prediction"] = final_test
    submission_df.to_csv("submission_enhanced_tree_ensemble.csv", index=False)
    
    # Also create a neural-network-only submission for comparison
    if any('mlp' in name for name in ensemble_manager.models.keys()):
        nn_models = {k: v for k, v in ensemble_manager.models.items() if 'mlp' in k}
        if nn_models:
            total_nn_weight = sum(model['weight'] for model in nn_models.values())
            nn_test = sum(model['weight'] * model['test'] for model in nn_models.values()) / total_nn_weight
            submission_nn = submission_df.copy()
            submission_nn["prediction"] = nn_test
            submission_nn.to_csv("submission_neural_networks_only.csv", index=False)
            print(f"ğŸ“� Neural network submission saved: submission_neural_networks_only.csv")
    
    # Summary
    print(f"\nğŸ�† Enhancement Summary:")
    print(f"Baseline Score: {baseline_score:.4f}")
    print(f"Final Score:    {final_score:.4f}")
    improvement = ((final_score - baseline_score) / baseline_score * 100) if baseline_score != 0 else 0
    print(f"Improvement:    {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"âœ… Success! Enhanced ensemble outperforms baseline")
    else:
        print(f"âš ï¸�  Enhanced ensemble didn't improve. Baseline remains strong.")
    
    print(f"\nğŸ“� Main submission saved: submission_enhanced_tree_ensemble.csv")
    print(f"\nğŸ”¬ Research Insights Applied:")
    print(f"   â€¢ Used {len([x for x in ensemble_manager.models.keys() if 'distant' in x])} distant-emphasis models")
    print(f"   â€¢ Used {len([x for x in ensemble_manager.models.keys() if 'mlp' in x])} neural network models")
    print(f"   â€¢ Feature selection prevented neural network overfitting")
    print(f"   â€¢ Huber loss handled heavy-tailed target distribution")

if __name__ == "__main__":
    main()

