from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

# Neural network imports
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612", "bid_qty",
        "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333","X817", 
        "X586",  "X292"
    ]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42

XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
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

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS}
]


# Neural Network Configuration
class MetaLearner(nn.Module):
    """Lightweight neural network for meta-learning on refined features"""
    def __init__(self, n_features):
        super(MetaLearner, self).__init__()
        # Input: refined features + XGBoost prediction
        input_dim = n_features + 1
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x).squeeze()

def train_meta_learner(X_train, y_train, xgb_pred_train, X_val, y_val, xgb_pred_val, sample_weights=None):
    """Train meta-learner neural network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare features: original features + XGBoost prediction
    X_train_meta = np.column_stack([X_train, xgb_pred_train])
    X_val_meta = np.column_stack([X_val, xgb_pred_val])
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_meta)
    X_val_scaled = scaler.transform(X_val_meta)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Sample weights
    if sample_weights is not None:
        weights_tensor = torch.FloatTensor(sample_weights).to(device)
    else:
        weights_tensor = None
    
    # Initialize model
    model = MetaLearner(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss(reduction='none')
    
    # Training
    model.train()
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(200):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(X_train_tensor)
        loss = criterion(pred, y_train_tensor)
        
        # Apply sample weights if provided
        if weights_tensor is not None:
            loss = (loss * weights_tensor).mean()
        else:
            loss = loss.mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor).mean().item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
            model.train()
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor).cpu().numpy()
        val_pred = model(X_val_tensor).cpu().numpy()
    
    return model, scaler, train_pred, val_pred

def add_features(df):
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']


    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'])
    df['selling_pressure'] = df['sell_qty'] / (df['volume'])
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'])
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'])
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'])
    

    return df

def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()
    
def load_data():
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    Config.FEATURES += ["log_volume", 'bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction', 'ask_buy_interaction',
                        'ask_sell_interaction']

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df


def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_90pct", "cutoff": int(0.10 * n_samples)},
        {"name": "last_85pct", "cutoff": int(0.15 * n_samples)},
        {"name": "last_80pct", "cutoff": int(0.20 * n_samples)},

    ]


# =========================
# Training and Evaluation
# =========================
def train_and_evaluate(train_df, test_df):
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)

    # Initialize prediction dictionaries for XGBoost
    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        for learner in LEARNERS
    }
    
    # Initialize prediction dictionaries for Neural Network
    nn_oof_preds = {s["name"]: np.zeros(n_samples) for s in model_slices}
    nn_test_preds = {s["name"]: np.zeros(len(test_df)) for s in model_slices}
    nn_models = {s["name"]: [] for s in model_slices}
    nn_scalers = {s["name"]: [] for s in model_slices}

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff

            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]

            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            # Train XGBoost (unchanged)
            for learner in LEARNERS:
                print(f"    Training XGBoost...")
                model = learner["Estimator"](**learner["params"])
                model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_valid, y_valid)], verbose=False)

                # XGBoost OOF predictions
                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                if cutoff > 0 and (~mask).any():
                    oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][
                        valid_idx[~mask]]

                # XGBoost test predictions
                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES])
                
                # Get XGBoost predictions for neural network training
                xgb_train_pred = model.predict(X_train)
                xgb_valid_pred = model.predict(X_valid)
                
                # Train Neural Network Meta-Learner
                print(f"    Training Neural Network Meta-Learner...")
                try:
                    nn_model, nn_scaler, nn_train_pred, nn_valid_pred = train_meta_learner(
                        X_train.values, y_train.values, xgb_train_pred,
                        X_valid.values, y_valid.values, xgb_valid_pred,
                        sample_weights=sw
                    )
                    
                    # Store neural network components
                    nn_models[slice_name].append(nn_model)
                    nn_scalers[slice_name].append(nn_scaler)
                    
                    # Neural Network OOF predictions
                    # nn_valid_pred corresponds to ALL validation samples (same order as X_valid)
                    mask = valid_idx >= cutoff
                    if mask.any():
                        idxs = valid_idx[mask]
                        # Use corresponding predictions for the masked validation indices
                        nn_oof_preds[slice_name][idxs] = nn_valid_pred[mask]
                    if cutoff > 0 and (~mask).any():
                        nn_oof_preds[slice_name][valid_idx[~mask]] = nn_oof_preds["full_data"][valid_idx[~mask]]
                    
                    # Neural Network test predictions
                    # Prepare test features with XGBoost predictions
                    xgb_test_pred = model.predict(test_df[Config.FEATURES])
                    test_meta_features = np.column_stack([test_df[Config.FEATURES].values, xgb_test_pred])
                    test_scaled = nn_scaler.transform(test_meta_features)
                    
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    nn_model.eval()
                    with torch.no_grad():
                        test_tensor = torch.FloatTensor(test_scaled).to(device)
                        nn_test_pred = nn_model(test_tensor).cpu().numpy()
                    
                    nn_test_preds[slice_name] += nn_test_pred
                    
                    print(f"      Neural Network trained successfully")
                    
                except Exception as e:
                    print(f"      Neural Network training failed: {str(e)}")
                    # Fallback: use XGBoost predictions
                    mask = valid_idx >= cutoff
                    if mask.any():
                        idxs = valid_idx[mask]
                        nn_oof_preds[slice_name][idxs] = oof_preds[learner["name"]][slice_name][idxs]
                    if cutoff > 0 and (~mask).any():
                        nn_oof_preds[slice_name][valid_idx[~mask]] = nn_oof_preds["full_data"][valid_idx[~mask]]
                    
                    nn_test_preds[slice_name] += test_preds[learner["name"]][slice_name]

    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= (Config.N_FOLDS-1)
    
    for slice_name in nn_test_preds:
        nn_test_preds[slice_name] /= (Config.N_FOLDS-1)

    return oof_preds, test_preds, model_slices, nn_oof_preds, nn_test_preds


# =========================
# Ensemble & Submission
# =========================
def safe_ensemble_strategy(train_df, xgb_oof, xgb_test, nn_oof, nn_test):
    """
    Safe ensemble strategy with fallback mechanism
    Only uses neural network if it improves performance
    """
    y_true = train_df[Config.LABEL_COLUMN]
    
    # Calculate individual scores
    xgb_score = pearsonr(y_true, xgb_oof)[0]
    nn_score = pearsonr(y_true, nn_oof)[0]
    
    print(f"\nModel Performance Comparison:")
    print(f"XGBoost OOF Score: {xgb_score:.4f}")
    print(f"Neural Network OOF Score: {nn_score:.4f}")
    
    # Safety threshold - only ensemble if NN is competitive
    improvement_threshold = -0.005  # Allow 0.5% degradation
    
    if nn_score < xgb_score + improvement_threshold:
        print(f"Neural Network performance is significantly worse. Using XGBoost only.")
        return "xgb_only", xgb_score, xgb_test
    
    # Try different ensemble weights
    best_score = xgb_score
    best_weight = 0.0
    best_test = xgb_test
    
    print(f"\nTesting ensemble weights:")
    for weight in np.arange(0.1, 0.8, 0.1):
        ensemble_oof = (1 - weight) * xgb_oof + weight * nn_oof
        ensemble_score = pearsonr(y_true, ensemble_oof)[0]
        print(f"  Weight {weight:.1f}: {ensemble_score:.4f}")
        
        if ensemble_score > best_score:
            best_score = ensemble_score
            best_weight = weight
            best_test = (1 - weight) * xgb_test + weight * nn_test
    
    if best_weight == 0.0:
        print(f"No ensemble weight improves performance. Using XGBoost only.")
        return "xgb_only", xgb_score, xgb_test
    else:
        improvement = ((best_score - xgb_score) / xgb_score) * 100
        print(f"Best ensemble weight: {best_weight:.1f}")
        print(f"Ensemble improvement: {improvement:+.2f}%")
        return "ensemble", best_score, best_test

def ensemble_and_submit(train_df, oof_preds, test_preds, submission_df, nn_oof_preds=None, nn_test_preds=None):
    learner_name = 'xgb'
    weights = np.array([1,1,1,1])

    # XGBoost ensemble (original logic)
    xgb_oof_weighted = pd.DataFrame(oof_preds[learner_name]).values @ weights
    xgb_test_weighted = pd.DataFrame(test_preds[learner_name]).values @ weights
    xgb_score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], xgb_oof_weighted)[0]
    print(f"{learner_name.upper()} Weighted Ensemble Pearson: {xgb_score_weighted:.4f}")

    # If neural network predictions are available, try ensemble
    if nn_oof_preds is not None and nn_test_preds is not None:
        print(f"\n" + "="*50)
        print("NEURAL NETWORK ENSEMBLE EVALUATION")
        print("="*50)
        
        # Neural Network ensemble
        nn_oof_weighted = pd.DataFrame(nn_oof_preds).values @ weights
        nn_test_weighted = pd.DataFrame(nn_test_preds).values @ weights
        nn_score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], nn_oof_weighted)[0]
        print(f"Neural Network Weighted Ensemble Pearson: {nn_score_weighted:.4f}")
        
        # Safe ensemble strategy
        strategy, final_score, final_test = safe_ensemble_strategy(
            train_df, xgb_oof_weighted, xgb_test_weighted, 
            nn_oof_weighted, nn_test_weighted
        )
        
        print(f"\n" + "="*50)
        print("FINAL MODEL SELECTION")
        print("="*50)
        print(f"Selected Strategy: {strategy}")
        print(f"Final Score: {final_score:.4f}")
        
        submission_df["prediction"] = final_test
        filename = f"submission_with_nn_{strategy}.csv"
        
    else:
        print("Neural Network predictions not available. Using XGBoost only.")
        submission_df["prediction"] = xgb_test_weighted
        filename = "submission_xgb_only.csv"

    submission_df.to_csv(filename, index=False)
    print(f"Saved: {filename}")
    print(submission_df.head(10))

if __name__ == "__main__":
    print("="*60)
    print("XGBoost + Neural Network Meta-Learning Pipeline")
    print("="*60)
    print("Strategy: Conservative ensemble with fallback mechanism")
    
    train_df, test_df, submission_df = load_data()
    oof_preds, test_preds, model_slices, nn_oof_preds, nn_test_preds = train_and_evaluate(train_df, test_df)
    ensemble_and_submit(train_df, oof_preds, test_preds, submission_df, nn_oof_preds, nn_test_preds)










