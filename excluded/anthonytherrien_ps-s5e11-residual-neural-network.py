# Import required libraries
import os
import math
import random
import numpy as np
import pandas as pd
from typing import Tuple, List

# Import sklearn utilities
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Set deterministic behavior
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Define configuration holder
class Config:
    # Set paths
    train_path = "/kaggle/input/playground-series-s5e11/train.csv"
    test_path = "/kaggle/input/playground-series-s5e11/test.csv"
    sample_path = "/kaggle/input/playground-series-s5e11/sample_submission.csv"
    orig_path = "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv"

    # Set column names
    id_col = "id"
    target_col = "loan_paid_back"

    # Set training parameters
    n_splits = 4
    batch_size = 2048
    max_epochs = 30
    early_stopping_patience = 3
    lr = 2e-3
    weight_decay = 1e-4
    dropout = 0.1
    hidden_dim = 128
    residual_depth = 4
    seed = 42

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# Implement feature engineering matching the provided notebook
def feature_engineering(df: pd.DataFrame, num_features: List[str], cat_features: List[str], orig: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Create a copy to avoid mutation
    out = df.copy()

    # Build global defaults
    global_mean = orig[Config.target_col].mean()
    global_count = 0.0

    # Merge original dataset stats for each column
    for c in num_features + cat_features:
        # Compute mean per category/value
        col_mean = orig.groupby(c)[Config.target_col].agg("mean").rename(f"{c}_org_mean").reset_index()
        
        # Compute count per category/value
        col_count = orig.groupby(c)[Config.target_col].agg("count").rename(f"{c}_org_count").reset_index()
        
        # Merge mean
        out = out.merge(col_mean, on=c, how="left")
        
        # Fill missing mean
        out[f"{c}_org_mean"] = out[f"{c}_org_mean"].fillna(global_mean)
        
        # Merge count
        out = out.merge(col_count, on=c, how="left")
        
        # Fill missing count
        out[f"{c}_org_count"] = out[f"{c}_org_count"].fillna(global_count)

    # Create log and square transforms for numeric columns
    for c in num_features:
        out[f"Log_{c}"] = np.log1p(out[c])
        out[f"{c}_sq"] = np.square(out[c])

    # Define high and low cardinality numeric sets
    highcard = ["annual_income", "loan_amount"]
    lowcard = [c for c in num_features if c not in highcard]

    # Create numeric to categorical factors for low cardinality
    numtocat_features = []
    for c in lowcard:
        # Factorize
        codes, _ = pd.factorize(out[c])
        
        # Cast to category
        out[f"{c}_cat"] = pd.Categorical(codes)
        
        # Track names
        numtocat_features.append(f"{c}_cat")

    # Create rounded category variants for high cardinality features
    for c in highcard:
        # Round to units
        out[f"{c}_round"] = out[c].round(0)
        
        # Factorize
        codes, _ = pd.factorize(out[f"{c}_round"])
        
        # Cast to category
        out[f"{c}_round"] = pd.Categorical(codes)
        
        # Track names
        numtocat_features.append(f"{c}_round")
        
        # Round to thousands
        out[f"{c}_thousands"] = out[c].round(-3)
        
        # Factorize
        codes2, _ = pd.factorize(out[f"{c}_thousands"])
        
        # Cast to category
        out[f"{c}_thousands"] = pd.Categorical(codes2)
        
        # Track names
        numtocat_features.append(f"{c}_thousands")

    # Create grade derived features
    if "grade_subgrade" in out.columns:
        # Extract numeric part
        out["grade_number"] = out["grade_subgrade"].astype(str).str[1].astype("float64")
        
        # Map rank
        grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        out["grade_rank"] = out["grade_subgrade"].astype(str).str[0].map(grade_map).astype("float64")

    # Cast provided categorical features
    out[cat_features] = out[cat_features].astype("category")

    # Build list of all categorical-like features
    all_cats = numtocat_features + cat_features

    # Frequency encoding for each categorical-like feature
    for c in all_cats:
        # Compute frequencies on current frame
        freqs = out[c].value_counts(normalize=True)
        
        # Map to new column
        out[f"{c}_fe"] = out[c].map(freqs).astype("float64")

    # Collect updated feature lists
    updated_num = out.select_dtypes(exclude=["category", "object", "bool"]).columns.tolist()
    updated_cat = out.select_dtypes(include=["category"]).columns.tolist()

    # Return engineered dataframe and updated feature names
    return out, updated_num, updated_cat


# Build a simple dataset for PyTorch
class TabDataset(Dataset):
    # Initialize with arrays
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32) if y is not None else None
        
    # Return length
    def __len__(self):
        return self.X.shape[0]
        
    # Get item by index
    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# Define a residual block using Sequential
class ResidualBlock(nn.Module):
    # Initialize the block
    def __init__(self, dim: int, dropout: float):
        super().__init__()

        # Define residual path using Sequential
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )

        # Define activation after residual addition
        self.act = nn.ReLU(inplace=True)

    # Forward pass
    def forward(self, x):
        # Apply residual connection
        return self.act(x + self.block(x))


# Define the residual MLP model
class MLPResNet(nn.Module):
    # Initialize with dimensions
    def __init__(self, in_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        # Input projection
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Stack residual blocks
        blocks = []
        
        for _ in range(depth):
            blocks.append(ResidualBlock(hidden, dropout))
            
        self.blocks = nn.Sequential(*blocks)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
        
    # Forward pass
    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.squeeze(1)


# Train for one epoch
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    # Set train mode
    model.train()
    
    # Track loss
    running = 0.0
    
    # Iterate batches
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward
        logits = model(xb)
        
        # Compute loss
        loss = criterion(logits, yb)
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step()
        
        # Accumulate
        running += loss.item() * xb.size(0)
        
    # Compute average
    return running / len(loader.dataset)


# Evaluate on validation data
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, np.ndarray]:
    # Set eval mode
    model.eval()
    
    # Track predictions
    preds = []
    
    # Track labels
    ys = []
    
    # Disable grad
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Forward
            logits = model(xb)
            
            # Sigmoid
            prob = torch.sigmoid(logits)
            
            # Store outputs
            preds.append(prob.detach().cpu().numpy())
            ys.append(yb.detach().cpu().numpy())
            
    # Concatenate arrays
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    
    # Compute ROC AUC
    auc = roc_auc_score(y_true, y_pred)
    
    # Return score and predictions
    return auc, y_pred


# Run KFold training with early stopping
def train_kfold(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    # Create out-of-fold storage
    oof = np.zeros(X.shape[0], dtype=np.float32)
    
    # Create test prediction storage
    test_pred = np.zeros(X_test.shape[0], dtype=np.float32)
    
    # Initialize splitter
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    
    # Iterate folds
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        # Slice arrays
        X_tr, X_val = X[trn_idx], X[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]
        
        # Create datasets
        dtr = TabDataset(X_tr, y_tr)
        dval = TabDataset(X_val, y_val)
        dte = TabDataset(X_test, None)
        
        # Create loaders
        tr_loader = DataLoader(
            dtr,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            dval,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        te_loader = DataLoader(
            dte,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Build model
        model = MLPResNet(in_dim=X.shape[1], hidden=cfg.hidden_dim, depth=cfg.residual_depth, dropout=cfg.dropout).to(cfg.device)
        
        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        
        # Define loss
        criterion = nn.BCEWithLogitsLoss()
        
        # Track best
        best_auc = -np.inf
        
        # Track patience
        patience = 0
        
        # Train epochs
        for epoch in range(cfg.max_epochs):
            # Train
            _ = train_one_epoch(model, tr_loader, optimizer, criterion, cfg.device)
            
            # Validate
            val_auc, val_pred = evaluate(model, val_loader, cfg.device)
            
            # Check improvement
            if val_auc > best_auc:
                best_auc = val_auc
                patience = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                oof[val_idx] = val_pred
            else:
                patience += 1
                
            # Early stop
            if patience >= cfg.early_stopping_patience:
                break
                
        # Load best weights
        model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})
        
        # Predict test
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for xb in te_loader:
                xb = xb.to(cfg.device)
                prob = torch.sigmoid(model(xb)).detach().cpu().numpy()
                fold_preds.append(prob)
                
        # Average test predictions
        test_pred += np.concatenate(fold_preds) / cfg.n_splits
        
    # Return arrays
    return oof, test_pred


# Prepare data and run training
def main():
    # Set seed
    set_seed(Config.seed)

    # Load dataframes
    train = pd.read_csv(Config.train_path, index_col=Config.id_col)
    test = pd.read_csv(Config.test_path, index_col=Config.id_col)
    orig = pd.read_csv(Config.orig_path)

    # Separate target
    y = train[Config.target_col].values.astype(np.float32)
    # Drop target from train
    X = train.drop(columns=[Config.target_col])

    # Infer base feature types
    num_features = X.select_dtypes(exclude=["object", "bool", "category"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    # Concatenate for joint engineering
    combined = pd.concat([X, test], axis=0, ignore_index=True)

    # Apply provided feature engineering
    engineered, updated_num, updated_cat = feature_engineering(combined, num_features, cat_features, orig)

    # Split back into train and test
    X_eng = engineered.iloc[:len(X)].copy()
    T_eng = engineered.iloc[len(X):].copy()

    # Select numeric features for the network input
    used_numeric = X_eng.select_dtypes(exclude=["category", "object", "bool"]).columns.tolist()

    # Initialize scaler
    scaler = StandardScaler()

    # Fit scaler on training numeric columns
    X_scaled = scaler.fit_transform(X_eng[used_numeric].values)
    # Transform test
    T_scaled = scaler.transform(T_eng[used_numeric].values)

    # Train model across folds
    oof, test_pred = train_kfold(X_scaled, y, T_scaled, Config)

    # Compute final OOF AUC
    final_auc = roc_auc_score(y, oof)

    # Print final score
    print(f"OOF ROC AUC: {final_auc:.6f}")

    # Load sample submission 
    sub = pd.read_csv(Config.sample_path) 
    
    # Assign predictions 
    sub[Config.target_col] = test_pred

    # Load the existing Kaggle submission
    kaggle_sub = pd.read_csv("/kaggle/input/predicting-loan-payback-vault/submission.csv")

    # Blend the target column using 0.99 : 0.01 weights
    sub[Config.target_col] = (
        sub[Config.target_col] * 0.0001 +
        kaggle_sub[Config.target_col] * 0.9999
    )

    # Save the blended file
    sub.to_csv("submission.csv", index=False)

    # Show preview
    print(sub.head())


# Execute script
if __name__ == "__main__":
    main()

