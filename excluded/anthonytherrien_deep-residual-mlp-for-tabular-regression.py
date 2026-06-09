# Import section
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from itertools import combinations
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import torch

# Suppress warnings
warnings.filterwarnings("ignore")


# ==============================
# Config
# ==============================
class Config:
    # Define the target column name
    target = 'BeatsPerMinute'

    # Load training, test, and sample submission data
    train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

    # Select computation device (GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Global random seed for reproducibility
    state = 42

    # Number of folds for cross-validation
    n_splits = 10

    # Early stopping patience for training
    early_stop = 20

    # Define evaluation metric
    metric = 'rmse'

    # Define task type (regression/classification/multiclass)
    task_type = "regression"

    # Boolean flag if the task is regression
    task_is_regression = task_type == 'regression'

    # Flags for preprocessing and feature engineering
    outliers = False      # Whether to remove outliers
    log_trf = False       # Whether to log-transform target
    feature_eng = True    # Whether to create new features
    missing = False       # Whether to impute missing values
    training = True       # Whether to run training (True) or load saved preds (False)


# ==============================
# Transform (same as before, no EDA)
# ==============================
class Transform(Config):
    def __init__(self):
        super().__init__()
        
        # Identify numeric and categorical feature columns
        self.num_features = self.train.drop(self.target, axis=1).select_dtypes(exclude=['object', 'bool']).columns.tolist()
        self.cat_features = self.train.drop(self.target, axis=1).select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Handle missing values if enabled
        if self.missing:
            self.missing_values()
        
        # Add engineered features if enabled
        if self.feature_eng:
            self.train = self.new_features(self.train)
            self.test = self.new_features(self.test)
        
        # Remove outliers if enabled
        if self.outliers:
            self.remove_outliers()
        
        # Apply log transformation on target if enabled
        if self.log_trf:
            self.log_transformation()
        
        # Encode categorical features and scale numerics
        self.encode()

    def __call__(self):
        # Split target and features
        self.y = self.train[self.target]
        self.X = self.train.drop(self.target, axis=1)
        self.X_enc = self.train_enc.drop(self.target, axis=1)
        
        # Return processed datasets and metadata
        return self.X, self.X_enc, self.y, self.test, self.test_enc, self.cat_features, self.num_features, self.cat_features_card

    def encode(self):
        # Copy train and test to avoid mutation
        self.train_enc = self.train.copy()
        self.test_enc = self.test.copy()
        
        # Concatenate for consistent encoding
        data = pd.concat([self.train_enc, self.test_enc])
        
        # Encode categorical features as integers
        oe = OrdinalEncoder()
        data[self.cat_features] = oe.fit_transform(data[self.cat_features]).astype('int')
        
        # Standardize numeric features
        scaler = StandardScaler()
        data[self.num_features] = scaler.fit_transform(data[self.num_features])
        
        # Split back into train and test encodings
        self.train_enc = data[~data[self.target].isna()]
        self.test_enc = data[data[self.target].isna()].drop(self.target, axis=1)
        
        # Store cardinality of categorical features
        self.cat_features_card = [data[f].nunique() for f in self.cat_features]

    def new_features(self, data):
        # Generate pairwise interaction features
        for c1, c2 in list(combinations(self.num_features, 2)):
            data[f"{c1}_{c2}"] = data[c1] * data[c2]               # product
            data[f'{c1}_div_{c2}'] = data[c1] / (data[c2] + 1e-6)  # ratio
        
        # Generate quartile and decile bins for each numeric feature
        for c in self.num_features:
            data[f"{c}_quartile"] = pd.cut(data[c], bins=4, labels=False, include_lowest=True)
            data[f"{c}_decile"] = pd.cut(data[c], bins=10, labels=False, include_lowest=True)
        
        return data

    def log_transformation(self):
        # Log-transform the target column
        self.train[self.target] = np.log1p(self.train[self.target])


# ==============================
# RMSE
# ==============================
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ==============================
# Deep Residual MLP
# ==============================

# Define a single residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        
        # Main transformation path: Linear -> BatchNorm -> ReLU -> Dropout
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Shortcut connection:
        #   - Identity if input and output dimensions match
        #   - Linear projection if dimensions differ
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        # Pass input through the main path
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)

        # Compute residual (skip connection)
        res = self.shortcut(x)

        # Combine main path + residual, then apply activation
        return self.act(out + res)


# Define the full residual MLP regressor
class MLPRegressor(nn.Module):
    def __init__(self, in_features, hidden_dims, dropout):
        super().__init__()
        
        layers = []
        prev = in_features

        # Stack multiple residual blocks according to hidden_dims list
        for h in hidden_dims:
            layers.append(ResidualBlock(prev, h, dropout))
            prev = h

        # Final regression head (outputs a single value)
        layers.append(nn.Linear(prev, 1))

        # Register all blocks in a Sequential container
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the entire network
        return self.net(x).squeeze(1)  # squeeze to shape [batch]



# ==============================
# Training utils
# ==============================

# Function to fix random seeds for reproducibility
def set_seed(seed):
    # Set NumPy seed (controls randomness in NumPy operations)
    np.random.seed(seed)
    
    # Set PyTorch CPU seed
    torch.manual_seed(seed)
    
    # Set PyTorch GPU seed (all CUDA devices)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True  # Disable nondeterministic algorithms
    torch.backends.cudnn.benchmark = False     # Turn off autotuner for convolution algorithms


# ==============================
# Train one fold (clean + flat with logging)
# ==============================
def train_one_fold(X_tr, y_tr, X_va, y_va, params, epochs, batch_size, patience):
    # Convert arrays to TensorDatasets
    ds_tr = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    ds_va = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float())

    # Create DataLoaders with workers and persistence
    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )
    
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True
    )

    # Initialize model
    model = MLPRegressor(
        in_features=X_tr.shape[1],
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"]
    ).to(Config.device)

    # Optimizer, loss, scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=max(patience // 4, 10), factor=0.5
    )

    # Tracking best model
    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        # ---------------- Training ----------------
        model.train()
        train_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(Config.device), yb.to(Config.device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(ds_tr)

        # ---------------- Validation ----------------
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(Config.device), yb.to(Config.device)
                preds = model(xb)
                va_loss += criterion(preds, yb).item() * xb.size(0)
        va_loss /= len(ds_va)

        # Scheduler step
        scheduler.step(va_loss)

        # Print metrics every `patience` epochs
        if (epoch + 1) % patience == 0:
            print(f"Epoch {epoch+1:4d} | Train Loss: {train_loss:.6f} | Val Loss: {va_loss:.6f}")

        # ---------------- Early stopping ----------------
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}, best val loss = {best_loss:.6f}")
                break

    # Load best weights
    model.load_state_dict({k: v.to(Config.device) for k, v in best_state.items()})
    model.eval()

    # Final validation predictions
    with torch.no_grad():
        va_out = model(torch.from_numpy(X_va).float().to(Config.device)).cpu().numpy()

    rmse = root_mean_squared_error(y_va, va_out)
    return va_out, rmse, model


# ==============================
# Cross-validation training + prediction
# ==============================
def fit_predict_cv(X, y, X_test, seed=42):
    # Fix random seed for reproducibility
    set_seed(seed)

    # Convert pandas DataFrames/Series to NumPy arrays (float32 for PyTorch)
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    # Define K-Fold cross-validation
    kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=seed)

    # Initialize storage for out-of-fold predictions and test predictions
    oof = np.zeros(len(X_np), dtype=np.float32)         # OOF preds for training data
    test_pred = np.zeros(len(X_test_np), dtype=np.float32)  # Averaged preds for test data

    # Define model hyperparameters
    params = {
        'hidden_dims': [448, 224, 192, 160, 128, 96],  # architecture (deep residual MLP)
        'dropout': 0.4,                                    # dropout rate
        'lr': 1e-3,                                        # learning rate
        'weight_decay': 1e-3                               # L2 regularization
    }

    # ==============================
    # Cross-validation loop
    # ==============================
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_np, y_np), 1):
        print(f"Fold {fold}")

        # Train on training fold, validate on validation fold
        va_out, rmse, model = train_one_fold(
            X_np[tr_idx], y_np[tr_idx],
            X_np[va_idx], y_np[va_idx],
            params,
            epochs=2000,
            batch_size=768,
            patience=Config.early_stop
        )

        # Store OOF predictions for this fold
        oof[va_idx] = va_out

        # Accumulate test predictions (average across folds)
        with torch.no_grad():
            test_pred += model(
                torch.from_numpy(X_test_np).float().to(Config.device)
            ).cpu().numpy() / Config.n_splits

        print(f"Fold {fold} RMSE: {rmse:.6f}")

    # ==============================
    # Final OOF score
    # ==============================
    oof_rmse = root_mean_squared_error(y_np, oof)
    print(f"OOF RMSE: {oof_rmse:.6f}")

    # Return OOF preds, test preds, and OOF score
    return oof, test_pred, oof_rmse


# ==============================
# Main
# ==============================
def main():
    # ---------------------------------
    # Data preprocessing
    # ---------------------------------
    t = Transform()  
    # Returns: encoded train/test data, target, feature info
    X, X_enc, y, test, test_enc, cat_features, num_features, cat_cardinalities = t()

    # ---------------------------------
    # Cross-validation training + prediction
    # ---------------------------------
    oof_preds, test_preds, oof_rmse = fit_predict_cv(
        X_enc, 
        y, 
        test_enc, 
        seed=Config.state
    )

    # ---------------------------------
    # Create submission file
    # ---------------------------------
    submission = Config.submission.copy()
    submission[Config.target] = test_preds

    # Save submission to Kaggle output directory
    submission.to_csv("/kaggle/working/submission.csv", index=False)

    # Print preview of submission
    print(submission.head())


# ==============================
# Script entry point
# ==============================
if __name__ == "__main__":
    main()

