# Silence warnings
import warnings
warnings.simplefilter('ignore')

# Import standard libraries
import os
import glob
import random

# Import third-party libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

# Import PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Define constants
TARGET = 'accident_risk'
BATCH_SIZE = 768
MAX_EPOCHS = 50
LEARNING_RATE = 5e-4
LR_DECAY = 0.925
PATIENCE = 7  # Early stopping patience
N_FOLDS = 5  # KFold CV splits
SEED_LIST = [9375, 1418, 2783, 8364, 5464, 6930, 3489, 4641]


# Define device selection
def get_device():
    # Choose CUDA if available, else CPU
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define reproducibility setup
def set_seed(seed):
    # Set Python seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch CPU seed
    torch.manual_seed(seed)

    # Set PyTorch CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define feature engineering
def create_frequency_features(train_df, test_df, cols, num, cat, target):
    """
    Add frequency and binning features to the dataset.
    - For each column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5 and 10 quantile bins (groups) to show rank or range.
    """
    train, test = train_df.copy(), test_df.copy()

    for col in cols:
        # Frequency encoding: how common each value is
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())

        # Binning: group numeric values into quantiles
        if col in num:
            for q in [5, 10, 15]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    new_num = train.drop(columns=cat+[target]).columns.tolist()
    return train, test, new_num


# Define MLP meta-model with regularization
class MetaMLP(nn.Module):
    # Initialize layers
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    # Forward pass
    def forward(self, x):
        return self.net(x)


# Define KFold CV training with early stopping
def train_with_kfold_cv(X_train, y_train, X_test, device, n_folds=N_FOLDS, seed=42):
    """
    Train meta-model with KFold cross-validation and early stopping.
    Returns OOF predictions, test predictions, and CV RMSE score.
    """
    # Initialize KFold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Initialize OOF and test prediction arrays
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    # Convert test to tensor once
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Track fold scores
    fold_scores = []
    
    # Loop through folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Split data
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create datasets
        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr.values, dtype=torch.float32).view(-1, 1)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        
        # Initialize model
        model = MetaMLP(input_dim=X_train.shape[1]).to(device)
        
        # Initialize optimizer with weight decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
        
        # Initialize loss criterion
        criterion = nn.MSELoss()
        
        # Early stopping tracking
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(MAX_EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            
            train_loss /= len(train_ds)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            
            val_loss /= len(val_ds)
            
            # Step scheduler
            scheduler.step()
            
            # Compute RMSE
            train_rmse = np.sqrt(train_loss)
            val_rmse = np.sqrt(val_loss)
            
            # Print progress every 5 epochs or on last epoch
            if (epoch + 1) % 5 == 0 or epoch == MAX_EPOCHS - 1:
                print(f"Epoch {epoch + 1:03d} | LR: {scheduler.get_last_lr()[0]:.6f} | Train RMSE: {train_rmse:.5f} | Val RMSE: {val_rmse:.5f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
        # Generate OOF predictions for validation set
        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
            oof_preds[val_idx] = model(val_tensor).cpu().view(-1).numpy()
            
            # Generate test predictions
            test_preds += model(test_tensor.to(device)).cpu().view(-1).numpy() / n_folds
        
        # Compute fold score
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_scores.append(fold_rmse)
        print(f"Fold {fold + 1} OOF RMSE: {fold_rmse:.5f}")
    
    # Compute overall CV score
    cv_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\n{'='*50}")
    print(f"CV RMSE: {cv_rmse:.5f} (Â±{np.std(fold_scores):.5f})")
    print(f"{'='*50}")
    
    return oof_preds, test_preds, cv_rmse


# Define dataframe merge by ID
def merge_dataframes_by_id(data_list, id_col='id', feature_col=TARGET):
    # Select the first dataframe in the list
    first = data_list[0]

    # Rename the feature column of the first dataframe using its model name
    merged = first['df'].rename(columns={feature_col: f"{feature_col}_{first['name']}"})

    # Iterate over the remaining dataframes in the list
    for data in data_list[1:]:
        # Rename the feature column in the current dataframe
        renamed = data['df'].rename(columns={feature_col: f"{feature_col}_{data['name']}"})

        # Merge the renamed dataframe with the accumulated merged dataframe
        merged = pd.merge(merged, renamed, on=id_col, how='outer')

    # Return the final merged dataframe
    return merged


# Define Ridge baseline with KFold CV
def train_ridge_baseline(X_train, y_train, X_test, n_folds=N_FOLDS, seed=42):
    """
    Train Ridge regression baseline with KFold CV for comparison.
    Returns OOF predictions, test predictions, and CV RMSE score.
    """
    # Initialize KFold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Initialize OOF and test prediction arrays
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    # Track fold scores
    fold_scores = []
    
    # Loop through folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        # Split data
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train Ridge model
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_tr)
        
        # Generate OOF predictions
        oof_preds[val_idx] = model.predict(X_val)
        
        # Generate test predictions
        test_preds += model.predict(X_test) / n_folds
        
        # Compute fold score
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        fold_scores.append(fold_rmse)
    
    # Compute overall CV score
    cv_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\nRidge Baseline CV RMSE: {cv_rmse:.5f} (Â±{np.std(fold_scores):.5f})")
    
    return oof_preds, test_preds, cv_rmse


# Define the main execution
def main():
    # Load training data
    train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')

    # Load test data
    test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

    # Discover all OOF (out-of-fold) prediction files
    oof_files = glob.glob('/kaggle/input/**/oof_*.csv', recursive=True)

    # Print the number of OOF files found
    print(f"Found {len(oof_files)} OOF files.")

    # Initialize containers for OOF and test data
    all_oof_data = []
    all_test_data = []

    # Iterate through all OOF file paths
    for oof_path in oof_files:
        # Construct corresponding test file path
        test_path = oof_path.replace('oof_', 'test_')

        # Extract model name from file path
        model_name = os.path.basename(oof_path).replace('oof_', '').replace('.csv', '')

        # Load OOF data and store with model name
        all_oof_data.append({'df': pd.read_csv(oof_path), 'name': model_name})

        # Load test data and store with model name
        all_test_data.append({'df': pd.read_csv(test_path), 'name': model_name})

    # Merge OOF dataframes by ID
    oof_df = merge_dataframes_by_id(all_oof_data)

    # Merge test dataframes by ID
    test_df = merge_dataframes_by_id(all_test_data)

    # Attach ground truth target to OOF dataframe
    oof_df[TARGET] = train[TARGET].values

    # ======================================================
    # --- Start: Your Custom Feature Engineering Logic ---
    # ======================================================
    
    # Merge in *all* original features to run your FE
    train_features = train.drop(columns=[TARGET], errors='ignore')
    oof_df = pd.merge(oof_df, train_features, on='id', how='left')
    test_df = pd.merge(test_df, test, on='id', how='left')

    # Identify feature columns from the *original* train data
    cols = train.drop(columns=[TARGET, 'id']).columns.tolist()
    cat = [col for col in cols if train[col].dtype in ["object","category"]]
    num = [col for col in cols if train[col].dtype not in ["object","category","bool"]]

    # Apply frequency and binning features
    # We pass test_df.copy() to avoid modifying it in place inside the function
    oof_df, test_df, _ = create_frequency_features(oof_df, test_df.copy(), cols, num, cat, TARGET)

    # Preparing categorical features
    # Find the cat cols that actually exist in the merged dataframes
    cat_cols_oof = [col for col in oof_df.columns if col in cat]
    cat_cols_test = [col for col in test_df.columns if col in cat]
    oof_df[cat_cols_oof] = oof_df[cat_cols_oof].astype("category")
    test_df[cat_cols_test] = test_df[cat_cols_test].astype("category")

    # Mapping a column
    map_col = "num_reported_accidents"
    map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
    if map_col in oof_df.columns:
        oof_df[map_col] = oof_df[map_col].map(map_num_reported)
        test_df[map_col] = test_df[map_col].map(map_num_reported)

    # Dropping unnecessary columns
    remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_freq"]
    
    # Find columns to drop that exist in oof_df
    cols_to_drop_oof = [col for col in remove if col in oof_df.columns]
    oof_df = oof_df.drop(columns=cols_to_drop_oof)
    
    # Find columns to drop that exist in test_df
    cols_to_drop_test = [col for col in remove if col in test_df.columns]
    test_df = test_df.drop(columns=cols_to_drop_test)

    # Note: We DO NOT drop 'id' or 'duplicates' from oof_df
    # as it's required for the stacking model to align.

    # ======================================================
    # --- End: Your Custom Feature Engineering Logic ---
    # ======================================================

    # Identify numerical feature columns for the NN
    # This will automatically pick up your new '_freq' and '_bin' features
    # and *exclude* the original categorical columns you converted.
    num_features = oof_df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude ID and target columns from features
    FEATURES = [f for f in num_features if f not in ['id', TARGET]]

    # Prepare feature matrix and target vector
    X = oof_df[FEATURES]
    y = oof_df[TARGET]

    # Copy test feature matrix
    # Ensure test_df has the same feature columns as oof_df
    X_test_full = test_df[FEATURES].copy()

    # Initialize standard scaler
    scaler = StandardScaler()

    # Fit scaler on full training data and transform all splits
    X_train_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test_full)

    # Select computation device (CPU or GPU)
    device = get_device()
    print(f"\nUsing device: {device}")
    print(f"Number of features: {len(FEATURES)}")
    print(f"Training samples: {len(X_train_scaled)}, Test samples: {len(X_test_scaled)}")

    # ======================================================
    # Train Ridge Baseline for comparison
    # ======================================================
    print("\n" + "="*50)
    print("Training Ridge Baseline")
    print("="*50)
    
    ridge_oof, ridge_test, ridge_cv_rmse = train_ridge_baseline(
        X_train_scaled,
        y,
        X_test_scaled,
        n_folds=N_FOLDS,
        seed=SEED_LIST[0]
    )

    # ======================================================
    # Train Neural Network with KFold CV and seed averaging
    # ======================================================
    print("\n" + "="*50)
    print("Training Neural Network Meta-Model")
    print("="*50)
    
    # Initialize arrays for averaging across seeds
    nn_oof_accum = np.zeros(len(X_train_scaled))
    nn_test_accum = np.zeros(len(X_test_scaled))
    cv_scores = []

    # Loop through each random seed for model averaging
    for idx, seed in enumerate(SEED_LIST):
        # Set reproducible random seed
        set_seed(seed)

        # Print current seed
        print(f"\n{'='*50}")
        print(f"Training with seed {seed} ({idx + 1}/{len(SEED_LIST)})")
        print(f"{'='*50}")

        # Train with KFold CV
        oof_preds, test_preds, cv_rmse = train_with_kfold_cv(
            X_train_scaled,
            y,
            X_test_scaled,
            device,
            n_folds=N_FOLDS,
            seed=seed
        )

        # Accumulate predictions
        nn_oof_accum += oof_preds / len(SEED_LIST)
        nn_test_accum += test_preds / len(SEED_LIST)
        cv_scores.append(cv_rmse)

    # Compute final averaged CV score
    final_nn_cv_rmse = np.sqrt(mean_squared_error(y, nn_oof_accum))
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Ridge Baseline CV RMSE: {ridge_cv_rmse:.5f}")
    print(f"Neural Network CV RMSE: {final_nn_cv_rmse:.5f}")
    print(f"NN Improvement over Ridge: {((ridge_cv_rmse - final_nn_cv_rmse) / ridge_cv_rmse * 100):.2f}%")
    print(f"NN Seed Variance: Â±{np.std(cv_scores):.5f}")
    print("="*50)

    # Choose best model predictions (use NN if better, else Ridge)
    if final_nn_cv_rmse < ridge_cv_rmse:
        print("\nâœ“ Using Neural Network predictions (better CV score)")
        final_test_preds = nn_test_accum
    else:
        print("\nâœ“ Using Ridge predictions (better CV score)")
        final_test_preds = ridge_test

    # Create dataframe for current model predictions
    new_submission = pd.DataFrame({'id': test.id, TARGET: final_test_preds})

    # ======================================================
    # Blend with existing saved submissions using weights
    # ======================================================

    # Define paths to other saved submissions
    blend_files = [
        '/kaggle/input/predicting-road-accident-risk-vault/autogluon15.csv',
        '/kaggle/input/predicting-road-accident-risk-vault/submission.csv'
    ]

    # Define blending weights (must sum to 1 with new model)
    blend_weights = [0.6, 0.375, 0.025]  # [modelA, modelB, new_model]

    # Initialize blended submission with zeros
    blended = pd.DataFrame({'id': test.id, TARGET: np.zeros(len(test))})

    # Loop through saved submissions and blend
    for path, weight in zip(blend_files, blend_weights[:-1]):
        # Load existing submission
        sub = pd.read_csv(path)

        # Add weighted predictions
        blended[TARGET] += sub[TARGET] * weight

    # Add current model predictions with its weight
    blended[TARGET] += new_submission[TARGET] * blend_weights[-1]

    # Save final blended submission
    blended.to_csv('submission.csv', index=False)

    # Print confirmation
    print("\nBlended submission saved as 'submission.csv'.")


# Invoke main execution
if __name__ == "__main__":
    main()

