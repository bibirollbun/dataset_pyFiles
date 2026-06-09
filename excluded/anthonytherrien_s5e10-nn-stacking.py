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

# Import PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Define constants
TARGET = 'accident_risk'
BATCH_SIZE = 768
MAX_EPOCHS = 30
LEARNING_RATE = 5e-4
LR_DECAY = 0.925
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
def add_engineered_features(df):
    # Compute engineered feature
    df['engineered_feature'] = (
        0.3 * df["curvature"] +
        0.2 * (df["lighting"] == "night").astype(int) +
        0.1 * (df["weather"] != "clear").astype(int) +
        0.2 * (df["speed_limit"] >= 60).astype(int) +
        0.1 * (df["num_reported_accidents"] > 2).astype(int)
    )

    # Return dataframe with new feature
    return df


# Define MLP meta-model
class MetaMLP(nn.Module):
    # Initialize layers
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    # Forward pass
    def forward(self, x):
        return self.net(x)


# Define training routine for one seed using full data
def run_one_seed(X_train, y_train, X_test, device):
    # Convert full training dataset to tensors
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    )

    # Convert test dataset to tensor
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Build training data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    # Initialize model
    model = MetaMLP(input_dim=X_train.shape[1]).to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Initialize exponential learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

    # Initialize loss criterion
    criterion = nn.MSELoss()

    # Start training loop
    for epoch in range(MAX_EPOCHS):
        # Set model to training mode
        model.train()

        # Initialize cumulative training loss
        train_loss = 0.0

        # Iterate over batches in training data
        for xb, yb in train_loader:
            # Move batch to the selected device
            xb, yb = xb.to(device), yb.to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # Forward pass through model
            preds = model(xb)

            # Compute loss
            loss = criterion(preds, yb)

            # Backpropagate gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Accumulate total training loss
            train_loss += loss.item() * xb.size(0)

        # Step the scheduler to decay learning rate
        scheduler.step()

        # Compute average training loss
        train_loss /= len(train_ds)

        # Compute training RMSE
        train_rmse = np.sqrt(train_loss)

        # Print epoch-level performance
        print(f"Epoch {epoch + 1:03d} | LR: {scheduler.get_last_lr()[0]:.6f} | Train RMSE: {train_rmse:.5f}")

    # Set model to evaluation mode for predictions
    model.eval()

    # Disable gradient computation for prediction
    with torch.no_grad():
        # Generate training predictions
        train_preds = model(
            torch.tensor(X_train, dtype=torch.float32, device=device)
        ).cpu().view(-1).numpy()

        # Generate test predictions
        test_preds = model(
            test_tensor.to(device)
        ).cpu().view(-1).numpy()

    # Return training and test predictions
    return train_preds, test_preds


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

    # Merge in base train/test features to apply feature engineering
    oof_df = pd.merge(oof_df, train[['id', 'curvature', 'lighting', 'weather', 'speed_limit', 'num_reported_accidents']], on='id', how='left')
    test_df = pd.merge(test_df, test[['id', 'curvature', 'lighting', 'weather', 'speed_limit', 'num_reported_accidents']], on='id', how='left')

    # Apply feature engineering
    oof_df = add_engineered_features(oof_df)
    test_df = add_engineered_features(test_df)

    # Identify numerical feature columns
    num_features = oof_df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude ID and target columns from features
    FEATURES = [f for f in num_features if f not in ['id', TARGET]]

    # Prepare feature matrix and target vector
    X = oof_df[FEATURES]
    y = oof_df[TARGET]

    # Copy test feature matrix
    X_test_full = test_df[FEATURES].copy()

    # Initialize standard scaler
    scaler = StandardScaler()

    # Fit scaler on full training data and transform all splits
    X_train_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test_full)

    # Select computation device (CPU or GPU)
    device = get_device()

    # Initialize arrays for storing averaged predictions
    train_pred_accum = np.zeros(len(X))
    test_pred_accum = np.zeros(len(X_test_full))

    # Loop through each random seed for model averaging
    for seed in SEED_LIST:
        # Set reproducible random seed
        set_seed(seed)

        # Print current seed
        print(f"\n--- Training with seed {seed} ---")

        # Train and predict with one seed
        train_preds_seed, test_preds_seed = run_one_seed(
            X_train_scaled,
            y,
            X_test_scaled,
            device
        )

        # Accumulate averaged training predictions
        train_pred_accum += train_preds_seed / len(SEED_LIST)

        # Accumulate averaged test predictions
        test_pred_accum += test_preds_seed / len(SEED_LIST)

    # Compute RMSE on full training set
    train_rmse = mean_squared_error(y, train_pred_accum, squared=False)

    # Print overall training performance
    print(f"\nFull Training RMSE: {train_rmse:.5f}")

    # Create dataframe for current model predictions
    new_submission = pd.DataFrame({'id': test.id, TARGET: test_pred_accum})

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

