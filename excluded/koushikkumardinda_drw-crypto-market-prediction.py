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


import gc
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from scipy.stats import pearsonr
import cudf # For GPU-accelerated dataframes
import cupy as cp # For GPU-accelerated numpy operations
import lightgbm as lgb # A strong baseline/ensemble candidate, even without explicit time features


# Function to reduce memory usage of a DataFrame (for pandas if you choose to use it for some steps)
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.2f}%')
    return df

# Load data with Polars first for efficiency, then convert to cuDF
# Or directly with cuDF if the file is not too large for direct cuDF read_parquet
try:
    train_df_pl = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
    test_df_pl = pl.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
    
    # Convert Polars DataFrames to cuDF DataFrames for GPU operations
    train_df = cudf.DataFrame(train_df_pl.to_pandas())
    test_df = cudf.DataFrame(test_df_pl.to_pandas())

    del train_df_pl, test_df_pl
    gc.collect()

    print("Data loaded into cuDF successfully!")

except Exception as e:
    print(f"Error loading data with cuDF/Polars, trying pandas: {e}")
    # Fallback to pandas with memory reduction if cuDF direct read or conversion fails due to size
    train_df = pd.read_parquet("train.parquet")
    test_df = pd.read_parquet("test.parquet")
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    print("Data loaded into pandas with memory reduction.")
    # If using pandas, move to CuPy/PyTorch tensors when needed to leverage GPU

# Define features and target
features = [col for col in train_df.columns if 'X' in col]
target = 'label'

# Handle potential NaN values (e.g., fill with median or mean for numerical features)
# For cuDF, use .fillna()
for col in features:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(train_df[col].median())
    if col in test_df.columns:
        test_df[col] = test_df[col].fillna(test_df[col].median())
    
# Convert target to float32
if target in train_df.columns:
    train_df[target] = train_df[target].astype(np.float32)

print(f"Shape of train_df: {train_df.shape}")
print(f"Shape of test_df: {test_df.shape}")

# Drop 'timestamp' from train_df if present, as it's not useful for direct time-series in test
if 'timestamp' in train_df.columns:
    train_df = train_df.drop('timestamp', axis=1)
if 'timestamp' in test_df.columns: # test timestamp is masked anyway
    test_df = test_df.drop('timestamp', axis=1)

# Ensure all features are float32 for GPU efficiency
for col in features:
    train_df[col] = train_df[col].astype(np.float32)
    test_df[col] = test_df[col].astype(np.float32)


# Simple feature engineering examples (can be expanded)
# Note: cuDF handles many pandas-like operations
# If you have specific domain knowledge about these 'X' features, you can create more tailored ones.

# Example: Mean and Std Dev across related features (if X1-X10 represent a group)
# This is a general example and might need adjustment based on feature relationships
# For instance, if X1-X5 are bid-related, X6-X10 are ask-related.
# For simplicity, let's create a few interaction terms and basic statistics.

# Note: These operations can be memory intensive, be selective.
# Using CuPy for complex numpy-like operations on GPU arrays can be beneficial.

def feature_engineer(df, features_list):
    # Convert to CuPy array for faster operations if df is cuDF
    if isinstance(df, cudf.DataFrame):
        df_cp = df[features_list].to_cupy()
    else: # If using pandas
        df_cp = cp.asarray(df[features_list].values)

    # Example: Simple sum and mean of all X features
    df['sum_X'] = cp.sum(df_cp, axis=1)
    df['mean_X'] = cp.mean(df_cp, axis=1)
    df['std_X'] = cp.std(df_cp, axis=1)
    
    # You can add more complex interactions here if RAM permits
    # e.g., ratios, differences, polynomial features for selected important features
    # df['X1_div_X2'] = df['X1'] / (df['X2'] + 1e-6) # Avoid division by zero

    # Convert back to cuDF if originally cuDF, or to pandas if originally pandas
    if isinstance(df, cudf.DataFrame):
        df['sum_X'] = df['sum_X'].astype(np.float32)
        df['mean_X'] = df['mean_X'].astype(np.float32)
        df['std_X'] = df['std_X'].astype(np.float32)
    else:
        df['sum_X'] = df['sum_X'].get().astype(np.float32) # .get() to move from CuPy to NumPy
        df['mean_X'] = df['mean_X'].get().astype(np.float32)
        df['std_X'] = df['std_X'].get().astype(np.float32)

    return df

train_df = feature_engineer(train_df, features)
test_df = feature_engineer(test_df, features)

# Update features list to include new engineered features
features.extend(['sum_X', 'mean_X', 'std_X'])

print("Feature Engineering complete.")


# Use StandardScaler for normalization. QuantileTransformer can also be effective.
# For large datasets, it's common to fit the scaler on a subset or in batches
# if RAM is very limited, but for numerical features, standard scaling is usually fine.
# Note: scikit-learn's scalers are CPU-based. If using cuDF, convert to NumPy then scale, or use RAPIDS equivalents.

X_train = train_df[features].to_pandas().values if isinstance(train_df, cudf.DataFrame) else train_df[features].values
y_train = train_df[target].to_pandas().values if isinstance(train_df, cudf.DataFrame) else train_df[target].values
X_test = test_df[features].to_pandas().values if isinstance(test_df, cudf.DataFrame) else test_df[features].values

# Free up memory
del train_df, test_df
gc.collect()

print("Data converted to NumPy arrays.")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to CuPy arrays for GPU processing with PyTorch
X_train_scaled = cp.asarray(X_train_scaled, dtype=cp.float32)
X_test_scaled = cp.asarray(X_test_scaled, dtype=cp.float32)
y_train = cp.asarray(y_train, dtype=cp.float32)

print("Data scaled and converted to CuPy arrays.")


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset for efficient loading
class CryptoDataset(Dataset):
    def __init__(self, features, labels=None):
        # Convert CuPy arrays to PyTorch tensors and move to device
        self.features = torch.from_numpy(cp.asnumpy(features)).to(device)
        self.labels = torch.from_numpy(cp.asnumpy(labels)).to(device) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

# Define the Neural Network Architecture
class CryptoPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super(CryptoPredictor, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(0.2)) # Regularization

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(0.2)) # Regularization

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Model Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    best_val_corr = -1.0 # Pearson correlation as evaluation metric
    
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        # Calculate Pearson correlation
        val_corr, _ = pearsonr(val_true, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Pearson Corr: {val_corr:.4f}")

        # Save best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with correlation: {best_val_corr:.4f}")
        
        model.train()
    return model

# Hyperparameters
input_dim = X_train_scaled.shape[1]
hidden_dims = [512, 256, 128] # Example: Adjust based on experimentation and RAM
output_dim = 1
learning_rate = 0.001
batch_size = 2048 # Larger batch size leverages GPU more but consumes more RAM
num_epochs = 15 # Adjust based on convergence

# K-Fold Cross-Validation for robust evaluation and prediction
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions = cp.zeros(len(y_train), dtype=cp.float32)
test_predictions = cp.zeros((N_SPLITS, len(X_test_scaled)), dtype=cp.float32)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    
    X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    train_dataset = CryptoDataset(X_train_fold, y_train_fold)
    val_dataset = CryptoDataset(X_val_fold, y_val_fold)
    test_dataset = CryptoDataset(X_test_scaled) # No labels for test set

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for Kaggle GPU
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CryptoPredictor(input_dim, hidden_dims).to(device)
    # Optionally use float16 for reduced memory usage (requires compatible GPU and PyTorch version)
    # model.half() 
    
    criterion = nn.MSELoss() # Or HuberLoss for robustness to outliers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Load the best model weights for prediction
    trained_model.load_state_dict(torch.load("best_model.pth"))
    trained_model.eval() # Set to evaluation mode

    fold_test_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            # if model.network[0].weight.dtype == torch.float16:
            #    inputs = inputs.half() # Match input type if model is half()
            outputs = trained_model(inputs).squeeze()
            fold_test_preds.extend(outputs.cpu().numpy())
    
    test_predictions[fold] = cp.asarray(fold_test_preds, dtype=cp.float32)

    # OOF predictions
    fold_oof_preds = []
    with torch.no_grad():
        for inputs, _ in val_loader: # Use val_loader for OOF predictions
            inputs = inputs.to(device)
            # if model.network[0].weight.dtype == torch.float16:
            #    inputs = inputs.half()
            outputs = trained_model(inputs).squeeze()
            fold_oof_preds.extend(outputs.cpu().numpy())
    oof_predictions[val_idx] = cp.asarray(fold_oof_preds, dtype=cp.float32)

print("\n--- Training Complete ---")

# Aggregate OOF predictions
oof_corr, _ = pearsonr(cp.asnumpy(y_train), cp.asnumpy(oof_predictions))
print(f"Overall OOF Pearson Correlation: {oof_corr:.4f}")

# Average test predictions across folds for final submission
final_test_predictions = cp.mean(test_predictions, axis=0)

print("Final test predictions aggregated.")


# Load sample submission to get the correct format
sample_submission_df = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")

# Ensure test_predictions is a NumPy array for pandas DataFrame creation
final_test_predictions_np = cp.asnumpy(final_test_predictions)

# Create submission DataFrame
submission_df = pd.DataFrame({'ID': sample_submission_df['ID'], 'prediction': final_test_predictions_np})

# Ensure the label column has the correct data type (float32 if required)
submission_df['prediction'] = submission_df['prediction'].astype(np.float32)

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
print(submission_df.head())

