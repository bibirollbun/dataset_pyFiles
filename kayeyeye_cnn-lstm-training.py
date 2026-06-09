# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from os.path import join,exists
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import polars as pl
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


CONFIG = {
    "seq_length": 20,  # Length of input sequences for the model
    "batch_size": 512,  # Number of samples per gradient update
    "learning_rate": 0.0001,  # Learning rate for the optimizer
    "num_epochs": 12,  # Number of training epochs
    "num_workers": 4,  # Number of subprocesses to use for data loading
    "num_heads": 8, 
    "num_hash_buckets": 64,
    "pin_memory": True,  # Whether to pin memory for faster data transfer to GPU
    "prefetch_factor": 16,  # Number of batches to prefetch
    "dropout_rate": 0.2,  # Dropout rate for regularization
    "input_channels": 79,  # Number of input features
    "output_features": 1,  # Number of output features after CNN
    "lstm_hidden_size": 128,  # Hidden size for LSTM
    "embedding_dim": 16,  # Dimension of the embedding for symbol_id
    "weight_decay": 0.0001, 
    # "num_symbols": 39,  # Number of unique symbol_ids (0 to 38)
    "feature_columns": [f'feature_{i:02d}' for i in range(79)],  # List of feature column names
    "target_column": "responder_6",  # Name of the target variable
    "data_path": "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/",  # Path to the dataset
    "dataset_save_path": "/kaggle/working/processed_dataset.parquet",  # Path to save the processed dataset
    "checkpoint_dir": "/kaggle/working/models",  # Path to save the trained model
}


class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size):
        super(GatedResidualNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        gate = torch.sigmoid(self.linear2(x))  # Gating mechanism
        residual = self.linear1(x)  # Linear transformation
        return gate * residual + x  # Residual connection

class AttCNNBiLSTM(nn.Module):
    def __init__(self, input_channels, lstm_hidden_size, output_features, num_hash_buckets, embedding_dim, num_heads, dropout_rate=0.2):
        super(AttCNNBiLSTM, self).__init__()
        
        # Embedding layer for symbol_id using hashing trick
        self.num_hash_buckets = num_hash_buckets
        self.embedding = nn.Embedding(num_hash_buckets, embedding_dim)  
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_channels + embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        #self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        #self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        #self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Additional GRN after CNN layers
        #self.grn_cnn_to_lstm = GatedResidualNetwork(32)  # Input size matches the output of the last CNN layer
        #self.add_norm_cnn_to_lstm = AddNorm(32)  # Add & Norm after GRN        

        # LSTM layers
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(dropout_rate)

        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2, num_heads=num_heads, dropout=0.2, add_zero_attn=False)

        # Gated Residual Network after Attention
        self.grn1 = GatedResidualNetwork(lstm_hidden_size * 2)  # Input size matches the output of the attention layer
        self.add_norm_attn = AddNorm(lstm_hidden_size * 2)  # Add & Norm after GRN

        # Fully connected layer for prediction
        #self.fc1 = nn.Linear(lstm_hidden_size * 2, 64)  # First fully connected layer
        #self.fc2 = nn.Linear(64, 16)  # Second fully connected layer
        self.fc_output = nn.Linear(lstm_hidden_size * 2, output_features)  # Final output layer

    def forward(self, x, symbol_ids, return_features=False):
        # Hashing trick to map symbol_ids to indices
        hashed_indices = torch.fmod(symbol_ids, self.num_hash_buckets)  # Hashing trick
        
        # Embedding for hashed symbol_ids
        embedded_symbols = self.embedding(hashed_indices)  # Shape: (batch_size, seq_length, embedding_dim)
        embedded_symbols = embedded_symbols.permute(0, 2, 1)  # Shape: (batch_size, embedding_dim, seq_length)

        # Concatenate embeddings with features
        x = torch.cat((x, embedded_symbols), dim=1)  # Shape: (batch_size, input_channels + embedding_dim, seq_length)

        # Apply Convolutional Layers
        x = F.relu(self.bn1(self.conv1(x)))
        #x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        #x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        #x = self.pool3(x)
        x = self.dropout3(x)
        
        # Apply Gated Residual Network
        #x = self.grn_cnn_to_lstm(x.permute(0, 2, 1))  # Shape: (batch_size, seq_length, num_channels)
        #x = x.permute(0, 2, 1)  # Shape: (batch_size, num_channels, seq_length)
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_length, num_channels)
        
        # Apply Add & Norm
        #x = self.add_norm_cnn_to_lstm(x, x)  # Use the same x for residual connection

        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)

        # Multi-Head Attention
        attn_out, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)

        # Apply Gated Residual Network and Add & Norm
        grn_out = self.grn1(attn_out.mean(dim=1))  # Shape: (batch_size, 128)
        grn_out = self.add_norm_attn(grn_out, attn_out.mean(dim=1))  # Apply Add & Norm

        if return_features:
            return grn_out

        # Fully connected layers
        #fc_out = F.relu(self.fc1(grn_out))  # First fully connected layer
        #fc_out = F.relu(self.fc2(fc_out))  # Second fully connected layer

        # Fully connected layer for prediction
        output = self.fc_output(grn_out)  # Final output layer
        return output


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, feature_columns, target_column):
        #unique_symbol_ids = data['symbol_id'].unique()  # Get unique symbol_ids
        #sorted_symbol_ids = np.sort(unique_symbol_ids)
        self.data = torch.tensor(data.select(feature_columns).to_numpy(), dtype=torch.float32)
        self.targets = torch.tensor(data.select(target_column).to_numpy().flatten(), dtype=torch.float32)
        self.symbol_ids = torch.tensor(data.select('symbol_id').to_numpy().flatten(), dtype=torch.long)
        # self.time_ids = torch.tensor(data.select('time_id').to_numpy().flatten(), dtype=torch.long)
        # self.date_ids = torch.tensor(data.select('date_id').to_numpy().flatten(), dtype=torch.long)
        
        # Ensure that the sequence length does not exceed the available data length
        if seq_length > len(self.data):
            raise ValueError("Sequence length must be less than or equal to the length of the data.")

        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_length]  # Shape: (seq_length, num_features)
        target = self.targets[idx + self.seq_length - 1]  # Shape: (1,)
        symbol_ids = self.symbol_ids[idx:idx + self.seq_length]  # Get corresponding symbol_ids
        # Ensure that seq and symbol_ids are of the same length
        # if len(seq) != self.seq_length or len(symbol_ids) != self.seq_length:
        #     raise ValueError(f"Sequence length mismatch: seq length {len(seq)}, symbol_ids length {len(symbol_ids)}")
        return seq.permute(1, 0), target.unsqueeze(0), symbol_ids  # Return (num_features, seq_length), (1,), (seq_length,)

def weighted_r2_score(y_true, y_pred, weights):
    # Convert to numpy arrays if they are PyTorch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().detach().numpy()

    # Ensure all arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    weights = weights.flatten()

    # Calculate the weighted mean of y_true
    weighted_mean = np.sum(weights * y_true) / np.sum(weights) if np.sum(weights) != 0 else 0

    # Calculate the weighted residuals
    residuals = y_true - y_pred
    weighted_residual_sum = np.sum(weights * (residuals ** 2))
    weighted_total_sum = np.sum(weights * ((y_true - weighted_mean) ** 2))

    # Calculate R²
    r2 = 1 - (weighted_residual_sum / weighted_total_sum) if weighted_total_sum != 0 else 0
    return r2


data_folder = "/kaggle/input/updated-janestreet-data/"
feature_columns = CONFIG['feature_columns']
target_column = CONFIG['target_column']
seq_length = CONFIG['seq_length']

dataframe_0 = pl.read_parquet("/kaggle/input/updated-janestreet-data/dataframe_(0,).parquet")
data = TimeSeriesDataset(dataframe_0, seq_length, feature_columns, target_column)
# Load a sample from each file
for i in range(1,39):
    file_path = join(data_folder, f"dataframe_({i},).parquet")
    chunk = pl.read_parquet(file_path)
    temp_dataset = TimeSeriesDataset(chunk, seq_length, feature_columns, target_column)
    data.data = torch.cat((data.data, temp_dataset.data),dim=0)
    data.targets = torch.cat((data.targets, temp_dataset.targets),dim=0)
    data.symbol_ids = torch.cat((data.symbol_ids, temp_dataset.symbol_ids),dim=0)

# # Concatenate all samples into one DataFrame if needed
# train_df = pl.concat(samples)




# Create dataset and dataloader
# dataset = TimeSeriesDataset(data, seq_length, feature_columns, target_column)
train_loader = DataLoader(data, batch_size=CONFIG['batch_size'], pin_memory=True, num_workers=4, shuffle=False)


# Step 3: Train the AttCNNBiLSTM Model
input_channels = CONFIG['input_channels']
output_features = CONFIG['output_features']
lstm_hidden_size = CONFIG['lstm_hidden_size']
num_hash_buckets = CONFIG['num_hash_buckets']
embedding_dim = CONFIG['embedding_dim']
# num_symbols = CONFIG['num_symbols']
num_heads = CONFIG['num_heads']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if multiple GPUs are available

model = AttCNNBiLSTM(input_channels, lstm_hidden_size, output_features, num_hash_buckets, embedding_dim, num_heads).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

criterion = torch.nn.SmoothL1Loss()
#criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
#optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
#optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9, weight_decay=CONFIG['weight_decay'])
from tqdm import tqdm  # Import tqdm for progress bar
# Initialize variables for checkpointing
best_val_loss = float('inf')  # Set to infinity initially
checkpoint_dir = CONFIG['checkpoint_dir']  # Directory to save checkpoints
os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists
num_epochs = CONFIG['num_epochs']
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    total_batches = len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Total Batches: {total_batches}')
    
    # Training loop
    for batch_X, batch_y, symbol_ids in tqdm(train_loader, desc="Processing Batches", total=total_batches):
        batch_X, batch_y, symbol_ids = batch_X.to(device), batch_y.to(device), symbol_ids.to(device)  # Move data to GPU
        optimizer.zero_grad()
        
        outputs = model(batch_X, symbol_ids)  # Pass symbol_ids to the model
        loss = criterion(outputs, batch_y)  # Compute loss
        
        if torch.isnan(loss).any():
            print("Loss is NaN. Stopping training.")
            break
        
        loss.backward()  # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()  # Update weights
        
        epoch_loss += loss.item()
    
    train_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}')
    
    # # Validation loop
    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for batch_X, batch_y, symbol_ids in val_loader:
    #         batch_X, batch_y, symbol_ids = batch_X.to(device), batch_y.to(device), symbol_ids.to(device)
    #         outputs = model(batch_X, symbol_ids)
    #         loss = criterion(outputs, batch_y)
    #         val_loss += loss.item()
    
    # val_loss /= len(val_loader)
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
    
    # # Checkpointing
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch + 1}.pth")
    #     torch.save(model.state_dict(), checkpoint_path)
    #     print(f'Checkpoint saved at {checkpoint_path}')
    
    # Save model after every epoch (optional)
    epoch_checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), epoch_checkpoint_path)
    print(f'Epoch checkpoint saved at {epoch_checkpoint_path}')


# Function to evaluate the model
def evaluate_model(model, data_loader, device):
    model.eval()
    model.to(device)
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y, symbol_ids in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            symbol_ids = symbol_ids.to(device)
            outputs = model(batch_X, symbol_ids)  # Get model predictions
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    return all_predictions, all_targets

# Evaluate the model
predictions, targets = evaluate_model(model, train_loader, device)

