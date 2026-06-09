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


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math

# Define a PyTorch Dataset for Time Series Data
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets=None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.sequences[idx], self.targets[idx]
        return self.sequences[idx]

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Transformer Model for Time Series
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dim_feedforward, dropout):
        super(TransformerTimeSeries, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
        self.input_projection = nn.Linear(input_dim, d_model)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        output = self.fc_out(x)
        return output.squeeze(-1)

# Load data and preprocess
data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')

# Extract sequences and targets
sequence_length = 1440  # 3 years of daily data
future_steps = 32  # 1 month prediction

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['num_sold']].values)

sequences = []
labels = []
for i in range(len(scaled_data) - sequence_length - future_steps + 1):
    seq = scaled_data[i:i + sequence_length]
    label = scaled_data[i + sequence_length:i + sequence_length + future_steps]
    sequences.append(seq)
    labels.append(label)

sequences = np.array(sequences)
labels = np.array(labels)

# Train-validation split
train_size = int(0.8 * len(sequences))
X_train, X_val = sequences[:train_size], sequences[train_size:]
y_train, y_val = labels[:train_size], labels[train_size:]

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model initialization
input_dim = 1
model = TransformerTimeSeries(input_dim=input_dim, d_model=64, n_heads=4, num_layers=3, dim_feedforward=128, dropout=0.1)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

# Load the best model
model.load_state_dict(torch.load('best_transformer_model.pth'))
model.eval()

# Predict future sequences
def predict_future(model, input_sequence, steps):
    model.eval()
    input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            pred = model(input_sequence)
            predictions.append(pred.cpu().numpy())
            input_sequence = torch.cat((input_sequence[:, 1:, :], pred.unsqueeze(1)), dim=1)
    return np.array(predictions).flatten()

# Generate submission file
submission_data = pd.read_csv('test.csv')
initial_sequence = scaled_data[-sequence_length:]
predictions = predict_future(model, initial_sequence, steps=future_steps)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Save predictions to a CSV file
submission_data['num_sold'] = predictions
submission_data[['id', 'num_sold']].to_csv('submission.csv', index=False)

print("Submission file created successfully.")


