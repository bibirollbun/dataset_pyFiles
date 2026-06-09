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


import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load and preprocess the data
file_path = '/kaggle/input/playground-series-s5e1/train.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert 'date' column to datetime and ordinal
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].apply(lambda x: x.toordinal())

# Fill missing values in 'num_sold'
data['num_sold'] = data['num_sold'].fillna(data['num_sold'].mean())

# Encode categorical columns
label_encoders = {}
for col in ['country', 'store', 'product']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Normalize features
feature_scaler = StandardScaler()
features = data[['date', 'country', 'store', 'product']]
features = feature_scaler.fit_transform(features)

# Normalize target
target_scaler = StandardScaler()
target = data[['num_sold']]
target = target_scaler.fit_transform(target)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Create data loaders
batch_size = 32  # Smaller batch size
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 2. Define the Transformer model
class TransformerForecast(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerForecast, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, d_model))  # Positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        for layer in self.transformer_encoder.layers:
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1) + self.pos_encoder  # Add positional encoding
        x = self.transformer_encoder(x)
        x = self.fc_out(x).squeeze(1)
        return x

# Instantiate model
input_dim = X_train_tensor.size(1)
d_model = 64
nhead = 4
num_encoder_layers = 3
dim_feedforward = 256
dropout = 0.1
model = TransformerForecast(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)

# 3. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Added weight decay

# 4. Training loop with gradient clipping
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'transformer_forecast_model.pth')



# Load and preprocess the test data
test_file_path = '/kaggle/input/playground-series-s5e1/test.csv'  # Replace with your test file path
test_data = pd.read_csv(test_file_path)

# Convert 'date' column to datetime and ordinal
test_data['date'] = pd.to_datetime(test_data['date'])
test_data['date'] = test_data['date'].apply(lambda x: x.toordinal())

# Encode categorical columns using the same label encoders as training
for col in ['country', 'store', 'product']:
    le = label_encoders[col]
    test_data[col] = le.transform(test_data[col])

# Normalize features using the same scaler as training
test_features = test_data[['date', 'country', 'store', 'product']]
test_features = feature_scaler.transform(test_features)

# Convert to PyTorch tensor
X_test_tensor = torch.tensor(test_features, dtype=torch.float32)

# Load the trained model
model = TransformerForecast(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
model.load_state_dict(torch.load('ridge_transformer_forecast_model.pth'))
model.to(device)
model.eval()

# Make predictions
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    predictions = model(X_test_tensor)

# Convert predictions back to original scale
predictions = target_scaler.inverse_transform(predictions.cpu().numpy())

# Create the output DataFrame with 'id' and 'predicted_num_sold'
output_data = test_data[['id']]
output_data['num_sold'] = predictions

# Save the predictions to a CSV file with only 'id' and 'predicted_num_sold' columns
output_data.to_csv('test_predictions1.csv', index=False)

