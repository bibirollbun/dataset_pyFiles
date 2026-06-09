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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW

# Load the dataset
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

# Remove rows with NaN values in the target column
train = train.dropna(subset=['num_sold'])

# Data preprocessing

def extract_date_features(df):
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day'] = pd.to_datetime(df['date']).dt.day
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

train = extract_date_features(train)
test = extract_date_features(test)

categorical_features = ['country', 'store', 'product']
numerical_features = ['year', 'month', 'day', 'day_of_week', 'is_weekend']
target = 'num_sold'

X = train[categorical_features + numerical_features]
y = np.log1p(train[target])  # Log transform to reduce skew

# Preprocessing: One-hot encode categorical features and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[ 
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ], remainder='passthrough')

X = preprocessor.fit_transform(X)

# Define a PyTorch Dataset
class SalesDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Datasets and Dataloaders
train_dataset = SalesDataset(X_train, y_train)
val_dataset = SalesDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the model
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Model hyperparameters
input_dim = X.shape[1]
model = SimpleRegressor(input_dim=input_dim)

# Loss and optimizer
criterion = nn.L1Loss()  # MAPE is more aligned with MAE, using L1 loss for stability
optimizer = AdamW(model.parameters(), lr=1e-3)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop

def train_model(model, train_loader, val_loader, epochs=50):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).squeeze()
                loss = criterion(preds, y_batch)
                val_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

# Train the model
train_model(model, train_loader, val_loader, epochs=50)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Predict on the test set
X_test = preprocessor.transform(test[categorical_features + numerical_features])
test_dataset = SalesDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).squeeze()
        predictions.append(preds.cpu().numpy())

# Create submission
test['num_sold'] = np.expm1(np.concatenate(predictions))  # Inverse transform
submission = test[['id', 'num_sold']]
submission.to_csv('submission.csv', index=False)


