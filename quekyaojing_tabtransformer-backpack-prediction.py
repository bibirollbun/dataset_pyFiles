!pip install tab-transformer-pytorch
!pip install hyper-connections


import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from colorama import Fore, Style
import random
import pickle
import time
import sys
import os
import gc
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import AUROC
from tab_transformer_pytorch import TabTransformer

import warnings
warnings.filterwarnings('ignore')
gc.enable()

print('PyTorch version',torch.__version__)


# Load dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

use_sample = False
if use_sample:
    train_df = train_df[:3000]
    test_df = test_df[:2000]

print(f'number_of_train_data: {len(train_df)}')
print(f'number_of_test_data: {len(test_df)}')


numerical_columns = ['Compartments', 'Weight Capacity (kg)']
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
target = 'Price'


print(train_df.info())
print(test_df.info())


def print_null_values():
    print(train_df.isnull().sum())
    print('\n')
    print(test_df.isnull().sum())

numerical_columns = ['Compartments', 'Weight Capacity (kg)']
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
target = 'Price'

for col in categorical_columns:
    train_df[col] = train_df[col].fillna('Unknown')
    train_df[col] = train_df[col].astype('category')

    test_df[col] = test_df[col].fillna('Unknown')
    test_df[col] = test_df[col].astype('category')

for col in numerical_columns:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(test_df[col].median())
    
print_null_values()


sns.histplot(data=train_df, x = 'Price')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.boxplot(data=train_df, x="Brand", y="Price", ax=axes[0])
sns.boxplot(data=train_df, x="Material", y="Price", ax=axes[1]);


fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.boxplot(data=train_df, x="Size", y="Price", ax=axes[0])
sns.boxplot(data=train_df, x="Laptop Compartment", y="Price", ax=axes[1]);


fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.boxplot(data=train_df, x="Waterproof", y="Price", ax=axes[0])
sns.boxplot(data=train_df, x="Style", y="Price", ax=axes[1]);


plt.figure(figsize=(10, 8))

sns.boxplot(data=train_df, x="Color", y="Price");


## Feature Selection
numerical_columns = ['Compartments', 'Weight Capacity (kg)']
categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
target = 'Price'

## Number of unique values in each categorical features.
categorical_n_unique = {cc: train_df[cc].nunique() \
                        for cc in categorical_columns}
categorical_n_unique


## Statistics of Numerical Features
train_df.describe().T.style.bar(subset=['mean'],)\
                        .background_gradient(subset=['std'], cmap='coolwarm')\
                        .background_gradient(subset=['50%'], cmap='coolwarm')


import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from torch.nn.functional import mse_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define categorical and continuous columns
cat_cols = categorical_columns
cont_cols = numerical_columns

print(cat_cols)
print(cont_cols)

# Label Encode categorical columns
label_encoders = {col: LabelEncoder().fit(train_df[col]) for col in cat_cols}
for col in cat_cols:
    train_df[col] = label_encoders[col].transform(train_df[col])

# Scale continuous features
scaler = StandardScaler().fit(train_df[cont_cols])
train_df[cont_cols] = scaler.transform(train_df[cont_cols])

# Save mean and std for TabTransformer (ensure correct dtype)
cont_mean_std = torch.from_numpy(np.array([scaler.mean_, scaler.scale_], dtype=np.float32)).to(device)

# Convert categorical & continuous features to tensors & move to GPU
x_categ = torch.tensor(train_df[cat_cols].values, dtype=torch.long).to(device)  # Categorical = long
x_cont = torch.tensor(train_df[cont_cols].values, dtype=torch.float).to(device)  # Continuous = float

# Handle target (ensure correct dtype & move to GPU)
y = torch.tensor(train_df['Price'].values, dtype=torch.float).to(device)  # Regression (float)
# y = torch.tensor(train_df['Price'].values, dtype=torch.long).to(device)  # Classification (long)

# Create dataset and DataLoader
dataset = TensorDataset(x_categ, x_cont, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Initialize model and move it to GPU
model = TabTransformer(
    categories=(6, 5, 4, 3, 3, 4, 7),  # Unique values per categorical column
    num_continuous=len(cont_cols),
    dim=32,
    dim_out=1,  # Use `num_classes` for classification
    depth=6,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    mlp_hidden_mults=(4, 2),
    mlp_act=nn.ReLU(),
    continuous_mean_std=cont_mean_std
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# Training loop
epochs = 30
for epoch in range(epochs):
    epoch_loss = 0.0
    total_batches = len(train_loader)
    for x_categ_batch, x_cont_batch, y_batch in train_loader:
        # Move data to GPU
        x_categ_batch = x_categ_batch.to(device)
        x_cont_batch = x_cont_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        preds = model(x_categ_batch, x_cont_batch).squeeze()

        # Compute loss
        mse_loss = loss_fn(preds, y_batch)

        # Backpropagation
        mse_loss.backward()
        optimizer.step()
        
        epoch_loss += mse_loss.item()
        avg_loss = epoch_loss / total_batches if total_batches > 0 else 0
    
    print(f"Epoch {epoch+1}, MSE Loss: {avg_loss:.4f}")


from sklearn.preprocessing import LabelEncoder

# Define categorical and continuous columns
cat_cols = categorical_columns
cont_cols = numerical_columns

# Label Encode categorical columns
label_encoders = {col: LabelEncoder().fit(test_df[col]) for col in cat_cols}
for col in cat_cols:
    test_df[col] = label_encoders[col].transform(test_df[col])

# Scale continuous features
scaler = StandardScaler().fit(test_df[cont_cols])
test_df[cont_cols] = scaler.transform(test_df[cont_cols])

# Save mean and std for TabTransformer (ensure correct dtype)
cont_mean_std = torch.from_numpy(np.array([scaler.mean_, scaler.scale_], dtype=np.float32)).to(device)

# Convert categorical & continuous features to tensors & move to GPU
x_categ_test = torch.tensor(test_df[cat_cols].values, dtype=torch.long).to(device)  # Categorical = long
x_cont_test = torch.tensor(test_df[cont_cols].values, dtype=torch.float).to(device)  # Continuous = float

# Create test dataset and DataLoader
test_dataset = TensorDataset(x_categ_test, x_cont_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Predict on test data
predictions = []
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient tracking
    for x_categ_batch, x_cont_batch in test_loader:
        x_categ_batch = x_categ_batch.to(device)
        x_cont_batch = x_cont_batch.to(device)

        preds = model(x_categ_batch, x_cont_batch).squeeze()
        predictions.append(preds.cpu())  # Move to CPU for easier handling

# Convert predictions to numpy array
predictions = torch.cat(predictions).numpy()

# Display predictions
print(predictions)


len(predictions)


# Load dataset
submission = pd.read_csv("../input/playground-series-s5e2/sample_submission.csv")
submission["Price"] = predictions

display(submission.head(10))
submission.to_csv("submission.csv", index=False)

