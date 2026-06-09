import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import zscore
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load and preprocess data
train_data = pd.read_csv('/kaggle/input/sehe-4678-ai-202/train.csv')
print(train_data)

test_data = pd.read_csv('/kaggle/input/sehe-4678-ai-202/test_without_labels.csv')
print(test_data)

# Drop 'NOX' and 'B' columns
train_data = train_data.drop(columns=['NOX', 'B'])  # Remove NOX and B columns
test_data = test_data.drop(columns=['NOX', 'B'])  # Ensure test data matches
print(train_data)
print(test_data)

# Split features and labels
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]


# Z-score for normal features
z_scores = np.abs(zscore(X))
X = X[(z_scores < 3).all(axis=1)] 
y = y[X.index] # Ensure the labels match the filtered features


# Ensure test data columns match training data
X_test_submission = test_data[X.columns]

# Log transform the target variable
y = np.log1p(y)


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_submission_scaled = scaler.transform(X_test_submission)

# Reshape target variable for PyTorch
y = y.values.reshape(-1, 1)

# Define the model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, dropout_rate, activation):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()

        # Select activation function
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
           raise ValueError("Unsupported activation function.")

        # Input layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        self.hidden_layers.append(nn.BatchNorm1d(hidden_size))
        self.hidden_layers.append(self.activation)
        self.hidden_layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))
            self.hidden_layers.append(self.activation)
            self.hidden_layers.append(nn.Dropout(dropout_rate))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

# K-Fold Cross Validation
kfold = KFold(n_splits=8, shuffle=True, random_state=42)
all_test_losses = []
all_predictions = []
fold = 1
best_model_paths = []
lowest_test_loss = float('inf')
start_time_total = time.time()

for train_idx, val_idx in kfold.split(X_scaled):
    print(f"Fold {fold} in progress...")

    # Split data for this fold
    X_train_fold, X_temp_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_temp_fold = y[train_idx], y[val_idx]

    # Further split validation set from the fold's training data
    X_train, X_val, y_train, y_val = train_test_split(X_train_fold, y_train_fold, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_temp_tensor = torch.tensor(X_temp_fold, dtype=torch.float32)
    y_temp_tensor = torch.tensor(y_temp_fold, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model for this fold
    input_size = X_train.shape[1]
    hidden_size = 128
    hidden_layers = 3
    dropout_rate = 0.2
    activation_function = 'elu'
    model = MLP(input_size, hidden_size, hidden_layers, dropout_rate, activation=activation_function)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Train the model
    num_epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    fold_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Save the best model for this fold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"best_model_fold_{fold}.pth"
            torch.save(model.state_dict(), best_model_path)

        # Print progress every 10 epochs
        epoch_time = time.time() - epoch_start_time
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f} sec")

    fold_training_time = time.time() - fold_start_time
    print(f"Fold {fold} Training Time: {fold_training_time:.2f} sec")

    # Load the best model for this fold and evaluate on the test set
    model.load_state_dict(torch.load(best_model_path,weights_only=True))
    model.eval()

    test_start_time = time.time()
    with torch.no_grad():
        test_loss = criterion(model(X_temp_tensor), y_temp_tensor).item()
    test_evaluation_time = time.time() - test_start_time
    print(f"Test Evaluation Time: {test_evaluation_time:.4f} sec")
    print(f"Fold {fold} Test Loss (MSE): {test_loss:.4f}")
    all_test_losses.append(test_loss)
    best_model_paths.append(best_model_path)

    # Generate predictions for submission
    with torch.no_grad():
        fold_predictions = model(torch.tensor(X_submission_scaled, dtype=torch.float32)).numpy()
    all_predictions.append(fold_predictions)

    fold += 1

# Print overall results
total_training_time = time.time() - start_time_total
print(f"Average Test Loss (MSE) across folds: {np.mean(all_test_losses):.4f}")
print(f"Total Training Time: {total_training_time:.2f} seconds")

# Generate final submission predictions
final_predictions = np.mean(all_predictions, axis=0)
final_predictions = np.expm1(final_predictions)

# Save predictions to CSV
submission = pd.read_csv('/kaggle/input/sehe-4678-ai-202/sample submission.csv')
submission["y_pred"] = final_predictions
submission.to_csv("sample_submission.csv", index=False)
print("Submission file saved")
print(submission["y_pred"])

# Plot training and validation loss curves of the best fold
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Training and Validation Loss Curve (Best Fold: {best_model_paths[np.argmin(all_test_losses)]}), Test Loss (MSE):{min(all_test_losses):.4f}")
plt.legend()
plt.show()



