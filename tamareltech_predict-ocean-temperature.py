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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("=== Ocean Temperature Profile Prediction ===\n")

# 1. DATA LOADING
print("1. Loading Data...")
data_path = '/kaggle/input/prediciton-of-ocean-temperature-profile-2025/'

# Load training data
train_data = np.load(data_path + 'train_data_target.npz')
print("Available keys in train_data_target.npz:", list(train_data.keys()))

# Extract training features and targets
# Using the correct key names from the npz file
data_train = train_data['data_train']
target_train = train_data['target_train']

print(f"\nOriginal training data shape: {data_train.shape}")
print(f"Original training target shape: {target_train.shape}")

# Additional debugging info
if len(data_train.shape) == 3:
    print("Note: Training data appears to be missing the batch dimension")
    print("Expected shape: (n_samples, 625, 37, 12)")
    print("Actual shape:", data_train.shape)
    # If batch dimension is missing, add it
    if data_train.shape == (625, 37, 12):
        data_train = data_train[np.newaxis, :]
        target_train = target_train[np.newaxis, :]
        print("Added batch dimension. New shapes:")
        print(f"Training data: {data_train.shape}")
        print(f"Target data: {target_train.shape}")

# Load test data
test_data = np.load(data_path + 'test_data.npz')
print("\nAvailable keys in test_data.npz:", list(test_data.keys()))

# Try to get the correct key for test data
# Common key names: 'data', 'data_test', 'test', 'x_test'
test_keys = list(test_data.keys())
if 'data' in test_keys:
    data_test = test_data['data']
elif 'data_test' in test_keys:
    data_test = test_data['data_test']
elif 'test' in test_keys:
    data_test = test_data['test']
elif 'x_test' in test_keys:
    data_test = test_data['x_test']
else:
    # Use the first available key
    print(f"Warning: Using first available key: {test_keys[0]}")
    data_test = test_data[test_keys[0]]

print(f"Test data shape: {data_test.shape}")

# Check if test data shape is correct
if data_test.shape[0] != 304:
    print(f"Warning: Expected 304 test samples, but got {data_test.shape[0]}")
    print("Test data might need reshaping or the structure might be different")

# Load sample submission
sample_submission = pd.read_csv(data_path + 'sample_submission.csv')
print(f"\nSample submission shape: {sample_submission.shape}")
print("Sample submission columns:", sample_submission.columns.tolist())
print("First few rows of sample submission:")
print(sample_submission.head())

# 2. EXPLORATORY DATA ANALYSIS
print("\n2. Exploratory Data Analysis...")

# Reshape data to expose spatial structure
n_samples = data_train.shape[0]
data_train_reshaped = data_train.reshape(n_samples, 25, 25, 37, 12)
target_train_reshaped = target_train.reshape(n_samples, 25, 25, 37, 1)

print(f"\nReshaped training data: {data_train_reshaped.shape}")
print(f"Reshaped target data: {target_train_reshaped.shape}")

# Basic statistics
print("\nData Statistics:")
print(f"Training data - Min: {data_train.min():.2f}, Max: {data_train.max():.2f}, Mean: {data_train.mean():.2f}, Std: {data_train.std():.2f}")
print(f"Target data - Min: {target_train.min():.2f}, Max: {target_train.max():.2f}, Mean: {target_train.mean():.2f}, Std: {target_train.std():.2f}")

# Check for NaN values
print(f"\nNaN values in training data: {np.isnan(data_train).sum()}")
print(f"NaN values in target data: {np.isnan(target_train).sum()}")

# Visualize sample data
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sample_idx = 0

# Plot temperature at different depths for first sample
for i, depth_idx in enumerate([0, 18, 36]):  # Surface, middle, bottom
    # Spatial view at specific depth and time
    ax = axes[0, i]
    im = ax.imshow(data_train_reshaped[sample_idx, :, :, depth_idx, 0], cmap='coolwarm')
    ax.set_title(f'Temperature at Depth Index {depth_idx} (t=0)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax)

# Plot temporal evolution at center point
center_x, center_y = 12, 12
for i, depth_idx in enumerate([0, 18, 36]):
    ax = axes[1, i]
    temps = data_train_reshaped[sample_idx, center_x, center_y, depth_idx, :]
    ax.plot(range(12), temps, 'b-o')
    ax.set_title(f'Temporal Evolution at Depth {depth_idx}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Temperature')
    ax.grid(True)

plt.tight_layout()
plt.savefig('eda_visualization.png')
plt.close()

# Analyze vertical profiles
print("\n3. Vertical Profile Analysis...")
mean_profile = data_train_reshaped.mean(axis=(0, 1, 2, 4))  # Average over samples, spatial, and time
plt.figure(figsize=(8, 10))
plt.plot(mean_profile, range(37), 'b-')
plt.gca().invert_yaxis()
plt.xlabel('Mean Temperature')
plt.ylabel('Depth Level')
plt.title('Average Vertical Temperature Profile')
plt.grid(True)
plt.savefig('vertical_profile.png')
plt.close()

# 3. DATA PREPROCESSING
print("\n4. Data Preprocessing...")

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Flatten for normalization
data_train_flat = data_train.reshape(n_samples, -1)
target_train_flat = target_train.reshape(n_samples, -1)

data_train_normalized = scaler_X.fit_transform(data_train_flat)
target_train_normalized = scaler_y.fit_transform(target_train_flat)

# Reshape back
data_train_normalized = data_train_normalized.reshape(n_samples, 625, 37, 12)
target_train_normalized = target_train_normalized.reshape(n_samples, 625, 37, 1)

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    data_train_normalized, target_train_normalized, 
    test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# 4. MODEL DEFINITION
print("\n5. Defining Model Architecture...")

class OceanTempPredictor(nn.Module):
    def __init__(self, input_channels=12, hidden_dim=128, num_layers=2):
        super(OceanTempPredictor, self).__init__()
        
        # Spatial feature extraction
        self.conv1 = nn.Conv2d(input_channels * 37, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Depth-wise processing
        self.depth_conv = nn.Conv1d(256 * 25 * 25, hidden_dim, kernel_size=3, padding=1)
        
        # Output projection
        self.output_conv = nn.Conv2d(256, 37, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, 625, 37, 12)
        batch_size = x.shape[0]
        
        # Reshape to (batch, 25, 25, 37, 12)
        x = x.reshape(batch_size, 25, 25, 37, 12)
        
        # Combine depth and time channels: (batch, 37*12, 25, 25)
        x = x.permute(0, 3, 4, 1, 2).reshape(batch_size, 37*12, 25, 25)
        
        # Spatial convolutions
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        
        # Output projection: (batch, 37, 25, 25)
        x = self.output_conv(x)
        
        # Reshape to match target: (batch, 625, 37, 1)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, 625, 37, 1)
        
        return x

# Alternative: Simple LSTM-based model
class SimpleLSTMPredictor(nn.Module):
    def __init__(self, input_dim=37*12, hidden_dim=256, output_dim=37):
        super(SimpleLSTMPredictor, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, 625, 37, 12)
        batch_size, n_points = x.shape[0], x.shape[1]
        
        # Reshape to (batch*625, 37*12)
        x = x.reshape(batch_size * n_points, -1).unsqueeze(1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        
        # Reshape to (batch, 625, 37, 1)
        output = output.reshape(batch_size, n_points, 37, 1)
        
        return output

# 5. TRAINING SETUP
print("\n6. Setting up Training...")

# Convert to PyTorch tensors
train_x_tensor = torch.from_numpy(X_train).float()
train_y_tensor = torch.from_numpy(y_train).float()
val_x_tensor = torch.from_numpy(X_val).float()
val_y_tensor = torch.from_numpy(y_val).float()

# Create datasets and dataloaders
batch_size = 16
train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
val_dataset = TensorDataset(val_x_tensor, val_y_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model (using simpler LSTM for faster training)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SimpleLSTMPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# 6. TRAINING LOOP
print("\n7. Training Model...")

num_epochs = 20  # Adjust based on your needs
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png')
plt.close()

# 7. MAKE PREDICTIONS
print("\n8. Making Predictions on Test Set...")

# Normalize test data
data_test_flat = data_test.reshape(304, -1)
data_test_normalized = scaler_X.transform(data_test_flat)
data_test_normalized = data_test_normalized.reshape(304, 625, 37, 12)

# Convert to tensor
test_x_tensor = torch.from_numpy(data_test_normalized).float()
test_dataset = TensorDataset(test_x_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Make predictions
model.eval()
predictions = []

with torch.no_grad():
    for batch_x in test_loader:
        batch_x = batch_x[0].to(device)
        outputs = model(batch_x)
        predictions.append(outputs.cpu().numpy())

# Concatenate predictions
predictions = np.concatenate(predictions, axis=0)
print(f"Predictions shape: {predictions.shape}")

# Denormalize predictions
predictions_flat = predictions.reshape(304, -1)
predictions_denorm = scaler_y.inverse_transform(predictions_flat)

# 8. CREATE SUBMISSION FILE
print("\n9. Creating Submission File...")

# Create submission dataframe
submission_df = pd.DataFrame(predictions_denorm)
submission_df.insert(0, 'ID', range(1, 305))

# Save submission
submission_df.to_csv('submission.csv', index=False)
print(f"Submission file created with shape: {submission_df.shape}")
print("First few rows of submission:")
print(submission_df.head())

# Final summary
print("\n=== Pipeline Complete ===")
print(f"Training samples: {n_samples}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Final validation loss: {val_losses[-1]:.4f}")
print("\nFiles created:")
print("- eda_visualization.png")
print("- vertical_profile.png")
print("- training_history.png")
print("- submission.csv")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'train_losses': train_losses,
    'val_losses': val_losses
}, 'ocean_temp_model.pth')
print("- ocean_temp_model.pth")

print("\nNext steps:")
print("1. Check the actual key names in the .npz files and update the data loading section")
print("2. Experiment with the CNN-based model for potentially better spatial feature extraction")
print("3. Try ensemble methods combining multiple models")
print("4. Implement data augmentation techniques")
print("5. Consider physics-informed constraints for ocean temperature modeling")

