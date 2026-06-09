# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error
from cuml.preprocessing import TargetEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_ex = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


# Merging Train and Train_ex Data

train = pd.concat([train, train_ex], axis=0, ignore_index=True)


# Print the few rows of the train dataset

train.head()


# Print the few rows of the test dataset

test.head()


# Print the shape of the data

print('Shape of the train data:')
print(train.shape)

print('Shape of the test data:')
print(test.shape)


# print information about the columns in the train dataset

print('Information about the columns:')
print(train.info())


# print information about the columns in the test dataset

print('Information about the columns:')
print(test.info())


# Print the summary statistics for all variables of the train dataset

print('Summary statistics for all variables:')
train.describe()


# Print the summary statistics for all variables of the test dataset

print('Summary statistics for all variables:')
test.describe()


# Check percentage of missing values for train data

missing_train = train.isnull().mean() * 100
print(missing_train[missing_train > 0].sort_values(ascending=False))


# Check percentage of missing values for test data

missing_test = test.isnull().mean() * 100
print(missing_test[missing_test > 0].sort_values(ascending=False))


test_ids = test["id"].copy()


# Impute missing for numerical data with the median values

num_cols = test.select_dtypes(include=['number']).columns

imp_value = train[num_cols].median()

train[num_cols] = train[num_cols].fillna(imp_value)
test[num_cols] = test[num_cols].fillna(imp_value)


print("Missing Values for Train Dataset")

display(train.isnull().sum())


print("Missing Values for Test Dataset")

display(test.isnull().sum())


# Impute Missing Values for categorical data 

cat_cols = train.select_dtypes(include=['object']).columns

train[cat_cols] = train[cat_cols].fillna('None')
test[cat_cols] = test[cat_cols].fillna('None')


print("Missing Values and for Train Dataset")

display(train.isnull().sum())


print("Missing Values and Train Dataset")

display(test.isnull().sum())


# 1. Distribution of Price in Train Dataset
plt.figure(figsize=(8, 5))
sns.histplot(train['Price'], bins=30, kde=True, color='blue')
plt.title("Distribution of Price")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


# 2. Count of Different Brands
plt.figure(figsize=(10, 5))
sns.countplot(y=train['Brand'], order=train['Brand'].value_counts().index, palette="viridis")
plt.title("Count of Different Brands")
plt.xlabel("Count")
plt.ylabel("Brand")
plt.show()


# 3. Material Distribution
plt.figure(figsize=(10, 5))
sns.countplot(y=train['Material'], order=train['Material'].value_counts().index, palette="coolwarm")
plt.title("Distribution of Materials")
plt.xlabel("Count")
plt.ylabel("Material")
plt.show()


# 4. Price Distribution by Brand
plt.figure(figsize=(12, 6))
sns.boxplot(x='Brand', y='Price', data=train)
plt.xticks(rotation=45)
plt.title("Price Distribution by Brand")
plt.xlabel("Brand")
plt.ylabel("Price")
plt.show()


# 5. Size Distribution
plt.figure(figsize=(8, 5))
sns.countplot(y=train['Size'], order=train['Size'].value_counts().index, palette="pastel")
plt.title("Distribution of Bag Sizes")
plt.xlabel("Count")
plt.ylabel("Size")
plt.show()


# 6. Style Distribution
plt.figure(figsize=(10, 5))
sns.countplot(y=train['Style'], order=train['Style'].value_counts().index, palette="Set2")
plt.title("Distribution of Bag Styles")
plt.xlabel("Count")
plt.ylabel("Style")
plt.show()


# 7. Waterproof vs. Non-Waterproof Bags
plt.figure(figsize=(6, 4))
sns.countplot(x=train['Waterproof'], palette=["#3498db", "#e74c3c"])
plt.title("Count of Waterproof vs. Non-Waterproof Bags")
plt.xlabel("Waterproof")
plt.ylabel("Count")
plt.show()


# 8. Laptop Compartment Availability
plt.figure(figsize=(6, 4))
sns.countplot(x=train['Laptop Compartment'], palette=["#2ecc71", "#f1c40f"])
plt.title("Laptop Compartment Availability")
plt.xlabel("Laptop Compartment")
plt.ylabel("Count")
plt.show()


# 9. Color Distribution
plt.figure(figsize=(12, 5))
sns.countplot(y=train['Color'], order=train['Color'].value_counts().index, palette="coolwarm")
plt.title("Color Distribution of Bags")
plt.xlabel("Count")
plt.ylabel("Color")
plt.show()


target_encoder = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

features = test.columns.tolist()

for col in features:
    target_encoder.fit(train[col], train['Price'])
    train[col] = target_encoder.transform(train[col])
    test[col] = target_encoder.transform(test[col])


train.head()


test.head()


# Features (X)
X = train.drop(["Price"], axis=1)  # Drop non-feature columns
y = train["Price"]  # Use Price target


# Convert data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # Reshape for single output


# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)         # Hidden layer
        self.fc3 = nn.Linear(64, 1)           # Output layer
        self.relu = nn.ReLU()                 # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU after first layer
        x = self.relu(self.fc2(x))  # Apply ReLU after second layer
        x = self.fc3(x)             # Output layer (no activation)
        return x

# Initialize the model
input_size = X_train.shape[1]  # Number of features
model = RegressionModel(input_size)


criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer


import torch

num_epochs = 10
train_losses = []
val_losses = []
train_rmse = []
val_rmse = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    running_rmse = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()
        
        # Compute RMSE for training
        rmse = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
        running_rmse += rmse

    # Calculate average training loss and RMSE
    train_loss = running_loss / len(train_loader)
    train_rmse_epoch = running_rmse / len(train_loader)
    train_losses.append(train_loss)
    train_rmse.append(train_rmse_epoch)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_rmse_epoch = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            
            # Compute RMSE for validation
            rmse = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
            val_rmse_epoch += rmse

    # Calculate average validation loss and RMSE
    val_loss = val_loss / len(val_loader)
    val_rmse_epoch = val_rmse_epoch / len(val_loader)
    val_losses.append(val_loss)
    val_rmse.append(val_rmse_epoch)

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse_epoch:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse_epoch:.4f}")

print("Training complete!")



# Predict on validation set
model.eval()
with torch.no_grad():
    val_preds = model(X_val).numpy()

# Calculate RMSE
val_rmse = np.sqrt(mean_squared_error(y_val.numpy(), val_preds))
print(f"Validation RMSE: {val_rmse:.4f}")


# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(test.values, dtype=torch.float32)


# Ensure the model is in evaluation mode
model.eval()

# Make predictions
with torch.no_grad():  # Disable gradient calculation
    test_preds = model(X_test_tensor).numpy()  # Convert predictions to NumPy array


# Prepare submission DataFrame
submission = pd.DataFrame({
    "id": test_ids,  # Use the 'id' column from the test data
    "Price": test_preds.flatten()  # Flatten predictions to 1D array
})

# Ensure prices are rounded to three decimal places
submission["Price"] = submission["Price"].apply(lambda x: round(x, 3))

# Save to CSV
submission.to_csv("submission.csv", index=False, float_format="%.3f")


submission.head()

