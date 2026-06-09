import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


calories_train =pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
calories_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


print(calories_train.shape, calories_test.shape)


print(calories_train.head(5))


print(calories_train.describe())


print(calories_train.info())


# Encode column Sex: 0: male and 1: female
print(calories_train['Sex'].value_counts())


calories_train['Sex'] = calories_train['Sex'].map({'male':0, 'female':1})


# Plot histogram for the target variable (Calorie burnt)
plt.hist(calories_train['Calories'], bins=30, edgecolor='k')
plt.title('Distribution of Calories Burnt')
plt.xlabel('Calories')
plt.ylabel('Count')
plt.show()


# Exploration correlation
plt.figure(figsize= (10, 10))
sns.heatmap(calories_train.corr(), annot=True)


calories_train = calories_train.drop(columns=['id'], axis=1)


# Check columns of dataset
calories_train.columns


# Separate features and targets
X = calories_train.drop(columns=['Calories'], axis=1).values
y = calories_train['Calories'].values

print(X.shape, y.shape)


from sklearn.preprocessing import StandardScaler
# Standardize inputs 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.reshape(-1,1), dtype=torch.float32)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size = 0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)


batch_size = 32
# Data Loader 
train_loader = DataLoader(train_data, batch_size)
test_loader = DataLoader(test_data, batch_size)


input_size = X.shape[1]
print(input_size)


# Custom MLP class for regression
class MLPModel(nn.Module):
    def __init__(self, input_size=input_size, hidden_size=16, output_size=1):
        super(MLPModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu = nn.ReLU()                          # Activation
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer
    
    def forward(self, x):
        x = self.fc1(x)   # Apply first linear layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)   # Apply output layer
        return x

# Create model instance
model = MLPModel()
print(model)


# Hyper parameters
model = MLPModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


num_epochs = 300 

# List to store loss values per epoch
train_losses = []

# Training loop
for epoch in range(num_epochs):
    # Forward Pass
    y_pred = model(X_train)

    # Compute Loss
    loss = criterion(y_pred, y_train)

    # Backward Pass
    loss.backward()

    # Optimizer
    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()

    # Append loss
    train_losses.append(loss.item())

    # Print
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


import matplotlib.pyplot as plt

# Plot the training loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(num_epochs), train_losses, linestyle='-', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Training Loss (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.show()


import torch

# Function to get RMSLE loss
def rmsle_loss(y_pred, y_true):
    # Prevent from negative
    y_pred = torch.clamp(y_pred, min=0)
    y_true = torch.clamp(y_true, min=0)
    
    # log(1 + y)
    log_pred = torch.log1p(y_pred)
    log_true = torch.log1p(y_true)
    
    return torch.sqrt(torch.mean((log_pred - log_true) ** 2))



# Evaluate model performance

with torch.no_grad():
    # Predict on train set
    train_preds = model(X_train)
    train_loss = criterion(train_preds, y_train)
    train_rmsle = rmsle_loss(train_preds, y_train)

    # Predict on test set
    test_preds = model(X_test)
    test_loss = criterion(test_preds, y_test)
    test_rmsle = rmsle_loss(test_preds, y_test)

print(f"Final Training Loss (MSE): {train_loss.item():.4f}")
print(f"Final Testing Loss (MSE): {test_loss.item():.4f}")
print(f"Training RMSLE: {train_rmsle.item():.4f}")
print(f"Test RMSLE: {test_rmsle.item():.4f}")


# Calculate RMSE
train_rmse = torch.sqrt(train_loss)
test_rmse = torch.sqrt(test_loss)

print(f"Training RMSE: {train_rmse.item():.2f}")
print(f"Testing RMSE: {test_rmse.item():.2f}")

