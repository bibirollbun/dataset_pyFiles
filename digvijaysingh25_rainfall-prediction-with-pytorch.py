!pip install torchviz

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from torchsummary import summary
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Loading Data
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


# Display few rows of each dataset
print(f"Train Data Preview:")
display(train.head(), train.shape)

print("\nTest Data Preview:")
display(test.head(), test.shape)


# Total missing values in train and test
missing_values = pd.DataFrame({
    'train': train.isnull().sum(),
    'test': test.isnull().sum(), 
    'data_types': train.dtypes
})
print(missing_values)


# Replace missing values in the 'winddirection' column of the test dataset with its mean
test.winddirection = test.winddirection.fillna(test.winddirection.mean())


# some new features
train['humidity_cloud_interaction'] = train['humidity'] * train['cloud']
train['humidity_sunshine_interaction'] = train['humidity'] * train['sunshine']
train['cloud_sunshine_ratio'] = train['cloud'] / (train['sunshine'] + 1e-5)
train['relative_dryness'] = 100 - train['humidity']
train['sunshine_percentage'] = train['sunshine'] / (train['sunshine'] + train['cloud'] + 1e-5)
train['weather_index'] = (0.4 * train['humidity']) + (0.3 * train['cloud']) - (0.3 * train['sunshine'])

test['humidity_cloud_interaction'] = test['humidity'] * test['cloud']
test['humidity_sunshine_interaction'] = test['humidity'] * test['sunshine']
test['cloud_sunshine_ratio'] = test['cloud'] / (test['sunshine'] + 1e-5)
test['relative_dryness'] = 100 - test['humidity']
test['sunshine_percentage'] = test['sunshine'] / (test['sunshine'] + test['cloud'] + 1e-5)
test['weather_index'] = (0.4 * test['humidity']) + (0.3 * test['cloud']) - (0.3 * test['sunshine'])


train.columns


# Separate features and target
feature_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed', 'humidity_cloud_interaction',
       'humidity_sunshine_interaction', 'cloud_sunshine_ratio',
       'relative_dryness', 'sunshine_percentage', 'weather_index']

X = train[feature_cols].values
y = train['rainfall'].values


# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,   # 20% data for validation
    random_state=42, # Ensures reproducibility
    stratify=y       # Maintains class balance
)


# Scale features (helps neural networks converge faster)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)


# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # shape (batch_size, 1)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)


class SimpleRainfallNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.5):
        super(SimpleRainfallNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Added Batch Normalization layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, 1) # No sigmoid here; handled by BCEWithLogitsLoss

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Normalize the output of fc1
        x = self.relu(x)
        x = self.dropout(x)  
        x = self.fc2(x)
        return x

# Example initialization:
input_dim = len(feature_cols)
hidden_dim = 16
dropout = 0.3
model = SimpleRainfallNet(input_dim, hidden_dim, dropout=dropout)

# Summary
summary(model, (input_dim,))


# Create a dummy input tensor with the right dimensions (batch_size, input_dim)
dummy_input = torch.randn(2, input_dim)

# Perform a forward pass
output = model(dummy_input)

# Generate the architecture diagram
dot = make_dot(output, params=dict(model.named_parameters()))
dot


train.rainfall.value_counts()


pos_weight_value = (len(train)-sum(train['rainfall'])) / sum(train['rainfall'])


criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value]))


# weight_decay applies L2 regularization to reduce overfitting
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)


num_epochs = 200
batch_size = 32


def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


best_val_loss = float('inf')  # Best validation loss starts at infinity
patience = 50  # Stop training if no improvement for 10 epochs
no_improve_count = 0  # Tracks epochs without improvement


for epoch in range(num_epochs):
    # Enable training mode (activates dropout layers)
    model.train()
    epoch_loss = 0.0
    
    # Shuffle data each epoch
    perm = torch.randperm(X_train_t.size(0))
    X_train_t = X_train_t[perm]
    y_train_t = y_train_t[perm]
    
    for X_batch, y_batch in get_batches(X_train_t, y_train_t, batch_size):
        optimizer.zero_grad()  # Reset gradients
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        epoch_loss += loss.item() * X_batch.size(0)
    
    epoch_loss /= len(X_train_t)  # Normalize loss

    # VALIDATION
    model.eval()  # disables dropout
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
        
        # Convert outputs and true labels to numpy arrays for roc_auc_score
        val_outputs_np = val_outputs.detach().cpu().numpy().flatten()
        y_val_np = y_val_t.detach().cpu().numpy().flatten()
        roc_auc = roc_auc_score(y_val_np, val_outputs_np)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"ROC AUC: {roc_auc:.4f}")
    
    # EARLY STOPPING CHECK
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break



model.eval()
with torch.no_grad():
    val_preds_np = torch.sigmoid(model(X_val_t)).cpu().numpy().ravel()
    y_val_np = y_val_t.cpu().numpy().ravel()


roc_auc = roc_auc_score(y_val_np, val_preds_np)
print(f"Validation ROC AUC: {roc_auc:.4f}")


# Compute False Positive Rate (FPR), True Positive Rate (TPR), and Thresholds
fpr, tpr, thresholds = roc_curve(y_val_np, val_preds_np)

# Plot ROC Curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Classifier (AUC = 0.5)')

# Labels and Titles
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid()
plt.show()


# Suppose 'test' is your test DataFrame
test_ids = test["id"]  # or the appropriate ID column

# The same feature columns used in training
feature_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed', 'humidity_cloud_interaction',
       'humidity_sunshine_interaction', 'cloud_sunshine_ratio',
       'relative_dryness', 'sunshine_percentage', 'weather_index']


# Preprocess the test data (apply same scaling)
X_test = test[feature_cols].values
X_test = scaler.transform(X_test)  # Use the same scaler from training


# Convert to torch tensor
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Set model to evaluation mode
model.eval()

# Disable gradient computation during inference
with torch.no_grad():
    # Forward pass: get probabilities
    y_pred_proba = model(X_test_t).cpu().numpy().ravel() # Apply sigmoid to convert logits to probabilities [0,1]


# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': y_pred_proba
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("PyTorch submission file created!")

