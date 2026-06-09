import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, Normalizer, OrdinalEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, mean_squared_error#, root_mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from tqdm import tqdm

from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split



def root_mean_squared_error(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE)
    
    Parameters:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
    
    Returns:
        float: The RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')
origin_train = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')
origin_test = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Test.csv')


# Combine train and origin_train for training (data augmentation)
df_train = pd.concat([train.copy(), origin_train.copy()], axis=0).reset_index(drop=True)
df_test = test.copy()

# Identify numerical and categorical features
categorical_features = test.select_dtypes(include=['object', 'category']).columns
numerical_features = test.select_dtypes(include=['number']).columns
target_feature = 'BeatsPerMinute'


# Normalize numerical features using StandardScaler
scaler = StandardScaler()
df_train[numerical_features] = scaler.fit_transform(df_train[numerical_features])
df_test[numerical_features] = scaler.transform(df_test[numerical_features])

# Preview the preprocessed training data
df_train.head()


# Convert features and target to PyTorch tensors
X = torch.tensor(df_train[numerical_features].values, dtype=torch.float32)
y = torch.tensor(df_train[target_feature].values, dtype=torch.float32).view(-1, 1)

# Split the training data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Prepare test set tensors
X_test = torch.tensor(df_test[numerical_features].values, dtype=torch.float32)


# Custom dataset class to support training and inference (with or without target)
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

# Create dataset and dataloaders
train_dataset = TabularDataset(X_train, y_train)
valid_dataset = TabularDataset(X_valid, y_valid)

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TabularDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define a feedforward neural network model for tabular data
class TabularModel(nn.Module):
    def __init__(self, n_numerical_features):
        super().__init__()

        n_inputs = n_numerical_features
        
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

n_numerical_features = len(numerical_features)

# Instantiate the model
model = TabularModel(n_numerical_features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
model.to(device)
# Display model architecture
print("\nModèle architecture:")
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


fold_loss_train = []
fold_loss_val = []
fold_score_train = []
fold_score_val = []


col_widths = [10, 12, 12, 12, 12]
def print_separator():
    print("+" + "+".join("-" * w for w in col_widths) + "+")

# Affichage de l'entête
print_separator()
print(f"|{'Epoch':^{col_widths[0]}}|{'Train Loss':^{col_widths[1]}}|"
      f"{'Train RMSE':^{col_widths[2]}}|{'Val Loss':^{col_widths[3]}}|"
      f"{'Val RMSE':^{col_widths[4]}}|")
print_separator()

# Training loop
epochs = 50
for epoch in range(epochs):
    # Training phase
    model.train()
    total_train_loss  = 0
    all_train_targets = []
    all_train_outputs = []
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        all_train_targets.append(y_batch.cpu().numpy())
        all_train_outputs.append(outputs.cpu().detach().numpy())
        
    avg_train_loss = total_train_loss / len(train_loader)
    fold_loss_train.append(avg_train_loss)

    # Calcul du RMSE d'entraînement
    all_train_targets = np.concatenate(all_train_targets, axis=0)
    all_train_outputs = np.concatenate(all_train_outputs, axis=0)
    train_auc_score = root_mean_squared_error(all_train_targets, all_train_outputs)
    fold_score_train.append(train_auc_score)

    # Validation phase
    model.eval()
    total_val_loss = 0
    all_val_targets = []
    all_val_outputs = []

    with torch.no_grad():
        for x_batch_val, y_batch_val in valid_loader:
            x_batch_val = x_batch_val.to(device)
            y_batch_val = y_batch_val.to(device)

            val_outputs = model(x_batch_val)

            val_loss = criterion(val_outputs, y_batch_val.float())
            total_val_loss += val_loss.item()

            all_val_targets.append(y_batch_val.cpu().numpy())
            all_val_outputs.append(val_outputs.cpu().numpy())

    avg_val_loss = total_val_loss / len(valid_loader)
    fold_loss_val.append(avg_val_loss)

    all_val_targets = np.concatenate(all_val_targets, axis=0)
    all_val_outputs = np.concatenate(all_val_outputs, axis=0)

    val_auc_score = root_mean_squared_error(all_val_targets, all_val_outputs)
    fold_score_val.append(val_auc_score)

    # Print metrics for the epoch
    print(f"|{epoch+1:>3}/{epochs:<6}|"
          f"{avg_train_loss:^12.4f}|{train_auc_score:^12.4f}|"
          f"{avg_val_loss:^12.4f}|{val_auc_score:^12.4f}|")

print("\nTraining done !")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 4))

ax1.plot(fold_score_train)
ax1.plot(fold_score_val)
ax1.set_title('model RMSE score')
ax1.set_xlabel('epochs')
ax1.legend(['train RMSE score', 'val RMSE score'])

ax2.plot(fold_loss_train)
ax2.plot(fold_loss_val)
ax2.set_title('model loss')
ax2.set_xlabel('epochs')
ax2.legend(['train loss', 'val loss'])


predictions = []
with torch.no_grad():
    for x_batch in test_loader:
        x_batch = x_batch.to(device)
        y_pred = model(x_batch).cpu().detach().numpy()
        predictions.append(y_pred)

predictions_torch = np.concatenate(predictions)


submission_torch = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
submission_torch['BeatsPerMinute'] = predictions_torch

submission_torch


plt.hist(submission_torch.BeatsPerMinute, bins=100)
plt.title('Test Preds')
plt.ylim((0, 10_000))
plt.show()


submission_torch.to_csv("submission_torch.csv", index = False)


xgb = XGBRegressor()
xgb.fit(X_train, y_train)


y_pred = xgb.predict(X_train)
print(root_mean_squared_error(y_pred, y_train))


predictions_xgb = xgb.predict(X_test)

submission_xgb = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
submission_xgb['BeatsPerMinute'] = predictions_xgb

submission_xgb


plt.hist(submission_xgb.BeatsPerMinute, bins=100)
plt.title('Test Preds')
plt.ylim((0, 10_000))
plt.show()


submission_xgb.to_csv("submission_xgb.csv", index = False)


cat = CatBoostRegressor(verbose=0)
cat.fit(X_train.numpy(), y_train.numpy())


y_pred = cat.predict(X_train.numpy())
print(root_mean_squared_error(y_pred, y_train))


predictions_cat = cat.predict(X_test.numpy())

submission_cat = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
submission_cat['BeatsPerMinute'] = predictions_cat

submission_cat


plt.hist(submission_cat.BeatsPerMinute, bins=100)
plt.title('Test Preds')
plt.ylim((0, 10_000))
plt.show()


submission_cat.to_csv("submission_cat.csv", index = False)


lgbm = LGBMRegressor()
lgbm.fit(X_train, y_train.ravel())


y_pred = lgbm.predict(X_train)
print(root_mean_squared_error(y_pred, y_train))


predictions_lgbm = lgbm.predict(X_test)

submission_lgbm = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
submission_lgbm['BeatsPerMinute'] = predictions_lgbm

submission_lgbm


plt.hist(submission_lgbm.BeatsPerMinute, bins=100)
plt.title('Test Preds')
plt.ylim((0, 10_000))
plt.show()


submission_lgbm.to_csv("submission_lgbm.csv", index = False)

