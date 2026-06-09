import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
import torch.functional as f 
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error,mean_squared_log_error
from sklearn.model_selection import KFold,train_test_split
from sklearn.preprocessing import StandardScaler


# train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')

train = pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')


def ohe(df,column):
    return pd.concat([
        df.drop(column,axis=1),
        pd.get_dummies(df[column],drop_first=True, prefix=column).astype(int)
    ],axis=1)
    
train = ohe(train, 'Sex')
test = ohe(test, 'Sex')


best_param_dict = {
 'latent_dim': 512,
 'batch_size': 128,
 'lr': 0.0005,
 'weight_decay': 5e-05
}

PATIENCE = 3
N_ITER = 100
# FOLDS = 4


import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU")


class RegressionModel(nn.Module):
    
    def __init__(self, input_dim, latent_dim, dropout=0.2):
        
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim*2, 1),
            nn.Softplus() 
        )
        
    def forward(self, x):
        
        return self.model(x)


# ===== 1️⃣ Model Definition =====
class RegressionModel(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, 1)
        )

    def forward(self, x):
        return self.model(x)


# ===== 2️⃣ Hyperparameters ===== 
best_param_dict = best_param_dict
N_iterations = N_ITER
latent_dim   = best_param_dict['latent_dim']
batch_size   = best_param_dict['batch_size']
lr           = best_param_dict['lr']
weight_decay = best_param_dict['weight_decay']


# ====== 3️⃣ Data Scaling =====
sc = StandardScaler()

train_set, val_set = train_test_split(train, test_size=0.1)
train_scaled = sc.fit_transform(train_set.drop(['id', 'Rings'], axis=1))
val_scaled = sc.transform(val_set.drop(['id', 'Rings'], axis=1))
test_scaled  = sc.transform(test.drop(['id'], axis=1))

X_train = torch.tensor(train_scaled, dtype=torch.float32).to(device)
y_train = torch.tensor(train_set['Rings'].values, dtype=torch.float32).to(device)
X_val = torch.tensor(val_scaled, dtype=torch.float32).to(device)
y_val = torch.tensor(val_set['Rings'].values, dtype=torch.float32).to(device)
X_test  = torch.tensor(test_scaled, dtype=torch.float32).to(device)


# ===== 4. Model Initialization =====
input_dim = X_train.size(1)
model = RegressionModel(input_dim=input_dim, latent_dim=latent_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = torch.nn.MSELoss()

train_dataset = TensorDataset(X_train, y_train)
train_DL = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# ===== 5. Training Loop =====
train_loss_list = []
val_loss_list = []
best_val_loss = float('inf')
wait = 0  # counter for early stopping
patience = PATIENCE 
for epoch in range(N_iterations):
    model.train()
    epoch_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in train_DL:
        y_pred = model(X_batch)
        batch_loss = loss_fn(y_pred, y_batch.unsqueeze(1))  # ensure same shape
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item() * len(y_batch)  # batch_sq_error = batch_mean_sq_error * batch_n
        total_samples += len(y_batch)

    avg_loss = epoch_loss / total_samples # total_sq_error / total_n = toal_mean_sq_error
    train_loss_list.append(avg_loss)
        
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = loss_fn(y_val_pred, y_val.unsqueeze(1)).item()  # ensure same shape
        val_loss_list.append(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        best_model_state = model.state_dict()  # save best model
    else:
        wait += 1

    if wait >= patience:
        print(f"Early stopping triggered at iteration {epoch}")
        break

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: train_loss = {avg_loss:.6f} | val_loss = {val_loss:.6f} ")


plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend() 
plt.grid(True)
plt.show()


# ====================================================
# 6️⃣ Predictions
# ====================================================
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train)
    y_val_pred   = model(X_val)
    y_test_pred  = model(X_test)

# ====================================================
# 7️⃣ Evaluate on Training & Val Set
# ====================================================
y_true_train = y_train.detach().cpu().numpy()
y_pred_train = np.clip(y_train_pred.detach().cpu().numpy(), a_min=0, a_max=None)
rmsle_train = np.sqrt(mean_squared_log_error(y_true_train, y_pred_train))

y_true_val = y_val.detach().cpu().numpy()
y_pred_val = np.clip(y_val_pred.detach().cpu().numpy(), a_min=0, a_max=None)
rmsle_val = np.sqrt(mean_squared_log_error(y_true_val, y_pred_val))

print("-" * 50)
print("Train RMSLE:", rmsle_train)
print("Val RMSLE  :", rmsle_val)


# ====================================================
# 8️⃣ Test Predictions
# ====================================================
y_test_pred_np = np.clip(y_test_pred.detach().cpu().numpy(), a_min=0, a_max=None)
print("Test predictions shape:", y_test_pred_np.shape)


submission = test[['id']].copy()
submission['Rings'] = y_test_pred_np


submission['Rings'].describe()


submission.to_csv('v1_vanilla_DL.csv', index=False)




