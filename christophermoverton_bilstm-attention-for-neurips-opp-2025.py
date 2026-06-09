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


!pip install -q rdkit



# Step 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load the dataset
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

# Step 3: Basic overview
print(train_df.head())
print(train_df.info())
print(train_df.describe())



target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train_df[target_cols].hist(bins=30, figsize=(14, 10))
plt.suptitle("Target Distributions", fontsize=16)
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 8))
sns.heatmap(train_df[target_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



# Reload train.csv to ensure it's not empty
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
print(train_df.shape)
train_df.head()



targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train_df[targets].isna().sum()


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import copy
import os

# Config
batch_size = 64
n_epochs = 100
max_len = 120
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Targets
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Tokenizer (shared for all targets)
df_all = train_df[['SMILES']].dropna().copy()
charset = sorted(set("".join(df_all['SMILES'])))
char_to_idx = {c: i + 1 for i, c in enumerate(charset)}
vocab_size = len(char_to_idx) + 1

# SMILES to sequence
def smiles_to_seq(smiles):
    return [char_to_idx.get(c, 0) for c in smiles]

# Create models folder
os.makedirs('models', exist_ok=True)




class SmilesBiLSTM_Attn(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(SmilesBiLSTM_Attn, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out



for target in targets:
    print(f"\n========== Training for target: {target} ==========")
    
    # Filter rows with this target
    df_target = train_df[['SMILES', target]].dropna().copy()
    print(f"Samples for {target}: {len(df_target)}")
    
    # Tokenize + pad
    df_target['SMILES_seq'] = df_target['SMILES'].apply(smiles_to_seq)
    X = pad_sequences(df_target['SMILES_seq'], maxlen=max_len, padding='post', truncating='post')
    y = df_target[target].values.reshape(-1, 1)

    # Scale target
    scaler = StandardScaler()
    y = scaler.fit_transform(y)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # Model
    model = SmilesBiLSTM_Attn(vocab_size=vocab_size).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    best_val_loss = np.inf
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss = loss_fn(preds, y_batch)
                val_loss_total += val_loss.item() * X_batch.size(0)
        
        epoch_val_loss = val_loss_total / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save scaler also
            scaler_y = scaler
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save model + scaler
    torch.save(model.state_dict(), f'models/model_{target}.pth')
    np.save(f'models/scaler_{target}.npy', scaler_y.mean_)
    np.save(f'models/scale_{target}.npy', scaler_y.scale_)
    
    # Plot learning curve
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(f'Learning Curve — {target}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



# Load test.csv
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
print(f"Test samples: {len(test_df)}")



test_df.head()


# Tokenize
test_df['SMILES_seq'] = test_df['SMILES'].apply(smiles_to_seq)

# Pad
X_test = pad_sequences(test_df['SMILES_seq'], maxlen=max_len, padding='post', truncating='post')
X_test = torch.tensor(X_test, dtype=torch.long).to(device)



# Prepare submission df
submission = pd.DataFrame({'id': test_df['id']})

# Loop over targets
for target in targets:
    print(f"\n=== Predicting target: {target} ===")
    
    # Load model
    model = SmilesBiLSTM_Attn(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(f'models/model_{target}.pth'))
    model.eval()
    
    # Load scaler
    scaler_mean = np.load(f'models/scaler_{target}.npy')
    scaler_scale = np.load(f'models/scale_{target}.npy')
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            batch_preds = model(X_batch).detach().cpu().numpy()
            preds.append(batch_preds)
    
    preds = np.vstack(preds).flatten()
    
    # Inverse scale
    preds = preds * scaler_scale + scaler_mean
    
    # Add to submission df
    submission[target] = preds



# Save submission.csv
submission.to_csv('submission.csv', index=False)
print("Saved: submission.csv")





