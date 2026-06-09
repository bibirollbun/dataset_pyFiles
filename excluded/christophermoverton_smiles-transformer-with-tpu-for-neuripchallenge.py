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



from collections import Counter

# Build character vocab from all SMILES strings
all_smiles = ''.join(train_df['SMILES'].dropna().tolist())
char_counts = Counter(all_smiles)

# Assign an index to each character
vocab = {c: i+1 for i, c in enumerate(char_counts.keys())}  # +1 because 0 = padding
vocab['<PAD>'] = 0

print("Vocab size:", len(vocab))
print(vocab)



!pip install -q rdkit


targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train_df[targets].isna().sum()


# === Hyperparameters ===
max_len = 120
batch_size = 64
num_epochs = 60  # you can use more epochs now



# === Imports ===
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import pandas as pd
import numpy as np
import math
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

# === Load data ===
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

# === Randomized SMILES ===
def randomize_smiles(smiles, num_aug=1):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * num_aug
    randomized = []
    for _ in range(num_aug):
        randomized_smiles = Chem.MolToSmiles(mol, doRandom=True)
        randomized.append(randomized_smiles)
    return randomized

# === Tokenize + Pad ===
def smiles_to_token_ids(smiles, vocab):
    return [vocab.get(c, 0) for c in smiles]

def pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    padded = np.full((len(sequences), maxlen), value)
    for i, seq in enumerate(sequences):
        if truncating == 'pre':
            trunc = seq[-maxlen:]
        else:
            trunc = seq[:maxlen]
        trunc = np.array(trunc)
        if padding == 'post':
            padded[i, :len(trunc)] = trunc
        else:
            padded[i, -len(trunc):] = trunc
    return padded

# === Dataset class ===
from torch.utils.data import Dataset

class SmilesDataset(Dataset):
    def __init__(self, df, target_col, vocab, max_len=120, augment=False):
        self.smiles_list = df['SMILES'].tolist()
        self.targets = df[target_col].tolist()
        self.vocab = vocab
        self.max_len = max_len
        self.augment = augment
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]
        
        if self.augment:
            smiles = randomize_smiles(smiles, num_aug=1)[0]
        
        token_ids = smiles_to_token_ids(smiles, self.vocab)
        token_ids = pad_sequences([token_ids], maxlen=self.max_len)[0]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(target, dtype=torch.float)

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# === Transformer model ===
class SmilesTransformerRegressorPE(nn.Module):
    def __init__(self, vocab_size, embedding_dim=384, nhead=8, num_layers=6, dim_feedforward=768, dropout=0.3, max_len=120):
        super(SmilesTransformerRegressorPE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        transformer_out = self.transformer(embedded)
        context = transformer_out.mean(dim=1)
        out = self.fc(context)
        return out

# === TPU device ===
device = xm.xla_device()

# === Hyperparameters ===
max_len = 120
batch_size = 64
num_epochs = 60

# === Target list ===
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# === Main loop: one target at a time ===
for target_col in targets:
    print(f"\n========== TRAINING TARGET: {target_col} ==========")
    
    df_target = train_df[train_df[target_col].notnull()].reset_index(drop=True)
    
    # Scale target
    scaler = StandardScaler()
    df_target[f'{target_col}_scaled'] = scaler.fit_transform(df_target[[target_col]])
    
    # Build vocab
    all_smiles = ''.join(df_target['SMILES'].dropna().tolist())
    char_counts = Counter(all_smiles)
    vocab = {c: i+1 for i, c in enumerate(char_counts.keys())}
    vocab['<PAD>'] = 0
    
    print(f"Vocab size: {len(vocab)}")
    
    # Split train/valid
    train_df_split, valid_df_split = train_test_split(df_target, test_size=0.1, random_state=42)
    
    # Create datasets
    train_dataset = SmilesDataset(train_df_split, target_col=f'{target_col}_scaled', vocab=vocab, max_len=max_len, augment=True)
    valid_dataset = SmilesDataset(valid_df_split, target_col=f'{target_col}_scaled', vocab=vocab, max_len=max_len, augment=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    valid_device_loader = pl.MpDeviceLoader(valid_loader, device)
    
    # Build model
    model = SmilesTransformerRegressorPE(vocab_size=len(vocab), max_len=max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_device_loader:
            optimizer.zero_grad()
            inputs, targets_batch = batch
            inputs = inputs.to(device)
            targets_batch = targets_batch.to(device)
            
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets_batch)
            
            loss.backward()
            xm.optimizer_step(optimizer)
            xm.mark_step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_device_loader:
                inputs, targets_batch = batch
                inputs = inputs.to(device)
                targets_batch = targets_batch.to(device)
                
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets_batch)
                total_valid_loss += loss.item()
                xm.mark_step()
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.5f} | Valid Loss: {avg_valid_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save model + scaler + vocab
    torch.save(model.state_dict(), f'/kaggle/working/smiles_transformer_{target_col.lower()}_scaled.pth')
    joblib.dump(scaler, f'/kaggle/working/scaler_{target_col.lower()}.pkl')
    joblib.dump(vocab, f'/kaggle/working/vocab_{target_col.lower()}.pkl')
    
    print(f"âœ… Saved model, scaler, vocab for {target_col}!\n")



# === Imports ===
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from collections import Counter


# === Tokenize + Pad ===
def smiles_to_token_ids(smiles, vocab):
    return [vocab.get(c, 0) for c in smiles]

def pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    padded = np.full((len(sequences), maxlen), value)
    for i, seq in enumerate(sequences):
        if truncating == 'pre':
            trunc = seq[-maxlen:]
        else:
            trunc = seq[:maxlen]
        trunc = np.array(trunc)
        if padding == 'post':
            padded[i, :len(trunc)] = trunc
        else:
            padded[i, -len(trunc):] = trunc
    return padded

# === Model class (must match training) ===
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class SmilesTransformerRegressorPE(nn.Module):
    def __init__(self, vocab_size, embedding_dim=384, nhead=8, num_layers=6, dim_feedforward=768, dropout=0.3, max_len=120):
        super(SmilesTransformerRegressorPE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        transformer_out = self.transformer(embedded)
        context = transformer_out.mean(dim=1)
        out = self.fc(context)
        return out

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# === Load test set ===
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
test_df['SMILES'] = test_df['SMILES'].fillna('')

# === Define target names ===
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# === Model paths ===
model_info = {}
for target in targets:
    model_info[target] = {
        'model_path': f'/kaggle/working/smiles_transformer_{target.lower()}_scaled.pth',
        'scaler_path': f'/kaggle/working/scaler_{target.lower()}.pkl',
        'vocab_path': f'/kaggle/working/vocab_{target.lower()}.pkl'
    }

# === Predict for each target ===
max_len = 120
batch_size = 64

# To collect predictions:
all_predictions = {'id': test_df['id'].values}

for target in targets:
    print(f"\n--- Predicting target: {target} ---")
    
    # Load vocab
    vocab = joblib.load(model_info[target]['vocab_path'])
    vocab_size = len(vocab)
    
    # Build model
    model = SmilesTransformerRegressorPE(vocab_size=vocab_size, max_len=max_len)
    model.load_state_dict(torch.load(model_info[target]['model_path']))
    model = model.to(device)
    model.eval()
    
    # Load scaler
    scaler = joblib.load(model_info[target]['scaler_path'])
    
    # Predict in batches
    preds = []
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i+batch_size]
        smiles_list = batch['SMILES'].tolist()
        
        token_ids_batch = []
        for smiles in smiles_list:
            token_ids = smiles_to_token_ids(smiles, vocab)
            token_ids = pad_sequences([token_ids], maxlen=max_len)[0]
            token_ids_batch.append(token_ids)
        
        input_tensor = torch.tensor(token_ids_batch, dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs_scaled = model(input_tensor).squeeze().cpu().numpy()
            outputs = scaler.inverse_transform(outputs_scaled.reshape(-1, 1)).reshape(-1)
            preds.extend(outputs)
    
    # Store predictions
    all_predictions[target] = preds
    print(f"âœ… Done target: {target}")

# === Build submission DataFrame ===
submission_df = pd.DataFrame(all_predictions)
submission_df = submission_df[['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# === Save submission ===
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("\nðŸŽ‰ Submission file saved: /kaggle/working/submission.csv")





