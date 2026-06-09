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
        pass
        # print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd

# Define the data folder
data_dir = "/kaggle/input/stanford-rna-3d-folding"

# Define file paths for the sequence and label files
train_sequences_path = os.path.join(data_dir, "train_sequences.csv")
train_labels_path = os.path.join(data_dir, "train_labels.csv")
val_sequences_path   = os.path.join(data_dir, "validation_sequences.csv")
val_labels_path      = os.path.join(data_dir, "validation_labels.csv")

# Load CSV files into DataFrames
train_seq_df = pd.read_csv(train_sequences_path)
train_labels_df = pd.read_csv(train_labels_path)
val_seq_df = pd.read_csv(val_sequences_path)
val_labels_df = pd.read_csv(val_labels_path)

# Convert 'temporal_cutoff' column to datetime for proper filtering
train_seq_df['temporal_cutoff'] = pd.to_datetime(train_seq_df['temporal_cutoff'], errors='coerce')
val_seq_df['temporal_cutoff'] = pd.to_datetime(val_seq_df['temporal_cutoff'], errors='coerce')

# Filter training sequences based on temporal cutoff.
# As noted in the dataset description, only train sequences with a cutoff before 2022-05-27 should be used.
cutoff_date = pd.to_datetime('2022-05-27')
filtered_train_seq_df = train_seq_df[train_seq_df['temporal_cutoff'] < cutoff_date].copy()

print("Number of training sequences after filtering:", len(filtered_train_seq_df))
print("Number of validation sequences:", len(val_seq_df))

# The label files have an 'ID' column in the form 'target_id_resid'
# We extract the target_id (everything before the last underscore) for merging.
train_labels_df['target_id'] = train_labels_df['ID'].str.rsplit('_', n=1, expand=True)[0]
val_labels_df['target_id']   = val_labels_df['ID'].str.rsplit('_', n=1, expand=True)[0]

# Merge the sequences with the labels based on 'target_id'
# Note: Each RNA sequence (in train_seq_df or val_seq_df) corresponds to multiple rows in the labels files (one per residue)
train_data = pd.merge(filtered_train_seq_df, train_labels_df, on="target_id", how="inner")
val_data   = pd.merge(val_seq_df, val_labels_df, on="target_id", how="inner")

print("Training data shape (after merging):", train_data.shape)
print("Validation data shape (after merging):", val_data.shape)

# Optionally, save the processed data for later use
train_data.to_csv("processed_train_data.csv", index=False)
val_data.to_csv("processed_val_data.csv", index=False)



import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Utility Functions and Setup
# --------------------------
torch.manual_seed(42)

# Mapping nucleotides (with unknown token for non-ACGU)
nucleotide_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
unknown_token_index = 4  # for any unknown characters
vocab_size = len(nucleotide_to_idx) + 1  # total vocab size

def sequence_to_tensor(seq):
    """Convert nucleotide sequence string to tensor of indices; unknown tokens get unknown_token_index."""
    indices = [nucleotide_to_idx.get(nt, unknown_token_index) for nt in seq]
    return torch.tensor(indices, dtype=torch.long)

# --------------------------
# Dataset Definition with Filtering
# --------------------------

class RNA3DDataset(Dataset):
    """
    Groups processed data by target_id.
    Returns:
      - seq_tensor: Tensor of indices (one per residue) from the 'resname' column.
      - coords_tensor: Tensor of ground truth coordinates (from x_1, y_1, z_1) for each residue.
    """
    def __init__(self, processed_csv, coord_threshold=-1e10):
        self.data = pd.read_csv(processed_csv)
        
        # Ensure coordinate columns are numeric
        for col in ['x_1', 'y_1', 'z_1']:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Drop rows with NaN in coordinate columns
        self.data.dropna(subset=['x_1', 'y_1', 'z_1'], inplace=True)
        
        # Filter out rows where any coordinate is extremely low (likely a placeholder)
        mask = (self.data['x_1'] > coord_threshold) & \
               (self.data['y_1'] > coord_threshold) & \
               (self.data['z_1'] > coord_threshold)
        self.data = self.data[mask]
        
        # Group by target_id so that each group represents one RNA target
        self.groups = self.data.groupby('target_id')
        self.target_ids = list(self.groups.groups.keys())
    
    def __len__(self):
        return len(self.target_ids)
    
    def __getitem__(self, idx):
        target_id = self.target_ids[idx]
        group = self.groups.get_group(target_id)
        
        # Construct the input sequence using the 'resname' column
        residue_list = group['resname'].tolist()
        seq_tensor = torch.tensor(
            [nucleotide_to_idx.get(nt, unknown_token_index) for nt in residue_list],
            dtype=torch.long
        )
        
        # Extract the coordinates (using only x_1, y_1, z_1)
        coords = group[['x_1', 'y_1', 'z_1']].values  # shape: (num_residues, 3)
        coords_tensor = torch.tensor(coords, dtype=torch.float)
        
        # Sanity check: the length of the sequence should equal number of coordinate rows
        if seq_tensor.size(0) != coords_tensor.size(0):
            raise ValueError(f"Mismatch for target_id {target_id}: sequence length {seq_tensor.size(0)} vs. {coords_tensor.size(0)} coordinates")
        
        # Optionally, you might normalize coordinates per target here (e.g., center them)
        # coords_tensor = coords_tensor - coords_tensor.mean(dim=0, keepdim=True)
        
        return seq_tensor, coords_tensor

# --------------------------
# Positional Encoding Module
# --------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# --------------------------
# RNATransformer Model Definition
# --------------------------

class RNATransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=3, vocab_size=vocab_size):
        super(RNATransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 3)  # Predict 3 coordinates per residue
        
    def forward(self, src):
        # src: (batch_size, seq_len) with token indices
        x = self.embedding(src)       # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        coords = self.fc_out(x)          # (batch_size, seq_len, 3)
        # Center predicted coordinates for translation invariance
        coords = coords - coords.mean(dim=1, keepdim=True)
        return coords

# --------------------------
# Data Loading and Preparation
# --------------------------
train_csv = "processed_train_data.csv"
val_csv   = "processed_val_data.csv"  # or you may use a filtered version if you saved one

train_dataset = RNA3DDataset(train_csv)
val_dataset = RNA3DDataset(val_csv)

# Use batch size 1 because sequences have variable lengths
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# --------------------------
# Model Training Setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNATransformer(d_model=128, nhead=8, num_layers=3, vocab_size=vocab_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10
max_grad_norm = 1.0  # For gradient clipping

# --------------------------
# Training Loop
# --------------------------
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for seq_tensor, coords_tensor in train_loader:
        seq_tensor = seq_tensor.to(device)         # shape: (1, seq_len)
        coords_tensor = coords_tensor.to(device)     # shape: (1, seq_len, 3)
        
        optimizer.zero_grad()
        pred_coords = model(seq_tensor)              # (1, seq_len, 3)
        
        # Optional: check for NaNs in predictions
        if torch.isnan(pred_coords).any():
            print("NaN detected in predictions for one batch!")
            continue
        
        loss = criterion(pred_coords, coords_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seq_tensor, coords_tensor in val_loader:
            seq_tensor = seq_tensor.to(device)
            coords_tensor = coords_tensor.to(device)
            pred_coords = model(seq_tensor)
            loss = criterion(pred_coords, coords_tensor)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), "rna_transformer_model.pth")
print("Model training complete and saved.")



import os
import pandas as pd
import torch
import numpy as np

# --------------------------
# Utility: Map nucleotides to indices
# --------------------------
nucleotide_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
unknown_token_index = 4
vocab_size = len(nucleotide_to_idx) + 1

def sequence_to_tensor(seq):
    """Converts a nucleotide sequence string to a tensor of indices.
       Unknown tokens are mapped to unknown_token_index."""
    indices = [nucleotide_to_idx.get(nt, unknown_token_index) for nt in seq]
    return torch.tensor(indices, dtype=torch.long)

# --------------------------
# Model Definition (should match your training code)
# --------------------------
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RNATransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=3, vocab_size=vocab_size):
        super(RNATransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 3)  # predict 3 coordinates per residue
        
    def forward(self, src):
        # src shape: (batch_size, seq_len)
        x = self.embedding(src)            # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)      # (batch_size, seq_len, d_model)
        coords = self.fc_out(x)              # (batch_size, seq_len, 3)
        # Center coordinates (translation invariance)
        coords = coords - coords.mean(dim=1, keepdim=True)
        return coords

# --------------------------
# Load the Trained Model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNATransformer(d_model=128, nhead=8, num_layers=3, vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("rna_transformer_model.pth", map_location=device))
# We'll use Monte Carlo dropout to generate diverse predictions:
# Set the model to train mode so that dropout is active at inference.
model.train()

# --------------------------
# Function to Generate Five Predictions Using Monte Carlo Dropout
# --------------------------
def predict_five_structures(model, seq_tensor, n_predictions=5):
    """
    Generates n_predictions for the given sequence tensor.
    seq_tensor: a tensor of shape (seq_len,) representing nucleotide indices.
    Returns a list of numpy arrays each of shape (seq_len, 3).
    """
    predictions = []
    with torch.no_grad():
        for i in range(n_predictions):
            # Note: by leaving dropout on (model.train()), we sample different predictions.
            pred = model(seq_tensor.unsqueeze(0))  # (1, seq_len, 3)
            predictions.append(pred.squeeze(0).cpu().numpy())
    return predictions

# --------------------------
# Generate Submission File
# --------------------------
# Load the test sequences.
test_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

submission_rows = []

# Loop over each target in the test set.
for idx, row in test_df.iterrows():
    target_id = row['target_id']
    sequence = row['sequence'].strip()  # assume sequence is a string of A, C, G, U
    seq_tensor = sequence_to_tensor(sequence).to(device)  # shape: (seq_len,)
    
    # Generate five predictions for this sequence.
    preds = predict_five_structures(model, seq_tensor, n_predictions=5)
    seq_len = len(sequence)
    
    # For each residue in the sequence, form a submission row.
    # Residue numbering: use 1-indexing.
    for i in range(seq_len):
        resid = i + 1
        # Build the unique ID for the residue: e.g. "R1107_1", "R1107_2", etc.
        row_id = f"{target_id}_{resid}"
        resname = sequence[i]  # nucleotide letter at this position
        
        # Gather coordinates from all five predictions.
        # Each prediction is a (seq_len, 3) array.
        coords = []
        for p in range(5):
            x, y, z = preds[p][i]  # coordinates for residue i in prediction p
            coords.extend([x, y, z])
        
        # Create a dictionary for this residue.
        submission_row = {
            "ID": row_id,
            "resname": resname,
            "resid": resid,
            "x_1": coords[0],
            "y_1": coords[1],
            "z_1": coords[2],
            "x_2": coords[3],
            "y_2": coords[4],
            "z_2": coords[5],
            "x_3": coords[6],
            "y_3": coords[7],
            "z_3": coords[8],
            "x_4": coords[9],
            "y_4": coords[10],
            "z_4": coords[11],
            "x_5": coords[12],
            "y_5": coords[13],
            "z_5": coords[14],
        }
        submission_rows.append(submission_row)

# Create a DataFrame for submission.
submission_df = pd.DataFrame(submission_rows)

# Save to CSV.
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")


