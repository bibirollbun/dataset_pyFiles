!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


import pickle
from sklearn.preprocessing import StandardScaler

# Load tokenizer and scaler/content/artifacts
with open("/kaggle/input/model_artifacts/pytorch/default/1/char2idx.pkl", "rb") as f:
    char2idx = pickle.load(f)

with open("/kaggle/input/model_artifacts/pytorch/default/1/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Set max_len used in training
max_len = 306   # Update to your actual value


# Model definition (same as training)
import torch.nn as nn

class PolymerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, ff_dim, desc_dim, num_tasks, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(embed_dim + desc_dim, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, num_tasks)

    def forward(self, seq, desc):
        x = self.embedding(seq) + self.pos_embedding[:, :seq.size(1), :]
        x = x.permute(1, 0, 2)
        pad_mask = (seq == 0)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = x.permute(1, 0, 2)
        mask = (~pad_mask).unsqueeze(-1).float()
        x = x * mask
        seq_feat = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        features = torch.cat([seq_feat, desc], dim=1)
        out = self.relu(self.fc1(features))
        return self.fc2(out)



def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 8
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.RingCount(mol)
    ]



test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
test_ids = test_df["id"].tolist()
test_smiles = test_df["SMILES"].tolist()


# Tokenize
tokenized = []
for sm in test_smiles:
    seq = [char2idx.get(c, 0) for c in sm]
    seq += [0] * (max_len - len(seq))
    tokenized.append(seq)
seq_tensor = torch.tensor(tokenized, dtype=torch.long).to(device)

# Descriptors
desc_raw = np.array([compute_descriptors(sm) for sm in test_smiles], dtype=np.float32)
desc_raw = np.nan_to_num(desc_raw, nan=0.0)
desc_scaled = scaler.transform(desc_raw)
desc_tensor = torch.tensor(desc_scaled, dtype=torch.float32).to(device)



# Model params (same as training)
embed_dim = 64
n_heads = 4
ff_dim = 128
desc_dim = desc_tensor.shape[1]
num_tasks = 5
vocab_size = len(char2idx) + 1

# Predict from each fold
fold_preds = []

for fold in range(1, 6):
    model = PolymerModel(vocab_size, embed_dim, n_heads, ff_dim, desc_dim, num_tasks, max_len).to(device)
    model_path = f"/kaggle/input/model_artifacts/pytorch/default/1/fold{fold}_best.pt"
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        preds = model(seq_tensor, desc_tensor)
        fold_preds.append(preds.cpu().numpy())

# Average predictions
avg_preds = np.mean(fold_preds, axis=0)



submission = pd.DataFrame(avg_preds, columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'])
submission.insert(0, 'id', test_ids)
submission = submission.round(4)
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved!")



submission.head()




