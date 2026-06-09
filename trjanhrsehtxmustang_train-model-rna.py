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





import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# === 1) Load & Combine Training + Validation for Training ===
dftrain_seq   = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.v2.csv")
dftrain_lab   = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.v2.csv")
dfvalid_seq   = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
dfvalid_lab   = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")

combined_seqs_df  = pd.concat([dftrain_seq, dfvalid_seq], ignore_index=True)
combined_labs_df  = pd.concat([dftrain_lab, dfvalid_lab], ignore_index=True)

# === 2) Helper Functions ===
def encode_sequence(seq):
    mapping = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'U':[0,0,0,1]}
    return [mapping.get(b,[0,0,0,0]) for b in seq]

def clean_coords(c):
    c = c.copy()
    c[~np.isfinite(c)] = np.nan
    for j in range(c.shape[1]):
        col = c[:, j]
        if np.isnan(col).all():
            c[:, j] = 0.0
        else:
            m = np.nanmean(col)
            col[np.isnan(col)] = m if np.isfinite(m) else 0.0
            c[:, j] = col
    return np.clip(c, -100, 100)

def merge_labels(seqs_df, labs_df):
    S, C = [], []
    for _, row in seqs_df.iterrows():
        tid = row['target_id']
        enc = encode_sequence(row['sequence'])
        L   = len(enc)

        lab = labs_df[labs_df['ID'].str.startswith(tid + '_')]
        if lab.empty: continue
        lab = lab.sort_values('resid')

        coord = np.full((L, 15), np.nan, dtype=np.float32)
        for i, (x,y,z) in enumerate(zip(
            ['x_1','x_2','x_3','x_4','x_5'],
            ['y_1','y_2','y_3','y_4','y_5'],
            ['z_1','z_2','z_3','z_4','z_5']
        )):
            if x in lab.columns:
                vals = lab[[x,y,z]].to_numpy(dtype=np.float32)
                m = min(vals.shape[0], L)
                coord[:m, i*3:(i+1)*3] = vals[:m]

        coord = clean_coords(coord)
        S.append(torch.tensor(enc,  dtype=torch.float32))
        C.append(torch.tensor(coord, dtype=torch.float32))
    return S, C

# === 3) Prepare Training Data ===
seq_tensors, coord_tensors = merge_labels(combined_seqs_df, combined_labs_df)
maxL = max(s.shape[0] for s in seq_tensors)
for i in range(len(seq_tensors)):
    pad = maxL - seq_tensors[i].shape[0]
    if pad>0:
        seq_tensors[i] = torch.cat([seq_tensors[i],   torch.zeros(pad,4)])
        coord_tensors[i] = torch.cat([coord_tensors[i], torch.zeros(pad,15)])

class RNADataset(Dataset):
    def __init__(self, seqs, coords):
        self.seqs, self.coords = seqs, coords
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        return self.seqs[i], self.coords[i]

train_loader = DataLoader(RNADataset(seq_tensors, coord_tensors),
                          batch_size=1, shuffle=True)

# === 4) Define Model ===
class SimpleRNA3DModel(nn.Module):
    def __init__(self, feat_dim=4, hid=128, out=15):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hid)
        self.fc2 = nn.Linear(hid, out)
    def forward(self,x):
        return self.fc2(torch.relu(self.fc1(x)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleRNA3DModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
lossf     = nn.MSELoss()

# === 5) Training Loop ===
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for seq_batch, coord_batch in train_loader:
        sb = seq_batch.squeeze(0).to(device)  # [L,4]
        cb = coord_batch.squeeze(0).to(device)  # [L,15]

        optimizer.zero_grad()
        pred = model(sb)
        loss = lossf(pred, cb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10 — Avg Loss: {total_loss/len(train_loader):.4f}")

# === 6) Robust Inference & Submission ===
# 6a) Load sample_submission
sample = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")
test_ids = sample['ID'].str.extract(r'(^.+)_\d+$')[0].unique()

# 6b) Pick correct sequences file
base = "/kaggle/input/stanford-rna-3d-folding"
if   os.path.exists(f"{base}/test_sequences.csv"):
    test_df = pd.read_csv(f"{base}/test_sequences.csv")
elif os.path.exists(f"{base}/validation_sequences.csv"):
    test_df = pd.read_csv(f"{base}/validation_sequences.csv")
else:
    raise FileNotFoundError("No test_sequences.csv or validation_sequences.csv found")

test_df = test_df[test_df['target_id'].isin(test_ids)].reset_index(drop=True)

# 6c) Encode & pad
seqs_test = [encode_sequence(s) for s in test_df['sequence']]
for i,s in enumerate(seqs_test):
    pad = maxL - len(s)
    if pad>0:
        seqs_test[i] = s + [[0,0,0,0]]*pad

# 6d) Predict one by one
model.eval()
preds = []
with torch.no_grad():
    for s in seqs_test:
        x = torch.tensor(s, dtype=torch.float32).to(device)
        out = model(x).cpu().numpy()  # [L,15]
        preds.append(out)

# 6e) Build submission rows
rows = []
for i, tid in enumerate(test_df['target_id']):
    L = len(encode_sequence(test_df.loc[i,'sequence']))
    for j in range(L):
        row = {'ID': f"{tid}_{j+1}"}
        for k in range(5):
            row[f"x_{k+1}"] = float(preds[i][j, 3*k+0])
            row[f"y_{k+1}"] = float(preds[i][j, 3*k+1])
            row[f"z_{k+1}"] = float(preds[i][j, 3*k+2])
        rows.append(row)

sub = pd.DataFrame(rows)
sub = pd.merge(sample[['ID']], sub, on='ID', how='left').fillna(0.0)
sub.to_csv("submission.csv", index=False)
print("✅ submission.csv written")











