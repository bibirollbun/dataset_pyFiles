import os
import pandas as pd
import numpy as np
import RNA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
import subprocess

MAX_SEQ_LEN = None
vocab = {'A': 1, 'C': 2, 'G': 3, 'U': 4, '<pad>': 0}

def encode(seq: str) -> torch.LongTensor:
    return torch.LongTensor([vocab.get(nt, 0) for nt in seq])

def compute_ss_matrix(seq: str) -> np.ndarray:
    dotbracket, _ = RNA.fold(seq)
    N = len(seq)
    ss_mat = np.zeros((N, N), dtype=np.float32)
    stack = []
    for i, ch in enumerate(dotbracket):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                ss_mat[i, j] = 1.0
                ss_mat[j, i] = 1.0
    return ss_mat

def compute_tmscore(ref_coords, pred_coords):
    np.savetxt('ref.xyz', ref_coords, header='', comments='')
    np.savetxt('pred.xyz', pred_coords, header='', comments='')
    result = subprocess.run(['./TMscore', 'ref.xyz', 'pred.xyz'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith('TM-score'):
            return float(line.split('=')[1].strip().split()[0])
    return None

def pairwise_dist(coords: torch.Tensor):
    return torch.cdist(coords, coords)

def extract_base_id(full_id: str) -> str:
    return full_id.rsplit('_', 1)[0]



class RNADataset(Dataset):
    def __init__(self, seq_df: pd.DataFrame, coord_df: pd.DataFrame):
        self.seq_df = seq_df.copy()
        self.coord_df = coord_df.copy()
        self.groups = self.coord_df.groupby('target_id')
        self.ids = sorted(self.groups.groups.keys())
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx: int):
        tid = self.ids[idx]
        seq_str = self.seq_df.loc[tid, 'sequence']
        seq_encoded = encode(seq_str)
        grp = self.groups.get_group(tid).sort_values('resid')
        coords = grp[['x_1', 'y_1', 'z_1']].values
        ss_mat = compute_ss_matrix(seq_str)
        return seq_encoded, coords, ss_mat, tid

def custom_collate(batch):
    if MAX_SEQ_LEN is None:
        max_len = max(x[0].shape[0] for x in batch)
    else:
        max_len = min(max(x[0].shape[0] for x in batch), MAX_SEQ_LEN)

    seq_list, coords_list, ss_list, mask_list, tids = [], [], [], [], []

    for (seq_enc, coords, ss_mat, tid) in batch:
        L = seq_enc.shape[0]
        if MAX_SEQ_LEN is not None and L > MAX_SEQ_LEN:
            seq_enc = seq_enc[:MAX_SEQ_LEN]
            coords = coords[:MAX_SEQ_LEN]
            ss_mat = ss_mat[:MAX_SEQ_LEN, :MAX_SEQ_LEN]
            L = MAX_SEQ_LEN

        # Sequence is used for an embedding layer => keep it long/int
        seq_pad = F.pad(seq_enc, (0, max_len - L), value=0).long()

        # Convert coords to float
        coords_pad = F.pad(
            torch.from_numpy(coords).float(),  # <-- .float() here
            (0, 0, 0, max_len - L),
            value=0.0
        )

        # Convert ss_mat to float
        ss_pad = F.pad(
            torch.from_numpy(ss_mat).float(),  # <-- .float() here
            (0, max_len - L, 0, max_len - L),
            value=0.0
        )

        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:L] = True

        seq_list.append(seq_pad)
        coords_list.append(coords_pad)
        ss_list.append(ss_pad)
        mask_list.append(mask)
        tids.append(tid)

    seq_tensor = torch.stack(seq_list, dim=0)
    coords_tensor = torch.stack(coords_list, dim=0)
    ss_tensor = torch.stack(ss_list, dim=0)
    mask_tensor = torch.stack(mask_list, dim=0)

    return seq_tensor, coords_tensor, ss_tensor, mask_tensor, tids


class RNATestDataset(Dataset):
    def __init__(self, test_seq_df: pd.DataFrame):
        self.test_seq_df = test_seq_df.copy()
        self.ids = sorted(self.test_seq_df.index.unique())
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        tid = self.ids[idx]
        seq_str = self.test_seq_df.loc[tid, 'sequence']
        seq_enc = encode(seq_str)
        ss_mat = compute_ss_matrix(seq_str)
        return seq_enc, ss_mat, tid

def test_collate(batch):
    if MAX_SEQ_LEN is None:
        max_len = max(x[0].shape[0] for x in batch)
    else:
        max_len = min(max(x[0].shape[0] for x in batch), MAX_SEQ_LEN)
    seq_list, ss_list, mask_list, tids = [], [], [], []
    for (seq_enc, ss_mat, tid) in batch:
        L = seq_enc.shape[0]
        if MAX_SEQ_LEN is not None and L > MAX_SEQ_LEN:
            seq_enc = seq_enc[:MAX_SEQ_LEN]
            ss_mat = ss_mat[:MAX_SEQ_LEN, :MAX_SEQ_LEN]
            L = MAX_SEQ_LEN
        seq_pad = F.pad(seq_enc, (0, max_len - L), value=0)
        ss_pad = F.pad(torch.from_numpy(ss_mat), (0, max_len - L, 0, max_len - L), value=0.0)
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:L] = True
        seq_list.append(seq_pad)
        ss_list.append(ss_pad)
        mask_list.append(mask)
        tids.append(tid)
    seq_tensor = torch.stack(seq_list, dim=0)
    ss_tensor = torch.stack(ss_list, dim=0)
    mask_tensor = torch.stack(mask_list, dim=0)
    return seq_tensor, ss_tensor, mask_tensor, tids



class RelativeSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_relative_position=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.max_rel = max_relative_position
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * max_relative_position - 1, self.head_dim))
        self.ss_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, ss_mat: torch.Tensor = None, mask: torch.Tensor = None):
        B, L, _ = x.size()
        Q = self.q_proj(x).view(B, self.num_heads, L, self.head_dim)
        K = self.k_proj(x).view(B, self.num_heads, L, self.head_dim)
        V = self.v_proj(x).view(B, self.num_heads, L, self.head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        idx = torch.arange(L, device=x.device)
        rel_idx = (idx[None, :] - idx[:, None]).clamp(-self.max_rel+1, self.max_rel-1) + (self.max_rel-1)
        rel_emb = self.rel_pos_emb[rel_idx.long()]
        scores = scores + torch.einsum('bnhd,ljd->bnhl', Q, rel_emb)

        if ss_mat is not None:
            scores = scores + self.ss_weight * ss_mat[:, None, :, :]

        if mask is not None:
            attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))

        weights = torch.softmax(scores, dim=-1)

        if mask is not None:
            query_mask = mask.unsqueeze(1).unsqueeze(-1)
            weights = weights.masked_fill(~query_mask, 0.0)

        out = torch.matmul(weights, V).transpose(1, 2).reshape(B, L, self.embed_dim)
        out = self.out_proj(out)

        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)

        return out

class RNA3DPredictor(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed_dim = 64
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
        self.attn = RelativeSelfAttention(self.embed_dim, num_heads=8)
        self.mlp = nn.Sequential(nn.Linear(self.embed_dim, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, seq: torch.Tensor, ss_mat: torch.Tensor = None, mask: torch.Tensor = None):
        x = self.embed(seq)
        x = self.attn(x, ss_mat, mask)
        x = self.mlp(x)
        return x



def extract_base_id(full_id: str) -> str:
    return "_".join(full_id.split("_")[:2])  # e.g. "1SCL_A_1" -> "1SCL_A"

def drop_incomplete_sequences(lbl_df, seq_df):
    drop_ids = set()
    seq_map = dict(zip(seq_df["target_id"], seq_df["sequence"]))
    for tid, group in lbl_df.groupby("target_id"):
        if group[["x_1", "y_1", "z_1"]].isna().any().any():
            drop_ids.add(tid)
            continue
        if tid not in seq_map:
            drop_ids.add(tid)
            continue
        expected_len = len(seq_map[tid])
        sorted_resid = group["resid"].astype(int).sort_values().values
        if not np.array_equal(sorted_resid, np.arange(1, expected_len + 1)):
            drop_ids.add(tid)
    return drop_ids

base = '/kaggle/input/stanford-rna-3d-folding'
train_seq = pd.read_csv(f'{base}/train_sequences.csv')
train_lbl = pd.read_csv(f'{base}/train_labels.csv')

train_lbl["target_id"] = train_lbl["ID"].apply(extract_base_id)
# If needed:
# train_seq["target_id"] = train_seq["target_id"].apply(extract_base_id)

drop_ids = drop_incomplete_sequences(train_lbl, train_seq)
train_lbl_clean = train_lbl[~train_lbl["target_id"].isin(drop_ids)].copy()
train_seq_clean = train_seq[~train_seq["target_id"].isin(drop_ids)].copy()

valid_ids = set(train_lbl_clean["target_id"]) & set(train_seq_clean["target_id"])
train_lbl_clean = train_lbl_clean[train_lbl_clean["target_id"].isin(valid_ids)].copy()
train_seq_clean = train_seq_clean[train_seq_clean["target_id"].isin(valid_ids)].copy()

print("Remaining targets in train_lbl_clean:", train_lbl_clean["target_id"].nunique())
print("Remaining targets in train_seq_clean:", train_seq_clean["target_id"].nunique())

# Set the index once, on the cleaned DataFrames
train_seq_clean = train_seq_clean.set_index("target_id")
train_lbl_clean = train_lbl_clean.set_index("target_id")

train_dset = RNADataset(train_seq_clean, train_lbl_clean)
train_loader = DataLoader(train_dset, batch_size=4, shuffle=True, collate_fn=custom_collate)
print(f"Train dataset size: {len(train_dset)} sequences.")


# Load raw validation data
val_seq = pd.read_csv(f'{base}/validation_sequences.csv')
val_lbl = pd.read_csv(f'{base}/validation_labels.csv')

# Normalize IDs
val_lbl["target_id"] = val_lbl["ID"].apply(extract_base_id)
val_seq["target_id"] = val_seq["target_id"].apply(extract_base_id)

# Drop incomplete/mismatched entries
drop_ids = drop_incomplete_sequences(val_lbl, val_seq)
val_lbl_clean = val_lbl[~val_lbl["target_id"].isin(drop_ids)].copy()
val_seq_clean = val_seq[~val_seq["target_id"].isin(drop_ids)].copy()

# Keep only IDs present in both
common = set(val_lbl_clean["target_id"]) & set(val_seq_clean["target_id"])
val_lbl_clean = val_lbl_clean[val_lbl_clean["target_id"].isin(common)].set_index("target_id")
val_seq_clean = val_seq_clean[val_seq_clean["target_id"].isin(common)].set_index("target_id")

print("Cleaned validation targets:", len(common))

# Create DataLoader
val_dset = RNADataset(val_seq_clean, val_lbl_clean)
val_loader = DataLoader(val_dset, batch_size=4, shuffle=False, collate_fn=custom_collate)
print("Validation loader batches:", len(val_loader))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RNA3DPredictor(len(vocab)).to(device)
opt = optim.Adam(model.parameters(), lr=1e-5)
relative_loss_fn = nn.MSELoss()

# ——— Insert debug here ———
loader = DataLoader(train_dset, batch_size=4, shuffle=False, collate_fn=custom_collate)
seq_batch, coords_batch, ss_batch, mask_batch, _ = next(iter(loader))
seq_batch, coords_batch, ss_batch, mask_batch = [t.to(device) for t in (seq_batch, coords_batch, ss_batch, mask_batch)]

model.eval()
with torch.no_grad():
    pred = model(seq_batch, ss_batch, mask_batch)

print("Any NaNs in coords_batch? ", torch.isnan(coords_batch).any().item())
print("Any NaNs in ss_batch?    ", torch.isnan(ss_batch).any().item())
print("Any NaNs in model output?", torch.isnan(pred).any().item())

mask_flat = mask_batch.view(-1)
pred_masked = pred[mask_batch]
coords_masked = coords_batch[mask_batch]
dist_pred = pairwise_dist(pred_masked).unsqueeze(0)
dist_ref = pairwise_dist(coords_masked).unsqueeze(0)
loss = relative_loss_fn(dist_pred, dist_ref)
print("Loss is NaN? ", torch.isnan(loss).item())

n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    for seq_batch, coords_batch, ss_batch, mask_batch, tids in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        seq_batch = seq_batch.to(device)
        coords_batch = coords_batch.to(device)
        ss_batch = ss_batch.to(device)
        mask_batch = mask_batch.to(device)
        pred_coords = model(seq_batch, ss_batch, mask_batch)
        batch_loss = 0.0
        for b_idx in range(seq_batch.size(0)):
            L_b = mask_batch[b_idx].sum().item()
            pred_b = pred_coords[b_idx, :L_b, :]
            coords_b = coords_batch[b_idx, :L_b, :]
            dist_pred = pairwise_dist(pred_b).unsqueeze(0)
            dist_ref = pairwise_dist(coords_b).unsqueeze(0)
            sample_loss = relative_loss_fn(dist_pred, dist_ref)
            batch_loss += sample_loss
        batch_loss = batch_loss / seq_batch.size(0)
        opt.zero_grad()
        batch_loss.backward()
        opt.step()
        total_loss += batch_loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{n_epochs}] Mean pairwise-dist MSE: {avg_loss:.4f}")

model.eval()

# Re-create the test DataLoader (in case it was deleted)
test_seq = pd.read_csv(f'{base}/test_sequences.csv')
test_seq["target_id"] = test_seq["target_id"].apply(extract_base_id)
test_seq = test_seq.set_index("target_id")

test_dset = RNATestDataset(test_seq)
test_loader = DataLoader(test_dset, batch_size=4, shuffle=False, collate_fn=test_collate)

# Build submission with required columns
columns = [
    "ID","resname","resid",
    "x_1","y_1","z_1",
    "x_2","y_2","z_2",
    "x_3","y_3","z_3",
    "x_4","y_4","z_4",
    "x_5","y_5","z_5",
]

results = []
model.eval()

for seq_batch, ss_batch, mask_batch, tids in test_loader:
    seq_batch = seq_batch.to(device)
    ss_batch = ss_batch.to(device)
    mask_batch = mask_batch.to(device)

    with torch.no_grad():
        preds = model(seq_batch, ss_batch, mask_batch)

    for b, tid in enumerate(tids):
        seq_str = test_seq.loc[tid, "sequence"]
        L = mask_batch[b].sum().item()
        for i in range(L):
            x, y, z = preds[b, i].cpu().numpy()
            results.append({
                "ID": f"{tid}_{i+1}",
                "resname": seq_str[i],
                "resid": i+1,
                "x_1": x, "y_1": y, "z_1": z,
                **{f"x_{j}": 0.0 for j in range(2,6)},
                **{f"y_{j}": 0.0 for j in range(2,6)},
                **{f"z_{j}": 0.0 for j in range(2,6)},
            })

submission = pd.DataFrame(results, columns=columns)
submission.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv with {len(submission)} rows")


print(submission)


import numpy as np

def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    A_cent = A - A.mean(axis=0)
    B_cent = B - B.mean(axis=0)
    H = B_cent.T @ A_cent
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    B_rot = B_cent @ R
    diff = A_cent - B_rot
    return np.sqrt((diff**2).sum() / A.shape[0])

# === Clean validation DataFrames ===
val_seq = pd.read_csv(f'{base}/validation_sequences.csv')
val_lbl = pd.read_csv(f'{base}/validation_labels.csv')

val_lbl["target_id"] = val_lbl["ID"].apply(lambda x: x.rsplit("_",1)[0])
val_seq["target_id"] = val_seq["target_id"]

common = set(val_lbl["target_id"]) & set(val_seq["target_id"])
val_lbl_clean = val_lbl[val_lbl["target_id"].isin(common)].set_index("target_id")
val_seq_clean = val_seq[val_seq["target_id"].isin(common)].set_index("target_id")

sentinel = -1e18
bad_ids = val_lbl_clean[val_lbl_clean[['x_1','y_1','z_1']].eq(sentinel).any(axis=1)].index.unique()
val_lbl_clean = val_lbl_clean.drop(index=bad_ids)
val_seq_clean = val_seq_clean.drop(index=bad_ids)

print("Final validation targets:", len(val_lbl_clean.index.unique()))

val_dset = RNADataset(val_seq_clean, val_lbl_clean)
val_loader = DataLoader(val_dset, batch_size=4, shuffle=False, collate_fn=custom_collate)
print("Validation loader batches:", len(val_loader))

# === RMSD Validation ===
model.eval()
rmsds = []
for seq_batch, coords_batch, ss_batch, mask_batch, tids in val_loader:
    seq_batch, ss_batch, mask_batch = seq_batch.to(device), ss_batch.to(device), mask_batch.to(device)
    with torch.no_grad():
        preds = model(seq_batch, ss_batch, mask_batch)
    for i, tid in enumerate(tids):
        L = mask_batch[i].sum().item()
        ref = val_lbl_clean.loc[tid].sort_values('resid')[['x_1','y_1','z_1']].values
        pred = preds[i, :L].cpu().numpy()
        rmsds.append(kabsch_rmsd(ref, pred))

if rmsds:
    print(f"Validation mean RMSD: {np.mean(rmsds):.4f} Å")
else:
    print("No validation examples left after cleaning.")


