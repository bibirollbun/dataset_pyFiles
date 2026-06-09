import os, glob, random, warnings, joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# --- SPEED SETTINGS (Cleaned Syntax) ---
SEED = 42
MAX_SEQS_PER_BAG = 500  # Turbo speed: Focuses on top sequences for efficiency
BATCH_SIZE = 12         # Optimized for Kaggle P100/T4 GPUs
EPOCHS = 15             
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# --- Path Configuration ---
BASE_DIR = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
TRAIN_DIR = os.path.join(BASE_DIR, "train_datasets")
TEST_DIR = os.path.join(BASE_DIR, "test_datasets")

# Finding metadata dynamically and safely
meta_search = glob.glob(os.path.join(BASE_DIR, "**/metadata.csv"), recursive=True)
METADATA_PATH = meta_search[0] if meta_search else None

if not METADATA_PATH:
    print("âš ï¸� Warning: metadata.csv not found. Check your dataset attachment.")

# --- FAST GENE MAPPING (200-file sample) ---
v_set, j_set = set(), set()
all_train_files = glob.glob(os.path.join(TRAIN_DIR, "**/*.tsv"), recursive=True)

# Ensure we don't try to sample more files than available
sample_size = min(len(all_train_files), 200)
sample_files = random.sample(all_train_files, sample_size)

for f in tqdm(sample_files, desc="Building Gene Vocabulary"):
    try:
        # Optimization: Only read the 2 necessary columns
        d = pd.read_csv(f, sep='\t', usecols=['v_call', 'j_call']).dropna()
        # Clean gene names (remove alleles after '*')
        v_set.update(d['v_call'].str.split('*').str[0])
        j_set.update(d['j_call'].str.split('*').str[0])
    except Exception as e:
        continue

# Create mappings (1-based indexing for Embedding padding)
V_MAP = {v: i+1 for i, v in enumerate(sorted(list(v_set)))}
J_MAP = {j: i+1 for i, j in enumerate(sorted(list(j_set)))}

print(f"\nâœ… Environment Ready on: {DEVICE}")
print(f"âœ… Vocab Size: V-Genes={len(V_MAP)}, J-Genes={len(J_MAP)}")


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}

class FastAIRRDataset(Dataset):
    def __init__(self, rep_ids, paths, labels=None):
        self.rep_ids = rep_ids
        self.paths = paths
        self.labels = labels

    def __len__(self): return len(self.rep_ids)

    def __getitem__(self, idx):
        rid = self.rep_ids[idx]
        df = pd.read_csv(self.paths[rid], sep='\t', usecols=['junction_aa', 'v_call', 'j_call']).dropna(subset=['junction_aa'])
        if len(df) > MAX_SEQS_PER_BAG: df = df.sample(n=MAX_SEQS_PER_BAG, random_state=SEED)
        
        seqs = torch.tensor([[AA_TO_INT.get(aa, 0) for aa in str(s)[:30]] + [0]*(30-len(str(s)[:30])) for s in df['junction_aa']], dtype=torch.long)
        v_ids = torch.tensor([V_MAP.get(str(v).split('*')[0], 0) for v in df['v_call']], dtype=torch.long)
        j_ids = torch.tensor([J_MAP.get(str(j).split('*')[0], 0) for j in df['j_call']], dtype=torch.long)
        label = torch.tensor(self.labels[rid], dtype=torch.float) if self.labels else torch.tensor(0.0)
        return seqs, v_ids, j_ids, label, rid, df

def fast_collate(batch):
    seqs, v, j, lbls, rids, dfs = zip(*batch)
    indices = []
    curr = 0
    for s in seqs:
        indices.append((curr, curr + len(s)))
        curr += len(s)
    return torch.cat(seqs), torch.cat(v), torch.cat(j), torch.stack(lbls), rids, dfs, indices

class ProAttentionMIL(nn.Module):
    def __init__(self, v_size, j_size):
        super().__init__()
        self.emb = nn.Embedding(len(AA_VOCAB)+1, 32, padding_idx=0)
        self.gru = nn.GRU(32, 64, batch_first=True) 
        self.v_emb = nn.Embedding(v_size+1, 16); self.j_emb = nn.Embedding(j_size+1, 16)
        
        # Gated Attention Mechanism
        self.attention = nn.Sequential(nn.Linear(64+32, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Linear(64+32, 1)

    def forward(self, seqs, v, j, indices):
        _, h = self.gru(self.emb(seqs))
        combined = torch.cat([h[-1], self.v_emb(v), self.j_emb(j)], dim=1)
        
        bag_reps = []
        for s, e in indices:
            bag_data = combined[s:e]
            weights = torch.softmax(self.attention(bag_data), dim=0)
            bag_reps.append(torch.sum(weights * bag_data, dim=0))
            
        return self.classifier(torch.stack(bag_reps)).squeeze(1)


# 1. Synchronize Paths and Metadata
train_paths_raw = {os.path.basename(f).replace('.tsv', ''): f for f in all_train_files}
meta_df = pd.read_csv(METADATA_PATH)
label_map = dict(zip(meta_df['repertoire_id'].astype(str), meta_df['label_positive']))

# Filter IDs
valid_ids = [rid for rid in train_paths_raw.keys() if rid in label_map]
train_paths = {rid: train_paths_raw[rid] for rid in valid_ids}

print(f"ğŸ“Š Labeled Repertoires: {len(valid_ids)} (out of {len(train_paths_raw)} files)")

# 2. Split IDs
train_ids, val_ids = train_test_split(valid_ids, test_size=0.1, random_state=SEED)

# 3. Data Loaders
train_loader = DataLoader(FastAIRRDataset(train_ids, train_paths, label_map), 
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=fast_collate)

# 4. Model & Optimizer
model = ProAttentionMIL(len(V_MAP), len(J_MAP)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

# 5. Device-Aware Scaler & Training Logic
use_cuda = DEVICE.type == 'cuda'
# Fixed: GradScaler needs a string for the device type, usually 'cuda'
scaler = torch.amp.GradScaler('cuda', enabled=use_cuda) 

# Fixed: Use str(DEVICE).upper() to avoid AttributeError
print(f"ğŸ”¥ Training Started on {str(DEVICE).upper()}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for seqs, v, j, lbls, _, _, idxs in pbar:
        optimizer.zero_grad()
        
        # Fixed: Modern Autocast syntax
        with torch.amp.autocast(device_type=DEVICE.type, enabled=use_cuda):
            out = model(seqs.to(DEVICE), v.to(DEVICE), j.to(DEVICE), idxs)
            loss = criterion(out, lbls.to(DEVICE))
        
        if use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward for CPU
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss/len(pbar))

torch.save(model.state_dict(), "best_model.pt")
print("âœ… Training Complete. Model Saved.")


model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Task 1 Inference
test_files = glob.glob(os.path.join(TEST_DIR, "**/*.tsv"), recursive=True)
test_paths = {os.path.basename(f).replace('.tsv', ''): f for f in test_files}
test_loader = DataLoader(FastAIRRDataset(list(test_paths.keys()), test_paths), batch_size=1, collate_fn=fast_collate)

predictions = []
with torch.no_grad():
    for seqs, v, j, _, rids, _, idxs in tqdm(test_loader, desc="Inference"):
        prob = torch.sigmoid(model(seqs.to(DEVICE), v.to(DEVICE), j.to(DEVICE), idxs))
        predictions.append({'repertoire_id': rids[0], 'probability': prob.item()})

df_task1 = pd.DataFrame(predictions)

# Align with competition format
for col in ['dataset_id', 'junction_aa', 'v_call', 'j_call', 'rank']:
    df_task1[col] = -999.0

df_task1.to_csv("submission.csv", index=False)
print("ğŸ�† Submission file saved successfully!")

