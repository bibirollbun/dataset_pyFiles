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


# ============================================================
# ACCELEROMETER-ONLY TRANSFORMER
# Single-file Kaggle version with analysis & plots
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"  # <<< CHANGE
BATCH_SIZE = 32
SEQ_LEN = 150
EPOCHS = 10
LR = 3e-4
TOP_K = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ============================================================
# DATASET
# ============================================================

class AccDataset(Dataset):
    ACC_COLS = ["acc_x", "acc_y", "acc_z"]

    def __init__(self, csv_path, seq_len=150):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len

        # Encode gestures
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_

        # Group by movement (= sequence)
        self.sequences = list(self.df.groupby("sequence_id"))
        print(f"Loaded {len(self.sequences)} movements")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _, seq = self.sequences[idx]

        acc = seq[self.ACC_COLS].values.astype(np.float32)

        # Normalize per sequence
        acc = (acc - acc.mean(axis=0)) / (acc.std(axis=0) + 1e-6)

        L = min(len(acc), self.seq_len)
        acc = acc[:L]

        # Pad
        pad = np.zeros((self.seq_len, 3), dtype=np.float32)
        pad[:L] = acc

        mask = np.zeros(self.seq_len, dtype=np.bool_)
        mask[:L] = 1

        label = seq["gesture_id"].mode().iloc[0]

        return {
            "acc": torch.tensor(pad),
            "mask": torch.tensor(mask),
            "label": torch.tensor(label)
        }

# ============================================================
# MODEL
# ============================================================

class AccTransformer(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, layers=4):
        super().__init__()

        self.input_proj = nn.Linear(3, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask):
        x = self.input_proj(x) + self.pos_emb
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = x.mean(dim=1)
        return self.cls_head(x)

# ============================================================
# LOAD DATA
# ============================================================

dataset = AccDataset(CSV_PATH, SEQ_LEN)
num_classes = len(dataset.gesture_classes)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ============================================================
# TRAIN
# ============================================================

model = AccTransformer(num_classes).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        acc = batch["acc"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(acc, mask)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

# ============================================================
# EVALUATION
# ============================================================

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        acc = batch["acc"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        logits = model(acc, mask)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ============================================================
# METRICS
# ============================================================

overall_acc = (all_preds == all_labels).mean()
print(f"\nOverall accuracy (accelerometer only): {overall_acc:.3f}")

# ---- Per-gesture accuracy
gesture_acc = {}
gesture_counts = {}

for g in range(num_classes):
    idx = all_labels == g
    if idx.sum() == 0:
        continue
    gesture_acc[g] = (all_preds[idx] == g).mean()
    gesture_counts[g] = idx.sum()

# ---- Ranking
ranking = sorted(
    gesture_acc.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nGesture ranking (accelerometer detectability):")
print("-" * 50)
for g, acc in ranking:
    print(f"{dataset.gesture_classes[g]:20s} | Acc: {acc:.3f} | N={gesture_counts[g]}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

conf = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_labels, all_preds):
    conf[t, p] += 1

plt.figure(figsize=(12, 10))
sns.heatmap(
    conf,
    xticklabels=dataset.gesture_classes,
    yticklabels=dataset.gesture_classes,
    cmap="Blues",
    fmt="d"
)
plt.title("Confusion Matrix (Accelerometer Only)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ============================================================
# TOP-K ACCURACY
# ============================================================

correct_topk = 0

with torch.no_grad():
    for batch in test_loader:
        acc = batch["acc"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        logits = model(acc, mask)
        topk = logits.topk(TOP_K, dim=1).indices

        for i in range(label.size(0)):
            if label[i] in topk[i]:
                correct_topk += 1

topk_acc = correct_topk / len(all_labels)
print(f"\nTop-{TOP_K} accuracy: {topk_acc:.3f}")



# ============================================================
# ROTATION-ONLY TRANSFORMER
# Single-file Kaggle version with analysis & plots
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
BATCH_SIZE = 32
SEQ_LEN = 150
EPOCHS = 10
LR = 1e-4  # Reduced from 3e-4
TOP_K = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# DATASET
# ============================================================

class RotDataset(Dataset):
    ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]

    def __init__(self, csv_path, seq_len=150):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len

        # Encode gestures
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_

        # Group by movement (= sequence)
        self.sequences = list(self.df.groupby("sequence_id"))
        print(f"Loaded {len(self.sequences)} movements")
        print(f"Number of gesture classes: {len(self.gesture_classes)}")
        print(f"Gesture classes: {self.gesture_classes}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _, seq = self.sequences[idx]

        rot = seq[self.ROT_COLS].values.astype(np.float32)
        
        # Check for NaN values and replace with zeros
        if np.any(np.isnan(rot)):
            rot = np.nan_to_num(rot)

        # Normalize quaternions to unit length per row
        norms = np.linalg.norm(rot, axis=1, keepdims=True)
        # Handle zero norms - set to 1 to avoid division by zero
        norms[norms == 0] = 1.0
        rot = rot / norms

        L = min(len(rot), self.seq_len)
        rot = rot[:L]

        # Pad
        pad = np.zeros((self.seq_len, 4), dtype=np.float32)
        pad[:L] = rot

        # Create mask: 1 for real data, 0 for padding
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mask[:L] = 1

        label = seq["gesture_id"].mode().iloc[0]

        return {
            "rot": torch.tensor(pad, dtype=torch.float32),
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# MODEL with improved stability
# ============================================================

class RotTransformer(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, layers=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection with better initialization
        self.input_proj = nn.Linear(4, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # Positional embedding with smaller initialization
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.01)
        
        # Add layer normalization before transformer
        self.ln1 = nn.LayerNorm(d_model)
        
        # Transformer encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # More stable than relu
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head with more layers and better normalization
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize classification head properly
        for layer in self.cls_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, mask):
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_emb
        
        # Layer norm
        x = self.ln1(x)
        
        # Pass through transformer
        # Note: src_key_padding_mask expects True for padded positions
        x = self.encoder(x, src_key_padding_mask=~mask)
        
        # Mean pooling over sequence dimension (considering only non-padded positions)
        # Create attention mask for pooling
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        x_masked = x * mask_expanded
        sum_pool = torch.sum(x_masked, dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        x = sum_pool / count
        
        # Classification
        return self.cls_head(x)

# ============================================================
# LOAD DATA
# ============================================================

dataset = RotDataset(CSV_PATH, SEQ_LEN)
num_classes = len(dataset.gesture_classes)
print(f"\nNumber of classes: {num_classes}")

# Check class distribution
class_counts = dataset.df.groupby("gesture_id").size()
print("\nClass distribution:")
for gid, count in class_counts.items():
    print(f"  Class {gid} ({dataset.gesture_classes[gid]}): {count} samples")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ============================================================
# TRAINING with gradient clipping and learning rate warmup
# ============================================================

model = RotTransformer(num_classes, dropout=0.1).to(DEVICE)

# Use AdamW with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LR,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS * len(train_loader),
    eta_min=LR * 0.01
)

criterion = nn.CrossEntropyLoss()

# Track training metrics
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        rot = batch["rot"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(rot, mask)
        
        # Add label smoothing for stability
        loss = criterion(logits, label)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"WARNING: NaN loss at batch {batch_idx}")
            continue
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)
        
        # Print progress
        if batch_idx % 20 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
    
    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(avg_loss)
    
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

# ============================================================
# EVALUATION
# ============================================================

model.eval()
all_preds, all_labels = [], []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        rot = batch["rot"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        labels = batch["label"].numpy()

        logits = model(rot, mask)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels)
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ============================================================
# METRICS
# ============================================================

overall_acc = (all_preds == all_labels).mean()
print(f"\nOverall accuracy (rotation only): {overall_acc:.3f}")

# ---- Per-gesture accuracy
gesture_acc = {}
gesture_counts = {}

for g in range(num_classes):
    idx = all_labels == g
    if idx.sum() == 0:
        continue
    gesture_acc[g] = (all_preds[idx] == g).mean()
    gesture_counts[g] = idx.sum()

# ---- Ranking
ranking = sorted(
    gesture_acc.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nGesture ranking (rotation detectability):")
print("-" * 50)
for g, acc in ranking:
    print(f"{dataset.gesture_classes[g]:20s} | Acc: {acc:.3f} | N={gesture_counts[g]}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

conf = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_labels, all_preds):
    conf[t, p] += 1

plt.figure(figsize=(12, 10))
sns.heatmap(
    conf,
    xticklabels=dataset.gesture_classes,
    yticklabels=dataset.gesture_classes,
    cmap="Blues",
    fmt="d",
    annot=True,
    annot_kws={'size': 8}
)
plt.title(f"Confusion Matrix (Rotation Only) - Overall Acc: {overall_acc:.3f}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ============================================================
# TOP-K ACCURACY
# ============================================================

correct_topk = 0

for i in range(len(all_labels)):
    label = all_labels[i]
    prob = all_probs[i]
    top_k_indices = np.argsort(prob)[-TOP_K:][::-1]
    if label in top_k_indices:
        correct_topk += 1

topk_acc = correct_topk / len(all_labels)
print(f"\nTop-{TOP_K} accuracy: {topk_acc:.3f}")

# ============================================================
# ADDITIONAL DIAGNOSTICS
# ============================================================

# Plot training loss
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()

# Calculate per-class metrics
from sklearn.metrics import classification_report

print("\n" + "="*50)
print("Classification Report:")
print("="*50)
print(classification_report(
    all_labels, 
    all_preds, 
    target_names=dataset.gesture_classes,
    digits=3
))


# ============================================================
# ACCELEROMETER-ONLY TRANSFORMER
# Single-file Kaggle version with analysis & plots
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"  # <<< CHANGE
BATCH_SIZE = 32
SEQ_LEN = 21
EPOCHS = 10
LR = 3e-4
TOP_K = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ============================================================
# DATASET
# ============================================================

class RotDataset(Dataset):
    ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]

    def __init__(self, csv_path, seq_len=150):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len

        # Encode gestures
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_

        # Group by movement (= sequence)
        self.sequences = list(self.df.groupby("sequence_id"))
        print(f"Loaded {len(self.sequences)} movements")
        print(f"Number of gesture classes: {len(self.gesture_classes)}")
        print(f"Gesture classes: {self.gesture_classes}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _, seq = self.sequences[idx]

        rot = seq[self.ROT_COLS].values.astype(np.float32)
        
        # Check for NaN values and replace with zeros
        if np.any(np.isnan(rot)):
            rot = np.nan_to_num(rot)

        # Normalize quaternions to unit length per row
        norms = np.linalg.norm(rot, axis=1, keepdims=True)
        # Handle zero norms - set to 1 to avoid division by zero
        norms[norms == 0] = 1.0
        rot = rot / norms

        L = min(len(rot), self.seq_len)
        rot = rot[:L]

        # Pad
        pad = np.zeros((self.seq_len, 4), dtype=np.float32)
        pad[:L] = rot

        # Create mask: 1 for real data, 0 for padding
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mask[:L] = 1

        label = seq["gesture_id"].mode().iloc[0]

        return {
            "rot": torch.tensor(pad, dtype=torch.float32),
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# MODEL with improved stability
# ============================================================

class RotTransformer(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, layers=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection with better initialization
        self.input_proj = nn.Linear(4, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # Positional embedding with smaller initialization
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.01)
        
        # Add layer normalization before transformer
        self.ln1 = nn.LayerNorm(d_model)
        
        # Transformer encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # More stable than relu
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head with more layers and better normalization
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize classification head properly
        for layer in self.cls_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, mask):
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_emb
        
        # Layer norm
        x = self.ln1(x)
        
        # Pass through transformer
        # Note: src_key_padding_mask expects True for padded positions
        x = self.encoder(x, src_key_padding_mask=~mask)
        
        # Mean pooling over sequence dimension (considering only non-padded positions)
        # Create attention mask for pooling
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        x_masked = x * mask_expanded
        sum_pool = torch.sum(x_masked, dim=1)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        x = sum_pool / count
        
        # Classification
        return self.cls_head(x)
# ============================================================
class AccDataset(Dataset):
    ACC_COLS = ["acc_x", "acc_y", "acc_z"]

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Encode gestures
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_

        # Group by movement (= sequence)
        self.sequences = list(self.df.groupby("sequence_id"))
        print(f"Loaded {len(self.sequences)} movements")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        _, seq = self.sequences[idx]

        acc = seq[self.ACC_COLS].values.astype(np.float32)

        # Normalize per sequence
        acc = (acc - acc.mean(axis=0)) / (acc.std(axis=0) + 1e-6)

        label = seq["gesture_id"].mode().iloc[0]

        return {"acc": torch.tensor(acc), "length": len(acc), "label": torch.tensor(label)}


# ============================================================
# COLLATE FUNCTION FOR DYNAMIC PADDING
# ============================================================

def collate_acc_fn(batch):
    batch_size = len(batch)
    lengths = [b["length"] for b in batch]
    max_len = max(lengths)

    accs = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["length"]
        accs[i, :L] = b["acc"]
        masks[i, :L] = 1
        labels[i] = b["label"]

    return {"acc": accs, "mask": masks, "label": labels}


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class AccTransformer(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, layers=4):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layers)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask):
        # x: (batch, seq_len, 3)
        # mask: (batch, seq_len)
        batch_size, seq_len, _ = x.size()

        # Dynamic positional embeddings
        pos_emb = torch.randn(1, seq_len, self.input_proj.out_features, device=x.device)
        x = self.input_proj(x) + pos_emb

        x = self.encoder(x, src_key_padding_mask=~mask)
        x = x.mean(dim=1)
        return self.cls_head(x)
    
class AccWindowDataset(Dataset):
    ACC_COLS = ["acc_x", "acc_y", "acc_z"]

    def __init__(self, csv_path, window_before=10, window_after=10):
        self.df = pd.read_csv(csv_path)
        self.window_before = window_before
        self.window_after = window_after

        # Encode gestures
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_

        # Group by movement (= sequence)
        self.sequences = list(self.df.groupby("sequence_id"))

        self.windows = []
        for _, seq in self.sequences:
            gesture_rows = seq[seq["phase"] == "Gesture"]
            if gesture_rows.empty:
                center_pos = 0  # fallback
            else:
                gesture_pos = gesture_rows.index[0]
                center_pos = seq.index.get_loc(gesture_pos)           # convert to position 0..len-1

            start = max(0, center_pos - WINDOW_BEFORE)
            end   = min(len(seq), center_pos + WINDOW_AFTER)

            window_data = seq.iloc[start:end][self.ACC_COLS].values.astype(np.float32)
            if window_data.shape[0] == 0:
                continue  # skip empty windows

            label = seq["gesture_id"].mode().iloc[0]
            self.windows.append((window_data, label))

        print(f"Created {len(self.windows)} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data, label = self.windows[idx]
        L = data.shape[0]
        # Pad if needed
        pad_len = self.window_before + self.window_after + 1
        if L < pad_len:
            pad = np.zeros((pad_len, 3), dtype=np.float32)
            pad[:L] = data
            mask = np.zeros(pad_len, dtype=np.bool_)
            mask[:L] = 1
            data = pad
        else:
            mask = np.ones(L, dtype=np.bool_)
        return {
            "acc": torch.tensor(data, dtype=torch.float32),
            "mask": torch.tensor(mask),
            "label": torch.tensor(label)
        }

# ============================================================
# MODEL
# ============================================================

class AccWindowTransformer(nn.Module):
    def __init__(self, num_classes, window_len, d_model=64, nhead=4, layers=4):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, window_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, mask):
        x = self.input_proj(x) + self.pos_emb
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = x.mean(dim=1)
        return self.cls_head(x)

class RotWindowDataset(Dataset):
    ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]

    def __init__(self, csv_path, window_before=10, window_after=10):
        self.df = pd.read_csv(csv_path)
        self.window_before = window_before
        self.window_after = window_after
        self.window_len = window_before + window_after + 1

        # Encode gestures
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_

        # Group by movement (= sequence)
        self.sequences = list(self.df.groupby("sequence_id"))

        self.windows = []
        for _, seq in self.sequences:
            seq = seq.reset_index(drop=True)

            gesture_rows = seq[seq["phase"] == "Gesture"]
            if gesture_rows.empty:
                continue  # skip invalid sequences

            center_pos = gesture_rows.index[0]

            start = center_pos - self.window_before
            end = center_pos + self.window_after + 1

            start_clamped = max(0, start)
            end_clamped = min(len(seq), end)

            rot = seq.iloc[start_clamped:end_clamped][self.ROT_COLS].values.astype(np.float32)

            if rot.shape[0] == 0:
                continue

            # ---- Handle NaNs
            rot = np.nan_to_num(rot)

            # ---- Normalize quaternions (row-wise)
            norms = np.linalg.norm(rot, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            rot = rot / norms

            # ---- Pad to fixed window
            padded = np.zeros((self.window_len, 4), dtype=np.float32)
            mask = np.zeros(self.window_len, dtype=np.bool_)

            insert_pos = start_clamped - start
            padded[insert_pos:insert_pos + len(rot)] = rot
            mask[insert_pos:insert_pos + len(rot)] = True

            label = seq["gesture_id"].mode().iloc[0]

            self.windows.append((padded, mask, label))

        print(f"Created {len(self.windows)} rotation windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        rot, mask, label = self.windows[idx]
        return {
            "rot": torch.tensor(rot, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
        }

class RotationWinTransformer(nn.Module):
    def __init__(self, num_classes, window_len, d_model=64, nhead=4, layers=4, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(4, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        self.pos_emb = nn.Parameter(torch.randn(1, window_len, d_model) * 0.01)
        self.ln1 = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        for m in self.cls_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        """
        x: (B, T, 4)
        mask: (B, T)
        """

        x = self.input_proj(x)
        x = x + self.pos_emb
        x = self.ln1(x)

        x = self.encoder(x, src_key_padding_mask=~mask)

        # ---- Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return self.cls_head(x)
# ============================================================
# LOAD DATA
# ============================================================

BATCH_SIZE = 32
WINDOW_BEFORE = 10
WINDOW_AFTER = 10
D_MODEL = 64
NHEAD = 4
LAYERS = 4
EPOCHS = 10
LR = 3e-4
TOP_K = 3
# dataset = AccWindowDataset(CSV_PATH, WINDOW_BEFORE, WINDOW_AFTER)
dataset = RotWindowDataset(CSV_PATH, WINDOW_BEFORE, WINDOW_AFTER)
# dataset = AccDataset(CSV_PATH)
# dataset = RotDataset(CSV_PATH, SEQ_LEN)
num_classes = len(dataset.gesture_classes)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ============================================================
# TRAIN
# ============================================================

model = RotationWinTransformer(num_classes,21).to(DEVICE)
# model = AccWindowTransformer(num_classes,21, D_MODEL, NHEAD, LAYERS).to(DEVICE)
# model = AccTransformer(num_classes).to(DEVICE)
# model = RotTransformer(num_classes).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        # acc = batch["acc"].to(DEVICE)
        rot = batch["rot"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(rot, mask)
        # logits = model(acc, mask)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

# ============================================================
# EVALUATION
# ============================================================

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        rot = batch["rot"].to(DEVICE)
        # acc = batch["acc"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        logits = model(rot, mask)
        # logits = model(acc, mask)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ============================================================
# METRICS
# ============================================================

overall_acc = (all_preds == all_labels).mean()
print(f"\nOverall accuracy (rotation only): {overall_acc:.3f}")
# print(f"\nOverall accuracy (rotation only): {overall_acc:.3f}")

# ---- Per-gesture accuracy
gesture_acc = {}
gesture_counts = {}

for g in range(num_classes):
    idx = all_labels == g
    if idx.sum() == 0:
        continue
    gesture_acc[g] = (all_preds[idx] == g).mean()
    gesture_counts[g] = idx.sum()

# ---- Ranking
ranking = sorted(
    gesture_acc.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nGesture ranking (rotation detectability):")
# print("\nGesture ranking (accelerometer detectability):")
print("-" * 50)
for g, acc in ranking:
    print(f"{dataset.gesture_classes[g]:20s} | Acc: {acc:.3f} | N={gesture_counts[g]}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

conf = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_labels, all_preds):
    conf[t, p] += 1

plt.figure(figsize=(12, 10))
sns.heatmap(
    conf,
    xticklabels=dataset.gesture_classes,
    yticklabels=dataset.gesture_classes,
    cmap="Blues",
    fmt="d"
)
# plt.title("Confusion Matrix (Accelerometer Only)")
plt.title("Confusion Matrix (Rotation Only)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ============================================================
# TOP-K ACCURACY
# ============================================================

correct_topk = 0

with torch.no_grad():
    for batch in test_loader:
        rot = batch["rot"].to(DEVICE)
        # acc = batch["acc"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        logits = model(rot, mask)
        # logits = model(acc, mask)
        topk = logits.topk(TOP_K, dim=1).indices

        for i in range(label.size(0)):
            if label[i] in topk[i]:
                correct_topk += 1

topk_acc = correct_topk / len(all_labels)
print(f"\nTop-{TOP_K} accuracy: {topk_acc:.3f}")



# ============================================================
# IMPROVED FUSED TRANSFORMER FOR SUBTLE GESTURE RECOGNITION
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
BATCH_SIZE = 64
SEQ_LEN = 21  # 10 + 10 + 1
EPOCHS = 30
LR = 2e-4
TOP_K = 3
D_MODEL = 128  # Increased capacity
NHEAD = 8
NLAYERS = 4
DROPOUT = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# ENHANCED DATA HELPERS WITH MORE FEATURES
# ============================================================

def handle_quaternion_missing_values(rot_data: np.ndarray) -> np.ndarray:
    rot_cleaned = rot_data.copy()
    for i in range(len(rot_data)):
        row = rot_data[i]
        missing_count = np.isnan(row).sum()
        if missing_count == 0:
            norm = np.linalg.norm(row)
            rot_cleaned[i] = row / norm if norm > 1e-8 else [1.0, 0.0, 0.0, 0.0]
        elif missing_count == 1:
            missing_idx = np.where(np.isnan(row))[0][0]
            valid_values = row[~np.isnan(row)]
            sum_squares = np.sum(valid_values**2)
            missing_value = np.sqrt(max(0, 1.0 - sum_squares))
            if i > 0 and not np.isnan(rot_cleaned[i-1, missing_idx]):
                if rot_cleaned[i-1, missing_idx] < 0:
                    missing_value = -missing_value
            rot_cleaned[i, missing_idx] = missing_value
            rot_cleaned[i, ~np.isnan(row)] = valid_values
        else:
            rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
    return rot_cleaned

def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Convert acceleration from device to world coordinates"""
    try:
        rot_scipy = rot[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w] for scipy
        norms = np.linalg.norm(rot_scipy, axis=1)
        if np.any(norms < 1e-8):
            mask = norms < 1e-8
            rot_scipy[mask] = [0.0, 0.0, 0.0, 1.0]
        r = R.from_quat(rot_scipy)
        acc_world = r.apply(acc)
        return acc_world
    except Exception as e:
        print(f"Warning: World coordinate transformation failed: {e}")
        return acc.copy()

def compute_angular_velocity(rot_data, time_delta=1/200):
    """Compute angular velocity from quaternions"""
    num_samples = rot_data.shape[0]
    angular_vel = np.zeros((num_samples, 3))
    for i in range(num_samples - 1):
        try:
            r1 = R.from_quat(rot_data[i])
            r2 = R.from_quat(rot_data[i+1])
            delta = r1.inv() * r2
            angular_vel[i] = delta.as_rotvec() / time_delta
        except:
            continue
    return angular_vel

def compute_motion_intensity_features(acc, rot):
    """Compute features that help distinguish subtle vs large motions"""
    # Magnitude features
    acc_mag = np.sqrt(np.sum(acc**2, axis=1))
    acc_mag_smooth = np.convolve(acc_mag, np.ones(5)/5, mode='same')
    
    # Jerk (rate of change of acceleration)
    jerk = np.zeros_like(acc)
    if len(acc) > 1:
        jerk[1:] = np.diff(acc, axis=0) * 200  # 200Hz sampling
    jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
    
    # Angular features
    angular_vel = compute_angular_velocity(rot)
    angular_vel_mag = np.sqrt(np.sum(angular_vel**2, axis=1))
    
    # Motion variability (subtle motions are more periodic)
    motion_variability = np.zeros(len(acc))
    if len(acc) > 10:
        # Compute rolling standard deviation to detect repetitive motions
        window = min(10, len(acc))
        for i in range(len(acc)):
            start = max(0, i - window//2)
            end = min(len(acc), i + window//2)
            if end > start:
                motion_variability[i] = np.std(acc[start:end])
    
    return {
        'acc_mag': acc_mag,
        'acc_mag_smooth': acc_mag_smooth,
        'jerk_mag': jerk_mag,
        'angular_vel_mag': angular_vel_mag,
        'motion_variability': motion_variability
    }

def compute_temporal_features(data, window_size=5):
    """Compute temporal features (moving average) for each column of data"""
    n_samples = len(data)
    n_features = data.shape[1]
    features = np.zeros((n_samples, n_features))  # FIXED: Use n_features instead of 4
    kernel = np.ones(window_size) / window_size
    
    for i in range(n_features):
        features[:, i] = np.convolve(data[:, i], kernel, mode='same')
    
    return features

# ============================================================
# ENHANCED DATASET WITH CLASS-SPECIFIC PROCESSING
# ============================================================

class EnhancedMultimodalDataset(Dataset):
    ACC_COLS = ["acc_x", "acc_y", "acc_z"]
    ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]
    
    def __init__(self, csv_path, window_before=10, window_after=10, mode='train'):
        self.df = pd.read_csv(csv_path)
        self.window_before = window_before
        self.window_after = window_after
        self.window_len = window_before + window_after + 1
        self.mode = mode
        
        # Encode gestures
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_
        
        # Analyze gesture types
        self.gesture_type = {}
        for i, gesture in enumerate(self.gesture_classes):
            gesture_lower = gesture.lower()
            if any(word in gesture_lower for word in ['pinch', 'scratch', 'pull']):
                self.gesture_type[i] = 'subtle'
            elif any(word in gesture_lower for word in ['drink', 'feel', 'glasses', 'text', 'wave', 'write']):
                self.gesture_type[i] = 'large'
            else:
                self.gesture_type[i] = 'other'
        
        self.sequences = list(self.df.groupby("sequence_id"))
        self.windows = []
        
        print(f"Loaded {len(self.sequences)} sequences")
        subtle_count = sum(1 for v in self.gesture_type.values() if v == 'subtle')
        large_count = sum(1 for v in self.gesture_type.values() if v == 'large')
        print(f"Gesture types: Subtle={subtle_count}, Large={large_count}")

    def __len__(self):
        return len(self.windows)

    def process_sequence(self, seq):
        """Process a single sequence with enhanced features"""
        seq = seq.reset_index(drop=True)
        
        # Find gesture phase
        gesture_rows = seq[seq["phase"] == "Gesture"]
        if gesture_rows.empty:
            return None
        
        center = gesture_rows.index[0]
        start = max(0, center - self.window_before)
        end = min(len(seq), center + self.window_after + 1)
        
        # Extract raw data
        acc_raw = seq.iloc[start:end][self.ACC_COLS].values.astype(np.float32)
        rot_raw = seq.iloc[start:end][self.ROT_COLS].values.astype(np.float32)
        
        if len(acc_raw) == 0 or len(rot_raw) == 0:
            return None
        
        # Handle missing values in rotation
        rot_cleaned = handle_quaternion_missing_values(rot_raw)
        
        # Normalize quaternions
        norms = np.linalg.norm(rot_cleaned, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        rot_norm = rot_cleaned / norms
        
        # Compute world acceleration
        acc_world = compute_world_acceleration(acc_raw, rot_norm)
        
        # Compute enhanced features
        motion_features = compute_motion_intensity_features(acc_world, rot_norm)
        
        # Compute angular velocity
        angular_vel = compute_angular_velocity(rot_norm)
        
        # Create feature-rich representations
        # Accelerometer features: [world_x, world_y, world_z, magnitude, jerk_mag, variability]
        acc_features = np.column_stack([
            acc_world,
            motion_features['acc_mag'].reshape(-1, 1),
            motion_features['jerk_mag'].reshape(-1, 1),
            motion_features['motion_variability'].reshape(-1, 1)
        ])  # Shape: (n_samples, 6)
        
        # Rotation features: [quat_w, quat_x, quat_y, quat_z, angular_vel_x, angular_vel_y, angular_vel_z, angular_mag]
        rot_features = np.column_stack([
            rot_norm,
            angular_vel,
            motion_features['angular_vel_mag'].reshape(-1, 1)
        ])  # Shape: (n_samples, 8)
        
        # Add temporal features
        acc_temporal = compute_temporal_features(acc_features)
        rot_temporal = compute_temporal_features(rot_features[:, :4])  # Only on quaternions
        
        acc_features = np.column_stack([acc_features, acc_temporal])  # Now (n_samples, 12)
        rot_features = np.column_stack([rot_features, rot_temporal])  # Now (n_samples, 12)
        
        # Normalize per window
        acc_mean = acc_features.mean(axis=0, keepdims=True)
        acc_std = acc_features.std(axis=0, keepdims=True) + 1e-6
        acc_features = (acc_features - acc_mean) / acc_std
        
        rot_mean = rot_features.mean(axis=0, keepdims=True)
        rot_std = rot_features.std(axis=0, keepdims=True) + 1e-6
        rot_features = (rot_features - rot_mean) / rot_std
        
        # Padding
        actual_len = len(acc_features)
        acc_pad = np.zeros((self.window_len, acc_features.shape[1]), dtype=np.float32)
        rot_pad = np.zeros((self.window_len, rot_features.shape[1]), dtype=np.float32)
        mask = np.zeros(self.window_len, dtype=np.bool_)
        
        insert_start = 0
        acc_pad[insert_start:insert_start + actual_len] = acc_features
        rot_pad[insert_start:insert_start + actual_len] = rot_features
        mask[insert_start:insert_start + actual_len] = True
        
        label = seq["gesture_id"].mode().iloc[0]
        
        # Store gesture type for weighted sampling
        gesture_type = self.gesture_type.get(label, 'other')
        
        return acc_pad, rot_pad, mask, label, gesture_type

    def build_windows(self):
        """Build all windows (can be called after initialization if needed)"""
        self.windows = []
        self.gesture_types = []
        
        for _, seq in self.sequences:
            result = self.process_sequence(seq)
            if result is not None:
                acc_pad, rot_pad, mask, label, gtype = result
                self.windows.append((acc_pad, rot_pad, mask, label))
                self.gesture_types.append(gtype)
        
        print(f"Created {len(self.windows)} windows")
        if self.windows:
            print(f"Accelerometer features shape: {self.windows[0][0].shape}")
            print(f"Rotation features shape: {self.windows[0][1].shape}")

    def __getitem__(self, idx):
        acc, rot, mask, label = self.windows[idx]
        
        # Add modality-specific noise augmentation for training
        if self.mode == 'train':
            # More noise for subtle gestures to improve robustness
            if self.gesture_types[idx] == 'subtle':
                acc_noise = np.random.normal(0, 0.03, acc.shape).astype(np.float32)
                rot_noise = np.random.normal(0, 0.01, rot.shape).astype(np.float32)
            else:
                acc_noise = np.random.normal(0, 0.02, acc.shape).astype(np.float32)
                rot_noise = np.random.normal(0, 0.005, rot.shape).astype(np.float32)
            
            acc = acc + acc_noise
            rot = rot + rot_noise
        
        return {
            "acc": torch.tensor(acc, dtype=torch.float32),
            "rot": torch.tensor(rot, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# SIMPLIFIED BUT EFFECTIVE MODEL
# ============================================================

class ModalityEncoder(nn.Module):
    """Encoder for each modality with attention"""
    def __init__(self, input_dim, d_model=128, nhead=8, dropout=0.3, num_layers=2):
        super().__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # Input projection
        x = self.input_proj(x)
        x = x + self.pos_emb
        x = self.dropout(x)
        x = self.norm(x)
        
        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=~mask)
        
        # Attention pooling
        attn_weights = self.attention_pool(x)
        attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        pooled = torch.sum(attn_weights * x, dim=1)
        
        return pooled

class ImprovedFusedTransformer(nn.Module):
    """Simplified but effective fusion model"""
    def __init__(self, num_classes, acc_input_dim=12, rot_input_dim=12, 
                 d_model=128, dropout=0.3):
        super().__init__()
        
        # Modality encoders
        self.acc_encoder = ModalityEncoder(acc_input_dim, d_model, dropout=dropout)
        self.rot_encoder = ModalityEncoder(rot_input_dim, d_model, dropout=dropout)
        
        # Gated fusion
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, acc, rot, mask):
        # Encode each modality
        acc_features = self.acc_encoder(acc, mask)
        rot_features = self.rot_encoder(rot, mask)
        
        # Learn modality importance
        combined = torch.cat([acc_features, rot_features], dim=1)
        gate_weights = self.gate_network(combined)
        
        # Gated combination
        gated_acc = gate_weights[:, 0:1] * acc_features
        gated_rot = gate_weights[:, 1:2] * rot_features
        
        # Combine
        fused_features = torch.cat([gated_acc, gated_rot], dim=1)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, gate_weights

# ============================================================
# LOAD DATA WITH CLASS BALANCING
# ============================================================

dataset = EnhancedMultimodalDataset(CSV_PATH, window_before=10, window_after=10, mode='train')
dataset.build_windows()  # Build windows after initialization

num_classes = len(dataset.gesture_classes)
print(f"\nNumber of classes: {num_classes}")

if dataset.windows:
    acc_input_dim = dataset.windows[0][0].shape[1]
    rot_input_dim = dataset.windows[0][1].shape[1]
    print(f"Accelerometer input dim: {acc_input_dim}")
    print(f"Rotation input dim: {rot_input_dim}")
else:
    print("No windows created. Check dataset.")
    exit()

# Compute class weights for imbalance
class_counts = defaultdict(int)
for _, _, _, label in dataset.windows:
    class_counts[label] += 1

print(f"\nClass distribution:")
for class_id, count in sorted(class_counts.items()):
    print(f"  Class {class_id} ({dataset.gesture_classes[class_id]}): {count} samples")

class_weights = torch.zeros(num_classes)
for class_id, count in class_counts.items():
    class_weights[class_id] = 1.0 / count
class_weights = class_weights / class_weights.sum()

# Split data
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

# Create weighted sampler for training
train_labels = [dataset.windows[i][3] for i in train_ds.indices]
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(train_labels))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ============================================================
# TRAINING WITH FOCUS ON SUBTLE GESTURES
# ============================================================

model = ImprovedFusedTransformer(
    num_classes=num_classes,
    acc_input_dim=acc_input_dim,
    rot_input_dim=rot_input_dim,
    d_model=D_MODEL,
    dropout=DROPOUT
).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Focal loss for handling class imbalance and hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

# Use focal loss with higher weight for subtle gestures
subtle_indices = [i for i, gtype in dataset.gesture_type.items() if gtype == 'subtle']
alpha = torch.ones(num_classes)
for idx in subtle_indices:
    alpha[idx] = 2.0  # Higher weight for subtle gestures

criterion = FocalLoss(alpha=alpha.to(DEVICE), gamma=2.0)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS * len(train_loader),
    eta_min=LR * 0.01
)

# Training metrics
train_losses = []
val_losses = []
val_accs = []
best_val_acc = 0
patience = 10
patience_counter = 0

print(f"\nStarting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    # Training
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        acc = batch["acc"].to(DEVICE)
        rot = batch["rot"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        logits, _ = model(acc, rot, mask)
        loss = criterion(logits, label)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    subtle_correct = 0
    subtle_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            acc = batch["acc"].to(DEVICE)
            rot = batch["rot"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            
            logits, _ = model(acc, rot, mask)
            loss = criterion(logits, label)
            
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)
            
            # Track subtle gesture accuracy
            for i in range(len(label)):
                if label[i].item() in subtle_indices:
                    subtle_total += 1
                    if preds[i] == label[i]:
                        subtle_correct += 1
    
    val_loss = val_loss / len(test_loader)
    val_acc = val_correct / val_total
    subtle_acc = subtle_correct / max(subtle_total, 1)
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if subtle_total > 0:
        print(f"  Subtle Gesture Acc: {subtle_acc:.4f} ({subtle_correct}/{subtle_total})")
    
    # Early stopping with patience
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✓ Saved best model (Val Acc: {val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  ⚠ Early stopping at epoch {epoch+1}")
            break

# ============================================================
# EVALUATION AND ANALYSIS
# ============================================================

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds, all_labels, all_probs = [], [], []
gesture_results = defaultdict(lambda: {'correct': 0, 'total': 0})

with torch.no_grad():
    for batch in test_loader:
        acc = batch["acc"].to(DEVICE)
        rot = batch["rot"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        labels = batch["label"].numpy()
        
        logits, _ = model(acc, rot, mask)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs.cpu().numpy())
        
        for true_label, pred_label in zip(labels, preds):
            gesture = dataset.gesture_classes[true_label]
            gesture_results[gesture]['total'] += 1
            if true_label == pred_label:
                gesture_results[gesture]['correct'] += 1

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ============================================================
# IMPROVED METRICS WITH DETAILED ANALYSIS
# ============================================================

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

overall_acc = accuracy_score(all_labels, all_preds)
print(f"\n{'='*70}")
print(f"IMPROVED RESULTS")
print(f"{'='*70}")
print(f"\nOverall accuracy: {overall_acc:.4f}")

# Per-gesture accuracy with type classification
print("\nGesture Accuracy by Type:")
print("-" * 80)
print(f"{'Gesture':30s} | {'Type':10s} | {'Acc':6s} | {'Samples':8s}")
print("-" * 80)

for gesture in dataset.gesture_classes:
    if gesture in gesture_results:
        acc = gesture_results[gesture]['correct'] / max(gesture_results[gesture]['total'], 1)
        gtype = dataset.gesture_type.get(dataset.le.transform([gesture])[0], 'other')
        print(f"{gesture:30s} | {gtype:10s} | {acc:.4f} | {gesture_results[gesture]['total']:8d}")

# Calculate accuracy by gesture type
type_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})
for gesture, results in gesture_results.items():
    gtype = dataset.gesture_type.get(dataset.le.transform([gesture])[0], 'other')
    type_accuracies[gtype]['correct'] += results['correct']
    type_accuracies[gtype]['total'] += results['total']

print("\nAccuracy by Gesture Type:")
print("-" * 40)
for gtype in ['subtle', 'large', 'other']:
    if type_accuracies[gtype]['total'] > 0:
        acc = type_accuracies[gtype]['correct'] / type_accuracies[gtype]['total']
        print(f"{gtype:10s}: {acc:.4f} ({type_accuracies[gtype]['correct']}/{type_accuracies[gtype]['total']})")

# Confusion matrix focusing on confused gestures
conf = confusion_matrix(all_labels, all_preds)

# Identify most confused pairs
conf_pairs = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and conf[i, j] > 0:
            conf_pairs.append((conf[i, j], i, j))

conf_pairs.sort(reverse=True)

print("\nMost confused gesture pairs:")
print("-" * 60)
for count, i, j in conf_pairs[:10]:
    print(f"{dataset.gesture_classes[i]:30s} → {dataset.gesture_classes[j]:30s}: {count}")

# ============================================================
# VISUALIZE IMPROVEMENTS
# ============================================================

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_losses, label='Training Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(val_accs, label='Validation Accuracy', color='green')
ax2.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best: {best_val_acc:.4f}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Bar chart of gesture accuracies
gestures = []
accuracies = []
colors = []

for gesture in dataset.gesture_classes:
    if gesture in gesture_results and gesture_results[gesture]['total'] > 0:
        gestures.append(gesture)
        acc = gesture_results[gesture]['correct'] / gesture_results[gesture]['total']
        accuracies.append(acc)
        gtype = dataset.gesture_type.get(dataset.le.transform([gesture])[0], 'other')
        if gtype == 'subtle':
            colors.append('red')
        elif gtype == 'large':
            colors.append('green')
        else:
            colors.append('blue')

# Sort by accuracy
sorted_indices = np.argsort(accuracies)
gestures = [gestures[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]
colors = [colors[i] for i in sorted_indices]

plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(gestures)), accuracies, color=colors)
plt.xlabel('Accuracy')
plt.ylabel('Gesture')
plt.title('Gesture Recognition Accuracy (Red=Subtle, Green=Large)')
plt.yticks(range(len(gestures)), gestures)
plt.xlim([0, 1.0])
plt.grid(True, alpha=0.3, axis='x')

# Add accuracy values on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{acc:.3f}', va='center')

plt.tight_layout()
plt.show()

# ============================================================
# TOP-K ACCURACY
# ============================================================

correct_topk = 0
for i in range(len(all_labels)):
    label = all_labels[i]
    prob = all_probs[i]
    top_k_indices = np.argsort(prob)[-TOP_K:][::-1]
    if label in top_k_indices:
        correct_topk += 1

topk_acc = correct_topk / len(all_labels)
print(f"\nTop-{TOP_K} accuracy: {topk_acc:.4f}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")


# ============================================================
# TEMPORAL CNN + TRANSFORMER GESTURE RECOGNITION MODEL
# Combines local pattern extraction (CNN) with global context (Transformer)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
BATCH_SIZE = 64
SEQ_LEN = 41
EPOCHS = 30
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# DATA HELPERS
# ============================================================

def handle_quaternion_missing_values(rot_data: np.ndarray) -> np.ndarray:
    """Intelligently handle missing quaternion values"""
    rot_cleaned = rot_data.copy()
    for i in range(len(rot_data)):
        row = rot_data[i]
        missing_count = np.isnan(row).sum()
        if missing_count == 0:
            norm = np.linalg.norm(row)
            rot_cleaned[i] = row / norm if norm > 1e-8 else [1.0, 0.0, 0.0, 0.0]
        elif missing_count == 1:
            missing_idx = np.where(np.isnan(row))[0][0]
            valid_values = row[~np.isnan(row)]
            sum_squares = np.sum(valid_values**2)
            missing_value = np.sqrt(max(0, 1.0 - sum_squares))
            if i > 0 and not np.isnan(rot_cleaned[i-1, missing_idx]):
                if rot_cleaned[i-1, missing_idx] < 0:
                    missing_value = -missing_value
            rot_cleaned[i, missing_idx] = missing_value
            rot_cleaned[i, ~np.isnan(row)] = valid_values
        else:
            rot_cleaned[i] = [1.0, 0.0, 0.0, 0.0]
    return rot_cleaned

def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Convert acceleration from device to world coordinates"""
    try:
        rot_scipy = rot[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w] for scipy
        norms = np.linalg.norm(rot_scipy, axis=1)
        if np.any(norms < 1e-8):
            mask = norms < 1e-8
            rot_scipy[mask] = [0.0, 0.0, 0.0, 1.0]
        r = R.from_quat(rot_scipy)
        acc_world = r.apply(acc)
        return acc_world
    except Exception:
        return acc.copy()

def add_engineered_features(df):
    """Add engineered features to the dataframe"""
    # Acceleration magnitude and jerk
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['jerk_x'] = df.groupby('sequence_id')['acc_x'].diff().fillna(0) * 200
    df['jerk_y'] = df.groupby('sequence_id')['acc_y'].diff().fillna(0) * 200
    df['jerk_z'] = df.groupby('sequence_id')['acc_z'].diff().fillna(0) * 200
    df['jerk_mag'] = np.sqrt(df['jerk_x']**2 + df['jerk_y']**2 + df['jerk_z']**2)
    
    # Angular velocity approximation
    df['rot_diff_w'] = df.groupby('sequence_id')['rot_w'].diff().fillna(0) * 200
    df['rot_diff_x'] = df.groupby('sequence_id')['rot_x'].diff().fillna(0) * 200
    df['rot_diff_y'] = df.groupby('sequence_id')['rot_y'].diff().fillna(0) * 200
    df['rot_diff_z'] = df.groupby('sequence_id')['rot_z'].diff().fillna(0) * 200
    df['rot_diff_mag'] = np.sqrt(df['rot_diff_w']**2 + df['rot_diff_x']**2 + 
                                 df['rot_diff_y']**2 + df['rot_diff_z']**2)
    
    # Normalize quaternions
    df['rot_mag'] = np.sqrt(df['rot_w']**2 + df['rot_x']**2 + df['rot_y']**2 + df['rot_z']**2)
    mask = df['rot_mag'] < 1e-6
    df.loc[~mask, ['rot_w', 'rot_x', 'rot_y', 'rot_z']] = df.loc[~mask, ['rot_w', 'rot_x', 'rot_y', 'rot_z']].div(
        df.loc[~mask, 'rot_mag'], axis=0
    )
    df.loc[mask, ['rot_w', 'rot_x', 'rot_y', 'rot_z']] = [1.0, 0.0, 0.0, 0.0]
    
    # Rolling statistics for temporal patterns
    df['acc_mag_rolling_mean'] = df.groupby('sequence_id')['acc_mag'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df['acc_mag_rolling_std'] = df.groupby('sequence_id')['acc_mag'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    )
    
    return df

# ============================================================
# ENHANCED DATASET
# ============================================================

class GestureDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len=21, mode='train'):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.mode = mode
        
        # Remove bad subjects
        df = df[~df['subject'].isin({'SUBJ_045235', 'SUBJ_019262'})]
        
        # Add engineered features
        df = add_engineered_features(df)
        
        # Encode gestures
        self.gesture_encoder = LabelEncoder()
        df['gesture_encoded'] = self.gesture_encoder.fit_transform(df['gesture'])
        self.gesture_classes = self.gesture_encoder.classes_.tolist()
        
        print(f"Gesture classes: {len(self.gesture_classes)}")
        
        # Group by sequence
        self.sequences = list(df.groupby('sequence_id'))
        self.windows = []
        
        for seq_id, seq in self.sequences:
            seq = seq.reset_index(drop=True)
            
            # Find gesture phase
            gesture_rows = seq[seq['phase'] == 'Gesture']
            if gesture_rows.empty:
                # Use middle of sequence
                center = len(seq) // 2
            else:
                center = gesture_rows.index[0]
            
            # Create window around gesture
            start = max(0, center - (seq_len // 2))
            end = min(len(seq), center + (seq_len // 2) + 1)
            
            # Extract features
            features = seq.iloc[start:end][feature_cols].values.astype(np.float32)
            
            # Handle missing values
            features = np.nan_to_num(features, nan=0.0)
            
            # Compute world acceleration
            if 'acc_x' in feature_cols and 'rot_w' in feature_cols:
                acc_idx = [feature_cols.index(c) for c in ['acc_x', 'acc_y', 'acc_z']]
                rot_idx = [feature_cols.index(c) for c in ['rot_w', 'rot_x', 'rot_y', 'rot_z']]
                
                # Handle missing quaternions
                rot_features = features[:, rot_idx].copy()
                rot_features = handle_quaternion_missing_values(rot_features)
                
                # Compute world acceleration
                acc_world = compute_world_acceleration(
                    features[:, acc_idx],
                    rot_features
                )
                features[:, acc_idx] = acc_world
                features[:, rot_idx] = rot_features
            
            # Pad if necessary
            if len(features) < seq_len:
                pad = np.zeros((seq_len - len(features), features.shape[1]), dtype=np.float32)
                features = np.vstack([pad, features])
                pad_mask = np.zeros(seq_len, dtype=np.bool_)
                pad_mask[-len(features):] = True
            elif len(features) > seq_len:
                features = features[-seq_len:]
                pad_mask = np.ones(seq_len, dtype=np.bool_)
            else:
                pad_mask = np.ones(seq_len, dtype=np.bool_)
            
            # Get gesture label
            gesture_label = seq['gesture_encoded'].iloc[0]
            
            self.windows.append({
                'features': features,
                'pad_mask': pad_mask,
                'gesture': gesture_label,
                'sequence_id': seq_id
            })
        
        print(f"Created {len(self.windows)} windows")
        
        # Compute class weights for imbalance
        self._compute_class_weights()
    
    def _compute_class_weights(self):
        """Compute class weights for imbalanced gesture classes"""
        gesture_counts = defaultdict(int)
        for window in self.windows:
            gesture_counts[window['gesture']] += 1
        
        total = len(self.windows)
        self.gesture_weights = torch.zeros(len(self.gesture_classes))
        for class_id, count in gesture_counts.items():
            # Inverse frequency weighting with smoothing
            self.gesture_weights[class_id] = total / (len(gesture_counts) * count)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        features = window['features'].copy()
        
        # Data augmentation for training
        if self.mode == 'train':
            # Add noise
            noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
            features = features + noise
            
            # Random scaling
            scale = np.random.uniform(0.95, 1.05)
            features = features * scale
            
            # Random time warping (slight)
            if np.random.random() > 0.5:
                # Slightly stretch or compress the sequence
                original_len = features.shape[0]
                new_len = int(original_len * np.random.uniform(0.95, 1.05))
                if new_len != original_len:
                    # Simple linear interpolation
                    indices = np.linspace(0, original_len-1, new_len)
                    features = np.array([np.interp(indices, range(original_len), features[:, i]) 
                                        for i in range(features.shape[1])]).T
        
        return {
            'X': torch.tensor(features, dtype=torch.float32),
            'mask': torch.tensor(window['pad_mask'], dtype=torch.bool),
            'gesture': torch.tensor(window['gesture'], dtype=torch.long)
        }

# ============================================================
# TEMPORAL CNN + TRANSFORMER MODEL
# ============================================================

class TemporalBlock(nn.Module):
    """Basic temporal block with dilation for multi-scale receptive fields"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.gelu(out)
        out = self.dropout(out)
        
        return out

class MultiScaleTemporalCNN(nn.Module):
    """Multi-scale temporal CNN with parallel dilated convolutions"""
    def __init__(self, in_channels, hidden_channels=64, dropout=0.2):
        super().__init__()
        
        # Initial projection
        self.init_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.init_bn = nn.BatchNorm1d(hidden_channels)
        
        # Parallel dilated convolutions for different receptive fields
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                TemporalBlock(hidden_channels, hidden_channels, dilation=1, dropout=dropout),
                TemporalBlock(hidden_channels, hidden_channels, dilation=2, dropout=dropout),
            ),
            nn.Sequential(
                TemporalBlock(hidden_channels, hidden_channels, dilation=2, dropout=dropout),
                TemporalBlock(hidden_channels, hidden_channels, dilation=4, dropout=dropout),
            ),
            nn.Sequential(
                TemporalBlock(hidden_channels, hidden_channels, dilation=3, dropout=dropout),
                TemporalBlock(hidden_channels, hidden_channels, dilation=6, dropout=dropout),
            )
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_channels * 3, hidden_channels * 2, kernel_size=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        # x: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # -> (batch, channels, seq_len)
        
        # Initial projection
        x = F.gelu(self.init_bn(self.init_conv(x)))
        
        # Parallel dilated convolutions
        branch_outputs = []
        for conv_path in self.dilated_convs:
            branch_outputs.append(conv_path(x))
        
        # Concatenate and fuse
        x = torch.cat(branch_outputs, dim=1)
        x = self.fusion(x)
        
        # Temporal pooling
        x = self.temporal_pool(x)  # -> (batch, channels, 1)
        x = x.squeeze(-1)  # -> (batch, channels)
        
        return x

class AdaptiveTransformerEncoder(nn.Module):
    """Transformer encoder with adaptive input size"""
    def __init__(self, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Adaptive pooling
        self.pooling = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        batch_size = x.size(0)
        
        # Add positional encoding (learned)
        seq_len = x.size(1)
        if not hasattr(self, 'pos_emb') or self.pos_emb.size(1) != seq_len:
            self.pos_emb = nn.Parameter(torch.randn(1, seq_len, self.d_model) * 0.02).to(x.device)
        
        x = x + self.pos_emb
        x = self.dropout(x)
        
        # Transformer
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)
        
        # Adaptive pooling
        attn_weights = self.pooling(x)  # (batch, seq_len, 1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), -1e9)
        
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(attn_weights * x, dim=1)  # (batch, d_model)
        
        return pooled, attn_weights

class TemporalCNNTransformer(nn.Module):
    """Main model: Temporal CNN + Transformer"""
    def __init__(self, input_dim, num_classes, cnn_hidden=64, transformer_dim=256, 
                 transformer_heads=8, transformer_layers=3, dropout=0.2):
        super().__init__()
        
        # ========== MODALITY-SPECIFIC TEMPORAL CNNs ==========
        # Accelerometer CNN
        self.acc_cnn = MultiScaleTemporalCNN(
            in_channels=3,  # x, y, z
            hidden_channels=cnn_hidden // 2,
            dropout=dropout
        )
        
        # Rotation CNN
        self.rot_cnn = MultiScaleTemporalCNN(
            in_channels=4,  # w, x, y, z
            hidden_channels=cnn_hidden // 2,
            dropout=dropout
        )
        
        # Derived features CNN
        self.derived_cnn = MultiScaleTemporalCNN(
            in_channels=input_dim - 7,  # All other features
            hidden_channels=cnn_hidden // 2,
            dropout=dropout
        )
        
        # ========== FEATURE FUSION ==========
        self.feature_fusion = nn.Sequential(
            nn.Linear(cnn_hidden * 3 // 2, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ========== TRANSFORMER ENCODER ==========
        self.transformer_encoder = AdaptiveTransformerEncoder(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # ========== CLASSIFICATION HEAD ==========
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim // 2, transformer_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(transformer_dim // 4, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # Split into modalities
        # Assuming first 3 features are acceleration
        acc_features = x[:, :, :3]
        # Next 4 features are rotation
        rot_features = x[:, :, 3:7]
        # Remaining are derived features
        derived_features = x[:, :, 7:]
        
        # Process each modality with its own CNN
        # Need to reshape for CNN: (batch, seq_len, channels) -> (batch, channels, seq_len)
        
        # Process acceleration
        acc_processed = self.acc_cnn(acc_features)  # (batch, cnn_hidden//2)
        
        # Process rotation
        rot_processed = self.rot_cnn(rot_features)  # (batch, cnn_hidden//2)
        
        # Process derived features (if any)
        if derived_features.size(2) > 0:
            derived_processed = self.derived_cnn(derived_features)  # (batch, cnn_hidden//2)
        else:
            derived_processed = torch.zeros(batch_size, self.derived_cnn.fusion[0].in_channels // 3, 
                                          device=x.device)
        
        # Concatenate modality features
        fused_features = torch.cat([acc_processed, rot_processed, derived_processed], dim=1)
        
        # Project to transformer dimension
        transformer_input = self.feature_fusion(fused_features)
        transformer_input = transformer_input.unsqueeze(1)  # (batch, 1, transformer_dim)
        
        # Create a dummy mask for transformer (since we have sequence length 1 after CNN)
        if mask is not None:
            # We need to adapt the mask for the transformer input
            # For simplicity, we'll use a mask of all True (since we have single token)
            transformer_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        else:
            transformer_mask = None
        
        # Transformer encoder (processes the fused representation)
        transformer_output, attn_weights = self.transformer_encoder(
            transformer_input, 
            mask=transformer_mask
        )
        
        # Classification
        logits = self.classifier(transformer_output)
        
        return logits, attn_weights

# Alternative: Sequential CNN -> Transformer
class SequentialTemporalCNNTransformer(nn.Module):
    """Alternative: Sequential processing (CNN -> Transformer on reduced sequence)"""
    def __init__(self, input_dim, num_classes, cnn_channels=64, transformer_dim=256, 
                 reduced_seq_len=10, dropout=0.2):
        super().__init__()
        
        # ========== TEMPORAL CNN FOR SEQUENCE REDUCTION ==========
        self.cnn_encoder = nn.Sequential(
            # First conv block
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second conv block with stride for downsampling
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Third conv block
            nn.Conv1d(cnn_channels * 2, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Adaptive pooling to fixed length
            nn.AdaptiveAvgPool1d(reduced_seq_len)
        )
        
        # ========== TRANSFORMER ON REDUCED SEQUENCE ==========
        self.transformer_encoder = AdaptiveTransformerEncoder(
            d_model=transformer_dim,
            nhead=8,
            num_layers=3,
            dropout=dropout
        )
        
        # Project CNN features to transformer dimension
        self.cnn_to_transformer = nn.Sequential(
            nn.Linear(cnn_channels * 2, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ========== CLASSIFICATION HEAD ==========
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(transformer_dim // 2, num_classes)
        )
        
        self.reduced_seq_len = reduced_seq_len
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch, channels, seq_len)
        x_cnn = x.transpose(1, 2)
        
        # CNN encoding and downsampling
        x_cnn = self.cnn_encoder(x_cnn)  # (batch, cnn_channels*2, reduced_seq_len)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, reduced_seq_len, cnn_channels*2)
        
        # Project to transformer dimension
        x_transformer = self.cnn_to_transformer(x_cnn)
        
        # Adjust mask for reduced sequence length
        if mask is not None:
            # Downsample mask (keep only every other position due to stride)
            reduced_mask = mask[:, ::2]  # Simple downsampling
            # Ensure we have the right length
            if reduced_mask.size(1) > self.reduced_seq_len:
                reduced_mask = reduced_mask[:, :self.reduced_seq_len]
            elif reduced_mask.size(1) < self.reduced_seq_len:
                # Pad if needed
                pad_len = self.reduced_seq_len - reduced_mask.size(1)
                reduced_mask = F.pad(reduced_mask, (0, pad_len), value=False)
        else:
            reduced_mask = None
        
        # Transformer encoder
        transformer_output, attn_weights = self.transformer_encoder(
            x_transformer, 
            mask=reduced_mask
        )
        
        # Classification
        logits = self.classifier(transformer_output)
        
        return logits, attn_weights

# ============================================================
# TRAINING UTILITIES
# ============================================================

class FocalLoss(nn.Module):
    """Focal loss for class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_one_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0
    correct = 0
    n = 0
    
    for batch in loader:
        X = batch["X"].to(device)
        mask = batch["mask"].to(device)
        gesture = batch["gesture"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(X, mask)
        loss = criterion(logits, gesture)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Metrics
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(1)
        correct += (preds == gesture).sum().item()
        n += X.size(0)
    
    return total_loss / n, correct / n

def evaluate(model, loader, device, criterion, return_predictions=False):
    model.eval()
    total_loss = 0
    correct = 0
    n = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            mask = batch["mask"].to(device)
            gesture = batch["gesture"].to(device)
            
            # Forward pass
            logits, _ = model(X, mask)
            loss = criterion(logits, gesture)
            
            # Metrics
            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(1)
            correct += (preds == gesture).sum().item()
            n += X.size(0)
            
            # Store for analysis
            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(gesture.cpu().numpy())
                probs = F.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())
    
    results = {
        'loss': total_loss / n,
        'accuracy': correct / n,
    }
    
    if return_predictions:
        results.update({
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs)
        })
    
    return results

# ============================================================
# MODEL SELECTION AND TRAINING
# ============================================================

def main(model_type='parallel'):
    """Main training function with model selection"""
    
    # Load data
    df = pd.read_csv(CSV_PATH)
    
    # Define feature columns
    base_features = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    engineered_features = ['acc_mag', 'jerk_x', 'jerk_y', 'jerk_z', 'jerk_mag',
                          'rot_diff_w', 'rot_diff_x', 'rot_diff_y', 'rot_diff_z', 'rot_diff_mag',
                          'acc_mag_rolling_mean', 'acc_mag_rolling_std']
    feature_cols = base_features + engineered_features
    
    # Create dataset
    dataset = GestureDataset(df, feature_cols, seq_len=SEQ_LEN, mode='train')
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    # Update mode for test set
    test_ds.dataset.mode = 'test'
    
    # Create weighted sampler for training
    train_labels = [dataset.windows[i]['gesture'] for i in train_ds.indices]
    sample_weights = [dataset.gesture_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels))
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Model parameters
    num_classes = len(dataset.gesture_classes)
    input_dim = len(feature_cols)
    
    print(f"\nModel Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Sequence length: {SEQ_LEN}")
    
    # Select model architecture
    if model_type == 'parallel':
        print("  Architecture: Parallel Temporal CNN + Transformer")
        model = TemporalCNNTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            cnn_hidden=64,
            transformer_dim=256,
            transformer_heads=8,
            transformer_layers=3,
            dropout=0.2
        )
    elif model_type == 'sequential':
        print("  Architecture: Sequential CNN -> Transformer")
        model = SequentialTemporalCNNTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            cnn_channels=64,
            transformer_dim=256,
            reduced_seq_len=10,
            dropout=0.2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Loss function with class weights
    criterion = FocalLoss(
        alpha=dataset.gesture_weights.to(DEVICE),
        gamma=2.0
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Training loop
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    train_history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, criterion
        )
        
        # Validation
        val_results = evaluate(
            model, test_loader, DEVICE, criterion
        )
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        
        # Store history
        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_accuracy'].append(val_acc)
        
        # Print progress
        print(f"\nEpoch {epoch+1:02d}/{EPOCHS}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'train_history': train_history,
                'gesture_classes': dataset.gesture_classes,
                'model_type': model_type,
                'input_dim': input_dim,
                'num_classes': num_classes
            }, f'best_model_{model_type}.pth')
            print(f"  ✓ Saved best model (Accuracy: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠ Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    print(f"\nLoading best model...")
    checkpoint = torch.load(f'best_model_{model_type}.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION - {model_type.upper()} MODEL")
    print(f"{'='*70}")
    
    val_results = evaluate(
        model, test_loader, DEVICE, criterion,
        return_predictions=True
    )
    
    print(f"\nResults:")
    print(f"  Accuracy: {val_results['accuracy']:.4f}")
    print(f"  Loss: {val_results['loss']:.4f}")
    
    # Detailed analysis
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"\nClassification Report:")
    print(classification_report(
        val_results['labels'],
        val_results['predictions'],
        target_names=dataset.gesture_classes,
        digits=3
    ))
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training curves
    ax1.plot(train_history['loss'], label='Training Loss')
    ax1.plot(train_history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_history['accuracy'], label='Training Accuracy')
    ax2.plot(train_history['val_accuracy'], label='Validation Accuracy')
    ax2.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best: {best_val_acc:.4f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Accuracy by gesture
    gesture_accuracies = {}
    for i, gesture in enumerate(dataset.gesture_classes):
        mask = val_results['labels'] == i
        if mask.sum() > 0:
            acc = (val_results['predictions'][mask] == val_results['labels'][mask]).mean()
            gesture_accuracies[gesture] = acc
    
    # Sort by accuracy
    sorted_gestures = sorted(gesture_accuracies.items(), key=lambda x: x[1])
    gestures = [g[0] for g in sorted_gestures]
    accuracies = [g[1] for g in sorted_gestures]
    
    # Plot only top and bottom performers to avoid overcrowding
    n_show = min(10, len(gestures))
    show_gestures = gestures[:n_show//2] + gestures[-n_show//2:]
    show_accuracies = accuracies[:n_show//2] + accuracies[-n_show//2:]
    
    colors = ['red' if i < n_show//2 else 'green' for i in range(len(show_gestures))]
    bars = ax3.barh(range(len(show_gestures)), show_accuracies, color=colors)
    ax3.set_xlabel('Accuracy')
    ax3.set_ylabel('Gesture')
    ax3.set_title('Gesture Recognition Accuracy (Red=Worst, Green=Best)')
    ax3.set_yticks(range(len(show_gestures)))
    ax3.set_yticklabels(show_gestures)
    ax3.set_xlim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, show_accuracies)):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center')
    
    # Confusion matrix for top confused pairs
    cm = confusion_matrix(val_results['labels'], val_results['predictions'])
    
    # Find most confused pairs
    confusion_pairs = []
    for i in range(len(dataset.gesture_classes)):
        for j in range(len(dataset.gesture_classes)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((cm[i, j], i, j))
    
    confusion_pairs.sort(reverse=True)
    
    # Show confusion matrix as text
    ax4.axis('off')
    ax4.set_title('Top 5 Most Confused Gesture Pairs')
    
    confusion_text = "Most Confused Pairs:\n"
    for count, i, j in confusion_pairs[:5]:
        confusion_text += f"{dataset.gesture_classes[i]:25s} -> {dataset.gesture_classes[j]:25s}: {count}\n"
    
    ax4.text(0.1, 0.5, confusion_text, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'training_results_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compare architectures if both are trained
    if model_type == 'parallel':
        print(f"\n{'='*70}")
        print("ARCHITECTURE COMPARISON")
        print(f"{'='*70}")
        print("Parallel Temporal CNN + Transformer advantages:")
        print("1. Separate processing for different modalities")
        print("2. Multi-scale receptive fields with dilated convolutions")
        print("3. Better for capturing local patterns in each sensor modality")
        print("4. More parameters, potentially better accuracy")
        print(f"\nFinal Accuracy: {val_results['accuracy']:.4f}")
    
    return model, dataset, val_results

# ============================================================
# RUN EXPERIMENTS
# ============================================================

if __name__ == "__main__":
    # Choose architecture: 'parallel' or 'sequential'
    # Parallel: Separate CNNs for each modality, then transformer
    # Sequential: Single CNN for downsampling, then transformer
    
    # Run parallel architecture (recommended)
    print("Training Parallel Temporal CNN + Transformer model...")
    model_parallel, dataset_parallel, results_parallel = main(model_type='parallel')
    
    # Optionally run sequential architecture for comparison
    # print("\n" + "="*70)
    # print("Training Sequential CNN -> Transformer model...")
    # print("="*70)
    # model_sequential, dataset_sequential, results_sequential = main(model_type='sequential')


# ============================================================
# INCEPTIONTIME BASELINE FOR IMU GESTURE CLASSIFICATION
# Kaggle-ready single file
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
BATCH_SIZE = 64
WINDOW_BEFORE = 30
WINDOW_AFTER = 30
WINDOW_LEN = WINDOW_BEFORE + WINDOW_AFTER + 1
EPOCHS = 25
LR = 1e-3
TOP_K = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ============================================================
# DATASET
# ============================================================

class IMUWindowDataset(Dataset):
    ACC_COLS = ["acc_x", "acc_y", "acc_z"]
    ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        df["gesture"] = df["gesture"].astype(str)
        self.le = LabelEncoder()
        df["gesture_id"] = self.le.fit_transform(df["gesture"])
        self.classes = self.le.classes_

        self.windows = []

        for _, seq in df.groupby("sequence_id"):
            seq = seq.reset_index(drop=True)

            g_rows = seq[seq["phase"] == "Gesture"]
            if g_rows.empty:
                continue

            center = g_rows.index[0]
            start = center - WINDOW_BEFORE
            end = center + WINDOW_AFTER + 1

            s = max(0, start)
            e = min(len(seq), end)

            acc = seq.iloc[s:e][self.ACC_COLS].values.astype(np.float32)
            rot = seq.iloc[s:e][self.ROT_COLS].values.astype(np.float32)

            if len(acc) == 0:
                continue

            # ---- normalize acc (per window)
            acc = (acc - acc.mean(0)) / (acc.std(0) + 1e-6)

            # ---- normalize quaternion
            rot = np.nan_to_num(rot)
            norm = np.linalg.norm(rot, axis=1, keepdims=True)
            norm[norm < 1e-6] = 1.0
            rot = rot / norm

            x = np.concatenate([acc, rot], axis=1)  # (T, 7)

            pad = np.zeros((WINDOW_LEN, 7), dtype=np.float32)
            mask_start = s - start
            pad[mask_start:mask_start+len(x)] = x

            label = seq["gesture_id"].iloc[0]
            self.windows.append((pad, label))

        print(f"Created {len(self.windows)} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        # (C, T) for CNN
        return (
            torch.tensor(x.T, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

# ============================================================
# INCEPTION MODULE
# ============================================================

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.branch1 = nn.Conv1d(in_ch, out_ch, 1)

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
        )

        self.branch5 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.Conv1d(out_ch, out_ch, 5, padding=2),
        )

        self.pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, 1),
        )

        self.bn = nn.BatchNorm1d(out_ch * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.pool(x)
        ], dim=1)
        return self.relu(self.bn(x))

# ============================================================
# INCEPTIONTIME MODEL
# ============================================================

class InceptionTime(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.block1 = InceptionBlock(in_channels, 32)
        self.block2 = InceptionBlock(32 * 4, 32)
        self.block3 = InceptionBlock(32 * 4, 32)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32 * 4, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ============================================================
# LOAD DATA
# ============================================================

dataset = IMUWindowDataset(CSV_PATH)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ============================================================
# TRAIN
# ============================================================

model = InceptionTime(7, num_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total, correct, loss_sum = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch+1:02d} | Train Acc {correct/total:.3f} | Loss {loss_sum/len(train_loader):.3f}")

# ============================================================
# EVALUATION
# ============================================================

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = (all_preds == all_labels).mean()
print(f"\nOverall Accuracy: {acc:.3f}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

conf = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_labels, all_preds):
    conf[t, p] += 1

plt.figure(figsize=(12, 10))
sns.heatmap(conf, xticklabels=dataset.classes, yticklabels=dataset.classes,
            cmap="Blues", fmt="d")
plt.title("Confusion Matrix – InceptionTime")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ============================================================
# TOP-K ACCURACY
# ============================================================

correct_topk = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        topk = logits.topk(TOP_K, dim=1).indices
        correct_topk += sum(y[i] in topk[i] for i in range(y.size(0)))

print(f"Top-{TOP_K} Accuracy: {correct_topk / len(all_labels):.3f}")



# =====================================================
# DeepConvLSTM IMU Baseline — FULL Kaggle Script
# =====================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Config
# ------------------------
DATA_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"  # <-- CHANGE
WINDOW_LEN = 30
CENTER_OFFSET = 15
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ------------------------
# Load data
# ------------------------
df = pd.read_csv(DATA_PATH)

# Basic cleanup
df = df.sort_values(["sequence_id", "row_id"] if "row_id" in df.columns else ["sequence_id"])
df = df.reset_index(drop=True)

# Encode labels
le = LabelEncoder()
df["gesture_int"] = le.fit_transform(df["gesture"])
NUM_CLASSES = len(le.classes_)

print("Classes:", NUM_CLASSES)

# ------------------------
# Dataset
# ------------------------
class IMUWindowDataset(Dataset):
    def __init__(self, df):
        self.samples = []

        for seq_id, seq in df.groupby("sequence_id"):
            seq = seq.reset_index(drop=True)

            if "Gesture" not in seq["phase"].values:
                continue

            center_idx = seq.index[seq["phase"] == "Gesture"][0]

            start = center_idx - CENTER_OFFSET
            end = start + WINDOW_LEN

            if start < 0 or end > len(seq):
                continue

            window = seq.iloc[start:end]

            x = window[["acc_x", "acc_y", "acc_z"]].values.astype(np.float32)
            y = window["gesture_int"].iloc[CENTER_OFFSET]

            if np.isnan(x).any():
                continue

            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# ------------------------
# Train / Val split (by subject if exists)
# ------------------------
if "subject" in df.columns:
    subjects = df["subject"].unique()
    np.random.shuffle(subjects)
    split = int(0.8 * len(subjects))
    train_subj = set(subjects[:split])

    train_df = df[df["subject"].isin(train_subj)]
    val_df   = df[~df["subject"].isin(train_subj)]
else:
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

train_ds = IMUWindowDataset(train_df)
val_ds   = IMUWindowDataset(val_df)

print("Train windows:", len(train_ds))
print("Val windows:", len(val_ds))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ------------------------
# Model: DeepConvLSTM
# ------------------------
class DeepConvLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, C, T)
        x = self.conv(x)
        x = x.transpose(1, 2)          # (B, T, C)

        x, _ = self.lstm(x)
        x = x.mean(dim=1)

        return self.fc(x)

# ------------------------
# Training utils
# ------------------------
def topk_acc(logits, y, k=3):
    return (logits.topk(k, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean().item()

def train_epoch(model, loader):
    model.train()
    acc, loss_sum = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        acc += (out.argmax(1) == y).float().mean().item()
        loss_sum += loss.item()

    return loss_sum / len(loader), acc / len(loader)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    acc, top3, loss_sum = 0, 0, 0
    preds, labels = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)

        loss = loss_fn(out, y)
        loss_sum += loss.item()

        acc += (out.argmax(1) == y).float().mean().item()
        top3 += topk_acc(out, y, 3)

        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())

    return (
        loss_sum / len(loader),
        acc / len(loader),
        top3 / len(loader),
        np.concatenate(preds),
        np.concatenate(labels),
    )

# ------------------------
# Train
# ------------------------
model = DeepConvLSTM(NUM_CLASSES).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

best_acc = 0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_epoch(model, train_loader)
    val_loss, val_acc, val_top3, preds, labels = eval_epoch(model, val_loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Acc {tr_acc:.3f} | "
        f"Val Acc {val_acc:.3f} | "
        f"Top-3 {val_top3:.3f}"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

# ------------------------
# Confusion Matrix
# ------------------------
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# -------------------------
# Dataset
# -------------------------
class GestureDataset(Dataset):
    def __init__(self, df, imu_cols, thm_cols, tof_cols, label_col):
        self.df = df
        self.imu_cols = imu_cols
        self.thm_cols = thm_cols
        self.tof_cols = tof_cols
        self.labels = df[label_col].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imu = row[self.imu_cols].values.astype(np.float32)
        thm = row[self.thm_cols].values.astype(np.float32)
        tof = row[self.tof_cols].values.astype(np.float32)

        # Replace missing TOF with 500 and normalize
        tof[tof == -1] = 500
        tof = tof / 500.0

        x = np.concatenate([imu, thm, tof], axis=-1)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# -------------------------
# Model
# -------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.pool = nn.MaxPool1d(2)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out

class GestureModel(nn.Module):
    def __init__(self, imu_ch, thm_ch, tof_ch, n_classes):
        super().__init__()
        # Branches
        self.imu_branch = nn.Sequential(
            ResidualBlock1D(imu_ch, 128),
            ResidualBlock1D(128, 256)
        )
        self.thm_branch = nn.Sequential(
            ResidualBlock1D(thm_ch, 64),
            ResidualBlock1D(64, 128)
        )
        self.tof_branch = nn.Sequential(
            ResidualBlock1D(tof_ch, 128),
            ResidualBlock1D(128, 256)
        )

        # BiLSTM over concatenated features
        self.bilstm = nn.LSTM(256+256+128, 512, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(1024, 512)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x: [B, T, C]
        imu_ch = 12
        thm_ch = 5
        tof_ch = x.shape[-1] - imu_ch - thm_ch

        imu = x[:, :, :imu_ch].transpose(1,2)
        thm = x[:, :, imu_ch:imu_ch+thm_ch].transpose(1,2)
        tof = x[:, :, imu_ch+thm_ch:].transpose(1,2)

        imu_feat = self.imu_branch(imu).transpose(1,2)
        thm_feat = self.thm_branch(thm).transpose(1,2)
        tof_feat = self.tof_branch(tof).transpose(1,2)

        merged = torch.cat([imu_feat, thm_feat, tof_feat], dim=2)

        lstm_out, _ = self.bilstm(merged)
        attn_weights = F.softmax(self.attn_fc(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out

# -------------------------
# Training Loop
# -------------------------
def train_loop(model, loader, optimizer, criterion, device):
    model.train()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return correct/total

def val_loop(model, loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return correct/total

# -------------------------
# Usage Example
# -------------------------
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
    imu_cols = [c for c in df.columns if "acc" in c or "rot" in c]
    thm_cols = [c for c in df.columns if "thm" in c]
    tof_cols = [c for c in df.columns if "tof" in c]

    dataset = GestureDataset(df, imu_cols, thm_cols, tof_cols, "gesture")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GestureModel(len(imu_cols), len(thm_cols), len(tof_cols), n_classes=20)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        train_acc = train_loop(model, loader, optimizer, criterion, device)
        val_acc = val_loop(model, loader, criterion, device)
        print(f"Epoch {epoch+1} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
BATCH_SIZE = 64  # Original was good
WINDOW_BEFORE = 30  # Original values
WINDOW_AFTER = 30
WINDOW_LEN = WINDOW_BEFORE + WINDOW_AFTER + 1
EPOCHS = 35  # Increased slightly
LR = 1e-3  # Original was good
TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Window length: {WINDOW_LEN}")

# ============================================================
# MINIMAL PREPROCESSING (KEEP IT SIMPLE)
# ============================================================

def simple_quaternion_normalization(rot_data: np.ndarray) -> np.ndarray:
    """Fast quaternion normalization - keep it simple"""
    rot_norm = rot_data.copy()
    norms = np.linalg.norm(rot_data, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    return rot_norm / norms

def extract_simple_features(acc, rot):
    """Original approach worked well - keep it"""
    # 1. Normalize acceleration (per window)
    acc_norm = (acc - acc.mean(0)) / (acc.std(0) + 1e-6)
    
    # 2. Normalize quaternions
    rot_norm = simple_quaternion_normalization(rot)
    
    # 3. Add magnitude features (NEW - simple but effective)
    acc_magnitude = np.linalg.norm(acc_norm, axis=1, keepdims=True)
    rot_magnitude = np.linalg.norm(rot_norm, axis=1, keepdims=True)
    
    # 4. Add simple temporal difference (NEW - captures motion)
    if len(acc_norm) > 1:
        acc_diff = np.diff(acc_norm, axis=0, prepend=acc_norm[0:1])
        rot_diff = np.diff(rot_norm, axis=0, prepend=rot_norm[0:1])
    else:
        acc_diff = np.zeros_like(acc_norm)
        rot_diff = np.zeros_like(rot_norm)
    
    # Total: 7 (original) + 2 (magnitude) + 10 (differences) = 19 features
    features = np.concatenate([
        acc_norm,        # 3 features
        rot_norm,        # 4 features
        acc_magnitude,   # 1 feature
        rot_magnitude,   # 1 feature
        acc_diff,        # 3 features
        rot_diff         # 7 features (4 rot + 3 acc diff for rot?)
    ], axis=1)
    
    # Actually rot_diff is 4, so total is 3+4+1+1+3+4 = 16 features
    return features[:, :16]  # Ensure consistent shape

# ============================================================
# DATASET - SIMILAR TO ORIGINAL BUT BETTER
# ============================================================

class ImprovedIMUDataset(Dataset):
    ACC_COLS = ["acc_x", "acc_y", "acc_z"]
    ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]
    
    def __init__(self, csv_path, use_advanced_features=False):
        df = pd.read_csv(csv_path)
        
        # Encode labels
        df["gesture"] = df["gesture"].astype(str)
        self.le = LabelEncoder()
        df["gesture_id"] = self.le.fit_transform(df["gesture"])
        self.classes = self.le.classes_
        
        self.windows = []
        self.use_advanced_features = use_advanced_features
        
        print(f"Processing {len(df['sequence_id'].unique())} sequences...")
        
        sequence_count = 0
        for seq_id, seq in df.groupby("sequence_id"):
            sequence_count += 1
            # if sequence_count % 100 == 0:
            #     print(f"  Processed {sequence_count} sequences...")
                
            seq = seq.reset_index(drop=True).copy()
            
            # Fill NaN values with 0 (simple approach)
            seq[self.ACC_COLS] = seq[self.ACC_COLS].fillna(0)
            seq[self.ROT_COLS] = seq[self.ROT_COLS].fillna(0)
            
            # Find ALL gesture points in this sequence
            gesture_indices = seq[seq["phase"] == "Gesture"].index.tolist()
            
            for center_idx in gesture_indices[:2]:  # Use up to 2 gestures per sequence
                start_idx = center_idx - WINDOW_BEFORE
                end_idx = center_idx + WINDOW_AFTER + 1
                
                # Adjust boundaries
                start_idx = max(0, start_idx)
                end_idx = min(len(seq), end_idx)
                
                # Skip if window is too small
                if end_idx - start_idx < 20:
                    continue
                
                # Extract data
                acc_data = seq.iloc[start_idx:end_idx][self.ACC_COLS].values.astype(np.float32)
                rot_data = seq.iloc[start_idx:end_idx][self.ROT_COLS].values.astype(np.float32)
                
                # Skip empty windows
                if len(acc_data) == 0:
                    continue
                
                # Extract features
                if self.use_advanced_features:
                    features = extract_simple_features(acc_data, rot_data)
                else:
                    # Original approach
                    acc_norm = (acc_data - acc_data.mean(0)) / (acc_data.std(0) + 1e-6)
                    rot_norm = simple_quaternion_normalization(rot_data)
                    features = np.concatenate([acc_norm, rot_norm], axis=1)
                
                # Pad or truncate to exact window length
                if len(features) < WINDOW_LEN:
                    # Pad with zeros
                    pad_len = WINDOW_LEN - len(features)
                    features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
                else:
                    # Truncate to window length
                    features = features[:WINDOW_LEN]
                
                # Get label
                label = seq.loc[center_idx, "gesture_id"]
                
                # Store as (features, label)
                self.windows.append((features.T, label))
        
        print(f"\nCreated {len(self.windows)} windows")
        print(f"Feature dimension: {self.windows[0][0].shape if self.windows else 'No windows'}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        features, label = self.windows[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

# ============================================================
# IMPROVED INCEPTIONTIME ARCHITECTURE
# ============================================================

class BasicInceptionModule(nn.Module):
    """Basic Inception module (no residual connections)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # Branch 2: 1x1 -> 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Branch 3: 1x1 -> 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
        )
        
        # Branch 4: 3x3 max pool -> 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )
        
        # Batch normalization and activation (applied after concatenation)
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Process through all branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        # Concatenate
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        # Apply batch norm and activation
        out = self.relu(self.bn(out))
        
        return out

class BasicInceptionTime(nn.Module):
    """Basic InceptionTime model (original architecture)"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # Three basic Inception blocks (original uses 3)
        self.inception1 = BasicInceptionModule(in_channels, 32)
        self.inception2 = BasicInceptionModule(32 * 4, 32)
        self.inception3 = BasicInceptionModule(32 * 4, 32)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Single fully connected layer (original architecture)
        self.fc = nn.Linear(32 * 4, num_classes)
        
    def forward(self, x):
        # Through inception blocks
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        
        return x

# ============================================================
# EVEN SIMPLER VERSION (Closest to original paper)
# ============================================================

class PaperInceptionModule(nn.Module):
    """Inception module as described in the original InceptionTime paper"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 40 filters per branch (as in paper)
        filters_per_branch = out_channels
        
        self.bottleneck = nn.Conv1d(in_channels, filters_per_branch, kernel_size=1)
        
        self.conv1 = nn.Conv1d(filters_per_branch, filters_per_branch, kernel_size=1)
        self.conv3 = nn.Conv1d(filters_per_branch, filters_per_branch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(filters_per_branch, filters_per_branch, kernel_size=5, padding=2)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.convpool = nn.Conv1d(in_channels, filters_per_branch, kernel_size=1)
        
        self.bn = nn.BatchNorm1d(filters_per_branch * 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Bottleneck layer
        bottleneck_out = self.bottleneck(x)
        
        # Three convolutions
        out1 = self.conv1(bottleneck_out)
        out2 = self.conv3(bottleneck_out)
        out3 = self.conv5(bottleneck_out)
        
        # Max pooling branch
        out4 = self.maxpool(x)
        out4 = self.convpool(out4)
        
        # Concatenate and activate
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.relu(self.bn(out))
        
        return out

class PaperInceptionTime(nn.Module):
    """InceptionTime as described in the original paper"""
    def __init__(self, in_channels, num_classes, num_blocks=3, filters=32):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        # First block
        self.blocks.append(PaperInceptionModule(in_channels, filters))
        
        # Additional blocks
        for _ in range(1, num_blocks):
            self.blocks.append(PaperInceptionModule(filters * 4, filters))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(filters * 4, num_classes)
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# ============================================================
# MINIMALIST VERSION (Closest to your original code)
# ============================================================

class MinimalInceptionBlock(nn.Module):
    """Minimal Inception block matching your original code structure"""
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.branch1 = nn.Conv1d(in_ch, out_ch, 1)

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
        )

        self.branch5 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.Conv1d(out_ch, out_ch, 5, padding=2),
        )

        self.pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, 1),
        )

        self.bn = nn.BatchNorm1d(out_ch * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.pool(x)
        ], dim=1)
        return self.relu(self.bn(x))

class MinimalInceptionTime(nn.Module):
    """Minimal InceptionTime matching your original code exactly"""
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.block1 = MinimalInceptionBlock(in_channels, 32)
        self.block2 = MinimalInceptionBlock(32 * 4, 32)
        self.block3 = MinimalInceptionBlock(32 * 4, 32)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32 * 4, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
class InceptionModule(nn.Module):
    """Improved Inception module with better feature extraction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # Branch 2: 1x1 -> 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Branch 3: 1x1 -> 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
        )
        
        # Branch 4: 3x3 max pool -> 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
        )
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.activation = nn.ReLU()
        
        # Residual connection (NEW - helps with deeper networks)
        self.residual = None
        if in_channels != out_channels * 4:
            self.residual = nn.Conv1d(in_channels, out_channels * 4, kernel_size=1)
        
    def forward(self, x):
        # Store residual
        residual = x
        
        # Process through all branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        # Concatenate
        out = torch.cat([out1, out2, out3, out4], dim=1)
        
        # Apply residual if needed
        if self.residual is not None:
            residual = self.residual(residual)
        
        # Add residual and apply batch norm + activation
        out = out + residual
        out = self.activation(self.bn(out))
        
        return out

class ImprovedInceptionTime(nn.Module):
    """Improved InceptionTime with better depth and feature usage"""
    def __init__(self, in_channels, num_classes, bottleneck=True):
        super().__init__()
        
        # Initial convolution to increase channels
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        # Inception modules with increasing complexity
        self.inception1 = InceptionModule(32, 32)  # Output: 128 channels
        self.inception2 = InceptionModule(128, 32) # Output: 128 channels
        self.inception3 = InceptionModule(128, 32) # Output: 128 channels
        self.inception4 = InceptionModule(128, 32) # Output: 128 channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Optional bottleneck layer (reduces overfitting)
        self.bottleneck = bottleneck
        if bottleneck:
            self.fc1 = nn.Linear(128, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, num_classes)
        else:
            self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Stack of inception modules
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        if self.bottleneck:
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = self.fc(x)
        
        return x

# ============================================================
# TRAINING FUNCTIONS WITH IMPROVEMENTS
# ============================================================

def train_model(model, train_loader, val_loader, num_epochs=EPOCHS, learning_rate=LR):
    """Train with improvements"""
    device = DEVICE
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler (NEW - helps converge better)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Track metrics
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    print("Starting training...")
    print("-" * 80)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Add L2 regularization manually for more control
            l2_lambda = 0.0001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            
            # Gradient clipping (prevents explosions)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
            
            # if batch_idx % 20 == 0:
            #     print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "/kaggle/working/best_inceptiontime_model.pth")
        
        # Print progress
        print(f"Epoch {epoch+1:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    return model, history, best_val_acc
    
class RawTestDataset(Dataset):
    """
    Test dataset that returns pandas DataFrames grouped by sequence_id
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # Group by sequence_id and store each as DataFrame
        self.sequences = []  # List of DataFrames
        self.sequence_ids = []  # List of sequence IDs
        
        for seq_id, seq_df in df.groupby("sequence_id"):
            # Reset index for this sequence
            seq_df = seq_df.reset_index(drop=True)
            
            self.sequences.append(seq_df)
            self.sequence_ids.append(seq_id)
        
        print(f"Loaded {len(self.sequences)} sequences from {csv_path}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Returns: (sequence_data_as_DataFrame, sequence_id)
        return self.sequences[idx], self.sequence_ids[idx]

# Custom collate function for DataFrames
def df_collate_fn(batch):
    """
    Custom collate function to handle pandas DataFrames
    """
    # batch is a list of tuples: [(df1, id1), (df2, id2), ...]
    
    # Separate DataFrames and IDs
    data = [item[0] for item in batch]  # List of DataFrames
    seq_ids = [item[1] for item in batch]  # List of sequence IDs
    
    return data, seq_ids
def prepare_window(sequence_df):
    def simple_quaternion_normalization(rot_data: np.ndarray) -> np.ndarray:
        """Fast quaternion normalization"""
        rot_norm = rot_data.copy()
        norms = np.linalg.norm(rot_data, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        return rot_norm / norms
    
    def extract_features(acc, rot):
        """Extract features (must match training)"""
        # Handle empty arrays
        if len(acc) == 0:
            return np.zeros((WINDOW_LEN, 16), dtype=np.float32)
        
        # Normalize acceleration
        acc_mean = acc.mean(0)
        acc_std = acc.std(0) + 1e-6
        acc_norm = (acc - acc_mean) / acc_std
        
        # Normalize quaternions
        rot_norm = simple_quaternion_normalization(rot)
        
        # Add magnitude features
        acc_magnitude = np.linalg.norm(acc_norm, axis=1, keepdims=True)
        rot_magnitude = np.linalg.norm(rot_norm, axis=1, keepdims=True)
        
        # Add temporal differences
        if len(acc_norm) > 1:
            acc_diff = np.diff(acc_norm, axis=0, prepend=acc_norm[0:1])
            rot_diff = np.diff(rot_norm, axis=0, prepend=rot_norm[0:1])
        else:
            acc_diff = np.zeros_like(acc_norm)
            rot_diff = np.zeros_like(rot_norm)
        
        # Combine features (16 total)
        features = np.concatenate([
            acc_norm,        # 3
            rot_norm,        # 4
            acc_magnitude,   # 1
            rot_magnitude,   # 1
            acc_diff,        # 3
            rot_diff         # 4
        ], axis=1)
        
        return features[:, :16]  # Ensure consistent shape
    def find_activity_window(acc_data):
        """
        Find the most active window in the sequence when phase is not available.
        Uses variance of acceleration magnitude to find interesting parts.
        """
        if len(acc_data) < WINDOW_LEN:
            # If sequence is shorter than window, use the whole sequence
            return 0, len(acc_data)
        
        # Calculate acceleration magnitude
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        
        # Find window with maximum variance (most activity)
        max_variance = -1
        best_start = 0
        
        # Slide window through sequence
        for start in range(0, len(acc_data) - WINDOW_LEN + 1, WINDOW_LEN // 4):
            end = start + WINDOW_LEN
            window_magnitude = acc_magnitude[start:end]
            variance = np.var(window_magnitude)
            
            if variance > max_variance:
                max_variance = variance
                best_start = start
        
        return best_start, best_start + WINDOW_LEN
        
    """Prepare a window from sequence for prediction (NO PHASE COLUMN)"""
    # Convert to pandas for easier processing
    if hasattr(sequence_df, 'to_pandas'):
        df = sequence_df.to_pandas()
    else:
        df = sequence_df
    
    # Get accelerometer and rotation data
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    rot_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]
    
    # Check if phase column exists
    has_phase = "phase" in df.columns
    
    if has_phase and "Gesture" in df["phase"].values:
        # If phase column exists and has "Gesture", use it
        gesture_rows = df[df["phase"] == "Gesture"]
        center_idx = gesture_rows.index[0]
    else:
        # No phase column or no gesture phase - find most active window
        acc_data = df[acc_cols].values.astype(np.float32)
        start_idx, end_idx = self.find_activity_window(acc_data)
        
        # Extract window
        acc_data = df.iloc[start_idx:end_idx][acc_cols].values.astype(np.float32)
        rot_data = df.iloc[start_idx:end_idx][rot_cols].values.astype(np.float32)
        
        # Extract features and return
        features = extract_features(acc_data, rot_data)
        
        # Pad if necessary
        if len(features) < WINDOW_LEN:
            pad_len = WINDOW_LEN - len(features)
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
        
        features = features[:WINDOW_LEN]
        return features.T  # Shape: (16, WINDOW_LEN)
    
    # If we have a phase column with gesture, proceed as before
    start_idx = max(0, center_idx - WINDOW_BEFORE)
    end_idx = min(len(df), center_idx + WINDOW_AFTER + 1)
    
    acc_data = df.iloc[start_idx:end_idx][acc_cols].values.astype(np.float32)
    rot_data = df.iloc[start_idx:end_idx][rot_cols].values.astype(np.float32)
    
    # Handle missing values
    acc_data = np.nan_to_num(acc_data)
    rot_data = np.nan_to_num(rot_data)
    
    # Extract features
    features = extract_features(acc_data, rot_data)
    
    # Pad if necessary
    if len(features) < WINDOW_LEN:
        pad_len = WINDOW_LEN - len(features)
        features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
    
    # Trim to exact window length
    features = features[:WINDOW_LEN]
    
    # Transpose for model (channels first)
    return features.T  # Shape: (16, WINDOW_LEN)            
def evaluate_model(model, test_loader, dataset_classes):
    """Comprehensive evaluation"""
    device = DEVICE
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # features = prepare_window(data)
            # features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Calculate metrics
    accuracy = (all_preds == all_targets).mean()
    
    # Top-K accuracy
    topk_correct = 0
    for i in range(len(all_targets)):
        top_k_indices = np.argsort(all_probs[i])[-TOP_K:]
        if all_targets[i] in top_k_indices:
            topk_correct += 1
    topk_acc = topk_correct / len(all_targets)
    
    print(f"\nEvaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Top-{TOP_K} Accuracy: {topk_acc:.4f}")
    
    return all_preds, all_targets, all_probs, accuracy, topk_acc

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 80)
    print("IMPROVED INCEPTIONTIME FOR GESTURE RECOGNITION")
    print("=" * 80)
    
    # Load dataset with improved features
    print("\n1. Loading and preparing dataset...")
    dataset = ImprovedIMUDataset(CSV_PATH, use_advanced_features=True)
    
    if len(dataset) == 0:
        print("ERROR: No data loaded!")
        return
    
    num_classes = len(dataset.classes)
    in_channels = dataset.windows[0][0].shape[0]
    
    print(f"   Samples: {len(dataset)}")
    print(f"   Classes: {num_classes} ({dataset.classes})")
    print(f"   Input shape: {dataset.windows[0][0].shape}")
    
    # Split dataset (80% train, 10% val, 10% test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # test_dataset = RawTestDataset("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
    # collate_fn=df_collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\n2. Creating model...")
    print(f"   Input channels: {in_channels}")
    print(f"   Number of classes: {num_classes}")
    
    # Try different model configurations
    models_to_try = [
        ("Minimal_InceptionTime", MinimalInceptionTime(in_channels, num_classes)),
        ("Basic_InceptionTime", BasicInceptionTime(in_channels, num_classes)),
        ("Paper_InceptionTime", PaperInceptionTime(in_channels, num_classes, num_blocks=3, filters=32)),
        ("Standard_InceptionTime", ImprovedInceptionTime(in_channels, num_classes, bottleneck=False)),
        ("Improved_InceptionTime", ImprovedInceptionTime(in_channels, num_classes, bottleneck=True)),
    ]
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    for model_name, model in models_to_try:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        model = model.to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Train
        trained_model, history, val_acc = train_model(
            model, train_loader, val_loader, num_epochs=EPOCHS
        )
        
        # Evaluate on test set
        _, _, _, test_acc, test_topk = evaluate_model(trained_model, test_loader, dataset.classes)
        
        model_path = f'/kaggle/working/{model_name}_model.pth'
        torch.save(model.state_dict(), model_path)
        # Track best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = trained_model
            best_model_name = model_name
        
        # Plot training history for this model
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Loss History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{model_name} - Accuracy History')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Final evaluation with best model
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test Accuracy: {best_accuracy:.4f}")
    print(f"{'='*80}")
    
    # Get predictions from best model
    test_preds, test_targets, test_probs, final_acc, final_topk = evaluate_model(
        best_model, test_loader, dataset.classes
    )
    
    # Confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, 
                xticklabels=dataset.classes, 
                yticklabels=dataset.classes,
                cmap="Blues", 
                fmt="d",
                annot=True,
                cbar_kws={'label': 'Count'},
                annot_kws={"size": 10})
    plt.title(f"Confusion Matrix - {best_model_name}\nAccuracy: {final_acc:.3f}", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(test_targets, test_preds, 
                                   target_names=dataset.classes, digits=3)
    print(report)
    
    # Save final model
    torch.save(best_model.state_dict(), "/kaggle/working/final_gesture_model.pth")
    print(f"\nModel saved to: /kaggle/working/final_gesture_model.pth")
    
    return best_model, dataset

# ============================================================
# RUN THE CODE
# ============================================================

model, dataset = main()

# Quick test with a few samples
print("\n" + "="*80)
print("QUICK INFERENCE TEST")
print("="*80)

# Get a test sample
# test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
model.eval()

for i, (data, target) in enumerate(test_loader):
# for batch_data, batch_seq_ids in test_loader:
#     for sequence_df, seq_id in zip(batch_data, batch_seq_ids):
    if i >= 3:  # Test 3 samples
        break
    # features = prepare_window(data)
    # features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    # Predict
    with torch.no_grad():
        outputs = model(data)
        probs = F.softmax(outputs, dim=1)
        pred_idx = outputs.argmax(dim=1).item()
    
    # Get prediction
    prediction = dataset.classes_[pred_idx]
    confidence = probs[0, pred_idx].item()
    
    print(f"Sample {i+1}: True={true_label:10s} | Pred={pred_label:10s} | "
          f"Confidence={confidence:.3f} | "
          f"{'CORRECT' if pred.item() == target.item() else 'WRONG'}")
        





# ============================================================
# HYBRID CNN + TRANSFORMER FUSED MODEL
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.transform import Rotation as R

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
BATCH_SIZE = 32
WINDOW_BEFORE = 10
WINDOW_AFTER = 10
WINDOW_LEN = WINDOW_BEFORE + WINDOW_AFTER + 1
D_MODEL = 64
NHEAD = 4
LAYERS = 2
EPOCHS = 30
LR = 3e-4
TOP_K = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ============================================================
# HELPER FUNCTIONS (integrate previous ones)
# ============================================================

def handle_quaternion_missing_values(rot_data: np.ndarray) -> np.ndarray:
    # Same as your helper
    rot_cleaned = rot_data.copy()
    for i in range(len(rot_data)):
        row = rot_data[i]
        missing_count = np.isnan(row).sum()
        if missing_count == 0:
            norm = np.linalg.norm(row)
            rot_cleaned[i] = row / norm if norm > 1e-8 else [1.0,0.0,0.0,0.0]
        elif missing_count == 1:
            missing_idx = np.where(np.isnan(row))[0][0]
            valid_values = row[~np.isnan(row)]
            sum_squares = np.sum(valid_values**2)
            if sum_squares <= 1.0:
                missing_value = np.sqrt(max(0,1.0-sum_squares))
                if i>0 and not np.isnan(rot_cleaned[i-1,missing_idx]):
                    if rot_cleaned[i-1,missing_idx]<0: missing_value=-missing_value
                rot_cleaned[i,missing_idx]=missing_value
                rot_cleaned[i,~np.isnan(row)] = valid_values
            else: rot_cleaned[i]=[1.0,0.0,0.0,0.0]
        else:
            rot_cleaned[i]=[1.0,0.0,0.0,0.0]
    return rot_cleaned

def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
    try:
        rot_scipy = rot[:,[1,2,3,0]]
        norms = np.linalg.norm(rot_scipy, axis=1)
        mask = norms<1e-8
        rot_scipy[mask] = [0.0,0.0,0.0,1.0]
        r = R.from_quat(rot_scipy)
        return r.apply(acc)
    except:
        return acc.copy()

def remove_gravity_from_acc(acc_data, rot_data):
    num_samples = acc_data.shape[0]
    linear_accel = np.zeros_like(acc_data)
    gravity_world = np.array([0,0,9.81])
    for i in range(num_samples):
        if np.all(np.isnan(rot_data[i])) or np.all(np.isclose(rot_data[i],0)):
            linear_accel[i,:] = acc_data[i,:]
            continue
        try:
            rotation = R.from_quat(rot_data[i])
            gravity_sensor_frame = rotation.apply(gravity_world,inverse=True)
            linear_accel[i,:] = acc_data[i,:]-gravity_sensor_frame
        except:
            linear_accel[i,:] = acc_data[i,:]
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    num_samples = rot_data.shape[0]
    angular_vel = np.zeros((num_samples,3))
    for i in range(num_samples-1):
        q_t = rot_data[i]; q_t_plus = rot_data[i+1]
        if np.any(np.isnan(q_t)) or np.any(np.isnan(q_t_plus)): continue
        try:
            r1 = R.from_quat(q_t); r2 = R.from_quat(q_t_plus)
            delta_rot = r1.inv()*r2
            angular_vel[i,:] = delta_rot.as_rotvec()/time_delta
        except: continue
    return angular_vel

def calculate_angular_distance(rot_data):
    num_samples = rot_data.shape[0]
    angular_dist = np.zeros(num_samples)
    for i in range(num_samples-1):
        q1=rot_data[i]; q2=rot_data[i+1]
        if np.any(np.isnan(q1)) or np.any(np.isnan(q2)): continue
        try:
            r1=R.from_quat(q1); r2=R.from_quat(q2)
            rel_rot = r1.inv()*r2
            angular_dist[i]=np.linalg.norm(rel_rot.as_rotvec())
        except: continue
    return angular_dist

# ============================================================
# DATASET
# ============================================================

class AccRotHybridDataset(Dataset):
    ACC_COLS = ["acc_x","acc_y","acc_z"]
    ROT_COLS = ["rot_w","rot_x","rot_y","rot_z"]

    def __init__(self,csv_path,window_before=10,window_after=10):
        self.df=pd.read_csv(csv_path)
        self.window_before = window_before
        self.window_after = window_after
        self.window_len = window_before+window_after+1

        # Encode gestures
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_

        self.sequences = list(self.df.groupby("sequence_id"))
        self.windows=[]

        for _,seq in self.sequences:
            seq = seq.reset_index(drop=True)
            gesture_rows = seq[seq["phase"]=="Gesture"]
            if gesture_rows.empty: continue
            center = gesture_rows.index[0]
            start=center-self.window_before; end=center+self.window_after+1
            start_c = max(0,start); end_c=min(len(seq),end)

            acc = seq.iloc[start_c:end_c][self.ACC_COLS].values.astype(np.float32)
            rot = seq.iloc[start_c:end_c][self.ROT_COLS].values.astype(np.float32)

            # Preprocess rotation
            rot = handle_quaternion_missing_values(rot)
            norms = np.linalg.norm(rot,axis=1,keepdims=True); norms[norms<1e-6]=1.0
            rot = rot/norms

            # Compute world acceleration and remove gravity
            acc_world = compute_world_acceleration(acc,rot)
            acc_linear = remove_gravity_from_acc(acc,rot)

            # Add angular velocity and distance
            angular_vel = calculate_angular_velocity_from_quat(rot)
            angular_dist = calculate_angular_distance(rot)

            # Concatenate features: acc_world + linear acc + angular vel + distance
            acc_features = np.concatenate([acc_world, acc_linear, angular_vel, angular_dist[:,None]], axis=1)

            # Padding
            acc_pad = np.zeros((self.window_len, acc_features.shape[1]),dtype=np.float32)
            rot_pad = np.zeros((self.window_len, 4),dtype=np.float32)
            mask = np.zeros(self.window_len,dtype=np.bool_)

            insert = start_c-start
            acc_pad[insert:insert+len(acc_features)] = acc_features
            rot_pad[insert:insert+len(rot)] = rot
            mask[insert:insert+len(acc_features)] = True

            label = seq["gesture_id"].mode().iloc[0]
            self.windows.append((acc_pad,rot_pad,mask,label))

        print(f"Created {len(self.windows)} multimodal windows")

    def __len__(self): return len(self.windows)
    def __getitem__(self,idx):
        acc, rot, mask, label = self.windows[idx]
        return {
            "acc": torch.tensor(acc,dtype=torch.float32),
            "rot": torch.tensor(rot,dtype=torch.float32),
            "mask": torch.tensor(mask,dtype=torch.bool),
            "label": torch.tensor(label,dtype=torch.long)
        }

# ============================================================
# MODEL
# ============================================================

class AxisCNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (B, 1, T)
        return: (B, hidden_dim)
        """
        x = self.net(x)
        x = x.mean(dim=-1)  # temporal pooling
        return x


class EnhancedCNNEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, axis_hidden_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # AxisCNN for per-axis feature extraction
        self.axis_cnns = nn.ModuleList([
            AxisCNN(hidden_dim=axis_hidden_dim) for _ in range(input_dim)
        ])
        
        # Dimension adjustment layer
        self.axis_proj = nn.Linear(input_dim * axis_hidden_dim, d_model)
        
        # Temporal CNN for sequence modeling
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, F)
        return: (B, T, d_model)
        """
        B, T, F = x.shape
        
        # Extract per-axis features
        axis_features = []
        for i in range(F):
            # Extract axis i: (B, T, 1) -> (B, 1, T)
            axis_data = x[:, :, i:i+1].transpose(1, 2)
            # Process through axis-specific CNN
            axis_feat = self.axis_cnns[i](axis_data)  # (B, axis_hidden_dim)
            axis_features.append(axis_feat)
        
        # Stack axis features: (B, F * axis_hidden_dim)
        axis_features = torch.cat(axis_features, dim=1)
        
        # Project to d_model dimension
        projected = self.axis_proj(axis_features)  # (B, d_model)
        
        # Repeat across temporal dimension
        expanded = projected.unsqueeze(1).repeat(1, T, 1)  # (B, T, d_model)
        
        # Apply temporal CNN
        expanded = expanded.transpose(1, 2)  # (B, d_model, T)
        expanded = self.temporal_cnn(expanded)
        expanded = expanded.transpose(1, 2)  # (B, T, d_model)
        
        return self.ln(expanded)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model*4,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        
    def forward(self, x, mask):
        x = self.encoder(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


class HybridFusedModel(nn.Module):
    def __init__(self, num_classes, acc_feat_dim, rot_dim=4, d_model=64, axis_hidden_dim=32):
        super().__init__()
        
        # Enhanced encoders with AxisCNN integration
        self.acc_encoder = EnhancedCNNEncoder(acc_feat_dim, d_model, axis_hidden_dim)
        self.rot_encoder = EnhancedCNNEncoder(rot_dim, d_model, axis_hidden_dim)
        
        # Transformer encoders
        self.acc_trans = TransformerEncoder(d_model)
        self.rot_trans = TransformerEncoder(d_model)
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, acc, rot, mask):
        # Process through enhanced CNN encoders
        acc_feat = self.acc_encoder(acc)
        rot_feat = self.rot_encoder(rot)
        
        # Process through transformers
        acc_seq = self.acc_trans(acc_feat, mask)
        rot_seq = self.rot_trans(rot_feat, mask)
        
        # Fuse features
        z = torch.cat([acc_seq, rot_seq], dim=1)
        z = self.fusion(z)
        
        return self.head(z)

# ============================================================
# TRAINING
# ============================================================

dataset = AccRotHybridDataset(CSV_PATH,WINDOW_BEFORE,WINDOW_AFTER)
num_classes = len(dataset.gesture_classes)
acc_feat_dim = dataset[0]["acc"].shape[1]

train_size = int(0.8*len(dataset)); test_size=len(dataset)-train_size
train_ds, test_ds = random_split(dataset,[train_size,test_size])
train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=BATCH_SIZE)

model = HybridFusedModel(num_classes,acc_feat_dim).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(),lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train(); total_loss=0
    for batch in train_loader:
        acc=batch["acc"].to(DEVICE); rot=batch["rot"].to(DEVICE)
        mask=batch["mask"].to(DEVICE); label=batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(acc,rot,mask)
        loss = criterion(logits,label)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

# ============================================================
# EVALUATION
# ============================================================

model.eval()
all_preds, all_labels=[],[]
with torch.no_grad():
    for batch in test_loader:
        acc=batch["acc"].to(DEVICE); rot=batch["rot"].to(DEVICE)
        mask=batch["mask"].to(DEVICE)
        logits = model(acc,rot,mask)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].numpy())

all_preds=np.array(all_preds); all_labels=np.array(all_labels)
overall_acc = (all_preds==all_labels).mean()
print(f"\nOverall accuracy: {overall_acc:.3f}")



# ============================================================
# FLEXIBLE HYBRID CNN + TRANSFORMER MODEL FOR KAGGLE
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.transform import Rotation as R
import time

# ============================================================
# CONFIGURABLE VARIABLES
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
WINDOW_BEFORE = 10
WINDOW_AFTER = 10
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
D_MODEL = 64
AXIS_HIDDEN_DIM = 32
USE_AXISCNN = True
USE_TRANSFORMER = True
FUSION = 'concat'  # 'concat' or 'gated'
DROPOUT = 0.1
USE_ACC = True
USE_ROT = True
VAL_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def handle_quaternion_missing_values(rot_data: np.ndarray) -> np.ndarray:
    rot_cleaned = rot_data.copy()
    for i in range(len(rot_data)):
        row = rot_data[i]
        missing_count = np.isnan(row).sum()
        if missing_count == 0:
            norm = np.linalg.norm(row)
            rot_cleaned[i] = row / norm if norm > 1e-8 else [1.0,0.0,0.0,0.0]
        elif missing_count == 1:
            missing_idx = np.where(np.isnan(row))[0][0]
            valid_values = row[~np.isnan(row)]
            sum_squares = np.sum(valid_values**2)
            if sum_squares <= 1.0:
                missing_value = np.sqrt(max(0,1.0-sum_squares))
                if i>0 and not np.isnan(rot_cleaned[i-1,missing_idx]):
                    if rot_cleaned[i-1,missing_idx]<0: missing_value=-missing_value
                rot_cleaned[i,missing_idx]=missing_value
                rot_cleaned[i,~np.isnan(row)] = valid_values
            else: rot_cleaned[i]=[1.0,0.0,0.0,0.0]
        else:
            rot_cleaned[i]=[1.0,0.0,0.0,0.0]
    return rot_cleaned

def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
    try:
        rot_scipy = rot[:,[1,2,3,0]]
        norms = np.linalg.norm(rot_scipy, axis=1)
        mask = norms<1e-8
        rot_scipy[mask] = [0.0,0.0,0.0,1.0]
        r = R.from_quat(rot_scipy)
        return r.apply(acc)
    except:
        return acc.copy()

def remove_gravity_from_acc(acc_data, rot_data):
    num_samples = acc_data.shape[0]
    linear_accel = np.zeros_like(acc_data)
    gravity_world = np.array([0,0,9.81])
    for i in range(num_samples):
        if np.all(np.isnan(rot_data[i])) or np.all(np.isclose(rot_data[i],0)):
            linear_accel[i,:] = acc_data[i,:]
            continue
        try:
            rotation = R.from_quat(rot_data[i])
            gravity_sensor_frame = rotation.apply(gravity_world,inverse=True)
            linear_accel[i,:] = acc_data[i,:]-gravity_sensor_frame
        except:
            linear_accel[i,:] = acc_data[i,:]
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    num_samples = rot_data.shape[0]
    angular_vel = np.zeros((num_samples,3))
    for i in range(num_samples-1):
        q_t = rot_data[i]; q_t_plus = rot_data[i+1]
        if np.any(np.isnan(q_t)) or np.any(np.isnan(q_t_plus)): continue
        try:
            r1 = R.from_quat(q_t); r2 = R.from_quat(q_t_plus)
            delta_rot = r1.inv()*r2
            angular_vel[i,:] = delta_rot.as_rotvec()/time_delta
        except: continue
    return angular_vel

def calculate_angular_distance(rot_data):
    num_samples = rot_data.shape[0]
    angular_dist = np.zeros(num_samples)
    for i in range(num_samples-1):
        q1=rot_data[i]; q2=rot_data[i+1]
        if np.any(np.isnan(q1)) or np.any(np.isnan(q2)): continue
        try:
            r1=R.from_quat(q1); r2=R.from_quat(q2)
            rel_rot = r1.inv()*r2
            angular_dist[i]=np.linalg.norm(rel_rot.as_rotvec())
        except: continue
    return angular_dist

# ============================================================
# DATASET
# ============================================================

class AccRotHybridDataset(Dataset):
    ACC_COLS = ["acc_x","acc_y","acc_z"]
    ROT_COLS = ["rot_w","rot_x","rot_y","rot_z"]

    def __init__(self,csv_path,window_before=10,window_after=10):
        self.df=pd.read_csv(csv_path)
        self.window_before = window_before
        self.window_after = window_after
        self.window_len = window_before+window_after+1

        # Encode gestures
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_

        self.sequences = list(self.df.groupby("sequence_id"))
        self.windows=[]

        for _,seq in self.sequences:
            seq = seq.reset_index(drop=True)
            gesture_rows = seq[seq["phase"]=="Gesture"]
            if gesture_rows.empty: continue
            center = gesture_rows.index[0]
            start=center-self.window_before; end=center+self.window_after+1
            start_c = max(0,start); end_c=min(len(seq),end)

            acc = seq.iloc[start_c:end_c][self.ACC_COLS].values.astype(np.float32)
            rot = seq.iloc[start_c:end_c][self.ROT_COLS].values.astype(np.float32)

            rot = handle_quaternion_missing_values(rot)
            norms = np.linalg.norm(rot,axis=1,keepdims=True); norms[norms<1e-6]=1.0
            rot = rot/norms

            acc_world = compute_world_acceleration(acc,rot)
            acc_linear = remove_gravity_from_acc(acc,rot)
            angular_vel = calculate_angular_velocity_from_quat(rot)
            angular_dist = calculate_angular_distance(rot)

            acc_features = np.concatenate([acc_world, acc_linear, angular_vel, angular_dist[:,None]], axis=1)

            acc_pad = np.zeros((self.window_len, acc_features.shape[1]),dtype=np.float32)
            rot_pad = np.zeros((self.window_len, 4),dtype=np.float32)
            mask = np.zeros(self.window_len,dtype=np.bool_)

            insert = start_c-start
            acc_pad[insert:insert+len(acc_features)] = acc_features
            rot_pad[insert:insert+len(rot)] = rot
            mask[insert:insert+len(acc_features)] = True

            label = seq["gesture_id"].mode().iloc[0]
            self.windows.append((acc_pad,rot_pad,mask,label))

        print(f"Created {len(self.windows)} multimodal windows")

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        acc, rot, mask, label = self.windows[idx]
    
        if not USE_ACC:
            acc = np.zeros_like(acc, dtype=np.float32)
    
        if not USE_ROT:
            rot = np.zeros_like(rot, dtype=np.float32)
    
        return {
            "acc": torch.tensor(acc, dtype=torch.float32),
            "rot": torch.tensor(rot, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
        }

# ============================================================
# MODEL
# ============================================================

class AxisCNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
    def forward(self,x):
        x = self.net(x)
        x = x.mean(dim=-1)
        return x

class EnhancedCNNEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, axis_hidden_dim=32):
        super().__init__()
        self.axis_cnns = nn.ModuleList([AxisCNN(hidden_dim=axis_hidden_dim) for _ in range(input_dim)])
        self.axis_proj = nn.Linear(input_dim*axis_hidden_dim, d_model)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(d_model,d_model,3,padding=1), nn.BatchNorm1d(d_model), nn.ReLU(),
            nn.Conv1d(d_model,d_model,3,padding=1), nn.BatchNorm1d(d_model), nn.ReLU()
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self,x):
        B,T,F = x.shape
        feats = []
        for i in range(F):
            axis_data = x[:,:,i:i+1].transpose(1,2)
            feats.append(self.axis_cnns[i](axis_data))
        projected = self.axis_proj(torch.cat(feats,dim=1))
        expanded = projected.unsqueeze(1).repeat(1,T,1)
        expanded = self.temporal_cnn(expanded.transpose(1,2)).transpose(1,2)
        return self.ln(expanded)

class TransformerEncoder(nn.Module):
    def __init__(self,d_model=64,nhead=4,layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,batch_first=True,dim_feedforward=d_model*4,activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer,layers)
    def forward(self,x,mask):
        x = self.encoder(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return (x*m).sum(dim=1)/m.sum(dim=1).clamp(min=1)

class HybridFusedModel(nn.Module):
    def __init__(self,num_classes,acc_feat_dim,rot_dim=4,d_model=64):
        super().__init__()
        self.use_acc = USE_ACC
        self.use_rot = USE_ROT
        self.acc_encoder = EnhancedCNNEncoder(acc_feat_dim,d_model) if USE_ACC else None
        self.rot_encoder = EnhancedCNNEncoder(rot_dim,d_model) if USE_ROT else None
        self.acc_trans = TransformerEncoder(d_model) if USE_TRANSFORMER and USE_ACC else None
        self.rot_trans = TransformerEncoder(d_model) if USE_TRANSFORMER and USE_ROT else None
        self.fusion = nn.Sequential(nn.LayerNorm(d_model*2), nn.Linear(d_model*2,d_model), nn.GELU(), nn.Dropout(DROPOUT))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model,d_model//2), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(d_model//2,num_classes))
    def forward(self, acc, rot, mask):
        feats=[]
        if self.use_acc:
            acc_feat = self.acc_encoder(acc)
            acc_out = self.acc_trans(acc_feat,mask) if self.acc_trans else (acc_feat*mask.unsqueeze(-1).float()).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
            feats.append(acc_out)
        if self.use_rot:
            rot_feat = self.rot_encoder(rot)
            rot_out = self.rot_trans(rot_feat,mask) if self.rot_trans else (rot_feat*mask.unsqueeze(-1).float()).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
            feats.append(rot_out)
        if len(feats)==2:
            z = torch.cat(feats,dim=1)
            z = self.fusion(z)
        else:
            z = feats[0]
        return self.head(z)

# ============================================================
# TRAINING
# ============================================================
WINDOW_BEFORE = 20
WINDOW_AFTER = 20
EPOCHS = 20
USE_AXISCNN = False
USE_TRANSFORMER = True
FUSION = 'concat'  # 'concat' or 'gated'
USE_ACC = True
USE_ROT = False

torch.manual_seed(SEED)
dataset = AccRotHybridDataset(CSV_PATH,WINDOW_BEFORE,WINDOW_AFTER)
num_classes = len(dataset.gesture_classes)
acc_feat_dim = dataset[0]["acc"].shape[1] if USE_ACC else 0

train_size = int((1-VAL_RATIO)*len(dataset)); test_size=len(dataset)-train_size
train_ds, test_ds = random_split(dataset,[train_size,test_size])
train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=BATCH_SIZE)

model = HybridFusedModel(num_classes,acc_feat_dim).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(),lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train(); total_loss=0
    for batch in train_loader:
        acc=batch["acc"].to(DEVICE) if USE_ACC else None
        rot=batch["rot"].to(DEVICE) if USE_ROT else None
        mask=batch["mask"].to(DEVICE); label=batch["label"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(acc,rot,mask)
        loss = criterion(logits,label)
        loss.backward(); optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

# ============================================================
# EVALUATION
# ============================================================


model.eval()
all_preds, all_labels=[],[]
with torch.no_grad():
    for batch in test_loader:
        acc=batch["acc"].to(DEVICE) if USE_ACC else None
        rot=batch["rot"].to(DEVICE) if USE_ROT else None
        mask=batch["mask"].to(DEVICE)
        logits = model(acc,rot,mask)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].numpy())

all_preds=np.array(all_preds); all_labels=np.array(all_labels)
overall_acc = (all_preds==all_labels).mean()
print(f"\nOverall accuracy: {overall_acc:.3f}")



# ============================================================
# HYBRID ACC + ROT + ToF MODEL (KAGGLE, SINGLE FILE)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.transform import Rotation as R
import random

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"

WINDOW_BEFORE = 20
WINDOW_AFTER = 20
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4

D_MODEL = 64
AXIS_HIDDEN_DIM = 32
DROPOUT = 0.1

USE_ACC = True
USE_ROT = False
USE_TOF = True          # ⭐ NEW
FREEZE_TOF = False     # freeze autoencoder encoder

VAL_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# ToF AUTOENCODER
# ============================================================

class ToFAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# ============================================================
# HELPERS
# ============================================================

def handle_quaternion_missing_values(rot):
    out = rot.copy()
    for i in range(len(rot)):
        r = rot[i]
        if np.isnan(r).sum() == 0:
            n = np.linalg.norm(r)
            out[i] = r / n if n > 1e-6 else [1,0,0,0]
        else:
            out[i] = [1,0,0,0]
    return out

def compute_world_acc(acc, rot):
    try:
        r = R.from_quat(rot[:, [1,2,3,0]])
        return r.apply(acc)
    except:
        return acc.copy()

def remove_gravity(acc, rot):
    g = np.array([0,0,9.81])
    out = np.zeros_like(acc)
    for i in range(len(acc)):
        try:
            r = R.from_quat(rot[i])
            out[i] = acc[i] - r.apply(g, inverse=True)
        except:
            out[i] = acc[i]
    return out

# ============================================================
# DATASET
# ============================================================

class HybridDataset(Dataset):

    ACC_COLS = ["acc_x","acc_y","acc_z"]
    ROT_COLS = ["rot_w","rot_x","rot_y","rot_z"]

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # labels
        df["gesture"] = df["gesture"].astype(str)
        self.le = LabelEncoder()
        df["label"] = self.le.fit_transform(df["gesture"])
        self.classes = self.le.classes_

        # ToF
        self.tof_cols = [c for c in df.columns if c.startswith("tof")]
        self.use_tof = len(self.tof_cols) > 0

        if self.use_tof:
            scaler = StandardScaler()
            df[self.tof_cols] = scaler.fit_transform(
                df[self.tof_cols].fillna(0.0)
            )

        self.windows = []
        win_len = WINDOW_BEFORE + WINDOW_AFTER + 1

        for _, seq in df.groupby("sequence_id"):
            seq = seq.reset_index(drop=True)
            if "Gesture" not in seq["phase"].values:
                continue

            center = seq[seq["phase"]=="Gesture"].index[0]
            s = max(0, center - WINDOW_BEFORE)
            e = min(len(seq), center + WINDOW_AFTER + 1)

            # ACC
            acc = seq.iloc[s:e][self.ACC_COLS].values.astype(np.float32)
            rot = seq.iloc[s:e][self.ROT_COLS].values.astype(np.float32)

            rot = handle_quaternion_missing_values(rot)
            acc_w = compute_world_acc(acc, rot)
            acc_l = remove_gravity(acc, rot)

            acc_feat = np.concatenate([acc_w, acc_l], axis=1)

            acc_pad = np.zeros((win_len, acc_feat.shape[1]), np.float32)
            rot_pad = np.zeros((win_len, 4), np.float32)
            mask = np.zeros(win_len, bool)

            insert = WINDOW_BEFORE - (center - s)
            acc_pad[insert:insert+len(acc_feat)] = acc_feat
            rot_pad[insert:insert+len(rot)] = rot
            mask[insert:insert+len(acc_feat)] = True

            # ToF (center frame only)
            if self.use_tof:
                tof = seq.iloc[center][self.tof_cols].values.astype(np.float32)
            else:
                tof = np.zeros(1, dtype=np.float32)

            label = seq["label"].mode().iloc[0]
            self.windows.append((acc_pad, rot_pad, tof, mask, label))

        print(f"Created {len(self.windows)} samples")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        acc, rot, tof, mask, label = self.windows[idx]

        return {
            "acc": torch.tensor(acc),
            "rot": torch.tensor(rot),
            "tof": torch.tensor(tof),
            "mask": torch.tensor(mask),
            "label": torch.tensor(label)
        }

# ============================================================
# ENCODERS
# ============================================================

class AxisCNN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x).mean(-1)

class CNNEncoder(nn.Module):
    def __init__(self, feat_dim, d_model):
        super().__init__()
        self.cnns = nn.ModuleList([AxisCNN() for _ in range(feat_dim)])
        self.proj = nn.Linear(feat_dim * 32, d_model)

    def forward(self, x):
        feats = []
        for i in range(x.shape[2]):
            feats.append(self.cnns[i](x[:,:,i:i+1].transpose(1,2)))
        return self.proj(torch.cat(feats, dim=1))

# ============================================================
# FULL MODEL
# ============================================================

class HybridModel(nn.Module):
    def __init__(self, num_classes, acc_dim, tof_dim):
        super().__init__()

        self.acc_enc = CNNEncoder(acc_dim, D_MODEL) if USE_ACC else None
        self.rot_enc = CNNEncoder(4, D_MODEL) if USE_ROT else None

        self.tof_ae = ToFAutoencoder(tof_dim) if USE_TOF else None
        self.tof_proj = nn.Linear(8, D_MODEL) if USE_TOF else None

        fusion_dim = D_MODEL * sum([USE_ACC, USE_ROT, USE_TOF])
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, num_classes)
        )

    def forward(self, acc, rot, tof):
        feats = []

        if USE_ACC:
            feats.append(self.acc_enc(acc))
        if USE_ROT:
            feats.append(self.rot_enc(rot))
        if USE_TOF:
            _, z = self.tof_ae(tof)
            feats.append(self.tof_proj(z))

        z = torch.cat(feats, dim=1)
        return self.head(z)

# ============================================================
# TRAIN
# ============================================================

dataset = HybridDataset(CSV_PATH)
num_classes = len(dataset.classes)
acc_dim = dataset[0]["acc"].shape[1]
tof_dim = dataset[0]["tof"].shape[0]

train_size = int((1 - VAL_RATIO) * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = HybridModel(num_classes, acc_dim, tof_dim).to(DEVICE)

if USE_TOF and FREEZE_TOF:
    for p in model.tof_ae.encoder.parameters():
        p.requires_grad = False

opt = torch.optim.AdamW(model.parameters(), lr=LR)
clf_loss = nn.CrossEntropyLoss()
tof_loss = nn.MSELoss()

# ============================================================
# LOOP
# ============================================================

for epoch in range(EPOCHS):
    model.train()
    total = 0

    for b in train_loader:
        acc = b["acc"].to(DEVICE)
        rot = b["rot"].to(DEVICE)
        tof = b["tof"].to(DEVICE)
        y = b["label"].to(DEVICE)

        opt.zero_grad()
        logits = model(acc, rot, tof)
        loss = clf_loss(logits, y)

        if USE_TOF:
            recon, _ = model.tof_ae(tof)
            loss = loss + 0.1 * tof_loss(recon, tof)

        loss.backward()
        opt.step()
        total += loss.item()

    print(f"Epoch {epoch+1} | Loss {total/len(train_loader):.4f}")

# ============================================================
# EVAL
# ============================================================

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for b in val_loader:
        logits = model(
            b["acc"].to(DEVICE),
            b["rot"].to(DEVICE),
            b["tof"].to(DEVICE)
        )
        pred = logits.argmax(1)
        correct += (pred.cpu() == b["label"]).sum().item()
        total += len(pred)

print(f"\nValidation accuracy: {correct/total:.3f}")



# ============================================================
# HYBRID ACC + ROT + ToF MODEL (KAGGLE, SINGLE FILE)
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.transform import Rotation as R
import random

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"

WINDOW_BEFORE = 20
WINDOW_AFTER = 20
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4

D_MODEL = 64
AXIS_HIDDEN_DIM = 32
DROPOUT = 0.1

USE_ACC = True
USE_ROT = False
USE_TOF = True          # ⭐ NEW
FREEZE_TOF = False     # freeze autoencoder encoder

VAL_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# ToF AUTOENCODER
# ============================================================

class ToFAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# ============================================================
# HELPERS
# ============================================================

def handle_quaternion_missing_values(rot):
    out = rot.copy()
    for i in range(len(rot)):
        r = rot[i]
        if np.isnan(r).sum() == 0:
            n = np.linalg.norm(r)
            out[i] = r / n if n > 1e-6 else [1,0,0,0]
        else:
            out[i] = [1,0,0,0]
    return out

def compute_world_acc(acc, rot):
    try:
        r = R.from_quat(rot[:, [1,2,3,0]])
        return r.apply(acc)
    except:
        return acc.copy()

def remove_gravity(acc, rot):
    g = np.array([0,0,9.81])
    out = np.zeros_like(acc)
    for i in range(len(acc)):
        try:
            r = R.from_quat(rot[i])
            out[i] = acc[i] - r.apply(g, inverse=True)
        except:
            out[i] = acc[i]
    return out

# ============================================================
# DATASET
# ============================================================

class HybridDataset(Dataset):

    ACC_COLS = ["acc_x","acc_y","acc_z"]
    ROT_COLS = ["rot_w","rot_x","rot_y","rot_z"]

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # labels
        df["gesture"] = df["gesture"].astype(str)
        self.le = LabelEncoder()
        df["label"] = self.le.fit_transform(df["gesture"])
        self.classes = self.le.classes_

        # ToF
        self.tof_cols = [c for c in df.columns if c.startswith("tof")]
        self.use_tof = len(self.tof_cols) > 0

        if self.use_tof:
            scaler = StandardScaler()
            df[self.tof_cols] = scaler.fit_transform(
                df[self.tof_cols].fillna(0.0)
            )

        self.windows = []
        win_len = WINDOW_BEFORE + WINDOW_AFTER + 1

        for _, seq in df.groupby("sequence_id"):
            seq = seq.reset_index(drop=True)
            if "Gesture" not in seq["phase"].values:
                continue

            center = seq[seq["phase"]=="Gesture"].index[0]
            s = max(0, center - WINDOW_BEFORE)
            e = min(len(seq), center + WINDOW_AFTER + 1)

            # ACC
            acc = seq.iloc[s:e][self.ACC_COLS].values.astype(np.float32)
            rot = seq.iloc[s:e][self.ROT_COLS].values.astype(np.float32)

            rot = handle_quaternion_missing_values(rot)
            acc_w = compute_world_acc(acc, rot)
            acc_l = remove_gravity(acc, rot)

            acc_feat = np.concatenate([acc_w, acc_l], axis=1)

            acc_pad = np.zeros((win_len, acc_feat.shape[1]), np.float32)
            rot_pad = np.zeros((win_len, 4), np.float32)
            mask = np.zeros(win_len, bool)

            insert = WINDOW_BEFORE - (center - s)
            acc_pad[insert:insert+len(acc_feat)] = acc_feat
            rot_pad[insert:insert+len(rot)] = rot
            mask[insert:insert+len(acc_feat)] = True

            # ToF (center frame only)
            if self.use_tof:
                tof = seq.iloc[center][self.tof_cols].values.astype(np.float32)
            else:
                tof = np.zeros(1, dtype=np.float32)

            label = seq["label"].mode().iloc[0]
            self.windows.append((acc_pad, rot_pad, tof, mask, label))

        print(f"Created {len(self.windows)} samples")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        acc, rot, tof, mask, label = self.windows[idx]

        return {
            "acc": torch.tensor(acc),
            "rot": torch.tensor(rot),
            "tof": torch.tensor(tof),
            "mask": torch.tensor(mask),
            "label": torch.tensor(label)
        }

# ============================================================
# ENCODERS
# ============================================================

class AxisCNN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x).mean(-1)

class CNNEncoder(nn.Module):
    def __init__(self, feat_dim, d_model):
        super().__init__()
        self.cnns = nn.ModuleList([AxisCNN() for _ in range(feat_dim)])
        self.proj = nn.Linear(feat_dim * 32, d_model)

    def forward(self, x):
        feats = []
        for i in range(x.shape[2]):
            feats.append(self.cnns[i](x[:,:,i:i+1].transpose(1,2)))
        return self.proj(torch.cat(feats, dim=1))

# ============================================================
# FULL MODEL
# ============================================================

class HybridModel(nn.Module):
    def __init__(self, num_classes, acc_dim, tof_dim):
        super().__init__()

        self.acc_enc = CNNEncoder(acc_dim, D_MODEL) if USE_ACC else None
        self.rot_enc = CNNEncoder(4, D_MODEL) if USE_ROT else None

        self.tof_ae = ToFAutoencoder(tof_dim) if USE_TOF else None
        self.tof_proj = nn.Linear(8, D_MODEL) if USE_TOF else None

        fusion_dim = D_MODEL * sum([USE_ACC, USE_ROT, USE_TOF])
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, num_classes)
        )

    def forward(self, acc, rot, tof):
        feats = []

        if USE_ACC:
            feats.append(self.acc_enc(acc))
        if USE_ROT:
            feats.append(self.rot_enc(rot))
        if USE_TOF:
            _, z = self.tof_ae(tof)
            feats.append(self.tof_proj(z))

        z = torch.cat(feats, dim=1)
        return self.head(z)

# ============================================================
# TRAIN
# ============================================================

dataset = HybridDataset(CSV_PATH)
num_classes = len(dataset.classes)
acc_dim = dataset[0]["acc"].shape[1]
tof_dim = dataset[0]["tof"].shape[0]

train_size = int((1 - VAL_RATIO) * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = HybridModel(num_classes, acc_dim, tof_dim).to(DEVICE)

if USE_TOF and FREEZE_TOF:
    for p in model.tof_ae.encoder.parameters():
        p.requires_grad = False

opt = torch.optim.AdamW(model.parameters(), lr=LR)
clf_loss = nn.CrossEntropyLoss()
tof_loss = nn.MSELoss()

# ============================================================
# LOOP
# ============================================================

for epoch in range(EPOCHS):
    model.train()
    total = 0

    for b in train_loader:
        acc = b["acc"].to(DEVICE)
        rot = b["rot"].to(DEVICE)
        tof = b["tof"].to(DEVICE)
        y = b["label"].to(DEVICE)

        opt.zero_grad()
        logits = model(acc, rot, tof)
        loss = clf_loss(logits, y)

        if USE_TOF:
            recon, _ = model.tof_ae(tof)
            loss = loss + 0.1 * tof_loss(recon, tof)

        loss.backward()
        opt.step()
        total += loss.item()

    print(f"Epoch {epoch+1} | Loss {total/len(train_loader):.4f}")

# ============================================================
# EVAL
# ============================================================

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for b in val_loader:
        logits = model(
            b["acc"].to(DEVICE),
            b["rot"].to(DEVICE),
            b["tof"].to(DEVICE)
        )
        pred = logits.argmax(1)
        correct += (pred.cpu() == b["label"]).sum().item()
        total += len(pred)

print(f"\nValidation accuracy: {correct/total:.3f}")



for i, (data, target) in enumerate(test_loader):
    if i >= 3:  # Test 3 samples
        break
    
    data = data.to(DEVICE)
    with torch.no_grad():
        output = model(data)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1)
        
    true_label = dataset.classes[target.item()]
    pred_label = dataset.classes[pred.item()]
    confidence = probs[0, pred.item()].item()
    print(pred.item())
    print(f"Sample {i+1}: True={true_label:10s} | Pred={pred_label:10s} | "
          f"Confidence={confidence:.3f} | "
          f"{'CORRECT' if pred.item() == target.item() else 'WRONG'}")


def save_model_for_submission(model, label_encoder, feature_dim=16):
    """
    Save the trained model and label encoder for Kaggle submission.
    
    Args:
        model: Trained PyTorch model
        label_encoder: Fitted LabelEncoder
        feature_dim: Number of input features (should be 16)
    """
    import os
    import joblib
    
    # Create output directory
    os.makedirs("/kaggle/working", exist_ok=True)
    
    # 1. Save model weights
    model_path = "/kaggle/working/inceptiontime_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # 2. Save label encoder
    encoder_path = "/kaggle/working/label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path)
    print(f"✅ Label encoder saved to: {encoder_path}")
    
    # 3. Save metadata
    metadata = {
        'feature_dim': feature_dim,
        'window_before': WINDOW_BEFORE,
        'window_after': WINDOW_AFTER,
        'window_len': WINDOW_LEN,
        'classes': label_encoder.classes_.tolist(),
        'num_classes': len(label_encoder.classes_)
    }
    metadata_path = "/kaggle/working/model_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    # 4. Create a requirements.txt for reproducibility
    requirements = """torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
polars>=0.19.0
scikit-learn>=1.3.0
joblib>=1.3.0
"""
    
    with open("/kaggle/working/requirements.txt", "w") as f:
        f.write(requirements)
    print(f"✅ Requirements file saved to: /kaggle/working/requirements.txt")
save_model_for_submission(model, dataset.le, feature_dim=16)


print(dataset.classes)


import os

import polars as pl

import kaggle_evaluation.cmi_inference_server

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # Replace this function with your inference code.
    # You can return either a Pandas or Polars dataframe, though Polars is recommended.
    # Each prediction (except the very first) must be returned within 30 minutes of the batch features being provided.
    classes = ['Above ear - pull hair' 'Cheek - pinch skin' 'Drink from bottle/cup'
 'Eyebrow - pull hair' 'Eyelash - pull hair'
 'Feel around in tray and pull out an object' 'Forehead - pull hairline'
 'Forehead - scratch' 'Glasses on/off' 'Neck - pinch skin'
 'Neck - scratch' 'Pinch knee/leg skin' 'Pull air toward your face'
 'Scratch knee/leg skin' 'Text on phone' 'Wave hello' 'Write name in air'
 'Write name on leg']
    data = sequence.to(DEVICE)
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
        
    pred_label = classes[pred.item()]
    return pred_label


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

