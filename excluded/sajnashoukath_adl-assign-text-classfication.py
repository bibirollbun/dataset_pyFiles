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



!pip install -q "protobuf<4" transformers accelerate




import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Seed helper
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Paths (Kaggle default)
DATA_DIR = Path("/kaggle/input/jigsaw-toxic-comment-classification-challenge")
assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"

# Read files (pandas can read zipped csvs)
train_df = pd.read_csv(DATA_DIR / "train.csv.zip")
test_df  = pd.read_csv(DATA_DIR / "test.csv.zip")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv.zip")

# quick check
print("Train rows:", len(train_df))
print("Test rows :", len(test_df))
print("Sample sub rows:", len(sample_sub))



# Cell 2 - constants and small validation split
TARGET_COLUMNS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# small stratified validation split on 'toxic' (keeps distribution)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.05,
    random_state=42,
    stratify=train_df["toxic"]
)

print("Split sizes -> train:", len(train_df), "val:", len(val_df), "test:", len(test_df))

# Hyperparameters (tweak for debugging / final run)
MODEL_NAME = "roberta-base"     # change to a larger/smaller model if needed
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3                      # set to 1 for quick debug
LR = 2e-5
NUM_WORKERS = 2
SEEDS = [1, 2, 3]               # seeds to train and ensemble



# Cell 3 - Dataset wrapper
class JigsawDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, include_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_labels = include_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["comment_text"]) if pd.notna(row["comment_text"]) else ""
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }
        if self.include_labels:
            labels = torch.tensor(row[TARGET_COLUMNS].values.astype(float), dtype=torch.float32)
            item["labels"] = labels
        return item



## Model definition

# Cell 4 - Model
class ToxicModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(cfg.hidden_size, len(TARGET_COLUMNS))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]   # CLS pooling
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits



# Cell 5 - Loss functions & class-imbalance helpers

# Compute pos_weight for each label: pos_weight = (N - pos) / pos
label_counts = train_df[TARGET_COLUMNS].sum(axis=0).values.astype(float)
n_samples = len(train_df)
pos_weight_arr = (n_samples - label_counts) / (label_counts + 1e-12)   # small eps to avoid div by 0
pos_weight_tensor = torch.tensor(pos_weight_arr, dtype=torch.float32).to(device)
print("Label counts:", dict(zip(TARGET_COLUMNS, label_counts.astype(int))))
print("pos_weight per label:", pos_weight_arr)

# BCEWithLogitsLoss with pos_weight (this multiplies the positive examples' loss per class)
bce_pos_weight_loss = lambda: nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

# Optional: Focal Loss for multi-label (works on logits)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Multi-label focal loss based on BCE with logits.
        alpha: None or array-like of shape (num_classes,) for class-wise alpha scaling
        gamma: focusing parameter
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device) if alpha is not None else None
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B, C)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (B,C)
        p_t = torch.exp(-bce)  # p_t = sigmoid when target==1 else 1-sigmoid
        modulating_factor = (1 - p_t) ** self.gamma
        loss = modulating_factor * bce
        if self.alpha is not None:
            loss = loss * self.alpha  # broadcast over batch
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Choose which loss to use by instantiating one:
# For pos_weight BCE:
loss_fn = bce_pos_weight_loss()    # recommended baseline
# Or for focal loss (uncomment to use):
# focal_alpha = (1.0 / (label_counts + 1e-12))  # example alpha inversely proportional to freq
# loss_fn = FocalLoss(gamma=2.0, alpha=focal_alpha, reduction='mean')



# Cell 6 - train and evaluate utilities
def train_one_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad()
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * ids.size(0)
        if step % 200 == 0:
            print(f"  step {step}/{len(loader)} loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    elapsed = time.time() - start
    print(f"  epoch finished avg_loss={avg_loss:.4f} time={elapsed:.1f}s")
    return avg_loss

def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(ids, mask).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_labels.append(labels)
            all_probs.append(probs)

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_probs)

    aucs = []
    for j in range(y_true.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true[:, j], y_pred[:, j]))
        except ValueError:
            aucs.append(np.nan)
    mean_auc = np.nanmean(aucs)

    preds_bin = (y_pred >= 0.5).astype(int)
    per_label_acc = (preds_bin == y_true).mean(axis=0).tolist()
    micro_acc = (preds_bin == y_true).mean()
    subset_acc = (preds_bin == y_true).all(axis=1).mean()

    print("  AUC per label:", [round(x,4) if not np.isnan(x) else None for x in aucs])
    print("  mean AUC:", round(mean_auc,6))
    print("  per-label acc:", [round(x,4) for x in per_label_acc])
    print("  micro_acc:", round(micro_acc,6), "subset_acc:", round(subset_acc,6))
    return {"aucs": aucs, "mean_auc": mean_auc, "per_label_acc": per_label_acc, "micro_acc": micro_acc, "subset_acc": subset_acc}



# Cell 7 - run one seed (train EPOCHS, save checkpoint, save logits)
def run_seed(seed, model_name=MODEL_NAME, save_prefix="seed"):
    print(f"\n=== Starting seed {seed} ===")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = JigsawDataset(train_df, tokenizer, MAX_LEN, include_labels=True)
    val_ds   = JigsawDataset(val_df, tokenizer, MAX_LEN, include_labels=True)
    test_ds  = JigsawDataset(test_df, tokenizer, MAX_LEN, include_labels=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ToxicModel(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = loss_fn

    best_val = -np.inf
    best_path = f"best_{save_prefix}_{seed}.pt"

    for epoch in range(1, EPOCHS+1):
        print(f"\nSeed {seed} - Epoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
        metrics = evaluate(model, val_loader)
        if metrics["mean_auc"] > best_val or best_val == -np.inf:
            best_val = metrics["mean_auc"]
            torch.save(model.state_dict(), best_path)
            print(f"  Saved best model to {best_path} (mean_auc={best_val:.6f})")

    # Load best and predict test logits
    model.load_state_dict(torch.load(best_path))
    model.eval()
    logits_list = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask).cpu().numpy()
            logits_list.append(logits)
    all_logits = np.vstack(logits_list)
    np.save(f"logits_{save_prefix}_{seed}.npy", all_logits)
    print(f"Saved logits_{save_prefix}_{seed}.npy shape={all_logits.shape}")
    return all_logits



# Cell 8 - run seeds and create submission
all_logits = []
for s in SEEDS:
    lg = run_seed(s, model_name=MODEL_NAME, save_prefix="ens")
    all_logits.append(lg)

# if you'd prefer to load existing saved logits instead of retraining, uncomment:
# all_logits = [np.load(f"logits_ens_{s}.npy") for s in SEEDS]

# Ensure shapes align
for i,lg in enumerate(all_logits):
    print(f"seed {SEEDS[i]} logits shape:", lg.shape)
    assert lg.shape[1] == len(TARGET_COLUMNS), "logits must have same number of columns as TARGET_COLUMNS"

# Average logits (soft ensemble), then convert to probabilities
avg_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
probs = 1.0 / (1.0 + np.exp(-avg_logits))





