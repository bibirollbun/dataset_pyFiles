# ============================================================
# 1. Imports & basic setup
# ============================================================
import os
import re
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

print("Torch version:", torch.__version__)

# ============================================================
# 2. Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 3. Configuration
# ============================================================
class Config:
    data_dir = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
    model_name = "roberta-base"  # safe: already works in your env
    max_len = 220                # longer sequence than 160
    train_batch_size = 16
    valid_batch_size = 32
    grad_accum_steps = 2         # effective batch size 32
    epochs = 2                   # you can try 3 if you have time
    lr = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    num_workers = 2

CFG = Config()

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ============================================================
# 4. Load & preprocess data
# ============================================================
train_path = os.path.join(CFG.data_dir, "train.csv.zip")
test_path = os.path.join(CFG.data_dir, "test.csv.zip")
sample_sub_path = os.path.join(CFG.data_dir, "sample_submission.csv.zip")

train_df_full = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)

print("Original train shape:", train_df_full.shape)
print("Test shape:", test_df.shape)

def clean_text(text: str) -> str:
    text = str(text)
    text = text.lower()
    # replace URLs with special token
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    # remove line breaks
    text = re.sub(r"\n+", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df_full["comment_text"] = train_df_full["comment_text"].fillna("none").apply(clean_text)
test_df["comment_text"] = test_df["comment_text"].fillna("none").apply(clean_text)

# for stratified split: any toxic label
train_df_full["any_toxic"] = (train_df_full[label_cols].sum(axis=1) > 0).astype(int)

train_df, val_df = train_test_split(
    train_df_full,
    test_size=0.1,
    random_state=SEED,
    shuffle=True,
    stratify=train_df_full["any_toxic"],
)

print("Train rows:", len(train_df), "Val rows:", len(val_df))

# ============================================================
# 5. Dataset class
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, with_labels=True):
        self.texts = df["comment_text"].tolist()
        self.with_labels = with_labels
        self.max_len = max_len
        self.tokenizer = tokenizer
        if with_labels:
            self.labels = df[label_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.with_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item

train_dataset = JigsawDataset(train_df, tokenizer, CFG.max_len, with_labels=True)
val_dataset = JigsawDataset(val_df, tokenizer, CFG.max_len, with_labels=True)
test_dataset = JigsawDataset(test_df, tokenizer, CFG.max_len, with_labels=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.train_batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.valid_batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.valid_batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
)

print("Train steps per epoch:", len(train_loader))

# ============================================================
# 6. Model definition
# ============================================================
config = AutoConfig.from_pretrained(CFG.model_name)
config.num_labels = len(label_cols)
config.problem_type = "multi_label_classification"

model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name, config=config)
model.to(device)

# ============================================================
# 7. Optimizer, scheduler, loss, AMP
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

num_update_steps_per_epoch = len(train_loader) // CFG.grad_accum_steps + int(len(train_loader) % CFG.grad_accum_steps > 0)
num_training_steps = num_update_steps_per_epoch * CFG.epochs
num_warmup_steps = int(CFG.warmup_ratio * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

criterion = nn.BCEWithLogitsLoss()

use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# ============================================================
# 8. Train & validation loops
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss = loss / CFG.grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % CFG.grad_accum_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item() * input_ids.size(0)

        if (step + 1) % 500 == 0 or (step + 1) == len(loader):
            print(
                f"Epoch {epoch} - Step {step+1}/{len(loader)} "
                f"- loss: {loss.item() * CFG.grad_accum_steps:.4f}"
            )

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            running_loss += loss.item() * input_ids.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    roc_aucs = []
    per_label = {}
    for i, col in enumerate(label_cols):
        try:
            score = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            score = np.nan
        roc_aucs.append(score)
        per_label[col] = score

    mean_roc_auc = np.nanmean(roc_aucs)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, mean_roc_auc, per_label

best_val_auc = 0.0
best_state_dict = None

for epoch in range(1, CFG.epochs + 1):
    print("=" * 70)
    print(f"Epoch {epoch}/{CFG.epochs}")
    print("=" * 70)

    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
    val_loss, val_auc, per_label_auc = evaluate(model, val_loader, device)

    print("-" * 70)
    print(f"Train loss: {train_loss:.5f}")
    print(f"Val   loss: {val_loss:.5f}")
    print(f"Val ROC-AUC (mean): {val_auc:.6f}")
    print("Per-label ROC-AUC:")
    for k, v in per_label_auc.items():
        print(f"  {k:13s}: {v:.6f}")
    print("-" * 70)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state_dict = model.state_dict().copy()
        print(f"--> New best validation ROC-AUC: {best_val_auc:.6f}")

print("Best validation ROC-AUC:", best_val_auc)

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# ============================================================
# 9. Predict on test & create submission
# ============================================================
model.eval()
test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        test_preds.append(probs)

test_preds = np.concatenate(test_preds, axis=0)

submission = sample_submission.copy()
submission[label_cols] = test_preds
submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", submission.shape)
print(submission.head())

