# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/quora-question-pairs/train.csv.zip",)
train_df.head()


from statistics import mode

# Tính độ dài từng câu
lens_q1 = train_df["question1"].dropna().str.split().apply(len)
lens_q2 = train_df["question2"].dropna().str.split().apply(len)

# Max length
max_q1, max_q2 = lens_q1.max(), lens_q2.max()

# Mode length
mode_q1, mode_q2 = mode(lens_q1), mode(lens_q2)

print(f"question1 -> max: {max_q1}, mode: {mode_q1}")
print(f"question2 -> max: {max_q2}, mode: {mode_q2}")


train_df.info()


train_df['is_duplicate'].unique()


test_df = pd.read_csv("/kaggle/input/quora-question-pairs/test.csv",)
test_df.head()


test_df.info()


# DistilBERT for Quora Question Pairs (Kaggle)
# - Trains on a 10k sample to quickly gauge effectiveness
# - Switch between FP32 and FP16 with a single flag
# - Measures average batch time and estimates total training time BEFORE full training
#
# How to use (Kaggle Notebook):
# 1) Ensure the dataset is available at /kaggle/input/quora-question-pairs/train.csv
# 2) Toggle USE_FP16 = True/False below
# 3) Run all cells. The script will first estimate training time, then train, then evaluate.


import os
import time
import math
import random
import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


# =========================
# Configuration
# =========================
@dataclass
class Config:
    model_name: str = "/kaggle/input/vietledsdistilbert-base-uncased/distilbert-base-uncased"
    seed: int = 42
    use_fp16: bool = True            # <<< switch here: True = FP16, False = FP32
    max_length: int = 160
    sample_size: int = 500_000        # limit train set to 10k rows as requested
    valid_ratio: float = 0.1
    batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_workers: int = 2
    dataset_path: str = "/kaggle/input/quora-question-pairs/train.csv.zip"
    test_path = "/kaggle/input/quora-question-pairs/test.csv"
    submission_path = '/kaggle/working/submission.csv'
    test_size = None
    finetuning = False

CFG = Config()


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG.seed)


# =========================
# Device & Precision
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

if CFG.use_fp16 and DEVICE.type != "cuda":
    print("[Warning] FP16 requested but CUDA not available; falling back to FP32.")
    CFG.use_fp16 = False


# =========================
# Load data (limit to 10k rows)
# =========================
assert os.path.exists(CFG.dataset_path), f"Dataset not found at {CFG.dataset_path}"
df = pd.read_csv(CFG.dataset_path)
# Columns: id, qid1, qid2, question1, question2, is_duplicate


# Basic cleanup
df = df.dropna(subset=["question1", "question2", "is_duplicate"]).copy()
df["is_duplicate"] = df["is_duplicate"].astype(int)

# Sample 10k rows for a quick run
if len(df) > CFG.sample_size:
    df = df.sample(n=CFG.sample_size, random_state=CFG.seed)

# Train/valid split
train_df, valid_df = train_test_split(
    df,
    test_size=CFG.valid_ratio,
    random_state=CFG.seed,
    stratify=df["is_duplicate"],
)
print(f"Train size: {len(train_df)} | Valid size: {len(valid_df)}")


# =========================
# Tokenizer & Dataset
# =========================
Tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)


class QQPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, test_mode: bool = False):
        self.q1 = df["question1"].tolist()
        self.q2 = df["question2"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_mode = test_mode
        if not self.test_mode:
            self.labels = df["is_duplicate"].tolist()
        else:
            self.ids = df["test_id"].tolist()

    def __len__(self):
        return len(self.q1)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.q1[idx],
            self.q2[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if not self.test_mode:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            item["ids"] = self.ids[idx]
        return item

train_dataset = QQPDataset(train_df, Tokenizer, CFG.max_length)
valid_dataset = QQPDataset(valid_df, Tokenizer, CFG.max_length)


collator = DataCollatorWithPadding(tokenizer=Tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=(DEVICE.type == "cuda"),
    collate_fn=collator,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=(DEVICE.type == "cuda"),
    collate_fn=collator,
)


# =========================
# Model & Optimizer & Scheduler
# =========================
model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name, num_labels=2)
model.to(DEVICE)

# Typical setup
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)

total_steps = len(train_loader) * CFG.epochs
warmup_steps = max(1, int(CFG.warmup_ratio * total_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_fp16)


# =========================
# Utility: Estimate average batch time BEFORE training
# =========================
@torch.no_grad()
def _forward_pass(model, batch):
    return model(**batch)


def estimate_batch_time(model, loader, steps: int = 50, warmup: int = 5) -> float:
    """Run a few forward+backward steps to estimate avg batch time (in seconds).
    We actually include backward pass to reflect realistic cost.
    """
    model.train()

    # We'll clone a few batches
    batches = []
    for i, batch in enumerate(loader):
        if i >= steps:
            break
        # Move to device
        for k in batch:
            batch[k] = batch[k].to(DEVICE, non_blocking=True)
        batches.append(batch)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []
    # Use a temporary optimizer/scaler to avoid touching real states
    tmp_optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    tmp_scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_fp16)

    for i, batch in enumerate(batches):
        start = time.time()
        tmp_optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=CFG.use_fp16):
            outputs = model(**batch)
            loss = outputs.loss
        if CFG.use_fp16:
            tmp_scaler.scale(loss).backward()
            tmp_scaler.step(tmp_optimizer)
            tmp_scaler.update()
        else:
            loss.backward()
            tmp_optimizer.step()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start
        if i >= warmup:  # discard warmup
            times.append(elapsed)

    # Clean temporary optimizer states
    del tmp_optimizer, tmp_scaler
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    avg_time = float(np.mean(times)) if times else 0.0
    return avg_time

# --- Estimate & print ---
avg_batch_sec = estimate_batch_time(model, train_loader, steps=min(50, len(train_loader)), warmup=5)
num_batches_total = len(train_loader) * CFG.epochs
eta_total_sec = avg_batch_sec * num_batches_total

mins, secs = divmod(int(eta_total_sec), 60)
hrs, mins = divmod(mins, 60)
print(f"\n[Time Estimate] Avg seconds/batch: {avg_batch_sec:.3f}")
print(f"[Time Estimate] Total batches (@{CFG.epochs} epochs): {num_batches_total}")
print(f"[Time Estimate] Estimated total training time: {hrs}h {mins}m {secs}s\n")



# =========================
# Training Loop
# =========================

def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(loader, 1):
        # Move to device
        for k in batch:
            batch[k] = batch[k].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=CFG.use_fp16):
            outputs = model(**batch)
            loss = outputs.loss

        if CFG.use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        running_loss += loss.item()
        if step % 100 == 0 or step == 1:
            avg = running_loss / step
            print(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss: {avg:.4f}")

    return running_loss / max(1, len(loader))


def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].numpy()
            for k in batch:
                batch[k] = batch[k].to(DEVICE, non_blocking=True)
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()

            all_labels.append(labels)
            all_preds.append(preds)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print("\nValidation Metrics:")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, digits=4))
    return acc, f1


# --- Run training and Save ---
if CFG.finetuning:
    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
        print(f"Epoch {epoch} finished. Train Loss: {train_loss:.4f}")
        _ = evaluate(model, valid_loader)
    
    print("\nTraining complete.")


# Save model
if CFG.finetuning:
    save_dir = "/kaggle/working/my_distilbert"
    
    model.save_pretrained(save_dir)        # lưu kiến trúc + weights
    Tokenizer.save_pretrained(save_dir)    # rất quan trọng để load lại
    
    # Ngoài ra nếu bạn có Trainer, bạn có thể:
    # trainer.save_model(save_dir)

    print("Save model complete.")


# Zip model
if CFG.finetuning:
    import shutil
    
    save_dir = "/kaggle/working/my_distilbert"
    
    # nén thư mục thành my_distilbert.zip trong /kaggle/working
    shutil.make_archive(save_dir, 'zip', save_dir)
    
    print("Đã tạo:", save_dir + ".zip")


if not CFG.finetuning:
    load_dir = "/kaggle/input/quora-question-pairs-my-models/my_distilbert_full_training"
    model = AutoModelForSequenceClassification.from_pretrained(load_dir)
    model.to(DEVICE)
    Tokenizer = AutoTokenizer.from_pretrained(load_dir)


if not CFG.finetuning:
    evaluate(model, valid_loader)


# Nếu chạy tiếp, đọc file cũ để biết đã làm đến đâu
if os.path.exists(CFG.submission_path):
    done_df = pd.read_csv(CFG.submission_path)
    print(f"Resume mode: {len(done_df)} rows đã lưu")
else:
    done_ids = set()

test_df = pd.read_csv(CFG.test_path)
if CFG.test_size:
    test_df = test_df[:CFG.test_size]

if os.path.exists(CFG.submission_path):
    test_df = test_df[len(done_df):]
    print(f"Test size: {len(test_df)}")

# Dòng 379205 bị lỗi ValueError: text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).
# Vì chứa giá trị NaN


test_df.head()


print(test_df[test_df['question2'].isna()])


print(test_df[test_df['question1'].isna()])


test_df['question1'] = test_df['question1'].fillna('')
print(test_df[test_df['question1'].isna()])


test_df['question2'] = test_df['question2'].fillna('')
print(test_df[test_df['question2'].isna()])


test_dataset = QQPDataset(test_df, Tokenizer, CFG.max_length, test_mode=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=(DEVICE.type == "cuda"),
    collate_fn=collator,
)


from datetime import datetime
import os

model.eval()
save_every = 100       # bao nhiêu batch thì flush ra file

print("Start ~")
with torch.no_grad():
    for i, batch in enumerate(test_loader, 1):
        if i < start_batch:
            continue          # bỏ qua batch cũ

        ids = batch.pop("ids")
        if any(x in done_ids for x in ids):
            # tránh lưu trùng nếu resume nhiều lần
            continue

        for k in batch:
            batch[k] = batch[k].to(DEVICE, non_blocking=True)

        outputs = model(**batch)
        logits  = outputs.logits
        probs   = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        # Ghi ngay
        tmp = pd.DataFrame({"test_id": ids, "is_duplicate": probs})
        # mode='a' + header=False => append, nếu file chưa tồn tại thì header=True
        tmp.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

        if i == 10:
            start = time.time()

        if i == 11:
            end = time.time()
            avg_batch_sec = end - start
            
            num_batches_total = len(test_loader)
            eta_total_sec = avg_batch_sec * num_batches_total
            mins, secs = divmod(int(eta_total_sec), 60)
            hrs, mins = divmod(mins, 60)
            print(f"\n[Time Estimate] Avg seconds/batch: {avg_batch_sec:.3f}")
            print(f"[Time Estimate] Total batches: {num_batches_total}")
            print(f"[Time Estimate] Estimated total testing time: {hrs}h {mins}m {secs}s\n")

        if i % save_every == 0:
            time_point = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Time: {time_point} - Processed batch {i}")

print(f"Checkpointing hoàn tất. File đang ở: {save_path}")

