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


# ============================
# Kaggle 專用環境安裝（安全版）
# ============================

# ❗ Kaggle 已內建 CUDA-compatible PyTorch
# ❗ 不要安裝 / 升級 torch / torchvision / torchaudio

# 1️⃣ 固定 protobuf（避免 HF / tokenizer 相容問題）
!pip install -U protobuf==3.20.3

!pip install -U transformers==4.36.2 typing-extensions
# 2️⃣ 安裝 Kaggle 相容的 transformers（最高穩定版）

# 3️⃣ 其他常用套件
!pip install -U sentencepiece scikit-learn tqdm

print("✅ Environment install done. Please restart the kernel.")


import torch, transformers, sys
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)


# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os, random, html
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from torch.cuda.amp import autocast, GradScaler

# ----------------- Kaggle friendly -----------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_logging.set_verbosity_error()
torch.backends.cudnn.benchmark = True

# ---------------- CONFIG ----------------
MODEL_NAME = "/kaggle/input/roberta-base/roberta-base/"
WINDOW_SIZE = 512
WINDOW_STRIDE = 256
MAX_WINDOWS = 6
TOP_K = 2
BATCH_SIZE = 2
ACCUM_STEPS = 4
EPOCHS = 4
N_SPLITS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 13
LR = 2e-5
NUM_WORKERS = 2

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------- UTIL ----------------
def sliding_windows_for_ids(ids: List[int], max_len=WINDOW_SIZE, stride=WINDOW_STRIDE):
    windows = []
    start = 0
    while start < len(ids):
        end = min(start + max_len - 2, len(ids))
        windows.append(ids[start:end])
        if end == len(ids):
            break
        start += stride
    return windows

# ---------------- DATA ----------------
def load_data():
    train = pd.read_csv("/kaggle/input/mydataset/train.csv")
    test = pd.read_csv("/kaggle/input/mydataset/test.csv")
    sample = pd.read_csv("/kaggle/input/mydataset/sample_submission.csv")

    y = train.iloc[:, 11:].values
    X = train[['question_title','question_body','answer']].fillna('').apply(lambda c: c.map(html.unescape))
    X_test = test[['question_title','question_body','answer']].fillna('').apply(lambda c: c.map(html.unescape))
    return X, X_test, y, sample

class SlideDataset(Dataset):
    def __init__(self, df, y=None, tokenizer=None, max_title_len=128):
        self.titles = df['question_title'].tolist()
        self.bodies = df['question_body'].tolist()
        self.answers = df['answer'].tolist()
        self.tokenizer = tokenizer
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

        self.pre_title_ids = []
        self.pre_title_mask = []
        self.pre_windows = []

        for t, b, a in tqdm(
            zip(self.titles, self.bodies, self.answers),
            total=len(self.titles),
            desc="Pre-tokenize"
        ):
            t_enc = tokenizer(
                t,
                max_length=max_title_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.pre_title_ids.append(t_enc['input_ids'][0])
            self.pre_title_mask.append(t_enc['attention_mask'][0])

            all_windows = []
            cnt = 0
            for ids in (
                tokenizer.encode(b, add_special_tokens=False),
                tokenizer.encode(a, add_special_tokens=False)
            ):
                for w in sliding_windows_for_ids(ids):
                    if cnt >= MAX_WINDOWS:
                        break
                    enc = tokenizer.build_inputs_with_special_tokens(w)
                    all_windows.append(enc[:WINDOW_SIZE])
                    cnt += 1
                if cnt >= MAX_WINDOWS:
                    break

            # hidden dataset 防呆
            if len(all_windows) == 0:
                all_windows = [[tokenizer.cls_token_id]]

            self.pre_windows.append(all_windows)

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        item = {
            "title_ids": self.pre_title_ids[idx],
            "title_mask": self.pre_title_mask[idx],
            "windows": self.pre_windows[idx]
        }
        if self.y is not None:
            item["labels"] = self.y[idx]
        return item

def collate_fn(batch):
    pad_id = 1

    title_ids = torch.stack([b['title_ids'] for b in batch])
    title_mask = torch.stack([b['title_mask'] for b in batch])

    windows, sample_map = [], []
    for i, b in enumerate(batch):
        for w in b['windows']:
            windows.append(w)
            sample_map.append(i)

    padded, masks = [], []
    for w in windows:
        pad = WINDOW_SIZE - len(w)
        padded.append(w + [pad_id]*pad)
        masks.append([1]*len(w) + [0]*pad)

    windows_ids = torch.tensor(padded)
    windows_mask = torch.tensor(masks)
    sample_map = torch.tensor(sample_map)

    labels = torch.stack([b['labels'] for b in batch]) if 'labels' in batch[0] else None

    return {
        "title_ids": title_ids,
        "title_mask": title_mask,
        "windows_ids": windows_ids,
        "windows_mask": windows_mask,
        "window_to_sample": sample_map,
        "labels": labels
    }

# ---------------- MODEL ----------------
class SlideModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.encoder.gradient_checkpointing_enable()

        hidden = self.encoder.config.hidden_size
        self.win_score = nn.Linear(hidden * 3, 1)
        self.proj = nn.Linear(hidden * 9, 512)
        self.cls = nn.Linear(512, 30)

    def encode(self, ids, mask):
        out = self.encoder(ids, mask).last_hidden_state
        mask = mask.unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        maxv = (out + (1-mask)*-1e9).max(1).values
        cls = out[:, 0]
        return torch.cat([cls, mean, maxv], 1)

    def forward(self, title_ids, title_mask, win_ids, win_mask, win_map):
        title_vec = self.encode(title_ids, title_mask)
        B = title_vec.size(0)
        hidden = title_vec.size(1) // 3

        win_vec = self.encode(win_ids, win_mask)
        win_scores = self.win_score(win_vec).squeeze(-1)

        pooled = []
        for i in range(B):
            g = win_vec[win_map == i]
            s = win_scores[win_map == i]

            if g.size(0) == 0:
                pooled.append(torch.zeros(hidden * 6, device=title_vec.device))
                continue

            k = min(TOP_K, g.size(0))
            idx = s.topk(k).indices
            topk = g[idx]

            pooled.append(torch.cat([topk.mean(0), topk.max(0).values], 0))

        agg = torch.stack(pooled)
        x = torch.cat([title_vec, agg], 1)
        return self.cls(self.proj(x))

# ---------------- RUN ----------------
def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    X, X_test, y, sample = load_data()

    dataset = SlideDataset(X, y, tokenizer)
    test_dataset = SlideDataset(X_test, None, tokenizer)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    all_preds = []

    for fold, (tr, cv) in enumerate(kf.split(dataset)):
        print(f"\n========== Fold {fold} ==========")

        model = SlideModel(MODEL_NAME).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        scaler = GradScaler()
        loss_fn = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            Subset(dataset, tr),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS
        )

        val_loader = DataLoader(
            Subset(dataset, cv),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS
        )

        for epoch in range(EPOCHS):
            model.train()
            opt.zero_grad()
            running_loss = 0.0

            for step, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch+1}")):
                with autocast():
                    logits = model(
                        batch['title_ids'].to(DEVICE),
                        batch['title_mask'].to(DEVICE),
                        batch['windows_ids'].to(DEVICE),
                        batch['windows_mask'].to(DEVICE),
                        batch['window_to_sample'].to(DEVICE)
                    )
                    loss = loss_fn(logits, batch['labels'].to(DEVICE)) / ACCUM_STEPS

                scaler.scale(loss).backward()
                running_loss += loss.item()

                if (step + 1) % ACCUM_STEPS == 0 or step + 1 == len(train_loader):
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

            # -------- validation spearman --------
            model.eval()
            val_preds, val_targets = [], []

            with torch.no_grad():
                for batch in val_loader:
                    logits = model(
                        batch['title_ids'].to(DEVICE),
                        batch['title_mask'].to(DEVICE),
                        batch['windows_ids'].to(DEVICE),
                        batch['windows_mask'].to(DEVICE),
                        batch['window_to_sample'].to(DEVICE)
                    )
                    val_preds.append(torch.sigmoid(logits).cpu().numpy())
                    val_targets.append(batch['labels'].cpu().numpy())

            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)

            spearman = np.nanmean([
                spearmanr(val_targets[:, i], val_preds[:, i]).correlation
                for i in range(val_targets.shape[1])
            ])

            print(f"Epoch {epoch+1}/{EPOCHS} train_loss={running_loss:.4f} val_spearman={spearman:.4f}")

        # ---------- inference ----------
        model.eval()
        preds = []
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Infer"):
                logits = model(
                    batch['title_ids'].to(DEVICE),
                    batch['title_mask'].to(DEVICE),
                    batch['windows_ids'].to(DEVICE),
                    batch['windows_mask'].to(DEVICE),
                    batch['window_to_sample'].to(DEVICE)
                )
                preds.append(torch.sigmoid(logits).cpu().numpy())

        all_preds.append(np.vstack(preds))
        del model
        torch.cuda.empty_cache()

    # ---------- 正確 submission ----------
    final_preds = np.mean(all_preds, axis=0)
    assert final_preds.shape[0] == len(test_dataset)

    submission = pd.DataFrame(final_preds, columns=sample.columns[1:])
    submission.insert(0, sample.columns[0], sample.iloc[:, 0].values)
    submission.to_csv("submission.csv", index=False)

    print("✅ submission.csv generated:", submission.shape)

run()




