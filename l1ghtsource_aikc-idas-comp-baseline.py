# be careful, just gpt generated shit - validate it (or write ur own pipe)
# baseline public score: 0.626

# some my ideas for you:
# add scaling !!! (per dataset / per sequence / per batch)
# add more features: magnitudes / angles / angular vel & dist / cumsums
# remove gravity
# 6d quats (https://arxiv.org/pdf/1812.07035)
# use full sgkf validation and compute oof score
# improve model (base idea: lstm / gru / transformer / mamba / cnn1d, maybe try multibranch or combine these models)
# add augs: jitter / time warp / magnitude warp / scaling / window warp / moda / masking
# try filtering: kalman, savgol, firwin, butter
# ensemble
# tta


import os
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEED = 69

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed=SEED)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 64
EPOCHS = 50 # adjust it
LR = 1e-3 # adjust it
WEIGHT_DECAY = 1e-5 # adjust it
HIDDEN_SIZE = 128 # adjust it
NUM_LAYERS = 2 # adjust it
DROPOUT = 0.3 # adjust it
MAX_LEN = 800 # adjust it
INPUT_FEATURES = 7 # acc_x,y,z + rot_w,x,y,z
MIXUP_PROB = 0.0 # adjust it
MIXUP_ALPHA = 0.4 # adjust it
NOISE_STD = 0.02 # adjust it
N_FOLDS = 5


def build_sequences(df, is_train=True):
    sequences = []
    grouped = df.groupby('sequence_id', sort=False)
    for seq_id, g in grouped:
        # use more features there !! (generate them before)
        arr = g[['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']].to_numpy(dtype=np.float32)
        subject = g['subject_id'].values[0]
        rec = {'sequence_id': seq_id, 'subject_id': subject, 'X': arr}
        if is_train:
            rec['y'] = g['gesture'].values[0]
        sequences.append(rec)
    return sequences


class MotionSeqDataset(Dataset):
    def __init__(self, sequences, label_encoder=None, max_len=MAX_LEN, augment=False):
        self.sequences = sequences
        self.max_len = max_len
        self.augment = augment
        self.le = label_encoder

    def __len__(self):
        return len(self.sequences)

    def pad_truncate(self, x):
        T, F = x.shape
        if T >= self.max_len:
            return x[:self.max_len]
        else:
            out = np.zeros((self.max_len, F), dtype=np.float32)
            out[:T] = x
            return out

    def add_noise(self, x):
        if NOISE_STD <= 0:
            return x
        noise = np.random.normal(0, NOISE_STD, size=x.shape).astype(np.float32)
        return x + noise

    def __getitem__(self, idx):
        rec = self.sequences[idx]
        x = rec['X']
        x = self.pad_truncate(x)

        # test other augs
        # if self.augment:
            # x = self.add_noise(x)
            
        if 'y' in rec:
            y = self.le.transform([rec['y']])[0]
            return x, y, rec['sequence_id']
        else:
            return x, -1, rec['sequence_id']


def collate_fn(batch):
    # batch: list of (x, y, seqid)
    xs = np.stack([b[0] for b in batch]) # (B, T, F)
    ys = np.array([b[1] for b in batch], dtype=np.int64)
    seqids = [b[2] for b in batch]

    # mixup
    if random.random() < MIXUP_PROB:
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        perm = np.random.permutation(len(xs))
        xs2 = xs[perm]
        ys2 = ys[perm]
        xs = lam * xs + (1 - lam) * xs2
        return torch.from_numpy(xs).float(), torch.from_numpy(ys).long(), torch.from_numpy(ys2).long(), lam, seqids

    return torch.from_numpy(xs).float(), torch.from_numpy(ys).long(), None, None, seqids


class BiGRUModel(nn.Module):
    def __init__(self, input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, num_classes=6):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers>1 else 0.0
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.gru(x) # out: (B, T, 2*H)
        # pool across time dimension
        out = out.permute(0, 2, 1)
        pooled = self.pool(out).squeeze(-1) # (B, 2H)
        logits = self.fc(pooled)
        return logits


def mixup_criterion(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader):
        if batch[2] is not None:
            # mixup case: (xs, ys, ys2, lam, seqids)
            xs, ys, ys2, lam, _ = batch
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)
            ys2 = ys2.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xs)
            loss = mixup_criterion(criterion, logits, ys, ys2, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xs.size(0)
        else:
            xs, ys, _, _, _ = batch
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xs.size(0)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss


def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(loader):
            xs, ys, _, _, _ = batch
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)
            logits = model(xs)
            loss = criterion(logits, ys)
            running_loss += loss.item() * xs.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(ys.detach().cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return avg_loss, f1


train_df = pd.read_csv('/kaggle/input/aikc-idas-courses-2025-final-competition/train.csv')
test_df = pd.read_csv('/kaggle/input/aikc-idas-courses-2025-final-competition/test.csv')


train_seqs = build_sequences(train_df, is_train=True)
test_seqs = build_sequences(test_df, is_train=False)


gestures = sorted(list({rec['y'] for rec in train_seqs}))
le = LabelEncoder()
le.fit(gestures)
num_classes = len(le.classes_)
print(f'{num_classes=}')


subjects = np.array([rec['subject_id'] for rec in train_seqs])
seq_ids = np.array([rec['sequence_id'] for rec in train_seqs])
y_str = np.array([rec['y'] for rec in train_seqs])


gkf = GroupKFold(n_splits=N_FOLDS)
splits = list(gkf.split(seq_ids, y_str, groups=subjects))
train_idx, val_idx = splits[0] # pick fold 0

train_samples = [train_seqs[i] for i in train_idx]
val_samples = [train_seqs[i] for i in val_idx]


# datasets and loaders
train_ds = MotionSeqDataset(train_samples, label_encoder=le, max_len=MAX_LEN, augment=True)
val_ds = MotionSeqDataset(val_samples, label_encoder=le, max_len=MAX_LEN, augment=False)
test_ds_obj = MotionSeqDataset(test_seqs, label_encoder=le, max_len=MAX_LEN, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds_obj, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)


# model
model = BiGRUModel(input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, num_classes=num_classes)
model = model.to(DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)


best_val_f1 = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_f1 = validate_epoch(model, val_loader, criterion)
    scheduler.step(val_f1)

    print(f'{epoch=}: {val_loss=}, {val_f1=}')

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = model.state_dict()
        torch.save(best_state, 'best_model.pth')
        print('saved best model')


if best_state is not None:
    model.load_state_dict(torch.load('best_model.pth'))


# inference on test set
model.eval()
seqid_to_pred = {}
with torch.no_grad():
    for batch in test_loader:
        xs, _, _, _, seqids = batch
        xs = xs.to(DEVICE)
        logits = model(xs)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        for sid, p in zip(seqids, preds):
            seqid_to_pred[sid] = p


# build submission
sample_sub = pd.read_csv('/kaggle/input/aikc-idas-courses-2025-final-competition/sample_submission.csv')
# map preds; if some sequence missing (shouldn't), default to most frequent class
default_class = 0
sample_sub['gesture'] = sample_sub['sequence_id'].apply(lambda sid: le.inverse_transform([seqid_to_pred.get(sid, default_class)])[0])
sample_sub.to_csv('submission.csv', index=False)

