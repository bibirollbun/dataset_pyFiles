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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
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
EPOCHS = 90 # adjust it
LR = 3e-4 # adjust it
WEIGHT_DECAY = 1e-2 # adjust it
HIDDEN_SIZE = 512 # adjust it
NUM_LAYERS = 4 # adjust it
DROPOUT = 0.1 # adjust it
MAX_LEN = 100 # adjust it
INPUT_FEATURES = 11 # acc_x,y,z,abs + rot_w,x,y,z + angle_x,y,z
MIXUP_PROB = 0.2 # adjust it
MIXUP_ALPHA = 0.4 # adjust it
NOISE_STD = 0.02 # adjust it
N_FOLDS = 5
WINDOW_SIZE = 100
STRIDE = 50
ALPHA_RL = 0.3


def build_sequences(df, is_train=True):
    df['acc_abs'] = np.sqrt((df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2))
    df['angle_z'] = np.arccos(1 - 2 * (df['rot_x'] ** 2 + df['rot_y'] ** 2))
    df['angle_y'] = np.arccos(1 - 2 * (df['rot_x'] ** 2 + df['rot_z'] ** 2))
    df['angle_x'] = np.arccos(1 - 2 * (df['rot_z'] ** 2 + df['rot_y'] ** 2))
    sequences = []
    grouped = df.groupby('sequence_id', sort=False)
    for seq_id, g in grouped:
        # use more features there !! (generate them before)
        arr = g[['acc_x', 'acc_y', 'acc_z', 'acc_abs', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 'angle_x', 'angle_y', 'angle_z']].to_numpy(dtype=np.float32)
        subject = g['subject_id'].values[0]
        rec = {'sequence_id': seq_id, 'subject_id': subject, 'X': arr}
        if is_train:
            rec['y'] = g['gesture'].values[0]
        sequences.append(rec)
    return sequences


class MotionSeqDataset(Dataset):
    def __init__(self, sequences, label_encoder=None, max_len=MAX_LEN, augment=True):
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
        if self.augment:
            x = self.add_noise(x)
            
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

    B, T, F = xs.shape
    xs_reshaped = xs.reshape(-1, F)  
    scaler = StandardScaler()
    xs_scaled = scaler.fit_transform(xs_reshaped)
    xs = xs_scaled.reshape(B, T, F)

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
    def __init__(self, input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=2, dropout=0.3, num_classes=6):
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


class optimus_prime(nn.Module):
    def __init__(self, num_features=INPUT_FEATURES, max_len=MAX_LEN, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, num_layers=NUM_LAYERS, num_classes=6, d_model=64, nhead=4):
        super().__init__()
        self.embed = nn.Linear(num_features, d_model)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, d_model))
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.input_ln = nn.LayerNorm(num_features)
    def forward(self, x):
        x = self.input_ln(x)
        x = self.embed(x)
        B, T, D = x.shape

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1, :]

        out = self.transformer(x)
        cls_out = out[:, 0, :]
        return self.cls_head(cls_out)


class optimus_prime_v2(nn.Module):
    def __init__(
        self,
        num_features=INPUT_FEATURES,
        max_len=MAX_LEN,
        num_classes=6,
        d_model=128,
        nhead=4,
        num_layers=NUM_LAYERS,
        dim_feedforward=None,
        dropout=DROPOUT,
        use_conv_stem=True,
        pos_type="learned",
        activation="gelu",
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.use_conv_stem = use_conv_stem
        if use_conv_stem:
    
            self.conv_stem = nn.Sequential(
                nn.Conv1d(in_channels=num_features, out_channels=d_model, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )

            self.embed = None
        else:
            self.embed = nn.Linear(num_features, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.max_len = max_len
        if pos_type == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, d_model))
        else:
            pe = self._build_sinusoidal_pe(max_len + 1, d_model)
            self.register_buffer("pos_embedding", pe, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.input_ln = nn.LayerNorm(num_features)

        self._init_weights()

    def _build_sinusoidal_pe(self, length, d_model):
        pe = torch.zeros(1, length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_weights(self):
        # standard initialization for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # smaller std for cls token
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.input_ln(x)

        if self.use_conv_stem:
            x_conv = x.permute(0, 2, 1)  
            x_conv = self.conv_stem(x_conv)  
            x = x_conv.permute(0, 2, 1)  
        else:
            x = self.embed(x) 

        B, T, D = x.shape
        assert T <= self.max_len, f"Sequence length {T} > max_len {self.max_len}"

        cls = self.cls_token.expand(B, -1, -1) 
        x = torch.cat([cls, x], dim=1)  

        x = x + self.pos_embedding[:, : T + 1, :]

        out = self.transformer(x)  # (B, T+1, D)
        cls_out = out[:, 0, :]  # (B, D)
        mean_out = out[:, 1:, :].mean(dim=1)
        pooled = (cls_out + mean_out) / 2
        logits = self.cls_head(cls_out)
        return logits



class CNN1DActivityClassifier(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, n_channels=INPUT_FEATURES, n_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size * 2, n_classes)
        )

    def forward(self, x, return_features=False):
        x = x.permute(0, 2, 1)  
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        logits = self.classifier(x)
        if return_features:
            return logits, x
        return logits


class MLPBoost(nn.Module):
    def __init__(self, in_dim=HIDDEN_SIZE * 4, num_classes=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)



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
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, f1, acc


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
# model = BiGRUModel(input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, num_classes=num_classes)
model_transformer = optimus_prime_v2()
model_transformer = model_transformer.to(DEVICE)


model_bigru = BiGRUModel()
model_bigru = model_bigru.to(DEVICE)


model_cnn = CNN1DActivityClassifier()
model_cnn = model_cnn.to(DEVICE)


optimizer = torch.optim.Adam(model_cnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)


best_val_f1 = -1.0
best_state_cnn = None

for epoch in range(1, 30 + 1):
    train_loss = train_epoch(model_cnn, train_loader, optimizer, criterion)
    val_loss, val_f1, val_acc = validate_epoch(model_cnn, val_loader, criterion)

    print(f'{epoch=}: {val_loss=}, {val_f1=}, {val_acc=}')

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state_cnn = model_cnn.state_dict()
        torch.save(best_state_cnn, 'best_model_cnn.pth')
        print('saved best model')


if best_state_cnn is not None:
    model_cnn.load_state_dict(torch.load('best_model_cnn.pth'))


model_mlp = MLPBoost()
model_mlp = model_mlp.to(DEVICE)


optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)


best_val_f1 = -1.0
best_state_mlp = None

for epoch in range(1, 30 + 1):
    train_loss = train_epoch_rl(model_mlp, model_cnn, train_loader, optimizer, criterion)
    val_loss, val_f1, val_acc = validate_epoch_rl(model_mlp, model_cnn, val_loader, criterion)

    print(f'{epoch=}: {val_loss=}, {val_f1=}, {val_acc=}')

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state_mlp = model_mlp.state_dict()
        torch.save(best_state_mlp, 'best_model_mlp.pth')
        print('saved best model')


if best_state_mlp is not None:
    model_mlp.load_state_dict(torch.load('best_model_mlp.pth'))


optimizer = torch.optim.Adam(model_transformer.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)


best_val_f1 = -1.0
best_state_trans = None

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch_rl(model_transformer, model_cnn, train_loader, optimizer, criterion)
    val_loss, val_f1, val_acc = validate_epoch_rl(model_transformer, model_cnn, val_loader, criterion)
    scheduler.step(val_f1)

    print(f'{epoch=}: {val_loss=}, {val_f1=}, {val_acc=}')

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state_trans = model_transformer.state_dict()
        torch.save(best_state_trans, 'best_model_trans_rl.pth')
        print('saved best model')


if best_state_trans is not None:
    model_transformer.load_state_dict(torch.load('best_model_trans_rl.pth'))


optimizer = torch.optim.Adam(model_cnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)


def train_epoch_rl(model, trans, loader, optimizer, criterion):
    trans.eval()
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
            with torch.no_grad():
                logits_tr, feat = trans(xs, return_features=True)
            logits_cnn = model(feat)
            logits = logits_tr + ALPHA_RL * logits_cnn
            loss = mixup_criterion(criterion, logits, ys, ys2, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xs.size(0)
        else:
            xs, ys, _, _, _ = batch
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                logits_tr, feat = trans(xs, return_features=True)
            logits_cnn = model(feat)
            logits = logits_tr + ALPHA_RL * logits_cnn
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xs.size(0)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss


def validate_epoch_rl(model, trans, loader, criterion):
    model.eval()
    trans.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(loader):
            xs, ys, _, _, _ = batch
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)
            logits_tr, feat = trans(xs, return_features=True)
            logits_cnn = model(feat)
            logits = logits_tr + ALPHA_RL * logits_cnn
            loss = criterion(logits, ys)
            running_loss += loss.item() * xs.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(ys.detach().cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='macro')
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, f1, acc


best_val_f1 = -1.0
best_state_cnn_rl = None

model_transformer.eval()
for p in model_transformer.parameters():
    p.requires_grad = False
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch_rl(model_cnn, model_transformer, train_loader, optimizer, criterion)
    val_loss, val_f1, val_acc = validate_epoch_rl(model_cnn, model_transformer, val_loader, criterion)
    scheduler.step(val_f1)

    print(f'{epoch=}: {val_loss=}, {val_f1=}, {val_acc=}')

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state_cnn = model_cnn.state_dict()
        torch.save(best_state_cnn, 'best_model_cnn_rl.pth')
        print('saved best model')


if best_state_cnn is not None:
    model_cnn.load_state_dict(torch.load('best_model_cnn_rl.pth'))


optimizer = torch.optim.Adam(model_bigru.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)


best_val_f1 = -1.0
best_state_bigru = None

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model_bigru, train_loader, optimizer, criterion)
    val_loss, val_f1, val_acc = validate_epoch(model_bigru, val_loader, criterion)
    scheduler.step(val_f1)

    print(f'{epoch=}: {val_loss=}, {val_f1=}, {val_acc=}')

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state_bigru = model_bigru.state_dict()
        torch.save(best_state_bigru, 'best_model_bigru.pth')
        print('saved best model')


if best_state_bigru is not None:
    model_bigru.load_state_dict(torch.load('best_model_bigru.pth'))


# inference on test set
model_transformer.eval()
model_cnn.eval()
alpha = 0.5
seqid_to_pred = {}
with torch.no_grad():
    for batch in test_loader:
        xs, _, _, _, seqids = batch
        xs = xs.to(DEVICE)
        logits_t = model_transformer(xs)
        logits_b = model_cnn(xs)
        logits = alpha * logits_t + (1 - alpha) * logits_b
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        for sid, p in zip(seqids, preds):
            seqid_to_pred[sid] = p


# inference on test set
model_cnn.eval()
alpha = 0.8
seqid_to_pred = {}
with torch.no_grad():
    for batch in test_loader:
        xs, _, _, _, seqids = batch
        xs = xs.to(DEVICE)
        logits = model_cnn(xs)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        for sid, p in zip(seqids, preds):
            seqid_to_pred[sid] = p


# inference on test set
model_transformer.eval()
model_cnn.eval()
alpha = 0.5
seqid_to_pred = {}
with torch.no_grad():
    for batch in test_loader:
        xs, _, _, _, seqids = batch
        xs = xs.to(DEVICE)
        logits_t = model_transformer(xs)
        logits_b = model_cnn(xs)
        logits = logits_b + 0.5 * logits_t
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        for sid, p in zip(seqids, preds):
            seqid_to_pred[sid] = p


# inference on test set
model_mlp.eval()
model_cnn.eval()
alpha = 0.5
seqid_to_pred = {}
with torch.no_grad():
    for batch in test_loader:
        xs, _, _, _, seqids = batch
        xs = xs.to(DEVICE)
        logits_t, feat = model_cnn(xs, return_features=True)
        logits_b = model_mlp(feat)
        logits = logits_b + ALPHA_RL * logits_t
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        for sid, p in zip(seqids, preds):
            seqid_to_pred[sid] = p


# build submission
sample_sub = pd.read_csv('/kaggle/input/aikc-idas-courses-2025-final-competition/sample_submission.csv')
# map preds; if some sequence missing (shouldn't), default to most frequent class
default_class = 0
sample_sub['gesture'] = sample_sub['sequence_id'].apply(lambda sid: le.inverse_transform([seqid_to_pred.get(sid, default_class)])[0])
sample_sub.to_csv('submission.csv', index=False)




