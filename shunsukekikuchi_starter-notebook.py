import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import gc

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from scipy import signal
from tqdm import tqdm

from torchaudio import transforms as T
import albumentations as A

from torchvision.transforms import v2


class CFG:
    # basic
    model_name = "resnet34"
    seed = 42
    fold = 0

    # training setting
    n_epoch = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size=64
    lr = 1e-5
    img_size = (257, 600)
    train_transform=v2.Resize(img_size)
    valid_transform=v2.Resize(img_size)
    autocast=True # used for training, not for validation

    dataset_path = "/kaggle/input/hms-harmful-brain-activity-classification"


'''# if TPU
#!pip install cloud-tpu-client==0.10 torch==1.12.0 https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-1.12-cp37-cp37m-linux_x86_64.whl --force-reinstall 
import torch_xla
import torch_xla.core.xla_model as xm

class CFG:
    # basic
    model_name = "resnet34"
    seed = 42
    fold = 0

    # training setting
    n_epoch = 50
    device = xm.xla_device()
    batch_size=64
    lr = 1e-6
    img_size = (257, 600)
    train_transform=v2.Resize(img_size)
    valid_transform=v2.Resize(img_size)
    autocast=True # used for training, not for validation

    dataset_path = "/kaggle/input/hms-harmful-brain-activity-classification"'''


def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def eeg2spec(eeg):
    # x: (C,T) or (B, C, T)
    input_len = eeg.shape[-1]
    transform = T.Spectrogram(n_fft = 512,
                          win_length = 64,
                          hop_length = input_len // 600, # 
                          power = 1)
    spec = transform(eeg)**0.8
    spec = torch.nan_to_num(spec)
    spec = F.normalize(spec)
    return spec

set_seed(CFG.seed)


from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.seed)

# Load training data
df = pl.read_csv(f"{CFG.dataset_path}/train.csv")

for fold, (train_idx, valid_idx) in enumerate(sgkf.split(df, y=df["expert_consensus"], groups=df["patient_id"])):
    if fold == CFG.fold:
        break

train_df = df[train_idx]
valid_df = df[valid_idx]


labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]

train_labels = train_df.group_by('eeg_id', maintain_order=True).agg([
    *[pl.col(lbl+'_vote').sum() for lbl in labels],
    pl.len().alias('total_vote') 
]).with_columns(
    pl.sum_horizontal([pl.col(lbl+'_vote') for lbl in labels]).alias("total_vote").cast(pl.Float64)
).with_columns(
    *[pl.col(lbl+'_vote') / pl.col('total_vote') for lbl in labels],
    pl.concat_str(
    [
            pl.lit(f"{CFG.dataset_path}/train_eegs/"),
            pl.col('eeg_id').cast(pl.String),
            pl.lit(".parquet"),
        ],
    ).alias("path"),
).drop("total_vote")

valid_labels = valid_df.group_by('eeg_id', maintain_order=True).agg([
    *[pl.col(lbl+'_vote').sum() for lbl in labels],
    pl.len().alias('total_vote') 
]).with_columns(
    pl.sum_horizontal([pl.col(lbl+'_vote') for lbl in labels]).alias("total_vote").cast(pl.Float64)
).with_columns(
    *[pl.col(lbl+'_vote') / pl.col('total_vote') for lbl in labels],
    pl.concat_str(
    [
            pl.lit(f"{CFG.dataset_path}/train_eegs/"),
            pl.col('eeg_id').cast(pl.String),
            pl.lit(".parquet"),
        ],
    ).alias("path"),
).drop("total_vote")


del df, train_df, valid_df
gc.collect()


np.random.seed(40)
for _ in np.random.randint(len(train_labels), size=(10)):
    idx = int(_)
    data = pl.read_parquet(train_labels['path'][idx])
    spec = eeg2spec(data.drop("EKG").to_torch().transpose(1,0))
    print(f"seizure_vote: {train_labels['seizure_vote'][idx]:.2f} | lpd_vote: {train_labels['lpd_vote'][idx]:.2f} | gpd_vote: {train_labels['gpd_vote'][idx]:.2f} | lrda_vote: {train_labels['lrda_vote'][idx]:.2f} | grda_vote: {train_labels['grda_vote'][idx]:.2f} | other_vote: {train_labels['other_vote'][idx]:.2f}")
    plt.imshow(torch.cat([spec[i][:40, :] for i in range(len(spec))]))
    plt.show()


class HmsDataset(Dataset):
    def __init__(self, labels_df: pl.DataFrame, train=False, transform=None):
        '''
        in train/valid - set train True
        in inference - set train False
        '''
        self.labels_df = labels_df
        self.paths = labels_df['path'].to_list()
        self.train = train
        if train:
            self.targets = labels_df.select([
                    'seizure_vote', 'lpd_vote', 'gpd_vote',
                    'lrda_vote', 'grda_vote', 'other_vote'
                ]).to_torch()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        df = pl.read_parquet(self.paths[idx])
        eeg = df.drop("EKG").to_torch().transpose(1, 0)
        spec = eeg2spec(eeg)
        if self.transform is not None:
            spec = self.transform(spec)
        if self.train:
            target = self.targets[idx]
            return spec, target
        else:
            return spec


train_dataset = HmsDataset(train_labels, train=True, transform=CFG.train_transform)
train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
valid_dataset = HmsDataset(valid_labels, train=True, transform=CFG.valid_transform)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False)


import timm
model = timm.create_model(CFG.model_name, pretrained=True, num_classes=6, in_chans=19).to(CFG.device)
print("number of parameters: ", sum(p.numel() for p in model.parameters()))


optimizer = optim.SGD(model.parameters(), lr=CFG.lr)
kl_loss = nn.KLDivLoss(reduction="batchmean")
# use scheduler here? - skip this time


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    for spec, target in train_loader:
        spec   = spec.to(CFG.device)
        target = target.to(CFG.device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=CFG.autocast):
            log_pred = model(spec).log_softmax(dim=1)
            target_clipped = torch.clamp(target, 1e-8, 1.0)
            loss = criterion(log_pred, target_clipped)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"*** Epoch {epoch} TRAINING COMPLETE. Avg Loss: {avg_loss:.4f} ***")
    torch.cuda.empty_cache(); gc.collect()
    return avg_loss


def valid(model, valid_loader, criterion, epoch):
    model.eval()
    all_log_pred, all_target = [], []

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for spec, target in valid_loader:
            spec   = spec.to(CFG.device)
            target = target.to(CFG.device)

            log_pred = model(spec).log_softmax(dim=1)
            all_log_pred.append(log_pred)
            all_target.append(target)

        log_pred = torch.cat(all_log_pred, dim=0)
        target   = torch.cat(all_target , dim=0)
        target   = torch.clamp(target, 1e-8, 1.0)

        val_loss = criterion(log_pred, target)

    print(f"=== Epoch {epoch} VALIDATION: KL Divergence = {val_loss:.4f} ===")
    torch.cuda.empty_cache(); gc.collect()
    return val_loss


criterion = kl_loss
epoch = -1

model.train()
running_loss = 0.0

for spec, target in train_loader:
    spec   = spec.to(CFG.device)
    target = target.to(CFG.device)

    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", enabled=CFG.autocast):
        log_pred = model(spec).log_softmax(dim=1)
        target_clipped = torch.clamp(target, 1e-8, 1.0)
        loss = criterion(log_pred, target_clipped)

    loss.backward()
    optimizer.step()

    running_loss += loss.item()

avg_loss = running_loss / len(train_loader)
print(f"=== Epoch {epoch} TRAINING COMPLETE. Avg Loss: {avg_loss:.4f} ===")
torch.cuda.empty_cache(); gc.collect()


best_loss = np.inf

for epoch in range(1, CFG.n_epoch+1):
    avg_loss = train(model, train_loader, optimizer, kl_loss, epoch)
    avg_val_loss = valid(model, valid_loader, kl_loss, epoch)
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f">>> New best model saved (KL={best_loss:.4f}) <<<")

