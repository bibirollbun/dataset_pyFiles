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


import sys
from pathlib import Path
# update for 2d cnn
# sys.path.append("/kaggle/input/math156-model-1d")


def fix_keys(loaded_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in loaded_dict.items()}


class CFG:
    # basic
    model_name = "resnet18"
    seed = 42
    fold = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size=64
    img_size = (257, 600)
    train_transform=v2.Resize(img_size)
    valid_transform=v2.Resize(img_size)
    autocast=False # used for training, not for validation

    dataset_path = "/kaggle/input/hms-harmful-brain-activity-classification"
    target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']


# update for 2d cnn

import timm

model = timm.create_model(CFG.model_name, pretrained=False, num_classes=6, in_chans=19).to(CFG.device)
# modify below line for 2d 
model.load_state_dict(fix_keys(torch.load(f"/kaggle/input/resnet_spec/pytorch/default/1/{CFG.model_name}_best_model.pth")))
model.to(CFG.device)


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


df = pl.read_csv(f"{CFG.dataset_path}/test.csv").with_columns(pl.concat_str(
    [
            pl.lit(f"{CFG.dataset_path}/test_eegs/"),
            pl.col('eeg_id').cast(pl.String),
            pl.lit(".parquet"),
        ],
    ).alias("path")
           )


class HmsDataset(Dataset):
    def __init__(self, labels_df: pl.DataFrame, train=False, transform=None):
        '''
        in train/valid - set train True
        in inference - set train False
        '''
        self.labels_df = labels_df
        self.paths = labels_df['path'].to_list()
        self.train = train
        
        self.targets = labels_df.select([
                'eeg_id'
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
        target = self.targets[idx]
        return spec, target


test_dataset = HmsDataset(df, train=False, transform=CFG.valid_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, drop_last=False)


def _fix_row(row: np.ndarray) -> np.ndarray:
    """Return a copy whose elements sum to 1.0 by tweaking one entry."""
    s = row.sum(dtype=np.float64)
    delta = 1.0 - s
    if abs(delta) < 1e-6:           # already OK
        return row

    # choose index to adjust
    idx = row.argmin() if delta > 0 else row.argmax()
    row[idx] += delta               # add (or subtract) the difference
    # numerical safety - clip into [0,1]
    row[idx] = np.clip(row[idx], 0.0, 1.0)
    return row


model.eval()
all_log_pred, all_eeg_id = [], []

with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
    for spec, eeg_id in test_loader:
        spec = spec.to(CFG.device)

        log_pred = model(spec).log_softmax(dim=1).cpu()
        all_log_pred.append(log_pred)
        all_eeg_id.append(eeg_id)

log_preds = torch.cat(all_log_pred, dim=0)          # (N, n_classes)
eeg_ids   = torch.cat(all_eeg_id , dim=0)           # (N,)

log_preds = log_preds.reshape(-1, log_preds.size(-1))  # (N, C)
eeg_ids   = eeg_ids.reshape(-1)                        # (N,)

probs = log_preds.exp()


probs_np = probs.cpu().numpy().astype(np.float32)         # shape = (N, C)
probs_np = np.apply_along_axis(_fix_row, 1, probs_np)     # normalise rows


submit_df = pl.DataFrame({
    "eeg_id":   eeg_ids.cpu().numpy().astype("int64"),
    **{c: probs_np[:, i] for i, c in enumerate(CFG.target_cols)}
})

submit_df.write_csv("submission.csv")


submit_df

