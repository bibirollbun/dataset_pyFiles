import torch
from torch.utils.data import Dataset, DataLoader

import random

import numpy as np
import polars as pl

from scipy import signal
from scipy.signal import butter, lfilter
from tqdm import tqdm
from glob import glob

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union


import sys
sys.path.append("/kaggle/input/math156-model-1d")


def fix_keys(loaded_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in loaded_dict.items()}


#from model import resnet50_1d
#from model2 import EEGNet
from model3 import resnext


#model = resnet50_1d(num_classes=6, in_channels=19)
#model = EEGNet()
model = resnext(num_classes=6, in_channels=19, width_mult = 1.0)
model.load_state_dict(fix_keys(torch.load("/kaggle/input/hms-1dmodels-test/pytorch/default/6/best_model3.pth")))
model.to("cuda")
print("number of parameters: ", sum(p.numel() for p in model.parameters()))


class CFG:
    # basic
    seed = 42
    fold = 0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size=32
    autocast=False # used for training, not for validation

    dataset_path = "/kaggle/input/hms-harmful-brain-activity-classification"

    target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']


def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff, fs=fs, btype="low")
    return lfilter(b, a, data)

def eeg_from_parquet(
    parquet_path: str,
) -> np.ndarray:
    eeg = pl.read_parquet(parquet_path).drop("EKG").cast(pl.Float32)

    # 2) centre-crop to CFG.nsamples rows (gracefully handles short files)
    rows = len(eeg)
    offset = max((rows - 10000) // 2, 0)
    eeg_slice = eeg[offset : offset + 10000]

    # 3) convert to NumPy and fill NaNs
    data = eeg_slice.to_numpy()
    col_mean = np.nanmean(data, axis=0)
    # columns that are entirely NaN → col_mean becomes NaN, so we replace later
    nan_rows, nan_cols = np.where(np.isnan(data))
    data[nan_rows, nan_cols] = col_mean[nan_cols]
    data = np.nan_to_num(data, nan=0.0)  # all-NaN columns → 0

    return data

set_seed(CFG.seed)


df = pl.read_csv(f"{CFG.dataset_path}/test.csv").with_columns(pl.concat_str(
    [
            pl.lit(f"{CFG.dataset_path}/test_eegs/"),
            pl.col('eeg_id').cast(pl.String),
            pl.lit(".parquet"),
        ],
    ).alias("path")
           )


class EEGDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        train: bool = True,
    ) -> None:
        self.all_data = df.with_columns(
                        pl.col("path").map_elements(eeg_from_parquet).alias('eeg')
                        ).select(['eeg', 'eeg_id']).to_dicts()
        self.training = train

    def __len__(self) -> int:
        return len(self.all_data)

    def __getitem__(self, idx: int):
        row = self.all_data[idx]
        y = row['eeg_id']
        X = np.array(row['eeg']) # shape (nsamples, raw_channels)
            
        X = np.clip(X, -1024, 1024)
        X = np.nan_to_num(X) / 32.0  # scale down
        X = butter_lowpass_filter(
            X, cutoff=20, fs=200, order=6
        )

        return torch.tensor(X, dtype=torch.float32).permute(1,0),y


test_dataset = EEGDataset(df, train=False)
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
    for eeg, eeg_id in test_loader:
        eeg   = eeg.to(CFG.device)

        log_pred = model(eeg).log_softmax(dim=1).cpu()
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

