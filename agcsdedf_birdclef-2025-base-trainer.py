%%capture
!pip install -U transformers  --no-index --find-links /kaggle/input/pip-hub


import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import timm
import tqdm
import librosa
import numpy as np
import pandas as pd
from timm.optim.adan import Adan
# from timm.utils.model_ema import ModelEmaV2
from timm.scheduler.cosine_lr import CosineLRScheduler
from transformers import AutoModel

import random
import albumentations as A
# import torch_audiomentations as Audio
from albumentations.pytorch import ToTensorV2

import gc
import math
import glob
import dataclasses
from transformers import set_seed
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


train_csv_path = "/kaggle/input/birdclef-2025/train.csv"
labeldata_csv_path = "/kaggle/input/custom-label-data/custom_label-data.csv"
unlabel_data_path = "/kagglse/input/birdclef-2025/train_soundscapes"
submission_path = "/kaggle/input/birdclef-2025/sample_submission.csv"
SEED = 3407

VALID_DATA_RATE = .15
TRAIN_MIN_LENGTH = 60
EPOCHS = 30
BS = 32
LR = 3e-4
WD = 3e-2
NUM_WORKERS = 2
MODEL_NAME = "EfficientV2"
VALID_INTERVAL = 2

device = "cuda" if torch.cuda.is_available() else "cpu"


train_transform = None
train_audio_transform = None

test_transform = None
test_audio_transform = None


@dataclasses.dataclass
class AudioParam:
    SR: int=32_000
    NFFT: int=2048
    NMEL: int=128
    FMAX: int=16_000
    FMIN: int=20
    HOP_LENGTH: int=NFFT // 4


set_seed(SEED)
audio_param = AudioParam()


sub_csv = pd.read_csv(submission_path)
idx2cls = sub_csv.columns.drop("row_id").tolist()
cls2idx = {c: i for i, c in enumerate(idx2cls)}


def pcen(E, alpha=0.98, delta=2, r=0.5, s=0.025, eps=1e-6):   
    M = scipy.signal.lfilter([s], [1, s - 1], E)
    smooth = (eps + M)**(-alpha)
    return (E * smooth + delta)**r - delta**r


class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df_group,
        transform=None,
        audio_transform=None,
    ):
        self.file_path = df_group["file_path"].values
        self.label_id = df_group["label_id"].values[0]
        self.end_time = df_group["end_time"].values

        self.transform = transform
        self.audio_transform = audio_transform
        self.length = len(self.file_path)

    def __getitem__(self, idx):
        fp, y, end_time = (
            self.file_path[idx],
            self.label_id,
            self.end_time[idx],
        )
        x, sr = librosa.load(
            fp,
            sr=audio_param.SR,
            offset=random.uniform(0, end_time-5) if end_time > 5 else 0,
            duration=5,
        )
        if x.shape[0] < audio_param.SR * 5:
            x = np.concatenate([x, np.zeros((audio_param.SR * 5 - x.shape[0]), dtype=x.dtype)])

        if self.audio_transform is not None:
            x = self.audio_transform(sample=x, sample_rate=audio_param.SR)

        x = self.pipeline(x)

        if self.transform is not None:
            x = self.transform(image=x)["image"]

        return x, y

    def pipeline(self, x):
        mels = librosa.feature.melspectrogram(
            y=x,
            sr=audio_param.SR,
            n_fft=audio_param.NFFT,
            n_mels=audio_param.NMEL,
            fmax=audio_param.FMAX,
            fmin=audio_param.FMIN,
            hop_length=audio_param.HOP_LENGTH,
        )

        # db_map = pcen(mels).astype(np.float32)

        db_map = librosa.power_to_db(mels, ref=np.max)
        db_map = (db_map + 80) / 80

        return db_map[None]

    def __len__(self):
        return self.length


class ValidDataset(TrainDataset):
    def __init__(
        self, 
        df_group,
        transform=None,
        audio_transform=None,
    ):
        self.file_path = df_group["file_path"].values
        self.label_id = df_group["label_id"].values[0]
        self.end_time = df_group["end_time"].values

        self.transform = transform
        self.audio_transform = audio_transform

        sample_counts = [max(t, 5) // 5 for t in self.end_time]
        cumulative_counts = [0]
        for cnt in sample_counts:
            cumulative_counts.append(cumulative_counts[-1] + cnt)
        self.cumulative_counts = cumulative_counts
            
        self.length = int(sum(sample_counts))

    def __getitem__(self, idx):
        audio_idx = next(i for i in range(len(self.cumulative_counts) - 1) 
                    if self.cumulative_counts[i] <= idx < self.cumulative_counts[i + 1])
        inner_idx = idx - self.cumulative_counts[audio_idx]
        
        fp, y, end_time = (
            self.file_path[audio_idx],
            self.label_id,
            self.end_time[audio_idx],
        )
        x, sr = librosa.load(
            fp,
            sr=audio_param.SR,
            offset=inner_idx,
            duration=5,
        )

        if x.shape[0] < audio_param.SR * 5:
            x = np.concatenate([x, np.zeros((audio_param.SR * 5 - x.shape[0]), dtype=x.dtype)])

        if self.audio_transform is not None:
            x = self.audio_transform(sample=x, sample_rate=audio_param.SR)

        x = self.pipeline(x)

        if self.transform is not None:
            x = self.transform(image=x)["image"]

        return x, y

    def __len__(self):
        return self.length


data = pd.read_csv(labeldata_csv_path)
data["file_path"] = data["file_path"].apply(lambda x: x.replace("data/train_audio", "/kaggle/input/birdclef-2025/train_audio"))

train_data = []
valid_data = []
for group_id, group in data.groupby("label_id"):
    group = group.sample(frac=1).reset_index(drop=True)
    group_length = len(group)
    split = int(group_length * (1 - VALID_DATA_RATE))
    if group[:split]["end_time"].sum() < TRAIN_MIN_LENGTH:
        train_data.append(group)
    else:
        train_data.append(group[:split])
    valid_data.append(group[split:])

train_data = pd.concat(train_data).reset_index(drop=True)
valid_data = pd.concat(valid_data).reset_index(drop=True)


train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset(
        [
            TrainDataset(
                group,
                transform=train_transform,
                audio_transform=train_audio_transform,
            ) 
            for group_id, group in train_data.groupby("label_id")
        ]
    ),
    batch_size=BS,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=None,
)

valid_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset(
        [
            ValidDataset(
                group,
                transform=test_transform,
                audio_transform=test_audio_transform,
            ) 
            for group_id, group in valid_data.groupby("label_id")
        ]
    ),
    batch_size=BS // 2,
    shuffle=False,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=None,
)


del data, train_data, valid_data
gc.collect()


class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1, pretrained=False, dropout=.0):
        super().__init__()
        self.backbone = timm.create_model(
            "timm/tf_efficientnetv2_m.in21k",
            in_chans=1,
            pretrained=pretrained,
            features_only=True,
            drop_rate=dropout,
            drop_path_rate=dropout,
        )

        self.head = nn.Sequential(
            nn.Conv2d(512, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.head(x)

        return x


class EfficientNetB3(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3.ra2_in1k",
            pretrained=False, 
            features_only=True, 
        )
        self.backbone.load_state_dict(
            torch.load("/kaggle/input/birdclef-2025-model-hub/efficientnet_b3.ra2_in1k/pytorch_model.bin", weights_only=True),
            strict=False,
        )
        self.backbone.conv_stem = nn.Conv2d(1, 40, 3, stride=2, padding=1, bias=False)
        self.head = nn.Sequential(
            nn.Conv2d(384, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.head(x)

        return x


# model = EfficientNetB3(len(idx2cls))
model = EfficientNetV2(len(idx2cls), True, .1)

model.to(device);


def get_param_groups(model, nowd_keys=()):
    para_groups, para_groups_dbg = {}, {}
    
    for name, para in model.named_parameters():
        if not para.requires_grad:
            continue  # frozen weights
        if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
            wd_scale, group_name = 0., 'no_decay'
        else:
            wd_scale, group_name = 1., 'decay'
        
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
            para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)

    return list(para_groups.values())


def cal_score(y_hat, y):
    matrix = torch.zeros(y_hat.shape)
    matrix.scatter_(1, torch.from_numpy(y).reshape(-1, 1), 1)
    matrix = matrix.numpy()
    return roc_auc_score(
        y_true=matrix.reshape(-1),
        y_score=y_hat.reshape(-1),
        average="macro",
        multi_class="ovo",
    )


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    # dataloader = tqdm.tqdm(dataloader, desc="Train: ", disable=False)
    losses = 0
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        losses += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return losses / len(dataloader)


@torch.no_grad()
def valid_one_epoch(model, dataloader):
    model.eval()
    # dataloader = tqdm.tqdm(dataloader, desc="Valid: ", disable=False)
    target = []
    predict = []
    for i, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device)
        out = model(x).detach().cpu().sigmoid().numpy()
        y = y.cpu().numpy()
        predict.append(out)
        target.append(y)

    score = cal_score(
        np.concatenate(predict).reshape(-1, len(idx2cls)),
        np.concatenate(target).reshape(-1, 1),
    )

    return score


class Criterion(nn.BCEWithLogitsLoss):
    def __init__(self, normalize_targets=True, **kwargs):
        super().__init__(**kwargs)
        self.reduction = "none"
        
        self._normalize_targets = normalize_targets
        self._eps = torch.finfo(torch.float32).eps
        
    def forward(self, x, y):
        b = x.shape[0]
        m = torch.zeros_like(x, device=x.device)
        for i in range(b):
            m[i, y[i]] = 1.

        if self._normalize_targets:
            m /= self._eps + m.sum(dim=1, keepdim=True)
        per_sample_per_target_loss = -m * F.log_softmax(x, -1)
        per_sample_loss = torch.sum(per_sample_per_target_loss, -1).mean()

        return per_sample_loss


criterion = Criterion()
optimizer = torch.optim.AdamW(
    get_param_groups(model, (".bias", )),
    lr=LR,
    weight_decay=WD,
)

lr_scheduler = CosineLRScheduler(
    optimizer,
    EPOCHS,
    warmup_t=EPOCHS // 6,
    warmup_lr_init=LR / 10,
)


min_loss = float("inf")
score = 0

for epoch in range(1, EPOCHS+1):
    lr_scheduler.step(epoch-1)
    loss = train_one_epoch(model, train_dataloader, criterion, optimizer)
    if min_loss > loss:
        min_loss = loss
        torch.save(
            model.state_dict(),
            f"{MODEL_NAME}_min-loss"
        )
        
    s = -0.01
    if epoch % VALID_INTERVAL == 0:
        s = valid_one_epoch(model, valid_dataloader)
        if s > score:
            score = s
            torch.save(
                model.state_dict(),
                f"{MODEL_NAME}_max-score"
            )

    print(
        f"Epoch {epoch}/{EPOCHS}\t"
        f"Cur Epoch Loss: {loss :.4f}\t"
        f"Min Loss: {min_loss :.4f}\t"
        f"Cur Epoch Score: {s*100 :.2f}%\t"
        f"Max Score: {score*100 :.2f}%\t"
    )

    torch.save(
        model.state_dict(),
        f"{MODEL_NAME}_last"
    )




