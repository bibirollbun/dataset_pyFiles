from functools import cached_property
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import gc
import glob
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import polars as pl
import random
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


%cd /kaggle/input/cmi-detect-behavior-with-sensor-data


pd.set_option('display.width', 1000)


df = pd.read_csv('train.csv', index_col=0)
df = df.reset_index(drop=True)


df['sequence_id'].nunique()


# aggregate IMU features
acc_features = ['acc_x', 'acc_y', 'acc_z']
rot_features = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
imu_features = acc_features + rot_features

y_feature = 'gesture'


print(df[df['sequence_id'] == 'SEQ_000007'][imu_features + [y_feature]].head())


label_encoder = LabelEncoder()
_ = label_encoder.fit(df[y_feature])


def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return arr[idx, np.arange(idx.shape[1])]

def bfill(arr): 
    return ffill(arr[::-1, :])[::-1, :]

def _replace_nans(seq):
    arr = bfill(ffill(seq))
    return np.where(np.isnan(arr), 0, arr)


cache = Path('/kaggle/working')
imu_features_dir = cache / 'imu_features'
if not imu_features_dir.exists():
    grouped = df.groupby('sequence_id')
    imu_features_dir.mkdir(parents=True, exist_ok=True)
    # write IMU features to a file
    overwrite = False
    for seq_id, seq in tqdm(grouped):
        seq = seq[imu_features].values
        output_file = imu_features_dir / f"{seq_id}.npy"
        if overwrite or not output_file.exists():
            with open(output_file, "wb") as f:
                np.save(f, _replace_nans(seq))

    y_values = grouped[y_feature].first().values
    y_values = label_encoder.transform(y_values)
    gestures_file = cache / 'gestures.npy'
    np.save(gestures_file, y_values)


# --- Feature Engineering ---
def _remove_gravity_from_acc(acc_values, quat_values):
    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]
    return linear_accel

def _calculate_angular_velocity_from_quat(quat_values, time_delta=1/10):
    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))
    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]
        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
            np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue
        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
    return angular_vel

def _calculate_angular_distance(quat_values):
    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)
    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]
        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
            np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
            pass
    return angular_dist

def _imu_feature_engineering(imu):
    acc = imu[:,0:3]                                                                       # x, y, z
    rot = imu[:,3:7]                                                                       # w, x, y, z
    acc_mag = np.sqrt(acc[:,0]**2 + acc[:,1]**2 + acc[:,2]**2)                             # 1, Calculate acceleration magnitude
    rot_angle = 2 * np.arccos(rot[:,0].clip(-1, 1))                                        # 1, Calculate quaternion rotation angle (radians)
    acc_mag_jerk = np.diff(acc_mag, prepend=acc_mag[0])                                    # 1, Calculate jerk of acceleration magnitude (difference, prepend first element)
    rot_angle_vel = np.diff(rot_angle, prepend=rot_angle[0])                               # 1, Calculate angular velocity of rotation angle (difference, prepend first element)
    linear_acc = _remove_gravity_from_acc(acc, rot[:, [1, 2, 3, 0]])                  # 3, Linear acceleration after removing gravity (input quaternion format is [x, y, z, w])
    linear_acc_mag = np.sqrt(linear_acc[:,0]**2 + linear_acc[:,1]**2 + linear_acc[:,2]**2) # 1, Linear acceleration magnitude
    linear_acc_mag_jerk = np.diff(linear_acc_mag, prepend=linear_acc_mag[0])               # 1, Jerk of linear acceleration magnitude (difference, prepend first element)
    angular_vel = _calculate_angular_velocity_from_quat(rot[:, [1, 2, 3, 0]])         # 3, Calculate angular velocity (input quaternion format is [x, y, z, w])
    angular_distance = _calculate_angular_distance(rot[:, [1, 2, 3, 0]])              # 1, Calculate angular distance between adjacent frames (radians)
    # Concatenate all features
    new_imu = np.concatenate([
        acc,                          # 3
        rot,                          # 4
        acc_mag[:, None],             # 1
        rot_angle[:, None],           # 1
        acc_mag_jerk[:, None],        # 1
        rot_angle_vel[:, None],       # 1
        linear_acc,                   # 3
        linear_acc_mag[:, None],      # 1
        linear_acc_mag_jerk[:, None], # 1
        angular_vel,                  # 3
        angular_distance[:, None]     # 1
    ], axis=1)
    return new_imu

# this works even if we only have IMU features (the first 7 columns)
def _feature_engineering(seq):
    imu = seq[:,:7]
    new_imu = _imu_feature_engineering(imu)  # Apply IMU feature engineering
    other_features = seq[:, 7:]  # Other features besides IMU
    thm = other_features[:, :5]         # Temperature features
    tof = other_features[:, 5:5+320]    # TOF features
    demo = other_features[:, 5+320:]    # Demographic features
    # Concatenate all features: IMU(20) + Temperature(5) + Demographic(7) + TOF(320)
    features = np.concatenate([new_imu, thm, demo, tof], axis=1)
    return features


# --- Data Augmentation Methods (applied to normalized data) ---
def _jitter(sequence, sigma=0.1):
    return sequence + np.random.normal(loc=0., scale=sigma, size=sequence.shape)

def _time_mask(sequence, max_mask_size=25):
    seq_len = sequence.shape[0]
    mask_size = np.random.randint(1, max_mask_size)
    start = np.random.randint(0, max(1, seq_len - mask_size))
    sequence[start : start + mask_size] = 0
    return sequence

def _feature_mask(sequence, max_mask_size=0.5):
    num_features = sequence.shape[1]
    mask_size = np.random.randint(1, int(num_features * max_mask_size) + 1)
    masked_features = np.random.choice(num_features, mask_size, replace=False)
    sequence[:, masked_features] = 0
    return sequence

def _motion_drift(imu_features: np.ndarray, drift_std, drift_max) -> np.ndarray:
    """Simulate sensor drift on IMU features."""
    T = imu_features.shape[0]
    # Generate drift signal
    drift = np.cumsum(np.random.normal(scale=drift_std, size=(T, 1)), axis=0)
    drift = np.clip(drift, -drift_max, drift_max)
    # Apply drift to features related to acceleration and angular velocity
    # acc (cols 0-2), linear_acc (cols 10-12), angular_vel (cols 15-17)
    imu_features[:, 0:3] += drift
    imu_features[:, 10:13] += drift
    imu_features[:, 15:18] += drift
    return imu_features

def _apply_augmentations(sequence, *, drift_std, drift_max, time_masking):
    """Apply augmentations to the normalized sequence."""
    # Augmentation 1: Jitter
    if np.random.rand() < 0.7:
        sequence = _jitter(sequence, sigma=0.05)
    # Augmentation 2 & 3: Time and Feature Masking
    if time_masking and np.random.rand() < 0.5:
        sequence = _time_mask(sequence, max_mask_size=20)
    if np.random.rand() < 0.5:
        sequence = _feature_mask(sequence)
    # Augmentation 4: Motion Drift (only applied to the IMU part)
    if np.random.rand() < 0.5:
        imu_features = sequence[:, :20]
        other_features = sequence[:, 20:]
        augmented_imu = _motion_drift(imu_features, drift_std=drift_std, drift_max=drift_max)
        sequence = np.concatenate([augmented_imu, other_features], axis=1)
    return sequence


class IMUDataset(Dataset):
    def __init__(self, sequence_dir, max_len=256, transform=None):
        self.sequence_dir = Path(sequence_dir)
        self.sequence_ids = sorted([seq_id.stem for seq_id in self.sequence_dir.glob("*.npy")])
        self.max_len = max_len
        self.transform = transform

    def set_mask_ratio(self, mask_ratio):
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.sequence_ids)

    @cached_property
    def labels(self):
        gestures_file = cache / 'gestures.npy'
        return np.load(gestures_file, allow_pickle=True)

    def __getitem__(self, idx):
        sequence_file = self.sequence_dir / f"{self.sequence_ids[idx]}.npy"
        seq = np.load(sequence_file, allow_pickle=True)

        label = self.labels[idx]

        output = {"inputs": seq, "label": label}
        if self.transform:
            output = self.transform(output)
        
        return output

ds = IMUDataset(imu_features_dir)
ds[0]['inputs'].shape


imu_features_engineered_dir = cache / 'imu_features_engineered'
if not imu_features_engineered_dir.exists():
    engineered = [_feature_engineering(ds[i]['inputs']) for i in tqdm(range(len(ds)))]
    imu_features_engineered_dir.mkdir(parents=True, exist_ok=True)
    for seq_id, seq in tqdm(zip(ds.sequence_ids, engineered), total=len(ds)):
        output_file = imu_features_engineered_dir / f"{seq_id}.npy"
        if not output_file.exists():
            with open(output_file, "wb") as f:
                np.save(f, seq)


ds = IMUDataset(imu_features_engineered_dir)
ds[0]['inputs'].shape


# split dataset into train and validation
train_indices, val_indices = train_test_split(range(len(ds)), test_size=0.1, random_state=42, stratify=ds.labels)
len(train_indices), len(val_indices)


scaler = StandardScaler()
train_sequences = np.vstack([ds[i]['inputs'] for i in train_indices])
assert not np.isnan(train_sequences).any(), "There should be no NaNs in the training sequences"
_ = scaler.fit(train_sequences)


class CoordAttention(nn.Module):
    """
    Coordinate Attention for Sequences.
    Input Dimension: (B, C, T)
    Output Dimension: (B, C, T)
    """
    def __init__(self, channels, reduction=8):
        super(CoordAttention, self).__init__()
        self.mid_channels = max(8, channels // reduction)

        self.compression = nn.Sequential(
            nn.Conv1d(channels, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.mid_channels),
            nn.SiLU(inplace=True)
        )
        # Attention branches
        self.time_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.channel_conv = nn.Conv1d(self.mid_channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x is expected to be (B, C, T)
        
        # Squeeze features
        f = self.compression(x)  # (B, rC, T)
        
        # Time Attention (B, 1, T)
        f_t = f.mean(dim=1, keepdim=True)
        time_attn = self.sigmoid(self.time_conv(f_t))  # Stays (B, 1, T)
        
        # Channel Attention (B, C, 1)
        f_c = f.mean(dim=2, keepdim=True) # (B, rC, 1)
        channel_attn = self.sigmoid(self.channel_conv(f_c)) # (B, C, 1)
        
        # Apply attention and return
        out = x * time_attn * channel_attn
        return out


class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, attention, reduction=8, pool_size=2, dropout=0.3):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # attention block
        self.attention = attention(out_channels, reduction) # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        # First conv
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))
        # attention block
        out = self.attention(out)
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out


class Collator:
    def __call__(self, batch):
        inputs = [item['inputs'] for item in batch]
        labels = [item['label'] for item in batch]
        inputs = pad_sequence(inputs, batch_first=True)
        labels = torch.tensor(labels, dtype=torch.long)

        return {"input": inputs, "label": labels}


n_features = ds[0]['inputs'].shape[1]

class ScaleInputs:
    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, sample):
        n_features = self.scaler.n_features_in_
        sample["inputs"][:, :n_features] = self.scaler.transform(sample["inputs"][:, :n_features])
        return sample


class AugmentForFinetune:
    def __init__(self, drift_std=0.1, drift_max=0.5, time_masking=True):
        self.drift_std = drift_std
        self.drift_max = drift_max
        self.time_masking = time_masking

    def __call__(self, sample):
        sample["inputs"] = _apply_augmentations(
            sample["inputs"],
            drift_std=self.drift_std,
            drift_max=self.drift_max,
            time_masking=self.time_masking,
        )
        return sample

class ToTensor:
    def __call__(self, sample):
        sample["inputs"] = torch.from_numpy(sample["inputs"]).float()
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
        return sample

class TfDataset(Dataset):
    def __init__(self, ds, indices, transform=None):
        self.ds = ds
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.ds[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample


finetune_train_transform = transforms.Compose([
    ScaleInputs(scaler),
    AugmentForFinetune(drift_std=0.01, drift_max=0.5, time_masking=True),
    ToTensor(),
])
finetune_val_transform = transforms.Compose([
    ScaleInputs(scaler),
    ToTensor(),
])

ft_train_ds = TfDataset(ds, train_indices, transform=finetune_train_transform)
ft_val_ds = TfDataset(ds, val_indices, transform=finetune_val_transform)
ft_train_ds[0]['inputs'].shape


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context

class MLPAttention(nn.Module):
    def __init__(self, feature_dim):
        super(MLPAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//8),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim//8, 1)
        )
    def forward(self, x):
        # inputs shape: (B, T, C)
        weights = self.attn(x)  # (B, T, 1)
        weights = F.softmax(weights, dim=1)  # (B, T, 1)
        context = (x * weights).sum(dim=1)  # (B, C)
        return context

class IMUOnlyModel(nn.Module):
    def __init__(self, imu_dim, n_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.n_classes = n_classes
        # IMU deep branch
        self.imu_block1 = ResidualCNNBlock(imu_dim, 64, 3, dropout=0.3, attention=CoordAttention)
        self.imu_block2 = ResidualCNNBlock(64, 128, 5, dropout=0.3, attention=CoordAttention)
        # BiGRU
        self.bigru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.gru_dropout = nn.Dropout(0.4)
        # Attention
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional
        self.mlp_attention = MLPAttention(256)  # MLP attention for final context aggregation
        
        # Dense layers
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, imu):
        imu = imu.transpose(1, 2)  # (batch, imu_dim, seq_len)
        # IMU branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        merged = x1.transpose(1, 2)  # (batch, seq_len, 128)
        # BiGRU 
        gru_out, _ = self.bigru(merged)
        gru_out = self.gru_dropout(gru_out)
        # Attention
        attended = self.mlp_attention(gru_out)
        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        # Classification
        logits = self.classifier(x)
        return logits


def apply_label_smoothing(targets, num_classes, smoothing_factor=0.1):
    """
    Apply label smoothing to convert hard targets to soft targets.
    Args:
        targets (torch.Tensor): Hard label indices
        num_classes (int): Total number of classes
        smoothing_factor (float): Smoothing factor, typically a small value like 0.1 
    Returns:
        torch.Tensor: Smoothed label distribution (batch_size, num_classes)
    """
    # Create a tensor of zeros with shape [batch_size, num_classes]
    smoothed_labels = torch.zeros(targets.size(0), num_classes, device=targets.device)
    # Fill in the tensor with the smoothing value
    smoothed_labels.fill_(smoothing_factor / (num_classes - 1))
    # Set the correct class with the main probability mass
    smoothed_labels.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing_factor)
    return smoothed_labels


EPOCHS = 50
LR = 1e-3
LR_MIN = 1e-6
BATCH_SIZE = 32

n_classes = len(label_encoder.classes_)
device = 'cuda'

loader_train = DataLoader(ft_train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collator())
loader_val = DataLoader(ft_val_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collator())

model = IMUOnlyModel(imu_dim=ft_train_ds[0]['inputs'].size(1), n_classes=n_classes).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)

num_training_steps = EPOCHS * len(loader_train)

train_losses = []
val_losses = []
train_accs = []
val_accs = []

warmup_epochs = 3
label_smoothing = 0.1
use_label_smoothing = label_smoothing > 0

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0.
    epoch_train_acc = 0.
    progress_bar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        base_lr = optimizer.defaults['lr']
        eta_min = LR_MIN

        current_iter = epoch * len(loader_train) + batch_idx
        current_epoch = current_iter / len(loader_train)  # Convert to decimal epoch format
        # Phase 1: Linear warmup
        if current_epoch < warmup_epochs:
            # Linear warmup: increase from 0.001 * base_lr to base_lr
            lr = base_lr * (0.001 + (current_epoch / warmup_epochs) * 0.999)
        # Phase 2: Cosine annealing
        else:
            # Cosine annealing formula
            cos_factor = 0.5 * (1 + math.cos(math.pi * (current_epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))
            lr = eta_min + (base_lr - eta_min) * cos_factor
        # Apply calculated learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        inputs = batch["input"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)

        if use_label_smoothing:
            # Convert hard labels to soft labels
            soft_labels = apply_label_smoothing(labels, n_classes, label_smoothing)
            # Use cross-entropy loss function (for soft labels, usually use log_softmax + sum directly)
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(output, dim=1),
                soft_labels,
                reduction='batchmean'
            )
        else:
            # Use original loss function
            loss = criterion(output, labels)

        acc = (output.argmax(dim=1) == labels).float().mean()

        epoch_train_loss += loss.item()
        epoch_train_acc += acc.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}", lr=f"{lr:.6f}")

    model.eval()
    epoch_val_loss = 0.
    epoch_val_acc = 0.
    progress_bar = tqdm(loader_val, desc=f"Epoch (validation) {epoch+1}/{EPOCHS}", leave=False)
    for batch in progress_bar:
        inputs = batch["input"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            output = model(inputs)
            val_loss = criterion(output, labels)
            val_acc = (output.argmax(dim=1) == labels).float().mean()

        epoch_val_loss += val_loss.item()
        epoch_val_acc += val_acc.item()
        progress_bar.set_postfix(val_loss=f"{val_loss.item():.4f}, val_acc={val_acc.item():.4f}")

    epoch_train_loss /= len(loader_train)
    epoch_train_acc /= len(loader_train)
    epoch_val_loss /= len(loader_val)
    epoch_val_acc /= len(loader_val)
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)
    print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")


fig, ax = plt.subplots(figsize=(10, 5), nrows=2, sharex=True)

ax[0].plot(train_losses, label='Train Loss', color='blue')
ax[1].plot(train_accs, label='Train Accuracy', color='blue')
ax[0].plot(val_losses, label='Validation Loss', color='orange')
ax[1].plot(val_accs, label='Validation Accuracy', color='orange')
ax[0].set_ylabel('Loss')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[0].legend()
ax[1].legend()
ax[0].grid()
ax[1].grid()
plt.tight_layout()


class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass

class CompetitionMetric:
    """Hierarchical macro F1 for the CMI 2025 challenge."""
    def __init__(self):
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures

    def calculate_hierarchical_f1(
        self,
        sol: pd.Series,
        sub: pd.Series
    ) -> float:
        invalid_types = {i for i in sub.unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )
        y_true_bin = sol.isin(self.target_gestures).values
        y_pred_bin = sub.isin(self.target_gestures).values
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
        )
        y_true_mc = sol.apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub.apply(lambda x: x if x in self.target_gestures else 'non_target')
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )
        return 0.5 * f1_binary + 0.5 * f1_macro, f1_binary, f1_macro


model.eval()
all_out_labels = []
gt_labels = []
progress_bar = tqdm(loader_val, desc=f"Epoch (validation) {epoch+1}/{EPOCHS}", leave=False)
for batch in progress_bar:
    inputs = batch["input"].to(device)
    labels = batch["label"].to(device)

    with torch.no_grad():
        output = model(inputs)
        out_labels = output.argmax(dim=1)

    all_out_labels.extend([i.item() for i in out_labels])
    gt_labels.extend([i.item() for i in labels])


metric = CompetitionMetric()
all_out_labels = pd.Series(all_out_labels, name='gesture').map(lambda x: label_encoder.inverse_transform([x])[0])
gt_labels = pd.Series(gt_labels, name='gesture_gt').map(lambda x: label_encoder.inverse_transform([x])[0])
hierarchical_f1, f1_binary, f1_macro = metric.calculate_hierarchical_f1(gt_labels, all_out_labels)
print(f"Hierarchical F1: {hierarchical_f1:.4f}, Binary F1: {f1_binary:.4f}, Macro F1: {f1_macro:.4f}")




