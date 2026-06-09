import os
import glob
import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.cuda.amp import autocast, GradScaler
# !pip install torch-optimizer
# import torch_optimizer as optim
# !pip install torchinfo
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm.auto import tqdm
from pathlib import Path
import time
import warnings
import copy
from copy import deepcopy


# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Config:
    BASE_PATH = Path("/kaggle/input/waveform-inversion")
    TRAIN_PATH = BASE_PATH / "train_samples"
    TEST_PATH = BASE_PATH / "test"

    # Data dimensions
    SEISMIC_NUM_SOURCES = 5
    SEISMIC_TIME_STEPS = 1000
    SEISMIC_NUM_RECEIVERS = 70
    VELOCITY_MAP_HEIGHT = 70
    VELOCITY_MAP_WIDTH = 70
    POS_ENC_CHANNELS = 4

    # Model params
    UNET_INPUT_CHANNELS = SEISMIC_NUM_SOURCES * 2 + POS_ENC_CHANNELS
    UNET_OUTPUT_CHANNELS = 1
    BASE_CHANNELS = 96
    BILINEAR = True
    USE_TTA = True
    USE_EMA = True
    USE_SCSE = True

    # Training params
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100 # Increased epochs since EarlyStopping is used
    VALIDATION_SPLIT = 0.15
    RANDOM_SEED = 42
    PATIENCE = 10  # Increased patience for early stopping
    WEIGHT_DECAY = 1e-3

    # Loss params
    # ALPHA = 0.85
    # BETA  = 0.15
    W_MAE = 0.7
    W_SSIM = 0.15
    W_GRAD = 0.15

    
    # Normalization (to be computed)
    SEISMIC_MEAN = None
    SEISMIC_STD = None
    VELOCITY_MEAN = None
    VELOCITY_STD = None
    GROUP_STATS = None

    # Augmentation/Transformation
    MAX_SEISMIC_SHIFT = 50
    LOG_TRANSFORM_VELOCITY = True

    # Augmentation parameters
    NOISE_STD = 0.015  # Relative noise std for Gaussian noise
    RECEIVER_DROP_PROB = 0.1
    MAX_RECEIVER_DROPS = 2
    SCALE_MIN = 0.9
    SCALE_MAX = 1.1
    TRANSLATE_PIXELS = 2

    # Dropout for residual blocks
    RES_BLOCK_DROPOUT = 0.1

    # Sampling for stats
    NORMALIZATION_SAMPLE_FRACTION = 1
    CACHE_SIZE = 50
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()


# Reproducibility
torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
random.seed(cfg.RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = cfg.RANDOM_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def scan_files(root_dir: Path) -> list[dict[str, Path | str]]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory {root_dir} does not exist.")

    file_pairs: list[dict[str, Path | str]] = []
    for item in sorted(root_dir.iterdir()):
        if not item.is_dir():
            continue

        name = item.name
        if ("Vel" in name) or ("Style" in name):
            group = "Vel" if "Vel" in name else "Style"
            data_dir = item / "data"
            model_dir = item / "model"
            if data_dir.exists() and model_dir.exists():
                for data_file in sorted(data_dir.glob("data*.npy")):
                    idx_match = re.search(r"data(\d+)\.npy", data_file.name)
                    if idx_match:
                        idx = idx_match.group(1)
                        model_file = model_dir / f"model{idx}.npy"
                        if model_file.exists():
                            file_pairs.append({
                                "input": data_file,
                                "target": model_file,
                                "group": group
                            })
        elif "Fault" in name:
            for seis_file in sorted(item.glob("seis*.npy")):
                base_name = seis_file.name.replace("seis", "vel")
                vel_file = item / base_name
                if vel_file.exists():
                    file_pairs.append({
                        "input": seis_file,
                        "target": vel_file,
                        "group": "Fault"
                    })
    return file_pairs


def create_stratified_split(
    pairs: List[Dict[str, Path]],
    val_frac: float = 0.15
) -> Tuple[List[Dict[str, Path]], List[Dict[str, Path]]]:
    """
    Stratify by directory type so each family (Vel/Style/Fault)
    appears in both train and val.
    Returns full dictionaries with 'input', 'target', 'group'.
    """
    groups = {"Vel": [], "Style": [], "Fault": []}
    for p in pairs:
        group = p.get("group")
        if group in groups:
            groups[group].append(p)
        else:
            print(f"Warning: Unknown group type in {p}")

    train, val = [], []
    for group_name, items in groups.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    print(f"Stratified split → Train: {len(train)}, Val: {len(val)}")
    return train, val


def compute_stratified_stats(file_pairs):
    group_stats = {}

    for group in ['Vel', 'Style', 'Fault']:
        group_files = [(p['input'], p['target']) for p in file_pairs if p['group'] == group]
        if not group_files:
            continue

        ds = StatsDataset(group_files, log_transform_velocity=cfg.LOG_TRANSFORM_VELOCITY)
        loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS)

        s_mean, s_std, v_mean, v_std = compute_stats_gpu(loader, sample_fraction=1.0)
        group_stats[group] = {
            "seismic_mean": s_mean,
            "seismic_std": s_std,
            "vel_mean": v_mean,
            "vel_std": v_std,
        }
        print(f"[{group}] SEISMIC μ={s_mean:.2f}, σ={s_std:.2f} | VELOCITY μ={v_mean:.2f}, σ={v_std:.2f}")

    return group_stats


class StatsDataset(Dataset):
    """
    Dataset to compute normalization statistics on the final 14-channel processed data
    (10 FFT channels + 4 Positional Encoding channels).
    """
    def __init__(self, file_paths, log_transform_velocity=False, cache_in_memory=False):
        self.file_metadata = []
        self.log_transform_velocity = log_transform_velocity
        self.cache_in_memory = cache_in_memory
        self.seismic_cache = {}
        self.velocity_cache = {}

        # Pre-generate the positional encoding once, as it's the same for all samples
        self.pos_encoding = generate_positional_encoding(
            cfg.VELOCITY_MAP_HEIGHT,
            cfg.VELOCITY_MAP_WIDTH,
            cfg.POS_ENC_CHANNELS
        )

        for seismic_path, vel_path in file_paths:
            if self.cache_in_memory:
                self.seismic_cache[seismic_path] = np.load(seismic_path)
                self.velocity_cache[vel_path] = np.load(vel_path)

            # Correctly load the array to get its shape without a 'with' statement
            data = np.load(seismic_path)
            num_samples = data.shape[0]

            self.file_metadata.append((seismic_path, vel_path, num_samples))

        self.indices = []
        for file_idx, (_, _, num_samples) in enumerate(self.file_metadata):
            self.indices.extend([(file_idx, i) for i in range(num_samples)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.indices[idx]
        seismic_path, vel_path, _ = self.file_metadata[file_idx]

        if self.cache_in_memory:
            seismic_data = self.seismic_cache[seismic_path]
            vel_data = self.velocity_cache[vel_path]
        else:
            seismic_data = np.load(seismic_path, mmap_mode='r')
            vel_data = np.load(vel_path, mmap_mode='r')

        seismic = torch.from_numpy(seismic_data[sample_idx].copy()).float()
        vel = torch.from_numpy(vel_data[sample_idx].copy()).float()

        # 1. Perform FFT and separate into Real/Imaginary parts
        ffted = torch.fft.fft(seismic, dim=1)
        real_part = ffted.real[:, :cfg.SEISMIC_NUM_RECEIVERS, :]
        imag_part = ffted.imag[:, :cfg.SEISMIC_NUM_RECEIVERS, :]
        seismic_fft = torch.cat([real_part, imag_part], dim=0)

        # 2. Concatenate with positional encoding to create the final 14-channel tensor
        seismic = torch.cat([seismic_fft, self.pos_encoding], dim=0)

        # Process velocity map
        if self.log_transform_velocity:
            vel = torch.log(vel + 1e-6)

        if vel.dim() == 2:
            vel = vel.unsqueeze(0)

        return seismic, vel


class RunningStats:
    """Online mean & std via Welford's algorithm."""
    def __init__(self, device='cuda'):
        self.n = torch.tensor(0, dtype=torch.long, device=device)
        self.mean = torch.tensor(0.0, device=device)
        self.M2 = torch.tensor(0.0, device=device)

    def update(self, x):
        with torch.no_grad():
            flat = x.detach().flatten()
            batch_n = flat.numel()
            batch_mean = flat.mean()
            batch_M2 = flat.var(unbiased=False) * batch_n

            delta = batch_mean - self.mean
            total_n = self.n + batch_n

            self.mean = (self.n * self.mean + batch_n * batch_mean) / total_n
            self.M2 = self.M2 + batch_M2 + delta**2 * self.n * batch_n / total_n
            self.n = total_n

    @property
    def std(self):
        return torch.sqrt(self.M2 / self.n)


def compute_stats_gpu(dataloader, sample_fraction=0.1):
    """Compute mean/std on GPU for seismic and velocity."""
    seismic_stats = RunningStats(device=cfg.DEVICE)
    velocity_stats = RunningStats(device=cfg.DEVICE)

    n_samples = int(len(dataloader.dataset) * sample_fraction)
    count = 0

    with torch.no_grad():
        for seismic, velocity in dataloader:
            seismic = seismic.to(cfg.DEVICE, non_blocking=True)
            velocity = velocity.to(cfg.DEVICE, non_blocking=True)

            seismic_stats.update(seismic)
            velocity_stats.update(velocity)

            count += seismic.shape[0]
            if count >= n_samples:
                break

    return (
        seismic_stats.mean.item(), seismic_stats.std.item(),
        velocity_stats.mean.item(), velocity_stats.std.item()
    )


class WaveformInversionDataset(Dataset):
    """
    Main dataset for training and validation. It creates the 14-channel input,
    applies group-aware normalization, and performs data augmentation.
    """
    def __init__(self, file_pairs, group_stats,
                 augment=False, light_augment=False,
                 log_transform_velocity=False, cache_in_memory=False):
        self.pairs = file_pairs
        self.group_stats = group_stats
        self.augment = augment
        self.light_augment = light_augment
        self.log_transform_velocity = log_transform_velocity
        self.cache_in_memory = cache_in_memory
        self.seismic_cache = {}
        self.velocity_cache = {}

        # Pre-generate the positional encoding once
        self.pos_encoding = generate_positional_encoding(
            cfg.VELOCITY_MAP_HEIGHT,
            cfg.VELOCITY_MAP_WIDTH,
            cfg.POS_ENC_CHANNELS
        )

        self.metadata = []
        for pair in self.pairs:
            seismic_path = pair["input"]
            vel_path = pair["target"]
            group = pair["group"]

            if cache_in_memory:
                self.seismic_cache[seismic_path] = np.load(seismic_path)
                self.velocity_cache[vel_path] = np.load(vel_path)
            
            ### --- CORRECTED CODE --- ###
            # The 'with' statement is removed to fix the TypeError.
            data = np.load(seismic_path)
            num_samples = data.shape[0]
            ### --- END CORRECTION --- ###

            self.metadata.append((seismic_path, vel_path, num_samples, group))

        self.indices = []
        for i, (_, _, num_samples, _) in enumerate(self.metadata):
            self.indices.extend([(i, j) for j in range(num_samples)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        meta_idx, sample_idx = self.indices[idx]
        seismic_path, vel_path, _, group = self.metadata[meta_idx]
        stats = self.group_stats[group]

        if self.cache_in_memory:
            seismic_data = self.seismic_cache[seismic_path]
            vel_data = self.velocity_cache[vel_path]
        else:
            seismic_data = np.load(seismic_path, mmap_mode='r')
            vel_data = np.load(vel_path, mmap_mode='r')

        seismic = torch.from_numpy(seismic_data[sample_idx].copy()).float()
        vel = torch.from_numpy(vel_data[sample_idx].copy()).float()

        if self.log_transform_velocity:
            vel = torch.log(vel + 1e-6)

        # 1. Perform FFT and separate into Real/Imaginary parts
        ffted = torch.fft.fft(seismic, dim=1)
        real_part = ffted.real[:, :cfg.SEISMIC_NUM_RECEIVERS, :]
        imag_part = ffted.imag[:, :cfg.SEISMIC_NUM_RECEIVERS, :]
        seismic_fft = torch.cat([real_part, imag_part], dim=0)

        # 2. Concatenate with positional encoding
        seismic_processed = torch.cat([seismic_fft, self.pos_encoding], dim=0)

        # 3. Apply normalization to the final 14-channel tensor
        seismic = (seismic_processed - stats["seismic_mean"]) / stats["seismic_std"]

        # 4. Apply data augmentation (if specified)
        if self.augment:
            seismic += torch.randn_like(seismic) * cfg.NOISE_STD
            scale = random.uniform(cfg.SCALE_MIN, cfg.SCALE_MAX)
            seismic *= scale
            if random.random() < cfg.RECEIVER_DROP_PROB:
                for _ in range(random.randint(1, cfg.MAX_RECEIVER_DROPS)):
                    # Only drop FFT channels, not positional encoding channels
                    ch = random.randint(0, 10 - 1) 
                    seismic[ch, :, :] = 0.0

        elif self.light_augment:
            seismic += torch.randn_like(seismic) * (cfg.NOISE_STD * 0.5)
            seismic *= random.uniform(0.98, 1.02)

        if vel.dim() == 2:
            vel = vel.unsqueeze(0)

        vel = (vel - stats["vel_mean"]) / stats["vel_std"]
        
        return seismic, vel, group


def generate_positional_encoding(height, width, channels):
    """
    Generates a 2D positional encoding map of shape (channels, height, width).
    """
    if channels % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd number of channels or channels not divisible by 4.")

    pos_encoding = torch.zeros(channels, height, width)
    
    # Create a grid of coordinates
    y_pos, x_pos = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    # Define the division term for the sine/cosine frequencies
    div_term = torch.exp(torch.arange(0, channels // 2, 2) * -(math.log(10000.0) / (channels // 2)))

    # Calculate positional encoding for x coordinate
    pos_encoding[0::2, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
    pos_encoding[1::2, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
    
    # Calculate positional encoding for y coordinate
    # Note: We apply this to the second half of the channels
    y_channel_offset = channels // 2
    pos_encoding[y_channel_offset::2, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
    pos_encoding[y_channel_offset+1::2, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))

    return pos_encoding


class ResidualDoubleConv(nn.Module):
    """Residual double‐conv block with dropout."""
    def __init__(self, in_ch, out_ch, dropout=cfg.RES_BLOCK_DROPOUT):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.relu2 = nn.ReLU()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.dropout(out)
        return self.relu2(out + identity)


class SCSEBlock(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation (SCSE) block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Down(nn.Module):
    """Downscaling with maxpool then residual double‐conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualDoubleConv(in_ch, out_ch, dropout=cfg.RES_BLOCK_DROPOUT)
        )

    def forward(self, x):
        return self.layer(x)


class AttentionBlock(nn.Module):
    """Attention block used in Attention U-Net."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(torch.relu(g1 + x1))
        return x * psi


class Up(nn.Module):
    """Upscaling then attention then residual double‐conv."""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.attn = AttentionBlock(F_g=in_ch // 2, F_l=in_ch // 2, F_int=in_ch // 4)
        self.conv = ResidualDoubleConv(in_ch, out_ch, dropout=cfg.RES_BLOCK_DROPOUT)
        self.scse = SCSEBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # If spatial sizes don't match, apply padding
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)

        if diffY != 0 or diffX != 0:
            assert abs(diffY) <= 2 and abs(diffX) <= 2, (
                f"Padding too large: x={x.shape}, skip={skip.shape}, "
                f"diffY={diffY}, diffX={diffX}"
            )

            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        # Attention and concat
        attn_out = self.attn(g=x, x=skip)
        x = torch.cat([attn_out, x], dim=1)
        return self.scse(self.conv(x))


class OutConv(nn.Module):
    """Final 1×1 convolution to produce output channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Attention U-Net with residual double-conv blocks, modified for multi-scale outputs.
    """
    def __init__(self, n_channels, n_classes, bilinear=cfg.BILINEAR, base_channels=cfg.BASE_CHANNELS):
        super().__init__()
        B = base_channels
        self.inc = ResidualDoubleConv(n_channels, B, dropout=cfg.RES_BLOCK_DROPOUT)
        self.down1 = Down(B, B*2)
        self.down2 = Down(B*2, B*4)
        self.down3 = Down(B*4, B*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(B*8, B*16 // factor)

        self.up1 = Up(B*16, B*8 // factor, bilinear)
        self.up2 = Up(B*8, B*4 // factor, bilinear)
        self.up3 = Up(B*4, B*2 // factor, bilinear)
        self.up4 = Up(B*2, B, bilinear)
        
        # The final, full-resolution output convolution
        self.outc_final = OutConv(B, n_classes) ### <-- MODIFIED (renamed for clarity)
        
        ### --- NEW: Prediction heads for intermediate scales --- ###
        # Prediction head after up2 (1/4 resolution)
        self.outc_scale2 = OutConv(B*4 // factor, n_classes)
        # Prediction head after up3 (1/2 resolution)
        self.outc_scale1 = OutConv(B*2 // factor, n_classes)
        ### --- END NEW --- ###


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        # Final full-resolution prediction
        logits_final = self.outc_final(u4)
        
        ### --- NEW: Generate predictions from intermediate layers --- ###
        logits_scale1 = self.outc_scale1(u3) # Prediction at 1/2 resolution
        logits_scale2 = self.outc_scale2(u2) # Prediction at 1/4 resolution
        ### --- END NEW --- ###

        # Return a list of predictions, from lowest resolution to highest
        return [logits_scale2, logits_scale1, logits_final]


def create_sobel_filters(device):
    """Creates Sobel filters for X and Y gradients, moved to the specified device."""
    # Sobel filter for the x-gradient
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    # Sobel filter for the y-gradient
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    # Move filters to the same device as the model
    return sobel_x.to(device), sobel_y.to(device)


def gradient_loss(pred, target, sobel_x, sobel_y):
    """
    Calculates the L1 loss between the gradients of the prediction and the target.
    """
    # Calculate gradients for the prediction
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    
    # Calculate gradients for the target
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)
    
    # Calculate L1 loss on the gradients
    loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    
    return loss


def rescale_to_unit_range(x):
    """
    Safely rescales a tensor to the [0, 1] range for stable SSIM calculation.
    """
    min_val = x.amin(dim=(1, 2, 3), keepdim=True)
    max_val = x.amax(dim=(1, 2, 3), keepdim=True)
    # The epsilon in the denominator prevents division by zero for flat images
    return (x - min_val) / (max_val - min_val + 1e-8)


def safe_ssim_loss(pred, target):
    """
    Calculates a numerically stable SSIM loss.
    Returns the loss value (1.0 - SSIM).
    """
    # Rescale both tensors independently to the [0, 1] range
    pred_norm = rescale_to_unit_range(pred)
    target_norm = rescale_to_unit_range(target)
    
    # Calculate SSIM on the rescaled tensors with a fixed data_range of 1.0
    ssim_val = structural_similarity_index_measure(pred_norm, target_norm, data_range=1.0)
    
    # The SSIM loss is 1 - ssim
    ssim_term = 1.0 - ssim_val
    
    return ssim_term


def safe_ssim_metric(pred, target):
    """
    Calculates a numerically stable SSIM score (not the loss).
    """
    # Use the same rescaling helper as our loss function
    pred_norm = rescale_to_unit_range(pred)
    target_norm = rescale_to_unit_range(target)
    
    # Calculate and return the SSIM score on the rescaled tensors
    return structural_similarity_index_measure(pred_norm, target_norm, data_range=1.0)


# class CombinedLoss(nn.Module):
#     """
#     A combined loss function that includes MAE, a stable SSIM, and a gradient (Sobel) loss.
#     Loss = w_mae * MAE + w_ssim * SSIM_Loss + w_grad * Gradient_Loss
#     """
#     def __init__(self, w_mae=cfg.W_MAE, w_ssim=cfg.W_SSIM, w_grad=cfg.W_GRAD, device=cfg.DEVICE):
#         super().__init__()
#         self.w_mae = w_mae
#         self.w_ssim = w_ssim
#         self.w_grad = w_grad
        
#         # Create and store the sobel filters on the correct device
#         self.sobel_x, self.sobel_y = create_sobel_filters(device)

#     def forward(self, pred, target):
#         # 1. Mean Absolute Error (L1 Loss)
#         mae_term = F.l1_loss(pred, target)
        
#         # 2. Stable SSIM Loss (using our safe helper function)
#         ssim_term = safe_ssim_loss(pred, target)
        
#         # 3. Gradient Loss (using our sobel helper function)
#         grad_term = gradient_loss(pred, target, self.sobel_x, self.sobel_y)
        
#         # Combine the losses with their respective weights
#         total_loss = (self.w_mae * mae_term) + \
#                      (self.w_ssim * ssim_term) + \
#                      (self.w_grad * grad_term)
        
#         return total_loss


class CombinedLoss(nn.Module):
    """
    A combined loss function adapted for multi-scale supervision.
    """
    def __init__(self, w_mae=cfg.W_MAE, w_ssim=cfg.W_SSIM, w_grad=cfg.W_GRAD, device=cfg.DEVICE):
        super().__init__()
        self.w_mae = w_mae
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.sobel_x, self.sobel_y = create_sobel_filters(device)

    def forward(self, preds, target): ### <-- MODIFIED: 'pred' is now 'preds' (a list)
        
        total_loss = 0
        
        # Iterate through the predictions from each scale
        for p in preds:
            # Downsample the ground truth target to match the prediction's size
            # mode='area' is generally good for downsampling.
            resized_target = F.interpolate(target, size=p.shape[2:], mode='area')
            
            # --- Calculate the combined loss for the current scale ---
            mae_term = F.l1_loss(p, resized_target)
            ssim_term = safe_ssim_loss(p, resized_target)
            grad_term = gradient_loss(p, resized_target, self.sobel_x, self.sobel_y)
            
            scale_loss = (self.w_mae * mae_term) + \
                         (self.w_ssim * ssim_term) + \
                         (self.w_grad * grad_term)
            
            # Add the loss for the current scale to the total
            total_loss += scale_loss
            
        return total_loss


# class MAE_SSIM_Loss(nn.Module):
#     """
#     Combines L1 (MAE) with SSIM as a composite loss.
#     Total loss = alpha * MAE + beta * (1 - SSIM)
#     """
#     def __init__(self, alpha=cfg.ALPHA, beta=cfg.BETA):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta

#     def forward(self, pred, target):
#         mae_loss = F.l1_loss(pred, target)
#         # For SSIM, we expect the input to be in a predictable range, e.g. [0, 1] or [-1, 1]
#         # Since our targets are normalized around 0, we can use a large data_range.
#         # Alternatively, clamp to a range if we know it. Here, we assume a reasonable range after normalization.
#         data_range = target.max() - target.min()
#         if data_range < 1e-6: # Handle case of blank targets
#             data_range = 1.0

#         ssim_val = structural_similarity_index_measure(pred, target, data_range=data_range)
#         ssim_loss = 1.0 - ssim_val
#         return self.alpha * mae_loss + self.beta * ssim_loss


class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if device:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                model_v = model_v.to(ema_v.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, lambda e, m: m)


# def denormalize_velocity(norm_vel, groups):
#     """
#     Denormalize batched velocity using per-sample group stats.
#     groups: list of strings (length = batch_size)
#     """
#     out = []
#     for i, group in enumerate(groups):
#         stats = cfg.GROUP_STATS[group]
#         v = norm_vel[i] * stats["vel_std"] + stats["vel_mean"]
#         if cfg.LOG_TRANSFORM_VELOCITY:
#             v = torch.exp(v)
#         out.append(v.unsqueeze(0))  # keep batch dimension

#     return torch.cat(out, dim=0)


def denormalize_velocity(norm_vel, groups):
    """
    Denormalize batched velocity using per-sample group stats.
    Includes a clamp for numerical stability.
    """
    out = []
    for i, group in enumerate(groups):
        stats = cfg.GROUP_STATS[group]
        v = norm_vel[i] * stats["vel_std"] + stats["vel_mean"]
        if cfg.LOG_TRANSFORM_VELOCITY:
            # Clamp before the exponential to prevent overflow to infinity
            v = torch.clamp(v, max=20.0)
            v = torch.exp(v)
        out.append(v.unsqueeze(0))  # keep batch dimension

    return torch.cat(out, dim=0)


def denormalize_velocity_global(norm_vel):
    """Denormalize batched velocity using global stats for inference."""
    v = norm_vel * cfg.VELOCITY_STD + cfg.VELOCITY_MEAN
    if cfg.LOG_TRANSFORM_VELOCITY:
        v = torch.exp(v)
    return v


def plot_model_metrics(history, save_path='model_metrics.png'):
    """Plot MAE & SSIM curves, overfitting gap, and smoothed MAE."""
    epochs = np.arange(1, len(history['train_denorm_mae']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1) MAE curves
    axes[0, 0].plot(epochs, history['train_denorm_mae'], 'b-', label='Train MAE', linewidth=2)
    axes[0, 0].plot(epochs, history['val_denorm_mae'], 'r-', label='Val MAE', linewidth=2)
    axes[0, 0].set_title('MAE Curves (Denormalized)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2) SSIM curves
    axes[0, 1].plot(epochs, history['train_ssim'], 'b-', label='Train SSIM', linewidth=2)
    axes[0, 1].plot(epochs, history['val_ssim'], 'r-', label='Val SSIM', linewidth=2)
    axes[0, 1].set_title('SSIM Curves', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3) Overfitting gap (MAE difference)
    mae_gap = np.array(history['val_denorm_mae']) - np.array(history['train_denorm_mae'])
    axes[1, 0].plot(epochs, mae_gap, 'purple', linewidth=2)
    axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Overfitting Gap (Val MAE – Train MAE)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE Gap')
    axes[1, 0].grid(alpha=0.3)

    # 4) Smoothed MAE
    window = max(3, len(epochs) // 10)
    if len(epochs) >= window:
        smooth_train = np.convolve(history['train_denorm_mae'], np.ones(window) / window, mode='valid')
        smooth_val = np.convolve(history['val_denorm_mae'], np.ones(window) / window, mode='valid')
        smooth_epochs = epochs[window - 1:]
        axes[1, 1].plot(epochs, history['train_denorm_mae'], 'b-', alpha=0.3, label='Train MAE', linewidth=1)
        axes[1, 1].plot(epochs, history['val_denorm_mae'], 'r-', alpha=0.3, label='Val MAE', linewidth=1)
        axes[1, 1].plot(smooth_epochs, smooth_train, 'b-', linewidth=2, label=f'Train MA({window})')
        axes[1, 1].plot(smooth_epochs, smooth_val, 'r-', linewidth=2, label=f'Val MA({window})')
        axes[1, 1].set_title('Smoothed MAE Learning Curves', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "Not enough epochs to smooth", ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_scatter_and_residuals(model, val_loader, save_path='scatter_residuals.png'):
    """
    Plots predicted vs actual values, residuals, and error distribution.
    Updated to handle multi-scale model outputs.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, groups in tqdm(val_loader, desc="Gathering eval data"):
            inputs = inputs.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            # model(inputs) returns a list of predictions for different scales
            outputs = model(inputs)

            ### --- CORRECTED CODE --- ###
            # 1. Select the final, full-resolution prediction from the list.
            final_pred = outputs[-1]

            # 2. Pass this single tensor to the denormalization function.
            pred_denorm = denormalize_velocity(final_pred, groups).cpu().numpy().flatten()
            ### --- END CORRECTION --- ###
            
            tgt_denorm = denormalize_velocity(targets, groups).cpu().numpy().flatten()

            all_preds.append(pred_denorm)
            all_targets.append(tgt_denorm)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    residuals = all_preds - all_targets

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter: Pred vs Actual
    axes[0].scatter(all_targets, all_preds, alpha=0.3, s=1)
    axes[0].plot([all_targets.min(), all_targets.max()],
                 [all_targets.min(), all_targets.max()],
                 'r--', linewidth=2)
    axes[0].set_title(f'Pred vs Actual\nR²={r2:.4f}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Actual Velocity')
    axes[0].set_ylabel('Predicted Velocity')
    axes[0].grid(alpha=0.3)

    # Residuals
    axes[1].scatter(all_targets, residuals, alpha=0.3, s=1)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title(f'Residuals\nMAE={mae:.4f}, RMSE={rmse:.4f}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Actual Velocity')
    axes[1].set_ylabel('Residual (Pred – Actual)')
    axes[1].grid(alpha=0.3)

    # Error distribution
    axes[2].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[2].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[2].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Residual')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}


def visualize_predictions(model, loader, group_stats, num_samples=4):
    """
    Visualizes model predictions with group-aware denormalization.
    Updated to handle multi-scale model outputs.
    """
    model.eval()
    count = 0

    with torch.no_grad():
        for inputs, targets, groups in loader:
            inputs = inputs.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            
            # model(inputs) returns a list of predictions
            outputs = model(inputs)

            ### --- CORRECTED CODE --- ###
            # 1. Select the final, full-resolution prediction from the list.
            final_pred = outputs[-1]

            # 2. Pass this single tensor to the denormalization function.
            denorm_pred = denormalize_velocity(final_pred, groups)
            ### --- END CORRECTION --- ###
            
            denorm_target = denormalize_velocity(targets, groups)

            for i in range(inputs.size(0)):
                # Take first source as representative seismic input
                # This shows the Real part of the FFT for the first source
                input_img = inputs[i, 0].cpu().numpy() 
                pred_img = denorm_pred[i].squeeze().cpu().numpy()
                target_img = denorm_target[i].squeeze().cpu().numpy()
                group = groups[i]

                # Compute MAE and SSIM for this single sample
                pred_tensor = denorm_pred[i].unsqueeze(0)
                target_tensor = denorm_target[i].unsqueeze(0)
                mae = F.l1_loss(pred_tensor, target_tensor).item()
                
                # Use our stable SSIM metric function
                ssim_score = safe_ssim_metric(pred_tensor, target_tensor).item()

                # Plot
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(input_img, cmap='gray')
                axs[0].set_title("Seismic Input (Source 1, Real Part)")

                axs[1].imshow(pred_img, cmap='viridis')
                axs[1].set_title(f"Prediction\nMAE: {mae:.1f} | SSIM: {ssim_score:.3f}")

                axs[2].imshow(target_img, cmap='viridis')
                axs[2].set_title(f"Ground Truth\nGroup: {group}")

                for ax in axs:
                    ax.axis("off")

                plt.tight_layout()
                plt.show()

                count += 1
                if count >= num_samples:
                    return


class EarlyStopping:
    """
    Early stops if validation denormalized MAE doesn't improve after `patience` epochs.
    Saves the best model.
    """
    def __init__(self, patience: int = cfg.PATIENCE, min_delta: float = 0.0, path: str = 'best_model.pth', verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_denorm_mae: float, model_to_save: nn.Module):
        if self.best_score is None:
            self.best_score = val_denorm_mae
            self._save_checkpoint(model_to_save)
        elif val_denorm_mae < self.best_score - self.min_delta:
            self.best_score = val_denorm_mae
            self._save_checkpoint(model_to_save)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model_to_save: nn.Module):
        torch.save(model_to_save.state_dict(), self.path)
        if self.verbose:
            print(f"Validation MAE improved to {self.best_score:.4f}. Saving model to {self.path}")


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs):
    """
    The main training and validation loop, updated with all enhancements.
    """
    history = {
        "train_loss": [], "val_loss": [],
        "train_denorm_mae": [], "val_denorm_mae": [],
        "train_ssim": [], "val_ssim": [],
        "val_mae_group": {}, "val_ssim_group": {},
        "ema_val_denorm_mae": [], "ema_val_ssim": [],
        "ema_val_mae_group": {}, "ema_val_ssim_group": {}
    }

    # Use our new sophisticated and stable loss function
    criterion = CombinedLoss()
    
    early_stopper = EarlyStopping(path='best_model.pth', verbose=True)
    ema = ModelEMA(model, decay=0.99, device=cfg.DEVICE)
    scaler = GradScaler()

    def validate(model_to_eval, loader, label):
        """Inner function to perform validation."""
        model_to_eval.eval()
        loss_total, denorm_mae_total, ssim_total = 0, 0, 0
        group_mae, group_ssim = defaultdict(list), defaultdict(list)

        with torch.no_grad():
            for inputs, targets, groups in loader:
                inputs = inputs.to(cfg.DEVICE)
                targets = targets.to(cfg.DEVICE)

                with autocast():
                    outputs = model_to_eval(inputs)
                    loss = criterion(outputs, targets)

                loss_total += loss.item()

                # Use the final, full-resolution prediction for metrics
                final_pred = outputs[-1]

                denorm_out = denormalize_velocity(final_pred, groups)
                denorm_tar = denormalize_velocity(targets, groups)

                for i, group in enumerate(groups):
                    group_mae[group].append(F.l1_loss(denorm_out[i], denorm_tar[i]).item())
                    
                    # Use our numerically stable SSIM metric function to prevent NaNs
                    ssim_score = safe_ssim_metric(denorm_out[i].unsqueeze(0), denorm_tar[i].unsqueeze(0))
                    group_ssim[group].append(ssim_score.item())
                    
        avg_group_mae = {g: np.mean(v) for g, v in group_mae.items()}
        avg_group_ssim = {g: np.mean(v) for g, v in group_ssim.items()}
        denorm_mae = np.mean([mae for v in group_mae.values() for mae in v])
        ssim = np.mean([s for v in group_ssim.values() for s in v])
        avg_loss = loss_total / len(loader)

        print(f"[{label}] Loss: {avg_loss:.4f} | MAE: {denorm_mae:.2f} | SSIM: {ssim:.3f}")
        return avg_loss, denorm_mae, ssim, avg_group_mae, avg_group_ssim

    # --- Main Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_mae, train_ssim = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training")
        for inputs, targets, groups in pbar:
            inputs = inputs.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)

            optimizer.zero_grad()

            # Forward pass with mixed-precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            train_loss += loss.item()

            # Calculate training metrics for logging
            with torch.no_grad():
                # Use the final, full-resolution prediction for metrics
                final_pred = outputs[-1]
                denorm_out = denormalize_velocity(final_pred, groups)
                denorm_tar = denormalize_velocity(targets, groups)
                train_mae += F.l1_loss(denorm_out, denorm_tar).item()
                
                # Use our numerically stable SSIM metric function to prevent NaNs
                train_ssim += safe_ssim_metric(denorm_out, denorm_tar).item()

        # --- End of Epoch: Calculate and Print Metrics ---
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        train_ssim /= len(train_loader)

        # Print training metrics
        print(f"[TRAIN] Loss: {train_loss:.4f} | MAE: {train_mae:.2f} | SSIM: {train_ssim:.3f}")

        # Perform validation
        val_loss, val_mae, val_ssim, val_grp_mae, val_grp_ssim = validate(model, val_loader, "VAL (Raw)")
        ema_loss, ema_mae, ema_ssim, ema_grp_mae, ema_grp_ssim = validate(ema.module, val_loader, "VAL (EMA)")

        # Step the scheduler based on the validation MAE
        scheduler.step(ema_mae) 

        # Log history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_denorm_mae"].append(train_mae)
        history["val_denorm_mae"].append(val_mae)
        history["train_ssim"].append(train_ssim)
        history["val_ssim"].append(val_ssim)
        history["val_mae_group"][epoch] = val_grp_mae
        history["val_ssim_group"][epoch] = val_grp_ssim
        history["ema_val_denorm_mae"].append(ema_mae)
        history["ema_val_ssim"].append(ema_ssim)
        history["ema_val_mae_group"][epoch] = ema_grp_mae
        history["ema_val_ssim_group"][epoch] = ema_grp_ssim

        # Check for early stopping based on EMA model's performance
        early_stopper(ema_mae, ema.module)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best EMA MAE: {early_stopper.best_score:.4f}")
    return history


def visualize_seismic_input(val_loader, num_samples=3, save_path='seismic_input.png'):
    """Visualize the processed seismic input data (FFT magnitude)"""
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            break

    indices = np.random.choice(len(inputs), min(num_samples, len(inputs)), replace=False)

    fig, axes = plt.subplots(cfg.SEISMIC_NUM_SOURCES, num_samples, figsize=(6 * num_samples, 15))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, idx in enumerate(indices):
        seismic = inputs[idx].cpu().numpy()  # Shape: (5, 70, 70)

        for source in range(cfg.SEISMIC_NUM_SOURCES):
            im = axes[source, i].imshow(seismic[source], cmap='viridis', aspect='auto')
            axes[source, i].set_title(f'Sample {idx+1}, Source {source+1}\n(FFT Magnitude)',
                                    fontsize=12, fontweight='bold')
            axes[source, i].set_xlabel('Receiver Position')
            axes[source, i].set_ylabel('Frequency Bin')
            plt.colorbar(im, ax=axes[source, i], shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_groupwise_val_mae(history, title="Validation MAE by Group"):
    group_names = set()
    for epoch_data in history["val_mae_group"].values():
        group_names.update(epoch_data.keys())

    group_names = sorted(group_names)
    epochs = sorted(history["val_mae_group"].keys())

    for group in group_names:
        y = [history["val_mae_group"][epoch].get(group, float('nan')) for epoch in epochs]
        plt.plot(range(1, len(y) + 1), y, label=group)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_groupwise_val_ssim(history, title="Validation SSIM by Group"):
    group_names = set()
    for epoch_data in history["val_ssim_group"].values():
        group_names.update(epoch_data.keys())

    group_names = sorted(group_names)
    epochs = sorted(history["val_ssim_group"].keys())

    for group in group_names:
        y = [history["val_ssim_group"][epoch].get(group, float('nan')) for epoch in epochs]
        plt.plot(range(1, len(y) + 1), y, label=group)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def scan_test_files(test_dir: Path) -> List[Path]:
    """Scan test directory for .npy files (test samples)."""
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory {test_dir} not found.")
    test_files = sorted(test_dir.glob("*.npy"))
    print(f"Found {len(test_files)} test samples.")
    return test_files


class TestDataset(Dataset):
    """
    Loads a list of .npy test files and applies the full 14-channel preprocessing
    (FFT Real/Imag + Positional Encoding) and global normalization.
    """
    def __init__(self, file_paths: List[Path]):
        self.file_paths = file_paths
        # Pre-generate the positional encoding once
        self.pos_encoding = generate_positional_encoding(
            cfg.VELOCITY_MAP_HEIGHT,
            cfg.VELOCITY_MAP_WIDTH,
            cfg.POS_ENC_CHANNELS
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = np.load(path)
        tensor = torch.from_numpy(data).float()

        # 1. Perform FFT and separate into Real/Imaginary parts
        ffted = torch.fft.fft(tensor, dim=1)
        real_part = ffted.real[:, :cfg.SEISMIC_NUM_RECEIVERS, :]
        imag_part = ffted.imag[:, :cfg.SEISMIC_NUM_RECEIVERS, :]
        seismic_fft = torch.cat([real_part, imag_part], dim=0)

        # 2. Concatenate with positional encoding
        seismic_processed = torch.cat([seismic_fft, self.pos_encoding], dim=0)

        # 3. Normalize with the pre-calculated GLOBAL stats
        norm_seismic = (seismic_processed - cfg.SEISMIC_MEAN) / cfg.SEISMIC_STD
        
        return norm_seismic, path.stem


def run_test_inference(model, test_files: List[Path], batch_size: int = 8):
    """Batched inference over TestDataset, updated for multi-scale model."""
    ds = TestDataset(test_files)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    model.eval()
    all_preds = []
    all_oids  = []
    with torch.no_grad():
        for batch, stems in tqdm(loader, desc="Test inference"):
            batch = batch.to(cfg.DEVICE)
            
            # out is a list of predictions
            out = model(batch)
            
            ### --- CORRECTED CODE --- ###
            # Select the final, full-resolution prediction for submission
            final_out = out[-1]
            
            # Denormalize the final prediction
            den = denormalize_velocity_global(final_out).squeeze(1).cpu().numpy()
            ### --- END CORRECTION --- ###
            
            all_preds.append(den)
            all_oids.extend(stems)

    preds = np.concatenate(all_preds, axis=0)
    return preds, all_oids


# def make_submission(preds: np.ndarray, oids: List[str], filename="submission.csv"):
#     """
#     Build submission DataFrame: for each oid, for each y in [0..69],
#     output only odd x‐columns (1,3,5,…).
#     """
#     rows = []
#     x_cols = list(range(1, cfg.VELOCITY_MAP_WIDTH, 2))  # [1,3,...,69]
#     for oid, pred in zip(oids, preds):
#         for y in range(pred.shape[0]):
#             row_id = f"{oid}y{y}"
#             vals   = pred[y, x_cols].tolist()
#             rows.append([row_id] + vals)

#     cols = ["oid_ypos"] + [f"x{i}" for i in x_cols]
#     df   = pd.DataFrame(rows, columns=cols)
#     df.to_csv(filename, index=False)
#     print(f"Saved submission to {filename}")


def make_submission(preds: np.ndarray, oids: List[str], filename="submission.csv"):
    """
    Builds the submission DataFrame with the corrected header and row ID format.
    """
    rows = []
    x_cols = list(range(1, cfg.VELOCITY_MAP_WIDTH, 2))

    for oid, pred in zip(oids, preds):
        for y in range(pred.shape[0]):
            
            ### --- FIX 1: Correct the format of the values --- ###
            row_id = f"{oid}_y_{y}"

            vals   = pred[y, x_cols].tolist()
            rows.append([row_id] + vals)

    ### --- FIX 2: Correct the name of the header column --- ###
    cols = ["oid_ypos"] + [f"x{i}" for i in x_cols]
    
    df   = pd.DataFrame(rows, columns=cols)
    df.to_csv(filename, index=False)
    print(f"Saved submission to {filename}")


def main():
    print(f"\n === Starting Group-Aware Waveform Inversion Training === \n")

    # Load file pairs
    file_pairs = scan_files(cfg.TRAIN_PATH)

    # Stratified split
    train_pairs, val_pairs = create_stratified_split(
        file_pairs, val_frac=cfg.VALIDATION_SPLIT
    )

    # --- Compute Normalization Statistics ---
    print(f"\n === Computing normalization statistics === \n")
    # 1. Group-wise stats for training/validation
    print("Computing group-wise stats...")
    cfg.GROUP_STATS = compute_stratified_stats(train_pairs)

    # 2. Global stats for test set inference
    print("\nComputing global stats for test set...")
    train_file_tuples = [(p['input'], p['target']) for p in train_pairs]
    stats_ds = StatsDataset(train_file_tuples, log_transform_velocity=cfg.LOG_TRANSFORM_VELOCITY)
    stats_loader = DataLoader(
        stats_ds, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    (
        cfg.SEISMIC_MEAN, cfg.SEISMIC_STD,
        cfg.VELOCITY_MEAN, cfg.VELOCITY_STD
    ) = compute_stats_gpu(stats_loader, sample_fraction=1.0)
    print(f"GLOBAL SEISMIC: μ={cfg.SEISMIC_MEAN:.4f}, σ={cfg.SEISMIC_STD:.4f}")
    print(f"GLOBAL VELOCITY: μ={cfg.VELOCITY_MEAN:.4f}, σ={cfg.VELOCITY_STD:.4f}")

    # Create datasets using group-aware stats
    train_dataset = WaveformInversionDataset(
        train_pairs, group_stats=cfg.GROUP_STATS,
        augment=True, cache_in_memory=False, log_transform_velocity=cfg.LOG_TRANSFORM_VELOCITY
    )
    val_dataset = WaveformInversionDataset(
        val_pairs, group_stats=cfg.GROUP_STATS,
        light_augment=True, cache_in_memory=False, log_transform_velocity=cfg.LOG_TRANSFORM_VELOCITY
    )

    # Dataloaders
    g = torch.Generator()
    g.manual_seed(cfg.RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS, prefetch_factor=cfg.PREFETCH_FACTOR,
        worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
        persistent_workers=cfg.PERSISTENT_WORKERS, prefetch_factor=cfg.PREFETCH_FACTOR
    )

    # Optional: visualize some preprocessed seismic
    print(f"\n === Visualizing some preprocessed seismic waveform samples === \n")
    visualize_seismic_input(val_loader, num_samples=2)

    print(f"\n === Preparing to train the model === \n")

    # Initialize model & optimizer
    model = UNet(
        n_channels=cfg.UNET_INPUT_CHANNELS,
        n_classes=cfg.UNET_OUTPUT_CHANNELS,
        bilinear=cfg.BILINEAR,
        base_channels=cfg.BASE_CHANNELS
    ).to(cfg.DEVICE)

    summary(
        model,
        input_size=(cfg.BATCH_SIZE, cfg.UNET_INPUT_CHANNELS, 70, 70),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,
        verbose=1
    )

    # optimizer = optim.Lookahead(torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)


    # Train the model
    history = train_model(model, train_loader, val_loader, optimizer, scheduler, cfg.NUM_EPOCHS)

    # --- Post-Training Analysis ---
    print(f"\nTraining complete. Loading best model for evaluation.\n")
    model.load_state_dict(torch.load("best_model.pth"))
    torch.save(model.state_dict(), "final_model.pth") # Save a final copy
    print(f"Best model loaded and saved to 'final_model.pth'.\n")

    # Visualizations
    print(f"\n === Preparing visualizations === \n")
    plot_model_metrics(history, save_path='training_curves.png')
    metrics = plot_scatter_and_residuals(model, val_loader, save_path='residuals.png')
    plot_groupwise_val_mae(history)
    plot_groupwise_val_ssim(history)
    visualize_predictions(model, val_loader, cfg.GROUP_STATS, num_samples=3)

    # --- Inference ---
    print(f"\n === Starting Test Set Inference === \n")
    test_files = scan_test_files(cfg.TEST_PATH)
    preds, oids = run_test_inference(model, test_files, batch_size=cfg.BATCH_SIZE)

    # Make submission file
    make_submission(preds, oids, filename="submission.csv")

    return model, history, metrics


if __name__ == "__main__":
    # To prevent issues in environments like Jupyter
    try:
        from numba import cuda
        cuda.select_device(0)
        cuda.close()
    except:
        pass

    main()

