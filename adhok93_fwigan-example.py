# -*- coding: utf-8 -*-
"""
Script for training a FWIGAN (WGAN-GP) model for Full Waveform Inversion,
using data sourced solely from Kaggle input directories, with data augmentation.
Includes placeholder normalization - **USER MUST UPDATE NORMALIZATION VALUES**.
"""

!pip install webdataset -q # Install quietly

# %% Imports
import csv
import gc
import glob
import os
import random
import shutil
import sys
from pathlib import Path
import time # For timing epochs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.amp # For Automatic Mixed Precision
import torch.autograd as autograd # For gradient penalty
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF # For Augmentation
import webdataset as wds
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# %% Configuration
class cfg:
    """Configuration parameters for the workflow."""

    # --- Paths ---
    kaggle_train_dir = "/kaggle/input/waveform-inversion/train_samples"
    kaggle_test_dir = "/kaggle/input/waveform-inversion/test"
    shard_output_dir = "/kaggle/working/sharded_data"
    working_dir = "/kaggle/working/"
    submission_file = os.path.join(working_dir, "submission.csv")
    model_save_dir = os.path.join(working_dir, "models") # Directory to save models

    # --- Dataset Params ---
    dataset_name = "fwi_kaggle_only_augmented"
    # !!! IMPORTANT: Define your data normalization parameters !!!
    # !!! Determine these from your *entire* training dataset !!!
    VEL_MIN = 1400.0 # Placeholder min velocity
    VEL_MAX = 4600.0 # Placeholder max velocity
    SEIS_NORM_MODE = 'minmax_sample' # 'minmax_sample', 'std_sample', or None

    # --- Sharding Params ---
    maxsize = 1e9  # Approx 1 GB per shard
    force_shard_creation = False

    # --- Splitting & Loading Params ---
    num_used_shards = None  # Use all available
    test_size = 0.1  # Proportion for validation split
    batch_size = 8 # GANs are memory intensive, start lower
    num_workers = 2

    # --- Augmentation Params ---
    apply_augmentation = True
    aug_hflip_prob = 0.5
    # Noise added *before* normalization in this setup
    aug_seis_noise_std = 0.01 # Std dev relative to original seismic range

    # --- Model params (Generator - U-Net like) ---
    unet_in_channels = 5
    unet_out_channels = 1
    unet_init_features = 32 # Initial features for Generator
    unet_depth = 5
    unet_bilinear = True

    # --- Model params (Discriminator) ---
    # Input channels = vel_map (1) + processed_seismic (5)
    disc_in_channels = unet_out_channels + unet_in_channels
    disc_init_features = 64 # Initial features for Discriminator

    # --- Training params (WGAN-GP) ---
    n_epochs = 150 # GANs often need more epochs
    lr_g = 1e-4 # Learning rate for Generator
    lr_d = 1e-4 # Learning rate for Discriminator
    b1 = 0.5    # Adam beta1 (Common GAN value)
    b2 = 0.999  # Adam beta2
    lambda_gp = 10 # Gradient penalty coefficient
    lambda_l1 = 100 # L1 content loss coefficient
    n_critic = 5   # Train Discriminator n_critic times per Generator update

    plot_every_n_epochs = 10 # Plot history less frequently
    save_every_n_epochs = 10 # Save models periodically

    # --- Misc ---
    seed = 42
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Use float16 on CUDA, bfloat16 on CPU (if available) for AMP
    autocast_dtype = torch.float16 if use_cuda else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

# %% Helper Functions
def set_seed(seed=cfg.seed):
    """Sets seed for reproducibility across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # You might disable deterministic for performance in GANs if needed
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def find_best_model(model_dir=cfg.model_save_dir, model_prefix="generator_epoch", suffix=".pth"):
    """Finds the model file with the highest epoch number."""
    best_epoch = -1
    best_model_path = None
    pattern = os.path.join(model_dir, f"{model_prefix}_*_loss_*{suffix}") # Keep loss for potential future use
    all_model_files = glob.glob(os.path.join(model_dir, f"{model_prefix}_*{suffix}"))

    if not all_model_files:
        print(f"W: No models matching pattern '{model_prefix}_*{suffix}' found in {model_dir}.")
        return None

    parsed_models = []
    for f in all_model_files:
        try:
            epoch_str = f.split(model_prefix + "_")[-1].split("_")[0]
            epoch = int(epoch_str)
            parsed_models.append((epoch, f))
        except (ValueError, IndexError, AttributeError):
            print(f"W: Couldn't parse epoch from filename: {os.path.basename(f)}")

    if parsed_models:
        parsed_models.sort(key=lambda x: x[0], reverse=True) # Sort descending by epoch
        best_epoch, best_model_path = parsed_models[0]
        print(f"Found latest model by epoch: {os.path.basename(best_model_path)} (Epoch: {best_epoch})")
    else:
        # Fallback: Select most recently modified if parsing failed
        print("W: No epochs parsed. Selecting most recently modified model.")
        best_model_path = max(all_model_files, key=os.path.getmtime, default=None)
        if best_model_path:
             print(f"Using most recent modification time: {os.path.basename(best_model_path)}")

    return best_model_path

# %% Data Normalization Functions (PLACEHOLDERS - UPDATE VALUES)
def normalize_vel(vel_tensor):
    """Normalizes velocity tensor to [-1, 1] using global min/max."""
    # !!! UPDATE cfg.VEL_MIN and cfg.VEL_MAX with your dataset's actual values !!!
    if cfg.VEL_MAX == cfg.VEL_MIN:
        print("W: VEL_MAX equals VEL_MIN, cannot normalize velocity.")
        return vel_tensor
    # Formula: 2 * (x - min) / (max - min) - 1
    return 2.0 * (vel_tensor - cfg.VEL_MIN) / (cfg.VEL_MAX - cfg.VEL_MIN) - 1.0

def unnormalize_vel(norm_vel_tensor):
    """Un-normalizes velocity tensor from [-1, 1] to original range."""
    # !!! Uses cfg.VEL_MIN and cfg.VEL_MAX !!!
    if cfg.VEL_MAX == cfg.VEL_MIN:
        print("W: VEL_MAX equals VEL_MIN, cannot unnormalize velocity.")
        return norm_vel_tensor
    # Formula: ((y + 1) * (max - min) / 2) + min
    return ((norm_vel_tensor + 1.0) * (cfg.VEL_MAX - cfg.VEL_MIN) / 2.0) + cfg.VEL_MIN

def normalize_seis(seis_tensor):
    """Normalizes seismic tensor based on cfg.SEIS_NORM_MODE."""
    if cfg.SEIS_NORM_MODE == 'minmax_sample':
        # Normalize each sample (image) in the batch independently to [-1, 1]
        min_val = torch.amin(seis_tensor, dim=(-1, -2, -3), keepdim=True)
        max_val = torch.amax(seis_tensor, dim=(-1, -2, -3), keepdim=True)
        range_val = max_val - min_val
        # Add epsilon to avoid division by zero for constant samples
        range_val = torch.where(range_val == 0, torch.tensor(1e-6, device=seis_tensor.device), range_val)
        return 2.0 * (seis_tensor - min_val) / range_val - 1.0
    elif cfg.SEIS_NORM_MODE == 'std_sample':
        # Normalize each sample to mean 0, std 1
        mean_val = torch.mean(seis_tensor, dim=(-1, -2, -3), keepdim=True)
        std_val = torch.std(seis_tensor, dim=(-1, -2, -3), keepdim=True)
        # Add epsilon to avoid division by zero
        std_val = torch.where(std_val == 0, torch.tensor(1e-6, device=seis_tensor.device), std_val)
        return (seis_tensor - mean_val) / std_val
    elif cfg.SEIS_NORM_MODE is None:
        return seis_tensor # No normalization
    else:
        raise ValueError(f"Unknown SEIS_NORM_MODE: {cfg.SEIS_NORM_MODE}")

# %% WebDataset Preprocessing Functions
def search_data_path(target_dirs, root_dir, shuffle=True, seed=cfg.seed):
    """Finds input/output .npy file pairs within subdirectories."""
    # (Keep this function as is from your original code)
    files = []
    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"W: Root directory not found: {root_path}")
        return []

    print(f"Searching for data families {target_dirs} in root: {root_path}")
    total_pairs_found = 0
    for target_dir in target_dirs:
        data_dir = root_path / target_dir
        if not data_dir.is_dir():
            continue

        in_files, out_files = [], []
        data_subdir = data_dir / "data"
        model_subdir = data_dir / "model"

        if data_subdir.is_dir() and model_subdir.is_dir():
            in_files = sorted(data_subdir.glob("*.npy"))
            out_files = sorted(model_subdir.glob("*.npy"))
        else:
            in_files = sorted(data_dir.glob("seis*.npy"))
            out_files = sorted(data_dir.glob("vel*.npy"))

        if not in_files or len(in_files) != len(out_files):
            if in_files or out_files:
                print(
                    f"W: Mismatch or missing files in {data_dir} (in:{len(in_files)}, out:{len(out_files)}). Skipping."
                )
            continue

        current_pairs = list(zip(in_files, out_files))
        files.extend(current_pairs)
        total_pairs_found += len(current_pairs)

    print(f"Found {len(files)} total valid pairs across specified families.")
    if shuffle and files:
        print(f"Shuffling {len(files)} pairs (seed={seed}).")
        rng = np.random.default_rng(seed)
        rng.shuffle(files)

    return files


def generate_sample(in_file, out_file=None, base_dir=None):
    """
    Loads data from .npy files, prepares dicts for WebDataset, converts to float16.
    Handles errors during loading gracefully.
    """
    # (Keep this function as is from your original code)
    # Note: It loads data but doesn't normalize here. Normalization happens later.
    data = []
    seis = None
    vel = None
    try:
        if out_file is None:
            print("W: generate_sample called without out_file (test mode?), not implemented.")
            return []
        else:
            try:
                seis = np.load(in_file, mmap_mode="r")
            except Exception as e:
                print(f"E: Load fail for input {in_file.name}: {e}")
                return []

            try:
                vel = np.load(out_file, mmap_mode="r")
            except Exception as e:
                print(f"E: Load fail for output {out_file.name}: {e}")
                if seis is not None: del seis
                return []

            n_samples = 0
            if seis.ndim == 4 and vel.ndim == 4:
                if seis.shape[0] != vel.shape[0]:
                    print(f"W: Batch size mismatch in {in_file.name} ({seis.shape[0]}) vs {out_file.name} ({vel.shape[0]})")
                    del seis, vel
                    return []
                n_samples = seis.shape[0]
            elif seis.ndim == 3 and vel.ndim == 3:
                n_samples = 1
            else:
                 raise ValueError(f"Unexpected dims: seis {seis.shape}, vel {vel.shape} in {in_file.name}")

            if n_samples == 0:
                print(f"W: Found 0 samples in pair: {in_file.name}, {out_file.name}")
                del seis, vel
                return []

            common_part = f"{in_file.parent.name}_{in_file.stem}"
            if base_dir:
                try:
                    relative_path = in_file.relative_to(base_dir)
                    common_part = "_".join(relative_path.parts).replace(".npy", "")
                    common_part = common_part.replace(os.sep, "_").replace("\\", "_")
                except ValueError:
                    pass

            for i in range(n_samples):
                key = f"{common_part}_{i}"
                s_sample = (seis[i].copy().astype(np.float16) if seis.ndim == 4 else seis.copy().astype(np.float16))
                v_sample = (vel[i].copy().astype(np.float16) if vel.ndim == 4 else vel.copy().astype(np.float16))
                data.append({"__key__": key, "sample_id.txt": key, "seis.npy": s_sample, "vel.npy": v_sample})

            del seis
            del vel

    except Exception as e:
        print(f"E: Error during sample generation for {in_file.name}: {e}")
        if seis is not None:
            try: del seis
            except NameError: pass
        if vel is not None:
            try: del vel
            except NameError: pass
        return []

    return data


# %% WebDataset Loading Functions
def get_shard_paths(root_dir, dataset_name, stage, num_shards=None, test_size=cfg.test_size, seed=cfg.seed):
    """Gets list of shard paths, optionally selects subset, optionally splits train/val."""
    # (Keep this function as is from your original code)
    source_dir_name = f"train_{dataset_name}"
    dataset_dir = Path(root_dir) / source_dir_name
    print(f"Looking for shards for stage '{stage}' in: {dataset_dir}")

    if not dataset_dir.is_dir():
        print(f"W: Shard directory not found: {dataset_dir}")
        return (None, None) if stage == "train" else None

    shard_paths = sorted([str(p) for p in dataset_dir.glob("*.tar")])

    if not shard_paths:
        print(f"W: No .tar shards found in {dataset_dir}.")
        return (None, None) if stage == "train" else None

    print(f"Found {len(shard_paths)} total shards.")
    selected_paths = shard_paths
    available_count = len(shard_paths)
    if num_shards is not None:
        if 0 < num_shards < available_count:
            print(f"Selecting {num_shards} shards randomly (seed={seed}).")
            rng = np.random.default_rng(seed)
            indices = rng.choice(available_count, size=num_shards, replace=False)
            selected_paths = sorted([shard_paths[i] for i in indices])
        elif num_shards >= available_count:
            print(f"Requested {num_shards} or more shards, using all {available_count} available.")
        else:
            print(f"W: Invalid num_shards ({num_shards}). Using all {available_count} shards.")
    print(f"Using {len(selected_paths)} selected shards for stage '{stage}'.")

    if stage == "train":
        count = len(selected_paths)
        print(f"Splitting {count} selected shards (test_size={test_size}, seed={seed})")
        try:
            if not (0 <= test_size < 1): raise ValueError("test_size must be in [0, 1)")
            if count <= 1 or test_size == 0:
                reason = "only 1 shard" if count <= 1 else "test_size is 0"
                print(f"W: Cannot split for validation ({reason}). Assigning all to train.")
                return sorted(selected_paths), []
            else:
                trn_paths, val_paths = train_test_split(selected_paths, test_size=test_size, random_state=seed, shuffle=True)
                trn_paths.sort()
                val_paths.sort()
                print(f"# Train shards: {len(trn_paths)}, # Val shards: {len(val_paths)}")
                return trn_paths, val_paths
        except Exception as e:
            print(f"E: Failed to split shards: {e}")
            return None, None
    else:
        print(f"# Shards returned for stage '{stage}': {len(selected_paths)}")
        return sorted(selected_paths)


def get_dataset(paths, stage, seed=cfg.seed):
    """Creates WebDataset object. Applies augmentations and NORMALIZATION."""
    if not paths:
        print(f"W: No shard paths provided for stage '{stage}'. Cannot create dataset.")
        return None

    print(f"Creating WebDataset for stage '{stage}' from {len(paths)} shards.")
    is_train = stage == "train"
    map_handler = wds.warn_and_continue

    try:
        dataset = wds.WebDataset(
            paths, nodesplitter=wds.split_by_node, shardshuffle=is_train, seed=seed
        )
        dataset = dataset.decode(handler=map_handler) # Decode standard types

        def map_train_val(sample):
            """Inner function to process, augment, and NORMALIZE samples."""
            key_info = sample.get("__key__", "N/A")
            try:
                required = ["sample_id.txt", "seis.npy", "vel.npy"]
                if not all(k in sample for k in required):
                    raise KeyError(f"Missing required keys in sample {key_info}")

                sid = sample["sample_id.txt"]
                # Convert to float32 tensors first
                s_np = np.asarray(sample["seis.npy"]).astype(np.float32)
                v_np = np.asarray(sample["vel.npy"]).astype(np.float32)
                seis_tensor = torch.from_numpy(s_np)
                vel_tensor = torch.from_numpy(v_np)

                # --- Augmentation Block (Applied BEFORE Normalization potentially) ---
                if is_train and cfg.apply_augmentation:
                    # 1. Add Gaussian Noise to Seismic Data (relative to original range)
                    if cfg.aug_seis_noise_std > 0:
                        noise = torch.randn_like(seis_tensor) * cfg.aug_seis_noise_std
                        seis_tensor.add_(noise) # In-place addition

                    # 2. Horizontal Flip
                    if torch.rand(1).item() < cfg.aug_hflip_prob:
                        seis_tensor = TF.hflip(seis_tensor)
                        vel_tensor = TF.hflip(vel_tensor)

                # --- NORMALIZATION ---
                # !!! Ensure your normalization functions handle the tensor shapes correctly !!!
                seis_tensor_norm = normalize_seis(seis_tensor)
                vel_tensor_norm = normalize_vel(vel_tensor)

                return {"sample_id": sid, "seis": seis_tensor_norm, "vel": vel_tensor_norm}

            except Exception as map_e:
                print(f"E: Map function failed for sample {key_info}: {map_e}")
                raise map_e # Let handler decide fate

        if stage in ["train", "val"]:
            dataset = dataset.map(map_train_val, handler=map_handler)

        if is_train:
            dataset = dataset.shuffle(1000) # Buffer size

        return dataset

    except Exception as e:
        print(f"E: Error creating WebDataset pipeline for stage '{stage}': {e}")
        return None


# %% Kaggle TestSet Loading (Directly from .npy)
class KaggleTestDataset(Dataset):
    """Loads Kaggle test set .npy files and applies SEISMIC normalization."""
    def __init__(self, test_files_dir):
        self.test_files_dir = Path(test_files_dir)
        self.test_files = []
        try:
            if not self.test_files_dir.is_dir():
                raise FileNotFoundError(f"Kaggle test directory missing: {self.test_files_dir}")
            self.test_files = sorted(list(self.test_files_dir.glob("*.npy")))
            print(f"Found {len(self.test_files)} '.npy' files in Kaggle test dir: {self.test_files_dir}")
            if not self.test_files: print(f"W: No .npy files found in {self.test_files_dir}.")
        except Exception as e:
            print(f"E: Error accessing Kaggle test directory {self.test_files_dir}: {e}")

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):
        if not self.test_files or index >= len(self.test_files):
            raise IndexError(f"Index {index} out of bounds ({len(self.test_files)} files).")
        test_file_path = self.test_files[index]
        try:
            # Load numpy array, convert to float32 tensor
            data_np = np.load(test_file_path).astype(np.float32)
            data_tensor = torch.from_numpy(data_np)
            # Apply SEISMIC normalization consistent with training
            data_tensor_norm = normalize_seis(data_tensor)
            original_id = test_file_path.stem
            return data_tensor_norm, original_id
        except Exception as e:
            raise IOError(f"Error loading/normalizing Kaggle test file: {test_file_path}") from e


# %% U-Net Blocks (DoubleConv, Down, Up, OutConv - Keep as is)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__(); self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_channels + out_channels, out_channels, mid_channels=out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2); diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1); return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__(); self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)


# %% Generator Definition (Adapted U-Net)
class GeneratorUNet(nn.Module):
    """Generator using U-Net architecture, outputs normalized velocity map."""
    def __init__( self, n_channels=cfg.unet_in_channels, n_classes=cfg.unet_out_channels,
                 init_features=cfg.unet_init_features, depth=cfg.unet_depth, bilinear=cfg.unet_bilinear):
        super().__init__()
        self.n_channels = n_channels; self.n_classes = n_classes
        self.bilinear = bilinear; self.depth = depth
        self.initial_pool = nn.AvgPool2d(kernel_size=(14, 1), stride=(14, 1))
        self.encoder_blocks = nn.ModuleList(); self.inc = DoubleConv(n_channels, init_features)
        self.encoder_blocks.append(self.inc); current_features = init_features
        for _ in range(depth):
            down_block = Down(current_features, current_features * 2)
            self.encoder_blocks.append(down_block); current_features *= 2
        bottleneck_features = current_features
        self.decoder_blocks = nn.ModuleList(); current_features = bottleneck_features
        for _ in range(depth):
            up_block = Up(current_features, current_features // 2, bilinear)
            self.decoder_blocks.append(up_block); current_features //= 2
        self.outc = OutConv(current_features, n_classes)
        self.final_activation = nn.Tanh() # Output in [-1, 1] matching normalization
        self.processed_seismic = None # To store condition for Discriminator

    def _pad_or_crop(self, x, target_h=70, target_w=70):
        _, _, h, w = x.shape
        if h < target_h: pad_top = (target_h - h) // 2; pad_bottom = target_h - h - pad_top; x = F.pad(x, (0, 0, pad_top, pad_bottom)); h = target_h
        if w < target_w: pad_left = (target_w - w) // 2; pad_right = target_w - w - pad_left; x = F.pad(x, (pad_left, pad_right, 0, 0)); w = target_w
        if h > target_h: crop_top = (h - target_h) // 2; x = x[:, :, crop_top : crop_top + target_h, :]; h = target_h
        if w > target_w: crop_left = (w - target_w) // 2; x = x[:, :, :, crop_left : crop_left + target_w]; w = target_w
        return x

    def forward(self, x_seismic):
        x_pooled = self.initial_pool(x_seismic)
        x_resized = self._pad_or_crop(x_pooled, target_h=70, target_w=70)
        self.processed_seismic = x_resized # Store for D conditioning
        skip_connections = []; xi = x_resized
        for i, block in enumerate(self.encoder_blocks):
            xi = block(xi)
            if i < len(self.encoder_blocks) - 1: skip_connections.append(xi)
        xu = xi
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections[len(skip_connections) - 1 - i]; xu = block(xu, skip)
        logits = self.outc(xu); output = self.final_activation(logits)
        return output

# %% Discriminator Definition (Example CNN)

# %% Discriminator Definition (Corrected Final Layer)
class Discriminator(nn.Module):
    """Discriminator network for FWIGAN (Conditional)."""
    def __init__(self, in_channels=cfg.disc_in_channels, init_features=cfg.disc_init_features):
        super().__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [ nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1),
                      nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                 block.append(nn.BatchNorm2d(out_filters)) # Consider InstanceNorm/LayerNorm or removal if issues persist
            return block

        nf = init_features
        self.model = nn.Sequential(
            *discriminator_block(in_channels, nf, bn=False), # (B, 6, 70, 70) -> (B, 64, 35, 35)
            *discriminator_block(nf, nf * 2),                # -> (B, 128, 17, 17) ~ Corrected Size
            *discriminator_block(nf * 2, nf * 4),            # -> (B, 256, 8, 8)  ~ Corrected Size
            *discriminator_block(nf * 4, nf * 8),            # -> (B, 512, 4, 4)  ~ Corrected Size
            # *** FIXED KERNEL SIZE HERE ***
            # Final layer outputs a single score (no sigmoid for WGAN)
            nn.Conv2d(nf * 8, 1, kernel_size=4, stride=1, padding=0) # Output: (B, 1, 1, 1)
        )

    def forward(self, velocity_map, seismic_condition):
        # Concatenate along channel dimension
        img_input = torch.cat((velocity_map, seismic_condition), dim=1)
        return self.model(img_input) # Output is raw score





# %% Weight Initialization Function
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try: torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except: pass # handles bias items etc.
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# %% WGAN-GP Gradient Penalty Function
def compute_gradient_penalty(D, real_samples_vel, fake_samples_vel, seismic_cond, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    batch_size = real_samples_vel.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device) # Shape (B, 1, 1, 1)
    interpolates_vel = (alpha * real_samples_vel.data + ((1 - alpha) * fake_samples_vel.data)).requires_grad_(True)
    # Use the same seismic condition for interpolated samples
    d_interpolates = D(interpolates_vel, seismic_cond.data) # Pass condition

    # Use torch.ones_like for fake gradients to match output shape
    fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)

    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates_vel, grad_outputs=fake,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# %% Main Execution Script
print("--- Starting Full FWIGAN Workflow ---")
start_time = time.time()
set_seed(cfg.seed)
print(f"Device: {cfg.device}")
print(f"Using PyTorch version: {torch.__version__}")
if cfg.use_cuda: print(f"CUDA available: {torch.cuda.get_device_name(0)}")
print(f"AMP dtype: {cfg.autocast_dtype}")
print(f"Velocity Norm Range: [{cfg.VEL_MIN}, {cfg.VEL_MAX}] -> [-1, 1]")
print(f"Seismic Norm Mode: {cfg.SEIS_NORM_MODE}")


# ==============================================================================
# Cleanup Code
# ==============================================================================
print("\n--- Cleaning up previous run artifacts ---")
Path(cfg.model_save_dir).mkdir(parents=True, exist_ok=True) # Ensure model dir exists
paths_to_clean = [cfg.shard_output_dir]
# Clean previous models (adjust pattern if needed)
model_patterns = [
    os.path.join(cfg.model_save_dir, "generator_epoch_*.pth"),
    os.path.join(cfg.model_save_dir, "discriminator_epoch_*.pth")
]
for pattern in model_patterns: paths_to_clean.extend(glob.glob(pattern))
paths_to_clean.append(os.path.join(cfg.working_dir, "training_history.png"))
paths_to_clean.append(cfg.submission_file)

for path_str in paths_to_clean:
    path_obj = Path(path_str)
    try:
        if path_obj.is_dir(): shutil.rmtree(path_obj, ignore_errors=True)
        elif path_obj.is_file(): path_obj.unlink(missing_ok=True)
    except Exception as e: print(f"W: Error during cleanup of {path_obj}: {e}")
print("--- Cleanup finished ---")
gc.collect()

# ==============================================================================
# SECTION 0/1: Sharding from Kaggle Data Only
# ==============================================================================
print("\n--- 0/1. Sharding from Kaggle Data Only ---")
# (Keep this section largely as is from your original code)
shard_stage_dir = Path(cfg.shard_output_dir) / f"train_{cfg.dataset_name}"
kaggle_train_root = Path(cfg.kaggle_train_dir)
needs_creation = True; total_samples_written = 0
try:
    if shard_stage_dir.exists() and any(shard_stage_dir.glob("*.tar")):
        if cfg.force_shard_creation:
            print(f"Forcing shard creation. Removing existing shards in {shard_stage_dir}")
            shutil.rmtree(shard_stage_dir)
        else:
            print(f"Found existing shards at: {shard_stage_dir}. Skipping creation.")
            needs_creation = False
    Path(cfg.shard_output_dir).mkdir(parents=True, exist_ok=True)
    shard_stage_dir.mkdir(parents=True, exist_ok=True)

    if needs_creation:
        print(f"Starting shard creation from {kaggle_train_root} into {shard_stage_dir}")
        if not kaggle_train_root.is_dir(): raise FileNotFoundError(f"Kaggle train dir not found: {kaggle_train_root}")
        families = [d.name for d in kaggle_train_root.iterdir() if d.is_dir()]
        if not families: raise FileNotFoundError(f"No family subdirs found in {kaggle_train_root}")
        print(f"Searching Kaggle data families: {families}")
        kaggle_file_pairs = search_data_path(families, kaggle_train_root, shuffle=True, seed=cfg.seed)
        if not kaggle_file_pairs: raise RuntimeError("No valid data pairs found.")

        shard_pattern = str(shard_stage_dir / "%06d.tar")
        print(f"Writing shards using pattern {shard_pattern} (max size {cfg.maxsize / 1e9:.2f} GB)")
        with wds.ShardWriter(shard_pattern, maxsize=int(cfg.maxsize)) as writer:
            common_base_dir = kaggle_train_root
            for in_file, out_file in tqdm(kaggle_file_pairs, desc="Sharding Kaggle Data", unit="pair"):
                samples_from_pair = generate_sample(Path(in_file), Path(out_file), base_dir=common_base_dir)
                if samples_from_pair:
                    for sample_dict in samples_from_pair: writer.write(sample_dict)
                    total_samples_written += len(samples_from_pair)
        print(f"Finished writing {total_samples_written} samples to shards.")
    else: print(f"Using existing shards in {shard_stage_dir}.")

except Exception as e: print(f"E: Sharding process failed: {e}"); raise

# ==============================================================================
# SECTION 2: Create DataLoaders (Using modified get_dataset with Normalization)
# ==============================================================================
print("\n--- 2. Creating DataLoaders from Shards (with Normalization) ---")
dltrain, dlvalid = None, None; val_paths_saved = []
try:
    trn_paths, val_paths = get_shard_paths(cfg.shard_output_dir, cfg.dataset_name, "train")
    val_paths_saved = val_paths
    if trn_paths is None: raise RuntimeError("Failed to get or split shard paths.")

    # Check if shards exist if paths are empty
    shard_check_dir = Path(cfg.shard_output_dir) / f"train_{cfg.dataset_name}"
    if not trn_paths and not list(shard_check_dir.glob("*.tar")):
         raise RuntimeError(f"No training shards found in {shard_check_dir}.")
    if not trn_paths: print("W: No shards for training.")
    else: print(f"Using {len(trn_paths)} shards for training.")
    if not val_paths: print("W: No shards for validation.")
    else: print(f"Using {len(val_paths)} shards for validation.")

    trn_ds = get_dataset(trn_paths, "train", seed=cfg.seed) if trn_paths else None
    val_ds = get_dataset(val_paths, "val", seed=cfg.seed + 1) if val_paths else None # Use different seed for val

    if trn_ds:
        n_trn_w = min(cfg.num_workers, os.cpu_count(), len(trn_paths)) if trn_paths else 0
        p_trn = n_trn_w > 0
        dltrain = DataLoader(trn_ds.batched(cfg.batch_size), batch_size=None, shuffle=False,
                             num_workers=n_trn_w, pin_memory=cfg.use_cuda, persistent_workers=p_trn,
                             prefetch_factor=2 if p_trn else None)
        print(f"Train DataLoader created (workers={n_trn_w}, persistent={p_trn}).")
    if val_ds:
        n_val_w = min(cfg.num_workers, os.cpu_count(), len(val_paths)) if val_paths else 0
        p_val = n_val_w > 0
        dlvalid = DataLoader(val_ds.batched(cfg.batch_size), batch_size=None, shuffle=False,
                             num_workers=n_val_w, pin_memory=cfg.use_cuda, persistent_workers=p_val,
                             prefetch_factor=2 if p_val else None)
        print(f"Validation DataLoader created (workers={n_val_w}, persistent={p_val}).")

except Exception as e: print(f"E: DataLoader creation failed: {e}"); raise

# ==============================================================================
# SECTION 3: Initialize Models, Optimizers, Losses
# ==============================================================================
print("\n--- 3. Initializing Models, Losses, Optimizers ---")
try:
    generator = GeneratorUNet().to(cfg.device)
    discriminator = Discriminator().to(cfg.device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print("Applied normal weight initialization.")

    # Optimizers (Using Adam as recommended for GANs)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.lr_g, betas=(cfg.b1, cfg.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.b1, cfg.b2))
    print(f"Optimizers: Adam (G: lr={cfg.lr_g}, D: lr={cfg.lr_d}, beta1={cfg.b1})")

    # Loss Functions
    criterion_L1 = nn.L1Loss().to(cfg.device) # Content loss
    print(f"Losses: Adversarial (WGAN-GP), Content (L1, lambda={cfg.lambda_l1})")

    # AMP Grad Scalers

    # NEW Lines (Corrected GradScaler initialization)
# Use 'cuda' if using GPU, 'cpu' otherwise. enabled logic remains the same.
    device_type = 'cuda' if cfg.use_cuda else 'cpu'
    g_scaler = torch.amp.GradScaler(device_type, enabled=(cfg.use_cuda and cfg.autocast_dtype != torch.float32))
    d_scaler = torch.amp.GradScaler(device_type, enabled=(cfg.use_cuda and cfg.autocast_dtype != torch.float32))
    print(f"AMP GradScaler enabled: {g_scaler.is_enabled()}") # Keep check
    # g_scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_cuda and cfg.autocast_dtype != torch.float32))
    # d_scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_cuda and cfg.autocast_dtype != torch.float32))
    # print(f"AMP GradScaler enabled: {g_scaler.is_enabled()}")

    params_g = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    params_d = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator Params: {params_g:,}")
    print(f"Discriminator Params: {params_d:,}")

except Exception as e: print(f"E: Model/Optimizer initialization failed: {e}"); raise

# ==============================================================================
# SECTION 4: Training Loop
# ==============================================================================
print("\n--- 4. Starting FWIGAN Training ---")
history = {"epoch": [], "d_loss": [], "g_loss": [], "l1_loss": [], "val_l1_loss": []}
batches_done = 0

if dltrain is None:
    print("E: Training cannot proceed. Train DataLoader is missing.")
else:
    try:
        for epoch in range(1, cfg.n_epochs + 1):
            epoch_start_time = time.time()
            generator.train()
            discriminator.train()
            epoch_d_losses, epoch_g_losses, epoch_l1_losses = [], [], []

            pbar_train = tqdm(dltrain, desc=f"Train E{epoch}", leave=False, unit="batch")

            for i, batch in enumerate(pbar_train):
                if not batch or "seis" not in batch or "vel" not in batch:
                     print(f"W: Skipping invalid train batch {i}")
                     continue

                real_seis = batch["seis"].to(cfg.device, non_blocking=True).float()
                real_vel = batch["vel"].to(cfg.device, non_blocking=True).float()
                current_batch_size = real_seis.size(0) # Get actual batch size

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad(set_to_none=True)

                # Use AMP context manager for Discriminator forward/loss
                with torch.amp.autocast(device_type=cfg.device.type, dtype=cfg.autocast_dtype, enabled=g_scaler.is_enabled()):
                    # Generate fake velocity map (no grad for G here)
                    with torch.no_grad():
                         fake_vel = generator(real_seis)
                    # Get the processed seismic condition used by the generator
                    processed_seis_cond = generator.processed_seismic.detach()

                    # Calculate D scores
                    real_validity = discriminator(real_vel, processed_seis_cond)
                    fake_validity = discriminator(fake_vel.detach(), processed_seis_cond) # Detach fake_vel

                    # Gradient penalty
                    gradient_penalty = compute_gradient_penalty(
                        discriminator, real_vel, fake_vel, processed_seis_cond, cfg.device
                    )

                    # Adversarial loss (WGAN-GP)
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + cfg.lambda_gp * gradient_penalty

                # Scale loss and backpropagate (Discriminator)
                d_scaler.scale(d_loss).backward()
                d_scaler.step(optimizer_D)
                d_scaler.update()
                epoch_d_losses.append(d_loss.item())

                # -----------------
                #  Train Generator
                # -----------------
                # Train Generator every n_critic discriminator iterations
                if i % cfg.n_critic == 0:
                    optimizer_G.zero_grad(set_to_none=True)

                    # Use AMP context manager for Generator forward/loss
                    with torch.amp.autocast(device_type=cfg.device.type, dtype=cfg.autocast_dtype, enabled=g_scaler.is_enabled()):
                        # Generate fake velocity map (track grads for G now)
                        gen_vel = generator(real_seis)
                        # Get the corresponding processed seismic condition
                        processed_seis_cond_gen = generator.processed_seismic

                        # Get discriminator score for generated map
                        gen_validity = discriminator(gen_vel, processed_seis_cond_gen)

                        # Adversarial loss (aims for high D score)
                        g_adv_loss = -torch.mean(gen_validity)

                        # Content loss (L1 distance between normalized maps)
                        g_l1_loss = criterion_L1(gen_vel, real_vel)

                        # Total generator loss
                        g_loss = g_adv_loss + cfg.lambda_l1 * g_l1_loss

                    # Scale loss and backpropagate (Generator)
                    g_scaler.scale(g_loss).backward()
                    g_scaler.step(optimizer_G)
                    g_scaler.update()
                    epoch_g_losses.append(g_loss.item())
                    epoch_l1_losses.append(g_l1_loss.item())

                    batches_done += 1

                # Update progress bar description
                if i % 100 == 0: # Update less frequently
                    pbar_train.set_postfix(
                        D_Loss=f"{np.mean(epoch_d_losses[-50:]):.3f}",
                        G_Loss=f"{np.mean(epoch_g_losses[-10:]):.3f}",
                        L1=f"{np.mean(epoch_l1_losses[-10:]):.3f}"
                    )

            # --- End of Epoch ---
            avg_d_loss = np.mean(epoch_d_losses) if epoch_d_losses else 0
            avg_g_loss = np.mean(epoch_g_losses) if epoch_g_losses else 0
            avg_l1_loss = np.mean(epoch_l1_losses) if epoch_l1_losses else 0
            epoch_time = time.time() - epoch_start_time

            print(f"Epoch {epoch}/{cfg.n_epochs} [{epoch_time:.2f}s] - D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}, L1_Loss: {avg_l1_loss:.4f}")
            history["epoch"].append(epoch)
            history["d_loss"].append(avg_d_loss)
            history["g_loss"].append(avg_g_loss)
            history["l1_loss"].append(avg_l1_loss)

            # --- Validation Phase (Optional but recommended) ---
            if dlvalid is not None:
                generator.eval() # Generator only for validation
                val_l1 = []
                with torch.no_grad():
                    for batch_val in tqdm(dlvalid, desc=f"Valid E{epoch}", leave=False):
                         if not batch_val or "seis" not in batch_val or "vel" not in batch_val: continue
                         seis_val = batch_val["seis"].to(cfg.device).float()
                         vel_val = batch_val["vel"].to(cfg.device).float()
                         with torch.amp.autocast(device_type=cfg.device.type, dtype=cfg.autocast_dtype, enabled=g_scaler.is_enabled()):
                             pred_vel = generator(seis_val)
                             # Calculate L1 loss on normalized validation data
                             loss_l1 = criterion_L1(pred_vel, vel_val)
                         val_l1.append(loss_l1.item())
                avg_val_l1 = np.mean(val_l1) if val_l1 else 0
                print(f"Epoch {epoch} Avg Validation L1_Loss: {avg_val_l1:.4f}")
                history["val_l1_loss"].append(avg_val_l1)
            else:
                history["val_l1_loss"].append(None) # Append None if no validation

            # --- Save Models Periodically ---
            if epoch % cfg.save_every_n_epochs == 0 or epoch == cfg.n_epochs:
                g_path = os.path.join(cfg.model_save_dir, f"generator_epoch_{epoch}.pth")
                d_path = os.path.join(cfg.model_save_dir, f"discriminator_epoch_{epoch}.pth")
                torch.save(generator.state_dict(), g_path)
                torch.save(discriminator.state_dict(), d_path)
                print(f"Saved models at epoch {epoch} to {cfg.model_save_dir}")

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\nE: Training loop encountered a critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Training Loop Finished ---")


# ==============================================================================
# SECTION 5: Plot History
# ==============================================================================
print("\n--- 5. Plotting Training History ---")
if history["epoch"]:
    try:
        hist_df = pd.DataFrame(history)
        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2, 1) # Loss plot
        plt.plot(hist_df["epoch"], hist_df["d_loss"], "o-", label="Discriminator Loss")
        plt.plot(hist_df["epoch"], hist_df["g_loss"], "s-", label="Generator Loss")
        plt.title("GAN Losses vs. Epoch")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.5)

        plt.subplot(1, 2, 2) # L1 Loss plot
        plt.plot(hist_df["epoch"], hist_df["l1_loss"], "^-", label="Train L1 Loss (Content)")
        if not hist_df["val_l1_loss"].isnull().all():
            plt.plot(hist_df["epoch"], hist_df["val_l1_loss"], "v--", label="Validation L1 Loss")
        plt.title("L1 Content Loss vs. Epoch")
        plt.xlabel("Epoch"); plt.ylabel("L1 Loss"); plt.legend(); plt.grid(True, alpha=0.5)
        plt.ylim(bottom=0) # L1 should be non-negative

        plt.tight_layout()
        plot_fname = os.path.join(cfg.working_dir, "training_history.png")
        plt.savefig(plot_fname)
        print(f"Saved history plot: {plot_fname}")
        plt.show()
    except Exception as e: print(f"E: Failed plotting training history: {e}")
else: print("No training history recorded to plot.")

# ==============================================================================
# SECTION 6: Error Analysis / Validation Visualization (Placeholder)
# ==============================================================================
print("\n--- 6. Validation Set Visualization (Example) ---")
best_generator_path = find_best_model(model_prefix="generator_epoch") # Find latest Generator

if not best_generator_path:
    print("W: No generator model found. Skipping validation visualization.")
elif dlvalid is None:
    print("W: Validation loader unavailable. Skipping visualization.")
else:
    print(f"Visualizing using generator: {os.path.basename(best_generator_path)}")
    try:
        # Load the latest generator model found
        vis_generator = GeneratorUNet().to(cfg.device)
        vis_generator.load_state_dict(torch.load(best_generator_path, map_location=cfg.device))
        vis_generator.eval()

        # Get a sample batch from validation loader
        val_batch = next(iter(dlvalid))
        seis_val = val_batch["seis"].to(cfg.device).float()
        vel_val_norm = val_batch["vel"].to(cfg.device).float() # Ground truth (normalized)

        with torch.no_grad():
            with torch.amp.autocast(device_type=cfg.device.type, dtype=cfg.autocast_dtype, enabled=g_scaler.is_enabled()):
                pred_vel_norm = vis_generator(seis_val) # Predicted (normalized)

        # Un-normalize for visualization
        vel_val_unnorm = unnormalize_vel(vel_val_norm)
        pred_vel_unnorm = unnormalize_vel(pred_vel_norm)

        # Plot the first sample in the batch
        idx_to_plot = 0
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(vel_val_unnorm[idx_to_plot, 0].cpu().numpy(), cmap='viridis', aspect='auto')
        plt.title(f"Ground Truth Velocity (Sample {idx_to_plot})")
        plt.colorbar(label='Velocity')
        plt.subplot(1, 2, 2)
        plt.imshow(pred_vel_unnorm[idx_to_plot, 0].cpu().numpy(), cmap='viridis', aspect='auto')
        plt.title(f"Predicted Velocity (Epoch: {best_generator_path.split('_')[-1].split('.')[0]})")
        plt.colorbar(label='Velocity')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"E: Error during validation visualization: {e}")
        import traceback; traceback.print_exc()

# ==============================================================================
# SECTION 7: Prediction on Kaggle Test Set
# ==============================================================================
print("\n--- 7. Final Prediction on Kaggle Test Set ---")
best_generator_final_path = find_best_model(model_prefix="generator_epoch") # Find latest Generator

if not best_generator_final_path:
    print("W: No generator model found. Skipping final prediction.")
elif not Path(cfg.kaggle_test_dir).is_dir():
    print(f"W: Kaggle test directory '{cfg.kaggle_test_dir}' not found. Skipping prediction.")
else:
    try:
        print(f"Loading generator for prediction: {os.path.basename(best_generator_final_path)}")
        model_pred = GeneratorUNet().to(cfg.device)
        model_pred.load_state_dict(torch.load(best_generator_final_path, map_location=cfg.device))
        model_pred.eval()

        test_ds = KaggleTestDataset(cfg.kaggle_test_dir) # Applies seismic normalization
        if len(test_ds) == 0:
            print("W: Kaggle test dataset is empty. No submission generated.")
        else:
            t_bs = max(1, cfg.batch_size // 2) # Use smaller batch size for inference if needed
            t_nw = min(max(0, cfg.num_workers // 2), os.cpu_count())
            dl_test = DataLoader(test_ds, batch_size=t_bs, shuffle=False, num_workers=t_nw, pin_memory=cfg.use_cuda)
            print(f"Test DataLoader created (bs={t_bs}, workers={t_nw})")
            print(f"Writing submission file to: {cfg.submission_file}")

            rows_written = 0
            with open(cfg.submission_file, "wt", newline="") as csvfile:
                x_cols = [f"x_{i}" for i in range(1, 70, 2)]
                fieldnames = ["oid_ypos"] + x_cols
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                pbar_test = tqdm(dl_test, desc="Generating Submission", unit="batch")
                with torch.no_grad():
                    for inputs_norm, original_ids in pbar_test: # Input is normalized seismic
                        if isinstance(original_ids, str): original_ids = [original_ids]
                        try:
                            inputs_norm = inputs_norm.to(cfg.device).float()
                            with torch.amp.autocast(device_type=cfg.device.type, dtype=cfg.autocast_dtype, enabled=g_scaler.is_enabled()):
                                outputs_norm = model_pred(inputs_norm) # Output is normalized velocity

                            # !!! UN-NORMALIZE the prediction !!!
                            outputs_unnorm = unnormalize_vel(outputs_norm)

                            # Get predictions as numpy array (B, H, W)
                            preds = outputs_unnorm[:, 0].cpu().numpy()

                            for y_pred, oid in zip(preds, original_ids): # y_pred is (H=70, W=70)
                                for y_pos in range(y_pred.shape[0]):
                                    vals = y_pred[y_pos, 1::2].astype(np.float32) # Extract odd columns
                                    row = dict(zip(x_cols, vals))
                                    row["oid_ypos"] = f"{oid}_y_{y_pos}"
                                    writer.writerow(row); rows_written += 1
                        except Exception as e:
                            print(f"\nE: Prediction failed for batch (OID: {original_ids[0] if original_ids else '?'}) : {e}")

            print(f"Submission file created: {cfg.submission_file} ({rows_written} rows).")
            expected_rows = len(test_ds) * 70
            if rows_written != expected_rows:
                print(f"W: Row count mismatch! Expected {expected_rows}, but wrote {rows_written}.")

    except Exception as e:
        print(f"E: Final prediction process failed critically: {e}")
        import traceback; traceback.print_exc()

# ==============================================================================
# End of Workflow
# ==============================================================================
end_time = time.time()
print(f"\n--- Full Workflow Finished in {(end_time - start_time) / 60:.2f} minutes ---")

