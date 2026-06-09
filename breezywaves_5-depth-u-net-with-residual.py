# -*- coding: utf-8 -*-
"""
Script for training a U-Net model (with Residual Blocks) for Full Waveform
Inversion, using data sourced solely from Kaggle input directories, with
data augmentation. Corrected generate_sample function.
"""

# %% Imports
# Standard Library Imports
import csv
import gc
import glob
import os
import random
import shutil
import sys
from pathlib import Path

# Third-party Imports
# Install webdataset if not present (useful in notebook environments)
try:
    import webdataset as wds
except ImportError:
    print("Installing webdataset...")
    !pip install webdataset
    import webdataset as wds

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF  # For Augmentation
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from functools import partial


# %% Configuration
class cfg:
    """Configuration parameters for the workflow."""

    # --- Paths ---
    kaggle_train_dir = "/kaggle/input/waveform-inversion/train_samples"
    kaggle_test_dir = "/kaggle/input/waveform-inversion/test"
    shard_output_dir = "/kaggle/working/sharded_data"
    working_dir = "/kaggle/working/"
    submission_file = os.path.join(working_dir, "submission.csv")

    # --- Dataset Params ---
    dataset_name = "fwi_kaggle_only_augmented_resnet" # Updated name

    # --- Sharding Params ---
    maxsize = 1e9  # Approx 1 GB
    force_shard_creation = False

    # --- Splitting & Loading Params ---
    num_used_shards = None  # Use all available
    test_size = 0.1  # Proportion for validation split
    batch_size = 8  # Reduced from 16 to 8
    num_workers = 1  # Reduced from 2 to 1

    # --- Augmentation Params ---
    apply_augmentation = True
    aug_hflip_prob = 0.5  # Probability of horizontal flip
    aug_seis_noise_std = 0.01  # Std dev of Gaussian noise added to seismic

    # --- Model params (U-Net with Residual Blocks) ---
    unet_in_channels = 5
    unet_out_channels = 1
    unet_init_features = 24  # Reduced from 32 to 24
    unet_depth = 4  # Reduced from 5 to 4
    unet_bilinear = True # Upsampling method

    # --- Training params ---
    n_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    plot_every_n_epochs = 5

    # --- Misc ---
    seed = 42
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    autocast_dtype = torch.float16 if use_cuda else torch.bfloat16


# %% Helper Functions
def set_seed(seed=42):
    """Sets seed for reproducibility across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure reproducibility if desired, may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def find_best_model(model_dir=cfg.working_dir, model_prefix="unet_best_model"):
    """
    Finds the best model file based on filename pattern (lowest loss).
    Falls back to most recently created/modified if pattern fails or doesn't exist.
    """
    best_loss = float("inf")
    best_model_path = None
    pattern = os.path.join(model_dir, f"{model_prefix}_epoch_*_loss_*.pth")
    model_files = glob.glob(pattern)

    if not model_files:
        # Fallback 1: No pattern match -> find latest created .pth
        print(f"W: No models matching pattern '{pattern}'. Looking for *.pth")
        all_pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if all_pth_files:
            best_model_path = max(all_pth_files, key=os.path.getctime, default=None)
            if best_model_path:
                print(
                    f"Using most recently created: {os.path.basename(best_model_path)}"
                )
            else:
                print("W: No .pth models found.")
                return None
        else:
            print("W: No .pth models found in model directory.")
            return None

    elif "loss" in os.path.basename(pattern):
        # Try parsing loss from filename
        parsed_models = []
        for f in model_files:
            try:
                loss_str = f.split("_loss_")[-1].split(".pth")[0]
                loss = float(loss_str)
                parsed_models.append((loss, f))
            except (ValueError, IndexError, AttributeError):
                print(f"W: Couldn't parse loss from filename: {os.path.basename(f)}")

        if parsed_models:
            # Found models with parseable loss, sort by loss
            parsed_models.sort(key=lambda x: x[0])
            best_loss, best_model_path = parsed_models[0]
            print(
                f"Found best model by loss: {os.path.basename(best_model_path)} (Loss: {best_loss:.4f})"
            )
        elif model_files:
            # Pattern matched, but loss couldn't be parsed from any filename
            print(
                "W: Pattern matched but no losses parsed. Selecting most recently created."
            )
            best_model_path = max(model_files, key=os.path.getctime, default=None)
            if best_model_path:
                print(f"Using most recent creation time: {os.path.basename(best_model_path)}")

    else:
        # Pattern matched but doesn't contain "loss" part (unexpected)
        if model_files:
            print(
                f"W: Pattern matched but no loss info expected. Selecting most recently created."
            )
            best_model_path = max(model_files, key=os.path.getctime, default=None)
            if best_model_path:
                print(
                    f"Using most recent creation time match: {os.path.basename(best_model_path)}"
                )

    if not best_model_path:
        # Final fallback: If no model found yet, use the latest modified .pth file
        all_pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if all_pth_files:
            print("W: Fallback: Selecting most recently modified .pth file.")
            best_model_path = max(all_pth_files, key=os.path.getmtime, default=None)
            if best_model_path:
                print(
                    f"Using most recent modification time: {os.path.basename(best_model_path)}"
                )

    return best_model_path


# %% WebDataset Preprocessing Functions
def search_data_path(target_dirs, root_dir, shuffle=True, seed=42):
    """Finds input/output .npy file pairs within subdirectories of a root directory."""
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
            # print(f"W: Target directory {target_dir} not found in {root_path}")
            continue

        in_files, out_files = [], []
        data_subdir = data_dir / "data"
        model_subdir = data_dir / "model"

        # Check for HF structure first, then Kaggle structure
        if data_subdir.is_dir() and model_subdir.is_dir():
            in_files = sorted(data_subdir.glob("*.npy"))
            out_files = sorted(model_subdir.glob("*.npy"))
            # print(f"Found {len(in_files)}/{len(out_files)} files (HF style) in {target_dir}")
        else:
            in_files = sorted(data_dir.glob("seis*.npy"))
            out_files = sorted(data_dir.glob("vel*.npy"))
            # print(f"Found {len(in_files)}/{len(out_files)} files (Kaggle style) in {target_dir}")

        if not in_files or len(in_files) != len(out_files):
            if in_files or out_files:  # Only warn if some files were found
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


# ==================================================
# CORRECTED generate_sample function (No finally block)
# ==================================================
def generate_sample(in_file, out_file=None, base_dir=None):
    """
    Loads data from .npy files, prepares dicts for WebDataset, converts to float16.
    Handles errors during loading gracefully.
    """
    data = []
    seis = None  # Initialize to ensure variable exists for potential del
    vel = None
    try:
        if out_file is None:
            # Logic for test data sharding (if needed later) - not implemented here
            print("W: generate_sample called without out_file (test mode?), not implemented.")
            return []
        else:
            # --- Load Train/Validation data ---
            try:
                # Use mmap_mode='r' for memory efficiency if files are large
                seis = np.load(in_file, mmap_mode="r")
            except Exception as e:
                print(f"E: Load fail for input {in_file.name}: {e}")
                return []  # Exit early if input fails

            try:
                vel = np.load(out_file, mmap_mode="r")
            except Exception as e:
                print(f"E: Load fail for output {out_file.name}: {e}")
                # Clean up the already loaded seis if vel loading fails
                if seis is not None:
                    del seis
                return []  # Exit early if output fails

            # --- Validate shapes and determine number of samples ---
            n_samples = 0
            if seis.ndim == 4 and vel.ndim == 4:  # Batch of samples (N, C, H, W)
                if seis.shape[0] != vel.shape[0]:
                    print(
                        f"W: Batch size mismatch in {in_file.name} ({seis.shape[0]}) vs {out_file.name} ({vel.shape[0]})"
                    )
                    del seis, vel
                    return []
                n_samples = seis.shape[0]
            elif seis.ndim == 3 and vel.ndim == 3:  # Single sample (C, H, W)
                n_samples = 1
            else:
                # Raise error for unexpected dimensions
                raise ValueError(
                    f"Unexpected dims: seis {seis.shape}, vel {vel.shape} in {in_file.name}"
                )

            if n_samples == 0:
                print(f"W: Found 0 samples in pair: {in_file.name}, {out_file.name}")
                del seis, vel
                return []

            # --- Generate unique key based on file path relative to base_dir ---
            common_part = f"{in_file.parent.name}_{in_file.stem}"  # Default key
            if base_dir:
                try:
                    # Create key from relative path parts, removing .npy suffix
                    relative_path = in_file.relative_to(base_dir)
                    common_part = "_".join(relative_path.parts).replace(".npy", "")
                    # Ensure compatibility across OS path separators
                    common_part = common_part.replace(os.sep, "_").replace("\\", "_")
                except ValueError:
                    # If relative_to fails (e.g., different drives), use the default key
                    pass

            # --- Process and append each sample ---
            for i in range(n_samples):
                key = f"{common_part}_{i}"
                # Extract sample, explicitly copy, and convert to float16
                s_sample = (
                    seis[i].copy().astype(np.float16)
                    if seis.ndim == 4
                    else seis.copy().astype(np.float16)
                )
                v_sample = (
                    vel[i].copy().astype(np.float16)
                    if vel.ndim == 4
                    else vel.copy().astype(np.float16)
                )
                data.append(
                    {
                        "__key__": key,
                        "sample_id.txt": key,  # Store key as text too
                        "seis.npy": s_sample,
                        "vel.npy": v_sample,
                    }
                )

            # --- Explicitly delete mmap objects after copying data ---
            # This is important to release file handles, especially with mmap
            del seis
            del vel

    except Exception as e:
        # Catch other errors (ValueError from dim check, key gen errors, etc.)
        print(f"E: Error during sample generation for {in_file.name}: {e}")
        # Explicitly try deleting here, in case they were loaded before the error
        if seis is not None:
            try:
                del seis
            except NameError:  # Should not happen if assigned None initially
                pass
        if vel is not None:
            try:
                del vel
            except NameError:
                pass
        return []  # Return empty list on any error during processing

    # No finally block needed as del is handled within try/except scopes
    return data


# ==================================================


# %% WebDataset Loading Functions
def get_shard_paths(
    root_dir, dataset_name, stage, num_shards=None, test_size=0.2, seed=42
):
    """Gets list of shard paths, optionally selects subset, optionally splits train/val."""
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

    # --- Shard Selection Logic ---
    selected_paths = shard_paths
    available_count = len(shard_paths)
    if num_shards is not None:
        if 0 < num_shards < available_count:
            print(f"Selecting {num_shards} shards randomly (seed={seed}).")
            rng = np.random.default_rng(seed)
            indices = rng.choice(available_count, size=num_shards, replace=False)
            selected_paths = sorted([shard_paths[i] for i in indices])
        elif num_shards >= available_count:
            print(
                f"Requested {num_shards} or more shards, using all {available_count} available."
            )
        else:  # num_shards <= 0
            print(
                f"W: Invalid num_shards ({num_shards}). Using all {available_count} shards."
            )
    print(f"Using {len(selected_paths)} selected shards for stage '{stage}'.")

    # --- Train/Validation Split Logic ---
    if stage == "train":
        count = len(selected_paths)
        print(f"Splitting {count} selected shards (test_size={test_size}, seed={seed})")
        try:
            if not (0 <= test_size < 1):
                raise ValueError("test_size must be in [0, 1)")
            if count <= 1 or test_size == 0:
                reason = "only 1 shard" if count <= 1 else "test_size is 0"
                print(f"W: Cannot split for validation ({reason}). Assigning all to train.")
                return sorted(selected_paths), []
            else:
                trn_paths, val_paths = train_test_split(
                    selected_paths, test_size=test_size, random_state=seed, shuffle=True
                )
                trn_paths.sort()
                val_paths.sort()
                print(f"# Train shards: {len(trn_paths)}, # Val shards: {len(val_paths)}")
                return trn_paths, val_paths
        except Exception as e:
            print(f"E: Failed to split shards: {e}")
            return None, None
    else:  # Not 'train' stage (e.g., 'val' direct loading or 'test')
        print(f"# Shards returned for stage '{stage}': {len(selected_paths)}")
        return sorted(selected_paths)


def get_dataset(paths, stage, seed=42):
    """Creates WebDataset object. Applies augmentations if stage=='train'."""
    if not paths:
        print(f"W: No shard paths provided for stage '{stage}'. Cannot create dataset.")
        return None

    print(f"Creating WebDataset for stage '{stage}' from {len(paths)} shards.")
    is_train = stage == "train"
    
    # Custom handler that will just skip errors rather than warning 
    def silent_skip(exn):
        return None
        
    # Older WebDataset version compatible error handler setup
    map_handler = wds.ignore_and_continue
    
    try:
        # Create the dataset with proper error handling
        dataset = wds.WebDataset(
            paths, 
            nodesplitter=wds.split_by_node, 
            shardshuffle=1 if is_train else 0,  # True/False can cause issues in older versions
            seed=seed,
            handler=wds.ignore_and_continue  # Skip errors at shard level
        )
        
        # Decode standard types (.npy, .txt, etc.)
        dataset = dataset.decode(handler=map_handler)
        
        def map_train_val(sample):
            """Inner function to process decoded samples and apply augmentations."""
            key_info = sample.get("__key__", "N/A")  # For error reporting
            try:
                required = ["sample_id.txt", "seis.npy", "vel.npy"]
                if not all(k in sample for k in required):
                    raise KeyError(f"Missing required keys in sample {key_info}")

                sid = sample["sample_id.txt"]
                # Ensure numpy arrays and convert to float32 tensors
                s_np = np.asarray(sample["seis.npy"])
                v_np = np.asarray(sample["vel.npy"])
                
                # Verify correct shapes - should be (5, 1000, 70) and (1, 70, 70)
                expected_seis_shape = (5, 1000, 70)
                expected_vel_shape = (1, 70, 70)
                
                # Only log a small random subset of samples to prevent log overflow
                if random.random() < 0.01:  # 1% chance to log
                    print(f"\n--- DATASET INSPECTION: Sample Key: {key_info} ---")
                    print(f"Original seismic numpy shape: {s_np.shape}, dtype: {s_np.dtype}")
                    print(f"Original velocity numpy shape: {v_np.shape}, dtype: {v_np.dtype}")
                    
                # Fix shapes if needed (resize/pad to expected dimensions)
                if s_np.shape != expected_seis_shape:
                    print(f"Fixing seismic shape from {s_np.shape} to {expected_seis_shape} for {key_info}")
                    # Simple case: missing batch dimension
                    if len(s_np.shape) == 2 and s_np.shape[0] == 1000 and s_np.shape[1] == 70:
                        # Add channel dimension
                        s_np = np.expand_dims(s_np, axis=0)
                        # Repeat to get 5 channels
                        s_np = np.repeat(s_np, 5, axis=0)
                    # Other cases need proper resizing - convert to tensor for easier interpolation
                    elif s_np.shape != expected_seis_shape:
                        # Need to reshape or resize
                        temp_tensor = torch.from_numpy(s_np.astype(np.float32))
                        # Add missing dimensions if needed
                        while len(temp_tensor.shape) < 3:
                            temp_tensor = temp_tensor.unsqueeze(0)
                        # Ensure we have exactly 3 dimensions (C, H, W)
                        if len(temp_tensor.shape) > 3:
                            if temp_tensor.shape[0] == 1:  # First dim is batch
                                temp_tensor = temp_tensor.squeeze(0)
                            else:
                                # More complex case, just flatten and reshape
                                temp_tensor = temp_tensor.reshape(-1, 1000, 70)
                                temp_tensor = temp_tensor[:5] if temp_tensor.shape[0] >= 5 else temp_tensor
                        # Ensure we have 5 channels
                        if temp_tensor.shape[0] != 5:
                            if temp_tensor.shape[0] < 5:
                                # Repeat the first channel to get to 5
                                temp_tensor = torch.cat([temp_tensor, temp_tensor[0:1].repeat(5-temp_tensor.shape[0], 1, 1)], dim=0)
                            else:
                                # Take the first 5 channels
                                temp_tensor = temp_tensor[:5]
                        # Resize spatial dimensions if needed
                        if temp_tensor.shape[1:] != (1000, 70):
                            temp_tensor = F.interpolate(temp_tensor.unsqueeze(0), size=(1000, 70), mode='bilinear', align_corners=False).squeeze(0)
                        # Convert back to numpy
                        s_np = temp_tensor.numpy()
                
                # Similarly validate and fix velocity tensor shape
                if v_np.shape != expected_vel_shape:
                    print(f"Fixing velocity shape from {v_np.shape} to {expected_vel_shape} for {key_info}")
                    # Simple case: missing channel dimension
                    if len(v_np.shape) == 2 and v_np.shape[0] == 70 and v_np.shape[1] == 70:
                        v_np = np.expand_dims(v_np, axis=0)
                    # Other cases need proper resizing
                    else:
                        temp_tensor = torch.from_numpy(v_np.astype(np.float32))
                        # Add missing dimensions if needed
                        while len(temp_tensor.shape) < 3:
                            temp_tensor = temp_tensor.unsqueeze(0)
                        # Ensure we have exactly 3 dimensions (C, H, W)
                        if len(temp_tensor.shape) > 3:
                            if temp_tensor.shape[0] == 1:  # First dim is batch
                                temp_tensor = temp_tensor.squeeze(0)
                            else:
                                # More complex case, flatten and take first channel
                                temp_tensor = temp_tensor.reshape(-1, 70, 70)
                                temp_tensor = temp_tensor[:1]
                        # Ensure we have 1 channel
                        if temp_tensor.shape[0] != 1:
                            # Take first channel only
                            temp_tensor = temp_tensor[0:1]
                        # Resize spatial dimensions if needed
                        if temp_tensor.shape[1:] != (70, 70):
                            temp_tensor = F.interpolate(temp_tensor.unsqueeze(0), size=(70, 70), mode='bilinear', align_corners=False).squeeze(0)
                        # Convert back to numpy
                        v_np = temp_tensor.numpy()
                
                # Convert to tensors
                seis_tensor = torch.from_numpy(s_np).float()
                vel_tensor = torch.from_numpy(v_np).float()
                
                if random.random() < 0.01:  # Only log occasionally
                    print(f"Loaded seismic tensor: {seis_tensor.shape}")
                    print(f"Loaded velocity tensor: {vel_tensor.shape}")
                
                # --- Augmentation Block ---
                if is_train and cfg.apply_augmentation:
                    # 1. Horizontal Flip
                    if torch.rand(1).item() < cfg.aug_hflip_prob:
                        seis_tensor = TF.hflip(seis_tensor)
                        vel_tensor = TF.hflip(vel_tensor)
                    # 2. Add Gaussian Noise to Seismic Data
                    if cfg.aug_seis_noise_std > 0:
                        noise = torch.randn_like(seis_tensor) * cfg.aug_seis_noise_std
                        seis_tensor.add_(noise)  # In-place addition

                # ====== Compute Gradients/Normals ======
                # Compute gradients with large kernel (5x5 Sobel-like)
                def large_kernel_gradient(tensor):
                    kernel_x = torch.tensor([
                        [-1, -2, 0, 2, 1],
                        [-4, -8, 0, 8, 4],
                        [-6,-12, 0,12, 6],
                        [-4, -8, 0, 8, 4],
                        [-1, -2, 0, 2, 1]
                    ], dtype=torch.float32) / 48.0
                    
                    kernel_y = kernel_x.T
                    
                    # Ensure input is properly shaped for convolution
                    if tensor.dim() == 2:  # Add batch and channel dimensions if needed
                        tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
                    elif tensor.dim() == 3:  # Add channel dimension if needed
                        tensor_4d = tensor.unsqueeze(1)
                    else:
                        tensor_4d = tensor
                    
                    # Verify tensor shape and apply convolution
                    grad_x = F.conv2d(tensor_4d, kernel_x.view(1,1,5,5), padding=2)
                    grad_y = F.conv2d(tensor_4d, kernel_y.view(1,1,5,5), padding=2)
                    
                    # Remove batch dimension if input was 2D or 3D
                    if tensor.dim() <= 3:
                        grad_x = grad_x.squeeze(0)
                        grad_y = grad_y.squeeze(0)
                    
                    # Remove channel dimension always
                    grad_x = grad_x.squeeze(1)
                    grad_y = grad_y.squeeze(1)
                    
                    # Ensure output has same shape as input
                    if grad_x.shape != tensor.shape:
                        grad_x = F.interpolate(grad_x.unsqueeze(1), size=tensor.shape, mode='bilinear', align_corners=False).squeeze(1)
                        grad_y = F.interpolate(grad_y.unsqueeze(1), size=tensor.shape, mode='bilinear', align_corners=False).squeeze(1)
                    
                    return grad_x, grad_y

                # Compute gradients with our large kernel
                grad_x, grad_y = large_kernel_gradient(vel_tensor)
                
                # Debug prints for sizes
                if random.random() < 0.01:  # Print for ~1% of samples
                    print(f"Debug shapes - vel_tensor: {vel_tensor.shape}, grad_x: {grad_x.shape}, grad_y: {grad_y.shape}")
                
                # Handle surface layer (first row)
                if grad_y.dim() > 1 and grad_y.size(1) > 0:  # Check if there's a height dimension
                    grad_y[:,0,:] = vel_tensor[:,0,:]  # Surface velocity as first row
                
                # Compute normals from gradients
                norm = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
                normal_x = -grad_x / norm  # Negative gradient direction
                normal_y = -grad_y / norm

                # Ensure correct dimensions for gradient and normal tensors
                # If vel_tensor is [B, H, W], then grad components are [B, H, W]
                # We want to create [B, 2, H, W] tensors for both grad and normal
                
                # Stack gradients along new channel dimension
                if grad_x.dim() == 2:  # If we have a 2D tensor (H, W)
                    grad_stacked = torch.stack([grad_x, grad_y], dim=0)  # [2, H, W]
                    normal_stacked = torch.stack([normal_x, normal_y], dim=0)  # [2, H, W]
                else:  # We have a 3D tensor (B, H, W)
                    grad_stacked = torch.stack([grad_x, grad_y], dim=1)  # [B, 2, H, W]
                    normal_stacked = torch.stack([normal_x, normal_y], dim=1)  # [B, 2, H, W]
                
                # Make sure tensors have compatible shapes
                if grad_stacked.shape[-2:] != vel_tensor.shape[-2:]:
                    print(f"Shape mismatch: grad_stacked {grad_stacked.shape[-2:]} != vel_tensor {vel_tensor.shape[-2:]}")
                    new_shape = vel_tensor.shape[-2:]
                    
                    # Reshape to 4D if needed for interpolation
                    if grad_stacked.dim() == 3:  # [2, H, W]
                        grad_stacked = grad_stacked.unsqueeze(0)  # [1, 2, H, W]
                        normal_stacked = normal_stacked.unsqueeze(0)  # [1, 2, H, W]
                        need_squeeze = True
                    else:
                        need_squeeze = False
                        
                    # Do the interpolation
                    grad_stacked = F.interpolate(grad_stacked, size=new_shape, mode='bilinear', align_corners=False)
                    normal_stacked = F.interpolate(normal_stacked, size=new_shape, mode='bilinear', align_corners=False)
                    
                    # Squeeze back if we added a dimension
                    if need_squeeze:
                        grad_stacked = grad_stacked.squeeze(0)
                        normal_stacked = normal_stacked.squeeze(0)
                        
                    print(f"After interpolation: grad_stacked {grad_stacked.shape[-2:]}")
                
                # More debug prints
                if random.random() < 0.01:  # Print for ~1% of samples 
                    print(f"Final shapes - vel: {vel_tensor.shape}, grad: {grad_stacked.shape}, normal: {normal_stacked.shape}")

                # Ensure all tensors have batch dimension
                if seis_tensor.dim() == 2:
                    seis_tensor = seis_tensor.unsqueeze(0)
                if vel_tensor.dim() == 2:
                    vel_tensor = vel_tensor.unsqueeze(0)
                if grad_stacked.dim() == 2:
                    grad_stacked = grad_stacked.unsqueeze(0)
                if normal_stacked.dim() == 2:
                    normal_stacked = normal_stacked.unsqueeze(0)

                return {
                    "__key__": key_info,  # Keep the key for debugging
                    "sample_id": sid,
                    "seis": seis_tensor,
                    "vel": vel_tensor,
                    "grad": grad_stacked,
                    "normal": normal_stacked
                }

            except Exception as map_e:
                print(f"E: Map function failed for sample {key_info}: {map_e}")
                # Let the handler decide whether to skip or raise
                raise map_e

        # Apply the mapping function to train/val stages
        if stage in ["train", "val"]:
            dataset = dataset.map(map_train_val, handler=map_handler)

        # Shuffle buffer for training data
        if is_train:
            dataset = dataset.shuffle(1000)  # Buffer size for shuffling
        
        # Check for empty items (compatible with older WebDataset versions)
        def check_sample(sample):
            return sample is not None and len(sample) > 0
            
        # Make sure all items in a batch have the same tensor shapes
        def check_batch_consistency(batch):
            """Make sure all items in a batch have the same tensor shapes"""
            if not batch:
                return None
                
            try:
                # First filter out any non-dictionary items (like strings)
                valid_items = []
                for item in batch:
                    if isinstance(item, dict) and "seis" in item and "vel" in item:
                        valid_items.append(item)
                    else:
                        # If we encounter a non-dictionary or incomplete item, log it
                        item_type = type(item).__name__
                        print(f"Skipping invalid batch item of type {item_type}")
                
                # If no valid items remain, return None
                if not valid_items:
                    print("No valid items found in batch, skipping")
                    return None
                
                # Extract tensor shapes only from valid dictionary items
                seis_shapes = [item["seis"].shape for item in valid_items]
                vel_shapes = [item["vel"].shape for item in valid_items]
                
                # Check if all shapes are the same
                unique_seis_shapes = set(str(s) for s in seis_shapes)
                unique_vel_shapes = set(str(s) for s in vel_shapes)
                
                # Log shape info for debugging
                if len(unique_seis_shapes) > 1 or len(unique_vel_shapes) > 1:
                    print(f"Found inconsistent shapes in batch: seis={unique_seis_shapes}, vel={unique_vel_shapes}")
                    return None  # Skip this batch
                    
                # Use only the valid items for the batch
                return valid_items  # Return the filtered batch
            except Exception as e:
                print(f"Error in batch filtering: {e}")
                return None
        
        # WebDataset v0.2.111 compatible filtering and batching
        # For older versions, use to_tuple() and then from_tuple() as a replacement for select/filter
        
        # Check if this version has the select method
        if hasattr(dataset, 'select'):
            # Modern WebDataset
            dataset = dataset.select(check_sample)  # Filter out empty items
            dataset = dataset.batched(cfg.batch_size, partial=True)
            dataset = dataset.map(check_batch_consistency)
            dataset = dataset.select(lambda x: x is not None)  # Remove filtered out batches
        else:
            # Older WebDataset version without select/filter
            # Convert to tuples
            dataset = dataset.to_tuple("__key__ sample_id.txt seis.npy vel.npy".split())
            
            # Define a function to filter and process tuples
            def process_tuple(sample_tuple):
                try:
                    # Unpack the tuple
                    key, sample_id, seis_np, vel_np = sample_tuple
                    
                    # Skip if any component is None
                    if key is None or sample_id is None or seis_np is None or vel_np is None:
                        return None
                        
                    # Process like map_train_val would
                    key_info = key
                    s_np = np.asarray(seis_np)
                    v_np = np.asarray(vel_np)
                    
                    # Verify correct shapes - should be (5, 1000, 70) and (1, 70, 70)
                    expected_seis_shape = (5, 1000, 70)
                    expected_vel_shape = (1, 70, 70)
                    
                    # Fix shapes if needed
                    # (Shape fixing code would go here - simplified for readability)
                    
                    # Convert to tensors
                    seis_tensor = torch.from_numpy(s_np).float()
                    vel_tensor = torch.from_numpy(v_np).float()
                    
                    # Compute gradients/normals
                    # (Remaining logic from map_train_val)
                    
                    return {
                        "__key__": key_info,
                        "sample_id": sample_id,
                        "seis": seis_tensor,
                        "vel": vel_tensor
                        # Add grad and normal here too
                    }
                    
                except Exception as e:
                    print(f"Error processing sample tuple: {e}")
                    return None
                    
            # Map the function over the dataset
            dataset = dataset.map(process_tuple)
            # Skip None values
            dataset = dataset.compose(lambda src: (x for x in src if x is not None))
            # Batch the dataset
            dataset = dataset.batched(cfg.batch_size)

        return dataset

    except Exception as e:
        print(f"E: Error creating WebDataset pipeline for stage '{stage}': {e}")
        return None


# %% Kaggle TestSet Loading (Directly from .npy)
class KaggleTestDataset(Dataset):
    """Loads the final Kaggle test set directly from individual .npy files."""

    def __init__(self, test_files_dir):
        self.test_files_dir = Path(test_files_dir)
        self.test_files = []
        try:
            if not self.test_files_dir.is_dir():
                raise FileNotFoundError(
                    f"Kaggle test directory missing: {self.test_files_dir}"
                )
            self.test_files = sorted(list(self.test_files_dir.glob("*.npy")))
            print(
                f"Found {len(self.test_files)} '.npy' files in Kaggle test dir: {self.test_files_dir}"
            )
            if not self.test_files:
                print(f"W: No .npy files found in {self.test_files_dir}.")
        except Exception as e:
            print(
                f"E: Error accessing Kaggle test directory {self.test_files_dir}: {e}"
            )

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, index):
        if not self.test_files or index >= len(self.test_files):
            raise IndexError(
                f"Index {index} out of bounds for KaggleTestDataset ({len(self.test_files)} files)."
            )
        test_file_path = self.test_files[index]
        try:
            # Load numpy array and convert to float32 tensor
            data = torch.from_numpy(np.load(test_file_path).astype(np.float32))
            # Get the original ID (filename without extension)
            original_id = test_file_path.stem
            return data, original_id
        except Exception as e:
            # Raise a more informative error if loading fails
            raise IOError(f"Error loading Kaggle test file: {test_file_path}") from e


# %% U-Net Model Definition (Formatted for Readability with Residual Blocks)

class ResidualDoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2 + Residual Connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution layer
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle potential channel mismatch
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            # Projection shortcut: 1x1 conv + BN to match output channels and spatial dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x  # Store the input for the residual connection
        # Only print debug info rarely to reduce memory usage
        debug_logging = random.random() < 0.001  # 0.1% chance
        
        if debug_logging:
            print(f"ResidualDoubleConv input: {x.shape}")

        # First conv block
        out = self.conv1(x)
        if debug_logging:
            print(f"After conv1: {out.shape}")
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block (without final ReLU yet)
        out = self.conv2(out)
        if debug_logging:
            print(f"After conv2: {out.shape}")
        out = self.bn2(out)

        # Apply shortcut to the identity path
        identity_mapped = self.shortcut(identity)
        if debug_logging:
            print(f"Identity after shortcut: {identity_mapped.shape}")
            
        # Ensure dimensions match before adding residual connection
        # Both out and identity_mapped should have the same shape
        if out.shape != identity_mapped.shape:
            if debug_logging:
                print(f"Warning: Shape mismatch in residual connection - out: {out.shape}, identity: {identity_mapped.shape}")
                
            # Resize the smaller one to match the larger one
            if out.shape[-1] > identity_mapped.shape[-1] or out.shape[-2] > identity_mapped.shape[-2]:
                # Identity is smaller, resize it to match out
                identity_mapped = F.interpolate(identity_mapped, size=out.shape[-2:], mode='bilinear', align_corners=False)
                if debug_logging:
                    print(f"Resized identity to: {identity_mapped.shape}")
            else:
                # Out is smaller, resize it to match identity
                out = F.interpolate(out, size=identity_mapped.shape[-2:], mode='bilinear', align_corners=False)
                if debug_logging:
                    print(f"Resized out to: {out.shape}")

        # Add the residual connection
        out += identity_mapped

        # Apply final ReLU
        out = self.relu(out)
        
        # Final check - ensure output has desired dimensions (70x70)
        if out.shape[-1] != 70 or out.shape[-2] != 70:
            # Need to adjust dimensions
            if debug_logging:
                print(f"Adjusting output to 70x70 (currently {out.shape[-2]}x{out.shape[-1]})")
                
            # Use center crop or padding to get to 70x70
            # Pad if smaller
            if out.shape[-1] < 70 or out.shape[-2] < 70:
                diff_y = max(0, 70 - out.shape[-2])
                diff_x = max(0, 70 - out.shape[-1])
                pad_top = diff_y // 2
                pad_bottom = diff_y - pad_top
                pad_left = diff_x // 2
                pad_right = diff_x - pad_left
                out = F.pad(out, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
                
            # Crop if larger
            if out.shape[-1] > 70 or out.shape[-2] > 70:
                crop_y = out.shape[-2] - 70
                crop_x = out.shape[-1] - 70
                start_y = crop_y // 2
                start_x = crop_x // 2
                out = out[..., start_y:start_y+70, start_x:start_x+70]
                
            if debug_logging:
                print(f"Final adjusted output: {out.shape}")
        
        if debug_logging:
            print(f"ResidualDoubleConv output: {out.shape}")
        return out


class Up(nn.Module):
    """Upscaling then ResidualDoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            # For bilinear upsampling, we need to handle the channel concatenation
            self.conv = ResidualDoubleConv(in_channels + out_channels, out_channels)
        else:
            # For transposed conv, we first reduce channels then concatenate
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualDoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is from below (needs upsampling), x2 is skip connection
        x1 = self.up(x1)
        
        # Ensure spatial dimensions match
        if x1.shape[-1] != x2.shape[-1] or x1.shape[-2] != x2.shape[-2]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 Convolution for the output layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture implementation with Residual Blocks"""

    def __init__(
        self,
        n_channels=5,  # Fixed to 5 for seismic input channels
        n_classes=1,   # Fixed to 1 for velocity output
        init_features=24,
        depth=4,
        bilinear=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth

        # Initial pooling to reduce time dimension
        self.initial_pool = nn.AvgPool2d(kernel_size=(100, 1), stride=(100, 1))  # Reduce 1000 to 10

        # Encoder
        self.encoder_convs = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        # Initial conv block
        self.inc = ResidualDoubleConv(n_channels, init_features)
        self.encoder_convs.append(self.inc)

        current_features = init_features
        for _ in range(depth):
            # Use stride=2 in conv instead of separate pooling to maintain spatial dimensions
            conv = ResidualDoubleConv(current_features, current_features * 2, stride=2)
            self.encoder_convs.append(conv)
            current_features *= 2

        # Bottleneck
        self.bottleneck = ResidualDoubleConv(current_features, current_features)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for _ in range(depth):
            up_block = Up(current_features, current_features // 2, bilinear)
            self.decoder_blocks.append(up_block)
            current_features //= 2

        # Output layer for velocity
        self.outc = OutConv(current_features, n_classes)

        # Physics-guided prediction heads
        self.grad_head = nn.Sequential(
            nn.Conv2d(current_features, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)  # Predicts grad_x, grad_y
        )
        
        self.normal_head = nn.Sequential(
            nn.Conv2d(current_features, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)  # Predicts normal_x, normal_y
        )

    def forward(self, x):
        # Input shape could be [B, 5, 1000, 70] or [B, B, 5, 1000, 70] with double batching
        # Handle various potential tensor shapes
        orig_shape = x.shape

        try:
            # Handle the double-batching case
            if x.dim() > 4:
                print(f"UNet received {x.dim()}D input with shape {x.shape}, reshaping...")
                # If we have [B, B, C, H, W] shape, reshape to [B*B, C, H, W]
                if x.dim() == 5:
                    batch_size = x.size(0) * x.size(1)
                    x = x.reshape(batch_size, *x.shape[2:])
                    print(f"Reshaped to {x.shape}")
                else:
                    print(f"Unexpected tensor dimensionality: {x.dim()}, attempting to process")
                
            # Ensure we have a batch dimension
            if x.dim() == 3:  # [C, H, W] -> [1, C, H, W]
                x = x.unsqueeze(0)
                print(f"Added batch dimension, shape now: {x.shape}")
                
            # Ensure the input tensor is in the correct format [B, 5, 1000, 70]
            if x.dim() == 4:
                # Check channel dimension is 5
                if x.size(1) != 5:
                    print(f"Warning: Channel dimension is {x.size(1)}, expected 5")
                    
                # Handle different spatial dimensions with interpolation
                if x.size(2) != 1000 or x.size(3) != 70:
                    print(f"Warning: Spatial dimensions are {x.size(2)}x{x.size(3)}, expected 1000x70")
                    x = F.interpolate(x, size=(1000, 70), mode='bilinear', align_corners=False)
                    print(f"Interpolated to shape: {x.shape}")
            
            # Input shape: [B, 5, 1000, 70]
            # First reduce time dimension
            x = self.initial_pool(x)  # [B, 5, 10, 70]
            
        except Exception as e:
            print(f"Error preprocessing input tensor (shape {orig_shape}): {e}")
            # Emergency reshape - try to make it work with minimal assumptions
            try:
                if x.dim() >= 3:
                    # Extract relevant dimensions and reshape
                    target_shape = [-1, 5, 1000, 70]  # Target shape with unknown batch size
                    total_elements = x.numel()
                    batch_size = total_elements // (5 * 1000 * 70)  # Calculate batch size
                    
                    if batch_size > 0:
                        x = x.reshape(batch_size, 5, 1000, 70)
                        print(f"Emergency reshape to {x.shape}")
                        x = self.initial_pool(x)  # Try to continue with pipeline
                    else:
                        # If tensor is too small, pad it
                        print(f"Tensor too small, emergency padding")
                        x = torch.zeros(1, 5, 1000, 70, device=x.device, dtype=x.dtype)
                        x = self.initial_pool(x)
                else:
                    # Complete failure case, just create a dummy tensor
                    print(f"Unable to reshape tensor, creating emergency dummy")
                    x = torch.zeros(1, 5, 10, 70, device=x.device, dtype=x.dtype)
            except Exception as e2:
                print(f"Emergency reshape failed: {e2}, creating dummy tensor")
                # Last resort dummy tensor
                x = torch.zeros(1, 5, 10, 70, device=x.device, dtype=x.dtype)

        # Initial conv
        x1 = self.encoder_convs[0](x)  # [B, 24, 10, 70]
        
        # Store skip connections
        skips = [x1]
        
        # Encoder path
        for i in range(self.depth):
            x1 = self.encoder_convs[i+1](self.encoder_pools[i](x1))
            if i < self.depth - 1:
                skips.append(x1)
        
        # Bottleneck
        x1 = self.bottleneck(x1)
        
        # Decoder path
        for i in range(self.depth):
            # Get skip connection
            skip = skips[-(i+1)]
            
            # Ensure skip connection has correct number of channels
            if skip.shape[1] != x1.shape[1] // 2:
                # Create a 1x1 conv to adjust channels
                conv = nn.Conv2d(skip.shape[1], x1.shape[1] // 2, 1).to(skip.device)
                skip = conv(skip)
            
            x1 = self.decoder_blocks[i](x1, skip)
        
        # Main velocity prediction
        velocity = self.outc(x1)  # [B, 1, 70, 70]
        
        # Physics-guided predictions
        gradients = self.grad_head(x1)  # [B, 2, 70, 70]
        normals = self.normal_head(x1)  # [B, 2, 70, 70]
        
        # Ensure all outputs are exactly 70x70
        if velocity.shape[-1] != 70 or velocity.shape[-2] != 70:
            velocity = F.interpolate(velocity, size=(70, 70), mode='bilinear', align_corners=False)
        if gradients.shape[-1] != 70 or gradients.shape[-2] != 70:
            gradients = F.interpolate(gradients, size=(70, 70), mode='bilinear', align_corners=False)
        if normals.shape[-1] != 70 or normals.shape[-2] != 70:
            normals = F.interpolate(normals, size=(70, 70), mode='bilinear', align_corners=False)
        
        return velocity, gradients, normals


def ensure_tensor_sizes(pred, target, expected_size=70):
    """Ensure tensors have the exact size needed in spatial dimensions"""
    
    # First, determine if resizing is needed
    spatial_dim = pred.shape[-1]  # Get the last dimension (width)
    
    if spatial_dim != expected_size:
        print(f"Resizing tensors from spatial dim {spatial_dim} to {expected_size}")
        
        # Create paddings for spatial dimensions
        if spatial_dim < expected_size:
            # Need to pad
            pad_size = expected_size - spatial_dim
            left_pad = pad_size // 2
            right_pad = pad_size - left_pad
            
            # For 4D tensors: [batch, channels, height, width]
            # Format: (left_pad, right_pad, top_pad, bottom_pad)
            padding = (left_pad, right_pad, left_pad, right_pad)
            
            # Use padding instead of interpolation to preserve values
            pred_resized = F.pad(pred, padding, mode='replicate')
            target_resized = F.pad(target, padding, mode='replicate')
            
            print(f"Padded tensors: {pred.shape} -> {pred_resized.shape}, {target.shape} -> {target_resized.shape}")
        else:
            # Need to crop
            crop_size = spatial_dim - expected_size
            left_crop = crop_size // 2
            right_crop = left_crop + expected_size
            
            # Crop the tensors (using proper slicing)
            pred_resized = pred[..., :, left_crop:right_crop, left_crop:right_crop]
            target_resized = target[..., :, left_crop:right_crop, left_crop:right_crop]
            
            print(f"Cropped tensors: {pred.shape} -> {pred_resized.shape}, {target.shape} -> {target_resized.shape}")
        
        return pred_resized, target_resized
    
    # No resize needed
    return pred, target


def verify_batch_tensors(inputs, targets, grads, normals, expected_size=70):  # Changed to expect 70 not 71
    """Helper function to check and fix tensor shapes for compatibility."""
    
    def log_shape(name, tensor):
        # Only log occasionally to avoid log bloat
        if random.random() < 0.1:  # 10% chance to log
            print(f"{name} shape: {tensor.shape}")
    
    # Log shapes of all tensors
    log_shape("inputs", inputs)
    log_shape("targets", targets)
    log_shape("grads", grads)
    log_shape("normals", normals)
    
    try:
        # Reshape tensors if needed
        # Handle 5D grads tensor: [batch, 1, 2, h, w] -> [batch, 2, h, w]
        if grads.dim() == 5:
            if random.random() < 0.1:
                print(f"Reshaping 5D grads to 4D: {grads.shape} -> ", end="")
            grads = grads.squeeze(1)  # Remove the extra dimension
            if random.random() < 0.1:
                print(f"{grads.shape}")
        
        # Handle 5D normals tensor: [batch, 1, 2, h, w] -> [batch, 2, h, w]
        if normals.dim() == 5:
            if random.random() < 0.1:
                print(f"Reshaping 5D normals to 4D: {normals.shape} -> ", end="")
            normals = normals.squeeze(1)  # Remove the extra dimension
            if random.random() < 0.1:
                print(f"{normals.shape}")
        
        # Ensure targets have correct dimensions
        # If targets shape is [B, 1, H, W] and H=W=70, we need to ensure it is exactly 70x70
        if targets.shape[-1] != expected_size or targets.shape[-2] != expected_size:
            if targets.dim() == 3:  # [B, H, W]
                targets = targets.unsqueeze(1)  # [B, 1, H, W]
            padding_mode = 'replicate'
            if random.random() < 0.1:
                print(f"Padding targets from {targets.shape[-2:]} to [{expected_size}, {expected_size}] using {padding_mode} mode")
            
            # Calculate padding
            h_pad = max(0, expected_size - targets.shape[-2])
            w_pad = max(0, expected_size - targets.shape[-1])
            
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left
            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            
            # Apply padding
            targets = F.pad(targets, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode)
            if random.random() < 0.1:
                print(f"Padded targets to: {targets.shape}")
        
        # Same for gradients and normals - ensure 4D tensors
        if grads.dim() == 3:  # Missing batch dimension
            if random.random() < 0.1:
                print(f"Adding batch dimension to grads: {grads.shape} -> ", end="")
            grads = grads.unsqueeze(0)  # [1, 2, H, W]
            if random.random() < 0.1:
                print(f"{grads.shape}")
        
        if normals.dim() == 3:
            if random.random() < 0.1:
                print(f"Adding batch dimension to normals: {normals.shape} -> ", end="")
            normals = normals.unsqueeze(0)  # [1, 2, H, W]
            if random.random() < 0.1:
                print(f"{normals.shape}")
        
        # Pad/crop grads to expected size
        if grads.shape[-1] != expected_size or grads.shape[-2] != expected_size:
            padding_mode = 'replicate'
            if random.random() < 0.1:
                print(f"Padding grads from {grads.shape[-2:]} to [{expected_size}, {expected_size}] using {padding_mode} mode")
            
            # Calculate padding
            h_pad = max(0, expected_size - grads.shape[-2])
            w_pad = max(0, expected_size - grads.shape[-1])
            
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left
            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            
            # Apply padding
            grads = F.pad(grads, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode)
            if random.random() < 0.1:
                print(f"Padded grads to: {grads.shape}")
        
        # Same for normals
        if normals.shape[-1] != expected_size or normals.shape[-2] != expected_size:
            padding_mode = 'replicate'
            if random.random() < 0.1:
                print(f"Padding normals from {normals.shape[-2:]} to [{expected_size}, {expected_size}] using {padding_mode} mode")
            
            # Calculate padding
            h_pad = max(0, expected_size - normals.shape[-2])
            w_pad = max(0, expected_size - normals.shape[-1])
            
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left
            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            
            # Apply padding
            normals = F.pad(normals, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode)
            if random.random() < 0.1:
                print(f"Padded normals to: {normals.shape}")
            
    except Exception as e:
        print(f"Error during tensor verification: {e}")
        print("Using fallback approach for tensor reshaping...")
        
        # Fallback approach - just make sure dimensions are compatible for forward pass
        if grads.dim() == 5:
            grads = grads.squeeze(1)
        if normals.dim() == 5:
            normals = normals.squeeze(1)
            
        # Make sure tensors have 4 dimensions for interpolation
        if grads.dim() < 4:
            grads = grads.view(-1, grads.size(0), grads.size(1), grads.size(2))
        if normals.dim() < 4:
            normals = normals.view(-1, normals.size(0), normals.size(1), normals.size(2))
        
        # Emergency fallback - resize everything to expected size
        if targets.shape[-1] != expected_size:
            # Force targets to expected size
            targets = F.interpolate(targets, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
            grads = F.interpolate(grads, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
            normals = F.interpolate(normals, size=(expected_size, expected_size), mode='bilinear', align_corners=False)
            print(f"Emergency resizing to {expected_size}x{expected_size}")
    
    # One final check and force fix
    if targets.shape[-1] != expected_size:
        targets = F.pad(targets, (0, expected_size - targets.shape[-1], 0, expected_size - targets.shape[-2]), mode='replicate')
    if grads.shape[-1] != expected_size:
        grads = F.pad(grads, (0, expected_size - grads.shape[-1], 0, expected_size - grads.shape[-2]), mode='replicate')
    if normals.shape[-1] != expected_size:
        normals = F.pad(normals, (0, expected_size - normals.shape[-1], 0, expected_size - normals.shape[-2]), mode='replicate')
    
    return inputs, targets, grads, normals


# %% Main Execution
print("--- Starting Full Workflow (U-Net with Residual Blocks) ---")
set_seed(cfg.seed)
print(f"Device: {cfg.device}")
print(f"Using PyTorch version: {torch.__version__}")
if cfg.use_cuda:
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# Cleanup Code
# ==============================================================================



print("\n--- Cleaning up previous run artifacts ---")
paths_to_clean = [cfg.shard_output_dir]
# Find previous best model files based on pattern
model_pattern = os.path.join(cfg.working_dir, "unet_best_model_epoch_*_loss_*.pth")
paths_to_clean.extend(glob.glob(model_pattern))
# Add plot and submission files
paths_to_clean.append(os.path.join(cfg.working_dir, "training_history.png"))
paths_to_clean.append(cfg.submission_file)

for path_str in paths_to_clean:
    path_obj = Path(path_str)
    try:
        if path_obj.is_dir():
            print(f"Attempting to remove directory: {path_obj}")
            shutil.rmtree(path_obj, ignore_errors=True)
            print(f"Removed directory (if existed): {path_obj}")
        elif path_obj.is_file():
            print(f"Attempting to remove file: {path_obj}")
            path_obj.unlink(missing_ok=True)  # Ignore error if file doesn't exist
            print(f"Removed file (if existed): {path_obj}")
    except Exception as e:
        print(f"W: Error during cleanup of {path_obj}: {e}")
print("--- Cleanup finished ---")
gc.collect()
# ==============================================================================

# ==============================================================================
# SECTION 0/1: Sharding from Kaggle Data Only
# ==============================================================================
print("\n--- 0/1. Sharding from Kaggle Data Only ---")
# Use updated dataset name in shard path
shard_stage_dir = Path(cfg.shard_output_dir) / f"train_{cfg.dataset_name}"
kaggle_train_root = Path(cfg.kaggle_train_dir)
needs_creation = True
total_samples_written = 0

try:
    # --- Check if shards need creating ---
    if shard_stage_dir.exists() and any(shard_stage_dir.glob("*.tar")):
        if cfg.force_shard_creation:
            print(f"Forcing shard creation. Removing existing shards in {shard_stage_dir}")
            shutil.rmtree(shard_stage_dir)
        else:
            print(f"Found existing shards at: {shard_stage_dir}. Skipping creation.")
            needs_creation = False

    # --- Ensure output directories exist & Check Disk Space ---
    print("\n--- Checking Disk Space Before Directory Creation ---")
    try:
        total, used, free = shutil.disk_usage(cfg.working_dir)
        print(
            f"Disk Usage for {cfg.working_dir}: Total={total / 1e9:.2f}GB, Used={used / 1e9:.2f}GB, Free={free / 1e9:.2f}GB"
        )
    except Exception as du_e:
        print(f"W: Could not check disk usage: {du_e}")

    try:
        Path(cfg.shard_output_dir).mkdir(parents=True, exist_ok=True)
        shard_stage_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"E: Critical error creating output directories: {e}")
        raise  # Stop if directories can't be created

    # --- Sharding Process ---
    if needs_creation:
        print(
            f"Starting shard creation from {kaggle_train_root} into {shard_stage_dir}"
        )
        if not kaggle_train_root.is_dir():
            raise FileNotFoundError(f"Kaggle train directory not found: {kaggle_train_root}")

        # Find family subdirectories in the Kaggle train directory
        families = [d.name for d in kaggle_train_root.iterdir() if d.is_dir()]
        print(f"Searching Kaggle data families: {families}")
        if not families:
            raise FileNotFoundError(
                f"No family subdirectories found in {kaggle_train_root}"
            )

        print("Searching for all data pairs in Kaggle source...")
        kaggle_file_pairs = search_data_path(
            families, kaggle_train_root, shuffle=True, seed=cfg.seed
        )
        print(f"Found {len(kaggle_file_pairs)} total valid pairs from Kaggle source.")
        if not kaggle_file_pairs:
            raise RuntimeError(
                "No valid data pairs found in the specified Kaggle directories."
            )

        # --- Write Shards ---
        shard_pattern = str(shard_stage_dir / "%06d.tar")
        print(
            f"Writing shards using pattern {shard_pattern} (max size {cfg.maxsize / 1e9:.2f} GB)"
        )
        with wds.ShardWriter(shard_pattern, maxsize=int(cfg.maxsize)) as writer:
            common_base_dir = kaggle_train_root  # For relative path key generation
            for in_file, out_file in tqdm(
                kaggle_file_pairs, desc="Sharding Kaggle Data", unit="pair"
            ):
                # generate_sample handles potential errors for each pair
                samples_from_pair = generate_sample(
                    Path(in_file), Path(out_file), base_dir=common_base_dir
                )
                if samples_from_pair:
                    for sample_dict in samples_from_pair:
                        writer.write(sample_dict)
                    total_samples_written += len(samples_from_pair)

        print(
            f"Finished writing {total_samples_written} samples from Kaggle source to shards."
        )

    elif not needs_creation:
        existing_shard_count = len(list(shard_stage_dir.glob("*.tar")))
        print(f"Using {existing_shard_count} existing shards.")

except Exception as e:
    print(f"E: Kaggle-only sharding process failed critically: {e}")
    import traceback

    traceback.print_exc()
    raise
# ==============================================================================


# --- 2. Get Train/Val DataLoaders from Created Shards ---
print("\n--- 2. Creating DataLoaders from Shards ---")
dltrain, dlvalid = None, None
val_paths_saved = []  # Keep track of validation paths for potential later use
try:
    # Use updated dataset name for getting paths
    trn_paths, val_paths = get_shard_paths(
        cfg.shard_output_dir,
        cfg.dataset_name, # Use updated name
        "train",  # Request splitting
        num_shards=cfg.num_used_shards,
        test_size=cfg.test_size,
        seed=cfg.seed,
    )
    val_paths_saved = val_paths  # Save the validation paths

    if trn_paths is None:
        # get_shard_paths returns None, None on critical split error
        raise RuntimeError("Failed to get or split shard paths for train/val.")

    # Check if any shards actually exist if paths were returned empty
    # Use updated dataset name in check path
    shard_check_dir = Path(cfg.shard_output_dir) / f"train_{cfg.dataset_name}"
    if not trn_paths and not list(shard_check_dir.glob("*.tar")):
        raise RuntimeError(
            f"No training shards selected AND no .tar files found in {shard_check_dir}."
        )

    # Report shard counts
    if not trn_paths:
        print("W: No shards assigned for training. Training cannot proceed.")
    else:
        print(f"Using {len(trn_paths)} shards for training.")
    if not val_paths:
        print("W: No shards assigned for validation. Validation will be skipped.")
    else:
        print(f"Using {len(val_paths)} shards for validation.")

    # Create WebDatasets (Augmentation applied in get_dataset for 'train')
    trn_ds = get_dataset(trn_paths, "train", seed=cfg.seed) if trn_paths else None
    val_ds = get_dataset(val_paths, "val", seed=cfg.seed + 1) if val_paths else None

    # Check if dataset creation failed unexpectedly
    if trn_ds is None and trn_paths:
        raise RuntimeError("Failed to create train WebDataset pipeline.")
    if val_ds is None and val_paths:
        # Only warn if validation dataset failed, training might still proceed
        print("W: Failed to create validation WebDataset pipeline.")

    # Create DataLoaders
    if trn_ds:
        n_trn_w = min(cfg.num_workers, len(trn_paths)) if trn_paths else 0
        p_trn = n_trn_w > 0  # Use persistent workers only if num_workers > 0
        
        # For WebDataset 0.2.111, we don't need a custom collation function
        # because we're handling the batching at the WebDataset level
        dltrain = DataLoader(
            trn_ds,  # WebDataset already handles batching
            batch_size=None,  # No additional batching in DataLoader
            shuffle=False,  # Shuffling done by WebDataset
            num_workers=n_trn_w,
            pin_memory=cfg.use_cuda,
            persistent_workers=p_trn,
            prefetch_factor=2 if p_trn else None,  # Only relevant if num_workers > 0
        )
        print(f"Train DataLoader created with {n_trn_w} workers (WebDataset v0.2.111 compatible).")
        
    if val_ds:
        n_val_w = min(cfg.num_workers, len(val_paths)) if val_paths else 0
        p_val = n_val_w > 0
        dlvalid = DataLoader(
            val_ds,  # WebDataset already handles batching
            batch_size=None,
            shuffle=False,
            num_workers=n_val_w,
            pin_memory=cfg.use_cuda,
            persistent_workers=p_val,
            prefetch_factor=2 if p_val else None,
        )
        print(f"Validation DataLoader created with {n_val_w} workers (WebDataset v0.2.111 compatible).")

    # Final check (can sometimes trigger TypeError: 'IterableDataset' has no len())
    try:
        loaders_exist = bool(dltrain or dlvalid)
        if not loaders_exist and (trn_paths or val_paths):
             # Should not happen if datasets were created but loaders failed
             raise RuntimeError("Loaders missing despite dataset paths existing.")
        print("DataLoader(s) created successfully or skipped appropriately.")
    except TypeError as te:
        # Expected error for IterableDataset without explicit length
        if "has no len()" in str(te):
            print(f"W: Caught expected TypeError '{te}'. Assume DataLoaders are ok.")
        else:
            raise te  # Re-raise unexpected TypeError
except Exception as e:
    print(f"E: DataLoader creation failed critically: {e}")
    import traceback
    traceback.print_exc()
    raise


# --- 3. Initialize Model, Loss, Optimizer ---
print("\n--- 3. Initializing Model, Loss, Optimizer ---")
model = None
try:
    # Instantiate the modified U-Net
    model = UNet().to(cfg.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} (Residual), Trainable Params: {params:,}")
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    print(f"Loss Function: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__} (lr={cfg.learning_rate}, wd={cfg.weight_decay})")
except Exception as e:
    print(f"E: Model initialization failed: {e}")
    raise


# --- 4. Training Loop ---
print("\n--- 4. Starting Training ---")
history = []
best_val_loss = float("inf")

if dltrain is None or model is None:
    print("E: Training cannot proceed. Train DataLoader or Model is missing.")
else:
    try:
        for epoch in range(1, cfg.n_epochs + 1):
            print(f"\n=== Epoch {epoch}/{cfg.n_epochs} ===")
            # --- Training Phase ---
            gc.collect()  # Run garbage collection before each epoch
            if cfg.use_cuda:
                torch.cuda.empty_cache()
            model.train()
            train_losses = []
            pbar_train = tqdm(dltrain, desc=f"Train E{epoch}", leave=False, unit="batch")
            for i, batch in enumerate(pbar_train):
                if not batch or "seis" not in batch or "vel" not in batch:
                    print(f"W: Skipping invalid train batch {i}")
                    continue
                try:
                    # Check if batch is a valid dictionary with required fields
                    if not isinstance(batch, dict):
                        print(f"W: Skipping invalid train batch {i} (type: {type(batch).__name__})")
                        continue
                        
                    if "seis" not in batch or "vel" not in batch or "grad" not in batch or "normal" not in batch:
                        print(f"W: Skipping incomplete train batch {i} (missing keys)")
                        continue
                        
                    # Access batch dictionary elements and move them to device
                    inputs = batch["seis"].to(cfg.device, non_blocking=True)
                    targets = batch["vel"].to(cfg.device, non_blocking=True)
                    batch_grads = batch["grad"].to(cfg.device)
                    batch_normals = batch["normal"].to(cfg.device)

                    # Fix double batching issue - detect and reshape tensor if it has extra dimensions
                    # Error example: [8, 8, 5, 1000, 70] - has an extra batch dimension
                    if inputs.dim() > 4:
                        print(f"Fixing double-batched inputs: shape before = {inputs.shape}")
                        # Calculate new shape: flatten the first two dimensions if they're both batch dimensions
                        new_batch_size = inputs.size(0) * inputs.size(1)
                        new_shape = [new_batch_size] + list(inputs.shape[2:])
                        inputs = inputs.reshape(new_shape).float()
                        print(f"Fixed inputs shape = {inputs.shape}")

                    # Similarly fix other tensors if they have extra dimensions
                    if targets.dim() > 4:
                        print(f"Fixing double-batched targets: shape before = {targets.shape}")
                        new_batch_size = targets.size(0) * targets.size(1)
                        new_shape = [new_batch_size] + list(targets.shape[2:])
                        targets = targets.reshape(new_shape).float()
                        print(f"Fixed targets shape = {targets.shape}")

                    if batch_grads.dim() > 4:
                        print(f"Fixing double-batched grads: shape before = {batch_grads.shape}")
                        new_batch_size = batch_grads.size(0) * batch_grads.size(1)
                        new_shape = [new_batch_size] + list(batch_grads.shape[2:])
                        batch_grads = batch_grads.reshape(new_shape).float()
                        print(f"Fixed grads shape = {batch_grads.shape}")

                    if batch_normals.dim() > 4:
                        print(f"Fixing double-batched normals: shape before = {batch_normals.shape}")
                        new_batch_size = batch_normals.size(0) * batch_normals.size(1)
                        new_shape = [new_batch_size] + list(batch_normals.shape[2:])
                        batch_normals = batch_normals.reshape(new_shape).float()
                        print(f"Fixed normals shape = {batch_normals.shape}")
                        
                    # Make sure all tensors are in float format
                    inputs = inputs.float()
                    targets = targets.float()
                    batch_grads = batch_grads.float()
                    batch_normals = batch_normals.float()

                    # Verify tensor shapes on first batch of each epoch
                    if i == 0:
                        print(f"Training batch 0 tensor verification:")
                        print(f"  inputs: {inputs.shape}")
                        print(f"  targets: {targets.shape}")
                        print(f"  grads: {batch_grads.shape}")
                        print(f"  normals: {batch_normals.shape}")
                        inputs, targets, batch_grads, batch_normals = verify_batch_tensors(
                            inputs, targets, batch_grads, batch_normals
                        )

                    optimizer.zero_grad(set_to_none=True)
                    # Use Automatic Mixed Precision (AMP) if on CUDA
                    with torch.amp.autocast(
                        device_type=cfg.device.type,
                        dtype=cfg.autocast_dtype,
                        enabled=cfg.use_cuda,
                    ):
                        try:
                            # Run model forward pass
                            outputs, pred_grad, pred_normal = model(inputs)
                            
                            # Ensure consistent dimensions for targets (may have extra channel dim)
                            if targets.dim() == 4 and targets.size(1) == 1:
                                targets_flat = targets.squeeze(1)
                            else:
                                targets_flat = targets
                                
                            # Velocity loss remains the same
                            loss_vel = criterion(outputs, targets)
                            
                            # Check for dimension mismatch in model outputs vs targets
                            if outputs.shape[-1] != targets.shape[-1] or outputs.shape[-2] != targets.shape[-2]:
                                if random.random() < 0.1:  # Only log occasionally
                                    print(f"Dimension mismatch in batch {i}: outputs {outputs.shape} vs targets {targets.shape}")
                                # Resize tensors to match exactly
                                outputs, targets = ensure_tensor_sizes(outputs, targets, expected_size=70)
                                # Recalculate velocity loss with resized tensors
                                loss_vel = criterion(outputs, targets)
                            
                            # Reshape batch_grads if needed (handle any extra dimensions)
                            if batch_grads.dim() > 4:  # If it's 5D: [B, 1, 2, H, W] -> [B, 2, H, W]
                                batch_grads = batch_grads.squeeze(1)
                            
                            # Reshape batch_normals if needed
                            if batch_normals.dim() > 4:  # If it's 5D: [B, 1, 2, H, W] -> [B, 2, H, W]
                                batch_normals = batch_normals.squeeze(1)
                            
                            # Gradient loss (weighted MAE) - ensure shapes match
                            if pred_grad.shape[-2:] != batch_grads.shape[-2:]:
                                # Print detailed shape info for debugging 
                                if random.random() < 0.1:  # Reduce logging
                                    print(f"Shape mismatch in batch {i}: pred_grad {pred_grad.shape} vs batch_grads {batch_grads.shape}")
                                # Resize tensors to match exactly
                                pred_grad, batch_grads = ensure_tensor_sizes(pred_grad, batch_grads, expected_size=70)
                            
                            loss_grad = F.l1_loss(pred_grad, batch_grads)
                            
                            # Normal loss (cosine similarity) - ensure shapes match
                            if pred_normal.shape[-2:] != batch_normals.shape[-2:]:
                                # Print detailed shape info for debugging
                                if random.random() < 0.1:  # Reduce logging
                                    print(f"Shape mismatch in batch {i}: pred_normal {pred_normal.shape} vs batch_normals {batch_normals.shape}")
                                # Resize tensors to match exactly
                                pred_normal, batch_normals = ensure_tensor_sizes(pred_normal, batch_normals, expected_size=70)
                            
                            # Compute normalized dot product for cosine similarity
                            true_normal = batch_normals
                            # Ensure vectors are normalized for proper cosine similarity
                            pred_norm = torch.sqrt(torch.sum(pred_normal**2, dim=1, keepdim=True) + 1e-6)
                            true_norm = torch.sqrt(torch.sum(true_normal**2, dim=1, keepdim=True) + 1e-6)
                            
                            pred_normalized = pred_normal / pred_norm
                            true_normalized = true_normal / true_norm
                            
                            dot_product = (pred_normalized * true_normalized).sum(dim=1)
                            loss_normal = 1 - dot_product.mean()
                            
                            # Combined loss with adaptive weights
                            total_loss = loss_vel + 0.3*loss_grad + 0.2*loss_normal
                        
                        except RuntimeError as e:
                            if "size mismatch" in str(e) or "sizes of tensors must match" in str(e):
                                print(f"\nE: Tensor size mismatch in batch {i}. Details: {e}")
                                print(f"Shapes - inputs: {inputs.shape}, targets: {targets.shape}, "
                                     f"batch_grads: {batch_grads.shape}, batch_normals: {batch_normals.shape}")
                                
                                # Fall back to just velocity loss for this batch
                                outputs, _, _ = model(inputs)
                                total_loss = criterion(outputs, targets)
                                print("Falling back to velocity loss only for this batch.")
                            else:
                                # Re-raise other RuntimeErrors
                                raise e

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    train_losses.append(loss_vel.item())  # Track main velocity loss for history

                    # Garbage collect every few batches
                    if i % 5 == 0:  # Every 5 batches
                        gc.collect()
                        if cfg.use_cuda:
                            torch.cuda.empty_cache()

                    # Update progress bar description
                    if i % 100 == 0:
                        pbar_train.set_postfix(loss=f"{np.mean(train_losses):.5f}")

                except Exception as e:
                    print(f"\nE: Training batch {i} failed: {e}")
                    # Stop training on OOM error
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        print("E: CUDA Out of Memory during training. Exiting.")
                        raise e
                    # Continue on other errors if desired, or raise
                    # raise e # Uncomment to stop on any training error

            avg_train_loss = np.mean(train_losses) if train_losses else 0.0
            print(f"Epoch {epoch} Avg Train Loss: {avg_train_loss:.5f}")

            # --- Validation Phase ---
            if dlvalid is None:
                print("W: Skipping validation phase - no validation DataLoader.")
                history.append(
                    {"epoch": epoch, "train_loss": avg_train_loss, "valid_loss": None}
                )
                continue  # Skip to next epoch

            # Run garbage collection before validation
            gc.collect()
            if cfg.use_cuda:
                torch.cuda.empty_cache()

            model.eval()
            val_losses = []
            pbar_val = tqdm(dlvalid, desc=f"Valid E{epoch}", leave=False, unit="batch")
            with torch.no_grad():
                for i, batch in enumerate(pbar_val):
                    if not batch or "seis" not in batch or "vel" not in batch:
                        print(f"W: Skipping invalid validation batch {i}")
                        continue
                    try:
                        # Check if batch is a valid dictionary with required fields
                        if not isinstance(batch, dict):
                            print(f"W: Skipping invalid validation batch {i} (type: {type(batch).__name__})")
                            continue
                            
                        if "seis" not in batch or "vel" not in batch or "grad" not in batch or "normal" not in batch:
                            print(f"W: Skipping incomplete validation batch {i} (missing keys)")
                            continue
                            
                        # Access batch dictionary elements and move them to device
                        inputs = batch["seis"].to(cfg.device, non_blocking=True)
                        targets = batch["vel"].to(cfg.device, non_blocking=True)
                        batch_grads = batch["grad"].to(cfg.device)
                        batch_normals = batch["normal"].to(cfg.device)

                        # Fix double batching issue - detect and reshape tensor if it has extra dimensions
                        # Error example: [8, 8, 5, 1000, 70] - has an extra batch dimension
                        if inputs.dim() > 4:
                            print(f"Fixing double-batched inputs: shape before = {inputs.shape}")
                            # Calculate new shape: flatten the first two dimensions if they're both batch dimensions
                            new_batch_size = inputs.size(0) * inputs.size(1)
                            new_shape = [new_batch_size] + list(inputs.shape[2:])
                            inputs = inputs.reshape(new_shape).float()
                            print(f"Fixed inputs shape = {inputs.shape}")

                        # Similarly fix other tensors if they have extra dimensions
                        if targets.dim() > 4:
                            print(f"Fixing double-batched targets: shape before = {targets.shape}")
                            new_batch_size = targets.size(0) * targets.size(1)
                            new_shape = [new_batch_size] + list(targets.shape[2:])
                            targets = targets.reshape(new_shape).float()
                            print(f"Fixed targets shape = {targets.shape}")

                        if batch_grads.dim() > 4:
                            print(f"Fixing double-batched grads: shape before = {batch_grads.shape}")
                            new_batch_size = batch_grads.size(0) * batch_grads.size(1)
                            new_shape = [new_batch_size] + list(batch_grads.shape[2:])
                            batch_grads = batch_grads.reshape(new_shape).float()
                            print(f"Fixed grads shape = {batch_grads.shape}")

                        if batch_normals.dim() > 4:
                            print(f"Fixing double-batched normals: shape before = {batch_normals.shape}")
                            new_batch_size = batch_normals.size(0) * batch_normals.size(1)
                            new_shape = [new_batch_size] + list(batch_normals.shape[2:])
                            batch_normals = batch_normals.reshape(new_shape).float()
                            print(f"Fixed normals shape = {batch_normals.shape}")
                            
                        # Make sure all tensors are in float format
                        inputs = inputs.float()
                        targets = targets.float()
                        batch_grads = batch_grads.float()
                        batch_normals = batch_normals.float()

                        # Verify tensor shapes on first batch of validation
                        if i == 0:
                            print(f"Validation batch 0 tensor verification:")
                            print(f"  inputs: {inputs.shape}")
                            print(f"  targets: {targets.shape}")
                            print(f"  grads: {batch_grads.shape}")
                            print(f"  normals: {batch_normals.shape}")
                            inputs, targets, batch_grads, batch_normals = verify_batch_tensors(
                                inputs, targets, batch_grads, batch_normals
                            )

                        with torch.amp.autocast(
                            device_type=cfg.device.type,
                            dtype=cfg.autocast_dtype,
                            enabled=cfg.use_cuda,
                        ):
                            try:
                                # Run model forward pass
                                outputs, pred_grad, pred_normal = model(inputs)
                                
                                # Ensure consistent dimensions for targets (may have extra channel dim)
                                if targets.dim() == 4 and targets.size(1) == 1:
                                    targets_flat = targets.squeeze(1)
                                else:
                                    targets_flat = targets
                                    
                                # Velocity loss remains the same
                                loss_vel = criterion(outputs, targets)
                                
                                # Check for dimension mismatch in model outputs vs targets
                                if outputs.shape[-1] != targets.shape[-1] or outputs.shape[-2] != targets.shape[-2]:
                                    if random.random() < 0.1:  # Reduce logging
                                        print(f"Dimension mismatch in batch {i}: outputs {outputs.shape} vs targets {targets.shape}")
                                    # Resize tensors to match exactly
                                    outputs, targets = ensure_tensor_sizes(outputs, targets, expected_size=70)
                                    # Recalculate velocity loss with resized tensors
                                    loss_vel = criterion(outputs, targets)
                                
                                # Reshape batch_grads if needed (handle any extra dimensions)
                                if batch_grads.dim() > 4:  # If it's 5D: [B, 1, 2, H, W] -> [B, 2, H, W]
                                    batch_grads = batch_grads.squeeze(1)
                                
                                # Reshape batch_normals if needed
                                if batch_normals.dim() > 4:  # If it's 5D: [B, 1, 2, H, W] -> [B, 2, H, W]
                                    batch_normals = batch_normals.squeeze(1)
                                
                                # Gradient loss (weighted MAE) - ensure shapes match
                                if pred_grad.shape[-2:] != batch_grads.shape[-2:]:
                                    # Print detailed shape info for debugging 
                                    if random.random() < 0.1:  # Reduce logging
                                        print(f"Shape mismatch in batch {i}: pred_grad {pred_grad.shape} vs batch_grads {batch_grads.shape}")
                                    # Resize tensors to match exactly
                                    pred_grad, batch_grads = ensure_tensor_sizes(pred_grad, batch_grads, expected_size=70)
                                
                                loss_grad = F.l1_loss(pred_grad, batch_grads)
                                
                                # Normal loss (cosine similarity) - ensure shapes match
                                if pred_normal.shape[-2:] != batch_normals.shape[-2:]:
                                    # Print detailed shape info for debugging
                                    if random.random() < 0.1:  # Reduce logging
                                        print(f"Shape mismatch in batch {i}: pred_normal {pred_normal.shape} vs batch_normals {batch_normals.shape}")
                                    # Resize tensors to match exactly
                                    pred_normal, batch_normals = ensure_tensor_sizes(pred_normal, batch_normals, expected_size=70)
                                
                                # Compute normalized dot product for cosine similarity
                                true_normal = batch_normals
                                # Ensure vectors are normalized for proper cosine similarity
                                pred_norm = torch.sqrt(torch.sum(pred_normal**2, dim=1, keepdim=True) + 1e-6)
                                true_norm = torch.sqrt(torch.sum(true_normal**2, dim=1, keepdim=True) + 1e-6)
                                
                                pred_normalized = pred_normal / pred_norm
                                true_normalized = true_normal / true_norm
                                
                                dot_product = (pred_normalized * true_normalized).sum(dim=1)
                                loss_normal = 1 - dot_product.mean()
                                
                                # Combined loss with same weights as training
                                loss = loss_vel + 0.3*loss_grad + 0.2*loss_normal
                            
                            except RuntimeError as e:
                                if "size mismatch" in str(e) or "sizes of tensors must match" in str(e):
                                    print(f"\nE: Tensor size mismatch in validation batch {i}. Details: {e}")
                                    print(f"Shapes - inputs: {inputs.shape}, targets: {targets.shape}, "
                                        f"batch_grads: {batch_grads.shape}, batch_normals: {batch_normals.shape}")
                                    
                                    # Fall back to just velocity loss for this batch
                                    outputs, _, _ = model(inputs)
                                    loss = criterion(outputs, targets)
                                    print("Falling back to velocity loss only for this validation batch.")
                                else:
                                    # Re-raise other RuntimeErrors
                                    raise e

                        val_losses.append(loss_vel.item())  # Track main velocity loss for history
                        
                        # Garbage collect periodically during validation
                        if i % 5 == 0:  # Every 5 batches
                            gc.collect()
                            if cfg.use_cuda:
                                torch.cuda.empty_cache()

                        # Plotting validation examples periodically
                        if i == 0 and epoch % cfg.plot_every_n_epochs == 0:
                            # Add validation plotting code here if desired
                            pass # Placeholder

                    except Exception as e:
                        print(f"\nE: Validation batch {i} failed: {e}")
                        if isinstance(e, torch.cuda.OutOfMemoryError):
                            print("E: CUDA Out of Memory during validation. Exiting.")
                            raise e
                        # Continue on other errors if desired, or raise
                        # raise e # Uncomment to stop on any validation error

            avg_val_loss = np.mean(val_losses) if val_losses else float("inf")
            print(f"Epoch {epoch} Avg Valid Loss: {avg_val_loss:.5f}")
            history.append(
                {"epoch": epoch, "train_loss": avg_train_loss, "valid_loss": avg_val_loss}
            )

            # --- Save Best Model ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Clean previous best models before saving new one
                del_pattern = os.path.join(
                    cfg.working_dir, f"unet_best_model_epoch_*_loss_*.pth"
                )
                for old_model_path in glob.glob(del_pattern):
                    try:
                        print(f"    Removing old best model: {os.path.basename(old_model_path)}")
                        os.remove(old_model_path)
                    except OSError as e:
                        print(f"W: Could not delete old model {old_model_path}: {e}")

                # Save the new best model
                fname = f"unet_best_model_epoch_{epoch}_loss_{best_val_loss:.4f}.pth"
                fpath = os.path.join(cfg.working_dir, fname)
                print(f"*** New best validation loss: {best_val_loss:.5f}. Saving model: {fname} ***")
                torch.save(model.state_dict(), fpath)

    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\nE: Training loop encountered a critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Training Loop Finished ---")


# --- 5. Plot History ---
print("\n--- 5. Plotting Training History ---")
if history:
    try:
        hist_df = pd.DataFrame(history)
        plt.figure(figsize=(12, 6))
        plt.plot(hist_df["epoch"], hist_df["train_loss"], "o-", label="Train Loss")
        # Only plot validation loss if it exists and is not all None/NaN
        if "valid_loss" in hist_df.columns and not hist_df["valid_loss"].isnull().all():
            plt.plot(
                hist_df["epoch"],
                hist_df["valid_loss"],
                "s--",  # Square markers, dashed line
                label="Validation Loss",
            )
        plt.title("Training and Validation Loss vs. Epoch (Residual U-Net)")
        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss (Mean Absolute Error)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim(bottom=0)  # Loss should not be negative
        plt.tight_layout()
        plot_fname = os.path.join(cfg.working_dir, "training_history.png")
        plt.savefig(plot_fname)
        print(f"Saved history plot: {plot_fname}")
        plt.show()  # Display the plot
    except Exception as e:
        print(f"E: Failed plotting training history: {e}")
else:
    print("No training history recorded to plot.")


# --- 6. Error Analysis on Validation Set ---
print("\n--- 6. Error Analysis on Validation Set ---")
# Placeholder for error analysis code - requires dlvalid or val_paths_saved
best_model_path_analysis = find_best_model()
if not best_model_path_analysis:
    print("W: No best model found. Skipping analysis.")
elif dlvalid is None and not val_paths_saved:
    # Need either the original loader or the paths to recreate it
    print("W: Validation loader/paths unavailable. Skipping analysis.")
else:
    print(f"Performing analysis using model: {os.path.basename(best_model_path_analysis)}")
    # Add analysis code block here (e.g., load model, get batch, predict, plot errors)
    # Ensure to handle potential recreation of dlvalid if it was lost
    # Example: Load model, get a batch from dlvalid, predict, compare pred/target
    # model_analysis = UNet().to(cfg.device)
    # model_analysis.load_state_dict(torch.load(best_model_path_analysis, map_location=cfg.device))
    # model_analysis.eval()
    # with torch.no_grad():
    #     # Get a batch (handle if dlvalid needs recreation from val_paths_saved)
    #     # batch = next(iter(dlvalid_or_recreated))
    #     # inputs = batch["seis"].to(cfg.device)... targets = batch["vel"]...
    #     # preds = model_analysis(inputs)
    #     # Plot difference, calculate stats, etc.
    pass


# --- 7. Prediction (Using Kaggle Test Set) ---
print("\n--- 7. Final Prediction on Kaggle Test Set ---")
best_model_final_path = find_best_model()
if not best_model_final_path:
    print("W: No best model found. Skipping final prediction.")
elif not Path(cfg.kaggle_test_dir).is_dir():
    print(f"W: Kaggle test directory '{cfg.kaggle_test_dir}' not found. Skipping prediction.")
else:
    try:
        print(f"Loading model for final prediction: {os.path.basename(best_model_final_path)}")
        # Make sure to instantiate the correct model class (UNet)
        model_pred = UNet().to(cfg.device)
        model_pred.load_state_dict(torch.load(best_model_final_path, map_location=cfg.device))
        model_pred.eval()

        test_ds = KaggleTestDataset(cfg.kaggle_test_dir)
        if len(test_ds) == 0:
            print("W: Kaggle test dataset is empty. No submission generated.")
        else:
            # Setup DataLoader for test set
            t_bs = max(1, cfg.batch_size // 2)
            t_nw = min(
                max(0, cfg.num_workers // 2),
                (os.cpu_count() // 2 if os.cpu_count() else 1),
            )
            dl_test = DataLoader(
                test_ds,
                batch_size=t_bs,
                shuffle=False,
                num_workers=t_nw,
                pin_memory=cfg.use_cuda,
            )
            print(f"Test DataLoader created with bs={t_bs}, workers={t_nw}")
            print(f"Writing submission file to: {cfg.submission_file}")

            rows_written = 0
            with open(cfg.submission_file, "wt", newline="") as csvfile:
                # Define CSV header columns (x_1, x_3, ..., x_69)
                x_cols = [f"x_{i}" for i in range(1, 70, 2)]
                fieldnames = ["oid_ypos"] + x_cols
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                pbar_test = tqdm(dl_test, desc="Generating Submission", unit="batch")
                with torch.no_grad():
                    for inputs, original_ids in pbar_test:
                        # Handle batch size = 1 where original_ids might be a string
                        if isinstance(original_ids, str):
                            original_ids = [original_ids]
                        try:
                            inputs = inputs.to(cfg.device).float()
                            with torch.amp.autocast(
                                device_type=cfg.device.type,
                                dtype=cfg.autocast_dtype,
                                enabled=cfg.use_cuda,
                            ):
                                # Get velocity prediction (discard gradients/normals for submission)
                                velocity, _, _ = model_pred(inputs)
                                
                            # Output shape is (B, 1, H, W), get predictions (B, H, W)
                            if velocity.dim() == 4 and velocity.size(1) == 1:
                                preds = velocity[:, 0].cpu().numpy()
                            else:
                                preds = velocity.cpu().numpy()

                            # Iterate through samples in the batch
                            for y_pred, oid in zip(preds, original_ids): # y_pred is (H, W)
                                # Iterate through y-positions (rows) for this sample
                                for y_pos in range(y_pred.shape[0]): # y_pred.shape[0] should be 70
                                    # Extract values at odd x-indices (1, 3, ..., 69)
                                    vals = y_pred[y_pos, 1::2].astype(np.float32)
                                    # Create row dictionary
                                    row = dict(zip(x_cols, vals))
                                    row["oid_ypos"] = f"{oid}_y_{y_pos}"
                                    writer.writerow(row)
                                    rows_written += 1
                        except Exception as e:
                            # Report error but continue if possible
                            print(
                                f"\nE: Prediction failed for batch (OID: {original_ids[0] if original_ids else '?'}) : {e}"
                            )

            print(f"Submission file created: {cfg.submission_file} ({rows_written} rows).")
            # Sanity check row count
            expected_rows = len(test_ds) * 70  # 70 y-positions per test sample
            if rows_written != expected_rows:
                print(
                    f"W: Row count mismatch! Expected {expected_rows}, but wrote {rows_written}."
                )

    except Exception as e:
        print(f"E: Final prediction process failed critically: {e}")
        import traceback
        traceback.print_exc()

print("\n--- Full Workflow Finished ---")

