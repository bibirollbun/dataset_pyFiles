# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# -*- coding: utf-8 -*-
"""
Kaggle Notebook: BYU Flagellar Motor Localization (Revised for Directory Structure & Multi-Motor)

Goal: Detect the presence and (x, y, z) coordinates of bacterial
      flagellar motors in 3D cryo-ET tomograms stored as slice directories.

Inspired Concepts: Analogies from Non-Linear Electrodynamics / String Theory
    - Tomogram Intensity: Scalar field (like potential)
    - Gradient (∇I): Vector field (analogous to Electric Field E)
    - Structure Tensor (∇I ⊗ ∇I averaged): Symmetric Tensor (analogous to Stress Tensor T_µν or Metric Perturbation S_µν)
    - Hessian (∇²I): Tensor capturing curvature (related to field changes)
    - Invariants: Rotationally invariant features derived from gradient/structure tensor/Hessian (analogous to Lorentz invariants x, y)
    - Anisotropic Diffusion/Filtering: Guided propagation of information (analogous to wave propagation influenced by background fields/metrics)
    - Multi-Metric Idea: Using different analysis scales/features (like g_µν vs G_µν) - e.g., global search vs. local refinement.

MODIFICATION: This version SKIPS training data generation and model training entirely.
            It focuses prediction ONLY on specified IDs ['tomo_003acc', 'tomo_00e047', 'tomo_00e463']
            using NO trained model (will predict NO_MOTOR_COORD).
            The final output is a submission file template for ALL test IDs found.
"""

# %% [markdown]
# # 1. Setup and Imports
#
# Load necessary libraries and define constants.

# %%
# Attempt installation only if needed (e.g., in Kaggle environment)
try:
    import cv2
    import mrcfile # Though likely unused for loading now
    print("Libraries opencv-python, mrcfile seem available.")
except ImportError:
    print("Installing required libraries: opencv-python, mrcfile")
    # Use %pip instead of !pip in notebooks for better integration
    %pip install opencv-python mrcfile --quiet
    import cv2
    import mrcfile
    print("Libraries installed.")

import os
import glob
import numpy as np
import pandas as pd
import cv2 # Import OpenCV
# import mrcfile # Keep if other parts might use it, but not for loading slices
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as ndi
from skimage.feature import structure_tensor, structure_tensor_eigenvalues, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import gaussian, median
from skimage.measure import regionprops, label
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import fbeta_score, mean_squared_error, make_scorer
import gc # Garbage collection
import time # For timing

# Optional: Deep Learning Libraries (if used)
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import monai # Medical Imaging AI library

# Constants
COMPETITION_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
TRAIN_DIR = os.path.join(COMPETITION_DIR, 'train')
TEST_DIR = os.path.join(COMPETITION_DIR, 'test')
TRAIN_LABELS_PATH = os.path.join(COMPETITION_DIR, 'train_labels.csv')
SAMPLE_SUB_PATH = os.path.join(COMPETITION_DIR, 'sample_submission.csv')

# Default Voxel spacing (used for test set or if lookup fails)
DEFAULT_VOXEL_SPACING_A = 10.0
print(f"Using default voxel spacing: {DEFAULT_VOXEL_SPACING_A} Angstroms/pixel (will try to read from labels for training)")

# Evaluation Threshold (Angstroms)
DISTANCE_THRESHOLD_A = 1000.0
# DISTANCE_THRESHOLD_PX will be calculated dynamically based on actual voxel spacing

# F-beta Score Beta value
F_BETA = 2.0

# For submission: Coordinate value indicating no motor found
NO_MOTOR_COORD = -1.0 # Ensure float

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)

# %% [markdown]
# # 2. Load Data and Metadata
#
# Load the training labels and find the data directories. Check for consistency.

# %%
start_time = time.time()
print("Loading labels...")
try:
    train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Training labels shape: {train_labels_df.shape}")
    print(train_labels_df.head())
except FileNotFoundError:
    print(f"ERROR: Training labels file not found at {TRAIN_LABELS_PATH}")
    train_labels_df = pd.DataFrame() # Empty dataframe

# --- Debug: List directories to confirm paths ---
print("\n--- Directory Listing ---")
print(f"Listing {COMPETITION_DIR}:")
try:
    print(os.listdir(COMPETITION_DIR))
except FileNotFoundError:
    print(f"  Error: Directory not found: {COMPETITION_DIR}")

print(f"\nListing {TRAIN_DIR}:")
try:
    train_contents = os.listdir(TRAIN_DIR)
    print(f"  Found {len(train_contents)} items in train dir. First 10:")
    print(train_contents[:10])
    if len(train_contents) > 10: print("  ...")
except FileNotFoundError:
    print(f"  Error: Directory not found: {TRAIN_DIR}")
    train_contents = []

print(f"\nListing {TEST_DIR}:")
try:
    test_contents = os.listdir(TEST_DIR)
    print(f"  Found {len(test_contents)} items in test dir. First 10:")
    print(test_contents[:10])
    if len(test_contents) > 10: print("  ...")
except FileNotFoundError:
    print(f"  Error: Directory not found: {TEST_DIR}")
    test_contents = []
print("--- End Directory Listing ---\n")


# --- Find Tomogram Directories ---
print("Finding tomogram directories...")
train_dirs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'tomo_*')))
test_dirs = sorted(glob.glob(os.path.join(TEST_DIR, 'tomo_*')))

# Extract UNIQUE tomo_ids present in the directories found
train_ids_found = sorted([os.path.basename(d) for d in train_dirs if os.path.isdir(d)]) # Ensure it's a directory
test_ids_found = sorted([os.path.basename(d) for d in test_dirs if os.path.isdir(d)]) # Ensure it's a directory

print(f"Found {len(train_ids_found)} potential training tomogram directories.")
print(f"First 5 found: {train_ids_found[:5]}")
print(f"Found {len(test_ids_found)} potential testing tomogram directories.")
print(f"First 5 found: {test_ids_found[:5]}")

# --- Filter Labels based on Found Directories ---
if not train_labels_df.empty:
    # Check if 'tomo_id' column exists
    if 'tomo_id' not in train_labels_df.columns:
        print("ERROR: 'tomo_id' column not found in labels CSV. Cannot proceed.")
        train_labels_df_filtered = pd.DataFrame()
        unique_train_ids_with_data = []
        voxel_spacing_map = {}
    else:
        train_labels_df_filtered = train_labels_df[train_labels_df['tomo_id'].isin(train_ids_found)].copy()
        print(f"\nOriginal label rows: {len(train_labels_df)}")
        print(f"Label rows after filtering by found train directories: {len(train_labels_df_filtered)}")

        # Get unique tomogram IDs that have both labels AND a found directory
        unique_train_ids_with_data = sorted(train_labels_df_filtered['tomo_id'].unique())
        print(f"Number of unique train tomograms with labels AND data: {len(unique_train_ids_with_data)}")

        # Create map for Voxel Spacing from labels (use first entry per tomo_id)
        if 'Voxel spacing' in train_labels_df_filtered.columns and not train_labels_df_filtered.empty:
            voxel_spacing_map = train_labels_df_filtered.drop_duplicates(subset='tomo_id').set_index('tomo_id')['Voxel spacing'].to_dict()
            print(f"Example voxel spacing: {list(voxel_spacing_map.items())[:5]}")
        else:
             print("WARNING: 'Voxel spacing' column not found or no valid labels. Cannot create voxel spacing map.")
             voxel_spacing_map = {}

else:
    print("WARNING: Training labels dataframe is empty. Cannot filter or create maps.")
    train_labels_df_filtered = pd.DataFrame()
    unique_train_ids_with_data = []
    voxel_spacing_map = {}

# Map tomo_id to DIRECTORY path (even if labels are missing)
train_dir_map = {os.path.basename(d): d for d in train_dirs if os.path.isdir(d)}
test_dir_map = {os.path.basename(d): d for d in test_dirs if os.path.isdir(d)}


print(f"Data loading setup took {time.time() - start_time:.2f} seconds.")

if not unique_train_ids_with_data and not train_labels_df.empty:
    print("\n\nCRITICAL WARNING: No intersection between labels and found training directories. Training cannot proceed.")
elif not train_ids_found and not test_ids_found:
     print("\n\nCRITICAL WARNING: No training or testing directories found. Check data paths.")

# %% [markdown]
# # 3. Exploratory Data Analysis (EDA) & Concept Visualization
#
# Understand the data distribution, visualize tomograms and labels, and see how our physics-inspired concepts manifest.
# (Keeping these utility functions defined, even if visualization is partially skipped)

# %%
# --- Tomogram Loading Function ---
def load_tomogram(tomo_id, dir_map):
    """Loads a tomogram by reading and stacking slices from a directory."""
    if isinstance(dir_map, str):
        directory_path = dir_map
        if not os.path.isdir(directory_path): return None
    else:
        directory_path = dir_map.get(tomo_id)
        if not directory_path or not os.path.isdir(directory_path): return None

    slice_files = sorted(glob.glob(os.path.join(directory_path, 'slice_*.jpg')))
    if not slice_files: slice_files = sorted(glob.glob(os.path.join(directory_path, 'slice_*.png')))
    if not slice_files: slice_files = sorted(glob.glob(os.path.join(directory_path, 'slice_*.tif')))
    if not slice_files: return None

    def get_slice_number(filepath):
        try:
            filename = os.path.basename(filepath)
            num_str = filename.split('slice_')[1].split('.')[0]
            return int(num_str)
        except: return -1

    valid_slice_files = [(f, get_slice_number(f)) for f in slice_files]
    valid_slice_files = [(f, num) for f, num in valid_slice_files if num != -1]
    if not valid_slice_files: return None
    valid_slice_files.sort(key=lambda item: item[1])
    slice_files_sorted = [f for f, num in valid_slice_files]

    try:
        first_slice = cv2.imread(slice_files_sorted[0], cv2.IMREAD_GRAYSCALE)
        if first_slice is None: raise IOError(f"cv2.imread failed for {slice_files_sorted[0]}")
        height, width = first_slice.shape
        num_slices = len(slice_files_sorted)
        dtype = first_slice.dtype
    except Exception as e:
        print(f"Error reading first slice {slice_files_sorted[0]}: {e}"); return None

    tomogram_data = np.zeros((num_slices, height, width), dtype=dtype)
    tomogram_data[0, :, :] = first_slice

    for i in range(1, num_slices):
        try:
            slice_img = cv2.imread(slice_files_sorted[i], cv2.IMREAD_GRAYSCALE)
            if slice_img is None: raise IOError(f"cv2.imread failed for {slice_files_sorted[i]}")
            if slice_img.shape != (height, width): return None
            tomogram_data[i, :, :] = slice_img
        except Exception as e:
            print(f"Error reading slice {slice_files_sorted[i]}: {e}"); return None
    return tomogram_data

# --- Plotting Function ---
def plot_slices_with_label(tomo_data, voxel_spacing, label_coords_A=None, title="Tomogram Slices"):
    """Plots central slices along each axis, optionally marking the label(s)."""
    if tomo_data is None: print("No data to plot."); return
    if voxel_spacing <= 0: print("Invalid voxel spacing."); return
    shape = tomo_data.shape
    if not (len(shape) == 3 and all(s > 0 for s in shape)): print(f"Invalid shape {shape}."); return

    center_z, center_y, center_x = shape[0] // 2, shape[1] // 2, shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); fig.suptitle(title, fontsize=16)
    try:
        axes[0].imshow(tomo_data[center_z, :, :], cmap='gray'); axes[0].set_title(f'Z Slice (Z={center_z})')
        axes[1].imshow(tomo_data[:, center_y, :], cmap='gray', aspect=shape[0]/shape[2] if shape[2]>0 else 1); axes[1].set_title(f'Y Slice (Y={center_y})')
        axes[2].imshow(tomo_data[:, :, center_x], cmap='gray', aspect=shape[0]/shape[1] if shape[1]>0 else 1); axes[2].set_title(f'X Slice (X={center_x})')
    except IndexError as e: print(f"Error plotting slices: {e}"); plt.close(fig); return

    if label_coords_A is not None and len(label_coords_A) > 0:
        is_multiple = isinstance(label_coords_A[0], (list, np.ndarray))
        if not is_multiple: label_coords_A = [label_coords_A]
        plotted_legend = False
        for i, motor_A in enumerate(label_coords_A):
            if len(motor_A) < 3 or motor_A[0] == NO_MOTOR_COORD: continue
            lz_A, ly_A, lx_A = motor_A
            lz_px, ly_px, lx_px = int(lz_A / voxel_spacing), int(ly_A / voxel_spacing), int(lx_A / voxel_spacing)
            if not (0 <= lz_px < shape[0] and 0 <= ly_px < shape[1] and 0 <= lx_px < shape[2]): continue
            label_text = f'Motor {i+1}' if len(label_coords_A) > 1 else 'Motor'
            z_thresh=max(5, shape[0]*0.05); y_thresh=max(5, shape[1]*0.05); x_thresh=max(5, shape[2]*0.05)
            try:
                lbl = None
                if not plotted_legend: lbl = label_text; plotted_legend=True
                if abs(lz_px - center_z) < z_thresh: axes[0].plot(lx_px, ly_px, 'ro', ms=8, label=lbl)
                if abs(ly_px - center_y) < y_thresh: axes[1].plot(lx_px, lz_px, 'ro', ms=8)
                if abs(lx_px - center_x) < x_thresh: axes[2].plot(ly_px, lz_px, 'ro', ms=8)
                if lbl: lbl=None
            except Exception as plot_e: print(f"Error plotting marker: {plot_e}")
    if plotted_legend: axes[0].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

# --- Visualize a sample tomogram ---
# (Keep this visualization, it's quick and useful)
if unique_train_ids_with_data:
    sample_tomo_id = unique_train_ids_with_data[0]
    print(f"\nVisualizing sample tomogram: {sample_tomo_id}")
    sample_data = load_tomogram(sample_tomo_id, train_dir_map)
    sample_labels = train_labels_df_filtered[train_labels_df_filtered['tomo_id'] == sample_tomo_id]
    if sample_data is not None and not sample_labels.empty:
        motor_locations_A = sample_labels[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values.tolist()
        voxel_spacing = sample_labels.iloc[0]['Voxel spacing']
        print(f"Sample Tomogram Shape: {sample_data.shape}, Voxel Spacing: {voxel_spacing} A/px")
        print(f"Motor Location(s) (Angstroms): {motor_locations_A}")
        plot_slices_with_label(sample_data, voxel_spacing, motor_locations_A, title=f"{sample_tomo_id}")
        plt.figure(figsize=(10, 4)); plt.hist(sample_data.flatten(), bins=100, color='blue', alpha=0.7); plt.title(f'Intensity Distribution {sample_tomo_id}'); plt.xlabel('Intensity'); plt.ylabel('Frequency'); plt.grid(True, alpha=0.3); plt.show()
        del sample_data, sample_labels; gc.collect()
    else: print(f"Failed to load/find labels for sample {sample_tomo_id}")
else: print("\nSkipping Visualization: No valid training tomograms found.")

# %% [markdown]
# ### 3.1 Visualizing Physics-Inspired Concepts
# (Skip this visualization as it relies on patch extraction and calculations that mirror training data gen)

# %%
# --- Patch Extraction Function (Keep definition for prediction step) ---
def get_local_patch(tomo_data, center_px, patch_size_px=64):
    """Extracts a 3D patch centered at center_px. Patch size is in pixels."""
    if tomo_data is None or center_px is None: return None, None
    center_px = np.round(center_px).astype(int); z, y, x = center_px
    shape = tomo_data.shape
    if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]): return None, None
    half_size = patch_size_px // 2
    z_start, z_end = max(0, z - half_size), min(shape[0], z + half_size)
    y_start, y_end = max(0, y - half_size), min(shape[1], y + half_size)
    x_start, x_end = max(0, x - half_size), min(shape[2], x + half_size)
    dest_z_start = half_size - (z - z_start); dest_z_end = dest_z_start + (z_end - z_start)
    dest_y_start = half_size - (y - y_start); dest_y_end = dest_y_start + (y_end - y_start)
    dest_x_start = half_size - (x - x_start); dest_x_end = dest_x_start + (x_end - x_start)
    patch = np.zeros((patch_size_px, patch_size_px, patch_size_px), dtype=tomo_data.dtype)
    try:
        src_slice = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
        dest_slice = (slice(dest_z_start, dest_z_end), slice(dest_y_start, dest_y_end), slice(dest_x_start, dest_x_end))
        patch[dest_slice] = tomo_data[src_slice]
        patch_center_coords_in_patch = np.round([z - z_start + dest_z_start, y - y_start + dest_y_start, x - x_start + dest_x_start]).astype(int)
        if not (0 <= patch_center_coords_in_patch[0] < patch_size_px and 0 <= patch_center_coords_in_patch[1] < patch_size_px and 0 <= patch_center_coords_in_patch[2] < patch_size_px):
             patch_center_coords_in_patch = [patch_size_px // 2] * 3
        return patch, patch_center_coords_in_patch
    except (ValueError, IndexError) as e: return None, None

# --- Visualize physics concepts ---
print("\nSkipping Physics Concept Visualization (requires patch processing).")
# (The code for visualization is removed here)

# %% [markdown]
# # 4. Data Preprocessing & Feature Engineering
# (Keep function definitions as they are needed for prediction, even if models aren't trained)

# %%
# --- Simple Preprocessing ---
def preprocess_tomogram(data):
    """Normalizes and smooths the tomogram data."""
    if data is None: return None
    if not np.issubdtype(data.dtype, np.floating): data = data.astype(np.float32)
    mean, std = np.mean(data), np.std(data)
    if std > 1e-6: normalized_data = (data - mean) / std
    else: normalized_data = data - mean
    smoothed_data = gaussian(normalized_data, sigma=1.5, mode='reflect', preserve_range=True, truncate=4.0)
    return smoothed_data

# --- Feature Extraction (Keep definition) ---
def extract_features_patch(patch):
    """Extracts a feature vector from a 3D patch."""
    if patch is None or patch.size == 0: return None
    if not np.issubdtype(patch.dtype, np.floating): patch = patch.astype(np.float32)
    features = []
    EXPECTED_N_FEATURES = 15
    try:
        features.extend([np.mean(patch), np.std(patch), np.median(patch), np.min(patch), np.max(patch)])
        patch_smooth_grad = gaussian(patch, sigma=1.0, mode='reflect', preserve_range=True, truncate=4.0)
        patch_smooth_st = gaussian(patch, sigma=1.5, mode='reflect', preserve_range=True, truncate=4.0)
        patch_smooth_hess = gaussian(patch, sigma=2.0, mode='reflect', preserve_range=True, truncate=4.0)
        try:
            grad_z, grad_y, grad_x = np.gradient(patch_smooth_grad)
            grad_mag = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
            features.extend([np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag)])
        except Exception: features.extend([0.0] * 3)
        try:
            shape = patch_smooth_st.shape; slice_start = max(0, shape[0]//5); slice_end = max(slice_start+1, 4*shape[0]//5)
            center_slice = slice(slice_start, slice_end)
            if not (0<=center_slice.start<shape[0] and 0<center_slice.stop<=shape[0] and center_slice.start<center_slice.stop): center_patch_smooth = patch_smooth_st
            else: center_patch_smooth = patch_smooth_st[center_slice, center_slice, center_slice]
            if center_patch_smooth.size > 0:
                S_elems_c = structure_tensor(center_patch_smooth, sigma=1.5, mode='reflect')
                eigvals_S_c = structure_tensor_eigenvalues(S_elems_c); eigvals_S_c = np.sort(eigvals_S_c, axis=0)
                l3, l2, l1 = eigvals_S_c[0], eigvals_S_c[1], eigvals_S_c[2]
                den = l1 + l2 + l3 + 1e-9; coh = np.where(den > 1e-8, (l1 - l3) / den, 0.0)
                features.extend([np.mean(l1), np.mean(l3), np.mean(coh), np.max(coh)])
            else: features.extend([0.0] * 4)
        except Exception: features.extend([0.0] * 4)
        try:
            shape_h = patch_smooth_hess.shape; slice_start_h = max(0, shape_h[0]//5); slice_end_h = max(slice_start_h+1, 4*shape_h[0]//5)
            center_slice_h = slice(slice_start_h, slice_end_h)
            if not (0<=center_slice_h.start<shape_h[0] and 0<center_slice_h.stop<=shape_h[0] and center_slice_h.start<center_slice_h.stop): center_patch_smooth_h = patch_smooth_hess
            else: center_patch_smooth_h = patch_smooth_hess[center_slice_h, center_slice_h, center_slice_h]
            if center_patch_smooth_h.size > 0:
                h_matrix = hessian_matrix(center_patch_smooth_h, sigma=2.0, mode='reflect', use_gaussian_derivatives=False)
                eigvals_H_c = hessian_matrix_eigvals(h_matrix); eigvals_H_c = np.sort(eigvals_H_c, axis=0)
                h1, h2, h3 = eigvals_H_c[0], eigvals_H_c[1], eigvals_H_c[2]
                features.extend([np.mean(h1), np.mean(h3), np.std(h1)])
            else: features.extend([0.0] * 3)
        except Exception: features.extend([0.0] * 3)
    except Exception as outer_e: print(f"Outer FE error: {outer_e}"); return None
    if len(features) != EXPECTED_N_FEATURES:
        if len(features) < EXPECTED_N_FEATURES: features.extend([0.0] * (EXPECTED_N_FEATURES - len(features)))
        else: features = features[:EXPECTED_N_FEATURES]
        if len(features) != EXPECTED_N_FEATURES: return None
    features_arr = np.array(features, dtype=np.float32)
    if np.any(np.isnan(features_arr)) or np.any(np.isinf(features_arr)):
        features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(features_arr) != EXPECTED_N_FEATURES: return None
    return features_arr


# %% [markdown]
# # 5. Model Development (Example: Patch Classification + Regression)
#
# <<< SKIPPING Training Data Generation >>>

# %%
# --- Generate Training Data ---
PATCH_SIZE_PX = 64 # Keep constants defined
EXPECTED_N_FEATURES = 15

print("\nSKIPPING Training Data Generation as requested.")

# Define variables that would have been created as None or empty
features_list = []; labels_clf_list = []; labels_reg_list = []
failed_tomos = []
clf_model = None # Crucial: Ensure models are None
reg_model = None # Crucial: Ensure models are None
X = np.array([]).reshape(0, EXPECTED_N_FEATURES)
y_clf = np.array([], dtype=np.int32)
y_reg_all = np.array([]).reshape(0, 3)
X_reg = np.array([]).reshape(0, EXPECTED_N_FEATURES)
y_reg = np.array([]).reshape(0, 3)
X_train_clf = X_val_clf = np.array([]).reshape(0, EXPECTED_N_FEATURES)
y_train_clf = y_val_clf = np.array([], dtype=np.int32)
X_train_reg = X_val_reg = np.array([]).reshape(0, EXPECTED_N_FEATURES)
y_train_reg = y_val_reg = np.array([]).reshape(0, 3)

gc.collect() # Clean up memory in case any large objects were created before skip


# %% [markdown]
# ### 5.1 Train Models
#
# <<< SKIPPING Model Training >>>

# %%
# --- Split data and Train ---
print("\nSKIPPING Model Training as training data generation was skipped.")

# Models remain None as set in the previous step
print(f"\nModels after training attempts:")
print(f"  Classifier model: {'Available' if clf_model is not None else 'Not Available'}")
print(f"  Regressor model: {'Available' if reg_model is not None else 'Not Available'}")


# %% [markdown]
# # 6. Evaluation Metric Implementation
# (Keep definition, although it won't be used effectively without trained models/predictions)

# %%
def calculate_fbeta_and_distance(y_true_df, y_pred_df, voxel_spacing_map, default_voxel_spacing, beta=F_BETA, threshold_a=DISTANCE_THRESHOLD_A):
    """
    Calculates the competition metric (F-beta score based on distance).
    Handles multiple ground truth motors per tomogram.
    """
    tp = 0; fp = 0; fn = 0
    no_motor = float(NO_MOTOR_COORD)

    if y_pred_df.empty: pred_map = {}
    else:
        y_pred_df_unique = y_pred_df.drop_duplicates(subset='tomo_id', keep='first')
        pred_map = y_pred_df_unique.set_index('tomo_id')[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].apply(lambda row: row.tolist(), axis=1).to_dict()

    if y_true_df.empty: true_grouped = {}; total_true_motors = 0
    else:
        y_true_motors_only = y_true_df[y_true_df['Motor axis 0'] != no_motor].copy()
        true_grouped = y_true_motors_only.groupby('tomo_id')
        total_true_motors = len(y_true_motors_only)

    tp_final = 0; fp_final = 0
    matched_true_motor_indices = {}

    for tomo_id, pred_coords_a in pred_map.items():
        if pred_coords_a[0] == no_motor: continue
        true_motors_a = []; true_indices = []; is_true_motor_present = False
        if tomo_id in true_grouped.groups:
             true_df_tomo = true_grouped.get_group(tomo_id)
             true_motors_a = true_df_tomo[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
             true_indices = true_df_tomo.index.tolist(); is_true_motor_present = True
        if not is_true_motor_present: fp_final += 1; continue
        pred_vec = np.array(pred_coords_a)
        found_match = False; best_match_true_idx = -1; min_dist = float('inf')
        for i, true_motor_a in enumerate(true_motors_a):
            dist = np.linalg.norm(true_motor_a - pred_vec)
            original_idx = true_indices[i]
            if dist <= threshold_a:
                 already_matched = tomo_id in matched_true_motor_indices and original_idx in matched_true_motor_indices[tomo_id]
                 if not already_matched and dist < min_dist:
                      min_dist = dist; best_match_true_idx = original_idx; found_match = True
        if found_match:
            tp_final += 1
            if tomo_id not in matched_true_motor_indices: matched_true_motor_indices[tomo_id] = set()
            matched_true_motor_indices[tomo_id].add(best_match_true_idx)
        else: fp_final += 1
    fn_final = total_true_motors - tp_final
    f_beta_numerator = (1 + beta**2) * tp_final
    f_beta_denominator = (1 + beta**2) * tp_final + (beta**2 * fn_final) + fp_final
    if f_beta_denominator == 0: f_beta_score = 1.0 if total_true_motors == 0 else 0.0
    else: f_beta_score = f_beta_numerator / f_beta_denominator
    stats = {'TP': tp_final, 'FP': fp_final, 'FN': fn_final, 'Total True': total_true_motors}
    # Print score even if it's based on default predictions
    print(f"Evaluation Stats (based on default predictions): TP={tp_final}, FP={fp_final}, FN={fn_final} (Total True={total_true_motors}), F{beta}_Score={f_beta_score:.4f}")
    return f_beta_score, stats


# %% [markdown]
# # 7. Prediction Pipeline on Specific Tomograms
#
# Apply the (non-existent) models ONLY to the requested test set IDs: `tomo_003acc`, `tomo_00e047`, `tomo_00e463`.
# This will load the data but predict NO_MOTOR_COORD because models are None.

# %%
# --- Prediction Function (Definition unchanged, behavior changes as models are None) ---
def predict_motor_location(tomo_data, voxel_spacing, clf_model, reg_model, patch_size_px=PATCH_SIZE_PX, n_samples=1000, prob_threshold=0.6):
    """Predicts motor location by sampling patches."""
    # This check is now crucial and will always be true in this script version
    if clf_model is None:
        # No need to print warning every time here, we know it's None
        return [NO_MOTOR_COORD] * 3
    # The rest of the function will not execute if clf_model is None

    # --- Original function logic (will not run) ---
    if tomo_data is None or voxel_spacing <= 0: return [NO_MOTOR_COORD] * 3
    pred_start_time = time.time()
    preprocessed_data = preprocess_tomogram(tomo_data)
    if preprocessed_data is None: return [NO_MOTOR_COORD] * 3
    tomo_shape = preprocessed_data.shape
    candidate_features = []; candidate_centers_px = []
    n_samples_to_use = n_samples if n_samples > 0 else 2500
    for i in range(n_samples_to_use):
        center_px = np.random.randint(0, tomo_shape, size=3)
        patch, _ = get_local_patch(preprocessed_data, center_px, patch_size_px=patch_size_px)
        if patch is not None:
            features = extract_features_patch(patch)
            if features is not None: candidate_features.append(features); candidate_centers_px.append(center_px)
    if not candidate_features: return [NO_MOTOR_COORD] * 3
    candidate_features_arr = np.array(candidate_features, dtype=np.float32)
    try:
        if candidate_features_arr.shape[0] == 0: return [NO_MOTOR_COORD] * 3
        probs = clf_model.predict_proba(candidate_features_arr)[:, 1]
    except Exception as e: print(f"Error predict_proba: {e}"); return [NO_MOTOR_COORD] * 3
    if len(probs) == 0: return [NO_MOTOR_COORD] * 3
    max_prob_idx = np.argmax(probs); max_prob = probs[max_prob_idx]; best_center_px = candidate_centers_px[max_prob_idx]
    if max_prob >= prob_threshold:
        best_features = candidate_features_arr[max_prob_idx:max_prob_idx+1]; predicted_offset_px = np.zeros(3)
        if reg_model is not None:
            try:
                 if best_features.shape[0] > 0: predicted_offset_px = reg_model.predict(best_features)[0]
            except Exception as e: print(f"Warn reg_predict: {e}")
        final_predicted_coords_px = best_center_px + predicted_offset_px
        final_predicted_coords_a = final_predicted_coords_px * voxel_spacing
        max_coords_a = (np.array(tomo_shape) - 1) * voxel_spacing
        final_predicted_coords_a = np.clip(final_predicted_coords_a, 0, max_coords_a); return final_predicted_coords_a.tolist()
    else: return [NO_MOTOR_COORD] * 3
    # --- End of original function logic ---

# --- Generate Predictions ONLY for Specific Tomogram IDs ---
print("\nGenerating DEFAULT Predictions for SPECIFIC Tomograms (Models were not trained)...")
predictions = []
test_predict_start_time = time.time()

ids_to_predict = ['tomo_003acc', 'tomo_00e047', 'tomo_00e463']
print(f"Processing ONLY the following {len(ids_to_predict)} tomograms: {ids_to_predict}")

if not ids_to_predict:
     print("No specific tomograms defined to process.")
# No need to check clf_model here, predict_motor_location handles it
else:
    if voxel_spacing_map:
        valid_spacings = [v for v in voxel_spacing_map.values() if v > 0]
        avg_voxel_spacing = np.mean(valid_spacings) if valid_spacings else DEFAULT_VOXEL_SPACING_A
    else: avg_voxel_spacing = DEFAULT_VOXEL_SPACING_A
    print(f"Using average voxel spacing (fallback): {avg_voxel_spacing:.4f} A/px")

    for tomo_idx, tomo_id in enumerate(ids_to_predict):
        print(f"Generating default prediction for {tomo_idx+1}/{len(ids_to_predict)}: {tomo_id}...")
        tomo_dir_path = test_dir_map.get(tomo_id)
        source_dir = "test"
        if not tomo_dir_path:
            tomo_dir_path = train_dir_map.get(tomo_id)
            source_dir = "train"
        if not tomo_dir_path:
            print(f"  Skipping {tomo_id}: Directory not found.")
            pred_coords = [NO_MOTOR_COORD] * 3
            predictions.append({'tomo_id': tomo_id,'Motor axis 0': pred_coords[0],'Motor axis 1': pred_coords[1],'Motor axis 2': pred_coords[2]})
            continue

        # Load data to get shape info, even though model won't use features
        tomo_data = load_tomogram(tomo_id, {tomo_id: tomo_dir_path})
        current_voxel_spacing = voxel_spacing_map.get(tomo_id, avg_voxel_spacing)

        # Call prediction function - it will return default due to clf_model being None
        pred_coords = predict_motor_location(
            tomo_data, current_voxel_spacing, clf_model, reg_model, # Pass None models
            patch_size_px=PATCH_SIZE_PX, n_samples=10, prob_threshold=0.99 # Params don't matter much here
        )
        print(f"  Predicted Coords (Default) for {tomo_id}: {[f'{c:.2f}' for c in pred_coords]}")

        predictions.append({'tomo_id': tomo_id,'Motor axis 0': pred_coords[0],'Motor axis 1': pred_coords[1],'Motor axis 2': pred_coords[2]})
        if tomo_data is not None: del tomo_data; gc.collect()

    print(f"\nFinished default predictions. Total time: {time.time() - test_predict_start_time:.2f} seconds.")

# Create dataframe from the default predictions
if predictions:
    submission_df = pd.DataFrame(predictions)
    print("\nDefault Predictions for the specified IDs:")
    print(submission_df)
else:
    print("\nNo default predictions were generated (check specific IDs).")
    submission_df = pd.DataFrame(columns=['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2'])


# %% [markdown]
# # 8. Create Submission File
#
# Generate the submission file template, ensuring all original test IDs are present.

# %%
print("\nCreating Submission File Template...")
if 'submission_df' in locals():
    try:
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        expected_test_ids = test_ids_found # Use all IDs found in the test directory

        if not expected_test_ids:
             print("Warning: No test IDs were found in the test directory structure.")
             expected_test_ids = sample_sub['tomo_id'].tolist()
             print(f"Using {len(expected_test_ids)} IDs from sample submission as base.")
        else:
             print(f"Generating submission structure based on {len(expected_test_ids)} test IDs found in directory.")

        sub_df_final = pd.DataFrame({'tomo_id': expected_test_ids})

        # Merge the default predictions we generated for the *specific* IDs
        if not submission_df.empty:
             sub_df_final = sub_df_final.merge(submission_df, on='tomo_id', how='left')
        else:
             # If even the default predictions weren't made, fill everything
             for col in ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']: sub_df_final[col] = NO_MOTOR_COORD

        # Fill any remaining NaNs (for test IDs not in the specific prediction list) with default
        sub_df_final.fillna(NO_MOTOR_COORD, inplace=True)

        # Ensure column order and types match sample
        final_columns = sample_sub.columns.tolist()
        for col in final_columns:
             if col not in sub_df_final.columns:
                  print(f"Warning: Column '{col}' from sample not in generated df. Adding default.")
                  sub_df_final[col] = NO_MOTOR_COORD
        sub_df_final = sub_df_final[final_columns]
        for col in ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']:
            sub_df_final[col] = pd.to_numeric(sub_df_final[col], errors='coerce').fillna(NO_MOTOR_COORD).astype(float)

        sub_df_final.to_csv('submission.csv', index=False)
        print("Submission file template created: submission.csv")
        print("\nFinal Submission Head:")
        print(sub_df_final.head())
        print(f"Submission shape: {sub_df_final.shape}")

        if len(sub_df_final) != len(expected_test_ids):
            print(f"CRITICAL WARNING: Submission row count ({len(sub_df_final)}) "
                  f"mismatches expected test IDs ({len(expected_test_ids)})!")
        else:
             print("Submission row count matches expected test IDs.")

    except FileNotFoundError:
        print(f"Error: Sample submission file not found at {SAMPLE_SUB_PATH}. Cannot verify format.")
        # Try to save anyway if df exists
        if 'sub_df_final' in locals():
             sub_df_final.to_csv('submission.csv', index=False)
             print("Submission file template created (format not verified): submission.csv")
    except Exception as e:
        print(f"Error creating final submission file: {e}")
else:
    print("No prediction DataFrame ('submission_df') was generated. Cannot create submission template.")


# %% [markdown]
# # 9. Discussion and Future Work
#
# *   **Data Loading & Handling:** Code structure maintained for loading data, labels, and finding directories.
# *   **Feature Engineering:** Utility functions (`preprocess_tomogram`, `extract_features_patch`) remain defined but were not used for training.
# *   **Modeling:** **Training data generation and model training sections were entirely skipped.** `clf_model` and `reg_model` were explicitly set to `None`.
# *   **Targeted Prediction:** The prediction loop for specific IDs (`tomo_003acc`, `tomo_00e047`, `tomo_00e463`) was executed. However, since the models were `None`, the `predict_motor_location` function returned the default `NO_MOTOR_COORD` for these IDs.
# *   **Submission Generation:** A `submission.csv` file was successfully created containing *all* test IDs found in the `test` directory. The specifically targeted IDs have `NO_MOTOR_COORD` because no model was trained, and all other test IDs were also filled with `NO_MOTOR_COORD`. This serves as a valid submission template.
# *   **Limitations:** This script produces a baseline submission with no actual detection. Its primary purpose is to verify the data loading, directory handling, and submission file formatting logic.
# *   **Future Improvements:** To get actual predictions, the skipping of Sections 5 and 5.1 must be removed, and sufficient training data (likely more than the 20 used in the previous version) needs to be processed to train meaningful models. Then, the prediction pipeline in Section 7 will use the trained models. All other future improvements mentioned previously (CNNs, FCNs, augmentation, etc.) still apply for building a competitive model.

# %%
print("Script finished (Training and Model Fitting Skipped). Submission template generated.")





# -*- coding: utf-8 -*-
"""
Kaggle Notebook: BYU Flagellar Motor Localization (Revised for Directory Structure & Multi-Motor)

Goal: Detect the presence and (x, y, z) coordinates of bacterial
      flagellar motors in 3D cryo-ET tomograms stored as slice directories.

Inspired Concepts: Analogies from Non-Linear Electrodynamics / String Theory
    - Tomogram Intensity: Scalar field (like potential)
    - Gradient (∇I): Vector field (analogous to Electric Field E)
    - Structure Tensor (∇I ⊗ ∇I averaged): Symmetric Tensor (analogous to Stress Tensor T_µν or Metric Perturbation S_µν)
    - Hessian (∇²I): Tensor capturing curvature (related to field changes)
    - Invariants: Rotationally invariant features derived from gradient/structure tensor/Hessian (analogous to Lorentz invariants x, y)
    - Anisotropic Diffusion/Filtering: Guided propagation of information (analogous to wave propagation influenced by background fields/metrics)
    - Multi-Metric Idea: Using different analysis scales/features (like g_µν vs G_µν) - e.g., global search vs. local refinement.

MODIFICATION: This version SKIPS training data generation and model training entirely.
            It focuses prediction ONLY on specified IDs ['tomo_003acc', 'tomo_00e047', 'tomo_00e463']
            using NO trained model (will predict NO_MOTOR_COORD).
            The final output is a submission file template for ALL test IDs found.
"""

# %% [markdown]
# # 1. Setup and Imports
#
# Load necessary libraries and define constants.

# %%
# Attempt installation only if needed (e.g., in Kaggle environment)
try:
    import cv2
    import mrcfile # Though likely unused for loading now
    print("Libraries opencv-python, mrcfile seem available.")
except ImportError:
    print("Installing required libraries: opencv-python, mrcfile")
    # Use %pip instead of !pip in notebooks for better integration
    %pip install opencv-python mrcfile --quiet
    import cv2
    import mrcfile
    print("Libraries installed.")

import os
import glob
import numpy as np
import pandas as pd
import cv2 # Import OpenCV
# import mrcfile # Keep if other parts might use it, but not for loading slices
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as ndi
from skimage.feature import structure_tensor, structure_tensor_eigenvalues, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import gaussian, median
from skimage.measure import regionprops, label
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import fbeta_score, mean_squared_error, make_scorer
import gc # Garbage collection
import time # For timing

# Optional: Deep Learning Libraries (if used)
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import monai # Medical Imaging AI library

# Constants
COMPETITION_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
TRAIN_DIR = os.path.join(COMPETITION_DIR, 'train')
TEST_DIR = os.path.join(COMPETITION_DIR, 'test')
TRAIN_LABELS_PATH = os.path.join(COMPETITION_DIR, 'train_labels.csv')
SAMPLE_SUB_PATH = os.path.join(COMPETITION_DIR, 'sample_submission.csv')

# Default Voxel spacing (used for test set or if lookup fails)
DEFAULT_VOXEL_SPACING_A = 10.0
print(f"Using default voxel spacing: {DEFAULT_VOXEL_SPACING_A} Angstroms/pixel (will try to read from labels for training)")

# Evaluation Threshold (Angstroms)
DISTANCE_THRESHOLD_A = 1000.0
# DISTANCE_THRESHOLD_PX will be calculated dynamically based on actual voxel spacing

# F-beta Score Beta value
F_BETA = 2.0

# For submission: Coordinate value indicating no motor found
NO_MOTOR_COORD = -1.0 # Ensure float

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)

# %% [markdown]
# # 2. Load Data and Metadata
#
# Load the training labels and find the data directories. Check for consistency.

# %%
start_time = time.time()
print("Loading labels...")
try:
    train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Training labels shape: {train_labels_df.shape}")
    print(train_labels_df.head())
except FileNotFoundError:
    print(f"ERROR: Training labels file not found at {TRAIN_LABELS_PATH}")
    train_labels_df = pd.DataFrame() # Empty dataframe

# --- Debug: List directories to confirm paths ---
print("\n--- Directory Listing ---")
print(f"Listing {COMPETITION_DIR}:")
try:
    print(os.listdir(COMPETITION_DIR))
except FileNotFoundError:
    print(f"  Error: Directory not found: {COMPETITION_DIR}")

print(f"\nListing {TRAIN_DIR}:")
try:
    train_contents = os.listdir(TRAIN_DIR)
    print(f"  Found {len(train_contents)} items in train dir. First 10:")
    print(train_contents[:10])
    if len(train_contents) > 10: print("  ...")
except FileNotFoundError:
    print(f"  Error: Directory not found: {TRAIN_DIR}")
    train_contents = []

print(f"\nListing {TEST_DIR}:")
try:
    test_contents = os.listdir(TEST_DIR)
    print(f"  Found {len(test_contents)} items in test dir. First 10:")
    print(test_contents[:10])
    if len(test_contents) > 10: print("  ...")
except FileNotFoundError:
    print(f"  Error: Directory not found: {TEST_DIR}")
    test_contents = []
print("--- End Directory Listing ---\n")


# --- Find Tomogram Directories ---
print("Finding tomogram directories...")
train_dirs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'tomo_*')))
test_dirs = sorted(glob.glob(os.path.join(TEST_DIR, 'tomo_*')))

# Extract UNIQUE tomo_ids present in the directories found
train_ids_found = sorted([os.path.basename(d) for d in train_dirs if os.path.isdir(d)]) # Ensure it's a directory
test_ids_found = sorted([os.path.basename(d) for d in test_dirs if os.path.isdir(d)]) # Ensure it's a directory

print(f"Found {len(train_ids_found)} potential training tomogram directories.")
print(f"First 5 found: {train_ids_found[:5]}")
print(f"Found {len(test_ids_found)} potential testing tomogram directories.")
print(f"First 5 found: {test_ids_found[:5]}")

# --- Filter Labels based on Found Directories ---
if not train_labels_df.empty:
    # Check if 'tomo_id' column exists
    if 'tomo_id' not in train_labels_df.columns:
        print("ERROR: 'tomo_id' column not found in labels CSV. Cannot proceed.")
        train_labels_df_filtered = pd.DataFrame()
        unique_train_ids_with_data = []
        voxel_spacing_map = {}
    else:
        train_labels_df_filtered = train_labels_df[train_labels_df['tomo_id'].isin(train_ids_found)].copy()
        print(f"\nOriginal label rows: {len(train_labels_df)}")
        print(f"Label rows after filtering by found train directories: {len(train_labels_df_filtered)}")

        # Get unique tomogram IDs that have both labels AND a found directory
        unique_train_ids_with_data = sorted(train_labels_df_filtered['tomo_id'].unique())
        print(f"Number of unique train tomograms with labels AND data: {len(unique_train_ids_with_data)}")

        # Create map for Voxel Spacing from labels (use first entry per tomo_id)
        if 'Voxel spacing' in train_labels_df_filtered.columns and not train_labels_df_filtered.empty:
            voxel_spacing_map = train_labels_df_filtered.drop_duplicates(subset='tomo_id').set_index('tomo_id')['Voxel spacing'].to_dict()
            print(f"Example voxel spacing: {list(voxel_spacing_map.items())[:5]}")
        else:
             print("WARNING: 'Voxel spacing' column not found or no valid labels. Cannot create voxel spacing map.")
             voxel_spacing_map = {}

else:
    print("WARNING: Training labels dataframe is empty. Cannot filter or create maps.")
    train_labels_df_filtered = pd.DataFrame()
    unique_train_ids_with_data = []
    voxel_spacing_map = {}

# Map tomo_id to DIRECTORY path (even if labels are missing)
train_dir_map = {os.path.basename(d): d for d in train_dirs if os.path.isdir(d)}
test_dir_map = {os.path.basename(d): d for d in test_dirs if os.path.isdir(d)}


print(f"Data loading setup took {time.time() - start_time:.2f} seconds.")

if not unique_train_ids_with_data and not train_labels_df.empty:
    print("\n\nCRITICAL WARNING: No intersection between labels and found training directories. Training cannot proceed.")
elif not train_ids_found and not test_ids_found:
     print("\n\nCRITICAL WARNING: No training or testing directories found. Check data paths.")

# %% [markdown]
# # 3. Exploratory Data Analysis (EDA) & Concept Visualization
#
# Understand the data distribution, visualize tomograms and labels, and see how our physics-inspired concepts manifest.
# (Keeping these utility functions defined, even if visualization is partially skipped)

# %%
# --- Tomogram Loading Function ---
def load_tomogram(tomo_id, dir_map):
    """Loads a tomogram by reading and stacking slices from a directory."""
    if isinstance(dir_map, str):
        directory_path = dir_map
        if not os.path.isdir(directory_path): return None
    else:
        directory_path = dir_map.get(tomo_id)
        if not directory_path or not os.path.isdir(directory_path): return None

    slice_files = sorted(glob.glob(os.path.join(directory_path, 'slice_*.jpg')))
    if not slice_files: slice_files = sorted(glob.glob(os.path.join(directory_path, 'slice_*.png')))
    if not slice_files: slice_files = sorted(glob.glob(os.path.join(directory_path, 'slice_*.tif')))
    if not slice_files: return None

    def get_slice_number(filepath):
        try:
            filename = os.path.basename(filepath)
            num_str = filename.split('slice_')[1].split('.')[0]
            return int(num_str)
        except: return -1

    valid_slice_files = [(f, get_slice_number(f)) for f in slice_files]
    valid_slice_files = [(f, num) for f, num in valid_slice_files if num != -1]
    if not valid_slice_files: return None
    valid_slice_files.sort(key=lambda item: item[1])
    slice_files_sorted = [f for f, num in valid_slice_files]

    try:
        first_slice = cv2.imread(slice_files_sorted[0], cv2.IMREAD_GRAYSCALE)
        if first_slice is None: raise IOError(f"cv2.imread failed for {slice_files_sorted[0]}")
        height, width = first_slice.shape
        num_slices = len(slice_files_sorted)
        dtype = first_slice.dtype
    except Exception as e:
        print(f"Error reading first slice {slice_files_sorted[0]}: {e}"); return None

    tomogram_data = np.zeros((num_slices, height, width), dtype=dtype)
    tomogram_data[0, :, :] = first_slice

    for i in range(1, num_slices):
        try:
            slice_img = cv2.imread(slice_files_sorted[i], cv2.IMREAD_GRAYSCALE)
            if slice_img is None: raise IOError(f"cv2.imread failed for {slice_files_sorted[i]}")
            if slice_img.shape != (height, width): return None
            tomogram_data[i, :, :] = slice_img
        except Exception as e:
            print(f"Error reading slice {slice_files_sorted[i]}: {e}"); return None
    return tomogram_data

# --- Plotting Function ---
def plot_slices_with_label(tomo_data, voxel_spacing, label_coords_A=None, title="Tomogram Slices"):
    """Plots central slices along each axis, optionally marking the label(s)."""
    if tomo_data is None: print("No data to plot."); return
    if voxel_spacing <= 0: print("Invalid voxel spacing."); return
    shape = tomo_data.shape
    if not (len(shape) == 3 and all(s > 0 for s in shape)): print(f"Invalid shape {shape}."); return

    center_z, center_y, center_x = shape[0] // 2, shape[1] // 2, shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); fig.suptitle(title, fontsize=16)
    try:
        axes[0].imshow(tomo_data[center_z, :, :], cmap='gray'); axes[0].set_title(f'Z Slice (Z={center_z})')
        axes[1].imshow(tomo_data[:, center_y, :], cmap='gray', aspect=shape[0]/shape[2] if shape[2]>0 else 1); axes[1].set_title(f'Y Slice (Y={center_y})')
        axes[2].imshow(tomo_data[:, :, center_x], cmap='gray', aspect=shape[0]/shape[1] if shape[1]>0 else 1); axes[2].set_title(f'X Slice (X={center_x})')
    except IndexError as e: print(f"Error plotting slices: {e}"); plt.close(fig); return

    if label_coords_A is not None and len(label_coords_A) > 0:
        is_multiple = isinstance(label_coords_A[0], (list, np.ndarray))
        if not is_multiple: label_coords_A = [label_coords_A]
        plotted_legend = False
        for i, motor_A in enumerate(label_coords_A):
            if len(motor_A) < 3 or motor_A[0] == NO_MOTOR_COORD: continue
            lz_A, ly_A, lx_A = motor_A
            lz_px, ly_px, lx_px = int(lz_A / voxel_spacing), int(ly_A / voxel_spacing), int(lx_A / voxel_spacing)
            if not (0 <= lz_px < shape[0] and 0 <= ly_px < shape[1] and 0 <= lx_px < shape[2]): continue
            label_text = f'Motor {i+1}' if len(label_coords_A) > 1 else 'Motor'
            z_thresh=max(5, shape[0]*0.05); y_thresh=max(5, shape[1]*0.05); x_thresh=max(5, shape[2]*0.05)
            try:
                lbl = None
                if not plotted_legend: lbl = label_text; plotted_legend=True
                if abs(lz_px - center_z) < z_thresh: axes[0].plot(lx_px, ly_px, 'ro', ms=8, label=lbl)
                if abs(ly_px - center_y) < y_thresh: axes[1].plot(lx_px, lz_px, 'ro', ms=8)
                if abs(lx_px - center_x) < x_thresh: axes[2].plot(ly_px, lz_px, 'ro', ms=8)
                if lbl: lbl=None
            except Exception as plot_e: print(f"Error plotting marker: {plot_e}")
    if plotted_legend: axes[0].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

# --- Visualize a sample tomogram ---
# (Keep this visualization, it's quick and useful)
if unique_train_ids_with_data:
    sample_tomo_id = unique_train_ids_with_data[0]
    print(f"\nVisualizing sample tomogram: {sample_tomo_id}")
    sample_data = load_tomogram(sample_tomo_id, train_dir_map)
    sample_labels = train_labels_df_filtered[train_labels_df_filtered['tomo_id'] == sample_tomo_id]
    if sample_data is not None and not sample_labels.empty:
        motor_locations_A = sample_labels[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values.tolist()
        voxel_spacing = sample_labels.iloc[0]['Voxel spacing']
        print(f"Sample Tomogram Shape: {sample_data.shape}, Voxel Spacing: {voxel_spacing} A/px")
        print(f"Motor Location(s) (Angstroms): {motor_locations_A}")
        plot_slices_with_label(sample_data, voxel_spacing, motor_locations_A, title=f"{sample_tomo_id}")
        plt.figure(figsize=(10, 4)); plt.hist(sample_data.flatten(), bins=100, color='blue', alpha=0.7); plt.title(f'Intensity Distribution {sample_tomo_id}'); plt.xlabel('Intensity'); plt.ylabel('Frequency'); plt.grid(True, alpha=0.3); plt.show()
        del sample_data, sample_labels; gc.collect()
    else: print(f"Failed to load/find labels for sample {sample_tomo_id}")
else: print("\nSkipping Visualization: No valid training tomograms found.")

# %% [markdown]
# ### 3.1 Visualizing Physics-Inspired Concepts
# (Skip this visualization as it relies on patch extraction and calculations that mirror training data gen)

# %%
# --- Patch Extraction Function (Keep definition for prediction step) ---
def get_local_patch(tomo_data, center_px, patch_size_px=64):
    """Extracts a 3D patch centered at center_px. Patch size is in pixels."""
    if tomo_data is None or center_px is None: return None, None
    center_px = np.round(center_px).astype(int); z, y, x = center_px
    shape = tomo_data.shape
    if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]): return None, None
    half_size = patch_size_px // 2
    z_start, z_end = max(0, z - half_size), min(shape[0], z + half_size)
    y_start, y_end = max(0, y - half_size), min(shape[1], y + half_size)
    x_start, x_end = max(0, x - half_size), min(shape[2], x + half_size)
    dest_z_start = half_size - (z - z_start); dest_z_end = dest_z_start + (z_end - z_start)
    dest_y_start = half_size - (y - y_start); dest_y_end = dest_y_start + (y_end - y_start)
    dest_x_start = half_size - (x - x_start); dest_x_end = dest_x_start + (x_end - x_start)
    patch = np.zeros((patch_size_px, patch_size_px, patch_size_px), dtype=tomo_data.dtype)
    try:
        src_slice = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
        dest_slice = (slice(dest_z_start, dest_z_end), slice(dest_y_start, dest_y_end), slice(dest_x_start, dest_x_end))
        patch[dest_slice] = tomo_data[src_slice]
        patch_center_coords_in_patch = np.round([z - z_start + dest_z_start, y - y_start + dest_y_start, x - x_start + dest_x_start]).astype(int)
        if not (0 <= patch_center_coords_in_patch[0] < patch_size_px and 0 <= patch_center_coords_in_patch[1] < patch_size_px and 0 <= patch_center_coords_in_patch[2] < patch_size_px):
             patch_center_coords_in_patch = [patch_size_px // 2] * 3
        return patch, patch_center_coords_in_patch
    except (ValueError, IndexError) as e: return None, None

# --- Visualize physics concepts ---
print("\nSkipping Physics Concept Visualization (requires patch processing).")
# (The code for visualization is removed here)

# %% [markdown]
# # 4. Data Preprocessing & Feature Engineering
# (Keep function definitions as they are needed for prediction, even if models aren't trained)

# %%
# --- Simple Preprocessing ---
def preprocess_tomogram(data):
    """Normalizes and smooths the tomogram data."""
    if data is None: return None
    if not np.issubdtype(data.dtype, np.floating): data = data.astype(np.float32)
    mean, std = np.mean(data), np.std(data)
    if std > 1e-6: normalized_data = (data - mean) / std
    else: normalized_data = data - mean
    smoothed_data = gaussian(normalized_data, sigma=1.5, mode='reflect', preserve_range=True, truncate=4.0)
    return smoothed_data

# --- Feature Extraction (Keep definition) ---
def extract_features_patch(patch):
    """Extracts a feature vector from a 3D patch."""
    if patch is None or patch.size == 0: return None
    if not np.issubdtype(patch.dtype, np.floating): patch = patch.astype(np.float32)
    features = []
    EXPECTED_N_FEATURES = 15
    try:
        features.extend([np.mean(patch), np.std(patch), np.median(patch), np.min(patch), np.max(patch)])
        patch_smooth_grad = gaussian(patch, sigma=1.0, mode='reflect', preserve_range=True, truncate=4.0)
        patch_smooth_st = gaussian(patch, sigma=1.5, mode='reflect', preserve_range=True, truncate=4.0)
        patch_smooth_hess = gaussian(patch, sigma=2.0, mode='reflect', preserve_range=True, truncate=4.0)
        try:
            grad_z, grad_y, grad_x = np.gradient(patch_smooth_grad)
            grad_mag = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
            features.extend([np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag)])
        except Exception: features.extend([0.0] * 3)
        try:
            shape = patch_smooth_st.shape; slice_start = max(0, shape[0]//5); slice_end = max(slice_start+1, 4*shape[0]//5)
            center_slice = slice(slice_start, slice_end)
            if not (0<=center_slice.start<shape[0] and 0<center_slice.stop<=shape[0] and center_slice.start<center_slice.stop): center_patch_smooth = patch_smooth_st
            else: center_patch_smooth = patch_smooth_st[center_slice, center_slice, center_slice]
            if center_patch_smooth.size > 0:
                S_elems_c = structure_tensor(center_patch_smooth, sigma=1.5, mode='reflect')
                eigvals_S_c = structure_tensor_eigenvalues(S_elems_c); eigvals_S_c = np.sort(eigvals_S_c, axis=0)
                l3, l2, l1 = eigvals_S_c[0], eigvals_S_c[1], eigvals_S_c[2]
                den = l1 + l2 + l3 + 1e-9; coh = np.where(den > 1e-8, (l1 - l3) / den, 0.0)
                features.extend([np.mean(l1), np.mean(l3), np.mean(coh), np.max(coh)])
            else: features.extend([0.0] * 4)
        except Exception: features.extend([0.0] * 4)
        try:
            shape_h = patch_smooth_hess.shape; slice_start_h = max(0, shape_h[0]//5); slice_end_h = max(slice_start_h+1, 4*shape_h[0]//5)
            center_slice_h = slice(slice_start_h, slice_end_h)
            if not (0<=center_slice_h.start<shape_h[0] and 0<center_slice_h.stop<=shape_h[0] and center_slice_h.start<center_slice_h.stop): center_patch_smooth_h = patch_smooth_hess
            else: center_patch_smooth_h = patch_smooth_hess[center_slice_h, center_slice_h, center_slice_h]
            if center_patch_smooth_h.size > 0:
                h_matrix = hessian_matrix(center_patch_smooth_h, sigma=2.0, mode='reflect', use_gaussian_derivatives=False)
                eigvals_H_c = hessian_matrix_eigvals(h_matrix); eigvals_H_c = np.sort(eigvals_H_c, axis=0)
                h1, h2, h3 = eigvals_H_c[0], eigvals_H_c[1], eigvals_H_c[2]
                features.extend([np.mean(h1), np.mean(h3), np.std(h1)])
            else: features.extend([0.0] * 3)
        except Exception: features.extend([0.0] * 3)
    except Exception as outer_e: print(f"Outer FE error: {outer_e}"); return None
    if len(features) != EXPECTED_N_FEATURES:
        if len(features) < EXPECTED_N_FEATURES: features.extend([0.0] * (EXPECTED_N_FEATURES - len(features)))
        else: features = features[:EXPECTED_N_FEATURES]
        if len(features) != EXPECTED_N_FEATURES: return None
    features_arr = np.array(features, dtype=np.float32)
    if np.any(np.isnan(features_arr)) or np.any(np.isinf(features_arr)):
        features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(features_arr) != EXPECTED_N_FEATURES: return None
    return features_arr


# %% [markdown]
# # 5. Model Development (Example: Patch Classification + Regression)
#
# <<< SKIPPING Training Data Generation >>>

# %%
# --- Generate Training Data ---
PATCH_SIZE_PX = 64 # Keep constants defined
EXPECTED_N_FEATURES = 15

print("\nSKIPPING Training Data Generation as requested.")

# Define variables that would have been created as None or empty
features_list = []; labels_clf_list = []; labels_reg_list = []
failed_tomos = []
clf_model = None # Crucial: Ensure models are None
reg_model = None # Crucial: Ensure models are None
X = np.array([]).reshape(0, EXPECTED_N_FEATURES)
y_clf = np.array([], dtype=np.int32)
y_reg_all = np.array([]).reshape(0, 3)
X_reg = np.array([]).reshape(0, EXPECTED_N_FEATURES)
y_reg = np.array([]).reshape(0, 3)
X_train_clf = X_val_clf = np.array([]).reshape(0, EXPECTED_N_FEATURES)
y_train_clf = y_val_clf = np.array([], dtype=np.int32)
X_train_reg = X_val_reg = np.array([]).reshape(0, EXPECTED_N_FEATURES)
y_train_reg = y_val_reg = np.array([]).reshape(0, 3)

gc.collect() # Clean up memory in case any large objects were created before skip


# %% [markdown]
# ### 5.1 Train Models
#
# <<< SKIPPING Model Training >>>

# %%
# --- Split data and Train ---
print("\nSKIPPING Model Training as training data generation was skipped.")

# Models remain None as set in the previous step
print(f"\nModels after training attempts:")
print(f"  Classifier model: {'Available' if clf_model is not None else 'Not Available'}")
print(f"  Regressor model: {'Available' if reg_model is not None else 'Not Available'}")


# %% [markdown]
# # 6. Evaluation Metric Implementation
# (Keep definition, although it won't be used effectively without trained models/predictions)

# %%
def calculate_fbeta_and_distance(y_true_df, y_pred_df, voxel_spacing_map, default_voxel_spacing, beta=F_BETA, threshold_a=DISTANCE_THRESHOLD_A):
    """
    Calculates the competition metric (F-beta score based on distance).
    Handles multiple ground truth motors per tomogram.
    """
    tp = 0; fp = 0; fn = 0
    no_motor = float(NO_MOTOR_COORD)

    if y_pred_df.empty: pred_map = {}
    else:
        y_pred_df_unique = y_pred_df.drop_duplicates(subset='tomo_id', keep='first')
        pred_map = y_pred_df_unique.set_index('tomo_id')[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].apply(lambda row: row.tolist(), axis=1).to_dict()

    if y_true_df.empty: true_grouped = {}; total_true_motors = 0
    else:
        y_true_motors_only = y_true_df[y_true_df['Motor axis 0'] != no_motor].copy()
        true_grouped = y_true_motors_only.groupby('tomo_id')
        total_true_motors = len(y_true_motors_only)

    tp_final = 0; fp_final = 0
    matched_true_motor_indices = {}

    for tomo_id, pred_coords_a in pred_map.items():
        if pred_coords_a[0] == no_motor: continue
        true_motors_a = []; true_indices = []; is_true_motor_present = False
        if tomo_id in true_grouped.groups:
             true_df_tomo = true_grouped.get_group(tomo_id)
             true_motors_a = true_df_tomo[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
             true_indices = true_df_tomo.index.tolist(); is_true_motor_present = True
        if not is_true_motor_present: fp_final += 1; continue
        pred_vec = np.array(pred_coords_a)
        found_match = False; best_match_true_idx = -1; min_dist = float('inf')
        for i, true_motor_a in enumerate(true_motors_a):
            dist = np.linalg.norm(true_motor_a - pred_vec)
            original_idx = true_indices[i]
            if dist <= threshold_a:
                 already_matched = tomo_id in matched_true_motor_indices and original_idx in matched_true_motor_indices[tomo_id]
                 if not already_matched and dist < min_dist:
                      min_dist = dist; best_match_true_idx = original_idx; found_match = True
        if found_match:
            tp_final += 1
            if tomo_id not in matched_true_motor_indices: matched_true_motor_indices[tomo_id] = set()
            matched_true_motor_indices[tomo_id].add(best_match_true_idx)
        else: fp_final += 1
    fn_final = total_true_motors - tp_final
    f_beta_numerator = (1 + beta**2) * tp_final
    f_beta_denominator = (1 + beta**2) * tp_final + (beta**2 * fn_final) + fp_final
    if f_beta_denominator == 0: f_beta_score = 1.0 if total_true_motors == 0 else 0.0
    else: f_beta_score = f_beta_numerator / f_beta_denominator
    stats = {'TP': tp_final, 'FP': fp_final, 'FN': fn_final, 'Total True': total_true_motors}
    # Print score even if it's based on default predictions
    print(f"Evaluation Stats (based on default predictions): TP={tp_final}, FP={fp_final}, FN={fn_final} (Total True={total_true_motors}), F{beta}_Score={f_beta_score:.4f}")
    return f_beta_score, stats


# %% [markdown]
# # 7. Prediction Pipeline on Specific Tomograms
#
# Apply the (non-existent) models ONLY to the requested test set IDs: `tomo_003acc`, `tomo_00e047`, `tomo_00e463`.
# This will load the data but predict NO_MOTOR_COORD because models are None.

# %%
# --- Prediction Function (Definition unchanged, behavior changes as models are None) ---
def predict_motor_location(tomo_data, voxel_spacing, clf_model, reg_model, patch_size_px=PATCH_SIZE_PX, n_samples=1000, prob_threshold=0.6):
    """Predicts motor location by sampling patches."""
    # This check is now crucial and will always be true in this script version
    if clf_model is None:
        # No need to print warning every time here, we know it's None
        return [NO_MOTOR_COORD] * 3
    # The rest of the function will not execute if clf_model is None

    # --- Original function logic (will not run) ---
    if tomo_data is None or voxel_spacing <= 0: return [NO_MOTOR_COORD] * 3
    pred_start_time = time.time()
    preprocessed_data = preprocess_tomogram(tomo_data)
    if preprocessed_data is None: return [NO_MOTOR_COORD] * 3
    tomo_shape = preprocessed_data.shape
    candidate_features = []; candidate_centers_px = []
    n_samples_to_use = n_samples if n_samples > 0 else 2500
    for i in range(n_samples_to_use):
        center_px = np.random.randint(0, tomo_shape, size=3)
        patch, _ = get_local_patch(preprocessed_data, center_px, patch_size_px=patch_size_px)
        if patch is not None:
            features = extract_features_patch(patch)
            if features is not None: candidate_features.append(features); candidate_centers_px.append(center_px)
    if not candidate_features: return [NO_MOTOR_COORD] * 3
    candidate_features_arr = np.array(candidate_features, dtype=np.float32)
    try:
        if candidate_features_arr.shape[0] == 0: return [NO_MOTOR_COORD] * 3
        probs = clf_model.predict_proba(candidate_features_arr)[:, 1]
    except Exception as e: print(f"Error predict_proba: {e}"); return [NO_MOTOR_COORD] * 3
    if len(probs) == 0: return [NO_MOTOR_COORD] * 3
    max_prob_idx = np.argmax(probs); max_prob = probs[max_prob_idx]; best_center_px = candidate_centers_px[max_prob_idx]
    if max_prob >= prob_threshold:
        best_features = candidate_features_arr[max_prob_idx:max_prob_idx+1]; predicted_offset_px = np.zeros(3)
        if reg_model is not None:
            try:
                 if best_features.shape[0] > 0: predicted_offset_px = reg_model.predict(best_features)[0]
            except Exception as e: print(f"Warn reg_predict: {e}")
        final_predicted_coords_px = best_center_px + predicted_offset_px
        final_predicted_coords_a = final_predicted_coords_px * voxel_spacing
        max_coords_a = (np.array(tomo_shape) - 1) * voxel_spacing
        final_predicted_coords_a = np.clip(final_predicted_coords_a, 0, max_coords_a); return final_predicted_coords_a.tolist()
    else: return [NO_MOTOR_COORD] * 3
    # --- End of original function logic ---

# --- Generate Predictions ONLY for Specific Tomogram IDs ---
print("\nGenerating DEFAULT Predictions for SPECIFIC Tomograms (Models were not trained)...")
predictions = []
test_predict_start_time = time.time()

ids_to_predict = ['tomo_003acc', 'tomo_00e047', 'tomo_00e463']
print(f"Processing ONLY the following {len(ids_to_predict)} tomograms: {ids_to_predict}")

if not ids_to_predict:
     print("No specific tomograms defined to process.")
# No need to check clf_model here, predict_motor_location handles it
else:
    if voxel_spacing_map:
        valid_spacings = [v for v in voxel_spacing_map.values() if v > 0]
        avg_voxel_spacing = np.mean(valid_spacings) if valid_spacings else DEFAULT_VOXEL_SPACING_A
    else: avg_voxel_spacing = DEFAULT_VOXEL_SPACING_A
    print(f"Using average voxel spacing (fallback): {avg_voxel_spacing:.4f} A/px")

    for tomo_idx, tomo_id in enumerate(ids_to_predict):
        print(f"Generating default prediction for {tomo_idx+1}/{len(ids_to_predict)}: {tomo_id}...")
        tomo_dir_path = test_dir_map.get(tomo_id)
        source_dir = "test"
        if not tomo_dir_path:
            tomo_dir_path = train_dir_map.get(tomo_id)
            source_dir = "train"
        if not tomo_dir_path:
            print(f"  Skipping {tomo_id}: Directory not found.")
            pred_coords = [NO_MOTOR_COORD] * 3
            predictions.append({'tomo_id': tomo_id,'Motor axis 0': pred_coords[0],'Motor axis 1': pred_coords[1],'Motor axis 2': pred_coords[2]})
            continue

        # Load data to get shape info, even though model won't use features
        tomo_data = load_tomogram(tomo_id, {tomo_id: tomo_dir_path})
        current_voxel_spacing = voxel_spacing_map.get(tomo_id, avg_voxel_spacing)

        # Call prediction function - it will return default due to clf_model being None
        pred_coords = predict_motor_location(
            tomo_data, current_voxel_spacing, clf_model, reg_model, # Pass None models
            patch_size_px=PATCH_SIZE_PX, n_samples=10, prob_threshold=0.99 # Params don't matter much here
        )
        print(f"  Predicted Coords (Default) for {tomo_id}: {[f'{c:.2f}' for c in pred_coords]}")

        predictions.append({'tomo_id': tomo_id,'Motor axis 0': pred_coords[0],'Motor axis 1': pred_coords[1],'Motor axis 2': pred_coords[2]})
        if tomo_data is not None: del tomo_data; gc.collect()

    print(f"\nFinished default predictions. Total time: {time.time() - test_predict_start_time:.2f} seconds.")

# Create dataframe from the default predictions
if predictions:
    submission_df = pd.DataFrame(predictions)
    print("\nDefault Predictions for the specified IDs:")
    print(submission_df)
else:
    print("\nNo default predictions were generated (check specific IDs).")
    submission_df = pd.DataFrame(columns=['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2'])


# %% [markdown]
# # 8. Create Submission File
#
# Generate the submission file template, ensuring all original test IDs are present.

# %%
print("\nCreating Submission File Template...")
if 'submission_df' in locals():
    try:
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        expected_test_ids = test_ids_found # Use all IDs found in the test directory

        if not expected_test_ids:
             print("Warning: No test IDs were found in the test directory structure.")
             expected_test_ids = sample_sub['tomo_id'].tolist()
             print(f"Using {len(expected_test_ids)} IDs from sample submission as base.")
        else:
             print(f"Generating submission structure based on {len(expected_test_ids)} test IDs found in directory.")

        sub_df_final = pd.DataFrame({'tomo_id': expected_test_ids})

        # Merge the default predictions we generated for the *specific* IDs
        if not submission_df.empty:
             sub_df_final = sub_df_final.merge(submission_df, on='tomo_id', how='left')
        else:
             # If even the default predictions weren't made, fill everything
             for col in ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']: sub_df_final[col] = NO_MOTOR_COORD

        # Fill any remaining NaNs (for test IDs not in the specific prediction list) with default
        sub_df_final.fillna(NO_MOTOR_COORD, inplace=True)

        # Ensure column order and types match sample
        final_columns = sample_sub.columns.tolist()
        for col in final_columns:
             if col not in sub_df_final.columns:
                  print(f"Warning: Column '{col}' from sample not in generated df. Adding default.")
                  sub_df_final[col] = NO_MOTOR_COORD
        sub_df_final = sub_df_final[final_columns]
        for col in ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']:
            sub_df_final[col] = pd.to_numeric(sub_df_final[col], errors='coerce').fillna(NO_MOTOR_COORD).astype(float)

        sub_df_final.to_csv('submission.csv', index=False)
        print("Submission file template created: submission.csv")
        print("\nFinal Submission Head:")
        print(sub_df_final.head())
        print(f"Submission shape: {sub_df_final.shape}")

        if len(sub_df_final) != len(expected_test_ids):
            print(f"CRITICAL WARNING: Submission row count ({len(sub_df_final)}) "
                  f"mismatches expected test IDs ({len(expected_test_ids)})!")
        else:
             print("Submission row count matches expected test IDs.")

    except FileNotFoundError:
        print(f"Error: Sample submission file not found at {SAMPLE_SUB_PATH}. Cannot verify format.")
        # Try to save anyway if df exists
        if 'sub_df_final' in locals():
             sub_df_final.to_csv('submission.csv', index=False)
             print("Submission file template created (format not verified): submission.csv")
    except Exception as e:
        print(f"Error creating final submission file: {e}")
else:
    print("No prediction DataFrame ('submission_df') was generated. Cannot create submission template.")


# %% [markdown]
# # 9. Discussion and Future Work
#
# *   **Data Loading & Handling:** Code structure maintained for loading data, labels, and finding directories.
# *   **Feature Engineering:** Utility functions (`preprocess_tomogram`, `extract_features_patch`) remain defined but were not used for training.
# *   **Modeling:** **Training data generation and model training sections were entirely skipped.** `clf_model` and `reg_model` were explicitly set to `None`.
# *   **Targeted Prediction:** The prediction loop for specific IDs (`tomo_003acc`, `tomo_00e047`, `tomo_00e463`) was executed. However, since the models were `None`, the `predict_motor_location` function returned the default `NO_MOTOR_COORD` for these IDs.
# *   **Submission Generation:** A `submission.csv` file was successfully created containing *all* test IDs found in the `test` directory. The specifically targeted IDs have `NO_MOTOR_COORD` because no model was trained, and all other test IDs were also filled with `NO_MOTOR_COORD`. This serves as a valid submission template.
# *   **Limitations:** This script produces a baseline submission with no actual detection. Its primary purpose is to verify the data loading, directory handling, and submission file formatting logic.
# *   **Future Improvements:** To get actual predictions, the skipping of Sections 5 and 5.1 must be removed, and sufficient training data (likely more than the 20 used in the previous version) needs to be processed to train meaningful models. Then, the prediction pipeline in Section 7 will use the trained models. All other future improvements mentioned previously (CNNs, FCNs, augmentation, etc.) still apply for building a competitive model.

# %%
print("Script finished (Training and Model Fitting Skipped). Submission template generated.")


# -*- coding: utf-8 -*-
"""
Kaggle Notebook: Calculate and Visualize Physics-Inspired Features for Specific Tomograms

Goal: To load specific tomograms ('tomo_003acc', 'tomo_00e047', 'tomo_01a877'),
      extract a patch around their geometric center (as no motor is labeled),
      calculate the physics-inspired features
      (Smoothed, Gradient Mag, ST Coherence, ST λ_max, Hessian λ_min, Blobness),
      and display these calculated features in a 6-panel plot for each tomogram.
"""

# %% [markdown]
# # 1. Setup and Imports
#
# Load necessary libraries for data loading, calculation, and visualization.

# %%
# Attempt installation only if needed (e.g., in Kaggle environment)
try:
    import cv2
    import mrcfile # Although not used for loading slices here
    print("Libraries opencv-python, mrcfile seem available.")
except ImportError:
    print("Installing required libraries: opencv-python, mrcfile")
    %pip install opencv-python mrcfile --quiet
    import cv2
    # import mrcfile # Not strictly needed if only using cv2 for slices
    print("Libraries installed.")

import os
import glob
import numpy as np
import pandas as pd
import cv2 # Import OpenCV
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import structure_tensor, structure_tensor_eigenvalues, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import gaussian
import gc # Garbage collection
import time # For timing

# Constants
COMPETITION_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
TRAIN_DIR = os.path.join(COMPETITION_DIR, 'train')
TEST_DIR = os.path.join(COMPETITION_DIR, 'test') # Include test dir as some IDs might be there
TRAIN_LABELS_PATH = os.path.join(COMPETITION_DIR, 'train_labels.csv')

# Default Voxel spacing (fallback if lookup fails)
DEFAULT_VOXEL_SPACING_A = 10.0

# Target Tomogram IDs (those with no labeled motors)
TARGET_TOMO_IDS = ['tomo_003acc', 'tomo_00e047', 'tomo_01a877']

# Visualization Parameters
PATCH_SIZE_PX_VIS = 96        # Patch size for visualization
SIGMA_SMOOTH = 1.5            # Sigma for initial smoothing
SIGMA_ST = 2.0                # Sigma for Structure Tensor calculations
SIGMA_HESSIAN = 3.0           # Sigma for Hessian calculations

print(f"Target Tomograms for Visualization: {TARGET_TOMO_IDS}")
print(f"Patch Size: {PATCH_SIZE_PX_VIS}x{PATCH_SIZE_PX_VIS}x{PATCH_SIZE_PX_VIS}")

# %% [markdown]
# # 2. Load Metadata and Prepare Directory Maps
#
# Load the labels to get voxel spacing (if available) for our target tomograms. Create maps for both train and test directories.

# %%
print("Loading labels (primarily for voxel spacing)...")
voxel_spacing_map = {}
try:
    train_labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Training labels shape: {train_labels_df.shape}")
    # Create map for Voxel Spacing from labels (use first entry per tomo_id)
    if 'Voxel spacing' in train_labels_df.columns and not train_labels_df.empty:
        voxel_spacing_map = train_labels_df.drop_duplicates(subset='tomo_id').set_index('tomo_id')['Voxel spacing'].to_dict()
        print(f"Loaded voxel spacing for {len(voxel_spacing_map)} tomograms.")
    else:
        print("WARNING: 'Voxel spacing' column not found or labels empty. Will use default spacing.")

except FileNotFoundError:
    print(f"ERROR: Training labels file not found at {TRAIN_LABELS_PATH}. Will use default spacing.")
    train_labels_df = pd.DataFrame()

# --- Find Tomogram Directories ---
print("\nFinding tomogram directories...")
train_dirs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'tomo_*')))
test_dirs = sorted(glob.glob(os.path.join(TEST_DIR, 'tomo_*')))

train_dir_map = {os.path.basename(d): d for d in train_dirs if os.path.isdir(d)}
test_dir_map = {os.path.basename(d): d for d in test_dirs if os.path.isdir(d)}

# Combine maps for easier lookup, prioritizing test if duplicates exist (unlikely)
full_dir_map = {**train_dir_map, **test_dir_map}

print(f"Found {len(train_dir_map)} potential training directories.")
print(f"Found {len(test_dir_map)} potential testing directories.")
print(f"Total unique directories mapped: {len(full_dir_map)}")


# %% [markdown]
# # 3. Utility Functions (Load Tomogram, Extract Patch)

# %%
# --- Tomogram Loading Function ---
def load_tomogram(tomo_id, dir_map):
    """Loads a tomogram by reading and stacking slices from a directory."""
    directory_path = dir_map.get(tomo_id)
    if not directory_path or not os.path.isdir(directory_path):
        print(f"Warning: Tomogram directory for {tomo_id} not found in provided map.")
        return None
    slice_files = sorted(glob.glob(os.path.join(directory_path, 'slice_*.jpg')))
    if not slice_files: slice_files = sorted(glob.glob(os.path.join(directory_path, 'slice_*.png')))
    if not slice_files: slice_files = sorted(glob.glob(os.path.join(directory_path, 'slice_*.tif')))
    if not slice_files: print(f"Warning: No slice files found in {directory_path}."); return None
    def get_slice_number(filepath):
        try: return int(os.path.basename(filepath).split('slice_')[1].split('.')[0])
        except: return -1
    valid_slice_files = [(f, get_slice_number(f)) for f in slice_files if get_slice_number(f) != -1]
    if not valid_slice_files: print(f"Warning: No valid slice numbers found in {directory_path}."); return None
    valid_slice_files.sort(key=lambda item: item[1]); slice_files_sorted = [f for f, num in valid_slice_files]
    try:
        first_slice = cv2.imread(slice_files_sorted[0], cv2.IMREAD_GRAYSCALE)
        if first_slice is None: raise IOError("imread failed")
        height, width = first_slice.shape
        num_slices = len(slice_files_sorted); dtype = first_slice.dtype
    except Exception as e: print(f"Error reading first slice {slice_files_sorted[0]}: {e}"); return None
    tomogram_data = np.zeros((num_slices, height, width), dtype=dtype); tomogram_data[0, :, :] = first_slice
    for i in range(1, num_slices):
        try:
            slice_img = cv2.imread(slice_files_sorted[i], cv2.IMREAD_GRAYSCALE)
            if slice_img is None: raise IOError("imread failed")
            if slice_img.shape != (height, width): print("Shape mismatch"); return None
            tomogram_data[i, :, :] = slice_img
        except Exception as e: print(f"Error reading slice {slice_files_sorted[i]}: {e}"); return None
    print(f"  Successfully loaded {tomo_id}, shape: {tomogram_data.shape}")
    return tomogram_data

# --- Patch Extraction Function ---
def get_local_patch(tomo_data, center_px, patch_size_px=64):
    """Extracts a 3D patch centered at center_px. Patch size is in pixels."""
    if tomo_data is None or center_px is None: return None, None
    center_px = np.round(center_px).astype(int); z, y, x = center_px
    shape = tomo_data.shape
    if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
        print(f"Warning: Center px {center_px} out of bounds {shape}. Cannot extract patch.")
        return None, None
    half_size = patch_size_px // 2
    z_start, z_end = max(0, z - half_size), min(shape[0], z + half_size)
    y_start, y_end = max(0, y - half_size), min(shape[1], y + half_size)
    x_start, x_end = max(0, x - half_size), min(shape[2], x + half_size)
    dest_z_start = half_size - (z - z_start); dest_z_end = dest_z_start + (z_end - z_start)
    dest_y_start = half_size - (y - y_start); dest_y_end = dest_y_start + (y_end - y_start)
    dest_x_start = half_size - (x - x_start); dest_x_end = dest_x_start + (x_end - x_start)
    patch = np.zeros((patch_size_px, patch_size_px, patch_size_px), dtype=tomo_data.dtype)
    try:
        src_slice = (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))
        dest_slice = (slice(dest_z_start, dest_z_end), slice(dest_y_start, dest_y_end), slice(dest_x_start, dest_x_end))
        patch[dest_slice] = tomo_data[src_slice]
        # Calculate center coordinates *within the patch* (should be near half_size)
        patch_center_coords = np.round([z - z_start + dest_z_start, y - y_start + dest_y_start, x - x_start + dest_x_start]).astype(int)
        if not (0 <= patch_center_coords[0] < patch_size_px and 0 <= patch_center_coords[1] < patch_size_px and 0 <= patch_center_coords[2] < patch_size_px):
             patch_center_coords = [patch_size_px // 2] * 3 # Fallback
        print(f"  Extracted patch shape: {patch.shape}, Center within patch: {patch_center_coords}")
        return patch, patch_center_coords
    except (ValueError, IndexError) as e: print(f"Patch extraction error: {e}"); return None, None


# %% [markdown]
# # 4. Process Each Target Tomogram
#
# Loop through the specified IDs, load data, extract a central patch, calculate features, and visualize.

# %%
for tomo_id in TARGET_TOMO_IDS:
    print(f"\n--- Processing Tomogram: {tomo_id} ---")

    # --- Load Tomogram Data ---
    tomo_data = load_tomogram(tomo_id, full_dir_map)
    if tomo_data is None:
        print(f"  Skipping {tomo_id} due to loading error.")
        continue

    # --- Get Geometric Center and Voxel Spacing ---
    tomo_shape = tomo_data.shape
    geometric_center_px = np.array([tomo_shape[0] // 2, tomo_shape[1] // 2, tomo_shape[2] // 2])
    voxel_spacing = voxel_spacing_map.get(tomo_id, DEFAULT_VOXEL_SPACING_A)
    print(f"  Geometric Center (px): {geometric_center_px}")
    print(f"  Voxel Spacing (A/px): {voxel_spacing}")

    # --- Extract Central Patch ---
    print(f"  Extracting {PATCH_SIZE_PX_VIS}x{PATCH_SIZE_PX_VIS} patch around geometric center...")
    central_patch, patch_center_coords = get_local_patch(tomo_data, geometric_center_px, patch_size_px=PATCH_SIZE_PX_VIS)

    # Free memory of full tomogram
    del tomo_data; gc.collect()

    if central_patch is None:
        print(f"  Skipping {tomo_id} because patch extraction failed.")
        continue

    # --- Calculate Features ---
    patch_smooth = grad_mag = st_coherence = st_lambda_max = hessian_lambda_min = blobness = None
    calculation_successful = False
    print("\n  Calculating features on the central patch...")
    calc_start_time = time.time()
    try:
        if not np.issubdtype(central_patch.dtype, np.floating):
            patch_float = central_patch.astype(np.float32)
        else:
            patch_float = central_patch

        patch_smooth = gaussian(patch_float, sigma=SIGMA_SMOOTH, mode='reflect', preserve_range=True, truncate=4.0)
        grad_z, grad_y, grad_x = np.gradient(patch_smooth)
        grad_mag = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)

        S_elems = structure_tensor(patch_smooth, sigma=SIGMA_ST, mode='reflect')
        eigvals_S = structure_tensor_eigenvalues(S_elems)
        eigvals_S = np.sort(eigvals_S, axis=0)
        lambda3, lambda2, lambda1 = eigvals_S[0], eigvals_S[1], eigvals_S[2]
        denominator = lambda1 + lambda2 + lambda3 + 1e-9
        st_coherence = np.where(denominator > 1e-8, (lambda1 - lambda3) / denominator, 0.0)
        st_lambda_max = lambda1

        h_matrix = hessian_matrix(patch_smooth, sigma=SIGMA_HESSIAN, mode='reflect', use_gaussian_derivatives=False)
        eigvals_H = hessian_matrix_eigvals(h_matrix)
        eigvals_H = np.sort(eigvals_H, axis=0)
        h_lambda1, h_lambda2, h_lambda3 = eigvals_H[0], eigvals_H[1], eigvals_H[2]
        hessian_lambda_min = h_lambda1
        blobness = np.abs(h_lambda1 * h_lambda2 * h_lambda3) * (h_lambda1 < 0) * (h_lambda2 < 0) * (h_lambda3 < 0)

        calculation_successful = True
        print(f"  Feature calculation took {time.time() - calc_start_time:.2f} seconds.")

    except Exception as e:
        print(f"  ERROR during feature calculation for {tomo_id}: {e}")
        calculation_successful = False

    # --- Visualize Features ---
    if calculation_successful and patch_center_coords is not None:
        print("\n  Generating visualization...")

        # Use the Z-coordinate of the center *within the patch* for slicing
        center_z_in_patch = patch_center_coords[0]
        center_y_in_patch = patch_center_coords[1]
        center_x_in_patch = patch_center_coords[2]

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"Calculated Features for {tomo_id} (Central Patch, Z-Slice={center_z_in_patch})", fontsize=16)

        # Panel 1: Smoothed
        ax = axes[0, 0]; im = ax.imshow(patch_smooth[center_z_in_patch, :, :], cmap='gray'); ax.set_title(f'Smoothed (Z={center_z_in_patch})'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # No marker plotted

        # Panel 2: Gradient Magnitude
        ax = axes[0, 1]; im = ax.imshow(grad_mag[center_z_in_patch, :, :], cmap='magma'); ax.set_title('Gradient Mag'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 3: ST Coherence
        ax = axes[0, 2]; im = ax.imshow(st_coherence[center_z_in_patch, :, :], cmap='viridis', vmin=0, vmax=1); ax.set_title('ST Coherence'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 4: ST Lambda_max
        ax = axes[1, 0]; im = ax.imshow(st_lambda_max[center_z_in_patch, :, :], cmap='plasma'); ax.set_title('ST $\lambda_{max}$'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 5: Hessian Lambda_min
        ax = axes[1, 1]; vmin_h = np.percentile(hessian_lambda_min, 1); vmax_h = np.percentile(hessian_lambda_min, 99); im = ax.imshow(hessian_lambda_min[center_z_in_patch, :, :], cmap='coolwarm', vmin=vmin_h, vmax=vmax_h); ax.set_title('Hessian $\lambda_{min}$'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 6: Blobness
        ax = axes[1, 2]; vmax_b = np.percentile(blobness[blobness > 0], 99) if np.any(blobness > 0) else 1.0; im = ax.imshow(blobness[center_z_in_patch, :, :], cmap='cubehelix', vmin=0, vmax=vmax_b); ax.set_title('Blobness'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    elif central_patch is None:
        print(f"  Could not visualize features for {tomo_id} because patch extraction failed earlier.")
    else:
        print(f"  Could not visualize features for {tomo_id} because calculation failed.")

    # Clean up memory before next loop iteration
    del central_patch, patch_smooth, grad_mag, st_coherence, st_lambda_max, hessian_lambda_min, blobness
    gc.collect()


# %% [markdown]
# # 5. Discussion
#
# This notebook processed the tomograms `tomo_003acc`, `tomo_00e047`, and `tomo_01a877`. Since these tomograms did not have labeled motor coordinates in the provided `train_labels.csv` file, the analysis focused on a patch extracted from the geometric center of each volume.
#
# For each tomogram, the six physics-inspired features were calculated on this central patch and visualized. The plots show the characteristics of the cellular environment at the center of these specific tomograms according to each feature:
#
# *   **Smoothed:** Shows the general density and texture at the center.
# *   **Gradient Mag:** Highlights edges or textures present in the central region.
# *   **ST Coherence & λ_max:** Reveal the degree of orientation and anisotropy of structures at the center.
# *   **Hessian λ_min & Blobness:** Indicate the local curvature and presence of any blob-like structures near the center.
#
# Unlike visualizations centered on a known motor, these plots provide insight into the "background" characteristics captured by these features in tomograms potentially lacking the target structure (or where it wasn't labeled). This can be useful for understanding feature responses away from the target.

# %%
print("Notebook execution finished.")


import math # For sqrt symbol, though not strictly needed for text output

# Define a helper function for clarity (optional)
def print_concept(title, concept, equation, usability, equation2=None):
    """Prints the concept, equation, and usability in a standard format."""
    print(f"\n--- {title} ---")
    print(f"Concept: {concept}")
    print(f"General Equation: {equation}")
    if equation2:
        print(f"Derived/Related Equation: {equation2}")
    print(f"Usability: {usability}")

# --- Print Explanations ---

print("=== Conceptual Overview of Physics-Inspired Tomogram Features ===")

print_concept(
    title="1. Intensity: The Base Scalar Field (Φ)",
    concept="The fundamental data assigning a density/intensity value to every 3D point.",
    equation="I_s(x, y, z)  (Intensity, typically after smoothing)",
    usability="Provides the raw density map. Smoothing creates a stable base for derivatives."
)

print_concept(
    title="2. Gradient: Rate and Direction of Change (E)",
    concept="A vector field pointing in the direction of steepest intensity increase.",
    equation="∇I_s = (∂I_s/∂x, ∂I_s/∂y, ∂I_s/∂z)",
    equation2="|∇I_s| = sqrt[ (∂I_s/∂x)² + (∂I_s/∂y)² + (∂I_s/∂z)² ]  (Magnitude)",
    usability="Magnitude (|∇I_s|) detects edges and textures. Direction indicates orientation perpendicular to iso-surfaces."
)

print_concept(
    title="3. Structure Tensor: Local Orientation and Anisotropy (T_μν / S_μν)",
    concept="Tensor summarizing predominant gradient orientations via local averaging.",
    # Using <>_sigma notation for averaging over scale sigma
    equation="ST = <∇I_s ⊗ ∇I_s^T>_σ  (Locally averaged outer product of gradient)",
    # Explain eigenvalues and coherence conceptually
    equation2="Derived features based on Eigenvalues (λ₁≥λ₂≥λ₃≥0): λ_max=λ₁, Coherence C ≈ (λ₁-λ₃)/(λ₁+λ₂+λ₃)",
    usability="Describes local geometry: discriminates lines vs. planes vs. isotropic regions using eigenvalues. Coherence measures degree/strength of orientation."
)

print_concept(
    title="4. Hessian Matrix: Local Curvature",
    concept="Matrix of second-order partial derivatives describing the local curvature of the intensity landscape.",
    equation="H_ij = ∂²I_s / ∂xᵢ ∂xⱼ",
    # Explain eigenvalues and blobness conceptually
    equation2="Derived features based on Eigenvalues (h₁, h₂, h₃): λ_min=h₁, Blobness B ≈ |h₁h₂h₃| if all h<0",
    usability="Classifies local shape (blob, tube, sheet) via eigenvalue signs/magnitudes. λ_min helps find centers of dark structures. Blobness specifically highlights blob-like regions."
)

print("\n==============================================================")
print("Notebook execution finished. Conceptual explanations printed.")

