# This directory will store all our downloaded package files
!mkdir -p /kaggle/working/pip_packages

# --- Step 1: Download the specific, compatible version of PyTorch ---
# Version 2.6.0 is compatible with the pre-installed Kaggle libraries.
# We also download torchvision and torchaudio that match this version.
print("Downloading compatible PyTorch version...")
!pip download torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu121 -d /kaggle/working/pip_packages

# --- Step 2: Download PyTorch Geometric ---
# pip will automatically find a version of torch_geometric compatible with torch==2.6.0
# and also download its dependencies (like torch-scatter, torch-sparse, etc.)
print("\nDownloading PyTorch Geometric...")
!pip download torch_geometric -d /kaggle/working/pip_packages

# --- Step 3: Download Optuna and LightGBM ---
print("\nDownloading Optuna and LightGBM...")
!pip download optuna lightgbm -d /kaggle/working/pip_packages

print("\nAll packages downloaded successfully.")




