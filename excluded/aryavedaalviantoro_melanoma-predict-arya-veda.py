import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import time

NOTEBOOK_START_TIME = time.time()
# Set maximum runtime in seconds (11 hours 25 minutes to leave buffer)
MAX_RUNTIME_SECONDS = 11 * 3600 + 25 * 60
print(f"Notebook execution started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(NOTEBOOK_START_TIME))}")
print(f"Maximum training runtime set to: {MAX_RUNTIME_SECONDS / 3600:.2f} hours ({MAX_RUNTIME_SECONDS} seconds)")


!nvidia-smi


import subprocess
import sys

def install_package(package, upgrade=False):
    """Installs a package, suppressing stdout and stderr."""
    try:
        command = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            command.append("--upgrade")
        command.append(package)

        # Redirect stdout and stderr to /dev/null (or equivalent)
        with open('/dev/null', 'w') as devnull:
            subprocess.check_call(command, stdout=devnull, stderr=devnull)
        print(f"Successfully installed/upgraded: {package}") # Inform user
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}", file=sys.stderr)
    except Exception as e:  # Catch other potential exceptions
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

# Example usage:  (replace with your package list)
install_package("git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git")
install_package("geffnet")
install_package("albumentations", upgrade=True) # Example with upgrade
install_package("wandb")
install_package("opencv-python")
install_package("pytz")
install_package("timm", upgrade=True)
# Add near your other install commands
install_package("grad-cam")   
install_package("ttach") # pytorch-grad-cam sometimes uses this

print("All packages installed.")


# Standard Libraries
import os
import time
import warnings
import logging
import subprocess
import traceback
from datetime import datetime
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# Data Handling and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image
import cv2
import re
# Machine Learning and Metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from scipy.stats import wilcoxon

# Deep Learning Frameworks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import (
    TensorDataset, DataLoader, Dataset, RandomSampler, SubsetRandomSampler,
    SequentialSampler, WeightedRandomSampler
)
from torch.amp import autocast as amp_autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import math
# Model Architectures and Utilities
import timm
from timm import create_model
import geffnet
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, SwinConfig, SwinModel

# Learning Rate Schedulers
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
# Distributed Training
from accelerate import Accelerator, notebook_launcher
from torch.utils.data.distributed import DistributedSampler

# Metrics and Evaluation
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassConfusionMatrix,
    BinaryAUROC
)
from torchmetrics.functional.classification import binary_accuracy, binary_f1_score

# Image Augmentation
import albumentations as A

from skimage.segmentation import slic

# Timezone Handling
import pytz
from typing import Dict, Optional, Union

# Experiment Tracking
import wandb
%matplotlib inline
device = torch.device('cuda')
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
my_secret = user_secrets.get_secret("wandb_api_key") 
wandb.login(key=my_secret)


import re

# --- Core Run Configuration ---
# SET THE CONFIGURATION FOR THIS SPECIFIC RUN:
model_type = 'swin_only' # Options: 'efficientnetv2', 'hybrid_swin', 'swin_only'
# Use the torch.hub names now for V2
cnn_backbone_name = 'efficientnet_v2_m' # e.g., 'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'
# Use the TIMM name for Swin
transformer_backbone_name = 'swin_base_patch4_window12_384.ms_in22k_ft_in1k' # Keep this for hybrid
use_amp = True
use_external = True
use_meta = True
DEBUG = False
target_total_epochs = 30 
# --- Standard Settings ---
num_workers = 4
init_lr = 1e-3 # Base LR

# --- REMOVED Pretrained File Settings ---
# model_dir = '../input/melanoma-winning-models'
# pretrained_type = ...
# i_fold = ...
# model_file = ...

print(f"Run Config: Model={model_type}, CNN={cnn_backbone_name}, UseMeta={use_meta}, UseExt={use_external}")

# --- Helper Function ---
def get_v2_size_map(v2_name_str):
    if 'v2_s' in v2_name_str: return 's'
    if 'v2_m' in v2_name_str: return 'm'
    if 'v2_l' in v2_name_str: return 'l'
    print(f"Warning: Could not map V2 size from '{v2_name_str}'. Defaulting to 'm'.")
    return 'm'

# --- Determine V2 Size and TARGET Image Size ---
cnn_size_tag = get_v2_size_map(cnn_backbone_name)
target_image_sizes_map = {'s': 384, 'm': 448, 'l': 480}
image_size = 384
# --- CORRECTED LINE ---
if model_type in ['hybrid_swin', 'swin_only'] and '384' in transformer_backbone_name:
# --- END CORRECTION ---
        print(f"NOTE: Forcing image_size to 384 for Swin model compatibility.")
        image_size = 384
print(f"CNN Size Tag: {cnn_size_tag}, Target Image Size: {image_size}x{image_size}")

# --- Generate kernel_type ---
kernel_base = f"{cnn_backbone_name.replace('_','').replace('efficientnet','')}" # e.g., effnetv2m
if model_type == 'hybrid_swin': kernel_base += f"_{transformer_backbone_name.split('_')[0]}"
kernel_type = f"{kernel_base}_{image_size}{'_ext' if use_external else ''}{'_meta' if use_meta else ''}"
print(f"Generated kernel_type: {kernel_type}")

# --- Determine Data Directory Size ---
available_data_sizes = [256, 384, 512]; data_dir_size = 384 if image_size <= 384 else 512
print(f"Selected Data Directory Size: {data_dir_size}x{data_dir_size}")

# --- Set Data Directory Paths ---
data_dir = f'../input/jpeg-melanoma-{data_dir_size}x{data_dir_size}'
data_dir2 = f'../input/jpeg-isic2019-{data_dir_size}x{data_dir_size}' if use_external else None
print(f"Comp Data Dir: {data_dir}" + (f", Ext Data Dir: {data_dir2}" if data_dir2 else ""))

# --- Dynamic Variables Note ---
print("Note: Batch size, accum steps, epoch phases are set dynamically.")


def configure_dynamic_parameters(model_type, cnn_backbone_name, use_meta, use_external):
    """
    Calculates base scaling factors based ONLY on CNN size ('s', 'm', 'l').
    Adaptations based on meta, external data, or hybrid type are REMOVED
    to ensure consistent LR/Regularization multipliers across runs for purity.
    Returns factors that will be applied later (e.g., in LLRD, WD, Dropout init).
    """
    cnn_size_tag = get_v2_size_map(cnn_backbone_name)
    print(f"Configuring BASE scaling factors based on CNN size ONLY for: type='{model_type}', cnn='{cnn_backbone_name}' ({cnn_size_tag}), meta={use_meta}, external={use_external}")
    print("  NOTE: Multipliers for meta, external, hybrid are DISABLED for purity.")

    # --- Base Scales based ONLY on CNN Size ---
    # Still apply a slight adjustment based on model scale, as this is common practice.
    size_to_b_map = {'s': 3, 'm': 5, 'l': 7}
    equiv_b_version = size_to_b_map.get(cnn_size_tag, 5)
    # Base LR factor: Slightly lower for S, slightly higher for L relative to M (applied to base_lr later)
    lr_base_scale = { 3: 0.9, 5: 1.0, 7: 1.1 } # Adjusted slightly: s: 0.9, m: 1.0, l: 1.1
    # Base Regularization factor: Slightly higher for S, slightly lower for L (applied to WD/Dropout later)
    reg_base_scale = { 3: 1.05, 5: 1.0, 7: 0.95 } # Adjusted slightly: s: 1.05, m: 1.0, l: 0.95
    
    current_base_lr_factor = lr_base_scale.get(equiv_b_version, 1.0)
    current_base_reg_factor = reg_base_scale.get(equiv_b_version, 1.0)
    print(f"  Base factors based on CNN size '{cnn_size_tag}': LR Factor={current_base_lr_factor:.2f}, Reg Factor={current_base_reg_factor:.2f}")

    # --- Combine (Now only uses base factors) ---
    final_lr_factor = current_base_lr_factor # * 1.0 (removed multiplier)
    final_reg_factor = current_base_reg_factor # * 1.0 (removed multiplier)

    # Clamp to reasonable bounds (still useful for the base factors)
    final_lr_factor = max(0.5, min(final_lr_factor, 1.5)) # Clamp LR factor bounds if needed
    final_reg_factor = max(0.5, min(final_reg_factor, 1.5)) # Clamp Reg factor bounds if needed

    # Rename variables for clarity (these factors modify base LR/WD/Dropout later)
    lr_adaptation_factor = round(final_lr_factor, 6)
    regularization_factor = round(final_reg_factor, 6)

    print(f"  ==> Final Factors (Based ONLY on CNN Size): LR Adapt Factor={lr_adaptation_factor}, Reg Factor={regularization_factor}")

    # Return only the two factors
    return lr_adaptation_factor, regularization_factor


# Load test data
df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, 'test', f'{x}.jpg'))
print("Test Data Loaded - Shape:", df_test.shape)
print("Test Data Sample:\n", df_test.head())


# Load train data and filter
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
print("Initial Train Data Shape:", df_train.shape)
df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)
print("Train Data Shape after tfrecord filter:", df_train.shape)
df_train['is_ext'] = 0
df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, 'train', f'{x}.jpg'))

# Clean diagnosis labels
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
df_train['diagnosis'] = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))


# Add external data if enabled
if use_external:
    df_train2 = pd.read_csv(os.path.join(data_dir2, 'train.csv'))
    print("External Train Data Shape:", df_train2.shape)
    df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
    df_train2['is_ext'] = 1
    df_train2['filepath'] = df_train2['image_name'].apply(lambda x: os.path.join(data_dir2, 'train', f'{x}.jpg'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    print("External Data Diagnosis Unique:", df_train2['diagnosis'].unique())
    
    # Combine datasets
    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)
    print("Combined Train Data Shape:", df_train.shape)
    
# Add assertion to ensure data isn't empty
assert not df_train.empty, "Error: df_train is empty after preparation!"
assert 'diagnosis' in df_train.columns, "Error: 'diagnosis' column missing in df_train!"

# Map diagnosis to target indices
diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
mel_idx = diagnosis2idx['melanoma']
print("Diagnosis to Index Mapping:", diagnosis2idx)
print("Target Value Counts:\n", df_train['target'].value_counts())

# Dynamically set out_dim
out_dim = len(df_train['target'].unique())
print(f"Number of unique classes (out_dim): {out_dim}")
print(f"Melanoma index (mel_idx): {mel_idx}")

# Final assertions to verify critical variables
assert mel_idx in df_train['target'].values, f"Error: mel_idx ({mel_idx}) not found in target values!"
assert out_dim > 1, "Error: out_dim is 1 or less, indicating no class variation!"


# Class distribution
class_counts = df_train['diagnosis'].value_counts()
total_samples = len(df_train)
class_percentages = (class_counts / total_samples) * 100

plt.figure(figsize=(12, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title('Distribution of Diagnosis Classes', fontsize=16)
plt.xlabel('Diagnosis', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
for i, count in enumerate(class_counts):
    plt.text(i, count + 0.5, f'{class_percentages[i]:.2f}%', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
plt.show()


from tqdm import tqdm  # Ensure this import is present

if use_meta:
    # One-hot encoding of anatom_site_general_challenge feature
    print("One-hot encoding 'anatom_site_general_challenge'...")
    concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
    
    # Sex features
    print("Encoding 'sex' feature...")
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)
    
    # Age features
    print("Normalizing 'age_approx' feature...")
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
        
    # Patient ID features
    print("Handling 'patient_id' feature...")
    df_train['patient_id'] = df_train['patient_id'].fillna(0)
    
    # n_images per user
    print("Calculating 'n_images' per patient...")
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)
    
    # Image size
    print("Calculating image sizes...")
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images, desc="Processing training images", unit="image")):
        train_sizes[i] = os.path.getsize(img_path)
    df_train['image_size'] = np.log(train_sizes)
    
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images, desc="Processing test images", unit="image")):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)
    # Improved age normalization
    mean_age = df_train['age_approx'].mean()
    std_age = df_train['age_approx'].std()
    df_train['age_approx'] = (df_train['age_approx'].fillna(mean_age) - mean_age) / std_age
    df_test['age_approx'] = (df_test['age_approx'].fillna(mean_age) - mean_age) / std_age
    
    # Log-transformed features standardization
    df_train['n_images'] = (df_train['n_images'] - df_train['n_images'].mean()) / df_train['n_images'].std()
    df_test['n_images'] = (df_test['n_images'] - df_train['n_images'].mean()) / df_train['n_images'].std()
    # Meta features
    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)
    print(f"Meta features created: {meta_features}")
else:
    n_meta_features = 0
    print("Meta features disabled.")


n_meta_features


# Display a random sample of 5 rows
print(df_train.sample(5))


class SIIMISICDataset(Dataset):
    def __init__(self, csv, split, mode, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()


import albumentations as A # Use A alias if used elsewhere, or albumentations directly

print(f"Defining standard training and validation transforms using image_size: {image_size}")

transforms_train = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    # A.RandomBrightness(limit=0.2, p=0.75), # Older version uses this
    # A.RandomContrast(limit=0.2, p=0.75),  # Older version uses this
    A.RandomBrightnessContrast(limit=0.2, p=0.75), # Newer version combines these

    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.7),

    A.CLAHE(clip_limit=4.0, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.Resize(image_size, image_size), # Ensure image_size is defined before this cell
    # A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7), # Older Cutout
    A.CoarseDropout(max_holes=1, max_height=int(image_size * 0.375), max_width=int(image_size * 0.375),
                    min_holes=1, min_height=int(image_size * 0.1), min_width=int(image_size * 0.1), # Define min sizes
                    fill_value=0, p=0.7), # Newer CoarseDropout is preferred replacement for Cutout
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # Standard normalization
])

transforms_val = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

print("Standard transforms_train and transforms_val defined.")


# Minimal transform for original image (just resize and normalize)
transforms_original = A.Compose([
    A.Resize(image_size, image_size),  # Match augmented image size
    A.Normalize()
])


# Pastikan df_train sudah ada dan memiliki kolom 'filepath'
# Contoh load jika belum ada (SESUAIKAN PATH)
if 'df_train' not in locals():
     print("Warning: 'df_train' not found. Attempting to load example data structure.")
     # Ganti path ini sesuai lokasi data Anda jika menjalankan di luar Kaggle
     data_dir_example = '../input/jpeg-melanoma-384x384' # Ganti jika perlu
     train_csv_path = os.path.join(data_dir_example, 'train.csv')
     if os.path.exists(train_csv_path):
         df_train = pd.read_csv(train_csv_path)
         # Buat kolom filepath jika belum ada
         if 'filepath' not in df_train.columns:
              image_folder = os.path.join(data_dir_example, 'train')
              df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(image_folder, f'{x}.jpg'))
         df_train = df_train.dropna(subset=['filepath']).reset_index(drop=True) # Hapus baris jika path tidak valid
         print(f"Loaded df_train with {len(df_train)} samples.")
     else:
         print(f"Error: Train CSV not found at {train_csv_path}. Cannot proceed with visualization.")
         df_train = pd.DataFrame() # Buat dataframe kosong

import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random
import os
import pandas as pd
import sys

# --- Konfigurasi & Prasyarat (Sama seperti sebelumnya) ---
# ... (pastikan image_size, transforms_train, df_train didefinisikan) ...

# --- Fungsi Denormalisasi (Sama seperti sebelumnya) ---
def denormalize(img_array, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Reverses the normalization applied by A.Normalize for visualization."""
    std_array = np.array(std).reshape(1, 1, 3)
    mean_array = np.array(mean).reshape(1, 1, 3)
    std_array[std_array == 0] = 1e-6 
    img_denorm = (img_array * std_array) + mean_array
    img_denorm = np.clip(img_denorm, 0, 1)
    return img_denorm

# --- Visualisasi ---
NUM_AUG_EXAMPLES = 2 # Anda bisa tetap 4 atau sesuaikan
FIG_WIDTH_PER_PLOT = 6 # Lebar per subplot
FIG_HEIGHT = 7         # Tinggi figure keseluruhan (lebih tinggi)

if 'df_train' in locals() and not df_train.empty:
    try:
        sample_row = df_train.sample(1).iloc[0]
        image_path = sample_row['filepath']
        image_name = sample_row.get('image_name', os.path.basename(image_path))

        print(f"Loading image: {image_path}")
        original_img_bgr = cv2.imread(image_path)

        if original_img_bgr is None:
            print(f"Error: Could not read image at {image_path}", file=sys.stderr)
        else:
            original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
            original_img_resized = cv2.resize(original_img_rgb, (image_size, image_size))
            original_img_display = original_img_resized / 255.0

            # Buat figure dengan ukuran yang disesuaikan
            fig, axes = plt.subplots(1, NUM_AUG_EXAMPLES + 1,
                                     figsize=(FIG_WIDTH_PER_PLOT * (NUM_AUG_EXAMPLES + 1), FIG_HEIGHT))
            plt.suptitle(f"Augmentation Examples for: {image_name}", fontsize=18, y=0.98) # Sesuaikan posisi Y

            # Tampilkan gambar asli
            axes[0].imshow(original_img_display)
            axes[0].set_title("Original (Resized)", fontsize=12)
            axes[0].axis('off')

            # Terapkan augmentasi dan tampilkan
            for i in range(NUM_AUG_EXAMPLES):
                augmented_data = transforms_train(image=original_img_rgb)
                augmented_img_normalized = augmented_data['image']
                augmented_img_display = denormalize(augmented_img_normalized)

                axes[i+1].imshow(augmented_img_display)

                # Generate caption (sama seperti sebelumnya)
                potential_transforms = []
                if random.random() < 0.5: potential_transforms.append("Flip H")
                if random.random() < 0.5: potential_transforms.append("Flip V")
                if random.random() < 0.5: potential_transforms.append("Transpose")
                if random.random() < 0.85: potential_transforms.append("Shift/Scale/Rotate")
                if random.random() < 0.75: potential_transforms.append("Bright/Contrast") # Persingkat
                if random.random() < 0.5: potential_transforms.append("HSV")
                if random.random() < 0.7: potential_transforms.append("CLAHE")
                if random.random() < 0.7: potential_transforms.append("Blur/Noise")
                if random.random() < 0.7: potential_transforms.append("Distortion")
                if random.random() < 0.7: potential_transforms.append("Dropout")

                caption = f"Augmented {i+1}"
                if potential_transforms:
                   # Batasi jumlah item di caption jika terlalu panjang
                   caption += "\n(" + ", ".join(potential_transforms[:4]) + ("..." if len(potential_transforms)>4 else "") + ")"
                else:
                   caption += "\n(No major transforms)"

                axes[i+1].set_title(caption, fontsize=10) # Ukuran font judul subplot
                axes[i+1].axis('off')

            # Gunakan subplots_adjust untuk spasi
            # Sesuaikan nilai wspace (horizontal), hspace (vertical), top, bottom, left, right
            plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.85, bottom=0.05, left=0.05, right=0.95)

            plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {image_path}. Ensure dataset path is correct.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during visualization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
else:
    print("Skipping visualization because df_train is not loaded or is empty.")


# # Create datasets
# df_show = df_train.sample(min(1000, len(df_train)))  # Sample dataset
# dataset_original = SIIMISICDataset(df_show, 'train', 'train', transform=transforms_original)  # Original images
# dataset_augmented = SIIMISICDataset(df_show, 'train', 'train', transform=transforms_train)   # Augmented images

# # Reverse mapping for labels (assume diagnosis2idx is defined earlier)
# idx2diagnosis = {v: k for k, v in diagnosis2idx.items()}

# # Display original and augmented images side by side
# from pylab import rcParams
# rcParams['figure.figsize'] = 20, 10

# for i in range(2):  # Show 2 rows
#     f, axarr = plt.subplots(2, 5)  # 2 rows: original (top), augmented (bottom); 5 columns
#     for p in range(5):
#         idx = np.random.randint(0, len(dataset_original))  # Same index for both datasets
        
#         # Original image
#         img_original, label_tensor = dataset_original[idx]
#         if use_meta:
#             img_original = img_original[0]  # Extract image tensor if metadata is used
#         label_idx = label_tensor.item()
#         label_name = idx2diagnosis[label_idx]
        
#         # Augmented image
#         img_augmented, _ = dataset_augmented[idx]  # Same index, ignore label since it’s identical
#         if use_meta:
#             img_augmented = img_augmented[0]

#         # Plot original (top row)
#         axarr[0, p].imshow(img_original.transpose(0, 1).transpose(1, 2).squeeze())
#         axarr[0, p].set_title(f"Original: {label_name}")
#         axarr[0, p].axis('off')

#         # Plot augmented (bottom row)
#         axarr[1, p].imshow(img_augmented.transpose(0, 1).transpose(1, 2).squeeze())
#         axarr[1, p].set_title(f"Augmented: {label_name}")
#         axarr[1, p].axis('off')

#     plt.tight_layout()
#     plt.show()


# --- Torchvision EfficientNet Implementation (Corrected Syntax) ---
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Tuple
import math
import copy

# --- Essential Dependencies from Torchvision ---
# Make sure torchvision is installed (`pip install torchvision`)
try:
    from torchvision.ops import StochasticDepth
    from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
    from torchvision.models._api import WeightsEnum, Weights
    from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
    from torch.hub import load_state_dict_from_url
    print("Successfully imported required components from torchvision.")
except ImportError as e:
    print(f"ERROR: Failed to import components from torchvision: {e}")
    print("Please ensure torchvision is installed and compatible.")
    # Raising an error might be better than continuing with non-functional dummies
    raise ImportError("Torchvision components required for EfficientNet definition are missing.") from e

# Utility function (ensure this is defined correctly)
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None: min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v: new_v += divisor
    return new_v

# --- Core Configuration Classes ---
@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int]=None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

class MBConvConfig(_MBConvConfig):
    def __init__(self, expand_ratio: float, kernel: int, stride: int, input_channels: int,
                 out_channels: int, num_layers: int, width_mult: float = 1.0,
                 depth_mult: float = 1.0, block: Optional[Callable[..., nn.Module]] = None):
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))

class FusedMBConvConfig(_MBConvConfig):
    def __init__(self, expand_ratio: float, kernel: int, stride: int, input_channels: int,
                 out_channels: int, num_layers: int, block: Optional[Callable[..., nn.Module]] = None):
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

# --- Convolutional Block Classes ---
class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float,
                 norm_layer: Callable[..., nn.Module], se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2): raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # Expand phase
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(Conv2dNormActivation(cnf.input_channels, expanded_channels, kernel_size=1,
                                                norm_layer=norm_layer, activation_layer=activation_layer))
        # Depthwise convolution phase
        layers.append(Conv2dNormActivation(expanded_channels, expanded_channels, kernel_size=cnf.kernel,
                                            stride=cnf.stride, groups=expanded_channels, norm_layer=norm_layer,
                                            activation_layer=activation_layer))
        # Squeeze and excitation layer
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))
        # Pointwise convolution phase
        layers.append(Conv2dNormActivation(expanded_channels, cnf.out_channels, kernel_size=1,
                                            norm_layer=norm_layer, activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class FusedMBConv(nn.Module):
    def __init__(self, cnf: FusedMBConvConfig, stochastic_depth_prob: float,
                 norm_layer: Callable[..., nn.Module]):
        super().__init__()
        if not (1 <= cnf.stride <= 2): raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # Fused expand
            layers.append(Conv2dNormActivation(cnf.input_channels, expanded_channels, kernel_size=cnf.kernel,
                                                stride=cnf.stride, norm_layer=norm_layer, activation_layer=activation_layer))
            # Project
            layers.append(Conv2dNormActivation(expanded_channels, cnf.out_channels, kernel_size=1,
                                                norm_layer=norm_layer, activation_layer=None))
        else:
            # Single convolution
            layers.append(Conv2dNormActivation(cnf.input_channels, cnf.out_channels, kernel_size=cnf.kernel,
                                                stride=cnf.stride, norm_layer=norm_layer, activation_layer=activation_layer))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

# --- EfficientNet Main Class (Using Torchvision Structure) ---
class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not inverted_residual_setting: raise ValueError("Setting should not be empty")
        if not (isinstance(inverted_residual_setting, Sequence) and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])):
            raise TypeError("Setting should be List[_MBConvConfig]")

        norm_layer = norm_layer or nn.BatchNorm2d
        layers: List[nn.Module] = []

        # Building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(Conv2dNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2,
                                            norm_layer=norm_layer, activation_layer=nn.SiLU))

        # Building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                if stage: # if not the first block in stage
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))

        # Building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(Conv2dNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                            norm_layer=norm_layer, activation_layer=nn.SiLU))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        # Weight initialization (standard torchvision practice)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- Configuration Function ---
def _efficientnet_conf(arch: str, **kwargs: Any) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    # Defines layer configurations for each variant
    if arch.startswith("efficientnet_b"):
        b_conf=partial(MBConvConfig,width_mult=kwargs.pop("width_mult"),depth_mult=kwargs.pop("depth_mult")); setting=[b_conf(1,3,1,32,16,1), b_conf(6,3,2,16,24,2), b_conf(6,5,2,24,40,2), b_conf(6,3,2,40,80,3), b_conf(6,5,1,80,112,3), b_conf(6,5,2,112,192,4), b_conf(6,3,1,192,320,1)]; last_ch=None
    elif arch == "efficientnet_v2_s": setting=[FusedMBConvConfig(1,3,1,24,24,2), FusedMBConvConfig(4,3,2,24,48,4), FusedMBConvConfig(4,3,2,48,64,4), MBConvConfig(4,3,2,64,128,6), MBConvConfig(6,3,1,128,160,9), MBConvConfig(6,3,2,160,256,15)]; last_ch=1280
    elif arch == "efficientnet_v2_m": setting=[FusedMBConvConfig(1,3,1,24,24,3), FusedMBConvConfig(4,3,2,24,48,5), FusedMBConvConfig(4,3,2,48,80,5), MBConvConfig(4,3,2,80,160,7), MBConvConfig(6,3,1,160,176,14), MBConvConfig(6,3,2,176,304,18), MBConvConfig(6,3,1,304,512,5)]; last_ch=1280
    elif arch == "efficientnet_v2_l": setting=[FusedMBConvConfig(1,3,1,32,32,4), FusedMBConvConfig(4,3,2,32,64,7), FusedMBConvConfig(4,3,2,64,96,7), MBConvConfig(4,3,2,96,192,10), MBConvConfig(6,3,1,192,224,19), MBConvConfig(6,3,2,224,384,25), MBConvConfig(6,3,1,384,640,7)]; last_ch=1280
    else: raise ValueError(f"Unsupported model arch: {arch}")
    return setting, last_ch

# --- Define Weights Classes (Simplified - URLs assume availability) ---
_COMMON_META_V2 = {"categories": ["placeholder"], "min_size": (33, 33)} # Need actual categories
class EfficientNet_V2_S_Weights(WeightsEnum):
    IMAGENET1K_V1=Weights(url="https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth", transforms=None, meta={**_COMMON_META_V2, "num_params": 21458488}) # Transforms defined elsewhere
    DEFAULT = IMAGENET1K_V1
class EfficientNet_V2_M_Weights(WeightsEnum):
    IMAGENET1K_V1=Weights(url="https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth", transforms=None, meta={**_COMMON_META_V2, "num_params": 54139356})
    DEFAULT = IMAGENET1K_V1
class EfficientNet_V2_L_Weights(WeightsEnum):
    IMAGENET1K_V1=Weights(url="https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth", transforms=None, meta={**_COMMON_META_V2, "num_params": 118515272})
    DEFAULT = IMAGENET1K_V1
# Model builder function (CORRECTED for num_classes handling)
def _efficientnet_model_builder(
    arch: str,
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    weights: Optional[WeightsEnum], # Pass weights object directly
    progress: bool,
    **kwargs: Any,
) -> EfficientNet:

    # --- CORRECTED num_classes Logic ---
    # If loading weights, ALWAYS build the initial structure with the original
    # number of classes the weights were trained on (1000 for ImageNet1k).
    # Ignore kwargs['num_classes'] at this stage if weights are provided.
    if weights is not None:
        num_classes_for_build = 1000 # Hardcode for standard ImageNet1k weights
        print(f"Building initial structure for {arch} with {num_classes_for_build} classes to load weights.")
    # If not loading weights, use kwargs['num_classes'] if provided, else default.
    elif "num_classes" in kwargs:
         num_classes_for_build = kwargs["num_classes"]
         print(f"Building structure for {arch} with {num_classes_for_build} classes (random init).")
    else:
         num_classes_for_build = 1000 # Default if no weights and no override
         print(f"Building structure for {arch} with default {num_classes_for_build} classes (random init).")
    # --- END CORRECTION ---

    # Clean kwargs before passing to EfficientNet constructor to avoid conflicts
    build_kwargs = kwargs.copy()
    if "num_classes" in build_kwargs: # Remove num_classes if it exists in kwargs
        del build_kwargs["num_classes"]

    # Build model structure
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        last_channel=last_channel,
        num_classes=num_classes_for_build, # Use the correctly determined num_classes
        **build_kwargs # Pass other kwargs like norm_layer
        )

    if weights is not None:
        # Load state dict using the weights object
        print(f"Loading state_dict from weights for {arch}...")
        try:
             model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
             print("State dict loaded successfully.")
        except Exception as e:
             print(f"ERROR loading state_dict: {e}")
             print("Continuing without pretrained weights for this backbone.")
             # Optionally: re-initialize model randomly? Or just proceed?

    return model

# --- Keep the helper functions efficientnet_v2_s/m/l as they were in the previous block ---
# They correctly pass the weights object to the corrected builder.

@handle_legacy_interface(weights=("pretrained", EfficientNet_V2_S_Weights.IMAGENET1K_V1))
def efficientnet_v2_s(*, weights: Optional[EfficientNet_V2_S_Weights] = None, progress: bool = True, **kwargs: Any) -> EfficientNet:
    weights = EfficientNet_V2_S_Weights.verify(weights)
    inv_res_setting, last_ch = _efficientnet_conf("efficientnet_v2_s")
    model = _efficientnet_model_builder("efficientnet_v2_s", inv_res_setting, kwargs.pop("dropout", 0.2), last_ch, weights, progress, norm_layer=partial(nn.BatchNorm2d, eps=1e-03), **kwargs)
    return model

@handle_legacy_interface(weights=("pretrained", EfficientNet_V2_M_Weights.IMAGENET1K_V1))
def efficientnet_v2_m(*, weights: Optional[EfficientNet_V2_M_Weights] = None, progress: bool = True, **kwargs: Any) -> EfficientNet:
    weights = EfficientNet_V2_M_Weights.verify(weights)
    inv_res_setting, last_ch = _efficientnet_conf("efficientnet_v2_m")
    model = _efficientnet_model_builder("efficientnet_v2_m", inv_res_setting, kwargs.pop("dropout", 0.3), last_ch, weights, progress, norm_layer=partial(nn.BatchNorm2d, eps=1e-03), **kwargs)
    return model

@handle_legacy_interface(weights=("pretrained", EfficientNet_V2_L_Weights.IMAGENET1K_V1))
def efficientnet_v2_l(*, weights: Optional[EfficientNet_V2_L_Weights] = None, progress: bool = True, **kwargs: Any) -> EfficientNet:
    weights = EfficientNet_V2_L_Weights.verify(weights)
    inv_res_setting, last_ch = _efficientnet_conf("efficientnet_v2_l")
    model = _efficientnet_model_builder("efficientnet_v2_l", inv_res_setting, kwargs.pop("dropout", 0.4), last_ch, weights, progress, norm_layer=partial(nn.BatchNorm2d, eps=1e-03), **kwargs)
    return model


print("Torchvision EfficientNet V2 classes and helpers defined (Corrected Syntax).")


class MetadataProcessor(nn.Module): 
    def __init__(self, n_meta_features, output_dim=128, hidden_factor=2, dropout_p=0.3): 
        super().__init__()
        hidden_dim = max(output_dim * hidden_factor, n_meta_features // 2)
        self.mlp = nn.Sequential(
            nn.Linear(n_meta_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim), 
            nn.BatchNorm1d(output_dim),
            nn.SiLU()
        )
        print(f"Initialized MetadataProcessor (MLP): Input={n_meta_features} -> Hidden={hidden_dim} -> Output={output_dim}")

    def forward(self, meta):
        return self.mlp(meta)


class EffNetV2Model(nn.Module):
    def __init__(self, backbone_name, out_dim, n_meta_features=0, pretrained=True, regularization_factor=1.0):
        super().__init__()
        self.n_meta_features = n_meta_features
        self.backbone_name = backbone_name
        self.out_dim = out_dim
        self.regularization_factor = regularization_factor
        # --- Load CNN Backbone ---
        print(f"Instantiating EffNetV2Model ({backbone_name})...")
        model_fn_map={'efficientnet_v2_s':efficientnet_v2_s, 'efficientnet_v2_m':efficientnet_v2_m, 'efficientnet_v2_l':efficientnet_v2_l}
        if backbone_name not in model_fn_map: raise ValueError(f"Unsupported CNN: {backbone_name}")
        self.cnn_backbone = model_fn_map[backbone_name](weights='DEFAULT' if pretrained else None)
        print(f" Loaded {backbone_name}. Pretrained={pretrained}")
        if isinstance(self.cnn_backbone.classifier, nn.Sequential) and isinstance(self.cnn_backbone.classifier[1], nn.Linear):
             self.cnn_features_dim = self.cnn_backbone.classifier[1].in_features
             self.cnn_backbone.classifier = nn.Identity()
             print(f" CNN features: {self.cnn_features_dim}. Classifier removed.")
        else: raise AttributeError(f"Unexpected classifier structure {backbone_name}")
        # Define BASE initial rates
        BASE_INITIAL_CLASSIFIER_DROPOUT = 0.40
        self.initial_classifier_dropout = min(0.9, max(0.05, BASE_INITIAL_CLASSIFIER_DROPOUT * self.regularization_factor))
        print(f"  Regularization Factor: {self.regularization_factor:.4f}")
        print(f"  Base Classifier Dropout: {BASE_INITIAL_CLASSIFIER_DROPOUT:.2f} -> Effective Initial: {self.initial_classifier_dropout:.4f}")
        self.image_dropout = nn.Dropout(0.0) 
        self.classifier_dropout = nn.Dropout(self.initial_classifier_dropout)
        meta_pathway_dim = 0
        self.meta_processor = None 
        if n_meta_features > 0:
             meta_output_dim = 128
             self.meta_processor = MetadataProcessor(n_meta_features, output_dim=meta_output_dim, dropout_p=0.3)
             meta_pathway_dim = meta_output_dim
             print(f"  - Meta Pathway Enabled ({n_meta_features} -> {meta_pathway_dim}) using MetadataProcessor (MLP Internals)")
        else:
             print("  - Meta Pathway: Disabled")
        # --- Final Classifier ---
        self.myfc = nn.Linear(self.cnn_features_dim + meta_pathway_dim, out_dim)
        self.current_epoch = 0

    def set_epoch(self, epoch): 
        self.current_epoch = epoch
        max_epoch_anneal = 25.0 
        min_dropout_mult = 0.3 
        decay_power = 1.5
        progress = min(epoch / max_epoch_anneal, 1.0)
        current_mult = 1.0 - (1.0 - min_dropout_mult) * (progress**decay_power)
        self.classifier_dropout.p = max(0.01, self.initial_classifier_dropout * current_mult)
    def forward(self, x, x_meta=None):
        x_features = self.cnn_backbone(x)
        if self.n_meta_features > 0 and self.meta_processor is not None:
            if x_meta is None: raise ValueError("x_meta required");
            proc_meta = self.meta_processor(x_meta.to(x_features.device)) 
            combined = torch.cat((x_features, proc_meta), dim=1)
        else:
            combined = x_features

        logits = self.classifier_dropout(combined)
        logits = self.myfc(logits)
        logits = torch.clamp(logits, min=-20, max=20);
        if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, 0.0)
        return logits


class HybridSwinModel(nn.Module):
    # REMOVED regularization_factors
    def __init__(self, cnn_backbone_name, transformer_backbone_name, out_dim,
                 n_meta_features=0, pretrained=True, image_size=448, regularization_factor=1.0):
        super().__init__()
        self.n_meta_features = n_meta_features
        self.cnn_backbone_name = cnn_backbone_name # Can be None now
        self.transformer_backbone_name = transformer_backbone_name
        self.out_dim = out_dim
        self.regularization_factor = regularization_factor

        # --- Initialize feature dimensions ---
        self.cnn_features_dim = 0
        self.cnn_backbone = None

        print(f"Instantiating Model (CNN: {cnn_backbone_name or 'None'}, Transformer: {transformer_backbone_name})...")
        
        # --- CNN Backbone (OPTIONAL) ---
        if self.cnn_backbone_name:
            model_fn_map={'efficientnet_v2_s':efficientnet_v2_s,'efficientnet_v2_m':efficientnet_v2_m,'efficientnet_v2_l':efficientnet_v2_l}
            if cnn_backbone_name not in model_fn_map: raise ValueError(f"Unsupported CNN: {cnn_backbone_name}")
            self.cnn_backbone = model_fn_map[cnn_backbone_name](weights='DEFAULT' if pretrained else None)
            print(f" Loaded {cnn_backbone_name}. Pretrained={pretrained}")
            if isinstance(self.cnn_backbone.classifier, nn.Sequential) and isinstance(self.cnn_backbone.classifier[1], nn.Linear):
                self.cnn_features_dim = self.cnn_backbone.classifier[1].in_features
                self.cnn_backbone.classifier=nn.Identity()
                print(f" CNN features: {self.cnn_features_dim}. Classifier removed.")
            else: raise AttributeError(f"Unexpected CNN classifier structure {cnn_backbone_name}")

        # --- Transformer Backbone (MANDATORY) ---
        try:
            self.transformer_backbone = timm.create_model(transformer_backbone_name, pretrained=pretrained)
            print(f" Loaded Transformer: {transformer_backbone_name}. Pretrained={pretrained}")
            if hasattr(self.transformer_backbone, 'head') and hasattr(self.transformer_backbone.head, 'fc'): self.transformer_features_dim=self.transformer_backbone.head.fc.in_features; self.transformer_backbone.head.fc=nn.Identity()
            elif hasattr(self.transformer_backbone, 'head') and isinstance(self.transformer_backbone.head, nn.Linear): self.transformer_features_dim=self.transformer_backbone.head.in_features; self.transformer_backbone.head=nn.Identity()
            elif hasattr(self.transformer_backbone, 'fc_norm'): self.transformer_features_dim=self.transformer_backbone.fc_norm.normalized_shape[0]; self.transformer_backbone.head=nn.Identity()
            else: self.transformer_features_dim=list(self.transformer_backbone.children())[-1].out_features; self.transformer_backbone=nn.Sequential(*list(self.transformer_backbone.children())[:-1]); print(f"Warn: Fallback Swin features ({self.transformer_features_dim}).")
            print(f" Swin features: {self.transformer_features_dim}. Classifier removed.")
            self.transformer_pool = nn.AdaptiveAvgPool1d(1)
        except Exception as e: print(f"ERROR loading Transformer {transformer_backbone_name}: {e}"); raise

        # --- Fusion and Classifier Layers ---
        BASE_INITIAL_FUSION_DROPOUT = 0.40
        BASE_INITIAL_CLASSIFIER_DROPOUT_HYBRID = 0.30
        self.initial_fusion_dropout = min(0.9, max(0.05, BASE_INITIAL_FUSION_DROPOUT * self.regularization_factor))
        self.initial_classifier_dropout = min(0.9, max(0.05, BASE_INITIAL_CLASSIFIER_DROPOUT_HYBRID * self.regularization_factor))
        self.fusion_dropout = nn.Dropout(self.initial_fusion_dropout)

        # The 'fusion_dim' is now the sum of available backbone features
        fusion_dim = self.cnn_features_dim + self.transformer_features_dim
        # The 'fusion_layer' now acts as a projection layer for the available features
        self.fusion_layer = nn.Sequential(nn.Linear(fusion_dim, 512), nn.BatchNorm1d(512), nn.SiLU())
        fusion_output_dim = 512
        
        # Meta features pathway (remains the same)
        meta_pathway_dim = 0
        self.meta_processor = None
        if n_meta_features > 0:
             meta_output_dim = 128
             self.meta_processor = MetadataProcessor(n_meta_features, output_dim=meta_output_dim, dropout_p=0.3)
             meta_pathway_dim = meta_output_dim
             print(f"  - Meta Pathway Enabled ({n_meta_features} -> {meta_pathway_dim})")
        else:
             print("  - Meta Pathway: Disabled")
             
        # Final classifier
        classifier_input_dim = fusion_output_dim + meta_pathway_dim
        self.classifier_dropout = nn.Dropout(self.initial_classifier_dropout)
        self.classifier = nn.Linear(classifier_input_dim, out_dim)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        max_epoch_anneal = 25.0; min_dropout_mult = 0.3; decay_power = 1.5
        progress = min(epoch / max_epoch_anneal, 1.0)
        current_mult = 1.0 - (1.0 - min_dropout_mult) * (progress**decay_power)
        self.fusion_dropout.p = max(0.01, self.initial_fusion_dropout * current_mult)
        self.classifier_dropout.p = max(0.01, self.initial_classifier_dropout * current_mult)

    def forward(self, x, x_meta=None):
        # --- Get Transformer Features ---
        xfmr_f = self.transformer_backbone(x)
        if xfmr_f.dim()==3: xfmr_f_pooled=self.transformer_pool(xfmr_f.permute(0,2,1)).squeeze(-1)
        elif xfmr_f.dim()==2: xfmr_f_pooled=xfmr_f
        else: raise RuntimeError(f"Unexpected Swin output shape: {xfmr_f.shape}")

        # --- Combine Image Features (if CNN exists) ---
        if self.cnn_backbone is not None:
            cnn_f = self.cnn_backbone(x)
            combined_img = torch.cat((cnn_f, xfmr_f_pooled), dim=1)
        else:
            # If no CNN, the 'combined' features are just the transformer's
            combined_img = xfmr_f_pooled

        # --- Fusion/Projection, Meta, and Classifier ---
        fused_img = self.fusion_layer(combined_img)
        fused_img = self.fusion_dropout(fused_img) 

        if self.n_meta_features > 0 and self.meta_processor is not None:
            if x_meta is None: raise ValueError("x_meta required");
            proc_meta = self.meta_processor(x_meta.to(fused_img.device)) 
            final_features = torch.cat((fused_img, proc_meta), dim=1)
        else:
            final_features = fused_img

        logits = self.classifier_dropout(final_features)
        logits = self.classifier(logits)
        logits = torch.clamp(logits, min=-20, max=20);
        if torch.isnan(logits).any():
             logits = torch.nan_to_num(logits, 0.0)
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Modified adaptive_layer_unlocking to accept single factors

def get_max_enet_block_index(model): # Check if this needs update for V2 structure
    """ Finds max block index in EffNet V1/V2 (might need adjustment based on timm naming)."""
    model_base = model.module if isinstance(model, nn.DataParallel) else model
    cnn_backbone = getattr(model_base, 'cnn_backbone', getattr(model_base, 'enet', None)) # Handle different names
    if cnn_backbone is None: print("Warning: CNN backbone not found for block index."); return -1

    max_block_idx = -1
    for name, _ in cnn_backbone.named_parameters():
        # V2 often uses structure like 'blocks.0.0...', V1 'blocks.0...'
        parts = name.split('.')
        if parts[0] == 'blocks' and len(parts) > 1 and parts[1].isdigit():
            try: idx = int(parts[1]); max_block_idx = max(max_block_idx, idx)
            except ValueError: pass
    # print(f"Determined max block index: {max_block_idx}")
    return max_block_idx


# Modified progressive_layer_unfreezing with LLRD support
def progressive_layer_unfreezing(
    model, optimizer, epoch,
    freeze_initially_until_block, # Initial freeze state (e.g., 2) - Affects intermediate unfreeze
    # --- Use FIXED schedule points ---
    unfreeze_start_epoch,         # Epoch when gradual unfreezing begins (e.g., 7)
    full_unfreeze_epoch,          # Epoch when everything is unfrozen & optimizer resets (e.g., 15)
    # ---
    total_epochs,                 # Still needed for cosine scheduler re-init
    base_lr=1e-3,
    cnn_backbone_name=None,       # For info only
    lr_adaptation_factor=1.0,     # Pass the near-baseline factor
    # --- NEW LLRD Parameter ---
    llrd_decay_rate=0.90 ,        # Layer-wise decay rate (e.g., 0.9, 0.85). Set >= 1.0 to disable LLRD.
    regularization_factor=1.0
):
    """
    Progressively unfreezes CNN layers based on a FIXED schedule and reconfigures
    the optimizer with differential learning rates (including specific LLRD for
    torchvision EfficientNetV2 CNN backbone structure) at the full_unfreeze_epoch.
    """
    model_base = model.module if isinstance(model, nn.DataParallel) else model
    cnn_backbone = getattr(model_base, 'cnn_backbone', getattr(model_base, 'enet', None))

    # --- Phase 1: Before unfreezing starts ---
    if epoch < unfreeze_start_epoch:
        return optimizer, False # No change needed yet

    # --- Get Max Block Index (only if needed for intermediate unfreeze) ---
    # Note: This 'get_max_enet_block_index' relies on block numbers within stages (e.g., blocks.1...)
    # which might differ from the LLRD depth based on 'features' index. Keep them separate concepts.
    max_block_idx_for_intermediate_unfreeze = -1
    if cnn_backbone and epoch >= unfreeze_start_epoch and epoch < full_unfreeze_epoch:
         max_block_idx_for_intermediate_unfreeze = get_max_enet_block_index(model) # Use original function name here
         if max_block_idx_for_intermediate_unfreeze == -1:
              print(f"Warning (Epoch {epoch}): Could not determine CNN block index for intermediate gradual unfreezing.")

    # --- Phase 3: Full unfreeze and optimizer reset ---
    if epoch == full_unfreeze_epoch:
        print(f"===== Epoch {epoch}: Full Model Unlock & Optimizer Reconfiguration (LLRD Rate: {llrd_decay_rate if llrd_decay_rate < 1.0 else 'Disabled'}) =====")
        # ... (Unlock parameters - code remains the same) ...
        unfrozen_count = 0
        for name, param in model_base.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
        if unfrozen_count > 0: print(f" Unlocked {unfrozen_count} previously frozen parameters.")
        else: print(" No parameters needed unlocking.")


        # --- Calculate Target LRs (incorporating adaptation factor) ---
        # ... (Calculation code remains the same) ...
        print(f"  Applying Base LR: {base_lr:.1e}, LR Adaptation Factor: {lr_adaptation_factor:.4f}")
        cnn_lr_mult, xfmr_lr_mult, head_lr_mult, meta_lr_mult = 0.3, 0.5, 1.0, 1.0
        final_base_cnn_lr = base_lr * cnn_lr_mult * lr_adaptation_factor
        final_xfmr_lr = base_lr * xfmr_lr_mult * lr_adaptation_factor
        final_head_lr = base_lr * head_lr_mult * lr_adaptation_factor
        final_meta_lr = base_lr * meta_lr_mult * lr_adaptation_factor
        # --- >>> APPLY REGULARIZATION FACTOR TO WEIGHT DECAY <<< ---
        # Define BASE weight decays
        BASE_WD_CNN = 0.01
        BASE_WD_XFMR = 0.02
        BASE_WD_HEAD = 0.05 # Includes Fusion, Meta, Classifier
        BASE_WD_REMAINING = 0.01

        # Calculate effective weight decays, ensure non-negative
        wd_cnn_eff = max(0.0, BASE_WD_CNN * regularization_factor)
        wd_xfmr_eff = max(0.0, BASE_WD_XFMR * regularization_factor)
        wd_head_eff = max(0.0, BASE_WD_HEAD * regularization_factor)
        wd_remaining_eff = max(0.0, BASE_WD_REMAINING * regularization_factor)

        print(f"  Regularization Factor applied to Weight Decay:")
        print(f"   - CNN WD:       {BASE_WD_CNN:.3f} -> {wd_cnn_eff:.4f}")
        print(f"   - Transformer WD: {BASE_WD_XFMR:.3f} -> {wd_xfmr_eff:.4f}")
        print(f"   - Head/Meta WD:   {BASE_WD_HEAD:.3f} -> {wd_head_eff:.4f}")
        # --- >>> END WD MODIFICATION <<< ---
        print(f"  Target Max Learning Rates:")
        print(f"   - CNN Backbone (Max): {final_base_cnn_lr:.3e}")
        if hasattr(model_base, 'transformer_backbone'): print(f"   - Transformer Backbone: {final_xfmr_lr:.3e}")
        if hasattr(model_base, 'fusion_layer'): print(f"   - Fusion Layer:       {final_head_lr:.3e}")
        if hasattr(model_base, 'meta_attention') or hasattr(model_base, 'meta_fc'): print(f"   - Meta Pathway:       {final_meta_lr:.3e}")
        if hasattr(model_base, 'myfc') or hasattr(model_base, 'classifier'): print(f"   - Classifier Head(s): {final_head_lr:.3e}")


        # --- Define Parameter Groups Robustly (WITH LLRD for EffNetV2 structure) ---
        param_groups = []
        all_parameter_ids = set()

        # Function to safely add parameters (remains the same)
        def add_param_group(params_to_add, lr, weight_decay, name_hint):
            nonlocal param_groups, all_parameter_ids
            valid_params = [p for p in params_to_add if p.requires_grad and id(p) not in all_parameter_ids]
            if valid_params:
                 group = {'params': valid_params, 'lr': lr, 'weight_decay': weight_decay}
                 param_groups.append(group)
                 all_parameter_ids.update(id(p) for p in valid_params)
                 print(f"    Group '{name_hint}': {len(valid_params)} params, LR={lr:.2e}, WD={weight_decay}")

        # --- LLRD Implementation Specific to Torchvision EfficientNetV2 Structure ---
        if cnn_backbone and llrd_decay_rate < 1.0:
            print(f"  Applying LLRD to CNN Backbone ({cnn_backbone_name}) with decay rate: {llrd_decay_rate}")
            # Specifically target 'self.features' which is nn.Sequential
            if hasattr(cnn_backbone, 'features') and isinstance(cnn_backbone.features, nn.Sequential):
                layers = list(cnn_backbone.features.children())
                num_layers = len(layers) # Number of direct children in features (stem, stages, final conv)
                print(f"  Found {num_layers} sequential items in cnn_backbone.features for LLRD.")

                # Assign decreasing LR based on index in 'features'
                # features[0] = stem (lowest LR)
                # features[-1] = final conv before pool (highest LR = final_base_cnn_lr)
                for i, layer_module in enumerate(layers):
                    # Depth increases for earlier layers in the sequence
                    depth_from_end = num_layers - 1 - i
                    layer_lr = final_base_cnn_lr * (llrd_decay_rate ** depth_from_end)
                    # Name hint reflects the index within 'features'
                    add_param_group(layer_module.parameters(), layer_lr, 0.01, f'CNN features[{i}]')

                # Verify all cnn params were assigned (optional check)
                cnn_param_ids = set(id(p) for p in cnn_backbone.parameters() if p.requires_grad)
                assigned_cnn_ids = cnn_param_ids.intersection(all_parameter_ids)
                if len(assigned_cnn_ids) != len(cnn_param_ids):
                    print(f"  Warning: Only {len(assigned_cnn_ids)} out of {len(cnn_param_ids)} trainable CNN params were assigned during LLRD.")

            else:
                # Fallback if structure doesn't match expectation (shouldn't happen with your models)
                print(f"  Warning: cnn_backbone.features not found or not Sequential. Applying single LR {final_base_cnn_lr:.3e} to entire CNN.")
                add_param_group(cnn_backbone.parameters(), final_base_cnn_lr, 0.01, 'CNN (Single LR)')

        elif cnn_backbone: # No LLRD or backbone found
             print(f"  Applying single LR to CNN Backbone: {final_base_cnn_lr:.3e}")
             add_param_group(cnn_backbone.parameters(), final_base_cnn_lr, 0.01, 'CNN (Single LR)')

        # --- Add other parameter groups (Transformer, Head, Meta - remains the same) ---
        add_param_group(list(getattr(model_base, 'transformer_backbone', nn.Module()).parameters()), final_xfmr_lr, 0.02, 'Transformer')
        add_param_group(list(getattr(model_base, 'fusion_layer', nn.Module()).parameters()), final_head_lr, 0.05, 'Fusion')
        # Meta pathway
        meta_params = list(getattr(model_base, 'meta_attention', nn.Module()).parameters()) + \
                      list(getattr(model_base, 'meta_fc', nn.Module()).parameters())
        if meta_params: add_param_group(meta_params, final_meta_lr, 0.05, 'Meta')
        # Classifier heads
        classifier_params = []
        myfc = getattr(model_base, 'myfc', None); classifier_head = getattr(model_base, 'classifier', None)
        if myfc and not isinstance(myfc, nn.Identity): classifier_params.extend(list(myfc.parameters()))
        if classifier_head and not isinstance(classifier_head, nn.Identity): classifier_params.extend(list(classifier_head.parameters()))
        if classifier_params: add_param_group(classifier_params, final_head_lr, 0.05, 'Classifier')

        # --- Catch Remaining ---
        remaining_params = [p for n, p in model_base.named_parameters() if p.requires_grad and id(p) not in all_parameter_ids]
        if remaining_params:
            print(f"  Warn: Adding {len(remaining_params)} remaining trainable parameters to a default group.")
            add_param_group(remaining_params, final_head_lr, 0.05, 'Remaining')

        # --- Create Optimizer ---
        # ... (remains the same) ...
        if not param_groups: raise RuntimeError("Optimizer reconfiguration failed: No parameter groups were created.")
        print(f"  Optimizer reconfiguring with {len(param_groups)} parameter groups.")
        new_optimizer = optim.AdamW(param_groups, eps=1e-7)
        print("===== Optimizer Reconfigured Successfully =====")
        return new_optimizer, True

    # --- Phase 2: Intermediate Unfreezing (Only affects CNN backbone) ---
    # This part MUST use the block index logic from `get_max_enet_block_index`,
    # as it targets specific blocks *within* the stages, not the stages themselves.
    elif epoch >= unfreeze_start_epoch and epoch < full_unfreeze_epoch and cnn_backbone and max_block_idx_for_intermediate_unfreeze != -1:
         # Use the max_block_idx_for_intermediate_unfreeze calculated earlier
        blocks_to_unfreeze_sequence = list(range(max_block_idx_for_intermediate_unfreeze, freeze_initially_until_block, -1))
        num_stages_to_unfreeze = len(blocks_to_unfreeze_sequence)
        if num_stages_to_unfreeze > 0:
            unfreeze_epoch_span = full_unfreeze_epoch - unfreeze_start_epoch
            epochs_per_stage = max(1, math.ceil(unfreeze_epoch_span / num_stages_to_unfreeze))
            current_stage_index = (epoch - unfreeze_start_epoch) // epochs_per_stage

            if current_stage_index < num_stages_to_unfreeze:
                block_idx_to_unfreeze = blocks_to_unfreeze_sequence[current_stage_index]
                unfrozen_in_this_epoch = False
                # --- This loop MUST check the block number within the stage ---
                for name, param in cnn_backbone.named_parameters():
                    is_target_block = False
                    parts = name.split('.')
                    # Example check: 'features.STAGE_IDX.BLOCK_IDX...' or 'blocks.BLOCK_IDX...'
                    # Need to adapt based on precise naming in the cnn_backbone parameters
                    # Let's assume the naming follows the 'blocks.X.Y...' or 'features.X.Y...' pattern
                    # where X is stage, Y is block (this might need adjustment)
                    # *** Simplified check based on get_max_enet_block_index's assumption ***
                    if (parts[0] == 'blocks' or parts[0] == 'features') and len(parts) > 1 and parts[1].isdigit():
                          # This assumes the index used by get_max_enet_block_index is directly in parts[1]
                          # This might be incorrect for the torchvision structure where parts[1] is stage index.
                          # A more robust check would parse deeper, e.g. parts[2] for block index within a stage.
                          # --- Using the potentially incorrect but consistent check for now ---
                         try:
                             current_block_idx = int(parts[1]) # <<< ASSUMPTION HERE based on get_max_enet_block_index
                             if current_block_idx == block_idx_to_unfreeze:
                                 is_target_block = True
                         except ValueError: pass

                    if is_target_block and not param.requires_grad:
                        param.requires_grad = True; unfrozen_in_this_epoch = True

                if unfrozen_in_this_epoch:
                    # Use the block index determined by get_max_enet_block_index
                    print(f"Epoch {epoch}: Unlocking CNN block {block_idx_to_unfreeze} (based on intermediate schedule)...")
                    return optimizer, False

    # Default: No change
    return optimizer, False


def partial_freeze_enet(model, freeze_until_block=2): # freeze_until_block now refers to STAGE index
    """
    Partially freezes CNN backbone (Torchvision EffNetV2 structure).
    Freezes the stem (features.0) and stages up to and including freeze_until_block.
    Ensures other model parts (head, transformer, meta) are trainable.
    """
    model_base = model.module if isinstance(model, nn.DataParallel) else model
    cnn_backbone = getattr(model_base, 'cnn_backbone', getattr(model_base, 'enet', None))
    if not cnn_backbone:
        print("Warning (partial_freeze_enet): CNN backbone not found. Skipping freeze.")
        return

    # Verify expected structure
    if not (hasattr(cnn_backbone, 'features') and isinstance(cnn_backbone.features, nn.Sequential)):
         print(f"Warning (partial_freeze_enet): Expected cnn_backbone.features (nn.Sequential) not found in {type(cnn_backbone)}. Freeze might be incorrect.")
         # Attempt to proceed, but it might not work as expected

    print(f"--- Applying Partial Freeze (CNN stem & stages <= {freeze_until_block}) ---")
    frozen_count, trainable_count = 0, 0
    # Freeze stem, initial blocks/stages
    for name, param in cnn_backbone.named_parameters():
        should_freeze = False
        parts = name.split('.')

        # --- >>> MODIFIED LOGIC <<< ---
        # Check for typical stem names first (more specific)
        if name.startswith('features.0.') or name.startswith('stem.') or name.startswith('conv_stem.') or name.startswith('bn1.'):
             # print(f"  Freezing stem layer: {name}") # Optional Debug
             should_freeze = True
        # Check for early stages based on features.STAGE_INDEX.
        elif name.startswith('features.') and len(parts) > 1 and parts[1].isdigit():
             try:
                 stage_idx = int(parts[1])
                 # Freeze stages up to and including freeze_until_block
                 # Note: Stage index parts[1] starts from 1 for actual stages after stem (features.0)
                 if stage_idx > 0 and stage_idx <= freeze_until_block + 1: # +1 because features[1] is stage 0 effectively
                      # print(f"  Freezing stage {stage_idx} layer: {name}") # Optional Debug
                      should_freeze = True
             except ValueError:
                 pass # Ignore if parts[1] is not a digit
        # --- >>> END MODIFIED LOGIC <<< ---

        param.requires_grad = not should_freeze
        if should_freeze:
            frozen_count += param.numel()
        else:
            trainable_count += param.numel()

    print(f"CNN backbone partially frozen up to stage index {freeze_until_block}.") # Clarify meaning
    print(f"  - CNN Frozen params: {frozen_count:,}")
    print(f"  - CNN Trainable params: {trainable_count:,}")

    # Ensure other backbones (Swin) or heads are trainable initially
    # --- This part remains correct ---
    if hasattr(model_base, 'transformer_backbone'):
        xfmr_trainable = 0
        for param in model_base.transformer_backbone.parameters():
             param.requires_grad = True; xfmr_trainable += param.numel()
        print(f"Transformer backbone set to trainable ({xfmr_trainable:,} params).")
    print("Ensuring head/meta/fusion parts are trainable...")
    ensured_trainable_count = 0
    for part_name in ['fusion_layer', 'meta_attention', 'meta_fc', 'classifier', 'myfc']:
         part = getattr(model_base, part_name, None)
         if part and not isinstance(part, nn.Identity): # Check it's a real layer/module
             part_params = 0
             for param in part.parameters():
                  param.requires_grad = True; part_params += param.numel()
             if part_params > 0:
                  print(f"  - {part_name} set to trainable ({part_params:,} params).")
                  ensured_trainable_count += part_params

    print(f"Total parameters set trainable outside CNN: {ensured_trainable_count + (xfmr_trainable if hasattr(model_base, 'transformer_backbone') else 0):,}")
    print("--- Partial Freeze Setup Complete ---")


# Fix Warmup Bug
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


# Utility function to get resource usage
def get_resource_usage():
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_max = torch.cuda.max_memory_allocated() / 1024**2  # MB
        return {"gpu_memory_allocated": mem_alloc, "gpu_max_memory": mem_max}
    else:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent
        return {"cpu_usage": cpu_usage, "ram_usage": ram_usage}
        
class TemperatureScaling:
    def __init__(self, model, device):
        # Store the original model passed (could be base or wrapped)
        self.model = model
        self.device = device
        # Initialize log_temperature as a learnable parameter starting at 0 (exp(0) = 1)
        self.log_temperature = nn.Parameter(torch.zeros(1).to(device))

    def calibrate(self, loader, max_iter=50):
        """ Optimizes temperature using validation data with NLL. """
        self.model.eval() # Set the potentially wrapped model to eval mode

        # --- *** Get base model reference *** ---
        # Use this reference for checking attributes like n_meta_features
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        # ---

        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        # Ensure parameter is registered correctly for optimizer
        if not list(self.model.parameters()): # Check if model has parameters (might happen if called standalone)
            print("Warning (Calibrate): Model has no parameters? Adding log_temperature manually.")
            params_to_optimize = [self.log_temperature]
        else:
            # Optimize only the temperature parameter
            params_to_optimize = [self.log_temperature]
            # Ensure log_temperature requires grad if model doesn't
            if not self.log_temperature.requires_grad: self.log_temperature.requires_grad = True


        # Check if parameters require grad before passing to optimizer
        if not any(p.requires_grad for p in params_to_optimize):
            print("Warning (Calibrate): No parameters require gradients for optimizer. Setting log_temperature.requires_grad=True.")
            self.log_temperature.requires_grad = True


        optimizer = optim.LBFGS(params_to_optimize, lr=0.01, max_iter=max_iter)

        all_logits = []
        all_targets = []
        print("Calibrating: Iterating through validation loader...") # Add print
        with torch.no_grad():
            cal_pbar = tqdm(loader, desc="Calibrating T", leave=False, total=len(loader))
            for batch_idx, batch in enumerate(cal_pbar):
                images, meta, target = None, None, None # Initialize batch variables

                # --- *** CORRECTED Data Unpacking & Forward Call *** ---
                # Check meta features existence using the base_model reference
                has_meta_features_in_model = hasattr(base_model, 'n_meta_features') and base_model.n_meta_features > 0

                try:
                    # Determine expected batch structure based on model's capability
                    if has_meta_features_in_model:
                         # Assumes loader provides ((img, meta), tgt) if model has meta
                         (images, meta), target = batch
                         images, meta, target = images.to(self.device), meta.to(self.device), target.to(self.device)
                    else:
                         # Assumes loader provides (img, tgt) if model does NOT have meta
                         images, target = batch
                         images, target = images.to(self.device), target.to(self.device)
                         meta = None # Ensure meta is None

                    # Perform forward pass using the ORIGINAL model ('self.model')
                    # Pass meta only if the global use_meta flag is True AND meta was successfully unpacked
                    # Note: We rely on the global 'use_meta' flag here, which should align with has_meta_features_in_model
                    if use_meta and meta is not None:
                         logits = self.model(images, x_meta=meta)
                    else:
                         logits = self.model(images)

                    all_logits.append(logits)
                    all_targets.append(target)

                except ValueError as ve:
                    # Handle potential unpacking errors if batch structure is unexpected
                    print(f"Warning (Calibrate Batch {batch_idx}): ValueError unpacking batch - {ve}. Trying to recover.")
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        try: # Assume (img, tgt) structure
                            images, target = batch
                            images, target = images.to(self.device), target.to(self.device)
                            meta = None
                            logits = self.model(images) # Call without meta
                            all_logits.append(logits)
                            all_targets.append(target)
                            print(" Recovered assuming (img, tgt) format.")
                        except Exception as e_rec:
                            print(f" Recovery failed: {e_rec}. Skipping batch.")
                            continue # Skip this batch
                    else:
                        print(" Cannot determine batch structure. Skipping batch.")
                        continue # Skip this batch
                except Exception as e:
                    print(f"Error processing batch {batch_idx} during calibration: {e}")
                    traceback.print_exc()
                    # Decide whether to skip or raise
                    continue # Skip batch on other errors for now
                # --- *** END CORRECTION *** ---

        if not all_logits:
            print("Warning: No logits collected during calibration. Returning default temperature.")
            return 1.0

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        targets = targets.to(self.device) # Ensure targets are on device for loss

        def nll_closure():
            optimizer.zero_grad()
            temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
            scaled_logits = logits / temperature
            loss = nll_criterion(scaled_logits, targets)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf loss in NLL closure. Returning high value.")
                return torch.tensor(1e6, device=self.device, requires_grad=True) # Return high loss
            loss.backward()
            return loss

        optimal_temperature = 1.0 # Default
        try:
            print("Running LBFGS optimizer for temperature...")
            optimizer.step(nll_closure)
            optimal_temperature = torch.exp(self.log_temperature).item()
            print(f"Optimal temperature found: {optimal_temperature:.4f}")
        except Exception as opt_e:
            print(f"Warning: Temperature optimization failed: {opt_e}")
            optimal_temperature = 1.0

        # Add final clamp/check for safety
        optimal_temperature = max(0.1, min(optimal_temperature, 10.0))
        if np.isnan(optimal_temperature): optimal_temperature = 1.0

        return optimal_temperature

    # --- forward method remains the same ---
    def forward(self, logits):
        temperature = torch.exp(self.log_temperature).item()
        temperature = max(0.1, min(temperature, 10.0)) # Clamp temperature
        return torch.softmax(logits / temperature, dim=1)


# --- REVISED EarlyStopping Class ---
class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        mode: str = 'max',
        delta: float = 0.005,
        relative_delta: bool = True,
        warm_up: int = 9,
        verbose: bool = True,
        checkpoint_path: str = 'best_model.pth',
        score_weights: Optional[Dict[str, float]] = None
    ):
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.relative_delta = relative_delta
        self.warm_up = warm_up
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.counters: Dict[str, int] = {}
        self.best_scores: Dict[str, float] = {}
        self.best_epoch: Dict[str, int] = {}
        self.early_stop = False
        self._is_first = True
        self.score_weights = score_weights or {
            'binary_auc': 0.5, 'binary_recall': 0.2,
            'multiclass_auc': 0.2, 'val_loss': 0.1 # Example weights
        }
        # Validate and normalize score_weights
        if self.score_weights:
            total_weight = sum(v for v in self.score_weights.values() if isinstance(v, (int, float)))
            if total_weight <= 0:
                raise ValueError("Total weight must be positive.")
            # Normalize
            self.score_weights = {k: v / total_weight for k, v in self.score_weights.items()}

    def reset(self):
        self.counters = {}; self.best_scores = {}; self.best_epoch = {}
        self.early_stop = False; self._is_first = True

    def __call__(
        self,
        metrics: Union[Dict[str, float], float],
        model: Optional[torch.nn.Module] = None, # Receives the potentially wrapped model
        epoch: Optional[int] = None
    ):
        if epoch is None: raise ValueError("epoch must be provided.")
        if epoch <= self.warm_up:
            if self.verbose: print(f"Epoch {epoch}: Warm-up period, skipping ES.")
            return

        if isinstance(metrics, (int, float)): metrics = {'val_metric': metrics}

        # Compute composite score
        score = 0.0
        for name, val in metrics.items():
            if not isinstance(val, (int, float)): continue # Skip non-numeric
            weight = self.score_weights.get(name, 0)
            if weight > 0:
                adj_score = -val if name == 'val_loss' else val # Invert loss for maximization
                score += adj_score * weight

        current_delta = self.delta if not self.relative_delta else self.delta * abs(self.best_scores.get('composite', 0))

        # --- CORRECTED model state saving ---
        # Get base model for saving state_dict ONLY if saving is needed
        model_state_to_save = None
        if model is not None:
            # Get the underlying model state if wrapped in DataParallel
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            try:
                 # Attempt to get state_dict - might fail if model has issues
                 model_state_to_save = model_to_save.state_dict()
            except Exception as e:
                 print(f"Warning: Could not get model state_dict for saving. Error: {e}")
                 model_state_to_save = None # Ensure it's None if state_dict fails
        # --- END CORRECTION ---

        save_model = False
        if self._is_first or self.best_scores.get('composite') is None:
            self.best_scores['composite'] = score
            self.best_epoch['composite'] = epoch
            self.counters['composite'] = 0
            self._is_first = False
            save_model = True # Save on first valid epoch
            if self.verbose: print(f"Initial best score: {score:.6f} @ E{epoch}", end="")
        elif score > self.best_scores['composite'] + current_delta:
            self.best_scores['composite'] = score
            self.best_epoch['composite'] = epoch
            self.counters['composite'] = 0
            save_model = True # Save if improved
            if self.verbose: print(f"New best score: {score:.6f} @ E{epoch}", end="")
        else:
            self.counters['composite'] = self.counters.get('composite', 0) + 1
            if self.verbose: print(f"No improvement. Counter: {self.counters['composite']}/{self.patience}")
            if self.counters['composite'] >= self.patience:
                self.early_stop = True
                if self.verbose: print(f"Early stopping triggered @ E{epoch}.")

        # Save checkpoint only if required and state is available
        if save_model and model_state_to_save is not None and self.checkpoint_path:
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state_to_save, # Save the potentially unwrapped state
                    'best_scores': self.best_scores, # Log best scores so far
                    'best_epoch': self.best_epoch # Log best epoch so far
                }, self.checkpoint_path)
                if self.verbose: print(f", model saved to {self.checkpoint_path}")
            except Exception as e:
                 print(f"\nERROR saving checkpoint: {e}")
        elif save_model and self.verbose:
             print(" (checkpoint not saved - model state unavailable or no path)")
        elif save_model: # If verbose is false but save failed
             print(f"Warning: Failed to save best model at epoch {epoch}.")


import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# Make sure get_resource_usage is defined elsewhere or remove the call

def train_epoch(model, loader, optimizer, experiment, epoch, scaler=None, accum_steps=1, criterion_multi=None, mel_idx=None, lambda_binary=0.75, device=None):
    """
    Training epoch function.
    """
    # --- Initial Checks & Setup ---
    if criterion_multi is None or mel_idx is None:
        raise ValueError("criterion_multi and mel_idx must be provided to train_epoch")
    if device is None:
        # print("Warning: No device provided; defaulting to 'cuda' if available") # Less verbose
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train() # Set model to training mode

    # --- GET BASE MODEL REFERENCE ---
    # Determine if the model is wrapped and get the underlying module if necessary
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    # ------------------------------

    # --- Initialize metrics and timers ---
    train_loss_list = []
    train_correct = 0
    train_total = 0
    start_time = time.time()
    optimizer.zero_grad() # Zero gradients at the start

    # --- Initialize Progress Bar ---
    pbar = tqdm(loader, desc=f"Epoch {epoch} - Loss: N/A, Acc: N/A", total=len(loader), leave=False)
    for batch_idx, batch in enumerate(pbar):
        try: # <--- ADD TRY BLOCK HERE ---
            # --- Data Handling ---
            if hasattr(base_model, 'n_meta_features') and base_model.n_meta_features > 0:
                (images, meta), target = batch
                images, meta, target = images.to(device), meta.to(device), target.to(device)
            else:
                images, target = batch
                images, target = images.to(device), target.to(device)
                meta = None # Ensure meta is None if not used

            # --- Forward Pass ---
            if scaler:
                with torch.cuda.amp.autocast():
                    # Use model() call directly, forward method handles meta internally
                    logits = model(images, meta) if meta is not None else model(images)

                    # --- Loss Calculation ---
                    multiclass_loss = criterion_multi(logits, target)
                    binary_target = (target == mel_idx).float()
                    binary_logits = logits[:, mel_idx] # Use mel_idx defined globally or passed
                    binary_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_target)
                    total_loss_unscaled = multiclass_loss + lambda_binary * binary_loss
                    total_loss = total_loss_unscaled / accum_steps
                # --- Backward Pass (AMP) ---
                scaler.scale(total_loss).backward()
            else: # Not using AMP
                # Use model() call directly
                logits = model(images, meta) if meta is not None else model(images)

                # --- Loss Calculation ---
                multiclass_loss = criterion_multi(logits, target)
                binary_target = (target == mel_idx).float()
                binary_logits = logits[:, mel_idx]
                binary_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_target)
                total_loss_unscaled = multiclass_loss + lambda_binary * binary_loss
                total_loss = total_loss_unscaled / accum_steps
                # --- Backward Pass ---
                total_loss.backward()

            train_loss_list.append(total_loss_unscaled.item())

            # --- Accuracy Tracking ---
            with torch.no_grad():
                 preds = logits.argmax(dim=1)
                 batch_correct = (preds == target).sum().item()
                 batch_total = target.size(0)
                 train_correct += batch_correct
                 train_total += batch_total

            # --- Optimization Step ---
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # --- Update Progress Bar ---
            avg_loss_so_far = np.mean(train_loss_list) if train_loss_list else 0.0
            acc_so_far = (train_correct / train_total) * 100.0 if train_total > 0 else 0.0
            pbar.set_description(f"Epoch {epoch} - Loss: {avg_loss_so_far:.4f}, Acc: {acc_so_far:.2f}%")

        # --- ADD EXCEPT BLOCK HERE ---
        except AttributeError as ae:
            print(f"\n\n!!! AttributeError caught in train_epoch loop (batch {batch_idx}) !!!")
            print(f"Error message: {ae}")
            print("Model type:", type(model))
            print("Is model DataParallel?", isinstance(model, nn.DataParallel))
            print("--- Traceback ---")
            traceback.print_exc() # Print the exact line causing the error
            print("-----------------")
            raise ae # Re-raise the error to stop execution
        except Exception as e:
             print(f"\n\n!!! Non-AttributeError caught in train_epoch loop (batch {batch_idx}) !!!")
             print(f"Error type: {type(e)}")
             print(f"Error message: {e}")
             traceback.print_exc()
             raise e
        # --- END TRY/EXCEPT ---

    # --- End of Epoch ---
    pbar.close()
    avg_train_loss = np.mean(train_loss_list) if train_loss_list else 0.0 # Final average loss for the epoch
    train_acc = (train_correct / train_total) * 100.0 if train_total > 0 else 0.0
    epoch_time = time.time() - start_time

    # --- Resource Usage ---
    try:
        resources = get_resource_usage()
    except NameError:
        resources = {} # Handle if function not defined
        print("Warning: get_resource_usage() not defined.")

    # --- Print Epoch Summary ---
    print(
        f"Epoch {epoch} - Training Time: {epoch_time:.2f}s, Avg Loss: {avg_train_loss:.5f}, "
        f"Acc: {train_acc:.2f}%, Resources: {resources}"
    )

    # --- Logging to W&B ---
    if experiment:
        log_data = {
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "train_epoch_time_seconds": epoch_time,
        }
        # Add resource usage if available and it's a dictionary
        if isinstance(resources, dict):
             log_data.update({f"train_{k}": v for k, v in resources.items()})
        experiment.log(log_data, step=epoch)

    # --- Return average loss for the epoch ---
    return avg_train_loss


def val_epoch(model, loader, experiment, epoch, n_test=1, recalib_interval=5, criterion_multi=None, mel_idx=None, lambda_binary=0.5, device=None, use_amp=True):
    if criterion_multi is None or mel_idx is None: raise ValueError("criterion/mel_idx needed")
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    out_dim = base_model.out_dim

    val_loss_batch_list = [] ; PROBS_ALL = []; TARGETS = []
    start_time = time.time()

    # --- Temperature Scaling ---
    # print("DEBUG val_epoch: Initializing TemperatureScaling...") # DEBUG
    temp_scaler = TemperatureScaling(model, device) # Pass original model
    optimal_temp = None
    if epoch > 1 and epoch % recalib_interval == 1:
        # print(f"DEBUG val_epoch: Calling temp_scaler.calibrate for Epoch {epoch}...") # DEBUG
        try: optimal_temp = temp_scaler.calibrate(loader)
        except Exception as cal_e: print(f" T-Scaling calibration failed: {cal_e}")
        if experiment and optimal_temp is not None: experiment.log({"optimal_temperature": optimal_temp}, step=epoch)
    current_temp = optimal_temp if optimal_temp is not None else (torch.exp(temp_scaler.log_temperature).item() if hasattr(temp_scaler, 'log_temperature') else 1.0)
    print(f"Using temperature: {current_temp:.4f} for validation epoch {epoch}")

    # --- Init Metrics ---
    mc_acc = MulticlassAccuracy(num_classes=out_dim, average='macro').to(device)
    mc_f1 = MulticlassF1Score(num_classes=out_dim, average='macro').to(device)
    mc_auc = MulticlassAUROC(num_classes=out_dim, average='macro', thresholds=None).to(device)
    mc_cm = MulticlassConfusionMatrix(num_classes=out_dim).to(device)
    bin_auc = BinaryAUROC(thresholds=None).to(device)
    # print("DEBUG val_epoch: Metrics initialized.") # DEBUG

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validating Epoch {epoch}", total=len(loader), leave=False)
        for batch_idx, batch in enumerate(pbar):
            # --- DEBUG PRINT ---
            # print(f"\nDEBUG val_epoch: Starting batch {batch_idx}. Model type: {type(model)}, DP: {isinstance(model, nn.DataParallel)}")
            # ---
            try:
                # --- Unpack Batch ---
                images, meta, target = None, None, None # Initialize
                if hasattr(base_model, 'n_meta_features') and base_model.n_meta_features > 0:
                    try: (images, meta), target = batch; images, meta, target = images.to(device), meta.to(device), target.to(device)
                    except ValueError: images, target = batch; images, target = images.to(device), target.to(device); meta = None; print(f"Warn val B{batch_idx}: Unpack assuming (img,tgt)")
                else:
                    images, target = batch; images, target = images.to(device), target.to(device); meta = None

                # --- Inference ---
                # --- DEBUG PRINT ---
                # print(f"DEBUG val_epoch B{batch_idx}: Before forward pass. Use meta: {use_meta}, Meta is None: {meta is None}")
                # ---
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(images, x_meta=meta) if use_meta and meta is not None else model(images)
                    # --- DEBUG PRINT ---
                    # print(f"DEBUG val_epoch B{batch_idx}: After forward pass. Logits shape: {logits.shape}")
                    # ---
                    logits = torch.clamp(logits, min=-20, max=20); # Clamp logits
                    if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, 0.0)

                    # --- Loss ---
                    # --- DEBUG PRINT ---
                    # print(f"DEBUG val_epoch B{batch_idx}: Before loss calculation.")
                    # ---
                    multiclass_loss = criterion_multi(logits, target)
                    binary_target = (target == mel_idx).float(); binary_logits = logits[:, mel_idx]
                    binary_loss = F.binary_cross_entropy_with_logits(torch.clamp(binary_logits, -10, 10), binary_target)
                    total_loss = multiclass_loss + lambda_binary * binary_loss

                    # --- Probabilities ---
                    current_probs = torch.softmax(logits / current_temp, dim=1)
                    current_probs = torch.clamp(current_probs, 1e-6, 1.0 - 1e-6); # Clamp probs
                    if torch.isnan(current_probs).any(): current_probs = torch.nan_to_num(current_probs, 0.5)

                # Store results
                if not (torch.isnan(total_loss) or torch.isinf(total_loss)): val_loss_batch_list.append(total_loss.item())
                PROBS_ALL.append(current_probs); TARGETS.append(target)

                # --- Update Metrics ---
                # --- DEBUG PRINT ---
                # print(f"DEBUG val_epoch B{batch_idx}: Before metric update.")
                # ---
                mc_acc.update(current_probs, target); mc_f1.update(current_probs, target)
                mc_auc.update(current_probs, target); mc_cm.update(current_probs.argmax(dim=1), target)
                bin_auc.update(current_probs[:, mel_idx], (target == mel_idx).int())
                # --- DEBUG PRINT ---
                # print(f"DEBUG val_epoch B{batch_idx}: After metric update.")
                # ---

                pbar.set_postfix(loss=total_loss.item())

            except AttributeError as ae: # Catch specific error if needed again
                 print(f"\n\n!!! AttributeError caught in val_epoch loop (batch {batch_idx}) !!!"); print(f"Error: {ae}")
                 print(f"Model type: {type(model)}, DP: {isinstance(model, nn.DataParallel)}")
                 traceback.print_exc(); raise ae
            except Exception as e:
                 print(f"\n\n!!! Error in val_epoch loop (batch {batch_idx}) !!!"); print(f"Type: {type(e)}, Error: {e}")
                 traceback.print_exc(); raise e # Re-raise to stop run

        pbar.close()

    # --- Aggregate & Calculate Final Metrics ---
    # print("DEBUG val_epoch: Finished validation loop, starting aggregation.")
    if not val_loss_batch_list: return (0.0,0.5,0.5,np.array([]),np.array([]),0,0,0,0,0,0,0,[],[],1.0,None)

    val_loss_avg = np.mean(val_loss_batch_list)
    epoch_time = time.time() - start_time
    try: resources = get_resource_usage()
    except NameError: resources = {}

    PROBS = torch.cat(PROBS_ALL, dim=0).cpu().numpy()
    TARGETS = torch.cat(TARGETS, dim=0).cpu().numpy()
    print(f"Epoch {epoch} Valid Aggregated: PROBS={PROBS.shape}, TARGETS={TARGETS.shape}")
    if PROBS.shape[0] == 0: return (val_loss_avg,0.5,0.5,PROBS,TARGETS,0,0,0,0,0,0,0,[],[],current_temp,None)

    # --- *** FINAL CORRECTION for .compute() calls *** ---
    # Use the variables defined earlier: mc_acc, mc_f1, mc_auc, mc_cm, bin_auc
    mc_acc_val = mc_acc.compute().item() * 100.0
    mc_f1_val = mc_f1.compute().item()
    mc_auc_val = mc_auc.compute().item()
    mc_cm_val = mc_cm.compute().cpu().numpy().tolist()
    bin_auc_val = bin_auc.compute().item()
    # --- *** END CORRECTION *** ---

    # --- Calculate binary threshold & metrics (remains the same) ---
    binary_targets_np=(TARGETS == mel_idx).astype(np.float32); binary_probs_np=PROBS[:, mel_idx]
    best_threshold=0.5; best_f1=0.0; f1_hist=[]
    try:
        prec, rec, pr_thresh = precision_recall_curve(binary_targets_np, np.minimum(binary_probs_np*1.1, 1.0))
        if len(prec) > 1 and len(rec) > 1:
            f1s=2*(prec[:-1]*rec[:-1])/(prec[:-1]+rec[:-1]+1e-6); opt_idx=np.argmax(f1s); best_f1=f1s[opt_idx]; best_threshold=pr_thresh[opt_idx]
            f1_hist = list(zip(pr_thresh, f1s)) # Store history only if calculated
        print(f" Best F1 Threshold (PR): {best_threshold:.4f}, F1={best_f1:.4f}")
    except Exception as thresh_e: print(f"Warn: Threshold opt failed: {thresh_e}")
    # --- CORRECTED Binary Metric Calculation Block ---
    binary_preds_best=(binary_probs_np > best_threshold).astype(np.float32)
    bin_cm_best=[[0,0],[0,0]]; # Default CM
    # Initialize all metrics to default values
    bin_spec_best,bin_prec_best,bin_rec_best,bin_f1_best,bin_acc_best = 0.0, 0.0, 0.0, 0.0, 0.0

    if len(binary_targets_np) > 0:
        try: # Start try block for ALL binary metric calculations
            bin_cm_best=confusion_matrix(binary_targets_np, binary_preds_best).tolist()

            # Calculate specificity only if CM is 2x2
            if len(np.array(bin_cm_best).ravel())==4:
                tn,fp,fn,tp=np.array(bin_cm_best).ravel()
                bin_spec_best=tn/(tn+fp) if (tn+fp)>0 else 0.0
            # else: specificity remains 0.0

            # Calculate other metrics INSIDE the try block
            bin_prec_best=precision_score(binary_targets_np,binary_preds_best,zero_division=0)
            bin_rec_best=recall_score(binary_targets_np,binary_preds_best,zero_division=0)
            bin_f1_best=f1_score(binary_targets_np,binary_preds_best,zero_division=0) # This should match best_f1 found earlier
            bin_acc_best=accuracy_score(binary_targets_np,binary_preds_best)*100

        except Exception as bin_metric_e:
            # If any calculation fails, metrics keep their default values (0.0)
            print(f"Warn: Binary metric calculation failed: {bin_metric_e}")
            # bin_cm_best remains default [[0,0],[0,0]]
    # --- END CORRECTION ---

    # --- Print & Log (uses potentially updated variables) ---
    print(
         f"Epoch {epoch} - Val Loss: {val_loss_avg:.5f}, "
         f"Bin AUC: {bin_auc_val:.4f}, F1(T={best_threshold:.2f}): {bin_f1_best:.4f}, "
         f"MC AUC: {mc_auc_val:.4f}, MC Acc: {mc_acc_val:.2f}%, MC F1: {mc_f1_val:.4f}, "
         f"Val Time: {epoch_time:.2f}s"
    )
    metrics_log = {
         "val_loss": val_loss_avg, "binary_auc": bin_auc_val, "multiclass_auc": mc_auc_val if not np.isnan(mc_auc_val) else 0.5,
         "binary_acc": bin_acc_best, "binary_precision": bin_prec_best, "binary_recall": bin_rec_best,
         "binary_specificity": bin_spec_best, "binary_f1": bin_f1_best,
         "multiclass_acc": mc_acc_val, "multiclass_f1": mc_f1_val, # Use mc_..._val
         "val_epoch_time_seconds": epoch_time, **{f"val_{k}": v for k, v in resources.items()}
    }
    if experiment: # Check if wandb run object exists
         try:
             experiment.log(metrics_log, step=epoch)
             cls_nms = [k for k,v in sorted(globals().get('diagnosis2idx',{}).items(),key=lambda i:i[1])] or [str(i) for i in range(globals().get('out_dim',0))]
             if len(binary_targets_np)>0:
                 experiment.log({"bin_cm_best_f1": wandb.plot.confusion_matrix(y_true=binary_targets_np.astype(int), preds=binary_preds_best, class_names=['non-mel','mel'])}, step=epoch)
                 experiment.log({"mc_cm": wandb.plot.confusion_matrix(y_true=TARGETS.astype(int), preds=PROBS.argmax(axis=1), class_names=cls_nms)}, step=epoch)
                 if bin_cm_best: experiment.log({"bin_cm_table": wandb.Table(columns=['Pred Non-Mel','Pred Mel'], data=bin_cm_best)}, step=epoch)
                 if mc_cm_val: experiment.log({"mc_cm_table": wandb.Table(columns=[f"Pred {n}" for n in cls_nms], data=mc_cm_val)}, step=epoch) # Use mc_cm_val
             if f1_hist: experiment.log({"f1_thresh_hist": wandb.Table(columns=["thresh","f1"], data=f1_hist)}, step=epoch)
         except Exception as log_e: print(f"Wandb val log failed: {log_e}")

    # --- ROC Data ---
    roc_data = None # Initialize roc_data to None
    # --- CORRECTED SYNTAX ---
    if len(binary_targets_np) > 0 :
        try:
            fpr, tpr, roc_thresholds = roc_curve(binary_targets_np, binary_probs_np) # Use adjusted probs? Maybe not needed for roc_curve
            # Check if roc_curve returned valid data (at least two points needed)
            if len(fpr) > 1 and len(tpr) > 1:
                roc_data = (fpr, tpr, roc_thresholds)
            else:
                 print("Warning: ROC curve calculation returned insufficient points.")
        except ValueError as roc_e:
             print(f"Warning: Could not compute ROC curve: {roc_e}")
        except Exception as e: # Catch other potential errors
             print(f"Warning: Unexpected error during ROC curve calculation: {e}")
    # --- END CORRECTION ---

    # --- Return ---
    # The return statement should be correct from the previous fix
    return (val_loss_avg, bin_auc_val, mc_auc_val, PROBS, TARGETS,
            bin_acc_best, bin_prec_best, bin_rec_best, bin_f1_best, bin_spec_best,
            mc_acc_val, mc_f1_val, mc_cm_val, bin_cm_best,
            current_temp, roc_data)


def run_single_model(model_type='efficientnetv2',
                     cnn_backbone_name='efficientnetv2_m',
                     transformer_backbone_name=None,
                     use_meta_flag=True,
                     use_external_flag=True,
                     target_total_epochs=40): # Keep target epochs
    """
    Main training function adapted for EffNetV2 / Swin Hybrid models.
    Uses PLAUSIBLE adaptive settings based on configuration, NO forced bias.
    Uses CONSISTENT augmentation and training schedules.
    """
    global use_meta, use_external, NOTEBOOK_START_TIME, MAX_RUNTIME_SECONDS
    use_meta = use_meta_flag
    use_external = use_external_flag

    print(f"--- Starting Plausible Training Run ---") # Changed title
    print(f"Model Type: {model_type}, CNN: {cnn_backbone_name}" + (f", TF: {transformer_backbone_name}" if transformer_backbone_name else ""))
    print(f"Using Meta: {use_meta}, Using External: {use_external}")

    # --- 1. Configuration & Parameter Adaptation ---
    print("\n--- 1. Configuring Run Parameters ---")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); print(f"Device: {device}")
    if torch.cuda.device_count() > 1: print(f"GPUs available: {torch.cuda.device_count()}")

    cnn_size_tag = get_v2_size_map(cnn_backbone_name)
    # *** Unpack only TWO values ***
    lr_adaptation_factor, regularization_factor = configure_dynamic_parameters(
        model_type, cnn_backbone_name, use_meta, use_external
    )

    # Batch Size / Accumulation (Keep adaptive based on CNN size)
    _batch_config = {'s': {'bs': 24, 'acc': 3}, 'm': {'bs': 18, 'acc': 4}, 'l': {'bs': 12, 'acc': 5}} # Adjusted for V2+SwinBase
    _current_bs_cfg = _batch_config.get(cnn_size_tag, _batch_config['m'])
    batch_size, accum_steps = _current_bs_cfg['bs'], _current_bs_cfg['acc']
    effective_bs = batch_size * accum_steps
    print(f"Batch Config ({cnn_size_tag}): BS={batch_size}, Accum={accum_steps} (Eff={effective_bs})")

    # --- *** Use CONSISTENT Training Schedule *** ---
    n_epochs = target_total_epochs # Total loop iterations

    # --- <<< START DEBUG MODIFICATION >>> ---
    debug_epochs = 25 # Set the number of epochs for DEBUG mode
    if DEBUG:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! DEBUG MODE ACTIVE: Overriding epochs to {debug_epochs} !!!")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        n_epochs = debug_epochs
    # --- <<< END DEBUG MODIFICATION >>> ---

    

    # --- Calculate phase durations relative to n_epochs ---
    # Aim for roughly: 20% freeze, 30% warmup, 50% cosine
    # Ensure minimums and that phases fit within n_epochs
    freeze_epo = min(n_epochs - 2, max(1, round(n_epochs * 0.20))) if n_epochs > 1 else 0
    warmup_epo = min(n_epochs - freeze_epo - 1, max(1, round(n_epochs * 0.30))) if (n_epochs - freeze_epo) > 1 else 0
    cosine_epo = n_epochs - freeze_epo - warmup_epo
    cosine_epo = max(0, cosine_epo) # Ensure non-negative
    # Simplified adjustment for DEBUG:
    if DEBUG and n_epochs <= 5 : # Very short runs
        freeze_epo = 1
        warmup_epo = max(0, n_epochs - 2) # Warmup for remaining epochs except last
        cosine_epo = max(0, n_epochs - freeze_epo - warmup_epo) # Likely 0 or 1
        print(f"DEBUG Schedule Adjustment: F={freeze_epo}, W={warmup_epo}, C={cosine_epo}")
    # --- Calculate unfreeze points relative to n_epochs ---
    # Start unfreezing shortly after the initial freeze ends
    unfreeze_start_epoch = freeze_epo + 1 # Start right after freeze
    # Full unfreeze happens partway through the warmup phase (e.g., ~60% into warmup)
    # Ensure it happens after start and before cosine starts
    unfreeze_point_in_warmup = max(1, round(warmup_epo * 0.60))
    full_unfreeze_epoch = min(freeze_epo + warmup_epo, freeze_epo + unfreeze_point_in_warmup)
    # Make sure full unfreeze epoch is at least one after start epoch
    full_unfreeze_epoch = max(unfreeze_start_epoch + 1, full_unfreeze_epoch)
    # Guard against n_epochs being too small
    if n_epochs <= freeze_epo + 1: full_unfreeze_epoch = n_epochs + 1 # Prevent unfreeze if only freeze exists

    freeze_duration, warmup_duration, cosine_duration = freeze_epo, warmup_epo, cosine_epo

    print(f"Using Training Schedule: Freeze={freeze_duration}, Warmup={warmup_duration}, Cosine={cosine_duration} (Total={n_epochs})")
    print(f"Unfreezing Schedule Points (Dynamic): Start E{unfreeze_start_epoch}, Full E{full_unfreeze_epoch}")
    
    # Image Size & Data Directories
    target_image_sizes = {'s': 384, 'm': 448, 'l': 480}
    image_size = 384
    data_dir_size = 384 if image_size <= 384 else 512
    data_dir_comp = f'../input/jpeg-melanoma-{data_dir_size}x{data_dir_size}'
    data_dir_ext = f'../input/jpeg-isic2019-{data_dir_size}x{data_dir_size}' if use_external else None
    print(f"Target Image Size: {image_size}, Data Dir Size: {data_dir_size}")
    print(f"Competition Data: {data_dir_comp}")
    if data_dir_ext: print(f"External Data: {data_dir_ext}")

    # Kernel Type String for Logging/Saving
    kb = f"{cnn_backbone_name.replace('efficientnet','effnet')}"
    if model_type == 'hybrid_swin': kb += "_swin"
    kernel_type = f"{kb}_{image_size}{'_ext' if use_external else ''}{'_meta' if use_meta else ''}"
    print(f"Generated kernel_type: {kernel_type}")
    # --- Define hyperparameters USED LATER here ---
    debug_sample_size = 1000
    split_random_state = 42
    llrd_decay_rate = 0.90 # Fixed choice for this implementation
    lambda_binary = 0.5    # Fixed choice
    loss_label_smoothing = 0.1 # Fixed choice
    warmup_multiplier = 10    # Fixed choice
    plateau_factor = 0.5      # Fixed choice
    plateau_patience = 5       # Fixed choice
    es_patience = 20          # Fixed choice
    es_delta = 0.003          # Fixed choice
    es_relative_delta = True  # Fixed choice
    val_split_ratio = 0.2
    es_score_weights = {'binary_auc': 0.5, 'binary_recall': 0.2, 'multiclass_auc': 0.2, 'val_loss': 0.1} # Fixed choice

    # --- WandB Initialization ---
    print("\n--- WandB Initialization ---")
    wandb_run = None
    try:
        wandb_config = {
            "kernel_type": kernel_type, "model_type": model_type, "cnn_backbone": cnn_backbone_name,
            "transformer_backbone": transformer_backbone_name, 
            "cnn_size_tag": cnn_size_tag,
            "use_external": use_external, "use_meta": use_meta, "image_size": image_size,
            "DEBUG": DEBUG,
            "target_total_epochs": n_epochs, 
            "batch_size": batch_size, "accum_steps": accum_steps, "effective_bs": effective_bs,
            "use_amp": use_amp, "init_lr": init_lr,
            "lr_adaptation_factor": lr_adaptation_factor, 
            "regularization_factor": regularization_factor, 
            "llrd_decay_rate": llrd_decay_rate, 
            "lambda_binary_loss_weight": lambda_binary, 
            "loss_label_smoothing": loss_label_smoothing, 
            "freeze_epochs": freeze_duration, "warmup_epochs": warmup_duration, 
            "cosine_epochs": cosine_duration, 
            "unfreeze_start": unfreeze_start_epoch, "unfreeze_full": full_unfreeze_epoch,
            "warmup_multiplier": warmup_multiplier, 
            "plateau_factor": plateau_factor, "plateau_patience": plateau_patience, 
            "es_patience": es_patience, "es_delta": es_delta, "es_relative_delta": es_relative_delta, 
            "es_warm_up": freeze_duration, 
            "es_score_weights": es_score_weights, 
            "num_workers": num_workers, "split_random_state": split_random_state
        }
        tz = pytz.timezone('Asia/Jakarta'); run_name = f"{kernel_type}_{datetime.now(tz).strftime('%y%m%d_%H%M')}"
        wandb_run = wandb.init(project="SKRIPSI GACOR", entity="arveda-ava86-universitas-gadjah-mada-library", config=wandb_config, name=run_name, tags=[cnn_size_tag, model_type, kernel_type])
        print(f"WandB run initialized: {run_name} (ID: {wandb_run.id if wandb_run else 'N/A'})")
    except Exception as e: print(f"Error initializing WandB: {e}"); traceback.print_exc(); wandb_run = None
    torch.cuda.empty_cache()

    
    # --- 2. Data Sampling & Splitting ---
    print("\n--- 2. Data Sampling & Splitting ---")
    df_train_to_split = None # DataFrame to use for train/val split
    # --- >>> ADD THIS PRINT STATEMENT <<< ---
    print(f"DEBUG CHECK: Type of split_random_state before sampling: {type(split_random_state)}, Value: {split_random_state}")
    # --- >>> END ADD <<< ---
    if DEBUG:
        # --- >>> ADD TYPE CHECK AND CORRECTION <<< ---
        if isinstance(debug_sample_size, tuple):
            print(f"Warning: debug_sample_size was a tuple {debug_sample_size}. Taking the first element.")
            if len(debug_sample_size) > 0 and isinstance(debug_sample_size[0], int):
                debug_sample_size = debug_sample_size[0]
            else:
                print("Error: Cannot correct debug_sample_size tuple. Reverting to default 200.")
                debug_sample_size = 200 # Fallback default
        elif not isinstance(debug_sample_size, int):
             print(f"Warning: debug_sample_size was not an int ({type(debug_sample_size)}). Reverting to default 200.")
             debug_sample_size = 200 # Fallback default
        # --- >>> END TYPE CHECK <<< ---

        print(f"\n!!! DEBUG MODE: Sampling up to {debug_sample_size} instances from df_train !!!") # Changed wording slightly

        # Now the comparison should work
        if len(df_train) <= debug_sample_size:
             print(f" Original df_train ({len(df_train)}) smaller than/equal to debug sample size. Using all.")
             df_train_to_split = df_train.copy()
        elif len(df_train['target'].unique()) > 1:
             try:
                 # Try stratified sampling first
                 required_frac = debug_sample_size / len(df_train)
                 df_train_to_split = df_train.groupby('target', group_keys=False).apply(lambda x: x.sample(n=max(1, int(len(x) * required_frac)), random_state=split_random_state))

                 # If oversampled due to rounding up in apply, sample down
                 if len(df_train_to_split) > debug_sample_size:
                      df_train_to_split = df_train_to_split.sample(debug_sample_size, random_state=split_random_state)
                 # If undersampled (e.g., tiny classes), top up with random sample if needed (less critical for debug)
                 # elif len(df_train_to_split) < debug_sample_size * 0.8:
                 #    print(" Stratified fraction resulted in fewer samples than expected, taking random sample instead.")
                 #    df_train_to_split = df_train.sample(debug_sample_size, random_state=split_random_state)


                 print(f" Stratified sampling successful for DEBUG (Target: {debug_sample_size}, Actual: {len(df_train_to_split)}).")
             except Exception as e:
                 print(f" Warning: Stratified sampling for DEBUG failed ({e}). Falling back to random sampling.")
                 df_train_to_split = df_train.sample(debug_sample_size, random_state=split_random_state)
        else:
             # Cannot stratify
             print("Warning: Only one class present in df_train target for DEBUG. Performing random sampling.")
             df_train_to_split = df_train.sample(min(debug_sample_size, len(df_train)), random_state=split_random_state)

        print(f" DEBUG Sampled df_train shape: {df_train_to_split.shape}")
        if not df_train_to_split.empty:
            print(f" DEBUG Sampled target distribution:\n{df_train_to_split['target'].value_counts(normalize=True).sort_index()}")
            
    else: # Not DEBUG mode
        # --- <<< CHANGE THIS BLOCK >>> ---
        # if len(df_train) > production_sample_size: # REMOVE or comment out this check
        #     print(f"\n!!! PRODUCTION MODE: Undersampling df_train...")
        #     # ... (undersampling code) ...
        # else:
        #     print(f"\n--- PRODUCTION MODE: Using full df_train ({len(df_train)} instances, <= {production_sample_size}) ---")
        #     df_train_to_split = df_train.copy()
    
        # --- Replace with ---
        print(f"\n--- PRODUCTION MODE: Using full df_train ({len(df_train)} instances) ---")
        df_train_to_split = df_train.copy()
        # --- <<< END CHANGE >>> ---



    # --- Perform Train/Validation Split on the (potentially sampled) data ---
    print(f"\nSplitting data (shape: {df_train_to_split.shape}) with val ratio {val_split_ratio}...")
    df_train_set = None
    df_valid_set = None
    if df_train_to_split.empty:
        print("Warning: DataFrame to split is empty. Creating empty train/valid sets.")
        df_train_set = pd.DataFrame(columns=df_train.columns)
        df_valid_set = pd.DataFrame(columns=df_train.columns)
    elif val_split_ratio > 0 and val_split_ratio < 1:
        # Check stratification possibility again on the sampled data
        if len(df_train_to_split['target'].unique()) > 1:
            try:
                df_train_set, df_valid_set = train_test_split(
                    df_train_to_split,
                    test_size=val_split_ratio,
                    random_state=split_random_state,
                    stratify=df_train_to_split['target'] # Stratify on the sampled data
                )
                print(" Stratified split successful.")
            except ValueError as e:
                 print(f" Warning: Stratified split on sampled data failed ({e}). Falling back to non-stratified split.")
                 df_train_set, df_valid_set = train_test_split(
                     df_train_to_split,
                     test_size=val_split_ratio,
                     random_state=split_random_state
                 )
        else:
             print("Warning: Only one class present in sampled df_train target. Performing non-stratified split.")
             df_train_set, df_valid_set = train_test_split(
                 df_train_to_split,
                 test_size=val_split_ratio,
                 random_state=split_random_state
             )
    elif val_split_ratio == 0:
         print("Validation split ratio is 0. Using all sampled data for training.")
         df_train_set = df_train_to_split.copy()
         df_valid_set = pd.DataFrame(columns=df_train.columns)
    else: # val_split_ratio >= 1
         print("Validation split ratio is >= 1. Using all sampled data for validation.")
         df_valid_set = df_train_to_split.copy()
         df_train_set = pd.DataFrame(columns=df_train.columns)

    # Reset index after final split
    if df_train_set is not None: df_train_set = df_train_set.reset_index(drop=True)
    if df_valid_set is not None: df_valid_set = df_valid_set.reset_index(drop=True)

    print(f" Final Train set shape: {df_train_set.shape}, Final Validation set shape: {df_valid_set.shape}")
    if not df_train_set.empty: print(f" Final Train set target distribution:\n{df_train_set['target'].value_counts(normalize=True).sort_index()}")
    if not df_valid_set.empty: print(f" Final Valid set target distribution:\n{df_valid_set['target'].value_counts(normalize=True).sort_index()}")

    # --- Update WandB Config with final data counts ---
    if wandb_run:
        try: # Add try-except around wandb calls
            wandb_run.config.update({
                "n_samples_total_before_split": len(df_train_to_split),
                "n_samples_train": len(df_train_set),
                "n_samples_valid": len(df_valid_set),
                "out_dim": out_dim,
                "mel_idx": mel_idx
            }, allow_val_change=True) # Allow changes post-init
        except Exception as e: print(f"Wandb config update failed (data counts): {e}")


    # --- Augmentations & DataLoaders ---
    print("\nSetting Augmentations & Creating DataLoaders...")
    transforms_val = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    if 'transforms_train' not in globals(): raise NameError("Global 'transforms_train' not defined.")

    dataset_train = SIIMISICDataset(df_train_set, 'train', 'train', transform=transforms_train)
    dataset_valid = SIIMISICDataset(df_valid_set, 'train', 'val', transform=transforms_val)

    # --- Weighted Sampler Logic ---
    train_sampler = None
    # Run sampler if train set is not empty (applies to DEBUG=False undersampled case too)
    if not df_train_set.empty:
        target_counts = df_train_set['target'].value_counts().sort_index()
        target_counts = target_counts.replace(0, 1e-9)
        num_samples = len(df_train_set)
        num_classes = len(target_counts)

        if num_classes > 0 and num_samples > 0:
            class_weights_map = (num_samples / (num_classes * target_counts))
            sample_weights = torch.from_numpy(df_train_set['target'].map(class_weights_map).values).double()
            if (sample_weights <= 0).any():
                print("Warning: Clamping non-positive sample weights.")
                sample_weights = torch.clamp(sample_weights, min=1e-9)

            # Ensure num_samples for sampler is reasonable, especially for small debug sets
            sampler_num_samples = num_samples # Draw 'num_samples' per epoch by default
            if DEBUG and num_samples < batch_size * 2: # Heuristic for very small debug sets
                 sampler_num_samples = max(num_samples, batch_size) # Ensure at least one batch is drawn
                 print(f" Adjusting sampler num_samples to {sampler_num_samples} for small DEBUG set.")

            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=sampler_num_samples,
                replacement=True
            )
            print(f"WeightedRandomSampler Enabled (Classes: {num_classes}, Samples Drawn: {sampler_num_samples})")
        else:
            print("WeightedRandomSampler Disabled: Not enough classes or samples in train_set.")
    else:
        print(f"WeightedRandomSampler Disabled (Train set empty)")


    # --- DataLoaders ---
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None and not df_train_set.empty),
        num_workers=num_workers, pin_memory=True,
        drop_last= (not df_train_set.empty)
    ) if not df_train_set.empty else None

    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    ) if not df_valid_set.empty else None

    print(f"Loaders Ready:")
    print(f"  Train Loader: {'Created' if train_loader else 'Skipped'}" + (f" ({len(train_loader)} batches)" if train_loader else ""))
    print(f"  Valid Loader: {'Created' if valid_loader else 'Skipped'}" + (f" ({len(valid_loader)} batches)" if valid_loader else ""))



    # --- 3. Creating Model ---
    print("\n--- 3. Creating Model ---")
    model=None
    try:
        if model_type == 'hybrid_swin':
            model = HybridSwinModel(
                cnn_backbone_name=cnn_backbone_name,
                transformer_backbone_name=transformer_backbone_name,
                out_dim=out_dim,                 
                n_meta_features=n_meta_features, 
                pretrained=True,
                image_size=image_size,
                regularization_factor=regularization_factor
            )
        # --- ADD THIS NEW CONDITION ---
        elif model_type == 'swin_only':
            print(">>> Running in Swin Transformer ONLY mode <<<")
            model = HybridSwinModel(
                cnn_backbone_name=None, # <<< PASS NONE HERE TO DISABLE CNN
                transformer_backbone_name=transformer_backbone_name,
                out_dim=out_dim,                 
                n_meta_features=n_meta_features, 
                pretrained=True,
                image_size=image_size,
                regularization_factor=regularization_factor
            )
            
        elif model_type == 'efficientnetv2':
            model = EffNetV2Model(
                backbone_name=cnn_backbone_name,
                out_dim=out_dim,                 
                n_meta_features=n_meta_features,
                pretrained=True,
                regularization_factor=regularization_factor
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    except Exception as model_init_e:
        print(f"!!! ERROR initializing model: {model_init_e}")
        traceback.print_exc() # Print full traceback for model init errors
        raise model_init_e

    if model is None: raise RuntimeError("Model init failed silently.")
    # --- >>> ADD DROPOUT LOGGING AFTER MODEL CREATION <<< ---
    if wandb_run and 'model_ref' in locals() and model_ref is not None:
        try:
            dropout_config = {}
            # Log classifier dropout (exists in both models)
            if hasattr(model_ref, 'initial_classifier_dropout'):
                 dropout_config["initial_dropout_classifier"] = model_ref.initial_classifier_dropout
            # Log fusion dropout (only in hybrid)
            if hasattr(model_ref, 'initial_fusion_dropout'):
                 dropout_config["initial_dropout_fusion"] = model_ref.initial_fusion_dropout
            # Log image dropout if implemented and used
            # if hasattr(model_ref, 'initial_image_dropout'):
            #      dropout_config["initial_dropout_image"] = model_ref.initial_image_dropout

            if dropout_config: # Only update if we found dropout values
                wandb_run.config.update(dropout_config, allow_val_change=True)
                print(f"Logged effective initial dropout rates to WandB: {dropout_config}")
        except Exception as e: print(f"Wandb config update failed (dropout): {e}")
  
    # Determine base model reference and check for DataParallel
    model = model.to(device)
    model_ref = model 
    is_parallel = False
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs. Wrapping model with DataParallel.")
        model = nn.DataParallel(model)
        model_ref = model.module 
        is_parallel = True
    print(f"Model Instantiated ({type(model_ref).__name__}). Parallel: {is_parallel}")
  
    initial_freeze_block=2; print(f"Applying initial freeze up to CNN block {initial_freeze_block}...")
    partial_freeze_enet(model, freeze_until_block=initial_freeze_block)
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"Initially Trainable: {trainable_params:,}")
    print(f"Model check: Has meta? {'Yes' if hasattr(model_ref,'meta_attention') else 'No'}, n_meta={getattr(model_ref,'n_meta_features','N/A')}")

    # --- 4. Loss, Optimizer, Scheduler ---
    print("\n--- 4. Loss, Optimizer, Scheduler ---")
    
    if not df_train_set.empty:
        counts=df_train_set['target'].value_counts().reindex(range(out_dim),fill_value=1e-6)
        w_raw=torch.FloatTensor([1.0/c for c in counts]).to(device); class_weights=w_raw/w_raw.sum()*out_dim
    else: class_weights = torch.ones(out_dim, device=device)
    criterion_multi = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).to(device)
    lambda_binary = 0.5 


    base_lr = init_lr
    initial_lr = base_lr * 0.01 
    param_groups = []

    print("Defining initial optimizer groups...")
    initially_optimized_params = 0
    head_params = []
    backbone_params = [] 
    head_param_ids = set()
    for part_name in ['myfc', 'classifier', 'meta_attention', 'meta_fc', 'fusion_layer']:
        module = getattr(model_ref, part_name, None)
        if module and not isinstance(module, nn.Identity):
             head_param_ids.update(id(p) for p in module.parameters())
    # Assign parameters to groups based on requires_grad and location
    print(" Iterating through model parameters for initial optimizer...")
    for name, param in model.named_parameters(): # Iterate through potentially wrapped model
        if param.requires_grad: # Check if it's trainable after partial_freeze_enet
            if id(param) in head_param_ids:
                 head_params.append(param)
            else: # Assume it's part of a backbone if requires_grad and not in head
                 backbone_params.append(param)

    # Add groups if they have parameters
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': initial_lr, 'weight_decay': 0.01}) # Low WD for backbone parts
        count = sum(p.numel() for p in backbone_params)
        initially_optimized_params += count
        print(f"  Added initial optimizer group for 'Trainable Backbone Parts' ({len(backbone_params)} params, {count:,} elements) LR={initial_lr:.1e}, WD=0.01")

    if head_params:
        # Use a slightly higher LR for the head initially
        head_initial_lr = initial_lr * 5
        param_groups.append({'params': head_params, 'lr': head_initial_lr, 'weight_decay': 0.05}) # Higher WD for head parts
        count = sum(p.numel() for p in head_params)
        initially_optimized_params += count
        print(f"  Added initial optimizer group for 'Head Parts' ({len(head_params)} params, {count:,} elements) LR={head_initial_lr:.1e}, WD=0.05")

    if not param_groups:
        # This should only happen if partial_freeze_enet made nothing trainable
        raise RuntimeError("No parameters requiring grad found for initial optimizer! Check partial_freeze_enet.")

    optimizer = optim.AdamW(param_groups, eps=1e-7)
    print(f"Initial optimizer created with {len(param_groups)} groups.")
    print(f"Total parameters in initial optimizer: {initially_optimized_params:,}") 
    scaler = GradScaler() if use_amp else None

    warmup_multiplier = 10
    sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, cosine_duration), eta_min=base_lr * 0.001)
    sched_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=warmup_multiplier, total_epoch=max(1, warmup_duration), after_scheduler=sched_cosine)
    sched_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7) 
    print("Schedulers initialized.")
    es_checkpoint_path = f'{kernel_type}_best.pth'
    early_stopping = EarlyStopping(
        patience=20, mode='max', delta=0.003, relative_delta=True,
        warm_up=freeze_duration, verbose=True, checkpoint_path=es_checkpoint_path,
        score_weights={'binary_auc': 0.5, 'binary_recall': 0.2, 'multiclass_auc': 0.2, 'val_loss': 0.1} 
    )
    print(f"Early stopping active after epoch {freeze_duration}, checkpoint: {es_checkpoint_path}")


    # --- 5. WandB Setup ---
    # Watch model - log gradients and parameters (optional, can be verbose)
    if wandb and wandb.run:
        try:
            # Exclude embedding layers if they cause issues
            wandb.watch(model, log='all', log_freq=max(100, len(train_loader)//2), idx=0, log_graph=False)
            print("WandB watching model.")
        except Exception as e:
            print(f"WandB watch failed: {e}")


    # --- 6. Training Loop ---
    print(f"\n--- 6. Starting Training Loop (Target: {n_epochs} Epochs) ---")
    # ... (Loop initializations) ...
    train_losses, val_losses = [], []; best_model_state, best_metrics = None, None; best_PROBS, best_TARGETS = None, None; best_epoch_num=0; best_score=float('-inf'); total_start_time=time.time(); main_cosine_scheduler=None; last_epoch_completed=0; early_stopping.reset()
    main_scheduler = None # This will hold the active scheduler (Warmup->Cosine or just Cosine)
    time_limit_reached = False # Flag to indicate why the loop stopped
    try:
        for epoch in range(1, n_epochs + 1):
            last_epoch_completed = epoch; epoch_start_time = time.time()
            print(f"\n===== Epoch {epoch}/{n_epochs} =====") # Removed profile from print
            # <<< --- START RUNTIME CHECK --- >>>
            current_time = time.time()
            elapsed_seconds = current_time - NOTEBOOK_START_TIME
            if elapsed_seconds >= MAX_RUNTIME_SECONDS:
                print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"!!! RUNTIME LIMIT REACHED ({elapsed_seconds:.0f}s / {MAX_RUNTIME_SECONDS}s) !!!")
                print(f"!!! Stopping training loop before starting Epoch {epoch}.   !!!")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                time_limit_reached = True
                break # Exit the training loop
            else:
                print(f"Runtime Check: {elapsed_seconds/3600:.2f} hours elapsed (Limit: {MAX_RUNTIME_SECONDS/3600:.2f} hours)")
            # <<< --- END RUNTIME CHECK --- >>>
            # Set dynamic dropout (uses regularization_factor)
            if hasattr(model_ref, 'set_epoch'): model_ref.set_epoch(epoch)

            # --- Progressive Layer Unfreezing (Use FIXED schedule points) ---
            optimizer, optimizer_changed = progressive_layer_unfreezing(
                model, optimizer, epoch, initial_freeze_block,
                unfreeze_start_epoch=unfreeze_start_epoch,
                full_unfreeze_epoch=full_unfreeze_epoch,
                total_epochs=n_epochs, base_lr=base_lr,
                cnn_backbone_name=cnn_backbone_name,
                lr_adaptation_factor=lr_adaptation_factor,
                llrd_decay_rate=0.90,
                regularization_factor=regularization_factor                
                )

# --- >>> REVISED Scheduler Stepping Logic <<< ---

            # Phase 1: Initial Freeze (Epochs 1 to freeze_duration)
            if epoch <= freeze_duration:
                print("Scheduler Phase: Initial Freeze (No Step)")
                # Optimizer LR is low (initial_lr), no scheduler step needed

            # Phase 2: Initial Warmup (Epochs freeze_duration + 1 to freeze_duration + warmup_duration)
            elif epoch == freeze_duration + 1: # First epoch of initial warmup
                 print(f"Scheduler Phase: Starting Initial Warmup (Epochs {epoch} to {freeze_duration + warmup_duration})")
                 # Initialize the first main_scheduler (WarmupV2 -> Cosine)
                 sched_cosine_init = CosineAnnealingLR(optimizer, T_max=max(1, cosine_duration), eta_min=base_lr * 0.001)
                 sched_warmup_init = GradualWarmupSchedulerV2(optimizer, multiplier=warmup_multiplier, total_epoch=max(1, warmup_duration), after_scheduler=sched_cosine_init)
                 main_scheduler = sched_warmup_init
                 main_scheduler.step() # Step for the current epoch (epoch 1 of warmup)
            elif epoch <= freeze_duration + warmup_duration:
                 if main_scheduler: main_scheduler.step() # Continue stepping initial warmup/cosine
                 else: print("Error: Initial scheduler missing during warmup phase!")

            # Phase 3: Post-Warmup / Post-Unfreeze Reset
            else: # epoch > freeze_duration + warmup_duration
                if optimizer_changed: # This happens *exactly* at full_unfreeze_epoch
                     print("Optimizer Changed! Resetting FULL scheduler sequence.")
                     remaining_epochs_total = n_epochs - epoch + 1
                     # Define a *new, shorter* warmup + cosine for the remaining epochs
                     # Example: 5 epochs warmup, rest cosine. Adjust as needed.
                     new_warmup_epochs = min(5, max(1, remaining_epochs_total // 4))
                     new_cosine_epochs = remaining_epochs_total - new_warmup_epochs
                     print(f"  New schedule phase: Warmup={new_warmup_epochs}, Cosine={new_cosine_epochs}")

                     if new_warmup_epochs + new_cosine_epochs != remaining_epochs_total:
                          print("  Warning: New schedule epoch calculation mismatch.")
                          new_cosine_epochs = max(0, remaining_epochs_total - new_warmup_epochs) # Ensure non-negative

                     # Create NEW schedulers based on the *new* optimizer state
                     # Use the LRs set by progressive_layer_unfreezing
                     sched_cosine_reset = CosineAnnealingLR(optimizer, T_max=max(1, new_cosine_epochs), eta_min=base_lr * 0.0001) # Lower final LR maybe?
                     sched_warmup_reset = GradualWarmupSchedulerV2(optimizer, multiplier=warmup_multiplier, total_epoch=max(1, new_warmup_epochs), after_scheduler=sched_cosine_reset)
                     main_scheduler = sched_warmup_reset # OVERWRITE main_scheduler

                     # Reset Plateau scheduler with the new optimizer
                     sched_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7); sched_plateau._reset()
                     print("  Plateau scheduler reset.")

                     main_scheduler.step() # Step the *new* scheduler for the current epoch

                elif main_scheduler: # Optimizer didn't change (already past full_unfreeze_epoch)
                     print("Scheduler Phase: Stepping Existing Post-Warmup Scheduler")
                     main_scheduler.step() # Step the active scheduler (could be warmup_reset or cosine_reset)
                else:
                     # This case should ideally not be reached if logic is correct
                     print("Error: Main scheduler missing in post-warmup/post-reset phase!")
                     # Fallback: Create a simple cosine scheduler for remaining epochs
                     remaining_at_fallback = n_epochs - epoch + 1
                     main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, remaining_at_fallback), eta_min=base_lr * 0.001)
                     main_scheduler.step()

                # --- >>> END REVISED Scheduler Stepping Logic <<< ---
            # Log LR
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.2e}")
            if wandb and wandb.run: wandb.log({"learning_rate": current_lr}, step=epoch)

            # --- Train ---
            if len(train_loader) > 0:
                train_loss = train_epoch(
                    model, train_loader, optimizer, wandb, epoch, scaler=scaler,
                    accum_steps=accum_steps, criterion_multi=criterion_multi,
                    mel_idx=mel_idx, lambda_binary=lambda_binary, device=device
                )
                if train_loss is not None: train_losses.append(train_loss)
            else:
                print("Skipping train epoch: train_loader is empty.")
                train_loss = None # Signify no training loss

            # --- Validate ---
            if len(valid_loader) > 0:
                val_results = val_epoch(
                    model, valid_loader, wandb, epoch, n_test=1, # n_test=1 assumes no TTA
                    criterion_multi=criterion_multi, mel_idx=mel_idx,
                    lambda_binary=lambda_binary, device=device, use_amp=use_amp
                )
                # Unpack all 15 results
                (val_loss, bin_auc, mc_auc, PROBS, TARGETS, bin_acc, bin_prec,
                 bin_rec, bin_f1, bin_spec, mc_acc, mc_f1, mc_cm, bin_cm, temp, roc) = val_results
                val_losses.append(val_loss)

                # --- Early Stopping & Best Model Check ---
                # Normalize loss for ES (higher score is better)
                normalized_val_loss = max(0, 1.0 - min(val_loss / 5.0, 1.0)) # Adjust divisor if loss range differs
                metrics_for_es = {
                    'binary_auc': bin_auc if not np.isnan(bin_auc) else 0.0,
                    'val_loss': normalized_val_loss,
                    'binary_recall': bin_rec if not np.isnan(bin_rec) else 0.0,
                    'multiclass_auc': mc_auc if not np.isnan(mc_auc) else 0.0,
                }
                early_stopping(metrics_for_es, model, epoch) # Check ES

                composite_score = sum(
                    early_stopping.score_weights.get(m, 0) * metrics_for_es.get(m, 0)
                    for m in early_stopping.score_weights
                )

                # Log comprehensive validation metrics
                if wandb and wandb.run:
                    wandb.log({
                        "composite_score": composite_score, "val_loss_raw": val_loss,
                        **metrics_for_es, # Include metrics used for ES score
                        'binary_f1': bin_f1, 'binary_acc': bin_acc, 'binary_precision': bin_prec,
                        'binary_specificity': bin_spec, 'multiclass_acc': mc_acc, 'multiclass_f1': mc_f1,
                        'temperature': temp
                    }, step=epoch)

                # Save best model state if score improved
                if composite_score > best_score:
                    best_score = composite_score
                    best_model_state = model_ref.state_dict() # Get state from base model
                    best_epoch_num = epoch
                    best_metrics = { # Store detailed metrics for the best epoch
                        'epoch': epoch, 'composite_score': composite_score, 'val_loss_raw': val_loss,
                        'binary_auc': bin_auc, 'multiclass_auc': mc_auc, 'binary_recall': bin_rec,
                        'binary_f1': bin_f1, 'binary_acc': bin_acc, 'binary_precision': bin_prec,
                        'binary_specificity': bin_spec, 'multiclass_acc': mc_acc, 'multiclass_f1': mc_f1,
                        'temperature': temp, 'binary_cm': bin_cm, 'multiclass_cm': mc_cm
                    }
                    best_PROBS = PROBS
                    best_TARGETS = TARGETS
                    print(f"*** New Best Score: {best_score:.6f} at Epoch {epoch} ***")
                    # Save temporary best model (overwritten each time)
                    torch.save({'epoch': best_epoch_num, 'model_state_dict': best_model_state},
                               early_stopping.checkpoint_path) # Use ES path

                # --- Plateau Scheduler Step ---
                # Step based on composite score during cosine phase
                if epoch > freeze_duration + warmup_duration:
                    if hasattr(sched_plateau, 'optimizer') and sched_plateau.optimizer is optimizer:
                        sched_plateau.step(composite_score)
                    elif optimizer_changed:
                        pass # Optimizer was reset, plateau scheduler already updated
                    else: # Fallback if optimizer reference somehow differs
                        print("Warn: Recreating Plateau scheduler due to optimizer mismatch.")
                        sched_plateau=ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=5,verbose=True,min_lr=1e-7)
                        sched_plateau._reset()
                        sched_plateau.step(composite_score)

            else: # No validation data
                print("Skipping validation epoch: valid_loader is empty.")
                # Cannot check early stopping or update best model

            # --- Check Early Stopping Trigger ---
            if early_stopping.early_stop:
                print(f"EARLY STOPPING triggered after epoch {epoch}.")
                break

            print(f"Epoch {epoch} completed in {time.time() - epoch_start_time:.2f}s.")
            torch.cuda.empty_cache() # Clear cache at end of epoch
        # --- End MAIN TRAINING LOOP ---

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training loop: {e}")
        logging.error(traceback.format_exc()) # Log detailed error
    finally:
        # --- Final Operations ---
        total_time = time.time() - total_start_time
        loop_stop_reason = "Runtime Limit" if time_limit_reached else ("Early Stopping" if early_stopping.early_stop else "Completed Epochs")
        print(f"\n===== Training Finished / Stopped (Epoch {last_epoch_completed}/{n_epochs}, Reason: {loop_stop_reason}) =====") # <--- Modified print
        print(f"Total time: {total_time:.2f}s")

        # Reload best model state if early stopping happened and we have a best epoch recorded
        if early_stopping.early_stop and best_epoch_num > 0:
            best_model_path_es = early_stopping.checkpoint_path
            if os.path.exists(best_model_path_es):
                print(f"Reloading best model state from Early Stopping checkpoint (Epoch {best_epoch_num})...")
                try:
                    checkpoint = torch.load(best_model_path_es)
                    # Load into the base model structure first
                    temp_model = model_ref.__class__( # Re-instantiate base model
                         backbone_name=cnn_backbone_name, # Or use appropriate args for hybrid
                         # Add other necessary args based on model type...
                         transformer_backbone_name=transformer_backbone_name if model_type=='hybrid_swin' else None,
                         out_dim=out_dim, n_meta_features=n_meta_features, image_size=image_size,
                         pretrained=False # Don't need pretrained weights here
                    )
                    temp_model.load_state_dict(checkpoint['model_state_dict'])
                    best_model_state = temp_model.state_dict() # Get the clean state_dict
                    print(" Successfully reloaded best model state.")
                except Exception as load_err:
                    print(f" Warning: Failed to reload best model state from {best_model_path_es}. Error: {load_err}")
                    # Keep the state from before loop end if loading fails
            else:
                print(f"Warning: Early stopping triggered, but checkpoint file not found at {best_model_path_es}.")


        # Prepare final metrics dictionary
        final_log_metrics = {}
        if best_metrics:
            print(f"Using metrics from Best Epoch: {best_epoch_num}")
            final_log_metrics = best_metrics.copy() # Use copy
        elif last_epoch_completed > 0 and 'val_results' in locals():
            print(f"WARNING: No best metrics saved. Using metrics from last completed validation epoch ({last_epoch_completed}).")
             # Reconstruct from last val_results if needed (simplified example)
            final_log_metrics = {'epoch': last_epoch_completed,
                                'val_loss_raw': val_loss, 'binary_auc': bin_auc, 'multiclass_auc': mc_auc}
            if best_model_state is None and 'model' in locals(): # Use last model state if no best saved
                 best_model_state = model_ref.state_dict()
            best_epoch_num = last_epoch_completed # Mark as last epoch if using last metrics
        else:
            print("WARNING: No validation results available to log final metrics.")

        # Add training summary stats
        final_log_metrics['total_training_time_sec'] = round(total_time, 2)
        final_log_metrics['last_epoch_completed'] = last_epoch_completed
        final_log_metrics['best_epoch_logged'] = best_epoch_num # Epoch whose metrics are being logged

        # Log final summary to WandB
        if wandb and wandb.run:
             summary = {f"final_{k}": v for k, v in final_log_metrics.items() if isinstance(v, (int, float, bool, str))}
             print("\nFinal Summary Metrics:")
             for k, v in sorted(summary.items()): print(f"  - {k}: {v}")
             try: wandb.summary.update(summary)
             except Exception as wb_err: print(f"Warning: Failed to update WandB summary: {wb_err}")

        # --- Save Final Artifacts ---
        if best_model_state is not None and best_epoch_num > 0:
            # Use the explicitly saved ES checkpoint path as the final path now
            final_model_path = early_stopping.checkpoint_path
            # Re-save with additional info if needed, or just use the ES checkpoint
            print(f"\nFinal Best Model saved at: {final_model_path}")
            # Optionally save metrics dict separately or add to checkpoint
            # torch.save({ ... 'final_metrics': final_log_metrics ... }, final_model_path) # Example if re-saving

            # Log Model Artifact to WandB
            if wandb and wandb.run:
                model_artifact_name = f"model-{wandb.run.id}-final"
                description=f"Final model ({kernel_type}, Best E{best_epoch_num})"
                try:
                    model_artifact = wandb.Artifact(model_artifact_name, type="model",
                                                    description=description, metadata=summary)
                    if os.path.exists(final_model_path):
                         model_artifact.add_file(final_model_path)
                         wandb.log_artifact(model_artifact, aliases=["best", f"E{best_epoch_num}"])
                         print("Logged FINAL model artifact to WandB.")
                    else: print(f"Warning: Final model file {final_model_path} not found for artifact logging.")
                except Exception as art_err: print(f"Warning: Failed logging model artifact: {art_err}")

            # and if wandb run is active
            if best_PROBS is not None and best_TARGETS is not None and \
               wandb is not None and wandb.run is not None:
                preds_artifact_name = f"preds-{wandb.run.id}-final"
                preds_desc = f"Val preds/tgts (Best E{best_epoch_num})"
                probs_fn = f"best_probs_E{best_epoch_num}_{wandb.run.id}.npy"
                tgts_fn = f"best_tgts_E{best_epoch_num}_{wandb.run.id}.npy"
                valid_ids_fn = f"best_valid_ids_E{best_epoch_num}_{wandb.run.id}.csv"

                try:
                    probs_np = np.array(best_PROBS)
                    tgts_np = np.array(best_TARGETS)
                    if 'df_valid_set' in locals() and df_valid_set is not None and not df_valid_set.empty:
                         valid_ids_df = df_valid_set[['image_name']].copy() 
                         valid_ids_df.to_csv(valid_ids_fn, index=False)
                         print(f" Saved validation IDs ({len(valid_ids_df)}) to {valid_ids_fn}")
                    else:
                         print("Warning: Could not save validation IDs (df_valid_set unavailable or empty).")
                         valid_ids_fn = None 

                    np.save(probs_fn, probs_np); np.save(tgts_fn, tgts_np)
                    print(f" Saved best predictions ({probs_np.shape}) to {probs_fn}")
                    print(f" Saved best targets ({tgts_np.shape}) to {tgts_fn}")

                    pred_artifact = wandb.Artifact(preds_artifact_name, type="val_predictions", description=preds_desc)
                    pred_artifact.add_file(probs_fn); pred_artifact.add_file(tgts_fn)

                    if valid_ids_fn and os.path.exists(valid_ids_fn):
                         pred_artifact.add_file(valid_ids_fn)
                    wandb.log_artifact(pred_artifact, aliases=["best_preds", f"E{best_epoch_num}"])
                    print("Logged final predictions artifact (including validation IDs).")

                    # Clean up local files after logging
                    try:
                        os.remove(probs_fn); os.remove(tgts_fn)
                        if valid_ids_fn and os.path.exists(valid_ids_fn): os.remove(valid_ids_fn)
                    except OSError as e: print(f" Warning: Failed to remove temporary prediction files: {e}")

                except Exception as e: print(f"Error saving/logging prediction artifacts: {e}")
        else:
            print("\nNo best model state recorded or best epoch is 0, skipping final artifact saving.")

        # Log Final CM Tables from best_metrics if available
        if best_metrics and wandb and wandb.run:
            bin_cm=best_metrics.get('bin_cm',[]); multi_cm=best_metrics.get('mc_cm',[])
            cls_nms=[k for k,v in sorted(globals().get('diagnosis2idx',{}).items(),key=lambda i:i[1])] or [str(i) for i in range(globals().get('out_dim',0))]
            try:
                 if bin_cm: wandb.log({f"binary_cm_table_P": wandb.Table(columns=['Pred Non-Mel','Pred Mel'],data=bin_cm)}, step=n_epochs)
                 if multi_cm and len(multi_cm)==len(cls_nms): wandb.log({f"multiclass_cm_table_P": wandb.Table(columns=[f"Pred {n}" for n in cls_nms],data=multi_cm)}, step=n_epochs)
            except Exception as e: print(f"Warn: Failed log final CMs: {e}")

    # --- Function Return ---
    print("--- Exiting Adaptive Training Run ---")
    # Return single factors AND the wandb_run object
    return (model, best_model_state, final_log_metrics, train_losses, val_losses,
            best_PROBS, best_TARGETS, df_valid_set, diagnosis2idx, best_epoch_num,
            kernel_type, early_stopping, model_type, cnn_backbone_name, n_meta_features,
            image_size, out_dim,
            lr_adaptation_factor, regularization_factor,
            wandb_run # <<< ADD wandb_run HERE
           ) # 20 values returned now


import time
if __name__ == "__main__":

    # Use the globally defined configuration directly
    print(f"Starting run: {model_type} / {cnn_backbone_name}" + (f" / {transformer_backbone_name}" if model_type == 'hybrid_swin' else ""))
    start_time = time.time()

    # --- Call the modified function using GLOBAL variables ---
    (model, best_model_state, final_metrics, train_losses, val_losses,
     best_PROBS, best_TARGETS,
     df_valid_set, diagnosis2idx,
     best_epoch_num, kernel_type_ret, early_stopping,
     model_type_ret, cnn_backbone_ret, n_meta_features_ret,
     image_size_ret, out_dim_ret,
     lr_adaptation_factor_ret, regularization_factor_ret,
     wandb_run_ret
    ) = run_single_model(
        model_type=model_type, # Use global
        cnn_backbone_name=cnn_backbone_name, # Use global
        # --- CORRECTED LINE ---
        transformer_backbone_name=transformer_backbone_name if model_type in ['hybrid_swin', 'swin_only'] else None,
        # --- END CORRECTION ---
        use_meta_flag=use_meta, # Use global
        use_external_flag=use_external, # Use global
        target_total_epochs=target_total_epochs # Use global
        # DEBUG flag is used internally by run_single_model based on global value
    )
    # --- END Function Call ---

    total_time = time.time() - start_time
    print(f"\nTotal Run Time (Training): {total_time:.2f}s")
    print("\nFinal Run Metrics Summary (from run_single_model):")
    if final_metrics:
        # Print scalar metrics nicely
        scalar_metrics = {k: v for k, v in final_metrics.items() if isinstance(v, (int, float, bool, str))}
        for k, v in sorted(scalar_metrics.items()):
            print(f"  - {k}: {v}")
    else:
        print("  No final metrics available.")

# Grad-CAM prerequisites check
    if 'df_valid_set' in locals() and 'best_model_state' in locals():
        print("\nPrerequisites for Grad-CAM seem available.")
    else:
        print("\nWarning: Prerequisites for Grad-CAM might be missing.")


print("\nAttempting to finish WandB run...")
if 'wandb_run_ret' in locals() and wandb_run_ret is not None:
    try:
        # Check if the run associated with the object is still active before finishing
        # Note: Accessing wandb.run might be None even if wandb_run_ret exists if finish was called elsewhere
        if wandb.run and wandb.run.id == wandb_run_ret.id:
             print(f"Finishing active WandB run: {wandb_run_ret.id}")
             wandb.finish()
             print("WandB run finished.")
        elif wandb.run:
             print(f"Warning: Another WandB run seems active ({wandb.run.id}). Not finishing the target run ({wandb_run_ret.id}).")
        else:
             print(f"WandB run ({wandb_run_ret.id}) already finished or object is detached.")
    except Exception as e:
        print(f"Error finishing WandB run: {e}")
elif 'wandb' in locals() and wandb.run is not None:
    # Fallback if wandb_run_ret wasn't captured but a run is somehow still active
    print(f"Finishing potentially active global WandB run: {wandb.run.id}")
    wandb.finish()
    print("WandB run finished.")
else:
    print("No active WandB run object found to finish.")

