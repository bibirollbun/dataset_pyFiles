!pip install kaggle --quiet


# Install necessary libraries that may not be available by default in Kaggle

# pydicom: For reading DICOM files
# nibabel: For working with NIfTI files (reading, writing)
# monai: Medical imaging AI framework (includes augmentations for 3D data)
# SimpleITK: Useful for medical image processing including NIfTI and DICOM support
# torchio: Alternative 3D medical image processing and augmentation toolkit

# Only install if not already available
try:
    import pydicom, nibabel, monai, SimpleITK, torchio
except ImportError:
    !pip install -q pydicom nibabel monai SimpleITK torchio

# Import standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import json
import random
from pathlib import Path

# Import DICOM and NIfTI handling libraries
import pydicom
import nibabel as nib
import SimpleITK as sitk

# Import PyTorch and MONAI for model training and augmentations
import torch
import monai

# For progress bars
from tqdm.notebook import tqdm

# For warnings suppression (optional)
import warnings
warnings.filterwarnings('ignore')

# Ensure Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Automatically use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("Environment setup complete.")


import os
import shutil

# Make sure the .config/kaggle directory exists
os.makedirs("/root/.config/kaggle", exist_ok=True)

# Move kaggle.json to expected directory
shutil.copy("/kaggle/input/kaggle-json/kaggle (1).json", "/root/.config/kaggle/kaggle.json")

# Set permissions (optional but recommended)
os.chmod("/root/.config/kaggle/kaggle.json", 0o600)

# Now import and authenticate
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


# model_from_dataset_path = "/kaggle/input/densenet121-model/model_best.pth"

# if os.path.exists(model_from_dataset_path):
#     print(f"âœ… Found model in dataset: {model_from_dataset_path}")
#     model.load_state_dict(torch.load(model_from_dataset_path, map_location=device))
# else:
#     print("âš ï¸� No model found in Kaggle dataset input.")


# Paths
DATASET_ROOTS = [
    '/kaggle/input/abdominal-nifti-0-100',
    '/kaggle/input/abdominal-nifti-100-200',
    '/kaggle/input/abdominal-trauma-nifti-200-300',
    '/kaggle/input/abdominal-trauma-nifti-300-400',
    '/kaggle/input/abdominal-trauma-nifti-400-500',
    '/kaggle/input/abdominal-trauma-nifti-500-600',
    '/kaggle/input/abdominal-trauma-nifti-600-700',
    '/kaggle/input/abdominal-trauma-nifti-700-800',
    '/kaggle/input/abdominal-trauma-nifti-800-900',
    '/kaggle/input/abdominal-trauma-nifti-900-1000',
    
    '/kaggle/input/abdominal-nifti-1000-1100',
    '/kaggle/input/abdominal-nifti-1100-1200',
    '/kaggle/input/abdominal-nifti-1200-1300',
    '/kaggle/input/abdominal-nifti-1300-1400',
    '/kaggle/input/abdominal-nifti-1400-1500',
    '/kaggle/input/abdominal-nifti-1500-1600',
    '/kaggle/input/abdominal-nifti-1600-1700',
    '/kaggle/input/abdominal-nifti-1700-1800',
    '/kaggle/input/abdominal-nifti-1800-1900',
    '/kaggle/input/abdominal-nifti-1900-2000',
    
    '/kaggle/input/abdominal-trauma-nifti-2000-above',
    '/kaggle/input/abdominal-nifti-2100-2150',
    '/kaggle/input/abdominal-trauma-nifti-2230-2360',
    '/kaggle/input/abdominal-trauma-nifti-2300-2400',
    '/kaggle/input/abdominal-trauma-nifti-2400-2500',
    '/kaggle/input/abdominal-trauma-nifti-2500-2600',
    '/kaggle/input/abdominal-trauma-nifti-2600-2700',
    '/kaggle/input/abdominal-trauma-nifti-2700-2800',
    '/kaggle/input/abdominal-trauma-nifti-2800-2900',
    '/kaggle/input/abdominal-trauma-nifti-2900-3000',

    '/kaggle/input/abdominal-trauma-nifti-3000-3100',
    '/kaggle/input/abdominal-trauma-nifti-3100-3200',
    '/kaggle/input/abdominal-trauma-nifti-3200-3300',
    '/kaggle/input/abdominal-trauma-nifti-3300-3400',
    '/kaggle/input/abdominal-trauma-nifti-3400-3500',
    '/kaggle/input/abdominal-trauma-nifti-3500-3600',
    '/kaggle/input/abdominal-trauma-nifti-3600-3700',
    '/kaggle/input/abdominal-trauma-nifti-3700-3800',
    '/kaggle/input/abdominal-trauma-nifti-3800-3900',
    '/kaggle/input/abdominal-trauma-nifti-3900-4000',

    '/kaggle/input/abdominal-trauma-nifti-4000-4100',
    '/kaggle/input/abdominal-trauma-nifti-4100-4200',
    '/kaggle/input/abdominal-trauma-nifti-4200-4300',
    '/kaggle/input/abdominal-nifti-4290-4400',
    '/kaggle/input/abdominal-nifti-4380-4470',
    '/kaggle/input/abdominal-nifti-4470-4560',
    '/kaggle/input/abdominal-nifti-4560-4650',
    '/kaggle/input/abdominal-nifti-4650-4710',  
]

LABELS_CSV_PATH = '/kaggle/input/rsna-2023-abdominal-trauma-detection/train_2024.csv'
OUTPUT_JSON_PATH = '/kaggle/working/train_metadata.json'


# Load labels CSV
labels_df = pd.read_csv(LABELS_CSV_PATH)
labels_df['patient_id'] = labels_df['patient_id'].astype(str)
labels_dict_map = labels_df.set_index('patient_id').to_dict(orient='index')
label_cols = [col for col in labels_df.columns if col != 'patient_id']

metadata_list = []

for dataset_root in DATASET_ROOTS:
    nifti_files = sorted(Path(dataset_root).rglob("*.nii*"))  # .nii or .nii.gz both

    print(f"ğŸ”� Found {len(nifti_files)} NIfTI files in {dataset_root}")

    for nii_path in nifti_files:
        stem = nii_path.stem  # e.g. "12345_67890"
        try:
            patient_id, study_id = stem.split("_")
        except ValueError:
            print(f"âš ï¸� Skipping malformed filename: {stem}")
            continue

        if patient_id not in labels_dict_map:
            print(f"âš ï¸� No label for patient {patient_id}, skipping...")
            continue

        labels = {col: int(labels_dict_map[patient_id][col]) for col in label_cols}

        metadata_list.append({
            "patient_id": patient_id,
            "study_id": study_id,
            "nifti_path": str(nii_path),
            "labels": labels
        })

print(f"âœ… Total metadata entries: {len(metadata_list)}")

# Save JSON
with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(metadata_list, f, indent=2)

print(f"ğŸ“� Metadata saved to {OUTPUT_JSON_PATH}")


import json
from collections import Counter

# Load JSON file (assuming it's a list of entries)
with open("/kaggle/working/train_metadata.json", "r") as f:
    data = json.load(f)

# Collect (patient_id, study_id) pairs
id_pairs = [(entry["patient_id"], entry["study_id"]) for entry in data]

# Count how many times each pair appears
pair_counts = Counter(id_pairs)

# Find duplicates
duplicates = [pair for pair, count in pair_counts.items() if count > 1]

# Print results
if duplicates:
    print(f"Found {len(duplicates)} duplicate entries:")
    for pair in duplicates:
        print(f" - patient_id: {pair[0]}, study_id: {pair[1]}")
else:
    print("âœ… No duplicate (patient_id, study_id) entries found.")


import pandas as pd
import matplotlib.pyplot as plt

# --- Extract and flatten labels into a DataFrame ---
label_rows = []
for entry in metadata_list:
    row = entry["labels"]
    label_rows.append(row)

labels_df = pd.DataFrame(label_rows)

# --- Aggregate counts per label ---
labels_agg = pd.DataFrame({
    'bowel_healthy': [labels_df['bowel_healthy'].sum()],
    'bowel_injury': [labels_df['bowel_injury'].sum()],
    'extravasation_healthy': [labels_df['extravasation_healthy'].sum()],
    'extravasation_injury': [labels_df['extravasation_injury'].sum()],
    
    'kidney_healthy': [labels_df['kidney_healthy'].sum()],
    'kidney_low': [labels_df['kidney_low'].sum()],
    'kidney_high': [labels_df['kidney_high'].sum()],
    
    'liver_healthy': [labels_df['liver_healthy'].sum()],
    'liver_low': [labels_df['liver_low'].sum()],
    'liver_high': [labels_df['liver_high'].sum()],
    
    'spleen_healthy': [labels_df['spleen_healthy'].sum()],
    'spleen_low': [labels_df['spleen_low'].sum()],
    'spleen_high': [labels_df['spleen_high'].sum()]
})



# --- Prepare for plotting ---
labels_agg = labels_agg.T.reset_index()
labels_agg.columns = ['label', 'count']
labels_agg[['organ', 'status']] = labels_agg['label'].str.rsplit('_', n=1, expand=True)

# --- Print counts in console ---
print("Counts per class:")
print(labels_agg[['label', 'count']].to_string(index=False))


# Pivot table: index=organ, columns=status
pivot_df = labels_agg.pivot(index='organ', columns='status', values='count').fillna(0)

# --- Ensure consistent column order ---
status_order = ['healthy', 'injury', 'low',  'high']  # Include all possible status types
for status in status_order:
    if status not in pivot_df.columns:
        pivot_df[status] = 0  # add missing columns with 0
pivot_df = pivot_df[status_order]  # reorder columns

# --- Plot ---
color_map = {
    'healthy': 'skyblue',
    'low': 'orange',
    'high': 'salmon',
    'injury': 'red'
}

colors = [color_map[status] for status in pivot_df.columns]

pivot_df.plot(kind='bar', figsize=(10, 6), color=colors)
plt.title("Organ Injury Severity Distribution")
plt.ylabel("Number of Samples")
plt.xlabel("Organ")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Status')
plt.tight_layout()
plt.show()



# # Create a new column 'injury_status'
# labels_df['injury_status'] = labels_df['any_injury'].apply(lambda x: 'injured' if x == 1 else 'healthy')

# # Count samples by injury_status
# status_counts = labels_df['injury_status'].value_counts()

# print("Counts by injury status:")
# print(status_counts)

# # Plot pie chart
# status_counts.plot(
#     kind='pie',
#     colors=['skyblue', 'salmon'],
#     autopct='%1.1f%%',
#     startangle=90,
#     ylabel='',  # Hide ylabel for cleaner plot
#     title='Proportion of Healthy vs Injured Samples'
# )
# plt.tight_layout()
# plt.show()


# import seaborn as sns

# # Select injury columns only
# injury_cols = ['bowel_injury', 'extravasation_injury', 'kidney_injury',
#                'liver_injury', 'spleen_injury']

# injury_only = labels_df[injury_cols].copy()
# injury_only['kidney_injury'] = injury_only['kidney_low'] + injury_only['kidney_high']
# injury_only['liver_injury'] = injury_only['liver_low'] + injury_only['liver_high']
# injury_only['spleen_injury'] = injury_only['spleen_low'] + injury_only['spleen_high']

# # Keep only binary (0/1)
# injury_matrix = injury_only[['bowel_injury', 'extravasation_injury', 
#                              'kidney_injury', 'liver_injury', 'spleen_injury']]

# # Compute correlation/co-occurrence matrix
# co_occurrence = injury_matrix.T @ injury_matrix
# sns.heatmap(co_occurrence, annot=True, fmt='d', cmap="Reds")
# plt.title("Injury Co-occurrence Heatmap")
# plt.tight_layout()
# plt.show()


# for organ in ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']:
#     if organ == 'kidney' or organ == 'liver' or organ == 'spleen':
#         injury = labels_df[f'{organ}_low'] + labels_df[f'{organ}_high']
#     else:
#         injury = labels_df[f'{organ}_injury']
#     healthy = labels_df[f'{organ}_healthy']
#     ratio = injury.sum() / (injury.sum() + healthy.sum())
#     print(f"{organ.capitalize()} Injury Ratio: {ratio:.2%}")


# # Count how many injury labels each sample has
# injury_per_sample = injury_matrix.sum(axis=1)

# # Plot histogram
# plt.figure(figsize=(6, 4))
# injury_per_sample.hist(bins=range(0, 7), color='teal', rwidth=0.8)
# plt.xlabel("Number of Injured Organs")
# plt.ylabel("Number of Samples")
# plt.title("Histogram of Injury Counts per Sample")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


# # Optional: Visualize first 10 patients' labels as a heatmap
# subset = injury_matrix.iloc[:10]  # First 10 rows
# sns.heatmap(subset, annot=True, cbar=False, cmap="YlGnBu")
# plt.title("First 10 Patients - Injury Pattern Heatmap")
# plt.xlabel("Organ")
# plt.ylabel("Patient Index")
# plt.tight_layout()
# plt.show()


# for organ in ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']:
#         total = labels_df[f'{organ}_injury'] + labels_df[f'{organ}_healthy']
    
#     if not all(total == 1):
#         print(f"Inconsistency found in {organ} labels")


### Import MONAI Transform Classes

# MONAI is a deep learning framework for medical imaging.
# These are various image transformation tools used for preprocessing and augmentation.

from monai.transforms import (
    Compose, EnsureChannelFirst, EnsureType,
    Orientation, Spacing, RandAffine, RandFlip,
    NormalizeIntensity, RandScaleIntensity, RandShiftIntensity,
    RandGaussianNoise, RandGaussianSmooth, RandAdjustContrast,
    Resize, RandBiasField,
    ToTensor
)

from monai.data import MetaTensor
from monai.transforms import OneOf


### Configuration Class
# Stores all important constants and settings in one place (like a dictionary).
class Config:
    SEED = 42
    IMAGE_SIZE = (128, 128, 128)  
    BATCH_SIZE = 8
    EPOCHS = 20
    LR =1e-4
    
    # Target columns (i.e., all the labels you want to predict)
    TARGET_COLS = [
        "bowel_healthy", "extravasation_healthy",
        "bowel_injury", "extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
    ]

    NUM_CLASSES = len(TARGET_COLS)  # Assumes LABELS is a predefined list of column names

    VOXEL_SPACING = (1.0, 1.0, 1.0)  # Used to normalize spacing in 3D CT scans

    SPLIT_MODE = "group"  # Use 'group' split for stratified grouping, or 'random' for simple random split

# Create an instance of the Config class to use in other parts of the code
config = Config()


train_transforms = Compose([
    EnsureChannelFirst(),
    EnsureType(),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    Resize(spatial_size=config.IMAGE_SIZE),

    # Original transforms you had:
    OneOf([
        RandAffine(
            rotate_range=(0, 0, np.pi/12),  # Limited Z-axis rotation only
            shear_range=(0.1, 0.1, 0.1),
            translate_range=(10, 10, 5),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.5,
            mode="bilinear"
        ),
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
    ]),
    NormalizeIntensity(nonzero=True, channel_wise=True),
    RandScaleIntensity(factors=0.1, prob=1.0),
    RandShiftIntensity(offsets=0.1, prob=1.0),
    RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),
    RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5)),

    # New safe additions:
    RandGaussianSmooth(
        prob=0.2, 
        sigma_x=(0.25, 0.5),  # Very mild smoothing
        sigma_y=(0.25, 0.5),
        sigma_z=(0.25, 0.5)
    ),
    RandBiasField(
        prob=0.2, 
        coeff_range=(0.1, 0.3)  # Subtle intensity variations
    ),

    ToTensor()
])


### Validation Transforms

# These are simpler than training transforms and only do standard preprocessing.
# No random changes here, just make sure data is consistent and normalized.

val_transforms = Compose([
    EnsureChannelFirst(),  # (Z, H, W) â†’ (1, Z, H, W)
    EnsureType(),  # Convert to MetaTensor
    Orientation(axcodes="RAS"),  # Set standard orientation
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),  # Make voxel spacing uniform
    Resize(spatial_size=config.IMAGE_SIZE),
    NormalizeIntensity(nonzero=True, channel_wise=True),  # Normalize image intensities
    ToTensor()  # Convert to tensor
])


### Test Transforms
# Same as validation transformsâ€”no randomness. Used during testing and inference.

test_transforms = Compose([
    EnsureChannelFirst(),
    EnsureType(),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    Resize(spatial_size=config.IMAGE_SIZE),
    NormalizeIntensity(nonzero=True, channel_wise=True),
    ToTensor()
])


import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from monai.data import MetaTensor  

class RSNADataset(Dataset):
    def __init__(self, metadata_list, transforms=None, has_labels=True):
        """
        Args:
            metadata_list: List of dictionaries with 'nifti_path' and 'labels'.
            transforms: MONAI or array-style transforms (non-dict style).
            has_labels: Whether to return labels (True during training/val).
        """
        self.metadata_list = metadata_list
        self.transforms = transforms
        self.has_labels = has_labels

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        entry = self.metadata_list[idx]

        # --- Load the NIfTI file ---
        nifti_path = entry["nifti_path"]  # Full path to .nii.gz
        nifti_img = nib.load(nifti_path)
        volume = nifti_img.get_fdata().astype(np.float32)

        # --- Rearrange dimensions (X, Y, Z) â†’ (Z, Y, X) ---
        volume = np.transpose(volume, (2, 1, 0))

        # --- Add channel dimension: (1, Z, Y, X) ---
        volume = np.expand_dims(volume, axis=0)

        # --- Wrap in MetaTensor (optional, for MONAI compatibility) ---
        meta = {"original_channel_dim": 0}
        sample = MetaTensor(volume, meta=meta)

        # --- Apply transforms if provided ---
        if self.transforms:
            sample = self.transforms(sample)

        # --- Package label if available ---
        if self.has_labels:
            sample = {
                "image": sample,
                "label": np.array([entry["labels"][key] for key in config.TARGET_COLS], dtype=np.float32),
            }
        else:
            sample = {"image": sample}

        return sample



### Prepare Dataloaders for Training, Validation, and Testing
# This function handles data splitting and creates PyTorch DataLoader objects.

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def prepare_balanced_dataloaders(metadata_list, train_transforms, val_transforms, test_transforms, config):
    def filter_by_condition(label_key, value=1):
        return [m for m in metadata_list if m["labels"].get(label_key, 0) == value]

    # Use a dict to avoid duplicates (keyed by unique nifti_path)
    balanced_dict = {}

    def add_to_balanced(samples):
        for sample in samples:
            balanced_dict[sample["nifti_path"]] = sample

    # === Bowel (binary)
    bowel_injury = filter_by_condition("bowel_injury")
    bowel_healthy = np.random.choice(filter_by_condition("bowel_healthy"), size=len(bowel_injury), replace=False)
    add_to_balanced(bowel_injury)
    add_to_balanced(bowel_healthy)

    # === Extravasation (binary)
    extrav_injury = filter_by_condition("extravasation_injury")
    extrav_healthy = np.random.choice(filter_by_condition("extravasation_healthy"), size=len(extrav_injury), replace=False)
    add_to_balanced(extrav_injury)
    add_to_balanced(extrav_healthy)

    # === Kidney (3-class)
    kidney_low = filter_by_condition("kidney_low")
    kidney_high = filter_by_condition("kidney_high")
    kidney_healthy = np.random.choice(filter_by_condition("kidney_healthy"), size=len(kidney_low)+len(kidney_high), replace=False)
    add_to_balanced(kidney_low)
    add_to_balanced(kidney_high)
    add_to_balanced(kidney_healthy)

    # === Liver (3-class)
    liver_low = filter_by_condition("liver_low")
    liver_high = filter_by_condition("liver_high")
    liver_healthy = np.random.choice(filter_by_condition("liver_healthy"), size=len(liver_low)+len(liver_high), replace=False)
    add_to_balanced(liver_low)
    add_to_balanced(liver_high)
    add_to_balanced(liver_healthy)

    # === Spleen (3-class)
    spleen_low = filter_by_condition("spleen_low")
    spleen_high = filter_by_condition("spleen_high")
    spleen_healthy = np.random.choice(filter_by_condition("spleen_healthy"), size=len(spleen_low)+len(spleen_high), replace=False)
    add_to_balanced(spleen_low)
    add_to_balanced(spleen_high)
    add_to_balanced(spleen_healthy)

    # Convert dict values to list of unique samples
    balanced_metadata = list(balanced_dict.values())

    # === Split the Data === #
    if config.SPLIT_MODE == "random":
        train_meta, temp_meta = train_test_split(
            balanced_metadata,
            test_size=0.3,
            random_state=42,
            stratify=[str([m["labels"][k] for k in config.TARGET_COLS]) for m in balanced_metadata]
        )
        val_meta, test_meta = train_test_split(
            temp_meta,
            test_size=0.5,
            random_state=42,
            stratify=[str([m["labels"][k] for k in config.TARGET_COLS]) for m in temp_meta]
        )
    elif config.SPLIT_MODE == "group":
        train_meta, val_meta, test_meta = split_metadata_train_val_test(
            balanced_metadata,
            target_cols=config.TARGET_COLS,
            val_size=0.15,
            test_size=0.15,
            seed=42
        )

    # === Create Dataset and DataLoaders === #
    train_ds = RSNADataset(train_meta, transforms=train_transforms)
    val_ds = RSNADataset(val_meta, transforms=val_transforms)
    test_ds = RSNADataset(test_meta, transforms=test_transforms)

    return (
        train_ds,
        val_ds,
        test_ds,
        DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False),
        DataLoader(test_ds, batch_size=1, shuffle=False)
    )



def split_metadata_train_val_test(metadata_list, target_cols, val_size=0.1, test_size=0.1, seed=42):
    """
    Custom stratified split to maintain the label distribution for train, validation, and test sets.
    """
    # Convert metadata into a DataFrame
    df = pd.DataFrame(metadata_list)

    # Extract individual labels into separate columns
    label_df = pd.json_normalize(df['labels'])
    df = pd.concat([df.drop(columns='labels'), label_df], axis=1)

    # Group rows by all target label combinations
    grouped = df.groupby(target_cols)

    # Empty splits
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    val_test_size = val_size + test_size

    for _, group in grouped:
        n = len(group)
        if n == 1:
            r = np.random.rand()
            if r < test_size:
                test_df = pd.concat([test_df, group], ignore_index=True)
            elif r < val_test_size:
                val_df = pd.concat([val_df, group], ignore_index=True)
            else:
                train_df = pd.concat([train_df, group], ignore_index=True)
        else:
            train_split, val_test_split = train_test_split(group, test_size=val_test_size, random_state=seed)

            if len(val_test_split) < 2:
                val_split = val_test_split
                test_split = pd.DataFrame()
            else:
                relative_test_size = test_size / val_test_size if val_test_size > 0 else 0
                val_split, test_split = train_test_split(val_test_split, test_size=relative_test_size, random_state=seed)

            train_df = pd.concat([train_df, train_split], ignore_index=True)
            val_df = pd.concat([val_df, val_split], ignore_index=True)
            test_df = pd.concat([test_df, test_split], ignore_index=True)

    # Convert DataFrame rows back to metadata format
    def row_to_metadata(row):
        return {
            "nifti_path": row["nifti_path"],
            "labels": {col: row[col] for col in target_cols}
        }

    train_list = [row_to_metadata(row) for _, row in train_df.iterrows()]
    val_list   = [row_to_metadata(row) for _, row in val_df.iterrows()]
    test_list  = [row_to_metadata(row) for _, row in test_df.iterrows()]

    return train_list, val_list, test_list



### Prepare Data Using the Loader Function (with subset for tuning)
# It splits the metadata into train/val/test and creates DataLoader objects for batching and shuffling.

train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = prepare_balanced_dataloaders(
    metadata_list=metadata_list,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=test_transforms,
    config=config
)

print(f"Train size: {len(train_loader.dataset)}")
print(f"Val size:   {len(val_loader.dataset)}")
print(f"Test size:  {len(test_loader.dataset)}")    


# Reduce training data to 10 samples to test pipeline
# overfit_subset = torch.utils.data.Subset(train_ds, range(10))
# train_loader = DataLoader(overfit_subset, batch_size=2, shuffle=True)

# val_subset = torch.utils.data.Subset(val_ds, range(5))
# val_loader = DataLoader(val_subset, batch_size=2, shuffle=False)


import pandas as pd
import torch

def print_class_distribution_and_weights(metadata_list, target_cols):
    # Convert labels into a DataFrame
    labels_df = pd.DataFrame([entry["labels"] for entry in metadata_list])

    print("ğŸ“Š Class distribution:")

    for col in target_cols:
        counts = labels_df[col].value_counts().sort_index()
        zeros = counts.get(0, 0)
        ones = counts.get(1, 0)
        print(f"\nğŸ”¸ {col}")
        print(f"   Label 0: {zeros} samples")
        print(f"   Label 1: {ones} samples")

        # Calculate weight for positive class (inverse freq)
        total = zeros + ones
        pos_weight = (zeros / (ones + 1e-6)) if ones > 0 else 0.0
        print(f"   â†’ Positive class weight: {pos_weight:.2f}")

    # For multi-class groups (kidney, liver, spleen), calculate weights per class label
    for organ in ["kidney", "liver", "spleen"]:
        organ_cols = [f"{organ}_healthy", f"{organ}_low", f"{organ}_high"]
        print(f"\nğŸ”¸ {organ} class weights (inverse frequency):")
        means = labels_df[organ_cols].mean(axis=0)
        weights = 1.0 / (means + 1e-6)
        norm_weights = weights / weights.sum()
        for cls, w, m in zip(organ_cols, norm_weights, means):
            print(f"  {cls}: mean={m:.4f}, weight={w:.4f}")

    return labels_df

# Usage example:
target_cols = config.TARGET_COLS  # Your full target columns list

labels_df = print_class_distribution_and_weights(train_ds.metadata_list, target_cols)


import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121

class DenseNet121model(nn.Module):
    def __init__(self, in_channels=1, pretrained=False):
        super().__init__()
        
        # Grad-CAM hooks
        self.activations = None
        self.gradients = None
        
        # Backbone - using MONAI's DenseNet121 which already includes GAP
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=512,  # This is the output after GAP
            pretrained=pretrained
        )
        
        # Register hook for Grad-CAM on the last conv layer
        self.backbone.features[-1].register_forward_hook(self.save_activation)
        
        # Classification heads
        self.bowel_head = self._create_binary_head()
        self.extra_head = self._create_binary_head()
        self.liver_head = self._create_multiclass_head()
        self.kidney_head = self._create_multiclass_head()
        self.spleen_head = self._create_multiclass_head()
        
    def _create_binary_head(self):
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def _create_multiclass_head(self):
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    
    def save_activation(self, module, input, output):
        """Save activations for Grad-CAM"""
        self.activations = output
        if output.requires_grad:
            output.register_hook(self.save_gradient)
    
    def save_gradient(self, grad):
        """Save gradients for Grad-CAM"""
        self.gradients = grad
    
    def forward(self, x):
        # Forward pass through backbone
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        self.activations = x
        
        # Get features (shape: [B, 512] after GAP)
        features = self.backbone(x)
        
        return {
            "bowel": self.bowel_head(features),
            "extra": self.extra_head(features),
            "liver": self.liver_head(features),
            "kidney": self.kidney_head(features),
            "spleen": self.spleen_head(features)
        }
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations


### Initialize Model and Move to Device (GPU or CPU)
model = DenseNet121model().to(device)
print(f"Model moved to {next(model.parameters()).device}")


import torch.nn.functional as F
from scipy.ndimage import zoom

def compute_gradcam(model, input_tensor, target_head="bowel", class_index=None, target_shape=None):
    model.eval()
    model.zero_grad()

    # Forward pass
    output = model(input_tensor)

    # Select target class output
    if class_index is None:
        class_index = 0
    loss = output[target_head][0, class_index]
    loss.backward()

    # Grab activations and gradients from the model
    activations = model.activations  # Shape: (B, C, D, H, W)
    grads = model.gradients          # Same shape

    # Global average pooling of gradients over spatial dims
    pooled_grads = torch.mean(grads, dim=(2, 3, 4), keepdim=True)  # Shape: (B, C, 1, 1, 1)

    # Weighted sum of activations
    weighted_activations = activations * pooled_grads  # Shape: (B, C, D, H, W)
    cam = weighted_activations.sum(dim=1).squeeze()    # Shape: (D, H, W)

    # ReLU and normalize
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-5)

    # Convert to NumPy
    cam_np = cam.detach().cpu().numpy()  # Shape: (D, H, W)

    # Resize to original volume shape if provided
    if target_shape:
        if len(target_shape) != 3:
            target_shape = target_shape[-3:]
        zoom_factors = [t / c for t, c in zip(target_shape, cam_np.shape)]
        cam_np = zoom(cam_np, zoom=zoom_factors, order=1)  # Linear interpolation

    return cam_np



import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction="mean", label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Numerically stable BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss  



class MultiClassFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(metadata_list):
    labels_df = pd.DataFrame([entry["labels"] for entry in metadata_list])

    # Alpha for focal loss = pos / (pos + neg)
    alpha_dict = {
        "bowel": float(labels_df["bowel_injury"].sum() / (labels_df["bowel_injury"].sum() + labels_df["bowel_healthy"].sum())),
        "extra": float(labels_df["extravasation_injury"].sum() / (labels_df["extravasation_injury"].sum() + labels_df["extravasation_healthy"].sum()))
    }

    ce_weights = {}
    for organ in ["kidney", "liver", "spleen"]:
        counts = torch.tensor([
            labels_df[f"{organ}_healthy"].sum(),
            labels_df[f"{organ}_low"].sum(),
            labels_df[f"{organ}_high"].sum()
        ], dtype=torch.float)
        weights = 1.0 / (counts + 1e-6)
        weights /= weights.sum()
        ce_weights[organ] = weights

    return alpha_dict, ce_weights



### Define Loss Functions for Each Output Head
# - BCEWithLogitsLoss is used for binary classification (outputs NOT passed through sigmoid yet).
# - CrossEntropyLoss is used for multi-class classification (outputs NOT passed through softmax yet).
bce_weights, ce_weights = compute_class_weights(train_ds.metadata_list)

loss_fn_dict = {
    "bowel": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_weights["bowel"]).to(device)),
    "extra": nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_weights["extra"]).to(device)),
    "liver": nn.CrossEntropyLoss(weight=ce_weights["liver"].to(device)),
    "kidney": nn.CrossEntropyLoss(weight=ce_weights["kidney"].to(device)),
    "spleen": nn.CrossEntropyLoss(weight=ce_weights["spleen"].to(device)),
}



# Replace current optimizer setup with:
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
)


from tqdm import tqdm
from collections import defaultdict
import torch
import numpy as np

def train_one_epoch(model, loader, optimizer, loss_fn_dict, scheduler=None, grad_clip=None, debug=False):
    model.train()
    running_loss = 0.0
    task_losses = defaultdict(float)
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)

    for batch_idx, batch in pbar:
        inputs = batch["image"].to(device, dtype=torch.float32)
        labels = batch["label"].to(device, dtype=torch.float32)

        if inputs.ndim == 4:
            inputs = inputs.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)

        targets = {
            "bowel": labels[:, 0:2].max(dim=1)[0].float(),
            "extra": labels[:, 2:4].max(dim=1)[0].float(),
            "kidney": labels[:, 4:7].argmax(dim=1),
            "liver": labels[:, 7:10].argmax(dim=1),
            "spleen": labels[:, 10:13].argmax(dim=1),
        }

        loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for key in outputs:
            pred = outputs[key]
            target = targets[key]

            try:
                if key in ["bowel", "extra"]:
                    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
                    target = target.float()
                    task_loss = loss_fn_dict[key](pred, target)
                else:
                    target = target.long()
                    task_loss = loss_fn_dict[key](pred, target)
            except Exception as e:
                print(f"â�Œ Error computing loss for {key} at batch {batch_idx}: {e}")
                continue

            if not torch.is_tensor(task_loss):
                task_loss = torch.tensor(task_loss, device=device, dtype=torch.float32)

            if torch.isnan(task_loss).any():
                print(f"â�Œ NaN loss encountered in {key} at batch {batch_idx}. Skipping this task.")
                continue

            task_losses[key] += task_loss.item()
            loss += task_loss

        if torch.isnan(loss).any():
            print(f"â�Œ NaN total loss at batch {batch_idx}. Skipping update.")
            continue

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)

        postfix = {"loss": avg_loss}
        for key in task_losses:
            postfix[f"{key}_loss"] = task_losses[key] / (batch_idx + 1)

        pbar.set_postfix(postfix)

        if debug and batch_idx % 20 == 0:
            print(f"\n [Batch {batch_idx}]")
            print(f"  Bowel logits: {outputs['bowel'].detach().cpu().numpy()[:3].squeeze()}")
            print(f"  Bowel targets: {targets['bowel'].cpu().numpy()[:3]}")
            print(f"  Kidney pred: {outputs['kidney'].argmax(dim=1).cpu().numpy()[:3]}")
            print(f"  Kidney true: {targets['kidney'].cpu().numpy()[:3]}")
            print(f"  Total loss: {loss.item():.4f}")

    task_losses = {k: v / len(loader) for k, v in task_losses.items()}
    return avg_loss, task_losses



@torch.no_grad()
def validate(model, loader, loss_fn_dict):
    model.eval()
    val_loss = 0.0
    pbar = tqdm(loader, desc="Validation", leave=False)

    all_preds = defaultdict(list)
    all_targets = defaultdict(list)

    with torch.no_grad():
        for batch in pbar:
            inputs = batch["image"].to(device, dtype=torch.float32)
            labels = batch["label"].to(device, dtype=torch.float32)

            if inputs.ndim == 4:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)

            targets = {
                "bowel": labels[:, 0:2].max(dim=1)[0].float(),
                "extra": labels[:, 2:4].max(dim=1)[0].float(),
                "kidney": labels[:, 4:7].argmax(dim=1),
                "liver": labels[:, 7:10].argmax(dim=1),
                "spleen": labels[:, 10:13].argmax(dim=1),
            }

            loss = torch.tensor(0.0, device=device, dtype=torch.float32)

            for key in outputs:
                pred = outputs[key]
                target = targets[key]

                if pred.shape[-1] == 1:
                    prob = torch.sigmoid(pred).view(-1).cpu().numpy()
                    bin_pred = (prob >= 0.5).astype(int)
                    target_np = target.cpu().numpy().astype(int)

                    all_preds[key].extend(bin_pred)
                    all_targets[key].extend(target_np)
                    loss += loss_fn_dict[key](pred.view(-1), target.float().view(-1))

                else:
                    softmax_pred = torch.softmax(pred, dim=1)
                    class_pred = torch.argmax(softmax_pred, dim=1).cpu().numpy()
                    target_np = target.cpu().numpy()

                    all_preds[key].extend(class_pred)
                    all_targets[key].extend(target_np)
                    loss += loss_fn_dict[key](pred, target)

            val_loss += loss.item()
            pbar.set_postfix({"val_loss": val_loss / (pbar.n + 1)})

    metrics = {}
    print("\n--- Evaluation Metrics ---")
    for key in all_preds:
        y_true = np.array(all_targets[key])
        y_pred = np.array(all_preds[key])

        metrics[key] = {}

        if len(np.unique(y_true)) <= 1:
            print(f"{key}: Not enough class diversity in ground truth to compute metrics.")
            continue

        if set(np.unique(y_true)) <= {0, 1}:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            acc = accuracy_score(y_true, y_pred)
            try:
                roc = roc_auc_score(y_true, y_pred)
            except ValueError:
                roc = np.nan
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
            acc = accuracy_score(y_true, y_pred)
            try:
                NUM_CLASSES = {
                    "kidney": 3,
                    "liver": 3,
                    "spleen": 3,
                }
                roc = roc_auc_score(
                    y_true,
                    torch.nn.functional.one_hot(torch.tensor(y_pred), num_classes=NUM_CLASSES[key]),
                    multi_class='ovo')
            except ValueError:
                roc = np.nan

        metrics[key]['precision'] = precision
        metrics[key]['recall'] = recall
        metrics[key]['f1'] = f1
        metrics[key]['accuracy'] = acc
        metrics[key]['roc_auc'] = roc

        print(f"{key.capitalize()} | Acc: {acc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | ROC-AUC: {roc:.3f}")

    return val_loss / len(loader), metrics



import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import zipfile
import os
import torch
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Authenticate once globally
api = KaggleApi()
api.authenticate()

def upload_to_kaggle_model(dataset_owner, dataset_slug, model_path, checkpoint_path=None, version_note=""):
    import zipfile
    import os
    import json

    zip_path = "/kaggle/working/model_upload.zip"
    
    # Zip model(s)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(model_path, arcname=os.path.basename(model_path))
        if checkpoint_path:
            zipf.write(checkpoint_path, arcname=os.path.basename(checkpoint_path))
    
    print(f"Zipped model(s) to: {zip_path}")

    # Create dataset-metadata.json file for Kaggle API
    metadata = {
        "title": f"{dataset_slug} model",
        "id": f"{dataset_owner}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    metadata_path = "/kaggle/working/dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Created metadata file at {metadata_path}")

    # Upload as new version to Kaggle dataset
    api.dataset_create_version(
        folder="/kaggle/working",
        version_notes=version_note,
        delete_old_versions=False,
        convert_to_csv=False
    )
    print(f"Uploaded to Kaggle Dataset: {dataset_owner}/{dataset_slug}")



def extract_checkpoint_from_dataset(dataset_input_path="/kaggle/input/hypertuning-dataset",
                                    extract_path="/kaggle/working"):
    checkpoint_src = os.path.join(dataset_input_path, "checkpoint.pth")
    best_model_src = os.path.join(dataset_input_path, "model_best.pth")
    
    copied = False

    if os.path.exists(checkpoint_src):
        shutil.copy(checkpoint_src, os.path.join(extract_path, "checkpoint.pth"))
        print(f"Copied checkpoint.pth to {extract_path}")
        copied = True
    else:
        print("checkpoint.pth not found in dataset input.")

    if os.path.exists(best_model_src):
        shutil.copy(best_model_src, os.path.join(extract_path, "model_best.pth"))
        print(f"Copied model_best.pth to {extract_path}")
        copied = True
    else:
        print("model_best.pth not found in dataset input.")

    return extract_path if copied else None

def train(model, train_loader, val_loader, optimizer, loss_fn_dict, num_epochs,
          save_dir="/kaggle/working", resume=False,
          upload_to_kaggle=False, dataset_owner=None, dataset_slug=None, hyperparam_note=""):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    start_epoch = 0

    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    best_model_path = os.path.join(save_dir, "model_best.pth")

    history = {
        'train_loss': [],
        'val_loss': [],
        'metrics': {
            'bowel': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
            'extra': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
            'kidney': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
            'liver': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
            'spleen': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
        }
    }
    
    # Try to extract checkpoint if resume and dataset input zip exists
    if resume:
        extracted_path = extract_checkpoint_from_dataset()
        if extracted_path is not None:
            checkpoint_path_from_extract = os.path.join(extracted_path, "checkpoint.pth")
            best_model_path_from_extract = os.path.join(extracted_path, "model_best.pth")
            # Copy extracted checkpoint and best model to save_dir for training continuity
            if os.path.exists(checkpoint_path_from_extract):
                os.replace(checkpoint_path_from_extract, checkpoint_path)
                print(f"Checkpoint copied to {checkpoint_path}")
            if os.path.exists(best_model_path_from_extract):
                os.replace(best_model_path_from_extract, best_model_path)
                print(f"Best model copied to {best_model_path}")

    # Resume from local checkpoint if exists
    # Resume from local checkpoint if exists
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        history = checkpoint.get('history', history)
        print(f"Resumed from epoch {start_epoch}")
    
        # ğŸ”§ Patch: Add 'accuracy' if missing from metrics history
        for organ in history['metrics']:
            if 'accuracy' not in history['metrics'][organ]:
                history['metrics'][organ]['accuracy'] = []

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn_dict)
        val_loss, metrics = validate(model, val_loader, loss_fn_dict)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for organ in metrics:
            for metric in metrics[organ]:
                history['metrics'][organ][metric].append(metrics[organ][metric])

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
        }, checkpoint_path)

        # Save best model + upload if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Val Loss = {val_loss:.4f}")

            if upload_to_kaggle and dataset_owner and dataset_slug:
                note = f"{hyperparam_note} | Epoch {epoch+1}, Val Loss: {val_loss:.4f}"
                upload_to_kaggle_model(dataset_owner, dataset_slug, best_model_path, checkpoint_path, version_note=note)
        else:
            print(f"No improvement. Val Loss = {val_loss:.4f}")

            if upload_to_kaggle and dataset_owner and dataset_slug:
                note = f"{hyperparam_note} | Epoch {epoch+1}, No improvement"
                upload_to_kaggle_model(dataset_owner, dataset_slug, model_path=checkpoint_path, version_note=note)
                
    return model, history


from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

def train(model, train_loader, val_loader, optimizer, loss_fn_dict, num_epochs,
          save_dir="/kaggle/working", resume=False,
          upload_to_kaggle=False, dataset_owner=None, dataset_slug=None, hyperparam_note="", grad_clip=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    start_epoch = 0

    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    best_model_path = os.path.join(save_dir, "model_best.pth")

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'metrics': {
            'bowel': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
            'extra': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
            'kidney': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
            'liver': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
            'spleen': {'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'accuracy': []},
        }
    }
    
    # Try to extract checkpoint if resume and dataset input zip exists
    if resume:
        extracted_path = extract_checkpoint_from_dataset()
        if extracted_path is not None:
            checkpoint_path_from_extract = os.path.join(extracted_path, "checkpoint.pth")
            best_model_path_from_extract = os.path.join(extracted_path, "model_best.pth")
            # Copy extracted checkpoint and best model to save_dir for training continuity
            if os.path.exists(checkpoint_path_from_extract):
                os.replace(checkpoint_path_from_extract, checkpoint_path)
                print(f"Checkpoint copied to {checkpoint_path}")
            if os.path.exists(best_model_path_from_extract):
                os.replace(best_model_path_from_extract, best_model_path)
                print(f"Best model copied to {best_model_path}")

    # Resume from local checkpoint if exists
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        history = checkpoint.get('history', history)
        print(f"Resumed from epoch {start_epoch}")
    
        # Patch: Add 'accuracy' if missing from metrics history
        for organ in history['metrics']:
            if 'accuracy' not in history['metrics'][organ]:
                history['metrics'][organ]['accuracy'] = []

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        train_loss, train_task_losses = train_one_epoch(model, train_loader, optimizer, loss_fn_dict, grad_clip=grad_clip)
        val_loss, metrics = validate(model, val_loader, loss_fn_dict)

        # Step scheduler with validation loss
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for organ in metrics:
            for metric in metrics[organ]:
                history['metrics'][organ][metric].append(metrics[organ][metric])

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
        }, checkpoint_path)

        # Save best model + upload if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Val Loss = {val_loss:.4f}")

            if upload_to_kaggle and dataset_owner and dataset_slug:
                note = f"{hyperparam_note} | Epoch {epoch+1}, Val Loss: {val_loss:.4f}"
                upload_to_kaggle_model(dataset_owner, dataset_slug, best_model_path, checkpoint_path, version_note=note)
        else:
            print(f"No improvement. Val Loss = {val_loss:.4f}")

            if upload_to_kaggle and dataset_owner and dataset_slug:
                note = f"{hyperparam_note} | Epoch {epoch+1}, No improvement"
                upload_to_kaggle_model(dataset_owner, dataset_slug, model_path=checkpoint_path, version_note=note)
                
    return model, history


dataset_owner = "anjanatimsina"
dataset_slug = "hypertuning-dataset"
dataset_id = f"{dataset_owner}/{dataset_slug}"


# # Subset your training and validation loader to only 1 image
# from torch.utils.data import Subset

# train_subset = Subset(train_ds, [0])
# val_subset = Subset(val_ds, [0])

# train_loader = DataLoader(train_subset, batch_size=1)
# val_loader = DataLoader(val_subset, batch_size=1)

# # Train for 1 epoch
# model, history = train(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     loss_fn_dict,
#     num_epochs=1,
#     save_dir="/kaggle/working/",
#     resume=False,
#     upload_to_kaggle=True,
#     dataset_owner=dataset_owner,
#     dataset_slug=dataset_slug,
#     hyperparam_note="ğŸ§ª Test run with 1 sample"
# )


### Full Training Loop Over All Epochs
NUM_EPOCHS = config.EPOCHS
save_dir = '/kaggle/working/'

hyperparam_note=f"lr={config.LR}, bs={config.BATCH_SIZE}, image_size={config.IMAGE_SIZE}, focal_loss: fine tuned subset = 30% "

print("Is CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

model, history = train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn_dict,
    NUM_EPOCHS,
    save_dir,
    resume=True,
    upload_to_kaggle=True,
    dataset_owner=dataset_owner,
    dataset_slug=dataset_slug,
    hyperparam_note=hyperparam_note
)


# from torch.utils.data import Subset, DataLoader

# # Use only first 5 samples for training and validation
# small_train_ds = Subset(train_ds, range(5))
# small_val_ds = Subset(val_ds, range(5))

# # Create smaller loaders
# train_loader = DataLoader(small_train_ds, batch_size=1, shuffle=True)
# val_loader = DataLoader(small_val_ds, batch_size=1, shuffle=False)

# # Train only 1 epoch on small data
# NUM_EPOCHS = 1
# model, history = train(model, train_loader, val_loader, optimizer, loss_fn_dict, NUM_EPOCHS)


def load_model(path="model_best.pth"):
    model = DenseNet121model()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval().to(device)
    return model

load_model('checkpoints/model_best.pth')


def plot_training_history(history):
    # Create a figure with appropriate size
    plt.figure(figsize=(20, 15))
    
    # Plot losses
    plt.subplot(3, 2, 1)  # 3 rows, 2 columns, position 1
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics for each task
    tasks = list(history['metrics'].keys())
    metrics = ['precision', 'recall', 'f1', 'roc_auc']
    
    # Plot each metric in its own subplot
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+2)  # Positions 2-5
        for task in tasks:
            if history['metrics'][task][metric]:  # Check if metric exists
                plt.plot(history['metrics'][task][metric], label=f'{task}')
        plt.title(f'{metric.capitalize()} per epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrices(model, loader, loss_fn_dict):
    model.eval()
    all_preds = defaultdict(list)
    all_targets = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating predictions for confusion matrices"):
            inputs = batch["image"].to(device, dtype=torch.float32)
            labels = batch["label"].to(device, dtype=torch.float32)

            if inputs.ndim == 4:
                inputs = inputs.unsqueeze(2)

            outputs = model(inputs)

            targets = {
                "bowel": labels[:, 0:2].max(dim=1)[0].float(),
                "extra": labels[:, 2:4].max(dim=1)[0].float(),
                "kidney": labels[:, 4:7].argmax(dim=1),
                "liver": labels[:, 7:10].argmax(dim=1),
                "spleen": labels[:, 10:13].argmax(dim=1),
            }

            for key in outputs:
                pred = outputs[key]
                target = targets[key]

                if pred.shape[-1] == 1:  # Binary case
                    prob = torch.sigmoid(pred).view(-1).cpu().numpy()
                    bin_pred = (prob >= 0.5).astype(int)
                    target_np = target.cpu().numpy().astype(int)
                else:  # Multi-class
                    softmax_pred = torch.softmax(pred, dim=1)
                    class_pred = torch.argmax(softmax_pred, dim=1).cpu().numpy()
                    target_np = target.cpu().numpy()

                all_preds[key].extend(class_pred if pred.shape[-1] != 1 else bin_pred)
                all_targets[key].extend(target_np)
    
    # Plot confusion matrices
    plt.figure(figsize=(20, 15))
    for i, key in enumerate(all_preds, 1):
        plt.subplot(2, 3, i)
        cm = confusion_matrix(all_targets[key], all_preds[key])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(all_targets[key]), 
                   yticklabels=np.unique(all_targets[key]))
        plt.title(f'Confusion Matrix - {key.capitalize()}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.show()


# Plot training history
plot_training_history(history)

# Plot confusion matrices on validation set
plot_confusion_matrices(model, val_loader, loss_fn_dict)


@torch.no_grad()
def test_inference(model, loader, device="cuda", max_batches=None):
    model.eval()
    predictions = []
    indices = []

    pbar = tqdm(enumerate(loader), total=max_batches or len(loader), desc="Inference", leave=False)

    for i, batch in pbar:
        if max_batches is not None and i >= max_batches:
            break

        inputs = batch["image"].to(device, dtype=torch.float32)

        if inputs.ndim == 4:
            inputs = inputs.unsqueeze(2)  # Ensure 5D

        outputs = model(inputs)

        # Apply activations to get probabilities
        batch_preds = {
            "bowel": torch.sigmoid(outputs["bowel"]).cpu().numpy(),
            "extra": torch.sigmoid(outputs["extra"]).cpu().numpy(),
            "kidney": F.softmax(outputs["kidney"], dim=1).cpu().numpy(),
            "liver": F.softmax(outputs["liver"], dim=1).cpu().numpy(),
            "spleen": F.softmax(outputs["spleen"], dim=1).cpu().numpy(),
        }

        predictions.append(batch_preds)
        indices.append(i)  # You can append batch-level index or patient ID if available

    return predictions, indices


from torch.utils.data import Subset, DataLoader

# Create small subset of test dataset
small_test_ds = Subset(test_ds, list(range(10)))  # First 10 samples
small_test_loader =  DataLoader(small_test_ds, batch_size=1, shuffle=False)

# Run inference on this subset
preds, ids = test_inference(model, small_test_loader, device=device, max_batches=10)

test_sample = next(iter(small_test_loader))  # Get one sample from the test DataLoader
input_tensor = test_sample["image"] 

# Example output inspection
print("Predictions for first sample:")
print(preds[0])


for i, pred in enumerate(preds[:3]):
    print(f"\nSample {i}:")
    print("Bowel Injury Probability:", pred["bowel"].squeeze())
    print("Extravasation Probability:", pred["extra"].squeeze())
    print("Kidney Class Probabilities:", pred["kidney"].squeeze())
    print("Liver Class Probabilities:", pred["liver"].squeeze())
    print("Spleen Class Probabilities:", pred["spleen"].squeeze())


def interpret_predictions(preds, class_labels=None):
    if class_labels is None:
        class_labels = ["healthy", "low", "high"]  # for multiclass labels

    for i, pred in enumerate(preds):
        print(f"\nSample {i} Prediction:")

        # Binary: Injury Present or Not
        bowel_status = "Injured" if pred["bowel"].squeeze() > 0.5 else "Healthy"
        extra_status = "Extravasation" if pred["extra"].squeeze() > 0.5 else "No Extravasation"
        print(f"  â–¸ Bowel Injury: {bowel_status} ")
        print(f"  â–¸ Extravasation: {extra_status} ")

        # Multiclass: argmax for label
        kidney_label = class_labels[np.argmax(pred["kidney"])]
        liver_label = class_labels[np.argmax(pred["liver"])]
        spleen_label = class_labels[np.argmax(pred["spleen"])]

        print(f"  â–¸ Kidney Condition: {kidney_label}")
        print(f"  â–¸ Liver Condition: {liver_label}")
        print(f"  â–¸ Spleen Condition: {spleen_label}")

interpret_predictions(preds[:3])


from skimage import measure

def show_gradcam_overlay(cam, original_volume, slice_axis=0, alpha=0.5, threshold=0.5, show_contours=True, slice_range=5):
    """
    Show Grad-CAM overlay on multiple slices of the CT volume.
    
    Parameters:
    - cam: 3D Grad-CAM heatmap (D, H, W)
    - original_volume: 3D CT scan (D, H, W)
    - slice_axis: 0, 1, or 2 â†’ axis to slice along
    - alpha: blending factor for overlay
    - threshold: value (0â€“1) for contour visualization
    - show_contours: whether to show contour on top of heatmap
    - slice_range: number of slices to show before and after mid-slice
    """
    assert cam.shape == original_volume.shape, "CAM and CT shape mismatch"

    mid_slice = original_volume.shape[slice_axis] // 2
    slice_indices = range(mid_slice - slice_range, mid_slice + slice_range + 1)

    n_cols = 5
    n_rows = int(np.ceil(len(slice_indices) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    for i, idx in enumerate(slice_indices):
        if idx < 0 or idx >= original_volume.shape[slice_axis]:
            continue

        # Slice selection
        if slice_axis == 0:
            base = original_volume[idx, :, :]
            heat = cam[idx, :, :]
        elif slice_axis == 1:
            base = original_volume[:, idx, :]
            heat = cam[:, idx, :]
        elif slice_axis == 2:
            base = original_volume[:, :, idx]
            heat = cam[:, :, idx]

        # Normalize base image
        base = (base - np.min(base)) / (np.max(base) - np.min(base) + 1e-5)

        ax = axes[i]
        ax.imshow(base, cmap="gray")
        im = ax.imshow(heat, cmap="jet", alpha=alpha)
        
        if show_contours:
            contours = measure.find_contours(heat, threshold)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='white')

        ax.set_title(f"Slice {idx}")
        ax.axis("off")

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Grad-CAM Intensity')

    plt.suptitle("Grad-CAM Overlay (Multiple Slices)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


import scipy.ndimage

def preprocess_nifti(nifti_path, target_shape=(64, 64, 64)):
    nii = nib.load(nifti_path)
    volume = nii.get_fdata().astype(np.float32)
    volume = np.transpose(volume, (2, 1, 0))  # to Z, Y, X

    # Downsample/resample volume to target_shape
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    volume = scipy.ndimage.zoom(volume, zoom=factors, order=1)  # linear interpolation

    # Normalize
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-5)

    volume = np.expand_dims(volume, axis=0)  # Add channel
    volume = np.expand_dims(volume, axis=0)  # Add batch
    return torch.tensor(volume).to(device), volume[0, 0]

# Load model
model = DenseNet121model()
model.load_state_dict(torch.load("checkpoints/model_best.pth", map_location=device))
model.to(device)
    
# Run inference + Grad-CAM
input_tensor, original_vol = preprocess_nifti("/kaggle/input/abdominal-nifti-0-100/1316_43094.nii")

# Ensure target shape is 3D only (D, H, W)
target_shape_3d = original_vol.shape  # should be (D, H, W)
if len(target_shape_3d) != 3:
    target_shape_3d = target_shape_3d[-3:]  # slice off batch/channel if needed

# Compute Grad-CAM with resizing
cam = compute_gradcam(model, input_tensor, target_head="bowel", target_shape=target_shape_3d)

# Show overlay with enhanced visualization
show_gradcam_overlay(
    cam,
    original_vol,
    slice_axis=0,       # 0 = axial; 1 = coronal; 2 = sagittal
    alpha=0.5,          # blending of heatmap
    threshold=0.5,      # threshold for contour detection
    show_contours=True, # enable contour lines
    slice_range=5       # show midÂ±5 slices
)




