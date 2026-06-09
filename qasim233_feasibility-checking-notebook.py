import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import ast
import json
from collections import Counter

import pydicom
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import cv2

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm


train_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
train_df.head()


# Filter and count data
location_cols = [col for col in train_df.columns if col not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']]
location_counts = train_df[location_cols].sum().sort_values(ascending=False)

# Set style (dark theme similar to plotly_dark)
plt.style.use('dark_background')
sns.set_palette(sns.color_palette(["#00BFC4", "#C77CFF"], n_colors=len(location_counts)))

# Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(location_counts.index, location_counts.values)

# Add gradient coloring (manual approximation)
for i, bar in enumerate(bars):
    # Interpolate between #00BFC4 (start) and #C77CFF (end)
    ratio = i / len(bars)
    r = int(0x00 * (1 - ratio) + 0xC7 * ratio)
    g = int(0xBF * (1 - ratio) + 0x7C * ratio)
    b = int(0xC4 * (1 - ratio) + 0xFF * ratio)
    bar.set_color(f'#{r:02x}{g:02x}{b:02x}')

# Add labels and title
plt.title('Aneurysm Count by Location', fontsize=14, pad=20)
plt.xlabel('Location', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Adjust layout
plt.tight_layout()
plt.show()


# Get modality counts
modality_counts = train_df['Modality'].value_counts()

# Set dark theme (similar to plotly_dark)
plt.style.use('dark_background')

# Create pie chart
fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    modality_counts.values,
    labels=modality_counts.index,
    colors=['#00BFC4', '#C77CFF'],  # Custom colors
    autopct='%1.1f%%',              # Show percentages
    startangle=90,                  # Start at top
    wedgeprops={'linewidth': 1, 'edgecolor': 'black'},  # Edge styling
    pctdistance=0.8,                # Move percentages inward
    textprops={'color': 'white'},   # Text color
)

# Add a hole (donut chart)
centre_circle = plt.Circle((0, 0), 0.4, fc='#1f1f1f')  # Dark background for hole
ax.add_artist(centre_circle)

# Equal aspect ratio ensures pie is drawn as a circle
ax.axis('equal')

# Add title
plt.title('Imaging Modality Distribution', pad=20, fontsize=14)

plt.tight_layout()
plt.show()


plt.style.use('dark_background')
plt.figure(figsize=(10,6))
sns.histplot(
    data=train_df,
    x='PatientAge',
    hue='Aneurysm Present',
    bins=30,
    kde=True,
    palette={0: '#00BFC4', 1: '#C77CFF'}
)
plt.title("Age Distribution by Aneurysm Presence", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.grid(alpha=0.2)
plt.show()


pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'], normalize='index') * 100


# Set dark theme
plt.style.use('dark_background')

# Get counts
counts = train_df['Aneurysm Present'].value_counts().sort_index()

# Define colors
colors = ['#00BFC4', '#C77CFF']
labels = ["No Aneurysm", "Aneurysm"]

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars
bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=1)

# Add count labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', color='white', fontsize=12)

# Customize axes and title
ax.set_title('Class Imbalance: Any Aneurysm Present', pad=20, fontsize=14)
ax.set_ylabel('Count', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Remove top/right spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()


localizers_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')

# Convert coordinate strings to dicts
localizers_df['coords'] = localizers_df['coordinates'].apply(ast.literal_eval)
localizers_df['x'] = localizers_df['coords'].apply(lambda d: d['x'])
localizers_df['y'] = localizers_df['coords'].apply(lambda d: d['y'])

localizers_df.drop(columns=['coordinates', 'coords'], inplace=True)
localizers_df.head()


plt.style.use('dark_background')
sns.set_style("dark")

# Create figure
plt.figure(figsize=(10, 8))

# Create 2D histogram (heatmap)
heatmap, xedges, yedges = np.histogram2d(
    localizers_df['x'],
    localizers_df['y'],
    bins=(50, 50)  # Matching nbinsx=50, nbinsy=50
)

# Plot with Seaborn
ax = sns.heatmap(
    heatmap.T,  # Transpose to match orientation
    cmap='turbo',  # Matches 'Turbo' colorscale
    square=True,
    cbar_kws={'label': 'Density'}
)

# Reverse y-axis to match autorange="reversed"
ax.invert_yaxis()

# Add labels and title
ax.set_title('ðŸ§  Heatmap of Aneurysm Locations in Image Space', pad=20, fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Adjust layout
plt.tight_layout()
plt.show()


# Set dark theme
plt.style.use('dark_background')
palette = sns.color_palette("husl", n_colors=len(localizers_df['location'].unique()))

# Create figure
plt.figure(figsize=(10, 8))

# Create scatter plot
scatter = sns.scatterplot(
    data=localizers_df,
    x='x',
    y='y',
    hue='location',
    palette=palette,
    s=60,  # Marker size
    edgecolor='white',
    linewidth=0.3
)

# Reverse y-axis
plt.gca().invert_yaxis()

# Add title and labels
plt.title('2D Scatter of Aneurysm Coordinates by Location', pad=20, fontsize=14)
plt.xlabel('x')
plt.ylabel('y')

# Customize legend
plt.legend(
    title='Location',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0
)

# Adjust grid
plt.grid(alpha=0.2)

# Equal aspect ratio
plt.gca().set_aspect('equal')

plt.tight_layout()
plt.show()


df = train_df
# Identify location columns correctly
location_cols = df.columns[4:-1]  # skip UID, Age, Sex, Modality, and skip final label
location_df = df[location_cols].astype(int)  # just in case they're still object type

# Co-occurrence matrix
co_matrix = location_df.T.dot(location_df)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(co_matrix, cmap="magma", annot=True, fmt=".0f", linewidths=0.5)
plt.title("Aneurysm Co-occurrence Matrix", fontsize=16, color='white')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.gca().set_facecolor('black')
plt.gcf().set_facecolor('#111111')
plt.tight_layout()
plt.show()


plt.style.use('dark_background')
# Filter only rows where aneurysm is present
df_pos = df[df["Aneurysm Present"] == 1]

# List of all aneurysm location columns
location_cols = df_pos.columns[4:-1]  # From 'Left Infraclinoid...' to 'Other Posterior Circulation'

# Group by PatientSex and sum each location
sex_location = df_pos.groupby("PatientSex")[location_cols].sum().T

# Reset index for plotting
sex_location = sex_location.reset_index().melt(id_vars="index", var_name="Sex", value_name="Count")
sex_location = sex_location.rename(columns={"index": "Location"})

# Plot
plt.figure(figsize=(16, 6))
sns.barplot(data=sex_location, x="Location", y="Count", hue="Sex")
plt.xticks(rotation=45, ha="right")
plt.title("Aneurysm Location Frequency by Sex")
plt.ylabel("Aneurysm Count")
plt.xlabel("Location")
plt.tight_layout()
plt.show()


# List of location columns
location_cols = df.columns[4:-1]

# Filter only positive cases
df_pos = df[df["Aneurysm Present"] == 1]

# Prepare a DataFrame: for each location, collect patients with that location = 1
age_location = []

for loc in location_cols:
    subset = df_pos[df_pos[loc] == 1][["PatientAge"]].copy()
    subset["Location"] = loc
    age_location.append(subset)

# Combine all into one DataFrame
age_location_df = pd.concat(age_location)

# Plot
plt.figure(figsize=(16, 6))
sns.boxplot(data=age_location_df, x="Location", y="PatientAge", palette="crest")
plt.xticks(rotation=45, ha="right")
plt.title("Patient Age Distribution per Aneurysm Location")
plt.ylabel("Age")
plt.xlabel("Location")
plt.tight_layout()
plt.show()


# Filter rows with aneurysm present
df_pos = df[df["Aneurysm Present"] == 1].copy()

# Initialize a dictionary to store counts
modality_counts = {}

for loc in location_cols:
    subset = df_pos[df_pos[loc] == 1]
    modality_distribution = subset["Modality"].value_counts()
    modality_counts[loc] = modality_distribution

# Convert to DataFrame and fill missing values with 0
modality_df = pd.DataFrame(modality_counts).T.fillna(0).astype(int)
modality_df = modality_df[["CTA", "MRA"]] if "CTA" in modality_df.columns and "MRA" in modality_df.columns else modality_df

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(modality_df.T, annot=True, cmap="crest", fmt="d")
plt.title("Modality Preference per Aneurysm Location")
plt.xlabel("Location")
plt.ylabel("Modality")
plt.tight_layout()
plt.show()


train_localizations = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")
# Parse coordinates
train_localizations["coord_dict"] = train_localizations["coordinates"].apply(ast.literal_eval)
train_localizations["x"] = train_localizations["coord_dict"].apply(lambda d: d["x"])
train_localizations["y"] = train_localizations["coord_dict"].apply(lambda d: d["y"])

# Top 4-5 most frequent locations
top_locations = train_localizations["location"].value_counts().head(5).index

# Plot heatmap per location
for loc in top_locations:
    subset = train_localizations[train_localizations["location"] == loc]
    plt.figure(figsize=(6, 5))
    sns.kdeplot(data=subset, x="x", y="y", fill=True, cmap="crest")
    plt.title(f"Spatial Heatmap for {loc}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.show()


# Dataset Exploration and Setup
# Set up paths
BASE_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection"
TRAIN_CSV = f"{BASE_PATH}/train.csv"
TRAIN_LOCALIZERS_CSV = f"{BASE_PATH}/train_localizers.csv"
SEGMENTATIONS_PATH = f"{BASE_PATH}/segmentations"
SERIES_PATH = f"{BASE_PATH}/series"

# Load CSV files
print("Loading CSV files...")

train_df = pd.read_csv(TRAIN_CSV)
print(f"Train.csv shape: {train_df.shape}")
print(f"Train.csv columns: {train_df.columns.tolist()}")
print("\nTrain.csv head:")
train_df.head()


train_localizers_df = pd.read_csv(TRAIN_LOCALIZERS_CSV)
print(f"\nTrain_localizers.csv shape: {train_localizers_df.shape}")
print(f"Train_localizers.csv columns: {train_localizers_df.columns.tolist()}")
print("\nTrain_localizers.csv head:")
train_localizers_df.head()


# Explore directory structure
print("\n" + "="*50)
print("Segmentation Directory STRUCTURE ANALYSIS")
print("="*50)

# Count segmentation files
seg_folders = glob.glob(f"{SEGMENTATIONS_PATH}/*")
print(f"Number of segmentation folders: {len(seg_folders)}")

nii_files = glob.glob(f"{SEGMENTATIONS_PATH}/*/*.nii")
cowseg_files = glob.glob(f"{SEGMENTATIONS_PATH}/*/*_cowseg.nii")
regular_nii = [f for f in nii_files if not f.endswith('_cowseg.nii')]

print(f"Total .nii files: {len(nii_files)}")
print(f"Cowseg files: {len(cowseg_files)}")
print(f"Regular .nii files: {len(regular_nii)}")


print("\n" + "="*50)
print("Series Directory STRUCTURE ANALYSIS")
print("="*50)
# Count DICOM series
series_folders = glob.glob(f"{SERIES_PATH}/*")
print(f"Number of series folders: {len(series_folders)}")

dcm_files = glob.glob(f"{SERIES_PATH}/*/*.dcm")
print(f"Total .dcm files: {len(dcm_files)}")

if len(series_folders) > 0:
    # Sample a few folders to see DICOM count distribution
    sample_folders = series_folders[:5]
    for folder in sample_folders:
        dcm_count = len(glob.glob(f"{folder}/*.dcm"))
        folder_name = os.path.basename(folder)
        print(f"  {folder_name}: {dcm_count} DICOM files")



#Detailed Data Analysis
print("DETAILED DATA ANALYSIS")
print("="*50)

# Analyze train_localizers.csv in detail
if 'train_localizers_df' in locals():
    print("Analyzing localizer annotations...")
    
    # Parse coordinates
    def parse_coordinates(coord_str):
        try:
            if isinstance(coord_str, str):
                coord_dict = ast.literal_eval(coord_str)
                return coord_dict['x'], coord_dict['y']
            return None, None
        except:
            return None, None
    
    train_localizers_df['x_coord'] = train_localizers_df['coordinates'].apply(lambda x: parse_coordinates(x)[0])
    train_localizers_df['y_coord'] = train_localizers_df['coordinates'].apply(lambda x: parse_coordinates(x)[1])
    
    # Analyze locations
    location_counts = train_localizers_df['location'].value_counts()
    print(f"\nAneurysm locations distribution:")
    print(location_counts)
    
    # Coordinate statistics
    print(f"\nCoordinate statistics:")
    print(f"X coordinates - Min: {train_localizers_df['x_coord'].min():.2f}, Max: {train_localizers_df['x_coord'].max():.2f}")
    print(f"Y coordinates - Min: {train_localizers_df['y_coord'].min():.2f}, Max: {train_localizers_df['y_coord'].max():.2f}")
    
    # Plot location distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    location_counts.plot(kind='bar', rot=45)
    plt.title('Aneurysm Location Distribution')
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    plt.scatter(train_localizers_df['x_coord'], train_localizers_df['y_coord'], alpha=0.6)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Aneurysm Coordinate Distribution')
    plt.tight_layout()
    plt.show()


# Analyze unique series and studies
unique_series = train_localizers_df['SeriesInstanceUID'].nunique()
unique_sop = train_localizers_df['SOPInstanceUID'].nunique()
print(f"\nUnique SeriesInstanceUID: {unique_series}")
print(f"Unique SOPInstanceUID: {unique_sop}")
print(f"Total annotations: {len(train_localizers_df)}")
print(f"Average annotations per series: {len(train_localizers_df) / unique_series:.2f}")



# Medical Imaging Data Loader
def load_dicom_series(series_path):
    """Load a complete DICOM series and sort by instance number"""
    try:
        dcm_files = glob.glob(f"{series_path}/*.dcm")
        if not dcm_files:
            return None, None
        
        # Load all DICOM files
        dicoms = []
        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                dicoms.append((ds.InstanceNumber, ds))
            except:
                continue
        
        if not dicoms:
            return None, None
        
        # Sort by instance number
        dicoms.sort(key=lambda x: x[0])
        
        # Extract pixel arrays
        images = []
        metadata = []
        for _, ds in dicoms:
            if hasattr(ds, 'pixel_array'):
                img = ds.pixel_array.astype(np.float32)
                images.append(img)
                metadata.append({
                    'InstanceNumber': ds.InstanceNumber,
                    'SliceLocation': getattr(ds, 'SliceLocation', None),
                    'ImagePositionPatient': getattr(ds, 'ImagePositionPatient', None),
                    'PixelSpacing': getattr(ds, 'PixelSpacing', None),
                    'WindowCenter': getattr(ds, 'WindowCenter', None),
                    'WindowWidth': getattr(ds, 'WindowWidth', None)
                })
        
        if images:
            volume = np.stack(images, axis=0)
            return volume, metadata
        
    except Exception as e:
        print(f"Error loading DICOM series: {e}")
        return None, None
    
    return None, None



def load_nifti_file(nii_path):
    """Load NIfTI file"""
    try:
        nii_img = nib.load(nii_path)
        data = nii_img.get_fdata()
        return data, nii_img.header
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return None, None

def apply_window_level(image, window_center, window_width):
    """Apply window/level to DICOM image"""
    if window_center is None or window_width is None:
        return image
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    windowed = np.clip(image, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    return windowed


# Test loading functions
print("Testing medical imaging loaders...")

# Test DICOM loading
if len(series_folders) > 0:
    test_series = series_folders[0]
    print(f"Testing DICOM loading with: {os.path.basename(test_series)}")
    
    volume, metadata = load_dicom_series(test_series)
    if volume is not None:
        print(f"DICOM volume shape: {volume.shape}")
        print(f"Number of slices: {len(metadata)}")
        print(f"Pixel value range: {volume.min():.2f} to {volume.max():.2f}")
        
        # Display middle slice
        middle_slice = volume.shape[0] // 2
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(volume[middle_slice], cmap='gray')
        plt.title(f'DICOM Slice {middle_slice} (Raw)')
        plt.axis('off')
        
        # Apply windowing if available
        if metadata[middle_slice]['WindowCenter'] and metadata[middle_slice]['WindowWidth']:
            windowed = apply_window_level(
                volume[middle_slice], 
                metadata[middle_slice]['WindowCenter'], 
                metadata[middle_slice]['WindowWidth']
            )
            plt.subplot(1, 2, 2)
            plt.imshow(windowed, cmap='gray')
            plt.title(f'DICOM Slice {middle_slice} (Windowed)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# Test NIfTI loading
if len(regular_nii) > 0:
    test_nii = regular_nii[0]
    print(f"\nTesting NIfTI loading with: {os.path.basename(test_nii)}")
    
    nii_data, nii_header = load_nifti_file(test_nii)
    if nii_data is not None:
        print(f"NIfTI volume shape: {nii_data.shape}")
        print(f"NIfTI data range: {nii_data.min():.2f} to {nii_data.max():.2f}")
        
        # Display middle slice
        if len(nii_data.shape) == 3:
            middle_slice = nii_data.shape[2] // 2
            plt.figure(figsize=(8, 4))
            plt.imshow(nii_data[:, :, middle_slice], cmap='gray')
            plt.title(f'NIfTI Slice {middle_slice}')
            plt.axis('off')
            plt.show()


# Annotation Visualization
def visualize_annotations_on_dicom(series_uid, localizers_df):
    """Visualize aneurysm annotations on DICOM images"""
    
    # Find annotations for this series
    series_annotations = localizers_df[localizers_df['SeriesInstanceUID'] == series_uid]
    
    if len(series_annotations) == 0:
        print(f"No annotations found for series {series_uid}")
        return
    
    # Find corresponding DICOM series folder
    series_folder = None
    for folder in series_folders:
        if series_uid in folder:
            series_folder = folder
            break
    
    if series_folder is None:
        print(f"DICOM folder not found for series {series_uid}")
        return
    
    # Load DICOM series
    volume, metadata = load_dicom_series(series_folder)
    if volume is None:
        print(f"Failed to load DICOM series")
        return
    
    print(f"Found {len(series_annotations)} annotations for series")
    print(f"DICOM volume shape: {volume.shape}")
    
    # Group annotations by SOPInstanceUID (individual images)
    sop_groups = series_annotations.groupby('SOPInstanceUID')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for sop_uid, group in sop_groups:
        if plot_idx >= 6:  # Limit to 6 images
            break
            
        # For simplicity, show first slice (in practice, you'd match SOPInstanceUID)
        slice_idx = min(plot_idx, volume.shape[0] - 1)
        img = volume[slice_idx]
        
        # Apply windowing if available
        if metadata[slice_idx]['WindowCenter'] and metadata[slice_idx]['WindowWidth']:
            img = apply_window_level(
                img, 
                metadata[slice_idx]['WindowCenter'], 
                metadata[slice_idx]['WindowWidth']
            )
        
        axes[plot_idx].imshow(img, cmap='gray')
        
        # Add annotation points
        for _, row in group.iterrows():
            x, y = parse_coordinates(row['coordinates'])
            if x is not None and y is not None:
                circle = Circle((x, y), radius=5, color='red', fill=False, linewidth=2)
                axes[plot_idx].add_patch(circle)
                axes[plot_idx].text(x+10, y-10, row['location'][:20], 
                                   color='red', fontsize=8, weight='bold')
        
        axes[plot_idx].set_title(f'Slice {slice_idx}\n{len(group)} annotations')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 6):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()



# Test annotation visualization
if 'train_localizers_df' in locals() and len(train_localizers_df) > 0:
    # Get a series with multiple annotations
    series_counts = train_localizers_df['SeriesInstanceUID'].value_counts()
    test_series = series_counts.index[0]  # Series with most annotations
    
    print(f"Visualizing annotations for series with {series_counts.iloc[0]} annotations")
    visualize_annotations_on_dicom(test_series, train_localizers_df)



# Basic Model Development Setup
class AneurysmDataset(Dataset):
    """Dataset class for aneurysm detection"""
    
    def __init__(self, series_paths, annotations_df, transform=None, image_size=512):
        self.series_paths = series_paths
        self.annotations_df = annotations_df
        self.transform = transform
        self.image_size = image_size
        
        # Create mapping from series to annotations
        self.series_to_annotations = {}
        for series_uid in annotations_df['SeriesInstanceUID'].unique():
            series_annotations = annotations_df[annotations_df['SeriesInstanceUID'] == series_uid]
            self.series_to_annotations[series_uid] = series_annotations
    
    def __len__(self):
        return len(self.series_paths)
    
    def __getitem__(self, idx):
        series_path = self.series_paths[idx]
        series_uid = os.path.basename(series_path)
        
        # Load DICOM series
        volume, metadata = load_dicom_series(series_path)
        if volume is None:
            # Return dummy data if loading fails
            return torch.zeros(3, self.image_size, self.image_size), torch.zeros(1, 4)
        
        # For simplicity, use middle slice
        middle_slice = volume.shape[0] // 2
        image = volume[middle_slice]
        
        # Normalize and resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Convert to 3-channel
        image = np.stack([image, image, image], axis=0)
        
        # Get annotations for this series
        annotations = self.series_to_annotations.get(series_uid, pd.DataFrame())
        
        # Create bounding boxes (simplified - using point annotations as centers)
        boxes = []
        labels = []
        
        for _, row in annotations.iterrows():
            x, y = parse_coordinates(row['coordinates'])
            if x is not None and y is not None:
                # Scale coordinates to resized image
                x_scaled = x * (self.image_size / volume.shape[2])
                y_scaled = y * (self.image_size / volume.shape[1])
                
                # Create bounding box around point (simplified)
                box_size = 32
                x1 = max(0, x_scaled - box_size // 2)
                y1 = max(0, y_scaled - box_size // 2)
                x2 = min(self.image_size, x_scaled + box_size // 2)
                y2 = min(self.image_size, y_scaled + box_size // 2)
                
                boxes.append([x1, y1, x2, y2])
                labels.append(1)  # Aneurysm class
        
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]  # Dummy box
            labels = [0]  # Background
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
        
        return image, {'boxes': boxes, 'labels': labels}


def custom_collate(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)



class SimpleAneurysmDetector(nn.Module):
    """Simple CNN for aneurysm detection"""
    
    def __init__(self, num_classes=2):
        super(SimpleAneurysmDetector, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


# Initialize model and check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Create simple model
model = SimpleAneurysmDetector(num_classes=2)
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Model architecture:")
print(model)



# Training Pipeline Setup
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        
        # For simplicity, convert to classification task
        # (has_aneurysm = 1 if any annotations, 0 otherwise)
        labels = torch.tensor([1 if len(t['labels']) > 0 and t['labels'][0] > 0 else 0 
                              for t in targets], dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            labels = torch.tensor([1 if len(t['labels']) > 0 and t['labels'][0] > 0 else 0 
                                  for t in targets], dtype=torch.long).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(dataloader), 100. * correct / total


# Setup training components
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Create dataset (using available series)
available_series = [folder for folder in series_folders if os.path.exists(folder)][:50]  # Limit for testing
print(f"Using {len(available_series)} series for training")

if len(available_series) > 0 and 'train_localizers_df' in locals():
    # Split data
    train_series, val_series = train_test_split(available_series, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = AneurysmDataset(train_series, train_localizers_df, image_size=256)
    val_dataset = AneurysmDataset(val_series, train_localizers_df, image_size=256)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=custom_collate)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Test one batch
    print("\nTesting data loading...")
    try:
        sample_batch = next(iter(train_loader))
        images, targets = sample_batch

        print(f"Type of images: {type(images)}; Batch size: {len(images)}")
        for idx, img in enumerate(images):
            print(f"  img[{idx}]: type {type(img)}, shape {tuple(img.shape)}")
        print(f"Number of targets: {len(targets)}, first target keys: {targets[0].keys()}")
        
        # Forward pass (list of different-sized tensors)
        model.eval()
        with torch.no_grad():
            logits_list = []
            for img in images:
                img = img.to(device).unsqueeze(0)  # add batch dim = 1
                logits = model(img)                # e.g. shape [1, num_classes]
                logits_list.append(logits.squeeze(0))  # shape [num_classes]
        
            outputs = torch.stack(logits_list, dim=0)  # shape [batch, num_classes]
            print(f"Stacked model output: {outputs.shape}")
        
        print("Forward pass successful!")
            
    except Exception as e:
        print(f"Error in data loading/forward pass: {e}")

else:
    print("Insufficient data for training setup")



# Inference Pipeline
def predict_aneurysm(model, series_path, device, image_size=256):
    """Predict aneurysm presence in a DICOM series"""
    model.eval()
    
    # Load DICOM series
    volume, metadata = load_dicom_series(series_path)
    if volume is None:
        return None, "Failed to load DICOM series"
    
    predictions = []
    confidences = []
    
    with torch.no_grad():
        # Process each slice
        for slice_idx in range(volume.shape[0]):
            image = volume[slice_idx]
            
            # Preprocess
            image = cv2.resize(image, (image_size, image_size))
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image = np.stack([image, image, image], axis=0)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Predict
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)
            
            predictions.append(predicted.cpu().item())
            confidences.append(confidence.cpu().item())
    
    # Aggregate predictions (majority vote or max confidence)
    has_aneurysm = sum(predictions) > len(predictions) // 2
    max_confidence = max(confidences)
    
    return {
        'has_aneurysm': has_aneurysm,
        'confidence': max_confidence,
        'slice_predictions': predictions,
        'slice_confidences': confidences,
        'num_slices': len(predictions)
    }, None


def batch_inference(model, series_list, device, max_series=10):
    """Run inference on multiple series"""
    results = []
    
    print(f"Running inference on {min(len(series_list), max_series)} series...")
    
    for i, series_path in enumerate(series_list[:max_series]):
        series_name = os.path.basename(series_path)
        print(f"Processing {i+1}/{min(len(series_list), max_series)}: {series_name[:50]}...")
        
        result, error = predict_aneurysm(model, series_path, device)
        
        if error:
            print(f"  Error: {error}")
            continue
        
        results.append({
            'series_path': series_path,
            'series_name': series_name,
            'has_aneurysm': result['has_aneurysm'],
            'confidence': result['confidence'],
            'num_slices': result['num_slices']
        })
        
        print(f"  Result: {'Aneurysm' if result['has_aneurysm'] else 'No Aneurysm'} "
              f"(confidence: {result['confidence']:.3f})")
    
    return results


# Run sample inference
if len(available_series) > 0:
    print("Running sample inference...")
    
    # Test on a few series
    inference_results = batch_inference(model, available_series, device, max_series=5)
    
    if inference_results:
        print(f"\nInference Summary:")
        print(f"Total series processed: {len(inference_results)}")
        
        aneurysm_count = sum(1 for r in inference_results if r['has_aneurysm'])
        print(f"Predicted aneurysms: {aneurysm_count}")
        print(f"Predicted normal: {len(inference_results) - aneurysm_count}")
        
        avg_confidence = np.mean([r['confidence'] for r in inference_results])
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Show detailed results
        print("\nDetailed Results:")
        for result in inference_results:
            status = "ANEURYSM" if result['has_aneurysm'] else "NORMAL"
            print(f"  {result['series_name'][:40]}: {status} (conf: {result['confidence']:.3f})")



# Feasibility Analysis Report
def generate_feasibility_report():
    """Generate a comprehensive feasibility report"""
    
    print("="*60)
    print("RSNA INTRACRANIAL ANEURYSM DETECTION - FEASIBILITY REPORT")
    print("="*60)
    
    # Dataset Analysis
    print("\n1. DATASET ANALYSIS")
    print("-" * 30)
    
    if 'train_localizers_df' in locals():
        print(f"âœ“ Localizer annotations: {len(train_localizers_df)} samples")
        print(f"âœ“ Unique series: {train_localizers_df['SeriesInstanceUID'].nunique()}")
        print(f"âœ“ Unique locations: {train_localizers_df['location'].nunique()}")
        
        location_dist = train_localizers_df['location'].value_counts()
        print(f"âœ“ Most common location: {location_dist.index[0]} ({location_dist.iloc[0]} cases)")
        print(f"âœ“ Least common location: {location_dist.index[-1]} ({location_dist.iloc[-1]} cases)")
    
    print(f"âœ“ DICOM series folders: {len(series_folders)}")
    print(f"âœ“ Total DICOM files: {len(dcm_files)}")
    print(f"âœ“ NIfTI segmentation files: {len(nii_files)}")
    print(f"âœ“ Cowseg files: {len(cowseg_files)}")
    
    # Technical Feasibility
    print("\n2. TECHNICAL FEASIBILITY")
    print("-" * 30)
    
    print(f"âœ“ GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"âœ“ GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"âœ“ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"âœ“ DICOM Loading: {'Success' if 'volume' in locals() and volume is not None else 'Failed'}")
    print(f"âœ“ NIfTI Loading: {'Success' if 'nii_data' in locals() and nii_data is not None else 'Failed'}")
    print(f"âœ“ Model Creation: Success")
    print(f"âœ“ Data Pipeline: {'Success' if 'train_loader' in locals() else 'Partial'}")
    
    # Challenges and Recommendations
    print("\n3. CHALLENGES & RECOMMENDATIONS")
    print("-" * 30)
    
    challenges = [
        "Class imbalance - some aneurysm locations are rare",
        "3D nature of data - need to handle volumetric information",
        "Variable image quality and protocols across institutions",
        "Small aneurysm size - requires high-resolution processing",
        "Limited training data for some anatomical locations"
    ]
    
    recommendations = [
        "Use 3D CNNs or 2.5D approaches for better spatial context",
        "Implement data augmentation for rare classes",
        "Consider ensemble methods combining multiple views",
        "Use attention mechanisms to focus on relevant regions",
        "Implement proper cross-validation strategy",
        "Consider transfer learning from pre-trained medical imaging models"
    ]
    
    print("Challenges:")
    for i, challenge in enumerate(challenges, 1):
        print(f"  {i}. {challenge}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Model Architecture Suggestions
    print("\n4. SUGGESTED MODEL ARCHITECTURES")
    print("-" * 30)
    
    architectures = [
        "3D ResNet/DenseNet for volumetric analysis",
        "U-Net for segmentation-based detection",
        "YOLO/RetinaNet for object detection approach",
        "Vision Transformer (ViT) for attention-based detection",
        "Multi-scale CNN with Feature Pyramid Networks"
    ]
    
    for i, arch in enumerate(architectures, 1):
        print(f"  {i}. {arch}")
    
    # Performance Expectations
    print("\n5. PERFORMANCE EXPECTATIONS")
    print("-" * 30)
    
    print("Based on similar medical imaging competitions:")
    print("  â€¢ Baseline CNN: 0.65-0.75 AUC")
    print("  â€¢ Advanced 3D models: 0.75-0.85 AUC")
    print("  â€¢ Ensemble methods: 0.80-0.90 AUC")
    print("  â€¢ Top solutions: 0.85-0.95 AUC")
    
    # Resource Requirements
    print("\n6. RESOURCE REQUIREMENTS")
    print("-" * 30)
    
    print("Computational:")
    print("  â€¢ GPU: 16GB+ VRAM recommended for 3D models")
    print("  â€¢ RAM: 32GB+ for large batch processing")
    print("  â€¢ Storage: 100GB+ for preprocessed data")
    print("  â€¢ Training time: 2-5 days for full training")
    
    print("\nDevelopment:")
    print("  â€¢ Data preprocessing: 1-2 weeks")
    print("  â€¢ Model development: 2-3 weeks")
    print("  â€¢ Hyperparameter tuning: 1-2 weeks")
    print("  â€¢ Ensemble creation: 1 week")
    
    print("\n" + "="*60)
    print("CONCLUSION: Project is FEASIBLE with proper approach")
    print("="*60)

# Generate the report
generate_feasibility_report()

# Additional utility functions for competition
print("\n" + "="*40)
print("UTILITY FUNCTIONS FOR COMPETITION")
print("="*40)

def create_submission_format():
    """Create sample submission format"""
    # This would be based on the actual submission requirements
    sample_submission = pd.DataFrame({
        'SeriesInstanceUID': ['sample_series_1', 'sample_series_2'],
        'aneurysm_probability': [0.85, 0.23]
    })
    print("Sample submission format:")
    print(sample_submission)
    return sample_submission

def preprocessing_pipeline():
    """Outline preprocessing steps"""
    steps = [
        "1. Load DICOM series and sort by slice location",
        "2. Apply appropriate windowing (brain window: 80/40)",
        "3. Resample to consistent spacing (e.g., 1mm isotropic)",
        "4. Crop/pad to consistent dimensions",
        "5. Normalize intensity values",
        "6. Apply data augmentation (rotation, scaling, noise)",
        "7. Create 3D patches or 2.5D slices for training"
    ]
    
    print("Preprocessing Pipeline:")
    for step in steps:
        print(f"  {step}")

create_submission_format()
print()
preprocessing_pipeline()


