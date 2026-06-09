import os
import pydicom
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from glob import glob


# Set plot style
sns.set(style="whitegrid")


# Load train.csv
train_df = pd.read_csv('../input/rsna-intracranial-aneurysm-detection/train.csv')
print("Train shape:", train_df.shape)
display(train_df.head())
display(train_df.describe())


# Basic info
print("\nUnique modalities:", train_df['Modality'].unique())
print("Missing values:\n", train_df.isnull().sum())


plt.figure(figsize=(12, 5))
# Age distribution by sex
plt.subplot(1, 2, 1)
sns.histplot(data=train_df, x='PatientAge', hue='PatientSex', multiple='stack', bins=20)
plt.title('Age Distribution by Sex')
plt.xlabel('Age')
plt.ylabel('Count')

# Sex distribution
plt.subplot(1, 2, 2)
sns.countplot(x='PatientSex', data=train_df)
plt.title('Sex Distribution')

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(y='Modality', data=train_df)
plt.title('Modality Distribution')
plt.xlabel('Count')
plt.ylabel('Modality')
plt.show()


plt.figure(figsize=(6, 6))
train_df['Aneurysm Present'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Absent', 'Present'])
plt.title('Aneurysm Present Distribution')
plt.ylabel('')
plt.show()


location_cols = train_df.columns[4:-1]  # Location columns
label_counts = train_df[location_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=label_counts.values, y=label_counts.index)
plt.title('Aneurysm Counts by Location')
plt.xlabel('Count')
plt.ylabel('Location')
plt.show()

# Multi-label analysis: Number of aneurysms per positive case
positive_df = train_df[train_df['Aneurysm Present'] == 1]
positive_df['Num_Locations'] = positive_df[location_cols].sum(axis=1)
sns.countplot(x='Num_Locations', data=positive_df)
plt.title('Number of Aneurysm Locations per Positive Case')
plt.xlabel('Number of Locations')
plt.ylabel('Count')
plt.show()


corr = train_df[location_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Location Labels')
plt.show()


localizers_df = pd.read_csv('../input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
print("\nLocalizers shape:", localizers_df.shape)
display(localizers_df.head())

# Distribution of aneurysms per series
aneurysms_per_series = localizers_df.groupby('SeriesInstanceUID').size()
sns.histplot(aneurysms_per_series, bins=10)
plt.title('Number of Annotated Aneurysms per Series')
plt.xlabel('Number of Aneurysms')
plt.ylabel('Count')
plt.show()


# Find a sample series (e.g., first one with aneurysm)
sample_series = train_df[train_df['Aneurysm Present'] == 1]['SeriesInstanceUID'].iloc[0]
dicom_path = f'../input/rsna-intracranial-aneurysm-detection/series/{sample_series}/*.dcm'
dicom_files = sorted(glob(dicom_path))

if dicom_files:
    # Load middle slice
    mid_idx = len(dicom_files) // 2 + 1
    ds = pydicom.dcmread(dicom_files[mid_idx])
    
    plt.figure(figsize=(8, 8))
    plt.imshow(ds.pixel_array, cmap='gray')
    plt.title(f'Sample DICOM Slice from Series: {sample_series}')
    plt.axis('off')
    plt.show()
else:
    print("No DICOM files found for sample series.")


seg_paths = glob('../input/rsna-intracranial-aneurysm-detection/segmentations/*/*.nii')
print("\nNumber of segmentation files:", len(seg_paths))


import json
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import sample
from scipy.ndimage import zoom

# Set random seed for reproducibility
random.seed(22)

# Modified plot_around_label to visualize aneurysm location with rotated seg, label name, and modality
def plot_around_label(row_number, num_around=2, localizers_df=localizers_df, train_df=train_df):
    row = localizers_df.iloc[row_number]
    a = row['SeriesInstanceUID']
    b = row['SOPInstanceUID']
    c = row['coordinates']
    d = row['location']
    
    # Get Modality from train_df
    modality = train_df[train_df['SeriesInstanceUID'] == a]['Modality'].iloc[0] if not train_df[train_df['SeriesInstanceUID'] == a].empty else 'Unknown'
    
    folders = glob("../input/rsna-intracranial-aneurysm-detection/series/*")
    
    for i in folders:
        if a in i:
            break
    files = sorted(glob(os.path.join(i, "*")))
    i_n = list()
    for e, f in enumerate(files):
        ex = pydicom.dcmread(f)
        i_n.append([ex.InstanceNumber,f])
    
    i_n = sorted(i_n)
    
    for e, f in enumerate(i_n):
        if b in f[1]:
            break
    
    c = json.loads(c.replace("'",'"'))
    
    # Load seg if available
    seg_path = next((p for p in seg_paths if a in p), None)
    if seg_path:
        seg = nib.load(seg_path).get_fdata()
        if seg.shape[2] != len(i_n):
            zoom_factors = (1, 1, len(i_n) / seg.shape[2])
            seg = zoom(seg, zoom_factors, order=0)
    else:
        seg = None

    for i in range(e-num_around,e+num_around+1):
        if i < 0 or i >= len(i_n):
            continue
        if i == e:
            print("labeled slice")
        else:
            print(f"nearby slice {i}")
        ex = pydicom.dcmread(i_n[i][1])
        if len(ex.pixel_array.shape) > 2:
            continue
        
        img = ex.pixel_array
        img = np.flipud(img)  # Flip vertically to match DICOM top-left origin
        seg_slice = seg[:, :, i] if seg is not None else np.zeros_like(img)
        seg_slice = np.flipud(seg_slice)  # Flip segmentation to align with image
        
        height, width = img.shape
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Image
        axs[0].imshow(img, cmap='gray', origin='upper', aspect=1)  # Use upper origin for flipped image
        axs[0].set_title('Image')
        axs[0].axis('off')
        
        # Panel 2: Seg + Label (seg rotated -90 degrees, clockwise)
        seg_slice_rotated = np.rot90(seg_slice, k=-1)  # Rotate 90 degrees clockwise
        axs[1].imshow(seg_slice_rotated, cmap='jet', origin='upper', aspect=1)
        if i == e:
            # Use flipped coordinates without rotation for rectangle
            # flipud: (x, y) -> (x, height - 1 - y)
            x = c["x"]
            y = height - 1 - c["y"]
            rect = patches.Rectangle((x-10, y-10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
            axs[1].add_patch(rect)
            # Add label name and modality above rectangle
            axs[1].text(x, y+35, d, color='yellow', fontsize=10, ha='center', va='bottom')
            axs[1].text(x, y+55, f'Modality: {modality}', color='red', fontsize=10, ha='center', va='bottom')
        axs[1].set_title('Seg + Label')
        axs[1].axis('off')
        
        # Panel 3: Overlay (Image + Seg + Label, no rotation for seg)
        axs[2].imshow(img, cmap='gray', origin='upper', aspect=1)
        axs[2].imshow(seg_slice_rotated, cmap='jet', alpha=0.5, origin='upper', aspect=1)  # Use non-rotated seg
        if i == e:
            # Same coordinates for non-rotated, flipped image
            x = c["x"]
            y = height - 1 - c["y"]
            rect = patches.Rectangle((x-10, y-10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
            axs[2].add_patch(rect)
            # Add label name and modality above rectangle
            axs[2].text(x, y+35, d, color='yellow', fontsize=10, ha='center', va='bottom')
            axs[2].text(x, y+55, f'Modality: {modality}', color='red', fontsize=10, ha='center', va='bottom')
        axs[2].set_title('Overlay')
        axs[2].axis('off')
        
        plt.show()

# Select a random row, prefer one with seg
all_rows = range(len(localizers_df))
rows_with_seg = [r for r in all_rows if localizers_df.iloc[r]['SeriesInstanceUID'] in [os.path.basename(os.path.dirname(p)) for p in seg_paths]]
row_number = 52
print(f"\nDisplaying slides for row {row_number}")
plot_around_label(row_number, 2)




