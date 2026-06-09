from glob import glob
import pydicom as dicom #for dicom files
import nibabel as nib #for nii files

import os
import shutil
import gc
from collections import defaultdict
from typing import Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import pydicom
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import kaggle_evaluation.rsna_inference_server

import warnings
warnings.filterwarnings('ignore')


train_images = glob("/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647/*")


path = '/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381.nii'


train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
label_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")


plt.style.use('default')
fig, axes = plt.subplots(4,4, figsize=(12,12))
train_images
for i, ax in enumerate(axes.reshape(-1)):
    img_path = train_images[i]
    img = dicom.dcmread(img_path)  
    ax.imshow(img.pixel_array)
plt.show()


img = nib.load(path).get_fdata()
img.shape


plt.style.use('default')
fig, axes = plt.subplots(4,4, figsize=(12,12))
for i, ax in enumerate(axes.reshape(-1)):
    ax.imshow(img[:,:,1 + i])
plt.show()


# Check class imbalance
print("Aneurysm Present: 1 =", train_df['Aneurysm Present'].mean()*100, "%")
# Check modality distribution
print(train_df['Modality'].value_counts())
# Check location-wise prevalence (critical for multi-label)
locations = [col for col in train_df.columns if 'Artery' in col or 'Communicating' in col]
print(train_df[locations].sum() / len(train_df))


train_df.head()


train_df['PatientAge'].describe()


train_df['PatientAge'] = train_df['PatientAge'].astype(int)
plt.figure(figsize=(10,6))
sns.histplot(train_df['PatientAge'], bins=20, kde=False, color=sns.color_palette("rocket")[4])  
plt.xlabel('Patient Age')
plt.ylabel('Count')
plt.title('Distribution of Patient Age')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Create cross-tabulation with proportions or counts
ctab = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'])

# Plot grouped bar chart
ctab.plot(kind='bar', 
          color=sns.color_palette("pastel"), 
          figsize=(8, 6), 
          width=0.8)

# Labels and title
plt.xlabel('Patient Sex')
plt.ylabel('Count')
plt.title('Aneurysm Presence by Patient Sex')
plt.legend(title='Aneurysm Present', labels=['No', 'Yes'])
plt.xticks(rotation=0)

# Add value labels on bars (optional, improves readability)
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%d', padding=3)

plt.tight_layout()
plt.show()


# Plot pie chart with counts shown on each slice
train_df['Modality'].value_counts().plot(kind='pie', autopct='%d')

# Optional: Improve layout and title
plt.title('Distribution of Modality')
plt.ylabel('')  # Hide the y-label (default is 'Modality' from pandas)
plt.show()


label_df.head()


ID_COL = 'SeriesInstanceUID'

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

DICOM_TAG_ALLOWLIST = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows',
    'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID',
]

