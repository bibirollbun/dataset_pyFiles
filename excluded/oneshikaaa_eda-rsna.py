import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pydicom

import os

from tqdm.notebook import tqdm

sns.set_style("whitegrid")


BASE_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection/' # Make sure this path is correct

train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
localizers_df = pd.read_csv(os.path.join(BASE_PATH, 'train_localizers.csv'))

print(f"Number of scans in train_df: {len(train_df)}")
train_df.head()


plt.figure(figsize=(8, 5))
ax = sns.countplot(data=train_df, x='Aneurysm Present', palette='viridis')

total = len(train_df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.title('Distribution of "Aneurysm Present" Target', fontsize=16)
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(data=train_df, x='PatientAge', hue='Aneurysm Present', kde=True, ax=axes[0], palette='magma')
axes[0].set_title('Age Distribution by Aneurysm Presence')

sns.countplot(data=train_df, x='PatientSex', hue='Aneurysm Present', ax=axes[1], palette='plasma')
axes[1].set_title('Sex Distribution by Aneurysm Presence')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.countplot(data=localizers_df, y='location', order=localizers_df['location'].value_counts().index, palette='crest')
plt.title('Frequency of Aneurysm Locations', fontsize=16)
plt.xlabel('Count')
plt.ylabel('Anatomical Location')
plt.show()


aneurysms_per_scan = localizers_df.groupby('SeriesInstanceUID').size().reset_index(name='aneurysm_count')

plt.figure(figsize=(10, 5))
sns.countplot(data=aneurysms_per_scan, x='aneurysm_count', palette='rocket')
plt.title('Number of Aneurysms per Scan (for positive scans)', fontsize=16)
plt.xlabel('Number of Aneurysms')
plt.ylabel('Number of Scans')
plt.show()


import ast 

def view_aneurysm_in_scan(series_uid, localizers_df):
    """
    Loads a 3D scan and displays the slice with the aneurysm highlighted.
    Handles both list and dictionary coordinate formats.
    """
    print(f"Loading and visualizing scan: {series_uid}")
    
    # Path to the series directory
    series_path = os.path.join(BASE_PATH, 'series', series_uid)
    
    # Load all DICOM files for the series
    dicom_files = [pydicom.dcmread(os.path.join(series_path, f)) for f in os.listdir(series_path)]
    
    # Sort the slices by their position along the Z-axis
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Get the aneurysm info for this scan
    aneurysm_info = localizers_df[localizers_df['SeriesInstanceUID'] == series_uid].iloc[0]
    coords_str = aneurysm_info['coordinates']
    
    # Use ast.literal_eval to safely parse the string into a Python object (list or dict)
    coords_obj = ast.literal_eval(coords_str)
    
    # Check if the parsed object is a dictionary or a list and extract coordinates
    if isinstance(coords_obj, dict):
        x_coord, y_coord = int(coords_obj['x']), int(coords_obj['y'])
    elif isinstance(coords_obj, list):
        x_coord, y_coord = int(coords_obj[0]), int(coords_obj[1])
    else:
        print(f"Unknown coordinate format for series {series_uid}")
        return

    # Find which slice the aneurysm is on using its SOPInstanceUID
    sop_uid = aneurysm_info['SOPInstanceUID']
    target_slice_index = -1
    for i, s in enumerate(dicom_files):
        if s.SOPInstanceUID == sop_uid:
            target_slice_index = i
            break
            
    if target_slice_index == -1:
        print("Error: Could not find the specified slice in the series.")
        return

    # Stack the pixel data into a 3D numpy array
    volume = np.stack([d.pixel_array for d in dicom_files])
    
    # Display the target slice
    plt.figure(figsize=(10, 10))
    plt.imshow(volume[target_slice_index], cmap='bone')
    
    # Draw a circle to highlight the aneurysm
    circle = plt.Circle((x_coord, y_coord), radius=20, color='red', fill=False, lw=2)
    plt.gca().add_patch(circle)
    
    plt.title(f'Aneurysm in Scan {series_uid}\nSlice: {target_slice_index}, Location: ({x_coord}, {y_coord})', fontsize=16)
    plt.axis('off')
    plt.show()

# Now, when you run this, it should work correctly
positive_series_uid = localizers_df['SeriesInstanceUID'].iloc[0]
view_aneurysm_in_scan(positive_series_uid, localizers_df)




