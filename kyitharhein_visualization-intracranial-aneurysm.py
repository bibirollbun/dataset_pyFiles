import os
import json
import glob
import warnings
import pydicom
import seaborn as sns
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import sample
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")


# Define paths
train_csv_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv'     
localizers_csv_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv'
series_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'                

# Load train.csv as df
df = pd.read_csv(train_csv_path)
# Load train_localizers.csv as loc_df
loc_df = pd.read_csv(localizers_csv_path)


# Filter for series where Aneurysm Present is 1
df_aneurysm = df[df['Aneurysm Present'] == 1]

# Get unique SeriesInstanceUIDs from loc_df (series with localization data)
loc_series = loc_df['SeriesInstanceUID'].unique()

# Filter df_aneurysm to include only SeriesInstanceUIDs present in loc_df
df_aneurysm_loc = df_aneurysm[df_aneurysm['SeriesInstanceUID'].isin(loc_series)]

# Extract relevant columns: SeriesInstanceUID and Modality
modality_map = df_aneurysm_loc[['SeriesInstanceUID', 'Modality']].drop_duplicates()

# Initialize a dictionary to store modality to SeriesInstanceUID mappings
modality_series = {}

# Iterate through each unique SeriesInstanceUID and its modality
for _, row in modality_map.iterrows():
    series_uid = row['SeriesInstanceUID']
    modality = row['Modality']
    
    # Store the SeriesInstanceUID for this modality
    if modality not in modality_series:
        modality_series[modality] = []
    modality_series[modality].append(series_uid)

# Prepare data for plotting
modality_counts = {modality: len(uids) for modality, uids in modality_series.items()}
modality_df = pd.DataFrame(list(modality_counts.items()), columns=['Modality', 'Count'])

# Plot the number of series per modality
plt.figure(figsize=(10, 6))
g = sns.barplot(x='Modality', y='Count', data=modality_df, palette='viridis')
plt.title('Number of Series with Aneurysm and Location per Modality')
plt.xlabel('Modality')
plt.ylabel('Number of Series')
plt.xticks(rotation=45)

# Add count labels on top of each bar
for p in g.patches:
    g.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.show()


merged_df = loc_df.merge(df[['SeriesInstanceUID', 'Modality', 'Aneurysm Present']], on='SeriesInstanceUID', how='inner')
merged_df = merged_df[merged_df['Aneurysm Present'] == 1]

# Group by Modality and location to count occurrences
label_modality_counts = merged_df.groupby(['Modality', 'location']).size().reset_index(name='Count')

# Plot as a grouped bar plot
plt.figure(figsize=(12, 8))
g = sns.barplot(x='location', y='Count', hue='Modality', data=label_modality_counts, palette='Set2')
plt.title('Number of Aneurysms per Location and Modality')
plt.xlabel('Aneurysm Location')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Modality')

# Add count labels on top of each bar
for p in g.patches:
    height = p.get_height()
    if height > 0:  # Only annotate non-zero bars
        g.annotate(f'{height:.0f}', 
                   (p.get_x() + p.get_width() / 2., height), 
                   ha='center', va='center', 
                   xytext=(0, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.show()


CMAP = 'gray'

def plot_labeled_slice(row_number, localizers_df, train_df, ax_image, ax_label, ax_overlay):
    row = localizers_df.iloc[row_number]
    series_uid = row['SeriesInstanceUID']
    sop_uid = row['SOPInstanceUID']
    coordinates = row['coordinates']
    location = row['location']
    
    # Get Modality from train_df
    modality = train_df[train_df['SeriesInstanceUID'] == series_uid]['Modality'].iloc[0] if not train_df[train_df['SeriesInstanceUID'] == series_uid].empty else 'Unknown'
    
    # Find the DICOM file for the SOPInstanceUID
    folders = glob.glob("../input/rsna-intracranial-aneurysm-detection/series/*")
    series_path = None
    for folder in folders:
        if series_uid in folder:
            series_path = folder
            break
    
    if not series_path:
        print(f"No folder found for SeriesInstanceUID: {series_uid}")
        return
    
    # Get all DICOM files in the series
    files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
    dicom_file = None
    for file in files:
        if sop_uid in file:
            dicom_file = file
            break
    
    if not dicom_file:
        print(f"No DICOM file found for SOPInstanceUID: {sop_uid}")
        return
    
    # Load DICOM image
    ex = pydicom.dcmread(dicom_file)
    if len(ex.pixel_array.shape) > 2:
        print(f"Skipping {sop_uid}: Image has more than 2 dimensions")
        return
    
    img = ex.pixel_array
    img = np.flipud(img)  # Flip vertically to match DICOM top-left origin
    height, width = img.shape
    
    # Parse coordinates
    coordinates = json.loads(coordinates.replace("'", '"'))
    x = coordinates["x"]
    y = height - 1 - coordinates["y"]  # Adjust for flipud
    
    # Plot Image
    ax_image.imshow(img, cmap=CMAP, origin='upper', aspect=1)  # Use gray for DICOM
    ax_image.set_title(f'Image\nSeries: {series_uid[:8]}...')
    ax_image.axis('off')
    
    # Plot Label (black background with red rectangle and text)
    ax_label.imshow(np.zeros_like(img), cmap=CMAP, origin='upper', aspect=1)  # Black background
    rect = patches.Rectangle((x-10, y-10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax_label.add_patch(rect)
    ax_label.text(x, y+35, location, color='yellow', fontsize=8, ha='center', va='bottom')
    ax_label.text(x, y+55, f'Modality: {modality}', color='red', fontsize=8, ha='center', va='bottom')
    ax_label.set_title('Label')
    ax_label.axis('off')
    
    # Plot Overlay
    ax_overlay.imshow(img, cmap=CMAP, origin='upper', aspect=1)  # Use gray for DICOM
    rect = patches.Rectangle((x-10, y-10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax_overlay.add_patch(rect)
    ax_overlay.text(x, y+35, location, color='yellow', fontsize=8, ha='center', va='bottom')
    ax_overlay.text(x, y+55, f'Modality: {modality}', color='red', fontsize=8, ha='center', va='bottom')
    ax_overlay.set_title('Overlay')
    ax_overlay.axis('off')

# Function to plot for a given modality, optional label (location), and optional row indices
def plot_modality_slices(modality, label=None, row_indices=None):
    # Filter for series with Aneurysm Present and localization data
    df_aneurysm = df[df['Aneurysm Present'] == 1]
    loc_series = loc_df['SeriesInstanceUID'].unique()
    df_aneurysm_loc = df_aneurysm[df_aneurysm['SeriesInstanceUID'].isin(loc_series)]
    df_modality = df_aneurysm_loc[df_aneurysm_loc['Modality'] == modality]
    
    # Get sorted unique SeriesInstanceUID for the modality
    modality_series_sorted = sorted(df_modality['SeriesInstanceUID'].unique())
    
    # Select rows from loc_df for the modality
    modality_loc_rows = loc_df[loc_df['SeriesInstanceUID'].isin(modality_series_sorted)]
    
    # Optionally filter by label (aneurysm location)
    if label is not None:
        modality_loc_rows = modality_loc_rows[modality_loc_rows['location'] == label]
    
    # If specific row indices are provided, try to use them
    selected_rows = []
    if row_indices is not None:
        for idx in row_indices:
            if idx < len(modality_loc_rows) and modality_loc_rows.iloc[idx]['SeriesInstanceUID'] in modality_series_sorted:
                selected_rows.append(modality_loc_rows.index[idx])  # Use global index from loc_df
    
    # If fewer than 5 rows selected (or none), fall back to first 5 sorted series
    if len(selected_rows) < 5:
        if row_indices is not None:
            print(f"Requested indices {row_indices} invalid or insufficient. Selecting first {5 - len(selected_rows)} sorted series.")
        remaining_series = [s for s in modality_series_sorted if s not in [loc_df.iloc[idx]['SeriesInstanceUID'] for idx in selected_rows]]
        for series in remaining_series[:5 - len(selected_rows)]:
            series_rows = modality_loc_rows[modality_loc_rows['SeriesInstanceUID'] == series]
            if not series_rows.empty:
                selected_rows.append(series_rows.index[0])  # Pick first row for the series
    
    # If fewer than 5 series found, warn
    if len(selected_rows) < 5:
        print(f"Only {len(selected_rows)} {modality} series with aneurysms and localization data found (label: {label}).")
    
    # Create subplot grid (adjust rows based on available series)
    fig, axs = plt.subplots(len(selected_rows), 3, figsize=(15, 5 * len(selected_rows)))
    
    # Handle case of single row (2D array to 1D)
    if len(selected_rows) == 1:
        axs = [axs]
    
    # Plot for each selected row
    for i, row_number in enumerate(selected_rows):
        print(f"\nDisplaying labeled slice for series {loc_df.iloc[row_number]['SeriesInstanceUID'][:8]}... (row {row_number})")
        plot_labeled_slice(row_number, loc_df, df, axs[i][0], axs[i][1], axs[i][2])

    plt.tight_layout()
    plt.show()



plot_modality_slices(modality='CTA', label='Left Middle Cerebral Artery')


plot_modality_slices(modality='CTA')


plot_modality_slices(modality='MRA')


plot_modality_slices(modality='MRI T2', row_indices=[21, 40, 48, 49])


plot_modality_slices(modality='MRI T1post')


df.Modality.unique()




