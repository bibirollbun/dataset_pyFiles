import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import pydicom
import nibabel as nib
from glob import glob
import os

pio.renderers.default = "kaggle" 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Visualisation settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
pd.options.display.float_format = '{:,.4f}'.format
plt.rcParams['figure.figsize'] = (18, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14


data_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection'
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
localizers_df = pd.read_csv(os.path.join(data_dir, 'train_localizers.csv'))


print("Train DataFrame Info:")
train_df.info()
print("\n" + "="*50 + "\n")
print("Localizers DataFrame Info:")
localizers_df.info()

print("\nTrain DataFrame Head:")
display(train_df.head())


aneurysm_counts = train_df['Aneurysm Present'].value_counts()

fig = px.pie(
    names=['No Aneurysm', 'Aneurysm Present'],
    values=aneurysm_counts.values,
    title='Class Balance for "Aneurysm Present"',
    hole=0.3,
    color_discrete_sequence=px.colors.sequential.RdBu
)
fig.show()

print(f"Proportion of patients with aneurysm: {aneurysm_counts[1] / len(train_df):.2%}")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(data=train_df, x='PatientAge', hue='Aneurysm Present', kde=True, ax=axes[0])
axes[0].set_title('Age Distribution by Aneurysm Presence')

sns.countplot(data=train_df, x='PatientSex', hue='Aneurysm Present', ax=axes[1])
axes[1].set_title('Sex Distribution by Aneurysm Presence')

plt.tight_layout()
plt.show()


fig = px.histogram(
    train_df,
    x='Modality',
    color='Aneurysm Present',
    barmode='group',
    title='Modality Distribution by Aneurysm Presence',
    text_auto=True
)
fig.show()


location_cols = [col for col in train_df.columns if 'Artery' in col or 'Tip' in col or 'Circulation' in col]
location_counts = train_df[location_cols].sum().sort_values(ascending=False)

fig = px.bar(
    x=location_counts.values,
    y=location_counts.index,
    orientation='h',
    title='Number of Aneurysms by Location',
    labels={'y': 'Location', 'x': 'Count'}
)
fig.show()

# Let's check if one patient has aneurysms in several places.
train_df['num_aneurysms'] = train_df[location_cols].sum(axis=1)
print("Distribution of the number of aneurysms per series:")
print(train_df[train_df['Aneurysm Present'] == 1]['num_aneurysms'].value_counts())


def show_modality_comparison(df, data_dir):
    modalities = df['Modality'].unique()
    
    fig, axes = plt.subplots(1, len(modalities), figsize=(20, 7))
    if len(modalities) == 1:
        axes = [axes]
        
    fig.suptitle('Visual Comparison of Imaging Modalities (Corrected)', fontsize=20)

    for i, mod in enumerate(modalities):
        sample_series_uid = df[df['Modality'] == mod]['SeriesInstanceUID'].iloc[0]
        series_path = os.path.join(data_dir, 'series', sample_series_uid)
        
        dicom_files = glob(os.path.join(series_path, '*.dcm'))
        
        if not dicom_files:
            print(f"No DICOM files found for series {sample_series_uid} (Modality: {mod})")
            continue
            
        dcm_path = dicom_files[0]
        dcm = pydicom.dcmread(dcm_path)
        
        pixel_data = dcm.pixel_array
        
        if pixel_data.ndim == 3:
            mid_frame_idx = pixel_data.shape[0] // 2
            image_to_show = pixel_data[mid_frame_idx, :, :]
            print(f"Modality {mod}: Detected multi-frame DICOM with shape {pixel_data.shape}. Displaying frame {mid_frame_idx}.")
        else:
            image_to_show = pixel_data
            print(f"Modality {mod}: Detected single-frame DICOM with shape {pixel_data.shape}.")

        ax = axes[i]
        ax.imshow(image_to_show, cmap='gray')
        ax.set_title(f'Modality: {mod}')
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

show_modality_comparison(train_df, data_dir)


def load_scan(series_path):
    """
    Downloads DICOM series from a folder.
    It can process both folders with multiple files and a single multiframe file.
    Sorts slices by their position in space.
    """
    dicom_files = glob(os.path.join(series_path, '*.dcm'))
    
    if not dicom_files:
        print(f"No DICOM files found in {series_path}")
        return None, None
        
    if len(dicom_files) == 1:
        dcm = pydicom.dcmread(dicom_files[0])
        if hasattr(dcm, 'NumberOfFrames') and dcm.NumberOfFrames > 1:
            print(f"Detected multi-frame DICOM with {dcm.NumberOfFrames} frames.")
    
            z_positions = []
            if 'PerFrameFunctionalGroupsSequence' in dcm:
                for frame_info in dcm.PerFrameFunctionalGroupsSequence:
                    z_positions.append(frame_info.PlanePositionSequence[0].ImagePositionPatient[2])
            
            pixel_data = dcm.pixel_array
            
            if z_positions and len(z_positions) == pixel_data.shape[0]:
                sorted_indices = np.argsort(z_positions)
                pixel_data = pixel_data[sorted_indices]
            
            return pixel_data, dcm

    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    shape = slices[0].pixel_array.shape
    if not all(s.pixel_array.shape == shape for s in slices):
        print("Warning: Slices have different shapes. This may cause issues.")
        return None, None

    volume = np.stack([s.pixel_array for s in slices])
    
    return volume, slices[0]

series_to_show_uid = train_df['SeriesInstanceUID'].iloc[5]
series_path = os.path.join(data_dir, 'series', series_to_show_uid)

volume, meta = load_scan(series_path)

if volume is not None:
    print(f"Successfully loaded volume with shape: {volume.shape}")
    
    def plot_slice(slice_index):
        plt.figure(figsize=(8, 8))
        plt.imshow(volume[slice_index, :, :], cmap='gray')
        plt.title(f'Series: {series_to_show_uid[:15]}...\nSlice: {slice_index}/{volume.shape[0]-1}')
        plt.axis('off')
        plt.show()

    interact(
        plot_slice,
        slice_index=IntSlider(
            min=0,
            max=volume.shape[0] - 1,
            step=1,
            value=volume.shape[0] // 2,
            description='Slice:',
            continuous_update=False
        )
    );
else:
    print("Could not load the volume for this series.")

if volume is not None:
    mip_axial = np.max(volume, axis=0)
    mip_coronal = np.max(volume, axis=1)
    mip_sagittal = np.max(volume, axis=2)
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Maximum Intensity Projections (MIP)', fontsize=20)
    
    axes[0].imshow(mip_axial, cmap='gray')
    axes[0].set_title('Axial MIP (Top-Down View)')
    axes[1].imshow(mip_coronal, cmap='gray')
    axes[1].set_title('Coronal MIP (Front-Back View)')
    axes[2].imshow(mip_sagittal, cmap='gray')
    axes[2].set_title('Sagittal MIP (Side View)')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()


meta_data = []
for series_uid in train_df['SeriesInstanceUID'].head(200):
    dcm_path = glob(os.path.join(data_dir, 'series', series_uid, '*.dcm'))[0]
    dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    
    meta_data.append({
        'SeriesInstanceUID': series_uid,
        'PixelSpacing_row': dcm.PixelSpacing[0] if 'PixelSpacing' in dcm else None,
        'PixelSpacing_col': dcm.PixelSpacing[1] if 'PixelSpacing' in dcm else None,
        'SliceThickness': dcm.SliceThickness if 'SliceThickness' in dcm else None,
        'RescaleIntercept': dcm.RescaleIntercept if 'RescaleIntercept' in dcm else None,
        'RescaleSlope': dcm.RescaleSlope if 'RescaleSlope' in dcm else None,
    })
meta_df = pd.DataFrame(meta_data).dropna()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.histplot(meta_df['PixelSpacing_row'], kde=True, ax=axes[0, 0]).set_title('Pixel Spacing (X) Distribution')
sns.histplot(meta_df['SliceThickness'], kde=True, ax=axes[0, 1]).set_title('Slice Thickness (Z) Distribution')
sns.histplot(meta_df['RescaleIntercept'], kde=True, ax=axes[1, 0]).set_title('Rescale Intercept Distribution')
sns.histplot(meta_df['RescaleSlope'], kde=True, ax=axes[1, 1]).set_title('Rescale Slope Distribution')
plt.tight_layout()
plt.show()


location_cols = [col for col in train_df.columns if 'Artery' in col or 'Tip' in col or 'Circulation' in col]
is_aneurysm_by_loc = (train_df[location_cols].sum(axis=1) > 0).astype(int)
mismatch = (train_df['Aneurysm Present'] != is_aneurysm_by_loc).sum()

print(f"The number of discrepancies between the 'Aneurysm Present' and the total by location: {mismatch}")


aneurysm_df = train_df.query("`Aneurysm Present` == 1").copy()
aneurysm_df['num_aneurysms'] = aneurysm_df[location_cols].sum(axis=1)

total_loc_counts = aneurysm_df[location_cols].sum()

loc_with_coords_counts = localizers_df['location'].value_counts()

missing_coords_stats = pd.DataFrame({'Total': total_loc_counts, 'WithCoords': loc_with_coords_counts}).fillna(0)
missing_coords_stats['Missing_Percent'] = 1 - (missing_coords_stats['WithCoords'] / missing_coords_stats['Total'])
print("Percentage of missing coordinates by location:")
print(missing_coords_stats.sort_values('Missing_Percent', ascending=False))

