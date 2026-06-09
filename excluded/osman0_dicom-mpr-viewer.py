# 1. Import Required Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from ipywidgets import interact, IntSlider


# 2. Load DICOM Series from Directory
SERIES_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647'

dicom_files = [os.path.join(SERIES_PATH, f) for f in os.listdir(SERIES_PATH) if f.endswith('.dcm')]
dicoms = [pydicom.dcmread(f) for f in dicom_files]
# InstanceNumber veya SliceLocation
try:
    dicoms.sort(key=lambda x: int(x.InstanceNumber))
except:
    dicoms.sort(key=lambda x: float(getattr(x, 'SliceLocation', 0)))

# Hacimsel veri oluştur
volume = np.stack([d.pixel_array for d in dicoms])


# 3. Visualize Slices of the DICOM Series
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    idx = int(i * len(volume) / 5)
    ax.imshow(volume[idx], cmap='gray')
    ax.set_title(f'Slice {idx}')
    ax.axis('off')
plt.suptitle('Sample Slices from the Series')
plt.show()


# 4. Synchronized MPR Viewer (Axial, Sagittal, Coronal)
from ipywidgets import interactive
 
def show_mpr(slice_idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Axial (XY)
    axes[0].imshow(volume[slice_idx, :, :], cmap='hot') # can change 'gray' also check it colormaps =>  https://matplotlib.org/stable/users/explain/colors/colormaps.html
    axes[0].set_title(f'Axial (Slice {slice_idx})')
    axes[0].axis('off')
    # Sagittal (YZ)
    axes[1].imshow(volume[:, :, slice_idx], cmap='hot')
    axes[1].set_title(f'Sagittal (Slice {slice_idx})')
    axes[1].axis('off')
    # Coronal (XZ)
    axes[2].imshow(volume[:, slice_idx, :], cmap='hot')
    axes[2].set_title(f'Coronal (Slice {slice_idx})')
    axes[2].axis('off')
    plt.suptitle('Synchronized MPR Viewer')
    plt.show()

mpr_slider = IntSlider(min=0, max=volume.shape[0]-1, step=1, value=volume.shape[0]//2, description='Slice')


# interactive(show_mpr, slice_idx=mpr_slider)

