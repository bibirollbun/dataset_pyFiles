import os
import sys
import re
import time
import math
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import nibabel as nib
import pydicom
import SimpleITK as sitk
from scipy.ndimage import zoom
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display_html, display, Markdown, IFrame
from ipywidgets.embed import embed_minimal_html
import warnings
import ast
import cv2
from tqdm.auto import tqdm
from glob import glob
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)

class clr:
    B = '\033[1m'
    L = '\033[1m' + '\033[94m'
    T = '\033[1m' + '\033[91m'
    E = '\033[0m'

print(clr.B+'\n----- Font -----\n'+clr.E)
font_url = 'https://raw.githubusercontent.com/ammomahdavikia/asset-holding/main/carbonplus-regular-bl.otf'
output_path = 'carbonplus-regular-bl.otf'

response = requests.get(font_url, stream=True)

if response.status_code == 200:
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)
    print(f'[Success] downloaded: {output_path}')
else:
    print('[Error] Failed to download the font. Check the URL.')

font_path = '/kaggle/working/carbonplus-regular-bl.otf'
try:
    fm.fontManager.addfont(font_path)
    primary_font = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = [primary_font, "DejaVu Sans", "Noto Sans CJK JP"]
    print('[Success] loaded.')
except:
    print('[Error] Failed to load the font. Check the address.')

COLORS = ['#780000', '#c1121f', '#ffb703', '#02c39a', '#669bbc', '#003049']
modality_palette = {'MR': COLORS[3], 'CT': COLORS[4]}

print(clr.B+'\n----- Color -----\n'+clr.E)
sns.palplot(sns.color_palette(COLORS))


%%capture
!pip install nipreps-synthstrip
!pip install nipreps-synthstrip[nipype]


CONFIG = {
    'PATH': '/kaggle/input/rsna-intracranial-aneurysm-detection',
    'META': '/kaggle/input/rsna2025-iad-metadata',
    'IMG_SIZE': (196, 196),
    'ARTERIES': {
        1: "Other Posterior Circulation",
        2: "Basilar Tip",
        3: "Right Posterior Communicating Artery",
        4: "Left Posterior Communicating Artery",
        5: "Right Infraclinoid Internal Carotid Artery",
        6: "Left Infraclinoid Internal Carotid Artery",
        7: "Right Supraclinoid Internal Carotid Artery",
        8: "Left Supraclinoid Internal Carotid Artery",
        9: "Right Middle Cerebral Artery",
        10: "Left Middle Cerebral Artery",
        11: "Right Anterior Cerebral Artery",
        12: "Left Anterior Cerebral Artery",
        13: "Anterior Communicating Artery"
    },
}

fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)


def load_dicom(dir_path):
    """
    Load DICOM images from a directory and return as a 3D NumPy array.
    Automatically determines the slice orientation (sagittal, coronal, axial) 
    from the first file and sorts slices accordingly.

    Sorting is by InstanceNumber if available, otherwise by ImagePositionPatient[dim].
    Handles both multi-frame and single-slice DICOMs without resizing.
    """
    dcm_files = sorted(glob(os.path.join(dir_path, "*.dcm")))
    if not dcm_files:
        raise ValueError(f"No DICOM files found in {dir_path}")

    first_dcm = pydicom.dcmread(dcm_files[0])
    iop = getattr(first_dcm, "ImageOrientationPatient", None)
    if iop is None or len(iop) < 6:
        raise ValueError("ImageOrientationPatient tag missing or malformed")

    row_cosines = np.array(iop[0:3])
    col_cosines = np.array(iop[3:6])
    normal = np.cross(row_cosines, col_cosines)

    dim = int(np.argmax(np.abs(normal)))
    plane_map = {0: "sagittal", 1: "coronal", 2: "axial"}
    plane = plane_map[dim]

    if len(dcm_files) == 1:
        if hasattr(first_dcm, "NumberOfFrames"):
            return first_dcm.pixel_array
        else:
            return np.expand_dims(first_dcm.pixel_array, axis=0)
    else:
        slices = [pydicom.dcmread(f) for f in dcm_files]
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except (AttributeError, ValueError):
            slices.sort(key=lambda s: float(s.ImagePositionPatient[dim]))

        volume = np.stack([s.pixel_array for s in slices], axis=0)
        return volume

def load_nii(path, mask=False):
    series_id = os.path.basename(os.path.dirname(path))
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    nii_array = nii.get_fdata()  # convert to numpy array

    if mask:
        nii_array = np.clip(nii_array, 0, 13).astype(np.uint8)

    nii_array = nii_array[::-1, :, :]
    nii_array = np.rot90(nii_array, k=1, axes=(0, 1))
    nii_array = np.transpose(nii_array, (2, 0, 1))
    nii_array = np.ascontiguousarray(nii_array)
    
    return nii_array

def plot_image_slices(image_array, save_path, cmap='gray'):

    num_slices = image_array.shape[0]
    
    cols = math.ceil(math.sqrt(num_slices))
    rows = math.ceil(num_slices / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = axes.ravel()

    for i in range(rows*cols):
        ax = axes[i]
        if i < num_slices:
            ax.imshow(image_array[i, :, :], cmap=cmap)
            ax.set_title(i)
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, save_path+'.png'), dpi=300)
    fig.show()

def plot_preprocessing_process(processes):
    fig, axs = plt.subplots(1, len(processes), figsize=(len(processes)*3, 3))

    for idx, (process, array) in enumerate(processes.items()):
        if 'resample' in process.lower():
            if 'mask' in process.lower():
                resample_selected_slice = array.mean(axis=(1, 2)).argmax()
            else:
                resample_selected_slice = array.shape[0] * 2 // 3
        else:
            if 'mask' in process.lower():
                normal_selected_slice = array.mean(axis=(1, 2)).argmax()
            else:
                normal_selected_slice = array.shape[0] * 2 // 3

    for idx, (process, array) in enumerate(processes.items()):
        if 'resample' in process.lower():
            selected_slice = resample_selected_slice
        else:
            selected_slice = normal_selected_slice
        
        cmap='gray'
        if 'mask' in process.lower():
            cmap='jet'
            
        axs[idx].imshow(array[selected_slice, :, :], cmap=cmap)
        axs[idx].axis('off')
        axs[idx].set_title(f'{process}\n{array.shape}')

    fig.tight_layout()
    plt.show()


def load_data(metadata=False):

    if metadata:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        train_df = pd.read_csv(os.path.join(CONFIG['META'], 'train.csv'))
        train_loc = pd.read_csv(os.path.join(CONFIG['META'], 'train_localizers.csv'))
        train_instances = pd.read_csv(os.path.join(CONFIG['META'], 'train_instances.csv'))
        return train_df, train_loc, train_instances
    else:
        train_df = pd.read_csv(os.path.join(CONFIG['PATH'], 'train.csv'))
        train_loc = pd.read_csv(os.path.join(CONFIG['PATH'], 'train_localizers.csv'))
        return train_df, train_loc

def summarize(df, desc='Summary'):

    start = clr.T
    if 'Localization' in desc:
        start = clr.L
    
    print(start+f'\n----- {desc} -----\n'+clr.E)
    print(clr.B+'Shape:'+clr.E, df.shape)
    print(clr.B+'Missing:'+clr.E, df.isna().sum().sum())
    print(clr.B+'Columns:\n'+clr.E, df.columns.to_list())
    display_html(df.head(3))
    print('\n')


train_df, train_loc, train_instances = load_data(metadata=True)

for df, desc in zip([train_df, train_instances, train_loc],
                    ['New Train', 'Instances', 'New Localization']):
    summarize(df, desc)


seg_dir = os.path.join(CONFIG['PATH'], 'segmentations')
seg_series = [re.sub(r'(_cowseg)?\.nii$', '', f) for f in os.listdir(seg_dir)]
seg_df = train_df[train_df.SeriesInstanceUID.isin(seg_series)]
all_ct_series_uids = seg_df[seg_df.Modality=='CTA'].SeriesInstanceUID.tolist()

def summarize_series(series_uid, desc='Summary'):

    print(clr.B+'Chosen series_uid:'+clr.E)
    print(ct_series_uid)
    print(clr.B+'\nAge:'+clr.E)
    print(f'{train_df[train_df.SeriesInstanceUID == series_uid].PatientAge.iloc[0]}')
    print(clr.B+'\nSex:'+clr.E)
    print(f'{train_df[train_df.SeriesInstanceUID == series_uid].PatientSex.iloc[0]}')
    print(clr.B+'\nHas Aneurysm:'+clr.E)
    print(f'{train_df[train_df.SeriesInstanceUID == series_uid]["Aneurysm Present"].iloc[0]}')


ct_series_uid = '1.2.826.0.1.3680043.8.498.52363954882447190271251269039176558430'
#ct_series_uid = np.random.choice(all_ct_series_uids)
summarize_series(ct_series_uid)


ct_series_dir = os.path.join(CONFIG['PATH'], 'series', ct_series_uid)
ct_series_array = load_dicom(ct_series_dir)
print(clr.B+'Image shape:'+clr.E)
print(ct_series_array.shape)


plot_image_slices(ct_series_array, save_path='ct_series_wo_preprocessing')


ct_mask_path = os.path.join(seg_dir, ct_series_uid+'_cowseg.nii')
ct_mask_array = load_nii(ct_mask_path)
print(clr.B+'Mask shape:'+clr.E)
print(ct_mask_array.shape)


plot_image_slices(ct_mask_array, save_path='ct_mask_wo_preprocessing', cmap='jet')


scales = train_instances.groupby(
    ['SeriesInstanceUID', 'Modality']
).agg({
    'RescaleSlope': 'mean',
    'RescaleIntercept': 'mean',
    'RescaleType': 'first'
}).reset_index()


fig, ax = plt.subplots(figsize=(6, 4))

sns.histplot(
    scales,
    x='RescaleType',
    hue='Modality',
    palette=modality_palette,
    ax=ax
)
ax.set_title('RescaleType')
ax.set_xlabel('')

fig.tight_layout()
fig.show()


rscles = ['RescaleIntercept', 'RescaleSlope']

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axs = axs.flatten()

for idx, rscle in enumerate(rscles):
    sns.histplot(
        scales,
        x=rscle,
        hue='Modality',
        palette=modality_palette,
        ax=axs[idx],
        bins=50
    )
    axs[idx].set_title(rscle)
    axs[idx].set_xlabel('')

fig.tight_layout()
fig.show()
fig.savefig('figures/Rescale_distribution.png', dpi=300)


pd.DataFrame(scales[scales.Modality == 'CT'].value_counts(['RescaleSlope', 'RescaleIntercept']))


def apply_hounsfield_scaling(
    array,
    slope,
    intercept
):

    if slope is not None and intercept is not None:
        rescaled_array = array * slope + intercept
    else:
        rescaled_array = array

    return rescaled_array


def apply_hounsfield_range(array, hounsfield_range=(100, 600)):
    mask = (array >= hounsfield_range[0]) & (array <= hounsfield_range[1])
    range_array = np.where(mask, array, 0)
    return range_array


ct_dcm = pydicom.dcmread(glob(os.path.join(ct_series_dir, '*.dcm'))[0])
slope = ct_dcm.get('RescaleSlope')
intercept = ct_dcm.get('RescaleIntercept')
modality = ct_dcm.get('Modality')
assert modality == 'CT', f'series_uid :{ct_series_uid} should be CT.'
ct_rescaled_array = apply_hounsfield_scaling(ct_series_array, slope=slope, intercept=intercept)
ct_range_array = apply_hounsfield_range(ct_rescaled_array, hounsfield_range=(100, 600))

plot_preprocessing_process({
    'Raw': ct_series_array,
    'HU Scaled': ct_range_array,
    'Mask': ct_mask_array
})

print(clr.B+'RescaleSlope:'+clr.E)
print(slope)
print(clr.B+'\nRescaleIntercept:'+clr.E)
print(intercept)
print(clr.L+"\nRaw pixel values (5x5):"+clr.E)
print(ct_series_array[ct_series_array.shape[0]*2//3, :5, :5])
print(clr.L+'Mean:'+clr.E, ct_series_array.mean().round(2), clr.L+'Max:'+clr.E, ct_series_array.max(), clr.L+'Min:'+clr.E, ct_series_array.min())
print(clr.T+"\nRescaled pixel values (HU, 5x5):"+clr.E)
print(ct_rescaled_array[ct_rescaled_array.shape[0]*2//3, :5, :5])
print(clr.T+'Mean:'+clr.E, ct_rescaled_array.mean().round(2), clr.T+'Max:'+clr.E, ct_rescaled_array.max(), clr.T+'Min:'+clr.E, ct_rescaled_array.min())


def plot_hounsfield_interactive(array):
    mid_slice = array.shape[0] * 2 // 3

    hu_min_slider = widgets.IntSlider(value=100, min=-1000, max=3000, step=10, description='HU Min')
    hu_max_slider = widgets.IntSlider(value=600, min=-1000, max=3000, step=10, description='HU Max')

    out = widgets.Output()

    def update(hu_min, hu_max):
        mask = (array >= hu_min) & (array <= hu_max)
        filtered_array = np.where(mask, array, 0)
        
        with out:
            out.clear_output(wait=True)
            fig, axs = plt.subplots(1, 3, figsize=(8, 4))
            
            axs[0].imshow(array[mid_slice], cmap='gray')
            axs[0].axis('off')
            axs[0].set_title("Raw HU")
            
            axs[1].imshow(filtered_array[mid_slice], cmap='gray')
            axs[1].axis('off')
            axs[1].set_title(f"Filtered {hu_min}–{hu_max} HU")
            
            axs[2].imshow(ct_mask_array[mid_slice], cmap='jet')
            axs[2].axis('off')
            axs[2].set_title("Segmentation")
            
            plt.tight_layout()
            plt.show()

    widgets.interactive(update, hu_min=hu_min_slider, hu_max=hu_max_slider)

    update(hu_min_slider.value, hu_max_slider.value)

    ui = widgets.VBox([hu_min_slider, hu_max_slider, out])
    
    embed_minimal_html('hu_widget.html', views=[ui], title='HU Windowing')
    display(IFrame(src='hu_widget.html', width=900, height=600))


plot_hounsfield_interactive(ct_rescaled_array)


plot_image_slices(ct_range_array, save_path='ct_series_w_hounsfield_scaling')


def apply_normalization(array, norm='z-score'):
    if norm == 'min-max':
        return (array - array.min()) / (array.max() - array.min() + 1e-8)
    elif norm == 'z-score':
        return (array - array.mean()) / array.std()


ct_z_normalized_array = apply_normalization(ct_range_array, 'z-score')


plot_preprocessing_process({
    'HU Scaled': ct_range_array,
    'Z-score Normalized': ct_z_normalized_array,
    'Mask': ct_mask_array
})


ct_mm_normalized_array = apply_normalization(ct_range_array, 'min-max')


plot_preprocessing_process({
    'HU Scaled': ct_range_array,
    'Min-Max Normalized': ct_mm_normalized_array,
    'Mask': ct_mask_array
})


pixels = train_instances.groupby(['SeriesInstanceUID', 'Modality'])[['PixelSpacing_X', 'PixelSpacing_Y']].mean().reset_index()

g = sns.jointplot(
    data=pixels, 
    x="PixelSpacing_X", 
    y="PixelSpacing_Y",
    hue='Modality',
    kind='kde',
    palette=modality_palette,
    height=6
)

g.figure.savefig('figures/pixel_spacing_distribution.png', dpi=300)
plt.show()


slices = train_instances.groupby(['SeriesInstanceUID', 'Modality'])[['SliceThickness', 'SpacingBetweenSlices']].mean().reset_index()

g = sns.jointplot(
    data=slices, 
    x="SliceThickness", 
    y="SpacingBetweenSlices",
    hue='Modality',
    palette=modality_palette,
    kind='kde',
    height=6
)

g.figure.savefig('figures/slice_dimensions_distribution.png', dpi=300)
plt.show()


def apply_resampling(
    array, 
    pixel_spacing_x,
    pixel_spacing_y,
    pixel_spacing_z,
    target_spacing=(1.0, 1.0, 1.0),
    is_mask=False
):
    """
    Resample a 3D array to target spacing.

    Parameters
    ----------
    array : np.ndarray
        Input array with shape (Z, H, W)
    pixel_spacing_x, pixel_spacing_y, pixel_spacing_z : float
        Original voxel spacing in mm
    target_spacing : tuple of float
        Desired spacing (x, y, z) in mm
    is_mask : bool
        If True, use nearest neighbor interpolation to preserve labels

    Returns
    -------
    resampled_array : np.ndarray
        Resampled array with same type as input
    """
    # Transpose to (W, H, Z) for scipy zoom
    array = np.transpose(array, (2, 1, 0))
    pixdim = [pixel_spacing_x, pixel_spacing_y, pixel_spacing_z]
    zoom_factors = tuple(pixdim[i] / target_spacing[i] for i in range(3))

    order = 0 if is_mask else 3  # nearest neighbor for mask, cubic for image

    resampled_array = zoom(array, zoom_factors, order=order)
    # Transpose back to original (Z, H, W)
    resampled_array = np.transpose(resampled_array, (2, 1, 0))

    if is_mask:
        resampled_array = resampled_array.astype(array.dtype)

    return resampled_array


ct_pixel_spacing = ct_dcm.get('PixelSpacing')
ct_slice_thickness = ct_dcm.get('SliceThickness')
ct_spacing_slices = ct_dcm.get('SpacingBetweenSlices')

print(clr.B+'PixelSpacing:'+clr.E)
print(ct_pixel_spacing)
print(clr.B+'\nSliceThickness:'+clr.E)
print(ct_slice_thickness)
print(clr.B+'\nSpacingBetweenSlices:'+clr.E)
print(ct_spacing_slices)


ct_resampled_array = apply_resampling(
    ct_z_normalized_array,
    pixel_spacing_x=ct_pixel_spacing[0],
    pixel_spacing_y=ct_pixel_spacing[1],
    pixel_spacing_z=ct_spacing_slices if ct_spacing_slices else ct_slice_thickness,
    target_spacing=(1.0, 1.0, 1.0)
)

ct_resampled_mask = apply_resampling(
    ct_mask_array,
    pixel_spacing_x=ct_pixel_spacing[0],
    pixel_spacing_y=ct_pixel_spacing[1],
    pixel_spacing_z=ct_spacing_slices if ct_spacing_slices else ct_slice_thickness,
    target_spacing=(1.0, 1.0, 1.0),
    is_mask=True
)

plot_preprocessing_process({
    'Z-score Normalized': ct_z_normalized_array, 
    'Resampled': ct_resampled_array, 
    'Mask': ct_mask_array,
    'Resampled Mask': ct_resampled_mask
})

print(clr.L+"Raw array:"+clr.E)
print(clr.L+'Shape:'+clr.E, ct_z_normalized_array.shape)
print(clr.L+'Mean:'+clr.E, ct_z_normalized_array.mean().round(2), clr.L+'Max:'+clr.E, ct_z_normalized_array.max().round(2), clr.L+'Min:'+clr.E, ct_z_normalized_array.min().round(2))
print(clr.T+"\nResampled array:"+clr.E)
print(clr.T+'Shape:'+clr.E, ct_resampled_array.shape)
print(clr.T+'Mean:'+clr.E, ct_resampled_array.mean().round(2), clr.T+'Max:'+clr.E, ct_resampled_array.max().round(2), clr.T+'Min:'+clr.E, ct_resampled_array.min().round(2))


plot_image_slices(ct_resampled_array, 'ct_series_w_resampling')


plot_preprocessing_process({
    'Raw': ct_series_array,
    'HU scaled': ct_range_array,
    'Normalized': ct_z_normalized_array,
    'Resampled': ct_resampled_array,
    'Mask': ct_mask_array,
    'Mask Resampled': ct_resampled_mask
})


all_mr_series_uids = seg_df[seg_df.Modality!='CTA'].SeriesInstanceUID.tolist()


mr_series_uid = '1.2.826.0.1.3680043.8.498.92498800238576582506105430510381134234'
#mr_series_uid = np.random.choice(all_mr_series_uids)
summarize_series(mr_series_uid)


mr_series_dir = os.path.join(CONFIG['PATH'], 'series', mr_series_uid)
mr_series_array = load_dicom(mr_series_dir)
print(clr.B+'Image shape:'+clr.E)
print(mr_series_array.shape)


plot_image_slices(mr_series_array, save_path='mr_series_wo_preprocessing')


mr_mask_path = os.path.join(seg_dir, mr_series_uid+'_cowseg.nii')
mr_mask_array = load_nii(mr_mask_path)
print(clr.B+'Mask shape:'+clr.E)
print(mr_mask_array.shape)


plot_image_slices(mr_mask_array, save_path='mr_mask_wo_preprocessing', cmap='jet')


def apply_skull_stripping(series_uid):
    
    series_nii_path = os.path.join(seg_dir, series_uid+'.nii')
    ss_output_path = f"{mr_series_uid}_output.nii.gz"
    ss_mask_path = f"{mr_series_uid}_mask.nii.gz"

    from nipreps.synthstrip.cli import main
    
    sys.argv = [
        "synthstrip",
        "--image", series_nii_path,
        "--out", ss_output_path,
        "--mask", ss_mask_path,
        "--model", "/kaggle/input/synthstrip/pytorch/main/1/synthstrip.1.pt",
    ]
    
    main()

    ss_output_array = load_nii(ss_output_path)
    ss_mask_array = load_nii(ss_mask_path)
    ss_array = ss_mask_array * mr_series_array

    return ss_array, ss_mask_array


mr_ss_array, mr_ss_mask_array = apply_skull_stripping(mr_series_uid)


plot_preprocessing_process({
    'Raw': mr_series_array, 
    'Skull-Stripped': mr_ss_array, 
    'Brain Mask': mr_ss_mask_array, 
    'Mask': mr_mask_array
})


plot_image_slices(mr_ss_array, save_path='mr_mask_w_skull_stripping')


def apply_n4_correction(array):
    image_sitk = sitk.GetImageFromArray(array)
    image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
    
    # Create mask using Otsu's method
    mask = sitk.OtsuThreshold(image_sitk, 0, 1, 200)
    
    # Set up N4 Bias Field Correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 30, 20])
    
    start_time = time.time()
    corrected_sitk = corrector.Execute(image_sitk, mask)
    elapsed_time = time.time() - start_time
    
    corrected_array = sitk.GetArrayFromImage(corrected_sitk)
    
    print(clr.B+f"N4 Bias Field Correction took {elapsed_time:.2f} seconds"+clr.E)

    log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
    bias_field = sitk.Exp(log_bias_field)  # multiplicative bias map
    bias_array = sitk.GetArrayFromImage(bias_field)

    return corrected_array, bias_array


mr_corrected_array, mr_bias_array = apply_n4_correction(mr_ss_array)


plot_preprocessing_process({
    'Skull-Stripped': mr_ss_array, 
    'N4 Bias-Field Corrected': mr_corrected_array, 
    'Bias Map Mask': mr_bias_array
})


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

sns.histplot(mr_ss_array[mr_ss_array>0].flatten(), color=COLORS[4], ax=axs[0])
axs[0].set_title('Before')
sns.histplot(mr_corrected_array[mr_corrected_array>0].flatten(), color=COLORS[4], ax=axs[1])
axs[1].set_title('After')
axs[1].set_ylabel('')

fig.suptitle('N4 Bias-Field Correction\nPixelValue Distribution', fontsize=16)
fig.tight_layout()
fig.show()
fig.savefig('figures/n4_bias_field_correction.png', dpi=300)


mr_z_normalized_array = apply_normalization(mr_corrected_array, 'z-score')


plot_preprocessing_process({
    'N4 Bias-Field Corrected': mr_corrected_array,
    'Z-score Normalized': mr_z_normalized_array,
    'Mask': mr_mask_array
})


mr_mm_normalized_array = apply_normalization(mr_corrected_array, 'min-max')


plot_preprocessing_process({
    'N4 Bias-Field Corrected': mr_corrected_array,
    'Min-Max Normalized': mr_mm_normalized_array,
    'Mask': mr_mask_array
})


mr_dcm = pydicom.dcmread(glob(os.path.join(mr_series_dir, '*.dcm'))[0])
mr_pixel_spacing = mr_dcm.get('PixelSpacing')
mr_slice_thickness = mr_dcm.get('SliceThickness')
mr_spacing_slices = mr_dcm.get('SpacingBetweenSlices')
print(clr.B+'PixelSpacing:'+clr.E)
print(mr_pixel_spacing)
print(clr.B+'\nSliceThickness:'+clr.E)
print(mr_slice_thickness)
print(clr.B+'\nSpacingBetweenSlices:'+clr.E)
print(mr_spacing_slices)


mr_resampled_array = apply_resampling(
    mr_z_normalized_array,
    pixel_spacing_x=mr_pixel_spacing[0],
    pixel_spacing_y=mr_pixel_spacing[1],
    pixel_spacing_z=mr_spacing_slices if mr_spacing_slices else mr_slice_thickness,
    target_spacing=(1.0, 1.0, 1.0),
)

mr_resampled_mask = apply_resampling(
    mr_mask_array,
    pixel_spacing_x=mr_pixel_spacing[0],
    pixel_spacing_y=mr_pixel_spacing[1],
    pixel_spacing_z=mr_spacing_slices if mr_spacing_slices else mr_slice_thickness,
    target_spacing=(1.0, 1.0, 1.0),
    is_mask=True
)

plot_preprocessing_process({
    'Z-score Normalized': mr_z_normalized_array, 
    'Resampled': mr_resampled_array, 
    'Resampled Mask': mr_resampled_mask
})

print(clr.L+"Raw array:"+clr.E)
print(clr.L+'Shape:'+clr.E, mr_z_normalized_array.shape)
print(clr.L+'Mean:'+clr.E, mr_z_normalized_array.mean().round(2), clr.L+'Max:'+clr.E, mr_z_normalized_array.max().round(2), clr.L+'Min:'+clr.E, mr_z_normalized_array.min().round(2))
print(clr.T+"\nResampled array:"+clr.E)
print(clr.T+'Shape:'+clr.E, mr_resampled_array.shape)
print(clr.T+'Mean:'+clr.E, mr_resampled_array.mean().round(2), clr.T+'Max:'+clr.E, mr_resampled_array.max().round(2), clr.T+'Min:'+clr.E, mr_resampled_array.min().round(2))


plot_image_slices(mr_resampled_array, 'mr_series_w_resampling')


plot_preprocessing_process(
    {'Raw': mr_series_array,
     'Non-Skull mask': mr_ss_mask_array,
     'Skull-Stripped': mr_ss_array,
     'N4 Bias-Field': mr_corrected_array,
     'Normalized': mr_z_normalized_array,
     'Resampled': mr_resampled_array}
)


plot_preprocessing_process(
    {'Mask': mr_mask_array,
     'Mask Resampled': mr_resampled_mask}
)

