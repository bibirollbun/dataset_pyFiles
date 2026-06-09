


import os
import requests
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import nibabel as nib
import pydicom
import ipywidgets as widgets
from IPython.display import display_html, display, Markdown
import warnings
import ast
import cv2
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from tqdm.auto import tqdm
from glob import glob
from pathlib import Path
import plotly.io as pio
pio.renderers.default = 'iframe'
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
aneurysm_palette = {1: COLORS[1], 0: COLORS[-1]}
modality_palette = {'CTA': COLORS[4], 'MRA': COLORS[3],
                    'MRI T2': COLORS[2], 'MRI T1post': COLORS[1]}
plane_palette = {'axial': COLORS[0], 'coronal': COLORS[2], 'sagittal': COLORS[-1]}
print(clr.B+'\n----- Color -----\n'+clr.E)
sns.palplot(sns.color_palette(COLORS))


series_cols = ['SeriesInstanceUID', 'FrameOfReferenceUID', 'SOPClassUID', 'IsMultiFrame', 'Rows', 'Columns', 'NumberOfFrames', 'Plane']
instance_cols = ['SeriesInstanceUID', 'FrameOfReferenceUID', 'SOPClassUID', 'SOPInstanceUID', 'FilePath',
                 'Modality', 'Columns', 'Rows', 'PixelRepresentation', 'WindowWidth', 'WindowCenter',
                 'BitsAllocated', 'BitsStored', 'HighBit', 'RescaleIntercept', 'RescaleSlope', 'RescaleType',
                 'SliceThickness', 'SpacingBetweenSlices', 'PixelSpacing_X', 'PixelSpacing_Y',
                 'IPP_X', 'IPP_Y', 'IPP_Z', 'IOP_RowX', 'IOP_RowY', 'IOP_RowZ', 'IOP_ColX', 'IOP_ColY', 'IOP_ColZ',
                 'Plane', 'InstanceNumber']


def normalize_value(value):
    if isinstance(value, pydicom.multival.MultiValue):
        return list(value)
    elif isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.DSdecimal)):
        return float(value)
    elif isinstance(value, pydicom.valuerep.IS):
        return int(value)
    return value

def get_series_metadata(dataset):
    tags = ['SeriesInstanceUID', 'FrameOfReferenceUID', 
            'SOPClassUID', 'Rows', 'Columns']
    metadata = {tag: [normalize_value(dataset.get(tag))] for tag in tags}
    
    num_frames = getattr(dataset, 'NumberOfFrames', None)
    metadata["IsMultiFrame"] = [num_frames is not None and num_frames > 1]
    
    return metadata

def get_singleframe_instance_metadata(dataset, file_path):
    tags = ['BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 
            'HighBit', 'Modality', 'WindowWidth', 'WindowCenter',
            'PixelRepresentation', 'SeriesInstanceUID',
            'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 
            'SOPClassUID', 'SliceThickness', 
            'SpacingBetweenSlices']

    metadata = {}
    for tag in tags:
        metadata[tag] = normalize_value(dataset.get(tag))

    metadata["SOPInstanceUID"] = dataset.get("SOPInstanceUID")
    metadata["FilePath"] = file_path

    pi = dataset.get("PixelSpacing")
    if pi and len(pi) == 2:
        pi = list(map(float, pi))
        metadata["PixelSpacing_X"], metadata["PixelSpacing_Y"] = pi
    else:
        metadata["PixelSpacing_X"], metadata["PixelSpacing_Y"] = np.nan
    
    # ImagePositionPatient into X, Y, Z
    ipp = dataset.get("ImagePositionPatient")
    if ipp and len(ipp) == 3:
        ipp = list(map(float, ipp))
        metadata["IPP_X"], metadata["IPP_Y"], metadata["IPP_Z"] = ipp
    else:
        metadata["IPP_X"] = metadata["IPP_Y"] = metadata["IPP_Z"] = np.nan

    # ImageOrientationPatient into Row/Col direction cosines
    iop = dataset.get("ImageOrientationPatient")
    if iop and len(iop) == 6:
        iop = list(map(float, iop))
        metadata["IOP_RowX"], metadata["IOP_RowY"], metadata["IOP_RowZ"] = iop[:3]
        metadata["IOP_ColX"], metadata["IOP_ColY"], metadata["IOP_ColZ"] = iop[3:]
    else:
        for k in ["IOP_RowX", "IOP_RowY", "IOP_RowZ", "IOP_ColX", "IOP_ColY", "IOP_ColZ"]:
            metadata[k] = np.nan

    return metadata

def get_multiframe_instance_metadata(dataset, file_path):
    shared = dataset.SharedFunctionalGroupsSequence[0]
    per_frame = dataset.PerFrameFunctionalGroupsSequence

    # Shared attributes
    iop = shared.PlaneOrientationSequence[0].ImageOrientationPatient
    ps = shared.PixelMeasuresSequence[0].PixelSpacing
    st = shared.PixelMeasuresSequence[0].SliceThickness
    sbs = getattr(shared.PixelMeasuresSequence[0], 'SpacingBetweenSlices', None)

    shared_fields = {
        'IOP_RowX': float(iop[0]), 'IOP_RowY': float(iop[1]), 'IOP_RowZ': float(iop[2]),
        'IOP_ColX': float(iop[3]), 'IOP_ColY': float(iop[4]), 'IOP_ColZ': float(iop[5]),
        'PixelSpacing_X': float(ps[0]), 'PixelSpacing_Y': float(ps[1]),
        'SliceThickness': float(st),
        'SpacingBetweenSlices': float(sbs) if sbs is not None else None,
    }

    tags = ['BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 
            'HighBit', 'Modality', 'WindowWidth', 'WindowCenter',
            'PixelRepresentation', 'SeriesInstanceUID',
            'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 
            'SOPClassUID']

    for tag in tags:
        shared_fields[tag] = normalize_value(dataset.get(tag))

    metadata = []

    for frame_index, frame in enumerate(per_frame):
        ipp = frame.PlanePositionSequence[0].ImagePositionPatient
        ipp = list(map(float, ipp))

        instance = {
            "SOPInstanceUID": dataset.SOPInstanceUID,  # same for all frames
            "FilePath": file_path,
            "IPP_X": ipp[0], "IPP_Y": ipp[1], "IPP_Z": ipp[2],
            **shared_fields
        }
        metadata.append(instance)

    return metadata

def process_series(series_dir):
    dcm_files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.lower().endswith('.dcm')]
    instance_records = []
    ds_first = None

    for file in dcm_files:
        try:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            if ds_first is None:
                ds_first = ds
                
            num_frames = getattr(ds, 'NumberOfFrames', None)
            if isinstance(num_frames, (int, float)) and num_frames > 1:
                multi_instances = get_multiframe_instance_metadata(ds, file)
                instance_records.extend(multi_instances)
                continue

            metadata = get_singleframe_instance_metadata(ds, file)
            instance_records.append(metadata)

        except Exception as e:
            print(f"[ERROR] Could not read {file}: {e}")

    for i, rec in enumerate(instance_records):
        row_cos = np.array([rec['IOP_RowX'], rec['IOP_RowY'], rec['IOP_RowZ']])
        col_cos = np.array([rec['IOP_ColX'], rec['IOP_ColY'], rec['IOP_ColZ']])
        normal = np.cross(row_cos, col_cos)
        axis = np.abs(normal)
        labels = ['sagittal', 'coronal', 'axial']
        directions = ['X', 'Y', 'Z']
        rec['Plane'] = labels[int(np.argmax(axis))]
        slice_direction = directions[int(np.argmax(axis))]
    
    instance_records = sorted(instance_records, key=lambda x: x.get(f"IPP_{slice_direction}", float('inf')))

    for i, rec in enumerate(instance_records):
        rec["InstanceNumber"] = i + 1

    if ds_first:
        series_meta = get_series_metadata(ds_first)
        series_meta['Plane'] = instance_records[0]['Plane']
        series_meta['NumberOfFrames'] = len(instance_records)
    else:
        series_meta = {}

    instance_records = pd.DataFrame(instance_records)[instance_cols]
    series_meta = pd.DataFrame(series_meta)[series_cols]
    
    return instance_records, series_meta

def extract_dicom_metadata(root_dir):
    """
    Traverse all series directories in root_dir, extract instance and series metadata,
    and return concatenated DataFrames.
    
    Returns:
        all_instances_df: pd.DataFrame with instance-level metadata
        all_series_df: pd.DataFrame with series-level metadata
    """
    all_instances = []
    all_series = []

    series_dirs = [
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for idx, series_dir in enumerate(tqdm(series_dirs, desc='Extract DICOM Metadata')):
        try:
            instance_df, series_df = process_series(series_dir)

            all_instances.append(instance_df)
            all_series.append(series_df)
        except Exception as e:
            print(f"[ERROR] Failed to process {series_dir}: {e}")

    if all_instances:
        all_instances_df = pd.concat(all_instances, ignore_index=True)
    else:
        all_instances_df = pd.DataFrame(columns=instance_cols)

    if all_series:
        all_series_df = pd.concat(all_series, ignore_index=True)
    else:
        all_series_df = pd.DataFrame(columns=series_cols)

    return all_instances_df, all_series_df

def load_dicom(dir_path, target_shape=(512, 512)):
    """
    Load DICOM images from a directory and return as a 3D NumPy array.
    
    Handles both multi-frame and single-slice DICOMs. Resizes each slice to target_shape.
    """
    dcm_files = sorted(glob(os.path.join(dir_path, "*.dcm")))

    if len(dcm_files) == 1:
        dcm = pydicom.dcmread(dcm_files[0])
        if hasattr(dcm, 'NumberOfFrames'):
            frames = dcm.pixel_array
            resized = np.stack([cv2.resize(f, target_shape, interpolation=cv2.INTER_LINEAR)
                                for f in frames], axis=0)
            return resized
        else:
            single = dcm.pixel_array
            resized = cv2.resize(single, target_shape, interpolation=cv2.INTER_LINEAR)
            return np.expand_dims(dcm.pixel_array, 0)
    else:
        slices = [pydicom.dcmread(f) for f in dcm_files]
        slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
        resized_slices = [cv2.resize(s.pixel_array, target_shape, interpolation=cv2.INTER_LINEAR)
                          for s in slices]
        volume = np.stack(resized_slices, axis=0)
        return volume

def load_nii(path, mask=False):
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    nii_array = nii.get_fdata()
    
    pixdim = nii.header.get_zooms()[:3]
    target_spacing = (1.0, 1.0, 1.0)
    zoom_factors = tuple(pixdim[i] / target_spacing[i] for i in range(3))
    nii_array = zoom(nii_array, zoom_factors, order = 0 if mask else 1)
    
    if mask:
        nii_array = np.clip(nii_array, 0, 13).astype(np.uint8)

    nii_array = np.rot90(nii_array, k=1, axes=(0, 1))
    nii_array = np.ascontiguousarray(nii_array)
    
    return nii_array


CONFIG = {
    'PATH': '/kaggle/input/rsna-intracranial-aneurysm-detection',
    'META': '/kaggle/input/rsna2025-iad-metadata',
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

os.makedirs('Figures', exist_ok=True)


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
    print(f'Shape: {df.shape}')
    print(f'Missing: {df.isna().sum().sum()}')
    print(f'Columns: {df.columns.to_list()}\n')
    display_html(df.head(3))
    print('\n')


train_df, train_loc = load_data()

for df, desc in zip([train_df, train_loc],
                    ['Train', 'Localization']):
    summarize(df, desc)


def process_data(train_df, train_loc):
    
    root_dir = os.path.join(CONFIG['PATH'], 'series')
    instances_df, series_df = extract_dicom_metadata(root_dir)

    train_df = pd.merge(train_df, series_df, on='SeriesInstanceUID', how='left')

    coords = train_loc['coordinates'].apply(ast.literal_eval)
    coords_df = pd.DataFrame(coords.tolist(), index=train_loc.index)
    train_loc = train_loc.copy()
    train_loc.loc[:, 'x'] = coords_df['x']
    train_loc.loc[:, 'y'] = coords_df['y']

    shape_df = train_df[['SeriesInstanceUID', 'Rows', 'Columns']].copy()
    train_loc = pd.merge(train_loc, shape_df, on='SeriesInstanceUID', how='left')

    train_loc['x_norm'] = train_loc['x'] / train_loc['Columns']
    train_loc['y_norm'] = train_loc['y'] / train_loc['Rows']
    
    train_df.to_csv('train.csv', index=False)
    train_loc.to_csv('train_localizers.csv', index=False)
    instances_df.to_csv('train_instances.csv', index=False)

    return train_df, instances_df, train_loc


# train_df, train_instances, train_loc = process_data(train_df, train_loc)

train_df, train_loc, train_instances = load_data(metadata=True)


for df, desc in zip([train_df, train_instances, train_loc],
                    ['New Train', 'Instances', 'New Localization']):
    summarize(df, desc)


train_loc = train_loc.drop(columns=['Rows', 'Columns'])
merged_df = pd.merge(train_loc, train_df, on='SeriesInstanceUID', how='left')


train_pivot = pd.pivot_table(
    train_df,
    index='Aneurysm Present',
    columns='Modality',
    values='PatientAge',
    aggfunc='count',
    margins=True,
    margins_name='Total'
)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    train_pivot, 
    annot=True,
    fmt='g',
    cmap='Reds_r',
    linewidths=0.5, 
    linecolor='gray',
    ax=ax
    
)

ax.set_title('Patient Count by Modality and Aneurysm Presence')
ax.set_ylabel('Aneurysm Present')
ax.set_xlabel('Modality')
fig.tight_layout()
fig.show()
fig.savefig('Figures/series_count_per_modality_and_aneurysm.png', dpi=300)


aneurysm_count = train_df['Aneurysm Present'].value_counts()
aneurysm = aneurysm_count.index.to_list()
count = aneurysm_count.to_list()
total = sum(count)
proportions = [c / total for c in count]

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(
    nrows=2, ncols=3, 
    height_ratios=[1, 8], 
    width_ratios=[4, 3, 0.01])  # 3rd col is dummy

ax0 = fig.add_subplot(gs[0, :2])

ax0.barh(y=0, width=proportions[0], left=0, 
         color=aneurysm_palette[aneurysm[0]], 
         label=f'{aneurysm[0]}: {count[0]}')
ax0.barh(y=0, width=proportions[1], left=proportions[0], 
         color=aneurysm_palette[aneurysm[1]], 
         label=f'{aneurysm[1]}: {count[1]}')

ax0.set_title('Aneurysm Presence')
ax0.set_ylabel('')
ax0.set_xlabel('')

ax0.text(proportions[0]/2, 0, f'{count[0]}', 
         ha='center', va='center', color='white', fontsize=10)
ax0.text(proportions[0] + proportions[1]/2, 0, f'{count[1]}',
         ha='center', va='center', color='white', fontsize=10)

for spine in ax0.spines.values():
    spine.set_visible(False)
ax0.tick_params(left=False, bottom=False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])

ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])

# How many Aneurysm?
sns.histplot(
    train_df.loc[:, CONFIG['ARTERIES'].values()].sum(axis=1), 
    discrete=True,
    ax=ax1
)

n_bins = len(ax1.patches)
h_colors = [COLORS[-1]] + [COLORS[1] for i in range(1, n_bins)]

for patch, color in zip(ax1.patches, h_colors):
    patch.set_facecolor(color)

    height = patch.get_height()
    x = patch.get_x() + patch.get_width() / 2

    if height > 0:
        ax1.annotate(f'{int(height)}',
                    xy=(x, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    color=color,
                    fontsize=10, fontweight='bold')

ax1.set_title('How many Aneurysms?')
ax1.set_ylim(0, 2700)

# Sex distribution
sns.histplot(data=train_df, x='PatientSex', hue='Aneurysm Present', multiple='dodge',
             shrink=0.8, palette=aneurysm_palette, ax=ax2)
ax2.set_title('Sex distribution')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_ylim(0, 2700)

fig.tight_layout()
fig.savefig("Figures/aneurysm_and_sex_distribution.png", dpi=300)
plt.show()


fig = plt.figure(figsize=(10, 7))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

ax0 = fig.add_subplot(gs[0])
sns.histplot(
    data=train_df,
    x='PatientAge',
    hue='Aneurysm Present',
    kde=True,
    palette=aneurysm_palette,
    ax=ax0
)
ax0.set_title('Age distribution')

ax1 = fig.add_subplot(gs[1], sharex=ax0)
sns.boxplot(
    data=train_df,
    x='PatientAge',
    y='Aneurysm Present',
    palette=aneurysm_palette,
    ax=ax1,
    orient='h'
)

plt.setp(ax0.get_xticklabels(), visible=False)
ax0.set_xlabel('')

fig.savefig('Figures/age_distribution.png', dpi=300)
plt.show()


artery_count = train_df.loc[:, CONFIG['ARTERIES'].values()].sum(axis=0)
artery_count = artery_count.reset_index()
artery_count.columns = ['Artery', 'Count']

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=artery_count, 
    y='Artery', 
    x='Count', 
    palette='Reds_r',
    ax=ax
)

ax.set_title('Artery Aneurysm Count')
ax.set_xlabel('Count')
ax.set_ylabel('Artery')
fig.tight_layout()
fig.show()
fig.savefig('Figures/artery_distribution.png', dpi=300)


fig, axs = plt.subplots(ncols=2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 1]})

sns.histplot(
    data=train_df,
    x='NumberOfFrames',
    hue="Aneurysm Present",
    kde=True,
    palette=aneurysm_palette,
    ax=axs[0]
)

axs[0].set_title('Frame distribution')

multiframe = train_df.IsMultiFrame.value_counts()
axs[1].pie(
    multiframe,
    labels=multiframe.index,
    colors=[COLORS[1], COLORS[3]],
    autopct='%1.1f%%'
)
axs[1].set_title('Is Series MultiFrame?')

fig.tight_layout()
fig.show()
fig.savefig('Figures/frame_distribution_per_aneurysm.png', dpi=300)


g = sns.jointplot(
    data=train_df, 
    x="Columns", 
    y="Rows",
    palette=aneurysm_palette,
    hue="Aneurysm Present",
    height=6
)

g.figure.savefig('Figures/image_dimensions_per_aneurysm_distribution.png', dpi=300)
plt.show()


modality_counts = train_df['Modality'].value_counts().reset_index()
modality_counts.columns = ['Modality', 'Count']
modality_counts = modality_counts.sort_values('Modality')

bar_colors = modality_counts['Modality'].map(modality_palette)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=modality_counts,
    x='Modality',
    y='Count',
    palette=bar_colors,
    ax=ax
)

ax.set_title('Modality Distribution')
ax.set_xlabel('')
fig.tight_layout()
fig.savefig('Figures/modality_distribution.png', dpi=300)
plt.show()


artery_per_modality_count = (
    train_df
    .groupby('Modality')[list(CONFIG['ARTERIES'].values())]
    .sum()
    .T
)

artery_per_modality_count.reset_index(inplace=True)
artery_per_modality_count.rename(columns={'index': 'Artery'}, inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))

bottom = np.zeros(len(artery_per_modality_count))

modalities = artery_per_modality_count.columns[1:]
colors = [modality_palette[m] for m in modalities]

for modality, color in zip(modalities, colors):
    counts = artery_per_modality_count[modality]
    ax.barh(
        artery_per_modality_count['Artery'],
        counts,
        left=bottom,
        label=modality,
        color=color
    )
    bottom += counts

ax.set_title('Artery Aneurysm per Modality Count')
ax.set_xlabel('Count')
ax.set_ylabel('Artery')
ax.legend(title='Modality')
ax.invert_yaxis()

fig.tight_layout()
fig.show()
fig.savefig('Figures/artery_per_modality_distribution.png', dpi=300)


fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(
    data=train_df,
    x='NumberOfFrames',
    hue="Modality",
    kde=True,
    palette=modality_palette,
    ax=ax
)

ax.set_title('Frame distribution per modality')
fig.tight_layout()
fig.show()
fig.savefig('Figures/frame_per_modality_distribution.png', dpi=300)


g = sns.jointplot(
    data=train_df, 
    x="Columns", 
    y="Rows",
    palette=modality_palette,
    hue="Modality",
    height=6
)

g.figure.savefig('Figures/image_dimensions_per_modality_distribution.png', dpi=300)
plt.show()


def plot_multiple_series(modality=None, plane=None, n_samples=5):
    """
    Plots mid-slices from multiple randomly selected DICOM series 
    matching the specified modality and/or plane (if given).
    
    Parameters:
        modality (str or None): Filter by modality (optional).
        plane (str or None): Filter by plane (optional).
        n_samples (int): Number of series to plot (default: 8).
    """
    ncols = 5
    nrows = int(np.ceil(n_samples / ncols))

    if modality == 'MRA' and plane == 'sagittal':
        ncols = 1
        n_samples = 1
    
    condition = pd.Series(True, index=merged_df.index)
    if modality is not None:
        condition &= (merged_df['Modality'] == modality)
    if plane is not None:
        condition &= (merged_df['Plane'] == plane)

    series_list = merged_df.loc[condition, 'SeriesInstanceUID'].unique()

    if len(series_list) == 0:
        filters = []
        if modality: filters.append(f"modality='{modality}'")
        if plane: filters.append(f"plane='{plane}'")
        raise ValueError(f"No series found for {' and '.join(filters) or 'any condition'}.")

    if len(series_list) < n_samples:
        print(f"Warning: Only {len(series_list)} series available. Plotting all.")
        n_samples = len(series_list)

    selected_series = np.random.choice(series_list, size=n_samples, replace=False)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.flatten()

    for idx, series in enumerate(selected_series):
        series_path = os.path.join(CONFIG['PATH'], 'series', series)

        try:
            array = load_dicom(series_path)
            mid_slice = array.shape[0] // 2
            axs[idx].imshow(array[mid_slice], cmap='gray')
            axs[idx].axis('off')
            axs[idx].set_title(series, fontsize=6)
        except Exception as e:
            axs[idx].set_title(f"Error loading: {series}", fontsize=6)
            axs[idx].axis('off')

    for j in range(n_samples, len(axs)):
        axs[j].axis('off')

    mod_text = modality.upper() if modality else 'Any Modality'
    sec_text = plane if plane else 'Any Plane'
    fig.suptitle(f"{sec_text} {mod_text} â€” Random Mid-Slices", fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    os.makedirs("Figures", exist_ok=True)
    save_name = f"{plane or 'any'}_{(modality.replace(' ', '_') or 'any').lower().replace(' ', '_')}_multiple_series.png"
    fig.savefig(os.path.join("Figures", save_name), dpi=300)
    fig.show()


plot_multiple_series(modality='CTA')


plot_multiple_series(modality='MRA')


plot_multiple_series(modality='MRI T2')


plot_multiple_series(modality='MRI T1post')


ipps = ['IPP_X', 'IPP_Y', 'IPP_Z']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axs = axs.flatten()

for idx, ipp in enumerate(ipps):
    sns.histplot(
        train_instances,
        x=ipp,
        hue='Modality',
        kde=True,
        palette={'MR': COLORS[3], 'CT': COLORS[4]},
        bins=10,
        ax=axs[idx]
    )
    axs[idx].set_title(f'ImagePositionPatient{ipp[-1]}')
    axs[idx].set_xlabel('')

fig.tight_layout()
fig.show()
fig.savefig('Figures/ImagePositionPatient_distribution.png', dpi=300)


rscles = ['RescaleType', 'RescaleIntercept', 'RescaleSlope']

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axs = axs.flatten()

for idx, rscle in enumerate(rscles):
    sns.histplot(
        train_instances,
        x=rscle,
        hue='Modality',
        palette={'MR': COLORS[3], 'CT': COLORS[4]},
        ax=axs[idx]
    )
    axs[idx].set_title(rscle)
    axs[idx].set_xlabel('')

fig.tight_layout()
fig.show()
fig.savefig('Figures/Rescale_distribution.png', dpi=300)


slices = train_instances.groupby(['SeriesInstanceUID', 'Modality'])[['SliceThickness', 'SpacingBetweenSlices']].mean().reset_index()
g = sns.jointplot(
    data=slices, 
    x="SliceThickness", 
    y="SpacingBetweenSlices",
    hue='Modality',
    palette={'MR': COLORS[3], 'CT': COLORS[4]},
    kind='kde',
    height=6
)

g.figure.savefig('Figures/slice_dimensions_distribution.png', dpi=300)
plt.show()


pixels = train_instances.groupby(['SeriesInstanceUID', 'Modality'])[['PixelSpacing_X', 'PixelSpacing_Y']].mean().reset_index()

g = sns.jointplot(
    data=pixels, 
    x="PixelSpacing_X", 
    y="PixelSpacing_Y",
    hue='Modality',
    kind='kde',
    palette={'MR': COLORS[3], 'CT': COLORS[4]},
    height=6
)

g.figure.savefig('Figures/pixel_spacing_distribution.png', dpi=300)
plt.show()


windows = train_instances.groupby(['SeriesInstanceUID', 'Modality'])[['WindowWidth1', 'WindowCenter1']].mean().reset_index()

g = sns.jointplot(
    data=windows, 
    x="WindowWidth1", 
    y="WindowCenter1",
    hue='Modality',
    palette={'MR': COLORS[3], 'CT': COLORS[4]},
    height=6
)

g.figure.savefig('Figures/window_distribution.png', dpi=300)
plt.show()


windows = ['WindowWidth1', 'WindowCenter1']
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

for idx, window in enumerate(windows):
    sns.boxplot(
        train_instances,
        x=window,
        y='Modality',
        palette={'MR': COLORS[3], 'CT': COLORS[4]},
        ax=axs[idx],
        orient='h'
    )
    axs[idx].set_title(window)
    axs[idx].set_xlim(0, 3000)
    axs[idx].set_ylabel('')
    axs[idx].set_xlabel('')

fig.tight_layout()
fig.show()
fig.savefig('Figures/window_boxplot.png', dpi=300)


iops = ['IOP_RowX', 'IOP_RowY', 'IOP_RowZ', 'IOP_ColX', 'IOP_ColY', 'IOP_ColZ']

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

for idx, iop in enumerate(iops):
    axs[idx].hist(train_instances[iop], color=COLORS[-1], bins=10)
    axs[idx].set_title(f'ImageOrientationPatient{iop[-4:]}')
    axs[idx].set_xlim(-1, 1)
    axs[idx].set_xlabel('')

fig.tight_layout()
fig.show()
fig.savefig('Figures/ImageOrientationPatient_distribution.png', dpi=300)


plane_counts = train_df['Plane'].value_counts().reset_index()
plane_counts.columns = ['Plane', 'Count']
plane_counts = plane_counts.sort_values('Plane')

bar_colors = plane_counts['Plane'].map(plane_palette)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=plane_counts,
    x='Plane',
    y='Count',
    palette=bar_colors,
    ax=ax
)

ax.set_xlabel('')
ax.set_title('Plane Distribution')
fig.tight_layout()
fig.savefig('Figures/plane_distribution.png', dpi=300)
fig.show()


artery_per_plane_count = (
    train_df
    .groupby('Plane')[list(CONFIG['ARTERIES'].values())]
    .sum()
    .T
)

artery_per_plane_count.reset_index(inplace=True)
artery_per_plane_count.rename(columns={'index': 'Artery'}, inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))

bottom = np.zeros(len(artery_per_plane_count))

planes = artery_per_plane_count.columns[1:]
colors = [plane_palette[m] for m in planes]

for plane, color in zip(planes, colors):
    counts = artery_per_plane_count[plane]
    ax.barh(
        artery_per_plane_count['Artery'],
        counts,
        left=bottom,
        label=plane,
        color=color
    )
    bottom += counts

ax.set_title('Artery Aneurysm per Plane Count')
ax.set_xlabel('Count')
ax.set_ylabel('Artery')
ax.legend(title='Plane')
ax.invert_yaxis()

fig.tight_layout()
fig.show()
fig.savefig('Figures/artery_per_plane_distribution.png', dpi=300)


fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(
    data=train_df,
    x='NumberOfFrames',
    hue="Plane",
    kde=True,
    palette=plane_palette,
    ax=ax
)

ax.set_title('Frame distribution per plane')
fig.tight_layout()
fig.show()
fig.savefig('Figures/frame_per_plane_distribution.png', dpi=300)


g = sns.jointplot(
    data=train_df, 
    x="Columns", 
    y="Rows",
    palette=plane_palette,
    hue="Plane",
    height=6
)

g.figure.savefig('Figures/image_dimensions_per_plane_distribution.png', dpi=300)
plt.show()


def plot_single_series(modality=None, plane=None):
    """
    Plots all the slices of a randomly selected DICOM series 
    matching the specified modality and/or plane (if given).

    Parameters:
        modality (str or None): Filter by modality. If None, all modalities are included.
        plane (str or None): Filter by plane. If None, all sections are included.
    """

    condition = pd.Series(True, index=merged_df.index)
    if modality is not None:
        condition &= (merged_df['Modality'] == modality)
    if plane is not None:
        condition &= (merged_df['Plane'] == plane)

    series_list = merged_df.loc[condition, 'SeriesInstanceUID'].unique()

    if len(series_list) == 0:
        filters = []
        if modality: filters.append(f"modality='{modality}'")
        if plane: filters.append(f"plane='{plane}'")
        raise ValueError(f"No series found for {' and '.join(filters) or 'any condition'}.")

    selected_series = np.random.choice(series_list)
    series_path = os.path.join(CONFIG['PATH'], 'series', selected_series)

    try:
        series_array = load_dicom(series_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load DICOM from: {series_path}\n{e}")

    num_slices = series_array.shape[0]
    
    cols = math.ceil(math.sqrt(num_slices))
    rows = math.ceil(num_slices / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = axes.ravel()

    for i in range(rows*cols):
        ax = axes[i]
        if i < num_slices:
            ax.imshow(series_array[i, :, :], cmap='gray')
            ax.set_title(i)
        ax.axis('off')


    mod_text = modality.upper() if modality else 'Any Modality'
    sec_text = plane if plane else 'Any Plane'
    fig.suptitle(f"{sec_text} {mod_text}\n{selected_series}", fontsize=14)

    fig.tight_layout()
    save_name = f"{plane or 'any'}_{(modality.replace(' ', '_') or 'any').lower().replace(' ', '_')}_single_series.png"
    fig.savefig(os.path.join("Figures", save_name), dpi=300)
    fig.show()


plot_single_series(modality='MRI T1post', plane='axial')


plot_single_series(modality='MRI T1post', plane='sagittal')


plot_single_series(modality='MRI T1post', plane='coronal')


def plot_modality_per_plane():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    sampled = (
        merged_df.groupby(['Modality', 'Plane'], group_keys=False)
                 .apply(lambda g: g.sample(1, random_state=42))
                 .reset_index(drop=True)
    ).sort_values(['Plane', 'Modality'])

    planes = sorted(sampled['Plane'].unique())
    modalities = sorted(sampled['Modality'].unique())

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))

    for idx, (_, row) in enumerate(sampled.iterrows()):
        row_idx = planes.index(row['Plane'])
        col_idx = modalities.index(row['Modality'])

        ax = axs[row_idx, col_idx]
        series = row['SeriesInstanceUID']
        path = os.path.join(CONFIG['PATH'], 'series', series)

        try:
            array = load_dicom(path)
            slice_idx = array.shape[0] // 2
            ax.imshow(array[slice_idx], cmap='gray')
        except Exception as e:
            ax.set_title(f"Error: {series}", fontsize=8)
        finally:
            ax.axis('off')

        if row_idx == 0:
            ax.set_title(row['Modality'], fontsize=14)
        if col_idx == 0:
            ax.set_ylabel(row['Plane'], fontsize=14)

    fig.suptitle("One Random Series per Modality Ã— Plane", fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig("Figures/all_modality_per_plane.png", dpi=300)
    fig.show()


plot_modality_per_plane()


fig, axs = plt.subplot_mosaic(
    [["modality", "plane"],
     ["modality_vs_plane", "modality_vs_plane"]],
    figsize=(12, 9)
)

# First plot: Modality distribution per aneurysm
sns.histplot(
    data=train_df,
    x='Modality', 
    hue='Aneurysm Present', 
    multiple="dodge",
    shrink=0.8,
    palette=aneurysm_palette,
    ax=axs['modality']
)
axs['modality'].set_title('Modality Distribution')
axs['modality'].set_ylim(0, 2500)

# Second plot: Plane distribution per aneurysm
sns.histplot(
    data=train_df,
    x='Plane', 
    hue='Aneurysm Present', 
    multiple="dodge",
    shrink=0.8,
    palette=aneurysm_palette,
    ax=axs['plane']
)
axs['plane'].set_title('Plane Distribution')

# Third plot: Plane vs. Modality
sns.histplot(
    data=train_df,
    x='Plane', 
    hue='Modality', 
    multiple="dodge",
    shrink=0.8,
    palette=modality_palette,
    ax=axs['modality_vs_plane']
)
axs['modality_vs_plane'].set_title('Plane vs. Modality')
axs['modality_vs_plane'].set_ylim(0, 2500)

fig.tight_layout()
fig.show()
fig.savefig('Figures/combined_distribution.png', dpi=300)


all_loc_pivot = pd.pivot_table(
    merged_df,
    index='Aneurysm Present',
    columns='Modality',
    values='PatientAge',
    aggfunc='count',
    margins=True,
    margins_name='Total'
).iloc[:-1]

unique_loc_pivot = pd.pivot_table(
    merged_df.drop_duplicates('SeriesInstanceUID'),
    index='Aneurysm Present',
    columns='Modality',
    values='PatientAge',
    aggfunc='count',
    margins=True,
    margins_name='Total'
).iloc[:-1]

loc_pivot = pd.concat([all_loc_pivot, unique_loc_pivot])
loc_pivot.index = ['All Aneurysms', 'Unique series']

fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(
    loc_pivot, 
    annot=True,
    fmt='g',
    cmap='Reds_r',
    linewidths=0.5, 
    linecolor='gray',
    ax=ax
    
)

ax.set_title('Coordinate Count by Modality')
ax.set_ylabel('')
ax.set_xlabel('Modality')
fig.tight_layout()
fig.show()
fig.savefig('Figures/series_and_coordinate_count_per_modality.png', dpi=300)


fig = px.scatter(
    merged_df,
    x='x_norm',
    y='y_norm',
    color='location',
    category_orders={'location': CONFIG['ARTERIES'].values()},
    title='Normalized Coordinates by Location',
    labels={'x_norm': 'Normalized X', 'y_norm': 'Normalized Y'},
)

fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))
fig.update_layout(
    scattermode="group",
    template="plotly_white",
    legend_title_text='Location',
    width=1000,
    height=800
)

fig.show()
fig.write_html("Figures/aneurysm_normalized_coordinates.html")


def plot_multiple_localization(modality=None, plane=None, n_samples=5):
    """
    Plots annotated slices with aneurysm bounding boxes (32x32, clipped at borders)
    and an additional zoomed-in subplot showing the cropped region (with a red border).
    """
    condition = pd.Series(True, index=merged_df.index)
    if modality:
        condition &= (merged_df['Modality'] == modality)
    if plane:
        condition &= (merged_df['Plane'] == plane)

    all_instances = merged_df.loc[condition, 'SOPInstanceUID'].unique().tolist()
    if len(all_instances) == 0:
        filters = []
        if modality: filters.append(f"modality='{modality}'")
        if plane: filters.append(f"plane='{plane}'")
        raise ValueError(f"No SOPInstanceUIDs found for {' and '.join(filters) or 'any condition'}.")

    if len(all_instances) < n_samples:
        print(f"Warning: Only {len(all_instances)} instances available. Plotting all.")
        n_samples = len(all_instances)

    picked_instances = np.random.choice(all_instances, size=n_samples, replace=False)

    fig, axs = plt.subplots(nrows=2, ncols=n_samples, figsize=(5 * n_samples, 10), constrained_layout=True)
    if n_samples == 1:
        axs = np.array(axs).reshape(2, 1)

    for idx, instance in enumerate(picked_instances):
        try:
            row = merged_df.loc[merged_df.SOPInstanceUID == instance].iloc[0]
            series = row['SeriesInstanceUID']
            instance_number = train_instances.loc[
                train_instances.SOPInstanceUID == instance, 'InstanceNumber'
            ].iloc[0]
            series_path = os.path.join(CONFIG['PATH'], 'series', series)
            array = load_dicom(series_path)

            x, y = row['x'], row['y']
            rows, cols = row['Rows'], row['Columns']
            new_x = x * (512 / cols)
            new_y = y * (512 / rows)

            slice_img = array[instance_number]

            half = 16
            xmin = int(max(np.floor(new_x - half), 0))
            ymin = int(max(np.floor(new_y - half), 0))
            xmax = int(min(np.floor(new_x + half), slice_img.shape[1]))
            ymax = int(min(np.floor(new_y + half), slice_img.shape[0]))

            w = xmax - xmin
            h = ymax - ymin

            # --- Full slice with bbox ---
            axs[0, idx].imshow(slice_img, cmap="gray")
            axs[0, idx].add_patch(
                Rectangle((xmin - 0.5, ymin - 0.5), w, h, linewidth=2, edgecolor="red", facecolor="none")
            )
            axs[0, idx].axis("off")
            axs[0, idx].set_title(series, fontsize=8)

            # --- Cropped box (zoomed view) with perfectly fitting red border ---
            crop = slice_img[ymin:ymax, xmin:xmax]
            ch, cw = crop.shape[:2]
            axc = axs[1, idx]
            axc.imshow(crop, cmap="gray")
            axc.add_patch(
                Rectangle((-0.5, -0.5), cw, ch, linewidth=10, edgecolor="red", facecolor="none")
            )
            axc.set_xlim(-0.5, cw - 0.5)
            axc.set_ylim(ch - 0.5, -0.5)
            axc.set_aspect('equal')
            axc.axis("off")
            axc.set_title("Zoomed Crop", fontsize=8)

        except Exception as e:
            axs[0, idx].set_title(f"Error: {instance}", fontsize=6)
            axs[0, idx].axis("off")
            axs[1, idx].axis("off")
            print(f"[Warning] Failed to load or annotate {instance}: {e}")

    mod_text = modality.upper() if modality else "Any Modality"
    sec_text = plane if plane else "Any Plane"
    fig.suptitle(f"{sec_text} {mod_text} â€” Random Slices with BBoxes", fontsize=18)

    os.makedirs("Figures", exist_ok=True)
    save_name = f"{plane or 'any'}_{modality or 'any'}_bbox_zoom.png"
    save_name = save_name.lower().replace(" ", "_")
    fig.savefig(os.path.join("Figures", save_name), dpi=300)
    fig.show()


plot_multiple_localization(modality='CTA')


plot_multiple_localization(modality='MRA')


plot_multiple_localization(modality='MRI T2', plane='sagittal')


plot_multiple_localization(modality='MRI T1post', plane='coronal')


axial_single = merged_df[
    (~merged_df.IsMultiFrame) & 
    (merged_df.Plane == 'axial')
]

axial_merged = pd.merge(
    axial_single, 
    train_instances, 
    on=['SeriesInstanceUID']
)

first_slices = axial_merged.loc[axial_merged.InstanceNumber == 1]
last_slices = axial_merged.loc[axial_merged.InstanceNumber == axial_merged.NumberOfFrames]

instance_loc = pd.merge(
    axial_single, 
    train_instances, 
    on=['SeriesInstanceUID', 'SOPInstanceUID', 'Rows', 'Columns', 'SOPClassUID']
)

first_z = (
    first_slices[['SeriesInstanceUID', 'IPP_Z']]
    .rename(columns={'IPP_Z': 'FirstZ'})
)

last_z = (
    last_slices[['SeriesInstanceUID', 'IPP_Z']]
    .rename(columns={'IPP_Z': 'LastZ'})
)

instance_loc = instance_loc.merge(first_z, on='SeriesInstanceUID', how='left')
instance_loc = instance_loc.merge(last_z, on='SeriesInstanceUID', how='left')

instance_loc['ZDiffFirst'] = instance_loc['IPP_Z'] - instance_loc['FirstZ']
instance_loc['ZDiffLast'] = instance_loc['LastZ'] - instance_loc['IPP_Z']
instance_loc['ZDiffNorm'] = (instance_loc['IPP_Z'] - instance_loc['FirstZ']) / (
    instance_loc['LastZ'] - instance_loc['FirstZ']
)


fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(
    data=instance_loc,
    x='IPP_Z',
    hue='Modality_x',
    multiple='stack',
    palette=modality_palette,
    bins=50,
    ax=ax
)

ax.set_title('ImagePositionPatient distribution')

fig.tight_layout()
fig.show()
fig.savefig('Figures/ImagePositionPatient_per_modality_distribution.png', dpi=300)


def plot_instance_loc(feature='ZDiffNorm'):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    sns.histplot(
        data=instance_loc,
        x=feature,
        kde=True,
        bins=50,
        ax=axs[0]
    )
    axs[0].set_title('Normalized IPP_Z of Annotated Slice')
    
    sns.histplot(
        data=instance_loc,
        x=feature,
        hue='Modality_x',
        kde=True,
        palette=modality_palette,
        bins=50,
        ax=axs[1]
    )
    axs[1].set_title('Normalized IPP_Z x Modality')
    axs[1].set_ylabel('')
    
    sns.histplot(
        data=instance_loc,
        x=feature,
        hue='location',
        kde=True,
        bins=50,
        ax=axs[2]
    )
    axs[2].set_title('Normalized IPP_Z x Location')
    axs[2].set_ylabel('')

    fig.tight_layout()
    fig.savefig(f'Figures/ImagePositionPatient_with{feature}_distribution.png', dpi=300)
    fig.show()


plot_instance_loc(feature='ZDiffNorm')


plot_instance_loc(feature='ZDiffFirst')


plot_instance_loc(feature='ZDiffLast')


def get_folder_size(path):
    file_sizes = np.array([f.stat().st_size for f in Path(path).rglob('*') if f.is_file()], dtype=float)
    return file_sizes

seg_dir = os.path.join(CONFIG['PATH'], 'segmentations')
file_sizes = get_folder_size(seg_dir)
file_sizes /= 1024.0**2

print(clr.T+"----- Segmentation Size Stats -----\n"+clr.E)
print(f"Total Dir: {file_sizes.sum()/1024.0:.3f} GB\n")
print(f"Mean: {file_sizes.mean():.1f} MB\n")
print(f"Min: {file_sizes.min():.1f} MB\n")
print(f"Max: {file_sizes.max():.1f} MB\n")


seg_series = [re.sub(r'(_cowseg)?\.nii$', '', f) for f in os.listdir(seg_dir)]
seg_df = train_df[train_df.SeriesInstanceUID.isin(seg_series)]

all_seg_pivot = pd.pivot_table(
    seg_df,
    index='Aneurysm Present',
    columns='Modality',
    values='PatientAge',
    aggfunc='count',
    margins=True,
    margins_name='Total'
)


fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    all_seg_pivot, 
    annot=True,
    fmt='g',
    cmap='Reds_r',
    linewidths=0.5, 
    linecolor='gray',
    ax=ax
    
)

ax.set_title('Segmentation Count by Modality and Aneurysm Presence')
ax.set_ylabel('Aneurysm Present')
ax.set_xlabel('Modality')
fig.tight_layout()
fig.show()
fig.savefig('Figures/series_and_segmentation_count_per_modality.png', dpi=300)


def plot_3d_segment(series_uid=None, random=False):
    if random:
        series_uid = seg_df.SeriesInstanceUID.sample().iloc[0]

    seg_path = os.path.join(CONFIG['PATH'], 'segmentations', f'{series_uid}_cowseg.nii')
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    
    fig = go.Figure()
    
    for label, name in CONFIG['ARTERIES'].items():
        coords = np.where(seg == label)
        if coords[0].size == 0:
            continue
    
        x = coords[0]
        y = coords[1]
        z = coords[2]
    
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name=f"{label}: {name}",
            marker=dict(
                size=2,
                colorscale='Viridis',
                opacity=0.8)
        ))
    
    fig.update_layout(
        title="3D Segmentation",
        scene=dict(
            xaxis=dict(
                title='Right â†’ Left',
                autorange='reversed'
            ),
            yaxis_title='Posterior â†’ Anterior',
            zaxis=dict(
                title='Superior â†’ Inferior',
                autorange=True
            ),
            aspectmode='data'
        ),
        legend=dict(
            itemsizing='constant',
            font=dict(size=10)
        ),
        width=1000,
        height=900
    )
    
    fig.show()
    fig.write_html("Figures/intracranial_arteries_segmentation.html")


plot_3d_segment(random=True)


arteries_info = []

for idx, row in tqdm(seg_df.iterrows(), total=len(seg_df)):
    series_id = row['SeriesInstanceUID']

    image_path = os.path.join(seg_dir, series_id+'.nii')
    mask_path = os.path.join(seg_dir, series_id+'_cowseg.nii')

    image = load_nii(image_path, mask=False)
    mask = load_nii(mask_path, mask=True)

    shape = mask.shape
    for idx, artery in CONFIG['ARTERIES'].items():
        density = (mask == idx).mean()
        coords = np.argwhere(mask == idx)
    
        if coords.shape[0] != 0:
            
            min_x, min_y, min_z = coords.min(axis=0)
            max_x, max_y, max_z = coords.max(axis=0)
            len_x = max_x - min_x
            len_y = max_y - min_y
            len_z = max_z - min_z
            
            artery_info = [series_id, shape, artery, min_x, min_y, min_z, max_x, max_y, max_z, len_x, len_y, len_z, density]
        else:
            artery_info = [series_id, shape, artery] + [np.nan] * 10

        arteries_info.append(artery_info)

arteries_df = pd.DataFrame(arteries_info, columns=['SeriesInstanceUID', 'shape', 'location', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'X', 'Y', 'Z', 'Density'])
arteries_df.head()


missing_seg = pd.DataFrame(len(seg_df) - arteries_df.groupby('location').Density.count()[CONFIG['ARTERIES'].values()])
missing_seg


artery_size = arteries_df.groupby('location')[['X', 'Y', 'Z']].mean().loc[list(CONFIG['ARTERIES'].values())]
artery_size


fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

for i, feature in enumerate(['X', 'Y', 'Z']):
    
    sns.boxplot(
        data=arteries_df,
        x=feature,
        y='location',
        ax=axs[i]
    )
    axs[i].set_title(feature)
    axs[i].set_xlabel('')
    axs[i].set_xlim((0, 140))

plt.tight_layout()
plt.show()
fig.savefig('Figures/arteries_size_boxplot', dpi=300)


def create_cube_vertices(center, dimensions):
    """Create vertices for a cube centered at 'center' with given dimensions"""
    x_center, y_center, z_center = center
    x_size, y_size, z_size = dimensions
    
    dx, dy, dz = x_size/2, y_size/2, z_size/2
    
    vertices = np.array([
        [x_center-dx, y_center-dy, z_center-dz],  # 0
        [x_center+dx, y_center-dy, z_center-dz],  # 1
        [x_center+dx, y_center+dy, z_center-dz],  # 2
        [x_center-dx, y_center+dy, z_center-dz],  # 3
        [x_center-dx, y_center-dy, z_center+dz],  # 4
        [x_center+dx, y_center-dy, z_center+dz],  # 5
        [x_center+dx, y_center+dy, z_center+dz],  # 6
        [x_center-dx, y_center+dy, z_center+dz],  # 7
    ])
    return vertices

def get_cube_faces(vertices):
    """Define the 12 triangular faces of a cube (2 triangles per face)"""
    faces = [
        # Bottom face (z = z_min)
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[2], vertices[3]],
        # Top face (z = z_max)
        [vertices[4], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[6]],
        # Front face (y = y_min)
        [vertices[0], vertices[4], vertices[5]],
        [vertices[0], vertices[5], vertices[1]],
        # Back face (y = y_max)
        [vertices[2], vertices[6], vertices[7]],
        [vertices[2], vertices[7], vertices[3]],
        # Left face (x = x_min)
        [vertices[0], vertices[3], vertices[7]],
        [vertices[0], vertices[7], vertices[4]],
        # Right face (x = x_max)
        [vertices[1], vertices[5], vertices[6]],
        [vertices[1], vertices[6], vertices[2]],
    ]
    return faces


locations = artery_size.index.tolist()
coordinates = artery_size.values.tolist()

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab20(np.linspace(0, 1, len(locations)))

grid_size = int(np.ceil(np.sqrt(len(locations))))
spacing = 15  # Space between cubes

for i, (location, coord) in enumerate(zip(locations, coordinates)):
    x_dim, y_dim, z_dim = coord
    
    row = i // grid_size
    col = i % grid_size
    
    x_pos = col * spacing
    y_pos = row * spacing
    z_pos = z_dim * 0.15
    
    center = [x_pos, y_pos, z_pos]
    dimensions = [x_dim*0.3, y_dim*0.3, z_dim*0.3]
    
    vertices = create_cube_vertices(center, dimensions)
    
    faces = get_cube_faces(vertices)
    
    cube = Poly3DCollection(faces, alpha=0.7, facecolor=colors[i], edgecolor='black', linewidth=0.5)
    ax.add_collection3d(cube)

ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_zlabel('Z Coordinate', fontsize=12)
ax.set_title('3D Anatomical Locations - Size Comparison', 
             fontsize=14, fontweight='bold')

max_coord = max(grid_size * spacing, max([max(coord) for coord in coordinates]) * 0.3)
ax.set_xlim(-2, (grid_size-1) * spacing + 5)
ax.set_ylim(-2, (grid_size-1) * spacing + 5)
ax.set_zlim(0, max_coord)

legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, edgecolor='black') 
                  for i in range(len(locations))]

mid_point = len(locations) // 2
legend1 = ax.legend(legend_elements[:mid_point], locations[:mid_point], 
                   loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=8)
legend2 = ax.legend(legend_elements[mid_point:], locations[mid_point:], 
                   loc='upper left', bbox_to_anchor=(0.02, 0.5), fontsize=8)

ax.add_artist(legend1)

xx, yy = np.meshgrid(np.linspace(-2, (grid_size-1) * spacing + 5, 10),
                     np.linspace(-2, (grid_size-1) * spacing + 5, 10))
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

ax.grid(True, alpha=0.3)
ax.view_init(elev=20, azim=45)

fig.tight_layout()
fig.savefig('Figures/3d_size_comparison.png', dpi=300)
fig.show()

