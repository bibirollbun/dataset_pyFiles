# --- SETTINGS STUDENTS CAN TUNE ---

# If True, try to pick a series with an aneurysm. If False, pick any series.
PICK_POSITIVE = True

# Optionally force a specific SeriesInstanceUID (otherwise we'll sample one).
SERIES_TO_VIEW = None  # e.g., "1.2.840.113619...."  <<-- put a UID string here to lock a case

# How many slices to show in the grid, and how many columns in that grid.
NUM_SLICES = 18
GRID_COLS  = 6

# Random seed so results are repeatable when sampling.
RANDOM_SEED = 42


import os, glob, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

# NIfTI is optional; only needed if you want to overlay vessel segmentations.
try:
    import nibabel as nib
    HAVE_NIB = True
except Exception:
    HAVE_NIB = False



# Change this to your competition input path on Kaggle
# /kaggle/input/rsna-intracranial-aneurysm-detection

BASE_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection"  # <<-- EDIT if your path name differs

# train.csv = one row per series (scan), with 13 location labels + the main "Aneurysm Present".
df  = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))

# train_localizers.csv = points marking aneurysm centers (x, y) on specific slices.
loc = pd.read_csv(os.path.join(BASE_DIR, "train_localizers.csv"))

SERIES_DIR = os.path.join(BASE_DIR, "series")
SEG_DIR    = os.path.join(BASE_DIR, "segmentations")



df.head(8)


if SERIES_TO_VIEW:
    row = df[df["SeriesInstanceUID"] == SERIES_TO_VIEW].iloc[0]
else:
    if PICK_POSITIVE and "Aneurysm Present" in df.columns and df["Aneurysm Present"].sum() > 0:
        cand = df[df["Aneurysm Present"] == 1]
    else:
        cand = df
    row = cand.sample(1, random_state=0).iloc[0]

sid = row["SeriesInstanceUID"]
modality = row.get("Modality", "Unknown")
series_path = os.path.join(SERIES_DIR, sid)
assert os.path.isdir(series_path), f"Series folder not found: {series_path}"
# print(f"Series Path: {series_path}")
print(f"Chosen SeriesInstanceUID: {sid}")
# print(f"Modality: {modality}")


from IPython.display import display
display(row.to_frame().T)

# print(row.to_frame().T.to_string(index=False))


# files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))  
# print(f"Total slices: {len(files)}")
# names = [os.path.basename(p) for p in files]
# print(names[:8] + ["..."] + names[-4:]) # show the first 8 images and the last 4 images


# dcm_path = files[len(files)//2] 
# ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
# print("Modality:", getattr(ds,"Modality",None))
# print("PatientAge:", getattr(ds,"PatientAge",None), "PatientSex:", getattr(ds,"PatientSex",None))
# print("Rows x Cols:", getattr(ds,"Rows",None),"x",getattr(ds,"Columns",None))
# print("PixelSpacing (mm):", getattr(ds,"PixelSpacing",None))        
# print("SliceThickness (mm):", getattr(ds,"SliceThickness",None))
# print("ImagePositionPatient (mm):", getattr(ds,"ImagePositionPatient",None))  
# print("ImageOrientationPatient:", getattr(ds,"ImageOrientationPatient",None)) 
# print("InstanceNumber:", getattr(ds,"InstanceNumber",None))


def sort_key(fp):
    ds = pydicom.dcmread(fp, stop_before_pixels=True)
    # Use physical position (z) if available; else fall back to instance number.
    if "ImagePositionPatient" in ds:
        return float(ds.ImagePositionPatient[2])
    if "InstanceNumber" in ds:
        return int(ds.InstanceNumber)
    return 0

dcm_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")), key=sort_key)
assert len(dcm_files) > 0, "No DICOM files found in this series."

slices = []
for fp in dcm_files:
    ds = pydicom.dcmread(fp)
    arr = ds.pixel_array.astype(np.float32)

    # Most CT/CTA store a linear transform (slope/intercept) to convert to Hounsfield Units (HU).
    slope     = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    slices.append(arr)

vol = np.stack(slices, axis=0)  # shape: [Z, Y, X]
print("Volume shape [Z, Y, X]:", vol.shape)



def to_uint8(img3d, center=None, width=None):
    img = img3d.copy()
    if (center is not None) and (width is not None):
        vmin = center - width/2.0
        vmax = center + width/2.0
    else:
        vmin, vmax = np.percentile(img, (1, 99))  # generic for MRI/MRA
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-6)
    return (img * 255).astype(np.uint8)

use_ct_window = (str(modality).upper() in ["CT", "CTA"])
vol_u8 = to_uint8(vol, center=40, width=300) if use_ct_window else to_uint8(vol)



print("Shape of 'vol_u8' after windowing:", vol_u8.shape)


def show_montage(vol_u8, nslices=NUM_SLICES, ncols=GRID_COLS, title=""):
    idx = np.linspace(0, vol_u8.shape[0]-1, nslices).astype(int)
    nrows = int(np.ceil(nslices / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.5*nrows))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(idx):
            ax.imshow(vol_u8[idx[i]], cmap="gray")
            ax.set_title(f"z={idx[i]}")
    fig.suptitle(title)
    plt.show()

# show_montage(vol_u8, nslices=NUM_SLICES, ncols=GRID_COLS,
#              title=f"Series {sid} | {modality} | shape {vol_u8.shape}")

show_montage(vol_u8, title=f"Series {sid} | {modality} | shape {vol_u8.shape}")

