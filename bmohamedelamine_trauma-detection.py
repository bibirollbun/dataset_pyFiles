!pip install dicomsdl


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import SimpleITK as sitk
import matplotlib.patches as patches
import numpy.ma as ma
from ipywidgets import interact, IntSlider, HBox, Button, Output , Dropdown, VBox
from IPython.display import FileLink, display
import uuid
import pandas as pd
import dicomsdl
import nibabel as nib
from scipy.ndimage import zoom


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


ROOT_PATH = "/kaggle/input/rsna-2023-abdominal-trauma-detection"
MASKS_PATH = "/kaggle/input/datasets/bmohamedelamine/rsna-2023-spleen-masks"


def load_dicom_series(series_dir):
    files = glob(os.path.join(series_dir, "*.dcm"))
    if not files:
        raise FileNotFoundError("No DICOM files found")

    slices = []
    zs = []

    for f in files:
        ds = pydicom.dcmread(f)
        img = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            z = float(ipp[2])
        else:
            z = float(getattr(ds, "InstanceNumber", 0))

        slices.append(img)
        zs.append(z)

    order = np.argsort(np.array(zs))
    volume = np.stack([slices[i] for i in order])

    center, width = wc, ww
    low, high = center - width/2, center + width/2
    volume = np.clip(volume, low, high)
    volume = (volume - low) / (high - low + 1e-6)

    return volume




def __dataset__to_numpy_image(self, index=0):
    info = self.getPixelDataInfo()
    dtype = info['dtype']

    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]

    arr = np.empty(shape, dtype=dtype)
    self.copyFrameData(index, arr)
    return arr

# only patch if not already there
if not hasattr(dicomsdl._dicomsdl.DataSet, "to_numpy_image"):
    dicomsdl._dicomsdl.DataSet.to_numpy_image = __dataset__to_numpy_image
    
def fast_glob_sorted(path):
    # RSNA filenames: 1.dcm, 2.dcm ... → numeric sort is enough and cheap
    return sorted(glob(path), key=lambda x: int(os.path.basename(x).split('.')[0]))


def fast_window(img):
    WL,WW = wc,ww
    low  = WL - WW/2
    high = WL + WW/2
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-6)
    return img


def load_dicom_series_fast(series_dir):
    paths = fast_glob_sorted(os.path.join(series_dir, "*.dcm"))
    if not paths:
        raise FileNotFoundError("No DICOM files found")

    volume = []

    for p in paths:
        dcm = dicomsdl.open(p)

        img = dcm.to_numpy_image().astype(np.float32)

        slope = getattr(dcm, "RescaleSlope", 1.0)
        intercept = getattr(dcm, "RescaleIntercept", 0.0)
        img = img * slope + intercept

        img = fast_window(img)

        volume.append(img)

    return np.stack(volume).astype(np.float32)



def load_mask(mask_path, volume_shape):
    """Load a .nii_gz mask, fix orientation, and resize to match DICOM volume."""
    import shutil
    tmp = os.path.join("/kaggle/working", "tmp_mask.nii.gz")
    shutil.copy(mask_path, tmp)

    mask_nii = nib.load(tmp)
    mask_data = mask_nii.get_fdata()

    # NIfTI stores as (X, Y, Z) in RAS; DICOM volume is stacked as (Z, Y, X)
    mask_data = np.transpose(mask_data, (2, 1, 0))  # → (Z, Y, X)
    mask_data = mask_data[::-1, :, :]             # flip 

    # Resize to volume dims
    factors = np.array(volume_shape) / np.array(mask_data.shape)
    resized = zoom(mask_data, factors, order=0)
    return (resized > 0.5).astype(np.uint8)


def rsna_plain_viewer(ROOT_PATH, MASKS_PATH):
    csv_path = os.path.join(ROOT_PATH, "train_2024.csv")
    df = pd.read_csv(csv_path)

    # ---------------- LABEL FILTER DROPDOWNS ----------------
    spleen_dd = Dropdown(description="Spleen", options=["Any","Healthy","Low","High"], value="Any", layout={'width':'180px'})
    liver_dd  = Dropdown(description="Liver",  options=["Any","Healthy","Low","High"], value="Any", layout={'width':'180px'})
    kidney_dd = Dropdown(description="Kidney", options=["Any","Healthy","Low","High"], value="Any", layout={'width':'190px'})
    bowel_dd  = Dropdown(description="Bowel",  options=["Any","Healthy","Injury"],     value="Any", layout={'width':'180px'})
    extra_dd  = Dropdown(description="Extravasation", options=["Any","Healthy","Injury"], value="Any", layout={'width':'220px'})

    patient_dd = Dropdown(description="Patient:")
    series_dd  = Dropdown(description="Series:")
    slider     = IntSlider(description="Slice")
    save_btn   = Button(description="Save Slice", button_style="success")

    labels_out = Output()
    img_out    = Output()
    save_out   = Output()

    # ---------------- FILTERING LOGIC ----------------
    def apply_filter():
        temp = df.copy()
        if spleen_dd.value == "Low":       temp = temp[temp.spleen_low == 1]
        elif spleen_dd.value == "High":    temp = temp[temp.spleen_high == 1]
        elif spleen_dd.value == "Healthy": temp = temp[(temp.spleen_low == 0) & (temp.spleen_high == 0)]

        if liver_dd.value == "Low":        temp = temp[temp.liver_low == 1]
        elif liver_dd.value == "High":     temp = temp[temp.liver_high == 1]
        elif liver_dd.value == "Healthy":  temp = temp[(temp.liver_low == 0) & (temp.liver_high == 0)]

        if kidney_dd.value == "Low":       temp = temp[temp.kidney_low == 1]
        elif kidney_dd.value == "High":    temp = temp[temp.kidney_high == 1]
        elif kidney_dd.value == "Healthy": temp = temp[(temp.kidney_low == 0) & (temp.kidney_high == 0)]

        if bowel_dd.value == "Injury":     temp = temp[temp.bowel_injury == 1]
        elif bowel_dd.value == "Healthy":  temp = temp[temp.bowel_injury == 0]

        if extra_dd.value == "Injury":     temp = temp[temp.extravasation_injury == 1]
        elif extra_dd.value == "Healthy":  temp = temp[temp.extravasation_injury == 0]
        return temp

    def update_patients(change):
        filtered = apply_filter()
        if filtered.empty:
            patient_dd.options = []
            with labels_out:
                labels_out.clear_output()
                print("⚠️ No matching patients")
            return
        pts = filtered.patient_id.astype(str).tolist()
        patient_dd.options = pts
        patient_dd.value = pts[0]

    def show_labels(pid):
        with labels_out:
            labels_out.clear_output()
            row = df[df.patient_id == int(pid)]
            if row.empty:
                print("No label entry found.")
                return
            row = row.iloc[0]
            print(f"Patient {pid}")
            print("Labels:")
            for col in row.index:
                if col != "patient_id" and row[col] == 1:
                    print("  -", col)

    def update_series(change):
        pid = patient_dd.value
        patient_dir = os.path.join(ROOT_PATH, "train_images", pid)
        series = sorted(os.listdir(patient_dir))
        series_dd.options = series
        if series:
            series_dd.value = series[0]
        show_labels(pid)
        load_series(None)

    # ---------------- LOAD + VIEW (with mask) ----------------
    def load_series(change):
        pid = patient_dd.value
        sid = series_dd.value
        if pid is None or sid is None:
            return

        series_dir = os.path.join(ROOT_PATH, "train_images", pid, sid)
        volume = load_dicom_series_fast(series_dir)

        # Try to load the spleen mask
        mask = None
        mask_dir = os.path.join(MASKS_PATH, str(pid), str(sid))
        if os.path.isdir(mask_dir):
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith("nii_gz") or f.endswith(".nii.gz")]
            if mask_files:
                mask_path = os.path.join(mask_dir, mask_files[0])
                try:
                    mask = load_mask(mask_path, volume.shape)
                except Exception as e:
                    print(f"⚠️ Mask load failed: {e}")
                    mask = None

        slider.min = 0
        slider.max = volume.shape[0] - 1
        slider.value = volume.shape[0] // 2

        def _draw_overlay(ax, ct_slice, mask_slice):
            """Draw CT with green mask fill + lime contour on given axes."""
            ax.imshow(ct_slice, cmap="gray")
            if mask_slice is not None and mask_slice.any():
                # Semi-transparent green fill via RGBA overlay
                rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
                rgba[mask_slice == 1] = [0.0, 1.0, 0.2, 0.35]
                ax.imshow(rgba)
                # Sharp lime contour
                ax.contour(mask_slice, levels=[0.5], colors='lime', linewidths=1.5)

        def update_slice(change):
            z = slider.value
            with img_out:
                img_out.clear_output(wait=True)
                has_mask = mask is not None

                fig, axes = plt.subplots(1, 2 if has_mask else 1,
                                         figsize=(12 if has_mask else 6, 6))

                if has_mask:
                    ax_ct, ax_overlay = axes

                    ax_ct.imshow(volume[z], cmap="gray")
                    ax_ct.set_title(f"CT — Slice {z+1}/{volume.shape[0]}")
                    ax_ct.axis("off")

                    _draw_overlay(ax_overlay, volume[z], mask[z])
                    ax_overlay.set_title("Spleen Mask Overlay")
                    ax_overlay.axis("off")
                else:
                    axes.imshow(volume[z], cmap="gray")
                    axes.set_title(f"CT — Slice {z+1}/{volume.shape[0]} (no mask)")
                    axes.axis("off")

                fig.suptitle(f"Patient {pid} | Series {sid}", fontsize=13)
                plt.tight_layout()
                plt.show()

        slider.observe(update_slice, names="value")
        update_slice(None)

        def save_slice(b):
            z = slider.value
            filename = f"{pid}_{sid}_slice{z+1}_{uuid.uuid4().hex[:6]}.png"
            fig, ax = plt.subplots(figsize=(6, 6))
            _draw_overlay(ax, volume[z], mask[z] if mask is not None else None)
            ax.axis("off")
            fig.savefig(filename, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            with save_out:
                save_out.clear_output()
                print("Saved:")
                display(FileLink(filename))

        save_btn._click_handlers.callbacks = []
        save_btn.on_click(save_slice)

    # ---------------- HOOK EVENTS ----------------
    for w in [spleen_dd, liver_dd, kidney_dd, bowel_dd, extra_dd]:
        w.observe(update_patients, names="value")
    patient_dd.observe(update_series, names="value")
    series_dd.observe(load_series, names="value")
    update_patients(None)

    display(VBox([
        HBox([spleen_dd, liver_dd, kidney_dd, bowel_dd, extra_dd]),
        patient_dd, series_dd, labels_out,
        HBox([slider, save_btn]),
        img_out, save_out
    ]))





wc, ww = 50, 400 
rsna_plain_viewer(ROOT_PATH, MASKS_PATH)

