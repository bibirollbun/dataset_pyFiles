!pip -q install nnunetv2 nibabel pydicom tqdm


!nvidia-smi


import os, re, json, sys, shutil, zipfile, glob
from pathlib import Path
import numpy as np
import nibabel as nib
import pydicom
from tqdm import tqdm


INPUT_ROOT = Path("/kaggle/input/rsna-intracranial-aneurysm-detection")
RAW_ROOT   = Path("/kaggle/working/nnunet_raw")
PREP_ROOT  = Path("/kaggle/working/nnunet_preprocessed")
RES_ROOT   = Path("/kaggle/working/nnunet_results")

os.environ["nnUNet_raw"]         = str(RAW_ROOT)
os.environ["nnUNet_preprocessed"] = str(PREP_ROOT)
os.environ["nnUNet_results"]      = str(RES_ROOT)

DATASET_ID  = 601
DATASET_TAG = f"Dataset{DATASET_ID:03d}_RSNAIA"
DS_ROOT = RAW_ROOT / DATASET_TAG
IMAGES_TR = DS_ROOT / "imagesTr"
LABELS_TR = DS_ROOT / "labelsTr"
for p in [IMAGES_TR, LABELS_TR, PREP_ROOT, RES_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("INPUT_ROOT exists:", INPUT_ROOT.exists())
print("Will write nnU-Net raw data to:", DS_ROOT)


from pathlib import Path
import nibabel as nib
import numpy as np

# === Define your dataset paths ===
TRAIN_SERIES = Path("/kaggle/input/rsna-intracranial-aneurysm-detection/series")
TRAIN_SEGS   = Path("/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations")

# === Collect all series folders ===
series_candidates = [p for p in TRAIN_SERIES.iterdir() if p.is_dir()]
print(f"Found {len(series_candidates)} DICOM series folders")

# === Collect all segmentation NIfTI files ===
seg_map = {}
for seg_path in sorted(TRAIN_SEGS.glob("*.nii*")):
    uid = seg_path.stem.replace(".nii","").replace(".gz","")
    seg_map[uid] = seg_path
print(f"Found {len(seg_map)} segmentation masks")

# === Match series folders to segmentation files by UID ===
pairs = []
for s in series_candidates:
    uid = s.name  # folder name is the UID
    if uid in seg_map:
        pairs.append((uid, s, seg_map[uid]))
print(f"Matched {len(pairs)} series+mask pairs")

# Quick sanity check on one pair
if pairs:
    uid, sdir, smask = pairs[0]
    print(f"Example:\nUID: {uid}\nSeries folder: {sdir}\nMask: {smask}")



import pydicom, numpy as np, nibabel as nib

def _read_dicom_series_from_dir(series_dir):
    """Read a DICOM series (folder of .dcm) and return sorted list of slices."""
    dcm_files = [p for p in Path(series_dir).glob("**/*") if p.is_file()]
    ds_list = []
    for p in dcm_files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False, force=True)
            if hasattr(ds, "PixelData"):
                ds_list.append(ds)
        except Exception:
            pass
    if not ds_list:
        raise RuntimeError(f"No readable DICOMs in {series_dir}")

    # sort by slice location (or instance number fallback)
    def slice_key(ds):
        if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
            ipp = np.array(ds.ImagePositionPatient, dtype=float)
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            row, col = iop[:3], iop[3:]
            normal = np.cross(row, col)
            return float(np.dot(ipp, normal))
        return float(getattr(ds, "InstanceNumber", 0))
    ds_list.sort(key=slice_key)
    return ds_list


def _dicom_list_to_nifti(ds_list, out_path):
    """Stack DICOM slices and save as .nii.gz (in HU if CT)."""
    imgs = []
    for ds in ds_list:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        imgs.append(arr * slope + inter)
    vol = np.stack(imgs, axis=-1)

    # build simple affine
    ds0 = ds_list[0]
    ps = np.array(getattr(ds0, "PixelSpacing", [1.0, 1.0]), dtype=float)
    try:
        st = float(getattr(ds0, "SliceThickness"))
    except Exception:
        st = 1.0
    iop = np.array(getattr(ds0, "ImageOrientationPatient", [1,0,0,0,1,0]), dtype=float)
    row, col = iop[:3], iop[3:]
    nor = np.cross(row, col)
    origin = np.array(getattr(ds0, "ImagePositionPatient", [0,0,0]), dtype=float)

    affine = np.eye(4)
    affine[:3,0] = row * ps[1]
    affine[:3,1] = col * ps[0]
    affine[:3,2] = nor * st
    affine[:3,3] = origin

    vol = np.clip(vol, -1024, 3071)
    img = nib.Nifti1Image(vol.astype(np.int16), affine)
    nib.save(img, str(out_path))



from tqdm import tqdm
import shutil, nibabel as nib, numpy as np

TMP = Path("/kaggle/working/tmp_series")
TMP.mkdir(exist_ok=True, parents=True)

fail_count, done = 0, 0

for uid, series_src, seg_path in tqdm(pairs, desc="Converting to nnU-Net raw"):
    img_out = IMAGES_TR / f"{uid}_0000.nii.gz"
    lab_out = LABELS_TR / f"{uid}.nii.gz"
    if img_out.exists() and lab_out.exists():
        continue
    try:
        # --- convert DICOM folder -> NIfTI image ---
        ds_list = _read_dicom_series_from_dir(series_src)
        _dicom_list_to_nifti(ds_list, img_out)

        # --- load segmentation, ensure binary mask ---
        seg_img = nib.load(str(seg_path))
        seg_arr = seg_img.get_fdata()
        seg_bin = (seg_arr > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(seg_bin, affine=seg_img.affine), str(lab_out))
        done += 1
    except Exception as e:
        fail_count += 1
        print(f"[WARN] Failed {uid}: {e}")

print(f"✅ Converted {done} cases. ❌ Failed: {fail_count}")



from sklearn.model_selection import train_test_split
import json

# list of all converted cases
all_cases = sorted([p.stem.replace("_0000","") for p in IMAGES_TR.glob("*_0000.nii.gz")])
print("Total usable cases:", len(all_cases))

# simple 80/20 split
train_cases, val_cases = train_test_split(all_cases, test_size=0.2, random_state=42)

# build dataset.json
dataset_json = {
    "name": "RSNAIA",
    "description": "RSNA Intracranial Aneurysm Segmentation Dataset",
    "tensorImageSize": "3D",
    "reference": "Kaggle RSNA Intracranial Aneurysm Detection 2024",
    "licence": "Challenge rules apply",
    "release": "1.0",
    "modality": {"0": "CT"},
    "labels": {"0": "background", "1": "aneurysm"},
    "numTraining": len(all_cases),
    "file_ending": ".nii.gz",
    "training": [
        {"image": f"./imagesTr/{c}_0000.nii.gz", "label": f"./labelsTr/{c}.nii.gz"}
        for c in all_cases
    ],
    "test": []
}

# save to Dataset601_RSNAIA
with open(DS_ROOT / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

# save split lists (optional, but helpful later)
with open(DS_ROOT / "split_train.txt", "w") as f: f.write("\n".join(train_cases))
with open(DS_ROOT / "split_val.txt", "w") as f: f.write("\n".join(val_cases))

print("✅ dataset.json created at:", DS_ROOT / "dataset.json")
print("Train cases:", len(train_cases), "Val cases:", len(val_cases))



import os, sys, subprocess, glob

# Locate where nnunetv2 is actually installed
nnunet_path = subprocess.check_output(
    ["python3", "-c", "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))"]
).decode().strip()
print("nnUNetv2 package path:", nnunet_path)

# Look for the verify and plan scripts
verify_script = glob.glob(os.path.join(nnunet_path, "**/verify_dataset*.py"), recursive=True)
plan_script = glob.glob(os.path.join(nnunet_path, "**/plan_and_preprocess*.py"), recursive=True)

print("Found verify scripts:", verify_script)
print("Found plan scripts:", plan_script)



import os
os.environ["nnUNet_raw"] = "/kaggle/working/nnunet_raw"
os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnunet_preprocessed"
os.environ["nnUNet_results"] = "/kaggle/working/nnunet_results"

# confirm the correct dataset exists
!ls /kaggle/working/nnunet_raw/Dataset601_RSNAIA


import json

json_path = "/kaggle/working/nnunet_raw/Dataset601_RSNAIA/dataset.json"

with open(json_path, "r") as f:
    data = json.load(f)

# Add required "channel_names" field if missing
if "channel_names" not in data:
    data["channel_names"] = {"0": "CT"}

# (Optional) keep consistent order & re-save
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print("✅ Fixed dataset.json; added channel_names = {'0': 'CT'}")


import json

json_path = "/kaggle/working/nnunet_raw/Dataset601_RSNAIA/dataset.json"

with open(json_path, "r") as f:
    data = json.load(f)

# Fix label structure (keys should be names, values are integers)
data["labels"] = {"background": 0, "aneurysm": 1}

# Ensure channel_names still present
if "channel_names" not in data:
    data["channel_names"] = {"0": "CT"}

# Save back
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print("✅ Fixed dataset.json — labels now use correct format (background:0, aneurysm:1)")



from pathlib import Path
import nibabel as nib

IMAGES = Path("/kaggle/working/nnunet_raw/Dataset601_RSNAIA/imagesTr")
LABELS = Path("/kaggle/working/nnunet_raw/Dataset601_RSNAIA/labelsTr")

bad = []
for img_path in IMAGES.glob("*_0000.nii.gz"):
    uid = img_path.stem.replace("_0000","")
    lab_path = LABELS / f"{uid}.nii.gz"
    if not lab_path.exists():
        continue
    try:
        img = nib.load(str(img_path))
        lab = nib.load(str(lab_path))
        if img.shape != lab.shape:
            bad.append(uid)
    except Exception as e:
        print(f"{uid}: {e}")
        bad.append(uid)

print("❌ Problematic cases:", bad)
print("Count:", len(bad))



import SimpleITK as sitk
from pathlib import Path

IMAGES = Path("/kaggle/working/nnunet_raw/Dataset601_RSNAIA/imagesTr")
LABELS = Path("/kaggle/working/nnunet_raw/Dataset601_RSNAIA/labelsTr")

fixed = 0
for img_path in IMAGES.glob("*_0000.nii.gz"):
    uid = img_path.stem.replace("_0000","")
    lab_path = LABELS / f"{uid}.nii.gz"
    if not lab_path.exists():
        continue
    try:
        img = sitk.ReadImage(str(img_path))
        seg = sitk.ReadImage(str(lab_path))
        seg = sitk.Resample(seg, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, seg.GetPixelID())
        sitk.WriteImage(seg, str(lab_path))
        fixed += 1
    except Exception as e:
        print("⚠️ Failed to fix", uid, e)

print(f"✅ Realigned {fixed} masks to match image orientation/spacing.")


