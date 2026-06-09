# Cell 0: install libs (if needed) and imports
# On Kaggle you may have internet; if not, skip pip and ensure required packages are available.
try:
    import segment_anything as sam
except Exception:
    # try installing segment-anything (only works if internet is enabled)
    !pip install -q git+https://github.com/facebookresearch/segment-anything.git
    import segment_anything as sam

# common libs
import os, sys, math, json
import numpy as np
import pandas as pd
import cv2
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import torch

print("torch:", torch.__version__)
print("segment_anything imported")



# Cell 1: utils for DICOM read + window + mask selection + visualize
import numpy as np
import pydicom, cv2
from skimage.measure import regionprops, label

def read_windowed_dcm(path, window_center=None, window_width=None):
    """
    Read DICOM and apply basic windowing. Returns uint8 0..255 image.
    """
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # use DICOM window if present, else fallback to percentiles
    try:
        if window_center is None:
            wc = float(ds.WindowCenter)
        else:
            wc = float(window_center)
        if window_width is None:
            ww = float(ds.WindowWidth)
        else:
            ww = float(window_width)
    except Exception:
        wc = np.median(arr)
        ww = np.percentile(arr, 99) - np.percentile(arr, 1)

    mn = wc - ww/2.0
    mx = wc + ww/2.0
    img = np.clip(arr, mn, mx)
    img = ((img - mn) / max(1e-6, (mx - mn)) * 255.0).astype(np.uint8)
    return img, ds

def save_mask_png(mask, out_path):
    # mask: boolean or 0/1 array
    mask_u8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(out_path, mask_u8)

def mask_to_bbox(mask):
    """Return x1,y1,x2,y2 for nonzero area of mask. If empty -> None."""
    ys, xs = np.where(mask)
    if len(xs)==0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2)+1, int(y2)+1

def enlarge_bbox(x1,y1,x2,y2, img_w, img_h, pad=0.2):
    """Pad bbox by fraction pad (0.2 => 20%), clamp to image size."""
    w = x2 - x1; h = y2 - y1
    padx = int(round(w * pad)); pady = int(round(h * pad))
    nx1 = max(0, x1 - padx)
    ny1 = max(0, y1 - pady)
    nx2 = min(img_w, x2 + padx)
    ny2 = min(img_h, y2 + pady)
    return nx1, ny1, nx2, ny2

def mask_centroid(mask):
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return None
    p = props[0]
    return (p.centroid[1], p.centroid[0])  # x,y



# Cell 2: load SAM / MedSAM model
CHECKPOINT_PATH = "/kaggle/input/medsam-checkpoint/samcheckpoint.pth"  # <-- place your checkpoint here

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"MedSAM/SAM checkpoint not found at {CHECKPOINT_PATH}. "
                            "Upload the .pth file to /kaggle/working/ or change CHECKPOINT_PATH.")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# choose model_type matching checkpoint, e.g., "vit_b", "vit_l", "vit_h"
MODEL_TYPE = "vit_b"   # change if your checkpoint is different (vit_b, vit_l, vit_h)

print("Loading SAM model - may take a moment...")
sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model.to(device)
print("Model loaded on", device)

# create automatic mask generator (for full-slice masks)
mask_generator = SamAutomaticMaskGenerator(sam_model,
                                           points_per_batch=64,   # tune for speed
                                           pred_iou_thresh=0.3,
                                           stability_score_thresh=0.5,
                                           min_mask_region_area=100)  # smallest mask in px

# create predictor (for point prompt)
predictor = SamPredictor(sam_model)



import os, io, zipfile, cv2, pydicom
import numpy as np, pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops

# ================
# Paths and setup
# ================
LABELS_CSV = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv"
BASE_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
OUT_ZIP_MASKS = "/kaggle/working/medsam_masks.zip"
OUT_ZIP_CROPS = "/kaggle/working/medsam_crops.zip"

CROP_SIZE = 224
BOX_PAD = 0.18
USE_POINT_PROMPT = True
MAX_SAMPLES = None

df = pd.read_csv(LABELS_CSV).astype({"study_id": str, "series_id": str})
records = []

# Create writable ZIP files
zip_masks = zipfile.ZipFile(OUT_ZIP_MASKS, "w", compression=zipfile.ZIP_DEFLATED)
zip_crops = zipfile.ZipFile(OUT_ZIP_CROPS, "w", compression=zipfile.ZIP_DEFLATED)

# ================
# Helper functions
# ================
def read_windowed_dcm(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope"): img *= ds.RescaleSlope
    if hasattr(ds, "RescaleIntercept"): img += ds.RescaleIntercept
    img = np.clip(img, np.percentile(img, 0.5), np.percentile(img, 99.5))
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    img = (img * 255).astype(np.uint8)
    return img, ds

def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0: return None
    return np.min(xs), np.min(ys), np.max(xs), np.max(ys)

def enlarge_bbox(x1, y1, x2, y2, w, h, pad=0.15):
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * pad), int(bh * pad)
    return (
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(w, x2 + pad_w),
        min(h, y2 + pad_h),
    )

def save_mask_to_zip(mask, fname, zip_handle):
    ok, buf = cv2.imencode(".png", (mask * 255).astype(np.uint8))
    if ok:
        zip_handle.writestr(fname, buf.tobytes())

def save_crop_to_zip(crop, fname, zip_handle):
    ok, buf = cv2.imencode(".png", crop)
    if ok:
        zip_handle.writestr(fname, buf.tobytes())

# ================
# Main loop
# ================
count = 0
for idx, row in tqdm(df.iterrows(), total=len(df)):
    if MAX_SAMPLES and count >= MAX_SAMPLES:
        break

    study, series, inst = str(row.study_id), str(row.series_id), int(row.instance_number)
    x, y = float(row.x), float(row.y)
    dcm_path = os.path.join(BASE_DIR, study, series, f"{inst}.dcm")
    if not os.path.exists(dcm_path):
        continue

    img, ds = read_windowed_dcm(dcm_path)
    h, w = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # === segmentation ===
    predictor.set_image(rgb)
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
    mask = masks if masks.ndim == 2 else masks[0]
    mask = mask.astype(bool)

    # === crop ===
    bbox = mask_to_bbox(mask)
    if bbox is None: continue
    x1, y1, x2, y2 = enlarge_bbox(*bbox, w, h, pad=BOX_PAD)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: continue
    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

    # === save into zips ===
    mask_fname = f"{study}_{series}_{inst}_{idx}_mask.png"
    crop_fname = f"{study}_{series}_{inst}_{idx}_crop.png"
    save_mask_to_zip(mask, mask_fname, zip_masks)
    save_crop_to_zip(crop_resized, crop_fname, zip_crops)

    records.append({
        "filename_crop": crop_fname,
        "filename_mask": mask_fname,
        "study_id": study,
        "series_id": series,
        "instance_number": inst,
        "x": x, "y": y,
        "condition": row.condition,
        "level": row.level
    })
    count += 1

# Close and save zips
zip_masks.close()
zip_crops.close()

# Save metadata CSV
out_df = pd.DataFrame(records)
out_df.to_csv("/kaggle/working/medsam_crops_metadata.csv", index=False)

print(f"âœ… Done! Saved {len(records)} crops and masks directly into ZIP files.")
print(f"ðŸ“¦ Crops ZIP size: {os.path.getsize(OUT_ZIP_CROPS)/1e9:.2f} GB")
print(f"ðŸ“¦ Masks ZIP size: {os.path.getsize(OUT_ZIP_MASKS)/1e9:.2f} GB")





