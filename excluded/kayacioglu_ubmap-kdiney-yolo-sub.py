import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




import os

archive_path = "/kaggle/input/ultralytics-for-offline-install/archive.tar.gz"

# Install silently (no internet needed)
os.system(f"pip install --no-deps {archive_path} > /dev/null")

print("ultralytics installed offline successfully!")


!pip install --no-deps --force-reinstall /kaggle/input/imagecodecs/imagecodecs-2024.9.22-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl



import imagecodecs
print("imagecodecs version:", imagecodecs.__version__)



# Offline install: ultralytics + dependencies

import os

# Extract the archive (tar.gz) to a temp folder
os.system("mkdir -p /kaggle/temp_ultra && tar -xzf /kaggle/input/ultralytics-for-offline-install/archive.tar.gz -C /kaggle/temp_ultra")

# Install from the local wheels, no internet needed
os.system("pip install --no-index --find-links=/kaggle/temp_ultra/packages ultralytics==8.3.40 > /dev/null")

print("ultralytics and dependencies installed offline!")



from ultralytics import YOLO
print("Ultralytics imported successfully!")




import gc, torch, numpy as np, pandas as pd, tifffile as tiff
from pathlib import Path
from ultralytics import YOLO

# Paths
DATA = Path("/kaggle/input/hubmap-kidney-segmentation/")
TEST_DIR = DATA / "test"
SAMPLE_SUB = DATA / "sample_submission.csv"
WEIGHTS = Path("/kaggle/input/hubmap-yolo-seg/best.pt")
OUT_CSV = Path("/kaggle/working/submission.csv")

SAMPLE_PATH = "/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv"
FINAL_PATH  = "submission.csv"


import numpy as np
import pandas as pd
import gc, torch
from pathlib import Path
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tifffile as tiff
import imagecodecs

# CONFIG

#TEST_DIR = Path("/kaggle/input/hubmap-kidney-segmentation/test")
SUB_PATH = Path("/kaggle/working/submission.csv")
MODEL_PATH = Path("/kaggle/input/hubmap-yolo-seg/best.pt")  # adjust!
TILE, STRIDE = 512, 512
DOWNSAMPLE = 2  # 2â€“4 recommended for safety

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Safety setup
Image.MAX_IMAGE_PIXELS = 3_000_000_000  # allow huge TIFFs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Safe TIFF reader
def read_tiff_safe(path, downsample=2):
    with tiff.TiffFile(str(path)) as tif:
        arr = tif.pages[0].asarray()
    if downsample > 1:
        arr = arr[::downsample, ::downsample]
    if arr.ndim == 2:
        arr = np.stack([arr]*3, -1)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


# RLE encoder
def mask_to_rle(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# Tile-based prediction
def predict_tiff(model, path, tile=512, stride=512, downsample=2, device=0):
    img = read_tiff_safe(path, downsample=downsample)
    if img is None:
        print(f"Skipping {path.name} (unreadable TIFF)")
        return np.zeros((512, 512), dtype=np.uint8)

    H, W = img.shape[:2]
    print(f"ðŸ§¬ {path.name}: {W}x{H} (downsample={downsample})")

    full_mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            crop = img[y:y+tile, x:x+tile]
            if crop.shape[0] < tile or crop.shape[1] < tile:
                pad = np.zeros((tile, tile, 3), dtype=crop.dtype)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad

            results = model.predict(
                source=crop,
                conf=0.1,
                imgsz=tile,
                verbose=False,
                device=device
            )
            if results and results[0].masks is not None:
                mask = results[0].masks.data.sum(0).cpu().numpy()
                mask = (mask > 0).astype(np.uint8)
                full_mask[y:y+tile, x:x+tile] = np.maximum(
                    full_mask[y:y+tile, x:x+tile],
                    mask[:min(tile, H-y), :min(tile, W-x)]
                )
        gc.collect()
    return full_mask
import os

# Dynamically collect all test TIFFs for inference

# Recursively find all .tiff or .tif files
test_tiffs = []
for root, dirs, files in os.walk(TEST_DIR):
    for f in files:
        if f.lower().endswith((".tiff", ".tif")):
            test_tiffs.append(Path(root) / f)

test_tiffs = sorted(test_tiffs)
print(f"Found {len(test_tiffs)} test images:")
for t in test_tiffs:
    print("  ", t.name)

# Run inference and build submission

model = YOLO(str(MODEL_PATH))
model.to(device)
print("Model loaded!")

test_tiffs = sorted(TEST_DIR.glob("*.tiff"))
pred_rows = []

for wsi in test_tiffs:
    mask = predict_tiff(model, wsi, TILE, STRIDE, DOWNSAMPLE, device)
    rle = mask_to_rle(mask) if mask.sum() > 0 else ""
    pred_rows.append({"id": wsi.stem, "predicted": rle})
    gc.collect()

# Save submission
sub_df = pd.DataFrame(pred_rows)
sample = pd.read_csv("/kaggle/input/hubmap-kidney-segmentation/sample_submission.csv")
final = sample[["id"]].merge(sub_df, on="id", how="left").fillna("")
final.to_csv("submission.csv", index=False)
print("Submission saved with", len(final), "rows")

'''
# Optional overlay preview

try:
    preview = test_tiffs[0]
    print(f"Rendering overlay for {preview.stem}...")
    img = read_tiff_safe(preview, downsample=4)
    mask = (mask > 0).astype(np.uint8)
    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 0]  # red mask overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Overlay: {preview.stem}")
    plt.axis("off")
    plt.show()
except Exception as e:
    print("Visualization skipped:", e)
'''

