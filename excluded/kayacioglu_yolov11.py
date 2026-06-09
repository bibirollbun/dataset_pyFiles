# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Install dependencies
!pip install -q ultralytics opencv-python-headless rasterio shapely

import os
import cv2
import gc
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm.notebook import tqdm
from shapely.geometry import Polygon
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil




# Configuration
DATA_ROOT = "/kaggle/input/hubmap-kidney-segmentation"
WORK_DIR = "/kaggle/working/hubmap_yolo"

# Optimized for speed on Kaggle T4/P100
PATCH_SIZE = 512
STRIDE = 256     # Set to 512 (no overlap) for fastest training. Set to 256 for better accuracy but 4x slower.
MIN_TISSUE = 0.05 # Drop empty patches
BATCH_SIZE = 16   # Increased from 8
EPOCHS = 15      

# Create Directories
for split in ['train', 'val']:
    os.makedirs(f"{WORK_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{WORK_DIR}/{split}/labels", exist_ok=True)


def rle2mask(mask_rle, shape):
    """Decodes RLE string to numpy array."""
    if pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def tissue_ratio(patch):
    """Calculates the percentage of tissue in a patch."""
    if len(patch.shape) == 2 or patch.shape[-1] == 1:
        gray = patch
    else:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    
    # Otsu's thresholding
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Count non-background pixels (assuming background is light/white in H&E usually, 
    # but Otsu finds the separator. For HuBMAP, background is often white).
    # Inverted check: tissue is usually darker.
    tissue_pixels = np.sum(mask == 0) 
    return tissue_pixels / mask.size

def mask_to_polygons(mask):
    """Converts binary mask to normalized polygons for YOLO."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 200: # Filter tiny noise
            continue
        cnt = cnt.squeeze()
        if len(cnt.shape) < 2 or len(cnt) < 6: # Filter invalid lines
            continue
        polygons.append(cnt)
    return polygons


# Load metadata
df_masks = pd.read_csv(f"{DATA_ROOT}/train.csv").set_index('id')
image_ids = [f.split(".")[0] for f in os.listdir(f"{DATA_ROOT}/train") if f.endswith(".tiff")]

# Split by IMAGE ID, not patch (Prevents data leakage)
train_ids, val_ids = train_test_split(image_ids, test_size=0.1, random_state=42)
split_map = {img_id: 'train' for img_id in train_ids}
split_map.update({img_id: 'val' for img_id in val_ids})

print(f"Processing {len(train_ids)} Train images and {len(val_ids)} Validation images...")

for img_id in tqdm(image_ids):
    if img_id not in df_masks.index: continue
    
    split = split_map[img_id]
    img_path = f"{DATA_ROOT}/train/{img_id}.tiff"
    
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width
        # Load mask only once per image
        rle = df_masks.loc[img_id, 'encoding']
        full_mask = rle2mask(rle, (h, w))
        
        # Grid loop
        for y in range(0, h - PATCH_SIZE, STRIDE):
            for x in range(0, w - PATCH_SIZE, STRIDE):
                
                # 1. Read Patch
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                patch = src.read(window=window)
                patch = np.moveaxis(patch, 0, -1) # (C,H,W) -> (H,W,C)
                
                # 2. Tissue Check (Skip empty background)
                if tissue_ratio(patch) < MIN_TISSUE:
                    continue
                
                # 3. Handle Mask Patch
                mask_patch = full_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                # 4. Prepare Filename
                fname = f"{img_id}_{x}_{y}"
                save_path_img = f"{WORK_DIR}/{split}/images/{fname}.jpg"
                save_path_lbl = f"{WORK_DIR}/{split}/labels/{fname}.txt"
                
                # 5. Save Image (Force BGR for OpenCV/YOLO)
                if len(patch.shape) == 2 or patch.shape[-1] == 1:
                    save_img = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
                else:
                    save_img = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(save_path_img, save_img)
                
                # 6. Save Labels
                polys = mask_to_polygons(mask_patch)
                with open(save_path_lbl, "w") as f:
                    if polys:
                        for poly in polys:
                            # Normalize coordinates (0-1)
                            poly = poly.astype(float)
                            poly[:, 0] /= PATCH_SIZE
                            poly[:, 1] /= PATCH_SIZE
                            
                            # Flatten and write
                            coords = " ".join(map(str, poly.flatten()))
                            f.write(f"0 {coords}\n")
                    import random
                    # If the patch has NO glomeruli (background), drop 90% of them
                    if not polys:
                        if random.random() > 0.10: # Keep only 10% of empty background patches
                            continue
                    
        # Cleanup memory immediately
        del full_mask
        gc.collect()


yaml_content = f"""
path: {WORK_DIR}
train: train/images
val: val/images

nc: 1
names: ['glomerulus']
"""

with open(f"{WORK_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)

print("Data YAML created.")


# Load model
model = YOLO("yolo11n-seg.pt") 

# Train
results = model.train(
    data=f"{WORK_DIR}/data.yaml",
    epochs=EPOCHS,
    imgsz=PATCH_SIZE,
    batch=BATCH_SIZE,      # Higher batch size for speed
    patience=15,            # Stop early if no improvement
    optimizer="AdamW",
    lr0=1e-3,
    augment=True,
    workers=4,             # Use multiple cores for data loading
    project="hubmap_project",
    name="run_v1",
    exist_ok=True
)


# Run validation on the separate validation set
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"Seg mAP50-95: {metrics.seg.map}")

# Cleanup (Optional: deletes images to save space for commit)
# shutil.rmtree(WORK_DIR)

