
import os
import sys
import math
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Optional: use cv2 for faster drawing (falls back if not available)
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    print("OpenCV not available; using PIL/matplotlib for visualization.")

# ---- Configure paths ----
DATA_DIR = Path("/kaggle/input/airbus-ship-detection")  # change if your dataset is elsewhere
TRAIN_IMAGES_DIR = DATA_DIR / "train_v2"
TEST_IMAGES_DIR  = DATA_DIR / "test_v2"
CSV_PATH = DATA_DIR / "train_ship_segmentations_v2.csv"  # or 'train_ship_segmentations.csv'

# Output folders for YOLO labels and visualization
OUTPUT_DIR = Path("./airbus_output")
YOLO_DIR = OUTPUT_DIR / "yolo"
LABELS_TRAIN_DIR = YOLO_DIR / "labels" / "train"
LABELS_VAL_DIR   = YOLO_DIR / "labels" / "val"
IMAGES_TRAIN_DIR = YOLO_DIR / "images" / "train"
IMAGES_VAL_DIR   = YOLO_DIR / "images" / "val"
VIZ_DIR = OUTPUT_DIR / "viz"

for p in [LABELS_TRAIN_DIR, LABELS_VAL_DIR, IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, VIZ_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Paths set. You can adjust DATA_DIR/CSV_PATH above if needed.")




def rle_decode(encoded_pixels: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode RLE‑encoded pixels into a 2D mask (H, W) with values {0,1}.
    The encoding is 1‑based, row‑major (as in the Kaggle Airbus dataset).
    """
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if pd.isna(encoded_pixels) or encoded_pixels == "":
        return mask.reshape((h, w), order="F")
    s = list(map(int, encoded_pixels.split()))
    starts = s[0::2]
    lengths = s[1::2]
    starts = [x - 1 for x in starts]  # 1‑based to 0‑based
    for st, ln in zip(starts, lengths):
        mask[st:st+ln] = 1
    # The dataset uses column‑major (Fortran) orientation
    return mask.reshape((h, w), order="F")

def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask to RLE (1‑based, column‑major/Fortran order).
    Returns an empty string when mask has no positive pixels.
    """
    # Ensure binary and Fortran order flatten
    pixels = mask.T.flatten()  # transpose to switch to column‑major indexing
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x_min, y_min, x_max, y_max) for a binary mask.
    Returns None if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return int(x_min), int(y_min), int(x_max), int(y_max)

def bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert pixel bbox to YOLOv5/8 normalized (xc, yc, w, h)."""
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    xc = xmin + w / 2.0
    yc = ymin + h / 2.0
    return xc / img_w, yc / img_h, w / img_w, h / img_h

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    """Convert YOLO normalized bbox back to pixel coords (xmin, ymin, xmax, ymax)."""
    xc *= img_w
    yc *= img_h
    w  *= img_w
    h  *= img_h
    xmin = int(round(xc - w/2))
    ymin = int(round(yc - h/2))
    xmax = int(round(xc + w/2))
    ymax = int(round(yc + h/2))
    return xmin, ymin, xmax, ymax




# Load CSV (try common filenames)
if not CSV_PATH.exists():
    alt = DATA_DIR / "train_ship_segmentations.csv"
    if alt.exists():
        CSV_PATH = alt
    else:
        raise FileNotFoundError(f"Could not find {CSV_PATH} or {alt}. Adjust the CSV_PATH above.")

df = pd.read_csv(CSV_PATH)
print("Annotations shape:", df.shape)
display(df.head())

# Count ships per image
ships_per_image = df.groupby('ImageId')['EncodedPixels'].apply(lambda s: (s.notna() & (s != "")).sum()).reset_index(name='ship_count')
print("Unique images in CSV:", ships_per_image.shape[0])
display(ships_per_image.ship_count.describe())

# Images that are not listed in CSV have zero ships
all_train_images = sorted([p.name for p in TRAIN_IMAGES_DIR.glob("*.jpg")]) if TRAIN_IMAGES_DIR.exists() else []
images_in_csv = set(ships_per_image['ImageId'].tolist())
no_ship_images = [img for img in all_train_images if img not in images_in_csv]
print(f"Train images found on disk: {len(all_train_images)} | Not listed (assumed no-ship): {len(no_ship_images)}")




def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def draw_box(img: np.ndarray, bbox, color=(255, 0, 0), thickness=2) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    out = img.copy()
    if CV2_AVAILABLE:
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    else:
        from PIL import ImageDraw
        pil_img = Image.fromarray(out)
        draw = ImageDraw.Draw(pil_img)
        for t in range(thickness):
            draw.rectangle([x0-t, y0-t, x1+t, y1+t], outline=tuple(color))
        out = np.array(pil_img)
    return out

# Pick a sample that has ships (if possible)
sample_with_ship = None
if len(all_train_images) > 0:
    # Prefer an image present in CSV with at least 1 ship
    for img_id, cnt in ships_per_image.values:
        if cnt > 0:
            sample_with_ship = img_id
            break

if sample_with_ship is None and len(all_train_images) > 0:
    sample_with_ship = all_train_images[0]

if sample_with_ship is not None:
    img_path = TRAIN_IMAGES_DIR / sample_with_ship
    img = load_image(img_path)
    H, W = img.shape[:2]

    # Decode all masks for this image
    encs = df.loc[df['ImageId'] == sample_with_ship, 'EncodedPixels'].dropna().tolist()
    full_mask = np.zeros((H, W), dtype=np.uint8)
    bboxes = []
    for enc in encs:
        m = rle_decode(enc, (H, W))
        full_mask = np.maximum(full_mask, m)
        bb = mask_to_bbox(m)
        if bb is not None:
            bboxes.append(bb)

    # Visualize
    overlay = img.copy()
    if CV2_AVAILABLE:
        colored = np.dstack([full_mask*255, np.zeros_like(full_mask), np.zeros_like(full_mask)]).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.3, 0)
    else:
        red = np.zeros_like(img)
        red[..., 0] = full_mask * 255
        overlay = (0.7 * overlay + 0.3 * red).astype(np.uint8)

    for bb in bboxes:
        overlay = draw_box(overlay, bb, color=(0,255,0), thickness=2)

    plt.figure(figsize=(6,6))
    plt.title(f"Sample: {sample_with_ship} — {len(bboxes)} boxes")
    plt.imshow(overlay)
    plt.axis('off')
else:
    print("No images found to visualize. Check your TRAIN_IMAGES_DIR path.")




def make_split(image_ids: List[str], val_ratio=0.03, seed=42):
    rng = random.Random(seed)
    ids = image_ids.copy()
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]

# Prepare mapping from ImageId -> list of RLEs
grouped = df.groupby('ImageId')['EncodedPixels'].apply(list).to_dict()

# Build the complete list: include also no‑ship images
all_ids = sorted(set(all_train_images))
if not all_ids:
    # Fall back to only the images referenced in CSV (if images aren't on disk here)
    all_ids = sorted(set(df['ImageId'].tolist()))
print(f"Total images considered for split: {len(all_ids)}")

train_ids, val_ids = make_split(all_ids, val_ratio=0.03, seed=2024)
len(train_ids), len(val_ids)



from multiprocessing import Pool, cpu_count

# Assumindo imagens 768x768 (mude se necessário)
IMG_W, IMG_H = 768, 768

def write_yolo_label_file_fast(img_id: str, out_dir: Path, grouped_dict):
    """Escreve arquivo YOLO para uma imagem, pulando imagens sem navios"""
    rles = grouped_dict.get(img_id, [])
    
    # Pular se não houver navios
    if not rles or all(pd.isna(enc) or enc == "" for enc in rles):
        out_path = out_dir / (Path(img_id).stem + ".txt")
        out_path.touch()
        return
    
    lines = []
    for enc in rles:
        if pd.isna(enc) or enc == "":
            continue
        mask = rle_decode(enc, (IMG_H, IMG_W))
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        xc, yc, w, h = bbox_to_yolo(xmin, ymin, xmax, ymax, IMG_W, IMG_H)
        # Clamp
        xc, yc, w, h = map(lambda x: min(max(x,0.0),1.0), (xc, yc, w, h))
        if w <= 0 or h <= 0:
            continue
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # Escreve arquivo
    out_path = out_dir / (Path(img_id).stem + ".txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def process_images_parallel(img_ids, out_dir, grouped_dict, n_cores=None):
    """Processa lista de imagens em paralelo"""
    n_cores = n_cores or max(1, cpu_count()-1)
    args = [(img_id, out_dir, grouped_dict) for img_id in img_ids]
    with Pool(n_cores) as p:
        p.starmap(write_yolo_label_file_fast, args)

# ---- Executar para train/val ----
print("Gerando labels YOLO para train...")
process_images_parallel(train_ids, LABELS_TRAIN_DIR, grouped)

print("Gerando labels YOLO para val...")
process_images_parallel(val_ids, LABELS_VAL_DIR, grouped)

print("YOLO label files escritos:")
print("  Train labels:", len(list(LABELS_TRAIN_DIR.glob('*.txt'))))
print("  Val labels:  ", len(list(LABELS_VAL_DIR.glob('*.txt'))))




dataset_yaml = {
    'path': 'REPLACE_WITH_ABSOLUTE_PATH_TO_yolo_DIR',  # e.g., '/workspace/airbus_output/yolo'
    'train': 'images/train',  # if you linked/copied images
    'val':   'images/val',
    'names': {0: 'ship'}
}

from pathlib import Path
yaml_path = Path('./airbus_output/yolo/dataset.yaml')
yaml_path.parent.mkdir(parents=True, exist_ok=True)

with open(yaml_path, "w") as f:
    import yaml
    yaml.safe_dump(dataset_yaml, f, sort_keys=False)

print("Wrote", yaml_path)
print("Edit 'path' inside dataset.yaml to the absolute path of the yolo dir.")




def load_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = parts
            xc, yc, w, h = map(float, (xc, yc, w, h))
            boxes.append(yolo_to_bbox(xc, yc, w, h, img_w, img_h))
    return boxes

def show_sample(img_dir: Path, labels_dir: Path, img_id: str):
    img_path = img_dir / img_id
    if not img_path.exists():
        print(f"Image not found on disk: {img_path}")
        return
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    label_path = labels_dir / (Path(img_id).stem + ".txt")
    boxes = []
    if label_path.exists():
        boxes = load_yolo_boxes(label_path, W, H)

    out = img.copy()
    for b in boxes:
        out = draw_box(out, b, color=(0,255,0), thickness=2)
    plt.figure(figsize=(6,6))
    plt.title(f"{img_id} — {len(boxes)} boxes")
    plt.imshow(out)
    plt.axis('off')

# Draw a few random validations if available on disk
if TRAIN_IMAGES_DIR.exists() and len(val_ids) > 0:
    import random
    for img_id in random.sample(val_ids, k=min(10, len(val_ids))):
        show_sample(TRAIN_IMAGES_DIR, LABELS_VAL_DIR, img_id)
else:
    print("TRAIN_IMAGES_DIR not found on disk or empty val split — skipping visualization.")


