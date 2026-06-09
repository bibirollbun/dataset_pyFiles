# ===============================================================
# ğŸ“¦ VinBigData DICOM â†’ JPG + YOLO Label Converter
# Author: ChatGPT (GPT-5) | Verified for Kaggle 2025
# ===============================================================
!pip install -q pydicom tqdm pillow


import os, random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import pydicom
from tqdm import tqdm

# =====================
# CONFIG
# =====================
OUT_IMG_SIZE = (512, 512)   # Resize to 512x512
VAL_FRAC = 0.1              # 10% validation split
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =====================
# INPUT PATHS
# =====================
ROOT_IN = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection")
TRAIN_CSV = ROOT_IN / "train.csv"
SAMPLE_SUB = ROOT_IN / "sample_submission.csv"
TRAIN_DICOM_DIR = ROOT_IN / "train"
TEST_DICOM_DIR = ROOT_IN / "test"

# =====================
# OUTPUT PATHS
# =====================
WORK_DIR = Path("/kaggle/working/vindr")
IMG_DIR = WORK_DIR / "images"
LAB_DIR = WORK_DIR / "labels"
for split in ["train", "val", "test"]:
    (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
    (LAB_DIR / split).mkdir(parents=True, exist_ok=True)

# =====================
# LOAD TRAIN DATA
# =====================
df = pd.read_csv(TRAIN_CSV)
print("ğŸ“„ train.csv rows:", len(df))
print("Unique images in CSV:", df['image_id'].nunique())

# Ensure class_id is int
df["class_id"] = df["class_id"].astype(int)

# =====================
# SPLIT IMAGES
# =====================
all_images = sorted([f.stem for f in TRAIN_DICOM_DIR.glob("*.dicom")])
random.shuffle(all_images)
n_val = int(len(all_images) * VAL_FRAC)
val_ids = set(all_images[:n_val])
train_ids = set(all_images[n_val:])
print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

# =====================
# DICOM to JPEG helper
# =====================
def dicom_to_pil(dicom_path):
    ds = pydicom.dcmread(str(dicom_path))
    img = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(img, (0.5, 99.5))
    img = np.clip(img, lo, hi)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")

# =====================
# YOLO label writer
# =====================
def write_yolo_label(image_id, boxes_for_image, out_label_path, img_w, img_h):
    lines = []
    for _, row in boxes_for_image.iterrows():
        cls = int(row["class_id"])
        # skip "No finding" rows â€” no box, but keep empty file
        if cls == 14:
            continue
        # skip NaN coords
        if any(pd.isna(row[c]) for c in ["x_min","y_min","x_max","y_max"]):
            continue
        x_min, y_min, x_max, y_max = row[["x_min","y_min","x_max","y_max"]]
        if x_max <= x_min or y_max <= y_min:
            continue
        # normalize
        cx = (x_min + x_max) / 2 / img_w
        cy = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    # write label file
    with open(out_label_path, "w") as f:
        if lines:
            f.write("\n".join(lines))
        else:
            f.write("")  # empty = No Finding

# =====================
# CONVERSION LOOP
# =====================
def convert_split(ids_set, split_name):
    count, skipped = 0, 0
    for img_id in tqdm(ids_set, desc=f"Converting {split_name}"):
        dicom_path = TRAIN_DICOM_DIR / f"{img_id}.dicom"
        if not dicom_path.exists():
            skipped += 1
            continue
        try:
            pil = dicom_to_pil(dicom_path)
        except Exception as e:
            print(f"âš ï¸� Error reading {img_id}: {e}")
            skipped += 1
            continue

        if OUT_IMG_SIZE:
            pil = pil.resize(OUT_IMG_SIZE)
        out_img_path = IMG_DIR / split_name / f"{img_id}.jpg"
        out_lbl_path = LAB_DIR / split_name / f"{img_id}.txt"
        pil.save(out_img_path, quality=95)

        boxes = df[df["image_id"] == img_id]
        write_yolo_label(img_id, boxes, out_lbl_path, pil.width, pil.height)
        count += 1

    print(f"âœ… {split_name} done: {count} converted, {skipped} skipped")

def convert_test():
    sample_sub = pd.read_csv(SAMPLE_SUB)
    test_ids = sample_sub["image_id"].tolist()
    for img_id in tqdm(test_ids, desc="Converting test"):
        dicom_path = TEST_DICOM_DIR / f"{img_id}.dicom"
        if not dicom_path.exists():
            continue
        try:
            pil = dicom_to_pil(dicom_path)
        except Exception:
            continue
        if OUT_IMG_SIZE:
            pil = pil.resize(OUT_IMG_SIZE)
        out_img_path = IMG_DIR / "test" / f"{img_id}.jpg"
        out_lbl_path = LAB_DIR / "test" / f"{img_id}.txt"
        pil.save(out_img_path, quality=95)
        open(out_lbl_path, 'w').close()

# =====================
# RUN CONVERSIONS
# =====================
convert_split(train_ids, "train")
convert_split(val_ids, "val")
convert_test()

# =====================
# YAML for YOLOv8
# =====================
yaml_path = WORK_DIR / "vinbigdata_yolo.yaml"
class_names = [
    "Aortic_enlargement","Atelectasis","Calcification","Cardiomegaly",
    "Consolidation","ILD","Infiltration","Lung_Opacity","Nodule_Mass",
    "Other_lesion","Pleural_effusion","Pleural_thickening",
    "Pneumothorax","Pulmonary_fibrosis","No_finding"
]
with open(yaml_path, "w") as f:
    f.write(f"path: {WORK_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write("names: " + str(class_names) + "\n")

print(f"\nâœ… Conversion Completed!")
print(f"YAML file saved to: {yaml_path}")



import os, numpy as np

label_dir = "/kaggle/working/vindr/labels/train"
ids = set()
for f in os.listdir(label_dir):
    with open(os.path.join(label_dir, f)) as fp:
        for line in fp:
            if line.strip():
                cid = int(line.split()[0])
                ids.add(cid)
print("Unique class IDs in labels:", sorted(list(ids)))
print("Expected 0â€“13 for diseases, 14 empty = No finding handled via empty files.")



import shutil
from IPython.display import FileLink, display

# 1ï¸�âƒ£ Define the folder or files you want to zip
# Example: 'output' is your folder with generated files
folder_to_zip = '/kaggle/working/'
zip_filename = 'results.zip'

# 2ï¸�âƒ£ Create the ZIP file
shutil.make_archive('results', 'zip', folder_to_zip)

# 3ï¸�âƒ£ Display download link inside notebook
display(FileLink(zip_filename))

# âœ… Optional (for Google Colab-like behavior, may work in some browsers)
# from IPython.display import Javascript
# display(Javascript('window.open("/kaggle/working/results.zip")'))





