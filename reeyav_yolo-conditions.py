# Cell 1: Imports, Data Cleaning & YOLOâ€�label prep
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image, UnidentifiedImageError

# â”€â”€â”€ 1) Load raw CSVs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT       = Path('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/')
train      = pd.read_csv(ROOT/'train.csv')
label_df   = pd.read_csv(ROOT/'train_label_coordinates.csv')
series_df  = pd.read_csv(ROOT/'train_series_descriptions.csv')

# â”€â”€â”€ 2) Melt train.csv â†’ long format â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train_long = (
    train
    .melt(id_vars=['study_id'], var_name='cond_lvl', value_name='score')
    .assign(
        lvl1     = lambda df: df['cond_lvl'].str.extract(r'_(l\d+)_')[0].str.upper(),
        lvl2     = lambda df: df['cond_lvl'].str.extract(r'_(s\d+)$')[0].str.upper(),
        cond_key = lambda df: df['cond_lvl'].str.rsplit('_', n=2).str[0]
    )
    .assign(
        level     = lambda df: df['lvl1'] + '/' + df['lvl2'],
        condition = lambda df: df['cond_key'].map({
            'spinal_canal_stenosis':            'Spinal Canal Stenosis',
            'left_neural_foraminal_narrowing':  'Neural Foraminal Narrowing',
            'right_neural_foraminal_narrowing': 'Neural Foraminal Narrowing',
            'left_subarticular_stenosis':       'Subarticular Stenosis',
            'right_subarticular_stenosis':      'Subarticular Stenosis',
        })
    )[['study_id','condition','level','score']]
)

# â”€â”€â”€ 3) Merge everything â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
merged = (
    label_df
    .merge(train_long, on=['study_id','condition','level'], how='left')
    .merge(series_df[['study_id','series_id','series_description']],
           on=['study_id','series_id'], how='left')
)
print("After merge, sample:\n", merged.head())

# â”€â”€â”€ 4) Build the set of ROIâ€�containing stems â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
merged['stem'] = merged.apply(
    lambda r: f"{int(r.study_id)}_{int(r.series_id)}_{int(r.instance_number)}",
    axis=1
)
needed_stems = set(merged['stem'])
print(f"ğŸ”� Will convert only {len(needed_stems)} images that have an ROI")

# â”€â”€â”€ 5) Map stems â†’ DICOM paths & convert only those â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
IMG_DIR = Path('data/images/train/');  IMG_DIR.mkdir(parents=True, exist_ok=True)

stem2path = {}
for dcm_path in ROOT.rglob('*.dcm'):
    p = Path(dcm_path)
    rel = p.relative_to(ROOT).with_suffix('').parts
    stem = "_".join(rel[-3:])
    if stem in needed_stems:
        stem2path[stem] = dcm_path

print(f"ğŸ”� Found {len(stem2path)} DICOMs matching ROI stems")

converted = 0
for stem, dcm_path in tqdm(stem2path.items(),
                           total=len(stem2path),
                           desc="Converting ROIâ†’PNG"):
    png_fp = IMG_DIR/f"{stem}.png"
    if png_fp.exists():
        continue
    ds = pydicom.dcmread(str(dcm_path))
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    arr = ((arr - arr.min())/(arr.max()-arr.min())*255).astype(np.uint8)
    Image.fromarray(arr).save(png_fp)
    converted += 1

print(f"âœ… Converted {converted} ROI DICOMs to PNG in {IMG_DIR}")

# â”€â”€â”€ 6) Gather image shapes from the PNG folder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
png_paths = list(IMG_DIR.glob("*.png"))
print(f"ğŸ“‚ Found {len(png_paths)} PNG files in {IMG_DIR}")

image_shapes = {}
skipped = []
for p in tqdm(png_paths, desc="Loading image shapes"):
    try:
        H, W = Image.open(p).size[::-1]
        image_shapes[p.stem] = (H, W)
    except (UnidentifiedImageError, OSError):
        skipped.append(p.name)

print(f"âš ï¸� Skipped {len(skipped)} unreadable PNGs")
print(f"âœ… Collected shapes for {len(image_shapes)} images")

# â”€â”€â”€ 7) Write YOLO-format label files â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LABEL_DIR = Path('data/labels/train/');  LABEL_DIR.mkdir(parents=True, exist_ok=True)

ROI_SIZE = 128  # pixels

def get_cls_id(cond_text: str) -> int:
    """Return 0/1/2 based on the substring in the original condition field."""
    if 'Spinal Canal' in cond_text:
        return 0
    if 'Neural Foraminal' in cond_text:
        return 1
    if 'Subarticular' in cond_text:
        return 2
    raise ValueError(f"Unknown condition text: {cond_text!r}")

def write_yolo(stem, cls_id, x_c, y_c, w_n, h_n):
    with open(LABEL_DIR/f"{stem}.txt", 'a') as f:
        f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

for _, row in tqdm(merged.iterrows(),
                   total=len(merged),
                   desc="Writing YOLO labels"):
    stem = row.stem
    if stem not in image_shapes:
        continue
    H, W = image_shapes[stem]
    x_c, y_c = row.x / W, row.y / H
    w_n = h_n = ROI_SIZE / W
    cls_id = get_cls_id(row.condition)
    write_yolo(stem, cls_id, x_c, y_c, w_n, h_n)

print("âœ… Done writing YOLO labels to", LABEL_DIR)



!zip -r /kaggle/working/export_all.zip /kaggle/working/data /kaggle/working/runs



from pathlib import Path
import pandas as pd

# â”€â”€â”€ 1. Load CSVs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/')
train = pd.read_csv(ROOT / 'train.csv')
label_df = pd.read_csv(ROOT / 'train_label_coordinates.csv')
series_df = pd.read_csv(ROOT / 'train_series_descriptions.csv')

# â”€â”€â”€ 2. Melt train.csv â†’ long format â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train_long = train.melt(id_vars='study_id', var_name='cond_lvl', value_name='score')

# â”€â”€â”€ 3. Extract condition and levels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Extract condition prefix
train_long['cond_key'] = train_long['cond_lvl'].str.rsplit('_', n=2).str[0]

# Extract levels, including cases like l5_s1 or l2_l3
train_long['level'] = train_long['cond_lvl'].str.extract(r'_(l\d+_[ls]\d+)$')[0]
train_long['level'] = train_long['level'].str.upper().str.replace('_', '/')

# ğŸ”� Normalize conditions (combine left/right into one for 2 conditions)
def map_condition(key):
    if 'spinal_canal_stenosis' in key:
        return 'Spinal Canal Stenosis'
    elif 'neural_foraminal_narrowing' in key:
        return 'Neural Foraminal Narrowing'
    elif 'subarticular_stenosis' in key:
        return 'Subarticular Stenosis'
    else:
        return None

train_long['condition'] = train_long['cond_key'].map(map_condition)

# Drop rows with unmapped conditions or levels
train_long = train_long.dropna(subset=['condition', 'level'])

# â”€â”€â”€ 4. Normalize levels & condition in label_df â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
label_df['level'] = label_df['level'].str.upper().str.strip()
train_long['level'] = train_long['level'].str.upper().str.strip()

label_df['condition'] = label_df['condition'].replace({
    'Left Neural Foraminal Narrowing': 'Neural Foraminal Narrowing',
    'Right Neural Foraminal Narrowing': 'Neural Foraminal Narrowing',
    'Left Subarticular Stenosis': 'Subarticular Stenosis',
    'Right Subarticular Stenosis': 'Subarticular Stenosis',
})

# â”€â”€â”€ 5. Merge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
merged = (
    label_df
    .merge(train_long[['study_id', 'condition', 'level', 'score']],
           on=['study_id', 'condition', 'level'], how='left')
    .merge(series_df[['study_id', 'series_id', 'series_description']],
           on=['study_id', 'series_id'], how='left')
)

# â”€â”€â”€ 6. Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("âœ… Merged data shape:", merged.shape)
missing = merged['score'].isna().sum()
print(f"âš ï¸� Missing scores: {missing} (should now be close to 0)")
print("ğŸ•µï¸� Remaining unmatched rows (sample):")
print(merged[merged['score'].isna()][['study_id', 'condition', 'level']].head())

# Optional check
print("ğŸ§ª Unique conditions:", merged['condition'].unique())
print("ğŸ§ª Unique levels:", sorted(merged['level'].unique()))



merged = merged.dropna(subset=['score'])
print("After merge, sample:\n", merged.head())


import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# â”€â”€â”€ 1. Prepare â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
merged = merged.dropna(subset=['score'])  # Drop rows with no label
merged['stem'] = merged.apply(
    lambda r: f"{int(r.study_id)}_{int(r.series_id)}_{int(r.instance_number)}", axis=1
)

needed_stems = set(merged['stem'])
print(f"ğŸ”� Will convert only {len(needed_stems)} images that have an ROI")

# â”€â”€â”€ 2. DICOM to PNG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT = Path('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/')
IMG_DIR = Path('data/images/train/'); IMG_DIR.mkdir(parents=True, exist_ok=True)

stem2path = {}
for dcm_path in ROOT.rglob('*.dcm'):
    p = Path(dcm_path)
    rel = p.relative_to(ROOT).with_suffix('').parts
    stem = "_".join(rel[-3:])
    if stem in needed_stems:
        stem2path[stem] = dcm_path

print(f"ğŸ”� Found {len(stem2path)} DICOMs matching ROI stems")

converted = 0
for stem, dcm_path in tqdm(stem2path.items(), desc="Converting ROIâ†’PNG"):
    png_fp = IMG_DIR / f"{stem}.png"
    if png_fp.exists():
        continue
    try:
        ds = pydicom.dcmread(str(dcm_path))
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
        Image.fromarray(arr).save(png_fp)
        converted += 1
    except Exception as e:
        print(f"â�Œ Skipped {dcm_path}: {e}")
print(f"âœ… Converted {converted} DICOMs to PNG")

# â”€â”€â”€ 3. Image size lookup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
png_paths = list(IMG_DIR.glob("*.png"))
image_shapes = {}
skipped = []
for p in tqdm(png_paths, desc="Loading image shapes"):
    try:
        H, W = Image.open(p).size[::-1]
        image_shapes[p.stem] = (H, W)
    except (UnidentifiedImageError, OSError):
        skipped.append(p.name)
print(f"âš ï¸� Skipped {len(skipped)} unreadable PNGs")
print(f"âœ… Collected shapes for {len(image_shapes)} images")

# â”€â”€â”€ 4. YOLO Label Generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LABEL_DIR = Path('data/labels/train/'); LABEL_DIR.mkdir(parents=True, exist_ok=True)
ROI_SIZE = 128  # You can change this to match cropping logic

def get_cls_id(cond_text: str) -> int:
    if cond_text == 'Spinal Canal Stenosis':
        return 0
    elif cond_text == 'Neural Foraminal Narrowing':
        return 1
    elif cond_text == 'Subarticular Stenosis':
        return 2
    raise ValueError(f"Unknown condition: {cond_text!r}")

def write_yolo(stem, cls_id, x_c, y_c, w_n, h_n):
    with open(LABEL_DIR / f"{stem}.txt", 'a') as f:
        f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Writing YOLO labels"):
    stem = row.stem
    if stem not in image_shapes:
        continue
    H, W = image_shapes[stem]
    x_c, y_c = row.x / W, row.y / H
    w_n = h_n = ROI_SIZE / W
    cls_id = get_cls_id(row.condition)
    write_yolo(stem, cls_id, x_c, y_c, w_n, h_n)

print(f"âœ… YOLO labels saved to {LABEL_DIR}")


# Cell 2: Build train/val splits & data.yaml for YOLOv8

import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1) Gather all the stems you converted
IMG_DIR = Path('data/images/train/')
all_stems = [p.stem for p in IMG_DIR.glob("*.png")]
print(f"ğŸ–¼ï¸�  {len(all_stems)} total images found")

# 2) Split into train / val
train_stems, val_stems = train_test_split(
    all_stems,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
print(f"ğŸš‚  {len(train_stems)} train  |  {len(val_stems)} val")

# 3) Write file lists
with open('data/train.txt','w') as f:
    for s in train_stems:
        f.write(f"{IMG_DIR}/{s}.png\n")

with open('data/val.txt','w') as f:
    for s in val_stems:
        f.write(f"{IMG_DIR}/{s}.png\n")

print("âœ… train.txt & val.txt saved")

# 4) Create data.yaml
import yaml

data = {
    'path': '.',            # project root
    'train': 'data/train.txt',
    'val':   'data/val.txt',
    'nc':    3,             # number of classes
    'names': [
        'Spinal Canal Stenosis',
        'Neural Foraminal Narrowing',
        'Subarticular Stenosis'
    ]
}

with open('data/data.yaml','w') as f:
    yaml.dump(data, f, sort_keys=False)

print("âœ… data/data.yaml created:")
print(yaml.dump(data, sort_keys=False))



# Cell 3: Install YOLOv8 dependency
!pip install ultralytics --upgrade



from pathlib import Path

LABEL_DIR = Path("data/labels/train/")

log = []

for label_file in LABEL_DIR.glob("*.txt"):
    with open(label_file, "r") as f:
        lines = f.readlines()

    unique_lines = sorted(set(line.strip() for line in lines if line.strip()))
    removed = len(lines) - len(unique_lines)

    # Write cleaned labels to new directory
    with open(label_file, "w") as f:
        for line in unique_lines:
            f.write(line + "\n")

    if removed > 0:
        log.append((label_file.name, removed))

# ğŸ“� Print a summary log
print("âœ… Cleaned YOLO label files (in-place).")



# Cell 3 (updated): 5-Fold Cross-Validation Training per Condition with perâ€�epoch checkpoints

import yaml
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import StratifiedKFold

# Ensure `merged` from Cell 1 is in scope
conditions = {
    'Spinal Canal Stenosis':      'Spinal_Canal_Stenosis',
    'Neural Foraminal Narrowing': 'Neural_Foraminal_Narrowing',
    'Subarticular Stenosis':      'Subarticular_Stenosis'
}

IMG_DIR   = Path('data/images/train/')
BASE_DATA = Path('data')

for cond_text, cond_folder in conditions.items():
    df_cond = merged[merged['condition'].str.contains(cond_text)].copy()

    class_names = sorted(df_cond['level'].unique())
    cls_map = {lvl: i for i, lvl in enumerate(class_names)}
    df_cond['class_id'] = df_cond['level'].map(cls_map)

    stems = df_cond['stem'].tolist()
    labels = df_cond['class_id'].tolist()

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(stems, labels)):
        fold_dir = BASE_DATA/cond_folder/f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        # write train/val lists
        train_txt = fold_dir/'train.txt'
        val_txt   = fold_dir/'val.txt'
        with open(train_txt, 'w') as f:
            for i in train_idx:
                f.write(f"{IMG_DIR}/{stems[i]}.png\n")
        with open(val_txt, 'w') as f:
            for i in val_idx:
                f.write(f"{IMG_DIR}/{stems[i]}.png\n")

        # write data.yaml
        data_cfg = {
            'path': '.', 
            'train': str(train_txt), 
            'val':   str(val_txt), 
            'nc':    len(class_names), 
            'names': class_names
        }
        data_yaml = fold_dir/'data.yaml'
        with open(data_yaml, 'w') as f:
            yaml.dump(data_cfg, f, sort_keys=False)

        # train and save a checkpoint every epoch
        print(f"\nâ–¶ï¸�  Training {cond_folder}, fold {fold}")
        model = YOLO('yolov8n.pt')
        model.train(
            data=str(data_yaml),
            epochs=2,
            imgsz=512,
            batch=16,
            project='runs/detect',
            name=f"{cond_folder}_fold{fold}",
            save_period=1,    # save epoch_{i}.pt each epoch
            verbose=True,
            plots=True
        )



# Cell 4 (Debug): Inspect your runs/detect folder for .pt weights

import os
from pathlib import Path

RUN_DIR = Path('runs/detect')
if not RUN_DIR.exists():
    raise RuntimeError(f"{RUN_DIR} does not exist! Did you train your models?")

print(f"Directory tree under {RUN_DIR}:\n")
for root, dirs, files in os.walk(RUN_DIR):
    level = root.replace(str(RUN_DIR), '').count(os.sep)
    indent = '    ' * level
    print(f"{indent}- {Path(root).name}/")
    for fname in files:
        print(f"{indent}    - {fname}")



from pathlib import Path
from PIL import Image, UnidentifiedImageError

IMG_DIR = Path("data/images/train/")
bad_files = []

for img_path in IMG_DIR.glob("*.png"):
    try:
        _ = Image.open(img_path).verify()
    except (UnidentifiedImageError, OSError):
        bad_files.append(img_path)

print(f"â�Œ Found {len(bad_files)} corrupted files.")

# Delete them
for f in bad_files:
    print(f"Removing: {f}")
    f.unlink()

print("âœ… Corrupted PNGs removed.")



# Cell 4: Inference & Crop ROIs (clean, per-fold, only split images, deduplicated)

import os
import sys
import contextlib
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

conditions = {
    'Spinal_Canal_Stenosis':      'Spinal_Canal_Stenosis',
    'Neural_Foraminal_Narrowing': 'Neural_Foraminal_Narrowing',
    'Subarticular_Stenosis':      'Subarticular_Stenosis'
}

IMG_DIR = Path('data/images/train/')
CROPS_BASE = Path('data/crops/')
CROPS_BASE.mkdir(parents=True, exist_ok=True)

for cond_folder in conditions.values():
    fold_base = Path(f'data/{cond_folder}')
    if not fold_base.exists():
        continue

    fold_dirs = sorted(f for f in fold_base.iterdir() if f.name.startswith('fold_'))
    for fold_dir in fold_dirs:
        fold_idx = fold_dir.name.split('_')[-1]

        train_path = fold_dir / 'train.txt'
        val_path   = fold_dir / 'val.txt'
        if not train_path.exists() or not val_path.exists():
            print(f"âš ï¸� Skipping {fold_dir}: missing train/val split files.")
            continue

        with open(train_path) as f1, open(val_path) as f2:
            crop_list = [
                Path(p.strip()) for p in f1.readlines() + f2.readlines()
                if Path(p.strip()).exists()
            ]



        yolo_weights = Path(f'runs/detect/{cond_folder}_fold{fold_idx}/weights')
        best_ckpt = yolo_weights / 'best.pt'
        if not best_ckpt.exists():
            print(f"âš ï¸� Skipping {cond_folder}_fold{fold_idx}: no best.pt found.")
            continue

        with suppress_output():
            model = YOLO(str(best_ckpt))

        out_dir = CROPS_BASE / cond_folder / f'fold_{fold_idx}'
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_fp in tqdm(crop_list, desc=f"Cropping {cond_folder}_fold{fold_idx}"):
            with suppress_output():
                results = model(img_fp, imgsz=512, conf=0.3, verbose=False)

            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            if boxes.size == 0:
                continue

            im = Image.open(img_fp)
            stem = img_fp.stem
            seen_boxes = []

            for j, (box, conf) in enumerate(sorted(zip(boxes, confs), key=lambda x: -x[1])):
                x1, y1, x2, y2 = map(int, box)

                # Check for near-duplicate box
                duplicate = False
                for px1, py1, px2, py2 in seen_boxes:
                    if abs(x1 - px1) < 5 and abs(y1 - py1) < 5 and abs(x2 - px2) < 5 and abs(y2 - py2) < 5:
                        duplicate = True
                        break

                if duplicate:
                    continue

                seen_boxes.append((x1, y1, x2, y2))
                crop = im.crop((x1, y1, x2, y2))
                crop.save(out_dir / f"{stem}_roi{j}.png")





!zip -r /kaggle/working/export_all.zip /kaggle/working/data /kaggle/working/runs



# Add to end of Cell 1
merged.to_pickle('data/merged.pkl')



# Cell 5: Train severity classifiers for all 3 conditions from ROI crops

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

# Settings
CROP_ROOT = Path('data/crops')
MERGED = merged  # from Cell 1
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_MAP = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
IDX2LABEL = {v: k for k, v in LABEL_MAP.items()}

conditions = {
    'Spinal_Canal_Stenosis':      'Spinal Canal Stenosis',
    'Neural_Foraminal_Narrowing': 'Neural Foraminal Narrowing',
    'Subarticular_Stenosis':      'Subarticular Stenosis'
}

# Dataset class
class CropDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row['label_id'])

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

for crop_folder, original_condition in conditions.items():
    crop_dir = CROP_ROOT / crop_folder
    if not crop_dir.exists():
        print(f"âš ï¸� Skipping {crop_folder}: no crop directory found.")
        continue

    # Step 1: Load (path, label_id) pairs
    records = []
    for crop_fp in crop_dir.rglob("*.png"):
        stem = crop_fp.stem.split('_roi')[0]
        row = MERGED[(MERGED['stem'] == stem) &
                     (MERGED['condition'].str.contains(original_condition))]

        if row.empty or pd.isna(row.iloc[0]['score']):
            continue

        severity = row.iloc[0]['score']
        if severity not in LABEL_MAP:
            continue

        records.append({'path': str(crop_fp), 'label_id': LABEL_MAP[severity]})

    df = pd.DataFrame(records)
    if df.empty:
        print(f"âš ï¸� Skipping {crop_folder}: no labeled crops found.")
        continue

    print(f"\nğŸ§® Class distribution for {crop_folder}: {df['label_id'].value_counts().to_dict()}")

    # Step 2: Train/val split
    train_df, val_df = train_test_split(
        df, stratify=df['label_id'], test_size=0.2, random_state=42
    )

    train_loader = DataLoader(CropDataset(train_df, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CropDataset(val_df, transform), batch_size=BATCH_SIZE)

    # Step 3: Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_MAP))
    model = model.to(DEVICE)

    # Step 4: Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_acc = 0.0

    print(f"\nğŸš€ Training classifier for {crop_folder} with {len(df)} samples")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            _, preds = out.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item() * imgs.size(0)
        acc = correct / total
        print(f"Epoch {epoch:02d} - Train Acc: {acc:.4f} | Loss: {loss_sum/total:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            all_preds = []
            all_labels = []
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                _, preds = out.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            val_acc = correct / total
            print(f"            Val Acc:   {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = Path(f"runs/classify/{crop_folder}_best.pth")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(model.state_dict(), save_path)

    print(f"\nâœ… Best model saved for {crop_folder}: Val Acc = {best_acc:.4f}")

    # ğŸ”� Final Classification Report
    print("\nğŸ“Š Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=LABEL_MAP.keys()))

    # ğŸ”� Final Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_MAP.keys(), yticklabels=LABEL_MAP.keys())
    plt.title(f"Confusion Matrix for {crop_folder}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()





