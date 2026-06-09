# Cell 1: Install all dependencies once
%pip install ultralytics==8.0.111 torch torchvision pandas numpy pydicom albumentations scikit-learn tqdm



# Point to the RSNA CSVs in your Kaggle dataset
Data_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'

import os
import pandas as pd

# 1) Read train_label_coordinates.csv and train_series_descriptions.csv
train_label_coordinates = pd.read_csv(os.path.join(Data_path, 'train_label_coordinates.csv'))
train_series_descriptions = pd.read_csv(os.path.join(Data_path, 'train_series_descriptions.csv'))

# 2) Merge on ['study_id', 'series_id'] to pull in series_description
merged_csv = pd.merge(
    train_label_coordinates,
    train_series_descriptions[['study_id', 'series_id', 'series_description']],
    on=['study_id', 'series_id'],
    how='left'
)

# 3) Read train.csv (holds the severity scores)
train_df = pd.read_csv(os.path.join(Data_path, 'train.csv'))

# 4) Helper to look up the correct column (e.g. 'neural_foraminal_narrowing_l1_l2') and fetch its value
def get_score(row):
    study_id = row['study_id']
    condition = row['condition']
    level = row['level']  # like "L1/L2" or "R3/R4"

    # Split "L1/L2" → "L1", "L2" so we can build a column name same as in train.csv
    level_1, level_2 = level.split('/')
    condition_level = f"{condition}_{level_1}_{level_2}".replace(' ', '_').lower()
    # e.g., "neural_foraminal_narrowing_l1_l2"

    if condition_level in train_df.columns and study_id in train_df['study_id'].values:
        return train_df.loc[train_df['study_id'] == study_id, condition_level].values[0]
    else:
        return None

# 5) Apply it to every row
merged_csv['score'] = merged_csv.apply(get_score, axis=1)

# 6) Save the merged result into /kaggle/working so downstream steps can consume it
out_path = '/kaggle/working/dataset_description.csv'
merged_csv.to_csv(out_path, index=False)
print(f"✅ Wrote merged CSV with scores to: {out_path}")



import os
import pandas as pd

# 1) Path to the merged file you created earlier
input_csv = '/kaggle/working/dataset_description.csv'

# 2) Load the full dataset_description.csv
df = pd.read_csv(input_csv)

# 3) Define which 'condition' values belong to each output group
condition_groups = {
    'Spinal Canal Stenosis': ['Spinal Canal Stenosis'],
    'Neural Foraminal Narrowing': ['Right Neural Foraminal Narrowing', 'Left Neural Foraminal Narrowing'],
    'Subarticular Stenosis': ['Right Subarticular Stenosis', 'Left Subarticular Stenosis']
}

# 4) For each group, filter and save a separate CSV
for group_name, conditions in condition_groups.items():
    filtered_df = df[df['condition'].isin(conditions)].copy()
    # Make a filesystem‐friendly name, e.g. "Spinal_Canal_Stenosis.csv"
    out_name = group_name.replace(' ', '_') + '.csv'
    out_path = os.path.join('/kaggle/working', out_name)
    filtered_df.to_csv(out_path, index=False)
    print(f"→ Wrote {len(filtered_df)} rows to {out_name}")

print("✅ Done splitting into three CSVs.")



import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def cross_validation_2fold(csv_path, output_name):
    """
    Reads a condition‐specific CSV, creates 'class_id' from condition+level,
    performs a 2‐fold stratified split, and writes out a new CSV with a 'fold' column.
    """
    df = pd.read_csv(csv_path)

    # Create a combined “condition_level” string
    df['condition_level'] = df['condition'] + '_' + df['level']

    # Convert to numeric class IDs for stratification
    df['class_id'] = df['condition_level'].astype('category').cat.codes

    # Use 2 splits instead of 5
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    df['fold'] = -1
    for fold_number, (_, val_idx) in enumerate(skf.split(df, df['class_id'])):
        df.loc[val_idx, 'fold'] = fold_number

    output_path = os.path.join('/kaggle/working', f"{output_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved 2‐fold CSV to: {output_path}")


# Paths to the condition‐specific CSVs (from the “split by condition” step)
spinal_csv = '/kaggle/working/Spinal_Canal_Stenosis.csv'
neural_csv = '/kaggle/working/Neural_Foraminal_Narrowing.csv'
subart_csv = '/kaggle/working/Subarticular_Stenosis.csv'

# Run 2‐fold stratified splitting for each condition
cross_validation_2fold(spinal_csv, 'Spinal_Canal_Stenosis_2folds')
cross_validation_2fold(neural_csv, 'Neural_Foraminal_Narrowing_2folds')
cross_validation_2fold(subart_csv, 'Subarticular_Stenosis_2folds')



import os

print(os.listdir('/kaggle/working'))
# You should see:
# [
#   'dataset_description.csv',
#   'Spinal_Canal_Stenosis.csv',
#   'Neural_Foraminal_Narrowing.csv',
#   'Subarticular_Stenosis.csv',
#   'Spinal_Canal_Stenosis_folds.csv',
#   'Neural_Foraminal_Narrowing_folds.csv',
#   'Subarticular_Stenosis_folds.csv',
#   … (any other files)
# ]



import os
import cv2
import yaml
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import pydicom


class DetectorDataPreparation:
    def __init__(
        self,
        dataset_directory='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images',
        csv_path='',                     # e.g. '/kaggle/working/Spinal_Canal_Stenosis_2folds.csv'
        condition_level_classes={},
        condition_name='',
        val_fold=0,                      # will be 0 or 1
        width_box=16,
    ):
        self.dataset_directory = dataset_directory
        self.csv_path = csv_path
        self.condition_level_classes = condition_level_classes
        self.condition_name = condition_name
        self.val_fold = val_fold
        self.width_box = width_box

        self.save_directory = Path(f'/kaggle/working/{self.condition_name}')
        self._create_folders()
        self._read_cross_validation()
        self._dicom_to_png(self.training_data, self.train_image_path)
        self._save_height_width_csv()
        self._create_yolo_labels(self.training_data, self.train_labels_path)
        self._dicom_to_png(self.validation_data, self.val_images_path)
        self._save_height_width_csv()
        self._create_yolo_labels(self.validation_data, self.val_labels_path)
        self._create_yaml_file()


    def _create_folders(self):
        base = self.save_directory
        base.mkdir(parents=True, exist_ok=True)
        self.fold_path = base / f'fold_{self.val_fold}'
        self.fold_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.fold_path / 'datasets'
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        (self.dataset_path / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

        self.train_image_path = self.dataset_path / 'train' / 'images'
        self.train_labels_path = self.dataset_path / 'train' / 'labels'
        self.val_images_path   = self.dataset_path / 'val' / 'images'
        self.val_labels_path   = self.dataset_path / 'val' / 'labels'


    def _read_cross_validation(self):
        df = pd.read_csv(self.csv_path)
        self.validation_data = df[df['fold'] == self.val_fold].reset_index(drop=True)
        print(f"[{self.condition_name}][fold {self.val_fold}]  Validation rows: {len(self.validation_data)}")
        self.training_data = df[df['fold'] != self.val_fold].reset_index(drop=True)
        print(f"[{self.condition_name}][fold {self.val_fold}]  Training rows:   {len(self.training_data)}")


    def _read_dicom(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255.0
        img = np.stack([img]*3, axis=-1).astype('uint8')
        return img


    def _dicom_to_png(self, df: pd.DataFrame, image_directory: Path):
        self.height_width_info = []
        for study_id, study_grp in df.groupby('study_id'):
            for series_id, series_grp in study_grp.groupby('series_id'):
                series_folder = os.path.join(
                    self.dataset_directory,
                    str(study_id),
                    str(series_id)
                )
                if not os.path.isdir(series_folder):
                    print(f"⚠️ Missing folder: {series_folder}")
                    continue

                inst_map = {}
                for fname in os.listdir(series_folder):
                    if not fname.lower().endswith('.dcm'):
                        continue
                    full_path = os.path.join(series_folder, fname)
                    try:
                        ds = pydicom.dcmread(full_path, stop_before_pixels=True)
                        inst_num = int(ds.InstanceNumber)
                        inst_map[inst_num] = full_path
                    except Exception:
                        continue

                instance_list = series_grp['instance_number'].unique().tolist()
                if len(inst_map) == 0:
                    print(f"⚠️ No .dcm files found in {series_folder}")
                    continue

                first_inst = instance_list[0]
                if first_inst not in inst_map:
                    first_inst = next(iter(inst_map.keys()))
                first_path = inst_map[first_inst]
                img0 = self._read_dicom(first_path)
                h, w, _ = img0.shape
                self.height_width_info.append({
                    'study_id': study_id,
                    'series_id': series_id,
                    'height': h,
                    'width': w
                })

                for inst in instance_list:
                    if inst not in inst_map:
                        print(f"⚠️ InstanceNumber {inst} not found in {series_folder}")
                        continue
                    dcm_path = inst_map[inst]
                    img = self._read_dicom(dcm_path)
                    out_png = image_directory / f"{study_id}_{series_id}_{inst}.png"
                    cv2.imwrite(str(out_png), img)

        print(f"[{self.condition_name}][fold {self.val_fold}]  Saved PNGs to {image_directory}")


    def _save_height_width_csv(self):
        hw_path = self.fold_path / f"{self.condition_name}_height_weight.csv"
        with open(hw_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['study_id', 'series_id', 'height', 'width'])
            writer.writeheader()
            writer.writerows(self.height_width_info)

        fold_df = pd.read_csv(self.csv_path)
        hw_df   = pd.read_csv(hw_path)
        merged_df = pd.merge(
            fold_df,
            hw_df[['study_id', 'series_id', 'height', 'width']],
            on=['study_id', 'series_id'],
            how='left'
        )
        merged_df.to_csv(hw_path, index=False)
        print(f"[{self.condition_name}][fold {self.val_fold}]  Height/width CSV saved to {hw_path}")


    def _find_class_label(self, condition, level):
        cond_norm = condition.replace(' ', '_')
        lvl_norm  = level.replace('/', '_')
        key       = f"{cond_norm}_{lvl_norm}"
        return self.condition_level_classes[key]


    def _create_yolo_labels(self, df: pd.DataFrame, labels_directory: Path):
        merged_csv = pd.read_csv(self.fold_path / f"{self.condition_name}_height_weight.csv")

        for study_id, study_grp in merged_csv.groupby('study_id'):
            for series_id, series_grp in study_grp.groupby('series_id'):
                for inst_num, inst_grp in series_grp.groupby('instance_number'):
                    labels = []
                    h = inst_grp['height'].iloc[0]
                    w = inst_grp['width'].iloc[0]

                    for _, row in inst_grp.iterrows():
                        cond = row['condition']
                        lvl  = row['level']
                        x    = row['x']
                        y    = row['y']
                        class_id = self._find_class_label(cond, lvl)

                        x_norm = float(x) / w
                        y_norm = float(y) / h
                        box_w  = float(self.width_box) / w
                        box_h  = float(self.width_box) / h

                        labels.append((class_id, x_norm, y_norm, box_w, box_h))

                    txt_path = labels_directory / f"{study_id}_{series_id}_{inst_num}.txt"
                    with open(txt_path, 'w') as f:
                        for (cid, xn, yn, bw, bh) in labels:
                            f.write(f"{cid} {xn:.6f} {yn:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"[{self.condition_name}][fold {self.val_fold}]  Wrote YOLO labels to {labels_directory}")


    def _create_yaml_file(self):
        yaml_path = self.dataset_path / 'yolo_config.yaml'
        num_classes = len(self.condition_level_classes)
        names_list = list(self.condition_level_classes.keys())

        data = {
            'train': './train',
            'val': './val',
            'nc': num_classes,
            'names': names_list
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"[{self.condition_name}][fold {self.val_fold}]  Created YAML at {yaml_path}")


############################################
# Run for each condition, but only 2 folds (0 and 1):

# 1) Spinal Canal Stenosis
for fold in range(2):   # <— range(2) instead of range(5)
    spinal_classes = {
        'Spinal_Canal_Stenosis_L1_L2': 0,
        'Spinal_Canal_Stenosis_L2_L3': 1,
        'Spinal_Canal_Stenosis_L3_L4': 2,
        'Spinal_Canal_Stenosis_L4_L5': 3,
        'Spinal_Canal_Stenosis_L5_S1': 4,
    }

    DetectorDataPreparation(
        dataset_directory='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images',
        csv_path='/kaggle/working/Spinal_Canal_Stenosis_2folds.csv',  # new 2‐folds file
        condition_level_classes=spinal_classes,
        condition_name='Spinal_Canal_Stenosis',
        val_fold=fold,
        width_box=16,
    )
print("✅ Spinal Canal Stenosis data prep DONE for both folds\n")


# 2) Subarticular Stenosis
for fold in range(2):   # <— range(2)
    subart_classes = {
        'Left_Subarticular_Stenosis_L1_L2': 0,
        'Left_Subarticular_Stenosis_L2_L3': 1,
        'Left_Subarticular_Stenosis_L3_L4': 2,
        'Left_Subarticular_Stenosis_L4_L5': 3,
        'Left_Subarticular_Stenosis_L5_S1': 4,
        'Right_Subarticular_Stenosis_L1_L2': 5,
        'Right_Subarticular_Stenosis_L2_L3': 6,
        'Right_Subarticular_Stenosis_L3_L4': 7,
        'Right_Subarticular_Stenosis_L4_L5': 8,
        'Right_Subarticular_Stenosis_L5_S1': 9,
    }

    DetectorDataPreparation(
        dataset_directory='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images',
        csv_path='/kaggle/working/Subarticular_Stenosis_2folds.csv',
        condition_level_classes=subart_classes,
        condition_name='Subarticular_Stenosis',
        val_fold=fold,
        width_box=16,
    )
print("✅ Subarticular Stenosis data prep DONE for both folds\n")


# 3) Neural Foraminal Narrowing
for fold in range(2):   # <— range(2)
    neural_classes = {
        'Left_Neural_Foraminal_Narrowing_L1_L2': 0,
        'Left_Neural_Foraminal_Narrowing_L2_L3': 1,
        'Left_Neural_Foraminal_Narrowing_L3_L4': 2,
        'Left_Neural_Foraminal_Narrowing_L4_L5': 3,
        'Left_Neural_Foraminal_Narrowing_L5_S1': 4,
        'Right_Neural_Foraminal_Narrowing_L1_L2': 5,
        'Right_Neural_Foraminal_Narrowing_L2_L3': 6,
        'Right_Neural_Foraminal_Narrowing_L3_L4': 7,
        'Right_Neural_Foraminal_Narrowing_L4_L5': 8,
        'Right_Neural_Foraminal_Narrowing_L5_S1': 9,
    }

    DetectorDataPreparation(
        dataset_directory='/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images',
        csv_path='/kaggle/working/Neural_Foraminal_Narrowing_2folds.csv',
        condition_level_classes=neural_classes,
        condition_name='Neural_Foraminal_Narrowing',
        val_fold=fold,
        width_box=16,
    )
print("✅ Neural Foraminal Narrowing data prep DONE for both folds\n")


# ─────────────────────────────────────────────────────────────────────────────
# Cell: Zip & Clean Up Each Fold’s Detection Output
# ─────────────────────────────────────────────────────────────────────────────
import shutil
from pathlib import Path

BASE = Path('/kaggle/working')
CONDITIONS = ["Spinal_Canal_Stenosis", "Subarticular_Stenosis", "Neural_Foraminal_Narrowing"]
FOLDS      = [0, 1]

for cond in CONDITIONS:
    for f in FOLDS:
        src = BASE / cond / f'fold_{f}' / 'datasets'
        dst = BASE / f'{cond}_fold_{f}_datasets'
        if src.exists():
            # 1) create ZIP: /kaggle/working/<cond>_fold_<f>_datasets.zip
            shutil.make_archive(str(dst), 'zip', root_dir=src)
            print(f"✓ Zipped {src} → {dst}.zip")
            # 2) delete the uncompressed data
            shutil.rmtree(src)
            print(f"✓ Removed {src} to reclaim space\n")



import zipfile
from pathlib import Path

BASE       = Path('/kaggle/working')
CONDITIONS = ["Spinal_Canal_Stenosis","Subarticular_Stenosis","Neural_Foraminal_Narrowing"]
FOLDS      = [0,1]

for cond in CONDITIONS:
    for f in FOLDS:
        zip_fp = BASE/f'{cond}_fold_{f}_datasets.zip'
        out_dir = BASE/cond/f'fold_{f}'/'datasets'
        if zip_fp.exists() and not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_fp, 'r') as zp:
                zp.extractall(out_dir)
            print(f"✓ Unzipped {zip_fp} → {out_dir}")



# ─────────────────────────────────────────────────────────────────────────────
# Cell: 2-Fold YOLOv8 Training with WandB Disabled
# ─────────────────────────────────────────────────────────────────────────────

import os
# Disable Weights & Biases logging
os.environ['WANDB_MODE']    = 'disabled'
os.environ['WANDB_PROJECT'] = 'yolo_training'

from ultralytics import YOLO
from pathlib import Path

# CONFIGURATION
DATA_ROOT    = "/kaggle/working"
RESULTS_ROOT = "/kaggle/working/yolo_results"
CONDITIONS   = ["Spinal_Canal_Stenosis", "Subarticular_Stenosis", "Neural_Foraminal_Narrowing"]
FOLDS        = [0, 1]
EPOCHS       = 20
PATIENCE     = 5
BATCH_SIZE   = 8

# TRAINING LOOP
for condition in CONDITIONS:
    for fold in FOLDS:
        data_yaml  = f"{DATA_ROOT}/{condition}/fold_{fold}/datasets/yolo_config.yaml"
        project_dir = Path(RESULTS_ROOT) / condition / f"fold_{fold}"
        project_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ Training {condition}, fold {fold}")
        # Initialize from YAML (no .pt unpickle)
        model = YOLO("yolov8n.yaml")
        # Train with WandB disabled
        model.train(
            data     = data_yaml,
            project  = str(project_dir),
            name     = "exp",
            epochs   = EPOCHS,
            patience = PATIENCE,
            batch    = BATCH_SIZE
        )
        print(f"✅ Done {condition} fold {fold}")


import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

def splitted_data_2fold(condition):
    """
    Reads ./condition_csv/<condition>.csv, drops NaN scores,
    performs a 2-fold stratified split on 'score', and writes out:
      - ./<condition>/<condition>_2folds.csv
      - ./<condition>/fold_0/<condition>_train.csv, <condition>_val.csv
      - ./<condition>/fold_1/<condition>_train.csv, <condition>_val.csv
    """
    # 1) Load and clean
    df = pd.read_csv(f'./condition_csv/{condition}.csv')
    df = df.dropna(subset=['score']).reset_index(drop=True)

    # 2) Stratified 2-fold split
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df['score'])):
        df.loc[val_idx, 'fold'] = fold

    # 3) Save the combined 2-folds CSV
    condition_dir = Path(f'./{condition}')
    condition_dir.mkdir(exist_ok=True)
    df.to_csv(condition_dir / f'{condition}_2folds.csv', index=False)
    print(f"Saved 2-folds CSV to: {condition_dir / f'{condition}_2folds.csv'}")

    # 4) Write out per-fold train/val splits
    for fold in range(2):
        fold_dir = condition_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        train_df = df[df['fold'] != fold]
        val_df   = df[df['fold'] == fold]
        train_df.to_csv(fold_dir / f'{condition}_train.csv', index=False)
        val_df.to_csv(  fold_dir / f'{condition}_val.csv',   index=False)
        print(f"  Fold {fold}: {len(train_df)} train rows → {fold_dir / f'{condition}_train.csv'}")
        print(f"  Fold {fold}: {len(val_df)}   val rows → {fold_dir / f'{condition}_val.csv'}")

# Run for each condition
for cond in ['Neural_Foraminal_Narrowing', 'Spinal_Canal_Stenosis', 'Subarticular_Stenosis']:
    splitted_data_2fold(cond)



import os
import pandas as pd
import numpy as np
import random
import shutil

# Define augmentation methods
augmentations = ['rotate', 'horizontal_flip', 'vertical_flip', 'gaussian_noise']

# Function to balance minority classes
def augment_data(df: pd.DataFrame, augmentations: list) -> pd.DataFrame:
    """
    Oversample minority "score" classes to balance against the majority class.
    Adds an 'augmentation' column to each augmented sample.
    """
    counts = df['score'].value_counts()
    maj_cls = counts.idxmax()
    target1 = counts[maj_cls] // 3
    target2 = counts[maj_cls] // 2

    minors = [cls for cls, c in counts.items() if c < counts[maj_cls]]
    if len(minors) < 2:
        minors = minors * 2

    def sample_and_label(min_cls, target):
        existing = df[df['score'] == min_cls]
        needed = max(0, target - len(existing))
        if needed <= 0:
            return pd.DataFrame(columns=list(df.columns) + ['augmentation'])
        sampled = existing.sample(needed, replace=True).copy()
        sampled['augmentation'] = [random.choice(augmentations) for _ in range(needed)]
        return sampled

    aug1 = sample_and_label(minors[0], target1)
    aug2 = sample_and_label(minors[1], target2)

    df['augmentation'] = None
    return pd.concat([df, aug1, aug2], ignore_index=True)

# Two-fold augmentation for each condition and fold
conditions = ['Neural_Foraminal_Narrowing', 'Spinal_Canal_Stenosis', 'Subarticular_Stenosis']
folds = [0, 1]

for cond in conditions:
    for fold in folds:
        train_csv = f'./{cond}/fold_{fold}/{cond}_train.csv'
        val_csv   = f'./{cond}/fold_{fold}/{cond}_val.csv'

        df_train = pd.read_csv(train_csv)
        df_aug   = augment_data(df_train, augmentations)

        out_dir = os.path.join('augmented_output', cond, f'fold_{fold}')
        os.makedirs(out_dir, exist_ok=True)

        aug_train_path = os.path.join(out_dir, f'{cond}_augmented_train.csv')
        df_aug.to_csv(aug_train_path, index=False)
        print(f'Augmented train data saved to: {aug_train_path}')

        # Copy validation set unchanged
        aug_val_path = os.path.join(out_dir, f'{cond}_val.csv')
        shutil.copy(val_csv, aug_val_path)
        print(f'Validation data copied to: {aug_val_path}')



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Image Data Preparation for 2-Fold Splits
# ─────────────────────────────────────────────────────────────────────────────

import os
import pydicom
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

class DataPreparationImage:
    def __init__(
        self,
        dataset_directory: str,
        condition: str,
        csv_directory: str,
        num_folds: int = 2,
        augmentation_list=None,
    ):
        self.dataset_directory = dataset_directory
        self.condition = condition
        self.csv_directory = csv_directory
        self.num_folds = num_folds
        self.augmentation_list = augmentation_list or ['rotate','horizontal_flip','vertical_flip','gaussian_noise']

        print(f"Starting image data prep for {self.condition} ({self.num_folds} folds)")
        self._create_folders()
        self._process_all_folds()

    def _create_folders(self):
        base = Path(f"./{self.condition}")
        base.mkdir(exist_ok=True)
        for fold in range(self.num_folds):
            (base / f"fold_{fold}" / "train").mkdir(parents=True, exist_ok=True)
            (base / f"fold_{fold}" / "val").mkdir(parents=True, exist_ok=True)

    def _read_csv(self, fold: int, split: str) -> pd.DataFrame:
        filename = f"{self.condition}_{'augmented_train' if split=='train' else 'val'}.csv"
        path = Path(self.csv_directory) / self.condition / f"fold_{fold}" / filename
        return pd.read_csv(path)

    def _read_dicom(self, path: str) -> np.ndarray:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(float)
        img = (img - img.min())/(img.max()-img.min()+1e-6)*255.0
        return np.stack([img]*3,axis=-1).astype('uint8')

    def _crop(self, image: np.ndarray, x: float, y: float, box: int = 16) -> Image.Image:
        img = Image.fromarray(image)
        left, top = int(x-box), int(y-box)
        right, bottom = int(x+box), int(y+box)
        return img.crop((left, top, right, bottom))

    def _apply_augmentation(self, img: Image.Image, aug: str) -> Image.Image:
        if aug=='rotate':
            return img.rotate(np.random.uniform(-20,20), expand=True)
        if aug=='horizontal_flip':
            return ImageOps.mirror(img)
        if aug=='vertical_flip':
            return ImageOps.flip(img)
        if aug=='gaussian_noise':
            arr = np.array(img)
            noise = np.random.normal(0,25,arr.shape)
            return Image.fromarray(np.clip(arr+noise,0,255).astype('uint8'))
        return img

    def _process_all_folds(self):
        for fold in range(self.num_folds):
            # TRAIN SPLIT
            df_train = self._read_csv(fold, 'train')
            for _, row in df_train.iterrows():
                sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
                x, y = row['x'], row['y']
                aug = row.get('augmentation')
                dcm_path = os.path.join(self.dataset_directory, str(sid), str(seid), f"{inst}.dcm")
                img = self._read_dicom(dcm_path)
                cropped = self._crop(img, x, y)
                if aug in self.augmentation_list:
                    out_img = self._apply_augmentation(cropped, aug)
                    suffix = f"_{aug}"
                else:
                    out_img = cropped
                    suffix = ""
                fname = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}{suffix}.png"
                out_path = Path(self.condition) / f"fold_{fold}" / "train" / fname
                out_img.save(out_path)

            # VAL SPLIT (no augmentation)
            df_val = self._read_csv(fold, 'val')
            for _, row in df_val.iterrows():
                sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
                x, y = row['x'], row['y']
                dcm_path = os.path.join(self.dataset_directory, str(sid), str(seid), f"{inst}.dcm")
                img = self._read_dicom(dcm_path)
                cropped = self._crop(img, x, y)
                fname = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}.png"
                out_path = Path(self.condition) / f"fold_{fold}" / "val" / fname
                cropped.save(out_path)

# ─────────────────────────────────────────────────────────────────────────────
# Run for your three conditions:
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train"
CSV_DIR     = "/kaggle/working/augmented_output"

for cond in ["Subarticular_Stenosis", "Spinal_Canal_Stenosis", "Neural_Foraminal_Narrowing"]:
    DataPreparationImage(
        dataset_directory=DATASET_DIR,
        condition=cond,
        csv_directory=CSV_DIR,
        num_folds=2
    )



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Classification Label Preparation for 2-Fold Splits
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
from pathlib import Path

# 1) Parameters
CSV_DIR    = '../data_augmentation/augmented_output'   # where your augmented CSVs live
CONDITIONS = ['Spinal_Canal_Stenosis',
              'Neural_Foraminal_Narrowing',
              'Subarticular_Stenosis']
FOLDS      = [0, 1]  # only two folds
OUT_ROOT   = './'    # base for label output folders

# 2) Helper: build 'subject' and 'label' columns
def working_on_csv(csv_path: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # subject PNG filename
    def make_subject(r):
        name = f"{r.study_id}_{r.series_id}_{r.instance_number}_{int(r.x)}_{int(r.y)}.png"
        if split=='train' and pd.notna(r.augmentation) and r.augmentation:
            return name.replace('.png', '_augmented.png')
        return name
    # numeric label mapping
    def make_label(score):
        return {'Normal/Mild':1, 'Moderate':2, 'Severe':3}.get(score, None)

    df['subject'] = df.apply(make_subject, axis=1)
    df['label']   = df['score'].apply(make_label)
    return df[['subject','label']]

# 3) Create output folder structure
for cond in CONDITIONS:
    base = Path(OUT_ROOT) / f"{cond}_label"
    for fold in FOLDS:
        (base / f"fold_{fold}").mkdir(parents=True, exist_ok=True)

# 4) Process each condition & fold
for cond in CONDITIONS:
    for fold in FOLDS:
        # ←— **NOTE**: train file ends in `_augmented_train.csv`
        train_csv = f"{CSV_DIR}/{cond}/fold_{fold}/{cond}_augmented_train.csv"
        val_csv   = f"{CSV_DIR}/{cond}/fold_{fold}/{cond}_val.csv"

        # build label DataFrames
        train_labels = working_on_csv(train_csv, 'train')
        val_labels   = working_on_csv(val_csv,   'val')

        # write out
        out_base = Path(OUT_ROOT) / f"{cond}_label" / f"fold_{fold}"
        train_labels.to_csv(out_base / f"{cond}_augmented_labels.csv", index=False)
        val_labels.to_csv(  out_base / f"{cond}_val_labels.csv",       index=False)

        print(f"→ {cond} fold {fold}:")
        print(f"   • train labels → {out_base / f'{cond}_augmented_labels.csv'} ({len(train_labels)} rows)")
        print(f"   • val   labels → {out_base / f'{cond}_val_labels.csv'} ({len(val_labels)} rows)")



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Fast Severity Classification Training on 2 Folds
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

# CONFIGURATION
CONDITION     = 'Subarticular_Stenosis'    # or change to each condition
FOLDS         = [0, 1]                     # two folds
IMAGE_ROOT_TMPL = './{cond}/fold_{fold}'
LABEL_ROOT_TMPL = './{cond}_label/fold_{fold}'
EPOCHS        = 10                         # fewer epochs for quick results
BATCH_SIZE    = 16
PATIENCE      = 3
LEARNING_RATE = 1e-3

def load_data(df, image_root, split):
    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_root, split, row['subject'])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32,32))
        arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        X.append(arr)
        y.append(int(row['label']) - 1)
    return np.array(X), np.array(y)

for fold in FOLDS:
    print(f"\n▶ Training {CONDITION}, fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(cond=CONDITION, fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(cond=CONDITION, fold=fold)

    # Load labels
    train_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_augmented_labels.csv'))
    val_df   = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_val_labels.csv'))

    # Load image arrays and labels
    X_train, y_train = load_data(train_df, IMAGE_ROOT, 'train')
    X_val,   y_val   = load_data(val_df,   IMAGE_ROOT, 'val')

    # Build a small CNN
    model = Sequential([
        Conv2D(16, 3, activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping
    es = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    # Train
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=2
    )

    # Save model
    out_dir = Path(f'./results/{CONDITION}/fold_{fold}')
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / 'quick_model.h5')
    print(f"✅ Saved quick model for fold {fold} to: {out_dir/'quick_model.h5'}")



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Training + Evaluation with Confusion Matrix for 2 Folds
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from pathlib import Path

# CONFIGURATION
CONDITION       = 'Subarticular_Stenosis'
FOLDS           = [0, 1]
IMAGE_ROOT_TMPL = './{cond}/fold_{fold}'
LABEL_ROOT_TMPL = './{cond}_label/fold_{fold}'
EPOCHS          = 10
BATCH_SIZE      = 16
PATIENCE        = 3
LEARNING_RATE   = 1e-3
CLASS_NAMES     = ['Normal/Mild', 'Moderate', 'Severe']

def load_data(df, image_root, split):
    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_root, split, row['subject'])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32,32))
        arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        X.append(arr)
        y.append(int(row['label']) - 1)
    return np.array(X), np.array(y)

for fold in FOLDS:
    print(f"\n▶ Training {CONDITION}, fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(cond=CONDITION, fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(cond=CONDITION, fold=fold)

    # Load labels
    train_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_augmented_labels.csv'))
    val_df   = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_val_labels.csv'))

    # Load data arrays
    X_train, y_train = load_data(train_df, IMAGE_ROOT, 'train')
    X_val,   y_val   = load_data(val_df,   IMAGE_ROOT, 'val')

    # Build model
    model = Sequential([
        Conv2D(16, 3, activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)

    # Train
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=2)

    # Predict on validation set
    y_pred_probs = model.predict(X_val, batch_size=BATCH_SIZE)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    # Compute metrics
    acc   = accuracy_score(y_val, y_pred)
    prec  = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    rec   = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1    = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    cm    = confusion_matrix(y_val, y_pred)

    # Display results
    print(f"\nFold {fold} Validation Metrics:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 score : {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

    # Save model and metrics
    out_dir = Path(f'./results/{CONDITION}/fold_{fold}')
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / 'quick_model.h5')

    # Save metrics + confusion matrix to CSV
    metrics_df = pd.DataFrame({
        'metric':      ['accuracy','precision','recall','f1_score'],
        'value':       [acc,prec,rec,f1]
    })
    metrics_df.to_csv(out_dir / 'validation_metrics.csv', index=False)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(out_dir / 'confusion_matrix.csv')
    print(f"✅ Saved metrics and confusion matrix to {out_dir}\n")





