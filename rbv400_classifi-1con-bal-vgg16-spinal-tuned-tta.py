import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from pathlib import Path

def split_train_val_test_5fold(
    condition: str,
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42
):
    """
    1) Reads /kaggle/input/csv-files/<condition>.csv, drops NaN scores
    2) Splits off a stratified test set (test_size fraction)
    3) Performs a stratified n_splits-fold CV on the remaining data
    4) Writes out:
       - ./<condition>/test/<condition>_test.csv
       - ./<condition>/<condition>_trainval_<n_splits>folds.csv
       - ./<condition>/fold_0/... fold_{n_splits-1}
    """
    # 1) Load and clean
    df = pd.read_csv(f'/kaggle/input/csv-files/{condition}.csv')
    df = df.dropna(subset=['score']).reset_index(drop=True)

    # 2) Stratified shuffle split â†’ train_val vs test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss.split(df, df['score']))
    df_train_val = df.loc[train_val_idx].reset_index(drop=True)
    df_test      = df.loc[test_idx].reset_index(drop=True)

    # Save test set
    condition_dir = Path(f'./{condition}')
    (condition_dir / 'test').mkdir(parents=True, exist_ok=True)
    df_test.to_csv(
        condition_dir / 'test' / f'{condition}_test.csv', 
        index=False
    )
    print(f"Saved test set ({len(df_test)} rows) to: {condition_dir/'test'/f'{condition}_test.csv'}")

    # 3) Stratified K-Fold on train_val
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    df_train_val['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df_train_val, df_train_val['score'])):
        df_train_val.loc[val_idx, 'fold'] = fold

    # Save combined train_val folds CSV
    df_train_val.to_csv(
        condition_dir / f'{condition}_trainval_{n_splits}folds.csv',
        index=False
    )
    print(f"Saved train/val folds CSV to: {condition_dir/f'{condition}_trainval_{n_splits}folds.csv'}")

    # 4) Write out per-fold train/val splits
    for fold in range(n_splits):
        fold_dir = condition_dir / f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = df_train_val[df_train_val['fold'] != fold].reset_index(drop=True)
        val_df   = df_train_val[df_train_val['fold'] == fold].reset_index(drop=True)

        train_df.to_csv(fold_dir / f'{condition}_train.csv', index=False)
        val_df.to_csv(  fold_dir / f'{condition}_val.csv',   index=False)
        print(f"  Fold {fold}: {len(train_df)} train rows, {len(val_df)} val rows")

# Run for each condition
for cond in ['Neural_Foraminal_Narrowing', 'Spinal_Canal_Stenosis', 'Subarticular_Stenosis']:
    split_train_val_test_5fold(cond)



import os
import random
import shutil

import numpy as np
import pandas as pd

# â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# â”€â”€ Augmentation operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
augmentation_ops = [
    'rotate',
    'horizontal_flip',
    'vertical_flip',
    'gaussian_noise',
    'brightness',
    'zoom'
]

def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oversample each class up to the majority count.
    Adds an 'augmentation' column indicating which op to apply.
    """
    counts = df['score'].value_counts()
    target = counts.max()
    
    # original rows, no augmentation
    base = df.copy()
    base['augmentation'] = None
    
    aug_list = [base]
    for cls, cnt in counts.items():
        if cnt < target:
            needed = int(target - cnt)
            sampled = (
                df[df.score == cls]
                .sample(needed, replace=True, random_state=SEED)
                .copy()
            )
            # assign one random augment op to each row
            sampled['augmentation'] = [
                random.choice(augmentation_ops) for _ in range(needed)
            ]
            aug_list.append(sampled)
    
    df_aug = pd.concat(aug_list, ignore_index=True)
    return df_aug

# --- Adjusted for train/val/test splits ---
conditions = [
    'Neural_Foraminal_Narrowing',
    'Spinal_Canal_Stenosis',
    'Subarticular_Stenosis'
]
folds = range(5)  # 5-fold CV

for cond in conditions:
    for fold in folds:
        in_dir  = f'./{cond}/fold_{fold}'
        out_dir = f'./augmented_output/{cond}/fold_{fold}'
        os.makedirs(out_dir, exist_ok=True)

        # 1) load train, augment, save
        train_csv = os.path.join(in_dir, f'{cond}_train.csv')
        df_train  = pd.read_csv(train_csv)
        if df_train.empty:
            raise RuntimeError(f"No training data for {cond} fold {fold}")

        df_aug = augment_data(df_train)
        df_aug.to_csv(
            os.path.join(out_dir, f'{cond}_augmented_train.csv'),
            index=False
        )
        print(f'â†’ Augmented train saved: {out_dir}/{cond}_augmented_train.csv')

        # 2) copy val 
        val_csv = os.path.join(in_dir, f'{cond}_val.csv')
        shutil.copy(val_csv, os.path.join(out_dir, f'{cond}_val.csv'))
        print(f'â†’ Val copied:           {out_dir}/{cond}_val.csv')

    # 3) copy test 
    test_in  = f'./{cond}/test/{cond}_test.csv'
    test_out = f'./augmented_output/{cond}/test'
    os.makedirs(test_out, exist_ok=True)
    shutil.copy(test_in, os.path.join(test_out, f'{cond}_test.csv'))
    print(f'â†’ Test copied:          {test_out}/{cond}_test.csv')



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cell: Image Data Preparation for 5-Fold CV + Held-Out Test
#  (with brightness & zoom augmentations added)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cell: Image Data Preparation for 5-Fold CV + Held-Out Test
#  (with brightness & zoom augmentations added, 128Ã—128 â†’ 224Ã—224 crops)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import os
import pydicom
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path

class DataPreparationImage:
    def __init__(
        self,
        dataset_directory: str,
        condition: str,
        csv_directory: str,
        num_folds: int = 5,
        augmentation_list=None,
    ):
        self.dataset_directory = Path(dataset_directory)
        self.condition       = condition
        self.csv_directory   = Path(csv_directory)
        self.num_folds       = num_folds
        self.augmentation_list = augmentation_list or [
            'rotate','horizontal_flip','vertical_flip',
            'gaussian_noise','brightness','zoom'
        ]

        print(f"Starting image data prep for {self.condition} ({self.num_folds} folds + test)")
        self._create_folders()
        self._process_all_folds()
        self._process_test()

    def _create_folders(self):
        base = Path(f"./{self.condition}")
        base.mkdir(exist_ok=True)
        for fold in range(self.num_folds):
            (base / f"fold_{fold}" / "train").mkdir(parents=True, exist_ok=True)
            (base / f"fold_{fold}" / "val").mkdir(parents=True, exist_ok=True)
        (base / "test").mkdir(parents=True, exist_ok=True)

    def _read_csv(self, split: str, fold: int = None) -> pd.DataFrame:
        if split in ('train','val'):
            if fold is None:
                raise ValueError("Must provide fold for train/val")
            suffix = 'augmented_train' if split=='train' else 'val'
            path = self.csv_directory / self.condition / f"fold_{fold}" / f"{self.condition}_{suffix}.csv"
        elif split == 'test':
            path = self.csv_directory / self.condition / "test" / f"{self.condition}_test.csv"
        else:
            raise ValueError(f"Unknown split: {split}")
        return pd.read_csv(path)

    def _read_dicom(self, path: Path) -> np.ndarray:
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255.0
        return np.stack([img]*3, axis=-1).astype('uint8')

    def _crop(self, image: np.ndarray, x: float, y: float, box: int = 64) -> Image.Image:
        """
        Crop a 128Ã—128 window centered at (x,y) then resize to 224Ã—224 (bicubic).
        """
        img = Image.fromarray(image)
        left, top   = int(x - box), int(y - box)
        right, bottom = int(x + box), int(y + box)
        patch = img.crop((left, top, right, bottom))
        # resize up to model input size
        return patch.resize((224, 224), resample=Image.BICUBIC)
    def _apply_augmentation(self, img: Image.Image, aug: str) -> Image.Image:
        # rotate/flips/noise unchanged
        if aug == 'rotate':
            return img.rotate(np.random.uniform(-20,20), expand=True)
        if aug == 'horizontal_flip':
            return ImageOps.mirror(img)
        if aug == 'vertical_flip':
            return ImageOps.flip(img)
        if aug == 'gaussian_noise':
            arr = np.array(img)
            noise = np.random.normal(0,25,arr.shape)
            return Image.fromarray(np.clip(arr+noise,0,255).astype('uint8'))
        # new brightness augmentation
        if aug == 'brightness':
            enhancer = ImageEnhance.Brightness(img)
            factor = np.random.uniform(0.7, 1.3)
            return enhancer.enhance(factor)
        # new zoom augmentation
        if aug == 'zoom':
            w, h = img.size
            factor = np.random.uniform(1.0, 1.2)
            new_w, new_h = int(w*factor), int(h*factor)
            zoomed = img.resize((new_w, new_h), Image.BILINEAR)
            left = (new_w - w)//2
            top  = (new_h - h)//2
            return zoomed.crop((left, top, left + w, top + h))
        # fallback: no op
        return img

    def _process_all_folds(self):
        for fold in range(self.num_folds):
            print(f" Processing fold {fold}â€¦")
            # TRAIN (with augmentation)
            df_train = self._read_csv('train', fold)
            for _, row in df_train.iterrows():
                sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
                x, y            = row['x'], row['y']
                aug_op          = row.get('augmentation')
                dcm_path        = self.dataset_directory / str(sid) / str(seid) / f"{inst}.dcm"
                img             = self._read_dicom(dcm_path)
                patch           = self._crop(img, x, y)
                if aug_op in self.augmentation_list:
                    out_img = self._apply_augmentation(patch, aug_op)
                    suffix  = f"_{aug_op}"
                else:
                    out_img = patch
                    suffix  = ""
                fname = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}{suffix}.png"
                out_path = Path(self.condition) / f"fold_{fold}" / "train" / fname
                out_img.save(out_path)

            # VAL (no augmentation)
            df_val = self._read_csv('val', fold)
            for _, row in df_val.iterrows():
                sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
                x, y            = row['x'], row['y']
                img = self._read_dicom(self.dataset_directory / str(sid) / str(seid) / f"{inst}.dcm")
                patch = self._crop(img, x, y)
                fname = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}.png"
                out_path = Path(self.condition) / f"fold_{fold}" / "val" / fname
                patch.save(out_path)

    def _process_test(self):
        print(" Processing held-out test setâ€¦")
        df_test = self._read_csv('test')
        for _, row in df_test.iterrows():
            sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
            x, y            = row['x'], row['y']
            img = self._read_dicom(self.dataset_directory / str(sid) / str(seid) / f"{inst}.dcm")
            patch = self._crop(img, x, y)
            fname = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}.png"
            out_path = Path(self.condition) / "test" / fname
            patch.save(out_path)
        print(f" Saved {len(df_test)} test images to ./{self.condition}/test")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Run for your condition
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATASET_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
CSV_DIR     = "/kaggle/working/augmented_output"

DataPreparationImage(
    dataset_directory=DATASET_DIR,
    condition="Spinal_Canal_Stenosis",
    csv_directory=CSV_DIR,
    num_folds=5
)



import os
import pandas as pd
from pathlib import Path

# 1) Parameters
CSV_DIR    = Path('/kaggle/working/augmented_output')   # where your augmented CSVs live
CONDITIONS = [
    'Spinal_Canal_Stenosis',
    'Neural_Foraminal_Narrowing',
    'Subarticular_Stenosis'
]
FOLDS      = list(range(5))  # five folds: 0â€“4
OUT_ROOT   = Path('/kaggle/working')    # base for label output folders

# 2) Helper: build 'subject' and 'label' columns
def working_on_csv(csv_path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # ensure we always have an 'augmentation' column
    if 'augmentation' not in df.columns:
        df['augmentation'] = None

    def make_subject(r):
        name = f"{r.study_id}_{r.series_id}_{r.instance_number}_{int(r.x)}_{int(r.y)}.png"
        # only append "_augmented" on training samples that were actually augmented
        if split == 'train' and pd.notna(r.augmentation) and r.augmentation:
            name = name.replace('.png', '_augmented.png')
        return name

    def make_label(score):
        # if your original CSV already used 1,2,3 as numeric scores, just cast
        if pd.api.types.is_numeric_dtype(type(score)):
            return int(score)
        # otherwise map from the text labels you used
        return {
            'Normal/Mild': 1,
            'Moderate':    2,
            'Severe':      3
        }.get(score, None)

    df['subject'] = df.apply(make_subject, axis=1)
    df['label']   = df['score'].apply(make_label)

    return df[['subject', 'label']]

# 3) Create output folder structure (including test)
for cond in CONDITIONS:
    base = OUT_ROOT / f"{cond}_label"
    for fold in FOLDS:
        (base / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    (base / "test").mkdir(parents=True, exist_ok=True)

# 4) Process each condition: train/val and then test
for cond in CONDITIONS:
    label_root = OUT_ROOT / f"{cond}_label"

    # train & val
    for fold in FOLDS:
        train_csv = CSV_DIR / cond / f"fold_{fold}" / f"{cond}_augmented_train.csv"
        val_csv   = CSV_DIR / cond / f"fold_{fold}" / f"{cond}_val.csv"

        train_labels = working_on_csv(train_csv, 'train')
        val_labels   = working_on_csv(val_csv,   'val')

        out_base = label_root / f"fold_{fold}"
        train_labels.to_csv(out_base / f"{cond}_augmented_labels.csv", index=False)
        val_labels.to_csv(  out_base / f"{cond}_val_labels.csv",       index=False)

        print(f"â†’ {cond} fold {fold}: train={len(train_labels)}, val={len(val_labels)}")

    # test
    test_csv    = CSV_DIR / cond / "test" / f"{cond}_test.csv"
    test_labels = working_on_csv(test_csv, 'test')
    out_test    = label_root / "test"
    test_labels.to_csv(out_test / f"{cond}_test_labels.csv", index=False)

    print(f"â†’ {cond} test: {len(test_labels)} rows â†’ {out_test}/{cond}_test_labels.csv")



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cell: 5-Fold CV + Held-Out Test Evaluation (VGG16 strong tune)
#         - GAP head, label smoothing, class weights
#         - Two-phase fine-tune: head warm-up â†’ unfreeze block5+block4
#         - Robust callbacks on PR-AUC, ReduceLROnPlateau, EarlyStopping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import os
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CONDITION       = 'Spinal_Canal_Stenosis'
FOLDS           = list(range(5))
IMAGE_ROOT_TMPL = '/kaggle/working/Spinal_Canal_Stenosis/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Spinal_Canal_Stenosis_label/fold_{fold}'
TEST_IMAGE_DIR  = f'/kaggle/working/{CONDITION}/test'
TEST_LABEL_CSV  = f'/kaggle/working/{CONDITION}_label/test/{CONDITION}_test_labels.csv'
RESULTS_DIR     = Path(f'./results/{CONDITION}')

BATCH_SIZE      = 16
PATIENCE        = 5
PHASE_A_EPOCHS  = 8        # warm-up head
PHASE_B_EPOCHS  = 20       # fine-tune unfreezed blocks
LR_A            = 3e-4
LR_B            = 3e-5
IMG_SIZE        = (224, 224)
L2_WEIGHT       = 1e-4
DROP_RATE1      = 0.5
DROP_RATE2      = 0.4
LABEL_SMOOTH    = 0.05
NUM_CLASSES     = 3
UNFREEZE_BLOCKS = ("block5", "block4")  # progressive unfreezing

# â”€â”€ Metrics (Keras) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_metrics():
    return [
        tf.keras.metrics.AUC(name="auc_roc", curve="ROC", multi_label=True, num_labels=NUM_CLASSES),
        tf.keras.metrics.AUC(name="auc_pr",  curve="PR",  multi_label=True, num_labels=NUM_CLASSES),
        "accuracy",
    ]

# â”€â”€ Loss â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
loss_ce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)

# â”€â”€ Augmentation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_class_weights(class_indices, num_classes=NUM_CLASSES):
    """Inverse-frequency weights normalized to mean=1."""
    counts = np.bincount(class_indices, minlength=num_classes).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv /= inv.mean()
    return {i: float(inv[i]) for i in range(num_classes)}

def build_vgg16_model():
    base = VGG16(include_top=False, weights='imagenet', input_shape=IMG_SIZE+(3,))
    base.trainable = False  # Phase A: freeze all

    inp = Input(shape=IMG_SIZE+(3,))
    x = base(inp, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(DROP_RATE1)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(L2_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROP_RATE2)(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model, base

def compile_with(model, lr):
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=L2_WEIGHT)
    model.compile(optimizer=opt, loss=loss_ce, metrics=make_metrics())

def make_callbacks(out_dir):
    ckpt = ModelCheckpoint(out_dir/'best_model.keras',
                           monitor='val_auc_pr', mode='max',
                           save_best_only=True, verbose=1)
    es  = EarlyStopping(monitor='val_auc_pr', mode='max',
                        patience=PATIENCE, restore_best_weights=True, verbose=1)
    rl  = ReduceLROnPlateau(monitor='val_auc_pr', mode='max',
                            factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    return [ckpt, rl, es]

fold_results = []

for fold in FOLDS:
    print(f"\nâ–¶ Training fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(fold=fold)

    # â€” Load labels
    train_df = pd.read_csv(Path(LABEL_ROOT)/f'{CONDITION}_augmented_labels.csv')
    val_df   = pd.read_csv(Path(LABEL_ROOT)/f'{CONDITION}_val_labels.csv')
    for df in (train_df, val_df):
        df['subject'] = df['subject'].str.replace('_augmented','',regex=False)
        df['label']   = (df['label'].astype(int) - 1).astype(str)

    # â€” Generators (train with aug, val only rescale)
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=os.path.join(IMAGE_ROOT,'train'),
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True, seed=42
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        directory=os.path.join(IMAGE_ROOT,'val'),
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    # â€” Class weights (from training set distribution)
    class_weights = compute_class_weights(train_gen.classes, NUM_CLASSES)
    print("Class weights:", class_weights)

    # â€” Build & compile model
    model, base = build_vgg16_model()
    out_dir = RESULTS_DIR / f'fold_{fold}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # â€” Phase A: warm-up head (base frozen)
    compile_with(model, LR_A)
    cbs = make_callbacks(out_dir)
    history_a = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE_A_EPOCHS,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=2
    )
    with open(out_dir/'history_phaseA.json','w') as f:
        json.dump(history_a.history, f)

    # â€” Phase B: progressive unfreezing (block5 + block4)
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers:
        if any(layer.name.startswith(b) for b in UNFREEZE_BLOCKS):
            layer.trainable = True

    compile_with(model, LR_B)
    history_b = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE_B_EPOCHS,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=2
    )
    with open(out_dir/'history_phaseB.json','w') as f:
        json.dump(history_b.history, f)

    # â€” Evaluate on validation
    val_gen.reset()
    probs_val = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(probs_val, axis=1)
    y_true = val_gen.classes

    # Weighted + Macro metrics
    fm = {
        'fold': fold,
        'Accuracy_weighted':  accuracy_score(y_true, y_pred),
        'Precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall_weighted':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1_weighted':        f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Precision_macro':    precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall_macro':       recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1_macro':           f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    # Macro ROC-AUC & PR-AUC on val
    y_true_bin = label_binarize(y_true, classes=[0,1,2])
    roc_aucs, pr_aucs = [], []
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs_val[:, i])
        roc_aucs.append(auc(fpr, tpr))
        pr_aucs.append(average_precision_score(y_true_bin[:, i], probs_val[:, i]))
    fm['ROC_AUC_macro'] = float(np.mean(roc_aucs))
    fm['PR_AUC_macro']  = float(np.mean(pr_aucs))

    fold_results.append(fm)
    with open(out_dir/'metrics_val.json','w') as f:
        json.dump(fm, f, indent=2)
    print(f"âœ… Fold {fold} metrics:", fm)

    tf.keras.backend.clear_session()
    gc.collect()

# â”€â”€ Pick best fold by Macro F1 & evaluate held-out test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
best = max(fold_results, key=lambda x: x['F1_macro'])
best_fold = best['fold']
print(f"\nâ–¶ Best fold selected: {best_fold} (Macro F1 = {best['F1_macro']:.4f})")

print("â–¶ Evaluating held-out test setâ€¦")
# No custom_objects needed (using standard CE loss)
model = load_model(RESULTS_DIR/f'fold_{best_fold}'/'best_model.keras')

test_df = pd.read_csv(TEST_LABEL_CSV)
test_df['label'] = (test_df['label'].astype(int)-1).astype(str)
test_gen = val_datagen.flow_from_dataframe(
    test_df,
    directory=TEST_IMAGE_DIR,
    x_col='subject', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

probs  = model.predict(test_gen, verbose=1)
y_pred = np.argmax(probs, axis=1)
y_true = test_gen.classes
y_true_bin = label_binarize(y_true, classes=[0,1,2])

# Weighted + Macro metrics on test
acc_test   = accuracy_score(y_true, y_pred)
prec_w     = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec_w      = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1_w       = f1_score(y_true, y_pred, average='weighted', zero_division=0)
prec_m     = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec_m      = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_m       = f1_score(y_true, y_pred, average='macro', zero_division=0)

# Per-class ROC-AUC & PR-AUC
roc_lines, pr_lines = [], []
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc_i = auc(fpr, tpr)
    ap_i      = average_precision_score(y_true_bin[:, i], probs[:, i])
    roc_lines.append(f"Class {i} ROC-AUC: {roc_auc_i:.4f}")
    pr_lines.append(f"Class {i} PR-AUC (AP): {ap_i:.4f}")

cm = confusion_matrix(y_true, y_pred)

test_out = RESULTS_DIR / 'test'
test_out.mkdir(exist_ok=True)
with open(test_out/'metrics_test.txt','w') as f:
    f.write(
        "Weighted:\n"
        f"  Accuracy:  {acc_test:.4f}\n"
        f"  Precision: {prec_w:.4f}\n"
        f"  Recall:    {rec_w:.4f}\n"
        f"  F1:        {f1_w:.4f}\n\n"
        "Macro:\n"
        f"  Precision: {prec_m:.4f}\n"
        f"  Recall:    {rec_m:.4f}\n"
        f"  F1:        {f1_m:.4f}\n"
    )
with open(test_out/'roc_auc.txt','w') as f:
    f.write("\n".join(roc_lines) + "\n")
with open(test_out/'pr_auc.txt','w') as f:
    f.write("\n".join(pr_lines) + "\n")

np.savetxt(test_out/'confusion_matrix.csv', cm, delimiter=',', fmt='%d')

# ROC Curves
plt.figure()
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:,i], probs[:,i])
    plt.plot(fpr, tpr, label=f'Class {i} (AUC={auc(fpr,tpr):.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curves on Held-Out Test Set')
plt.legend(loc='lower right')
plt.savefig(test_out/'roc_curves.png')
plt.close()

# PR Curves
plt.figure()
for i in range(NUM_CLASSES):
    # precision-recall curve via sklearn
    # (Use average_precision_score for area; plot points using sklearn if desired)
    # Simplified plotting using thresholds from roc_curve's probabilities is omitted for brevity.
    pass
plt.close()

print(f"âœ… Test evaluation saved to {test_out}")



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cell: Summary for CV & Held-Out Test â€“ Loss/PR Curves, Avg Metrics, Confusion & ROC/PR
#   (aligned with tuned VGG16 training cell: CE+label smoothing, PR-AUC monitor,
#    history_phaseA.json / history_phaseB.json, best fold by macro F1)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import json, os, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CONDITION      = 'Spinal_Canal_Stenosis'
RESULTS_DIR    = Path(f'./results/{CONDITION}')
LABEL_DIR      = Path(f'/kaggle/working/{CONDITION}_label')
IMG_DIR        = Path(f'/kaggle/working/{CONDITION}')
TEST_IMAGE_DIR = IMG_DIR / 'test'
TEST_LABEL_CSV = LABEL_DIR / 'test' / f'{CONDITION}_test_labels.csv'

BATCH_SIZE = 16
IMG_SIZE   = (224, 224)
NUM_CLASSES = 3

# â”€â”€ Find all fold directories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fold_dirs = sorted(RESULTS_DIR.glob('fold_*'),
                   key=lambda p: int(p.name.split('_')[1]))
assert len(fold_dirs) > 0, "No fold_* directories found under RESULTS_DIR."

# â”€â”€ Containers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train_metrics, val_metrics = [], []
loss_curves, vloss_curves = {}, {}
vpr_curves = {}   # per-epoch val PR-AUC if available
vroc_curves = {}  # per-epoch val ROC-AUC if available

val_macro_auc = []  # per-fold macro ROC-AUC (computed from predictions)
val_macro_ap  = []  # per-fold macro PR-AUC (computed from predictions)

# â”€â”€ Generators (rescale only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
val_datagen  = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# â”€â”€ Loop over folds to collect histories & metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for fd in fold_dirs:
    fold = int(fd.name.split('_')[1])

    # --- load histories (phaseA + phaseB or fallback to single history.json)
    hist_a_path = fd / 'history_phaseA.json'
    hist_b_path = fd / 'history_phaseB.json'
    if hist_a_path.exists() and hist_b_path.exists():
        hist_a = json.load(open(hist_a_path))
        hist_b = json.load(open(hist_b_path))
        loss   = list(hist_a.get('loss', []))    + list(hist_b.get('loss', []))
        vloss  = list(hist_a.get('val_loss', []))+ list(hist_b.get('val_loss', []))
        vpr    = list(hist_a.get('val_auc_pr', []))  + list(hist_b.get('val_auc_pr', []))
        vroc   = list(hist_a.get('val_auc_roc', [])) + list(hist_b.get('val_auc_roc', []))
    else:
        # backward-compatible fallback
        hist = json.load(open(fd/'history.json'))
        loss  = hist.get('loss', [])
        vloss = hist.get('val_loss', [])
        vpr   = hist.get('val_auc_pr', [])
        vroc  = hist.get('val_auc_roc', [])

    loss_curves[fold] = loss
    vloss_curves[fold] = vloss
    if vpr:  vpr_curves[fold]  = vpr
    if vroc: vroc_curves[fold] = vroc

    # --- load best model (no custom_objects needed)
    model = load_model(fd/'best_model.keras')

    # --- build train & val generators for metrics (no shuffling)
    train_csv = LABEL_DIR/f'fold_{fold}'/f'{CONDITION}_augmented_labels.csv'
    df_train  = pd.read_csv(train_csv)
    df_train['subject'] = df_train['subject'].str.replace('_augmented', '', regex=False)
    df_train['label']   = (df_train['label'].astype(int) - 1).astype(str)
    train_gen = val_datagen.flow_from_dataframe(
        df_train,
        directory=IMG_DIR/f'fold_{fold}'/'train',
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    val_csv = LABEL_DIR/f'fold_{fold}'/f'{CONDITION}_val_labels.csv'
    df_val  = pd.read_csv(val_csv)
    df_val['subject'] = df_val['subject'].str.replace('_augmented','', regex=False)
    df_val['label']   = (df_val['label'].astype(int) - 1).astype(str)
    val_gen = val_datagen.flow_from_dataframe(
        df_val,
        directory=IMG_DIR/f'fold_{fold}'/'val',
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    # --- predictions
    probs_train = model.predict(train_gen, verbose=0)
    y_train_pred = np.argmax(probs_train, axis=1)
    y_train_true = train_gen.classes

    probs_val = model.predict(val_gen, verbose=0)
    y_val_pred = np.argmax(probs_val, axis=1)
    y_val_true = val_gen.classes

    # --- metrics: weighted + macro
    train_metrics.append({
        'Accuracy':  accuracy_score(y_train_true, y_train_pred),
        'Precision': precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_train_true, y_train_pred, average='weighted', zero_division=0),
        'F1 Score':  f1_score(y_train_true, y_train_pred, average='weighted', zero_division=0),
        'F1 Macro':  f1_score(y_train_true, y_train_pred, average='macro', zero_division=0),
    })
    val_metrics.append({
        'Accuracy':  accuracy_score(y_val_true, y_val_pred),
        'Precision': precision_score(y_val_true, y_val_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_val_true, y_val_pred, average='weighted', zero_division=0),
        'F1 Score':  f1_score(y_val_true, y_val_pred, average='weighted', zero_division=0),
        'F1 Macro':  f1_score(y_val_true, y_val_pred, average='macro', zero_division=0),
    })

    # --- macro ROC-AUC & PR-AUC on validation (from probs)
    y_true_bin = label_binarize(y_val_true, classes=list(range(NUM_CLASSES)))
    fold_roc_aucs, fold_pr_aucs = [], []
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs_val[:, i])
        fold_roc_aucs.append(auc(fpr, tpr))
        fold_pr_aucs.append(average_precision_score(y_true_bin[:, i], probs_val[:, i]))
    val_macro_auc.append(np.mean(fold_roc_aucs))
    val_macro_ap.append(np.mean(fold_pr_aucs))

    tf.keras.backend.clear_session()
    gc.collect()

# â”€â”€ 1) Plot train/val loss curves per fold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.figure(figsize=(9,5))
for fold in sorted(loss_curves):
    epochs = range(1, len(loss_curves[fold]) + 1)
    plt.plot(epochs, loss_curves[fold],     label=f'Fold {fold} Train', alpha=0.7)
    plt.plot(epochs, vloss_curves[fold], '--', label=f'Fold {fold} Val',   alpha=0.7)
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Train & Val Loss Curves (All Folds)')
plt.legend(ncol=2, fontsize='small')
plt.tight_layout()
plt.show()

# (Optional) plot val PR-AUC curves if logged
if len(vpr_curves) > 0:
    plt.figure(figsize=(9,5))
    for fold in sorted(vpr_curves):
        plt.plot(range(1, len(vpr_curves[fold])+1), vpr_curves[fold], label=f'Fold {fold}')
    plt.xlabel('Epoch'); plt.ylabel('Val PR-AUC')
    plt.title('Validation PR-AUC across epochs')
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()
    plt.show()

# â”€â”€ 2) Average CV metrics (weighted + macro + AUCs) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def avg(dicts):
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

avg_train = avg(train_metrics)
avg_val   = avg(val_metrics)

print("â�¡ï¸�  Average CV TRAIN metrics:")
for k,v in avg_train.items():
    print(f"   {k}: {v:.3f}")
print("â�¡ï¸�  Average CV   VAL metrics:")
for k,v in avg_val.items():
    print(f"   {k}: {v:.3f}")
print(f"â�¡ï¸�  Average CV   VAL ROC-AUC (macro): {np.mean(val_macro_auc):.3f}")
print(f"â�¡ï¸�  Average CV   VAL PR-AUC  (macro): {np.mean(val_macro_ap):.3f}")

# â”€â”€ 3) Evaluate held-out test on best fold (by Val Macro F1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
best_idx  = int(np.argmax([m['F1 Macro'] for m in val_metrics]))
best_fold = int(fold_dirs[best_idx].name.split('_')[1])
print(f"\nâœ¨ Best fold = {best_fold} (Val Macro F1 = {val_metrics[best_idx]['F1 Macro']:.4f})")

model = load_model(RESULTS_DIR/f'fold_{best_fold}'/'best_model.keras')

test_df = pd.read_csv(TEST_LABEL_CSV)
test_df['label'] = (test_df['label'].astype(int)-1).astype(str)
test_gen = test_datagen.flow_from_dataframe(
    test_df,
    directory=TEST_IMAGE_DIR,
    x_col='subject', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

probs  = model.predict(test_gen, verbose=0)
y_test = test_gen.classes
y_pred = np.argmax(probs, axis=1)

# Weighted + Macro metrics on test
acc_w  = accuracy_score(y_test, y_pred)
prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec_w  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_w   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
prec_m = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec_m  = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_m   = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\nâ�¡ï¸� Held-out TEST metrics:")
print(f"   Weighted  - Acc: {acc_w:.3f}  Prec: {prec_w:.3f}  Rec: {rec_w:.3f}  F1: {f1_w:.3f}")
print(f"   Macro     - Prec: {prec_m:.3f} Rec: {rec_m:.3f}  F1: {f1_m:.3f}")

# â”€â”€ 4) Confusion Matrix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=list(test_gen.class_indices.keys()))
plt.figure(figsize=(4.2,4))
disp.plot(ax=plt.gca(), cmap='Blues', colorbar=False)
plt.title('Held-out Test Confusion')
plt.tight_layout()
plt.show()

# â”€â”€ 5) ROC & PR Curves with per-class AUC/AP â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

# ROC
plt.figure(figsize=(6.2,5))
for i, label in enumerate(test_gen.class_indices):
    fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--', linewidth=1)
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('Held-out Test ROC Curves')
plt.legend(loc='lower right', fontsize='small')
plt.tight_layout()
plt.show()

# PR
plt.figure(figsize=(6.2,5))
for i, label in enumerate(test_gen.class_indices):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
    ap = average_precision_score(y_bin[:, i], probs[:, i])
    plt.plot(recall, precision, label=f"{label} (AP={ap:.2f})")
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Held-out Test PR Curves')
plt.legend(loc='lower left', fontsize='small')
plt.tight_layout()
plt.show()



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cell: Per-Class Precision / Recall / F1 on Held-Out Test Set (fixed)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from pathlib import Path

# invert class_indices: idx -> name
inv_map = {v: k for k, v in test_gen.class_indices.items()}
labels = sorted(inv_map)                 # e.g., [0,1,2]
class_names = [inv_map[i] for i in labels]

# Use y_test from the previous cell (not y_true)
precisions, recalls, f1s, supports = precision_recall_fscore_support(
    y_test, y_pred, labels=labels, zero_division=0
)

df_metrics = pd.DataFrame({
    'Class':     class_names,
    'LabelIdx':  labels,
    'Support':   supports,
    'Precision': precisions,
    'Recall':    recalls,
    'F1 Score':  f1s
}).sort_values('LabelIdx')

print(df_metrics.to_string(index=False))

print("\nFull classification report:\n")
print(classification_report(
    y_test, y_pred, labels=labels, target_names=class_names, zero_division=0
))

# (optional) save alongside other test artifacts
out_dir = Path(f'./results/{CONDITION}/test')
out_dir.mkdir(parents=True, exist_ok=True)
df_metrics.to_csv(out_dir / 'per_class_metrics.csv', index=False)



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cell: Per-class Threshold Tuning (on Val of best fold) + Test-Time Aug (TTA)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ---- Config -----------------------------------------------------------------
N_TTA = 6                     # number of TTA passes
BETA_FOR_CLASS2 = 1.5         # use FÎ² for class 2 (Î²>1 favors recall); set 1.0 to disable
SEED0 = 123                    # base seed for TTA gens

# Rebuild the best model (if not already in memory)
best_model_path = RESULTS_DIR / f'fold_{best_fold}' / 'best_model.keras'
model = load_model(best_model_path)

# ---- Make best-fold VAL generator (needed to tune thresholds) ---------------
val_csv = LABEL_DIR / f'fold_{best_fold}' / f'{CONDITION}_val_labels.csv'
df_val  = pd.read_csv(val_csv)
df_val['subject'] = df_val['subject'].str.replace('_augmented','', regex=False)
df_val['label']   = (df_val['label'].astype(int) - 1).astype(str)

val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_dataframe(
    df_val,
    directory=IMG_DIR / f'fold_{best_fold}' / 'val',
    x_col='subject', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# ---- Predict on VAL and tune per-class thresholds ---------------------------
val_probs = model.predict(val_gen, verbose=0)
y_val_int = val_gen.classes
y_val_bin = label_binarize(y_val_int, classes=list(range(NUM_CLASSES)))

best_thr = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
for c in range(NUM_CLASSES):
    thrs = np.linspace(0.05, 0.95, 19)
    scores = []
    for t in thrs:
        y_hat = (val_probs[:, c] >= t).astype(int)
        if c == 2 and BETA_FOR_CLASS2 != 1.0:
            # FÎ² for class 2 (favor recall)
            p = precision_score(y_val_bin[:, c], y_hat, zero_division=0)
            r = recall_score(y_val_bin[:, c], y_hat, zero_division=0)
            beta2 = BETA_FOR_CLASS2 ** 2
            fbeta = (1+beta2) * p * r / (beta2 * p + r + 1e-8)
            scores.append(fbeta)
        else:
            scores.append(f1_score(y_val_bin[:, c], y_hat, zero_division=0))
    best_thr[c] = thrs[int(np.argmax(scores))]

print("ğŸ”§ Tuned per-class thresholds:", best_thr.tolist())

# ---- Build TTA generators for TEST and average predictions ------------------
def make_tta_gen(seed):
    return ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        fill_mode='nearest'
    ).flow_from_dataframe(
        test_df,  # from previous summary cell
        directory=TEST_IMAGE_DIR,
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False, seed=seed
    )

probs_tta = []
for i in range(N_TTA):
    gen_i = make_tta_gen(SEED0 + i)
    probs_i = model.predict(gen_i, verbose=0)
    probs_tta.append(probs_i)
probs_tta = np.mean(probs_tta, axis=0)

# ---- Apply thresholds -> binary preds; fall back to argmax if needed --------
y_test = test_gen.classes  # from previous cell (non-aug test_gen)
y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

bin_pred = (probs_tta >= best_thr).astype(int)

# ensure exactly one label per sample (single-label task)
none_mask = bin_pred.sum(axis=1) == 0
multi_mask = bin_pred.sum(axis=1) > 1
bin_pred[none_mask] = 0
bin_pred[none_mask, np.argmax(probs_tta[none_mask], axis=1)] = 1
bin_pred[multi_mask] = 0
bin_pred[multi_mask, np.argmax(probs_tta[multi_mask], axis=1)] = 1

y_pred_thresh = np.argmax(bin_pred, axis=1)

# ---- Recompute metrics with thresholds + TTA --------------------------------
acc_w  = accuracy_score(y_test, y_pred_thresh)
prec_w = precision_score(y_test, y_pred_thresh, average='weighted', zero_division=0)
rec_w  = recall_score(y_test, y_pred_thresh, average='weighted', zero_division=0)
f1_w   = f1_score(y_test, y_pred_thresh, average='weighted', zero_division=0)
prec_m = precision_score(y_test, y_pred_thresh, average='macro', zero_division=0)
rec_m  = recall_score(y_test, y_pred_thresh, average='macro', zero_division=0)
f1_m   = f1_score(y_test, y_pred_thresh, average='macro', zero_division=0)

print("\nğŸ“ˆ TEST (with thresholds + TTA)")
print(f"   Weighted  - Acc: {acc_w:.3f}  Prec: {prec_w:.3f}  Rec: {rec_w:.3f}  F1: {f1_w:.3f}")
print(f"   Macro     - Prec: {prec_m:.3f} Rec: {rec_m:.3f}  F1: {f1_m:.3f}")

# Per-class AP (PR-AUC) after TTA
for c in range(NUM_CLASSES):
    ap = average_precision_score(y_test_bin[:, c], probs_tta[:, c])
    print(f"   Class {c} AP (PR-AUC): {ap:.3f}")

# ---- Confusion matrix --------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_thresh)
print("\nConfusion matrix:\n", cm)

# Optional: quick PR curves with TTA probs
plt.figure(figsize=(6.2,5))
for c, label in enumerate(test_gen.class_indices):
    p, r, _ = precision_recall_curve(y_test_bin[:, c], probs_tta[:, c])
    ap = average_precision_score(y_test_bin[:, c], probs_tta[:, c])
    plt.plot(r, p, label=f"{label} (AP={ap:.2f})")
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Held-out Test PR Curves (TTA)')
plt.legend(loc='lower left', fontsize='small'); plt.tight_layout(); plt.show()


