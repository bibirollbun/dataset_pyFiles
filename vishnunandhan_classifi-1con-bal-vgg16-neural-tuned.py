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
#   - brightness & zoom augmentations
#   - safe crops (clamped), always end at 224Ã—224
#   - optional anatomy-safe aug toggles
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import os
import pydicom
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path
from typing import Tuple, Optional

# Optional reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

class DataPreparationImage:
    def __init__(
        self,
        dataset_directory: str,
        condition: str,
        csv_directory: str,
        num_folds: int = 5,
        augmentation_list=None,
        crop_size: int = 128,     # square crop size before resize
        out_size: Tuple[int,int] = (224,224),
        allow_vertical_flip: bool = False,  # anatomy-safe default
        add_contrast: bool = True           # light contrast jitter
    ):
        self.dataset_directory = Path(dataset_directory)
        self.condition       = condition
        self.csv_directory   = Path(csv_directory)
        self.num_folds       = num_folds
        self.crop_size       = crop_size
        self.out_size        = out_size
        self.allow_vertical_flip = allow_vertical_flip
        self.add_contrast    = add_contrast

        self.augmentation_list = augmentation_list or [
            'rotate','horizontal_flip',
            # 'vertical_flip',   # toggled via flag below
            'gaussian_noise','brightness','zoom'
        ]
        if self.allow_vertical_flip and 'vertical_flip' not in self.augmentation_list:
            self.augmentation_list.append('vertical_flip')

        print(f"Starting image data prep for {self.condition} ({self.num_folds} folds + test)")
        self._create_folders()
        self._process_all_folds()
        self._process_test()

    # â”€â”€ I/O helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        df = pd.read_csv(path)
        # ensure ints/floats
        for c in ('study_id','series_id','instance_number'):
            df[c] = df[c].astype(str)
        return df

    def _read_dicom(self, path: Path) -> np.ndarray:
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)
        # normalize to 0..255 robustly
        vmin, vmax = np.percentile(img, [1, 99])
        img = np.clip((img - vmin) / max(vmax - vmin, 1e-6), 0, 1) * 255.0
        img = img.astype('uint8')
        return np.stack([img]*3, axis=-1)

    # â”€â”€ geometry helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _safe_crop_box(self, w:int, h:int, cx:float, cy:float, box:int) -> Tuple[int,int,int,int]:
        half = box // 2
        left   = int(max(0, min(cx - half, w - box)))
        top    = int(max(0, min(cy - half, h - box)))
        right  = left + box
        bottom = top  + box
        return left, top, right, bottom

    def _crop(self, image: np.ndarray, x: float, y: float, box: Optional[int]=None) -> Image.Image:
        """Crop a boxÃ—box window centered at (x,y) (clamped to bounds)."""
        if box is None: box = self.crop_size
        img = Image.fromarray(image)
        w, h = img.size
        left, top, right, bottom = self._safe_crop_box(w, h, x, y, box)
        return img.crop((left, top, right, bottom))

    # â”€â”€ augmentations (guarantee final size) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _apply_augmentation(self, img: Image.Image, aug: str) -> Image.Image:
        # we operate at crop size, then we will finalize -> resize out_size
        if aug == 'rotate':
            angle = rng.uniform(-15, 15)  # milder than 20 to preserve structures
            # rotate around center, no expand; then center crop to keep box size
            rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)
            rw, rh = rotated.size
            cx, cy = rw // 2, rh // 2
            box = self.crop_size
            left = int(cx - box//2); top = int(cy - box//2)
            right = left + box;       bottom = top + box
            rotated = rotated.crop((max(0,left), max(0,top),
                                    min(rw,right), min(rh,bottom)))
            rotated = rotated.resize((self.crop_size, self.crop_size), Image.BICUBIC)
            img = rotated

        elif aug == 'horizontal_flip':
            img = ImageOps.mirror(img)

        elif aug == 'vertical_flip' and self.allow_vertical_flip:
            img = ImageOps.flip(img)

        elif aug == 'gaussian_noise':
            arr = np.array(img).astype(np.float32)
            sigma = rng.uniform(10, 25)  # adjustable
            noise = rng.normal(0, sigma, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype('uint8')
            img = Image.fromarray(arr)

        elif aug == 'brightness':
            enhancer = ImageEnhance.Brightness(img)
            factor = rng.uniform(0.85, 1.15)
            img = enhancer.enhance(factor)

        elif aug == 'zoom':
            # random slight in-zoom with keep-size crop (random resized crop flavor)
            w, h = img.size
            factor = rng.uniform(1.0, 1.2)
            new_w, new_h = int(w*factor), int(h*factor)
            zoomed = img.resize((new_w, new_h), Image.BICUBIC)
            # random crop window of original size
            max_x = new_w - w
            max_y = new_h - h
            left = rng.integers(0, max(1, max_x + 1))
            top  = rng.integers(0, max(1, max_y + 1))
            img = zoomed.crop((left, top, left + w, top + h))

        # tiny optional contrast jitter after any aug
        if self.add_contrast:
            c_enh = ImageEnhance.Contrast(img)
            img = c_enh.enhance(rng.uniform(0.95, 1.08))

        return img

    def _finalize(self, img: Image.Image) -> Image.Image:
        """Ensure output is exactly out_size and RGB uint8."""
        if img.size != self.out_size:
            img = img.resize(self.out_size, Image.BICUBIC)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    # â”€â”€ runners â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _process_all_folds(self):
        for fold in range(self.num_folds):
            print(f" Processing fold {fold}â€¦")
            # TRAIN (with augmentation as specified in CSV)
            df_train = self._read_csv('train', fold)
            for _, row in df_train.iterrows():
                sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
                x, y            = float(row['x']), float(row['y'])
                aug_op          = row.get('augmentation')
                dcm_path        = self.dataset_directory / sid / seid / f"{inst}.dcm"
                img_arr         = self._read_dicom(dcm_path)
                patch           = self._crop(img_arr, x, y, self.crop_size)

                out_img = patch
                suffix  = ""
                if aug_op in self.augmentation_list:
                    out_img = self._apply_augmentation(out_img, aug_op)
                    suffix  = f"_{aug_op}"

                out_img = self._finalize(out_img)
                fname = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}{suffix}.png"
                out_path = Path(self.condition) / f"fold_{fold}" / "train" / fname
                out_img.save(out_path, format='PNG', optimize=True)

            # VAL (no augmentation)
            df_val = self._read_csv('val', fold)
            for _, row in df_val.iterrows():
                sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
                x, y            = float(row['x']), float(row['y'])
                img_arr = self._read_dicom(self.dataset_directory / sid / seid / f"{inst}.dcm")
                patch   = self._crop(img_arr, x, y, self.crop_size)
                patch   = self._finalize(patch)
                fname   = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}.png"
                out_path = Path(self.condition) / f"fold_{fold}" / "val" / fname
                patch.save(out_path, format='PNG', optimize=True)

    def _process_test(self):
        print(" Processing held-out test setâ€¦")
        df_test = self._read_csv('test')
        for _, row in df_test.iterrows():
            sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
            x, y            = float(row['x']), float(row['y'])
            img_arr = self._read_dicom(self.dataset_directory / sid / seid / f"{inst}.dcm")
            patch   = self._crop(img_arr, x, y, self.crop_size)
            patch   = self._finalize(patch)
            fname   = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}.png"
            out_path = Path(self.condition) / "test" / fname
            patch.save(out_path, format='PNG', optimize=True)
        print(f" Saved {len(df_test)} test images to ./{self.condition}/test")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Run for your condition
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATASET_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
CSV_DIR     = "/kaggle/working/augmented_output"

DataPreparationImage(
    dataset_directory=DATASET_DIR,
    condition="Neural_Foraminal_Narrowing",
    csv_directory=CSV_DIR,
    num_folds=5,
    crop_size=128,              # 128Ã—128 crop around (x,y)
    out_size=(224,224),         # resize to model input
    allow_vertical_flip=False,  # anatomy-safe default
    add_contrast=True
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
        base = f"{r.study_id}_{r.series_id}_{r.instance_number}_{int(r.x)}_{int(r.y)}.png"
        # for TRAIN rows with an augmentation op, append that op name to match saved files
        if split == 'train' and pd.notna(r.augmentation) and str(r.augmentation).strip():
            op = str(r.augmentation).strip()
            return base.replace('.png', f'_{op}.png')
        return base

    def make_label(score):
        # numeric scores (1,2,3)
        try:
            return int(score)
        except Exception:
            pass
        # text -> numeric
        mapping = {
            'Normal/Mild': 1,
            'Moderate':    2,
            'Severe':      3
        }
        return mapping.get(str(score), None)

    df['subject'] = df.apply(make_subject, axis=1)
    df['label']   = df['score'].apply(make_label)

    # basic sanity checks
    if df['label'].isna().any():
        missing = df[df['label'].isna()][['subject','score']].head()
        print(f"[WARN] Some labels are NaN in {csv_path}. Example rows:\n{missing}")

    # keep only what training expects
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
# Cell: 5-Fold CV + Held-Out Test Evaluation (VGG16, CB-Focal + ROC-AUC monitor)
#   - GAP head
#   - Class-Balanced Focal loss (effective number of samples)
#   - Two-phase fine-tune: head warm-up â†’ unfreeze blocks (toggle below)
#   - Checkpoint/EarlyStopping/ReduceLROnPlateau on val_auc_roc
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import os
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CONDITION       = 'Neural_Foraminal_Narrowing'
FOLDS           = list(range(5))
IMAGE_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing_label/fold_{fold}'
TEST_IMAGE_DIR  = f'/kaggle/working/{CONDITION}/test'
TEST_LABEL_CSV  = f'/kaggle/working/{CONDITION}_label/test/{CONDITION}_test_labels.csv'
RESULTS_DIR     = Path(f'./results/{CONDITION}')

BATCH_SIZE      = 16
PATIENCE        = 5
PHASE_A_EPOCHS  = 8
PHASE_B_EPOCHS  = 12          # shorter; peaks early
LR_A            = 3e-4
LR_B            = 2e-5
IMG_SIZE        = (224, 224)
L2_WEIGHT       = 1e-4
DROP_RATE1      = 0.5
DROP_RATE2      = 0.4
NUM_CLASSES     = 3

# Toggle deeper unfreezing for Phase-B:
#   default: ("block5","block4")
#   deeper:  ("block5","block4","block3")
UNFREEZE_BLOCKS = ("block5", "block4")

# â”€â”€ Metrics (Keras) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def make_metrics():
    return [
        tf.keras.metrics.AUC(name="auc_roc", curve="ROC", multi_label=True, num_labels=NUM_CLASSES),
        tf.keras.metrics.AUC(name="auc_pr",  curve="PR",  multi_label=True, num_labels=NUM_CLASSES),
        "accuracy",
    ]

# â”€â”€ Class-Balanced Focal (Cui et al.) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_cb_focal(train_class_indices, num_classes=NUM_CLASSES, beta=0.9999, gamma=2.0):
    counts = np.bincount(train_class_indices, minlength=num_classes).astype(np.float32)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    weights = weights / np.mean(weights)  # normalize to mean=1
    class_w = tf.constant(weights, dtype=tf.float32)

    @tf.function
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal = tf.pow(1.0 - pt, gamma) * ce
        w = tf.reduce_sum(y_true * class_w, axis=-1)
        return w * focal
    return loss

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
    return Model(inputs=inp, outputs=out), base

def compile_with(model, lr, loss_fn):
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=L2_WEIGHT, clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn, metrics=make_metrics())

def make_callbacks(out_dir):
    ckpt = ModelCheckpoint(out_dir/'best_model.keras',
                           monitor='val_auc_roc', mode='max',
                           save_best_only=True, verbose=1)
    es  = EarlyStopping(monitor='val_auc_roc', mode='max',
                        patience=PATIENCE, restore_best_weights=True, verbose=1)
    rl  = ReduceLROnPlateau(monitor='val_auc_roc', mode='max',
                            factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    return [ckpt, rl, es]

fold_results = []
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for fold in FOLDS:
    print(f"\nâ–¶ Training fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(fold=fold)

    # â€” Load labels
    train_df = pd.read_csv(Path(LABEL_ROOT)/f'{CONDITION}_augmented_labels.csv')
    val_df   = pd.read_csv(Path(LABEL_ROOT)/f'{CONDITION}_val_labels.csv')
    for df in (train_df, val_df):
        # filenames produced by your prep cell already include aug op; no '_augmented' suffix
        df['label'] = (df['label'].astype(int) - 1).astype(str)

    # â€” Generators
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

    # â€” Build loss (per fold, from class counts)
    cb_focal_loss = build_cb_focal(train_gen.classes, NUM_CLASSES, beta=0.9999, gamma=2.0)

    # â€” Build & compile model
    model, base = build_vgg16_model()
    out_dir = RESULTS_DIR / f'fold_{fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    cbs = make_callbacks(out_dir)

    # â€” Phase A: warm-up head (base frozen)
    compile_with(model, LR_A, cb_focal_loss)
    history_a = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE_A_EPOCHS,
        callbacks=cbs,
        verbose=2
    )
    with open(out_dir/'history_phaseA.json','w') as f:
        json.dump(history_a.history, f)

    # â€” Phase B: unfreeze selected blocks
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers:
        if any(layer.name.startswith(b) for b in UNFREEZE_BLOCKS):
            layer.trainable = True

    compile_with(model, LR_B, cb_focal_loss)
    history_b = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE_B_EPOCHS,
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
# Load without recompiling (custom loss not needed for inference)
model = load_model(RESULTS_DIR/f'fold_{best_fold}'/'best_model.keras', compile=False)

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

print(f"âœ… Test evaluation saved to {test_out}")



# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Cell: Summary for CV & Held-Out Test â€” Loss/ROC Curves, Avg Metrics, Confusion & ROC/PR
#   (matches CB-Focal training + ROC-AUC monitoring)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import os, json, gc
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, average_precision_score,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ---------------- Config (match training cell) ----------------
CONDITION     = 'Neural_Foraminal_Narrowing'
RESULTS_DIR   = Path(f'./results/{CONDITION}')
LABEL_DIR     = Path(f'/kaggle/working/{CONDITION}_label')
IMG_DIR       = Path(f'/kaggle/working/{CONDITION}')
TEST_IMAGE_DIR= IMG_DIR / 'test'
TEST_LABEL_CSV= LABEL_DIR / 'test' / f'{CONDITION}_test_labels.csv'

BATCH_SIZE = 16
IMG_SIZE   = (224,224)
NUM_CLASSES = 3

# Keras generators (same normalization as training)
val_datagen  = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# ---------------- Gather histories & compute per-fold metrics ----------------
fold_dirs = sorted(RESULTS_DIR.glob('fold_*'), key=lambda p: int(p.name.split('_')[1]))

val_metrics = []
loss_curves, roc_curves = {}, {}

for fd in fold_dirs:
    fold = int(fd.name.split('_')[1])

    # --- Load history (phase A/B) and stitch
    hist_a = json.load(open(fd/'history_phaseA.json')) if (fd/'history_phaseA.json').exists() else {}
    hist_b = json.load(open(fd/'history_phaseB.json')) if (fd/'history_phaseB.json').exists() else {}

    loss = list(hist_a.get('loss', []))     + list(hist_b.get('loss', []))
    vloss= list(hist_a.get('val_loss', [])) + list(hist_b.get('val_loss', []))
    vroc = list(hist_a.get('val_auc_roc',[]))+list(hist_b.get('val_auc_roc',[]))
    loss_curves[fold] = (loss, vloss)
    roc_curves[fold]  = vroc

    # --- Build val generator for this fold
    val_csv = LABEL_DIR/f'fold_{fold}'/f'{CONDITION}_val_labels.csv'
    df_val  = pd.read_csv(val_csv)
    df_val['label'] = (df_val['label'].astype(int)-1).astype(str)
    val_gen = val_datagen.flow_from_dataframe(
        df_val,
        directory=IMG_DIR/f'fold_{fold}'/'val',
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    # --- Load best model for this fold (inference only)
    model = load_model(fd/'best_model.keras', compile=False)

    # --- Predict on validation and compute metrics
    probs_val = model.predict(val_gen, verbose=0)
    y_val     = val_gen.classes
    y_pred    = np.argmax(probs_val, axis=1)

    vm = {
        'fold': fold,
        'Accuracy_weighted':  accuracy_score(y_val, y_pred),
        'Precision_weighted': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'Recall_weighted':    recall_score(y_val, y_pred, average='weighted', zero_division=0),
        'F1_weighted':        f1_score(y_val, y_pred, average='weighted', zero_division=0),
        'Precision_macro':    precision_score(y_val, y_pred, average='macro', zero_division=0),
        'Recall_macro':       recall_score(y_val, y_pred, average='macro', zero_division=0),
        'F1_macro':           f1_score(y_val, y_pred, average='macro', zero_division=0),
    }
    y_val_bin = label_binarize(y_val, classes=list(range(NUM_CLASSES)))
    roc_aucs, pr_aucs = [], []
    for c in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_val_bin[:, c], probs_val[:, c])
        roc_aucs.append(auc(fpr, tpr))
        pr_aucs.append(average_precision_score(y_val_bin[:, c], probs_val[:, c]))
    vm['ROC_AUC_macro'] = float(np.mean(roc_aucs))
    vm['PR_AUC_macro']  = float(np.mean(pr_aucs))
    val_metrics.append(vm)

    gc.collect()

# ---------------- Plots: Loss & Val ROC-AUC (all folds) ----------------
plt.figure(figsize=(8,5))
for fold, (l, vl) in loss_curves.items():
    if len(l)==0: continue
    e = range(1, len(l)+1)
    plt.plot(e, l,  label=f'Fold {fold} Train', alpha=0.6)
    plt.plot(e, vl, '--', label=f'Fold {fold} Val', alpha=0.7)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train & Val Loss (All Folds)')
plt.legend(ncol=2, fontsize='small'); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
for fold, vroc in roc_curves.items():
    if len(vroc)==0: continue
    e = range(1, len(vroc)+1)
    plt.plot(e, vroc, label=f'Fold {fold}')
plt.xlabel('Epoch'); plt.ylabel('Val ROC-AUC'); plt.title('Val ROC-AUC (All Folds)')
plt.legend(ncol=3, fontsize='small'); plt.tight_layout(); plt.show()

# ---------------- Averages across folds ----------------
def avg(d, keys):
    return {k: float(np.mean([m[k] for m in d])) for k in keys}

keys_w = ['Accuracy_weighted','Precision_weighted','Recall_weighted','F1_weighted']
keys_m = ['Precision_macro','Recall_macro','F1_macro','ROC_AUC_macro','PR_AUC_macro']

avg_w = avg(val_metrics, keys_w)
avg_m = avg(val_metrics, keys_m)

print("â�¡ï¸�  Average CV VAL metrics:")
for k,v in avg_w.items(): print(f"   {k}: {v:.3f}")
for k,v in avg_m.items(): print(f"   {k}: {v:.3f}")

# ---------------- Choose best fold (by Val Macro-F1, consistent with training) ----------------
best_fold = int(max(val_metrics, key=lambda x: x['F1_macro'])['fold'])
print(f"\nâœ¨ Best fold = {best_fold} (Val Macro F1 = {max([m for m in val_metrics if m['fold']==best_fold], key=lambda x:x['F1_macro'])['F1_macro']:.4f})")

# ---------------- Held-out TEST evaluation on best fold ----------------
best_model = load_model(RESULTS_DIR/f'fold_{best_fold}'/'best_model.keras', compile=False)

test_df = pd.read_csv(TEST_LABEL_CSV)
test_df['label'] = (test_df['label'].astype(int)-1).astype(str)
test_gen = test_datagen.flow_from_dataframe(
    test_df,
    directory=TEST_IMAGE_DIR,
    x_col='subject', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

probs  = best_model.predict(test_gen, verbose=0)
y_test = test_gen.classes
y_pred = np.argmax(probs, axis=1)

metrics_test = {
    'Accuracy_weighted':  accuracy_score(y_test, y_pred),
    'Precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
    'Recall_weighted':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
    'F1_weighted':        f1_score(y_test, y_pred, average='weighted', zero_division=0),
    'Precision_macro':    precision_score(y_test, y_pred, average='macro', zero_division=0),
    'Recall_macro':       recall_score(y_test, y_pred, average='macro', zero_division=0),
    'F1_macro':           f1_score(y_test, y_pred, average='macro', zero_division=0),
}
print("\nâ�¡ï¸� Held-out TEST metrics:")
for k,v in metrics_test.items():
    print(f"   {k}: {v:.3f}")

# ---------------- Confusion matrix ----------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n", cm)

# ---------------- ROC & PR curves on TEST (per class) ----------------
y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

# ROC
plt.figure(figsize=(6.2,5))
for i,label in enumerate(test_gen.class_indices):
    fpr, tpr, _ = roc_curve(y_bin[:,i], probs[:,i])
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Held-out Test ROC Curves')
plt.legend(loc='lower right', fontsize='small'); plt.tight_layout(); plt.show()

# PR
plt.figure(figsize=(6.2,5))
for i,label in enumerate(test_gen.class_indices):
    prec, rec, _ = precision_recall_curve(y_bin[:,i], probs[:,i])
    ap = average_precision_score(y_bin[:,i], probs[:,i])
    plt.plot(rec, prec, label=f"{label} (AP={ap:.2f})")
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Held-out Test PR Curves')
plt.legend(loc='lower left', fontsize='small'); plt.tight_layout(); plt.show()

# ---------------- Save artifacts ----------------
test_out = RESULTS_DIR / 'test'
test_out.mkdir(exist_ok=True)

with open(test_out/'metrics_test.txt','w') as f:
    for k,v in metrics_test.items(): f.write(f"{k}: {v:.4f}\n")
np.savetxt(test_out/'confusion_matrix.csv', cm, delimiter=',', fmt='%d')

# Save per-class ROC-AUC & PR-AUC
y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
roc_lines, pr_lines = [], []
for i,label in enumerate(test_gen.class_indices):
    fpr, tpr, _ = roc_curve(y_bin[:,i], probs[:,i]); roc_lines.append(f"{label}: {auc(fpr,tpr):.4f}")
    ap = average_precision_score(y_bin[:,i], probs[:,i]); pr_lines.append(f"{label}: {ap:.4f}")
with open(test_out/'roc_auc.txt','w') as f: f.write("\n".join(roc_lines)+"\n")
with open(test_out/'pr_auc.txt','w') as f:  f.write("\n".join(pr_lines)+"\n")

print(f"\nâœ… Saved test artifacts to: {test_out}")



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
SEED0 = 123                   # base seed for TTA gens

# Rebuild the best model (if not already in memory)
best_model_path = RESULTS_DIR / f'fold_{best_fold}' / 'best_model.keras'
model = load_model(best_model_path, compile=False)   # <-- important for CB-Focal

# ---- Make best-fold VAL generator (needed to tune thresholds) ---------------
val_csv = LABEL_DIR / f'fold_{best_fold}' / f'{CONDITION}_val_labels.csv'
df_val  = pd.read_csv(val_csv)
# do NOT strip '_augmented' â€“ your saved filenames already match the CSV
df_val['label'] = (df_val['label'].astype(int) - 1).astype(str)

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
        test_df,                  # defined in previous summary cell
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
y_test = test_gen.classes  # from previous (non-aug) test_gen
y_test_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

bin_pred = (probs_tta >= best_thr).astype(int)   # broadcast per class
none_mask  = bin_pred.sum(axis=1) == 0
multi_mask = bin_pred.sum(axis=1) > 1
bin_pred[none_mask]  = 0
bin_pred[none_mask,  np.argmax(probs_tta[none_mask],  axis=1)] = 1
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

for c in range(NUM_CLASSES):
    ap = average_precision_score(y_test_bin[:, c], probs_tta[:, c])
    print(f"   Class {c} AP (PR-AUC): {ap:.3f}")

cm = confusion_matrix(y_test, y_pred_thresh)
print("\nConfusion matrix:\n", cm)

# PR curves with TTA probs
plt.figure(figsize=(6.2,5))
for c, label in enumerate(test_gen.class_indices):
    p, r, _ = precision_recall_curve(y_test_bin[:, c], probs_tta[:, c])
    ap = average_precision_score(y_test_bin[:, c], probs_tta[:, c])
    plt.plot(r, p, label=f"{label} (AP={ap:.2f})")
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Held-out Test PR Curves (TTA)')
plt.legend(loc='lower left', fontsize='small'); plt.tight_layout(); plt.show()

# (optional) save artifacts
out_dir = RESULTS_DIR / 'test'
out_dir.mkdir(exist_ok=True)
np.savetxt(out_dir/'tta_thresholds.csv', best_thr, fmt='%.3f', delimiter=',')


