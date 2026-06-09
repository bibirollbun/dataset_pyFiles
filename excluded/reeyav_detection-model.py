# Cell 1: Install all dependencies once
# ⛔ Uninstall existing potentially problematic versions
#%pip uninstall -y torch torchvision torchaudio

# ✅ Install clean versions that avoid all UnpicklingErrors (torch < 2.6)
%pip install torch<2.6.0 torchvision<0.20.0 torchaudio --upgrade

# ✅ Install compatible ultralytics + common YOLO training libraries
%pip install ultralytics==8.0.111 pandas numpy pydicom albumentations scikit-learn tqdm



# Uninstall ray to avoid YOLOv8 ray tune callback error
!pip uninstall -y ray



import torch
import dill
import os
from pathlib import Path
import yaml
from ultralytics import YOLO

# Ultralytics components
from ultralytics.nn.modules.block import Bottleneck, C2f, DFL, SPPF
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.utils.tal import TaskAlignedAssigner
from ultralytics.yolo.utils import IterableSimpleNamespace
from ultralytics.yolo.utils.loss import BboxLoss, v8DetectionLoss

import numpy as np

# Torch components
from torch.nn import (
    Conv2d, BatchNorm2d, MaxPool2d, Sequential,
    ModuleList, SiLU, Upsample, BCEWithLogitsLoss
)

torch.serialization.add_safe_globals([
    # PyTorch
    Conv2d, BatchNorm2d, MaxPool2d, Sequential,
    ModuleList, SiLU, Upsample, BCEWithLogitsLoss, 
    BboxLoss,TaskAlignedAssigner,

    np.dtype, np.float64, np.dtypes.Float64DType, np.core.multiarray.scalar,


    # Ultralytics
    DetectionModel,
    Bottleneck, C2f, DFL, SPPF,
    Conv, Concat, Detect,
    IterableSimpleNamespace,
    v8DetectionLoss,

    # Dill (used for pickled config objects)
    dill._dill._load_type
])





# ✅ Disable WandB
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_PROJECT'] = 'yolo_training'

# ✅ Disable Ray Tune callback
from ultralytics.yolo.utils import callbacks
callbacks.default_callbacks.pop('on_fit_epoch_end', None)

# ✅ Custom global Ultralytics settings
ultralytics_settings_path = Path("/kaggle/working/ultralytics_custom_settings.yaml")
ultralytics_settings_path.parent.mkdir(parents=True, exist_ok=True)

if not ultralytics_settings_path.exists():
    default_settings = {
        "datasets_dir": "/kaggle/working/yolo_cache",  # Custom cache location
        "runs_dir": "/kaggle/working/yolo_results"     # Output folder
    }
    with open(ultralytics_settings_path, 'w') as f:
        yaml.dump(default_settings, f)

# ✅ Point Ultralytics to the settings file
os.environ["ULTRALYTICS_SETTINGS"] = str(ultralytics_settings_path)

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = "/kaggle/input"
RESULTS_ROOT = "/kaggle/working/yolo_results"
CONDITIONS = ["spinal-stenosis"]
FOLDS = [0, 1]
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 8

# -----------------------------
# TRAINING LOOP
# -----------------------------
for condition in CONDITIONS:
    for fold in FOLDS:
        data_yaml = f"{DATA_ROOT}/{condition}/fold_{fold}/datasets/yolo_config.yaml"
        save_dir = Path(RESULTS_ROOT) / condition / f"fold_{fold}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ Training {condition}, fold {fold}")
        
        # ✅ From config — no pretrained weights
        model = YOLO("yolov8n.yaml")
        
        # ✅ Training
        model.train(
            data=data_yaml,
            project="yolo_results",
            name=f"{condition.replace('-', '_')}_fold_{fold}",
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH_SIZE,
            amp=False,
            save=True,
            exist_ok=True  # avoids errors if dir exists
        )
        
        print(f"✅ Done {condition}, fold {fold}")



# ✅ Disable WandB
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_PROJECT'] = 'yolo_training'

# ✅ Disable Ray Tune callback
from ultralytics.yolo.utils import callbacks
callbacks.default_callbacks.pop('on_fit_epoch_end', None)

# ✅ Custom global Ultralytics settings
ultralytics_settings_path = Path("/kaggle/working/ultralytics_custom_settings.yaml")
ultralytics_settings_path.parent.mkdir(parents=True, exist_ok=True)

if not ultralytics_settings_path.exists():
    default_settings = {
        "datasets_dir": "/kaggle/working/yolo_cache",  # Custom cache location
        "runs_dir": "/kaggle/working/yolo_results"     # Output folder
    }
    with open(ultralytics_settings_path, 'w') as f:
        yaml.dump(default_settings, f)

# ✅ Point Ultralytics to the settings file
os.environ["ULTRALYTICS_SETTINGS"] = str(ultralytics_settings_path)

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = "/kaggle/input"
RESULTS_ROOT = "/kaggle/working/yolo_results"
CONDITIONS = ["subarticular-stenosis"]
FOLDS = [0, 1]
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 8

# -----------------------------
# TRAINING LOOP
# -----------------------------
for condition in CONDITIONS:
    for fold in FOLDS:
        data_yaml = f"{DATA_ROOT}/{condition}/fold_{fold}/datasets/yolo_config.yaml"
        save_dir = Path(RESULTS_ROOT) / condition / f"fold_{fold}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ Training {condition}, fold {fold}")
        
        # ✅ From config — no pretrained weights
        model = YOLO("yolov8n.yaml")
        
        # ✅ Training
        model.train(
            data=data_yaml,
            project="yolo_results",
            name=f"{condition.replace('-', '_')}_fold_{fold}",
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH_SIZE,
            amp=False,
            save=True,
            exist_ok=True  # avoids errors if dir exists
        )
        
        print(f"✅ Done {condition}, fold {fold}")



# ✅ Disable WandB
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_PROJECT'] = 'yolo_training'

# ✅ Disable Ray Tune callback
from ultralytics.yolo.utils import callbacks
callbacks.default_callbacks.pop('on_fit_epoch_end', None)

# ✅ Custom global Ultralytics settings
ultralytics_settings_path = Path("/kaggle/working/ultralytics_custom_settings.yaml")
ultralytics_settings_path.parent.mkdir(parents=True, exist_ok=True)

if not ultralytics_settings_path.exists():
    default_settings = {
        "datasets_dir": "/kaggle/working/yolo_cache",  # Custom cache location
        "runs_dir": "/kaggle/working/yolo_results"     # Output folder
    }
    with open(ultralytics_settings_path, 'w') as f:
        yaml.dump(default_settings, f)

# ✅ Point Ultralytics to the settings file
os.environ["ULTRALYTICS_SETTINGS"] = str(ultralytics_settings_path)

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = "/kaggle/input"
RESULTS_ROOT = "/kaggle/working/yolo_results"
CONDITIONS = ["neural-foraminal-narrowing"]
FOLDS = [0, 1]
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 8

# -----------------------------
# TRAINING LOOP
# -----------------------------
for condition in CONDITIONS:
    for fold in FOLDS:
        data_yaml = f"{DATA_ROOT}/{condition}/fold_{fold}/datasets/yolo_config.yaml"
        save_dir = Path(RESULTS_ROOT) / condition / f"fold_{fold}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n▶ Training {condition}, fold {fold}")
        
        # ✅ From config — no pretrained weights
        model = YOLO("yolov8n.yaml")
        
        # ✅ Training
        model.train(
            data=data_yaml,
            project="yolo_results",
            name=f"{condition.replace('-', '_')}_fold_{fold}",
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH_SIZE,
            amp=False,
            save=True,
            exist_ok=True  # avoids errors if dir exists
        )
        
        print(f"✅ Done {condition}, fold {fold}")


!zip -r yolo_results.zip /kaggle/working/yolo_results



rm -rf /kaggle/working/scs_yolo_fold0.zip


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
    df = pd.read_csv(f'/kaggle/input/csv-files/{condition}.csv')
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

DATASET_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
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
CSV_DIR    = '/kaggle/working/augmented_output'   # where your augmented CSVs live
CONDITIONS = ['Spinal_Canal_Stenosis',
              'Neural_Foraminal_Narrowing',
              'Subarticular_Stenosis']
FOLDS      = [0, 1]  # only two folds
OUT_ROOT   = '/kaggle/working/'    # base for label output folders

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
# Cell: Neural_Foraminal_Narrowing Fast Severity Classification Training on 2 Folds
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
CONDITION = 'Neural_Foraminal_Narrowing'
FOLDS = [0, 1]
IMAGE_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing_label/fold_{fold}'
BATCH_SIZE = 16
PATIENCE = 5
LEARNING_RATE = 1e-4
IMG_SIZE = (224, 224)

# Focal loss implementation
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

def load_data(df, image_root, split):
    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_root, split, row['subject'])
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            arr = img_to_array(img) / 255.0
            X.append(arr)
            y.append(int(row['label']) - 1)
        except:
            print(f"Failed to load {img_path}")
    return np.array(X), tf.keras.utils.to_categorical(y, num_classes=3)

for fold in FOLDS:
    print(f"\n▶ Training {CONDITION}, fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(cond=CONDITION, fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(cond=CONDITION, fold=fold)

    # Load labels
    train_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_augmented_labels.csv'))
    val_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_val_labels.csv'))

    X_train, y_train = load_data(train_df, IMAGE_ROOT, 'train')
    X_val, y_val = load_data(val_df, IMAGE_ROOT, 'val')

    # Load ResNet50 base
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=focal_loss(gamma=2.0, alpha=0.5),
                  metrics=['accuracy'])

    out_dir = Path(f'./results/{CONDITION}/fold_{fold}')
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=out_dir / 'best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=callbacks,
              verbose=2)

    # Evaluation
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    with open(out_dir / 'metrics.txt', 'w') as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"F1 Score: {f1}\n")

    print(f"✅ Saved model and metrics for fold {fold} to: {out_dir}")



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Spinal Canal Stenosis Fast Severity Classification Training on 2 Folds
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Cell: Fast Severity Classification Training on 2 Folds with ResNet & Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
CONDITION = 'Spinal_Canal_Stenosis'
FOLDS = [0, 1]
IMAGE_ROOT_TMPL = '/kaggle/working/Spinal_Canal_Stenosis/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Spinal_Canal_Stenosis_label/fold_{fold}'
BATCH_SIZE = 16
PATIENCE = 5
LEARNING_RATE = 1e-4
EPOCHS = 10
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal/Mild', 'Moderate', 'Severe']

# Focal loss implementation
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

def load_data(df, image_root, split):
    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_root, split, row['subject'])
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            arr = img_to_array(img) / 255.0
            X.append(arr)
            y.append(int(row['label']) - 1)
        except:
            print(f"Failed to load {img_path}")
    return np.array(X), tf.keras.utils.to_categorical(y, num_classes=3)

for fold in FOLDS:
    print(f"\n▶ Training {CONDITION}, fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(cond=CONDITION, fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(cond=CONDITION, fold=fold)

    train_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_augmented_labels.csv'))
    val_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_val_labels.csv'))

    X_train, y_train = load_data(train_df, IMAGE_ROOT, 'train')
    X_val, y_val = load_data(val_df, IMAGE_ROOT, 'val')

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=focal_loss(gamma=2.0, alpha=0.5),
                  metrics=['accuracy'])

    out_dir = Path(f'./results/{CONDITION}/fold_{fold}')
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=out_dir / 'best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=callbacks,
              verbose=2)

    # Evaluation
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    with open(out_dir / 'metrics.txt', 'w') as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"F1 Score: {f1}\n")

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / 'confusion_matrix.csv')
    print(f"✅ Saved model and metrics for fold {fold} to: {out_dir}")




# ─────────────────────────────────────────────────────────────────────────────
# Cell: Fast Severity Classification Training on 2 Folds with ResNet & Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
CONDITION = 'Subarticular_Stenosis'
FOLDS = [0, 1]
IMAGE_ROOT_TMPL = '/kaggle/working/Subarticular_Stenosis/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Subarticular_Stenosis_label/fold_{fold}'
BATCH_SIZE = 16
PATIENCE = 5
LEARNING_RATE = 1e-4
EPOCHS = 10
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal/Mild', 'Moderate', 'Severe']

# Focal loss implementation
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

def load_data(df, image_root, split):
    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_root, split, row['subject'])
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            arr = img_to_array(img) / 255.0
            X.append(arr)
            y.append(int(row['label']) - 1)
        except:
            print(f"Failed to load {img_path}")
    return np.array(X), tf.keras.utils.to_categorical(y, num_classes=3)

for fold in FOLDS:
    print(f"\n▶ Training {CONDITION}, fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(cond=CONDITION, fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(cond=CONDITION, fold=fold)

    train_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_augmented_labels.csv'))
    val_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_val_labels.csv'))

    X_train, y_train = load_data(train_df, IMAGE_ROOT, 'train')
    X_val, y_val = load_data(val_df, IMAGE_ROOT, 'val')

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=focal_loss(gamma=2.0, alpha=0.5),
                  metrics=['accuracy'])

    out_dir = Path(f'./results/{CONDITION}/fold_{fold}')
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=out_dir / 'best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=callbacks,
              verbose=2)

    # Evaluation
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    with open(out_dir / 'metrics.txt', 'w') as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"F1 Score: {f1}\n")

    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / 'confusion_matrix.csv')
    print(f"✅ Saved model and metrics for fold {fold} to: {out_dir}")






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





