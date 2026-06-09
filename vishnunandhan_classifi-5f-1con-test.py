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

    # 2) Stratified shuffle split → train_val vs test
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
import pandas as pd
import numpy as np
import random
import shutil

# Define augmentation methods
augmentations = ['rotate', 'horizontal_flip', 'vertical_flip', 'gaussian_noise']

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

    df = df.copy()
    df['augmentation'] = None
    return pd.concat([df, aug1, aug2], ignore_index=True)

# --- Adjusted for train/val/test splits ---
conditions = ['Neural_Foraminal_Narrowing',
              'Spinal_Canal_Stenosis',
              'Subarticular_Stenosis']
folds = list(range(5))  # 5-fold CV

for cond in conditions:
    # Augment each train fold and copy its val split
    for fold in folds:
        base_in  = f'./{cond}/fold_{fold}'
        base_out = f'./augmented_output/{cond}/fold_{fold}'
        os.makedirs(base_out, exist_ok=True)

        # load train, augment, save
        train_csv = os.path.join(base_in, f'{cond}_train.csv')
        df_train  = pd.read_csv(train_csv)
        df_aug    = augment_data(df_train, augmentations)
        df_aug.to_csv(os.path.join(base_out, f'{cond}_augmented_train.csv'), index=False)
        print(f'→ Augmented train saved: {base_out}/{cond}_augmented_train.csv')

        # copy val unchanged
        val_csv = os.path.join(base_in, f'{cond}_val.csv')
        shutil.copy(val_csv, os.path.join(base_out, f'{cond}_val.csv'))
        print(f'→ Val copied:           {base_out}/{cond}_val.csv')

    # Copy test set unchanged
    test_in  = f'./{cond}/test/{cond}_test.csv'
    test_out = f'./augmented_output/{cond}/test'
    os.makedirs(test_out, exist_ok=True)
    shutil.copy(test_in, os.path.join(test_out, f'{cond}_test.csv'))
    print(f'→ Test copied:          {test_out}/{cond}_test.csv')



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Image Data Preparation for 5-Fold CV + Held-Out Test
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
        num_folds: int = 5,
        augmentation_list=None,
    ):
        self.dataset_directory = Path(dataset_directory)
        self.condition       = condition
        self.csv_directory   = Path(csv_directory)
        self.num_folds       = num_folds
        self.augmentation_list = augmentation_list or [
            'rotate', 'horizontal_flip', 'vertical_flip', 'gaussian_noise'
        ]

        print(f"Starting image data prep for {self.condition} ({self.num_folds} folds + test)")
        self._create_folders()
        self._process_all_folds()
        self._process_test()

    def _create_folders(self):
        base = Path(f"./{self.condition}")
        base.mkdir(exist_ok=True)
        # train / val
        for fold in range(self.num_folds):
            (base / f"fold_{fold}" / "train").mkdir(parents=True, exist_ok=True)
            (base / f"fold_{fold}" / "val").mkdir(parents=True, exist_ok=True)
        # held-out test
        (base / "test").mkdir(parents=True, exist_ok=True)

    def _read_csv(self, split: str, fold: int = None) -> pd.DataFrame:
        """
        split: 'train', 'val', or 'test'
        fold: required for train/val, ignored for test
        """
        if split in ('train', 'val'):
            if fold is None:
                raise ValueError("Must provide fold for train/val")
            suffix = 'augmented_train' if split == 'train' else 'val'
            filename = f"{self.condition}_{suffix}.csv"
            path = self.csv_directory / self.condition / f"fold_{fold}" / filename
        elif split == 'test':
            filename = f"{self.condition}_test.csv"
            path = self.csv_directory / self.condition / "test" / filename
        else:
            raise ValueError(f"Unknown split: {split}")
        return pd.read_csv(path)

    def _read_dicom(self, path: Path) -> np.ndarray:
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255.0
        rgb = np.stack([img] * 3, axis=-1)
        return rgb.astype('uint8')

    def _crop(self, image: np.ndarray, x: float, y: float, box: int = 16) -> Image.Image:
        img = Image.fromarray(image)
        left, top   = int(x - box), int(y - box)
        right, bottom = int(x + box), int(y + box)
        return img.crop((left, top, right, bottom))

    def _apply_augmentation(self, img: Image.Image, aug: str) -> Image.Image:
        if aug == 'rotate':
            return img.rotate(np.random.uniform(-20,20), expand=True)
        if aug == 'horizontal_flip':
            return ImageOps.mirror(img)
        if aug == 'vertical_flip':
            return ImageOps.flip(img)
        if aug == 'gaussian_noise':
            arr   = np.array(img)
            noise = np.random.normal(0,25,arr.shape)
            return Image.fromarray(np.clip(arr+noise,0,255).astype('uint8'))
        return img

    def _process_all_folds(self):
        for fold in range(self.num_folds):
            print(f" Processing fold {fold}…")
            # TRAIN (with augmentation)
            df_train = self._read_csv('train', fold)
            for _, row in df_train.iterrows():
                sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
                x, y    = row['x'], row['y']
                aug_op  = row.get('augmentation')
                dcm     = self.dataset_directory / str(sid) / str(seid) / f"{inst}.dcm"
                img     = self._read_dicom(dcm)
                patch   = self._crop(img, x, y)
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
                x, y    = row['x'], row['y']
                dcm     = self.dataset_directory / str(sid) / str(seid) / f"{inst}.dcm"
                img     = self._read_dicom(dcm)
                patch   = self._crop(img, x, y)
                fname   = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}.png"
                out_path = Path(self.condition) / f"fold_{fold}" / "val" / fname
                patch.save(out_path)

    def _process_test(self):
        print(" Processing held-out test set…")
        df_test = self._read_csv('test')
        for _, row in df_test.iterrows():
            sid, seid, inst = row['study_id'], row['series_id'], row['instance_number']
            x, y    = row['x'], row['y']
            dcm     = self.dataset_directory / str(sid) / str(seid) / f"{inst}.dcm"
            img     = self._read_dicom(dcm)
            patch   = self._crop(img, x, y)
            fname   = f"{sid}_{seid}_{inst}_{int(x)}_{int(y)}.png"
            out_path = Path(self.condition) / "test" / fname
            patch.save(out_path)
        print(f" Saved {len(df_test)} test images to ./{self.condition}/test")


# ─────────────────────────────────────────────────────────────────────────────
# Run for your condition (example: Neural Foraminal Narrowing)
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images"
CSV_DIR     = "/kaggle/working/augmented_output"

DataPreparationImage(
    dataset_directory=DATASET_DIR,
    condition="Neural_Foraminal_Narrowing",
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
FOLDS      = list(range(5))  # five folds: 0–4
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

        print(f"→ {cond} fold {fold}: train={len(train_labels)}, val={len(val_labels)}")

    # test
    test_csv    = CSV_DIR / cond / "test" / f"{cond}_test.csv"
    test_labels = working_on_csv(test_csv, 'test')
    out_test    = label_root / "test"
    test_labels.to_csv(out_test / f"{cond}_test_labels.csv", index=False)

    print(f"→ {cond} test: {len(test_labels)} rows → {out_test}/{cond}_test_labels.csv")



# ─────────────────────────────────────────────────────────────────────────────
# Cell: 5-Fold CV + Held-Out Test Evaluation for Neural Foraminal Narrowing
# ─────────────────────────────────────────────────────────────────────────────

import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import json

# ── Configuration ────────────────────────────────────────────────────────────
CONDITION     = 'Neural_Foraminal_Narrowing'
FOLDS         = list(range(5))
IMAGE_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing_label/fold_{fold}'
TEST_IMAGE_DIR  = f'/kaggle/working/{CONDITION}/test'
TEST_LABEL_CSV  = f'/kaggle/working/{CONDITION}_label/test/{CONDITION}_test_labels.csv'
RESULTS_DIR     = Path(f'./results/{CONDITION}')
BATCH_SIZE   = 16
PATIENCE     = 5
EPOCHS       = 10
LEARNING_RATE= 1e-4
IMG_SIZE     = (224, 224)

# ── Focal Loss ────────────────────────────────────────────────────────────────
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce     = -y_true * tf.math.log(y_pred)
        loss   = alpha * tf.math.pow(1 - y_pred, gamma) * ce
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# ── Keep track of each fold’s validation F1 ─────────────────────────────────
fold_results = []

for fold in FOLDS:
    print(f"\n▶ Training fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(fold=fold)

    # — Load labels
    train_df = pd.read_csv(Path(LABEL_ROOT) / f'{CONDITION}_augmented_labels.csv')
    val_df   = pd.read_csv(Path(LABEL_ROOT) / f'{CONDITION}_val_labels.csv')

    # — Strip “_augmented” and re-index to 0/1/2
    for df in (train_df, val_df):
        df['subject'] = df['subject'].str.replace('_augmented', '', regex=False)
        df['label']   = (df['label'].astype(int) - 1).astype(str)

    # — Generators (only rescale)
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_dataframe(
        train_df, directory=os.path.join(IMAGE_ROOT,'train'),
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True
    )
    val_gen = datagen.flow_from_dataframe(
        val_df, directory=os.path.join(IMAGE_ROOT,'val'),
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    # — Build & compile model
    base = ResNet50(include_top=False, weights='imagenet', input_shape=IMG_SIZE+(3,))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(3, activation='softmax')(x)
    model = Model(base.input, preds)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=focal_loss(gamma=2.0, alpha=0.5),
        metrics=['accuracy']
    )

    # — Callbacks & checkpoints
    out_dir = RESULTS_DIR / f'fold_{fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(out_dir/'best_model.keras', monitor='val_loss',
                           save_best_only=True, verbose=1)
    es   = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                         restore_best_weights=True, verbose=1)

    # — Train
    history = model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS, callbacks=[es,ckpt], verbose=2
    )
    
    hist_path = out_dir / 'history.json'
    with open(hist_path, 'w') as f:
        json.dump(history.history, f)

    # — Evaluate on validation
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    y_true = val_gen.classes
    fold_metrics = {
        'fold': fold,
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall':    recall_score(y_true, y_pred, average='weighted'),
        'F1 Score':  f1_score(y_true, y_pred, average='weighted')
    }
    fold_results.append(fold_metrics)

    # — Save fold metrics
    with open(out_dir/'metrics.txt','w') as f:
        for k,v in fold_metrics.items():
            if k!='fold': f.write(f"{k}: {v:.4f}\n")
    print(f"✅ Fold {fold} metrics:", fold_metrics)

    # — Cleanup
    tf.keras.backend.clear_session()
    gc.collect()

# ── Pick best fold by validation F1 ─────────────────────────────────────────
best = max(fold_results, key=lambda x: x['F1 Score'])
best_fold = best['fold']
print(f"\n▶ Best fold selected: {best_fold} (F1 = {best['F1 Score']:.4f})")

# ── Held-Out Test Evaluation ────────────────────────────────────────────────
print("▶ Evaluating on held-out test set…")

# — Load best model
best_model_path = RESULTS_DIR / f'fold_{best_fold}' / 'best_model.keras'
model = load_model(
    best_model_path,
    custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.5)}
)

# — Prepare test DataFrame
test_df = pd.read_csv(TEST_LABEL_CSV)
test_df['label'] = (test_df['label'].astype(int) - 1).astype(str)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_dataframe(
    test_df, directory=TEST_IMAGE_DIR,
    x_col='subject', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# — Predict
probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(probs, axis=1)
y_true = test_gen.classes

# — Compute metrics
acc_test      = accuracy_score(y_true, y_pred)
err_test      = 1 - acc_test
prec_test     = precision_score(y_true, y_pred, average='weighted')
rec_test      = recall_score(y_true, y_pred, average='weighted')
f1_test       = f1_score(y_true, y_pred, average='weighted')
cm            = confusion_matrix(y_true, y_pred)
y_true_bin    = label_binarize(y_true, classes=[0,1,2])

# — ROC / AUC per class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], probs[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# — Save test results
test_out = RESULTS_DIR / 'test'
test_out.mkdir(parents=True, exist_ok=True)

with open(test_out/'metrics_test.txt','w') as f:
    f.write(f"Accuracy:  {acc_test:.4f}\n")
    f.write(f"Error:     {err_test:.4f}\n")
    f.write(f"Precision: {prec_test:.4f}\n")
    f.write(f"Recall:    {rec_test:.4f}\n")
    f.write(f"F1 Score:  {f1_test:.4f}\n")

np.savetxt(test_out/'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
with open(test_out/'roc_auc.txt','w') as f:
    for i,aucv in roc_auc.items():
        f.write(f"Class {i} AUC: {aucv:.4f}\n")

# — Plot & save ROC curves
plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={roc_auc[i]:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves on Held-Out Test Set')
plt.legend(loc='lower right')
plt.savefig(test_out/'roc_curves.png')
plt.close()

print(f"✅ Saved test evaluation to: {test_out}")



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Summary for Best Fold – Loss Curve, Confusion Matrix & ROC
# ─────────────────────────────────────────────────────────────────────────────

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ── Config ────────────────────────────────────────────────────────────────────
CONDITION   = 'Neural_Foraminal_Narrowing'
RESULTS_DIR = Path(f'./results/{CONDITION}')
LABEL_DIR   = Path(f'/kaggle/working/{CONDITION}_label')
IMG_DIR     = Path(f'/kaggle/working/{CONDITION}')
BATCH_SIZE  = 16
IMG_SIZE    = (224, 224)

# ── 1) Pick best fold by saved metrics ────────────────────────────────────────
fold_dirs = sorted(RESULTS_DIR.glob('fold_*'), key=lambda p: int(p.name.split('_')[1]))
best_fold = None
best_f1   = -1.0

for fd in fold_dirs:
    # read the F1 from metrics.txt
    txt = (fd / 'metrics.txt').read_text().splitlines()
    f1_line = [l for l in txt if l.startswith('F1 Score')][0]
    f1_val  = float(f1_line.split(':')[-1])
    fold_idx = int(fd.name.split('_')[1])
    if f1_val > best_f1:
        best_f1   = f1_val
        best_fold = fold_idx

print(f"✨ Best fold is {best_fold} with val F1 = {best_f1:.4f}")

best_dir = RESULTS_DIR / f'fold_{best_fold}'

# ── 2) Plot Train vs Val Loss ────────────────────────────────────────────────
hist = json.load(open(best_dir / 'history.json'))
epochs = range(1, len(hist['loss'])+1)

plt.figure()
plt.plot(epochs, hist['loss'],    label='Train Loss')
plt.plot(epochs, hist['val_loss'],label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title(f'Fold {best_fold} Loss Curve')
plt.legend()
plt.show()

# ── 3) Build validation generator for best fold ─────────────────────────────
# load labels
val_csv = LABEL_DIR / 'fold_{}'.format(best_fold) / f'{CONDITION}_val_labels.csv'
val_df  = pd.read_csv(val_csv)
val_df['subject'] = val_df['subject'].str.replace('_augmented', '', regex=False)
val_df['label']   = (val_df['label'].astype(int) - 1).astype(str)

test_datagen = ImageDataGenerator(rescale=1./255)
val_gen = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=IMG_DIR / f'fold_{best_fold}' / 'val',
    x_col='subject', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# class names
inv_map = {v:k for k,v in val_gen.class_indices.items()}
classes = [inv_map[i] for i in range(len(inv_map))]

# ── 4) Load best model & predict ────────────────────────────────────────────
model = load_model(
    best_dir / 'best_model.keras',
    custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.5)}
)

print("Val folder listing:", sorted((IMG_DIR/f'fold_{best_fold}'/'val').iterdir())[:5])
print("Val subjects:", val_df['subject'].unique()[:5])

val_gen.reset()
probs   = model.predict(val_gen, verbose=0)
y_true  = val_gen.classes
y_pred  = np.argmax(probs, axis=1)

# ── 5) Plot Confusion Matrix ────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)

plt.figure(figsize=(5,5))
disp.plot(ax=plt.gca(), cmap='Blues', colorbar=False)
plt.title(f'Fold {best_fold} Validation Confusion Matrix')
plt.show()

# ── 6) Plot ROC Curve ───────────────────────────────────────────────────────
# binarize for multiclass
y_bin   = label_binarize(y_true, classes=list(range(len(classes))))
fpr, tpr, roc_auc = {}, {}, {}

for i in range(len(classes)):
    fpr[i], tpr[i], _   = roc_curve(y_bin[:,i], probs[:,i])
    roc_auc[i]          = auc(fpr[i], tpr[i])

plt.figure()
for i, cls in enumerate(classes):
    plt.plot(fpr[i], tpr[i], label=f'{cls} (AUC={roc_auc[i]:.2f})')
plt.plot([0,1],[0,1], 'k--', alpha=0.3)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title(f'Fold {best_fold} ROC Curve')
plt.legend(loc='lower right')
plt.show()





