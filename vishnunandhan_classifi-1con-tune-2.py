import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



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

        # copy val 
        val_csv = os.path.join(base_in, f'{cond}_val.csv')
        shutil.copy(val_csv, os.path.join(base_out, f'{cond}_val.csv'))
        print(f'→ Val copied:           {base_out}/{cond}_val.csv')

    # Copy test set
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

    def _crop(self, image: np.ndarray, x: float, y: float, box: int = 64) -> Image.Image:
        """
        Crop a (2*box)x(2*box)=128x128 patch centered at (x,y),
        then resize to 224x224 with bicubic interpolation.
        """
        img = Image.fromarray(image)
        left, top   = int(x - box), int(y - box)
        right, bottom = int(x + box), int(y + box)
        patch = img.crop((left, top, right, bottom))
        return patch.resize((224, 224), Image.BICUBIC)
    

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
#           (50 layers frozen, Flatten-only head, L2 on hidden layers, no reg on final)
#   RADNET IMAGE
# ─────────────────────────────────────────────────────────────────────────────

import os
import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────────
CONDITION       = 'Neural_Foraminal_Narrowing'
FOLDS           = list(range(5))
IMAGE_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing_label/fold_{fold}'
TEST_IMAGE_DIR  = f'/kaggle/working/{CONDITION}/test'
TEST_LABEL_CSV  = f'/kaggle/working/{CONDITION}_label/test/{CONDITION}_test_labels.csv'
RESULTS_DIR     = Path(f'./results/{CONDITION}')

BATCH_SIZE      = 16
IMG_SIZE        = (224, 224)
EPOCHS          = 10
PATIENCE        = 5
LEARNING_RATE   = 1e-4
L2_WEIGHT       = 1e-4
DROP_RATE       = 0.5

# ── Focal Loss (per-class α) ──────────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=None):
    def loss_fn(y_true, y_pred):
        eps    = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        ce     = -y_true * K.log(y_pred)
        mod    = K.pow(1. - y_pred, gamma)
        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_factor = y_true * alpha_tensor
            ce = ce * alpha_factor
        return K.sum(mod * ce, axis=-1)
    return loss_fn

fold_results = []

for fold in FOLDS:
    print(f"\n▶ Training fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(fold=fold)

    # — Load & preprocess labels
    train_df = pd.read_csv(Path(LABEL_ROOT)/f'{CONDITION}_augmented_labels.csv')
    val_df   = pd.read_csv(Path(LABEL_ROOT)/f'{CONDITION}_val_labels.csv')
    for df in (train_df, val_df):
        df['subject'] = df['subject'].str.replace('_augmented','',regex=False)
        df['label']   = (df['label'].astype(int)-1).astype(str)

    # — Compute class weights
    y_train_int = train_df['label'].astype(int).values
    classes     = np.unique(y_train_int)
    cw_array    = class_weight.compute_class_weight(
                     class_weight='balanced',
                     classes=classes,
                     y=y_train_int)
    class_weight_dict = dict(zip(classes, cw_array))

    # — Compute per-class α for focal loss
    counts = np.bincount(y_train_int, minlength=len(classes))
    freqs  = counts / counts.sum()
    alpha  = ((1 - freqs)/(1 - freqs).sum()).tolist()

    # — Generators
    datagen    = ImageDataGenerator(rescale=1./255)
    train_gen  = datagen.flow_from_dataframe(
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

     # RADNET IMAGE WEIGHTS
    # 1) instantiate ResNet50 without any weights
    base = ResNet50(
        include_top=False,
        weights=None,
        input_shape=IMG_SIZE + (3,)
    )
    # 2) load RadImageNet “notop” weights
    rad_path = "/kaggle/input/notop-wg-radimagenet/RadImageNet-ResNet50_notop.h5"
    base.load_weights(rad_path)
    

    # freeze 50 layers
    for layer in base.layers[:50]:
        layer.trainable = False
    for layer in base.layers[50:]:
        layer.trainable = True

    # — add our head
    x = Flatten()(base.output)
    x = Dense(256, activation='relu', kernel_regularizer=l2(L2_WEIGHT))(x)
    x = BatchNormalization()(x)
    x = Dropout(DROP_RATE)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(L2_WEIGHT))(x)
    x = Dropout(DROP_RATE)(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=preds)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(gamma=2.0, alpha=alpha),
        metrics=['accuracy']
    )

    # — Callbacks
    out_dir = RESULTS_DIR / f'fold_{fold}'; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt  = ModelCheckpoint(out_dir/'best_model.keras', monitor='val_loss',
                             save_best_only=True, verbose=1)
    es    = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                          restore_best_weights=True, verbose=1)
    rl    = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=1e-6, verbose=1)

    # — Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[es, ckpt, rl],
        class_weight=class_weight_dict,
        verbose=2
    )
    with open(out_dir/'history.json','w') as f:
        json.dump(history.history, f)

    # — Validate
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    y_true = val_gen.classes
    fm = {
        'fold': fold,
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall':    recall_score(y_true, y_pred, average='weighted'),
        'F1 Score':  f1_score(y_true, y_pred, average='weighted')
    }
    fold_results.append(fm)
    with open(out_dir/'metrics.txt','w') as f:
        for k,v in fm.items():
            if k!='fold': f.write(f"{k}: {v:.4f}\n")
    print(f"✅ Fold {fold} metrics:", fm)

    tf.keras.backend.clear_session()
    gc.collect()

# ── Pick best fold & evaluate on held-out test ───────────────────────────────
best = max(fold_results, key=lambda x: x['F1 Score'])
best_fold = best['fold']
print(f"\n▶ Best fold selected: {best_fold} (F1 = {best['F1 Score']:.4f})")

print("▶ Evaluating held-out test…")
model = load_model(
    RESULTS_DIR/f'fold_{best_fold}'/'best_model.keras',
    custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=alpha)}
)

test_df = pd.read_csv(TEST_LABEL_CSV)
test_df['label'] = (test_df['label'].astype(int) - 1).astype(str)
test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df, directory=TEST_IMAGE_DIR,
    x_col='subject', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

probs  = model.predict(test_gen, verbose=1)
y_pred = np.argmax(probs, axis=1)
y_true = test_gen.classes

acc_test   = accuracy_score(y_true, y_pred)
prec_test  = precision_score(y_true, y_pred, average='weighted')
rec_test   = recall_score(y_true, y_pred, average='weighted')
f1_test    = f1_score(y_true, y_pred, average='weighted')
cm         = confusion_matrix(y_true, y_pred)
y_true_bin = label_binarize(y_true, classes=[0,1,2])

test_out = RESULTS_DIR / 'test'
test_out.mkdir(exist_ok=True)
with open(test_out/'metrics_test.txt','w') as f:
    f.write(f"Accuracy: {acc_test:.4f}\n"
            f"Precision: {prec_test:.4f}\n"
            f"Recall:    {rec_test:.4f}\n"
            f"F1 Score:  {f1_test:.4f}\n")
np.savetxt(test_out/'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
with open(test_out/'roc_auc.txt','w') as f:
    for i in range(3):
        f.write(f"Class {i} AUC: {auc(*roc_curve(y_true_bin[:,i], probs[:,i])[:2]):.4f}\n")

plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve(y_true_bin[:,i], probs[:,i])
    plt.plot(fpr, tpr, label=f'Class {i} (AUC={auc(fpr,tpr):.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curves on Held-Out Test Set')
plt.legend(loc='lower right')
plt.savefig(test_out/'roc_curves.png')
plt.close()

print(f"✅ Test evaluation saved to {test_out}")



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Summary for CV & Held-Out Test – Loss Curves, Avg Metrics, Confusion & ROC
# ─────────────────────────────────────────────────────────────────────────────

import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ── Config ───────────────────────────────────────────────────────────────────
CONDITION      = 'Neural_Foraminal_Narrowing'
RESULTS_DIR    = Path(f'./results/{CONDITION}')
LABEL_DIR      = Path(f'/kaggle/working/{CONDITION}_label')
IMG_DIR        = Path(f'/kaggle/working/{CONDITION}')
TEST_IMAGE_DIR = IMG_DIR / 'test'
TEST_LABEL_CSV = LABEL_DIR / 'test' / f'{CONDITION}_test_labels.csv'

BATCH_SIZE = 16
IMG_SIZE   = (224, 224)

# ── Focal Loss (per‐class α) – must exactly match training cell! ──────────────
def focal_loss(gamma=2.0, alpha=None):
    """
    gamma: focusing parameter
    alpha: None or list of length num_classes
    """
    def loss_fn(y_true, y_pred):
        eps    = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        ce     = -y_true * K.log(y_pred)               # cross‐entropy
        mod    = K.pow(1. - y_pred, gamma)             # modulating factor

        if alpha is not None:
            # per‐class alpha weighting
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_factor = y_true * alpha_tensor
            ce = ce * alpha_factor

        loss = mod * ce
        return K.sum(loss, axis=-1)                    # sum over classes
    return loss_fn

# ── Gather fold dirs ─────────────────────────────────────────────────────────
fold_dirs = sorted(RESULTS_DIR.glob('fold_*'),
                   key=lambda p: int(p.name.split('_')[1]))

# ── Containers ───────────────────────────────────────────────────────────────
train_metrics, val_metrics = [], []
loss_curves = {}

# ── 1) Collect CV train/val losses & metrics ─────────────────────────────────
for fd in fold_dirs:
    fold = int(fd.name.split('_')[1])
    # a) load history
    hist = json.load(open(fd/'history.json'))
    loss_curves[fold] = (hist['loss'], hist['val_loss'])

    # b) load model (only for naming the custom loss – won't re‐train)
    model = load_model(
        fd/'best_model.keras',
        custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=None)}
    )

    # c) prepare train generator
    train_csv = LABEL_DIR/f'fold_{fold}'/f'{CONDITION}_augmented_labels.csv'
    df_train  = pd.read_csv(train_csv)
    df_train['subject'] = df_train['subject'].str.replace('_augmented','',regex=False)
    df_train['label']   = (df_train['label'].astype(int)-1).astype(str)
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        df_train,
        directory=IMG_DIR/f'fold_{fold}'/'train',
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    # d) prepare val generator
    val_csv = LABEL_DIR/f'fold_{fold}'/f'{CONDITION}_val_labels.csv'
    df_val  = pd.read_csv(val_csv)
    df_val['subject'] = df_val['subject'].str.replace('_augmented','',regex=False)
    df_val['label']   = (df_val['label'].astype(int)-1).astype(str)
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        df_val,
        directory=IMG_DIR/f'fold_{fold}'/'val',
        x_col='subject', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    # e) get predictions to compute metrics
    y_tr_pred = np.argmax(model.predict(train_gen), axis=1)
    y_tr_true = train_gen.classes
    y_vl_pred = np.argmax(model.predict(val_gen),   axis=1)
    y_vl_true = val_gen.classes

    train_metrics.append({
        'Accuracy':  accuracy_score(y_tr_true, y_tr_pred),
        'Precision': precision_score(y_tr_true, y_tr_pred, average='weighted'),
        'Recall':    recall_score(y_tr_true, y_tr_pred, average='weighted'),
        'F1 Score':  f1_score(y_tr_true, y_tr_pred, average='weighted')
    })
    val_metrics.append({
        'Accuracy':  accuracy_score(y_vl_true, y_vl_pred),
        'Precision': precision_score(y_vl_true, y_vl_pred, average='weighted'),
        'Recall':    recall_score(y_vl_true, y_vl_pred, average='weighted'),
        'F1 Score':  f1_score(y_vl_true, y_vl_pred, average='weighted')
    })

    # clear for next fold
    tf.keras.backend.clear_session()
    gc.collect()

# ── 2) Plot all CV loss curves ────────────────────────────────────────────────
plt.figure(figsize=(8,5))
for fold,(loss,vloss) in loss_curves.items():
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss,  label=f'Fold {fold} Train', alpha=0.6)
    plt.plot(epochs, vloss, '--', label=f'Fold {fold} Val',   alpha=0.6)
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Train & Val Loss Curves (All Folds)')
plt.legend(ncol=2, fontsize='small')
plt.tight_layout()
plt.show()

# ── 3) Average CV metrics ────────────────────────────────────────────────────
def avg(ms):
    return {k: np.mean([m[k] for m in ms]) for k in ms[0]}

avg_tr = avg(train_metrics)
avg_vl = avg(val_metrics)

print("➡️  Average CV TRAIN metrics:")
for k,v in avg_tr.items():
    print(f"   {k}: {v:.3f}")
print("➡️  Average CV   VAL metrics:")
for k,v in avg_vl.items():
    print(f"   {k}: {v:.3f}")

# ── 4) Held-out test on best fold ────────────────────────────────────────────
best_fold = int(np.argmax([m['F1 Score'] for m in val_metrics]))
print(f"\n✨ Best fold = {best_fold}")

model = load_model(
    RESULTS_DIR/f'fold_{best_fold}'/'best_model.keras',
    custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=None)}
)

test_df = pd.read_csv(TEST_LABEL_CSV)
test_df['label'] = (test_df['label'].astype(int)-1).astype(str)
test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df,
    directory=TEST_IMAGE_DIR,
    x_col='subject', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

probs  = model.predict(test_gen, verbose=0)
y_test = test_gen.classes
y_pred = np.argmax(probs, axis=1)

metrics_test = {
    'Accuracy':  accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, average='weighted'),
    'Recall':    recall_score(y_test, y_pred, average='weighted'),
    'F1 Score':  f1_score(y_test, y_pred, average='weighted'),
    'Error':     1 - accuracy_score(y_test, y_pred)
}

print("\n➡️ Held-out TEST metrics:")
for k,v in metrics_test.items():
    print(f"   {k}: {v:.3f}")

# ── 5) Confusion & ROC on held-out test ───────────────────────────────────────
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=list(test_gen.class_indices))
plt.figure(figsize=(4,4))
disp.plot(ax=plt.gca(), cmap='Blues', colorbar=False)
plt.title('Held-out Test Confusion')
plt.show()

# ROC
y_bin = label_binarize(y_test, classes=list(range(len(test_gen.class_indices))))
plt.figure(figsize=(6,5))
for i,label in enumerate(test_gen.class_indices):
    fpr, tpr, _ = roc_curve(y_bin[:,i], probs[:,i])
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--', color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('Held-out Test ROC Curves')
plt.legend(loc='lower right')
plt.show()



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Per-Class Precision / Recall / F1 on Held-Out Test Set
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

# invert the mapping so we get label → class name
inv_map = {v:k for k,v in test_gen.class_indices.items()}
# sort by label so class order is consistent
labels = sorted(inv_map)
class_names = [inv_map[i] for i in labels]

# compute per‐class metrics
precisions, recalls, f1s, supports = precision_recall_fscore_support(
    y_true, y_pred, labels=labels, zero_division=0
)

df_metrics = pd.DataFrame({
    'Class':      class_names,
    'Support':    supports,
    'Precision':  precisions,
    'Recall':     recalls,
    'F1 Score':   f1s
})

print(df_metrics.to_string(index=False))

# print the sklearn report
print("\nFull classification report:\n")
print(classification_report(
    y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
))



# Cell: Visualize 

import random
import matplotlib.pyplot as plt
from PIL import Image

# how many samples to show
N = 4

# load the test labels DataFrame (must match the one used above)
test_df = pd.read_csv(TEST_LABEL_CSV)
test_df['label_int'] = (test_df['label'].astype(int) - 1)

# get class names from your test generator
inv_map = {v:k for k,v in test_gen.class_indices.items()}
classes = [inv_map[i] for i in range(len(inv_map))]

# pick N random rows
sample_df = test_df.sample(N, random_state=42).reset_index(drop=True)

fig, axes = plt.subplots(1, N, figsize=(4*N, 4))
for i, row in sample_df.iterrows():
    # load image
    img_path = TEST_IMAGE_DIR / row['subject']
    img = Image.open(img_path)
    
    # find its index in the generator so we can look up the prediction
    idx = test_gen.filenames.index(row['subject'])
    pred_idx = y_pred[idx]
    true_idx = row['label_int']
    
    axes[i].imshow(img, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(
        f"True: {classes[true_idx]}\nPred: {classes[pred_idx]}",
        fontsize=12
    )

plt.tight_layout()
plt.show()



# ─────────────────────────────────────────────────────────────────────────────
# Cell: Visualize Sample Cropped Patches
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Directory containing cropped patches (e.g. fold 0 train)
crop_dir = './Neural_Foraminal_Narrowing/fold_0/train'

# List all PNG files
img_files = [f for f in os.listdir(crop_dir) if f.lower().endswith('.png')]
if not img_files:
    raise RuntimeError(f"No images found in {crop_dir}")

# Randomly pick up to 9 samples
sample_files = random.sample(img_files, min(9, len(img_files)))

# Display in a 3×3 grid
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for ax, fname in zip(axes, sample_files):
    img = Image.open(os.path.join(crop_dir, fname))
    ax.imshow(img)
    ax.set_title(fname, fontsize=8)
    ax.axis('off')

# Turn off any unused subplots
for ax in axes[len(sample_files):]:
    ax.axis('off')

plt.suptitle('Sample Cropped Patches – Fold 0 Train', fontsize=16)
plt.tight_layout()
plt.show()



# Cell: Plot Train vs Val Loss per Fold
import json
import matplotlib.pyplot as plt
from pathlib import Path

for fold in FOLDS:
    hist_path = RESULTS_DIR / f'fold_{fold}' / 'history.json'
    hist = json.load(open(hist_path))
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, hist['loss'],    marker='o', label='Train Loss')
    plt.plot(epochs, hist['val_loss'],marker='s', label='Val Loss')
    plt.title(f'Fold {fold} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


