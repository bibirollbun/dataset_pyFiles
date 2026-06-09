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



#neural foraminal training cell
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
CONDITION = 'Neural_Foraminal_Narrowing'
FOLDS = [0, 1]
IMAGE_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Neural_Foraminal_Narrowing_label/fold_{fold}'
BATCH_SIZE = 16
PATIENCE = 5
EPOCHS = 10
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

# For each fold
for fold in FOLDS:
    print(f"\n▶ Training {CONDITION}, fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(fold=fold)

    # Load and process CSVs
    train_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_augmented_labels.csv'))
    val_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_val_labels.csv'))

    # Remove "_augmented" from filenames
    train_df['subject'] = train_df['subject'].str.replace('_augmented', '', regex=False)
    val_df['subject'] = val_df['subject'].str.replace('_augmented', '', regex=False)

    # Adjust label indexing (ensure string classes: "0", "1", "2")
    train_df['label'] = (train_df['label'].astype(int) - 1).astype(str)
    val_df['label'] = (val_df['label'].astype(int) - 1).astype(str)

    # Data generators
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(IMAGE_ROOT, 'train'),
        x_col='subject',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=os.path.join(IMAGE_ROOT, 'val'),
        x_col='subject',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(gamma=2.0, alpha=0.5),
        metrics=['accuracy']
    )

    out_dir = Path(f'./results/{CONDITION}/fold_{fold}')
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=out_dir / 'best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluation
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    y_true = val_gen.classes
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

    # Clean up memory
    tf.keras.backend.clear_session()
    import gc; gc.collect()



#spinal canal steno
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
CONDITION = 'Spinal_Canal_Stenosis'
FOLDS = [0, 1]
IMAGE_ROOT_TMPL = '/kaggle/working/Spinal_Canal_Stenosis/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Spinal_Canal_Stenosis_label/fold_{fold}'
BATCH_SIZE = 16
PATIENCE = 5
EPOCHS = 10
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

# For each fold
for fold in FOLDS:
    print(f"\n▶ Training {CONDITION}, fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(fold=fold)

    # Load and process CSVs
    train_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_augmented_labels.csv'))
    val_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_val_labels.csv'))

    # Remove "_augmented" from filenames
    train_df['subject'] = train_df['subject'].str.replace('_augmented', '', regex=False)
    val_df['subject'] = val_df['subject'].str.replace('_augmented', '', regex=False)

    # Adjust label indexing (ensure string classes: "0", "1", "2")
    train_df['label'] = (train_df['label'].astype(int) - 1).astype(str)
    val_df['label'] = (val_df['label'].astype(int) - 1).astype(str)

    # Data generators
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(IMAGE_ROOT, 'train'),
        x_col='subject',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=os.path.join(IMAGE_ROOT, 'val'),
        x_col='subject',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(gamma=2.0, alpha=0.5),
        metrics=['accuracy']
    )

    out_dir = Path(f'./results/{CONDITION}/fold_{fold}')
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=out_dir / 'best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluation
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    y_true = val_gen.classes
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

    # Clean up memory
    tf.keras.backend.clear_session()
    import gc; gc.collect()



#spinal canal stenosis training cell
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
CONDITION = 'Subarticular_Stenosis'
FOLDS = [0, 1]
IMAGE_ROOT_TMPL = '/kaggle/working/Subarticular_Stenosis/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/Subarticular_Stenosis_label/fold_{fold}'
BATCH_SIZE = 16
PATIENCE = 5
EPOCHS = 10
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

# For each fold
for fold in FOLDS:
    print(f"\n▶ Training {CONDITION}, fold {fold}")
    IMAGE_ROOT = IMAGE_ROOT_TMPL.format(fold=fold)
    LABEL_ROOT = LABEL_ROOT_TMPL.format(fold=fold)

    # Load and process CSVs
    train_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_augmented_labels.csv'))
    val_df = pd.read_csv(os.path.join(LABEL_ROOT, f'{CONDITION}_val_labels.csv'))

    # Remove "_augmented" from filenames
    train_df['subject'] = train_df['subject'].str.replace('_augmented', '', regex=False)
    val_df['subject'] = val_df['subject'].str.replace('_augmented', '', regex=False)

    # Adjust label indexing (ensure string classes: "0", "1", "2")
    train_df['label'] = (train_df['label'].astype(int) - 1).astype(str)
    val_df['label'] = (val_df['label'].astype(int) - 1).astype(str)

    # Data generators
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(IMAGE_ROOT, 'train'),
        x_col='subject',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=os.path.join(IMAGE_ROOT, 'val'),
        x_col='subject',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(gamma=2.0, alpha=0.5),
        metrics=['accuracy']
    )

    out_dir = Path(f'./results/{CONDITION}/fold_{fold}')
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=out_dir / 'best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluation
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    y_true = val_gen.classes
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

    # Clean up memory
    tf.keras.backend.clear_session()
    import gc; gc.collect()



import os
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

# --- must match your training config ---
CONDITIONS = [
    'Neural_Foraminal_Narrowing',
    'Spinal_Canal_Stenosis',
    'Subarticular_Stenosis'
]
FOLDS = [0, 1]
IMAGE_ROOT_TMPL = '/kaggle/working/{condition}/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/{condition}_label/fold_{fold}'
BATCH_SIZE = 16
IMG_SIZE = (224, 224)

# focal loss factory (needed to load)
def focal_loss(gamma=2., alpha=0.25):
    def loss_fn(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.-eps)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(alpha * (1.-y_pred)**gamma * ce, axis=1)
    return loss_fn

for condition in CONDITIONS:
    # accumulate across folds
    y_true_all = []
    y_score_all = []
    class_indices = None

    for fold in FOLDS:
        print(f"→ Loading {condition} / fold {fold}")
        # 1) reload val labels
        label_root = LABEL_ROOT_TMPL.format(condition=condition, fold=fold)
        val_df = pd.read_csv(os.path.join(label_root, f'{condition}_val_labels.csv'))
        val_df['subject'] = val_df['subject'].str.replace('_augmented','',regex=False)
        val_df['label']   = (val_df['label'].astype(int)-1).astype(str)

        # 2) rebuild val generator
        datagen = ImageDataGenerator(rescale=1./255)
        img_dir = os.path.join(IMAGE_ROOT_TMPL.format(condition=condition, fold=fold), 'val')
        val_gen = datagen.flow_from_dataframe(
            val_df, directory=img_dir,
            x_col='subject', y_col='label',
            target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', shuffle=False
        )

        if class_indices is None:
            class_indices = val_gen.class_indices  # save mapping str→int

        # 3) load model
        model_path = Path(f'./results/{condition}/fold_{fold}/best_model.keras')
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'loss_fn': focal_loss(2.0,0.5),
                            'focal_loss_fixed': focal_loss(2.0,0.5)}
        )

        # 4) predict
        val_gen.reset()
        prob = model.predict(val_gen, verbose=0)
        y_score_all.append(prob)
        y_true_all.append(val_gen.classes)

        tf.keras.backend.clear_session()

    # concat across folds
    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)
    y_pred  = np.argmax(y_score, axis=1)

    # invert class_indices to get labels in order
    inv_map = {v:k for k,v in class_indices.items()}
    labels = [inv_map[i] for i in range(len(inv_map))]

    # ----- confusion matrix -----
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{condition} — Confusion Matrix\n(all folds)')
    plt.show()

    # ----- ROC curves -----
    n_classes = len(inv_map)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr = dict(); tpr = dict(); roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # macro-average
    fpr["macro"], tpr["macro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label=f'Class {labels[i]} (AUC={roc_auc[i]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], linestyle='--',
             label=f'Macro-avg (AUC={roc_auc["macro"]:.2f})')
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{condition} — ROC Curves (all folds)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- must match your training config ---
CONDITIONS = [
    'Neural_Foraminal_Narrowing',
    'Spinal_Canal_Stenosis',
    'Subarticular_Stenosis'
]
FOLDS = [0, 1]
IMAGE_ROOT_TMPL = '/kaggle/working/{condition}/fold_{fold}'
LABEL_ROOT_TMPL = '/kaggle/working/{condition}_label/fold_{fold}'
RESULTS_DIR    = './results/{condition}/fold_{fold}/best_model.keras'
BATCH_SIZE = 16
IMG_SIZE   = (224, 224)

# focal loss factory (for loading .keras)
def focal_loss(gamma=2., alpha=0.25):
    def loss_fn(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.-eps)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(alpha * (1.-y_pred)**gamma * ce, axis=1)
    return loss_fn

for condition in CONDITIONS:
    for fold in FOLDS:
        print(f"\n=== {condition} | fold {fold} ===")

        # 1) load & prep train labels + generator
        label_root = LABEL_ROOT_TMPL.format(condition=condition, fold=fold)
        train_df = pd.read_csv(os.path.join(label_root, f'{condition}_augmented_labels.csv'))
        train_df['subject'] = train_df['subject'].str.replace('_augmented','',regex=False)
        train_df['label']   = (train_df['label'].astype(int)-1).astype(str)

        datagen = ImageDataGenerator(rescale=1./255)
        train_gen = datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=os.path.join(IMAGE_ROOT_TMPL.format(condition=condition, fold=fold),'train'),
            x_col='subject', y_col='label',
            target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', shuffle=False
        )

        # 2) load & prep val labels + generator
        val_df = pd.read_csv(os.path.join(label_root, f'{condition}_val_labels.csv'))
        val_df['subject'] = val_df['subject'].str.replace('_augmented','',regex=False)
        val_df['label']   = (val_df['label'].astype(int)-1).astype(str)

        val_gen = datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=os.path.join(IMAGE_ROOT_TMPL.format(condition=condition, fold=fold),'val'),
            x_col='subject', y_col='label',
            target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', shuffle=False
        )

        # invert class_indices → sorted label names
        inv_map = {v:k for k,v in train_gen.class_indices.items()}
        labels = [inv_map[i] for i in range(len(inv_map))]

        # 3) load model once
        model = tf.keras.models.load_model(
            Path(RESULTS_DIR.format(condition=condition, fold=fold)),
            custom_objects={'loss_fn':         focal_loss(2.0,0.5),
                            'focal_loss_fixed': focal_loss(2.0,0.5)}
        )

        # 4) eval & plot for both splits
        for split, gen in [('train', train_gen), ('val', val_gen)]:
            print(f"\n→ {split.upper()} results:")
            gen.reset()
            y_score = model.predict(gen, verbose=0)
            y_true  = gen.classes
            y_pred  = np.argmax(y_score, axis=1)

            # metrics
            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            print(f"  Acc: {acc:.3f}  Prec: {prec:.3f}  Rec: {rec:.3f}  F1: {f1:.3f}")

            # confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5,5))
            disp = ConfusionMatrixDisplay(cm, display_labels=labels)
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'{condition} | fold {fold} | {split} Confusion Matrix')
            plt.show()

            # ROC
            n_classes = len(labels)
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:,i], y_score[:,i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            fpr["macro"], tpr["macro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            plt.figure(figsize=(6,5))
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i],
                         label=f'{labels[i]} (AUC={roc_auc[i]:.2f})')
            plt.plot(fpr["macro"], tpr["macro"], linestyle='--',
                     label=f'Macro (AUC={roc_auc["macro"]:.2f})')
            plt.plot([0,1],[0,1],'k--', alpha=0.3)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{condition} | fold {fold} | {split} ROC Curves')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.show()

        tf.keras.backend.clear_session()





