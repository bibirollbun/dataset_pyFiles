import os
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ========== CONFIGURATION ==========
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50
WARMUP_EPOCHS = 5
MIXUP_ALPHA = 0.2
DATA_DIR = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/'

# ========== LOAD DATA ==========
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')
series_desc_df = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')

# ========== LABEL ENCODING ==========
label_cols = train_df.columns[1:]
label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}

def encode_labels(row):
    return [label_map.get(row[col], 0) for col in label_cols]

train_df['encoded_labels'] = train_df.apply(encode_labels, axis=1)

def get_max_severity(encoded_labels):
    return max(encoded_labels)

train_df['max_severity'] = train_df['encoded_labels'].apply(get_max_severity)

# ========== BALANCED SAMPLING ==========
MAX_IMAGES = 400000
IMAGES_PER_STUDY = 3
MAX_STUDIES = MAX_IMAGES // IMAGES_PER_STUDY
studies_per_class = MAX_STUDIES // 3

print(f"Sampling up to {studies_per_class} studies per class...")
dfs = []
for severity in [0, 1, 2]:
    subset = train_df[train_df['max_severity'] == severity]
    sampled = subset.sample(n=min(len(subset), studies_per_class), random_state=42)
    dfs.append(sampled)
    print(f"  Class {severity}: {len(sampled)} studies")

balanced_train_df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total balanced dataset: {len(balanced_train_df)} studies\n")

# ========== HELPER FUNCTIONS ==========
def get_series_ids(study_id):
    sub_df = series_desc_df[series_desc_df['study_id'] == study_id]
    views = {'Sagittal T1': None, 'Sagittal T2/STIR': None, 'Axial T2': None}
    for view in views:
        found = sub_df[sub_df['series_description'].str.contains(view, case=False, na=False)]
        if not found.empty:
            views[view] = found.iloc[0]['series_id']
    return views

# ========== DATA AUGMENTATION ==========
def mixup_batch(X1, X2, X3, y, alpha=MIXUP_ALPHA):
    batch_size = len(X1)
    indices = np.random.permutation(batch_size)
    lam = np.random.beta(alpha, alpha, batch_size)
    
    X1_mixed = np.array([lam[i] * X1[i] + (1 - lam[i]) * X1[indices[i]] for i in range(batch_size)])
    X2_mixed = np.array([lam[i] * X2[i] + (1 - lam[i]) * X2[indices[i]] for i in range(batch_size)])
    X3_mixed = np.array([lam[i] * X3[i] + (1 - lam[i]) * X3[indices[i]] for i in range(batch_size)])
    y_mixed = np.array([lam[i] * y[i] + (1 - lam[i]) * y[indices[i]] for i in range(batch_size)])
    
    return X1_mixed, X2_mixed, X3_mixed, y_mixed

def augment_image(img):
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
    
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img * factor, 0, 1)
    
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
    
    if np.random.rand() < 0.3:
        zoom = np.random.uniform(0.9, 1.1)
        h, w = img.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)
        img = cv2.resize(img, (new_w, new_h))
        
        if zoom > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img = img[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_CONSTANT)
    
    return img

# ========== DICOM LOADING ==========
def load_dicom_image(path, augment=False):
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img / (np.max(img) + 1e-8)
        
        if augment:
            img = augment_image(img)
        
        return img
    except Exception as e:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

def load_study_images(study_id, augment=False):
    views = get_series_ids(study_id)
    images = []
    
    for view in ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']:
        series_id = views[view]
        if pd.isna(series_id):
            images.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
        else:
            series_path = os.path.join(DATA_DIR, str(study_id), str(series_id))
            if os.path.exists(series_path):
                instances = sorted(os.listdir(series_path))
                if instances:
                    img_path = os.path.join(series_path, instances[len(instances)//2])
                    images.append(load_dicom_image(img_path, augment=augment))
                else:
                    images.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
            else:
                images.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    
    return images

# ========== DATA GENERATOR ==========
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, shuffle=True, augment=False, mixup=False):
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mixup = mixup
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_ids]
        X1, X2, X3, y = [], [], [], []
        
        for _, row in batch_df.iterrows():
            imgs = load_study_images(row['study_id'], augment=self.augment)
            X1.append(imgs[0])
            X2.append(imgs[1])
            X3.append(imgs[2])
            y.append(row['encoded_labels'])
        
        X1, X2, X3 = np.array(X1), np.array(X2), np.array(X3)
        y = to_categorical(np.array(y), num_classes=3)
        
        if self.mixup and self.augment:
            X1, X2, X3, y = mixup_batch(X1, X2, X3, y)
        
        return (X1, X2, X3), y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# ========== MODEL ARCHITECTURE ==========
def create_backbone():
    base = applications.MobileNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(IMG_SIZE, IMG_SIZE, 3), 
        pooling='avg'
    )
    for layer in base.layers[-65:]:
        layer.trainable = True
    return base

def build_mvcnn():
    input1 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='sagittal_t1')
    input2 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='sagittal_t2')
    input3 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='axial_t2')
    
    backbone = create_backbone()
    feat1 = backbone(input1)
    feat2 = backbone(input2)
    feat3 = backbone(input3)
    
    merged = layers.Concatenate()([feat1, feat2, feat3])
    
    x = layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(768, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.35)(x)
    
    x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    
    output = layers.Dense(len(label_cols) * 3, activation='softmax')(x)
    output = layers.Reshape((len(label_cols), 3))(output)
    
    model = models.Model(inputs=[input1, input2, input3], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

# ========== TRAIN/VAL SPLIT ==========
print("Creating train/validation split...")
train_ids, val_ids = train_test_split(
    balanced_train_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=balanced_train_df['max_severity']
)

print(f"Train: {len(train_ids)} studies")
print(f"Val: {len(val_ids)} studies\n")

# ========== DATA GENERATORS ==========
train_gen = DataGenerator(train_ids, augment=True, mixup=True)
val_gen = DataGenerator(val_ids, augment=False, mixup=False)

# ========== BUILD MODEL ==========
print("Building model...")
model = build_mvcnn()
print(f"Total parameters: {model.count_params():,}\n")

# ========== LEARNING RATE SCHEDULE ==========
def lr_schedule(epoch, lr):
    if epoch < WARMUP_EPOCHS:
        return 1e-5 + (2e-4 - 1e-5) * (epoch / WARMUP_EPOCHS)
    else:
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 2e-4 * 0.5 * (1 + np.cos(np.pi * progress))

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

# ========== CALLBACKS ==========
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        'best_model_balanced.keras', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    ),
    lr_callback
]

# ========== TRAINING ==========
print("Starting training with warmup + MixUp...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ========== SAVE MODEL ==========
model.save('mvc_MobileNetV2_balanced_90pct.keras')
print(f"\n{'='*60}")
print("Training complete!")
print(f"Best model saved as: 'best_model_balanced.keras'")
print(f"Final model saved as: 'mvc_MobileNetV2_balanced_90pct.keras'")
print(f"{'='*60}")




