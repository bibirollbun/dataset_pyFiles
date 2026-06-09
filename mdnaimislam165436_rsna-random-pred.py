import os
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet201, EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
import glob
from tqdm import tqdm


DATA_DIR = '/kaggle/input/rsna-intracranial-aneurysm-detection'
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train.csv')
SERIES_DIR = os.path.join(DATA_DIR, 'series')
TEST_DIR = os.path.join(DATA_DIR, 'test')

LABEL_COLUMNS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

IMG_SIZE = (300, 300)  # EfficientNet-B3 default
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 0.0001


train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df[LABEL_COLUMNS] = train_df[LABEL_COLUMNS].fillna(0)



class BrainAneurysmDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, series_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True):
        self.df = df
        self.series_dir = series_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for _, row in batch_df.iterrows():
            series_id = row['SeriesInstanceUID']
            series_path = os.path.join(self.series_dir, series_id)
            dcm_files = glob.glob(os.path.join(series_path, "*.dcm"))
            if len(dcm_files) > 0:
                num_slices = min(10, len(dcm_files))
                selected_files = np.random.choice(dcm_files, num_slices, replace=False)
                imgs = [preprocess_dicom(f) for f in selected_files]
                series_img = np.mean(imgs, axis=0)
            else:
                series_img = np.zeros((*self.img_size, 3), dtype=np.float32)
            X.append(series_img)
            y.append(row[LABEL_COLUMNS].values)
        X = np.array(X, dtype=np.float32) / 255.0  # normalize
        y = np.array(y, dtype=np.float32)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)



def preprocess_dicom(dcm_path):
    try:
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)
        img = img - np.min(img)
        if np.max(img) != 0:
            img = img / np.max(img)
        img = cv2.resize(img, IMG_SIZE)
        # 3-channel
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape != (*IMG_SIZE, 3):
            img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        return img
    except:
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)


class BrainAneurysmDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, series_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True):
        self.df = df
        self.series_dir = series_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for _, row in batch_df.iterrows():
            series_id = row['SeriesInstanceUID']
            series_path = os.path.join(self.series_dir, series_id)
            dcm_files = glob.glob(os.path.join(series_path, "*.dcm"))
            if len(dcm_files) > 0:
                num_slices = min(10, len(dcm_files))
                selected_files = np.random.choice(dcm_files, num_slices, replace=False)
                imgs = [preprocess_dicom(f) for f in selected_files]
                series_img = np.mean(imgs, axis=0)
            else:
                series_img = np.zeros((*self.img_size, 3), dtype=np.float32)
            X.append(series_img)
            y.append(row[LABEL_COLUMNS].values)
        X = np.array(X, dtype=np.float32) / 255.0  # normalize
        y = np.array(y, dtype=np.float32)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)



train_data, val_data = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['Aneurysm Present']
)

train_generator = BrainAneurysmDataGenerator(train_data, SERIES_DIR, batch_size=BATCH_SIZE)
val_generator = BrainAneurysmDataGenerator(val_data, SERIES_DIR, batch_size=BATCH_SIZE, shuffle=False)



def create_model(base_model_class):
    base_model = base_model_class(
        weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    predictions = layers.Dense(len(LABEL_COLUMNS), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model



efficientnet_model = create_model(EfficientNetB3)
densenet_model = create_model(DenseNet201)


efficientnet_checkpoint = ModelCheckpoint(
    'efficientnet_model.h5', monitor='val_auc', save_best_only=True, mode='max', verbose=1
)
densenet_checkpoint = ModelCheckpoint(
    'densenet_model.h5', monitor='val_auc', save_best_only=True, mode='max', verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_auc', patience=5, mode='max', verbose=1, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_auc', factor=0.2, patience=2, min_lr=1e-6, mode='max', verbose=1
)



print("Training EfficientNet-B3 model...")
efficientnet_history = efficientnet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[efficientnet_checkpoint, early_stopping, reduce_lr]
)


print("Training DenseNet-201 model...")
densenet_history = densenet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[densenet_checkpoint, early_stopping, reduce_lr]
)


def create_ensemble_model(models):
    input_layer = layers.Input(shape=(*IMG_SIZE, 3))
    outputs = [model(input_layer) for model in models]
    avg_output = layers.Average()(outputs)
    ensemble_model = Model(inputs=input_layer, outputs=avg_output)
    ensemble_model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return ensemble_model

ensemble_model = create_ensemble_model([efficientnet_model, densenet_model])

ensemble_checkpoint = ModelCheckpoint(
    'ensemble_model.h5', monitor='val_auc', save_best_only=True, mode='max', verbose=1
)

print("Training Ensemble model...")
ensemble_history = ensemble_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[ensemble_checkpoint, early_stopping, reduce_lr]
)



def predict_test_data(model, test_dir):
    test_series_ids = os.listdir(test_dir)
    predictions = []
    for series_id in tqdm(test_series_ids):
        series_path = os.path.join(test_dir, series_id)
        dcm_files = glob.glob(os.path.join(series_path, "*.dcm"))
        if len(dcm_files) > 0:
            num_slices = min(10, len(dcm_files))
            selected_files = np.random.choice(dcm_files, num_slices, replace=False)
            imgs = [preprocess_dicom(f) for f in selected_files]
            series_img = np.mean(imgs, axis=0)
        else:
            series_img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        series_img = np.expand_dims(series_img / 255.0, axis=0)
        pred = model.predict(series_img)[0]
        predictions.append(pred)
    return predictions

print("Predicting on test data...")
test_predictions = predict_test_data(ensemble_model, TEST_DIR)

submission_df = pd.DataFrame(test_predictions, columns=LABEL_COLUMNS)
submission_df.insert(0, 'SeriesInstanceUID', os.listdir(TEST_DIR))
submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

ensemble_model.save('final_ensemble_model.h5')
print("Model saved successfully!")


