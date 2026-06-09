import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tqdm import tqdm

!pip install segmentation-models -q
import segmentation_models as sm


BASE_DIR = '/kaggle/input/severstal-steel-defect-detection/'
TRAIN_CSV_PATH = os.path.join(BASE_DIR, 'train.csv')
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, 'train_images/')
TEST_IMAGE_DIR = os.path.join(BASE_DIR, 'test_images/')

IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 16
EPOCHS = 40
NUM_CLASSES = 4
N_TILES_PER_IMAGE = 6
TARGET_DIM = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
THRESHOLD = 0.5


df_train = pd.read_csv(TRAIN_CSV_PATH)
print(f"Eğitim verisinde {df_train.shape[0]} satır bulunuyor.")
print(df_train.head())

print(f"Eşsiz görüntü sayısı: {df_train['ImageId'].nunique()}")

df_train['has_mask'] = ~df_train['EncodedPixels'].isna()
class_counts = df_train[df_train['has_mask']]['ClassId'].value_counts().sort_index()
print("\nKusur Sınıfı Dağılımları (maskesi olanlar için):")
print(class_counts)

plt.figure(figsize=(8, 5))
class_counts.plot(kind='bar')
plt.title('Kusur Sınıfı Dağılımları')
plt.xlabel('Sınıf ID')
plt.ylabel('Sayı')
plt.xticks(rotation=0)
plt.show()


def rle_to_mask(rle_string, height, width):
    rows, cols = height, width
    if pd.isna(rle_string):
        return np.zeros(rows * cols, dtype=np.uint8).reshape(rows, cols)

    rle_numbers = [int(numstring) for numstring in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1, 2)

    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index:index+length] = 1

    img = img.reshape(cols, rows)
    return img.T


sample_df = df_train[df_train['EncodedPixels'].notnull()].iloc[0]
image_id = sample_df['ImageId']
class_id = sample_df['ClassId']
rle_pixels = sample_df['EncodedPixels']

original_height = 256
original_width = 1600

img_path = os.path.join(TRAIN_IMAGE_DIR, image_id)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = rle_to_mask(rle_pixels, original_height, original_width)

print(f"Görüntü ID: {image_id}, Sınıf ID: {class_id}")
print(f"Görüntü boyutu: {img.shape}")
print(f"Maske boyutu: {mask.shape}")
print(f"Maskedeki piksel değerleri: {np.unique(mask)}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(f"Orijinal Görüntü: {image_id}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title(f"Maske (Sınıf {class_id})")
plt.axis('off')

plt.show()


class SteelDataGenerator(Sequence):
    def __init__(self, image_ids, annotations_df, batch_size, target_dim=TARGET_DIM,
                 n_channels=3, n_classes=NUM_CLASSES, shuffle=True, augmentations=None,
                 base_image_dir=TRAIN_IMAGE_DIR, n_tiles_per_image=N_TILES_PER_IMAGE):
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.annotations_df = annotations_df
        self.base_image_dir = base_image_dir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.n_tiles_per_image = n_tiles_per_image

        self.samples = []
        for img_id in image_ids:
            for tile_idx in range(self.n_tiles_per_image):
                self.samples.append((img_id, tile_idx))

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_samples = [self.samples[k] for k in indexes]
        X, y = self.__data_generation(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_samples):
        X = np.empty((self.batch_size, *self.target_dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.target_dim, self.n_classes), dtype=np.uint8)

        original_height = 256
        original_width = 1600

        for i, (image_id, tile_idx) in enumerate(batch_samples):
            img_path = os.path.join(self.base_image_dir, image_id)
            img = cv2.imread(img_path)
            img_annotations = self.annotations_df[self.annotations_df['ImageId'] == image_id]
            combined_mask_original_size = np.zeros((original_height, original_width, self.n_classes), dtype=np.uint8)

            for _, row in img_annotations.iterrows():
                class_id = int(row['ClassId'])
                rle_pixels = row['EncodedPixels']
                if pd.notna(rle_pixels):
                    individual_mask = rle_to_mask(rle_pixels, original_height, original_width)
                    combined_mask_original_size[:, :, class_id - 1] = individual_mask
            
            start_col = tile_idx * self.target_dim[1]
            end_col = start_col + self.target_dim[1]

            img_tile = img[:, start_col:end_col, :]
            mask_tile_multi_channel = combined_mask_original_size[:, start_col:end_col, :]
            
            X[i,] = img_tile.astype(np.float32) / 255.0
            y[i,] = mask_tile_multi_channel.astype(np.uint8)
            
            if self.augmentations:
                 augmented = self.augmentations(image=X[i,], mask=y[i,])
                 X[i,] = augmented['image']
                 y[i,] = augmented['mask']
        return X, y


unique_image_ids = df_train['ImageId'].unique()
train_ids, val_ids = train_test_split(unique_image_ids, test_size=0.15, random_state=42)

print(f"Eğitim için eşsiz görüntü sayısı: {len(train_ids)}")
print(f"Doğrulama için eşsiz görüntü sayısı: {len(val_ids)}")

train_generator = SteelDataGenerator(train_ids, df_train, BATCH_SIZE)
val_generator = SteelDataGenerator(val_ids, df_train, BATCH_SIZE, shuffle=False)

sample_batch_x, sample_batch_y = train_generator[0]
print(f"X batch shape: {sample_batch_x.shape}")
print(f"y batch shape: {sample_batch_y.shape}")
print(f"y batch data type: {sample_batch_y.dtype}")
print(f"Unique values in y batch sample mask (first sample, first channel): {np.unique(sample_batch_y[0,:,:,0])}")


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


BACKBONE = 'efficientnetb4'
model = sm.Unet(BACKBONE,
                input_shape=INPUT_SHAPE,
                classes=NUM_CLASSES,
                activation='sigmoid',
                encoder_weights='imagenet')

model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=[dice_coefficient])
model.summary()


checkpoint_filepath = 'best_unet_model_with_backbone.keras'

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_dice_coefficient',
    mode='max',
    save_best_only=True,
    verbose=1)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_dice_coefficient',
    factor=0.2,
    patience=3, 
    min_lr=1e-6,
    mode='max',
    verbose=1)

early_stopping_callback = EarlyStopping(
    monitor='val_dice_coefficient',
    patience=10, 
    mode='max',
    restore_best_weights=True,
    verbose=1)

callbacks_list = [
    model_checkpoint_callback,
    reduce_lr_callback,
    early_stopping_callback
]

print("Model eğitimi başlıyor...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks_list,
    verbose=1
)
print("Model eğitimi tamamlandı!")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['dice_coefficient'], label='Eğitim Dice Coefficient')
plt.plot(history.history['val_dice_coefficient'], label='Doğrulama Dice Coefficient')
plt.title('Dice Coefficient vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Doğrulama Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (1 - Dice Coefficient)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


best_model_path = checkpoint_filepath

custom_objects = {
    'dice_loss': dice_loss,
    'dice_coefficient': dice_coefficient
}
if not os.path.exists(best_model_path):
    print(f"Kaydedilmiş model bulunamadı: {best_model_path}. Lütfen önce modeli eğitin ve kaydedin.")
else:
    loaded_best_model = load_model(best_model_path, custom_objects=custom_objects)
    print("En iyi model başarıyla yüklendi.")

    print("Yüklenen modelin doğrulama seti üzerinde değerlendirilmesi:")
    results = loaded_best_model.evaluate(val_generator, steps=len(val_generator), verbose=1)
    print(f"Doğrulama Loss: {results[0]:.4f}")
    print(f"Doğrulama Dice Coefficient: {results[1]:.4f}")

    sample_val_images, sample_val_masks_true = val_generator[0]
    sample_val_masks_pred = loaded_best_model.predict(sample_val_images, batch_size=BATCH_SIZE)
    sample_val_masks_pred_binary = (sample_val_masks_pred > THRESHOLD).astype(np.uint8)

    num_samples_to_show = 3
    plt.figure(figsize=(15, num_samples_to_show * 5))

    for i in range(num_samples_to_show):
        plt.subplot(num_samples_to_show, 3, i * 3 + 1)
        plt.imshow(sample_val_images[i])
        plt.title(f"Örnek {i+1}: Orijinal Görüntü")
        plt.axis('off')

        plt.subplot(num_samples_to_show, 3, i * 3 + 2)
        plt.imshow(sample_val_masks_true[i, :, :, 2], cmap='gray')
        plt.title(f"Gerçek Maske (Sınıf 3)")
        plt.axis('off')

        plt.subplot(num_samples_to_show, 3, i * 3 + 3)
        plt.imshow(sample_val_masks_pred_binary[i, :, :, 2], cmap='gray')
        plt.title(f"Tahmini Maske (Sınıf 3)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def mask_to_rle(mask_img):
    pixels = mask_img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if 'loaded_best_model' in locals() or 'loaded_best_model' in globals():
    test_image_files = os.listdir(TEST_IMAGE_DIR)
    if not test_image_files:
        print("Test edilecek görüntü bulunamadı.")
    else:
        print(f"Test edilecek görüntü sayısı: {len(test_image_files)}")
        submission_data = []
        original_height = 256
        original_width = 1600

        for img_file in tqdm(test_image_files):
            img_path = os.path.join(TEST_IMAGE_DIR, img_file)
            img = cv2.imread(img_path)

            full_pred_prob_mask = np.zeros((original_height, original_width, NUM_CLASSES), dtype=np.float32)

            for tile_idx in range(N_TILES_PER_IMAGE):
                start_col = tile_idx * TARGET_DIM[1]
                end_col = start_col + TARGET_DIM[1]

                img_tile = img[:, start_col:end_col, :]
                img_tile_processed = img_tile.astype(np.float32) / 255.0
                img_tile_processed = np.expand_dims(img_tile_processed, axis=0)

                tile_pred_prob = loaded_best_model.predict(img_tile_processed, verbose=0)[0]
                full_pred_prob_mask[:, start_col:end_col, :] = tile_pred_prob

            full_pred_binary_mask = (full_pred_prob_mask > THRESHOLD).astype(np.uint8)

            for class_id_idx in range(NUM_CLASSES):
                class_id_actual = class_id_idx + 1
                rle_encoded_pixels = mask_to_rle(full_pred_binary_mask[:, :, class_id_idx])

                if len(rle_encoded_pixels) == 0:
                    rle_encoded_pixels = ''

                submission_data.append({
                    'ImageId_ClassId': f"{img_file}_{class_id_actual}",
                    'EncodedPixels': rle_encoded_pixels
                })

        submission_df = pd.DataFrame(submission_data)
        print("\nÖrnek Submission Satırları:")
        print(submission_df.head())

        submission_df.to_csv('submission.csv', index=False)
        print("\submission.csv dosyası başarıyla oluşturuldu!")
else:
    print("Model yüklenemediği için submission dosyası oluşturulamadı.")

