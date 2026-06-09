import os
import cv2
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
# import albumentations as A
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


import warnings
warnings.filterwarnings("ignore")


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU available. Using CPU.")


TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"
TEST_IMG_DIR  = "../input/aptos2019-blindness-detection/test_images"


train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")


def get_image_path(id_code, is_train=True):
    ext = ".png"  # Change extension if needed.
    if is_train:
        return os.path.join(TRAIN_IMG_DIR, id_code + ext)
    else:
        return os.path.join(TEST_IMG_DIR, id_code + ext)


train_df["filepath"] = train_df["id_code"].apply(lambda x: get_image_path(x, is_train=True))
test_df["filepath"]  = test_df["id_code"].apply(lambda x: get_image_path(x, is_train=False))


# train_transform = A.Compose([
#     A.HorizontalFlip(p=0.5) if True else A.NoOp(),  # do_mirror=True
#     A.RandomBrightnessContrast(brightness_limit=20/255, contrast_limit=0.2, p=0.5),
#     A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=0, p=0.5),
#     A.OneOf([
#          A.Blur(blur_limit=3, p=0.5),
#          A.Sharpen(p=0.5)
#     ], p=0.5) if True else A.NoOp(),  # blur_and_sharpen
#     A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=180, 
#                          shear_limit=0.2, border_mode=cv2.BORDER_REFLECT_101, p=0.5)
# ])


train_transform = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    rotation_range=180,
    shear_range=20,
    zoom_range=(0.8, 1.2),
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='reflect',
    # If desired, rescale pixel values before augmentations.
    rescale=1./255
)


# valid_transform = A.Compose([])


valid_transform = ImageDataGenerator()


IMG_SIZE = 299


def load_and_preprocess_image(path, transform=None):
    # Load the image from disk.
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Image not found at path: {path}")
    
    # Convert BGR to RGB and resize.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    if transform:
        # Apply a random augmentation.
        image = transform.random_transform(image)
        # Optionally, standardize the image using the generator's settings.
        # For example, if you have set rescale or samplewise normalization in your ImageDataGenerator,
        # this call will apply that standardization.
        image = transform.standardize(image)
    
    # Convert to float32 and preprocess for InceptionResNetV2.
    image = image.astype(np.float32)
    image = preprocess_input(image)
    return image



def __getitem__(self, index):
    batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
    batch_df = self.df.iloc[batch_indexes]
    
    images = []
    labels = []
    for _, row in batch_df.iterrows():
        image = load_and_preprocess_image(row["filepath"], transform=self.transform)
        images.append(image)
        if self.is_train:
            label = tf.keras.utils.to_categorical(row["diagnosis"], num_classes=self.num_classes)
            labels.append(label)
    
    images = np.stack(images, axis=0)
    if self.is_train:
        labels = np.stack(labels, axis=0)
        return images, labels
    else:
        return images


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, transform=None, is_train=True, num_classes=5, shuffle=True):
        self.df = df.copy().reset_index(drop=True)
        self.batch_size = batch_size
        self.transform = transform
        self.is_train = is_train
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        
        images = []
        labels = []
        for _, row in batch_df.iterrows():
            image = load_and_preprocess_image(row["filepath"], transform=self.transform)
            images.append(image)
            if self.is_train:
                label = tf.keras.utils.to_categorical(row["diagnosis"], num_classes=self.num_classes)
                labels.append(label)
        
        images = np.stack(images, axis=0)
        if self.is_train:
            labels = np.stack(labels, axis=0)
            return images, labels
        else:
            return images


train_df_split, valid_df_split = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df["diagnosis"])


BATCH_SIZE = 32
train_gen = DataGenerator(train_df_split, batch_size=BATCH_SIZE, transform=train_transform, is_train=True)
valid_gen = DataGenerator(valid_df_split, batch_size=BATCH_SIZE, transform=valid_transform, is_train=True)


weights_path = '/kaggle/input/inceptionresnetv2/keras/default/1/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'


base_model = InceptionResNetV2(include_top=False, 
                               weights=weights_path, 
                               input_shape=(IMG_SIZE, IMG_SIZE, 3))


for layer in base_model.layers:
    layer.trainable = True


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100)(x)
x = Dropout(0.3)(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', 'precision', 'recall'])
# model.summary()


checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop  = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min', restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7)


EPOCHS = 50
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=valid_gen,
    # callbacks=[checkpoint, earlystop, reduce_lr],
    callbacks=[TqdmCallback(verbose=1), checkpoint, earlystop, reduce_lr],
    # workers=4,
    # verbose = 1,
    # use_multiprocessing=True
)


test_gen = DataGenerator(test_df, 
                         batch_size=BATCH_SIZE, 
                         transform=valid_transform, 
                         is_train=False, 
                         shuffle=False)


preds = model.predict(test_gen, verbose=1)
test_df["diagnosis"] = np.argmax(preds, axis=1)


submission_csv = "submission.csv"
test_df[["id_code", "diagnosis"]].to_csv(submission_csv, index=False)
print(f"Submission file saved as {submission_csv}")

