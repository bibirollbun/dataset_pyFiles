
!pip install albumentations
# !unzip skin-cancer.v6i.folder.zip -d /content/skin-cancer-dataset



# import pandas as pd
# import os

# # Step 1: Load Metadata
# # Assuming the metadata file is in the extracted directory
# metadata_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
# try:
#     metadata_df = pd.read_csv(metadata_path)
#     print("Metadata loaded successfully.")
#     display(metadata_df.head())
#     display(metadata_df.info())
# except FileNotFoundError:
#     print(f"Error: Metadata file not found at {metadata_path}")
#     # If not found, try listing files in the expected directory to help debug
#     extracted_dir = 'skin-cancer-dataset/'
#     print(f"Listing contents of {extracted_dir}:")
#     if os.path.exists(extracted_dir):
#         print(os.listdir(extracted_dir))
#     else:
#         print(f"Directory not found: {extracted_dir}")


# # Step 2: Filter Classes and Limit Samples
# # Filter for 'nevus' (0) and 'melanoma' (1)
# filtered_df = metadata_df[metadata_df['target'].isin([0, 1])].copy()

# # Limit to a maximum of 500 images per class
# limited_df = filtered_df.groupby('target').head(500).reset_index(drop=True)

# print("\nFiltered and limited data:")
# display(limited_df['target'].value_counts())


# # Step 3: Split Data
# from sklearn.model_selection import train_test_split

# # Split into training and the rest (validation + test)
# train_df, rest_df = train_test_split(limited_df, test_size=0.3, random_state=42, stratify=limited_df['target'])

# # Split the rest into validation and test sets
# val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42, stratify=rest_df['target'])

# print("\nDataset split distribution:")
# print("Train set shape:", train_df.shape)
# print("Validation set shape:", val_df.shape)
# print("Test set shape:", test_df.shape)


# # Step 4: Organize Data by Class
# import shutil

# base_img_dir = '/kaggle/input/isic-2024-challenge/train-image/image' # Assuming images are in the train folder within the extracted data

# # Create directories
# train_dir = 'split_dataset/train'
# valid_dir = 'split_dataset/valid'
# test_dir = 'split_dataset/test'

# for directory in [train_dir, valid_dir, test_dir]:
#     os.makedirs(os.path.join(directory, '0'), exist_ok=True) # Class 0
#     os.makedirs(os.path.join(directory, '1'), exist_ok=True) # Class 1

# # Function to copy images
# def copy_images(df, dest_dir, base_img_dir):
#     for index, row in df.iterrows():
#         img_name = row['isic_id'] + '.jpg' # Assuming images are .jpg
#         src_path = os.path.join(base_img_dir, str(row['target']), img_name) # Assuming images are already in class subdirectories
#         dest_path = os.path.join(dest_dir, str(row['target']), img_name)

#         # Check if source file exists before copying
#         if os.path.exists(src_path):
#             shutil.copy(src_path, dest_path)
#         else:
#             # If not found in class subdirectories, try the base image directory directly
#             src_path_flat = os.path.join(base_img_dir, img_name)
#             if os.path.exists(src_path_flat):
#                  shutil.copy(src_path_flat, dest_path)
#             else:
#                 print(f"Warning: Image not found: {src_path} or {src_path_flat}")


# print("\nCopying training images...")
# copy_images(train_df, train_dir, base_img_dir)

# print("Copying validation images...")
# copy_images(val_df, valid_dir, base_img_dir)

# print("Copying test images...")
# copy_images(test_df, test_dir, base_img_dir)

# print("Image copying complete.")


import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# Step 1: Load Metadata from ISIC 2024 Challenge
metadata_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
try:
    metadata_df = pd.read_csv(metadata_path)
    print("ISIC 2024 Metadata loaded successfully.")
    print(f"ISIC 2024 dataset shape: {metadata_df.shape}")
    display(metadata_df.head())
    display(metadata_df['target'].value_counts())
except FileNotFoundError:
    print(f"Error: Metadata file not found at {metadata_path}")
    extracted_dir = 'skin-cancer-dataset/'
    print(f"Listing contents of {extracted_dir}:")
    if os.path.exists(extracted_dir):
        print(os.listdir(extracted_dir))
    else:
        print(f"Directory not found: {extracted_dir}")

# Step 2: Load Metadata from SIIM-ISIC Melanoma Classification
siim_metadata_path = '/kaggle/input/siim-isic-melanoma-classification/train.csv'
try:
    siim_metadata_df = pd.read_csv(siim_metadata_path)
    print("\nSIIM-ISIC Metadata loaded successfully.")
    print(f"SIIM-ISIC dataset shape: {siim_metadata_df.shape}")
    display(siim_metadata_df.head())
    display(siim_metadata_df['target'].value_counts())
except FileNotFoundError:
    print(f"Error: SIIM-ISIC metadata file not found at {siim_metadata_path}")

# Step 3: Process ISIC 2024 data (same as before)
# Filter for 'nevus' (0) and 'melanoma' (1)
filtered_df_2024 = metadata_df[metadata_df['target'].isin([0, 1])].copy()
# Limit to a maximum of 500 images per class
limited_df_2024 = filtered_df_2024.groupby('target').head(500).reset_index(drop=True)
print("\nISIC 2024 - Filtered and limited data:")
display(limited_df_2024['target'].value_counts())

# Step 4: Process SIIM-ISIC data
# Filter for classes 0 and 1, and get more melanoma samples (class 1)
filtered_df_siim = siim_metadata_df[siim_metadata_df['target'].isin([0, 1])].copy()
# Get additional melanoma samples (prioritize melanoma class 1)
melanoma_siim = filtered_df_siim[filtered_df_siim['target'] == 1].head(1000)  # Get up to 1000 melanoma samples
nevus_siim = filtered_df_siim[filtered_df_siim['target'] == 0].head(200)     # Get some nevus samples too
limited_df_siim = pd.concat([melanoma_siim, nevus_siim], ignore_index=True)

print("\nSIIM-ISIC - Filtered and limited data:")
display(limited_df_siim['target'].value_counts())

# Step 5: Combine both datasets
# Add a source column to track which dataset each sample comes from
limited_df_2024['source'] = 'isic_2024'
limited_df_siim['source'] = 'siim_isic'

# Rename columns to match (assuming SIIM uses 'image_name' instead of 'isic_id')
if 'image_name' in limited_df_siim.columns:
    limited_df_siim = limited_df_siim.rename(columns={'image_name': 'isic_id'})

# Combine datasets
combined_df = pd.concat([limited_df_2024, limited_df_siim], ignore_index=True)
print("\nCombined dataset:")
print(f"Total samples: {len(combined_df)}")
display(combined_df['target'].value_counts())
display(combined_df['source'].value_counts())

# Step 6: Split Combined Data
# Split into training and the rest (validation + test)
train_df, rest_df = train_test_split(combined_df, test_size=0.3, random_state=42, stratify=combined_df['target'])
# Split the rest into validation and test sets
val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42, stratify=rest_df['target'])

print("\nCombined dataset split distribution:")
print("Train set shape:", train_df.shape)
print("Validation set shape:", val_df.shape)
print("Test set shape:", test_df.shape)
print("\nTrain set class distribution:")
display(train_df['target'].value_counts())

# Step 7: Create Directory Structure
base_img_dir_2024 = '/kaggle/input/isic-2024-challenge/train-image/image'
base_img_dir_siim = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train'

# Create directories in skin-cancer-dataset folder
extracted_dir = 'skin-cancer-dataset'
train_dir = os.path.join(extracted_dir, 'train')
valid_dir = os.path.join(extracted_dir, 'valid')
test_dir = os.path.join(extracted_dir, 'test')

for directory in [train_dir, valid_dir, test_dir]:
    os.makedirs(os.path.join(directory, '0'), exist_ok=True)  # Class 0
    os.makedirs(os.path.join(directory, '1'), exist_ok=True)  # Class 1

# Step 8: Enhanced Function to Copy Images from Both Sources
def copy_images_combined(df, dest_dir, base_img_dir_2024, base_img_dir_siim):
    copied_count = 0
    not_found_count = 0
    
    for index, row in df.iterrows():
        img_name = row['isic_id']
        target_class = str(row['target'])
        source = row['source']
        
        # Determine source directory and file extension
        if source == 'isic_2024':
            img_file = img_name + '.jpg'
            # Try different possible paths for ISIC 2024
            possible_paths = [
                os.path.join(base_img_dir_2024, target_class, img_file),
                os.path.join(base_img_dir_2024, img_file)
            ]
        else:  # siim_isic
            img_file = img_name + '.jpg'
            # SIIM-ISIC images are typically in the train folder directly
            possible_paths = [
                os.path.join(base_img_dir_siim, img_file)
            ]
        
        # Try to find and copy the image
        copied = False
        for src_path in possible_paths:
            if os.path.exists(src_path):
                dest_path = os.path.join(dest_dir, target_class, img_file)
                try:
                    shutil.copy(src_path, dest_path)
                    copied_count += 1
                    copied = True
                    break
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
        
        if not copied:
            not_found_count += 1
            print(f"Warning: Image not found: {img_name} from {source}")
    
    print(f"Successfully copied: {copied_count} images")
    print(f"Images not found: {not_found_count}")

# Step 9: Copy Images for All Splits
print("\nCopying training images...")
copy_images_combined(train_df, train_dir, base_img_dir_2024, base_img_dir_siim)

print("\nCopying validation images...")
copy_images_combined(val_df, valid_dir, base_img_dir_2024, base_img_dir_siim)

print("\nCopying test images...")
copy_images_combined(test_df, test_dir, base_img_dir_2024, base_img_dir_siim)

print("\nDataset aggregation complete!")

# Step 10: Final Summary
print("\n" + "="*50)
print("FINAL DATASET SUMMARY")
print("="*50)
print(f"Total images processed: {len(combined_df)}")
print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Test images: {len(test_df)}")
print("\nClass distribution in training set:")
for class_label in [0, 1]:
    count = len(train_df[train_df['target'] == class_label])
    class_name = 'nevus' if class_label == 0 else 'melanoma'
    print(f"Class {class_label} ({class_name}): {count} images")

print(f"\nImages stored in: {extracted_dir}/")
print("Directory structure:")
print(f"â”œâ”€â”€ train/")
print(f"â”‚   â”œâ”€â”€ 0/ (nevus)")
print(f"â”‚   â””â”€â”€ 1/ (melanoma)")
print(f"â”œâ”€â”€ valid/")
print(f"â”‚   â”œâ”€â”€ 0/ (nevus)")
print(f"â”‚   â””â”€â”€ 1/ (melanoma)")
print(f"â””â”€â”€ test/")
print(f"    â”œâ”€â”€ 0/ (nevus)")
print(f"    â””â”€â”€ 1/ (melanoma)")


train_dir = '/kaggle/working/skin-cancer-dataset/train'
valid_dir = '/kaggle/working/skin-cancer-dataset/valid'
test_dir = '/kaggle/working/skin-cancer-dataset/test'



# import os
# import shutil

# # unlabeled_dir = '/content/skin-cancer-dataset/skin-cancer.v5i.folder/unlabeled'
# dummy_dir = os.path.join(unlabeled_dir, 'dummy')

# # Create dummy folder if it doesn't exist
# os.makedirs(dummy_dir, exist_ok=True)

# # Move all images into /unlabeled/dummy/
# for filename in os.listdir(unlabeled_dir):
#     filepath = os.path.join(unlabeled_dir, filename)
#     if os.path.isfile(filepath) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         shutil.move(filepath, os.path.join(dummy_dir, filename))



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB0, EfficientNetV2B0, EfficientNetV2M
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback


from tensorflow.keras.utils import Sequence

class AlbumentationsDataGenerator(Sequence):
    """
    Custom data generator using Albumentations for augmentation.
    """
    def __init__(self, folder_path, batch_size, transforms=None, shuffle=True):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.transforms = transforms
        self.shuffle = shuffle
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(folder_path))
        self.class_indices = {name: i for i, name in enumerate(self.class_names)}


        for class_folder in self.class_names:
            class_path = os.path.join(folder_path, class_folder)
            for img_file in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_file))
                self.labels.append(self.class_indices[class_folder])


        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        batch_x = []
        batch_y = []

        for i in batch_indexes:
            img = cv2.imread(self.image_paths[i])
            if img is None:
                print(f"Error loading image: {self.image_paths[i]}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transforms:
                augmented = self.transforms(image=img)
                img = augmented['image']

            batch_x.append(img)
            batch_y.append(self.labels[i])

        batch_x = np.array(batch_x)
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=len(self.class_names))

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# class AlbumentationsDataset(Sequence):
#     def __init__(self, folder_path, batch_size=32, transform=None, shuffle=True):
#         self.folder_path = folder_path
#         self.batch_size = batch_size
#         self.transform = transform
#         self.shuffle = shuffle
#         self.image_paths = []
#         self.labels = []
#         self.class_names = sorted(os.listdir(folder_path))
#         self.images = []  # Or self.image_paths if you store paths
#         self.classes = [] # Initialize for class names


#         self.images = [] # Initialize to store images
#         for idx, class_folder in enumerate(self.class_names):
#             class_path = os.path.join(folder_path, class_folder)
#             for img_file in os.listdir(class_path):
#                 self.image_paths.append(os.path.join(class_path, img_file))
#                 self.labels.append(idx)
#                 img_path = os.path.join(class_path, img_file)
#                 self.images.append(img_path)  # If storing paths instead
#                 self.classes.append(class_folder) # Actual class name
#                 self.labels.append(idx)  # Numerical label



#         self.indexes = np.arange(len(self.image_paths))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)



#     def __len__(self):
#         return int(np.floor(len(self.image_paths) / self.batch_size))

#     def __getitem__(self, index):
#         batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

#         batch_x = []
#         batch_y = []

#         for i in batch_indexes:
#             img = cv2.imread(self.image_paths[i])

#             # Check if image was loaded successfully
#             if img is None:
#                 print(f"Error loading image: {self.image_paths[i]}")
#                 continue  # Skip this image if it couldn't be loaded

#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (224, 224))

#             if self.transform:
#                 augmented = self.transform(image=img)
#                 img = augmented['image'].numpy()
#             else:
#                 img = img / 255.0

#             batch_x.append(img)
#             batch_y.append(self.labels[i])

#         # Check if any images were loaded
#         if not batch_x:  # If batch_x is empty
#             return None, None  # Skip this batch

#         batch_x = np.array(batch_x)

#         # Check if batch_x has the expected dimensions before transposing
#         if batch_x.ndim == 4:
#             batch_x = batch_x.transpose(0, 2, 3, 1)
#         else:
#             print(f"Skipping batch with unexpected shape: {batch_x.shape}")
#             return None, None # or handle differently, e.g., return empty batch

#         batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=len(self.class_names))

#         return batch_x, batch_y


#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

# class AlbumentationsDatasetFromArrays(Sequence):
#     def __init__(self, images, labels, batch_size=32, transform=None, shuffle=True, classes=None):  # Add classes
#         """
#         Args:
#             images: numpy array of shape (N, H, W, C)
#             labels: numpy array of shape (N,) or (N, num_classes)
#             batch_size: int
#             transform: albumentations.Compose object
#             shuffle: bool
#         """
#         self.images = images
#         self.labels = labels
#         self.batch_size = batch_size
#         self.transform = transform
#         self.shuffle = shuffle
#         self.indexes = np.arange(len(self.images))
#         self.classes = classes

#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def __len__(self):
#         return int(np.floor(len(self.images) / self.batch_size))

#     def __getitem__(self, index):
#         batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

#         batch_x = []
#         batch_y = []

#         for i in batch_indexes:
#             img = self.images[i]

#             # If transform provided (Albumentations), apply it
#             if self.transform:
#                 augmented = self.transform(image=img)
#                 img = augmented['image'].numpy()
#             else:
#                 img = img / 255.0

#             batch_x.append(img)
#             batch_y.append(self.labels[i])

#         batch_x = np.array(batch_x)

#         # If labels are integers, convert to categorical
#         if len(batch_y) > 0 and len(np.array(batch_y).shape) == 1:
#             batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=np.unique(self.labels).shape[0])
#         else:
#             batch_y = np.array(batch_y)

#         return batch_x, batch_y

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indexes)



# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# # Easy augmentations (start here)
# # easy_gen = ImageDataGenerator(
# #     rescale=1./255,
# #     horizontal_flip=True,
# #     vertical_flip=True,
# #     rotation_range=45,
# #     zoom_range=0.1,
# #     width_shift_range=0.05,
# #     height_shift_range=0.05,
# #     fill_mode='nearest'
# # )

# # Hard augmentations (switch later)
# train_aug = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     rotation_range=90,
#     brightness_range=[0.8, 1.2],
#     width_shift_range=0.15,
#     height_shift_range=0.15,
#     shear_range=0.2,
#     zoom_range=0.3,
#     fill_mode='nearest'
# )

# valid_aug = ImageDataGenerator(rescale=1./255)


# train_gen = datagen.flow_from_directory(
#     '/content/skin-cancer-dataset/train',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# val_gen = datagen.flow_from_directory(
#     '/content/skin-cancer-dataset/valid',
#     target_size=(224, 224),
#     batch_size=16,
#     class_mode='categorical',
#     shuffle=False
# )

# test_gen = datagen.flow_from_directory(
#     '/content/skin-cancer-dataset/test',
#     target_size=(224, 224),
#     batch_size=16,
#     class_mode='categorical'
# )



import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

IMG_SIZE = 224

train_transforms = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),

    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.0),
        A.ElasticTransform(alpha=3),
    ], p=0.7),

    A.CLAHE(clip_limit=4.0, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.85), # Use cv2 border mode

    A.Resize(IMG_SIZE, IMG_SIZE),

    A.CoarseDropout(
        max_holes=1,
        max_height=int(IMG_SIZE * 0.3),
        max_width=int(IMG_SIZE * 0.3),
        num_holes_range=(1, 1),
        p=0.5
    ),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),

    # ToTensorV2() # ToTensorV2 is for PyTorch, remove for TensorFlow/Keras
], p=1.0)



valid_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),

    # ToTensorV2() # ToTensorV2 is for PyTorch, remove for TensorFlow/Keras
], p=1.0)

train_gen = AlbumentationsDataGenerator(
    folder_path=train_dir,
    batch_size=32,
    transforms=train_transforms,
    shuffle=True
)

val_gen = AlbumentationsDataGenerator(
    folder_path=valid_dir,
    batch_size=16,
    transforms=valid_transforms,
    shuffle=False # No shuffling for validation
)

test_gen = AlbumentationsDataGenerator(
    folder_path=test_dir,
    batch_size=16,
    transforms=valid_transforms, # Use validation transforms for testing
    shuffle=False # No shuffling for testing
)


# Step 6: Inspect Data Generators
print("\nTrain Generator:")
print(f"Number of batches: {len(train_gen)}")
print(f"Number of images: {len(train_gen.image_paths)}")
print(f"Class names: {train_gen.class_names}")
print(f"Class indices: {train_gen.class_indices}")


print("\nValidation Generator:")
print(f"Number of batches: {len(val_gen)}")
print(f"Number of images: {len(val_gen.image_paths)}")
print(f"Class names: {val_gen.class_names}")
print(f"Class indices: {val_gen.class_indices}")


print("\nTest Generator:")
print(f"Number of batches: {len(test_gen)}")
print(f"Number of images: {len(test_gen.image_paths)}")
print(f"Class names: {test_gen.class_names}")
print(f"Class indices: {test_gen.class_indices}")

# Verify class distribution in generators
from collections import Counter

train_labels = [train_gen.class_names[label] for label in train_gen.labels]
val_labels = [val_gen.class_names[label] for label in val_gen.labels]
test_labels = [test_gen.class_names[label] for label in test_gen.labels]

print("\nClass distribution in generators:")
print("Train:", Counter(train_labels))
print("Validation:", Counter(val_labels))
print("Test:", Counter(test_labels))


def load_test_dataset_no_aug(folder_path, batch_size=32):
    """Loads the test dataset without augmentations (only resizing and normalization)."""

    class SimpleDataset(Sequence):
        def __init__(self, image_paths, labels, batch_size):
            self.image_paths = image_paths
            self.labels = labels
            self.batch_size = batch_size
            self.indexes = np.arange(len(self.image_paths))

        def __len__(self):
            return int(np.floor(len(self.image_paths) / self.batch_size))

        def __getitem__(self, index):
            batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            batch_x = []
            batch_y = []

            for i in batch_indexes:
                img = cv2.imread(self.image_paths[i])
                if img is None:
                    print(f"Error loading image: {self.image_paths[i]}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0

                batch_x.append(img)
                batch_y.append(self.labels[i])

            if not batch_x:
                return None, None

            batch_x = np.array(batch_x)
            batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=3) # Assuming 3 classes

            return batch_x, batch_y

    image_paths = []
    labels = []
    class_names = sorted(os.listdir(folder_path))
    for idx, class_folder in enumerate(class_names):
        class_path = os.path.join(folder_path, class_folder)
        for img_file in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_file))
            labels.append(idx)

    return SimpleDataset(image_paths, labels, batch_size)




class SAMModel(tf.keras.Model):
    def __init__(self, base_model, rho=0.05):
        super(SAMModel, self).__init__()
        self.base_model = base_model
        self.rho = rho

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            pred = self.base_model(x, training=True)
            loss = self.compiled_loss(y, pred)
        grads = tape.gradient(loss, self.base_model.trainable_variables)

        # Compute perturbation e_w
        e_ws = [g * self.rho / (tf.norm(g) + 1e-12) for g in grads]

        # Snapshot current weights (FIX: use tf.identity instead of numpy)
        old_weights = [tf.identity(w) for w in self.base_model.trainable_variables]

        # Perturb weights
        for w, e in zip(self.base_model.trainable_variables, e_ws):
            w.assign_add(e)

        # Second forward-backward pass
        with tf.GradientTape() as tape2:
            pred2 = self.base_model(x, training=True)
            loss2 = self.compiled_loss(y, pred2)
        grads2 = tape2.gradient(loss2, self.base_model.trainable_variables)

        # Restore original weights
        for w, old_w in zip(self.base_model.trainable_variables, old_weights):
            w.assign(old_w)

        # Apply gradients from second loss
        self.optimizer.apply_gradients(zip(grads2, self.base_model.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, pred2)
        return {m.name: m.result() for m in self.metrics}



def mixup(batch_x, batch_y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = batch_x.shape[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
    mixed_y = lam * batch_y + (1 - lam) * batch_y[index]
    return mixed_x, mixed_y

class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator, alpha=0.2):
        self.generator = generator
        self.alpha = alpha

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, idx):
        x, y = self.generator[idx]
        return mixup(x, y, self.alpha)

def focal_loss(alpha, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)  # prevent log(0)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

def sharpen(p, T=0.5):
    p = np.array(p)
    p_sharpen = p ** (1 / T)
    return p_sharpen / np.sum(p_sharpen, axis=1, keepdims=True)

def mixup(x1, y1, x2, y2, alpha=0.75):
    lam = np.random.beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2
    return x_mix, y_mix

def mixmatch_generator(labeled_gen, unlabeled_gen, model, batch_size, K=2, T=0.5, alpha=0.75):
    while True:
        # Get labeled data
        x_l, y_l = next(labeled_gen)

        # Get unlabeled data and predict labels
        x_u = next(unlabeled_gen)  # Assuming unlabeled_gen yields (images, None)
        #x_u = x_u[0] #Removing the slice as the shape is already (batch_size, 224, 224, 3)
        preds = [model.predict(x_u, verbose=0) for _ in range(K)]
        avg_preds = np.mean(preds, axis=0)
        y_u = sharpen(avg_preds, T)

        # Concatenate and apply MixUp
        x_all = np.concatenate([x_l, x_u], axis=0)
        y_all = np.concatenate([y_l, y_u], axis=0)

        indices = np.random.permutation(len(x_all))
        x_all, y_all = x_all[indices], y_all[indices]

        # Fix: Ensure both batches have the same size for mixup
        x1, y1 = x_all[:batch_size], y_all[:batch_size]
        x2, y2 = x_all[batch_size:2 * batch_size], y_all[batch_size:2 * batch_size]

        #Handle case where there are not enough samples for the second batch
        num_samples = x_all.shape[0]
        if num_samples < 2 * batch_size:
            x2, y2 = x_all[:batch_size], y_all[:batch_size] #Reuse first batch if not enough samples


        x_mix, y_mix = mixup(x1, y1, x2, y2, alpha)

        yield x_mix, y_mix

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mix_cut_batch(gen):
    while True:
        x1, y1 = next(gen)
        x2, y2 = next(gen)
        if np.random.rand() < 0.5:
            lam = np.random.beta(0.4, 0.4)
            x_mix = lam * x1 + (1 - lam) * x2
            y_mix = lam * y1 + (1 - lam) * y2
        else:
            lam = np.random.beta(1.0, 1.0)
            bbx1, bby1, bbx2, bby2 = rand_bbox(x1.shape, lam)
            x1[:, bbx1:bbx2, bby1:bby2, :] = x2[:, bbx1:bbx2, bby1:bby2, :]
            lam_adj = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.shape[1] * x1.shape[2]))
            y_mix = lam_adj * y1 + (1 - lam_adj) * y2
            x_mix = x1
        yield x_mix, y_mix

def weighted_focal_loss(class_weights, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * K.log(y_pred)
        weights = class_weights * y_true
        weights = tf.reduce_sum(weights, axis=-1)
        focal = K.pow(1 - y_pred, gamma)
        focal = tf.reduce_sum(focal * y_true, axis=-1)
        return weights * focal * K.sum(ce, axis=-1)
    return loss


def build_model(num_classes=3, input_shape=(224,224,3)):
    base_model = tf.keras.applications.EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # freeze base model initially

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    return model, base_model

class CurriculumSwitchCallback(Callback):
    def __init__(self, train_gen, hard_gen, switch_epoch=15):
        self.train_gen = train_gen
        self.hard_gen = hard_gen
        self.switch_epoch = switch_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.switch_epoch:
            print(f"\nðŸ§  Switching to HARD ImageDataGenerator augmentations at epoch {epoch}")
            self.model.stop_training = True  # Hack to restart with new generator


from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers.schedules import CosineDecay


base_model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze for now

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(2, activation='softmax')(x) # Changed to 2 for 2 classes

model = Model(inputs=base_model.input, outputs=output)

y_train = train_gen.labels
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)


class_weights_tensor = tf.constant(class_weights_array, dtype=tf.float32)
# model = load_model("skin_cancer_detector.keras", compile=False)  # compile=False to recompile later with new loss
loss_fn_focal = weighted_focal_loss(class_weights_tensor)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

initial_lr = 1e-4
decay_steps = len(train_gen)
cosine_lr = CosineDecay(
    initial_learning_rate = initial_lr,
    decay_steps=decay_steps,
    alpha=1e-2
)
# model = SAMModel(model)

class_weights = dict(enumerate(class_weights_array))



# from sklearn.utils.class_weight import compute_class_weight
# import tensorflow.keras.backend as K

# y_train = train_gen.classes
# class_weights_array = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )

# class_weights_tensor = tf.constant(class_weights_array, dtype=tf.float32)
# loss_fn = weighted_focal_loss(class_weights_tensor)

# model, base_model = build_model(num_classes=3)

model.compile(optimizer=Adam(1e-4), loss=loss_fn, metrics=['accuracy'])
callbacks = [
    ModelCheckpoint("efficientnetv2m_best.h5", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3),
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
]
# callbacks = [
#     ModelCheckpoint("best_model.h5", save_best_only=True),
#     ReduceLROnPlateau(patience=3)
# ]

# base_model_name = 'efficientnetv2-b0'
# base_model = model.layers[0]  # Replace 0 with the index of the EfficientNetV2B0 layer if needed



history = model.fit(
    train_gen,  # 3k training generator
    validation_data=val_gen,
    epochs=25,
    class_weight=class_weights, # Commented out as it might not be compatible with ImageDataGenerator or SAMModel
    callbacks=callbacks
)

# Switch to hard aug
# train_gen = hard_gen.flow_from_directory(
#     '/content/skin-cancer-dataset/train',
#     target_size=(224, 224),
#     batch_size=16,
#     class_mode='categorical'
# )

# # Phase 2: Continue with harder augmentations
# model.fit(
#     train_gen,
#     steps_per_epoch=len(train_gen),
#     epochs=10,  # or more
#     validation_data=val_gen,
#     callbacks=callbacks
# )


# Step 1: Unfreeze
base_model.trainable = True

loss_fn_base = tf.keras.losses.CategoricalCrossentropy()

initial_lr_finetune = 1e-5 # Starting lower learning rate for fine-tuning
# Calculate decay steps for the fine-tuning phase
# Assuming 15 fine-tuning epochs and the same steps_per_epoch as the initial training phase
decay_steps_finetune = len(train_gen) * 20
# Step 2: Use a lower learning rate for fine-tuning
cosine_lr_finetune = CosineDecay(
    initial_learning_rate=initial_lr_finetune,
    decay_steps=decay_steps_finetune,
    alpha=1e-2 # Can adjust alpha for minimum learning rate
)

# Compile the model with the new lower learning rate schedule
opt_finetune = Adam(learning_rate=initial_lr_finetune) # Use the new schedule

model.compile(
    optimizer=opt_finetune,
    loss=loss_fn_base,  # your weighted focal loss
    metrics=['accuracy']
)

# Step 3: Fine-tune
history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weights,
    callbacks=[
        ModelCheckpoint("finetuned_with_base.keras", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    ]
)




# # Predict Unlabeled Set
# pseudo_preds = model.predict(unlabeled_dataset, verbose=1)

# # Get confidence and predicted class
# confidence_scores = np.max(pseudo_preds, axis=1)
# predicted_classes = np.argmax(pseudo_preds, axis=1)

# # Filter by confidence threshold
# threshold = 0.50
# confident_indices = np.where(confidence_scores >= threshold)[0]

# # Select pseudo-labeled data
# print(f"Number of confident samples: {len(confident_indices)}")
# pseudo_images = unlabeled_dataset.images[confident_indices]
# pseudo_labels = predicted_classes[confident_indices]

# print(f"Number of confident pseudo-labeled samples: {len(pseudo_images)}")

# # Merge images and labels
# final_train_images = np.concatenate([train_dataset.images, pseudo_images], axis=0)
# final_train_labels = np.concatenate([train_dataset.labels, pseudo_labels], axis=0)

# # Create new dataset
# merged_train_dataset = AlbumentationsDatasetFromArrays(
#     images=final_train_images,
#     labels=final_train_labels,
#     batch_size=16,
#     transform=easy_transform  # curriculum will switch this later
# )

# # Same curriculum callback but attached to merged_train_dataset
# curriculum_callback_merged = CurriculumCallback(
#     dataset_obj=merged_train_dataset,
#     new_transform=hard_transform,
#     switch_epoch=15
# )

# train_generator = custom_batch_generator(merged_train_dataset, use_mixup_cutmix=True)
# steps_per_epoch = len(merged_train_dataset)

# history = model.fit(
#     train_generator,
#     steps_per_epoch=steps_per_epoch,
#     validation_data=val_dataset,
#     epochs=25,
#     callbacks=[
#         ModelCheckpoint("best_model_after_pseudo.h5", save_best_only=True),
#         ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3),
#         curriculum_callback_merged
#     ]
# )

# # Unfreeze base again (already unfrozen but good practice)
# base_model.trainable = True

# # Even Lower LR now for delicate fine-tuning
# opt = Adam(learning_rate=1e-6)

# model.compile(
#     optimizer=opt,
#     loss=focal_loss(alpha=loss_fn, gamma=2),
#     metrics=['accuracy']
# )

# history_finetune = model.fit(
#     train_generator,
#     steps_per_epoch=steps_per_epoch,
#     validation_data=val_dataset,
#     epochs=10,
#     callbacks=[
#         ModelCheckpoint("final_best_model.h5", save_best_only=True),
#     ]
# )


import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import tensorflow.keras.backend as K

# Method 1: Improved Focal Loss Function
def focal_loss_fixed(y_true, y_pred):
    """
    Focal Loss implementation with numerical stability improvements
    """
    gamma = 2.0
    alpha = 0.25
    epsilon = K.epsilon()  # Small value to prevent log(0)
    
    # Clip predictions to prevent log(0)
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate focal loss for positive and negative classes
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    
    # Add epsilon to prevent log(0) and pow(0, gamma)
    pt_1 = K.clip(pt_1, epsilon, 1.0 - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1.0 - epsilon)
    
    # Calculate focal loss components
    loss_1 = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)
    loss_0 = -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)
    
    return K.mean(loss_1 + loss_0)

# Method 2: Alternative Focal Loss (more stable)
def focal_loss_stable(y_true, y_pred):
    """
    More numerically stable focal loss implementation
    """
    gamma = 2.0
    alpha = 0.25
    epsilon = 1e-8
    
    # Ensure y_pred is in valid range
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Convert to float32 for numerical stability
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Calculate cross entropy
    ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    
    # Calculate focal weight
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = alpha * tf.pow(1 - pt, gamma)
    
    # Apply focal weight
    focal_loss = focal_weight * ce
    
    return tf.reduce_mean(focal_loss)

# Method 3: Try loading with different approaches
def load_model_safely(model_path):
    """
    Try multiple approaches to load the model safely
    """
    print("Attempting to load model...")
    
    # Approach 1: Load with original focal loss
    try:
        print("Trying with focal_loss_fixed...")
        model = keras.models.load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': focal_loss_fixed}
        )
        print("âœ“ Successfully loaded with focal_loss_fixed")
        return model
    except Exception as e:
        print(f"âœ— Failed with focal_loss_fixed: {str(e)}")
    
    # Approach 2: Load with stable focal loss
    try:
        print("Trying with focal_loss_stable...")
        model = keras.models.load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': focal_loss_stable}
        )
        print("âœ“ Successfully loaded with focal_loss_stable")
        return model
    except Exception as e:
        print(f"âœ— Failed with focal_loss_stable: {str(e)}")
    
    # Approach 3: Load without compiling (ignore the loss function)
    try:
        print("Trying to load without compiling...")
        model = keras.models.load_model(model_path, compile=False)
        print("âœ“ Successfully loaded without compiling")
        print("Note: You'll need to compile the model before training")
        return model
    except Exception as e:
        print(f"âœ— Failed loading without compiling: {str(e)}")
    
    # Approach 4: Load architecture and weights separately
    try:
        print("Trying to load weights only...")
        # This assumes you have the model architecture defined elsewhere
        # You would need to recreate your model architecture first
        print("This approach requires recreating the model architecture")
        return None
    except Exception as e:
        print(f"âœ— Failed loading weights only: {str(e)}")
    
    print("All loading approaches failed!")
    return None

# Load the model
model_path = '/kaggle/input/keras-skin-cancer/keras/default/1/finetuned_with_base (3).keras'
model = load_model_safely(model_path)

if model is not None:
    print("\nModel loaded successfully!")
    print(f"Model summary:")
    model.summary()
    
    # If loaded without compiling, you can recompile with a working loss function
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        print("\nRecompiling model...")
        model.compile(
            optimizer='adam',
            loss=focal_loss_stable,  # Use the stable version
            metrics=['accuracy']
        )
        print("Model recompiled successfully!")
        
else:
    print("\nFailed to load model. Consider these alternatives:")
    print("1. Check if the model file exists and is not corrupted")
    print("2. Verify the focal loss function used during training")
    print("3. Try loading without custom objects and redefine the loss")
    print("4. Recreate the model architecture and load weights separately")

# Alternative if all else fails - create a dummy focal loss
def dummy_focal_loss(y_true, y_pred):
    """
    Dummy focal loss that just returns categorical crossentropy
    Use this as a last resort to load the model
    """
    return keras.losses.categorical_crossentropy(y_true, y_pred)

# Last resort loading attempt
if model is None:
    try:
        print("\nLast resort: Loading with dummy focal loss...")
        model = keras.models.load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': dummy_focal_loss}
        )
        print("âœ“ Loaded with dummy focal loss")
        print("Warning: Loss function is now categorical crossentropy, not focal loss")
    except Exception as e:
        print(f"âœ— Even dummy focal loss failed: {str(e)}")


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import tensorflow as tf

def evaluate_model_robust(model, test_gen, max_batches=None):
    """
    Robust model evaluation with comprehensive error handling
    """
    print("Starting model evaluation...")
    
    # Step 1: Reset generator and collect predictions
    y_true = []
    y_pred = []
    y_probs = []
    
    try:
        # Reset the generator to start from beginning
        test_gen.reset()
        print(f"Test generator reset. Total samples: {test_gen.samples}")
        print(f"Batch size: {test_gen.batch_size}")
        print(f"Number of classes: {test_gen.num_classes}")
        
    except Exception as e:
        print(f"Warning: Could not reset generator: {e}")
    
    # Get class names safely
    try:
        if hasattr(test_gen, 'class_indices'):
            class_names = list(test_gen.class_indices.keys())
        elif hasattr(test_gen, 'class_names'):
            class_names = test_gen.class_names
        else:
            class_names = [f'Class_{i}' for i in range(test_gen.num_classes)]
        print(f"Class names: {class_names}")
    except Exception as e:
        print(f"Warning: Could not get class names: {e}")
        class_names = ['melanoma', 'nevus']  # Default for your case
    
    # Iterate through batches
    batch_count = 0
    total_samples = 0
    
    try:
        for batch_x, batch_y in test_gen:
            print(f"Processing batch {batch_count + 1}...")
            
            # Check batch shapes
            print(f"Batch X shape: {batch_x.shape}")
            print(f"Batch Y shape: {batch_y.shape}")
            
            # Make predictions
            try:
                batch_preds = model.predict(batch_x, verbose=0)
                print(f"Predictions shape: {batch_preds.shape}")
                
                # Handle different prediction formats
                if len(batch_preds.shape) == 1:
                    # Binary classification with single output
                    batch_pred_classes = (batch_preds > 0.5).astype(int)
                    batch_preds_2d = np.column_stack([1-batch_preds, batch_preds])
                elif batch_preds.shape[1] == 1:
                    # Binary classification with single column
                    batch_pred_classes = (batch_preds.flatten() > 0.5).astype(int)
                    batch_preds_2d = np.column_stack([1-batch_preds.flatten(), batch_preds.flatten()])
                else:
                    # Multi-class classification
                    batch_pred_classes = np.argmax(batch_preds, axis=1)
                    batch_preds_2d = batch_preds
                
                # Handle true labels
                if len(batch_y.shape) == 1:
                    # Already class indices
                    batch_true_classes = batch_y.astype(int)
                elif batch_y.shape[1] == 1:
                    # Single column (binary)
                    batch_true_classes = batch_y.flatten().astype(int)
                else:
                    # One-hot encoded
                    batch_true_classes = np.argmax(batch_y, axis=1)
                
                # Store results
                y_true.extend(batch_true_classes)
                y_pred.extend(batch_pred_classes)
                y_probs.extend(batch_preds_2d)
                
                batch_count += 1
                total_samples += len(batch_x)
                
                # Debug info for first batch
                if batch_count == 1:
                    print(f"First batch - True classes: {batch_true_classes[:5]}")
                    print(f"First batch - Pred classes: {batch_pred_classes[:5]}")
                    print(f"First batch - Pred probs: {batch_preds_2d[:5]}")
                
            except Exception as e:
                print(f"Error making predictions for batch {batch_count}: {e}")
                break
            
            # Safety check - avoid infinite loops
            if max_batches and batch_count >= max_batches:
                print(f"Reached maximum batches limit: {max_batches}")
                break
                
            if total_samples >= test_gen.samples:
                print(f"Processed all samples: {total_samples}")
                break
                
    except Exception as e:
        print(f"Error during batch processing: {e}")
        if len(y_true) == 0:
            print("No predictions were made. Check your generator and model compatibility.")
            return None, None, None
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    print(f"\nEvaluation completed!")
    print(f"Total samples processed: {len(y_true)}")
    print(f"True class distribution: {collections.Counter(y_true)}")
    print(f"Predicted class distribution: {collections.Counter(y_pred)}")
    
    # Basic accuracy
    if len(y_true) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
    
    return y_true, y_pred, y_probs, class_names

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix with error handling
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
        
        return cm
        
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        return None

def print_classification_report(y_true, y_pred, class_names):
    """
    Print classification report with error handling
    """
    try:
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names,
                                     zero_division=0)
        print("\nClassification Report:")
        print(report)
        
    except Exception as e:
        print(f"Error generating classification report: {e}")
        # Fallback to basic metrics
        if len(y_true) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Basic accuracy: {accuracy:.4f}")

# Main evaluation execution
print("=" * 50)
print("STARTING MODEL EVALUATION")
print("=" * 50)

# Run evaluation
try:
    y_true, y_pred, y_probs, class_names = evaluate_model_robust(model, test_gen, max_batches=100)
    
    if y_true is not None and len(y_true) > 0:
        # Plot confusion matrix
        print("\n" + "=" * 30)
        print("CONFUSION MATRIX")
        print("=" * 30)
        cm = plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Print classification report
        print("\n" + "=" * 30)
        print("CLASSIFICATION REPORT")
        print("=" * 30)
        print_classification_report(y_true, y_pred, class_names)
        
        # Additional statistics
        print("\n" + "=" * 30)
        print("ADDITIONAL STATISTICS")
        print("=" * 30)
        
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        print(f"Unique true classes: {unique_true}")
        print(f"Unique predicted classes: {unique_pred}")
        
        # Per-class accuracy
        if cm is not None:
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            for i, acc in enumerate(per_class_acc):
                print(f"{class_names[i]} accuracy: {acc:.4f}")
                
    else:
        print("No valid predictions were obtained. Please check:")
        print("1. Model and generator compatibility")
        print("2. Generator configuration")
        print("3. Model output shape")
        
except Exception as e:
    print(f"Critical error during evaluation: {e}")
    print("\nTroubleshooting steps:")
    print("1. Check if test_gen is properly configured")
    print("2. Verify model input/output shapes")
    print("3. Try with a single batch first")
    
    # Emergency single batch test
    print("\nTrying single batch test...")
    try:
        single_batch_x, single_batch_y = next(iter(test_gen))
        print(f"Single batch X shape: {single_batch_x.shape}")
        print(f"Single batch Y shape: {single_batch_y.shape}")
        
        single_pred = model.predict(single_batch_x[:1])  # Just first sample
        print(f"Single prediction shape: {single_pred.shape}")
        print(f"Single prediction: {single_pred}")
        
    except Exception as e2:
        print(f"Single batch test also failed: {e2}")


import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import tensorflow.keras.backend as K

# Method 1: Improved Focal Loss Function
def focal_loss_fixed(y_true, y_pred):
    """
    Focal Loss implementation with numerical stability improvements
    """
    gamma = 2.0
    alpha = 0.25
    epsilon = K.epsilon()  # Small value to prevent log(0)
    
    # Clip predictions to prevent log(0)
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate focal loss for positive and negative classes
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    
    # Add epsilon to prevent log(0) and pow(0, gamma)
    pt_1 = K.clip(pt_1, epsilon, 1.0 - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1.0 - epsilon)
    
    # Calculate focal loss components
    loss_1 = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)
    loss_0 = -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)
    
    return K.mean(loss_1 + loss_0)

# Method 2: Alternative Focal Loss (more stable)
def focal_loss_stable(y_true, y_pred):
    """
    More numerically stable focal loss implementation
    """
    gamma = 2.0
    alpha = 0.25
    epsilon = 1e-8
    
    # Ensure y_pred is in valid range
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Convert to float32 for numerical stability
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Calculate cross entropy
    ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    
    # Calculate focal weight
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = alpha * tf.pow(1 - pt, gamma)
    
    # Apply focal weight
    focal_loss = focal_weight * ce
    
    return tf.reduce_mean(focal_loss)

# Method 3: Try loading with different approaches
def load_model_safely(model_path):
    """
    Try multiple approaches to load the model safely
    """
    print("Attempting to load model...")
    
    # Approach 1: Load with original focal loss
    try:
        print("Trying with focal_loss_fixed...")
        model = keras.models.load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': focal_loss_fixed}
        )
        print("âœ“ Successfully loaded with focal_loss_fixed")
        return model
    except Exception as e:
        print(f"âœ— Failed with focal_loss_fixed: {str(e)}")
    
    # Approach 2: Load with stable focal loss
    try:
        print("Trying with focal_loss_stable...")
        model = keras.models.load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': focal_loss_stable}
        )
        print("âœ“ Successfully loaded with focal_loss_stable")
        return model
    except Exception as e:
        print(f"âœ— Failed with focal_loss_stable: {str(e)}")
    
    # Approach 3: Load without compiling (ignore the loss function)
    try:
        print("Trying to load without compiling...")
        model = keras.models.load_model(model_path, compile=False)
        print("âœ“ Successfully loaded without compiling")
        print("Note: You'll need to compile the model before training")
        return model
    except Exception as e:
        print(f"âœ— Failed loading without compiling: {str(e)}")
    
    # Approach 4: Load architecture and weights separately
    try:
        print("Trying to load weights only...")
        # This assumes you have the model architecture defined elsewhere
        # You would need to recreate your model architecture first
        print("This approach requires recreating the model architecture")
        return None
    except Exception as e:
        print(f"âœ— Failed loading weights only: {str(e)}")
    
    print("All loading approaches failed!")
    return None

# Load the model
model_path_nt = "/kaggle/input/skin-cancer-notebook/finetuned_with_base.keras"
model_notebook = load_model_safely(model_path_nt)

if model is not None:
    print("\nModel loaded successfully!")
    print(f"Model summary:")
    model.summary()
    
    # If loaded without compiling, you can recompile with a working loss function
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        print("\nRecompiling model...")
        model.compile(
            optimizer='adam',
            loss=focal_loss_stable,  # Use the stable version
            metrics=['accuracy']
        )
        print("Model recompiled successfully!")
        
else:
    print("\nFailed to load model. Consider these alternatives:")
    print("1. Check if the model file exists and is not corrupted")
    print("2. Verify the focal loss function used during training")
    print("3. Try loading without custom objects and redefine the loss")
    print("4. Recreate the model architecture and load weights separately")

# Alternative if all else fails - create a dummy focal loss
def dummy_focal_loss(y_true, y_pred):
    """
    Dummy focal loss that just returns categorical crossentropy
    Use this as a last resort to load the model
    """
    return keras.losses.categorical_crossentropy(y_true, y_pred)

# Last resort loading attempt
if model is None:
    try:
        print("\nLast resort: Loading with dummy focal loss...")
        model = keras.models.load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': dummy_focal_loss}
        )
        print("âœ“ Loaded with dummy focal loss")
        print("Warning: Loss function is now categorical crossentropy, not focal loss")
    except Exception as e:
        print(f"âœ— Even dummy focal loss failed: {str(e)}")


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import tensorflow as tf

def evaluate_model_robust(model, test_gen, max_batches=None):
    """
    Robust model evaluation with comprehensive error handling
    """
    print("Starting model evaluation...")
    
    # Step 1: Reset generator and collect predictions
    y_true = []
    y_pred = []
    y_probs = []
    
    try:
        # Reset the generator to start from beginning
        test_gen.reset()
        print(f"Test generator reset. Total samples: {test_gen.samples}")
        print(f"Batch size: {test_gen.batch_size}")
        print(f"Number of classes: {test_gen.num_classes}")
        
    except Exception as e:
        print(f"Warning: Could not reset generator: {e}")
    
    # Get class names safely
    try:
        if hasattr(test_gen, 'class_indices'):
            class_names = list(test_gen.class_indices.keys())
        elif hasattr(test_gen, 'class_names'):
            class_names = test_gen.class_names
        else:
            class_names = [f'Class_{i}' for i in range(test_gen.num_classes)]
        print(f"Class names: {class_names}")
    except Exception as e:
        print(f"Warning: Could not get class names: {e}")
        class_names = ['melanoma', 'nevus']  # Default for your case
    
    # Iterate through batches
    batch_count = 0
    total_samples = 0
    
    try:
        for batch_x, batch_y in test_gen:
            print(f"Processing batch {batch_count + 1}...")
            
            # Check batch shapes
            print(f"Batch X shape: {batch_x.shape}")
            print(f"Batch Y shape: {batch_y.shape}")
            
            # Make predictions
            try:
                batch_preds = model.predict(batch_x, verbose=0)
                print(f"Predictions shape: {batch_preds.shape}")
                
                # Handle different prediction formats
                if len(batch_preds.shape) == 1:
                    # Binary classification with single output
                    batch_pred_classes = (batch_preds > 0.5).astype(int)
                    batch_preds_2d = np.column_stack([1-batch_preds, batch_preds])
                elif batch_preds.shape[1] == 1:
                    # Binary classification with single column
                    batch_pred_classes = (batch_preds.flatten() > 0.5).astype(int)
                    batch_preds_2d = np.column_stack([1-batch_preds.flatten(), batch_preds.flatten()])
                else:
                    # Multi-class classification
                    batch_pred_classes = np.argmax(batch_preds, axis=1)
                    batch_preds_2d = batch_preds
                
                # Handle true labels
                if len(batch_y.shape) == 1:
                    # Already class indices
                    batch_true_classes = batch_y.astype(int)
                elif batch_y.shape[1] == 1:
                    # Single column (binary)
                    batch_true_classes = batch_y.flatten().astype(int)
                else:
                    # One-hot encoded
                    batch_true_classes = np.argmax(batch_y, axis=1)
                
                # Store results
                y_true.extend(batch_true_classes)
                y_pred.extend(batch_pred_classes)
                y_probs.extend(batch_preds_2d)
                
                batch_count += 1
                total_samples += len(batch_x)
                
                # Debug info for first batch
                if batch_count == 1:
                    print(f"First batch - True classes: {batch_true_classes[:5]}")
                    print(f"First batch - Pred classes: {batch_pred_classes[:5]}")
                    print(f"First batch - Pred probs: {batch_preds_2d[:5]}")
                
            except Exception as e:
                print(f"Error making predictions for batch {batch_count}: {e}")
                break
            
            # Safety check - avoid infinite loops
            if max_batches and batch_count >= max_batches:
                print(f"Reached maximum batches limit: {max_batches}")
                break
                
            if total_samples >= test_gen.samples:
                print(f"Processed all samples: {total_samples}")
                break
                
    except Exception as e:
        print(f"Error during batch processing: {e}")
        if len(y_true) == 0:
            print("No predictions were made. Check your generator and model compatibility.")
            return None, None, None
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    print(f"\nEvaluation completed!")
    print(f"Total samples processed: {len(y_true)}")
    print(f"True class distribution: {collections.Counter(y_true)}")
    print(f"Predicted class distribution: {collections.Counter(y_pred)}")
    
    # Basic accuracy
    if len(y_true) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
    
    return y_true, y_pred, y_probs, class_names

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix with error handling
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
        
        return cm
        
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        return None

def print_classification_report(y_true, y_pred, class_names):
    """
    Print classification report with error handling
    """
    try:
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names,
                                     zero_division=0)
        print("\nClassification Report:")
        print(report)
        
    except Exception as e:
        print(f"Error generating classification report: {e}")
        # Fallback to basic metrics
        if len(y_true) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            print(f"Basic accuracy: {accuracy:.4f}")

# Main evaluation execution
print("=" * 50)
print("STARTING MODEL EVALUATION")
print("=" * 50)

# Run evaluation
try:
    y_true, y_pred, y_probs, class_names = evaluate_model_robust(model, test_gen, max_batches=100)
    
    if y_true is not None and len(y_true) > 0:
        # Plot confusion matrix
        print("\n" + "=" * 30)
        print("CONFUSION MATRIX")
        print("=" * 30)
        cm = plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Print classification report
        print("\n" + "=" * 30)
        print("CLASSIFICATION REPORT")
        print("=" * 30)
        print_classification_report(y_true, y_pred, class_names)
        
        # Additional statistics
        print("\n" + "=" * 30)
        print("ADDITIONAL STATISTICS")
        print("=" * 30)
        
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        print(f"Unique true classes: {unique_true}")
        print(f"Unique predicted classes: {unique_pred}")
        
        # Per-class accuracy
        if cm is not None:
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            for i, acc in enumerate(per_class_acc):
                print(f"{class_names[i]} accuracy: {acc:.4f}")
                
    else:
        print("No valid predictions were obtained. Please check:")
        print("1. Model and generator compatibility")
        print("2. Generator configuration")
        print("3. Model output shape")
        
except Exception as e:
    print(f"Critical error during evaluation: {e}")
    print("\nTroubleshooting steps:")
    print("1. Check if test_gen is properly configured")
    print("2. Verify model input/output shapes")
    print("3. Try with a single batch first")
    
    # Emergency single batch test
    print("\nTrying single batch test...")
    try:
        single_batch_x, single_batch_y = next(iter(test_gen))
        print(f"Single batch X shape: {single_batch_x.shape}")
        print(f"Single batch Y shape: {single_batch_y.shape}")
        
        single_pred = model.predict(single_batch_x[:1])  # Just first sample
        print(f"Single prediction shape: {single_pred.shape}")
        print(f"Single prediction: {single_pred}")
        
    except Exception as e2:
        print(f"Single batch test also failed: {e2}")


from sklearn.metrics import cohen_kappa_score
print("Cohenâ€™s Kappa Score:", cohen_kappa_score(y_true, y_pred))

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

n_classes = 3
y_true_onehot = label_binarize(y_true, classes=[0, 1, 2])
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROCs
plt.figure(figsize=(8,6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC = {roc_auc[i]:.2f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve per Class")
plt.legend()
plt.grid()
plt.show()




from sklearn.metrics import cohen_kappa_score
print("Cohenâ€™s Kappa Score:", cohen_kappa_score(y_true, y_pred))

import matplotlib.pyplot as plt

# Get filenames (requires shuffle=False)
filenames = test_gen.filenames
errors = np.where(y_pred != y_true)[0]

for i in errors[:5]:  # show first 5 mistakes
    img_path = test_gen.filepaths[i]
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(f"True: {y_true[i]}, Pred: {y_pred[i]}")
    plt.axis('off')
    plt.show()

import seaborn as sns

cm = confusion_matrix(y_true, y_pred, normalize='true')  # row-normalized
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Purples', xticklabels=['Nevus', 'Atypical', 'Melanoma'], yticklabels=['Nevus', 'Atypical', 'Melanoma'])
plt.title("Normalized Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from sklearn.metrics import precision_recall_curve

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_probs[:, i])
    plt.plot(recall, precision, label=f'Class {i}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Get one batch (you can loop this to check multiple)
batch_data, batch_labels = next(train_dataset)

# If it's one-hot encoded, convert to class indices
if batch_labels.ndim > 1:
    batch_classes = np.argmax(batch_labels, axis=1)
else:
    batch_classes = batch_labels  # Already class indices

# Count how many samples per class
class_counts = Counter(batch_classes)

# Print raw counts
print("Batch class distribution:", class_counts)

# Optional: visualize as a bar chart
plt.bar(class_counts.keys(), class_counts.values(), tick_label=['Nevus', 'Atypical', 'Melanoma'])
plt.title("Class distribution in one training batch")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

x_l, _ = next(train_gen)
x_u = next(unlabeled_gen)

# Visualize both
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)
axs[0].imshow(x_l[0])
axs[0].set_title("Labeled Augmented")
axs[1].imshow(x_u[0])
axs[1].set_title("Unlabeled Augmented")
plt.show()

import os

def get_class_distribution(directory):
  """Calculates the number of images in each class within a directory."""
  class_distribution = {}
  for class_name in os.listdir(directory):
    class_path = os.path.join(directory, class_name)
    if os.path.isdir(class_path):
      class_distribution[class_name] = len(os.listdir(class_path))
  return class_distribution

# Get distributions for train, validation, and test sets
train_distribution = get_class_distribution(train_dir)
val_distribution = get_class_distribution(valid_dir)
test_distribution = get_class_distribution(test_dir)

# Print the results
print("Training set class distribution:", train_distribution)
print("Validation set class distribution:", val_distribution)
print("Testing set class distribution:", test_distribution)

print(train_gen.class_indices)
print("Samples per class:", dict(zip(np.unique(train_gen.classes, return_counts=True)[0],
                                     np.unique(train_gen.classes, return_counts=True)[1])))

def get_gradcam_heatmap(model, img_array, class_index, last_conv_layer_name):
    # Create a model that maps input image to activations of the last conv layer and predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Gradient tape to get gradients of the class output wrt last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    # Get gradients of loss wrt conv layer output
    grads = tape.gradient(loss, conv_outputs)

    # Mean intensity of gradients (importance of each feature map)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply feature maps by importance weights
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize to [0, 1]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-6

    return heatmap # Return the numpy array directly


def overlay_heatmap(img_path, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)

    # Overlay heatmap on image
    overlayed = cv2.addWeighted(img, alpha, heatmap_colored, 1 - alpha, 0)

    return overlayed
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Select a test image path
img_path = "/content/skin-cancer-dataset/skin-cancer.v4i.folder/test/1/IMD023_bmp.rf.5afc020009f4b9d030c4c43db7b529c9.jpg"  # change as needed

# Load and preprocess image
img_tensor = load_and_preprocess_image(img_path)

# Predict class
preds = model.predict(img_tensor)
predicted_class = np.argmax(preds)

# Use name of last conv layer in EfficientNetV2B0
last_conv_layer_name = "top_conv"  # usually this layer in EfficientNetV2*

# Generate heatmap
heatmap = get_gradcam_heatmap(model, img_tensor, predicted_class, last_conv_layer_name)

# Overlay and plot
result = overlay_heatmap(img_path, heatmap)

plt.figure(figsize=(6, 6))
plt.imshow(result)
plt.axis('off')
plt.title(f"Predicted: {predicted_class} | Class Prob: {preds[0][predicted_class]:.2f}")
plt.show()

filenames = test_gen.filenames
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# Identify misclassified images
errors = np.where(y_pred != y_true)[0]

# Visualize Grad-CAM for misclassified images
last_conv_layer_name = "top_conv"
for i in errors[:10]:  # Visualize first 5 misclassifications
    img_path = test_gen.filepaths[i]
    img_tensor = load_and_preprocess_image(img_path)
    heatmap = get_gradcam_heatmap(model, img_tensor, y_pred[i], last_conv_layer_name)
    result = overlay_heatmap(img_path, heatmap)

    plt.figure(figsize=(6, 6))
    plt.imshow(result)
    plt.title(f"True: {y_true[i]}, Pred: {y_pred[i]}")
    plt.axis('off')
    plt.show()

model.save("skin_cancer_detector_70.keras")

model = keras.models.load_model('/content/skin_cancer_detector_70.keras')

# Assuming you have preprocessed your input data as 'input_data'
predictions = model.predict(input_data)


# # Update the model training to use the Albumentations generators
# history = model.fit(
#     train_alb_gen,
#     validation_data=val_alb_gen,
#     epochs=25, # You can adjust the number of epochs
#     class_weight=class_weights,
#     callbacks=callbacks
# )

