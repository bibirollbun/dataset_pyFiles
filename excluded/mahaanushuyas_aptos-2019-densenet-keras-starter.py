import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm

%matplotlib inline


np.random.seed(2019)
tf.set_random_seed(2019)


# train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
# test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
# print(train_df.shape)
# print(test_df.shape)
# train_df.head()


import os, random

base_dir = "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/train"

# Lists to store info
image_ids = []
diagnoses = []

# Loop through each class folder
for label in sorted(os.listdir(base_dir)):
    class_dir = os.path.join(base_dir, label)
    if not os.path.isdir(class_dir):
        continue
    
    # Get all image files in this class
    files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle for randomness
    random.shuffle(files)
    
    # Take only one tenth (at least 1 image to avoid empty)
    n = max(1, len(files) // 2)
    files = files[:n]
    
    for img_file in files:
        image_id = os.path.splitext(img_file)[0]
        image_ids.append(image_id)
        diagnoses.append(int(label))

print("Total images selected:", len(image_ids))



train_df = pd.DataFrame({
    "id_code": image_ids,
    "diagnosis": diagnoses
})

# Shuffle (optional)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Inspect
print(train_df.head())
print(train_df['diagnosis'].value_counts())


import os, random
import pandas as pd

base_dir = "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/val"

# Lists to store info
image_ids = []
diagnoses = []

# Loop through each class folder
for label in sorted(os.listdir(base_dir)):
    class_dir = os.path.join(base_dir, label)
    if not os.path.isdir(class_dir):
        continue

    # Collect all images for this class
    img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Randomly shuffle and keep 1/10
    random.shuffle(img_files)
    take_n = max(1, len(img_files) // 2)   # at least 1 image
    img_files = img_files[:take_n]

    for img_file in img_files:
        image_id = os.path.splitext(img_file)[0]
        image_ids.append(image_id)
        diagnoses.append(int(label))

# Create dataframe
val_df = pd.DataFrame({
    "id_code": image_ids,
    "diagnosis": diagnoses
})

# Shuffle final dataframe
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Inspect
print(val_df.head())
print(val_df['diagnosis'].value_counts())
print("Total validation images selected:", len(val_df))



# base_dir = "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/test"

# # Lists to store info
# image_ids = []
# diagnoses = []
# # Loop through each class folder
# for label in sorted(os.listdir(base_dir)):
#     class_dir = os.path.join(base_dir, label)
#     if not os.path.isdir(class_dir):
#         continue
#     for img_file in os.listdir(class_dir):
#         if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#             # Use the filename without extension as id_code
#             image_id = os.path.splitext(img_file)[0]
#             image_ids.append(image_id)
#             diagnoses.append(int(label))


# test_df = pd.DataFrame({
#     "id_code": image_ids,
#     "diagnosis": diagnoses
# })

# # Shuffle (optional)
# test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Inspect
# print(test_df.head())
# print(test_df['diagnosis'].value_counts())


# import os, random
# import pandas as pd

# base_dir = "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/test"

# # Lists to store info
# image_ids = []
# diagnoses = []

# # Loop through each class folder
# for label in sorted(os.listdir(base_dir)):
#     class_dir = os.path.join(base_dir, label)
#     if not os.path.isdir(class_dir):
#         continue

#     # Collect all images for this class
#     img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     # Randomly shuffle and keep 1/10
#     random.shuffle(img_files)
#     take_n = max(1, len(img_files) // 2.5)   # at least 1 image
#     img_files = img_files[:take_n]

#     for img_file in img_files:
#         image_id = os.path.splitext(img_file)[0]
#         image_ids.append(image_id)
#         diagnoses.append(int(label))

# # Create dataframe
# test_df = pd.DataFrame({
#     "id_code": image_ids,
#     "diagnosis": diagnoses
# })

# # Shuffle final dataframe
# test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Inspect
# print(test_df.head())
# print(test_df['diagnosis'].value_counts())
# print("Total test images selected:", len(test_df))



test_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

# Collect all image files
all_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle and take 10%
# random.shuffle(all_images)
take_n = max(1, len(all_images))
selected_images = all_images[:take_n]

# Prepare dataframe
test_df = pd.DataFrame({
    "id_code": [os.path.splitext(f)[0] for f in selected_images]
})

print("Total test images selected:", len(test_df))
print(test_df.head())


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/train/{image_id}/{image_path}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_df)


def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im

def preprocess_image_nature(image_path, desired_size=224, clahe_clip=2.0, clahe_grid=(8,8)):
    """
    Preprocessing inspired by Nature paper:
    - Convert to grayscale
    - Apply CLAHE
    - Resize to desired size
    """
    # Read the image
    im = cv2.imread(image_path)
    
    # Convert to grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    im_clahe = clahe.apply(im_gray)
    
    # Convert back to 3 channels (so model can take 3-channel input)
    im_final = cv2.merge([im_clahe, im_clahe, im_clahe])
    
    # Resize
    im_resized = cv2.resize(im_final, (desired_size, desired_size), interpolation=cv2.INTER_LANCZOS4)
    
    return im_resized


# N = train_df.shape[0]
# # x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)
# x_train = np.empty((N, 600, 600, 3), dtype=np.uint8)

# for i, image_id, diag in enumerate(tqdm(train_df['id_code'], train_df['diagnosis'])):
#     x_train[i, :, :, :] = preprocess_image(
#         f'/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/train/{diag}/{image_id}.jpg'
#     )

# N = train_df.shape[0]
# x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

# for i, (image_id, diag) in enumerate(tqdm(zip(train_df['id_code'], train_df['diagnosis']), total=N)):
#     x_train[i, :, :, :] = preprocess_image(
#         f'/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/train/{diag}/{image_id}.jpg'
#     )





# N = test_df.shape[0]
# x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

# # for i, image_id in enumerate(tqdm(test_df['id_code'])):
# for i, (image_id, diag) in enumerate(tqdm(zip(test_df['id_code'], test_df['diagnosis']), total=N)):
#     x_test[i, :, :, :] = preprocess_image(
#         f'/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/test/{diag}/{image_id}.jpg'
#     )


# N = len(test_df)
# x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)
# # 
# for i, image_id in enumerate(tqdm(test_df['id_code'], total=N)):
#     img_path = os.path.join(test_dir, f"{image_id}.png")  # adjust extension if needed
#     x_test[i, :, :, :] = preprocess_image(img_path)


# y_train = pd.get_dummies(train_df['diagnosis']).values

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)


# y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
# y_train_multi[:, 4] = y_train[:, 4]

# for i in range(3, -1, -1):
#     y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

# print("Original y_train:", y_train.sum(axis=0))
# print("Multilabel version:", y_train_multi.sum(axis=0))


# # x_train, x_val, y_train, y_val = train_test_split(
# #     x_train, y_train_multi, 
# #     test_size=0.15, 
# #     random_state=2019
# # )

# N_val = val_df.shape[0]
# x_val = np.empty((N_val, 224, 224, 3), dtype=np.uint8)
# y_val = val_df['diagnosis'].values  # or one-hot encode if needed

# for i in tqdm(range(N_val)):
#     image_id = val_df.loc[i, 'id_code']
#     diag = val_df.loc[i, 'diagnosis']
#     x_val[i] = preprocess_image(
#         f"/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/val/{diag}/{image_id}.jpg"
#     )



# y_val = pd.get_dummies(val_df['diagnosis']).values

# print(x_val.shape)
# print(y_val.shape)



# # Instead of loading all images into memory, create a custom generator
# class MemoryEfficientDataGenerator(Sequence):
#     def __init__(self, df, base_dir, batch_size=32, shuffle=True, preprocessing_func=preprocess_image):
#         self.df = df
#         self.base_dir = base_dir
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.preprocessing_func = preprocessing_func
#         self.indices = np.arange(len(df))
#         if self.shuffle:
#             np.random.shuffle(self.indices)
        
#     def __len__(self):
#         return int(np.ceil(len(self.df) / self.batch_size))
    
#     def __getitem__(self, idx):
#         batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
#         return self._generate_batch(batch_indices)
    
#     def _generate_batch(self, batch_indices):
#         batch_x = np.empty((len(batch_indices), 224, 224, 3), dtype=np.uint8)
#         batch_y = np.zeros((len(batch_indices), 5))
        
#         for i, idx in enumerate(batch_indices):
#             row = self.df.iloc[idx]
#             image_id = row['id_code']
#             diagnosis = row['diagnosis']
            
#             img_path = f"{self.base_dir}/{diagnosis}/{image_id}.jpg"
#             img = self.preprocessing_func(img_path)
#             # Convert PIL Image to numpy array
#             batch_x[i] = np.array(img)
            
#             # Multi-label encoding (cumulative)
#             for j in range(diagnosis + 1):
#                 batch_y[i, j] = 1
            
#         return batch_x, batch_y
    
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)

# # Memory-efficient Mixup Generator
# class MemoryEfficientMixupGenerator:
#     def __init__(self, df, base_dir, batch_size=32, alpha=0.2, shuffle=True, 
#                  datagen=None, preprocessing_func=preprocess_image):
#         self.df = df
#         self.base_dir = base_dir
#         self.batch_size = batch_size
#         self.alpha = alpha
#         self.shuffle = shuffle
#         self.datagen = datagen
#         self.preprocessing_func = preprocessing_func
#         self.sample_num = len(df)
        
#     def __call__(self):
#         while True:
#             indexes = self._get_exploration_order()
#             itr_num = int(len(indexes) // (self.batch_size * 2))
            
#             for i in range(itr_num):
#                 batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
#                 X, y = self._data_generation(batch_ids)
#                 yield X, y
    
#     def _get_exploration_order(self):
#         indexes = np.arange(self.sample_num)
#         if self.shuffle:
#             np.random.shuffle(indexes)
#         return indexes
    
#     def _load_image(self, idx):
#         row = self.df.iloc[idx]
#         image_id = row['id_code']
#         diagnosis = row['diagnosis']
#         img_path = f"{self.base_dir}/{diagnosis}/{image_id}.jpg"
#         return self.preprocessing_func(img_path)
    
#     def _data_generation(self, batch_ids):
#         l = np.random.beta(self.alpha, self.alpha, self.batch_size)
#         X_l = l.reshape(self.batch_size, 1, 1, 1)
#         y_l = l.reshape(self.batch_size, 1)
        
#         # Load images on-demand
#         X1 = np.array([self._load_image(idx) for idx in batch_ids[:self.batch_size]])
#         X2 = np.array([self._load_image(idx) for idx in batch_ids[self.batch_size:]])
        
#         X = X1 * X_l + X2 * (1 - X_l)
        
#         if self.datagen:
#             for i in range(self.batch_size):
#                 X[i] = self.datagen.random_transform(X[i])
#                 X[i] = self.datagen.standardize(X[i])
        
#         # Create multi-label encoded labels
#         y1 = np.zeros((self.batch_size, 5))
#         y2 = np.zeros((self.batch_size, 5))
        
#         for i, idx in enumerate(batch_ids[:self.batch_size]):
#             diagnosis = self.df.iloc[idx]['diagnosis']
#             for j in range(diagnosis + 1):
#                 y1[i, j] = 1
                
#         for i, idx in enumerate(batch_ids[self.batch_size:]):
#             diagnosis = self.df.iloc[idx]['diagnosis']
#             for j in range(diagnosis + 1):
#                 y2[i, j] = 1
        
#         y = y1 * y_l + y2 * (1 - y_l)
        
#         return X, y


class MemoryEfficientDataGenerator(Sequence):
    def __init__(self, df, base_dir, batch_size=32, shuffle=True, preprocessing_func=preprocess_image_nature):
        self.df = df
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocessing_func = preprocessing_func
        self.indices = np.arange(len(df))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._generate_batch(batch_indices)
    
    def _generate_batch(self, batch_indices):
        batch_x = np.empty((len(batch_indices), 224, 224, 3), dtype=np.uint8)
        batch_y = np.zeros((len(batch_indices), 5))
        
        for i, idx in enumerate(batch_indices):
            row = self.df.iloc[idx]
            image_id = row['id_code']
            diagnosis = row['diagnosis']
            
            img_path = f"{self.base_dir}/{diagnosis}/{image_id}.jpg"
            img = self.preprocessing_func(img_path)
            # Convert PIL Image to numpy array
            batch_x[i] = np.array(img)
            
            # Multi-label encoding (cumulative)
            for j in range(diagnosis + 1):
                batch_y[i, j] = 1
            
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Memory-efficient Mixup Generator
class MemoryEfficientMixupGenerator:
    def __init__(self, df, base_dir, batch_size=32, alpha=0.2, shuffle=True, 
                 datagen=None, preprocessing_func=preprocess_image):
        self.df = df
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.datagen = datagen
        self.preprocessing_func = preprocessing_func
        self.sample_num = len(df)
        
    def __call__(self):
        while True:
            indexes = self._get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))
            
            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self._data_generation(batch_ids)
                yield X, y
    
    def _get_exploration_order(self):
        indexes = np.arange(self.sample_num)
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes
    
    def _load_image(self, idx):
        row = self.df.iloc[idx]
        image_id = row['id_code']
        diagnosis = row['diagnosis']
        img_path = f"{self.base_dir}/{diagnosis}/{image_id}.jpg"
        img = self.preprocessing_func(img_path)
        # Convert PIL Image to numpy array
        return np.array(img)
    
    def _data_generation(self, batch_ids):
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)
        
        # Load images on-demand
        X1 = np.array([self._load_image(idx) for idx in batch_ids[:self.batch_size]])
        X2 = np.array([self._load_image(idx) for idx in batch_ids[self.batch_size:]])
        
        X = X1 * X_l + X2 * (1 - X_l)
        
        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])
        
        # Create multi-label encoded labels
        y1 = np.zeros((self.batch_size, 5))
        y2 = np.zeros((self.batch_size, 5))
        
        for i, idx in enumerate(batch_ids[:self.batch_size]):
            diagnosis = self.df.iloc[idx]['diagnosis']
            for j in range(diagnosis + 1):
                y1[i, j] = 1
                
        for i, idx in enumerate(batch_ids[self.batch_size:]):
            diagnosis = self.df.iloc[idx]['diagnosis']
            for j in range(diagnosis + 1):
                y2[i, j] = 1
        
        y = y1 * y_l + y2 * (1 - y_l)
        
        return X, y


class TestDataGenerator:
    def __init__(self, df, test_dir, batch_size=32):
        self.df = df
        self.test_dir = test_dir
        self.batch_size = batch_size
        
    def predict_all(self, model):
        predictions = []
        
        for i in range(0, len(self.df), self.batch_size):
            batch_df = self.df.iloc[i:i+self.batch_size]
            batch_x = np.empty((len(batch_df), 224, 224, 3), dtype=np.uint8)
            
            for j, (idx, row) in enumerate(batch_df.iterrows()):
                image_id = row['id_code']
                img_path = os.path.join(self.test_dir, f"{image_id}.png")
                batch_x[j] = np.array(preprocess_image_nature(img_path))
            
            batch_pred = model.predict(batch_x, verbose=0)
            predictions.append(batch_pred)
            
        return np.concatenate(predictions, axis=0)





# BATCH_SIZE = 32

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

# # Using original generator
# data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)
# # Using Mixup
# mixup_generator = MixupGenerator(x_train, y_train, batch_size=BATCH_SIZE, alpha=0.2, datagen=create_datagen())()


# Don't load x_train into memory! Instead use generators:
BATCH_SIZE = 32

# For regular training
train_generator = MemoryEfficientDataGenerator(
    train_df, 
    "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/train",
    batch_size=BATCH_SIZE,
    preprocessing_func=preprocess_image_nature
)

# For validation (you can keep your current val loading if it's small enough)
val_generator = MemoryEfficientDataGenerator(
    val_df,
    "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/val",
    batch_size=BATCH_SIZE,
    shuffle=False,
    preprocessing_func=preprocess_image_nature
)

# For mixup training
mixup_generator = MemoryEfficientMixupGenerator(
    train_df,
    "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2/train",
    batch_size=BATCH_SIZE,
    alpha=0.2,
    datagen=create_datagen(),
    preprocessing_func=preprocess_image_nature
)


true_labels = np.array([1, 0, 1, 1, 0, 1])
pred_labels = np.array([1, 0, 0, 0, 0, 1])


accuracy_score(true_labels, pred_labels)


cohen_kappa_score(true_labels, pred_labels)


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


class GeneratorCompatibleKappa(Callback):
    def __init__(self, val_generator, val_steps):
        super().__init__()
        self.val_generator = val_generator
        self.val_steps = val_steps
        self.val_kappas = []  # ADD THIS LINE
        
    def on_epoch_end(self, epoch, logs={}):
        # Collect all validation predictions and true labels
        y_pred_list = []
        y_true_list = []
        
        # Reset generator to start
        self.val_generator.on_epoch_end()
        
        # Get predictions for entire validation set
        for i in range(self.val_steps):
            X_val_batch, y_val_batch = self.val_generator[i]
            
            # Get predictions
            y_pred_batch = self.model.predict(X_val_batch, verbose=0)
            
            y_pred_list.append(y_pred_batch)
            y_true_list.append(y_val_batch)
        
        # Concatenate all batches
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)
        
        # Convert multi-label to single label for kappa calculation
        y_true_single = y_true.sum(axis=1) - 1  # Convert back to 0-4 scale
        y_pred_single = y_pred.sum(axis=1) - 1  # Convert back to 0-4 scale
        
        # Round predictions to nearest integer
        y_pred_single = np.round(np.clip(y_pred_single, 0, 4)).astype(int)
        
        # Calculate quadratic weighted kappa
        kappa = cohen_kappa_score(y_true_single, y_pred_single, weights='quadratic')
        
        self.val_kappas.append(kappa)  # ADD THIS LINE
        
        print(f" - val_kappa: {kappa:.4f}")
        logs['val_kappa'] = kappa


densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model


model = build_model()
model.summary()


# kappa_metrics = Metrics()

# history = model.fit_generator(
#     data_generator,
#     steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
#     epochs=50,
#     validation_data=(x_val, y_val),
#     callbacks=[kappa_metrics]
# )

# Training becomes:
# BATCH_SIZE = 32

val_steps = math.ceil(len(val_df) / BATCH_SIZE)
kappa_metrics = GeneratorCompatibleKappa(val_generator, val_steps)

history = model.fit_generator(
    mixup_generator(),  # or train_generator for regular training
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    epochs=15,
    validation_data=val_generator,  # Now this will work!
    validation_steps = val_steps,
    callbacks=[kappa_metrics]
)


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


plt.plot(kappa_metrics.val_kappas)


model.save_weights('model.h5')



# Then load and use
model.load_weights('model.h5')

# y_val_pred = model.predict(x_val)
# y_val_pred = model.predict(x_val)

# def compute_score_inv(threshold):
#     y1 = y_val_pred > threshold
#     y1 = y1.astype(int).sum(axis=1) - 1
#     y2 = y_val.sum(axis=1) - 1
#     score = cohen_kappa_score(y1, y2, weights='quadratic')
    
#     return 1 - score

# simplex = scipy.optimize.minimize(
#     compute_score_inv, 0.5, method='nelder-mead'
# )

# best_threshold = simplex['x'][0]

# Get predictions using the generator
y_val_pred_list = []
y_val_true_list = []

for i in range(len(val_generator)):
    X_batch, y_batch = val_generator[i]
    pred_batch = model.predict(X_batch, verbose=0)
    
    y_val_pred_list.append(pred_batch)
    y_val_true_list.append(y_batch)

y_val_pred = np.concatenate(y_val_pred_list, axis=0)
y_val = np.concatenate(y_val_true_list, axis=0)

# Rest of the code remains the same
def compute_score_inv(threshold):
    y1 = y_val_pred > threshold
    y1 = y1.astype(int).sum(axis=1) - 1
    y2 = y_val.sum(axis=1) - 1
    score = cohen_kappa_score(y1, y2, weights='quadratic')
    
    return 1 - score

import scipy.optimize
simplex = scipy.optimize.minimize(
    compute_score_inv, 0.5, method='nelder-mead'
)
best_threshold = simplex['x'][0]
print(f"Best threshold: {best_threshold}")


# Use test generator
test_generator = TestDataGenerator(test_df, test_dir, batch_size=32)
y_test_pred = test_generator.predict_all(model)

# Apply optimized threshold and convert to diagnosis
y_test = y_test_pred > best_threshold  # Use optimized threshold
y_test = y_test.astype(int).sum(axis=1) - 1

test_df['diagnosis'] = y_test
test_df.to_csv('submission.csv', index=False)
print(f"Submission saved! Predictions range: {y_test.min()} to {y_test.max()}")


# y_test = model.predict(x_test) > 0.5
# y_test = y_test.astype(int).sum(axis=1) - 1

# test_df['diagnosis'] = y_test
# test_df.to_csv('submission.csv',index=False)


# loss, acc = model.evaluate(x_val, y_val, verbose=0)
# print("Validation Accuracy:", acc)





