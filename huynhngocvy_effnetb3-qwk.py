!pip install albumentations


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warning, 3 = error only
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print('âœ…')


# import os
# import sys
# # Repository source: https://github.com/qubvel/efficientnet
# sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))
# from efficientnet import EfficientNetB3
# print('âœ…')


from tensorflow.keras.applications import EfficientNetB3
print('âœ…')


# Standard dependencies
import cv2
import time
import scipy as sp
import numpy as np
import random as rn
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt

# Machine Learning
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Rotate, Resize, RandomGamma

import tensorflow as tf
import keras
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import backend as K
from tensorflow.keras.activations import elu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score
print('âœ…')


# Path specifications
KAGGLE_DIR = '../input/aptos2019-blindness-detection/'
train_df_path = KAGGLE_DIR + "train.csv"
#test_df_path = KAGGLE_DIR + 'test.csv'
train_img_path = KAGGLE_DIR + "train_images/"
#test_img_path = KAGGLE_DIR + 'test_images/'
SAVED_MODEL_NAME = '/kaggle/working/efficientnetb3_best_kappa_model.keras'

save_img_path = '/kaggle/working/images'
os.makedirs(save_img_path, exist_ok=True)
aug_img_path = "/kaggle/working/aug_images/"
os.makedirs(aug_img_path, exist_ok=True)

print('âœ…')


from sklearn.model_selection import train_test_split

# === Ä�á»�c dá»¯ liá»‡u gá»‘c ===
train_df = pd.read_csv(train_df_path)

# === Chia dá»¯ liá»‡u: 80% train_val, 20% test ===
train_val_df, test_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['diagnosis'],
    random_state=42
)

# === ThÃªm Ä‘uÃ´i .png vÃ o cá»™t id_code ===
train_val_df['id_code'] = train_val_df['id_code'].astype(str) + ".png"
test_df['id_code'] = test_df['id_code'].astype(str) + ".png"
print('âœ…')


print("Image IDs and Labels (TRAIN + VAL)")
# Add extension to id_code
#train_val_df['id_code'] = train_val_df['id_code'] + ".png"
print(f"Training images: {train_val_df.shape[0]}")
display(train_val_df.head())

print("Image IDs (TEST)")
# Add extension to id_code
#test_df['id_code'] = test_df['id_code'] + ".png"
print(f"Testing Images: {test_df.shape[0]}")
display(test_df.head())
print('âœ…')


# === LÆ°u file ra /kaggle/working ===
train_val_path = '/kaggle/working/train_val.csv'
test_path = '/kaggle/working/test.csv'

train_val_df.to_csv(train_val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"âœ… Saved train_val.csv to {train_val_path}")
print(f"âœ… Saved test.csv to {test_path}")


# Specify image size
IMG_WIDTH = 224
IMG_HEIGHT = 224
CHANNELS = 3
print('âœ…')


# def get_preds_and_labels(model, generator):
#     """
#     Get predictions and labels from the generator
    
#     :param model: A Keras model object
#     :param generator: A Keras ImageDataGenerator object
    
#     :return: A tuple with two Numpy Arrays. One containing the predictions
#     and one containing the labels
#     """
#     preds = []
#     labels = []
#     for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
#         x, y = next(generator)
#         preds.append(model.predict(x))
#         labels.append(y)
#     # Flatten list of numpy arrays
#     return np.concatenate(preds).ravel(), np.concatenate(labels)#.ravel()
# print('âœ…')


def get_preds_and_labels(model, generator):
    """
    Get predictions and true labels from a data generator.
    
    :param model: A compiled Keras model
    :param generator: A data generator (e.g., ImageDataGenerator or custom generator)
    
    :return: preds (softmax output), labels (as integer class indices)
    """
    steps = int(np.ceil(generator.samples / generator.batch_size))
    preds = model.predict(generator, steps=steps, verbose=0)

    try:
        # Náº¿u cÃ³ .classes (ImageDataGenerator): tráº£ vá»� trá»±c tiáº¿p
        labels = generator.classes
    except AttributeError:
        # Vá»›i custom generator: thu tháº­p labels
        labels_list = []
        for i in range(steps):
            _, y_batch = next(generator)
            labels_list.append(y_batch)

        labels = np.concatenate(labels_list)

        # Ã‰p vá»� class index náº¿u lÃ  one-hot
        if isinstance(labels[0], (np.ndarray, list)):  # má»—i label lÃ  vector
            labels = np.argmax(labels, axis=1)

    return preds, labels

print('âœ…')


class Metrics(Callback):
    def __init__(self, val_generator):
        super().__init__()
        self.val_generator = val_generator
        self.val_kappas = []

    def on_train_begin(self, logs=None):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred, labels = get_preds_and_labels(self.model, self.val_generator)
        y_pred = y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred

        # KhÃ´ng dÃ¹ng ndim cho labels ná»¯a vÃ¬ Ä‘Ã£ xá»­ lÃ½ trong get_preds_and_labels
        # labels Ä‘Ã£ cháº¯c cháº¯n lÃ  vector class index rá»“i

        val_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic')
        self.val_kappas.append(val_kappa)
        print(f"val_kappa: {round(val_kappa, 4)}")

        if val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")

        if logs is not None:
            logs['val_kappa'] = val_kappa
        return
print('âœ…')


# Label distribution
counts = train_val_df['diagnosis'].value_counts().sort_index()
ax = counts.plot(kind="bar", figsize=(12, 5), rot=0, color='skyblue')

ax.set_title("Label Distribution", fontsize=15, fontweight='bold')
ax.set_xlabel("Class", fontsize=13)
ax.set_ylabel("Count", fontsize=13)

# Hiá»ƒn thá»‹ sá»‘ trÃªn Ä‘áº§u má»—i cá»™t
for i, value in enumerate(counts.values):
    ax.text(i, value + 10, str(value), ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_img_path,'label_distribution.png'))
plt.show()
print('âœ…')


def check_duplicates(train_val_df, test_df):
    train_set = set(train_val_df['id_code'])
    test_set = set(test_df['id_code'])

    print("ğŸ”� Duplicates between train & test:", len(train_set & test_set))

check_duplicates(train_val_df, test_df)


from collections import Counter
import matplotlib.pyplot as plt

for name, df in [("Train + Val", train_val_df), ("Test", test_df)]:
    counts = Counter(df['diagnosis'])
    labels = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color='skyblue')
    plt.title(f"Label Distribution - {name}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    # Hiá»ƒn thá»‹ giÃ¡ trá»‹ trÃªn tá»«ng cá»™t
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 10, str(int(height)), 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()



fig, ax = plt.subplots(1, 5, figsize=(15, 6))
for i in range(5):
    sample = train_val_df[train_val_df['diagnosis'] == i].sample(1)
    image_name = sample['id_code'].item()
    if not image_name.endswith('.png'):
        image_name += '.png'

    img_path = os.path.join(train_img_path, image_name)
    X = cv2.imread(img_path)

    if X is None:
        print(f"[â�Œ] Cannot read image: {img_path}")
        ax[i].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12)
        ax[i].axis('off')
        continue

    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    ax[i].set_title(f"Image: {image_name}\n Label = {sample['diagnosis'].item()}", 
                    weight='bold', fontsize=10)
    ax[i].axis('off')
    ax[i].imshow(X)

plt.tight_layout()
plt.savefig(os.path.join(save_img_path, 'sample_each_class.png'), dpi=300)
plt.show()


def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and 
    returns the a preprocessed image with 
    3 channels
    
    :param img: A NumPy Array that will be cropped
    :param tol: The tolerance used for masking
    
    :return: A NumPy array containing the cropped image
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image, sigmaX=10):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    
    :param img: A NumPy Array that will be cropped
    :param sigmaX: Value used for add GaussianBlur to the image
    
    :return: A NumPy array containing the preprocessed image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image
print('âœ…')


fig, ax = plt.subplots(1, 5, figsize=(15, 6))
for i in range(5):
    sample = train_val_df[train_val_df['diagnosis'] == i].sample(1)
    image_name = sample['id_code'].item()
    if not image_name.endswith('.png'):
        image_name += '.png'

    image_path = os.path.join(train_img_path, image_name)
    img_raw = cv2.imread(image_path)

    if img_raw is None:
        print(f"[â�Œ] Cannot read image: {image_path}")
        ax[i].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12)
        ax[i].axis('off')
        continue

    X = preprocess_image(img_raw)
    ax[i].set_title(f"Image: {image_name}\nLabel = {sample['diagnosis'].item()}", 
                    weight='bold', fontsize=10)
    ax[i].axis('off')
    ax[i].imshow(X)

plt.tight_layout()
plt.savefig(os.path.join(save_img_path, 'sample_each_class_preprocessed.png'), dpi=300)
plt.show()



from collections import Counter

# === In phÃ¢n phá»‘i vÃ  káº¿ hoáº¡ch augment ===
label_counts = Counter(train_val_df['diagnosis'])
max_count = max(label_counts.values())
augment_plan = {cls: max_count - count for cls, count in label_counts.items() if count < max_count}

print(f"[INFO] Original label counts:")
for k, v in sorted(label_counts.items()):
    print(f"    Class {k}: {v} samples")
print(f"[INFO] Auto augment plan:")
for k, v in sorted(augment_plan.items()):
    print(f"    Class {k}: need {v} augmented samples")


import os, cv2
import numpy as np
import pandas as pd
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
    RandomGamma, Resize, Compose
)
from sklearn.utils import shuffle
from collections import Counter
from tqdm import tqdm

# === Augmentation pipeline ===
augmenter = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Rotate(limit=30, p=0.5),
    RandomBrightnessContrast(p=0.5),
    RandomGamma(p=0.5),
    Resize(224, 224)
])

# === HÃ m augment má»™t class ===
def augment_class_images(class_df, label, n_to_generate, augmenter, save_dir):
    new_records = []
    samples = class_df.copy().reset_index(drop=True)
    i = 0
    while len(new_records) < n_to_generate:
        row = samples.sample(1).iloc[0]
        # Chuáº©n hÃ³a tÃªn file áº£nh
        img_filename = row['id_code'].replace('.png', '') + '.png'
        img_path = os.path.join(train_img_path, img_filename)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Cannot read image: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        aug_img = augmenter(image=img)['image']

        new_name = img_filename.replace(".png", f"_aug{i}.png")
        save_path = os.path.join(save_dir, new_name)
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        new_records.append({'id_code': new_name, 'diagnosis': label})
        i += 1
    return new_records

# === HÃ m tá»•ng augment toÃ n bá»™ táº­p train Ä‘á»ƒ cÃ¡c lá»›p báº±ng nhau ===
def augment_dataset_to_balance(df, augmenter, save_dir):
    label_counts = Counter(df['diagnosis'])
    max_count = max(label_counts.values())

    print(f"[INFO] Balancing to {max_count} samples per class")

    augmented_records = []

    for label in sorted(df['diagnosis'].unique()):
        class_df = df[df['diagnosis'] == label].reset_index(drop=True)
        
        for _, row in class_df.iterrows():
            # Chuáº©n hÃ³a tÃªn file áº£nh
            img_filename = row['id_code'].replace('.png', '') + '.png'
            img_path = os.path.join(train_img_path, img_filename)

            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Cannot read image: {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            save_path = os.path.join(save_dir, img_filename)
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            augmented_records.append({'id_code': img_filename, 'diagnosis': label})
        
        # Augment náº¿u cáº§n
        n_current = len(class_df)
        n_to_generate = max_count - n_current
        if n_to_generate > 0:
            aug_records = augment_class_images(class_df, label, n_to_generate, augmenter, save_dir)
            augmented_records.extend(aug_records)

    final_df = pd.DataFrame(augmented_records)
    return shuffle(final_df, random_state=42)

# === Thá»±c thi augment ===
augmented_df = augment_dataset_to_balance(train_val_df, augmenter, aug_img_path)
print(f"âœ… Augment complete. New balanced dataset shape: {augmented_df.shape}")


from sklearn.model_selection import train_test_split

# Chia láº¡i tá»« augmented_df
train_df, val_df = train_test_split(
    augmented_df,
    test_size=0.15,  # báº¡n muá»‘n 15% cho validation
    stratify=augmented_df['diagnosis'],
    random_state=42
)

print(f"Train set: {len(train_df)} samples")
print(f"Val set:   {len(val_df)} samples")


train_df['id_code'] = train_df['id_code'].apply(lambda x: x if x.endswith('.png') else f"{x}.png")
val_df['id_code'] = val_df['id_code'].apply(lambda x: x if x.endswith('.png') else f"{x}.png")
test_df['id_code'] = test_df['id_code'].apply(lambda x: x if x.endswith('.png') else f"{x}.png")
print('âœ…')


from pathlib import Path

missing = []
for fname in test_df['id_code']:
    if not Path(os.path.join(train_img_path, fname)).exists():
        missing.append(fname)

print(f"[â�Œ] Missing {len(missing)} images in test set.")
if missing:
    print("First few missing:", missing[:5])


train_df['diagnosis'] = train_df['diagnosis'].astype(str)
val_df['diagnosis'] = val_df['diagnosis'].astype(str)
test_df['diagnosis'] = test_df['diagnosis'].astype(str)



# We use a small batch size so we can handle large images easily
BATCH_SIZE = 32 # 32 hoáº·c 64

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1/128.)

# Use the dataframe to define train and validation generators
train_generator = train_datagen.flow_from_dataframe(train_df, 
                                                    x_col='id_code', 
                                                    y_col='diagnosis',
                                                    directory = aug_img_path,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1/128.)
    
val_generator = val_datagen.flow_from_dataframe(val_df, 
                                                  x_col='id_code', 
                                                  y_col='diagnosis',
                                                  directory = aug_img_path,
                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical')

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1/128.)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=train_img_path,  
    x_col='id_code',
    y_col='diagnosis',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
print('âœ…')


import matplotlib.pyplot as plt

# Ä�áº¿m sá»‘ lÆ°á»£ng áº£nh tá»«ng lá»›p trÆ°á»›c vÃ  sau
label_counts_before = train_val_df['diagnosis'].value_counts().sort_index()
label_counts_after = train_df['diagnosis'].value_counts().sort_index()

# TÃªn lá»›p
label_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Váº½ biá»ƒu Ä‘á»“
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# === BEFORE AUGMENTATION ===
bars0 = axes[0].bar(label_names, label_counts_before.values, color='skyblue')
axes[0].set_title("Before Augmentation", fontsize=15, fontweight='bold')
axes[0].set_xlabel("Class", fontsize=13)
axes[0].set_ylabel("Count", fontsize=13)
axes[0].tick_params(axis='x', labelsize=11)
axes[0].tick_params(axis='y', labelsize=11)
for bar in bars0:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, height + 10, f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# === AFTER AUGMENTATION ===
bars1 = axes[1].bar(label_names, label_counts_after.values, color='lightgreen')
axes[1].set_title("After Augmentation", fontsize=15, fontweight='bold')
axes[1].set_xlabel("Class", fontsize=13)
axes[1].set_ylabel("Count", fontsize=13)
axes[1].tick_params(axis='x', labelsize=11)
axes[1].tick_params(axis='y', labelsize=11)
for bar in bars1:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, height + 10, f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
# âœ… LÆ°u biá»ƒu Ä‘á»“
plt.savefig(os.path.join(save_img_path, 'augmentation_distribution.png'))
plt.show()
print('âœ…')


class GroupNormalization(Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
print('âœ…')


# Load in EfficientNetB3
effnet = EfficientNetB3(weights='imagenet',
                        include_top=False,
                        input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))
# effnet.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b3_imagenet_1000_notop.h5')
print('âœ…')


# Replace all Batch Normalization layers by Group Normalization layers
for i, layer in enumerate(effnet.layers):
    if "batch_normalization" in layer.name:
        effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
print('âœ…')


from sklearn.utils import class_weight
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
import numpy as np

# 1. TÃ­nh toÃ¡n class weights tá»« dá»¯ liá»‡u huáº¥n luyá»‡n
class_labels = train_df['diagnosis'].values
classes = np.unique(class_labels)
raw_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=class_labels
)

# 2. Boost cÃ¡c lá»›p hiáº¿m báº±ng lÅ©y thá»«a
boost_power = 1.5
alpha_boosted = raw_class_weights ** boost_power

# 3. Chuáº©n hÃ³a alpha vá»� tá»•ng = sá»‘ lá»›p
alpha_values = (alpha_boosted / np.sum(alpha_boosted)) * len(classes)
print("âœ… Boosted alpha_values:", alpha_values)

# 4. Focal Loss vá»›i label_smoothing vÃ  gamma cao
@register_keras_serializable()
def focal_loss_with_custom_alpha(alpha_list, gamma=2.0, label_smoothing=0.1):
    alpha = tf.constant(alpha_list, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        # VÃ¬ generator output lÃ  one-hot â†’ dÃ¹ng luÃ´n
        y_true_onehot = tf.cast(y_true, tf.float32)

        # Label smoothing
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true_smooth = y_true_onehot * (1.0 - label_smoothing) + (label_smoothing / num_classes)

        # Focal loss
        cross_entropy = -y_true_smooth * tf.math.log(y_pred)
        focal = tf.pow(1.0 - y_pred, gamma)

        # Alpha per class
        alpha_factor = tf.reduce_sum(alpha * y_true_onehot, axis=-1, keepdims=True)
        loss = alpha_factor * focal * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return loss_fn
    
# 5. Khá»Ÿi táº¡o loss function vá»›i tham sá»‘ rÃµ rÃ ng
gamma = 2.0
label_smoothing = 0.1
loss_fn = focal_loss_with_custom_alpha(alpha_values, gamma=gamma, label_smoothing=label_smoothing)

print(f'âœ… Custom focal loss with boosted alpha, gamma={gamma} and label_smoothing={label_smoothing} ready.')


from tensorflow.keras.optimizers import AdamW

def build_model():
    """
    A custom implementation of EfficientNetB3
    for the APTOS 2019 competition
    (Regression)
    """
    effnet = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze háº¿t, chá»‰ fine-tune 60 layer cuá»‘i
    for layer in effnet.layers:
        layer.trainable = False
    for layer in effnet.layers[-60:]:
        layer.trainable = True
        
    model = Sequential()
    model.add(effnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(218, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(5, activation='softmax'))
    # model.add(Dense(1, activation="linear"))
    
    model.compile(loss=loss_fn, #'mse',
                  optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
                  metrics=['mae'])
    model.summary()
    return model

# Initialize model
model = build_model()
print('âœ…')


from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Custom QWK metrics callback (náº¿u Ä‘Ã£ Ä‘á»‹nh nghÄ©a lá»›p Metrics nhÆ° báº¡n nÃ³i)
kappa_metrics = Metrics(val_generator=val_generator)

# Early stopping - trÃ¡nh overfitting, phá»¥c há»“i best weights
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=5,
    restore_best_weights=True
)

# Reduce learning rate khi val_loss khÃ´ng giáº£m
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    verbose=1,
    mode='min',
    min_delta=1e-4,
    min_lr=1e-6
)

# Gá»™p láº¡i callback list
callbacks = [kappa_metrics, early_stopping, reduce_lr]

print('âœ…')


import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# Begin training
history = model.fit(train_generator,
          steps_per_epoch=train_generator.samples // BATCH_SIZE,
          epochs=50,
          validation_data=val_generator,
          validation_steps = val_generator.samples // BATCH_SIZE,
          callbacks=callbacks)
print('âœ…')


FULL_MODEL_PATH = '/kaggle/working/efficientnetb3_final_model.keras'
model.save(FULL_MODEL_PATH)
print(f"âœ… Saved full model to {FULL_MODEL_PATH}")

import pickle

with open('/kaggle/working/efficientnetb3_training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("âœ… Saved training history to efficientnetb3_training_history.pkl")


# Load saved model
model = tf.keras.models.load_model(FULL_MODEL_PATH, compile=False)

with open('/kaggle/working/efficientnetb3_training_history.pkl', 'rb') as f:
    history = pickle.load(f)

IMG_SIZE = 224
CATEGORIES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath)
    if img_array is None:
        raise FileNotFoundError(f"â�Œ KhÃ´ng thá»ƒ Ä‘á»�c áº£nh: {filepath}")
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_img_path = '/kaggle/working/aug_images/001639a390f0_aug322.png'
# Regression-based prediction
prediction = model.predict([prepare(test_img_path)])
rounded_pred = int(np.rint(prediction).clip(0, 4)[0][0])
print(f"Predicted class: {CATEGORIES[rounded_pred]}")
print(f"Prediction (raw regression value): {prediction[0][0]}")


import pickle

with open('/kaggle/working/efficientnetb3_training_history.pkl', 'rb') as f:
    history = pickle.load(f)

print(type(history))
print(history)  # Xem trá»±c tiáº¿p bÃªn trong



import matplotlib.pyplot as plt
import pandas as pd

history_df = pd.DataFrame(history)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Loss (MSE)
axs[0].plot(history_df['loss'], label='Train Loss ')
axs[0].plot(history_df['val_loss'], label='Val Loss')
axs[0].set_title('Loss ', fontsize=14, weight='bold')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

# MAE
axs[1].plot(history_df['mae'], label='Train MAE')
axs[1].plot(history_df['val_mae'], label='Val MAE')
axs[1].set_title('Mean Absolute Error (MAE)', fontsize=14, weight='bold')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('MAE')
axs[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(save_img_path, 'loss_mae_plot.png'))
plt.show()



val_kappa_history = kappa_metrics.val_kappas

plt.figure(figsize=(10, 5))
plt.plot(val_kappa_history, marker='o', linestyle='-')
plt.title('Validation Quadratic Weighted Kappa (QWK)', fontsize=16, weight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('QWK Score', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_img_path, 'val_kappa_history.png'))
plt.show()



class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa score
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']
print('âœ…')


# Get predictions on train set
y_train_preds, train_labels = get_preds_and_labels(model, train_generator)

# Convert one-hot labels to class indices if needed
if isinstance(train_labels[0], (np.ndarray, list)):
    train_labels = np.argmax(train_labels, axis=1)

# Convert prediction to class index (rounding only if regression)
if y_train_preds.shape[1] == 1:
    y_train_preds = np.rint(y_train_preds).astype(np.uint8).clip(0, 4).flatten()
else:
    y_train_preds = np.argmax(y_train_preds, axis=1)

# Compute training QWK
train_score = cohen_kappa_score(train_labels, y_train_preds, weights="quadratic")

# Get predictions on validation set
y_val_preds, val_labels = get_preds_and_labels(model, val_generator)

# Convert one-hot labels to class indices if needed
if isinstance(val_labels[0], (np.ndarray, list)):
    val_labels = np.argmax(val_labels, axis=1)

# Convert prediction to class index
if y_val_preds.shape[1] == 1:
    y_val_preds = np.rint(y_val_preds).astype(np.uint8).clip(0, 4).flatten()
else:
    y_val_preds = np.argmax(y_val_preds, axis=1)

# Compute validation QWK
val_score = cohen_kappa_score(val_labels, y_val_preds, weights="quadratic")

# Print results
print(f"The Training Cohen Kappa Score is: {round(train_score, 5)}")
print(f"The Validation Cohen Kappa Score is: {round(val_score, 5)}")
print('âœ…')



from sklearn.metrics import confusion_matrix, classification_report

print("Train Confusion Matrix:")
print(confusion_matrix(train_labels, y_train_preds))
print("Train Classification Report:")
print(classification_report(train_labels, y_train_preds))

print("Val Confusion Matrix:")
print(confusion_matrix(val_labels, y_val_preds))
print("Val Classification Report:")
print(classification_report(val_labels, y_val_preds))




# Optimize on validation data and evaluate again
y_val_preds, val_labels = get_preds_and_labels(model, val_generator)
optR = OptimizedRounder()
optR.fit(y_val_preds, val_labels)
coefficients = optR.coefficients()
opt_val_predictions = optR.predict(y_val_preds, coefficients)
new_val_score = cohen_kappa_score(val_labels, opt_val_predictions, weights="quadratic")

print(f"Optimized Thresholds:\n{coefficients}\n")
print(f"The Validation Quadratic Weighted Kappa (QWK)\n\
with optimized rounding thresholds is: {round(new_val_score, 5)}\n")
print(f"This is an improvement of {round(new_val_score - val_score, 5)}\n\
over the unoptimized rounding")
print('âœ…')


img_array = preprocess_image(img_path)
predictions = model.predict(img_array)
pred_raw = predictions[0][0]
print(f"Raw output before thresholding: {pred_raw}")


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
print('âœ…')


plot_confusion_matrix(val_labels, opt_val_predictions, classes=np.array(['0', '1', '2', '3', '4']),
                      title='Confusion matrix, without normalization')

plt.savefig(os.path.join(save_img_path, 'val_conf_matrix.png'))
plt.show()
print('âœ…')


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# === Load EfficientNetB3 model Ä‘Ã£ train ===
model = keras.models.load_model(FULL_MODEL_PATH)

# Dummy call Ä‘á»ƒ model cÃ³ .input vÃ  .output
_ = model(tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3)))

# === Grad-CAM heatmap ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # TÃ¡ch EfficientNetB3 backbone
    backbone = model.get_layer('efficientnetb3')
    last_conv_layer = backbone.get_layer(last_conv_layer_name)
    feature_extractor = keras.Model(inputs=backbone.input, outputs=last_conv_layer.output)

    # Build classifier
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model = keras.Model(inputs=classifier_input, outputs=x)

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_array)
        tape.watch(conv_outputs)
        preds = classifier_model(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# === Preprocess áº£nh ===
def preprocess_image(img_path):
    img = keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === Overlay Grad-CAM ===
def save_and_display_gradcam(img_path, heatmap, true_label, save_path, alpha=0.5):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)
    
    os.makedirs(save_path, exist_ok=True)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    predictions = model.predict(preprocess_image(img_path))
    pred = int(np.rint(predictions).clip(0, 4)[0][0])

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img.astype('uint8'))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Grad-CAM')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f"Overlayed\nTrue: {true_label}, Pred: {pred}")
    plt.axis('off')

    plt.tight_layout()
    filename = os.path.basename(img_path).replace('.png', '_gradcam.png')
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.show()




# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# # Load model
# model = keras.models.load_model('/content/drive/MyDrive/NCKH/Book Chapter/model/vgg16_final.keras')

# # Build model vá»›i dummy input
# dummy_input = tf.zeros((1, 224, 224, 3))
# _ = model(dummy_input)

# # HÃ m táº¡o Grad-CAM heatmap
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     # Láº¥y pháº§n backbone VGG16
#     vgg_model = model.get_layer('vgg16')

#     # TÃ¡ch cÃ¡c layer
#     last_conv_layer = vgg_model.get_layer(last_conv_layer_name)
#     feature_extractor = keras.Model(vgg_model.input, last_conv_layer.output)

#     # Pháº§n classifier
#     classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
#     x = classifier_input
#     for layer in model.layers[1:]:  # Bá»� vgg16, láº¥y cÃ¡c lá»›p sau
#         x = layer(x)
#     classifier_model = keras.Model(classifier_input, x)

#     # Gradient Tape
#     with tf.GradientTape() as tape:
#         conv_outputs = feature_extractor(img_array)
#         tape.watch(conv_outputs)
#         preds = classifier_model(conv_outputs)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]

#     # TÃ­nh gradient
#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # Chuáº©n hÃ³a heatmap
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()

# # Load áº£nh Ä‘á»ƒ Grad-CAM
# def preprocess_image(img_path, target_size=(224,224)):
#     img = keras.utils.load_img(img_path, target_size=target_size)
#     img_array = keras.utils.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     # img_array = img_array / 255.0  # normalize náº¿u cáº§n
#     return img_array

# # Overlay heatmap lÃªn áº£nh gá»‘c
# def save_and_display_gradcam(img_path, heatmap, true_label, svg_path, alpha=0.8, class_names=['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'seg_neutrophil']):
#     # Load áº£nh gá»‘c
#     img = keras.utils.load_img(img_path)
#     img = keras.utils.img_to_array(img)

#     # Chuyá»ƒn heatmap vá»� dáº¡ng Ä‘Ãºng
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     # Resize heatmap vá»� cÃ¹ng kÃ­ch thÆ°á»›c áº£nh gá»‘c
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

#     # Chuyá»ƒn Ä‘á»•i heatmap tá»« BGR sang RGB
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#     # Overlay heatmap lÃªn áº£nh gá»‘c
#     superimposed_img = heatmap * alpha + img
#     superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

#     # Dá»± Ä‘oÃ¡n nhÃ£n cá»§a áº£nh
#     predictions = model.predict(preprocess_image(img_path))
#     predicted_class = np.argmax(predictions)
#     predicted_label = class_names[predicted_class] if class_names else f"Class {predicted_class}"

#     # Hiá»ƒn thá»‹ áº£nh gá»‘c, heatmap vÃ  áº£nh overlay
#     plt.figure(figsize=(15,5))

#     # áº¢nh gá»‘c
#     plt.subplot(1, 3, 1)
#     plt.imshow(img.astype('uint8'))
#     plt.title('Original Image')
#     plt.axis('off')

#     # Heatmap
#     plt.subplot(1, 3, 2)
#     plt.imshow(heatmap)
#     plt.title('Grad-CAM Heatmap')
#     plt.axis('off')

#     # áº¢nh overlay
#     plt.subplot(1, 3, 3)
#     plt.imshow(superimposed_img)
#     plt.title(f'Overlayed Image\nTrue: {true_label}, Pred: {predicted_label}')
#     plt.axis('off')

#     plt.tight_layout()

#     # Save the figure as SVG
#     plt.savefig(svg_path, format='svg')

#     plt.show()

# # =============== CÃ�CH Gá»ŒI ===============

# # Ä�Æ°á»�ng dáº«n áº£nh báº¡n muá»‘n test
# img_path = '/content/drive/MyDrive/NCKH/Book Chapter/Blood Cell images for Cancer detection/basophil/BA_396998.jpg'  # sá»­a Ä‘Ãºng file cá»§a báº¡n
# # img_path = '/content/drive/MyDrive/NCKH/Book Chapter/Blood Cell images for Cancer detection/seg_neutrophil/NGS_0022.jpg'
# # Tiá»�n xá»­ lÃ½ áº£nh
# img_array = preprocess_image(img_path)

# # Táº¡o heatmap
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3")

# # Overlay lÃªn áº£nh
# save_and_display_gradcam(img_path, heatmap, true_label='basophil',svg_path='gradcam_output.svg')



# === VÃ­ dá»¥ sá»­ dá»¥ng ===
img_path = '/kaggle/working/aug_images/001639a390f0.png'
img_array = preprocess_image(img_path)
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv')
save_and_display_gradcam(img_path, heatmap, true_label=2, save_path='/kaggle/working/gradcam_results')


from sklearn.metrics import classification_report

# Sau khi Ä‘Ã£ cÃ³ dá»± Ä‘oÃ¡n rá»�i ráº¡c (0â€“4) vÃ  nhÃ£n tháº­t:
print("Classification Report (Validation Set):")
print(classification_report(val_labels, opt_val_predictions, digits=4, target_names=[
    'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']))



model.predict(np.expand_dims(preprocess_image(cv2.imread(val_df.iloc[0]['id_code'])), axis=0))



import random

sample_idx = random.sample(range(len(val_df)), 5)

for i in sample_idx:
    img_path = os.path.join(aug_img_path, val_df.iloc[i]['id_code'])  # áº£nh augment
    true_label = int(val_df.iloc[i]['diagnosis'])
    pred_label = int(opt_val_predictions[i])  # Ä‘Ã£ tá»‘i Æ°u hÃ³a threshold rá»“i

    print(f"Image: {img_path} | True: {true_label} | Pred: {pred_label}")
    generate_gradcam(img_path, model, class_idx=pred_label, save_path=save_img_path)



# Dá»± Ä‘oÃ¡n trÃªn test set
y_test_preds, test_labels = get_preds_and_labels(model, test_generator)
y_test_preds = np.rint(y_test_preds).astype(np.uint8).clip(0, 4)

# QWK gá»‘c
test_score = cohen_kappa_score(test_labels, y_test_preds, weights="quadratic")
print(f"The Test Cohen Kappa Score is: {round(test_score, 5)}")

# Tá»‘i Æ°u hÃ³a threshold
optR = OptimizedRounder()
optR.fit(y_test_preds, test_labels)
coefficients = optR.coefficients()
opt_test_predictions = optR.predict(y_test_preds, coefficients)
new_test_score = cohen_kappa_score(test_labels, opt_test_predictions, weights="quadratic")

print(f"Optimized Thresholds:\n{coefficients}\n")
print(f"The Test Quadratic Weighted Kappa (QWK)\n\
with optimized rounding thresholds is: {round(new_test_score, 5)}\n")
print(f"Improvement over unoptimized rounding: {round(new_test_score - test_score, 5)}")



# Ä�Ã¡nh giÃ¡ mÃ´ hÃ¬nh trÃªn táº­p test
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Accuracy : {accuracy * 100}')


# Dá»± Ä‘oÃ¡n trÃªn táº­p test
pred = np.argmax(model.predict(X_test), axis = -1)


# Confusion Matrix
plot_confusion_matrix(test_labels, opt_test_predictions, classes=np.array(['0', '1', '2', '3', '4']),
                      title='Test Confusion Matrix (Optimized)')

plt.savefig(os.path.join(save_img_path, 'test_conf_matrix.png'))
plt.show()

# Classification Report
print("Classification Report (Test Set):")
print(classification_report(test_labels, opt_test_predictions, digits=4, target_names=[
    'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']))



#confusion matrix
cf = confusion_matrix(y_test, pred, normalize = 'true')
sns.heatmap(cf, annot. = True, cmap = 'crest');
plt.xlabel('PREDICTIONS');
plt.ylabel('ACTUAL');


# === VÃ­ dá»¥ sá»­ dá»¥ng ===
img_path = '/kaggle/working/aug_images/001639a390f0.png'
img_array = preprocess_image(img_path)
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv')
save_and_display_gradcam(img_path, heatmap, true_label=2, save_path='/kaggle/working/gradcam_results')

