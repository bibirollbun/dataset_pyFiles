# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pip install scikit-image scipy


import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.transform import resize
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from skimage.io import imread, imsave
import tensorflow as tf
import numpy as np
import os
from skimage import  io, transform, color, util
from skimage.io import imread
from scipy.ndimage import center_of_mass
from scipy.ndimage import gaussian_filter,binary_opening, binary_closing
import os

import warnings
warnings.filterwarnings('ignore')


TRAIN_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'

train_labels = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')


sample = train_labels.iloc[1]
sample


sorted(os.listdir(os.path.join(TRAIN_DIR,sample['tomo_id'])))[int(sample['Motor axis 0'])]


def read_image(sample, folder_path):
    tomo_id = sample['tomo_id']
    files = sorted(os.listdir(os.path.join(folder_path, tomo_id)))
    file_name = files[int(sample['Motor axis 0'])]
    img_path = os.path.join(folder_path, tomo_id, file_name)
    
    img = imread(img_path)

    # If grayscale (2D), convert to 3 channels by stacking
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)  # (H, W) → (H, W, 3)

    # If RGBA (4 channels), discard alpha
    elif img.shape[-1] == 4:
        img = img[..., :3]

    return img


sample = train_labels.iloc[2]


def generate_gaussian_heatmap(shape, center, radius):
    """
    Generates a 2D Gaussian heatmap.

    Parameters:
        shape  : Tuple[int, int] - Shape of the output heatmap (height, width)
        center : Tuple[int, int] - (x, y) coordinates of the center
        radius : float           - Radius or standard deviation of the Gaussian

    Returns:
        heatmap: 2D NumPy array of shape `shape`
    """
    x = np.arange(0, shape[1], 1)
    y = np.arange(0, shape[0], 1)
    xx, yy = np.meshgrid(x, y)

    x0, y0 = center

    # Gaussian formula
    heatmap = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * radius ** 2))

    # Normalize the heatmap to [0, 1]
    heatmap = heatmap / np.max(heatmap)
    return heatmap

img = read_image(sample,TRAIN_DIR)


# If image is RGB, convert it to grayscale
# if img.ndim == 3:
#     img = color.rgb2gray(img)  # Returns float64 in [0, 1]

# Optional: scale to 0–255 and convert to uint8 for consistent datatype
# img = (img * 255).astype('uint8')
heat_map = generate_gaussian_heatmap(img.shape,(int(sample['Motor axis 2']),int(sample['Motor axis 1'])),40)
img_copy = resize(
    img,
    (256,256),
    anti_aliasing=True
)
heat_map = resize(
    heat_map,
    (256,256),
    anti_aliasing=True
)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(img_copy)
plt.subplot(1,2,2)
plt.imshow(heat_map)
plt.show()


print(img.shape)


def get_mask(img,center,sigma=3.0,quantile_=0.70,length=40):

    if img.ndim == 3:
        img = np.mean(img, axis=-1).astype(np.uint8)  # Fast grayscale conversion

    # print(img.shape)
    mask = np.zeros_like(img)
    mask1 = mask.copy()
    mask2 = mask.copy()
    
    height = img.shape[0]
    width = img.shape[1]
    y = center[1]
    x = center[0]
    y1, y2 = max(0, y-length), min(height, y+length)
    x1, x2 = max(0, x-length), min(width, x+length)
    patch = img[y1:y2, x1:x2]
    patch2 = patch.copy()

    patch2 = gaussian_filter(patch2, sigma=sigma)
    
    q = np.quantile(patch,quantile_)
    segmented_patch = patch <= q
    mask[y1:y2,x1:x2] = segmented_patch
    
    q2 = np.quantile(patch2,quantile_)
    segmented_patch2 = patch2 <= q2
    mask1[y1:y2,x1:x2] = segmented_patch2
    print(segmented_patch2.shape)

    segmented_patch3 = segmented_patch2.copy()
    segmented_patch3 = binary_opening(segmented_patch3, structure=np.ones(( 3, 3)))
    segmented_patch3 = binary_closing(segmented_patch3, structure=np.ones(( 10, 10)))
    mask2[y1:y2,x1:x2] = segmented_patch3
    
    return mask, mask1, mask2

center = (int(sample['Motor axis 2']),int(sample['Motor axis 1']))

mask1, mask2, mask3 = get_mask(img,center,length=60)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('sample')
plt.subplot(2,2,2)
plt.imshow(mask1)
plt.title('mask')
plt.subplot(2,2,3)
plt.imshow(mask2)
plt.title('mask with gaussin blur')
plt.subplot(2,2,4)
plt.imshow(mask3)
plt.title('mask with morphing')


train_labels_2 = train_labels[train_labels['Number of motors'] > 0]
train_labels_2.describe()


plt.hist(train_labels_2['Number of motors'],bins=10,rwidth=0.8)
plt.ylabel("No of samples")
plt.xlabel("No of motors")
plt.show()


DATASET_DIR = '/kaggle/working/dataset'
os.makedirs(DATASET_DIR,exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR,"train"),exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR,"test"),exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR,"val"),exist_ok=True)
TRAIN_DATASET_DIR = os.path.join(DATASET_DIR,"train")
TEST_DATASET_DIR = os.path.join(DATASET_DIR,"test")
VAL_DATASET_DIR = os.path.join(DATASET_DIR,"val")


EXTRA_DATASET_DIR = '/kaggle/input/cryoet-flagellar-motors-dataset/jpgs'
extra_dataset_labels = pd.read_csv('/kaggle/input/cryoet-flagellar-motors-dataset/labels.csv')
extra_dataset_labels_new = pd.read_csv('/kaggle/input/cryoet-flagellar-motors-dataset/labels_new.csv')
display(extra_dataset_labels_new.info())
print(f'No of samples in extra dataset: {len(os.listdir(EXTRA_DATASET_DIR))}')
files_in_dir = os.listdir(EXTRA_DATASET_DIR)
missing_samples = extra_dataset_labels_new[~extra_dataset_labels_new['tomo_id'].isin(files_in_dir)]

print(f"No of samples missing in dir: {len(missing_samples)}")
display(extra_dataset_labels_new)


clean_extra_data = extra_dataset_labels_new[extra_dataset_labels_new['tomo_id'].isin(files_in_dir)]
clean_extra_data = clean_extra_data[(clean_extra_data['z']>0) & (clean_extra_data['y']>0) & (clean_extra_data['x']>0)]
display(clean_extra_data.info())
display(clean_extra_data.describe())


# def generate_mask(sample,folder_path , dataset_dir=TRAIN_DIR):
#     tomo_id = sample['tomo_id']
#     mask_folder = os.path.join(folder_path,'mask')
#     mask_file = os.path.join(mask_folder,tomo_id,f"slice_{sample['Motor axis 0']}.png")
#     if os.path.exist(mask_file):
        

# def flagellar_dataset(samples,dataset_dir):
#     for idx, sample in samples.iterrows():
#         img = read_image(sample,TRAIN_DIR)
        

def get_custom_mask(img,center,sigma=3.0,quantile_=0.70,length=40):

    if img.ndim == 3:
        img = np.mean(img, axis=-1).astype(np.uint8)  # Fast grayscale conversion

    # print(img.shape)
    mask = np.zeros_like(img)
    mask1 = mask.copy()
    mask2 = mask.copy()
    
    height = img.shape[0]
    width = img.shape[1]
    y = center[1]
    x = center[0]
    y1, y2 = max(0, y-length), min(height, y+length)
    x1, x2 = max(0, x-length), min(width, x+length)
    patch = img[y1:y2, x1:x2]
    patch2 = patch.copy()

    patch2 = gaussian_filter(patch2, sigma=sigma)
    
    q = np.quantile(patch,quantile_)
    segmented_patch = patch <= q
    mask[y1:y2,x1:x2] = segmented_patch
    
    q2 = np.quantile(patch2,quantile_)
    segmented_patch2 = patch2 <= q2
    mask1[y1:y2,x1:x2] = segmented_patch2
    # print(segmented_patch2.shape)

    segmented_patch3 = segmented_patch2.copy()
    segmented_patch3 = binary_opening(segmented_patch3, structure=np.ones(( 3, 3)))
    segmented_patch3 = binary_closing(segmented_patch3, structure=np.ones(( 10, 10)))
    mask2[y1:y2,x1:x2] = segmented_patch3
    
    return mask2

# --------------------- Gaussian Heatmap --------------------- #
def generate_gaussian_heatmap(shape, center, radius):
    x = np.arange(0, shape[1], 1)
    y = np.arange(0, shape[0], 1)
    xx, yy = np.meshgrid(x, y)
    x0, y0 = center
    heatmap = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * radius ** 2))
    heatmap = heatmap / np.max(heatmap)
    return heatmap

# --------------------- Read Image --------------------- #
def read_image(sample, folder_path):
    files = sorted(os.listdir(os.path.join(folder_path, sample['tomo_id'])))
    file_name = files[int(sample['Motor axis 0'])]
    return imread(os.path.join(folder_path, sample['tomo_id'], file_name))

# --------------------- Generate Mask & Save --------------------- #
def generate_mask(sample, dataset_dir, output_base_dir, radius=60):
    tomo_id = sample['tomo_id']
    slice_idx = int(sample['Motor axis 0'])
    center_x = float(sample['Motor axis 2'])
    center_y = float(sample['Motor axis 1'])

    # Read image
    image = read_image(sample, dataset_dir)
    image_shape = image.shape[:2]

    # # 1. Save image to output/img/{tomo_id}/slice_{idx}.png
    img_dir = os.path.join(output_base_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(output_base_dir, 'img', f"{ tomo_id}_slice_{slice_idx}.png")
    if not os.path.exists(img_path):  # avoid overwriting
        imsave(img_path, image)

    # 2. Create Gaussian heatmap
    # heatmap = generate_gaussian_heatmap(image_shape, center=(center_x, center_y), radius=radius)

    # get quantile mask
    heatmap = get_custom_mask(image,(int(center_x), int(center_y)))
    
    # # 3. Save mask to output/mask/{tomo_id}/slice_{idx}.png
    mask_dir = os.path.join(output_base_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(output_base_dir, 'mask',  f"{tomo_id}_slice_{slice_idx}.png")

    if os.path.exists(mask_path):
        # Load existing and add heatmap
        existing_mask = imread(mask_path).astype(np.float32) / 255.0
        combined_mask = np.clip(existing_mask + heatmap, 0, 1)
    else:
        combined_mask = heatmap

    imsave(mask_path, (combined_mask * 255).astype(np.uint8))

# --------------------- Process All Samples --------------------- #
def flagellar_dataset(samples, output_base_dir,dataset_dir=TRAIN_DIR):
    for idx, sample in samples.iterrows():
        generate_mask(sample, dataset_dir=dataset_dir, output_base_dir=output_base_dir)



print(f"No of samples: {len(train_labels_2['tomo_id'])}")
train_split = int(len(train_labels_2['tomo_id'])*0.70)
test_split = int(len(train_labels_2['tomo_id'])*0.15)
train_samples = train_labels_2.iloc[:train_split]
test_samples = train_labels_2.iloc[train_split:train_split+test_split]
val_samples = train_labels_2.iloc[train_split+test_split:]
print(f"No of train samples: {len(train_samples['tomo_id'])}")
print(f"No of tset samples: {len(test_samples['tomo_id'])}")
print(f"No of val samples: {len(val_samples['tomo_id'])}")
# flagellar_dataset(train_labels_2.iloc[:4],)


flagellar_dataset(train_samples,output_base_dir=TRAIN_DATASET_DIR)


flagellar_dataset(test_samples,output_base_dir=TEST_DATASET_DIR)


flagellar_dataset(val_samples,output_base_dir=VAL_DATASET_DIR)


test_dir = '/kaggle/working/dataset/test/img'
train_dir = '/kaggle/working/dataset/train/img'
val_dir = '/kaggle/working/dataset/val/img'
print(f"No of train samples:{len(os.listdir(train_dir))}")
print(f"No of test samples:{len(os.listdir(test_dir))}")
print(f"No of val samples:{len(os.listdir(val_dir))}")


train_labels_2[(train_labels_2['Number of motors']==4) & (train_labels_2['tomo_id']=='tomo_1b82d1')]


train_labels_2[(train_labels_2['Number of motors']==10)& (train_labels_2['tomo_id']=='tomo_226cd8')]


clean_extra_data = clean_extra_data.rename(columns={'z':'Motor axis 0','y':'Motor axis 1','x':'Motor axis 2'})
clean_extra_data


extra_data_train_split = int(len(clean_extra_data['tomo_id'])*0.80)
extra_data_test_split = int(len(clean_extra_data['tomo_id'])*0.10)
extra_train_data = clean_extra_data.iloc[:extra_data_train_split]
extra_test_data = clean_extra_data.iloc[extra_data_train_split:extra_data_train_split+extra_data_test_split]
extra_val_data = clean_extra_data.iloc[extra_data_train_split+extra_data_test_split:]
print(f"no of extra train samples: {len(extra_train_data['tomo_id'])}")
print(f"no of extra test samples: {len(extra_test_data['tomo_id'])}")
print(f"no of extra val samples: {len(extra_val_data['tomo_id'])}")


flagellar_dataset(extra_train_data,output_base_dir=TRAIN_DATASET_DIR,dataset_dir=EXTRA_DATASET_DIR)


flagellar_dataset(extra_test_data,output_base_dir=TEST_DATASET_DIR,dataset_dir=EXTRA_DATASET_DIR)


flagellar_dataset(extra_val_data,output_base_dir=VAL_DATASET_DIR,dataset_dir=EXTRA_DATASET_DIR)


test_dir = '/kaggle/working/dataset/test/img'
train_dir = '/kaggle/working/dataset/train/img'
val_dir = '/kaggle/working/dataset/val/img'
print(f"No of training samples: {len(os.listdir(train_dir))}")
print(f"No of test samples: {len(os.listdir(test_dir))}")
print(f"No of val samples: {len(os.listdir(val_dir))}")


IMG_SIZE = (256, 256)
MASK_SIZE = (256, 256)
def load_image_and_mask(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, MASK_SIZE)
    mask = tf.cast(mask, tf.float32) / 255.0  # normalize heatmap

    return img, mask

def create_heatmap_dataset(image_dir, mask_dir, batch_size=16, shuffle=True):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

    def gen():
        for img_path, mask_path in zip(image_paths, mask_paths):
            yield load_image_and_mask(img_path, mask_path)

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



# import tensorflow as tf
# import os

# IMG_SIZE = (256, 256)

# def load_image_and_mask(img_path, mask_path):
#     img = tf.io.read_file(img_path)
#     img = tf.image.decode_png(img, channels=3)
#     img = tf.image.resize(img, IMG_SIZE)
#     img = tf.cast(img, tf.float32) / 255.0

#     mask = tf.io.read_file(mask_path)
#     mask = tf.image.decode_png(mask, channels=1)
#     mask = tf.image.resize(mask, IMG_SIZE)
#     mask = tf.cast(mask, tf.float32) / 255.0

#     return img, mask

# def augment(img, mask):
#     if tf.random.uniform(()) > 0.5:
#         img = tf.image.flip_left_right(img)
#         mask = tf.image.flip_left_right(mask)
#     if tf.random.uniform(()) > 0.5:
#         img = tf.image.flip_up_down(img)
#         mask = tf.image.flip_up_down(mask)
#     k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
#     img = tf.image.rot90(img, k=k)
#     mask = tf.image.rot90(mask, k=k)
#     img = tf.image.random_brightness(img, max_delta=0.2)
#     img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
#     return img, mask

# def create_augmented_dataset(image_dir, mask_dir, batch_size=16, augment_factor=5, shuffle=True):
#     image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
#     mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

#     image_paths = [os.path.join(image_dir, f) for f in image_files]
#     mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

#     # Repeat each file path augment_factor times
#     image_paths = image_paths * augment_factor
#     mask_paths = mask_paths * augment_factor

#     dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
#     if shuffle:
#         dataset = dataset.shuffle(buffer_size=1024)

#     # Load and augment
#     dataset = dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

#     return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



import tensorflow as tf
import os

IMG_SIZE = (256, 256)

def load_image_and_mask(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE)
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask

def augment(img, mask):
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    # Random rotation
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    mask = tf.image.rot90(mask, k)

    # Color-only augmentations
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img, mask

def create_augmented_dataset(image_dir, mask_dir, batch_size=16, augment_multiplier=5, shuffle=True):
    """
    augment_multiplier: how many times to repeat each image with different augmentations.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

    base_dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        base_dataset = base_dataset.shuffle(buffer_size=1024)

    # Expand dataset by repeating and augmenting
    augmented_dataset = base_dataset.map(
        lambda x, y: tf.py_function(load_image_and_mask, [x, y], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Set shapes for TensorFlow graph
    augmented_dataset = augmented_dataset.map(
        lambda x, y: (tf.ensure_shape(x, (256, 256, 3)), tf.ensure_shape(y, (256, 256, 1)))
    )

    # Apply augmentation multiple times
    datasets = []
    for _ in range(augment_multiplier):
        ds = augmented_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        datasets.append(ds)

    final_dataset = tf.data.Dataset.concatenate(datasets[0], datasets[1]) if augment_multiplier > 1 else datasets[0]
    for i in range(2, augment_multiplier):
        final_dataset = tf.data.Dataset.concatenate(final_dataset, datasets[i])

    final_dataset = final_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return final_dataset



import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50
import numpy as np
inputs = Input((256, 256, 3))
# base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
# base_model.summary()


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Activation, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras import layers
def unet_model_final_with_dropout(input_shape, backbone='EfficientNetB0', dropout_rate=0.3):

    inputs = Input(input_shape)
    IMG_SIZE = input_shape[0]
    resize_layer = layers.Resizing(IMG_SIZE, IMG_SIZE)(inputs)
    # --- Encoder Path ---
    if backbone == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=resize_layer)
        skip_layers = [
            'block2a_expand_activation',
            'block3a_expand_activation',
            'block4a_expand_activation',
            'block6a_expand_activation',
            'top_activation'
        ]
        skip_connections = [base_model.get_layer(name).output for name in skip_layers]
        encoder_output = skip_connections[-1]
        skip_connections = skip_connections[:-1]
    
    elif backbone == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        skip_layers = [
            'conv2_block3_out',
            'conv3_block4_out',
            'conv4_block6_out',
            'conv5_block3_out'
        ]
        skip_connections = [base_model.get_layer(name).output for name in skip_layers]
        encoder_output = skip_connections[-1]
        skip_connections = skip_connections[:-1]
    else:
        raise ValueError("Unsupported backbone. Choose 'EfficientNetB0' or 'ResNet50'.")

    # --- Decoder Path with Dropout ---
    
    # Bottleneck upsampling
    d1 = UpSampling2D(size=(2, 2))(encoder_output)
    d1 = Conv2D(512, (5, 5), padding='same')(d1)
    d1 = Concatenate()([d1, skip_connections[-1]])
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = Conv2D(512, (5, 5), padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = Dropout(dropout_rate)(d1) 

    # Decoder 2
    d2 = UpSampling2D(size=(2, 2))(d1)
    d2 = Conv2D(256, (5, 5), padding='same')(d2)
    d2 = Concatenate()([d2, skip_connections[-2]])
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = Conv2D(256, (5, 5), padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = Dropout(dropout_rate)(d2)

    # Decoder 3
    d3 = UpSampling2D(size=(2, 2))(d2)
    d3 = Conv2D(128, (5, 5), padding='same')(d3)
    d3 = Concatenate()([d3, skip_connections[-3]])
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    d3 = Conv2D(128, (5, 5), padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    d3 = Dropout(dropout_rate)(d3) 

    # Decoder 4
    if backbone == 'EfficientNetB0':
        d4 = UpSampling2D(size=(2, 2))(d3)
        d4 = Conv2D(64, (5, 5), padding='same')(d4)
        d4 = Concatenate()([d4, skip_connections[-4]])
        d4 = BatchNormalization()(d4)
        d4 = Activation('relu')(d4)
        d4 = Conv2D(64, (5, 5), padding='same')(d4)
        d4 = BatchNormalization()(d4)
        d4 = Activation('relu')(d4)
        d4 = Dropout(dropout_rate)(d4) 
        final_conv = d4
    else: 
        d4 = UpSampling2D(size=(2, 2))(d3)
        d4 = Conv2D(64, (5, 5), padding='same')(d4)
        d4 = Concatenate()([d4, skip_connections[-4]])
        d4 = BatchNormalization()(d4)
        d4 = Activation('relu')(d4)
        d4 = Conv2D(64, (5, 5), padding='same')(d4)
        d4 = BatchNormalization()(d4)
        d4 = Activation('relu')(d4)
        d4 = Dropout(dropout_rate)(d4) 
        final_conv = d4
    
    # Final upsampling to 256x256
    final_upsample = UpSampling2D(size=(2, 2))(final_conv)
    final_upsample = Conv2D(32, (5, 5), padding='same')(final_upsample)
    final_upsample = BatchNormalization()(final_upsample)
    final_upsample = Activation('relu')(final_upsample)
    final_upsample = Dropout(dropout_rate)(final_upsample)

    # Output layer
    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(final_upsample)

    model = Model(inputs, outputs, name='unet_with_' + backbone + '_backbone')
    
    return model

# Example usage with corrected code
try:
    model = unet_model_final_with_dropout(input_shape=(256, 256, 3), backbone='EfficientNetB0', dropout_rate=0.4)
    model.summary()
except Exception as e:
    print(f"An error occurred: {e}")


train_ds = create_augmented_dataset('/kaggle/working/dataset/train/img', '/kaggle/working/dataset/train/mask',augment_multiplier=5)
test_ds = create_augmented_dataset('/kaggle/working/dataset/test/img', '/kaggle/working/dataset/test/mask')
val_ds = create_augmented_dataset('/kaggle/working/dataset/val/img', '/kaggle/working/dataset/val/mask',augment_multiplier=5)


import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Calculates the Dice Coefficient (Sørensen–Dice coefficient) as a metric.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
@register_keras_serializable()
def dice_loss(y_true, y_pred):
    """
    Calculates the Dice Loss.
    """
    return 1 - dice_coefficient(y_true, y_pred)

@register_keras_serializable()
def iou_score(y_true, y_pred, smooth=1.0):
    """
    Calculates the Intersection over Union (IoU) score as a metric.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


import tensorflow.keras.backend as K
@register_keras_serializable()
def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    TP = K.sum(y_true_f * y_pred_f)
    FP = K.sum((1 - y_true_f) * y_pred_f)
    FN = K.sum(y_true_f * (1 - y_pred_f))

    return 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# @register_keras_serializable()
# def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
#     # Assuming edema is the 3rd channel (index 2)
#     y_true_edema = y_true[:, :, :, 0]
#     y_pred_edema = y_pred[:, :, :, 0]

#     intersection = K.sum(K.abs(y_true_edema * y_pred_edema))
#     denominator = K.sum(K.square(y_true_edema)) + K.sum(K.square(y_pred_edema)) + epsilon

#     return (2. * intersection) / denominator


@register_keras_serializable()
def combined_loss_penalizing_dice_and_sensitivity(y_true, y_pred, alpha=0.3, beta=0.7, lambda_dice=1.0):
    # Global Tversky loss to focus on sensitivity
    tversky = tversky_loss(y_true, y_pred, alpha=alpha, beta=beta)

    # Dice loss for whole mask
    # dice_1 = dice_coef_edema(y_true, y_pred)
    dice_2 = dice_coef(y_true, y_pred)
    # dice_loss_1 = 1 - dice_1
    dice_loss_2 = 1 - dice_2

    # Combine
    return tversky + lambda_dice * (dice_loss_2)


# Create the model
# input_size = (256, 256, 3)
# model = unet_model_revised(input_shape=input_size, backbone='EfficientNetB0')

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=combined_loss_penalizing_dice_and_sensitivity, 
    metrics=[iou_score, dice_coefficient, 'accuracy']
)


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
]

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks
)


model.evaluate(test_ds)


best_model = tf.keras.models.load_model('/kaggle/working/best_model.h5',custom_objects={'combined_loss_penalizing_dice_and_sensitivity': combined_loss_penalizing_dice_and_sensitivity,'iou_score':iou_score,'dice_coefficient':dice_coefficient})
best_model.evaluate(test_ds)


# img = imread('/kaggle/working/dataset/test/img/tomo_adc026_slice_157.png')
# img = resize(img,IMG_SIZE)
img = io.imread('/kaggle/working/dataset/test/img/tomo_adc026_slice_157.png')
mask = io.imread('/kaggle/working/dataset/test/mask/tomo_adc026_slice_157.png')
print(f"Initial image shape: {img.shape}")

# 2. Add the 3rd channel for RGB conversion
# This step is necessary if the model expects a 3-channel image.
if img.ndim == 2:
    img = color.gray2rgb(img)
    print(f"Shape after converting to RGB: {img.shape}")

# 3. Resize the image with anti-aliasing
# The `transform.resize` function handles 3-channel images automatically.
# Use `anti_aliasing=True` for better image quality, especially when downsampling.
resized_img = resize(
    img,
    IMG_SIZE,
    anti_aliasing=True
)
resized_msk = resize(
    mask,
    IMG_SIZE,
    anti_aliasing=True
)
print(f"resized image shape: {resized_img.shape}")
plt.subplot(1,2,1)
plt.title("Original image")
plt.imshow(resized_img)
plt.subplot(1,2,2)
plt.title("Mask")
plt.imshow(resized_msk)


input_img = np.expand_dims(resized_img, axis=0)
pred_1 = model.predict(input_img)


print(pred_1[0].shape)


plt.imshow(pred_1[0])


pred_2 = best_model.predict(input_img)


plt.imshow(pred_2[0])


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_curves(history):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    metric_key = None
    for k in history.history.keys():
        if 'dice' in k.lower():
            metric_key = k
            break
        elif 'accuracy' in k.lower():
            metric_key = k
            break

    if metric_key:
        plt.subplot(1, 2, 2)
        plt.plot(history.history[metric_key], label=f'Train {metric_key}')
        plt.plot(history.history[f'val_{metric_key}'], label=f'Val {metric_key}')
        plt.title(f'Training and Validation {metric_key}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_key)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_ds, threshold=0.5):
    y_true = []
    y_pred = []

    for images, masks in test_ds:
        preds = model.predict(images, verbose=0)
        preds = (preds > threshold).astype(np.uint8)  
        y_true.append(masks.numpy().flatten())
        y_pred.append(preds.flatten())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Object'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix (Flattened Pixel-wise)")
    plt.show()


plot_training_curves(history)

# plot_confusion_matrix(model, test_ds)


# # import tensorflow as tf
# # import os

# # IMG_SIZE = (256, 256)
# # MASK_SIZE = (256, 256)

# # def load_image_and_mask(img_path, mask_path):
# #     """
# #     Loads and preprocesses a single image and mask from file paths.
# #     """
# #     img = tf.io.read_file(img_path)
# #     img = tf.image.decode_png(img, channels=3)
# #     img = tf.image.resize(img, IMG_SIZE)
# #     img = tf.cast(img, tf.float32) / 255.0

# #     mask = tf.io.read_file(mask_path)
# #     mask = tf.image.decode_png(mask, channels=1)
# #     mask = tf.image.resize(mask, MASK_SIZE)
# #     mask = tf.cast(mask, tf.float32) / 255.0

# #     return img, mask

# # def augment(img, mask):
# #     """
# #     Applies on-the-fly data augmentation to the image and mask.
# #     """
# #     # Random horizontal flip
# #     if tf.random.uniform(()) > 0.5:
# #         img = tf.image.flip_left_right(img)
# #         mask = tf.image.flip_left_right(mask)

# #     # Random vertical flip
# #     if tf.random.uniform(()) > 0.5:
# #         img = tf.image.flip_up_down(img)
# #         mask = tf.image.flip_up_down(mask)
    
# #     # Random rotation (90, 180, 270 degrees)
# #     k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
# #     img = tf.image.rot90(img, k=k)
# #     mask = tf.image.rot90(mask, k=k)

# #     # Note: Color augmentations should only be applied to the image
# #     img = tf.image.random_brightness(img, max_delta=0.1)
# #     img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

# #     return img, mask

# # def create_heatmap_dataset(image_dir, mask_dir, batch_size=16, shuffle=True):
# #     """
# #     Creates a tf.data.Dataset pipeline for images and masks with augmentation.
# #     """
# #     image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
# #     mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

# #     image_paths = [os.path.join(image_dir, f) for f in image_files]
# #     mask_paths = [os.path.join(mask_dir, f) for f in mask_files]
    
# #     # Use from_tensor_slices for efficiency
# #     dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
# #     if shuffle:
# #         dataset = dataset.shuffle(buffer_size=1024)
        
# #     # Map the loading and parsing function
# #     dataset = dataset.map(
# #         lambda x, y: tf.py_function(load_image_and_mask, [x, y], (tf.float32, tf.float32)),
# #         num_parallel_calls=tf.data.AUTOTUNE
# #     )
    
# #     # Ensure correct tensor shapes after the py_function call
# #     dataset = dataset.map(
# #         lambda x, y: (tf.ensure_shape(x, (256, 256, 3)), tf.ensure_shape(y, (256, 256, 1)))
# #     )

# #     # Apply the augmentation function
# #     dataset = dataset.map(
# #         augment, 
# #         num_parallel_calls=tf.data.AUTOTUNE
# #     )

# #     # Batch and prefetch the dataset for training
# #     return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# pred_coords = pred[0]  # Extract from batch
# print("Predicted coordinates (x, y):", pred_coords)

# plt.imshow(resized_img.astype('uint8'))  # Or .numpy() if using TensorFlow tensor
# plt.scatter(pred_coords[0], pred_coords[1], c='red', marker='x')
# plt.title("Predicted Motor Location")
# plt.show()


# def generate_3d_gaussian_heatmap(shape, center, radius):
#     """
#     Generates a 3D Gaussian heatmap.

#     Parameters:
#         shape  : Tuple[int, int, int] - Shape of the 3D volume (depth, height, width)
#         center : Tuple[int, int, int] - (z, y, x) coordinates of the center
#         radius : float                - Radius (standard deviation) of the Gaussian

#     Returns:
#         heatmap: 3D NumPy array of shape `shape`
#     """
#     z = np.arange(0, shape[0])
#     y = np.arange(0, shape[1])
#     x = np.arange(0, shape[2])
#     zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

#     z0, y0, x0 = center

#     heatmap = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2) / (2 * radius ** 2))

#     # Normalize to [0, 1]
#     heatmap /= np.max(heatmap)
#     return heatmap

# # Example usage
# volume_shape = (500,  924,956)     # (depth, height, width)
# center_point = (235,  403,137)       # Center of the blob
# radius = 10

# heatmap_3d = generate_3d_gaussian_heatmap(volume_shape, center_point, radius)

# # Visualize a few central slices
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].imshow(heatmap_3d[center_point[0], :, :], cmap='hot')  # Z slice
# axs[0].set_title("Axial Slice (Z)")
# axs[1].imshow(heatmap_3d[:, center_point[1], :], cmap='hot')  # Y slice
# axs[1].set_title("Coronal Slice (Y)")
# axs[2].imshow(heatmap_3d[:, :, center_point[2]], cmap='hot')  # X slice
# axs[2].set_title("Sagittal Slice (X)")
# plt.show()

# def pad_to_shape(volume, target_shape,value):
#     """
#     Pads a 3D volume to the target shape using constant padding (0).
#     """
#     pad_width = []
#     for i in range(3):  # For z, y, x
#         total_pad = target_shape[i] - volume.shape[i]
#         pad_before = total_pad // 2
#         pad_after = total_pad - pad_before
#         pad_width.append((pad_before, pad_after))
    
#     return np.pad(volume, pad_width, mode='constant', constant_values=value)


# def get_volume_and_mask(sample, folder_path=TRAIN_DIR, trust=120, radius=60):
#     tomo_id = sample['tomo_id']
#     files = sorted(os.listdir(os.path.join(folder_path, tomo_id)))
#     z = int(sample['Motor axis 0'])
#     y = int(sample['Motor axis 1'])
#     x = int(sample['Motor axis 2'])
#     slice_dir = os.path.join(folder_path, tomo_id)
#     volume = np.stack([
#         img_as_float(imread(os.path.join(slice_dir, f)))
#         for f in files
#     ])
#     mask = generate_3d_gaussian_heatmap(volume.shape, (z, y, x), radius)

#     # Extract patch
#     z1, z2 = max(0, z - trust), min(volume.shape[0], z + trust)
#     y1, y2 = max(0, y - trust), min(volume.shape[1], y + trust)
#     x1, x2 = max(0, x - trust), min(volume.shape[2], x + trust)

#     patch_vol = volume[z1:z2, y1:y2, x1:x2]
#     patch_mask = mask[z1:z2, y1:y2, x1:x2]

#     # Pad if necessary
#     target_shape = (2 * trust, 2 * trust, 2 * trust)
#     patch_vol = pad_to_shape(patch_vol, target_shape,1)
#     patch_mask = pad_to_shape(patch_mask, target_shape,0)

#     return patch_vol.astype(np.float32), patch_mask.astype(np.float32)


# vol, mask = get_volume_and_mask(sample,TRAIN_DIR)
# print(f"vol shape: {vol.shape}, mask shape: {mask.shape}")

# import gc
# # import segmentation_models_3D as sm
# import keras
# import keras.backend as K
# import tensorflow as tf

# CHANNELS = 3



# def datagenerator(samples,batch_size=8,DATASET_DIR=TRAIN_DIR,TRUST=120,REDIUS=40):
#     while True:
#         x_batch = []
#         y_batch = []

#         # Collect one full batch
#         for _ in range(batch_size):
#             sample = samples.sample(n=1).iloc[0]
#             volume, mask = get_volume_and_mask(sample)

#             # Add channel dimensions: volume → (D, H, W, 3), mask → (D, H, W, 1)
#             volume = np.repeat(volume[..., np.newaxis], CHANNELS, axis=-1)
#             mask = mask[..., np.newaxis]

            
#             x_batch.append(volume)
#             y_batch.append(mask)

#         # Stack into batch: (B, D, H, W, C)
#         x_batch = np.stack(x_batch, axis=0)
#         y_batch = np.stack(y_batch, axis=0)

#         yield x_batch, y_batch
#         del volume, mask, x_batch, y_batch
#         gc.collect()

# train_labels_2 = train_labels[train_labels['Number of motors']>0]
# train_labels_2.describe()

# print(f"Total no of samples: {len(train_labels_2['tomo_id'])}")

# train_split = int(len(train_labels_2['tomo_id'])*0.70)
# test_split = int(len(train_labels_2)*0.15)
# val_split = int(len(train_labels_2)*0.15)
# train_samples = train_labels_2.iloc[:train_split]
# test_samples = train_labels_2.iloc[train_split:train_split+test_split]
# val_samples = train_labels_2.iloc[train_split+test_split:]
# print(f"no of train samples: {len(train_samples['tomo_id'])}")
# print(f"no of test samples: {len(test_samples['tomo_id'])}")
# print(f"no of val samples: {len(val_samples['tomo_id'])}")

# train_gen = datagenerator(train_samples)
# test_gen = datagenerator(test_samples)
# val_gen = datagenerator(val_samples)

# x_batch, y_batch = next(train_gen)
# print(f"x batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")

# from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Dropout, concatenate
# from tensorflow.keras.models import Model
# import tensorflow as tf

# def build_unet(input_shape=(240, 240, 240, 3),num_classes=1, ker_init='he_normal', dropout=0.3):
#     inputs = Input(input_shape)

#     # Encoder
#     conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(inputs)
#     conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv1)
#     pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

#     conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool1)
#     conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv2)
#     pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

#     conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool2)
#     conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv3)
#     pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

#     conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool3)
#     conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv4)
#     pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

#     # Bottleneck
#     conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool4)
#     conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv5)
#     drop5 = Dropout(dropout)(conv5)

#     # Decoder
#     up6 = UpSampling3D(size=(2, 2, 2))(drop5)
#     up6 = Conv3D(256, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(up6)
#     merge6 = concatenate([conv4, up6])
#     conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge6)
#     conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv6)

#     up7 = UpSampling3D(size=(2, 2, 2))(conv6)
#     up7 = Conv3D(128, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(up7)
#     merge7 = concatenate([conv3, up7])
#     conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge7)
#     conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv7)

#     up8 = UpSampling3D(size=(2, 2, 2))(conv7)
#     up8 = Conv3D(64, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(up8)
#     merge8 = concatenate([conv2, up8])
#     conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge8)
#     conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv8)

#     up9 = UpSampling3D(size=(2, 2, 2))(conv8)
#     up9 = Conv3D(32, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(up9)
#     merge9 = concatenate([conv1, up9])
#     conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge9)
#     conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv9)

#     # Output layer
#     conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

#     model = Model(inputs=inputs, outputs=conv10)
#     return model

# from keras.saving import register_keras_serializable
# @register_keras_serializable()
# def dice_coef(y_true, y_pred, smooth=1.0):
#     class_num = 1
#     for i in range(class_num):
#         y_true_f = K.flatten(y_true[:,:,:,i])
#         y_pred_f = K.flatten(y_pred[:,:,:,i])
#         intersection = K.sum(y_true_f * y_pred_f)
#         loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
#    #     K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
#         if i == 0:
#             total_loss = loss
#         else:
#             total_loss = total_loss + loss
#     total_loss = total_loss / class_num
# #    K.print_tensor(total_loss, message=' total dice coef: ')
#     return total_loss


 
# # define per class evaluation of dice coef
# # inspired by https://github.com/keras-team/keras/issues/9395
# @register_keras_serializable()
# def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
#     intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
#     return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)
# @register_keras_serializable()
# def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
#     intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
#     return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)
# @register_keras_serializable()
# def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
#     intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
#     return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)



# # Computing Precision 
# @register_keras_serializable()
# def precision(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision

    
# # Computing Sensitivity   
# @register_keras_serializable()
# def sensitivity(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     return true_positives / (possible_positives + K.epsilon())


# # Computing Specificity
# @register_keras_serializable()
# def specificity(y_true, y_pred):
#     true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
#     possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
#     return true_negatives / (possible_negatives + K.epsilon())


# @register_keras_serializable()
# def iou_3d(y_true, y_pred, threshold=0.5, smooth=1e-6):
#     """
#     Calculates 3D IoU for batches of volumetric predictions.
    
#     Args:
#         y_true: Ground truth tensor of shape (B, D, H, W, 1)
#         y_pred: Predicted tensor of shape (B, D, H, W, 1)
#         threshold: Threshold to binarize predictions
#         smooth: Smoothing factor to avoid division by zero

#     Returns:
#         IoU score (scalar tensor)
#     """
#     # Binarize prediction
#     y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
#     y_true_bin = tf.cast(y_true > threshold, tf.float32)

#     # Flatten
#     y_pred_f = tf.reshape(y_pred_bin, [tf.shape(y_pred_bin)[0], -1])
#     y_true_f = tf.reshape(y_true_bin, [tf.shape(y_true_bin)[0], -1])

#     intersection = tf.reduce_sum(y_pred_f * y_true_f, axis=1)
#     union = tf.reduce_sum(y_pred_f + y_true_f, axis=1) - intersection

#     iou = (intersection + smooth) / (union + smooth)
#     return tf.reduce_mean(iou)

# import tensorflow.keras.backend as K
# @register_keras_serializable()
# def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)

#     TP = K.sum(y_true_f * y_pred_f)
#     FP = K.sum((1 - y_true_f) * y_pred_f)
#     FN = K.sum(y_true_f * (1 - y_pred_f))

#     return 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
# @register_keras_serializable()
# def dice_coef(y_true, y_pred, smooth=1e-6):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# @register_keras_serializable()
# def combined_loss_penalizing_dice_and_sensitivity(y_true, y_pred, alpha=0.3, beta=0.7, lambda_dice=1.0):
#     # Global Tversky loss to focus on sensitivity
#     tversky = tversky_loss(y_true, y_pred, alpha=alpha, beta=beta)

#     # Dice loss for whole mask
#     dice_1 = dice_coef_edema(y_true, y_pred)
#     dice_2 = dice_coef(y_true, y_pred)
#     dice_loss_1 = 1 - dice_1
#     dice_loss_2 = 1 - dice_2

#     # Combine
#     return tversky + lambda_dice * (dice_loss_1 +dice_loss_2)



# # Final Dice Coefficient for Metrics
# @register_keras_serializable()
# def dice_coef_metric(y_true, y_pred, smooth=1.0):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     union = K.sum(y_true_f) + K.sum(y_pred_f)
#     return (2. * intersection + smooth) / (union + smooth)



# model = build_unet(num_classes=1)
# # Compile
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss=combined_loss_penalizing_dice_and_sensitivity,
#     metrics=[
#         dice_coef_metric,
#         precision,
#         sensitivity,
#         specificity,
#         dice_coef_necrotic,
#         dice_coef_edema,
#         dice_coef_enhancing,
#         iou_3d
#     ]
# )


# model.summary()

# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# checkpoint_cb = ModelCheckpoint(
#     "best_model.keras",                     # File to save to
#     monitor="val_dice_bce_loss",                 # Metric to monitor
#     mode="min",                         # Minimize the loss
#     save_best_only=True,               # Only save when it's the best so far
#     save_weights_only=False,           # Save full model
#     verbose=1
# )
# checkpoint_cb_2 = ModelCheckpoint(
#     "best_model_dice_coef_edema.keras",                     # File to save to
#     monitor="val_dice_coef_edema",                 # Metric to monitor
#     mode="max",                         # Minimize the loss
#     save_best_only=True,               # Only save when it's the best so far
#     save_weights_only=False,           # Save full model
#     verbose=1
# )
# # EarlyStopping to stop training if no improvement in 5 epochs
# early_stopping_cb = EarlyStopping(
#     monitor="val_dice_coef_edema",
#     mode="min",
#     patience=5,
#     restore_best_weights=True,   # Optional: restores weights from best epoch
#     verbose=1
# )

# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=15,
#     callbacks=[checkpoint_cb_2,early_stopping_cb]
# )
#     # steps_per_epoch=350,
#     # validation_steps=100,

# steps = 100  # or any value you want
# results_2 = model.evaluate(test_gen, verbose=1)

# # Show metrics
# for name, value in zip(model.metrics_names, results_2):
#     print(f"{name}: {value:.4f}")

# # results_2 = model.evaluate(extra_test_gen, steps=steps, verbose=1)

# # Debug output
# print("Returned metrics:", results_2)
# print("Metric names:", model.metrics_names)
# print(f"Length of results: {len(results_2)}, Length of metric names: {len(model.metrics_names)}")

# # Show metrics safely
# if len(results_2) == len(model.metrics_names):
#     for name, value in zip(model.metrics_names, results_2):
#         print(f"{name}: {value:.4f}")
# else:
#     print("Mismatch between metrics_names and evaluation results.")
#     for idx, value in enumerate(results_2):
#         print(f"Metric {idx}: {value:.4f}")

