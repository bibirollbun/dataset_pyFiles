import numpy as np
import pandas as pd
import os, math, re, copy
# uncomment below line to make computations fully deterministic, but with larger execution time
# os.environ["TF_DETERMINISTIC_OPS"] = "1"    
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
import keras
import random
from keras import layers, optimizers, metrics, Input, regularizers, initializers
from pathlib import Path
from functools import partial
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
import cv2
from skimage import exposure
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, recall_score, precision_score

import warnings

print("Tensorflow version " + tf.__version__)


# As of now, there seems to be issues with Kaggle TPUs. Hence using GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available:", len(tf.config.list_physical_devices('CPU')))


DIR_PATH = "/kaggle/input/siim-isic-melanoma-classification/"
TRAIN_PATH = DIR_PATH + "tfrecords/train*.tfrec"
TEST_PATH = DIR_PATH + "tfrecords/test*.tfrec"

#Since input jpegs are of varying size and mostly as large as 6000x4000 using the 512x512 dataset generously provided here
#                                  https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/164092
TRAIN_JPEG_PATH = "/kaggle/input/jpeg-melanoma-512x512/train/"  
TEST_JPEG_PATH = "/kaggle/input/jpeg-melanoma-512x512/test/"

TRAIN_TABDATA_PATH = DIR_PATH + "train.csv"
TEST_TABDATA_PATH = DIR_PATH + "test.csv"

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
BATCH_SIZE_TEST = 256
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 16
IMAGE_RESIZE = [384, 384]

GRAD_ACCU_STEPS = 4

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 5

#Does not support setting both TRAIN_VALID_SPLIT and TRAIN_ON_FULL_DATA as True 
TRAIN_VALID_SPLIT = False
TRAIN_ON_FULL_DATA = True
TEST_PREDICT = True
STAGE2 = True

VALID_TTA = False  # obtain valid data predictions by taking predictions from multiple augmented valid datasets and taking mean/median

RANDOM_SEED = 0

TABULAR_FEATURE_DIM = 10

CATEGORIES_SEX = ['male', 'female', 'unknown']
CATEGORIES_SITE = ['head/neck', 'upper extremity', 'lower extremity',
                   'torso', 'palms/soles', 'oral/genital', 'unknown']

MODEL_STAGE1_FILE_NAME = "effnetv2_m.keras"
MODEL_STAGE2_FILE_NAME = "effnetv2_m_stage2.keras"

MODEL_LOAD_FILE_PATH = "/kaggle/input/melanomaclassification/tensorflow2/effnetv2_384x384/3/effnetv2_m_tv_stage2.keras"

SAVE_TRAIN_VALID_MODEL = False
MODEL_TRAIN_VALID_STAGE1_FILE_NAME = "effnetv2_m_tv.keras"
MODEL_TRAIN_VALID_STAGE2_FILE_NAME = "effnetv2_m_tv_stage2.keras"

LOAD_STAGE1_TRAIN_VALID_MODEL = False

LOAD_MODEL_FOR_TEST_PREDICT = False

if TRAIN_ON_FULL_DATA and TRAIN_VALID_SPLIT:
    raise Exception("Setting both TRAIN_VALID_SPLIT and TRAIN_ON_FULL_DATA as True not supported")


# ensure reproducibility(to some extent) across different runs
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


train = pd.read_csv(TRAIN_TABDATA_PATH)
train["image_path"] = train["image_name"].apply(lambda x: os.path.join(TRAIN_JPEG_PATH, f"{x}.jpg"))
test = pd.read_csv(TEST_TABDATA_PATH)
test["image_path"] = test["image_name"].apply(lambda x: os.path.join(TEST_JPEG_PATH, f"{x}.jpg"))


train.head()


test.head()


print(train.shape, test.shape)


print("Train")
print(train.isna().sum())

print("\n----------------\n")

print("Test")
print(test.isna().sum())


train.age_approx.describe()


sum(train.age_approx == 0)


train.loc[train.age_approx == 0, :]


train.loc[train.patient_id == "IP_1300691", :]


diff_age_patients = train.dropna(subset="age_approx").groupby("patient_id")["age_approx"].agg(["min", "max", "median", "count"])
diff_age_patients["diff"] = diff_age_patients["max"] - diff_age_patients["min"]
diff_age_patients.sort_values("diff", ascending=False)


train.loc[train.patient_id == "IP_6796539", :]


train.age_approx.unique(), test.age_approx.unique()


train.loc[train.age_approx != 0, :].describe()


train["sex"] = train["sex"].fillna("unknown")

train["anatom_site_general_challenge"] = train["anatom_site_general_challenge"].fillna("unknown")
test["anatom_site_general_challenge"] = test["anatom_site_general_challenge"].fillna("unknown")

train["age_missing"] = train.age_approx.isna()
train["age_missing"] = train["age_missing"].astype(int)

test["age_missing"] = test.age_approx.isna()
test["age_missing"] = test["age_missing"].astype(int)

train["age_approx"] = train["age_approx"].fillna(-10)


train.age_approx.unique(), test.age_approx.unique()


print("Train")
print(train.isna().sum())

print("\n----------------\n")

print("Test")
print(test.isna().sum())


train.head()


test.head()


target_counts = train.target.value_counts()
target_counts


target_counts[0]*100/sum(target_counts), target_counts[1]*100/sum(target_counts)


print(train.shape, test.shape)
print(len(train.image_name.unique()), len(test.image_name.unique()))


train_patientids = pd.DataFrame(train.patient_id.value_counts())
display(train_patientids.describe())
test_patientids = pd.DataFrame(test.patient_id.value_counts())
display(test_patientids.describe())


def hist_and_box(df, col_name, main_title, title, hist_xlabel, bin_range=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.histplot(data=df[col_name].values, binwidth=5, binrange=bin_range, kde=True, ax=axes[0])
    axes[0].set_title("Histogram of " + title)
    axes[0].set_xlabel(hist_xlabel)
    axes[0].set_ylabel("Frequency")
    
    sns.boxplot(data=df[col_name].values, ax=axes[1])
    axes[1].set_title("Boxplot of " + title)
    axes[1].set_ylabel(hist_xlabel)

    plt.suptitle(main_title)
    
    plt.tight_layout()
    plt.show()


warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")

hist_and_box(train_patientids, "count", "Train data", "images per patient", "Number of images for a patient")

hist_and_box(test_patientids, "count", "Test data", "images per patient", "Number of images for a patient")


venn2(subsets = (set(train.patient_id.unique()), set(test.patient_id.unique())),
      set_labels = ('Train Patient IDs', 'Test Patient IDs'))
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.countplot(data=train, x="sex", ax=axes[0])
for container in axes[0].containers:
    axes[0].bar_label(container)
axes[0].set_title("Train data")

axes[1] = sns.countplot(data=test, x="sex", ax=axes[1])
for container in axes[1].containers:
    axes[1].bar_label(container)
axes[1].set_title("Test data")    

plt.tight_layout() 
plt.show()


ax = sns.countplot(data=train, x="sex",
                   hue="target")
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("Sex and target count")
ax.tick_params(axis='x', labelrotation=90)

plt.tight_layout()
plt.show()


print(np.sort(train.anatom_site_general_challenge.unique()))
print(np.sort(test.anatom_site_general_challenge.unique()))


anatom_site_order = train.anatom_site_general_challenge.value_counts().index
anatom_site_order


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.countplot(data=train, x="anatom_site_general_challenge", 
              order=anatom_site_order,
              ax=axes[0])
for container in axes[0].containers:
    axes[0].bar_label(container)
axes[0].set_title("Train data")
axes[0].tick_params(axis='x', labelrotation=90)

sns.countplot(data=test, x="anatom_site_general_challenge",
              order=anatom_site_order,
              ax=axes[1])
for container in axes[1].containers:
    axes[1].bar_label(container)
axes[1].set_title("Test data")   
axes[1].tick_params(axis='x', labelrotation=90)

plt.tight_layout()
plt.show()


ax = sns.countplot(data=train, x="anatom_site_general_challenge",
                   hue="target",
                   order=anatom_site_order)
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("Anatom_site and target count")
ax.tick_params(axis='x', labelrotation=90)

plt.tight_layout()
plt.show()


print("Train")
print(train.age_approx.describe())

print("\n----------------\n")

print("Test")
print(test.age_approx.describe())


hist_and_box(train, "age_approx", "Train data", "patient age", "Patient age", (-10,100))
hist_and_box(test, "age_approx", "Test data", "patient age", "Patient age", (-10,100))


hist_and_box(train.loc[train.target==1, :], "age_approx", "Malignant", "patient age", "Patient age", (-10,100))
hist_and_box(train.loc[train.target==0, :], "age_approx", "Benign", "patient age", "Patient age", (-10,100))


train.loc[(train.target==1) & (train.age_approx==-10.0), :]


def show_images(imgs):
    plt.figure(figsize=(20,8))
    for i,k in enumerate(imgs):
        img = cv2.imread(k)
        img = cv2.resize(img, IMAGE_RESIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2,5,i+1); plt.axis('off')
        plt.imshow(img)
    plt.show()

print("Examples : Malignant (With Melanoma)")
imgs = train.loc[train.target==1].sample(10).image_path.values
show_images(imgs)

print("Examples : Benign (Without Melanoma)")
imgs = train.loc[train.target==0].sample(10).image_path.values
show_images(imgs)

print("Examples : Test")
imgs = test.sample(10).image_path.values
show_images(imgs)


def get_tabular_features(df, ref_age_min, ref_age_max):
    """ 
        df : Input dataframe
        ref_age_min : Min age_approx of reference dataset
        ref_age_max : Max age_approx of reference dataset
        return tabular_features one-hot encoded for use with jpeg image data 
           and dense_hash hashtable for use with TFRecord
    """

    tabular_features = pd.get_dummies(df, columns=["sex", "anatom_site_general_challenge"], dtype=np.float32)

    #represent unknown of both columns with all 0s in 1-hot encoding
    drop_columns = ["sex_unknown", "anatom_site_general_challenge_unknown"]
    tabular_features.drop(columns=[c for c in drop_columns if c in tabular_features.columns], inplace=True) 
    
    required_columns = (["sex_"+col for col in CATEGORIES_SEX[:-1]] +
                        ["age_approx", "age_missing"] +
                        ["anatom_site_general_challenge_"+col for col in CATEGORIES_SITE[:-1]])
    tabular_features = tabular_features[required_columns]

    tabular_features["age_approx"] = (tabular_features["age_approx"] - ref_age_min) / (ref_age_max - ref_age_min)

    dense_hash = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string,
                                                       value_dtype=tf.float32,
                                                       default_value=tf.constant([-1.0] * len(required_columns), dtype=tf.float32),
                                                       empty_key='',
                                                       deleted_key='$')
    dense_hash.insert(tf.constant(df["image_name"].values), 
                      tf.constant(tabular_features.values.tolist(), dtype=tf.float32))

    return tabular_features, dense_hash


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_RESIZE)
    return image


# uncomment code below to see raw tfrecord info and verify data type

# raw_dataset = tf.data.TFRecordDataset("/kaggle/input/siim-isic-melanoma-classification/tfrecords/train09-2071.tfrec")
# for raw_record in raw_dataset.take(5):
#     example = tf.train.Example()
#     example.ParseFromString(raw_record.numpy())
#     print(example)


def read_tfrecord(example, dense_hash, labeled):
    tfrecord_format = {
        "image"                        : tf.io.FixedLenFeature([], tf.string),
        "image_name"                   : tf.io.FixedLenFeature([], tf.string),      
        "target"                       : tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image"                        : tf.io.FixedLenFeature([], tf.string),          
        "image_name"                   : tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image_name = example["image_name"]
    image = decode_image(example["image"])
    tabular_features = dense_hash.lookup(image_name)
    tabular_features = tf.ensure_shape(tabular_features, (TABULAR_FEATURE_DIM,)) 
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return (image, tabular_features), label
    return (image, tabular_features), image_name

def load_dataset(filenames, dense_hash, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(partial(read_tfrecord, dense_hash=dense_hash, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return dataset


train_filenames = tf.io.gfile.glob(TRAIN_PATH) 
test_filenames = tf.io.gfile.glob(TEST_PATH)

print('Train TFRecord Files:', len(train_filenames))
print('Test TFRecord Files:', len(test_filenames))


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

num_training_images = count_data_items(train_filenames)
num_test_images = count_data_items(test_filenames)
print(
    'Dataset: {} training images, {} unlabeled test images'.format(
        num_training_images, num_test_images
    )
)


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = decode_image(image)
    return image

def process_data(image_path, label, tabular_row):
    return (load_image(image_path), tf.cast(tabular_row, tf.float32)), tf.cast(label, tf.int32)

def load_jpeg_dataset(image_paths, labels, tabular_data):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels, tabular_data))
    dataset = dataset.map(lambda x, y, z: process_data(x, y, z), num_parallel_calls=AUTOTUNE)
    return dataset


# # Visualize augmentations

# image = tf.io.decode_jpeg(tf.io.read_file("/kaggle/input/jpeg-melanoma-512x512/train/ISIC_0149568.jpg"))  # malignant
# image = tf.io.decode_jpeg(tf.io.read_file("/kaggle/input/jpeg-melanoma-512x512/train/ISIC_0188432.jpg"))  # malignant
# image = tf.io.decode_jpeg(tf.io.read_file("/kaggle/input/jpeg-melanoma-512x512/train/ISIC_2637011.jpg"))  # benign
# image = tf.io.decode_jpeg(tf.io.read_file("/kaggle/input/cat-image/cat.jpg"))                             # cat image

# image = tf.image.resize(image, (224, 224)).numpy() / 255.0
# image = tf.expand_dims(image, axis=0)  # Add batch dimension

# num_samples = 6

# # random_layer = layers.RandomZoom((-0.2, -0.19), fill_mode="constant")
# random_layer = layers.RandomRotation((-0.3, -0.1), fill_mode="nearest")
# # random_layer = layers.RandomFlip(mode="horizontal_and_vertical")
# # random_layer = layers.RandomBrightness(0.15, value_range=(0, 1))

# # random_layer = layers.RandomBrightness((0.05, 0.15), value_range=(0, 1))
# # random_layer = layers.RandomBrightness((-0.15, -0.05), value_range=(0, 1))
# # random_layer = layers.Lambda(lambda x: tf.image.random_saturation(x, 0.9, 0.91))
# # random_layer = layers.Lambda(lambda x: tf.image.random_hue(x, 0.03))
# # random_layer = layers.Lambda(lambda x: tf.image.random_contrast(x, 0.7, 0.9))
# # random_layer = layers.Lambda(lambda x: tf.image.random_contrast(x, 1.39, 1.4))

# # Generate augmented images
# augmented_images = [random_layer(image, training=True)[0].numpy() for _ in range(num_samples)]

# # Plot the original and augmented images
# fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 5))
# axes[0].imshow(image[0])
# axes[0].set_title("Original")
# axes[0].axis("off")

# for i in range(num_samples):
#     axes[i + 1].imshow(augmented_images[i])
#     axes[i + 1].set_title(f"Augmented {i+1}")
#     axes[i + 1].axis("off")

# plt.tight_layout()
# plt.show()


layer_flip = layers.RandomFlip(mode="horizontal_and_vertical", seed=RANDOM_SEED)
layer_zoom_in = layers.RandomZoom((-0.2, -0.1), fill_mode="nearest", seed=RANDOM_SEED)
layer_zoom_out = layers.RandomZoom((0.1, 0.2), fill_mode="nearest", seed=RANDOM_SEED)
layer_rotate_clockwise = layers.RandomRotation((-0.3, -0.1), fill_mode="nearest", seed=RANDOM_SEED)
layer_rotate_anticlockwise = layers.RandomRotation((0.1, 0.3), fill_mode="nearest", seed=RANDOM_SEED)
layer_brightness_up = layers.RandomBrightness((0.05, 0.1), seed=RANDOM_SEED)
layer_brightness_down = layers.RandomBrightness((-0.1, -0.05), seed=RANDOM_SEED)

# Augmentation functions
def identity(x): return x
def apply_flip(x): return layer_flip(x, training=True)
# def apply_zoom(x): return layer_zoom(x, training=True)
def apply_zoom_in(x): return layer_zoom_in(x, training=True)
def apply_zoom_out(x): return layer_zoom_out(x, training=True)
# def apply_brightness(x): return layer_brightness(x, training=True)
def apply_rot_clock(x): return layer_rotate_clockwise(x, training=True)
def apply_rot_anticlock(x): return layer_rotate_anticlockwise(x, training=True)    
def apply_brightness_up(x): return layer_brightness_up(x, training=True)
def apply_brightness_down(x): return layer_brightness_down(x, training=True)
def apply_saturation_up(x): return tf.image.random_saturation(x, 1.1, 1.25, seed=RANDOM_SEED)
def apply_saturation_down(x): return tf.image.random_saturation(x, 0.75, 0.9, seed=RANDOM_SEED)    
def apply_contrast_up(x): return tf.image.random_contrast(x, 1.1, 1.25, seed=RANDOM_SEED)
def apply_contrast_down(x): return tf.image.random_contrast(x, 0.75, 0.9, seed=RANDOM_SEED)
def apply_hue(x): return tf.image.random_hue(x, 0.03, seed=RANDOM_SEED)    

augmentation_functions_1 = [identity, apply_hue, apply_brightness_up, apply_brightness_down, 
                            apply_saturation_up, apply_saturation_down, apply_contrast_up, apply_contrast_down]
augmentation_functions_2 = [apply_rot_clock, apply_rot_anticlock, apply_flip, apply_zoom_in]

def apply_augmentation(idx1, idx2, img):
    img = augmentation_functions_1[idx1](img)
    img = augmentation_functions_2[idx2](img)    
    return img
    
def augmentation_pipeline(data, label, rand_idx_1=None, rand_idx_2=None):
    image, tab_features = data
    
    # Generate a random integers
    if rand_idx_1 is None:
        rand_idx_1 = np.random.randint(0, len(augmentation_functions_1), dtype=np.int32)
    if rand_idx_2 is None:
        rand_idx_2 = np.random.randint(0, len(augmentation_functions_2), dtype=np.int32)
    
    # Apply the randomly selected augmentation
    image = apply_augmentation(rand_idx_1, rand_idx_2, image)

    return (image, tab_features), label


# # image = cv2.imread("/kaggle/input/cat-image/cat.jpg")
# # image = cv2.imread("/kaggle/input/jpeg-melanoma-512x512/train/ISIC_2637011.jpg")
# image = cv2.imread("/kaggle/input/jpeg-melanoma-512x512/train/ISIC_0188432.jpg")
# image = cv2.resize(image, IMAGE_RESIZE)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert NumPy image to TensorFlow tensor

# # Generate random indices (explicitly as int32 scalar tensors)
# # rand_nums_1 = tf.constant([0, 1, 2, 3], dtype=tf.int32)
# # rand_nums_2 = tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)

# l1 = len(augmentation_functions_1)
# l2 = len(augmentation_functions_2)

# rand_nums_1 = np.array([i for i in range(l1)], dtype=np.int32)
# rand_nums_2 = np.array([i for i in range(l2)], dtype=np.int32)

# pairs = tf.experimental.numpy.meshgrid(rand_nums_1, rand_nums_2, indexing="ij")
# pairs = tf.stack([tf.reshape(pairs[0], [-1]), tf.reshape(pairs[1], [-1])], axis=1)

# # Apply augmentations
# augmented_images = []
# for pair in pairs.numpy():
#     # print(pair[0], pair[1])
#     aug_image = apply_augmentation(pair[0], pair[1], image)
#     augmented_images.append(aug_image)
# print(len(augmented_images))

# # Convert tensors to NumPy for visualization
# augmented_images = [img.numpy().astype(np.uint8) for img in augmented_images]

# fig, axes = plt.subplots(l1, l2, figsize=(l2*2, l1*2))

# for i in range(len(augmentation_functions_1)):
#     for j in range(len(augmentation_functions_2)):  
#         axes[i, j].imshow(augmented_images[i*l2+j])
#         axes[i, j].set_title(f"{pairs.numpy()[i*l2+j][0]}, {pairs.numpy()[i*l2+j][1]}")
#         axes[i, j].axis("off")

# plt.tight_layout()
# plt.show()


#For tfrecord, pass tfrecord=True and provide dense_hash as tabular_data and also provide filenames
#For jpeg, pass tfrecord=False and provide image_paths and labels and tabular_features as tabular_data

def get_training_dataset(tfrecord=True, filenames=None, image_paths=None, labels=None, tabular_data=None):
    if tfrecord:
        dataset = load_dataset(filenames, dense_hash=tabular_data, labeled=True)
    else: #jpeg
        dataset = load_jpeg_dataset(image_paths, labels, tabular_data)
    dataset = dataset.map(augmentation_pipeline, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE, seed=RANDOM_SEED)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_validation_dataset(tfrecord=True, filenames=None, image_paths=None, 
                           labels=None, tabular_data=None, augment=False,
                           rand_idx_1=None, rand_idx_2=None):
    if tfrecord:
        dataset = load_dataset(filenames, dense_hash=tabular_data, labeled=True) 
    else: #jpeg
        dataset = load_jpeg_dataset(image_paths, labels, tabular_data)
    if augment:
        dataset = dataset.map(partial(augmentation_pipeline, rand_idx_1=rand_idx_1, rand_idx_2=rand_idx_2), 
                              num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_test_dataset(filenames=None, tabular_data=None, 
                     augment=False,
                     rand_idx_1=None, rand_idx_2=None):   # only use TFRecord for predicting test data
    dataset = load_dataset(filenames, dense_hash=tabular_data, labeled=False)
    if augment:
        dataset = dataset.map(partial(augmentation_pipeline, rand_idx_1=rand_idx_1, rand_idx_2=rand_idx_2), 
                              num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE_TEST)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def show_batch(image_batch, tab_data_batch, element3_batch, labeled=True):
    plt.figure(figsize=(10,10))
    image_batch = tf.cast(image_batch, tf.float32) / 255.0
    for n in range(16):
        ax = plt.subplot(4,4,n+1)
        plt.imshow(image_batch[n])

        if labeled:
            element_3 = "Malignant" if element3_batch[n] else "Benign"
        else:
            element_3 = element3_batch[n].decode("utf-8") # gets image name from test images without label
        tabular_info = tab_data_batch[n].round(2)

        sex = "male" if tabular_info[0] else "female" if tabular_info[1] else "unknown"
        age = tabular_info[2]
        age_missing = (tabular_info[2] == 1)
        a_site = CATEGORIES_SITE[np.where(tabular_info[4:])[0][0] if sum(tabular_info[4:]) else 6]
        
        plt.title(f"{element_3}\n{tabular_info}\nSex : {sex}\n Age scaled : {age:.3f} Age_missing : {age_missing}\n Anatomical Site : {a_site}", 
                  fontsize=6)
        
        plt.axis("off")

    plt.tight_layout()
    plt.show()


_, train_hash = get_tabular_features(train, train.age_approx.min(), train.age_approx.max())
train_dataset = get_training_dataset(filenames=train_filenames, tabular_data=train_hash)

(image_batch, tab_data_batch), label_batch = next(iter(train_dataset))
show_batch(image_batch.numpy(), tab_data_batch.numpy(), label_batch.numpy())


train_index, valid_index = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(train.image_name, train.target, train.patient_id))

print(len(train_index), len(valid_index))
print(len(train_index)/train.shape[0], len(valid_index)/train.shape[0])


train_data = train.loc[train_index, :]
valid_data = train.loc[valid_index, :]


venn2(subsets = (set(train_data.patient_id.unique()), 
                 set(valid_data.patient_id.unique())),
      set_labels = ('Trainset Patient IDs', 'Validset Patient IDs'))
plt.show()


t_target_counts = train_data.target.value_counts()
print(t_target_counts[0]*100/sum(t_target_counts), t_target_counts[1]*100/sum(t_target_counts))

v_target_counts = valid_data.target.value_counts()
print(v_target_counts[0]*100/sum(v_target_counts), v_target_counts[1]*100/sum(v_target_counts))


print(train_data.sex.value_counts())
print(valid_data.sex.value_counts())


print(train_data.anatom_site_general_challenge.value_counts())
print(valid_data.anatom_site_general_challenge.value_counts())


train_tabular_features, _ = get_tabular_features(train_data, train_data.age_approx.min(), train_data.age_approx.max())
train_jpeg_dataset = get_training_dataset(tfrecord=False, 
                                          image_paths=train_data.image_path.values, 
                                          labels=train_data.target.values,
                                          tabular_data=train_tabular_features.values)
(image_batch, tab_data_batch), label_batch = next(iter(train_jpeg_dataset))
show_batch(image_batch.numpy(), tab_data_batch.numpy(), label_batch.numpy())


                                                    # Use train_data age for min-max normalization
valid_tabular_features, _ = get_tabular_features(valid_data, train_data.age_approx.min(), train_data.age_approx.max())
valid_jpeg_dataset = get_validation_dataset(tfrecord=False,
                                            image_paths=valid_data.image_path.values,
                                            labels=valid_data.target.values,
                                            tabular_data=valid_tabular_features.values)
(image_batch, tab_data_batch), label_batch = next(iter(valid_jpeg_dataset))
show_batch(image_batch.numpy(), tab_data_batch.numpy(), label_batch.numpy())


if TRAIN_ON_FULL_DATA or TRAIN_VALID_SPLIT:

    # Image Model
    base_model = keras.applications.EfficientNetV2M(
        include_top=False,
        input_shape=(*IMAGE_RESIZE, 3),
        include_preprocessing=True
    )
	# for i, layer in enumerate(base_model.layers):
	#     if "a_expand_conv" in layer.name:
	#         print(f"Layer index: {i}, Layer name: {layer.name}")
	# print(len(base_model.layers))
	# # Layer index: 19, Layer name: block2a_expand_conv
	# # Layer index: 52, Layer name: block3a_expand_conv
	# # Layer index: 85, Layer name: block4a_expand_conv
	# # Layer index: 188, Layer name: block5a_expand_conv
	# # Layer index: 396, Layer name: block6a_expand_conv
	# # Layer index: 664, Layer name: block7a_expand_conv
	# # 740
    base_model.trainable = False
    for layer in base_model.layers[396:]:  
        layer.trainable = True  
 
    image_inputs = Input(shape=(*IMAGE_RESIZE, 3), name="image_input_layer")
    x = base_model(image_inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Tabular Model
    tabular_inputs = Input(shape=(TABULAR_FEATURE_DIM,), name="tabular_input_layer")
    t = layers.BatchNormalization()(tabular_inputs)

    #Merge image and tabular
    merged = layers.Concatenate()([x, t])
    merged = layers.Dense(128, 
                          activation="relu",
                          kernel_initializer=initializers.HeNormal(seed=RANDOM_SEED),
                          kernel_regularizer=regularizers.l2(1e-4)
                         )(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.2)(merged)
    merged = layers.Dense(16, 
                          activation="relu",
                          kernel_initializer=initializers.HeNormal(seed=RANDOM_SEED),
                          kernel_regularizer=regularizers.l2(1e-4)
                         )(merged)
    merged = layers.BatchNormalization()(merged)
    outputs = layers.Dense(1, 
                           activation='sigmoid',
                           kernel_initializer=initializers.GlorotNormal(seed=RANDOM_SEED),
                          )(merged)        
    
    model = keras.Model([image_inputs, tabular_inputs], outputs)
    
    model.summary(show_trainable=True)


if TRAIN_VALID_SPLIT:
    steps_per_epoch = int(np.ceil(train_data.shape[0] / BATCH_SIZE))
elif TRAIN_ON_FULL_DATA:
    steps_per_epoch = int(np.ceil(train.shape[0] / BATCH_SIZE))


if TRAIN_ON_FULL_DATA or TRAIN_VALID_SPLIT:

    initial_lr = 1e-4
    warmup_target_lr = 1e-2
    final_target_lr = 1e-3
    warmup_epochs = 5
    alpha = final_target_lr / warmup_target_lr
    decay_epochs = STAGE1_EPOCHS - warmup_epochs
    lr_scheduler = optimizers.schedules.CosineDecay(
        initial_learning_rate = initial_lr,
        warmup_target = warmup_target_lr,
        warmup_steps = steps_per_epoch * warmup_epochs,
        alpha = alpha,
        decay_steps = steps_per_epoch * decay_epochs,
    )
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_scheduler, gradient_accumulation_steps=GRAD_ACCU_STEPS),
        loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.05, alpha=0.25),  
        metrics=[metrics.AUC(), metrics.Recall(), metrics.Precision()]
    )


def plot_lr_scheduler(lr_scheduler, total_epochs):
	#step-wise
	steps = np.arange(steps_per_epoch * total_epochs)
	lr_values = [lr_scheduler(step) for step in steps]
	
	# epoch-wise
	epochs = np.arange(total_epochs)
	epoch_lr_values = [lr_scheduler(epoch * steps_per_epoch) for epoch in epochs]
	
	# Plot step-wise and epoch-wise learning rate schedules side by side
	fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
	
	# step-wise LR plot
	axes[0].plot(steps, lr_values, label="LR per Step", color="b")
	axes[0].set_xlabel("Steps")
	axes[0].set_ylabel("Learning Rate")
	axes[0].set_title("Step-wise Learning Rate Schedule")
	axes[0].set_yscale("log")
	axes[0].legend()
	axes[0].grid(True, which="both", linestyle="--")
	
	# epoch-wise LR plot
	axes[1].plot(epochs, epoch_lr_values, marker='o', linestyle='dotted', label="LR per Epoch", color="g")
	axes[1].set_xlabel("Epochs")
	axes[1].set_title("Epoch-wise Learning Rate Schedule")
	axes[1].set_yscale("log")
	axes[1].legend()
	axes[1].grid(True, which="both", linestyle="--")
	
	# Show plots
	plt.tight_layout()
	plt.show()


if TRAIN_ON_FULL_DATA or TRAIN_VALID_SPLIT:
    plot_lr_scheduler(lr_scheduler, STAGE1_EPOCHS)


if TRAIN_VALID_SPLIT:
    #Load the train and valid dataset again, since we have iterated over them to show_batch
    train_jpeg_dataset = get_training_dataset(tfrecord=False, 
                                              image_paths=train_data.image_path.values, 
                                              labels=train_data.target.values,
                                              tabular_data=train_tabular_features.values)
    valid_jpeg_dataset = get_validation_dataset(tfrecord=False,
                                                image_paths=valid_data.image_path.values,
                                                labels=valid_data.target.values,
                                                tabular_data=valid_tabular_features.values)

    # Get class weights since the data is highly unbalanced
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]),
                                         y=train_data.target.values)
    class_weights_dict = {0: round(class_weights[0], 3), 1: round(class_weights[1], 3)}
    # print(class_weights_dict)
    # {0: 0.509, 1: 28.479}

    if not LOAD_STAGE1_TRAIN_VALID_MODEL:
        history = model.fit(train_jpeg_dataset, 
                            epochs=STAGE1_EPOCHS,
                            validation_data=valid_jpeg_dataset,
                            class_weight=class_weights_dict)


if TRAIN_VALID_SPLIT and not LOAD_STAGE1_TRAIN_VALID_MODEL:
    history_frame = pd.DataFrame(history.history)
    display(history_frame)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    history_frame.loc[:, ['loss', 'val_loss']].plot(ax=axes[0,0])
    history_frame.loc[:, ['auc', 'val_auc']].plot(ax=axes[0,1])
    history_frame.loc[:, ['precision', 'val_precision']].plot(ax=axes[1,0])
    history_frame.loc[:, ['recall', 'val_recall']].plot(ax=axes[1,1])
    history_frame.loc[:, ['precision', 'recall']].plot(ax=axes[2,0])
    history_frame.loc[:, ['val_precision', 'val_recall']].plot(ax=axes[2,1])
    plt.tight_layout()
    plt.show()


if TRAIN_VALID_SPLIT and SAVE_TRAIN_VALID_MODEL and not LOAD_STAGE1_TRAIN_VALID_MODEL:
    model.save(MODEL_TRAIN_VALID_STAGE1_FILE_NAME)


if TRAIN_VALID_SPLIT and LOAD_STAGE1_TRAIN_VALID_MODEL:
    model = keras.models.load_model(MODEL_LOAD_FILE_PATH)
    model.summary(show_trainable=True)


# predict on valid_dataset to verify if metrics match the last epoch metrics of training the loaded model
if TRAIN_VALID_SPLIT and LOAD_STAGE1_TRAIN_VALID_MODEL:
    prediction_prob = model.predict(valid_jpeg_dataset)[:, 0]
    auc = roc_auc_score(valid_data.target.values, prediction_prob)
    recall = recall_score(valid_data.target.values, (prediction_prob > 0.5).astype(int))
    precision = precision_score(valid_data.target.values, (prediction_prob > 0.5).astype(int))
    print({"AUC": auc, "Recall": recall, "Precision": precision})


if TRAIN_ON_FULL_DATA:
    #Load the dataset again, since we have iterated over it to show_batch
    train_dataset = get_training_dataset(filenames=train_filenames, tabular_data=train_hash)

    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]),
                                         y=train.target.values)
    class_weights_dict = {0: round(class_weights[0], 3), 1: round(class_weights[1], 3)}
    # print(class_weights_dict)
    # {0: 0.509, 1: 28.361}    
    
    history = model.fit(train_dataset, 
                        epochs=STAGE1_EPOCHS,
                        class_weight=class_weights_dict)


if TRAIN_ON_FULL_DATA:
    history_frame = pd.DataFrame(history.history)
    display(history_frame)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    history_frame.loc[:, ['auc']].plot(ax=axes[0,0])
    history_frame.loc[:, ['loss']].plot(ax=axes[0,1])
    history_frame.loc[:, ['precision']].plot(ax=axes[1,0])
    history_frame.loc[:, ['recall']].plot(ax=axes[1,1])
    plt.tight_layout()
    plt.show()


if TRAIN_ON_FULL_DATA:
    model.save(MODEL_STAGE1_FILE_NAME)


if (TRAIN_ON_FULL_DATA or TRAIN_VALID_SPLIT) and STAGE2:

    model.get_layer("efficientnetv2-m").trainable = False
    for layer in model.get_layer("efficientnetv2-m").layers[85:]:  
        layer.trainable = True 
        
    model.summary(show_trainable=True)

    initial_lr = 1e-5
    warmup_target_lr = 5e-5
    final_target_lr = 2.5e-5
    warmup_epochs = 3
    alpha = final_target_lr / warmup_target_lr
    decay_epochs = STAGE2_EPOCHS - warmup_epochs
    lr_scheduler = optimizers.schedules.CosineDecay(
        initial_learning_rate = initial_lr,
        warmup_target = warmup_target_lr,
        warmup_steps = steps_per_epoch * warmup_epochs,
        alpha = alpha,
        decay_steps = steps_per_epoch * decay_epochs,
    )    
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_scheduler, gradient_accumulation_steps=GRAD_ACCU_STEPS),
        loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0, label_smoothing=0.1, alpha=0.25),  
        metrics=[metrics.AUC(), metrics.Recall(), metrics.Precision()]
    )


if (TRAIN_ON_FULL_DATA or TRAIN_VALID_SPLIT) and STAGE2:
    plot_lr_scheduler(lr_scheduler, STAGE2_EPOCHS)


if TRAIN_VALID_SPLIT and STAGE2:
    history = model.fit(train_jpeg_dataset, 
                        epochs=STAGE2_EPOCHS,
                        validation_data=valid_jpeg_dataset,
                        class_weight=class_weights_dict)


if TRAIN_VALID_SPLIT and STAGE2:
    history_frame = pd.DataFrame(history.history)
    display(history_frame)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    history_frame.loc[:, ['loss', 'val_loss']].plot(ax=axes[0,0])
    history_frame.loc[:, ['auc_1', 'val_auc_1']].plot(ax=axes[0,1])
    history_frame.loc[:, ['precision_1', 'val_precision_1']].plot(ax=axes[1,0])
    history_frame.loc[:, ['recall_1', 'val_recall_1']].plot(ax=axes[1,1])
    history_frame.loc[:, ['precision_1', 'recall_1']].plot(ax=axes[2,0])
    history_frame.loc[:, ['val_precision_1', 'val_recall_1']].plot(ax=axes[2,1])
    plt.tight_layout()
    plt.show()


if TRAIN_VALID_SPLIT and STAGE2 and SAVE_TRAIN_VALID_MODEL:
    model.save(MODEL_TRAIN_VALID_STAGE2_FILE_NAME)    


if TRAIN_ON_FULL_DATA and STAGE2: 
    history = model.fit(train_dataset, 
                        epochs=STAGE2_EPOCHS,
                        class_weight=class_weights_dict)


if TRAIN_ON_FULL_DATA and STAGE2:
    history_frame = pd.DataFrame(history.history)
    display(history_frame)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    history_frame.loc[:, ['auc_1']].plot(ax=axes[0,0])
    history_frame.loc[:, ['loss']].plot(ax=axes[0,1])
    history_frame.loc[:, ['precision_1']].plot(ax=axes[1,0])
    history_frame.loc[:, ['recall_1']].plot(ax=axes[1,1])
    plt.tight_layout()
    plt.show()


if TRAIN_ON_FULL_DATA and STAGE2:
    model.save(MODEL_STAGE2_FILE_NAME)    


if TRAIN_VALID_SPLIT and VALID_TTA:

    #rot_anticlock, brightness down and contrast down found to give poor results in valid_data TTA. 
    #So redefining augmentation_functions avoiding those.
    augmentation_functions_1 = [identity, apply_hue, apply_brightness_up, 
                                apply_saturation_up, apply_saturation_down, 
                                apply_contrast_up]
    augmentation_functions_2 = [apply_rot_clock, apply_flip, apply_zoom_in]

    valid_datasets = [get_validation_dataset(tfrecord=False,
                                             image_paths=valid_data.image_path.values,
                                             labels=valid_data.target.values,
                                             tabular_data=valid_tabular_features.values)
                     ]

    for _ in range(len(augmentation_functions_1)*len(augmentation_functions_2)):
        valid_datasets.append(
            get_validation_dataset(tfrecord=False, image_paths=valid_data.image_path.values, 
                                   labels=valid_data.target.values, tabular_data=valid_tabular_features.values, 
                                   augment=True)
        )
    
    tta_predictions = []
    for dataset in valid_datasets:
        prediction_prob = model.predict(dataset)
        tta_predictions.append(prediction_prob[:, 0])


if TRAIN_VALID_SPLIT and VALID_TTA:

    results = []
    
    # Compute metrics for non-augmented (Id = 0)
    auc = roc_auc_score(valid_data.target.values, tta_predictions[0])
    recall = recall_score(valid_data.target.values, (tta_predictions[0] > 0.5).astype(int))
    precision = precision_score(valid_data.target.values, (tta_predictions[0] > 0.5).astype(int))
    results.append({"Id": "Non-augmented", "AUC": auc, "Recall": recall, "Precision": precision})
    
    # Compute metrics for augmented versions
    for i, preds in enumerate(tta_predictions[1:], start=1):
        auc = roc_auc_score(valid_data.target.values, preds)
        recall = recall_score(valid_data.target.values, (preds > 0.5).astype(int))
        precision = precision_score(valid_data.target.values, (preds > 0.5).astype(int))
        results.append({"Id": "Augmented_"+str(i), "AUC": auc, "Recall": recall, "Precision": precision})
    
    # Compute metrics for averaged TTA predictions 
    tta_mean_predictions = np.mean(tta_predictions, axis=0)
    auc = roc_auc_score(valid_data.target.values, tta_mean_predictions)
    recall = recall_score(valid_data.target.values, (tta_mean_predictions > 0.5).astype(int))
    precision = precision_score(valid_data.target.values, (tta_mean_predictions > 0.5).astype(int))
    results.append({"Id": "Combined_with_mean", "AUC": auc, "Recall": recall, "Precision": precision})
    
    tta_median_predictions = np.median(tta_predictions, axis=0)
    auc = roc_auc_score(valid_data.target.values, tta_median_predictions)
    recall = recall_score(valid_data.target.values, (tta_median_predictions > 0.5).astype(int))
    precision = precision_score(valid_data.target.values, (tta_median_predictions > 0.5).astype(int))
    results.append({"Id": "Combined_with_median", "AUC": auc, "Recall": recall, "Precision": precision})
    
    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)
    
    display(df_results)


if TEST_PREDICT and LOAD_MODEL_FOR_TEST_PREDICT:
    model = keras.models.load_model(MODEL_LOAD_FILE_PATH)
    model.summary(show_trainable=True)


keras.utils.plot_model(model, "model.png", show_shapes=True, show_layer_names=True)


if TEST_PREDICT:
                                                # use train age for min-max scaling
    _, test_hash = get_tabular_features(test, train.age_approx.min(), train.age_approx.max())
    test_dataset = get_test_dataset(filenames=test_filenames, tabular_data=test_hash)
    (image_batch, tab_data_batch), name_batch = next(iter(test_dataset))
    show_batch(image_batch.numpy(), tab_data_batch.numpy(), name_batch.numpy(), labeled=False)


if TEST_PREDICT:
    test_dataset2 = get_test_dataset(filenames=test_filenames, tabular_data=test_hash, augment=True, rand_idx_1=0, rand_idx_2=0)
    (image_batch, tab_data_batch), name_batch = next(iter(test_dataset2))
    show_batch(image_batch.numpy(), tab_data_batch.numpy(), name_batch.numpy(), labeled=False)


def get_test_prediction(test_dataset, model):

    test_image_name = []
    prediction_probs = []

    for (image_batch, tab_data_batch), image_name_batch in test_dataset:
        # Convert image names to strings
        image_name_batch = [image_name.decode("utf-8") for image_name in image_name_batch.numpy()]
        
        # Get predictions batch-wise
        prediction_prob_batch = model.predict((image_batch, tab_data_batch), batch_size=BATCH_SIZE_TEST)

        # Store predictions and names
        test_image_name.extend(image_name_batch)
        prediction_probs.extend(prediction_prob_batch[:, 0])

    # Convert to NumPy arrays
    test_image_name = np.array(test_image_name)
    prediction_probs = np.array(prediction_probs)

    # Sort based on image names
    sorted_indices = np.argsort(test_image_name)
    test_image_name = test_image_name[sorted_indices]
    prediction_probs = prediction_probs[sorted_indices]

    return prediction_probs, test_image_name


if TEST_PREDICT:

    #generate test dataset
    test_dataset = get_test_dataset(filenames=test_filenames, tabular_data=test_hash)

    #rot_anticlock, brightness down and contrast down found to give poor results in  valid_data TTA. 
    #So redefining augmentation_functions avoiding those.
    augmentation_functions_1 = [identity, apply_hue, apply_brightness_up, 
                                apply_saturation_up, apply_saturation_down, 
                                apply_contrast_up]
    augmentation_functions_2 = [apply_rot_clock, apply_flip, apply_zoom_in]

    #generate augmented test datasets
    test_datasets = []
    for _ in range(len(augmentation_functions_1)*len(augmentation_functions_2)):
        test_datasets.append(
            get_test_dataset(filenames=test_filenames, tabular_data=test_hash, augment=True)
        ) 

    tta_predictions = []
    
    #get predictions from original test dataset   
    #        obtaining predictions once outside the below loop, so as to get test_image_name
    #        predictions are sorted on image names, so same order for original and augmented versions
    prediction_prob, test_image_name = get_test_prediction(test_dataset, model)
    tta_predictions.append(prediction_prob)

    #get predictions from augmented test datasets
    for dataset in test_datasets:
        prediction_prob, _ = get_test_prediction(dataset, model)
        tta_predictions.append(prediction_prob)

    tta_mean_predictions = np.mean(tta_predictions, axis=0)
    tta_median_predictions = np.median(tta_predictions, axis=0)


if TEST_PREDICT:
    submission = pd.DataFrame(dict(image_name=test_image_name, target=tta_mean_predictions))
    submission.to_csv('submission.csv', index=False)
    !head submission.csv

