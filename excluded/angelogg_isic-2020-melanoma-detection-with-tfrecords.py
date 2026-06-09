DEVICE = "TPU"

if DEVICE == "TPU":
    !pip install -q pydicom

RUN_TRAINING = False


# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pydicom # needed to load .dcm images
from keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D, Layer
from tensorflow.keras.layers import (
    RandomFlip, RandomRotation, RandomZoom, RandomTranslation,
    RandomBrightness, RandomContrast, GaussianNoise, Resizing, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, classification_report, precision_recall_curve,
    auc, precision_recall_fscore_support
)
from sklearn.model_selection import KFold, StratifiedGroupKFold
from tensorflow.keras import regularizers
import math
from tensorflow.keras import backend as keras_backend
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import json
import re
from enum import Enum

SEED = 42
np.random.seed(SEED)
num_additional_features = 0
AUTO = tf.data.experimental.AUTOTUNE

# Define normalization constants
MEAN = tf.constant([0.485, 0.456, 0.406], shape=(1, 1, 3), dtype=tf.float32)  # Pretrained mean
STD = tf.constant([0.229, 0.224, 0.225], shape=(1, 1, 3), dtype=tf.float32)   # Pretrained std

IMAGE_RESIZE = 224


if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

FOLDS = 3
# BATCH_SIZES = [128]*FOLDS
BATCH_SIZES = [128]*FOLDS
EPOCHS = [15]*FOLDS


# Define your TFRecord files
#INPUT_DIR = '/kaggle/input/siim-isic-melanoma-classification/tfrecords/'
# INPUT_DIR = '/kaggle/input/isic-2020-melanoma-images-and-metadata/enhanced_dataset/'
INPUT_DIR = '/kaggle/input/isic-2020-melanoma-images-and-metadata/isic2020-tfrecords-with-metadata/'
input_tfrec_train_pattern = INPUT_DIR + 'train*.tfrec'
input_tfrec_train = tf.io.gfile.glob(input_tfrec_train_pattern)
print(f"Number of training TFRecord files found: {len(input_tfrec_train)}")

input_tfrec_test_pattern = INPUT_DIR + 'test*.tfrec'
input_tfrec_test = tf.io.gfile.glob(input_tfrec_test_pattern)
print(f"Number of training TFRecord files found: {len(input_tfrec_test)}")


def inspect_tfrecord(file_path, num_samples=3):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    for raw_record in raw_dataset.take(num_samples):
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            print(example)
        except Exception as e:
            print(f"Error parsing record: {e}")
            continue

#inspect_tfrecord(INPUT_DIR + 'train00-2071-enhanced.tfrec', num_samples=3)


def extract_target(example):
    tfrec_format = {
        'target': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example, tfrec_format)
    return parsed['target']


def compute_class_ratio_from_tfrecords(tfrecord_files, max_samples=5000):
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(extract_target, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.take(max_samples)  # limit for speed

    num_pos = 0
    num_total = 0

    for label in dataset:
        label_val = label.numpy()
        num_total += 1
        if label_val == 1:
            num_pos += 1

    if num_total == 0:
        print("Warning: No samples found for oversampling calculation.")
        return 0.0, 0, 1

    pos_ratio = num_pos / num_total

    # Compute safe oversampling factor
    if pos_ratio == 0:
        repeat_factor = 1  # Can't oversample if no positives
    elif pos_ratio >= 0.5:
        repeat_factor = 1  # No need to oversample if positives dominate
    else:
        repeat_factor = int((1 - pos_ratio) / pos_ratio)

    return pos_ratio, num_total, repeat_factor


def compute_class_counts_from_tfrecords(tfrecord_files):
    dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(extract_target, num_parallel_calls=tf.data.AUTOTUNE)

    num_pos = 0
    num_neg = 0
    num_total = 0

    for label in dataset:
        val = label.numpy()
        if val == 1:
            num_pos += 1
        else:
            num_neg += 1
        num_total += 1

    return num_pos, num_neg, num_total



# Compute Global Age Statistics (Mean and Max Age) from TFRecords
def extract_age_approx(example):
    tfrec_format = {
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, tfrec_format)
    age = tf.cast(parsed_example['age_approx'], tf.float32)
    return age

def create_age_dataset(tfrecord_files):
    age_dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=AUTO)
    age_dataset = age_dataset.map(extract_age_approx, num_parallel_calls=AUTO)
    return age_dataset

def compute_age_statistics(age_dataset):
    # Initialize accumulators
    sum_age = tf.constant(0.0, dtype=tf.float32)
    count = tf.constant(0, dtype=tf.int64)
    max_age = tf.constant(0.0, dtype=tf.float32)
    
    def reducer(accum, age):
        sum_age, count, max_age = accum
        # Only consider non-missing values (assuming missing values are represented as NaN)
        condition = tf.logical_not(tf.math.is_nan(age))
        sum_age += tf.where(condition, age, 0.0)
        count += tf.cast(condition, tf.int64)
        max_age = tf.maximum(max_age, tf.where(condition, age, 0.0))
        return sum_age, count, max_age
    
    sum_age, count, max_age = age_dataset.reduce(
        (sum_age, count, max_age),
        reducer
    )
    
    # Avoid division by zero
    mean_age = tf.cond(
        tf.equal(count, 0),
        lambda: tf.constant(0.0, dtype=tf.float32),
        lambda: sum_age / tf.cast(count, tf.float32)
    )
    
    mean_age = mean_age.numpy()
    max_age = max_age.numpy()
    
    print(f"Global Mean Age: {mean_age}")
    print(f"Maximum Age: {max_age}")
    
    return mean_age, max_age


# Create the age dataset
age_dataset = create_age_dataset(input_tfrec_train)
# Compute statistics
global_mean_age, global_max_age = compute_age_statistics(age_dataset)

num_additional_features += 1 # age is one numerical feature
print(f'Number of features after age computation: {num_additional_features}')


def extract_sex(example):
    tfrec_format = {
        'sex': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, tfrec_format)
    sex = parsed_example['sex']
    # Decode bytes to string
    sex = sex.numpy().decode('utf-8')
    return sex

sex_set = set()

for tfrecord in input_tfrec_train:
    # Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(tfrecord, num_parallel_reads=tf.data.AUTOTUNE)
    
    # Iterate through each serialized example in the TFRecord
    for raw_record in dataset:
        try:
            sex = extract_sex(raw_record)
            # Replace empty strings or specific missing indicators with 'unknown'
            if not sex or sex.lower() == 'nan':
                sex = 'unknown'
            sex_set.add(sex)
        except Exception as e:
            print(f"Error parsing record: {e}")
            continue

# Add 'unknown' category to handle missing values
sex_set.add('unknown')

# Convert the set to a sorted list
sex_categories = sorted(sex_set)
print(f"Unique 'sex' Categories: {sex_categories}")
num_sex_categories = len(sex_categories)
print(f"Number of sex categories: {num_sex_categories}")

sex_to_index = {sex: idx for idx, sex in enumerate(sex_categories)}
print(f"'sex' Mapping: {sex_to_index}")

num_additional_features +=  num_sex_categories
print(f'Number of features after sex computation: {num_additional_features}')

def create_sex_lookup_table(sex_to_index):
    # Create TensorFlow tensors for keys and values
    keys = tf.constant(list(sex_to_index.keys()))
    values = tf.constant(list(sex_to_index.values()), dtype=tf.int64)

    # Set the default value (e.g., index for 'unknown')
    default_value = sex_to_index.get('unknown', 2)
    
    # Create a KeyValueTensorInitializer
    initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
    
    # Use StaticHashTable which accepts default_value
    table = tf.lookup.StaticHashTable(initializer, default_value)
    return table

# Create the lookup table
sex_lookup_table = create_sex_lookup_table(sex_to_index)



def extract_anatom_site(example):
    tfrec_format = {
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, tfrec_format)
    anatom_site = parsed_example['anatom_site_general_challenge']
    # Decode bytes to string
    anatom_site = anatom_site.numpy().decode('utf-8')
    return anatom_site

# Initialize an empty set to store unique 'anatom_site_general_challenge' categories
anatom_site_set = set()

# Iterate through each TFRecord file
for tfrecord in input_tfrec_train:
    # Create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(tfrecord, num_parallel_reads=tf.data.AUTOTUNE)
    
    # Iterate through each serialized example in the TFRecord
    for raw_record in dataset:
        try:
            anatom_site = extract_anatom_site(raw_record)
            # Replace empty strings or specific missing indicators with 'unknown'
            if not anatom_site or anatom_site.lower() == 'nan':
                anatom_site = 'unknown'
            anatom_site_set.add(anatom_site)
        except Exception as e:
            print(f"Error parsing record: {e}")
            continue

# Add 'unknown' category to handle missing values
anatom_site_set.add('unknown')

# Convert the set to a sorted list
anatom_site_categories = sorted(anatom_site_set)
print(f"Unique 'anatom_site_general_challenge' Categories: {anatom_site_categories}")
num_anatom_sites = len(anatom_site_categories)
print(f"Number of anatom sites: {num_anatom_sites}")

# Create a mapping from 'anatom_site_general_challenge' categories to indices
anatom_site_to_index = {site: idx for idx, site in enumerate(anatom_site_categories)}
print(f"'anatom_site_general_challenge' Mapping: {anatom_site_to_index}")

num_additional_features += num_anatom_sites
print(f"Total number of additional tabular features = {num_additional_features}")

def create_anatom_site_lookup_table(anatom_site_to_index):
    # Create TensorFlow tensors for keys and values
    keys = tf.constant(list(anatom_site_to_index.keys()))
    values = tf.constant(list(anatom_site_to_index.values()), dtype=tf.int64)

    # Initialize the lookup table with a default value (index of 'unknown')
    default_value = anatom_site_to_index.get('unknown', 5)  # Default to 'unknown' if not found
    initializer = tf.lookup.KeyValueTensorInitializer(keys, values)
    table = tf.lookup.StaticHashTable(initializer, default_value)
    return table

# Create the lookup table
anatom_site_lookup_table = create_anatom_site_lookup_table(anatom_site_to_index)


def check_missing_data(input_df):
    print("=== Checking for NaNs ===")
    print(input_df.isna().sum())
    
    print("\n=== Checking numeric columns for inf ===")
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(input_df[col]).any():
            print(f"Column '{col}' has infinite values!")
    
    print("\n=== Checking 'target' column uniqueness ===")
    print("Unique target labels:", input_df["target"].unique())
    
    print("\n=== Some numeric bounds ===")
    for col in numeric_cols:
        cmin = input_df[col].min()
        cmax = input_df[col].max()
        print(f"{col} -> min: {cmin}, max: {cmax}")
    
    print("Mixed precision policy:", tf.keras.mixed_precision.global_policy()) # must be "float32"


def validate_training_dfs(train_df, validation_df):

    positives_samples = train_df['target'].value_counts()
    print(f"Training set class distribution (neg:pos): {1-positives_samples}:{positives_samples}")
    positives_samples = validation_df['target'].value_counts()
    print(f"Validation set class distribution (neg:pos): {1-positives_samples}:{positives_samples}")

    # Verify no 'patient id' leakage
    train_patient_ids = set(train_df['patient_id'])
    val_patient_ids = set(validation_df['patient_id'])
    leakage_check1 = train_patient_ids.intersection(val_patient_ids)
    assert not leakage_check1, f"Leakage check ::: 'Patient ID' leakage detected for patients: {leakage_check1}"

    # Check duplicates within validation set
    train_images = set(train_df['image_name'])
    val_images = set(validation_df['image_name'])
    leakage_check2 = train_images.intersection(val_images)
    assert not leakage_check2, f"Leakage check ::: 'Image' leakage detected for images: {leakage_check2}"

    check_missing_data(train_df)
    check_missing_data(validation_df)
    


def microscope_crop_tf(img, radius_offset=0):
    # Assume square image [H, W, 3]
    shape = tf.shape(img)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    center_x = width / 2
    center_y = height / 2
    radius = (tf.minimum(width, height) / 2) - radius_offset

    # Create meshgrid
    y = tf.range(0.0, height)
    x = tf.range(0.0, width)
    Y, X = tf.meshgrid(y, x, indexing='ij')

    dist_from_center = tf.sqrt(tf.square(X - center_x) + tf.square(Y - center_y))
    circular_mask = tf.cast(dist_from_center <= radius, tf.float32)  # shape (H, W)

    # Expand to 3 channels
    circular_mask = tf.expand_dims(circular_mask, axis=-1)
    circular_mask = tf.tile(circular_mask, [1, 1, 3])  # shape (H, W, 3)

    return img * circular_mask



def label_counter(tfrecords):
    tfrec_format = {'target': tf.io.FixedLenFeature([], tf.int64)}
    counter = {0: 0, 1: 0}

    for tfrec in tfrecords:
        raw_ds = tf.data.TFRecordDataset(tfrec)
        parsed = raw_ds.map(lambda x: tf.io.parse_single_example(x, tfrec_format))

        for example in parsed:
            label = example['target'].numpy()
            counter[label] += 1

    return counter



def parse_tfrec(example, training, num_sex_categories, num_anatom_sites, mean, std,
                global_mean_age, global_max_age, sex_lookup_table, anatom_site_lookup_table, 
                has_label=True):
    tfrec_format = {
        'target'                       : tf.io.FixedLenFeature([], tf.int64),
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'sex'                          : tf.io.FixedLenFeature([], tf.string),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.string),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.string),
    }
    
    if not has_label:
        tfrec_format.pop('target')
    
    parsed_example = tf.io.parse_single_example(example, tfrec_format)

    ####################
    ###### IMAGE #######
    ####################
    # Decode image
    image = tf.image.decode_jpeg(parsed_example['image'], channels=3)
    image = tf.image.resize(image, [IMAGE_RESIZE, IMAGE_RESIZE])
    
    # Apply data augmentation if Training set
    # Val and Test sets are training=False
    if training:
        image = data_augmentation(image)
    
    # # Apply microscope crop
    # # image = microscope_crop(image, radius_offset=5)
    image = microscope_crop_tf(image, radius_offset=5)
    
    # Normalize image
    image = tf.cast(image, tf.float32) / 255.0  # Scale pixel values to [0, 1]
    image = (image - mean) / std

    ####################
    ### TABULAR DATA ###
    ####################
    
    # Handle 'age_approx': cast to float32, impute missing values and normalize
    age_approx = tf.cast(parsed_example['age_approx'], tf.float32)
    # Check for NaN
    age_approx = tf.where(tf.math.is_nan(age_approx), 
                          tf.constant(global_mean_age, dtype=tf.float32), 
                          age_approx)
    # Normalize by max_age
    age_approx = age_approx / global_max_age
    age_approx = tf.expand_dims(age_approx, axis=-1)  # Shape: (1,)

    # Handle 'sex': impute missing with 'unknown' and map to index
    sex = parsed_example['sex']
    # Replace empty strings or 'nan' with 'unknown'
    sex = tf.cond(
        tf.logical_or(tf.equal(sex, ''), tf.equal(tf.strings.lower(sex), 'nan')),
        lambda: tf.constant('unknown'),
        lambda: sex
    )
    # Lookup the 'sex' index
    sex_index = sex_lookup_table.lookup(sex)
    sex_encoded = tf.one_hot(sex_index, depth=num_sex_categories, dtype=tf.float32)

    # Handle 'anatom_site_general_challenge': impute missing with 'unknown' and map to index
    anatom_site = parsed_example['anatom_site_general_challenge']
    # Replace empty strings or 'nan' with 'unknown'
    anatom_site = tf.cond(
        tf.logical_or(tf.equal(anatom_site, ''), tf.equal(tf.strings.lower(anatom_site), 'nan')),
        lambda: tf.constant('unknown'),
        lambda: anatom_site
    )
    # Lookup the 'anatom_site_general_challenge' index
    anatom_site_index = anatom_site_lookup_table.lookup(anatom_site)
    # One-hot encode 'anatom_site_general_challenge'
    anatom_site_encoded = tf.one_hot(tf.cast(anatom_site_index, tf.int32), depth=num_anatom_sites)
    
    
    # Concatenate additional features
    additional_features = tf.concat([sex_encoded, age_approx, anatom_site_encoded], axis=-1)
    
    if has_label:
        label = tf.cast(parsed_example['target'], tf.float32)
        return (image, additional_features), label
    else:
        return {
            'image_input': image,
            'tabular_input': additional_features,
            'image_name': parsed_example['image_name']
        }



data_augmentation = Sequential([
    RandomFlip('horizontal_and_vertical'),
    RandomRotation(0.05),          # Â±18 degrees
    RandomZoom(0.01),               # Â±1%
    RandomTranslation(0.05, 0.05),   # Â±5%
    RandomBrightness(0.15),
    RandomContrast(0.15),
    GaussianNoise(0.01)
])

class DatasetType(Enum):
    TRAINING = 'training'
    VALIDATION = 'validation'
    TEST = 'test'

def get_dataset(tfrecords, df_type, batch_size=128,
                num_sex_categories=None, num_anatom_sites=None,
                mean=MEAN, std=STD, global_mean_age=0.0, global_max_age=1.0,
                sex_lookup_table=None, anatom_site_lookup_table=None,
                has_label=True):

    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=AUTO)
    dataset = dataset.cache()

    
    if df_type == DatasetType.TRAINING:
        
        dataset = dataset.repeat()
    
        dataset = dataset.shuffle(8192)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        dataset = dataset.with_options(opt)
        
        dataset = dataset.map(
            lambda x: parse_tfrec(
                x,
                training=True,
                num_sex_categories=num_sex_categories, 
                num_anatom_sites=num_anatom_sites, 
                mean=mean, 
                std=std,
                global_mean_age=global_mean_age, 
                global_max_age=global_max_age, 
                sex_lookup_table=sex_lookup_table, 
                anatom_site_lookup_table=anatom_site_lookup_table, 
                has_label=has_label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )       
        
        dataset = dataset.batch(batch_size * REPLICAS)


    elif df_type == DatasetType.VALIDATION or df_type == DatasetType.TEST:
        
        dataset = dataset.map(
            lambda x: parse_tfrec(
                x,
                training=False,
                num_sex_categories=num_sex_categories, 
                num_anatom_sites=num_anatom_sites, 
                mean=mean, 
                std=std,
                global_mean_age=global_mean_age, 
                global_max_age=global_max_age, 
                sex_lookup_table=sex_lookup_table, 
                anatom_site_lookup_table=anatom_site_lookup_table, 
                has_label=has_label
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.batch(batch_size * REPLICAS)
        
    else:
        raise ValueError('Wrong dataset type. Abort.')
    
    return dataset.prefetch(AUTO)


class BaseModel(Enum):
    DENSENET121 = 'DenseNet121'
    EFFICIENTNETB0 = 'EfficientNetB0'

def build_model(fold_no, model_name, num_additional_features):
    # Define input layers
    image_input = layers.Input(shape=(224, 224, 3), name='image_input')
    tabular_input = layers.Input(shape=(num_additional_features,), name='tabular_input')

    # Base model
    if model_name == BaseModel.EFFICIENTNETB0:
        base_model = EfficientNetB0(weights='imagenet', include_top=False)
        print(f"Fold {fold_no} ::: Base model: {BaseModel.EFFICIENTNETB0.value}.")
    elif model_name == BaseModel.DENSENET121:
        base_model = DenseNet121(weights='imagenet', include_top=False)
        print(f"Fold {fold_no} ::: Base model: {BaseModel.DENSENET121.value}.")
    else:
        base_model = DenseNet121(weights='imagenet', include_top=False)
        print(f"Fold {fold_no} ::: Base model: default.")
    
    # Train entire base model
    for layer in base_model.layers:
        layer.trainable = True

    # Image processing
    x1 = base_model(image_input)
    x1 = layers.GlobalAveragePooling2D()(x1)
    x1 = layers.BatchNormalization()(x1)

    # Process tabular features
    x2 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(tabular_input)
    x2 = layers.BatchNormalization()(x2)
    # x2 = layers.Dropout(0.05)(x2)
    x2 = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x2)
    x2 = layers.BatchNormalization()(x2)
    # x2 = layers.Dropout(0.05)(x2)

    # Combine image and tabular features
    combined = layers.concatenate([x1, x2])
    combined = layers.BatchNormalization()(combined)

    # Add final dense layers
    x = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(combined)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.1)(x)
    output = layers.Dense(1, activation='sigmoid')(x)  # Binary classification

    # Create the model
    model = models.Model(inputs=[image_input, tabular_input], outputs=output)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    # loss = binary_focal_loss(alpha=0.5, gamma=2.25)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(name='AUC'),
            tf.keras.metrics.Precision(name='Precision'),
            tf.keras.metrics.Recall(name='Recall')]
    )
    
    return model

def get_lr_callback(batch_size):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * batch_size * REPLICAS
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

def get_callbacks(fold_no):
    early_stopping = EarlyStopping(
        monitor='val_AUC', patience=5, mode='max', restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=f'MelanomaModel_fold_{fold_no}_AUC_{{val_AUC:.5f}}.keras',
        monitor='val_AUC',
        mode='max',
        save_best_only=True,
        save_weights_only=False,  # saves full model (set True for just weights)
        verbose=1
    )

    return [early_stopping, get_lr_callback(BATCH_SIZES[fold_no]), checkpoint]
    # return [get_lr_callback(BATCH_SIZES[fold_no]), checkpoint]



#input_df.head()


def count_data_items(filenames):
    n = []
    for filename in filenames:
        match = re.search(r"-(\d+)-enhanced\.tfrec", filename)
        if match:
            n.append(int(match.group(1)))
        else:
            print(f"âš ï¸� Warning: Could not extract count from {filename}")
    return np.sum(n)



def check_for_problems(df):
    print("=== Checking for NaNs ===")
    print(df.isna().sum())
    
    print("\n=== Checking numeric columns for inf ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            print(f"Column '{col}' has infinite values!")
    
    print("\n=== Checking 'target' column uniqueness ===")
    print("Unique target labels:", df["target"].unique())
    
    print("\n=== Some numeric bounds ===")
    for col in numeric_cols:
        cmin = df[col].min()
        cmax = df[col].max()
        print(f"{col} -> min: {cmin}, max: {cmax}")
    
    print("Mixed precision policy:", tf.keras.mixed_precision.global_policy()) # must be "float32"


def display_images(dataset, mean, std, num_batches=1, images_per_batch=12):
    for (images, additional_features), labels in dataset.take(num_batches):
        num_images = min(images_per_batch, images.shape[0])

        # Calculate grid size dynamically
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)

        plt.figure(figsize=(cols * 3, rows * 3))

        for i in range(num_images):
            plt.subplot(rows, cols, i + 1)
            img = images[i].numpy() * std.numpy() + mean.numpy()
            img = np.clip(img, 0.0, 1.0)
            plt.imshow(img)
            # Display label as "Positive" or "Negative"
            label_text = "Positive" if labels[i].numpy() == 1 else "Negative"
            plt.title(label_text)
            plt.axis("off")

        # Hide any remaining subplots if the grid has more slots than images
        total_slots = rows * cols
        if total_slots > num_images:
            for i in range(num_images, total_slots):
                plt.subplot(rows, cols, i + 1)
                plt.axis("off")

        plt.tight_layout()
        plt.show()


def display_balanced_images(dataset, mean, std, num_batches=1, images_per_batch=12):
    import random  # for shuffling

    for (images, additional_features), labels in dataset.take(num_batches):
        # Convert tensors to numpy arrays
        images_np = images.numpy()
        labels_np = labels.numpy()

        # Separate positives and negatives
        positives = [(img, 1) for img, label in zip(images_np, labels_np) if label == 1]
        negatives = [(img, 0) for img, label in zip(images_np, labels_np) if label == 0]

        # Shuffle for variety
        random.shuffle(positives)
        random.shuffle(negatives)

        # Choose half from each class
        half = images_per_batch // 2
        selected = positives[:half] + negatives[:half]

        # In case we don't have enough of one class
        if len(selected) < images_per_batch:
            additional = negatives[half:] + positives[half:]
            selected += additional[:images_per_batch - len(selected)]

        # Final safety check
        selected = selected[:images_per_batch]

        # Calculate grid size dynamically
        cols = math.ceil(math.sqrt(images_per_batch))
        rows = math.ceil(images_per_batch / cols)

        plt.figure(figsize=(cols * 3, rows * 3))

        for i, (img, label) in enumerate(selected):
            plt.subplot(rows, cols, i + 1)
            # Unnormalize
            img = img * std.numpy() + mean.numpy()
            img = np.clip(img, 0.0, 1.0)
            plt.imshow(img)
            plt.title("Positive" if label == 1 else "Negative")
            plt.axis("off")

        plt.tight_layout()
        plt.show()



if RUN_TRAINING:
    all_fold_ids = np.arange(len(input_tfrec_train))
    skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
    
    # fold_no = 1
    results = []
    optimal_thresholds = []
    
    def count_examples(tfrecords):
        count = 0
        for _ in tf.data.TFRecordDataset(tfrecords):
            count += 1
        return count
    
    for fold_no, (train_index, val_index) in enumerate(skf.split(all_fold_ids)):
        print(f"::: Fold {fold_no} START :::")
        
        # Split the data
        # For each index in train_index, build the corresponding file pattern and expand it.
        train_tfrecords = []
        for idx in train_index:
            pattern = f"{INPUT_DIR}train{idx:02d}*.tfrec"
            # Expand the pattern to get the actual file names
            files = tf.io.gfile.glob(pattern)
            train_tfrecords.extend(files)
        # Similarly for validation
        validation_tfrecords = []
        for idx in val_index:
            pattern = f"{INPUT_DIR}train{idx:02d}*.tfrec"
            files = tf.io.gfile.glob(pattern)
            validation_tfrecords.extend(files)
        
        print(f"Fold {fold_no} ::: Train tfrecords: {len(train_tfrecords)}")
        print(f"Fold {fold_no} ::: Validation tfrecords: {len(validation_tfrecords)}")
    
    
        print(f"Number of examples in Train TFRecords: {count_examples(train_tfrecords)}")
        
    
        # # Compute class stats for the training split only
        num_pos, num_neg, total_count = compute_class_counts_from_tfrecords(train_tfrecords)
        label_stats = label_counter(train_tfrecords)
        print(f"âœ… Label counts from Train TFRecords: {label_stats}")
    
        # _, _, val_total_count = compute_class_counts_from_tfrecords(validation_tfrecords)
    
        # steps_per_epoch = math.ceil(total_count / BATCH_SIZES[fold_no])
        print(f"Fold {fold_no} ::: total count = {total_count}")
        
        # steps_per_epoch = math.ceil(total_count / (BATCH_SIZES[fold_no] * REPLICAS))
        steps_per_epoch=int(count_data_items(train_tfrecords)/BATCH_SIZES[fold_no]//REPLICAS)

        
        print(f"Fold {fold_no} ::: steps_per_epoch = {steps_per_epoch}")
        # validation_steps = math.ceil(val_total_count / BATCH_SIZES[fold_no])
    
        zeros_weight = total_count / (2.0 * num_neg)
        ones_weight  = total_count / (2.0 * num_pos)
        class_weight_dict = {0: zeros_weight, 1: ones_weight}
        print(f"Fold {fold_no} ::: class weights: {class_weight_dict}")

        train_dataset = get_dataset(
            tfrecords=train_tfrecords,
            df_type=DatasetType.TRAINING,
            batch_size=BATCH_SIZES[fold_no],
            num_sex_categories=num_sex_categories,
            num_anatom_sites=num_anatom_sites,
            mean=MEAN,
            std=STD,
            global_mean_age=global_mean_age,
            global_max_age=global_max_age,
            sex_lookup_table=sex_lookup_table,
            anatom_site_lookup_table=anatom_site_lookup_table,
            has_label=True
        )
        print(f"Fold {fold_no} ::: Train dataset created successfully.")
        
        print(f"Fold {fold_no} ::: Sample of Training images:")
        # display_images(train_dataset, mean=MEAN, std=STD, num_batches=1, images_per_batch=12)
        display_balanced_images(train_dataset, mean=MEAN, std=STD, num_batches=1, images_per_batch=12)
        
        # for (img_batch, meta_batch), labels in train_dataset.take(1):
        #     print("âœ… image batch shape:", img_batch.shape)
        #     print("âœ… metadata shape:", meta_batch.shape)
        #     print("âœ… labels shape:", labels.shape)
    
        
        validation_dataset = get_dataset(
            tfrecords=validation_tfrecords,
            df_type=DatasetType.VALIDATION,
            batch_size=BATCH_SIZES[fold_no],
            num_sex_categories=num_sex_categories,
            num_anatom_sites=num_anatom_sites,
            mean=MEAN,
            std=STD,
            global_mean_age=global_mean_age,
            global_max_age=global_max_age,
            sex_lookup_table=sex_lookup_table,
            anatom_site_lookup_table=anatom_site_lookup_table,
            has_label=True
        )
        print(f"Fold {fold_no} ::: Validation dataset created successfully.")
        print(f"Fold {fold_no} ::: Sample of Validation images:")
    
        # display_images(validation_dataset, mean=MEAN, std=STD, num_batches=1, images_per_batch=12)
        display_balanced_images(validation_dataset, mean=MEAN, std=STD, num_batches=1, images_per_batch=12)
    
        K.clear_session()
        
        print(f"::: Fold {fold_no} ::: TRAINING START")
        with strategy.scope():
            model = build_model(fold_no, BaseModel.DENSENET121, num_additional_features)
            history = model.fit(
                train_dataset,
                epochs=EPOCHS[fold_no],
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_dataset,
                callbacks=get_callbacks(fold_no),
                verbose=2,
                class_weight=class_weight_dict
            )
        
        # Evaluate the model
        print(f"Fold {fold_no} ::: Starting model evaluation against validation dataset.")
        true_labels = []
        pred_probs = []
    
        for batch in validation_dataset:
            (images, tabular_inputs), labels = batch
            preds = model.predict((images, tabular_inputs), verbose=0)
            pred_probs.extend(preds.flatten())
            true_labels.extend(labels.numpy())
    
        true_labels = np.array(true_labels)
        pred_probs = np.array(pred_probs)
    
        # Compute AUC (threshold-independent)
        auc_value = roc_auc_score(true_labels, pred_probs)
        
        # ------------------------------------------
        # Evaluate multiple predefined thresholds
        # ------------------------------------------
        
        min_val = 0.20
        max_val = 0.80
        step_val = 0.05
        thresholds_to_eval = [round(x, 2) for x in np.arange(min_val, max_val + step_val, step_val)]
    
        results_table = []
        
        for t in thresholds_to_eval:
            predicted_labels_t = (pred_probs >= t).astype(int)
        
            # Compute confusion matrix for threshold t
            conf = confusion_matrix(true_labels, predicted_labels_t)
            tn, fp, fn, tp = conf.ravel()
        
            # Calculate precision and recall
            prec_t = precision_score(true_labels, predicted_labels_t, zero_division=0)
            rec_t = recall_score(true_labels, predicted_labels_t, zero_division=0)
    
            f1 = f1_score(true_labels, predicted_labels_t, zero_division=0)
            
            # Collect results in a dictionary
            results_table.append({
                'Threshold': t,
                'AUC': f"{auc_value:.4f}",   # same AUC for all thresholds, it's threshold-independent
                'Precision': f"{prec_t:.4f}",
                'TNs': tn,
                'FPs': fp,
                'Recall': f"{rec_t:.4f}",
                'TPs': tp,
                'FNs': fn,
                'F1': f"{f1:.4f}",
            })
        
        # Convert to a DataFrame for a neat table
        df_results = pd.DataFrame(results_table)
        print("Evaluation at Fixed Thresholds:")
        print(df_results.to_string(index=False))
        
        # Compute ROC curve
        fpr, tpr, thresholds_roc = roc_curve(true_labels, pred_probs)
    
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='r', label=f'ROC curve (AUC = {auc_value:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line for reference
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.tight_layout()  
        plt.show()
    
    
        f1_scores = []
        for t in thresholds_to_eval:
            predicted_labels_t = (pred_probs >= t).astype(int)
            f1 = f1_score(true_labels, predicted_labels_t, zero_division=0)
            f1_scores.append(f1)
        
        # Find the threshold that gives the highest F1
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds_to_eval[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"\nâœ… Best Threshold by F1 Score: {best_threshold}")
        print(f"ğŸ§® Best F1 Score: {best_f1:.4f}")
    
        best_preds = (pred_probs >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, best_preds).ravel()
        print(f"Confusion matrix at best F1 threshold ({best_threshold}):")
        print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
        prec, rec, _ = precision_recall_curve(true_labels, pred_probs)
        pr_auc = auc(rec, prec)
        
        print(f"ğŸ“ˆ PR AUC: {pr_auc:.4f}")
        
        # Plot PR Curve
        plt.figure(figsize=(6, 6))
        plt.plot(rec, prec, color='green', label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
        
        print(f"::: Fold {fold_no} COMPLETE :::")
        print(f"::: ::::::::::::::::::::::: :::")



# Create the test dataset
test_batch_size = BATCH_SIZES[0]

test_dataset = get_dataset(
    tfrecords=input_tfrec_test,
    df_type=DatasetType.TEST,
    batch_size=test_batch_size,
    num_sex_categories=num_sex_categories,
    num_anatom_sites=num_anatom_sites,
    mean=MEAN,
    std=STD,
    global_mean_age=global_mean_age,
    global_max_age=global_max_age,
    sex_lookup_table=sex_lookup_table,
    anatom_site_lookup_table=anatom_site_lookup_table,
    has_label=False
)

print(f"Test dataset created successfully.")

# for batch in test_dataset.take(1):
#     print(batch['image_name'])  # should print a list of string tensors

# print(f"Fold {fold_no} ::: Sample of Validation images:")
# display_balanced_images(validation_dataset, mean=MEAN, std=STD, num_batches=1, images_per_batch=12)

# for (image_batch, additional_features_batch) in test_dataset.take(1):
#     print(f"Image batch shape: {image_batch.shape}")  # Expected: (batch_size, 224, 224, 3)
#     print(f"Additional features batch shape: {additional_features_batch.shape}")  # Expected: (batch_size, num_additional_features)

# Paths to the saved models
model_paths = [
    'MelanomaModel_fold_0_AUC_0.83144.keras',
    'MelanomaModel_fold_1_AUC_0.89147.keras',
    'MelanomaModel_fold_2_AUC_0.92443.keras'
]

# Fold 0
# Evaluation at Fixed Thresholds:
#  Threshold    AUC Precision   TNs   FPs Recall  TPs  FNs     F1
#       0.20 0.8244    0.0184     0 12197 1.0000  229    0 0.0362
#       0.25 0.8244    0.0185    23 12174 1.0000  229    0 0.0363
#       0.30 0.8244    0.0186   140 12057 1.0000  229    0 0.0366
#       0.35 0.8244    0.0193   558 11639 1.0000  229    0 0.0379
#       0.40 0.8244    0.0202  1358 10839 0.9738  223    6 0.0395
#       0.45 0.8244    0.0226  2719  9478 0.9563  219   10 0.0441
#       0.50 0.8244    0.0277  4836  7361 0.9170  210   19 0.0538
#       0.55 0.8244    0.0382  7159  5038 0.8734  200   29 0.0732
#       0.60 0.8244    0.0505  8734  3463 0.8035  184   45 0.0949
#       0.65 0.8244    0.0634  9804  2393 0.7074  162   67 0.1164
#       0.70 0.8244    0.0804 10596  1601 0.6114  140   89 0.1421
#       0.75 0.8244    0.1032 11163  1034 0.5197  119  110 0.1722
#       0.80 0.8244    0.1266 11583   614 0.3886   89  140 0.1910
#       0.85 0.8244    0.1683 11856   341 0.3013   69  160 0.2160

# Fold 1
# Evaluation at Fixed Thresholds:
#  Threshold    AUC Precision  TNs  FPs Recall  TPs  FNs     F1
#       0.20 0.8432    0.0225 2466 7702 1.0000  177    0 0.0439
#       0.25 0.8432    0.0248 3213 6955 1.0000  177    0 0.0484
#       0.30 0.8432    0.0265 3714 6454 0.9944  176    1 0.0517
#       0.35 0.8432    0.0280 4129 6039 0.9831  174    3 0.0545
#       0.40 0.8432    0.0295 4484 5684 0.9774  173    4 0.0573
#       0.45 0.8432    0.0306 4880 5288 0.9435  167   10 0.0593
#       0.50 0.8432    0.0328 5306 4862 0.9322  165   12 0.0634
#       0.55 0.8432    0.0353 5767 4401 0.9096  161   16 0.0679
#       0.60 0.8432    0.0381 6209 3959 0.8870  157   20 0.0731
#       0.65 0.8432    0.0426 6685 3483 0.8757  155   22 0.0813
#       0.70 0.8432    0.0478 7198 2970 0.8418  149   28 0.0904
#       0.75 0.8432    0.0544 7717 2451 0.7966  141   36 0.1018
#       0.80 0.8432    0.0564 8227 1941 0.6554  116   61 0.1038
#       0.85 0.8432    0.0677 8804 1364 0.5593   99   78 0.1207

# Fold 2
# Evaluation at Fixed Thresholds:
#  Threshold    AUC Precision  TNs  FPs Recall  TPs  FNs     F1
#       0.20 0.8970    0.0180  472 9705 1.0000  178    0 0.0354
#       0.25 0.8970    0.0231 2698 7479 0.9944  177    1 0.0452
#       0.30 0.8970    0.0314 4808 5369 0.9775  174    4 0.0608
#       0.35 0.8970    0.0380 5820 4357 0.9663  172    6 0.0731
#       0.40 0.8970    0.0436 6489 3688 0.9438  168   10 0.0833
#       0.45 0.8970    0.0488 6978 3199 0.9213  164   14 0.0926
#       0.50 0.8970    0.0548 7415 2762 0.8989  160   18 0.1032
#       0.55 0.8970    0.0603 7762 2415 0.8708  155   23 0.1128
#       0.60 0.8970    0.0676 8095 2082 0.8483  151   27 0.1253
#       0.65 0.8970    0.0775 8438 1739 0.8202  146   32 0.1415
#       0.70 0.8970    0.0885 8756 1421 0.7753  138   40 0.1589
#       0.75 0.8970    0.1046 9090 1087 0.7135  127   51 0.1825
#       0.80 0.8970    0.1211 9357  820 0.6348  113   65 0.2034
#       0.85 0.8970    0.1431 9644  533 0.5000   89   89 0.2225


optimal_thresholds = [0.60 ,0.60 , 0.55]

models = [
    load_model(path)  # No need for custom_objects
    for path in model_paths
]

# Step 1: Predict
predictions = []
for idx, model in enumerate(models):
    print(f"Making predictions with Model {idx}...")
    preds = model.predict(test_dataset, verbose=2)
    predictions.append(preds.flatten())

predictions = np.array(predictions)
avg_predictions = predictions.mean(axis=0)
averaged_threshold = np.mean(optimal_thresholds)
final_predictions = (avg_predictions >= averaged_threshold).astype(int)


# Step 2: Extract image names from test_dataset
image_names = []

for batch in test_dataset:
    batch_image_names = batch['image_name']
    decoded_names = [name.numpy().decode('utf-8') for name in batch_image_names]
    image_names.extend(decoded_names)

# Step 3: Create the submission DataFrame
submission_df = pd.DataFrame({
    'image_name': image_names,
    'target': final_predictions
})

# Step 4: Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("âœ… submission.csv created.")

