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



!pip install pydicom
!pip install matplotlib seaborn scikit-learn




import cv2
import os
import time
import json
import glob
import random
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom as dicom
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras import backend as K
from pydicom.pixel_data_handlers.util import apply_voi_lut




# Define the path to the training data
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

# Load CSV files containing metadata and labels
train = pd.read_csv(os.path.join(train_path, 'train.csv'))
label = pd.read_csv(os.path.join(train_path, 'train_label_coordinates.csv'))
train_desc = pd.read_csv(os.path.join(train_path, 'train_series_descriptions.csv'))
test_desc = pd.read_csv(os.path.join(train_path, 'test_series_descriptions.csv'))
sub = pd.read_csv(os.path.join(train_path, 'sample_submission.csv'))




# # Display the first few rows of each dataframe
# print("Test Descriptions:")
# print(test_desc.head(5))

# print("\nTrain Data:")
# print(train.head(5))

# print("\nTrain Series Descriptions:")
# print(train_desc.head(5))

import random

# Generate Image Paths
# ---------------------
def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        # Sort images by filename to maintain DICOM sequence
        images = sorted(os.listdir(series_dir))
        image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths

# Generate image paths for training and testing datasets
train_image_paths = generate_image_paths(train_desc, os.path.join(train_path, 'train_images'))
test_image_paths = generate_image_paths(test_desc, os.path.join(train_path, 'test_images'))

# Example usage
print("\nSample Train Image Path:", train_image_paths[2])
print("Number of Train Descriptions:", len(train_desc))
print("Number of Train Image Paths:", len(train_image_paths))

# Display DICOM Images
# ---------------------
import pydicom
def display_dicom_images(image_paths, num_images=3):
    """
    Display a specified number of DICOM images.

    Parameters:
        image_paths (List[str]): List of DICOM image file paths.
        num_images (int): Number of images to display.
    """
    plt.figure(figsize=(15, 5))
    for i, path in enumerate(image_paths[:num_images]):
        ds = pydicom.dcmread(path)
        plt.subplot(1, num_images, i+1)
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
        plt.title(f"Image {i+1}")
        plt.axis('off')
    plt.show()

# Display the first three DICOM images from training data
display_dicom_images(train_image_paths)

# Display DICOM Images with Coordinates
# -------------------------------------
def display_dicom_with_coordinates(image_paths, label_df):
    """
    Display DICOM images with annotated coordinates.

    Parameters:
        image_paths (List[str]): List of DICOM image file paths.
        label_df (DataFrame): DataFrame containing label coordinates.
    """
    fig, axs = plt.subplots(1, len(image_paths), figsize=(18, 6))

    for idx, path in enumerate(image_paths):
        study_id = int(path.split('/')[-3])
        series_id = int(path.split('/')[-2])

        # Filter labels for the current study and series
        filtered_labels = label_df[
            (label_df['study_id'] == study_id) &
            (label_df['series_id'] == series_id)
        ]

        # Read DICOM image
        ds = pydicom.dcmread(path)

        # Plot DICOM image
        axs[idx].imshow(ds.pixel_array, cmap='gray')
        axs[idx].set_title(f"Study ID: {study_id}, Series ID: {series_id}")
        axs[idx].axis('off')

        # Plot coordinates
        for _, row in filtered_labels.iterrows():
            axs[idx].plot(row['x'], row['y'], 'ro', markersize=5)

    plt.tight_layout()
    plt.show()

def load_dicom_files(path_to_folder):
    """
    Load and sort DICOM files from a specified folder.

    Parameters:
        path_to_folder (str): Directory containing DICOM files.

    Returns:
        List[str]: Sorted list of DICOM file paths.
    """
    files = [os.path.join(path_to_folder, f) for f in os.listdir(path_to_folder) if f.endswith('.dcm')]
    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('-')[-1]))
    return files

# Example: Display DICOM images with coordinates for a specific study
study_id = "100206310"
study_folder = os.path.join(train_path, 'train_images', study_id)

image_paths = []
for series_folder in os.listdir(study_folder):
    series_folder_path = os.path.join(study_folder, series_folder)
    dicom_files = load_dicom_files(series_folder_path)
    if dicom_files:
        image_paths.append(dicom_files[0])  # Add the first image from each series

display_dicom_with_coordinates(image_paths, label)

# Data Reshaping and Merging
# --------------------------
def reshape_row(row):
    """
    Reshape a single row of the DataFrame to separate conditions, levels, and severities.

    Parameters:
        row (Series): A row from the DataFrame.

    Returns:
        DataFrame: Reshaped DataFrame.
    """
    data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}

    for column, value in row.items():
        if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
            parts = column.split('_')
            condition = ' '.join([word.capitalize() for word in parts[:-2]])
            level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
            data['study_id'].append(row['study_id'])
            data['condition'].append(condition)
            data['level'].append(level)
            data['severity'].append(value)

    return pd.DataFrame(data)

# Reshape the training DataFrame
new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)

# Display the first few rows of the reshaped DataFrame
print("\nReshaped Train Data:")
print(new_train_df.head(5))

# Print columns for verification
print("\nColumns in new_train_df:")
print(", ".join(new_train_df.columns))

print("\nColumns in label:")
print(", ".join(label.columns))

print("\nColumns in test_desc:")
print(", ".join(test_desc.columns))

print("\nColumns in sub:")
print(", ".join(sub.columns))

# Merge DataFrames
# -----------------
# Merge reshaped training data with label coordinates
merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')

# Further merge with training series descriptions
final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')

# Display the first few rows of the final merged DataFrame
print("\nFinal Merged Data:")
print(final_merged_df.head(5))

# Example Queries
print("\nEntries for Study ID 100206310:")
print(final_merged_df[final_merged_df['study_id'] == 100206310].sort_values(['x','y'], ascending=True))

print("\nEntries for Series ID 1012284084:")
print(final_merged_df[final_merged_df['series_id'] == 1012284084].sort_values("instance_number"))

# Create the 'row_id' column by combining study_id, condition, and level
final_merged_df['row_id'] = (
    final_merged_df['study_id'].astype(str) + '_' +
    final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
    final_merged_df['level'].str.lower().str.replace('/', '_')
)

# Create the 'image_path' column based on directory structure
final_merged_df['image_path'] = (
    os.path.join(train_path, 'train_images') + '/' +
    final_merged_df['study_id'].astype(str) + '/' +
    final_merged_df['series_id'].astype(str) + '/' +
    final_merged_df['instance_number'].astype(str) + '.dcm'
)

# Display the updated DataFrame
print("\nUpdated Final Merged DataFrame:")
print(final_merged_df.head(5))

normal_mild_count = final_merged_df[final_merged_df["severity"] == "Normal/Mild"].shape[0]
moderate_count = final_merged_df[final_merged_df["severity"] == "Moderate"].shape[0]
severe_count = final_merged_df[final_merged_df["severity"] == "Severe"].shape[0]

print(f"\nNormal/Mild Count: {normal_mild_count}")
print(f"Moderate Count: {moderate_count}")
print(f"Severe Count: {severe_count}")

base_path = '/content/test_images/'

def get_image_paths(row):
    series_path = os.path.join(base_path, str(row['study_id']), str(row['series_id']))
    if os.path.exists(series_path):
        return [
            os.path.join(series_path, f)
            for f in os.listdir(series_path)
            if os.path.isfile(os.path.join(series_path, f))
        ]
    return []

# Mapping of series_description to conditions
condition_mapping = {
    'Sagittal T1': {
        'left': 'left_neural_foraminal_narrowing',
        'right': 'right_neural_foraminal_narrowing'
    },
    'Axial T2': {
        'left': 'left_subarticular_stenosis',
        'right': 'right_subarticular_stenosis'
    },
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

# Expand the test descriptions by adding new rows for each image path and condition
expanded_rows = []

for index, row in test_desc.iterrows():
    image_paths = get_image_paths(row)
    conditions = condition_mapping.get(row['series_description'], {})

    # Handle single or multiple conditions
    if isinstance(conditions, str):
        conditions = {'left': conditions, 'right': conditions}

    for side, condition in conditions.items():
        for image_path in image_paths:
            expanded_rows.append({
                'study_id': row['study_id'],
                'series_id': row['series_id'],
                'series_description': row['series_description'],
                'image_path': image_path,
                'condition': condition,
                'row_id': f"{row['study_id']}_{condition}"
            })

# Create a new DataFrame from the expanded rows
expanded_test_desc = pd.DataFrame(expanded_rows)

# Display the first few rows of the expanded test descriptions
print("\nExpanded Test Descriptions:")
print(expanded_test_desc.head(5))

# Update Severity Labels
# -----------------------
# Map severity labels to simplified categories
final_merged_df['severity'] = final_merged_df['severity'].map({
    'Normal/Mild': 'normal_mild',
    'Moderate': 'moderate',
    'Severe': 'severe'
})

# Assign train and test data
train_data = final_merged_df
test_data = expanded_test_desc

# Display sample data
print("\nSample Train Data:")
print(train_data.head(10))

print("\nSample Test Data:")
print(test_data.head(10))

# Display the shape of the training data
print("\nTrain Data Shape:", train_data.shape)

# Verify File Paths
# -----------------
def check_exists(path):
    return os.path.exists(path)

def check_study_id(row):
    study_id = row['study_id']
    path = os.path.join(train_path, 'train_images', str(study_id))
    return check_exists(path)

def check_series_id(row):
    study_id = row['study_id']
    series_id = row['series_id']
    path = os.path.join(train_path, 'train_images', str(study_id), str(series_id))
    return check_exists(path)

def check_image_exists(row):
    image_path = row['image_path']
    return check_exists(image_path)

# Apply existence checks to the training data
train_data['study_id_exists'] = train_data.apply(check_study_id, axis=1)
train_data['series_id_exists'] = train_data.apply(check_series_id, axis=1)
train_data['image_exists'] = train_data.apply(check_image_exists, axis=1)

# Filter training data to include only existing paths
train_data = train_data[
    train_data['study_id_exists'] &
    train_data['series_id_exists'] &
    train_data['image_exists']
]
print("\nTrain Data Shape after Filtering:", train_data.shape)

# Load and Display Sample Images
# ------------------------------
def load_dicom(path):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)  # Normalize the image
    return (data * 255).astype(np.uint8)

images = []
row_ids = []
selected_indices = random.sample(range(len(train_data)), 2)
for i in selected_indices:
    image = load_dicom(train_data.iloc[i]['image_path'])
    images.append(image)
    row_ids.append(train_data.iloc[i]['row_id'])

# Plot the selected images
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for i in range(2):
    ax[i].imshow(images[i], cmap='gray')
    ax[i].set_title(f'Row ID: {row_ids[i]}', fontsize=8)
    ax[i].axis('off')
plt.tight_layout()
plt.show()

# Remove any rows with missing values
train_data = train_data.dropna()

# Define Visualization Functions
# -------------------------------
def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()

import cv2

# Data Loading and Preprocessing Functions
# ----------------------------------------
def load_dicom_tf(path):
    dicom = pydicom.dcmread(path.numpy().decode('utf-8'))
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    return data.astype(np.float32)


#2
import tensorflow as tf
import cv2
import pydicom
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import applications, layers, optimizers, callbacks

from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras import layers, models
# from tensorflow.keras import applications, layers, optimizers, callbacks
# from sklearn.utils.class_weight import compute_class_weight


#3
class_counts = [37626, 7950, 3081]
total_samples = sum(class_counts)
class_weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=np.repeat([0,1,2], class_counts))
class_weights = dict(enumerate(class_weights))
# class_weights1 = list(class_weights)

def focal_loss(y_true, y_pred, alpha=list(class_weights.values()), gamma=2.0):
    y_true = tf.cast(y_true, tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    probs = tf.nn.softmax(y_pred, axis=-1)
    probs = tf.gather(probs, y_true, batch_dims=1)
    alpha = tf.gather(alpha, y_true)
    modulating_factor = tf.pow(1.0 - probs, gamma)
    return tf.reduce_mean(alpha * modulating_factor * ce)

def apply_clahe(image, clipLimit=2, tileGridSize=(16, 16)):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if np.issubdtype(image.dtype, np.floating) else image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)

def apply_bilateral_filter(image, diameter=5, sigma_color=10, sigma_space=10):
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            sigma_color = sigma_color / 255.0
        else:
            image = (image / 255.0).astype(np.float32)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

import tensorflow as tf

def augment_image_normalized(image):
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=5/255.0)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image_uint8 = tf.image.convert_image_dtype(image, tf.uint8)
    image_uint8 = tf.image.random_jpeg_quality(image_uint8, 80, 100)
    image = tf.image.convert_image_dtype(image_uint8, tf.float32)

    return image

def remove_background(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.dtype == np.float32 else image.astype(np.uint8)
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_and(image, image, mask=mask)

def preprocess_image(image, label=None, is_training=False):
    def opencv_process(img_tensor):
        img_np = img_tensor.numpy().squeeze(axis=-1)  # Convert to 2D (H, W)
        img_np = apply_bilateral_filter(img_np)
        img_np = apply_clahe(img_np)
        img_np = remove_background(img_np)
        if is_training:
           img_np = augment_image_normalized(tf.expand_dims(img_np, axis=-1)).numpy().squeeze(axis=-1)
        img_np = np.expand_dims(img_np, axis=-1)  # Add channel back (H, W, 1)
        return img_np.astype(np.float32)

    image = tf.expand_dims(image, axis=-1)
    image = tf.py_function(opencv_process, [image], tf.float32)
    image.set_shape([None, None, 1])  # Set dynamic shape
    image = tf.image.resize(image, [224, 224])
    image = tf.image.grayscale_to_rgb(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return (image, label) if label is not None else image

def create_dataset(df, batch_size=64, is_test=False, is_training=False):
    if is_test:
        def load_wrapper(path):
            image = tf.py_function(load_dicom_tf, [path], tf.float32)
            image.set_shape((None, None))
            return preprocess_image(image)

        dataset = tf.data.Dataset.from_tensor_slices(df['image_path'])
        dataset = dataset.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def load_wrapper(path, label):
            # Load DICOM and ensure shape
            image = tf.py_function(load_dicom_tf, [path], tf.float32)
            image.set_shape([None, None])  # Shape [H, W]
            return preprocess_image(image, label, is_training=True)

        labels = df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}).astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((df['image_path'], labels))
        dataset = dataset.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

        if is_training:
            dataset = dataset.apply(tf.data.experimental.rejection_resample(
                class_func=lambda image, label: label,
                target_dist=list(class_weights.values()),
                initial_dist=list(class_weights.values())
            )).map(lambda resampled_label, original_sample: original_sample)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

#copilot

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')


# Custom weighted loss function
def weighted_log_loss(y_true, y_pred):
    weights = tf.gather([1.0, 2.0, 4.0], tf.cast(y_true, tf.int32))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss * weights)


def build_efficientnetb1(num_classes=3):
    base_model = applications.EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    # فقط بلاک‌های آخر را آموزش‌پذیر بگذار
    for layer in base_model.layers:
        if 'block6' in layer.name or 'block7' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)  # لایه‌ی واسط اختیاری
    outputs = layers.Dense(num_classes, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
# 

# Focal_loss=weighted_log_loss

# Train model
def train_model(model, train_dataset, val_dataset, series_name):
    model.compile(
        optimizer=optimizers.Adam(0.0001),
        loss=focal_loss,
        metrics=['accuracy']
    )

    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(f'best_{series_name}.keras', save_best_only=True)
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        callbacks=callbacks_list
    )

    return history

# Evaluate model
def evaluate_model(model, dataset):
    y_true = []
    y_pred = []

    for batch in dataset:
        images, labels = batch[0], batch[1]
        y_true.extend(labels.numpy())
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_indices = ~np.isnan(y_true)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    return {
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cm': confusion_matrix(y_true, y_pred)
    }


from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


# Initialize models for each series
series_models = {
    'Sagittal T1': build_efficientnetb1(),
    'Axial T2': build_efficientnetb1(),
    'Sagittal T2/STIR': build_efficientnetb1()
}

class_names = ['normal_mild', 'moderate', 'severe']
results = {}

# # Ensure final_merged_df is defined correctly
# if 'final_merged_df' not in globals():
#     raise NameError("final_merged_df is not defined. Please make sure your dataset is loaded correctly.")

for series_name, model in series_models.items():
    series_df = final_merged_df[
        (final_merged_df['series_description'] == series_name) &
        (final_merged_df['severity'].isin(class_names))
    ].copy()

    if series_df.empty:
        print(f"Skipping {series_name} - no valid data.")
        continue

    # Split dataset
    train_df, temp_df = train_test_split(
        series_df,
        test_size=0.3,
        stratify=series_df['severity'],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.6667,
        stratify=temp_df['severity'],
        random_state=42
    )

    # Display class distribution
    print(f"\nClass distribution for {series_name}:")
    print("Train:", train_df['severity'].value_counts())
    print("Validation:", val_df['severity'].value_counts())
    print("Test:", test_df['severity'].value_counts())

    train_ds = create_dataset(train_df, batch_size=64,is_test=False, is_training=True)
    val_ds = create_dataset(val_df, batch_size=64 ,is_test=False,)
    test_ds = create_dataset(test_df, batch_size=64,is_test=False,)

    # Train model
    history = train_model(model, train_ds, val_ds, series_name)

    # Evaluate model on TEST set
    result = evaluate_model(model, test_ds)
    results[series_name] = result

    plot_confusion_matrix(result['cm'], class_names, f'{series_name} Confusion Matrix')

    print(f"\nMetrics for {series_name}:")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-Score: {result['f1']:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{series_name} Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{series_name} Accuracy')
    plt.legend()

    plt.show()



import os
import numpy as np
import pandas as pd
import cv2
import pydicom
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB1
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


# Set up mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# Define dataset path
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

# Load CSV files
train = pd.read_csv(os.path.join(train_path, 'train.csv'))
label = pd.read_csv(os.path.join(train_path, 'train_label_coordinates.csv'))
train_desc = pd.read_csv(os.path.join(train_path, 'train_series_descriptions.csv'))

# Reshape train.csv
def reshape_row(row):
    data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
    for column, value in row.items():
        if column != 'study_id':
            parts = column.split('_')
            condition = ' '.join([word.capitalize() for word in parts[:-2]])
            level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
            data['study_id'].append(row['study_id'])
            data['condition'].append(condition)
            data['level'].append(level)
            data['severity'].append(value)
    return pd.DataFrame(data)

# Create and merge DataFrames
new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)
merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')

# Create image paths
final_merged_df['image_path'] = (
    train_path + 'train_images/' +
    final_merged_df['study_id'].astype(str) + '/' +
    final_merged_df['series_id'].astype(str) + '/' +
    final_merged_df['instance_number'].astype(str) + '.dcm'
)

# Map severity labels
final_merged_df['severity'] = final_merged_df['severity'].map({
    'Normal/Mild': 'normal_mild',
    'Moderate': 'moderate',
    'Severe': 'severe'
})

# Filter invalid rows
final_merged_df = final_merged_df[final_merged_df['severity'].isin(['normal_mild', 'moderate', 'severe'])]

# Compute class weights
class_counts = final_merged_df['severity'].value_counts().sort_index().values
class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), 
                                    y=final_merged_df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}))
class_weights_dict = dict(enumerate(class_weights))

# Define Focal Loss
def focal_loss(y_true, y_pred, alpha=list(class_weights_dict.values()), gamma=2.0):
    y_true = tf.cast(y_true, tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    probs = tf.nn.softmax(y_pred, axis=-1)
    probs = tf.gather(probs, y_true, batch_dims=1)
    alpha = tf.gather(alpha, y_true)
    modulating_factor = tf.pow(1.0 - probs, gamma)
    return tf.reduce_mean(alpha * modulating_factor * ce)

# Image preprocessing functions
def apply_clahe(image, clip_limit=2, tile_grid_size=(16, 16)):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if np.issubdtype(image.dtype, np.floating) else image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def apply_bilateral_filter(image, diameter=5, sigma_color=10, sigma_space=10):
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            sigma_color = sigma_color / 255.0
        else:
            image = (image / 255.0).astype(np.float32)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def augment_image(image):
    image = tf.image.random_brightness(image, max_delta=5/255.0)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image_uint8 = tf.image.convert_image_dtype(image, tf.uint8)
    image_uint8 = tf.image.random_jpeg_quality(image_uint8, 80, 100)
    return tf.image.convert_image_dtype(image_uint8, tf.float32)

def remove_background(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.dtype == np.float32 else image.astype(np.uint8)
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_and(image, image, mask=mask)

def load_dicom_tf(path):
    dicom = pydicom.dcmread(path.numpy().decode('utf-8'))
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    return data.astype(np.float32)

def preprocess_image(image, label=None, is_training=False):
    def process_with_opencv(img_tensor):
        img_np = img_tensor.numpy().squeeze()
        img_np = apply_bilateral_filter(img_np)
        img_np = apply_clahe(img_np)
        img_np = remove_background(img_np)
        if is_training:
            img_np = augment_image(tf.expand_dims(img_np, axis=-1)).numpy().squeeze()
        img_np = np.expand_dims(img_np, axis=-1)
        return img_np.astype(np.float32)

    image = tf.py_function(load_dicom_tf, [image], tf.float32)
    image.set_shape([None, None])
    image = tf.py_function(process_with_opencv, [image], tf.float32)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [224, 224])
    image = tf.image.grayscale_to_rgb(image)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return (image, label) if label is not None else image

# Create dataset
def create_dataset(df, batch_size=32, is_training=False):
    labels = df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}).astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'], labels))
    dataset = dataset.map(lambda x, y: preprocess_image(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.apply(tf.data.experimental.rejection_resample(
            class_func=lambda image, label: label,
            target_dist=list(class_weights_dict.values()),
            initial_dist=list(class_weights_dict.values())
        )).map(lambda resampled_label, original_sample: original_sample)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Build U-Net encoder
def build_unet_encoder(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    c5 = layers.Conv2D(1024, 3, padding='same', activation='relu')(p4)
    c5 = layers.Conv2D(1024, 3, padding='same', activation='relu')(c5)
    
    return models.Model(inputs, c5)

# Build hybrid U-Net + EfficientNetB1 model
def build_hybrid_model(input_shape=(224, 224, 3), num_classes=3):
    # U-Net encoder
    unet_encoder = build_unet_encoder(input_shape)
    
    # EfficientNetB1
    efficientnet = EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze most EfficientNet layers
    for layer in efficientnet.layers:
        if 'block6' in layer.name or 'block7' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    
    # Get features from U-Net encoder
    unet_features = unet_encoder(inputs)
    
    # Get features from EfficientNetB1
    eff_features = efficientnet(inputs)
    
    # Combine features (concatenate after global average pooling)
    unet_pooled = layers.GlobalAveragePooling2D()(unet_features)
    eff_pooled = layers.GlobalAveragePooling2D()(eff_features)
    combined = layers.Concatenate()([unet_pooled, eff_pooled])
    
    # Dense layers for classification
    x = layers.Dense(512, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='linear')(x)
    
    return models.Model(inputs, outputs)

# Train model
def train_model(model, train_ds, val_ds, series_name):
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
                  loss=focal_loss, 
                  metrics=['accuracy'])
    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(f'best_{series_name}_hybrid.keras', save_best_only=True)
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=callbacks_list)
    return history

# Evaluate model
def evaluate_model(model, test_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    cm_details = {}
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        cm_details[class_name] = {
            'True Positives': int(tp),
            'False Positives': int(fp),
            'False Negatives': int(fn),
            'True Negatives': int(tn)
        }
    
    return {
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cm': cm,
        'cm_details': cm_details
    }

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    file_path = f'{title.lower().replace(" ", "_")}.png'
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(file_path)
    plt.close()




# Main loop for each series
series_types = ['Sagittal T1', 'Axial T2', 'Sagittal T2/STIR']
class_names = ['normal_mild', 'moderate', 'severe']

for series_name in series_types:
    series_df = final_merged_df[final_merged_df['series_description'] == series_name].copy()
    if series_df.empty:
        print(f"No valid data for {series_name}.")
        continue
    
    # Split data
    train_df, temp_df = train_test_split(series_df, test_size=0.3, stratify=series_df['severity'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.6667, stratify=temp_df['severity'], random_state=42)
    
    print(f"\nClass distribution for {series_name}:")
    print("Train:", train_df['severity'].value_counts().to_dict())
    print("Validation:", val_df['severity'].value_counts().to_dict())
    print("Test:", test_df['severity'].value_counts().to_dict())
    
    # Create datasets
    train_ds = create_dataset(train_df, batch_size=32, is_training=True)
    val_ds = create_dataset(val_df, batch_size=32)
    test_ds = create_dataset(test_df, batch_size=32)
    
    # Build and train hybrid model
    model = build_hybrid_model()
    history = train_model(model, train_ds, val_ds, series_name)
    
    # Evaluate
    results = evaluate_model(model, test_ds, class_names)
    print(f"\nMetrics for {series_name}:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    
    # Print confusion matrix details
    print(f"\nConfusion Matrix Details for {series_name}:")
    for class_name, metrics in results['cm_details'].items():
        print(f"\nClass: {class_name}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value}")
            
    # Plot confusion matrix
    plot_confusion_matrix(results['cm'], class_names, f'{series_name} Confusion Matrix')
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{series_name} Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{series_name} Accuracy')
    plt.legend()
    file_path = f'{series_name.lower().replace(" ", "_")}_plots.png'
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(file_path)
    plt.close()

