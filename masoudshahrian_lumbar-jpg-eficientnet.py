# !pip install -q kaggle
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
# !unzip -qq /content/rsna-2024-lumbar-spine-degenerative-classification.zip

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
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight


# Define the path to the training data
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'
train_path2 = '/kaggle/input/lumbar-2024-imagesdicom2jpg/'  # مسیر فرضی برای دیتاست JPG

# Load CSV files containing metadata and labels
train = pd.read_csv(os.path.join(train_path, 'train.csv'))
label = pd.read_csv(os.path.join(train_path, 'train_label_coordinates.csv'))
train_desc = pd.read_csv(os.path.join(train_path, 'train_series_descriptions.csv'))
test_desc = pd.read_csv(os.path.join(train_path, 'test_series_descriptions.csv'))
sub = pd.read_csv(os.path.join(train_path, 'sample_submission.csv'))

# Display the first few rows of each dataframe
print("Test Descriptions:")
print(test_desc.head(5))

print("\nTrain Data:")
print(train.head(5))

print("\nTrain Series Descriptions:")
print(train_desc.head(5))


# Generate Image Paths (تغییر به JPG)
# ---------------------
def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        images = sorted([f for f in os.listdir(series_dir) if f.endswith('.jpg')])
        image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths

# Generate image paths for training and testing datasets
train_image_paths = generate_image_paths(train_desc, os.path.join(train_path2, 'train_images'))
test_image_paths = generate_image_paths(test_desc, os.path.join(train_path2, 'test_images'))

# Example usage
print("\nSample Train Image Path:", train_image_paths[2])
print("Number of Train Descriptions:", len(train_desc))
print("Number of Train Image Paths:", len(train_image_paths))



# Display JPG Images
# ---------------------
def display_jpg_images(image_paths, num_images=3):
    plt.figure(figsize=(15, 5))
    for i, path in enumerate(image_paths[:num_images]):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, num_images, i+1)
        plt.imshow(img, cmap=plt.cm.bone)
        plt.title(f"Image {i+1}")
        plt.axis('off')
    plt.show()

# Display the first three JPG images from training data
display_jpg_images(train_image_paths)

# Display JPG Images with Coordinates
# -------------------------------------
def display_jpg_with_coordinates(image_paths, label_df):
    fig, axs = plt.subplots(1, len(image_paths), figsize=(18, 6))
    for idx, path in enumerate(image_paths):
        study_id = int(path.split('/')[-3])
        series_id = int(path.split('/')[-2])
        filtered_labels = label_df[
            (label_df['study_id'] == study_id) &
            (label_df['series_id'] == series_id)
        ]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        axs[idx].imshow(img, cmap='gray')
        axs[idx].set_title(f"Study ID: {study_id}, Series ID: {series_id}")
        axs[idx].axis('off')
        for _, row in filtered_labels.iterrows():
            axs[idx].plot(row['x'], row['y'], 'ro', markersize=5)
    plt.tight_layout()
    plt.show()

def load_jpg_files(path_to_folder):
    files = [os.path.join(path_to_folder, f) for f in os.listdir(path_to_folder) if f.endswith('.jpg')]
    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('-')[-1]))
    return files

# Example: Display JPG images with coordinates for a specific study
study_id = "100206310"
study_folder = os.path.join(train_path2, 'train_images', study_id)
image_paths = []
for series_folder in os.listdir(study_folder):
    series_folder_path = os.path.join(study_folder, series_folder)
    jpg_files = load_jpg_files(series_folder_path)
    if jpg_files:
        image_paths.append(jpg_files[0])
display_jpg_with_coordinates(image_paths, label)



# Data Reshaping and Merging
# --------------------------
def reshape_row(row):
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

new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)
print("\nReshaped Train Data:")
print(new_train_df.head(5))


# Merge DataFrames
# -----------------
merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')
final_merged_df['row_id'] = (
    final_merged_df['study_id'].astype(str) + '_' +
    final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
    final_merged_df['level'].str.lower().str.replace('/', '_')
)
final_merged_df['image_path'] = (
    os.path.join(train_path2, 'train_images') + '/' +
    final_merged_df['study_id'].astype(str) + '/' +
    final_merged_df['series_id'].astype(str) + '/' +
    final_merged_df['instance_number'].astype(str) + '.jpg'
)
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
        return [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.jpg')]
    return []

condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

expanded_rows = []
for index, row in test_desc.iterrows():
    image_paths = get_image_paths(row)
    conditions = condition_mapping.get(row['series_description'], {})
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

expanded_test_desc = pd.DataFrame(expanded_rows)
print("\nExpanded Test Descriptions:")
print(expanded_test_desc.head(5))

final_merged_df['severity'] = final_merged_df['severity'].map({
    'Normal/Mild': 'normal_mild',
    'Moderate': 'moderate',
    'Severe': 'severe'
})









train_data = final_merged_df
test_data = expanded_test_desc

print("\nSample Train Data:")
print(train_data.head(10))
print("\nSample Test Data:")
print(test_data.head(10))
print("\nTrain Data Shape:", train_data.shape)



# Verify File Paths
# -----------------
def check_exists(path):
    return os.path.exists(path)

def check_study_id(row):
    study_id = row['study_id']
    path = os.path.join(train_path2, 'train_images', str(study_id))
    return check_exists(path)

def check_series_id(row):
    study_id = row['study_id']
    series_id = row['series_id']
    path = os.path.join(train_path2, 'train_images', str(study_id), str(series_id))
    return check_exists(path)

def check_image_exists(row):
    image_path = row['image_path']
    return check_exists(image_path)

train_data['study_id_exists'] = train_data.apply(check_study_id, axis=1)
train_data['series_id_exists'] = train_data.apply(check_series_id, axis=1)
train_data['image_exists'] = train_data.apply(check_image_exists, axis=1)

train_data = train_data[
    train_data['study_id_exists'] &
    train_data['series_id_exists'] &
    train_data['image_exists']
]
print("\nTrain Data Shape after Filtering:", train_data.shape)


# Load and Display Sample Images
# ------------------------------
def load_jpg(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)
    return (img * 255).astype(np.uint8)

images = []
row_ids = []
selected_indices = random.sample(range(len(train_data)), 2)
for i in selected_indices:
    image = load_jpg(train_data.iloc[i]['image_path'])
    images.append(image)
    row_ids.append(train_data.iloc[i]['row_id'])

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for i in range(2):
    ax[i].imshow(images[i], cmap='gray')
    ax[i].set_title(f'Row ID: {row_ids[i]}', fontsize=8)
    ax[i].axis('off')
plt.tight_layout()
plt.show()

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



# Data Loading and Preprocessing Functions
# ----------------------------------------
def load_jpg_tf(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.cast(img, tf.float32)
    img = img - tf.reduce_min(img)
    img = img / tf.reduce_max(img) if tf.reduce_max(img) != 0 else img
    return img

class_counts = [37626, 7950, 3081]
class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=np.repeat([0, 1, 2], class_counts))
class_weights = dict(enumerate(class_weights))

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

def augment_image_normalized(image):
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

def preprocess_image(image, label=None, is_training=False):
    def opencv_process(img_tensor):
        img_np = img_tensor.numpy().squeeze()
        img_np = apply_bilateral_filter(img_np)
        img_np = apply_clahe(img_np)
        img_np = remove_background(img_np)
        if is_training:
            img_np = augment_image_normalized(tf.expand_dims(img_np, axis=-1)).numpy().squeeze()
        return np.expand_dims(img_np, axis=-1).astype(np.float32)

    image = tf.expand_dims(image, axis=-1)
    image = tf.py_function(opencv_process, [image], tf.float32)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [224, 224])
    image = tf.image.grayscale_to_rgb(image)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return (image, label) if label is not None else image




def create_dataset(df, batch_size=64, is_test=False, is_training=False):
    if is_test:
        def load_wrapper(path):
            image = load_jpg_tf(path)
            return preprocess_image(image)
        dataset = tf.data.Dataset.from_tensor_slices(df['image_path'])
        dataset = dataset.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def load_wrapper(path, label):
            image = load_jpg_tf(path)
            return preprocess_image(image, label, is_training)
        labels = df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}).astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((df['image_path'], labels))
        dataset = dataset.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset




# Train EfficientNetB2 model
# --------------------------
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

def weighted_log_loss(y_true, y_pred):
    weights = tf.gather([1.0, 2.0, 4.0], tf.cast(y_true, tf.int32))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss * weights)

def build_efficientnet_b2(num_classes=3):
    base_model = applications.EfficientNetB2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    for layer in base_model.layers:
        if 'block7' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='linear')(x)
    return tf.keras.Model(inputs, outputs)

def train_model(model, train_dataset, val_dataset, series_name):
    model.compile(
        optimizer=optimizers.Adam(0.0001),
        loss=weighted_log_loss,
        metrics=['accuracy']
    )
    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(f'best_{series_name}.keras', save_best_only=True),
    ]
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        callbacks=callbacks_list
    )
    return history

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

series_models = {
    'Sagittal T1': build_efficientnet_b2(),
    'Axial T2': build_efficientnet_b2(),
    'Sagittal T2/STIR': build_efficientnet_b2()
}

class_names = ['normal_mild', 'moderate', 'severe']
results = {}

for series_name, model in series_models.items():
    series_df = final_merged_df[
        (final_merged_df['series_description'] == series_name) &
        (final_merged_df['severity'].isin(class_names))
    ].copy()
    if series_df.empty:
        print(f"Skipping {series_name} - no valid data.")
        continue
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





    print(f"\nClass distribution for {series_name}:")
    print("Train:", train_df['severity'].value_counts())
    print("Validation:", val_df['severity'].value_counts())
    print("Test:", test_df['severity'].value_counts())
    train_ds = create_dataset(train_df, batch_size=64, is_test=False, is_training=True)
    val_ds = create_dataset(val_df, batch_size=64, is_test=False)
    test_ds = create_dataset(test_df, batch_size=64, is_test=False)
    history = train_model(model, train_ds, val_ds, series_name)
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


# ResNet50 Section (بازنویسی برای JPG)
# -------------------------------------
def build_resnet50(num_classes=3):
    base_model = applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    for layer in base_model.layers:
        if 'conv4_block' in layer.name or 'conv5_block' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='linear')(x)
    return tf.keras.Model(inputs, outputs)

series_models_resnet = {
    'Sagittal T1': build_resnet50(),
    'Axial T2': build_resnet50(),
    'Sagittal T2/STIR': build_resnet50()
}

results_resnet = {}

for series_name, model in series_models_resnet.items():
    series_df = final_merged_df[
        (final_merged_df['series_description'] == series_name) &
        (final_merged_df['severity'].isin(class_names))
    ].copy()
    if series_df.empty:
        print(f"Skipping {series_name} - no valid data.")
        continue
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
    print(f"\nClass distribution for {series_name} (ResNet50):")
    print("Train:", train_df['severity'].value_counts())
    print("Validation:", val_df['severity'].value_counts())
    print("Test:", test_df['severity'].value_counts())
    train_ds = create_dataset(train_df, batch_size=64, is_test=False, is_training=True)
    val_ds = create_dataset(val_df, batch_size=64, is_test=False)
    test_ds = create_dataset(test_df, batch_size=64, is_test=False)
    history = train_model(model, train_ds, val_ds, series_name)  # Using weighted_log_loss here too
    result = evaluate_model(model, test_ds)
    results_resnet[series_name] = result
    plot_confusion_matrix(result['cm'], class_names, f'{series_name} Confusion Matrix (ResNet50)')
    print(f"\nMetrics for {series_name} (ResNet50):")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-Score: {result['f1']:.4f}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{series_name} Loss (ResNet50)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{series_name} Accuracy (ResNet50)')
    plt.legend()
    plt.show()

