!pip install matplotlib seaborn scikit-learn


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
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
from keras.applications import ResNet50
from keras.layers import Conv2D, UpSampling2D, concatenate, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define the path to the training data
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

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



# Generate Image Paths
# ---------------------
def generate_image_paths(df, data_dir):
    """
    Generate a list of image file paths based on study_id and series_id.
    
    Parameters:
        df (DataFrame): DataFrame containing 'study_id' and 'series_id' columns.
        data_dir (str): Base directory where images are stored.
    
    Returns:
        List[str]: List of sorted image file paths.
    """
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



# Filter the DataFrame for a specific study_id and sort by instance_number
filtered_df = final_merged_df[final_merged_df['study_id'] == 1013589491].sort_values("instance_number")

# Display the filtered DataFrame
print("\nFiltered DataFrame for Study ID 1013589491:")
print(filtered_df)

# Sort the final merged DataFrame by study_id, series_id, and series_description
sorted_final_merged_df = final_merged_df[
    final_merged_df['study_id'] == 1013589491
].sort_values(by=['series_id', 'series_description', 'instance_number'])

print("\nSorted Final Merged DataFrame:")
print(sorted_final_merged_df)




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



# Calculate the number of entries for each severity level
normal_mild_count = final_merged_df[final_merged_df["severity"] == "Normal/Mild"].shape[0]
moderate_count = final_merged_df[final_merged_df["severity"] == "Moderate"].shape[0]
severe_count = final_merged_df[final_merged_df["severity"] == "Severe"].shape[0]

print(f"\nNormal/Mild Count: {normal_mild_count}")
print(f"Moderate Count: {moderate_count}")
print(f"Severe Count: {severe_count}")


# Define the base path for test images
base_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

def get_image_paths(row):
    """
    Retrieve all image file paths for a given series.
    
    Parameters:
        row (Series): A row from the test_desc DataFrame.
    
    Returns:
        List[str]: List of image file paths.
    """
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
    """
    Check if a given file or directory path exists.
    
    Parameters:
        path (str): File or directory path.
    
    Returns:
        bool: True if path exists, False otherwise.
    """
    return os.path.exists(path)

def check_study_id(row):
    """
    Check if the study_id directory exists.
    
    Parameters:
        row (Series): A row from the DataFrame.
    
    Returns:
        bool: True if study_id directory exists, False otherwise.
    """
    study_id = row['study_id']
    path = os.path.join(train_path, 'train_images', str(study_id))
    return check_exists(path)

def check_series_id(row):
    """
    Check if the series_id directory exists.
    
    Parameters:
        row (Series): A row from the DataFrame.
    
    Returns:
        bool: True if series_id directory exists, False otherwise.
    """
    study_id = row['study_id']
    series_id = row['series_id']
    path = os.path.join(train_path, 'train_images', str(study_id), str(series_id))
    return check_exists(path)

def check_image_exists(row):
    """
    Check if the image file exists.
    
    Parameters:
        row (Series): A row from the DataFrame.
    
    Returns:
        bool: True if image file exists, False otherwise.
    """
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

# Display the shape after filtering
print("\nTrain Data Shape after Filtering:", train_data.shape)



# Load and Display Sample Images
# ------------------------------
def load_dicom(path):
    """
    Load a DICOM image and normalize its pixel data.
    
    Parameters:
        path (str): Path to the DICOM file.
    
    Returns:
        np.ndarray: Normalized image data as uint8.
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)  # Normalize the image
    return (data * 255).astype(np.uint8)

# Select two random samples from the training data
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
    """
    Plot a confusion matrix using Seaborn heatmap.
    
    Parameters:
        cm (ndarray): Confusion matrix.
        classes (List[str]): List of class names.
        title (str): Title of the plot.
    """
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
def load_dicom_tf(path):
    """
    Load a DICOM image within a TensorFlow graph.
    
    Parameters:
        path (tf.Tensor): Path tensor.
    
    Returns:
        tf.Tensor: Image data as float32.
    """
    dicom = pydicom.dcmread(path.numpy().decode('utf-8'))
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    return data.astype(np.float32)

def preprocess_image(image, label=None):
    """
    Preprocess the image by resizing and converting to RGB.
    
    Parameters:
        image (tf.Tensor): Grayscale image tensor.
        label (tf.Tensor, optional): Label tensor.
    
    Returns:
        Tuple: Preprocessed image and label.
    """
    image = tf.expand_dims(image, -1)  # Add channel dimension
    image = tf.image.resize(image, [224, 224])  # Resize to 224x224
    image = tf.image.grayscale_to_rgb(image)  # Convert to RGB
    return (image, label) if label is not None else image

def create_dataset(df, batch_size=64, is_test=False):
    """
    Create a TensorFlow dataset from a DataFrame.
    
    Parameters:
        df (DataFrame): DataFrame containing image paths and labels.
        batch_size (int): Batch size.
        is_test (bool): Flag indicating whether it's test data.
    
    Returns:
        tf.data.Dataset: Prepared dataset.
    """
    if is_test:
        def load_wrapper(path):
            image = tf.py_function(load_dicom_tf, [path], tf.float32)
            image.set_shape((None, None))
            return preprocess_image(image)
        
        dataset = tf.data.Dataset.from_tensor_slices(df['image_path'])
        dataset = dataset.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def load_wrapper(path, label):
            image = tf.py_function(load_dicom_tf, [path], tf.float32)
            image.set_shape((None, None))
            return preprocess_image(image, label)
        
        labels = df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}).astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((df['image_path'], labels))
        dataset = dataset.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset



# Define Custom Loss Function
# ---------------------------
def weighted_log_loss(y_true, y_pred):
    """
    Compute a weighted sparse categorical cross-entropy loss.
    
    Parameters:
        y_true (Tensor): True labels.
        y_pred (Tensor): Predicted logits.
    
    Returns:
        Tensor: Weighted loss.
    """
    weights = tf.gather([1.0, 2.0, 4.0], tf.cast(y_true, tf.int32))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss * weights)



# Define Model Building Function
# ------------------------------
def build_resnet50(num_classes=3):
    """
    Build a ResNet50-based model for classification.
    
    Parameters:
        num_classes (int): Number of output classes.
    
    Returns:
        tf.keras.Model: Compiled ResNet50 model.
    """
    base_model = applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    # Freeze all layers except for conv4_block and conv5_block
    for layer in base_model.layers:
        if 'conv4_block' in layer.name or 'conv5_block' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model



# Define Training Function
# ------------------------
def train_model(model, train_dataset, val_dataset, series_name):
    """
    Compile and train the model.
    
    Parameters:
        model (tf.keras.Model): The model to train.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.
        series_name (str): Name of the series for checkpointing.
    
    Returns:
        History: Training history.
    """
    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss=weighted_log_loss,
        metrics=['accuracy']
    )

    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(f'best_{series_name}.keras', save_best_only=True),
        # callbacks.ReduceLROnPlateau(factor=0.1, patience=1)
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        callbacks=callbacks_list
    )
    return history



# Define Evaluation Function
# --------------------------
def evaluate_model(model, dataset):
    """
    Evaluate the model on a dataset and compute metrics.
    
    Parameters:
        model (tf.keras.Model): Trained model.
        dataset (tf.data.Dataset): Dataset for evaluation.
    
    Returns:
        dict: Dictionary containing precision, recall, f1-score, and confusion matrix.
    """
    y_true = []
    y_pred = []
    for batch in dataset:
        # Extract images and labels from the batch
        images, labels = batch[0], batch[1]
        y_true.extend(labels.numpy())
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))

    # Remove NaN values if any
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



# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Conv2D, UpSampling2D, concatenate, GlobalAveragePooling2D, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping


from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Initialize models for each series
series_models = {
    'Sagittal T1': build_resnet50(),
    'Axial T2': build_resnet50(),
    'Sagittal T2/STIR': build_resnet50()
}

class_names = ['normal_mild', 'moderate', 'severe']
results = {}

for series_name, model in series_models.items():
    # Filter data for the current series and valid severity levels
    series_df = final_merged_df[
        (final_merged_df['series_description'] == series_name) &
        (final_merged_df['severity'].isin(class_names))
    ].copy()

    if series_df.empty:
        print(f"Skipping {series_name} - no valid data.")
        continue

    # Split data into train (70%), temp (30%)
    train_df, temp_df = train_test_split(
        series_df,
        test_size=0.3,
        stratify=series_df['severity'],
        random_state=42
    )

    # Split temp into validation (10%) and test (20%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.6667,  # 20%/(10%+20%) = 2/3
        stratify=temp_df['severity'],
        random_state=42
    )

    # Display class distribution
    print(f"\nClass distribution for {series_name}:")
    print("Train:", train_df['severity'].value_counts())
    print("Validation:", val_df['severity'].value_counts())
    print("Test:", test_df['severity'].value_counts())

    # Create TensorFlow datasets
    train_ds = create_dataset(train_df, batch_size=64, is_test=False)
    val_ds = create_dataset(val_df, batch_size=64, is_test=False)
    test_ds = create_dataset(test_df, batch_size=64, is_test=False)
    # Train the model
    history = train_model(
        model,
        train_ds,
        val_ds,
        series_name,
    )

    # Evaluate the model on TEST set
    result = evaluate_model(model, test_ds)
    results[series_name] = result




    # Plot confusion matrix
    plot_confusion_matrix(result['cm'], class_names, f'{series_name} Confusion Matrix')

    # Print evaluation metrics
    print(f"\nMetrics for {series_name}:")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-Score: {result['f1']:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{series_name} Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{series_name} Accuracy')
    plt.legend()
    
    plt.show()


from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def check_image_paths(df, image_col='image_path'):
    invalid_paths = []
    for path in df[image_col]:
        if not os.path.isfile(path):
            invalid_paths.append(path)
    return invalid_paths

def filter_valid_image_paths(df, image_col='image_path'):
    valid_df = df[df[image_col].apply(lambda x: os.path.isfile(x))]
    return valid_df

def build_resnet50_fpn(input_shape=(224, 224, 3), num_classes=3):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    C2 = base_model.get_layer('conv2_block3_out').output
    C3 = base_model.get_layer('conv3_block4_out').output
    C4 = base_model.get_layer('conv4_block6_out').output
    C5 = base_model.get_layer('conv5_block3_out').output
    
    P5 = Conv2D(512, (1, 1), name='fpn_c5p5')(C5)
    P4 = Conv2D(512, (1, 1), name='fpn_c4p4')(C4)
    P4 = UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5) + P4
    
    P3 = Conv2D(512, (1, 1), name='fpn_c3p3')(C3)
    P3 = UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4) + P3
    
    P2 = Conv2D(512, (1, 1), name='fpn_c2p2')(C2)
    P2 = UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3) + P2
    
    P2 = Conv2D(512, (3, 3), padding='same', name='fpn_p2')(P2)
    P3 = Conv2D(512, (3, 3), padding='same', name='fpn_p3')(P3)
    P4 = Conv2D(512, (3, 3), padding='same', name='fpn_p4')(P4)
    P5 = Conv2D(512, (3, 3), padding='same', name='fpn_p5')(P5)
    
    pooled = [GlobalAveragePooling2D()(p) for p in [P2, P3, P4, P5]]
    merged = concatenate(pooled, axis=-1)
    
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    
    outputs = Dense(num_classes, activation='softmax')(merged)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    return model

# افزایش داده‌ها
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# مدل‌ها برای هر سری داده
series_models = {
    'Sagittal T1': build_resnet50_fpn(),
    'Axial T2': build_resnet50_fpn(),
    'Sagittal T2/STIR': build_resnet50_fpn()
}

class_names = ['normal_mild', 'moderate', 'severe']
results = {}

# آموزش مدل برای هر سری داده
for series_name, model in series_models.items():
    series_df = final_merged_df[
        (final_merged_df['series_description'] == series_name) &
        (final_merged_df['severity'].isin(class_names))
    ].copy()

    if series_df.empty:
        print(f"Skipping {series_name} - no valid data.")
        continue

    # بررسی مسیرهای نامعتبر
    invalid_paths = check_image_paths(series_df)
    if invalid_paths:
        print(f"Found {len(invalid_paths)} invalid image paths in {series_name}.")
        print(invalid_paths[:10])  # نمایش 10 مسیر نامعتبر اول
        series_df = filter_valid_image_paths(series_df)

    if series_df.empty:
        print(f"Skipping {series_name} - no valid image paths after filtering.")
        continue

    # تقسیم داده‌ها به آموزش (70%) و موقت (30%)
    train_df, temp_df = train_test_split(
        series_df,
        test_size=0.3,
        stratify=series_df['severity'],
        random_state=42
    )

    # تقسیم موقت به اعتبارسنجی (10%) و تست (20%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.6667,
        stratify=temp_df['severity'],
        random_state=42
    )

    # نمایش توزیع کلاس‌ها
    print(f"\nClass distribution for {series_name}:")
    print("Train:", train_df['severity'].value_counts())
    print("Validation:", val_df['severity'].value_counts())
    print("Test:", test_df['severity'].value_counts())

    # بررسی خالی نبودن DataFrame‌ها
    if train_df.empty or val_df.empty or test_df.empty:
        print(f"Skipping {series_name} - one of the datasets is empty.")
        continue

    # ایجاد مجموعه‌های داده TensorFlow با افزایش داده
    train_ds = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='severity',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical'
    )
    
    val_ds = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='severity',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical'
    )
    
    test_ds = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='severity',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical'
    )

    # کامپایل مدل
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Callback‌ها
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.1, patience=5)
    ]

    # آموزش مدل
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=callbacks
        )
    except ValueError as e:
        print(f"Error during model training: {e}")
        continue

    # ارزیابی مدل بر روی مجموعه تست
    result = model.evaluate(test_ds)
    results[series_name] = result

    # نمایش معیارهای ارزیابی
    print(f"\nMetrics for {series_name}:")
    print(f"Loss: {result[0]:.4f}")
    print(f"Accuracy: {result[1]:.4f}")

    # ترسیم تاریخچه آموزش
    plt.figure(figsize=(12, 5))
    
    # ترسیم Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{series_name} Loss')
    plt.legend()
    
    # ترسیم دقت
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{series_name} Accuracy')
    plt.legend()
    
    plt.show()


# from keras.applications import ResNet50
# from keras.layers import Conv2D, UpSampling2D, concatenate, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
# from keras.models import Model
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator  # تغییر این خط

# def build_resnet50_fpn(input_shape=(224, 224, 3), num_classes=3):
#     # Load ResNet50 backbone
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
#     # Get feature maps from different levels of ResNet50
#     C2 = base_model.get_layer('conv2_block3_out').output
#     C3 = base_model.get_layer('conv3_block4_out').output
#     C4 = base_model.get_layer('conv4_block6_out').output
#     C5 = base_model.get_layer('conv5_block3_out').output
    
#     # FPN Top-down pathway and lateral connections
#     P5 = Conv2D(512, (1, 1), name='fpn_c5p5')(C5)
#     P4 = Conv2D(512, (1, 1), name='fpn_c4p4')(C4)
#     P4 = UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5) + P4
    
#     P3 = Conv2D(512, (1, 1), name='fpn_c3p3')(C3)
#     P3 = UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4) + P3
    
#     P2 = Conv2D(512, (1, 1), name='fpn_c2p2')(C2)
#     P2 = UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3) + P2
    
#     # Final convolution layers for each level
#     P2 = Conv2D(512, (3, 3), padding='same', name='fpn_p2')(P2)
#     P3 = Conv2D(512, (3, 3), padding='same', name='fpn_p3')(P3)
#     P4 = Conv2D(512, (3, 3), padding='same', name='fpn_p4')(P4)
#     P5 = Conv2D(512, (3, 3), padding='same', name='fpn_p5')(P5)
    
#     # Global Average Pooling and classification head
#     pooled = [GlobalAveragePooling2D()(p) for p in [P2, P3, P4, P5]]
#     merged = concatenate(pooled, axis=-1)
    
#     # Add BatchNormalization and Dropout
#     merged = BatchNormalization()(merged)
#     merged = Dropout(0.5)(merged)
    
#     # Output layer
#     outputs = Dense(num_classes, activation='softmax')(merged)
    
#     # Build the model
#     model = Model(inputs=base_model.input, outputs=outputs)
    
#     return model

# # Data Augmentation
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Initialize models for each series with FPN
# series_models = {
#     'Sagittal T1': build_resnet50_fpn(),
#     'Axial T2': build_resnet50_fpn(),
#     'Sagittal T2/STIR': build_resnet50_fpn()
# }

# class_names = ['normal_mild', 'moderate', 'severe']
# results = {}

# for series_name, model in series_models.items():
#     # Filter data for the current series and valid severity levels
#     series_df = final_merged_df[
#         (final_merged_df['series_description'] == series_name) &
#         (final_merged_df['severity'].isin(class_names))
#     ].copy()

#     if series_df.empty:
#         print(f"Skipping {series_name} - no valid data.")
#         continue

#     # Split data into train (70%), temp (30%)
#     train_df, temp_df = train_test_split(
#         series_df,
#         test_size=0.3,
#         stratify=series_df['severity'],
#         random_state=42
#     )

#     # Split temp into validation (10%) and test (20%)
#     val_df, test_df = train_test_split(
#         temp_df,
#         test_size=0.6667,  # 20%/(10%+20%) = 2/3
#         stratify=temp_df['severity'],
#         random_state=42
#     )

#     # Display class distribution
#     print(f"\nClass distribution for {series_name}:")
#     print("Train:", train_df['severity'].value_counts())
#     print("Validation:", val_df['severity'].value_counts())
#     print("Test:", test_df['severity'].value_counts())

#     # Create TensorFlow datasets with data augmentation
#     train_ds = datagen.flow_from_dataframe(
#         dataframe=train_df,
#         x_col='image_path',  # Replace with your image column name
#         y_col='severity',
#         target_size=(224, 224),
#         batch_size=64,
#         class_mode='categorical'
#     )
    
#     val_ds = datagen.flow_from_dataframe(
#         dataframe=val_df,
#         x_col='image_path',  # Replace with your image column name
#         y_col='severity',
#         target_size=(224, 224),
#         batch_size=64,
#         class_mode='categorical'
#     )
    
#     test_ds = datagen.flow_from_dataframe(
#         dataframe=test_df,
#         x_col='image_path',  # Replace with your image column name
#         y_col='severity',
#         target_size=(224, 224),
#         batch_size=64,
#         class_mode='categorical'
#     )

#     # Compile the model
#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#     # Callbacks
#     callbacks = [
#         EarlyStopping(patience=10, restore_best_weights=True),
#         ReduceLROnPlateau(factor=0.1, patience=5)
#     ]

#     # Train the model
#     history = model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=50,
#         callbacks=callbacks
#     )

#     # Evaluate the model on TEST set
#     result = model.evaluate(test_ds)
#     results[series_name] = result

#     # Print evaluation metrics
#     print(f"\nMetrics for {series_name}:")
#     print(f"Loss: {result[0]:.4f}")
#     print(f"Accuracy: {result[1]:.4f}")

#     # Plot training history
#     plt.figure(figsize=(12, 5))
    
#     # Plot Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title(f'{series_name} Loss')
#     plt.legend()
    
#     # Plot Accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Train Acc')
#     plt.plot(history.history['val_accuracy'], label='Val Acc')
#     plt.title(f'{series_name} Accuracy')
#     plt.legend()
    
#     plt.show()


# from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split

# # Initialize models for each series
# series_models = {
#     'Sagittal T1': build_resnet50(),
#     'Axial T2': build_resnet50(),
#     'Sagittal T2/STIR': build_resnet50()
# }

# class_names = ['normal_mild', 'moderate', 'severe']
# results = {}

# for series_name, model in series_models.items():
#     # Filter data for the current series and valid severity levels
#     series_df = final_merged_df[
#         (final_merged_df['series_description'] == series_name) &
#         (final_merged_df['severity'].isin(class_names))
#     ].copy()

#     if series_df.empty:
#         print(f"Skipping {series_name} - no valid data.")
#         continue

#     # Split data into train (70%), temp (30%)
#     train_df, temp_df = train_test_split(
#         series_df,
#         test_size=0.3,
#         stratify=series_df['severity'],
#         random_state=42
#     )

#     # Split temp into validation (10%) and test (20%)
#     val_df, test_df = train_test_split(
#         temp_df,
#         test_size=0.6667,  # 20%/(10%+20%) = 2/3
#         stratify=temp_df['severity'],
#         random_state=42
#     )

#     # Display class distribution
#     print(f"\nClass distribution for {series_name}:")
#     print("Train:", train_df['severity'].value_counts())
#     print("Validation:", val_df['severity'].value_counts())
#     print("Test:", test_df['severity'].value_counts())

#     # Create TensorFlow datasets
#     train_ds = create_dataset(train_df, batch_size=64, is_test=False)
#     val_ds = create_dataset(val_df, batch_size=64, is_test=False)
#     test_ds = create_dataset(test_df, batch_size=64, is_test=False)
#     # Train the model
#     history = train_model(
#         model,
#         train_ds,
#         val_ds,
#         series_name,
#     )

#     # Evaluate the model on TEST set
#     result = evaluate_model(model, test_ds)
#     results[series_name] = result

#     # Plot confusion matrix
#     plot_confusion_matrix(result['cm'], class_names, f'{series_name} Confusion Matrix')

#     # Print evaluation metrics
#     print(f"\nMetrics for {series_name}:")
#     print(f"Precision: {result['precision']:.4f}")
#     print(f"Recall: {result['recall']:.4f}")
#     print(f"F1-Score: {result['f1']:.4f}")

#     # Plot training history
#     plt.figure(figsize=(12, 5))
    
#     # Plot Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title(f'{series_name} Loss')
#     plt.legend()
    
#     # Plot Accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Train Acc')
#     plt.plot(history.history['val_accuracy'], label='Val Acc')
#     plt.title(f'{series_name} Accuracy')
#     plt.legend()
    
#     plt.show()


import os
import pandas as pd
import numpy as np
import tensorflow as tf

def predict_test_and_generate_submission(series_models, expanded_test_desc, class_names):
    """
    Generate predictions on the test set and create a submission file.
    Fixed to handle series names with special characters and FutureWarnings.
    """
    # Use a list to collect prediction DataFrames
    dfs = []

    for series_name, model in series_models.items():
        try:
            # Split the series_name by '/'
            parts = series_name.split('/')
            
            if len(parts) == 2:
                # Construct path: best_<first_part>/<second_part>.keras
                weight_dir = f'best_{parts[0]}'
                weight_filename = f"{parts[1]}.keras"
                weight_path = os.path.join(weight_dir, weight_filename)
            else:
                # Construct path: best_<series_name>.keras
                weight_path = f'best_{series_name}.keras'
            
            # Ensure the weight_path is correct
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Weight file not found: {weight_path}")

            # Load model weights
            model.load_weights(weight_path)
            print(f"Loaded weights for '{series_name}' from '{weight_path}'")

            # Filter test data for current series
            series_test = expanded_test_desc[expanded_test_desc['series_description'] == series_name]

            if series_test.empty:
                print(f"No test data for '{series_name}'. Skipping.")
                continue

            # Create test dataset and predict
            test_ds = create_dataset(series_test, batch_size=64, is_test=True)
            preds = model.predict(test_ds, verbose=1)

            # Handle model outputs
            if preds.ndim == 2 and preds.shape[1] > 1:
                if not np.allclose(preds.sum(axis=1), 1):
                    probs = tf.nn.softmax(preds, axis=1).numpy()
                else:
                    probs = preds
            elif preds.ndim == 1:
                # If model outputs a single probability, convert to multi-class
                probs = np.vstack([1 - preds, preds, np.zeros_like(preds)]).T
            else:
                raise ValueError(f"Unexpected prediction shape: {preds.shape}")

            # Ensure probs has exactly three columns
            if probs.shape[1] != 3:
                raise ValueError(f"Expected 3 classes, but got {probs.shape[1]} for '{series_name}'")

            # Create temporary DataFrame
            temp_df = series_test[['row_id']].copy()
            temp_df[['normal_mild', 'moderate', 'severe']] = probs

            # Ensure all probability columns are numeric
            temp_df[['normal_mild', 'moderate', 'severe']] = temp_df[['normal_mild', 'moderate', 'severe']].apply(pd.to_numeric, errors='coerce')

            # Check for any NaNs introduced by non-numeric values
            if temp_df[['normal_mild', 'moderate', 'severe']].isnull().values.any():
                raise ValueError(f"Non-numeric prediction probabilities encountered for '{series_name}'")

            # Append to list
            dfs.append(temp_df)
            print(f"Added predictions for '{series_name}'")

        except Exception as e:
            print(f"Error in '{series_name}': {str(e)}")
            continue

    # Combine all predictions
    submission = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['row_id', 'normal_mild', 'moderate', 'severe'])

    # Handle missing rows
    all_row_ids = expanded_test_desc['row_id'].unique()
    submission_row_ids = submission['row_id'].unique()
    missing_ids = set(all_row_ids) - set(submission_row_ids)
    
    if missing_ids:
        print(f"Adding default probabilities for {len(missing_ids)} missing rows")
        missing_df = pd.DataFrame({
            'row_id': list(missing_ids),
            'normal_mild': 1/3,
            'moderate': 1/3,
            'severe': 1/3
        })
        submission = pd.concat([submission, missing_df], ignore_index=True)

    # Aggregate and normalize
    numeric_cols = ['normal_mild', 'moderate', 'severe']
    submission = submission.groupby('row_id')[numeric_cols].mean().reset_index()
    
    # Normalize the probabilities to ensure they sum to 1
    submission[numeric_cols] = (
        submission[numeric_cols]
        .div(submission[numeric_cols].sum(axis=1), axis=0)
        .fillna(1/3)
        .round(3)
    )

    # Verify that all probability columns are numeric using numpy
    if not np.isreal(submission[numeric_cols].values).all():
        raise ValueError("Non-numeric values detected in the probability columns after aggregation.")

    # Save submission
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully")

predict_test_and_generate_submission(series_models, expanded_test_desc, class_names)


import pandas as pd
df = pd.read_csv("/kaggle/working/submission.csv")
df


# import pandas as pd
# df = pd.read_csv("/kaggle/working/submission.csv")
# df













