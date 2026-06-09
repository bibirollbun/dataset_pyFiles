# !pip install pydicom
# !pip install matplotlib seaborn scikit-learn


# import os
# import numpy as np
# import pandas as pd
# import cv2
# import pydicom
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers, callbacks
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.utils.class_weight import compute_class_weight


!pip install pydicom
!pip install matplotlib seaborn scikit-learn


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





# !pip install pydicom
# !pip install matplotlib seaborn scikit-learn

# import os
# import numpy as np
# import pandas as pd
# import cv2
# import pydicom
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers, callbacks
# from tensorflow.keras.applications import EfficientNetB1
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.utils.class_weight import compute_class_weight
# import matplotlib.pyplot as plt
# import seaborn as sns

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





# !pip install pydicom
# !pip install matplotlib seaborn scikit-learn

# import os
# import numpy as np
# import pandas as pd
# import cv2
# import pydicom
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers, callbacks
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.utils.class_weight import compute_class_weight
# # Set up mixed precision
# tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# # Define dataset path
# train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

# # Load CSV files
# train = pd.read_csv(os.path.join(train_path, 'train.csv'))
# label = pd.read_csv(os.path.join(train_path, 'train_label_coordinates.csv'))
# train_desc = pd.read_csv(os.path.join(train_path, 'train_series_descriptions.csv'))

# # Reshape train.csv
# def reshape_row(row):
#     data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
#     for column, value in row.items():
#         if column != 'study_id':
#             parts = column.split('_')
#             condition = ' '.join([word.capitalize() for word in parts[:-2]])
#             level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
#             data['study_id'].append(row['study_id'])
#             data['condition'].append(condition)
#             data['level'].append(level)
#             data['severity'].append(value)
#     return pd.DataFrame(data)

# # Create and merge DataFrames
# new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)
# merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
# final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')

# # Create image paths
# final_merged_df['image_path'] = (
#     train_path + 'train_images/' +
#     final_merged_df['study_id'].astype(str) + '/' +
#     final_merged_df['series_id'].astype(str) + '/' +
#     final_merged_df['instance_number'].astype(str) + '.dcm'
# )

# # Map severity labels
# final_merged_df['severity'] = final_merged_df['severity'].map({
#     'Normal/Mild': 'normal_mild',
#     'Moderate': 'moderate',
#     'Severe': 'severe'
# })

# # Filter invalid rows
# final_merged_df = final_merged_df[final_merged_df['severity'].isin(['normal_mild', 'moderate', 'severe'])]

# # Compute class weights
# class_counts = final_merged_df['severity'].value_counts().sort_index().values
# class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), 
#                                     y=final_merged_df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}))
# class_weights_dict = dict(enumerate(class_weights))

# # Define Focal Loss
# def focal_loss(y_true, y_pred, alpha=list(class_weights_dict.values()), gamma=2.0):
#     y_true = tf.cast(y_true, tf.int32)
#     ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
#     probs = tf.nn.softmax(y_pred, axis=-1)
#     probs = tf.gather(probs, y_true, batch_dims=1)
#     alpha = tf.gather(alpha, y_true)
#     modulating_factor = tf.pow(1.0 - probs, gamma)
#     return tf.reduce_mean(alpha * modulating_factor * ce)

# # Image preprocessing functions
# def apply_clahe(image, clip_limit=2, tile_grid_size=(16, 16)):
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8) if np.issubdtype(image.dtype, np.floating) else image.astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     return clahe.apply(image)

# def apply_bilateral_filter(image, diameter=5, sigma_color=10, sigma_space=10):
#     if image.dtype == np.float32 or image.dtype == np.float64:
#         if image.max() <= 1.0:
#             sigma_color = sigma_color / 255.0
#         else:
#             image = (image / 255.0).astype(np.float32)
#     elif image.dtype != np.uint8:
#         image = image.astype(np.uint8)
#     return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

# def augment_image(image):
#     image = tf.image.random_brightness(image, max_delta=5/255.0)
#     image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
#     image_uint8 = tf.image.convert_image_dtype(image, tf.uint8)
#     image_uint8 = tf.image.random_jpeg_quality(image_uint8, 80, 100)
#     return tf.image.convert_image_dtype(image_uint8, tf.float32)

# def remove_background(image):
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8) if image.dtype == np.float32 else image.astype(np.uint8)
#     _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return cv2.bitwise_and(image, image, mask=mask)

# def load_dicom_tf(path):
#     dicom = pydicom.dcmread(path.numpy().decode('utf-8'))
#     data = dicom.pixel_array
#     data = data - np.min(data)
#     if np.max(data) != 0:
#         data = data / np.max(data)
#     return data.astype(np.float32)

# def preprocess_image(image, label=None, is_training=False):
#     def process_with_opencv(img_tensor):
#         img_np = img_tensor.numpy().squeeze()
#         img_np = apply_bilateral_filter(img_np)
#         img_np = apply_clahe(img_np)
#         img_np = remove_background(img_np)
#         if is_training:
#             img_np = augment_image(tf.expand_dims(img_np, axis=-1)).numpy().squeeze()
#         img_np = np.expand_dims(img_np, axis=-1)
#         return img_np.astype(np.float32)

#     image = tf.py_function(load_dicom_tf, [image], tf.float32)
#     image.set_shape([None, None])
#     image = tf.py_function(process_with_opencv, [image], tf.float32)
#     image.set_shape([None, None, 1])
#     image = tf.image.resize(image, [224, 224])
#     image = tf.image.grayscale_to_rgb(image)
#     image = tf.keras.applications.resnet50.preprocess_input(image)
#     return (image, label) if label is not None else image

# # Create dataset
# def create_dataset(df, batch_size=32, is_training=False):
#     labels = df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}).astype(np.int32)
#     dataset = tf.data.Dataset.from_tensor_slices((df['image_path'], labels))
#     dataset = dataset.map(lambda x, y: preprocess_image(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
    
#     if is_training:
#         dataset = dataset.apply(tf.data.experimental.rejection_resample(
#             class_func=lambda image, label: label,
#             target_dist=list(class_weights_dict.values()),
#             initial_dist=list(class_weights_dict.values())
#         )).map(lambda resampled_label, original_sample: original_sample)
    
#     dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return dataset

# # Build U-Net encoder
# def build_unet_encoder(input_shape=(224, 224, 3), num_classes=3):
#     inputs = layers.Input(shape=input_shape)
#     c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
#     c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
#     p1 = layers.MaxPooling2D((2, 2))(c1)
    
#     c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
#     c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
#     p2 = layers.MaxPooling2D((2, 2))(c2)
    
#     c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
#     c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
#     p3 = layers.MaxPooling2D((2, 2))(c3)
    
#     c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
#     c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(c4)
#     p4 = layers.MaxPooling2D((2, 2))(c4)
    
#     c5 = layers.Conv2D(1024, 3, padding='same', activation='relu')(p4)
#     c5 = layers.Conv2D(1024, 3, padding='same', activation='relu')(c5)
    
#     x = layers.GlobalAveragePooling2D()(c5)
#     x = layers.Dense(512, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(num_classes, activation='linear')(x)
    
#     return models.Model(inputs, outputs)

# # Train model
# def train_model(model, train_ds, val_ds, series_name):
#     model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
#                   loss=focal_loss, 
#                   metrics=['accuracy'])
#     callbacks_list = [
#         callbacks.EarlyStopping(patience=10, restore_best_weights=True),
#         callbacks.ModelCheckpoint(f'best_{series_name}_unet.keras', save_best_only=True)
#     ]
#     history = model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=callbacks_list)
#     return history

# # Evaluate model and extract confusion matrix details
# def evaluate_model(model, test_ds, class_names):
#     y_true, y_pred = [], []
#     for images, labels in test_ds:
#         y_true.extend(labels.numpy())
#         preds = model.predict(images)
#         y_pred.extend(np.argmax(preds, axis=1))
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     cm = confusion_matrix(y_true, y_pred)
    
#     # Calculate per-class metrics
#     cm_details = {}
#     for i, class_name in enumerate(class_names):
#         tp = cm[i, i]
#         fp = cm[:, i].sum() - tp
#         fn = cm[i, :].sum() - tp
#         tn = cm.sum() - (tp + fp + fn)
#         cm_details[class_name] = {
#             'True Positives': int(tp),
#             'False Positives': int(fp),
#             'False Negatives': int(fn),
#             'True Negatives': int(tn)
#         }
    
#     return {
#         'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
#         'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
#         'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
#         'cm': cm,
#         'cm_details': cm_details
#     }

# # Plot confusion matrix
# def plot_confusion_matrix(cm, classes, title):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.title(title)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     file_path = f'{title.lower().replace(" ", "_")}.png'
#     directory = os.path.dirname(file_path)
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory)
#     plt.savefig(file_path)
#     plt.close()

# # Main loop for each series
# series_types = ['Sagittal T1', 'Axial T2', 'Sagittal T2/STIR']
# class_names = ['normal_mild', 'moderate', 'severe']

# for series_name in series_types:
#     series_df = final_merged_df[final_merged_df['series_description'] == series_name].copy()
#     if series_df.empty:
#         print(f"No valid data for {series_name}.")
#         continue
    
#     # Split data
#     train_df, temp_df = train_test_split(series_df, test_size=0.3, stratify=series_df['severity'], random_state=42)
#     val_df, test_df = train_test_split(temp_df, test_size=0.6667, stratify=temp_df['severity'], random_state=42)
    
#     print(f"\nClass distribution for {series_name}:")
#     print("Train:", train_df['severity'].value_counts().to_dict())
#     print("Validation:", val_df['severity'].value_counts().to_dict())
#     print("Test:", test_df['severity'].value_counts().to_dict())
    
#     # Create datasets
#     train_ds = create_dataset(train_df, batch_size=32, is_training=True)
#     val_ds = create_dataset(val_df, batch_size=32)
#     test_ds = create_dataset(test_df, batch_size=32)
    
#     # Build and train model
#     model = build_unet_encoder()
#     history = train_model(model, train_ds, val_ds, series_name)
    
#     # Evaluate
#     results = evaluate_model(model, test_ds, class_names)
#     print(f"\nMetrics for {series_name}:")
#     print(f"Precision: {results['precision']:.4f}")
#     print(f"Recall: {results['recall']:.4f}")
#     print(f"F1-Score: {results['f1']:.4f}")
    
#     # Print confusion matrix details
#     print(f"\nConfusion Matrix Details for {series_name}:")
#     for class_name, metrics in results['cm_details'].items():
#         print(f"\nClass: {class_name}")
#         for metric_name, value in metrics.items():
#             print(f"{metric_name}: {value}")
            
#     # Plot confusion matrix
#     plot_confusion_matrix(results['cm'], class_names, f'{series_name} Confusion Matrix')
    
#     # Plot training metrics
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title(f'{series_name} Loss')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Train Acc')
#     plt.plot(history.history['val_accuracy'], label='Val Acc')
#     plt.title(f'{series_name} Accuracy')
#     plt.legend()
#     file_path = f'{series_name.lower().replace(" ", "_")}_plots.png'
#     directory = os.path.dirname(file_path)
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory)
#     plt.savefig(file_path)
#     plt.close()






# !pip install pydicom
# !pip install matplotlib seaborn scikit-learn
# import os
# import numpy as np
# import pandas as pd
# import cv2
# import pydicom
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers, callbacks
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.utils.class_weight import compute_class_weight
# # Set up mixed precision for better performance
# tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# # Define dataset path
# train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

# # Load CSV files
# train = pd.read_csv(os.path.join(train_path, 'train.csv'))
# label = pd.read_csv(os.path.join(train_path, 'train_label_coordinates.csv'))
# train_desc = pd.read_csv(os.path.join(train_path, 'train_series_descriptions.csv'))

# # Reshape train.csv to separate conditions, levels, and severities
# def reshape_row(row):
#     data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
#     for column, value in row.items():
#         if column != 'study_id':
#             parts = column.split('_')
#             condition = ' '.join([word.capitalize() for word in parts[:-2]])
#             level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
#             data['study_id'].append(row['study_id'])
#             data['condition'].append(condition)
#             data['level'].append(level)
#             data['severity'].append(value)
#     return pd.DataFrame(data)

# # Create and merge DataFrames
# new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)
# merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
# final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')

# # Create image paths
# final_merged_df['image_path'] = (
#     train_path + 'train_images/' +
#     final_merged_df['study_id'].astype(str) + '/' +
#     final_merged_df['series_id'].astype(str) + '/' +
#     final_merged_df['instance_number'].astype(str) + '.dcm'
# )

# # Map severity labels
# final_merged_df['severity'] = final_merged_df['severity'].map({
#     'Normal/Mild': 'normal_mild',
#     'Moderate': 'moderate',
#     'Severe': 'severe'
# })

# # Filter out invalid rows
# final_merged_df = final_merged_df[final_merged_df['severity'].isin(['normal_mild', 'moderate', 'severe'])]

# # Compute class weights for imbalanced classes
# class_counts = final_merged_df['severity'].value_counts().sort_index().values
# class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), 
#                                     y=final_merged_df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}))
# class_weights_dict = dict(enumerate(class_weights))

# # Define Focal Loss
# def focal_loss(y_true, y_pred, alpha=list(class_weights_dict.values()), gamma=2.0):
#     y_true = tf.cast(y_true, tf.int32)
#     ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
#     probs = tf.nn.softmax(y_pred, axis=-1)
#     probs = tf.gather(probs, y_true, batch_dims=1)
#     alpha = tf.gather(alpha, y_true)
#     modulating_factor = tf.pow(1.0 - probs, gamma)
#     return tf.reduce_mean(alpha * modulating_factor * ce)

# # Image preprocessing functions
# def apply_clahe(image, clip_limit=2, tile_grid_size=(16, 16)):
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8) if np.issubdtype(image.dtype, np.floating) else image.astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     return clahe.apply(image)

# def apply_bilateral_filter(image, diameter=5, sigma_color=10, sigma_space=10):
#     if image.dtype == np.float32 or image.dtype == np.float64:
#         if image.max() <= 1.0:
#             sigma_color = sigma_color / 255.0
#         else:
#             image = (image / 255.0).astype(np.float32)
#     elif image.dtype != np.uint8:
#         image = image.astype(np.uint8)
#     return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

# def augment_image(image):
#     image = tf.image.random_brightness(image, max_delta=5/255.0)
#     image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
#     image_uint8 = tf.image.convert_image_dtype(image, tf.uint8)
#     image_uint8 = tf.image.random_jpeg_quality(image_uint8, 80, 100)
#     return tf.image.convert_image_dtype(image_uint8, tf.float32)

# def remove_background(image):
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8) if image.dtype == np.float32 else image.astype(np.uint8)
#     _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return cv2.bitwise_and(image, image, mask=mask)

# def load_dicom_tf(path):
#     dicom = pydicom.dcmread(path.numpy().decode('utf-8'))
#     data = dicom.pixel_array
#     data = data - np.min(data)
#     if np.max(data) != 0:
#         data = data / np.max(data)
#     return data.astype(np.float32)

# def preprocess_image(image, label=None, is_training=False):
#     def process_with_opencv(img_tensor):
#         img_np = img_tensor.numpy().squeeze()
#         img_np = apply_bilateral_filter(img_np)
#         img_np = apply_clahe(img_np)
#         img_np = remove_background(img_np)
#         if is_training:
#             img_np = augment_image(tf.expand_dims(img_np, axis=-1)).numpy().squeeze()
#         img_np = np.expand_dims(img_np, axis=-1)
#         return img_np.astype(np.float32)

#     image = tf.py_function(load_dicom_tf, [image], tf.float32)
#     image.set_shape([None, None])
#     image = tf.py_function(process_with_opencv, [image], tf.float32)
#     image.set_shape([None, None, 1])
#     image = tf.image.resize(image, [224, 224])
#     image = tf.image.grayscale_to_rgb(image)
#     image = tf.keras.applications.resnet50.preprocess_input(image)
#     return (image, label) if label is not None else image

# # Create dataset
# def create_dataset(df, batch_size=32, is_training=False):
#     labels = df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}).astype(np.int32)
#     dataset = tf.data.Dataset.from_tensor_slices((df['image_path'], labels))
#     dataset = dataset.map(lambda x, y: preprocess_image(x, y, is_training), num_parallel_calls=tf.data.AUTOTUNE)
    
#     if is_training:
#         dataset = dataset.apply(tf.data.experimental.rejection_resample(
#             class_func=lambda image, label: label,
#             target_dist=list(class_weights_dict.values()),
#             initial_dist=list(class_weights_dict.values())
#         )).map(lambda resampled_label, original_sample: original_sample)
    
#     dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return dataset

# # Build U-Net encoder
# def build_unet_encoder(input_shape=(224, 224, 3), num_classes=3):
#     inputs = layers.Input(shape=input_shape)
#     c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
#     c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
#     p1 = layers.MaxPooling2D((2, 2))(c1)
    
#     c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
#     c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
#     p2 = layers.MaxPooling2D((2, 2))(c2)
    
#     c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
#     c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
#     p3 = layers.MaxPooling2D((2, 2))(c3)
    
#     c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
#     c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(c4)
#     p4 = layers.MaxPooling2D((2, 2))(c4)
    
#     c5 = layers.Conv2D(1024, 3, padding='same', activation='relu')(p4)
#     c5 = layers.Conv2D(1024, 3, padding='same', activation='relu')(c5)
    
#     x = layers.GlobalAveragePooling2D()(c5)
#     x = layers.Dense(512, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(num_classes, activation='linear')(x)
    
#     return models.Model(inputs, outputs)

# # Train model
# def train_model(model, train_ds, val_ds, series_name):
#     model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
#                   loss=focal_loss, 
#                   metrics=['accuracy'])
#     callbacks_list = [
#         callbacks.EarlyStopping(patience=10, restore_best_weights=True),
#         callbacks.ModelCheckpoint(f'best_{series_name}_unet.keras', save_best_only=True)
#     ]
#     history = model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=callbacks_list)
#     return history

# # Evaluate model
# def evaluate_model(model, test_ds):
#     y_true, y_pred = [], []
#     for images, labels in test_ds:
#         y_true.extend(labels.numpy())
#         preds = model.predict(images)
#         y_pred.extend(np.argmax(preds, axis=1))
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return {
#         'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
#         'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
#         'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
#         'cm': confusion_matrix(y_true, y_pred)
#     }

# # Plot confusion matrix
# def plot_confusion_matrix(cm, classes, title):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.title(title)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     file_path = f'{title.lower().replace(" ", "_")}.png'
#     directory = os.path.dirname(file_path)
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory)
#     plt.savefig(file_path)
#     plt.close()

# # Main loop for each series
# series_types = ['Sagittal T1', 'Axial T2', 'Sagittal T2/STIR']
# class_names = ['normal_mild', 'moderate', 'severe']

# for series_name in series_types:
#     series_df = final_merged_df[final_merged_df['series_description'] == series_name].copy()
#     if series_df.empty:
#         print(f"No valid data for {series_name}.")
#         continue
    
#     # Split data
#     train_df, temp_df = train_test_split(series_df, test_size=0.3, stratify=series_df['severity'], random_state=42)
#     val_df, test_df = train_test_split(temp_df, test_size=0.6667, stratify=temp_df['severity'], random_state=42)
    
#     print(f"\nClass distribution for {series_name}:")
#     print("Train:", train_df['severity'].value_counts().to_dict())
#     print("Validation:", val_df['severity'].value_counts().to_dict())
#     print("Test:", test_df['severity'].value_counts().to_dict())
    
#     # Create datasets
#     train_ds = create_dataset(train_df, batch_size=32, is_training=True)
#     val_ds = create_dataset(val_df, batch_size=32)
#     test_ds = create_dataset(test_df, batch_size=32)
    
#     # Build and train model
#     model = build_unet_encoder()
#     history = train_model(model, train_ds, val_ds, series_name)
    
#     # Evaluate
#     results = evaluate_model(model, test_ds)
#     print(f"\nMetrics for {series_name}:")
#     print(f"Precision: {results['precision']:.4f}")
#     print(f"Recall: {results['recall']:.4f}")
#     print(f"F1-Score: {results['f1']:.4f}")


#     # Plot confusion matrix
#     plot_confusion_matrix(results['cm'], class_names, f'{series_name} Confusion Matrix')
    
#     # Plot training metrics
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title(f'{series_name} Loss')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Train Acc')
#     plt.plot(history.history['val_accuracy'], label='Val Acc')
#     plt.title(f'{series_name} Accuracy')
#     plt.legend()
#     file_path = f'{series_name.lower().replace(" ", "_")}_plots.png'
#     directory = os.path.dirname(file_path)
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory)
#     plt.savefig(file_path)
#     plt.close()




