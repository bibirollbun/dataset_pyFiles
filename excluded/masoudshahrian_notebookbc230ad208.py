!pip install pydicom
!pip install matplotlib seaborn scikit-learn
# !pip install tensorflow_addons


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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Print session info
print(f"Training started at: 2025-06-05 16:52:20 UTC")
print(f"User: masoudshahrian")

# Create output directories
os.makedirs('model_results', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# Define dataset path
train_path = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/'

def load_and_process_data(train_path):
    print("Loading and processing data...")
    
    train = pd.read_csv(os.path.join(train_path, 'train.csv'))
    label = pd.read_csv(os.path.join(train_path, 'train_label_coordinates.csv'))
    train_desc = pd.read_csv(os.path.join(train_path, 'train_series_descriptions.csv'))
    
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
    
    new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)
    merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
    final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')
    
    final_merged_df['image_path'] = (
        train_path + 'train_images/' +
        final_merged_df['study_id'].astype(str) + '/' +
        final_merged_df['series_id'].astype(str) + '/' +
        final_merged_df['instance_number'].astype(str) + '.dcm'
    )
    
    final_merged_df['severity'] = final_merged_df['severity'].map({
        'Normal/Mild': 'normal_mild',
        'Moderate': 'moderate',
        'Severe': 'severe'
    })
    
    valid_data = final_merged_df[final_merged_df['severity'].isin(['normal_mild', 'moderate', 'severe'])]
    print(f"Total valid samples: {len(valid_data)}")
    
    return valid_data

@tf.function
def load_dicom_tf(path):
    try:
        dicom = pydicom.dcmread(path.numpy().decode('utf-8'))
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = np.expand_dims(data, axis=-1)
        return data.astype(np.float32)
    except Exception as e:
        print(f"Error loading DICOM: {e}")
        return np.zeros((224, 224, 1), dtype=np.float32)

@tf.function
def rotate_image(image, angle):
    # Convert angle from radians to degrees
    angle_deg = angle * 180.0 / np.pi
    
    # Rotate image using tf.image
    rotated = tf.image.rot90(
        image,
        k=tf.cast(angle_deg / 90, tf.int32)
    )
    return rotated

@tf.function
def preprocess_image(image_path, label=None, is_training=False):
    # Load and preprocess image
    image = tf.py_function(load_dicom_tf, [image_path], tf.float32)
    image.set_shape([None, None, 1])
    
    # Resize image
    image = tf.image.resize(image, [224, 224])
    image = tf.image.grayscale_to_rgb(image)
    
    if is_training:
        # Data augmentation for training
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Add random rotation
        random_angle = tf.random.uniform([], -0.5, 0.5)  # Random angle between -0.5 and 0.5 radians
        image = rotate_image(image, random_angle)
    
    # Preprocess for EfficientNet
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    
    return (image, label) if label is not None else image

def create_dataset(df, batch_size=32, is_training=False):
    labels = df['severity'].map({'normal_mild': 0, 'moderate': 1, 'severe': 2}).astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((df['image_path'], labels))
    
    dataset = dataset.map(
        lambda x, y: preprocess_image(x, y, is_training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if is_training:
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_hybrid_model(input_shape=(224, 224, 3), num_classes=3):
    efficientnet = EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    for layer in efficientnet.layers[:100]:
        layer.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = efficientnet(inputs)
    
    attention = layers.Conv2D(x.shape[-1], 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)
    
    return models.Model(inputs, outputs)

def focal_loss(gamma=2., alpha=4.):
    def focal_loss_with_logits(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        probs = tf.nn.softmax(y_pred)
        probs = tf.gather(probs, y_true, batch_dims=1)
        return tf.reduce_mean(alpha * tf.pow(1. - probs, gamma) * ce)
    return focal_loss_with_logits

def train_model(model, train_ds, val_ds, series_name, epochs=100, steps_per_epoch=100):
    optimizer = optimizers.Adam(learning_rate=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(),
        metrics=['accuracy']
    )
    
    log_dir = f"logs/{series_name}_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=30,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history

def evaluate_model(model, test_ds, class_names):
    y_true = []
    y_pred = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        predictions = model.predict(images)
        y_pred_probs.extend(tf.nn.softmax(predictions).numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)
    
    cm = confusion_matrix(y_true, y_pred)
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        true_class = (y_true == i)
        pred_class = (y_pred == i)
        
        tp = np.sum((true_class) & (pred_class))
        fp = np.sum((~true_class) & (pred_class))
        fn = np.sum((true_class) & (~pred_class))
        tn = np.sum((~true_class) & (~pred_class))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        class_metrics[class_name] = {
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'F1-Score': f1
        }
    
    overall_metrics = {
        'accuracy': np.mean(y_pred == y_true),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted'),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    return {
        'confusion_matrix': cm,
        'class_metrics': class_metrics,
        'overall_metrics': overall_metrics,
        'predictions': y_pred_probs
    }

def plot_results(history, results, class_names, series_name):
    output_dir = 'model_results'
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 15))
    
    # Training History - Loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{series_name} - Training Loss', pad=20)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Training History - Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{series_name} - Training Accuracy', pad=20)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Confusion Matrix with percentages
    plt.subplot(2, 2, 3)
    cm = results['confusion_matrix']
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    
    sns.heatmap(cm, annot=np.array([[f'{int(x)}\n({y:.1f}%)' for x, y in zip(row_true, row_perc)] 
                                   for row_true, row_perc in zip(cm, cm_perc)]),
                fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{series_name} - Confusion Matrix\nwith Counts and Percentages', pad=20)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Metrics Table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    table_data = []
    table_colors = []
    metrics_display = ['Sensitivity', 'Specificity', 'Precision', 'F1-Score']
    
    header = ['Metrics'] + class_names + ['Overall']
    table_data.append(header)
    table_colors.append(['lightgray'] * len(header))
    
    for metric in metrics_display:
        row = [metric]
        for class_name in class_names:
            value = results['class_metrics'][class_name][metric]
            row.append(f'{value:.3f}')
        if metric == 'Precision':
            row.append(f'{results["overall_metrics"]["weighted_precision"]:.3f}')
        elif metric == 'Sensitivity':
            row.append(f'{results["overall_metrics"]["weighted_recall"]:.3f}')
        elif metric == 'F1-Score':
            row.append(f'{results["overall_metrics"]["weighted_f1"]:.3f}')
        else:
            row.append('-')
        table_data.append(row)
        table_colors.append(['white'] * len(header))
    
    acc_row = ['Accuracy'] + ['-'] * len(class_names) + [f'{results["overall_metrics"]["accuracy"]:.3f}']
    table_data.append(acc_row)
    table_colors.append(['white'] * len(header))
    
    table = plt.table(cellText=table_data,
                     cellColours=table_colors,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title(f'{series_name} - Performance Metrics', pad=20)
    
    plt.tight_layout(h_pad=1.0, w_pad=1.0)
    safe_series_name = series_name.lower().replace("/", "_").replace(" ", "_")
    output_file = os.path.join(output_dir, f'{safe_series_name}_detailed_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed metrics
    print(f"\nDetailed Results for {series_name}:")
    print("=" * 50)
    print("Overall Metrics:")
    for metric, value in results['overall_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPer-Class Metrics:")
    for class_name in class_names:
        print(f"\n{class_name}:")
        metrics = results['class_metrics'][class_name]
        print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}, TN: {metrics['TN']}")
        print(f"Sensitivity: {metrics['Sensitivity']:.4f}")
        print(f"Specificity: {metrics['Specificity']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"F1-Score: {metrics['F1-Score']:.4f}")

def main():
    final_merged_df = load_and_process_data(train_path)
    
    series_types = ['Sagittal T1', 'Axial T2', 'Sagittal T2/STIR']
    class_names = ['normal_mild', 'moderate', 'severe']
    
    for series_name in series_types:
        print(f"\nProcessing {series_name}")
        
        series_df = final_merged_df[final_merged_df['series_description'] == series_name].copy()
        if series_df.empty:
            print(f"No data for {series_name}")
            continue
        
        train_df, temp_df = train_test_split(series_df, test_size=0.3, stratify=series_df['severity'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['severity'])
        
        print(f"\nDataset splits for {series_name}:")
        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        train_ds = create_dataset(train_df, batch_size=32, is_training=True)
        val_ds = create_dataset(val_df, batch_size=32)
        test_ds = create_dataset(test_df, batch_size=32)
        
        model = build_hybrid_model()
        history = train_model(model, train_ds, val_ds, series_name)
        
        results = evaluate_model(model, test_ds, class_names)
        plot_results(history, results, class_names, series_name)
        
        safe_series_name = series_name.lower().replace("/", "_").replace(" ", "_")
        model_file = os.path.join('saved_models', f'{safe_series_name}_model.h5')
        model.save(model_file)
        
        print(f"Completed {series_name} at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

if __name__ == "__main__":
    main()











# cause of low efficiency commented for improving with sonet3.5
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
#     image = tf.keras.applications.efficientnet.preprocess_input(image)
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
# def build_unet_encoder(input_shape=(224, 224, 3)):
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
    
#     return models.Model(inputs, c5)

# # Build hybrid U-Net + EfficientNetB1 model
# def build_hybrid_model(input_shape=(224, 224, 3), num_classes=3):
#     unet_encoder = build_unet_encoder(input_shape)
#     efficientnet = EfficientNetB1(
#         include_top=False,
#         weights='imagenet',
#         input_shape=input_shape
#     )
    
#     for layer in efficientnet.layers:
#         if 'block6' in layer.name or 'block7' in layer.name:
#             layer.trainable = True
#         else:
#             layer.trainable = False
    
#     inputs = layers.Input(shape=input_shape)
#     unet_features = unet_encoder(inputs)
#     eff_features = efficientnet(inputs)
    
#     unet_pooled = layers.GlobalAveragePooling2D()(unet_features)
#     eff_pooled = layers.GlobalAveragePooling2D()(eff_features)
#     combined = layers.Concatenate()([unet_pooled, eff_pooled])
    
#     x = layers.Dense(512, activation='relu')(combined)
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
#         callbacks.ModelCheckpoint(f'best_{series_name}_hybrid.keras', save_best_only=True)
#     ]
#     history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks_list)
#     return history

# # Evaluate model
# def evaluate_model(model, test_ds, class_names):
#     y_true, y_pred = [], []
#     for images, labels in test_ds:
#         y_true.extend(labels.numpy())
#         preds = model.predict(images)
#         y_pred.extend(np.argmax(preds, axis=1))
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     cm = confusion_matrix(y_true, y_pred)
    
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
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=classes, yticklabels=classes,
#                 annot_kws={"size": 12}, cbar=True)
#     plt.title(title, fontsize=14, pad=15)
#     plt.xlabel('Predicted Label', fontsize=12)
#     plt.ylabel('True Label', fontsize=12)
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#     plt.tight_layout()
    
#     # Save the plot
#     file_path = f'confusion_matrix_{title.lower().replace(" ", "_")}.png'
#     directory = os.path.dirname(file_path)
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory)
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     plt.show()  # Display the plot
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
    
#     # Build and train hybrid model
#     model = build_hybrid_model()
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
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Train Acc')
#     plt.plot(history.history['val_accuracy'], label='Val Acc')
#     plt.title(f'{series_name} Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     file_path = f'{series_name.lower().replace(" ", "_")}_plots.png'
#     directory = os.path.dirname(file_path)
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory)
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()




