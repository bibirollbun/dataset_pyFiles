# Standard Library Imports
import os
import math
import random

# Third-Party Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from tabulate import tabulate
from tqdm.notebook import tqdm
import h5py
import cv2
from PIL import Image, ImageEnhance, ImageOps

# TensorFlow and Keras Imports
import tensorflow as tf
from tensorflow.keras import layers
import keras_cv
import keras


metadata = pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv", low_memory=False)

print("---------------------------------------Shape-------------------------------------------------")
print(metadata.shape)
print("---------------------------------------Info--------------------------------------------------")
print(metadata.info())
print("---------------------------------------Describe----------------------------------------------")
print(metadata.describe())
print("---------------------------------------Missing Data------------------------------------------")
print(metadata.isnull().sum())


# Count class instances before oversampling
class_counts = metadata['target'].value_counts()

# Plot class distribution before oversampling
plt.figure(figsize=(6, 4))
plt.bar(class_counts.index, class_counts.values, color=['blue', 'green'])
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution Before Oversampling")
plt.show()


# Define numerical columns for outlier detection
numeric_cols = [
    'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio',
    'tbp_lv_minorAxisMM', 'tbp_lv_perimeterMM', 'tbp_lv_deltaLBnorm',
    'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_radial_color_std_max',
    'tbp_lv_symm_2axis', "tbp_lv_color_std_mean", "tbp_lv_nevi_confidence"
]
# Compute Z-scores
z_scores = np.abs(stats.zscore(metadata[numeric_cols]))

# Set threshold (e.g., values greater than 3 std deviations)
threshold = 3
outliers_zscore = metadata[(z_scores > threshold).any(axis=1)]

print(f"Number of outliers detected using Z-Score: {len(outliers_zscore)}")


# Plot histograms for all selected numerical features
plt.figure(figsize=(15, 12))

for i, col in enumerate(numeric_cols):
    plt.subplot(4, 4, i+1)  # Adjust layout based on the number of features
    sns.histplot(metadata[col], bins=50, kde=True)
    plt.axvline(metadata[col].mean(), color='red', linestyle='dashed', label='Mean')
    plt.title(col)
    plt.legend()

plt.tight_layout()
plt.show()


class TrainingConfig:
    """
    Configuration class for training and model setup.
    """
    verbose = 1  
    seed = 42 
    batch_size = 32
    majority_ratio = 0.7
    minority_ratio = 0.3
    preset = "efficientnetv2_b2_imagenet"
    image_size = [100, 100]
    lr_mode = "cos"  # Learning rate scheduler mode: "cos", "step", or "exp"
    class_names = ['target']
    num_classes = 1
    sample_size = 0.3


# Categorical features which will be one hot encoded
CATEGORICAL_COLUMNS = ["sex", "anatom_site_general",
            "tbp_tile_type","tbp_lv_location", "tbp_lv_location_simple"]

# Numeraical features which will be normalized
NUMERIC_COLUMNS = [
    'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio',
    'tbp_lv_minorAxisMM', 'tbp_lv_perimeterMM', 'tbp_lv_deltaLBnorm',
    'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_radial_color_std_max',
    'tbp_lv_symm_2axis', "tbp_lv_color_std_mean", "tbp_lv_nevi_confidence"]

# Tabular feature columns
FEAT_COLS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def preprocess_metadata(metadata, minority_ratio, majority_ratio, seed, train=True, encoder=None, scaler=None):
    """
    Preprocess the metadata for the ISIC 2024 challenge.
    """
    
    # Remove unnecessary columns (only for training data)
    if train:
        columns_to_drop = [
            'lesion_id', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 'mel_mitotic_index',
            'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence', 'iddx_1', 'iddx_full',
            'patient_id', 'image_type', 'attribution', 'copyright_license'
        ]
        metadata = metadata.drop(columns=columns_to_drop)

        # Filter out rows with missing values in key columns (only for training data)
        metadata = metadata.dropna(subset=['age_approx', 'anatom_site_general', 'sex'])

        # Filter biologically implausible age values (only for training data)
        metadata = metadata[(metadata['age_approx'] >= 5) & (metadata['age_approx'] <= 100)]
    else:
        columns_to_drop = ['patient_id', 'image_type', 'attribution', 'copyright_license']
        metadata = metadata.drop(columns=columns_to_drop)
        
    # Apply Winsorization to lesion size-related features
    size_cols = ['clin_size_long_diam_mm', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM', 'tbp_lv_perimeterMM']
    for col in size_cols:
        metadata[col] = winsorize(metadata[col], limits=[0.01, 0.01])  # Cap at 1st and 99th percentile

    # Apply RobustScaler to features where outliers may be informative
    scale_cols = [
        'tbp_lv_area_perim_ratio', 'tbp_lv_deltaLBnorm', 'tbp_lv_norm_border',
        'tbp_lv_norm_color', 'tbp_lv_radial_color_std_max', 'tbp_lv_symm_2axis'
    ]
    r_scaler = RobustScaler()
    metadata[scale_cols] = r_scaler.fit_transform(metadata[scale_cols])
    
    # Hot encode categorical features
    if train:
        # Fit the OneHotEncoder on the training data
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_categories = encoder.fit_transform(metadata[CATEGORICAL_COLUMNS])
        encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS), index=metadata.index)
    else:
        # Transform the test data using the encoder fitted on the training data
        encoded_categories = encoder.transform(metadata[CATEGORICAL_COLUMNS])
        encoded_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS), index=metadata.index)
        
    # Drop the original categorical columns and concatenate the encoded ones
    metadata = metadata.drop(columns=CATEGORICAL_COLUMNS)
    metadata = pd.concat([metadata, encoded_df], axis=1)
    
    # Discritize age feature
    metadata['age_approx'] = pd.cut(metadata['age_approx'], bins=5, labels=False)
    metadata['age_approx'] = metadata['age_approx'].astype(float)
    
    # Normalize numerical features
    if train:
        scaler = MinMaxScaler()
        metadata[NUMERIC_COLUMNS[1:]] = scaler.fit_transform(metadata[NUMERIC_COLUMNS[1:]])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for test data preprocessing.")
        metadata[NUMERIC_COLUMNS[1:]] = scaler.transform(metadata[NUMERIC_COLUMNS[1:]])

    # Balance the dataset by sampling minority and majority classes (only for training data)
    if train:
        original_minority_ratio = metadata['target'].mean()
        minority_class = metadata.query("target == 1").sample(
            frac=minority_ratio / original_minority_ratio, random_state=seed, replace=True
        )
        majority_class = metadata.query("target == 0").sample(
            frac=majority_ratio, random_state=seed
        )
        metadata = pd.concat([minority_class, majority_class], axis=0).reset_index(drop=True)

    return metadata, encoder, scaler


metadata = pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv", low_memory=False)
df, t_encoder, t_scaler = preprocess_metadata(metadata, TrainingConfig.minority_ratio, TrainingConfig.majority_ratio, TrainingConfig.seed)


# Count class instances after oversampling
class_counts = df['target'].value_counts()

# Plot class distribution after oversampling
plt.figure(figsize=(6, 4))
plt.bar(class_counts.index, class_counts.values, color=['blue', 'green'])
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution After Oversampling")
plt.show()

# Display the first few rows of the preprocessed metadata
df.head()


BASE_PATH = "isic-2024-challenge"
training_validation_hdf5 = h5py.File(f"/kaggle/input/isic-2024-challenge/train-image.hdf5", 'r')
testing_hdf5 = h5py.File(f"/kaggle/input/isic-2024-challenge/test-image.hdf5", 'r')


isic_id = metadata['isic_id'].iloc[0]

# Image as Byte String
byte_string = training_validation_hdf5[isic_id][()]
print(f"Byte String: {byte_string[:20]}....")

# Convert byte string to numpy array
nparr = np.frombuffer(byte_string, np.uint8)
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,::-1] # reverse last axis for bgr -> rgb
image_size = image.shape[:2]

# Compute image statistics
pixel_mean = image.mean()
pixel_std = image.std()
pixel_min = image.min()
pixel_max = image.max()

# Display the image with properties
plt.figure(figsize=(4, 4))
plt.imshow(image)
plt.title(f"Size: {image_size}\n" f"Pixel Mean: {pixel_mean:.2f}, Std: {pixel_std:.2f}\n" f"Min: {pixel_min}, Max: {pixel_max}")
plt.show()


# Initialize GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=TrainingConfig.seed)

# Random sample of the data
df_sample = df.sample(frac=TrainingConfig.sample_size, random_state=TrainingConfig.seed)

# Split the data
train_idx, val_idx = next(gss.split(df_sample, groups=df_sample['isic_id']))

# Create training and validation DataFrames
training_df = df_sample.iloc[train_idx]
validation_df = df_sample.iloc[val_idx]

print(f"# Num Train: {len(training_df)} | Num Valid: {len(validation_df)}")


def buid_augmenter():
    """
    Build an image augmenter using the Keras Sequential API.
    """
    aug_layers = [
        keras_cv.layers.RandomFlip("horizontal"),
        keras_cv.layers.RandomCutout(height_factor=(0.02, 0.06), width_factor=(0.02, 0.06)),
        keras_cv.layers.RandomContrast(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomBrightness(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
    ]

    # Apply augmentations to random samples
    aug_layers = [keras_cv.layers.RandomApply(x, rate=0.5) for x in aug_layers]
    
    # Build the augment layer
    augmenter = keras_cv.layers.Augmenter(aug_layers)
    
    # Apply augmentations
    def augment(inp, label):
        images = inp["images"]
        aug_data = {"images": images}
        aug_data = augmenter(aug_data)
        inp["images"] = aug_data["images"]
        return inp, label
    return augment


def build_decoder(with_labels=True, target_size=TrainingConfig.image_size):
    """
    Create a function to decode images and optionally labels.
    """
    def decode_image(inp):
        # Read jpeg image
        file_bytes = inp["images"]
        image = tf.io.decode_jpeg(file_bytes)
        
        # Resize
        image = tf.image.resize(image, size=target_size, method="area")
        
        # Normalize
        image = tf.cast(image, tf.float32)
        image /= 255.0
        
        # Reshape
        image = tf.reshape(image, [*target_size, 3])
        
        inp["images"] = image
        return inp

    def decode_label(label, num_classes):
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [num_classes])
        return label

    def decode_with_labels(inp, label=None):
        inp = decode_image(inp)
        label = decode_label(label, num_classes=TrainingConfig.num_classes)
        return (inp, label)

    return decode_with_labels if with_labels else decode_image


def build_dataset(
    isic_ids,
    hdf5,
    features,
    labels=None,
    batch_size=32,
    decode_fn=None,
    augment_fn=None,
    augment=False,
    shuffle=1024,
    cache=True,
    drop_remainder=False,
):
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = buid_augmenter()

    AUTO = tf.data.experimental.AUTOTUNE

    images = [None]*len(isic_ids)
    for i, isic_id in enumerate(tqdm(isic_ids, desc="Loading Images ")):
        images[i] = hdf5[isic_id][()]
        
    features = np.array(features, dtype=np.float32)
    
    inp = {"images": images, "features": features}
    slices = (inp, labels) if labels is not None else inp

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.cache() if cache else ds
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    if shuffle:
        ds = ds.shuffle(shuffle, seed=TrainingConfig.seed)
        opt = tf.data.Options()
        opt.deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds


# Train
print("# Training:")
training_features = np.array(training_df.drop(columns=["isic_id", "target"]).values.tolist())
training_ids = training_df.isic_id.values
training_labels = training_df.target.values
training_ds = build_dataset(training_ids, training_validation_hdf5, training_features, 
                         training_labels, batch_size=TrainingConfig.batch_size,
                         shuffle=True, augment=True)

# Valid
print("# Validation:")
validation_features = np.array(validation_df.drop(columns=["isic_id", "target"]).values.tolist())
validation_ids = validation_df.isic_id.values
validation_labels = validation_df.target.values
validation_ds = build_dataset(validation_ids, training_validation_hdf5, validation_features,
                         validation_labels, batch_size=TrainingConfig.batch_size,
                         shuffle=False, augment=False)


print("Training Dataset:")
for x, y in training_ds.take(1):
    print("images.shape:", x["images"].shape)
    feat = x["features"]
    print("features.shape:", feat.shape)
    print("features.dtype:", feat.dtype)
    print("label.shape:", y.shape)

print("\nValidation Dataset:")  
for x, y in validation_ds.take(1):
    print("images.shape:", x["images"].shape)
    feat = x["features"]
    print("features.shape:", feat.shape)
    print("features.dtype:", feat.dtype)
    print("label.shape:", y.shape)


training_ds = training_ds.map(
    lambda x, y: ({"images": x["images"],
                   "features": x["features"]}, y), num_parallel_calls=tf.data.AUTOTUNE)

validation_ds = validation_ds.map(
    lambda x, y: ({"images": x["images"],
                   "features": x["features"]}, y), num_parallel_calls=tf.data.AUTOTUNE)


for images, labels in training_ds.take(1):
    print(f"Images Shape: {images['images'].shape}")
    print(f"Labels Shape: {labels.shape}")
    break

image = images['images'][0].numpy()
image_size = image.shape[:2]

# Compute image statistics
pixel_mean = image.mean()
pixel_std = image.std()
pixel_min = image.min()
pixel_max = image.max()

# Display the image with properties
plt.figure(figsize=(4, 4))
plt.imshow(image)
plt.title(f"Size: {image_size}\n" f"Pixel Mean: {pixel_mean:.2f}, Std: {pixel_std:.2f}\n" f"Min: {pixel_min}\n" f"Max: {pixel_max}")
plt.show()


image_input = keras.Input(shape=(*TrainingConfig.image_size, 3), name="images")
feat_input = keras.Input(shape=(len(df.columns) - 2,), name="features")
inp = {"images":image_input, "features":feat_input}

# Branch for image input
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(TrainingConfig.preset)
x1 = backbone(image_input)
x1 = keras.layers.GlobalAveragePooling2D()(x1)
x1 = keras.layers.Dropout(0.2)(x1)

# Branch for tabular/feature input
x2 = keras.layers.Dense(96, activation="selu")(feat_input)
x2 = keras.layers.Dense(128, activation="selu")(x2)
x2 = keras.layers.Dropout(0.1)(x2)

# Concatenate both branches
concat = keras.layers.Concatenate()([x1, x2])

# Output layer
out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(concat)

# Build model
model = keras.models.Model(inp, out)

# Compile the model
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.01, from_logits=False),
    metrics=[keras.metrics.AUC(name="auc")]
)

# Model Summary
model.summary()
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, dpi=60)


# Model for Image Only
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(TrainingConfig.preset)
x1 = backbone(image_input)
x1 = keras.layers.GlobalAveragePooling2D()(x1)
x1 = keras.layers.Dropout(0.2)(x1)

# Output layer
out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x1)

# Build model
model_img = keras.models.Model(inp, out)

# Compile the model
model_img.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.01, from_logits=False),
    metrics=[keras.metrics.AUC(name="auc")]
)

# Model Summary
model_img.summary()
keras.utils.plot_model(model_img, show_shapes=True, show_layer_names=True, dpi=60)


# Model for Meta Data Only
x2 = keras.layers.Dense(96, activation="selu")(feat_input)
x2 = keras.layers.Dense(128, activation="selu")(x2)
x2 = keras.layers.Dropout(0.1)(x2)

# Output layer
out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x2)

# Build model
model_meta = keras.models.Model(inp, out)

# Compile the model
model_meta.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.01, from_logits=False),
    metrics=[keras.metrics.AUC(name="auc")]
)

# Model Summary
model_meta.summary()
keras.utils.plot_model(model_meta, show_shapes=True, show_layer_names=True, dpi=60)


def get_lr_callback(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 2.5e-5, 5e-6 * batch_size, 0.8e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('lr')
        plt.title('LR Scheduler')
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create lr callback

lr_cb = get_lr_callback(TrainingConfig.batch_size, mode="exp", plot=True)


ckpt_cb = keras.callbacks.ModelCheckpoint(
    "best_model.keras",   # Filepath where the model will be saved.
    monitor="val_auc",    # Metric to monitor (validation AUC in this case).
    save_best_only=True,  # Save only the model with the best performance.
    save_weights_only=False,  # Save the entire model (not just the weights).
    mode="max",           # The model with the maximum 'val_auc' will be saved.
)
es_cb = keras.callbacks.EarlyStopping(
    monitor="val_auc",   # Monitor validation AUC
    patience=3,          # Number of epochs with no improvement before stopping
    mode="max",          # Stop when 'val_auc' is no longer increasing
    restore_best_weights=True # Restore best weights when stopping
)
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau( 
    monitor="val_auc", 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1 
)


majority_weight = 1 / TrainingConfig.majority_ratio
minority_weight = 1 / TrainingConfig.minority_ratio
class_weights = {0: majority_weight, 1: minority_weight}

# Train the model with class weights
CNN_Model = model.fit(
    training_ds,
    epochs=10,  # Increased number of epochs
    callbacks=[lr_cb, ckpt_cb, es_cb, reduce_lr_cb],
    steps_per_epoch=150,
    validation_data=validation_ds,
    validation_steps=50,  # Reduced validation steps for faster debugging
    verbose=1,
    shuffle=True,
    class_weight=class_weights,
)

# Train the model with class weights
CNN_Model_img = model_img.fit(
    training_ds,
    epochs=10,  # Increased number of epochs
    callbacks=[lr_cb, es_cb, reduce_lr_cb],
    steps_per_epoch=150,
    validation_data=validation_ds,
    validation_steps=50,  # Reduced validation steps for faster debugging
    verbose=1,
    shuffle=True,
    class_weight=class_weights,
)

# Train the model with class weights
Model_meta = model_meta.fit(
    training_ds,
    epochs=10,  # Increased number of epochs
    callbacks=[lr_cb, es_cb, reduce_lr_cb],
    steps_per_epoch=150,
    validation_data=validation_ds,
    validation_steps=50,  # Reduced validation steps for faster debugging
    verbose=1,
    shuffle=True,
    class_weight=class_weights,
)


# plt.plot(CNN_Model.history['auc'], label='train accuracy')
# plt.plot(CNN_Model.history['val_auc'], label = 'test accuracy')
# plt.xlabel('Epoch'); plt.ylabel('Accuracy')
# plt.legend(); plt.show()

# plt.plot(CNN_Model.history['loss'], label='train loss')
# plt.plot(CNN_Model.history['val_loss'], label = 'test loss')


# Extract AUC and validation AUC from history
auc = CNN_Model.history['auc']
val_auc = CNN_Model.history['val_auc']
epochs = range(1, len(auc) + 1)

# Find the epoch with the maximum val_auc
max_val_auc_epoch = np.argmax(val_auc)
max_val_auc = val_auc[max_val_auc_epoch]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, auc, 'o-', label='Training AUC', markersize=5, color='tab:blue')
plt.plot(epochs, val_auc, 's-', label='Validation AUC', markersize=5, color='tab:orange')

# Highlight the max val_auc
plt.scatter(max_val_auc_epoch + 1, max_val_auc, color='red', s=100, label=f'Max Val AUC: {max_val_auc:.4f}')
plt.annotate(f'Max Val AUC: {max_val_auc:.4f}', 
             xy=(max_val_auc_epoch + 1, max_val_auc), 
             xytext=(max_val_auc_epoch + 1 + 0.5, max_val_auc - 0.05),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=12,
             color='tab:red')

# Enhancing the plot
plt.title('AUC over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.xticks(epochs)

# Show the plot
plt.show()


# Extract AUC and validation AUC from history
auc_img = CNN_Model_img.history['auc']
val_auc_img = CNN_Model_img.history['val_auc']
epochs_img = range(1, len(auc_img) + 1)

# Find the epoch with the maximum val_auc
max_val_auc_epoch_img = np.argmax(val_auc_img)
max_val_auc_img = val_auc_img[max_val_auc_epoch_img]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs_img, auc_img, 'o-', label='Training AUC', markersize=5, color='tab:blue')
plt.plot(epochs_img, val_auc_img, 's-', label='Validation AUC', markersize=5, color='tab:orange')

# Highlight the max val_auc
plt.scatter(max_val_auc_epoch_img + 1, max_val_auc_img, color='red', s=100, label=f'Max Val AUC: {max_val_auc_img:.4f}')
plt.annotate(f'Max Val AUC: {max_val_auc_img:.4f}', 
             xy=(max_val_auc_epoch_img + 1, max_val_auc_img), 
             xytext=(max_val_auc_epoch_img + 1 + 0.5, max_val_auc_img - 0.05),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=12,
             color='tab:red')

# Enhancing the plot
plt.title('AUC over Epochs Image Only', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.xticks(epochs_img)

# Show the plot
plt.show()

#############################
# Extract AUC and validation AUC from history
auc_meta = Model_meta.history['auc']
val_auc_meta = Model_meta.history['val_auc']
epochs_meta = range(1, len(auc_meta) + 1)

# Find the epoch with the maximum val_auc
max_val_auc_epoch_meta = np.argmax(val_auc_meta)
max_val_auc_meta = val_auc_meta[max_val_auc_epoch_meta]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs_meta, auc_meta, 'o-', label='Training AUC', markersize=5, color='tab:blue')
plt.plot(epochs_meta, val_auc_meta, 's-', label='Validation AUC', markersize=5, color='tab:orange')

# Highlight the max val_auc
plt.scatter(max_val_auc_epoch_meta + 1, max_val_auc_meta, color='red', s=100, label=f'Max Val AUC: {max_val_auc_meta:.4f}')
plt.annotate(f'Max Val AUC: {max_val_auc_meta:.4f}', 
             xy=(max_val_auc_epoch_meta + 1, max_val_auc_meta), 
             xytext=(max_val_auc_epoch_meta + 1 + 0.5, max_val_auc_meta - 0.05),
             arrowprops=dict(facecolor='black', arrowstyle="->"),
             fontsize=12,
             color='tab:red')

# Enhancing the plot
plt.title('AUC over Epochs Metadata Only', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.xticks(epochs_meta)

# Show the plot
plt.show()


# Best Result
best_score = max(CNN_Model.history['val_auc'])
best_epoch = np.argmax(CNN_Model.history['val_auc']) + 1
print("#" * 10 + " Result " + "#" * 10)
print(f"Best AUC: {best_score:.5f}")
print(f"Best Epoch: {best_epoch}")
print("#" * 28)


model.load_weights("best_model.keras")


test_metadata = pd.read_csv('/kaggle/input/isic-2024-challenge/test-metadata.csv')
testing_df, _, _= preprocess_metadata(test_metadata, TrainingConfig.minority_ratio, TrainingConfig.majority_ratio, TrainingConfig.seed, train=False, encoder=t_encoder, scaler=t_scaler)

# Test
print("# Testing:")
testing_features = np.array(testing_df.drop(columns=["isic_id"]).values.tolist())
testing_ids = testing_df.isic_id.values
testing_ds = build_dataset(testing_ids, testing_hdf5,
                        testing_features, batch_size=TrainingConfig.batch_size,
                         shuffle=False, augment=False, cache=False)
# Apply feature space processing
testing_ds = testing_ds.map(
    lambda x: {"images": x["images"],
               "features": x["features"]}, num_parallel_calls=tf.data.AUTOTUNE)


preds = model.predict(testing_ds).squeeze()


inputs = next(iter(testing_ds))
images = inputs["images"]

# Plotting
plt.figure(figsize=(10, 4))

for i in range(3):
    plt.subplot(1, 3, i+1)  # 1 row, 3 columns, i+1th subplot
    plt.imshow(images[i])  # Show image
    plt.title(f'Prediction: {preds[i]:.2f}')  # Set title with prediction
    plt.axis('off')  # Hide axis

plt.suptitle('Model Predictions on Testing Images', fontsize=16)
plt.tight_layout()
plt.show()


pred_df = testing_df[["isic_id"]].copy()
pred_df["target"] = preds.tolist()

sub_df = pd.read_csv(f'/kaggle/input/isic-2024-challenge/sample_submission.csv')
sub_df = sub_df[["isic_id"]].copy()
sub_df = sub_df.merge(pred_df, on="isic_id", how="left")
sub_df.to_csv("submission.csv", index=False, float_format="%.6f")
sub_df.head()


try:
    # %%
    validation_df = validation_df.drop_duplicates(subset=['isic_id'])
    print(sum(validation_df_df['target']))
    
    # %% [markdown]
    # #### XGBoost w/ CV
    
    # %%
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, classification_report, make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # %%
    x_train = training_df.drop(columns=['isic_id', 'target'])
    y_train = training_df['target']
    x_validation = validation_df.drop(columns=['isic_id', 'target'])
    y_validation = validation_df['target']
    
    majority_weight = 1 / TrainingConfig.majority_ratio
    minority_weight = 1 / TrainingConfig.minority_ratio
    class_weights = {0: majority_weight, 1: minority_weight}
    
    model_xgb = XGBClassifier(objective='binary:logistic', eval_metric = 'auc', 
                              random_state = TrainingConfig.seed, 
                              scale_pos_weight=minority_weight, device = 'gpu')
    
    parameters_xgb = {'max_depth': [6, 8], 'learning_rate': [0.01, 0.05], 
                      'n_estimators': [100, 500], 'subsample': [0.8, 1], 
                      'colsample_bytree': [0.8, 1], 'reg_alpha': [0.1, 0.5], 
                      'reg_lambda': [1, 1.5]}
    
    auc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")
    
    grid_search_xgb = GridSearchCV(estimator=model_xgb, param_grid=parameters_xgb, 
                                   scoring=auc_scorer, cv = 3, verbose=1, n_jobs=-1)
    
    
    # %%
    grid_search_xgb.fit(x_train, y_train)
    
    # %%
    print('Best parameters found: ', grid_search_xgb.best_params_)
    print('Best AUC score: ', grid_search_xgb.best_score_)
    
    best_xgb = grid_search_xgb.best_estimator_
    results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)
    
    y_val_pred_xgb = best_xgb.predict(x_validation)
    y_val_pred_xgb_proba = best_xgb.predict_proba(x_validation)[:,1]
    
    print(classification_report(y_validation, y_val_pred_xgb))
    print('Validation AUC: ', roc_auc_score(y_validation, y_val_pred_xgb_proba))
    
    # %%
    # change threshold for predicting positive class
    threshold = 0.1
    y_val_pred_xgb_th = (y_val_pred_xgb_proba >= threshold).astype(int)
    
    print(classification_report(y_validation, y_val_pred_xgb_th))
    
    # %%
    # classification report for training data
    y_train_pred_xgb = best_xgb.predict(x_train)
    print(classification_report(y_train, y_train_pred_xgb))
    
    # %% [markdown]
    # #### Logistic Regression
    
    # %%
    model_lr = LogisticRegression(random_state=TrainingConfig.seed, class_weight=class_weights, max_iter=100)
    
    parameters_lr = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2', 'elasticnet'], 
                     'solver': ['lbfgs', 'sag', 'saga', 'newton-cholesky'], 'l1_ratio': [0.25, 0.5, 0.75]}
    
    grid_search_lr = GridSearchCV(estimator=model_lr, param_grid=parameters_lr, scoring=auc_scorer, cv = 5, verbose=1, n_jobs=-1)
    
    # %%
    grid_search_lr.fit(x_train, y_train)
    
    # %%
    print('Best parameters found: ', grid_search_lr.best_params_)
    print('Best AUC score: ', grid_search_lr.best_score_)
    
    best_lr = grid_search_lr.best_estimator_
    results_lr = pd.DataFrame(grid_search_lr.cv_results_)
    
    y_val_pred_lr = best_lr.predict(x_validation)
    y_val_pred_lr_proba = best_lr.predict_proba(x_validation)[:,1]
    
    print(classification_report(y_validation, y_val_pred_lr))
    print('Validation AUC: ', roc_auc_score(y_validation, y_val_pred_lr_proba))
    
    # %%
    # change threshold for predicting positive class
    threshold = 0.3
    y_val_pred_lr_th = (y_val_pred_lr_proba >= threshold).astype(int)
    
    print(classification_report(y_validation, y_val_pred_lr_th))
    
    # %%
    # classification report for training data
    y_train_pred_lr = best_lr.predict(x_train)
    print(classification_report(y_train, y_train_pred_lr))
    
    # %% [markdown]
    # #### SVM w/ CV
    
    # %%
    model_svm = SVC(probability=True, random_state=TrainingConfig.seed, class_weight=class_weights)
    
    parameters_svm = {'C': [0.1, 1, 5, 10], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 
                      'gamma': ['scale', 'auto']}
    
    grid_search_svm = GridSearchCV(estimator=model_svm, param_grid=parameters_svm, 
                                   scoring=auc_scorer, cv = 5, verbose=1, n_jobs=-1)
    
    
    # %%
    #grid_search_svm.fit(x_train, y_train)
    
    # %%
    print('Best parameters found: ', grid_search_svm.best_params_)
    print('Best AUC score: ', grid_search_svm.best_score_)
    
    best_svm = grid_search_svm.best_estimator_
    results_svm = pd.DataFrame(grid_search_svm.cv_results_)
    
    y_val_pred_svm = best_svm.predict(x_validation)
    y_val_pred_svm_proba = best_svm.predict_proba(x_validation)[:,1]
    
    print(classification_report(y_validation, y_val_pred_svm))
    print('Validation AUC: ', roc_auc_score(y_validation, y_val_pred_svm_proba))
except:
    print("Something went wrong")



