import tensorflow as tf
from tensorflow import data as tf_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras import layers
import keras_tuner as kt
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.special import expit
import glob
import tifffile as tiff

sns.set(style='whitegrid')
print("Tensorflow version " + tf.__version__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


test_dir = '/kaggle/input/histopathologic-cancer-detection/test/'
train_dir = '/kaggle/input/histopathologic-cancer-detection/train/'
train_labels_filepath = '/kaggle/input/histopathologic-cancer-detection/train_labels.csv'
test_labels_filepath = '/kaggle/input/histopathologic-cancer-detection/sample_submission.csv'


train_labels = pd.read_csv(train_labels_filepath)
test_labels = pd.read_csv(test_labels_filepath)
train_labels['path'] = train_dir + "/" + train_labels['id'].astype(str) + ".tif"
test_labels['path'] = test_dir + "/" + test_labels['id'].astype(str) + ".tif"


train_labels.head()


# Compare training labels and number of images
num_labels_train = train_labels.shape[0]
num_images_train = len(os.listdir(train_dir))
print(f'Training data contains {num_images_train} with {num_labels_train} labels.')

# Compare test labels and number of images
num_labels_test = test_labels.shape[0]
num_images_test = len(os.listdir(test_dir))
print(f'Training data contains {num_images_test} with {num_labels_test} labels.')


# Count labels in each class
label_counts = train_labels['label'].value_counts(normalize=True).reset_index()
label_counts.columns = ['label', 'proportion']

# Visualize class distribution
sns.barplot(data=label_counts, x='label', y='proportion')

plt.title('Relative Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Proportion')
plt.ylim(0, 1)
plt.show()


# Visualize benign and malignant samples
benign_samples = train_labels[train_labels["label"] == 0].sample(3)["id"].values
malignant_samples = train_labels[train_labels["label"] == 1].sample(3)["id"].values

fig, axes = plt.subplots(2, 3, figsize=(10,6))

def plot_sample(sample_id, sample_dir, ax, title):
    img_path = os.path.join(sample_dir, sample_id + ".tif")
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

for i, img_id in enumerate(benign_samples):
    ax = axes[0, i]
    plot_sample(img_id, train_dir, ax, 'Benign')

for i, img_id in enumerate(malignant_samples):
    ax = axes[1, i]
    plot_sample(img_id, train_dir, ax, 'Malignant')

plt.tight_layout()
plt.show()


benign_samples = train_labels[train_labels["label"] == 0].sample(100)["id"].values
malignant_samples = train_labels[train_labels["label"] == 1].sample(100)["id"].values
num_bins = 32

histograms = []

def extract_color_histograms(samples, num_bins, label_name):
    # Initialize histograms for each color channel
    hist_r = np.zeros(num_bins)
    hist_g = np.zeros(num_bins)
    hist_b = np.zeros(num_bins)
    
    # Extract and combine data
    for img_id in samples:
        img = cv2.imread(os.path.join(train_dir, img_id + ".tif"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist_r += cv2.calcHist([img], [0], None, [num_bins], [0, 256]).flatten()
        hist_g += cv2.calcHist([img], [1], None, [num_bins], [0, 256]).flatten()
        hist_b += cv2.calcHist([img], [2], None, [num_bins], [0, 256]).flatten()
        
    # Normalize histograms
    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()

    histogram_dict = {'red': hist_r, 'green': hist_g, 'blue': hist_b}
    histograms.append({'label': label_name, 'histograms': histogram_dict})

# Process samples
extract_color_histograms(benign_samples, num_bins, 'Benign')
extract_color_histograms(malignant_samples, num_bins, 'Malignant')

# Plot
fig, axes = plt.subplots(2, 3, figsize=(10,6), sharey=True)

def plot_color_histogram(histogram_values, label, color, ax):
    ax.plot(histogram_values, color=color)
    ax.set_title(f'{label} {color.title()} Channel Histogram')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Proportion')
    ax.grid(True)

for row, result_dict in enumerate(histograms):
    label = result_dict['label']
    histogram_dict = result_dict['histograms']
    for i, color in enumerate(histogram_dict):
        ax = axes[row, i]
        plot_color_histogram(histogram_dict[color], label, color, ax)
        
plt.tight_layout()
plt.show()      


intensities = []

# Function to compute mean grayscale intensity per image
def extract_grayscale_intensity(sample_ids, label_name):
    for img_id in sample_ids:
        img_path = os.path.join(train_dir, img_id + ".tif")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mean_intensity = np.mean(img)
        intensities.append({'label': label_name, 'intensity': mean_intensity})

# Process samples
extract_grayscale_intensity(benign_samples, 'Benign')
extract_grayscale_intensity(malignant_samples, 'Malignant')

# Create DataFrame for Seaborn
df_gray = pd.DataFrame(intensities)

# Plot
plt.figure(figsize=(8, 5))

sns.histplot(data=df_gray, x='intensity', hue='label', kde=True, stat='density', common_norm=False, bins=30)

plt.title('Grayscale Intensity Distribution by Class')
plt.xlabel('Mean Grayscale Intensity')
plt.ylabel('Density')
plt.tight_layout()
plt.show()


# Balance training data (subsample for debugging)
n_train = train_labels['label'].value_counts().min()
n_train = round(n_train * 0.05)  # Limit to 10% of total for debugging

benign = train_labels[train_labels['label'] == 0].sample(n_train)
malignant = train_labels[train_labels['label'] == 1].sample(n_train)
df_train_all = pd.concat([benign, malignant], axis=0).reset_index(drop=True)


# Split training and testing data
train_df, val_df = train_test_split(
    df_train_all, 
    test_size=0.2, 
    stratify=df_train_all['label'], 
    random_state=1337
)


# Set up TensorFlow Data Service
def make_dataset(df, image_dir, image_size, batch_size, shuffle=True, repeat=True, include_id=False):
    file_paths = df['path'].astype(str).tolist()
    labels = df['label'].tolist()
    ids = df['id'].astype(str).tolist()

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels, ids))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    def map_fn(path, label, id_):
        def _load_image(p):
            p = p.numpy().decode("utf-8")
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            return img.astype(np.float32) / 255.0

        image = tf.py_function(_load_image, inp=[path], Tout=tf.float32)
        image.set_shape((*image_size, 3))
        return (image, label, id_) if include_id else (image, label)

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# Create data streams
image_size = (96, 96)
batch_size = 128

train_ds = make_dataset(train_df, train_dir, image_size, batch_size, shuffle=True)
val_ds = make_dataset(val_df, train_dir, image_size, batch_size, shuffle=False)


# Augment data with rotation and flip
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    # x = layers.Rescaling(1./255)(x)
    x = layers.CenterCrop(32, 32)(x)

    # Entry block
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

xception_model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(xception_model, show_shapes=True)


class MinimumEpochEarlyStopping(keras.callbacks.EarlyStopping):
    def __init__(self, min_epochs=10, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Only start checking for stopping after reaching min_epochs
        if epoch + 1 >= self.min_epochs:
            super().on_epoch_end(epoch, logs)


epochs = 25
steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
validation_steps = int(np.ceil(len(val_df) / batch_size))
    
callbacks = [
    MinimumEpochEarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


xception_model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
)

history_xception = xception_model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    validation_data=val_ds,
)


def build_basic_model(hp):
    model = keras.Sequential()
    
    model.add(keras.Input(shape=image_size + (3,)))
    # model.add(layers.Rescaling(1./255))
    model.add(layers.CenterCrop(32, 32))

    # Tune number of layers and layer sizes
    for i in range(hp.Int('num_blocks', 1, 3)):
        filters = hp.Choice(f'block_filters_{i}', values=[32, 64, 128])
        model.add(layers.Conv2D(filters, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    # Tune dropout 
    model.add(layers.Dropout(hp.Choice('dropout', values=[0.2, 0.3, 0.5])))
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile model with variable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")]
    )

    return model


# Delete the tuner directory if it exists
import shutil
tuner_dir = 'my_tuners/basic'
if os.path.exists(tuner_dir):
    shutil.rmtree(tuner_dir)

# Initialize tuner
basic_tuner = kt.RandomSearch(
    lambda hp: build_basic_model(hp),
    objective='val_accuracy',
    directory='my_tuners',
    project_name='basic'
)

# Summarize tuner
basic_tuner.search_space_summary()


# Run tuner
basic_tuner.search(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    validation_data=val_ds,
)


basic_tuner.results_summary()


# Get the top model
models = basic_tuner.get_best_models(num_models=1)
best_model = models[0]
best_model.summary()


# Balance training data (subsample for debugging)
n_train = train_labels['label'].value_counts().min()

benign = train_labels[train_labels['label'] == 0].sample(n_train)
malignant = train_labels[train_labels['label'] == 1].sample(n_train)
df_train_all = pd.concat([benign, malignant], axis=0).reset_index(drop=True)

# Split training and testing data
train_df, val_df = train_test_split(
    df_train_all, 
    test_size=0.2, 
    stratify=df_train_all['label'], 
    random_state=1337
)

steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
validation_steps = int(np.ceil(len(val_df) / batch_size))

# Make dataset
train_ds_full = make_dataset(train_df, train_dir, image_size, batch_size, shuffle=True)
val_ds_full = make_dataset(val_df, train_dir, image_size, batch_size, shuffle=False)


# Fit final basic model
best_basic_hps = basic_tuner.get_best_hyperparameters(num_trials=1)
best_basic_model = build_basic_model(best_basic_hps[0])
history_basic = best_basic_model.fit(
    train_ds_full,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    validation_data=val_ds_full  
)


# Fit Xception model to all data
history_xception = xception_model.fit(
    train_ds_full,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    validation_data=val_ds_full,
)


def plot_accuracy(model, ax, title):
    ax.plot(model.history['accuracy'], label='Train Accuracy', color='#1f77b4', linestyle='dashed')
    ax.plot(model.history['val_accuracy'], label='Val Accuracy', color='#1f77b4')
    ax.set_title(f'CNN ({title}) Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()

fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

# Plot history_basic
plot_accuracy(history_basic, axes[0], "Basic")
plot_accuracy(history_xception, axes[1], "Xception")

plt.tight_layout()
plt.show()


# Get true and predicted labels
y_true = np.concatenate([y.numpy() for _, y in val_ds.take(validation_steps)], axis=0)
y_pred_basic = best_basic_model.predict(val_ds, steps=validation_steps).ravel()
y_pred_xception = xception_model.predict(val_ds, steps=validation_steps).ravel()


def plot_roc(y_true, y_pred, ax, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_title(f"Receiver Operating Characteristic (ROC) | {title}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

# Plot
plot_roc(y_true, y_pred_basic, axes[0], "Basic")
plot_roc(y_true, y_pred_xception, axes[1], "Xception")

plt.tight_layout()
plt.show()


# Create test data stream
test_ds = make_dataset(test_labels, test_dir, image_size, batch_size, 
                       shuffle=False, repeat=False, include_id=True)

# Strip out the IDs for prediction
predict_ds = test_ds.map(lambda image, label, id_: image)


all_ids = []
for batch in test_ds:
    _, _, ids = batch
    all_ids.extend([id_.numpy().decode("utf-8") for id_ in ids])
len(all_ids)


# Create submission
y_pred_basic = best_basic_model.predict(predict_ds).ravel()
submission_basic_df = pd.DataFrame(
    {
        'id': all_ids,
        'label': y_pred_basic
    }
)
submission_basic_df.to_csv('submission_basic.csv', index=False)


y_pred_xception = xception_model.predict(predict_ds).ravel()
y_pred_xception_probs = expit(y_pred_xception)
submission_xception_df = pd.DataFrame(
    {
        'id': all_ids,
        'label': y_pred_xception_probs
    }
)
submission_xception_df.to_csv('submission_xception.csv', index=False)




