import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
from functools import partial
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm # Use tqdm.notebook for Kaggle
import gc


# Environment Setup
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU detected. Running on CPU.")

# TPU setup removed, using default strategy for GPU/CPU
strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

# Constants
GCS_PATH = KaggleDatasets().get_gcs_path("siim-isic-melanoma-classification")
KAGGLE_PATH = '/kaggle/input/siim-isic-melanoma-classification'
# Adjust BATCH_SIZE for GPU memory (original was 16 * replicas for TPU)
BATCH_SIZE = 32 * strategy.num_replicas_in_sync # Start with 32 or 64 per replica (GPU)
IMAGE_SIZE = [256, 256] # Target size for the model
AUTOTUNE = tf.data.experimental.AUTOTUNE

print("Batch Size:", BATCH_SIZE)
print("Image Size:", IMAGE_SIZE)
print("GCS Path:", GCS_PATH)


# Get TFRecord file paths
ALL_TRAINING_FILENAMES = tf.io.gfile.glob(KAGGLE_PATH + "/tfrecords/train*.tfrec")
TEST_FILENAMES = tf.io.gfile.glob(KAGGLE_PATH + "/tfrecords/test*.tfrec")

# Split training files for validation
TRAIN_FILENAMES, VAL_FILENAMES = train_test_split(
    ALL_TRAINING_FILENAMES,
    test_size=0.1, # 10% for validation
    random_state=42 # Use a fixed random state for reproducibility
)

print("Number of training TFRecord files:", len(TRAIN_FILENAMES))
print("Number of validation TFRecord files:", len(VAL_FILENAMES))
print("Number of test TFRecord files:", len(TEST_FILENAMES))


def decode_image(image_data):
    """Decodes JPEG image, casts to float32, and normalizes."""
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
    # No initial resize here, will be done in augmentation/preprocessing step
    return image

def read_tfrecord(example, labeled):
    """Parses a single TFRecord example."""
    if labeled:
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64)
        }
    else:
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string)
        }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    else:
        image_name = example['image_name']
        return image, image_name

def preprocess_image(image, label=None, is_training=False):
    """Resizes and optionally augments the image."""
    image = tf.image.resize(image, IMAGE_SIZE)
    if is_training:
        # Basic augmentation
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_flip_up_down(image) # Optional
        # image = tf.image.random_saturation(image, 0.8, 1.2) # Optional
        # image = tf.image.random_brightness(image, 0.1) # Optional
        # image = tf.image.random_contrast(image, 0.8, 1.2) # Optional
    if label is None:
        return image
    else:
        return image, label


def load_dataset(filenames, labeled=True, ordered=False, is_training=False):
    """Loads TFRecords, preprocesses, and batches the dataset."""
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    # Apply preprocessing and augmentation
    dataset = dataset.map(partial(preprocess_image, is_training=is_training), num_parallel_calls=AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(2048) # Shuffle buffer size
        dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE) # prefetch next batch while training
    return dataset


train_dataset = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=False, is_training=True)
val_dataset = load_dataset(VAL_FILENAMES, labeled=True, ordered=False, is_training=False) # No augmentation/repeat/shuffle for val
test_dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=True, is_training=False) # Ordered for submission

# Calculate number of images and steps
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

num_training_images = count_data_items(TRAIN_FILENAMES)
num_validation_images = count_data_items(VAL_FILENAMES)
num_test_images = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH_TRAIN = num_training_images // BATCH_SIZE

print(f"Training images: {num_training_images}, Steps/epoch: {STEPS_PER_EPOCH_TRAIN}")
print(f"Test images: {num_test_images}")


# def show_batch(image_batch, label_batch):
#     plt.figure(figsize=(15, 15))
#     for n in range(min(8, BATCH_SIZE)): # Show up to 8 images
#         ax = plt.subplot(2, 4, n + 1)
#         plt.imshow(image_batch[n])
#         if label_batch[n] == 0:
#             plt.title("BENIGN")
#         else:
#             plt.title("MALIGNANT")
#         plt.axis("off")
#     plt.tight_layout()
#     plt.show()

# # Fetch a batch from the training dataset to visualize
# image_batch, label_batch = next(iter(train_dataset))
# show_batch(image_batch.numpy(), label_batch.numpy())

# # Clean up memory
# del image_batch, label_batch
# gc.collect()


# --- Class Weights and Bias Initialization (Requires train.csv) ---
# Load train CSV briefly to calculate weights
train_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
malignant_count = train_df['target'].sum()
total_count = len(train_df)
benign_count = total_count - malignant_count

weight_malignant = (total_count / malignant_count) / 2.0
weight_benign = (total_count / benign_count) / 2.0
class_weight = {0: weight_benign, 1: weight_malignant}
initial_bias = np.log([malignant_count / benign_count]) # Calculate initial bias

print(f"Benign cases: {benign_count}, Malignant cases: {malignant_count}")
print(f"Weight for class 0 (Benign): {class_weight[0]:.2f}")
print(f"Weight for class 1 (Malignant): {class_weight[1]:.2f}")
print(f"Initial bias: {initial_bias[0]:.2f}")

del train_df # Free memory
gc.collect()
# --- End Class Weights ---


# --- Model Definition ---
# No strategy.scope() needed for default strategy
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False, # Exclude the final classification layer
    weights='imagenet' # Use pre-trained ImageNet weights
)
base_model.trainable = False # Freeze the base model layers initially

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(20, activation="relu"), # Smaller dense layers from original
    tf.keras.layers.Dropout(0.4),                 # Dropout for regularization
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid', # Output layer for binary classification
                          bias_initializer=tf.keras.initializers.Constant(initial_bias)) # Set initial bias
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # Standard Adam optimizer
    loss='binary_crossentropy', # Suitable for binary classification
    metrics=[tf.keras.metrics.AUC(name='auc')] # Competition metric
)

model.summary()
# --- End Model Definition ---


# Callbacks for training
callback_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', # Monitor validation AUC
    patience=15,         # Stop after 15 epochs with no improvement
    mode='max',         # Maximize AUC
    verbose=1,
    restore_best_weights=True # Restore weights from the epoch with the best val_auc
)

callback_lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc', # Monitor validation AUC
    factor=0.1,        # Reduce LR by factor of 10
    patience=5,        # Reduce after 5 epochs with no improvement
    mode='max',        # Maximize AUC
    verbose=1,
    min_lr=1e-6        # Minimum learning rate
)

# Checkpoint saving the best weights based on validation AUC
callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "melanoma_best.weights.h5", # File path
    monitor='val_auc',          # Monitor validation AUC
    mode='max',                 # Maximize AUC
    save_best_only=True,        # Only save the best model
    save_weights_only=True,     # Save only the weights
    verbose=0                   # Less verbose output
)


EPOCHS = 40 # Set a reasonable number of epochs, EarlyStopping will likely stop it sooner

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
    validation_data=val_dataset,
    validation_steps=None,
    callbacks=[callback_early_stopping, callback_lr_reduce, callback_checkpoint],
    class_weight=class_weight, # Use calculated class weights
    verbose=1 # Show progress bar
)


def plot_history(history):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], 'bo-', label='Training loss')
    plt.plot(epochs, hist['val_loss'], 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot AUC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['auc'], 'bo-', label='Training AUC')
    plt.plot(epochs, hist['val_auc'], 'ro-', label='Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Plot the results (EarlyStopping restores best weights)
plot_history(history)


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns # For a prettier confusion matrix

# --- Important: Ensure val_dataset for evaluation covers all data and is ordered ---
# If your original val_dataset was created with ordered=False, it's best to create
# a new one for evaluation to ensure labels and predictions align perfectly.
# Also, ensure we use all validation images.

print("Re-creating validation dataset for full evaluation (ordered)...")
# Use the same BATCH_SIZE as training, or adjust if needed for prediction memory/speed
# For evaluation, is_training should be False, and ordered should be True.
EVAL_BATCH_SIZE = BATCH_SIZE # Can be different from training BATCH_SIZE if desired
val_eval_dataset = load_dataset(VAL_FILENAMES, labeled=True, ordered=True, is_training=False)

# Calculate the correct number of steps to cover the entire validation set
num_validation_images = count_data_items(VAL_FILENAMES)
VAL_EVAL_STEPS = (num_validation_images + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
print(f"Total validation images: {num_validation_images}")
print(f"Evaluation batch size: {EVAL_BATCH_SIZE}")
print(f"Validation evaluation steps: {VAL_EVAL_STEPS}")

# --- Get True Labels ---
print("Extracting true labels from the validation dataset...")
y_true_val = []
# .take(VAL_EVAL_STEPS) ensures we iterate through the entire dataset once
for images, labels in tqdm(val_eval_dataset.take(VAL_EVAL_STEPS), total=VAL_EVAL_STEPS):
    y_true_val.extend(labels.numpy())
y_true_val = np.array(y_true_val)
print(f"Extracted {len(y_true_val)} true labels.")

# --- Get Model Predictions (Probabilities) ---
# The model should already have the best weights loaded if EarlyStopping's restore_best_weights=True
# Or, if you saved weights: model.load_weights("melanoma_best.weights.h5")
print("Generating predictions on the full validation dataset...")
# model.predict will iterate through the dataset.
# Providing 'steps' ensures it processes the correct amount of data if the dataset could be infinite (though ours is not here).
y_pred_probs_val = model.predict(val_eval_dataset, steps=VAL_EVAL_STEPS, verbose=1)
# Ensure predictions match the number of true labels
y_pred_probs_val = y_pred_probs_val[:len(y_true_val)] # Trim if predict gives more due to batching
print(f"Generated {len(y_pred_probs_val)} predictions.")


# --- Calculate Metrics ---
# For AUC, we use the probabilities
auc_val = roc_auc_score(y_true_val, y_pred_probs_val)

# For other metrics, we need binary predictions (threshold at 0.5)
THRESHOLD = 0.5
y_pred_binary_val = (y_pred_probs_val > THRESHOLD).astype(int).flatten() # flatten in case of (N,1) shape

accuracy_val = accuracy_score(y_true_val, y_pred_binary_val)
precision_val = precision_score(y_true_val, y_pred_binary_val)
recall_val = recall_score(y_true_val, y_pred_binary_val)
f1_val = f1_score(y_true_val, y_pred_binary_val)
cm_val = confusion_matrix(y_true_val, y_pred_binary_val)

print("\n--- Validation Set Evaluation Results ---")
print(f"AUC: {auc_val:.4f}")
print(f"Accuracy: {accuracy_val:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"F1-Score: {f1_val:.4f}")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign (0)', 'Malignant (1)'],
            yticklabels=['Benign (0)', 'Malignant (1)'])
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ## End of Notebook

