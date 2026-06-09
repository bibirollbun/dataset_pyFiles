"""
Mel Spectrogram Instrument Classification using Transfer Learning
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

print("TensorFlow Version:", tf.__version__)
# -=- Check for GPU availability -=-
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU available:", physical_devices)
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(physical_devices)} Physical GPUs configured.")
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU")

# -=- User Defined Parameters -=-
BASE_DATA_DIR = '/kaggle/input/musical-instrumemts-sound-classification/Melspectogram_split'
IMG_HEIGHT = 160
IMG_WIDTH = 160
IMG_CHANNELS = 3 # RGB
BATCH_SIZE = 32
EPOCHS_INITIAL = 15 
LEARNING_RATE_INITIAL = 0.001

# -=- Define paths -=-
train_dir = os.path.join(BASE_DATA_DIR, 'train')
val_dir = os.path.join(BASE_DATA_DIR, 'val')
test_dir = os.path.join(BASE_DATA_DIR, 'test')

# -=- Define image size tuple -=-
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
AUTOTUNE = tf.data.AUTOTUNE

print("Loading datasets...")

# -=- Load Train and Validation Datasets -=-
train_dataset_raw = image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='categorical',
    color_mode='rgb',
    seed=42
)

validation_dataset_raw = image_dataset_from_directory(
    val_dir,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='categorical',
    color_mode='rgb',
    seed=42
)

# -=- Get class names from training data -=-
class_names = train_dataset_raw.class_names
NUM_CLASSES = len(class_names)
if not class_names:
     raise ValueError("Could not infer class names from training directory.")
print(f"Found {NUM_CLASSES} classes from training data: {class_names}")

# -=- Preprocessing Function for MobileNetV2 -=-
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# -=- Prepare Training Dataset -=-
print("\nPreparing Training Dataset...")
train_dataset = train_dataset_raw.map(
    lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE
)
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
print("Training dataset prepared.")

# -=- Prepare Validation Dataset -=-
print("\nPreparing Validation Dataset...")
validation_dataset = validation_dataset_raw.map(
    lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE
)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print("Validation dataset prepared.")

# -=- Prepare Test Dataset -=-
print("\nLoading Test Dataset (structured) and Extracting Filenames...")
# -=- Load test data using image_dataset_from_directory -=-
test_dataset_raw = image_dataset_from_directory(
    test_dir,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='categorical',
    color_mode='rgb'
)

# -=- Extract the ordered file paths -=-
ordered_full_paths = test_dataset_raw.file_paths
if not ordered_full_paths:
    print("Warning: No files found by image_dataset_from_directory in test_dir.")
    ordered_test_filenames = []
    test_file_count = 0
else:
    # -=- Extract just the file name from the full path -=-
    ordered_test_filenames = [os.path.basename(p) for p in ordered_full_paths]
    test_file_count = len(ordered_test_filenames)
    print(f"Found {test_file_count} test files.")
    print(f"Successfully extracted {len(ordered_test_filenames)} filenames in order.")

# -=- Create a separate dataset FOR PREDICTION (images only, preprocessed) -=-
print("\nPreparing test dataset for prediction (preprocessing images)...")
# -=- Map to apply preprocessing and DISCARD labels (y) -=-
test_dataset_for_predict = test_dataset_raw.map(
    lambda image, label: preprocess_input(image),
    num_parallel_calls=AUTOTUNE
)
test_dataset_for_predict = test_dataset_for_predict.prefetch(buffer_size=AUTOTUNE)
print("Test dataset prepared for model.predict().")

# -=- Visualize Raw Training Samples (Optional) -=-
print("\nShowing raw training samples:")
plt.figure(figsize=(10, 10))
for images, labels in train_dataset_raw.take(1):
    num_to_show = min(BATCH_SIZE, 9)
    for i in range(num_to_show):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_index = tf.argmax(labels[i])
        plt.title(f"Raw: {class_names[label_index]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    break # -=- Show only the first batch -=-

# -=- Build the Model (Transfer Learning) -=-
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False, weights='imagenet')
base_model.trainable = False
print(f"\nBase model ({base_model.name}) loaded. Trainable: {base_model.trainable}")
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
trainable_count = sum(tf.size(w).numpy() for w in model.trainable_weights)
non_trainable_count = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
print(f"\nTotal Trainable Params: {trainable_count}")
print(f"Total Non-Trainable Params: {non_trainable_count}")

# -=- Initial Training -=-
print(f"\n*** Starting Training ({EPOCHS_INITIAL} epochs) ***")
start_time = time.time()
history = model.fit(
    train_dataset,
    epochs=EPOCHS_INITIAL,
    validation_data=validation_dataset
)
end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# -=- Visualize Training Results -=-
print("\n*** Visualizing Training Results ***")
if history and hasattr(history, 'history'):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(epochs_range, acc, label='Training Accuracy'); plt.plot(epochs_range, val_acc, label='Validation Accuracy'); plt.legend(loc='lower right'); plt.title('Training and Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.subplot(1, 2, 2); plt.plot(epochs_range, loss, label='Training Loss'); plt.plot(epochs_range, val_loss, label='Validation Loss'); plt.legend(loc='upper right'); plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.tight_layout(); plt.show()

else: print("No training history found to plot.")

# -=- Generate Predictions and Save to CSV -=-
print("\*** Generating Predictions and Saving to CSV ***")

print(f"Predicting on {len(ordered_test_filenames)} test images...")
# -=- Predict using the dataset prepared for prediction (images only) -=-
predictions = model.predict(test_dataset_for_predict)
predicted_indices = np.argmax(predictions, axis=1)

if not class_names:
     print("Error: class_names not defined. Cannot map predictions.")
else:
    predicted_names = [class_names[i] for i in predicted_indices]

    print("Prediction counts match filename counts.")
    # -=- Create DataFrame using the ORDERED filenames extracted earlier -=-
    df_preds = pd.DataFrame({
        'Filename': ordered_test_filenames,
        'Instrument_name': predicted_names
    })

    # -=- Define CSV output path -=-
    csv_path = '/kaggle/working/predictions.csv'

    # -=- Save to CSV -=-
    df_preds.to_csv(csv_path, index=False)
    print(f"\nPredictions saved successfully to: {csv_path}")
    print("\nFirst 5 predictions:")
    print(df_preds.head())


