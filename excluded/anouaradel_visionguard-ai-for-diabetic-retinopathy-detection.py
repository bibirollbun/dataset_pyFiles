# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Define the file path for our training data labels
TRAIN_CSV_PATH = '/kaggle/input/aptos2019-blindness-detection/train.csv'

# Read the CSV file into a pandas DataFrame
df_train = pd.read_csv(TRAIN_CSV_PATH)

# Display the first 5 rows of the DataFrame to understand its structure
print("The first 5 rows of our training data:")
df_train.head()


# Define the path to the directory containing training images
TRAIN_IMG_PATH = '/kaggle/input/aptos2019-blindness-detection/train_images/'

# Create a new column 'image_path' by joining the image directory path, the id_code, and the .png extension
df_train['image_path'] = df_train['id_code'].apply(lambda x: os.path.join(TRAIN_IMG_PATH, x + '.png'))

# Convert the 'diagnosis' column to a string type, so it is treated as a category rather than a number
# This is important for stratified splitting and for many deep learning frameworks when setting up labels for classification
df_train['diagnosis'] = df_train['diagnosis'].astype(str)

# Display the first 5 rows again to see our new 'image_path' column and the updated 'diagnosis' type
print("DataFrame with full image paths and categorical labels:")
df_train.head()


# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for our plots
sns.set_style("whitegrid")

# --- Data Exploration ---
# Count the number of images in each diagnosis category
diagnosis_counts = df_train['diagnosis'].value_counts().sort_index()

print("Number of images per diagnosis category:")
print(diagnosis_counts)

# --- Visualization ---
# Create a bar chart to visualize the distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, palette="viridis")

# Add titles and labels for clarity
plt.title('Distribution of Diabetic Retinopathy Stages in the Dataset', fontsize=16)
plt.xlabel('Diagnosis (Stage)', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=0) # Keep the x-axis labels horizontal

# Show the plot
plt.show()


from sklearn.model_selection import train_test_split

# We use the entire DataFrame for the split, as it contains both the image paths and the labels.
# test_size=0.2 specifies that we want to allocate 20% of the data to the validation set.
# random_state=42 is a seed for the random number generator, ensuring that our split is reproducible.
# stratify=df_train['diagnosis'] is the key parameter that ensures both training and validation sets
# have the same proportion of samples for each diagnosis class.

train_df, val_df = train_test_split(
    df_train,
    test_size=0.2,
    random_state=42,
    stratify=df_train['diagnosis']
)

# Reset the index of the new dataframes to avoid potential issues later
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# --- Verification ---
# Print the number of images in each new dataframe
print(f"Total original images: {len(df_train)}")
print(f"Number of images in the new training set: {len(train_df)}")
print(f"Number of images in the new validation set: {len(val_df)}")
print("-" * 50)

# Verify the distribution in the training set by checking the percentage of each class
print("Distribution of diagnoses in the training set (%):")
print(train_df['diagnosis'].value_counts(normalize=True).sort_index() * 100)
print("-" * 50)

# Verify the distribution in the validation set
print("Distribution of diagnoses in the validation set (%):")
print(val_df['diagnosis'].value_counts(normalize=True).sort_index() * 100)


import tensorflow as tf

# --- Define Constants ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# --- Create a data augmentation layer ---
# This is a modern way to do augmentation directly on the GPU for efficiency
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")


# --- Function to load and preprocess images ---
# This function will be the core of our pipeline
def load_and_preprocess_image(image_path, label):
    # Read the file from disk
    image = tf.io.read_file(image_path)
    # Decode the image from its compressed format (e.g., PNG)
    image = tf.image.decode_png(image, channels=3)
    # Resize the image to the target dimensions
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # Normalize pixel values to the [0, 1] range
    image = image / 255.0
    return image, label

# --- Build the Training and Validation Datasets ---
# AUTOTUNE allows tf.data to find the best parallel settings automatically
AUTOTUNE = tf.data.AUTOTUNE

# Create the training dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_df['image_path'], train_df['diagnosis'].astype(int).values))
train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().shuffle(buffer_size=len(train_df)) # Cache and shuffle
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE) # Apply augmentation
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # Prefetch the next batch

# Create the validation dataset (no shuffling or augmentation)
val_ds = tf.data.Dataset.from_tensor_slices((val_df['image_path'], val_df['diagnosis'].astype(int).values))
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache() # Cache for faster evaluation
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("tf.data pipelines created successfully.")


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB3
from sklearn.utils import class_weight

# --- 1. Load the Base Model (EfficientNetB3) ---
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# --- 2. Freeze the Base Model ---
# We initially freeze the base model layers. They will not be trained in the first phase.
base_model.trainable = False

# --- 3. Add Custom Layers on Top ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)

# --- 4. Create the Final Model ---
model = Model(inputs=base_model.input, outputs=predictions)

# --- 5. Compile the Model with the Correct Loss Function ---
# We use `SparseCategoricalCrossentropy` because our `tf.data` pipeline provides integer labels (0, 1, 2, etc.)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# --- 6. Print Model Summary ---
print("Model architecture is ready and compiled.")
model.summary()


# Check if TensorFlow can detect the GPU
print("TensorFlow version:", tf.__version__)
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("GPU is available:", gpu_devices)
else:
    print("GPU is not available.")


# --- 1. Manually Define Class Weights ---
# Based on experimentation, these manually-tuned weights provide a better balance
# for this dataset than the automatic 'balanced' mode.
class_weights_dict = {
    0: 0.75,
    1: 2.0,
    2: 0.8,
    3: 2.8,
    4: 2.5
}
print("Using Manually Adjusted Class Weights:")
print(class_weights_dict)
print("-" * 60)

# --- 2. Define Callbacks for smarter training ---
# EarlyStopping will stop training if val_loss doesn't improve for 5 epochs
# and will restore the best model weights.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ReduceLROnPlateau will reduce the learning rate if the validation loss plateaus.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7
)

print("Callbacks defined. Starting initial training on the model head...")
print("-" * 60)

# --- 3. Start Initial Training ---
# The model is trained with the class weights and callbacks.
EPOCHS = 15 # A maximum number of epochs to run

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr]
)

print("\nModel initial training complete.")


# --- 1. Unfreeze the base model ---
base_model.trainable = True

print(f"Number of layers in the base model: {len(base_model.layers)}")

# --- Optional: Fine-tune only from a certain layer onwards ---
# We freeze the first 100 layers and only fine-tune the rest.
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# --- 2. Re-compile the model with a very low learning rate ---
fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

model.compile(
    optimizer=fine_tune_optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# --- 3. Print the model summary again ---
# This will now show a much larger number of trainable parameters.
print("\nModel re-compiled for fine-tuning.")
model.summary()


# --- Continue training the model (fine-tuning) ---

# Define how many additional epochs to train for.
FINE_TUNE_EPOCHS = 15

# Calculate the total number of epochs to run until.
last_epoch = history.epoch[-1]
TOTAL_EPOCHS = last_epoch + 1 + FINE_TUNE_EPOCHS

print(f"Initial training stopped at epoch {last_epoch + 1}.")
print(f"Resuming training for fine-tuning up to epoch {TOTAL_EPOCHS}.")
print("-" * 60)


# Resume training the model.
history_fine_tune = model.fit(
    train_ds,
    epochs=TOTAL_EPOCHS,
    initial_epoch=last_epoch + 1,
    validation_data=val_ds,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr]
)

# --- Append the fine-tuning history to our original history ---
for key in history_fine_tune.history:
    if key not in history.history:
        history.history[key] = []
    history.history[key].extend(history_fine_tune.history[key])

print("\nModel fine-tuning complete and history has been updated.")


# --- Extract Combined History ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a range of epochs for the x-axis
epochs_range = range(1, len(history.history['accuracy']) + 1)

# Determine the epoch where fine-tuning started
initial_epochs_run = len(acc) - len(history_fine_tune.history['accuracy'])


# --- Plotting ---
plt.figure(figsize=(16, 8))

# Subplot 1: Combined Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.axvline(x=initial_epochs_run, color='r', linestyle='--', label='Start Fine-Tuning')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Full Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

# Subplot 2: Combined Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.axvline(x=initial_epochs_run, color='r', linestyle='--', label='Start Fine-Tuning')
plt.ylim([0, max(plt.ylim())])
plt.legend(loc='upper right')
plt.title('Full Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.suptitle('Full Training History: Initial Training + Fine-Tuning', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# --- 1. Get True Labels and Model Predictions ---
val_labels = np.concatenate([y for x, y in val_ds], axis=0)
predictions = model.predict(val_ds)
predicted_labels = np.argmax(predictions, axis=1)

# --- 2. Calculate and Plot the Confusion Matrix ---
class_names = ['0', '1', '2', '3', '4']
cm = confusion_matrix(val_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Diagnosis (True Label)', fontsize=12)
plt.xlabel('Predicted Diagnosis (Predicted Label)', fontsize=12)
plt.show()

# --- 3. Print the Classification Report ---
print("\nClassification Report:")
print(classification_report(val_labels, predicted_labels, target_names=class_names))


# --- Save the Final Model ---
# We use model.save() to store the entire model—architecture, weights,
# and optimizer state—into a single '.keras' file.

model.save('retinaguard_model.keras')

print("Model saved successfully as 'visionguard_model.keras'")


# --- FOR REFERENCE: The load_and_preprocess_image function with CLAHE ---

# import cv2
# import numpy as np

# def apply_clahe(image):
#     # Convert image to LAB color space
#     lab = cv2.cvtColor(image.numpy().astype(np.uint8), cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#
#     # Apply CLAHE to the L-channel
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#
#     # Merge the CLAHE enhanced L-channel back
#     limg = cv2.merge((cl, a, b))
#
#     # Convert back to RGB
#     final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
#     return final.astype(np.float32)
#
# def load_and_preprocess_image(image_path, label):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_png(image, channels=3)
#     image = tf.py_function(func=apply_clahe, inp=[image], Tout=tf.float32)
#     image.set_shape([None, None, 3])
#     image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
#     image = image / 255.0
#     return image, label

