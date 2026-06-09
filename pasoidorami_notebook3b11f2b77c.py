import pandas as pd
import os

DATA_DIR = "/kaggle/input/histopathologic-cancer-detection"
labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
print(labels.head())


# English language is used for code and comments as requested.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import time

# Record the start time
start_time = time.time()

# --- 1. PROBLEM AND DATA DESCRIPTION ---
# The goal is to build a binary classifier to identify metastatic cancer in small
# image patches taken from larger digital pathology scans.
# The data consists of TIFF images and a CSV file with labels.
# A '1' label means the center 32x32px of the image contains tumor tissue.

# Define constants
DATA_DIR = "/kaggle/input/histopathologic-cancer-detection"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMAGE_SIZE = 96  # The images are 96x96 pixels
BATCH_SIZE = 64  # Number of images to process in a batch

# Load the labels
full_labels_df = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))

# --- FOR SPEED: Use a small fraction of the data ---
# We will use 20,000 images out of the ~220,000 available to speed up the process.
# We use stratify to maintain the original distribution of positive/negative labels.
print(f"Using a subset of the data for speed. Full dataset has {len(full_labels_df)} images.")
_, sample_df = train_test_split(full_labels_df, test_size=0.1, random_state=42, stratify=full_labels_df['label'])
print(f"Subset size: {len(sample_df)} images.")

# Add the '.tif' extension to the id for the ImageDataGenerator
sample_df['id'] = sample_df['id'].apply(lambda x: f"{x}.tif")
sample_df['label'] = sample_df['label'].astype(str) # Convert labels to string for the generator

# Split the subset into training and validation sets
train_df, valid_df = train_test_split(sample_df, test_size=0.2, random_state=42, stratify=sample_df['label'])

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(valid_df)}")


# --- 2. EXPLORATORY DATA ANALYSIS (EDA) ---
print("\n--- Starting EDA ---")

# Plot the distribution of labels in our training subset
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=train_df)
plt.title('Distribution of Labels in Training Subset (0 = No Cancer, 1 = Cancer)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Display a few sample images
print("Displaying one example for each class:")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Positive sample
positive_id = train_df[train_df['label'] == '1'].iloc[0]['id']
positive_img = plt.imread(os.path.join(TRAIN_DIR, positive_id))
axes[0].imshow(positive_img)
axes[0].set_title(f"Class: 1 (Cancer)\nImage: {positive_id}")
axes[0].axis('off')

# Negative sample
negative_id = train_df[train_df['label'] == '0'].iloc[0]['id']
negative_img = plt.imread(os.path.join(TRAIN_DIR, negative_id))
axes[1].imshow(negative_img)
axes[1].set_title(f"Class: 0 (No Cancer)\nImage: {negative_id}")
axes[1].axis('off')

plt.show()


# --- 3. DATA PREPARATION AND MODEL ARCHITECTURE ---
print("\n--- Preparing Data Generators and Building Model ---")

# We use ImageDataGenerator to load images in batches and perform augmentation.
# We only rescale the images, as more augmentation would slow down training.
train_datagen = ImageDataGenerator(rescale=1./255.)
valid_datagen = ImageDataGenerator(rescale=1./255.)

# Create data generators from our dataframes
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col='id',
    y_col='label',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=TRAIN_DIR,
    x_col='id',
    y_col='label',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # No need to shuffle validation data
)

# Define a simple CNN model for speed
model = Sequential([
    # Input Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D((2, 2)),
    
    # Second Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(), # Helps stabilize training
    Dropout(0.5), # Reduces overfitting
    
    # Output Layer (Binary Classification)
    Dense(1, activation='sigmoid')
])

# Compile the model
# Using Adam optimizer, which is a good default.
# Binary Crossentropy is the standard loss function for binary classification.
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()


# --- 4. STARTING MODEL TRAINING ---
# This is the main part. To make it extremely fast, we will use:
# - epochs=1: Train for only one full pass over the data.
# - steps_per_epoch=50: Use only 50 batches for training in this epoch.
# This ensures the training step completes in under a minute.
print("\n--- Starting Model Training (Fast Version) ---")

EPOCHS = 1 # Set to 1 for maximum speed as requested
STEPS_PER_EPOCH = 50 # Limit steps to make it even faster

history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(valid_df) // BATCH_SIZE
)


# --- 5. RESULTS AND ANALYSIS ---
print("\n--- Training Finished. Displaying Results. ---")

# Plot training & validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# --- 6. CONCLUSION ---
print("\n--- Conclusion and Next Steps ---")
print("This script completed a full cycle of EDA, model building, and training very quickly.")
print("The resulting model performance is low because we used a small data subset and only one training epoch.")
print("\nTo improve performance, you can try the following:")
print("1. Increase the 'test_size' in the train_test_split to use more data (e.g., use all data).")
print("2. Increase the number of EPOCHS (e.g., to 10, 15, or more).")
print("3. Remove the 'STEPS_PER_EPOCH' limit to train on the full dataset each epoch.")
print("4. Experiment with a more complex model architecture (e.g., add more Conv2D layers or use a pre-trained model like ResNet50).")
print("5. Add more data augmentation in the ImageDataGenerator (e.g., flips, rotations).")

# Calculate and print the total execution time
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds.")


# --- 5.1. SAVE VALIDATION PREDICTIONS TO CSV (FINAL FIX) ---
print("\n--- Saving Validation Predictions to CSV (ONLY id, label) ---")

# Get predictions on validation data
val_preds = model.predict(validation_generator, verbose=1)

# Copy validation dataframe
val_results_df = valid_df.copy()

# Replace label column with predicted labels (binary 0/1)
# If threshold = 0.5, probability >= 0.5 -> 1 else 0
val_results_df['label'] = (val_preds >= 0.5).astype(int)

# Keep only required columns
val_results_df = val_results_df[['id', 'label']]

# Save to CSV
output_pred_csv = "submission.csv"
val_results_df.to_csv(output_pred_csv, index=False)

print(f"Submission file saved to: {output_pred_csv}")
print(val_results_df.head())













