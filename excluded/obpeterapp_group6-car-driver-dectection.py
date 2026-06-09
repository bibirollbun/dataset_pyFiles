# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set parameters
IMG_SIZE = (128, 128)  # Resize all images to 128x128
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 10  # 10 driving behavior classes

# Paths to dataset (modify as needed)
TRAIN_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TEST_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test"

# Data Augmentation for better generalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  
])

# Display model architecture
model.summary()

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy", verbose=1)

# Train model
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[lr_reduction, checkpoint])

# Plot accuracy and loss graphs
def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Loss plot
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.show()

plot_training_history(history)

# Load test data for predictions
test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = None
if os.path.isdir(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:  # Ensure test images exist
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode=None,
        shuffle=False  
    )
else:
    print("Test directory is empty or does not contain subdirectories.")

if test_generator:
    # Predict probabilities for each test image
    predictions = model.predict(test_generator)
    filenames = test_generator.filenames

    # Create submission DataFrame
    submission_df = pd.DataFrame(predictions, columns=[f'c{i}' for i in range(NUM_CLASSES)])
    submission_df.insert(0, 'img', [f.split('/')[-1] for f in filenames])

    # Save to CSV
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file saved as 'submission.csv'.")

# Evaluate model on validation data
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc:.4f}")

# Confusion Matrix & Classification Report
y_true = val_generator.classes
y_pred = np.argmax(model.predict(val_generator), axis=1)

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))



import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Set parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30  # Increased for better learning
NUM_CLASSES = 10  # Driving behavior classes

# Paths to dataset
TRAIN_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
TEST_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

# Load DenseNet121 as feature extractor
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base model initially

# Define Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

# Callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-4 * 0.1 ** (epoch // 10))
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy", verbose=1)

# Train model
history = model.fit(
    train_generator, validation_data=val_generator,
    epochs=EPOCHS, class_weight=class_weights,
    callbacks=[lr_reduction, checkpoint, early_stopping, lr_scheduler]
)

# Unfreeze top layers for fine-tuning
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

# Fine-tune model
history_fine = model.fit(
    train_generator, validation_data=val_generator,
    epochs=10, class_weight=class_weights,
    callbacks=[lr_reduction, checkpoint, early_stopping, lr_scheduler]
)

# Evaluate model
val_loss, val_acc = model.evaluate(val_generator)
print(f"Final Validation Accuracy: {val_acc:.4f}")


