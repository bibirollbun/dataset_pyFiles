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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, InceptionResNetV2, InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import zipfile
import shutil


# Clean up working directory
workspace_path = "/kaggle/working"

for item in os.listdir(workspace_path):
    target = os.path.join(workspace_path, item)
    try:
        if os.path.isfile(target) or os.path.islink(target):
            os.unlink(target)
        elif os.path.isdir(target):
            shutil.rmtree(target)
    except Exception as error:
        print(f"Could not delete {target}. Reason: {error}")


# Unzip data and organize structure

# Input zip file locations
path_to_train_zip = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
path_to_test_zip = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'

# Output directory structure
project_dir = '/kaggle/working/pet-classifier'
raw_train_dir = os.path.join(project_dir, 'unzipped_train')
train_split_dir = os.path.join(project_dir, 'training')
val_split_dir = os.path.join(project_dir, 'validation')
test_output_dir = os.path.join(project_dir, 'testing')

os.makedirs(raw_train_dir, exist_ok=True)
with zipfile.ZipFile(path_to_train_zip, 'r') as archive:
    archive.extractall(raw_train_dir)

os.makedirs(test_output_dir, exist_ok=True)
with zipfile.ZipFile(path_to_test_zip, 'r') as archive:
    archive.extractall(test_output_dir)

image_directory = os.path.join(raw_train_dir, 'train')
all_filenames = os.listdir(image_directory)

# Create directories for categorized split
for species in ['cats', 'dogs']:
    os.makedirs(os.path.join(train_split_dir, species), exist_ok=True)
    os.makedirs(os.path.join(val_split_dir, species), exist_ok=True)

# Split 80/20 per class
train_ratio = 0.8
cat_files = sorted([img for img in all_filenames if 'cat' in img])
dog_files = sorted([img for img in all_filenames if 'dog' in img])

cat_train_count = int(len(cat_files) * train_ratio)
dog_train_count = int(len(dog_files) * train_ratio)

# Move files accordingly
for img in cat_files[:cat_train_count]:
    shutil.move(os.path.join(image_directory, img), os.path.join(train_split_dir, 'cats', img))
for img in cat_files[cat_train_count:]:
    shutil.move(os.path.join(image_directory, img), os.path.join(val_split_dir, 'cats', img))

for img in dog_files[:dog_train_count]:
    shutil.move(os.path.join(image_directory, img), os.path.join(train_split_dir, 'dogs', img))
for img in dog_files[dog_train_count:]:
    shutil.move(os.path.join(image_directory, img), os.path.join(val_split_dir, 'dogs', img))

print("Split complete. Images placed into training/, validation/, and testing/ folders.")


# Check how many images remain
if os.path.exists(raw_train_dir):
    leftover_imgs = os.listdir(raw_train_dir)
    rem_dogs = len([i for i in leftover_imgs if 'dog' in i])
    rem_cats = len([i for i in leftover_imgs if 'cat' in i])
else:
    rem_dogs = rem_cats = 0

print(f"Remaining in original folder: Dogs = {rem_dogs}, Cats = {rem_cats}")

# Count split images
dogs_train = len(os.listdir(os.path.join(train_split_dir, 'dogs')))
cats_train = len(os.listdir(os.path.join(train_split_dir, 'cats')))
dogs_val = len(os.listdir(os.path.join(val_split_dir, 'dogs')))
cats_val = len(os.listdir(os.path.join(val_split_dir, 'cats')))

print("\nTraining Split:")
print(f"Dogs: {dogs_train}")
print(f"Cats: {cats_train}")

print("\nValidation Split:")
print(f"Dogs: {dogs_val}")
print(f"Cats: {cats_val}")


# Image generators

train_aug = ImageDataGenerator(rescale=1./255)
val_aug = ImageDataGenerator(rescale=1./255)

train_flow = train_aug.flow_from_directory(
    train_split_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

val_flow = val_aug.flow_from_directory(
    val_split_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)


# === Constants ===
INPUT_SHAPE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 10

# === Plot Training History ===
def plot_training_metrics(history, chart_title="Model Performance Overview"):
    epochs_ran = history.epoch
    metrics = history.history

    plt.figure(figsize=(14, 5))

    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_ran, metrics['accuracy'], label='Training Accuracy')
    plt.plot(epochs_ran, metrics['val_accuracy'], linestyle='--', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True)

    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_ran, metrics['loss'], label='Training Loss')
    plt.plot(epochs_ran, metrics['val_loss'], linestyle='--', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)

    plt.suptitle(chart_title)
    plt.tight_layout()
    plt.show()


from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_transfer_model(base_architecture, image_dim=299, dropout_prob=0.5, freeze_base=True):
    base_model = base_architecture(
        weights='imagenet',
        include_top=False,
        input_shape=(image_dim, image_dim, 3)
    )
    base_model.trainable = not freeze_base

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_prob)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# === Callbacks ===
training_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1)
]


# === Custom CNN Builder ===
def create_custom_cnn():
    model = models.Sequential([
        layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# === Train CNN ===
custom_cnn_model = create_custom_cnn()
custom_cnn_history = custom_cnn_model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    callbacks=training_callbacks)

# === Plot Results ===
plot_training_metrics(custom_cnn_history, "Custom CNN Performance")


# === Build and Train MobileNetV2 ===
mobilenet_model = create_transfer_model(
    base_architecture=MobileNetV2,
    image_dim=INPUT_SHAPE[0],
    dropout_prob=0.5,
    freeze_base=True)

mobilenet_history = mobilenet_model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    callbacks=training_callbacks)

# === Plot Performance ===
plot_training_metrics(mobilenet_history, "MobileNetV2 Transfer Learning")


# === Build and Train InceptionV3 ===
inceptionv3_model = create_transfer_model(
    base_architecture=InceptionV3,
    image_dim=INPUT_SHAPE[0],
    dropout_prob=0.5,
    freeze_base=True)

inceptionv3_history = inceptionv3_model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    callbacks=training_callbacks)

# === Plot Performance ===
plot_training_metrics(inceptionv3_history, "InceptionV3 Transfer Learning")


# FROM HERE
# === Path to Test Images ===
test_image_dir = '/kaggle/working/pet-classifier/testing/test'

# === Load Test Filenames ===
test_files = sorted([file for file in os.listdir(test_image_dir) if file.endswith('.jpg')])
test_dataframe = pd.DataFrame({'filename': test_files})

# === Test Data Generator ===
test_datagen = ImageDataGenerator(rescale=1./255)
test_data_generator = test_datagen.flow_from_dataframe(
    dataframe=test_dataframe,
    directory=test_image_dir,
    x_col='filename',
    y_col=None,
    target_size=INPUT_SHAPE,
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False)

# === Make Predictions ===
test_predictions = inceptionv3_model.predict(test_data_generator, verbose=1).flatten()
predicted_labels = (test_predictions > 0.5).astype(int)

# === Extract Image IDs ===
image_ids = [int(name.split('.')[0]) for name in test_files]

# === Create Submission File ===
submission = pd.DataFrame({
    'id': image_ids,
    'label': test_predictions
})

submission = submission.sort_values('id')
submission.to_csv('submission.csv', index=False)

print("submission.csv with probabilities created successfully.")

