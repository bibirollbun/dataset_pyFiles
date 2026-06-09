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


# Unzip training and test data with overwrite
!unzip -o -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip -d /kaggle/working/train
!unzip -o -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip -d /kaggle/working/test


import os, shutil
from tqdm import tqdm

# Correct source path (after unzip)
source_folder = '/kaggle/working/train/train'
target_folder = '/kaggle/working/train_split'

# Create directories
os.makedirs(f'{target_folder}/cat', exist_ok=True)
os.makedirs(f'{target_folder}/dog', exist_ok=True)

# Move images into respective folders
for fname in tqdm(os.listdir(source_folder)):
    src_path = os.path.join(source_folder, fname)
    if fname.startswith('cat'):
        shutil.move(src_path, os.path.join(target_folder, 'cat', fname))
    elif fname.startswith('dog'):
        shutil.move(src_path, os.path.join(target_folder, 'dog', fname))



# Confirm folders exist and contain .jpg files
print("\nFolders inside /kaggle/working/train_split:", os.listdir('/kaggle/working/train_split'))

print("\nSample images:")
print(os.listdir('/kaggle/working/train_split/')[:5])


print("Moved cat images:", len(os.listdir('/kaggle/working/train_split/cat')))
print("Moved dog images:", len(os.listdir('/kaggle/working/train_split/dog')))


print(os.listdir('/kaggle/working/test/test')[:5])


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
img_size = (150, 150)
batch_size = 32
train_path = '/kaggle/working/train_split'
test_path = '/kaggle/working/test/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# CNN Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=10,
    callbacks=[early_stop]
)

# Plot Training Performance
def plot_history(hist):
    plt.figure(figsize=(12, 4))
    for i, metric in enumerate(['accuracy', 'loss']):
        plt.subplot(1, 2, i+1)
        plt.plot(hist.history[metric], label='Train')
        plt.plot(hist.history[f'val_{metric}'], label='Val')
        plt.title(f'Model {metric.title()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(history)

# Generate Predictions
submission = []
for fname in tqdm(sorted(os.listdir(test_path))):
    if fname.endswith('.jpg'):
        img_path = os.path.join(test_path, fname)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)[0][0]
        submission.append((int(fname.split('.')[0]), pred))

# Save Submission
submission_df = pd.DataFrame(submission, columns=['id', 'label']).sort_values('id')
submission_df.to_csv('MSBA.Session002.UjjwalKhanna_CNN.csv', index=False)  # Rename accordingly


# ResNet50 for Dogs vs Cats - Clean 5 Epoch Version

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import layers, models, optimizers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configurations
img_size = (224, 224)
batch_size = 64
train_path = '/kaggle/working/train_split'
test_path = '/kaggle/working/test/test'

# Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# ResNet50 Architecture with Functional API
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_tensor, outputs=out)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.0005),
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_resnet.keras', monitor='val_loss', save_best_only=True, verbose=1)

# Train for 5 Epochs (Initial + Fine-Tuning Combined)
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=5,
    callbacks=[early_stop, checkpoint]
)

# Plot History (if needed)
def plot_history(hist):
    plt.figure(figsize=(12, 4))
    for i, metric in enumerate(['accuracy', 'loss']):
        plt.subplot(1, 2, i+1)
        plt.plot(hist.history[metric], label='Train')
        plt.plot(hist.history[f'val_{metric}'], label='Val')
        plt.title(f'Model {metric.title()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.title())
        plt.legend()
    plt.tight_layout()
    plt.show()


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# Inference on Test Set
submission = []
for fname in tqdm(sorted(os.listdir(test_path))):
    if fname.endswith('.jpg'):
        img_path = os.path.join(test_path, fname)
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)[0][0]
        submission.append((int(fname.split('.')[0]), pred))

# Save Submission
submission_df = pd.DataFrame(submission, columns=['id', 'label']).sort_values('id')
submission_df.to_csv('MSBA.Session002.UjjwalKhanna_ResNet.csv', index=False)

