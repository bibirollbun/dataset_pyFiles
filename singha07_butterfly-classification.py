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
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Base directory
directory = '/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train'

# Read CSV
df = pd.read_csv('/kaggle/input/2025-bamboo-summer-competiton-dl-pr/train.csv')

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)

# Create tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_df['filename'].values, train_df['label_encoded'].values))
val_ds   = tf.data.Dataset.from_tensor_slices((val_df['filename'].values, val_df['label_encoded'].values))

# Load image from path
def load_image(filename, label):
    image_path = tf.strings.join([directory, filename], separator='/')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image, label

# Data augmentation function
def data_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image

# Preprocessing for training (with augmentation)
def process_train(image, label):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    image = data_augmentation(image)
    return image, label

# Preprocessing for validation (no augmentation)
def process_val(image, label):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Constants
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Apply pipeline
train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(process_train, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_val, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)




test_dir = '/kaggle/input/2025-bamboo-summer-competiton-dl-pr/test'
test_df = pd.read_csv('/kaggle/input/2025-bamboo-summer-competiton-dl-pr/test.csv')


# Create tf.data.Dataset from filenames
test_ds = tf.data.Dataset.from_tensor_slices(test_df['filename'].values)

# Test image loader and processor (same size & normalization)
def load_test_image(filepath):
    image_path = tf.strings.join([test_dir, filepath], separator='/')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (256, 256))                     # Match train/val size
    image = tf.cast(image, tf.float32) / 255.0                     # Normalize to [0,1]
    return image

# Apply preprocessing and batch
test_ds = test_ds.map(load_test_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)



from tensorflow.keras.applications import Xception		
# Load base model
base_model = Xception(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.layers


from tensorflow.keras.applications import Xception	
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Flatten

# Load base model
base_model = Xception(include_top=False, weights='imagenet', input_shape=(256, 256, 3))


trainable = False
for layer in base_model.layers:
    if 'block14' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = trainable
        
# Build model
model = models.Sequential([
    base_model,
    # layers.GlobalAveragePooling2D()
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(le.classes_), activation='softmax')
])

# Compile with a lower learning rate for fine-tuning
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)



preds = model.predict(test_ds)  # shape: (num_samples, num_classes)
pred_labels = tf.argmax(preds, axis=1).numpy()



# pred_labels: [3, 0, 1, 4, ...]
# Decode them back to original class names
class_names = le.inverse_transform(pred_labels)



test_df = pd.read_csv('/kaggle/input/2025-bamboo-summer-competiton-dl-pr/test.csv')

submission = pd.DataFrame({
    'filename': test_df['filename'],
    'label': class_names  # These should be the predicted class names
})

submission.to_csv('submission.csv', index=False)



submission.head()


submission.to_csv('submission.csv', index=False)

