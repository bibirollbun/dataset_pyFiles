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
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras import layers, models, callbacks, optimizers
import keras
from PIL import ImageDraw, ImageFont, Image


dataset_dir = "/kaggle/input/heads-or-tails-image-classification/train"
img_size = (300, 300)
batch_size = 8


samples_per_class = 3

# Get class folders
class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

# Load and display images
plt.figure(figsize=(15, 5 * len(class_folders)))

img_index = 1

for class_name in class_folders:
    class_path = os.path.join(dataset_dir, class_name)
    image_files = os.listdir(class_path)[:samples_per_class]

    for img_file in image_files:
        img_path = os.path.join(class_path, img_file)

        # Load and resize image
        image = load_img(img_path, target_size=img_size)
        image_np = img_to_array(image).astype(np.uint8)
        
        # Convert to PIL to draw text
        pil_img = Image.fromarray(image_np)
        draw = ImageDraw.Draw(pil_img)

        # Draw class name on top
        draw.text((10, 10), class_name, fill="white")

        # Plot
        plt.subplot(len(class_folders), samples_per_class, img_index)
        plt.imshow(pil_img)
        plt.axis("off")
        img_index += 1

plt.tight_layout()
plt.show()


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)


val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42
)


base_model = EfficientNetV2B3(
    include_top=False,
    input_shape=(300, 300, 3),
    weights='imagenet'
)

base_model.trainable = False


model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')  # 2 classes
])


model.summary()



optimizer = optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)


cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
    callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)
]


history = model.fit(
    train_generator,
    epochs=3,
    validation_data=val_generator,
    callbacks=cb,
    verbose=1
)


# Unfreeze last 40 layers
for layer in base_model.layers[-40:]:
    layer.trainable = True


# Re-compiling with lower learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)


model.summary()


fine_tune_history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=cb,
    verbose=1
)


model.save("Heads-tails.keras")


def plot_metrics(history):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    plt.figure(figsize=(16, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i + 1)
        plt.plot(history.history[metric], label='train_' + metric)
        plt.plot(history.history['val_' + metric], label='val_' + metric)
        plt.title(metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
    plt.tight_layout()
    plt.show()


plot_metrics(fine_tune_history)


test_dir = "/kaggle/input/heads-or-tails-image-classification/test"


# Test ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

# Test generator
test_generator = test_datagen.flow_from_directory(
    directory=os.path.dirname(test_dir),  # parent directory
    target_size=img_size,
    batch_size=1,
    shuffle=False,
    class_mode='binary',
    classes=[os.path.basename(test_dir)]  # folder name e.g., "test"
)

# Predicted probabilities using trained model
test_generator.reset()
probabilities = model.predict(test_generator, verbose=1)

# File names
file_paths = test_generator.filenames  # e.g., ['test/image_23.jpg']
file_names = [os.path.basename(path) for path in file_paths]

# Dynamically getting the class name being predicted (from training generator)
class_name = list(train_generator.class_indices.keys())[1]  # assuming class '1' is the positive class

# Submission DataFrame using the real class name
submission = pd.DataFrame({
    'filename': file_names,
    f'probability_of_{class_name}': probabilities.flatten()  # ðŸ‘ˆ dynamic column name
})


# Save to CSV
submission.to_csv('submission.csv', index=False)
print(f"âœ… Submission file created with class column: probability_of_{class_name}")


submission




