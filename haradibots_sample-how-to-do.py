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


import pandas as pd

df = pd.read_csv('/kaggle/input/handmade-metadata/metaData.csv')
df


# ============================================================
# Cat or Dog Image Classification Challenge
# Starter Notebook
# ============================================================

import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# Load metadata
# ---------------------------
meta_path = "/kaggle/input/handmade-metadata/metaData.csv"
df = pd.read_csv(meta_path)

print(" Metadata loaded successfully!")
print(df.head())

# ---------------------------
# Define image folder
# ---------------------------
img_folder = "/kaggle/input/cat-or-dog-image-classification-challenge/data_mixed"

# ---------------------------
# Helper function to load image by ID
# ---------------------------
def load_image(img_id):
    img_path = os.path.join(img_folder, f"{img_id}.jpg")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ---------------------------
# Example: Display few images
# ---------------------------
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    img_id = df.loc[i, "img"]
    label = df.loc[i, "is_dog"]
    image = load_image(img_id)
    ax.imshow(image)
    ax.set_title(f"ID: {img_id} | Label: {'Dog' if label == 1 else 'Cat'}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------
# Train-Test Split (Example)
# ---------------------------
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["is_dog"], random_state=42)
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

# ---------------------------
# Ready for model training!
# ---------------------------



!pip install --upgrade pip
!pip install "protobuf<4.24" --force-reinstall



# ============================================================
# Cat or Dog Image Classification Challenge - Full Starter Code
# With Data Augmentation
# ============================================================

import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------
# Load Metadata
# ---------------------------
meta_path = "/kaggle/input/handmade-metadata/metaData.csv"
df = pd.read_csv(meta_path)
print(" Metadata loaded successfully!")
print(df.head())

# ---------------------------
# Base Image Folder
# ---------------------------
IMG_DIR = "/kaggle/input/cat-or-dog-image-classification-challenge/data_mixed"

# ---------------------------
# Image Size and Parameters
# ---------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50  # participants can increase this later

# ---------------------------
# Load Images into Arrays
# ---------------------------
def load_image(img_id):
    path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img

# Load all images and labels
images = np.array([load_image(i) for i in df["img"]])
labels = np.array(df["is_dog"])

print(f"Loaded {len(images)} images successfully!")

# ---------------------------
# Split Train and Validation
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# ---------------------------
# Data Augmentation
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# ---------------------------
# Build CNN Model
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# Train Model
# ---------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1
)

# ---------------------------
# Visualize Training
# ---------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# ---------------------------
# Generate Predictions for Submission
# ---------------------------
preds = model.predict(val_gen, verbose=1).ravel()
preds = np.clip(preds, 0, 1)

submission = pd.DataFrame({
    "img": df["img"].iloc[y_val.index] if hasattr(y_val, "index") else np.arange(len(y_val)),
    "is_dog": preds[:len(y_val)]
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")
print(submission.head())


