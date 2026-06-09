import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, LeakyReLU, Input
from keras.datasets import cifar10
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/train.csv")
train


# Extract filename only (remove folder name)
train["filename"] = train["file_name"].apply(lambda x: os.path.basename(x))

# Create a mapping of filename -> label
label_dict = dict(zip(train["filename"], train["label"]))

# Path to images
image_dir = "/kaggle/input/ai-vs-human-generated-dataset/train_data"

# Lists to store images and labels
x, y = [], []

# Process each image
for file in os.listdir(image_dir):
    if file in label_dict:  # Ensure the file exists in the CSV
        # Load and preprocess image
        image = Image.open(os.path.join(image_dir, file))
        image = image.convert('RGB')
        image = image.resize((128, 128))
        x.append(np.array(image))
        
        # Get corresponding label
        y.append(label_dict[file])

# Convert to NumPy arrays
x = np.array(x, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int32)

print("Dataset Loaded:", x.shape, y.shape)


# Get the first image path from CSV
image_path = os.path.join(trainpath, train.iloc[0]['file_name'])

print("Image Path:", image_path)
print("File Exists:", os.path.exists(image_path))


from tensorflow.keras import layers, models

# Load EfficientNetB7 with pre-trained weights
base_model = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base layers

# Build the Model
nnmodel = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Classification output
])

# Compile Model
nnmodel.compile(
    optimizer='adam', 
    loss='binary_crossentropy',  # For binary classification
    metrics=['accuracy']
)

# Check Model Summary
nnmodel.summary()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
trained = nnmodel.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test), verbose=1)

