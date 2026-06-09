# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


df = pd.read_csv('../input/animal-clef-2025/metadata.csv')
df.tail()


#By Jocelyn Dumlao

# Directories for each animal category
data_dirs = {    
    "SeaTurtlesD": "/kaggle/input/animal-clef-2025/images/SeaTurtleID2022/database/turtles-data/data/images/t001",
    "SeaTurtlesQ": "/kaggle/input/animal-clef-2025/images/SeaTurtleID2022/query/images",
    "LynxsD": "/kaggle/input/animal-clef-2025/images/LynxID2025/database",
    "LynxsQ": "/kaggle/input/animal-clef-2025/images/LynxID2025/query",
    "SalamandersD": "/kaggle/input/animal-clef-2025/images/SalamanderID2025/database/images",
    "SalamandersQ": "/kaggle/input/animal-clef-2025/images/SalamanderID2025/query/images"
        
}


#By Jocelyn Dumlao

# Function to load and display images from each directory
def show_sample_images():
    for label, dir_path in data_dirs.items():
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue

        # Load the first few images from the directory
        sample_images = []
        for root, _, files in os.walk(dir_path):
            for img_name in files[:5]:  # Load 5 images for display
                img_path = os.path.join(root, img_name)
                if img_path.lower().endswith(('.JPG', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (150, 150))
                        sample_images.append(img)
                    if len(sample_images) == 5:
                        break
            if len(sample_images) == 5:
                break
                
        # Plot the images
        plt.figure(figsize=(10, 10))
        for i, img in enumerate(sample_images):
            plt.subplot(1, 5, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"{label}")
        plt.show()

# Call the function to display images
show_sample_images()


#By Jocelyn Dumlao

# Set random seed for reproducibility
tf.random.set_seed(42)

# Directories for each bird category
data_dirs = {
    "SeaTurtlesD": "/kaggle/input/animal-clef-2025/images/SeaTurtleID2022/database/turtles-data/data/images/t001",
    "SeaTurtlesQ": "/kaggle/input/animal-clef-2025/images/SeaTurtleID2022/query/images",
    "LynxsD": "/kaggle/input/animal-clef-2025/images/LynxID2025/database",
    "LynxsQ": "/kaggle/input/animal-clef-2025/images/LynxID2025/query",
    "SalamandersD": "/kaggle/input/animal-clef-2025/images/SalamanderID2025/database/images",
    "SalamandersQ": "/kaggle/input/animal-clef-2025/images/SalamanderID2025/query/images"    
}

IMG_SIZE = 150
X, Y = [], []


#By Jocelyn Dumlao
# Function to load images and labels
def load_images():
    for label, dir_path in data_dirs.items():
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            continue

        for root, _, files in os.walk(dir_path):  # Walk through all subdirectories
            for img_name in tqdm(files):
                img_path = os.path.join(root, img_name)

                # Ensure it's a valid image file
                if not img_path.lower().endswith(('.JPG', '.jpg', '.jpeg')):
                    print(f"Skipping {img_path}, not a valid image file.")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {img_path}. Skipping.")
                    continue  # Skip unreadable images

                try:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    Y.append(label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


#By Jocelyn Dumlao

# Load the dataset
load_images()

# Convert to numpy arrays
if len(X) == 0 or len(Y) == 0 or len(X) != len(Y):
    raise ValueError(f"Mismatch in dataset sizes: X={len(X)}, Y={len(Y)}")

X = np.array(X, dtype='float32') / 255.0  # Normalize images
Y = np.array([list(data_dirs.keys()).index(y) for y in Y], dtype='int32')

# Shuffle and split dataset
X, Y = shuffle(X, Y, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Dataset successfully loaded: Train={len(X_train)}, Test={len(X_test)}")


#By Jocelyn Dumlao

# Model building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(data_dirs), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#By Jocelyn Dumlao

# Train the model
history = model.fit(X_train, Y_train, batch_size=32, epochs=2, validation_split=0.2) #Epochs=10

# Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Evaluate on test data
loss, acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

