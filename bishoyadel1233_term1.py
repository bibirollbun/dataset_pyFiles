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


from glob import glob

from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
# Install required libraries (uncomment if needed)
# !pip install pydicom opencv-python tensorflow

import pydicom
import numpy as np
import cv2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input


# --- Step 1: Load DICOM ---
def load_dicom(path):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array.astype(np.float32)
    image /= np.max(image)
        
    return image

def preprocess_dicom(image, mean, std):
    # Convert to 3 channels if grayscale
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    
    # Clip top and bottom 0.5% of pixel values
    image = image.astype(np.float32)
    low_val, high_val = np.percentile(image, [0.5, 99.5])
    image = np.clip(image, low_val, high_val)
    
    # Normalize to [0, 1] range
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # Apply model's mean/std normalization
    image = (image - mean) / std
    
    # Resize to target size
    image = cv2.resize(image, (320, 320))
    
    return image

# --- Step 3: Load Model and Predict ---
# Load pretrained DenseNet121
clf = DenseNet121(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
)


x = clf.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(14, activation="sigmoid")(x)
for layer in clf.layers:
    layer.trainable = False
clf = Model(inputs=clf.input, outputs=predictions)



clf.load_weights('/kaggle/input/bisho-mo3edo-3medo/xray_class_weights.best.hdf5')









# 2. Function to load and preprocess images
def load_and_preprocess_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, input_size)  # Resize to model input size
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    return img

mean = np.array([0.52625063, 0.52625063, 0.52625063])
std = np.array([0.25296255, 0.25296255, 0.25296255])
print(mean, std)


# Predict
dicom_path = "/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train/321c111713c3ee5385db0effb54ff568.dicom"  # <- CHANGE THIS

# Load and preprocess
dicom_image = load_dicom(dicom_path)
processed_image = preprocess_dicom(dicom_image, mean, std)  # Shape: (320, 320, 3)
input_array = np.expand_dims(processed_image, axis=0)
print(input_array.shape)

prediction = clf.predict(input_array)
print("Prediction shape:", prediction.shape)  
print(prediction)



threshold = 0.5
binary_predictions = (prediction > threshold).astype(int)

# Get the specific disease predictions (convert from 1-based to 0-based indices)
disease_indices = {
    'Emphysema': 3,  # 3 (0-based)
    'Hernia': 4,     # 4 (0-based)
    'Edema': 6,      # 6 (0-based)
    'Pneumonia': 7   # 7 (0-based)
}

# Extract and print the specific predictions
print("Thresholded predictions for specific diseases:")
for disease, idx in disease_indices.items():
    pred = binary_predictions[0, idx]  # [0] because prediction shape is (1, 14)
    print(f"{disease}: {'Positive' if pred == 1 else 'Negative'} (Probability: {prediction[0, idx]:.4f})")




