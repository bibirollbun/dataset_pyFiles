import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization

print("Loaded all libraries")


# 1. DATA LOADING AND EXPLORATION


# Đường dẫn đến thư mục chứa ảnh và file labels
train_path = "/kaggle/input/histopathologic-cancer-detection/train"
labels_path = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"
random_seed = 42

# Đọc file labels
df_labels = pd.read_csv(labels_path)
print(f"Total images: {len(df_labels)}")
print(f"\nLabel distribution:\n{df_labels['label'].value_counts()}")
print(f"\nFirst few rows:\n{df_labels.head()}")


# 2. LOAD IMAGES AND LABELS


def load_images_and_labels(df, img_path, img_size=(227, 227), max_samples=None):
    """
    Load images and labels from Histopathologic Cancer Detection dataset
    
    Parameters:
    - df: DataFrame containing image ids and labels
    - img_path: Path to image directory
    - img_size: Target size for resizing (default: 227x227 for AlexNet)
    - max_samples: Limit number of samples (None = load all)
    """
    img_lst = []
    labels = []
    
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
    
    for idx, row in df.iterrows():
        img_id = row['id']
        label = row['label']
        
        # Construct full image path
        img_file = os.path.join(img_path, f"{img_id}.tif")
        
        # Check if file exists
        if not os.path.exists(img_file):
            continue
            
        # Read and process image
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size (227x227 for AlexNet)
        img_array = Image.fromarray(img, 'RGB')
        resized_img = img_array.resize(img_size)
        
        img_lst.append(np.array(resized_img))
        labels.append(label)
        
        # Progress indicator
        if (idx + 1) % 10000 == 0:
            print(f"Loaded {idx + 1} images...")
    
    return np.array(img_lst), np.array(labels)

# Load images (sử dụng max_samples để test nhanh, bỏ tham số này để load toàn bộ)
print("\nLoading images...")
images, labels = load_images_and_labels(df_labels, train_path, max_samples=5000)

print(f"\nImages shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Data types: {images.dtype}, {labels.dtype}")


# 3. VISUALIZE RANDOM SAMPLES


def display_rand_images(images, labels, title_prefix="Label"):
    """Display 9 random images with their labels"""
    plt.figure(figsize=(15, 10))
    
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        
        # Random index
        idx = np.random.randint(0, len(images))
        
        plt.imshow(images[idx])
        plt.title(f'{title_prefix}: {labels[idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

print("\nDisplaying sample images...")
display_rand_images(images, labels)

