import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


# âœ… 1ï¸�âƒ£ Load Dataset Metadata
train_dir = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
labels_csv = "/kaggle/input/state-farm-distracted-driver-detection/driver_imgs_list.csv"


labels_df = pd.read_csv(labels_csv)
labels_df


# âœ… 2ï¸�âƒ£ Extract Image Paths & Labels (Ensuring Correct Mapping)
image_paths = []
labels = []

for class_name in sorted(os.listdir(train_dir)):  # Ensure correct order
    class_dir = os.path.join(train_dir, class_name)
    if os.path.isdir(class_dir):  # Ensure it's a folder
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            labels.append(int(class_name[1]))  # Convert 'c0' â†’ 0, ..., 'c9' â†’ 9

# Convert to NumPy Arrays
image_paths = np.array(image_paths)
labels = np.array(labels)


# âœ… 3ï¸�âƒ£ Split Data (80% Train, 20% Validation)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)


# âœ… 4ï¸�âƒ£ Preprocess Images Before Training (Save as NumPy)
def preprocess_and_save(image_paths, labels, save_path):
    images = []
    for path in image_paths:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224]) / 255.0  # Normalize
        images.append(image.numpy())  # Convert to NumPy
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    
    np.savez_compressed(save_path, images=images, labels=labels)
    print(f"âœ… Saved preprocessed dataset at {save_path}")


# Preprocess and Save Data
preprocess_and_save(train_paths, train_labels, "train_data.npz")
preprocess_and_save(val_paths, val_labels, "val_data.npz")



# âœ… 5ï¸�âƒ£ Load Preprocessed Data
train_data = np.load("/kaggle/working/train_data.npz")
val_data = np.load("/kaggle/working/val_data.npz")


train_images, train_labels = train_data["images"], train_data["labels"]
val_images, val_labels = val_data["images"], val_data["labels"]


# âœ… 6ï¸�âƒ£ Create TensorFlow Datasets (No Preprocessing Needed)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE)


# âœ… 7ï¸�âƒ£ Load Pretrained MobileNetV2
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model


# âœ… 8ï¸�âƒ£ Build Custom Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")  # 10 classes
])


# âœ… 9ï¸�âƒ£ Compile Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# âœ… ğŸ”Ÿ Train the Model (Now Faster & Stable)
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)




