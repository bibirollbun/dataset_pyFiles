import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Define Constants
IMG_SIZE = 224
BATCH_SIZE = 32

# Kaggle APTOS 2019 Dataset Paths
APTOS_DATASET_DIR = "/kaggle/input/aptos2019-blindness-detection"
APTOS_CSV = os.path.join(APTOS_DATASET_DIR, "train.csv")
APTOS_IMAGES_DIR = os.path.join(APTOS_DATASET_DIR, "train_images")

# Your Custom Dataset Paths
MY_DATASET_DIR = "/kaggle/input/mydataset"  # Update with your dataset path
MY_CSV = os.path.join(MY_DATASET_DIR, "gan_test_csv.csv")
MY_IMAGES_DIR = os.path.join(MY_DATASET_DIR, "generated_images")

# Validate Paths
assert os.path.exists(APTOS_DATASET_DIR), f"❌ Kaggle dataset not found: {APTOS_DATASET_DIR}"
assert os.path.exists(APTOS_CSV), f"❌ Kaggle CSV not found: {APTOS_CSV}"
assert os.path.exists(APTOS_IMAGES_DIR), f"❌ Kaggle image directory not found: {APTOS_IMAGES_DIR}"
assert os.path.exists(MY_DATASET_DIR), f"❌ Custom dataset not found: {MY_DATASET_DIR}"
assert os.path.exists(MY_CSV), f"❌ Custom CSV not found: {MY_CSV}"
assert os.path.exists(MY_IMAGES_DIR), f"❌ Custom image directory not found: {MY_IMAGES_DIR}"

# Load CSV Files
aptos_df = pd.read_csv(APTOS_CSV)
my_df = pd.read_csv(MY_CSV)

# Ensure image filenames have .png extension
aptos_df["id_code"] = aptos_df["id_code"].astype(str) + ".png"
my_df["id_code"] = my_df["id_code"].astype(str) + ".png"

# Verify that all image files exist
aptos_images = set(os.listdir(APTOS_IMAGES_DIR))
my_images = set(os.listdir(MY_IMAGES_DIR))

aptos_df = aptos_df[aptos_df["id_code"].isin(aptos_images)]
my_df = my_df[my_df["id_code"].isin(my_images)]

# Merge Both Datasets
train_df = pd.concat([aptos_df, my_df], ignore_index=True)

# Convert labels to string format
train_df["diagnosis"] = train_df["diagnosis"].astype(str)

# Split Data into Training & Validation (Stratified)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["diagnosis"])

# Data Augmentation & Preprocessing
data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

# Custom Generator to Handle Multiple Image Directories
class MultiDirectoryDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size, mode="train"):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.mode = mode
        self.indices = np.arange(len(self.dataframe))
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indices]
        
        images = []
        labels = []
        
        for _, row in batch_data.iterrows():
            img_path = os.path.join(APTOS_IMAGES_DIR, row["id_code"])
            if not os.path.exists(img_path):  # If not in Kaggle dataset, check your dataset
                img_path = os.path.join(MY_IMAGES_DIR, row["id_code"])
            
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            
            images.append(img)
            labels.append(int(row["diagnosis"]))
        
        return np.array(images), np.array(labels)

# Use the custom generator
train_gen = MultiDirectoryDataGenerator(train_df, BATCH_SIZE, mode="train")
val_gen = MultiDirectoryDataGenerator(val_df, BATCH_SIZE, mode="val")

# Load VGG16 Model (Transfer Learning)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True  # Fine-tuning enabled

# Define Model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 classes (0–4 for DR severity)
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define Callbacks
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "vgg16_best_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=14,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[model_checkpoint, early_stopping]
)

# Save Final Model
model.save("vgg16_finetuned.keras")

# Evaluate Model
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Get Predictions
y_true = val_df["diagnosis"].astype(int).values
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Plot Loss and Accuracy
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Loss plot
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Val Loss')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss vs Epochs")
ax[0].legend()

# Accuracy plot
ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Val Accuracy')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Accuracy vs Epochs")
ax[1].legend()

plt.show()



# Training Accuracy
train_loss, train_accuracy = model.evaluate(train_gen)
print(f"Recalculated Training Accuracy: {train_accuracy:.4f}")

# Validation Accuracy
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Recalculated Validation Accuracy: {val_accuracy:.4f}")



import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
DATASET_DIR = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train_images")

# Validate dataset paths
assert os.path.exists(DATASET_DIR), f"❌ Dataset directory not found: {DATASET_DIR}"
assert os.path.exists(TRAIN_CSV), f"❌ CSV file not found: {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMAGES_DIR), f"❌ Image directory not found: {TRAIN_IMAGES_DIR}"

# Load train.csv
train_df = pd.read_csv(TRAIN_CSV)

# Ensure image filenames have .png extension
train_df["id_code"] = train_df["id_code"].astype(str) + ".png"

# Verify that all image files exist
all_image_files = set(os.listdir(TRAIN_IMAGES_DIR))  # List all images in directory
train_df = train_df[train_df["id_code"].isin(all_image_files)]  # Keep only existing images

if train_df.empty:
    raise ValueError("❌ No valid image filenames found in the dataset. Check filenames and dataset directory.")

# Convert labels to string format for flow_from_dataframe()
train_df["diagnosis"] = train_df["diagnosis"].astype(str)

# Split data into training and validation sets (stratified)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["diagnosis"])

# Data Augmentation & Preprocessing
data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

# Create Data Generators
train_gen = data_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_IMAGES_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)
val_gen = val_gen.flow_from_dataframe(
    dataframe=val_df,
    directory=TRAIN_IMAGES_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# Load VGG16 Model (Transfer Learning)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True  # Fine-tuning enabled

# Define Model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 classes (0–4 for DR severity)
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define Callbacks
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "vgg16_best_model.keras",  # ✅ Use .keras instead of .h5
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stops training if no improvement for 5 epochs
    restore_best_weights=True
)

# Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=14,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE,
    callbacks=[model_checkpoint, early_stopping]  # ✅ Callbacks added
)

# Save Final Model
model.save("vgg16_finetuned.keras")

# Evaluate Model
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Get Predictions
y_true = val_df["diagnosis"].astype(int).values
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Plot Loss and Accuracy
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Loss plot
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Val Loss')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss vs Epochs")
ax[0].legend()

# Accuracy plot
ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Val Accuracy')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Accuracy vs Epochs")
ax[1].legend()

plt.show()



import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Define Constants
IMG_SIZE = 224
BATCH_SIZE = 32

# Kaggle APTOS 2019 Dataset Paths
APTOS_DATASET_DIR = "/kaggle/input/aptos2019-blindness-detection"
APTOS_CSV = os.path.join(APTOS_DATASET_DIR, "train.csv")
APTOS_IMAGES_DIR = os.path.join(APTOS_DATASET_DIR, "train_images")

# Your Custom Dataset Paths
MY_DATASET_DIR = "/kaggle/input/mydataset"  # Update with your dataset path
MY_CSV = "/kaggle/input/vishal-images/vishal_images.csv"
MY_IMAGES_DIR = os.path.join(MY_DATASET_DIR, "generated_images")

# Validate Paths
assert os.path.exists(APTOS_DATASET_DIR), f"❌ Kaggle dataset not found: {APTOS_DATASET_DIR}"
assert os.path.exists(APTOS_CSV), f"❌ Kaggle CSV not found: {APTOS_CSV}"
assert os.path.exists(APTOS_IMAGES_DIR), f"❌ Kaggle image directory not found: {APTOS_IMAGES_DIR}"
assert os.path.exists(MY_DATASET_DIR), f"❌ Custom dataset not found: {MY_DATASET_DIR}"
assert os.path.exists(MY_CSV), f"❌ Custom CSV not found: {MY_CSV}"
assert os.path.exists(MY_IMAGES_DIR), f"❌ Custom image directory not found: {MY_IMAGES_DIR}"

# Load CSV Files
aptos_df = pd.read_csv(APTOS_CSV)
my_df = pd.read_csv(MY_CSV)

# Ensure image filenames have .png extension
aptos_df["id_code"] = aptos_df["id_code"].astype(str) + ".png"
my_df["id_code"] = my_df["id_code"].astype(str) + ".png"

# Verify that all image files exist
aptos_images = set(os.listdir(APTOS_IMAGES_DIR))
my_images = set(os.listdir(MY_IMAGES_DIR))

aptos_df = aptos_df[aptos_df["id_code"].isin(aptos_images)]
my_df = my_df[my_df["id_code"].isin(my_images)]

# Merge Both Datasets
train_df = pd.concat([aptos_df, my_df], ignore_index=True)

# Convert labels to string format
train_df["diagnosis"] = train_df["diagnosis"].astype(str)

# Split Data into Training & Validation (Stratified)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["diagnosis"])

# Data Augmentation & Preprocessing
data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

# Custom Generator to Handle Multiple Image Directories
class MultiDirectoryDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size, mode="train"):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.mode = mode
        self.indices = np.arange(len(self.dataframe))
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indices]
        
        images = []
        labels = []
        
        for _, row in batch_data.iterrows():
            img_path = os.path.join(APTOS_IMAGES_DIR, row["id_code"])
            if not os.path.exists(img_path):  # If not in Kaggle dataset, check your dataset
                img_path = os.path.join(MY_IMAGES_DIR, row["id_code"])
            
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            
            images.append(img)
            labels.append(int(row["diagnosis"]))
        
        return np.array(images), np.array(labels)

# Use the custom generator
train_gen = MultiDirectoryDataGenerator(train_df, BATCH_SIZE, mode="train")
val_gen = MultiDirectoryDataGenerator(val_df, BATCH_SIZE, mode="val")

# Load VGG16 Model (Transfer Learning)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True  # Fine-tuning enabled

# Define Model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 classes (0–4 for DR severity)
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define Callbacks
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "vgg16_best_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=14,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
    callbacks=[model_checkpoint, early_stopping]
)

# Save Final Model
model.save("vgg16_finetuned.keras")

# Evaluate Model
val_loss, val_accuracy = model.evaluate(val_gen)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Get Predictions
y_true = val_df["diagnosis"].astype(int).values
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Plot Loss and Accuracy
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Loss plot
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Val Loss')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss vs Epochs")
ax[0].legend()

# Accuracy plot
ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Val Accuracy')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Accuracy vs Epochs")
ax[1].legend()

plt.show()


