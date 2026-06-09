# Hide warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Common imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

np.random.seed(42)
tf.random.set_seed(42)

%matplotlib inline

print("Imports loaded!")


# Check Kaggle environment
import os

print("Running in Kaggle Notebook")

# Data paths for Dogs vs Cats competition
BASE_PATH = '/kaggle/input/dogs-vs-cats/'
TRAIN_ZIP = BASE_PATH + 'train.zip'
TEST_ZIP = BASE_PATH + 'test1.zip'

# Verify it works
import os
print(f"✓ Data found: {os.path.exists(BASE_PATH)}")
print(f"✓ Contents: {os.listdir(BASE_PATH)}")

print("Kaggle API already configured!")



# Create directory structure
import os
import zipfile
import shutil

# Use the paths we defined earlier
TRAIN_ZIP = '/kaggle/input/dogs-vs-cats/train.zip'
TEST_ZIP = '/kaggle/input/dogs-vs-cats/test1.zip'

# Create working directory
os.makedirs('/kaggle/working/data', exist_ok=True)

# Extract train data
print("Extracting training data...")
with zipfile.ZipFile(TRAIN_ZIP, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/data/')

# Extract test data
print("Extracting test data...")
with zipfile.ZipFile(TEST_ZIP, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/data/')

print("✓ Dataset extracted!")
print(f"Training images: {len(os.listdir('/kaggle/working/data/train'))}")
print(f"Test images: {len(os.listdir('/kaggle/working/data/test1'))}")


# Organize data into train/validation split
organized_dir = '/kaggle/working/data/organized'
os.makedirs(f'{organized_dir}/cats', exist_ok=True)
os.makedirs(f'{organized_dir}/dogs', exist_ok=True)

# Move files to category folders
print("Organizing by category...")
for filename in os.listdir('/kaggle/working/data/train'):
    src = f'/kaggle/working/data/train/{filename}'
    if filename.startswith('cat'):
        shutil.copy(src, f'{organized_dir}/cats/{filename}')
    else:
        shutil.copy(src, f'{organized_dir}/dogs/{filename}')

print(f"✓ Organized into cats ({len(os.listdir(organized_dir + '/cats'))}) and dogs ({len(os.listdir(organized_dir + '/dogs'))})")


# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Show some cats
cat_files = os.listdir(organized_dir + '/cats')[:5]
for i, filename in enumerate(cat_files):
    img_path = os.path.join(organized_dir, 'cats', filename)
    img = plt.imread(img_path)
    axes[0, i].imshow(img)
    axes[0, i].set_title('Cat', fontsize=12)
    axes[0, i].axis('off')

# Show some dogs
dog_files = os.listdir(organized_dir + '/dogs')[:5]
for i, filename in enumerate(dog_files):
    img_path = os.path.join(organized_dir, 'dogs', filename)
    img = plt.imread(img_path)
    axes[1, i].imshow(img)
    axes[1, i].set_title('Dog', fontsize=12)
    axes[1, i].axis('off')

plt.suptitle('Sample Images from Dataset', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# Image parameters
IMG_SIZE = 150
BATCH_SIZE = 32

# Load datasets with 80/20 split
training_ds = tf.keras.utils.image_dataset_from_directory(
    'data/organized',
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    'data/organized',
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

print("Data loaded")


# Create preprocessing layer (Chapter 14 approach)
preprocess = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255)
])

# Apply preprocessing to datasets
training_ds = training_ds.map(lambda x, y: (preprocess(x), y))
training_ds = training_ds.prefetch(1)

validation_ds = validation_ds.map(lambda x, y: (preprocess(x), y))
validation_ds = validation_ds.prefetch(1)

print("Data Normalized")


# Display sample images from training set
plt.figure(figsize=(15, 6))

for images, labels in training_ds.take(1):
    for i in range(min(10, len(images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title('Dog' if labels[i] == 1 else 'Cat', fontsize=12)
        plt.axis('off')

plt.suptitle('Sample Images from Training Set (with preprocessing)', fontsize=16)
plt.tight_layout()
plt.show()

print("Note: These images have been resized to 224x224 and normalized to [0,1] range")


from functools import partial

# Create a partial function for default Conv2D layers
# Keep it consistant
DefaultConv2D = partial(tf.keras.layers.Conv2D,
                       kernel_size=3,
                       padding="same",
                       activation="relu",
                       kernel_initializer="he_normal")

# NEWER MORE COMPLEX LIKE THE BOOK
def create_cnn_from_scratch():
    """
    Custom CNN architecture
    Using 4 convolutional blocks with progressive filter increases.
    """
    model = tf.keras.Sequential([
        # First convolutional block - Chapter 14 pattern
        DefaultConv2D(filters=32, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(),

        # Second convolutional block
        DefaultConv2D(filters=64),
        tf.keras.layers.MaxPooling2D(),

        # Third convolutional block
        DefaultConv2D(filters=128),
        tf.keras.layers.MaxPooling2D(),

        # Fourth convolutional block just to follow the book's deeper architecture
        #tf.keras.layers.Conv2D(filters=256),
        #tf.keras.layers.MaxPooling2D(),

        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),  # Regularization
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=1, activation='sigmoid')  # Binary classification
    ])

    return model

# Create Model A
model_a = create_cnn_from_scratch()

# Compile with Adam optimizer
model_a.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Standard learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model A: CNN from Scratch")
model_a.summary()
print(f"\nTotal parameters: {model_a.count_params():,}")


# NEW WAY - TRAINING MODEL A
# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop if no improvement for 5 epochs
    restore_best_weights=True,
    verbose=1
)

print("Training Model A (CNN from Scratch)...")
# Track training time
from datetime import datetime
import time

start_time = time.time()
start_datetime = datetime.now()

history_a = model_a.fit(
    training_ds,
    epochs=10,
    validation_data=validation_ds,
    callbacks=[early_stopping],
    verbose=1
)

print("\nModel A training complete!")
end_time = time.time()
end_datetime = datetime.now()
training_time_a = end_time - start_time
print(f"Stopped at epoch {len(history_a.history['accuracy'])}")
print(f"Training Time: {training_time_a:.2f} seconds ({training_time_a/60:.2f} minutes)")


print("Creating ResNet50-specific datasets...")

# Load fresh datasets WITHOUT [0,1] normalization (ResNet needs [0, 255])
training_ds_resnet = tf.keras.utils.image_dataset_from_directory(
    'data/organized',
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

validation_ds_resnet = tf.keras.utils.image_dataset_from_directory(
    'data/organized',
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Define ResNet50 preprocessing function
def preprocess_for_resnet50(image, label):
    # Images are already in [0, 255] range from raw loading
    # Apply ResNet50's preprocessing (BGR conversion + ImageNet mean subtraction)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label

# Apply preprocessing
training_ds_resnet = training_ds_resnet.map(preprocess_for_resnet50)
training_ds_resnet = training_ds_resnet.prefetch(1)

validation_ds_resnet = validation_ds_resnet.map(preprocess_for_resnet50)
validation_ds_resnet = validation_ds_resnet.prefetch(1)

print("ResNet50 datasets ready..")


# Load pretrained tf.keras.applications.ResNet50
print("Loading pretrained tf.keras.applications.ResNet50 from ImageNet...")

base_model = tf.keras.applications.ResNet50(
    weights='imagenet',  # Use ImageNet pretrained weights
    include_top=False,   # Exclude the final classification layer
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model (only train the new top layers)
base_model.trainable = False

print(f"Base model loaded: {base_model.name}")
print(f"Total layers in base: {len(base_model.layers)}")


# Build Model B with custom classification head

# Build the model page 520
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)  # Note: training=False for inference mode
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # book recommends this over Flatten
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_b = tf.keras.Model(inputs, outputs)

# Only the top layers
model_b.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Same as Model A
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model B: Transfer Learning with tf.keras.applications.ResNet50")
model_b.summary()
print(f"\nTrainable parameters: {sum([tf.size(w).numpy() for w in model_b.trainable_weights]):,}")
print(f"Non-trainable parameters: {sum([tf.size(w).numpy() for w in model_b.non_trainable_weights]):,}")


print("Training only the top layers...\n")

# Same callbacks as Model A
early_stopping_b = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

#Record timer
start_time = time.time()
start_datetime = datetime.now()

# Training should converge quickly bec of pretrained features
history_b = model_b.fit(
    training_ds_resnet,
    epochs=5,  # Fewer epochs for B, 10 is too long
    validation_data=validation_ds_resnet,
    callbacks=[early_stopping_b],
    verbose=1
)

print("\nModel B Training complete!")
end_time = time.time()
end_datetime = datetime.now()
training_time_b = end_time - start_time
print(f"Stopped at epoch {len(history_b.history['accuracy'])}")
print(f"Training Time: {training_time_b:.2f} seconds ({training_time_b/60:.2f} minutes)")


def plot_models_comparison(history_a, history_b):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Model A - Accuracy
    axes[0, 0].plot(history_a['accuracy'], label='Training', linewidth=2)
    axes[0, 0].plot(history_a['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model A - CNN from Scratch - Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Model A - Loss
    axes[0, 1].plot(history_a['loss'], label='Training', linewidth=2)
    axes[0, 1].plot(history_a['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model A - CNN from Scratch - Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Model B - Accuracy
    axes[1, 0].plot(history_b['accuracy'], label='Training', linewidth=2, color='green')
    axes[1, 0].plot(history_b['val_accuracy'], label='Validation', linewidth=2, color='orange')
    axes[1, 0].set_title('Model B - Transfer Learning - Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Model B - Loss
    axes[1, 1].plot(history_b['loss'], label='Training', linewidth=2, color='green')
    axes[1, 1].plot(history_b['val_loss'], label='Validation', linewidth=2, color='orange')
    axes[1, 1].set_title('Model B - Transfer Learning - Loss', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_models_comparison(history_a.history, history_b.history)


## Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Model A - collect everything in one loop
y_true_a, y_pred_a = [], []
for images, labels in validation_ds:
    preds = model_a.predict(images, verbose=0)
    y_true_a.extend(labels.numpy())
    y_pred_a.extend((preds > 0.5).astype(int).flatten())

# Model B - collect everything in one loop
y_true_b, y_pred_b = [], []
for images, labels in validation_ds_resnet:
    preds = model_b.predict(images, verbose=0)
    y_true_b.extend(labels.numpy())
    y_pred_b.extend((preds > 0.5).astype(int).flatten())

# Create matrices
cm_a = confusion_matrix(y_true_a, y_pred_a)
cm_b = confusion_matrix(y_true_b, y_pred_b)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
ax1.set_title('Model A: From Scratch')

sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
ax2.set_title('Model B: Transfer Learning')

plt.tight_layout()
plt.show()

