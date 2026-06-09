
import numpy as np                
import pandas as pd               # tabular data handling
import matplotlib.pyplot as plt   # plotting and visualizations

# File handling
import os                         # filesystem operations (paths, listing files)
import zipfile                    # open .zip files if dataset is zipped

# Image processing
import cv2                        # OpenCV: image I/O and transformations

# Machine learning utilities
from sklearn.model_selection import train_test_split  # split data into train/validation

# Keras (TensorFlow backend) high-level model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D,
    Dropout, Conv2DTranspose, Input
)
from tensorflow.keras.callbacks import EarlyStopping



# Extracting datasets from .zip archives

# Paths to the dataset location (input) and our working folder
data_dir = "/kaggle/input/denoising-dirty-documents/"
work_dir = "/kaggle/working/"

# Helper function to unzip any file into the working directory
def unzip_file(zip_filename, destination):
    """
    Extracts the contents of a zip file into the specified folder.
    
    Parameters:
        zip_filename (str): Path to the .zip file
        destination (str): Folder where contents will be extracted
    """
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(destination)

# Unzip all required archives
unzip_file(os.path.join(data_dir, "train.zip"), work_dir)
unzip_file(os.path.join(data_dir, "test.zip"), work_dir)
unzip_file(os.path.join(data_dir, "train_cleaned.zip"), work_dir)
unzip_file(os.path.join(data_dir, "sampleSubmission.csv.zip"), work_dir)



train_images = sorted(os.listdir(os.path.join(work_dir, "train")))
clean_images = sorted(os.listdir(os.path.join(work_dir, "train_cleaned")))
test_images  = sorted(os.listdir(os.path.join(work_dir, "test")))

print(len(train_images), len(clean_images), len(test_images))



def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # load in grayscale directly
    img = cv2.resize(img, (540, 420)).astype("float32") / 255.0  # resize & normalize
    return img.reshape(420, 540, 1)                # add channel dimension



# Preprocess all images into NumPy-ready format

# Process noisy training images
train = [load_image(os.path.join(work_dir, "train", fname)) for fname in train_images]

# Process cleaned training images
train_cleaned = [load_image(os.path.join(work_dir, "train_cleaned", fname))for fname in clean_images]

# Process test images
test = [load_image(os.path.join(work_dir, "test", fname)) for fname in test_images]



# sample noisy vs. cleaned image pairs
plt.figure(figsize=(15, 25))

for i in range(0, 6, 2):
    # Noisy image
    plt.subplot(3, 2, i + 1)
    plt.imshow(train[i][:, :, 0], cmap='gray')
    plt.title(f"Noisy: {train_images[i]}")
    plt.xticks([]); plt.yticks([])

    # Cleaned image
    plt.subplot(3, 2, i + 2)
    plt.imshow(train_cleaned[i][:, :, 0], cmap='gray')
    plt.title(f"Cleaned: {train_images[i]}")
    plt.xticks([]); plt.yticks([])

plt.tight_layout()
plt.show()



# Convert lists to NumPy arrays and create train/validation split
X_train = np.array(train, dtype="float32")
y_train = np.array(train_cleaned, dtype="float32")
X_test  = np.array(test, dtype="float32")

# Split data: 85% training, 15% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)




c_autoencoder = Sequential()

# Encoder
# Input: (420, 540, 1)
c_autoencoder.add(
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(420, 540, 1)))
# Output shape after Conv2D (padding='same', stride=1):
# (420, 540, 32) â€” spatial dims unchanged, channels = 32

c_autoencoder.add(
    MaxPooling2D(pool_size=(2, 2), padding='same'))
# Output shape after MaxPooling2D (pool=2x2, padding='same'):
# (ceil(420/2), ceil(540/2), 32) = (210, 270, 32)

c_autoencoder.add(
    Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same'))
# Output shape after Conv2D:
# (210, 270, 16)

c_autoencoder.add(
    MaxPooling2D(pool_size=(2, 2), padding='same'))
# Output shape after MaxPooling2D:
# (ceil(210/2), ceil(270/2), 16) = (105, 135, 16)


# Decoder
c_autoencoder.add(
    Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='relu', padding='same'))
# Output shape after Conv2DTranspose (stride=2, padding='same'):
# approximately doubles spatial dims
# (105*2, 135*2, 8) = (210, 270, 8)

c_autoencoder.add(
    Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='relu', padding='same'))
# Output shape:
# (210*2, 270*2, 16) = (420, 540, 16)


# Output layer
c_autoencoder.add(
    Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same'))
# Output shape after final Conv2D:
# (420, 540, 1) â€” matches original input size

c_autoencoder.summary()



# Compile and train the autoencoder

# Compile the model
c_autoencoder.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']  # Mean Absolute Error
)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='loss',           # watch the training loss
    patience=10,               # stop if no improvement for 10 epochs
    restore_best_weights=True  # revert to the best model
)

# Train the model
history = c_autoencoder.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)



import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

epoch_loss = history.history['loss']
epoch_val_loss = history.history['val_loss']


plt.figure(figsize=(18,6))

# Loss plot
plt.plot(epoch_loss, label='Train Loss', color='red')
plt.plot(epoch_val_loss, label='Validation Loss', color='black')
plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle=':', linewidth=0.8)

plt.tight_layout()
plt.show()



y_pred = c_autoencoder.predict(X_test, batch_size=16)



import seaborn as sns
sns.set(style='whitegrid')

plt.figure(figsize=(12, 18))

for i in range(0, 6, 2):
    # Noisy Image subplot
    plt.subplot(3, 2, i + 1)
    plt.imshow(X_test[i][:, :, 0], cmap='gray')
    plt.title("Noisy image", fontsize=12, fontweight='bold')
    plt.axis('off')

    # Denoised Image subplot
    plt.subplot(3, 2, i + 2)
    plt.imshow(y_pred[i][:, :, 0], cmap='gray')
    plt.title("Denoised by autoencoder", fontsize=12, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.show()





