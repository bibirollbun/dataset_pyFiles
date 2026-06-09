# Import libraries
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Unzipping train.zip
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/train")

train_path = "/kaggle/working/train/train"


# Organize data into separate folders for ImageDataGenerator
import shutil

# Create new directory structure
base_dir = "/kaggle/working/dataset"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(base_dir + '/train/cats', exist_ok=True)
os.makedirs(base_dir + '/train/dogs', exist_ok=True)

# Split files into cats and dogs folders
for file in os.listdir(train_path):
    if "cat" in file:
        shutil.move(os.path.join(train_path, file), base_dir + '/train/cats/' + file)
    else:
        shutil.move(os.path.join(train_path, file), base_dir + '/train/dogs/' + file)


# Prepare the data generators

IMG_SIZE = 224
BATCH_SIZE = 32

# Data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    base_dir + '/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    base_dir + '/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)


# Build ResNet50 model

# Load pre-trained ResNet50 without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze base model

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# Fine-tuning (Optional)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3
)


# Evaluate model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")


# Plotting training results
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

