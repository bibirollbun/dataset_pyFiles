# Import the main library for building and training neural networks.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import our visualization library.
import matplotlib.pyplot as plt

# Import utilities for file and directory management.
import os
import shutil
import zipfile
import random


# --- Step 1: Unzip the main training file ---
# The original data is in /kaggle/input/dogs-vs-cats/train.zip
print("Unzipping the training data...")
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('.') # Extract to the current working directory
print("Unzipping complete.")

# --- Step 2: Get all image filenames and shuffle them ---
all_filenames = os.listdir('train')
# We shuffle the filenames to ensure a random mix for our training and validation sets.
random.shuffle(all_filenames)

# --- Step 3: Define directories and create them ---
# We'll create a main 'data' directory to hold everything.
if os.path.exists('data'):
    shutil.rmtree('data') # Remove old directory if it exists

# Create the folder structure
train_cat_dir = 'data/train/cats'
train_dog_dir = 'data/train/dogs'
val_cat_dir = 'data/validation/cats'
val_dog_dir = 'data/validation/dogs'

os.makedirs(train_cat_dir, exist_ok=True)
os.makedirs(train_dog_dir, exist_ok=True)
os.makedirs(val_cat_dir, exist_ok=True)
os.makedirs(val_dog_dir, exist_ok=True)

# --- Step 4: Split filenames and copy files ---
# We'll use 10,000 images per class for training, and the rest for validation.
TRAIN_SIZE = 10000
VAL_SIZE = 2500 # 12500 total images per class

train_cat_count, train_dog_count = 0, 0
val_cat_count, val_dog_count = 0, 0

for filename in all_filenames:
    source_path = os.path.join('train', filename)
    
    # Check if the file is a cat image
    if filename.startswith('cat'):
        if train_cat_count < TRAIN_SIZE:
            shutil.copy(source_path, os.path.join(train_cat_dir, filename))
            train_cat_count += 1
        elif val_cat_count < VAL_SIZE:
            shutil.copy(source_path, os.path.join(val_cat_dir, filename))
            val_cat_count += 1
            
    # Check if the file is a dog image
    elif filename.startswith('dog'):
        if train_dog_count < TRAIN_SIZE:
            shutil.copy(source_path, os.path.join(train_dog_dir, filename))
            train_dog_count += 1
        elif val_dog_count < VAL_SIZE:
            shutil.copy(source_path, os.path.join(val_dog_dir, filename))
            val_dog_count += 1

print("\nData organization complete.")
print(f"Total training cat images: {train_cat_count}")
print(f"Total training dog images: {train_dog_count}")
print(f"Total validation cat images: {val_cat_count}")
print(f"Total validation dog images: {val_dog_count}")


# Define some key parameters for our project.
IMAGE_SIZE = (128, 128) # We'll make all images a standard 128x128 size.
BATCH_SIZE = 32         # We'll show the child 32 pictures at a time.
DATA_DIR = 'data'      # The path to our newly created data directory.

# Create the training and validation datasets from our organized folders.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# Let's see the class names it found from the folder names.
class_names = train_dataset.class_names
print(f"The classes are: {class_names}")

# Let's visualize a few images to confirm everything loaded correctly.
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


# Let's create our "magic photocopier" as a sequence of layers.
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)


# We build our model as a sequence of layers.
model = keras.Sequential([
    # --- Input and Preprocessing ---
    layers.Input(shape=IMAGE_SIZE + (3,)),
    layers.Rescaling(1./255),
    data_augmentation,
    
    # --- The "Seeing" Part of the Brain ---
    # First Conv/Pool Block: Learn 32 simple filters.
    layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    
    # Second Conv/Pool Block: Learn 64 more complex filters.
    layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    
    # Third Conv/Pool Block: Learn 128 even more complex filters.
    layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    
    # --- The "Reasoning" Part of the Brain ---
    layers.Flatten(),
    # A Dense layer to learn complex combinations of clues.
    layers.Dense(512, activation="relu"),
    # The final output neuron. We use 'sigmoid' because this is a yes/no (binary) question.
    # It will output a single number between 0 (cat) and 1 (dog).
    layers.Dense(1, activation="sigmoid"),
])

model.summary()


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# Let's start the training!
history = model.fit(
    train_dataset,
    epochs=5,   #increase or decrease epoch as per your liking. Beaware! increasing epoch doesn't necessarily means increase in accuracy. Experiment Yourself...
    validation_data=validation_dataset,
)


# Get the accuracy and loss values from the training history.
acc = history.history['accuracy']
val_acc = history.history['validation_accuracy']
loss = history.history['loss']
val_loss = history.history['validation_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Let's use an image from our validation set to test a prediction.
for val_images, val_labels in validation_dataset.take(1):
    first_image = val_images[0]
    true_label = val_labels[0]
    
    plt.imshow(first_image.numpy().astype("uint8"))
    plt.title(f"True Label: {class_names[true_label]}")
    plt.axis("off")
    plt.show()

    img_array = tf.expand_dims(first_image, 0)
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    confidence = 100 * (1 - score) if score < 0.5 else 100 * score
    predicted_class = "Cat" if score < 0.5 else "Dog"
    
    print(f"This image is a {predicted_class} with {confidence:.2f}% confidence.")

