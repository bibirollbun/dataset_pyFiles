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


!unzip -qq /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip
!unzip -qq /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import os, shutil, pathlib

import warnings
warnings.filterwarnings('ignore')


# Define the base directory
base_dir = pathlib.Path("/kaggle/working/")

# Define the source directory
original_dir = base_dir / "train"

# Define the destination directories
train_dir = base_dir / "train_organized" # Changed directory name for subset
validation_dir = base_dir / "validation_organized"
test_dir = base_dir / "test_organized"

# Create the main destination directories
for directory in [train_dir, validation_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

# Create subdirectories for cats and dogs within each destination directory
for directory in [train_dir, validation_dir, test_dir]:
    os.makedirs(directory / "cats", exist_ok=True)
    os.makedirs(directory / "dogs", exist_ok=True)
# Function to get category from filename
def get_category(filename):
    if "cat" in filename:
        return "cats"
    elif "dog" in filename:
        return "dogs"
    else:
        return None

# Get list of all image files
all_files = list(original_dir.glob("*.jpg")) # Assuming all images are jpg

# Separate cat and dog files
cat_files = [f for f in all_files if "cat" in f.name]
dog_files = [f for f in all_files if "dog" in f.name]

# Shuffle files to create random train/validation/test splits
np.random.shuffle(cat_files)
np.random.shuffle(dog_files)


# Count the number of cat and dog files
num_cats = len(cat_files)
num_dogs = len(dog_files)

# Create a bar plot to visualize class balance
labels = ['Cats', 'Dogs']
counts = [num_cats, num_dogs]

plt.bar(labels, counts, color=['blue', 'orange'])
plt.title('Class Balance: Cats vs Dogs')
plt.ylabel('Number of Images')
plt.xlabel('Classes')
plt.show()


# Define split sizes for each label in training
train_split = 5500
validation_split = 2000
test_split = 5000     

train_cat_files = cat_files[:train_split]
train_dog_files = dog_files[:train_split]

validation_cat_files = cat_files[train_split : train_split + validation_split]
validation_dog_files = dog_files[train_split : train_split + validation_split]

test_cat_files = cat_files[train_split + validation_split : train_split + validation_split + test_split]
test_dog_files = dog_files[train_split + validation_split : train_split + validation_split + test_split]


# Function to copy files to their respective directories
def copy_files(file_list, destination_dir):
    for f in file_list:
        category = get_category(f.name)
        if category:
            shutil.copy(f, destination_dir / category / f.name)

# Copy files to train, validation, and test directories
print("Copying training files (subset)...")
copy_files(train_cat_files + train_dog_files, train_dir)
print("Copying validation files...")
copy_files(validation_cat_files + validation_dog_files, validation_dir)
print("Copying test files...")
copy_files(test_cat_files + test_dog_files, test_dir)



train_dataset = image_dataset_from_directory(
    train_dir, # Use the new training directory
    image_size=(224, 224),
    batch_size=64)
validation_dataset = image_dataset_from_directory(
    validation_dir ,
    image_size=(224, 224),
    batch_size=64)
test_dataset = image_dataset_from_directory(
    test_dir ,
    image_size=(224, 224),
    batch_size=64)


# Take one batch from the training dataset
for images, labels in train_dataset.take(1):
    plt.figure(figsize=(12, 12))
    for i in range(20):
        ax = plt.subplot(5, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_dataset.class_names[labels[i]])
        plt.axis("off")
    plt.show()


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models



# Load base model (exclude top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers initially
base_model.trainable = False

# Build model with embedded normalization
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),  # Regularization
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.summary()  # Check architecture



model.compile(
    optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'] 
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('ResNet50-Based Model.h5', monitor='val_accuracy', save_best_only=True)
]


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,  
    callbacks=callbacks
)



 import matplotlib.pyplot as plt
 acc = history.history["accuracy"]
 val_acc = history.history["val_accuracy"]
 loss = history.history["loss"]
 val_loss = history.history["val_loss"]
 epochs = range(1, len(acc) + 1)
 plt.plot(epochs, acc, "bo", label="Training accuracy")
 plt.plot(epochs, val_acc, "b", label="Validation accuracy")
 plt.title("Training and validation accuracy")
 plt.legend()
 plt.figure()
 plt.plot(epochs, loss, "bo", label="Training loss")
 plt.plot(epochs, val_loss, "b", label="Validation loss")
 plt.title("Training and validation loss")
 plt.legend()
 plt.show()


 test_model = keras.models.load_model(
 "/kaggle/working/ResNet50-Based Model.h5")


from sklearn.metrics import classification_report, confusion_matrix

# Get predictions
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred_probs = test_model.predict(test_dataset)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


import matplotlib.pyplot as plt
import tensorflow as tf
import random

# Get the filenames from the test directory
test_images_dir = pathlib.Path("/kaggle/working/test")
test_image_files = list(test_images_dir.glob("*.jpg")) # Get all files


images_to_show = [random.randint(0, 5000) for _ in range(20)]
images_to_display = []
filenames_to_display = []
plt.figure(figsize=(12, 12))
for idx, i in enumerate(images_to_show):
    img_path = test_image_files[i]
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    images_to_display.append(img)
    filenames_to_display.append(img_path.name)

    # Make prediction for this single image
    prediction = test_model.predict(img_array)
    predicted_label = "Dog" if prediction[0][0] > 0.5 else "Cat"
    confidence = prediction[0][0] if predicted_label == "Dog" else 1 - prediction[0][0]
    plt.subplot(4, 5, idx + 1)
    plt.imshow(images_to_display[idx])
    plt.title(f"{predicted_label}: {confidence:.2f}")
    plt.axis("off")

plt.tight_layout()
plt.show()


test_images_dir = pathlib.Path("/kaggle/working/test")

test_data = image_dataset_from_directory(
    test_images_dir,
    image_size=(224, 224),
    batch_size=64,
    shuffle=False,
    labels=None
)


import os

# Get the file paths from the dataset
file_paths = test_data.file_paths

# Extract just the filenames
filenames = [os.path.basename(path) for path in file_paths]

# Now filenames contains the name of each image file
print(f"Found {len(filenames)} images")
print("First 5 filenames:", filenames[:5])


id = [int(filename.split('.')[0]) for filename in filenames]
print(f"Found {len(id)} images")
print("First 5 filenames:", id[:5])


pred = test_model.predict(test_data, steps = len(test_data), verbose = 1)


id_fd=pd.Series(id,name = "id")
id_lb=pd.DataFrame(pred, columns=['label'])
submission = pd.concat([id_fd,id_lb],axis = 1)
submission.to_csv("submission.csv",index=False)
srt_sub=submission.sort_values(by=['id'])
srt_sub.head(10)

