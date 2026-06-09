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


import os
print(os.listdir("../input"))

import zipfile

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")


from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
folder = 'train/'
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'dog.' + str(i) + '.jpg'
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()


import os
import shutil
import zipfile
from matplotlib import pyplot
from matplotlib.image import imread


# Fix: Correct dataset path
original_dataset_dir = './train/'

# Create directories
base_dir = '/kaggle/working/dogs-vs-cats'
os.makedirs(base_dir, exist_ok=True)

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

for directory in [train_dir, validation_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

for directory in [train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir, test_cats_dir, test_dogs_dir]:
    os.makedirs(directory, exist_ok=True)

# Function to copy images safely
def copy_images(file_list, src_folder, dest_folder):
    for fname in file_list:
        src = os.path.join(src_folder, fname)
        dst = os.path.join(dest_folder, fname)
        if os.path.exists(src):  # Fix: Avoid file not found errors
            shutil.copyfile(src, dst)

# Splitting dataset
copy_images(['cat.{}.jpg'.format(i) for i in range(1000)], original_dataset_dir, train_cats_dir)
copy_images(['cat.{}.jpg'.format(i) for i in range(1000, 1500)], original_dataset_dir, validation_cats_dir)
copy_images(['cat.{}.jpg'.format(i) for i in range(1500, 2000)], original_dataset_dir, test_cats_dir)

copy_images(['dog.{}.jpg'.format(i) for i in range(1000)], original_dataset_dir, train_dogs_dir)
copy_images(['dog.{}.jpg'.format(i) for i in range(1000, 1500)], original_dataset_dir, validation_dogs_dir)
copy_images(['dog.{}.jpg'.format(i) for i in range(1500, 2000)], original_dataset_dir, test_dogs_dir)


# Sanity check: Counting images in each dataset split
print('Total training cat images:', len(os.listdir(train_cats_dir)))
print('Total training dog images:', len(os.listdir(train_dogs_dir)))
print('Total validation cat images:', len(os.listdir(validation_cats_dir)))
print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('Total test cat images:', len(os.listdir(test_cats_dir)))
print('Total test dog images:', len(os.listdir(test_dogs_dir)))



from keras import layers, models, optimizers

# Define the CNN model
model = models.Sequential()

# First Conv-Pool Layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Second Conv-Pool Layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Conv-Pool Layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fourth Conv-Pool Layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flattening & Fully Connected Layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),  # Updated API
    metrics=['accuracy']
)

# Print model summary
model.summary()



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation for Training (Rescaling only)
train_datagen = ImageDataGenerator(rescale=1.0/255)

# Rescaling Validation Data
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Training Data Generator
train_generator = train_datagen.flow_from_directory(
    train_dir,               # Directory path for training images
    target_size=(150, 150),  # Resize images to match CNN input
    batch_size=20,           # Batch size
    class_mode='binary'      # Binary classification (Cats vs. Dogs)
)

# Validation Data Generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,          # Directory path for validation images
    target_size=(150, 150),  # Resize images
    batch_size=20,           # Batch size
    class_mode='binary'      # Binary classification
)



history = model.fit(
    train_generator,          # Training Data Generator
    steps_per_epoch=100,      # Number of batches per epoch
    epochs=30,                # Total epochs
    validation_data=validation_generator,  # Validation Data Generator
    validation_steps=50       # Number of batches per validation step
)



model.save('cats_and_dogs_small_1.h5')


import matplotlib.pyplot as plt

# Extract training history
acc = history.history['accuracy']  # ✅ Corrected key
val_acc = history.history['val_accuracy']  # ✅ Corrected key
loss = history.history['loss']
val_loss = history.history['val_loss']

# Find the correct number of epochs to avoid mismatch
num_epochs = min(len(acc), len(val_acc))  # ✅ Use the shorter length
epochs = range(1, num_epochs + 1)

# Plot Training & Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, acc[:num_epochs], 'bo', label='Training Accuracy')  # ✅ No mismatch
plt.plot(epochs, val_acc[:num_epochs], 'b', label='Validation Accuracy')  # ✅ No mismatch
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Training & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss[:num_epochs], 'bo', label='Training Loss')  # ✅ No mismatch
plt.plot(epochs, val_loss[:num_epochs], 'b', label='Validation Loss')  # ✅ No mismatch
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()



datagen = ImageDataGenerator(
    rotation_range=40,              # Random rotation between -40 and +40 degrees
    width_shift_range=0.2,          # Random horizontal shift
    height_shift_range=0.2,         # Random vertical shift
    shear_range=0.2,                # Random shear
    zoom_range=0.2,                 # Random zoom
    horizontal_flip=True,           # Randomly flip images horizontally
    fill_mode='nearest'             # Use nearest pixel value for filling empty pixels
)


import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming 'train_cats_dir' is defined and contains the images
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[3]  # Get a specific image file path (3rd image in the directory)

# Load the image and resize it to (150, 150)
img = image.load_img(img_path, target_size=(150, 150))

# Convert image to array
x = image.img_to_array(img)

# Reshape image to add a batch dimension
x = x.reshape((1,) + x.shape)

# Define the data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create augmented images and display them
i = 0
for batch in datagen.flow(x, batch_size=1):
    # Plot the augmented image
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))  # Convert array back to image and display
    i += 1
    
    # Stop after generating and displaying 4 augmented images
    if i % 4 == 0:
        break

# Show the generated images
plt.show()



from keras import models, layers, optimizers

# Define the model architecture
model = models.Sequential()

# Add the first convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Add the second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Add the third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Add the fourth convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the 3D feature maps to 1D vector
model.add(layers.Flatten())

# Add dropout layer to avoid overfitting
model.add(layers.Dropout(0.5))

# Add a fully connected layer
model.add(layers.Dense(512, activation='relu'))

# Output layer with sigmoid activation for binary classification
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and RMSprop optimizer
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-4),  # Updated syntax for learning rate
    metrics=['accuracy']  # Use 'accuracy' instead of 'acc'
)

# Model summary to check architecture
model.summary()



# from keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Data preprocessing for validation images (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Directory where training images are located
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,  # Number of images per batch
    class_mode='binary',  # Binary classification (cats vs dogs)
)

# Validation data generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,  # Directory where validation images are located
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,  # Number of images per batch
    class_mode='binary',  # Binary classification (cats vs dogs)
)

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Number of batches per epoch
    epochs=100,  # Number of epochs to train
    validation_data=validation_generator,
    validation_steps=50,  # Number of validation batches per epoch
)



model.save('cats_and_dogs_small_1.h5')


import matplotlib.pyplot as plt

# Assuming 'history' is the variable storing your model's training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Plotting training and validation accuracy
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training and validation loss
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Display the plots
plt.show()



 from keras.applications import VGG16
 conv_base = VGG16(weights='imagenet',
 include_top=False,
 input_shape=(150, 150, 3))


import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Define base directory paths for datasets
base_dir = '/kaggle/working/dogs-vs-cats'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Load the pre-trained VGG16 model, excluding the top classification layers
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Define the ImageDataGenerator for rescaling images
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# Extract features and labels for training, validation, and test datasets
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)



from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# Define the model
model = models.Sequential()
model.add(layers.Flatten(input_shape=(4, 4, 512)))  # Flatten the input
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))



import matplotlib.pyplot as plt

# Assuming 'history' is the variable storing your model's training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Plotting training and validation accuracy
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training and validation loss
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Display the plots
plt.show()



from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


 model.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers

# Define the ImageDataGenerator for training and validation datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up the data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)



print(len(acc), len(val_acc), len(loss), len(val_loss))


# Ensure epochs array matches the shortest metric array
epochs = range(1, min(len(acc), len(val_acc)) + 1)

# Plotting training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc[:len(epochs)], 'bo', label='Training accuracy')  # Truncate acc
plt.plot(epochs, val_acc[:len(epochs)], 'b', label='Validation accuracy')  # Truncate val_acc
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss[:len(epochs)], 'ro', label='Training loss')  # Truncate loss
plt.plot(epochs, val_loss[:len(epochs)], 'r', label='Validation loss')  # Truncate val_loss
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


from keras.models import load_model
model = load_model('/kaggle/input/dogs-vs-cats/cats_and_dogs_small_1.h5')
model.summary() 

