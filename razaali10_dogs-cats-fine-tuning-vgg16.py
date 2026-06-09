import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator



with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")



print(os.listdir("train")) #just for checking


filenames = os.listdir("train")
labels = []
for filename in filenames:
    if "cat" in filename:
        labels.append('cat')
    elif "dog" in filename:
        labels.append('dog')
    
df = pd.DataFrame({
    'Filename': filenames,
    'Label': labels
})

df.head()


df.info()


# Function to display images
def display_images(image_paths, category):
    plt.figure(figsize=(24, 4))
    for i, img_path in enumerate(image_paths):
        plt.subplot(1, 3, i + 1)
        img = mpimg.imread(os.path.join('train', img_path))
        plt.imshow(img)
        plt.title(category)
        plt.axis('off')
    plt.show()

# Selecting three random images of cats
cat_images = [filename for filename in filenames if 'cat' in filename]
selected_cat_images = random.sample(cat_images, 3)

# Selecting three random images of dogs
dog_images = [filename for filename in filenames if 'dog' in filename]
selected_dog_images = random.sample(dog_images, 3)

# Display the images
display_images(selected_cat_images, 'Cat')
display_images(selected_dog_images, 'Dog')



# Constants
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
BATCH_SIZE = 32


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='train/',
    x_col='Filename',
    y_col='Label',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='train/',
    x_col='Filename',
    y_col='Label',
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)


# Load the VGG16 model, pretrained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False


# Create the model
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


model.summary()


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=5
)


# Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')


#Unfreeze the last 10 layers
for layer in base_model.layers[-10:]:
    layer.trainable = True


model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Continue training
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=5  # You can adjust the number of epochs
)


# Evaluate the model again
plt.plot(history_fine.history['accuracy'], label='accuracy')
plt.plot(history_fine.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.92, 1])
plt.legend(loc='lower right')


with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as test1_zip:
    test1_zip.extractall('.')  


test_dir = "../working/test1/"
filenames = os.listdir(test_dir)
test_data = pd.DataFrame({"Filename": filenames})
test_data.head()


test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(
    dataframe=test_data,
    directory='test1/',
    x_col='Filename',
    y_col=None,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode=None,
)


test1_predict = model.predict(test_generator)


test1_predict


threshold = 0.5
test_data['Label'] = np.where(test1_predict > threshold, 'dog', 'cat')


test_data.tail()


sample_test = test_data.sample(n=6)
plt.figure(figsize=(24, 8)) 

subplot_index = 1
for _, row in sample_test.iterrows():
    image_path = os.path.join(test_dir, row['Filename'])
    img = mpimg.imread(image_path)
    plt.subplot(2, 3, subplot_index)
    plt.imshow(img)
    plt.title(row['Label'])
    plt.axis('off')
    subplot_index += 1

plt.show()




