import os
import numpy as np
import pandas as pd
import shutil
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryCrossentropy
import gc
from tensorflow.keras.backend import clear_session

def clear_memory():
    clear_session()
    gc.collect()



data_base_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
train_zip_path = os.path.join(data_base_dir, "train.zip")
test_zip_path = os.path.join(data_base_dir, "test.zip")

train_dir = "/kaggle/working/train"
test_dir = "/kaggle/working/test"

if not os.path.exists(train_dir):
    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall("/kaggle/working")

if not os.path.exists(test_dir):
    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        zip_ref.extractall("/kaggle/working")

# Check the decompressed contents
#print("Train directory contents:", os.listdir(train_dir))
gc.collect()


# Create cats and dogs subfolders
cats_dir = os.path.join(train_dir, 'cats')
dogs_dir = os.path.join(train_dir, 'dogs')

os.makedirs(cats_dir, exist_ok=True)
os.makedirs(dogs_dir, exist_ok=True)

for filename in os.listdir(train_dir):
    file_path = os.path.join(train_dir, filename)
    if filename.endswith('.jpg'): 
        if 'cat' in filename.lower():
            shutil.move(file_path, os.path.join(cats_dir, filename))
        elif 'dog' in filename.lower():
            shutil.move(file_path, os.path.join(dogs_dir, filename))

# Ensure folder structure
#print("Cats directory contents:", os.listdir(cats_dir))
#print("Dogs directory contents:", os.listdir(dogs_dir))
gc.collect()



batch_size = 16
img_size = (224, 224)

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)



_input = Input((224, 224, 3))  # The input is an RGB image

# Convolution block
conv1 = Conv2D(64, (3, 3), padding="same", activation="relu")(_input)
conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv1)
pool1 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool1)
conv4 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv3)
pool2 = MaxPooling2D((2, 2))(conv4)

conv5 = Conv2D(256, (3, 3), padding="same", activation="relu")(pool2)
conv6 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv5)
pool3 = MaxPooling2D((2, 2))(conv6)

conv7 = Conv2D(512, (3, 3), padding="same", activation="relu")(pool3)
conv8 = Conv2D(512, (3, 3), padding="same", activation="relu")(conv7)
pool4 = MaxPooling2D((2, 2))(conv8)

conv9 = Conv2D(512, (3, 3), padding="same", activation="relu")(pool4)
conv10 = Conv2D(512, (3, 3), padding="same", activation="relu")(conv9)
pool5 = MaxPooling2D((2, 2))(conv10)

# Fully connected layer
flat = Flatten()(pool5)
dense1 = Dense(4096, activation="relu")(flat)
dropout1 = Dropout(0.5)(dense1)
dense2 = Dense(4096, activation="relu")(dropout1)
dropout2 = Dropout(0.5)(dense2)

# Output layer (single node for binary classification)
output = Dense(1, activation="sigmoid")(dropout2)

vgg16_model = Model(inputs=_input, outputs=output)
vgg16_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])



vgg16_model.summary()


def log_loss(y_true, y_pred):
    y_pred = tf.squeeze(y_pred, axis=-1)  
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

vgg16_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', log_loss]
)


checkpoint = ModelCheckpoint("vgg16_model.keras", monitor="val_loss", save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=5)



epochs = 10

history = vgg16_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)
clear_memory()


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

plot_history(history)



def prepare_test_images(test_dir, img_size=(224, 224)):
    filenames = os.listdir(test_dir)
    images = []
    for filename in filenames:
        img_path = os.path.join(test_dir, filename)
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) 
        img = img / 255.0  
        images.append(img)
    return filenames, np.vstack(images)

test_filenames, test_images = prepare_test_images(test_dir, img_size=(224, 224))




predictions = vgg16_model.predict(test_images, batch_size=batch_size)


submission_df = pd.DataFrame({
    'id': [filename.split('.')[0] for filename in test_filenames],  
    'label': predictions.flatten()                                 
})

submission_df.to_csv('submission.csv', index=False)


submission_df.head()

