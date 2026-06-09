import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization 
from PIL import Image # Image editing
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Image Transformation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input ,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.utils import class_weight
import tensorflow as tf
import zipfile # Zip archives

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




# Train images are in the zip files, Extract the files
zip_path = "/kaggle/input/datasciencebowl/train.zip"
extract_path = "/kaggle/working/train"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("File classes: ",os.listdir(extract_path)[:5])


# Image show example from train file
extract_path = "/kaggle/working/train/train"

class_folder = os.listdir(extract_path)[0]
class_path = os.path.join(extract_path, class_folder)

img_file = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))][0]
img_path = os.path.join(class_path, img_file)

img = Image.open(img_path)
plt.imshow(img, cmap='gray')
plt.title(f"Class: {class_folder}")
plt.axis("off")
plt.show()



train_dir = "/kaggle/working/train/train"
num_classes = len(os.listdir("/kaggle/working/train/train"))
img_size = 80 # size of image
batch_size = 64 # loaded image numbers


# MobilNetV2 base model
base_model = MobileNetV2(
    input_shape=(img_size, img_size, 3), 
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False # freezing the model

inputs = Input(shape=(img_size, img_size, 3))  
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# Data generator and Normalization
datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2,
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode='rgb',  
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


# Callbacks

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)

checkpoint = ModelCheckpoint("best_mobilenetv2.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

callbacks = [early_stop, reduce_lr, checkpoint]

# Model fit
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)


# Visualization

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.legend()

plt.tight_layout()
plt.show()


# Test images are in the zip files, Extract the files

import zipfile
test_zip_path = "/kaggle/input/datasciencebowl/test.zip"
extract_test_path = "/kaggle/working/test"

with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_test_path)

test_dir = extract_test_path + "/test"


# Test ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

# Test generator
test_generator = test_datagen.flow_from_directory(
    directory=extract_test_path,
    target_size=(img_size, img_size),
    batch_size=64,
    class_mode=None,
    shuffle=False
)

# Predict
preds = model.predict(test_generator, verbose=1)
test_preds = np.argmax(preds, axis=1)

# image filenames
image_files = test_generator.filenames
image_files = [os.path.basename(f) for f in image_files]  

# Class indices
idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
pred_labels = [idx_to_label[i] for i in test_preds]

class_names = list(train_gen.class_indices.keys())

# Submission 
submission = pd.DataFrame(preds, columns=class_names)
submission.insert(0, "image", image_files)

submission.to_csv("submission.csv", index=False) # Score: 1.60754 Private score: 1.64601
submission.head()


img_size = 160
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    class_mode='categorical',
    subset='training',
    batch_size=batch_size,
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    batch_size=batch_size,
    shuffle=False
)


def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Convert grayscale to RGB
    x = tf.keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)
        
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False  # freeze
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)


model = build_model((img_size, img_size, 1), train_gen.num_classes)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


labels = train_gen.classes
cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(cw))


callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)
]


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights 
)


# Visualization

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.legend()

plt.tight_layout()
plt.show()


# Test ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

# Test generator
test_generator = test_datagen.flow_from_directory(
    directory=extract_test_path,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode=None,
    shuffle=False,
    color_mode='grayscale'
)

# Predict
preds = model.predict(test_generator, verbose=1)
test_preds = np.argmax(preds, axis=1)

# image filenames
image_files = test_generator.filenames
image_files = [os.path.basename(f) for f in image_files]  

# Class indices
idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
pred_labels = [idx_to_label[i] for i in test_preds]

class_names = list(train_gen.class_indices.keys())

# Submission 
submission = pd.DataFrame(preds, columns=class_names)
submission.insert(0, "image", image_files)

submission.to_csv("submission.csv", index=False) # Score: 5.54002 Private score: 5.53441
submission.head()

