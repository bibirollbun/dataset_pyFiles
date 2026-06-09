import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Input, layers, models, optimizers
from keras import optimizers


# This is our folder with all of the data
data_path = "/kaggle/input/histopathologic-cancer-detection"
print("Files in dataset folder:", os.listdir(data_path), '\n')

# The 'train' and 'test' folders have images to be classified by the model
train_path = os.path.join(data_path, "train")
print("Sample images:", os.listdir(train_path)[:5], '\n')

# train_labels.csv provides the labels for training images (1=cancerous, 0=non-cancerous)
labels = pd.read_csv(os.path.join(data_path, "train_labels.csv"))[:5000]
print(labels.head())


random_index = np.random.randint(0, len(labels))

image_id = labels.iloc[random_index]['id']
label = labels.iloc[random_index]['label']

img_path = f'{data_path}/train/{image_id}.tif'
image = load_img(img_path)

plt.imshow(image)
plt.axis('off')
plt.title(f"Label: {label}")
plt.show()


num_image_files = len([f for f in os.listdir(f'{data_path}/train/') if f.endswith('.tif')])
print(f"Number of image files: {num_image_files}")

print(f"Labels dataset shape: {labels.shape}")  # Expecting 5,000 rows for the training set


first_image_file = [f for f in os.listdir(f'{data_path}/train/') if f.endswith('.tif')][0]

img_path = os.path.join(f'{data_path}/train/', first_image_file)
image = load_img(img_path)

image_array = img_to_array(image)
print(f"Image shape: {image_array.shape}")


labels['label'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Labels (Cancer Detection)')
plt.xlabel('Label (0 = No Cancer, 1 = Cancer)')
plt.ylabel('Number of Images')
plt.xticks([0, 1])
plt.tight_layout()
plt.show()


print(labels.describe())


labeled_ids = set(labels['id'])
image_ids_in_folder = set(f.replace('.tif', '') for f in os.listdir(f'{data_path}/train/') if f.endswith('.tif'))
unlabeled_images = image_ids_in_folder - labeled_ids
print(f"Number of image files without a label: {len(unlabeled_images)}")


random_index = np.random.randint(0, len(labels))
image_path = os.path.join(f'{data_path}/train/', os.listdir(f'{data_path}/train/')[random_index])
img = load_img(image_path)
img_array = img_to_array(img).astype(np.uint8)

red_channel = img_array[:, :, 0].flatten()
green_channel = img_array[:, :, 1].flatten()
blue_channel = img_array[:, :, 2].flatten()

# Plot histograms
plt.hist(red_channel, bins=256, color='red', alpha=0.5)
plt.hist(green_channel, bins=256, color='green', alpha=0.5)
plt.hist(blue_channel, bins=256, color='blue', alpha=0.5)
plt.title('Pixel Value Distribution from a Random Image')
plt.xlabel('Pixel Intensity (0-255)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


labels['label'] = labels['label'].astype(str)


labels['id'] = labels['id'].astype(str) + '.tif'  # Add .tif to the end of each image ID in the dataframe
train, val = train_test_split(labels, test_size=0.2, stratify=labels['label'], random_state=42)


# This is the generator that will provide batches of images to the model
# It also handles the normalization step with the rescale parameter
imggen = ImageDataGenerator(rescale=1./255)


# Generator connected to the training set
train_generator = imggen.flow_from_dataframe(
    train,
    directory=f'{data_path}/train/',
    x_col='id',
    y_col='label',
    batch_size=16,
    target_size=(96, 96),
    color_mode='rgb',
    class_mode='binary',
    shuffle=False
)

# Generator connected to the validation set
val_generator = imggen.flow_from_dataframe(
    val,
    directory=f'{data_path}/train/',
    x_col='id',
    y_col='label',
    batch_size=16,
    target_size=(96, 96),
    color_mode='rgb',
    class_mode='binary',
    shuffle=False
)


simple_ann = models.Sequential([
    Input(shape=(96, 96, 3)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

simple_ann.compile(
    optimizer=optimizers.RMSprop(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

simple_ann_start = time.time()
simple_ann_results = simple_ann.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=20,
    epochs=5,
)
simple_ann_end = time.time()


simple_cnn = models.Sequential([
    layers.Input(shape=(96, 96, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

simple_cnn.compile(
    optimizer=optimizers.RMSprop(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

simple_cnn.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=10,
    epochs=20,
)


simple_cnn = models.Sequential([
    layers.Input(shape=(96, 96, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

simple_cnn.compile(
    optimizer=optimizers.RMSprop(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

simple_cnn_start = time.time()
simple_cnn_results = simple_cnn.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=40,
    epochs=75,
)
simple_cnn_end = time.time()


plt.plot(simple_cnn_results.history['accuracy'], label='Training Accuracy')
plt.plot(simple_cnn_results.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


batch_cnn = models.Sequential([
    layers.Input(shape=(96, 96, 3)),
    
    layers.Conv2D(32, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(1, activation='sigmoid')
])

batch_cnn.compile(
    optimizer=optimizers.RMSprop(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

batch_cnn_results = batch_cnn.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=40,
    epochs=75,
)


plt.plot(batch_cnn_results.history['accuracy'], label='Training Accuracy')
plt.plot(batch_cnn_results.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


batch_cnn = models.Sequential([
    layers.Input(shape=(96, 96, 3)),
    
    layers.Conv2D(32, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    
    layers.Dense(1, activation='sigmoid')
])

batch_cnn.compile(
    optimizer=optimizers.RMSprop(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

batch_cnn_start = time.time()
batch_cnn_results = batch_cnn.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=40,
    epochs=20,
)
batch_cnn_end = time.time()


vgg = VGG16(
    weights='imagenet', 
    include_top=False, 
    input_shape=(96, 96, 3)
)
vgg.trainable = False

vgg = models.Sequential([
    vgg,
    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(1, activation='sigmoid')
])

vgg.compile(
    optimizer=optimizers.RMSprop(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

vgg_start = time.time()
vgg_history = vgg.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=40,
    validation_steps=20
)
vgg_end = time.time()


model_summary = pd.DataFrame()
model_summary['Model'] = ['Simple Artificial NN', 'Convolutional NN', 'Convolutional NN w/ Batch Norm.', 'VGG Pre-trained NN']
model_summary['Accuracy'] = [simple_ann_results.history['val_accuracy'][-1],
                             simple_cnn_results.history['val_accuracy'][-1],
                             batch_cnn_results.history['val_accuracy'][-1],
                             vgg_history.history['val_accuracy'][-1]]
model_summary['Training Time (sec)'] = [simple_ann_end - simple_ann_start,
                                        simple_cnn_end - simple_cnn_start,
                                        batch_cnn_end - batch_cnn_start,
                                        vgg_end - vgg_start]
model_summary['Training Batches'] = [5, 75, 20, 10]
model_summary['Training Time per Batch (sec)'] = model_summary['Training Time (sec)'] / model_summary['Training Batches']
model_summary


y_true = val_generator.classes
y_pred_prob = simple_cnn.predict(val_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))


sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Simple CNN Confusion Matrix')
plt.show()


# Get the names of the images in the test dataset
test_image_dir = f'{data_path}/test/'
test_df = pd.DataFrame({'id': os.listdir(test_image_dir)})


test_datagen = ImageDataGenerator(rescale=1./255)

# This generator will serve test images to the model to make predictions
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=f'{data_path}/test/',
    x_col='id',
    y_col=None,
    target_size=(96, 96),
    color_mode='rgb',
    batch_size=16,
    class_mode=None,
    shuffle=False
)

# Get predictions from test images
predictions = simple_cnn.predict(test_generator)


# Bring predictions together which 
predicted_classes = (predictions > 0.5).astype("int32")
test_df['id'] = test_df['id'].str[:-4]
test_df['label'] = predicted_classes
print(test_df.head())


# Output CSV for submission to competition
test_df.to_csv('submission.csv', index=False)

