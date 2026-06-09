# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense, Multiply, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Define directories
directory  = '../input/histopathologic-cancer-detection'  
labels_path = os.path.join(directory, 'train_labels.csv')
train_path = os.path.join(directory, 'train')
test_path = os.path.join(directory, 'test')


print('There are ',len(os.listdir(train_path)), 'training images and ',len(os.listdir(test_path)), 'testing images.')


# load the CSV file containing labels
labels = pd.read_csv(labels_path)

# create a DataFrame to match training data images with labels
train = pd.DataFrame({'path': glob(os.path.join(train_path, '*.tif'))}) 
train['id'] = train['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])  # Extract image ID to merge with training labels
train = train.merge(labels, on='id')  

# convert labels to strings
train['label'] = train['label'].astype(str)

# ensure paths are relative to train_path
train['path'] = train['path'].apply(lambda x: os.path.basename(x))

print(train.head())


# check for null values
print('There are ',train.isnull().sum().sum(), ' null values')
# check for duplicates
print('There are ', train.duplicated().sum(),' duplicated images')


# randomly select one image path
image_path = random.choice(train['path'].values)
# full path to the image
full_image_path = os.path.join(train_path, image_path)
# Load the selected image 
selected_image = cv2.imread(full_image_path)

print('Image shape is ', selected_image.shape)
print('The maximum number of pixels is ', selected_image.max())


# Separate cancerous and non-cancerous data
cancerous = train[train['label'] == '1']['id'].sample(n=5).values  # Treat labels as strings
noncancerous = train[train['label'] == '0']['id'].sample(n=5).values

cancerous_images = []
for id in cancerous:
    image_path = os.path.join(train_path, id + '.tif')
    cancerous_images.append(Image.open(image_path))

noncancerous_images = []
for id in noncancerous:
    image_path = os.path.join(train_path, id + '.tif')
    noncancerous_images.append(Image.open(image_path))

plt.figure(figsize=(10, 5))
for i, image in enumerate(cancerous_images):
    plt.subplot(1, 5, i+1)
    plt.imshow(image)
    plt.title('Cancerous')
    plt.axis('off')
plt.show()

plt.figure(figsize=(10, 5))
for i, image in enumerate(noncancerous_images):
    plt.subplot(1, 5, i+1)
    plt.imshow(image)
    plt.title('Non-Cancerous')
    plt.axis('off')
plt.show()


# Counts for each class
label_counts = labels['label'].value_counts()

# Calculate the percentages of each class
positive_percentage = label_counts[1] / (label_counts[0] + label_counts[1])
print(f'Positive labels in training data: {positive_percentage:.2%}')

# Bar plot 
plt.figure(figsize=(7, 5))
ax = label_counts.sort_index().plot(kind='bar', color=['blue', 'red'])

plt.xticks([0, 1], labels=[f"Cancer Negative N={label_counts[0]}", f"Cancer Positive N={label_counts[1]}"], rotation=0)

for i, count in enumerate(label_counts):
    percentage = count / label_counts.sum() * 100
    ax.text(i, count + 500, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=12)

plt.title(f'Distribution of Training Data')
plt.ylabel('Count')
plt.xlabel(" ")
plt.tight_layout()

# Save plot to kaggle output directory
#plt.savefig('/kaggle/working/training_data_distribution.png', dpi=300)

plt.show()


def gaussian_smooth(image):
    # Convert image to unit8
    image = (image * 255).astype(np.uint8)

    #Convert image to RGB
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Gaussian smoothing (blur)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Convert float 32 and range of [0,1]
    return image.astype(np.float32) / 255.0



# Split the data into 80% training and 20% validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Data generator for training images
train_datagen = ImageDataGenerator(
    rescale=1./255  # scale images from 0-255 to 0-1
    #rotation_range=20,          # Degree range for random rotations
    #width_shift_range=0.1,      # Fraction of total width for random horizontal shifts
    #height_shift_range=0.1,     # Fraction of total height for random vertical shifts
    #shear_range=0.1,
    #zoom_range=0.1,
    #horizontal_flip=True, 
    #preprocessing_function=gaussian_smooth
    
)

# Data generator for validation images without augmentation
validation_datagen = ImageDataGenerator(
    rescale=1./255  # scale images from 0-255 to 0-1
)

batch_size = 64

# Generator for the training set
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_path,
    x_col='path',
    y_col='label',
    target_size=(96, 96), # image shape
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

# Generator for the validation set
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=val_df,  
    directory=train_path,
    x_col='path',
    y_col='label',
    target_size=(96, 96),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)


def spatial_attention_block(input_tensor, kernel_size=7):
    # Generates attention features through a separate pathway
    attention = Conv2D(filters=1, kernel_size=kernel_size, padding='same', activation='sigmoid')(input_tensor)
    #attention = BatchNormalization()(attention)

    # Apply attention to input
    refined_features = Multiply()([input_tensor, attention])

    return refined_features


reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.5, # 50% Reduction in learning Rate
    patience = 2, # If no improvement after 2 epochs
    verbose = 1,
    min_lr = 1e-6 # minimum learning rate not to go below
)


# Define input
inputs = Input(shape=(96, 96, 3))

x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(inputs)
# Spatial attention layer
# Using kernel_size=9 then kernel_size=5 to capture both global and local spatial dependencies
x = spatial_attention_block(x, kernel_size=9)
x = spatial_attention_block(x, kernel_size=5)
x = Dropout(0.05)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.15)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)
# 2nd spatial attention layer, allows the model to re-focus after deep feature extraction
x = spatial_attention_block(x, kernel_size=9)
x = spatial_attention_block(x, kernel_size=5)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Global Average Pooling to reduce spatial dimensions
x = GlobalAveragePooling2D()(x)

# Fully connected layers
x = Dense(256)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Output layer
outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))(x)

# Build model
model_spatial = Model(inputs, outputs)


model_spatial.summary()


model_spatial.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'auc']
)


#steps_per_epoch = len(train_df)//batch_size
#validation_steps = len(val_df)//batch_size

hist_spatial = model_spatial.fit(
    train_generator,
    #steps_per_epoch = steps_per_epoch, 
    validation_data = validation_generator,
    #validation_steps = validation_steps,
    epochs = 20,
    callbacks=[reduce_lr]
)


plt.figure(figsize=(10, 5))

# accuracy plot
plt.subplot(1, 3, 1)
plt.plot(hist_spatial.history['accuracy'], label='Train Accuracy')
plt.plot(hist_spatial.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.700, 1.000)
plt.title('Model Accuracy')
plt.legend(loc='lower right')

# loss plot
plt.subplot(1, 3, 2)
plt.plot(hist_spatial.history['loss'], label='Train Loss')
plt.plot(hist_spatial.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend(loc='upper right')

# AUC plot
plt.subplot(1, 3, 3)
plt.plot(hist_spatial.history['auc'], label='Train AUC')
plt.plot(hist_spatial.history['val_auc'], label='Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('ROC AUC')
plt.legend(loc='lower right')

plt.tight_layout()
# Save plot to kaggle output directory (optional)
plt.savefig('/kaggle/working/cnn_training_results.png', dpi=300)
plt.show()


# calculations for confusion matrix
true_labels = validation_generator.classes
pred_probabilities = model_spatial.predict(validation_generator, steps=len(validation_generator), verbose=1)

# for binary classification get predicted classes based on probability threshold of 0.5
pred_classes = (pred_probabilities > 0.5).astype(int)

# confusion matrix
cm = confusion_matrix(true_labels, pred_classes)

# plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Cancerous', 'Cancerous'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')

# Save plot to kaggle output directory (optional)
plt.savefig('/kaggle/working/cnn_training_confusion.png', dpi=300)
plt.show()


# Load test image paths
test_df = pd.DataFrame({'path': glob(os.path.join(test_path, '*.tif'))})
test_df['id'] = test_df['path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
test_df['filename'] = test_df['id'] + '.tif'
print(test_df.head(5))
print(test_df.info())
print(test_df['path'][0])


#Create the test image datagenerator
test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                    directory=test_path,
                                                    x_col='filename',
                                                    y_col=None,
                                                    target_size=(96, 96),
                                                    batch_size=32,
                                                    class_mode=None,
                                                    shuffle=False)


# Make predictions with test data
test_preds = model_spatial.predict(test_generator, verbose=1)
test_preds_probs = (test_preds > 0.5).astype(int)
submission = test_df.copy()
submission = submission.drop(columns=['filename', 'path'])
submission['label'] = test_preds_probs
print(submission.head())

submission.to_csv('/kaggle/working/submission.csv', index=False)

