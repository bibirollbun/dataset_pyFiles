import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(2)
import random
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras import layers, models, regularizers
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.regularizers import l1_l2 
from keras.layers import *
import pandas as pd
from keras.optimizers import Adam
from keras.initializers import GlorotUniform 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Added ReduceLROnPlateau
from keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG16, ResNet50, ResNet101
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.applications import EfficientNetB4
import os
import itertools


# Define paths
directory = '/kaggle/input/ai-vs-human-generated-dataset'
train_data = '/kaggle/input/ai-vs-human-generated-dataset/train_data'
train_csv = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'
test_data = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2'
test_csv = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'


# Define image size and ELA quality
image_size = (48, 48)  # Desired image size
# ela_quality = 90  # ELA quality parameter


# Load the training CSV file
train_df = pd.read_csv(train_csv)
test_df= pd.read_csv(test_csv)


# Remove the 'train_data/' prefix from the file_name column
train_df['file_name'] = train_df['file_name'].str.replace('train_data/', '', regex=False)
# Remove the 'train_data/' prefix from the file_name column
test_df['id'] = test_df['id'].str.replace('test_data_v2/', '', regex=False)


# def convert_to_ela_image(image, quality):
#     """Convert an image to ELA (Error Level Analysis) format."""
#     # Ensure the image is in uint8 format and has the correct shape
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    
#     # Convert the NumPy array to a PIL Image
#     original_image = Image.fromarray(image)
    
#     # Save the original image to a BytesIO object with a specific quality
#     temp_buffer = io.BytesIO()
#     original_image.save(temp_buffer, 'JPEG', quality=quality)
#     temp_buffer.seek(0)
    
#     # Open the compressed image from the BytesIO object
#     temp_image = Image.open(temp_buffer)
    
#     # Compute the ELA image
#     ela_image = Image.fromarray(np.abs(np.array(original_image) - np.array(temp_image)))
#     return ela_image


# # Function to preprocess an image (convert to ELA, resize, normalize)
# def ela_preprocessing(image):
#     """Preprocess an image by converting it to ELA, resizing, and normalizing."""
#     # Convert the image to ELA format
#     ela_image = convert_to_ela_image(image, ela_quality)
    
#     # Resize the image
#     resized_image = ela_image.resize(image_size)
    
#     # Normalize the image to [0, 1]
#     return np.array(resized_image) / 255.0


# Split the dataset into two categories
category_0 = train_df[train_df['label'] == 0]  # Images with label 0
category_1 = train_df[train_df['label'] == 1]  # Images with label 1


print(len(category_0))
print(len(category_1))


# Randomly sample 20,000 images from each category
num_samples = 39975
category_0_sampled = category_0.sample(n=num_samples, random_state=42)  # Sample 20,000 from category 0
category_1_sampled = category_1.sample(n=num_samples, random_state=42)  # Sample 20,000 from category 1

# Combine the sampled data into a balanced DataFrame
balanced_df = pd.concat([category_0_sampled, category_1_sampled])

# Shuffle the balanced DataFrame
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)


# Split the balanced dataset into training and validation sets
train_data_df, val_data_df = train_test_split(balanced_df, test_size=0.2, random_state=42)


# Convert the 'label' column to strings
train_data_df['label'] = train_data_df['label'].astype(str)
val_data_df['label'] = val_data_df['label'].astype(str)


# Trim whitespace from filenames
test_df['id'] = test_df['id'].str.strip()


epochs = 50
batch_size = 128


# Define data augmentation for training
train_datagen = ImageDataGenerator(
    #preprocessing_function=ela_preprocessing,  
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define validation and test generators (without augmentation)
val_datagen = ImageDataGenerator(
    #preprocessing_function=ela_preprocessing  
)

test_datagen = ImageDataGenerator(
    #preprocessing_function=ela_preprocessing  
)

# Create generators for training
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data_df,
    directory=train_data,
    x_col='file_name',
    y_col='label',
    target_size=image_size,  
    batch_size=batch_size,
    class_mode='categorical',  
    shuffle=True  
)

# Create generators for validation
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data_df,
    directory=train_data,
    x_col='file_name',
    y_col='label',
    target_size=image_size, 
    batch_size=batch_size,
    class_mode='categorical',  
    shuffle=False  
)

# Create the test generator again after checks
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_data,
    x_col='id',  
    y_col=None,  
    target_size=image_size, 
    batch_size=batch_size,
    class_mode=None, 
    shuffle=False 
)


# Print generator details
print("Training generator:", train_generator)
print("Validation generator:", val_generator)
print("Test generator:", test_generator)


# def build_InceptionV3_model():
#     base_model = InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3))  # Changed to ResNet101

#     # Add new layers on top of the model
#     x = base_model.output  # Output of the base model
#     x = GaussianNoise(0.1)(x)  # Add Gaussian noise with a standard deviation of 0.1
#     x = GlobalAveragePooling2D()(x)  # Global average pooling layer
#     x = Dense(1024, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l1_l2(l1=0.02, l2=0.04))(x)  # Dense layer with ReLU activation and L2 regularization
#     x = Dropout(0.4)(x)  # Dropout layer for regularization
#     predictions = Dense(2, activation='softmax')(x)  # Output layer with softmax activation for multi-class classification

#     # Define the model
#     model = Model(inputs=base_model.input, outputs=predictions)  # Combined model

#     # Freeze base layers
#     for layer in base_model.layers:
#         layer.trainable = True  # Freeze base layers for training

#     return model


def build_Sequential_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32,(3,3), activation='relu', input_shape = (48,48,3)),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv2D(64,(3,3), activation='relu'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv2D(128,(3,3), activation='relu'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
        
        keras.layers.Conv2D(256,(3,3), activation='relu'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Dropout(0.2),
            
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(2, activation='softmax')  
    ])
    return model


lr_schedule = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.1,  # Reduce learning rate by a factor of 0.1
    patience=3,  # Wait for 3 epochs before reducing the learning rate
    min_lr=1e-6  # Minimum learning rate
)


# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop training if no improvement for 5 epochs
    restore_best_weights=True
)


import keras
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Set the learning rate
learning_rate = 1e-4

model = build_Sequential_model()
model.summary()
# Compile the model with the specified learning rate
model.compile(optimizer=Adam(learning_rate=learning_rate), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


# Train the model using the generators
hist = model.fit(
    train_generator,  # Use the train_generator created with flow_from_dataframe
    steps_per_epoch=len(train_data_df) // batch_size,  # Number of batches per epoch
    epochs=epochs,
    validation_data=val_generator,  # Use the val_generator created with flow_from_dataframe
    validation_steps=len(val_data_df) // batch_size,  # Number of validation batches
    callbacks=[early_stopping, lr_schedule],  # Callbacks for early stopping and learning rate scheduling
)


model.save('model_Sequential_run1.h5')


import numpy as np

# Make predictions using the test generator
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

# Convert predictions to class labels (assuming 2 classes)
predicted_labels = np.argmax(predictions, axis=1)  # Get the index of the highest probability

# Add predictions to the test DataFrame
test_df['label'] = predicted_labels


# # Make predictions using the test generator
# predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

# # Convert predictions to binary labels
# predicted_labels = [1 if pred > 0.5 else 0 for pred in predictions]

# # Add predictions to the test DataFrame
# test_df['label'] = predicted_labels


# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': test_df['label'].astype(int)  # Convert predictions to integers
})


# # Save the submission file, remove rows with NaN labels
# submission_df = submission_df.dropna(subset=['label'])
submission_df.to_csv('submission_sh_Sequential.csv', index=False)
print("Submission file 'submission.csv' created successfully.")

