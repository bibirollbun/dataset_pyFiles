import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
import os


# Define paths
train_csv = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')
test_csv = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/test.csv')


train_data, val_data = train_test_split(train_csv, test_size=0.2, random_state=42)

print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")



IMG_SIZE = (48, 48)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Create data generators for training and validation
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory='/kaggle/input/ai-vs-human-generated-dataset',
    x_col="file_name",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory='/kaggle/input/ai-vs-human-generated-dataset',
    x_col="file_name",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    
)

test_datagen = ImageDataGenerator(
    #preprocessing_function=ela_preprocessing  
)
# Create the test generator again after checks
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_csv,
    directory='/kaggle/input/ai-vs-human-generated-dataset',
    x_col='id',  
    y_col=None,  
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,
    class_mode=None, 
    shuffle=False 
)


# Load EfficientNetB3 model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
base_model.trainable = False  # Freeze base model


# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=out)


# Compile model
model.compile(optimizer=AdamW(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks

early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)





# Train model
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

model.fit(train_generator, validation_data=val_generator, epochs=5,steps_per_epoch=STEP_SIZE_TRAIN,validation_steps=STEP_SIZE_VAL)





model.save('/kaggle/working/model_efficientnetb3_run2.h5')


import numpy as np

# Make predictions using the test generator
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

# Convert predictions to class labels (assuming 2 classes)
predicted_labels = np.argmax(predictions, axis=1)  # Get the index of the highest probability

# Add predictions to the test DataFrame
test_csv['label'] = predicted_labels


# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': test_csv['id'],
    'label': test_csv['label'].astype(int)  # Convert predictions to integers
})


# # Save the submission file, remove rows with NaN labels
# submission_df = submission_df.dropna(subset=['label'])
submission_df.to_csv('submission_2_efficientnetb3.csv', index=False)
print("Submission file 'submission.csv' created successfully.")

