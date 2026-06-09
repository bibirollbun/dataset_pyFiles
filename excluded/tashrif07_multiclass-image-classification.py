import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os

IMAGE_DIR = '/kaggle/input/dog-breed-identification/train'
CSV_PATH = '/kaggle/input/dog-breed-identification/labels.csv'


labels_df = pd.read_csv(CSV_PATH)


# Prepare the DataFrame: Append '.jpg' to the image IDs
# This is crucial for flow_from_dataframe to match the filenames in the directory
labels_df['id'] = labels_df['id'] + '.jpg'

# Define Constants
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = labels_df['breed'].nunique() # Automatically determine the number of classes

print(f"Total images loaded: {len(labels_df)}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Image directory: {IMAGE_DIR}")
print("-" * 50)


# Initialize the ImageDataGenerator with augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1.0 / 255,      # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.3    # Reserve 30% for validation
)

# Training data generator
train_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=IMAGE_DIR,    
    x_col='id',
    y_col='breed',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
val_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=IMAGE_DIR,    # Using your defined image path
    x_col='id',
    y_col='breed',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"Training batches: {len(train_generator)}")
print(f"Validation batches: {len(val_generator)}")
print("-" * 50)


# Simple Sequential CNN Model
model = Sequential([
    # Input Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Block 4 (More complex)
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    
    # Classifier Head
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax') # Output layer
])

# Model Compilation
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()


# Model Training
print("Starting Model Training...")
history = model.fit(
    train_generator,
    epochs=20, # Set your desired number of training epochs
    validation_data=val_generator,
    verbose=1
)

print("Training complete.")


import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- PATHS (Using the paths you previously provided) ---
TEST_DIR = '/kaggle/input/dog-breed-identification/test' 
SAMPLE_SUBMISSION_PATH = '/kaggle/input/dog-breed-identification/sample_submission.csv'
IMG_SIZE = 224
BATCH_SIZE = 16

# 1. PREPARE TEST DATA LIST
# Load the sample submission file to get the image IDs in the required order.
submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

# Create a DataFrame of filenames (ID + .jpg) for the generator
test_df = pd.DataFrame({'id': submission_df['id'] + '.jpg'})


# 2. CREATE TEST GENERATOR (No shuffle, no augmentation, just scaling)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TEST_DIR,
    x_col='id',
    y_col=None,            # IMPORTANT: No labels for the test set
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,       # IMPORTANT: Set to None for prediction/inference
    shuffle=False          # IMPORTANT: Must be False to maintain file order
)

# 3. GENERATE PREDICTIONS
print("Generating predictions...")
probabilities = model.predict(test_generator, verbose=1)


# 4. CREATE FINAL SUBMISSION FILE
# Get the class names in the exact order the model was trained on
class_names = list(train_generator.class_indices.keys())

# Create a new DataFrame with the prediction probabilities and correct column headers
submission_output = pd.DataFrame(probabilities, columns=class_names)

# Re-insert the original 'id' column from the sample submission file (without the .jpg)
submission_output.insert(0, 'id', submission_df['id'])

# Save the final CSV file
submission_file_name = 'submission.csv'
submission_output.to_csv(submission_file_name, index=False)

print(f"First 5 rows:\n{submission_output.head()}")

