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


# ==============================================================================
# 1. SETUP AND IMPORTS
# ==============================================================================
# Standard library imports
import os
import zipfile
import time

# Third-party library imports
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Scikit-learn for data splitting
from sklearn.model_selection import train_test_split

# TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("TensorFlow Version:", tf.__version__)

# ==============================================================================
# 2. CONFIGURATION PARAMETERS
# ==============================================================================
# --- Data Paths ---
# Assumes the zip files are in the same directory as the script.
TRAIN_ZIP_PATH = 'train.zip'
TEST_ZIP_PATH = 'test.zip'
LABELS_CSV_PATH = '/kaggle/input/dog-breed-identification/labels.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/dog-breed-identification/sample_submission.csv'

# --- Directories for Extracted Data ---
BASE_DIR = '/kaggle/input/dog-breed-identification/'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# --- Model & Training Hyperparameters ---
# The Xception model was trained on 299x299 images.
IMG_WIDTH, IMG_HEIGHT = 299, 299
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32  # Adjust based on your GPU memory
EPOCHS_HEAD_TRAINING = 10  # Epochs for training the new classification layers
EPOCHS_FINE_TUNING = 20  # Epochs for fine-tuning the full model
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
VALIDATION_SPLIT = 0.2 # 20% of training data will be used for validation
RANDOM_STATE = 42 # For reproducible splits

# ==============================================================================
# 3. DATA PREPARATION
# ==============================================================================
def prepare_data():
    """
    Extracts data from zip files, creates directories, and prepares the
    pandas DataFrame with file paths and labels.
    """
    print("--- Starting Data Preparation ---")
    start_time = time.time()

    # Create base directory if it doesn't exist
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print(f"Created directory: {BASE_DIR}")

    # Unzip training data
    if not os.path.exists(TRAIN_DIR):
        print(f"Extracting {TRAIN_ZIP_PATH}...")
        with zipfile.ZipFile(TRAIN_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
    else:
        print(f"Training data already extracted at {TRAIN_DIR}")

    # Unzip test data
    if not os.path.exists(TEST_DIR):
        print(f"Extracting {TEST_ZIP_PATH}...")
        with zipfile.ZipFile(TEST_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
    else:
        print(f"Test data already extracted at {TEST_DIR}")
        
    # Load labels
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    print(f"\nLoaded {LABELS_CSV_PATH} with {len(labels_df)} entries.")
    
    # Add image extension and full path to the dataframe
    labels_df['id'] = labels_df['id'] + '.jpg'
    labels_df['filepath'] = labels_df['id'].apply(lambda x: os.path.join(TRAIN_DIR, x))
    
    # Get the number of classes
    num_classes = labels_df['breed'].nunique()
    print(f"Number of dog breeds (classes): {num_classes}")
    
    # Check for missing files
    labels_df['file_exists'] = labels_df['filepath'].apply(os.path.exists)
    if not labels_df['file_exists'].all():
        print("Warning: Some image files listed in labels.csv are missing!")
        labels_df = labels_df[labels_df['file_exists']]

    # Split data into training and validation sets
    train_df, val_df = train_test_split(
        labels_df,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=labels_df['breed'] # Ensures balanced class distribution
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    print(f"--- Data Preparation Finished in {time.time() - start_time:.2f}s ---\n")
    return train_df, val_df, num_classes

# ==============================================================================
# 4. DATA GENERATORS
# ==============================================================================
def create_data_generators(train_df, val_df):
    """
    Creates Keras ImageDataGenerators for training and validation.
    Applies data augmentation to the training generator.
    """
    print("--- Creating Data Generators ---")
    
    # Training generator with data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.xception.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation generator (only rescaling, no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.xception.preprocess_input
    )
    
    # Create generators from dataframes
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='breed',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='breed',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # No need to shuffle validation data
    )
    
    print("--- Data Generators Created ---\n")
    return train_generator, validation_generator

# ==============================================================================
# 5. MODEL BUILDING
# ==============================================================================
def build_model(num_classes):
    """
    Builds the classification model using transfer learning with Xception.
    """
    print("--- Building Model ---")
    
    # Load the base Xception model, pre-trained on ImageNet
    base_model = Xception(
        weights='imagenet',       # Load weights pre-trained on ImageNet
        include_top=False,        # Exclude the final classification layer
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )
    
    # Freeze the base model layers to prevent them from being updated
    # during the initial training phase.
    base_model.trainable = False
    
    # Create the new model head
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Convert features to a single vector per image
    x = Dense(1024, activation='relu')(x) # A fully-connected layer
    x = Dropout(0.5)(x) # Dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x) # The final output layer
    
    # Combine the base model and the new head
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print("--- Model Built Successfully ---\n")
    return model, base_model

# ==============================================================================
# 6. MODEL TRAINING & FINE-TUNING
# ==============================================================================
def train_model(model, base_model, train_gen, val_gen, num_classes):
    """
    Trains the model in two phases:
    1. Trains only the new head with the base model frozen.
    2. Fine-tunes the top layers of the base model with a low learning rate.
    """
    # --- Phase 1: Train the Head ---
    print("--- Phase 1: Training the custom head ---")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_HEAD),
        loss='categorical_crossentropy', # Correct for multi-class classification
        metrics=['accuracy']
    )
    
    # Callbacks for Phase 1
    # Save the best model based on validation loss
    checkpoint_head = ModelCheckpoint('best_model_head.h5', monitor='val_loss', save_best_only=True, mode='min')
    # Stop training if validation loss doesn't improve for 3 epochs
    early_stop_head = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history_head = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD_TRAINING,
        validation_data=val_gen,
        callbacks=[checkpoint_head, early_stop_head]
    )

    print("\n--- Phase 2: Fine-tuning the model ---")
    # Unfreeze the base model to allow fine-tuning
    base_model.trainable = True
    
    # We'll fine-tune from the top. A common practice is to keep the
    # early layers (like batch normalization) frozen.
    # For Xception, let's unfreeze the top ~30% of layers.
    fine_tune_at = len(base_model.layers) - 40 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Re-compile the model with a very low learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks for Phase 2
    checkpoint_fine_tune = ModelCheckpoint('best_model_fine_tuned.h5', monitor='val_loss', save_best_only=True, mode='min')
    early_stop_fine_tune = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history_fine_tune = model.fit(
        train_gen,
        epochs=EPOCHS_FINE_TUNING,
        validation_data=val_gen,
        callbacks=[checkpoint_fine_tune, early_stop_fine_tune]
    )

    print("--- Model Training Complete ---\n")
    # The best fine-tuned model is automatically restored by EarlyStopping
    return model

# ==============================================================================
# 7. PREDICTION AND SUBMISSION
# ==============================================================================
def create_submission(model, train_generator):
    """
    Generates predictions on the test set and creates the submission file.
    """
    print("--- Generating Predictions for Submission ---")
    
    # Get test file paths
    test_files = os.listdir(TEST_DIR)
    test_filepaths = [os.path.join(TEST_DIR, fname) for fname in test_files]
    test_df = pd.DataFrame({'filepath': test_filepaths})
    
    # Create a test generator
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.xception.preprocess_input
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col=None, # No labels for test data
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False # IMPORTANT: Do not shuffle test data
    )
    
    # Make predictions
    predictions = model.predict(test_generator, verbose=1)
    
    # Get the class labels in the correct order
    class_indices = train_generator.class_indices
    # Invert the dictionary to map index to label
    labels = dict((v, k) for k, v in class_indices.items())
    # Sort by index to get an ordered list of breed names
    breed_columns = [labels[i] for i in range(len(labels))]

    # Create the submission DataFrame
    submission_df = pd.DataFrame(predictions, columns=breed_columns)
    
    # Get the image IDs from the filenames
    test_ids = [os.path.basename(f).split('.')[0] for f in test_generator.filenames]
    submission_df.insert(0, 'id', test_ids)
    
    # Ensure the columns match the sample submission format
    sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission_df = submission_df[sample_df.columns]

    # Save the submission file
    submission_df.to_csv('submission.csv', index=False)
    print("\n--- Submission file 'submission.csv' created successfully! ---")


# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # Step 1: Prepare the data (unzip, load labels, split)
    train_df, val_df, num_classes = prepare_data()
    
    # Step 2: Create data generators
    train_gen, val_gen = create_data_generators(train_df, val_df)
    
    # Step 3: Build the model
    model, base_model = build_model(num_classes)
    
    # Step 4: Train the model (head training + fine-tuning)
    trained_model = train_model(model, base_model, train_gen, val_gen, num_classes)
    
    # Step 5: Create the submission file
    # Note: We can also load the best saved model from disk if needed:
    # from tensorflow.keras.models import load_model
    # best_model = load_model('best_model_fine_tuned.h5')
    # create_submission(best_model, train_gen)
    create_submission(trained_model, train_gen)




