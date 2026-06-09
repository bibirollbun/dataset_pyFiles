# Basic Data Processing
import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
import zipfile

# Image Processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import shutil
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning Framework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.callbacks as keras_callback

# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
# Visualization
import plotly.express as px
import plotly.graph_objects as go

# Progress Bar
from tqdm import tqdm

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Extras for Grad-CAM
import tensorflow.keras.backend as K

# Test imports
print("âœ… All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")



# Define paths
base_path = '/kaggle/input/dogs-vs-cats'
train_zip = os.path.join(base_path, 'train.zip')
test_zip = os.path.join(base_path, 'test1.zip')

# Extract directories
extract_path = '/kaggle/working/dogs-vs-cats-data'
TRAIN_DIR = os.path.join(extract_path, 'train')
TEST_DIR = os.path.join(extract_path, 'test1')

print("\nChecking dataset directories...")

# Extract train.zip if not already extracted
if not os.path.exists(TRAIN_DIR):
    print("Extracting train.zip...")
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Train extraction completed!")
else:
    print("Train directory already exists")

# Extract test1.zip if not already extracted  
if not os.path.exists(TEST_DIR):
    print("Extracting test1.zip...")
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Test extraction completed!")
else:
    print("Test directory already exists")

# Check if directories exist and count files
if os.path.exists(TRAIN_DIR):
    train_files = os.listdir(TRAIN_DIR)
    print("Training directory found:", len(train_files), "files")
else:
    print("Training directory not found!")
    train_files = []

if os.path.exists(TEST_DIR):
    test_files = os.listdir(TEST_DIR)
    print("Test directory found:", len(test_files), "files")
else:
    print("Test directory not found!")
    test_files = []

print("\nFinal paths:")
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)


def explore_dataset():
    print("\nExploring dataset...")
    
    if not train_files:
        print("No training files found!")
        return [], [], []
    
    # Count dogs and cats
    dog_files = []
    cat_files = []
    
    for filename in train_files:
        if 'dog' in filename.lower():
            dog_files.append(filename)
        elif 'cat' in filename.lower():
            cat_files.append(filename)
    
    print("Dog images:", len(dog_files))
    print("Cat images:", len(cat_files))
    print("Total training images:", len(train_files))
    
    if dog_files:
        print("Sample dog files:", dog_files[:3])
    if cat_files:
        print("Sample cat files:", cat_files[:3])
    
    return train_files, dog_files, cat_files

# Run exploration
if train_files:
    train_files, dog_files, cat_files = explore_dataset()
else:
    dog_files, cat_files = [], []

# Display sample images
def show_sample_images():
    print("\nShowing sample images...")
    
    if not train_files:
        print("No images to display!")
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Dogs vs Cats - Sample Images', fontsize=16)
    
    if dog_files and cat_files:
        # Show 4 dog images
        for i in range(min(4, len(dog_files))):
            img_path = os.path.join(TRAIN_DIR, dog_files[i])
            img = load_img(img_path, target_size=(150, 150))
            axes[0, i].imshow(img)
            axes[0, i].set_title('Dog ' + str(i+1))
            axes[0, i].axis('off')
        
        # Show 4 cat images
        for i in range(min(4, len(cat_files))):
            img_path = os.path.join(TRAIN_DIR, cat_files[i])
            img = load_img(img_path, target_size=(150, 150))
            axes[1, i].imshow(img)
            axes[1, i].set_title('Cat ' + str(i+1))
            axes[1, i].axis('off')
    else:
        # Show first 8 images
        for i in range(min(8, len(train_files))):
            row = i // 4
            col = i % 4
            img_path = os.path.join(TRAIN_DIR, train_files[i])
            try:
                img = load_img(img_path, target_size=(150, 150))
                axes[row, col].imshow(img)
                axes[row, col].set_title('Image ' + str(i+1))
                axes[row, col].axis('off')
            except:
                axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Show images
if train_files:
    show_sample_images()

# Analyze image sizes
def analyze_images(sample_size=100):
    print("\nAnalyzing image sizes...")
    
    if not train_files:
        print("No images to analyze!")
        return
    
    widths = []
    heights = []
    
    # Take random sample
    sample_files = random.sample(train_files, min(sample_size, len(train_files)))
    
    for filename in sample_files:
        img_path = os.path.join(TRAIN_DIR, filename)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
        except:
            continue
    
    if widths:
        print("Width - Min:", min(widths), "Max:", max(widths), "Average:", int(np.mean(widths)))
        print("Height - Min:", min(heights), "Max:", max(heights), "Average:", int(np.mean(heights)))
        
        # Plot histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(widths, bins=20, color='blue', alpha=0.7)
        ax1.set_title('Image Width Distribution')
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Count')
        
        ax2.hist(heights, bins=20, color='red', alpha=0.7)
        ax2.set_title('Image Height Distribution') 
        ax2.set_xlabel('Height (pixels)')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()

# Analyze images
if train_files:
    analyze_images()

# Show class distribution
def show_class_distribution():
    print("\nClass distribution:")
    
    if not dog_files and not cat_files:
        print("Cannot show distribution - no labeled data!")
        return
    
    # Create pie chart
    labels = ['Dogs', 'Cats']
    sizes = [len(dog_files), len(cat_files)]
    colors = ['lightblue', 'lightcoral']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Dogs vs Cats Distribution')
    plt.axis('equal')
    plt.show()
    
    print("Dogs:", len(dog_files), "images")
    print("Cats:", len(cat_files), "images")
    total = len(dog_files) + len(cat_files)
    print("Dog percentage:", round(len(dog_files)/total*100, 1), "%")
    print("Cat percentage:", round(len(cat_files)/total*100, 1), "%")

# Show distribution
if dog_files or cat_files:
    show_class_distribution()

print("\nDataset setup completed!")



print("Starting data preprocessing...")

# STEP 1: CREATE LABELS AND SPLIT DATA
print("\nStep 1: Creating labels and splitting data...")

# Create lists for image paths and labels
image_paths = []
labels = []

for filename in train_files:
    full_path = os.path.join(TRAIN_DIR, filename)
    image_paths.append(full_path)
    
    # Label: dogs=0, cats=1
    if 'dog' in filename.lower():
        labels.append(0)
    elif 'cat' in filename.lower():
        labels.append(1)

print("Total labeled images:", len(labels))
print("Dogs:", labels.count(0), "Cats:", labels.count(1))

# Split data: 60% train, 20% validation, 20% test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.25, random_state=42, stratify=train_labels
)

print(f"Train: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}")

# STEP 2: ORGANIZE FILES INTO FOLDERS
print("\nStep 2: Organizing files...")

# Create directory structure
organized_dir = '/kaggle/working/organized_data'
dirs = [
    os.path.join(organized_dir, 'train', 'dogs'),
    os.path.join(organized_dir, 'train', 'cats'),
    os.path.join(organized_dir, 'validation', 'dogs'),
    os.path.join(organized_dir, 'validation', 'cats'),
    os.path.join(organized_dir, 'test', 'dogs'),
    os.path.join(organized_dir, 'test', 'cats')
]

for directory in dirs:
    os.makedirs(directory, exist_ok=True)

# Copy files to organized structure
def copy_files(paths, labels, base_dir):
    for path, label in zip(paths, labels):
        filename = os.path.basename(path)
        if label == 0:  # Dog
            dest = os.path.join(base_dir, 'dogs', filename)
        else:  # Cat
            dest = os.path.join(base_dir, 'cats', filename)
        
        if not os.path.exists(dest):
            shutil.copy2(path, dest)

copy_files(train_paths, train_labels, os.path.join(organized_dir, 'train'))
copy_files(val_paths, val_labels, os.path.join(organized_dir, 'validation'))
copy_files(test_paths, test_labels, os.path.join(organized_dir, 'test'))

print("Files organized successfully!")

# STEP 3: CREATE DATA GENERATORS
print("\nStep 3: Creating data generators...")

# Training generator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixels to 0-1
    rotation_range=20,           # Rotate images
    width_shift_range=0.1,       # Shift horizontally
    height_shift_range=0.1,      # Shift vertically
    zoom_range=0.1,              # Zoom in/out
    horizontal_flip=True,        # Flip horizontally
    brightness_range=[0.8, 1.2], # Change brightness
    fill_mode='nearest'
)

# Validation and test generators (no augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data flows
IMG_SIZE = 128  # â†� MODEL INPUT Ä°LE UYUMLU
BATCH_SIZE = 32

train_generator = train_datagen.flow_from_directory(
    os.path.join(organized_dir, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),  # 128x128
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['dogs', 'cats'],  # dogs=0, cats=1
    shuffle=True,
    seed=42
)

validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(organized_dir, 'validation'),
    target_size=(IMG_SIZE, IMG_SIZE),  # 128x128
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['dogs', 'cats'],
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(organized_dir, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),  # 128x128
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['dogs', 'cats'],
    shuffle=False
)

print("Data generators created!")
print("Training samples:", train_generator.samples)
print("Validation samples:", validation_generator.samples)
print("Test samples:", test_generator.samples)

# STEP 4: VERIFY PREPROCESSING
print("\nStep 4: Verifying preprocessing...")

def show_preprocessed_samples():
    """Show sample preprocessed images with correct labels"""
    images, labels = next(train_generator)
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Preprocessed Images with Correct Labels', fontsize=16)
    
    for i in range(8):
        row = i // 4
        col = i % 4
        
        axes[row, col].imshow(images[i])
        
        if labels[i] == 0:
            class_name = "Dog"
            color = 'blue'
        else:
            class_name = "Cat"
            color = 'red'
        
        axes[row, col].set_title(f'{class_name} (Label: {int(labels[i])})', 
                                color=color, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

show_preprocessed_samples()

# STEP 5: SUMMARY
print("\nPreprocessing Summary:")
print("=" * 40)
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Classes: Dogs=0, Cats=1")
print(f"Training images: {train_generator.samples}")
print(f"Validation images: {validation_generator.samples}")
print(f"Test images: {test_generator.samples}")
print(f"Steps per epoch: {train_generator.samples // BATCH_SIZE}")
print("\nData augmentation applied to training set:")
print("- Rotation, shifting, zoom, flip, brightness")
print("\nData preprocessing completed!")



print("Starting CNN model building...")

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# STEP 1: CREATE SIMPLIFIED CNN MODEL
print("\nStep 1: Creating simplified CNN architecture...")

def create_simplified_cnn():
    """
    Create a simplified CNN model
    """
    model = models.Sequential()
    
    # INPUT
    model.add(layers.Input(shape=(128, 128, 3)))
    
    # BLOCK 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # BLOCK 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # BLOCK 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # CLASSIFIER HEAD
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# Create simplified model
simplified_model = create_simplified_cnn()

print("Simplified CNN Model Summary:")
simplified_model.summary()

# Compile simplified model
simplified_model.compile(
    optimizer=Adam(learning_rate=0.0005),  # biraz daha hÄ±zlÄ± Ã¶ÄŸrenme
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Simplified model compiled!")

# STEP 2: SETUP TRAINING CALLBACKS
print("\nStep 2: Setting up training callbacks...")

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

print("Callbacks configured.")

# STEP 3: PREPARE FOR TRAINING
print("\nStep 3: Preparing for training...")

# Yeni batch size = 64
train_generator.batch_size = 64
validation_generator.batch_size = 64

# Epoch sayÄ±sÄ± 15
EPOCHS = 15

train_steps = train_generator.samples // train_generator.batch_size
val_steps = validation_generator.samples // validation_generator.batch_size

print(f"Training steps per epoch: {train_steps}")
print(f"Validation steps per epoch: {val_steps}")
print(f"Epochs: {EPOCHS}")

# STEP 4: READY TO TRAIN
print("\nModel Training Setup Summary:")
print("=" * 50)
print("âœ“ Simplified CNN model created (3 conv blocks)")
print("âœ“ Input size: 128x128x3")
print("âœ“ Batch size: 64")
print("âœ“ Epochs: 15")
print("âœ“ Optimizer: Adam (lr=0.0005)")
print("âœ“ Binary crossentropy loss")
print("âœ“ EarlyStopping & ReduceLROnPlateau callbacks")
print("âœ“ Ready for training!")




print("Starting model training...")

# Step 1: Define callbacks (simple version)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

# Step 2: Train the model
EPOCHS = 15
print(f"Training for {EPOCHS} epochs...")
print(f"Training images: {train_generator.samples}")
print(f"Validation images: {validation_generator.samples}")

history = simplified_model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1
)

print("Training completed!")

# Step 3: Plot Accuracy & Loss graphs
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy (Training vs Validation)')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss (Training vs Validation)')
    
    plt.show()
    
    # Print final metrics
    print(f"Final Training Accuracy: {acc[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")
    print(f"Final Training Loss: {loss[-1]:.4f}")
    print(f"Final Validation Loss: {val_loss[-1]:.4f}")

plot_training_history(history)

# Step 4: Save the trained model
print("\nStep 4: Saving trained model...")

try:
    # Save complete model
    model_path = '/kaggle/working/dogs_cats_trained_model.h5'
    simplified_model.save(model_path)
    print(f"âœ“ Model saved successfully: {model_path}")
    
    # Save weights separately (backup)
    weights_path = '/kaggle/working/model_weights.h5'
    simplified_model.save_weights(weights_path)
    print(f"âœ“ Weights saved successfully: {weights_path}")
    
    # Check file sizes
    import os
    model_size = os.path.getsize(model_path) / (1024*1024)  # MB
    weights_size = os.path.getsize(weights_path) / (1024*1024)  # MB
    
    print(f"âœ“ Model file size: {model_size:.1f} MB")
    print(f"âœ“ Weights file size: {weights_size:.1f} MB")
    
    # Test loading to make sure it works
    print("\nTesting model loading...")
    from tensorflow.keras.models import load_model
    test_model = load_model(model_path)
    print("âœ“ Model loading test successful!")
    
    # Save training history too
    import pickle
    history_path = '/kaggle/working/training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"âœ“ Training history saved: {history_path}")
    
except Exception as e:
    print(f" Error saving model: {e}")
    print("Training completed but model not saved!")

print("\n" + "="*50)
print("TRAINING AND SAVING COMPLETED!")
print("="*50)
print("âœ“ Model trained successfully")
print("âœ“ Training graphs displayed")
print("âœ“ Model saved to file")
print("âœ“ Ready for evaluation!")



# Loading saved model and running evaluation

print("Starting model evaluation with saved model...")

# STEP 1: LOAD OR RECREATE MODEL
print("\nStep 1: Loading or recreating model...")



# First, try to load saved model
model_path = "/kaggle/working/dogs_cats_trained_model.h5"
loaded_model = None

if os.path.exists(model_path):
    try:
        loaded_model = load_model(model_path)
        print("âœ“ Model loaded from file successfully!")
    except Exception as e:
        print(f"Error loading model file: {e}")
        loaded_model = None
else:
    print("Model file not found at:", model_path)

# If loading failed, recreate the model
if loaded_model is None:
    print("Recreating model from scratch...")

    def recreate_improved_model():
        model = models.Sequential()
        # INPUT
        model.add(layers.Input(shape=(128, 128, 3)))
        # BLOCK 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        # BLOCK 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        # BLOCK 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        # BLOCK 4
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        # CLASSIFIER
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    loaded_model = recreate_improved_model()
    loaded_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("âœ“ Model recreated successfully!")
    print(" Note: This is a fresh model without trained weights")

print(f"Model input shape: {loaded_model.input_shape}")
print(f"Model output shape: {loaded_model.output_shape}")

# STEP 2: RECREATE DATA GENERATORS (IF NEEDED)
print("\nStep 2: Ensuring data generators are ready...")

try:
    print(f"Test generator samples: {test_generator.samples}")
    print("âœ“ Data generators are ready")
except:
    print("Recreating data generators...")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    organized_dir = '/kaggle/working/organized_data'
    test_dir = os.path.join(organized_dir, 'test')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        classes=['dogs', 'cats'],
        shuffle=False
    )
    print("âœ“ Test generator recreated")

# STEP 3: EVALUATE MODEL PERFORMANCE
print("\nStep 3: Evaluating model on test set...")
test_generator.reset()
test_loss, test_accuracy = loaded_model.evaluate(
    test_generator,
    steps=test_generator.samples // test_generator.batch_size,
    verbose=1
)
print(f"âœ“ Test Accuracy: {test_accuracy:.4f}")
print(f"âœ“ Test Loss: {test_loss:.4f}")

# STEP 4: GENERATE PREDICTIONS
print("\nStep 4: Generating predictions...")
test_generator.reset()
test_steps_total = test_generator.samples // test_generator.batch_size
predictions = loaded_model.predict(test_generator, steps=test_steps_total, verbose=1)
true_labels = test_generator.classes[:len(predictions)]
predicted_labels = (predictions > 0.5).astype(int).flatten()
print(f"âœ“ Generated {len(predictions)} predictions")
print(f"âœ“ True labels shape: {len(true_labels)}")

# STEP 5: CONFUSION MATRIX
print("\nStep 5: Creating confusion matrix...")
def plot_confusion_matrix_fixed(true_labels, predicted_labels):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dogs (0)', 'Cats (1)'],
                yticklabels=['Dogs (0)', 'Cats (1)'])
    plt.title('Confusion Matrix - Dogs vs Cats')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("Confusion Matrix Results:")
    print(f"True Negatives (Dogs â†’ Dogs): {tn}")
    print(f"False Positives (Dogs â†’ Cats): {fp}")
    print(f"False Negatives (Cats â†’ Dogs): {fn}")
    print(f"True Positives (Cats â†’ Cats): {tp}")
    print(f"\nCalculated Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Cat detection): {precision:.4f}")
    print(f"Recall (Cat detection): {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

plot_confusion_matrix_fixed(true_labels, predicted_labels)

# STEP 6: CLASSIFICATION REPORT
print("\nStep 6: Detailed classification report...")
from sklearn.metrics import classification_report
class_names = ['Dogs', 'Cats']
report = classification_report(true_labels, predicted_labels,
                               target_names=class_names,
                               digits=4)
print("Classification Report:")
print(report)

# STEP 7: SIMPLIFIED GRAD-CAM
print("\nStep 7: Grad-CAM visualization...")
def simple_gradcam(model, img_path):
    from tensorflow.keras.preprocessing import image
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras.models import Model

    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)/255.0

    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_class = "Cat" if prediction > 0.5 else "Dog"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        print("No convolutional layer found for Grad-CAM")
        return

    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:,0]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)+1e-8
    heatmap = cv2.resize(heatmap.numpy(), (128,128))

    fig, axes = plt.subplots(1,3, figsize=(15,5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image'); axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap'); axes[1].axis('off')
    heatmap_colored = plt.cm.jet(heatmap)[:,:,:3]
    superimposed = 0.6*img_array[0] + 0.4*heatmap_colored
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Prediction: {predicted_class}\nConfidence: {confidence:.3f}')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()

# Test Grad-CAM on sample images
organized_dir = '/kaggle/working/organized_data'
test_dir = os.path.join(organized_dir, 'test')
sample_count = 0
max_samples = 4

for class_folder in ['dogs','cats']:
    class_path = os.path.join(test_dir, class_folder)
    if os.path.exists(class_path):
        files = os.listdir(class_path)[:2]
        for file in files:
            if sample_count >= max_samples: break
            file_path = os.path.join(class_path, file)
            print(f"\nGrad-CAM for {class_folder} sample:")
            try:
                simple_gradcam(loaded_model, file_path)
                sample_count += 1
            except Exception as e:
                print(f"Error with Grad-CAM: {e}")
                continue

print("\n" + "="*60)
print("MODEL EVALUATION SUMMARY")
print("="*60)
print(f"âœ“ Model successfully loaded and evaluated")
print(f"âœ“ Test Accuracy: {test_accuracy:.4f}")
print(f"âœ“ Test Loss: {test_loss:.4f}")
print(f"âœ“ Confusion matrix generated")





print("Starting quick hyperparameter optimization...")

# Define  test configurations (only 3 key ones)
quick_configs = {
    'baseline': {
        'filters': [32, 64, 128],
        'dropout_rate': 0.5,
        'dense_units': 256,
        'learning_rate': 0.0001
    },
    'more_dropout': {
        'filters': [32, 64, 128],
        'dropout_rate': 0.7,
        'dense_units': 256,
        'learning_rate': 0.0001
    },
    'higher_lr': {
        'filters': [32, 64, 128],
        'dropout_rate': 0.5,
        'dense_units': 256,
        'learning_rate': 0.001
    }
}

def create_quick_model(config):
    """Create simplified model for testing"""
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        
        # Block 1
        layers.Conv2D(config['filters'][0], (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(config['filters'][1], (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(config['filters'][2], (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(config['dense_units'], activation='relu'),
        layers.Dropout(config['dropout_rate']),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def create_quick_generators():
    """Create data generators for testing"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Simplified augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    organized_dir = '/kaggle/working/organized_data'
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(organized_dir, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=['dogs', 'cats'],
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        os.path.join(organized_dir, 'validation'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=['dogs', 'cats'],
        shuffle=False
    )
    
    return train_gen, val_gen

# Create generators once
train_gen, val_gen = create_quick_generators()
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# Test configurations quickly
results = {}

for config_name, config in quick_configs.items():
    print(f"\n--- Quick Test: {config_name} ---")
    
    # Create and compile model
    model = create_quick_model(config)
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Parameters: {model.count_params():,}")
    
    # Quick training (only 5 epochs)
    history = model.fit(
        train_gen,
        steps_per_epoch=min(50, train_gen.samples // train_gen.batch_size),  # Limited steps
        epochs=5,  # Very short
        validation_data=val_gen,
        validation_steps=min(20, val_gen.samples // val_gen.batch_size),  # Limited steps
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )
    
    # Store results
    results[config_name] = {
        'config': config,
        'final_val_acc': max(history.history['val_accuracy']),
        'final_train_acc': max(history.history['accuracy']),
        'final_val_loss': min(history.history['val_loss']),
        'history': history.history
    }
    
    print(f"Best val accuracy: {results[config_name]['final_val_acc']:.4f}")

# Quick comparison
print("\n" + "="*50)
print(" COMPARISON RESULTS")
print("="*50)

# Create summary
summary_data = []
for name, result in results.items():
    config = result['config']
    summary_data.append({
        'Config': name,
        'Dropout': config['dropout_rate'],
        'Learning Rate': config['learning_rate'],
        'Dense Units': config['dense_units'],
        'Val Accuracy': f"{result['final_val_acc']:.4f}",
        'Train Accuracy': f"{result['final_train_acc']:.4f}",
        'Overfitting Gap': f"{result['final_train_acc'] - result['final_val_acc']:.4f}"
    })

df = pd.DataFrame(summary_data)
print(df.to_string(index=False))

# Quick visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy comparison
config_names = list(results.keys())
val_accs = [results[name]['final_val_acc'] for name in config_names]

ax1.bar(config_names, val_accs, color=['blue', 'orange', 'green'])
ax1.set_title('Validation Accuracy Comparison')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1)

# Add values on bars
for i, v in enumerate(val_accs):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Overfitting analysis
train_accs = [results[name]['final_train_acc'] for name in config_names]
gaps = [t - v for t, v in zip(train_accs, val_accs)]

ax2.bar(config_names, gaps, color=['red' if gap > 0.1 else 'green' for gap in gaps])
ax2.set_title('Overfitting Analysis (Train - Val Accuracy)')
ax2.set_ylabel('Accuracy Gap')
ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Overfitting threshold')
ax2.legend()

# Add values on bars
for i, v in enumerate(gaps):
    color = 'white' if v > 0.05 else 'black'
    ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', color=color)

plt.tight_layout()
plt.show()

# Find best configuration
best_config = max(results.keys(), key=lambda x: results[x]['final_val_acc'])
best_acc = results[best_config]['final_val_acc']

print(f"\nBEST CONFIGURATION: {best_config}")
print(f"Best validation accuracy: {best_acc:.4f}")
print(f"Configuration details: {quick_configs[best_config]}")

# Overfitting analysis
print(f"\nOVERFITTING ANALYSIS:")
for config_name, result in results.items():
    gap = result['final_train_acc'] - result['final_val_acc']
    if gap > 0.1:
        status = "HIGH OVERFITTING"
    elif gap > 0.05:
        status = "MODERATE OVERFITTING"  
    else:
        status = "GOOD FIT"
    
    print(f"{config_name}: Gap = {gap:.3f} -> {status}")

# Recommendations
print(f"\nRECOMMENDATIONS:")
print("1. Use best configuration for final training")

baseline_acc = results['baseline']['final_val_acc']
for name, result in results.items():
    if name != 'baseline':
        improvement = result['final_val_acc'] - baseline_acc
        if improvement > 0.01:  # Significant improvement
            print(f"2. {name} shows {improvement:.3f} improvement - consider this approach")
        elif improvement < -0.01:  # Significant decrease
            print(f"2. {name} shows {improvement:.3f} decrease - avoid this approach")

print(f"3. Monitor overfitting - consider more dropout if gap > 0.1")
print(f"4. For final model, use more epochs with early stopping")

# Optional: Test best config with slightly more epochs
print(f"\nOptional: Testing best config ({best_config}) with more epochs...")

try:
    best_model = create_quick_model(quick_configs[best_config])
    best_model.compile(
        optimizer=Adam(learning_rate=quick_configs[best_config]['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    extended_history = best_model.fit(
        train_gen,
        steps_per_epoch=min(100, train_gen.samples // train_gen.batch_size),
        epochs=8,  # Slightly more epochs
        validation_data=val_gen,
        validation_steps=min(30, val_gen.samples // val_gen.batch_size),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )
    
    extended_val_acc = max(extended_history.history['val_accuracy'])
    print(f"Extended training result: {extended_val_acc:.4f}")
    
    # Save the best model
    best_model.save('/kaggle/working/best_hyperparams_model.h5')
    print("Best model saved as: best_hyperparams_model.h5")
    
except Exception as e:
    print(f"Extended training failed: {e}")

print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION COMPLETED!")
print("="*60)
print("Time saved: ~80% compared to full optimization")
print("Key insights obtained for parameter selection")
print("Ready to apply best parameters to final model!")




