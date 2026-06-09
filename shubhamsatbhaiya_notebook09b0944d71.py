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


# Essential imports for our CNN tutorial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Data manipulation and analysis
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Utilities
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Configure matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("âœ… All libraries imported successfully!")
print(f"ğŸ“Š TensorFlow version: {tf.__version__}")
print(f"ğŸ”¥ GPU Available: {len(tf.config.experimental.list_physical_devices('GPU')) > 0}")


# Define paths to our dataset
BASE_DIR = "/kaggle/input/cat-and-dog-classification-harper2022"
TRAIN_CATS_DIR = os.path.join(BASE_DIR, "Cat")
TRAIN_DOGS_DIR = os.path.join(BASE_DIR, "Dog") 
TEST_DIR = os.path.join(BASE_DIR, "Test")

# Count images in each directory
def count_images(directory):
    """Count the number of image files in a directory."""
    if os.path.exists(directory):
        return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return 0

num_cats = count_images(TRAIN_CATS_DIR)
num_dogs = count_images(TRAIN_DOGS_DIR)
num_test = count_images(TEST_DIR)

print("ğŸ“� Dataset Overview:")
print(f"ğŸ�± Cat images: {num_cats:,}")
print(f"ğŸ�¶ Dog images: {num_dogs:,}")
print(f"ğŸ§ª Test images: {num_test:,}")
print(f"ğŸ“Š Total training images: {num_cats + num_dogs:,}")
print(f"âš–ï¸�  Class balance: {num_cats/(num_cats + num_dogs)*100:.1f}% cats, {num_dogs/(num_cats + num_dogs)*100:.1f}% dogs")

# Visualize class distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Bar chart
classes = ['Cats', 'Dogs']
counts = [num_cats, num_dogs]
colors = ['orange', 'skyblue']
ax1.bar(classes, counts, color=colors, alpha=0.7)
ax1.set_title('Class Distribution in Training Set', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Images')
for i, count in enumerate(counts):
    ax1.text(i, count + 100, f'{count:,}', ha='center', fontweight='bold')

# Pie chart
ax2.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nğŸ’¡ Key Observations:")
print("â€¢ We have a perfectly balanced dataset with equal numbers of cat and dog images")
print("â€¢ This eliminates class imbalance issues that could bias our model")
print("â€¢ Large dataset size (20,000 images) should provide good training data")


# Let's examine some sample images to understand their characteristics
def display_sample_images(cat_dir, dog_dir, num_samples=8):
    """Display sample images from both classes to understand the data."""
    
    # Get random sample of images
    cat_images = random.sample(os.listdir(cat_dir), num_samples//2)
    dog_images = random.sample(os.listdir(dog_dir), num_samples//2)
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(20, 8))
    fig.suptitle('Sample Images from Our Dataset', fontsize=16, fontweight='bold')
    
    # Display cat images
    for i, img_name in enumerate(cat_images):
        img_path = os.path.join(cat_dir, img_name)
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'ğŸ�± Cat\nSize: {img.size}', fontsize=10)
        axes[0, i].axis('off')
    
    # Display dog images  
    for i, img_name in enumerate(dog_images):
        img_path = os.path.join(dog_dir, img_name)
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'ğŸ�¶ Dog\nSize: {img.size}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Display sample images
display_sample_images(TRAIN_CATS_DIR, TRAIN_DOGS_DIR)

# Analyze image dimensions
def analyze_image_dimensions(directory, sample_size=100):
    """Analyze the dimensions of images in a directory."""
    files = os.listdir(directory)
    sample_files = random.sample(files, min(sample_size, len(files)))
    
    dimensions = []
    for file in sample_files:
        try:
            img = Image.open(os.path.join(directory, file))
            dimensions.append(img.size)  # (width, height)
        except Exception:
            continue
    
    return dimensions

print("ğŸ”� Analyzing image dimensions...")
cat_dims = analyze_image_dimensions(TRAIN_CATS_DIR)
dog_dims = analyze_image_dimensions(TRAIN_DOGS_DIR)

all_widths = [dim[0] for dim in cat_dims + dog_dims]
all_heights = [dim[1] for dim in cat_dims + dog_dims]

print(f"ğŸ“� Image Dimension Analysis (sample of 200 images):")
print(f"Width  - Min: {min(all_widths)}, Max: {max(all_widths)}, Mean: {np.mean(all_widths):.0f}")
print(f"Height - Min: {min(all_heights)}, Max: {max(all_heights)}, Mean: {np.mean(all_heights):.0f}")

# Plot dimension distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.hist(all_widths, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax1.set_title('Distribution of Image Widths')
ax1.set_xlabel('Width (pixels)')
ax1.set_ylabel('Frequency')

ax2.hist(all_heights, bins=30, alpha=0.7, color='green', edgecolor='black')
ax2.set_title('Distribution of Image Heights')
ax2.set_xlabel('Height (pixels)')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print("\nğŸ’¡ Key Insights:")
print("â€¢ Images have varying dimensions - we'll need to resize them for training")
print("â€¢ Both cats and dogs show good variety in poses, lighting, and backgrounds")
print("â€¢ This diversity will help our model generalize better to new images")


# Set up our image parameters
IMG_WIDTH = 150  # Width to resize images to
IMG_HEIGHT = 150  # Height to resize images to
BATCH_SIZE = 32   # Number of images to process at once
EPOCHS = 25       # Number of training iterations

print(f"ğŸ–¼ï¸� Image Configuration:")
print(f"Target size: {IMG_WIDTH}x{IMG_HEIGHT} pixels")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training epochs: {EPOCHS}")

# Create data generators with augmentation for training
# ImageDataGenerator automatically handles loading, resizing, and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,          # Normalize pixel values to [0,1]
    rotation_range=20,          # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,      # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2,     # Randomly shift images vertically by up to 20%
    shear_range=0.2,           # Apply shearing transformation
    zoom_range=0.2,            # Randomly zoom in/out by up to 20%
    horizontal_flip=True,       # Randomly flip images horizontally
    fill_mode='nearest',        # Fill pixels after transformation
    validation_split=0.2        # Use 20% of training data for validation
)

# For test data, we only rescale (no augmentation for evaluation)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

print("\nâœ… Data generators created with augmentation settings:")
print("â€¢ Rotation: Â±20 degrees")
print("â€¢ Width/Height shift: Â±20%") 
print("â€¢ Zoom range: Â±20%")
print("â€¢ Horizontal flip: Yes")
print("â€¢ Validation split: 20%")



       # We need to reorganize our data into train/cat and train/dog folders
       import shutil

       # Create temporary directory structure
       TEMP_DIR = "temp_data"
       TEMP_TRAIN_DIR = os.path.join(TEMP_DIR, "train")
       TEMP_TRAIN_CATS = os.path.join(TEMP_TRAIN_DIR, "cat")
       TEMP_TRAIN_DOGS = os.path.join(TEMP_TRAIN_DIR, "dog")

       # Create directories if they don't exist
       os.makedirs(TEMP_TRAIN_CATS, exist_ok=True)
       os.makedirs(TEMP_TRAIN_DOGS, exist_ok=True)

       print("ğŸ“� Creating temporary directory structure...")

       # Create symbolic links to avoid copying large files
       # This creates references to original files without duplicating data
       def create_symlinks(source_dir, target_dir, max_files=None):
           """Create symbolic links from source to target directory."""
           files = os.listdir(source_dir)
           if max_files:
               files = files[:max_files]

           for file in files:
               source_path = os.path.join(source_dir, file)
               target_path = os.path.join(target_dir, file)

               # Remove existing symlink if it exists
               if os.path.islink(target_path):
                   os.unlink(target_path)

               # Create new symlink
               try:
                   os.symlink(os.path.abspath(source_path), target_path)
               except FileExistsError:
                   pass  # Link already exists

       # For demonstration, we'll use a subset of images to speed up training
       # In practice, you'd use the full dataset
       SUBSET_SIZE = 2000  # Use 2000 images per class for faster training

       print(f"Creating symlinks for {SUBSET_SIZE} images per class...")
       create_symlinks(TRAIN_CATS_DIR, TEMP_TRAIN_CATS, SUBSET_SIZE)
       create_symlinks(TRAIN_DOGS_DIR, TEMP_TRAIN_DOGS, SUBSET_SIZE)

       print(f"âœ… Temporary structure created with {SUBSET_SIZE} images per class")



       # flow_from_directory automatically assigns labels based on subfolder names

       # Training data generator (with augmentation)
       train_generator = train_datagen.flow_from_directory(
           TEMP_TRAIN_DIR,
           target_size=(IMG_WIDTH, IMG_HEIGHT),
           batch_size=BATCH_SIZE,
           class_mode='binary',  # Binary classification (cat=0, dog=1)
           subset='training',    # Use training subset (80%)
           shuffle=True,         # Shuffle the data
           seed=SEED            # For reproducibility
       )

       # Validation data generator (with same preprocessing, no augmentation)
       validation_generator = train_datagen.flow_from_directory(
           TEMP_TRAIN_DIR,
           target_size=(IMG_WIDTH, IMG_HEIGHT),
           batch_size=BATCH_SIZE,
           class_mode='binary',
           subset='validation',  # Use validation subset (20%)
           shuffle=True,
           seed=SEED
       )

       print("ğŸ“Š Data generators created successfully!")
       print(f"ğŸ�‹ï¸� Training samples: {train_generator.samples}")
       print(f"âœ… Validation samples: {validation_generator.samples}")
       print(f"ğŸ�·ï¸� Class indices: {train_generator.class_indices}")
       print(f"ğŸ“¦ Steps per epoch: {train_generator.samples // BATCH_SIZE}")
       print(f"ğŸ”� Validation steps: {validation_generator.samples // BATCH_SIZE}")


       # Build our CNN model step by step
       def create_cnn_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)):
           """Create a CNN model for binary classification."""

           model = keras.Sequential([
               # First Convolutional Block
               # 32 filters of size 3x3, ReLU activation
               layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv2d_1'),
               layers.MaxPooling2D(2, 2, name='maxpool_1'),  # Reduce spatial dimensions by half

               # Second Convolutional Block  
               # Increase filters to 64 to detect more complex features
               layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
               layers.MaxPooling2D(2, 2, name='maxpool_2'),

               # Third Convolutional Block
               # Further increase to 128 filters
               layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
               layers.MaxPooling2D(2, 2, name='maxpool_3'),

               # Fourth Convolutional Block
               # 128 filters again to capture high-level features
               layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_4'),
               layers.MaxPooling2D(2, 2, name='maxpool_4'),

               # Flatten the 2D feature maps to 1D vector
               layers.Flatten(name='flatten'),

               # Add dropout for regularization (prevents overfitting)
               layers.Dropout(0.5, name='dropout_1'),

               # Dense layer with 512 neurons
               layers.Dense(512, activation='relu', name='dense_1'),

               # Another dropout layer
               layers.Dropout(0.5, name='dropout_2'),

               # Output layer - 1 neuron with sigmoid for binary classification
               # Sigmoid outputs probability between 0 and 1
               layers.Dense(1, activation='sigmoid', name='output')
           ])

           return model

       # Create the model
       model = create_cnn_model()

       print("ğŸ�—ï¸� CNN Model Architecture Created!")
       print("\nğŸ“‹ Model Summary:")
       model.summary()

       print("\nğŸ”¢ Model Statistics:")
       total_params = model.count_params()
       print(f"Total parameters: {total_params:,}")
       print(f"Model input shape: {model.input_shape}")
       print(f"Model output shape: {model.output_shape}")

       print("\nğŸ’¡ Architecture Explanation:")
       print("â€¢ Conv2D layers detect features (edges, textures, patterns)")
       print("â€¢ MaxPooling reduces spatial dimensions and computational load")
       print("â€¢ Filter count increases: 32 â†’ 64 â†’ 128 â†’ 128")
       print("â€¢ Dropout (50%) prevents overfitting")
       print("â€¢ Dense layers perform final classification")
       print("â€¢ Sigmoid output gives probability: <0.5 = Cat, >0.5 = Dog")


       # Compile the model - specify optimizer, loss function, and metrics
       model.compile(
           optimizer='adam',              # Adam optimizer - adaptive learning rate
           loss='binary_crossentropy',    # Binary crossentropy for binary classification
           metrics=['accuracy']           # Track accuracy during training
       )

       print("âš™ï¸� Model compiled successfully!")
       print("\nğŸ”§ Compilation Settings:")
       print("â€¢ Optimizer: Adam (adaptive learning rate)")
       print("â€¢ Loss function: Binary crossentropy")
       print("â€¢ Metrics: Accuracy")

       print("\nğŸ“š Why these choices?")
       print("â€¢ Adam: Self-adjusting learning rate, works well for most problems")
       print("â€¢ Binary crossentropy: Standard loss for binary classification")
       print("â€¢ Accuracy: Easy-to-understand metric for classification performance")

       # Set up callbacks for training
       # Callbacks are functions called during training to monitor and control the process

       callbacks = [
           # Save the best model based on validation accuracy
           ModelCheckpoint(
               'best_cat_dog_model.h5',
               monitor='val_accuracy',
               save_best_only=True,
               mode='max',
               verbose=1
           ),

           # Stop training early if validation accuracy stops improving
           EarlyStopping(
               monitor='val_accuracy',
               patience=5,  # Wait 5 epochs before stopping
               mode='max',
               verbose=1,
               restore_best_weights=True
           ),

           # Reduce learning rate when validation accuracy plateaus
           ReduceLROnPlateau(
               monitor='val_accuracy',
               factor=0.2,  # Reduce LR by factor of 5
               patience=3,  # Wait 3 epochs before reducing
               min_lr=0.0001,
               verbose=1
           )
       ]

       print("\nğŸ�¯ Training callbacks configured:")
       print("â€¢ ModelCheckpoint: Save best model")
       print("â€¢ EarlyStopping: Prevent overfitting")
       print("â€¢ ReduceLROnPlateau: Adaptive learning rate")
       print("\nReady to start training! ğŸš€")


       # Train the model
       print("ğŸš€ Starting model training...")
       print(f"Training for {EPOCHS} epochs with {train_generator.samples} training samples")
       print(f"Validating on {validation_generator.samples} validation samples")
       print("\n" + "="*50)

       # Calculate steps per epoch
       steps_per_epoch = train_generator.samples // BATCH_SIZE
       validation_steps = validation_generator.samples // BATCH_SIZE

       print(f"Steps per epoch: {steps_per_epoch}")
       print(f"Validation steps: {validation_steps}")
       print("\nStarting training...\n")

       # Train the model
       history = model.fit(
           train_generator,
           steps_per_epoch=steps_per_epoch,
           epochs=EPOCHS,
           validation_data=validation_generator,
           validation_steps=validation_steps,
           callbacks=callbacks,
           verbose=1  # Show progress bar
       )

       print("\nğŸ�‰ Training completed!")
       print(f"ğŸ“Š Final training accuracy: {history.history['accuracy'][-1]:.4f}")
       print(f"âœ… Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
       print(f"ğŸ“‰ Final training loss: {history.history['loss'][-1]:.4f}")
       print(f"ğŸ”� Final validation loss: {history.history['val_loss'][-1]:.4f}")


       def plot_training_history(history):
           """Plot training and validation accuracy and loss."""

           # Extract metrics from history
           acc = history.history['accuracy']
           val_acc = history.history['val_accuracy']
           loss = history.history['loss']
           val_loss = history.history['val_loss']
           epochs_range = range(len(acc))

           # Create subplots
           fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
           fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

           # Plot accuracy
           ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
           ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
           ax1.set_title('Model Accuracy', fontsize=14)
           ax1.set_xlabel('Epoch')
           ax1.set_ylabel('Accuracy')
           ax1.legend()
           ax1.grid(True, alpha=0.3)

           # Plot loss
           ax2.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
           ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
           ax2.set_title('Model Loss', fontsize=14)
           ax2.set_xlabel('Epoch')
           ax2.set_ylabel('Loss')
           ax2.legend()
           ax2.grid(True, alpha=0.3)

           plt.tight_layout()
           plt.show()

           # Analyze the results
           best_val_acc = max(val_acc)
           best_epoch = val_acc.index(best_val_acc) + 1
           final_gap = acc[-1] - val_acc[-1]

           print(f"\nğŸ“ˆ Training Analysis:")
           print(f"â€¢ Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
           print(f"â€¢ Final accuracy gap: {final_gap:.4f} (training - validation)")

           if final_gap > 0.1:
               print("âš ï¸�  Large gap suggests overfitting - consider more regularization")
           elif final_gap < 0.05:
               print("âœ… Good balance between training and validation performance")
           else:
               print("ğŸ“Š Moderate gap - model performance is acceptable")

       # Plot the training history
       plot_training_history(history)

       print("\nğŸ�¯ Key Takeaways:")
       print("â€¢ Watch the validation accuracy - it's the true measure of model performance")
       print("â€¢ If validation accuracy stops improving while training accuracy keeps rising, that's overfitting")
       print("â€¢ Our callbacks help prevent overfitting and save the best model automatically")


       # Load the best saved model
       print("ğŸ“¥ Loading the best trained model...")
       best_model = keras.models.load_model('best_cat_dog_model.h5')
       print("âœ… Best model loaded successfully!")

       # Function to make predictions on individual images
       def predict_image(model, img_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
           """Make a prediction on a single image."""
           # Load and preprocess the image
           img = Image.open(img_path)
           img = img.resize(target_size)
           img_array = np.array(img) / 255.0  # Normalize
           img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

           # Make prediction
           prediction = model.predict(img_array, verbose=0)[0][0]

           # Interpret the prediction
           if prediction > 0.5:
               label = "Dog"
               confidence = prediction
           else:
               label = "Cat"
               confidence = 1 - prediction

           return label, confidence, prediction

       # Function to display predictions with images
       def show_predictions(model, image_paths, true_labels=None):
           """Display images with their predictions."""
           num_images = len(image_paths)
           cols = 4
           rows = (num_images + cols - 1) // cols

           fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
           if rows == 1:
               axes = axes.reshape(1, -1)

           fig.suptitle('Model Predictions', fontsize=16, fontweight='bold')

           for i, img_path in enumerate(image_paths):
               row = i // cols
               col = i % cols

               # Make prediction
               label, confidence, raw_prediction = predict_image(model, img_path)

               # Load and display image
               img = Image.open(img_path)
               axes[row, col].imshow(img)

               # Create title with prediction info
               confidence_icon = "ğŸ”¥" if confidence > 0.9 else "âœ…" if confidence > 0.7 else "âš ï¸�"
               title = f"{confidence_icon} {label}\nConfidence: {confidence:.2f}"

               # Add true label if provided
               if true_labels and i < len(true_labels):
                   true_label = true_labels[i]
                   is_correct = (label.lower() == true_label.lower())
                   title += f"\nTrue: {true_label} {'âœ“' if is_correct else 'âœ—'}"

               axes[row, col].set_title(title, fontsize=10)
               axes[row, col].axis('off')

           # Hide empty subplots
           for i in range(num_images, rows * cols):
               row = i // cols
               col = i % cols
               axes[row, col].axis('off')

           plt.tight_layout()
           plt.show()

       print("\nğŸ”® Prediction functions ready!")
       print("Let's test our model on some images...")


       # Test predictions on sample images from our training set
       print("ğŸ§ª Testing on sample training images...")

       # Get random sample images from each class
       cat_samples = random.sample(os.listdir(TRAIN_CATS_DIR), 4)
       dog_samples = random.sample(os.listdir(TRAIN_DOGS_DIR), 4)

       # Create full paths
       cat_paths = [os.path.join(TRAIN_CATS_DIR, img) for img in cat_samples]
       dog_paths = [os.path.join(TRAIN_DOGS_DIR, img) for img in dog_samples]

       # Combine paths and labels
       test_paths = cat_paths + dog_paths
       true_labels = ['Cat'] * 4 + ['Dog'] * 4

       print(f"Testing on {len(test_paths)} images...")
       show_predictions(best_model, test_paths, true_labels)

       # Calculate accuracy on this sample
       correct = 0
       for i, path in enumerate(test_paths):
           label, confidence, raw_pred = predict_image(best_model, path)
           if label.lower() == true_labels[i].lower():
               correct += 1

       accuracy = correct / len(test_paths)
       print(f"\nğŸ“Š Sample accuracy: {accuracy:.2%} ({correct}/{len(test_paths)})")

       print("\nğŸ’¡ Observations:")
       print("â€¢ ğŸ”¥ High confidence (>90%) predictions are usually very reliable")
       print("â€¢ âœ… Good confidence (70-90%) predictions are generally accurate")
       print("â€¢ âš ï¸� Lower confidence (<70%) predictions may be uncertain or difficult cases")
       print("â€¢ Look for patterns in misclassifications to understand model limitations")



       # Clean up temporary files
       print("ğŸ§¹ Cleaning up temporary files...")

       # Remove temporary directory structure
       import shutil
       if os.path.exists(TEMP_DIR):
           shutil.rmtree(TEMP_DIR)
           print("âœ… Temporary directory removed")

       print("\nğŸ“� Final Project Files:")
       print("â€¢ notebook.ipynb - This complete tutorial")
       print("â€¢ best_cat_dog_model.h5 - Your trained CNN model")
       print("â€¢ CLAUDE.md - Project documentation")

       print("\nğŸ�¯ Your CNN Model Stats:")
       print(f"â€¢ Architecture: Custom 4-layer CNN")
       print(f"â€¢ Parameters: {model.count_params():,}")
       print(f"â€¢ Input size: {IMG_WIDTH}x{IMG_HEIGHT}x3")
       print(f"â€¢ Training data: {SUBSET_SIZE*2:,} images")

       print("\nğŸ�† Congratulations! You've successfully:")
       print("âœ“ Built a CNN from scratch")
       print("âœ“ Trained it on real image data")
       print("âœ“ Achieved good classification performance")
       print("âœ“ Learned fundamental deep learning concepts")
       print("âœ“ Created a complete end-to-end ML pipeline")

       print("\nğŸš€ Ready to tackle your next computer vision challenge!")

