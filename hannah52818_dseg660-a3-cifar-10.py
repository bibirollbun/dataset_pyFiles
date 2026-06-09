import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import subprocess
import zipfile

# Install py7zr for .7z extraction
try:
    import py7zr
    print("py7zr is available")
except ImportError:
    print("Installing py7zr...")
    subprocess.run(["pip", "install", "py7zr"], check=True)
    import py7zr
    print("py7zr installed successfully")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Load CIFAR-10 data for training
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation data shape: {x_val.shape}")
print(f"Validation labels shape: {y_val.shape}")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1
)

def create_cnn_model():
    model = models.Sequential([
        # First block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),   # ↑ from 0.25
        
        # Second block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),   # ↑ from 0.25
        
        # Third block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),   # ↑ from 0.25
        
        # Fourth block
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),   # ↑ from 0.25
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),   # ↓ from 0.5
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),   # ↓ from 0.5
        layers.Dense(10, activation='softmax')
    ])
    
    return model



# # Create CNN model (inspired by VGG but smaller)
# def create_cnn_model():
#     model = models.Sequential([
#         # First block
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
#         layers.BatchNormalization(),
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
        
#         # Second block
#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
        
#         # Third block
#         layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
        
#         # Fourth block
#         layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
#         layers.BatchNormalization(),
#         layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Dropout(0.25),
        
#         # Dense layers
#         layers.Flatten(),
#         layers.Dense(512, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),
#         layers.Dense(256, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),
#         layers.Dense(10, activation='softmax')
#     ])
    
#     return model

# Create and compile model
model = create_cnn_model()

# Compile model with Adam optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Define callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Train the model
print("Starting training...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=len(x_train) // 128
)

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Evaluate on CIFAR-10 validation set
val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"CIFAR-10 Validation Accuracy: {val_accuracy:.4f}")

# Function to load Kaggle test images (using the suggested approach)
def load_test_images():
    """
    Load test images from the Kaggle dataset.
    This function will search for the test images, including extracting from .7z files.
    """
    import zipfile
    import py7zr
    
    # Check what's available in the input directory
    print("Available directories in /kaggle/input/:")
    if os.path.exists('/kaggle/input/'):
        for item in os.listdir('/kaggle/input/'):
            item_path = f'/kaggle/input/{item}'
            print(f"  {item}/")
            # Show what's inside each directory
            if os.path.isdir(item_path):
                try:
                    contents = os.listdir(item_path)
                    for content in contents[:10]:  # Show first 10 items
                        print(f"    {content}")
                    if len(contents) > 10:
                        print(f"    ... and {len(contents) - 10} more items")
                except:
                    print("    (cannot read contents)")
    
    # Check if we have the test.7z file
    test_7z_path = '/kaggle/input/cifar-10/test.7z'
    if os.path.exists(test_7z_path):
        print(f"\nFound test.7z file at: {test_7z_path}")
        print("Extracting test images from .7z file...")
        
        # Create a directory to extract to
        extract_dir = '/kaggle/working/test_images'
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            # Extract the .7z file
            with py7zr.SevenZipFile(test_7z_path, mode='r') as archive:
                archive.extractall(path=extract_dir)
            
            # Find the actual test directory after extraction
            test_dir = None
            for root, dirs, files in os.walk(extract_dir):
                if any(f.endswith('.png') for f in files):
                    test_dir = root
                    break
            
            if test_dir:
                print(f"Successfully extracted test images to: {test_dir}")
                png_files = glob.glob(os.path.join(test_dir, '*.png'))
                print(f"Found {len(png_files)} PNG files")
            else:
                print("No PNG files found after extraction")
                return None, None
                
        except Exception as e:
            print(f"Error extracting .7z file: {e}")
            print("Trying alternative extraction method...")
            
            # Alternative: try using zipfile (sometimes .7z files work with this)
            try:
                with zipfile.ZipFile(test_7z_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find the actual test directory after extraction
                test_dir = None
                for root, dirs, files in os.walk(extract_dir):
                    if any(f.endswith('.png') for f in files):
                        test_dir = root
                        break
                
                if test_dir:
                    print(f"Successfully extracted test images to: {test_dir}")
                    png_files = glob.glob(os.path.join(test_dir, '*.png'))
                    print(f"Found {len(png_files)} PNG files")
                else:
                    print("No PNG files found after extraction")
                    return None, None
                    
            except Exception as e2:
                print(f"Alternative extraction also failed: {e2}")
                print("The .7z file might need manual extraction or a different method")
                return None, None
    
    else:
        # Try other possible paths
        possible_paths = [
            '/kaggle/input/cifar-10/test',
            '/kaggle/input/cifar10/test', 
            '/kaggle/input/cifar-10-test-images/test',
            '/kaggle/input/d/safios/cifar-10/test',
            '/kaggle/input/safios-cifar-10/test',
            '/kaggle/input/cifar10-test/test'
        ]
        
        # Also check all subdirectories for PNG files
        if os.path.exists('/kaggle/input/'):
            for root_item in os.listdir('/kaggle/input/'):
                root_path = f'/kaggle/input/{root_item}'
                if os.path.isdir(root_path):
                    # Check if there's a test subdirectory
                    test_path = os.path.join(root_path, 'test')
                    if os.path.exists(test_path):
                        possible_paths.append(test_path)
                    # Also check if PNG files are directly in the root directory
                    try:
                        files_in_dir = os.listdir(root_path)
                        if any(f.endswith('.png') for f in files_in_dir):
                            possible_paths.append(root_path)
                        # Check subdirectories too
                        for subdir in files_in_dir:
                            subdir_path = os.path.join(root_path, subdir)
                            if os.path.isdir(subdir_path):
                                try:
                                    sub_files = os.listdir(subdir_path)
                                    if any(f.endswith('.png') for f in sub_files):
                                        possible_paths.append(subdir_path)
                                except:
                                    pass
                    except:
                        pass
        
        test_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                # Check if there are actually PNG files in this directory
                png_files = glob.glob(os.path.join(path, '*.png'))
                if png_files:
                    test_dir = path
                    print(f"Found {len(png_files)} PNG files in: {path}")
                    break
        
        if test_dir is None:
            print("No test.7z file found and no PNG directories located")
            return None, None
    
    print(f"Using test directory: {test_dir}")
    
    # Collect only PNG files
    test_files = sorted(
        glob.glob(os.path.join(test_dir, '*.png')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])  # removes ".png"
    )
    
    print(f"Found {len(test_files)} test images")
    
    test_images = []
    test_ids = []
    
    for file_path in test_files:
        try:
            img = Image.open(file_path).convert('RGB')
            img = img.resize((32, 32))
            img_array = np.array(img)
            test_images.append(img_array)
            
            # Extract ID (filename without extension)
            file_id = int(os.path.splitext(os.path.basename(file_path))[0])
            test_ids.append(file_id)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return np.array(test_images), test_ids

# Load test images
print("Loading Kaggle test images...")
test_images, test_ids = load_test_images()

if test_images is not None:
    print(f"Loaded {len(test_images)} test images")
    
    # Preprocess test images
    test_images = test_images.astype('float32') / 255.0
    
    # Make predictions on test set
    print("Making predictions...")
    test_predictions = model.predict(test_images, batch_size=128, verbose=1)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    # Convert predictions to class names
    test_pred_labels = [class_names[pred] for pred in test_pred_classes]
    
    # Create Kaggle Submission
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': test_pred_labels
    })
    
    # Sort by ID
    submission_df = submission_df.sort_values('id').reset_index(drop=True)
    
    # Display sample predictions
    print("\nSample predictions:")
    print(submission_df.head(10))
    
    # Save submission file
    submission_df.to_csv('submission.csv', index=False)
    print(f"\nSubmission file saved as 'submission.csv' with {len(submission_df)} predictions")
    
    # Display submission statistics
    print("\nPrediction distribution:")
    print(submission_df['label'].value_counts().sort_index())
    
else:
    print("\n" + "="*70)
    print("SOLUTION: You need a DIFFERENT dataset with individual PNG images!")
    print("="*70)
    print("\nWhat you currently have:")
    print("- Standard CIFAR-10 dataset (with pickled/binary files)")
    print("\nWhat you need:")
    print("- CIFAR-10 competition dataset with individual PNG test images")
    print("\nHow to get it:")
    print("1. Click '+ Add input' in your notebook")
    print("2. Search for these specific datasets:")
    print("   - 'cifar-10 competition test'")
    print("   - 'safios/cifar-10' (this one should work)")
    print("   - 'cifar 10 png images'")
    print("3. Look for a dataset that mentions 'PNG files' or 'individual images'")
    print("4. The dataset should be around 100-300MB (not just a few MB)")
    print("\nOR download from the competition directly:")
    print("https://www.kaggle.com/c/cifar-10/data")
    print("="*70)
    
    # Create a submission using validation predictions for now
    print("\nCreating temporary submission using validation set predictions...")
    
    # Use the CIFAR-10 validation set to create a sample submission
    val_predictions = model.predict(x_val, batch_size=128, verbose=1)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    val_pred_labels = [class_names[pred] for pred in val_pred_classes]
    
    # Create a sample submission (first 300 predictions to match typical test size)
    sample_size = min(300, len(val_pred_labels))
    submission_df = pd.DataFrame({
        'id': list(range(1, sample_size + 1)),
        'label': val_pred_labels[:sample_size]
    })
    
    submission_df.to_csv('submission.csv', index=False)
    print(f"\nTemporary submission.csv created with {len(submission_df)} entries")
    print("NOTE: This uses validation set predictions - you need real test images for final submission!")
    
    print(f"\nSample of temporary submission:")
    print(submission_df.head(10))
    print(f"\nPrediction distribution:")
    print(submission_df['label'].value_counts().sort_index())

# Save the model
model.save('cifar10_model.h5')
print("Model saved as 'cifar10_model.h5'")

print(f"\nFinal validation accuracy: {val_accuracy:.4f}")
print("Training completed successfully!")

