
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(len(filenames))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Install specific package versions for compatibility
%pip install numpy==1.22.4
%pip install --upgrade scipy


# CORE LIBRARIES
import os
import random
import warnings
from pathlib import Path
from tqdm import tqdm


# DATA PROCESSING & SCIENTIFIC COMPUTING
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest


# DEEP LEARNING & TENSORFLOW
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, ReLU, Concatenate, AvgPool2D, 
    GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K


# VISUALIZATION
import matplotlib.pyplot as plt


# CONFIGURATION
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print(f"âœ… TensorFlow version: {tf.__version__}")
print(f"âœ… GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"âœ… Environment setup complete!")



# ENVIRONMENT CONFIGURATION
# Set environment variable to handle large histopathological images
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))


# DATASET PATHS
competition_dataset_directory = Path('/kaggle/input/UBC-OCEAN')


# IMAGE PROCESSING PARAMETERS
PROCESSED_IMAGE_SIZE = 224     # Target image size for DenseNet-121
JPEG_QUALITY = 80              # Compression quality for processed images


# TRAINING PARAMETERS
VALIDATION_SPLIT = 0.2         # 20% for validation
RANDOM_STATE = 42              # For reproducible results
TARGET_TRAINING_SAMPLES = 30000 # Desired dataset size
MAX_OUTLIER_PERCENT = 0.05     # Maximum 5% outliers to remove

print(f"ğŸ�¯ Target image size: {PROCESSED_IMAGE_SIZE}x{PROCESSED_IMAGE_SIZE}")
print(f"ğŸ“Š Validation split: {VALIDATION_SPLIT*100}%")
print(f"ğŸ”¢ Random state: {RANDOM_STATE}")



# LOAD DATASET FILES
print("ğŸ“– Loading dataset files...")
df_train = pd.read_csv(competition_dataset_directory / 'train.csv')
df_test = pd.read_csv(competition_dataset_directory / 'test.csv')

print(f"âœ… Training samples: {len(df_train):,}")
print(f"âœ… Test samples: {len(df_test):,}")
print(f"âœ… Label distribution:")
print(df_train['label'].value_counts())

# CREATE PROCESSING DIRECTORIES
print("\nğŸ“� Creating directories for processed images...")
train_processed_dir = Path('./train_processed_images')
test_processed_dir = Path('./test_processed_images')
train_processed_dir.mkdir(exist_ok=True, parents=True)   
test_processed_dir.mkdir(exist_ok=True, parents=True)

print(f"âœ… Training directory: {train_processed_dir}")
print(f"âœ… Test directory: {test_processed_dir}")



# Install OpenSlide for handling whole slide images (WSI)
%pip install openslide-python


def resize_with_aspect_ratio(image, target_size):
    """
    Resize image to target size while preserving aspect ratio
    
    Parameters
    ----------
    image: numpy.ndarray of shape (height, width, 3)
        Image array
    target_size: int
        Desired size for both dimensions
        
    Returns
    -------
    resized_image: numpy.ndarray of shape (target_size, target_size, 3)
        Resized and padded image array
    """
    height, width = image.shape[:2]
    
    # Calculate scaling factor to preserve aspect ratio
    scale = min(target_size / height, target_size / width)
    
    # Calculate new dimensions
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas of target size
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
     # Calculate offsets to center the image
    y_offset = (target_size - new_height) // 2
    x_offset = (target_size - new_width) // 2
    
    # Place the resized image on the canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas



def process_by_tiles(raw_image_path, processed_image_path, tile_size=1024):
    """
    Process very large images by tiling them and taking the center tile
    
    Parameters
    ----------
    raw_image_path: str
        Path to the input image
    processed_image_path: str
        Path to save the processed image
    tile_size: int
        Size of tiles to extract
    """
    try:
        from PIL import Image
        img = Image.open(raw_image_path)
        width, height = img.size
        
        # Take center tile
        center_x, center_y = width // 2, height // 2
        left = max(0, center_x - tile_size // 2)
        top = max(0, center_y - tile_size // 2)
        right = min(width, left + tile_size)
        bottom = min(height, top + tile_size)
        
        # Crop and resize
        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize((PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE), Image.LANCZOS)
        resized.save(processed_image_path, quality=JPEG_QUALITY)
        
    except Exception as e:
        print(f"Tiling process failed: {e}")
        raise e


def process_large_images(df, source_dir, target_dir, is_train=True):
    processed_paths = []
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
        image_id = row["image_id"]
        raw_image_path = str(source_dir / f'{image_id}.png')
        processed_image_path = str(target_dir / f'{image_id}.jpg')

        if os.path.exists(processed_image_path):
            processed_paths.append(processed_image_path)
            continue

        # Check file extension to determine processing method
        file_extension = Path(raw_image_path).suffix.lower()
        
        # 1. For WSI formats, try OpenSlide first
        if file_extension in ['.svs', '.ndpi', '.tiff', '.tif', '.vms', '.vmu', '.scn', '.mrxs', '.bif']:
            try:
                import openslide
                from PIL import Image

                slide = openslide.OpenSlide(raw_image_path)
                width, height = slide.dimensions
                scale = min(PROCESSED_IMAGE_SIZE / height, PROCESSED_IMAGE_SIZE / width)
                new_w, new_h = int(width * scale), int(height * scale)

                thumbnail = slide.get_thumbnail((new_w, new_h))

                # Center on canvas
                result = Image.new("RGB", (PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE), (0, 0, 0))
                result.paste(thumbnail, ((PROCESSED_IMAGE_SIZE - new_w) // 2, (PROCESSED_IMAGE_SIZE - new_h) // 2))
                result.save(processed_image_path, quality=JPEG_QUALITY)
                processed_paths.append(processed_image_path)
                continue
            except Exception as e:
                print(f"OpenSlide failed for {raw_image_path}: {e}")

        # 2. For standard image formats (PNG, JPG, etc.), use PIL as primary method
        try:
            from PIL import Image
            import warnings
            warnings.simplefilter('ignore', Image.DecompressionBombWarning)
            Image.MAX_IMAGE_PIXELS = None

            img = Image.open(raw_image_path)
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            width, height = img.size
            scale = min(PROCESSED_IMAGE_SIZE / height, PROCESSED_IMAGE_SIZE / width)
            new_w, new_h = int(width * scale), int(height * scale)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)

            result = Image.new("RGB", (PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE), (0, 0, 0))
            result.paste(img_resized, ((PROCESSED_IMAGE_SIZE - new_w) // 2, (PROCESSED_IMAGE_SIZE - new_h) // 2))
            result.save(processed_image_path, quality=JPEG_QUALITY)
            processed_paths.append(processed_image_path)
            continue
        except Exception as e:
            print(f"PIL failed for {raw_image_path}: {e}")

        # 3. Fallback to OpenCV
        try:
            import cv2
            image = cv2.imread(raw_image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized = resize_with_aspect_ratio(image, PROCESSED_IMAGE_SIZE)
                resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                cv2.imwrite(processed_image_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                processed_paths.append(processed_image_path)
                continue
        except Exception as e:
            print(f"OpenCV failed for {raw_image_path}: {e}")

        # 4. Last resort: tiling
        try:
            process_by_tiles(raw_image_path, processed_image_path)
            processed_paths.append(processed_image_path)
        except Exception as e:
            print(f"Tiling failed for {raw_image_path}: {e}")
            # Create black placeholder
            try:
                blank = np.zeros((PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE, 3), dtype=np.uint8)
                cv2.imwrite(processed_image_path, blank)
                processed_paths.append(processed_image_path)
                print(f"Created black placeholder for {image_id}")
            except:
                pass

    return processed_paths


# Process training and test images
train_image_paths = process_large_images(df_train, competition_dataset_directory / 'train_images', train_processed_dir, is_train=True)
test_image_paths = process_large_images(df_test, competition_dataset_directory / 'test_images', test_processed_dir, is_train=False)


# Create a label mapping
label_mapping = {label: idx for idx, label in enumerate(df_train['label'].unique())}
num_classes = len(label_mapping)
print(f"Number of classes: {num_classes}")
print(f"Label mapping: {label_mapping}")

# Map labels to numeric values
df_train['label_idx'] = df_train['label'].map(label_mapping)



def extract_image_features(image_path):
    """
    Extract basic features from an image for outlier detection
    
    Parameters
    ----------
    image_path: str
        Path to the image file
        
    Returns
    -------
    features: numpy.ndarray
        Array of extracted features or None if extraction fails
    """
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract color statistics for each channel
        features = []
        for c in range(3):
            channel = img[:,:,c]
            features.extend([
                np.mean(channel),       # Mean
                np.std(channel),        # Standard deviation
                np.percentile(channel, 5),  # 5th percentile
                np.percentile(channel, 95), # 95th percentile
            ])
        
        # Add texture features (using basic edge detection)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        features.append(np.mean(edges))
        features.append(np.std(edges))
        
        # Add basic shape features
        features.append(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))  # Edge density
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None



#Extract features for outlier detection
print("Extracting features for outlier detection...")
features_list = []
valid_image_indices = []

for i, path in enumerate(tqdm(train_image_paths, desc="Extracting features")):
    features = extract_image_features(path)
    if features is not None:
        features_list.append(features)
        valid_image_indices.append(i)

# Convert list to numpy array
X_features = np.array(features_list)
# Apply Isolation Forest for outlier detection
print("Detecting outliers using Isolation Forest...")
contamination = min(MAX_OUTLIER_PERCENT, 0.05)  # Max 5% outliers
outlier_detector = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
outlier_predictions = outlier_detector.fit_predict(X_features)

# Filter out outliers
inlier_indices = [valid_image_indices[i] for i, pred in enumerate(outlier_predictions) if pred == 1]
outlier_indices = [valid_image_indices[i] for i, pred in enumerate(outlier_predictions) if pred == -1]

print(f"Detected {len(outlier_indices)} outliers out of {len(valid_image_indices)} images ({len(outlier_indices)/len(valid_image_indices)*100:.2f}%)")

# Create filtered DataFrame
df_train_filtered = df_train.iloc[inlier_indices].reset_index(drop=True)
print(f"After outlier removal: {len(df_train_filtered)} training samples")



# DenseNet-121 Implementation following the tutorial

def bottleneck_layer(x, filters, strides=1):
    """Create bottleneck layer for DenseNet"""
    skip_connection = x
    # BN-ReLU-Conv(1Ã—1)-BN-ReLU-Conv(3Ã—3)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(4*filters, kernel_size=1, strides=strides, padding='same')(x)
    
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x)
    
    x = Concatenate()([x, skip_connection])
    return x

def transition_layer(x):
    """Create transition layer for DenseNet"""
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(K.int_shape(x)[-1]//2, kernel_size=1, strides=1, padding='same')(x)
    x = AvgPool2D(2, strides=2, padding='same')(x)
    return x

def dense_block(x, repetition=1, growth_rate=32):
    """Create dense block with multiple bottleneck layers"""
    for _ in range(repetition):
        x = bottleneck_layer(x, growth_rate)
    return x

def densenet121(input_shape, num_classes, growth_rate=32):
    """Create DenseNet-121 model"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial layer
    x = BatchNormalization()(inputs)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    
    # Pooling layer
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # First dense and transition layer (6 layers)
    x = dense_block(x, 6, growth_rate)
    x = transition_layer(x)
    
    # Second dense and transition layer (12 layers)
    x = dense_block(x, 12, growth_rate)
    x = transition_layer(x)
    
    # Third dense and transition layer (24 layers)
    x = dense_block(x, 24, growth_rate)
    x = transition_layer(x)
    
    # Last dense layer (16 layers)
    x = dense_block(x, 16, growth_rate)
    
    # Global average pooling layer
    x = GlobalAveragePooling2D()(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model


# Helper function to load an image
def load_image(image_path):
    """Load and return RGB image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Improved data generator with proper augmentation
class ImprovedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, image_dir, batch_size, datagen, 
                 target_size=(224, 224), shuffle=True, num_classes=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.datagen = datagen
        self.target_size = target_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        # Generate indexes for this batch
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.dataframe))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_size_actual = len(batch_indexes)
        X = np.empty((batch_size_actual, *self.target_size, 3), dtype=np.float32)
        y = np.empty((batch_size_actual, self.num_classes), dtype=np.float32)
        
        # Generate data
        for i, idx in enumerate(batch_indexes):
            # Get image path
            image_id = self.dataframe.loc[idx, 'image_id']
            img_path = self.image_dir / f'{image_id}.jpg'
            
            try:
                # Load image
                img = load_image(str(img_path))
                
                # Ensure correct size
                if img.shape[:2] != self.target_size:
                    img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
                
                # Apply augmentation
                img = img.astype(np.float32)
                if hasattr(self.datagen, 'random_transform'):
                    img = self.datagen.random_transform(img)
                
                # Apply preprocessing
                if hasattr(self.datagen, 'preprocessing_function') and self.datagen.preprocessing_function:
                    img = self.datagen.preprocessing_function(img)
                else:
                    img = img / 255.0  # Default normalization
                    
                X[i] = img
                
                # Get label
                label_idx = self.dataframe.loc[idx, 'label_idx']
                y[i] = to_categorical(label_idx, self.num_classes)
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Create black image as fallback
                X[i] = np.zeros((*self.target_size, 3), dtype=np.float32)
                label_idx = self.dataframe.loc[idx, 'label_idx']
                y[i] = to_categorical(label_idx, self.num_classes)
        
        return X, y


# Split into training and validation sets
train_df, val_df = train_test_split(
    df_train_filtered, 
    test_size=VALIDATION_SPLIT, 
    random_state=RANDOM_STATE,
    stratify=df_train_filtered['label_idx']
)

print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")


# Define augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='constant',
    cval=0
)

# Minimal augmentation for validation
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# Create improved data generators
batch_size = 32

# Training generator with augmentation
train_generator = ImprovedDataGenerator(
    train_df,
    train_processed_dir,
    batch_size=batch_size,
    datagen=train_datagen,
    target_size=(PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE),
    shuffle=True,
    num_classes=num_classes
)

# Validation generator without augmentation
val_generator = ImprovedDataGenerator(
    val_df,
    train_processed_dir,
    batch_size=batch_size,
    datagen=val_datagen,
    target_size=(PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE),
    shuffle=False,
    num_classes=num_classes
)

print(f"Training batches: {len(train_generator)}")
print(f"Validation batches: {len(val_generator)}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")


# Print dataset statistics
print(f"\n=== Dataset Statistics ===")
print(f"Number of classes: {num_classes}")
print(f"Label mapping: {label_mapping}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Training steps per epoch: {len(train_generator)}")
print(f"Validation steps: {len(val_generator)}")
print(f"Batch size: {batch_size}")
print(f"Image size: {PROCESSED_IMAGE_SIZE}x{PROCESSED_IMAGE_SIZE}")


# Create DenseNet-121 model
print("Creating DenseNet-121 model...")
model = densenet121(
    input_shape=(PROCESSED_IMAGE_SIZE, PROCESSED_IMAGE_SIZE, 3),
    num_classes=num_classes,
    growth_rate=32
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001, epsilon=0.05),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
print(f"\nModel Parameters: {model.count_params():,}")
model.summary()


# Define callbacks
callbacks = [
    ModelCheckpoint(
        'densenet121_ocean_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

# Train the model
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

print("Training completed!")


# Plot training history
plt.figure(figsize=(15, 5))

# Plot training & validation accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot learning rate if available
plt.subplot(1, 3, 3)
if 'lr' in history.history:
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
else:
    plt.text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
             horizontalalignment='center', verticalalignment='center')
    plt.title('Learning Rate')

plt.tight_layout()
plt.show()


# Evaluate the model
print("\n=== Final Evaluation ===")
train_loss, train_accuracy = model.evaluate(train_generator, verbose=0)
val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Loss: {train_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")


# Visualize some predictions
def plot_predictions(generator, model, num_samples=8):
    """Plot sample predictions"""
    # Get a batch of data
    X_batch, y_batch = generator[0]
    
    # Make predictions
    predictions = model.predict(X_batch)
    
    # Create reverse label mapping
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    
    # Plot samples
    plt.figure(figsize=(16, 8))
    for i in range(min(num_samples, len(X_batch))):
        plt.subplot(2, 4, i + 1)
        
        # Denormalize image for display
        img = X_batch[i]
        if img.max() <= 1.0:  # If normalized
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        plt.imshow(img)
        
        # Get true and predicted labels
        true_label_idx = np.argmax(y_batch[i])
        pred_label_idx = np.argmax(predictions[i])
        
        true_label = reverse_label_mapping[true_label_idx]
        pred_label = reverse_label_mapping[pred_label_idx]
        confidence = predictions[i][pred_label_idx]
        
        # Set title with color coding
        color = 'green' if true_label_idx == pred_label_idx else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', y=1.02)
    plt.show()

# Show sample predictions
print("\n=== Sample Predictions ===")
plot_predictions(val_generator, model, num_samples=8)

