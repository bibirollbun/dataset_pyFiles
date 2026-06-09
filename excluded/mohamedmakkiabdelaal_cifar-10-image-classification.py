import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, UpSampling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import optimizers
import pickle


def load_cifar10_batch(file_path):
    """
    Load a single CIFAR-10 batch file
    
    Parameters:
    file_path (str): Path to the batch file
    
    Returns:
    tuple: (images, labels) arrays
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    # Extract images and labels
    images = batch[b'data']
    labels = batch[b'labels']
    
    # Reshape images from flat array to 32x32x3
    images = images.reshape(-1, 3, 32, 32)
    # Convert from channels-first to channels-last format
    images = images.transpose(0, 2, 3, 1)
    
    return images, labels


def extract_7z_files():
    """
    Extract 7z files if they exist and load CIFAR-10 data
    
    Returns:
    tuple: (X_train, y_train, X_test, y_test) or None if failed
    """
    print(f"\nLOADING CIFAR-10 FROM KAGGLE")
    print("-" * 30)
    
    try:
        # Check available files
        available_files = os.listdir(KAGGLE_INPUT_PATH)
        print(f"Available files: {available_files}")
        
        # Try to use TensorFlow's built-in CIFAR-10 dataset as fallback
        # This is more reliable on Kaggle
        print("Loading CIFAR-10 using TensorFlow datasets...")
        
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Flatten labels (they come as [[0], [1], ...] format)
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        print(f"Training data loaded: {X_train.shape}")
        print(f"Training labels: {y_train.shape}")
        print(f"Test data loaded: {X_test.shape}")
        print(f"Test labels: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        return None, None, None, None

def load_and_explore_labels():
    """
    Load and explore the CIFAR-10 labels using standard mapping
    
    Returns:
    dict: Label dictionary mapping names to indices
    """
    print(f"\nSETTING UP CIFAR-10 LABELS")
    print("-" * 30)
    
    # CIFAR-10 standard label mapping
    labels_dict = {
        'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
    }
    
    # Reverse mapping for getting names from indices
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"CIFAR-10 class mapping:")
    for name, idx in labels_dict.items():
        print(f"   - {idx}: {name}")
    
    return labels_dict, class_names


KAGGLE_INPUT_PATH = '/kaggle/input/cifar-10'


print(f"ğŸ”� Environment: Kaggle")
print(f"ğŸ”� Input path: {KAGGLE_INPUT_PATH}")


X_train, y_train, X_test, y_test = extract_7z_files()

if X_train is not None:
    print(f"\nCIFAR-10 DATA LOADED SUCCESSFULLY")
    print("-" * 30)
    print(f"Training images: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test images: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")
    
    # Load label mappings
    labels_dict, class_names = load_and_explore_labels()
    
    # Show data distribution
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print(f"\nTraining data distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"   - {class_names[label]}: {count} images")
        
else:
    print(f"\nCREATING SAMPLE DATA")
    print("-" * 30)
    print("Using synthetic CIFAR-10 like data for demonstration...")
    
    # Create sample data that matches CIFAR-10 structure
    X_train = np.random.randint(0, 255, (10000, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.randint(0, 10, 10000)
    X_test = np.random.randint(0, 255, (2000, 32, 32, 3), dtype=np.uint8)
    y_test = np.random.randint(0, 10, 2000)
    
    labels_dict, class_names = load_and_explore_labels()
    print(f"Sample data created: Train {X_train.shape}, Test {X_test.shape}")



print(f"\nDATA PREPROCESSING")
print("-" * 30)

# Ù„Ù„ØªØ¯Ø±ÙŠØ¨ Ø§Ù„Ø³Ø±ÙŠØ¹ Ø¹Ù„Ù‰ KaggleØŒ Ø§Ø³ØªØ®Ø¯Ù… Ø¹ÙŠÙ†Ø© Ù…Ù† Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
SAMPLE_SIZE = 10000  # ÙŠÙ…ÙƒÙ†Ùƒ ØªØºÙŠÙŠØ± Ù‡Ø°Ø§ Ø§Ù„Ø±Ù‚Ù… Ø­Ø³Ø¨ Ø§Ù„Ø­Ø§Ø¬Ø©
if len(X_train) > SAMPLE_SIZE:
    # Ø£Ø®Ø° Ø¹ÙŠÙ†Ø© Ø¹Ø´ÙˆØ§Ø¦ÙŠØ© Ù…ØªÙˆØ§Ø²Ù†Ø©
    indices = []
    for class_idx in range(10):
        class_indices = np.where(y_train == class_idx)[0]
        selected_indices = np.random.choice(class_indices, min(SAMPLE_SIZE//10, len(class_indices)), replace=False)
        indices.extend(selected_indices)
    
    indices = np.array(indices)
    np.random.shuffle(indices)
    
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print(f"Using sample of {len(X_train)} training images for faster training")


X_train_scaled = X_train.astype('float32') / 255.0
X_test_scaled = X_test.astype('float32') / 255.0


def create_transfer_learning_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Create transfer learning model using ResNet50
    
    Parameters:
    input_shape (tuple): Shape of input images
    num_classes (int): Number of output classes
    
    Returns:
    keras.Sequential: Compiled transfer learning model
    """
    print(f"\nBUILDING TRANSFER LEARNING MODEL")
    print("-" * 30)
    
    # Load pre-trained ResNet50 (without top layers)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)  # ResNet50 requires larger input
    )
    
    print(f"ResNet50 base model loaded")
    print(f"Base model parameters: {base_model.count_params():,}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build complete model
    model = models.Sequential([
        # Upsampling layers to match ResNet50 input requirements
        UpSampling2D((2, 2), input_shape=input_shape),  # 32x32 -> 64x64
        UpSampling2D((2, 2)),                          # 64x64 -> 128x128
        UpSampling2D((2, 2)),                          # 128x128 -> 256x256
        
        # Pre-trained ResNet50 base
        base_model,
        
        # Custom classification head
        Flatten(),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with lower learning rate for transfer learning
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=2e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Transfer learning model created")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {np.sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights]):,}")
    
    return model


model = create_transfer_learning_model()

print(f"\nTRAINING TRANSFER LEARNING MODEL")
print("-" * 30)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=10,
    validation_split=0.1,
    batch_size=32,
    verbose=1
)

print(f"Transfer learning model training completed")


test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Get training history
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")


# Create training plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_title('Model Loss Progress', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax2.set_title('Model Accuracy Progress', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

# Get predictions for first 10 test images
sample_predictions = model.predict(X_test_scaled[:10], verbose=0)
predicted_classes = np.argmax(sample_predictions, axis=1)
confidences = np.max(sample_predictions, axis=1)

for i in range(10):
    axes[i].imshow(X_test[i])
    axes[i].set_title(
        f'True: {class_names[y_test[i]]}\n'
        f'Pred: {class_names[predicted_classes[i]]}\n'
        f'Conf: {confidences[i]:.2f}',
        fontsize=10
    )
    axes[i].axis('off')
    
    # Add border color based on correctness
    if predicted_classes[i] == y_test[i]:
        axes[i].add_patch(plt.Rectangle((0, 0), 32, 32, fill=False, edgecolor='green', lw=3))
    else:
        axes[i].add_patch(plt.Rectangle((0, 0), 32, 32, fill=False, edgecolor='red', lw=3))

plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


def predict_cifar10_class(model, image_array, class_names):
    """
    Predict CIFAR-10 class for a single image
    
    Parameters:
    model: Trained transfer learning model
    image_array: Input image as numpy array (32x32x3)
    class_names: List of class names
    
    Returns:
    tuple: Prediction result and confidence
    """
    try:
        # Preprocess the image
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Normalize
        image_processed = image_array.astype('float32') / 255.0
        
        # Make prediction
        predictions = model.predict(image_processed, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class_name = class_names[predicted_class]
        
        return predicted_class, predicted_class_name, confidence
        
    except Exception as e:
        return None, f"Error: {e}", 0.0


if len(X_test) > 0:
    sample_idx = 0
    sample_image = X_test[sample_idx]
    actual_class = y_test[sample_idx]
    
    pred_class, pred_name, confidence = predict_cifar10_class(model, sample_image, class_names)
    actual_name = class_names[actual_class]
    
    print(f"Sample Prediction Results:")
    print(f"  Actual class: {actual_class} ({actual_name})")
    print(f"  Predicted class: {pred_class} ({pred_name})")
    print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"  Result: {'Correct' if pred_class == actual_class else 'Incorrect'}")


print(f"\nMODEL ARCHITECTURE SUMMARY")
print("-" * 30)
model.summary()

# Calculate accuracy per class
y_pred_all = model.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred_all, axis=1)

print(f"\nPER-CLASS ACCURACY")
print("-" * 30)
for i, class_name in enumerate(class_names):
    class_mask = y_test == i
    if np.sum(class_mask) > 0:
        class_accuracy = np.mean(y_pred_classes[class_mask] == y_test[class_mask])
        print(f"  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

print(f"\nCIFAR-10 Classification System Ready!")
print(f"Model can classify images into 10 categories")
print(f"Classes: {', '.join(class_names)}")

