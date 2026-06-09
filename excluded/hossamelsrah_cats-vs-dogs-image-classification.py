!pip install split-folders


# Import essential libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import zipfile
from sklearn.model_selection import train_test_split
import splitfolders
import warnings

# Basic warnings filter
warnings.filterwarnings('ignore')


# Unzip dataset files (Kaggle stores them as zip files)
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/temp_train')
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/temp_test')

# Verify extraction
print("Train files:", len(os.listdir('/kaggle/working/temp_train/train')))
print("Test files:", len(os.listdir('/kaggle/working/temp_test/test1')))

# Create organized directory structure
os.makedirs('/kaggle/working/data/train/cats', exist_ok=True)
os.makedirs('/kaggle/working/data/train/dogs', exist_ok=True)
os.makedirs('/kaggle/working/data/val/cats', exist_ok=True)
os.makedirs('/kaggle/working/data/val/dogs', exist_ok=True)


# Display sample images function
def display_samples(folder, num_samples=5):
    plt.figure(figsize=(15, 3))
    plt.suptitle(f"Sample {folder.split('/')[-1]} Images", fontsize=16)
    
    # Get random samples
    samples = random.sample(os.listdir(folder), num_samples)
    
    for i, img_name in enumerate(samples):
        img_path = os.path.join(folder, img_name)
        img = mpimg.imread(img_path)
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display cat and dog samples
train_dir = '/kaggle/working/temp_train/train'
display_samples(train_dir)  # Will show mixed samples


# Organize files into class folders
for filename in os.listdir('/kaggle/working/temp_train/train'):
    src = os.path.join('/kaggle/working/temp_train/train', filename)
    
    if filename.startswith('cat'):
        dst = os.path.join('/kaggle/working/data/train/cats', filename)
    elif filename.startswith('dog'):
        dst = os.path.join('/kaggle/working/data/train/dogs', filename)
    
    os.rename(src, dst)

# Split into train/validation sets
splitfolders.ratio(
    '/kaggle/working/data/train',  # Source folder
    output='/kaggle/working/data/split',  # Output folder
    seed=42,
    ratio=(0.9, 0.1),  # Train/val ratio
    group_prefix=None
)


# Remove temporary files (optional)
!rm -r /kaggle/working/temp_train
!rm -r /kaggle/working/temp_test


# Display samples from the organized dataset
def display_organized_samples(folder, title, num_samples=5):
    plt.figure(figsize=(15, 3))
    plt.suptitle(title, fontsize=16)
    
    # Get random samples
    samples = random.sample(os.listdir(folder), num_samples)
    
    for i, img_name in enumerate(samples):
        img_path = os.path.join(folder, img_name)
        img = mpimg.imread(img_path)
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display samples from each category
print("\n" + "="*50)
print("Training Samples After Organization")
print("="*50)

# Cat samples
display_organized_samples(
    '/kaggle/working/data/split/train/cats',
    "Sample Training Cat Images"
)

# Dog samples
display_organized_samples(
    '/kaggle/working/data/split/train/dogs', 
    "Sample Training Dog Images"
)

# Validation samples
print("\n" + "="*50)
print("Validation Samples")
print("="*50)

# Validation cats
display_organized_samples(
    '/kaggle/working/data/split/val/cats',
    "Sample Validation Cat Images"
)

# Validation dogs
display_organized_samples(
    '/kaggle/working/data/split/val/dogs',
    "Sample Validation Dog Images"
)

# Show counts
print("\nFinal Counts:")
print(f"Training Cats: {len(os.listdir('/kaggle/working/data/split/train/cats'))}")
print(f"Training Dogs: {len(os.listdir('/kaggle/working/data/split/train/dogs'))}")
print(f"Validation Cats: {len(os.listdir('/kaggle/working/data/split/val/cats'))}")
print(f"Validation Dogs: {len(os.listdir('/kaggle/working/data/split/val/dogs'))}")


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Identify new paths
train_dir = '/kaggle/working/data/split/train'  
val_dir = '/kaggle/working/data/split/val'      

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='reflect'
)

# Validation data generator (only normalization)
val_datagen = ImageDataGenerator(rescale=1./255)

# Generate training batches from directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    classes=['cats', 'dogs'] 
)

# Generate validation batches from directory
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    classes=['cats', 'dogs'],
    shuffle=False
)

# Print a summary of the data
print("\nTraining Class Indices:", train_generator.class_indices)
print("Validation Class Indices:", val_generator.class_indices)


val_generator.class_indices


# Import Our Model Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2
import os

# Enhanced Callbacks Configuration
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('/kaggle/working/best_model.keras', save_best_only=True), 
    ReduceLROnPlateau(factor=0.5, patience=5)
]

# Create model 
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(150,150,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Classifier
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=L2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Load pre-trained weights if available 
os.makedirs('/kaggle/working/output', exist_ok=True)
if os.path.exists("/kaggle/working/output/weights.best.hdf5"):
    model.load_weights("/kaggle/working/output/weights.best.hdf5")
    print("Loaded pre-trained weights successfully!")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile model 
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy',
                      tf.keras.metrics.Precision(),
                      tf.keras.metrics.Recall()])
model.summary()


# Training Configuration
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Save Final Model
model.save('/kaggle/working/output/dogs_vs_cats_model.h5')
print("Training completed and model saved in ")
!ls -lh /kaggle/working/output/


# Evaluation
train_acc = model.evaluate(train_generator)[1]
valid_acc = model.evaluate(val_generator)[1]
print("Our New Model Accuracy on Training Data: ", train_acc)
print("Our New Model Accuracy on Validation Data: ", valid_acc)


import pandas as pd
import matplotlib.pyplot as plt

# Convert training history to DataFrame for easier manipulation
history_df = pd.DataFrame(history.history)

# Create figure with 2 subplots (side by side)
plt.figure(figsize=(12, 5))

# 1. Loss Subplot (Left)
plt.subplot(1, 2, 1)
plt.plot(history_df['loss'], 'b-', label='Training Loss')
plt.plot(history_df['val_loss'], 'r--', label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Training Epoch')
plt.ylabel('Loss Value')
plt.grid(True)
plt.legend()

# 2. Accuracy Subplot (Right)
plt.subplot(1, 2, 2)
plt.plot(history_df['accuracy'], 'g-', label='Training Accuracy')
plt.plot(history_df['val_accuracy'], 'm--', label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Training Epoch')
plt.ylabel('Accuracy Score')
plt.grid(True)
plt.ylim(0, 1)
plt.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


# Training Configuration With More Epochs
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Save Final Model 
model.save('/kaggle/working/output/dogs_vs_cats_model.h5')
print("Training completed and model saved in ")
!ls -lh /kaggle/working/output/


# Evaluation
train_acc = model.evaluate(train_generator)[1]
valid_acc = model.evaluate(val_generator)[1]
print("Our New Model Accuracy on Training Data: ", train_acc)
print("Our New Model Accuracy on Validation Data: ", valid_acc)


from sklearn.metrics import confusion_matrix, classification_report

# 1. Get model predictions
val_preds = model.predict(val_generator)
val_preds = (val_preds > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

# 2. Get true labels
val_true = val_generator.classes

# 3. Generate Confusion Matrix
cm = confusion_matrix(val_true, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cats', 'Dogs'],
            yticklabels=['Cats', 'Dogs'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 4. Generate Classification Report
print("\nDetailed Classification Report:")
print(classification_report(val_true, val_preds,
                          target_names=['Cats', 'Dogs']))

# 5. Calculate Overall Accuracy
val_acc = np.sum(val_preds == val_true) / len(val_true)
print(f"\nFinal Validation Accuracy: {val_acc:.4f}")


# Load Our Model
from tensorflow.keras.models import load_model
model = load_model('/kaggle/working/output/dogs_vs_cats_model.h5')


import requests
from PIL import Image

def predict_from_url(image_url):
    try:
        # Download The Image From URL
        response = requests.get(image_url, stream=True)
        img = Image.open(response.raw)

        # Processing
        img = img.convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predicition
        pred = model.predict(img_array)[0][0]
        confidence = pred if pred > 0.5 else 1 - pred
        return {
            'prediction': 'Dog' if pred > 0.5 else 'Cat',
            'confidence': float(confidence * 100)
        }
    
    except Exception as e:
        return {'error': str(e)}

# The Link URL
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZ0L7ciOGctocVYVpcLUIxcjI5Or3tj-e_jQ&s"  
result = predict_from_url(url)

# Show Result
if 'error' in result:
    print(f"Error: {result['error']}")
else:
    print(f"Predicition: {result['prediction']}")
    print(f"confidence: {result['confidence']:.2f}%")


from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# 1. Load pre-trained VGG16 model
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# 2. Partial freezing - unfreeze last 10 layers for fine-tuning
for layer in base_model.layers[:-10]:
    layer.trainable = False
for layer in base_model.layers[-10:]:
    layer.trainable = True
print("Last 10 layers of VGG16 unfrozen for fine-tuning")

# 3. Build enhanced model on top
VGG16_model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
print("Enhanced classification head with more layers and regularization")

# 4. Compile with different optimizer options
# Option 1: Adam with lower learning rate (good default)
# VGG16_model.compile(
#     optimizer=Adam(learning_rate=1e-5),
#     loss='binary_crossentropy',
#     metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
# )

# Option 2: RMSprop (often works well for fine-tuning)
VGG16_model.compile(
    optimizer=RMSprop(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Display model architecture
VGG16_model.summary()


# Declare Needed Callbacks Before Fitting
callbacks = [
    ModelCheckpoint(
        '/kaggle/working/output/vgg16_best_weights.weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

# Training
VGG16_history = VGG16_model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=val_generator,
    validation_steps=50
)

# Save our VGG16 Model
VGG16_model.save('/kaggle/working/output/vgg16_dogs_vs_cats_model.h5')
VGG16_model.save_weights('/kaggle/working/output/vgg16_dogs_vs_cats_weights.weights.h5')
print("Training completed and model saved in ")
!ls -lh /kaggle/working/output/


# Evaluation
train_acc = VGG16_model.evaluate(train_generator)[1]
valid_acc = VGG16_model.evaluate(val_generator)[1]
print("VGG16 Model Accuracy on Training Data: ", train_acc)
print("VGG16 Model Accuracy on Validation Data: ", valid_acc)


# Training
VGG16_history = VGG16_model.fit(
    train_generator,
    steps_per_epoch=200,
    epochs=100,
    validation_data=val_generator,
    validation_steps=50
)

# Save our VGG16 Model
VGG16_model.save('/kaggle/working/output/vgg16_dogs_vs_cats_model.h5')
VGG16_model.save_weights('/kaggle/working/output/vgg16_dogs_vs_cats_weights.weights.h5')
print("Training completed and model saved in ")
!ls -lh /kaggle/working/output/


# Evaluation
train_acc = VGG16_model.evaluate(train_generator)[1]
valid_acc = VGG16_model.evaluate(val_generator)[1]
print("VGG16 Model Accuracy on Training Data: ", train_acc)
print("VGG16 Model Accuracy on Validation Data: ", valid_acc)


# Convert training history to DataFrame for easier manipulation
VGG16_history_df = pd.DataFrame(VGG16_history.history)

# Create figure with 2 subplots (side by side)
plt.figure(figsize=(12, 5))

# 1. Loss Subplot (Left)
plt.subplot(1, 2, 1)
plt.plot(VGG16_history_df['loss'], 'b-', label='Training Loss')
plt.plot(VGG16_history_df['val_loss'], 'r--', label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Training Epoch')
plt.ylabel('Loss Value')
plt.legend()

# 2. Accuracy Subplot (Right)
plt.subplot(1, 2, 2)
plt.plot(VGG16_history_df['accuracy'], 'g-', label='Training Accuracy')
plt.plot(VGG16_history_df['val_accuracy'], 'm--', label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Training Epoch')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
plt.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


# 1. Get model predictions
val_preds = VGG16_model.predict(val_generator)
val_preds = (val_preds > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

# 2. Get true labels
val_true = val_generator.classes

# 3. Generate Confusion Matrix
cm = confusion_matrix(val_true, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cats', 'Dogs'],
            yticklabels=['Cats', 'Dogs'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 4. Generate Classification Report
print("\nDetailed Classification Report:")
print(classification_report(val_true, val_preds,
                          target_names=['Cats', 'Dogs']))

# 5. Calculate Overall Accuracy
val_acc = np.sum(val_preds == val_true) / len(val_true)
print(f"\nFinal Validation Accuracy for VGG16 model: {val_acc:.4f}")


# Extract the test zip file
test_zip_path = '/kaggle/input/dogs-vs-cats/test1.zip'
extract_path = '/kaggle/working/test_data'

# Create extraction directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Unzip the test files
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
    
print(f"Test files extracted to: {extract_path}")


# Verify the extracted files
test_files = os.listdir(os.path.join(extract_path, 'test1'))
print(f"\nNumber of test images: {len(test_files)}")
print("First 5 files:", test_files[:5])


# Prepare ImageDataGenerator for test images
test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

test_generator = test_datagen.flow_from_directory(
    extract_path,               
    target_size=(150, 150),     
    batch_size=32,             
    class_mode=None,            
    shuffle=False               
)


# Make predictions on test data
print("\nRunning predictions on test data...")
predictions = VGG16_model.predict(test_generator)

# Convert probability predictions to labels
predicted_labels = ['dog' if pred > 0.5 else 'cat' for pred in predictions]

# Display random prediction samples
sample_indices = random.sample(range(len(predicted_labels)), 5)

print("\nRandom prediction samples:")
for idx in sample_indices:
    confidence = max(predictions[idx][0], 1-predictions[idx][0]) * 100
    print(f"Image: {test_generator.filenames[idx]} - Prediction: {predicted_labels[idx]} (Confidence: {confidence:.1f}%)")


# Save predictions for competition submission
submission = pd.DataFrame({
    'id': [os.path.splitext(f)[0] for f in test_generator.filenames],  # Remove file extension
    'label': [1 if pred > 0.5 else 0 for pred in predictions]  # 1=dog, 0=cat
})

submission_path = '/kaggle/working/output/test_predictions.csv'
submission.to_csv(submission_path, index=False)
print(f"\nPredictions saved to: {submission_path}")

