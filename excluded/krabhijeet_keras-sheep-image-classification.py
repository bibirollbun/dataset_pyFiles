import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import keras

import warnings
warnings.filterwarnings("ignore")


# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 7
TTA_STEPS = 5  # Number of augmentations per test image


# Paths
TRAIN_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'
TEST_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test'
TRAIN_CSV = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'


# Load and prepare data
train_df = pd.read_csv(TRAIN_CSV)
print(f"Found {len(train_df)} training images")
print("Class distribution:\n", train_df['label'].value_counts())

# Stratified split of the dataframe
train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['label'],  # This ensures stratified sampling
    random_state=42
)


# Create data generators with separate dataframes for train and validation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=TRAIN_DIR,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory=TRAIN_DIR,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Verify the class distribution in both sets
print("\nTraining set class distribution:")
print(train_data['label'].value_counts(normalize=True))
print("\nValidation set class distribution:")
print(val_data['label'].value_counts(normalize=True)) 


# Create model (ConvNeXtXLarge)
def create_model():
    base_model = applications.ConvNeXtXLarge(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = False  # Freeze base model initially
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
#        layers.Dropout(0.2), # Regularize with dropout
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = create_model()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

model.summary()


# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping, checkpoint]
)


model.load_weights("best_model.h5")  # ensure best weights are loaded


def plot_training_curves_seaborn(history):
    # Prepare data
    history_dict = history.history
    epochs = range(1, len(history_dict['accuracy']) + 1)
    
    df = pd.DataFrame({
        'epoch': list(epochs) * 4,
        'value': history_dict['accuracy'] + history_dict['val_accuracy'] + history_dict['loss'] + history_dict['val_loss'],
        'metric': ['train_acc'] * len(epochs) + ['val_acc'] * len(epochs) + ['train_loss'] * len(epochs) + ['val_loss'] * len(epochs)
    })
    
    # Plot accuracy
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    sns.lineplot(data=df[df.metric.isin(['train_acc', 'val_acc'])], x='epoch', y='value', hue='metric')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Plot loss
    plt.subplot(1,2,2)
    sns.lineplot(data=df[df.metric.isin(['train_loss', 'val_loss'])], x='epoch', y='value', hue='metric')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.show()

# Usage:
plot_training_curves_seaborn(history)


# Prepare test data
test_files = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({'filename': test_files})

# Create TTA generator - same augmentations as training but for inference
tta_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

test_generator = tta_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TEST_DIR,
    x_col='filename',
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)

# Verify test generator found images
print(f"\nFound {test_generator.samples} test images")



# Make predictions with TTA
if test_generator.samples > 0:
    # Initialize array to store predictions
    all_predictions = []
    
    print(f"\nPerforming Test Time Augmentation with {TTA_STEPS} steps...")
    for i in range(TTA_STEPS):
        print(f"TTA step {i+1}/{TTA_STEPS}")
        # Reset generator to avoid randomness between epochs
        test_generator.reset()
        preds = model.predict(test_generator, verbose=1)
        all_predictions.append(preds)
    
    # Average predictions across all TTA steps
    avg_predictions = np.mean(all_predictions, axis=0)
    predicted_classes = tf.argmax(avg_predictions, axis=1)
    class_names = list(train_generator.class_indices.keys())
    predicted_labels = [class_names[i] for i in predicted_classes]
    
    # Prepare submission
    submission = pd.DataFrame({
        'filename': test_files,
        'label': predicted_labels
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file with TTA created successfully!")
else:
    print("Error: No test images found. Check your TEST_DIR path.")

