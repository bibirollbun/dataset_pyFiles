# The code is used for training on kaggle the input is cassava disease classification
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        # Check if the file extension is not '.jpg'
        if not filename.lower().endswith('.jpg'):
            print(os.path.join(dirname, filename))


import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define paths
PATHS = {
    'TRAIN_CSV': '/kaggle/input/cassava-leaf-disease-classification/train.csv',
    'TEST_CSV': '/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv',
    'DISEASE_MAP': '/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json',
    'TRAIN_IMAGES': '/kaggle/input/cassava-leaf-disease-classification/train_images',
    'TEST_IMAGES': '/kaggle/input/cassava-leaf-disease-classification/test_images',
    'OUTPUT': '/kaggle/working/submission.csv',
    'MODEL_CACHE': '/kaggle/working/model_cache',
    'WEIGHTS': '/kaggle/working/weights',
    'PLOTS': '/kaggle/working/plots',
    'SAVED_MODEL': '/kaggle/working/cassava_disease_model_tf'
}

# Create necessary directories
for directory in ['MODEL_CACHE', 'WEIGHTS', 'PLOTS', 'SAVED_MODEL']:
    os.makedirs(PATHS[directory], exist_ok=True)

# Load disease mapping
with open(PATHS['DISEASE_MAP'], 'r') as f:
    disease_map = json.load(f)
    
# Convert from string keys to integer keys
disease_map = {int(k): v for k, v in disease_map.items()}
num_classes = len(disease_map)
print(f"Number of classes: {num_classes}")
print("Disease mapping:", disease_map)

# Load training data
train_df = pd.read_csv(PATHS['TRAIN_CSV'])
print(f"Training data shape: {train_df.shape}")
print(train_df.head())

# Check class distribution
class_distribution = train_df['label'].value_counts().sort_index()
print("Class distribution:")
for class_id, count in class_distribution.items():
    print(f"Class {class_id} ({disease_map[class_id]}): {count} images")

# Split data into train and validation sets
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)
print(f"Training set: {train_df.shape[0]} images")
print(f"Validation set: {val_df.shape[0]} images")

# Data augmentation for training - slightly reduced parameters for more stability
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,          # Reduced from 20
    width_shift_range=0.15,     # Reduced from 0.2
    height_shift_range=0.15,    # Reduced from 0.2
    shear_range=0.15,           # Reduced from 0.2
    zoom_range=0.15,            # Reduced from 0.2
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Image dimensions
img_height, img_width = 224, 224
batch_size = 32

# Convert label integers to strings to work with categorical mode
train_df['label_str'] = train_df['label'].astype(str)
val_df['label_str'] = val_df['label'].astype(str)

# Create data generators with explicit shuffle setting
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=PATHS['TRAIN_IMAGES'],
    x_col='image_id',
    y_col='label_str',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True  # Explicit setting
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=PATHS['TRAIN_IMAGES'],
    x_col='image_id',
    y_col='label_str',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # No shuffling for validation
)

# Verify shuffling is working
def verify_shuffling(generator, num_batches=2):
    """Check if a generator is actually shuffling data between epochs"""
    # Get first batch indices from first epoch
    batch_indices_epoch1 = []
    for i in range(num_batches):
        batch_x, _ = next(generator)
        # Store some sample data from this batch to identify it
        batch_indices_epoch1.append(batch_x[0, 0, 0, 0])  # Use first pixel value as identifier
    
    # Reset the generator to simulate a new epoch
    generator.reset()
    
    # Get first batch indices from second "epoch"
    batch_indices_epoch2 = []
    for i in range(num_batches):
        batch_x, _ = next(generator)
        batch_indices_epoch2.append(batch_x[0, 0, 0, 0])
    
    # Compare the batches
    different_batches = sum(abs(epoch1 - epoch2) > 1e-5 
                           for epoch1, epoch2 in zip(batch_indices_epoch1, batch_indices_epoch2))
    
    print(f"Shuffling verification: {different_batches}/{num_batches} batches were different between epochs")
    print(f"First epoch batch identifiers: {batch_indices_epoch1}")
    print(f"Second epoch batch identifiers: {batch_indices_epoch2}")
    
    # Reset the generator again for actual training
    generator.reset()
    return different_batches > 0

# Run the verification
is_shuffling = verify_shuffling(train_generator)
print(f"Training data is being shuffled: {is_shuffling}")

# Validate dataset splits
def validate_dataset_splits(train_gen, val_gen):
    """Validate that the dataset splits have reasonable sizes and class distributions"""
    # Check total samples
    print("\n=== Dataset Split Validation ===")
    print(f"Total training samples: {train_gen.n}")
    print(f"Total validation samples: {val_gen.n}")
    
    # Ensure reasonable split ratio
    total_samples = train_gen.n + val_gen.n
    train_ratio = train_gen.n / total_samples
    val_ratio = val_gen.n / total_samples
    
    print(f"Training split: {train_ratio:.2%}")
    print(f"Validation split: {val_ratio:.2%}")
    
    # Warn if validation set is too small or too large
    if val_gen.n < 100:
        print("WARNING: Validation set may be too small (<100 samples)")
    if val_ratio < 0.1:
        print("WARNING: Validation set may be too small (<10% of data)")
    if val_ratio > 0.3:
        print("WARNING: Validation set may be too large (>30% of data)")
    
    # Check class distribution in both splits
    print("\n--- Class Distribution ---")
    class_counts_train = train_df['label'].value_counts().sort_index()
    class_counts_val = val_df['label'].value_counts().sort_index()
    
    print("Class distribution in training set:")
    for class_id in sorted(class_counts_train.index):
        count = class_counts_train.get(class_id, 0)
        percentage = count / train_gen.n * 100
        class_name = disease_map.get(class_id, f"Class {class_id}")
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    print("\nClass distribution in validation set:")
    for class_id in sorted(class_counts_val.index):
        count = class_counts_val.get(class_id, 0)
        percentage = count / val_gen.n * 100
        class_name = disease_map.get(class_id, f"Class {class_id}")
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Check if any class has very few samples
    min_samples_warning = 50  # Arbitrary threshold
    for class_id in sorted(class_counts_val.index):
        if class_counts_val.get(class_id, 0) < min_samples_warning:
            print(f"WARNING: Class {class_id} has fewer than {min_samples_warning} samples in validation set")
    
    print("=== End of Dataset Validation ===\n")

# Run the validation
validate_dataset_splits(train_generator, validation_generator)

# Calculate steps properly - IMPORTANT FIX
steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = validation_generator.n // validation_generator.batch_size

# Print steps information
print(f"Training generator has {train_generator.n} samples with batch size {train_generator.batch_size}")
print(f"Using {steps_per_epoch} steps per epoch for training")
print(f"Validation generator has {validation_generator.n} samples with batch size {validation_generator.batch_size}")
print(f"Using {validation_steps} steps per epoch for validation")

# Load the pretrained model
cropnet_path = "/kaggle/input/cropnet/tensorflow1/classifier-cassava-disease-v1/1"

# Load the base model using TFSMLayer
base_model_layer = tf.keras.layers.TFSMLayer(
    cropnet_path,
    call_endpoint='default'
)

# Examine the model's input/output signature
loaded = tf.saved_model.load(cropnet_path)
print("Model signature info:", loaded.signatures['default'])

# Create a new model with the pretrained base
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
base_outputs = base_model_layer(inputs)

# Print the output type and content to understand its structure
print(f"Base model output type: {type(base_outputs)}")
print(f"Base model output keys: {base_outputs.keys() if isinstance(base_outputs, dict) else 'Not a dictionary'}")

# Extract the appropriate tensor from the dictionary
if isinstance(base_outputs, dict):
    # Try to find the most likely output tensor from the dictionary
    output_key = list(base_outputs.keys())[0]
    print(f"Using output key: {output_key}")
    x = base_outputs[output_key]
else:
    x = base_outputs

print(f"Selected output shape: {x.shape}")

# Add new classification head
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Freeze the base model initially
base_model_layer.trainable = False

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Function to validate label configurations
def validate_label_configuration(train_gen, model):
    """Verify that the class_mode and model output layer are compatible"""
    print("\n=== Label Configuration Validation ===")
    
    # Check class_mode setting
    class_mode = train_gen.class_mode
    print(f"Generator class_mode: {class_mode}")
    
    # Get the model's output layer
    output_layer = model.layers[-1]
    
    # Get output shape correctly - this is the fix
    output_shape = model.output_shape
    
    # Get activation function
    output_activation = output_layer.activation.__name__ if hasattr(output_layer.activation, '__name__') else 'unknown'
    
    print(f"Model output layer shape: {output_shape}")
    print(f"Model output activation function: {output_activation}")
    
    # Validate compatibility
    is_compatible = True
    error_message = None
    
    if class_mode == 'categorical':
        if output_activation != 'softmax':
            is_compatible = False
            error_message = "Using 'categorical' class_mode but output layer activation is not 'softmax'"
        if output_shape[-1] != num_classes:
            is_compatible = False
            error_message = f"Output layer has {output_shape[-1]} units but there are {num_classes} classes"
    elif class_mode == 'binary':
        if output_activation != 'sigmoid':
            is_compatible = False
            error_message = "Using 'binary' class_mode but output layer activation is not 'sigmoid'"
        if output_shape[-1] != 1:
            is_compatible = False
            error_message = f"Binary classification should have 1 output unit, but found {output_shape[-1]}"
    elif class_mode == 'sparse':
        if output_activation != 'softmax':
            is_compatible = False
            error_message = "Using 'sparse' class_mode but need 'softmax' activation for multi-class"
    
    # Print validation results
    if is_compatible:
        print("✓ Class mode and model output layer are compatible")
    else:
        print(f"⚠ CONFIGURATION ERROR: {error_message}")
        print("This will likely cause training issues!")
    
    # Verify label encoding by checking a sample batch
    batch_x, batch_y = next(train_gen)
    print(f"\nSample batch shape - X: {batch_x.shape}, Y: {batch_y.shape}")
    
    # For categorical, we expect one-hot encoding (shape should be (batch_size, num_classes))
    if class_mode == 'categorical':
        if batch_y.shape[1] != num_classes:
            print(f"⚠ LABEL ERROR: Expected y shape to be (batch_size, {num_classes}) but got {batch_y.shape}")
        else:
            print(f"✓ Labels are correctly one-hot encoded with {num_classes} classes")
    
    # Reset the generator
    train_gen.reset()
    print("=== End of Label Configuration Validation ===\n")

# Run label validation
validate_label_configuration(train_generator, model)

# Define callbacks
checkpoint = ModelCheckpoint(
    os.path.join(PATHS['WEIGHTS'], 'best_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Train the model with frozen base - using corrected steps_per_epoch
history_frozen = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Use calculated value
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps,  # Use calculated value
    callbacks=callbacks
)

# Unfreeze the base model for fine-tuning
base_model_layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
# IMPROVEMENT: Reduced learning rate to prevent overfitting
model.compile(
    optimizer=optimizers.Adam(learning_rate=5e-6),  # Reduced from 1e-5 to 5e-6
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training with unfrozen base - using corrected steps_per_epoch
# IMPROVEMENT: Reduced number of fine-tuning epochs
history_unfrozen = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Use calculated value
    epochs=20,  # Reduced from 30 to 20
    validation_data=validation_generator,
    validation_steps=validation_steps,  # Use calculated value
    callbacks=callbacks,
    initial_epoch=history_frozen.epoch[-1] + 1  # Continue from where we left off
)

# Improved function to plot training history
def plot_improved_training_history(history_frozen, history_unfrozen=None):
    """Plot training history with properly combined epochs for frozen and unfrozen training"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Merge histories for continuous plotting
    merged_history = {}
    
    # Start with frozen history metrics
    for metric in history_frozen.history:
        merged_history[metric] = list(history_frozen.history[metric])
    
    # Append unfrozen history if available
    if history_unfrozen:
        for metric in history_unfrozen.history:
            if metric in merged_history:
                merged_history[metric].extend(history_unfrozen.history[metric])
            else:
                # Handle case where a metric might only exist in one history
                merged_history[metric] = list(history_unfrozen.history[metric])
    
    # Create a single x-axis for all epochs
    epochs = range(1, len(merged_history.get('accuracy', [])) + 1)
    
    # Add a vertical line to mark the transition from frozen to unfrozen
    frozen_epochs = len(history_frozen.history.get('accuracy', []))
    
    # Plot accuracy
    ax1.plot(epochs, merged_history.get('accuracy', []), 'b-', label='Training Accuracy')
    if 'val_accuracy' in merged_history:
        ax1.plot(epochs, merged_history.get('val_accuracy', []), 'r-', label='Validation Accuracy')
    
    # Add transition line for accuracy plot
    if history_unfrozen:
        ax1.axvline(x=frozen_epochs, color='g', linestyle='--', 
                   label='Transition: Frozen → Unfrozen')
    
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot loss
    ax2.plot(epochs, merged_history.get('loss', []), 'b-', label='Training Loss')
    if 'val_loss' in merged_history:
        ax2.plot(epochs, merged_history.get('val_loss', []), 'r-', label='Validation Loss')
    
    # Add transition line for loss plot
    if history_unfrozen:
        ax2.axvline(x=frozen_epochs, color='g', linestyle='--',
                   label='Transition: Frozen → Unfrozen')
    
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add a text annotation to explain the phases
    if history_unfrozen:
        plt.figtext(0.5, 0.01, 
                   f"Phase 1 (Epochs 1-{frozen_epochs}): Base model frozen | "
                   f"Phase 2 (Epochs {frozen_epochs+1}-{len(epochs)}): Full model fine-tuning",
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the text
    
    # Save figure
    plt.savefig(os.path.join(PATHS['PLOTS'], 'improved_training_history.png'), dpi=300)
    plt.show()
    
    # Print summary statistics
    print("\n=== Training History Summary ===")
    
    # Initial phase stats
    print(f"Phase 1 (Frozen base model) - {frozen_epochs} epochs:")
    print(f"  Starting train accuracy: {history_frozen.history['accuracy'][0]:.4f}")
    print(f"  Final train accuracy: {history_frozen.history['accuracy'][-1]:.4f}")
    
    if 'val_accuracy' in history_frozen.history:
        print(f"  Starting validation accuracy: {history_frozen.history['val_accuracy'][0]:.4f}")
        print(f"  Final validation accuracy: {history_frozen.history['val_accuracy'][-1]:.4f}")
    
    # Fine-tuning phase stats
    if history_unfrozen:
        unfrozen_epochs = len(history_unfrozen.history['accuracy'])
        print(f"\nPhase 2 (Fine-tuning) - {unfrozen_epochs} epochs:")
        print(f"  Starting train accuracy: {history_unfrozen.history['accuracy'][0]:.4f}")
        print(f"  Final train accuracy: {history_unfrozen.history['accuracy'][-1]:.4f}")
        
        if 'val_accuracy' in history_unfrozen.history:
            print(f"  Starting validation accuracy: {history_unfrozen.history['val_accuracy'][0]:.4f}")
            print(f"  Final validation accuracy: {history_unfrozen.history['val_accuracy'][-1]:.4f}")
        
        # Improvement calculation
        acc_improvement = history_unfrozen.history['accuracy'][-1] - history_frozen.history['accuracy'][-1]
        print(f"\nImprovement from fine-tuning: {acc_improvement:.4f} accuracy")
        
        if 'val_accuracy' in history_unfrozen.history and 'val_accuracy' in history_frozen.history:
            val_acc_improvement = history_unfrozen.history['val_accuracy'][-1] - history_frozen.history['val_accuracy'][-1]
            print(f"Validation accuracy improvement: {val_acc_improvement:.4f}")
    
    print("=== End of Training History Summary ===\n")

# Use the improved plotting function
plot_improved_training_history(history_frozen, history_unfrozen)

# Export model as SavedModel (TF2 format)
@tf.function(input_signature=[tf.TensorSpec(shape=[None, img_height, img_width, 3], dtype=tf.float32, name='input_image')])
def serving_fn(input_image):
    return {'predictions': model(input_image, training=False)}

# Save the model in SavedModel format
tf.saved_model.save(
    model,
    PATHS['SAVED_MODEL'],
    signatures={'serving_default': serving_fn}
)

print(f"Model saved to {PATHS['SAVED_MODEL']} in SavedModel format")

# Create a zip file of the SavedModel directory for easy download
import shutil
shutil.make_archive(
    os.path.join('/kaggle/working', 'cassava_disease_model'),  # output name
    'zip',                                                     # format
    PATHS['SAVED_MODEL']                                      # source directory
)

print(f"SavedModel zipped to /kaggle/working/cassava_disease_model.zip")

# Load test data
test_df = pd.read_csv(PATHS['TEST_CSV'])
print(f"Test data shape: {test_df.shape}")

# Create test generator (only rescaling, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=PATHS['TEST_IMAGES'],
    x_col='image_id',
    y_col=None,  # No labels for test data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,  # No labels
    shuffle=False  # Keep the order for submission
)

# Calculate steps for test predictions - CRITICAL FIX: Convert to integer
test_steps = int(np.ceil(test_generator.n / test_generator.batch_size))
print(f"Test generator has {test_generator.n} samples")
print(f"Using {test_steps} steps for prediction")

# Predict on test data
predictions = model.predict(test_generator, steps=test_steps)
predicted_classes = np.argmax(predictions, axis=1)

# Create submission file
test_df['label'] = predicted_classes
test_df.to_csv(PATHS['OUTPUT'], index=False)
print(f"Submission file saved to {PATHS['OUTPUT']}")

# Print final model summary
model.summary()

# Verify the saved model can be loaded
print("\nVerifying SavedModel by loading it and making a test prediction...")
try:
    # Load the model
    loaded_model = tf.saved_model.load(PATHS['SAVED_MODEL'])
    
    # Get the serving signature
    serving_signature = loaded_model.signatures['serving_default']
    print(f"Model loaded successfully with signature: {serving_signature}")
    
    # Create a sample input (random tensor with correct shape)
    sample_input = np.random.random((1, img_height, img_width, 3)).astype(np.float32)
    
    # Make a prediction
    test_prediction = serving_signature(tf.constant(sample_input))
    print(f"Test prediction shape: {list(test_prediction.values())[0].shape}")
    print("SavedModel verification successful!")
except Exception as e:
    print(f"Error verifying SavedModel: {e}")

