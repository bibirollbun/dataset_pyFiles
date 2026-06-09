import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Add, Layer
from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, Conv2D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = (224, 224)  # ViT input size
BATCH_SIZE = 16  # Reduced batch size for transformer model
EPOCHS = 50  # Increased from 1 to get meaningful training
BASE_DIR = '/kaggle/input/plant-seedlings-classification'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
OUTPUT_DIR = '/kaggle/working'

# Create necessary directories to avoid permission errors
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define directories for storing split data
SPLIT_DATA_DIR = os.path.join(OUTPUT_DIR, 'split_data')
if not os.path.exists(SPLIT_DATA_DIR):
    os.makedirs(SPLIT_DATA_DIR)
    os.makedirs(os.path.join(SPLIT_DATA_DIR, 'train'))
    os.makedirs(os.path.join(SPLIT_DATA_DIR, 'validation'))

# Function to analyze class distribution
def analyze_class_distribution(data_dir):
    """Analyze the class distribution in the dataset"""
    class_counts = {}
    classes = []
    
    # Get all classes (subdirectories)
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            classes.append(class_name)
            n_samples = len(os.listdir(class_path))
            class_counts[class_name] = n_samples
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Plant Species')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
    plt.close()
    
    return class_counts, classes

# Create a DataFrame for all images
def create_dataframe(data_dir):
    """Create a DataFrame with file paths and labels"""
    image_paths = []
    labels = []
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_path, img_name))
                    labels.append(class_name)
    
    return pd.DataFrame({'image_path': image_paths, 'class': labels})

# Function to create data generators
def create_data_generators(train_df, val_df, class_weights=None):
    """Create data generators for training and validation"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

# Function to calculate class weights to handle class imbalance
def calculate_class_weights(class_counts):
    """Calculate class weights to handle class imbalance"""
    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)
    class_weights = {i: total_samples / (n_classes * count) for i, count in enumerate(class_counts.values())}
    return class_weights

# Vision Transformer components
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation="gelu")(x)
        x = Dropout(dropout_rate)(x)
    return x

def create_patches(images, patch_size):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches

# Create a custom layer that extracts patches
class PatchExtractor(Layer):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size
        
    def call(self, images):
        return create_patches(images, self.patch_size)

# Custom layer to create a class token
class AddClassToken(Layer):
    def __init__(self, embed_dim):
        super(AddClassToken, self).__init__()
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.class_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True,
            name="class_token"
        )
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_tokens = tf.repeat(self.class_token, repeats=batch_size, axis=0)
        return tf.concat([class_tokens, inputs], axis=1)

# Create and compile the model
def create_model(n_classes):
    """Create a Vision Transformer model"""
    patch_size = 16  # Size of the patches to be extracted from the input images
    projection_dim = 64  # Dimension for patch projection
    num_patches = (IMG_SIZE[0] // patch_size) * (IMG_SIZE[1] // patch_size)  # Number of patches
    transformer_layers = 8  # Number of transformer layers
    num_heads = 4  # Number of attention heads
    transformer_units = [projection_dim * 2, projection_dim]  # MLP units
    mlp_head_units = [2048, 1024]  # MLP head units
    
    # Input shape (224, 224, 3)
    inputs = Input(shape=(*IMG_SIZE, 3))
    
    # Encode patches using Conv2D
    x = Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size, padding="VALID")(inputs)
    x = Reshape((-1, projection_dim))(x)
    
    # Add positional embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + position_embedding
    
    # Add class token
    x = AddClassToken(projection_dim)(x)
    
    # Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        y = LayerNormalization(epsilon=1e-6)(x)
        
        # Multi-head attention
        y = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads
        )(y, y)
        
        # Skip connection 1
        x = Add()([x, y])
        
        # Layer normalization 2
        y = LayerNormalization(epsilon=1e-6)(x)
        
        # MLP
        y = mlp(y, transformer_units, 0.1)
        
        # Skip connection 2
        x = Add()([x, y])
    
    # Get class token output
    x = LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0]  # Take the first token (class token)
    
    # MLP head
    x = mlp(x, mlp_head_units, 0.1)
    
    # Classification head
    outputs = Dense(n_classes, activation="softmax")(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model

# Define callbacks
def get_callbacks():
    """Define callbacks for training"""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    return [early_stopping, checkpoint, reduce_lr]

# Function to plot training history
def plot_history(history):
    """Plot the training and validation metrics"""
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()

# Function to evaluate the model
def evaluate_model(model, val_generator, classes):
    """Evaluate the model and generate performance metrics"""
    # Reset the generator to get all images
    val_generator.reset()
    
    # Get predictions
    preds = model.predict(val_generator, steps=len(val_generator))
    pred_classes = np.argmax(preds, axis=1)
    
    # Get true labels
    true_classes = val_generator.classes
    
    # Calculate metrics
    acc = accuracy_score(true_classes, pred_classes)
    precision = precision_score(true_classes, pred_classes, average='weighted', zero_division=0)
    recall = recall_score(true_classes, pred_classes, average='weighted')
    f1 = f1_score(true_classes, pred_classes, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(true_classes, pred_classes, target_names=classes)
    print(report)
    
    # Save classification report to file
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Create a confusion matrix as a percentage of true cases
    cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.png'))
    plt.close()

# Main function
def main():
    # Analyze class distribution
    print("Analyzing class distribution...")
    class_counts, classes = analyze_class_distribution(TRAIN_DIR)
    print(f"Class distribution: {class_counts}")
    
    # Create dataframe with all images
    print("Creating dataframe for all images...")
    all_df = create_dataframe(TRAIN_DIR)
    
    # Split data into train and validation
    print("Splitting data into train and validation sets...")
    train_df, val_df = train_test_split(all_df, test_size=0.2, stratify=all_df['class'], random_state=42)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Calculate class weights if needed
    class_imbalance = max(class_counts.values()) / min(class_counts.values())
    print(f"Class imbalance ratio: {class_imbalance:.2f}")
    
    if class_imbalance > 1.5:  # Threshold for considering class imbalance
        print("Class imbalance detected. Applying class weights...")
        class_weights = calculate_class_weights(class_counts)
        print(f"Class weights: {class_weights}")
    else:
        print("No significant class imbalance detected.")
        class_weights = None
    
    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator = create_data_generators(train_df, val_df)
    
    # Create and compile the model
    print("Creating Vision Transformer model...")
    model = create_model(len(classes))
    model.summary()
    
    # Train the model
    print("Starting training...")
    callbacks = get_callbacks()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=len(val_df) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Safe way to handle and plot history
    if history and hasattr(history, 'history') and history.history:
        plot_history(history.history)
    
    # Evaluate the model on validation set
    print("Evaluating model on validation set...")
    evaluate_model(model, val_generator, classes)

# Run the main function
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Try to print more detailed error information
        import traceback
        print(traceback.format_exc()) 




