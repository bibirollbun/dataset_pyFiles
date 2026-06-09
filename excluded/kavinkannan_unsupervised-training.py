import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define constants
IMG_SIZE = 150
NUM_CATEGORIES = 10
INPUT_DIR = '/kaggle/input/2025-computer-vision-homework-2/cv_data'
TRAIN_DIR = os.path.join(INPUT_DIR, 'train')
TEST_DIR = os.path.join(INPUT_DIR, 'test')
VALID_DIR = os.path.join(INPUT_DIR, 'valid')

# Load images function
def load_images(directory, max_samples=None):
    images = []
    filenames = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return np.array([]), []
    
    image_files = [f for f in os.listdir(directory) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if max_samples is not None:
        image_files = image_files[:max_samples]
    
    for filename in image_files:
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize
            images.append(img)
            filenames.append(filename)
    
    print(f"Loaded {len(images)} images from {directory}")
    return np.array(images), filenames

# Custom feature extractor
def extract_features_custom(images):
    if len(images) == 0:
        return np.array([])
    
    print("Using custom CNN feature extractor")
    
    # Create a simple feature extractor - fixed to use proper input creation
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    feature_extractor = tf.keras.Model(inputs, x)
    
    # Process in smaller batches to avoid memory issues
    batch_size = 32
    features = []
    for i in range(0, len(images), batch_size):
        end_idx = min(i + batch_size, len(images))
        batch = images[i:end_idx]
        batch_features = feature_extractor.predict(batch, verbose=0)
        features.append(batch_features)
        print(f"Processed batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
    
    return np.vstack(features)

# Clustering (K-Means)
def cluster_images(features, num_clusters):
    if len(features) == 0:
        print("No features to cluster")
        return np.array([]), None
        
    print(f"Clustering {len(features)} feature vectors into {num_clusters} clusters")
    
    # Use KMeans with multiple initializations for better results
    kmeans = KMeans(
        n_clusters=num_clusters, 
        random_state=42, 
        n_init=10,
        max_iter=300
    )
    
    clusters = kmeans.fit_predict(features)
    
    # Print cluster distribution
    unique, counts = np.unique(clusters, return_counts=True)
    distribution = dict(zip(unique, counts))
    print("Cluster distribution:", distribution)
    
    return clusters, kmeans

# Custom callback to stop at 95% accuracy
class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.95):
        super(AccuracyThresholdCallback, self).__init__()
        self.threshold = threshold
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get('accuracy') >= self.threshold:
            print(f'\nReached {self.threshold*100}% accuracy, stopping training.')
            self.model.stop_training = True

# Main execution flow
print("Loading training images...")
train_images, train_filenames = load_images(TRAIN_DIR, max_samples=5000)
print("Loading validation images...")
valid_images, valid_filenames = load_images(VALID_DIR, max_samples=1000)
print("Loading test images...")
test_images, test_filenames = load_images(TEST_DIR)  # Load all test images

# Extract features for training data
print("Extracting features from training images...")
train_features = extract_features_custom(train_images)
print(f"Extracted features shape: {train_features.shape}")

# Cluster training data
print("Clustering training features...")
train_labels, kmeans_model = cluster_images(train_features, NUM_CATEGORIES)

# Extract features for validation data
print("Extracting features from validation images...")
valid_features = extract_features_custom(valid_images)

# Use the same clustering model to predict validation labels
print("Assigning validation data to clusters...")
valid_labels = kmeans_model.predict(valid_features) if len(valid_features) > 0 else np.array([])

# Build CNN model to predict the clusters 
print("Building CNN model...")
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CATEGORIES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define model checkpoint callback - fixed filepath to end with .keras
model_save_path = 'kavins_unsupervised_trained_cnn.keras'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_save_path,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Define callbacks
callbacks = [
    AccuracyThresholdCallback(threshold=0.95),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    ),
    checkpoint_callback
]

# Train the model
print("Training CNN model...")
history = model.fit(
    train_images, 
    train_labels,
    epochs=20,  # Maximum of 20 epochs (was 10)
    validation_data=(valid_images, valid_labels) if len(valid_images) > 0 and len(valid_labels) > 0 else None,
    callbacks=callbacks,
    batch_size=32,
    verbose=1
)

# Save the final model (in case best model wasn't saved by checkpoint)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Save KMeans model
import joblib 
kmeans_save_path = 'kmeans_clustering_model.joblib'
joblib.dump(kmeans_model, kmeans_save_path)
print(f"KMeans model saved to {kmeans_save_path}")

# Load the saved model for prediction (to ensure we're using the best model)
print("Loading the saved model for prediction...")
model = tf.keras.models.load_model(model_save_path)

# Generate predictions for all test images
print(f"Generating predictions for {len(test_images)} test images...")
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

ids = []
for filename in test_filenames:
    id_name = os.path.splitext(filename)[0]
    ids.append(id_name)

results_df = pd.DataFrame({
    'id': ids,
    'label': predicted_labels
})

output_path = 'predictions.csv'
results_df.to_csv(output_path, index=False)

print(f"Prediction file saved to {output_path}")
print(f"Total predictions: {len(results_df)}")
print("All predictions:")
print(results_df) 

