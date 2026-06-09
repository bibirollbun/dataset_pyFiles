import tensorflow as tf
import os

print("TensorFlow version:", tf.__version__)

# Check for TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("âœ… TPU connected")
except Exception as e:
    print("â�Œ TPU not found:", e)
    tpu = None

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth (optional but avoids errors)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
        print("âœ… GPU connected:", gpus)
    except RuntimeError as e:
        print("âš ï¸� GPU error:", e)
else:
    print("â�Œ No GPU found. Using CPU.")
    strategy = tf.distribute.get_strategy()

print("\n>>> Final Strategy:", strategy)
print("Number of accelerators in sync:", strategy.num_replicas_in_sync)



import os
import pandas as pd
import tensorflow as tf

# --- 1. Setup base path ---
BASE_PATH = '/kaggle/input/grand-xray-slam-division-b/'

# Confirm files
print("Files in dataset:", os.listdir(BASE_PATH))

# âœ… Use the correct CSV file
df = pd.read_csv(os.path.join(BASE_PATH, "train2.csv"))
print("CSV loaded with shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- 2. Labels ---
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# --- 3. Sample a subset for quick testing ---
sample_df = df.sample(2000, random_state=42)

def get_image_path(image_name):
    return os.path.join(BASE_PATH, "train2", image_name)  # inside the train2 folder
# âœ… Use 'Image_name' (lowercase n)
sample_df['full_path'] = sample_df['Image_name'].apply(get_image_path)

# --- 4. Build TensorFlow Data Pipeline ---
IMG_SIZE = 224
BATCH_SIZE = 32 * strategy.num_replicas_in_sync  # use GPU strategy

def parse_image_and_labels(file_path, labels):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, labels

image_paths = sample_df['full_path'].values
image_labels = sample_df[LABELS].values

dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
dataset = dataset.map(parse_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print("âœ… Data pipeline created successfully!")

# --- 5. Inspect a batch ---
for images, labels in dataset.take(1):
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)



# --- NEW CELL FOR SPLITTING DATA ---

# Based on your logs, you have 31 batches of data (2000 sample size / 64 batch size)
DATASET_SIZE = 31 
TRAIN_SPLIT = int(0.8 * DATASET_SIZE) # 80% for training

# It's crucial to shuffle the data before splitting
dataset = dataset.shuffle(buffer_size=1024, seed=42)

train_dataset = dataset.take(TRAIN_SPLIT)
validation_dataset = dataset.skip(TRAIN_SPLIT)

print(f"âœ… Dataset split into {len(train_dataset)} training batches and {len(validation_dataset)} validation batches.")


# We must define and compile the model inside the strategy scope
# This tells TensorFlow to use the two GPUs we have enabled
with strategy.scope():
    # 1. Load the pre-trained base model
    # We'll use EfficientNetB0, a strong and efficient model for images
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights='imagenet',   # Load weights pre-trained on millions of internet images
        include_top=False     # Exclude the final layer that classifies 1000 objects
    )
    
    # 2. Freeze the base model
    # This prevents the pre-trained knowledge from being erased during initial training
    base_model.trainable = False
    
    # 3. Add our own custom layers on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), # A layer to reduce the feature dimensions
        tf.keras.layers.Dense(14, activation='sigmoid') # FINAL LAYER: 14 outputs, 1 for each condition
    ])
    
    # 4. Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', # The correct loss function for multi-label problems
        metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')] # The competition's evaluation metric
    )

# Print a summary of our new model's architecture
print("âœ… Model built and compiled successfully!")
model.summary()


from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Configure Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 2. Train The Model
print("ğŸš€ Starting model training with Early Stopping...")
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)
print("\nâœ… Model training complete!")

# 3. Plot The Updated Results
print("ğŸ“Š Generating training & validation history plot...")
plt.figure(figsize=(12, 5))

# Plot AUC
plt.subplot(1, 2, 1)
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Model AUC Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# --- THIS IS THE CORRECTED LINE FOR KAGGLE ---
# This saves the plot to Kaggle's output directory
plt.savefig('/kaggle/working/training_validation_history.png', bbox_inches='tight')

plt.show()

print("âœ… Plot saved to Kaggle's output directory.")


# --- 1. Load Test Data Information ---
print("Loading test data information...")
# We use the sample submission file to get the list of test image names in the correct order
submission_df = pd.read_csv(BASE_PATH + 'sample_submission_2.csv')

# Create the full file paths for the test images
def get_test_image_path(image_name):
    # Note that test images are in the 'test2' folder
    return f"{BASE_PATH}test2/{image_name}"

submission_df['full_path'] = submission_df['Image_name'].apply(get_test_image_path)

print(f"Found {len(submission_df)} images in the test set.")

# --- 2. Create the Test Data Pipeline ---
# This pipeline is similar to the training one, but it doesn't process labels
def parse_test_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img

test_image_paths = submission_df['full_path'].values

# Create a dataset of file paths, then map the parsing function
test_dataset = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_dataset = test_dataset.map(parse_test_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("âœ… Test data pipeline created successfully.")

# --- 3. Make Predictions ---
print("\nğŸ§  Making predictions on the test set...")
# The model.predict() function will return the probabilities for each of the 14 labels
predictions = model.predict(test_dataset)
print("âœ… Predictions complete.")

# --- 4. Create and Save the Submission File ---
# Assign the predictions to the correct label columns in our submission DataFrame
submission_df[LABELS] = predictions

# Drop the extra 'full_path' column we created
submission_df = submission_df.drop('full_path', axis=1)

# Save the final DataFrame to a submission.csv file
# index=False is crucial to avoid an extra unwanted column
submission_df.to_csv('submission.csv', index=False)

print("\nâœ… Submission file 'submission.csv' created successfully!")
print("Here's a preview of your submission file:")
print(submission_df.head())

