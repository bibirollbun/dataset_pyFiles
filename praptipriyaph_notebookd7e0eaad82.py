import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications import resnet, vgg16 # For preprocessing

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# --- Define Constants ---
DATA_DIR = "/kaggle/input/aptos2019-blindness-detection/"
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

# Model constants
IMG_SIZE = 224 # As required by the models
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
NUM_CLASSES = 5


# 1. Load the CSV file
df = pd.read_csv(TRAIN_CSV_PATH)

# 2. Create the full image path
#    Example: 'id_code_123' -> '/kaggle/input/.../train_images/id_code_123.png'
df['image_path'] = df['id_code'].apply(lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png"))

# 3. Get labels as integers
#    The 'diagnosis' column is already our label (0, 1, 2, 3, 4)
df['label'] = df['diagnosis']

# 4. Split the data into training and validation sets
#    We use stratify=df['label'] to ensure both sets have a similar
#    distribution of classes, which is crucial for imbalanced datasets.
train_df, val_df = train_test_split(
    df,
    test_size=0.2, # 20% for validation
    random_state=42,
    stratify=df['label']
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")


# 1. Define the image loading and resizing function
def load_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label

# 2. Define the model-specific preprocessing functions
#    Each pre-trained model has its own way of normalizing pixels
def preprocess_vgg16(image, label):
    image = vgg16.preprocess_input(image) # Uses VGG16's specific normalization
    return image, label

def preprocess_resnet50(image, label):
    image = resnet.preprocess_input(image) # Uses ResNet's specific normalization
    return image, label

def preprocess_alexnet(image, label):
    image = image / 255.0 # Scale to [0, 1]
    # Normalize using ImageNet mean and std dev
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return image, label

# 3. Create a helper function to build the full dataset
def create_dataset(df, preprocess_fn):
    # Create a dataset from the dataframe slices
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_path'].values,
        df['label'].values
    ))
    
    # Load and resize images
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply model-specific preprocessing
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch, shuffle, and prefetch for performance
    # For the validation set, we don't need to shuffle
    if 'train' in df.iloc[0]['image_path']:
        dataset = dataset.shuffle(buffer_size=len(df))
        
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# 4. Create the three distinct datasets
#    (We'll do this just before training each model)
#
#    Example:
#    train_ds_vgg16 = create_dataset(train_df, preprocess_vgg16)
#    val_ds_vgg16 = create_dataset(val_df, preprocess_vgg16)


# =============================================================================
# Model 1: VGG-16 (From Keras Applications)
# =============================================================================
def create_vgg16_baseline():
    base_model = VGG16(input_shape=IMG_SHAPE,
                       include_top=False,
                       weights='imagenet')
    base_model.trainable = False # FREEZE
    
    inputs = layers.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# =============================================================================
# Model 2: ResNet-50 (From Keras Applications)
# =============================================================================
def create_resnet50_baseline():
    base_model = ResNet50(input_shape=IMG_SHAPE,
                          include_top=False,
                          weights='imagenet')
    base_model.trainable = False # FREEZE
    
    inputs = layers.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model




# =============================================================================
# Model 3: AlexNet (Manually Implemented) - CORRECTED
# =============================================================================
def create_alexnet_baseline():
    
    # 1. Define the entire AlexNet model + our new head in one stack
    model = models.Sequential([
        layers.Input(shape=IMG_SHAPE),
        layers.Resizing(227, 227), # AlexNet uses 227x227
        
        # --- Start of "base_model" layers ---
        layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        
        layers.Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        
        layers.Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'),
        layers.Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'),
        layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
        
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        # --- End of "base_model" layers ---
        
        # --- Our new classification head ---
        layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')
    
    ], name="alexnet_model")
    
    # 2. FREEZE all layers *except* the last one
    #    This follows the rule "train only the new classification head"
    for layer in model.layers[:-1]:
        layer.trainable = False
        
    return model


# Create a dictionary to store our final results for the table
baseline_results = {}

# We need the true labels from the validation set for final scoring
y_true = val_df['label'].values

# --- 1. Train and Evaluate VGG-16 ---
print("\n--- Training VGG-16 ---")
vgg16_model = create_vgg16_baseline()
vgg16_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use this for integer labels
    metrics=['accuracy']  # <-- FIX: Removed problematic Keras metrics
)

# Create the VGG-16 specific datasets
train_ds_vgg16 = create_dataset(train_df, preprocess_vgg16)
val_ds_vgg16 = create_dataset(val_df, preprocess_vgg16)

# Train the model
vgg16_model.fit(
    train_ds_vgg16,
    epochs=5, # You can increase this, but 5 is a good start
    validation_data=val_ds_vgg16
)

# Get predictions on the validation set
preds_vgg16 = vgg16_model.predict(val_ds_vgg16)
y_pred_vgg16 = np.argmax(preds_vgg16, axis=1) # Convert probabilities to class labels

# Calculate and store metrics
baseline_results['VGG-16'] = {
    'Accuracy': accuracy_score(y_true, y_pred_vgg16),
    'F1 Score': f1_score(y_true, y_pred_vgg16, average='weighted'),
    'Recall': recall_score(y_true, y_pred_vgg16, average='weighted'),
    'Precision': precision_score(y_true, y_pred_vgg16, average='weighted', zero_division=0)
}


# --- 2. Train and Evaluate ResNet-50 ---
print("\n--- Training ResNet-50 ---")
resnet50_model = create_resnet50_baseline()
resnet50_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']  # <-- FIX: Removed problematic Keras metrics
)

# Create the ResNet-50 specific datasets
train_ds_resnet50 = create_dataset(train_df, preprocess_resnet50)
val_ds_resnet50 = create_dataset(val_df, preprocess_resnet50)

# Train the model
resnet50_model.fit(
    train_ds_resnet50,
    epochs=5,
    validation_data=val_ds_resnet50
)

# Get predictions
preds_resnet50 = resnet50_model.predict(val_ds_resnet50)
y_pred_resnet50 = np.argmax(preds_resnet50, axis=1)

# Calculate and store metrics
baseline_results['ResNet-50'] = {
    'Accuracy': accuracy_score(y_true, y_pred_resnet50),
    'F1 Score': f1_score(y_true, y_pred_resnet50, average='weighted'),
    'Recall': recall_score(y_true, y_pred_resnet50, average='weighted'),
    'Precision': precision_score(y_true, y_pred_resnet50, average='weighted', zero_division=0)
}





# --- 3. Train and Evaluate AlexNet ---
print("\n--- Training AlexNet ---")
alexnet_model = create_alexnet_baseline()
alexnet_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']  # <-- FIX: Removed problematic Keras metrics
)

# Create the AlexNet specific datasets
train_ds_alexnet = create_dataset(train_df, preprocess_alexnet)
val_ds_alexnet = create_dataset(val_df, preprocess_alexnet)

# Train the model
alexnet_model.fit(
    train_ds_alexnet,
    epochs=5,
    validation_data=val_ds_alexnet
)

# Get predictions
preds_alexnet = alexnet_model.predict(val_ds_alexnet)
y_pred_alexnet = np.argmax(preds_alexnet, axis=1)

# Calculate and store metrics
baseline_results['AlexNet'] = {
    'Accuracy': accuracy_score(y_true, y_pred_alexnet),
    'F1 Score': f1_score(y_true, y_pred_alexnet, average='weighted'),
    'Recall': recall_score(y_true, y_pred_alexnet, average='weighted'),
    'Precision': precision_score(y_true, y_pred_alexnet, average='weighted', zero_division=0)
}


# Convert the results dictionary to a pandas DataFrame
results_df = pd.DataFrame(baseline_results).T

# Re-order columns to match your assignment
results_df = results_df[['Accuracy', 'F1 Score', 'Recall', 'Precision']]

print("\n--- Baseline Model Results ---")
print(results_df)

# You can now copy this output into Table 1 of your assignment

