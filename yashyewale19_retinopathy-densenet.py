# ## Cell 1: Imports and Setup (Updated for Research Paper)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn for metrics and splitting
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, fbeta_score

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import load_model

# CRITICAL IMPORT: All 3 architectures for comparison
from tensorflow.keras.applications import EfficientNetB0, ResNet50, DenseNet121

print(f"TensorFlow Version: {tf.__version__}")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) available: {len(gpus)}")
    # Optional: Set Mixed Precision for T4 GPUs to speed up training
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed Precision enabled.")
else:
    print("No GPU detected.")


# ## Cell 2: Configuration

# Paths to your Kaggle data
TRAIN_CSV = '/kaggle/input/aptos2019-blindness-detection/train.csv'
TRAIN_IMG_DIR = '/kaggle/input/aptos2019-blindness-detection/train_images'

# Model and training hyperparameters
IMG_SIZE = (384, 384)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 25  # Increased from 20 to allow more time for augmentation
NUM_CLASSES = 5

# Set seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ## Cell 3: Load and Prepare Data

# Load the CSV and ensure the diagnosis column is a string for the generator
df = pd.read_csv(TRAIN_CSV)
df['id_code'] = df['id_code'].astype(str) + '.png'

# This line fixes the error by converting the diagnosis labels to strings
df['diagnosis'] = df['diagnosis'].astype(str)

print("Data loaded successfully.")
print(f"Total samples: {len(df)}")


# ## Cell 4: Quick EDA
# Visualize the class distribution
plt.figure(figsize=(10, 6))
df['diagnosis'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Class Distribution of Diabetic Retinopathy')
plt.xlabel('Diagnosis Level')
plt.ylabel('Number of Images')
plt.xticks(rotation=0)
plt.show()


# ## Cell 5: Train/Validation Split
# Create a stratified split to maintain class distribution in both sets
train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=SEED,
    stratify=df['diagnosis']
)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")


# ## Cell 6: Data Generators (with Augmentations)

# Add basic augmentations to the training generator to combat overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# The validation generator should NOT have augmentations
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_IMG_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=TRAIN_IMG_DIR,
    x_col='id_code',
    y_col='diagnosis',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)


# ## Cell 7: Calculate Class Weights
# Correctly compute class weights on the integer labels to handle imbalance
classes = np.unique(train_df['diagnosis'])
class_weights = compute_class_weight(
    'balanced',
    classes=classes,
    y=train_df['diagnosis'].values
)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}

print("Class weights calculated:")
print(class_weight_dict)


# ## Cell 8: Build the DenseNet121 Model (Experiment A - Part 3)

def build_densenet_model(input_shape=IMG_SIZE + (3,), n_classes=NUM_CLASSES):
    """Builds a DenseNet121 model for comparative analysis."""
    
    # Base model: Switched to DenseNet121
    # DenseNet connects each layer to every other layer in a feed-forward fashion.
    # This architecture is excellent for feature reuse in medical imaging.
    base = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Full Fine-Tuning: We unfreeze the base to learn specific diabetic retinopathy features
    base.trainable = True 

    # Model architecture (Kept IDENTICAL to EfficientNet/ResNet for fair comparison)
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    # Compile (Same optimizer and loss as previous models)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Build the DenseNet model
model = build_densenet_model()

# Print summary to confirm architecture
# Note: DenseNet121 usually has around 7-8 Million parameters (lighter than ResNet)
model.summary()


# ## Cell 9: Define Callbacks for DenseNet

# 1. ModelCheckpoint: Save to a NEW file 'best_model_densenet.h5'
# We rename this so we don't overwrite the ResNet or EfficientNet models
checkpoint = ModelCheckpoint(
    'best_model_densenet.h5',  
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1)

# 2. CSVLogger: Save the training history to a CSV file
# This allows you to plot the Learning Curves for your paper later
csv_logger = CSVLogger('training_log_densenet.csv', separator=',', append=False)

# 3. EarlyStopping (Same patience as previous models for fair comparison)
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5, 
    restore_best_weights=True,
    mode='max',
    verbose=1)

# 4. ReduceLROnPlateau (Same settings as previous models)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-7)

# Combine all callbacks into a list
callbacks_list = [checkpoint, early_stopping, reduce_lr, csv_logger]


# ## Cell 10: Train the DenseNet121 Model

# We store history in 'history_densenet' variable
# This allows you to compare it against 'history' (EfficientNet) and 'history_resnet' (ResNet)
history_densenet = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks_list,
    verbose=1
)


# ## Cell 11: Plot Training History (DenseNet121)

def plot_history(history, model_name="Model"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} Loss')
    
    # Save the plot for your Research Paper
    plt.savefig(f'{model_name}_training_curves.png')
    plt.show()

# Plot the history for DenseNet
# IMPORTANT: Ensure you pass 'history_densenet' here
if 'history_densenet' in locals():
    plot_history(history_densenet, model_name="DenseNet121")
else:
    # Fallback: If you restarted the notebook, load from CSV
    print("History variable not found in RAM. Loading from CSV...")
    if os.path.exists('training_log_densenet.csv'):
        history_df = pd.read_csv('training_log_densenet.csv')
        # Mock a history object structure for the function
        class HistoryObj:
            pass
        mock_history = HistoryObj()
        mock_history.history = history_df.to_dict(orient='list')
        plot_history(mock_history, model_name="DenseNet121")
    else:
        print("No history found.")


# ## Cell 12: Evaluate the DenseNet121 Model (Experiment A - Part 3 Results)

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# -----------------------------------------------------------
# FIX 1: Define Custom Layer and FIX 2: Automatic Path Search
# -----------------------------------------------------------

# Define the custom Cast layer (Needed because of Mixed Precision saving)
@tf.keras.utils.register_keras_serializable()
class Cast(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Cast, self).__init__(**kwargs)
    def call(self, inputs):
        return inputs
    def get_config(self):
        return super(Cast, self).get_config()

custom_objects_dict = {'Cast': Cast}
MODEL_FILENAME = 'best_model_densenet.h5' # <--- TARGETS DENSENET MODEL
MODEL_PATH = None
model_found = False

# Search for the model file in the entire Kaggle file system
for root, _, files in os.walk('/kaggle/'):
    if MODEL_FILENAME in files:
        MODEL_PATH = os.path.join(root, MODEL_FILENAME)
        model_found = True
        break

if model_found:
    print(f"Loading model from: {MODEL_PATH}")
    try:
        # Load the ENTIRE model (architecture + weights + custom objects)
        model = load_model(MODEL_PATH, custom_objects=custom_objects_dict)
        print("âœ… Model loaded successfully!")
    except Exception as e:
        print(f"â�Œ Critical Error during model loading: {e}")
        model = None 
else:
    print(f"â�Œ File Not Found: Could not locate '{MODEL_FILENAME}'. Ensure you ran the training commit successfully.")
    model = None

# -----------------------------------------------------------
# Evaluation Logic
# -----------------------------------------------------------

if model:
    # Make predictions on the validation set
    # val_gen must be defined by Cell 6 before running this cell!
    preds = model.predict(val_gen)
    pred_classes = np.argmax(preds, axis=1)

    # Get true labels directly from the generator
    true_classes = val_gen.classes

    # Calculate Metrics
    qwk = cohen_kappa_score(true_classes, pred_classes, weights='quadratic')
    print(f"\nğŸ“ˆ Validation Quadratic Weighted Kappa (QWK): {qwk:.4f}\n")

    f2 = fbeta_score(true_classes, pred_classes, beta=2, average='weighted')
    print(f"ğŸ”¬ Weighted F2 Score (Sensitivity Emphasis): {f2:.4f}\n")

    # Print Classification Report
    print("ğŸ“Š Classification Report (DenseNet121):\n")
    print(classification_report(true_classes, pred_classes, target_names=[str(i) for i in classes]))

    # Display Confusion Matrix
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (DenseNet121)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
else:
    print("\nEvaluation skipped because the model could not be loaded.")




