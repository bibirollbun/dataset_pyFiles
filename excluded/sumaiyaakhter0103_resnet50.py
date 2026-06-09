# =======================
# IMPORTS & GPU SETUP
# =======================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision

import warnings
warnings.filterwarnings('ignore')

# =======================
# GPU CHECK & CONFIG
# =======================
# Set TensorFlow to use mixed precision for faster GPU training
mixed_precision.set_global_policy('mixed_float16')

# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))
if gpus:
    print("GPU Name:", gpus[0].name)
else:
    print("No GPU detected. Please enable GPU in your runtime.")

# =======================
# REPRODUCIBILITY
# =======================
tf.random.set_seed(42)
np.random.seed(42)

# =======================
# NOTES
# =======================
# - TensorFlow will automatically utilize GPU if available
# - Mixed precision reduces memory usage and speeds up training
# - Use batch sizes appropriate for your GPU (usually 16-64 for EfficientNetB0)




# =============================================================================
# 2ï¸�âƒ£ LOAD AND PREPROCESS DATA
# =============================================================================
import os
import pandas as pd
import numpy as np

# Dataset paths
BASE_DIR = "/kaggle/input/siim-isic-melanoma-classification"
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "jpeg/train")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "jpeg/test")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(BASE_DIR, "test.csv")

# Load metadata
print("ğŸ“Š Loading dataset metadata...")
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

# Create full image paths
train_df['image_path'] = train_df['image_name'].apply(lambda x: os.path.join(TRAIN_IMAGES_DIR, f"{x}.jpg"))
test_df['image_path'] = test_df['image_name'].apply(lambda x: os.path.join(TEST_IMAGES_DIR, f"{x}.jpg"))

# Handle missing values
train_df['age_approx'] = train_df['age_approx'].fillna(train_df['age_approx'].median()).astype(np.float32)
train_df['sex'] = train_df['sex'].fillna('unknown')
train_df['anatom_site_general_challenge'] = train_df['anatom_site_general_challenge'].fillna('unknown')

# Basic dataset info
print(f"âœ… Training set size: {len(train_df)} images")
print(f"âœ… Test set size: {len(test_df)} images")
print(f"âœ… Malignant cases: {train_df['target'].sum()} ({train_df['target'].mean()*100:.2f}%)")

# Optional: Quick visualization of class distribution
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='target', data=train_df)
plt.title("Class Distribution")
plt.show()




# =============================================================================
# 3ï¸�âƒ£ EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\nğŸ”� Performing Exploratory Data Analysis...")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18, 10))

# Plot 1: Class distribution (pie chart)
plt.subplot(2, 3, 1)
class_counts = train_df['target'].value_counts()
plt.pie(class_counts.values, labels=['Benign (0)', 'Malignant (1)'], autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution')

# Plot 2: Age distribution by diagnosis
plt.subplot(2, 3, 2)
sns.histplot(data=train_df, x='age_approx', hue='target', bins=30, alpha=0.6, palette=['green','red'])
plt.title('Age Distribution by Diagnosis')
plt.xlabel('Age')

# Plot 3: Anatomical site distribution
plt.subplot(2, 3, 3)
site_counts = train_df['anatom_site_general_challenge'].value_counts().head(6)
sns.barplot(x=site_counts.index, y=site_counts.values, palette='viridis')
plt.title('Top Anatomical Sites')
plt.xticks(rotation=45)

# Plot 4: Sex distribution by diagnosis
plt.subplot(2, 3, 4)
sex_diagnosis = pd.crosstab(train_df['sex'], train_df['target'])
sex_diagnosis.plot(kind='bar', stacked=True, ax=plt.gca(), color=['green','red'])
plt.title('Sex Distribution by Diagnosis')
plt.legend(['Benign', 'Malignant'])
plt.xticks(rotation=0)

# Plot 5: Sample images placeholder
plt.subplot(2, 3, 5)
plt.text(0.5, 0.5, 'Sample Images Preview\n(2 benign + 2 malignant)', 
         ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
plt.axis('off')
plt.title('Sample Images Preview')

# Plot 6: Patient-level lesions distribution
plt.subplot(2, 3, 6)
patient_lesions = train_df['patient_id'].value_counts()
sns.histplot(patient_lesions.values, bins=30, color='purple')
plt.title('Lesions per Patient Distribution')
plt.xlabel('Number of Lesions')

plt.tight_layout()
plt.show()

# Print detailed statistics
print("\nğŸ“ˆ Dataset Statistics:")
print(f"Class distribution:\n{train_df['target'].value_counts()}")
print(f"\nAge statistics:\n{train_df['age_approx'].describe()}")
print(f"\nSex distribution:\n{train_df['sex'].value_counts()}")
print(f"\nTop anatomical sites:\n{train_df['anatom_site_general_challenge'].value_counts().head()}")



# =============================================================================
# 4ï¸�âƒ£ DATA PREPARATION WITH IMBALANCE HANDLING (CRITICAL FIX)
# =============================================================================
print("\nğŸ”„ Preparing data with imbalance handling...")

import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import os

# ============================================
# FIRST: Reload or use existing train_df
# ============================================
# If train_df is not defined, reload it
try:
    train_df
    print("âœ“ train_df already loaded")
except NameError:
    print("âš ï¸�  train_df not found. Reloading data...")
    BASE_DIR = "/kaggle/input/siim-isic-melanoma-classification"
    TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "jpeg/train")
    TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")
    
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df['image_path'] = train_df['image_name'].apply(
        lambda x: os.path.join(TRAIN_IMAGES_DIR, f"{x}.jpg")
    )
    
    # Handle missing values
    train_df['age_approx'] = train_df['age_approx'].fillna(train_df['age_approx'].median()).astype(np.float32)
    train_df['sex'] = train_df['sex'].fillna('unknown')
    train_df['anatom_site_general_challenge'] = train_df['anatom_site_general_challenge'].fillna('unknown')
    
    print(f"âœ… Reloaded training set: {len(train_df)} images")

# ============================================
# CRITICAL STEP 1: Handle Extreme Imbalance
# ============================================
print("\nâš–ï¸�  Balancing the dataset...")
print(f"Original class distribution: {train_df['target'].value_counts().to_dict()}")
print(f"Malignant percentage: {(train_df['target'].sum()/len(train_df))*100:.2f}%")

# Strategy: Undersample majority class (benign) 
benign_samples = train_df[train_df['target'] == 0]
malignant_samples = train_df[train_df['target'] == 1]

# Create balanced dataset (1:3 ratio - malignant:benign)
# This gives us 3 benign for every 1 malignant
benign_downsampled = benign_samples.sample(n=len(malignant_samples)*3, random_state=42)
balanced_df = pd.concat([benign_downsampled, malignant_samples])

# Shuffle the balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nğŸ“Š After balancing:")
print(f"  Balanced dataset size: {len(balanced_df)}")
print(f"  Balanced class distribution: {balanced_df['target'].value_counts().to_dict()}")
print(f"  New class ratio: 1:{len(benign_downsampled)/len(malignant_samples):.1f}")
print(f"  Malignant percentage: {(balanced_df['target'].sum()/len(balanced_df))*100:.2f}%")

# ============================================
# CRITICAL STEP 2: Use Stratified Split
# ============================================
print("\nğŸ“‚ Creating train/validation split...")
train_data, val_data = train_test_split(
    balanced_df, 
    test_size=0.15, 
    random_state=42, 
    stratify=balanced_df['target']  # THIS IS CRITICAL
)

print(f"  Training samples: {len(train_data)}")
print(f"  Validation samples: {len(val_data)}")
print(f"  Training class dist: {train_data['target'].value_counts().to_dict()}")
print(f"  Validation class dist: {val_data['target'].value_counts().to_dict()}")

# ============================================
# CRITICAL STEP 3: Adjust Class Weights
# ============================================
# Use MORE REASONABLE class weights for balanced data
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_data['target']),
    y=train_data['target']
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nğŸ�¯ Class weights for balanced data: {class_weight_dict}")

# ============================================
# CRITICAL STEP 4: Enhanced Data Augmentation
# ============================================
# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Function to load and preprocess image
def process_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize to [0,1]
    return img, label

# ENHANCED AUGMENTATION for MALIGNANT cases only
def augment_malignant(img, label):
    # Apply more aggressive augmentation to malignant cases
    if label == 1:  # Only augment malignant cases
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.3)
        img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
        img = tf.image.random_saturation(img, lower=0.7, upper=1.3)
        # Add rotation for malignant cases only
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # Mild augmentation for all cases
    else:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
    return img, label

# ============================================
# CRITICAL STEP 5: Create Dataset Pipelines
# ============================================
print("\nğŸ”§ Creating TensorFlow datasets...")

# Training dataset with targeted augmentation
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_data['image_path'].values, train_data['target'].values)
)
train_dataset = train_dataset.shuffle(buffer_size=len(train_data), seed=42, reshuffle_each_iteration=True)
train_dataset = train_dataset.map(process_image, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(augment_malignant, num_parallel_calls=AUTOTUNE)  # Targeted augmentation
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Validation dataset (NO augmentation, only basic preprocessing)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_data['image_path'].values, val_data['target'].values)
)
val_dataset = val_dataset.map(process_image, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print(f"âœ… Training batches: {len(train_dataset)}")
print(f"âœ… Validation batches: {len(val_dataset)}")
print(f"âœ… Samples per batch: {BATCH_SIZE}")

# ============================================
# Quick sanity check: Show sample distribution
# ============================================
print("\nğŸ“‹ Final dataset summary:")
print(f"  Original dataset: {len(train_df)} samples")
print(f"    - Benign: {len(benign_samples)}")
print(f"    - Malignant: {len(malignant_samples)}")
print(f"  Balanced dataset: {len(balanced_df)} samples")
print(f"    - Benign: {len(benign_downsampled)}")
print(f"    - Malignant: {len(malignant_samples)}")
print(f"  Training set: {len(train_data)} samples")
print(f"  Validation set: {len(val_data)} samples")
print(f"  Image size: {IMG_SIZE}")
print(f"  Class weights: {class_weight_dict}")


# =============================================================================
# 5ï¸�âƒ£ BUILD RESNET50 MODEL (for balanced dataset)
# =============================================================================
print("\nğŸ§  Building ResNet50 Model for balanced data...")

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import ResNet50

# Ensure mixed precision is enabled
mixed_precision.set_global_policy('mixed_float16')

def create_resnet_model(input_shape=(224, 224, 3)):
    # Load pre-trained ResNet50
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze base model layers (optional)
    # base_model.trainable = False
    
    # Add custom layers on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid', dtype='float32')
    ])
    
    return model

# Create and compile model
model = create_resnet_model()
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower learning rate is better for transfer learning
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']  # â†� lowercase!
)
print("âœ… Model architecture (ResNet50 based):")
model.summary()


# =============================================================================
# 6ï¸�âƒ£ CALLBACKS AND TRAINING PREPARATION (GPU-Optimized)
# =============================================================================
print("\nâ�° Setting up training callbacks...")

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Callbacks for efficient GPU training
callbacks = [
    EarlyStopping(
        monitor='val_loss',       # Stop training when validation loss stops improving
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',       # Reduce learning rate when validation loss plateaus
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_melanoma_model.h5',  # Save the best model during training
        monitor='val_accuracy',     # Track validation accuracy
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print("âœ… Callbacks ready for training!")



# =============================================================================
# 7ï¸�âƒ£ TRAIN THE MODEL (GPU-Optimized)
# =============================================================================
print("\nğŸš€ Starting GPU-optimized model training...")

# Parameters
EPOCHS = 50

# Train the model using tf.data.Dataset pipelines
history = model.fit(
    train_dataset,                      # GPU-optimized training dataset
    validation_data=val_dataset,        # Validation dataset
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

print("âœ… Training completed!")



# =============================================================================
# 8ï¸�âƒ£ MODEL EVALUATION (GPU-Optimized) - FIXED VERSION
# =============================================================================
print("\nğŸ“Š Evaluating model performance on GPU...")

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
import numpy as np

# Load best saved model
model.load_weights('best_melanoma_model.h5')

# ============================================
# FIX: Evaluate model (returns 5 values now)
# ============================================
eval_results = model.evaluate(val_dataset, verbose=1)

# Unpack based on number of metrics
if len(eval_results) == 5:  # loss + 4 metrics (accuracy, precision, recall, auc)
    val_loss, val_accuracy, val_precision, val_recall, val_auc_metric = eval_results
    print(f"ğŸ�¯ Validation Loss: {val_loss:.4f}")
    print(f"ğŸ�¯ Validation Accuracy: {val_accuracy:.4f}")
    print(f"ğŸ�¯ Validation Precision: {val_precision:.4f}")
    print(f"ğŸ�¯ Validation Recall: {val_recall:.4f}")
    print(f"ğŸ�¯ Validation AUC (from metric): {val_auc_metric:.4f}")
elif len(eval_results) == 4:  # loss + 3 metrics
    val_loss, val_accuracy, val_precision, val_recall = eval_results
    print(f"ğŸ�¯ Validation Loss: {val_loss:.4f}")
    print(f"ğŸ�¯ Validation Accuracy: {val_accuracy:.4f}")
    print(f"ğŸ�¯ Validation Precision: {val_precision:.4f}")
    print(f"ğŸ�¯ Validation Recall: {val_recall:.4f}")
else:
    print(f"âš ï¸� Unexpected number of evaluation metrics: {len(eval_results)}")

# ============================================
# Make predictions for additional metrics
# ============================================
print("\nğŸ“ˆ Making predictions for detailed analysis...")
val_predictions = model.predict(val_dataset)
val_pred_binary = (val_predictions > 0.5).astype(int).flatten()

# True labels
val_true = np.concatenate([y for x, y in val_dataset], axis=0)

# Calculate AUC-ROC (better to calculate it ourselves for consistency)
auc_roc = roc_auc_score(val_true, val_predictions)
print(f"ğŸ�¯ AUC-ROC Score (calculated): {auc_roc:.4f}")

# Calculate F1-Score
f1 = f1_score(val_true, val_pred_binary)
print(f"ğŸ�¯ F1-Score: {f1:.4f}")

# Classification report
print("\nğŸ“‹ Classification Report:")
print(classification_report(val_true, val_pred_binary, target_names=['Benign', 'Malignant']))

# Confusion matrix
cm = confusion_matrix(val_true, val_pred_binary)
print("\nğŸ—‚ Confusion Matrix:")
print(cm)

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\nğŸ“Š Detailed Metrics from Confusion Matrix:")
print(f"  True Negatives (Benign correctly identified): {tn}")
print(f"  False Positives (Benign misclassified as Malignant): {fp}")
print(f"  False Negatives (Malignant missed): {fn}")
print(f"  True Positives (Malignant correctly identified): {tp}")

# Calculate sensitivity and specificity
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity: {specificity:.4f}")


# =============================================================================
# 9ï¸�âƒ£ VISUALIZE RESULTS
# =============================================================================
print("\nğŸ“ˆ Visualizing training results and model performance...")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

# --- Training history plots ---
plt.figure(figsize=(18, 5))

# 1ï¸�âƒ£ Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 2ï¸�âƒ£ Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 3ï¸�âƒ£ ROC Curve
plt.subplot(1, 3, 3)
fpr, tpr, _ = roc_curve(val_true, val_predictions)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# --- Confusion Matrix ---
plt.figure(figsize=(6, 5))
cm = confusion_matrix(val_true, val_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()



# =============================================================================
# ğŸ”Ÿ MODEL INTERPRETATION AND FINAL RESULTS
# =============================================================================
print("\nğŸ’¡ Model Performance Summary:")
print("="*50)
print(f"âœ… Final Validation Accuracy : {val_accuracy:.4f}")
print(f"âœ… Final Validation Precision: {val_precision:.4f}")
print(f"âœ… Final Validation Recall   : {val_recall:.4f}")
print(f"âœ… AUC-ROC Score            : {auc_roc:.4f}")
print(f"âœ… Training Samples         : {len(train_data)}")
print(f"âœ… Validation Samples       : {len(val_data)}")
print(f"âœ… Class Weights Applied    : {class_weight_dict}")

# Additional metric: F1-Score
from sklearn.metrics import f1_score
f1 = f1_score(val_true, val_pred_binary)
print(f"âœ… F1-Score                 : {f1:.4f}")

# Save the final model
model.save('melanoma_classification_final_model.h5')
print("ğŸ’¾ Model saved as 'melanoma_classification_final_model.h5'")

print("\nğŸ�‰ Training, evaluation, and final results completed successfully!")


