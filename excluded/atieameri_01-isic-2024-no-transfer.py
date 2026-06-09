# Environment Settings
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["KERAS_BACKEND"] = "tensorflow"
import random
import shutil
from glob import glob

# Core Libraries
import joblib
import time

# Data Manipulation
import numpy as np
import pandas as pd

# Image Processing
from PIL import Image
import cv2

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-v0_8-darkgrid")

# Machine Learning & Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Accuracy, Precision
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K

# Keras Utilities
from keras import ops

# Model Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Progress Tracking
from tqdm.notebook import tqdm
print("Libraries loaded successfully!")


print(f"Tensdorflow vesrion: {tf.__version__}")


# Clear the current TensorFlow graph
tf.keras.backend.clear_session()

# Optional: Release GPU memory
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
    # Release GPU memory
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
else:
    print("GPU is not available")


SEED = 42

# Set seed for NumPy
np.random.seed(SEED)

# Set seed for Python's built-in random module
random.seed(SEED)

# Set seed for TensorFlow
tf.random.set_seed(SEED)


import glob
for i in glob.glob ("/kaggle/working/*/*"):
    os.remove(i)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# Constants
BATCH_SIZE = 32
IMG_SIZE = 299
EPOCHS = 20
LEARNING_RATE = 1e-4


# Load metadata

data = pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv")
print(f"The shape of the dataframe : { data.shape}"  )



colors = {
    'benign': '#8fb9aa',    # soft teal for benign
    'malignant': '#e8c2ca', # soft pink for malignant
    'bg': '#f8f9fa',        # light background
    'text': '#2d3436'       # dark text
}

# Count class distribution
benign_count = (data['target'] == 0).sum()
malignant_count = (data['target'] == 1).sum()
total = len(data)

# Create figure
fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor(colors['bg'])

# Create pie chart
wedges, texts, autotexts = ax.pie(
    [benign_count, malignant_count],
    labels=['Benign', 'Malignant'],
    autopct='%1.1f%%',
    startangle=45,
    colors=[colors['benign'], colors['malignant']],
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'color': colors['text'], 'fontsize': 10}
)

# Style the percentage text
for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
    autotext.set_color('white')

# Add title
ax.set_title('Distribution of Target Classes', fontsize=16, fontweight='bold', pad=20, color=colors['text'])

# Add legend with count information
plt.legend(
    wedges,
    [f'Benign: {benign_count} ({benign_count/total:.1%})',
     f'Malignant: {malignant_count} ({malignant_count/total:.1%})'],
    loc='upper left',
    frameon=False,
    fontsize=8
)

plt.tight_layout()
plt.savefig('target_distribution_pie.png', dpi=200, bbox_inches='tight', facecolor=colors['bg'])
plt.show()


negative_df = data[data["target"] == 0].sample(frac=0.02, random_state=SEED)
positive_df = data[data["target"] == 1]
IMAGE_ROOT = "/kaggle/input/isic-2024-challenge/train-image/image"




for i in glob.glob("/kaggle/working/aug_folder/*"):
   os.remove(i)

# Create augmentation directory if it doesn't exist
os.makedirs("/kaggle/working/aug_folder", exist_ok=True)

# Augmentation settings
augmentations_per_image = 10
augmentation_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

#Store augmented images metadata
augmented_images = []
AUGMENTED_DIR = "/kaggle/working/aug_folder"

# Process positive images for augmentation
for _, row in positive_df.iterrows():
    img_id = row.isic_id
    img_path = os.path.join(IMAGE_ROOT, img_id + ".jpg")
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skipping.")
        continue
    try:
        # Load and preprocess image
        image = plt.imread(img_path)
        image = np.expand_dims(image, 0)  # Add batch dimension
        for i in range(augmentations_per_image):
            aug_batch = next(augmentation_datagen.flow(image, batch_size=1))
            aug_img = aug_batch[0]
            # Normalize if needed
            if aug_img.max() > 1.0:
                aug_img /= 255.0
            save_filename = f"aug_{img_id}_{i}.jpg"
            save_path = os.path.join(AUGMENTED_DIR, save_filename)
            plt.imsave(save_path, aug_img)
            augmented_images.append([save_filename,
                                    1,
                                    row.sex,
                                    row.age_approx, 
                                    row.anatom_site_general,
                                    save_path])
    
    except Exception as e:  
        print(f"Error processing {img_path}: {e}")


### add the type as augmented for stratifying when splitting
df_aug = pd.DataFrame(augmented_images, columns= ["isic_id", "target", "sex",
                                                  "age_approx",'anatom_site_general','image_path'])
df_aug["type"]= "augmented"
df_aug


data= data[["isic_id", "target", "sex",
"age_approx",'anatom_site_general',]]

data["image_path"] = data["isic_id"].apply(lambda x: os.path.join(IMAGE_ROOT, x + ".jpg"))
data["type"]= "original"

# # Undersample negative class (1.5% of negatives)
negative_df = data[data["target"] == 0].sample(frac=0.015, random_state=SEED)

# # Posiitve class 
positive_df = data[data["target"] == 1]
positive_df

print(f" Lenght of Positive Oiginal Data: {len(positive_df)} ")
print(f" Lenght of Negative Oiginal Data: {len(negative_df)} ")
print(f" Lenght of Positive Augmented Data: {len(df_aug)} ")


df_combined = pd.concat([positive_df, negative_df, df_aug], ignore_index=True)
print(df_combined["image_path"][0])

Image.open(df_combined["image_path"][0])


df_combined["stratified"]= df_combined["target"].astype("str") +df_combined["type"]
df_combined.head()


import random
import matplotlib.pyplot as plt
from PIL import Image

# Define your groups with better labels
group_labels = {
    "1original": "Positive (Original)",
    "1augmented": "Positive (Augmented)",
    "0original": "Negative (Original)"
}

# Create a figure with a more compact size
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 8))
fig.suptitle('Dataset Visualization by Category', fontsize=14)

# Get the groups in the order you want
groups = ["1original", "1augmented", "0original"]

for row_idx, group in enumerate(groups):
    # Get random sample of 3 images from this group
    group_samples = df_combined[df_combined["stratified"] == group]
    if len(group_samples) > 3:
        group_df = group_samples.sample(3)  # Random sampling
    else:
        group_df = group_samples.head(3)  # Take all if less than 3
    
    for col_idx, (_, row) in enumerate(group_df.iterrows()):
        ax = axes[row_idx, col_idx]
        try:
            image = Image.open(row["image_path"])
            ax.imshow(image)
            ax.axis("off")
            
            # Add title in the middle column
            if col_idx == 1:
                ax.set_title(group_labels[group], fontsize=10)
                
            # Add sample count in bottom right of first column
            if col_idx == 0:
                total = len(df_combined[df_combined["stratified"] == group])
                ax.text(0.02, 0.02, f"n={total}", transform=ax.transAxes, 
                        fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.7))
                
        except Exception as e:
            ax.text(0.5, 0.5, "Image not found", ha="center", fontsize=8)
            ax.axis("off")

# Increase spacing between rows
plt.subplots_adjust(wspace=0.1, hspace=0.4)  # Increased hspace for more distance between rows
plt.tight_layout()
plt.show()


# Split dataset while maintaining class balance
rest_df, test_df = train_test_split(df_combined, test_size=0.15, stratify=df_combined["stratified"], random_state=42)

train_df, val_df = train_test_split(rest_df, test_size=0.2, stratify=rest_df["stratified"], random_state=42)

rest_df.drop(columns=['stratified'], inplace=True)
test_df.drop(columns=['stratified'], inplace=True)
val_df.drop(columns=['stratified'], inplace=True)


##Define callbacks
checkpoint_cb = ModelCheckpoint(
    filepath="models/xception_skincancer_notransfer.keras",
    monitor='val-auc', 
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor='val-auc',  # Using underscore, not hyphen
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    mode='max',
    verbose=1
)

early_stopping_cb = EarlyStopping(
    monitor='val_loss',  # Using underscore, not hyphen
    patience=8,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.003,
    mode='min'
)


callbacks = [checkpoint_cb, reduce_lr_cb,early_stopping_cb ]




def f1B_score(y_true, y_pred, beta=2):
    """
    Computes the normalized F1 score (scaled between 0 and 1).
    - If all predictions are correct → F1 = 1.
    - If all predictions are wrong → F1 = 0.
    - Handles class imbalance by normalizing relative to the best possible F1 for the given labels.

    Args:
        y_true: True labels (binary, 0 or 1).
        y_pred: Predicted probabilities.
    
    Returns:
        Normalized F1 score (0 to 1).
    """
    # Ensure y_pred and y_true are flat tensors
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    # Sort predictions and corresponding true values in descending order
    sorted_indices = tf.argsort(y_pred, direction='DESCENDING')
    sorted_y_true = tf.gather(y_true, sorted_indices)
    sorted_y_pred = tf.gather(y_pred, sorted_indices)

    # Calculate cumulative true positives (TP), false positives (FP), and false negatives (FN)
    cumulative_tp = tf.cumsum(sorted_y_true)
    cumulative_fp = tf.cumsum(1 - sorted_y_true)
    total_positives = tf.reduce_sum(y_true)
    total_negatives = tf.reduce_sum(1 - y_true)
    fn = total_positives - cumulative_tp

    # Calculate precision and recall
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + K.epsilon())
    recall = cumulative_tp / (total_positives + K.epsilon())

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())


    beta_squared = beta ** 2
    fbeta = ((1 + beta_squared) * precision * recall) / \
            ((beta_squared * precision) + recall +  K.epsilon())
    

    # Get the maximum possible F1 score (best achievable F1 for this y_true)
    best_fbeta = tf.reduce_max(fbeta)

    # Normalize the F1 score (if best_f1 is 0, avoid division by zero)
    normalized_f1B = best_fbeta / (1.0 + K.epsilon())  # Since F1 is already in [0,1], this ensures no division issues

    return normalized_f1B


# # Test with some sample values
# y_true_1 = np.array([1, 0, 1, 1, 0])
# y_pred_1 = np.array([0.9, 0.8, 0.7, 0.3, 0.2])

# y_true_2 = np.array([1, 1, 1, 0, 0])
# y_pred_2 = np.array([0.9, 0.8, 0.3, 0.7, 0.2])

# # Convert to TensorFlow tensors
# y_true_1_tf = tf.convert_to_tensor(y_true_1, dtype=tf.float32)
# y_pred_1_tf = tf.convert_to_tensor(y_pred_1, dtype=tf.float32)
# y_true_2_tf = tf.convert_to_tensor(y_true_2, dtype=tf.float32)
# y_pred_2_tf = tf.convert_to_tensor(y_pred_2, dtype=tf.float32)

# # Test with beta=1 (standard F1 score)
# print("Example 1 with beta=1:")
# print(f1B_score(y_true_1_tf, y_pred_1_tf, beta=1))

# print("\nExample 2 with beta=1:")
# print(f1B_score(y_true_2_tf, y_pred_2_tf, beta=1))

# # Test with beta=2 (emphasizes recall)
# print("\nExample 1 with beta=2:")
# print(f1B_score(y_true_1_tf, y_pred_1_tf, beta=2))

# print("\nExample 2 with beta=2:")
# print(f1B_score(y_true_2_tf, y_pred_2_tf, beta=2))

# # Let's manually calculate an F1 score for comparison
# def calculate_manual_f1(y_true, y_pred, threshold=0.5):
#     predictions = (y_pred >= threshold).astype(int)
#     tp = np.sum((predictions == 1) & (y_true == 1))
#     fp = np.sum((predictions == 1) & (y_true == 0))
#     fn = np.sum((predictions == 0) & (y_true == 1))
    
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#     return f1, precision, recall

# # Calculate traditional F1 scores using a threshold
# threshold = 0.5
# f1_1, prec_1, rec_1 = calculate_manual_f1(y_true_1, y_pred_1, threshold)
# f1_2, prec_2, rec_2 = calculate_manual_f1(y_true_2, y_pred_2, threshold)

# print("\nTraditional F1 scores with threshold 0.5:")
# print(f"Example 1: F1={f1_1:.4f} (Precision={prec_1:.4f}, Recall={rec_1:.4f})")
# print(f"Example 2: F1={f1_2:.4f} (Precision={prec_2:.4f}, Recall={rec_2:.4f})")


# Create the base model 
base_model = tf.keras.applications.Xception(
    include_top=False, 
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
print("Base model output shape:", base_model.output_shape)



# Define Constants
EPOCHS_FINE_TUNE = 50  # Fine-tune with last 10 layers unfrozen
LEARNING_RATE = 1e-4


result = []

# Ensure labels are integers
train_df['target'] = train_df['target'].astype('str')
val_df['target'] = val_df['target'].astype('str')

# Check class distribution
print("Overall class distribution:")
print(train_df['target'].value_counts())

# Set up Stratified K-Fold to maintain class distribution
# skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)



# Create Image Generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="target",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="target",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

# Print class distribution
unique_train_classes = np.unique(train_generator.classes)
unique_val_classes = np.unique(val_generator.classes)
print(f"Unique training classes: {unique_train_classes}")
print(f"Unique validation classes: {unique_val_classes}")



# Compute class weights if both classes exist
if len(unique_train_classes) < 2:
    print("WARNING: Only one class found in training data. Using equal weights.")
    class_weights_dict = {0: 1.0, 1: 1.0}
else:
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_generator.classes
    )
    class_weights_dict = {int(c): float(w) for c, w in zip([0, 1], class_weights)}

print(f"Class weights: {class_weights_dict}")


# Freeze all layers initially
base_model.trainable = True

# Unfreeze last 10 layers for fine-tuning
print("\nUnfreezing last 10 layers for fine-tuning...")
for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers
    layer.trainable = True



# Add custom classification head with regularization
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)  # L2 regularization here
x = BatchNormalization(name="custom_bn_1")(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # And here
x = BatchNormalization(name="custom_bn_2")(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)


# Create model
model = Model(inputs=base_model.input, outputs=outputs)



# Recompile model with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(), 
             tf.keras.metrics.Recall(),
             f1B_score
             ]
)

# Record the start time
start_time = time.time()

print("\nStarting fine-tuning with last 10 layers unfrozen...")
history_fine_tune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=callbacks,
    class_weight=class_weights_dict,
)


# Record the end time
end_time = time.time()

# Calculate the time spent
time_spent = end_time - start_time


# Save Model
model.save(f"models/xception_isic2024_fold_PAUC.keras")
model.save(f"models/xception_isic2024_fold_PAUC.h5")


# Store Results
result.append({
    "val_accuracy": history_fine_tune.history["val_accuracy"],
    "val_loss": history_fine_tune.history["val_loss"],
    "val_recall": history_fine_tune.history.get("val_recall"),
    "val_auc": history_fine_tune.history.get("val_auc"),
    "val_F1_score": history_fine_tune.history.get("val_f1B_score"),
})


print(f"Time of computation: {time_spent}")


# Print the time in a readable format
hours, remainder = divmod(time_spent, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\nFine-tuning completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")



###save the model

history_dict= history_fine_tune.history
history_df = pd.DataFrame(history_fine_tune.history)

path ="training_history.csv"
history_df.to_csv(path)



print(history_df.tail(10))
print("Training history saved to training_history.csv")


test_datagen = ImageDataGenerator(rescale=1.0/255)
IMG_SIZE= 299
test_df['target']= test_df['target'].astype('str')
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='target',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False  # Keep False for predictions
)


y_pred = model.predict(test_generator)



# Make predictions
predictions = model.predict(test_generator)
print("Raw prediction probabilities (first 5):", predictions[:5])

# Convert to class predictions (0 or 1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
print("Predicted classes (first 5):", predicted_classes[:5])

# Get true labels
y_true = test_generator.classes

# Calculate and print basic metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

accuracy = accuracy_score(y_true, predicted_classes)
precision = precision_score(y_true, predicted_classes)
recall = recall_score(y_true, predicted_classes)
auc_score = roc_auc_score(y_true, predictions)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC: {auc_score:.4f}")




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

# Define sensitivity (TPR) range
min_sensitivity = 0.8
max_sensitivity = 1.0

# Get indices where TPR is in desired range
indices = np.where((tpr >= min_sensitivity) & (tpr <= max_sensitivity))[0]

# Handle edge cases
if len(indices) < 2:
    print("Warning: Not enough points in sensitivity range. Expanding...")
    min_sensitivity = 0.7
    indices = np.where((tpr >= min_sensitivity) & (tpr <= max_sensitivity))[0]
    if len(indices) < 2:
        idx_min = np.argmin(np.abs(tpr - min_sensitivity))
        idx_max = np.argmin(np.abs(tpr - max_sensitivity))
        if idx_min == idx_max:
            if idx_min > 0:
                idx_min -= 1
            else:
                idx_max += 1
        indices = np.array([idx_min, idx_max])

# Sort indices
indices = np.sort(indices)

# Compute actual range
actual_min_tpr = np.min(tpr[indices])
actual_max_tpr = np.max(tpr[indices])
tpr_range = actual_max_tpr - actual_min_tpr

# Slice FPR and TPR in region of interest
fpr_partial = fpr[indices]
tpr_partial = tpr[indices]

# Compute regular partial AUC
partial_auc_value = auc(fpr_partial, tpr_partial)

# Compute baseline area under horizontal line y = 0.8
baseline_area = 0.8 * (fpr_partial[-1] - fpr_partial[0])

# Compute area above 0.8 line (AIC)
partial_aic_value = partial_auc_value - baseline_area
normalized_partial_aic = partial_aic_value / (0.2 * (fpr_partial[-1] - fpr_partial[0])) if (fpr_partial[-1] - fpr_partial[0]) > 0 else 0.0

# Print results
print(f"Using sensitivity (TPR) range: {actual_min_tpr:.3f}–{actual_max_tpr:.3f}")
print(f"Partial AUC: {partial_auc_value:.4f}")
print(f"Area between curve and 0.8 line (AIC): {partial_aic_value:.4f}")
print(f"Normalized AIC (0-1): {normalized_partial_aic:.4f}")

# Plot ROC and shaded region above 0.8 line
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')

# Highlight region between TPR curve and TPR=0.8 line
plt.fill_between(fpr_partial, 0.8, tpr_partial, 
                 where=(tpr_partial >= 0.8), 
                 color='lightblue', alpha=0.5,
                 label=f'AIC (TPR > 0.8) = {normalized_partial_aic:.3f}')

# Horizontal line at 0.8
plt.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, label='TPR = 0.8')

# Mark point where TPR = 0.8 (closest)
sensitivity_80_idx = np.argmin(np.abs(tpr - 0.8))
fpr_at_80 = fpr[sensitivity_80_idx]
tpr_at_80 = tpr[sensitivity_80_idx]
plt.plot([fpr_at_80], [tpr_at_80], 'ro', ms=8)
plt.annotate(f'FPR: {fpr_at_80:.3f}', 
             xy=(fpr_at_80, tpr_at_80),
             xytext=(fpr_at_80 + 0.1, tpr_at_80 - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10)

# Plot random classifier line
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

# Final plot config
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve with AIC (Above 80% Sensitivity Line)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_aic_above_80.png', dpi=300)
plt.show()


# Make sure y_true is properly formatted as integers
y_true = test_df["target"].astype(int).values

# Convert probabilities to binary predictions if not already done
y_pred_binary = (predictions > 0.5).astype(int).flatten()

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred_binary)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
plt.savefig(os.path.join('confusion_matrix.png'), dpi=300)
plt.close()

# Open and display the saved image
img = Image.open(save_path)
img.show()

# Print confusion matrix values
print("Confusion Matrix:")
print(f"True Negative: {cm[0,0]}")
print(f"False Positive: {cm[0,1]}")
print(f"False Negative: {cm[1,0]}")
print(f"True Positive: {cm[1,1]}")

# Calculate sensitivity and specificity
sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)

print(f"\nSensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")



# Plot training history
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# Set up a sophisticated minimal style
plt.style.use('seaborn-v0_8-whitegrid')
history = pd.read_csv("/kaggle/working/training_history.csv")


# Custom color palette with soft, sophisticated colors
colors = {
    'train': '#8fb9aa',  # soft teal
    'val': '#e8c2ca',    # soft pink
    'bg': '#FFFFFF',     # light background
    'grid': '#dce1e3',   # soft grid lines
    'text': '#2d3436'    # dark text
}

# Create figure with proper spacing
fig = plt.figure(constrained_layout=True)
spec = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)  # Increased spacing for titles
# Add a single title to the entire figure
fig.suptitle('Model Performance', fontsize=16, fontweight='bold', y=0.00)

# Plot Accuracy
ax1 = fig.add_subplot(spec[0, 0])
ax1.plot(history_dict['accuracy'], color=colors['train'], label='Train', marker='o', markersize=4, linewidth=2)
ax1.plot(history_dict['val_accuracy'], color=colors['val'], label='Validation', marker='o', markersize=4, linewidth=2)
ax1.set_xlabel('Epochs', fontsize=10)
ax1.set_ylabel('Accuracy', fontsize=10)
ax1.set_title('Accuracy', fontsize=12, pad=10)  # Add title above plot
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='lower right', frameon=False)  # Position legend at bottom right
ax1.set_ylim(0, 1.1)

# Plot Recall
ax2 = fig.add_subplot(spec[0, 1])
ax2.plot(history_dict['recall'], color=colors['train'], label='Train', marker='o', markersize=4, linewidth=2)
ax2.plot(history_dict['val_recall'], color=colors['val'], label='Validation', marker='o', markersize=4, linewidth=2)
ax2.set_xlabel('Epochs', fontsize=10)
ax2.set_ylabel('Recall', fontsize=10)
ax2.set_title('Recall', fontsize=12, pad=10)  # Add title above plot
ax2.legend(loc='lower right', frameon=False)  # Position legend at bottom right
ax2.set_ylim(0, 1.1)

# Plot AUC
ax3 = fig.add_subplot(spec[1, 0])
ax3.plot(history_dict['auc'], color=colors['train'], label='Train',  linewidth=2)
ax3.plot(history_dict['val_auc'], color=colors['val'], label='Validation',  linewidth=2)
ax3.set_xlabel('Epochs', fontsize=10)
ax3.set_ylabel('AUC', fontsize=10)
ax3.set_title('AUC', fontsize=12, pad=10)  # Add title above plot
ax3.legend(loc='lower right', frameon=False)  # Position legend at bottom right
ax3.set_ylim(0, 1.1)


# Plot Loss
ax4 = fig.add_subplot(spec[1, 1])
ax4.plot(history_dict['loss'], color=colors['train'], label='Train', marker='o', markersize=4, linewidth=2)
ax4.plot(history_dict['val_loss'], color=colors['val'], label='Validation', marker='o', markersize=4, linewidth=2)
ax4.set_xlabel('Epochs', fontsize=10)
ax4.set_ylabel('Loss', fontsize=10)
ax4.set_title('Loss', fontsize=12, pad=10)  # Add title above plot
ax4.legend(loc='lower right', frameon=False)  # Position legend at bottom right
ax4.set_ylim(0, .25)

# Remove the corner text labels since we now have proper titles
# Save figure with high quality
plt.savefig('plots/sisi2024_metrics.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])





