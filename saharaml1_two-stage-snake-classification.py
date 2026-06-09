# Setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF warnings

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Load metadata
df = pd.read_csv('/kaggle/input/snakeclef2022/SnakeCLEF2022-TrainMetadata.csv')

print(f"\nMetadata: {len(df):,} records")
df.head()


# Extract genus
df['genus'] = df['binomial_name'].str.split().str[0]

# Filter for North and Central American countries
north_central_american = ['US', 'MX', 'CA', 'CR', 'PA', 'GT', 'HN', 'NI', 'SV', 'BZ']
mask = df['code'].isin(north_central_american)
df = df[mask].reset_index(drop=True)

# ALL North/Central American venomous Species list
venomous_list = [
    'Crotalus',      # Rattlesnakes
    'Sistrurus',     # Pygmy/Massasauga rattlesnakes
    'Agkistrodon',   # Copperheads/Cottonmouths
    'Micrurus',      # Coral snakes
    'Bothrops',      # Fer-de-lance
    'Lachesis',      # Bushmaster
    'Porthidium',    # Hognosed pit vipers
    'Cerrophidion',  # Montane pit vipers
    'Ophryacus',     # Mexican horned pit vipers
]

df['is_venomous'] = df['genus'].isin(venomous_list).astype(int)

# Build full image paths
IMAGE_DIR = '/kaggle/input/snakeclef2022/SnakeCLEF2022-medium_size/SnakeCLEF2022-medium_size'
df['image_path'] = IMAGE_DIR + '/' + df['file_path']


# Preview venomous snake records
df[df['is_venomous'] == 1].head(5)


# Quick stats
print("NORTH AND CENTRAL AMERICAN SNAKE DATASET (ALL)")
print(f"Total: {len(df):,} images")
print(f"Total Species: {df['binomial_name'].nunique()}")
df['is_venomous'].value_counts()


# Venomous Species distribution
df[df.is_venomous == 1].genus.value_counts()


most_common_names = {
    # Venomous
    'Crotalus': 'Rattlesnake',
    'Micrurus': 'Coral Snake',
    'Agkistrodon': 'Copperhead',
    'Sistrurus': 'Pygmy Rattlesnake',
    'Bothrops': 'Fer-de-lance',
    'Porthidium': 'Hognosed Pit Viper',
    'Cerrophidion': 'Montane Pit Viper',
    'Ophryacus': 'Horned Pit Viper',
    'Lachesis': 'Bushmaster',
    # Non-venomous (common ones)
    'Thamnophis': 'Garter Snake',
    'Lampropeltis': 'Kingsnake',
    'Pantherophis': 'Rat Snake',
    'Nerodia': 'Water Snake',
    'Pituophis': 'Gopher Snake',
}


# Visualize Venomous Group Distribution
venomous_counts = df[df.is_venomous == 1].genus.value_counts()
labels = [most_common_names.get(g, g) for g in venomous_counts.index]

plt.figure(figsize=(10, 5))
plt.bar(labels, venomous_counts.values)
plt.xlabel('Snake Group')
plt.ylabel('Number of Images')
plt.title('Venomous Snake Images by Group')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Class distribution
counts = df['is_venomous'].value_counts()
labels = ['Non-Venomous', 'Venomous']

plt.figure(figsize=(6, 4))
plt.bar(labels, [counts[0], counts[1]])
plt.ylabel('Number of Images')
plt.title('Dataset Class Distribution')
plt.show()

df['is_venomous'].value_counts()


# Sample images - 1 venomous, 1 non-venomous
from PIL import Image

venomous_sample = df[df.is_venomous == 1].iloc[0]
nonvenomous_sample = df[df.is_venomous == 0].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

v_name = most_common_names.get(venomous_sample['genus'], venomous_sample['genus'])
axes[0].imshow(Image.open(venomous_sample['image_path']))
axes[0].set_title(f"Venomous: {v_name}")
axes[0].axis('off')

nv_name = most_common_names.get(nonvenomous_sample['genus'], nonvenomous_sample['genus'])
axes[1].imshow(Image.open(nonvenomous_sample['image_path']))
axes[1].set_title(f"Non-Venomous: {nv_name}")
axes[1].axis('off')

plt.tight_layout()
plt.show()


# Data Cleaning - Remove corrupted images
# source: https://www.kaggle.com/code/parkjohnychae/check-and-remove-corrupted-files-from-the-metadata
valid_indices = []
corrupt_count = 0
for i, path in enumerate(df['image_path'].values):
    try:
        img = tf.io.read_file(path)
        tf.image.decode_jpeg(img, channels=3)
        valid_indices.append(i)
    except:
        corrupt_count += 1

    if (i + 1) % 20000 == 0:
        print(f"  Checked {i+1:,}/{len(df):,}...")
        
# Keep only valid images
df = df.iloc[valid_indices].reset_index(drop=True)
print(f"\nRemoved {corrupt_count} corrupted images")
print(f"Clean dataset: {len(df):,} images")


# 80/20 Split
# Train/val split (stratified to maintain class balance)
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['is_venomous'], 
    random_state=42
)

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
train_df['is_venomous'].value_counts()


# Visualize train/val split
fig, ax = plt.subplots(figsize=(8, 4))

train_counts = train_df['is_venomous'].value_counts().sort_index()
val_counts = val_df['is_venomous'].value_counts().sort_index()

x = np.arange(2)
width = 0.35

bars1 = ax.bar(x - width/2, train_counts.values, width, label='Train')
bars2 = ax.bar(x + width/2, val_counts.values, width, label='Validation')

ax.set_xticks(x)
ax.set_xticklabels(['Non-Venomous', 'Venomous'])
ax.set_ylabel('Number of Images')
ax.set_title('Train/Validation Split (80/20)')
ax.legend()

# Add count labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_df['is_venomous'].values
)

# convert to dictionary format that Keras exopects
class_weight_dict = {0: class_weights[0], # non-venomous weight
                     1: class_weights[1]} # venomous weight

print(f"{class_weight_dict}")


# Chapter 13
#IMG_SIZE = 64
IMG_SIZE = 224
BATCH_SIZE = 32

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img / 255.0, label


from tensorflow.data import Dataset

# paths and labels
train_paths = train_df['image_path'].values
train_labels = train_df['is_venomous'].values
val_paths = val_df['image_path'].values
val_labels = val_df['is_venomous'].values

# Create datasets
# Training set
train_ds = Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.map(load_image)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(1)

# Validation set
val_ds = Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_image)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(1)


# Baseline CNN (from scratch)
from tensorflow import keras

baseline_model = keras.Sequential([
    # First conv block
    keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', 
                       input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.MaxPooling2D(pool_size=2),
    
    # Second conv block
    keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    
    # Third conv block
    keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=2),
    
    # Classifier head
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')  # Binary output
])

baseline_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

baseline_model.summary()


# Train baseline CNN
history = baseline_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    verbose=1
)


# Get predictions
y_pred_probs = baseline_model.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = val_df['is_venomous'].values

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-venomous', 'Venomous'],
            yticklabels=['Non-venomous', 'Venomous'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Baseline CNN - Confusion Matrix')
plt.tight_layout()
plt.show()

print(classification_report(y_true, y_pred, 
                           target_names=['Non-venomous', 'Venomous']))


# Transfer Learning with MobileNetV2
from tensorflow.keras.applications import MobileNetV2

# Load pretrained model (without top classification layer)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model
base_model.trainable = False


mobilenet_model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=1, activation='sigmoid')
])

mobilenet_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

mobilenet_model.summary()


# Train MobileNetV2
history_mobilenet = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weight_dict,
    verbose=1
)


y_pred_probs = mobilenet_model.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = val_df['is_venomous'].values

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-venomous', 'Venomous'],
            yticklabels=['Non-venomous', 'Venomous'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('MobileNetV2 - Confusion Matrix')
plt.tight_layout()
plt.show()

print(classification_report(y_true, y_pred, 
                           target_names=['Non-venomous', 'Venomous']))


# EfficientNet expects pixels in [0-255] range, not [0-1]
def load_image_efficientnet(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label

# Create datasets for EfficientNet
# Training set
train_ds_eff = Dataset.from_tensor_slices((train_paths, train_labels))
train_ds_eff = train_ds_eff.shuffle(1000)
train_ds_eff = train_ds_eff.map(load_image_efficientnet)
train_ds_eff = train_ds_eff.batch(BATCH_SIZE)
train_ds_eff = train_ds_eff.prefetch(1)

# Validation set
val_ds_eff = Dataset.from_tensor_slices((val_paths, val_labels))
val_ds_eff = val_ds_eff.map(load_image_efficientnet)
val_ds_eff = val_ds_eff.batch(BATCH_SIZE)
val_ds_eff = val_ds_eff.prefetch(1)


from tensorflow.keras.applications import EfficientNetB0

# Load pretrained model (without top classification layer)
efficient_base = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model
efficient_base.trainable = False


efficientnet_model = keras.Sequential([
    efficient_base,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=1, activation='sigmoid')
])

efficientnet_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

efficientnet_model.summary()


# Train EfficientNet
history_efficient = efficientnet_model.fit(
    train_ds_eff,
    validation_data=val_ds_eff,
    epochs=5,
    class_weight=class_weight_dict,
    verbose=1
)


y_pred_probs = efficientnet_model.predict(val_ds_eff)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = val_df['is_venomous'].values

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-venomous', 'Venomous'],
            yticklabels=['Non-venomous', 'Venomous'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('EfficientNetB0 - Confusion Matrix')
plt.tight_layout()
plt.show()

print(classification_report(y_true, y_pred, target_names=['Non-venomous', 'Venomous']))


from sklearn.metrics import roc_curve, auc

# Get predictions from each model
# Note: baseline and mobilenet use val_ds, efficientnet uses val_ds_eff
y_true = val_df['is_venomous'].values

pred_baseline = baseline_model.predict(val_ds).flatten()
pred_mobilenet = mobilenet_model.predict(val_ds).flatten()
pred_efficient = efficientnet_model.predict(val_ds_eff).flatten()

# Calculate ROC curves
fpr_base, tpr_base, _ = roc_curve(y_true, pred_baseline)
fpr_mobile, tpr_mobile, _ = roc_curve(y_true, pred_mobilenet)
fpr_efficient, tpr_efficient, _ = roc_curve(y_true, pred_efficient)

# Calculate AUC scores
auc_base = auc(fpr_base, tpr_base)
auc_mobile = auc(fpr_mobile, tpr_mobile)
auc_efficient = auc(fpr_efficient, tpr_efficient)


# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_base, tpr_base, label=f'Baseline CNN (AUC = {auc_base:.3f})', linewidth=2)
plt.plot(fpr_mobile, tpr_mobile, label=f'MobileNetV2 (AUC = {auc_mobile:.3f})', linewidth=2)
plt.plot(fpr_efficient, tpr_efficient, label=f'EfficientNetB0 (AUC = {auc_efficient:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Sample predictions
import random

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, idx in enumerate(random.sample(range(len(val_df)), 6)):
    img_path = val_df.iloc[idx]['image_path']
    true_label = val_df.iloc[idx]['is_venomous']
    
    # Load and predict
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    pred_prob = efficientnet_model.predict(tf.expand_dims(img, 0), verbose=0)[0][0]
    
    # Display
    true_text = 'Venomous' if true_label == 1 else 'Non-venomous'
    pred_text = 'Venomous' if pred_prob > 0.5 else 'Non-venomous'
    color = 'green' if (pred_prob > 0.5) == true_label else 'red'
    
    axes.flat[i].imshow(img.numpy().astype('uint8'))
    axes.flat[i].axis('off')
    axes.flat[i].set_title(f'Actual: {true_text}\nPredicted: {pred_text} ({pred_prob:.1%})', 
                           color=color, fontsize=10)

plt.suptitle('EfficientNetB0 Sample Predictions', fontweight='bold')
plt.tight_layout()
plt.show()


# Filter to venomous snakes only
venomous_df = df[df['is_venomous'] == 1].copy()

print(f"Total images: {len(venomous_df):,}")
print(f"Total species: {venomous_df['binomial_name'].nunique()}")
print(f"Groups: {venomous_df['genus'].nunique()}")


# Quick stats
venomous_df['binomial_name'].value_counts().describe()


# Species distribution (top 20 most common)
venomous_df['binomial_name'].value_counts().head(20).plot(kind='barh', figsize=(10, 8))
plt.xlabel('Number of Images')
plt.ylabel('Species')
plt.title('Top 20 Venomous Species by Image Count')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Step 1: Get all unique class_ids and sort them
unique_classes = sorted(venomous_df['class_id'].unique())

# Step 2: Create a mapping dictionary {original_id: new_index}
# Example: {21: 0, 22: 1, 23: 2, ...}
class_to_idx = {}
for i, class_id in enumerate(unique_classes):
    class_to_idx[class_id] = i

# Step 3: Create reverse mapping for decoding predictions later
# Example: {0: 21, 1: 22, 2: 23, ...}
idx_to_class = {}
for class_id, idx in class_to_idx.items():
    idx_to_class[idx] = class_id

# Step 4: Apply the mapping to create encoded labels
venomous_df['encoded_label'] = venomous_df['class_id'].map(class_to_idx)


# Compute class weights to handle imbalanced species
from sklearn.utils.class_weight import compute_class_weight

y_all = venomous_df['encoded_label'].values
class_weights = compute_class_weight('balanced', classes=np.unique(y_all), y=y_all)
class_weight_dict = dict(enumerate(class_weights))

print(f"Weight range: {min(class_weights):.2f} - {max(class_weights):.2f}")


# Train/val split
# 80/20
train_df, val_df = train_test_split(
    venomous_df, 
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(train_df):,}")
print(f"Validation samples: {len(val_df):,}")


# Visualize train/val split
fig, ax = plt.subplots(figsize=(6, 4))

counts = [len(train_df), len(val_df)]
labels = ['Train', 'Validation']

bars = ax.bar(labels, counts, color=['steelblue', 'darkorange'])

ax.set_ylabel('Number of Images')
ax.set_title('Stage 2: Train/Validation Split (80/20)')

# Add count labels on bars
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


# Augmentation function
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


# EfficientNet preprocessing
# paths and labels
train_paths_s2 = train_df['image_path'].values
train_labels_s2 = train_df['encoded_label'].values

val_paths_s2 = val_df['image_path'].values
val_labels_s2 = val_df['encoded_label'].values

# Create Datasets
# Training set
train_ds = Dataset.from_tensor_slices((train_paths_s2, train_labels_s2))
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.map(load_image_efficientnet)
train_ds = train_ds.map(augment_image)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(1)

# Validation set
val_ds = Dataset.from_tensor_slices((val_paths_s2, val_labels_s2))
val_ds = val_ds.map(load_image_efficientnet)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(1)


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

NUM_CLASSES = venomous_df['encoded_label'].nunique()

effnet_base = EfficientNetB0(weights='imagenet', 
                            include_top=False, 
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model -all layers
effnet_base.trainable = False


species_model = keras.Sequential([
    effnet_base,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(units=256, activation='relu', kernel_initializer='he_normal'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
])

species_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

species_model.summary()


# Train EfficientNet for Species Classification
species_history = species_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weight_dict
)


# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
axes[0].plot(species_history.history['accuracy'], label='Train')
axes[0].plot(species_history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Loss
axes[1].plot(species_history.history['loss'], label='Train')
axes[1].plot(species_history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.show()


# Unfreeze the base model, just the last 10 layers
effnet_base.trainable = True
for layer in effnet_base.layers[:-10]:
    layer.trainable = False

# Recompile with smaller learning rate
species_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

species_model.summary()


# Train with fine-tuning
fine_tune_history = species_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight_dict
)


# Combine histories
full_acc = species_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
full_loss = species_history.history['val_loss'] + fine_tune_history.history['val_loss']

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(full_acc)
plt.axvline(x=19, color='r', linestyle='--', label='Fine-tuning start')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(full_loss)
plt.axvline(x=19, color='r', linestyle='--', label='Fine-tuning start')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()


# Collect predictions
val_predictions = []
val_labels = []

for images, labels in val_ds:
    preds = species_model.predict(images, verbose=0)
    val_predictions.extend(np.argmax(preds, axis=1))
    val_labels.extend(labels.numpy())

val_predictions = np.array(val_predictions)
val_labels = np.array(val_labels)


# Top 10 Species Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

unique, counts = np.unique(val_labels, return_counts=True)
top10_indices = unique[np.argsort(counts)[-10:]]

label_to_species = dict(zip(venomous_df['encoded_label'], venomous_df['binomial_name']))
top10_names = [label_to_species[l].split()[-1][:12] for l in top10_indices]

mask = np.isin(val_labels, top10_indices)
cm = confusion_matrix(val_labels[mask], val_predictions[mask], labels=top10_indices)

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(cm, display_labels=top10_names)
disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')
plt.title('Top 10 Species')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))

for images, labels in val_ds.shuffle(1000).take(1):
    preds = species_model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        img = images[i].numpy()
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        
        actual = label_to_species.get(labels[i].numpy(), "Unknown").split()[-1]
        predicted = label_to_species.get(pred_labels[i], "Unknown").split()[-1]
        correct = pred_labels[i] == labels[i].numpy()
        color = 'green' if correct else 'red'
        
        plt.title(f"Actual: {actual}\nPredicted: {predicted} ({confidences[i]:.1%})", 
                  fontsize=10, color=color)
        plt.axis('off')

plt.suptitle('Sample Predictions', fontsize=14)
plt.tight_layout()
plt.show()

