import os, random, shutil, csv, math, itertools, pathlib, time
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(True)
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import models, Model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, top_k_accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import json
from datetime import datetime

sns.set()

from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.losses import CategoricalCrossentropy


# Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_CLASSES = 4000  # Updated for your dataset
IMG_SIZE = (64, 64 )
BATCH_SIZE = 32  # Reduced for memory efficiency with 4000 classes
EMBEDDING_DIM = 512  # Larger embedding for more classes

base_dir = "/kaggle/input/11-785-fall-20-homework-2-part-2/classification_data"
train_dir = base_dir + "/train_data"
test_dir = base_dir + "/test_data"
val_dir = base_dir + "/val_data"

# Create experiment directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
# os.makedirs(experiment_dir, exist_ok=True)
experiment_dir = os.path.join("/kaggle/working", f"experiment_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

# Save configuration
config = {
    "num_classes": NUM_CLASSES,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "embedding_dim": EMBEDDING_DIM,
    "timestamp": timestamp
}
with open(os.path.join(experiment_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

# =============================================================================
# GPU SETUP
# =============================================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) available: {len(gpus)}")
else:
    print("No GPU found - training will be slow!")

mixed_precision.set_global_policy('mixed_float16')
print(f"Mixed precision policy: {mixed_precision.global_policy()}")
# =============================================================================
# DATA LOADING
# =============================================================================
print("Loading datasets...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    label_mode='categorical'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    label_mode='categorical'
)

# Save class names
class_names = train_ds.class_names
with open(os.path.join(experiment_dir, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=4)
print(f"Total classes: {len(class_names)}")

# Preprocessing
train_ds = train_ds.map(lambda x, y: (x, y))
val_ds = val_ds.map(lambda x, y: (x, y))
test_ds = test_ds.map(lambda x, y: (x, y))

# Optimize pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

#data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),        # horizontal flip
    layers.RandomRotation(0.1),             # rotate ±10%
    layers.RandomZoom(0.1),                 # zoom in/out ±10%
    layers.RandomContrast(0.1),             # adjust contrast ±10%
    layers.RandomTranslation(0.1, 0.1)      # translate ±10% horizontally and vertically
], name="data_augmentation")
# =============================================================================
# MODEL DEFINITION
# =============================================================================
def l2_norm_layer(x):
    return tf.nn.l2_normalize(x, axis=1)

def create_model(num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM):
    """Create face recognition model with embedding layer"""
    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    x = data_augmentation(inputs)

    # Preprocessing for ResNet
    x = preprocess_input(x)
    
    base_model = ResNet50V2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Embedding layer (unnormalized)
    x = layers.Dense(embedding_dim, activation=None, name="embedding_dense")(x)
    x = layers.BatchNormalization()(x)
    
    # L2 normalized embedding
    embedding = layers.Lambda(l2_norm_layer, name="l2_embedding", output_shape=(embedding_dim,))(x)
    
    # Classification head
    output = layers.Dense(num_classes, activation="softmax", dtype='float32', name="softmax_output")(embedding)
    
    model = Model(inputs=inputs, outputs=output, name="FaceRecognitionModel")
    return model, base_model

model, base_model = create_model()
print(f"Model created with {NUM_CLASSES} output classes")

# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================
class DetailedMetricsCallback(keras.callbacks.Callback):
    """Save detailed metrics after each epoch"""
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.metrics_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_metrics = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **logs
        }
        self.metrics_history.append(epoch_metrics)
        
        # Save after each epoch
        with open(os.path.join(self.log_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        
        print(f"\nEpoch {epoch+1} metrics saved.")

# =============================================================================
# STAGE 1: TRAIN FROZEN BASE
# =============================================================================
print("\n" + "="*80)
print("STAGE 1: Training with frozen base model")
print("="*80 + "\n")

stage1_dir = os.path.join(experiment_dir, "stage1")
os.makedirs(stage1_dir, exist_ok=True)

optimizer_stage1 = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

model.compile(
    optimizer=optimizer_stage1,
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy", 
             tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy'),
             tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top10_accuracy')]
)

callbacks_stage1 = [
    ModelCheckpoint(
        os.path.join(stage1_dir, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(stage1_dir, "model_epoch_{epoch:02d}_acc_{val_accuracy:.4f}.keras"),
        save_freq='epoch',
        verbose=0
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(
        os.path.join(stage1_dir, "training_log.csv"),
        separator=',',
        append=False
    ),
    TensorBoard(
        log_dir=os.path.join(stage1_dir, "tensorboard"),
        histogram_freq=1,
        write_graph=True
    ),
    DetailedMetricsCallback(stage1_dir)
]

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=callbacks_stage1,
    verbose=1
)

# Save final model and weights
model.save(os.path.join(stage1_dir, "final_model.keras"))
model.save_weights(os.path.join(stage1_dir, "final_model.weights.h5"))
print(f"Stage 1 complete. Models saved to {stage1_dir}")

# =============================================================================
# STAGE 2: FINE-TUNING
# =============================================================================
print("\n" + "="*80)
print("STAGE 2: Fine-tuning with unfrozen layers")
print("="*80 + "\n")

stage2_dir = os.path.join(experiment_dir, "stage2")
os.makedirs(stage2_dir, exist_ok=True)

# Load best model from stage 1
model = tf.keras.models.load_model(
    os.path.join(stage1_dir, "best_model.keras"),
    custom_objects={"l2_norm_layer": l2_norm_layer}
)

# Unfreeze top layers
base_model = model.get_layer("resnet50v2")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False
for i, layer in enumerate(model.layers):
    print(i, layer.name, type(layer))

print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

optimizer_stage2 = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)

model.compile(
    optimizer=optimizer_stage2,
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy",
             tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy'),
             tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top10_accuracy')]
)

callbacks_stage2 = [
    ModelCheckpoint(
        os.path.join(stage2_dir, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(stage2_dir, "model_epoch_{epoch:02d}_acc_{val_accuracy:.4f}.keras"),
        save_freq='epoch',
        verbose=0
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    CSVLogger(
        os.path.join(stage2_dir, "training_log.csv"),
        separator=',',
        append=False
    ),
    TensorBoard(
        log_dir=os.path.join(stage2_dir, "tensorboard"),
        histogram_freq=1
    ),
    DetailedMetricsCallback(stage2_dir)
]

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    callbacks=callbacks_stage2,
    verbose=1
)

# Save final model
model.save(os.path.join(stage2_dir, "final_model.keras"))
model.save_weights(os.path.join(stage2_dir, "final_model.weights.h5"))
print(f"Stage 2 complete. Models saved to {stage2_dir}")


# =============================================================================
# FINAL EVALUATION
# =============================================================================
print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80 + "\n")

# Load best model
best_model_path = os.path.join(stage2_dir, "best_model.keras")
model = tf.keras.models.load_model(best_model_path, custom_objects={"l2_norm_layer": l2_norm_layer})

# Get predictions
y_true_list, y_pred_list = [], []
for batch_x, batch_y in test_ds:
    preds = model.predict(batch_x, verbose=0)
    y_true_list.append(batch_y.numpy())
    y_pred_list.append(preds)

y_true = np.concatenate(y_true_list, axis=0)
y_pred = np.concatenate(y_pred_list, axis=0)
y_true_labels = np.argmax(y_true, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Compute metrics
acc = accuracy_score(y_true_labels, y_pred_labels)
top5_acc = top_k_accuracy_score(y_true_labels, y_pred_labels, k=5, labels=range(NUM_CLASSES))
top10_acc = top_k_accuracy_score(y_true_labels, y_pred_labels, k=10, labels=range(NUM_CLASSES))
f1 = f1_score(y_true_labels, y_pred_labels, average='macro')

confidences = np.max(y_pred, axis=1)
avg_confidence = np.mean(confidences)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")
print(f"Top-10 Accuracy: {top10_acc:.4f}")
print(f"Macro F1-score: {f1:.4f}")
print(f"Average Confidence: {avg_confidence:.4f}")

# Save final metrics
final_metrics = {
    "accuracy": float(acc),
    "top5_accuracy": float(top5_acc),
    "top10_accuracy": float(top10_acc),
    "f1_macro": float(f1),
    "avg_confidence": float(avg_confidence),
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(experiment_dir, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

# Save predictions
np.savez(
    os.path.join(experiment_dir, "predictions.npz"),
    y_true=y_true,
    y_pred=y_pred,
    y_true_labels=y_true_labels,
    y_pred_labels=y_pred_labels
)

# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_combined_history(h1, h2, save_dir):
    """Plot training history from both stages"""
    combined = {
        k: h1.history.get(k, []) + h2.history.get(k, [])
        for k in set(h1.history.keys()).union(h2.history.keys())
    }
    epochs = range(1, len(combined["accuracy"]) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(epochs, combined["accuracy"], label="Train")
    axes[0, 0].plot(epochs, combined["val_accuracy"], label="Validation")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(epochs, combined["loss"], label="Train")
    axes[0, 1].plot(epochs, combined["val_loss"], label="Validation")
    axes[0, 1].set_title("Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()a
    axes[0, 1].grid(True)
    
    # Top-5 Accuracy
    axes[1, 0].plot(epochs, combined["top5_accuracy"], label="Train")
    axes[1, 0].plot(epochs, combined["val_top5_accuracy"], label="Validation")
    axes[1, 0].set_title("Top-5 Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Top-10 Accuracy
    axes[1, 1].plot(epochs, combined["top10_accuracy"], label="Train")
    axes[1, 1].plot(epochs, combined["val_top10_accuracy"], label="Validation")
    axes[1, 1].set_title("Top-10 Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=200, bbox_inches='tight')
    plt.show()

plot_combined_history(history_stage1, history_stage2, experiment_dir)

# Confidence distribution
plt.figure(figsize=(8, 5))
plt.hist(confidences, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Model Confidence Scores")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.savefig(os.path.join(experiment_dir, "confidence_distribution.png"), dpi=200, bbox_inches='tight')
plt.show()

print(f"\n{'='*80}")
print(f"Training complete! All results saved to:")
print(f"{experiment_dir}")
print(f"{'='*80}\n")


import os, random, shutil, csv, math, itertools, pathlib, time
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(True)
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import models, Model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, top_k_accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import json
from datetime import datetime

sns.set()

from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.losses import CategoricalCrossentropy


# Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_CLASSES = 4000  # Updated for your dataset
IMG_SIZE = (64, 64)
BATCH_SIZE = 32  # Reduced for memory efficiency with 4000 classes
EMBEDDING_DIM = 512  # Larger embedding for more classes
EPOCH_STAGE1 = 10
EPOCH_STAGE2 = 20

base_dir = "/kaggle/input/11-785-fall-20-homework-2-part-2/classification_data"
# base_dir = "/facerec/data"
train_dir = base_dir + "/train_data"
test_dir = base_dir + "/test_data"
val_dir = base_dir + "/val_data"

# Create experiment directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
# os.makedirs(experiment_dir, exist_ok=True)
experiment_dir = os.path.join("/kaggle/working/", f"experiment_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

# Save configuration
config = {
    "num_classes": NUM_CLASSES,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "embedding_dim": EMBEDDING_DIM,
    "timestamp": timestamp
}
with open(os.path.join(experiment_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

# =============================================================================
# GPU SETUP
# =============================================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) available: {len(gpus)}")
else:
    print("No GPU found - training will be slow!")

mixed_precision.set_global_policy('mixed_float16')
print(f"Mixed precision policy: {mixed_precision.global_policy()}")
# =============================================================================
# DATA LOADING
# =============================================================================
print("Loading datasets...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    label_mode='categorical'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    label_mode='categorical'
)

# Save class names
class_names = train_ds.class_names
with open(os.path.join(experiment_dir, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=4)
print(f"Total classes: {len(class_names)}")

# Preprocessing
train_ds = train_ds.map(lambda x, y: (x, y))
val_ds = val_ds.map(lambda x, y: (x, y))
test_ds = test_ds.map(lambda x, y: (x, y))

# Optimize pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

#data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),        # horizontal flip
    layers.RandomRotation(0.1),             # rotate ±10%
    layers.RandomZoom(0.1),                 # zoom in/out ±10%
    layers.RandomContrast(0.1),             # adjust contrast ±10%
    layers.RandomTranslation(0.1, 0.1)      # translate ±10% horizontally and vertically
], name="data_augmentation")
# =============================================================================
# MODEL DEFINITION
# =============================================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
def l2_norm_layer(x):
    return tf.nn.l2_normalize(x, axis=1)

def create_model(num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM):
    """Create face recognition model with embedding layer"""
    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    x = data_augmentation(inputs)

    # Preprocessing for ResNet
    x = preprocess_input(x)
    
    base_model = ResNet50V2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Embedding layer (unnormalized)
    x = layers.Dense(embedding_dim, activation=None, name="embedding_dense")(x)
    x = layers.BatchNormalization()(x)
    
    # L2 normalized embedding
    embedding = layers.Lambda(l2_norm_layer, name="l2_embedding", output_shape=(embedding_dim,))(x)
    
    # Classification head
    output = layers.Dense(num_classes, activation="softmax", dtype='float32', name="softmax_output")(embedding)
    
    model = Model(inputs=inputs, outputs=output, name="FaceRecognitionModel")
    return model, base_model

model, base_model = create_model()
print(f"Model created with {NUM_CLASSES} output classes")

# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================
class DetailedMetricsCallback(keras.callbacks.Callback):
    """Save detailed metrics after each epoch"""
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.metrics_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_metrics = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **logs
        }
        self.metrics_history.append(epoch_metrics)
        
        # Save after each epoch
        with open(os.path.join(self.log_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        
        print(f"\nEpoch {epoch+1} metrics saved.")

# =============================================================================
# STAGE 1: TRAIN FROZEN BASE
# =============================================================================
print("\n" + "="*80)
print("STAGE 1: Training with frozen base model")
print("="*80 + "\n")

stage1_dir = os.path.join(experiment_dir, "stage1")
os.makedirs(stage1_dir, exist_ok=True)

optimizer_stage1 = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

model.compile(
    optimizer=optimizer_stage1,
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy", 
             tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy'),
             tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top10_accuracy')]
)

callbacks_stage1 = [
    ModelCheckpoint(
        os.path.join(stage1_dir, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(stage1_dir, "model_epoch_{epoch:02d}_acc_{val_accuracy:.4f}.keras"),
        save_freq='epoch',
        verbose=0
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(
        os.path.join(stage1_dir, "training_log.csv"),
        separator=',',
        append=False
    ),
    TensorBoard(
        log_dir=os.path.join(stage1_dir, "tensorboard"),
        histogram_freq=1,
        write_graph=True
    ),
    DetailedMetricsCallback(stage1_dir)
]

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCH_STAGE1,
    callbacks=callbacks_stage1,
    verbose=20
)

# Save final model and weights
model.save(os.path.join(stage1_dir, "final_model.keras"))
model.save_weights(os.path.join(stage1_dir, "final_model_stage1.weights.h5"))
print(f"Stage 1 complete. Models saved to {stage1_dir}")

# =============================================================================
# STAGE 2: FINE-TUNING
# =============================================================================
print("\n" + "="*80)
print("STAGE 2: Fine-tuning with unfrozen layers")
print("="*80 + "\n")

stage2_dir = os.path.join(experiment_dir, "stage2")
os.makedirs(stage2_dir, exist_ok=True)

# Load best model from stage 1
model = tf.keras.models.load_model(
    os.path.join(stage1_dir, "best_model.keras"),
    custom_objects={"l2_norm_layer": l2_norm_layer}
)

# Unfreeze top layers
# base_model = model.get_layer("resnet50v2")
# base_model.trainable = True

try:
    base_model = model.get_layer("resnet50v2")
except ValueError:
    print("ResNet50V2 base not found, scanning layers...")
    for layer in model.layers:
        if isinstance(layer, tf.keras.applications.ResNet50V2):
            base_model = layer
            break

for layer in base_model.layers[:-50]:
    layer.trainable = False
for i, layer in enumerate(model.layers):
    print(i, layer.name, type(layer))

print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

optimizer_stage2 = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)

model.compile(
    optimizer=optimizer_stage2,
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy",
             tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy'),
             tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top10_accuracy')]
)

callbacks_stage2 = [
    ModelCheckpoint(
        os.path.join(stage2_dir, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(stage2_dir, "model_epoch_{epoch:02d}_acc_{val_accuracy:.4f}.keras"),
        save_freq='epoch',
        verbose=0
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    CSVLogger(
        os.path.join(stage2_dir, "training_log.csv"),
        separator=',',
        append=False
    ),
    TensorBoard(
        log_dir=os.path.join(stage2_dir, "tensorboard"),
        histogram_freq=1
    ),
    DetailedMetricsCallback(stage2_dir)
]

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCH_STAGE2,
    callbacks=callbacks_stage2,
    verbose=1
)

# Save final model
model.save(os.path.join(stage2_dir, "final_model.keras"))
model.save_weights(os.path.join(stage2_dir, "final_model_stage2.weights.h5"))
print(f"Stage 2 complete. Models saved to {stage2_dir}")


# =============================================================================
# FINAL EVALUATION
# =============================================================================
print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80 + "\n")

# Load best model
best_model_path = os.path.join(stage2_dir, "best_model.keras")
model = tf.keras.models.load_model(best_model_path, custom_objects={"l2_norm_layer": l2_norm_layer})

# Get predictions
y_true_list, y_pred_list = [], []
for batch_x, batch_y in test_ds:
    preds = model.predict(batch_x, verbose=0)
    y_true_list.append(batch_y.numpy())
    y_pred_list.append(preds)

y_true = np.concatenate(y_true_list, axis=0)
y_pred = np.concatenate(y_pred_list, axis=0)
y_true_labels = np.argmax(y_true, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Compute metrics
acc = accuracy_score(y_true_labels, y_pred_labels)
# top5_acc = top_k_accuracy_score(y_true_labels, y_pred, k=5, labels=range(NUM_CLASSES))
# top10_acc = top_k_accuracy_score(y_true_labels, y_pred, k=10, labels=range(NUM_CLASSES))
top5_acc = top_k_accuracy_score(y_true_labels, y_pred, k=5)  # remove labels param
top10_acc = top_k_accuracy_score(y_true_labels, y_pred, k=10)

f1 = f1_score(y_true_labels, y_pred_labels, average='macro')

confidences = np.max(y_pred, axis=1)
avg_confidence = np.mean(confidences)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")
print(f"Top-10 Accuracy: {top10_acc:.4f}")
print(f"Macro F1-score: {f1:.4f}")
print(f"Average Confidence: {avg_confidence:.4f}")

# Save final metrics
final_metrics = {
    "accuracy": float(acc),
    "top5_accuracy": float(top5_acc),
    "top10_accuracy": float(top10_acc),
    "f1_macro": float(f1),
    "avg_confidence": float(avg_confidence),
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(experiment_dir, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

# Save predictions
np.savez(
    os.path.join(experiment_dir, "predictions.npz"),
    y_true=y_true,
    y_pred=y_pred,
    y_true_labels=y_true_labels,
    y_pred_labels=y_pred_labels
)




# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_combined_history(h1, h2, save_dir):
    """Plot training history from both stages"""
    combined = {
        k: h1.history.get(k, []) + h2.history.get(k, [])
        for k in set(h1.history.keys()).union(h2.history.keys())
    }
    
    # combined = {}
    # all_keys = set(h1.history.keys()).union(h2.history.keys())
    # for k in all_keys:
    #     combined[k] = h1.history.get(k, []) + h2.history.get(k, [])
    
    epochs = range(1, len(combined["accuracy"]) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # def safe_plot(ax, key_train, key_val, title):
    # if key_train in combined and key_val in combined:
    #     ax.plot(range(1, len(combined[key_train])+1), combined[key_train], label="Train")
    #     ax.plot(range(1, len(combined[key_val])+1), combined[key_val], label="Validation")
    #     ax.set_title(title)
    #     ax.legend()
    #     ax.grid(True)
    
    # Accuracy
    axes[0, 0].plot(epochs, combined["accuracy"], label="Train")
    axes[0, 0].plot(epochs, combined["val_accuracy"], label="Validation")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(epochs, combined["loss"], label="Train")
    axes[0, 1].plot(epochs, combined["val_loss"], label="Validation")
    axes[0, 1].set_title("Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-5 Accuracy
    axes[1, 0].plot(epochs, combined["top5_accuracy"], label="Train")
    axes[1, 0].plot(epochs, combined["val_top5_accuracy"], label="Validation")
    axes[1, 0].set_title("Top-5 Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Top-10 Accuracy
    axes[1, 1].plot(epochs, combined["top10_accuracy"], label="Train")
    axes[1, 1].plot(epochs, combined["val_top10_accuracy"], label="Validation")
    axes[1, 1].set_title("Top-10 Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=200, bbox_inches='tight')
    plt.show()

plot_combined_history(history_stage1, history_stage2, experiment_dir)

# Confidence distribution
plt.figure(figsize=(8, 5))
plt.hist(confidences, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Model Confidence Scores")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.savefig(os.path.join(experiment_dir, "confidence_distribution.png"), dpi=200, bbox_inches='tight')
plt.show()

print(f"\n{'='*80}")
print(f"Training complete! All results saved to:")
print(f"{experiment_dir}")
print(f"{'='*80}\n")

def save_all_artifacts(experiment_dir, metrics, histories, confidences, y_true, y_pred):
    """Save metrics, histories, confidence distribution, and predictions"""
    os.makedirs(experiment_dir, exist_ok=True)

    # Save final metrics
    with open(os.path.join(experiment_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save training histories (combined)
    with open(os.path.join(experiment_dir, "history_stage1.json"), "w") as f:
        json.dump(histories[0].history, f, indent=4)
    with open(os.path.join(experiment_dir, "history_stage2.json"), "w") as f:
        json.dump(histories[1].history, f, indent=4)

    # Save predictions
    np.savez(
        os.path.join(experiment_dir, "predictions.npz"),
        y_true=y_true,
        y_pred=y_pred
    )

    # Confidence distribution
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Model Confidence Scores")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(experiment_dir, "confidence_distribution.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Save all artifacts
save_all_artifacts(
    experiment_dir,
    final_metrics,
    (history_stage1, history_stage2),
    confidences,
    y_true,
    y_pred
)



import os, json, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.losses import CategoricalCrossentropy
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
import matplotlib.pyplot as plt

# =============================================================================
# 1. RELOAD CONFIGURATION
# =============================================================================
experiment_dir = "/kaggle/working/experiment_20251112_0833056"  # ← CHANGE to your experiment folder
stage1_dir = os.path.join(experiment_dir, "stage1")
stage2_dir = os.path.join(experiment_dir, "stage2")
os.makedirs(stage2_dir, exist_ok=True)

# Load config
with open(os.path.join(experiment_dir, "config.json")) as f:
    config = json.load(f)

NUM_CLASSES = config["num_classes"]
IMG_SIZE = tuple(config["img_size"])
BATCH_SIZE = config["batch_size"]
EMBEDDING_DIM = config["embedding_dim"]

# =============================================================================
# 2. DATASETS (train, val, test)
# =============================================================================
base_dir = "/kaggle/input/11-785-fall-20-homework-2-part-2/classification_data"
train_dir = base_dir + "/train_data"
val_dir = base_dir + "/val_data"
test_dir = base_dir + "/test_data"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='categorical'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='categorical'
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =============================================================================
# 3. LOAD STAGE 1 MODEL
# =============================================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
def l2_norm_layer(x):
    return tf.nn.l2_normalize(x, axis=1)

model = tf.keras.models.load_model(
    os.path.join(stage1_dir, "best_model.keras"),
    custom_objects={"l2_norm_layer": l2_norm_layer}
)

# Try to locate the ResNet50V2 backbone
try:
    base_model = model.get_layer("resnet50v2")
except ValueError:
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "resnet" in layer.name.lower():
            base_model = layer
            break

# Unfreeze the top 50 layers for fine-tuning
if base_model is not None:
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    print(f"Unfroze {sum([1 for l in base_model.layers if l.trainable])} layers in base model.")
else:
    print("⚠️ Could not find ResNet base model; skipping fine-tuning unfreeze.")

# =============================================================================
# 4. COMPILE FOR FINE-TUNING
# =============================================================================
optimizer_stage2 = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)

model.compile(
    optimizer=optimizer_stage2,
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top10_accuracy')
    ]
)

# =============================================================================
# 5. CALLBACKS
# =============================================================================
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger, TensorBoard
)

class DetailedMetricsCallback(keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.metrics_history = []
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_metrics = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **logs
        }
        self.metrics_history.append(epoch_metrics)
        with open(os.path.join(self.log_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        print(f"\nEpoch {epoch+1} metrics saved.")

callbacks_stage2 = [
    ModelCheckpoint(
        os.path.join(stage2_dir, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    CSVLogger(os.path.join(stage2_dir, "training_log.csv")),
    TensorBoard(log_dir=os.path.join(stage2_dir, "tensorboard")),
    DetailedMetricsCallback(stage2_dir)
]

# =============================================================================
# 6. RUN STAGE 2 TRAINING
# =============================================================================
EPOCH_STAGE2 = 20  # adjust as desired

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCH_STAGE2,
    callbacks=callbacks_stage2,
    verbose=1
)

# Save final fine-tuned model
model.save(os.path.join(stage2_dir, "final_model.keras"))
model.save_weights(os.path.join(stage2_dir, "final_model_stage2.weights.h5"))
print(f"✅ Stage 2 complete. Models saved to {stage2_dir}")

# =============================================================================
# 7. FINAL EVALUATION ON TEST SET
# =============================================================================
print("\n" + "="*80)
print("FINAL EVALUATION ON TEST SET")
print("="*80 + "\n")

best_model_path = os.path.join(stage2_dir, "best_model.keras")
model = tf.keras.models.load_model(best_model_path, custom_objects={"l2_norm_layer": l2_norm_layer})

y_true_list, y_pred_list = [], []
for batch_x, batch_y in test_ds:
    preds = model.predict(batch_x, verbose=0)
    y_true_list.append(batch_y.numpy())
    y_pred_list.append(preds)

y_true = np.concatenate(y_true_list, axis=0)
y_pred = np.concatenate(y_pred_list, axis=0)
y_true_labels = np.argmax(y_true, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Compute metrics
acc = accuracy_score(y_true_labels, y_pred_labels)
top5_acc = top_k_accuracy_score(y_true_labels, y_pred, k=5)
top10_acc = top_k_accuracy_score(y_true_labels, y_pred, k=10)
f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
confidences = np.max(y_pred, axis=1)
avg_confidence = np.mean(confidences)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")
print(f"Top-10 Accuracy: {top10_acc:.4f}")
print(f"Macro F1-score: {f1:.4f}")
print(f"Average Confidence: {avg_confidence:.4f}")

# Save final metrics
final_metrics = {
    "accuracy": float(acc),
    "top5_accuracy": float(top5_acc),
    "top10_accuracy": float(top10_acc),
    "f1_macro": float(f1),
    "avg_confidence": float(avg_confidence),
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(experiment_dir, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

# Save predictions
np.savez(
    os.path.join(experiment_dir, "predictions_stage2.npz"),
    y_true=y_true,
    y_pred=y_pred,
    y_true_labels=y_true_labels,
    y_pred_labels=y_pred_labels
)

# =============================================================================
# 8. VISUALIZATION
# =============================================================================
plt.figure(figsize=(8, 5))
plt.hist(confidences, bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of Model Confidence Scores")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.savefig(os.path.join(experiment_dir, "confidence_distribution_stage2.png"), dpi=200, bbox_inches='tight')
plt.show()

print(f"\n{'='*80}")
print(f"Stage 2 and final evaluation complete! All results saved to:")
print(f"{experiment_dir}")
print(f"{'='*80}\n")



!zip -r /kaggle/working/experiment_stage1.zip /kaggle/working/experiment_20251112_083056*



from IPython.display import FileLink
FileLink('/kaggle/working/experiment_stage1.zip')



!kaggle datasets create -p /kaggle/working/experiment_20251112_083056 \
    -r zip --title "Stage1 ResNet model" \
    --description "Saved Stage 1 experiment for future fine-tuning"



!kaggle datasets create -p /kaggle/working/experiment_20251112_083056 -r zip



!mkdir -p ~/.kaggle
!cp /kaggle/input/meinapi/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json



!kaggle datasets list -s imagenet


