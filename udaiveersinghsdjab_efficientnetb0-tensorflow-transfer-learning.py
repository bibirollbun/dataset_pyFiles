# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
import keras_tuner as kt
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import f1_score
from tensorflow.keras.metrics import Metric


# Set paths
TRAIN_CSV = "/kaggle/input/ai-vs-human-generated-dataset/train.csv"
TEST_CSV = "/kaggle/input/ai-vs-human-generated-dataset/test.csv"
TRAIN_PATH = "/kaggle/input/ai-vs-human-generated-dataset/train_data/"
TEST_PATH = "/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/"

# Load CSV files
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Check dataset
print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
print(train_df.head())


train_df.info()


test_df.info()


# Drop rows with missing values
train_df = train_df.dropna().reset_index(drop=True)
test_df = test_df.dropna().reset_index(drop=True)

print(f"After dropping NaN: Train shape: {train_df.shape}, Test shape: {test_df.shape}")


print(train_df.columns)
print(test_df.columns)


# Extract image paths and labels correctly
DATASET_PATH = "/kaggle/input/ai-vs-human-generated-dataset"

train_df["file_name"] = train_df["file_name"].apply(lambda x: os.path.join(DATASET_PATH, "train_data", os.path.basename(x)))
test_df["id"] = test_df["id"].apply(lambda x: os.path.join(DATASET_PATH, "test_data_v2", os.path.basename(x)))

# Split into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_df['file_name'].values, train_df['label'].values, test_size=0.2, random_state=42, stratify=train_df['label']
)

print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")


print("Any NaN in train paths?", pd.isna(train_paths).sum())
print("Any NaN in train labels?", pd.isna(train_labels).sum())


# Image size required for EfficientNet
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Function to load and preprocess images
def load_image(image_path, label=None):
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Ensure RGB
    image = tf.image.resize(image, IMG_SIZE)  # Resize to EfficientNet input size  
    image = preprocess_input(image)
    # Return (image, label) for training, (image) for test
    return (image, label) if label is not None else image

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    def _gaussian_kernel(size, sigma):
        x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        x = tf.exp(-0.5 * tf.square(x) / tf.square(sigma))
        kernel = tf.tensordot(x, x, axes=0)
        kernel /= tf.reduce_sum(kernel)
        return kernel[:, :, tf.newaxis, tf.newaxis]

    kernel = _gaussian_kernel(kernel_size, sigma)
    kernel = tf.repeat(kernel, repeats=3, axis=2)  # For RGB
    image = tf.expand_dims(image, axis=0)  # Add batch dim
    image = tf.nn.depthwise_conv2d(image, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(image, axis=0)  # Remove batch dim

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image)  # Flip 90 degrees
    # image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.2)
    
    # Randomly darken image
    if tf.random.uniform([]) < 0.5:
        image = tf.image.adjust_brightness(image, -0.2)
    
    # # Randomly apply blur
    # if tf.random.uniform([]) < 0.3:
    #     image = gaussian_blur(image)

    return image, label


# Create training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_image, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(augment, num_parallel_calls=AUTOTUNE)  # Augment training data
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Create validation dataset (no augmentation)
val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_dataset = val_dataset.map(load_image, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("Datasets prepared successfully!")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Convert to numpy arrays if they're tensors
train_class_labels = train_labels.numpy() if hasattr(train_labels, 'numpy') else train_labels
val_class_labels = val_labels.numpy() if hasattr(val_labels, 'numpy') else val_labels

# Set style
sns.set(style="whitegrid")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot training label distribution
sns.countplot(x=train_class_labels, ax=axes[0], palette="coolwarm")
axes[0].set_title("Train Label Distribution")
axes[0].set_xlabel("Class (0 or 1)")
axes[0].set_ylabel("Count")
axes[0].set_xticks([0, 1])

# Plot validation label distribution
sns.countplot(x=val_class_labels, ax=axes[1], palette="coolwarm")
axes[1].set_title("Validation Label Distribution")
axes[1].set_xlabel("Class (0 or 1)")
axes[1].set_ylabel("Count")
axes[1].set_xticks([0, 1])

plt.tight_layout()
plt.show()


# # Custom F1 Score Metric for training
# class F1Score(Metric):
#     def __init__(self, name='f1_score', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name='tp', initializer='zeros')
#         self.false_positives = self.add_weight(name='fp', initializer='zeros')
#         self.false_negatives = self.add_weight(name='fn', initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
#         y_true = tf.cast(y_true, tf.float32)

#         tp = tf.reduce_sum(y_true * y_pred)
#         fp = tf.reduce_sum((1 - y_true) * y_pred)
#         fn = tf.reduce_sum(y_true * (1 - y_pred))

#         self.true_positives.assign_add(tp)
#         self.false_positives.assign_add(fp)
#         self.false_negatives.assign_add(fn)

#     def result(self):
#         precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
#         recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)
#         return 2 * ((precision * recall) / (precision + recall + 1e-6))

#     def reset_states(self):
#         for var in self.variables:
#             var.assign(0)


# # Model builder for Keras Tuner
# def model_builder(hp):
#     base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#     base_model.trainable = True
#     for layer in base_model.layers[:-30]:
#         layer.trainable = False

#     model = models.Sequential()
#     model.add(base_model)
#     model.add(layers.GlobalAveragePooling2D())

#     # Tune number of dense layers
#     for i in range(hp.Int("num_dense_layers", 1, 3)):
#         units = hp.Int(f"dense_units_{i}", min_value=64, max_value=512, step=64)
#         activation = hp.Choice("activation", ["relu", "swish", "leaky_relu", "mish"])

#         if activation == "leaky_relu":
#             model.add(layers.Dense(units, kernel_initializer="he_normal"))
#             model.add(layers.LeakyReLU(alpha=0.1))
#         else:
#             model.add(layers.Dense(units, activation=activation, kernel_initializer="he_normal"))

#         if hp.Boolean(f"batchnorm_{i}"):
#             model.add(layers.BatchNormalization())

#         if hp.Boolean(f"dropout_{i}"):
#             model.add(layers.Dropout(hp.Float(f"dropout_rate_{i}", 0.3, 0.5)))

#     model.add(layers.Dense(1, activation='sigmoid'))

#     model.compile(
#         optimizer=tf.keras.optimizers.AdamW(hp.Float("learning_rate", 1e-4, 1e-2, sampling='log')),
#         loss="binary_crossentropy",
#         metrics=[F1Score()]
#     )
#     return model

# trial_f1_log = {}
    
# # Callback to track F1 score on train & validation
# class F1ScoreCallback(tf.keras.callbacks.Callback):
#     def __init__(self, train_dataset, val_dataset, trial_id=None):
#         super().__init__()
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.trial_id = trial_id
#         self.best_train_f1 = 0

#     def on_epoch_end(self, epoch, logs=None):
#         ...
#         train_f1 = f1_score(train_true, train_preds)
#         val_f1 = f1_score(val_true, val_preds)

#         # Save best train F1
#         self.best_train_f1 = max(self.best_train_f1, train_f1)

#         print(f"\nEpoch {epoch + 1} - Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

#     def on_train_end(self, logs=None):
#         # Save best train F1 after all epochs
#         trial_f1_log[self.trial_id] = self.best_train_f1

# # Early stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss",
#     patience=3,
#     restore_best_weights=True
# )

# # Keras Tuner setup
# tuner = kt.Hyperband(
#     model_builder,
#     objective="val_loss", 
#     max_epochs=10,
#     factor=3,
#     directory='keras_tuner_logs',
#     project_name='f1_tuning_model'
# )

# # Manual training loop to track best TRAIN F1 across trials
# trial_f1_log = {}

# for trial in range(5):  # Run 5 hyperparameter trials (change as needed)
#     # Sample a new hyperparameter set from the search space
#     hp = tuner.oracle.get_space().copy()
#     model = tuner.hypermodel.build(hp)

#     trial_id = f"trial_{trial}"
#     print(f"\nğŸ”� Running {trial_id}")

#     history = model.fit(
#         train_dataset,
#         validation_data=val_dataset,
#         epochs=10,
#         callbacks=[
#             early_stopping,
#             F1ScoreCallback(train_dataset, val_dataset, trial_id=trial_id)
#         ]
#     )

# # Find best trial based on TRAIN F1 score
# best_trial = max(trial_f1_log, key=trial_f1_log.get)
# print(f"\nğŸ�† Best Trial: {best_trial} | Train F1: {trial_f1_log[best_trial]:.4f}")


import optuna
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def build_model(trial):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    
    # ğŸ”§ Tune how many layers to unfreeze in the base model
    unfreeze_layers = trial.suggest_int("unfreeze_layers", 0, 30)
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())

    # ğŸ”§ Tune number of dense layers
    num_dense_layers = trial.suggest_int("num_dense_layers", 1, 3)
    for i in range(num_dense_layers):
        units = trial.suggest_int(f"dense_units_{i}", 64, 1024, step=64)
        activation = trial.suggest_categorical(f"activation_{i}", ["relu", "swish", "leaky_relu", "mish"])

        model.add(layers.Dense(units, kernel_initializer="he_normal"))
        if activation == "leaky_relu":
            model.add(layers.LeakyReLU(alpha=0.1))
        elif activation == "mish":
            model.add(tf.keras.layers.Activation(tf.nn.swish))  # Mish approximation
        else:
            model.add(layers.Activation(activation))

        if trial.suggest_categorical(f"batchnorm_{i}", [True, False]):
            model.add(layers.BatchNormalization())

        if trial.suggest_categorical(f"dropout_{i}", [True, False]):
            dropout_rate = trial.suggest_float(f"dropout_rate_{i}", 0.3, 0.5)
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

def objective(trial):
    model = build_model(trial)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    # Train model (no callbacks for simplicity; you can add EarlyStopping)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=5, verbose=0)

    # Get validation predictions
    y_true = np.concatenate([y for _, y in val_dataset], axis=0)
    y_pred_prob = model.predict(val_dataset).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    f1 = f1_score(y_true, y_pred)

    print(f"Trial {trial.number}: F1 Score = {f1:.4f}")
    return f1

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Best trial result
print("\nâœ… Best Trial:")
print(f"F1 Score: {study.best_value:.4f}")
print("Best Hyperparameters:")
for k, v in study.best_trial.params.items():
    print(f"{k}: {v}")


# Load EfficientNetB0 (Pretrained on ImageNet, without top layers)
base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
# base_model.trainable = False  # Freeze base model

base_model.trainable = True  # Unfreeze all layers

# Freeze all layers except the last `unfreeze_layers`
unfreeze_layers = 0  
for layer in base_model.layers[:-unfreeze_layers]:
    layer.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  
    
    # Dense Layer 0
    layers.Dense(640, kernel_initializer="he_normal"),
    layers.LeakyReLU(alpha=0.1),  # activation_0: leaky_relu
    layers.BatchNormalization(),
    layers.Dropout(0.3359853061113047),

    # Dense Layer 1
    layers.Dense(256, kernel_initializer="he_normal", activation="mish"),
    layers.BatchNormalization(),

    # Dense Layer 2
    layers.Dense(512, activation='relu', kernel_initializer="he_normal"),  # activation_2: relu

    # Output
    layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Print model summary
model.summary()


# Define Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy", mode="max")

# Train the Model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,  # Adjust based on performance
    callbacks=[early_stopping, model_checkpoint]
)


import matplotlib.pyplot as plt
import pandas as pd

# Convert history to DataFrame for easy plotting
history_df = pd.DataFrame(history.history)

# Plot Loss Curve
plt.figure(figsize=(10, 4))
plt.plot(history_df["loss"], label="Training Loss")
plt.plot(history_df["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

# Plot Accuracy Curve
plt.figure(figsize=(10, 4))
plt.plot(history_df["accuracy"], label="Training Accuracy")
plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.show()


IMG_SIZE = 224
BATCH_SIZE = 32

def decode_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3) # Adjust if images are PNG
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE]) # Resize to model input size
    image = preprocess_input(image)  # Required for EfficientNet
    return image
    
# Load test image paths
test_paths = test_df["id"].values

# Create a TensorFlow dataset for the test images
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Test dataset loaded successfully!")


print(test_paths[:5])  # Print first 5 paths to verify
print("Any NaN in test paths?", pd.isna(test_paths).sum())


# Function to load and preprocess images (adjust based on your preprocessing)
# def decode_image(filename, label):
#     image = tf.io.read_file(filename)
#     image = tf.image.decode_jpeg(image, channels=3)  # Adjust for your dataset format
#     image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])  # Resize to model input size
#     image = preprocess_input(image)  # Normalize 
#     return image, label
    
# # Create validation dataset
# val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
# val_ds = val_ds.map(decode_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# print("Validation dataset loaded successfully!")


from matplotlib import pyplot as plt
import random

# View val sample
# val_sample = random.choice(val_paths)
# img = tf.keras.preprocessing.image.load_img(val_sample)
# plt.imshow(img)
# plt.title("Validation Image")
# plt.show()

# View test sample
# test_sample = random.choice(test_paths)
# img = tf.keras.preprocessing.image.load_img(test_sample)
# plt.imshow(img)
# plt.title("Test Image")
# plt.show()

# Pick 15 random samples
random_samples = random.sample(list(test_paths), 15)

# Create a 3x5 grid
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for i, img_path in enumerate(random_samples):
    img = tf.keras.preprocessing.image.load_img(img_path)
    ax = axes[i // 5, i % 5]  # Determine row and column
    ax.imshow(img)
    ax.set_title(f"Image {i+1}")
    ax.axis('off')  # Hide axis

plt.tight_layout()
plt.show()


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

true_train_labels = []
for _, labels in train_dataset:
    true_train_labels.extend(labels.numpy())

true_val_labels = []
for _, labels in val_dataset:
    true_val_labels.extend(labels.numpy())

# Step 1: Get predictions on training set
train_predictions = model.predict(train_dataset).flatten()
train_preds_binary = (train_predictions > 0.5).astype(int)

# Step 2: Get predictions on validation set
val_predictions = model.predict(val_dataset).flatten()
val_preds_binary = (val_predictions > 0.5).astype(int)

# Step 3: Get true labels (you already have these)
# Assuming train_labels and val_labels are available and are numpy arrays

# Step 4: Calculate metrics
train_precision = precision_score(true_train_labels, train_preds_binary)
train_recall = recall_score(true_train_labels, train_preds_binary)
train_f1 = f1_score(true_train_labels, train_preds_binary)
train_accuracy = accuracy_score(true_train_labels, train_preds_binary)

val_precision = precision_score(true_val_labels, val_preds_binary)
val_recall = recall_score(true_val_labels, val_preds_binary)
val_f1 = f1_score(true_val_labels, val_preds_binary)
val_accuracy = accuracy_score(true_val_labels, val_preds_binary)

# Step 5: Print results
print("=== Training Metrics ===")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")
print(f"F1 Score:  {train_f1:.4f}")
print(f"Training Accuracy:   {train_accuracy:.4f}")

print("\n=== Validation Metrics ===")
print(f"Precision: {val_precision:.4f}")
print(f"Recall:    {val_recall:.4f}")
print(f"F1 Score:  {val_f1:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")


# from sklearn.metrics import

# # Convert model predictions (probabilities) to binary class labels (0 or 1)
# train_preds_class = (train_predictions > 0.5).astype(int).flatten()
# val_preds_class = (val_predictions > 0.5).astype(int).flatten()

# # True labels (already binary and not one-hot encoded)
# true_train_labels = train_labels.flatten()
# true_val_labels = val_labels.flatten()

# # Calculate accuracy
# train_accuracy = accuracy_score(true_train_labels, train_preds_class)
# val_accuracy = accuracy_score(true_val_labels, val_preds_class)

# # Print results
# print("=== Accuracy Metrics ===")
# print(f"Training Accuracy:   {train_accuracy:.4f}")
# print(f"Validation Accuracy: {val_accuracy:.4f}")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure predictions are binary class labels (0 or 1)
train_preds_class = (train_predictions > 0.5).astype(int).flatten()
val_preds_class = (val_predictions > 0.5).astype(int).flatten()

# Set style
sns.set(style="whitegrid")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot predicted training labels
sns.countplot(x=train_preds_binary, ax=axes[0], palette="mako")
axes[0].set_title("Predicted Train Label Distribution")
axes[0].set_xlabel("Predicted Class (0 or 1)")
axes[0].set_ylabel("Count")
axes[0].set_xticks([0, 1])

# Plot predicted validation labels
sns.countplot(x=val_preds_binary, ax=axes[1], palette="mako")
axes[1].set_title("Predicted Validation Label Distribution")
axes[1].set_xlabel("Predicted Class (0 or 1)")
axes[1].set_ylabel("Count")
axes[1].set_xticks([0, 1])

plt.tight_layout()
plt.show()


# Get validation predictions
val_predictions = model.predict(val_dataset).flatten()  # Ensure 1D array
print(f"Min: {np.min(val_predictions)}, Max: {np.max(val_predictions)}, Mean: {np.mean(val_predictions)}")


# Make predictions
predictions = model.predict(test_ds)
predictions = predictions.flatten()  # Convert to 1D array

print(f"Min: {np.min(predictions)}, Max: {np.max(predictions)}, Mean: {np.mean(predictions)}")

# Apply threshold
predicted_labels = (predictions >= 0.07).astype(int)


# How many 0s and 1s are predicted
unique, counts = np.unique(predicted_labels, return_counts=True)
print(dict(zip(unique, counts)))


plt.hist(val_predictions, bins=50, alpha=0.7, label='Validation')
plt.hist(predictions, bins=50, alpha=0.7, label='Test')
plt.legend()
plt.title("Prediction Distribution")
plt.show()


# Create a submission DataFrame
submission = pd.DataFrame({
    "id": test_df["id"].apply(lambda x: "/".join(x.split("/")[-2:])),  # Use actual test IDs from the dataset
    "label": predicted_labels.flatten().astype(int)  # Convert predictions to integers
})

# Save as CSV without index and header
submission.to_csv("/kaggle/working/submission.csv", index=False, encoding="utf-8")

print("Predictions saved to 'submission.csv'")


df = pd.read_csv("/kaggle/working/submission.csv")
print(df.head())
print(df.dtypes)  # Check if id and label have correct types
df['label'].value_counts()

