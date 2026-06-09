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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. LOAD DATA
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/lex-ai-kaggle-june-comp-1-digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/lex-ai-kaggle-june-comp-1-digit-recognizer/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 2. EXPLORATORY DATA ANALYSIS
print("\n=== Exploratory Data Analysis ===")

# Check for missing values
print(f"\nMissing values in train: {train_df.isnull().sum().sum()}")
print(f"Missing values in test: {test_df.isnull().sum().sum()}")

# Distribution of labels
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
train_df['label'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Digits in Training Set')
plt.xlabel('Digit')
plt.ylabel('Count')

# Display some sample images
plt.subplot(1, 2, 2)
sample_indices = np.random.choice(train_df.index, 9, replace=False)
for i, idx in enumerate(sample_indices):
    plt.subplot(3, 3, i+1)
    image = train_df.iloc[idx, 1:].values.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {train_df.iloc[idx, 0]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# 3. DATA PREPROCESSING
print("\n=== Data Preprocessing ===")

# Separate features and labels
X = train_df.drop('label', axis=1).values
y = train_df['label'].values
X_test = test_df.values

# Normalize pixel values (0-255 to 0-1)
X = X / 255.0
X_test = X_test / 255.0

# Reshape for CNN (28x28x1)
X = X.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Split train data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# Convert labels to categorical
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_val_cat = keras.utils.to_categorical(y_val, 10)

# 4. BUILD AND TRAIN MODEL
print("\n=== Building CNN Model ===")

# Build CNN model
model = keras.Sequential([
    # First Conv Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Conv Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Train model
print("\nTraining model...")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=128),
    validation_data=(X_val, y_val_cat),
    epochs=30,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 5. EVALUATE MODEL
print("\n=== Model Evaluation ===")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Validation performance
val_predictions = model.predict(X_val)
val_pred_classes = np.argmax(val_predictions, axis=1)
print(f"\nValidation Accuracy: {np.mean(val_pred_classes == y_val):.4f}")

# Confusion matrix
cm = confusion_matrix(y_val, val_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.title('Confusion Matrix on Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 6. MAKE PREDICTIONS AND CREATE SUBMISSION
print("\n=== Making Predictions ===")

# Predict on test set
test_predictions = model.predict(X_test)
test_pred_classes = np.argmax(test_predictions, axis=1)

# Create submission dataframe
submission = pd.DataFrame({
    'ImageId': range(1, len(test_pred_classes) + 1),
    'Label': test_pred_classes
})

# Save submission
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved! Shape: {submission.shape}")
print(submission.head(10))

# 7. ENSEMBLE WITH SIMPLE MODELS (Optional Enhancement)
print("\n=== Training Additional Models for Ensemble ===")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Flatten data for traditional ML models
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_flat, y_train)
rf_val_acc = rf.score(X_val_flat, y_val)
print(f"Random Forest Validation Accuracy: {rf_val_acc:.4f}")

# Make ensemble predictions
cnn_test_proba = model.predict(X_test)
rf_test_proba = rf.predict_proba(X_test_flat)

# Weighted average ensemble (CNN gets more weight due to better performance)
ensemble_proba = 0.8 * cnn_test_proba + 0.2 * rf_test_proba
ensemble_pred = np.argmax(ensemble_proba, axis=1)

# Create ensemble submission
ensemble_submission = pd.DataFrame({
    'ImageId': range(1, len(ensemble_pred) + 1),
    'Label': ensemble_pred
})

ensemble_submission.to_csv('ensemble_submission.csv', index=False)
print(f"\nEnsemble submission saved! Shape: {ensemble_submission.shape}")

print("\n=== Process Complete! ===")
print("Files created:")
print("- submission.csv (CNN predictions)")
print("- ensemble_submission.csv (Ensemble predictions)")
print("\nThe ensemble submission typically performs better!")

# Display some test predictions
plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {test_pred_classes[i]}')
    plt.axis('off')
plt.suptitle('Sample Test Predictions')
plt.show()

