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


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv('/kaggle/input/exploring-machine-learning-with-mnist/mnist_train.csv')
test_df = pd.read_csv('/kaggle/input/exploring-machine-learning-with-mnist/mnist_test.csv')

print("Train shape:", train_df.shape)  # (60000, 785)
print("Test shape:", test_df.shape)    # (10000, 785)



X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values 



X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels for training and test data
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)



plt.figure(figsize=(6, 4))
random_indices = np.random.choice(len(X_train), 6, replace=False)

for i, idx in enumerate(random_indices):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_train[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_train[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, val_idx in sss.split(X_train, y_train):
    X_train, X_val = X_train[train_idx], X_train[val_idx]
    y_train_cat, y_val_cat = y_train_cat[train_idx], y_train_cat[val_idx]



datagen = ImageDataGenerator(
    rotation_range=10,         # rotate images ±10 degrees
    width_shift_range=0.1,     # shift images horizontally by 10%
    height_shift_range=0.1,    # shift images vertically by 10%
    zoom_range=0.1             # zoom in/out by 10%
)

# Fit the Generator on Training Data
datagen.fit(X_train)



model = tf.keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Dense Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]



history = model.fit(datagen.flow(X_train, y_train_cat, batch_size=64),
                    validation_data=(X_val, y_val_cat),
                    epochs=20,
                    callbacks=callbacks,
                    verbose=2)

# Save training history to JSON
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

print("Training history saved to 'training_history.json'")



plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



# Load the best model saved during training
model = load_model("/kaggle/working/best_model.h5")
print("Best model loaded successfully.")



test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")



pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)



# Confirm shape
print("pred_probs shape:", pred_probs.shape)
print("y_test shape:", y_test_cat.shape)

# Compute per-class AUC
auc_scores = []
for i in range(10):
    auc = roc_auc_score(y_test_cat[:, i], pred_probs[:, i])
    auc_scores.append(auc)
    print(f"Class {i} AUC: {auc:.4f}")
    
# Compute overall macro-average
macro_auc = np.mean(auc_scores)
print(f"Macro-average AUC: {macro_auc:.4f}")



# Classification report
print("Classification Report:\n")
print(classification_report(y_test, pred_labels))

# Confusion matrix
cm = confusion_matrix(y_test, pred_labels)
# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()



def visualize_random_samples(X, y_true, y_pred=None):
    """
    Displays 6 randomly selected grayscale images along 
    with their true and optional predicted labels.
    
    Args:
        X (ndarray): Array of image data.
        y_true (ndarray): True labels.
        y_pred (ndarray, optional): Predicted probabilities or labels. Defaults to None.
    """
    plt.figure(figsize=(12, 8))
    indices = np.random.choice(len(X), 6, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        actual_label = y_true[idx]
        pred_label = np.argmax(y_pred[idx]) if y_pred is not None else None
        title = f"Actual: {actual_label}"
        if y_pred is not None:
            title += f"\nPred: {pred_label}"
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize random 6 images with actual and predicted labels
visualize_random_samples(X_test, y_test, pred_probs)


submission = pd.DataFrame({
    'ID': test_df.index + 1,
    'Label': list(pred_labels)
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created.")





