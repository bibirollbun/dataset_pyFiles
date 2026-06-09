!pip install -q pydicom tensorflow

# Imports
import os
import numpy as np
import pandas as pd
import cv2
import pydicom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load MGMT labels
train_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/'
labels_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'
labels_df = pd.read_csv(labels_path)


# Load multiple slices per patient into a bag
def load_patient_slices(patient_id, img_size=(128,128), max_slices=10):
    folder = os.path.join(train_path, str(patient_id).zfill(5), "T1w")
    if not os.path.exists(folder): return None
    files = sorted(os.listdir(folder))
    if len(files) == 0: return None
    slice_indices = np.linspace(0, len(files) - 1, max_slices, dtype=int)
    slices = []
    for idx in slice_indices:
        path = os.path.join(folder, files[idx])
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        img = cv2.resize(img, img_size)
        img = img / 255.0
        slices.append(img)
    return np.expand_dims(np.array(slices), -1)  # (slices, 128, 128, 1)


# Build bags and labels
bags = []
bag_labels = []

for _, row in labels_df.iterrows():
    slices = load_patient_slices(row['BraTS21ID'])
    if slices is not None:
        bags.append(slices)
        bag_labels.append(row['MGMT_value'])

bags = np.array(bags)
bag_labels = np.array(bag_labels)


# Train/Val/Test split
X_train, X_test, y_train, y_test = train_test_split(bags, bag_labels, test_size=0.2, stratify=bag_labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

# Attention MIL Layer
class MILAttentionLayer(layers.Layer):
    def __init__(self, hidden_units):
        super(MILAttentionLayer, self).__init__()
        self.dense = layers.Dense(hidden_units, activation="tanh")
        self.attention = layers.Dense(1)

    def call(self, inputs):
        x = self.dense(inputs)
        alpha = self.attention(x)
        alpha = tf.nn.softmax(alpha, axis=1)
        return tf.reduce_sum(alpha * inputs, axis=1)


# MIL Model Definition
def create_mil_model(input_shape=(10, 128, 128, 1)):
    inputs = keras.Input(shape=input_shape)
    x = layers.TimeDistributed(layers.Conv2D(32, 3, activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Conv2D(64, 3, activation='relu'))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    x = MILAttentionLayer(128)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    return model


# Create and compile model
mil_model = create_mil_model()
mil_model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

# Train model
history = mil_model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=30,
                        batch_size=8,
                        verbose=1)


# Evaluate
test_loss, test_acc = mil_model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = mil_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
f1 = f1_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)


print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"AUC: {roc_auc:.4f}")


# Accuracy/Loss plots
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_val = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_val:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - MIL Model')
plt.legend()
plt.show()




plt.figure(figsize=(12, 8))
for i in range(9):
    p_idx = np.random.randint(0, len(X_train))
    s_idx = np.random.randint(0, X_train.shape[1])
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[p_idx][s_idx].squeeze(), cmap='gray')
    plt.title(f"Label: {y_train[p_idx]}")
    plt.axis('off')
plt.suptitle('Sample Training Slices from MIL Bags')
plt.tight_layout()
plt.show()


