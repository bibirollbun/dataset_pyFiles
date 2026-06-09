!pip install -q pydicom tensorflow

# ✅ Imports
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

# Load RSNA-MICCAI Dataset
labels_df = pd.read_csv('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
train_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/'




def load_image(patient_id, img_size=(128, 128)):
    folder = os.path.join(train_path, str(patient_id).zfill(5), "T1w")
    if not os.path.exists(folder): return None
    files = sorted(os.listdir(folder))
    if len(files) == 0: return None
    path = os.path.join(folder, files[len(files)//2])
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, -1)

X, y = [], []
for _, row in labels_df.iterrows():
    img = load_image(row['BraTS21ID'])
    if img is not None:
        X.append(img)
        y.append(row['MGMT_value'])

X = np.array(X)
y = np.array(y)

# === Train/Val/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)



# Define Swin Transformer Components
input_shape = (128, 128, 1)
patch_size = 4
embed_dim = 64
num_heads = 4
window_size = 4
mlp_dim = 128
dropout_rate = 0.1

class WindowAttention(layers.Layer):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)

    def call(self, x):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, N, C])
        return self.proj(out)




def swin_block(x, dim, num_heads, window_size, mlp_dim):
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = WindowAttention(dim, num_heads, window_size)(x)
    x = layers.Add()([shortcut, x])
    shortcut2 = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dense(mlp_dim, activation='gelu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dim)(x)
    x = layers.Dropout(dropout_rate)(x)
    return layers.Add()([shortcut2, x])

def patch_embedding(inputs, patch_size, embed_dim):
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)(inputs)
    x = layers.Reshape((-1, embed_dim))(x)
    return x

def build_swin_model():
    inputs = keras.Input(shape=input_shape)
    x = patch_embedding(inputs, patch_size, embed_dim)
    for _ in range(2):
        x = swin_block(x, embed_dim, num_heads, window_size, mlp_dim)
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(mlp_dim, activation="gelu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)



# Compile and Train
swin_model = build_swin_model()
swin_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                   loss="binary_crossentropy",
                   metrics=["accuracy"])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = swin_model.fit(X_train, y_train,
                         validation_data=(X_val, y_val),
                         epochs=30,
                         batch_size=16,
                         callbacks=[early_stop],
                         verbose=1)



# Evaluate 
test_loss, test_acc = swin_model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = swin_model.predict(X_test)
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



# Plots
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_val = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_val:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Swin Transformer (New Method III)')
plt.legend()
plt.grid()
plt.show()


