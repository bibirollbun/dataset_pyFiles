# Install dependencies
!pip install -q pydicom tensorflow

# Imports
import os, cv2, pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess RSNA-MICCAI MRI data 
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

# Train/Validation/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)




# Define EANet Block 
def eanet_block(x, filters):
    conv = layers.Conv2D(filters, (3, 3), padding='same')(x)
    conv = layers.BatchNormalization()(conv)
    conv = layers.ReLU()(conv)
    
    spatial_att = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv)
    channel_avg = tf.reduce_mean(conv, axis=[1, 2], keepdims=True)
    channel_max = tf.reduce_max(conv, axis=[1, 2], keepdims=True)
    channel_att = layers.Conv2D(filters, (1, 1), activation='sigmoid')(channel_avg + channel_max)
    
    x = conv * spatial_att * channel_att
    return x



# Define EANet Model
def build_eanet(input_shape=(128, 128, 1)):
    inputs = keras.Input(shape=input_shape)
    
    x = eanet_block(inputs, 32)
    x = layers.MaxPooling2D()(x)

    x = eanet_block(x, 64)
    x = layers.MaxPooling2D()(x)

    x = eanet_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model



# Compile and Train EANet
eanet_model = build_eanet()
eanet_model.compile(optimizer=keras.optimizers.Adam(1e-4),
                    loss="binary_crossentropy",
                    metrics=["accuracy"])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = eanet_model.fit(X_train, y_train,
                          validation_data=(X_val, y_val),
                          epochs=30,
                          batch_size=16,
                          callbacks=[early_stop],
                          verbose=1)



# Evaluate Model
test_loss, test_acc = eanet_model.evaluate(X_test, y_test, verbose=0)
y_pred_prob = eanet_model.predict(X_test)
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



# Plot Accuracy & Loss
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




# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_val = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_val:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - EANet (New Method V)')
plt.legend()
plt.grid()
plt.show()


