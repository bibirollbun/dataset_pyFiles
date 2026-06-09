import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


IMG_SIZE = 227
BATCH_SIZE = 32
EPOCHS = 16
LEARNING_RATE = 1e-5

DICOM_DATA_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images"
LABELS_CSV = "/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
PNG_OUTPUT_DIR = "/kaggle/working/rsna_pneumonia_png_images"

os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)

# --- DICOM'dan PNG'ye dÃ¶nÃ¼ÅŸÃ¼m (senin verdiÄŸin kodla aynÄ±) ---
df_labels = pd.read_csv(LABELS_CSV)
patient_ids = df_labels['patientId'].unique()

print(f"{len(patient_ids)} adet DICOM dosyasÄ± PNG'ye dÃ¶nÃ¼ÅŸtÃ¼rÃ¼lÃ¼yor...")

for i, patient_id in enumerate(patient_ids):
    dicom_path = os.path.join(DICOM_DATA_DIR, patient_id + ".dcm")
    output_path = os.path.join(PNG_OUTPUT_DIR, patient_id + ".png")

    if os.path.exists(output_path):
        continue

    try:
        dicom = pydicom.dcmread(dicom_path)
        img = dicom.pixel_array.astype(np.float32)

        if 'WindowCenter' in dicom and 'WindowWidth' in dicom:
            window_center = dicom.WindowCenter
            window_width = dicom.WindowWidth
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]

            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2

            img = np.clip(img, min_val, max_val)
            img = ((img - min_val) / (max_val - min_val + 1e-5)) * 255
        else:
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5) * 255

        img = img.astype(np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(output_path, img)

    except Exception as e:
        print(f"Hata: {patient_id}.dcm dÃ¶nÃ¼ÅŸtÃ¼rÃ¼lÃ¼rken hata oluÅŸtu: {e}")
        continue

    if (i + 1) % 1000 == 0:
        print(f"{i + 1} gÃ¶rÃ¼ntÃ¼ dÃ¶nÃ¼ÅŸtÃ¼rÃ¼ldÃ¼.")

print("TÃ¼m DICOM dosyalarÄ± PNG'ye dÃ¶nÃ¼ÅŸtÃ¼rÃ¼ldÃ¼.")


# --- Veri setini hazÄ±rla ---
df_labels['filename'] = df_labels['patientId'] + ".png"
df_labels['Target'] = df_labels['Target'].astype(str)

train_val_df, test_df = train_test_split(df_labels, test_size=0.2, stratify=df_labels["Target"], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df["Target"], random_state=42)

print(f"EÄŸitim seti boyutu: {len(train_df)}")
print(f"DoÄŸrulama seti boyutu: {len(val_df)}")
print(f"Test seti boyutu: {len(test_df)}")

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_df["Target"]), y=train_df["Target"])
class_weights = dict(enumerate(class_weights))
print("SÄ±nÄ±f AÄŸÄ±rlÄ±klarÄ± (Class Weights):", class_weights)


# --- TF Dataset fonksiyonu ---
def process_path(filename, label):
    img_path = tf.strings.join([PNG_OUTPUT_DIR, "/", filename])
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)  # Gri tonlama
    img = tf.image.grayscale_to_rgb(img)    # 3 kanal yapÄ±yoruz
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0-1 arasÄ± normalize

    # Veri artÄ±rma (sadece eÄŸitim iÃ§in)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_zoom(img, (0.85, 1.15)) if hasattr(tf.image, "random_zoom") else img  # TF 2.10 ve Ã¼zeri iÃ§in, deÄŸilse silebilirsin
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    return img, label

def process_path_no_aug(filename, label):
    img_path = tf.strings.join([PNG_OUTPUT_DIR, "/", filename])
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label


# Label'larÄ± int yapÄ±yoruz
train_labels = train_df['Target'].astype(int).values
val_labels = val_df['Target'].astype(int).values
test_labels = test_df['Target'].astype(int).values

train_ds = tf.data.Dataset.from_tensor_slices((train_df['filename'].values, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_df['filename'].values, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_df['filename'].values, test_labels))

train_ds = train_ds.shuffle(1000).map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(process_path_no_aug, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(process_path_no_aug, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



def build_vgg16_from_scratch(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    l2_reg = regularizers.l2(0.0005)

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg, input_shape=input_shape))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    # Block 2
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    # Block 3
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    # Block 4
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    # Block 5
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid', dtype='float32'))

    return model

model = build_vgg16_from_scratch()

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacklar
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Model Ã¶zet
model.summary()


# --- Modeli eÄŸit ---
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)


import matplotlib.pyplot as plt
def plot_history(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14,5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'r-', label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.show()

plot_history(history)



# GerÃ§ek etiketleri ve tahminleri toplamak iÃ§in listeler
y_true = []
y_pred_probs = []

# Dataset'ten verileri Ã§ek
for batch in val_ds:
    X_batch, y_batch = batch
    y_true.extend(y_batch.numpy())  # GerÃ§ek etiketleri topla
    preds = model.predict(X_batch, verbose=0)  # Tahmin olasÄ±lÄ±klarÄ±
    y_pred_probs.extend(preds)

# Listeyi numpy dizisine Ã§evir
y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

# Metrikleri hesapla
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_probs)

print(f"ðŸ”¹ Accuracy     : {acc:.4f}")
print(f"ðŸ”¹ Precision    : {prec:.4f}")
print(f"ðŸ”¹ Recall       : {rec:.4f}")
print(f"ðŸ”¹ F1-Score     : {f1:.4f}")
print(f"ðŸ”¹ ROC AUC      : {roc_auc:.4f}")


