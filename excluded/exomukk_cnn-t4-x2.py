# 1. Cài đặt lại thư viện cần thiết
!pip install "protobuf<4.21"
!pip install "tensorflow-io-gcs-filesystem>=0.23.1"

import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.layers import LeakyReLU # [MỚI] Hàm kích hoạt tốt hơn
from sklearn.model_selection import train_test_split
# Bỏ import class_weight vì nó gây nhiễu model lúc này

# --- 1. CẤU HÌNH HỆ THỐNG & GPU ---
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU!")
except ValueError:
    strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} device(s) (GPU/CPU)")

# Cấu hình đường dẫn
IMG_DIR = '/kaggle/input/microsoft-malware/processed_images'
LABEL_FILE = '../input/malware-classification/trainLabels.csv'
IMG_SIZE = (226, 226)
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
EPOCHS = 40 # Tăng epoch vì kiến trúc mới (GAP) học chậm mà chắc

# --- 2. LOAD DỮ LIỆU ---
print("Loading data...")
labels_df = pd.read_csv(LABEL_FILE)
labels_df['Class'] = labels_df['Class'] - 1 

# [QUAN TRỌNG] Sắp xếp ID để khớp với XGBoost
labels_df = labels_df.sort_values('Id').reset_index(drop=True)

X = []
y = []

if os.path.exists(IMG_DIR):
    valid_files = set(os.listdir(IMG_DIR))
    print(f"Found {len(valid_files)} images.")
    
    for index, row in labels_df.iterrows():
        file_name = row['Id'] + '.png'
        if file_name in valid_files:
            try:
                path = os.path.join(IMG_DIR, file_name)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    X.append(img)
                    y.append(row['Class'])
            except:
                pass
    print(f"Successfully loaded: {len(X)} images.")
else:
    print("ERROR: Image directory not found!")

# Chuẩn hóa
X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0
y = np.array(y)

# Chia tập dữ liệu (Random State 20251226 khớp XGBoost)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20251226, stratify=y)
print(f"Train shape: {X_train.shape}")

# --- 3. XÂY DỰNG MODEL (KIẾN TRÚC NÂNG CẤP) ---
with strategy.scope():
    model = models.Sequential()
    
    model.add(layers.Input(shape=(226, 226, 1)))
    
    # Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1)) # [MỚI] Thay ReLU
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Block 3
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4)) 
    
    # Block 4
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4)) 
    
    # [NÂNG CẤP QUAN TRỌNG NHẤT] Thay Flatten bằng GlobalAveragePooling2D
    # Giúp model khái quát hóa tốt hơn, tăng Val Acc
    model.add(layers.GlobalAveragePooling2D())
    
    # Head
    model.add(layers.Dense(1024))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(9, activation='softmax'))

    # Dùng learning rate nhỏ hơn chút để hội tụ mượt
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# --- 4. TRAIN ---
checkpoint = callbacks.ModelCheckpoint('best_cnn_sync.keras', 
                                       monitor='val_accuracy', 
                                       save_best_only=True, 
                                       mode='max', verbose=1)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                        factor=0.5, 
                                        patience=4, 
                                        min_lr=1e-6, verbose=1)

early_stop = callbacks.EarlyStopping(monitor='val_loss', 
                                     patience=10, 
                                     restore_best_weights=True, verbose=1)

print("Starting training (Optimized Architecture - No Class Weight)...")
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint, reduce_lr, early_stop])
                    # Đã bỏ class_weight để tránh lỗi tụt accuracy

# --- 5. LƯU KẾT QUẢ ---
print("Saving results...")
model.load_weights('best_cnn_sync.keras')

y_pred_probs = model.predict(X_test)
np.save('cnn_probs.npy', y_pred_probs)
np.save('y_test_labels.npy', y_test)

print(f"Final Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

