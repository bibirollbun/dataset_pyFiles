import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


df = pd.read_csv("/kaggle/input/isic-2024-challenge/train-metadata.csv", low_memory=False)
img_dir = "/kaggle/input/isic-2024-challenge/train-image/image/"


df.sample(5)


df.isnull().sum()


df_new = df[['isic_id', 'target']]


class_counts = df_new['target'].value_counts()
print(f"Số lượng của mỗi loại: {class_counts}")
imbalance_ratio = class_counts[0] / class_counts[1]
print(f"Tỉ lệ mất cân bằng ảnh giữa (class 0 : class 1): {imbalance_ratio:.2f} : 1")


IMG_SIZE = 128
BATCH_SIZE = 64  # Tăng batch size để tăng tốc độ huấn luyện
EPOCHS = 25      # Tăng số epoch
AUTOTUNE = tf.data.AUTOTUNE # theo tìm hiểu là giúp tăng tốc độ chạy


image_paths = [f"{img_dir}{img_id}.jpg" for img_id in df_new['isic_id']]
labels = df_new['target'].values



X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)
print(f"Training images: {len(X_train_paths)}")
print(f"Testing images: {len(X_test_paths)}")


import random
def load_and_preprocess_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Chuẩn hóa về [0,1]
    return img, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label



def prepare_dataset(paths, labels, augment_data=False, balance_data=False, target_ratio=0.4):
    # 1. Balancing dữ liệu nếu cần
    if balance_data:
        paths_0 = [p for p, l in zip(paths, labels) if l == 0]
        paths_1 = [p for p, l in zip(paths, labels) if l == 1]

        repeat_factor = int(len(paths_0) * target_ratio / len(paths_1))
        augmented_paths = paths_1 * repeat_factor
        augmented_labels = [1] * len(augmented_paths)

        new_paths = paths_0 + augmented_paths
        new_labels = [0] * len(paths_0) + augmented_labels
    else:
        new_paths = paths
        new_labels = labels

    # 2. Tạo tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((new_paths, new_labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    # 3. Conditional augmentation: chỉ augment label==1
    if augment_data:
        def conditional_augment(image, label):
            return tf.cond(
                tf.equal(label, 1),
                lambda: augment(image, label),  # Gọi đúng augment(image, label)
                lambda: (image, label)          # Giữ nguyên nếu label != 1
            )
        dataset = dataset.map(conditional_augment, num_parallel_calls=AUTOTUNE)

    # 4. Shuffle, batch, prefetch
    dataset = (
        dataset
        .shuffle(buffer_size=min(len(new_paths), 10000))
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return dataset


train_dataset = prepare_dataset(X_train_paths, y_train,
                                augment_data=True,
                                balance_data=True,
                                target_ratio=0.4)
test_dataset  = prepare_dataset(X_test_paths, y_test)


from collections import Counter

def get_balanced_labels(orig_labels, balance_data=False, target_ratio=0.4):
    if balance_data:
        # tách labels
        labels_0 = [l for l in orig_labels if l == 0]
        labels_1 = [l for l in orig_labels if l == 1]
        # số lần lặp
        repeat_factor = max(1, int(len(labels_0) * target_ratio / len(labels_1)))
        # nhãn đã augment
        augmented_labels = labels_1 * repeat_factor
        # list mới
        return labels_0 + augmented_labels
    else:
        return orig_labels

# ví dụ với y_train
new_labels = get_balanced_labels(y_train, balance_data=True, target_ratio=0.4)
counts = Counter(new_labels)
print("Sau balance/augmentation:", counts)  
# Ví dụ output: Counter({0: 40000, 1: 40000})



new_labels = get_balanced_labels(y_train, balance_data=True, target_ratio=0.4)
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(new_labels), y=new_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights sau balance: {class_weight_dict}")



def visualize_balanced_augmentation(paths, labels, num_samples_per_class=3):
    # Separate paths by class
    paths_class0 = [path for path, label in zip(paths, labels) if label == 0]
    paths_class1 = [path for path, label in zip(paths, labels) if label == 1]
    
    # Get paths for visualization (num_samples_per_class from each class)
    vis_paths_class0 = paths_class0[:num_samples_per_class]
    vis_paths_class1 = paths_class1[:num_samples_per_class]
    
    # Combine paths and create corresponding labels
    vis_paths = vis_paths_class0 + vis_paths_class1
    vis_labels = [0] * len(vis_paths_class0) + [1] * len(vis_paths_class1)
    
    # Total number of samples to visualize
    total_samples = len(vis_paths)
    
    # Create dataset for visualization
    vis_dataset = tf.data.Dataset.from_tensor_slices((vis_paths, vis_labels))
    vis_dataset = vis_dataset.map(load_and_preprocess_image)
    
    # Create figure for visualization
    plt.figure(figsize=(15, 5))
    
    # Plot each sample
    for i, (image, label) in enumerate(vis_dataset.take(total_samples)):
        # Original image
        plt.subplot(2, total_samples, i+1)
        plt.imshow(image.numpy())
        plt.title(f"Original: Class {label.numpy()}")
        plt.axis('off')
        
        # Augmented image
        augmented_img, _ = augment(image, label)
        plt.subplot(2, total_samples, i+1+total_samples)
        plt.imshow(augmented_img.numpy())
        plt.title(f"Augmented: Class {label.numpy()}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage example
visualize_balanced_augmentation(X_train_paths, y_train, num_samples_per_class=3)


model = tf.keras.Sequential([
    # Block 1
    tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3), use_bias = False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    
    # Block 2
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    
    # Block 3
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    
    # Block 4
    tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    
    # Fully connected layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Tóm tắt kiến trúc mô hình
model.summary()


# Thiết lập learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9, # mỗi lần
    staircase=True # giảm theo từng bước nhảy 
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# Biên dịch mô hình
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
)


# Định nghĩa callbacks
callbacks = [
    # Early stopping với patience cao hơn
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Giảm learning rate khi model đi vào plateau
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    ),
    # Lưu model tốt nhất - với định dạng .keras mới
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',  # Thay đổi từ .h5 sang .keras
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    # TensorBoard logging (tùy chọn)
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]


# Huấn luyện mô hình
history = model.fit(
    train_dataset,
    epochs=25,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)






