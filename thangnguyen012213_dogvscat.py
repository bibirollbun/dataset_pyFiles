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


# ===== 1.1 Import =====
import tensorflow as tf
import zipfile, os, shutil
from sklearn.model_selection import train_test_split

# ===== 1.2 Giải nén dữ liệu =====
zip_train_path = "/kaggle/input/dogs-vs-cats/train.zip"
zip_test_path  = "/kaggle/input/dogs-vs-cats/test1.zip"
extract_root   = "/kaggle/working/data"

with zipfile.ZipFile(zip_train_path, 'r') as zip_ref:
    zip_ref.extractall(extract_root)

with zipfile.ZipFile(zip_test_path, 'r') as zip_ref:
    zip_ref.extractall(extract_root + "/test")

print("✅ Đã giải nén train & test")

# ===== 1.3 Tạo thư mục train/val cho cats và dogs =====
base_dir = "/kaggle/working/split_data"
train_dir = os.path.join(base_dir, 'train')
val_dir   = os.path.join(base_dir, 'val')

for category in ['cats', 'dogs']:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

# ===== 1.4 Lấy danh sách file ảnh & chia train/val =====
cat_files = [f for f in os.listdir(extract_root + "/train") if f.startswith("cat")]
dog_files = [f for f in os.listdir(extract_root + "/train") if f.startswith("dog")]

cat_train, cat_val = train_test_split(cat_files, test_size=0.2, random_state=123)
dog_train, dog_val = train_test_split(dog_files, test_size=0.2, random_state=123)

# ===== 1.5 Di chuyển file ảnh vào thư mục mới =====
for f in cat_train:
    shutil.move(os.path.join(extract_root + "/train", f),
                os.path.join(train_dir, 'cats', f))
for f in cat_val:
    shutil.move(os.path.join(extract_root + "/train", f),
                os.path.join(val_dir, 'cats', f))

for f in dog_train:
    shutil.move(os.path.join(extract_root + "/train", f),
                os.path.join(train_dir, 'dogs', f))
for f in dog_val:
    shutil.move(os.path.join(extract_root + "/train", f),
                os.path.join(val_dir, 'dogs', f))

print("✅ Đã chia dữ liệu train/val đều cho cats & dogs")

# ===== 1.6 Load dataset =====
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=32
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(150, 150),
    batch_size=32
)

# ===== 1.7 Chuẩn hóa và one-hot encode =====
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, depth=2)))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), tf.one_hot(y, depth=2)))



from tensorflow.keras import layers, models

# ===== 2.1 Xây dựng kiến trúc CNN =====
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)), # Conv layer 1
    layers.MaxPooling2D((2,2)),                                            # Pooling 1
    
    layers.Conv2D(64, (3,3), activation='relu'),  # Conv layer 2
    layers.MaxPooling2D((2,2)),                   # Pooling 2
    
    layers.Conv2D(128, (3,3), activation='relu'), # Conv layer 3
    layers.MaxPooling2D((2,2)),                   # Pooling 3
    
    layers.Flatten(),                             # Flatten to vector
    layers.Dense(128, activation='relu'),         # Fully connected layer
    layers.Dense(2, activation='sigmoid')         
])

# ===== 2.2 In model summary =====
model.summary()



from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt


# ===== 3.1 Compile model =====
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===== 3.2 Train model =====
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# ===== 3.3 Vẽ biểu đồ loss & accuracy =====
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.show()

# ===== 3.4 Đánh giá trên validation set =====
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_acc:.4f}")

# ===== 3.5 Precision, Recall, F1-score =====
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=['Cat','Dog']))



from tensorflow.keras import regularizers

# ===== 4.1 Thêm Dropout & L2 Regularization =====
model_reg = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),  # Dropout layer
    
    layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

# ===== 4.2 Data augmentation =====
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

aug_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# ===== 4.3 Compile & train =====
model_reg.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history_reg = model_reg.fit(aug_train_ds, validation_data=val_ds, epochs=5)



from tensorflow.keras.applications import VGG16

# ===== 5.1 Load VGG16 pre-trained =====
base_model = VGG16(input_shape=(150,150,3), include_top=False, weights='imagenet')

# Freeze các layer ban đầu
for layer in base_model.layers:
    layer.trainable = False

# ===== 5.2 Thêm các layer cho classification =====
x = layers.Flatten()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(2, activation='softmax')(x)

transfer_model = models.Model(inputs=base_model.input, outputs=x)

# ===== 5.3 Compile & train =====
transfer_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history_transfer = transfer_model.fit(train_ds, validation_data=val_ds, epochs=5)



#đổi optimizer
from tensorflow.keras.optimizers import RMSprop, SGD

model_sgd = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model_sgd.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model_sgd.fit(train_ds, validation_data=val_ds, epochs=5)



# Custom CNN giống AlexNet nhỏ gọn
model_custom = models.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model_custom.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model_custom.fit(train_ds, validation_data=val_ds, epochs=5)


