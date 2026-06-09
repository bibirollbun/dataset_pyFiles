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


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Đường dẫn tới dữ liệu
# Thư mục chứa ảnh train và test
data_dir = '/kaggle/input/aichallenge/old_oranges_data_1/old_oranges_data'

train_dir = os.path.join(data_dir, 'train_set')
test_dir = os.path.join(data_dir, 'test_set')


# Tăng cường dữ liệu cho tập test
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)


# Tăng cường dữ liệu cho tập train
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    rotation_range=45,  # Xoay ảnh ngẫu nhiên trong khoảng 45 độ
    width_shift_range=0.2,  # Dịch chuyển ngang tối đa 20%
    height_shift_range=0.2,  # Dịch chuyển dọc tối đa 20%
    shear_range=0.2,  # Biến dạng cắt
    zoom_range=0.2,  # Phóng to/thu nhỏ ngẫu nhiên
    horizontal_flip=True,  # Lật ngang ảnh
    brightness_range=[0.8, 1.2],  # Thay đổi độ sáng
    fill_mode='nearest'  # Điền giá trị pixel khi thiếu
)


# Khởi tạo generator cho tập train
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),  # Kích thước ảnh đầu vào (200x200)
    batch_size=32,  # Kích thước dữ liệu
    class_mode='binary'  # Phân loại cam tốt và cam dở
)


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary',
    shuffle=False # đúng thứ tự
)


# Xây dựng mô hình
model = Sequential([
    # Lớp tích chập (Convolutional Layer) đầu tiên
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    BatchNormalization(),  # Chuẩn hóa batch để tăng tốc độ học
    MaxPooling2D(pool_size=(2, 2)),  # Lấy đặc trưng với max pooling
    Dropout(0.25),  # Giảm overfitting
    # Lớp tích chập thứ hai
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    # Lớp tích chập thứ ba
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    # Lớp dàn phẳng và kết nối đầy đủ
    Flatten(),
    Dense(256, activation='relu'),  # Lớp kết nối đầy đủ
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Lớp đầu ra với sigmoid
])


# 5. Compile model
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])


# --- Lưu mô hình tốt nhất trong quá trình huấn luyện ---
checkpoint_path = 'best_model.h5.keras'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')


# --- Huấn luyện mô hình ---
history = model.fit(
    train_generator,
    epochs=20,  # Số vòng lặp huấn luyện
    validation_data=test_generator,  # Dữ liệu kiểm tra
    callbacks=[checkpoint]  # Callback lưu mô hình tốt nhất
)


# --- Đánh giá mô hình tốt nhất ---
best_model = tf.keras.models.load_model(checkpoint_path)


# Đánh giá trên tập test
test_loss, test_acc = best_model.evaluate(test_generator)
print(f"Độ chính xác trên tập test: {test_acc}")


# --- Dự đoán và lưu kết quả ---
predictions = (best_model.predict(test_generator) > 0.5).astype(int).flatten()


# Lấy danh sách tên file ảnh
filenames = [os.path.basename(file).replace('.jpg', '') for file in test_generator.filenames]


#lưu kết quả
results = pd.DataFrame({
    'image_name': filenames,  # Tên ảnh
    'label': predictions  # Nhãn(0: good, 1: bad)
})


results


#chỉnh lại label
results['label'] = results['label'].apply(lambda x: 1 if x == 0 else 0)


results


# Lưu kết quả vào file CSV
results.to_csv('results.csv', sep=',', index=False)
print("Dự đoán đã lưu vào 'results.csv'")

