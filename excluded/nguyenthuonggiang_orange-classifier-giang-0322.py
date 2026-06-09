import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


data_dir = '/kaggle/input/ai-training-challenge-hutech-orange-classifier/old_oranges_data_1/old_oranges_data/'
train_dir = os.path.join(data_dir, 'train_set')
test_dir = os.path.join(data_dir, 'test_set')


# Hàm đọc dữ liệu
def load_data(data_dir):
    images = []
    labels = []
    filenames = []

    # Duyệt qua các thư mục con
    for label, folder in enumerate(["Orange_Bad", "Orange_Good"]):
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith((".jpg", ".png")):
                file_path = os.path.join(folder_path, file)
                
                # Mở và resize ảnh
                img = Image.open(file_path).resize((224, 224))
                images.append(np.array(img))
                
                # Gán nhãn
                labels.append(1 if folder == "Orange_Bad" else 0)
                filenames.append(file)
    
    return np.array(images), np.array(labels), filenames

# Đọc dữ liệu
data_dir = '/kaggle/input/ai-training-challenge-hutech-orange-classifier/old_oranges_data_1/old_oranges_data/'
train_dir = os.path.join(data_dir, 'train_set')
test_dir = os.path.join(data_dir, 'test_set')

X_train_raw, y_train_raw, _ = load_data(train_dir)
X_test_raw, _, test_filenames = load_data(test_dir)




# Chuẩn hóa dữ liệu
X_train_raw = X_train_raw / 255.0
X_test_raw = X_test_raw / 255.0

# Chia tập train và validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_raw, y_train_raw, test_size=0.2, random_state=42
)

# Khởi tạo ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()

# Tạo generator
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)



# Xây dựng mô hình
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile mô hình
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# In mô hình
model.summary()



import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Khởi tạo EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',    # Theo dõi giá trị loss trên tập validation
    patience=5,            # Dừng nếu không cải thiện sau 5 epochs
    restore_best_weights=True  # Phục hồi trọng số tốt nhất
)

# Huấn luyện mô hình với EarlyStopping
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping]  # Thêm EarlyStopping vào callbacks
)

# Vẽ đồ thị quá trình huấn luyện và validation
plt.figure(figsize=(12, 5))

# Vẽ loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Vẽ accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



# Dự đoán nhãn cho tập test
predictions = model.predict(X_test_raw)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Tạo file submission
submission_df = pd.DataFrame({
    "image_name": test_filenames,  
    "label": predicted_labels     
})

# Xuất file CSV
submission_df.to_csv("submission.csv", index=False)

print("File submission.csv đã được tạo!")



# Đảm bảo sắp xếp tên file đúng thứ tự
submission_df = submission_df.sort_values(by="image_name").reset_index(drop=True)

# Xuất lại file CSV
submission_df.to_csv("submission.csv", index=False)


