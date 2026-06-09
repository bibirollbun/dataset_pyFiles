# 导入必要库
!pip install py7zr
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import py7zr  # 用于解压.7z文件


# 解压数据集函数
def extract_7z(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Extracting {file_path} to {output_dir}...")
    with py7zr.SevenZipFile(file_path, mode='r') as z:
        z.extractall(output_dir)
    print(f"Extraction complete. Files in {output_dir}: {len(os.listdir(output_dir))}")

# 解压数据集（在Kaggle环境中路径如下）
extract_7z('/kaggle/input/cifar-10/train.7z', '/kaggle/working/train')
extract_7z('/kaggle/input/cifar-10/test.7z', '/kaggle/working/test')


import glob
# 加载训练标签
train_labels = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
print(f"Loaded {len(train_labels)} training labels")
label_dict = dict(zip(train_labels['id'], train_labels['label']))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
print("Class names:", class_names)

# 加载图像数据的函数
def load_images(folder_path, label_dict=None):
    # 获取文件夹中所有图像文件
    image_files = glob.glob(os.path.join(folder_path, '*.png'))
    print(f"Found {len(image_files)} PNG files in {folder_path}")
    
    if len(image_files) == 0:
        raise ValueError(f"No PNG files found in {folder_path}")
    
    images = []
    labels = [] if label_dict else None
    ids = []
    
    for img_path in tqdm(image_files, desc=f"Loading {os.path.basename(folder_path)} images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        images.append(img)
        
        if label_dict:
            filename = os.path.basename(img_path)
            img_id = int(os.path.splitext(filename)[0])
            ids.append(img_id)
            labels.append(label_dict.get(img_id, None))
    
    images = np.array(images)
    print(f"Successfully loaded {len(images)} images")
    
    if label_dict:
        return images, np.array(labels), ids
    return images, None, [int(os.path.splitext(os.path.basename(f))[0]) for f in image_files]

# 加载训练和测试数据
print("\nLoading training images...")
X_train, y_train, train_ids = load_images('/kaggle/working/train/train', label_dict)

print("\nLoading test images...")
X_test, _, test_ids = load_images('/kaggle/working/test/test')


# 数据预处理
def preprocess(images):
    images = images.astype('float32') / 255.0
    return images

X_train = preprocess(X_train)
X_test = preprocess(X_test)
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# 标签编码
label_to_index = {name: i for i, name in enumerate(class_names)}
y_train_encoded = np.array([label_to_index[label] for label in y_train])
y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, 10)
print(f"Labels encoded. One-hot shape: {y_train_onehot.shape}")

# 划分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train_onehot, test_size=0.2, random_state=42
)
print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")


# 构建增强型CNN模型
def create_enhanced_model():
    model = models.Sequential()
    
    # 数据增强层
    model.add(layers.RandomFlip("horizontal", input_shape=(32, 32, 3)))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))
    
    # 卷积块1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # 卷积块2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.35))
    
    # 卷积块3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.45))
    
    # 全连接层
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    # 优化器配置
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

model = create_enhanced_model()
model.summary()

# 设置回调函数
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
model_checkpoint = callbacks.ModelCheckpoint(
    'best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'
)

# 训练模型
print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# 加载最佳模型
model = models.load_model('best_model.h5')
print("Loaded best model for predictions")


# 预测测试集
print("\nPredicting on test set...")
probabilities = model.predict(X_test, batch_size=128, verbose=1)
predictions = np.argmax(probabilities, axis=1)
predicted_labels = [class_names[idx] for idx in predictions]

# 生成提交文件
submission = pd.DataFrame({
    'id': test_ids,
    'label': predicted_labels
})
submission.to_csv('submission.csv', index=False)
print(f"Submission file created with {len(submission)} predictions")

# 可视化训练过程
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()


