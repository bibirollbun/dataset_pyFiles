import os, glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import TFSMLayer
from keras.layers import TFSMLayer, Input, Dropout, Dense
from tensorflow.keras import layers, models, regularizers, callbacks
from keras.models import Model
import tensorflow as tf
import tensorflow as tfa
import random

import zipfile
import os

print("✅ ")



# 预处理

# 定义解压目标目录（可自定义）
train_zip_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
test_zip_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'

extract_dir = '/kaggle/working/'

# 解压 train.zip
with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 解压 test.zip
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ ")


# ===============================
# 1. 参数设置
# ===============================
DATA_DIR = "/kaggle/working/train"
IMG_SIZE = 299
BATCH_SIZE = 32
SEED = 42

print("✅ ")



# ===============================
# 2. 数据准备
# ===============================
# 获取图像路径与标签（dog=1，cat=0）
all_images = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
labels = [1 if "dog" in os.path.basename(p) else 0 for p in all_images]

# 划分训练/验证集
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_images, labels, test_size=0.15, stratify=labels, random_state=SEED
)

# 图像解码与预处理函数
def decode_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label

# 构建 tf.data.Dataset
def build_dataset(paths, labels, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.shuffle(1024)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = build_dataset(train_paths, train_labels)
val_ds = build_dataset(val_paths, val_labels, is_train=False)

print("✅ ")



# ===============================
# 3. 构建模型
# ===============================

import tensorflow as tf
from tensorflow.keras import layers, models

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# 直接用官方MobileNetV2预训练模型，不包含顶层分类头
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # 冻结
for layer in base_model.layers[:-30]:  # 解冻最后 30 层
    layer.trainable = False

x = base_model(inputs, training=False)  # 前向传播
x = layers.GlobalAveragePooling2D(name="MobileNetV2")(x)  # GAP压平
# x = Dense(16, activation='relu')(x)  # 中间隐藏层
x = layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-4))(x)  # 添加L2正则化
x = layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(2e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  # 二分类输出

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()



# ===============================
# 4. 模型训练
# ===============================

# ✅ 设置 EarlyStopping 回调
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=1,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    callbacks=[early_stop]
)

print("✅ ")



# kaggle's score：0.057


# 保存为 keras 原生格式（推荐）
model.save('/kaggle/working/MobileNetV2.keras')

print("✅ ")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.show()



# ===============================
# 5. 生成预测文件
# ===============================


# MODEL_SAVE_PATH = '/kaggle/working/mbt2_plus.keras'
# model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# 测试数据
test_dir = "/kaggle/working/test"
test_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpg")), key=lambda x: int(os.path.basename(x).split('.')[0]))

# 构建测试集
def build_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: decode_img(p, 0)[0], num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = build_test_ds(test_paths)

# 模型预测
preds = model.predict(test_ds)
preds = preds.ravel()
preds = preds.clip(min=0.005, max=0.995)   # 裁剪一下两头

print("✅ ")



print(f"预测最大值：{preds.max():.4f}")
print(f"预测最小值：{preds.min():.4f}")
print(f"预测均值：{preds.mean():.4f}")
plt.hist(preds, bins=100)
plt.title("Test Prediction Distribution")
plt.show()



# 写入提交文件
import pandas as pd
submission = pd.DataFrame({
    "id": [int(os.path.basename(p).split('.')[0]) for p in test_paths],
    "label": preds
})
submission.to_csv("submission.csv", index=False)
print("✅ ")



import os
import shutil

# 指定要删除的文件夹路径
folder_path = '/kaggle/working/train'

# 删除文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'删除 {file_path} 失败。原因: {e}')

# 删除文件夹
try:
    os.rmdir(folder_path)
    print(f"文件夹 '{folder_path}' 已成功删除。")
except FileNotFoundError:
    print(f"文件夹 '{folder_path}' 不存在。")
except PermissionError:
    print(f"没有权限删除文件夹 '{folder_path}'。")
except OSError as e:
    print(f"删除文件夹时出错: {e}")


# 指定要删除的文件夹路径
folder_path = '/kaggle/working/test'

# 删除文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'删除 {file_path} 失败。原因: {e}')

# 删除文件夹
try:
    os.rmdir(folder_path)
    print(f"文件夹 '{folder_path}' 已成功删除。")
except FileNotFoundError:
    print(f"文件夹 '{folder_path}' 不存在。")
except PermissionError:
    print(f"没有权限删除文件夹 '{folder_path}'。")
except OSError as e:
    print(f"删除文件夹时出错: {e}")

