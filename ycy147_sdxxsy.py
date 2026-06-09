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


!pip install py7zr


import os
import py7zr

# 设置输入和输出路径
INPUT_PATH = "/kaggle/input/cifar-10"
OUTPUT_PATH = "/kaggle/working"  # Kaggle工作目录，可写

# 定义文件路径
TRAIN_7Z = os.path.join(INPUT_PATH, "train.7z")
TEST_7Z = os.path.join(INPUT_PATH, "test.7z")
TRAIN_OUTPUT = os.path.join(OUTPUT_PATH, "train")
TEST_OUTPUT = os.path.join(OUTPUT_PATH, "test")

# 确保输出目录存在
os.makedirs(TRAIN_OUTPUT, exist_ok=True)
os.makedirs(TEST_OUTPUT, exist_ok=True)

print("开始解压训练数据...")
with py7zr.SevenZipFile(TRAIN_7Z, mode='r') as z:
    z.extractall(path=TRAIN_OUTPUT)
print(f"训练数据已解压到: {TRAIN_OUTPUT}")

print("\n开始解压测试数据...")
with py7zr.SevenZipFile(TEST_7Z, mode='r') as z:
    z.extractall(path=TEST_OUTPUT)
print(f"测试数据已解压到: {TEST_OUTPUT}")

# 检查解压后的文件
print("\n解压完成! 文件列表:")
print("训练目录:", os.listdir(TRAIN_OUTPUT))
print("测试目录:", os.listdir(TEST_OUTPUT))


# 导入必要的库
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint


# 设置路径（根据解压结果调整）
TRAIN_IMG_PATH = "/kaggle/working/train/train"
TEST_IMG_PATH = "/kaggle/working/test/test"
TRAIN_LABELS_PATH = "/kaggle/input/cifar-10/trainLabels.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/cifar-10/sampleSubmission.csv"
MODEL_SAVE_PATH = "/kaggle/working/cifar10_resnet18_model.h5"

# 超参数设置
BATCH_SIZE = 128
EPOCHS = 40
VAL_SPLIT = 0.15
LEARNING_RATE = 1e-3
L2_REG = 1e-4

# 数据标准化参数
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def load_data():
    """加载和预处理数据"""
    # 加载标签数据
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    print("标签数据预览:")
    print(labels_df.head())
    
    # 创建标签编码器
    label_encoder = LabelEncoder()
    labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
    class_names = label_encoder.classes_
    print("\n类别名称:", class_names)
    
    # 数据加载函数
    def load_images(image_folder, label_df=None):
        images = []
        labels = []
        filenames = []
        
        # 按文件名排序确保与标签顺序一致
        file_list = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')], 
                          key=lambda x: int(x.split('.')[0]))
        
        print(f"找到 {len(file_list)} 个图像文件")
        
        for i, filename in enumerate(file_list):
            if i % 5000 == 0:  # 每5000张打印一次进度
                print(f"处理图像 {i}/{len(file_list)}...")
                
            img_path = os.path.join(image_folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(32, 32))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            
            img_id = int(filename.split('.')[0])
            filenames.append(img_id)
            
            if label_df is not None:
                label_row = label_df[label_df['id'] == img_id]
                if not label_row.empty:
                    label = label_row['label_encoded'].values[0]
                    labels.append(label)
        
        images = np.array(images, dtype='float32')
        
        # CIFAR-10专用数据标准化
        images = (images / 255.0 - CIFAR10_MEAN) / CIFAR10_STD
        
        if label_df is not None and labels:
            labels = np.array(labels)
            return images, labels, filenames
        else:
            return images, filenames
    
    # 加载训练数据
    print("\n加载训练数据...")
    X_train, y_train, train_filenames = load_images(TRAIN_IMG_PATH, labels_df)
    print(f"训练数据形状: {X_train.shape}, 标签形状: {y_train.shape}")
    
    # 加载测试数据
    print("\n加载测试数据...")
    X_test, test_filenames = load_images(TEST_IMG_PATH)
    print(f"测试数据形状: {X_test.shape}")
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SPLIT, random_state=42, stratify=y_train
    )
    print(f"\n训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, test_filenames, label_encoder

# 调用数据加载函数
X_train, y_train, X_val, y_val, X_test, test_filenames, label_encoder = load_data()


def create_data_augmenter():
    """创建数据增强生成器"""
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='constant',
        cval=0.0
    )
    datagen.fit(X_train)
    return datagen

# 创建数据增强器
datagen = create_data_augmenter()


def residual_block(x, filters, downsample=False, name=None):
    """残差块实现"""
    strides = 2 if downsample else 1
    identity = x
    
    # 主路径
    x = layers.Conv2D(filters, kernel_size=3, strides=strides, 
                      padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size=3, strides=1, 
                      padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    
    # 捷径连接（如果下采样或通道数变化）
    if downsample:
        identity = layers.Conv2D(filters, kernel_size=1, strides=2, 
                                 padding='valid', kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(L2_REG))(identity)
        identity = layers.BatchNormalization()(identity)
    
    # 添加捷径连接
    x = layers.Add()([x, identity])
    x = layers.Activation('relu')(x)
    return x

def build_resnet18(input_shape=(32, 32, 3), num_classes=10):
    """构建ResNet-18模型"""
    inputs = layers.Input(shape=input_shape)
    
    # 初始卷积层
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', 
                      kernel_initializer='he_normal', 
                      kernel_regularizer=regularizers.l2(L2_REG))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 残差块序列
    # Stage 1
    x = residual_block(x, 64, name='block1_1')
    x = residual_block(x, 64, name='block1_2')
    
    # Stage 2 (下采样)
    x = residual_block(x, 128, downsample=True, name='block2_1')
    x = residual_block(x, 128, name='block2_2')
    
    # Stage 3 (下采样)
    x = residual_block(x, 256, downsample=True, name='block3_1')
    x = residual_block(x, 256, name='block3_2')
    
    # Stage 4 (下采样)
    x = residual_block(x, 512, downsample=True, name='block4_1')
    x = residual_block(x, 512, name='block4_2')
    
    # 全局平均池化和输出层
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_regularizer=regularizers.l2(L2_REG))(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet18')
    return model

# 构建ResNet-18模型
model = build_resnet18(input_shape=(32, 32, 3), num_classes=10)
model.summary()


def lr_schedule(epoch):
    """学习率调度 - 在指定epoch减少学习率"""
    lr = LEARNING_RATE
    if epoch > 75:
        lr *= 0.1e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 45:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    print(f'Epoch {epoch+1}: 学习率 = {lr:.6f}')
    return lr

# 回调函数
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    LearningRateScheduler(lr_schedule),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', 
                   save_best_only=True, save_weights_only=False)
]


# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
print("\n开始训练ResNet-18模型...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)


# 评估模型
print("\n模型评估:")
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"验证集准确率: {val_acc*100:.2f}%")
print(f"验证集损失: {val_loss:.4f}")

# 可视化训练过程
def plot_training_history(history):
    # 第一个图表：准确率
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], linestyle='--', color='m', label='train acc')
    plt.plot(history.history['val_accuracy'], linestyle='-.', color='g', label='valid acc')
    plt.title('training and validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 第二个图表：损失值 - 注意这里缩进与第一个图表相同
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], linestyle='-', color='b', label='train loss')
    plt.plot(history.history['val_loss'], linestyle='-', color='orange', label='valid loss')
    plt.title('training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制训练历史（注意这行在函数定义之外，没有缩进）
plot_training_history(history)


# 预测测试集
print("\n预测测试集...")
test_probs = model.predict(X_test, batch_size=128, verbose=1)
test_preds = np.argmax(test_probs, axis=1)
test_labels = label_encoder.inverse_transform(test_preds)

# 创建提交文件
def create_submission(test_filenames, test_labels):
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'label': test_labels
    })
    
    # 确保id顺序与sampleSubmission一致
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission_df = submission_df.sort_values('id').reset_index(drop=True)
    
    # 保存提交文件
    submission_df.to_csv('resnet18_submission.csv', index=False)
    print("\n提交文件已保存: resnet18_submission.csv")
    print(f"提交文件预览:\n{submission_df.head()}")
    return submission_df

# 生成提交文件
submission_df = create_submission(test_filenames, test_labels)

