import numpy as np
import pandas as pd
import pydicom
import os
import json
from pathlib import Path
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import nibabel as nib
from scipy import ndimage


# 设置随机种子以确保可重复性
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 定义文件路径
BASE_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection'
TRAIN_CSV_PATH = f'{BASE_PATH}/train.csv'
SERIES_PATH = f'{BASE_PATH}/train'

# 加载训练数据
train_df = pd.read_csv(TRAIN_CSV_PATH)
print(f"训练集大小: {train_df.shape}")

# 检查数据列
print("数据列:", train_df.columns.tolist())
print("前几行数据:")
print(train_df.head())


def load_dicom_series(series_path, series_uid):
    """加载DICOM系列并转换为3D体积"""
    try:
        series_dir = Path(series_path) / series_uid
        if not series_dir.exists():
            print(f"警告: 目录 {series_dir} 不存在")
            return np.zeros((64, 64, 64)), []
            
        dicom_files = list(series_dir.glob("*.dcm"))
        
        if not dicom_files:
            print(f"警告: 在 {series_dir} 中未找到DICOM文件")
            return np.zeros((64, 64, 64)), []
        
        # 按切片位置排序
        slices = []
        positions = []
        
        for file_path in dicom_files:
            try:
                dicom = pydicom.dcmread(str(file_path))
                slices.append(dicom)
                if hasattr(dicom, 'ImagePositionPatient') and dicom.ImagePositionPatient:
                    positions.append(float(dicom.ImagePositionPatient[2]))
                else:
                    # 如果没有位置信息，使用切片位置
                    positions.append(float(getattr(dicom, 'SliceLocation', 0)))
            except Exception as e:
                print(f"读取DICOM文件时出错 {file_path}: {e}")
                continue
        
        # 检查是否有有效的切片
        if not slices:
            print(f"警告: 没有有效的DICOM切片可用于系列 {series_uid}")
            return np.zeros((64, 64, 64)), []
            
        # 按位置排序
        if positions and len(set(positions)) > 1:
            sorted_slices = [s for _, s in sorted(zip(positions, slices), key=lambda x: x[0])]
        else:
            sorted_slices = slices
        
        # 创建3D体积
        try:
            # 确保所有切片具有相同的尺寸
            first_shape = sorted_slices[0].pixel_array.shape
            for i, s in enumerate(sorted_slices):
                if s.pixel_array.shape != first_shape:
                    print(f"警告: 切片 {i} 的尺寸不一致")
                    # 可以在这里添加调整尺寸的逻辑，或者跳过不一致的切片
            
            volume = np.stack([s.pixel_array for s in sorted_slices], axis=-1)
            
            # 应用Rescale斜率/截距
            if hasattr(sorted_slices[0], 'RescaleSlope') and hasattr(sorted_slices[0], 'RescaleIntercept'):
                slope = sorted_slices[0].RescaleSlope
                intercept = sorted_slices[0].RescaleIntercept
                volume = volume * slope + intercept
            
            return volume, sorted_slices
        except Exception as e:
            print(f"创建3D体积时出错: {e}")
            return np.zeros((64, 64, 64)), sorted_slices
            
    except Exception as e:
        print(f"加载DICOM系列时发生未知错误: {e}")
        return np.zeros((64, 64, 64)), []

def preprocess_volume(volume, target_shape=(64, 64, 64)):
    """预处理3D体积"""
    if volume.size == 0 or np.all(volume == 0):
        return np.zeros((*target_shape, 1))
    
    # 重采样到目标形状
    try:
        zoom_factors = [
            target_shape[0] / volume.shape[0],
            target_shape[1] / volume.shape[1],
            target_shape[2] / volume.shape[2]
        ]
        
        volume_resampled = ndimage.zoom(volume, zoom_factors, order=1)
    except:
        # 如果重采样失败，使用零填充
        volume_resampled = np.zeros(target_shape)
        min_shape = [min(volume_resampled.shape[i], volume.shape[i]) for i in range(3)]
        for i in range(min_shape[0]):
            for j in range(min_shape[1]):
                for k in range(min_shape[2]):
                    volume_resampled[i, j, k] = volume[i, j, k]
    
    # 强度归一化
    volume_normalized = (volume_resampled - np.mean(volume_resampled)) / (np.std(volume_resampled) + 1e-6)
    
    # 添加通道维度
    volume_normalized = np.expand_dims(volume_normalized, axis=-1)
    
    return volume_normalized

def augment_volume(volume):
    """对3D体积进行数据增强"""
    # 随机翻转
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=0)  # 沿x轴翻转
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=1)  # 沿y轴翻转
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=2)  # 沿z轴翻转
    
    # 随机旋转 (0, 90, 180, 270度)
    k = np.random.randint(0, 4)
    volume = np.rot90(volume, k=k, axes=(0, 1))
    
    # 随机亮度调整
    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        volume = volume * factor
    
    return volume


class AneurysmClassificationGenerator(tf.keras.utils.Sequence):
    """3D动脉瘤分类数据生成器"""
    
    def __init__(self, series_df, base_path, batch_size=4, 
                 target_shape=(64, 64, 64), shuffle=True, use_cache=True, augment=False):
        self.series_df = series_df.reset_index(drop=True)
        self.base_path = base_path
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        self.cache = {}  # 用于缓存预处理后的体积
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.series_df) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_series = self.series_df.iloc[batch_indices]
        
        X_volumes = np.empty((len(batch_series), *self.target_shape, 1))
        y = np.empty((len(batch_series), 1))
        
        for i, idx in enumerate(batch_indices):
            series_uid = self.series_df.iloc[idx]['SeriesInstanceUID']
            
            # 检查是否已缓存
            if self.use_cache and series_uid in self.cache:
                processed_volume = self.cache[series_uid]
            else:
                # 加载和预处理3D体积
                volume, _ = load_dicom_series(self.base_path, series_uid)
                processed_volume = preprocess_volume(volume, self.target_shape)
                if self.use_cache:
                    self.cache[series_uid] = processed_volume
            
            # 数据增强
            if self.augment:
                processed_volume = augment_volume(processed_volume)
            
            X_volumes[i] = processed_volume
            
            # 获取标签
            y[i] = self.series_df.iloc[idx].get('Aneurysm Present', 0)
        
        return X_volumes, y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.series_df))
        if self.shuffle:
            np.random.shuffle(self.indices)
        # 清空缓存，确保每个epoch使用不同的数据增强
        self.cache.clear()



def create_3d_classification_model(input_shape, num_classes=1):
    """创建3D动脉瘤分类模型"""
    # 输入层
    inputs = layers.Input(shape=input_shape)
    
    # 3D特征提取
    x = layers.Conv3D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    
    x = layers.Conv3D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    
    x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    
    x = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)
    
    # 添加一些全连接层
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # 输出层
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# 训练设置
TARGET_SHAPE = (64, 64, 64)  # 缩小尺寸以适应内存
BATCH_SIZE = 4  # 小批量以适应内存

# 检查是否有足够的GPU内存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 创建模型
model = create_3d_classification_model((*TARGET_SHAPE, 1), num_classes=1)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# 显示模型架构
model.summary()

# 准备数据 - 使用部分数据进行演示
sample_size = min(50, len(train_df))  # 使用50个样本进行演示
sample_df = train_df.sample(sample_size, random_state=SEED)

# 检查是否有Aneurysm Present列
if 'Aneurysm Present' not in sample_df.columns:
    print("警告: 数据集中没有'Aneurysm Present'列，将使用第一列作为标签")
    # 假设第一列是标签列
    label_col = sample_df.columns[0]
else:
    label_col = 'Aneurysm Present'

train_series, val_series = train_test_split(
    sample_df, test_size=0.2, random_state=SEED, 
    stratify=sample_df[label_col] if label_col in sample_df.columns else None
)

print(f"训练样本数: {len(train_series)}, 验证样本数: {len(val_series)}")

train_generator = AneurysmClassificationGenerator(
    train_series, SERIES_PATH, 
    batch_size=BATCH_SIZE, target_shape=TARGET_SHAPE,
    augment=True  # 训练时使用数据增强
)

val_generator = AneurysmClassificationGenerator(
    val_series, SERIES_PATH, 
    batch_size=BATCH_SIZE, target_shape=TARGET_SHAPE, 
    shuffle=False, augment=False  # 验证时不使用数据增强
)


# 修复ModelCheckpoint的文件名问题
checkpoint_filepath = 'best_model.weights.h5'  # 使用.weights.h5扩展名

# 定义回调函数
callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_auc',
        save_best_only=True,
        save_weights_only=True,  # 只保存权重
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_auc',
        patience=5,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        mode='max',
        verbose=1
    )
]

# 训练模型
print("开始训练模型...")
history = model.fit(
    train_generator,
    epochs=10,  # 减少epoch数量进行演示
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# 加载最佳权重
model.load_weights(checkpoint_filepath)


# 评估模型
print("评估模型...")
val_loss, val_accuracy, val_auc = model.evaluate(val_generator, verbose=1)
print(f"验证集损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}, AUC: {val_auc:.4f}")

# 绘制训练历史
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='训练AUC')
plt.plot(history.history['val_auc'], label='验证AUC')
plt.title('模型AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# 进行预测
print("进行预测...")
predictions = model.predict(val_generator, verbose=1)
print(f"预测结果形状: {predictions.shape}")

# 计算AUC分数
true_labels = []
for i in range(len(val_generator)):
    _, labels = val_generator[i]
    true_labels.extend(labels.flatten())

true_labels = np.array(true_labels)
if len(true_labels) > 0:
    auc_score = roc_auc_score(true_labels, predictions[:len(true_labels)].flatten())
    print(f"最终AUC得分: {auc_score:.4f}")
else:
    print("无法计算AUC得分: 没有真实标签数据")

