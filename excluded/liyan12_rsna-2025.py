# 步骤1: 环境设置与数据加载
# 安装必要的库（在Kaggle Notebook中通常不需要运行，因为预装好了）
!pip install pydicom opencv-python scikit-image

import numpy as np
import pandas as pd
import pydicom
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 设置随机种子以确保可重复性
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 定义文件路径
BASE_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection'
TRAIN_CSV_PATH = f'{BASE_PATH}/train.csv'
SERIES_PATH = f'{BASE_PATH}/series'

# 加载训练数据
train_df = pd.read_csv(TRAIN_CSV_PATH)
print(f"训练集大小: {train_df.shape}")

# 正确定义标签列（14个标签）
label_columns = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

# 特征列
feature_columns = ['PatientAge', 'PatientSex', 'Modality']

print("步骤1完成: 环境设置与数据加载")


# 数据可视化 - 标签分布
plt.figure(figsize=(15, 8))
label_counts = train_df[label_columns].sum()
plt.bar(range(len(label_counts)), label_counts.values)
plt.xticks(range(len(label_counts)), label_counts.index, rotation=45, ha='right')
plt.title('Distribution of training set labels')#训练集标签分布
plt.ylabel('Distribution of training set labels')#样本数量
plt.tight_layout()
plt.show()

# 数据可视化 - 患者年龄分布
plt.figure(figsize=(10, 6))
plt.hist(train_df['PatientAge'], bins=30, edgecolor='black')
plt.title('Age distribution of patients')#患者年龄分布
plt.xlabel('age')#年龄
plt.ylabel('frequency')#频数
plt.show()

# 数据可视化 - 患者性别分布
plt.figure(figsize=(8, 6))
gender_counts = train_df['PatientSex'].value_counts()
plt.bar(gender_counts.index, gender_counts.values)
plt.title('Distribution of patients genders')#患者性别分布
plt.xlabel('Gender')#性别
plt.ylabel('Frequency')#频数
plt.xticks(range(len(gender_counts)), gender_counts.index)
plt.show()


# 步骤2: 数据预处理
# 编码分类变量
label_encoders = {}
for col in ['PatientSex', 'Modality']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    label_encoders[col] = le
    print(f"{col}编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 标准化数值特征
scaler = StandardScaler()
train_df['PatientAge'] = scaler.fit_transform(train_df[['PatientAge']])

# 计算类别权重
class_weights = {}
for i, col in enumerate(label_columns):
    positive_count = train_df[col].sum()
    negative_count = len(train_df) - positive_count
    class_weights[i] = negative_count / (positive_count + 1e-6)  # 防止除零错误

print("\n类别权重:", class_weights)

# 数据可视化 - 患者性别分布（编码后）
plt.figure(figsize=(8, 6))
gender_counts = train_df['PatientSex'].value_counts()
plt.bar(range(len(gender_counts)), gender_counts.values)
plt.title('Distribution of patients genders')#患者性别分布
plt.xlabel('Gender')#性别
plt.ylabel('Frequency')#频数
plt.xticks(range(len(gender_counts)), [f'{idx} ({label_encoders["PatientSex"].classes_[idx]})' for idx in gender_counts.index])
plt.show()

print("步骤2完成: 数据预处理")


# 步骤3: 图像处理函数
# DICOM图像处理函数
def read_dicom_file(path):
    """读取DICOM文件并提取像素数据和元信息"""
    try:
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        
        # 应用模态特定的预处理
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            image = image * dicom.RescaleSlope + dicom.RescaleIntercept
        
        return image, dicom
    except:
        # 如果无法读取DICOM文件，返回零数组
        return np.zeros((512, 512)), None

def preprocess_medical_image(image, target_size=(256, 256)):
    """
    预处理医学图像
    """
    # 调整大小
    if image.shape != target_size:
        image = cv2.resize(image, target_size)
    
    # 对比度限制自适应直方图均衡化（CLAHE）
    image = exposure.equalize_adapthist(image, clip_limit=0.03)
    
    # 标准化
    image = (image - np.mean(image)) / (np.std(image) + 1e-6)
    
    # 将单通道图像复制为三通道（适应EfficientNet）
    image = np.stack([image, image, image], axis=-1)
    
    return image

print("步骤3完成: 图像处理函数定义")


# 可视化一些样本图像
def visualize_sample_images(df, num_samples=5):
    """可视化样本图像"""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for i in range(num_samples):
        sample_row = df.iloc[i]
        image_path = f"{SERIES_PATH}/{sample_row['SeriesInstanceUID']}.dcm"
        
        image, _ = read_dicom_file(image_path)
        processed_image = preprocess_medical_image(image)
        
        axes[i].imshow(processed_image[:, :, 0], cmap='gray')
        axes[i].set_title(f"样本 {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 可视化一些样本
print("随机样本图像可视化:")
visualize_sample_images(train_df)


# 步骤4: 数据生成器
# 创建改进的数据生成器
class MedicalDataGenerator(tf.keras.utils.Sequence):
    """自定义医学数据生成器"""
    
    def __init__(self, df, base_path, batch_size=32, target_size=(256, 256), shuffle=True, **kwargs):
        # 调用父类初始化方法
        super().__init__(**kwargs)
        
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        X_images = np.empty((len(batch_df), *self.target_size, 3))  # 改为3通道
        X_features = np.empty((len(batch_df), len(feature_columns)))
        y = np.empty((len(batch_df), len(label_columns)))
        
        for i, idx in enumerate(batch_indices):
            row = self.df.iloc[idx]
            
            # 加载和预处理图像
            image_path = f"{self.base_path}/{row['SeriesInstanceUID']}.dcm"
            image, _ = read_dicom_file(image_path)
            processed_image = preprocess_medical_image(image, self.target_size)
            X_images[i] = processed_image
            
            # 提取特征 (已经预处理过)
            X_features[i, 0] = row['PatientAge']  # 已经标准化
            X_features[i, 1] = row['PatientSex']  # 已经编码
            X_features[i, 2] = row['Modality']    # 已经编码
            
            # 提取标签
            y[i] = row[label_columns].values.astype(np.float32)
        
        # 返回格式改为字典形式，符合Keras多输入模型的要求
        return {'image_input': X_images, 'feature_input': X_features}, y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

print("步骤4完成: 数据生成器定义")


# 步骤5: 模型构建
# 模型架构
def create_multi_input_model(image_shape=(256, 256, 3), num_features=3, num_classes=14):
    """
    创建多输入模型，同时处理图像和特征
    """
    # 图像输入分支
    image_input = tf.keras.Input(shape=image_shape, name='image_input')
    
    # 使用EfficientNetB0作为基础模型
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=image_shape
    )
    
    # 冻结基础层
    base_model.trainable = False
    
    x = base_model(image_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    image_features = layers.Dense(128, activation='relu')(x)
    
    # 特征输入分支
    feature_input = tf.keras.Input(shape=(num_features,), name='feature_input')
    feature_branch = layers.Dense(16, activation='relu')(feature_input)
    feature_branch = layers.Dropout(0.3)(feature_branch)
    
    # 合并分支
    combined = layers.Concatenate()([image_features, feature_branch])
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)
    
    # 输出层
    outputs = layers.Dense(num_classes, activation='sigmoid')(combined)
    
    # 创建模型
    model = tf.keras.Model(inputs=[image_input, feature_input], outputs=outputs)
    
    return model

# 创建模型
model = create_multi_input_model()
model.summary()

print("步骤5完成: 模型构建")


# 步骤6: 模型编译与数据准备
# 自定义加权损失函数
def weighted_binary_crossentropy(class_weights):
    weights = tf.constant(list(class_weights.values()), dtype=tf.float32)
    
    def loss_function(y_true, y_pred):
        # 计算基础损失
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # 应用类别权重
        weight_vector = tf.reduce_sum(weights * y_true, axis=1)
        weighted_bce = bce * weight_vector
        
        return tf.reduce_mean(weighted_bce)
    
    return loss_function

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss=weighted_binary_crossentropy(class_weights),
    metrics=['accuracy', 'AUC']
)

# 划分训练集和验证集
train_data, val_data = train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=SEED,
    stratify=train_df['Aneurysm Present']
)

# 创建数据生成器
train_generator = MedicalDataGenerator(train_data, SERIES_PATH, batch_size=16)
val_generator = MedicalDataGenerator(val_data, SERIES_PATH, batch_size=16, shuffle=False)

# 设置回调函数
callbacks = [
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_auc',
        save_best_only=True,
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
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("步骤6完成: 模型编译与数据准备")


# 步骤7: 模型训练
# 训练模型
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("步骤7完成: 模型训练")


# 绘制训练历史 - Loss和AUC曲线
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='training loss')#训练损失
plt.plot(history.history['val_loss'], label='Validation loss')#验证损失
plt.title('Model loss')#模型损失
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制AUC曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['AUC'], label='Training AUC')#训练
plt.plot(history.history['val_AUC'], label='Verify AUC')#验证
plt.title('Model AUC')#模型
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()



# 步骤8: 模型评估
# 评估模型
def evaluate_model(model, generator):
    # 获取所有预测和真实标签
    all_y_true = []
    all_y_pred = []
    
    for i in range(len(generator)):
        X, y_true = generator[i]
        y_pred = model.predict(X, verbose=0)
        
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
    
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    
    # 计算每个标签的AUC
    auc_scores = []
    for i, col in enumerate(label_columns):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_scores.append(auc)
            print(f"{col}: AUC = {auc:.4f}")
        except:
            auc_scores.append(0.5)
            print(f"{col}: AUC计算失败，使用默认值0.5")
    
    # 计算加权AUC
    weights = [1] * 13 + [13]  # 13个部位权重1，存在性权重13
    weighted_auc = np.average(auc_scores, weights=weights)
    
    # 计算最终得分
    final_score = 0.5 * (auc_scores[-1] + np.mean(auc_scores[:-1]))
    
    print(f"\n加权AUC: {weighted_auc:.4f}")
    print(f"最终得分: {final_score:.4f}")
    
    return y_true, y_pred, auc_scores, weighted_auc, final_score

# 在验证集上评估模型
print("模型评估结果:")
y_true, y_pred, auc_scores, weighted_auc, final_score = evaluate_model(model, val_generator)

print("步骤8完成: 模型评估")


# 步骤9: 提交文件生成
# 创建提交函数
def create_submission(model, test_df, base_path, label_encoders, scaler):
    """
    创建竞赛提交文件
    """
    # 预处理测试数据（与训练数据相同的方式）
    test_df_processed = test_df.copy()
    
    # 编码分类变量
    for col, le in label_encoders.items():
        # 处理未知类别
        test_df_processed[col] = test_df_processed[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )
    
    # 标准化数值特征
    if 'PatientAge' in test_df_processed.columns:
        test_df_processed['PatientAge'] = scaler.transform(test_df_processed[['PatientAge']])
    
    # 创建测试数据生成器
    test_generator = MedicalDataGenerator(
        test_df_processed, 
        base_path, 
        batch_size=16, 
        shuffle=False
    )
    
    # 生成预测
    predictions = model.predict(test_generator, verbose=1)
    
    # 创建提交DataFrame
    submission_df = pd.DataFrame(predictions, columns=label_columns)
    submission_df.insert(0, 'ID', test_df['SeriesInstanceUID'])
    
    # 保存为CSV
    submission_df.to_csv('submission.csv', index=False)
    
    return submission_df

# 注意：在实际使用中，你需要加载测试集数据
# test_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/test.csv')
# submission = create_submission(model, test_df, SERIES_PATH, label_encoders, scaler)

print("步骤9完成: 提交文件生成函数定义")




