# 导入必要的库
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import Sequence

# 设置随机种子保证可重复性
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 数据路径设置（Kaggle默认路径）
BASE_PATH = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train_images")
TEST_IMG_PATH = os.path.join(BASE_PATH, "test_images")

# 读取数据
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# 图像预处理函数（使用Ben Graham的预处理方法）
def preprocess_image(image_path, sigmaX=10):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)
    return image

# 加载并预处理训练图像
X_train = []
y_train = []

for i, row in train_df.iterrows():
    img_path = os.path.join(TRAIN_IMG_PATH, row['id_code'] + '.png')  # 确认训练集是.png格式
    img = preprocess_image(img_path)
    X_train.append(img)
    y_train.append(row['diagnosis'])

X_train = np.array(X_train)
y_train = np.array(y_train)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=SEED,
    stratify=y_train
)

# 数据增强
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator()

# 自定义生成器类
class CustomDataGenerator(Sequence):
    def __init__(self, x, y, batch_size, augmenter=None, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.shuffle = shuffle
        self.indexes = np.arange(len(x))
        self.on_epoch_end()
        super().__init__()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        
        if self.augmenter:
            x_batch = np.array([self.augmenter.random_transform(img) for img in x_batch])
            
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# 初始化生成器
train_generator = CustomDataGenerator(
    X_train, y_train,
    batch_size=32,
    augmenter=train_datagen,
    shuffle=True
)

val_generator = CustomDataGenerator(
    X_val, y_val,
    batch_size=32,
    augmenter=val_datagen,
    shuffle=False
)

# 计算类别权重
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# 构建模型
def build_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(5, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = build_model()

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 回调函数设置
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
)

# 训练参数
EPOCHS = 20

# 训练模型
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Evolution')
plt.show()

# 加载最佳模型进行预测
model = tf.keras.models.load_model('best_model.keras')  # 使用新格式加载

# 预处理测试数据（关键修改：添加异常处理和多后缀支持）
X_test = []
failed_ids = []
valid_extensions = ['.png', '.jpeg', '.jpg']  # 尝试多种图片后缀

for i, row in test_df.iterrows():
    img_found = False
    for ext in valid_extensions:
        img_path = os.path.join(TEST_IMG_PATH, row['id_code'] + ext)
        if os.path.exists(img_path):
            try:
                img = preprocess_image(img_path)
                X_test.append(img)
                img_found = True
                break
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                failed_ids.append(row['id_code'])
                img_found = False
                break
    if not img_found:
        print(f"警告: {row['id_code']} 没有找到有效图像文件，使用全黑占位")
        X_test.append(np.zeros((224, 224, 3)))  # 占位黑图

X_test = np.array(X_test)

# 生成预测结果
print("正在生成预测...")
preds = model.predict(X_test)
pred_classes = np.argmax(preds, axis=1)

# 生成提交文件（关键修改：显式使用绝对路径）
submission = pd.DataFrame({
    'id_code': test_df['id_code'],
    'diagnosis': pred_classes
})
submission.to_csv('/kaggle/working/submission.csv', index=False)  # 绝对路径确保写入

# 验证文件生成
if os.path.exists('/kaggle/working/submission.csv'):
    print("提交文件已成功生成！")
    print("文件路径: /kaggle/working/submission.csv")
else:
    print("错误: 提交文件生成失败！")

