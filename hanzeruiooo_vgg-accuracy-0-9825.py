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
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 路径和文件
data_file = '/kaggle/input/skku-2024-1-machine-learning-third-project/train.csv'
image_dir = '/kaggle/input/skku-2024-1-machine-learning-third-project/SceneImages/'
test_data_file = '/kaggle/input/skku-2024-1-machine-learning-third-project/test.csv'

# 加载标签数据
df = pd.read_csv(data_file)

# 加载测试集数据
test_df = pd.read_csv(test_data_file)

# 图像大小
img_size = (224, 224)
num_classes = df['label'].nunique()
df['label'] = df['label'].astype(str) 


# 数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 图像像素值归一化到[0, 1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20%的数据作为验证集
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 使用 flow_from_dataframe 读取图像数据，并进行训练集与验证集的拆分
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,  # 这里直接指定图像目录
    x_col='image_name',  # 使用 'image_name' 列作为图片文件名
    y_col='label',
    target_size=img_size,
    batch_size=32,
    class_mode='sparse',  # 改为 sparse，标签不需要独热编码
    subset='training'  # 训练集
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,  # 这里直接指定图像目录
    x_col='image_name',  # 使用 'image_name' 列作为图片文件名
    y_col='label',
    target_size=img_size,
    batch_size=32,
    class_mode='sparse',  # 标签为整数，不需要独热编码
    subset='validation'  # 验证集
)

# 创建输入层
input_tensor = Input(shape=(224, 224, 3))

# 加载VGG16模型，不包含顶层（即不包含全连接层），并设置输入形状
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# 冻结VGG16的前几层，保持其预训练权重不变
for layer in base_model.layers:
    layer.trainable = False

# 解冻顶部几层卷积层（fine-tuning）
for layer in base_model.layers[-10:]:  # 解冻最后10层
    layer.trainable = True

# 添加全连接层和输出层
x = GlobalAveragePooling2D()(base_model.output)  # 池化层，转化为一个向量
x = Dense(256, activation='relu', kernel_regularizer='l2')(x)  # 全连接层 + L2正则化
x = Dropout(0.5)(x)  # Dropout层，防止过拟合
x = Dense(num_classes, activation='softmax')(x)  # 输出层，用于多分类

# 构建模型
model = Model(inputs=input_tensor, outputs=x)

# 打印模型摘要
model.summary()

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',  # 使用 sparse_categorical_crossentropy
              metrics=['accuracy'])

# 回调函数：检查点和早停
checkpoint = ModelCheckpoint('best_vgg_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)



from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import os

# 加载训练好的模型
model = load_model('best_vgg_model.keras')

# 定义图像目录
image_dir = '/kaggle/input/skku-2024-1-machine-learning-third-project/SceneImages/'

# 读取 test_df 并获取图像名称
test_df = pd.read_csv('/kaggle/input/skku-2024-1-machine-learning-third-project/test.csv')  # 根据需要修改文件路径
image_names = test_df['image_name'].tolist()

# 初始化一个空的列表，用来存储预处理后的图像
test_images = []

# 加载和预处理每个图像
for image_name in image_names:
    img_path = os.path.join(image_dir, image_name)
    img = image.load_img(img_path, target_size=(224, 224))  # 修改为模型输入的尺寸，例如 224x224
    img_array = image.img_to_array(img)  # 转换为数组
    img_array = np.expand_dims(img_array, axis=0)  # 增加一个维度作为 batch_size 维度
    img_array = img_array / 255.0  # 归一化处理，视模型训练时是否有此步骤
    test_images.append(img_array)

# 将图像列表转换为 numpy 数组
test_images = np.vstack(test_images)

# 对 test_images 进行预测
predictions = model.predict(test_images)

# 获取预测的标签（通过 np.argmax 获取每个样本的类别）
predicted_labels = np.argmax(predictions, axis=1)

# 将预测结果与 image_name 合并
test_df['label'] = predicted_labels

# 将 'image_name' 列设置为索引
output_df = test_df[['image_name', 'label']].set_index('image_name')

# 打印输出
print(output_df)

# 保存到 CSV 文件，包含索引
output_df.to_csv('predicted.csv', index=False)


