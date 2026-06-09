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
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import preprocess_input



# 路径和文件
data_file = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv'
image_test = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/'
image_train = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train/'

# 加载标签数据
df = pd.read_csv(data_file)
df['image_path'] = df['image'].apply(lambda x: os.path.join(image_train, x))

n_classes = df['label'].nunique()

df.head()  # 显示数据的前几行，检查路径和标签


# 初始化空列表 x 用于存储图像
x = []

# 遍历每一行读取图像
for index, row in df.iterrows():
    image_path = row['image_path']  # 获取图像路径
    img = cv2.imread(image_path)  # 使用 cv2 读取图像
    
    if img is not None:
        img_resized = cv2.resize(img, (256, 256))  # 调整图像尺寸为 (256, 256)
        x.append(img_resized)  # 将读取的图像添加到列表 x 中
    else:
        print(f"图像 {row['image_path']} 读取失败")  # 打印失败的路径

# x 列表现在包含了所有读取的图像
print(f"总共有 {len(x)} 张图像被读取")





# 将图像转换为 NumPy 数组
x = np.array(x)

# 标签映射并进行 one-hot 编码
y = df['label'].map({'real': 1, 'editada': 0})
y = np.array(y)
y = to_categorical(y, num_classes=2)  # 二分类

# 分割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 检查转换后的结果
print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_test.shape: {y_test.shape}")




from tensorflow.keras.regularizers import l2
# 加载预训练的VGG16卷积基（不包括顶部的全连接层）
vgg16_model = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

# 冻结VGG16的卷积层
for layer in vgg16_model.layers:
    layer.trainable = False

# 创建一个新的模型
model_fine_tuning = Sequential()

# 将VGG16的卷积基添加到新模型中
model_fine_tuning.add(vgg16_model)  # 添加VGG16卷积基
model_fine_tuning.add(Flatten())  # 将卷积特征图展平

# 添加新的全连接层并进行正则化
model_fine_tuning.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))  # L2正则化
model_fine_tuning.add(Dropout(0.3))  # Dropout层，减少过拟合
model_fine_tuning.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))  # 较小的全连接层
model_fine_tuning.add(Dropout(0.3) ) # 再次使用Dropout层

# 输出层
model_fine_tuning.add(Dense(2, activation='softmax'))  # 对于二分类问题，使用softmax

# 查看模型架构
model_fine_tuning.summary()




# 编译模型
model_fine_tuning.compile(loss='binary_crossentropy', 
                          optimizer=Adam(), 
                          metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input)  # 使用VGG16的预处理函数

# 对原始图像进行增强，并进行训练
history = model_fine_tuning.fit(datagen.flow(x_train, y_train, batch_size=32),
                                epochs=20,
                                validation_data=(x_test, y_test),
                                callbacks=[ModelCheckpoint('best_model.keras', save_best_only=True),
                                           EarlyStopping(patience=5)])


import matplotlib.pyplot as plt

# 获取训练过程中的损失和准确率数据
history_dict = history.history
loss = history_dict['loss']
accuracy = history_dict['accuracy']
val_loss = history_dict['val_loss']
val_accuracy = history_dict['val_accuracy']

# 绘制损失图
plt.figure(figsize=(12, 6))

# 损失图
plt.subplot(1, 2, 1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 准确率图
plt.subplot(1, 2, 2)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 展示图像
plt.tight_layout()
plt.show()



import os
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

# 设置测试数据文件夹路径
image_test = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test/'

# 获取测试数据中的所有图像文件名
image_files = os.listdir(image_test)

# 只获取图片文件（jpg和png）
image_files = [f for f in image_files if f.endswith('.jpg') or f.endswith('.png')]

# 用于存储预测结果
predictions = []

# 加载训练好的模型
model_fine_tuning = load_model('best_model.keras')

# 处理每张图像并进行预测
for img_file in image_files:
    # 构建图像的完整路径
    img_path = os.path.join(image_test, img_file)
    
    # 使用 OpenCV 读取图像并调整大小
    img = cv2.imread(img_path)
    if img is not None:
        img_resized = cv2.resize(img, (256, 256))  # 调整为训练时使用的大小
        
        # 将图像转换为数组并进行预处理
        img_array = np.expand_dims(img_resized, axis=0)  # 增加batch维度
        img_array = preprocess_input(img_array)  # 使用VGG16的预处理函数
        
        # 使用模型进行预测
        prediction = model_fine_tuning.predict(img_array)
        
        # 预测结果为概率值，转换为0或1（二分类）
        label = 1 if prediction[0][1] > 0.5 else 0
        
        # 将图像文件名和预测标签存储到列表中
        predictions.append([img_file, label])
    else:
        print(f"图像 {img_file} 读取失败")  # 如果图像读取失败，输出失败信息

# 将结果转换为DataFrame
df_predictions = pd.DataFrame(predictions, columns=['image', 'label'])

# 显示或保存结果
print(df_predictions.head())

df_predictions.to_csv('predictions.csv', index=False)


