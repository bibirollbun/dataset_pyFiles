# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import psutil

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout


data = pd.read_csv("/kaggle/input/severstal-steel-defect-detection/train.csv")


l1=[]
l2=[] 
for img,ClassId,EncodedPixels in tqdm(data.values):
    image=cv2.imread("/kaggle/input/severstal-steel-defect-detection/train_images/{}".format(img),cv2.IMREAD_GRAYSCALE)
    image=cv2.resize(image, (120,120))
    l1.append(image)
    l2.append(ClassId)


encoder = LabelEncoder()

X= np.array(l1)
X = X/255

y = encoder.fit_transform(l2)
y = to_categorical(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,shuffle=True, random_state=42)


model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(120,120,1),activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(256,activation="relu"))
model.add(Dense(4,activation="softmax"))


early_stopping = tf.keras.callbacks.EarlyStopping(patience=5,min_delta=0.001,restore_best_weights=True)


model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=["accuracy"])


history = model.fit(X_train,y_train,epochs=15,validation_data=(X_test,y_test),batch_size=1,
                    verbose=1, callbacks=[early_stopping])


def rle_encode(mask):
    rle_encoded_mask = []
    prev_value = mask[0][0]  # Access first element
    count = 1

    for value in mask[1:]:
        if value.any() != prev_value:  # Check if any element in value differs from prev_value
            rle_encoded_mask.append(int(prev_value))
            prev_value = value[0]  # Access first element for next comparison
            count = 1
        else:
            count += 1

        rle_encoded_mask.append(int(prev_value))
    return ' '.join(map(str, rle_encoded_mask))



test_images_dir = "/kaggle/input/severstal-steel-defect-detection/test_images"
test_filenames = os.listdir(test_images_dir)


# 类别预测和 RLE 编码掩码
y_pred = []
encoded_pixels = []


for filename in tqdm(test_filenames):
    # 图像加载和转换
    image = cv2.imread(os.path.join(test_images_dir, filename), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (120, 120))
    _, mask = cv2.threshold(image, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

    # RLE 编码掩码
    rle_encoded_mask = rle_encode(mask)
    encoded_pixels.append(rle_encoded_mask)

    # 类别预测
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_class = np.argmax(prediction, axis=1)[0] + 1
    y_pred.append(predicted_class)

# 创建 dataframe
submission_df = pd.DataFrame({
    "ImageId": test_filenames,
    "EncodedPixels": encoded_pixels,
    "ClassId": y_pred
})


import resource
import psutil
 
# 获取当前进程的内存使用情况（以字节为单位）
memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(memory_usage)
 
# 转换为GB
memory_usage_gb = memory_usage / 1024 / 1024 / 1024
print(f"Memory usage in GB: {memory_usage_gb:.2f}")



def mask_to_rle(mask):
    """
    将二值掩码转换为游程编码 (RLE)，假设目标区域是255，背景区域是0
    :param mask: 二值掩码，形状为 (height, width)，目标区域为 255，背景区域为 0
    :return: 游程编码字符串，记录每段255的起始位置和长度
    """
    pixels = np.array(mask).flatten().astype(int)  # 将掩码展平成一维数组
    runs = []
    prev = 0  # 初始背景值
    start = None  # 当前连续255区域的起始位置
    length = 0  # 当前255区域的长度

    for i, pixel in enumerate(pixels):
        if pixel == 255:
            if prev == 0:  # 遇到新的255区域，开始记录
                start = i  # 记录新区域的起始位置
            length += 1
        else:
            if prev == 255:  # 如果遇到0且之前是255区域，结束255区域
                runs.append((start, length))
                length = 0  # 重置长度

        prev = pixel  # 更新前一个像素的值

    # 处理最后一个255区域，如果它以255结尾
    if prev == 255:
        runs.append((start, length))

    # 将每个连续区域的起始位置和长度拼接成字符串
    rle = ' '.join([f"{start} {length}" for start, length in runs])
    return rle


# 修改输出格式

# data = pd.read_csv("submission.csv", header=None)

imgs_classid = []
predict = []

for img,EncodedPixels,ClassId in tqdm(submission_df.values):
    for i in range(1, 4):
        imgs_classid.append(f'{img}_{i}')
        predict.append('')
    print(len(predict), -(4-ClassId+1))
    predict[-(4-ClassId+1)] = mask_to_rle(EncodedPixels.split())

df = pd.DataFrame({
    "ImageId_ClassId": imgs_classid,
    "EncodedPixels": predict
})
df.to_csv("/kaggle/working/submission.csv", index=False)

