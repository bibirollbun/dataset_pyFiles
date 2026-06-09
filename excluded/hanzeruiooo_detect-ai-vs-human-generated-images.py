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


train_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/train.csv')
test_df = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/test.csv')

train_df.head()


import kagglehub

# Download latest version
path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")

print("Path to dataset files:", path)


train_df.info()


test_df.info()


test_df.head()


import pandas as pd
import tensorflow as tf
from tensorflow.keras import applications, layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 数据准备
train_df = pd.read_csv("/kaggle/input/ai-vs-human-generated-dataset/train.csv")  # 假设路径需要调整
base_dir = "/kaggle/input/ai-vs-human-generated-dataset"
img_size = (224, 224)
batch_size = 32

# 数据增强
train_datagen = ImageDataGenerator(
    preprocessing_function=applications.resnet50.preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# 创建数据流
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=base_dir,  # 基础路径
    x_col="file_name",   # 包含相对路径的列
    y_col="label",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="raw",    # 直接使用原始标签
    subset="training"
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=base_dir,
    x_col="file_name",
    y_col="label",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="raw",
    subset="validation"
)

# 构建模型
base_model = applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# 冻结基础模型
base_model.trainable = False

# 添加自定义层
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 训练模型
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size
)

# 微调（可选）
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    initial_epoch=history.epoch[-1],
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size
)

# 保存模型
model.save("ai_vs_human_classifier.h5")



import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 加载完整的测试数据（假设已正确读取）
test_df = pd.read_csv("/kaggle/input/ai-vs-human-generated-dataset/test.csv")  # 请确认实际路径
base_dir = "/kaggle/input/ai-vs-human-generated-dataset"

# 创建测试数据生成器
test_datagen = ImageDataGenerator(
    preprocessing_function=applications.resnet50.preprocess_input
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=base_dir,
    x_col="id",          # 直接使用包含完整相对路径的列
    y_col=None,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False        # 保持原始顺序
)

# 加载训练好的模型
model = tf.keras.models.load_model("ai_vs_human_classifier.h5")

# 进行预测
predictions = model.predict(test_generator)
predictions_labels = (predictions >= 0.49999).astype(int).flatten()

# 构建结果DataFrame
results = pd.DataFrame({
    'id': test_df['id'],    # 保留原始文件名
    'label': predictions_labels
})

# 保存结果（不保留索引）
results.to_csv("submission.csv", index=False)


