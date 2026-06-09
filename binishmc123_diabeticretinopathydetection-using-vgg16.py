import numpy as np  
import pandas as pd 
import os 

input_dir = '/kaggle/input'
file_paths = []

for dirname, _, filenames in os.walk(input_dir):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        file_paths.append(file_path)

print("Total files found:", len(file_paths))
print("files:", file_paths[:])  

expected_files = ['trainLabels.csv.zip', 'sampleSubmission.csv.zip']
for ef in expected_files:
    if not any(ef in path for path in file_paths):
        print(f"Warning: Expected file {ef} not found!")



import numpy as np  # 用於數值計算
import matplotlib.pyplot as plt  # 用於資料視覺化
import seaborn as sns  # 用於強化視覺化的繪圖風格
import tensorflow as tf  # TensorFlow 深度學習框架
from glob import glob  # 用於查找文件路徑
from skimage.io import imread  # 用於圖像讀取

from tensorflow.keras.applications import EfficientNetB0  # 預訓練模型
from tensorflow.keras.optimizers import Adam  # 優化器
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # 損失函數
from tensorflow.keras.layers import Dense, Flatten, Dropout  # 常用層
from tensorflow.keras.models import Model  # 模型
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping  # 訓練回調
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 圖像生成工具
from tensorflow.keras.utils import to_categorical  # 類別處理工具
from sklearn.metrics import classification_report, confusion_matrix  # 評估工具
from sklearn.model_selection import train_test_split  # 資料分割工具

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # 只忽略FutureWarning

print("All necessary modules have been successfully imported!")



import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    cm = np.round(cm,2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Loading data
info = pd.read_csv("../input/prepossessed-arrays-of-binary-data/1000_Binary Dataframe")
info = info.drop('Unnamed: 0', axis=1)
Binary_90 = np.load('../input/prepossessed-arrays-of-binary-data/1000_Binary_images_data_90.npz')
X_90 = Binary_90['a']
Binary_128 = np.load('../input/prepossessed-arrays-of-binary-data/1000_Binary_images_data_128.npz')
X_128 = Binary_128['a']
Binary_264 = np.load('../input/prepossessed-arrays-of-binary-data/1000_Binary_images_data_264.npz')
X_264 = Binary_264['a']
y = info['level'].values

# Reshape images
X_90 = X_90.reshape(1000, 90, 90, 3)
X_128 = X_128.reshape(1000, 128, 128, 3)
X_264 = X_264.reshape(1000, 264, 264, 3)

# Display images
plt.title("90*90*3 Image")
plt.imshow(X_90[1])
plt.show()

plt.title("128*128*3 Image")
plt.imshow(X_128[1])
plt.show()

plt.title("264*264*3 Image")
plt.imshow(X_264[1])
plt.show()


# Prepare data for training
X = np.array(X_264)
Y = np.array(y)
Y = to_categorical(Y, 5)
x_train, x_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test1, y_test1, test_size=0.5, random_state=42)

print(f"Training set size: {len(x_train)}")
print(f"Validation set size: {len(x_val)}")
print(f"Test set size: {len(x_test)}")


# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)


# Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=2,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0.001
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

callbacks = [reduce_lr, early_stopping]


# Model with transfer learning (VGG16)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn import metrics  
# 使用 Functional API 來構建模型
input_shape = (264, 264, 3)
inputs = Input(shape=input_shape)

# 加載預訓練的 VGG16 模型，不包括頂部的全連接層
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

# 凍結預訓練模型的所有層
for layer in base_model.layers:
    layer.trainable = False

# 添加自定義的全連接層
x = Flatten()(base_model.output)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
outputs = Dense(5, activation='softmax')(x)

# 定義模型
model = Model(inputs, outputs)

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])

# 顯示模型結構
model.summary()



# Train the model

history = model.fit(
    train_datagen.flow(x_train, y_train, batch_size=8),
    validation_data=val_datagen.flow(x_val, y_val),
    epochs=20,
    callbacks=callbacks,
)

# Evaluate the model
evaluation = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {evaluation[1]*100:.2f}%')

# Predictions and report
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(model.predict(x_test), axis=-1)

print("Performance Report:")
print('Accuracy score:', metrics.accuracy_score(y_test_labels, y_pred_labels))
print('Precision score:', metrics.precision_score(y_test_labels, y_pred_labels, average='weighted'))
print('Recall score:', metrics.recall_score(y_test_labels, y_pred_labels, average='weighted'))
print('F1 Score:', metrics.f1_score(y_test_labels, y_pred_labels, average='weighted'))
print('Cohen Kappa Score:', metrics.cohen_kappa_score(y_test_labels, y_pred_labels))
print('\t\tClassification Report:\n', classification_report(y_test_labels, y_pred_labels))


import matplotlib.pyplot as plt

# Confusion Matrix visualization
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Extracting the metrics using correct keys
auc = history.history['AUC']
val_auc = history.history['val_AUC']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting the metrics
epochs = range(len(auc))
plt.figure(figsize=(18, 4.8))

# Plotting Training and Validation AUC
plt.subplot(1, 3, 1)
plt.plot(epochs, auc, 'r', label='Training AUC')
plt.plot(epochs, val_auc, 'b', label='Validation AUC')
plt.ylim(0, 1)
plt.title('Training and Validation AUC')
plt.legend(loc=0)

# Plotting Training and Validation Loss
plt.subplot(1, 3, 2)
plt.plot(epochs, loss, 'y-.', label='Training Loss')
plt.plot(epochs, val_loss, 'g-.', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylim(0, 2)
plt.legend(loc=0)

plt.show()

