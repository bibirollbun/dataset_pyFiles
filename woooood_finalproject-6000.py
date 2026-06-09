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
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pydicom
from skimage.transform import resize
import matplotlib.patches as patches


from sklearn.utils import shuffle

train_labels = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
train_labels.shape
train_labels = shuffle(train_labels, random_state=0)
train_labels.reset_index(inplace=True, drop=True)



input_size = 244

# resize img and make the max dimension be input_size
def format_image(img, box):
    height, width = img.shape 
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    
    # 0 padding
    new_image = np.zeros((input_size, input_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized
    
    # resize the box too
    x, y, w, h = (box[0], box[1], box[2], box[3]) if box[0] else (0.0,0.0,0.0,0.0)
    new_box = [int((x)/ r), int((y)/ r), int(w/ r), int(h/ r)] if box[0] else [0.0,0.0,0.0,0.0]

    return new_image, new_box


dcm_path = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm'

image_array = pydicom.dcmread(dcm_path).pixel_array
# image_array = cv.resize(image_array, (224, 224))

print(image_array.shape)

# 繪圖
fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # 建立圖表與子圖
ax.imshow(image_array, cmap='bone')          # 顯示影像

# 繪製標註框
rect = patches.Rectangle((264.0, 152.0), 213.0, 379.0, 
                         edgecolor='r', facecolor='none', linewidth=2)
ax.add_patch(rect)                           # 在軸上新增標註框

plt.show()    



datapath = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm'
temp_img = pydicom.dcmread(datapath).pixel_array
temp_box = [264.0, 152.0, 213.0, 379.0]

temp_img_formated, box = format_image(temp_img, temp_box)
print(box)
temp_color_img = cv.cvtColor(temp_img_formated, cv.COLOR_GRAY2RGB)

cv.rectangle(temp_color_img, box, (0, 255, 0), 1)

plt.imshow(temp_color_img)
# plt.axis("off")
plt.show()


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disabling verbose tf logging

# uncomment the following line if you want to force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print(tf.__version__)


from tqdm import tqdm  # 引入 tqdm
import os
import pydicom
import numpy as np
import tensorflow as tf
import math

def data_load(dataset, batch_size=3, full_data_path="/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/", image_ext=".dcm",ds_type='not_trian'):
    X = []
    Y = []

    # 使用 tqdm 包裝迭代器，顯示進度條
    for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Loading data"):
        filename = row['patientId']  # 根據欄位名稱取值

        # 讀取 DICOM 影像
        temp_img = pydicom.dcmread(os.path.join(full_data_path, filename + image_ext)).pixel_array
        
        # 確認標註框是否有效
        temp_box = [row['x'], row['y'], row['width'], row['height']] if not math.isnan(row['x']) else [0.0, 0.0, 0.0, 0.0]

        # 格式化影像與標註框
        img, box = format_image(temp_img, temp_box)

        # 正規化影像與標註框
        img = img.astype(float) / 255.
        box = np.asarray(box, dtype=float) / input_size
        
        # 合併標註與目標標籤
        label = np.append(box, row['Target'])

        # 將資料加入 X 和 Y
        X.append(img)
        Y.append(label)
    # print(len(X))
    # print(len(Y))
    
    # 將資料轉換為 TensorFlow 格式
    X = np.array(X)
    # if ds_type=="train":
    #     X = np.tile(X, (3, 1, 1))  # 重複 3 次，沿第 0 軸 (樣本數量) 增加
    #     Y = np.array(Y)  
    #     Y = np.tile(Y,(3 ,1))
    #     np.random.shuffle(X)
    #     print(len(X))
    data_X_len = len(X)
    X = np.expand_dims(X, axis=3)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)
    
    # 建立 TensorFlow 資料集
    result = tf.data.Dataset.from_tensor_slices((X, Y))

    return result,data_X_len
raw_train_ds,train_len = data_load(train_labels[:6001],ds_type="train")
print(train_len)
raw_valid_ds,valid_len = data_load(train_labels[6001:6301],ds_type="not train")
raw_test_ds, test_len = data_load(train_labels[6301:6501],ds_type="not train")


# plt.figure(figsize=(20, 10))
# BATCH_SIZE = 32
# i = 0
# for images, labels in raw_train_ds:
        
#         print(labels)
#         ax = plt.subplot(4, BATCH_SIZE//4, i + 1)
#         label = labels[4]
#         box = (labels[:4] * input_size)
#         box = tf.cast(box, tf.int32)

#         image = images.numpy().astype("float") * 255.0
#         image = image.astype(np.uint8)
#         image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

#         color = (0, 0, 255)
#         if label > 0:
#             color = (0, 255, 0)

#         cv.rectangle(image_color, box.numpy(), color, 2)

#         plt.imshow(image_color)
#         plt.axis("off")
#         i += 1


import os
cpu_count = os.cpu_count()
print(f"Available CPU cores: {cpu_count}")



CLASSES = 2

def format_instance(image, label):
    return image, (tf.one_hot(int(label[4]), CLASSES), [label[0], label[1], label[2], label[3]])


BATCH_SIZE = 32

# see https://www.tensorflow.org/guide/data_performance

def tune_training_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat() # The dataset be repeated indefinitely.
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def tune_validation_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(len(dataset))# // 4)
    dataset = dataset.repeat()
    return dataset


train_ds = tune_training_ds(raw_train_ds)
validation_ds = tune_validation_ds(raw_valid_ds)



plt.figure(figsize=(20, 10))
for images, labels in train_ds.take(1):
    for i in range(BATCH_SIZE):
        # print(labels.shape)
        ax = plt.subplot(4, BATCH_SIZE//4, i + 1)
        label = labels[0][i]
        box = (labels[1][i] * input_size)
        box = tf.cast(box, tf.int32)

        image = images[i].numpy().astype("float") * 255.0
        image = image.astype(np.uint8)
        image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        color = (0, 0, 255)
        if label[0] > 0:
            color = (0, 255, 0)

        cv.rectangle(image_color, box.numpy(), color, 2)

        plt.imshow(image_color)
        plt.axis("off")


DROPOUT_FACTOR = 0.5

def build_feature_extractor(inputs):

    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(input_size, input_size, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_FACTOR)(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    return x

def build_feature_extractor2(inputs):

    x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(input_size, input_size, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_FACTOR)(x)
    x = tf.keras.layers.AveragePooling2D(2,2)(x)

    return x

def build_model_adaptor(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return x

def build_classifier_head(inputs):
    return tf.keras.layers.Dense(CLASSES, activation='softmax', name = 'classifier_head')(inputs)

def build_regressor_head(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    # x = tf.keras.layers.Dense(10000, activation='relu')(x)
    x = tf.keras.layers.Dense(1000, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return tf.keras.layers.Dense(4, name = 'regressor_head')(x)

def build_model(inputs):
    
    feature_extractor = build_feature_extractor(inputs)

    model_adaptor = build_model_adaptor(feature_extractor)

    classification_head = build_classifier_head(model_adaptor)

    regressor_head = build_regressor_head(feature_extractor)

    model = tf.keras.Model(inputs = inputs, outputs = [classification_head, regressor_head])

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss = {'classifier_head' : 'categorical_crossentropy', 'regressor_head' : 'mse' }, 
              metrics = {'classifier_head' : 'accuracy', 'regressor_head' : 'mse' })

    return model


model = build_model(tf.keras.layers.Input(shape=(input_size, input_size, 1,)))

model.summary()


# plot_model requires graphviz & pydot
# see https://github.com/XifengGuo/CapsNet-Keras/issues/7#issuecomment-370745440
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True, show_layer_names=True)


EPOCHS = 50

history = model.fit(train_ds,
                    steps_per_epoch=(6000 // BATCH_SIZE),
                    validation_data=validation_ds, validation_steps=1, 
                    epochs=EPOCHS)


plt.plot(history.history['classifier_head_accuracy'])
plt.plot(history.history['val_classifier_head_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# # adapted from: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
# def intersection_over_union(boxA, boxB):
# 	xA = max(boxA[0], boxB[0])
# 	yA = max(boxA[1], boxB[1])
# 	xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
# 	yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
# 	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
# 	boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
# 	boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
# 	iou = interArea / float(boxAArea + boxBArea - interArea)
# 	return iou
def intersection_over_union(boxA, boxB):
    # 提取座標
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # 計算交集區域
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # 計算各框面積
    boxAArea = boxA[2] * boxA[3]  # 預測框面積
    boxBArea = boxB[2] * boxB[3]  # 實際框面積

    # 確實無框
    if boxAArea == 0 and boxBArea == 0:
        return 0, False

    if boxAArea == 0 or boxBArea == 0:
        return 0.0, True

    # 計算 IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou, True



def tune_test_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1) 
    dataset = dataset.repeat()
    return dataset


plt.figure(figsize=(12, 10))

test_ds = tune_test_ds(raw_test_ds)
# test_ds = train_ds

test_list = list(test_ds.take(20).as_numpy_iterator())

print(len(test_list))

image, labels = test_list[0]

for i in range(len(test_list)):

    ax = plt.subplot(4, 5, i + 1)
    image, labels = test_list[i]

    predictions = model(image)

    predicted_box = predictions[1][0] * input_size
    predicted_box = tf.cast(predicted_box, tf.int32)

    predicted_label = predictions[0][0][1]

    image = image[0]

    actual_label = labels[0][0][1]
    actual_box = labels[1][0] * input_size
    actual_box = tf.cast(actual_box, tf.int32)

    image = image.astype("float") * 255.0
    image = image.astype(np.uint8)
    image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    color = (255, 0, 0)
    # print box red if predicted and actual label do not match
    if (predicted_label > 0.5 and actual_label > 0) or (predicted_label < 0.5 and actual_label == 0):
        color = (0, 255, 0)

    img_label = "negative"
    if predicted_label > 0.5:
        img_label = "positive"

    predicted_box_n = predicted_box.numpy()
    cv.rectangle(image_color, predicted_box_n, color, 2)
    cv.rectangle(image_color, actual_box.numpy(), (0, 0, 255), 2)
    cv.rectangle(image_color, (predicted_box_n[0], predicted_box_n[1] + predicted_box_n[3] - 20), (predicted_box_n[0] + predicted_box_n[2], predicted_box_n[1] + predicted_box_n[3]), color, -1)
    cv.putText(image_color, img_label, (predicted_box_n[0] + 5, predicted_box_n[1] + predicted_box_n[3] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))

    IoU = intersection_over_union(predicted_box.numpy(), actual_box.numpy())[0]

    plt.title("IoU:" + format(IoU, '.4f'))
    plt.imshow(image_color)
    plt.axis("off")





# 建立儲存比較圖的資料夾
output_dir = "output_predictions"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 10))


# 將 test_ds 資料轉換為可迭代的列表
test_list = list(test_ds.take(500).as_numpy_iterator())
print(f"Test Data Size: {len(test_list)}")

# 初始化計算變數
correct_count = 0
total_count = 0
iou_list = []

# 開始處理每張圖片
for i in range(len(test_list)):

    # ax = plt.subplot(4, 5, i + 1)

    # 取得影像與標籤
    image, labels = test_list[i]
    predictions = model(image)

    # 預測標籤與框
    predicted_box = predictions[1][0] * input_size
    predicted_box = tf.cast(predicted_box, tf.int32)
    predicted_label = predictions[0][0][1]

    # 取得實際標籤與框
    image = image[0]
    actual_label = labels[0][0][1]
    actual_box = labels[1][0] * input_size
    actual_box = tf.cast(actual_box, tf.int32)

    # 預處理影像
    image = image.astype("float") * 255.0
    image = image.astype(np.uint8)
    image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    # 比較預測標籤與實際標籤
    color = (255, 0, 0)  # 預設紅色
    if (predicted_label > 0.5 and actual_label > 0) or (predicted_label < 0.5 and actual_label == 0):
        color = (0, 255, 0)  # 預測正確顯示綠色
        correct_count += 1

    total_count += 1

    # 繪製預測標籤
    img_label = "negative"
    if predicted_label > 0.5:
        img_label = "postive"

    # 繪製預測框
    predicted_box_n = predicted_box.numpy()
    cv.rectangle(image_color, predicted_box_n, color, 2)
    cv.rectangle(image_color, actual_box.numpy(), (0, 0, 255), 2)  # 實際標籤框紅色
    cv.rectangle(image_color, (predicted_box_n[0], predicted_box_n[1] + predicted_box_n[3] - 20), 
                 (predicted_box_n[0] + predicted_box_n[2], predicted_box_n[1] + predicted_box_n[3]), color, -1)
    cv.putText(image_color, img_label, (predicted_box_n[0] + 5, predicted_box_n[1] + predicted_box_n[3] - 5), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))

    # 計算 IoU
    IoU = intersection_over_union(predicted_box.numpy(), actual_box.numpy())
    if(IoU[1]):
        iou_list.append(IoU[0])

    # 顯示圖片與 IoU 值
    # plt.title(f"IoU: {IoU:.4f}")
    # plt.imshow(image_color)
    # plt.axis("off")

    # 儲存圖片到資料夾
    output_path = os.path.join(output_dir, f"prediction_{i + 1}.png")
    cv.imwrite(output_path, cv.cvtColor(image_color, cv.COLOR_RGB2BGR))  # OpenCV 儲存格式為 BGR

# 計算準確率與 IoU 平均值
accuracy = correct_count / total_count
average_iou = np.mean(iou_list)

print(f"準確率 (Accuracy): {accuracy:.4f}")
print(f"平均 IoU (Mean IoU): {average_iou:.4f}")

# 儲存圖表
plt.savefig(os.path.join(output_dir, "all_predictions.png"))
plt.show()



import shutil

# 壓縮 output_predictions 資料夾為 predictions.zip
shutil.make_archive('predictions', 'zip', output_dir)





