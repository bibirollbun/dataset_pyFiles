# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#!pip install keras-resnet --quiet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
#from keras_resnet.models import ResNet18
from tensorflow.keras.applications.resnet import preprocess_input


file_path_train = "/kaggle/input/mnist-dataset-number-classification/train_mnist.csv"
train_csv = pd.read_csv(file_path_train)
file_path_test = "/kaggle/input/mnist-dataset-number-classification/test_mnist.csv"
test_csv = pd.read_csv(file_path_test)
file_path_submit = "/kaggle/input/mnist-dataset-number-classification/sample_submission.csv"
submit_csv = pd.read_csv(file_path_submit)
# Thông tin tổng quan về dữ liệu
print(train_csv.info())
print(test_csv.info())

# Kích thước dữ liệu (số hàng, số cột)
print("Shape train csv: ", train_csv.shape)
print("Shape test csv: ", test_csv.shape)


# In số lượng cột train 
#print("Số lượng cột:", len(train_csv.columns))

# Nếu muốn xem tên các cột train
#print("Tên các cột:", train_csv.columns.tolist())


# Lấy ngẫu nhiên 9 dòng
random_rows = train_csv.sample(n=9)

# Tạo figure 3x3
plt.figure(figsize=(8, 8))

for i, (_, row) in enumerate(random_rows.iterrows()):
    label = row.iloc[1]                               # nhãn ở cột 1
    pixels = row.iloc[2:786].astype(float).values     # các pixel từ cột 2 -> 786
    image = pixels.reshape(28, 28)                    # reshape thành 28x28

    plt.subplot(3, 3, i + 1)                          # lưới 3x3
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# Đếm số lượng mỗi nhãn
label_counts = train_csv['label'].value_counts().sort_index()  # Đổi 'label' thành tên cột nhãn thực tế

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.bar(label_counts.index.astype(str), label_counts.values)
plt.xlabel("Nhãn (Label)")
plt.ylabel("Số lượng")
plt.title("Phân bố số lượng nhãn")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Tách nhãn và pixel
X = train_csv.iloc[:, 2:786].values.astype("float32") / 255.0  # chuẩn hóa 0-1
y = train_csv.iloc[:, 1].values

# Reshape thành (num_samples, 28, 28, 1) cho CNN
X = X.reshape(-1, 28, 28, 1)

# Resize lên 32x32 và 3 Channel
#X = tf.image.resize(X, [32,32])
#X = tf.image.grayscale_to_rgb(X)

# Preprocess cho ResNet
#X = preprocess_input(X)

# One-hot encode nhãn
#y = to_categorical(y, num_classes=10)

# Chia train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)


#Model CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss = 'sparse_categorical_crossentropy',
    #loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    verbose=1
)



loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {acc*100:.2f}%")



import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('CNN Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



from sklearn.metrics import confusion_matrix
import seaborn as sns

# Dự đoán nhãn
# Dự đoán nhãn trên tập validation
y_pred_probs = model.predict(X_val)           # Output là xác suất (shape: [num_samples, 10])
y_pred = np.argmax(y_pred_probs, axis=1)      # Chọn nhãn có xác suất cao nhất

# Tạo confusion matrix
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


from sklearn.metrics import classification_report

# In báo cáo chi tiết
report = classification_report(y_val, y_pred, digits=4)
print(report)



# Chuyển sang numpy và chuẩn hóa
X_test = test_csv.iloc[:,1:785].values.astype("float32") / 255.0

# Reshape cho CNN
X_test = X_test.reshape(-1, 28, 28, 1)

# Resize lên 32x32 và 3 chanel
#X_test = tf.image.resize(X_test, [32,32])
#X_test = tf.image.grayscale_to_rgb(X_test)

# Preprocess cho ResNet
#X_test = preprocess_input(X_test)

print("Test shape:", X_test.shape)


# Dự đoán xác suất cho 10 lớp
y_pred_prob = model.predict(X_test)

# Lấy nhãn dự đoán (0-9)
y_pred_labels = np.argmax(y_pred_prob, axis=1)

print("Sample predictions:", y_pred_labels[:10])



predictions = y_pred_labels  

def generate_submission(predictions, output_file="submission.csv"):
    if len(predictions) != len(submit_csv):
        raise ValueError("Prediction length must match sample_submission")
    submit_csv["label"] = predictions
    submit_csv.to_csv(output_file, index=False)

generate_submission(predictions)

