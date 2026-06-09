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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report


# Đường dẫn dữ liệu
train_dir = '/kaggle/input/ai-training-challenge-hutech-orange-classifier/old_oranges_data_1/old_oranges_data/train_set'
test_dir = '/kaggle/input/ai-training-challenge-hutech-orange-classifier/old_oranges_data_1/old_oranges_data/test_set'

# 1. Chuẩn bị và tiền xử lý dữ liệu
data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_data = data_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_data = data_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_data_gen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)
# 2. Xây dựng mô hình
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Phân loại nhị phân
])

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Huấn luyện mô hình
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# 4. Đánh giá và báo cáo kết quả
predictions = (model.predict(test_data) > 0.5).astype('int32')
true_labels = test_data.classes
class_indices = list(test_data.class_indices.keys())

test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

print(classification_report(true_labels, predictions, target_names=class_indices))



# **Vẽ lịch sử huấn luyện**
def plot_training_history(history):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(14, 6))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], 'orange', label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'orange', label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

