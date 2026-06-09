import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("✅ GPU is enabled!")
    except RuntimeError as e:
        print("❌ GPU error:", e)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed Precision Enabled!")

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)

seed_everything()

df = pd.read_csv('../input/happy-whale-and-dolphin/train.csv')


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, validation_split=0.2, horizontal_flip=True, zoom_range=0.2)

train_generator = train_datagen.flow_from_dataframe(
    df, '../input/happy-whale-and-dolphin/train_images/', x_col='image',
    y_col='individual_id', target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')

val_generator = train_datagen.flow_from_dataframe(
    df, '../input/happy-whale-and-dolphin/train_images/', x_col='image',
    y_col='individual_id', target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')


base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(len(train_generator.class_indices), activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=val_generator, epochs=4)

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


