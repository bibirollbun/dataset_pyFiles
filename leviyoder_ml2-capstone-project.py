# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, dirnames, _ in os.walk('/kaggle/input'):
    print(dirname)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from functools import partial
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras import layers
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.model_selection import train_test_split

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

tf.random.set_seed(72)


train_path = '/kaggle/input/aptos2019-blindness-detection/train_images'
test_path = '/kaggle/input/aptos2019-blindness-detection/test_images'


print("Number of training images:", len(os.listdir(train_path)))
print("Number of test images:", len(os.listdir(test_path)))


train = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
test = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")


class_counts = train['diagnosis'].value_counts().sort_index()

print(class_counts)


train['file_path'] = train['id_code'].apply(lambda x: 
        f'/kaggle/input/aptos2019-blindness-detection/train_images/{x}.png')


print(train.info())


train_df, val_df = train_test_split(
    train, 
    test_size=0.15, 
    stratify=train['diagnosis'],
    random_state=42
)


print("Entries in train set", len(train_df))
print("Entries in validation set", len(val_df))


class_names = [str(c) for c in sorted(train_df['diagnosis'].unique())]


IMG_SIZE = 224

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(df, batch_size=32, shuffle=False):
    paths = df['file_path'].values
    labels = df['diagnosis'].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(1024)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, shuffle=True)
val_ds = make_dataset(val_df)


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',      
    patience=5,              
    restore_best_weights=True  
)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomContrast(0.2),
])

def load_and_augment(path, label):
    img, label = load_image(path, label)
    img = data_augmentation(img, training=True)
    return img, label

def make_augmented_dataset(df, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((df['file_path'].values, df['diagnosis'].values))
    ds = ds.map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

augmented_ds = make_augmented_dataset(train_df)


DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")

model_1 = tf.keras.Sequential([
    Input(shape=(224, 224, 3)),
    DefaultConv2D(filters=64, kernel_size=7,),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=512),
    DefaultConv2D(filters=512),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=5, activation="softmax")
])



model_1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_1.summary()


history_1 = model_1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stop]
)


h1 = history_1.history
epochs_1 = range(1, len(h1['loss']) + 1)
plt.plot(epochs_1, h1['loss'], 'b-', label='Model 1 Training Loss')
plt.plot(epochs_1, h1['val_loss'], 'b--', label='Model 1 Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



plt.plot(epochs_1, h1['accuracy'], 'b-', label='Model 1 Training Accuracy')
plt.plot(epochs_1, h1['val_accuracy'], 'b--', label='Model 1 Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


y_true = np.concatenate([y for _, y in val_ds], axis=0)

y_pred_probs = model_1.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


model_2 = clone_model(model_1);
model_2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_2.summary()


history_2 = model_2.fit(
    augmented_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stop]
)


h2 = history_2.history
epochs_2 = range(1, len(h2['loss']) + 1)
plt.plot(epochs_2, h2['loss'], 'b-', label='Model 2 Training Loss')
plt.plot(epochs_2, h2['val_loss'], 'b--', label='Model 2 Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.plot(epochs_2, h2['accuracy'], 'b-', label='Model 2 Training Accuracy')
plt.plot(epochs_2, h2['val_accuracy'], 'b--', label='Model 2 Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


y_true_2 = np.concatenate([y for _, y in val_ds], axis=0)

y_pred_probs_2 = model_2.predict(val_ds)
y_pred_2 = np.argmax(y_pred_probs, axis=1)

cm_2 = confusion_matrix(y_true_2, y_pred_2)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_2,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
    weights='imagenet',        
    include_top=False,         
    input_shape=(224, 224, 3) 
)

base_model.trainable = False


model_3 = Sequential([
    base_model,
    GlobalAveragePooling2D(),  
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),
    Dense(5, activation='softmax')  
])


model_3.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_3.summary()


history_3 = model_3.fit(
    augmented_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stop]
)


h3 = history_3.history
epochs_3 = range(1, len(h3['loss']) + 1)
plt.plot(epochs_3, h3['loss'], 'b-', label='Model 3 Training Loss')
plt.plot(epochs_3, h3['val_loss'], 'b--', label='Model 3 Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.plot(epochs_3, h3['accuracy'], 'b-', label='Model 3 Training Accuracy')
plt.plot(epochs_3, h3['val_accuracy'], 'b--', label='Model 3 Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


y_true_3 = np.concatenate([y for _, y in val_ds], axis=0)

y_pred_probs_3 = model_3.predict(val_ds)
y_pred_3 = np.argmax(y_pred_probs_3, axis=1)

cm_3 = confusion_matrix(y_true_3, y_pred_3)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_3,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


test.head()
test['file_path'] = test['id_code'].apply(
    lambda x: f"/kaggle/input/aptos2019-blindness-detection/test_images/{x}.png"
)


def load_test_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image


test_paths = test['file_path'].values

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)


y_pred_probs_submission = model_3.predict(test_ds)
y_pred_submission = np.argmax(y_pred_probs_submission, axis=1)

submission = pd.DataFrame({
    "id_code": test["id_code"],
    "diagnosis": y_pred_submission
})
submission.to_csv("submission.csv", index=False)


from tensorflow.keras.preprocessing.image import save_img

def augment_and_save(image_path, save_dir, prefix="aug"):
   
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    
    
    augmented_img = data_augmentation(img, training=True)
    
   
    augmented_img_uint8 = tf.image.convert_image_dtype(augmented_img, dtype=tf.uint8)
    
    
    os.makedirs(save_dir, exist_ok=True)
    
    
    filename = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"{prefix}_{filename}")
    
    
    save_img(save_path, augmented_img_uint8.numpy())
    
    return save_path


image_paths = [
    "/kaggle/input/aptos2019-blindness-detection/train_images/000c1434d8d7.png",
    "/kaggle/input/aptos2019-blindness-detection/train_images/001639a390f0.png",
    "/kaggle/input/aptos2019-blindness-detection/train_images/0024cdab0c1e.png"
]

save_directory = "/kaggle/working/augmented_images/"

for path in image_paths:
    new_path = augment_and_save(path, save_directory)
    print("Saved:", new_path)

