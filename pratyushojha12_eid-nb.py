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


# /kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/dummy_sub.csv
# /kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv


# train_df = pd.read_csv("/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv")
# test_df = pd.read_csv("/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/dummy_sub.csv")
# train_df



# test_df



# import pandas as pd
# import os

# # Path setup
# data_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/"
# train_dir = os.path.join(data_dir, "train")
# test_dir = os.path.join(data_dir, "test")
# labels_path = os.path.join(data_dir, "train_labels.csv")

# # Load CSV
# df = pd.read_csv(labels_path)
# print(df.head())



# print("Class distribution:\n", df['label'].value_counts())



# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# df['encoded_label'] = le.fit_transform(df['label'])

# # For decoding later
# #label_to_class = dict(zip(le.transform(le.classes_), le.classes_))


# import cv2
# import numpy as np

# IMG_SIZE = 128  # you can use 224 or 256 for large models

# def load_images(image_dir, df, size=(IMG_SIZE, IMG_SIZE)):
#     images = []
#     labels = []
    
#     for _, row in df.iterrows():
#         img_path = os.path.join(image_dir, row['filename'])
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, size)
#         img = img / 255.0  # Normalize
#         images.append(img)
#         labels.append(row['encoded_label'])
        
#     return np.array(images), np.array(labels)

# X, y = load_images(train_dir, df)
# print("Shape of X:", X.shape)
# print("Shape of y:", y.shape)



# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )



# # import tensorflow as tf

# # model = tf.keras.Sequential([
# #     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
# #     tf.keras.layers.MaxPooling2D(2,2),
# #     tf.keras.layers.Flatten(),
# #     tf.keras.layers.Dense(128, activation='relu'),
# #     tf.keras.layers.Dense(len(le.classes_), activation='softmax')
# # ])

# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
# import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
# from tensorflow.keras.callbacks import EarlyStopping

# # ðŸŸ© STEP 0: Preprocess with EfficientNet-specific preprocessing
# X_train = preprocess_input(X_train)
# X_val = preprocess_input(X_val)

# # ðŸŸ© STEP 1: Load base EfficientNetB0
# base_model = EfficientNetB0(
#     include_top=False,
#     weights='imagenet',
#     input_shape=(IMG_SIZE, IMG_SIZE, 3)
# )
# base_model.trainable = False

# # ðŸŸ© STEP 2: Add classification head
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.4)(x)
# x = Dense(256, activation='relu')(x)
# output = Dense(len(le.classes_), activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)

# # ðŸŸ© STEP 3: Compile & train head
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(X_train, y_train,
#           validation_data=(X_val, y_val),
#           epochs=10,
#           batch_size=32,
#           callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])



# # Unfreeze top 20 layers only
# for layer in base_model.layers[-20:]:
#     layer.trainable = True

# # Compile with low learning rate
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(X_train, y_train,
#           validation_data=(X_val, y_val),
#           epochs=10,
#           batch_size=32,
#           callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])



# import os
# import cv2
# import numpy as np

# test_dir = os.path.join(data_dir, 'test')
# test_filenames = os.listdir(test_dir)

# def preprocess_test_images(file_list, directory, size=(128, 128)):
#     X_test = []
#     for fname in file_list:
#         path = os.path.join(directory, fname)
#         img = cv2.imread(path)
#         img = cv2.resize(img, size)
#         img = img / 255.0
#         X_test.append(img)
#     return np.array(X_test)

# X_test = preprocess_test_images(test_filenames, test_dir)



# pred_probs = model.predict(X_test)
# pred_indices = pred_probs.argmax(axis=1)
# pred_labels = le.inverse_transform(pred_indices)



# submission = pd.DataFrame({
#     'filename': test_filenames,
#     'label': pred_labels
# })

# # Make sure filenames match the order of dummy_sub.csv if required
# submission = submission.sort_values(by='filename')

# # Save
# submission.to_csv('submission.csv', index=False)
# print(submission.head())



import pandas as pd
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path setup
data_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
labels_path = os.path.join(data_dir, "train_labels.csv")

# Load CSV
df = pd.read_csv(labels_path)
print("Class distribution:\n", df['label'].value_counts())

# Encode labels
le = LabelEncoder()
df['encoded_label'] = le.fit_transform(df['label'])

# Image preprocessing
IMG_SIZE = 224  # Increase to match EfficientNetB0

def load_images(image_dir, df, size=(IMG_SIZE, IMG_SIZE)):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        images.append(img)  # Load raw images (0-255)
        labels.append(row['encoded_label'])
    return np.array(images), np.array(labels)

X, y = load_images(train_dir, df)
X = preprocess_input(X)  # Apply EfficientNet preprocessing
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Load base model
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Increase dropout
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
output = Dense(len(le.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
)

# Callbacks
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train head
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=30,  # Increase epochs
    callbacks=[early_stopping, lr_scheduler]
)

# Fine-tuning
for layer in base_model.layers[-50:]:  # Unfreeze more layers
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # Slightly higher learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
)

model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=[early_stopping, lr_scheduler]
)

# Preprocess test images
def preprocess_test_images(file_list, directory, size=(IMG_SIZE, IMG_SIZE)):
    X_test = []
    for fname in file_list:
        path = os.path.join(directory, fname)
        img = cv2.imread(path)
        img = cv2.resize(img, size)
        X_test.append(img)
    X_test = np.array(X_test)
    return preprocess_input(X_test)  # Apply EfficientNet preprocessing

test_filenames = os.listdir(test_dir)
X_test = preprocess_test_images(test_filenames, test_dir)

# Predict
pred_probs = model.predict(X_test)
pred_indices = pred_probs.argmax(axis=1)
pred_labels = le.inverse_transform(pred_indices)

# Submission
submission = pd.DataFrame({
    'filename': test_filenames,
    'label': pred_labels
})
submission = submission.sort_values(by='filename')
submission.to_csv('submission.csv', index=False)
print(submission.head())

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()




