import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from zipfile import ZipFile
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import squarify


path = "/kaggle/input/aerial-cactus-identification/"
train_labels = pd.read_csv(path + 'train.csv')


class_names = ["Has cactus", "Hasn\'t cactus"]
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

class_names_label


train_labels.info()


train_labels.head()


train_labels.id.shape


train_labels.size


with ZipFile(path + "train.zip") as zipper:
    zipper.extractall()

with ZipFile(path + "test.zip") as zipper:
    zipper.extractall()


train_path = '/kaggle/working/train'
test_path = '/kaggle/working/test'


def load_data(train_labels, train_path):
    x_train = []
    y_train = []

    for idx in range(len(train_labels)):
        img_path = os.path.join(train_path, train_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = train_labels.iloc[idx, 1]

        x_train.append(image)
        y_train.append(label)

    return x_train, y_train


x_train, y_train = load_data(train_labels, train_path)


def display_examples(class_names, images, labels):
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("plots of a sample of the data", fontsize=10)
    for i in range(20):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


display_examples(class_names, x_train, y_train)


unique_labels, train_counts = np.unique(y_train, return_counts=True)
print(f"{unique_labels[0]}: {train_counts[0]}\n{unique_labels[1]}: {train_counts[1]}")


plt.figure(figsize=(4, 4))
plt.bar(unique_labels, train_counts, color='violet', edgecolor='black')
plt.xlabel('Class Labels')
plt.ylabel('Number of Samples')
plt.title('Training Set Class Distribution')
plt.xticks(unique_labels)
plt.grid(axis='y', linestyle='', alpha=0.4)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 6))
squarify.plot(sizes=train_counts, label=class_names, alpha=0.8, color=plt.cm.Set3.colors)
plt.title('Training Set Class Distribution')
plt.show()


x_train=np.array(x_train, dtype='float32')
y_train=np.array(y_train, dtype='int32')


print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train, random_state=42)


model=Sequential([

    Conv2D(32,3,activation='relu', input_shape=(32, 32, 3), padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64,3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(128,3, activation='relu'),
    BatchNormalization(),

    Conv2D(256,3, activation='relu'),
    BatchNormalization(),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(16, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
    
]
    
)


early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)


model.summary()


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
class_weights


history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights
)


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid()
plt.show()


preds = model.predict(x_val)
preds_labels = (preds > 0.5).astype(int).flatten()
print(classification_report(y_val, preds_labels, digits=4))


conf_matrix = confusion_matrix(y_val, preds_labels)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Reds')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



test_images = glob.glob(test_path+'/*.jpg')


x_test = []

for path in test_images:
    img = load_img(path)  
    img_array = img_to_array(img)
    x_test.append(img_array)

x_test = np.array(x_test)


pred_probs = model.predict(x_test)
y_pred_labels = (pred_probs > 0.5).astype(int).reshape(-1)


image_names = [os.path.basename(p) for p in test_images]

df_submission = pd.DataFrame({
    'id': image_names,
    'has_cactus': y_pred_labels.astype(int)
})

df_submission.to_csv('submission.csv', index=False)

