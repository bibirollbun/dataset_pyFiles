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
import zipfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# TensorFlow loglarÄ±nÄ± sessize al
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



train_zip = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/train")

# KlasÃ¶rler
base_dir = "/kaggle/working/cats_vs_dogs"
cat_dir = os.path.join(base_dir, "cat")
dog_dir = os.path.join(base_dir, "dog")
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

# Resimleri taÅŸÄ±
src_dir = "/kaggle/working/train/train"
for fname in os.listdir(src_dir):
    if fname.startswith("cat"):
        shutil.copy(os.path.join(src_dir, fname), cat_dir)
    elif fname.startswith("dog"):
        shutil.copy(os.path.join(src_dir, fname), dog_dir)

print("Toplam cat resmi:", len(os.listdir(cat_dir)))
print("Toplam dog resmi:", len(os.listdir(dog_dir)))



img_size = (64,64)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)



inputs = Input(shape=(64,64,3))

x = Conv2D(32, (3,3), activation='relu', padding="same", name="conv1")(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)

x = Conv2D(64, (3,3), activation='relu', padding="same", name="conv2")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)

x = Conv2D(128, (3,3), activation='relu', padding="same", name="conv3")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)

x = Conv2D(256, (3,3), activation='relu', padding="same", name="conv4")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2,2)(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()



from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[early_stop, reduce_lr]
)



plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()



val_generator.reset()
Y_pred = model.predict(val_generator, verbose=1)
y_pred = np.round(Y_pred).astype(int)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Tahmin')
plt.ylabel('GerÃ§ek')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=['Cat','Dog']))



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_array, heatmap, pred_label, true_label, confidence, alpha=0.4):
    import cv2
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.image.resize(jet_heatmap, (img_array.shape[1], img_array.shape[2]))
    superimposed_img = jet_heatmap * alpha + img_array[0]
    superimposed_img = np.clip(superimposed_img, 0, 1)
    plt.figure(figsize=(4,4))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title(f"GerÃ§ek: {true_label} | Tahmin: {pred_label} ({confidence:.2f})")
    plt.show()



class_names = ['Cat','Dog']

val_images, val_labels = next(iter(val_generator))
_ = model.predict(val_images[:1])  # tensorlarÄ± baÅŸlat

last_conv_layer_name = "conv3"

for i in range(5):
    img_array = np.expand_dims(val_images[i], axis=0)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    pred_prob = model.predict(img_array, verbose=0)[0][0]
    pred_class = 1 if pred_prob > 0.5 else 0
    pred_label = class_names[pred_class]
    true_label = class_names[int(val_labels[i])]
    display_gradcam(img_array, heatmap, pred_label, true_label, pred_prob)


