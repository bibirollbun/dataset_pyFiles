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


import numpy as np
import pandas as pd
import os
import cv2
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping



with zipfile.ZipFile("/kaggle/input/street-view-getting-started-with-julia/trainResized.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working")

df = pd.read_csv("/kaggle/input/street-view-getting-started-with-julia/trainLabels.csv")
df = df[df["Class"].apply(lambda x: str(x).isdigit())]
df["Class"] = df["Class"].astype(int)

# Visualize class distribution
plt.figure(figsize=(10, 4))
sns.countplot(x=df["Class"])
plt.title("Class Distribution")
plt.show()


import os

train_resized_files = os.listdir("/kaggle/working/trainResized")
print("Total files:", len(train_resized_files))
print("First 5 files:", train_resized_files[:5])


X, y = [], []

for _, row in df.iterrows():
    path = f"/kaggle/working/trainResized/{row['ID']}.Bmp"
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        X.append(img)
        y.append(row["Class"])

X = np.array(X)
y = np.array(y)

# Show 10 random samples before any preprocessing
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X[i])
    plt.title(y[i])
    plt.axis("off")
plt.suptitle("Sample Images (Raw 64x64)")
plt.show()



X_m = mobilenet_preprocess(X.copy())
X_r = resnet_preprocess(X.copy())
y_cat = to_categorical(y, num_classes=10)

X_train_m, X_val_m, y_train, y_val = train_test_split(X_m, y_cat, test_size=0.2, random_state=42)
X_train_r, X_val_r, _, _ = train_test_split(X_r, y_cat, test_size=0.2, random_state=42)

# Show few MobileNet-preprocessed images (scaled, might look odd but valid)
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(((X_train_m[i] + 1) * 127.5).astype(np.uint8))
    plt.axis("off")
plt.suptitle("MobileNetV2 Preprocessed")
plt.show()



datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train_m)

# Show augmented versions of one image
sample = X_train_m[0].reshape(1, 64, 64, 3)
plt.figure(figsize=(10, 2))
for i, batch in enumerate(datagen.flow(sample, batch_size=1)):
    plt.subplot(1, 10, i+1)
    plt.imshow(((batch[0] + 1) * 127.5).astype(np.uint8))
    plt.axis("off")
    if i == 9:
        break
plt.suptitle("Augmented Versions of One Image")
plt.show()



mobilenet = MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights=None)
x = GlobalAveragePooling2D()(mobilenet.output)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

model_m = Model(inputs=mobilenet.input, outputs=preds)
model_m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history_m = model_m.fit(datagen.flow(X_train_m, y_train, batch_size=64),
                        validation_data=(X_val_m, y_val),
                        epochs=30,
                        callbacks=callbacks)



plt.plot(history_m.history['accuracy'], label='train acc')
plt.plot(history_m.history['val_accuracy'], label='val acc')
plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()



resnet = ResNet50(input_shape=(64, 64, 3), include_top=False, weights=None)
x = GlobalAveragePooling2D()(resnet.output)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)

model_r = Model(inputs=resnet.input, outputs=preds)
model_r.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_r = model_r.fit(datagen.flow(X_train_r, y_train, batch_size=64),
                        validation_data=(X_val_r, y_val),
                        epochs=30,
                        callbacks=callbacks)



plt.plot(history_r.history['accuracy'], label='train acc')
plt.plot(history_r.history['val_accuracy'], label='val acc')
plt.title("ResNet50 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()



with zipfile.ZipFile("/kaggle/input/street-view-getting-started-with-julia/testResized.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working")

X_test, image_ids = [], []

for fname in sorted(os.listdir("/kaggle/working/testResized")):
    path = f"/kaggle/working/testResized/{fname}"
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        X_test.append(img)
        image_ids.append(fname.replace(".Bmp", ""))

X_test = np.array(X_test)
X_test_m = mobilenet_preprocess(X_test.copy())
X_test_r = resnet_preprocess(X_test.copy())

# Show sample test images
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i])
    plt.axis("off")
plt.suptitle("Sample Test Images")
plt.show()



preds_m = model_m.predict(X_test_m, verbose=1)
preds_r = model_r.predict(X_test_r, verbose=1)

ensemble_preds = (preds_m + preds_r) / 2
final_preds = np.argmax(ensemble_preds, axis=1)

# Show some prediction examples
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i])
    plt.title(final_preds[i])
    plt.axis("off")
plt.suptitle("Predictions on Test Images")
plt.show()



submission = pd.DataFrame({
    'ID': image_ids,
    'Class': final_preds
})
submission.to_csv("submission.csv", index=False)
submission.head()




import tensorflow as tf
import matplotlib.cm as cm

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



def overlay_heatmap(img, heatmap, alpha=0.4, cmap='jet'):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cm.get_cmap(cmap)(heatmap)
    heatmap_colored = np.uint8(255 * heatmap_colored[..., :3])
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay



# Pick 5 test images
indices = [0, 5, 10, 15, 20]
selected_images = X_test[indices]
selected_images_m = X_test_m[indices]
selected_images_r = X_test_r[indices]

for i, (img_raw, img_m, img_r) in enumerate(zip(selected_images, selected_images_m, selected_images_r)):
    input_m = np.expand_dims(img_m, axis=0)
    input_r = np.expand_dims(img_r, axis=0)

    heatmap_m = make_gradcam_heatmap(input_m, model_m, last_conv_layer_name="Conv_1")
    heatmap_r = make_gradcam_heatmap(input_r, model_r, last_conv_layer_name="conv5_block3_out")

    overlay_m = overlay_heatmap(img_raw, heatmap_m)
    overlay_r = overlay_heatmap(img_raw, heatmap_r)

    # Show both side by side
    plt.figure(figsize=(8, 2))
    plt.subplot(1, 3, 1)
    plt.imshow(img_raw)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(overlay_m)
    plt.title("MobileNetV2")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay_r)
    plt.title("ResNet50")
    plt.axis("off")

    plt.suptitle(f"Grad-CAM Visualization for Test Image {indices[i]}")
    plt.tight_layout()
    plt.show()


