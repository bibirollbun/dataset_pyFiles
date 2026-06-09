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


import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, ReLU, GlobalAveragePooling2D, DepthwiseConv2D, Concatenate, Lambda
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from pathlib import Path


base_path = Path("/kaggle/input/histopathologic-cancer-detection")
train_dir = base_path / "train"
labels_csv = base_path / "train_labels.csv"



labels = pd.read_csv(labels_csv)
labels["id"] = labels["id"] + ".tif"



labels = pd.concat([
    labels[labels.label == 0].sample(5000, random_state=42),
    labels[labels.label == 1].sample(5000, random_state=42)
]).reset_index(drop=True)



def load_images(ids, directory, img_size=128):
    images = []
    for image_id in ids:
        path = directory / image_id
        with Image.open(path) as img:
            img = img.resize((img_size, img_size))
            img = img_to_array(img)
            images.append(img)
    return np.array(images)

X = load_images(labels["id"], train_dir)
y = labels["label"].values



# --------------------- ğŸ–¼ï¸� Better Visualization of Sample Images ---------------------
import matplotlib.pyplot as plt

# Randomly sample 6 images from each class
cancer_indices = np.where(y == 1)[0]
non_cancer_indices = np.where(y == 0)[0]

sample_indices = np.concatenate([
    np.random.choice(non_cancer_indices, 6, replace=False),
    np.random.choice(cancer_indices, 6, replace=False)
])

# Show 12 images (6 Non-Cancer + 6 Cancer)
plt.figure(figsize=(14, 7))
for i, idx in enumerate(sample_indices):
    plt.subplot(3, 4, i + 1)
    img = X[idx]
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)  # force full contrast
    label = "Cancer" if y[idx] == 1 else "Non-Cancer"
    plt.title(label, color="red" if label == "Cancer" else "green")
    plt.axis("off")

plt.suptitle("Sample Images - Cancer vs Non-Cancer")
plt.tight_layout()
plt.show()



# Normalize
X = X / 255.0


# --------------------- Train/Test Split ---------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# TensorFlow Datasets
def preprocess_tf(image, label):
    return tf.convert_to_tensor(image, dtype=tf.float32), tf.convert_to_tensor(label)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(preprocess_tf).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(preprocess_tf).batch(32)



# --------------------- Channel Shuffle ---------------------
class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self.groups = groups

    def call(self, inputs):
        batch_size, height, width, channels = tf.unstack(tf.shape(inputs))
        channels_per_group = channels // self.groups
        x = tf.reshape(inputs, [batch_size, height, width, self.groups, channels_per_group])
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = tf.reshape(x, [batch_size, height, width, channels])
        return x



# --------------------- Shuffle Block ---------------------
def shuffle_block(x, out_channels, stride):
    mid_channels = out_channels // 2
    if stride == 2:
        left = DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False)(x)
        left = BatchNormalization()(left)
        left = Conv2D(mid_channels, kernel_size=1, padding="same", use_bias=False)(left)
        left = BatchNormalization()(left)
        left = ReLU()(left)

        right = Conv2D(mid_channels, kernel_size=1, padding="same", use_bias=False)(x)
        right = BatchNormalization()(right)
        right = ReLU()(right)
        right = DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False)(right)
        right = BatchNormalization()(right)
        right = Conv2D(mid_channels, kernel_size=1, padding="same", use_bias=False)(right)
        right = BatchNormalization()(right)
        right = ReLU()(right)

        x = Concatenate()([left, right])
    else:
        left, right = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(x)
        right = Conv2D(mid_channels, kernel_size=1, padding="same", use_bias=False)(right)
        right = BatchNormalization()(right)
        right = ReLU()(right)
        right = DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=False)(right)
        right = BatchNormalization()(right)
        right = Conv2D(mid_channels, kernel_size=1, padding="same", use_bias=False)(right)
        right = BatchNormalization()(right)
        right = ReLU()(right)
        x = Concatenate()([left, right])

    x = ChannelShuffle(groups=2)(x)
    return x



# --------------------- Build ShuffleNet V2 ---------------------
def build_shufflenet_v2(input_shape=(128, 128, 3), num_classes=2):
    input_layer = Input(shape=input_shape)
    x = Conv2D(24, kernel_size=3, strides=1, padding="same", use_bias=False)(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for out_channels, num_blocks, stride in [(48, 2, 2), (96, 4, 2), (192, 8, 2)]:
        x = shuffle_block(x, out_channels, stride)
        for _ in range(num_blocks - 1):
            x = shuffle_block(x, out_channels, 1)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=input_layer, outputs=x)



# --------------------- Compile & Train ---------------------
model = build_shufflenet_v2()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)



# --------------------- Plot Accuracy ---------------------
import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Training Progress")
plt.show()


import tensorflow.keras.backend as K
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    
    pooled_grads = tf.reduce_mean(tape.gradient(loss, conv_outputs), axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()



idx = 10  # change index to test different images
sample_image = X_val[idx]
sample_tensor = tf.expand_dims(sample_image, axis=0)



# Automatically find last Conv2D layer
last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]



heatmap = make_gradcam_heatmap(sample_tensor, model, last_conv_layer_name)



import matplotlib.pyplot as plt
import cv2
import numpy as np

def display_gradcam(original_image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    overlay = heatmap_colored * alpha + np.uint8(original_image * 255)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Grad-CAM Overlay")
    plt.show()



display_gradcam(sample_image, heatmap)



prediction = model.predict(sample_tensor)[0]
predicted_class = np.argmax(prediction)
print(f"Predicted Class: {predicted_class} (Confidence: {prediction[predicted_class]:.4f})")



# --------------------- ğŸ–¼ï¸� Better Visualization of Sample Images ---------------------
import matplotlib.pyplot as plt

# Randomly sample 6 images from each class
cancer_indices = np.where(y == 1)[0]
non_cancer_indices = np.where(y == 0)[0]

sample_indices = np.concatenate([
    np.random.choice(non_cancer_indices, 6, replace=False),
    np.random.choice(cancer_indices, 6, replace=False)
])

# Show 12 images (6 Non-Cancer + 6 Cancer)
plt.figure(figsize=(14, 7))
for i, idx in enumerate(sample_indices):
    plt.subplot(3, 4, i + 1)
    img = X[idx]
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)  # force full contrast
    label = "Cancer" if y[idx] == 1 else "Non-Cancer"
    plt.title(label, color="red" if label == "Cancer" else "green")
    plt.axis("off")

plt.suptitle("Sample Images - Cancer vs Non-Cancer")
plt.tight_layout()
plt.show()



# --------------------- ğŸ“ˆ Plot Accuracy & Loss ---------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.show()



# --------------------- ğŸ”® Predictions & Confusion Matrix ---------------------
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Predict on validation set
val_preds = model.predict(X_val, batch_size=32)
val_pred_labels = np.argmax(val_preds, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_val, val_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Cancer", "Cancer"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_val, val_pred_labels, target_names=["Non-Cancer", "Cancer"]))





