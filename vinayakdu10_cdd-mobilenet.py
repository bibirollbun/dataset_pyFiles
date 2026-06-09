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
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# ========== Paths ==========
WORK_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(WORK_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
LABEL_MAP_JSON = os.path.join(WORK_DIR, "label_num_to_disease_map.json")

# ========== Load dataset ==========
train_df = pd.read_csv(TRAIN_CSV)
with open(LABEL_MAP_JSON) as f:
    label_map = json.load(f)

train_df["label"] = train_df["label"].astype(str)

# Stratified train-validation split
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=42)

# ========== Parameters ==========
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = len(label_map)

# ========== Data Generators ==========
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_dataframe(
    train_df,
    directory=TRAIN_IMG_DIR,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_data = val_datagen.flow_from_dataframe(
    val_df,
    directory=TRAIN_IMG_DIR,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ========== Soft Attention Module ==========
def soft_attention_module(input_tensor):
    # Channel Attention
    channel_avg = layers.GlobalAveragePooling2D()(input_tensor)
    channel_avg = layers.Reshape((1, 1, channel_avg.shape[1]))(channel_avg)
    channel_weights = layers.Conv2D(filters=input_tensor.shape[-1] // 8, kernel_size=1, activation='relu')(channel_avg)
    channel_weights = layers.Conv2D(filters=input_tensor.shape[-1], kernel_size=1, activation='sigmoid')(channel_weights)
    channel_refined = layers.Multiply()([input_tensor, channel_weights])
    
    # Spatial Attention
    spatial_weights = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')(channel_refined)
    refined_features = layers.Multiply()([channel_refined, spatial_weights])
    return refined_features

# ========== Build Model ==========
def build_cddnet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze backbone initially
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = soft_attention_module(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name='CDDNet')
    return model

model = build_cddnet()

# ========== Compile ==========
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== Callbacks ==========
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

# ========== Train ==========
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[reduce_lr, early_stop]
)

# ========== Optional Fine-tuning ==========
# Unfreeze backbone for fine-tuning
base_model = model.get_layer('mobilenetv3small_100')
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[reduce_lr, early_stop]
)



from tensorflow.keras import callbacks

checkpoint_best = callbacks.ModelCheckpoint(
    "cassava_best.h5",          # filename for best model
    monitor="val_accuracy",     # metric to monitor
    mode="max",                 # want to maximize accuracy
    save_best_only=True,        # save only when val_accuracy improves
    verbose=1
)

checkpoint_last = callbacks.ModelCheckpoint(
    "cassava_last.h5",          # filename for last epoch model
    save_best_only=False,       # always save last epoch model
    verbose=1
)



# Fix: get backbone layer by correct name
base_model = model.get_layer('MobileNetV3Small')
base_model.trainable = True

# Recompile before fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Visualize model architecture
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image
Image('model_architecture.png')



val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {val_accuracy:.4f}")



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict class probabilities
val_data.reset()  # Make sure data is at start
y_pred_probs = model.predict(val_data, verbose=1)

# Convert predicted probabilities to class indices
y_pred = np.argmax(y_pred_probs, axis=1)

# True labels from validation generator
y_true = val_data.classes

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_data.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()



import matplotlib.pyplot as plt

def plot_training_history(history, title_suffix=""):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot for initial training
plot_training_history(history, title_suffix="(Initial Training)")

# Plot for fine-tuning if available
if 'history_finetune' in locals():
    plot_training_history(history_finetune, title_suffix="(Fine-tuning)")



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os

# ========== Paths ==========
img_path = "/kaggle/input/cassava-leaf-disease-classification/train_images/288080098.jpg"

# ========== Parameters ==========
IMG_SIZE = 224
last_conv_layer_name = 'multiply_10'  # Last conv layer with spatial info
class_names = ['Cassava Bacterial Blight', 'Cassava Brown Streak Disease', 'Cassava Green Mottle', 'Cassava Mosaic Disease', 'Healthy']

# ========== Load model (assuming it's already loaded as `model`) ==========
# If not loaded, load here:
# model = tf.keras.models.load_model('your_model_path')

# ========== Preprocess Image ==========
def preprocess_img(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========== Generate Grad-CAM heatmap ==========
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
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
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()

# ========== Overlay heatmap on image ==========
def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    overlayed_img = heatmap_colored * alpha + img
    overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)
    return overlayed_img

# ========== Plot training history ==========
def plot_history(history):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# ========== Run Grad-CAM and display ==========
img_array = preprocess_img(img_path)
preds = model.predict(img_array)
predicted_class = np.argmax(preds[0])
print(f"Predicted Class: {class_names[predicted_class]} (Confidence: {preds[0][predicted_class]:.3f})")

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=predicted_class)
overlay_img = overlay_heatmap(img_path, heatmap)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.axis('off')
plt.imshow(image.load_img(img_path))

plt.subplot(1, 2, 2)
plt.title('Grad-CAM Overlay')
plt.axis('off')
plt.imshow(overlay_img)
plt.show()



for layer in model.layers:
    print(layer.name, layer.output.shape)



import time

def measure_fps(model, data_generator, steps=100):
    start_time = time.time()
    for i, (x_batch, _) in enumerate(data_generator):
        if i >= steps:
            break
        _ = model.predict(x_batch)
    end_time = time.time()
    total_time = end_time - start_time
    fps = (steps * data_generator.batch_size) / total_time
    return fps

# Measure your model's FPS on validation set
fps_your_model = measure_fps(model, val_data, steps=50)
print(f"Your model FPS: {fps_your_model:.2f}")

# For SOTA models, you'd need their implementations and run similar timing.
# You can create a dictionary to compare:
fps_comparison = {
    "Your Model (CDDNet)": fps_your_model,
    "MobileNetV2": 50,  # Hypothetical FPS
    "ResNet50": 30,
    "EfficientNetB0": 45
}

# Plot FPS comparison
plt.bar(fps_comparison.keys(), fps_comparison.values())
plt.ylabel("FPS")
plt.title("FPS Comparison with SOTA Models")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# ========== Paths ==========
IMG_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images"
IMG_SIZE = 224
last_conv_layer_name = 'multiply_10'
class_names = ['Cassava Bacterial Blight', 'Cassava Brown Streak Disease', 'Cassava Green Mottle', 'Cassava Mosaic Disease', 'Healthy']

# ========== Helper functions ==========

def preprocess_img(img_path, target_size=(IMG_SIZE, IMG_SIZE)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
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
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    overlayed_img = heatmap_colored * alpha + img
    overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)
    return overlayed_img

# ========== Main code ==========

# Randomly pick 4 images from training directory
all_images = os.listdir(IMG_DIR)
sample_images = random.sample(all_images, 4)

plt.figure(figsize=(20, 5))
for i, img_name in enumerate(sample_images):
    img_path = os.path.join(IMG_DIR, img_name)
    img_array = preprocess_img(img_path)
    
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]
    
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_class)
    overlay_img = overlay_heatmap(img_path, heatmap)
    
    # Plot original image
    plt.subplot(2, 4, i + 1)
    plt.imshow(image.load_img(img_path))
    plt.title(f"Orig\n{img_name[:15]}")
    plt.axis('off')
    
    # Plot Grad-CAM overlay
    plt.subplot(2, 4, i + 5)
    plt.imshow(overlay_img)
    plt.title(f"Pred: {class_names[pred_class]}\nConf: {confidence:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()


