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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pydicom
from glob import glob
import matplotlib.patches as patches


labels = pd.read_csv("/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")
class_info = pd.read_csv("/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv")

df = pd.merge(labels, class_info, on="patientId", how="left")
df.sample(4)


print("NaN rows:", df['class'].isna().sum())


print("Total Images:", df['patientId'].nunique())
print("Total Rows:", df.shape[0])
# Plot target distribution
sns.countplot(data=df, x='Target')
plt.title('Pneumonia (1) vs No Pneumonia (0)')
plt.show()
# Unique patientId per class
print("Patients with Pneumonia:", df[df['Target']==1]['patientId'].nunique())
print("Patients without Pneumonia:", df[df['Target']==0]['patientId'].nunique())


sns.countplot(data=df, x='class')
plt.title('Class Distribution')
plt.xticks(rotation=20)
plt.show()


def show_dicom_image(patient_id, bbox=False):
   dicom_path = f"/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/{patient_id}.dcm"
   ds = pydicom.dcmread(dicom_path)
   fig, ax = plt.subplots(1, 1, figsize=(8, 8))
   ax.imshow(ds.pixel_array, cmap='gray')
   if bbox:
       records = df[df['patientId'] == patient_id]
       for _, row in records.iterrows():
           if row['Target'] == 1:
               rect = patches.Rectangle(
                   (row['x'], row['y']), row['width'], row['height'],
                   linewidth=2, edgecolor='red', facecolor='none'
               )
               ax.add_patch(rect)
   plt.title(patient_id)
   plt.show()


pneumonia_patients = df[df['Target']==1]['patientId'].unique()
show_dicom_image(pneumonia_patients[0], bbox=True)


!pip install -q pydicom tensorflow-addons


import os, pydicom, cv2, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, mixed_precision

# Enable Mixed Precision
mixed_precision.set_global_policy('mixed_float16')

IMG_SIZE = 416
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 1  # Only one class: Pneumonia

# Load and prepare dataset
label_df = pd.read_csv('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
label_df = label_df[label_df['Target'] == 1]
img_dir = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images/'
grouped = label_df.groupby("patientId").agg(list).reset_index()

# Split dataset
train_group, val_group = train_test_split(grouped, test_size=0.2, random_state=42)

# Data Loader
def load_image_and_labels(row):
    pid = row['patientId']
    dicom_path = os.path.join(img_dir, pid + ".dcm")
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.uint8)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.stack([img]*3, axis=-1) / 255.0

    w_ratio = IMG_SIZE / ds.Columns
    h_ratio = IMG_SIZE / ds.Rows
    boxes = []
    for x, y, w, h in zip(row['x'], row['y'], row['width'], row['height']):
        x1 = x * w_ratio
        y1 = y * h_ratio
        x2 = (x + w) * w_ratio
        y2 = (y + h) * h_ratio
        boxes.append([x1/IMG_SIZE, y1/IMG_SIZE, x2/IMG_SIZE, y2/IMG_SIZE])
    if len(boxes) == 0:
        boxes = [[0,0,0,0]]
    return img, np.array(boxes[0], dtype=np.float32).reshape(-1, 4)

def make_dataset(group):
    def generator():
        for i in range(len(group)):
            yield load_image_and_labels(group.iloc[i])
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
        )
    ).padded_batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_dataset = make_dataset(train_group)
val_dataset = make_dataset(val_group)

# Model
def build_yolo(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=1, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(4, dtype="float32")(x)  # [x1, y1, x2, y2]
    return tf.keras.Model(inputs, x)

# Compile
model = build_yolo()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Callbacks
ckpt = callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss')
early = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
with tf.device('/GPU:0'):
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[ckpt, early]
    )

# Evaluation
val_preds = []
for i, row in val_group.iterrows():
    img, _ = load_image_and_labels(row)
    pred = model.predict(img[np.newaxis, ...])[0]
    x1, y1, x2, y2 = pred * IMG_SIZE
    val_preds.append({
        "patientId": row['patientId'],
        "x": x1, "y": y1,
        "width": x2 - x1,
        "height": y2 - y1
    })

# Export
pd.DataFrame(val_preds).to_csv("rsna_val_predictions.csv", index=False)
print("Exported results to rsna_val_predictions.csv")











