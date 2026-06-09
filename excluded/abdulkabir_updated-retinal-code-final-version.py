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
import os
import shutil

# Paths
csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
img_folder = "/kaggle/input/aptos2019-blindness-detection/train_images"
output_folder = "/kaggle/working/aptos_cleaned"

# Class labels mapping
label_map = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

# Load CSV
df = pd.read_csv(csv_path)

# Create class folders
for class_name in label_map.values():
    os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

# Copy images to class folders
for _, row in df.iterrows():
    img_file = row['id_code'] + ".png"
    label = row['diagnosis']
    class_name = label_map[label]

    src_path = os.path.join(img_folder, img_file)
    dst_path = os.path.join(output_folder, class_name, img_file)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)

print("âœ… Images copied to class folders successfully!")



import shutil
import os

# Source folders
old_dataset = "/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images"
new_dataset = "/kaggle/working/aptos_cleaned"

# Combined target
combined_folder = "/kaggle/working/combined_data"
os.makedirs(combined_folder, exist_ok=True)

# Classes
classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]

for cls in classes:
    os.makedirs(os.path.join(combined_folder, cls), exist_ok=True)

    # Move old data
    old_cls_path = os.path.join(old_dataset, cls)
    if os.path.exists(old_cls_path):
        for file in os.listdir(old_cls_path):
            src = os.path.join(old_cls_path, file)
            dst = os.path.join(combined_folder, cls, f"old_{file}")
            shutil.copyfile(src, dst)

    # Move new data
    new_cls_path = os.path.join(new_dataset, cls)
    if os.path.exists(new_cls_path):
        for file in os.listdir(new_cls_path):
            src = os.path.join(new_cls_path, file)
            dst = os.path.join(combined_folder, cls, f"new_{file}")
            shutil.copyfile(src, dst)

print("âœ… Combined dataset created!")



import os

combined_path = "/kaggle/working/combined_data"
for cls in os.listdir(combined_path):
    cls_path = os.path.join(combined_path, cls)
    print(f"{cls} ğŸ“� => {len(os.listdir(cls_path))} images")



import matplotlib.pyplot as plt
import cv2
import random

folder = "/kaggle/working/combined_data/Mild"
img_name = random.choice(os.listdir(folder))
img_path = os.path.join(folder, img_name)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title(f"Class: Mild | Image: {img_name}")
plt.axis('off')
plt.show()



import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight

# âœ… Paths & Config
data_dir = "/kaggle/working/combined_data"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8

# âœ… Data Generators with Augmentation
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# âœ… Model Architecture
base_model = EfficientNetB7(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# âœ… Handle Class Imbalance
labels = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))
print("âœ… Class weights applied:", class_weights_dict)

# âœ… Callbacks
checkpoint = ModelCheckpoint("retina_model_best.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# âœ… Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, early_stop],
    class_weight=class_weights_dict
)

# âœ… Save clean weights
model.load_weights("retina_model_best.keras")
model.save_weights("/kaggle/working/retina_dr_model_b7_weights_only_final.weights.h5")
print("âœ… .weights.h5 file saved successfully!")

# Optional: âœ… Export to TFLite
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open("/kaggle/working/retina_model.tflite", "wb") as f:
#     f.write(tflite_model)
# print("âœ… .tflite model exported successfully!")



import tensorflow as tf

# âš’ï¸� Rebuild model architecture
def build_model():
    base_model = EfficientNetB7(include_top=False, weights=None, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(5, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

# ğŸ”„ Build & load weights
model = build_model()
model.load_weights("/kaggle/working/retina_dr_model_b7_weights_only_final.weights.h5")
print("âœ… Model rebuilt and weights loaded")

# ğŸ�¯ Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ğŸ’¾ Save the TFLite model
tflite_path = "/kaggle/working/retina_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"âœ… .tflite model saved at: {tflite_path}")



import zipfile

# ğŸ“¦ Zip the TFLite and Weights file
with zipfile.ZipFile("/kaggle/working/retina_model_files_final.zip", 'w') as zipf:
    zipf.write("/kaggle/working/retina_dr_model_b7_weights_only_final.weights.h5")
    zipf.write("/kaggle/working/retina_model.tflite")

print("âœ… Final zip created")




import os

os.makedirs("/kaggle/working/output", exist_ok=True)



import shutil

shutil.rmtree("/kaggle/working/aptos_cleaned", ignore_errors=True)
shutil.rmtree("/kaggle/working/combined_data", ignore_errors=True)
shutil.rmtree("/kaggle/working/Mild", ignore_errors=True)
shutil.rmtree("/kaggle/working/Moderate", ignore_errors=True)
shutil.rmtree("/kaggle/working/No_DR", ignore_errors=True)
shutil.rmtree("/kaggle/working/Severe", ignore_errors=True)
shutil.rmtree("/kaggle/working/Proliferate_DR", ignore_errors=True)

print("âœ… Unused folders removed. You should have enough space now.")



import os, shutil

os.makedirs("/kaggle/working/output", exist_ok=True)

files_to_copy = [
    "/kaggle/working/retina_dr_model_b7_weights_only_final.weights.h5",
    "/kaggle/working/retina_model.tflite",
    "/kaggle/working/retina_model_best.keras",
    "/kaggle/working/retina_model_files_final.zip"
]

for file in files_to_copy:
    shutil.copy(file, "/kaggle/working/output")

print("âœ… All files copied to /kaggle/working/output/")

















