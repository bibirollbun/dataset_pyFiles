import os

os.listdir("/kaggle/input")


for folder in ['kaggle-json', 'dogs-vs-cats', 'dogs-vs-cats-redux-kernels-edition']:
    print(f"\nContents of: /kaggle/input/{folder}")
    print(os.listdir(f"/kaggle/input/{folder}")[:5])


import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import time


# Paths
DATA_DIR = "/kaggle/working/split_dataset"


import os

data_dir = "/kaggle/input/dogs-vs-cats/train/train"
print("Sample files:", os.listdir(data_dir)[:5])


import os, shutil

data_dir = "/kaggle/input/dogs-vs-cats/train/train"
split_dir = "/kaggle/working/split_dataset"
cat_dir = os.path.join(split_dir, "cats")
dog_dir = os.path.join(split_dir, "dogs")

os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

for fname in os.listdir(data_dir):
    src = os.path.join(data_dir, fname)
    if fname.startswith("cat"):
        shutil.copy(src, os.path.join(cat_dir, fname))
    elif fname.startswith("dog"):
        shutil.copy(src, os.path.join(dog_dir, fname))


print("Cats:", len(os.listdir(cat_dir)))
print("Dogs:", len(os.listdir(dog_dir)))



from tensorflow.keras.utils import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    split_dir,
    labels="inferred",
    label_mode="binary",
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = image_dataset_from_directory(
    split_dir,
    labels="inferred",
    label_mode="binary",
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    subset="validation",
    seed=42
)



AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)





import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



# === MODEL SETUP ===
def build_final_mobilenetv2():
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = True  # Enable fine-tuning

    # Freeze bottom layers (optional)
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, x)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# === TRAINING ===
final_model = build_final_mobilenetv2()

early_stop = EarlyStopping(patience=3, restore_best_weights=True)

history = final_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stop],
    verbose=1
)


import tensorflow as tf
import numpy as np
import pandas as pd
import os

# === Define test directory path ===
test_dir = "/kaggle/input/dogs-vs-cats/test/test"

# === Get sorted list of .jpg test filenames ===
test_filenames = sorted(
    [f for f in os.listdir(test_dir) if f.endswith('.jpg')],
    key=lambda x: int(x.split('.')[0])
)

# === Preprocessing function for test images ===
def preprocess_test_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img

# === Create dataset from file paths ===
file_paths = [os.path.join(test_dir, fname) for fname in test_filenames]
file_paths_tensor = tf.constant(file_paths, dtype=tf.string)
test_ds = tf.data.Dataset.from_tensor_slices(file_paths_tensor)
test_ds = test_ds.map(preprocess_test_image).batch(32)

# === Predict using final model ===
preds = final_model.predict(test_ds, verbose=1).squeeze()

# === Extract IDs from filenames ===
ids = [int(fname.split('.')[0]) for fname in test_filenames]

# === Create submission DataFrame and save ===
submission = pd.DataFrame({"id": ids, "label": preds})
submission.to_csv("submission.csv", index=False)



submission




