#!pip install -q keras-cv


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras import mixed_precision


BATCH_SIZE = 32
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

train_csv_path = "/kaggle/input/diabetic-retinopathy-detection/trainLabels.csv.zip"
train_images_dir = "/kaggle/input/diabetic-retinopathy-train-unzipped/train/"
test_images_dir = "/kaggle/input/diabetic-retinopathy-test-unzipped/test/"
submission_csv_path = "/kaggle/input/diabetic-retinopathy-detection/sampleSubmission.csv.zip"


df_train = pd.read_csv(train_csv_path)
df_train["filepath"] = df_train["image"].apply(lambda x: os.path.join(train_images_dir, f"{x}.jpeg"))

train_df, val_df = train_test_split(df_train, test_size=0.2, stratify=df_train["level"], random_state=42)

df_submission = pd.read_csv(submission_csv_path)
df_submission["filepath"] = df_submission["image"].apply(lambda x: os.path.join(test_images_dir, f"{x}.jpeg"))


def load_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return (img, label) if label is not None else img


train_ds = tf.data.Dataset.from_tensor_slices((train_df["filepath"], train_df["level"]))
val_ds = tf.data.Dataset.from_tensor_slices((val_df["filepath"], val_df["level"]))
train_ds = train_ds.shuffle(1024).map(load_image, AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.map(load_image, AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)


base_model = ConvNeXtTiny(include_top=False, input_shape=(*IMG_SIZE, 3), weights="imagenet")
base_model.trainable = True

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(5, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# CALLBACKS
cb = [
    callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
    callbacks.EarlyStopping(patience=4, restore_best_weights=True)
]

# TRAIN
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cb)

# TEST
test_paths = sorted(glob('/kaggle/input/diabetic-retinopathy-test-unzipped/test/*.jpeg'))
test_df = pd.DataFrame({
    'image': [os.path.basename(p).split('.')[0] for p in test_paths],
    'filepath': test_paths
})


test_ds = tf.data.Dataset.from_tensor_slices(df_submission["filepath"].values)
test_ds = test_ds.map(load_image, AUTOTUNE).batch(BATCH_SIZE)

# PREDICT
preds = model.predict(test_ds, verbose=0)


df_submission["level"] = np.argmax(preds, axis=1)
df_submission[["image", "level"]].to_csv("submission.csv", index=False)
print("submission.csv saved")

