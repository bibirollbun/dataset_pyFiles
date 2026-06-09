# ==========================================
# 1 — IMPORTS
# ==========================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==========================================
# 2 — LOAD & PREPARE LABELS
# ==========================================

train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
train_df["image_id"] = train_df["image_id"] + ".jpg"

# Shuffle dataframe (VERY IMPORTANT)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert one-hot to single label
train_df["label"] = train_df[["healthy", "multiple_diseases", "rust", "scab"]].idxmax(axis=1)

label_map = {"healthy":0, "multiple_diseases":1, "rust":2, "scab":3}
train_df["label"] = train_df["label"].map(label_map).astype(str)

train_df.head()


# ==========================================
# 3 — DATA GENERATORS
# ==========================================

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# train_gen = datagen.flow_from_dataframe(
#     dataframe=train_df,
#     directory="/kaggle/input/plant-pathology-2020-fgvc7/images",
#     x_col="image_id",
#     y_col="label",
#     target_size=(128, 128),
#     batch_size=16,
#     class_mode="sparse",
#     subset="training"
# )

# val_gen = datagen.flow_from_dataframe(
#     dataframe=train_df,
#     directory="/kaggle/input/plant-pathology-2020-fgvc7/images",
#     x_col="image_id",
#     y_col="label",
#     target_size=(128, 128),
#     batch_size=16,
#     class_mode="sparse",
#     subset="validation"
# )

# ==========================================
# 3A — ADVANCED DATA AUGMENTATION (WEEK 6)
# ==========================================

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    validation_split=0.2
)

val_aug = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_aug.flow_from_dataframe(
    dataframe=train_df,
    directory="/kaggle/input/plant-pathology-2020-fgvc7/images",
    x_col="image_id",
    y_col="label",
    target_size=(224, 224),   # <- required for MobileNetV2
    batch_size=16,
    class_mode="sparse",
    subset="training"
)

val_gen = val_aug.flow_from_dataframe(
    dataframe=train_df,
    directory="/kaggle/input/plant-pathology-2020-fgvc7/images",
    x_col="image_id",
    y_col="label",
    target_size=(224, 224),
    batch_size=16,
    class_mode="sparse",
    subset="validation"
)

# ==========================================
# 4 — CUSTOM CNN (OFFLINE, NO ERRORS)
# ==========================================

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,3)),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(4, activation='softmax')
# ])

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()
# ==========================================
# 4A — TRANSFER LEARNING MODEL (WEEK 6)
# ==========================================

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None   # <- CHANGE BACK
)

base_model.trainable = False  # Freeze for transfer learning

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output_layer = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 5 — TRAIN
# ==========================================

# # ==========================================
# 5A — TRAIN (WEEK 6 IMPROVED MODEL)
# ==========================================

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)
# ==========================================
# 5B — FINE-TUNE MOBILENETV2
# ==========================================

base_model.trainable = True   # unfreeze

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)


# ==========================================
# 6 — TEST DATA
# ==========================================

test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
test_df["image_id"] = test_df["image_id"] + ".jpg"

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    dataframe=test_df,
    directory="/kaggle/input/plant-pathology-2020-fgvc7/images",
    x_col="image_id",
    y_col=None,
    target_size=(224, 224),   # <<< MATCH THE TRAINING SIZE
    batch_size=16,
    class_mode=None,
    shuffle=False
)
preds = model.predict(test_gen)


# ==========================================
# 7 — PREDICT
# ==========================================

preds = model.predict(test_gen)


# ==========================================
# 8 — SUBMISSION
# ==========================================

submission = pd.DataFrame({
    "image_id": test_df["image_id"],
    "healthy": preds[:, 0],
    "multiple_diseases": preds[:, 1],
    "rust": preds[:, 2],
    "scab": preds[:, 3],
})

submission.to_csv("submission.csv", index=False)
submission.head()


