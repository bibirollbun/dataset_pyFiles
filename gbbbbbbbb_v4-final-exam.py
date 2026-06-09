import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, mixed_precision


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("Mixed precision enabled:", policy)


IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
FROZEN_EPOCHS = 3

train_csv_path = "/kaggle/input/diabetic-retinopathy-detection/trainLabels.csv.zip"
train_images_dir = "/kaggle/input/diabetic-retinopathy-train-unzipped/train/"
test_images_dir = "/kaggle/input/diabetic-retinopathy-test-unzipped/test/"
submission_csv_path = "/kaggle/input/diabetic-retinopathy-detection/sampleSubmission.csv.zip"


df_train = pd.read_csv(train_csv_path)
df_train["filepath"] = df_train["image"].apply(lambda x: os.path.join(train_images_dir, f"{x}.jpeg"))

train_df, val_df = train_test_split(df_train, test_size=0.2, stratify=df_train["level"], random_state=42)

df_submission = pd.read_csv(submission_csv_path)
df_submission["filepath"] = df_submission["image"].apply(lambda x: os.path.join(test_images_dir, f"{x}.jpeg"))


# ====================================================
# 4. tf.data Pipelines
# ====================================================
def load_image_and_label(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

def one_hot_encode(image, label):
    label = tf.one_hot(label, 5)
    return image, label

def load_test_image(filepath):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image


# Training set
ds_train = tf.data.Dataset.from_tensor_slices((train_df["filepath"].values, train_df["level"].values))
ds_train = ds_train.shuffle(len(train_df)).map(load_image_and_label, tf.data.AUTOTUNE)
ds_train = ds_train.map(augment, tf.data.AUTOTUNE).map(one_hot_encode, tf.data.AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Validation set
ds_val = tf.data.Dataset.from_tensor_slices((val_df["filepath"].values, val_df["level"].values))
ds_val = ds_val.map(load_image_and_label, tf.data.AUTOTUNE).map(one_hot_encode, tf.data.AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Test set
ds_test = tf.data.Dataset.from_tensor_slices(df_submission["filepath"].values)
ds_test = ds_test.map(load_test_image, tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ====================================================
# 5. Build Model
# ====================================================
def build_model(name):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    if name == "EfficientNetB2":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base = tf.keras.applications.EfficientNetB2(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif name == "ConvNeXtTiny":
        preprocess = tf.keras.applications.convnext.preprocess_input
        base = tf.keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = layers.Lambda(preprocess)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(5, activation="softmax", dtype="float32")(x)
    return models.Model(inputs, outputs), base


# ====================================================
# 6. Train and Store Models
# ====================================================
def lr_scheduler(epoch, lr):
    if epoch < FROZEN_EPOCHS:
        return lr
    else:
        return lr * 0.9 if epoch % 3 == 0 else lr

lr_callback = callbacks.LearningRateScheduler(lr_scheduler)

model_names = ["EfficientNetB2", "ConvNeXtTiny"]
models_dict = {}
val_accuracies = {}

for name in model_names:
    print(f"\nTraining {name}...")
    model, base = build_model(name)

    # Phase 1: Freeze base
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds_train, validation_data=ds_val, epochs=FROZEN_EPOCHS, callbacks=[lr_callback], verbose=2)

    # Phase 2: Unfreeze base
    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS - FROZEN_EPOCHS, callbacks=[lr_callback], verbose=2)

    val_loss, val_acc = model.evaluate(ds_val, verbose=0)
    val_accuracies[name] = val_acc
    models_dict[name] = model
    print(f"{name} validation accuracy: {val_acc:.4f}")





# ====================================================
# 7. Weighted Ensemble on Test Set
# ====================================================
total_acc = sum(val_accuracies.values())
weights = {name: acc / total_acc for name, acc in val_accuracies.items()}
print("Ensemble Weights:", weights)

preds_list = []
for name in model_names:
    preds = models_dict[name].predict(ds_test, verbose=0)
    preds_list.append(preds)

ensemble_preds = np.zeros_like(preds_list[0])
for i, name in enumerate(model_names):
    ensemble_preds += preds_list[i] * weights[name]

final_preds = np.argmax(ensemble_preds, axis=1)


# ====================================================
# 8. Submission
# ====================================================
df_submission["level"] = final_preds
df_submission[["image", "level"]].to_csv("submission.csv", index=False)
print("✅ Submission file saved as 'submission.csv'")

