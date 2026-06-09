import os

print("Contents of /kaggle/input:")
for f in os.listdir("/kaggle/input"):
    print(" -", f)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

root_dir = "/kaggle/input/aptos2019-blindness-detection"
train_img_dir = os.path.join(root_dir, "train_images")
test_img_dir  = os.path.join(root_dir, "test_images")

# Read CSVs
train_df = pd.read_csv(os.path.join(root_dir, "train.csv"))
test_df  = pd.read_csv(os.path.join(root_dir, "test.csv"))

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
train_df.head()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add file paths
train_df["file_path"] = train_df["id_code"].apply(lambda x: os.path.join(train_img_dir, f"{x}.png"))
test_df["file_path"]  = test_df["id_code"].apply(lambda x: os.path.join(test_img_dir, f"{x}.png"))

# Labels must be string for flow_from_dataframe
train_df["diagnosis"] = train_df["diagnosis"].astype(str)

# Check missing images
missing = train_df[~train_df["file_path"].apply(os.path.exists)]
print("Missing training images:", len(missing))

# ImageDataGenerator as in template (DO NOT CHANGE)
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

# Train generator
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="file_path",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# Validation generator
val_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="file_path",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Test generator
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="file_path",
    y_col=None,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

print("Training samples:", train_generator.samples)
print("Validation samples:", val_generator.samples)
print("Test samples:", test_generator.samples)


import matplotlib.pyplot as plt

# Just to *view* images nicely – reverse ResNet preprocess for display only
def undo_resnet_preprocess(img):
    img = img.copy()
    img = img + [103.939, 116.779, 123.68]   # add mean
    img = img[..., ::-1]                    # BGR → RGB
    img = np.clip(img / 255.0, 0, 1)
    return img

def show_sample(generator, title):
    imgs, labels = next(generator)
    plt.figure(figsize=(12,4))
    for i in range(3):
        plt.subplot(1,3,i+1)
        restored = undo_resnet_preprocess(imgs[i])
        plt.imshow(restored)
        plt.title(f"Label: {np.argmax(labels[i])}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# Train samples
show_sample(train_generator, "Training Images (Sample)")

# Test samples (no labels)
test_imgs = next(test_generator)
plt.figure(figsize=(12,4))
for i in range(3):
    plt.subplot(1,3,i+1)
    restored = undo_resnet_preprocess(test_imgs[i])
    plt.imshow(restored)
    plt.axis("off")
plt.suptitle("Test Images (Sample)")
plt.show()


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

# Base EfficientNetB0
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg"
)

# Phase 1: freeze the entire backbone
base_model.trainable = False

# Simple classification head
x = base_model.output
x = Dropout(0.4)(x)
outputs = Dense(5, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase1 = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

print("Phase 1: training classifier head (base frozen)...")

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=8,
    callbacks=callbacks_phase1
)

print("Phase 1 done.")


# Unfreeze the top 40 layers of the EfficientNet backbone
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

print("Phase 2: fine-tuning top EfficientNet layers...")

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks_phase2
)

print("Phase 2 done.")


# Save model after training
model.save("model.h5")
print("Model saved as model.h5")


def plot_history(histories, labels):
    plt.figure(figsize=(14,6))

    # Accuracy
    plt.subplot(1,2,1)
    for h, lbl in zip(histories, labels):
        plt.plot(h.history['accuracy'], label=f"{lbl} Train")
        plt.plot(h.history['val_accuracy'], label=f"{lbl} Val", linestyle="--")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    for h, lbl in zip(histories, labels):
        plt.plot(h.history['loss'], label=f"{lbl} Train")
        plt.plot(h.history['val_loss'], label=f"{lbl} Val", linestyle="--")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

print("Plotting learning curves...")
plot_history([history1, history2], ["Phase 1", "Phase 2"])


val_loss, val_acc = model.evaluate(val_generator)
print("FINAL VALIDATION ACCURACY:", val_acc)
print("FINAL VALIDATION LOSS:", val_loss)


print("Predicting on test set...")

test_predictions = model.predict(test_generator)
predicted_labels = np.argmax(test_predictions, axis=1)

submission_df = pd.DataFrame({
    "id_code": test_df["id_code"],
    "diagnosis": predicted_labels
})

submission_df.to_csv("submission.csv", index=False)

print("submission.csv saved!")
submission_df.head()


submission_df

