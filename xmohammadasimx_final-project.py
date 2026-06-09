!pip install --quiet tensorflow-addons

import os, math, numpy as np, pandas as pd, tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt



AUTO = tf.data.experimental.AUTOTUNE

GCS = KaggleDatasets().get_gcs_path()

IMAGE_SIZE = (512, 768)  # reduced, still keeps aspect ratio
EPOCHS = 30
BATCH_SIZE = 4        

def format_path(name):
    return f"{GCS}/images/{name}.jpg"



train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test_df  = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

train_paths = train_df.image_id.apply(format_path).values
test_paths  = test_df.image_id.apply(format_path).values

y_train = train_df.loc[:, "healthy":].values.astype("float32")

train_paths, val_paths, y_train, y_val = train_test_split(
    train_paths, y_train, test_size=0.10, shuffle=True, random_state=42)



def decode_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return (img, label) if label is not None else img

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img, label

train_ds = (tf.data.Dataset.from_tensor_slices((train_paths, y_train))
            .map(decode_image, AUTO)
            .map(augment, AUTO)
            .shuffle(512)
            .batch(BATCH_SIZE)
            .prefetch(AUTO))

val_ds = (tf.data.Dataset.from_tensor_slices((val_paths, y_val))
          .map(decode_image, AUTO)
          .batch(BATCH_SIZE)
          .cache()
          .prefetch(AUTO))

test_ds = (tf.data.Dataset.from_tensor_slices(test_paths)
           .map(decode_image, AUTO)
           .batch(BATCH_SIZE)
           .prefetch(AUTO))



def cosine_schedule(epoch, lr_max=3e-4, lr_min=1e-6, ramp=5):
    if epoch < ramp:
        return lr_max * (epoch + 1) / ramp
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * (epoch - ramp) / (EPOCHS - ramp))
    )

lr_cb = LearningRateScheduler(cosine_schedule, verbose=0)



lrs = [cosine_schedule(e) for e in range(EPOCHS)]
plt.figure(figsize=(6,4))
plt.plot(lrs)
plt.title("Cosine Warmup Learning Rate Schedule")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.show()



def build_model():
    base = InceptionResNetV2(
        input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
    )
    x = L.GlobalAveragePooling2D()(base.output)
    x = L.Dense(4, activation="softmax")(x)
    return tf.keras.Model(inputs=base.input, outputs=x)

model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[lr_cb],
    verbose=1
)



import matplotlib.pyplot as plt

def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(14,5))

    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_training(history)



def tta_predict(model, ds):
    preds = []

    for images in ds:
        p1 = model.predict(images, verbose=0)
        p2 = model.predict(tf.image.flip_left_right(images), verbose=0)
        p3 = model.predict(tf.image.flip_up_down(images), verbose=0)
        p4 = model.predict(tf.image.flip_left_right(tf.image.flip_up_down(images)), verbose=0)

        preds.append((p1 + p2 + p3 + p4) / 4.0)

    return np.concatenate(preds, axis=0)

# Run it
test_pred = tta_predict(model, test_ds)



import numpy as np
import matplotlib.pyplot as plt

labels = ['healthy','multiple_diseases','rust','scab']

# show 3 predictions (change number if needed)
for i in range(3):
    img_path = test_paths[i]
    
    # load image normally (no label)
    img = decode_image(img_path)
    
    pred = test_pred[i]
    cls = labels[np.argmax(pred)]
    prob = np.max(pred)

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(f"Prediction: {cls}\nConfidence: {prob:.3f}")
    plt.axis('off')
    plt.show()



pseudo = test_pred.max(axis=1) >= 0.95
pseudo_y = test_pred[pseudo]
pseudo_x = test_paths[pseudo]

x_all = np.concatenate([train_paths, pseudo_x])
y_all = np.concatenate([y_train, pseudo_y])

full_ds = (tf.data.Dataset.from_tensor_slices((x_all, y_all))
            .map(decode_image, AUTO)
            .shuffle(512)
            .map(augment, AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO))

model.fit(full_ds, epochs=5, verbose=1)



final_pred = tta_predict(model, test_ds)

sub = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")
sub.loc[:, "healthy":] = final_pred
sub.to_csv("submission_pseudo.csv", index=False)
sub.head()


