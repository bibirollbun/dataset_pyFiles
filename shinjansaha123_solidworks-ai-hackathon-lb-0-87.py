import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


TRAIN_IMG_DIR = "/kaggle/input/solidworks-ai-hackathon/train/train"
TEST_IMG_DIR  = "/kaggle/input/solidworks-ai-hackathon/test/test"

TRAIN_LABELS = "/kaggle/input/solidworks-ai-hackathon/train_labels.csv"
TRAIN_BBOXES = "/kaggle/input/solidworks-ai-hackathon/train_bboxes.csv"

labels_df = pd.read_csv(TRAIN_LABELS)
bboxes_df = pd.read_csv(TRAIN_BBOXES)

INPUT_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60

CLASSES = ["bolt", "locatingpin", "nut", "washer"]


labels_df.head()


bboxes_df.head()


max_counts = labels_df[CLASSES].max().to_dict()
print(max_counts)


for c in CLASSES:
    print(c)
    print(labels_df[c].value_counts().sort_index())
    print()


def load_image(path, labels):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, labels

def make_dataset(df, shuffle=False):
    paths = df["image_name"].apply(
        lambda x: os.path.join(TRAIN_IMG_DIR, x)
    ).values

    labels = {
        p: df[p].values.astype("int32")
        for p in CLASSES
    }

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(1024)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

train_ds = make_dataset(train_df, shuffle=True)
val_ds   = make_dataset(val_df)


print("Shape of train_df: ", train_df.shape)
print("Shape of val_df: ", val_df.shape)


inputs = tf.keras.Input(shape=(224, 224, 3))

x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)


outputs = {
    p: layers.Dense(
        max_counts[p] + 1,
        activation="softmax",
        name=p
    )(x)
    for p in CLASSES
}

model = tf.keras.Model(inputs, outputs)
model.summary()


losses = {p: "sparse_categorical_crossentropy" for p in CLASSES}
metrics = {
    "bolt": "accuracy",
    "locatingpin": "accuracy",
    "nut": "accuracy",
    "washer": "accuracy",
}

model.compile(
    optimizer="adam",
    loss=losses,
    metrics=metrics
)


class ExactMatchCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        self.val_ds = val_ds
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        correct, total = 0, 0

        for x, y_true in self.val_ds:
            y_pred = self.model.predict(x, verbose=0)

            preds = [tf.argmax(y_pred[p], axis=1) for p in CLASSES]
            trues = [tf.cast(y_true[p], tf.int64) for p in CLASSES]

            match = tf.reduce_all(
                tf.stack([tf.equal(p, t) for p, t in zip(preds, trues)], axis=1),
                axis=1
            )

            correct += tf.reduce_sum(tf.cast(match, tf.int32)).numpy()
            total += match.shape[0]

        exact_acc = correct / total
        self.history.append(exact_acc)
        print(f"\nval_exact_match_accuracy: {exact_acc:.4f}")


exact_cb = ExactMatchCallback(val_ds)
early_stop = EarlyStopping(
    monitor="val_exact_match_accuracy",
    mode="max",
    patience=12,
    restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    "/kaggle/working/best_model.keras",
    monitor="val_exact_match_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[exact_cb, early_stop, checkpoint]
)


plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("/kaggle/working/loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(8,5))
for part in CLASSES:
    plt.plot(history.history[f"val_{part}_accuracy"], label=f"{part}")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Part")
plt.legend()
plt.grid(True)
plt.savefig("/kaggle/working/val_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(6,4))
plt.plot(exact_cb.history, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Exact Match Accuracy")
plt.title("Validation Exact-Match Accuracy")
plt.grid(True)
plt.savefig("/kaggle/working/exact_match_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()


def visualize_prediction(idx, save=True):
    row = val_df.iloc[idx]
    img_path = os.path.join(TRAIN_IMG_DIR, row["image_name"])

    img = tf.image.decode_png(tf.io.read_file(img_path), channels=3)
    img = tf.image.resize(img, (224,224)) / 255.0

    preds = model.predict(img[None], verbose=0)
    pred_counts = {c: int(np.argmax(preds[c])) for c in CLASSES}
    gt_counts = row[CLASSES].to_dict()

    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"GT: {gt_counts}\nPred: {pred_counts}")

    if save:
        plt.savefig(f"/kaggle/working/prediction_{idx}.png", dpi=300, bbox_inches="tight")

    plt.show()

visualize_prediction(0)
visualize_prediction(10)
visualize_prediction(50)


# Load sample submission
sample_sub = pd.read_csv(
    "/kaggle/input/solidworks-ai-hackathon/sample_submission.csv"
)

test_images = sample_sub["image_name"].tolist()
test_paths = [os.path.join(TEST_IMG_DIR, x) for x in test_images]

def load_test_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    return img

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
)

preds = model.predict(test_ds)

# Fill predictions
sample_sub["bolt"]        = np.argmax(preds["bolt"], axis=1)
sample_sub["locatingpin"] = np.argmax(preds["locatingpin"], axis=1)
sample_sub["nut"]         = np.argmax(preds["nut"], axis=1)
sample_sub["washer"]      = np.argmax(preds["washer"], axis=1)

# Save submission
sample_sub.to_csv("/kaggle/working/submission.csv", index=False)

sample_sub.head()

