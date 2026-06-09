# ============================================================
# CELL 0 — Protobuf compatibility patch (run FIRST)
# ============================================================
import google.protobuf
from google.protobuf import message_factory as _message_factory

print("protobuf version:", google.protobuf.__version__)

# Some TF 2.x builds expect MessageFactory.GetPrototype, which newer
# protobuf versions removed. This patch restores a compatible method.
if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
    def _GetPrototype(self, descriptor):
        from google.protobuf import message_factory as mf_mod
        return mf_mod.GetMessageClass(descriptor)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype
    print("Patched MessageFactory.GetPrototype for compatibility.")
else:
    print("MessageFactory already has GetPrototype; no patch needed.")



# ============================================================
# CELL 1 — Access and Preprocess Data (Q2 setup)
# ============================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TF version:", tf.__version__)

# ---- Root directory from competition ----
root_dir = "/kaggle/input/aptos2019-blindness-detection"
train_img_dir = os.path.join(root_dir, "train_images")
test_img_dir  = os.path.join(root_dir, "test_images")

print("Contents of /kaggle/input:", os.listdir("/kaggle/input"))
print("Contents of aptos2019-blindness-detection:", os.listdir(root_dir))

# ---- Load CSVs ----
train_df = pd.read_csv(os.path.join(root_dir, "train.csv"))
test_df  = pd.read_csv(os.path.join(root_dir, "test.csv"))

# Keep numeric labels for EDA
train_df["diagnosis_int"] = train_df["diagnosis"].copy()

# Build file paths for train and test images
train_df["file_path"] = train_df["id_code"].apply(
    lambda x: os.path.join(train_img_dir, f"{x}.png")
)
test_df["file_path"] = test_df["id_code"].apply(
    lambda x: os.path.join(test_img_dir, f"{x}.png")
)

# Convert labels to string for flow_from_dataframe (categorical)
train_df["diagnosis"] = train_df["diagnosis"].astype(str)

# Sanity checks: confirm all files exist
missing_train_files = train_df[~train_df["file_path"].apply(os.path.exists)]
missing_test_files  = test_df[~test_df["file_path"].apply(os.path.exists)]
print(f"Missing training files: {len(missing_train_files)}")
print(f"Missing test files: {len(missing_test_files)}")

# ============================================================
# Shared ImageDataGenerator (rescale + augmentation)
# Used for: autoencoder pretraining AND classifier training
# ============================================================
img_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  # 20% validation
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- Classifier generators (supervised) ---
clf_train_generator = img_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="file_path",
    y_col="diagnosis",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

clf_val_generator = img_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="file_path",
    y_col="diagnosis",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print(f"Classifier training samples: {clf_train_generator.samples}")
print(f"Classifier validation samples: {clf_val_generator.samples}")



# ============================================================
# CELL 2 — Simple visual EDA (class balance + sample images)
# ============================================================
import matplotlib.pyplot as plt

# 1) Class distribution bar plot
class_counts = train_df["diagnosis_int"].value_counts().sort_index()
plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar")
plt.xlabel("Diagnosis class")
plt.ylabel("Count")
plt.title("Class distribution in training data")
plt.show()

print("Class counts:")
print(class_counts)

# 2) Show 3 sample training images with labels
sample_train = train_df.sample(3, random_state=42)

plt.figure(figsize=(10, 4))
for i, row in enumerate(sample_train.itertuples(), 1):
    img = tf.keras.utils.load_img(row.file_path, target_size=IMG_SIZE)
    plt.subplot(1, 3, i)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Train\n{row.id_code}\nlabel={row.diagnosis_int}")
plt.tight_layout()
plt.show()

# 3) Show 3 sample test images
sample_test = test_df.sample(3, random_state=42)

plt.figure(figsize=(10, 4))
for i, row in enumerate(sample_test.itertuples(), 1):
    img = tf.keras.utils.load_img(row.file_path, target_size=IMG_SIZE)
    plt.subplot(1, 3, i)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Test\n{row.id_code}")
plt.tight_layout()
plt.show()



# ============================================================
# CELL 3 — Autoencoder Data Generators (unsupervised)
# ============================================================
# For the autoencoder we only need images (no labels).
# We reuse img_datagen with the same augmentations/validation split.

# These raw generators return ONLY X (images), no labels.
ae_train_gen_raw = img_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="file_path",
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    subset="training",
    shuffle=True
)

ae_val_gen_raw = img_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="file_path",
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    subset="validation",
    shuffle=False
)

# Wrap the raw generators so they output (x, x) pairs for autoencoder training
def make_autoencoder_generator(raw_gen):
    while True:
        batch_x = next(raw_gen)
        yield (batch_x, batch_x)

ae_train_generator = make_autoencoder_generator(ae_train_gen_raw)
ae_val_generator   = make_autoencoder_generator(ae_val_gen_raw)

print(f"AE training steps per epoch: {len(ae_train_gen_raw)}")
print(f"AE validation steps: {len(ae_val_gen_raw)}")



# ============================================================
# CELL 4 — Build Convolutional Autoencoder
# ============================================================
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
inputs = layers.Input(shape=input_shape)

# ---------- Encoder ----------
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D((2, 2), padding="same")(x)   # 112x112

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)   # 56x56

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)   # 28x28

x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
encoded = layers.MaxPooling2D((2, 2), padding="same", name="latent_feature")(x)  # 14x14x256

# ---------- Decoder ----------
x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(encoded)
x = layers.UpSampling2D((2, 2))(x)   # 28x28

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)   # 56x56

x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)   # 112x112

x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)   # 224x224

decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder = models.Model(inputs, decoded, name="retina_autoencoder")

autoencoder.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse"
)

autoencoder.summary()



# ============================================================
# CELL 5 — Train Autoencoder
# ============================================================
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

ae_callbacks = [
    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, verbose=1),
]

AE_EPOCHS = 5  # keep small for Kaggle runtime

history_ae = autoencoder.fit(
    ae_train_generator,
    steps_per_epoch=len(ae_train_gen_raw),
    epochs=AE_EPOCHS,
    validation_data=ae_val_generator,
    validation_steps=len(ae_val_gen_raw),
    callbacks=ae_callbacks
)



# ============================================================
# CELL 6 — (Optional) Plot Autoencoder Loss Curves
# ============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(history_ae.history["loss"], label="train_loss")
plt.plot(history_ae.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Autoencoder reconstruction loss")
plt.legend()
plt.show()



# ============================================================
# CELL 7 — Build Classifier Using the Encoder
# ============================================================
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

num_classes = len(clf_train_generator.class_indices)
print("Classes:", clf_train_generator.class_indices)

# Extract encoder (from inputs to latent feature layer)
encoder = Model(
    inputs=autoencoder.input,
    outputs=autoencoder.get_layer("latent_feature").output,
    name="retina_encoder"
)

# First, freeze encoder to train only classifier head
encoder.trainable = False

# Classifier model: encoder + global pooling + dense layers
clf_inputs = layers.Input(shape=input_shape)
x = encoder(clf_inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
clf_outputs = Dense(num_classes, activation="softmax")(x)

clf_model = Model(clf_inputs, clf_outputs, name="AE_classifier")

clf_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

clf_model.summary()



# ============================================================
# CELL 8 — Train Classifier Head (encoder frozen)
# ============================================================
from tensorflow.keras.callbacks import ModelCheckpoint

callbacks_head = [
    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, verbose=1),
    ModelCheckpoint("ae_classifier_head.weights.h5", monitor="val_loss", save_best_only=True, verbose=1),
]

HEAD_EPOCHS = 5  # short for assignment

history_head = clf_model.fit(
    clf_train_generator,
    epochs=HEAD_EPOCHS,
    validation_data=clf_val_generator,
    callbacks=callbacks_head
)



# ============================================================
# CELL 9 — Fine-tune Upper Encoder Layers
# ============================================================
# Unfreeze the encoder and fine-tune only the last few blocks.
encoder.trainable = True

# For simplicity, unfreeze the last N convolutional layers.
# (You can adjust N to control how much you fine-tune.)
N_UNFREEZE = 15
for layer in encoder.layers[:-N_UNFREEZE]:
    layer.trainable = False

clf_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_ft = [
    EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, verbose=1),
    ModelCheckpoint("ae_classifier_finetuned.weights.h5", monitor="val_loss", save_best_only=True, verbose=1),
]

FT_EPOCHS = 5  # again short run

history_ft = clf_model.fit(
    clf_train_generator,
    epochs=FT_EPOCHS,
    validation_data=clf_val_generator,
    callbacks=callbacks_ft
)



# ============================================================
# CELL 10 — Plot Classifier Learning Curves
# ============================================================
def plot_history(hist, title_prefix=""):
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()
    plt.show()

    if "accuracy" in hist.history:
        plt.figure(figsize=(6, 4))
        plt.plot(hist.history["accuracy"], label="train_acc")
        plt.plot(hist.history["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{title_prefix} Accuracy")
        plt.legend()
        plt.show()

plot_history(history_head, "AE Classifier (Head)")
plot_history(history_ft, "AE Classifier (Fine-tune)")

# ➜ Add a Markdown cell after this:
#    - Briefly describe whether val_loss/val_accuracy improved.
#    - Compare qualitatively with your Q1 EfficientNet model.



# ============================================================
# CELL 11 — Build Test Generator & Create submission.csv
# ============================================================
# For test data we only need images (no labels), same rescale.
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="file_path",
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)

print(f"Test samples: {test_generator.samples}")

# Predict
pred_probs = clf_model.predict(test_generator)
predicted_classes = tf.argmax(pred_probs, axis=1).numpy()

submission_df = pd.DataFrame({
    "id_code": test_df["id_code"],
    "diagnosis": predicted_classes
})

# VERY IMPORTANT: competition expects exactly this filename
submission_df.to_csv("submission.csv", index=False)
submission_df.head()


