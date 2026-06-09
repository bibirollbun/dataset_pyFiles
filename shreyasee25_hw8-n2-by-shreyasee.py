import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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

# Labels as string for classifier generator
train_df["diagnosis"] = train_df["diagnosis"].astype(str)

# Check for missing files
missing = train_df[~train_df["file_path"].apply(os.path.exists)]
print("Missing training images:", len(missing))

# Base ImageDataGenerator (same style as Q1)
base_datagen = ImageDataGenerator(
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

# 1) Generators for CLASSIFIER (like Q1)
train_classifier_gen = base_datagen.flow_from_dataframe(
    train_df,
    x_col="file_path",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_classifier_gen = base_datagen.flow_from_dataframe(
    train_df,
    x_col="file_path",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# 2) Generators for AUTOENCODER (input = output image)
train_ae_gen = base_datagen.flow_from_dataframe(
    train_df,
    x_col="file_path",
    y_col="file_path",         # dummy, we ignore labels; just use images as both input/output
    target_size=(224, 224),
    batch_size=32,
    class_mode="input",        # Keras will use images as y
    subset="training"
)

val_ae_gen = base_datagen.flow_from_dataframe(
    train_df,
    x_col="file_path",
    y_col="file_path",
    target_size=(224, 224),
    batch_size=32,
    class_mode="input",
    subset="validation"
)

# Test generator (for final predictions)
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

print("Train classifier samples:", train_classifier_gen.samples)
print("Val classifier samples:",   val_classifier_gen.samples)
print("Train AE samples:",         train_ae_gen.samples)
print("Val AE samples:",           val_ae_gen.samples)
print("Test samples:",             test_generator.samples)


# For visualization
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
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# Training images
show_sample(train_ae_gen, "Training Images (Autoencoder)")

# Test images
test_imgs = next(test_generator)
plt.figure(figsize=(12,4))
for i in range(3):
    plt.subplot(1,3,i+1)
    restored = undo_resnet_preprocess(test_imgs[i])
    plt.imshow(restored)
    plt.axis("off")
plt.suptitle("Test Images (Sample)")
plt.show()


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

input_img = Input(shape=(224, 224, 3))

# ----- Encoder -----
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same', name="encoded_layer")(x)  # bottleneck

# ----- Decoder -----
x = UpSampling2D((2,2))(encoded)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)

x = UpSampling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)

x = UpSampling2D((2,2))(x)
decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded, name="autoencoder")
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()


callbacks_ae = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

print("Training autoencoder...")

history_ae = autoencoder.fit(
    train_ae_gen,
    validation_data=val_ae_gen,
    epochs=10,
    callbacks=callbacks_ae
)

print("Autoencoder training complete.")


# Build encoder model: from input to bottleneck "encoded_layer"
encoder = Model(inputs=autoencoder.input,
                outputs=autoencoder.get_layer("encoded_layer").output,
                name="encoder")

encoder.summary()

# build classifier on top of encoder
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten

encoded_output = encoder.output

# Option 1: flatten features
x = Flatten()(encoded_output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
classifier_output = Dense(5, activation='softmax')(x)

ae_classifier = Model(inputs=encoder.input, outputs=classifier_output, name="ae_classifier")

ae_classifier.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

ae_classifier.summary()


callbacks_cls = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

print("Training classifier on top of pretrained encoder...")

history_cls = ae_classifier.fit(
    train_classifier_gen,
    validation_data=val_classifier_gen,
    epochs=20,
    callbacks=callbacks_cls
)

print("Classifier training complete.")


val_loss_cls, val_acc_cls = ae_classifier.evaluate(val_classifier_gen)
print("Autoencoder-based classifier validation accuracy:", val_acc_cls)
print("Autoencoder-based classifier validation loss:", val_loss_cls)


def plot_history_pair(history_ae, history_cls):
    plt.figure(figsize=(14,6))

    # Autoencoder loss
    plt.subplot(1,2,1)
    plt.plot(history_ae.history["loss"], label="AE Train Loss")
    plt.plot(history_ae.history["val_loss"], label="AE Val Loss", linestyle="--")
    plt.title("Autoencoder Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()

    # Classifier accuracy
    plt.subplot(1,2,2)
    plt.plot(history_cls.history["accuracy"], label="Classifier Train Acc")
    plt.plot(history_cls.history["val_accuracy"], label="Classifier Val Acc", linestyle="--")
    plt.title("AE-based Classifier Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

print("Plotting autoencoder + classifier learning curves...")
plot_history_pair(history_ae, history_cls)


print("Predicting test set with AE-based classifier...")

test_preds_cls = ae_classifier.predict(test_generator)
test_labels_cls = np.argmax(test_preds_cls, axis=1)

submission_ae_df = pd.DataFrame({
    "id_code": test_df["id_code"],
    "diagnosis": test_labels_cls
})

submission_ae_df.to_csv("submission_ae.csv", index=False)

print("submission_ae.csv saved!")
submission_ae_df.head()


submission_ae_df

