# ====================================================
# CIFAR-10 Classification | Custom CNN with Residual Connections
# ====================================================
# This script:
# 1) Loads CIFAR-10 dataset for training/testing
# 2) Constructs a custom CNN with residual connections
# 3) Trains with light augmentation and cosine LR scheduler
# 4) Saves the top-performing model checkpoint
# 5) Loads the optimal model
# 6) Generates predictions for Kaggle submission
# ====================================================

import os, math, random, glob, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import AdamW

warnings.filterwarnings("ignore")

print("TensorFlow Version:", tf.__version__)
print("Detected GPUs:", tf.config.list_physical_devices('GPU'))

# --------------------------
# Fixed random seed for reproducibility
# --------------------------
RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --------------------------
# Paths and filenames
# --------------------------
DATA_FOLDER = "/kaggle/input/cifar10-object-recognition-in-images-zip-file"
SUBMISSION_TEMPLATE_PATH = os.path.join(DATA_FOLDER, "sampleSubmission.csv")

# Candidate folders where test images might be
TEST_FOLDER_CANDIDATES = [
    os.path.join(DATA_FOLDER, "train_test"),
    os.path.join(DATA_FOLDER, "train_test", "test"),
    os.path.join(DATA_FOLDER, "train_test", "test", "test"),
]

BEST_MODEL_PATH = "/kaggle/working/top_model_loss.h5"

# ====================================================
# 1) Load CIFAR-10 dataset
# ====================================================
(x_train_set, y_train_set), (x_val_set, y_val_set) = cifar10.load_data()

x_train_set = x_train_set.astype("float32") / 255.0
x_val_set = x_val_set.astype("float32") / 255.0

y_train_set = to_categorical(y_train_set, 10)
y_val_set = to_categorical(y_val_set, 10)

print("Training Data Shape:", x_train_set.shape, y_train_set.shape)
print("Validation Data Shape:", x_val_set.shape, y_val_set.shape)

# ====================================================
# 2) Define Custom CNN with Residual Connections
# ====================================================
def conv_bn_relu(x, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same",
                      use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def residual_block(x, filters, downsample=False):
    stride = 2 if downsample else 1
    shortcut = x

    y = conv_bn_relu(x, filters, stride=stride)
    y = layers.Conv2D(filters, 3, strides=1, padding="same",
                      use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(y)
    y = layers.BatchNormalization()(y)

    if downsample or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                 use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.add([shortcut, y])
    out = layers.ReLU()(out)
    return out

def create_custom_residual_cnn():
    inp = layers.Input(shape=(32, 32, 3))
    x = conv_bn_relu(inp, 64)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)

    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    return models.Model(inp, outputs)

# Instantiate and compile model
model_variant = create_custom_residual_cnn()
opt = AdamW(learning_rate=3e-4, weight_decay=1e-5, clipnorm=1.0)
model_variant.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])


plot_model(
    model_variant,
    to_file="cnn_architecture.png",
    show_shapes=True,
    show_layer_names=True
)

from IPython.display import Image
Image(filename="cnn_architecture.png")

print("\nModel Summary:")
model_variant.summary()

# ====================================================
# 3) Augmentation pipeline
# ====================================================
augmentation_gen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
augmentation_gen.fit(x_train_set)

# Callbacks
model_save_cb = ModelCheckpoint(
    BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
)
stop_early_cb = EarlyStopping(
    monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
)

def cosine_lr_schedule(epoch_num):
    max_epochs = 60
    base_lr = 3e-4
    return 0.5 * base_lr * (1 + math.cos(math.pi * epoch_num / max_epochs))

lr_cb = LearningRateScheduler(cosine_lr_schedule)

# ====================================================
# 4) Model training
# ====================================================
BATCH_SIZE = 128
MAX_EPOCHS = 60

if not os.path.exists(BEST_MODEL_PATH):
    print("No existing model found, starting training...")
    history_data = model_variant.fit(
        augmentation_gen.flow(x_train_set, y_train_set, batch_size=BATCH_SIZE),
        validation_data=(x_val_set, y_val_set),
        steps_per_epoch=x_train_set.shape[0] // BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[model_save_cb, stop_early_cb, lr_cb],
        verbose=1
    )
else:
    print("Model checkpoint found, skipping training phase.")

# ====================================================
# 5) Load best model
# ====================================================
def load_best_model():
    candidate_paths = [BEST_MODEL_PATH, "top_model_loss.h5", "/kaggle/working/top_model.h5"]
    for path in candidate_paths:
        if os.path.exists(path):
            print("Loading model from path:", path)
            return tf.keras.models.load_model(path, custom_objects={"conv_bn_relu": conv_bn_relu, "residual_block": residual_block})
    saved_model_search = glob.glob("/kaggle/working/**/saved_model.pb", recursive=True)
    if saved_model_search:
        directory = os.path.dirname(saved_model_search[0])
        print("Found saved model in directory:", directory)
        return tf.keras.models.load_model(directory)
    raise FileNotFoundError("No saved model found. Please run training.")

best_model_variant = load_best_model()
print("Model loaded successfully.")

# ====================================================
# 6) Kaggle submission generation
# ====================================================
assert os.path.exists(SUBMISSION_TEMPLATE_PATH), "sampleSubmission.csv not found in dataset!"
sample = pd.read_csv(SUBMISSION_TEMPLATE_PATH)
id_list = sample["id"].tolist()
print("IDs to predict:", len(id_list))

def locate_image_path(img_id):
    for base in TEST_FOLDER_CANDIDATES:
        png_path = os.path.join(base, f"{img_id}.png")
        if os.path.exists(png_path):
            return png_path
        jpg_path = os.path.join(base, f"{img_id}.jpg")
        if os.path.exists(jpg_path):
            return jpg_path
    return None

resolved_paths = []
missing_count = 0
for img_id in id_list:
    path = locate_image_path(img_id)
    if path is None:
        missing_count += 1
    else:
        resolved_paths.append(path)

if missing_count > 0:
    raise FileNotFoundError(f"{missing_count} test images could not be located.")

print(f"Resolved test image paths count: {len(resolved_paths)}")

def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [32, 32])
    return img

BATCH_SIZE_PRED = 1024
dataset_pred = tf.data.Dataset.from_tensor_slices(resolved_paths)
dataset_pred = dataset_pred.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset_pred = dataset_pred.batch(BATCH_SIZE_PRED).prefetch(tf.data.AUTOTUNE)

predictions = best_model_variant.predict(dataset_pred, verbose=1)
pred_indices = np.argmax(predictions, axis=1)

label_names_map = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
pred_labels_final = [label_names_map[i] for i in pred_indices]

submission_result = pd.DataFrame({"id": id_list, "label": pred_labels_final})
submission_result.to_csv("submission.csv", index=False)

print("Submission file generated: submission.csv")
print(submission_result.head())

import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Save CNN diagram as an image
plot_model(
    best_model_variant,
    to_file="cnn_architecture.png",
    show_shapes=True,
    show_layer_names=True
)




from tensorflow.keras.utils import model_to_dot
import graphviz
from IPython.display import Image

dot = model_to_dot(model_variant, show_shapes=True, show_layer_names=True)
graph = graphviz.Source(dot.to_string())
graph.render("cnn_architecture", format="png", cleanup=True)
Image(filename="cnn_architecture.png")


