!pip install py7zr --quiet


import os, math, random, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import AdamW
import tensorflow_probability as tfp

warnings.filterwarnings("ignore")
print("TensorFlow:", tf.__version__)
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATA_PATH = "/kaggle/input/cifar10-object-recognition-in-images-zip-file"
SUBMISSION_TEMPLATE = os.path.join(DATA_PATH, "sampleSubmission.csv")
BEST_MODEL_PATH = "/kaggle/working/best_custom_rescnn.weights.h5"


(x_train, y_train), (x_val, y_val) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)


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
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(y)
    y = layers.BatchNormalization()(y)

    if downsample or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                 use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.add([shortcut, y])
    return layers.ReLU()(out)

def build_custom_rescnn(input_shape=(32, 32, 3), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = conv_bn_relu(inputs, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

model = build_custom_rescnn()
model.summary()


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps, min_lr=1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.min_lr + (self.base_lr - self.min_lr) * (step / self.warmup_steps)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + tf.cos(math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        )
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: cosine_lr)

EPOCHS = 400
BATCH_SIZE = 128
steps_per_epoch = x_train.shape[0] // BATCH_SIZE
total_steps = EPOCHS * steps_per_epoch
warmup_steps = int(0.1 * total_steps)

lr_schedule = WarmupCosine(base_lr=3e-4, total_steps=total_steps, warmup_steps=warmup_steps)
optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


AUTO = tf.data.AUTOTUNE

# --- Safe Cutout ---
def cutout(image, size=8):
    h, w, c = image.shape
    y = tf.random.uniform([], 0, h, tf.int32)
    x = tf.random.uniform([], 0, w, tf.int32)
    y1 = tf.clip_by_value(y - size // 2, 0, h)
    y2 = tf.clip_by_value(y + size // 2, 0, h)
    x1 = tf.clip_by_value(x - size // 2, 0, w)
    x2 = tf.clip_by_value(x + size // 2, 0, w)

    # Create mask by cropping and padding
    mask = tf.pad(
        tf.zeros((y2 - y1, x2 - x1, c)),
        [[y1, h - y2], [x1, w - x2], [0, 0]],
        constant_values=1.0
    )
    return image * mask

# --- Safe CutMix ---
def cutmix(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, index)
    shuffled_labels = tf.gather(labels, index)

    # Sample lambda (convert to float32 for TensorFlow math)
    lam = tf.cast(tfp.distributions.Beta(alpha, alpha).sample([]), tf.float32)

    r_x = tf.random.uniform([], 0, 32, tf.int32)
    r_y = tf.random.uniform([], 0, 32, tf.int32)
    r_w = tf.cast(32 * tf.sqrt(1 - lam), tf.int32)
    r_h = tf.cast(32 * tf.sqrt(1 - lam), tf.int32)
    x1 = tf.clip_by_value(r_x - r_w // 2, 0, 32)
    y1 = tf.clip_by_value(r_y - r_h // 2, 0, 32)
    x2 = tf.clip_by_value(r_x + r_w // 2, 0, 32)
    y2 = tf.clip_by_value(r_y + r_h // 2, 0, 32)

    pad_left = x1
    pad_top = y1
    pad_right = 32 - x2
    pad_bottom = 32 - y2

    patch = shuffled_images[:, y1:y2, x1:x2, :]
    patch = tf.pad(patch, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    mask = tf.cast(tf.not_equal(patch, 0.0), tf.float32)

    images = images * (1 - mask) + patch
    labels = lam * tf.cast(labels, tf.float32) + (1 - lam) * tf.cast(shuffled_labels, tf.float32)
    return images, labels

# --- Augmentation pipeline ---
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = cutout(image)
    return image, label
    
def augment_batch(images, labels):
    images, labels = cutmix(images, labels)
    return images, labels
# --- Dataset build ---
train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000)
            .map(augment, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .map(augment_batch, num_parallel_calls=AUTO)
            .prefetch(AUTO))

val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
          .batch(BATCH_SIZE)
          .prefetch(AUTO))



ckpt = keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
    save_weights_only=True 
)
early = EarlyStopping(monitor="val_accuracy", patience=50,
                      restore_best_weights=True, verbose=1)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt, early],
    verbose=1
)


val_acc = max(history.history["val_accuracy"])
print(f"Best Validation Accuracy: {val_acc:.4f}")

if val_acc >= 0.942:
    print("ğŸ�† Expected Grade: 6/6 (Excellent)")
elif val_acc >= 0.931:
    print("ğŸ�¯ Expected Grade: 5/6 (Great)")
elif val_acc >= 0.921:
    print("âœ… Expected Grade: 4/6 (Strong)")
elif val_acc >= 0.911:
    print("âš™ï¸� Expected Grade: 3/6 (Good)")
elif val_acc >= 0.901:
    print("ğŸ“ˆ Expected Grade: 2/6 (Fair)")
elif val_acc >= 0.88:
    print("ğŸ§© Expected Grade: 1/6 (Passable)")
else:
    print("â�Œ Expected Grade: 0/6 (Needs Improvement)")


import os
import py7zr
import pandas as pd
import tensorflow as tf

print("âœ… Loading best model architecture and weights...")
best_model = build_custom_rescnn()  # same function you used to define your model
best_model.load_weights(BEST_MODEL_PATH)
print("âœ… Weights loaded successfully!")

# --- Step 1: Extract archives if necessary ---
input_dir = "/kaggle/input/cifar-10"
extract_dir = "./extracted_data"

os.makedirs(extract_dir, exist_ok=True)

for fname in ["train.7z", "test.7z"]:
    archive_path = os.path.join(input_dir, fname)
    if os.path.exists(archive_path):
        print(f"ğŸ“¦ Extracting {fname} ...")
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(path=extract_dir)
    else:
        print(f"âš ï¸� Archive not found: {archive_path}")

# --- Step 2: Find test directory automatically ---
possible_test_dirs = [
    os.path.join(extract_dir, "test"),
    os.path.join(extract_dir, "train_test/test"),
    os.path.join(extract_dir, "train_test/test/test"),
]

test_dir = None
for d in possible_test_dirs:
    if os.path.exists(d) and len(os.listdir(d)) > 0:
        test_dir = d
        break

if test_dir is None:
    raise FileNotFoundError("â�Œ Could not find test directory after extraction!")

print(f"âœ… Using test directory: {test_dir}")

# --- Step 3: Locate submission template ---
possible_templates = [
    os.path.join(input_dir, "sampleSubmission.csv"),
    "/kaggle/input/cifar10-object-recognition-in-images-zip-file/sampleSubmission.csv"
]
submission_template = next((p for p in possible_templates if os.path.exists(p)), None)

if submission_template is None:
    raise FileNotFoundError("â�Œ sampleSubmission.csv not found!")
print(f"âœ… Found submission template: {submission_template}")

# --- Step 4: Gather image paths ---
paths = sorted([
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.endswith(".png")
])
if len(paths) == 0:
    raise FileNotFoundError(f"â�Œ No .png files found in {test_dir}")
print(f"âœ… Found {len(paths)} test images.")

# --- Step 5: Preprocess and predict ---
def preprocess_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [32, 32])
    return tf.cast(img, tf.float32) / 255.0

AUTO = tf.data.AUTOTUNE
test_ds = (tf.data.Dataset.from_tensor_slices(paths)
           .map(preprocess_img, num_parallel_calls=AUTO)
           .batch(1024)
           .prefetch(AUTO))

print("ğŸš€ Running inference ...")
preds = best_model.predict(test_ds, verbose=1)
pred_classes = tf.argmax(preds, axis=1).numpy()

# --- Step 6: Prepare submission ---
labels = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

pred_labels = [labels[i] for i in pred_classes]
ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]

submission = pd.DataFrame({"id": ids, "label": pred_labels})
submission.to_csv("submission.csv", index=False)

print("âœ… submission.csv generated successfully!")
print(submission.head())


