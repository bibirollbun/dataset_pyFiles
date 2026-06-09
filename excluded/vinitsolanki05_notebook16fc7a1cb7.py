# ==============================================================
#  Khasi Musical Instrument Classification - Initial Setup (Fixed)
#  Kaggle Notebook Compatible
# ==============================================================

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------
# 1. Paths Setup
# --------------------------------------------------------------
BASE_DIR = "/kaggle/input/musical-instrumemts-sound-classification/Melspectogram_split"

train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")
test_dir = os.path.join(BASE_DIR, "test")

# --------------------------------------------------------------
# 2. Parameters
# --------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# --------------------------------------------------------------
# 3. Load Datasets (Keep one copy raw before caching)
# --------------------------------------------------------------
train_raw = image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_raw = image_dataset_from_directory(
    val_dir,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_raw = image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Extract class names *before* applying cache/prefetch
class_names = train_raw.class_names
print("Classes Detected:", class_names)
print(f"Total Classes: {len(class_names)}")

# --------------------------------------------------------------
# 4. Optimize Dataset for GPU Performance
# --------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_raw.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_raw.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_raw.cache().prefetch(buffer_size=AUTOTUNE)

# --------------------------------------------------------------
# 5. Data Augmentation
# --------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# --------------------------------------------------------------
# 6. Visualize Few Samples
# --------------------------------------------------------------
plt.figure(figsize=(10, 10))
for images, labels in train_raw.take(1):  # use train_raw instead of train_ds
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[tf.argmax(labels[i])])
        plt.axis("off")
plt.show()

# --------------------------------------------------------------
# 7. GPU Check (ignore cuInit warnings if running on CPU)
# --------------------------------------------------------------
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))
if gpus:
    print("✅ GPU detected and ready.")
else:
    print("⚠️ No GPU found — Kaggle may be using CPU runtime. Go to: Settings → Accelerator → GPU")

