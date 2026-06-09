# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# CIFAR-10 | ResNet-18 (scratch) + Kaggle submission
import os, math, random, glob, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import AdamW

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

SEED = 1337
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)



# Paths / constants (Kaggle)
DATA_DIR = "/kaggle/input/cifar10-object-recognition-in-images-zip-file"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sampleSubmission.csv")

# Typical locations of test images in this dataset
TEST_DIR_CANDIDATES = [
    os.path.join(DATA_DIR, "train_test"),
    os.path.join(DATA_DIR, "train_test", "test"),
    os.path.join(DATA_DIR, "train_test", "test", "test"),
    "test", "/kaggle/working/test",  # in case you extracted elsewhere
]

CKPT_PATH = "/kaggle/working/best_model_loss.h5"
print("Sample sub exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Checkpoint target:", CKPT_PATH)



# Load CIFAR-10 (for training/validation)
(x_train, y_train), (x_val, y_val) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_val   = x_val.astype("float32")/255.0
y_train = to_categorical(y_train, 10)
y_val   = to_categorical(y_val, 10)
print("Train:", x_train.shape, y_train.shape, "| Val:", x_val.shape, y_val.shape)



# ResNet-18 style building blocks + model (random init)
def _conv(x, f, s=1):
    y = layers.Conv2D(f, 3, strides=s, padding="same",
                      use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    y = layers.BatchNormalization()(y)
    return layers.Activation("relu")(y)

def _resblock(x, f, down=False):
    s = 2 if down else 1
    y = _conv(x, f, s)
    y = _conv(y, f, 1)
    if down or x.shape[-1] != f:
        x = layers.Conv2D(f, 1, strides=s, padding="same",
                          use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
    z = layers.add([x, y])
    return layers.Activation("relu")(z)

def build_resnet18():
    inp = layers.Input((32, 32, 3))
    x = _conv(inp, 64)
    x = _resblock(x, 64); x = _resblock(x, 64)
    x = _resblock(x, 128, down=True); x = _resblock(x, 128)
    x = _resblock(x, 256, down=True); x = _resblock(x, 256)
    x = _resblock(x, 512, down=True); x = _resblock(x, 512)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return models.Model(inp, out)

_tmp = build_resnet18()
print("âœ… Built ResNet-18 | layers:", len(_tmp.layers), "| params:", _tmp.count_params())
_tmp.summary(); del _tmp



# Instantiate + compile (AdamW)
model = build_resnet18()
model.compile(optimizer=AdamW(learning_rate=3e-4, weight_decay=1e-5, clipnorm=1.0),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
print("âœ… Compiled.")



# Augmentation + callbacks
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)
datagen.fit(x_train)

def cosine_lr(epoch, total=60, base=3e-4):
    return 0.5*base*(1+math.cos(math.pi*epoch/total))

checkpoint_cb = ModelCheckpoint(CKPT_PATH, monitor="val_loss", save_best_only=True, verbose=1)
earlystop_cb  = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1)
lrsched_cb    = LearningRateScheduler(lambda e: cosine_lr(e))

BATCH_SIZE = 128
EPOCHS     = 60
print("âœ… Callbacks ready. Saving best to:", CKPT_PATH)



# Train (skip if best checkpoint already present)
if not os.path.exists(CKPT_PATH):
    print("No checkpoint found â†’ training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_val, y_val),
        steps_per_epoch=len(x_train)//BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb, lrsched_cb],
        verbose=1
    )
else:
    print("Checkpoint exists â†’ skipping training.")



# Cell 7C â€” Force a fresh 60-epoch log (separate model; checkpoint untouched)
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import AdamW
import math

model_log = build_resnet18()
model_log.compile(optimizer=AdamW(3e-4, weight_decay=1e-5, clipnorm=1.0),
                  loss="categorical_crossentropy", metrics=["accuracy"])

def cosine_lr(e, total=60, base=3e-4):
    return 0.5 * base * (1 + math.cos(math.pi * e / total))

hist60 = model_log.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(x_val, y_val),
    steps_per_epoch=len(x_train)//BATCH_SIZE,
    epochs=60,
    callbacks=[LearningRateScheduler(lambda e: cosine_lr(e))],
    verbose=1
)

for e, (a, va) in enumerate(zip(hist60.history['accuracy'], hist60.history['val_accuracy']), start=1):
    print(f"Epoch {e:02d}/60 - accuracy: {a:.4f} - val_accuracy: {va:.4f}")



# Load best model (robust)
def load_best():
    for p in [CKPT_PATH, "best_model_loss.h5", "/kaggle/working/best_model.h5"]:
        if os.path.exists(p):
            print("Loading:", p); return tf.keras.models.load_model(p)
    sm = glob.glob("/kaggle/working/**/saved_model.pb", recursive=True)
    if sm:
        mdir = os.path.dirname(sm[0])
        print("Loading SavedModel from:", mdir)
        return tf.keras.models.load_model(mdir)
    raise FileNotFoundError("No best model found; (re)run training.")

best_model = load_best()
print("âœ… Best model loaded.")



# FAST Cell â€” index test images once, then create the tf.data pipeline

import os, glob, tensorflow as tf, pandas as pd, numpy as np

# 1) Load IDs
assert os.path.exists(SAMPLE_SUB_PATH), "sampleSubmission.csv missing!"
sub_df  = pd.read_csv(SAMPLE_SUB_PATH)
id_list = sub_df["id"].tolist()
print("IDs to predict:", len(id_list))

# 2) Build an index {id -> path} by scanning folders once (much faster)
index = {}
for base in TEST_DIR_CANDIDATES:
    if not os.path.isdir(base): 
        continue
    for ext in ("png","jpg","jpeg"):
        for p in glob.glob(os.path.join(base, f"*.{ext}")):
            name = os.path.splitext(os.path.basename(p))[0]
            if not name.isdigit(): 
                continue
            img_id = int(name)
            if img_id not in index:     # first hit wins
                index[img_id] = p
print("Indexed files:", len(index))

# 3) Build the paths aligned with Kaggle ID order
missing = [i for i in id_list if i not in index]
assert not missing, f"Missing {len(missing)} ids, e.g. {missing[:5]}"
paths = [index[i] for i in id_list]
print("âœ… Paths resolved:", len(paths), "| example:", paths[0])

# 4) tf.data pipeline for inference
def _load(path):
    b = tf.io.read_file(path)
    x = tf.io.decode_image(b, channels=3, expand_animations=False)
    x = tf.image.convert_image_dtype(x, tf.float32)
    x = tf.image.resize(x, [32, 32])
    return x

BATCH_PRED = 1024
ds = tf.data.Dataset.from_tensor_slices(paths).map(_load, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_PRED).prefetch(tf.data.AUTOTUNE)
print("âœ… Test dataset ready.")



# Predict â†’ build submission.csv
probs = best_model.predict(ds, verbose=1)
pred  = np.argmax(probs, axis=1)

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
labels  = [classes[i] for i in pred]

submission = pd.DataFrame({"id": id_list, "label": labels})
submission.to_csv("submission.csv", index=False)

print("ğŸ“� Saved submission.csv")
print(submission.head())
print("Shape:", submission.shape, "| Unique IDs?:", submission['id'].is_unique)
print("NaNs:", submission.isna().sum().to_dict())
print("Label distribution (top):")
print(submission["label"].value_counts().head(10))


