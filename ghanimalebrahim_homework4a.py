# CIFAR-10 | ResNet-18 (from scratch) + high-score Kaggle submission
import os, math, random, glob, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import SGD

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

SEED = 1337
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)



# One *single* checkpoint path used by BOTH training and inference
CKPT_PATH = "/kaggle/working/resnet18_best.keras"

# Dataset mount
DATA_DIR = "/kaggle/input/cifar10-object-recognition-in-images-zip-file"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sampleSubmission.csv")

# Likely places for test images (dataset variants differ)
TEST_DIR_CANDIDATES = [
    os.path.join(DATA_DIR, "train_test"),
    os.path.join(DATA_DIR, "train_test", "test"),
    os.path.join(DATA_DIR, "train_test", "test", "test"),
    "test", "/kaggle/working/test"  # in case you extracted elsewhere
]

print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Checkpoint target:", CKPT_PATH)



# Official CIFAR-10 for training/validation (mirrors Kaggle labels)
(x_train, y_train), (x_val, y_val) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_val   = x_val.astype("float32")   / 255.0
y_train = to_categorical(y_train, 10)
y_val   = to_categorical(y_val, 10)
print("Train:", x_train.shape, y_train.shape, "| Val:", x_val.shape, y_val.shape)



# Strong baseline: pad+random-crop, flip, contrast, L2=5e-4, He init (random), Dropout 0.5
reg = regularizers.l2(5e-4)

# Channel-wise normalization (fit on training)
norm = layers.Normalization()
norm.adapt(x_train)  # x_train already in [0,1]

def conv3(x, f, s=1):
    y = layers.Conv2D(f, 3, strides=s, padding="same",
                      kernel_initializer="he_normal",
                      use_bias=False, kernel_regularizer=reg)(x)
    y = layers.BatchNormalization()(y)
    return layers.ReLU()(y)

def resblock(x, f, down=False):
    s = 2 if down else 1
    y = conv3(x, f, s)
    y = conv3(y, f, 1)
    if down or x.shape[-1] != f:
        skip = layers.Conv2D(f, 1, strides=s, padding="same",
                             kernel_initializer="he_normal",
                             use_bias=False, kernel_regularizer=reg)(x)
        skip = layers.BatchNormalization()(skip)
    else:
        skip = x
    z = layers.Add()([y, skip])
    return layers.ReLU()(z)

def build_resnet18_strong():
    inp = layers.Input((32, 32, 3))
    x = layers.ZeroPadding2D(4)(inp)
    x = layers.RandomCrop(32, 32)(x)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomContrast(0.1)(x)
    x = norm(x)

    x = conv3(x, 64)
    x = resblock(x, 64);  x = resblock(x, 64)
    x = resblock(x, 128, down=True); x = resblock(x, 128)
    x = resblock(x, 256, down=True); x = resblock(x, 256)
    x = resblock(x, 512, down=True); x = resblock(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return models.Model(inp, out)

_tmp = build_resnet18_strong()
print("âœ… Built strong ResNet-18 | layers:", len(_tmp.layers), "| params:", _tmp.count_params())
_tmp.summary(); del _tmp



# Train settings that consistently reach â‰¥0.88 on CIFAR-10
EPOCHS = 120
BATCH  = 128
INIT_LR = 0.1  # with SGD this is standard for CIFAR-10

def lr_cosine_warmup(epoch, total=EPOCHS, base=INIT_LR, warmup=5):
    if epoch < warmup:
        return base * (epoch + 1) / warmup
    t = (epoch - warmup) / (total - warmup)
    return 0.5 * base * (1 + math.cos(math.pi * t))

model = build_resnet18_strong()
model.compile(
    optimizer=SGD(learning_rate=INIT_LR, momentum=0.9, nesterov=True),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

callbacks = [
    LearningRateScheduler(lambda e: lr_cosine_warmup(e), verbose=1),
    ModelCheckpoint(CKPT_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
]
print("âœ… Compiled. Training for", EPOCHS, "epochs; best will be saved to:", CKPT_PATH)



# Always train (fresh run). If you want to skip when a good checkpoint exists, guard with os.path.exists.
history = model.fit(
    x_train, y_train,                      # in-graph augmentation handles transforms
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)
print("âœ… Training done. Best saved to:", CKPT_PATH)



# Load exactly the saved file and evaluate on official CIFAR-10 test set
assert os.path.exists(CKPT_PATH), f"Missing: {CKPT_PATH}"
best_model = tf.keras.models.load_model(CKPT_PATH)

(_, _), (x_te, y_te) = cifar10.load_data()
x_te = x_te.astype("float32") / 255.0
y_te = to_categorical(y_te, 10)

offline_acc = best_model.evaluate(x_te, y_te, verbose=0)[1]
print(f"âœ… Offline CIFAR-10 test accuracy: {offline_acc:.4f}")



# Read Kaggle IDs and build a fast {id -> path} index by scanning folders once
assert os.path.exists(SAMPLE_SUB_PATH), "sampleSubmission.csv not found!"
sub_df  = pd.read_csv(SAMPLE_SUB_PATH)
id_list = sub_df["id"].tolist()
print("IDs to predict:", len(id_list))

index = {}
for base in [p for p in TEST_DIR_CANDIDATES if os.path.isdir(p)]:
    for ext in ("png","jpg","jpeg"):
        for p in glob.glob(os.path.join(base, f"*.{ext}")):
            name = os.path.splitext(os.path.basename(p))[0]
            if name.isdigit():
                i = int(name)
                if i not in index:  # first hit wins
                    index[i] = p

missing = [i for i in id_list if i not in index]
assert not missing, f"Missing {len(missing)} ids, e.g. {missing[:5]}"
paths = [index[i] for i in id_list]
print("âœ… Paths resolved:", len(paths), "| example:", paths[0])



# tf.data pipeline â†’ predict class indices
def load_and_preprocess(path):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)   # [0,1]
    img = tf.image.resize(img, [32, 32])
    return img

BATCH_PRED = 1024
ds = tf.data.Dataset.from_tensor_slices(paths)
ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_PRED).prefetch(tf.data.AUTOTUNE)

print("Predicting with:", CKPT_PATH)
probs = best_model.predict(ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)



# Map to label strings, keep original ID order, write CSV
label_names = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]
pred_labels = [label_names[i] for i in pred_idx]

submission = pd.DataFrame({"id": id_list, "label": pred_labels})
submission.to_csv("submission.csv", index=False)

print("ğŸ“� Saved submission.csv")
print("Shape:", submission.shape, "| Unique IDs?:", submission['id'].is_unique)
print(submission.head())
print("Label distribution:")
print(submission["label"].value_counts().head(10))





