!pip -q install py7zr


import os, math, glob, numpy as np, tensorflow as tf, pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility & performance
SEED = 1337
tf.keras.utils.set_random_seed(SEED)
AUTO = tf.data.AUTOTUNE
tf.config.optimizer.set_jit(False) 

# Hyperparameters
BATCH_TRAIN = 64
BATCH_TEST  = 64
EPOCHS      = 100
LR          = 1e-2
MOMENTUM    = 0.9
WEIGHT_DECAY = 1e-6  # L2 via kernel_regularizer
NUM_CLASSES = 10
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']



(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


train_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
    layers.RandomZoom(0.05, 0.05, fill_mode="reflect"),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
])

def preprocess(x, y=None):
    x = tf.cast(x, tf.float32) / 255.0
    return (x, y) if y is not None else x

def make_ds(x, y, train=False, batch=100):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        ds = ds.shuffle(10000, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda a,b: (train_aug(a), b), num_parallel_calls=AUTO)
    ds = ds.map(lambda a,b: (preprocess(a), b), num_parallel_calls=AUTO)
    ds = ds.batch(batch).prefetch(AUTO)
    return ds

train_ds = make_ds(x_train, y_train, train=True,  batch=BATCH_TRAIN)
test_ds  = make_ds(x_test,  y_test,  train=False, batch=BATCH_TEST)  # used as validation proxy for Kaggle



USE_CUTOUT = False  # set True for a small boost

def cutout(img, length=8):
    H = tf.shape(img)[0]; W = tf.shape(img)[1]
    y = tf.random.uniform([], 0, H, dtype=tf.int32)
    x = tf.random.uniform([], 0, W, dtype=tf.int32)
    y1 = tf.clip_by_value(y - length//2, 0, H); y2 = tf.clip_by_value(y + length//2, 0, H)
    x1 = tf.clip_by_value(x - length//2, 0, W); x2 = tf.clip_by_value(x + length//2, 0, W)
    mask = tf.ones([y2-y1, x2-x1, 3], dtype=img.dtype)
    paddings = [[y1, H - y2], [x1, W - x2], [0, 0]]
    mask = tf.pad(mask, paddings, constant_values=0)
    return img * (1 - mask)

if USE_CUTOUT:
    def _train_map(a, b):
        a = cutout(a, length=8)
        return a, b
    train_ds = train_ds.map(_train_map, num_parallel_calls=AUTO).prefetch(AUTO)



L2 = keras.regularizers.l2(WEIGHT_DECAY)

def conv_bn_relu(x, filters, k=3, bn=False):
    x = layers.Conv2D(filters, k, padding="same", use_bias=not bn,
                      kernel_regularizer=L2)(x)
    if bn:
        x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

inputs = keras.Input(shape=(32, 32, 3))
x = inputs

# Block 1: 3 convs -> MaxPool
x = conv_bn_relu(x, 64*2, bn=True)   # conv1 + BN
x = conv_bn_relu(x, 64*2)            # conv2
x = conv_bn_relu(x, 64*2)            # conv3
x = layers.MaxPooling2D(2)(x)

# Block 2: 3 convs -> MaxPool -> Dropout
x = conv_bn_relu(x, 128*2, bn=True)  # conv4 + BN
x = conv_bn_relu(x, 128*2)           # conv5
x = conv_bn_relu(x, 128*2)           # conv6
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.2)(x)

# Block 3: 3 convs -> MaxPool -> Dropout
x = conv_bn_relu(x, 256*2, bn=True)  # conv7 + BN
x = conv_bn_relu(x, 256*2)           # conv8
x = conv_bn_relu(x, 256*2)           # conv9
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.2)(x)

# Classifier: 4*4*512 = 8192 -> 8192 -> 4096 -> 10
x = layers.Flatten()(x)
x = layers.Dense(4096*2, activation="relu", kernel_regularizer=L2)(x)
x = layers.Dense(2048*2, activation="relu", kernel_regularizer=L2)(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()



opt = keras.optimizers.SGD(learning_rate=LR, momentum=MOMENTUM, nesterov=False)

model.compile(
    optimizer=opt,
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
    metrics=["accuracy"]
)



ckpt = keras.callbacks.ModelCheckpoint(
    filepath="adv_best.weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)
plateau = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=10, min_lr=1e-5, verbose=1
)
es = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=20, restore_best_weights=True
)
log = keras.callbacks.CSVLogger("adv_training_log.csv")



history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[ckpt, plateau, es, log]
)
print("Final val_acc (official test):", round(float(history.history["val_accuracy"][-1]), 4))



loss, acc = model.evaluate(test_ds, verbose=0)
print("Official CIFAR-10 test accuracy (Kaggle-score estimate):", round(float(acc), 4))



import py7zr, os, glob

input_dir = "/kaggle/input/cifar-10"
test_7z = os.path.join(input_dir, "test.7z")

if os.path.exists(test_7z):
    with py7zr.SevenZipFile(test_7z, mode='r') as z:
        z.extractall(path=".")
    print("Extracted test images:", len(glob.glob("test/*.png")))
else:
    print("test.7z not found at", test_7z)



from PIL import Image

def load_png(fp):
    im = Image.open(fp).convert("RGB")
    # Should already be 32x32; resize defensively just in case
    im = im.resize((32,32), resample=Image.BILINEAR)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

test_files = sorted(glob.glob("test/*.png"), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

BATCH = 1024
pred_labels = []
for i in range(0, len(test_files), BATCH):
    batch_files = test_files[i:i+BATCH]
    batch = np.stack([load_png(f) for f in batch_files], axis=0)
    probs = model.predict(batch, verbose=0)
    idx  = np.argmax(probs, axis=1)
    pred_labels.extend([CLASS_NAMES[j] for j in idx])

ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in test_files]
sub = pd.DataFrame({"id": ids, "label": pred_labels})
sub.to_csv("submission.csv", index=False)
sub.head()



# Check format and label set
assert set(sub.columns)=={"id","label"}
assert set(sub["label"].unique()).issubset(set(CLASSES))
assert sub["id"].min()==1 and sub["id"].max()==300000 and len(sub)==300000
print("Submission looks well-formed:", sub.shape, "rows")


