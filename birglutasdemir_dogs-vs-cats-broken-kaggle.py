import os, sys, glob, random, shutil, math, time, json, zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print("Python :", sys.version)
print("TensorFlow :", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

IN_KAGGLE = Path("/kaggle/input").exists()
if IN_KAGGLE:
    candidates = [p for p in Path("/kaggle/input").glob("*") if p.is_dir()]
    comp_root = None
    for c in candidates:
        if (c/"train.zip").exists() and (c/"test1.zip").exists():
            comp_root = c; break
    if comp_root is None:
        raise FileNotFoundError("Couldn't find train.zip & test1.zip under /kaggle/input. Make sure you added the 'Dogs vs Cats' competition on the right panel.")
    DATA_DIR = Path("/kaggle/working/data/dogs-vs-cats")
else:
    DATA_DIR = Path("data/dogs-vs-cats")
    comp_root = None 

RAW_TRAIN_DIR = DATA_DIR / "train"
RAW_TEST_DIR  = DATA_DIR / "test1"
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CLEAR_OLD = True

def _rm_children(folder: Path):
    """Delete all children of a folder (but keep the folder)."""
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return
    for p in folder.iterdir():
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")

def _extract(zip_path: Path, out_dir: Path, clear_old: bool = CLEAR_OLD):
    out_dir.mkdir(parents=True, exist_ok=True)
    if clear_old:
        print(f"Clearing old contents under: {out_dir}")
        _rm_children(out_dir)

    print(f"Extracting {zip_path.name} -> {out_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)

    inner_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    if inner_dirs and not any(out_dir.glob("*.jpg")):
        for d in inner_dirs:
            for f in d.rglob("*.jpg"):
                dest = out_dir / f.name
                if dest.exists():
                    dest = out_dir / f"{d.name}__{f.name}"
                f.replace(dest)
            try: shutil.rmtree(d)
            except Exception: pass

if IN_KAGGLE:
    _extract(comp_root / "train.zip", RAW_TRAIN_DIR)
    _extract(comp_root / "test1.zip", RAW_TEST_DIR)

if CLEAR_OLD:
    CACHE_DIR = Path("cache")
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.cache"):
            try: f.unlink()
            except Exception as e: print(f"[WARN] Could not delete cache {f}: {e}")

n_train = len(list(RAW_TRAIN_DIR.glob("*.jpg")))
n_test  = len(list(RAW_TEST_DIR.glob("*.jpg")))
assert n_train > 0, f"Beklenen klasÃ¶r yok veya boÅŸ: {RAW_TRAIN_DIR}. Kaggle 'train' iÃ§eriÄŸini buraya Ã§Ä±karÄ±n."
assert n_test  > 0, f"Beklenen klasÃ¶r yok veya boÅŸ: {RAW_TEST_DIR}. Kaggle 'test1' iÃ§eriÄŸini buraya Ã§Ä±karÄ±n."
print("Toplam eÄŸitim gÃ¶rÃ¼ntÃ¼sÃ¼:", n_train)
print("Toplam test gÃ¶rÃ¼ntÃ¼sÃ¼  :", n_test)

IMG_SIZE   = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BINARY = True
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


all_files = sorted(glob.glob(str(RAW_TRAIN_DIR / '*.jpg')))
labels = np.array([1 if Path(f).name.startswith('dog') else 0 for f in all_files], dtype=np.int32)
class_names = ['cat', 'dog']
print("Ã–rnek dosya:", Path(all_files[0]).name, "â†’ etiket:", class_names[labels[0]])

train_files, val_files, y_train, y_val = train_test_split(
    all_files, labels, test_size=0.2, random_state=SEED, stratify=labels
)
print(f"EÄŸitim: {len(train_files)}, DoÄŸrulama: {len(val_files)}")

# Decode/resize/scale 
RESCALE = layers.Rescaling(1./255)

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = RESCALE(img)  # float32 in [0,1]
    return img, tf.cast(label, tf.int32)

# Augmentasyon 
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0), name="clip01")
], name="data_augmentation")

opts = tf.data.Options()
opts.experimental_deterministic = True

CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CACHE = str(CACHE_DIR / "train.cache")
VAL_CACHE   = str(CACHE_DIR / "val.cache")

train_base = (tf.data.Dataset.from_tensor_slices((train_files, y_train))
              .map(load_image, num_parallel_calls=AUTOTUNE)
              .with_options(opts)
              .cache(TRAIN_CACHE))

val_ds = (tf.data.Dataset.from_tensor_slices((val_files, y_val))
          .map(load_image, num_parallel_calls=AUTOTUNE)
          .with_options(opts)
          .cache(VAL_CACHE)
          .batch(32)
          .prefetch(AUTOTUNE))
train_ds = (train_base
            .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
            .map(lambda x, y: (data_augmentation(x, training=True), y),
                 num_parallel_calls=AUTOTUNE)
            .batch(32)
            .prefetch(AUTOTUNE))

print("Disk cache etkin. Cache dosyalarÄ±:\n ", TRAIN_CACHE, "\n ", VAL_CACHE)
print("train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("val batches  :", tf.data.experimental.cardinality(val_ds).numpy())

preview_ds = (train_base.take(9)
              .map(lambda x, y: (data_augmentation(x, training=False), y))
              .batch(9))

sample_imgs, sample_labels = next(iter(preview_ds))
imgs_np = np.clip(sample_imgs.numpy(), 0.0, 1.0)

plt.figure(figsize=(8,8))
for i in range(imgs_np.shape[0]):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(imgs_np[i])
    plt.title(class_names[int(sample_labels[i])])
    plt.axis('off')
plt.tight_layout(); plt.show()


def build_simple_cnn(input_shape=INPUT_SHAPE,
                     
                     base_filters=32,
                     kernel_size=3,
                     dropout_rate=0.2,
                     dense_units=128):
    """
    Improved baseline CNN:
    - He-normal initialization (better for ReLU)
    - L2 kernel regularization (reduces overfitting)
    - Batch Normalization after each conv (stabilizes/accelerates training)
    - GlobalAveragePooling2D instead of Flatten (fewer params, CAM-friendly)
    - Keep `last_conv` layer name for Gradâ€‘CAM
    """
    kr = keras.regularizers.l2(5e-5) # noise
    ki = keras.initializers.HeNormal() #? standra

    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(base_filters, kernel_size, padding='same', use_bias=False,
                      kernel_initializer=ki, kernel_regularizer=kr, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D(name='pool1')(x)

    # Block 2
    x = layers.Conv2D(base_filters*2, kernel_size, padding='same', use_bias=False,
                      kernel_initializer=ki, kernel_regularizer=kr, name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu', name='relu2')(x)
    x = layers.MaxPooling2D(name='pool2')(x)

    # Block 3 (keep name for Gradâ€‘CAM)
    x = layers.Conv2D(base_filters*4, kernel_size, padding='same', use_bias=False,
                      kernel_initializer=ki, kernel_regularizer=kr, name='last_conv')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('relu', name='relu3')(x)
    x = layers.MaxPooling2D(name='pool3')(x)

    # Head: GAP -> Dense -> Dropout
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(dropout_rate, name='dropout_head')(x)
    x = layers.Dense(dense_units, activation='relu', kernel_initializer=ki,
                     kernel_regularizer=kr, name='dense')(x)
    x = layers.Dropout(dropout_rate, name='dropout_out')(x)

    if BINARY:
        outputs = layers.Dense(1, activation='sigmoid', name='out')(x)
    else:
        outputs = layers.Dense(len(class_names), activation='softmax', name='out')(x)

    model = keras.Model(inputs, outputs, name='cnn_baseline_v2')
    return model

model = build_simple_cnn()
model.summary()



LR = 5e-4
EPOCHS = 25

loss = keras.losses.BinaryCrossentropy(label_smoothing=0.05)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=loss,
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
    keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    keras.callbacks.ModelCheckpoint(filepath="models/cnn_baseline.keras",
                                    monitor="val_accuracy", save_best_only=True),
]

from math import ceil
train_steps = ceil(len(train_files) / BATCH_SIZE)
val_steps   = ceil(len(val_files)   / BATCH_SIZE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1
)




def plot_training_curves(history):
    hist = history.history
    epochs = range(1, len(hist['loss'])+1)

    plt.figure(figsize=(12,4))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, hist['loss'], label='train')
    plt.plot(epochs, hist['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, hist['accuracy'], label='train')
    plt.plot(epochs, hist['val_accuracy'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_curves(history)



y_true = []
y_pred = []

for imgs, labels in val_ds:
    probs = model.predict(imgs, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    y_true.extend(labels.numpy().astype(int).tolist())
    y_pred.extend(preds.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation='nearest')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
       xticklabels=class_names, yticklabels=class_names,
       xlabel='Predicted label', ylabel='True label', title='Confusion Matrix')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center")
plt.tight_layout()
plt.show()


def get_img_array(path, size=(IMG_SIZE, IMG_SIZE)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32)/255.0
    return img.numpy()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='last_conv'):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(img_array, axis=0), training=False)
        if BINARY:
            loss = predictions[:, 0]
        else:
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]              # (H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))         # (C,)

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def show_gradcam(image_path, model, last_conv_layer_name='last_conv'):
    img = get_img_array(image_path, size=(IMG_SIZE, IMG_SIZE))
    heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name)

    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img)
    ax1.set_title("Girdi GÃ¶rÃ¼ntÃ¼")
    ax1.axis('off')

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(img)
    ax2.imshow(heatmap, cmap='jet', alpha=0.4)
    ax2.set_title("Gradâ€‘CAM IsÄ± HaritasÄ±")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

example_path = val_files[89]
print("Ã–rnek:", Path(example_path).name)
show_gradcam(example_path, model, last_conv_layer_name='last_conv')


USE_TL = True

if USE_TL:
    IMG_TL = 224
    tl_input = layers.Input(shape=(IMG_TL, IMG_TL, 3))
    base = keras.applications.MobileNetV2(
        input_tensor=tl_input, include_top=False, weights='imagenet')
    base.trainable = False

    tl_model = keras.Sequential([
        layers.Resizing(IMG_TL, IMG_TL),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name="mobilenetv2_transfer")

    tl_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss='binary_crossentropy', metrics=['accuracy'])

    tl_history = tl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(MODELS_DIR/'mobilenetv2_tl.h5'),
                                            monitor='val_accuracy', save_best_only=True)
        ]
    )

    plot_training_curves(tl_history)


import keras_tuner as kt

def build_model(hp):
    base_filters = hp.Choice("base_filters", [16, 32, 64])
    kernel_size  = hp.Choice("kernel_size",  [3, 5])
    dropout_rate = hp.Choice("dropout_rate", [0.3, 0.5])
    dense_units  = hp.Choice("dense_units",  [64, 128, 256])
    lr           = hp.Choice("lr",           [1e-3, 5e-4, 1e-4])

    m = build_simple_cnn(base_filters=base_filters,
                         kernel_size=kernel_size,
                         dropout_rate=dropout_rate,
                         dense_units=dense_units)
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss='binary_crossentropy', metrics=['accuracy'])
    return m

HPO_TRAIN_SAMPLES = 800
HPO_VAL_SAMPLES   = 200
HPO_BATCH_SIZE    = 32

tr_small = (tf.data.Dataset.from_tensor_slices((train_files, y_train))
            .shuffle(2048, seed=SEED)
            .map(load_image, num_parallel_calls=AUTOTUNE)
            .map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            .take(HPO_TRAIN_SAMPLES)
            .batch(HPO_BATCH_SIZE)
            .repeat()
            .prefetch(AUTOTUNE))

va_small = (tf.data.Dataset.from_tensor_slices((val_files, y_val))
            .map(load_image, num_parallel_calls=AUTOTUNE)
            .take(HPO_VAL_SAMPLES)
            .batch(HPO_BATCH_SIZE)
            .repeat()
            .prefetch(AUTOTUNE))

STEPS_PER_EPOCH    = HPO_TRAIN_SAMPLES // HPO_BATCH_SIZE
VAL_STEPS_PER_EPOCH = HPO_VAL_SAMPLES // HPO_BATCH_SIZE

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=6,        
    factor=3,
    directory=str(MODELS_DIR / 'kt'),
    project_name='dogs_vs_cats'
)

early = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
print(f"[KT] Arama baÅŸlÄ±yorâ€¦ steps/epoch={STEPS_PER_EPOCH}, val_steps={VAL_STEPS_PER_EPOCH}")
tuner.search(
    tr_small,
    validation_data=va_small,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS_PER_EPOCH,
    epochs=6,
    callbacks=[early],
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_cfg = {k: best_hp.get(k) for k in ["base_filters","kernel_size","dropout_rate","dense_units","lr"]}
print("[KT] En iyi hiperparametreler:", best_cfg)

(MODELS_DIR / 'kt').mkdir(parents=True, exist_ok=True)
with open(MODELS_DIR / 'kt' / 'best_hparams.json', 'w') as f:
    json.dump(best_cfg, f, indent=2)
print(f"[KT] Hiperparametreler kaydedildi: {MODELS_DIR / 'kt' / 'best_hparams.json'}")

best_model = tuner.hypermodel.build(best_hp)
best_model.summary()




from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score
)

def collect_preds(model, dataset, num_classes=None, max_batches=None):
    """
    dataset .repeat() ile sonsuzsa max_batches=val_steps verin.
    """
    y_true_all, y_prob_all = [], []
    for i, (xb, yb) in enumerate(dataset):
        probs = model.predict(xb, verbose=0)
        y_true_all.append(yb.numpy() if hasattr(yb, "numpy") else yb)
        y_prob_all.append(probs)
        if max_batches is not None and (i + 1) >= max_batches:
            break

    y_true = np.concatenate([np.asarray(t).reshape(-1) for t in y_true_all])
    y_prob = np.concatenate(y_prob_all)

    # SÄ±nÄ±f sayÄ±sÄ±nÄ±/Ã§Ä±kÄ±ÅŸ biÃ§imini Ã§Ä±kar
    if num_classes is None:
        n_out = 1 if y_prob.ndim == 1 else y_prob.shape[-1]
        num_classes = 2 if n_out == 1 else n_out

    # SÄ±nÄ±f tahmini
    if num_classes == 2:
        p1 = y_prob.reshape(-1) if y_prob.ndim > 1 else y_prob
        y_pred = (p1 >= 0.5).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)

    return y_true, y_pred, y_prob, num_classes

# EÄŸer val_ds .repeat() kullanÄ±yorsa ve daha Ã¶nce val_steps hesapladÄ±ysanÄ±z, aÅŸaÄŸÄ±daki Ã§aÄŸrÄ±da max_batches=val_steps kullanÄ±n.
try:
    y_true, y_pred, y_prob, NUM_CLASSES = collect_preds(model, val_ds, max_batches=val_steps)
except NameError:
    # val_steps yoksa normal topla (val_ds sonsuz olmamalÄ±)
    y_true, y_pred, y_prob, NUM_CLASSES = collect_preds(model, val_ds)

# Metrikler
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

auc_val = None
if NUM_CLASSES == 2:
    try:
        p1 = y_prob.reshape(-1) if y_prob.ndim > 1 else y_prob
        auc_val = roc_auc_score(y_true, p1)
    except Exception:
        auc_val = None

print("\n=========== BAÅ�ARI Ã‡IKTISI ===========")
print(f"F1-macro: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
if auc_val is not None:
    print(f"AUC: {auc_val:.4f}")
print("======================================\n")

print("â€” AyrÄ±ntÄ±lÄ± Rapor â€”")
print(classification_report(y_true, y_pred, digits=4))

print("â€” Confusion Matrix â€”")
print(confusion_matrix(y_true, y_pred))

