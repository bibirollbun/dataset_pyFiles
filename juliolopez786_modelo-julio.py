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


import os, random, glob, re
import numpy as np
import pandas as pd
from pathlib import Path, PurePosixPath
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ========= Config =========
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_BASE = 12
EPOCHS_FT = 6
VAL_SPLIT = 0.2

tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ========= Helpers =========
def has_imgs(p: Path) -> bool:
    if not p.exists() or not p.is_dir(): return False
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        if list(p.rglob(ext)): return True
    return False

def unwrap(p: Path) -> Path:
    if p.exists() and p.is_dir():
        subs = [s for s in p.iterdir() if s.is_dir()]
        if len(subs) == 1:
            inner = subs[0]
            if sum(1 for s in inner.iterdir() if s.is_dir()) >= 2:
                return inner
    return p

def find_train_test_dirs():
    train_candidates = [
        Path("/kaggle/input/fotosgrupo/Fotos Grupo"),
        Path("/kaggle/input/fotosgrupo/Fotos_Grupo"),
        Path("/kaggle/input/fotosgrupo/fotos_grupo"),
        Path("/kaggle/input/fotosgrupo"),
    ]
    test_candidates = [
        Path("/kaggle/input/facial-recognition-7th-gen/Fotos Grupo"),
        Path("/kaggle/input/facial-recognition-7th-gen/Fotos_Grupo"),
        Path("/kaggle/input/facial-recognition-7th-gen/fotos_grupo"),
        Path("/kaggle/input/facial-recognition-7th-gen"),
    ]
    train_dir = None
    for ct in train_candidates:
        ct = unwrap(ct)
        if has_imgs(ct): train_dir = ct; break
    test_dir = None
    for ct in test_candidates:
        ct = unwrap(ct)
        if has_imgs(ct): test_dir = ct; break
    return train_dir, test_dir

def gather_images(folder: Path):
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        files += list(folder.rglob(ext))
    return sorted(files)

def build_infer_ds(paths, img_size=IMG_SIZE, batch=BATCH_SIZE):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    def _load(p):
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img, tf.float32))
        return img
    return ds.map(_load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)

def most_common(values):
    return pd.Series(values).mode().iloc[0]

# ========= Locate data =========
TRAIN_DIR, TEST_DIR = find_train_test_dirs()
if TRAIN_DIR is None: raise RuntimeError("No se encontró TRAIN_DIR con imágenes.")
if TEST_DIR is None:  raise RuntimeError("No se encontró TEST_DIR con imágenes.")

# ========= Datasets =========
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels="inferred", label_mode="categorical",
    validation_split=VAL_SPLIT, subset="training", seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels="inferred", label_mode="categorical",
    validation_split=VAL_SPLIT, subset="validation", seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
)
class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1024).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ========= Model (MobileNetV2) + fine-tuning =========
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])

base = tf.keras.applications.MobileNetV2(
    include_top=False, weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg"
)
base.trainable = False

inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
x = data_augmentation(x)
x = base(x, training=False)
x = layers.Dropout(0.4)(x)
out = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inp, out)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
cbs = [
    ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max"),
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_BASE, callbacks=cbs, verbose=1)

base.trainable = True
for layer in base.layers[:-30]: layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=cbs, verbose=1)

# ========= Predict all test images (unique by name) =========
all_paths = gather_images(TEST_DIR)
# deduplicate by lowercase filename to avoid duplicates
seen = {}
for p in all_paths:
    k = p.name.lower()
    if k not in seen: seen[k] = p
uniq_paths = [seen[k] for k in sorted(seen.keys())]

infer_ds_all = build_infer_ds(uniq_paths)
probs_all = model.predict(infer_ds_all, verbose=1)
pred_labels_all = [class_names[i] for i in probs_all.argmax(1)]

# build flexible lookup dicts
name2label = {}
for p, lab in zip(uniq_paths, pred_labels_all):
    rel = p.relative_to(TEST_DIR)
    rel_posix = str(PurePosixPath(rel))
    name2label[p.name] = lab
    name2label[p.name.lower()] = lab
    name2label[rel_posix] = lab
    name2label[rel_posix.lower()] = lab
    stem = p.stem
    name2label[stem] = lab
    name2label[stem.lower()] = lab
    if stem.isdigit():
        name2label[int(stem)] = lab
        name2label[str(int(stem))] = lab

fallback_label = most_common(pred_labels_all)

# ========= Build submission following sample if present =========
root = Path("/kaggle/input")
sample_candidates = list(root.rglob("sample_submission.csv"))
final_df = None
if sample_candidates:
    sample_df = pd.read_csv(sample_candidates[0])
    id_col = sample_df.columns[0]
    label_col = sample_df.columns[1] if len(sample_df.columns) > 1 else "label"
    ids = sample_df[id_col].tolist()

    labels = []
    for v in ids:
        cand = [
            v,
            str(v),
            str(PurePosixPath(str(v))).replace("\\","/"),
            Path(str(v)).name,
            Path(str(v)).name.lower(),
            str(v).lower(),
        ]
        lab = None
        for c in cand:
            if c in name2label:
                lab = name2label[c]; break
        if lab is None: lab = fallback_label
        labels.append(lab)

    final_df = pd.DataFrame({id_col: ids, label_col: labels})
else:
    # fallback: try 3 formats and write all; set main to relpath version
    rel_sorted = sorted([str(PurePosixPath(p.relative_to(TEST_DIR))) for p in uniq_paths])
    if len(rel_sorted) < 50: raise RuntimeError("No hay al menos 50 imágenes únicas en TEST.")
    ids50_rel = rel_sorted[:50]
    idx_map = {str(PurePosixPath(p.relative_to(TEST_DIR))): i for i,p in enumerate(uniq_paths)}
    paths50 = [uniq_paths[idx_map[i]] for i in ids50_rel]
    infer_ds50 = build_infer_ds(paths50)
    probs50 = model.predict(infer_ds50, verbose=1)
    preds50 = [class_names[i] for i in probs50.argmax(1)]

    df_rel = pd.DataFrame({"id": ids50_rel, "label": preds50})
    df_rel.to_csv("/kaggle/working/submission_by_relpath.csv", index=False)

    df_fn  = pd.DataFrame({"id": [p.name for p in paths50], "label": preds50})
    df_fn.to_csv("/kaggle/working/submission_by_filename.csv", index=False)

    df_num = pd.DataFrame({"id": list(range(50)), "label": preds50})
    df_num.to_csv("/kaggle/working/submission_by_numeric.csv", index=False)

    final_df = df_rel.copy()

# ========= Save submission (ensure it is published) =========
final_df.to_csv("/kaggle/working/submission.csv", index=False)

# also copy to current working dir so it appears in Output
import shutil
shutil.copy("/kaggle/working/submission.csv", "submission.csv")

# quick sanity print
print("Saved:", os.path.exists("/kaggle/working/submission.csv"), 
      "rows:", len(final_df), 
      "cols:", list(final_df.columns))



import pandas as pd, numpy as np, os, glob
from pathlib import Path, PurePosixPath
import tensorflow as tf

COMP_ROOT = Path("/kaggle/input/facial-recognition-7th-gen")
TEST_DIRS = [
    COMP_ROOT, 
    COMP_ROOT / "Fotos Grupo", 
    COMP_ROOT / "Fotos_Grupo", 
    COMP_ROOT / "fotos_grupo"
]

def gather_images(root_dirs):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    files = []
    for d in root_dirs:
        if d.exists():
            for e in exts:
                files += list(d.rglob(e))
    return sorted(files)

def build_infer_ds(paths, size=IMG_SIZE, batch=BATCH_SIZE):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    def _load(p):
        x = tf.io.read_file(p)
        x = tf.io.decode_image(x, channels=3, expand_animations=False)
        x = tf.image.resize(x, (size, size))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(x, tf.float32))
        return x
    return ds.map(_load, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)

# 1) IDs esperados (solo dentro de la carpeta de la competencia)
sample_path = next((p for p in (COMP_ROOT.rglob("sample_submission.csv"))), None)
testcsv_path = next((p for p in (COMP_ROOT.rglob("test.csv"))), None)

expected_ids = None
id_col_name = "id"
label_col_name = "label"

if sample_path is not None:
    sample_df = pd.read_csv(sample_path)
    id_col_name = sample_df.columns[0]
    if len(sample_df.columns) > 1:
        label_col_name = sample_df.columns[1]
    expected_ids = sample_df[id_col_name].astype(str).tolist()
elif testcsv_path is not None:
    tdf = pd.read_csv(testcsv_path)
    candidates = [c for c in tdf.columns if c.lower() == "id"]
    if not candidates:
        candidates = [c for c in tdf.columns if any(k in c.lower() for k in ["id","file","image","path","name"])]
    id_col_name = candidates[0]
    expected_ids = tdf[id_col_name].astype(str).tolist()
else:
    raise RuntimeError("No se encontró sample_submission.csv ni test.csv dentro de la carpeta de la competencia.")

# 2) Predicciones para todas las imágenes encontradas en TEST_DIRS
all_imgs = gather_images(TEST_DIRS)
if len(all_imgs) == 0:
    raise RuntimeError("No se encontraron imágenes de test en la carpeta de la competencia.")

# deduplicar por nombre en minúsculas
seen = {}
for p in all_imgs:
    k = p.name.lower()
    if k not in seen:
        seen[k] = p
uniq_paths = [seen[k] for k in sorted(seen.keys())]

infer_ds = build_infer_ds(uniq_paths)
probs = model.predict(infer_ds, verbose=1)
pred_idx = probs.argmax(1)
pred_lbl = [class_names[i] for i in pred_idx]

# diccionarios de búsqueda flexibles
name2label = {}
for p, lab in zip(uniq_paths, pred_lbl):
    rel = None
    try:
        # calcular ruta relativa al COMP_ROOT
        rel = p.relative_to(COMP_ROOT)
    except Exception:
        pass
    keys = set()
    keys.add(p.name)
    keys.add(p.name.lower())
    keys.add(p.stem)
    keys.add(p.stem.lower())
    if rel is not None:
        rp = str(PurePosixPath(rel))
        keys.add(rp)
        keys.add(rp.lower())
    for k in list(keys):
        # también variantes con extensión
        if "." not in str(k):
            keys.add(f"{k}.jpg"); keys.add(f"{k}.jpeg"); keys.add(f"{k}.png")
            keys.add(f"{str(k).lower()}.jpg"); keys.add(f"{str(k).lower()}.jpeg"); keys.add(f"{str(k).lower()}.png")
    for k in keys:
        name2label[str(k)] = lab

fallback = pd.Series(pred_lbl).mode().iloc[0]

# 3) Armar submission EXACTO con los IDs esperados del sample/test.csv
final_labels = []
for x in expected_ids:
    cand = [
        str(x),
        str(PurePosixPath(str(x))).replace("\\","/"),
        Path(str(x)).name,
        Path(str(x)).name.lower(),
        str(x).lower()
    ]
    lab = None
    for c in cand:
        if c in name2label:
            lab = name2label[c]; break
    if lab is None: lab = fallback
    final_labels.append(lab)

submission = pd.DataFrame({id_col_name: expected_ids, label_col_name: final_labels})

# Garantizar 50 filas si el sample las tiene
if sample_path is not None and len(submission) != len(sample_df):
    submission = submission.iloc[:len(sample_df)].copy()

# 4) Guardar en Outputs publicados
out_main = "/kaggle/working/submission.csv"
submission.to_csv(out_main, index=False)

import shutil
shutil.copy(out_main, "submission.csv")  # también en el directorio actual para que aparezca en Output

# Verificación mínima
print("submission guardado:", os.path.exists(out_main), "filas:", len(submission), "cols:", list(submission.columns))


