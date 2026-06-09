%%capture
# Unzip the competition dataset to the Kaggle working directory
!mkdir -p /kaggle/working/flower_102
!unzip -oq /kaggle/input/oxford-102-flower-pytorch/flower_data.zip -d /kaggle/working/flower_102


from pathlib import Path
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = Path("/kaggle/input/flower-model-tinyml-70k-90-top-1-accuracy/keras/default/2/flower_model/flower_model.h5")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# If the model used only standard Keras layers/losses/metrics, this will work as-is.
# If there were custom objects, you'll need to pass a custom_objects dict.
model = keras.models.load_model(MODEL_PATH, compile=False)

# Print a layer-by-layer summary
model.summary()



from pathlib import Path
import tensorflow as tf
from tensorflow import keras

ROOT_DIR   = Path("/kaggle/working/flower_102/flower_data")
TRAIN_DIR  = ROOT_DIR / "train"
VALID_DIR  = ROOT_DIR / "valid"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 64

def load_split(split_dir: Path,
               img_size=(224,224),
               batch_size=64,
               label_mode="int"):
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode=label_mode,
        color_mode="rgb", 
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
    )

    class_names = ds.class_names
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds, class_names

train_ds, class_names = load_split(TRAIN_DIR, IMG_SIZE, BATCH_SIZE)
valid_ds, _ = load_split(VALID_DIR, IMG_SIZE, BATCH_SIZE)


# %% [code]
#!/usr/bin/env python3
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------
# Core helpers (unchanged)
# ---------------------------
def _take_one_batch(ds):
    """Return (images, labels) numpy arrays from the first batch of a tf.data.Dataset."""
    x, y = next(iter(ds.take(1)))
    return x.numpy(), y.numpy()

def _label_indices(y):
    """
    Convert labels to integer indices:
      - if one-hot (2D), argmax along last axis
      - if already int (1D), return as-is
    """
    y = np.asarray(y)
    if y.ndim == 2:   # one-hot
        return y.argmax(axis=-1)
    elif y.ndim == 1: # int labels
        return y
    else:
        # Handle unexpected shapes defensively (e.g., sparse or ragged)
        return np.squeeze(y)

def _print_stats(name, x, y, y_pred=None):
    """Print dataset stats: shapes, dtypes, min/max for images and labels. Show pred shape if provided."""
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    print(f"[{name}]")
    print(f"  images: shape={x.shape}, dtype={x.dtype}, value_range=[{x_min:.4g}, {x_max:.4g}]")
    print(f"  labels: shape={y.shape}, dtype={y.dtype}, value_range=[{y_min:.4g}, {y_max:.4g}]")
    if y_pred is not None:
        yp = np.asarray(y_pred)
        yp_min, yp_max = float(yp.min()), float(yp.max())
        print(f"  preds : shape={yp.shape}, dtype={yp.dtype}, value_range=[{yp_min:.4g}, {yp_max:.4g}]")

# ---------------------------
# Prediction helpers
# ---------------------------
def _extract_tensor(pred):
    """
    Normalize various model outputs to a numpy array:
      - Tensor -> numpy
      - list/tuple -> take first item
      - dict -> prefer keys like 'logits','predictions','output_0', else first value
    """
    if isinstance(pred, tf.Tensor):
        return pred.numpy()
    if isinstance(pred, (list, tuple)) and pred:
        return _extract_tensor(pred[0])
    if isinstance(pred, dict) and pred:
        for k in ("logits","predictions","probs","outputs","output_0","y_pred"):
            if k in pred:
                v = pred[k]
                return v.numpy() if isinstance(v, tf.Tensor) else np.asarray(v)
        # fallback: first value
        v = next(iter(pred.values()))
        return v.numpy() if isinstance(v, tf.Tensor) else np.asarray(v)
    # final fallback
    return np.asarray(pred)

def _pred_indices(model, x):
    """
    Get predicted class indices from model:
      - If shape (...,1): binary -> (p>0.5).astype(int)
      - Else multiclass -> argmax(-1)
    """
    raw = model(x, training=False)
    y_pred = _extract_tensor(raw)
    y_pred = np.asarray(y_pred)

    while y_pred.ndim >= 3 and 1 in y_pred.shape[1:]:
        y_pred = np.squeeze(y_pred, axis=tuple(i for i,s in enumerate(y_pred.shape[1:], start=1) if s==1))

    if y_pred.ndim == 1:
        return y_pred.astype(int)
    if y_pred.shape[-1] == 1:
        return (y_pred[...,0] > 0.5).astype(int)
    return y_pred.argmax(axis=-1)

# ---------------------------
# Visualization with correctness highlighting
# ---------------------------
def _show_grid_with_correctness(x, y_idx, y_hat_idx, class_names=None, title=""):
    """
    Show a 4x4 grid; title colored green if correct, red if incorrect.
    """
    B = x.shape[0]
    n = min(16, B)
    cols = 4
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = x[i]

        # Display normalization for visualization only
        vmax = img.max()
        img_disp = img / 255.0 if vmax > 1.0 else img
        plt.imshow(img_disp)
        plt.axis("off")

        yt = int(y_idx[i]) if i < len(y_idx) else -1
        yp = int(y_hat_idx[i]) if i < len(y_hat_idx) else -1

        # Derive labels
        def name_of(k):
            if class_names is not None and 0 <= k < len(class_names):
                return class_names[k]
            return str(k)

        correct = (yp == yt)
        color = ("tab:green" if correct else "tab:red")
        title_txt = f"P:{name_of(yp)} | T:{name_of(yt)}"
        plt.title(title_txt, fontsize=10, color=color)

        # Also color the frame for extra clarity
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)

    if title:
        plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

# ---------------------------
# Public API
# ---------------------------
def display_ds_examples_and_eval(ds, model, class_names=None, name="dataset"):
    """
    Prints stats (+pred stats), shows a 4x4 grid with green/red correctness,
    and prints the batch accuracy.
    """
    # One batch
    x, y = _take_one_batch(ds)
    y_idx = _label_indices(y)
    y_hat_idx = _pred_indices(model, x)

    # Stats + batch acc
    _print_stats(name, x, y, y_pred=y_hat_idx)
    batch_acc = float((y_hat_idx[:len(y_idx)] == y_idx[:len(y_hat_idx)]).mean())
    print(f"[{name}] batch accuracy: {batch_acc*100:.2f}%")

    # Grid
    _show_grid_with_correctness(
        x, y_idx, y_hat_idx, class_names,
        title=f"{name}: {x.shape[1]}x{x.shape[2]}x{x.shape[3]}"
    )

def eval_dataset_accuracy(ds, model, max_batches=None, verbose_every=0):
    """
    Iterate the whole dataset (or first `max_batches`), compute accuracy.
    Returns (accuracy_float, n_correct, n_total).
    """
    n_correct = 0
    n_total = 0
    it = ds.as_numpy_iterator()
    for b_idx, (x, y) in enumerate(it):
        if (max_batches is not None) and (b_idx >= max_batches):
            break
        y_idx = _label_indices(y)
        y_hat_idx = _pred_indices(model, x)

        # Align lengths defensively
        m = min(len(y_idx), len(y_hat_idx))
        n_correct += int((y_hat_idx[:m] == y_idx[:m]).sum())
        n_total   += int(m)

        if verbose_every and (b_idx % verbose_every == 0):
            print(f"  batch {b_idx:4d}: running acc={100.0*n_correct/max(n_total,1):.2f}%")

    acc = (n_correct / n_total) if n_total else 0.0
    print(f"[full] accuracy: {acc*100:.4f}%  ({n_correct}/{n_total}) | incorrect: {(1-acc)*100:.4f}%")
    return acc, n_correct, n_total



display_ds_examples_and_eval(train_ds, model, class_names, name="train")
eval_dataset_accuracy(train_ds, model)


display_ds_examples_and_eval(valid_ds, model, class_names, name="valid")
eval_dataset_accuracy(valid_ds, model)


# %% [code]
import csv, os
from pathlib import Path
import numpy as np
import tensorflow as tf

TRAIN_DIR = Path("/kaggle/working/flower_102/flower_data/train")
TEST_DIR  = Path("/kaggle/working/flower_102/flower_data/test")
IMG_SIZE = (224,224)
BATCH_SIZE = 128
SUB_PATH = Path("/kaggle/working/submission.csv")

cn = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
idx2id = np.array([int(n) for n in cn], dtype=np.int64)

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".gif",".webp"}
paths = sorted([p for p in TEST_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
if not paths: raise RuntimeError(f"No images found under: {TEST_DIR}")

def _load_resize(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_image(x, channels=3, expand_animations=False)
    x = tf.image.resize(x, IMG_SIZE, method="bilinear")
    return tf.cast(x, tf.float32)

path_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
test_ds = path_ds.map(_load_resize, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

try: model
except NameError: raise NameError("`model` is not defined.")

def _to_array(o):
    if isinstance(o, tf.Tensor): return o.numpy()
    if isinstance(o, (list, tuple)) and o: return _to_array(o[0])
    if isinstance(o, dict) and o:
        for k in ("logits","predictions","probs","outputs","output_0","y_pred"):
            if k in o: return _to_array(o[k])
        return _to_array(next(iter(o.values())))
    a = np.asarray(o)
    if a.dtype==object and a.size: return _to_array(a[0])
    return a

pred = _to_array(model.predict(test_ds, verbose=0))
if pred.ndim==0: raise RuntimeError(f"Unexpected scalar prediction shape: {pred.shape}")
while pred.ndim>=3 and 1 in pred.shape[1:]:
    pred = np.squeeze(pred, axis=tuple(i for i,s in enumerate(pred.shape[1:], start=1) if s==1))
if pred.ndim==1:
    idx = pred.astype(int)
elif pred.shape[-1]==1:
    idx = (pred[...,0]>0.5).astype(int)
else:
    idx = pred.argmax(-1)

if idx.max() >= len(idx2id): raise RuntimeError(f"Pred index out of range: max {idx.max()} >= {len(idx2id)}")
pred_ids = idx2id[idx]

SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
with SUB_PATH.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["file_name","id"])
    for p, cid in zip(paths, pred_ids):
        w.writerow([p.name, int(cid)])

print(f"[ok] Wrote {len(paths)} rows to {SUB_PATH}")



print(idx2id)


!rm -rf /kaggle/working/flower_102


from zipfile import ZipFile, ZIP_DEFLATED
ZipFile("/kaggle/working/submission.zip","w",ZIP_DEFLATED).write(
    "/kaggle/working/submission.csv", arcname="submission.csv"
)




