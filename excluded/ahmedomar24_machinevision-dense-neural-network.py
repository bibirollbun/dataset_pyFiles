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


# ================================================
# Driver Distraction (State Farm) - Pipeline Bench
# One-technique-per-section + 70/15/15 split
# ================================================

import os, glob, random, gc, math, json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -------------------- Config --------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
CLASS_NAMES = [f"c{i}" for i in range(10)]

IMG_SIZE = 64              # keep small for speed
SCALE_RANGE = "[0,1]"      # "[0,1]" or "[-1,1]"
PER_CLASS_LIMIT = 400      # <= for quick runs; raise if you have time
BATCH = 256
EPOCHS = 25                # early stopping will usually stop earlier
PCA_KEEP = 0.95            # retain 95% variance
SHOW_CM = False            # set True to render confusion matrices

# Pipelines to train (one per rubric section)
PIPELINE_ORDER = [
    "Standard",
    "Lighting",
    "Noise_Gaussian",
    "Texture_LBP",        # <-- enabled
    "Feature_Edges",
    "BackgroundRemoval",
    "MultiScale",
    "Aug_Rotate"
]

# ------------------ Helpers: scaling ------------------
def _scale01(imgf32):
    return np.clip(imgf32, 0.0, 1.0).astype(np.float32)

def _scale_out(x01):
    if SCALE_RANGE == "[-1,1]":
        return (x01 * 2.0 - 1.0).astype(np.float32)
    return x01.astype(np.float32)  # [0,1]

def _resize_rgb(img_bgr, size=IMG_SIZE):
    return cv2.cvtColor(cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)

# ------------------ 1) Standard ------------------
def pp_standard(img_bgr):
    rgb = _resize_rgb(img_bgr)
    x = rgb.astype(np.float32) / 255.0
    return _scale_out(x)

# -------- 2) Lighting (CLAHE + gamma + B/C) -------
def pp_lighting(img_bgr, gamma=1.15, alpha=1.05, beta=0.02, clip=2.0, tile=8):
    rgb = _resize_rgb(img_bgr)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    L2 = clahe.apply(L)
    rgb2 = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    rgb2 = np.power(np.clip(rgb2, 0, 1), gamma).astype(np.float32)   # gamma
    rgb2 = np.clip(alpha * rgb2 + beta, 0.0, 1.0)                    # B/C
    return _scale_out(rgb2)

# -------- 3) Noise reduction (Gaussian) -----------
def pp_noise_gaussian(img_bgr, k=3):
    rgb = _resize_rgb(img_bgr)
    x8 = cv2.GaussianBlur(rgb, (k, k), 0)
    x = x8.astype(np.float32) / 255.0
    # tiny morph open for dot noise
    k3 = np.ones((2,2), np.uint8)
    x8 = (x * 255).astype(np.uint8)
    x8 = cv2.morphologyEx(x8, cv2.MORPH_OPEN, k3)
    return _scale_out(x8.astype(np.float32) / 255.0)

# -------- 4a) Feature: Edge detection (Canny) ----
def pp_feature_edges(img_bgr):
    rgb = _resize_rgb(img_bgr)
    g8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g8, 100, 200).astype(np.float32) / 255.0
    e3 = np.repeat(e[..., None], 3, axis=-1)  # keep 3 channels for fairness
    return _scale_out(e3)

# -------- 4b) Background removal (GrabCut) -------
def pp_bg_grabcut(img_bgr):
    rgb = _resize_rgb(img_bgr)
    h, w = rgb.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    rect = (w//8, h//8, 3*w//4, 3*h//4)
    bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fgmask = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        out = np.zeros_like(rgb, dtype=np.float32)
        out[fgmask] = rgb[fgmask].astype(np.float32) / 255.0
        return _scale_out(out)
    except:
        return pp_standard(img_bgr)

# -------- 4c) Multi-scale (pyramid mean) ---------
def pp_multiscale(img_bgr):
    base = _resize_rgb(img_bgr).astype(np.float32) / 255.0
    s2  = cv2.resize(base, (IMG_SIZE*3//4, IMG_SIZE*3//4), interpolation=cv2.INTER_AREA)
    s2  = cv2.resize(s2, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    s3  = cv2.resize(base, (IMG_SIZE//2, IMG_SIZE//2), interpolation=cv2.INTER_AREA)
    s3  = cv2.resize(s3, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    out = np.clip((base + s2 + s3) / 3.0, 0, 1).astype(np.float32)
    return _scale_out(out)

# -------- 4d) Texture (LBP) — vectorized ----------
def _lbp_simple(gray_u8):
    g = gray_u8.astype(np.float32)
    H, W = g.shape
    c = g[1:-1, 1:-1]
    code = np.zeros((H-2, W-2), dtype=np.uint8)
    # (dy, dx, bit)
    SHIFTS = [(-1,-1,7), (-1,0,6), (-1,1,5), (0,1,4),
              (1,1,3),  (1,0,2),  (1,-1,1), (0,-1,0)]
    for dy, dx, bit in SHIFTS:
        nbr = g[1+dy:H-1+dy, 1+dx:W-1+dx]
        code |= ((nbr >= c).astype(np.uint8) << bit)
    # pad back to HxW (zeros border)
    out = np.zeros((H, W), dtype=np.float32)
    out[1:-1, 1:-1] = code.astype(np.float32)
    # normalize 0..1
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out

def pp_texture_lbp(img_bgr):
    rgb = _resize_rgb(img_bgr)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp01 = _lbp_simple(g)
    e3 = np.repeat(lbp01[..., None], 3, axis=-1)
    return _scale_out(e3)

# -------- 5) Augmentation (Rotation only) --------
def _rotate_only(rgb, max_deg=15):
    h, w = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), np.random.uniform(-max_deg, max_deg), 1.0)
    return cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT101)

def pp_aug_rotate(img_bgr):
    rgb = _resize_rgb(img_bgr).astype(np.float32) / 255.0
    rgb = _rotate_only(rgb)
    return _scale_out(np.clip(rgb, 0, 1))

# ------------ Register pipelines -------------
PIPELINES = {
    "Standard":           pp_standard,
    "Lighting":           pp_lighting,
    "Noise_Gaussian":     pp_noise_gaussian,
    "Texture_LBP":        pp_texture_lbp,   # <-- now registered
    "Feature_Edges":      pp_feature_edges,
    "BackgroundRemoval":  pp_bg_grabcut,
    "MultiScale":         pp_multiscale,
    "Aug_Rotate":         pp_aug_rotate,
}

# -------------- Data loading ----------------
def list_files(root, per_class=None):
    X, y = [], []
    for ci, cname in enumerate(CLASS_NAMES):
        p = os.path.join(root, cname, "*.jpg")
        files = sorted(glob.glob(p))
        if per_class:
            files = files[:per_class]
        X += files
        y += [ci] * len(files)
    return np.array(X), np.array(y, dtype=np.int32)

ALL_FILES, ALL_LABELS = list_files(DATA_DIR, PER_CLASS_LIMIT)
print(f"Loaded paths: {len(ALL_FILES)}")

def load_images(paths, preprocess_fn):
    xs = []
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        xs.append(preprocess_fn(bgr))
    return np.array(xs, dtype=np.float32)

# -------------- Split 70/15/15 --------------
def split_70_15_15(X, y):
    idx = np.arange(len(y))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    tr_idx, tmp_idx = next(sss1.split(idx, y))
    y_tr, y_tmp = y[tr_idx], y[tmp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    va_rel, te_rel = next(sss2.split(tmp_idx, y_tmp))
    val_idx = tmp_idx[va_rel]
    tst_idx = tmp_idx[te_rel]
    return tr_idx, val_idx, tst_idx

# -------------- Model -----------------------
def build_dense(input_dim, num_classes=10):
    m = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

# -------------- Train one pipeline ----------
def train_one_pipeline(pname, pfunc):
    print(f"\n===== Pipeline: {pname} =====")
    tr_idx, va_idx, te_idx = split_70_15_15(ALL_FILES, ALL_LABELS)

    Xtr = load_images(ALL_FILES[tr_idx], pfunc)
    Xva = load_images(ALL_FILES[va_idx], pfunc)
    Xte = load_images(ALL_FILES[te_idx], pfunc)
    ytr = ALL_LABELS[tr_idx]; yva = ALL_LABELS[va_idx]; yte = ALL_LABELS[te_idx]

    # flatten -> PCA (fit on train only)
    Xtr_flat = Xtr.reshape(len(Xtr), -1).astype(np.float32)
    Xva_flat = Xva.reshape(len(Xva), -1).astype(np.float32)
    Xte_flat = Xte.reshape(len(Xte), -1).astype(np.float32)

    pca = PCA(n_components=PCA_KEEP, svd_solver="full", random_state=SEED)
    Xtr_p = pca.fit_transform(Xtr_flat)
    Xva_p = pca.transform(Xva_flat)
    Xte_p = pca.transform(Xte_flat)

    input_dim = Xtr_p.shape[1]
    model = build_dense(input_dim)

    es = callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="max",
                                 restore_best_weights=True, verbose=1)
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                      min_lr=1e-5, verbose=1)

    hist = model.fit(
        Xtr_p.astype(np.float32), ytr,
        validation_data=(Xva_p.astype(np.float32), yva),
        epochs=EPOCHS, batch_size=BATCH, verbose=1,
        callbacks=[es, rlr]
    )

    va_acc = float(model.evaluate(Xva_p.astype(np.float32), yva, verbose=0)[1])
    te_acc = float(model.evaluate(Xte_p.astype(np.float32), yte, verbose=0)[1])
    print(f"[{pname}] Val Acc: {va_acc:.4f} | Test Acc: {te_acc:.4f} | PCA dims: {input_dim}")

    if SHOW_CM:
        y_pred = np.argmax(model.predict(Xte_p.astype(np.float32), verbose=0), axis=1)
        cm = confusion_matrix(yte, y_pred, labels=list(range(10)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f"Confusion Matrix — {pname}")
        plt.show()

    out = {
        "val_acc": va_acc,
        "test_acc": te_acc,
        "pca_dims": int(input_dim),
        "history": {k: [float(vv) for vv in hist.history[k]] for k in hist.history},
    }
    del Xtr, Xva, Xte, Xtr_flat, Xva_flat, Xte_flat, Xtr_p, Xva_p, Xte_p; gc.collect()
    return out

# -------------- Run all selected pipelines --------------
RESULTS = {}
for pname in PIPELINE_ORDER:
    if pname not in PIPELINES:
        continue
    RESULTS[pname] = train_one_pipeline(pname, PIPELINES[pname])

# -------------- Accuracy histogram --------------
df_rows = [{"pipeline": k, "test_acc": RESULTS[k]["test_acc"]} for k in RESULTS]
df = pd.DataFrame(df_rows).sort_values("test_acc", ascending=False)
print("\n=== Test Accuracies ===")
print(df)

plt.figure(figsize=(8, 4.5))
plt.bar(df["pipeline"], df["test_acc"])
plt.ylim(0, 1.0)
plt.ylabel("Test accuracy")
plt.title("Per-pipeline accuracy (70/15/15 split)")
plt.xticks(rotation=25)
plt.grid(axis="y", alpha=0.3)
plt.show()

# -------------- Enhancement samples --------------
rng = np.random.default_rng(SEED)
sample_idx = int(rng.integers(0, len(ALL_FILES)))
sample_path = ALL_FILES[sample_idx]
orig_bgr = cv2.imread(sample_path, cv2.IMREAD_COLOR)
orig_rgb = cv2.cvtColor(cv2.resize(orig_bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)

n_show = min(len(PIPELINE_ORDER), 6)  # keep the panel compact
to_show = PIPELINE_ORDER[:n_show]

plt.figure(figsize=(3*(n_show+1), 3))
plt.subplot(1, n_show+1, 1)
plt.imshow(orig_rgb); plt.axis("off"); plt.title("Original")

for i, name in enumerate(to_show, start=2):
    proc = (PIPELINES[name](orig_bgr) if name in PIPELINES else orig_rgb)
    if proc.dtype != np.uint8:
        if SCALE_RANGE == "[-1,1]":
            vis = np.clip((proc + 1.0)/2.0, 0, 1)
        else:
            vis = np.clip(proc, 0, 1)
        vis = (vis * 255).astype(np.uint8)
    else:
        vis = proc
    plt.subplot(1, n_show+1, i)
    plt.imshow(vis); plt.axis("off"); plt.title(name)

plt.suptitle(f"Enhancement samples — {os.path.basename(sample_path)}", y=1.03)
plt.tight_layout()
plt.show()



# ============ Multi-sample enhancement panel ============
def show_enhancement_samples(sample_count=4, max_pipelines=6, seed=SEED):
    """
    Displays a grid:
      rows   -> different random images
      cols   -> Original + selected pipelines
    """
    rng = np.random.default_rng(seed)
    n_samples = min(sample_count, len(ALL_FILES))
    idxs = rng.choice(len(ALL_FILES), size=n_samples, replace=False)

    # choose which pipelines to visualize (in the same order you train)
    to_show = [p for p in PIPELINE_ORDER if p in PIPELINES][:max_pipelines]
    ncols = 1 + len(to_show)
    nrows = n_samples

    plt.figure(figsize=(3*ncols, 3*nrows))
    for r, idx in enumerate(idxs):
        path = ALL_FILES[idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        # Original
        orig_rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        ax = plt.subplot(nrows, ncols, r*ncols + 1)
        ax.imshow(orig_rgb); ax.axis("off")
        if r == 0:
            ax.set_title("Original", fontsize=10)

        # Each pipeline
        for c, name in enumerate(to_show, start=1):
            proc = PIPELINES[name](bgr)
            # Convert to displayable RGB uint8
            if proc.dtype != np.uint8:
                vis = (np.clip((proc + 1.0)/2.0, 0, 1)
                       if SCALE_RANGE == "[-1,1]" else np.clip(proc, 0, 1))
                vis = (vis * 255).astype(np.uint8)
            else:
                vis = proc

            ax = plt.subplot(nrows, ncols, r*ncols + c + 1)
            ax.imshow(vis); ax.axis("off")
            if r == 0:
                ax.set_title(name, fontsize=10)

    plt.suptitle("Enhancement samples (multiple)", y=1.02)
    plt.tight_layout()
    plt.show()

# Example: show 5 samples across the first 6 pipelines
show_enhancement_samples(sample_count=5, max_pipelines=6)


