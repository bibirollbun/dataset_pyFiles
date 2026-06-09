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


# ============================================================
# High-Accuracy CNN Multi-Pipeline (State Farm) — seed=42
# From-scratch CNN, one-technique-per-pipeline, train-only:
#   Lighting & Aug_Rotate. Shared 70/15/15 split saved to disk.
# ============================================================
import os, glob, random, math, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

# -------------------- Config --------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_DIR = "/kaggle/input/state-farm-distracted-driver-detection/imgs/train"
CLASS_NAMES = [f"c{i}" for i in range(10)]
NUM_CLASSES = 10

IMG_SIZE = 224
SCALE_RANGE = "[0,1]"     # or "[-1,1]"
MAX_IMAGES = 4000         # raise for more data if VRAM allows
CAP_PER_CLASS = MAX_IMAGES // len(CLASS_NAMES) if MAX_IMAGES else None

# Accuracy toggles (ON by default)
HIGH_ACCURACY = True
EPOCHS = 20 if HIGH_ACCURACY else 5
BATCH  = 32
LR     = 1e-3
WEIGHT_DECAY = 1e-4 if HIGH_ACCURACY else 0.0
LABEL_SMOOTH = 0.05 if HIGH_ACCURACY else 0.0
USE_STANDARDIZE = True if HIGH_ACCURACY and SCALE_RANGE=="[0,1]" else False

SAVE_DIR   = "/kaggle/working/cnn_multipipeline"
PANEL_DIR  = f"{SAVE_DIR}/panels"
SPLIT_FILE = "/kaggle/working/dd_split_shared.npz"
os.makedirs(SAVE_DIR, exist_ok=True); os.makedirs(PANEL_DIR, exist_ok=True)

# Pipelines (fixed order)
PIPELINE_ORDER = [
    "Standard",
    "Lighting",         # TRAIN-only (val/test Standard)
    "Noise_Gaussian",
    "Feature_Edges",
    "Texture_LBP",
    "BackgroundRemoval",
    "MultiScale",
    "Aug_Rotate"        # TRAIN-only (val/test Standard)
]
TRAIN_ONLY = {"Lighting", "Aug_Rotate"}

# -------------------- Helpers --------------------
def _scale_out(x01):
    if SCALE_RANGE == "[-1,1]": return (x01*2.0 - 1.0).astype(np.float32)
    return x01.astype(np.float32)

def _imread_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _resize(img_rgb, size=IMG_SIZE):
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)

# -------------------- Pipelines --------------------
def pp_standard(img_bgr):
    rgb = _resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return _scale_out(rgb.astype(np.float32)/255.0)

def pp_lighting(img_bgr, clip=2.0, tile=8, gamma=1.15, alpha=1.05, beta=0.02):
    rgb = _resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    L2 = clahe.apply(L)
    rgb2 = cv2.cvtColor(cv2.merge([L2,A,B]), cv2.COLOR_LAB2RGB).astype(np.float32)/255.0
    rgb2 = np.power(np.clip(rgb2,0,1), gamma)
    rgb2 = np.clip(alpha*rgb2 + beta, 0.0, 1.0)
    return _scale_out(rgb2)

def pp_noise_gaussian(img_bgr, k=3):
    rgb = _resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    x = cv2.GaussianBlur(rgb, (k,k), 0)
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    return _scale_out(x.astype(np.float32)/255.0)

def pp_feature_edges(img_bgr):
    rgb = _resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    g  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    e  = cv2.Canny(g, 100, 200).astype(np.float32)/255.0
    e3 = np.repeat(e[...,None], 3, axis=-1)
    return _scale_out(e3)

def _lbp_simple(gray_u8):
    g = gray_u8.astype(np.float32); H,W = g.shape
    c = g[1:-1,1:-1]; code = np.zeros((H-2,W-2), dtype=np.uint8)
    for dy,dx,bit in [(-1,-1,7),(-1,0,6),(-1,1,5),(0,1,4),(1,1,3),(1,0,2),(1,-1,1),(0,-1,0)]:
        nbr = g[1+dy:H-1+dy, 1+dx:W-1+dx]
        code |= ((nbr >= c).astype(np.uint8) << bit)
    out = np.zeros((H,W), dtype=np.float32); out[1:-1,1:-1] = code.astype(np.float32)
    out = (out - out.min())/(out.max()-out.min()+1e-6)
    return out

def pp_texture_lbp(img_bgr):
    rgb = _resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    g   = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lbp = _lbp_simple(g)
    e3  = np.repeat(lbp[...,None], 3, axis=-1)
    return _scale_out(e3)

def pp_bg_grabcut(img_bgr):
    rgb = _resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    h,w = rgb.shape[:2]
    mask = np.full((h,w), cv2.GC_PR_BGD, np.uint8)
    rect = (w//8, h//8, 3*w//4, 3*h//4)
    bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fgmask = (mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)
        out = np.zeros_like(rgb, dtype=np.float32)
        out[fgmask] = rgb[fgmask].astype(np.float32)/255.0
        return _scale_out(out)
    except:
        return pp_standard(img_bgr)

def pp_multiscale(img_bgr):
    base = _resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).astype(np.float32)/255.0
    s2   = cv2.resize(base, (IMG_SIZE*3//4, IMG_SIZE*3//4), interpolation=cv2.INTER_AREA)
    s2   = cv2.resize(s2, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    s3   = cv2.resize(base, (IMG_SIZE//2, IMG_SIZE//2), interpolation=cv2.INTER_AREA)
    s3   = cv2.resize(s3, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    out  = np.clip((base+s2+s3)/3.0, 0, 1).astype(np.float32)
    return _scale_out(out)

def _rotate_rgb01(rgb01, max_deg=15):
    u8 = (np.clip(rgb01,0,1)*255).astype(np.uint8)
    h,w = u8.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), np.random.uniform(-max_deg, max_deg), 1.0)
    rot = cv2.warpAffine(u8, M, (w,h), borderMode=cv2.BORDER_REFLECT101)
    return rot.astype(np.float32)/255.0

def pp_aug_rotate(img_bgr):
    # rotation applied in loader when split == "train"
    rgb = _resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).astype(np.float32)/255.0
    return _scale_out(rgb)

PIPELINES = {
    "Standard":           pp_standard,
    "Lighting":           pp_lighting,
    "Noise_Gaussian":     pp_noise_gaussian,
    "Feature_Edges":      pp_feature_edges,
    "Texture_LBP":        pp_texture_lbp,
    "BackgroundRemoval":  pp_bg_grabcut,
    "MultiScale":         pp_multiscale,
    "Aug_Rotate":         pp_aug_rotate,
}

# -------------------- Files & split --------------------
def list_files(root, per_class=None):
    X, y = [], []
    for ci, cname in enumerate(CLASS_NAMES):
        files = sorted(glob.glob(os.path.join(root, cname, "*.jpg")))
        if per_class: files = files[:per_class]
        X += files; y += [ci]*len(files)
    return np.array(X), np.array(y, dtype=np.int32)

ALL_FILES, ALL_LABELS = list_files(DATA_DIR, CAP_PER_CLASS)
print(f"Total files (capped): {len(ALL_FILES)}")

if os.path.exists(SPLIT_FILE):
    s = np.load(SPLIT_FILE, allow_pickle=True)
    tr_idx, va_idx, te_idx = s["tr"], s["va"], s["te"]
    print("Loaded shared 70/15/15 split.")
else:
    idx = np.arange(len(ALL_LABELS))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    (tr_idx, tmp_idx), = sss1.split(idx, ALL_LABELS)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    (va_rel, te_rel), = sss2.split(tmp_idx, ALL_LABELS[tmp_idx])
    va_idx, te_idx = tmp_idx[va_rel], tmp_idx[te_rel]
    np.savez(SPLIT_FILE, tr=tr_idx, va=va_idx, te=te_idx)
    print("Created shared 70/15/15 split and saved.")

TR_FILES, VA_FILES, TE_FILES = ALL_FILES[tr_idx], ALL_FILES[va_idx], ALL_FILES[te_idx]
TR_Y,     VA_Y,     TE_Y     = ALL_LABELS[tr_idx], ALL_LABELS[va_idx], ALL_LABELS[te_idx]

# -------------------- Standardization (optional) --------------------
MEAN, STD = None, None
def estimate_mean_std(paths, sample_n=2000):
    sel = paths if len(paths) <= sample_n else np.random.RandomState(SEED).choice(paths, sample_n, replace=False)
    s_mean = np.zeros(3, dtype=np.float64); s_sq = np.zeros(3, dtype=np.float64); n_pix = 0
    for p in sel:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None: continue
        x = pp_standard(bgr)  # [0,1]
        h,w,_ = x.shape
        s_mean += x.reshape(-1,3).sum(axis=0)
        s_sq   += (x.reshape(-1,3)**2).sum(axis=0)
        n_pix  += h*w
    mean = s_mean / max(n_pix,1)
    var  = s_sq / max(n_pix,1) - mean**2
    std  = np.sqrt(np.clip(var, 1e-8, None))
    return mean.astype(np.float32), std.astype(np.float32)

if USE_STANDARDIZE:
    MEAN, STD = estimate_mean_std(TR_FILES)
    print("Channel mean/std (TRAIN/Standard):", MEAN, STD)

def standardize(x):
    if MEAN is None or STD is None: return x
    return (x - MEAN) / STD

# -------------------- Loader Sequence --------------------
class PipelineSequence(keras.utils.Sequence):
    def __init__(self, files, labels, pname, split, batch=BATCH, shuffle=True):
        self.files = files.copy(); self.labels = labels.copy()
        self.pname = pname; self.split = split; self.batch = batch
        self.shuffle = shuffle; self.rng = np.random.RandomState(SEED)
        self.on_epoch_end()
    def __len__(self): return math.ceil(len(self.files)/self.batch)
    def on_epoch_end(self):
        if self.shuffle and self.split=="train":
            idx = self.rng.permutation(len(self.files))
            self.files, self.labels = self.files[idx], self.labels[idx]
    def __getitem__(self, i):
        sl = slice(i*self.batch, (i+1)*self.batch)
        batch_files = self.files[sl]; batch_labels = self.labels[sl]
        Xb, yb = [], []
        pfunc = PIPELINES[self.pname]
        for p, lab in zip(batch_files, batch_labels):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None: continue
            if self.pname in TRAIN_ONLY and self.split != "train":
                x = pp_standard(bgr)  # eval = Standard
            else:
                x = pfunc(bgr)
                if self.pname == "Aug_Rotate" and self.split == "train":
                    x = _rotate_rgb01(x, 15)
            if USE_STANDARDIZE and SCALE_RANGE=="[0,1]":
                x = standardize(x)
            Xb.append(x); yb.append(lab)
        X = np.array(Xb, dtype=np.float32)
        y = to_categorical(np.array(yb, dtype=np.int32), num_classes=NUM_CLASSES)
        return X, y

def show_panel(pname):
    sel = TR_FILES[:min(5, len(TR_FILES))]
    grid = []
    for p in sel:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None: continue
        if pname in TRAIN_ONLY:
            x = PIPELINES[pname](bgr)
            if pname=="Aug_Rotate": x = _rotate_rgb01(x, 15)
        else:
            x = PIPELINES[pname](bgr)
        if USE_STANDARDIZE and SCALE_RANGE=="[0,1]": x = standardize(x)
        grid.append(x)
    if not grid: return None
    cols = len(grid); plt.figure(figsize=(3*cols, 3))
    for i, im in enumerate(grid, 1):
        if USE_STANDARDIZE:
            # visualize standardized by de-standardizing to [0,1]
            vis = np.clip(im*STD + MEAN, 0, 1) if MEAN is not None else np.clip(im,0,1)
        else:
            vis = np.clip(im,0,1) if SCALE_RANGE=="[0,1]" else np.clip((im+1)/2,0,1)
        plt.subplot(1, cols, i); plt.imshow(vis); plt.axis("off")
    plt.suptitle(f"{pname} — TRAIN samples")
    out = f"{PANEL_DIR}/panel_{pname}.png"
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    return out

# -------------------- Model (4 conv blocks, BN, Dropout) --------------------
def build_cnn():
    L2 = keras.regularizers.l2(WEIGHT_DECAY) if WEIGHT_DECAY>0 else None
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    def block(x, ch, dp=0.0):
        x = layers.Conv2D(ch, 3, padding="same", use_bias=False, kernel_regularizer=L2)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(ch, 3, padding="same", use_bias=False, kernel_regularizer=L2)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        if dp>0: x = layers.Dropout(dp)(x)
        return x
    x = block(inp, 32, dp=0.1)
    x = block(x,   64, dp=0.15)
    x = block(x,  128, dp=0.2)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_regularizer=L2)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation=None, use_bias=False, kernel_regularizer=L2)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.3 if HIGH_ACCURACY else 0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inp, out)
    # Optimizer: AdamW if available, else Adam
    try:
        opt = keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    except Exception:
        opt = keras.optimizers.Adam(learning_rate=LR)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model

# -------------------- Train one pipeline --------------------
def train_one_pipeline(pname):
    print(f"\n===== Pipeline: {pname} =====")
    panel_path = show_panel(pname)

    train_seq = PipelineSequence(TR_FILES, TR_Y, pname, split="train", batch=BATCH, shuffle=True)
    val_seq   = PipelineSequence(VA_FILES, VA_Y, pname, split="val",   batch=BATCH, shuffle=False)
    test_seq  = PipelineSequence(TE_FILES, TE_Y, pname, split="test",  batch=BATCH, shuffle=False)

    model = build_cnn()
    cbs = [
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, mode="max", verbose=1),
    ]
    hist = model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS, verbose=2, callbacks=cbs)

    va_acc  = float(model.evaluate(val_seq,  verbose=0)[1])
    te_acc  = float(model.evaluate(test_seq, verbose=0)[1])

    out_dir = f"{SAVE_DIR}/{pname}"; os.makedirs(out_dir, exist_ok=True)
    model.save(f"{out_dir}/cnn_model.keras")
    with open(f"{out_dir}/history.json","w") as f:
        json.dump({k:[float(v) for v in hist.history[k]] for k in hist.history}, f)
    if panel_path:
        os.replace(panel_path, f"{out_dir}/" + os.path.basename(panel_path))

    print(f"[{pname}] Val Acc: {va_acc:.4f} | Test Acc: {te_acc:.4f}")
    return {"val_acc": va_acc, "test_acc": te_acc}

# -------------------- Run all pipelines --------------------
results = {}
for pname in PIPELINE_ORDER:
    results[pname] = train_one_pipeline(pname); gc.collect()

with open(f"{SAVE_DIR}/results.json","w") as f: json.dump(results, f, indent=2)
df = pd.DataFrame([{"pipeline":k, **v} for k,v in results.items()])[["pipeline","val_acc","test_acc"]]
print("\n=== Accuracies ==="); print(df.sort_values("test_acc", ascending=False))

# Bar chart
order = PIPELINE_ORDER
xs = np.arange(len(order)); vals = [results[p]["test_acc"] for p in order]
plt.figure(figsize=(10,4)); plt.bar(xs, vals)
plt.xticks(xs, order, rotation=20); plt.ylabel("Test Accuracy")
plt.title("CNN — Test Accuracy by Pipeline (70/15/15 shared split)")
plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/bar_test_accuracy.png", dpi=130); plt.show()

print(f"\nArtifacts saved under: {SAVE_DIR}")


