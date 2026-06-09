# === Grand X-Ray Slam: Division B ===
# Uses:
#   DATA_ROOT = "/kaggle/input/grand-xray-slam-division-b"
#   TRAIN_CSV = "/kaggle/input/grand-xray-slam-division-b/train2.csv"
#   OUT_DIR   = "/kaggle/working/outputs_effv2s_fold0"

import os, json, math, random, gc
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
import timm

# ----------------------- USER PATHS (fixed) -----------------------
DATA_ROOT = "/kaggle/input/grand-xray-slam-division-b"        # contains train2/ and test2/
TRAIN_CSV = "/kaggle/input/grand-xray-slam-division-b/train2.csv"
OUT_DIR   = "/kaggle/working/outputs_effv2s_fold0"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------- CONFIG -----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum",
    "Fracture","Lung Lesion","Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

FOLD         = 0
IMG_SIZE     = 256
BATCH_SIZE   = 4
NUM_WORKERS  = 0    # start with 0 to surface errors clearly; bump to 2/4 later
MODEL_NAME   = "tf_efficientnetv2_s"
LR           = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 10    # increase to 10–15 for real training

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV exists?", os.path.isfile(TRAIN_CSV))
print("OUT_DIR:", OUT_DIR)

# ----------------------- ARTIFACTS (pos_weight, folds, manifest) -----------------------
df = pd.read_csv(TRAIN_CSV)
df.columns = df.columns.str.strip()

# Ensure label columns exist; if any are missing, create as 0.0
for c in LABELS:
    if c not in df.columns:
        df[c] = 0.0

# Sanitize label dtypes to float32 in [0,1]
for c in LABELS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[LABELS] = df[LABELS].fillna(0.0).clip(0, 1).astype("float32")

# pos_weight for BCEWithLogitsLoss
total = len(df)
pos = df[LABELS].sum().astype(float)
neg = (total - pos).astype(float)
pos_weight = (neg / pos.replace(0, np.nan)).fillna(100.0)
with open(os.path.join(OUT_DIR, "pos_weight.json"), "w") as f:
    json.dump({c: float(pos_weight[c]) for c in LABELS}, f, indent=2)

# Patient-wise 5-fold split
gkf = GroupKFold(n_splits=5)
fold_ids = np.zeros(total, dtype=int)
for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df["Patient_ID"].astype(str).values)):
    fold_ids[val_idx] = fold
df_folds = df[["Image_name","Patient_ID"]].copy()
df_folds["fold"] = fold_ids
df_folds.to_csv(os.path.join(OUT_DIR, "train_folds.csv"), index=False)

# Minimal manifest for dataloader
manifest = df[["Image_name","Patient_ID","Study","Sex","Age","ViewCategory","ViewPosition"] + LABELS].copy()
manifest.to_csv(os.path.join(OUT_DIR, "train_manifest.csv"), index=False)

print("Artifacts saved:",
      os.path.join(OUT_DIR, "pos_weight.json"),
      os.path.join(OUT_DIR, "train_folds.csv"),
      os.path.join(OUT_DIR, "train_manifest.csv"))

# ----------------------- DATASET (hardened) + SAFE COLLATE -----------------------
class CXRDataset(Dataset):
    """
    - Forces square resize with OpenCV to (size, size) BEFORE transforms.
    - Pre-sanitizes label array to float32.
    - Robustly resolves image root (DATA_ROOT/img_dir or DATA_ROOT itself if already pointing to images).
    """
    def __init__(self, df: pd.DataFrame, data_root: str, img_dir: str, train: bool=True, size: int=512):
        self.df = df.reset_index(drop=True)
        # pre-store labels as float32 array (N,14)
        lab = self.df[LABELS].copy()
        for c in LABELS: lab[c] = pd.to_numeric(lab[c], errors="coerce")
        lab = lab.fillna(0.0).clip(0,1).astype("float32")
        self.labels = lab.values

        root_candidate = Path(data_root) / img_dir
        if not root_candidate.exists():
            if Path(data_root).exists() and any(Path(data_root).glob("*.jpg")):
                root_candidate = Path(data_root)
        if not root_candidate.exists():
            raise FileNotFoundError(f"Image root not found: tried '{Path(data_root)/img_dir}' and '{data_root}'")
        self.root = root_candidate

        self.train = train
        self.size  = int(size)
        self.train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ToTensor()
        ])
        self.val_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        try:
            r = self.df.iloc[idx]
            p = self.root / r["Image_name"]
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            img = np.stack([img, img, img], axis=-1)
            tf = self.train_tf if self.train else self.val_tf
            x = tf(img)
            y = torch.from_numpy(self.labels[idx])
            return x, y, r["Image_name"]
        except Exception:
            return None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0), []
    xs, ys, names = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(names)

# ----------------------- BUILD LOADERS -----------------------
dfm = pd.read_csv(os.path.join(OUT_DIR, "train_manifest.csv"))
dff = pd.read_csv(os.path.join(OUT_DIR, "train_folds.csv"))
df_all = dfm.merge(dff, on=["Image_name","Patient_ID"], how="left")

tr_df = df_all[df_all.fold != FOLD].copy()
va_df = df_all[df_all.fold == FOLD].copy()

tr_ds = CXRDataset(tr_df, DATA_ROOT, "train2", train=True,  size=IMG_SIZE)
va_ds = CXRDataset(va_df, DATA_ROOT, "train2", train=False, size=IMG_SIZE)

# smoke test a mini-batch
chk = DataLoader(tr_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=safe_collate)
xb, yb, names = next(iter(chk))
print("Sanity batch:", xb.shape, yb.shape, "first:", names[0] if names else None)

tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, collate_fn=safe_collate)
va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=safe_collate)

# ----------------------- METRICS -----------------------
@torch.no_grad()
def evaluate_logits(y_true, y_pred_logits, search_thresholds=True):
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        return np.nan, np.nan, [0.5]*len(LABELS)
    p = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()
    aucs=[]
    for i in range(y_true.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true[:,i], p[:,i]))
        except Exception:
            aucs.append(np.nan)
    thresholds = [0.5]*y_true.shape[1]
    macro_f1 = np.nan
    if search_thresholds:
        ts=[]
        for i in range(y_true.shape[1]):
            best_t, best_f1 = 0.5, 0.0
            for t in np.linspace(0.05,0.95,19):
                if y_true[:,i].max() > 0:
                    f1 = f1_score(y_true[:,i], (p[:,i]>=t).astype(int))
                else:
                    f1 = 0.0
                if f1>best_f1: best_f1, best_t = f1, t
            ts.append(best_t)
        thresholds = ts
        macro_f1 = np.nanmean([f1_score(y_true[:,i], (p[:,i]>=thresholds[i]).astype(int)) for i in range(y_true.shape[1])])
    return np.nanmean(aucs), macro_f1, thresholds

# ----------------------- MODEL / LOSS / OPT -----------------------
with open(os.path.join(OUT_DIR, "pos_weight.json")) as f:
    pos_weight_dict = json.load(f)
pos_weight_vec = torch.tensor([pos_weight_dict[c] for c in LABELS], dtype=torch.float32)

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(LABELS)).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_vec.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
use_amp = (device=="cuda")
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler(enabled=use_amp)

# ----------------------- TRAIN (quick) -----------------------
best_f1 = -1.0
history = []
for epoch in range(1, EPOCHS+1):
    model.train(); train_loss, n = 0.0, 0
    for x, y, _ in tr_ld:
        if x.numel()==0: continue  # all skipped
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(x)
            loss   = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        train_loss += loss.item()*x.size(0); n += x.size(0)
    train_loss /= max(n,1)

    # validate
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in va_ld:
            if x.numel()==0: continue
            x = x.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(x)
            ps.append(logits.float().cpu().numpy())
            ys.append(y.numpy())
    y_true   = np.concatenate(ys) if ys else np.zeros((0,len(LABELS)))
    y_logits = np.concatenate(ps) if ps else np.zeros((0,len(LABELS)))
    auc, mf1, thresholds = evaluate_logits(y_true, y_logits, search_thresholds=True)

    history.append({"epoch": epoch, "train_loss": float(train_loss), "val_auc": float(auc), "val_macro_f1": float(mf1)})
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_auc={auc:.4f} val_macro_f1={mf1:.4f}")

    ckpt = {"model": model.state_dict(), "cfg": {"MODEL_NAME": MODEL_NAME, "IMG_SIZE": IMG_SIZE, "LABELS": LABELS}, "epoch": epoch}
    torch.save(ckpt, os.path.join(OUT_DIR, "last.ckpt"))
    if (mf1==mf1) and (mf1>best_f1):   # mf1==mf1 guards against NaN
        best_f1 = mf1
        torch.save(ckpt, os.path.join(OUT_DIR, "best.ckpt"))
        with open(os.path.join(OUT_DIR, "thresholds.json"), "w") as f:
            json.dump({"thresholds": thresholds, "labels": LABELS}, f, indent=2)
    with open(os.path.join(OUT_DIR, "val_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

print("Training done. Best macro-F1:", best_f1, " Checkpoints @", OUT_DIR)

# ----------------------- INFERENCE ON test2/ -----------------------
from timm import create_model as _cm

ckpt_path = Path(OUT_DIR)/"best.ckpt"
if not ckpt_path.exists():
    print("best.ckpt not found; using last.ckpt")
    ckpt_path = Path(OUT_DIR)/"last.ckpt"
assert ckpt_path.exists(), f"No checkpoint in {OUT_DIR}"

ckpt = torch.load(ckpt_path, map_location="cpu")
labels = ckpt.get("cfg", {}).get("LABELS", LABELS)
model_name = ckpt.get("cfg", {}).get("MODEL_NAME", MODEL_NAME)
img_size = ckpt.get("cfg", {}).get("IMG_SIZE", IMG_SIZE)

model_inf = _cm(model_name, pretrained=False, num_classes=len(labels))
model_inf.load_state_dict(ckpt["model"], strict=True)
model_inf = model_inf.to(device).eval()

img_folder = Path(DATA_ROOT) / "test2"
assert img_folder.exists(), f"Image folder not found: {img_folder}"
image_names = sorted([f.name for f in img_folder.iterdir() if f.suffix.lower() in (".jpg",".png")])
assert len(image_names) > 0, f"No images in {img_folder}"

# minimal DF with dummy labels
d = {"Image_name": image_names}
for lab in labels: d[lab] = 0
df_dummy = pd.DataFrame(d)

ds = CXRDataset(df_dummy, DATA_ROOT, "test2", train=False, size=img_size)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=safe_collate)

probs_all = []
with torch.no_grad():
    for x, _, _ in dl:
        if x.numel()==0: continue
        x = x.to(device, non_blocking=True)
        logits = model_inf(x)
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
probs = np.concatenate(probs_all, axis=0)

sub = pd.DataFrame({"Image_name": image_names})
for i, lab in enumerate(labels):
    sub[lab] = probs[:, i]
sub_soft = str(Path(OUT_DIR)/"submission.csv")
sub.to_csv(sub_soft, index=False)
print("Saved soft-prob submission to", sub_soft)

thr_path = Path(OUT_DIR)/"thresholds.json"
if thr_path.exists():
    th = np.array(json.load(open(thr_path))["thresholds"], dtype=float)
    hard = (probs >= th).astype(int)
    hard_df = pd.DataFrame({"Image_name": image_names})
    for i, lab in enumerate(labels):
        hard_df[lab] = hard[:, i]
    sub_hard = str(Path(OUT_DIR)/"submission_hard.csv")
    hard_df.to_csv(sub_hard, index=False)
    print("Saved hard-label CSV to", sub_hard)

print("DONE ")


