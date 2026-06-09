import os, sys, gc, math, random, time, json, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Repro
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 8
LR = 3e-4
IMG_SIZE = 384           # EfficientNet likes 224–380; 384 is a sweet spot
NUM_WORKERS = 2

# Try to find the competition data folder automatically
# ================================
# Robust PATH discovery (drop-in replacement)
# ================================
import os, glob, re

def has_benign_malignant(d):
    try:
        entries = [e.lower() for e in os.listdir(d)]
        return ("benign" in entries) and ("malignant" in entries)
    except Exception:
        return False

def find_training_and_testing_roots():
    # 1) Search all first/second-level dirs under /kaggle/input for a training_set with benign/malignant
    candidates = []

    for path in glob.glob("/kaggle/input/*"):
        if os.path.isdir(path):
            # prefer "complete_set/training_set"
            c_train = os.path.join(path, "complete_set", "training_set")
            c_test  = os.path.join(path, "complete_set", "testing_set")
            if os.path.isdir(c_train) and has_benign_malignant(c_train):
                candidates.append(("complete", c_train, c_test if os.path.isdir(c_test) else None))

            # top-level "training_set"
            t_train = os.path.join(path, "training_set")
            if os.path.isdir(t_train) and has_benign_malignant(t_train):
                # test may be alongside as "testing_set" or "test"
                t_test = None
                for name in ["testing_set", "test", "Testing_set", "Test"]:
                    p = os.path.join(path, name)
                    if os.path.isdir(p):
                        t_test = p; break
                candidates.append(("top", t_train, t_test))

            # any nested place that directly has benign/malignant
            for nested in glob.glob(os.path.join(path, "**"), recursive=False):
                if os.path.isdir(nested) and has_benign_malignant(nested):
                    # try to find a sibling test folder
                    base = os.path.dirname(nested)
                    test_sibling = None
                    for name in ["testing_set", "test"]:
                        p = os.path.join(base, name)
                        if os.path.isdir(p):
                            test_sibling = p; break
                    candidates.append(("nested", nested, test_sibling))

    # Choose best match: prefer one that also has a testing_set
    with_test = [c for c in candidates if c[2] is not None]
    chosen = with_test[0] if with_test else (candidates[0] if candidates else None)
    if not chosen:
        raise FileNotFoundError("Could not find a training folder containing 'benign' and 'malignant' under /kaggle/input.")

    _, train_dir, test_dir = chosen
    return train_dir, test_dir

TRAIN_DIR, TEST_DIR = find_training_and_testing_roots()

def find_sample_submission_near(train_dir):
    # look near training dir and at its parent & root
    roots = [
        os.path.dirname(train_dir),
        os.path.dirname(os.path.dirname(train_dir)),
        "/kaggle/input"
    ]
    seen = set()
    for r in roots:
        if r and os.path.isdir(r) and r not in seen:
            seen.add(r)
            for p in glob.glob(os.path.join(r, "**", "sample_submission.csv"), recursive=True):
                return p
    return None

SS_PATH = find_sample_submission_near(TRAIN_DIR)

print("Resolved paths:")
print("  TRAIN_DIR:", TRAIN_DIR)
print("  TEST_DIR :", TEST_DIR)
print("  SS_PATH  :", SS_PATH)

# Light sanity check (no hard assert that caused the crash)
print("  Has benign?:", os.path.isdir(os.path.join(TRAIN_DIR, "benign")) or os.path.isdir(os.path.join(TRAIN_DIR, "Benign")))
print("  Has malignant?:", os.path.isdir(os.path.join(TRAIN_DIR, "malignant")) or os.path.isdir(os.path.join(TRAIN_DIR, "Malignant")))



def list_images(dir_path, label=None):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(dir_path, e))
        files += glob.glob(os.path.join(dir_path, "**", e), recursive=True)  # just in case
    rows = [{"path":f, "id":os.path.basename(f)} for f in sorted(set(files))]
    if label is not None:
        for r in rows: r["label"] = label
    return rows

train_benign   = list_images(os.path.join(TRAIN_DIR, "benign"), "benign")
train_malign   = list_images(os.path.join(TRAIN_DIR, "malignant"), "malignant")
train_df       = pd.DataFrame(train_benign + train_malign)

label2id = {"benign":0, "malignant":1}
train_df["target"] = train_df["label"].map(label2id)

# patient heuristic to reduce leakage (prefix before first underscore)
def patient_from_name(name):
    base = os.path.splitext(name)[0]
    return base.split("_")[0]
train_df["patient"] = train_df["id"].map(patient_from_name)

print("Train samples:", len(train_df))
display(train_df.head())


from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_idx, va_idx = next(sgkf.split(train_df, y=train_df["target"], groups=train_df["patient"]))
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)
print("Train/Valid sizes:", tr_df.shape, va_df.shape)


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.1)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

valid_tfms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# simple 2x TTA: identity + horizontal flip
tta_tfms = [
    valid_tfms,
    transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
]


class ImgDS(Dataset):
    def __init__(self, df, tfm, with_label=True):
        self.df = df
        self.tfm = tfm
        self.with_label = with_label
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        img = self.tfm(img)
        if self.with_label:
            return img, row["target"]
        return img, row["id"], row["path"]

tr_loader = DataLoader(ImgDS(tr_df, train_tfms, True), batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True)
va_loader = DataLoader(ImgDS(va_df, valid_tfms, True), batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)


from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)  # binary
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)



from tqdm.auto import tqdm

def run_epoch(loader, train_mode=True, epoch=0, phase="train"):
    model.train(train_mode)
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"{phase.capitalize()} E{epoch:02d}", leave=False)
    for imgs, targets in pbar:
        imgs = imgs.to(DEVICE)
        targets = targets.float().to(DEVICE)

        with torch.set_grad_enabled(train_mode):
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, targets)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # metrics
        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == targets.long()).sum().item()
        total += imgs.size(0)

        # live progress
        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

    return running_loss/total, correct/total

best_acc, best_path = 0.0, "best_model.pt"
for epoch in tqdm(range(1, EPOCHS+1), desc="Epochs"):
    tr_loss, tr_acc = run_epoch(tr_loader, True, epoch, "train")
    va_loss, va_acc = run_epoch(va_loader, False, epoch, "valid")
    scheduler.step()
    # Use tqdm.write so the line isn’t overwritten by the bars
    tqdm.write(f"Epoch {epoch:02d} | TR loss {tr_loss:.4f} acc {tr_acc:.4f} "
               f"| VA loss {va_loss:.4f} acc {va_acc:.4f}")
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(model.state_dict(), best_path)

tqdm.write(f"Best valid acc: {best_acc:.4f}")

# Free a bit
gc.collect(); torch.cuda.empty_cache()



def list_test_images(dir_path):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(dir_path, e))
        files += glob.glob(os.path.join(dir_path, "**", e), recursive=True)
    files = sorted(set(files))
    return pd.DataFrame({"path": files, "id": [os.path.basename(f) for f in files]})

test_df = list_test_images(TEST_DIR)
print("Test images:", len(test_df))
display(test_df.head())

# Load best model
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.eval();

def predict_with_tta(paths):
    probs_all = []
    for tfm in tta_tfms:
        ds = ImgDS(pd.DataFrame({"path": paths, "id":[os.path.basename(p) for p in paths]}),
                   tfm, with_label=False)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
        part = []
        for imgs, ids, _ in dl:
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                logits = model(imgs).squeeze(1)
                part.append(torch.sigmoid(logits).detach().cpu().numpy())
        probs_all.append(np.concatenate(part))
    return np.mean(probs_all, axis=0)

p_malignant = predict_with_tta(test_df["path"].tolist())
p_benign    = 1.0 - p_malignant


def build_submission(ids, p_benign, p_malignant, ss_path=None):
    if ss_path and os.path.exists(ss_path):
        ss = pd.read_csv(ss_path)
        out = ss.copy()
        cols = list(out.columns)
        # If two probability columns exist:
        if {"Benign","Malignant"}.issubset(set(cols)):
            out["Benign"]    = p_benign[:len(out)]
            out["Malignant"] = p_malignant[:len(out)]
            return out
        # If there is a single label-like column:
        label_like = [c for c in cols if c.lower() in ("label","target","prediction","predicted","diagnosis")]
        if label_like:
            c = label_like[0]
            # If strings expected (M/B), output those; else 0/1
            if out[c].dtype == object and out[c].dropna().isin(["M","B"]).any():
                out[c] = np.where(p_malignant[:len(out)] >= 0.5, "M", "B")
            else:
                out[c] = (p_malignant[:len(out)] >= 0.5).astype(int)
            return out
        # Otherwise, append probability columns:
        if "Benign" not in out.columns:    out["Benign"] = p_benign[:len(out)]
        if "Malignant" not in out.columns: out["Malignant"] = p_malignant[:len(out)]
        return out
    # No sample submission: make a simple, safe file
    return pd.DataFrame({
        "id": ids,
        "Benign": p_benign,
        "Malignant": p_malignant
    })

sub_df = build_submission(test_df["id"].tolist(), p_benign, p_malignant, SS_PATH)
sub_df.to_csv("submission.csv", index=False)
print("Saved:", "submission.csv")
display(sub_df.head())

