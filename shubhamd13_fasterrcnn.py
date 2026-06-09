# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

'''
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



csv_path = Path("/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv")
df = pd.read_csv(csv_path)

# quick peek
print(df.head())
# â�œ patientId x y width height Target



# collapse the rows belonging to the same study
per_image = (
    df.groupby("patientId")["Target"]
      .max()                     # â€œmaxâ€� is 1 if any row has a box
      .rename("has_box")
)

num_images      = len(per_image)
num_positives   = per_image.sum()
fraction_pos    = num_positives / num_images

print(f"{num_positives} / {num_images} images "
      f"({fraction_pos:.1%}) contain at least one bounding box.")



boxes_per_image = (
    df.query("Target == 1")
      .groupby("patientId").size()
)

boxes_per_image.hist(bins=range(1, 10), rwidth=0.8)
plt.xlabel("# boxes in the study")
plt.ylabel("count of studies")
plt.title("Lesion count distribution")
plt.show()



import numpy as np

pos = df.query("Target == 1")
areas      = (pos["width"] * pos["height"]).values
aspect_rat = (pos["width"] / pos["height"]).values

fig, ax = plt.subplots()
ax.hist(np.sqrt(areas), bins=40)        # use sqrt so the axis is in â€œpixelsâ€�
ax.set_xlabel("âˆšarea (â‰ˆ lesion diameter in pixels)")
ax.set_title("Lesion size distribution")
plt.show()

plt.figure()
plt.hist(aspect_rat, bins=40)
plt.xlabel("width / height")
plt.title("Aspect-ratio distribution")
plt.show()



import cv2, pydicom, matplotlib.patches as patches

sample_ids = boxes_per_image.sample(4, random_state=0).index
root = Path("/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images")

for pid in sample_ids:
    dcm = pydicom.dcmread(root/f"{pid}.dcm")
    img = dcm.pixel_array
    fig, ax = plt.subplots(1, figsize=(5,5))
    ax.imshow(img, cmap="gray")
    for _, row in pos[pos.patientId == pid].iterrows():
        rect = patches.Rectangle(
            (row.x, row.y), row.width, row.height,
            linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
    ax.set_title(pid)
    ax.axis("off")
    plt.show()



import cv2, pydicom, torch.nn.functional as F
from torch.utils.data import Dataset
RAW = Path("/kaggle/input/rsna-pneumonia-detection-challenge")  # read-only

LABELS = RAW/"stage_2_train_labels.csv"
df = pd.read_csv(LABELS)

class RSNADataset(Dataset):
    def __init__(self, df, root, transforms=None):
        self.img_ids = df.patientId.unique()
        self.df   = df
        self.root = Path(root)
        self.tfm  = transforms

    def __getitem__(self, idx):
        pid = self.img_ids[idx]

        # -------- image ----------
        ds  = pydicom.dcmread(self.root / f"{pid}.dcm")
        img = ds.pixel_array.astype("float32")
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = torch.as_tensor(np.stack([img, img, img]), dtype=torch.float32)

        # -------- target ----------
        recs = self.df[self.df.patientId == pid]
        pos  = recs[recs.Target == 1]

        if len(pos):
            xyxy = np.column_stack(
                [pos.x, pos.y, pos.x + pos.width, pos.y + pos.height]
            )
            boxes = torch.as_tensor(xyxy, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)  # class 1
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)      # 2-D!
            labels = torch.zeros((0,),  dtype=torch.int64)

        target = {
            "boxes":   boxes,
            "labels":  labels,
            "image_id": torch.tensor([idx]),
        }

        if self.tfm:                       # (no transforms in this baseline)
            img = self.tfm(img)

        return img, target

    def __len__(self):
        return len(self.img_ids)



import torch, torchvision, platform, subprocess, os, random, numpy as np, pandas as pd
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

from pathlib import Path

RAW = Path("/kaggle/input/rsna-pneumonia-detection-challenge")
TRAIN_IMG_DIR = RAW / "stage_2_train_images"
TEST_IMG_DIR  = RAW / "stage_2_test_images"

print("sample train file:", next(TRAIN_IMG_DIR.glob("*.dcm")))


# ---------------------------------------------------------
# 1)  Train / validation split (stratified by Target = 0/1)
# ---------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
import pandas as pd

RAW = Path("/kaggle/input/rsna-pneumonia-detection-challenge")
TRAIN_IMG_DIR = RAW / "stage_2_train_images"

df   = pd.read_csv(RAW / "stage_2_train_labels.csv")
ids  = df.groupby("patientId")["Target"].max().reset_index()

DEBUG = False
POS   = 100       # â†� how many positive studies you want
NEG   = 100       # â†� how many negative studies you want

ids   = df.groupby("patientId")["Target"].max().reset_index()   # 0/1 label
if DEBUG:
    pos_ids = ids[ids.Target == 1].sample(POS,  random_state=0)
    neg_ids = ids[ids.Target == 0].sample(NEG,  random_state=0)
    ids = pd.concat([pos_ids, neg_ids]).reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(ids.patientId, ids.Target))

train_df = df[df.patientId.isin(ids.patientId.iloc[train_idx])]
val_df   = df[df.patientId.isin(ids.patientId.iloc[val_idx])]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(ids.patientId, ids.Target))

train_df = df[df.patientId.isin(ids.patientId.iloc[train_idx])]
val_df   = df[df.patientId.isin(ids.patientId.iloc[val_idx])]

# ---------------------------------------------------------
# 2)  Datasets (root points directly at the DICOM folders)
# ---------------------------------------------------------
train_ds = RSNADataset(train_df, TRAIN_IMG_DIR)
val_ds   = RSNADataset(val_df,   TRAIN_IMG_DIR)

# ---------------------------------------------------------
# 3)  DataLoaders  (keyword args â†’ no â€œsampler vs shuffleâ€� clash)
# ---------------------------------------------------------
def collate(batch):
    return tuple(zip(*batch))

train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=12,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    collate_fn=collate,
)

val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    collate_fn=collate,
)



from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # â†� NEW path
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_ch = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_ch, num_classes=2)

# â¬…ï¸� anchors: lesions tiny â†’ include 16 px, 32 px
model.rpn.anchor_generator.sizes = ((16, 32, 64, 128, 256),)

model.to("cuda")



# === 0.  Imports === -------------------------------------------------------
from pathlib import Path
import torch, math, os
from tqdm.auto import tqdm

# Preferred AMP spelling in PyTorch â‰¥2.3
from torch.amp import autocast          #  â†� NEW import path
from torch.cuda.amp import GradScaler

# === 1.  Optimizer / LR / AMP scaler === ----------------------------------
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4
)
lr_sched  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scaler    = GradScaler()

# === 2.  Epoch runner with tqdm  (loss-dict even in val) === --------------
def run_epoch(loader, train=True, desc="train"):
    """
    One full pass over `loader`.
    If `train=False` we disable gradients, but still keep the model in
    training *mode* so Faster R-CNN returns a loss dict instead of detections.
    """
    model.train(True)                        # <-- ALWAYS train() for loss
    torch.set_grad_enabled(train)            # gradients only in train phase

    running, seen = 0.0, 0
    bar = tqdm(loader, desc=desc, leave=False)

    for imgs, tgts in bar:
        imgs = [img.cuda(non_blocking=True) for img in imgs]
        tgts = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in tgts]

        with autocast(device_type="cuda"):
            loss_dict = model(imgs, tgts)    # guaranteed dict
            loss = sum(loss_dict.values())

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = len(imgs)
        running += loss.item() * bs
        seen    += bs
        bar.set_postfix(loss=f"{loss.item():.3f}",
                        avg=f"{running/seen:.3f}")

    return running / seen

# === 3.  Checkpoint helpers === -------------------------------------------
CKPT_DIR = Path("/kaggle/working/checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

def save_ckpt(epoch):
    fname = CKPT_DIR / f"{epoch:03d}.pth"
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
    }, fname)
    print(f"âœ”ï¸�  Saved checkpoint â†’ {fname}")

def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    print(f"ğŸ”„  Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"] + 1   # resume at next epoch

# === 4.  Optional resume === ----------------------------------------------
latest = sorted(CKPT_DIR.glob("*.pth"))[-1:]   # pick the newest file, if any
start_epoch = load_ckpt(latest[0]) if latest else 1

# === 5.  Training loop === -------------------------------------------------
EPOCHS = 3          # change as you like

for epoch in range(start_epoch, EPOCHS + 1):
    tr_loss  = run_epoch(train_dl, train=True,  desc=f"Ep{epoch:02d}[train]")
    val_loss = run_epoch(val_dl,   train=False, desc=f"Ep{epoch:02d}[val]  ")

    lr_sched.step()
    print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {val_loss:.4f}")

    save_ckpt(epoch)



# ================================================================
#   Inference, simple stats, and 4-image visual sanity-check
# ================================================================
from pathlib import Path
import torch, pydicom, cv2, numpy as np, pandas as pd, random, matplotlib.pyplot as plt
from tqdm.auto import tqdm
plt.rcParams["figure.figsize"] = (5, 5)

# ---------- 0. Locate & load checkpoint ---------- #
CKPT_DIR = Path("/kaggle/working/checkpoints")
latest   = sorted(CKPT_DIR.glob("*.pth"))[-1]
ckpt     = torch.load(latest, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.to("cuda").eval()
print(f"âœ”ï¸�  Restored epoch {ckpt['epoch']} weights from {latest.name}")

# ---------- 1. Build test file list & helper ---------- #
RAW           = Path("/kaggle/input/rsna-pneumonia-detection-challenge")
TEST_IMG_DIR  = RAW / "stage_2_test_images"
test_files    = sorted(TEST_IMG_DIR.glob("*.dcm"))
BATCH_SIZE    = 6
SCORE_THR     = 0.30          # tweak to taste

def dcms_to_tensor(paths):
    """Read list[Path] â†’ list[torch.Tensor] (3Ã—HÃ—W, 0-1)."""
    out = []
    for p in paths:
        dcm  = pydicom.dcmread(p)
        img  = dcm.pixel_array.astype("float32")
        img  = (img - img.min()) / (img.max() - img.min() + 1e-6)
        out.append(torch.as_tensor(np.stack([img, img, img]), dtype=torch.float32))
    return out

# ---------- 2. Run inference over test set ---------- #
rows, all_scores, box_counts = [], [], []
for i in tqdm(range(0, len(test_files), BATCH_SIZE), desc="Infer"):
    batch_paths = test_files[i : i + BATCH_SIZE]
    imgs  = [t.cuda(non_blocking=True) for t in dcms_to_tensor(batch_paths)]
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        preds = model(imgs)

    for pth, pred in zip(batch_paths, preds):
        keep = pred["scores"] > SCORE_THR
        n_kept = int(keep.sum())
        box_counts.append(n_kept)

        if n_kept == 0:
            pred_str = "0 0 1 1 0.1"
        else:
            boxes  = pred["boxes"][keep].cpu().numpy()
            scores = pred["scores"][keep].cpu().numpy()
            all_scores.extend(scores.tolist())
            parts = [
                f"{x1:.0f} {y1:.0f} {x2-x1:.0f} {y2-y1:.0f} {s:.2f}"
                for (x1, y1, x2, y2), s in zip(boxes, scores)
            ]
            pred_str = " ".join(parts)

        rows.append({"patientId": pth.stem, "PredictionString": pred_str})

sub = pd.DataFrame(rows)
sub.to_csv("submission.csv", index=False)
print(f"\nSaved submission.csv with {len(sub)} rows")

# ---------- 3. Quick test-set stats ---------- #
total_imgs   = len(test_files)
with_boxes   = sum(b > 0 for b in box_counts)
avg_boxes    = np.mean(box_counts)
print(f"\nTEST-SET STATS  (score â‰¥ {SCORE_THR})")
print("-" * 40)
print(f"Images processed     : {total_imgs}")
print(f"Images w/ detections : {with_boxes} ({with_boxes/total_imgs:.1%})")
print(f"Avg boxes / image    : {avg_boxes:.2f}")
if all_scores:
    print(f"Score range          : {min(all_scores):.2f} â€“ {max(all_scores):.2f}")
    print(f"Median score         : {np.median(all_scores):.2f}")

# ---------- 4. Visualise 4 random predictions ---------- #
sample_paths = random.sample(test_files, 4)
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for ax, pth in zip(axes, sample_paths):
    dcm  = pydicom.dcmread(pth)
    img  = dcm.pixel_array
    ax.imshow(img, cmap="gray")
    ax.set_title(pth.stem, fontsize=8)
    ax.axis("off")

    # Get prediction we just computed
    pred_row   = sub[sub.patientId == pth.stem].iloc[0]
    if pred_row["PredictionString"].startswith("0 0 1 1"):
        continue  # no detection

    vals = list(map(float, pred_row["PredictionString"].split()))
    for x1, y1, w, h, sc in np.array(vals).reshape(-1, 5):
        x2, y2 = x1 + w, y1 + h
        ax.add_patch(plt.Rectangle(
            (x1, y1), w, h, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, y1, f"{sc:.2f}", color="yellow", fontsize=6,
                bbox=dict(facecolor="black", alpha=0.5, pad=1))

plt.tight_layout(); plt.show()



# ==================  VISUAL CHECK: GT vs. PRED  ==================
import random, torch, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches
import pydicom, pandas as pd
from pathlib import Path

NUM_SAMPLES = 4     # how many images to display
SCORE_THR   = 0.30  # draw preds with score >= this

# -----------------------------------------------------------------
# 1)  Restore latest checkpoint
# -----------------------------------------------------------------
CKPT_DIR = Path("/kaggle/working/checkpoints")
latest   = sorted(CKPT_DIR.glob("*.pth"))[-1]           # newest file
ckpt     = torch.load(latest, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.to("cuda").eval()
print(f"Loaded epoch {ckpt['epoch']} checkpoint â�œ {latest.name}")

# -----------------------------------------------------------------
# 2)  Pick random *positive* validation studies
# -----------------------------------------------------------------
ids_with_box = (val_df.groupby("patientId")["Target"].max() == 1)
pos_ids      = ids_with_box[ids_with_box].index.tolist()
sample_ids   = random.sample(pos_ids, NUM_SAMPLES)

RAW   = Path("/kaggle/input/rsna-pneumonia-detection-challenge")
IMG_DIR = RAW / "stage_2_train_images"                  # same folder as training

def load_img_tensor(pid):
    dcm = pydicom.dcmread(IMG_DIR/f"{pid}.dcm")
    img = dcm.pixel_array.astype("float32")
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    tens = torch.as_tensor(np.stack([img]*3), dtype=torch.float32)
    return img, tens

# -----------------------------------------------------------------
# 3)  Plot
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, NUM_SAMPLES, figsize=(5*NUM_SAMPLES, 5))
if NUM_SAMPLES == 1: axes = [axes]

for ax, pid in zip(axes, sample_ids):
    # --- load image & GT boxes --------------------------
    img_np, tensor = load_img_tensor(pid)
    gt_rows = val_df[(val_df.patientId == pid) & (val_df.Target == 1)]
    gt_boxes = gt_rows[['x', 'y', 'width', 'height']].values

    # --- model prediction -------------------------------
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        pred = model([tensor.cuda()])[0]
    keep   = pred['scores'] > SCORE_THR
    boxes  = pred['boxes'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()

    # --- draw -------------------------------------------
    ax.imshow(img_np, cmap="gray")
    ax.set_title(pid, fontsize=8); ax.axis("off")

    # ground-truth (red)
    for (x, y, w, h) in gt_boxes:
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2,
                                       edgecolor="red", facecolor="none"))

    # predictions (lime)
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                       edgecolor="lime", facecolor="none"))
        ax.text(x1, y1, f"{sc:.2f}", color="yellow", fontsize=6,
                bbox=dict(facecolor="black", alpha=0.4, pad=1))

plt.tight_layout(); plt.show()



from torchvision.ops import box_iou
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
import pydicom

# === Setup ===
RAW = Path("/kaggle/input/rsna-pneumonia-detection-challenge")
TRAIN_IMG_DIR = RAW / "stage_2_train_images"
LABELS_CSV = RAW / "stage_2_train_labels.csv"
BATCH_SIZE = 6
SCORE_THR = 0.30

# === Load Ground Truth ===
df = pd.read_csv(LABELS_CSV)
df_pos = df[df.Target == 1]
gt_patients = df_pos.patientId.unique()
test_files = [f for f in sorted(TRAIN_IMG_DIR.glob("*.dcm")) if f.stem in gt_patients]

# === Helper ===
def get_gt_boxes(pid):
    rows = df_pos[df_pos.patientId == pid]
    return torch.tensor([
        [row.x, row.y, row.x + row.width, row.y + row.height]
        for _, row in rows.iterrows()
    ], dtype=torch.float32)

def dcms_to_tensor(paths):
    out = []
    for p in paths:
        dcm = pydicom.dcmread(p)
        img = dcm.pixel_array.astype("float32")
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        out.append(torch.as_tensor(np.stack([img, img, img]), dtype=torch.float32))
    return out

# === Inference & IoU ===
rows, ious, iou_per_patient = [], [], {}

for i in tqdm(range(0, len(test_files), BATCH_SIZE), desc="Inference"):
    batch_paths = test_files[i : i + BATCH_SIZE]
    imgs = [t.cuda(non_blocking=True) for t in dcms_to_tensor(batch_paths)]
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        preds = model(imgs)

    for pth, pred in zip(batch_paths, preds):
        pid = pth.stem
        gt_boxes = get_gt_boxes(pid)
        keep = pred["scores"] > SCORE_THR

        if keep.sum() == 0 or gt_boxes.numel() == 0:
            rows.append({"patientId": pid, "PredictionString": "0 0 1 1 0.1"})
            continue

        boxes = pred["boxes"][keep].cpu()
        scores = pred["scores"][keep].cpu().numpy()
        pred_str = " ".join([
            f"{x1:.0f} {y1:.0f} {x2-x1:.0f} {y2-y1:.0f} {s:.2f}"
            for (x1, y1, x2, y2), s in zip(boxes, scores)
        ])
        rows.append({"patientId": pid, "PredictionString": pred_str})

        iou_matrix = box_iou(boxes, gt_boxes)
        best_ious = iou_matrix.max(dim=1)[0].tolist()
        ious.extend(best_ious)
        iou_per_patient[pid] = np.mean(best_ious)

# Save predictions
sub = pd.DataFrame(rows)
sub.to_csv("submission_trainset.csv", index=False)

# IoU stats
print(f"\nIoU Statistics on {len(ious)} predictions:")
print("-" * 40)
print(f"Mean IoU       : {np.mean(ious):.3f}")
print(f"Median IoU     : {np.median(ious):.3f}")
print(f"IoU > 0.5 count      : {sum(i > 0.5 for i in ious)} ({sum(i > 0.5 for i in ious)/len(ious)*100:.1f}%)")





