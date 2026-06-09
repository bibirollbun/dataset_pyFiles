from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

DATA_ROOT = Path("/kaggle/input/global-wheat-detection")
print("Root exists:", DATA_ROOT.exists())
print("Contents:", [p.name for p in DATA_ROOT.iterdir()])



train_csv = DATA_ROOT / "train.csv"
df = pd.read_csv(train_csv)
df.head()



# choose one image id
image_id = df["image_id"].iloc[0]
print("Using image_id:", image_id)

# all rows (boxes) for this image
df_img = df[df["image_id"] == image_id].reset_index(drop=True)
print("Number of boxes:", len(df_img))

# image path
img_path = DATA_ROOT / "train" / f"{image_id}.jpg"
print("Image path:", img_path, "exists:", img_path.exists())



# load image
img = Image.open(img_path).convert("RGB")
img_np = np.array(img).astype(np.float32) / 255.0
H, W, _ = img_np.shape
print("Image size:", W, "x", H)






# --- Color filter: keep only green & yellow-ish pixels ---

R = img_np[..., 0]
G = img_np[..., 1]
B = img_np[..., 2]

# very simple rules (you can tweak thresholds):
# green: G is strongest channel and not too dark
mask_green = (G > R) & (G > B) & (G > 0.25)

# yellow: R and G are both high, B is low
mask_yellow = (R > 0.4) & (G > 0.4) & (B < 0.3)

mask_plant = mask_green | mask_yellow

# make a copy and zero-out everything that is not plant-colored
img_filtered = img_np.copy()
img_filtered[~mask_plant] = 0.0

# from here on, use img_filtered instead of img_np
img_np = img_filtered







# parse bboxes: each bbox string -> x, y, w, h
boxes = []
for s in df_img["bbox"]:
    # s looks like "[x, y, w, h]"
    x, y, w, h = eval(s)  # safe here: controlled csv format
    boxes.append((x, y, w, h))

boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
print("Parsed boxes shape:", boxes.shape)



# choose grid resolution (tune if you like)
GRID_H, GRID_W = 10, 10

head_counts = np.zeros((GRID_H, GRID_W), dtype=np.float32)

for x, y, w, h in boxes:
    cx = x + w / 2.0
    cy = y + h / 2.0
    j = min(int(cx / W * GRID_W), GRID_W - 1)  # column
    i = min(int(cy / H * GRID_H), GRID_H - 1)  # row
    head_counts[i, j] += 1.0

# normalize density (1 = max density in this image)
if head_counts.max() > 0:
    density_norm = head_counts / head_counts.max()
else:
    density_norm = head_counts

density_norm.shape



##Yellow as Bad

"""GI = np.zeros((GRID_H, GRID_W), dtype=np.float32)
cell_h = H // GRID_H
cell_w = W // GRID_W

for i in range(GRID_H):
    for j in range(GRID_W):
        y0 = i * cell_h
        y1 = (i + 1) * cell_h if i < GRID_H - 1 else H
        x0 = j * cell_w
        x1 = (j + 1) * cell_w if j < GRID_W - 1 else W

        tile = img_np[y0:y1, x0:x1, :]
        R = tile[..., 0].mean()
        G = tile[..., 1].mean()
        B = tile[..., 2].mean()
        GI[i, j] = G / (R + G + B + 1e-6)

GI_mean = GI.mean()
GI_diff = np.clip(GI_mean - GI, 0, None)    # >0 where less green (more yellow)
GI_norm = GI_diff / (GI_diff.max() + 1e-6)

# suspicion: high when density is low & greenness is low
suspicion = 0.7 * (1.0 - density_norm) + 0.3 * GI_norm"""



## Yellow as Bad only if Lesser heads..

GI = np.zeros((GRID_H, GRID_W), dtype=np.float32)
cell_h = H // GRID_H
cell_w = W // GRID_W

for i in range(GRID_H):
    for j in range(GRID_W):
        y0 = i * cell_h
        y1 = (i + 1) * cell_h if i < GRID_H - 1 else H
        x0 = j * cell_w
        x1 = (j + 1) * cell_w if j < GRID_W - 1 else W

        tile = img_np[y0:y1, x0:x1, :]
        R = tile[..., 0].mean()
        G = tile[..., 1].mean()
        B = tile[..., 2].mean()
        GI[i, j] = G / (R + G + B + 1e-6)

GI_mean = GI.mean()
GI_diff = np.clip(GI_mean - GI, 0, None)    # >0 where more yellow
GI_norm = GI_diff / (GI_diff.max() + 1e-6)

# --- NEW PART: yellow only bad when head density is low ---
# density_norm is already in [0,1]; high density = good crop
color_term = GI_norm.copy()
color_term[density_norm >= 0.5] = 0.0   # yellow ignored where there are many heads

# suspicion: base = few heads, plus extra if yellow in low-head tiles
base_susp = 1.0 - density_norm
suspicion = 0.7 * base_susp + 0.3 * color_term




import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- reload true original image ---
img = Image.open(img_path).convert("RGB")
img_np_orig = np.array(img).astype(np.float32) / 255.0
H, W, _ = img_np_orig.shape

# --- grid + head density from GT boxes ---
GRID_H, GRID_W = 10, 10
cell_h = H // GRID_H
cell_w = W // GRID_W

head_counts = np.zeros((GRID_H, GRID_W), dtype=np.float32)
for x, y, w, h in boxes:
    cx = x + w / 2.0
    cy = y + h / 2.0
    j = min(int(cx / W * GRID_W), GRID_W - 1)
    i = min(int(cy / H * GRID_H), GRID_H - 1)
    head_counts[i, j] += 1.0

if head_counts.max() > 0:
    density_norm = head_counts / head_counts.max()
else:
    density_norm = head_counts.copy()

base_susp = 1.0 - density_norm  # few heads â‡’ more suspicious

# ---------- heatmap WITHOUT filter ----------
GI_unf = np.zeros((GRID_H, GRID_W), dtype=np.float32)
for i in range(GRID_H):
    for j in range(GRID_W):
        y0 = i * cell_h
        y1 = (i + 1) * cell_h if i < GRID_H - 1 else H
        x0 = j * cell_w
        x1 = (j + 1) * cell_w if j < GRID_W - 1 else W

        tile = img_np_orig[y0:y1, x0:x1, :]
        Rm = tile[..., 0].mean()
        Gm = tile[..., 1].mean()
        Bm = tile[..., 2].mean()
        GI_unf[i, j] = Gm / (Rm + Gm + Bm + 1e-6)

GI_mean_unf = GI_unf.mean()
GI_diff_unf = np.clip(GI_mean_unf - GI_unf, 0, None)
GI_norm_unf = GI_diff_unf / (GI_diff_unf.max() + 1e-6)

color_term_unf = GI_norm_unf.copy()
color_term_unf[density_norm >= 0.5] = 0.0   # yellow only bad if few heads

suspicion_unf = 0.7 * base_susp + 0.3 * color_term_unf

# ---------- build filtered image (green + yellow only) ----------
R = img_np_filt[..., 0]
G = img_np_filt[..., 1]
B = img_np_filt[..., 2]

# green: G clearly strongest
mask_green = (G > R) & (G > B) & (G > 0.25)

# yellow: R and G both fairly high, similar to each other, and B lower
'''mask_yellow = (R > 0.3) & (G > 0.3) & (B < 0.6) & (np.abs(R - G) < 0.25)'''
mask_yellow = (R > 0.25) & (G > 0.2) & (B < 0.8)

mask_plant = mask_green | mask_yellow


# grab MORE yellow: lower thresholds and remove the strict Râ‰ˆG condition


mask_plant = mask_green | mask_yellow






# ---------- heatmap WITH filter ----------
GI_filt = np.zeros((GRID_H, GRID_W), dtype=np.float32)
for i in range(GRID_H):
    for j in range(GRID_W):
        y0 = i * cell_h
        y1 = (i + 1) * cell_h if i < GRID_H - 1 else H
        x0 = j * cell_w
        x1 = (j + 1) * cell_w if j < GRID_W - 1 else W

        tile = img_np_filt[y0:y1, x0:x1, :]
        Rm = tile[..., 0].mean()
        Gm = tile[..., 1].mean()
        Bm = tile[..., 2].mean()
        GI_filt[i, j] = Gm / (Rm + Gm + Bm + 1e-6)

GI_mean_filt = GI_filt.mean()
GI_diff_filt = np.clip(GI_mean_filt - GI_filt, 0, None)
GI_norm_filt = GI_diff_filt / (GI_diff_filt.max() + 1e-6)

color_term_filt = GI_norm_filt.copy()
color_term_filt[density_norm >= 0.5] = 0.0

suspicion_filt = 0.7 * base_susp + 0.3 * color_term_filt

# ---------- plot 4 panels ----------
plt.figure(figsize=(20, 8))

# 1) original image
plt.subplot(2, 2, 1)
plt.imshow(img_np_orig)
plt.title("Original image")
plt.axis("off")

# 2) heatmap WITHOUT filter
plt.subplot(2, 2, 2)
plt.imshow(img_np_orig)
plt.imshow(
    suspicion_unf,
    cmap="jet",
    alpha=0.45,
    extent=(0, W, H, 0),
    interpolation="bilinear",
)
plt.title("Heatmap WITHOUT filter")
plt.axis("off")

# 3) filtered image (green + yellow only)
plt.subplot(2, 2, 3)
plt.imshow(img_np_filt)
plt.title("Filtered image (green + yellow kept)")
plt.axis("off")

# 4) heatmap WITH filter
plt.subplot(2, 2, 4)
plt.imshow(img_np_filt)
plt.imshow(
    suspicion_filt,
    cmap="jet",
    alpha=0.45,
    extent=(0, W, H, 0),
    interpolation="bilinear",
)
plt.title("Heatmap WITH filter")
plt.axis("off")

plt.tight_layout()
plt.show()



#HEADCOUNT USING REGRESSION - LIGHT & FAST ====> FOR DRONE


# HC1: Build a dataset that returns (image_tensor, 10x10 head-count map) for each wheat image

from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Global Wheat dataset root + annotations
DATA_ROOT = Path("/kaggle/input/global-wheat-detection")
df_wheat = pd.read_csv(DATA_ROOT / "train.csv")

# Use a subset for faster training (you can increase max_images later)
all_ids = df_wheat["image_id"].unique()
max_images = 1200
image_ids = all_ids[:max_images]

GRID_H, GRID_W = 10, 10  # we keep the same 10x10 grid

class WheatHeadCountDataset(Dataset):
    """
    For each image_id:
    - load the RGB image
    - build a 10x10 head-count map using the bbox centres
    - return (image_tensor, head_map_tensor)
    """
    def __init__(self, data_root, df, image_ids, grid_h=10, grid_w=10, transform=None):
        self.data_root = data_root
        self.df = df
        self.image_ids = list(image_ids)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.data_root / "train" / f"{image_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # build 10x10 head-count map (same logic as before using bbox centres)
        head_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        df_img = self.df[self.df["image_id"] == image_id]

        for s in df_img["bbox"]:
            x, y, w, h = eval(s)  # bbox = [x_min, y_min, w, h] in pixels
            cx = x + w / 2.0
            cy = y + h / 2.0
            j = min(int(cx / W * self.grid_w), self.grid_w - 1)
            i = min(int(cy / H * self.grid_h), self.grid_h - 1)
            head_map[i, j] += 1.0  # increment count for that tile

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        head_map_tensor = torch.from_numpy(head_map)  # shape [10,10], dtype float32

        return img_tensor, head_map_tensor

# Simple image transform: resize to 320x320, random flip, to tensor
transform_hc = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

full_hc_ds = WheatHeadCountDataset(
    DATA_ROOT, df_wheat, image_ids,
    grid_h=GRID_H, grid_w=GRID_W,
    transform=transform_hc,
)

# 80/20 train/val split
train_size = int(0.8 * len(full_hc_ds))
val_size   = len(full_hc_ds) - train_size
hc_train_ds, hc_val_ds = random_split(full_hc_ds, [train_size, val_size])

hc_train_loader = DataLoader(hc_train_ds, batch_size=8, shuffle=True,  num_workers=2)
hc_val_loader   = DataLoader(hc_val_ds,   batch_size=8, shuffle=False, num_workers=2)

print("Head-count dataset size:", len(full_hc_ds))
print("Train:", len(hc_train_ds), " Val:", len(hc_val_ds))



# HC2: Define a small fully-convolutional network that predicts a 10x10 head-count map from the image

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for head-count model:", device)

class HeadCountNet(nn.Module):
    """
    Input:   [B, 3, 320, 320] RGB image batch
    Output:  [B, 1, 10, 10] head-count map (counts per tile)

    We use only conv + pooling so output is a map, not a single number.
    ReLU at the end keeps predicted counts >= 0.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),    # 160x160

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),    # 80x80

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),    # 40x40

            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),    # 20x20

            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),    # 10x10
        )
        # 1x1 conv to convert 128 channels -> 1 head-count channel
        self.head_map = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)           # [B, 128, 10, 10]
        x = self.head_map(x)           # [B, 1, 10, 10]
        x = torch.relu(x)              # ReLU: f(z)=max(0,z); keep counts non-negative
        return x

hc_model = HeadCountNet().to(device)

# MSE (Mean Squared Error) = average of (predicted - true)^2 over all cells
hc_criterion = nn.MSELoss()
hc_optimizer = torch.optim.Adam(hc_model.parameters(), lr=1e-3)

print(hc_model)



# HC3 (low-head focused): train with weighted MSE that emphasises low-head tiles

def weighted_mse(preds, targets, alpha=4.0, low_thresh=1.0):
    """
    preds, targets: [B, 10, 10]
    alpha: how much extra weight for low-head tiles
    low_thresh: tiles with GT < low_thresh are considered 'low head'
    """
    # base weight = 1
    weights = torch.ones_like(targets)

    # tiles with few heads in GT
    low_mask = (targets < low_thresh).float()   # 1 for low-head tiles

    # increase their weight
    weights = weights + alpha * low_mask        # low tiles -> 1+alpha

    sq_err = (preds - targets) ** 2
    loss = (sq_err * weights).mean()
    return loss

def evaluate_headcount(model, loader, alpha=4.0, low_thresh=1.0):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for imgs, maps_gt in loader:
            imgs = imgs.to(device)
            maps_gt = maps_gt.to(device)
            preds = model(imgs).squeeze(1)       # [B,10,10]
            loss = weighted_mse(preds, maps_gt, alpha=alpha, low_thresh=low_thresh)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)

num_epochs = 15
alpha = 4.0        # importance boost for low-head tiles
low_thresh = 1.0   # GT < 1 head -> "low"

for epoch in range(1, num_epochs + 1):

    # simple LR decay
    if epoch == 6:
        for g in hc_optimizer.param_groups:
            g["lr"] *= 0.2

    hc_model.train()
    running_loss = 0.0
    batches = 0

    for imgs, maps_gt in hc_train_loader:
        imgs = imgs.to(device)
        maps_gt = maps_gt.to(device)

        hc_optimizer.zero_grad()
        preds = hc_model(imgs).squeeze(1)
        loss = weighted_mse(preds, maps_gt, alpha=alpha, low_thresh=low_thresh)
        loss.backward()
        hc_optimizer.step()

        running_loss += loss.item()
        batches += 1

    train_loss = running_loss / max(batches, 1)
    val_loss   = evaluate_headcount(hc_model, hc_val_loader, alpha=alpha, low_thresh=low_thresh)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")



# HC4: Visual check â€“ show image, true 10x10 head map, and predicted 10x10 head map for one sample

import matplotlib.pyplot as plt

hc_model.eval()

# pick one sample from validation set
img_tensor, head_map_gt = hc_val_ds[1]
img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,C] for plotting

with torch.no_grad():
    pred_map = hc_model(img_tensor.unsqueeze(0).to(device))
    pred_map = pred_map.squeeze(0).squeeze(0).cpu().numpy()  # [10,10]

gt_display   = head_map_gt.numpy()
pred_display = pred_map

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.axis("off")
plt.title("Input image (resized)")

plt.subplot(1, 3, 2)
plt.imshow(gt_display, cmap="viridis")
plt.colorbar()
plt.title("Ground-truth head-count map\n(10x10)")

plt.subplot(1, 3, 3)
plt.imshow(pred_display, cmap="viridis")
plt.colorbar()
plt.title("Predicted head-count map\n(10x10)")

plt.tight_layout()
plt.show()



# HC5: measure average inference time of the head-count model (per image)

import time
import torch

hc_model.eval()

n_batches = 20          # how many batches to test over (can increase)
total_imgs = 0
start = time.time()

with torch.no_grad():
    for b, (imgs, maps_gt) in enumerate(hc_val_loader):
        imgs = imgs.to(device)

        _ = hc_model(imgs)   # forward pass only

        total_imgs += imgs.size(0)
        if (b + 1) >= n_batches:
            break

elapsed = time.time() - start

print("Images processed:", total_imgs)
print(f"Total time: {elapsed:.4f} s")
print(f"Avg time per image: {elapsed / total_imgs:.4f} s")
print(f"Approx FPS (images per second): {total_imgs / elapsed:.1f}")









#YOLO for DRONE








!pip install -q ultralytics



# ONE-CELL YOLO PIPELINE ON GLOBAL-WHEAT + 10x10 HEATMAP COMPARISON

import os, shutil, yaml, random
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from IPython.display import display, Image as IPyImage
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.modules.container
from ultralytics.nn.modules import Conv

# ---------- 1. CONFIG ----------
GWD_PATH = '/kaggle/input/global-wheat-detection'
WORK_DIR = '/kaggle/working/yolo_gwd_data'
MODEL_NAME = 'yolov8n_gwd_wheat'
EPOCHS = 10          # you can set back to 50 if you want
IMG_SIZE = 640
BATCH_SIZE = 16      # reduce if OOM
VAL_SPLIT_RATIO = 0.1
ORIGINAL_IMG_SIZE = 1024
GRID_H = GRID_W = 10
random.seed(42)

print("Setup OK.")

# ---------- 2. SAFE LOAD PATCH (light) ----------
try:
    torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
    torch.serialization.add_safe_globals([Conv])
except Exception:
    pass

# ---------- 3. DATA PREP: CSV -> YOLO FORMAT ----------
print("\n--- Preparing YOLO dataset from Global Wheat ---")
df = pd.read_csv(os.path.join(GWD_PATH, 'train.csv'))
image_ids = df['image_id'].unique()
random.shuffle(image_ids)

val_samples = int(len(image_ids) * VAL_SPLIT_RATIO)
val_ids = image_ids[:val_samples]
train_ids = image_ids[val_samples:]

# Clean & create dirs
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
os.makedirs(os.path.join(WORK_DIR, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, 'labels/val'), exist_ok=True)

def coco_to_yolo(bbox_str):
    x_min, y_min, w, h = eval(bbox_str)
    x_center = (x_min + w / 2) / ORIGINAL_IMG_SIZE
    y_center = (y_min + h / 2) / ORIGINAL_IMG_SIZE
    width = w / ORIGINAL_IMG_SIZE
    height = h / ORIGINAL_IMG_SIZE
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

IMAGE_SOURCE_DIR = os.path.join(GWD_PATH, 'train')

for image_id in tqdm(image_ids, desc="Processing images"):
    is_val = image_id in val_ids
    img_dst_dir = os.path.join(WORK_DIR, 'images/val') if is_val else os.path.join(WORK_DIR, 'images/train')
    lbl_dst_dir = os.path.join(WORK_DIR, 'labels/val') if is_val else os.path.join(WORK_DIR, 'labels/train')

    src_img_path = os.path.join(IMAGE_SOURCE_DIR, f'{image_id}.jpg')
    dst_img_path = os.path.join(img_dst_dir, f'{image_id}.jpg')
    shutil.copyfile(src_img_path, dst_img_path)

    annotations = df[df['image_id'] == image_id]['bbox'].dropna()
    yolo_lines = [coco_to_yolo(bbox) for bbox in annotations]
    with open(os.path.join(lbl_dst_dir, f'{image_id}.txt'), 'w') as f:
        f.write('\n'.join(yolo_lines))

print("âœ… Annotation conversion & split done.")

# ---------- 4. data.yaml ----------
DATA_YAML_PATH = os.path.join(WORK_DIR, 'wheat_data.yaml')
data_yaml = dict(
    path=WORK_DIR,
    train='images/train',
    val='images/val',
    names={0: 'wheat_head'},
    cache='disk'
)
with open(DATA_YAML_PATH, 'w') as out:
    yaml.dump(data_yaml, out, default_flow_style=False)
print(f"data.yaml written at: {DATA_YAML_PATH}")

# ---------- 5. TRAIN YOLOv8n VIA CLI ----------
print("\n--- Training YOLOv8n via CLI ---")
!yolo train model=yolov8n.pt data={DATA_YAML_PATH} epochs={EPOCHS} imgsz={IMG_SIZE} batch={BATCH_SIZE} name={MODEL_NAME} project="/kaggle/working/runs/detect"

print("\n--- Training command finished (check logs above) ---")

# ---------- 6. LOAD BEST WEIGHTS & RUN INFERENCE ----------
BEST_WEIGHTS = f'/kaggle/working/runs/detect/{MODEL_NAME}/weights/best.pt'
if not os.path.exists(BEST_WEIGHTS):
    print(f"â�Œ best.pt not found at {BEST_WEIGHTS}. Training may have failed.")
else:
    print(f"\nâœ… Found best weights at: {BEST_WEIGHTS}")
    best_model = YOLO(BEST_WEIGHTS)

    val_img_dir = os.path.join(WORK_DIR, 'images/val')
    val_images = os.listdir(val_img_dir)
    if not val_images:
        print("No val images found for inference.")
    else:
        sample_img_name = val_images[0]
        sample_img_path = os.path.join(val_img_dir, sample_img_name)
        image_id = sample_img_name.replace('.jpg', '')
        print(f"\nRunning inference on sample: {sample_img_name}")

        # YOLO prediction
        pred = best_model.predict(source=sample_img_path, conf=0.25, verbose=False, save=True, name='gwd_inference_test')[0]
        boxes = pred.boxes
        wheat_head_count = len(boxes) if boxes is not None else 0
        print(f"Predicted wheat heads: {wheat_head_count}")

        # Display YOLO result image
        result_path = f'/kaggle/working/runs/detect/gwd_inference_test/{sample_img_name}'
        if os.path.exists(result_path):
            display(IPyImage(filename=result_path))

        # ---------- 7. BUILD 10x10 GT vs YOLO HEATMAPS ----------
        # reload df (already loaded, but safe)
        # df already in memory

        # Load image for size + display
        with Image.open(sample_img_path) as im:
            W, H = im.size
            img_np = np.array(im).astype(np.float32) / 255.0

        # GT tile map
        gt_map = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        rows = df[df['image_id'] == image_id]
        for s in rows['bbox']:
            x, y, w, h = eval(s)
            cx = x + w/2
            cy = y + h/2
            j = min(int(cx / W * GRID_W), GRID_W - 1)
            i = min(int(cy / H * GRID_H), GRID_H - 1)
            gt_map[i, j] += 1.0

        # YOLO tile map
        yolo_map = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2) in xyxy:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                j = min(int(cx / W * GRID_W), GRID_W - 1)
                i = min(int(cy / H * GRID_H), GRID_H - 1)
                yolo_map[i, j] += 1.0

        gt_norm   = gt_map   / (gt_map.max()   + 1e-6)
        yolo_norm = yolo_map / (yolo_map.max() + 1e-6)

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.axis("off")
        plt.title(f"Image {image_id}")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_norm, cmap="viridis")
        plt.colorbar()
        plt.title("GT head density (10Ã—10)")

        plt.subplot(1, 3, 3)
        plt.imshow(yolo_norm, cmap="viridis")
        plt.colorbar()
        plt.title("YOLO head density (10Ã—10)")

        plt.tight_layout()
        plt.show()






# Measure YOLO inference FPS and example drone speed

import os, time
from ultralytics import YOLO

# use the same paths as in your YOLO cell
BEST_WEIGHTS = f'/kaggle/working/runs/detect/yolov8n_gwd_wheat/weights/best.pt'
WORK_DIR = '/kaggle/working/yolo_gwd_data'

model = YOLO(BEST_WEIGHTS)

val_dir = os.path.join(WORK_DIR, "images/val")
val_images = sorted([f for f in os.listdir(val_dir) if f.endswith(".jpg")])

# number of images to time
N = min(100, len(val_images))
paths = [os.path.join(val_dir, f) for f in val_images[:N]]

print(f"Timing on {N} validation images...")

# warm-up once (loads model on GPU, etc.)
_ = model.predict(source=paths[0], conf=0.25, verbose=False, save=False)

t0 = time.time()
for p in paths:
    _ = model.predict(source=p, conf=0.25, verbose=False, save=False)
t1 = time.time()

total_time = t1 - t0
avg_time = total_time / N
fps = N / total_time

print(f"Total time       : {total_time:.4f} s")
print(f"Avg time / image : {avg_time:.4f} s")
print(f"Approx FPS       : {fps:.1f} images/s")

# simple drone-speed estimate (change numbers as you like)
GROUND_SWATH_M = 15.0   # ground length covered per frame along flight path
OVERLAP = 0.3           # want 30% overlap between frames

max_speed = GROUND_SWATH_M * fps * (1 - OVERLAP)
print(f"Example max drone speed â‰ˆ {max_speed:.1f} m/s "
      f"for {GROUND_SWATH_M} m/frame and {OVERLAP*100:.0f}% overlap.")















import numpy as np

# ground-truth stress from density (0..1)
low_thr = 0.4                      # you can tune this
gt_stress = (density_norm < low_thr).astype(int)   # 1 = stressed, 0 = ok

def compute_metrics(gt, pred):
    gt = gt.flatten()
    pred = pred.flatten()
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }



# choose suspicion threshold (tune if you like)
thr = 0.6

pred_unf  = (suspicion_unf  >= thr).astype(int)
pred_filt = (suspicion_filt >= thr).astype(int)

metrics_unf  = compute_metrics(gt_stress, pred_unf)
metrics_filt = compute_metrics(gt_stress, pred_filt)

print("WITHOUT filter:", metrics_unf)
print("WITH filter:   ", metrics_filt)



# plant mask already computed when we built img_np_filt:
# mask_plant = (green or yellow)  -> shape H x W

plant_frac = np.zeros((GRID_H, GRID_W), dtype=np.float32)
for i in range(GRID_H):
    for j in range(GRID_W):
        y0 = i * cell_h
        y1 = (i + 1) * cell_h if i < GRID_H - 1 else H
        x0 = j * cell_w
        x1 = (j + 1) * cell_w if j < GRID_W - 1 else W

        tile_mask = mask_plant[y0:y1, x0:x1]
        plant_frac[i, j] = tile_mask.mean()   # fraction of pixels that are plant

# example: tiles with <5% plant -> ignore in later steps
ignore_tiles = plant_frac < 0.05
print("Tiles to ignore (no crop there):", ignore_tiles.sum(), "out of", GRID_H * GRID_W)



import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# reload original image to be safe
img = Image.open(img_path).convert("RGB")
img_np_orig = np.array(img).astype(np.float32) / 255.0
H, W, _ = img_np_orig.shape

# make float masks for plotting
gt_mask   = gt_stress.astype(float)
unf_mask  = pred_unf.astype(float)
filt_mask = pred_filt.astype(float)

plt.figure(figsize=(18, 5))

# 1) Ground-truth stress (from head counts)
plt.subplot(1, 3, 1)
plt.imshow(img_np_orig)
plt.imshow(
    gt_mask,
    cmap="Reds",
    alpha=0.4,
    extent=(0, W, H, 0),
    interpolation="nearest",
)
plt.title("GT stressed tiles (from head count)")
plt.axis("off")

# 2) Predicted stress WITHOUT filter
plt.subplot(1, 3, 2)
plt.imshow(img_np_orig)
plt.imshow(
    unf_mask,
    cmap="Reds",
    alpha=0.4,
    extent=(0, W, H, 0),
    interpolation="nearest",
)
plt.title("Predicted stressed tiles\nWITHOUT color filter")
plt.axis("off")

# 3) Predicted stress WITH filter
plt.subplot(1, 3, 3)
plt.imshow(img_np_orig)
plt.imshow(
    filt_mask,
    cmap="Reds",
    alpha=0.4,
    extent=(0, W, H, 0),
    interpolation="nearest",
)
plt.title("Predicted stressed tiles\nWITH green+yellow filter")
plt.axis("off")

plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

DATA_ROOT = Path("/kaggle/input/global-wheat-detection")
df = pd.read_csv(DATA_ROOT / "train.csv")

# metric helper
def compute_metrics(gt, pred):
    gt = gt.flatten()
    pred = pred.flatten()
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    
    return acc, prec, rec, f1, tp, tn, fp, fn

def process_one_image(image_id, grid_h=10, grid_w=10,
                      stress_thr=0.4, susp_thr=0.6):
    # subset annotations for this image
    df_img = df[df["image_id"] == image_id]
    img_path = DATA_ROOT / "train" / f"{image_id}.jpg"
    
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    H, W, _ = img_np.shape
    
    # parse bbox list
    boxes = []
    for s in df_img["bbox"]:
        x, y, w, h = eval(s)
        boxes.append((x, y, w, h))
    boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
    
    # grid
    GRID_H, GRID_W = grid_h, grid_w
    cell_h = H // GRID_H
    cell_w = W // GRID_W
    
    # head counts
    head_counts = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for x, y, w, h in boxes:
        cx = x + w / 2.0
        cy = y + h / 2.0
        j = min(int(cx / W * GRID_W), GRID_W - 1)
        i = min(int(cy / H * GRID_H), GRID_H - 1)
        head_counts[i, j] += 1.0
    
    if head_counts.max() > 0:
        density_norm = head_counts / head_counts.max()
    else:
        density_norm = head_counts.copy()
    
    # ground-truth stress from density
    gt_stress = (density_norm < stress_thr).astype(int)   # 1 = stressed
    
    # plant mask (green + yellow-ish)
    R = img_np[..., 0]
    G = img_np[..., 1]
    B = img_np[..., 2]
    mask_green  = (G > R) & (G > B) & (G > 0.25)
    mask_yellow = (R > 0.25) & (G > 0.2) & (B < 0.8)
    mask_plant  = mask_green | mask_yellow
    
    # plant fraction per tile
    plant_frac = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for i in range(GRID_H):
        for j in range(GRID_W):
            y0 = i * cell_h
            y1 = (i + 1) * cell_h if i < GRID_H - 1 else H
            x0 = j * cell_w
            x1 = (j + 1) * cell_w if j < GRID_W - 1 else W
            tile_mask = mask_plant[y0:y1, x0:x1]
            plant_frac[i, j] = tile_mask.mean()
    
    ignore_tiles = plant_frac < 0.05   # almost no crop
    
    base_susp = 1.0 - density_norm
    
    # ---------- suspicion WITHOUT filter ----------
    GI_unf = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for i in range(GRID_H):
        for j in range(GRID_W):
            y0 = i * cell_h
            y1 = (i + 1) * cell_h if i < GRID_H - 1 else H
            x0 = j * cell_w
            x1 = (j + 1) * cell_w if j < GRID_W - 1 else W
            tile = img_np[y0:y1, x0:x1, :]
            Rm = tile[..., 0].mean()
            Gm = tile[..., 1].mean()
            Bm = tile[..., 2].mean()
            GI_unf[i, j] = Gm / (Rm + Gm + Bm + 1e-6)
    
    GI_mean_unf = GI_unf.mean()
    GI_diff_unf = np.clip(GI_mean_unf - GI_unf, 0, None)
    GI_norm_unf = GI_diff_unf / (GI_diff_unf.max() + 1e-6)
    
    color_term_unf = GI_norm_unf.copy()
    color_term_unf[density_norm >= 0.5] = 0.0
    
    suspicion_unf = 0.7 * base_susp + 0.3 * color_term_unf
    
    # ---------- suspicion WITH filter ----------
    img_np_filt = img_np.copy()
    img_np_filt[~mask_plant] = 0.0
    
    GI_filt = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for i in range(GRID_H):
        for j in range(GRID_W):
            y0 = i * cell_h
            y1 = (i + 1) * cell_h if i < GRID_H - 1 else H
            x0 = j * cell_w
            x1 = (j + 1) * cell_w if j < GRID_W - 1 else W
            tile = img_np_filt[y0:y1, x0:x1, :]
            Rm = tile[..., 0].mean()
            Gm = tile[..., 1].mean()
            Bm = tile[..., 2].mean()
            GI_filt[i, j] = Gm / (Rm + Gm + Bm + 1e-6)
    
    GI_mean_filt = GI_filt.mean()
    GI_diff_filt = np.clip(GI_mean_filt - GI_filt, 0, None)
    GI_norm_filt = GI_diff_filt / (GI_diff_filt.max() + 1e-6)
    
    color_term_filt = GI_norm_filt.copy()
    color_term_filt[density_norm >= 0.5] = 0.0
    
    suspicion_filt = 0.7 * base_susp + 0.3 * color_term_filt
    
    # predictions from suspicion maps
    pred_unf  = (suspicion_unf  >= susp_thr).astype(int)
    pred_filt = (suspicion_filt >= susp_thr).astype(int)
    
    # metrics
    acc_u, prec_u, rec_u, f1_u, tp_u, tn_u, fp_u, fn_u = compute_metrics(gt_stress, pred_unf)
    acc_f, prec_f, rec_f, f1_f, tp_f, tn_f, fp_f, fn_f = compute_metrics(gt_stress, pred_filt)
    
    # resource usage: how many tiles flagged (and not ignored)
    flagged_unf  = pred_unf.sum()
    flagged_filt = pred_filt.sum()
    flagged_unf_valid  = ((pred_unf == 1)  & (~ignore_tiles)).sum()
    flagged_filt_valid = ((pred_filt == 1) & (~ignore_tiles)).sum()
    
    return {
        "image_id": image_id,
        "acc_unf": acc_u,
        "f1_unf": f1_u,
        "acc_filt": acc_f,
        "f1_filt": f1_f,
        "flagged_unf": int(flagged_unf),
        "flagged_filt": int(flagged_filt),
        "flagged_unf_valid": int(flagged_unf_valid),
        "flagged_filt_valid": int(flagged_filt_valid),
        "ignored_tiles": int(ignore_tiles.sum()),
        "total_tiles": GRID_H * GRID_W,
    }



# run on many images and summarise

# take first N images for a quick test (you can increase N later)
image_ids = df["image_id"].unique()
N = 30   # try 30 or 50 depending on time
image_ids = image_ids[:N]

results = []
for k, iid in enumerate(image_ids, 1):
    print(f"{k}/{N}: {iid}")
    res = process_one_image(iid)
    results.append(res)

res_df = pd.DataFrame(results)
res_df.head()



### VISUAL SUMMARY

import numpy as np
import matplotlib.pyplot as plt

# x-axis: image index
x = np.arange(len(res_df))

plt.figure(figsize=(16, 4))

# 1) Accuracy per image
plt.subplot(1, 3, 1)
plt.plot(x, res_df["acc_unf"], marker="o", linestyle="-", label="no filter")
plt.plot(x, res_df["acc_filt"], marker="x", linestyle="--", label="with filter")
plt.title("Accuracy per image")
plt.xlabel("Image index")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()

# 2) F1-score per image
plt.subplot(1, 3, 2)
plt.plot(x, res_df["f1_unf"], marker="o", linestyle="-", label="no filter")
plt.plot(x, res_df["f1_filt"], marker="x", linestyle="--", label="with filter")
plt.title("F1-score per image")
plt.xlabel("Image index")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.legend()

# 3) Average number of stressed tiles (with crop)
mean_unf_valid  = res_df["flagged_unf_valid"].mean()
mean_filt_valid = res_df["flagged_filt_valid"].mean()

plt.subplot(1, 3, 3)
plt.bar(["no filter", "with filter"], [mean_unf_valid, mean_filt_valid])
plt.title("Avg. stressed tiles with crop\n(per image)")
plt.ylabel("Tiles per image")

plt.tight_layout()
plt.show()



# New training cell: longer training + LR decay + best-model saving
import torch

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0

num_epochs = 12          # was 3
best_acc   = 0.0

for epoch in range(1, num_epochs + 1):
    # simple LR decay after a few epochs
    if epoch == 6:
        for g in optimizer.param_groups:
            g["lr"] *= 0.2   # 1e-3 -> 2e-4

    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(val_loader)

    # track best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(
            {"model_state": model.state_dict(),
             "class_to_idx": full_ds.class_to_idx},
            "leaf_cnn_lcdcd_best.pth",
        )

    print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, val_acc={val_acc:.3f}, best={best_acc:.3f}")

print("Training done. Best val_acc =", best_acc)
print("Best model saved as leaf_cnn_lcdcd_best.pth")



from pathlib import Path

# your dataset root
leaf_base = Path("/kaggle/input/lcdcd-wheat-disease-inceptionresnetv2")
print("leaf_base:", leaf_base)
print("Immediate subfolders:")
for p in leaf_base.iterdir():
    print("  ", p)



import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# We will search for all folders under leaf_base that actually contain images
LEAF_ROOT = leaf_base
print("Using LEAF_ROOT =", LEAF_ROOT)

class LeafDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # find all dirs that contain at least one image file
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        candidate_dirs = []
        for d in root.rglob("*"):
            if d.is_dir():
                has_img = any(
                    (f.is_file() and f.suffix.lower() in img_exts)
                    for f in d.iterdir()
                )
                if has_img:
                    candidate_dirs.append(d)

        candidate_dirs = sorted(candidate_dirs)
        classes = [d.name for d in candidate_dirs]
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        print("Found classes:", self.class_to_idx)

        for d in candidate_dirs:
            label = self.class_to_idx[d.name]
            for img_path in d.glob("*"):
                if img_path.suffix.lower() in img_exts:
                    self.samples.append((img_path, label))

        print("Total images:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

full_ds = LeafDataset(LEAF_ROOT, transform=transform)

# 80/20 train/val split
train_size = int(0.8 * len(full_ds))
val_size   = len(full_ds) - train_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)



#SIMPLE CNN


import torch.nn as nn

num_classes = len(full_ds.class_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, " | num_classes:", num_classes)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64x64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16x16
            nn.AdaptiveAvgPool2d(1),                                      # 1x1
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



from pathlib import Path

root = Path("/kaggle/input")
for p in root.iterdir():
    print(" -", p)





# Cell L1: locate LCDCD 2020 dataset folder and inspect contents
from pathlib import Path

root = Path("/kaggle/input")

print("All folders in /kaggle/input:")
for p in root.iterdir():
    print(" -", p)

leaf_base = None
for p in root.iterdir():
    name = p.name.lower()
    if "lcdcd-2020-dataset" in name:
        leaf_base = p
        break

print("\nSelected leaf_base:", leaf_base)

if leaf_base is not None:
    print("\nSubfolders inside leaf_base:")
    for p in leaf_base.iterdir():
        print("  ", p)
else:
    raise RuntimeError("LCDCD dataset not found. Attach 'lcdcd-2020-dataset' in Add data.")



# Cell L2: build LeafDataset from leaf_base and create train/val loaders
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

LEAF_ROOT = leaf_base
print("Using LEAF_ROOT =", LEAF_ROOT)

class LeafDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        # find all dirs that contain at least one image
        candidate_dirs = []
        for d in root.rglob("*"):
            if d.is_dir():
                has_img = any(
                    (f.is_file() and f.suffix.lower() in img_exts)
                    for f in d.iterdir()
                )
                if has_img:
                    candidate_dirs.append(d)

        candidate_dirs = sorted(candidate_dirs)
        if not candidate_dirs:
            raise RuntimeError(f"No image folders found under {root}")

        classes = [d.name for d in candidate_dirs]
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        print("Found classes:", self.class_to_idx)

        for d in candidate_dirs:
            label = self.class_to_idx[d.name]
            for img_path in d.glob("*"):
                if img_path.suffix.lower() in img_exts:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError("No image files collected. Check dataset structure.")

        print("Total images:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# basic transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

full_ds = LeafDataset(LEAF_ROOT, transform=transform)
print("Total images in dataset:", len(full_ds))

# 80/20 train-val split
train_size = int(0.8 * len(full_ds))
val_size   = len(full_ds) - train_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)



# Cell L2: build LeafDataset from leaf_base and create train/val loaders
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

LEAF_ROOT = leaf_base
print("Using LEAF_ROOT =", LEAF_ROOT)

class LeafDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        # find all dirs that contain at least one image
        candidate_dirs = []
        for d in root.rglob("*"):
            if d.is_dir():
                has_img = any(
                    (f.is_file() and f.suffix.lower() in img_exts)
                    for f in d.iterdir()
                )
                if has_img:
                    candidate_dirs.append(d)

        candidate_dirs = sorted(candidate_dirs)
        if not candidate_dirs:
            raise RuntimeError(f"No image folders found under {root}")

        classes = [d.name for d in candidate_dirs]
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        print("Found classes:", self.class_to_idx)

        for d in candidate_dirs:
            label = self.class_to_idx[d.name]
            for img_path in d.glob("*"):
                if img_path.suffix.lower() in img_exts:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError("No image files collected. Check dataset structure.")

        print("Total images:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# basic transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

full_ds = LeafDataset(LEAF_ROOT, transform=transform)
print("Total images in dataset:", len(full_ds))

# 80/20 train-val split
train_size = int(0.8 * len(full_ds))
val_size   = len(full_ds) - train_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)



# Cell L3: define a small CNN model, loss, optimizer
import torch.nn as nn

num_classes = len(full_ds.class_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| num_classes:", num_classes)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64x64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16x16
            nn.AdaptiveAvgPool2d(1),                                      # 1x1
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



# Cell L4: train for a few epochs and print validation accuracy
'''def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0

num_epochs = 10  # you can increase later

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.3f}")
'''


# New training cell: longer training + LR decay + best-model saving
import torch

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0

num_epochs = 12          # was 3
best_acc   = 0.0

for epoch in range(1, num_epochs + 1):
    # simple LR decay after a few epochs
    if epoch == 6:
        for g in optimizer.param_groups:
            g["lr"] *= 0.2   # 1e-3 -> 2e-4

    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(val_loader)

    # track best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(
            {"model_state": model.state_dict(),
             "class_to_idx": full_ds.class_to_idx},
            "leaf_cnn_lcdcd_best.pth",
        )

    print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, val_acc={val_acc:.3f}, best={best_acc:.3f}")

print("Training done. Best val_acc =", best_acc)
print("Best model saved as leaf_cnn_lcdcd_best.pth")



# Quick sanity-check on a few validation images
import torch, random
import matplotlib.pyplot as plt

# 1) Load best checkpoint (if not already loaded)
ckpt = torch.load("leaf_cnn_lcdcd_best.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
class_to_idx = ckpt.get("class_to_idx", full_ds.class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}
model.eval()

# 2) Sample some images from validation set
n_samples = 8  # change if you want more/less
indices = random.sample(range(len(val_ds)), n_samples)

fig, axes = plt.subplots(2, n_samples // 2, figsize=(3*(n_samples//2), 6))
axes = axes.flatten()

for ax, i in zip(axes, indices):
    img_tensor, label = val_ds[i]           # tensor CxHxW in [0,1]
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # HxWxC for imshow

    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        conf = float(probs[pred_idx])

    true_class = idx_to_class[int(label)]
    pred_class = idx_to_class[pred_idx]

    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"T:{true_class}\nP:{pred_class}\n{conf:.2f}")

plt.tight_layout()
plt.show()



# Confusion matrix and per-class accuracy for the leaf CNN
import torch
import numpy as np

# load best model (if not already loaded)
ckpt = torch.load("leaf_cnn_lcdcd_best.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
class_to_idx = ckpt.get("class_to_idx", full_ds.class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}

num_classes = len(class_to_idx)
model.eval()

# build confusion matrix: rows = true class, cols = predicted class
conf_mat = np.zeros((num_classes, num_classes), dtype=int)

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            conf_mat[t, p] += 1

print("Classes (index -> name):")
for idx, name in sorted(idx_to_class.items()):
    print(f"  {idx}: {name}")

print("\nConfusion matrix (rows=true, cols=pred):")
print(conf_mat)

# per-class accuracy
print("\nPer-class accuracy:")
for c in range(num_classes):
    true_total = conf_mat[c].sum()
    correct    = conf_mat[c, c]
    acc_c = correct / true_total if true_total > 0 else 0.0
    print(f"{idx_to_class[c]}: {acc_c:.3f}")



# Cell R1: define simple "action" for each disease class and demo on some images
import random
import torch
import matplotlib.pyplot as plt

# load best model again (safe)
ckpt = torch.load("leaf_cnn_lcdcd_best.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
class_to_idx = ckpt.get("class_to_idx", full_ds.class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}
model.eval()

# simple text recommendations for each class
action_map = {
    "Healthy Wheat": "No action â€“ area OK, just monitor.",
    "Leaf Rust": "Apply leaf fungicide for rust in this tile.",
    "Crown and Root Rot": "Check soil moisture/drainage; root fungicide treatment.",
    "Wheat Loose Smut": "Treat seed for next season; consider crop rotation.",
}

# pick a few random validation samples
n_samples = 6
indices = random.sample(range(len(val_ds)), n_samples)

fig, axes = plt.subplots(2, n_samples // 2, figsize=(3*(n_samples//2), 6))
axes = axes.flatten()

for ax, i in zip(axes, indices):
    img_tensor, label = val_ds[i]
    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        conf = float(probs[pred_idx])

    true_class = idx_to_class[int(label)]
    pred_class = idx_to_class[pred_idx]
    action = action_map.get(pred_class, "No action defined.")

    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"T:{true_class}\nP:{pred_class}\nConf:{conf:.2f}\n{action}",
        fontsize=8,
    )

plt.tight_layout()
plt.show()



# YOLO-1 (robust): install/import ultralytics and show exact status
from pathlib import Path
import pandas as pd
import numpy as np
import shutil, os
from PIL import Image

print("Trying to import ultralytics...")

try:
    from ultralytics import YOLO
    print("Imported ultralytics successfully (no install needed).")
except ImportError as e:
    print("ImportError, trying to pip install ultralytics...")
    try:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
        from ultralytics import YOLO
        print("Installed and imported ultralytics successfully.")
    except Exception as e2:
        print("Failed to install/import ultralytics.")
        print("Exact error:")
        print(e2)




# YOLO-2: convert part of global-wheat-detection into YOLO format
DATA_ROOT = Path("/kaggle/input/global-wheat-detection")
df = pd.read_csv(DATA_ROOT / "train.csv")

# take a small subset (you can increase N later)
image_ids = df["image_id"].unique()
N = 200
image_ids = image_ids[:N]

# simple train/val split
split = int(0.8 * len(image_ids))
train_ids = set(image_ids[:split])
val_ids   = set(image_ids[split:])

yolo_root = Path("/kaggle/working/yolo_wheat")
img_train_dir = yolo_root / "images" / "train"
img_val_dir   = yolo_root / "images" / "val"
lbl_train_dir = yolo_root / "labels" / "train"
lbl_val_dir   = yolo_root / "labels" / "val"

for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
    d.mkdir(parents=True, exist_ok=True)

def write_yolo_labels(image_id, target_dir_img, target_dir_lbl):
    # copy image
    src_img = DATA_ROOT / "train" / f"{image_id}.jpg"
    dst_img = target_dir_img / f"{image_id}.jpg"
    if not dst_img.exists():
        shutil.copy(str(src_img), str(dst_img))

    # get all bboxes for this image
    df_img = df[df["image_id"] == image_id]

    # open image to get width/height
    with Image.open(src_img) as im:
        W, H = im.size

    # YOLO label file
    lbl_path = target_dir_lbl / f"{image_id}.txt"
    with open(lbl_path, "w") as f:
        for s in df_img["bbox"]:
            x, y, w, h = eval(s)  # [x_min, y_min, w, h] in pixels
            xc = x + w / 2.0
            yc = y + h / 2.0

            # normalize to 0..1
            xc_n = xc / W
            yc_n = yc / H
            w_n  = w  / W
            h_n  = h  / H

            cls_id = 0  # single class: wheat head
            f.write(f"{cls_id} {xc_n:.6f} {yc_n:.6f} {w_n:.6f} {h_n:.6f}\n")

# create YOLO train/val files
for iid in train_ids:
    write_yolo_labels(iid, img_train_dir, lbl_train_dir)

for iid in val_ids:
    write_yolo_labels(iid, img_val_dir, lbl_val_dir)

print("YOLO dataset built at:", yolo_root)
print("Train images:", len(list(img_train_dir.glob('*.jpg'))))
print("Val images:",   len(list(img_val_dir.glob('*.jpg'))))



# YOLO-3: create data.yaml for YOLO training
import yaml

data_yaml = {
    "path": str(yolo_root),         # base path
    "train": "images/train",
    "val":   "images/val",
    "names": {0: "wheat_head"},     # one class
}

with open(yolo_root / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print((yolo_root / "data.yaml").read_text())



# YOLO-4: train a small YOLO model on wheat heads
# "yolov8n.yaml" = tiny YOLOv8 model architecture (no pretrained weights needed)
model_yolo = YOLO("yolov8n.yaml")

results = model_yolo.train(
    data=str(yolo_root / "data.yaml"),
    epochs=5,          # you can increase later
    imgsz=640,
    batch=8,
    project="wheat_yolo_runs",
    name="yolov8n_wheat",
)

print("YOLO training done.")





