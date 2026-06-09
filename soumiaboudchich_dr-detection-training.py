from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, random, numpy as np, pandas as pd
from glob import glob
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score
import time, copy
from tqdm import tqdm
##Phase 1
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

##Phase 2
import torch.nn as nn
from torchvision import models
import torch.optim as optim



def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


file_lbl="/kaggle/input/diabetic-retinopathy-detection/trainLabels.csv.zip"
df_lbl=pd.read_csv(file_lbl,sep=',')
df_lbl.head()


# list all train and test images
paths_train = sorted(glob('/kaggle/input/diabetic-retinopathy-train-unzipped/train/*.jpeg'))
paths_test  = sorted(glob('/kaggle/input/diabetic-retinopathy-test-unzipped/test/*.jpeg'))

print(len(paths_train))
print(len(paths_test))


img_bgr = cv2.imread(paths_train[0])
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb); plt.axis('off')


img_bgr = cv2.imread(paths_train[35125])
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb); plt.axis('off')


img_bgr = cv2.imread(paths_test[0])
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb); plt.axis('off')


stem_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in paths_train}

df = df_lbl.copy()
df["path"] = df["image"].map(stem_to_path)

print("Total rows in labels:", len(df))
df.head(2)


df["patient_id"] = df["image"].str.split("_").str[0].astype(str)
df["level"] = df["level"].astype(int)

print("Label counts:", df["level"].value_counts().sort_index().to_dict())
df.head()


N_SPLITS = 5
FOLD_IDX = 0  # which fold to use as validation

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(sgkf.split(df, y=df["level"], groups=df["patient_id"]))
train_idx, val_idx = splits[FOLD_IDX]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx].reset_index(drop=True)



# safety: no patient overlap
assert set(df_train.patient_id) & set(df_val.patient_id) == set()


def show_dist(name, s):
    c = s.value_counts().sort_index()
    r = (c / c.sum()).round(4).to_dict()
    print(f"{name} counts:", c.to_dict())
    print(f"{name} ratios:", r)

show_dist("TRAIN", df_train["level"])
show_dist("VAL  ", df_val["level"])


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crop dark borders using a grayscale mask.
    Always returns a NumPy array.
    """
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask = gray > tol
    if not np.any(mask):
        return img  # return original NumPy image

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1]



# We will need to make all images the same size . we will start with the size 224 x 224
IMG_SIZE = 224


# Save preprocessed images to new path
out_dir = "/kaggle/working/processed"
os.makedirs(out_dir, exist_ok=True)

def preprocess_and_save(df, out_dir):
    new_paths = []

    os.makedirs(out_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        path = row["path"]
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)  # convert to NumPy array first

        img_np = crop_image_from_gray(img_np, tol=7) 
        img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)

        img_pil = Image.fromarray(img_np)  #convert back to PIL

        # save
        filename = os.path.basename(path)
        save_path = os.path.join(out_dir, filename)
        img_pil.save(save_path, format="JPEG", quality=95)

        new_paths.append(save_path)

    df = df.copy()
    df["proc_path"] = new_paths
    return df


#df_train_subset = df_train.sample(8000, random_state=42)
#df_train_subset = preprocess_and_save(df_train_subset, "/kaggle/working/processed/train_test")
#df_val   = preprocess_and_save(df_val,   "/kaggle/working/processed/val")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


CROP_BEFORE_TRANSFORM = True


class DRDatasetLite(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["proc_path"]).convert("RGB")
        x = self.tfm(img)
        y = int(row["level"])
        return x, y



sample_img, sample_label = DRDataset(df_train, train_tf)[0]
print(sample_img.shape, sample_label)


NUM_WORKERS = min(8, os.cpu_count())
BATCH_SIZE = 64

train_ds = DRDatasetLite(df_train_subset, train_tf)
val_ds   = DRDatasetLite(df_val,   val_tf)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2, drop_last=True
)

val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2
)



# We have 5 classes in this competition: 0,1,2,3,4
NUM_CLASSES = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 1) Create a ResNet-18 backbone
def build_resnet18(num_classes=NUM_CLASSES):
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet18(weights=None)
    
    # Replace final classification layer
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    
    return model.to(device)


model = build_resnet18()
print(next(model.parameters()).device)


dummy = torch.randn(1, 3, 1024, 1024).to(device)
out = model(dummy)


# Class weights to handle imbalanc
class_counts = df_train["level"].value_counts().sort_index()
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * len(class_counts)

# send to tensor 
class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32, device=device)


criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


EPOCHS = 4
best_qwk = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    print(f"\n Epoch {epoch}/{EPOCHS}")
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    # ---- VALIDATE ----
    model.eval()
    val_loss_sum = 0.0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss_sum += loss.item() * xb.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(yb.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    val_qwk = qwk(y_true, y_pred)

    # ---- LOG RESULTS ----
    duration = time.time() - t0
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val   Loss: {val_loss:.4f}")
    print(f"Val   QWK : {val_qwk:.4f}")
    print(f"Time taken: {duration:.1f} seconds")

    # ---- SAVE BEST MODEL ----
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "/kaggle/working/best_resnet18_224.pth")
        print("Saved new best model!")

print(f"\n Training complete. Best Val QWK: {best_qwk:.4f}")



def plot_originals(df, per_class=6, thumb_size=224):
    levels = [0, 1, 2, 3, 4]
    cols = len(levels)
    rows = per_class

    plt.figure(figsize=(cols * 3, rows * 3))

    for c, L in enumerate(levels):
        idxs = df.index[df["level"] == L].tolist()
        if len(idxs) == 0:
            continue
        choose = np.random.choice(idxs, size=min(rows, len(idxs)), replace=False)

        for r, idx in enumerate(choose):
            path = df.loc[idx, "path"]
            img = Image.open(path).convert("RGB")
            img = img.resize((thumb_size, thumb_size))

            ax = plt.subplot(rows, cols, r * cols + (c + 1))
            ax.imshow(img)
            if r == 0:
                ax.set_title(f"Level {L}", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


plot_originals(df_train_subset, per_class=6, thumb_size=224)


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        gray = img
    else:  # assume color HxWxC
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask = gray > tol
    if not np.any(mask):
        return img  # too dark â†’ nothing to crop

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1]



def load_enhace_color(img_rgb: np.ndarray, sigma: float = 10) -> np.ndarray:
    imgf = img_rgb.astype(np.float32)
    blur = cv2.GaussianBlur(imgf, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(imgf, 4.0, blur, -4.0, 128.0)
    return np.clip(out, 0, 255).astype(np.uint8)


NUM_SAMP = 7
SEED = 42

classes = sorted(df_train["level"].unique())
fig = plt.figure(figsize=(25, 16))

for row_i, class_id in enumerate(classes):
    
    rows_cls = df_train.loc[df_train["level"] == class_id]
    sample_rows = rows_cls.sample(min(NUM_SAMP, len(rows_cls)), random_state=SEED)

    for col_i, (idx, row) in enumerate(sample_rows.iterrows()):
        # compute subplot position using row_i (not class_id)
        ax = fig.add_subplot(len(classes), NUM_SAMP, row_i * NUM_SAMP + col_i + 1)
        ax.set_xticks([]); ax.set_yticks([])

        # Get path
        if "path" in row:
            path = row["path"]
        else:
            path = f"{BASE_DIR}/{row['id_code']}.png"

        # preprocess -> returns PIL.Image in your function
        pil_img = preprocess_base(path, out_size=224, sigma=10)

        # show on the axis
        ax.imshow(np.array(pil_img))
        ax.set_title(f"Class {class_id} | idx {idx}", fontsize=9)

plt.tight_layout()
plt.show()



import shutil
import os

# List of folders to remove
folders_to_remove = [
    "/kaggle/working/processed/train_gray",
    "/kaggle/working/processed/val_gray",
    "/kaggle/working/processed/train_test",
    "/kaggle/working/processed/val",           # optional, if you want to remove all
    "/kaggle/working/processed/train",
    "/kaggle/working/processed/train_Color",
    "/kaggle/working/processed/train_circle",
    "/kaggle/working/processed/val_Color",
    "/kaggle/working/processed/val_circle"
]

for folder in folders_to_remove:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"âœ… Removed: {folder}")
    else:
        print(f"âš ï¸� Not found (already removed?): {folder}")



def preprocess_gray_and_save(df, out_dir, out_size=224, tol=7):
    os.makedirs(out_dir, exist_ok=True)
    new_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing GRAY"):
        path = row["path"]

        img = Image.open(path).convert("L")  # Grayscale
        img_np = np.array(img)

        img_np = crop_image_from_gray(img_np, tol=tol)
        img_np = cv2.resize(img_np, (out_size, out_size), interpolation=cv2.INTER_AREA)
        img_np = np.stack([img_np]*3, axis=-1)  # Convert to 3-channel gray

        img_pil = Image.fromarray(img_np)
        filename = os.path.basename(path)
        save_path = os.path.join(out_dir, filename)
        img_pil.save(save_path, format="JPEG", quality=95)

        new_paths.append(save_path)

    df = df.copy()
    df["proc_path"] = new_paths
    return df



df_train_subset = df_train.sample(8000, random_state=42)

df_train_gray = preprocess_gray_and_save(df_train_subset, "/kaggle/working/processed/train_gray")
df_val_gray   = preprocess_gray_and_save(df_val,       "/kaggle/working/processed/val_gray")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf_gray = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_tf_gray = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class DRDatasetLite(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["proc_path"]).convert("RGB")  # already 3-channel gray
        x = self.tfm(img)
        y = int(row["level"])
        return x, y



NUM_WORKERS = min(8, os.cpu_count())
BATCH_SIZE = 64

train_ds = DRDatasetLite(df_train_gray, train_tf_gray)
val_ds   = DRDatasetLite(df_val_gray, val_tf_gray)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2, drop_last=True
)

val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2
)


EPOCHS = 4
best_qwk = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    # ---- VALIDATE ----
    model.eval()
    val_loss_sum = 0.0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * xb.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(yb.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    val_qwk = qwk(y_true, y_pred)

    # ---- LOG RESULTS ----
    duration = time.time() - t0
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val   Loss: {val_loss:.4f}")
    print(f"Val   QWK : {val_qwk:.4f}")
    print(f"Time taken: {duration:.1f} sec")

    # ---- SAVE BEST MODEL ----
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "/kaggle/working/best_resnet18_gray.pth")
        print("âœ… Saved new best model!")

print(f"\n Training complete. Best Val QWK: {best_qwk:.4f}")


def preprocess_and_save_color(df, out_dir, out_size=224, sigma=10):
    os.makedirs(out_dir, exist_ok=True)
    new_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Color"):
        path = row["path"]
        img_bgr = cv2.imread(path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img, tol=7)
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
        img = load_enhace_color(img, sigma=30)
        img_pil = Image.fromarray(img)

        save_path = os.path.join(out_dir, os.path.basename(path))
        img_pil.save(save_path, format="JPEG", quality=95)
        new_paths.append(save_path)

    df = df.copy()
    df["proc_path"] = new_paths
    return df



df_train_subset = df_train.sample(8000, random_state=42)

df_train_Color = preprocess_and_save_color(df_train_subset, "/kaggle/working/processed/train_Color")
df_val_Color  = preprocess_and_save_color(df_val,       "/kaggle/working/processed/val_Color")


from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf_color = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_tf_color = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])



from torch.utils.data import Dataset
from PIL import Image

class DRDatasetLite(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["proc_path"]).convert("RGB")
        x = self.tfm(img)
        y = int(row["level"])
        return x, y



NUM_WORKERS = min(8, os.cpu_count())
BATCH_SIZE = 64

train_ds = DRDatasetLite(df_train_Color, train_tf_color)
val_ds   = DRDatasetLite(df_val_Color, val_tf_color)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2, drop_last=True
)

val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2
)


EPOCHS = 4
best_qwk = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    print(f"\n Epoch {epoch}/{EPOCHS}")
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    # ---- VALIDATE ----
    model.eval()
    val_loss_sum = 0.0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(yb.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    val_qwk = qwk(y_true, y_pred)

    duration = time.time() - t0
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val   Loss: {val_loss:.4f}")
    print(f"Val   QWK : {val_qwk:.4f}")
    print(f"Time taken: {duration:.1f} seconds")

    if val_qwk > best_qwk:
        best_qwk = val_qwk
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "/kaggle/working/best_resnet18_color.pth")
        print("Saved new best model!")

print(f"\nTraining complete. Best Val QWK: {best_qwk:.4f}")



def circle_crop(path: str, out_size=224, sigmaX=10) -> np.ndarray:
    """
    Perform circular crop + enhancement.
    """
    img = cv2.imread(path)
    img = crop_image_from_gray(img)  # remove borders
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    x, y = w // 2, h // 2
    r = min(x, y)

    # Create circular mask and apply
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), r, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)

    img = crop_image_from_gray(img)  # final border crop
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # Enhance contrast with unsharp masking
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX)
    img = cv2.addWeighted(img, 4, blur, -4, 128)
    return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_and_save_circle(df, out_dir, out_size=224, sigma=30):
    os.makedirs(out_dir, exist_ok=True)
    new_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Circular"):
        path = row["path"]
        img_np = circle_crop(path, out_size=out_size, sigmaX=sigma)
        img_pil = Image.fromarray(img_np)

        save_path = os.path.join(out_dir, os.path.basename(path))
        img_pil.save(save_path, format="JPEG", quality=95)
        new_paths.append(save_path)

    df = df.copy()
    df["proc_path"] = new_paths
    return df


df_train_subset = df_train.sample(8000, random_state=42)
df_train_circle = preprocess_and_save_circle(df_train_subset, "/kaggle/working/processed/train_circle")
df_val_circle   = preprocess_and_save_circle(df_val, "/kaggle/working/processed/val_circle")


def plot_processed_by_level(df, per_class=5, label_col='level', path_col='proc_path'):
    levels = sorted(df[label_col].unique())
    cols = len(levels)
    rows = per_class

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))

    for col, level in enumerate(levels):
        subset = df[df[label_col] == level].sample(per_class, random_state=42)
        for row, (_, r) in enumerate(subset.iterrows()):
            ax = axes[row, col] if rows > 1 else axes[col]
            img = Image.open(r[path_col])
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Level {level}", fontsize=12)

    plt.tight_layout()
    plt.show()


plot_processed_by_level(df_train_circle)


train_tf_circle = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf_circle = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class DRDatasetLite(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["proc_path"]).convert("RGB")
        x = self.tfm(img)
        y = int(row["level"])
        return x, y


train_ds = DRDatasetLite(df_train_circle, train_tf_circle)
val_ds   = DRDatasetLite(df_val_circle, val_tf_circle)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4,
                          pin_memory=True, persistent_workers=True, prefetch_factor=2)

val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)



EPOCHS = 4
best_qwk = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    print(f"\n Epoch {epoch}/{EPOCHS}")
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    # ---- VALIDATE ----
    model.eval()
    val_loss_sum = 0.0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss_sum += loss.item() * xb.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(yb.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    val_qwk = qwk(y_true, y_pred)

    # ---- LOG RESULTS ----
    duration = time.time() - t0
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val   Loss: {val_loss:.4f}")
    print(f"Val   QWK : {val_qwk:.4f}")
    print(f"Time taken: {duration:.1f} seconds")

    # ---- SAVE BEST MODEL ----
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "/kaggle/working/best_resnet18_circle.pth")
        print("âœ… Saved new best model!")

print(f"\nâœ… Training complete. Best Val QWK: {best_qwk:.4f}")



def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        gray = img
    else:  # assume color HxWxC
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask = gray > tol
    if not np.any(mask):
        return img  # too dark â†’ nothing to crop

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1]

def load_enhace_color(img_rgb: np.ndarray, sigma: float = 10) -> np.ndarray:
    imgf = img_rgb.astype(np.float32)
    blur = cv2.GaussianBlur(imgf, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(imgf, 4.0, blur, -4.0, 128.0)
    return np.clip(out, 0, 255).astype(np.uint8)

def circle_crop(path: str, out_size=224, sigmaX=10) -> np.ndarray:
    """
    Perform circular crop + enhancement.
    """
    img = cv2.imread(path)
    img = crop_image_from_gray(img)  # remove borders
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    x, y = w // 2, h // 2
    r = min(x, y)

    # Create circular mask and apply
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), r, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)

    img = crop_image_from_gray(img)  # final border crop
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # Enhance contrast with unsharp masking
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX)
    img = cv2.addWeighted(img, 4, blur, -4, 128)
    return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_and_save_circle(df, out_dir, out_size=448, sigma=30):
    os.makedirs(out_dir, exist_ok=True)
    new_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Circular"):
        path = row["path"]
        img_np = circle_crop(path, out_size=out_size, sigmaX=sigma)
        img_pil = Image.fromarray(img_np)

        save_path = os.path.join(out_dir, os.path.basename(path))
        img_pil.save(save_path, format="JPEG", quality=95)
        new_paths.append(save_path)

    df = df.copy()
    df["proc_path"] = new_paths
    return df


df_train_subset = df_train.sample(8000, random_state=42)
#df_val_subset = df_val.sample(3000, random_state=42)
df_train_448 = preprocess_and_save_circle(df_train_subset, "/kaggle/working/processed/train_circle_448", out_size=448)
#df_val_448   = preprocess_and_save_circle(df_val_subset, "/kaggle/working/processed/val_circle_448", out_size=448)


#df_train_subset = df_train.sample(8000, random_state=42)
df_val_subset = df_val.sample(3000, random_state=42)
#df_train_448 = preprocess_and_save_circle(df_train_subset, "/kaggle/working/processed/train_circle_448", out_size=448)
df_val_448   = preprocess_and_save_circle(df_val_subset, "/kaggle/working/processed/val_circle_448", out_size=448)


def show_dist(name, s):
    c = s.value_counts().sort_index()
    r = (c / c.sum()).round(4).to_dict()
    print(f"{name} counts:", c.to_dict())
    print(f"{name} ratios:", r)

show_dist("TRAIN", df_train_448["level"])
show_dist("VAL  ", df_val_448["level"])


plot_processed_by_level(df_train_448)


IMG_SIZE = 448
#BATCH_SIZE = 32  # reduce if needed due to memory
BATCH_SIZE = 16  #test for ResNet-50
NUM_WORKERS = min(8, os.cpu_count())

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class DRDatasetLite(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["proc_path"]).convert("RGB")
        x = self.tfm(img)
        y = int(row["level"])
        return x, y



train_ds = DRDatasetLite(df_train_448, train_tf)
val_ds   = DRDatasetLite(df_val_448, val_tf)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2, drop_last=True
)

val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True, prefetch_factor=2
)



def build_resnet18(num_classes=5):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model.to(device)
model = build_resnet18(num_classes=5)
model.load_state_dict(torch.load("/kaggle/input/model_qwk_0.5/pytorch/default/1/best_resnet18_circle.pth"))
print("âœ… Loaded 224x224 weights!")


#for name, param in model.named_parameters():
#    if "fc" not in name:
#        param.requires_grad = False

#Unfreeze everything later (e.g. after 1 epoch)
for param in model.parameters():
    param.requires_grad = True



class_counts = df_train_448["level"].value_counts().sort_index()
weights = 1.0 / (class_counts + 1e-6)
weights = weights / weights.sum() * len(class_counts)

class_weights_tensor = torch.tensor(weights.values, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)



EPOCHS = 4
best_qwk = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    print(f"\nğŸ“˜ Epoch {epoch}/{EPOCHS}")
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    # ---- VALIDATE ----
    model.eval()
    val_loss_sum = 0.0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(yb.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    val_qwk = qwk(y_true, y_pred)

    # ---- LOG RESULTS ----
    duration = time.time() - t0
    print(f"âœ… Train Loss: {train_loss:.4f}")
    print(f"âœ… Val   Loss: {val_loss:.4f}")
    print(f"âœ… Val   QWK : {val_qwk:.4f}")
    print(f"â�±ï¸�  Time taken: {duration:.1f} seconds")

    # ---- SAVE BEST ----
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "/kaggle/working/best_resnet18_448.pth")
        print("ğŸ’¾ Saved new best model!")

print(f"\nğŸ�� Training complete. Best Val QWK: {best_qwk:.4f}")



# Cell 1: Build ResNet-50
from torchvision import models
import torch.nn as nn

def build_resnet50(num_classes=5):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model.to(device)



model = build_resnet50(num_classes=5)


class_counts = df_train_448["level"].value_counts().sort_index()
weights = 1.0 / (class_counts + 1e-6)
weights = weights / weights.sum() * len(class_counts)

class_weights_tensor = torch.tensor(weights.values, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)



EPOCHS = 4
best_qwk = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    print(f"\n Epoch {epoch}/{EPOCHS}")
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    # ---- VALIDATE ----
    model.eval()
    val_loss_sum = 0.0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(yb.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    val_qwk = qwk(y_true, y_pred)

    # ---- LOG RESULTS ----
    duration = time.time() - t0
    print(f" Train Loss: {train_loss:.4f}")
    print(f" Val   Loss: {val_loss:.4f}")
    print(f" Val   QWK : {val_qwk:.4f}")
    print(f" Time taken: {duration:.1f} seconds")

    # ---- SAVE BEST ----
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "/kaggle/working/best_resnet50_448.pth")
        print("Saved new best model!")

print(f"\n Training complete. Best Val QWK: {best_qwk:.4f}")



def preprocess_and_save_circle(df, out_dir, out_size=768, sigma=30):
    os.makedirs(out_dir, exist_ok=True)
    new_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing Circular"):
        path = row["path"]
        img_np = circle_crop(path, out_size=out_size, sigmaX=sigma)
        img_pil = Image.fromarray(img_np)

        save_path = os.path.join(out_dir, os.path.basename(path))
        img_pil.save(save_path, format="JPEG", quality=95)
        new_paths.append(save_path)

    df = df.copy()
    df["proc_path"] = new_paths
    return df


df_train_subset = df_train.sample(8000, random_state=42)

df_train_768 = preprocess_and_save_circle(df_train_subset, "/kaggle/working/processed/train_768", out_size=768)



df_val_subset = df_val.sample(1000, random_state=42)
df_val_768   = preprocess_and_save_circle(df_val, "/kaggle/working/processed/val_768", out_size=768)


def plot_processed_by_level(df, per_class=5, label_col='level', path_col='proc_path'):
    levels = sorted(df[label_col].unique())
    cols = len(levels)
    rows = per_class

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))

    for col, level in enumerate(levels):
        subset = df[df[label_col] == level].sample(per_class, random_state=42)
        for row, (_, r) in enumerate(subset.iterrows()):
            ax = axes[row, col] if rows > 1 else axes[col]
            img = Image.open(r[path_col])
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Level {level}", fontsize=12)

    plt.tight_layout()
    plt.show()


plot_processed_by_level(df_train_768)


IMG_SIZE = 448
#BATCH_SIZE = 32  # reduce if needed due to memory
BATCH_SIZE = 16  #test for ResNet-50
NUM_WORKERS = min(8, os.cpu_count())

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20, fill=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class DRDatasetLite(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["proc_path"]).convert("RGB")
        x = self.tfm(img)
        y = int(row["level"])
        return x, y



train_ds_768 = DRDatasetLite(df_train_768, train_tf)
val_ds_768   = DRDatasetLite(df_val_768, val_tf)

train_loader = DataLoader(train_ds_768, batch_size=16, shuffle=True,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

val_loader = DataLoader(val_ds_768, batch_size=16, shuffle=False,
                            num_workers=2, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)



class_counts = df_train_768["level"].value_counts().sort_index()
weights = 1.0 / (class_counts + 1e-6)
weights = weights / weights.sum() * len(class_counts)

class_weights_tensor = torch.tensor(weights.values, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)



def build_resnet18(num_classes=5):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model.to(device)

# Load pre-trained model from 448 phase
model = build_resnet18(num_classes=5)
model.load_state_dict(torch.load("/kaggle/input/best_resnet18_qwk_0.58/pytorch/default/1/best_resnet18_448.pth"))
print("Loaded ResNet-18 448x448 weights!")



EPOCHS = 4
best_qwk = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    print(f"\n Epoch {epoch}/{EPOCHS}")
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    # ---- VALIDATE ----
    model.eval()
    val_loss_sum = 0.0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(yb.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    val_qwk = qwk(y_true, y_pred)

    # ---- LOG RESULTS ----
    duration = time.time() - t0
    print(f" Train Loss: {train_loss:.4f}")
    print(f" Val   Loss: {val_loss:.4f}")
    print(f" Val   QWK : {val_qwk:.4f}")
    print(f" Time taken: {duration:.1f} seconds")

    # ---- SAVE BEST ----
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "/kaggle/working/best_resnet18-768.pth")
        print("Saved new best model!")

print(f"\n Training complete. Best Val QWK: {best_qwk:.4f}")



import gc
import torch

gc.collect()                      # Python garbage collection
torch.cuda.empty_cache()          # Releases unused memory back to the GPU
torch.cuda.ipc_collect()          # Optional: Collects interprocess references

# Delete large objects
for var in ['model', 'optimizer', 'train_loader', 'val_loader']:
    if var in globals():
        del globals()[var]

gc.collect()
torch.cuda.empty_cache()




def clean_gpu():
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("âœ… GPU memory cleaned")

clean_gpu()



def build_resnet50(num_classes=5):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model.to(device)

# Load pre-trained model from 448 phase
model = build_resnet50(num_classes=5)
model.load_state_dict(torch.load("/kaggle/input/resnet50_448_qwk_0.619/pytorch/default/1/best_resnet50_448_QWK_0.619.pth"))
print("Loaded ResNet-50 448x448 weights!")



EPOCHS = 4
best_qwk = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    print(f"\n Epoch {epoch}/{EPOCHS}")
    t0 = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * xb.size(0)

    train_loss = train_loss_sum / len(train_loader.dataset)

    # ---- VALIDATE ----
    model.eval()
    val_loss_sum = 0.0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(yb.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_gts)
    val_qwk = qwk(y_true, y_pred)

    # ---- LOG RESULTS ----
    duration = time.time() - t0
    print(f" Train Loss: {train_loss:.4f}")
    print(f" Val   Loss: {val_loss:.4f}")
    print(f" Val   QWK : {val_qwk:.4f}")
    print(f" Time taken: {duration:.1f} seconds")

    # ---- SAVE BEST ----
    if val_qwk > best_qwk:
        best_qwk = val_qwk
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "/kaggle/working/best_resnet50-768.pth")
        print("Saved new best model!")

print(f"\n Training complete. Best Val QWK: {best_qwk:.4f}")


