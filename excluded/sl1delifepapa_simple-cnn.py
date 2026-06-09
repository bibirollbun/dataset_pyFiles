import os, math, random, glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn
import torch
import torch.nn.functional as F     # To build neural networks
import torch.optim as optim   # Optimization functions
import torchvision
from torchvision.transforms import v2 as transforms   # For the PIL to tensor format
from torch.utils.data.sampler import SubsetRandomSampler # For validation set
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import zipfile
from torchvision.transforms import InterpolationMode

#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)


BASE = Path("/kaggle/input/galaxy-zoo-the-galaxy-challenge")

if not os.path.exists("/kaggle/working/all_ones_benchmark.csv"):
    for file in BASE.glob("*"):
        with zipfile.ZipFile(file) as zip_ref:
            zip_ref.extractall("/kaggle/working/")


BASE = Path("/kaggle/working")
TRAIN_DIR = BASE / "images_training_rev1"
TEST_DIR = BASE / "images_test_rev1"
LABELS_CSV = BASE / "training_solutions_rev1.csv"

assert TRAIN_DIR.exists() and TEST_DIR.exists() and LABELS_CSV.exists(), "Check BASE path."


def list_ids(images_dir):
    ids = []
    for p in images_dir.glob("*.jpg"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            pass
    ids.sort()
    return np.array(ids, dtype=np.int64)

train_ids_all = list_ids(TRAIN_DIR)
test_ids = list_ids(TEST_DIR)

print(f"Got {len(train_ids_all)} training images and {len(test_ids)} test images.")


df = pd.read_csv(LABELS_CSV)
df = df[df.GalaxyID.isin(train_ids_all)].copy()
df.set_index("GalaxyID", inplace=True)
target_cols = [c for c in df.columns]
y_all = df.loc[train_ids_all, target_cols].values.astype("float32")


# ==== TRAIN / VALID SPLIT (last 10% as valid, deterministic) ==================
num_train_total = len(train_ids_all)
num_valid = num_train_total // 10
num_train = num_train_total - num_valid

train_ids = train_ids_all[:num_train]
valid_ids = train_ids_all[num_train:]
y_train   = y_all[:num_train]
y_valid   = y_all[num_train:]

train_indices = np.arange(num_train, dtype=np.int64)
valid_indices = np.arange(num_train, num_train + num_valid, dtype=np.int64)
test_indices  = np.arange(len(test_ids), dtype=np.int64)

print(f"Train: {len(train_ids)}  Valid: {len(valid_ids)}  Test: {len(test_ids)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def id_to_path(images_dir, gid):
    return images_dir / f"{int(gid)}.jpg"

def unnormalize(x):
    return x * 0.5 + 0.5

# Transforms for validation and testing (applied on-the-fly)
transform_normal = transforms.Compose([
    transforms.CenterCrop(207),                         # crop to 207×207
    transforms.Resize((69, 69),                         # downsample ×3
                      interpolation=InterpolationMode.BILINEAR,
                      antialias=True),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Transforms for training (applied during pre-transformation)
transform_train = transforms.Compose([
    transforms.CenterCrop(207), 
    transforms.Resize((69, 69),interpolation=InterpolationMode.BILINEAR,antialias=True),
    transforms.RandomAffine(degrees=180, translate=(0.05, 0.05), scale=(0.9, 1.15), fill=0),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class GalaxyDataset(Dataset):
    def __init__(self, ids, images_dir, labels=None, transform=None, pretransformed_dir=None):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.images_dir = images_dir
        self.labels = labels  # None for test
        self.transform = transform
        self.pretransformed_dir = pretransformed_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        gid = int(self.ids[idx])
        if self.pretransformed_dir:
            img_path = self.pretransformed_dir / f"{gid}.jpg"
            img = Image.open(img_path).convert("RGB")
            img = transforms.ToTensor()(img) # Convert to tensor
            # Normalization was applied during pre-transformation, so no need to apply again
        else:
            img_path = id_to_path(self.images_dir, gid)
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        if self.labels is None:
            return img, gid  # test set: no labels
        target = torch.from_numpy(self.labels[idx].astype(np.float32))
        return img, target


# Define batch_size, pin, and num_workers
batch_size = 512 # Example batch size
pin = (device.type == "cuda")
num_workers = 2 # Example num_workers setting


# Create a new DataLoader for the training data
train_dataset = GalaxyDataset(train_ids, TRAIN_DIR, labels=y_train, transform=transform_train)
valid_dataset = GalaxyDataset(valid_ids, TRAIN_DIR, labels=y_valid, transform=transform_normal)
test_dataset  = GalaxyDataset(test_ids,  TEST_DIR,  labels=None,   transform=transform_normal)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=pin,
    persistent_workers=(num_workers > 0), prefetch_factor=2
)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=pin,
    persistent_workers=(num_workers > 0), prefetch_factor=2
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=pin,
    persistent_workers=(num_workers > 0), prefetch_factor=2
)
print(f"DataLoaders: {len(train_loader)} train batches, {len(valid_loader)} valid batches, {len(test_loader)} test batches.")
print(f"DataLoaders: {len(train_loader)} train batches, {len(valid_loader)} valid batches, {len(test_loader)} test batches.")


# Sanity check
for _ in range(3):
    k = np.random.randint(0, len(train_dataset))
    gid = int(train_dataset.ids[k])
    img, target = train_dataset[k]
    csv_row = df.loc[gid].values.astype(np.float32)
    print(f"GID {gid}  match:", np.allclose(target.numpy(), csv_row))


#Plot items directly from the dataset with highest class on top
nx, ny = 4, 4
fig, ax = plt.subplots(nx, ny, figsize=(10,10))
k = 0
for i in range(nx):
    for j in range(ny):
        img = train_dataset[k][0]          # C,H,W
        img = unnormalize(img).permute(1,2,0)  # H,W,C
        ax[i][j].imshow(img)
        label = train_dataset[k][1].numpy()
        top_class = np.argmax(label)
        second_class = np.argsort(label)[-2]
        img_id = train_dataset.ids[k]
        ax[i][j].set_title(f"ID: {img_id}\nClass1: {top_class}\nClass2: {second_class}", fontsize=8)
        ax[i][j].axis("off")
        k += 1
plt.tight_layout()
plt.show()


#Print pixel size
img = test_dataset[0][0]
print(f"Image size: {img.shape[0]}x{img.shape[1]}x{img.shape[2]} pixels")
print(y_all.shape[1])


#Create simple CNN model

class SimpleCNN(nn.Module):
    def __init__(self, num_outputs):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(32, num_outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x).flatten(1)  # [B, 32]
        x = self.fc(x)              # raw logits
        return x


# Model + AMP + channels_last on CUDA
num_outputs = y_all.shape[1]
model = SimpleCNN(num_outputs).to(device)

def count_parameters(model):
    table = {}
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table[name] = params
        total_params += params
    return table, total_params

# Number of filters in each convolutional layer
print(f"Number of filters in conv1: {model.conv1.out_channels}")
print(f"Number of filters in conv2: {model.conv2.out_channels}")

# Number of parameters per layer
param_table, total_params = count_parameters(model)
print("\nParameters per layer:")
for name, params in param_table.items():
    print(f"{name}: {params}")
print(f"\nTotal parameters: {total_params}")


if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
amp_enabled = (device.type == "cuda")
scaler = torch.amp.GradScaler(enabled=amp_enabled)

start_time = time.time()
num_epochs = 10000
patience = 16
best_vloss = float("inf")
epochs_no_improve = 0
train_loss, valid_loss = [], []
ckpt = {}

for epoch in range(num_epochs):
    # Training loop
    model.train()
    running = 0.0
    i = 0
    for xb, yb in train_loader: # Use the new train_loader
        # if i % 1 == 0:
        #     print(f"Batch {i} of {len(train_loader)}")
        i += 1
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        # Transformations are now pre-applied in the dataset for training data
        # No need to apply transform_train(xb) here

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            out = model(xb)
            loss = criterion(out, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
    train_epoch_loss = running / max(1, len(train_loader))
    train_loss.append(train_epoch_loss)

    # Validation loop
    model.eval()
    vloss_sum, vcount = 0.0, 0
    with torch.no_grad():
        for vx, vy in valid_loader:
            vx = vx.to(device, non_blocking=True)
            if device.type == "cuda":
                vx = vx.contiguous(memory_format=torch.channels_last)
            vy = vy.to(device, non_blocking=True)
            # Keep transformation for validation data
            # vx = transform_normal(vx) # Transformation is handled in the dataset

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                vout = model(vx)
                vloss_sum += criterion(vout, vy).item() * vx.size(0)
                vcount += vx.size(0)
    vepoch_loss = vloss_sum / max(1, vcount)
    valid_loss.append(vepoch_loss)

    scheduler.step(vepoch_loss)
    print(f"Epoch {epoch+1}  Train: {train_epoch_loss:.4f}  Valid: {vepoch_loss:.4f}  LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Time passed: {((time.time()-start_time)/60):.2f}m")

    if vepoch_loss < best_vloss - 1e-4:
        best_vloss = vepoch_loss
        epochs_no_improve = 0
        saved_epoch = epoch
        ckpt["model"] = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

print(f"Done in {time.time() - start_time:.1f}s. Best valid BCE: {best_vloss:.4f}")


#Save model
ckpt2 = {
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict() if amp_enabled else None,
    "epoch": epoch,  # last finished epoch
    "saved_epoch": saved_epoch,
    "train_loss": train_loss,          # list of per-epoch losses
    "valid_loss": valid_loss,
    "best_vloss": best_vloss,
    "patience": patience,
    "batch_size": batch_size,
}

ckpt.update(ckpt2)
torch.save(ckpt, "/kaggle/working/simple_cnn_patience_transform.pth")

