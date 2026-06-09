# ============================================================
# ğŸš€ Smart image-to-CSV matching dataset loader for H&M dataset
# ============================================================

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# -------------- CONFIG -------------------
CSV_PATH = "/kaggle/input/merges-articles-and-transactions/merged_articles_transactions.csv"
IMAGES_ROOT = "/kaggle/input/h-and-m-personalized-fashion-recommendations/images"
# -----------------------------------------

# Known class mappings (from your project definition)
COLOR_CLASSES = ["Black", "White", "Red", "Blue", "Navy", "Grey", "Beige", "Pink", "Green", "Brown"]
PRODUCT_CLASSES = ["T-shirt", "Dress", "Shirt", "Blouse", "Sweater", "Jacket", "Trousers", "Shorts", "Skirt", "Vest Top"]

color_map = {c.lower(): i for i, c in enumerate(COLOR_CLASSES)}
product_map = {p.lower(): i for i, p in enumerate(PRODUCT_CLASSES)}

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Sanity check
if not os.path.exists(IMAGES_ROOT):
    raise FileNotFoundError(f"Images directory not found at: {IMAGES_ROOT}")

print("âœ… Imports and config successful.")



# ------------------------------------------------------------
# Step 1 â€” Read and preprocess the CSV
# ------------------------------------------------------------
print("Reading merged CSV...")
df = pd.read_csv(CSV_PATH, dtype=str)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Ensure required column exists
if "article_id" not in df.columns:
    raise KeyError("Missing required column 'article_id' in CSV")

# Clean article IDs
df["article_id"] = df["article_id"].astype(str).str.strip().str.strip('"').str.strip("'")
df = df.sort_values("article_id").reset_index(drop=True)

print(f"âœ… Loaded {len(df)} rows from CSV.")
print("Unique article_ids:", df["article_id"].nunique())
print("Sample rows:")
display(df.head(3))



# ------------------------------------------------------------
# âœ… Step 2 â€” Map each article_id to its actual image path (with leading zero fix)
# ------------------------------------------------------------
print("Building image path mapping directly from CSV IDs (auto-correcting leading zeros)...")

image_records = []

# Fix dropped leading zeros
df["article_id"] = df["article_id"].astype(str).str.strip().str.strip('"').str.strip("'")
df["article_id_fixed"] = df["article_id"].apply(lambda x: x if x.startswith("0") else "0" + x)

# Get cleaned + sorted IDs
csv_ids = sorted(df["article_id_fixed"].tolist())

for aid in csv_ids:
    folder = aid[:3]  # first three digits â†’ image folder
    folder_path = os.path.join(IMAGES_ROOT, folder)

    if not os.path.isdir(folder_path):
        continue

    found = False
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = os.path.join(folder_path, aid + ext)
        if os.path.exists(img_path):
            image_records.append((aid, img_path))
            found = True
            break

    if not found:
        pass  # skip unmatched IDs silently

print(f"âœ… Successfully matched {len(image_records)} image files out of {len(csv_ids)} total article IDs.")

if len(image_records) == 0:
    raise RuntimeError("â�Œ No matching images found â€” check IMAGES_ROOT and CSV formatting.")

# Build DataFrame and merge (using fixed IDs)
img_df = pd.DataFrame(image_records, columns=["article_id_fixed", "image_path"])
merged_df = pd.merge(img_df, df, on="article_id_fixed", how="left")

print(f"âœ… Final merged dataset: {len(merged_df)} samples")
display(merged_df.head(5))



# ------------------------------------------------------------
# Step 3 â€” Define custom Dataset
# ------------------------------------------------------------
class FashionImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Handle flexible column naming
        prod_col = None
        color_col = None
        for c in ["product_type_name", "product_name", "prod_name"]:
            if c in self.df.columns:
                prod_col = c
                break
        for c in ["colour_group_name", "perceived_colour_master_name", "colour_name"]:
            if c in self.df.columns:
                color_col = c
                break

        prod_raw = str(row.get(prod_col, "")).lower().strip()
        color_raw = str(row.get(color_col, "")).lower().strip()

        prod_label = product_map.get(prod_raw, -1)
        color_label = color_map.get(color_raw, -1)

        return img, prod_label, color_label, row["article_id"]

print("âœ… FashionImageDataset class ready.")



# Ensure all labels are within your defined classes
valid_prod = [p.lower() for p in PRODUCT_CLASSES]
valid_color = [c.lower() for c in COLOR_CLASSES]

before = len(merged_df)

merged_df = merged_df[
    merged_df["product_type_name"].str.lower().isin(valid_prod)
    & merged_df["colour_group_name"].str.lower().isin(valid_color)
]

print(f"âœ… Filtered dataset from {before} â†’ {len(merged_df)} valid entries")


# ------------------------------------------------------------
# Step 4 â€” Setup transforms and dataloaders
# ------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Optional: subsample for faster debugging (comment out for full dataset)
# merged_df = merged_df.sample(5000, random_state=42).reset_index(drop=True)

# Shuffle and split
indices = np.arange(len(merged_df))
np.random.shuffle(indices)
split = int(0.85 * len(indices))
train_df = merged_df.iloc[indices[:split]]
val_df = merged_df.iloc[indices[split:]]

# Create datasets
train_ds = FashionImageDataset(train_df, transform=train_transform)
val_ds = FashionImageDataset(val_df, transform=val_transform)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

print(f"ğŸ“¦ Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")



import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Step 5 â€” Visualize a few samples
# ------------------------------------------------------------
def show_samples(dataset, n=9):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(n):
        img, prod_label, color_label, aid = dataset[i]
        ax = axes[i // 3, i % 3]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 0.229 + 0.485).clip(0, 1)
        ax.imshow(img_np)
        ax.set_title(f"{PRODUCT_CLASSES[prod_label] if prod_label >= 0 else '?'} / "
                     f"{COLOR_CLASSES[color_label] if color_label >= 0 else '?'}\n({aid})",
                     fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

show_samples(train_ds)



# ------------------------------------------------------------
# Step 6 â€” Define Multi-Head CNN Model (ResNet50 Backbone)
# ------------------------------------------------------------
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class ResNetMultiHead(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True,
                 num_product_classes=10, num_color_classes=10, dropout=0.5):
        super().__init__()

        if backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Only ResNet50 is implemented in this version.")

        # Define two separate heads
        self.product_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_product_classes)
        )
        self.color_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_color_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        product_output = self.product_head(features)
        color_output = self.color_head(features)
        return product_output, color_output

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetMultiHead(pretrained=True,
                        num_product_classes=len(PRODUCT_CLASSES),
                        num_color_classes=len(COLOR_CLASSES)).to(device)

print("âœ… Model initialized on device:", device)



# ------------------------------------------------------------
# Step 8 â€” Safer loss and accuracy with masking
# ------------------------------------------------------------
criterion = nn.CrossEntropyLoss(reduction="none")

def compute_masked_loss(outputs, targets):
    # Ignore invalid labels (negative or >= num_classes)
    valid_mask = (targets >= 0) & (targets < outputs.size(1))
    if not valid_mask.any():
        return torch.tensor(0.0, device=targets.device, requires_grad=True)
    losses = criterion(outputs, targets)
    return losses[valid_mask].mean()

def accuracy(outputs, targets):
    valid_mask = (targets >= 0) & (targets < outputs.size(1))
    if not valid_mask.any():
        return 0.0
    preds = outputs.argmax(1)
    correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
    total = valid_mask.sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss, running_acc_p, running_acc_c = 0.0, 0.0, 0.0
    total_p, total_c = 0, 0

    for imgs, prod_labels, color_labels, _ in loader:
        # âœ… Convert to tensors on CPU first (safe)
        if isinstance(prod_labels, torch.Tensor) is False:
            prod_labels = torch.tensor(prod_labels, dtype=torch.long)
        if isinstance(color_labels, torch.Tensor) is False:
            color_labels = torch.tensor(color_labels, dtype=torch.long)

        # âœ… Clamp invalid values (-1 or >class range)
        prod_labels = torch.clamp(prod_labels, 0, len(PRODUCT_CLASSES) - 1)
        color_labels = torch.clamp(color_labels, 0, len(COLOR_CLASSES) - 1)

        imgs = imgs.to(device)
        prod_labels = prod_labels.to(device)
        color_labels = color_labels.to(device)

        optimizer.zero_grad()
        out_prod, out_color = model(imgs)

        loss_p = compute_masked_loss(out_prod, prod_labels)
        loss_c = compute_masked_loss(out_color, color_labels)
        loss = loss_p + loss_c

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_acc_p += accuracy(out_prod, prod_labels) * imgs.size(0)
        running_acc_c += accuracy(out_color, color_labels) * imgs.size(0)
        total_p += imgs.size(0)
        total_c += imgs.size(0)

    return (
        running_loss / len(loader.dataset),
        running_acc_p / total_p,
        running_acc_c / total_c
    )



@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc_prod, total_acc_color = 0.0, 0.0, 0.0
    count = 0

    for imgs, prod_labels, color_labels, _ in loader:
        imgs = imgs.to(device)
        prod_labels = torch.tensor(prod_labels, dtype=torch.long, device=device)
        color_labels = torch.tensor(color_labels, dtype=torch.long, device=device)

        out_prod, out_color = model(imgs)
        loss_p = compute_masked_loss(out_prod, prod_labels)
        loss_c = compute_masked_loss(out_color, color_labels)
        loss = loss_p + loss_c

        total_loss += loss.item()
        total_acc_prod += accuracy(out_prod, prod_labels)
        total_acc_color += accuracy(out_color, color_labels)
        count += 1

    return total_loss / count, total_acc_prod / count, total_acc_color / count



# ------------------------------------------------------------
# Step 8 â€” Define Loss and Optimizer
# ------------------------------------------------------------

import torch.nn as nn
import torch.optim as optim

# CrossEntropy for both heads
criterion = nn.CrossEntropyLoss(reduction="none")

# AdamW optimizer works best with ResNet
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

print("Optimizer and criterion defined successfully.")
print(optimizer)



# ------------------------------------------------------------
# Step 9 â€” Run Training Loop
# ------------------------------------------------------------
num_epochs = 5
best_val_loss = float("inf")
save_path = "best_model.pth"

for epoch in range(num_epochs):
    train_loss, train_acc_p, train_acc_c = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc_p, val_acc_c = evaluate(model, val_loader, device)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  ğŸ”¹ Train Loss: {train_loss:.4f} | Product Acc: {train_acc_p:.2f}% | Color Acc: {train_acc_c:.2f}%")
    print(f"  ğŸ”¸ Val   Loss: {val_loss:.4f} | Product Acc: {val_acc_p:.2f}% | Color Acc: {val_acc_c:.2f}%")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "product_classes": PRODUCT_CLASSES,
            "color_classes": COLOR_CLASSES
        }, save_path)
        print(f"âœ… Saved new best model to {save_path}")

print(f"Training complete. Best validation loss: {best_val_loss:.4f}")



# ------------------------------------------------------------
# Step 10 â€” Convert & Save Model as best_model.h5
# ------------------------------------------------------------
import h5py

# Save standard PyTorch weights
torch.save(model.state_dict(), "best_model.pth")
print("âœ… Saved PyTorch model as best_model.pth")

# Create a small .h5 checkpoint with metadata
with h5py.File("best_model.h5", "w") as f:
    f.attrs["description"] = "ResNet50 multi-head model for H&M product + color classification"
    f.create_dataset("product_classes", data=np.array(PRODUCT_CLASSES, dtype="S"))
    f.create_dataset("color_classes", data=np.array(COLOR_CLASSES, dtype="S"))
print("âœ… Created metadata file: best_model.h5")

!ls -lh best_model.*



# ------------------------------------------------------------
# Step 11 â€” Quick Inference on Random Samples
# ------------------------------------------------------------
import random
import matplotlib.pyplot as plt

model.eval()

samples = random.sample(range(len(val_ds)), 3)
for idx in samples:
    img, prod_label, color_label, aid = val_ds[idx]
    with torch.no_grad():
        out_p, out_c = model(img.unsqueeze(0).to(device))
        pred_prod = PRODUCT_CLASSES[out_p.argmax(1).item()]
        pred_color = COLOR_CLASSES[out_c.argmax(1).item()]

    plt.imshow(img.permute(1, 2, 0).numpy() * 0.25 + 0.5)
    plt.axis("off")
    plt.title(f"Pred: {pred_prod} / {pred_color}\nID: {aid}", fontsize=8)
    plt.show()


