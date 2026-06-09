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


# ─── Cell 1: verify that Kaggle has mounted everything  ─────────────────────────
import os

INPUT_DIR = "/kaggle/input/histopathologic-cancer-detection"
print("Contents of /kaggle/input/histopathologic-cancer-detection/:")
for fname in sorted(os.listdir(INPUT_DIR)):
    print("  ", fname)




# ─── Cell 2: read the CSV of training labels ───────────────────────────────────────────
import pandas as pd

labels_df = pd.read_csv(os.path.join(INPUT_DIR, "train_labels.csv"))
print("Total train_labels rows:", len(labels_df))
print(labels_df["label"].value_counts())
labels_df.head()



# ─── Cell 3: inspect “train/” folder structure and show a few random images ─────────────
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_DIR = os.path.join(INPUT_DIR, "train")
all_train_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".tif")]
print("Number of .tif files in train/:", len(all_train_files))
print("Example filenames:", all_train_files[:5])

# Define a small helper to load a patch directly from train/
def load_patch(train_dir, patch_id):
    """
    patch_id is the 40-char string (without “.tif”). 
    We assume train/ contains exactly files named "<patch_id>.tif".
    """
    full_path = os.path.join(train_dir, patch_id + ".tif")
    img = Image.open(full_path).convert("RGB")
    return img

# Display 4 random train patches (2 positives, 2 negatives)
plt.figure(figsize=(8, 8))
for i in range(4):
    # pick a random row in labels_df 
    idx = random.randint(0, len(labels_df) - 1)
    pid = labels_df.loc[idx, "id"]
    lab = labels_df.loc[idx, "label"]
    patch_img = load_patch(TRAIN_DIR, pid)
    ax = plt.subplot(2, 2, i+1)
    ax.imshow(patch_img)
    ax.set_title(f"id={pid[:8]}…  label={lab}")
    ax.axis("off")

plt.tight_layout()
plt.show()



# ─── Cell 4: same for “test/”—peek at a few test images (unlabeled) ───────────────────
TEST_DIR = os.path.join(INPUT_DIR, "test")
all_test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".tif")]
print("Number of .tif files in test/:", len(all_test_files))
print("First 5 test filenames:", all_test_files[:5])

# Display 4 random “test” patches (though unlabeled, just for sanity check)
plt.figure(figsize=(8, 8))
for i in range(4):
    tfn = random.choice(all_test_files)
    img = Image.open(os.path.join(TEST_DIR, tfn)).convert("RGB")
    ax = plt.subplot(2, 2, i+1)
    ax.imshow(img)
    ax.set_title(f"test/{tfn[:8]}…")
    ax.axis("off")

plt.tight_layout()
plt.show()



# ─── Cell 5: Quick class‐balance bar chart for the train set ────────────────────────────
counts = labels_df["label"].value_counts()
plt.figure(figsize=(4, 4))
plt.bar(["Non-Metastasis (0)", "Metastasis (1)"], counts.values, color=["steelblue","crimson"])
plt.ylabel("Count of Patches")
plt.title("Class Distribution in the 220k-patch Train Set")
for i, v in enumerate(counts.values):
    plt.text(i, v + 2000, str(v), ha="center")
plt.show()



# ─── Cell 6: build a PyTorch Dataset/output DataLoader (since "train/" is already unzipped)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HistopathFolderDataset(Dataset):
    def __init__(self, labels_df, train_folder, transform=None):
        super().__init__()
        self.df = labels_df.reset_index(drop=True)
        self.train_folder = train_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_id = row["id"]
        label = torch.tensor(row["label"], dtype=torch.float32)
        img_path = os.path.join(self.train_folder, patch_id + ".tif")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

import numpy as np
from sklearn.model_selection import train_test_split

nsamp = 2000
rng = np.random.default_rng(seed=42)
indices = rng.choice(len(labels_df), size=nsamp, replace=False)

means = []
stds = []
for i in indices:
    pid = labels_df.loc[i, "id"]
    arr = np.array(
        Image.open(os.path.join(TRAIN_DIR, pid + ".tif")).convert("RGB")
    ).astype(np.float32) / 255.0
    means.append(arr.mean(axis=(0,1)))
    stds.append(arr.std(axis=(0,1)))
means = np.vstack(means)
stds  = np.vstack(stds)
global_mean = means.mean(axis=0).tolist()
global_std  = stds.mean(axis=0).tolist()
print("Global mean:", global_mean)
print("Global std: ", global_std)

train_df, val_df = train_test_split(
    labels_df, test_size=0.20, stratify=labels_df["label"], random_state=42
)

train_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=global_mean, std=global_std),
])
val_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=global_mean, std=global_std),
])

train_ds = HistopathFolderDataset(train_df, TRAIN_DIR, transform=train_transforms)
val_ds   = HistopathFolderDataset(val_df,   TRAIN_DIR, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

print("Train batches per epoch:", len(train_loader))
print("Val   batches per epoch:", len(val_loader))



# ─── Cell 7: Quick EDA / Patch Visualization (using .iloc and skipping missing files) ────────────────────────

import random
import matplotlib.pyplot as plt
from PIL import Image

plt.figure(figsize=(10,4))

shown = 0
attempts = 0
max_attempts = 30    # stop if we can’t find 6 valid files after 30 tries

while shown < 6 and attempts < max_attempts:
    attempts += 1
    pos = random.randint(0, len(train_df) - 1)
    pid = train_df.iloc[pos]["id"]
    lbl = train_df.iloc[pos]["label"]
    img_path = os.path.join(TRAIN_DIR, pid + ".tif")

    if not os.path.isfile(img_path):
        continue

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        continue

    ax = plt.subplot(2, 3, shown + 1)
    ax.imshow(img)
    ax.set_title(f"Label = {lbl}")
    ax.axis("off")

    shown += 1

if shown < 6:
    print(f"Only {shown} valid patches were found after {attempts} attempts.")

plt.suptitle("Six Random Training Patches", fontsize=16)
plt.tight_layout(rect=[0,0,1,0.9])
plt.show()



# ─── Cell 8: Define a Simple CNN Model ─────────────────────────────────────────────

import torch.nn as nn
import torch.optim as optim

class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # → 16×96×96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),                                       # → 16×48×48

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # → 32×48×48
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),                                       # → 32×24×24
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                            # → 32×24×24 = 18432
            nn.Linear(32*24*24, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1)   # output raw logit
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN().to(device)
print("Model architecture:", model)



# ─── Cell 9: Loss, Optimizer, and a Mini‐Batch AUC Helper (on small subsets) ─────

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def compute_subset_auc(model, loader, device, subset_size=512):
    """
    Runs inference on at most `subset_size` samples (not the full loader)
    to get a quick AUC estimate.
    """
    model.eval()
    collected_logits = []
    collected_labels = []
    seen = 0

    with torch.no_grad():
        for images, labels in loader:
            b = images.size(0)
            if seen + b > subset_size:
                take = subset_size - seen
                images = images[:take]
                labels = labels[:take]
                b = take

            images = images.to(device)
            labels = labels.to(device).view(-1)
            logits = model(images).squeeze(1).cpu().numpy()
            collected_logits.append(logits)
            collected_labels.append(labels.cpu().numpy())
            seen += b

            if seen >= subset_size:
                break

    all_logits = np.concatenate(collected_logits, axis=0)
    all_labels = np.concatenate(collected_labels, axis=0)
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    auc = roc_auc_score(all_labels, probs)
    return auc



# ─── Cell 10: Full 5‐Epoch Training Loop ─────────────────────────────────

num_epochs = 5
train_losses      = []
train_subset_aucs = []
val_subset_aucs   = []

for epoch in range(1, num_epochs + 1):
    print(f"\n→ Starting Epoch {epoch}/{num_epochs}")
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        if batch_idx % 500 == 0:
            print(f"   Batch {batch_idx}/{len(train_loader)} →  loss {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"→ Epoch {epoch} done. Avg. loss (full epoch): {epoch_loss:.4f}")

    train_auc = compute_subset_auc(model, train_loader, device, subset_size=1000)
    val_auc   = compute_subset_auc(model, val_loader,   device, subset_size=1000)
    train_subset_aucs.append(train_auc)
    val_subset_aucs.append(val_auc)

    print(f"→ Epoch {epoch} summary: Train Loss = {epoch_loss:.4f}   "
          f"TrainSubset AUC = {train_auc:.4f}   ValSubset AUC = {val_auc:.4f}")

print(f"\n===== Finished {num_epochs} epochs. Best val‐subset AUC ≈ {max(val_subset_aucs):.4f} =====")



# ─── Cell 11: Plot Training Loss and AUC Curves ──────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# Plot training loss over epochs
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o', color='darkblue', label="Train Loss")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Plot subset‐AUC over epochs (train‐subset vs. val‐subset)
plt.subplot(1, 2, 2)
plt.plot(epochs, train_subset_aucs, marker='o', color='forestgreen', label="Train‐subset AUC")
plt.plot(epochs, val_subset_aucs,   marker='o', color='firebrick',  label="Val‐subset AUC")
plt.title("Subset AUC over Epochs")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.ylim(0.5, 1.0)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



# ─── Cell 12: Full Validation Confusion Matrix and Classification Report ────────

from sklearn.metrics import confusion_matrix, classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        logits = model(images).squeeze(1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

cm = confusion_matrix(all_labels, all_preds)
cr = classification_report(all_labels, all_preds, target_names=["Non‐Meta (0)", "Meta (1)"])

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)



# ─── Cell 13: Test‐Time Inference & Create Submission ────────────────────────────

test_ids = [f.replace(".tif","") for f in all_test_files]
submission = []

for image_id in test_ids:
    filename = image_id + ".tif"
    img = Image.open(os.path.join(TEST_DIR, filename)).convert("RGB")
    img_tensor = val_transforms(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logit = model(img_tensor).item()
        prob = 1.0 / (1.0 + np.exp(-logit))
    submission.append((image_id, prob))

sub_df = pd.DataFrame(submission, columns=["id", "label"])
sub_df.to_csv("submission.csv", index=False)
print("Saved submission.csv")



# ─── Cell 14: Discussion / Conclusion Notes ───────────────────────────────────────

**Summary of Work:**

1. **Data Loading & Exploration**  
   - We loaded 220,025 labeled patches (96×96) from `train/` and inspected their class balance.  
   - We visualized several random positive/negative patches.

2. **Data Preparation**  
   - Computed approximate per-channel mean/std on 2,000 random samples for normalization.  
   - Performed an 80/20 stratified train/validation split.  
   - Built a custom `Dataset` and `DataLoader` to read patches directly from disk.

3. **Model Building**  
   - Defined a small CNN (`TinyCNN`) with two convolutional layers plus a two-layer MLP head.  
   - Moved the model to GPU (if available), used `BCEWithLogitsLoss` + Adam optimizer.

4. **Training & Validation**  
   - Trained for **5 epochs**, tracking average training loss, train‐subset AUC, and val‐subset AUC each epoch.  
   - Observed fast convergence:  
     - Epoch 1: Train Loss 0.3746 → Val Subset AUC 0.9225  
     - Epoch 2: Train Loss 0.3593 → Val Subset AUC 0.9299  
     - Epoch 3: Train Loss 0.3453 → Val Subset AUC 0.9367  
     - Epoch 4: Train Loss 0.3311 → Val Subset AUC 0.9402 (**best**)  
     - Epoch 5: Train Loss 0.3206 → Val Subset AUC 0.9393  
   - Final validation accuracy ≈ 0.87, F1‐scores ~0.84–0.89.

5. **Results & Next Steps**  
   - We generated a `submission.csv` on the 57k‐image test set.  
   - **Next improvements**: experiment with deeper architectures (ResNet50, EfficientNet), add more augmentations, fine‐tune a pretrained model, and run longer/more epochs.

**Conclusion:**  
Our TinyCNN achieved a validation‐subset AUC ≈ 0.9402 after 4 epochs. This baseline can be improved by adopting transfer learning (e.g., ResNet50) or stronger augmentation. The full pipeline—from raw `train/` patches → model training → submission generation—is contained in this notebook.

---


