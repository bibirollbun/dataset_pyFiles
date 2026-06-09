#################################
# 1. IMPORTS & CONFIG
#################################
import os
import gc
import cv2
import torch
import random
import numpy as np
import pandas as pd

from torchvision import models

from glob import glob
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision
import torchvision.transforms as T
from sklearn.metrics import f1_score
from PIL import Image  # Ù�ÙŠ Ø¨Ø¯Ø§ÙŠØ© Ø§Ù„Ù…Ù„Ù�
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")


# Ø§Ù„ØªØ­Ù‚Ù‚ Ù…Ù† ØªÙˆÙ�Ø± Ø§Ù„Ù€ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("ğŸ–¥ï¸� Using device:", device)

# Ø¥Ø°Ø§ ÙƒØ§Ù† Ù‡Ù†Ø§Ùƒ GPU Ù…ØªØ§Ø­ØŒ Ø§Ø·Ø¨Ø¹ Ø§Ø³Ù…Ù‡
if device.type == "cuda":
    print("âœ… GPU name:", torch.cuda.get_device_name(0))
    print("âœ… GPU memory allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB")
    print("âœ… GPU memory reserved:", round(torch.cuda.memory_reserved(0) / 1024**3, 2), "GB")
else:
    print("âš ï¸� GPU not available. Using CPU instead.")


#################################
# 2. BASIC SETTINGS
#################################
class CFG:
    seed = 42
    n_splits = 5          # Ù†Ù�Ø¨Ù‚ÙŠÙ‡ ÙƒÙ…Ø§ Ù‡Ùˆ Ù„Ø£Ù†Ùƒ ØªØ³ØªØ®Ø¯Ù… K-Fold
    epochs = 10            # ØªÙ‚Ù„ÙŠÙ„ Ø¹Ø¯Ø¯ Ø§Ù„Ù€ Epochs Ù„ØªØ³Ø±ÙŠØ¹ Ø§Ù„ØªØ¬Ø±Ø¨Ø© (3 Ù�Ù‚Ø· ÙƒØ¨Ø¯Ø§ÙŠØ©)
    train_bs = 8          # ØªÙ‚Ù„ÙŠÙ„ Batch Size Ù„ÙŠØªÙ†Ø§Ø³Ø¨ Ù…Ø¹ 100 ØµÙˆØ±Ø© (ÙˆÙŠÙˆÙ�Ø± Ø§Ù„Ø°Ø§ÙƒØ±Ø©)
    valid_bs = 16         # Ù†Ù�Ø³ Ø§Ù„Ø³Ø¨Ø¨ â€” Ù„Ø§ Ø­Ø§Ø¬Ø© Ù„Ø­Ø¬Ù… ÙƒØ¨ÙŠØ±
    lr = 1e-3             # Ù†ØªØ±ÙƒÙ‡ ÙƒÙ…Ø§ Ù‡Ùˆ (Ù…Ø¹Ø¯Ù„ ØªØ¹Ù„Ù… Ø«Ø§Ø¨Øª ÙƒØ¨Ø¯Ø§ÙŠØ©)
    num_workers = 0       # Ù„ØªÙ�Ø§Ø¯ÙŠ Ù…Ø´Ø§ÙƒÙ„ ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¹Ù„Ù‰ Google Colab
    img_size = 224        # Ù†ØªØ±ÙƒÙ‡ ÙƒÙ…Ø§ Ù‡Ùˆ (Ù…Ù†Ø§Ø³Ø¨ Ù„Ù€ EfficientNet-B0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)


#################################
# 3. LOAD DATA
#################################
train_df = pd.read_csv(r"/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv")
train_df["path"] = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train//" + train_df["filename"]
train_df.drop("filename", axis=1, inplace=True)
train_df = train_df[["path","label"]]
#train_df = train_df.iloc[:100]  # ÙŠØ­ØªÙ�Ø¸ Ù�Ù‚Ø· Ø¨Ø£ÙˆÙ„ 100 ØµÙ�

print("Train Shape:", train_df.shape)
print(train_df.head())

# Map string labels to integer indices
label_mapping = {
    "Harri":        0,
    "Najdi":     1,
    "Naeimi":     2,
    "Goat":  3,
    "Sawakni":     4,
    "Barbari":      5,
    "Roman":      6
}
train_df["label_idx"] = train_df["label"].map(label_mapping)


from collections import Counter

# Ù†Ø­Ù�Ø¸ Ø§Ù„Ù…Ù‚Ø§Ø³Ø§Øª Ù‡Ù†Ø§
sizes = []
# Ø§Ù�Ø­Øµ ÙƒÙ„ ØµÙˆØ±Ø© Ù�ÙŠ Ù…Ø³Ø§Ø±Ø§Øª train_df
for path in train_df["path"]:
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        sizes.append((w, h))

# ØªØ­ÙˆÙŠÙ„ Ø¥Ù„Ù‰ DataFrame
size_df = pd.DataFrame(sizes, columns=["width", "height"])

# Ø¥Ø­ØµØ§Ø¦ÙŠØ§Øª
size_counts = Counter(sizes)
most_common = size_counts.most_common(5)

print("ğŸ“Š Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ±:", len(sizes))
print("ğŸ“� Ø£ØµØºØ± Ø­Ø¬Ù…:", size_df.min().to_dict())
print("ğŸ“� Ø£ÙƒØ¨Ø± Ø­Ø¬Ù…:", size_df.max().to_dict())
print("ğŸ“Œ Ø£ÙƒØ«Ø± Ø§Ù„Ù…Ù‚Ø§Ø³Ø§Øª ØªÙƒØ±Ø§Ø±Ù‹Ø§:", most_common)
# Ø­Ø³Ø§Ø¨ Ø§Ù„ØµÙˆØ± Ø§Ù„ØªÙŠ Ø¹Ø±Ø¶Ù‡Ø§ Ø£Ùˆ Ø·ÙˆÙ„Ù‡Ø§ Ø£Ù‚Ù„ Ù…Ù† 300 Ø¨ÙƒØ³Ù„
small_images = size_df[(size_df["width"] < 300) | (size_df["height"] < 300)]
# Ø·Ø¨Ø§Ø¹Ø© Ø¹Ø¯Ø¯Ù‡Ø§
print(f"ğŸ“‰ Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ø§Ù„ØªÙŠ Ø£Ø¨Ø¹Ø§Ø¯Ù‡Ø§ Ø£ØµØºØ± Ù…Ù† 300x300:", len(small_images))

# Ø±Ø³Ù… Box Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x=size_df["width"])
plt.title("Box Plot of Image Widths")

plt.subplot(1, 2, 2)
sns.boxplot(x=size_df["height"])
plt.title("Box Plot of Image Heights")

plt.tight_layout()
plt.show()


classes = train_df["label"].unique()
print("ğŸŸ¢ Classes:", classes)
print("ğŸ”¢ Number of classes:", len(classes))
print(train_df["label"].value_counts())


# Ø£Ø®Ø° Ø­ØªÙ‰ 250 ØµÙˆØ±Ø© Ù…Ù† ÙƒÙ„ ÙƒÙ„Ø§Ø³ (Ø¥Ø°Ø§ ÙƒØ§Ù†Øª Ù…ØªÙˆÙ�Ø±Ø©)
balanced_df = train_df.groupby("label").apply(
    lambda x: x.sample(n=min(250, len(x)), random_state=CFG.seed)
).reset_index(drop=True)

# Ø¥Ø¹Ø§Ø¯Ø© ØªØ±ØªÙŠØ¨ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø©
balanced_df = balanced_df[["path", "label", "label_idx"]]

# Ø§Ù„ØªØ£ÙƒØ¯ Ù…Ù† Ø§Ù„Ù†ØªÙŠØ¬Ø©
print(balanced_df["label"].value_counts())
print("Total samples:", len(balanced_df))


#################################
# 4. K-FOLD Split
#################################
balanced_df["fold"] = -1
kf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

for fold_number, (tr_idx, val_idx) in enumerate(kf.split(balanced_df, balanced_df["label"])):
    balanced_df.loc[val_idx, "fold"] = fold_number

print(balanced_df.groupby("fold").size())


# ØªÙˆØ²ÙŠØ¹ Ø§Ù„ØµÙˆØ± Ø­Ø³Ø¨ Ø§Ù„Ù�Ø¦Ø©
label_counts = balanced_df["label"].value_counts()

# ØªØ¹Ø±ÙŠÙ� Ø§Ù„Ø¯Ø§Ù„Ø© Ø§Ù„ØªÙŠ ØªØ­Ø¯Ø¯ Ù†ÙˆØ¹ Ø§Ù„Ù€ augmentation
def assign_aug(label):
    count = label_counts[label]
    if count <= 70:  # Ø§Ù„ÙƒÙ„Ø§Ø³Ø§Øª Ø§Ù„Ø£Ù‚Ù„ (Barbari, Harri)
        return "strong_aug"
    elif 70 < count <= 110:  # Ø§Ù„ÙƒÙ„Ø§Ø³Ø§Øª Ø§Ù„Ù…ØªÙˆØ³Ø·Ø© (Najdi, Roman, Sawakni, Goat)
        return "basic_aug"
    else:  # Naeimi Ø£Ùˆ Ø£ÙŠ Ù�Ø¦Ø© Ø¹Ø¯Ø¯Ù‡Ø§ ÙƒØ¨ÙŠØ±
        return "none"

# ØªØ·Ø¨ÙŠÙ‚ Ø§Ù„Ø¯Ø§Ù„Ø© ÙˆØ¥Ø¶Ø§Ù�Ø© Ø§Ù„Ø¹Ù…ÙˆØ¯
train_df["aug"] = train_df["label"].apply(assign_aug)



#################################
# 5. DATASET & TRANSFORMS
#################################

# Augmentations Ø­Ø³Ø¨ Ù†ÙˆØ¹ Ø§Ù„ÙƒÙ„Ø§Ø³
resize_only = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor()
])

basic_aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    
    T.Resize((300, 300)),
    T.ToTensor()
])

strong_aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomResizedCrop(size=(300, 300), scale=(0.8, 1.0)),
    
    T.Resize((300, 300)),
    T.ToTensor()
])

# Ù†Ù�Ø³ Ø§Ù„Ø´ÙŠØ¡ Ù„Ù„Ù€ Validation
valid_transforms = T.Compose([
    T.Resize((300, 300)),  # âœ… Ø¶Ø±ÙˆØ±ÙŠ
    T.ToTensor()           # âœ… Ø¶Ø±ÙˆØ±ÙŠ
])


class SheepDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image_path = row["path"]
        label_idx = row["label_idx"] if "label_idx" in row else None
        aug_type = row.get("aug", "none")  # Ù†Ø­ØµÙ„ Ø¹Ù„Ù‰ Ù†ÙˆØ¹ Ø§Ù„ØªØ­ÙˆÙŠÙ„

        # Ù‚Ø±Ø§Ø¡Ø© Ø§Ù„ØµÙˆØ±Ø© ÙˆØªØ­ÙˆÙŠÙ„Ù‡Ø§ Ø¥Ù„Ù‰ RGB Ø«Ù… PIL
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Ø§Ø®ØªÙŠØ§Ø± Ø§Ù„ØªØ­ÙˆÙŠÙ„ Ø§Ù„Ù…Ù†Ø§Ø³Ø¨ Ø¨Ù†Ø§Ø¡Ù‹ Ø¹Ù„Ù‰ ÙˆØ¶Ø¹ Ø§Ù„ØªØ´ØºÙŠÙ„ ÙˆÙ†ÙˆØ¹ Ø§Ù„ØªØ­ÙˆÙŠÙ„
        if self.mode == "train":
            if aug_type == "strong_aug":
                image = strong_aug(image)
            elif aug_type == "basic_aug":
                image = basic_aug(image)
            else:
                image = resize_only(image)
        else:
            image = valid_transforms(image)

        if self.mode != "test":
            return image, label_idx
        else:
            return image


#################################
# 6. MODEL DEFINITION (EffNet-B0)
#################################
# We'll modify the final layer for the number of classes we have (7).
def get_model(num_classes=7, pretrained=True):
    model = torchvision.models.efficientnet_b3(pretrained=pretrained)
    model.classifier[1] = nn.Linear(1536, num_classes)
    return model


#################################
# 7. TRAIN & VALID FUNCTIONS
#################################
def train_one_epoch(model, optimizer, dataloader, device, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    epoch_loss = total_loss / len(dataloader.dataset)
    return epoch_loss

def valid_one_epoch(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, f1


#################################
# 8. K-FOLD TRAINING LOOP
#################################
def run_training(fold):
    print(f"========== Fold: {fold} ==========")

    # Split
    train_data = balanced_df[balanced_df["fold"] != fold].reset_index(drop=True)
    valid_data = balanced_df[balanced_df["fold"] == fold].reset_index(drop=True)

    # Datasets
    train_dataset = SheepDataset(train_data, mode="train")
    valid_dataset = SheepDataset(valid_data, mode="valid")

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs,
                              shuffle=True, num_workers=CFG.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs,
                              shuffle=False, num_workers=CFG.num_workers)

    # Model, Optimizer, Loss
    model = get_model(num_classes=len(label_mapping), pretrained=True)
    model.to(CFG.device)
    print("ğŸ”� Model is on device:", next(model.parameters()).device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    best_f1 = 0.0
    for epoch in range(CFG.epochs):
        print(f"Fold {fold} | Epoch {epoch+1}/{CFG.epochs}")

        train_loss = train_one_epoch(model, optimizer, train_loader, CFG.device, criterion)
        valid_loss, valid_f1 = valid_one_epoch(model, valid_loader, CFG.device, criterion)

        print(f"  [Train Loss: {train_loss:.4f}]  [Valid Loss: {valid_loss:.4f}]  [Valid F1 Score: {valid_f1:.4f}]")

        if valid_f1 > best_f1:
            best_f1 = valid_f1
            save_path = f"effb0_fold_{fold}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Model saved to {save_path}")

    print(f"Fold {fold} best F1 Score: {best_f1:.4f}\n")



for fold in range(CFG.n_splits):
    run_training(fold)



# ØªØ£ÙƒØ¯ Ø£Ù† Ù„Ø¯ÙŠÙƒ Ù‡Ø°Ø§ Ù…Ù† Ù‚Ø¨Ù„

# ===== Ø¥Ø¹Ø¯Ø§Ø¯Ø§Øª Ø¹Ø§Ù…Ø© =====
device = CFG.device
NUM_FOLDS = CFG.n_splits
BATCH_SIZE = CFG.valid_bs

# Ù…Ø§Ø¨ Ø§Ù„ÙƒÙ„Ø§Ø³Ø§Øª
idx_to_label = {
    0: "Harri",
    1: "Najdi",
    2: "Naeimi",
    3: "Goat",
    4: "Sawakni",
    5: "Barbari",
    6: "Roman"
}


# ===============================
# 2. Ø¥Ø¹Ø¯Ø§Ø¯ test_df
test_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test"  # Ø¶Ø¹ Ø§Ù„Ù…Ø³Ø§Ø± Ø§Ù„Ù…Ù†Ø§Ø³Ø¨ Ù‡Ù†Ø§
# Ø¥Ù†Ø´Ø§Ø¡ DataFrame ÙŠØ­ØªÙˆÙŠ Ø¹Ù„Ù‰ Ù…Ø³Ø§Ø±Ø§Øª Ø§Ù„ØµÙˆØ±
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
test_paths = [os.path.join(test_dir, f) for f in test_files]
test_df = pd.DataFrame({
    "filename": test_files,
    "path": test_paths
})

# ===============================
# 3. Ø¥Ø¹Ø¯Ø§Ø¯ Ø§Ù„Ø¯Ø§ØªØ§Ø³Øª ÙˆØ§Ù„Ù„ÙˆØ¯Ø±
# Ø¥Ø¹Ø¯Ø§Ø¯ Ø¯Ø§ØªØ§ Ù„ÙˆÙˆØ¯Ø± Ù„Ù„Ø§Ø®ØªØ¨Ø§Ø±
test_dataset = SheepDataset(test_df, mode="test")
test_loader = DataLoader(test_dataset, batch_size=CFG.valid_bs, shuffle=False, num_workers=CFG.num_workers)

# ===============================
# 4. ØªÙ†Ø¨Ø¤ Ø¨Ø§Ø³ØªØ®Ø¯Ø§Ù… Ø¬Ù…ÙŠØ¹ Ø§Ù„Ù€ Folds
all_predictions = []

print("\nğŸ”� Generating test predictions...")
for fold in range(NUM_FOLDS):
    model_path = f"effb0_fold_{fold}.pth"
    if not os.path.exists(model_path):
        print(f"â�Œ Model for fold {fold} not found, skipping.")
        continue

    print(f"âœ… Fold {fold + 1}/{NUM_FOLDS}")
    model = get_model(num_classes=7, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    fold_preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc=f"Fold {fold+1} Inference"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            fold_preds.append(probs)

    fold_preds = np.concatenate(fold_preds, axis=0)
    all_predictions.append(fold_preds)

    # ØªÙ†Ø¸ÙŠÙ�
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ===============================
# 5. ØªØ¬Ù…ÙŠØ¹ Ø§Ù„ØªÙˆÙ‚Ø¹Ø§Øª Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠØ©
avg_predictions = np.mean(all_predictions, axis=0)
final_indices = np.argmax(avg_predictions, axis=1)
final_labels = [idx_to_label[idx] for idx in final_indices]

# ===============================
# 6. Ø¥Ù†Ø´Ø§Ø¡ Ù…Ù„Ù� submission
submission_df = pd.DataFrame({
    "filename": test_df["filename"],
    "label": final_labels
})
submission_df.to_csv("submission.csv", index=False)
print("\nğŸ“� Saved submission.csv!")
print(submission_df.head())

# ===============================
# 7. Ø­Ù�Ø¸ Ù…Ù„Ù� Ø§Ù„Ø§Ø­ØªÙ…Ø§Ù„Ø§Øª
prob_df = pd.DataFrame({
    "filename": test_df["filename"],
    "pred_label": final_labels
})
for i, class_name in idx_to_label.items():
    prob_df[f"prob_{class_name}"] = avg_predictions[:, i]

prob_df.to_csv("test_predictions_with_probs.csv", index=False)
print("ğŸ“� Saved test_predictions_with_probs.csv!")

