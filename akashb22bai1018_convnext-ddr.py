import os, shutil, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
import timm
import seaborn as sns

csvpath = "/kaggle/input/ddrdataset/DR_grading.csv"
imgdir = "/kaggle/input/ddrdataset/DR_grading/DR_grading"
df = pd.read_csv(csvpath)
print("Total images:", len(df))
orig_dist = Counter(df['diagnosis'])
print("Original class distribution:", orig_dist)

# Remove duplicates
df = df.drop_duplicates(subset='id_code')
print("After removing duplicates:", len(df))

# Combine into binary classes
class0 = df[df['diagnosis'] == 0].copy()
other_classes = df[df['diagnosis'] != 0].copy()

class0['diagnosis'] = 0
other_classes['diagnosis'] = 1

df_binary = pd.concat([class0, other_classes], axis=0).reset_index(drop=True)

print("Dataset size (no removal from class 0):", len(df_binary))
print("Binary class distribution:", Counter(df_binary['diagnosis']))

# Train/test split
train_df, test_df = train_test_split(
    df_binary, test_size=0.2, stratify=df_binary['diagnosis'], random_state=42
)
print("Train size:", len(train_df))
print("Test size:", len(test_df))
print("Train distribution:", Counter(train_df['diagnosis']))
print("Test distribution:", Counter(test_df['diagnosis']))

# Folder Creation
traindir = "/kaggle/working/ddr_split/train"
testdir = "/kaggle/working/ddr_split/test"
os.makedirs(traindir, exist_ok=True)
os.makedirs(testdir, exist_ok=True)

for c in df_binary['diagnosis'].unique():
    os.makedirs(os.path.join(traindir, str(c)), exist_ok=True)
    os.makedirs(os.path.join(testdir, str(c)), exist_ok=True)

def copy_imgs(data, splitdir):
    for _, row in data.iterrows():
        src = os.path.join(imgdir, row['id_code'])
        dst_dir = os.path.join(splitdir, str(row['diagnosis']))
        dst = os.path.join(dst_dir, row['id_code'])
        if not os.path.exists(src):
            continue
        try:
            shutil.copy2(src, dst)
        except Exception:
            try:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, dst)
            except Exception:
                continue

copy_imgs(train_df, traindir)
copy_imgs(test_df, testdir)

# Augmentations
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(traindir, transform=train_tfms)
test_data = datasets.ImageFolder(testdir, transform=test_tfms)

print("Train dataset size:", len(train_data))
print("Test dataset size:", len(test_data))

# Weighted Sampler
targets = [lbl for _, lbl in train_data.samples]
class_counts = Counter(targets)
class_weights = [1.0 / class_counts[t] for t in targets]
sampler = WeightedRandomSampler(class_weights, num_samples=len(targets), replacement=True)

train_loader = DataLoader(train_data, batch_size=32, sampler=sampler, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_data.classes)

model = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    num_classes=num_classes,
    drop_path_rate=0.2,
    drop_rate=0.3
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.25, verbose=True)

# Warm-up Freeze
for param in model.stem.parameters():
    param.requires_grad = False
for param in model.stages.parameters():
    param.requires_grad = False

# Training
def train_model(epochs=20, unfreeze_epoch=5, patience=5):
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0
    wait = 0
    best_preds, best_labels = [], []

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            for param in model.parameters():
                param.requires_grad = True
            for g in optimizer.param_groups:
                g["lr"] = 1e-5

        model.train()
        tot, cor, runloss = 0, 0, 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            runloss += loss.item() * imgs.size(0)
            _, preds = torch.max(out, 1)
            cor += (preds == lbls).sum().item()
            tot += lbls.size(0)
        train_acc = cor / tot
        train_loss = runloss / tot

        model.eval()
        tot, cor, runloss = 0, 0, 0.0
        allp, alll = [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                runloss += loss.item() * imgs.size(0)
                _, preds = torch.max(out, 1)
                allp.extend(preds.cpu().numpy())
                alll.extend(lbls.cpu().numpy())
                cor += (preds == lbls).sum().item()
                tot += lbls.size(0)
        test_acc = cor / tot
        test_loss = runloss / tot

        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Test Loss {test_loss:.4f} Acc {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            wait = 0
            best_preds, best_labels = allp, alll
            torch.save(model.state_dict(), "best_model_v1.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping!")
                break

    return history, np.array(best_preds), np.array(best_labels)

# Run
history, y_pred, y_true = train_model(epochs=25)

# Plots
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["test_loss"], label="Test Loss")
plt.legend(); plt.title("Loss Curve"); plt.show()

plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["test_acc"], label="Test Acc")
plt.legend(); plt.title("Accuracy Curve"); plt.show()

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=train_data.classes,
            yticklabels=train_data.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix")
plt.show()

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=train_data.classes))
print("MCC:", matthews_corrcoef(y_true, y_pred))


import IPython
IPython.Application.instance().kernel.do_shutdown(True)


import os
import torch
import pandas as pd
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt
import timm
import torch.nn as nn

csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
img_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(csv_path)
df = df.drop_duplicates(subset='id_code')
df["diagnosis"] = df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)
print("Class distribution:", Counter(df["diagnosis"]))

# Transforms
test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset
class AptosBinaryDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['id_code'] + ".png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["diagnosis"]
        return image, label

# Dataloader
dataset = AptosBinaryDataset(df, img_dir, test_tfms)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

# Load Model
model = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    num_classes=2,
    drop_path_rate=0.2,
    drop_rate=0.3
)
model.load_state_dict(torch.load("/kaggle/working/best_model_v1.pth", map_location=device))
model = model.to(device)
model.eval()

# Inference
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=["No DR", "DR"]))
print("MCC:", matthews_corrcoef(all_labels, all_preds))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["No DR", "DR"], yticklabels=["No DR", "DR"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix")
plt.show()




