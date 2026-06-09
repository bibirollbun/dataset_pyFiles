!pip -q install timm==1.0.3 --no-warn-conflicts
!apt install libarchive-dev
!pip install libarchive


import os, math, random, re, json, gc, time, glob, shutil, pathlib, asyncio, csv, glob
import libarchive.public
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    # transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

val_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

DATA_DIR = "./data"

train_set = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tfms)
val_set   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_tfms)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

CLASSES = train_set.classes
class_to_idx = {c:i for i,c in enumerate(CLASSES)}


# total number of samples
print("Total images:", len(train_set) + len(val_set))

# training and validation separately
print("Train set size:", len(train_set))
print("Val set size:", len(val_set))

# classes
print("Classes:", CLASSES)




def imshow(ax, img):
    # img: CxHxW (tensor)
    img = img.detach().cpu().numpy().transpose(1, 2, 0)   # → HxWxC
    img = img * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)  # unnormalize
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.axis("off")

images, labels = next(iter(train_loader))

fig, axes = plt.subplots(2, 4, figsize=(7, 3))
axes = axes.flatten()
for i in range(8):
    imshow(axes[i], images[i])
    axes[i].set_title(CLASSES[labels[i]])
plt.tight_layout()
plt.show()



def resnet18_cifar10(num_classes=10):
    m = models.resnet18(weights=None)                 # <- NO PRETRAINED WEIGHTS
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, num_classes, bias=True)
    return m

model = resnet18_cifar10().to(DEVICE)
no_params = sum(p.numel() for p in model.parameters())/1e6
print(f"Number of Parameters: {no_params:.2f} M")


EPOCHS = 100
LR = 0.1
WEIGHT_DECAY = 5e-4

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def accuracy(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()

best_acc = 0.0
best_path = "best_model.pth"
best_path_final = best_path.replace(".pth", "_final.pth")

train_losses, val_losses = [], []
train_accs, val_accs = [], []
for epoch in range(1, EPOCHS+1):
    # train
    model.train()
    tr_loss, tr_acc, n = 0.0, 0.0, 0
    for x,y in train_loader:
        x,y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        tr_loss += loss.item()*bs
        tr_acc  += accuracy(logits, y)*bs
        n += bs

    scheduler.step()

    # validate
    model.eval()
    va_loss, va_acc, n_val = 0.0, 0.0, 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            va_loss += loss.item()*bs
            va_acc  += accuracy(logits, y)*bs
            n_val += bs

    tr_loss/=n; tr_acc/=n
    va_loss/=n_val; va_acc/=n_val

    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    train_accs.append(tr_acc)
    val_accs.append(va_acc)

    print(f"Epoch {epoch:03d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({"model": model.state_dict(), "classes": CLASSES}, best_path)
    torch.save({"model": model.state_dict(), "classes": CLASSES}, best_path_final)

print("Best val acc:", best_acc)



fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Loss subplot
axes[0].plot(range(1, EPOCHS+1), train_losses, label="Train Loss")
axes[0].plot(range(1, EPOCHS+1), val_losses, label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training vs Validation Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy subplot
axes[1].plot(range(1, EPOCHS+1), train_accs, label="Train Acc")
axes[1].plot(range(1, EPOCHS+1), val_accs, label="Val Acc")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Training vs Validation Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()



# model = torch.load("/kaggle/input/resnet18-ass4/pytorch/default/1/best_model.pth", map_location=DEVICE)
# Load checkpoint
checkpoint = torch.load("/kaggle/working/best_model.pth", map_location=DEVICE)

# Rebuild model
model = resnet18_cifar10(num_classes=len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()


all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())

# concat to 1D arrays
y_true = np.concatenate(all_targets)
y_pred = np.concatenate(all_preds)

# --- Metrics ---
acc = accuracy_score(y_true, y_pred)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

print(f"Accuracy: {acc:.4f}")
print(f"Macro  - Precision: {prec_macro:.4f}  Recall: {rec_macro:.4f}  F1: {f1_macro:.4f}")
print(f"Weighted - Precision: {prec_weighted:.4f}  Recall: {rec_weighted:.4f}  F1: {f1_weighted:.4f}")

# Per-class breakdown
print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# raw counts
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax[0], xticks_rotation=45, colorbar=False)
ax[0].set_title("Confusion Matrix (Counts)")

# normalized by true class (row-normalized)
cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASSES)
disp_norm.plot(ax=ax[1], xticks_rotation=45, colorbar=False, values_format=".2f")
ax[1].set_title("Confusion Matrix (Row-Normalized)")

plt.tight_layout()
plt.show()



cnt = 0

for entry in libarchive.public.file_pour('/kaggle/input/cifar-10/test.7z'):
    cnt += 1
    if cnt % 75000 == 0: print(cnt)


# len(glob.glob('test/*'))


test_tfms = val_tfms

def load_image(path):
    return test_tfms(Image.open(path).convert("RGB"))

# IMPORTANT: iterate ids in numeric order
test_dir = "test"
test_ids = sorted([int(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(os.path.join(test_dir, "*.png"))])
print("Test images:", len(test_ids))

BATCH = 512
pred_labels = []

with torch.no_grad():
    batch = []
    current_ids = []
    for i in test_ids:
        x = load_image(os.path.join(test_dir, f"{i}.png"))
        batch.append(x)
        current_ids.append(i)
        if len(batch) == BATCH:
            xb = torch.stack(batch).to(DEVICE)
            probs = model(xb).softmax(1)
            preds = probs.argmax(1).cpu().numpy()
            pred_labels += [(j, CLASSES[p]) for j,p in zip(current_ids, preds)]
            batch, current_ids = [], []
    if batch:
        xb = torch.stack(batch).to(DEVICE)
        probs = model(xb).softmax(1)
        preds = probs.argmax(1).cpu().numpy()
        pred_labels += [(j, CLASSES[p]) for j,p in zip(current_ids, preds)]



with open("submission.csv","w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id","label"])
    for i,lab in sorted(pred_labels, key=lambda t:t[0]):
        w.writerow([i, lab])

print("Wrote submission.csv")


