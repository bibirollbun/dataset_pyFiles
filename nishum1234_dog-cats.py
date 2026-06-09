import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# ===== 1. ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ =====
BATCH_SIZE = 64
EPOCHS = 6
IMG_SIZE = 128
LR = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"âœ… Using: {device}")

# ===== 2. zipãƒ•ã‚¡ã‚¤ãƒ«ã�®è§£å‡� =====
input_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
work_dir = "/kaggle/working/"

with zipfile.ZipFile(input_dir + "train.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "train/")
with zipfile.ZipFile(input_dir + "test.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "test/")

train_dir = os.path.join(work_dir, "train/train")
test_dir = os.path.join(work_dir, "test/test")

# ===== 3. ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆå®šç¾© =====
class CatsDogsDataset(Dataset):
    def __init__(self, filepaths, labels=None, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image

# ===== 4. ãƒ•ã‚¡ã‚¤ãƒ«ã�¨ãƒ©ãƒ™ãƒ«ã�®æº–å‚™ï¼ˆé †åº�ä¿�è¨¼ï¼‰ =====
file_names = sorted(os.listdir(train_dir))  # sortã�§é †åº�å›ºå®š
train_files = [os.path.join(train_dir, f) for f in file_names]
labels = [1 if 'dog' in f else 0 for f in file_names]

X_train, X_val, y_train, y_val = train_test_split(train_files, labels, test_size=0.1, random_state=42)

# ===== 5. Transformå®šç¾©ï¼ˆData Augmentation + Normalizeï¼‰ =====
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== 6. DataLoaderä½œæˆ� =====
train_dataset = CatsDogsDataset(X_train, y_train, transform=train_transform)
val_dataset = CatsDogsDataset(X_val, y_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===== 7. ãƒ¢ãƒ‡ãƒ«æ§‹ç¯‰ï¼ˆVGG11ï¼‰ =====
from torchvision.models import vgg11, VGG11_Weights
model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===== 8. å­¦ç¿’ãƒ«ãƒ¼ãƒ— =====
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {running_loss / len(train_loader):.4f}")
print("âœ… å­¦ç¿’å®Œäº†")

# ===== 9. æ¤œè¨¼ç²¾åº¦è©•ä¾¡ =====
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f"ğŸ”� Val Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

# æ··å�ˆè¡Œåˆ—è¡¨ç¤º
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Validation)")
plt.show()

# ===== 10. æ�¨è«–ç”¨ï¼ˆãƒ†ã‚¹ãƒˆã‚»ãƒƒãƒˆï¼‰ =====
test_images = [f for f in os.listdir(test_dir) if f.endswith(".jpg")]
test_images = sorted(test_images, key=lambda x: int(x.split('.')[0]))
test_paths = [os.path.join(test_dir, fname) for fname in test_images]

test_dataset = CatsDogsDataset(test_paths, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # çŠ¬ã�®ç¢ºç�‡
        predictions.extend(probs.cpu().numpy())

# ===== 11. æ��å‡ºç”¨CSVä½œæˆ� =====
submission = pd.DataFrame({
    "id": [int(fname.split(".")[0]) for fname in test_images],
    "label": predictions
})
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv ã‚’ä¿�å­˜ã�—ã�¾ã�—ã�Ÿ")

# ãƒ¢ãƒ‡ãƒ«ä¿�å­˜
torch.save(model.state_dict(), "model.pth")

 
 
 
 
 
 

 
 
 
 
 
 
 
 

# äºˆæ¸¬
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# äºˆæ¸¬ã�¨æ­£è§£ã‚’è“„ç©�
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ç²¾åº¦ãƒ»F1
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("ğŸ”� å­¦ç¿’ãƒ¢ãƒ‡ãƒ«ã�®è©•ä¾¡çµ�æ�œ")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# æ··å�ˆè¡Œåˆ—
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
print("\nğŸ“Š æ··å�ˆè¡Œåˆ—:")
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Model)")
plt.show()





