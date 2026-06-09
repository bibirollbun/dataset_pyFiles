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


import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# ===== 1. zipãƒ•ã‚¡ã‚¤ãƒ«ã�®è§£å‡� =====
input_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
work_dir = "/kaggle/working/"

with zipfile.ZipFile(input_dir + "train.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "train/")

with zipfile.ZipFile(input_dir + "test.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "test/")



train_root = os.path.join(work_dir, "train")
test_root = os.path.join(work_dir, "test")

print("ğŸ“� train ãƒ•ã‚©ãƒ«ãƒ€ã�®å†…å®¹ï¼ˆä¸Šä½�10ä»¶ï¼‰:")
print(os.listdir(train_root)[:10])  # trainç›´ä¸‹ã�«trainãƒ•ã‚©ãƒ«ãƒ€ã�Œã�‚ã‚‹ã�‹ç¢ºèª�

print("\nğŸ“� train/train ãƒ•ã‚©ãƒ«ãƒ€ã�®å†…å®¹ï¼ˆä¸Šä½�10ä»¶ï¼‰:")
print(os.listdir(os.path.join(train_root, "train"))[:10])

print("\nğŸ“� test/test ãƒ•ã‚©ãƒ«ãƒ€ã�®å†…å®¹ï¼ˆä¸Šä½�10ä»¶ï¼‰:")
print(os.listdir(os.path.join(test_root, "test"))[:10])


# ===== 3. ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ =====
BATCH_SIZE = 64
EPOCHS = 3
IMG_SIZE = 128
LR = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nâœ… Using:", device)


# ===== 4. Datasetå®šç¾© =====
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


# ===== 5. ãƒ•ã‚¡ã‚¤ãƒ«ã�¨ãƒ©ãƒ™ãƒ«æº–å‚™ =====
train_dir = os.path.join(train_root, "train")
test_dir = os.path.join(test_root, "test")

train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
labels = [1 if 'dog' in f else 0 for f in os.listdir(train_dir)]

X_train, X_val, y_train, y_val = train_test_split(train_files, labels, test_size=0.1, random_state=42)


# ===== 6. ãƒ‡ãƒ¼ã‚¿å¤‰æ�›ã�¨DataLoader =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dataset = CatsDogsDataset(X_train, y_train, transform)
val_dataset = CatsDogsDataset(X_val, y_val, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ===== 7. ãƒ¢ãƒ‡ãƒ«æº–å‚™ï¼ˆResNet18ï¼‰ =====
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
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
    print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}")
print("train finish")


# ===== 9. ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿ã�¨äºˆæ¸¬ =====
model.eval()
test_images = sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0]))
test_paths = [os.path.join(test_dir, fname) for fname in test_images]

test_dataset = CatsDogsDataset(test_paths, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # çŠ¬ã�®ç¢ºç�‡
        predictions.extend(probs.cpu().numpy())


# ===== 10. æ��å‡ºç”¨CSVä½œæˆ� =====
submission = pd.DataFrame({
    "id": [int(fname.split(".")[0]) for fname in test_images],
    "label": predictions
})
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv ã‚’ä½œæˆ�ã�—ã�¾ã�—ã�Ÿ")


from sklearn.metrics import accuracy_score, f1_score

# äºˆæ¸¬ã�¨æ­£è§£ãƒ©ãƒ™ãƒ«ã‚’ãƒªã‚¹ãƒˆã�§è“„ç©�
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

# ç²¾åº¦ãƒ»F1ã‚’è¨ˆç®—
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Val Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")



# PyTorchãƒ¢ãƒ‡ãƒ«ã�®ä¿�å­˜
torch.save(model.state_dict(), "model.pth")

