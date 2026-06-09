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


DATASET_PATH = "/kaggle/input/solidworks-ai-hackathon"

print("Root contents:")
print(os.listdir(DATASET_PATH))

print("\nTrain folder:")
print(os.listdir(f"{DATASET_PATH}/train")[:5])

print("\nTest folder:")
print(os.listdir(f"{DATASET_PATH}/test"))



TRAIN_IMG_DIR = "/kaggle/input/solidworks-ai-hackathon/train/train"
TEST_IMG_DIR  = "/kaggle/input/solidworks-ai-hackathon/test/test"

print("Train images count:", len(os.listdir(TRAIN_IMG_DIR)))
print("Test images count :", len(os.listdir(TEST_IMG_DIR)))



import pandas as pd

LABELS_PATH = "/kaggle/input/solidworks-ai-hackathon/train_labels.csv"
labels = pd.read_csv(LABELS_PATH)

labels.head()



import cv2
import matplotlib.pyplot as plt

# pick a random training sample
row = labels.sample(1).iloc[0]
img_name = row["image_name"]

img_path = f"{TRAIN_IMG_DIR}/{img_name}"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(4,4))
plt.imshow(img)
plt.title(
    f"bolt={row.bolt}, pin={row.locatingpin}, nut={row.nut}, washer={row.washer}"
)
plt.axis("off")



from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



#Custom Dataset class 
import torch
from torch.utils.data import Dataset

class PartsDataset(Dataset):
    def __init__(self, img_dir, labels_df=None, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        if self.labels_df is not None:
            row = self.labels_df[self.labels_df.image_name == img_name]
            label = row[['bolt','locatingpin','nut','washer']].values[0]
            label = torch.tensor(label, dtype=torch.float32)
            return image, label
        else:
            return image, img_name



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



from torch.utils.data import DataLoader

train_dataset = PartsDataset(
    img_dir=TRAIN_IMG_DIR,
    labels_df=labels,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

len(train_loader)



#Load pretrained ResNet18
import torch.nn as nn
from torchvision import models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)

model



#Loss function + optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for imgs, targets in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")



#Create test Dataset + DataLoader
test_dataset = PartsDataset(
    img_dir=TEST_IMG_DIR,
    labels_df=None,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

len(test_loader)



#Inference
model.eval()
results = []

with torch.no_grad():
    for imgs, names in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs)

        preds = torch.round(preds).clamp(min=0).cpu().numpy()

        for name, p in zip(names, preds):
            results.append([name, int(p[0]), int(p[1]), int(p[2]), int(p[3])])





submission = pd.DataFrame(
    results,
    columns=["image_name", "bolt", "locatingpin", "nut", "washer"]
)

submission.head()



submission.to_csv("submission.csv", index=False)

