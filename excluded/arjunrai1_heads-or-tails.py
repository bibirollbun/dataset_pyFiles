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


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])
])


dataset = datasets.ImageFolder(root="/kaggle/input/heads-or-tails-image-classification/train", transform=transform)


print(dataset.class_to_idx)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                 
            nn.Linear(256 * 16 * 16, 1024),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),             
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),                 
            nn.Sigmoid()                       
        )
    def forward(self, x):
            x = self.conv(x)
            x = self.fc(x)
            return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = CustomModel().to(device)


from torchvision import models
backbone = models.resnet18(pretrained=True)
backbone.fc = nn.Linear(backbone.fc.in_features, 1)
model = backbone.to(device)
criterion = nn.BCEWithLogitsLoss()


optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")


import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

model.eval()
all_logits = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)               
        logits = logits.squeeze(1)        
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())


all_logits = torch.cat(all_logits).numpy()
all_labels = torch.cat(all_labels).numpy()   


p_tails = 1 / (1 + np.exp(-all_logits))      
p_heads = 1.0 - p_tails


auc_score = roc_auc_score((all_labels == 0).astype(int), p_heads)
print(f"ROC AUC (heads) = {auc_score:.4f}")

fpr, tpr, thresholds = roc_curve((all_labels == 0).astype(int), p_heads)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC for Heads Detection")
plt.legend(loc="lower right")
plt.show()



from PIL import Image
from torch.utils.data import Dataset
import csv
import os
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5], [0.5])
])

test_dataset = UnlabeledImageDataset(root_dir='/kaggle/input/heads-or-tails-image-classification/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model.eval()
results = []

with torch.no_grad():
    for batch_idx, images in enumerate(test_loader):
        images = images.to(device)
        logits = model(images).squeeze(1)           
        p_tails = torch.sigmoid(logits).cpu()     
        p_heads = (1.0 - p_tails).tolist()           

        for i, ph in enumerate(p_heads):
            prediction_id = batch_idx * test_loader.batch_size + i + 1
            results.append([prediction_id, ph])


with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['prediction_id', 'probability_of_heads'])
    writer.writerows(results)




