# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


df = pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')
df.head()


df.info()


#label names
label_names = sorted(df['label'].unique())
class_to_idx = {label: idx for idx, label in enumerate(label_names)}


class_to_idx


class SheepData(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, class_to_idx=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['filename']
        label_name = self.dataframe.iloc[idx]['label']
        label = self.class_to_idx[label_name]
        
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
])
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)



image_dir='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'


train_dataset = SheepData(train_df, image_dir, transform=train_transform, class_to_idx=class_to_idx)
val_dataset = SheepData(val_df, image_dir, transform=valid_transform, class_to_idx=class_to_idx)




train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.resnet18(pretrained=True)
# model = models.resnext101_32x8d(pretrained=True)
# model = models.mobilenet_v3_large(pretrained=True)
# model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_to_idx))
# model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
model = models.maxvit_t(weights='IMAGENET1K_V1')
in_features = model.classifier[5].in_features
model.classifier[5] = nn.Linear(in_features, len(class_to_idx))

model = model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 25


from sklearn.metrics import f1_score

train_losses = []
val_losses = []
val_accuracies = []
val_f1s = []  

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    epoch_train_loss = train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    print(f'Epoch {epoch}: Train Loss = {epoch_train_loss}')

    model.eval()
    val_loss = 0
    val_correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_pred = model(x)

            loss = loss_fn(val_pred, y)
            val_loss += loss.item()

            preds = val_pred.argmax(dim=1)
            val_correct += (preds == y).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = val_correct / len(val_loader.dataset)
    epoch_val_f1 = f1_score(all_labels, all_preds, average='weighted')  # Compute weighted F1

    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    val_f1s.append(epoch_val_f1)

    print(f'Epoch {epoch}: Val Loss = {epoch_val_loss}, Val Acc = {epoch_val_acc}, Val F1 = {epoch_val_f1:.4f}')



#Need this as there are no labels
class Testdata(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = Testdata("/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test", transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)




# Reverse map: idx → class name
idx_to_class = {v: k for k, v in class_to_idx.items()}

model.eval()
results = []

with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for fname, pred in zip(filenames, preds):
            label = idx_to_class[pred.item()]
            results.append({"filename": fname, "label": label})

# Save to CSV
df_sub = pd.DataFrame(results)
df_sub.to_csv("submission.csv", index=False)





