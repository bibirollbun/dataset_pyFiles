import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image


import csv
import sys
import json

csv.field_size_limit(sys.maxsize)  

X_train = []
y_train = []

with open('/kaggle/input/fruits/Train.csv', 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        data = json.loads(row['Image'])
        X_train.append(np.array(data, dtype=np.uint8))
        y_train.append(row['Label'])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
with open('/kaggle/input/fruits/Test.csv', 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        data = json.loads(row['Image'])
        X_test.append(np.array(data, dtype=np.uint8))

X_test = np.array(X_test)


class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, label_encoder=None):
        self.images = images
        self.transform = transform

        if labels is not None:
            self.has_labels = True
            self.labels = labels
            if label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.encoded_labels = self.label_encoder.fit_transform(labels)
            else:
                self.label_encoder = label_encoder
                self.encoded_labels = self.label_encoder.transform(labels)
        else:
            self.has_labels = False
            self.encoded_labels = [-1] * len(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        img = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return img, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

label_encoder = LabelEncoder().fit(y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

train_dataset = ImageDataset(X_train, y_train, transform=transform, label_encoder=label_encoder)
val_dataset = ImageDataset(X_val, y_val, transform=transform, label_encoder=label_encoder)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DummyModel(nn.Module):
    def __init__(self, output_class=0, num_classes=2):
        super().__init__()
        self.output_class = output_class
        self.num_classes = num_classes

    def forward(self, x):
        batch_size = x.size(0)
        out = torch.zeros(batch_size, self.num_classes)
        out[:, self.output_class] = 1 
        return out

model = DummyModel(num_classes=len(label_encoder.classes_)).to(device)





test_dataset = ImageDataset(X_test, transform=transform, label_encoder=label_encoder)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


model.eval()
all_preds = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())

predicted_labels = label_encoder.inverse_transform(all_preds)


with open('./submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'Label'])
    for id, label in enumerate(predicted_labels):
        writer.writerow([id, label])


!pip install -q kaggle


import os
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key'


!kaggle competitions submit -c "juicy-or-junk-fruit-quality-detection" -f "submission.csv" -m "Baseline submission from notebook"

