import os
import zipfile
from PIL import Image
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# === Пути к архивам ===
zip_train = '/kaggle/input/the-nature-conservancy-fisheries-monitoring/train.zip'
zip_test = '/kaggle/input/the-nature-conservancy-fisheries-monitoring/test_stg1.zip'

# === Распаковка train.zip с удалением верхнего уровня 'train/' ===
with zipfile.ZipFile(zip_train, 'r') as zip_ref:
    for member in zip_ref.namelist():
        if member.startswith("train/") and not member.endswith('/'):
            target_path = os.path.join('/kaggle/working/train_clean', os.path.relpath(member, "train"))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, 'wb') as f:
                f.write(zip_ref.read(member))

# Распаковка test_stg1.zip
with zipfile.ZipFile(zip_test, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/test_stg1')

# === Пути к директориям ===
train_dir = '/kaggle/working/train_clean'
test_dir = '/kaggle/working/test_stg1/test_stg1'


# === Классы ===
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_names)
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# === Dataset ===
class FishDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.labels.append(class_to_idx[class_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# === DataLoader ===
train_dataset = FishDataset(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

# === Модель ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


# === Обучение ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")


# === Сохранение модели ===
torch.save(model.state_dict(), "/kaggle/working/fish_model.pth")


# === Инференс на тесте ===
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.filenames = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        self.filepaths = [os.path.join(root_dir, f) for f in self.filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.filenames[idx]

test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
predictions = []
filenames = []

with torch.no_grad():
    for images, names in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions.extend(probs)
        filenames.extend(names)


# === Генерация submission.csv ===
submission = pd.DataFrame(predictions, columns=class_names)
submission.insert(0, 'image', filenames)
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv готов")

