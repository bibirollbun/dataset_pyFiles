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


# ========================================================
# Final Project: Cassava Leaf Disease Classification
# Deluxe Version with Full Plots for Each Model
# 1. CNN From Scratch
# 2. MobileNetV3 (Pretrained)
# 3. EfficientNet-B3 (Pretrained)
# ========================================================


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.io import read_image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ------------------ Paths ------------------
CSV_PATH = '/kaggle/input/cassava-leaf-disease-classification/train.csv'
IMG_DIR  = '/kaggle/input/cassava-leaf-disease-classification/train_images'

# ------------------ Load & Split ------------------
df = pd.read_csv(CSV_PATH)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df['label'], random_state=42)

print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")


# ------------------ Dataset ------------------
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        image = read_image(os.path.join(IMG_DIR, img_name)).float() / 255.0
        if image.shape[0] == 1: image = image.repeat(3, 1, 1)
        label = self.df.iloc[idx, 1]
        if self.transform: image = self.transform(image)
        return image, label

# ------------------ Transforms  ------------------
transform = transforms.Compose([
    transforms.Resize((336, 336)),  
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.3,0.3,0.3),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CassavaDataset(train_df, transform)
val_dataset   = CassavaDataset(val_df,   transform)
test_dataset  = CassavaDataset(test_df,  transform)


# ------------------ Loaders (batch_size=32 ) ------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, pin_memory=True)


# ------------------ Models ------------------
# 1. CNN From Scratch
class CNNFromScratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 21 * 21, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 5)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. MobileNetV3 Pretrained
mobilenet = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, 5)

# 3. EfficientNet-B3 Pretrained
effnet = models.efficientnet_b3(weights='IMAGENET1K_V1')
effnet.classifier[1] = nn.Linear(1536, 5)

# Move to GPU
scratch_model = CNNFromScratch().to('cuda')
mobilenet.to('cuda')
effnet.to('cuda')


# ------------------ Training Function with Best Model Saving ------------------
def train_model(model, name, epochs=14, lr=3e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    best_val_acc = 0.0
    best_path = f"best_{name.lower().replace(' ', '_')}.pth"

    print(f"\nTraining {name} ({epochs} epochs)")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct, total = 0, 0
        for images, labels in tqdm(train_loader, desc=f"{name} Epoch {epoch+1:02d}/{epochs}"):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        train_accs.append(train_acc)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_accs.append(val_acc)
        val_losses.append(avg_val_loss)

        print(f"{name} - Epoch {epoch+1:02d} → Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"   → Best {name} saved! (Val Acc: {val_acc:.2f}%)")

    return train_accs, val_accs, train_losses, val_losses, best_path, best_val_acc


# ------------------ Train All Models ------------------
scratch_train_acc, scratch_val_acc, scratch_train_loss, scratch_val_loss, scratch_path, scratch_best = train_model(scratch_model, "CNN From Scratch", epochs=8, lr=1e-3)
mobilenet_train_acc, mobilenet_val_acc, mobilenet_train_loss, mobilenet_val_loss, mobilenet_path, mobilenet_best = train_model(mobilenet, "MobileNetV3", epochs=8, lr=3e-4)
torch.cuda.empty_cache()
effnet_train_acc, effnet_val_acc, effnet_train_loss, effnet_val_loss, effnet_path, effnet_best = train_model(effnet, "EfficientNet-B3", epochs=8, lr=3e-4)


# ------------------ Load Best Models & Test Accuracy ------------------
def load_and_test(path, model, name):
    model.load_state_dict(torch.load(path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"{name} - Test Accuracy (10%): {acc:.2f}%")
    return acc

scratch_test = load_and_test(scratch_path, scratch_model, "CNN From Scratch")
mobilenet_test = load_and_test(mobilenet_path, mobilenet, "MobileNetV3")
effnet_test = load_and_test(effnet_path, effnet, "EfficientNet-B3")


# ------------------ Deluxe Plots ------------------
epochs = range(1, len(scratch_val_acc)+1)

# Accuracy Curves (Train + Val)
plt.figure(figsize=(14, 6))
plt.plot(epochs, scratch_train_acc, label='Scratch Train', linestyle='--', color='red')
plt.plot(epochs, scratch_val_acc, label='Scratch Val', color='red')
plt.plot(epochs, mobilenet_train_acc, label='MobileNet Train', linestyle='--', color='orange')
plt.plot(epochs, mobilenet_val_acc, label='MobileNet Val', color='orange')
plt.plot(epochs, effnet_train_acc, label='EfficientNet Train', linestyle='--', color='green')
plt.plot(epochs, effnet_val_acc, label='EfficientNet Val', color='green')
plt.title('Training & Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

# Loss Curves
plt.figure(figsize=(14, 6))
plt.plot(epochs, scratch_train_loss, label='Scratch Train Loss', linestyle='--', color='red')
plt.plot(epochs, scratch_val_loss, label='Scratch Val Loss', color='red')
plt.plot(epochs, mobilenet_train_loss, label='MobileNet Train Loss', linestyle='--', color='orange')
plt.plot(epochs, mobilenet_val_loss, label='MobileNet Val Loss', color='orange')
plt.plot(epochs, effnet_train_loss, label='EfficientNet Train Loss', linestyle='--', color='green')
plt.plot(epochs, effnet_val_loss, label='EfficientNet Val Loss', color='green')
plt.title('Training & Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Test Accuracy Bar
models = ['CNN Scratch', 'MobileNetV3', 'EfficientNet-B3']
test_accs = [scratch_test, mobilenet_test, effnet_test]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, test_accs, color=['red', 'orange', 'green'])
plt.title('Test Accuracy on 10% Hold-out Set')
plt.ylabel('Accuracy (%)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", ha='center')
plt.ylim(0, 100)
plt.show()


# Final Table
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
print(f"{'Model':<25} {'Best Val Acc':<15} {'Test Acc (10%)'}")
print("-"*80)
print(f"{'CNN From Scratch':<25} {scratch_best:.2f}%{' ':>10} {scratch_test:.2f}%")
print(f"{'MobileNetV3':<25} {mobilenet_best:.2f}%{' ':>10} {mobilenet_test:.2f}%")
print(f"{'EfficientNet-B3':<25} {effnet_best:.2f}%{' ':>10} {effnet_test:.2f}%")
print("="*80)







