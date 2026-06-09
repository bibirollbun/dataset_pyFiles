# âœ… 1. Setup
!pip install timm py7zr -q

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import timm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# âœ… 2. Extract Dataset
import py7zr

os.makedirs("/kaggle/working/train", exist_ok=True)
os.makedirs("/kaggle/working/test", exist_ok=True)

with py7zr.SevenZipFile("/kaggle/input/cifar-10/train.7z", mode='r') as z:
    z.extractall(path="/kaggle/working/train")

with py7zr.SevenZipFile("/kaggle/input/cifar-10/test.7z", mode='r') as z:
    z.extractall(path="/kaggle/working/test")

# âœ… 3. Custom Dataset Class

labels_df = pd.read_csv("/kaggle/input/cifar-10/trainLabels.csv")
train_image_folder = "/kaggle/working/train/train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class CIFAR10CustomDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform
        self.classes = sorted(self.dataframe.label.unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        label = self.class_to_idx[self.dataframe.iloc[idx, 1]]
        img_path = os.path.join(self.image_folder, f"{img_name}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# âœ… 4. Data Splitting
full_dataset = CIFAR10CustomDataset(labels_df, train_image_folder, transform)

train_len = int(0.7 * len(full_dataset))
val_len = int(0.2 * len(full_dataset))
test_len = len(full_dataset) - train_len - val_len

train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)
test_loader = DataLoader(test_set, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# âœ… 5. Load and Modify Models

def get_model(model_name):
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif model_name == 'resnext':
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
    return model.to(device)

# âœ… 6. Training Function

def train_model(model, name, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        val_acc = correct / total
        val_losses.append(val_loss / len(val_loader))
        print(f"{name.upper()} | Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Acc={val_acc:.4f}")
    
    return train_losses, val_losses

# âœ… 7. Evaluation Function

def evaluate_model(model, name):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            _, preds = outputs.max(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(f"Classification Report for {name.upper()}")
    print(classification_report(y_true, y_pred, target_names=full_dataset.classes))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes, cmap="Blues")
    plt.title(f"Confusion Matrix - {name.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# âœ… 8. Train All Models & Compare

results = {}
models_to_train = ['alexnet', 'vgg', 'resnext', 'vit']

for model_name in models_to_train:
    print(f"\n----- Training {model_name.upper()} -----")
    model = get_model(model_name)
    start = time.time()
    train_losses, val_losses = train_model(model, model_name)
    duration = time.time() - start

    # Inference time
    dummy = torch.randn(1, 3, 224, 224).to(device)
    t0 = time.time()
    for _ in range(10):
        _ = model(dummy)
    inference_time = (time.time() - t0) / 10

    results[model_name] = {
        "model": model,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_time_sec": round(duration, 2),
        "inference_time_ms": round(inference_time * 1000, 2),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    }

    evaluate_model(model, model_name)

 # âœ… 9. Visualization & Summary

df = pd.DataFrame({
    name.upper(): {
        'Train Time (s)': res['train_time_sec'],
        'Inference Time (ms)': res['inference_time_ms'],
        'Params (M)': res['params_m']
    } for name, res in results.items()
}).T

print("ðŸ“Š Model Comparison Summary:")
display(df)

df[['Train Time (s)', 'Inference Time (ms)']].plot(kind='bar', figsize=(10,5))
plt.title("Model Training & Inference Time")
plt.ylabel("Time")
plt.grid(True)
plt.show()

