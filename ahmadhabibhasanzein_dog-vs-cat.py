import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import os
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from collections import Counter


# buat folder output
os.makedirs("/kaggle/working/data/train", exist_ok=True)

# unzip train.zip
with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/train.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/data/train")


base_path = "/kaggle/working/data/train/train" 
output_path = "/kaggle/working/data/train_split"

os.makedirs(f"{output_path}/cat", exist_ok=True)
os.makedirs(f"{output_path}/dog", exist_ok=True)

for fname in os.listdir(base_path):
    if fname.startswith("cat"):
        shutil.move(os.path.join(base_path, fname), f"{output_path}/cat/{fname}")
    elif fname.startswith("dog"):
        shutil.move(os.path.join(base_path, fname), f"{output_path}/dog/{fname}")


base_path = "/kaggle/working/data/train_split"

classes = ["cat", "dog"]
num_images = 5

plt.figure(figsize=(15, 6))

index = 1
for cls in classes:
    class_path = os.path.join(base_path, cls)
    images = os.listdir(class_path)[:num_images]

    for img_name in images:
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path)

        plt.subplot(2, num_images, index)
        plt.imshow(img)
        plt.title(f"{cls}")
        plt.axis("off")

        index += 1

plt.tight_layout()
plt.show()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder("/kaggle/working/data/train_split", transform=transform)

# Ambil semua indeks & label
indices = list(range(len(dataset)))
labels = [dataset.samples[i][1] for i in indices]

# Stratified split
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print("=== Distribusi Kelas Overall ===")
print(Counter(labels))

print("\n=== Distribusi Kelas TRAIN ===")
print(Counter([labels[i] for i in train_idx]))

print("\n=== Distribusi Kelas VAL ===")
print(Counter([labels[i] for i in val_idx]))

# Buat subset sesuai indeks hasil split
train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

# DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

print("\nTotal:", len(dataset))
print("Train:", len(train_ds))
print("Val:", len(val_ds))

# Mapping
print("\nClass mapping:", dataset.class_to_idx)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze semua feature layers
for param in model.parameters():
    param.requires_grad = False

# Ganti FC untuk 2 kelas
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()

        train_acc = correct / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()

        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss/len(val_loader):.4f}  Acc: {val_acc:.4f}\n")

    return model


model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)


torch.save(model.state_dict(), "resnet50_dogcat.pth")

