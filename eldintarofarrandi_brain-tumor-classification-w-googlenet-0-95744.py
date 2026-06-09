import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
from tqdm import tqdm
from torch.utils.data import random_split
import random


dir_path = "/kaggle/input/it1244-brain-tumor-dataset/data/"

train_path = os.path.join(dir_path, "train")
test_path = os.path.join(dir_path, "test")

def count_files_in_folders(path):
    if not os.path.exists(path):
        print(f"Path {path} tidak ditemukan!")
        return 0 
    
    total_files = 0
    for root, dirs, files in os.walk(path):
        print(f"Memeriksa direktori: {root}")  
        total_files += len(files)  
    return total_files

train_data_count = count_files_in_folders(train_path)
test_data_count = count_files_in_folders(test_path)

print("\nJumlah data pada folder Train:")
print(f"Total file: {train_data_count}")

print("\nJumlah data pada folder Test:")
print(f"Total file: {test_data_count}")


train_csv_path = os.path.join(train_path, "data.csv")
test_csv_path = os.path.join(test_path, "data.csv")

class BrainTumorTrainDataset(Dataset):
    def __init__(self, path, csv_file, transform=None):
        self.path = path
        self.data = pd.read_csv(csv_file, header=None, names=["id", "label"])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        img_path = os.path.join(self.path, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Label encoding: benign -> 0, malignant -> 1 (jika label lain ada)
        label = 0 if label == "benign" else 1

        return image, label

class BrainTumorTestDataset(Dataset):
    def __init__(self, path, csv_file, transform=None):
        self.path = path
        self.data = pd.read_csv(csv_file, header=None, names=["id"]) 
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ambil ID gambar
        img_id = self.data.iloc[idx, 0]

        # Load image
        img_path = os.path.join(self.path, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Transformasi (opsional)
        if self.transform:
            image = self.transform(image)

        return image, img_id  


# Step 1: Definisikan transformasi untuk menghitung mean & std (tanpa normalisasi/augmentasi)
temp_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Sesuaikan dengan ukuran di train_transform
    transforms.ToTensor(),
])

# Step 2: Buat dataset sementara dengan transformasi di atas
temp_dataset = BrainTumorTrainDataset(
    train_path, 
    train_csv_path, 
    transform=temp_transform
)

# Step 3: Hitung mean & std per channel
mean = torch.zeros(3)
std = torch.zeros(3)
total_images = len(temp_dataset)

for img, _ in temp_dataset:
    # Hitung mean dan std per channel (RGB)
    mean += img.mean(dim=(1, 2))  # Rata-rata per channel (3 nilai)
    std += img.std(dim=(1, 2))    # Std per channel (3 nilai)

# Rata-rata dari seluruh dataset
mean /= total_images
std /= total_images

print(f"Mean (RGB): {mean.tolist()}")
print(f"Std (RGB): {std.tolist()}")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean.tolist(),  # Gunakan mean yang dihitung
        std=std.tolist()     # Gunakan std yang dihitung
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean.tolist(),
        std=std.tolist()
    ),
])


train_dataset = BrainTumorTrainDataset(train_path, train_csv_path, transform=train_transform)
test_dataset = BrainTumorTestDataset(test_path, test_csv_path, transform=test_transform)

train_ratio = 0.8  
train_size = int(train_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size

train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for images, labels in train_loader:
    print(f"Train - Image batch shape: {images.shape}, Label batch shape: {labels.shape}")
    break

for images, img_ids in test_loader:
    print(f"Test - Image batch shape: {images.shape}, ID batch: {img_ids}")
    break


model = models.googlenet(pretrained=True)

num_classes = 2  
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)


criterion = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.001)  


from sklearn.metrics import roc_auc_score, f1_score

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store labels and predictions for F1 and ROC-AUC calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        # Calculate F1-Score and Train Accuracy
        train_accuracy = 100 * correct / total
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Train F1-Score: {train_f1:.4f}")

        model.eval()
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Store labels and predictions for F1 and ROC-AUC calculation
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())

        # Calculate Validation Accuracy, F1-Score, and ROC-AUC
        val_accuracy = 100 * val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_roc_auc = roc_auc_score(val_labels, val_preds, multi_class='ovr')

        print(f"Validation Accuracy: {val_accuracy:.2f}%, Validation F1-Score: {val_f1:.4f}, Validation ROC-AUC: {val_roc_auc:.4f}")

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)


def predict_and_save(model, test_loader, output_csv="submission.csv"):
    model.eval()
    test_ids = []
    test_preds = []
    label_map = {0: "benign", 1: "malignant"}  

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_ids.extend(ids.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())

    submission_df = pd.DataFrame({
        "id": test_ids,
        "classification": [label_map[pred] for pred in test_preds]
    })

    submission_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")



predict_and_save(model, test_loader, output_csv="submission.csv")


torch.save(model.state_dict(), "model.pth")

