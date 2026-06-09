import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image


# ===========================
# CONFIGURATION
# ===========================
dataset_path = "/kaggle/input/itj-labs-fruit-classification-challenge/fruit_dataset_10_classes"
train_csv = os.path.join(dataset_path, "train_dataset.csv")
val_csv = os.path.join(dataset_path, "val_dataset.csv")
test_csv = os.path.join(dataset_path, "test_dataset.csv")


# ===========================
# CUSTOM DATASET CLASS
# ===========================
class FruitDataset(Dataset):
    """Custom PyTorch Dataset to load images from CSV."""
    
    def __init__(self, csv_file, transform=None, test=False):
        self.data = pd.read_csv(csv_file)  # Load CSV
        self.transform = transform
        self.test = test
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(dataset_path, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.test:
            return image
        else:
            label = self.data.iloc[idx, 1]  # Label column
            return image, label


# ===========================
# DATA LOADING & TRANSFORMATIONS
# ===========================
transform = Compose([
    Resize((64, 64)),  # Resize images
    ToTensor()
])

train_ds = FruitDataset(train_csv, transform=transform)
val_ds = FruitDataset(val_csv, transform=transform)
test_ds = FruitDataset(test_csv, transform=transform, test=True)

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=batch_size * 2, num_workers=4)


# ===========================
# SIMPLE BUT BAD CNN MODEL
# ===========================
class SimpleBadCNN(nn.Module):
    """A very simple and underperforming CNN model."""
    
    def __init__(self, num_classes):
        super(SimpleBadCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

num_classes = 10  # Adjust based on dataset
model = SimpleBadCNN(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("✅ Simple but bad CNN model initialized.")


# ===========================
# SUBMISSION FILE GENERATION
# ===========================
def generate_submission_file(model, test_loader, device, filename="submission.csv"):
    """Generates a Kaggle-compatible submission file."""
    model.eval()
    y_pred = []
    image_ids = list(range(len(test_loader.dataset)))
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
    
    submission = pd.DataFrame({
        'ID': image_ids[:len(y_pred)],
        'Outcome': y_pred
    })
    
    submission.to_csv(filename, index=False)
    print(f"✅ Submission file '{filename}' created successfully!")

# ===========================
# FINAL NOTES
# ===========================

generate_submission_file(model, test_loader, device, filename="submission.csv")
print("✅ Participants should improve the CNN model!")


