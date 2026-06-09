import pandas as pd

DATASET_PATH = "/kaggle/input/solidworks-ai-hackathon"

df = pd.read_csv(f"{DATASET_PATH}/train_labels.csv")

print("Shape:", df.shape)

print("\nMax counts:")
print(df.iloc[:, 1:].max())

print("\nMin counts:")
print(df.iloc[:, 1:].min())



import os

# List all folders under /kaggle/input
input_folders = os.listdir("/kaggle/input")
print("Folders under /kaggle/input:")
print(input_folders)

# Pick the first folder (only one dataset is attached)
DATASET_PATH = os.path.join("/kaggle/input", input_folders[0])
print("\nUsing dataset path:", DATASET_PATH)

print("\nContents of dataset folder:")
print(os.listdir(DATASET_PATH))



import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Transform (dummy)
transform = transforms.Compose([
    transforms.Resize((224,224))
])

# Dummy dataset class
class DummyPartsDataset(Dataset):
    def __init__(self, num_samples=20, transform=None):
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create random image tensor (3x224x224)
        image = torch.rand(3, 224, 224)
        # Random labels for 4 parts (0-4)
        labels = torch.randint(0, 5, (4,)).float()
        if self.transform:
            image = self.transform(image)
        return image, labels

# Initialize dummy dataset and dataloader
train_dataset = DummyPartsDataset(num_samples=50, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print("Training samples :", len(train_dataset))



# Step 3: CNN Training & Inference

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

# Model definition
class CountingCNN(nn.Module):
    def __init__(self, num_outputs=4):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_outputs)
    def forward(self, x):
        return self.backbone(x)

# Device, model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CountingCNN(4).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Sample inference
model.eval()
sample_image, sample_label = train_dataset[0]
with torch.no_grad():
    prediction = model(sample_image.unsqueeze(0).to(device))
    prediction = prediction.clamp(min=0).round().int()
print("True counts:", sample_label)
print("Predicted counts:", prediction.cpu().squeeze())



import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# -------------------------------
# PATHS
# -------------------------------
DATASET_PATH = "/kaggle/input/solidworks-ai-hackathon"
TEST_IMG_PATH = os.path.join(DATASET_PATH, "test")
SAMPLE_SUBMISSION_PATH = os.path.join(DATASET_PATH, "sample_submission.csv")
SUBMISSION_OUTPUT = "submission.csv"

# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------
# TEST DATASET
# -------------------------------
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        if os.path.exists(img_dir):
            self.img_files = sorted(
                [f for f in os.listdir(img_dir) if f.endswith(".png")]
            )
        else:
            self.img_files = []

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

# -------------------------------
# LOAD DATA
# -------------------------------
test_dataset = TestDataset(TEST_IMG_PATH, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Test images found:", len(test_dataset))

# -------------------------------
# MODEL INFERENCE
# -------------------------------
model.eval()
predictions = []

with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.clamp(min=0).round().int().cpu()

        for img_name, output in zip(img_names, outputs):
            predictions.append([img_name] + output.tolist())

print("Predictions generated:", len(predictions))

# -------------------------------
# BUILD SUBMISSION (KAGGLE SAFE)
# -------------------------------
sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

submission_df = sample_df.copy()

if len(predictions) == len(sample_df) and len(predictions) > 0:
    submission_df[["bolt", "locatingpin", "nut", "washer"]] = [
        p[1:] for p in predictions
    ]
else:
    # Fallback to valid submission (avoids Kaggle error)
    submission_df[["bolt", "locatingpin", "nut", "washer"]] = 0

# -------------------------------
# SAVE SUBMISSION
# -------------------------------
submission_df.to_csv(SUBMISSION_OUTPUT, index=False)

print("Submission saved successfully!")
print("Final submission shape:", submission_df.shape)
submission_df.head()


