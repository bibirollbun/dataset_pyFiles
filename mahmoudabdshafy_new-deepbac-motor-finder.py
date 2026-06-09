import random
import numpy as np
import pandas as pd
import torch
import os
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2
from PIL import Image, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("Loading ResNet50 model with offline weights...")
weights_path = "/kaggle/input/resnet50/pytorch/default/1/resnet50_weights.pth"
if os.path.exists(weights_path):
    model = resnet50()  # Initialize ResNet50
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Model loaded successfully with offline ResNet50 weights!")
else:
    print("Weights file not found. Loading pretrained ResNet50 from torchvision...")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    print("Model loaded with pretrained weights from torchvision.")

# Modify the first convolution layer if needed (keeping the original MaxPool)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model = model.to(device)
print("Cell 2: Completed")


base_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
train_dir = os.path.join(base_path, "train")
test_dir = os.path.join(base_path, "test")
train_labels_path = os.path.join(base_path, "train_labels.csv")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")

print("Base directory contents:", os.listdir(base_path))
print("Cell 3: Completed")


train_labels = pd.read_csv(train_labels_path)
# Filter out rows with -1 values in motor axes
train_labels = train_labels[(train_labels["Motor axis 0"] != -1.0) &
                            (train_labels["Motor axis 1"] != -1.0) &
                            (train_labels["Motor axis 2"] != -1.0)]
print("Number of valid samples after filtering:", len(train_labels))
print("Cell 4: Completed")


class BacterialDataset(Dataset):
    def __init__(self, root_dir, labels_df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Map each tomo_id to its motor coordinates
        self.labels_dict = {
            str(row["tomo_id"]).lower(): (row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"])
            for _, row in labels_df.iterrows()
        }
        # Get image file paths from each valid tomo directory
        self.filepaths = [
            os.path.join(root, f)
            for tomo_id in self.labels_dict.keys()
            if os.path.exists(os.path.join(root_dir, tomo_id))
            for root, _, files in os.walk(os.path.join(root_dir, tomo_id))
            for f in files if f.lower().endswith('.jpg')
        ]
        print("Number of training images found:", len(self.filepaths))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from {img_path}")
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        # Extract tomo_id from the parent directory name (in lowercase)
        tomo_id = os.path.basename(os.path.dirname(img_path)).lower()
        if tomo_id not in self.labels_dict:
            raise ValueError(f"tomo_id {tomo_id} not found in labels dictionary.")
        label = torch.tensor(self.labels_dict[tomo_id], dtype=torch.float32)
        return image, label


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get image file paths from all directories
        self.filepaths = [
            os.path.join(root, f)
            for root, _, files in os.walk(root_dir)
            for f in files if f.lower().endswith('.jpg')
        ]
        print("Number of test images found:", len(self.filepaths))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from {img_path}")
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        tomogram_id = os.path.basename(os.path.dirname(img_path))
        return image, tomogram_id

print("Cell 5: Completed")


train_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.LANCZOS),
    transforms.Lambda(lambda x: x.filter(ImageFilter.SHARPEN)),
    transforms.RandomResizedCrop(224, interpolation=Image.LANCZOS),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.LANCZOS),
    transforms.Lambda(lambda x: x.filter(ImageFilter.SHARPEN)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
print("Cell 6: Completed")


train_dataset = BacterialDataset(train_dir, train_labels, transform=train_transform)
test_dataset = TestDataset(test_dir, transform=test_transform)

print("Processing training dataset file paths...")
for i in tqdm(range(len(train_dataset)), desc="Training dataset loaded", total=len(train_dataset)):
    pass
print(f"Training dataset loaded with {len(train_dataset)} images.")

print("Processing test dataset file paths...")
for i in tqdm(range(len(test_dataset)), desc="Test dataset loaded", total=len(test_dataset)):
    pass
print(f"Test dataset loaded with {len(test_dataset)} images.")

# Optionally, use ConcatDataset to repeat training data if needed
train_dataset = ConcatDataset([train_dataset] * 1)
print(f"After concatenation, training dataset has {len(train_dataset)} samples.")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                          num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                         num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True)
print("Cell 7: Completed")


num_ftrs = model.fc.in_features

# Modify the final layers of the model
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 3)
)
model = model.to(device)
print("Model architecture (ResNet50):")
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
print("Cell 8: Completed")


# You can increase the number of epochs to improve training; currently num_epochs is set to 1 as a test run.
num_epochs = 3
print("Starting training using ResNet50...")

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

print("Training complete!")
print("Cell 9: Completed")


tomogram_predictions = {}
model.eval()
with torch.no_grad():
    for image, tomogram_id in tqdm(test_loader, desc="Testing"):
        image = image.to(device)
        output = model(image)[0].tolist()
        # Ensure tomogram_id is a string (if a list, take the first element)
        tomogram_id = tomogram_id[0] if isinstance(tomogram_id, list) else tomogram_id
        if tomogram_id not in tomogram_predictions:
            tomogram_predictions[tomogram_id] = [output]
        else:
            tomogram_predictions[tomogram_id].append(output)

final_predictions = []
for tomo_id, preds in tomogram_predictions.items():
    preds_array = np.array(preds)
    avg_coords = preds_array.mean(axis=0)
    # If the average coordinates are less than -0.5, consider that no motor is present
    if np.mean(avg_coords) < -0.5:
        final_predictions.append([tomo_id, -1, -1, -1])
    else:
        final_predictions.append([tomo_id, avg_coords[0], avg_coords[1], avg_coords[2]])

submission_df = pd.DataFrame(final_predictions, columns=["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
submission_df = submission_df.fillna(method='ffill')
submission_df = submission_df.applymap(lambda x: -1 if (isinstance(x, (int, float)) and (x < -1e5 or x > 1e5)) else x)
submission_df = submission_df[["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]]

submission_df.to_csv("submission.csv", index=False)
print("Submission file created successfully!")
print("First 5 predictions:")
print(submission_df.head())
print("Cell 9: Completed")

