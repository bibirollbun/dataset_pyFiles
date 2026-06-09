import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import timm
from sklearn.model_selection import train_test_split


# --- 1. Configuration ---
# IMPORTANT: Update these paths to match the Division B data
DATA_DIR = "/kaggle/input/grand-xray-slam-division-b/train2"
CSV_PATH = "/kaggle/input/grand-xray-slam-division-b/train2.csv"

MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
NUM_WORKERS = 2 


# --- 2. Load Data and Define Labels ---
df = pd.read_csv(CSV_PATH)

# IMPORTANT: This line helps you find the correct column names to prevent errors!
print("Columns in your CSV file:")
print(df.columns)
print("-" * 25)

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]


class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, image_dir, labels, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        correct_column_name = 'Image_name' 
        img_path = os.path.join(self.image_dir, self.df.iloc[idx][correct_column_name])

        try:
            # --- Try to open the image ---
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # --- If it fails, handle the error ---
            print(f"Warning: Could not load image {img_path}. Error: {e}. Loading a replacement.")
            # Load the first image of the dataset as a fallback
            replacement_path = os.path.join(self.image_dir, self.df.iloc[0][correct_column_name])
            image = Image.open(replacement_path).convert("RGB")

        # Get the labels for the ORIGINAL index
        labels_vector = self.df.iloc[idx][self.labels].values.astype(np.float32)
        labels_tensor = torch.tensor(labels_vector, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels_tensor



# --- 4. Transforms and Data Augmentation ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# --- 5. Data Splitting and Loaders ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = ChestXRayDataset(train_df, DATA_DIR, LABELS, transform=train_transform)
val_dataset = ChestXRayDataset(val_df, DATA_DIR, LABELS, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# --- 6. Model, Loss Function, and Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(LABELS))
model.to(device)

# Calculate positive weights to handle class imbalance
pos_weights = []
for label in LABELS:
    num_pos = df[label].sum()
    num_neg = len(df) - num_pos
    pos_weights.append(num_neg / (num_pos + 1e-6)) # Added epsilon for stability
pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# --- 7. Training and Validation Functions ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(loader)


# --- 8. Main Training Loop ---
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

    val_loss = validate_one_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")

print("\nFinished Training")
print(f"Best validation loss achieved: {best_val_loss:.4f}")

