import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os



# --- Configuration ---
DATA_DIR = "/kaggle/input/grand-xray-slam-division-a/train1"
CSV_PATH = "/kaggle/input/grand-xray-slam-division-a/train1.csv" 
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10


# --- Load Data ---

df = pd.read_csv(CSV_PATH)
print("Actual columns in your CSV file:")
print(df.columns)
print("-" * 25)

# --- Define Labels ---
LABELS = [ ... ]


# --- Define Labels ---
# These are the 14 conditions we need to predict
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]


# --- Handle Missing Values ---

# Fill missing Age with the median age
median_age = df['Age'].median()
df.loc[:, 'Age'] = df['Age'].fillna(median_age)

# Fill missing Sex with the most frequent value (mode)
mode_sex = df['Sex'].mode()[0]
df.loc[:, 'Sex'] = df['Sex'].fillna(mode_sex)

# We might also want to normalize Age
df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())

# Let's verify
print("NaNs after cleaning:")
print(df[['Age', 'Sex']].isnull().sum())


class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, image_dir, labels, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # --- FIX: Changed 'Image_Name' to 'Image_name' to match CSV file ---
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['Image_name'])
        
        image = Image.open(img_path).convert("RGB")
        
        labels_vector = self.df.iloc[idx][self.labels].values.astype(np.float32)
        labels_tensor = torch.tensor(labels_vector, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels_tensor


# ImageNet stats for normalization
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


# Calculate positive weights for the loss function
# weight = number of negative samples / number of positive samples
pos_weights = []
for label in LABELS:
    num_pos = df[label].sum()
    num_neg = len(df) - num_pos
    pos_weights.append(num_neg / num_pos)

pos_weights = torch.tensor(pos_weights, dtype=torch.float32)


from sklearn.model_selection import train_test_split
import timm

# --- Create the Model ---
def create_model(num_classes):
    model = timm.create_model(MODEL_NAME, pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


# --- Split Data into Training and Validation Sets ---
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Create Datasets and DataLoaders ---
train_dataset = ChestXRayDataset(train_df, DATA_DIR, LABELS, transform=train_transform)
val_dataset = ChestXRayDataset(val_df, DATA_DIR, LABELS, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# --- Instantiate Model, Loss, and Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=len(LABELS)).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training and Validation Functions ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f"Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")
    return running_loss / len(loader)


# --- THIS IS THE MISSING FUNCTION ---
def validate_one_epoch(model, loader, criterion, device):
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(loader)


# --- Main Training Loop ---
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    
    # Training phase
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

    # Validation phase
    val_loss = validate_one_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

    # Save the best model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")

print("Finished Training")
print(f"Best validation loss achieved: {best_val_loss:.4f}")

