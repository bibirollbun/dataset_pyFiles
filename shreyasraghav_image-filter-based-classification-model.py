import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR

# -----------------------------
# Dataset Class for Training and Validation
# -----------------------------
class AIVSHumanDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing images and labels for training and validation.

    Args:
        csv_file (str): Path to the CSV file containing image file names and their corresponding labels.
        root_dir (str): Directory containing the images. Leave empty if the paths in CSV are absolute.
        transform (callable, optional): A function/transform to apply to the images.
    """
    def __init__(self, csv_file, root_dir='', transform=None):
        self.data = pd.read_csv(csv_file)  # Load the CSV file into a Pandas DataFrame
        self.root_dir = root_dir          # Set the root directory for images
        self.transform = transform        # Define any image transformations

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its label.
        """
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['file_name'])  # Get the full image path
        image = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB
        label = int(self.data.iloc[idx]['label'])    # Get the corresponding label

        if self.transform:
            image = self.transform(image)           # Apply transformations if provided

        return image, label

# -----------------------------
# Data Augmentation and Transformations
# -----------------------------
# Train transformations: These augmentations increase data variability to improve model generalization.
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally with a 50% probability
    transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color properties
    transforms.RandomAffine(degrees=15, scale=(0.8, 1.2)),  # Apply random scaling
    transforms.RandomErasing(p=0.2),  # Randomly erase part of the image to simulate occlusions
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
])

# Validation transformations: Keep these minimal to avoid data augmentation during evaluation.
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------
# Data Splitting and Dataset Creation
# -----------------------------
csv_path = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'  # Path to the training CSV file
df = pd.read_csv(csv_path)  # Load the CSV file into a DataFrame

# Split the data into training and validation sets (80% train, 20% validation)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Save the split data into separate CSV files
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)

# Create PyTorch Dataset objects for training and validation
train_dataset = AIVSHumanDataset(
    csv_file='train_split.csv',
    root_dir='/kaggle/input/ai-vs-human-generated-dataset/',  # Root directory for images
    transform=train_transforms
)

val_dataset = AIVSHumanDataset(
    csv_file='val_split.csv',
    root_dir='/kaggle/input/ai-vs-human-generated-dataset/',
    transform=val_transforms
)

# -----------------------------
# DataLoaders
# -----------------------------
# DataLoader parameters
batch_size = 96  # Number of samples per batch
num_workers = 2  # Number of worker threads for data loading

# Create DataLoaders for training and validation
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle data during training
    num_workers=num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,  # No shuffling during validation
    num_workers=num_workers
)

# -----------------------------
# Model Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

# Load a pre-trained ResNeXt model
model = models.resnext50_32x4d(pretrained=True)  # Alternatively, use resnext101_32x8d for a deeper model

# Freeze early layers if needed (optional)
for param in model.parameters():
    param.requires_grad = True  # Unfreeze all layers for fine-tuning

# Replace the fully connected layer for binary classification (2 classes)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),  # Add dropout for regularization
    nn.Linear(num_ftrs, 2)  # Binary classification
)

model = model.to(device)  # Move the model to the selected device

# -----------------------------
# Loss Function and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # AdamW optimizer with weight decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs

# -----------------------------
# Training and Validation Functions
# -----------------------------
def train_one_epoch(model, criterion, optimizer, dataloader, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        criterion: Loss function.
        optimizer: Optimizer for model parameters.
        dataloader: DataLoader for training data.
        device: Device to perform computations on (CPU or GPU).

    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device

        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        loss.backward()  # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()  # Update model parameters

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)  # Get predictions
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, criterion, dataloader, device):
    """
    Validates the model for one epoch.

    Args:
        model: The PyTorch model.
        criterion: Loss function.
        dataloader: DataLoader for validation data.
        device: Device to perform computations on (CPU or GPU).

    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"Validation Metrics: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return epoch_loss, epoch_acc

# -----------------------------
# Training Loop
# -----------------------------
num_epochs = 50  # Number of training epochs
best_val_acc = 0.0  # Track the best validation accuracy

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device)
    val_loss, val_acc = validate_one_epoch(model, criterion, val_loader, device)

    scheduler.step()  # Step the learning rate scheduler

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save model checkpoint if validation accuracy improves
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), f'model_best.pth')
        best_val_acc = val_acc
        print(f"Best model saved at epoch {epoch+1}")

print("Training completed!")



# -----------------------------
# 1. Load test.csv
# -----------------------------
test_df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/test.csv')  

# -----------------------------
# 2. Define a test Dataset
# -----------------------------
class AIVSHumanTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the file_name from the row
        file_name = '/kaggle/input/ai-vs-human-generated-dataset/' + self.df.iloc[idx]['id']
        # Load image
        image = Image.open(file_name).convert('RGB')
        # Apply transforms (if any)
        if self.transform:
            image = self.transform(image)
        return image

# -----------------------------
# 3. Create test transforms
# -----------------------------
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 4. Instantiate the Dataset & DataLoader
# -----------------------------
test_dataset = AIVSHumanTestDataset(test_df, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# 5. Load your trained model
# -----------------------------
# Ensure the model is loaded properly with its trained weights
model = models.resnext50_32x4d(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
model.load_state_dict(torch.load('model_best.pth'))  # Load the best model checkpoint
model.eval()
model = model.to(device)

# -----------------------------
# 6. Generate predictions
# -----------------------------
all_preds = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)  # shape [batch_size, 2] if using nn.CrossEntropyLoss
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy().tolist())

# -----------------------------
# 7. Create submission DataFrame
# -----------------------------
submission_df = pd.DataFrame({
    'id': test_df['id'],      # Matches the IDs from test.csv
    'label': all_preds        # Model's predictions
})

# -----------------------------
# 8. Save submission (no index)
# -----------------------------
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv!")























