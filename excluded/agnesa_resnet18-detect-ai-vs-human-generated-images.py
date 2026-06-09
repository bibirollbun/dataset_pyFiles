import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch import tensor
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import os


# Define the root directory of the dataset
dataset_root = '/kaggle/input/ai-vs-human-generated-dataset/'
# Load the train CSV file
train_df = pd.read_csv(os.path.join(dataset_root, 'train.csv'))
# Load the test CSV file
test_df = pd.read_csv(os.path.join(dataset_root, 'test.csv'))

# Split into training and validation (80% train, 20% validation)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")


# Define the custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 1])  # Use the path directly from CSV
        image = Image.open(img_path).convert('RGB')
        label = int(self.df.iloc[idx, 2])  # Convert label to integer
        label = tensor(label, dtype=torch.long)  # Convert to PyTorch tensor
        if self.transform:
            image = self.transform(image)
        return image, label


class TestImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]  # Get file name
        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])  # Use the first column (id/file_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name  # Return file name along with image


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = ImageDataset(train_df, dataset_root, transform=transform)
val_dataset = ImageDataset(val_df, dataset_root, transform=transform)
test_dataset = TestImageDataset(test_df, dataset_root, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader =  DataLoader(test_dataset, batch_size=32, shuffle=False)


from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: AI-generated and human-created


# Freeze early layers (optional)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers
for param in model.fc.parameters():
    param.requires_grad = True  # Unfreeze the final layer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


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
        
        running_loss += loss.item()
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")


# Make predictions on test set
model.eval()
predictions = []

with torch.no_grad():
    for images, _ in test_loader:  # Ignore labels if they exist in the dataset
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# Ensure IDs are correctly extracted from test_df
submission_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})

# Save predictions to CSV
submission_df.to_csv('submission.csv', index=False)

# Check the first few rows
print(submission_df.head())




