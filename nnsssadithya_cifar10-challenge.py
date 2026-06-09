import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


train_index = pd.read_csv('/kaggle/input/cifar-10neucalssification/train_labels.csv')
train_index.head(5)


val_index = pd.read_csv('/kaggle/input/cifar-10neucalssification/val_labels.csv')
val_index.head(5)


label_decode = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def base_path(BASE_PATH,data_index):
    data_index['full_path'] = data_index['filename'].apply(lambda x: os.path.join(BASE_PATH, x))
    # 2. Collect the files and their labels
    image_paths = data_index['full_path'].tolist()
    labels = data_index['label'].tolist()
    return image_paths,labels
    


# 3. Visualize images with labels
def visualize_images(image_paths, labels, num_images=9, figsize=(15, 15)):
    """
    Visualize images in a grid with their labels
    
    Parameters:
    - image_paths: list of image file paths
    - labels: list of corresponding labels
    - num_images: number of images to display (default 9)
    - figsize: figure size for the plot
    """
    # Limit to available images or requested number
    num_images = min(num_images, len(image_paths))
    
    # Calculate grid dimensions
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx in range(num_images):
        try:
            # Load and display image
            img = Image.open(image_paths[idx])
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f'Label: {label_decode[labels[idx]]}', fontsize=12, fontweight='bold')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error loading image\n{e}', 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


train_data,train_label = base_path("/kaggle/input/cifar-10neucalssification/train/",train_index)


print(train_data[0])
print(label_decode[train_label[0]])


# Visualize the first 10 images
visualize_images(train_data, train_label, num_images=10)


val_data,val_label = base_path("/kaggle/input/cifar-10neucalssification/val/",val_index)


print(val_data[0])
print(label_decode[val_label[0]])


visualize_images(val_data, val_label, num_images=10)


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Convert labels to unique integers
        self.unique_labels = list(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.label_to_idx[self.labels[idx]]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

# Keep validation transform simple (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


train_set = ImageDataset(train_data, train_label, transform=train_transform)


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# Test it
print(f"Dataset size: {len(train_set)}")
print(f"Number of classes: {len(train_set.unique_labels)}")
print(f"Classes: {train_set.unique_labels}")

# Get one batch
images, labels = next(iter(train_loader))
print(f"\nBatch image shape: {images.shape}")
print(f"Batch labels shape: {labels.shape}")


val_set = ImageDataset(val_data, val_label, transform=val_transform)


val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

# Test it
print(f"Dataset size: {len(val_set)}")
print(f"Number of classes: {len(val_set.unique_labels)}")
print(f"Classes: {val_set.unique_labels}")

# Get one batch
images, labels = next(iter(val_loader))
print(f"\nBatch image shape: {images.shape}")
print(f"Batch labels shape: {labels.shape}")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split



# Simple Small CNN for 32x32 images
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Setup (using your existing train_loader and val_loader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(train_set.unique_labels)
model = SmallCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)

print(f"Device: {device}")
print(f"Classes: {num_classes}")
print(f"Train: {len(train_set)}, Val: {len(val_set)}\n")

# Training
num_epochs = 35

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    # Validate
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train: Loss={train_loss/len(train_loader):.3f}, Acc={train_acc:.2f}%")
    print(f"  Val:   Loss={val_loss/len(val_loader):.3f}, Acc={val_acc:.2f}%\n")

print("Training completed!")


def predict_image(image_path, model, device):
    """
    Predict the label of an image
    
    Args:
        image_path: Path to the image file
        model: Trained model
        device: Device to run prediction on
    
    Returns:
        predicted_label: The predicted class label
        confidence: Confidence score (probability)
    """
    # Define the same transform used during training
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    # Get the label name
    idx_to_label = {idx: label for label, idx in train_set.label_to_idx.items()}
    predicted_label = idx_to_label[predicted_idx.item()]
    
    return predicted_label, confidence.item()


# Example usage:
# Single prediction
test_image_path = '/kaggle/input/cifar-10neucalssification/test/00020.png'
image = Image.open(test_image_path).convert('RGB')
plt.imshow(image)
plt.show()
predicted_label, confidence = predict_image(test_image_path, model, device)
print(f"Predicted: {predicted_label} , Decode: {label_decode[predicted_label]}")
print(f"Confidence: {confidence*100:.2f}%")


test_data = pd.read_csv('/kaggle/input/cifar-10neucalssification/sample_submission.csv')
test_data.head(5)


test_data['full_path'] = test_data['filename'].apply(lambda x: os.path.join("/kaggle/input/cifar-10neucalssification/test/", x))
image_paths = test_data['full_path'].tolist()
print(image_paths[0])


def predict_multiple(image_paths, model, device):
    """Predict labels for multiple images"""
    results = []
    for img_path in image_paths:
        label, conf = predict_image(img_path, model, device)
        results.append({
            'image_path': img_path,
            'predicted_label': label,
            'confidence': conf
        })
    return results


results = predict_multiple(image_paths,model,device)
predicted_labels = [item['predicted_label'] for item in results]


col0 = pd.read_csv("/kaggle/input/cifar-10neucalssification/sample_submission.csv").drop(['label'],axis=1)


col1 = pd.Series(predicted_labels, name='label')
submission = pd.concat([col0,col1],axis = 1)
submission.to_csv("submission.csv",index=False)

