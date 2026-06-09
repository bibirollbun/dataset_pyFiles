import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms




BASE_DIR = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images'
df = pd.read_csv(os.path.join(BASE_DIR, 'train_labels.csv'))
df['filepath'] = df['filename'].apply(lambda x: os.path.join(BASE_DIR, 'train', x))




le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))

label_map


import matplotlib.pyplot as plt
from PIL import Image


sample_df = df.sample(9, random_state=42).reset_index(drop=True)


plt.figure(figsize=(12, 8))


for idx, row in sample_df.iterrows():
    image = Image.open(row['filepath'])
    plt.subplot(3, 3, idx + 1)
    plt.imshow(image)
    plt.title(f"Breed: {row['label']}")
    plt.axis('off')

plt.tight_layout()
plt.show()



train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)




class SheepDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.loc[idx, 'filepath']).convert('RGB')
        label = self.df.loc[idx, 'label_encoded']
        if self.transform:
            image = self.transform(image)
        return image, label




IMAGE_SIZE = 224
BATCH_SIZE = 32

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
}

train_dataset = SheepDataset(train_df, transform=data_transforms['train'])
val_dataset = SheepDataset(val_df, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)




class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE//8) * (IMAGE_SIZE//8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = df['label_encoded'].nunique()

model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



EPOCHS = 40
patience = 10
trigger_times = 0

train_loss_history, val_loss_history = [], []
best_val_acc = 0.0
best_val_loss = float('inf')
best_epoch = -1
best_model_path = 'best_model.pth'

for epoch in range(EPOCHS):
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
    
    avg_train_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_loss_history.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    
    if val_acc > best_val_acc and avg_val_loss < best_val_loss:
        best_val_acc = val_acc
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)
        print(f"   Best model updated at Epoch {best_epoch} with Val Acc: {best_val_acc:.2f}% and Val Loss: {best_val_loss:.4f}")
        trigger_times = 0  # Reset patience counter
    else:
        trigger_times += 1
        print(f"  âš ï¸� No improvement. Trigger times: {trigger_times}/{patience}")
        if trigger_times >= patience:
            print(f"\nğŸ›‘ Early stopping triggered at epoch {epoch+1} due to no improvement in validation performance.")
            break

print(f"\n Training Complete. Best Val Acc: {best_val_acc:.2f}% at Epoch {best_epoch}")



MODEL_PATH = 'sheep_cnn.pth'
torch.save(model.state_dict(), MODEL_PATH)



model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=le.classes_))



plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()


import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os


label_map = {
    0: 'Barbari',
    1: 'Goat',
    2: 'Harri',
    3: 'Naeimi',
    4: 'Najdi',
    5: 'Roman',
    6: 'Sawakni'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

import torch.nn as nn

class SheepCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SheepCNN, self).__init__()
        self.features = nn.Sequential(   # âœ… changed from conv_layers
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(  
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model
model = SheepCNN(num_classes=7)
model.load_state_dict(torch.load('sheep_cnn.pth', map_location=device))
model.to(device)
model.eval()

# Test directory
test_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test"

# Transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Get all image filenames
image_filenames = sorted(os.listdir(test_dir))
results = []

# Predict each image
for filename in image_filenames:
    img_path = os.path.join(test_dir, filename)
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = label_map[pred_idx]  

    results.append({
        "Filename": filename,
        "Label": pred_label
    })


submission_df = pd.DataFrame(results)
submission_df.to_csv("sheep_submission.csv", index=False)

print(" Submission CSV saved as 'sheep_submission.csv'")



import os
import shutil

# Create the output directory if it doesn't exist
os.makedirs("/kaggle/working/output", exist_ok=True)

# Copy the model into this directory
shutil.copy('/kaggle/working/sheep_cnn.pth', '/kaggle/working/output/sheep_cnn.pth')





