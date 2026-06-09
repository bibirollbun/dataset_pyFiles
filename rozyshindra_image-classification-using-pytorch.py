# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings 
warnings.filterwarnings('ignore')


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import zipfile
import os

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/train")

with zipfile.ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/test")



train_dir = "/kaggle/working/train/train"
test_dir = "/kaggle/working/test/test"
submission_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"


# Train folder
train_files = os.listdir(train_dir)
train_files = sorted(train_files)  # optional: to keep filenames in order

print("Train samples:", train_files[:5])

# Test folder
test_files = os.listdir(test_dir)
test_files = sorted(test_files)

print("Test samples:", test_files[:5])



# Show 5 train images
plt.figure(figsize=(15, 3))
for i in range(5):
    img_path = os.path.join(train_dir, train_files[i])
    img = Image.open(img_path)
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(train_files[i])
    plt.axis("off")
plt.suptitle("Train Samples")
plt.show()

# Show 5 test images
plt.figure(figsize=(15, 3))
for i in range(5):
    img_path = os.path.join(test_dir, test_files[i])
    img = Image.open(img_path)
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(test_files[i])
    plt.axis("off")
plt.suptitle("Test Samples")
plt.show()



BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
IMG_SIZE = 224


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        label = 1 if 'dog' in image_name else 0  # dog=1, cat=0 (for training)

        if self.transform:
            image = self.transform(image)

        return image, label



transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = CustomDataset(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)



model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification (dog/cat)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)



for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()

    acc = 100.0 * correct / len(train_dataset)
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")



class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir), key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_name

test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



model.eval()
predictions = []

with torch.no_grad():
    for images, image_names in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        for img_name, pred in zip(image_names, predicted):
            id_ = int(img_name.split('.')[0])
            predictions.append((id_, pred.item()))


submission_df = pd.DataFrame(predictions, columns=["id", "label"])
submission_df = submission_df.sort_values(by="id")
submission_df.to_csv("submission.csv", index=False)


import pandas as pd
import torch.nn.functional as F

model.eval()
submission = []

with torch.no_grad():
    for images, image_names in tqdm(test_loader):
        images = images.to(device)

        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)  # get probabilities
        dog_probs = probabilities[:, 1]  # probability that the image is a dog (class 1)

        for img_name, prob in zip(image_names, dog_probs):
            img_id = int(img_name.split('.')[0])
            submission.append((img_id, prob.item()))

# Sort by id to match the required order
submission.sort(key=lambda x: x[0])

# Create a dataframe
df_submission = pd.DataFrame(submission, columns=["id", "label"])

# Save to CSV
df_submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
df_submission.head()



len(df_submission)





import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNNModel(nn.Module):
    def __init__(self):
        super(CustomNNModel, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 3, 224, 224] -> [B, 32, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> [B, 32, 112, 112]
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [B, 64, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> [B, 64, 56, 56]
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> [B, 128, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> [B, 128, 28, 28]
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # -> [B, 256, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> [B, 256, 14, 14]
        )

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 2)  # Output layer (2 classes: cat, dog)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNNModel().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



train_losses = []
train_accuracies = []
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / len(train_dataset)

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")



import matplotlib.pyplot as plt

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, marker='o', color='green')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()



history = {'loss': [], 'acc': []}

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / len(train_dataset)

    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")



import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plotting training and validation accuracy/loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Assuming `all_labels` are true labels and `all_preds` are predicted labels on validation data
def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


