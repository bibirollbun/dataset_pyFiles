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



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.io import read_image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# GPU Memory Management
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------ Paths ------------------
CSV_PATH = '/kaggle/input/cassava-leaf-disease-classification/train.csv'
IMG_DIR  = '/kaggle/input/cassava-leaf-disease-classification/train_images'


# ------------------ Load & Exact 70/20/10 Split ------------------
df = pd.read_csv(CSV_PATH)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df['label'], random_state=42)

print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")


# ------------------ Dataset ------------------
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        image = read_image(os.path.join(IMG_DIR, img_name)).float() / 255.0
        if image.shape[0] == 1: image = image.repeat(3, 1, 1)
        label = self.df.iloc[idx, 1]
        if self.transform: image = self.transform(image)
        return image, label


# ------------------ Transforms & Loaders ------------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CassavaDataset(train_df, transform)
val_dataset   = CassavaDataset(val_df,   transform)
test_dataset  = CassavaDataset(test_df,  transform)

train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=48, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=48, shuffle=False, pin_memory=True)


# ------------------ Model ------------------
model = models.efficientnet_b3(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(1536, 5)
model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)


# ------------------ Training Loop with Train & Val Accuracy per Epoch ------------------
# This loop trains the model for 8 epochs max, fine-tuning to avoid overfitting.
# Checkpoints are saved from epoch 4 to 8 for later comparison.

best_val_acc = 0.0
print(f"{'Epoch':<6} {'Train Accuracy':<16} {'Validation Accuracy':<20} {'Status'}")
print("-" * 60)

for epoch in range(8):
    # Training Phase
    model.train()
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/08 [Training]", leave=False):
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    # Validation Phase
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    status = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        status = "→ BEST MODEL SAVED!"

    # Save checkpoint if in range 4-8
    if epoch + 1 >= 4 and epoch + 1 <= 8:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
        status += " → CHECKPOINT SAVED!"

    print(f"{epoch+1:02d}     {train_acc:6.2f}%          {val_acc:7.2f}%            {status}")


print("\n" + "="*65)
print("TRAINING COMPLETED – FINAL SUMMARY")
print("="*65)
print(f"Best Validation Accuracy (20%): {best_val_acc:.2f}%")
print(f"Last Train Accuracy     (70%): {train_acc:.2f}%")
print(f"Last Validation Accuracy(20%): {val_acc:.2f}%")
print("="*65)


# ------------------ Final Evaluation on Test Set (10%) ------------------
# Load the best checkpoint from comparison (or fallback to best_model.pth) and evaluate on test set.
# Includes accuracy, confusion matrix, and classification report.

# Assuming best_checkpoint is defined from previous cell; fallback if not
try:
    model.load_state_dict(torch.load(best_checkpoint))
except NameError:
    model.load_state_dict(torch.load("best_model.pth"))  # Fallback to original best

model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        correct += (model(images.cuda()).argmax(1) == labels.cuda()).sum().item()
test_acc = correct / len(test_df)

print(f"Accuracy on 10% Test set: {test_acc*100:.2f}\n")

preds = []
trues = []
with torch.no_grad():
    for images, labels in test_loader:
        preds.extend(model(images.cuda()).argmax(1).cpu().numpy())
        trues.extend(labels.numpy())

cm = confusion_matrix(trues, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - 10% Test Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(trues, preds))

