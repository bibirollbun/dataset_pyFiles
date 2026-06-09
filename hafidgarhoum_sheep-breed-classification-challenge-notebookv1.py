import os
# Check input directory structure (optional but useful for exploration)
print("Files in /kaggle/input directory:\n")
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

#  Basic libraries for data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  PyTorch and torchvision for deep learning
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

from tqdm import tqdm

#  Image processing
from PIL import Image

#  Scikit-learn for metrics
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


#  Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ðŸ“¥ Load training labels
train_labels_path = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
df = pd.read_csv(train_labels_path)
print("\n Sample of training labels:")
df.head()


# Only keep valid breeds
valid_classes = df['label'].value_counts().index.tolist()
print("Valid breeds:", valid_classes)

# Drop invalid labels
df = df[df['label'].isin(valid_classes)]

# Visualize class distribution
plt.figure(figsize=(10,5))
sns.countplot(x='label', data=df)
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.show()


# Encode labels
label2idx = {label: idx for idx, label in enumerate(valid_classes)}
idx2label = {idx: label for label, idx in label2idx.items()}
df['label_idx'] = df['label'].map(label2idx)

# Split into train/val
train_df, val_df = train_test_split(df, stratify=df['label'], test_size=0.15, random_state=42)

# Albumentations transforms

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


class SheepDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row.filename)
        image = np.array(Image.open(img_path).convert('RGB'))
        label = row.label_idx

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


class SheepModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = SheepModel(len(valid_classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def eval_model(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1, y_true, y_pred


# Paths
train_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'

# Stratified 5-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
epochs = 10

best_models = []
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label_idx'])):
    print(f"\n===== Fold {fold + 1} / 5 =====")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_ds = SheepDataset(train_df, train_dir, train_transform)
    val_ds = SheepDataset(val_df, train_dir, valid_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # Re-init model & optimizer for each fold
    model = SheepModel(len(valid_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_f1, y_true, y_pred = eval_model(model, val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
            print("Saved best model for this fold.")

    fold_results.append(best_f1)
    print(f"Best val macro F1 for fold {fold}: {best_f1}")

print(f"\n=== Average F1 over folds: {np.mean(fold_results):.4f} ===")


# ðŸ§ª Step 6: Inference on Test Set & Submission

test_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test/"
test_files = sorted(os.listdir(test_dir))


class TestDataset(Dataset):
    def __init__(self, file_list, root_dir, transform):
        self.file_list = file_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_list[idx])
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image, self.file_list[idx]


test_ds = TestDataset(test_files, test_dir, valid_transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

# Load all fold models and average predictions (Ensembling)
all_preds = []

for fold in range(5):
    model = SheepModel(len(valid_classes)).to(device)
    model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
    model.eval()

    preds_fold = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds_fold.append(outputs.softmax(dim=1).cpu().numpy())

    preds_fold = np.concatenate(preds_fold, axis=0)
    all_preds.append(preds_fold)

# Average predictions across folds
final_preds = np.mean(all_preds, axis=0)
final_labels_idx = final_preds.argmax(axis=1)
final_labels = [idx2label[idx] for idx in final_labels_idx]

submission = pd.DataFrame({'filename': test_files, 'label': final_labels})
submission.to_csv('submission.csv', index=False)
print(submission.head())


fold_index = 1  # Fold 3 (zero-indexed)
train_idx, val_idx = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, df['label_idx']))[fold_index]
val_df = df.iloc[val_idx]

val_ds = SheepDataset(val_df, train_dir, valid_transform)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

model = SheepModel(len(valid_classes)).to(device)
model.load_state_dict(torch.load(f"best_model_fold{fold_index}.pth"))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in val_loader:  # replace with your validation DataLoader
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['label'])  # Fit using training labels

y_true = le.inverse_transform(y_true)
y_pred = le.inverse_transform(y_pred)

# Now use the confusion matrix code
valid_classes = sorted(df['label'].unique())
cm = confusion_matrix(y_true, y_pred, labels=valid_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=valid_classes, yticklabels=valid_classes, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()




!ls /kaggle/working/


# Paths
train_dir = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'
labels_csv = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'

# Load labels
df_labels = pd.read_csv(labels_csv)

# Classes to compare
classes_to_compare = ['Goat', 'Naeimi', 'Najdi','Sawakni']  # Change/add your classes here

num_images = 5  # Number of images per class to show

fig, axs = plt.subplots(len(classes_to_compare), num_images, figsize=(num_images * 3, len(classes_to_compare) * 3))

for row, class_name in enumerate(classes_to_compare):
    # Filter dataframe by class
    class_df = df_labels[df_labels['label'] == class_name]
    if class_df.empty:
        print(f"No images found for class {class_name}")
        continue
    
    # Random sample of filenames
    sampled_filenames = class_df['filename'].sample(min(num_images, len(class_df))).values

    for col, filename in enumerate(sampled_filenames):
        img_path = os.path.join(train_dir, filename)
        img = Image.open(img_path).convert('RGB')

        # Handle single-row case where axs might be 1D
        ax = axs[row, col] if len(classes_to_compare) > 1 else axs[col]
        ax.imshow(img)
        ax.axis('off')
        if col == 0:
            ax.set_title(class_name, fontsize=14, loc='left')

plt.tight_layout()
plt.show()

