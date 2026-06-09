import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


# Base paths
root_path = "/kaggle/input/sheep-classification-challenge-2025"
image_dir = os.path.join(root_path, "Sheep Classification Images")
csv_files = ["dummy_sub.csv", "train_labels.csv"]

# 1. Explore directory structure
print(f"Exploring: {image_dir}")
print("-" * 50)

items = os.listdir(image_dir)
for item in items:
    full_path = os.path.join(image_dir, item)
    if os.path.isdir(full_path):
        print(f"[DIR]  {item}")
    else:
        print(f"[FILE] {item}")

# 2. Count images in each subfolder
image_counts = {}
for item in items:
    folder_path = os.path.join(image_dir, item)
    if os.path.isdir(folder_path):
        img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_counts[item] = len(img_files)

if image_counts:
    print("\nImage counts per folder (possibly breed names):")
    for folder, count in image_counts.items():
        print(f"{folder}: {count} images")

# 3. Load and preview CSVs
for csv_file in csv_files:
    csv_path = os.path.join(image_dir, csv_file)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\nPreview of '{csv_file}':")
        print(df.head())
    else:
        print(f"CSV not found: {csv_file}")

# 4. Show label distribution in train_labels.csv
train_labels_path = os.path.join(image_dir, "train_labels.csv")
if os.path.exists(train_labels_path):
    df_train = pd.read_csv(train_labels_path)
    print("\nClass distribution in 'train_labels.csv':")
    print(df_train['label'].value_counts())



class SheepDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, label2id=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        # Use external label2id mapping if provided (e.g., for test set)
        if label2id is None:
            self.label2id = {label: idx for idx, label in enumerate(df['label'].unique())}
        else:
            self.label2id = label2id

        self.id2label = {idx: label for label, idx in self.label2id.items()}

        # Map label strings to IDs (skip for test with no labels)
        if 'label' in self.df.columns:
            self.df['label_id'] = self.df['label'].map(self.label2id)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if 'label_id' in self.df.columns:
            label = self.df.loc[idx, 'label_id']
            return image, label
        else:
            return image, img_name  # For test set



def get_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



def get_dataloader(df, img_dir, transform, batch_size=32, shuffle=True, label2id=None):
    dataset = SheepDataset(df, img_dir, transform=transform, label2id=label2id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset



# Paths for Kaggle
train_img_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train"
train_labels_path = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv"

# Load label CSV
df = pd.read_csv(train_labels_path)

# Get transformations
train_transform = get_transforms()

# Create DataLoader and Dataset
train_loader, train_dataset = get_dataloader(df, train_img_dir, train_transform, batch_size=32, shuffle=True)

# Store label-to-ID mapping for later use
label2id = train_dataset.label2id



# Define test image directory (Kaggle path)
test_img_dir = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test"

# Create dataframe of test image filenames
df_test = pd.DataFrame(sorted(os.listdir(test_img_dir)), columns=['filename'])

# Get the same transforms used for training/validation (usually just normalization and resizing)
test_transform = get_transforms()

# Create test DataLoader and Dataset
test_loader, test_dataset = get_dataloader(
    df_test,
    test_img_dir,
    test_transform,
    batch_size=32,
    shuffle=False,
    label2id=label2id  # for consistent label encoding (if needed)
)



# Compute class frequency and percentage
label_counts = df['label'].value_counts()
label_percentages = df['label'].value_counts(normalize=True) * 100

# Combine into a single DataFrame
class_stats = pd.DataFrame({
    'Count': label_counts,
    'Percentage': label_percentages.round(2)
})

# Display results
print("ğŸ“Š Class Frequencies and Percentages in Training Set:")
print(class_stats)



# Count label frequencies
label_counts = df['label'].value_counts().sort_values(ascending=False).reset_index()
label_counts.columns = ['label', 'count']

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=label_counts, x='label', y='count', palette="Set2")
plt.title("Class Distribution in Training Set", fontsize=14)
plt.xlabel("Sheep Breed", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Get reverse label mapping
id2label = train_dataset.id2label

# Unnormalize function
def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean

# Plot 5 random images
plt.figure(figsize=(15, 5))
for i in range(5):
    idx = random.randint(0, len(train_dataset) - 1)
    image, label_id = train_dataset[idx]
    image = unnormalize(image).permute(1, 2, 0).numpy().clip(0, 1)
    label = id2label[label_id]

    plt.subplot(1, 5, i+1)
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')

plt.tight_layout()
plt.show()



# ---------------- PATHS ----------------
DATA_DIR = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images"
IMAGE_DIR = os.path.join(DATA_DIR, "train")
LABEL_CSV = os.path.join(DATA_DIR, "train_labels.csv")

# ---------------- CONSTANTS ----------------
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 9
NUM_EPOCHS = 10
NUM_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- DATASET ----------------
class SheepDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------- MODEL FACTORY ----------------
from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    vit_b_16, ViT_B_16_Weights,
)

def get_model(name, num_classes):
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(DEVICE)

# ---------------- TRAIN / EVAL LOOP ----------------
def train_one_fold(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images).argmax(dim=1).cpu().numpy()
            preds.extend(outputs)
            truths.extend(labels.numpy())
    return accuracy_score(truths, preds), f1_score(truths, preds, average='macro')

# ---------------- MAIN CROSS-VALIDATION LOOP ----------------
df = pd.read_csv(LABEL_CSV)
df['label'] = pd.factorize(df['label'])[0]  # Convert breed names to integer labels

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

f1_results = {}
acc_results = {}

model_names = ["resnet50", "densenet121", "efficientnet_b0", "mobilenet_v3_large", "convnext_tiny", "vit_b_16"]

for model_name in model_names:
    print(f"\nğŸ”� Evaluating: {model_name.upper()}")
    fold_acc, fold_f1 = [], []

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"ğŸ”� Fold {fold+1}")
        train_ds = Subset(SheepDataset(df, IMAGE_DIR, transform), train_idx)
        val_ds = Subset(SheepDataset(df, IMAGE_DIR, transform), val_idx)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model = get_model(model_name, NUM_CLASSES)
        acc, f1 = train_one_fold(model, train_loader, val_loader)
        print(f"âœ… Fold {fold+1} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        fold_acc.append(acc)
        fold_f1.append(f1)

    acc_results[model_name] = np.mean(fold_acc)
    f1_results[model_name] = np.mean(fold_f1)

# ---------------- SUMMARY ----------------
print("\nğŸ“Š Summary of Mean Scores by Model:")
print(f"{'Model':<25} {'Accuracy':<10} {'F1 Score':<10}")
print("-" * 50)
for name in model_names:
    print(f"{name:<25} {acc_results[name]:<10.4f} {f1_results[name]:<10.4f}")

# ğŸ�† Best model
best_model_name = max(f1_results, key=f1_results.get)
print(f"\nğŸ�† Best Model: {best_model_name} with F1: {f1_results[best_model_name]:.4f}")


