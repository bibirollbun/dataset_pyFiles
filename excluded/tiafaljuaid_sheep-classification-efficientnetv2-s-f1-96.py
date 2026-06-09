# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from PIL import ImageOps
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Hyperparameters & Device 
SEED = 42
NUM_CLASSES = 7
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 30
IMG_SIZE = 300 #384
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility Setup
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

# Paths & Label Mapping  
label_map = {"Naeimi": 0, "Goat": 1, "Sawakni": 2, "Roman": 3, "Najdi": 4, "Harri": 5, "Barbari": 6}
inv_label_map = {v: k for k, v in label_map.items()}

DATA_DIR = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")

df = pd.read_csv(LABELS_CSV)


# Class Distribution (%)
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x='label', hue='label', order=df['label'].value_counts().index, 
                   palette="viridis", legend=False)
plt.title("Class Distribution (%)")
plt.xticks(rotation=45)
total = len(df)

for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black')

plt.tight_layout()
plt.show()



num_images_per_class = 1  # one image per breed
classes = df['label'].unique()
samples = []

# Take one sample image from each breed
for label in classes:
    sample_df = df[df['label'] == label].sample(num_images_per_class, random_state=SEED)
    samples.extend(sample_df.itertuples())

total_images = len(samples)
n_cols = 3  # number of columns
n_rows = (total_images + n_cols - 1) // n_cols  # calculate rows needed

plt.figure(figsize=(n_cols * 5, n_rows * 5))  # bigger figure size

for idx, row in enumerate(samples):
    img_path = os.path.join(TRAIN_DIR, row.filename)
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.expand(image, border=6, fill='white')  # slightly thicker border

    ax = plt.subplot(n_rows, n_cols, idx + 1)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(row.label, fontsize=14, fontweight='bold', pad=10)
    ax.title.set_position([.5, 1.05])  # center title above image

plt.suptitle("Samples of Sheep Breeds", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.4)
plt.show()


# Data Augmentation & Normalization
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Improve robustness with color variations
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# loads images and applies transforms
class SheepDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        label = label_map[self.df.iloc[idx]['label']]
        if self.transform: image = self.transform(image)
        return image, label

class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image


# EfficientNetV2-S pretrained on ImageNet, adjusted for 7 classes
def get_model():
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model

# Training with Stratified K-Fold
all_labels = df['label'].map(label_map).values
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n--- Fold {fold+1} ---")
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(SheepDataset(train_df, TRAIN_DIR, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SheepDataset(val_df, TRAIN_DIR, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Use class-weighted CrossEntropyLoss to handle class imbalance
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_f1, early_stop = 0, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_preds, train_labels = [], []
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_preds.extend(out.argmax(1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                out = model(x)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(y.numpy())
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        scheduler.step(val_f1)
        print(f"Epoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"/kaggle/working/model_fold{fold}.pth")
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= PATIENCE:
            print("Early stopping.")
            break




print("\n--- Validation Performance Across Folds ---")

all_probs, all_preds, all_labels = [], [], []

def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.any(mask):
            bin_accuracy = np.mean(accuracies[mask])
            bin_confidence = np.mean(confidences[mask])
            ece += (np.sum(mask) / len(probs)) * np.abs(bin_confidence - bin_accuracy)

    return ece

for fold, (_, val_idx) in enumerate(skf.split(df, df['label'])):
    val_df = df.iloc[val_idx].reset_index(drop=True)
    val_loader = DataLoader(SheepDataset(val_df, TRAIN_DIR, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(f"/kaggle/working/model_fold{fold}.pth"))
    model.eval()

    fold_probs, fold_preds, fold_labels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            fold_probs.extend(probs)
            fold_preds.extend(np.argmax(probs, axis=1))
            fold_labels.extend(y.numpy())

    all_probs.extend(fold_probs)
    all_preds.extend(fold_preds)
    all_labels.extend(fold_labels)

    print(f"Fold {fold+1} F1 Score: {f1_score(fold_labels, fold_preds, average='macro'):.4f}")

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=list(label_map.keys())))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Compute ECE
ece_score = compute_ece(np.array(all_probs), np.array(all_labels))
print(f"\nExpected Calibration Error (ECE): {ece_score:.4f}")


# Test Time Augmentation (TTA) + Weighted Ensemble Integration

# Add this transform block before inference section
tta_transforms = [
    val_transform,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
]

# Replace inference block starting from: print("\n--- Inference on Test Set ---") with the following:

print("\n--- Inference on Test Set with TTA + Weighted Ensemble ---")
test_files = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({"filename": test_files})

# Store each fold's weighted TTA prediction
all_outputs = []
fold_f1_scores = []

for fold, (_, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"Inferencing with Fold {fold+1} model...")
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(f"/kaggle/working/model_fold{fold}.pth"))
    model.eval()

    tta_outputs = []
    for tta_transform in tta_transforms:
        test_loader = DataLoader(TestDataset(test_df, TEST_DIR, tta_transform), batch_size=BATCH_SIZE, shuffle=False)
        outputs = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(DEVICE)
                out = model(images)
                outputs.append(torch.softmax(out, dim=1).cpu().numpy())
        tta_outputs.append(np.concatenate(outputs))

    # Average the TTA outputs for this fold
    tta_avg_output = np.mean(tta_outputs, axis=0)
    all_outputs.append(tta_avg_output)

    # Collect fold F1 score for ensemble weighting
    val_df = df.iloc[val_idx].reset_index(drop=True)
    val_loader = DataLoader(SheepDataset(val_df, TRAIN_DIR, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    val_preds, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            out = model(x)
            val_preds.extend(out.argmax(1).cpu().numpy())
            val_labels.extend(y.numpy())
    f1 = f1_score(val_labels, val_preds, average="macro")
    fold_f1_scores.append(f1)
    print(f"Fold {fold+1} Val F1: {f1:.4f}")

# Weighted average of all TTA predictions
weights = np.array(fold_f1_scores)
weights = weights / weights.sum()

weighted_output = np.zeros_like(all_outputs[0])
for i in range(NUM_FOLDS):
    weighted_output += all_outputs[i] * weights[i]

final_preds = np.argmax(weighted_output, axis=1)
final_labels = [inv_label_map[p] for p in final_preds]

submission = pd.DataFrame({"filename": test_files, "label": final_labels})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved weighted TTA submission to submission.csv")

