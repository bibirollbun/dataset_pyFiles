!pip install iterative-stratification albumentations


import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from transformers import ViTForImageClassification

# Check for multiple GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}, Number of GPUs: {torch.cuda.device_count()}")


# Load data
df = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/train1.csv')
df.dropna(inplace=True)
label_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
              'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
              'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
df = df[['Image_name', 'Patient_ID'] + label_cols]

# Aggregate by patient
patient_df = df.groupby('Patient_ID')[label_cols].max().reset_index()
patient_ids = patient_df['Patient_ID']
patient_labels = patient_df[label_cols]

# Multi-label stratified split (single 80-20 split)
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in msss.split(patient_ids, patient_labels):
    train_patients = patient_ids.iloc[train_idx]
    val_patients = patient_ids.iloc[val_idx]
    train_df = df[df['Patient_ID'].isin(train_patients)]
    val_df = df[df['Patient_ID'].isin(val_patients)]
print(f"Train shape {train_df.shape}, Val shape {val_df.shape}")
print("Train label distribution:\n", train_df[label_cols].mean())
print("Val label distribution:\n", val_df[label_cols].mean())

# Verify no patient overlap
assert len(set(train_df['Patient_ID']) & set(val_df['Patient_ID'])) == 0, "Patient overlap detected!"

# Remove Patient_ID and save to CSV
train_df = train_df[['Image_name'] + label_cols]
val_df = val_df[['Image_name'] + label_cols]
train_df.to_csv('/kaggle/working/train_fold_1.csv', index=False)
val_df.to_csv('/kaggle/working/val_fold_1.csv', index=False)

# Verify saved files
print("\nSaved train CSV columns:", pd.read_csv('/kaggle/working/train_fold_1.csv').columns.tolist())
print("Saved val CSV columns:", pd.read_csv('/kaggle/working/val_fold_1.csv').columns.tolist())


# Define separate pipelines for training (with augmentation) and validation (only resizing/normalization)
def get_transforms(train=False):
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(p=0.3),
            A.Normalize(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])),
            ToTensorV2()
        ])

class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, label_cols: list, transforms=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.transforms = transforms
        self.labels = label_cols
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['Image_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = torch.tensor(row[self.labels].values.astype(float), dtype=torch.float32)
        augmented = self.transforms(image=image)
        image = augmented['image']
        return image if self.is_test else (image, labels)

# Create dataset instances
train_dataset = ChestXrayDataset(
    df=train_df,
    image_dir='/kaggle/input/grand-xray-slam-division-a/train1',
    label_cols=label_cols,
    transforms=get_transforms(train=True)
)

val_dataset = ChestXrayDataset(
    df=val_df,
    image_dir='/kaggle/input/grand-xray-slam-division-a/train1',
    label_cols=label_cols,
    transforms=get_transforms(train=False)
)

# --- Multi-GPU change: Increase batch size to utilize both GPUs ---
batch_size = 64  # Doubled from 32 to leverage GPU memory (adjust if OOM occurs)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# Load model
model = ViTForImageClassification.from_pretrained(
    'codewithdark/vit-chest-xray',
    num_labels=len(label_cols),
    problem_type='multi_label_classification',
    ignore_mismatched_sizes=True
)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)

# Print parameter stats
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")


# weighted focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha=alpha,
        self.gamma=gamma,
        self.reduction=reduction
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        if self.alpha is not None:
            BCE_loss = self.alpha[0] * BCE_loss
        F_loss = (1-pt)**self.gamma[0] * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        else :
            return F_loss.sum()


# Loss function with positive weights
pos_weights = torch.tensor([len(train_df) / (train_df[col].sum() + 1e-6) for col in label_cols]).float().to(device)
loss_fn = FocalLoss(alpha=pos_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

# Training loop
epochs = 10  # Define epochs (was missing in original code)
best_auc = 0.0


for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []

    train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch}/{epochs} Training")

    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        
        train_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        train_labels.append(labels.detach().cpu().numpy())
        
        train_loop.set_postfix(loss=loss.item())
    epoch_train_loss = train_loss / len(train_dataset)
    train_preds = np.vstack(train_preds)
    train_labels = np.vstack(train_labels)
    train_auc = roc_auc_score(train_labels, train_preds, average='macro')

    # Validation
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []

    val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch}/{epochs} Validation")

    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * images.size(0)

            val_preds.append(torch.sigmoid(outputs).cpu().numpy())
            val_labels.append(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_dataset)
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        val_auc = roc_auc_score(val_labels, val_preds, average='macro')

        print(f"\nEpoch {epoch}/{epochs} Complete: Train Loss: {epoch_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val AUC: {val_auc:.4f}")

        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), '/kaggle/working/best_model.pth')
            else:
                torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            print(f"Checkpoint Model saved. best AUC: {best_auc}")
        


# Generate Sample Submission
sample_submission = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv')


test_dataset = ChestXrayDataset(
    df=sample_submission,
    image_dir='/kaggle/input/grand-xray-slam-division-a/test1',
    label_cols=label_cols,
    transforms=get_transforms(train=False),
    is_test=True
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
# test_loader = pl.MpDeviceLoader(test_loader, device)

# Load model
model = ViTForImageClassification.from_pretrained(
    'codewithdark/vit-chest-xray',
    num_labels=len(label_cols),
    problem_type='multi_label_classification',
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load('best_model.pth'))
print("model laoded successfully.")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model = model.to(device)

model.eval()
predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images).logits
        batch_preds = torch.sigmoid(outputs).cpu().numpy()
        predictions.append(batch_preds)

predictions = np.vstack(predictions)
predictions = predictions[:len(sample_submission)]

submission_df = sample_submission.copy()
submission_df[label_cols] = predictions
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created: submission.csv")

