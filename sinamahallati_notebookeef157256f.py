import os
import random
import json
import time

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
N_FOLDS = 5
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 5

DATA_DIR   = '/kaggle/input/cassava-leaf-disease-classification'
TRAIN_DIR  = os.path.join(DATA_DIR, 'train_images')
TEST_DIR   = os.path.join(DATA_DIR, 'test_images')
TRAIN_CSV  = os.path.join(DATA_DIR, 'train.csv')
SAMPLE_SUB = os.path.join(DATA_DIR, 'sample_submission.csv')

print('Using device:', DEVICE)
print('Data dir:', DATA_DIR)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


train_df = pd.read_csv(TRAIN_CSV)
print(train_df.head())
print('\nClass distribution:')
print(train_df['label'].value_counts())


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = Image.open(img_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        if self.is_test:
            return image, row['image_id']

        label = int(row['label'])
        return image, label

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes=NUM_CLASSES):
    
    model = None
    
    try:
        weights_enum = getattr(models, 'ResNet50_Weights', None)
        if weights_enum is not None:
            weights = weights_enum.IMAGENET1K_V2
            model = models.resnet50(weights=weights)
            print("Loaded ResNet50 with IMAGENET1K_V2 weights.")
    except Exception as e:
        print("Could not load new API weights:", e)
        
    if model is None:
        try:
            model = models.resnet50(pretrained=True)
            print("Loaded ResNet50 with pretrained=True.")
        except Exception as e:
            print("Could not load pretrained weights, using random init:", e)
            model = models.resnet50(pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(model, train_loader, criterion, optimizer,
                    epoch, scaler, device=DEVICE):
    model.train()
    running_loss = 0.0
    running_correct = 0
    num_samples = 0

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = outputs.max(1)
        batch_size = labels.size(0)
        num_samples += batch_size
        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()

        if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
            print(
                f"Epoch {epoch} Step {step+1}/{len(train_loader)} "
                f"Loss {running_loss/num_samples:.4f} "
                f"Acc {running_correct/num_samples:.4f}",
                end='\r'
            )

    epoch_loss = running_loss / num_samples
    epoch_acc = running_correct / num_samples
    return epoch_loss, epoch_acc


def validate_one_epoch(model, valid_loader, criterion, device=DEVICE):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    num_samples = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast(enabled=(device == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, preds = outputs.max(1)
            batch_size = labels.size(0)
            num_samples += batch_size
            running_loss += loss.item() * batch_size
            running_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / num_samples
    epoch_acc = running_correct / num_samples
    return epoch_loss, epoch_acc


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(train_df['image_id'], train_df['label']))

oof_predictions = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
oof_targets = train_df['label'].values

for fold, (train_idx, val_idx) in enumerate(folds):
    print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")

    train_data = train_df.iloc[train_idx].reset_index(drop=True)
    val_data   = train_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = CassavaDataset(train_data, TRAIN_DIR,
                                   transforms=train_transforms)
    val_dataset   = CassavaDataset(val_data,   TRAIN_DIR,
                                   transforms=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler(enabled=(DEVICE == 'cuda'))

    best_val_acc = 0.0
    best_model_path = f'/kaggle/working/best_model_fold{fold}.pth'

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, scaler, device=DEVICE
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device=DEVICE
        )

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"\nFold {fold+1} Epoch {epoch}/{EPOCHS} "
            f"Train loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved best model for fold {fold+1} with val_acc {best_val_acc:.4f}")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_outputs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            with autocast(enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
            all_outputs.append(probs.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    oof_predictions[val_idx] = all_outputs

    fold_acc = (oof_predictions[val_idx].argmax(axis=1) == oof_targets[val_idx]).mean()
    print(f"Fold {fold+1} OOF accuracy: {fold_acc:.4f}")



oof_pred_labels = oof_predictions.argmax(axis=1)
oof_accuracy = (oof_pred_labels == oof_targets).mean()
print(f"OOF accuracy across all folds: {oof_accuracy:.4f}")


sub_df = pd.read_csv(SAMPLE_SUB)
print('Sample submission head:')
print(sub_df.head())

test_df = sub_df.copy()

test_dataset = CassavaDataset(
    test_df, TEST_DIR, transforms=val_transforms, is_test=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

all_preds = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)

for fold in range(N_FOLDS):
    print(f"Inference with fold {fold+1} model...")

    model = build_model().to(DEVICE)
    best_model_path = f'/kaggle/working/best_model_fold{fold}.pth'
    state_dict = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    fold_preds = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(DEVICE)
            with autocast(enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
            fold_preds.append(probs.cpu().numpy())

    fold_preds = np.concatenate(fold_preds, axis=0)
    all_preds += fold_preds / N_FOLDS  # average over folds

pred_labels = all_preds.argmax(axis=1)
sub_df['label'] = pred_labels

sub_df.to_csv('submission.csv', index=False)
print('submission.csv saved!')
sub_df.head()




