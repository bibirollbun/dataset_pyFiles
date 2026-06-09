import os
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm.auto import tqdm
import torch.nn.functional as F

class Config:
    DATA_PATH = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/'
    IMG_SIZE = 416
    BATCH_SIZE = 2
    GRAD_ACCUMULATION_STEPS = 8
    EPOCHS = 20
    MODEL_NAME = 'convnext_large.fb_in22k_ft_in1k_384'
    LEARNING_RATE = 5e-5
    LR_MIN = 1e-6
    WARMUP_EPOCHS = 1
    WEIGHT_DECAY = 1e-6
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0
    MIXUP_CUTMIX_ALPHA = 0.5

def get_transforms(img_size):
    train_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.3, scale_limit=0.2, rotate_limit=30, border_mode=0),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.CoarseDropout(max_holes=8, max_height=int(img_size*0.1), max_width=int(img_size*0.1), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    valid_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transforms, valid_transforms

class SheepDataset(Dataset):
    def __init__(self, df, transforms=None, is_test=False):
        self.df = df
        self.transforms = transforms
        self.is_test = is_test
        self.image_dir = os.path.join(Config.DATA_PATH, 'test' if is_test else 'train')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        if self.transforms:
            image = self.transforms(image=image)['image']
        if self.is_test:
            return image, row['filename']
        else:
            return image, torch.tensor(row['label_int'], dtype=torch.long)

def mixup_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(Config.DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(Config.DEVICE)
    y_a, y_b = y, y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss, correct_predictions, total_samples = 0, 0, 0
    device_type = Config.DEVICE.split(':')[0]
    progress = tqdm(dataloader, desc='Training', leave=False)
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(progress):
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        total_samples += labels.size(0)
        
        r = np.random.rand()
        if Config.MIXUP_CUTMIX_ALPHA > 0 and r < 0.5:
            images, targets_a, targets_b, lam = mixup_data(images, labels, Config.MIXUP_CUTMIX_ALPHA)
        elif Config.MIXUP_CUTMIX_ALPHA > 0 and r < 0.9:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, Config.MIXUP_CUTMIX_ALPHA)
        else:
            targets_a, targets_b, lam = labels, labels, 1.0

        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        loss = loss / Config.GRAD_ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        
        if (i + 1) % Config.GRAD_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        scheduler.step()
        total_loss += loss.item() * Config.GRAD_ACCUMULATION_STEPS
        
        preds = outputs.argmax(dim=1)
        correct_predictions += (lam * (preds == targets_a).sum().item() + (1 - lam) * (preds == targets_b).sum().item())
        progress.set_postfix(loss=loss.item() * Config.GRAD_ACCUMULATION_STEPS, lr=scheduler.get_last_lr()[0])

    return total_loss / len(dataloader), correct_predictions / total_samples

@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    total_loss, correct_predictions, total_samples = 0, 0, 0
    device_type = Config.DEVICE.split(':')[0]
    for images, labels in dataloader:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        total_samples += len(labels)
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item()
        correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(dataloader), correct_predictions / total_samples

def main():
    print(f"Using device: {Config.DEVICE}")
    df = pd.read_csv(os.path.join(Config.DATA_PATH, 'train_labels.csv'))

    class_names = sorted(df['label'].unique())
    class_to_int = {name: i for i, name in enumerate(class_names)}
    int_to_class = {i: name for i, name in enumerate(class_names)}
    df['label_int'] = df['label'].map(class_to_int)
    
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

    train_transforms, valid_transforms = get_transforms(Config.IMG_SIZE)
    train_dataset = SheepDataset(train_df, transforms=train_transforms)
    val_dataset = SheepDataset(val_df, transforms=valid_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=len(class_names), drop_path_rate=0.25)
    model.to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    scaler = torch.amp.GradScaler(enabled=(Config.DEVICE == 'cuda'))
    
    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE, total_steps=num_train_steps,
        pct_start=float(Config.WARMUP_EPOCHS) / Config.EPOCHS
    )

    best_accuracy = 0
    best_model_path = 'best_model.pth'

    for epoch in range(Config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            print(f"ðŸš€ Accuracy improved! Saving model to {best_model_path}")
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    print("\nTraining finished. Loading best model for inference with TTA.")
    model.load_state_dict(torch.load(best_model_path))

    test_image_dir = os.path.join(Config.DATA_PATH, 'test')
    test_filenames = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    test_df = pd.DataFrame({'filename': test_filenames})
    
    test_dataset = SheepDataset(test_df, transforms=valid_transforms, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, num_workers=Config.NUM_WORKERS)

    model.eval()
    predictions, filenames = [], []
    device_type = Config.DEVICE.split(':')[0]
    with torch.no_grad():
        for images, fns in tqdm(test_loader, desc='Predicting with TTA'):
            images = images.to(Config.DEVICE)
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                outputs_original = model(images)
                outputs_flipped = model(torch.flip(images, dims=[3]))
            avg_probs = (F.softmax(outputs_original, dim=1) + F.softmax(outputs_flipped, dim=1)) / 2
            preds = avg_probs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
            filenames.extend(fns)

    submission_df = pd.DataFrame({'filename': filenames, 'label_int': predictions})
    submission_df['label'] = submission_df['label_int'].map(int_to_class)
    final_submission = submission_df[['filename', 'label']]
    final_submission.to_csv('submission.csv', index=False)

    print("\n Submission file created successfully!")

if __name__ == '__main__':
    main()




