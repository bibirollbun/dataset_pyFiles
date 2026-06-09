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






import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')




# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # Paths
    DATA_PATH = "../input/cassava-leaf-disease-classification"
    TRAIN_PATH = "../input/cassava-leaf-disease-classification/train_images/"
    TEST_PATH = "../input/cassava-leaf-disease-classification/test_images/"
    
    # Model
    IMG_SIZE = 384
    N_CLASSES = 5
    
    # Training
    MODE = "train"  # "train" or "inference"
    N_FOLDS = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]  # Which folds to train
    N_EPOCHS = 15
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 2  # Gradient accumulation
    
    # Optimizer
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    SCHEDULER = "cosine"  # "cosine" or "step"
    
    # Augmentation
    MIXUP_ALPHA = 0.2
    LABEL_SMOOTHING = 0.1
    
    # Other
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = True  # Mixed precision training
    
    # Inference
    USE_TTA = True
    
cfg = Config()




# =============================================================================
# SEED
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(cfg.SEED)




# =============================================================================
# CUSTOM CNN MODEL - BUILT FROM SCRATCH
# =============================================================================
class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.skip(residual)
        out = self.relu(out)
        return out

class CassavaNet(nn.Module):
    """Custom CNN for Cassava Classification - Built from scratch"""
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x




# =============================================================================
# DATA AUGMENTATION
# =============================================================================
def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(size=(cfg.IMG_SIZE, cfg.IMG_SIZE), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=cfg.IMG_SIZE//8, max_width=cfg.IMG_SIZE//8, 
                        fill_value=0, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])




# =============================================================================
# DATASET
# =============================================================================
class CassavaDataset(Dataset):
    def __init__(self, df, img_path, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_path = img_path
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image_id']
        label = self.df.loc[idx, 'label']
        img_path = os.path.join(self.img_path, img_name)
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        return image, label

class TestDataset(Dataset):
    def __init__(self, image_ids, img_path, transforms=None):
        self.image_ids = image_ids
        self.img_path = img_path
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_path, img_name)
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        return image




# =============================================================================
# MIXUP
# =============================================================================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [TRAIN]')
    optimizer.zero_grad()
    
    for i, (images, labels) in enumerate(pbar):
        images = images.to(cfg.DEVICE)
        labels = labels.to(cfg.DEVICE)
        
        # Mixup
        if cfg.MIXUP_ALPHA > 0 and np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)
            use_mixup = True
        else:
            use_mixup = False
        
        # Forward with AMP
        if cfg.USE_AMP:
            with autocast():
                outputs = model(images)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
                loss = loss / cfg.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % cfg.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler and cfg.SCHEDULER == "step":
                    scheduler.step()
        else:
            outputs = model(images)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss = loss / cfg.ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % cfg.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler and cfg.SCHEDULER == "step":
                    scheduler.step()
        
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        total_loss += loss.item() * cfg.ACCUMULATION_STEPS
        total_acc += acc.item()
        
        pbar.set_postfix({'loss': loss.item() * cfg.ACCUMULATION_STEPS, 'acc': acc.item()})
    
    return total_loss / len(loader), total_acc / len(loader)

def validate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [VALID]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
    
    return total_loss / len(loader), total_acc / len(loader)

def train_fold(fold, train_df, valid_df):
    print(f'\n{"="*70}')
    print(f'FOLD {fold + 1}')
    print(f'{"="*70}')
    
    # Datasets
    train_dataset = CassavaDataset(train_df, cfg.TRAIN_PATH, get_train_transforms())
    valid_dataset = CassavaDataset(valid_df, cfg.TRAIN_PATH, get_valid_transforms())
    
    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    
    # Model
    model = CassavaNet(num_classes=cfg.N_CLASSES)
    model.to(cfg.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    
    # Scheduler
    if cfg.SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.N_EPOCHS)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    scaler = GradScaler() if cfg.USE_AMP else None
    
    # Training loop
    best_acc = 0
    for epoch in range(cfg.N_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, epoch
        )
        valid_loss, valid_acc = validate(model, valid_loader, criterion, epoch)
        
        if cfg.SCHEDULER == "cosine":
            scheduler.step()
        
        print(f'\nEpoch {epoch+1}/{cfg.N_EPOCHS}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            print(f'✅ New best! Saving model...')
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold': fold,
                'epoch': epoch,
                'accuracy': best_acc
            }, f'cassava_fold{fold}_best.pth')
    
    print(f'\nFold {fold+1} Best Accuracy: {best_acc:.4f}')
    return best_acc




# =============================================================================
# INFERENCE
# =============================================================================
def predict_tta(model, image_path):
    """Predict with Test Time Augmentation"""
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # TTA transforms
    tta_transforms = [
        get_valid_transforms(),
        A.Compose([
            A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([
            A.Resize(height=cfg.IMG_SIZE, width=cfg.IMG_SIZE),
            A.Rotate(limit=15, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]
    
    predictions = []
    with torch.no_grad():
        for transform in tta_transforms:
            aug_img = transform(image=image)['image']
            aug_img = aug_img.unsqueeze(0).to(cfg.DEVICE)
            output = model(aug_img)
            probs = F.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy())
    
    return np.mean(predictions, axis=0)

def inference():
    print(f'\n{"="*70}')
    print('INFERENCE MODE')
    print(f'{"="*70}')
    
    # Find models
    import glob
    model_files = sorted(glob.glob('cassava_fold*_best.pth'))
    
    if len(model_files) == 0:
        print("❌ ERROR: No trained models found!")
        print("Please run training first with cfg.MODE = 'train'")
        return None
    
    print(f'Found {len(model_files)} trained model(s)')
    
    # Load test data
    test_df = pd.read_csv(os.path.join(cfg.DATA_PATH, 'sample_submission.csv'))
    
    all_predictions = []
    
    for model_file in model_files:
        print(f'\nLoading: {model_file}')
        
        model = CassavaNet(num_classes=cfg.N_CLASSES)
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(cfg.DEVICE)
        model.eval()
        
        predictions = []
        
        for img_id in tqdm(test_df['image_id'], desc='Predicting'):
            img_path = os.path.join(cfg.TEST_PATH, img_id)
            
            if cfg.USE_TTA:
                pred = predict_tta(model, img_path)
            else:
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
                image = get_valid_transforms()(image=image)['image']
                image = image.unsqueeze(0).to(cfg.DEVICE)
                
                with torch.no_grad():
                    output = model(image)
                    pred = F.softmax(output, dim=1).cpu().numpy()
            
            predictions.append(pred[0])
        
        all_predictions.append(np.array(predictions))
    
    # Ensemble
    final_predictions = np.mean(all_predictions, axis=0)
    final_labels = np.argmax(final_predictions, axis=1)
    
    # Create submission
    submission = pd.DataFrame({
        'image_id': test_df['image_id'],
        'label': final_labels
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print('\n✅ Submission saved!')
    print(f'\nPrediction distribution:')
    print(submission.label.value_counts().sort_index())
    
    return submission




# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f'Device: {cfg.DEVICE}')
    print(f'Mode: {cfg.MODE}')
    print(f'Image Size: {cfg.IMG_SIZE}')
    print(f'Custom CNN Architecture: CassavaNet')
    
    if cfg.MODE == 'train':
        # Load data
        df = pd.read_csv(os.path.join(cfg.DATA_PATH, 'train.csv'))
        
        # K-Fold CV
        skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
        
        fold_scores = []
        
        for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df['label'])):
            if fold not in cfg.TRAIN_FOLDS:
                continue
            
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[valid_idx]
            
            best_acc = train_fold(fold, train_df, valid_df)
            fold_scores.append(best_acc)
        
        print(f'\n{"="*70}')
        print('TRAINING COMPLETE')
        print(f'Average CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}')
        print(f'{"="*70}')
        
        print('\n✅ To run inference, set: cfg.MODE = "inference"')
    
    elif cfg.MODE == 'inference':
        submission = inference()
        if submission is not None:
            print('\n✅ Ready to submit!')
    
    else:
        print(f'Invalid MODE: {cfg.MODE}. Use "train" or "inference"')

if __name__ == '__main__':
    main()


# =============================================================================
# RUN INFERENCE AFTER TRAINING
# ADD THIS AS A NEW CELL AFTER THE MAIN CELL
# =============================================================================

import gc
import time

# Clear memory after training
gc.collect()
torch.cuda.empty_cache()
time.sleep(2)

print("\n" + "="*70)
print("SWITCHING TO INFERENCE MODE")
print("="*70)

# Change mode to inference
cfg.MODE = "inference"

# Run main again for inference
main()










