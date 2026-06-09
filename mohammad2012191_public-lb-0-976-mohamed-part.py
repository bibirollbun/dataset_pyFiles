# Sheep Classification with ConvNeXt V2 Small

## Complete Training Pipeline

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import random
import timm
import warnings
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

# ==================== SETTINGS ====================
# Model settings
MODEL_NAME = 'convnext_xlarge_384_in22ft1k'
IMG_SIZE = 384
NUM_CLASSES = 7

# Training settings
BATCH_SIZE = 16
EPOCHS = 10
NUM_FOLDS = 5
NUM_WORKERS = 2

# Learning rate settings
LR_START = 1e-4
LR_END = 1e-6
WARMUP_EPOCHS = 0

# Early stopping
PATIENCE = 999999999

# Augmentation settings
MIXUP_ALPHA = 0.25
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5
CUTMIX_PROB = 0.5

# Random augmentation parameters
ROTATION_DEGREES = 45
SCALE_RANGE = (0.8, 1.2)
BRIGHTNESS_RANGE = 0.3
CONTRAST_RANGE = 0.3
SATURATION_RANGE = 0.3
HUE_RANGE = 0.1
NOISE_STD = 0.02
BLUR_KERNEL_SIZE = 5
PERSPECTIVE_SCALE = 0.2
ERASING_PROB = 0.5
ERASING_SCALE = (0.02, 0.33)

# Paths
TRAIN_DIR = 'Sheep Classification Images/train'
TEST_DIR = 'Sheep Classification Images/test'
TRAIN_CSV = 'Sheep Classification Images/train_labels.csv'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==================== DATASET CLASS ====================
class SheepDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.is_test:
            img_name = self.df[idx]
            img_path = os.path.join(self.img_dir, img_name)
        else:
            img_name = self.df.iloc[idx]['filename']
            img_path = os.path.join(self.img_dir, img_name)
            label = self.df.iloc[idx]['encoded_label']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image, img_name
        else:
            return image, label

# ==================== AUGMENTATIONS ====================
class AddGaussianNoise:
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=ROTATION_DEGREES),
    transforms.RandomAffine(degrees=0, scale=SCALE_RANGE),
    transforms.RandomPerspective(distortion_scale=PERSPECTIVE_SCALE, p=0.3),
    transforms.ColorJitter(
        brightness=BRIGHTNESS_RANGE,
        contrast=CONTRAST_RANGE,
        saturation=SATURATION_RANGE,
        hue=HUE_RANGE
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(p=ERASING_PROB, scale=ERASING_SCALE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== MIXUP & CUTMIX ====================
# Settings for class-aware mixing
INTRA_CLASS_MIX_PROB = 0.9  # Probability of mixing within same class
INTER_CLASS_MIX_PROB = 0.1  # Probability of mixing between different classes

def get_class_indices(labels):
    """Get indices for each class in the batch"""
    class_indices = {}
    for idx, label in enumerate(labels):
        label_item = label.item()
        if label_item not in class_indices:
            class_indices[label_item] = []
        class_indices[label_item].append(idx)
    return class_indices

def mixup_data(x, y, alpha=0.2, intra_class_prob=0.7):
    """Apply mixup with preference for intra-class mixing"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    
    # Decide whether to do intra-class or inter-class mixing
    use_intra_class = np.random.rand() < intra_class_prob
    
    if use_intra_class:
        # Get indices for each class
        class_indices = get_class_indices(y)
        index = torch.zeros(batch_size, dtype=torch.long).to(device)
        
        # For each sample, find another sample from the same class
        for i in range(batch_size):
            label = y[i].item()
            same_class_indices = class_indices[label]
            if len(same_class_indices) > 1:
                # Choose from same class (excluding self)
                choices = [idx for idx in same_class_indices if idx != i]
                if choices:
                    index[i] = random.choice(choices)
                else:
                    index[i] = i
            else:
                # If only one sample of this class, keep the same
                index[i] = i
    else:
        # Random mixing between all classes
        index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, intra_class_prob=0.7):
    """Apply cutmix with preference for intra-class mixing"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    
    # Decide whether to do intra-class or inter-class mixing
    use_intra_class = np.random.rand() < intra_class_prob
    
    if use_intra_class:
        # Get indices for each class
        class_indices = get_class_indices(y)
        index = torch.zeros(batch_size, dtype=torch.long).to(device)
        
        # For each sample, find another sample from the same class
        for i in range(batch_size):
            label = y[i].item()
            same_class_indices = class_indices[label]
            if len(same_class_indices) > 1:
                # Choose from same class (excluding self)
                choices = [idx for idx in same_class_indices if idx != i]
                if choices:
                    index[i] = random.choice(choices)
                else:
                    index[i] = i
            else:
                # If only one sample of this class, use random
                index[i] = torch.randint(0, batch_size, (1,)).item()
    else:
        # Random mixing between all classes
        index = torch.randperm(batch_size).to(device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# ==================== MODEL ====================
def create_model(num_classes):
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    return model.to(device)

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, dataloader, criterion, optimizer, scaler, use_mixup=True, use_cutmix=True):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup or cutmix
        r = np.random.rand(1)
        if use_mixup and r < MIXUP_PROB:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA, INTRA_CLASS_MIX_PROB)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        elif use_cutmix and r < (MIXUP_PROB + CUTMIX_PROB):
            images, labels_a, labels_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA, INTRA_CLASS_MIX_PROB)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else:
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        
        # For metrics, use original labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1

def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1, np.array(all_probs), np.array(all_labels)

# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ==================== MAIN TRAINING ====================
# Load data
train_df = pd.read_csv(TRAIN_CSV)
print(f'Train data shape: {train_df.shape}')
print(f'Classes: {train_df["label"].unique()}')

# Encode labels
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])

#########################################
#########################################
#########################################
### To change between the 3 variants: ###
#########################################
#########################################
#########################################

# 1st variant: No Psuedo Labels, No class weights
# pass

# 2nd variant: Add class_weights from previous sub (uncomment to use):
# # 1. Load sub
# pseudo_df = pd.read_csv('submission_pseudo-Copy2.csv')
# pseudo_df['encoded_label'] = label_encoder.transform(pseudo_df['label'])
# combined_labels = np.concatenate([
#     pseudo_df['encoded_label'].values
# ])
# # 2. Compute class weights
# from sklearn.utils.class_weight import compute_class_weight
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(combined_labels),
#     y=combined_labels
# )
# class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

# Load test filenames
test_files = sorted(os.listdir(TEST_DIR))
print(f'Number of test images: {len(test_files)}')

# Initialize K-Fold (not stratified)
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Store models and OOF predictions
fold_models = []
oof_predictions = np.zeros((len(train_df), NUM_CLASSES))
oof_labels = np.zeros(len(train_df))

# Training loop
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f'\n{"="*50}')
    print(f'Fold {fold + 1}/{NUM_FOLDS}')
    print(f'{"="*50}')
    
    # Split data
    train_data = train_df.iloc[train_idx].reset_index(drop=True)
    val_data = train_df.iloc[val_idx].reset_index(drop=True)
    
    # Create datasets and dataloaders
    train_dataset = SheepDataset(train_data, TRAIN_DIR, transform=train_transform)
    val_dataset = SheepDataset(val_data, TRAIN_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Initialize model
    model = create_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_START)
    scaler = torch.cuda.amp.GradScaler()
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS, 
        eta_min=LR_END
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, scaler)
        
        # Validate
        val_loss, val_f1, val_probs, val_true_labels = validate_epoch(model, val_loader, criterion)
        
        # Scheduler step
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            }, f'best_model_fold_{fold}.pth')
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print('Early stopping triggered')
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    fold_models.append(model)
    
    # Get OOF predictions
    _, _, best_val_probs, best_val_labels = validate_epoch(model, val_loader, criterion)
    oof_predictions[val_idx] = best_val_probs
    oof_labels[val_idx] = best_val_labels
    
    print(f'\nFold {fold + 1} Best Val F1: {best_val_f1:.4f}')

# ==================== OOF EVALUATION ====================
oof_preds = np.argmax(oof_predictions, axis=1)
oof_f1 = f1_score(oof_labels, oof_preds, average='macro')
print(f'\nOverall OOF F1 Score: {oof_f1:.4f}')
print('\nOOF Classification Report:')
print(classification_report(oof_labels, oof_preds, target_names=label_encoder.classes_))

# Save OOF predictions
oof_df = pd.DataFrame({
    'filename': train_df['filename'].values,
    'true_label': label_encoder.inverse_transform(oof_labels.astype(int)),
    'pred_label': label_encoder.inverse_transform(oof_preds),
})
for i, class_name in enumerate(label_encoder.classes_):
    oof_df[f'prob_{class_name}'] = oof_predictions[:, i]
oof_df.to_csv('oof_predictions.csv', index=False)


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import random
import timm
import warnings
from sklearn.utils.class_weight import compute_class_weight
warnings.filterwarnings('ignore')

# ==================== SETTINGS ====================
# Model settings
MODEL_NAME = 'convnext_xlarge_384_in22ft1k'
IMG_SIZE = 384
NUM_CLASSES = 7

# Training settings
BATCH_SIZE = 16
EPOCHS = 10
NUM_FOLDS = 5
NUM_WORKERS = 2

# Pseudo labeling settings
CONFIDENCE_THRESHOLD = 0.8  # Only use predictions with confidence > this
USE_PSEUDO_LABELS = True
PSEUDO_LABEL_WEIGHT = 0.5  # Weight for pseudo label loss vs real label loss

# Learning rate settings
LR_START = 1e-4
LR_END = 1e-6
WARMUP_EPOCHS = 0

# Early stopping
PATIENCE = 999999999

# Augmentation settings
MIXUP_ALPHA = 0.25
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5
CUTMIX_PROB = 0.5

# Random augmentation parameters
ROTATION_DEGREES = 45
SCALE_RANGE = (0.8, 1.2)
BRIGHTNESS_RANGE = 0.3
CONTRAST_RANGE = 0.3
SATURATION_RANGE = 0.3
HUE_RANGE = 0.1
NOISE_STD = 0.02
BLUR_KERNEL_SIZE = 5
PERSPECTIVE_SCALE = 0.2
ERASING_PROB = 0.5
ERASING_SCALE = (0.02, 0.33)

# Paths
TRAIN_DIR = 'Sheep Classification Images/train'
TEST_DIR = 'Sheep Classification Images/test'
TRAIN_CSV = 'Sheep Classification Images/train_labels.csv'
OOF_PREDICTIONS_CSV = 'oof_predictions.csv'
SUBMISSION_CSV = 'submission.csv'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==================== DATASET CLASS ====================
class SheepDatasetWithPseudo(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False, is_pseudo=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.is_pseudo = is_pseudo
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.is_test:
            img_name = self.df[idx]
            img_path = os.path.join(self.img_dir, img_name)
        else:
            img_name = self.df.iloc[idx]['filename']
            img_path = os.path.join(self.img_dir, img_name)
            label = self.df.iloc[idx]['encoded_label']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image, img_name
        else:
            # Return additional flag for pseudo labels
            return image, label, self.is_pseudo

# ==================== AUGMENTATIONS ====================
class AddGaussianNoise:
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=ROTATION_DEGREES),
    transforms.RandomAffine(degrees=0, scale=SCALE_RANGE),
    transforms.RandomPerspective(distortion_scale=PERSPECTIVE_SCALE, p=0.3),
    transforms.ColorJitter(
        brightness=BRIGHTNESS_RANGE,
        contrast=CONTRAST_RANGE,
        saturation=SATURATION_RANGE,
        hue=HUE_RANGE
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(p=ERASING_PROB, scale=ERASING_SCALE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== MIXUP & CUTMIX ====================
INTRA_CLASS_MIX_PROB = 0.9
INTER_CLASS_MIX_PROB = 0.1

def get_class_indices(labels):
    """Get indices for each class in the batch"""
    class_indices = {}
    for idx, label in enumerate(labels):
        label_item = label.item()
        if label_item not in class_indices:
            class_indices[label_item] = []
        class_indices[label_item].append(idx)
    return class_indices

def mixup_data(x, y, alpha=0.2, intra_class_prob=0.7):
    """Apply mixup with preference for intra-class mixing"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    
    use_intra_class = np.random.rand() < intra_class_prob
    
    if use_intra_class:
        class_indices = get_class_indices(y)
        index = torch.zeros(batch_size, dtype=torch.long).to(device)
        
        for i in range(batch_size):
            label = y[i].item()
            same_class_indices = class_indices[label]
            if len(same_class_indices) > 1:
                choices = [idx for idx in same_class_indices if idx != i]
                if choices:
                    index[i] = random.choice(choices)
                else:
                    index[i] = i
            else:
                index[i] = i
    else:
        index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, intra_class_prob=0.7):
    """Apply cutmix with preference for intra-class mixing"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    
    use_intra_class = np.random.rand() < intra_class_prob
    
    if use_intra_class:
        class_indices = get_class_indices(y)
        index = torch.zeros(batch_size, dtype=torch.long).to(device)
        
        for i in range(batch_size):
            label = y[i].item()
            same_class_indices = class_indices[label]
            if len(same_class_indices) > 1:
                choices = [idx for idx in same_class_indices if idx != i]
                if choices:
                    index[i] = random.choice(choices)
                else:
                    index[i] = i
            else:
                index[i] = torch.randint(0, batch_size, (1,)).item()
    else:
        index = torch.randperm(batch_size).to(device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# ==================== PSEUDO LABEL FUNCTIONS ====================
def get_confident_predictions(df, label_encoder, confidence_threshold=0.8):
    """Extract confident predictions for pseudo labeling"""
    # Get probability columns
    prob_columns = [col for col in df.columns if col.startswith('prob_')]
    
    # Calculate max probability (confidence) for each sample
    probs = df[prob_columns].values
    max_probs = np.max(probs, axis=1)
    
    # Filter confident predictions
    confident_mask = max_probs >= confidence_threshold
    confident_df = df[confident_mask].copy()
    
    print(f'Found {len(confident_df)} confident predictions out of {len(df)} total ({len(confident_df)/len(df)*100:.1f}%)')
    print(f'Confidence threshold: {confidence_threshold}')
    
    return confident_df

def create_pseudo_labeled_data(train_data, oof_df, submission_df, val_idx, label_encoder):
    """Create combined dataset with real labels + pseudo labels"""
    
    # 1. Get confident OOF predictions for validation fold
    val_filenames = set(train_data.iloc[val_idx]['filename'].values)
    oof_val = oof_df[oof_df['filename'].isin(val_filenames)].copy()
    confident_oof = get_confident_predictions(oof_val, label_encoder, CONFIDENCE_THRESHOLD)
    
    print(f'Confident OOF predictions for val fold: {len(confident_oof)}')
    
    # 2. Get confident test predictions
    confident_test = get_confident_predictions(submission_df, label_encoder, CONFIDENCE_THRESHOLD)
    print(f'Confident test predictions: {len(confident_test)}')
    
    # 3. Create pseudo label datasets
    pseudo_datasets = []
    
    if len(confident_oof) > 0:
        # OOF pseudo labels (validation fold images)
        oof_pseudo = pd.DataFrame({
            'filename': confident_oof['filename'],
            'label': confident_oof['pred_label'],
            'encoded_label': label_encoder.transform(confident_oof['pred_label'])
        })
        pseudo_datasets.append(('oof', oof_pseudo))
    
    if len(confident_test) > 0:
        # Test pseudo labels
        test_pseudo = pd.DataFrame({
            'filename': confident_test['filename'],
            'label': confident_test['label'],
            'encoded_label': label_encoder.transform(confident_test['label'])
        })
        pseudo_datasets.append(('test', test_pseudo))
    
    return pseudo_datasets

# ==================== MODEL ====================
def create_model(num_classes):
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    return model.to(device)

# ==================== TRAINING FUNCTIONS ====================
def train_epoch_with_pseudo(model, real_loader, pseudo_loaders, criterion, optimizer, scaler, use_mixup=True, use_cutmix=True):
    model.train()
    running_loss = 0.0
    running_real_loss = 0.0
    running_pseudo_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Create iterators for pseudo loaders
    pseudo_iterators = [(name, iter(loader)) for name, loader in pseudo_loaders]
    
    for images, labels, is_pseudo_flags in tqdm(real_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        is_pseudo_flags = is_pseudo_flags.numpy()
        
        # Separate real and pseudo samples in the batch
        real_mask = ~is_pseudo_flags
        pseudo_mask = is_pseudo_flags
        
        total_loss = 0
        batch_count = 0
        
        # Process real samples
        if np.any(real_mask):
            real_images = images[real_mask]
            real_labels = labels[real_mask]
            
            # Apply augmentations to real samples
            r = np.random.rand(1)
            if use_mixup and r < MIXUP_PROB:
                real_images, labels_a, labels_b, lam = mixup_data(real_images, real_labels, MIXUP_ALPHA, INTRA_CLASS_MIX_PROB)
                
                with torch.cuda.amp.autocast():
                    outputs = model(real_images)
                    real_loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    
            elif use_cutmix and r < (MIXUP_PROB + CUTMIX_PROB):
                real_images, labels_a, labels_b, lam = cutmix_data(real_images, real_labels, CUTMIX_ALPHA, INTRA_CLASS_MIX_PROB)
                
                with torch.cuda.amp.autocast():
                    outputs = model(real_images)
                    real_loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    
            else:
                with torch.cuda.amp.autocast():
                    outputs = model(real_images)
                    real_loss = criterion(outputs, real_labels)
            
            total_loss += real_loss
            running_real_loss += real_loss.item()
            batch_count += 1
            
            # For metrics
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(real_labels.cpu().numpy())
        
        # Add pseudo label samples
        pseudo_loss_total = 0
        pseudo_count = 0
        
        for name, pseudo_iter in pseudo_iterators:
            try:
                pseudo_images, pseudo_labels, _ = next(pseudo_iter)
                pseudo_images, pseudo_labels = pseudo_images.to(device), pseudo_labels.to(device)
                
                with torch.cuda.amp.autocast():
                    pseudo_outputs = model(pseudo_images)
                    pseudo_loss = criterion(pseudo_outputs, pseudo_labels)
                
                pseudo_loss_total += pseudo_loss * PSEUDO_LABEL_WEIGHT
                running_pseudo_loss += pseudo_loss.item()
                pseudo_count += 1
                
            except StopIteration:
                # Restart iterator if exhausted
                pseudo_iterators = [(n, iter(l)) for n, l in pseudo_loaders if n == name] + \
                                 [(n, it) for n, it in pseudo_iterators if n != name]
                continue
        
        if pseudo_count > 0:
            total_loss += pseudo_loss_total / pseudo_count
            batch_count += pseudo_count
        
        # Backward pass
        if batch_count > 0:
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += total_loss.item()
    
    epoch_loss = running_loss / len(real_loader)
    epoch_real_loss = running_real_loss / len(real_loader)
    epoch_pseudo_loss = running_pseudo_loss / len(real_loader) if running_pseudo_loss > 0 else 0
    epoch_f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0
    
    return epoch_loss, epoch_real_loss, epoch_pseudo_loss, epoch_f1

def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1, np.array(all_probs), np.array(all_labels)

# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ==================== MAIN TRAINING ====================
# Load data
train_df = pd.read_csv(TRAIN_CSV)
print(f'Train data shape: {train_df.shape}')
print(f'Classes: {train_df["label"].unique()}')

# Load OOF and submission predictions
oof_df = pd.read_csv(OOF_PREDICTIONS_CSV)
submission_df = pd.read_csv(SUBMISSION_CSV)
print(f'OOF predictions shape: {oof_df.shape}')
print(f'Submission predictions shape: {submission_df.shape}')

# Encode labels
label_encoder = LabelEncoder()
train_df['encoded_label'] = label_encoder.fit_transform(train_df['label'])

# Load test filenames
test_files = sorted(os.listdir(TEST_DIR))
print(f'Number of test images: {len(test_files)}')

# Initialize K-Fold
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Store models and OOF predictions
fold_models = []
oof_predictions = np.zeros((len(train_df), NUM_CLASSES))
oof_labels = np.zeros(len(train_df))

# Training loop
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f'\n{"="*60}')
    print(f'Fold {fold + 1}/{NUM_FOLDS} - Training with Pseudo Labels')
    print(f'{"="*60}')
    
    # Split data
    train_data = train_df.iloc[train_idx].reset_index(drop=True)
    val_data = train_df.iloc[val_idx].reset_index(drop=True)
    
    print(f'Real training samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')
    
    # Create pseudo labeled data
    if USE_PSEUDO_LABELS:
        pseudo_datasets = create_pseudo_labeled_data(train_df, oof_df, submission_df, val_idx, label_encoder)
        
        # Create combined training dataset
        combined_train_data = train_data.copy()
        combined_train_data['is_pseudo'] = False
        
        pseudo_loaders = []
        total_pseudo_samples = 0
        
        for name, pseudo_df in pseudo_datasets:
            if len(pseudo_df) > 0:
                pseudo_df['is_pseudo'] = True
                
                if name == 'oof':
                    # OOF pseudo labels use train directory
                    pseudo_dataset = SheepDatasetWithPseudo(pseudo_df, TRAIN_DIR, transform=train_transform, is_pseudo=True)
                else:
                    # Test pseudo labels use test directory  
                    pseudo_dataset = SheepDatasetWithPseudo(pseudo_df, TEST_DIR, transform=train_transform, is_pseudo=True)
                
                pseudo_loader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE//2, shuffle=True, num_workers=NUM_WORKERS)
                pseudo_loaders.append((name, pseudo_loader))
                total_pseudo_samples += len(pseudo_df)
                print(f'Added {len(pseudo_df)} {name} pseudo samples')
        
        print(f'Total pseudo samples: {total_pseudo_samples}')
    else:
        combined_train_data = train_data.copy()
        combined_train_data['is_pseudo'] = False
        pseudo_loaders = []
    
    # Create datasets and dataloaders
    train_dataset = SheepDatasetWithPseudo(combined_train_data, TRAIN_DIR, transform=train_transform)
    val_dataset = SheepDatasetWithPseudo(val_data, TRAIN_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Initialize model
    model = create_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_START)
    scaler = torch.cuda.amp.GradScaler()
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS, 
        eta_min=LR_END
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print(f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Train
        if USE_PSEUDO_LABELS and pseudo_loaders:
            train_loss, real_loss, pseudo_loss, train_f1 = train_epoch_with_pseudo(
                model, train_loader, pseudo_loaders, criterion, optimizer, scaler
            )
            print(f'Train Loss: {train_loss:.4f} (Real: {real_loss:.4f}, Pseudo: {pseudo_loss:.4f}), Train F1: {train_f1:.4f}')
        else:
            # Standard training without pseudo labels
            train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, scaler)
            print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
        
        # Validate
        val_loss, val_f1, val_probs, val_true_labels = validate_epoch(model, val_loader, criterion)
        
        # Scheduler step
        scheduler.step()
        
        print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
                'pseudo_samples': total_pseudo_samples if USE_PSEUDO_LABELS else 0
            }, f'best_model_fold_{fold}_pseudo.pth')
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print('Early stopping triggered')
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    fold_models.append(model)
    
    # Get OOF predictions
    _, _, best_val_probs, best_val_labels = validate_epoch(model, val_loader, criterion)
    oof_predictions[val_idx] = best_val_probs
    oof_labels[val_idx] = best_val_labels
    
    print(f'\nFold {fold + 1} Best Val F1: {best_val_f1:.4f}')
    if USE_PSEUDO_LABELS:
        print(f'Used {total_pseudo_samples} pseudo samples')

# Standard training function for compatibility
def train_epoch(model, dataloader, criterion, optimizer, scaler, use_mixup=True, use_cutmix=True):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc='Training'):
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup or cutmix
        r = np.random.rand(1)
        if use_mixup and r < MIXUP_PROB:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA, INTRA_CLASS_MIX_PROB)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        elif use_cutmix and r < (MIXUP_PROB + CUTMIX_PROB):
            images, labels_a, labels_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA, INTRA_CLASS_MIX_PROB)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else:
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        
        # For metrics, use original labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1

# ==================== OOF EVALUATION ====================
oof_preds = np.argmax(oof_predictions, axis=1)
oof_f1 = f1_score(oof_labels, oof_preds, average='macro')
print(f'\nOverall OOF F1 Score with Pseudo Labels: {oof_f1:.4f}')
print('\nOOF Classification Report:')
print(classification_report(oof_labels, oof_preds, target_names=label_encoder.classes_))

# Save OOF predictions
oof_df_new = pd.DataFrame({
    'filename': train_df['filename'].values,
    'true_label': label_encoder.inverse_transform(oof_labels.astype(int)),
    'pred_label': label_encoder.inverse_transform(oof_preds),
})
for i, class_name in enumerate(label_encoder.classes_):
    oof_df_new[f'prob_{class_name}'] = oof_predictions[:, i]
oof_df_new.to_csv('oof_predictions_pseudo.csv', index=False)


# ==================== TEST INFERENCE ====================
test_dataset = SheepDataset(test_files, TEST_DIR, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

import glob
ckpt_paths = sorted(glob.glob("*.pth"))
fold_models = []
for ckpt in tqdm(ckpt_paths):
    # load checkpoint
    checkpoint = torch.load(ckpt, map_location=device)
    # rebuild model architecture
    model = create_model(NUM_CLASSES)
    # load the saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    fold_models.append(model)

all_predictions = []
print('\nGenerating test predictions...')
for i, model in enumerate(fold_models):
    print(f'Processing fold {i+1}/{NUM_FOLDS}')
    model.eval()
    fold_predictions = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f'Fold {i+1} Test Inference'):
            images = images.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
            
            fold_predictions.append(probabilities.cpu().numpy())
    
    fold_predictions = np.concatenate(fold_predictions, axis=0)
    all_predictions.append(fold_predictions)

# Average predictions
avg_predictions = np.mean(all_predictions, axis=0)
final_predictions = np.argmax(avg_predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(final_predictions)

# ==================== SAVE RESULTS ====================
# Submission
submission_df = pd.DataFrame({
    'filename': test_files,
    'label': predicted_labels
})
submission_df.to_csv('submission_all.csv', index=False)

# Test predictions with probabilities
test_pred_df = pd.DataFrame({'filename': test_files, 'pred_label': predicted_labels})
for i, class_name in enumerate(label_encoder.classes_):
    test_pred_df[f'prob_{class_name}'] = avg_predictions[:, i]
test_pred_df.to_csv('test_predictions_with_probs.csv', index=False)


import pandas as pd
from PIL import Image

# 1. Load your low-confidence flags
test_pred_df = pd.read_csv('test_predictions_with_probs.csv')
prob_cols    = [f'prob_{c}' for c in label_encoder.classes_]
max_probs    = test_pred_df[prob_cols].max(axis=1)
low_conf_mask= max_probs < 0.90
low_conf_files = test_pred_df.loc[low_conf_mask, 'filename'].tolist()

# 2. Build a map for any updated labels
updated_labels = {}

_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=ROTATION_DEGREES),
    transforms.RandomAffine(degrees=0, scale=SCALE_RANGE),
    transforms.RandomPerspective(distortion_scale=PERSPECTIVE_SCALE, p=0.3),
    transforms.ColorJitter(
        brightness=BRIGHTNESS_RANGE,
        contrast=CONTRAST_RANGE,
        saturation=SATURATION_RANGE,
        hue=HUE_RANGE
    ),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=ERASING_PROB, scale=ERASING_SCALE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Loop over each low-confidence image
for fname in tqdm(low_conf_files, desc='TTA thresholding'):
    img = Image.open(os.path.join(TEST_DIR, fname)).convert('RGB')
    for attempt in range(1, 201):  # up to 200 TTA attempts
        # apply one random augmentation
        aug_tensor = _transforms(img).unsqueeze(0).to(device)
        
        # ensemble inference
        with torch.no_grad():
            fold_probs = [torch.softmax(m(aug_tensor), dim=1).cpu().numpy()[0]
                          for m in fold_models]
        avg_probs = np.mean(fold_probs, axis=0)

        if attempt == 200: print("Failed")
        if avg_probs.max() >= 0.9 or attempt == 200:
            # pick the highest-scoring class
            new_label = label_encoder.inverse_transform([avg_probs.argmax()])[0]
            updated_labels[fname] = new_label
            break

# 4. Overwrite only those filenames in your original submission
submission_df = pd.read_csv('submission_all.csv')
submission_df['label'] = submission_df.apply(
    lambda row: updated_labels.get(row['filename'], row['label']),
    axis=1
)

# 5. Save out the new submission
submission_df.to_csv('submission_tta.csv', index=False)
print("Saved TTA-refined submission as submission_tta.csv")



import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Paths
CSV_PATH = 'test_predictions_with_probs.csv'
TEST_DIR = 'Sheep Classification Images/test'

# Load predictions CSV
df = pd.read_csv(CSV_PATH)

# Identify probability columns and compute max prob
prob_cols = [col for col in df.columns if col.startswith('prob_')]
df['max_prob'] = df[prob_cols].max(axis=1)

# Select low-confidence rows (<0.90)
low_conf = df[df['max_prob'] < 0.90][['filename', 'pred_label', 'max_prob']].reset_index(drop=True)

# Indices of the 10 failures from logs
failed = low_conf.iloc[[1, 7, 9, 10, 15, 16, 17, 22, 31, 32]]

# Plot 2×5 grid with filename, predicted class, and max probability
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for ax, (_, row) in zip(axes, failed.iterrows()):
    fname = row['filename']
    cls   = row['pred_label']
    maxp  = row['max_prob']
    img_path = os.path.join(TEST_DIR, fname)
    img = Image.open(img_path).convert('RGB')
    
    ax.imshow(img)
    ax.set_title(f"{fname}\n{cls}: {maxp:.2f}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()



failed


import pandas as pd

# Paths to input/output CSVs
PROB_CSV = 'test_predictions_with_probs.csv'
SUB_CSV  = 'submission_tta.csv'
OUT_CSV  = 'submission_tta_updated.csv'

# Load the probabilities and current submission
df_probs = pd.read_csv(PROB_CSV)
df_sub   = pd.read_csv(SUB_CSV)

# Identify files where top prediction is Roman but Sawakni probability > 20%
mask = (df_probs['pred_label'] == 'Roman') & (df_probs['prob_Sawakni'] > 0.20)
swap_files = df_probs.loc[mask, 'filename']

# Update submission labels
df_sub.loc[df_sub['filename'].isin(swap_files), 'label'] = 'Sawakni'

# Save new submission
df_sub.to_csv(OUT_CSV, index=False)

# Show the first few updated rows and count of swaps
updated = df_sub[df_sub['filename'].isin(swap_files)]
print(f"Total files switched to Sawakni: {len(updated)}\n")
display(updated.head())



import os
import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import timm

# Paths and settings
CSV_PROBS    = 'test_predictions_with_probs.csv'
SUB_IN       = 'submission_tta_updated.csv'
SUB_OUT      = 'submission_tta_updated_2.csv'
TEST_DIR     = 'Sheep Classification Images/test'
CKPT_PATTERN = 'best_model_fold_*_grayscale.pth'  ### --------> Rerun the 1st variant code but with grayscale imgs
MODEL_NAME   = 'convnext_xlarge_384_in22ft1k'
NUM_CLASSES  = 7
IMG_SIZE      = 384

# 1. Load probabilities and current submission
df_probs = pd.read_csv(CSV_PROBS)
df_sub   = pd.read_csv(SUB_IN)

# 2. Merge to filter only those currently labeled 'goat' with Sawakni > 8%
df_merged = df_sub.merge(
    df_probs[['filename', 'prob_Sawakni']],
    on='filename', how='left'
)
files_to_rescore = df_merged.loc[
    (df_merged['label'] == 'Goat') &
    (df_merged['prob_Sawakni'] > 0.08),
    'filename'
].tolist()

# 3. Grayscale transform (1-channel)
gray_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

# 4. Load grayscale checkpoints into 1-channel ConvNeXt models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gray_models = []
for ckpt_path in sorted(glob.glob(CKPT_PATTERN)):
    chk = torch.load(ckpt_path, map_location=device)
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        in_chans=1,
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(chk['model_state_dict'])
    model.to(device).eval()
    gray_models.append(model)

K = 1  

# 5. Reinfer and choose only between goat and sawakni, with K TTA
classes     = list(label_encoder.classes_)
goat_idx    = classes.index('Goat')
sawakni_idx = classes.index('Sawakni')

updates = {}
for fname in tqdm(files_to_rescore, desc='Grayscale Rescore (goat→sawakni?)'):
    img = Image.open(os.path.join(TEST_DIR, fname)).convert('RGB')
    
    # accumulate probs over K augmentations and M models
    accum = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _ in range(K):
        x = gray_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            # per‐model probabilities for this TTA
            tta_probs = [torch.softmax(m(x), dim=1).cpu().numpy()[0] 
                         for m in gray_models]
        accum += np.mean(tta_probs, axis=0)
    
    # average across K runs
    avg_probs = accum / K

    # pick best between goat and sawakni
    goat_p    = avg_probs[goat_idx]
    sawak_p   = avg_probs[sawakni_idx]
    choice    = 'Sawakni' if sawak_p >= 0.2  else 'Goat' 
    updates[fname] = choice

    # print out final probs
    print(f"{fname} → Goat: {goat_p:.3f}, Sawakni: {sawak_p:.3f}  ⇒  {choice}")

# 6. Update submission and save (unchanged)
df_sub['label'] = df_sub.apply(
    lambda r: updates.get(r['filename'], r['label']), axis=1
)
df_sub.to_csv(SUB_OUT, index=False)
print(f"Saved updated submission to {SUB_OUT}  (swapped {len(updates)} entries)")

