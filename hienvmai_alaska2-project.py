!pip install --upgrade pip
!pip install -qU timm albumentations


import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from tqdm.notebook import tqdm
import timm
import glob

# for reproducibility
import warnings
warnings.filterwarnings('ignore')


class CFG:
    # General
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    data_path = '/kaggle/input/alaska2-image-steganalysis/'

    # Take a subset of the original dataset
    n_stego_samples_per_type = 20000
    
    # To create a 1:1 balance, we need an equal number of cover images
    n_cover_samples = 3 * n_stego_samples_per_type
    
    image_size = 512
    
    # Model
    model_name = 'tf_efficientnet_b1_ns' 
    num_classes = 1
    
    # Training
    n_folds = 5
    fold_to_train = 0
    epochs = 1
    train_batch_size = 16
    valid_batch_size = 32
    
    # Optimizer & Scheduler
    lr = 1e-4
    weight_decay = 1e-6
    T_0 = 5 
    eta_min = 1e-6
    
    # Checkpointing
    checkpoint_save_path = '/kaggle/working/latest_checkpoint.pth'
    best_model_save_path = f'/kaggle/working/best_model_fold_{fold_to_train}.pth'
    checkpoint_load_path = glob.glob('/kaggle/input/*/latest_checkpoint.pth')
    if len(checkpoint_load_path) > 0:
        checkpoint_load_path = checkpoint_load_path[0]
    else:
        checkpoint_load_path = None


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(CFG.seed)


def alaska_weighted_auc(y_true, y_pred):
    """
    Calculates the weighted AUC score for the ALASKA2 competition.
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    if len(fpr) < 2: return 0.5
    
    areas = np.array([0.0] * len(weights))
    for i, lower in enumerate(tpr_thresholds[:-1]):
        upper = tpr_thresholds[i+1]
        mask = (tpr >= lower) & (tpr < upper)
        if np.any(mask):
            mask_indices = np.where(mask)[0]
            start_idx, end_idx = mask_indices[0], mask_indices[-1]
            tpr_slice = np.concatenate([[lower], tpr[start_idx:end_idx+1], [upper]])
            fpr_slice = np.concatenate([[np.interp(lower, tpr, fpr)], fpr[start_idx:end_idx+1], [np.interp(upper, tpr, fpr)]])
            tpr_slice, unique_indices = np.unique(tpr_slice, return_index=True)
            fpr_slice = fpr_slice[unique_indices]
            areas[i] = np.trapz(fpr_slice, tpr_slice)
            
    return np.sum(areas * weights) / np.sum(weights)


all_files = []

# --- Sample Cover Images ---
cover_folder = os.path.join(CFG.data_path, 'Cover')
cover_files = [os.path.join(cover_folder, f) for f in os.listdir(cover_folder)]
random.shuffle(cover_files)
all_files.extend(cover_files[:CFG.n_cover_samples])
print(f"Sampled {len(cover_files[:CFG.n_cover_samples])} images from Cover.")

# --- Sample Stego Images ---
stego_folders = ['JMiPOD', 'JUNIWARD', 'UERD']
print(f"Sampling {CFG.n_stego_samples_per_type} images from each stego type...")
for folder in stego_folders:
    folder_path = os.path.join(CFG.data_path, folder)
    files_in_folder = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    random.shuffle(files_in_folder)
    all_files.extend(files_in_folder[:CFG.n_stego_samples_per_type])
    print(f"  - Took {len(files_in_folder[:CFG.n_stego_samples_per_type])} images from {folder}")

# --- Create DataFrame and Folds ---
df = pd.DataFrame({'image_path': all_files})
df['label'] = df['image_path'].apply(lambda x: 0 if 'Cover' in x else 1)
df['image_id'] = df['image_path'].apply(os.path.basename)

# Now the dataset is balanced 1:1.
# StratifiedKFold will ensure this ratio is maintained in train/valid splits.
skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    df.loc[val_idx, 'fold'] = fold

print("\nBalanced subset dataset distribution:")
print(df['label'].value_counts())
print("\nFold distribution:")
print(df.groupby('fold')['label'].value_counts())


# Augmentations and Dataset
def get_transforms(data_type='train'):
    if data_type == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=CFG.image_size, width=CFG.image_size, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=CFG.image_size, width=CFG.image_size, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

class AlaskaDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.image_paths = df['image_path'].values
        self.labels = df['label'].values
        self.transforms = transforms
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms: image = self.transforms(image=image)['image']
        return image, label


def train_fn(loader, model, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    return running_loss / len(loader)

def eval_fn(loader, model, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    val_loss = running_loss / len(loader)
    score = alaska_weighted_auc(all_labels, all_preds)
    return val_loss, score


# Training Loop with Checkpointing
def run_training(fold):
    print(f"========== Starting Training for Fold {fold} ==========")
    
    # --- Data Setup ---
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    train_dataset = AlaskaDataset(train_df, transforms=get_transforms('train'))
    valid_dataset = AlaskaDataset(valid_df, transforms=get_transforms('valid'))
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # --- Model, Optimizer, Scheduler Setup ---
    model = timm.create_model(CFG.model_name, pretrained=True, num_classes=CFG.num_classes).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0 * len(train_loader), eta_min=CFG.eta_min)
    criterion = nn.BCEWithLogitsLoss()

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_score = 0.0
    if CFG.checkpoint_load_path and os.path.exists(CFG.checkpoint_load_path):
        print(f"Resuming training from checkpoint: {CFG.checkpoint_load_path}")
        checkpoint = torch.load(CFG.checkpoint_load_path, map_location=CFG.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = 10 # checkpoint['epoch'] + 1 # Start from the next epoch
        best_score = checkpoint['best_score']
        print(f"Loaded model from epoch {start_epoch-1} with best score: {best_score:.4f}")
    else:
        print("No checkpoint found, starting training from scratch.")

    # --- Main Loop ---
    for epoch in range(start_epoch, CFG.epochs):
        print(f"\n--- Epoch {epoch+1}/{CFG.epochs} ---")
        train_loss = train_fn(train_loader, model, criterion, optimizer, scheduler, CFG.device)
        val_loss, val_score = eval_fn(valid_loader, model, criterion, CFG.device)
        
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Valid Weighted AUC: {val_score:.4f}")

        # Save best model based on validation score
        if val_score > best_score:
            print(f"Validation score improved! ({best_score:.4f} -> {val_score:.4f}). Saving best model...")
            best_score = val_score
            torch.save(model.state_dict(), CFG.best_model_save_path)
        
        # Save current state for resuming
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_score': best_score,
        }
        torch.save(checkpoint, CFG.checkpoint_save_path)
        print(f"Epoch {epoch+1} state saved to checkpoint: {CFG.checkpoint_save_path}")

    print(f"\n========== Finished Training for Fold {fold}. Best Score: {best_score:.4f} ==========")

# Start the training process
run_training(CFG.fold_to_train)


def create_submission():
    print("\nStarting inference on the test set...")
    
    test_folder = os.path.join(CFG.data_path, 'Test')
    test_image_ids = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    
    test_df = pd.DataFrame({'image_id': test_image_ids})
    test_df['image_path'] = test_df['image_id'].apply(lambda x: os.path.join(test_folder, x))
    test_df['label'] = 0 # Dummy label
    
    test_dataset = AlaskaDataset(test_df, transforms=get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=2)
    
    # Load the best performing model for inference
    model = timm.create_model(CFG.model_name, pretrained=False, num_classes=CFG.num_classes)
    model.load_state_dict(torch.load(CFG.best_model_save_path))
    model.to(CFG.device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Predicting")
        for images, _ in pbar:
            images = images.to(CFG.device)
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy().flatten())
            
    submission_df = pd.DataFrame({'Id': test_image_ids, 'Label': predictions}).sort_values('Id')
    submission_df.to_csv('submission.csv', index=False)
    
    print("\nSubmission file created successfully!")
    print(submission_df)

create_submission()

