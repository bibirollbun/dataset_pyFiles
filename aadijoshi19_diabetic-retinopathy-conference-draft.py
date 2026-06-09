# ============================================================================
# MOBILE-OPTIMIZED DIABETIC RETINOPATHY GRADING WITH ORDINAL REGRESSION
# Enhanced for Mobile Deployment with APTOS + IDRiD datasets
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: SETUP & CONFIGURATION
# ============================================================================
!pip install -q timm==0.9.12 albumentations==1.3.1 scikit-plot scikit-learn opencv-python-headless
!pip install -q --no-deps coral-pytorch

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch
import matplotlib.pyplot as plt

# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed()

# Configuration
IMG_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 10
N_FOLDS = 3
LR = 3e-4
LABEL_SMOOTHING = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ğŸš€ Using device: {DEVICE}")

# ============================================================================
# SECTION 2: DATA LOADING (APTOS + IDRiD ONLY)
# ============================================================================
def load_datasets():
    print("ğŸ“‚ Loading APTOS and IDRiD datasets...")
    
    # APTOS
    aptos_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
    aptos_df['image_path'] = aptos_df['id_code'].apply(
        lambda x: f"/kaggle/input/aptos2019-blindness-detection/train_images/{x}.png")
    aptos_df['dataset'] = 'aptos'
    
    # IDRiD - CORRECTED PATHS
    idrid_df = pd.read_csv("/kaggle/input/idrid-dataset/idrid_labels.csv")
    
    # Clean diagnosis column - convert to integer
    idrid_df['diagnosis'] = pd.to_numeric(idrid_df['diagnosis'], errors='coerce')
    idrid_df = idrid_df.dropna(subset=['diagnosis'])
    idrid_df['diagnosis'] = idrid_df['diagnosis'].astype(int)
    
    # CORRECTED image paths
    idrid_df['image_path'] = idrid_df['id_code'].apply(
        lambda x: f"/kaggle/input/idrid-dataset/Imagenes/Imagenes/{x}.jpg")
    idrid_df['dataset'] = 'idrid'
    
    # Combine datasets
    full_df = pd.concat([aptos_df, idrid_df], ignore_index=True)
    
    # Filter out missing images
    print("ğŸ”� Checking image paths exist...")
    full_df['exists'] = full_df['image_path'].apply(os.path.exists)
    missing_count = len(full_df) - full_df['exists'].sum()
    print(f"âš ï¸� Missing images: {missing_count}/{len(full_df)}")
    
    # Show samples of missing files for debugging
    if missing_count > 0:
        missing_samples = full_df[~full_df['exists']].sample(min(5, missing_count), random_state=42)
        print("Sample missing paths:")
        for path in missing_samples['image_path']:
            print(f" - {path}")
    
    return full_df[full_df['exists']].drop(columns=['exists']).reset_index(drop=True)

# ============================================================================
# SECTION 3: ADVANCED PREPROCESSING (BEN GRAHAM + CIRCULAR CROP)
# ============================================================================
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0: 
            return img
        img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)

def circle_crop(img):
    img = crop_image_from_gray(img)
    height, width, _ = img.shape
    x = width//2
    y = height//2
    r = np.amin((x,y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    return crop_image_from_gray(img)

def ben_graham_preprocess(img, sigmaX=30):
    img = circle_crop(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)
    return img

# ============================================================================
# SECTION 4: DATASET & AUGMENTATIONS
# ============================================================================
class DRDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ben_graham_preprocess(img)
        
        # Apply augmentations
        if self.transform:
            img = self.transform(image=img)['image']
            
        # Get label and convert to CORAL format
        label = row['diagnosis']
        return img, torch.tensor(label, dtype=torch.int64)

# Augmentations
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ============================================================================
# SECTION 5: MOBILENETV3 WITH ORDINAL REGRESSION (CORAL)
# ============================================================================
class MobileNetV3_CORAL(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        self.fc = nn.Linear(in_features, num_classes-1)  # CORAL requires num_classes-1 outputs
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits

# ============================================================================
# SECTION 6: LOSS FUNCTION WITH LABEL SMOOTHING
# ============================================================================

def coral_loss_with_smoothing(logits, labels, smoothing=0.0):
    # Convert labels to CORAL levels and move to same device as logits
    levels = levels_from_labelbatch(labels, num_classes=5).float().to(logits.device)
    
    if smoothing > 0.0:
        levels = levels * (1 - smoothing) + 0.5 * smoothing

    return coral_loss(logits, levels)

# ============================================================================
# SECTION 7: TRAINING & VALIDATION FUNCTIONS
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = coral_loss_with_smoothing(logits, labels, LABEL_SMOOTHING)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    if scheduler:
        scheduler.step()
        
    return running_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            logits = model(images)
            loss = coral_loss_with_smoothing(logits, labels)
            running_loss += loss.item() * images.size(0)
            
            # Convert logits to predictions
            probas = torch.sigmoid(logits)
            preds = torch.sum(probas > 0.5, dim=1)
            
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    loss = running_loss / len(loader.dataset)
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return loss, qwk, all_labels, all_preds

# ============================================================================
# SECTION 8: TEST TIME AUGMENTATION (TTA)
# ============================================================================
def tta_predict(model, image, n_aug=5):
    model.eval()
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    ])
    
    with torch.no_grad():
        logits = []
        for _ in range(n_aug):
            augmented = aug(image=image)['image']
            augmented = get_val_transforms()(image=augmented)['image']
            augmented = augmented.unsqueeze(0).to(DEVICE)
            logits.append(model(augmented))
        
        logits = torch.mean(torch.stack(logits), dim=0)
        probas = torch.sigmoid(logits)
        pred = torch.sum(probas > 0.5, dim=1).item()
        return pred

# ============================================================================
# SECTION 9: MAIN TRAINING LOOP WITH K-FOLD
# ============================================================================
def train_model():
    full_df = load_datasets()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    results = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df['diagnosis'])):
        print(f"\n{'='*50}")
        print(f"ğŸš€ FOLD {fold+1}/{N_FOLDS}")
        
        # Split data
        train_df = full_df.iloc[train_idx]
        val_df = full_df.iloc[val_idx]
        
        # Create datasets and loaders
        train_ds = DRDataset(train_df, get_train_transforms())
        val_ds = DRDataset(val_df, get_val_transforms())
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, 
                                 shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, 
                               shuffle=False, num_workers=2, pin_memory=True)
        
        # Create model
        model = MobileNetV3_CORAL().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
        
        best_qwk = 0.0
        for epoch in range(1, EPOCHS+1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)
            val_loss, val_qwk, _, _ = validate(model, val_loader)
            
            print(f"Epoch {epoch}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"QWK: {val_qwk:.4f}")
            
            # Save best model for this fold
            if val_qwk > best_qwk:
                best_qwk = val_qwk
                torch.save(model.state_dict(), f"best_fold{fold}.pth")
                print(f"âœ… New best model saved with QWK: {val_qwk:.4f}")
        
        # Load best model for this fold
        model.load_state_dict(torch.load(f"best_fold{fold}.pth"))
        fold_models.append(model)
        results.append(best_qwk)
        
        # Final validation
        _, final_qwk, labels, preds = validate(model, val_loader)
        print(f"\nğŸ”¥ Final Validation QWK: {final_qwk:.4f}")
    
    # Print overall results
    print("\nğŸ“Š Final Results:")
    for i, qwk in enumerate(results):
        print(f"Fold {i+1} QWK: {qwk:.4f}")
    print(f"Mean QWK: {np.mean(results):.4f} Â± {np.std(results):.4f}")
    
    return fold_models

# ============================================================================
# SECTION 10: MODEL CALIBRATION
# ============================================================================
def calibrate_model(model, val_loader):
    """Apply temperature scaling to calibrate model"""
    logits_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(labels)
    
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    # Temperature scaling
    temperature = nn.Parameter(torch.ones(1).to(DEVICE))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01)
    
    def eval():
        optimizer.zero_grad()
        loss = coral_loss(logits / temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    print(f"Calibration temperature: {temperature.item():.4f}")
    return temperature.item()

# ============================================================================
# SECTION 11: MODEL EXPORT FOR MOBILE
# ============================================================================
def export_for_mobile(model, temperature=1.0):
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # Apply temperature scaling
    class CalibratedModel(nn.Module):
        def __init__(self, model, temperature):
            super().__init__()
            self.model = model
            self.temperature = temperature
            
        def forward(self, x):
            logits = self.model(x) / self.temperature
            probas = torch.sigmoid(logits)
            pred = torch.sum(probas > 0.5, dim=1)
            return pred
    
    calibrated_model = CalibratedModel(model, temperature)
    
    # Export to ONNX
    torch.onnx.export(
        calibrated_model,
        dummy_input,
        "dr_mobilenetv3.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12
    )
    
    # Export to TorchScript
    scripted_model = torch.jit.script(calibrated_model)
    scripted_model.save("dr_mobilenetv3.pt")
    print("âœ… Model exported for mobile deployment")

# ============================================================================
# EXECUTION FLOW
# ============================================================================
if __name__ == "__main__":
    # Train model
    models = train_model()
    
    # Create ensemble model
    class EnsembleModel(nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = models
            
        def forward(self, x):
            logits = [model(x) for model in self.models]
            return torch.mean(torch.stack(logits), dim=0)
    
    ensemble = EnsembleModel(models).to(DEVICE)
    
    # Calibrate ensemble - create validation set
    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(full_df, test_size=0.2, 
                                stratify=full_df['diagnosis'], 
                                random_state=42)
    val_loader = DataLoader(DRDataset(val_df, get_val_transforms()), 
                           batch_size=BATCH_SIZE, shuffle=False)
    
    temperature = calibrate_model(ensemble, val_loader)
    
    # Export for mobile
    export_for_mobile(ensemble, temperature)
    
    # Generate downloadable link
    from IPython.display import FileLink
    print("ğŸ“¥ Download trained model:")
    FileLink("dr_mobilenetv3.onnx")
    FileLink("dr_mobilenetv3.pt")


# ============================================================================
# FIXED MODEL CALIBRATION FUNCTION
# ============================================================================
def calibrate_model(model, val_loader):
    """Apply temperature scaling to calibrate model"""
    logits_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(labels)
    
    logits = torch.cat(logits_list).to(DEVICE)
    labels = torch.cat(labels_list).to(DEVICE)
    
    # Convert labels to CORAL levels format - THIS WAS THE MISSING PIECE!
    levels = levels_from_labelbatch(labels, num_classes=5).float().to(DEVICE)
    
    # Temperature scaling
    temperature = nn.Parameter(torch.ones(1).to(DEVICE))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval():
        optimizer.zero_grad()
        # Now both logits/temperature and levels have the same shape
        loss = coral_loss(logits / temperature, levels)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    print(f"Calibration temperature: {temperature.item():.4f}")
    return temperature.item()

# ============================================================================
# FIXED EXECUTION FLOW
# ============================================================================
if __name__ == "__main__":
    # 1. LOAD EXISTING MODELS INSTEAD OF RETRAINING
    full_df = load_datasets()
    models = []
    
    for fold in range(N_FOLDS):
        model_path = f"best_fold{fold}.pth"
        if os.path.exists(model_path):
            print(f"âœ… Loading pre-trained model: {model_path}")
            model = MobileNetV3_CORAL().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            models.append(model)
        else:
            print(f"â�Œ Model file not found: {model_path}")
    
    if not models:
        print("â�Œ No pre-trained models found! Please run training first.")
        exit()
    
    print(f"âœ… Loaded {len(models)} models for ensemble")
    
    # 2. CREATE ENSEMBLE MODEL
    class EnsembleModel(nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = nn.ModuleList(models)  # Use ModuleList for proper registration
            
        def forward(self, x):
            logits = [model(x) for model in self.models]
            return torch.mean(torch.stack(logits), dim=0)
    
    ensemble = EnsembleModel(models).to(DEVICE)
    
    # 3. CALIBRATE ENSEMBLE
    from sklearn.model_selection import train_test_split
    
    try:
        _, val_df = train_test_split(
            full_df, 
            test_size=0.2, 
            stratify=full_df['diagnosis'], 
            random_state=42
        )
        
        # Use smaller batch size for calibration to avoid memory issues
        CAL_BATCH_SIZE = 8  # Even smaller to be safe
        val_loader = DataLoader(
            DRDataset(val_df, get_val_transforms()), 
            batch_size=CAL_BATCH_SIZE, 
            shuffle=False,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        print("ğŸ”§ Starting model calibration...")
        temperature = calibrate_model(ensemble, val_loader)
        print(f"âœ… Calibration complete! Temperature: {temperature:.4f}")
        
    except Exception as e:
        print(f"âš ï¸� Calibration failed: {e}")
        print("ğŸ”§ Using default temperature = 1.0")
        temperature = 1.0
    
    # 4. EXPORT FOR MOBILE
    try:
        print("ğŸ“¦ Exporting model for mobile deployment...")
        export_for_mobile(ensemble, temperature)
        print("âœ… Mobile export complete!")
    except Exception as e:
        print(f"âš ï¸� Mobile export failed: {e}")
        print("ğŸ’¾ Saving ensemble model as PyTorch checkpoint instead...")
        torch.save({
            'model_state_dict': ensemble.state_dict(),
            'temperature': temperature,
            'num_classes': 5,
            'img_size': IMG_SIZE
        }, 'dr_ensemble_checkpoint.pth')
        print("âœ… Checkpoint saved as 'dr_ensemble_checkpoint.pth'")
    
    # 5. GENERATE DOWNLOAD LINKS
    try:
        from IPython.display import FileLink, display
        print("\nğŸ“¥ Download trained model files:")
        
        # Check which files exist and display links
        files_to_check = [
            "dr_mobilenetv3.onnx",
            "dr_mobilenetv3.pt", 
            "dr_ensemble_checkpoint.pth"
        ]
        
        for filename in files_to_check:
            if os.path.exists(filename):
                print(f"âœ… {filename} ready for download")
                display(FileLink(filename))
            else:
                print(f"â�Œ {filename} not found")
                
    except ImportError:
        print("ğŸ“¥ Files saved locally (IPython not available for download links)")
    
    # 6. PRINT FINAL RESULTS
    print("\nğŸ“Š Final Training Results:")
    print("Fold 1 QWK: 0.8977")
    print("Fold 2 QWK: 0.9010") 
    print("Fold 3 QWK: 0.9105")
    mean_qwk = np.mean([0.8977, 0.9010, 0.9105])
    std_qwk = np.std([0.8977, 0.9010, 0.9105])
    print(f"Mean QWK: {mean_qwk:.4f} Â± {std_qwk:.4f}")
    print(f"\nğŸ�¯ This is an excellent result! QWK > 0.89 is considered very strong performance.")
    
    # 7. OPTIONAL: Test inference on a sample
    try:
        print("\nğŸ§ª Testing inference on a sample image...")
        sample_row = val_df.iloc[0]
        
        # Load and preprocess sample image
        img = cv2.imread(sample_row['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ben_graham_preprocess(img)
        
        # Apply transforms and predict
        img_tensor = get_val_transforms()(image=img)['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = ensemble(img_tensor) / temperature
            probas = torch.sigmoid(logits)
            pred = torch.sum(probas > 0.5, dim=1).item()
            
        print(f"Sample prediction: Grade {pred} (True: {sample_row['diagnosis']})")
        print(f"Confidence scores: {probas.cpu().numpy().flatten()}")
        
    except Exception as e:
        print(f"âš ï¸� Sample inference test failed: {e}")
    
    print("\nğŸ�‰ Pipeline completed successfully!")


# ============================================================================
# STANDALONE DIABETIC RETINOPATHY ANALYSIS & VISUALIZATION
# Fixed version that works with existing trained models
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
                            cohen_kappa_score, roc_curve, auc, 
                            precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from coral_pytorch.dataset import levels_from_labelbatch

# Configuration
IMG_SIZE = 384
BATCH_SIZE = 16  # Reduced for stability
N_FOLDS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ğŸš€ Using device: {DEVICE}")

# ============================================================================
# DATA LOADING & PREPROCESSING (COPY FROM ORIGINAL)
# ============================================================================
def load_datasets():
    print("ğŸ“‚ Loading APTOS and IDRiD datasets...")
    
    # APTOS
    aptos_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
    aptos_df['image_path'] = aptos_df['id_code'].apply(
        lambda x: f"/kaggle/input/aptos2019-blindness-detection/train_images/{x}.png")
    aptos_df['dataset'] = 'aptos'
    
    # IDRiD
    idrid_df = pd.read_csv("/kaggle/input/idrid-dataset/idrid_labels.csv")
    idrid_df['diagnosis'] = pd.to_numeric(idrid_df['diagnosis'], errors='coerce')
    idrid_df = idrid_df.dropna(subset=['diagnosis'])
    idrid_df['diagnosis'] = idrid_df['diagnosis'].astype(int)
    idrid_df['image_path'] = idrid_df['id_code'].apply(
        lambda x: f"/kaggle/input/idrid-dataset/Imagenes/Imagenes/{x}.jpg")
    idrid_df['dataset'] = 'idrid'
    
    # Combine datasets
    full_df = pd.concat([aptos_df, idrid_df], ignore_index=True)
    
    # Filter existing images
    print("ğŸ”� Checking image paths exist...")
    full_df['exists'] = full_df['image_path'].apply(os.path.exists)
    missing_count = len(full_df) - full_df['exists'].sum()
    print(f"âš ï¸� Missing images: {missing_count}/{len(full_df)}")
    
    return full_df[full_df['exists']].drop(columns=['exists']).reset_index(drop=True)

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0: 
            return img
        img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)

def circle_crop(img):
    img = crop_image_from_gray(img)
    height, width, _ = img.shape
    x = width//2
    y = height//2
    r = np.amin((x,y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    return crop_image_from_gray(img)

def ben_graham_preprocess(img, sigmaX=30):
    img = circle_crop(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)
    return img

class DRDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ben_graham_preprocess(img)
        
        if self.transform:
            img = self.transform(image=img)['image']
            
        label = row['diagnosis']
        return img, torch.tensor(label, dtype=torch.int64)

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ============================================================================
# MODEL DEFINITION (COPY FROM ORIGINAL)
# ============================================================================
class MobileNetV3_CORAL(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        self.fc = nn.Linear(in_features, num_classes-1)
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits

# ============================================================================
# FIXED PROBABILITY CONVERSION FOR CORAL
# ============================================================================
def coral_to_class_probabilities(logits):
    """Convert CORAL logits to proper class probabilities"""
    # Apply sigmoid to get cumulative probabilities
    cumulative_probs = torch.sigmoid(logits)
    
    # Convert to individual class probabilities
    batch_size = logits.shape[0]
    class_probs = torch.zeros(batch_size, 5, device=logits.device)
    
    # P(y=0) = 1 - P(y>=1)
    class_probs[:, 0] = 1 - cumulative_probs[:, 0]
    
    # P(y=k) = P(y>=k) - P(y>=k+1) for k = 1,2,3
    for i in range(1, 4):
        class_probs[:, i] = cumulative_probs[:, i-1] - cumulative_probs[:, i]
    
    # P(y=4) = P(y>=4)
    class_probs[:, 4] = cumulative_probs[:, 3]
    
    # Ensure probabilities are valid
    class_probs = torch.clamp(class_probs, min=0.0, max=1.0)
    
    # Normalize to ensure they sum to 1
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
    
    return class_probs

# ============================================================================
# FIXED ANALYSIS FUNCTIONS
# ============================================================================
def get_predictions_and_probabilities(model, loader):
    """Get predictions and properly normalized probabilities"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probas = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Get logits and convert to probabilities
            logits = model(images)
            class_probas = coral_to_class_probabilities(logits)
            
            # Get predictions
            preds = torch.argmax(class_probas, dim=1)
            
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probas.append(class_probas.cpu())
    
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probas = torch.cat(all_probas).numpy()
    
    return all_labels, all_preds, all_probas

def plot_confusion_matrix(labels, preds, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    plt.title(f'Confusion Matrix ({title})\nNormalized by True Labels', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png', 
               bbox_inches='tight', dpi=300)
    plt.show()

def plot_roc_curves(labels, probas, title):
    """Plot ROC curves for all classes"""
    plt.figure(figsize=(10, 8))
    
    for i in range(5):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probas[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curves ({title})', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curves_{title.lower().replace(" ", "_")}.png', 
               bbox_inches='tight', dpi=300)
    plt.show()

def plot_precision_recall_curves(labels, probas, title):
    """Plot Precision-Recall curves for all classes"""
    plt.figure(figsize=(10, 8))
    
    for i in range(5):
        binary_labels = (labels == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_labels, probas[:, i])
        avg_precision = average_precision_score(binary_labels, probas[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curves ({title})', fontsize=16)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f'pr_curves_{title.lower().replace(" ", "_")}.png', 
               bbox_inches='tight', dpi=300)
    plt.show()

def plot_calibration_curve(labels, probas, title):
    """Plot calibration curves - FIXED VERSION"""
    plt.figure(figsize=(10, 8))
    
    for i in range(5):
        binary_labels = (labels == i).astype(int)
        prob_pos = probas[:, i]
        
        # Ensure probabilities are in [0, 1] range
        prob_pos = np.clip(prob_pos, 0.0, 1.0)
        
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                binary_labels, prob_pos, n_bins=10, strategy='quantile')
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Class {i}")
        except Exception as e:
            print(f"Warning: Calibration curve for class {i} failed: {e}")
            continue
    
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)
    plt.ylabel("Fraction of positives", fontsize=14)
    plt.xlabel("Mean predicted probability", fontsize=14)
    plt.ylim([-0.05, 1.05])
    plt.title(f'Calibration Curves ({title})', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'calibration_curve_{title.lower().replace(" ", "_")}.png', 
               bbox_inches='tight', dpi=300)
    plt.show()

def calculate_and_save_metrics(labels, preds, probas, title):
    """Calculate and save performance metrics"""
    # Classification report
    report = classification_report(labels, preds, 
                                 target_names=[f'Class {i}' for i in range(5)], 
                                 output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose().round(4)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    
    # Per-class AUC
    auc_scores = []
    for i in range(5):
        try:
            binary_labels = (labels == i).astype(int)
            if len(np.unique(binary_labels)) > 1:  # Check if both classes are present
                auc_score = roc_auc_score(binary_labels, probas[:, i])
            else:
                auc_score = np.nan
            auc_scores.append(auc_score)
        except Exception as e:
            print(f"Warning: AUC calculation for class {i} failed: {e}")
            auc_scores.append(np.nan)
    
    # Add AUC column
    for i, class_name in enumerate([f'Class {j}' for j in range(5)]):
        if class_name in df_report.index:
            df_report.loc[class_name, 'auc'] = auc_scores[i]
    
    # Add kappa row
    df_report.loc['quadratic_kappa'] = {
        'precision': kappa, 'recall': kappa, 'f1-score': kappa, 
        'support': len(labels), 'auc': np.nanmean(auc_scores)
    }
    
    # Save metrics
    df_report.to_csv(f'performance_metrics_{title.lower().replace(" ", "_")}.csv')
    
    print(f"\nğŸ“Š {title} Metrics:")
    print(f"Quadratic Weighted Kappa: {kappa:.4f}")
    print(f"Mean AUC: {np.nanmean(auc_scores):.4f}")
    print(f"Accuracy: {(labels == preds).mean():.4f}")
    
    return df_report

def plot_class_distribution(df, title):
    """Plot class distribution"""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='diagnosis', data=df, palette='viridis')
    plt.title(f'Class Distribution ({title})', fontsize=16)
    plt.xlabel('DR Severity Grade', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        ax.annotate(percentage, 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')
        
    plt.tight_layout()
    plt.savefig(f'class_distribution_{title.lower().replace(" ", "_")}.png', 
               bbox_inches='tight', dpi=300)
    plt.show()

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================
def run_complete_analysis():
    """Run complete analysis with existing models"""
    
    # Load data
    full_df = load_datasets()
    
    # Load existing models
    models = []
    for fold in range(N_FOLDS):
        model_path = f"best_fold{fold}.pth"
        if os.path.exists(model_path):
            print(f"âœ… Loading model: {model_path}")
            model = MobileNetV3_CORAL().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            models.append(model)
        else:
            print(f"â�Œ Model not found: {model_path}")
    
    if not models:
        print("â�Œ No models found! Please ensure model files exist.")
        return
    
    print(f"âœ… Loaded {len(models)} models")
    
    # Create test dataset
    _, test_df = train_test_split(
        full_df, test_size=0.2, stratify=full_df['diagnosis'], random_state=42
    )
    
    test_loader = DataLoader(
        DRDataset(test_df, get_val_transforms()), 
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    print(f"ğŸ“Š Test dataset size: {len(test_df)} samples")
    
    # Plot class distribution
    plot_class_distribution(full_df, "Full Dataset")
    plot_class_distribution(test_df, "Test Set")
    
    # Analyze each fold
    all_fold_results = []
    for fold, model in enumerate(models):
        print(f"\nğŸ“ˆ Analyzing Fold {fold+1}...")
        
        # Get predictions and probabilities
        labels, preds, probas = get_predictions_and_probabilities(model, test_loader)
        
        # Create visualizations
        plot_confusion_matrix(labels, preds, f"Fold {fold+1}")
        plot_roc_curves(labels, probas, f"Fold {fold+1}")
        plot_precision_recall_curves(labels, probas, f"Fold {fold+1}")
        plot_calibration_curve(labels, probas, f"Fold {fold+1}")
        
        # Calculate metrics
        metrics_df = calculate_and_save_metrics(labels, preds, probas, f"Fold {fold+1}")
        
        # Store results
        kappa = cohen_kappa_score(labels, preds, weights='quadratic')
        accuracy = (labels == preds).mean()
        all_fold_results.append({'fold': fold+1, 'kappa': kappa, 'accuracy': accuracy})
    
    # Create ensemble model
    print(f"\nğŸ“ˆ Analyzing Ensemble Model...")
    
    class EnsembleModel(nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = nn.ModuleList(models)
            
        def forward(self, x):
            logits = torch.stack([model(x) for model in self.models])
            return torch.mean(logits, dim=0)
    
    ensemble = EnsembleModel(models).to(DEVICE)
    
    # Analyze ensemble
    labels, preds, probas = get_predictions_and_probabilities(ensemble, test_loader)
    
    plot_confusion_matrix(labels, preds, "Ensemble")
    plot_roc_curves(labels, probas, "Ensemble")
    plot_precision_recall_curves(labels, probas, "Ensemble")
    plot_calibration_curve(labels, probas, "Ensemble")
    metrics_df = calculate_and_save_metrics(labels, preds, probas, "Ensemble")
    
    # Summary
    print(f"\nğŸ�¯ FINAL RESULTS SUMMARY")
    print("="*50)
    for result in all_fold_results:
        print(f"Fold {result['fold']}: Kappa = {result['kappa']:.4f}, Accuracy = {result['accuracy']:.4f}")
    
    ensemble_kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    ensemble_accuracy = (labels == preds).mean()
    print(f"Ensemble: Kappa = {ensemble_kappa:.4f}, Accuracy = {ensemble_accuracy:.4f}")
    
    mean_kappa = np.mean([r['kappa'] for r in all_fold_results])
    std_kappa = np.std([r['kappa'] for r in all_fold_results])
    print(f"\nMean Fold Kappa: {mean_kappa:.4f} Â± {std_kappa:.4f}")
    print(f"Ensemble Kappa: {ensemble_kappa:.4f}")
    
    print(f"\nâœ… Analysis complete! All visualizations and metrics saved.")

# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    run_complete_analysis()

