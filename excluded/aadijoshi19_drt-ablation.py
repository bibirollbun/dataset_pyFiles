# ===================================================================
# SECTION 1: SETUP AND CONFIGURATION
# ===================================================================
import warnings
warnings.filterwarnings('ignore')

# Install necessary libraries
!pip install -q timm albumentations scikit-plot scikit-learn opencv-python-headless coral-pytorch

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from coral_pytorch.dataset import levels_from_labelbatch
import matplotlib.pyplot as plt
import time
from IPython.display import FileLink
import seaborn as sns
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed()

# Configuration (Enhanced for Publication Quality)
IMG_SIZE = 384
BATCH_SIZE = 32
NUM_EPOCHS = 10  # Increased for thoroughness
N_FOLDS = 3      # Increased to the gold standard for validation
NUM_CLASSES = 5
LR = 3e-4
LABEL_SMOOTHING = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===================================================================
# SECTION 2: DATASET HANDLING AND VISUALIZATION (FROM YOUR WORKING NOTEBOOK)
# ===================================================================
# This block is restored from your original, working notebook and error logs.
# It correctly handles the column names and paths.
try:
    print("Loading datasets with your original, verified paths...")

    # --- APTOS 2019 paths ---
    aptos_df = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
    aptos_df['image_path'] = aptos_df['id_code'].apply(
        lambda x: os.path.join('/kaggle/input/aptos2019-blindness-detection/train_images', f'{x}.png')
    )
    aptos_df = aptos_df[['image_path', 'diagnosis']]
    aptos_df['dataset'] = 'aptos'

    # --- IDRiD paths (Corrected based on your original notebook and logs) ---
    idrid_df = pd.read_csv('/kaggle/input/idrid-dataset/idrid_labels.csv')
    
    # THE CORRECT LOGIC: The columns are 'id_code' and 'diagnosis' in this specific CSV.
    idrid_df['image_path'] = idrid_df['id_code'].apply(
        lambda x: os.path.join('/kaggle/input/idrid-dataset/Imagenes/Imagenes', f'{x}.jpg')
    )
    # The 'diagnosis' column is already named correctly in this file.
    idrid_df = idrid_df[['image_path', 'diagnosis']]
    idrid_df['dataset'] = 'idrid'

    # Combine and verify paths
    full_df = pd.concat([aptos_df, idrid_df], ignore_index=True)
    full_df['exists'] = full_df['image_path'].apply(os.path.exists)

    print(f"Found {full_df['exists'].sum()} images out of {len(full_df)} total records.")
    if full_df['exists'].sum() < len(full_df):
        print("Warning: Some image paths could not be found.")

except (FileNotFoundError, KeyError) as e:
    print(f"An error occurred while loading data: {e}")
    full_df = pd.DataFrame()


def visualize_dataset_statistics(df):
    if df.empty or 'exists' not in df.columns or df['exists'].sum() == 0:
        print("DataFrame is empty or no images found. Skipping visualization.")
        return
        
    df_existing = df[df['exists']].drop(columns=['exists'])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    class_dist = df_existing['diagnosis'].value_counts().sort_index()
    axes[0].bar(range(5), class_dist.values, color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'])
    axes[0].set_title('Class Distribution Across All Datasets', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Diabetic Retinopathy Grade', fontweight='bold')
    axes[0].set_ylabel('Number of Images', fontweight='bold')
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(['No DR\n(0)', 'Mild\n(1)', 'Moderate\n(2)', 'Severe\n(3)', 'Proliferative\n(4)'])
    
    class_weights_dict = dict(zip(range(5), 1.0 / class_dist.values))
    weights_df = pd.DataFrame(list(class_weights_dict.items()), columns=['Grade', 'Weight'])
    axes[1].bar(weights_df['Grade'], weights_df['Weight'], color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'])
    axes[1].set_title('Class Balancing Weights', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Diabetic Retinopathy Grade', fontweight='bold')
    axes[1].set_ylabel('Class Weight (Inverse Frequency)', fontweight='bold')
    axes[1].set_xticks(range(5))

    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

visualize_dataset_statistics(full_df)

# ===================================================================
# SECTION 3: IMAGE PREPROCESSING AND AUGMENTATION
# ===================================================================
def adaptive_circle_crop(img, tol=7):
    if img is None: return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if not np.any(mask): return cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img[np.ix_(mask.any(1), mask.any(0))]

def ben_graham_preprocess(img):
    img = adaptive_circle_crop(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 30), -4, 128)
    return img

class DRDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_path'])
        if img is None: 
            print(f"Warning: Could not read image {row['image_path']}. Returning blank image.")
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.tensor(0, dtype=torch.long)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ben_graham_preprocess(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, torch.tensor(row['diagnosis'], dtype=torch.long)

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ===================================================================
# SECTION 4: MODEL ARCHITECTURE DEFINITIONS (REVISED AND FIXED)
# ===================================================================
class MobileNetV3_CORAL(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        # 1. Create the backbone as before
        self.backbone = timm.create_model(
            'mobilenetv3_large_100',
            pretrained=pretrained,
            num_classes=0  # Removes the original classifier
        )
        
        # 2. Determine the TRUE feature size with a dummy forward pass.
        # This is the robust way to find the size, which solves the error.
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            # The 'in_features' will now correctly be 1280
            in_features = self.backbone(dummy_input).shape[1]

        self.num_features = in_features
        
        # 3. Create the head layer with the CORRECTLY determined size
        self.head = nn.Linear(self.num_features, num_classes - 1)
        
    def forward(self, x):
        # The forward pass is now simple and correct
        return self.head(self.backbone(x))

class BaselineResNet50(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)


# ===================================================================
# SECTION 5: LOSS FUNCTION AND TRAINING UTILITIES
# ===================================================================
def coral_loss_with_smoothing(logits, labels, smoothing=0.0):
    levels = levels_from_labelbatch(labels, num_classes=NUM_CLASSES).float().to(logits.device)
    if smoothing > 0.0:
        levels = levels * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, levels, reduction='mean')

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images.to(DEVICE))
        loss = coral_loss_with_smoothing(logits, labels.to(DEVICE), LABEL_SMOOTHING)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            preds = torch.sum(torch.sigmoid(logits) > 0.5, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return cohen_kappa_score(all_labels, all_preds, weights='quadratic'), accuracy_score(all_labels, all_preds), all_labels, all_preds
    
def train_and_evaluate_baseline(train_loader, val_loader, model, epochs):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = {'kappa': [], 'accuracy': []}
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
        model.eval()
        val_labels, val_preds = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(DEVICE))
                _, predicted = torch.max(outputs.data, 1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
        kappa, acc = cohen_kappa_score(val_labels, val_preds, weights='quadratic'), accuracy_score(val_labels, val_preds)
        history['kappa'].append(kappa); history['accuracy'].append(acc)
        print(f"Epoch {epoch+1}/{epochs} -> Val Kappa: {kappa:.4f}, Val Accuracy: {acc:.4f}")
    return model, {'Kappa': history['kappa'][-1], 'Accuracy': history['accuracy'][-1]}

# ===================================================================
# SECTION 6: MAIN EXPERIMENT - K-FOLD CROSS-VALIDATION
# ===================================================================
# print("\n" + "="*80 + "\nRUNNING MAIN EXPERIMENT: MobileNetV3 + CORAL with 5-Fold Cross-Validation\n" + "="*80)
# skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
data_df = full_df[full_df['exists']].reset_index(drop=True)
# fold_models, main_model_results_list = [], []

# for fold, (train_idx, val_idx) in enumerate(skf.split(data_df, data_df['diagnosis'])):
#     print(f"\n----- Fold {fold+1}/{N_FOLDS} -----")
#     train_df, val_df = data_df.iloc[train_idx], data_df.iloc[val_idx]
#     class_weights = 1. / train_df['diagnosis'].value_counts().sort_index()
#     sample_weights = train_df['diagnosis'].map(class_weights.to_dict()).values
#     train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
#     train_loader = DataLoader(DRDataset(train_df, get_train_transforms()), batch_size=BATCH_SIZE, sampler=train_sampler)
#     val_loader = DataLoader(DRDataset(val_df, get_val_transforms()), batch_size=BATCH_SIZE, shuffle=False)
#     model = MobileNetV3_CORAL().to(DEVICE)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#     best_kappa = -1
#     for epoch in range(NUM_EPOCHS):
#         train_loss = train_one_epoch(model, train_loader, optimizer)
#         val_kappa, val_acc, _, _ = evaluate(model, val_loader)
#         print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.4f}, Val Kappa: {val_kappa:.4f}")
#         if val_kappa > best_kappa:
#             best_kappa = val_kappa
#             torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
#     model.load_state_dict(torch.load(f'best_model_fold_{fold+1}.pth'))
#     fold_models.append(model)
#     final_kappa, final_acc, _, _ = evaluate(model, val_loader)
#     main_model_results_list.append({'Kappa': final_kappa, 'Accuracy': final_acc})

# ===================================================================
# SECTION 7: BASELINE AND ABLATION STUDIES
# ===================================================================
train_df_single, val_df_single = train_test_split(data_df, test_size=0.2, random_state=42, stratify=data_df['diagnosis'])
baseline_loader_train = DataLoader(DRDataset(train_df_single, get_train_transforms()), batch_size=BATCH_SIZE, shuffle=True)
baseline_loader_val = DataLoader(DRDataset(val_df_single, get_val_transforms()), batch_size=BATCH_SIZE, shuffle=False)

print("\n" + "="*80 + "\nRUNNING BASELINE: ResNet50 with Cross-Entropy Loss\n" + "="*80)
_, baseline_results = train_and_evaluate_baseline(baseline_loader_train, baseline_loader_val, BaselineResNet50(), epochs=NUM_EPOCHS)

print("\n" + "="*80 + "\nRUNNING ABLATION 1: MobileNetV3 without CORAL\n" + "="*80)
ablation_nocoral_model = MobileNetV3_CORAL()
ablation_nocoral_model.head = nn.Linear(ablation_nocoral_model.num_features, NUM_CLASSES)
_, ablation_nocoral_results = train_and_evaluate_baseline(baseline_loader_train, baseline_loader_val, ablation_nocoral_model, epochs=NUM_EPOCHS)

print("\n" + "="*80 + "\nRUNNING ABLATION 2: CORAL Model without Advanced Preprocessing\n" + "="*80)
class DRDatasetSimple(Dataset):
    def __init__(self, df, transform): self.df, self.transform = df, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = cv2.imread(self.df.iloc[idx]['image_path'])
        if img is not None:
            img = self.transform(img)
        else:
            print(f"Warning: Could not read image {self.df.iloc[idx]['image_path']}. Returning blank image.")
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.tensor(0, dtype=torch.long)
        return img, torch.tensor(self.df.iloc[idx]['diagnosis'], dtype=torch.long)

simple_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
nopreproc_loader_train = DataLoader(DRDatasetSimple(train_df_single, simple_transforms), batch_size=BATCH_SIZE, shuffle=True)
nopreproc_loader_val = DataLoader(DRDatasetSimple(val_df_single, simple_transforms), batch_size=BATCH_SIZE, shuffle=False)
model_nopreproc = MobileNetV3_CORAL().to(DEVICE)
optimizer_nopreproc = torch.optim.Adam(model_nopreproc.parameters(), lr=LR)
for epoch in range(NUM_EPOCHS):
    train_one_epoch(model_nopreproc, nopreproc_loader_train, optimizer_nopreproc)
    val_kappa, _, _, _ = evaluate(model_nopreproc, nopreproc_loader_val)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Val Kappa: {val_kappa:.4f}")
ablation_nopreproc_kappa, ablation_nopreproc_accuracy, _, _ = evaluate(model_nopreproc, nopreproc_loader_val)
ablation_nopreproc_results = {'Kappa': ablation_nopreproc_kappa, 'Accuracy': ablation_nopreproc_accuracy}

# ===================================================================
# SECTION 8: RESULTS VISUALIZATION AND ANALYSIS (FOR ABLATION NOTEBOOK)
# ===================================================================
# Note: This section is simplified to only show results from this notebook
try:
    models_names = ['Baseline (ResNet50)', 'Ablation (No CORAL)', 'Ablation (No Preprocessing)']
    kappa_scores = [baseline_results['Kappa'], ablation_nocoral_results['Kappa'], ablation_nopreproc_results['Kappa']]
    accuracy_scores = [baseline_results['Accuracy'], ablation_nocoral_results['Accuracy'], ablation_nopreproc_results['Accuracy']]
    
    x = np.arange(len(models_names))
    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.bar(x - 0.2, kappa_scores, 0.4, label='Quadratic Kappa', color='skyblue')
    rects2 = ax.bar(x + 0.2, accuracy_scores, 0.4, label='Accuracy', color='salmon')
    
    ax.set_title('Performance of Baseline and Ablation Models', fontsize=16, fontweight='bold')
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    plt.savefig('ablation_model_comparison.png', dpi=300)
    plt.show()

except NameError as e:
    print(f"Skipping visualization as some results are not defined: {e}")
    print("This is expected if the ablation runs haven't completed.")

# ===================================================================
# SECTION 9: MODEL CALIBRATION AND EXPORT FOR MOBILE
# ===================================================================
# print("\n" + "="*80 + "\nEXPORTING BEST MODEL FOR MOBILE DEPLOYMENT\n" + "="*80)
# best_model.eval()
# logits_list, labels_list = [], []
# with torch.no_grad():
#     for images, labels in best_val_loader:
#         logits_list.append(best_model(images.to(DEVICE)))
#         labels_list.append(labels)
# logits_all, labels_all = torch.cat(logits_list), torch.cat(labels_list)
# temperature = nn.Parameter(torch.ones(1).to(DEVICE))
# optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
# def calib_eval():
#     optimizer.zero_grad()
#     loss = coral_loss_with_smoothing(logits_all / temperature, labels_all.to(DEVICE))
#     loss.backward()
#     return loss
# optimizer.step(calib_eval)
# final_temp = temperature.item()
# print(f"Optimal calibration temperature found: {final_temp:.4f}")

# class FinalMobileModel(nn.Module):
#     def __init__(self, model, temp): super().__init__(); self.model, self.temperature = model, temp
#     def forward(self, x): return torch.sum(torch.sigmoid(self.model(x) / self.temperature) > 0.5, dim=1)

# exportable_model = FinalMobileModel(best_model.to('cpu'), final_temp)
# exportable_model.eval()
# traced_model = torch.jit.trace(exportable_model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
# from torch.utils.mobile_optimizer import optimize_for_mobile
# optimized_model = optimize_for_mobile(traced_model)
# optimized_model._save_for_lite_interpreter("dr_mobilenetv3_mobile.ptl")
# print("\nModel successfully exported to 'dr_mobilenetv3_mobile.ptl'")
# FileLink("dr_mobilenetv3_mobile.ptl")

