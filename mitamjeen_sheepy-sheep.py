import os
import random
import math
from pathlib import Path
from typing import Any, Dict
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from collections import Counter
from scipy.spatial.distance import cdist
import umap

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    seed = 42
    img_size = 384
    model_name = 'convnext_large.fb_in22k_ft_in1k_384'
    num_classes = 7
    batch_size = 8
    epochs = 10
    lr = 5e-5
    min_lr = 1e-6
    n_folds = 3
    patience = 7
    mixup_cutmix_alpha = 0.5
    
    base_data_path = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/'
    train_csv = os.path.join(base_data_path, 'train_labels.csv')
    train_dir = os.path.join(base_data_path, 'train')
    test_dir = os.path.join(base_data_path, 'test')
    
    models_dir = 'models'
    results_dir = 'results'
    processed_data_dir = 'processed_data'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

class SeedManager:
    def __init__(self, seed: int):
        self.seed = seed

    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seeds set to: {self.seed}")

CONFIG = Config()
SeedManager(CONFIG.seed).seed_everything()

def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(size=(CONFIG.img_size, CONFIG.img_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.3, scale_limit=0.15, rotate_limit=25, border_mode=cv2.BORDER_REFLECT),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        A.CoarseDropout(max_holes=1, max_height=int(CONFIG.img_size*0.4), max_width=int(CONFIG.img_size*0.4), min_holes=1, min_height=int(CONFIG.img_size*0.1), min_width=int(CONFIG.img_size*0.1), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_valid_transforms():
    return A.Compose([
        A.Resize(CONFIG.img_size, CONFIG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class SheepDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False, source_map=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self.source_map = source_map if source_map and not is_test else {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]

        if self.is_test:
            img_dir = CONFIG.test_dir
        else:
            source = self.source_map.get(filename, 'train')
            img_dir = CONFIG.train_dir if source == 'train' else CONFIG.test_dir

        img_path = os.path.join(img_dir, filename)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: File not found at {img_path}")
            if self.is_test:
                return None, None
            else:
                return None, None, None

        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]

        if self.is_test:
            return image, filename
        else:
            label = row["label"]
            confidence = row.get("confidence", 1.0)
            return image, torch.tensor(label, dtype=torch.long), torch.tensor(confidence, dtype=torch.float)

def mixup_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(CONFIG.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1.0
    
    rand_index = torch.randperm(x.size()[0]).to(CONFIG.device)
    y_a, y_b = y, y[rand_index]
    
    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, sample_weights=None):
    loss_a = criterion(pred, y_a, reduction='none')
    loss_b = criterion(pred, y_b, reduction='none')
    loss = lam * loss_a + (1 - lam) * loss_b
    if sample_weights is not None:
        loss *= sample_weights
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, reduction='mean'):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if reduction == "mean":
            return focal_loss.mean()
        elif reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class SheepClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, dropout_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, drop_path_rate=0.2)
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        with torch.no_grad():
            return self.backbone(x)

def train_one_epoch(model, loader, optimizer, criterion, scheduler, scaler):
    model.train()
    total_loss, correct_predictions, total_samples = 0, 0, 0
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        if batch[0] is None: continue
        images, labels, confidences = batch
        images, labels, confidences = images.to(CONFIG.device), labels.to(CONFIG.device), confidences.to(CONFIG.device)
        
        use_mixup_cutmix = CONFIG.mixup_cutmix_alpha > 0 and random.random() < 0.5
        if use_mixup_cutmix:
            if random.random() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, labels, CONFIG.mixup_cutmix_alpha)
            else:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, CONFIG.mixup_cutmix_alpha)
        else:
            targets_a, targets_b, lam = labels, labels, 1.0

        with torch.cuda.amp.autocast():
            outputs = model(images)
            if use_mixup_cutmix:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam, sample_weights=confidences)
            else:
                base_loss = criterion(outputs, labels, reduction='none')
                loss = (base_loss * confidences).mean()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()
        total_samples += labels.size(0)
        preds = outputs.argmax(dim=1)
        if use_mixup_cutmix:
            correct_predictions += (lam * (preds == targets_a).sum() + (1 - lam) * (preds == targets_b).sum()).item()
        else:
            correct_predictions += (preds == labels).sum().item()
            
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_predictions / total_samples:.4f}")
        
    return total_loss / len(loader), correct_predictions / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    for batch in tqdm(loader, desc="Validating"):
        if batch[0] is None: continue
        images, labels, _ = batch
        images, labels = images.to(CONFIG.device), labels.to(CONFIG.device)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return total_loss / len(loader), accuracy, f1_macro

@torch.no_grad()
def get_predictions_and_features(models, loader):
    all_probs = []
    all_features = []
    all_filenames = []

    for images, filenames in tqdm(loader, desc="Extracting Features & Probs"):
        if images is None: continue
        images = images.to(CONFIG.device)
        
        batch_probs_list = []
        batch_feats_list = []
        
        for model in models:
            model.eval()
            with torch.cuda.amp.autocast():
                features = model.get_features(images)
                outputs = model.classifier(features)
            batch_probs_list.append(torch.softmax(outputs, dim=1).cpu().numpy())
            batch_feats_list.append(features.cpu().numpy())
            
        avg_probs = np.mean(batch_probs_list, axis=0)
        avg_feats = np.mean(batch_feats_list, axis=0)
        
        all_probs.append(avg_probs)
        all_features.append(avg_feats)
        all_filenames.extend(filenames)

    return np.concatenate(all_probs), np.concatenate(all_features), all_filenames

def train_cross_validation(df, is_pseudo=False):
    fold_scores = []
    skf = StratifiedKFold(n_splits=CONFIG.n_folds, shuffle=True, random_state=CONFIG.seed)
    
    source_map = dict(zip(df.filename, df.source)) if 'source' in df.columns else {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.label)):
        print(f"\n===== FOLD {fold+1} / {CONFIG.n_folds} =====")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_ds = SheepDataset(train_df, transform=get_train_transforms(), source_map=source_map)
        val_ds = SheepDataset(val_df, transform=get_valid_transforms(), source_map=source_map)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=CONFIG.batch_size * 2, shuffle=False, num_workers=4)

        model = SheepClassifier(CONFIG.model_name, CONFIG.num_classes).to(CONFIG.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.epochs * len(train_loader), eta_min=CONFIG.min_lr)
        
        class_weights = compute_class_weight("balanced", classes=np.unique(df.label), y=df.label)
        criterion = FocalLoss(alpha=torch.tensor(class_weights, dtype=torch.float).to(CONFIG.device))
        
        scaler = torch.cuda.amp.GradScaler()
        best_f1 = 0
        patience_counter = 0

        for epoch in range(CONFIG.epochs):
            print(f"--- Epoch {epoch+1}/{CONFIG.epochs} ---")
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                model_prefix = "pseudo" if is_pseudo else "cv"
                torch.save(model.state_dict(), os.path.join(CONFIG.models_dir, f"{model_prefix}_fold_{fold+1}.pth"))
                print(f"ðŸš€ Model saved with F1: {best_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= CONFIG.patience:
                print("Early stopping triggered.")
                break
        fold_scores.append(best_f1)
    
    print(f"\nCV finished. Mean F1: {np.mean(fold_scores):.4f}")
    return fold_scores

def load_models(model_prefix):
    models = []
    for fold in range(1, CONFIG.n_folds + 1):
        model_path = os.path.join(CONFIG.models_dir, f"{model_prefix}_fold_{fold}.pth")
        if os.path.exists(model_path):
            model = SheepClassifier(CONFIG.model_name, CONFIG.num_classes).to(CONFIG.device)
            model.load_state_dict(torch.load(model_path))
            models.append(model)
    print(f"Loaded {len(models)} models with prefix '{model_prefix}'")
    return models

@torch.no_grad()
def ensemble_predict_with_tta(models, test_loader, model_weights=None):
    all_preds, all_filenames = [], []
    
    for images, filenames in tqdm(test_loader, desc="Predicting with TTA Ensemble"):
        if images is None: continue
        images = images.to(CONFIG.device)
        
        batch_probs = []
        for model in models:
            model.eval()
            with torch.cuda.amp.autocast():
                outputs_original = model(images)
                outputs_flipped = model(torch.flip(images, dims=[3]))
            
            probs = (torch.softmax(outputs_original, dim=1) + torch.softmax(outputs_flipped, dim=1)) / 2
            batch_probs.append(probs.cpu().numpy())
            
        if model_weights is None:
            avg_probs = np.mean(batch_probs, axis=0)
        else:
            avg_probs = np.average(np.stack(batch_probs, axis=0), axis=0, weights=model_weights)
            
        preds = np.argmax(avg_probs, axis=1)
        all_preds.extend(preds)
        all_filenames.extend(filenames)
        
    return all_preds, all_filenames

def main():
    print("Starting the enhanced training pipeline.")
    
    df = pd.read_csv(CONFIG.train_csv)
    labels = sorted(df["label"].unique())
    label2idx = {v: i for i, v in enumerate(labels)}
    idx2label = {i: v for v, i in label2idx.items()}
    df["label"] = df["label"].map(label2idx)
    df['source'] = 'train'
    df['confidence'] = 1.0

    print("\n--- PHASE 1: Initial CV Training ---")
    cv_scores = train_cross_validation(df, is_pseudo=False)
    np.save(os.path.join(CONFIG.models_dir, "cv_scores.npy"), cv_scores)

    print("\n--- PHASE 2: Data Enrichment ---")
    initial_models = load_models("cv")
    
    test_files = [f for f in os.listdir(CONFIG.test_dir) if f.lower().endswith('.jpg')]
    test_df_initial = pd.DataFrame({'filename': test_files})
    test_ds = SheepDataset(test_df_initial, transform=get_valid_transforms(), is_test=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG.batch_size*2, shuffle=False)

    test_probs, test_features, test_filenames = get_predictions_and_features(initial_models, test_loader)
    
    confidences = np.max(test_probs, axis=1)
    predictions = np.argmax(test_probs, axis=1)

    pseudo_threshold = 0.97
    pseudo_indices = np.where(confidences >= pseudo_threshold)[0]
    pseudo_df = pd.DataFrame({
        'filename': [test_filenames[i] for i in pseudo_indices],
        'label': predictions[pseudo_indices],
        'confidence': confidences[pseudo_indices],
        'source': 'pseudo'
    })
    print(f"Generated {len(pseudo_df)} pseudo-labels.")

    k = CONFIG.num_classes
    kmeans = KMeans(n_clusters=k, random_state=CONFIG.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(test_features)

    cluster_df_list = []
    purity_threshold = 0.90
    for i in range(k):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) == 0: continue
        
        cluster_preds = predictions[cluster_indices]
        major_class, count = Counter(cluster_preds).most_common(1)[0]
        purity = count / len(cluster_indices)

        if purity >= purity_threshold:
            print(f"Cluster {i} is pure. Label: {major_class}, Purity: {purity:.2f}")
            for idx in cluster_indices:
                if confidences[idx] < pseudo_threshold:
                    cluster_df_list.append({
                        'filename': test_filenames[idx],
                        'label': major_class,
                        'confidence': purity,
                        'source': 'cluster'
                    })
    
    cluster_df = pd.DataFrame(cluster_df_list)
    print(f"Generated {len(cluster_df)} labels from clustering.")

    merged_df = pd.concat([df, pseudo_df, cluster_df], ignore_index=True)
    print(f"Total training samples for Phase 3: {len(merged_df)}")
    
    print("\n--- PHASE 3: Re-training on Combined Data ---")
    pseudo_scores = train_cross_validation(merged_df, is_pseudo=True)
    np.save(os.path.join(CONFIG.models_dir, "pseudo_scores.npy"), pseudo_scores)
    
    print("\n--- FINAL PREDICTION ---")
    final_cv_models = load_models("cv")
    final_pseudo_models = load_models("pseudo")
    all_models = final_cv_models + final_pseudo_models
    
    all_scores = np.concatenate([
        np.load(os.path.join(CONFIG.models_dir, "cv_scores.npy")),
        np.load(os.path.join(CONFIG.models_dir, "pseudo_scores.npy"))
    ])
    
    final_preds, final_filenames = ensemble_predict_with_tta(all_models, test_loader, model_weights=all_scores)
    
    submission_df = pd.DataFrame({'filename': final_filenames, 'label': [idx2label[p] for p in final_preds]})
    submission_df.to_csv('submission.csv', index=False)

    print("\nâœ… Enhanced pipeline complete. Submission file created: submission.csv")

if __name__ == '__main__':
    main()

