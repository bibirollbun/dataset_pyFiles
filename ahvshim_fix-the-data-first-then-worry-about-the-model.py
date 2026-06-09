# ======================
# Standard Library Imports
# ======================
import os
import random
import math
from pathlib import Path
from typing import Any, Dict
import warnings
warnings.filterwarnings('ignore')

# ======================
# 3rd-Party Imports
# ======================
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
from torchsummary import summary

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ======================
# MAIN CONFIGURATIONS
# ======================
class Config:

    seed=7
    img_size=224
    model_name='vit_base_patch16_224.augreg_in21k_ft_in1k'
    num_classes=7
    batch_size=8
    epochs=50
    lr=1e-4
    min_lr=1e-6
    n_folds=5
    patience=5
    train_csv='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv'
    train_dir='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'
    test_dir='/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test'
    models_dir='models'
    results_dir='results'
    processed_data_dir='processed_data'
    device="cuda" if torch.cuda.is_available() else "cpu"

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
# Setting a seed for reproducibility
SeedManager(CONFIG.seed).seed_everything()

# Creating new folders, for initial results, and final results (will be used later)
INITIAL_RESULTS_DIR = os.path.join(CONFIG.results_dir, "initial_results")
os.makedirs(INITIAL_RESULTS_DIR, exist_ok=True)

PSEUDO_RESULTS_DIR = os.path.join(CONFIG.results_dir, "final_results")
os.makedirs(PSEUDO_RESULTS_DIR, exist_ok=True)


# ======================
# CUSTOM DATASET CLASSES
# ======================
class SheepDataset(Dataset):
    def __init__(self, df=None, image_dir=None, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        if self.is_test:
            # For test set, just get sorted list of image files
            self.img_files = sorted(os.listdir(image_dir))
        else:
            # For train/val set, use dataframe with filenames and labels
            self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.img_files) if self.is_test else len(self.df)

    def __getitem__(self, idx):
        if self.is_test:
            filename = self.img_files[idx]
            img_path = os.path.join(self.image_dir, filename)
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, filename
        else:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.image_dir, row["filename"])
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            if self.transform:
                image = self.transform(image=image)["image"]
            label = row["label"]
            return image, torch.tensor(label, dtype=torch.long)


class PseudoDataset(Dataset):
    def __init__(self, df, train_dir, test_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        label = row["label"]
        source = row.get("source", "train")
        confidence = row.get(
            "confidence", 1.0
        )  # default to 1.0 for clean training data

        img_dir = self.test_dir if source == "pseudo" or source == "cluster" else self.train_dir
        img_path = os.path.join(img_dir, filename)

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(confidence, dtype=torch.float),
        )
    

# ======================
# DATA AUGMENTATION FUNCTIONS
# ======================
def get_train_transforms():
    return A.Compose(
        [
            A.Resize(CONFIG.img_size, CONFIG.img_size),

            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.1),
                rotate=(-15, 15),
                shear=5,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                p=0.7,
            ),

            # Color & contrast adjustments
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),

            # Blur / distortion
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.1),
            A.ElasticTransform(alpha=1, sigma=50, p=0.1),

            # Weather simulation
            A.RandomFog(
                fog_coef_range=(0.1, 0.3),  # tuple of min and max fog intensity
                alpha_coef=0.08,
                p=0.2
            ),
            A.RandomRain(blur_value=3, brightness_coefficient=0.9, p=0.1),

            # Occlusion
            A.CoarseDropout(
                num_holes_range=(3, 6),
                hole_height_range=(10, 32),
                hole_width_range=(10, 32),
                fill=0,
                p=0.3,
            ),

            # Normalize and tensor
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(CONFIG.img_size, CONFIG.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


# ======================
# COMPUTE CLASS WEIGHTS
# ======================
def compute_class_weights(labels, method="balanced"):
    labels = np.asarray(labels)
    assert np.issubdtype(labels.dtype, np.integer), "Labels must be integers."
    classes = np.unique(labels)

    if method == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=labels)
    elif method == "effective":
        # Effective number of samples
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, np.bincount(labels))
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(classes)

    return torch.tensor(weights, dtype=torch.float)


# ======================
# MODEL ARCHITECTURE
# ======================

class ViTClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, dropout_rate=0.4):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        in_features = self.backbone.num_features
        self.backbone.reset_classifier(0)  # Remove default classifier head

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
            feats = self.backbone.forward_head(
                feats, pre_logits=True
            )  # get final embedding
            return feats
        

# ======================
# FOCAL LOSS
# ======================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, weights=None):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if weights is not None:
            focal_loss = focal_loss * weights

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
        

# ======================
# OPTIMIZER AND SCHEDULER
# ======================

def get_optimizer_scheduler(model, train_loader, epochs):

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_head = "classifier" in name

        if "bias" in name or "norm" in name:
            if is_head:
                head_no_decay.append(param)
            else:
                backbone_no_decay.append(param)
        else:
            if is_head:
                head_decay.append(param)
            else:
                backbone_decay.append(param)

    param_groups = [
        {"params": backbone_decay, "weight_decay": 0.01, "lr": float(CONFIG.lr) * 0.1},
        {"params": backbone_no_decay, "weight_decay": 0.0, "lr": float(CONFIG.lr) * 0.1},
        {"params": head_decay, "weight_decay": 0.01, "lr": float(CONFIG.lr)},
        {"params": head_no_decay, "weight_decay": 0.0, "lr": float(CONFIG.lr)},
    ]

    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.999), eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * 3,
        T_mult=2,
        eta_min=float(CONFIG.min_lr),
    )

    return optimizer, scheduler


# ======================
# EARLY STOPPING
# ======================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


# ======================
# TRAINING AND EVALUATION
# ======================

def train_cross_validation(df, pseudo_train=False, results_dir=None):
    if results_dir is None:
        results_dir = CONFIG.results_dir

    class_weights = compute_class_weights(df["label"].values, method="effective").to(
        CONFIG.device
    )
    # print(f"Class weights: {class_weights}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    skf = StratifiedKFold(
        n_splits=CONFIG.n_folds, shuffle=True, random_state=CONFIG.seed
    )

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df.label)):
        print(f"\n{'=' * 70} FOLD {fold+1} {'=' * 70}")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # --- Create fold-specific directories ---
        if pseudo_train:
            fold_results_dir = os.path.join(results_dir, f"fold_{fold+1}")
            model_name = f"pseudo_fold_{fold+1}.pth"
        else:
            fold_results_dir = os.path.join(results_dir, f"fold_{fold+1}")
            model_name = f"cv_fold_{fold+1}.pth"
        os.makedirs(fold_results_dir, exist_ok=True)
        # ----------------------------------------

        # Print fold class distribution
        print(
            f"Train distribution: {train_df['label'].value_counts().sort_index().tolist()}, Length: {len(train_df)}"
        )
        print(
            f"Val distribution: {val_df['label'].value_counts().sort_index().tolist()}, Length: {len(val_df)}"
        )
        if pseudo_train:
            train_ds = PseudoDataset(
                train_df, CONFIG.train_dir, CONFIG.test_dir, get_train_transforms()
            )
            val_ds = PseudoDataset(
                val_df, CONFIG.train_dir, CONFIG.test_dir, get_valid_transforms()
            )
        else:
            train_ds = SheepDataset(train_df, CONFIG.train_dir, get_train_transforms())
            val_ds = SheepDataset(val_df, CONFIG.train_dir, get_valid_transforms())

        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=CONFIG.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model = ViTClassifier(CONFIG.model_name, CONFIG.num_classes).to(CONFIG.device)
        optimizer, scheduler = get_optimizer_scheduler(
            model, train_loader, CONFIG.epochs
        )
        scaler = torch.amp.GradScaler(device=CONFIG.device)
        early_stopping = EarlyStopping(patience=CONFIG.patience)

        # Initialize history tracking for this fold
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
            "val_f1_weighted": [],
        }

        best_f1 = 0
        class_report, cm = "", ""

        for epoch in range(CONFIG.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, scheduler, scaler, epoch
            )

            eval_results = evaluate(model, val_loader, criterion)

            val_f1_macro = eval_results["metrics"]["f1_macro"]
            val_f1_weighted = eval_results["metrics"]["f1_weighted"]
            val_acc = eval_results["metrics"]["accuracy"]
            val_loss = eval_results["metrics"]["avg_loss"]
            all_labels = eval_results["predictions"]["all_labels"]
            all_preds = eval_results["predictions"]["all_preds"]

            # Store metrics in history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1_macro"].append(val_f1_macro)
            history["val_f1_weighted"].append(val_f1_weighted)

            print(
                f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val F1 Macro: {val_f1_macro:.4f} | Val F1 Weighted: {val_f1_weighted:.4f}"
            )

            if val_f1_macro > best_f1:
                best_f1 = val_f1_macro
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG.models_dir, model_name),
                )
                # Generate and store the classification report only when a new best model is found
                class_report = classification_report(all_labels, all_preds, digits=4)
                print(
                    f"New best F1-macro for Fold {fold+1} at epoch {epoch+1}. Model saved."
                    f" Current Best F1-macro: {best_f1:.4f}"
                )
                # Generate confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                cm = str(cm)  # Convert numpy array to string for saving

            if early_stopping(val_f1_macro, model):
                print(f"Early stopping at epoch {epoch+1}")
                break

        # --- Save best classification report for this fold ---
        if class_report:
            report_path = os.path.join(fold_results_dir, f"fold_{fold+1}_report.txt")
            with open(report_path, "w") as f:
                f.write(class_report)
            print("\n----- Classification Report -----")
            print(class_report)

        # --- Save best confusion matrix for this fold ---
        if cm:
            cm_path = os.path.join(
                fold_results_dir, f"fold_{fold+1}_confusion_matrix.txt"
            )
            with open(cm_path, "w") as f:
                f.write(cm)
            print("\n----- Confusion Matrix -----")
            print(cm)

        # --- Plot metrics ---
        plot_metrics(
            history, os.path.join(fold_results_dir, f"fold_{fold+1}_metrics.png")
        )

        # --- Save history ---
        pd.DataFrame(history).to_csv(
            os.path.join(fold_results_dir, f"history_fold_{fold+1}.csv"), index=False
        )

        fold_scores.append(best_f1)
        print(f"\nFold {fold+1} best F1: {best_f1:.4f}")

    print("\nCross-validation results:")
    print(f"Mean F1: {np.mean(fold_scores):.4f} Â± {np.std(fold_scores):.4f}")
    print(f"Individual fold scores: {fold_scores}")

    if pseudo_train:
        fold_score_path = os.path.join(CONFIG.models_dir, "pseudo_fold_scores.npy")
    else:
        fold_score_path = os.path.join(CONFIG.models_dir, "cv_fold_scores.npy")
    np.save(fold_score_path, np.array(fold_scores))

    return fold_scores


def train_one_epoch(model, loader, optimizer, criterion, scheduler, scaler, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for batch in pbar:
        if len(batch) == 3:
            images, labels, confidences = batch
            confidences = confidences.to(CONFIG.device)
        else:
            images, labels = batch
            confidences = None

        images = images.to(CONFIG.device)
        labels = labels.to(CONFIG.device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=CONFIG.device):
            outputs = model(images)
            loss = criterion(outputs, labels, weights=confidences)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        # _, predicted = torch.max(outputs.data, 1)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
        )

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            if len(batch) == 3:
                images, labels, confidences = batch
                confidences = confidences.to(CONFIG.device)
            else:
                images, labels = batch
                confidences = None

            images = images.to(CONFIG.device)
            labels = labels.to(CONFIG.device)

            with torch.amp.autocast(device_type=CONFIG.device):
                outputs = model(images)
                if criterion is not None:
                    if confidences is not None:
                        loss = criterion(outputs, labels, weights=confidences)
                    else:
                        loss = criterion(outputs, labels)
                    total_loss += loss.item()

            preds = torch.argmax(outputs, 1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader) if criterion is not None else 0

    return {
        "metrics": {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": accuracy,
            "avg_loss": avg_loss,
        },
        "predictions": {
            "all_preds": all_preds,
            "all_labels": all_labels,
        },
    }


# ======================
# PLOTTING METRICS
# ======================

def plot_metrics(history, save_path):
    sns.set_style("darkgrid")
    colors = {"train": "#124467", "val": "#084d02"}

    # Make all metrics same length (minimum length across)
    min_len = min(len(v) for v in history.values())
    for k in history:
        history[k] = history[k][:min_len]
    epochs = range(1, min_len + 1)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))  # 1 row, 4 columns

    # Loss Plot
    ax = axes[0]
    ax.plot(
        epochs,
        history["train_loss"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=colors["train"],
        label="Train Loss",
    )
    ax.plot(
        epochs,
        history["val_loss"],
        marker="s",
        markersize=6,
        linewidth=2,
        color=colors["val"],
        label="Val Loss",
    )
    ax.set_title("Train and Val Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Acc Plot
    ax = axes[1]
    ax.plot(
        epochs,
        history["train_acc"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=colors["train"],
        label="Train Acc",
    )
    ax.plot(
        epochs,
        history["val_acc"],
        marker="s",
        markersize=6,
        linewidth=2,
        color=colors["val"],
        label="Val Acc",
    )
    ax.set_title("Train and Val Acc", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    # F1 Macro Score
    ax = axes[2]
    ax.plot(
        epochs,
        history["val_f1_macro"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=colors["val"],
        label="Val F1 Macro",
    )
    ax.set_title("Val F1 Macro", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    # F1 Weighted Score
    ax = axes[3]
    ax.plot(
        epochs,
        history["val_f1_weighted"],
        marker="D",
        markersize=6,
        linewidth=2,
        color=colors["val"],
        label="Val F1 Weighted",
    )
    ax.set_title("Val F1 Weighted", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(
        save_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return fig


# ======================
# ENSEMBLE PREDICTION
# ======================

def ensemble_predict(
    models,
    test_loader,
    fold_scores=None
):
    """
    Generate ensemble predictions from a list of models
    """
    if fold_scores is not None:
        assert len(fold_scores) == len(models), f"Length of model scores ({len(fold_scores)}) " \
            f"must match number of models ({len(models)})."
        
        if isinstance(fold_scores, list):
            fold_scores = np.asarray(fold_scores)
        # Normalize weights
        model_weights = fold_scores / np.sum(fold_scores)
    else:
        model_weights = None

    all_preds, all_confidences, all_filenames, all_class_probs = [], [], [], []

    for images, filenames in tqdm(test_loader, desc="Predicting"):
        images = images.to(CONFIG.device)
        batch_logits = []

        with torch.no_grad():
            for model in models:
                with torch.amp.autocast(device_type=CONFIG.device):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    batch_logits.append(probs.cpu().numpy())
        
        for i, logit in enumerate(batch_logits):
            assert logit.shape == batch_logits[0].shape, f"Inconsistent shape in model {i}"

        if model_weights is None:
            avg_probs = np.mean(batch_logits, axis=0)
        else:
            # Weighted soft voting with model weights
            # 
            stacked_logits = np.stack(batch_logits, axis=0)  # shape: (num_models, B, C)
            avg_probs = np.average(stacked_logits, axis=0, weights=model_weights) # shape: (B, C)

        preds = np.argmax(avg_probs, axis=1)
        confidences = np.max(avg_probs, axis=1)

        all_preds.extend(preds)
        all_confidences.extend(confidences)
        all_filenames.extend(filenames)
        all_class_probs.extend(avg_probs)
    
    # ===== Print some stats =====
    print(f"\nTotal predictions: {len(all_preds)}")
    print(f"Average confidence: {np.mean(all_confidences):.4f}")
    print(f"Min confidence: {np.min(all_confidences):.4f}")
    print(f"Max confidence: {np.max(all_confidences):.4f}\n")

    # ===== Print model weights and fold scores =====
    if model_weights is not None:
        for i, (w, s) in enumerate(zip(model_weights, fold_scores)):
            print(
                f"Model {i+1} | Weight: {w:.3f} | Fold score: {s:.4f} | Type: {'Initial' if i < CONFIG.n_folds else 'Pseudo'}"
            )

    return all_preds, all_confidences, all_filenames, all_class_probs


# ======================
# HELPER FUNCTIONS
# ======================

def get_label_maps():
    df = pd.read_csv(CONFIG.train_csv)
    labels = sorted(df["label"].unique())
    label2idx = {v: i for i, v in enumerate(labels)}
    idx2label = {i: v for v, i in label2idx.items()}
    return label2idx, idx2label

def denormalize(img_tensor, mean, std):
    # img_tensor shape: (C, H, W)
    img = img_tensor.cpu().numpy()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = img.clip(0, 1)
    # transpose from (C, H, W) to (H, W, C) for imshow
    img = img.transpose(1, 2, 0)
    return img

def plot_augmented_samples(image_path, n=6):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug = get_train_transforms()
    _, axs = plt.subplots(1, n, figsize=(18, 6))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(n):
        augmented = aug(image=img)
        denorm_img = denormalize(augmented['image'], mean, std)
        axs[i].imshow(denorm_img)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()


def load_models(model_paths):
    models = []
    for model_file in tqdm(model_paths, desc="Loading models"):
        model_path = os.path.join(CONFIG.models_dir, model_file)
        model = ViTClassifier(CONFIG.model_name, CONFIG.num_classes).to(CONFIG.device)
        state_dict = torch.load(
            model_path, map_location=CONFIG.device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    print(f"\nLoaded {len(models)} models")
    return models


def load_test_data():
    test_files = sorted(
        [f for f in os.listdir(CONFIG.test_dir) if f.lower().endswith(".jpg")]
    )
    test_ds = SheepDataset(
        image_dir=CONFIG.test_dir, transform=get_valid_transforms(), is_test=True
    )
    test_ds.img_files = test_files

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Test set size: {len(test_loader.dataset)}, batch size: {CONFIG.batch_size}")
    return test_loader


# ======================
# PSEUDO LABEL GENERATION
# ======================

def generate_pseudo_labels(models, test_loader, threshold):
    assert len(models) > 0, "No models provided"
    assert len(test_loader) > 0, "No test loader provided"

    pseudo_data = []
    for images, filenames in tqdm(test_loader, desc=f"Generating pseudo labels â‰¥ {threshold:.2f} confidence"):
        images = images.to(CONFIG.device)
        batch_logits = []

        with torch.no_grad():
            for model in models:
                with torch.amp.autocast(device_type=CONFIG.device):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    batch_logits.append(probs.cpu().numpy())

        avg_probs = np.mean(batch_logits, axis=0)
        preds = np.argmax(avg_probs, axis=1)
        confidences = np.max(avg_probs, axis=1)

        for fname, pred, conf in zip(filenames, preds, confidences):
            if conf >= threshold:
                pseudo_data.append(
                    {"filename": fname, "label": pred, "confidence": conf}
                )

    pseudo_df = pd.DataFrame(pseudo_data)
    pseudo_df["source"] = (
        "pseudo"  # adding this column to know the source of the pseudo labels
    )
    print(f"Generated {len(pseudo_df)} pseudo-labels out of {len(test_loader.dataset)} test images")
    print(f"Excluded {len(test_loader.dataset) - len(pseudo_df)} low-confidence predictions")
    return pseudo_df

def load_pseudo_labels(pseudo_df):
    # create a map filename -> (label, confidence)
    label_map = dict(zip(pseudo_df["filename"], pseudo_df["label"]))
    conf_map = dict(zip(pseudo_df["filename"], pseudo_df["confidence"]))
    return label_map, conf_map


# ======================
# KMEANS CLUSTERING
# ======================

class KMeansClustering:
    def __init__(self, pseudo_df, train_df, output_dir, purity_threshold=0.9):
        self.config = CONFIG
        self.train_df = train_df
        self.output_dir = output_dir
        self.pseudo_df = pseudo_df
        self.purity_threshold = purity_threshold
        self.models = self._load_models()
        self.test_loader = self._load_test_loader()
        self.filenames = self.test_loader.dataset.img_files

    def _load_models(self):
        model_files = sorted(
            [f for f in os.listdir(self.config.models_dir) if f.endswith(".pth")]
        )
        return load_models(model_files)

    def _load_test_loader(self):
        files = sorted(
            [f for f in os.listdir(self.config.test_dir) if f.lower().endswith(".jpg")]
        )
        dataset = SheepDataset(
            image_dir=self.config.test_dir,
            transform=get_valid_transforms(),
            is_test=True,
        )
        dataset.img_files = files
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def run(self):
        print(f"{len(self.filenames)} test images loaded.")
        print(f"Using {len(self.models)} ensemble models.")

        features, filenames = extract_features(self.models, self.test_loader)

        # Run clustering
        embedding, cluster_labels = run_clustering(features, k=self.config.num_classes)

        # Load pseudo-labels
        pseudo_label_map, pseudo_conf_map = load_pseudo_labels(self.pseudo_df)

        predicted_label_names = get_cluster_labels_from_pseudo(
            cluster_labels, filenames, pseudo_label_map
        )

        df_clusters = pd.DataFrame(
            {
                "filename": filenames,
                "cluster": cluster_labels,
                "u1": embedding[:, 0],
                "u2": embedding[:, 1],
                "pred_label": predicted_label_names,
                "pconf": [pseudo_conf_map.get(f, np.nan) for f in filenames],
            }
        )
        # Visualize the clusters
        show_clusters(df_clusters, output_dir=self.output_dir)

        df_clusters.to_csv(
            os.path.join(self.output_dir, "clustered_test_results.csv"), index=False
        )
        print(f"Clustered CSV saved to {os.path.join(self.output_dir, 'clustered_test_results.csv')}")

        print("Calculating cluster purity...")
        purity_map, cluster_label_map = calc_cluster_purity(df_clusters)

        print("Building merged CSV...")
        merged_df=build_csv(
            train_df=self.train_df,
            cluster_df=df_clusters,
            purity_map=purity_map,
            label_map=cluster_label_map,
            feats=features,
            output_dir=self.output_dir,
            purity_threshold=self.purity_threshold,
            return_df=True,
        )

        return df_clusters, merged_df


def run_clustering(features, k=7):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    reducer = umap.UMAP(n_components=2, random_state=CONFIG.seed, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(scaled)
    clusterer = KMeans(k, random_state=CONFIG.seed)
    cluster_labels = clusterer.fit_predict(scaled)
    return embedding, cluster_labels


def get_cluster_labels_from_pseudo(cluster_labels, filenames, pseudo_map):
    """
    Generally the cluster labels are organized as `cluster_1`, `cluster_2`, etc.
    This function will map the cluster labels to the pseudo labels, and return the label names.
    """
    cluster_map = {}
    unique_clusters = np.unique(cluster_labels)
    print(
        f"Number of clusters found: {len(unique_clusters[unique_clusters >= 0])}"
    )

    pseudo_labels = [pseudo_map.get(f, None) for f in filenames]

    for c in unique_clusters:
        if c == -1:
            cluster_map[c] = "noise"
            continue
        # Get pseudo labels of samples in this cluster
        plabels = [
            pseudo_labels[i]
            for i in range(len(filenames))
            if cluster_labels[i] == c and pseudo_labels[i] is not None
        ]
        cluster_map[c] = Counter(plabels).most_common(1)[0][0] if plabels else "unknown"

    label_names = [cluster_map.get(c, "unknown") for c in cluster_labels]
    return label_names


def show_clusters(df_clusters, output_dir=None):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_clusters,
        x='u1', y='u2',
        hue='pred_label',
        palette='tab10',
        s=20, alpha=0.8
    )
    plt.title("UMAP projection by pseudo-label")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "cluster_umap.png"))
    plt.show()

def calc_cluster_purity(df):
    purity, label_map = {}, {}
    for c in df["cluster"].unique():
        if c == -1:
            continue
        group = df[df["cluster"] == c]["pred_label"]
        freq = group.value_counts(normalize=True)
        purity[c] = freq.max()
        label_map[c] = freq.idxmax()
    return purity, label_map


def filter_core_samples(df, features, pct=0.5):
    df = df[df["cluster"] != -1].copy()
    selected = []
    for c in df["cluster"].unique():
        idxs = df[df["cluster"] == c].index
        feats = features[idxs]
        dists = cdist(feats, feats.mean(axis=0, keepdims=True)).flatten()
        cutoff = np.quantile(dists, pct)
        selected.extend(i for i, d in zip(idxs, dists) if d <= cutoff)
    return df.loc[selected].copy()


def build_csv(
    train_df,
    cluster_df,
    purity_map,
    label_map,
    feats,
    output_dir,
    purity_threshold=0.9,
    return_df=False,
):
    print(f"Selected purity threshold: {purity_threshold}")

    # Original train data
    train = train_df[["filename", "label"]].copy()
    train["conf"] = 1.0
    train["src"] = "train"
    print(f"Original train samples: {len(train)}")

    # Pseudo labeled data
    pseudo = cluster_df[~cluster_df["pconf"].isna()].copy()
    pseudo = pseudo[["filename", "pred_label", "pconf"]].rename(
        columns={"pred_label": "label", "pconf": "conf"}
    )
    pseudo["src"] = "pseudo"
    print(f"Pseudo labeled samples: {len(pseudo)}")

    # Cluster data filtered by purity threshold and core samples
    cluster = cluster_df[cluster_df["pconf"].isna()].copy()
    cluster = cluster[cluster["cluster"].map(purity_map) >= purity_threshold].copy()
    cluster = cluster[cluster['cluster'].map(label_map) != 'unknown']
    cluster = cluster[cluster['cluster'].map(label_map) != 'noise']
    cluster = filter_core_samples(cluster, feats)
    cluster["label"] = cluster["cluster"].map(label_map)
    cluster["conf"] = purity_threshold
    cluster["src"] = "cluster"
    cluster = cluster[["filename", "label", "conf", "src"]]
    print(
        f"Cluster samples after purity filtering and core sample filtering: {len(cluster)}"
    )

    # Merge all
    merged_df = pd.concat([train, pseudo, cluster], ignore_index=True)
    merged_df.rename(columns={
        'conf': 'confidence',
        'src': 'source'
    }, inplace=True)
    merged_df.to_csv(
        os.path.join(output_dir, "pseudo_clustered_merged.csv"), index=False
    )
    print(
        f"Merged CSV saved to {os.path.join(output_dir, 'pseudo_clustered_merged.csv')}"
    )
    print(f"Total samples after merging: {len(merged_df)}")

    if return_df:
        return merged_df

def extract_features(models, loader):
    assert len(models) > 0, "No models provided"

    print("Using {} models for feature extraction".format(len(models)))
    all_features = []
    all_filenames = []
    for images, filenames in tqdm(loader, desc="Extracting features"):
        images = images.to(CONFIG.device)
        batch_feats = []
        for model in models:
            feats = model.get_features(images)
            batch_feats.append(feats.cpu().numpy())
        # Average ensemble features from all models
        batch_feats = np.stack(batch_feats).mean(axis=0)
        all_features.append(batch_feats)
        all_filenames.extend(filenames)
    all_features = np.concatenate(all_features, axis=0)
    return all_features, all_filenames


plot_augmented_samples("/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train/0048b660.jpg")


# ======================
# Data Loading
# ======================

# Load the training data
train_df = pd.read_csv(CONFIG.train_csv)

label2idx, idx2label = get_label_maps()

# Map labels to indices
train_df["label"] = train_df["label"].map(label2idx)
display(train_df)


# Compute class weights
class_weights = compute_class_weights(train_df["label"].values, method="effective").to(CONFIG.device)
print(class_weights)


# Plot class distribution and effective weights
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

# Class distribution
counts = np.bincount(train_df["label"].values)
ax1.bar(range(len(counts)), counts)
ax1.set_title("Class Distribution")
ax1.set_xlabel("Class")
ax1.set_ylabel("Count")

# Class weights
ax2.bar(range(len(class_weights)), class_weights.cpu().numpy())
ax2.set_title("Effective Number Weights")
ax2.set_xlabel("Class")
ax2.set_ylabel("Weight")

plt.tight_layout()
plt.show()


# computing weights using sklearnâ€™s built-in class weight computation based on inverse class frequency
# we are not using this method, but it's good to know about it
_class_weights = compute_class_weights(train_df["label"].values, method="balanced").to(CONFIG.device)
print(_class_weights)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

# Class distribution
counts = np.bincount(train_df["label"].values)
ax1.bar(range(len(counts)), counts, color="red")
ax1.set_title("Class Distribution")
ax1.set_xlabel("Class")
ax1.set_ylabel("Count")

# Class weights
ax2.bar(range(len(_class_weights)), _class_weights.cpu().numpy(), color="red")
ax2.set_title("Balanced Number of Weights")
ax2.set_xlabel("Class")
ax2.set_ylabel("Weight")

plt.tight_layout()
plt.show()

# delete the unused variable
del _class_weights


model = ViTClassifier(CONFIG.model_name, CONFIG.num_classes).to(CONFIG.device)
summary(model, (3, CONFIG.img_size, CONFIG.img_size))


# ======================
# Cross-validation training
# ======================

print("=" * 60)
print("PHASE 1: Initial Cross-Validation Training")
print("=" * 60)

fold_scores = train_cross_validation(
    train_df, pseudo_train=False, results_dir=INITIAL_RESULTS_DIR
)


# Load the 5 models we just trained
model_files = [f for f in os.listdir(CONFIG.models_dir) if f.endswith(".pth")]
sorted_model_files = sorted([f for f in model_files if f.startswith("cv_fold_")])

models = load_models(sorted_model_files)

# Load the test set
test_loader = load_test_data()


# ======================
# Predict on test set
# ======================

fold_scores = np.load(os.path.join(CONFIG.models_dir, "cv_fold_scores.npy"))
all_preds, all_confidences, all_filenames, all_probs = ensemble_predict(models, test_loader, fold_scores=fold_scores)

# We can also treat all models equally, without using cross-validation scores
# all_preds, all_confidences, all_filenames, all_probs = ensemble_predict(models, test_loader)

# Convert labels back to original labels
all_labels = [idx2label[pred] for pred in all_preds]


# ===== Save predictions =====
preds_df = pd.DataFrame(
    {
        "filename": all_filenames,
        "label": all_labels,
    }
)
preds_df.to_csv(os.path.join(CONFIG.processed_data_dir, "initial_submission.csv"), index=False)

# ===== Save predictions with confidence =====
conf_preds_df = pd.DataFrame(
    {"filename": all_filenames, "label": all_labels, "confidence": all_confidences}
)
conf_preds_df.to_csv(
    os.path.join(CONFIG.processed_data_dir, "initial_submission_with_confidence.csv"),
    index=False,
)


display(preds_df.head())
display(conf_preds_df.head())


# Plot confidence distribution
plt.figure(figsize=(20, 8))
sns.histplot(conf_preds_df['confidence'], bins=20, kde=True)
plt.title("Prediction Confidence Distribution")
plt.show()


# Zooming in, restricting the plot to the range 0.91 and more
plt.figure(figsize=(20, 8))
sns.histplot(conf_preds_df['confidence'], bins=20, kde=True)

plt.xlim(0.91, 1.00)
plt.xticks(np.round(np.arange(0.91, 1.001, 0.01), 2))
plt.title("Prediction Confidence Distribution")
plt.show()


# Load the predictions with confidence
temp_df = pd.read_csv(os.path.join(CONFIG.processed_data_dir, "initial_submission_with_confidence.csv"))

# Check how many samples we get for each threshold
for threshold in [0.95, 0.96, 0.97, 0.98, 0.99]:
    num_samples = (temp_df["confidence"] >= threshold).sum()
    print(f"Number of pseudo-labels with confidence â‰¥ {threshold}: {num_samples}")


subset = temp_df[(temp_df["confidence"] >= 0.96) & (temp_df["confidence"] < 0.97)].reset_index(drop=True)

for i in range(0, len(subset), 5):
    batch = subset.iloc[i:i+5]
    plt.figure(figsize=(15, 3))
    for j, row in batch.iterrows():
        img_path = os.path.join(CONFIG.test_dir, row["filename"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 5, j % 5 + 1)
        plt.imshow(img)
        plt.title(f'{row["label"]} ({row["confidence"]})')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


print("=" * 60)
print("PHASE 2: Pseudo-Label Generation")
print("=" * 60)

# Generate pseudo-labels with high threshold
pseudo_df = generate_pseudo_labels(models, test_loader, threshold=0.96)
# convert index to categorcal labels (to be matched with train_df when we combine)
pseudo_df["label"] = pseudo_df["label"].map(idx2label)


# Saving it
pseudo_df.to_csv(
    os.path.join(CONFIG.processed_data_dir, "pseudo_labels.csv"), index=False
)


# Analyze pseudo-label distribution
print(f"\nPseudo-label class distribution:")
print(pseudo_df["label"].value_counts().sort_index())


pseudo_df = pd.read_csv(os.path.join(CONFIG.processed_data_dir, "pseudo_labels.csv"))
train_df = pd.read_csv(CONFIG.train_csv)
output_dir = CONFIG.processed_data_dir

print("=" * 60)
print("PHASE 3: K-Mean Clustering")
print("=" * 60)

clusterer = KMeansClustering(pseudo_df=pseudo_df, train_df=train_df, output_dir=output_dir)
df_clusters, merged_df = clusterer.run()


# test dataset, with pseudo confidence (model's output), predicted label (pseudo-labeling), and cluster id (KMeans clustering)

display(df_clusters)


# The new dataset we will use in the next phase, they are a merge of train, and test data (pseudo, and cluster-labeled)
# The source tells us how we got the label.
display(merged_df)


# number of samples in each cluster
df_clusters['cluster'].value_counts().sort_index()


# # The dominant pseudo-labels in each cluster
cluster_label_dist = df_clusters.groupby('cluster')['pred_label'].value_counts(normalize=True).unstack().fillna(0)
cluster_label_dist.style.background_gradient(cmap='Blues')


show_clusters(df_clusters)


clustered_samples = merged_df[merged_df['source'] == 'cluster']
print(f"Number of clustered confident samples: {len(clustered_samples)}")
# Distribution of classes
clustered_samples['label'].value_counts().plot(kind='bar', title='Clustered Label Distribution')


clustered_samples[['filename', 'label', 'confidence']].head(30)


# lets plot them 
n_images = len(clustered_samples)
cols = 8
rows = math.ceil(n_images / cols)

fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
axes = axes.flatten()

for i, row in enumerate(clustered_samples.itertuples()):
    img_path = os.path.join(CONFIG.test_dir, row.filename) # test dir / filename
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(row.label) # label
    axes[i].axis('off')

# turn off any unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# count how many samples come from each source (train / pseudo / cluster)
merged_df['source'].value_counts().plot(kind='bar', title='Sample Count by Source')


# we still got imbalanced dataset, but it's better than before
plt.figure(figsize=(16, 5))
sns.countplot(data=merged_df, y='label', hue='source', palette='Set2')
plt.title('Label Distribution by Data Source')
plt.show()


# conf histogram by source type (train = 1.0, pseudo â‰ˆ 0.97, cluster = 0.9)
# cluster confidence is set to 0.9 as the purity threshold
# pseudo confidence is set to 0.97 as the threshold for pseudo-labeling
# train confidence is 1.0 as we know it's 100% confident
sns.histplot(
    data=merged_df,
    x='confidence',
    hue='source',
    bins=20,
    multiple='stack',
    palette='Set1'
)
plt.title('Confidence Distribution by Source')
plt.show()


# Loading the original train set
merged_df = pd.read_csv(os.path.join(CONFIG.processed_data_dir, "pseudo_clustered_merged.csv"))
print("Length of original training set: {}".format(len(merged_df)))
display(merged_df)


label2idx, idx2label = get_label_maps()
# Map labels to indices
merged_df["label"] = merged_df["label"].map(label2idx)

merged_df.head()


# ===== Train with merged dataset =====
print("=" * 60)
print("PHASE 4: Retraining with Train + Pseudo-Labeled + Clustered Synthetic Data")
print("=" * 60)

final_fold_scores = train_cross_validation(
    merged_df, 
    pseudo_train=True, # MUST BE TRUE
    results_dir=PSEUDO_RESULTS_DIR
)


# Load the 10 models ( 5 initial + 5 pseudo-trained)
# also, make sure the models are loaded in order: CV first, then pseudo, 
# as we want to use the cross-validation scores as model weights

all_model_files = [f for f in os.listdir(CONFIG.models_dir) if f.endswith(".pth")]

cv_model_files = sorted([f for f in all_model_files if f.startswith("cv_fold_")])
pseudo_model_files = sorted([f for f in all_model_files if f.startswith("pseudo_fold_")])
sorted_model_files = cv_model_files + pseudo_model_files

models = load_models(sorted_model_files)

# Load the test set
test_loader = load_test_data()

# Load CV scores for weighting
cv_scores = np.load(os.path.join(CONFIG.models_dir, "cv_fold_scores.npy"))
pseudo_scores = np.load(os.path.join(CONFIG.models_dir, "pseudo_fold_scores.npy"))
print(f"\nCV Scores (Clean): {cv_scores}")
print(f"CV Scores (Pseudo-labeled + Clustered): {pseudo_scores}")


# Concatenate the scores
scores = np.concatenate([
    cv_scores,  # Initial 5 models (clean training)
    pseudo_scores, # * 0.9,  # Pseudo 5 models (no penalty)
])


# ======================
# Predict on test set
# ======================


all_preds, all_confidences, all_filenames, all_probs = ensemble_predict(models, test_loader, fold_scores=scores)

# We can also treat all models equally, without using model weights (cross-validation mean scores and pseudo cross-validation mean scores)
# all_preds, all_confidences, all_filenames, all_probs = ensemble_predict(models, test_loader)


# Convert labels back to original labels
label2idx, idx2label = get_label_maps()
all_labels = [idx2label[pred] for pred in all_preds]


# Create submission
submission_df = pd.DataFrame({
    "filename": all_filenames,
    "label": all_labels
})

submission_df.to_csv(
    os.path.join(CONFIG.processed_data_dir, "final_submission.csv"), 
    index=False
)

# Create submission with confidence
submission_conf_df = pd.DataFrame({
    "filename": all_filenames,
    "label": all_labels,
    "confidence": all_confidences,
})

submission_conf_df.to_csv(
    os.path.join(CONFIG.processed_data_dir, "final_submission_with_confidence.csv"), 
    index=False
)

print(f"Final submission saved with {len(submission_df)} predictions")


display(submission_df)
display(submission_conf_df)


initial_res = pd.read_csv(os.path.join(CONFIG.processed_data_dir, "initial_submission_with_confidence.csv"))
final_res = pd.read_csv(os.path.join(CONFIG.processed_data_dir, "final_submission_with_confidence.csv"))
merged = initial_res.merge(final_res, on='filename', suffixes=('_initial', '_final'))

pred_changed = merged[merged['label_initial'] != merged['label_final']]
for _, row in pred_changed.iterrows():
    im = row['filename']
    
    img = cv2.imread(os.path.join(CONFIG.test_dir, im))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Initial: {row['label_initial']}\nConfidence: {row['confidence_initial']:.2f}")
    axes[0].axis('off')
    
    axes[1].imshow(img)
    axes[1].set_title(f"Final: {row['label_final']}\nConfidence: {row['confidence_final']:.2f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()




