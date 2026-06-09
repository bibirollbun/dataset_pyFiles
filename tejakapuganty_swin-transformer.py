from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import albumentations as A
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import h5py
import torch
import random
import timm
import cv2
import io
import os
import gc
warnings.filterwarnings('ignore')


class CFG:
    train_metadata_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
    test_metadata_path = '/kaggle/input/isic-2024-challenge/test-metadata.csv'
    train_img_path = '/kaggle/input/isic-2024-challenge/train-image.hdf5'
    test_img_path = '/kaggle/input/isic-2024-challenge/test-image.hdf5'
    sample_sub_path = '/kaggle/input/isic-2024-challenge/sample_submission.csv'
    checkpoint_path = '/kaggle/working/checkpoints'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 224
    max_epochs = 50
    learning_rate = 1e-5
    weight_decay = 1e-6
    min_lr = 1e-10
    t_max = 1000
    train_batch_size = 32
    val_batch_size = 64
    n_folds = 5
    seed = 35555
    model_name = "swin_base_patch4_window7_224"
    es_patience = 15


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)


from sklearn.model_selection import train_test_split

train_metadata = pd.read_csv(CFG.train_metadata_path)

print(f"Total samples: {len(train_metadata)}")
print(f"Target distribution:\n{train_metadata['target'].value_counts()}")

# MODIFIED: First split - separate test set (15%)
train_val_metadata, test_metadata = train_test_split(
    train_metadata,
    test_size=0.15,
    random_state=CFG.seed,
    stratify=train_metadata['target']
)

# MODIFIED: Second split - apply K-Fold on remaining train+val data (85%)
sgkf = StratifiedGroupKFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True)
split = sgkf.split(train_val_metadata, train_val_metadata.target, groups=train_val_metadata.patient_id)

for i, (_, val_index) in enumerate(split):
    train_val_metadata.loc[train_val_metadata.index[val_index], 'fold'] = i

test_metadata['fold'] = -1

print(f"\n{'='*60}")
print(f"DATASET SPLITS:")
print(f"{'='*60}")
print(f"Train+Val: {len(train_val_metadata):>6} samples ({len(train_val_metadata)/len(train_metadata)*100:>5.2f}%)")
print(f"Test:      {len(test_metadata):>6} samples ({len(test_metadata)/len(train_metadata)*100:>5.2f}%)")
print(f"{'='*60}")

print(f"\nTest set target distribution:\n{test_metadata['target'].value_counts()}")


class ISICDataset(Dataset):
    def __init__(self, data_path, metadata, transform=None):
        self.data_path = data_path
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform
        # Don't open HDF5 in _init, open in __getitem_ for each worker
        self._data = None
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Lazy load HDF5 file (each worker opens its own handle)
        if self._data is None:
            self._data = h5py.File(self.data_path, 'r', libver='latest', swmr=True)
        
        img_name = self.metadata.iloc[idx]['isic_id']
        
        try:
            # Read image bytes from HDF5
            image_bytes = np.array(self._data[img_name])
            # Decode JPEG
            image = np.array(Image.open(io.BytesIO(image_bytes)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if 'target' in self.metadata.columns:
            label = int(self.metadata['target'].iloc[idx])
            return image, label
        else:
            return image, 0


transforms = {
    'train': A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
        ], p=0.5),
        A.Resize(CFG.image_size, CFG.image_size),
        A.Normalize(
            mean=[0.4815, 0.4578, 0.4082],
            std=[0.2686, 0.2613, 0.2758],
            max_pixel_value=255.0),
        ToTensorV2(),
    ]),
    'val': A.Compose([
        A.Resize(CFG.image_size, CFG.image_size),
        A.Normalize(
            mean=[0.4815, 0.4578, 0.4082],
            std=[0.2686, 0.2613, 0.2758],
            max_pixel_value=255.0),
        ToTensorV2(),
    ])
}


class SwinTransformer(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.model.num_features * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class Trainer:
    def __init__(
        self,
        device,
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        checkpoint_path,
        fold_idx,
        es_patience
    ):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_path = checkpoint_path
        self.best_pauc = 0.0
        self.best_model_path = None
        self.fold_idx = fold_idx
        self.es_patience = es_patience
        self.create_checkpoint_path()

    def create_checkpoint_path(self):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    @staticmethod
    def p_roc_auc_score(y_true, y_preds, min_tpr: float = 0.80):
        v_gt = abs(np.asarray(y_true)-1)
        v_pred = -1.0 * np.asarray(y_preds)
        max_fpr = abs(1-min_tpr)
        fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
        if max_fpr is None or max_fpr == 1:
            return auc(fpr, tpr)
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        stop = np.searchsorted(fpr, max_fpr, "right")
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
        partial_auc = auc(fpr, tpr)
        return partial_auc

    def train(self, current_epoch_nr):
        self.model.train()
        num_batches = len(self.train_dataloader)
        running_loss = 0.0
        total = 0
        preds = []
        targets = []

        loop = tqdm(self.train_dataloader, total=num_batches)
        for batch in loop:
            x, y = batch
            x, y = x.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()
            y_hat = self.model(x).squeeze()

            preds.extend(y_hat.detach().cpu().numpy())
            targets.extend(y.detach().cpu().numpy())

            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x.size(0)
            total += y.size(0)

            loop.set_description(f'Epoch {current_epoch_nr}')
            loop.set_postfix(train_loss=round(running_loss / total, 6))

            if self.scheduler is not None:
                self.scheduler.step()

        train_pauc = self.p_roc_auc_score(targets, preds)
        train_loss = running_loss / num_batches

        return train_pauc, train_loss

    def evaluate(self, current_epoch_nr):
        self.model.eval()
        num_batches = len(self.val_dataloader)
        running_loss = 0.0
        total = 0
        preds = []
        targets = []

        with torch.no_grad():
            loop = tqdm(self.val_dataloader, total=num_batches)
            for batch in loop:
                x, y = batch
                x, y = x.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.float)

                self.optimizer.zero_grad()
                y_hat = self.model(x).squeeze()

                preds.extend(y_hat.detach().cpu().numpy())
                targets.extend(y.detach().cpu().numpy())

                loss = self.criterion(y_hat, y)
                running_loss += loss.item() * x.size(0)
                total += y.size(0)

                loop.set_description(f'Epoch {current_epoch_nr}')
                loop.set_postfix(val_loss=round(running_loss / total, 6))

        val_pauc = self.p_roc_auc_score(targets, preds)
        val_loss = running_loss / num_batches

        if val_pauc > self.best_pauc:
            self.es_patience = CFG.es_patience
            if self.best_pauc != 0.0:
                print(f'New best model found: pAUC = {val_pauc:.6f} (previous best: {self.best_pauc:.6f})')
            self.best_pauc = val_pauc
            checkpoint_name = f'fold_{self.fold_idx}epoch{current_epoch_nr}pauc{round(val_pauc, 6)}.pth'

            for file in os.listdir(self.checkpoint_path):
                if file.startswith(f'fold_{self.fold_idx}epoch'):
                    os.remove(os.path.join(self.checkpoint_path, file))

            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_path, checkpoint_name)
            )
            self.best_model_path = os.path.join(
                self.checkpoint_path, checkpoint_name)

        else:
            self.es_patience -= 1
            if self.es_patience == 0:
                print(f'Early stopping triggered at epoch {current_epoch_nr}. Best pAUC = {self.best_pauc:.6f}')
                return val_pauc, val_loss, True

        return val_pauc, val_loss, False

    def predict(self, dataloader):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.best_model_path))
        model.eval()

        preds = []
        with torch.no_grad():
            loop = tqdm(dataloader, total=len(dataloader))
            for batch in loop:
                x, _ = batch
                x = x.to(self.device, dtype=torch.float)
                y_hat = model(x).squeeze()
                preds.extend(y_hat.detach().cpu().numpy())
                loop.set_description(f'Prediction')

        return preds


# Diagnostic: Check data sizes before training
print(f"{'='*80}")
print(f"DATA SIZE DIAGNOSTIC")
print(f"{'='*80}")

for fold_idx in range(1):  # Just check fold 0
    _train = train_val_metadata[train_val_metadata['fold'] != fold_idx]
    _train_positives = _train[_train.target == 1]
    _train_negatives = _train[_train.target == 0]
    
    print(f"\nFold {fold_idx}:")
    print(f"  Total available: {len(_train)}")
    print(f"  Positives: {len(_train_positives)}")
    print(f"  Negatives available: {len(_train_negatives)}")
    print(f"  Negatives to sample: {20 * len(_train_positives)}")
    
    _train_sampled = pd.concat([_train_positives, _train_negatives.sample(n=20 * len(_train_positives), random_state=CFG.seed)])
    print(f"  Final training size: {len(_train_sampled)}")
    print(f"  Batches (batch_size={CFG.train_batch_size}): {len(_train_sampled) // CFG.train_batch_size}")
    
    estimated_time_per_epoch = (len(_train_sampled) // CFG.train_batch_size) * 0.5  # Assume 0.5 sec per batch
    print(f"  Estimated time per epoch: {estimated_time_per_epoch / 60:.1f} minutes")

print(f"{'='*80}")


histories = {}
oof_pred_probs_df = []
test_preds = []
best_model_paths = []

for fold_idx in range(CFG.n_folds):
    print(f'---------------------------------- Fold {fold_idx + 1} ----------------------------------')
    
    _train = train_val_metadata[train_val_metadata['fold'] != fold_idx].copy()
    _train_positives = _train[_train.target == 1]
    _train_negatives = _train[_train.target == 0]
    _train = pd.concat([_train_positives, _train_negatives.sample(n=20 * len(_train_positives), random_state=CFG.seed)])
    _train = _train.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    
    print(f"Training samples: {len(_train)}")
    
    train_dataset = ISICDataset(CFG.train_img_path, _train, transforms['train'])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=CFG.train_batch_size, 
        shuffle=True, 
        num_workers=0,  # CRITICAL: Set to 0 for HDF5!
        pin_memory=True
    )
    
    _val = train_val_metadata[train_val_metadata['fold'] == fold_idx].copy()
    _val_positives = _val[_val.target == 1]
    _val_negatives = _val[_val.target == 0]
    _val = pd.concat([_val_positives, _val_negatives.sample(n=20 * len(_val_positives), random_state=CFG.seed)])
    _val = _val.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    
    print(f"Validation samples: {len(_val)}")
    
    val_dataset = ISICDataset(CFG.train_img_path, _val, transforms['val'])
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=CFG.val_batch_size, 
        shuffle=False, 
        num_workers=0,  # CRITICAL: Set to 0 for HDF5!
        pin_memory=True
    )
    
    _val_oof = train_val_metadata[train_val_metadata['fold'] == fold_idx].copy()
    val_oof_dataset = ISICDataset(CFG.train_img_path, _val_oof, transforms['val'])
    val_oof_dataloader = DataLoader(
        val_oof_dataset, 
        batch_size=CFG.val_batch_size, 
        shuffle=False, 
        num_workers=0,  # CRITICAL: Set to 0 for HDF5!
        pin_memory=True
    )
    
    CFG.t_max = _train.shape[0] * (CFG.n_folds-1) * CFG.max_epochs // CFG.train_batch_size // CFG.n_folds
    
    model = SwinTransformer(CFG.model_name).to(CFG.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.t_max, eta_min=CFG.min_lr)
    
    trainer = Trainer(
        device=CFG.device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        checkpoint_path=CFG.checkpoint_path,
        fold_idx=fold_idx,
        es_patience=CFG.es_patience
    )
    
    history = {
        'epoch': [],
        'train_pauc': [],
        'train_loss': [],
        'val_pauc': [],
        'val_loss': []
    }
    
    for epoch in range(1, CFG.max_epochs + 1):
        train_pauc, train_loss = trainer.train(current_epoch_nr=epoch)
        val_pauc, val_loss, es_triggered = trainer.evaluate(current_epoch_nr=epoch)
        
        print(f"Epoch {epoch}: Train pAUC: {train_pauc:.6f} - Val pAUC: {val_pauc:.6f} | Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}\n")
        
        history['epoch'].append(epoch)
        history['train_pauc'].append(train_pauc)
        history['train_loss'].append(train_loss)
        history['val_pauc'].append(val_pauc)
        history['val_loss'].append(val_loss)
        
        if es_triggered:
            break
    
    histories[f'Fold {fold_idx + 1}'] = history
    
    best_model_paths.append(trainer.best_model_path)
    
    pred_probs = trainer.predict(val_oof_dataloader)
    _val_oof[CFG.model_name] = pred_probs
    oof_pred_probs_df.append(_val_oof)
    
    del _train, _val, _val_oof, train_dataset, val_dataset, model, criterion, optimizer, scheduler, trainer
    gc.collect()
    torch.cuda.empty_cache()


oof_pred_probs_df = pd.concat(oof_pred_probs_df)[['isic_id', 'fold', 'target', CFG.model_name]]
oof_pred_probs_df.to_csv(f'{CFG.model_name}_oof_preds.csv', index=False)
oof_pred_probs_df.head()


histories = pd.concat([pd.DataFrame(data).assign(Fold=fold) for fold, data in histories.items()])


sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.lineplot(x='epoch', y='train_pauc', hue='Fold', data=histories, ax=axes[0])
axes[0].set_title('Training pAUC')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('pAUC')
axes[0].legend(loc='best')

sns.lineplot(x='epoch', y='val_pauc', hue='Fold', data=histories, ax=axes[1])
axes[1].set_title('Validation pAUC')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('pAUC')
axes[1].legend(loc='best')

fig.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.lineplot(x='epoch', y='train_loss', hue='Fold', data=histories, ax=axes[0])
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(loc='best')

sns.lineplot(x='epoch', y='val_loss', hue='Fold', data=histories, ax=axes[1])
axes[1].set_title('Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(loc='best')

fig.tight_layout()
plt.show()


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    average_precision_score, matthews_corrcoef, cohen_kappa_score
)
import json

print(f"\n{'='*80}")
print(f"FINAL EVALUATION ON TEST SET (15% HELD-OUT DATA)")
print(f"{'='*80}\n")

test_dataset = ISICDataset(CFG.train_img_path, test_metadata, transforms['val'])
test_dataloader = DataLoader(test_dataset, batch_size=CFG.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

all_fold_preds = []

for fold_idx, model_path in enumerate(best_model_paths):
    print(f"Loading Fold {fold_idx + 1} model: {model_path}")
    
    model = SwinTransformer(CFG.model_name).to(CFG.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    fold_preds = []
    with torch.no_grad():
        loop = tqdm(test_dataloader, total=len(test_dataloader), desc=f'Fold {fold_idx + 1} Test Prediction')
        for batch in loop:
            x, _ = batch
            x = x.to(CFG.device, dtype=torch.float)
            y_hat = model(x).squeeze()
            fold_preds.extend(y_hat.cpu().numpy())
    
    all_fold_preds.append(fold_preds)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

test_probs = np.mean(all_fold_preds, axis=0)
test_preds = (test_probs >= 0.5).astype(int)
test_labels = test_metadata['target'].values

test_pauc = Trainer.p_roc_auc_score(test_labels, test_probs, min_tpr=0.80)

accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds, zero_division=0)
recall = recall_score(test_labels, test_preds, zero_division=0)
f1 = f1_score(test_labels, test_preds, zero_division=0)
roc_auc = roc_auc_score(test_labels, test_probs)
avg_precision = average_precision_score(test_labels, test_probs)
mcc = matthews_corrcoef(test_labels, test_preds)
kappa = cohen_kappa_score(test_labels, test_preds)

precision_weighted = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
recall_weighted = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

precision_macro = precision_score(test_labels, test_preds, average='macro', zero_division=0)
recall_macro = recall_score(test_labels, test_preds, average='macro', zero_division=0)
f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)

print(f"\n{'='*80}")
print(f"TEST SET PERFORMANCE METRICS")
print(f"{'='*80}")
print(f"\n{'Metric':<40} {'Score':>15}")
print("-" * 80)
print(f"{'Partial AUC (min_tpr=0.80)':<40} {test_pauc:>15.6f}")
print(f"{'ROC AUC Score':<40} {roc_auc:>15.6f}")
print(f"{'Average Precision Score':<40} {avg_precision:>15.6f}")
print(f"{'Accuracy':<40} {accuracy:>15.4f} ({accuracy*100:.2f}%)")
print(f"{'Matthews Correlation Coefficient':<40} {mcc:>15.4f}")
print(f"{'Cohen Kappa Score':<40} {kappa:>15.4f}")
print("-" * 80)
print(f"\n{'BINARY METRICS (Class 1 - Malignant)':<40}")
print("-" * 80)
print(f"{'Precision':<40} {precision:>15.4f}")
print(f"{'Recall (Sensitivity)':<40} {recall:>15.4f}")
print(f"{'F1 Score':<40} {f1:>15.4f}")
print("-" * 80)
print(f"\n{'WEIGHTED METRICS':<40}")
print("-" * 80)
print(f"{'Precision (Weighted)':<40} {precision_weighted:>15.4f}")
print(f"{'Recall (Weighted)':<40} {recall_weighted:>15.4f}")
print(f"{'F1 Score (Weighted)':<40} {f1_weighted:>15.4f}")
print("-" * 80)
print(f"\n{'MACRO METRICS':<40}")
print("-" * 80)
print(f"{'Precision (Macro)':<40} {precision_macro:>15.4f}")
print(f"{'Recall (Macro)':<40} {recall_macro:>15.4f}")
print(f"{'F1 Score (Macro)':<40} {f1_macro:>15.4f}")
print("=" * 80)

cm = confusion_matrix(test_labels, test_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*80}")
print(f"CONFUSION MATRIX")
print(f"{'='*80}\n")
print(f"                    Predicted")
print(f"                Benign (0)    Malignant (1)")
print(f"Actual Benign (0)    {tn:>6}         {fp:>6}")
print(f"       Malignant (1) {fn:>6}         {tp:>6}\n")

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"{'='*80}")
print(f"DETAILED CLASSIFICATION METRICS")
print(f"{'='*80}")
print(f"\n{'Metric':<40} {'Score':>15}")
print("-" * 80)
print(f"{'Specificity (TNR)':<40} {specificity:>15.4f}")
print(f"{'Sensitivity (TPR/Recall)':<40} {sensitivity:>15.4f}")
print(f"{'Positive Predictive Value (PPV)':<40} {ppv:>15.4f}")
print(f"{'Negative Predictive Value (NPV)':<40} {npv:>15.4f}")
print(f"{'False Positive Rate (FPR)':<40} {fpr_rate:>15.4f}")
print(f"{'False Negative Rate (FNR)':<40} {fnr_rate:>15.4f}")
print("-" * 80)
print(f"{'True Negatives (TN)':<40} {tn:>15}")
print(f"{'False Positives (FP)':<40} {fp:>15}")
print(f"{'False Negatives (FN)':<40} {fn:>15}")
print(f"{'True Positives (TP)':<40} {tp:>15}")
print("=" * 80)

print(f"\n{'='*80}")
print(f"DETAILED CLASSIFICATION REPORT")
print(f"{'='*80}\n")
print(classification_report(test_labels, test_preds, 
                          target_names=['Benign (0)', 'Malignant (1)'],
                          digits=4))

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign (0)', 'Malignant (1)'],
            yticklabels=['Benign (0)', 'Malignant (1)'],
            ax=axes[0, 0], cbar_kws={'label': 'Count'},
            annot_kws={'size': 14})
axes[0, 0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Confusion Matrix - Test Set (Counts)', fontsize=14, fontweight='bold')

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=['Benign (0)', 'Malignant (1)'],
            yticklabels=['Benign (0)', 'Malignant (1)'],
            ax=axes[0, 1], cbar_kws={'label': 'Percentage'},
            annot_kws={'size': 14})
axes[0, 1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Confusion Matrix - Test Set (Normalized)', fontsize=14, fontweight='bold')

fpr_curve, tpr_curve, thresholds = roc_curve(test_labels, test_probs)
axes[1, 0].plot(fpr_curve, tpr_curve, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
axes[1, 0].axvline(x=0.20, color='red', linestyle=':', alpha=0.7, 
                  label='Max FPR = 0.20 (min TPR = 0.80)')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
axes[1, 0].set_title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc="lower right", fontsize=10)
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(test_probs[test_labels == 0], bins=50, alpha=0.6, 
               label='Benign (0)', color='blue', edgecolor='black')
axes[1, 1].hist(test_probs[test_labels == 1], bins=50, alpha=0.6, 
               label='Malignant (1)', color='red', edgecolor='black')
axes[1, 1].axvline(x=0.5, color='green', linestyle='--', linewidth=2, 
                  label='Threshold = 0.5')
axes[1, 1].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='best', fontsize=10)
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('test_set_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

test_results_df = test_metadata.copy()
test_results_df['predicted_prob'] = test_probs
test_results_df['predicted_class'] = test_preds
test_results_df.to_csv(f'{CFG.model_name}_test_results.csv', index=False)

test_metrics = {
    'partial_auc_min_tpr_0.80': float(test_pauc),
    'roc_auc': float(roc_auc),
    'average_precision': float(avg_precision),
    'accuracy': float(accuracy),
    'matthews_correlation_coef': float(mcc),
    'cohen_kappa': float(kappa),
    'binary_metrics': {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'ppv': float(ppv),
        'npv': float(npv),
        'fpr': float(fpr_rate),
        'fnr': float(fnr_rate)
    },
    'weighted_metrics': {
        'precision': float(precision_weighted),
        'recall': float(recall_weighted),
        'f1_score': float(f1_weighted)
    },
    'macro_metrics': {
        'precision': float(precision_macro),
        'recall': float(recall_macro),
        'f1_score': float(f1_macro)
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'test_set_size': len(test_labels),
    'test_positive_samples': int(test_labels.sum()),
    'test_negative_samples': int(len(test_labels) - test_labels.sum())
}

with open(f'{CFG.model_name}_test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=4)

print(f"\n{'='*80}")
print(f"FILES SAVED:")
print(f"{'='*80}")
print(f"✓ Test predictions: {CFG.model_name}_test_results.csv")
print(f"✓ Test metrics: {CFG.model_name}_test_metrics.json")
print(f"✓ Visualization: test_set_evaluation.png")
print(f"{'='*80}")

print(f"\n{'='*80}")
print(f"EVALUATION COMPLETE!")
print(f"{'='*80}")
print(f"Best metric - Partial AUC (min_tpr=0.80): {test_pauc:.6f}")
print(f"ROC AUC: {roc_auc:.6f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 Score: {f1:.4f}")
print(f"{'='*80}\n")

