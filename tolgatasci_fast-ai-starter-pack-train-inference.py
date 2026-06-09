!unzip -qo /kaggle/input/timm-with-dependencies/timm_all -d timm-with-dependencies
!pip install --no-index --find-links timm-with-dependencies timm -q

# Kontrol
import timm
print(f'Timm: {timm.__version__}')
print('Kurulum BASARILI!')

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')

# Cihaz kontrolu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Cihaz: {device}')


class CFG:
    # ========== VERI YOLLARI ==========
    # Kaggle'da bu yollar otomatik gelir
    DATA_PATH = '/kaggle/input/rsna-breast-cancer-detection'
    IMAGE_PATH = '/kaggle/input/rsna-breast-cancer-512-pngs/images'  # onceden hazirlanmis PNG'ler
    
    # ========== MODEL AYARLARI ==========
    # Model secenekleri:
    # - 'tf_efficientnetv2_s' : Hizli, iyi sonuc (TAVSIYE)
    # - 'convnext_small'      : Daha yeni, iyi sonuc
    # - 'resnet50'            : Klasik, hizli
    MODEL_NAME = 'tf_efficientnetv2_s'
    
    # ========== EGITIM AYARLARI ==========
    EPOCHS = 5           # Kac tur egitim? (5-10 arasi iyi)
    BATCH_SIZE = 16      # GPU bellegine gore ayarla (8, 16, 32)
    IMG_SIZE = 512       # Goruntu boyutu (256, 384, 512)
    
    # ========== OGRENME HIZI ==========
    LR = 1e-4            # 0.0001 - guvenli deger
    MIN_LR = 1e-6        # Minimum ogrenme hizi
    
    # ========== CROSS VALIDATION ==========
    N_FOLDS = 4          # Kac parcaya bolunecek
    TRAIN_FOLDS = [0]    # Hangi fold'lar egitilecek (hizli test icin [0])
    
    # ========== DIGER ==========
    SEED = 42
    NUM_WORKERS = 2
    
    # ========== DENGESIZ VERI COZUMU ==========
    POS_WEIGHT = 10.0    # Kanserli orneklere kac kat fazla agirlik? (5-20 arasi)

print('Ayarlar yuklendi!')
print(f'Model: {CFG.MODEL_NAME}')
print(f'Epochs: {CFG.EPOCHS}')
print(f'Batch Size: {CFG.BATCH_SIZE}')
print(f'Image Size: {CFG.IMG_SIZE}')


def set_seed(seed=42):
    """Tekrarlanabilirlik icin seed ayarla"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CFG.SEED)

def pfbeta(labels, predictions, beta=1):
    """
    pF-beta skoru hesapla (RSNA yarisma metrigi)
    Bu metrik kanser tespitinde onemli!
    """
    y_true_count = labels.sum()
    ctp = (predictions[labels == 1]).sum()
    cfp = (predictions[labels == 0]).sum()
    
    beta_squared = beta * beta
    
    if ctp + cfp == 0:
        return 0
    
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count if y_true_count > 0 else 0
    
    if c_precision + c_recall == 0:
        return 0
    
    result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
    return result

print('Yardimci fonksiyonlar yuklendi!')


# CSV dosyasini oku
train_df = pd.read_csv(f'{CFG.DATA_PATH}/train.csv')

print(f'Toplam goruntu sayisi: {len(train_df)}')
print(f'Kanserli goruntu sayisi: {train_df.cancer.sum()}')
print(f'Normal goruntu sayisi: {len(train_df) - train_df.cancer.sum()}')
print(f'Kanser orani: %{100*train_df.cancer.mean():.2f}')

# Goruntu yolunu ekle
train_df['image_path'] = train_df.apply(
    lambda x: f"{CFG.IMAGE_PATH}/{x['patient_id']}_{x['image_id']}.png", axis=1
)

train_df.head()


# Hasta bazinda kanser durumu (ayni hastanin tum goruntulerini ayni fold'a koymak icin)
patient_cancer = train_df.groupby('patient_id')['cancer'].max().reset_index()

# Stratified K-Fold (kanser oranini koruyarak bolme)
skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

patient_cancer['fold'] = -1
for fold, (_, val_idx) in enumerate(skf.split(patient_cancer, patient_cancer['cancer'])):
    patient_cancer.loc[val_idx, 'fold'] = fold

# Ana dataframe'e fold bilgisini ekle
train_df = train_df.merge(patient_cancer[['patient_id', 'fold']], on='patient_id')

print('Fold dagilimi:')
print(train_df.groupby('fold')['cancer'].agg(['count', 'sum', 'mean']))


class MammographyDataset(Dataset):
    """
    Mamografi veri seti
    
    Bu sinif:
    - Goruntuleri diskten okur
    - Augmentation uygular (egitim icin)
    - Tensore cevirir
    """
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Goruntuyu oku
        img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Dosya bulunamazsa siyah goruntu
            img = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE), dtype=np.uint8)
        
        # 3 kanala cevir (model icin)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Augmentation
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        # Label
        label = torch.tensor(row['cancer'], dtype=torch.float32)
        
        return img, label

# Egitim icin augmentation (veri cogaltma)
train_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),       # Yatay cevirme
    A.VerticalFlip(p=0.5),         # Dikey cevirme
    A.RandomBrightnessContrast(    # Parlaklik/kontrast
        brightness_limit=0.2, 
        contrast_limit=0.2, 
        p=0.5
    ),
    A.ShiftScaleRotate(            # Kaydirma/olcekleme/dondurme
        shift_limit=0.1, 
        scale_limit=0.1, 
        rotate_limit=15, 
        p=0.5
    ),
    A.Normalize(                   # Normalizasyon
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()                   # Tensore cevir
])

# Validation icin (augmentation yok, sadece resize)
val_transforms = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

print('Dataset ve transforms hazirlandi!')


class MammographyModel(nn.Module):
    """
    Mamografi siniflandirma modeli
    
    Pretrained model + ozel head
    """
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        
        # Pretrained backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Head'i kaldir
            global_pool='avg'
        )
        
        # Ozellik boyutu
        self.num_features = self.backbone.num_features
        
        # Siniflandirma katmani
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output.squeeze(-1)

# Test
model = MammographyModel(CFG.MODEL_NAME)
print(f'Model: {CFG.MODEL_NAME}')
print(f'Ozellik boyutu: {model.num_features}')
print(f'Toplam parametre: {sum(p.numel() for p in model.parameters()):,}')


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Bir epoch egitim
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='Egitim')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Validation
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Validation')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrikleri hesapla
    auc = roc_auc_score(all_labels, all_preds) if all_labels.sum() > 0 else 0
    pf1 = pfbeta(all_labels, all_preds)
    
    return total_loss / len(loader), auc, pf1, all_preds, all_labels

print('Egitim fonksiyonlari hazirlandi!')


def train_fold(fold):
    """
    Tek bir fold icin egitim
    """
    print(f'\n{"="*50}')
    print(f'FOLD {fold} EGITIMI BASLIYOR')
    print(f'{"="*50}')
    
    # Train/Val ayir
    train_data = train_df[train_df['fold'] != fold]
    val_data = train_df[train_df['fold'] == fold]
    
    print(f'Egitim: {len(train_data)} goruntu')
    print(f'Validation: {len(val_data)} goruntu')
    
    # Dataset
    train_dataset = MammographyDataset(train_data, train_transforms)
    val_dataset = MammographyDataset(val_data, val_transforms)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CFG.BATCH_SIZE, 
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CFG.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True
    )
    
    # Model
    model = MammographyModel(CFG.MODEL_NAME).to(device)
    
    # Loss (pos_weight ile dengesiz veri cozumu)
    pos_weight = torch.tensor([CFG.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=CFG.LR, weight_decay=1e-4)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=CFG.MIN_LR)
    
    # Egitim
    best_auc = 0
    best_pf1 = 0
    history = []
    
    for epoch in range(CFG.EPOCHS):
        print(f'\nEpoch {epoch+1}/{CFG.EPOCHS}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Egitim
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_loss, val_auc, val_pf1, preds, labels = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Sonuclari kaydet
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_pf1': val_pf1
        })
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val AUC: {val_auc:.4f}')
        print(f'Val pF1: {val_pf1:.4f}')
        
        # En iyi modeli kaydet
        if val_auc > best_auc:
            best_auc = val_auc
            best_pf1 = val_pf1
            torch.save(model.state_dict(), f'best_model_fold{fold}.pt')
            print(f'*** YENi EN IYI MODEL KAYDEDILDI! AUC: {best_auc:.4f} ***')
    
    print(f'\nFold {fold} tamamlandi!')
    print(f'En iyi AUC: {best_auc:.4f}')
    print(f'En iyi pF1: {best_pf1:.4f}')
    
    return history, best_auc, best_pf1


# EGITIMI BASLAT!
all_histories = []
all_aucs = []
all_pf1s = []

for fold in CFG.TRAIN_FOLDS:
    history, best_auc, best_pf1 = train_fold(fold)
    all_histories.append(history)
    all_aucs.append(best_auc)
    all_pf1s.append(best_pf1)

print('\n' + '='*50)
print('EGITIM TAMAMLANDI!')
print('='*50)
print(f'Ortalama AUC: {np.mean(all_aucs):.4f}')
print(f'Ortalama pF1: {np.mean(all_pf1s):.4f}')


# Egitim grafiklerini ciz
if all_histories:
    history_df = pd.DataFrame(all_histories[0])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Grafigi')
    axes[0].legend()
    axes[0].grid(True)
    
    # AUC
    axes[1].plot(history_df['epoch'], history_df['val_auc'], 'g-o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Validation AUC')
    axes[1].grid(True)
    
    # pF1
    axes[2].plot(history_df['epoch'], history_df['val_pf1'], 'r-o')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('pF1')
    axes[2].set_title('Validation pF1')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    
    # Sonuclari CSV'ye kaydet
    history_df.to_csv('training_history.csv', index=False)
    print('Sonuclar kaydedildi!')

