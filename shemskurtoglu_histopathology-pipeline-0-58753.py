import os
import time
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import imgaug
import wandb  
import warnings
warnings.filterwarnings("ignore")

#I had the code rewritten by AI to make it more readable, it ended up being a bit long

#ckpt file from https://github.com/ozanciga/self-supervised-histopathology/releases/tag/tenpercent


import os
import random
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T

from scipy.stats import spearmanr
from tqdm import tqdm
import imgaug
import wandb

CONFIG = {
    "seed": 42,
    "data_path": "/kaggle/input/el-hackathon-2025",
    "output_dir": "/kaggle/working/",
    "batch_size": 32*4,
    "num_workers": 6,
    "learning_rate": 0.0018,
    "weight_decay": 3e-5,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.1,
    "num_classes": 35,
    "image_size": (162, 162),
    "patch_size": 54,
    "max_epochs": 12, # Best scoring epoch will be used for submission.csv. Epoch 4 performed the best among 10 epochs.
    "patience": 9,  # Number of epochs to wait for improvement before early stopping
    "min_delta": 0.001,  # Minimum change to qualify as improvement
    "save_best_only": True,
    "checkpoint_epochs": [], # epoch 3 performed best
    "use_wandb": False,  # Set to False if you don't want to use wandb
    "model_type": "eff",
    "mixed_precision": True,  # Use mixed precision training
    "preprocess_image": True
}



def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    imgaug.seed(seed)


def get_device():
    """Get the appropriate device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # For Apple Silicon
    else:
        return torch.device("cpu")


def spearman_rank_correlation(x, y):
    """Calculate Spearman rank correlation"""
    # Handle edge cases
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    return spearmanr(x, y)[0]

def spearman_corr(preds, targets):
    """Calculate mean Spearman correlation across all samples"""
    correlations = []
    for i in range(len(preds)):
        corr = spearman_rank_correlation(preds[i], targets[i])
        # Only count valid correlations
        if not np.isnan(corr):
            correlations.append(corr)
    return np.mean(correlations)


def check_image(img):
    # Veri tipini doÄŸru ÅŸekilde dÃ¶nÃ¼ÅŸtÃ¼r
    if img.dtype != np.uint8:
        if img.max() <= 1.0:  # [0,1] aralÄ±ÄŸÄ±nda normalize edilmiÅŸ gÃ¶rÃ¼ntÃ¼
            img = (img * 255).astype(np.uint8)
        else:  # GÃ¶rÃ¼ntÃ¼ zaten uygun aralÄ±kta
            img = img.astype(np.uint8)
    
    return img

def whitespace_filtering(img, filter_strength='moderate', preserve_tissue=True):
    """
    H&E boyalÄ± histopatoloji gÃ¶rÃ¼ntÃ¼lerinde beyaz alanlarÄ± daha hassas filtreleyen geliÅŸmiÅŸ fonksiyon
    
    Parametreler:
        img: Ä°ÅŸlenecek gÃ¶rÃ¼ntÃ¼
        filter_strength: Filtreleme ÅŸiddeti ('light', 'moderate', 'aggressive')
        preserve_tissue: Doku yapÄ±sÄ±nÄ± koruma modunu etkinleÅŸtir
        
    DÃ¶nÃ¼ÅŸ:
        filtered_img: FiltrelenmiÅŸ gÃ¶rÃ¼ntÃ¼
        mask: Doku maskesi
    """
    # GÃ¶rÃ¼ntÃ¼yÃ¼ kontrol et
    img_rgb = check_image(img)
    
    # ParlaklÄ±k ve HSV kanallarÄ±nÄ± hesapla
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Filtreleme ÅŸiddetine gÃ¶re parametreleri ayarla
    if filter_strength == 'light':
        brightness_threshold = 220
        sensitivity = 0.6
    elif filter_strength == 'moderate':
        brightness_threshold = 210
        sensitivity = 0.8
    else:  # aggressive
        brightness_threshold = 200
        sensitivity = 1.0
    
    # 1. ParlaklÄ±k bazlÄ± temel beyaz alan maskesi
    _, brightness_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Doygunluk kanalÄ± kullanarak doku tespiti (dÃ¼ÅŸÃ¼k doygunluk = beyaz alanlar)
    s_channel = hsv[:,:,1]
    _, saturation_mask = cv2.threshold(s_channel, 30 * sensitivity, 255, cv2.THRESH_BINARY)
    
    # 3. H&E boyalÄ± dokularda mor bÃ¶lgeleri koru
    lower_purple = np.array([120, 20, 20])
    upper_purple = np.array([170, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # 4. Eosin boyalÄ± bÃ¶lgeleri (pembe) koru
    lower_pink = np.array([150, 20, 100])
    upper_pink = np.array([180, 150, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # TÃ¼m doku maskelerini birleÅŸtir
    tissue_mask = cv2.bitwise_or(cv2.bitwise_or(brightness_mask, saturation_mask), 
                                 cv2.bitwise_or(purple_mask, pink_mask))
    
    # Doku koruma seÃ§eneÄŸi etkinse
    if preserve_tissue:
        # ZayÄ±f dokularÄ± da koru (daha dÃ¼ÅŸÃ¼k eÅŸikle tekrar tespit et)
        _, weak_tissue_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Renk bilgisini kullanarak zayÄ±f dokularÄ± filtrele
        texture_enhanced = cv2.addWeighted(gray, 0.5, hsv[:,:,1], 0.5, 0)
        _, texture_mask = cv2.threshold(texture_enhanced, 30, 255, cv2.THRESH_BINARY)
        
        # Ä°nce doku yapÄ±larÄ±nÄ± koru
        kernel = np.ones((3,3), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_DILATE, kernel)
        tissue_mask = cv2.bitwise_or(tissue_mask, cv2.bitwise_and(weak_tissue_mask, texture_mask))
    
    # Son maske iÅŸlemleri
    kernel = np.ones((3,3), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Maskeyi orijinal gÃ¶rÃ¼ntÃ¼ye uygula
    filtered_img = cv2.bitwise_and(img_rgb, img_rgb, mask=tissue_mask)
    
    return filtered_img, tissue_mask
    
def histopathology_tissue_mask(img):
    img_rgb = check_image(img)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # KenarlÄ±k noktalarÄ±nÄ± tespit et
    _, dots_mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    
    # HSV renk uzayÄ±na dÃ¶nÃ¼ÅŸtÃ¼r
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Hematoksilin boyalÄ± bÃ¶lgeler (mor-mavi)
    lower_purple = np.array([90, 30, 20])
    upper_purple = np.array([170, 255, 255])
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Eosin boyalÄ± bÃ¶lgeler (pembe-kÄ±rmÄ±zÄ±)
    lower_pink = np.array([150, 20, 100])
    upper_pink = np.array([180, 150, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Otsu eÅŸikleme
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    # TÃ¼m doku maskelerini birleÅŸtir
    combined_tissue_mask = cv2.bitwise_or(purple_mask, pink_mask)
    
    # Devam eden iÅŸlemler...
    kernel_small = np.ones((5, 5), np.uint8)
    dilated_dots = cv2.dilate(dots_mask, kernel_small, iterations=2)
    tissue_mask = cv2.bitwise_and(combined_tissue_mask, cv2.bitwise_not(dilated_dots))
    
    # BoÅŸluklarÄ± doldur - daha kÃ¼Ã§Ã¼k kernel boyutu
    kernel_large = np.ones((120, 120), np.uint8)  # 180 yerine daha kÃ¼Ã§Ã¼k
    filled_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
    
    # GÃ¼rÃ¼ltÃ¼yÃ¼ temizle
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Doku alanlarÄ±nÄ± seÃ§ - eÅŸik deÄŸerini dÃ¼ÅŸÃ¼rdÃ¼k
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(filled_mask)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Daha fazla doku yakalamak iÃ§in 800 yerine 500
            cv2.drawContours(final_mask, [contour], -1, 255, -1)
    
    return final_mask

    

def process_histopathology_image(img):
    """
    Histopatoloji gÃ¶rÃ¼ntÃ¼sÃ¼nÃ¼ iÅŸleyip beyaz alanlarÄ± kaldÄ±rarak dokularÄ± vurgular
    """
    img_rgb = check_image(img)
    
    # Doku maskesini oluÅŸtur
    mask = histopathology_tissue_mask(img_rgb)
    
    # Maskeyi gÃ¶rÃ¼ntÃ¼ye uygula
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    
    # CLAHE uygula (mevcut kodunuz)
    lab = cv2.cvtColor(masked_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    if l.dtype != np.uint8:
        l = l.astype(np.uint8)

    # https://pmc.ncbi.nlm.nih.gov/articles/PMC5498226/
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl, a, b))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # GeliÅŸmiÅŸ beyaz alan filtreleme
    result, _ = whitespace_filtering(result, filter_strength='light')
    
    return result



class HackhathonDataset(Dataset):
    """Dataset class for the hackathon competition"""

    def __init__(self, data_path, transform=None, mode="train"):
        self.data_path = data_path
        self.materials = []
        self.transform = transform

        # Define slides for train and validation sets
        train_slides = ["S_1","S_2","S_3", "S_4", "S_5"]
        val_slide = ["S_6"]
        test_slide = ["S_7"]
        self.mode = mode

        slide_list = train_slides if mode == "train" else val_slide if mode == "val" else test_slide

        with h5py.File(f"{self.data_path}/elucidata_ai_challenge_data.h5", "r") as h5file:
            images_group = "images/Train" if mode != "test" else "images/Test"
            spots_group = "spots/Train" if mode != "test" else "spots/Test"

            train_images = h5file[images_group]
            train_spots = h5file[spots_group]

            for slide_name in tqdm(slide_list, desc=f"Loading {mode} data"):
                if slide_name in train_images.keys():
                    image = np.array(train_images[slide_name])
                    # Ã–n iÅŸleme uygula
                    if CONFIG["preprocess_image"]:
                        image = process_histopathology_image(image)
                    spots = np.array(train_spots[slide_name])
                    df = pd.DataFrame(spots)
                    self._split_into_patches(image, df, CONFIG["patch_size"])

        print(f"{len(self.materials)} patches initialized for {mode} set")

    def __len__(self):
        return len(self.materials)

    def __getitem__(self, idx):
        image, stats = self.materials[idx]

        if self.transform:
            image = self.transform(image)

        stats = torch.tensor(stats[2:], dtype=torch.float32)

        return image, stats

    def _split_into_patches(self, arr, df, patch_size):
        """Split the image into patches centered on spot coordinates"""
        h, w, c = arr.shape

        for idx in range(len(df)):
            row = df.iloc[idx]
            x, y = int(row["x"]), int(row["y"])

            half_size = patch_size // 2

            # Ensure patches don't go outside image boundaries
            y_min = max(y - half_size, 0)
            y_max = min(y + half_size, h)
            x_min = max(x - half_size, 0)
            x_max = min(x + half_size, w)

            patch = arr[y_min:y_max, x_min:x_max, :]

            # Only include complete patches
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                # 
                self.materials.append([patch, row])
            else:
                # Handle incomplete patches by padding
                padded_patch = np.zeros((patch_size, patch_size, c), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                self.materials.append([padded_patch, row])




class SEBlock(nn.Module):
    """Squeeze-and-Excitation bloÄŸu - dikkat mekanizmasÄ±"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Ã–ncelikle SEBlock sÄ±nÄ±fÄ±nÄ± doÄŸru ÅŸekilde tanÄ±mlayalÄ±m
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HistoSwinEfficientNet(nn.Module):
    """Histopatoloji iÃ§in Ã¶zelleÅŸtirilmiÅŸ EfficientNet"""
    def __init__(self, num_classes=35, pretrained=True, freeze_layers=6):
        super(HistoSwinEfficientNet, self).__init__()
        
        # 1. EfficientNet-B3 temel modeli
        self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # EfficientNet'in Ã¶zellik boyutlarÄ±nÄ± al
        eff_features = self.efficientnet.classifier[1].in_features
        
        # 2. Swin Transformer modeli
        self.swin = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Swin'in Ã¶zellik boyutlarÄ±nÄ± al
        swin_features = self.swin.head.in_features
        
        # Orjinal sÄ±nÄ±flandÄ±rÄ±cÄ±larÄ± kaldÄ±r
        self.efficientnet.classifier = nn.Identity()
        self.swin.head = nn.Identity()
        
        # EfficientNet iÃ§in progressive unfreezing
        children = list(self.efficientnet.features.children())
        total_blocks = len(children)
        
        # Belirtilen sayÄ±da katmanÄ± dondur
        for i, child in enumerate(children):
            if i < freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
        
        # EfficientNet'e SE bloklarÄ± ekle
        for i in range(max(0, total_blocks-3), total_blocks):
            try:
                channels = self._get_out_channels(children[i])
                print(f"Blok {i}'a {channels} kanallÄ± SEBlock ekleniyor")
                
                children[i] = nn.Sequential(
                    children[i],
                    SEBlock(channels)
                )
            except Exception as e:
                print(f"UyarÄ±: Blok {i}'e SEBlock eklenirken hata oluÅŸtu: {e}")
        
        self.efficientnet.features = nn.Sequential(*children)
        
        """# Swin Transformer iÃ§in isteÄŸe baÄŸlÄ± dondurma
        for param in self.swin.parameters():
            param.requires_grad = False
            
        # Son 2 stage'i eÄŸitim iÃ§in Ã§Ã¶z
        for layer in [self.swin.features.norm, self.swin.features.stage4, self.swin.features.stage3]:
            for param in layer.parameters():
                param.requires_grad = True"""
        
        # 3. Ã–zellik birleÅŸtirme katmanÄ±
        self.feature_fusion = nn.Sequential(
            nn.Linear(eff_features + swin_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.4),
        )
        
        # 4. GeliÅŸmiÅŸ sÄ±nÄ±flandÄ±rÄ±cÄ±
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        
        # 5. Swin transformer iÃ§in normalizasyon 
        self.swin_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def _get_out_channels(self, block):
        """Bir bloktan Ã§Ä±kÄ±ÅŸ kanalÄ± sayÄ±sÄ±nÄ± gÃ¼venli ÅŸekilde belirleme"""
        # YÃ¶ntem 1: Blokun kendisinde out_channels var mÄ±?
        if hasattr(block, 'out_channels'):
            return block.out_channels
            
        # YÃ¶ntem 2: Sequential modÃ¼l ise son alt modÃ¼lÃ¼n out_channels'Ä±nÄ± kontrol et
        if isinstance(block, nn.Sequential):
            for module in reversed(list(block.modules())):
                if hasattr(module, 'out_channels'):
                    return module.out_channels
        
        # YÃ¶ntem 3: EfficientNet MBConv bloklarÄ± iÃ§in Ã¶zel durum
        if hasattr(block, 'project') and hasattr(block.project[0], 'out_channels'):
            return block.project[0].out_channels
            
        # YÃ¶ntem 4: Blok iÃ§indeki tÃ¼m modÃ¼lleri incele
        for name, module in block.named_modules():
            if isinstance(module, nn.Conv2d):
                return module.out_channels
                
        # Son Ã§are: Ã§Ä±ktÄ± ÅŸeklini analiz et
        print("Blok iÃ§in Ã§Ä±kÄ±ÅŸ kanallarÄ± belirlenemedi")
        # VarsayÄ±lan deÄŸer olarak EfficientNet-B3'Ã¼n son bloÄŸu genellikle 1536 kanallÄ±dÄ±r
        return 1536

    def forward(self, x):
        # 1. EfficientNet ile Ã¶zellik Ã§Ä±karÄ±mÄ±
        eff_features = self.efficientnet.features(x)
        eff_features = F.adaptive_avg_pool2d(eff_features, (1, 1))
        eff_features = torch.flatten(eff_features, 1)
        
        # 2. Swin Transformer iÃ§in normalizasyon ve Ã¶zellik Ã§Ä±karÄ±mÄ±
        x_swin = self.swin_norm(x)
        swin_features = self.swin(x_swin)
        
        # 3. Ã–zellikleri birleÅŸtir
        combined_features = torch.cat([eff_features, swin_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 4. SÄ±nÄ±flandÄ±rma
        output = self.classifier(fused_features)
        
        return output
    
    def unfreeze_efficientnet(self, num_layers=3):
        """EfficientNet'in belirtilen katmanlarÄ±nÄ± Ã§Ã¶z"""
        children = list(self.efficientnet.features.children())
        total_blocks = len(children)
        
        for i in range(max(0, total_blocks - num_layers), total_blocks):
            for param in children[i].parameters():
                param.requires_grad = True
        
        return self
    
    def unfreeze_swin_stage(self, stage_num=3):
        """Swin Transformer'Ä±n belirli bir stage'ini eÄŸitim iÃ§in aÃ§"""
        stage_name = f"stage{stage_num}"
        if hasattr(self.swin.features, stage_name):
            stage = getattr(self.swin.features, stage_name)
            for param in stage.parameters():
                param.requires_grad = True
        
        return self


class DifferentiableSpearmanLoss(nn.Module):
    """Differentiable approximation of Spearman correlation loss"""

    def __init__(self, regularization_strength=1.0):
        super().__init__()
        self.regularization_strength = regularization_strength

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Calculate soft ranks
        pred_rank = self._soft_rank(y_pred)
        true_rank = self._soft_rank(y_true)

        # Normalize ranks
        pred_rank = F.normalize(pred_rank, dim=1)
        true_rank = F.normalize(true_rank, dim=1)

        # Calculate correlation
        spearman = torch.sum(pred_rank * true_rank, dim=1)
        return 1 - spearman.mean()

    def _soft_rank(self, x, regularization_strength=None):
        if regularization_strength is None:
            regularization_strength = self.regularization_strength

        x = x.unsqueeze(-1)  # [batch, n, 1]
        diff = x - x.transpose(-1, -2)  # [batch, n, n]
        P = torch.sigmoid(-regularization_strength * diff)  # pairwise comparisons
        ranks = P.sum(dim=-1)  # approximate ranks
        return ranks


class CombinedLoss(nn.Module):
    """Combined loss function using L1 and Spearman correlation with linear alpha adjustment"""

    def __init__(self, initial_alpha=0.9, later_alpha=0.5, switch_epoch=5, regularization_strength=1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.spearman = DifferentiableSpearmanLoss(regularization_strength)
        self.initial_alpha = initial_alpha  # BaÅŸlangÄ±Ã§ alpha deÄŸeri
        self.later_alpha = later_alpha      # Hedef alpha deÄŸeri
        self.switch_epoch = switch_epoch    # GeÃ§iÅŸin tamamlanacaÄŸÄ± epoch
        self.current_epoch = 0              # Mevcut epoch sayacÄ±
        
    def forward(self, y_pred, y_true):
        l1_loss = self.l1(y_pred, y_true)
        spearman_loss = self.spearman(y_pred, y_true)
        
        # DoÄŸrusal kademeli alpha deÄŸiÅŸimi
        if self.current_epoch < self.switch_epoch:
            # DoÄŸrusal interpolasyon: 0. epoch'ta initial_alpha, switch_epoch'ta later_alpha
            progress = self.current_epoch / self.switch_epoch  # 0'dan 1'e
            alpha = self.initial_alpha - (self.initial_alpha - self.later_alpha) * progress
        else:
            # switch_epoch sonrasÄ± sabit deÄŸer
            alpha = self.later_alpha
        
        combined_loss = l1_loss * alpha + spearman_loss * (1 - alpha*.5)
        
        # EÄŸitim sÄ±rasÄ±nda ara sÄ±ra alpha deÄŸerini ve kayÄ±p bileÅŸenlerini yazdÄ±r
        if not hasattr(self, 'print_counter'):
            self.print_counter = 0
        
        self.print_counter += 1
        if self.print_counter % 100 == 0:  # Her 100 adÄ±mda bir yazdÄ±r
            print(f"Epoch: {self.current_epoch}, Alpha: {alpha:.4f}, "
                  f"L1Loss: {l1_loss.item():.4f}, SpearmanLoss: {spearman_loss.item():.4f}")
            self.print_counter = 0
        
        return combined_loss#, l1_loss, spearman_loss
    
    def update_epoch(self, epoch):
        """Her epoch baÅŸÄ±nda Ã§aÄŸrÄ±larak mevcut epoch bilgisini gÃ¼nceller"""
        self.current_epoch = epoch
        print(f"\nEpoch {epoch+1} baÅŸladÄ±. GeÃ§erli alpha aralÄ±ÄŸÄ±: "
              f"{self.initial_alpha:.4f} - {self.later_alpha:.4f}, "
              f"GeÃ§iÅŸ tamamlanma epoch'u: {self.switch_epoch}")




def get_transforms():
    """Get data transformations for training and validation"""
    train_transform = T.Compose([
        T.ToTensor(),
        T.Resize(CONFIG["image_size"]),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Resize(CONFIG["image_size"]),
    ])

    return train_transform, val_transform


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)



def train_one_epoch(model, dataloader, loss_fn, optimizer, device, scaler=None):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc="Training")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        if scaler:  # Using mixed precision
            with torch.cuda.amp.autocast():

                outputs = model(images)
                loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        # Store predictions and labels for metrics calculation
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate metrics
    avg_loss = epoch_loss / len(dataloader)
    spearman_score = spearman_corr(all_preds, all_labels)

    return avg_loss, spearman_score, all_preds, all_labels
       

def validate(model, dataloader, loss_fn, device):
    """Validate model on validation set"""
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

            # Store predictions and labels for metrics calculation
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = val_loss / len(dataloader)
    spearman_score = spearman_corr(all_preds, all_labels)

    return avg_loss, spearman_score, all_preds, all_labels


def get_tta_transforms():
    """Get test-time augmentation transformations"""
    tta_transforms = [
        # Original image
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
        ]),
        # Horizontal flip
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.RandomHorizontalFlip(p=1.0),
        ]),
        # Vertical flip
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.RandomVerticalFlip(p=1.0),
        ]),
        # 90 degree rotation
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.RandomRotation(degrees=(90, 90)),
        ]),
        # 180 degree rotation
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.RandomRotation(degrees=(180, 180)),
        ]),
        # Color jitter
        T.Compose([
            T.ToTensor(),
            T.Resize(CONFIG["image_size"]),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]),

    ]
    return tta_transforms



def predict_test_set_with_tta(model, data_path, device):
    """Generate predictions for the test set using Test Time Augmentation"""
    tta_transforms = get_tta_transforms()

    with h5py.File(f"{data_path}/elucidata_ai_challenge_data.h5", "r") as f:
        test_spots = f["spots/Test"]
        test_images = f["images/Test"]
        sample = 'S_7'  # Test sample
        image = np.array(test_images[sample])
        # Ã–n iÅŸleme uygula
        if CONFIG["preprocess_image"]:
            image = process_histopathology_image(image)
            
        spots = np.array(test_spots[sample])
        x, y = spots["x"], spots["y"]

        outputs = []

        with torch.inference_mode():
            model.eval()
            patch_size = CONFIG["patch_size"]
            for x_, y_ in tqdm(zip(x, y), desc="Generating predictions with TTA", total=len(x)):
                half_size = patch_size // 2
                # Ensure indices are within bounds
                y_min = max(y_ - half_size, 0)
                y_max = min(y_ + half_size, image.shape[0])
                x_min = max(x_ - half_size, 0)
                x_max = min(x_ + half_size, image.shape[1])
  
                patch = image[y_min:y_max, x_min:x_max, :]

                # Handle incomplete patches
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded_patch

                # Apply TTA and get predictions for each augmentation
                patch_predictions = []
                for transform in tta_transforms:
                    patch_tensor = transform(patch)
                    patch_tensor = patch_tensor.to(device)
                    with torch.no_grad():
                     output = model(patch_tensor.unsqueeze(0)).cpu().numpy()
                     patch_predictions.append(output[0])

                # Average predictions from all augmentations
                avg_prediction = np.mean(patch_predictions, axis=0)
                outputs.append(avg_prediction)

    return np.array(outputs), x, y



def save_submission(predictions):
    """Save predictions to submission file"""
    example_df = pd.read_csv("/kaggle/input/sample-submission/submission (1).csv")
    ID = example_df["ID"]
    output_df = pd.DataFrame(predictions)
    submission_df = pd.concat([ID, output_df], axis=1)
    submission_df.columns = example_df.columns

    output_file = "submission.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"Saved submission to {output_file}")

    return output_file

def save_submission(predictions, data_path, epoch, model_name):
    """Save predictions to submission file"""
    example_df = pd.read_csv(f"/kaggle/input/sample-submission/submission (1).csv")
    ID = example_df["ID"]
    output_df = pd.DataFrame(predictions)
    submission_df = pd.concat([ID, output_df], axis=1)
    submission_df.columns = example_df.columns

    output_file = "submission.csv"
    submission_df.to_csv(output_file, index=False)
    print(f"Saved submission to {output_file}")

    return output_file


def main():

    set_seed(CONFIG["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # Create data transformations, datasets, and dataloaders as before...
    train_transform, val_transform = get_transforms()
    train_dataset = HackhathonDataset(CONFIG["data_path"], transform=train_transform, mode="train")
    val_dataset = HackhathonDataset(CONFIG["data_path"], transform=val_transform, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    # Create model, loss function, optimizer, and scheduler as before...
    #model = create_model(CONFIG["model_type"], CONFIG["num_classes"])
    model = HistoSwinEfficientNet(num_classes=35, pretrained=True, freeze_layers=6)
    model = model.to(device)
    loss_fn = CombinedLoss(initial_alpha=0.9, later_alpha=0.8, switch_epoch=5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG["scheduler_gamma"],
        patience=3,
    )

    # Initialize mixed precision scaler if needed
    scaler = torch.cuda.amp.GradScaler() if CONFIG["mixed_precision"] and device.type == "cuda" else None

    # Initialize wandb if enabled
    if CONFIG["use_wandb"]:
        wandb.init(
            project="hackathon-gene-expression",
            config=CONFIG,
            name=f"{CONFIG['model_type']}_run"
        )
        wandb.watch(model)

    # Training loop with early stopping
    best_val_spearman = -1
    no_improvement_count = 0
    best_model_path = f"{CONFIG['output_dir']}/best_{CONFIG['model_type']}_model.pt"

    for epoch in range(CONFIG["max_epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['max_epochs']}")
        loss_fn.update_epoch(epoch)

        # Train one epoch
        train_loss, train_spearman, _, _ = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler
        )

        # Validate
        val_loss, val_spearman, _, _ = validate(
            model, val_loader, loss_fn, device
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Log metrics
        metrics = {
            "train_loss": train_loss,
            "train_spearman": train_spearman,
            "val_loss": val_loss,
            "val_spearman": val_spearman,
            "learning_rate": optimizer.param_groups[0]['lr']
        }

        print(f"Train Loss: {train_loss:.4f}, Train Spearman: {train_spearman:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Spearman: {val_spearman:.4f}")

        if CONFIG["use_wandb"]:
            wandb.log(metrics)

        # Check if this is the best model so far
        improved = val_spearman > best_val_spearman + CONFIG["min_delta"]

        if improved:
            best_val_spearman = val_spearman
            no_improvement_count = 0

            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_spearman': val_spearman,
            }, best_model_path)

            print(f"Saved best model with validation Spearman: {val_spearman:.4f}")

            # Generate test predictions with the best model so far
            test_preds, x, y = predict_test_set_with_tta(model, CONFIG["data_path"], device)

            # Save submission file
            submission_file = save_submission(
                test_preds, CONFIG["data_path"], f"best_epoch_{epoch}", CONFIG["model_type"]
            )

            if CONFIG["use_wandb"]:
                wandb.save(submission_file)
        else:
            no_improvement_count += 1

        # Generate and save predictions at checkpoint epochs regardless of improvement
        if epoch in CONFIG["checkpoint_epochs"]:
            # Generate test predictions with Test Time Augmentation
            test_preds, _, _ = predict_test_set_with_tta(model, CONFIG["data_path"], device)

            # Save submission file
            submission_file = save_submission(
                test_preds, CONFIG["data_path"], epoch, CONFIG["model_type"]
            )

            if CONFIG["use_wandb"]:
                wandb.save(submission_file)

        # Check for early stopping
        if no_improvement_count >= CONFIG["patience"]:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation Spearman: {checkpoint['val_spearman']:.4f}")

    # Final evaluation and prediction
    final_preds, _, _ = predict_test_set_with_tta(model, CONFIG["data_path"], device)
    final_submission = save_submission(
        final_preds, CONFIG["data_path"], "final", CONFIG["model_type"]
    )

    if CONFIG["use_wandb"]:
        wandb.save(final_submission)
        wandb.finish()

    print("Training completed!")


main()

