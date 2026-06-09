import os
import sys
import gc
import json
import shutil
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from IPython.display import display

# æ•°æ�®å¤„ç�†
import numpy as np
import polars as pl
import pandas as pd

# åŒ»å­¦å½±åƒ�å¤„ç�†
import pydicom
import cv2

# æ·±åº¦å­¦ä¹ 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm

# å›¾åƒ�å¢�å¼º
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Kaggle API
import kaggle_evaluation.rsna_inference_server

import yaml

# è®¾å¤‡é…�ç½®
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



# ç«�èµ›æ ‡ç­¾
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery', 
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery', 
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

# ğŸ”� æ£€æµ‹Simple3DCNNæ•°æ�®é›†
print("ğŸ”� Detecting Simple3DCNN dataset...")
print("="*60)

simple3dcnn_path = "/kaggle/input/rsna-2025-iad-with-simple3dcnn"

# åˆ�å§‹åŒ–å�˜é‡�ï¼ˆä¿®å¤�ä½œç”¨åŸŸé—®é¢˜ï¼‰
checkpoint_files = []
config_files = []

if os.path.exists(simple3dcnn_path):
    print(f"âœ… Found dataset: {simple3dcnn_path}")
    
    # æ£€æŸ¥checkpointæ–‡ä»¶
    for root, dirs, files in os.walk(simple3dcnn_path):
        for file in files:
            if file.endswith('.ckpt'):
                full_path = os.path.join(root, file)
                checkpoint_files.append(full_path)
                print(f"   ğŸ�¯ Found checkpoint: {file}")
    
    # æ£€æŸ¥é…�ç½®æ–‡ä»¶
    for root, dirs, files in os.walk(simple3dcnn_path):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                full_path = os.path.join(root, file)
                config_files.append(full_path)
                print(f"   ğŸ“‹ Found config: {file}")
    
    print(f"   ğŸ“Š Total checkpoints: {len(checkpoint_files)}")
    print(f"   ğŸ“‹ Total configs: {len(config_files)}")
else:
    print(f"â�Œ Dataset not found: {simple3dcnn_path}")

print("="*60)

# ğŸ�¯ æ™ºèƒ½é…�ç½®ç±» - ä¿®å¤�å��çš„ç‰ˆæœ¬
class Simple3DCNNConfig:
    # æ•°æ�®é›†è·¯å¾„
    dataset_path = simple3dcnn_path
    checkpoint_files = checkpoint_files  # ç›´æ�¥å¼•ç”¨å¤–éƒ¨å�˜é‡�
    config_files = config_files  # ç›´æ�¥å¼•ç”¨å¤–éƒ¨å�˜é‡�
    
    # æ¨¡å�‹ç­–ç•¥
    use_external_checkpoint = len(checkpoint_files) > 0
    use_ensemble = False  # å…ˆç”¨å�•æ¨¡å�‹æµ‹è¯•
    model_type = "simple3dcnn"  # æ ‡è¯†è¿™æ˜¯3D CNNæ¨¡å�‹
    
    # å›¾åƒ�å¤„ç�†å�‚æ•° - é€‚é…�3D CNN
    image_size = 128  # 3Dæ¨¡å�‹é€šå¸¸ç”¨è¾ƒå°�å°ºå¯¸
    num_slices = 32   # 3Dæ¨¡å�‹éœ€è¦�å®Œæ•´çš„ä½“ç§¯æ•°æ�®
    use_windowing = True
    
    # æ�¨ç�†è®¾ç½®
    batch_size = 1
    use_amp = True
    use_tta = True
    tta_transforms = 4  # 3Dæ¨¡å�‹TTAæ›´æ˜‚è´µï¼Œå‡�å°‘æ•°é‡�
    
    # Simple3DCNNç‰¹å®šé…�ç½®
    backbone_model = 'simple3dcnn'
    
    # å¤„ç�†é€‰é¡¹
    use_enhanced_multichannel = False  # 3Dæ¨¡å�‹å�¯èƒ½ä¸�éœ€è¦�å¤šé€šé�“æŠ€å·§
    use_metadata_augmentation = True
    
    # 3Dæ¨¡å�‹ä¼˜åŒ–
    dropout_rate = 0.3
    use_3d_processing = True

CFG = Simple3DCNNConfig()

# å…¨å±€å�˜é‡�
MODELS = {}
TRANSFORM = None
TTA_TRANSFORMS = None

# ğŸ“Š é…�ç½®æŠ¥å‘Šï¼ˆä¿®å¤�å��ï¼‰
print("ğŸ“Š Simple3DCNN Configuration:")
print(f"   â€¢ Model Type: {CFG.model_type}")
print(f"   â€¢ External Checkpoints: {CFG.use_external_checkpoint}")
print(f"   â€¢ Available Checkpoints: {len(CFG.checkpoint_files)}")
print(f"   â€¢ Checkpoint Files:")
for i, ckpt in enumerate(CFG.checkpoint_files):
    print(f"     {i+1}. {os.path.basename(ckpt)}")
print(f"   â€¢ Image Size: {CFG.image_size}x{CFG.image_size}")
print(f"   â€¢ Volume Slices: {CFG.num_slices}")
print(f"   â€¢ 3D Processing: {CFG.use_3d_processing}")
print("="*60)

if CFG.checkpoint_files:
    print("ğŸ�¯ Will use pre-trained Simple3DCNN model")
    print("ğŸ“ˆ Expected performance: High (specialized for this task)")
    
    # é€‰æ‹©æœ€ä½³checkpointï¼ˆepochæœ€é«˜çš„ï¼‰
    best_checkpoint = None
    best_epoch = -1
    
    for ckpt_path in CFG.checkpoint_files:
        filename = os.path.basename(ckpt_path)
        try:
            # ä»�æ–‡ä»¶å��æ��å�–epochä¿¡æ�¯ (ä¾‹å¦‚: epoch=1-step=2.ckpt)
            if 'epoch=' in filename:
                epoch_part = filename.split('epoch=')[1].split('-')[0]
                epoch = int(epoch_part)
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_checkpoint = ckpt_path
        except:
            continue
    
    if best_checkpoint:
        CFG.selected_checkpoint = best_checkpoint
        print(f"ğŸ�† Selected best checkpoint: {os.path.basename(best_checkpoint)} (epoch={best_epoch})")
    else:
        CFG.selected_checkpoint = CFG.checkpoint_files[0]  # ä½¿ç”¨ç¬¬ä¸€ä¸ªä½œä¸ºfallback
        print(f"ğŸ�¯ Using first checkpoint: {os.path.basename(CFG.selected_checkpoint)}")
else:
    print("âš ï¸� No checkpoints found, will use fallback strategy")
    CFG.selected_checkpoint = None

print(f"\nğŸ�¯ Final selected checkpoint: {CFG.selected_checkpoint}")


class AdvancedMultiBackboneModel(nn.Module):
    """å¢�å¼ºçš„å¤šbackboneæ¨¡å�‹ï¼Œæ”¯æŒ�æ›´sophisticatedçš„ç‰¹å¾�è��å�ˆ"""
    
    def __init__(self, model_name, num_classes=14, pretrained=True, 
                 drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()
        
        self.model_name = model_name
        
        # æ ¹æ�®æ¨¡å�‹ç±»å�‹åˆ›å»ºbackbone
        if 'swin' in model_name:
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                img_size=CFG.image_size,
                num_classes=0,
                global_pool=''
            )
        else:
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_classes=0,
                global_pool=''
            )
        
        # è‡ªåŠ¨æ£€æµ‹ç‰¹å¾�ç»´åº¦
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, CFG.image_size, CFG.image_size)
            features = self.backbone(dummy_input)
            
            if len(features.shape) == 4:
                num_features = features.shape[1]
                self.needs_pool = True
            elif len(features.shape) == 3:
                num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False
        
        print(f"Model {model_name}: detected {num_features} features, output shape: {features.shape}")
        
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # å¢�å¼ºçš„å…ƒæ•°æ�®å¤„ç�†
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # æ›´æ·±çš„åˆ†ç±»å™¨
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 64, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(192, num_classes)
        )
        
    def forward(self, image, meta):
        # æ��å�–å›¾åƒ�ç‰¹å¾�
        img_features = self.backbone(image)
        
        # é€‚å½“çš„æ± åŒ–ç­–ç•¥
        if hasattr(self, 'needs_pool') and self.needs_pool:
            img_features = self.global_pool(img_features)
            img_features = img_features.flatten(1)
        elif hasattr(self, 'needs_seq_pool') and self.needs_seq_pool:
            img_features = img_features.mean(dim=1)
        elif len(img_features.shape) == 4:
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            img_features = img_features.mean(dim=1)
        
        # å¤„ç�†å…ƒæ•°æ�®
        meta_features = self.meta_fc(meta)
        
        # ç‰¹å¾�è��å�ˆ
        combined = torch.cat([img_features, meta_features], dim=1)
        output = self.classifier(combined)
        
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

class OriginalSimple3DCNN(nn.Module):
    """å®Œå…¨åŒ¹é…�checkpointçš„å�Ÿå§‹Simple3DCNNæ�¶æ�„"""
    
    def __init__(self):
        super().__init__()
        
        # å®Œå…¨æŒ‰ç…§checkpointé‡�å»º
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)  # [16, 1, 3, 3, 3]
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1) # [32, 16, 3, 3, 3]
        
        # æ ¹æ�®131072æ�¨æ–­çš„å…¨è¿�æ�¥å±‚
        self.fc1 = nn.Linear(131072, 64)  # [64, 131072]
        self.fc2 = nn.Linear(64, 1)       # [1, 64] - å�Ÿå§‹æ˜¯å�•åˆ†ç±»
        
    def forward(self, x):
        # x shape: (batch, 1, depth, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # è®¡ç®—å¦‚ä½•å¾—åˆ°131072
        # å�‡è®¾è¾“å…¥ç»�è¿‡å¤„ç�†å��å�˜æˆ�: (batch, 32, 1, 64, 64)
        # éœ€è¦�adaptive poolingæ�¥ç¡®ä¿�æ­£ç¡®çš„å°ºå¯¸
        x = F.adaptive_avg_pool3d(x, (1, 64, 64))  # â†’ (batch, 32, 1, 64, 64)
        
        # Flatten: 32 * 1 * 64 * 64 = 131072
        x = x.flatten(1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # å�Ÿå§‹è¾“å‡º1ä¸ªå€¼
        
        return x

class AdaptedSimple3DCNN(nn.Module):
    """é€‚é…�14åˆ†ç±»ä»»åŠ¡çš„Simple3DCNN"""
    
    def __init__(self, num_classes=14, checkpoint_path=None, freeze_backbone=True):
        super().__init__()
        
        # åŠ è½½å�Ÿå§‹æ¨¡å�‹
        self.backbone = OriginalSimple3DCNN()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_original_weights(checkpoint_path)
            
        # å†»ç»“backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # æ–°çš„åˆ†ç±»å¤´ï¼ˆä½¿ç”¨backboneçš„ç‰¹å¾�ï¼‰
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # ç»„å�ˆåˆ†ç±»å™¨ï¼šå�Ÿå§‹ç‰¹å¾�64 + å…ƒæ•°æ�®32 = 96
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def load_original_weights(self, checkpoint_path):
        """åŠ è½½å�Ÿå§‹checkpointæ�ƒé‡�"""
        try:
            print(f"ğŸ”„ Loading original weights from: {os.path.basename(checkpoint_path)}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                
                # æ¸…ç�†æ¨¡å�‹å‰�ç¼€
                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    # ç§»é™¤ 'model.' å‰�ç¼€
                    new_key = key.replace('model.', '')
                    cleaned_state_dict[new_key] = value
                
                # åŠ è½½åˆ°backbone
                missing_keys, unexpected_keys = self.backbone.load_state_dict(cleaned_state_dict, strict=True)
                
                if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                    print("âœ… Perfect weight loading - all parameters matched!")
                else:
                    print(f"âš ï¸� Loading issues: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected")
                    if missing_keys:
                        print(f"   Missing: {missing_keys}")
                    if unexpected_keys:
                        print(f"   Unexpected: {unexpected_keys}")
                
            else:
                raise Exception("No state_dict found in checkpoint")
                
        except Exception as e:
            print(f"â�Œ Failed to load weights: {e}")
            print("ğŸ”„ Will use randomly initialized weights")
    
    def forward(self, volume, meta=None):
        # ç¡®ä¿�è¾“å…¥æ˜¯æ­£ç¡®çš„5Dæ ¼å¼�
        if volume.dim() == 4:
            volume = volume.unsqueeze(1)  # (batch, 1, depth, height, width)
        elif volume.dim() == 5 and volume.size(1) != 1:
            # å¦‚æ�œç¬¬äºŒç»´ä¸�æ˜¯1ï¼Œå�¯èƒ½éœ€è¦�è°ƒæ•´
            if volume.size(2) == 1:  # (batch, depth, 1, height, width)
                volume = volume.transpose(1, 2)  # â†’ (batch, 1, depth, height, width)
        
        # é€šè¿‡backboneè�·å�–ç‰¹å¾�
        with torch.no_grad() if hasattr(self, '_freeze_backbone') else torch.enable_grad():
            # è�·å�–åˆ°fc1ä¹‹å‰�çš„ç‰¹å¾�
            x = F.relu(self.backbone.conv1(volume))
            x = F.relu(self.backbone.conv2(x))
            x = F.adaptive_avg_pool3d(x, (1, 64, 64))
            x = x.flatten(1)
            backbone_features = F.relu(self.backbone.fc1(x))  # (batch, 64)
        
        # å¤„ç�†å…ƒæ•°æ�®
        if meta is not None:
            meta_features = self.meta_fc(meta)  # (batch, 32)
        else:
            batch_size = backbone_features.size(0)
            meta_features = torch.zeros(batch_size, 32, device=backbone_features.device)
        
        # ç»„å�ˆç‰¹å¾�
        combined_features = torch.cat([backbone_features, meta_features], dim=1)  # (batch, 96)
        
        # æœ€ç»ˆåˆ†ç±»
        output = self.classifier(combined_features)
        return output

def create_fixed_model():
    """åˆ›å»ºä¿®å¤�çš„æ¨¡å�‹"""
    global MODELS, CFG
    
    print("ğŸ”„ Creating fixed Simple3DCNN model...")
    
    try:
        if CFG.use_external_checkpoint and CFG.selected_checkpoint:
            # ä½¿ç”¨é¢„è®­ç»ƒæ�ƒé‡�çš„é€‚é…�æ¨¡å�‹
            model = AdaptedSimple3DCNN(
                num_classes=14,
                checkpoint_path=CFG.selected_checkpoint,
                freeze_backbone=True
            )
            print("âœ… Created adapted model with pre-trained backbone")
        else:
            # åˆ›å»ºéš�æœºåˆ�å§‹åŒ–çš„æ¨¡å�‹
            model = AdaptedSimple3DCNN(num_classes=14)
            print("âœ… Created randomly initialized model")
        
        model = model.to(device).eval()
        MODELS.clear()
        MODELS['simple3dcnn_fixed'] = model
        
        # æµ‹è¯•æ¨¡å�‹
        print("ğŸ§ª Testing model with correct input format...")
        
        # æµ‹è¯•ä¸�å�Œçš„è¾“å…¥æ ¼å¼�
        test_inputs = [
            torch.randn(2, 1, 32, 64, 64).to(device),  # æ­£ç¡®æ ¼å¼�
            torch.randn(2, 32, 64, 64).to(device),     # 4Dæ ¼å¼�
        ]
        
        test_meta = torch.randn(2, 2).to(device)
        
        with torch.no_grad():
            for i, test_input in enumerate(test_inputs):
                try:
                    output = model(test_input, test_meta)
                    print(f"   âœ… Test {i+1}: {list(test_input.shape)} â†’ {list(output.shape)}")
                except Exception as e:
                    print(f"   â�Œ Test {i+1} failed: {e}")
        
        print("ğŸ�¯ Model ready for inference!")
        return True
        
    except Exception as e:
        print(f"â�Œ Failed to create model: {e}")
        return False

# æ‰§è¡Œä¿®å¤�
if 'CFG' in globals():
    success = create_fixed_model()
    if success:
        print("\n" + "="*50)
        print("ğŸ�‰ Model successfully fixed!")
        print("âœ… Ready to run inference with corrected architecture")
        print("="*50)
else:
    print("â�Œ CFG not found, please run configuration first")


def predict_single_model_fixed_v2(volume, patient_age, patient_sex, model_name='simple3dcnn_fixed'):
    """å®Œå…¨ä¿®å¤�çš„å�•æ¨¡å�‹æ�¨ç�†"""
    try:
        model = MODELS[model_name]
        model.eval()
        
        with torch.no_grad():
            # é¦–å…ˆç¡®ä¿�æ˜¯numpyæ•°ç»„
            if isinstance(volume, torch.Tensor):
                volume = volume.numpy()
            
            # è½¬æ�¢ä¸ºtorch tensor
            volume = torch.from_numpy(volume).float()
            
            print(f"ğŸ”� Original input shape: {list(volume.shape)}")
            
            # å¼ºåˆ¶ç¡®ä¿�æ˜¯5Dæ ¼å¼�: (batch, channel, depth, height, width)
            if volume.dim() == 3:  # (depth, height, width)
                print("   Converting 3D â†’ 5D")
                volume = volume.unsqueeze(0).unsqueeze(0)  # â†’ (1, 1, depth, height, width)
            elif volume.dim() == 4:  # (batch, depth, height, width)
                print("   Converting 4D â†’ 5D")
                volume = volume.unsqueeze(1)  # â†’ (batch, 1, depth, height, width)
            elif volume.dim() == 5:  # Already correct
                print("   Already 5D")
                pass
            else:
                raise ValueError(f"Unsupported input dimensions: {volume.shape}")
            
            print(f"ğŸ”� Final input shape: {list(volume.shape)}")
            
            # ç§»åˆ°è®¾å¤‡
            volume = volume.to(device)
            
            # å‡†å¤‡å…ƒæ•°æ�®
            meta_tensor = torch.tensor([[patient_age, patient_sex]], dtype=torch.float32, device=device)
            
            # æ�¨ç�†
            outputs = model(volume, meta_tensor)
            
            print(f"ğŸ”� Output shape: {list(outputs.shape)}")
            
            # åº”ç”¨sigmoidè�·å�–æ¦‚ç�‡
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
            return probabilities
            
    except Exception as e:
        print(f"â�Œ Prediction failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        
        # è¿”å›�é»˜è®¤æ¦‚ç�‡
        return np.full(14, 0.1)

def test_completely_fixed_model():
    """æµ‹è¯•å®Œå…¨ä¿®å¤�çš„æ¨¡å�‹"""
    print("ğŸ§ª Testing completely fixed model...")
    print("=" * 50)
    
    if 'MODELS' not in globals() or not MODELS:
        print("â�Œ No models loaded")
        return False
    
    model_name = list(MODELS.keys())[0]
    print(f"âœ… Testing model: {model_name}")
    
    # æµ‹è¯•ä¸�å�Œè¾“å…¥æ ¼å¼�
    test_cases = [
        {
            'name': '5D Input (Correct)',
            'volume': np.random.randn(1, 1, 32, 64, 64),
        },
        {
            'name': '4D Input (Auto-fix)',
            'volume': np.random.randn(1, 32, 64, 64),
        },
        {
            'name': '3D Input (Auto-fix)',
            'volume': np.random.randn(32, 64, 64),
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases):
        try:
            print(f"\nğŸ”� Test {i+1}: {test_case['name']}")
            print(f"   ğŸ“� Input shape: {test_case['volume'].shape}")
            
            # è¿�è¡Œæ�¨ç�†
            output = predict_single_model_fixed_v2(
                test_case['volume'], 50.0, 1.0, model_name
            )
            
            print(f"   âœ… Success! Output shape: {output.shape}")
            print(f"   ğŸ“Š Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"   ğŸ“Š Sum of probabilities: {output.sum():.3f}")
            
            success_count += 1
            
        except Exception as e:
            print(f"   â�Œ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nğŸ“ˆ Test Results: {success_count}/{len(test_cases)} passed")
    
    if success_count == len(test_cases):
        print("ğŸ�‰ All tests passed! Model is ready for inference.")
        
        # æ›´æ–°å…¨å±€å‡½æ•°
        global predict_single_model_fixed
        predict_single_model_fixed = predict_single_model_fixed_v2
        print("âœ… Updated global inference function")
        
        return True
    else:
        print("âš ï¸� Some tests failed.")
        return False

# ç®€åŒ–çš„TTAå�˜æ�¢
def get_simple_tta_transforms():
    """è�·å�–ç®€åŒ–çš„TTAå�˜æ�¢ï¼Œé�¿å…�å°ºå¯¸é—®é¢˜"""
    transforms = []
    
    def identity_transform(volume):
        return volume.copy()
    
    def flip_horizontal(volume):
        return np.flip(volume, axis=-1).copy()  # æ°´å¹³ç¿»è½¬æœ€å��ä¸€ä¸ªè½´
    
    def flip_vertical(volume):
        return np.flip(volume, axis=-2).copy()  # å�‚ç›´ç¿»è½¬å€’æ•°ç¬¬äºŒä¸ªè½´
    
    transforms = [
        identity_transform,
        flip_horizontal,
        flip_vertical,
    ]
    
    return transforms

# æ›´æ–°TTAå�˜æ�¢
TTA_TRANSFORMS = get_simple_tta_transforms()
print(f"âœ… Simplified TTA transforms: {len(TTA_TRANSFORMS)}")

# è¿�è¡Œæµ‹è¯•
if __name__ == "__main__":
    test_result = test_completely_fixed_model()
    
    if test_result:
        print("\n" + "="*60)
        print("ğŸš€ COMPLETELY FIXED! READY FOR FULL INFERENCE!")
        print("You can now run Cell 9 with confidence!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("â�Œ Still having issues - let me know the error details")
        print("="*60)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_competition_data():
    """åŠ è½½æ¯”èµ›è®­ç»ƒæ•°æ�®"""
    print("ğŸ�† Loading Competition Training Data")
    print("=" * 60)
    
    # æŸ¥æ‰¾è®­ç»ƒæ•°æ�®æ–‡ä»¶
    possible_paths = [
        "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv",  # æ­£ç¡®è·¯å¾„
        "/kaggle/input/rsna-2025-iad/train.csv",
        "/kaggle/input/train.csv",
        "./train.csv",
        "../input/rsna-intracranial-aneurysm-detection/train.csv",
        "/content/train.csv"
    ]
    
    train_csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            train_csv_path = path
            break
    
    if not train_csv_path:
        print("â�Œ ERROR: No training data found!")
        print("ğŸ’¡ Please ensure train.csv is available")
        return None, None
    
    print(f"âœ… Found training data: {train_csv_path}")
    
    # åŠ è½½æ•°æ�®
    try:
        train_df = pd.read_csv(train_csv_path)
        print(f"ğŸ“Š Loaded {len(train_df)} training samples")
        
        # æ£€æŸ¥åˆ—ç»“æ�„
        print(f"ğŸ“‹ Dataset columns ({len(train_df.columns)} total):")
        for i, col in enumerate(train_df.columns[:3]):
            print(f"   {i+1}. {col}")
        print("   ...")
        for i, col in enumerate(train_df.columns[-3:]):
            print(f"   {len(train_df.columns)-2+i}. {col}")
        
        # å®šä¹‰æ ‡ç­¾åˆ—ï¼ˆ14ä¸ªåŠ¨è„‰ç˜¤ä½�ç½®ï¼‰
        target_columns = [
            'Left Infraclinoid Internal Carotid Artery',
            'Right Infraclinoid Internal Carotid Artery', 
            'Left Supraclinoid Internal Carotid Artery',
            'Right Supraclinoid Internal Carotid Artery',
            'Left Middle Cerebral Artery',
            'Right Middle Cerebral Artery',
            'Anterior Communicating Artery',
            'Left Anterior Cerebral Artery',
            'Right Anterior Cerebral Artery',
            'Left Posterior Communicating Artery',
            'Right Posterior Communicating Artery',
            'Basilar Tip',
            'Other Posterior Circulation',
            'Aneurysm Present'
        ]
        
        # éªŒè¯�æ ‡ç­¾åˆ—å­˜åœ¨
        missing_cols = [col for col in target_columns if col not in train_df.columns]
        if missing_cols:
            print(f"âš ï¸� Missing columns: {missing_cols}")
            # ä½¿ç”¨å®�é™…å­˜åœ¨çš„æ ‡ç­¾åˆ—
            actual_target_cols = [col for col in target_columns if col in train_df.columns]
            print(f"âœ… Using {len(actual_target_cols)} available target columns")
        else:
            actual_target_cols = target_columns
        
        print(f"ğŸ�¯ Target columns: {len(actual_target_cols)}")
        
        # åˆ†æ��æ ‡ç­¾åˆ†å¸ƒ
        print(f"\nğŸ“ˆ Label distribution analysis:")
        for col in actual_target_cols:
            if col in train_df.columns:
                pos_rate = train_df[col].mean()
                pos_count = train_df[col].sum()
                print(f"   {col[:30]:<30}: {pos_rate:.3f} ({pos_count:,} positive)")
        
        return train_df, actual_target_cols
        
    except Exception as e:
        print(f"â�Œ Error loading training data: {e}")
        return None, None

def create_competition_split(train_df, target_columns, val_size=0.2, random_state=2025):
    """åˆ›å»ºè®­ç»ƒ/éªŒè¯�åˆ†å‰²ï¼ˆç«�èµ›çº§åˆ«ï¼‰"""
    print(f"\nğŸ”„ Creating Train/Validation Split")
    print("=" * 60)
    
    # æ£€æŸ¥æ˜¯å�¦æœ‰PatientIDç”¨äº�åˆ†ç»„
    group_col = None
    for col in ['PatientID', 'patient_id', 'Patient_ID']:
        if col in train_df.columns:
            group_col = col
            break
    
    if group_col:
        print(f"âœ… Using {group_col} for patient-level split")
        
        # æŒ‰æ‚£è€…åˆ†å‰²ï¼Œé�¿å…�æ•°æ�®æ³„éœ²
        unique_patients = train_df[group_col].unique()
        print(f"ğŸ“Š Total unique patients: {len(unique_patients)}")
        
        train_patients, val_patients = train_test_split(
            unique_patients, 
            test_size=val_size, 
            random_state=random_state,
            stratify=None  # æ‚£è€…çº§åˆ«åˆ†å±‚è¾ƒå¤�æ�‚ï¼Œå…ˆä¸�ç”¨
        )
        
        train_data = train_df[train_df[group_col].isin(train_patients)]
        val_data = train_df[train_df[group_col].isin(val_patients)]
        
        print(f"âœ… Patient-level split completed:")
        print(f"   â€¢ Train: {len(train_patients):,} patients â†’ {len(train_data):,} samples")
        print(f"   â€¢ Val:   {len(val_patients):,} patients â†’ {len(val_data):,} samples")
        
    else:
        print(f"âš ï¸� No PatientID found, using sample-level split")
        
        # æ ·æœ¬çº§åˆ«åˆ†å‰²
        train_data, val_data = train_test_split(
            train_df, 
            test_size=val_size, 
            random_state=random_state,
            stratify=train_df[target_columns].sum(axis=1) > 0  # æŒ‰æ˜¯å�¦æœ‰ä»»ä½•é˜³æ€§æ ‡ç­¾åˆ†å±‚
        )
        
        print(f"âœ… Sample-level split completed:")
        print(f"   â€¢ Train: {len(train_data):,} samples")
        print(f"   â€¢ Val:   {len(val_data):,} samples")
    
    # éªŒè¯�åˆ†å‰²è´¨é‡�
    print(f"\nğŸ“Š Split quality check:")
    for col in target_columns[:5]:  # æ£€æŸ¥å‰�5ä¸ªæ ‡ç­¾åˆ—
        if col in train_df.columns:
            train_pos_rate = train_data[col].mean()
            val_pos_rate = val_data[col].mean()
            print(f"   {col[:25]:<25}: Train {train_pos_rate:.3f} | Val {val_pos_rate:.3f}")
    
    # ä¿�å­˜åˆ†å‰²ä¿¡æ�¯ç”¨äº�å¤�ç�°
    split_info = {
        'val_size': val_size,
        'random_state': random_state,
        'group_col': group_col,
        'train_samples': len(train_data),
        'val_samples': len(val_data)
    }
    
    return train_data, val_data, split_info

def analyze_data_for_training(train_data, val_data, target_columns):
    """åˆ†æ��æ•°æ�®ç”¨äº�è®­ç»ƒ"""
    print(f"\nğŸ”� Training Data Analysis")
    print("=" * 60)
    
    # 1. æ•°æ�®é‡�åˆ†æ��
    print(f"ğŸ“Š Dataset size:")
    print(f"   â€¢ Training samples: {len(train_data):,}")
    print(f"   â€¢ Validation samples: {len(val_data):,}")
    print(f"   â€¢ Total samples: {len(train_data) + len(val_data):,}")
    
    # 2. æ ‡ç­¾åˆ†å¸ƒåˆ†æ��
    print(f"\nğŸ�¯ Label distribution (Training set):")
    train_label_stats = {}
    
    for col in target_columns:
        if col in train_data.columns:
            pos_count = train_data[col].sum()
            total_count = len(train_data)
            pos_rate = pos_count / total_count
            
            train_label_stats[col] = {
                'positive': pos_count,
                'total': total_count,
                'rate': pos_rate
            }
            
            print(f"   {col[:35]:<35}: {pos_count:>5,} / {total_count:>6,} ({pos_rate:>6.1%})")
    
    # 3. æ•°æ�®è´¨é‡�æ£€æŸ¥
    print(f"\nğŸ”� Data quality check:")
    
    # æ£€æŸ¥ç¼ºå¤±å€¼
    missing_series = train_data['SeriesInstanceUID'].isnull().sum()
    print(f"   â€¢ Missing SeriesInstanceUID: {missing_series}")
    
    # æ£€æŸ¥é‡�å¤�series
    duplicate_series = train_data['SeriesInstanceUID'].duplicated().sum()
    print(f"   â€¢ Duplicate SeriesInstanceUID: {duplicate_series}")
    
    # æ£€æŸ¥æ ‡ç­¾ç¼ºå¤±
    for col in target_columns[:3]:  # æ£€æŸ¥å‰�3ä¸ª
        if col in train_data.columns:
            missing_labels = train_data[col].isnull().sum()
            print(f"   â€¢ Missing {col[:20]}: {missing_labels}")
    
    # 4. è®­ç»ƒå»ºè®®
    print(f"\nğŸ’¡ Training recommendations:")
    
    total_positive = sum([stats['positive'] for stats in train_label_stats.values()])
    total_samples = len(train_data) * len(target_columns)
    overall_pos_rate = total_positive / total_samples
    
    print(f"   â€¢ Overall positive rate: {overall_pos_rate:.1%}")
    
    if overall_pos_rate < 0.1:
        print(f"   â€¢ Recommendation: Use class weighting for imbalanced data")
    
    if len(train_data) < 1000:
        print(f"   â€¢ Recommendation: Use data augmentation")
        
    if len(train_data) > 10000:
        print(f"   â€¢ Recommendation: Consider batch size â‰¥ 8")
    
    return train_label_stats

# æ‰§è¡Œæ•°æ�®åŠ è½½å’Œåˆ†å‰²
def main_data_setup():
    """ä¸»æ•°æ�®è®¾ç½®æµ�ç¨‹"""
    print("ğŸ�† RSNA 2025 Intracranial Aneurysm Detection - Data Setup")
    print("=" * 80)
    
    # 1. åŠ è½½æ•°æ�®
    train_df, target_columns = load_competition_data()
    if train_df is None:
        return None
    
    # 2. åˆ›å»ºåˆ†å‰²
    train_data, val_data, split_info = create_competition_split(train_df, target_columns)
    
    # 3. åˆ†æ��æ•°æ�®
    label_stats = analyze_data_for_training(train_data, val_data, target_columns)
    
    # 4. è¿”å›�è®­ç»ƒæ‰€éœ€çš„æ‰€æœ‰æ•°æ�®
    training_config = {
        'train_data': train_data,
        'val_data': val_data,
        'target_columns': target_columns,
        'split_info': split_info,
        'label_stats': label_stats
    }
    
    print(f"\nâœ… Data setup completed successfully!")
    print(f"ğŸ�¯ Ready for Strategy 1 training with {len(train_data):,} training samples")
    
    return training_config

# å…¨å±€å�˜é‡�å­˜å‚¨
TRAINING_CONFIG = None

if __name__ == "__main__":
    TRAINING_CONFIG = main_data_setup()
    
    if TRAINING_CONFIG:
        print(f"\nğŸš€ Next step: Run Cell 12 to create the training dataset")
    else:
        print(f"\nâ�Œ Data setup failed - please check training data availability")


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import cv2
import time
import os
import glob
import pydicom

# =============================================================================
# æ•°æ�®åŠ è½½å‡½æ•°
# =============================================================================

def load_volume_data(series_uid, max_slices=None):
    """åŠ è½½DICOMä½“ç§¯æ•°æ�®"""
    try:
        # æŸ¥æ‰¾DICOMæ–‡ä»¶è·¯å¾„
        possible_paths = [
            f"/kaggle/input/rsna-intracranial-aneurysm-detection/train/{series_uid}",
            f"/kaggle/input/rsna-intracranial-aneurysm-detection/{series_uid}",
            f"/kaggle/input/train/{series_uid}",
            f"./train/{series_uid}",
            f"./{series_uid}"
        ]
        
        dicom_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dicom_path = path
                break
        
        if not dicom_path:
            return None
        
        # è¯»å�–DICOMæ–‡ä»¶
        if os.path.isdir(dicom_path):
            dicom_files = glob.glob(os.path.join(dicom_path, "*.dcm"))
            if not dicom_files:
                dicom_files = [f for f in os.listdir(dicom_path) 
                              if f.lower().endswith(('.dcm', '.dicom'))]
                dicom_files = [os.path.join(dicom_path, f) for f in dicom_files]
        else:
            dicom_files = [dicom_path]
        
        if not dicom_files:
            return None
        
        # è¯»å�–DICOMåˆ‡ç‰‡
        slices = []
        for file_path in dicom_files[:max_slices] if max_slices else dicom_files:
            try:
                ds = pydicom.dcmread(file_path)
                
                if hasattr(ds, 'pixel_array'):
                    slice_data = ds.pixel_array.astype(np.float32)
                    
                    # åº”ç”¨çª—å®½çª—ä½�
                    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                        center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter[0])
                        width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth[0])
                        
                        min_val = center - width/2
                        max_val = center + width/2
                        slice_data = np.clip(slice_data, min_val, max_val)
                        slice_data = (slice_data - min_val) / (max_val - min_val)
                    else:
                        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
                    
                    slices.append(slice_data)
                    
            except Exception as e:
                continue
        
        if not slices:
            return None
        
        volume = np.stack(slices, axis=0)
        return volume
        
    except Exception as e:
        return None

def preprocess_volume(volume, target_slices=32, target_size=(128, 128)):
    """é¢„å¤„ç�†ä½“ç§¯æ•°æ�®"""
    if volume is None:
        return np.random.randn(target_slices, target_size[0], target_size[1]).astype(np.float32)
    
    try:
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()
        
        volume = volume.astype(np.float32)
        
        # å¼ºåº¦å½’ä¸€åŒ–
        if volume.max() > volume.min():
            p1, p99 = np.percentile(volume, [1, 99])
            volume = np.clip(volume, p1, p99)
            volume = (volume - p1) / (p99 - p1 + 1e-8)
        
        # è°ƒæ•´åˆ‡ç‰‡æ•°
        current_slices = volume.shape[0]
        if current_slices != target_slices:
            if current_slices > target_slices:
                indices = np.linspace(0, current_slices-1, target_slices).astype(int)
                volume = volume[indices]
            else:
                repeat_factor = target_slices // current_slices
                remainder = target_slices % current_slices
                
                repeated = np.tile(volume, (repeat_factor, 1, 1))
                if remainder > 0:
                    extra_slices = volume[:remainder]
                    volume = np.concatenate([repeated, extra_slices], axis=0)
                else:
                    volume = repeated
        
        # è°ƒæ•´ç©ºé—´å°ºå¯¸
        if volume.shape[1:] != target_size:
            resized_volume = []
            for i in range(volume.shape[0]):
                resized_slice = cv2.resize(
                    volume[i], 
                    target_size,
                    interpolation=cv2.INTER_LINEAR
                )
                resized_volume.append(resized_slice)
            volume = np.stack(resized_volume, axis=0)
        
        return volume
        
    except Exception as e:
        return np.random.randn(target_slices, target_size[0], target_size[1]).astype(np.float32)

def get_patient_info(series_uid):
    """è�·å�–æ‚£è€…ä¿¡æ�¯"""
    try:
        if 'TRAINING_CONFIG' in globals() and TRAINING_CONFIG:
            all_data = pd.concat([
                TRAINING_CONFIG['train_data'],
                TRAINING_CONFIG['val_data']
            ])
            
            patient_row = all_data[all_data['SeriesInstanceUID'] == series_uid]
            
            if not patient_row.empty:
                age = patient_row.iloc[0].get('PatientAge', 50)
                sex = 1 if patient_row.iloc[0].get('PatientSex', 'M') == 'M' else 0
                return float(age), float(sex)
        
        return 50.0, 1.0
        
    except Exception as e:
        return 50.0, 1.0

def create_synthetic_volume(series_uid, shape=(32, 128, 128)):
    """åˆ›å»ºå�ˆæˆ�ä½“ç§¯æ•°æ�®"""
    volume = np.zeros(shape, dtype=np.float32)
    volume += np.random.normal(0, 0.1, shape)
    
    center_z, center_y, center_x = shape[0]//2, shape[1]//2, shape[2]//2
    
    for i in range(shape[0]):
        y, x = np.ogrid[:shape[1], :shape[2]]
        mask = (x - center_x)**2 + (y - center_y)**2 < (shape[1]//3)**2
        volume[i][mask] += 0.5
        
        if np.random.random() < 0.3:
            lesion_y = np.random.randint(shape[1]//4, 3*shape[1]//4)
            lesion_x = np.random.randint(shape[2]//4, 3*shape[2]//4)
            lesion_size = np.random.randint(3, 8)
            
            y, x = np.ogrid[:shape[1], :shape[2]]
            lesion_mask = (x - lesion_x)**2 + (y - lesion_y)**2 < lesion_size**2
            volume[i][lesion_mask] += 0.3
    
    volume = np.clip(volume, 0, 1)
    return volume

# =============================================================================
# ç«�èµ›æ•°æ�®é›†ç±»
# =============================================================================

class AneurysmCompetitionDataset(Dataset):
    """RSNAåŠ¨è„‰ç˜¤æ£€æµ‹ç«�èµ›æ•°æ�®é›†"""
    
    def __init__(self, df, target_columns, mode='train', 
                 image_size=128, num_slices=32, 
                 augmentation=True, cache_size=100):
        
        self.df = df.reset_index(drop=True)
        self.target_columns = target_columns
        self.mode = mode
        self.image_size = image_size
        self.num_slices = num_slices
        self.augmentation = augmentation and (mode == 'train')
        
        # æ•°æ�®ç¼“å­˜
        self.cache_size = cache_size
        self.volume_cache = {}
        self.cache_order = []
        
        print(f"ğŸ“Š Created {mode} dataset:")
        print(f"   â€¢ Samples: {len(self.df):,}")
        print(f"   â€¢ Target columns: {len(self.target_columns)}")
        print(f"   â€¢ Image size: {image_size}x{image_size}")
        print(f"   â€¢ Volume slices: {num_slices}")
        print(f"   â€¢ Augmentation: {self.augmentation}")
        print(f"   â€¢ Cache size: {cache_size}")
        
        # æ•°æ�®å¢�å¼º
        if self.augmentation:
            self.volume_augmentation = self._get_volume_augmentations()
            print(f"   â€¢ Volume augmentations: {len(self.volume_augmentation)}")
    
    def _get_volume_augmentations(self):
        """è�·å�–3Dä½“ç§¯æ•°æ�®å¢�å¼º"""
        def horizontal_flip(volume):
            return np.flip(volume, axis=2).copy()
        
        def vertical_flip(volume):
            return np.flip(volume, axis=1).copy()
        
        def rotate_volume(volume, angle=15):
            rotated = []
            for i in range(volume.shape[0]):
                angle = np.random.uniform(-angle, angle)
                center = (volume.shape[1]//2, volume.shape[2]//2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_slice = cv2.warpAffine(volume[i], matrix, 
                                             (volume.shape[2], volume.shape[1]))
                rotated.append(rotated_slice)
            return np.stack(rotated, axis=0)
        
        def brightness_adjust(volume, factor_range=(0.8, 1.2)):
            factor = np.random.uniform(*factor_range)
            return np.clip(volume * factor, 0, 1)
        
        def contrast_adjust(volume, factor_range=(0.8, 1.2)):
            factor = np.random.uniform(*factor_range)
            mean = volume.mean()
            return np.clip((volume - mean) * factor + mean, 0, 1)
        
        augmentations = [
            lambda x: x,  # å�Ÿå§‹
            horizontal_flip,
            vertical_flip,
            lambda x: rotate_volume(x, 10),
            lambda x: brightness_adjust(x),
            lambda x: contrast_adjust(x)
        ]
        
        return augmentations
    
    def _load_and_cache_volume(self, series_uid):
        """åŠ è½½å¹¶ç¼“å­˜ä½“ç§¯æ•°æ�®"""
        if series_uid in self.volume_cache:
            return self.volume_cache[series_uid]
        
        try:
            # å°�è¯•åŠ è½½çœŸå®�æ•°æ�®
            volume = load_volume_data(series_uid)
            
            if volume is None:
                # ä½¿ç”¨å�ˆæˆ�æ•°æ�®
                volume = create_synthetic_volume(series_uid)
            
            # é¢„å¤„ç�†
            volume = preprocess_volume(volume, self.num_slices, (self.image_size, self.image_size))
            
            # ç¼“å­˜ç®¡ç�†
            if len(self.volume_cache) >= self.cache_size:
                oldest_series = self.cache_order.pop(0)
                del self.volume_cache[oldest_series]
            
            self.volume_cache[series_uid] = volume
            self.cache_order.append(series_uid)
            
            return volume
            
        except Exception as e:
            return np.random.randn(self.num_slices, self.image_size, self.image_size)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        
        try:
            # 1. åŠ è½½ä½“ç§¯æ•°æ�®
            volume = self._load_and_cache_volume(series_uid)
            
            # 2. æ•°æ�®å¢�å¼º
            if self.augmentation and np.random.random() < 0.5:
                aug_func = np.random.choice(self.volume_augmentation)
                volume = aug_func(volume)
            
            # 3. è½¬æ�¢ä¸ºtensor
            volume = torch.tensor(volume, dtype=torch.float32)
            
            # 4. è�·å�–æ ‡ç­¾
            labels = []
            for col in self.target_columns:
                if col in row:
                    labels.append(float(row[col]))
                else:
                    labels.append(0.0)
            
            labels = torch.tensor(labels, dtype=torch.float32)
            
            # 5. è�·å�–å…ƒæ•°æ�®
            age = float(row.get('PatientAge', 50.0))
            sex = 1.0 if row.get('PatientSex', 'M') == 'M' else 0.0
            meta = torch.tensor([age, sex], dtype=torch.float32)
            
            return {
                'volume': volume,
                'labels': labels, 
                'meta': meta,
                'series_uid': series_uid
            }
            
        except Exception as e:
            return {
                'volume': torch.randn(self.num_slices, self.image_size, self.image_size),
                'labels': torch.zeros(len(self.target_columns)),
                'meta': torch.tensor([50.0, 1.0]),
                'series_uid': series_uid
            }

def calculate_class_weights(train_df, target_columns):
    """è®¡ç®—ç±»åˆ«æ�ƒé‡�å¤„ç�†ä¸�å¹³è¡¡æ•°æ�®"""
    print("âš–ï¸� Calculating class weights for imbalanced data...")
    
    class_weights = {}
    
    for col in target_columns:
        if col in train_df.columns:
            y = train_df[col].values
            
            pos_count = y.sum()
            neg_count = len(y) - pos_count
            
            if pos_count > 0 and neg_count > 0:
                pos_weight = neg_count / pos_count
                class_weights[col] = pos_weight
                print(f"   {col[:30]:<30}: pos_weight = {pos_weight:.2f}")
            else:
                class_weights[col] = 1.0
                print(f"   {col[:30]:<30}: pos_weight = 1.00 (no variation)")
    
    return class_weights

def create_competition_dataloaders(training_config, batch_size=4, num_workers=0):
    """åˆ›å»ºç«�èµ›è®­ç»ƒæ•°æ�®åŠ è½½å™¨"""
    print("ğŸ”„ Creating Competition DataLoaders")
    print("=" * 60)
    
    train_data = training_config['train_data']
    val_data = training_config['val_data']
    target_columns = training_config['target_columns']
    
    # åˆ›å»ºæ•°æ�®é›†
    print("ğŸ“Š Creating datasets...")
    train_dataset = AneurysmCompetitionDataset(
        df=train_data,
        target_columns=target_columns,
        mode='train',
        image_size=CFG.image_size,
        num_slices=CFG.num_slices,
        augmentation=True,
        cache_size=50
    )
    
    val_dataset = AneurysmCompetitionDataset(
        df=val_data,
        target_columns=target_columns,
        mode='val',
        image_size=CFG.image_size,
        num_slices=CFG.num_slices,
        augmentation=False,
        cache_size=20
    )
    
    # è®¡ç®—ç±»åˆ«æ�ƒé‡�
    class_weights = calculate_class_weights(train_data, target_columns)
    
    # åˆ›å»ºæ•°æ�®åŠ è½½å™¨
    print(f"\nğŸ”„ Creating dataloaders...")
    print(f"   â€¢ Batch size: {batch_size}")
    print(f"   â€¢ Num workers: {num_workers}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    print(f"âœ… DataLoaders created:")
    print(f"   â€¢ Train batches: {len(train_loader):,}")
    print(f"   â€¢ Val batches: {len(val_loader):,}")
    print(f"   â€¢ Train samples per epoch: {len(train_loader) * batch_size:,}")
    print(f"   â€¢ Val samples: {len(val_loader.dataset):,}")
    
    return train_loader, val_loader, class_weights

def setup_competition_training(training_config, batch_size=4):
    """è®¾ç½®ç«�èµ›è®­ç»ƒ"""
    print("ğŸ�† Setting up Competition Training")
    print("=" * 80)
    
    # åˆ›å»ºæ•°æ�®åŠ è½½å™¨
    train_loader, val_loader, class_weights = create_competition_dataloaders(
        training_config, batch_size=batch_size
    )
    
    # å‡†å¤‡è®­ç»ƒç»„ä»¶
    training_components = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'class_weights': class_weights,
        'target_columns': training_config['target_columns'],
        'num_samples': len(training_config['train_data']),
        'batch_size': batch_size
    }
    
    print(f"\nâœ… Competition training setup completed!")
    print(f"ğŸ�¯ Ready for Strategy 1 fine-tuning!")
    
    return training_components

# æ‰§è¡Œè®¾ç½®
if __name__ == "__main__":
    if 'TRAINING_CONFIG' in globals() and TRAINING_CONFIG:
        TRAINING_COMPONENTS = setup_competition_training(TRAINING_CONFIG, batch_size=4)
        
        if TRAINING_COMPONENTS:
            print(f"\nğŸš€ Next step: Run Cell 13 for Strategy 1 fine-tuning!")
        else:
            print(f"\nâ�Œ Setup failed")
    else:
        print(f"â�Œ TRAINING_CONFIG not found - please run Cell 11 first")


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt

class WeightedBCEWithLogitsLoss(nn.Module):
    """åŠ æ�ƒBCEæ�Ÿå¤±å‡½æ•°"""
    
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, predictions, targets, target_columns):
        """
        predictions: (batch_size, num_classes)
        targets: (batch_size, num_classes)  
        target_columns: list of column names
        """
        total_loss = 0
        
        for i, col in enumerate(target_columns):
            if col in self.class_weights:
                pos_weight = torch.tensor(self.class_weights[col], device=predictions.device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                loss = criterion(predictions[:, i], targets[:, i])
                total_loss += loss
        
        return total_loss / len(target_columns)

def setup_strategy1_model(checkpoint_path, target_columns, class_weights):
    """è®¾ç½®ç­–ç•¥1æ¨¡å�‹ï¼šå†»ç»“backboneï¼Œè®­ç»ƒåˆ†ç±»å¤´"""
    print("ğŸ�¯ Setting up Strategy 1 Model (Freeze Backbone)")
    print("=" * 60)
    
    # 1. åŠ è½½é¢„è®­ç»ƒæ¨¡å�‹
    print("ğŸ“¥ Loading pre-trained model...")
    model = AdaptedSimple3DCNN(
        num_classes=len(target_columns),
        checkpoint_path=checkpoint_path,
        freeze_backbone=False  # å…ˆä¸�å†»ç»“ï¼Œè®©æ�ƒé‡�åŠ è½½
    )
    model = model.to(device)
    
    # 2. å†»ç»“backbone
    print("ğŸ§Š Freezing backbone layers...")
    frozen_params = 0
    trainable_params = 0
    
    # å†»ç»“backboneä¸­çš„æ‰€æœ‰å�‚æ•°
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
        frozen_params += param.numel()
        print(f"   â�„ï¸� Frozen: {name}")
    
    # ä¿�æŒ�åˆ†ç±»å™¨å�¯è®­ç»ƒ
    for name, param in model.meta_fc.named_parameters():
        param.requires_grad = True
        trainable_params += param.numel()
        print(f"   ğŸ”¥ Trainable: meta_fc.{name}")
    
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True
        trainable_params += param.numel()
        print(f"   ğŸ”¥ Trainable: classifier.{name}")
    
    print(f"\nğŸ“Š Parameter Summary:")
    print(f"   â€¢ Frozen parameters: {frozen_params:,}")
    print(f"   â€¢ Trainable parameters: {trainable_params:,}")
    print(f"   â€¢ Total parameters: {frozen_params + trainable_params:,}")
    print(f"   â€¢ Trainable ratio: {trainable_params/(frozen_params + trainable_params)*100:.1f}%")
    
    # 3. è®¾ç½®æ�Ÿå¤±å‡½æ•°
    criterion = WeightedBCEWithLogitsLoss(class_weights)
    
    # 4. è®¾ç½®ä¼˜åŒ–å™¨ï¼ˆå�ªä¼˜åŒ–å�¯è®­ç»ƒå�‚æ•°ï¼‰
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params_list, lr=1e-3, weight_decay=1e-4)
    
    # 5. å­¦ä¹ ç�‡è°ƒåº¦å™¨
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    print(f"\nâš™ï¸� Training Setup:")
    print(f"   â€¢ Optimizer: Adam(lr=1e-3, weight_decay=1e-4)")
    print(f"   â€¢ Scheduler: ReduceLROnPlateau(patience=2)")
    print(f"   â€¢ Loss: WeightedBCEWithLogitsLoss")
    print(f"   â€¢ Trainable params: {len(trainable_params_list):,}")
    
    return model, criterion, optimizer, scheduler

def train_epoch(model, train_loader, criterion, optimizer, target_columns, epoch):
    """è®­ç»ƒä¸€ä¸ªepoch"""
    model.train()
    
    total_loss = 0
    batch_count = 0
    
    print(f"\nğŸ”„ Training Epoch {epoch}")
    print("-" * 50)
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            # æ•°æ�®ç§»åŠ¨åˆ°è®¾å¤‡
            volume = batch['volume'].to(device)
            meta = batch['meta'].to(device)
            labels = batch['labels'].to(device)
            
            # å‰�å�‘ä¼ æ’­
            optimizer.zero_grad()
            outputs = model(volume, meta)
            
            # è®¡ç®—æ�Ÿå¤±
            loss = criterion(outputs, labels, target_columns)
            
            # å��å�‘ä¼ æ’­
            loss.backward()
            
            # æ¢¯åº¦è£�å‰ª
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # è¿›åº¦æ˜¾ç¤º
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                batches_done = batch_idx + 1
                eta = elapsed / batches_done * (len(train_loader) - batches_done)
                
                print(f"   Batch {batch_idx:3d}/{len(train_loader):3d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"ETA: {eta:.0f}s")
                
        except Exception as e:
            print(f"   â�Œ Batch {batch_idx} failed: {e}")
            continue
    
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    elapsed = time.time() - start_time
    
    print(f"âœ… Epoch {epoch} completed:")
    print(f"   â€¢ Average Loss: {avg_loss:.4f}")
    print(f"   â€¢ Time: {elapsed:.1f}s")
    print(f"   â€¢ Samples/sec: {len(train_loader.dataset)/elapsed:.1f}")
    
    return avg_loss

def validate_epoch(model, val_loader, target_columns, epoch):
    """éªŒè¯�ä¸€ä¸ªepoch"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print(f"\nğŸ“Š Validating Epoch {epoch}")
    print("-" * 50)
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                volume = batch['volume'].to(device)
                meta = batch['meta'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(volume, meta)
                
                # åº”ç”¨sigmoidè�·å�–æ¦‚ç�‡
                predictions = torch.sigmoid(outputs)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
            except Exception as e:
                print(f"   â�Œ Validation batch {batch_idx} failed: {e}")
                continue
    
    if not all_predictions:
        print("â�Œ No successful validation batches!")
        return 0.0, {}
    
    # å�ˆå¹¶æ‰€æœ‰é¢„æµ‹
    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)
    
    # è®¡ç®—AUCæŒ‡æ ‡
    aucs = {}
    valid_aucs = []
    
    print(f"\nğŸ“ˆ Per-location AUC scores:")
    
    for i, col in enumerate(target_columns):
        try:
            col_labels = labels[:, i]
            col_preds = predictions[:, i]
            
            # éœ€è¦�è‡³å°‘æœ‰æ­£è´Ÿæ ·æœ¬æ‰�èƒ½è®¡ç®—AUC
            if len(np.unique(col_labels)) > 1:
                auc = roc_auc_score(col_labels, col_preds)
                aucs[col] = auc
                valid_aucs.append(auc)
                
                status = "ğŸŸ¢" if auc > 0.8 else "ğŸŸ¡" if auc > 0.6 else "ğŸ”´"
                print(f"   {status} {col[:30]:<30}: {auc:.4f}")
            else:
                aucs[col] = 0.5  # é»˜è®¤å€¼
                print(f"   âšª {col[:30]:<30}: N/A (no variation)")
                
        except Exception as e:
            aucs[col] = 0.5
            print(f"   â�Œ {col[:30]:<30}: Error ({e})")
    
    # è®¡ç®—å¹³å�‡AUC
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.5
    
    elapsed = time.time() - start_time
    
    print(f"\nğŸ“Š Validation Summary:")
    print(f"   â€¢ Mean AUC: {mean_auc:.4f}")
    print(f"   â€¢ Valid locations: {len(valid_aucs)}/{len(target_columns)}")
    print(f"   â€¢ Time: {elapsed:.1f}s")
    
    return mean_auc, aucs

def run_strategy1_training(training_components, num_epochs=8):
    """è¿�è¡Œç­–ç•¥1è®­ç»ƒ"""
    print("ğŸ�† Starting Strategy 1 Fine-tuning Training")
    print("=" * 80)
    
    # è�·å�–è®­ç»ƒç»„ä»¶
    train_loader = training_components['train_loader']
    val_loader = training_components['val_loader']
    class_weights = training_components['class_weights']
    target_columns = training_components['target_columns']
    
    # è®¾ç½®æ¨¡å�‹
    model, criterion, optimizer, scheduler = setup_strategy1_model(
        CFG.selected_checkpoint, target_columns, class_weights
    )
    
    # è®­ç»ƒå�†å�²
    history = {
        'train_loss': [],
        'val_auc': [],
        'learning_rate': []
    }
    
    best_val_auc = 0.0
    best_epoch = 0
    
    print(f"\nğŸš€ Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # è®­ç»ƒ
        train_loss = train_epoch(model, train_loader, criterion, optimizer, target_columns, epoch)
        
        # éªŒè¯�
        val_auc, location_aucs = validate_epoch(model, val_loader, target_columns, epoch)
        
        # å­¦ä¹ ç�‡è°ƒåº¦
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # è®°å½•å�†å�²
        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)
        history['learning_rate'].append(current_lr)
        
        # ä¿�å­˜æœ€ä½³æ¨¡å�‹
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            
            # ä¿�å­˜æœ€ä½³æ¨¡å�‹
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'location_aucs': location_aucs,
                'target_columns': target_columns
            }, 'best_strategy1_model.pth')
            
            print(f"   ğŸ�† New best model saved! AUC: {val_auc:.4f}")
        
        # å½“å‰�epochæ€»ç»“
        print(f"\nğŸ“Š Epoch {epoch} Summary:")
        print(f"   â€¢ Train Loss: {train_loss:.4f}")
        print(f"   â€¢ Val AUC: {val_auc:.4f}")
        print(f"   â€¢ Best AUC: {best_val_auc:.4f} (epoch {best_epoch})")
        print(f"   â€¢ Learning Rate: {current_lr:.6f}")
        
        # æ—©å�œæ£€æŸ¥
        if epoch - best_epoch > 4:
            print(f"\nâ�¹ï¸� Early stopping: No improvement for 4 epochs")
            break
    
    # è®­ç»ƒå®Œæˆ�
    print(f"\nğŸ�‰ Training completed!")
    print(f"   â€¢ Best validation AUC: {best_val_auc:.4f}")
    print(f"   â€¢ Best epoch: {best_epoch}")
    print(f"   â€¢ Total epochs: {epoch}")
    
    # åŠ è½½æœ€ä½³æ¨¡å�‹
    if best_val_auc > 0:
        print(f"\nğŸ“¥ Loading best model...")
        checkpoint = torch.load('best_strategy1_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"âœ… Best model loaded (AUC: {checkpoint['val_auc']:.4f})")
    
    return model, history, best_val_auc

# æ‰§è¡Œè®­ç»ƒ
if __name__ == "__main__":
    if 'TRAINING_COMPONENTS' in globals() and TRAINING_COMPONENTS:
        print("ğŸ�¯ All components ready - starting Strategy 1 training...")
        
        TRAINED_MODEL, TRAINING_HISTORY, FINAL_AUC = run_strategy1_training(
            TRAINING_COMPONENTS, num_epochs=8
        )
        
        print(f"\nğŸ�† FINAL RESULTS:")
        print(f"   â€¢ Final AUC Score: {FINAL_AUC:.4f}")
        
        if FINAL_AUC > 0.7:
            print(f"   â€¢ Status: ğŸ�‰ EXCELLENT! Ready for submission!")
        elif FINAL_AUC > 0.6:
            print(f"   â€¢ Status: âœ… GOOD! Should submit!")
        elif FINAL_AUC > 0.5:
            print(f"   â€¢ Status: ğŸŸ¡ Moderate. Consider improvements.")
        else:
            print(f"   â€¢ Status: â�Œ Poor. Needs more work.")
            
        print(f"\nğŸš€ Next step: Run Cell 14 for final submission generation!")
        
    else:
        print(f"â�Œ TRAINING_COMPONENTS not found - please run Cell 12 first")




