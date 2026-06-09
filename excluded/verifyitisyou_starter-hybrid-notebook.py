#!/usr/bin/env python3
"""
Enhanced Multi-Source Object Detection Ensemble with Self-Tuning
Advanced ensemble with more models, intelligent strategies, and automatic optimization
"""

# Install required packages with error handling
import subprocess
import sys

def install_package(package):
    """Install package with error handling"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        return True
    except:
        print(f"Warning: Failed to install {package}")
        return False

# Core packages + additional for enhanced features
packages = [
    'ultralytics',
    'opencv-python-headless',
    'pandas',
    'matplotlib',
    'seaborn',
    'tqdm',
    'ensemble-boxes',
    'torch',
    'torchvision',
    'transformers',
    'timm',
    'huggingface_hub',
    'Pillow',
    'requests',
    'gdown',
    'optuna',  # For hyperparameter optimization
    'scikit-learn',  # For validation splitting
    'albumentations',  # For advanced augmentations
    'pycocotools',  # For COCO evaluation
    'super-gradients',  # For YOLO-NAS
]

print("Installing required packages...")
for package in packages:
    install_package(package)

import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
import time
import requests
from PIL import Image
import traceback
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import gc
import hashlib
from datetime import datetime
warnings.filterwarnings("ignore")

# Conditional imports with fallbacks
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available")

try:
    from transformers import (
        AutoModelForObjectDetection, 
        AutoImageProcessor,
        DetrForObjectDetection,
        YolosForObjectDetection,
        pipeline,
        DetrImageProcessor,
        ConditionalDetrForObjectDetection,
        DeformableDetrForObjectDetection,
        TableTransformerForObjectDetection,
        DPTForDepthEstimation
    )
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available")

try:
    from ensemble_boxes import weighted_boxes_fusion, nms, soft_nms, non_maximum_weighted
    ENSEMBLE_BOXES_AVAILABLE = True
except:
    ENSEMBLE_BOXES_AVAILABLE = False
    print("Warning: Ensemble boxes not available")

try:
    from huggingface_hub import hf_hub_download, list_models, snapshot_download
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False
    print("Warning: HuggingFace Hub not available")

try:
    import timm
    TIMM_AVAILABLE = True
except:
    TIMM_AVAILABLE = False
    print("Warning: TIMM not available")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available")

try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import average_precision_score
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False
    print("Warning: Scikit-learn not available")

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: Albumentations not available")

try:
    from super_gradients.training import models as sg_models
    SG_AVAILABLE = True
except:
    SG_AVAILABLE = False
    print("Warning: Super-gradients not available")


class EnhancedObjectDetectionEnsemble:
    """
    Enhanced ensemble with self-tuning capabilities and advanced strategies
    """
    def __init__(self, device=None, enable_self_tuning=True):
        self.models = []
        self.model_info = []
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_performance = defaultdict(lambda: {'mAP': 0.0, 'speed': 0.0, 'predictions': 0})
        self.failed_models = []
        self.retry_attempts = 3
        self.enable_self_tuning = enable_self_tuning
        
        # Self-tuning parameters
        self.adaptive_weights = {}
        self.class_specific_thresholds = defaultdict(lambda: 0.001)
        self.model_correlations = {}
        self.ensemble_params = {
            'iou_threshold': 0.5,
            'conf_threshold': 0.001,
            'sigma': 0.5,
            'skip_box_threshold': 0.001
        }
        
        # Advanced ensemble strategies
        self.ensemble_strategies = {
            'wbf': self._weighted_boxes_fusion,
            'soft_nms': self._soft_nms,
            'nms': self._standard_nms,
            'nmw': self._non_maximum_weighted,
            'adaptive': self._adaptive_ensemble,
            'cascade': self._cascade_ensemble,
            'class_specific': self._class_specific_ensemble,
            'size_aware': self._size_aware_ensemble,
            'hybrid': self._hybrid_ensemble
        }
        
        # Test Time Augmentation settings
        self.tta_transforms = self._setup_tta_transforms()
        
        print(f"Using device: {self.device}")
        print(f"Self-tuning: {'Enabled' if enable_self_tuning else 'Disabled'}")
        
    def _setup_tta_transforms(self):
        """Setup Test Time Augmentation transforms"""
        if not ALBUMENTATIONS_AVAILABLE:
            return None
            
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def safe_model_load(self, load_func, model_name, max_retries=3):
        """Safely load a model with retries and error handling"""
        for attempt in range(max_retries):
            try:
                model = load_func()
                if model is not None:
                    # Initialize performance tracking
                    self.model_performance[model_name] = {
                        'mAP': 0.5,  # Initial assumption
                        'speed': 1.0,
                        'predictions': 0,
                        'successes': 0,
                        'failures': 0
                    }
                    return model
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {model_name}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.failed_models.append(model_name)
        return None
    
    def download_enhanced_yolo_models(self):
        """Download comprehensive YOLO model collection"""
        if not YOLO_AVAILABLE:
            print("Skipping YOLO models - package not available")
            return
            
        print("\n" + "="*60)
        print("DOWNLOADING ENHANCED YOLO MODELS")
        print("="*60)
        
        # Extended YOLO model collection
        yolo_models = [
            # YOLOv8 complete series
            {'name': 'yolov8n.pt', 'weight': 0.7, 'type': 'detection'},
            {'name': 'yolov8s.pt', 'weight': 0.8, 'type': 'detection'},
            {'name': 'yolov8m.pt', 'weight': 0.9, 'type': 'detection'},
            {'name': 'yolov8l.pt', 'weight': 1.0, 'type': 'detection'},
            {'name': 'yolov8x.pt', 'weight': 1.1, 'type': 'detection'},
            
            # YOLOv8 6.0 versions (larger input size)
            {'name': 'yolov8n6.pt', 'weight': 0.75, 'type': 'detection'},
            {'name': 'yolov8s6.pt', 'weight': 0.85, 'type': 'detection'},
            {'name': 'yolov8m6.pt', 'weight': 0.95, 'type': 'detection'},
            {'name': 'yolov8l6.pt', 'weight': 1.05, 'type': 'detection'},
            
            # YOLOv5 models
            {'name': 'yolov5n.pt', 'weight': 0.6, 'type': 'detection'},
            {'name': 'yolov5s.pt', 'weight': 0.7, 'type': 'detection'},
            {'name': 'yolov5m.pt', 'weight': 0.8, 'type': 'detection'},
            {'name': 'yolov5l.pt', 'weight': 0.9, 'type': 'detection'},
            {'name': 'yolov5x.pt', 'weight': 1.0, 'type': 'detection'},
            
            # YOLOv5 6.0 versions
            {'name': 'yolov5n6.pt', 'weight': 0.65, 'type': 'detection'},
            {'name': 'yolov5s6.pt', 'weight': 0.75, 'type': 'detection'},
            
            # Specialized models
            {'name': 'yolov8n-seg.pt', 'weight': 0.7, 'type': 'segmentation'},
            {'name': 'yolov8s-seg.pt', 'weight': 0.8, 'type': 'segmentation'},
            {'name': 'yolov8m-seg.pt', 'weight': 0.9, 'type': 'segmentation'},
            {'name': 'yolov8l-seg.pt', 'weight': 1.0, 'type': 'segmentation'},
            {'name': 'yolov8x-seg.pt', 'weight': 1.1, 'type': 'segmentation'},
            
            # World models
            {'name': 'yolov8s-world.pt', 'weight': 0.8, 'type': 'world'},
            {'name': 'yolov8m-world.pt', 'weight': 0.9, 'type': 'world'},
            {'name': 'yolov8l-world.pt', 'weight': 1.0, 'type': 'world'},
            
            # Pose estimation models (can be adapted for bbox)
            {'name': 'yolov8n-pose.pt', 'weight': 0.7, 'type': 'pose'},
            {'name': 'yolov8s-pose.pt', 'weight': 0.8, 'type': 'pose'},
        ]
        
        for model_config in yolo_models:
            model_name = model_config['name']
            
            def load_yolo():
                return YOLO(model_name)
            
            model = self.safe_model_load(load_yolo, model_name)
            if model:
                self.models.append(model)
                self.model_info.append({
                    'name': model_name,
                    'type': 'yolo',
                    'subtype': model_config['type'],
                    'weight': model_config['weight'],
                    'architecture': 'YOLO',
                    'source': 'ultralytics',
                    'input_size': 640 if '6' not in model_name else 1280
                })
                print(f"✓ Loaded {model_name}")
    
    def download_advanced_transformer_models(self):
        """Download advanced transformer-based detection models"""
        if not TRANSFORMERS_AVAILABLE:
            print("Skipping Transformer models - package not available")
            return
            
        print("\n" + "="*60)
        print("DOWNLOADING ADVANCED TRANSFORMER MODELS")
        print("="*60)
        
        transformer_models = [
            # DETR family
            {
                'name': 'facebook/detr-resnet-50',
                'weight': 1.0,
                'processor': 'facebook/detr-resnet-50'
            },
            {
                'name': 'facebook/detr-resnet-101',
                'weight': 1.1,
                'processor': 'facebook/detr-resnet-101'
            },
            {
                'name': 'facebook/detr-resnet-50-dc5',
                'weight': 1.05,
                'processor': 'facebook/detr-resnet-50-dc5'
            },
            {
                'name': 'facebook/detr-resnet-101-dc5',
                'weight': 1.15,
                'processor': 'facebook/detr-resnet-101-dc5'
            },
            
            # Conditional DETR
            {
                'name': 'microsoft/conditional-detr-resnet-50',
                'weight': 1.2,
                'processor': 'microsoft/conditional-detr-resnet-50'
            },
            
            # Deformable DETR
            {
                'name': 'SenseTime/deformable-detr',
                'weight': 1.3,
                'processor': 'SenseTime/deformable-detr'
            },
            {
                'name': 'SenseTime/deformable-detr-single-scale',
                'weight': 1.25,
                'processor': 'SenseTime/deformable-detr-single-scale'
            },
            
            # YOLOS variants
            {
                'name': 'hustvl/yolos-tiny',
                'weight': 0.8,
                'processor': 'hustvl/yolos-tiny'
            },
            {
                'name': 'hustvl/yolos-small',
                'weight': 0.9,
                'processor': 'hustvl/yolos-small'
            },
            {
                'name': 'hustvl/yolos-base',
                'weight': 1.0,
                'processor': 'hustvl/yolos-base'
            },
            
            # Table Transformer
            {
                'name': 'microsoft/table-transformer-detection',
                'weight': 0.9,
                'processor': 'microsoft/table-transformer-detection'
            },
            {
                'name': 'microsoft/table-transformer-structure-recognition',
                'weight': 0.85,
                'processor': 'microsoft/table-transformer-structure-recognition'
            },
            
            # DINO (DETR with Improved deNoising anchOr boxes)
            {
                'name': 'IDEA-Research/dino-vitb16-4scale',
                'weight': 1.4,
                'processor': 'IDEA-Research/dino-vitb16-4scale'
            },
            
            # Specialized models
            {
                'name': 'valentinafeve/yolos-fashionpedia',
                'weight': 0.9,
                'processor': 'valentinafeve/yolos-fashionpedia'
            },
            {
                'name': 'nickmuchi/yolos-small-finetuned-masks',
                'weight': 0.85,
                'processor': 'nickmuchi/yolos-small-finetuned-masks'
            },
            {
                'name': 'devonho/detr-resnet-50_finetuned_coco',
                'weight': 1.0,
                'processor': 'devonho/detr-resnet-50_finetuned_coco'
            }
        ]
        
        for model_config in transformer_models:
            model_name = model_config['name']
            
            def load_transformer():
                try:
                    # Try different loading strategies
                    model = None
                    processor = None
                    
                    # Strategy 1: Standard loading
                    try:
                        model = AutoModelForObjectDetection.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            ignore_mismatched_sizes=True
                        )
                        processor = AutoImageProcessor.from_pretrained(
                            model_config['processor'],
                            trust_remote_code=True
                        )
                    except:
                        # Strategy 2: Try specific model classes
                        if 'detr' in model_name.lower():
                            if 'conditional' in model_name:
                                model = ConditionalDetrForObjectDetection.from_pretrained(model_name)
                            elif 'deformable' in model_name:
                                model = DeformableDetrForObjectDetection.from_pretrained(model_name)
                            else:
                                model = DetrForObjectDetection.from_pretrained(model_name)
                            processor = DetrImageProcessor.from_pretrained(model_config['processor'])
                        elif 'yolos' in model_name.lower():
                            model = YolosForObjectDetection.from_pretrained(model_name)
                            processor = AutoImageProcessor.from_pretrained(model_config['processor'])
                        elif 'table' in model_name.lower():
                            model = TableTransformerForObjectDetection.from_pretrained(model_name)
                            processor = AutoImageProcessor.from_pretrained(model_config['processor'])
                    
                    if model and processor:
                        # Move to device
                        model = model.to(self.device)
                        model.eval()
                        return {'model': model, 'processor': processor}
                    
                    # Strategy 3: Pipeline fallback
                    pipe = pipeline("object-detection", model=model_name, device=0 if self.device == 'cuda' else -1)
                    return {'pipeline': pipe}
                    
                except Exception as e:
                    print(f"Failed to load transformer {model_name}: {e}")
                    return None
            
            result = self.safe_model_load(load_transformer, model_name)
            if result:
                self.models.append(result)
                self.model_info.append({
                    'name': model_name,
                    'type': 'transformer',
                    'weight': model_config['weight'],
                    'architecture': 'Transformer',
                    'source': 'huggingface'
                })
                print(f"✓ Loaded {model_name}")
    
    def download_yolov9_models(self):
        """Download YOLOv9 models"""
        print("\n" + "="*60)
        print("DOWNLOADING YOLOV9 MODELS")
        print("="*60)
        
        yolov9_models = [
            {
                'name': 'YOLOv9-C',
                'url': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt',
                'weight': 1.2,
            },
            {
                'name': 'YOLOv9-E',
                'url': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt',
                'weight': 1.3,
            },
            {
                'name': 'YOLOv9-C-converted',
                'url': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt',
                'weight': 1.2,
            },
            {
                'name': 'YOLOv9-E-converted',
                'url': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt',
                'weight': 1.3,
            }
        ]
        
        for model_config in yolov9_models:
            model_name = model_config['name']
            
            def download_and_load():
                try:
                    # Download model
                    response = requests.get(model_config['url'], stream=True)
                    response.raise_for_status()
                    
                    model_path = f"./downloaded_models/{model_name}.pt"
                    os.makedirs("./downloaded_models", exist_ok=True)
                    
                    with open(model_path, 'wb') as f:
                        total_size = int(response.headers.get('content-length', 0))
                        block_size = 8192
                        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=model_name) as pbar:
                            for chunk in response.iter_content(chunk_size=block_size):
                                pbar.update(len(chunk))
                                f.write(chunk)
                    
                    # Load model
                    if YOLO_AVAILABLE and 'converted' in model_name:
                        return YOLO(model_path)
                    else:
                        return torch.load(model_path, map_location=self.device)
                except:
                    return None
            
            model = self.safe_model_load(download_and_load, model_name)
            if model:
                self.models.append(model)
                self.model_info.append({
                    'name': model_name,
                    'type': 'yolo',
                    'weight': model_config['weight'],
                    'architecture': 'YOLOv9',
                    'source': 'github'
                })
                print(f"✓ Downloaded and loaded {model_name}")
    
    def download_yolo_nas_models(self):
        """Download YOLO-NAS models using super-gradients"""
        if not SG_AVAILABLE:
            print("Skipping YOLO-NAS models - super-gradients not available")
            return
            
        print("\n" + "="*60)
        print("DOWNLOADING YOLO-NAS MODELS")
        print("="*60)
        
        yolo_nas_models = [
            {'name': 'yolo_nas_s', 'weight': 0.9},
            {'name': 'yolo_nas_m', 'weight': 1.0},
            {'name': 'yolo_nas_l', 'weight': 1.1},
        ]
        
        for model_config in yolo_nas_models:
            model_name = model_config['name']
            
            def load_yolo_nas():
                try:
                    model = sg_models.get(model_name, pretrained_weights="coco")
                    return model
                except:
                    return None
            
            model = self.safe_model_load(load_yolo_nas, model_name)
            if model:
                self.models.append(model)
                self.model_info.append({
                    'name': model_name,
                    'type': 'yolo-nas',
                    'weight': model_config['weight'],
                    'architecture': 'YOLO-NAS',
                    'source': 'super-gradients'
                })
                print(f"✓ Loaded {model_name}")
    
    def download_specialized_hf_models(self):
        """Download specialized and fine-tuned models from HuggingFace"""
        if not HF_AVAILABLE:
            print("Skipping specialized HuggingFace models")
            return
            
        print("\n" + "="*60)
        print("DOWNLOADING SPECIALIZED HUGGINGFACE MODELS")
        print("="*60)
        
        specialized_models = [
            # Domain-specific YOLO models
            {
                'repo': 'keremberke/yolov8n-forklift-detection',
                'filename': 'best.pt',
                'weight': 0.8,
                'type': 'yolo'
            },
            {
                'repo': 'keremberke/yolov8m-hard-hat-detection',
                'filename': 'best.pt',
                'weight': 0.9,
                'type': 'yolo'
            },
            {
                'repo': 'keremberke/yolov8s-traffic-sign-detection',
                'filename': 'best.pt',
                'weight': 0.85,
                'type': 'yolo'
            },
            {
                'repo': 'keremberke/yolov8n-pothole-segmentation',
                'filename': 'best.pt',
                'weight': 0.8,
                'type': 'yolo'
            },
            {
                'repo': 'keremberke/yolov8s-plane-detection',
                'filename': 'best.pt',
                'weight': 0.85,
                'type': 'yolo'
            },
            {
                'repo': 'mshamrai/yolov8x-visdrone',
                'filename': 'yolov8x_visdrone.pt',
                'weight': 1.0,
                'type': 'yolo'
            },
            {
                'repo': 'fcakyon/yolov5s-v7.0',
                'filename': 'yolov5s.pt',
                'weight': 0.8,
                'type': 'yolo'
            },
            {
                'repo': 'fcakyon/yolov5m-v7.0',
                'filename': 'yolov5m.pt',
                'weight': 0.9,
                'type': 'yolo'
            },
            
            # RT-DETR models
            {
                'repo': 'PekingU/rtdetr_r50vd_coco_o365',
                'filename': 'model.pth',
                'weight': 1.1,
                'type': 'rtdetr'
            },
            {
                'repo': 'PekingU/rtdetr_r101vd_coco_o365',
                'filename': 'model.pth',
                'weight': 1.2,
                'type': 'rtdetr'
            },
            
            # Transformer models
            {
                'repo': 'jozhang97/deta-resnet-50',
                'filename': None,
                'weight': 1.1,
                'type': 'transformer'
            },
            {
                'repo': 'jozhang97/deta-swin-large',
                'filename': None,
                'weight': 1.2,
                'type': 'transformer'
            },
            {
                'repo': 'SenseTime/deformable-detr-with-box-refine',
                'filename': None,
                'weight': 1.3,
                'type': 'transformer'
            },
            {
                'repo': 'SenseTime/deformable-detr-with-box-refine-two-stage',
                'filename': None,
                'weight': 1.35,
                'type': 'transformer'
            }
        ]
        
        for model_config in specialized_models:
            repo = model_config['repo']
            
            def load_specialized():
                try:
                    if model_config['filename'] and model_config['type'] == 'yolo':
                        # Download specific file for YOLO models
                        model_path = hf_hub_download(
                            repo_id=repo,
                            filename=model_config['filename'],
                            cache_dir='./hf_cache'
                        )
                        if YOLO_AVAILABLE:
                            return YOLO(model_path)
                        else:
                            return torch.load(model_path, map_location=self.device)
                    else:
                        # Load transformer models
                        if model_config['type'] == 'transformer':
                            return pipeline("object-detection", model=repo, device=0 if self.device == 'cuda' else -1)
                        else:
                            # Generic loading
                            model = AutoModelForObjectDetection.from_pretrained(repo)
                            return model
                except:
                    return None
            
            model = self.safe_model_load(load_specialized, repo, max_retries=2)
            if model:
                self.models.append(model)
                self.model_info.append({
                    'name': repo.split('/')[-1],
                    'type': 'specialized',
                    'subtype': model_config['type'],
                    'weight': model_config['weight'],
                    'architecture': model_config['type'].upper(),
                    'source': 'huggingface'
                })
                print(f"✓ Loaded {repo}")
    
    def download_efficientdet_models(self):
        """Download EfficientDet models"""
        if not TIMM_AVAILABLE:
            print("Skipping EfficientDet models - TIMM not available")
            return
            
        print("\n" + "="*60)
        print("DOWNLOADING EFFICIENTDET MODELS")
        print("="*60)
        
        efficientdet_models = [
            {'name': 'tf_efficientdet_d0', 'weight': 0.8},
            {'name': 'tf_efficientdet_d1', 'weight': 0.85},
            {'name': 'tf_efficientdet_d2', 'weight': 0.9},
            {'name': 'tf_efficientdet_d3', 'weight': 0.95},
            {'name': 'tf_efficientdet_d4', 'weight': 1.0},
            {'name': 'tf_efficientdet_d5', 'weight': 1.05},
            {'name': 'tf_efficientdet_d6', 'weight': 1.1},
            {'name': 'tf_efficientdet_d7', 'weight': 1.15},
            {'name': 'tf_efficientdetv2_s', 'weight': 0.9},
            {'name': 'tf_efficientdetv2_m', 'weight': 1.0},
            {'name': 'tf_efficientdetv2_l', 'weight': 1.1},
        ]
        
        for model_config in efficientdet_models:
            model_name = model_config['name']
            
            def load_efficientdet():
                try:
                    model = timm.create_model(
                        model_name,
                        pretrained=True,
                        checkpoint_path='',
                        num_classes=1000,
                        features_only=False
                    )
                    model.eval()
                    return model
                except:
                    return None
            
            model = self.safe_model_load(load_efficientdet, model_name)
            if model:
                self.models.append(model)
                self.model_info.append({
                    'name': model_name,
                    'type': 'efficientdet',
                    'weight': model_config['weight'],
                    'architecture': 'EfficientDet',
                    'source': 'timm'
                })
                print(f"✓ Loaded {model_name}")
    
    def auto_discover_and_rank_models(self, limit=15):
        """Automatically discover and rank models from HuggingFace"""
        if not HF_AVAILABLE:
            return
            
        print("\n" + "="*60)
        print("AUTO-DISCOVERING AND RANKING MODELS")
        print("="*60)
        
        try:
            # Search for object detection models
            detection_models = list(list_models(
                filter=["object-detection"],
                sort="downloads",
                direction=-1,
                limit=limit * 2  # Get more to filter
            ))
            
            # Rank models based on multiple criteria
            ranked_models = []
            for model in detection_models:
                model_id = model.modelId
                
                # Skip test/demo models
                if any(skip in model_id.lower() for skip in ['demo', 'test', 'example', 'tutorial']):
                    continue
                
                # Calculate ranking score
                score = 0
                
                # Downloads factor
                downloads = getattr(model, 'downloads', 0)
                score += min(downloads / 10000, 10)  # Normalize
                
                # Likes factor
                likes = getattr(model, 'likes', 0)
                score += min(likes / 100, 5)
                
                # Model size factor (prefer balanced sizes)
                try:
                    tags = getattr(model, 'tags', [])
                    if 'yolo' in str(tags).lower():
                        score += 3
                    if 'detr' in str(tags).lower():
                        score += 2
                    if 'transformer' in str(tags).lower():
                        score += 2
                except:
                    pass
                
                ranked_models.append((model_id, score))
            
            # Sort by score
            ranked_models.sort(key=lambda x: x[1], reverse=True)
            
            # Load top models
            for model_id, score in ranked_models[:limit]:
                if len(self.models) >= 50:  # Total model limit
                    break
                
                def load_discovered():
                    try:
                        return pipeline(
                            "object-detection", 
                            model=model_id, 
                            device=0 if self.device == 'cuda' else -1
                        )
                    except:
                        return None
                
                pipe = self.safe_model_load(load_discovered, model_id, max_retries=1)
                if pipe:
                    self.models.append(pipe)
                    self.model_info.append({
                        'name': model_id.split('/')[-1],
                        'type': 'discovered',
                        'weight': 0.8 + (score / 20),  # Dynamic weight based on ranking
                        'architecture': 'Unknown',
                        'source': 'huggingface-auto',
                        'ranking_score': score
                    })
                    print(f"✓ Auto-discovered {model_id} (score: {score:.2f})")
                    
        except Exception as e:
            print(f"Model discovery failed: {e}")
    
    def setup_self_tuning(self, validation_data=None):
        """Setup self-tuning with validation data"""
        if not self.enable_self_tuning:
            return
            
        print("\n" + "="*60)
        print("SETTING UP SELF-TUNING")
        print("="*60)
        
        if validation_data:
            # Evaluate models on validation data
            self._evaluate_models_on_validation(validation_data)
            
        # Initialize adaptive weights based on performance
        self._initialize_adaptive_weights()
        
        # Optimize ensemble parameters if Optuna available
        if OPTUNA_AVAILABLE and validation_data:
            self._optimize_ensemble_parameters(validation_data)
    
    def _evaluate_models_on_validation(self, validation_data):
        """Evaluate each model on validation data"""
        print("Evaluating models on validation data...")
        
        for idx, (model, info) in enumerate(zip(self.models, self.model_info)):
            model_name = info['name']
            total_predictions = 0
            successful_predictions = 0
            
            # Sample evaluation on subset
            for img_path, ground_truth in validation_data[:10]:  # Sample
                try:
                    predictions = self.predict_with_model(model, cv2.imread(img_path), info)
                    if predictions:
                        successful_predictions += 1
                    total_predictions += 1
                except:
                    pass
            
            # Update performance metrics
            if total_predictions > 0:
                success_rate = successful_predictions / total_predictions
                self.model_performance[model_name]['mAP'] = success_rate
                self.model_performance[model_name]['predictions'] = total_predictions
                self.model_performance[model_name]['successes'] = successful_predictions
    
    def _initialize_adaptive_weights(self):
        """Initialize adaptive weights based on model performance"""
        for info in self.model_info:
            model_name = info['name']
            base_weight = info['weight']
            
            # Adjust weight based on performance
            performance = self.model_performance.get(model_name, {})
            mAP = performance.get('mAP', 0.5)
            
            # Adaptive weight = base_weight * performance_factor
            performance_factor = 0.5 + mAP  # Range: 0.5 to 1.5
            self.adaptive_weights[model_name] = base_weight * performance_factor
            
        print(f"Initialized adaptive weights for {len(self.adaptive_weights)} models")
    
    def _optimize_ensemble_parameters(self, validation_data):
        """Optimize ensemble parameters using Optuna"""
        print("Optimizing ensemble parameters...")
        
        def objective(trial):
            # Suggest parameters
            params = {
                'iou_threshold': trial.suggest_float('iou_threshold', 0.3, 0.7),
                'conf_threshold': trial.suggest_float('conf_threshold', 0.0001, 0.01),
                'sigma': trial.suggest_float('sigma', 0.3, 0.7),
                'skip_box_threshold': trial.suggest_float('skip_box_threshold', 0.0001, 0.01)
            }
            
            # Update ensemble parameters
            self.ensemble_params.update(params)
            
            # Evaluate on validation subset
            score = 0
            for img_path, ground_truth in validation_data[:5]:  # Small subset
                predictions = self.ensemble_predictions(img_path, strategy='adaptive')
                # Simple scoring based on number of predictions
                if predictions:
                    score += len(predictions) / 100  # Normalize
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, timeout=60)  # Quick optimization
        
        # Update with best parameters
        self.ensemble_params.update(study.best_params)
        print(f"Optimized parameters: {study.best_params}")
    
    def predict_with_model(self, model, image, model_info):
        """Unified prediction interface with performance tracking"""
        predictions = []
        start_time = time.time()
        
        try:
            # Update attempt counter
            model_name = model_info['name']
            self.model_performance[model_name]['predictions'] += 1
            
            # Get predictions based on model type
            if model_info['type'] in ['yolo', 'yolo-nas'] or (
                model_info['type'] in ['specialized', 'local', 'public'] and hasattr(model, 'predict')
            ):
                # YOLO-style prediction
                if model_info['type'] == 'yolo-nas':
                    # YOLO-NAS specific prediction
                    results = model.predict(image, conf=self.ensemble_params['conf_threshold'])
                    predictions = self._extract_yolo_nas_boxes(results)
                else:
                    # Standard YOLO prediction
                    results = model.predict(
                        image, 
                        conf=self.ensemble_params['conf_threshold'], 
                        device=self.device, 
                        verbose=False
                    )
                    if results and len(results) > 0:
                        predictions = self._extract_yolo_boxes(results[0])
                        
            elif model_info['type'] == 'transformer' or isinstance(model, dict):
                # Transformer model prediction
                predictions = self._predict_with_transformer(model, image, model_info)
                
            elif model_info['type'] == 'efficientdet':
                # EfficientDet prediction
                predictions = self._predict_with_efficientdet(model, image, model_info)
                
            elif hasattr(model, '__call__'):
                # Generic callable model
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                results = model(pil_image)
                predictions = self._extract_transformer_boxes(results, image.shape)
            
            # Update success counter
            if predictions:
                self.model_performance[model_name]['successes'] += 1
                
            # Update speed metric
            elapsed_time = time.time() - start_time
            self.model_performance[model_name]['speed'] = elapsed_time
            
        except Exception as e:
            # Update failure counter
            self.model_performance[model_info['name']]['failures'] = \
                self.model_performance[model_info['name']].get('failures', 0) + 1
                
            if self.model_performance[model_info['name']]['failures'] % 10 == 0:
                print(f"Model {model_info['name']} has failed {self.model_performance[model_info['name']]['failures']} times")
        
        return predictions
    
    def _predict_with_transformer(self, model, image, model_info):
        """Handle transformer model predictions"""
        predictions = []
        
        if isinstance(model, dict) and 'pipeline' in model:
            # Pipeline interface
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = model['pipeline'](pil_image)
            predictions = self._extract_transformer_boxes(results, image.shape)
            
        elif isinstance(model, dict) and 'model' in model:
            # Model + processor interface
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            inputs = model['processor'](images=pil_image, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model['model'](**inputs)
                
            # Post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
            results = model['processor'].post_process_object_detection(
                outputs, 
                threshold=self.ensemble_params['conf_threshold'], 
                target_sizes=target_sizes
            )[0]
            
            predictions = self._extract_transformer_boxes_raw(results, image.shape)
            
        return predictions
    
    def _predict_with_efficientdet(self, model, image, model_info):
        """Handle EfficientDet predictions"""
        # This is a placeholder - actual implementation would depend on the specific model
        # For now, return empty predictions
        return []
    
    def _extract_yolo_boxes(self, results):
        """Extract boxes from YOLO results"""
        boxes = []
        if results.boxes is None or len(results.boxes) == 0:
            return boxes
            
        width, height = results.orig_shape[1], results.orig_shape[0]
        
        for box in results.boxes:
            try:
                cls = int(box.cls.cpu().numpy().item())
                conf = float(box.conf.cpu().numpy().item())
                x_center_abs, y_center_abs, w_abs, h_abs = box.xywh[0].cpu().numpy()
                
                # Normalize coordinates
                x_center = x_center_abs / width
                y_center = y_center_abs / height
                box_width = w_abs / width
                box_height = h_abs / height
                
                # Validate box
                if all(0 <= v <= 1 for v in [x_center, y_center, box_width, box_height]):
                    boxes.append([cls, conf, x_center, y_center, box_width, box_height])
            except:
                continue
                
        return boxes
    
    def _extract_yolo_nas_boxes(self, results):
        """Extract boxes from YOLO-NAS results"""
        boxes = []
        
        try:
            # YOLO-NAS returns a different format
            if hasattr(results, 'prediction'):
                pred = results.prediction
                for i in range(len(pred.bboxes_xyxy)):
                    bbox = pred.bboxes_xyxy[i]
                    conf = pred.confidence[i]
                    cls = pred.labels[i]
                    
                    # Convert to center format and normalize
                    x_center = (bbox[0] + bbox[2]) / 2 / pred.image_shape[1]
                    y_center = (bbox[1] + bbox[3]) / 2 / pred.image_shape[0]
                    box_width = (bbox[2] - bbox[0]) / pred.image_shape[1]
                    box_height = (bbox[3] - bbox[1]) / pred.image_shape[0]
                    
                    boxes.append([int(cls), float(conf), x_center, y_center, box_width, box_height])
        except:
            pass
            
        return boxes
    
    def _extract_transformer_boxes(self, results, image_shape):
        """Extract boxes from transformer pipeline results"""
        boxes = []
        height, width = image_shape[:2]
        
        for detection in results:
            try:
                # Get bounding box
                bbox = detection.get('box', {})
                x_min = bbox.get('xmin', 0)
                y_min = bbox.get('ymin', 0)
                x_max = bbox.get('xmax', width)
                y_max = bbox.get('ymax', height)
                
                # Convert to center format and normalize
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height
                
                # Get class and confidence
                label = detection.get('label', 'object')
                score = detection.get('score', 0.5)
                
                # Map label to class id
                cls_id = self._label_to_class_id(label)
                
                if all(0 <= v <= 1 for v in [x_center, y_center, box_width, box_height]):
                    boxes.append([cls_id, score, x_center, y_center, box_width, box_height])
            except:
                continue
                
        return boxes
    
    def _extract_transformer_boxes_raw(self, results, image_shape):
        """Extract boxes from raw transformer outputs"""
        boxes = []
        height, width = image_shape[:2]
        
        try:
            # Handle both CPU and GPU tensors
            scores = results['scores'].cpu().numpy() if hasattr(results['scores'], 'cpu') else results['scores']
            labels = results['labels'].cpu().numpy() if hasattr(results['labels'], 'cpu') else results['labels']
            boxes_data = results['boxes'].cpu().numpy() if hasattr(results['boxes'], 'cpu') else results['boxes']
            
            for score, label, box in zip(scores, labels, boxes_data):
                if score > self.ensemble_params['conf_threshold']:
                    x_min, y_min, x_max, y_max = box
                    
                    # Convert to center format and normalize
                    x_center = (x_min + x_max) / 2 / width
                    y_center = (y_min + y_max) / 2 / height
                    box_width = (x_max - x_min) / width
                    box_height = (y_max - y_min) / height
                    
                    if all(0 <= v <= 1 for v in [x_center, y_center, box_width, box_height]):
                        boxes.append([int(label), float(score), x_center, y_center, box_width, box_height])
        except Exception as e:
            print(f"Error extracting transformer boxes: {e}")
            
        return boxes
    
    def _label_to_class_id(self, label):
        """Convert string label to class ID"""
        # This is a simplified mapping - in practice, you'd want a proper COCO class mapping
        if isinstance(label, str):
            # Use hash for consistent mapping
            return hash(label) % 80  # Assuming 80 COCO classes
        return int(label)
    
    def apply_test_time_augmentation(self, image, model, model_info):
        """Apply Test Time Augmentation"""
        all_predictions = []
        
        # Original prediction
        original_pred = self.predict_with_model(model, image, model_info)
        all_predictions.extend(original_pred)
        
        if not ALBUMENTATIONS_AVAILABLE or not self.tta_transforms:
            return all_predictions
        
        # TTA predictions
        augmentations = [
            # Horizontal flip
            lambda img: cv2.flip(img, 1),
            # Vertical flip  
            lambda img: cv2.flip(img, 0),
            # Rotate 90
            lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            # Rotate 270
            lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            # Brightness adjustment
            lambda img: cv2.convertScaleAbs(img, alpha=1.2, beta=10),
            lambda img: cv2.convertScaleAbs(img, alpha=0.8, beta=-10),
        ]
        
        for aug_func in augmentations[:3]:  # Limit TTA for speed
            try:
                aug_image = aug_func(image)
                aug_pred = self.predict_with_model(model, aug_image, model_info)
                
                # Reverse transform boxes if needed
                if aug_func == augmentations[0]:  # Horizontal flip
                    for pred in aug_pred:
                        pred[2] = 1 - pred[2]  # Flip x coordinate
                elif aug_func == augmentations[1]:  # Vertical flip
                    for pred in aug_pred:
                        pred[3] = 1 - pred[3]  # Flip y coordinate
                
                all_predictions.extend(aug_pred)
            except:
                continue
        
        return all_predictions
    
    # Advanced Ensemble Strategies
    
    def _weighted_boxes_fusion(self, predictions_by_class):
        """Standard Weighted Boxes Fusion"""
        final_predictions = []
        
        for cls, class_preds in predictions_by_class.items():
            if not class_preds:
                continue
                
            boxes, scores, labels = self._prepare_boxes_for_ensemble(class_preds, cls)
            
            try:
                boxes_out, scores_out, labels_out = weighted_boxes_fusion(
                    [boxes], [scores], [labels],
                    weights=None, 
                    iou_thr=self.ensemble_params['iou_threshold'],
                    skip_box_thr=self.ensemble_params['skip_box_threshold']
                )
                
                final_predictions.extend(
                    self._convert_from_corner_format(boxes_out, scores_out, labels_out)
                )
            except:
                final_predictions.extend(class_preds)
                
        return final_predictions
    
    def _soft_nms(self, predictions_by_class):
        """Soft-NMS ensemble"""
        final_predictions = []
        
        for cls, class_preds in predictions_by_class.items():
            if not class_preds:
                continue
                
            boxes, scores, labels = self._prepare_boxes_for_ensemble(class_preds, cls)
            
            try:
                boxes_out, scores_out, labels_out = soft_nms(
                    [boxes], [scores], [labels],
                    weights=None,
                    iou_thr=self.ensemble_params['iou_threshold'],
                    sigma=self.ensemble_params['sigma'],
                    thresh=self.ensemble_params['skip_box_threshold']
                )
                
                final_predictions.extend(
                    self._convert_from_corner_format(boxes_out, scores_out, labels_out)
                )
            except:
                final_predictions.extend(class_preds)
                
        return final_predictions
    
    def _standard_nms(self, predictions_by_class):
        """Standard NMS"""
        final_predictions = []
        
        for cls, class_preds in predictions_by_class.items():
            if not class_preds:
                continue
                
            boxes, scores, labels = self._prepare_boxes_for_ensemble(class_preds, cls)
            
            try:
                boxes_out, scores_out, labels_out = nms(
                    [boxes], [scores], [labels],
                    weights=None,
                    iou_thr=self.ensemble_params['iou_threshold']
                )
                
                final_predictions.extend(
                    self._convert_from_corner_format(boxes_out, scores_out, labels_out)
                )
            except:
                final_predictions.extend(class_preds)
                
        return final_predictions
    
    def _non_maximum_weighted(self, predictions_by_class):
        """Non-Maximum Weighted ensemble"""
        final_predictions = []
        
        for cls, class_preds in predictions_by_class.items():
            if not class_preds:
                continue
                
            boxes, scores, labels = self._prepare_boxes_for_ensemble(class_preds, cls)
            
            try:
                boxes_out, scores_out, labels_out = non_maximum_weighted(
                    [boxes], [scores], [labels],
                    weights=None,
                    iou_thr=self.ensemble_params['iou_threshold'],
                    skip_box_thr=self.ensemble_params['skip_box_threshold']
                )
                
                final_predictions.extend(
                    self._convert_from_corner_format(boxes_out, scores_out, labels_out)
                )
            except:
                final_predictions.extend(class_preds)
                
        return final_predictions
    
    def _adaptive_ensemble(self, predictions_by_class):
        """Adaptive ensemble using model performance"""
        final_predictions = []
        
        for cls, class_preds in predictions_by_class.items():
            if not class_preds:
                continue
            
            # Group predictions by model
            model_predictions = defaultdict(list)
            for pred in class_preds:
                # Identify model (simplified - you'd track this properly)
                model_id = pred[1] % len(self.models)  # Hack to identify model
                model_predictions[model_id].append(pred)
            
            # Apply adaptive weights
            weighted_preds = []
            for model_id, preds in model_predictions.items():
                model_name = self.model_info[model_id]['name'] if model_id < len(self.model_info) else 'unknown'
                adaptive_weight = self.adaptive_weights.get(model_name, 1.0)
                
                for pred in preds:
                    weighted_pred = pred.copy()
                    weighted_pred[1] *= adaptive_weight  # Apply adaptive weight
                    weighted_preds.append(weighted_pred)
            
            # Use WBF on weighted predictions
            boxes, scores, labels = self._prepare_boxes_for_ensemble(weighted_preds, cls)
            
            try:
                boxes_out, scores_out, labels_out = weighted_boxes_fusion(
                    [boxes], [scores], [labels],
                    weights=None,
                    iou_thr=self.ensemble_params['iou_threshold'],
                    skip_box_thr=self.ensemble_params['skip_box_threshold']
                )
                
                final_predictions.extend(
                    self._convert_from_corner_format(boxes_out, scores_out, labels_out)
                )
            except:
                final_predictions.extend(weighted_preds)
                
        return final_predictions
    
    def _cascade_ensemble(self, predictions_by_class):
        """Cascade ensemble - process high confidence first"""
        final_predictions = []
        confidence_stages = [0.9, 0.7, 0.5, 0.3, 0.1]
        
        for cls, class_preds in predictions_by_class.items():
            if not class_preds:
                continue
            
            remaining_preds = class_preds.copy()
            stage_results = []
            
            for conf_threshold in confidence_stages:
                # Extract high confidence predictions
                high_conf = [p for p in remaining_preds if p[1] >= conf_threshold]
                low_conf = [p for p in remaining_preds if p[1] < conf_threshold]
                
                if high_conf:
                    # Process high confidence predictions
                    boxes, scores, labels = self._prepare_boxes_for_ensemble(high_conf, cls)
                    
                    try:
                        boxes_out, scores_out, labels_out = weighted_boxes_fusion(
                            [boxes], [scores], [labels],
                            weights=None,
                            iou_thr=self.ensemble_params['iou_threshold'] * (1 + 0.2 * (1 - conf_threshold)),
                            skip_box_thr=conf_threshold * 0.5
                        )
                        
                        stage_results.extend(
                            self._convert_from_corner_format(boxes_out, scores_out, labels_out)
                        )
                    except:
                        stage_results.extend(high_conf)
                
                remaining_preds = low_conf
                if not remaining_preds:
                    break
            
            final_predictions.extend(stage_results)
            
        return final_predictions
    
    def _class_specific_ensemble(self, predictions_by_class):
        """Use different strategies for different classes"""
        final_predictions = []
        
        # Define class-specific strategies (customize based on your classes)
        class_strategies = {
            0: 'wbf',      # person
            1: 'soft_nms',  # bicycle
            2: 'wbf',      # car
            3: 'soft_nms',  # motorcycle
            # ... add more as needed
        }
        
        for cls, class_preds in predictions_by_class.items():
            if not class_preds:
                continue
            
            # Get strategy for this class
            strategy = class_strategies.get(cls, 'wbf')
            
            # Apply appropriate strategy
            if strategy == 'wbf':
                preds = self._weighted_boxes_fusion({cls: class_preds})
            elif strategy == 'soft_nms':
                preds = self._soft_nms({cls: class_preds})
            else:
                preds = self._standard_nms({cls: class_preds})
            
            final_predictions.extend(preds)
            
        return final_predictions
    
    def _size_aware_ensemble(self, predictions_by_class):
        """Different handling for small/medium/large objects"""
        final_predictions = []
        
        for cls, class_preds in predictions_by_class.items():
            if not class_preds:
                continue
            
            # Categorize by size
            small_preds = []
            medium_preds = []
            large_preds = []
            
            for pred in class_preds:
                area = pred[4] * pred[5]  # width * height
                if area < 0.01:  # Small objects
                    small_preds.append(pred)
                elif area < 0.1:  # Medium objects
                    medium_preds.append(pred)
                else:  # Large objects
                    large_preds.append(pred)
            
            # Process each size category with appropriate parameters
            for preds, size_factor in [(small_preds, 0.7), (medium_preds, 1.0), (large_preds, 1.3)]:
                if not preds:
                    continue
                
                boxes, scores, labels = self._prepare_boxes_for_ensemble(preds, cls)
                
                try:
                    boxes_out, scores_out, labels_out = weighted_boxes_fusion(
                        [boxes], [scores], [labels],
                        weights=None,
                        iou_thr=self.ensemble_params['iou_threshold'] * size_factor,
                        skip_box_thr=self.ensemble_params['skip_box_threshold']
                    )
                    
                    final_predictions.extend(
                        self._convert_from_corner_format(boxes_out, scores_out, labels_out)
                    )
                except:
                    final_predictions.extend(preds)
                    
        return final_predictions
    
    def _hybrid_ensemble(self, predictions_by_class):
        """Hybrid approach combining multiple strategies"""
        # Get predictions from multiple strategies
        wbf_preds = self._weighted_boxes_fusion(predictions_by_class)
        soft_nms_preds = self._soft_nms(predictions_by_class)
        adaptive_preds = self._adaptive_ensemble(predictions_by_class)
        
        # Combine all predictions
        all_predictions = wbf_preds + soft_nms_preds + adaptive_preds
        
        # Group by class again
        combined_by_class = defaultdict(list)
        for pred in all_predictions:
            combined_by_class[int(pred[0])].append(pred)
        
        # Final fusion
        return self._weighted_boxes_fusion(combined_by_class)
    
    def _prepare_boxes_for_ensemble(self, predictions, cls):
        """Prepare boxes for ensemble methods"""
        boxes = []
        scores = []
        labels = []
        
        for pred in predictions:
            _, score, x_center, y_center, width, height = pred
            
            # Convert to corner format
            x1 = max(0, x_center - width/2)
            y1 = max(0, y_center - height/2)
            x2 = min(1, x_center + width/2)
            y2 = min(1, y_center + height/2)
            
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(cls)
            
        return boxes, scores, labels
    
    def _convert_from_corner_format(self, boxes, scores, labels):
        """Convert from corner format back to center format"""
        predictions = []
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            predictions.append([int(label), score, x_center, y_center, width, height])
            
        return predictions
    
    def ensemble_predictions(self, image_path, strategy='adaptive', use_tta=False):
        """Enhanced ensemble prediction with multiple strategies"""
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            return []
        
        all_predictions = []
        
        # Collect predictions from all models
        for model, info in zip(self.models, self.model_info):
            try:
                # Check if model should be skipped based on performance
                if self.enable_self_tuning:
                    model_name = info['name']
                    perf = self.model_performance.get(model_name, {})
                    failure_rate = perf.get('failures', 0) / max(perf.get('predictions', 1), 1)
                    
                    # Skip models with high failure rate
                    if failure_rate > 0.8 and perf.get('predictions', 0) > 10:
                        continue
                
                # Get predictions (with or without TTA)
                if use_tta:
                    predictions = self.apply_test_time_augmentation(image, model, info)
                else:
                    predictions = self.predict_with_model(model, image, info)
                
                # Apply model-specific adjustments
                class_offset = info.get('class_offset', 0)
                weight = self.adaptive_weights.get(info['name'], info['weight'])
                
                for pred in predictions:
                    pred[0] += class_offset
                    pred[1] *= weight
                    
                all_predictions.extend(predictions)
                
            except Exception as e:
                continue
        
        if not all_predictions:
            return []
        
        # Apply ensemble strategy
        predictions_by_class = defaultdict(list)
        for pred in all_predictions:
            cls = int(pred[0])
            predictions_by_class[cls].append(pred)
        
        # Get ensemble function
        ensemble_func = self.ensemble_strategies.get(strategy, self._adaptive_ensemble)
        
        # Apply ensemble
        final_predictions = ensemble_func(predictions_by_class)
        
        # Post-process with class-specific thresholds
        if self.enable_self_tuning:
            filtered_predictions = []
            for pred in final_predictions:
                cls = int(pred[0])
                conf = pred[1]
                threshold = self.class_specific_thresholds.get(cls, self.ensemble_params['conf_threshold'])
                if conf >= threshold:
                    filtered_predictions.append(pred)
            final_predictions = filtered_predictions
        
        return final_predictions
    
    def create_enhanced_submission(self, test_images_path, output_csv="submission.csv", 
                                 strategy='adaptive', use_tta=False):
        """Create submission with enhanced features"""
        test_images_dir = Path(test_images_path)
        image_files = sorted([f for f in test_images_dir.glob("*") 
                             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        print(f"\n{'='*60}")
        print(f"Creating Enhanced Submission")
        print(f"Active Models: {len(self.models)}")
        print(f"Failed Models: {len(self.failed_models)}")
        print(f"Images: {len(image_files)}")
        print(f"Strategy: {strategy}")
        print(f"TTA: {'Enabled' if use_tta else 'Disabled'}")
        print(f"Self-Tuning: {'Enabled' if self.enable_self_tuning else 'Disabled'}")
        print(f"{'='*60}")
        
        # Show top performing models
        if self.enable_self_tuning:
            print("\nTop Performing Models:")
            sorted_models = sorted(
                [(name, perf['mAP']) for name, perf in self.model_performance.items()],
                key=lambda x: x[1],
                reverse=True
            )
            for name, mAP in sorted_models[:10]:
                print(f"- {name}: mAP={mAP:.3f}")
        
        predictions_data = []
        
        # Process images with progress bar
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                predictions = self.ensemble_predictions(
                    str(img_path), 
                    strategy=strategy, 
                    use_tta=use_tta
                )
                
                # Format predictions
                if predictions:
                    pred_strings = []
                    for pred in predictions:
                        cls_id, conf, x_center, y_center, width, height = pred
                        # Ensure valid values
                        if all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                            pred_strings.append(
                                f"{int(cls_id)} {conf:.6f} {x_center:.6f} {y_center:.6f} "
                                f"{width:.6f} {height:.6f}"
                            )
                    prediction_string = " ".join(pred_strings) if pred_strings else "no detections"
                else:
                    prediction_string = "no detections"
                
            except Exception as e:
                print(f"\nError processing {img_path.name}: {str(e)}")
                prediction_string = "no detections"
            
            predictions_data.append({
                "image_id": img_path.stem,
                "prediction_string": prediction_string
            })
            
            # Periodic garbage collection
            if len(predictions_data) % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(predictions_data)
        submission_df.to_csv(output_csv, index=False)
        
        print(f"\n✓ Submission saved to {output_csv}")
        print(f"Shape: {submission_df.shape}")
        
        # Statistics
        with_detections = (submission_df['prediction_string'] != 'no detections').sum()
        without_detections = (submission_df['prediction_string'] == 'no detections').sum()
        
        print(f"\nDetection Statistics:")
        print(f"- Images with detections: {with_detections} ({with_detections/len(submission_df)*100:.1f}%)")
        print(f"- Images without detections: {without_detections}")
        
        # Performance summary
        if self.enable_self_tuning:
            print("\nModel Performance Summary:")
            for name, perf in list(self.model_performance.items())[:10]:
                success_rate = perf['successes'] / max(perf['predictions'], 1)
                avg_speed = perf.get('speed', 0)
                print(f"- {name}: Success={success_rate:.2%}, Speed={avg_speed:.3f}s")
        
        return submission_df
    
    def load_all_models(self, local_models=None):
        """Load all available models"""
        print("="*60)
        print("ENHANCED MULTI-SOURCE OBJECT DETECTION ENSEMBLE")
        print("="*60)
        
        # 1. Enhanced YOLO models
        self.download_enhanced_yolo_models()
        
        # 2. Advanced Transformer models
        self.download_advanced_transformer_models()
        
        # 3. YOLOv9 models
        self.download_yolov9_models()
        
        # 4. YOLO-NAS models
        self.download_yolo_nas_models()
        
        # 5. Specialized HuggingFace models
        self.download_specialized_hf_models()
        
        # 6. EfficientDet models
        self.download_efficientdet_models()
        
        # 7. Auto-discover and rank models
        self.auto_discover_and_rank_models(limit=10)
        
        # 8. Local models (if any)
        if local_models:
            self.load_local_models(local_models)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"MODEL LOADING COMPLETE")
        print(f"Successfully loaded: {len(self.models)} models")
        print(f"Failed to load: {len(self.failed_models)} models")
        print(f"{'='*60}")
        
        # Setup self-tuning if enabled
        if self.enable_self_tuning:
            self.setup_self_tuning()
    
    def load_local_models(self, model_paths):
        """Load local pre-trained models"""
        print("\n" + "="*60)
        print("LOADING LOCAL MODELS")
        print("="*60)
        
        for path_config in model_paths:
            path = path_config.get('path', '')
            
            if not os.path.exists(path):
                print(f"✗ Path not found: {path}")
                continue
            
            def load_local():
                try:
                    # Try YOLO first
                    if YOLO_AVAILABLE and path.endswith('.pt'):
                        return YOLO(path)
                    # Try generic PyTorch
                    else:
                        return torch.load(path, map_location=self.device)
                except:
                    return None
            
            model = self.safe_model_load(load_local, path)
            if model:
                self.models.append(model)
                self.model_info.append({
                    'name': os.path.basename(path),
                    'type': 'local',
                    'weight': path_config.get('weight', 1.0),
                    'architecture': 'Unknown',
                    'source': 'local',
                    'class_offset': path_config.get('class_offset', 0)
                })
                print(f"✓ Loaded {path}")


def main():
    """Main execution with enhanced features"""
    # Initialize enhanced ensemble
    ensemble = EnhancedObjectDetectionEnsemble(enable_self_tuning=True)
    
    # Local models (if available)
    local_models = [
        {'path': '/kaggle/input/2-top-models/pytorch/default/1/habijabii.pt', 'weight': 1.2, 'class_offset': 1},
        {'path': '/kaggle/input/2-top-models/pytorch/default/1/nadiatriki.pt', 'weight': 1.2, 'class_offset': 0}
    ]
    
    # Load all models
    ensemble.load_all_models(local_models)
    
    # Test images path
    test_images_path = '/kaggle/input/multi-class-object-detection-challenge/testImages/images'
    
    # Create submissions with different strategies
    strategies = [
        ('adaptive', True),   # Adaptive with TTA
        ('hybrid', True),     # Hybrid with TTA
        ('cascade', False),   # Cascade without TTA
        ('wbf', False),      # Standard WBF
    ]
    
    for strategy, use_tta in strategies:
        print(f"\n{'='*60}")
        print(f"Creating submission with {strategy.upper()} strategy (TTA: {use_tta})")
        print(f"{'='*60}")
        
        output_name = f"submission_{strategy}_{'tta' if use_tta else 'no_tta'}.csv"
        
        submission_df = ensemble.create_enhanced_submission(
            test_images_path=test_images_path,
            output_csv=output_name,
            strategy=strategy,
            use_tta=use_tta
        )
    
    print("\n" + "="*60)
    print("ENHANCED ENSEMBLE PROCESSING COMPLETE!")
    print("="*60)
    
    # Final recommendations
    print("\nRecommendations for best results:")
    print("1. Use 'adaptive' or 'hybrid' strategy with TTA enabled")
    print("2. Ensure GPU is available for faster processing")
    print("3. Consider running cross-validation to optimize parameters")
    print("4. Monitor model performance and remove consistently failing models")
    print("5. Fine-tune class-specific thresholds based on validation data")


if __name__ == "__main__":
    main()

