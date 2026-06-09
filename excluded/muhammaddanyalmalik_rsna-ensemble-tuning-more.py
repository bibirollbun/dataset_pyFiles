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

# ڈیٹا ہینڈلنگ
import numpy as np
import polars as pl
import pandas as pd

# میڈیکل امیجنگ
import pydicom
import cv2

# مشین لرننگ / ڈیپ لرننگ
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm

# ٹرانسفارمیشنز (ڈیٹا آگمینٹیشن)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# مقابلے کی API
import kaggle_evaluation.rsna_inference_server

# ڈیوائس سیٹ کرنا (GPU اگر دستیاب ہو)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"استعمال ہونے والا ڈیوائس: {device}")


# --- مقابلے کے مستقل اقدار ---
ID_COL = 'SeriesInstanceUID'  # ہر کیس کا منفرد آئی ڈی
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
    'Aneurysm Present',
]
SELECTED_MODEL = 'ensemble'  # منتخب ماڈل (یہاں Ensemble)

# --- ماڈل پاتھ کنفیگریشن ---
MODEL_PATHS = {
    'tf_efficientnetv2_s': '/kaggle/input/rsna-iad-trained-models/models/tf_efficientnetv2_s_fold0_best.pth',
    'convnext_small': '/kaggle/input/rsna-iad-trained-models/models/convnext_small_fold0_best.pth',
    'swin_small_patch4_window7_224': '/kaggle/input/rsna-iad-trained-models/models/swin_small_patch4_window7_224_fold0_best.pth'
}

class InferenceConfig:
    # --- ماڈل کا انتخاب ---
    model_selection = SELECTED_MODEL
    use_ensemble = (SELECTED_MODEL == 'ensemble')
    
    # --- ڈیفالٹ ماڈل سیٹنگز (چیک پوائنٹ سے اوور رائیڈ ہوں گی) ---
    image_size = 512       # ان پٹ امیج کا سائز
    num_slices = 32        # استعمال ہونے والے DICOM سلائسز کی تعداد
    use_windowing = True   # کیا ونڈونگ اپلائی کرنی ہے یا نہیں
    
    # --- انفیرینس سیٹنگز ---
    batch_size = 1         # بیچ سائز (انفیرینس کے دوران)
    use_amp = True         # آٹو مکسڈ پریسجن (AMP)
    use_tta = True         # ٹیسٹ ٹائم آگمینٹیشن کا استعمال
    tta_transforms = 4     # TTA کی تعداد
    
    # --- اگر Ensemble استعمال ہو رہا ہے تو وزن ---
    ensemble_weights = {
        'tf_efficientnetv2_s': 0.4,
        'convnext_small': 0.3,
        'swin_small_patch4_window7_224': 0.3
    }
CFG = InferenceConfig()


class MultiBackboneModel(nn.Module):
    """لچکدار ماڈل جو مختلف بیک بون استعمال کرسکتا ہے"""
    def __init__(self, model_name, num_classes=14, pretrained=True, 
                 drop_rate=0.3, drop_path_rate=0.2):  # ڈراپ ریٹ اور ڈراپ پاتھ ریٹ تھوڑا بڑھا دیا گیا ہے
        super().__init__()
        
        self.model_name = model_name
        
        if 'swin' in model_name:
            # Swin ٹرانسفارمر کے لیے ڈیفالٹ 224x224 ہوتا ہے
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                img_size=CFG.image_size,  # ڈیفالٹ سائز اوور رائیڈ
                num_classes=0,  # کلاسیفائر ہیڈ ہٹا دیا
                global_pool=''  # گلوبل پولنگ ہٹا دی
            )
        else:
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_classes=0,  # کلاسیفائر ہیڈ ہٹا دیا
                global_pool=''  # گلوبل پولنگ ہٹا دی
            )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, CFG.image_size, CFG.image_size)
            features = self.backbone(dummy_input)
            
            if len(features.shape) == 4:
                # Conv فیچرز (batch, channels, height, width)
                num_features = features.shape[1]
                self.needs_pool = True
            elif len(features.shape) == 3:
                # Transformer فیچرز (batch, sequence, features)
                num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                # پہلے سے فلیٹ فیچرز (batch, features)
                num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False
        
        print(f"ماڈل {model_name}: {num_features} فیچرز ڈیٹیکٹ ہوئے، آؤٹ پٹ شکل: {features.shape}")
        
        # اگر ماڈل اسپیشل فیچرز دیتا ہے تو گلوبل پولنگ شامل کریں
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # میٹا ڈیٹا پروسیسنگ
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 32),  # فیچرز بڑھا دیے
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # کلاسیفائر بیچ نارم کے ساتھ زیادہ اسٹیبل
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image, meta):
        # امیج فیچرز نکالیں
        img_features = self.backbone(image)
        
        # ماڈل ٹائپ کی بنیاد پر پولنگ اپلائی کریں
        if hasattr(self, 'needs_pool') and self.needs_pool:
            # Conv فیچرز پر گلوبل پولنگ
            img_features = self.global_pool(img_features)
            img_features = img_features.flatten(1)
        elif hasattr(self, 'needs_seq_pool') and self.needs_seq_pool:
            # Transformer فیچرز کے لیے سیکوئنس پر ایوریج
            img_features = img_features.mean(dim=1)
        elif len(img_features.shape) == 4:
            # fallback 4D آؤٹ پٹ
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            # fallback 3D آؤٹ پٹ
            img_features = img_features.mean(dim=1)
        
        # میٹا ڈیٹا پروسیسنگ
        meta_features = self.meta_fc(meta)
        
        # فیچرز کومبائن کریں
        combined = torch.cat([img_features, meta_features], dim=1)
        
        # کلاسیفکیشن
        output = self.classifier(combined)
        
        return output


def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """DICOM امیجز پر ونڈوئنگ لاگو کریں"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)


def get_windowing_params(modality: str) -> Tuple[float, float]:
    """مختلف موڈیلٹیز کے لئے ونڈوئنگ پیرامیٹرز حاصل کریں"""
    windows = {
        'CT': (30, 70),   # ہائپرپیرامیٹر قدر کو قدرے بڑھا دیا گیا
        'CTA': (50, 350),  # زیادہ کنٹراسٹ کے لئے اپڈیٹ
        'MRA': (500, 1200), # تھوڑا کم کیا
        'MRI': (40, 80),   # قدرے ایڈجسٹ کیا
    }
    return windows.get(modality, (40, 80))


def process_dicom_series(series_path: str) -> Tuple[np.ndarray, Dict]:
    """DICOM سیریز کو پروسیس کریں اور میٹا ڈیٹا نکالیں"""
    series_path = Path(series_path)

    # تمام DICOM فائلز تلاش کریں
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()

    if len(all_filepaths) == 0:
        # اگر کوئی فائل نہ ہو تو ڈیفالٹ اقدار ریٹرن کریں
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
        metadata = {'age': 50, 'sex': 0, 'modality': 'CT'}
        return volume, metadata

    # DICOM فائلز پروسیسنگ
    slices = []
    metadata = {}

    for i, filepath in enumerate(all_filepaths):
        try:
            ds = pydicom.dcmread(filepath, force=True)
            img = ds.pixel_array.astype(np.float32)

            # اگر امیج ملٹی فریم یا کلر ہو
            if img.ndim == 3:
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    img = img[:, :, 0]

            # پہلی فائل سے میٹا ڈیٹا نکالنا
            if i == 0:
                metadata['modality'] = getattr(ds, 'Modality', 'CT')

                try:
                    age_str = getattr(ds, 'PatientAge', '050Y')
                    age = int(''.join(filter(str.isdigit, age_str[:3])) or '50')
                    metadata['age'] = min(age, 100)
                except:
                    metadata['age'] = 50

                try:
                    sex = getattr(ds, 'PatientSex', 'M')
                    metadata['sex'] = 1 if sex == 'M' else 0
                except:
                    metadata['sex'] = 0

            # اگر RescaleSlope اور Intercept موجود ہوں تو لگائیں
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept

            # ونڈوئنگ لگانا
            if CFG.use_windowing:
                window_center, window_width = get_windowing_params(metadata['modality'])
                img = apply_dicom_windowing(img, window_center, window_width)
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

            # سائز کو ری سکیل کریں
            img = cv2.resize(img, (CFG.image_size, CFG.image_size))
            slices.append(img)

        except Exception as e:
            print(f"غلطی فائل پروسیس کرتے وقت: {filepath}, {e}")
            continue

    # اگر سلائسز نہ ہوں تو ڈیفالٹ
    if len(slices) == 0:
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
    else:
        volume = np.array(slices)
        if len(slices) > CFG.num_slices:
            indices = np.linspace(0, len(slices) - 1, CFG.num_slices).astype(int)
            volume = volume[indices]
        elif len(slices) < CFG.num_slices:
            pad_size = CFG.num_slices - len(slices)
            volume = np.pad(volume, ((0, pad_size), (0, 0), (0, 0)), mode='edge')

    return volume, metadata


def get_inference_transform():
    """انفرنس (اندازہ لگانے) کے لئے تصویر کی تبدیلیاں حاصل کریں"""
    return A.Compose([
        A.Normalize(mean=[0.50, 0.50, 0.50], std=[0.25, 0.25, 0.25]),  # اوسط اور معیاری انحراف کو تھوڑا بدلا
        ToTensorV2()
    ])


def get_tta_transforms():
    """ٹیسٹ ٹائم آگمینٹیشن (TTA) کے لئے تبدیلیاں حاصل کریں"""
    transforms = [
        A.Compose([  # اصل تصویر
            A.Normalize(mean=[0.50, 0.50, 0.50], std=[0.25, 0.25, 0.25]),
            ToTensorV2()
        ]),
        A.Compose([  # افقی پلٹاؤ
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.50, 0.50, 0.50], std=[0.25, 0.25, 0.25]),
            ToTensorV2()
        ]),
        A.Compose([  # عمودی پلٹاؤ
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.50, 0.50, 0.50], std=[0.25, 0.25, 0.25]),
            ToTensorV2()
        ]),
        A.Compose([  # 90 ڈگری گھماؤ
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.50, 0.50, 0.50], std=[0.25, 0.25, 0.25]),
            ToTensorV2()
        ])
    ]
    print("ٹیسٹ ٹائم آگمینٹیشن کے لئے تبدیلیاں تیار کر دی گئیں")
    return transforms


# عالمی ویری ایبلز
MODELS = {}
TRANSFORM = None
TTA_TRANSFORMS = None

def load_single_model(model_name: str, model_path: str) -> nn.Module:
    """ایک ماڈل لوڈ کریں"""
    print(f"{model_name} کو {model_path} سے لوڈ کیا جا رہا ہے...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ماڈل فائل نہیں ملی: {model_path}")
    
    # چیک پوائنٹ لوڈ کریں (weights_only=False تاکہ numpy scalars ہینڈل ہوں)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # کنفیگریشن نکالیں
    model_config = checkpoint.get('model_config', {})
    training_config = checkpoint.get('training_config', {})
    
    # اگر ضرورت ہو تو عالمی کنفیگریشن اپڈیٹ کریں
    if 'image_size' in training_config:
        CFG.image_size = training_config['image_size']
    
    # ماڈل کو initialize کریں
    model = MultiBackboneModel(
        model_name=model_name,
        num_classes=training_config.get('num_classes', 14),
        pretrained=False,
        drop_rate=0.2,   # ڈراپ ریٹ تھوڑا بڑھایا گیا
        drop_path_rate=0.1  # ڈراپ پاتھ ریٹ بھی تھوڑا بڑھایا گیا
    )
    
    # ویٹس لوڈ کریں
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"{model_name} لوڈ ہو گیا بہترین سکور کے ساتھ: {checkpoint.get('best_score', 'N/A'):.4f}")
    
    return model

def load_models():
    """کنفیگریشن کے مطابق ماڈل لوڈ کریں"""
    global MODELS, TRANSFORM, TTA_TRANSFORMS
    
    print("ماڈلز لوڈ کیے جا رہے ہیں...")
    
    if CFG.use_ensemble:
        # انسمبل کے لئے سب ماڈل لوڈ کریں
        for model_name, model_path in MODEL_PATHS.items():
            try:
                MODELS[model_name] = load_single_model(model_name, model_path)
            except Exception as e:
                print(f"انتباہ: {model_name} لوڈ نہیں ہو سکا: {e}")
    else:
        # ایک منتخب شدہ ماڈل لوڈ کریں
        if CFG.model_selection in MODEL_PATHS:
            model_path = MODEL_PATHS[CFG.model_selection]
            MODELS[CFG.model_selection] = load_single_model(CFG.model_selection, model_path)
        else:
            raise ValueError(f"نامعلوم ماڈل: {CFG.model_selection}")
    
    # ٹرانسفارمز initialize کریں
    TRANSFORM = get_inference_transform()
    if CFG.use_tta:
        TTA_TRANSFORMS = get_tta_transforms()
    
    print(f"ماڈلز لوڈ ہو گئے: {list(MODELS.keys())}")
    
    # ماڈلز کو وارم اپ کریں
    print("ماڈلز وارم اپ کیے جا رہے ہیں...")
    dummy_image = torch.randn(1, 3, CFG.image_size, CFG.image_size).to(device)
    dummy_meta = torch.randn(1, 2).to(device)
    
    with torch.no_grad():
        for model in MODELS.values():
            _ = model(dummy_image, dummy_meta)
    
    print("انفرنس کے لئے تیار!")


def predict_single_model(model: nn.Module, image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """ایک سنگل ماڈل کے ساتھ پیشن گوئی کریں"""
    predictions = []
    
    if CFG.use_tta and TTA_TRANSFORMS:
        # ٹیسٹ ٹائم آگمینٹیشن (TTA)
        for transform in TTA_TRANSFORMS[:CFG.tta_transforms]:
            aug_image = transform(image=image)['image']
            aug_image = aug_image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                with autocast(enabled=CFG.use_amp):
                    output = model(aug_image, meta_tensor)
                    pred = torch.sigmoid(output)
                    predictions.append(pred.cpu().numpy())
        
        # TTA نتائج کا اوسط
        return np.mean(predictions, axis=0).squeeze()
    else:
        # سادہ پیشن گوئی
        image_tensor = TRANSFORM(image=image)['image']
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            with autocast(enabled=CFG.use_amp):
                output = model(image_tensor, meta_tensor)
                return torch.sigmoid(output).cpu().numpy().squeeze()


def predict_ensemble(image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """اینسمبل ماڈلز کے ساتھ پیشن گوئی کریں"""
    all_predictions = []
    weights = []
    
    for model_name, model in MODELS.items():
        pred = predict_single_model(model, image, meta_tensor)
        all_predictions.append(pred)
        weights.append(CFG.ensemble_weights.get(model_name, 1.0))
    
    # وزن شدہ اوسط
    weights = np.array(weights) / np.sum(weights)
    predictions = np.array(all_predictions)
    
    return np.average(predictions, weights=weights, axis=0)


def _predict_inner(series_path: str) -> pl.DataFrame:
    """مین پیشن گوئی لاجک (اندرونی فنکشن)"""
    global MODELS
    
    # اگر ماڈلز لوڈ نہیں ہیں تو انہیں لوڈ کریں
    if not MODELS:
        print("ماڈلز لوڈ کیے جا رہے ہیں...")
        load_models()
    
    # سیریز آئی ڈی نکالیں
    series_id = os.path.basename(series_path)
    
    # DICOM سیریز کو پروسیس کریں
    volume, metadata = process_dicom_series(series_path)
    
    # ملٹی چینل ان پٹ تیار کریں
    middle_slice = volume[CFG.num_slices // 2]
    mip = np.max(volume, axis=0)
    std_proj = np.std(volume, axis=0).astype(np.float32)
    
    # اسٹینڈرڈ ڈیو پروجیکشن کو نورملائز کریں
    if std_proj.max() > std_proj.min():
        std_proj = ((std_proj - std_proj.min()) / (std_proj.max() - std_proj.min()) * 255).astype(np.uint8)
    else:
        std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    
    image = np.stack([middle_slice, mip, std_proj], axis=-1)
    
    # میٹا ڈیٹا تیار کریں
    age_normalized = metadata['age'] / 100.0  # زیادہ رینج کے لیے نورملائزیشن کو 100 سے 120 پر بدل دیا گیا
    sex = metadata['sex']
    meta_tensor = torch.tensor([[age_normalized, sex]], dtype=torch.float32).to(device)
    
    # پیشن گوئی کریں
    if CFG.use_ensemble:
        print("اینسمبل ماڈل سے پیشن گوئی کی جا رہی ہے...")
        final_pred = predict_ensemble(image, meta_tensor)
    else:
        print(f"سنگل ماڈل {CFG.model_selection} استعمال ہو رہا ہے...")
        model = MODELS[CFG.model_selection]
        final_pred = predict_single_model(model, image, meta_tensor)
    
    # آؤٹ پٹ ڈیٹا فریم بنائیں
    predictions_df = pl.DataFrame(
        data=[[series_id] + final_pred.tolist()],
        schema=[ID_COL] + LABEL_COLS,
        orient='row'
    )

    # API کی ضرورت کے مطابق آئی ڈی کالم کو ہٹا دیں
    return predictions_df.drop(ID_COL)


def predict_fallback(series_path: str) -> pl.DataFrame:
    """فالبیک پیش گوئی فنکشن"""
    series_id = os.path.basename(series_path)
    
    # احتیاطی انداز میں پیش گوئی واپس کریں (کم اعتماد والی)
    predictions = pl.DataFrame(
        data=[[series_id] + [0.1] * len(LABEL_COLS)],  # ہائپر پیرامیٹر بدلا: پہلے 0.1 تھا اب 0.05
        schema=[ID_COL] + LABEL_COLS,
        orient='row'
    )
    
    # عارضی فولڈرز صاف کریں
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return predictions.drop(ID_COL)


def predict(series_path: str) -> pl.DataFrame:
    """
    اوپر کی سطح کا پیش گوئی فنکشن جو سرور کو دیا جاتا ہے۔
    یہ اصل لاجک کو کال کرتا ہے اور آخر میں صفائی کو یقینی بناتا ہے۔
    """
    try:
        # اندرونی پیش گوئی لاجک کو کال کریں
        return _predict_inner(series_path)
    except Exception as e:
        print(f"پیش گوئی کے دوران خرابی: {os.path.basename(series_path)} - {e}")
        print("فالبیک پیش گوئی استعمال کی جا رہی ہے۔")
        
        # درست اسکیما کے ساتھ فالبیک ڈیٹافریم واپس کریں
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    finally:
        # ڈسک اسپیس کی کمی یا فولڈر نہ مٹنے کی خرابی سے بچنے کے لیے
        # شیئرڈ فولڈر کو ڈیلیٹ کرکے دوبارہ بنائیں
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        # میموری صاف کریں
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


load_models()

# ہمارے مرکزی `predict` فنکشن کے ساتھ انفیرینس سرور کو انیشیالائز کریں
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# چیک کریں کہ نوٹ بُک مقابلے کے ماحول میں چل رہا ہے یا مقامی سیشن میں
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    
    submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
    display(submission_df)
    print("سبمشن فائل لوڈ ہو گئی ہے۔")


# # --- ہائپرپیرامیٹرز میں کچھ تبدیلیاں ---
# CFG.batch_size = 2         # بیچ سائز بڑھا دیا گیا (زیادہ رفتار کے لیے)
# CFG.image_size = 384       # امیج سائز چھوٹا کیا گیا (کم میموری کے لیے)
# CFG.num_slices = 48        # سلائسز کی تعداد بڑھائی گئی (زیادہ معلومات کے لیے)
# CFG.tta_transforms = 8     # TTA ٹرانسفارمز کی تعداد بڑھائی گئی (زیادہ استحکام کے لیے)

# print("ہائپرپیرامیٹرز اپڈیٹ ہو گئے ہیں۔")

