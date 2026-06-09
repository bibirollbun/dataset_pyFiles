# ====================================================
# CELL 1: IMPORTS & CONFIG
# ====================================================

import os
import shutil
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import timm
import cv2
import pydicom
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import label, center_of_mass
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import kaggle_evaluation.rsna_inference_server
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Competition Configuration
class Config:
    # Paths
    TRAIN_CSV_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv'
    SERIES_DIR = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/'
    SEGMENTATION_DIR = '/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/'
    STAGE1_MODEL_PATH = '/kaggle/input/pytorch-aneurysmnet-intracranial-e15-nb153/pytorch/default/6/stage1_segmentation_best.pth'
    
    # Stage 2 Configuration
    ROI_SIZE = (224, 224)
    ROIS_PER_SERIES = 5
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    N_FOLDS = 5
    
    # Competition constants
    ID_COL = 'SeriesInstanceUID'
    LABEL_COLS = [
        'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery', 'Anterior Communicating Artery',
        'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery',
        'Left Posterior Communicating Artery', 'Right Posterior Communicating Artery',
        'Basilar Tip', 'Other Posterior Circulation', 'Aneurysm Present',
    ]
    
    # Device and training
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIXED_PRECISION = True
    STAGE2_CACHE_DIR = '/kaggle/working/stage2_cache'
    # Optional: reuse Stage 1 external cache volumes directly for exact preprocessing parity
    STAGE1_EXTERNAL_CACHE_DIR = '/kaggle/input/rsna2025-v2-intracranial-aneurysm-detection-nb153/stage1_AneurysmNet_prebuilt_v2'  # e.g., '/kaggle/input/rsna2025aneurysmnetprebuildnb153/stage1_AneurysmNet_prebuilt'
    
    # Debug
    DEBUG_MODE = False
    DEBUG_SAMPLES = 0
    # Cache/throughput
    REUSE_EXISTING_ROIS = True  # if cached training_df exists, reuse to skip long ROI extraction

print(f"âœ… Configuration loaded - Device: {Config.DEVICE}")

# Speed-friendly backend settings
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# ====================================================
# CELL 1.5: CUSTOM 3D UNET (REPLACES MONAI BASICUNET)
# ====================================================

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, base=24):
        super().__init__()
        b = base
        self.enc1 = ConvBlock3D(in_ch, b)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(b, b*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(b*2, b*4)
        self.pool3 = nn.MaxPool3d(2)
        self.bott = ConvBlock3D(b*4, b*8)
        self.up3  = nn.ConvTranspose3d(b*8, b*4, 2, stride=2); self.dec3 = ConvBlock3D(b*8, b*4)
        self.up2  = nn.ConvTranspose3d(b*4, b*2, 2, stride=2); self.dec2 = ConvBlock3D(b*4, b*2)
        self.up1  = nn.ConvTranspose3d(b*2, b, 2, stride=2);   self.dec1 = ConvBlock3D(b*2, b)
        self.seg_head = nn.Conv3d(b, 1, 1)
        self.cls_pool = nn.AdaptiveAvgPool3d(1); self.cls_head = nn.Linear(b*8, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2)); b = self.bott(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg = self.seg_head(d1); cls = self.cls_head(self.cls_pool(b).flatten(1))
        return seg, cls

class CustomTransforms:
    """Pure PyTorch transforms to replace MONAI transforms"""
    
    def __init__(self, keys=['volume']):
        self.keys = keys
        
    def __call__(self, data_dict):
        """Apply transforms to data dictionary"""
        result = {}
        
        for key in data_dict:
            if key in self.keys:
                # Convert numpy array to tensor if needed
                if isinstance(data_dict[key], np.ndarray):
                    result[key] = torch.from_numpy(data_dict[key]).float()
                else:
                    result[key] = data_dict[key]
            else:
                result[key] = data_dict[key]
        
        return result

print("âœ… Custom 3D UNet and transforms loaded (MONAI-free!)")


# ====================================================
# CELL 2: DATA LOADING & ROI EXTRACTION
# ====================================================

class Simple3DSegmentationNet(nn.Module):
    """Wrapper to match Stage 1 UNet3D interface for Stage 3 loading."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.unet = UNet3D(in_ch=in_channels, base=24)
    def forward(self, x):
        return self.unet(x)
        
class SimpleDICOMProcessor:
    """Simplified DICOM processor aligned with Stage 1"""
    def __init__(self, target_size=(48, 112, 112)):
        self.target_size = target_size
        # Prefer working prebuilt if available, else input dataset path
        working_prebuilt = '/kaggle/input/rsna2025-v2-intracranial-aneurysm-detection-nb153/stage1_AneurysmNet_prebuilt_v2'
        default_prebuilt = getattr(Config, 'STAGE1_EXTERNAL_CACHE_DIR', '')
        if os.path.isdir(working_prebuilt):
            self.external_cache_dir = working_prebuilt
        else:
            self.external_cache_dir = default_prebuilt
        # brainmask path (v2 prebuilt has brainmasks/ with npz files)
        self.brainmask_key = 'm'
        
    def load_dicom_series(self, series_path):
        """Load DICOM series with Stage 1-style preprocessing (CT HU windowing + isotropic resample)."""
        try:
            # Prefer loading from Stage 1 external cache if available
            if self.external_cache_dir:
                sid = os.path.basename(series_path.rstrip('/'))
                cand = [
                    os.path.join(self.external_cache_dir, 'volumes', f'{sid}.npy'),
                    os.path.join(self.external_cache_dir, 'volumes', f'{sid}.npy.tmp.npy'),
                ]
                for p in cand:
                    if os.path.exists(p):
                        try:
                            vol = np.load(p, allow_pickle=False, mmap_mode='r')
                            # Ensure correct target size
                            if vol.shape != self.target_size:
                                target_d, target_h, target_w = self.target_size
                                D, H, W = vol.shape
                                if D != target_d:
                                    idx = np.linspace(0, max(D - 1, 0), num=target_d).astype(int) if D > 0 else np.zeros(target_d, dtype=int)
                                    vol = vol[idx]
                                if (H, W) != (target_h, target_w):
                                    resized = np.empty((target_d, target_h, target_w), dtype=np.float32)
                                    for i in range(target_d):
                                        resized[i] = cv2.resize(vol[i].astype(np.float32), (target_w, target_h))
                                    vol = resized
                            # Optional brainmask gating (as in Stage 1 training)
                            bm_path = os.path.join(self.external_cache_dir, 'brainmasks', f'{sid}_brainmask.npz')
                            if os.path.exists(bm_path):
                                try:
                                    bm = np.load(bm_path)[self.brainmask_key].astype(np.float32)
                                    if bm.shape == vol.shape and bm.sum() > 0:
                                        vol = (np.asarray(vol, dtype=np.float32) * bm)
                                except Exception:
                                    pass
                            if Config.DEBUG_MODE:
                                print(f"DEBUG: Loaded volume from external cache {p}, stats: min={float(np.min(vol)):.3f}, max={float(np.max(vol)):.3f}, mean={float(np.mean(vol)):.3f}")
                            return vol.astype(np.float32)
                        except Exception:
                            pass

            # Collect DICOMs
            dicoms = []
            for root, _, files in os.walk(series_path):
                for f in files:
                    if f.endswith('.dcm'):
                        try:
                            ds = pydicom.dcmread(os.path.join(root, f), force=True)
                            if hasattr(ds, 'PixelData'):
                                dicoms.append(ds)
                        except Exception:
                            continue
            if not dicoms:
                return np.zeros(self.target_size, dtype=np.float32)

            # Sort by orientation vector dot IPP; fallback InstanceNumber
            try:
                orient = np.array(dicoms[0].ImageOrientationPatient, dtype=np.float32)
                row = orient[:3]; col = orient[3:]
                normal = np.cross(row, col)
                def sort_key(ds):
                    ipp = np.array(getattr(ds, 'ImagePositionPatient', [0,0,0]), dtype=np.float32)
                    return float(np.dot(ipp, normal))
                dicoms = sorted(dicoms, key=sort_key)
            except Exception:
                dicoms = sorted(dicoms, key=lambda ds: getattr(ds, 'InstanceNumber', 0))

            # Spacing
            try:
                dy, dx = map(float, dicoms[0].PixelSpacing)
            except Exception:
                ps = getattr(dicoms[0], 'PixelSpacing', [1.0, 1.0])
                dy, dx = float(ps[0]), float(ps[1])
            zs = []
            for i in range(1, len(dicoms)):
                p0 = np.array(getattr(dicoms[i-1], 'ImagePositionPatient', [0,0,0]), dtype=np.float32)
                p1 = np.array(getattr(dicoms[i], 'ImagePositionPatient', [0,0,0]), dtype=np.float32)
                d = np.linalg.norm(p1 - p0)
                if d > 0:
                    zs.append(d)
            dz = float(np.median(zs)) if zs else float(getattr(dicoms[0], 'SliceThickness', 1.0))
            dz = dz if (dz > 0 and np.isfinite(dz)) else 1.0
            dy = dy if (dy > 0 and np.isfinite(dy)) else 1.0
            dx = dx if (dx > 0 and np.isfinite(dx)) else 1.0

            # Build volume with HU and CT windowing
            base_h = int(getattr(dicoms[0], 'Rows', 256))
            base_w = int(getattr(dicoms[0], 'Columns', 256))
            vol_slices = []
            modality = (getattr(dicoms[0], 'Modality', '') or '').upper()
            c = 300.0; w = 700.0
            lo, hi = c - w/2.0, c + w/2.0
            for ds in dicoms:
                try:
                    arr = ds.pixel_array
                except Exception:
                    continue
                if arr.ndim >= 3:
                    h, w2 = arr.shape[-2], arr.shape[-1]
                    frames = arr.reshape(int(np.prod(arr.shape[:-2])), h, w2)
                else:
                    frames = arr[np.newaxis, ...]
                for sl in frames:
                    sl = sl.astype(np.float32)
                    if getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
                        sl = sl.max() - sl
                    slope = float(getattr(ds, 'RescaleSlope', 1.0)); intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                    sl = sl * slope + intercept
                    if sl.shape != (base_h, base_w):
                        sl = cv2.resize(sl, (base_w, base_h))
                    if modality == 'CT':
                        s = np.clip(sl, lo, hi)
                        s = (s - lo) / (hi - lo + 1e-6)
                    else:
                        mean = float(sl.mean()); std = float(sl.std() + 1e-6)
                        s = (sl - mean) / std; zc = 3.0
                        s = np.clip(s, -zc, zc); s = (s + zc) / (2.0*zc)
                    vol_slices.append(s.astype(np.float32))
            if not vol_slices:
                return np.zeros(self.target_size, dtype=np.float32)

            volume = np.stack(vol_slices, axis=0).astype(np.float32)
            # Isotropic resample to 1.0 mm
            z, y, x = volume.shape
            newD = max(1, int(round(z * dz / 1.0)))
            newH = max(1, int(round(y * dy / 1.0)))
            newW = max(1, int(round(x * dx / 1.0)))
            volume = ndimage.zoom(volume, (newD / z, newH / y, newW / x), order=1)
            # Resize to target grid
            target_d, target_h, target_w = self.target_size
            D, H, W = volume.shape
            if D != target_d:
                idx = np.linspace(0, max(D - 1, 0), num=target_d).astype(int) if D > 0 else np.zeros(target_d, dtype=int)
                volume = volume[idx]
            if (H, W) != (target_h, target_w):
                resized = np.empty((target_d, target_h, target_w), dtype=np.float32)
                for i in range(target_d):
                    resized[i] = cv2.resize(volume[i].astype(np.float32), (target_w, target_h))
                volume = resized
            return volume.astype(np.float32)
        except Exception as e:
            print(f"Failed to load {series_path}: {e}")
            return np.zeros(self.target_size, dtype=np.float32)
    
    def preprocess_volume(self, volume):
        """Simple preprocessing (match Stage 1)"""
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        denom = (p99 - p1) if (p99 - p1) > 1e-6 else 1e-6
        volume = (volume - p1) / denom
        
        if volume.shape != self.target_size:
            target_d, target_h, target_w = self.target_size
            D, H, W = volume.shape
            if D != target_d:
                idx = np.linspace(0, max(D - 1, 0), num=target_d).astype(int) if D > 0 else np.zeros(target_d, dtype=int)
                volume = volume[idx]
            if (H, W) != (target_h, target_w):
                resized = np.empty((target_d, target_h, target_w), dtype=np.float32)
                for i in range(target_d):
                    resized[i] = cv2.resize(volume[i].astype(np.float32), (target_w, target_h))
                volume = resized
        
        return volume.astype(np.float32)

class Stage1Predictor:
    """Load and use Stage 1 model for ROI extraction"""
    def __init__(self, model_path):
        self.device = Config.DEVICE
        self.processor = SimpleDICOMProcessor()
        
        # Load Stage 1 model (exact UNet used in Stage 1 training)
        print("Loading Stage 1 model...")
        self.model = UNet3D(in_ch=1, base=24).to(self.device)
        
        try:
            preferred = '/kaggle/input/pytorch-aneurysmnet-intracranial-e15-nb153/pytorch/default/6/stage1_segmentation_best.pth'
            load_path = preferred if os.path.exists(preferred) else model_path
            if load_path != model_path:
                print(f"Using Stage 1 checkpoint from working dir: {preferred}")
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
            print(f"Loaded Stage 1 checkpoint from: {load_path}")
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel wrapper
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            
            # Diagnostic: count matched keys and shapes
            model_state = self.model.state_dict()
            matched, total = 0, 0
            for k, v in model_state.items():
                total += 1
                if k in state_dict and state_dict[k].shape == v.shape:
                    matched += 1
            match_ratio = matched / max(1, total)
            print(f"Stage 1 checkpoint match ratio: {match_ratio:.2%} ({matched}/{total})")
            if matched == 0:
                # Try key remap: strip optional 'unet.' prefix from checkpoint or add if needed
                remapped = {}
                if any(k.startswith('unet.') for k in state_dict.keys()):
                    remapped = {k.replace('unet.', '', 1): v for k, v in state_dict.items()}
                else:
                    remapped = state_dict
                # Recompute match
                matched = 0
                for k, v in model_state.items():
                    if k in remapped and remapped[k].shape == v.shape:
                        matched += 1
                print(f"Remapped match: {matched}/{total}")
                state_dict = remapped
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print("âœ… Stage 1 model loaded successfully")
        except Exception as e:
            print(f"â�Œ Error loading Stage 1 model: {e}")
            self.model = None
    
    def predict_segmentation(self, series_path):
        """Get segmentation mask from Stage 1 model"""
        if self.model is None:
            return np.zeros((48, 112, 112), dtype=np.float32)
        
        try:
            # Load volume
            volume = self.processor.load_dicom_series(series_path)
            
            # Predict
            with torch.no_grad():
                volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(self.device)
                # channels_last_3d for throughput and parity
                try:
                    volume_tensor = volume_tensor.to(memory_format=torch.channels_last_3d)
                except Exception:
                    pass
                seg_logits, _ = self.model(volume_tensor)
                seg_mask = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]
            
            return seg_mask
        except Exception as e:
            print(f"Error predicting segmentation for {series_path}: {e}")
            return np.zeros((64, 128, 128), dtype=np.float32)

    def predict_segmentation_with_volume(self, series_path):
        """Return both seg mask and the preprocessed 3D volume used for Stage 1."""
        if self.model is None:
            zero = np.zeros((48, 112, 112), dtype=np.float32)
            return zero, zero
        try:
            volume = self.processor.load_dicom_series(series_path)
            with torch.no_grad():
                volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(self.device)
                try:
                    volume_tensor = volume_tensor.to(memory_format=torch.channels_last_3d)
                except Exception:
                    pass
                seg_logits, _ = self.model(volume_tensor)
                seg_mask = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]
            # One-time debug of mask stats to diagnose low-quality outputs
            try:
                if not hasattr(self, '_printed_stats'):
                    print(f"DEBUG: seg_mask stats -> min={float(seg_mask.min()):.4f}, max={float(seg_mask.max()):.4f}, mean={float(seg_mask.mean()):.4f}")
                    self._printed_stats = True
            except Exception:
                pass
            return seg_mask, volume
        except Exception as e:
            print(f"Error predicting segmentation (with volume) for {series_path}: {e}")
            zero = np.zeros((48, 112, 112), dtype=np.float32)
            return zero, zero

class ROIExtractor:
    """Research-backed ROI extraction with adaptive count and quality filtering"""
    def __init__(self, stage1_predictor, roi_size=(224, 224)):
        self.stage1_predictor = stage1_predictor
        self.roi_size = roi_size
        self.processor = SimpleDICOMProcessor()

        # Research-backed thresholds
        # Relaxed thresholds to avoid over-pruning when Stage 1 is weak
        self.min_confidence_threshold = 0.15
        self.high_confidence_threshold = 0.5
        self.max_rois_per_series = getattr(Config, 'ROIS_PER_SERIES', 3)
        # Post-process controls
        self.border_margin = 2            # suppress edge activations near skull
        self.min_region_size = 6         # minimum connected component size (pixels)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def extract_top3_rois(self, series_path):
        """Extract 0-5 ROIs based on segmentation quality (research-backed)"""
        # Cache ROI results per series to avoid recomputation
        try:
            os.makedirs(Config.STAGE2_CACHE_DIR, exist_ok=True)
            sid = os.path.basename(series_path)
            cache_path = os.path.join(Config.STAGE2_CACHE_DIR, f"{sid}_rois.npy")
            if os.path.exists(cache_path):
                arr = np.load(cache_path, allow_pickle=True)
                return list(arr)
        except Exception:
            cache_path = None
        rois = self.extract_adaptive_rois(series_path)
        try:
            if cache_path is not None:
                np.save(cache_path, np.array(rois, dtype=object), allow_pickle=True)
        except Exception:
            pass
        return rois

    def extract_adaptive_rois(self, series_path):
        """Extract 0-5 ROIs based on segmentation quality (research-backed)"""
        try:
            print(f"ğŸ”� DEBUG: Quality-based ROI extraction for {os.path.basename(series_path)}")
            
            # Get Stage 1 seg mask and the preprocessed volume (avoid reloading original DICOMs here)
            seg_mask, original_volume = self.stage1_predictor.predict_segmentation_with_volume(series_path)
            print(f"ğŸ”� DEBUG: Segmentation mask shape: {seg_mask.shape}; Volume shape: {original_volume.shape}")
            
            # STEP 1: Assess overall segmentation quality
            seg_quality = self._assess_segmentation_quality(seg_mask)
            print(f"ğŸ”� DEBUG: Segmentation quality score: {seg_quality:.3f}")
            
            # STEP 2: If segmentation is poor, still attempt candidate extraction; fallback only if none
            low_quality = seg_quality < self.min_confidence_threshold
            if low_quality:
                print(f"ğŸ”� DEBUG: Low segmentation quality ({seg_quality:.3f} < {self.min_confidence_threshold}), attempting candidate extraction anyway")
            
            # STEP 4: Extract ROIs with confidence-based filtering
            roi_candidates = self._find_quality_based_rois(seg_mask, original_volume)
            
            if low_quality and not roi_candidates:
                print("ğŸ”� DEBUG: No candidates under low-quality mask, using volume-based fallback")
                return self._get_quality_fallback_rois_from_volume(original_volume, self.max_rois_per_series)

            # STEP 5: Adaptive ROI count
            selected_rois = self._select_adaptive_rois(roi_candidates, seg_quality, original_volume)
            
            print(f"ğŸ”� DEBUG: Selected {len(selected_rois)} ROIs based on quality assessment")
            return selected_rois
            
        except Exception as e:
            print(f"â�Œ Error in quality-based ROI extraction: {e}")
            return self._get_emergency_fallback_rois()
    
    def _assess_segmentation_quality(self, seg_mask):
        """Assess segmentation quality using connected components and border penalties."""
        try:
            D, H, W = seg_mask.shape
            largest_area_frac = 0.0
            largest_mean_conf = 0.0
            total_components = 0
            border_touch_penalty = 0.0

            for z in range(D):
                sm = seg_mask[z]
                # suppress borders
                sm_proc = sm.copy()
                sm_proc[:self.border_margin, :] = 0
                sm_proc[-self.border_margin:, :] = 0
                sm_proc[:, :self.border_margin] = 0
                sm_proc[:, -self.border_margin:] = 0

               # Adaptive thresholding based on actual max values
                max_val = float(sm_proc.max())
                if max_val > 0.3:
                    thr = max(0.05, 0.3 * max_val)
                elif max_val > 0.1:
                    thr = max(0.03, 0.4 * max_val)
                else:
                    thr = max(0.02, 0.5 * max_val)
                binmask = (sm_proc > thr).astype(np.uint8)
                if binmask.max() == 0:
                    continue
                # small opening to remove speckle
                binmask = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, self.morph_kernel)

                labeled, n = label(binmask)
                if n == 0:
                    continue
                total_components += int(n)

                # evaluate components
                for comp_id in range(1, n + 1):
                    comp = (labeled == comp_id)
                    comp_size = int(comp.sum())
                    if comp_size < self.min_region_size:
                        continue
                    mean_conf = float(sm[comp].mean())
                    area_frac = comp_size / float(H * W)
                    if area_frac > largest_area_frac:
                        largest_area_frac = area_frac
                    if mean_conf > largest_mean_conf:
                        largest_mean_conf = mean_conf

                    # simple border-touch penalty if component abuts image edge
                    ys, xs = np.where(comp)
                    if ys.size > 0:
                        if (ys.min() <= self.border_margin or ys.max() >= H - self.border_margin - 1 or
                            xs.min() <= self.border_margin or xs.max() >= W - self.border_margin - 1):
                            border_touch_penalty += 0.02

            # compose quality score
            area_score = min(largest_area_frac / 0.02, 1.0)  # cap around ~2% of slice (aneurysm-sized)
            comp_penalty = min(0.1, 0.0015 * total_components) + min(0.1, border_touch_penalty)
            quality_score = max(0.0, 0.6 * largest_mean_conf + 0.4 * area_score - comp_penalty)

            # robust floor based on global mask stats to avoid spurious 0.0 quality
            max_val = float(seg_mask.max())
            mean_val = float(seg_mask.mean())
            if max_val >= 0.55:
                quality_score = max(quality_score, 0.35)
            elif max_val >= 0.45:
                quality_score = max(quality_score, 0.25)
            elif mean_val >= 0.25:
                quality_score = max(quality_score, 0.22)

            return float(quality_score)
        except Exception:
            return 0.1
    
    def _find_quality_based_rois(self, seg_mask, original_volume):
        """Find ROI candidates with confidence scores (no hardcoded count)"""
        print("ğŸ”� DEBUG: Finding quality-based ROI candidates...")
        
        # Resize segmentation mask to match original volume
        if seg_mask.shape != original_volume.shape:
            print("ğŸ”� DEBUG: Resizing segmentation mask with cv2...")
            seg_mask_resized = np.zeros(original_volume.shape, dtype=np.float32)
            for i in range(min(seg_mask.shape[0], original_volume.shape[0])):
                if i < seg_mask.shape[0]:
                    resized_slice = cv2.resize(
                        seg_mask[i],
                        (original_volume.shape[2], original_volume.shape[1])
                    )
                    seg_mask_resized[i] = resized_slice
        else:
            seg_mask_resized = seg_mask
        
        # 3D peak proposals first (relative peak logic; does not lower thresholds)
        roi_candidates = self._proposals_from_3d_peaks(seg_mask_resized)
        if len(roi_candidates) == 0:
            # Fall back to 2D slice-wise CC method
            roi_candidates = []
        
        H, W = original_volume.shape[1], original_volume.shape[2]
        for slice_idx in range(seg_mask_resized.shape[0]):
            slice_mask = seg_mask_resized[slice_idx].copy()

            # Suppress borders to avoid skull/edge activations
            slice_mask[:self.border_margin, :] = 0
            slice_mask[-self.border_margin:, :] = 0
            slice_mask[:, :self.border_margin] = 0
            slice_mask[:, -self.border_margin:] = 0

            # Adaptive dynamic threshold tied to local max (aligned with quality assessment)
            max_val = float(slice_mask.max())
            if max_val > 0.2:
                thr = max(self.min_confidence_threshold, 0.25 * max_val)
            elif max_val > 0.1:
                thr = max(0.03, 0.30 * max_val)
            else:
                thr = max(0.02, 0.25 * max_val)
            high_conf_regions = (slice_mask > thr).astype(np.uint8)
            if high_conf_regions.max() == 0:
                # Percentile-based fallback with small dilation to form blobs
                p90 = float(np.percentile(slice_mask, 90))
                if p90 > 0:
                    mask_peaks = (slice_mask >= p90).astype(np.uint8)
                    # small dilation to merge nearby high pixels
                    mask_peaks = cv2.dilate(mask_peaks, self.morph_kernel, iterations=1)
                    labeled_regions, num_regions = label(mask_peaks)
                    for region_id in range(1, num_regions + 1):
                        region_mask = (labeled_regions == region_id)
                        region_size = int(region_mask.sum())
                        if region_size < 3:
                            continue
                        ys, xs = np.where(region_mask)
                        if ys.size == 0:
                            continue
                        # Skip borders
                        if (ys.min() <= self.border_margin or ys.max() >= H - self.border_margin - 1 or
                            xs.min() <= self.border_margin or xs.max() >= W - self.border_margin - 1):
                            continue
                        com = center_of_mass(region_mask)
                        y, x = int(com[0]), int(com[1])
                        region_confidence = float(slice_mask[region_mask].mean())
                        roi_candidates.append({
                            'slice_idx': slice_idx,
                            'y': y,
                            'x': x,
                            'confidence': region_confidence,
                            'region_size': region_size
                        })
                continue
            # Apply opening only if region is sufficiently large; avoid eroding tiny blobs
            if int(high_conf_regions.sum()) > 50:
                high_conf_regions = cv2.morphologyEx(high_conf_regions, cv2.MORPH_OPEN, self.morph_kernel)

            labeled_regions, num_regions = label(high_conf_regions)
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_regions == region_id)
                region_size = int(region_mask.sum())
                if region_size < self.min_region_size:
                    continue
                ys, xs = np.where(region_mask)
                if ys.size == 0:
                    continue
                # Skip border-touching components
                if (ys.min() <= self.border_margin or ys.max() >= H - self.border_margin - 1 or
                    xs.min() <= self.border_margin or xs.max() >= W - self.border_margin - 1):
                    continue

                com = center_of_mass(region_mask)
                y, x = int(com[0]), int(com[1])
                region_confidence = float(slice_mask[region_mask].mean())

                roi_candidates.append({
                    'slice_idx': slice_idx,
                    'y': y,
                    'x': x,
                    'confidence': region_confidence,
                    'region_size': region_size
                })
        
        # Sort by confidence (descending)
        if not roi_candidates:
            # Volume-wise peak fallback: pick top maxima per slice (excluding borders)
            print("ğŸ”� DEBUG: No ROI components found; using volume-wise peak fallback")
            D = seg_mask_resized.shape[0]
            peak_candidates = []
            for z in range(D):
                m = seg_mask_resized[z].copy()
                # suppress borders
                m[:self.border_margin, :] = 0
                m[-self.border_margin:, :] = 0
                m[:, :self.border_margin] = 0
                m[:, -self.border_margin:] = 0
                yx = np.unravel_index(np.argmax(m), m.shape)
                y, x = int(yx[0]), int(yx[1])
                conf = float(m[y, x])
                if conf > 0:
                    peak_candidates.append({
                        'slice_idx': z,
                        'y': y,
                        'x': x,
                        'confidence': conf,
                        'region_size': 1
                    })
            # Keep strongest few peaks across volume
            peak_candidates.sort(key=lambda c: c['confidence'], reverse=True)
            roi_candidates.extend(peak_candidates[: max( self.max_rois_per_series * 3, 6)])

        roi_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"ğŸ”� DEBUG: Found {len(roi_candidates)} ROI candidates")
        return roi_candidates

    def _proposals_from_3d_peaks(self, seg_mask_zyx: np.ndarray):
        """3D local-max proposals with seeded relative growth (no absolute threshold lowering)."""
        try:
            D, H, W = seg_mask_zyx.shape
            # Light 3D smoothing to stabilize local maxima
            try:
                sm = ndimage.gaussian_filter(seg_mask_zyx.astype(np.float32), sigma=0.75)
            except Exception:
                sm = seg_mask_zyx.astype(np.float32)
            # 3D local maxima via maximum filter
            footprint = np.ones((3,3,3), dtype=np.uint8)
            max_f = ndimage.maximum_filter(sm, footprint=footprint, mode='nearest')
            peaks = (sm == max_f)
            # Suppress borders
            b = self.border_margin
            if b > 0:
                peaks[:, :b, :] = False; peaks[:, -b:, :] = False
                peaks[:, :, :b] = False; peaks[:, :, -b:] = False
            coords = np.argwhere(peaks)
            if coords.shape[0] == 0:
                return []
            # Rank peaks by value and keep top-K to control cost
            values = sm[peaks]
            order = np.argsort(values)[::-1]
            top_k = min(64, order.size)
            selected = coords[order[:top_k]]
            # Non-maximum suppression by 3D distance
            kept = []
            min_dist = 4.0
            for (cz, cy, cx) in selected:
                if any(((cz-kz)**2 + (cy-ky)**2 + (cx-kx)**2) ** 0.5 < min_dist for kz,ky,kx in kept):
                    continue
                kept.append((int(cz), int(cy), int(cx)))
                if len(kept) >= 64:
                    break
            # Seeded relative growth
            proposals = []
            for cz, cy, cx in kept:
                peak = float(sm[cz, cy, cx])
                if peak <= 0:
                    continue
                rel_thr = max(0.6*peak, 1e-6)  # relative to each peak
                # collect voxels that descend from the peak (thresholded region)
                region = sm >= rel_thr
                labeled, num = ndimage.label(region)
                cid = int(labeled[cz, cy, cx])
                if cid == 0:
                    continue
                comp = (labeled == cid)
                size = int(comp.sum())
                if size < self.min_region_size:
                    continue
                # score = peak * mean(comp)
                conf = peak * float(sm[comp].mean() + 1e-6)
                # project to a representative slice (peak slice)
                ys, xs = np.where(comp[cz])
                if ys.size == 0:
                    # fallback to COM over full comp
                    zc, yc, xc = ndimage.center_of_mass(comp)
                    zc = int(round(zc)); yc = int(round(yc)); xc = int(round(xc))
                    if yc <= self.border_margin or yc >= H - self.border_margin - 1 or xc <= self.border_margin or xc >= W - self.border_margin - 1:
                        continue
                    proposals.append({
                        'slice_idx': int(zc),
                        'y': int(yc),
                        'x': int(xc),
                        'confidence': float(conf),
                        'region_size': size,
                    })
                else:
                    y = int(ys.mean()); x = int(xs.mean())
                    if y <= self.border_margin or y >= H - self.border_margin - 1 or x <= self.border_margin or x >= W - self.border_margin - 1:
                        continue
                    proposals.append({
                        'slice_idx': int(cz),
                        'y': y,
                        'x': x,
                        'confidence': float(conf),
                        'region_size': size,
                    })
            proposals.sort(key=lambda c: c['confidence'], reverse=True)
            return proposals
        except Exception:
            return []
    
    def _select_adaptive_rois(self, roi_candidates, seg_quality, original_volume):
        """Adaptively select ROIs based on segmentation quality (research-backed)"""
        if not roi_candidates:
            print("ğŸ”� DEBUG: No candidates found, using fallback")
            return self._get_quality_fallback_rois_from_volume(original_volume)
        
        # Adaptive selection based on segmentation quality
        if seg_quality >= self.high_confidence_threshold:
            max_rois = self.max_rois_per_series
            min_confidence = 0.3
        elif seg_quality >= self.min_confidence_threshold + 0.2:
            max_rois = self.max_rois_per_series
            min_confidence = 0.2
        else:
            max_rois = self.max_rois_per_series
            min_confidence = 0.05
        
        # Filter and select ROIs
        filtered = [c for c in roi_candidates if c['confidence'] >= min_confidence]
        selected_candidates = filtered[:max_rois]
        # If not enough, top-off with next best candidates
        if len(selected_candidates) < max_rois:
            for c in roi_candidates:
                if c in selected_candidates:
                    continue
                selected_candidates.append(c)
                if len(selected_candidates) >= max_rois:
                    break
        
        # Convert to ROI format
        rois = []
        for i, candidate in enumerate(selected_candidates):
            roi_patch = self._extract_roi_patch(
                original_volume,
                candidate['slice_idx'], 
                candidate['y'], 
                candidate['x']
            )
            
            rois.append({
                'roi_image': roi_patch,
                'slice_idx': candidate['slice_idx'],
                'coordinates': (candidate['y'], candidate['x']),
                'confidence': candidate['confidence'],
                'roi_id': i
            })
        # Ensure at least max_rois via center-based fallback if still short
        if len(rois) < self.max_rois_per_series:
            needed = self.max_rois_per_series - len(rois)
            center_fallbacks = self._get_quality_fallback_rois_from_volume(original_volume, needed)
            rois.extend(center_fallbacks)
        print(f"ğŸ”� DEBUG: Adaptively selected {len(rois)} ROIs (quality: {seg_quality:.3f})")
        return rois[: self.max_rois_per_series]
    
    def _get_quality_fallback_rois(self, series_path, seg_mask):
        """Fallback for poor segmentation quality: generate multiple center-based ROIs"""
        print("ğŸ”� DEBUG: Using quality-aware fallback (multi-center ROIs)")
        original_volume = self._load_efficient_volume(series_path)
        return self._get_quality_fallback_rois_from_volume(original_volume, self.max_rois_per_series)

    def _get_quality_fallback_rois_from_volume(self, original_volume, count: int = 3):
        D, H, W = original_volume.shape
        # Choose slice indices: center and quartiles
        slices = sorted(set([D // 2, max(0, D // 4), min(D - 1, 3 * D // 4)]))
        # Ensure desired count
        while len(slices) < count:
            # Add random slices if needed
            slices.append(np.random.randint(0, D))
            slices = list(dict.fromkeys(slices))
        rois = []
        cy, cx = H // 2, W // 2
        for i, s in enumerate(slices[:count]):
            roi_patch = self._extract_roi_patch(original_volume, s, cy, cx)
            rois.append({
                'roi_image': roi_patch,
                'slice_idx': s,
                'coordinates': (cy, cx),
                'confidence': 0.2,
                'roi_id': i
            })
        return rois
    
    def _get_simple_fallback_rois(self):
        """Simple fallback when no quality ROIs found"""
        print("ğŸ”� DEBUG: Using simple fallback (single center ROI)")
        dummy_roi = np.random.random((*Config.ROI_SIZE, 3)).astype(np.float32)
        return [{
            'roi_image': dummy_roi,
            'slice_idx': 25,
            'coordinates': (128, 128),
            'confidence': 0.1,
            'roi_id': 0
        }]
    
    def _get_emergency_fallback_rois(self):
        """Emergency fallback when everything fails"""
        print("ğŸ”� DEBUG: Using emergency fallback ROI")
        dummy_roi = np.random.random((*Config.ROI_SIZE, 3)).astype(np.float32)
        return [{
            'roi_image': dummy_roi,
            'slice_idx': 0,
            'coordinates': (128, 128),
            'confidence': 0.1,
            'roi_id': 0
        }]

    
    def _load_efficient_volume(self, series_path):
        """Load volume with smart distributed sampling to cover entire brain"""
        try:
            # Cache original volume slices to reduce repeated I/O
            os.makedirs(Config.STAGE2_CACHE_DIR, exist_ok=True)
            sid = os.path.basename(series_path)
            vcache = os.path.join(Config.STAGE2_CACHE_DIR, f"{sid}_vol.npy")
            if os.path.exists(vcache):
                return np.load(vcache, allow_pickle=False)
            dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            pixel_arrays = []
            
            # SMART SAMPLING: Distribute 50 slices across entire volume
            total_files = len(dicom_files)
            if total_files > 50:
                # Calculate step size to distribute slices evenly
                step = total_files / 50
                selected_indices = [int(i * step) for i in range(50)]
                selected_files = [dicom_files[i] for i in selected_indices]
                print(f"ğŸ”� DEBUG: Smart sampling - selected {len(selected_files)} files from {total_files} total (every {step:.1f})")
            else:
                selected_files = dicom_files
                print(f"ğŸ”� DEBUG: Using all {len(selected_files)} files (less than 50)")
            
            for f in selected_files:
                try:
                    ds = pydicom.dcmread(os.path.join(series_path, f), force=True)
                    if hasattr(ds, 'pixel_array'):
                        arr = ds.pixel_array
                        if arr.ndim == 2:
                            pixel_arrays.append(arr)
                except:
                    continue
            
            if pixel_arrays:
                # SMALLER target shape to reduce memory usage
                target_shape = (256, 256)  # Reduced from (512, 512)
                
                resized_arrays = []
                for arr in pixel_arrays:
                    # Use cv2.resize instead of ndimage.zoom (more reliable)
                    if arr.shape != target_shape:
                        resized_arr = cv2.resize(arr.astype(np.float32), target_shape)
                        resized_arrays.append(resized_arr)
                    else:
                        resized_arrays.append(arr.astype(np.float32))
                
                volume = np.stack(resized_arrays, axis=0)
                
                # Simple normalization
                p1, p99 = np.percentile(volume, [1, 99])
                volume = np.clip(volume, p1, p99)
                volume = (volume - p1) / (p99 - p1 + 1e-8)
                
                try:
                    np.save(vcache, volume.astype(np.float32), allow_pickle=False)
                except Exception:
                    pass
                return volume
            
        except Exception as e:
            print(f"Error loading efficient volume: {e}")
        
        # Fallback volume (matches our smart sampling approach)
        return np.random.random((50, 256, 256)).astype(np.float32)

    
    def _extract_roi_patch(self, volume, slice_idx, center_y, center_x):
        """Extract ROI with adjacent-slice context as RGB channels (s-1, s, s+1)."""
        D, H, W = volume.shape
        s_indices = [max(0, slice_idx - 1), slice_idx, min(D - 1, slice_idx + 1)]
        channels = []
        half_size = Config.ROI_SIZE[0] // 2
        for s in s_indices:
            slice_data = volume[s]
            h, w = slice_data.shape
            y1 = max(0, center_y - half_size)
            y2 = min(h, center_y + half_size)
            x1 = max(0, center_x - half_size)
            x2 = min(w, center_x + half_size)
            patch = slice_data[y1:y2, x1:x2]
            patch_resized = cv2.resize(patch, Config.ROI_SIZE)
            channels.append(patch_resized)
        patch_3ch = np.stack(channels, axis=2)
        return patch_3ch
    

def create_training_data(df, stage1_predictor):
    """Create training data with 3 ROIs per series"""
    print("ğŸ”„ Extracting ROIs for training data...")
    
    # Reuse cached ROIs/training dataframe if available
    cache_dir = 'rois'
    os.makedirs(cache_dir, exist_ok=True)
    cached_df_path_parquet = os.path.join(cache_dir, 'training_df.parquet')
    external_cached_df_path = os.path.join(getattr(Config, 'ROIS_EXTERNAL_DIR', ''), 'training_df.parquet')
    if Config.REUSE_EXISTING_ROIS:
        # Prefer working cache
        if os.path.exists(cached_df_path_parquet):
            try:
                cached = pl.read_parquet(cached_df_path_parquet).to_pandas()
                if len(cached) > 0 and all(c in cached.columns for c in ['roi_path', 'roi_id', 'series_id'] + Config.LABEL_COLS):
                    print(f"âœ… Reusing cached training ROIs (working): {len(cached)} samples from {cached['series_id'].nunique()} series")
                    return cached
            except Exception:
                pass
        # Fallback to external cache
        if isinstance(external_cached_df_path, str) and len(external_cached_df_path) and os.path.exists(external_cached_df_path):
            try:
                cached = pl.read_parquet(external_cached_df_path).to_pandas()
                if len(cached) > 0 and all(c in cached.columns for c in ['roi_path', 'roi_id', 'series_id'] + Config.LABEL_COLS):
                    print(f"âœ… Reusing cached training ROIs (external): {len(cached)} samples from {cached['series_id'].nunique()} series")
                    # Optionally copy into working for faster subsequent access
                    try:
                        pl.from_pandas(cached).write_parquet(cached_df_path_parquet)
                    except Exception:
                        pass
                    return cached
            except Exception:
                pass
    roi_extractor = ROIExtractor(stage1_predictor)
    training_data = []
    
    os.makedirs('rois', exist_ok=True)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ROIs"):
        series_id = row[Config.ID_COL]
        series_path = os.path.join(Config.SERIES_DIR, series_id)
        
        if not os.path.exists(series_path):
            continue
        
        # Extract ROIs
        rois = roi_extractor.extract_top3_rois(series_path)
        
        # Create training samples
        for roi_data in rois:
            roi_filename = f"rois/{series_id}_roi_{roi_data['roi_id']}.png"
            
            # Save ROI image
            roi_image = (roi_data['roi_image'] * 255).astype(np.uint8)
            Image.fromarray(roi_image).save(roi_filename)
            
            # Create training record
            sample = {
                'roi_id': f"{series_id}_roi_{roi_data['roi_id']}",
                'roi_path': roi_filename,
                'series_id': series_id,
                'roi_confidence': roi_data['confidence'],
                'slice_idx': roi_data['slice_idx']
            }
            
            # Add all label columns
            for col in Config.LABEL_COLS:
                sample[col] = row[col]
            
            training_data.append(sample)
    
    training_df = pd.DataFrame(training_data)
    print(f"âœ… Created {len(training_df)} training samples from {len(df)} series")
    # Save for reuse next runs
    try:
        pl.from_pandas(training_df).write_parquet(cached_df_path_parquet)
        print(f"ğŸ’¾ Saved training ROI dataframe â†’ {cached_df_path_parquet}")
    except Exception:
        pass
    
    return training_df

print("âœ… Data loading and ROI extraction functions loaded")

# ====================================================
# CELL 3: MODEL DEFINITION
# ====================================================

class AneurysmClassificationDataset(Dataset):
    """Dataset for ROI-based classification"""
    def __init__(self, df, mode='train'):
        self.df = df
        self.mode = mode
        
        # Data augmentation for training
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load ROI image
        roi_path = row['roi_path']
        try:
            image = Image.open(roi_path).convert('RGB')
        except:
            # Fallback to dummy image
            image = Image.fromarray(np.random.randint(0, 255, (*Config.ROI_SIZE, 3), dtype=np.uint8))
        
        # Apply transforms
        image = self.transform(image)
        
        # Get labels
        labels = torch.tensor([row[col] for col in Config.LABEL_COLS], dtype=torch.float32)
        
        return {
            'image': image,
            'labels': labels,
            'roi_id': row['roi_id'],
            'confidence': torch.tensor(row['roi_confidence'], dtype=torch.float32)
        }

class AneurysmEfficientNet(nn.Module):
    """EfficientNet-B3 for aneurysm classification with offline weights"""
    def __init__(self, num_classes=len(Config.LABEL_COLS)):
        super().__init__()
        
        # Load EfficientNet-B3 with offline pre-trained weights
        import timm
        
        # Path to the pre-trained weights you added
        weights_path = '/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b3/1/tf_efficientnet_b3_aa-84b4657e.pth'
        
        try:
            # Create model without pre-trained weights first
            self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
            
            # Load the offline weights
            if os.path.exists(weights_path):
                print(f"ğŸ”„ Loading offline EfficientNet-B3 weights from: {weights_path}")
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
                
                # Load weights into the model (ignore classifier since we're using num_classes=0)
                self.backbone.load_state_dict(state_dict, strict=False)
                print("âœ… Successfully loaded offline EfficientNet-B3 weights!")
            else:
                print(f"âš ï¸� Weights file not found at {weights_path}, using random initialization")
                
        except Exception as e:
            print(f"â�Œ Error loading offline weights: {e}")
            print("ğŸ”„ Falling back to timm without pre-training...")
            self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits # Initialize model

# Using original EfficientNet approach

def calculate_class_weights(df):
    """Calculate class weights with 13x multiplier for Aneurysm Present"""
    pos_counts = df[Config.LABEL_COLS].sum()
    neg_counts = len(df) - pos_counts
    
    # Standard frequency-based weights
    class_weights = neg_counts / (pos_counts + 1e-8)
    class_weights = np.minimum(class_weights, 100.0)  # Cap at 100
    
    # Apply 13x multiplier to "Aneurysm Present" (matches competition metric)
    class_weights.iloc[-1] = class_weights.iloc[-1] * 13.0
    
    return torch.tensor(class_weights.values, dtype=torch.float32)

print("âœ… Model definition loaded")

# ====================================================
# CELL 4: TRAINING PIPELINE
# ====================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=Config.MIXED_PRECISION):
            logits = model(images)
            loss = criterion(logits, labels)
        
        # Backward pass
        if Config.MIXED_PRECISION:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=Config.MIXED_PRECISION):
                logits = model(images)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions for AUC
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
    
    # Calculate AUC
    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        try:
            auc_scores = []
            for i in range(len(Config.LABEL_COLS)):
                if len(np.unique(all_labels[:, i])) > 1:
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                    auc_scores.append(auc)
                else:
                    auc_scores.append(0.5)
            
            # Weighted AUC (13x weight for Aneurysm Present)
            weights = [1.0] * (len(Config.LABEL_COLS) - 1) + [13.0]
            weighted_auc = np.average(auc_scores, weights=weights)
        except:
            weighted_auc = 0.5
    else:
        weighted_auc = 0.5
    
    return total_loss / num_batches, weighted_auc

def main_training():
    print("ğŸš€ STAGE 2: ANEURYSM CLASSIFICATION WITH EFFICIENTNET-B3")
    
    # Load data
    train_df = pd.read_csv(Config.TRAIN_CSV_PATH)
    
    if Config.DEBUG_MODE:
        train_df = train_df.head(Config.DEBUG_SAMPLES)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Aneurysm cases: {train_df['Aneurysm Present'].sum()}")
    
    # Initialize Stage 1 predictor
    stage1_predictor = Stage1Predictor(Config.STAGE1_MODEL_PATH)
    
    # Create training data with ROIs
    training_df = create_training_data(train_df, stage1_predictor)
    
    # Calculate class weights
    class_weights = calculate_class_weights(training_df)
    print(f"Class weights: {class_weights}")
    
    # Create criterion with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(Config.DEVICE)
    
    # Mixed precision scaler
    global scaler
    scaler = torch.cuda.amp.GradScaler(enabled=Config.MIXED_PRECISION)
    
    # Cross-validation or single split
    # Use Aneurysm Present for stratification
    fold_scores = []
    if Config.N_FOLDS <= 1:
        idx_all = np.arange(len(training_df))
        train_idx, val_idx = train_test_split(
            idx_all,
            test_size=0.2,
            stratify=training_df['Aneurysm Present'],
            random_state=42,
        )
        fold_splits = [(train_idx, val_idx)]
    else:
        skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
        fold_splits = list(skf.split(training_df, training_df['Aneurysm Present']))
    
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{Config.N_FOLDS}")
        print(f"{'='*50}")
        
        # Split data
        train_fold_df = training_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = training_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train ROIs: {len(train_fold_df)}, Val ROIs: {len(val_fold_df)}")
        
        # Create datasets
        train_dataset = AneurysmClassificationDataset(train_fold_df, mode='train')
        val_dataset = AneurysmClassificationDataset(val_fold_df, mode='val')
        
        # Create loaders (tuned for throughput)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
        )
        
        # Initialize model
        model = AneurysmEfficientNet().to(Config.DEVICE)
        
        # Optimizer with different learning rates
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': Config.LEARNING_RATE * 0.1},  # Lower LR for backbone
            {'params': model.classifier.parameters(), 'lr': Config.LEARNING_RATE}
        ], weight_decay=1e-4)

        # Multi-GPU if available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
        
        # Training loop
        best_auc = 0
        
        for epoch in range(Config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
            
            # Validate
            val_loss, val_auc = validate_epoch(model, val_loader, criterion, Config.DEVICE)
            
            # Step scheduler
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'epoch': epoch,
                    'fold': fold
                }, f'stage2_fold_{fold}_best.pth')
                print(f"ğŸ’¾ Saved best model (AUC: {val_auc:.4f})")
        
        fold_scores.append(best_auc)
        print(f"Fold {fold + 1} best AUC: {best_auc:.4f}")
    
    # Final results
    mean_cv_score = np.mean(fold_scores)
    print(f"\nâœ… Cross-validation complete!")
    print(f"Mean CV AUC: {mean_cv_score:.4f} Â± {np.std(fold_scores):.4f}")
    print(f"Individual fold scores: {fold_scores}")

print("âœ… Training pipeline loaded")

# ====================================================
# CELL 5: INFERENCE & SUBMISSION
# ====================================================

class InferenceConfig:
    """Configuration for inference server"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ID_COL = 'SeriesInstanceUID'
    LABEL_COLS = [
        'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery', 'Anterior Communicating Artery',
        'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery',
        'Left Posterior Communicating Artery', 'Right Posterior Communicating Artery',
        'Basilar Tip', 'Other Posterior Circulation', 'Aneurysm Present',
    ]

class ModelEnsemble:
    """Ensemble of Stage 2 models for inference"""
    def __init__(self, model_paths, device):
        self.device = device
        self.models = []
        
        for path in model_paths:
            try:
                model = AneurysmEfficientNet().to(device)
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Handle DataParallel wrapper
                if any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                
                model.load_state_dict(state_dict)
                model.eval()
                self.models.append(model)
                print(f"Loaded model: {path}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
    def predict_single(self, series_path):
        """Predict for a single series"""
        # Initialize predictors once and reuse
        global _shared_stage1_predictor
        if '_shared_stage1_predictor' not in globals() or _shared_stage1_predictor is None:
            _shared_stage1_predictor = Stage1Predictor(Config.STAGE1_MODEL_PATH)
        roi_extractor = ROIExtractor(_shared_stage1_predictor)
        
        # Extract ROIs
        rois = roi_extractor.extract_top3_rois(series_path)
        
        # Prepare images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Disable flips to preserve artery laterality (left/right)
        tta_transforms = [
            lambda img: img,
        ]

        # Keep PIL images for TTA
        roi_pils = []
        roi_confs = []
        for roi_data in rois:
            roi_image = roi_data['roi_image']
            roi_pil = Image.fromarray((roi_image * 255).astype(np.uint8))
            roi_pils.append(roi_pil)
            roi_confs.append(float(roi_data.get('confidence', 0.0)))
        roi_confs = np.array(roi_confs, dtype=np.float32) if len(roi_confs) else np.array([1.0], dtype=np.float32)
        # normalize confidences to sum=1 (avoid division by zero)
        conf_sum = float(roi_confs.sum())
        if conf_sum <= 1e-6:
            roi_weights = np.ones_like(roi_confs) / max(1, roi_confs.size)
        else:
            roi_weights = roi_confs / conf_sum
        
        # Ensemble predictions
        all_predictions = []
        
        with torch.no_grad():
            for model in self.models:
                per_roi_preds = []
                for roi_pil in roi_pils:
                    tta_probs = []
                    for t in tta_transforms:
                        aug_pil = t(roi_pil)
                        roi_tensor = transform(aug_pil).unsqueeze(0).to(self.device)
                        logits = model(roi_tensor)
                        probs = torch.sigmoid(logits).cpu().numpy()[0]
                        tta_probs.append(probs)
                    roi_avg = np.mean(tta_probs, axis=0)
                    per_roi_preds.append(roi_avg)
                per_roi_preds = np.stack(per_roi_preds, axis=0)  # [R, C]
                # confidence-weighted mean and max-pool fusion
                weighted_mean = (per_roi_preds * roi_weights[:, None]).sum(axis=0)
                roi_max = per_roi_preds.max(axis=0)
                fused = np.maximum(weighted_mean, roi_max)
                all_predictions.append(fused)
        
        # Average ensemble predictions
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        return ensemble_pred

class InferenceDICOMProcessor:
    """DICOM processor for inference"""
    def __init__(self):
        pass

# Global variables for model ensemble
model_ensemble = None
processor = None

def initialize_models():
    """Initialize models - called once at startup"""
    global model_ensemble, processor
    
    print("Initializing models...")
    
    # Model paths - adjust these to match your uploaded dataset structure
    model_paths = [
        'stage2_fold_0_best.pth',
        'stage2_fold_1_best.pth',
        'stage2_fold_2_best.pth',
        'stage2_fold_3_best.pth',
        'stage2_fold_4_best.pth',
    ]
    
    # Check if models exist, use available ones
    available_models = [path for path in model_paths if os.path.exists(path)]
    
    if not available_models:
        print("Warning: No trained models found! Using dummy predictions.")
        model_ensemble = None
    else:
        try:
            model_ensemble = ModelEnsemble(available_models, InferenceConfig.DEVICE)
            print("Models initialized successfully!")
        except Exception as e:
            print(f"Error initializing models: {e}")
            model_ensemble = None
    
    processor = InferenceDICOMProcessor()

def predict(series_path: str) -> pl.DataFrame:
    """Make predictions for the competition API"""
    global model_ensemble, processor
    
    # Initialize models on first call (lazy loading)
    if model_ensemble is None and processor is None:
        initialize_models()
    
    series_id = os.path.basename(series_path)
    
    try:
        if model_ensemble is not None:
            # Use trained ensemble
            predictions = model_ensemble.predict_single(series_path)
        else:
            # Fallback: extract metadata and make informed dummy predictions
            print(f"Using fallback prediction for {series_id}")
            
            # Load DICOM metadata
            all_filepaths = []
            for root, _, files in os.walk(series_path):
                for file in files:
                    if file.endswith('.dcm'):
                        all_filepaths.append(os.path.join(root, file))
            
            if all_filepaths:
                ds = pydicom.dcmread(all_filepaths[0], force=True)
                modality = getattr(ds, 'Modality', 'UNKNOWN')
                
                # Slightly better informed predictions based on modality
                if modality in ['CTA', 'MRA']:
                    # Vascular imaging - slightly higher probability
                    base_prob = 0.1
                else:
                    # Other modalities - lower baseline
                    base_prob = 0.05
                
                # Add some noise to make predictions more realistic
                predictions = np.random.normal(base_prob, 0.02, len(InferenceConfig.LABEL_COLS))
                predictions = np.clip(predictions, 0.001, 0.999)
            else:
                # No DICOM files found
                predictions = np.full(len(InferenceConfig.LABEL_COLS), 0.5)

        # Ensure predictions is numpy array and convert to list safely
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        # Create prediction DataFrame
        prediction_df = pl.DataFrame(
            data=[[series_id] + predictions.tolist()],
            schema=[InferenceConfig.ID_COL, *InferenceConfig.LABEL_COLS],
            orient='row',
        )
        
    except Exception as e:
        print(f"Error processing {series_id}: {e}")
        # Return safe default predictions
        prediction_df = pl.DataFrame(
            data=[[series_id] + [0.5] * len(InferenceConfig.LABEL_COLS)],
            schema=[InferenceConfig.ID_COL, *InferenceConfig.LABEL_COLS],
            orient='row',
        )
    
    # IMPORTANT: Remove SeriesInstanceUID before returning (API requirement)
    prediction_df = prediction_df.drop(InferenceConfig.ID_COL)
    
    # IMPORTANT: Disk cleanup to prevent "out of disk space" errors
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return prediction_df



# ====================================================
# SERVER EXECUTION
# ====================================================

# Initialize the inference server
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

print("âœ… Inference and submission pipeline loaded")

# ====================================================
# CELL 6: MAIN EXECUTION
# ====================================================

if __name__ == "__main__":
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Production mode - serve the API
        print("Starting inference server...")
        inference_server.serve()
    else:
        # Training mode
        print("Ready for Stage 2 training!")
        print("Uncomment the line below to start training:")
        print("# main_training()")
        
        # Uncomment to start training
        main_training()
        
        # Or run local testing
        print("Running local gateway for testing...")
        inference_server.run_local_gateway()
        
        # Display results if available
        results_path = '/kaggle/working/submission.parquet'
        if os.path.exists(results_path):
            results_df = pl.read_parquet(results_path)
            print("Submission preview:")
            print(results_df.head())




