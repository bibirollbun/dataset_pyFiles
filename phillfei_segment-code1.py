# Environment setup and library imports
import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import functools
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import pydicom
import nibabel as nib  # For NII file handling
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    torch.cuda.empty_cache()
else:
    raise RuntimeError("CUDA is not available! This code requires GPU.")


class Config:

    TRAIN_CSV_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
    

    # Alternative paths to try if the main path doesn't exist
    SEGMENTATION_DIR_ALTERNATIVES = [
        "/kaggle/input/rsna-segmentation-masks",
        "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations",
        "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentation",
        "./segmentation"  # Local path
    ]
    DICOM_SERIES_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
    
    # Model parameters for 2D U-Net segmentation
    IMAGE_SIZE = 224
    NUM_CLASSES = 14  # Binary segmentation: 0=background, 1=foreground
    BATCH_SIZE = 16  # Adjusted for 2D processing
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    
    # Model configuration
    MODEL_TYPE = "unet"  # Changed from EfficientNet to U-Net
    USE_METADATA = True
    USE_WINDOWING = True
    USE_CLAHE = True
    USE_STRONG_AUGMENTATION = True
    
    # Segmentation specific settings
    USE_DICE_LOSS = True
    DICE_WEIGHT = 0.5
    BCE_WEIGHT = 0.5
    USE_FOCAL_LOSS = False
    
    # GPU optimization settings
    NUM_WORKERS = 2
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    
    # Training parameters with robust cross-validation
    NUM_FOLDS = 5
    FOLD = 0
    ACCUMULATION_STEPS = 4
    EARLY_STOPPING_PATIENCE = 5
    USE_GROUP_CV = True
    
    # Data loading optimization
    CACHE_SIZE = 100
    
    # Output
    OUTPUT_DIR = "/kaggle/working"
    MODEL_NAME = "2d_unet_segmentation"

config = Config()

print("=== Configuration Summary - 2D U-Net Segmentation ===")
print(f"Model Type: {config.MODEL_TYPE}")
print(f"Image Size: {config.IMAGE_SIZE}")
print(f"Number of Classes: {config.NUM_CLASSES}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Accumulation Steps: {config.ACCUMULATION_STEPS}")
print(f"Effective Batch Size: {config.BATCH_SIZE * config.ACCUMULATION_STEPS}")
print(f"CLAHE Enabled: {config.USE_CLAHE}")
print(f"Strong Augmentation: {config.USE_STRONG_AUGMENTATION}")
print(f"Group Cross-Validation: {config.USE_GROUP_CV}")
print(f"Dice Loss Weight: {config.DICE_WEIGHT}")
print(f"BCE Loss Weight: {config.BCE_WEIGHT}")


# Load data
print("Loading data...")
train_df = pd.read_csv(config.TRAIN_CSV_PATH)

print(f"Train data shape: {train_df.shape}")

# Define target columns for segmentation (13 anatomical locations + background)
TARGET_COLS = [ 
    'Other Posterior Circulation',
    'Basilar Tip',
    'Right Posterior Communicating Artery',
    'Left Posterior Communicating Artery',
    'Right Anterior Cerebral Artery', 
    'Left Anterior Cerebral Artery',
    'Anterior Communicating Artery', 
    'Right Middle Cerebral Artery', 
    'Left Middle Cerebral Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Infraclinoid Internal Carotid Artery',
]

# Class mapping for segmentation (0 = background, 1-13 = anatomical locations)
CLASS_MAPPING = {col: idx + 1 for idx, col in enumerate(TARGET_COLS)}
CLASS_MAPPING['Background'] = 0

print(f"Target columns: {len(TARGET_COLS)}")
print(f"Class mapping: {CLASS_MAPPING}")


def get_windowing_params(modality: str) -> Tuple[float, float]:
    """Get optimal windowing parameters for different modalities"""
    windows = {
        'CT': (40, 80),
        'CTA': (50, 350), 
        'MRA': (600, 1200),
        'MRI': (40, 80),
        'MR': (40, 80)
    }
    return windows.get(modality, (40, 80))

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Apply DICOM windowing to normalize image intensities"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def apply_clahe_normalization(img: np.ndarray, modality: str) -> np.ndarray:
    """Apply CLAHE with modality-specific optimization"""
    if not config.USE_CLAHE:
        return img
        
    if modality in ['CTA', 'MRA']:
        # Vascular imaging: stronger contrast improvement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = cv2.convertScaleAbs(img_clahe, alpha=1.1, beta=5)
    elif modality in ['MRI', 'MR']:
        # MRI: gentler improvement with gamma correction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = np.power(img_clahe / 255.0, 0.9) * 255
        img_clahe = img_clahe.astype(np.uint8)
    else:
        # CT: standard CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
    
    return img_clahe

def robust_normalization(volume: np.ndarray) -> np.ndarray:
    """Apply robust normalization using percentiles"""
    p1, p99 = np.percentile(volume.flatten(), [1, 99])
    volume_norm = np.clip(volume, p1, p99)
    
    if p99 > p1:
        volume_norm = (volume_norm - p1) / (p99 - p1 + 1e-7)
    else:
        volume_norm = np.zeros_like(volume_norm)
        
    return (volume_norm * 255).astype(np.uint8)


# 修改后的NII文件匹配函数，支持_cowseg后缀和文件夹验证
def find_matching_segmentation_file_enhanced(series_uid: str, segmentation_dir: str, dicom_series_dir: str) -> Optional[str]:
    """Find matching segmentation NII file for a given series UID by matching with DICOM folders
    
    Enhanced version that handles _cowseg suffix and verifies DICOM folder existence
    """
    
    # First, check if the corresponding DICOM series folder exists
    dicom_series_path = os.path.join(dicom_series_dir, series_uid)
    if not os.path.exists(dicom_series_path):
        return None
    
    # Get list of DICOM files in the series folder
    try:
        dicom_files = [f for f in os.listdir(dicom_series_path) if f.endswith('.dcm')]
        if len(dicom_files) == 0:
            return None
    except Exception as e:
        return None
    
    # Enhanced matching: Handle _cowseg suffix and verify DICOM folder exists
    try:
        for root, dirs, files in os.walk(segmentation_dir):
            for file in files:
                if file.endswith(('.nii', '.nii.gz')):
                    # Remove extensions to get the base name
                    file_base = file.replace('.nii.gz', '').replace('.nii', '')
                    
                    # Handle _cowseg suffix: remove it if present
                    if file_base.endswith('_cowseg'):
                        file_base = file_base[:-7]  # Remove '_cowseg' (7 characters)
                    else:
                        continue
                    # Check if the base name matches series_uid
                    if file_base == series_uid:
                        # Double-check that the corresponding DICOM folder exists
                        dicom_folder_path = os.path.join(dicom_series_dir, file_base)
                        if os.path.exists(dicom_folder_path):
                            # Verify the folder contains DICOM files
                            try:
                                dcm_files = [f for f in os.listdir(dicom_folder_path) if f.endswith('.dcm')]
                                if len(dcm_files) > 0:
                                    return os.path.join(root, file)
                            except Exception:
                                continue
                            
    except Exception as e:
        pass
    
    return None

# 更新validate_dicom_nii_matching函数以使用新的匹配逻辑
def validate_dicom_nii_matching_enhanced(segmentation_dir: str, dicom_series_dir: str) -> dict:
    """Validate the matching between DICOM series and NII files with enhanced matching logic"""
    print("Validating DICOM-NII matching with enhanced logic...")
    
    # Get all DICOM series folders
    dicom_series_folders = []
    try:
        for item in os.listdir(dicom_series_dir):
            item_path = os.path.join(dicom_series_dir, item)
            if os.path.isdir(item_path):
                # Check if it contains DICOM files
                dcm_files = [f for f in os.listdir(item_path) if f.endswith('.dcm')]
                if len(dcm_files) > 0:
                    dicom_series_folders.append(item)
    except Exception as e:
        print(f"Error reading DICOM series directory: {e}")
        return {}
    
    # Get all NII files
    nii_files = list_available_nii_files(segmentation_dir)
    
    print(f"Found {len(dicom_series_folders)} DICOM series folders")
    print(f"Found {len(nii_files)} NII files")
    
    # Enhanced matching: Handle _cowseg suffix and verify DICOM folder exists
    matches = []
    unmatched_dicom = []
    unmatched_nii = []
    
    for series_uid in dicom_series_folders:
        matched_nii = None
        for nii_file in nii_files:
            nii_basename = os.path.basename(nii_file)
            # Remove extensions to get the base name
            nii_base = nii_basename.replace('.nii.gz', '').replace('.nii', '')
            
            # Handle _cowseg suffix: remove it if present
            if nii_base.endswith('_cowseg'):
                nii_base = nii_base[:-7]  # Remove '_cowseg' (7 characters)
            
            # Check if the base name matches series_uid
            if series_uid == nii_base:
                # Double-check that the corresponding DICOM folder exists
                dicom_folder_path = os.path.join(dicom_series_dir, nii_base)
                if os.path.exists(dicom_folder_path):
                    # Verify the folder contains DICOM files
                    try:
                        dcm_files = [f for f in os.listdir(dicom_folder_path) if f.endswith('.dcm')]
                        if len(dcm_files) > 0:
                            matched_nii = nii_file
                            break
                    except Exception:
                        continue
        
        if matched_nii:
            matches.append((series_uid, matched_nii))
        else:
            unmatched_dicom.append(series_uid)
    
    # Find unmatched NII files
    matched_nii_files = [match[1] for match in matches]
    for nii_file in nii_files:
        if nii_file not in matched_nii_files:
            unmatched_nii.append(nii_file)
    
    print(f"\\nEnhanced Matching Results:")
    print(f"- Successful matches: {len(matches)}")
    print(f"- Unmatched DICOM series: {len(unmatched_dicom)}")
    print(f"- Unmatched NII files: {len(unmatched_nii)}")
    
    if len(matches) > 0:
        print(f"\\nSample matches:")
        for i, (series_uid, nii_file) in enumerate(matches[:5]):
            print(f"  {i+1}. {series_uid} <-> {os.path.basename(nii_file)}")
    
    if len(unmatched_dicom) > 0:
        print(f"\\nSample unmatched DICOM series:")
        for i, series_uid in enumerate(unmatched_dicom[:5]):
            print(f"  {i+1}. {series_uid}")
    
    if len(unmatched_nii) > 0:
        print(f"\\nSample unmatched NII files:")
        for i, nii_file in enumerate(unmatched_nii[:5]):
            print(f"  {i+1}. {os.path.basename(nii_file)}")
    
    return {
        'matches': matches,
        'unmatched_dicom': unmatched_dicom,
        'unmatched_nii': unmatched_nii,
        'total_dicom': len(dicom_series_folders),
        'total_nii': len(nii_files)
    }

print("✅ 已创建增强版的NII文件匹配函数")
print("   - find_matching_segmentation_file_enhanced: 支持_cowseg后缀和文件夹验证")
print("   - validate_dicom_nii_matching_enhanced: 增强版验证函数")



# 添加二值化处理函数
def apply_binary_thresholding(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    对分割掩码进行二值化处理
    大于threshold的像素值设为1，小于等于threshold的像素值设为0
    
    Args:
        mask: 输入的分割掩码
        threshold: 二值化阈值，默认为127
    
    Returns:
        二值化后的掩码 (0或1)
    """
    if mask.max() <= 1:
        # 如果已经是二值化的，直接返回
        return mask.astype(np.uint8)
    
    # 应用二值化阈值
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask

def load_nii_segmentation(nii_path: str) -> np.ndarray:
    """Load NII segmentation file and return 3D array"""
    try:
        nii_img = nib.load(nii_path)
        segmentation = nii_img.get_fdata()
        return segmentation.astype(np.uint8)
    except Exception as e:
        print(f"Error loading NII file {nii_path}: {e}")
        return None
import os
import time
import glob
import numpy as np
import cv2
from typing import List

def convert_3d_to_2d_slices_intelligent(volume_3d: np.ndarray, target_size: int = 224, 
                                       series_uid: str = None, dicom_series_dir: str = None) -> Tuple[List[np.ndarray], List[str]]:
    """使用OpenCV进行空间对齐的智能3D到2D转换（向量化批处理版本）
    
    Args:
        volume_3d: 3D分割体积
        target_size: 2D切片的目标尺寸
        series_uid: 序列UID，用于检查对应的DICOM文件
        dicom_series_dir: 包含DICOM序列的目录
    
    Returns:
        (slices, dicom_filenames) 二元组：
        - slices: 2D切片列表（顺序与排序后的DICOM文件名一致）
        - dicom_filenames: 对应的DICOM基文件名（已排序）
    """
    if volume_3d is None:
        return [], []
    
    # 如果没有提供series_uid或dicom_series_dir，直接忽略
    if not series_uid or not dicom_series_dir:
        print("未提供series_uid或dicom_series_dir，忽略此NII文件")
        return [], []
    
    # 检查对应的DICOM文件夹是否存在
    dicom_folder_path = os.path.join(dicom_series_dir, series_uid)
    if not os.path.exists(dicom_folder_path):
        print(f"未找到DICOM文件夹: {dicom_folder_path}，忽略此NII文件")
        return [], []
    
    # 记录总处理时间
    total_start_time = time.time()
    
    try:
        # 快速扫描目录获取.dcm文件
        print("扫描DICOM文件...")
        dicom_files = glob.glob(os.path.join(dicom_folder_path, "*.dcm"))
        dicom_files.extend(glob.glob(os.path.join(dicom_folder_path, "*.DCM")))
        
        if len(dicom_files) == 0:
            print("DICOM文件夹中没有找到有效的DICOM文件，忽略此NII文件")
            return [], []
        
        # 排序以建立稳定映射
        dicom_files = sorted(dicom_files)
        dicom_basenames = [os.path.basename(p) for p in dicom_files]
        
        # 检查NII的Z维与DICOM文件数量是否匹配（假定NII为 (W, H, Z)）
        nii_shape = volume_3d.shape
        print(f"NII体积形状: {nii_shape}, DICOM文件数量: {len(dicom_files)}")
        
        if len(nii_shape) < 3 or nii_shape[2] != len(dicom_files):
            print(f"NII的Z维 ({nii_shape[2] if len(nii_shape)>=3 else 'N/A'}) 与DICOM文件数量 ({len(dicom_files)}) 不匹配")
            return [], []
        
        # 调整NII的维度顺序以匹配DICOM（(W,H,Z) -> (Z,H,W)）
        print("调整NII维度顺序以匹配DICOM文件...")
        aligned_seg_vol = np.transpose(volume_3d, (2, 1, 0))  # 将NII从(W, H, Z)调整为(Z, H, W)
        
        print(f"调整后的NII体积形状: {aligned_seg_vol.shape} (Z, H, W)")
        
        # 向量化处理：一次性处理所有切片
        z_slices = aligned_seg_vol.shape[0]
        
        # 创建一个存储所有切片的列表
        all_slices = []
        
        for i in range(z_slices):
            slice_2d = aligned_seg_vol[i]
            all_slices.append(slice_2d)
        
        # 检查是否需要调整大小
        if aligned_seg_vol.shape[1:] != (target_size, target_size):
            start_time = time.time()
            print(f"调整到目标尺寸: {aligned_seg_vol.shape[1:]} -> ({target_size}, {target_size})")
            resized_slices = []
            
            # 批量调整大小
            for i in range(0, len(all_slices), 32):  # 每批处理32个切片
                batch_end = min(i + 32, len(all_slices))
                batch_slices = all_slices[i:batch_end]
                
                # 批量调整大小
                batch_resized = np.array([
                    cv2.resize(slice_2d, (target_size, target_size), 
                             interpolation=cv2.INTER_NEAREST)
                    for slice_2d in batch_slices
                ])
                resized_slices.append(batch_resized)
            
            # 合并所有批次
            all_slices = np.vstack(resized_slices)
            resize_time = time.time() - start_time
            print(f"图像resize耗时: {resize_time:.3f}秒")
        
        # 转换为列表格式
        slices = [all_slices[i] for i in range(len(all_slices))]
        
        total_time = time.time() - total_start_time
        print(f"=== 总处理耗时: {total_time:.3f}秒 ===")
        print(f"成功生成了{len(slices)}个对齐的2D切片")
        return slices, dicom_basenames
        
    except Exception as e:
        total_time = time.time() - total_start_time
        print(f"=== 处理失败，总耗时: {total_time:.3f}秒 ===")
        print(f"OpenCV处理过程中出错: {e}，忽略此NII文件")
        return [], []





def process_2d_slice_intelligent(slice_2d: np.ndarray, target_size: int) -> np.ndarray:
    """处理单个2D切片：调整大小并应用二值化"""
    # 调整到目标大小
    if slice_2d.shape != (target_size, target_size):
        slice_2d = cv2.resize(slice_2d, (target_size, target_size), 
                            interpolation=cv2.INTER_NEAREST)  # 对分割掩码使用最近邻插值
    
    # 应用二值化
    slice_2d = apply_binary_thresholding(slice_2d, threshold=0)
    
    return slice_2d

def analyze_3d_volume_structure(volume_3d: np.ndarray, series_uid: str = None, dicom_series_dir: str = None) -> dict:
    """使用SimpleITK分析3D体积结构
    
    Args:
        volume_3d: 3D分割体积
        series_uid: 序列UID
        dicom_series_dir: DICOM序列目录
    
    Returns:
        包含分析结果的字典
    """
    if volume_3d is None:
        return {}
    
    analysis = {
        'volume_shape': volume_3d.shape,
        'dicom_file_count': 0,
        'alignment_successful': False,
        'aligned_shape': None,
        'confidence': 'none'
    }
    
    if series_uid and dicom_series_dir:
        dicom_folder_path = os.path.join(dicom_series_dir, series_uid)
        if os.path.exists(dicom_folder_path):
            try:
                # 使用SimpleITK读取DICOM序列
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
                
                if len(dicom_names) > 0:
                    reader.SetFileNames(dicom_names)
                    dicom_image_sitk = reader.Execute()
                    dicom_vol = sitk.GetArrayFromImage(dicom_image_sitk)
                    
                    analysis['dicom_file_count'] = len(dicom_names)
                    analysis['dicom_shape'] = dicom_vol.shape
                    
                    # 尝试对齐
                    seg_image_sitk = sitk.GetImageFromArray(volume_3d)
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(dicom_image_sitk)
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampled_seg_sitk = resampler.Execute(seg_image_sitk)
                    aligned_seg_vol = sitk.GetArrayFromImage(resampled_seg_sitk)
                    
                    analysis['aligned_shape'] = aligned_seg_vol.shape
                    
                    # 检查对齐是否成功
                    if aligned_seg_vol.shape == dicom_vol.shape:
                        analysis['alignment_successful'] = True
                        analysis['confidence'] = 'high'
                    else:
                        analysis['alignment_successful'] = False
                        analysis['confidence'] = 'low'
                        
            except Exception as e:
                analysis['error'] = str(e)
                analysis['confidence'] = 'none'
        else:
            analysis['confidence'] = 'none'
    else:
        analysis['confidence'] = 'none'
    
    return analysis

print("✅ 已创建智能3D到2D转换函数")
print("   - convert_3d_to_2d_slices_intelligent: 根据NII形状和DICOM文件数量智能选择切分轴")
print("   - process_2d_slice_intelligent: 处理单个2D切片")
print("   - analyze_3d_volume_structure: 分析3D体积结构并提供切分建议")

def create_segmentation_mask_from_labels(labels: np.ndarray, class_mapping: dict) -> np.ndarray:
    """Create binary segmentation mask from labels (0=background, 1=foreground)"""
    mask = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
    
    # Check if any anatomical location is present
    has_aneurysm = any(labels == 1)
    
    if has_aneurysm:
        # Create a simple foreground region in the center
        # In practice, you'd use actual segmentation data
        center_y, center_x = config.IMAGE_SIZE // 2, config.IMAGE_SIZE // 2
        y_start = max(0, center_y - 15)
        y_end = min(config.IMAGE_SIZE, center_y + 15)
        x_start = max(0, center_x - 15)
        x_end = min(config.IMAGE_SIZE, center_x + 15)
        mask[y_start:y_end, x_start:x_end] = 1  # Binary: 1 = foreground
    
    return mask

def list_available_nii_files(segmentation_dir: str) -> List[str]:
    """List all available NII files in the segmentation directory for debugging"""
    nii_files = []
    try:
        for root, dirs, files in os.walk(segmentation_dir):
            for file in files:
                if file.endswith(('.nii', '.nii.gz')):
                    nii_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error listing NII files: {e}")
    
    return nii_files

def find_available_segmentation_dir(alternative_paths: List[str]) -> Optional[str]:
    """Find the first available segmentation directory from a list of alternatives"""
    for path in alternative_paths:
        if os.path.exists(path):
            nii_files = list_available_nii_files(path)
            if len(nii_files) > 0:
                print(f"Found segmentation directory: {path} with {len(nii_files)} NII files")
                return path
            else:
                print(f"Directory exists but no NII files found: {path}")
        else:
            print(f"Directory does not exist: {path}")
    
    print("No valid segmentation directory found!")
    return None







def extract_dicom_patient_info(series_uid: str) -> Tuple[str, str]:
    """Extract StudyInstanceUID and PatientID from DICOM metadata"""
    try:
        dicom_dir = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_uid}"
        if os.path.exists(dicom_dir):
            dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            if dcm_files:
                ds = pydicom.dcmread(
                    os.path.join(dicom_dir, dcm_files[0]), 
                    stop_before_pixels=True, 
                    force=True
                )
                study_uid = getattr(ds, 'StudyInstanceUID', None)
                patient_id = getattr(ds, 'PatientID', None)
                return study_uid or f"fallback_{series_uid[:32]}", patient_id
    except Exception:
        pass
    
    # Fallback: use longer prefix from series UID
    return f"fallback_{series_uid[:32]}", f"fallback_{series_uid[:32]}"

@functools.lru_cache(maxsize=5000)
def get_patient_group_cached(series_uid: str) -> str:
    """Get patient group with caching for performance"""
    study_uid, patient_id = extract_dicom_patient_info(series_uid)
    # Use StudyInstanceUID as primary identifier
    return study_uid if study_uid and not study_uid.startswith('fallback_') else patient_id

# 更新数据映射函数以使用智能3D到2D转换
def create_segmentation_data_mapping_intelligent():
    """进一步优化版分割数据映射 - 直接遍历NII文件匹配训练数据"""
    segmentation_mapping = {}
    found_nii_count = 0
    placeholder_count = 0
    skipped_no_positive = 0
    skipped_no_dicom = 0
    
    print("创建进一步优化版分割数据映射 - 直接遍历NII文件...")
    
    # 尝试找到可用的分割目录
    segmentation_dir = find_available_segmentation_dir(config.SEGMENTATION_DIR_ALTERNATIVES)
    if segmentation_dir is None:
        print("警告: 未找到分割目录。将仅使用占位符掩码。")
        segmentation_dir = config.SEGMENTATION_DIR  # 用于占位符创建的默认路径
    
    print(f"使用分割目录: {segmentation_dir}")
    
    # 一次性获取所有NII文件
    all_nii_files = list_available_nii_files(segmentation_dir)
    print(f"找到 {len(all_nii_files)} 个NII文件")
    
    # 创建训练数据的快速查找字典（不过滤阳性，全部纳入）
    train_data_dict = {}
    for _, row in train_df.iterrows():
        series_uid = row['SeriesInstanceUID']
        train_data_dict[series_uid] = row
    
    print(f"训练数据中共有 {len(train_data_dict)} 个序列")
    
    # 直接遍历NII文件进行处理
    print(f"\\n直接遍历NII文件进行匹配...")
    for nii_file in tqdm(all_nii_files, desc="处理NII文件"):
        # 从NII文件名提取series_uid
        if '_cowseg' in nii_file:
            series_uid = nii_file.split('/')[-1].replace('.nii','').replace('_cowseg','')
        else:
            continue
        # if os.path.exist'/kaggle/input/rsna-intracranial-aneurysm-detection/series'+series_uid)
        # if series_uid is None:
        #     print(f"⚠️  无法解析NII文件名: {os.path.basename(nii_file)}")
        #     continue
        
        # 检查这个series_uid是否在训练数据中（不过滤阳性）
        if series_uid not in train_data_dict:
            skipped_no_positive += 1
            continue
        
        # # 验证对应的DICOM文件夹是否存在
        # if not validate_dicom_folder_exists(series_uid, config.DICOM_SERIES_DIR):
        #     skipped_no_dicom += 1
        #     print(f"⚠️  {series_uid}: 未找到对应的DICOM文件夹")
        #     continue
        
        # 获取训练数据行
        train_row = train_data_dict[series_uid]
        
        # 加载分割数据
        segmentation_3d = load_nii_segmentation(nii_file)
        if segmentation_3d is not None:
            # 分析3D体积结构
            # analysis = analyze_3d_volume_structure(segmentation_3d, series_uid, config.DICOM_SERIES_DIR)
            
            # 使用智能转换转换为2D切片（返回切片与对应DICOM文件名）
            slices_2d, dicom_filenames = convert_3d_to_2d_slices_intelligent(
                segmentation_3d, config.IMAGE_SIZE, series_uid, config.DICOM_SERIES_DIR
            )
            
            if len(slices_2d) > 0:  # 确保成功生成了切片
                # 构建文件名 -> 切片 的映射
                slice_map = {fname: slices_2d[i] for i, fname in enumerate(dicom_filenames)}
                
                # 过滤全零掩码切片（仅保留含前景像素的切片）
                nonzero_indices = [i for i, s in enumerate(slices_2d) if (np.asarray(s) > 0).any()]
                if len(nonzero_indices) != len(slices_2d):
                    print(f"{series_uid}: 过滤掉 {len(slices_2d)-len(nonzero_indices)} 个全零掩码切片")
                
                if len(nonzero_indices) > 0:
                    slices_2d = [slices_2d[i] for i in nonzero_indices]
                    dicom_filenames = [dicom_filenames[i] for i in nonzero_indices]
                    slice_map = {dicom_filenames[i]: slices_2d[i] for i in range(len(slices_2d))}
                else:
                    # 若全部为零，置空（该序列将不会贡献样本）
                    slices_2d = []
                    dicom_filenames = []
                    slice_map = {}
                
                segmentation_mapping[series_uid] = {
                    'segmentation_file': nii_file,
                    'slices_2d': slices_2d,
                    'slice_map': slice_map,  # 新增：文件名到切片的映射（已过滤零掩码）
                    'dicom_filenames': dicom_filenames,  # 新增：已排序的文件名列表（与切片同步）
                    'labels': train_row[TARGET_COLS].values,
                    'volume_analysis': None
                }
                found_nii_count += 1
                
                if found_nii_count <= 5:  # 打印前5个匹配用于调试
                    print(f"✅ 处理NII文件: {os.path.basename(nii_file)}")
                    print(f"   对应序列: {series_uid}")
                    print(f"   生成了 {len(slices_2d)} 个2D切片")
            else:
                print(f"⚠️  {series_uid}: NII文件加载成功但未生成2D切片")
        else:
            print(f"❌ 加载NII文件失败: {nii_file}")
    
    # 为有阳性标签但没有NII文件的序列创建占位符
    print(f"\\n为没有NII文件的序列创建占位符...")
    for series_uid, train_row in train_data_dict.items():
        if series_uid not in segmentation_mapping:
            placeholder_mask = create_segmentation_mask_from_labels(
                train_row[TARGET_COLS].values, CLASS_MAPPING
            )
            segmentation_mapping[series_uid] = {
                'segmentation_file': None,
                'slices_2d': [placeholder_mask],  # 单个切片占位符
                'labels': train_row[TARGET_COLS].values,
                'volume_analysis': None
            }
            placeholder_count += 1
    
    print(f"\\n进一步优化版分割映射摘要:")
    print(f"- 总NII文件数: {len(all_nii_files)}")
    print(f"- 成功处理NII文件: {found_nii_count}")
    print(f"- 跳过(不在训练数据中): {skipped_no_positive}")
    print(f"- 跳过(无DICOM文件夹): {skipped_no_dicom}")
    print(f"- 使用占位符: {placeholder_count}")
    print(f"- 最终映射序列数: {len(segmentation_mapping)}")
    print(f"- NII文件利用率: {found_nii_count}/{len(all_nii_files)} = {found_nii_count/len(all_nii_files)*100:.1f}%")
    print(f"- 训练数据覆盖率: {len(segmentation_mapping)}/{len(train_data_dict)} = {len(segmentation_mapping)/len(train_data_dict)*100:.1f}%")
    
    return segmentation_mapping

print("✅ 已创建智能数据映射函数")
print("   - create_segmentation_data_mapping_intelligent: 使用智能3D到2D转换")

# Create segmentation data mapping
segmentation_data_dict = create_segmentation_data_mapping_intelligent()
print(f"Created segmentation mapping for {len(segmentation_data_dict)} series")

# Filter data to only include series with segmentation data
valid_series = list(segmentation_data_dict.keys())
train_df_filtered = train_df[train_df['SeriesInstanceUID'].isin(valid_series)].copy()
print(f"Filtered train data shape: {train_df_filtered.shape}")

# Check distribution
positive_labels_count = train_df_filtered[TARGET_COLS].sum().sum()
print(f"Total positive labels: {positive_labels_count}")
print(f"Average positive labels per series: {positive_labels_count / len(train_df_filtered):.2f}")






# 2D灰度图像二分类数据增强
print("=== 配置2D灰度图像二分类数据增强 ===")

if config.USE_STRONG_AUGMENTATION:
    print("使用强数据增强 - 针对2D灰度医学图像二分类优化")
    train_transform = A.Compose([
        # === 几何变换 (医学图像安全) ===
        # 旋转 - 医学图像通常可以小幅旋转
        A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
        
        # 翻转 - 对于脑血管图像，水平翻转是安全的
        A.HorizontalFlip(p=0.5),
        
        # 缩放和平移 - 模拟不同的扫描视野
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.15, 
            rotate_limit=10, 
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.6
        ),
        
        
        # === 图像质量变化 (模拟不同扫描仪/协议) ===
        # 亮度和对比度调整
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.6
        ),
        
        # CLAHE - 自适应直方图均衡化
        A.CLAHE(
            clip_limit=2.0, 
            tile_grid_size=(8, 8), 
            p=0.4
        ),
        
        # Gamma校正 - 模拟不同的显示特性
        A.RandomGamma(
            gamma_limit=(85, 115), 
            p=0.4
        ),
        
        # === 噪声模拟 (扫描仪差异) ===
        # 高斯噪声 - 模拟热噪声
        A.GaussNoise(
            var_limit=(5, 25), 
            mean=0,
            p=0.3
        ),
        
        # 模糊 - 模拟运动伪影或低分辨率
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
 
        
        # # 像素级dropout - 模拟噪声像素
        # A.PixelDropout(
        #     dropout_prob=0.01,
        #     p=0.1
        # ),
        
        # === 归一化和张量转换 ===
        # 针对灰度图像的归一化 (ImageNet单通道统计)
        # A.Normalize(
        #     mean=[0.485],  # 灰度图像单通道
        #     std=[0.229]
        # ),
        ToTensorV2()
    ])
else:
    print("使用标准数据增强 - 2D灰度医学图像二分类")
    train_transform = A.Compose([
        # 基础几何变换
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.5),
        
        # 基础图像质量调整
        A.RandomBrightnessContrast(
            brightness_limit=0.15, 
            contrast_limit=0.15, 
            p=0.4
        ),
        
        # 轻微噪声
        A.GaussNoise(
            var_limit=(5, 20), 
            p=0.2
        ),
        
        # 归一化
        # A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ])

# 验证集变换 - 仅归一化
val_transform = A.Compose([
    A.Normalize(mean=[0.485], std=[0.229]),  # 灰度图像归一化
    ToTensorV2()
])

print(f"训练变换: {'强增强' if config.USE_STRONG_AUGMENTATION else '标准增强'}")
print(f"验证变换: 仅归一化")
print("✅ 2D灰度图像二分类数据增强配置完成")



# 2D灰度图像二分类专用数据增强工具函数
print("=== 二分类任务专用数据增强工具 ===")

def get_classification_transforms_2d_grayscale(image_size=224, is_training=True, augmentation_level='strong'):
    """
    为2D灰度医学图像二分类任务创建专用的数据增强管道
    
    Args:
        image_size (int): 目标图像尺寸
        is_training (bool): 是否为训练模式
        augmentation_level (str): 增强级别 ('none', 'light', 'medium', 'strong')
    
    Returns:
        albumentations.Compose: 数据增强管道
    """
    
    if not is_training:
        # 验证/测试时只进行归一化
        return A.Compose([
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ])
    
    # 训练时的数据增强
    transforms_list = []
    
    if augmentation_level == 'none':
        # 无增强，仅归一化
        transforms_list = [
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ]
    
    elif augmentation_level == 'light':
        # 轻度增强 - 仅基础变换
        transforms_list = [
            # 几何变换
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # 图像质量
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            
            # 归一化
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ]
    
    elif augmentation_level == 'medium':
        # 中等增强
        transforms_list = [
            # 几何变换
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4
            ),
            
            # 图像质量
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            
            # 噪声
            A.GaussNoise(var_limit=(5, 20), p=0.2),
            
            # 归一化
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ]
    
    elif augmentation_level == 'strong':
        # 强增强 - 完整的增强管道
        transforms_list = [
            # === 几何变换 ===
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ShiftScaleRotate(
                shift_limit=0.15, scale_limit=0.2, rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.6
            ),
        
            
            # === 图像质量变换 ===
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            
            # === 噪声和模糊 ===
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            

            # 归一化
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ]
    
    return A.Compose(transforms_list)

def get_test_time_augmentation_2d_classification():
    """
    为2D灰度图像二分类创建测试时增强(TTA)变换
    
    Returns:
        list: TTA变换列表
    """
    tta_transforms = [
        # 原始图像
        A.Compose([
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ]),
        
        # 水平翻转
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ]),
        
        # 垂直翻转
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ]),
        
        # 90度旋转
        A.Compose([
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ]),
        
        # 水平+垂直翻转
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ])
    ]
    
    return tta_transforms

def visualize_augmentations_2d_classification(image, transforms, num_samples=4):
    """
    可视化2D灰度图像二分类的数据增强效果
    
    Args:
        image (np.ndarray): 输入图像 (H, W)
        transforms (albumentations.Compose): 数据增强管道
        num_samples (int): 生成的样本数量
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(20, 4))
    
    # 显示原始图像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 生成增强样本
    for i in range(num_samples):
        # 应用增强（需要移除最后的归一化和ToTensor步骤用于可视化）
        aug_transforms = A.Compose(transforms.transforms[:-2])  # 移除Normalize和ToTensorV2
        augmented = aug_transforms(image=image)
        aug_image = augmented['image']
        
        # 显示增强后的图像
        axes[i + 1].imshow(aug_image, cmap='gray')
        axes[i + 1].set_title(f'增强样本 {i + 1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# 应用新的数据增强配置
augmentation_level = 'strong' if config.USE_STRONG_AUGMENTATION else 'medium'

train_transform = get_classification_transforms_2d_grayscale(
    image_size=config.IMAGE_SIZE,
    is_training=True,
    augmentation_level=augmentation_level
)

val_transform = get_classification_transforms_2d_grayscale(
    image_size=config.IMAGE_SIZE,
    is_training=False
)

print(f"✅ 已配置 {augmentation_level} 级别的2D灰度图像二分类数据增强")
print(f"   - 训练变换: {len(train_transform.transforms)} 个步骤")
print(f"   - 验证变换: {len(val_transform.transforms)} 个步骤")
print(f"   - 目标图像尺寸: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
print("   - 支持测试时增强(TTA)和可视化功能")



# 测试和可视化2D灰度图像二分类数据增强
print("=== 测试2D灰度图像二分类数据增强效果 ===")

# 创建测试用的虚拟图像
def create_test_sample():
    """创建测试用的2D灰度图像"""
    # 创建一个模拟的脑血管图像
    test_image = np.zeros((224, 224), dtype=np.uint8)
    
    # 添加一些结构（模拟血管）
    # 主血管
    cv2.line(test_image, (50, 50), (170, 170), 200, 3)
    cv2.line(test_image, (170, 50), (50, 170), 180, 2)
    
    # 分支血管
    cv2.line(test_image, (100, 30), (100, 100), 160, 2)
    cv2.line(test_image, (30, 100), (100, 100), 160, 2)
    
    # 添加一些噪声和纹理
    noise = np.random.normal(0, 20, (224, 224))
    test_image = np.clip(test_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return test_image

# 测试不同增强级别
augmentation_levels = ['none', 'light', 'medium', 'strong']

for level in augmentation_levels:
    print(f"\n--- {level.upper()} 级别数据增强 ---")
    
    # 创建对应级别的变换
    transform = get_classification_transforms_2d_grayscale(
        image_size=config.IMAGE_SIZE,
        is_training=True,
        augmentation_level=level
    )
    
    print(f"变换步骤数: {len(transform.transforms)}")
    print("变换列表:")
    for i, t in enumerate(transform.transforms):
        print(f"  {i+1}. {type(t).__name__}")

# 可视化数据增强效果（如果需要）
if hasattr(config, 'VISUALIZE_AUGMENTATION') and config.VISUALIZE_AUGMENTATION:
    print("\n=== 可视化数据增强效果 ===")
    
    # 创建测试样本
    test_image = create_test_sample()
    
    # 可视化强增强效果
    strong_transform = get_classification_transforms_2d_grayscale(
        image_size=config.IMAGE_SIZE,
        is_training=True,
        augmentation_level='strong'
    )
    
    print("生成数据增强可视化...")
    # 注意：这里需要在有matplotlib的环境中运行
    # visualize_augmentations_2d_classification(test_image, strong_transform, num_samples=4)

# 测试数据增强的性能
print("\n=== 数据增强性能测试 ===")

import time

# 创建测试数据
test_image = create_test_sample()

# 测试不同级别的增强速度
for level in ['light', 'medium', 'strong']:
    transform = get_classification_transforms_2d_grayscale(
        image_size=config.IMAGE_SIZE,
        is_training=True,
        augmentation_level=level
    )
    
    # 性能测试
    start_time = time.time()
    for i in range(100):
        # 移除最后两个步骤（Normalize和ToTensorV2）进行速度测试
        test_transform = A.Compose(transform.transforms[:-2])
        augmented = test_transform(image=test_image)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
    
    print(f"{level.upper()} 级别: {avg_time:.2f}ms/样本 ({len(transform.transforms)} 个变换)")

# 验证数据增强的正确性
print("\n=== 数据增强正确性验证 ===")

# 测试图像变换
test_transform = get_classification_transforms_2d_grayscale(
    image_size=config.IMAGE_SIZE,
    is_training=True,
    augmentation_level='medium'
)

# 移除归一化步骤进行测试
test_transform_no_norm = A.Compose(test_transform.transforms[:-2])

for i in range(3):
    augmented = test_transform_no_norm(image=test_image)
    aug_image = augmented['image']
    
    print(f"测试 {i+1}:")
    print(f"  原始图像形状: {test_image.shape}, 数据类型: {test_image.dtype}")
    print(f"  增强图像形状: {aug_image.shape}, 数据类型: {aug_image.dtype}")
    print(f"  像素值范围: {aug_image.min():.2f} - {aug_image.max():.2f}")
    
    # 验证图像形状和数据类型
    assert aug_image.shape == test_image.shape, "图像形状不匹配"
    assert aug_image.dtype == test_image.dtype, "数据类型不匹配"

print("\n✅ 数据增强正确性验证通过")
print("✅ 2D灰度图像二分类数据增强配置和测试完成")

# 最终配置确认
print(f"\n=== 最终配置确认 ===")
print(f"当前增强级别: {'strong' if config.USE_STRONG_AUGMENTATION else 'medium'}")
print(f"训练变换步骤: {len(train_transform.transforms)}")
print(f"验证变换步骤: {len(val_transform.transforms)}")
print(f"图像尺寸: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
print(f"输入通道数: 1 (灰度图像)")
print(f"任务类型: 二分类 (0/1)")
print(f"支持TTA: 是")
# print(f"支持可视化: 是")"



def create_robust_cv_split(train_df, n_splits=5):
    """Create robust cross-validation split with true patient separation from DICOM"""
    
    print("Creating patient-separated cross-validation split...")
    print("Extracting true patient IDs from DICOM metadata...")
    print("This will take a few minutes but ensures proper patient separation.")
    
    # Extract true patient groups from DICOM metadata
    patient_groups = []
    for series_uid in tqdm(train_df['SeriesInstanceUID'], desc="Reading DICOM patient info"):
        patient_group = get_patient_group_cached(series_uid)
        patient_groups.append(patient_group)
    
    # Add patient groups to dataframe
    train_df = train_df.copy()
    train_df['patient_id'] = patient_groups
    
    n_groups = train_df['patient_id'].nunique()
    print(f"True patient groups found: {n_groups}")
    
    # Check if we have enough patient groups
    if n_groups < n_splits:
        print(f"Not enough patient groups ({n_groups}) for {n_splits}-fold CV.")
        print("Falling back to StratifiedKFold...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(skf.split(train_df, train_df['Aneurysm Present']))
    
    # Create stratification key combining modality and aneurysm presence
    train_df['stratify_key'] = (
        train_df['Modality'].astype(str) + '_' + 
        train_df['Aneurysm Present'].astype(str)
    )
    
    print(f"Stratification keys: {train_df['stratify_key'].unique()}")
    
    # Use GroupKFold to ensure patient-level separation
    group_kfold = GroupKFold(n_splits=n_splits)
    
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(group_kfold.split(
        train_df, 
        groups=train_df['patient_id']
    )):
        # Validate patient separation
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # Check for patient overlap (should be 0)
        train_patients = set(train_fold['patient_id'])
        val_patients = set(val_fold['patient_id'])
        overlap = train_patients.intersection(val_patients)
        
        train_dist = train_fold['Aneurysm Present'].value_counts(normalize=True)
        val_dist = val_fold['Aneurysm Present'].value_counts(normalize=True)
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_fold)} samples ({len(train_patients)} patients)")
        print(f"  Val: {len(val_fold)} samples ({len(val_patients)} patients)")
        print(f"  Patient overlap: {len(overlap)} (should be 0!)")
        print(f"  Aneurysm Present - Train: {train_dist.get(1, 0):.3f}, Val: {val_dist.get(1, 0):.3f}")
        
        if len(overlap) > 0:
            print(f"  WARNING: Found {len(overlap)} overlapping patients!")
        
        splits.append((train_idx, val_idx))
    
    return splits

# Create robust train/validation split
cv_splits = create_robust_cv_split(train_df_filtered, config.NUM_FOLDS)
train_indices, val_indices = cv_splits[config.FOLD]

train_fold_df = train_df_filtered.iloc[train_indices]
val_fold_df = train_df_filtered.iloc[val_indices]

print(f"\nRobust CV Fold {config.FOLD} Summary:")
print(f"Train fold size: {len(train_fold_df)}")
print(f"Validation fold size: {len(val_fold_df)}")

# Check distributions
print(f"Train Aneurysm Present: {train_fold_df['Aneurysm Present'].value_counts().to_dict()}")
print(f"Val Aneurysm Present: {val_fold_df['Aneurysm Present'].value_counts().to_dict()}")

# Check modality distribution
print(f"Train Modality distribution: {train_fold_df['Modality'].value_counts().to_dict()}")
print(f"Val Modality distribution: {val_fold_df['Modality'].value_counts().to_dict()}")


class SegmentationDataset(Dataset):
    """Dataset for 2D U-Net segmentation training with foreground pixel filtering"""
    def __init__(self, df, segmentation_data_dict, series_mapping_df=None, 
                 transform=None, is_training=True, min_foreground_ratio=0.01):
        self.df = df.reset_index(drop=True)
        self.segmentation_data_dict = segmentation_data_dict
        self.series_mapping_df = series_mapping_df
        self.transform = transform
        self.is_training = is_training
        self.min_foreground_ratio = min_foreground_ratio  # 最小前景像素比例
        
        # Create list of (series_uid, slice_idx) pairs for all available slices
        # 在初始化时就过滤掉前景像素太少的切片
        self.samples = []
        self._filter_samples()
        
        # Simple LRU cache for recently accessed data
        self._cache = {}
        self._cache_keys = []
        self._max_cache_size = config.CACHE_SIZE
        
        print(f"数据集初始化完成: {len(self.samples)} 个有效样本 (最小前景像素比例: {self.min_foreground_ratio})")
    
    def _filter_samples(self):
        """过滤掉前景像素比例太低的切片"""
        total_slices = 0
        filtered_slices = 0
        
        for series_uid in self.df['SeriesInstanceUID'].unique():
            if series_uid in self.segmentation_data_dict:
                slices_2d = self.segmentation_data_dict[series_uid]['slices_2d']
                num_slices = len(slices_2d)
                total_slices += num_slices
                
                for slice_idx in range(num_slices):
                    # 检查前景像素比例
                    mask = slices_2d[slice_idx]
                    foreground_ratio = np.sum(mask > 0) / mask.size
                    
                    # 只保留前景像素比例足够的切片
                    if foreground_ratio >= self.min_foreground_ratio:
                        self.samples.append((series_uid, slice_idx))
                        filtered_slices += 1
        
        print(f"切片过滤统计: 总切片 {total_slices}, 保留切片 {filtered_slices}, 过滤率 {(total_slices-filtered_slices)/total_slices*100:.1f}%")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]
        
        series_uid, slice_idx = self.samples[idx]
        row = self.df[self.df['SeriesInstanceUID'] == series_uid].iloc[0]
        
        # Get segmentation mask
        mask = self.segmentation_data_dict[series_uid]['slices_2d'][slice_idx]
        
        # 再次检查前景像素比例（双重保险）
        foreground_ratio = np.sum(mask > 0) / mask.size
        if foreground_ratio < self.min_foreground_ratio:
            # 如果前景像素太少，返回一个随机的有效样本
            return self.__getitem__(np.random.randint(0, len(self.samples)))
        
        # Load corresponding image slice
        image = self._load_image_slice(series_uid, slice_idx, row)
        
        # Extract metadata
        metadata = self._extract_metadata(row)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to tensor
        mask_tensor = mask.long()
        
        result = (image, mask_tensor, metadata)
        
        # Update cache
        self._update_cache(idx, result)
        
        return result
    
    def _update_cache(self, idx, data):
        """Update LRU cache"""
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_idx = self._cache_keys.pop(0)
            del self._cache[oldest_idx]
        
        self._cache[idx] = data
        self._cache_keys.append(idx)
    
    def _extract_metadata(self, row):
        """Extract metadata from row"""
        metadata = {
            'age': row.get('Patient Age', 0),
            'sex': 1 if row.get('Patient Sex', 'M') == 'M' else 0,
            'modality': row.get('Modality', 'CTA')
        }
        return metadata
    
    def _load_image_slice(self, series_uid: str, slice_idx: int, row) -> np.ndarray:
        """Load real DICOM image slice"""
        try:
            # 构建DICOM文件路径
            dicom_dir = os.path.join(config.DICOM_SERIES_DIR, series_uid)
            
            if not os.path.exists(dicom_dir):
                print(f"Warning: DICOM directory not found: {dicom_dir}")
                return self._create_fallback_image()
            
            # 获取DICOM文件列表
            dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
            if not dcm_files:
                print(f"Warning: No DICOM files found in {dicom_dir}")
                return self._create_fallback_image()
            
            # 按文件名排序确保顺序一致
            dcm_files.sort()
            
            # 检查slice_idx是否在有效范围内
            if slice_idx >= len(dcm_files):
                print(f"Warning: slice_idx {slice_idx} out of range for {series_uid}")
                return self._create_fallback_image()
            
            # 加载对应的DICOM文件
            dcm_path = os.path.join(dicom_dir, dcm_files[slice_idx])
            
            try:
                # 使用pydicom读取DICOM文件
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
                img = ds.pixel_array.astype(np.float32)
                
                # 关键检查：跳过3D DICOM文件
                if len(img.shape) != 2:
                    # print(f"跳过3D DICOM文件: {dcm_path}, 形状: {img.shape}")
                    return self._create_fallback_image()
                
                # 检查图像尺寸有效性
                if img.shape[0] == 0 or img.shape[1] == 0:
                    print(f"无效图像尺寸: {img.shape}")
                    return self._create_fallback_image()
                
                # 获取模态信息
                modality = row.get('Modality', 'CTA')
                
                # 应用窗宽窗位
                if config.USE_WINDOWING:
                    window_center, window_width = get_windowing_params(modality)
                    img = apply_dicom_windowing(img, window_center, window_width)
                else:
                    # 使用鲁棒归一化
                    img = robust_normalization(img)
                
                # 应用CLAHE增强
                if config.USE_CLAHE:
                    img = apply_clahe_normalization(img, modality)
                
                # 调整到目标尺寸
                if img.shape != (config.IMAGE_SIZE, config.IMAGE_SIZE):
                    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE), 
                                   interpolation=cv2.INTER_AREA)
                
                # 最终检查：确保输出是2D
                if len(img.shape) != 2:
                    print(f"输出图像不是2D: {img.shape}")
                    return self._create_fallback_image()
                
                return img.astype(np.uint8)
                
            except Exception as e:
                print(f"Error reading DICOM file {dcm_path}: {e}")
                return self._create_fallback_image()
                
        except Exception as e:
            print(f"Error loading image slice {slice_idx} for {series_uid}: {e}")
            return self._create_fallback_image()
    
    def _create_fallback_image(self) -> np.ndarray:
        """创建备用图像（当无法加载真实DICOM时）"""
        # 创建一个简单的测试图像而不是随机噪声
        img = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
        
        # 添加一些简单的几何形状作为测试
        center = config.IMAGE_SIZE // 2
        cv2.circle(img, (center, center), 30, 128, -1)
        cv2.rectangle(img, (center-20, center-20), (center+20, center+20), 200, 2)
        
        return img



# Create segmentation datasets
print("Creating segmentation datasets...")
train_dataset = SegmentationDataset(
    train_fold_df, 
    segmentation_data_dict, 
    series_mapping_df=None,  # 不再依赖series_mapping_df
    transform=train_transform,
    is_training=True,
    min_foreground_ratio=0.0
)

val_dataset = SegmentationDataset(
    val_fold_df,
    segmentation_data_dict,
    series_mapping_df=None,  # 不再依赖series_mapping_df
    transform=val_transform,
    is_training=False,
    min_foreground_ratio=0.0
)

# Create optimized data loaders
print("Creating optimized data loaders for segmentation...")
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
    drop_last=True,
    prefetch_factor=config.PREFETCH_FACTOR,
    persistent_workers=config.PERSISTENT_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
    prefetch_factor=config.PREFETCH_FACTOR,
    persistent_workers=config.PERSISTENT_WORKERS
)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# Test segmentation data loading speed and analyze label pixels
print("Testing segmentation data loading speed...")
import time

start_time = time.time()
label_stats = {
    'unique_values': set(),
    'value_counts': {},
    'total_pixels': 0,
    'non_zero_pixels': 0,
    'class_distribution': {i: 0 for i in range(config.NUM_CLASSES)}
}

for i, batch in enumerate(train_loader):
    if i >= 5:  # Test first 5 batches
        break
    images, masks, metadata = batch
    print(f"Batch {i+1}: Images shape: {images.shape}, Masks shape: {masks.shape}, Device: {images.device}")
    
    # 分析标签像素分布
    for mask in masks:
        # 转换为numpy进行分析
        mask_np = mask.numpy()
        
        # 收集唯一值
        unique_vals = np.unique(mask_np)
        label_stats['unique_values'].update(unique_vals)
        
        # 计算每个类别的像素数量
        for val in unique_vals:
            count = np.sum(mask_np == val)
            if val not in label_stats['value_counts']:
                label_stats['value_counts'][val] = 0
            label_stats['value_counts'][val] += count
            
            # 统计类别分布
            if val < config.NUM_CLASSES:
                label_stats['class_distribution'][val] += count
        
        # 统计总像素和非零像素
        total_pixels = mask_np.size
        non_zero_pixels = np.sum(mask_np > 0)
        label_stats['total_pixels'] += total_pixels
        label_stats['non_zero_pixels'] += non_zero_pixels

elapsed = time.time() - start_time
print(f"Loaded 5 batches in {elapsed:.2f} seconds ({elapsed/5:.2f}s per batch)")

# 打印标签像素分析结果
print("\n" + "="*60)
print("分割标签像素分析结果")
print("="*60)

print(f"发现的唯一标签值: {sorted(label_stats['unique_values'])}")
print(f"预期类别数: {config.NUM_CLASSES}")

print(f"\n标签值像素统计:")
for val in sorted(label_stats['value_counts'].keys()):
    count = label_stats['value_counts'][val]
    percentage = (count / label_stats['total_pixels']) * 100 if label_stats['total_pixels'] > 0 else 0
    class_name = list(CLASS_MAPPING.keys())[val] if val < len(CLASS_MAPPING) else f"未知类别{val}"
    print(f"  标签 {val} ({class_name}): {count:,} 像素 ({percentage:.2f}%)")

print(f"\n类别分布:")
for class_id in range(config.NUM_CLASSES):
    count = label_stats['class_distribution'][class_id]
    percentage = (count / label_stats['total_pixels']) * 100 if label_stats['total_pixels'] > 0 else 0
    class_name = list(CLASS_MAPPING.keys())[class_id] if class_id < len(CLASS_MAPPING) else f"类别{class_id}"
    print(f"  {class_name}: {count:,} 像素 ({percentage:.2f}%)")

print(f"\n总体统计:")
print(f"  总像素数: {label_stats['total_pixels']:,}")
print(f"  非零像素数: {label_stats['non_zero_pixels']:,}")
print(f"  背景像素数: {label_stats['total_pixels'] - label_stats['non_zero_pixels']:,}")
print(f"  非零像素比例: {(label_stats['non_zero_pixels'] / label_stats['total_pixels']) * 100:.2f}%")

# 检查数据平衡性
print(f"\n数据平衡性分析:")
background_pixels = label_stats['class_distribution'][0]
foreground_pixels = label_stats['non_zero_pixels'] - background_pixels
if foreground_pixels > 0:
    imbalance_ratio = background_pixels / foreground_pixels
    print(f"  背景/前景像素比例: {imbalance_ratio:.2f}:1")
    print(f"  数据不平衡程度: {'严重' if imbalance_ratio > 100 else '中等' if imbalance_ratio > 10 else '轻微'}")

print("\n" + "="*60)

# 检查图片和标签的压缩情况
print("\n检查图片和标签的压缩情况:")
print("="*60)

if len(train_dataset) > 0:
    sample_image, sample_mask, sample_metadata = train_dataset[0]
    
    # 检查图像
    if isinstance(sample_image, torch.Tensor):
        image_np = sample_image.squeeze().numpy()
        print(f"图像形状: {sample_image.shape}")
        print(f"图像数据类型: {sample_image.dtype}")
        print(f"图像值范围: {sample_image.min().item():.3f} - {sample_image.max().item():.3f}")
        print(f"图像是否归一化: {'是' if sample_image.min() >= 0 and sample_image.max() <= 1 else '否'}")
    else:
        image_np = sample_image
        print(f"图像形状: {sample_image.shape}")
        print(f"图像数据类型: {sample_image.dtype}")
        print(f"图像值范围: {sample_image.min():.3f} - {sample_image.max():.3f}")
    
    # 检查标签
    if isinstance(sample_mask, torch.Tensor):
        mask_np = sample_mask.numpy()
        print(f"标签形状: {sample_mask.shape}")
        print(f"标签数据类型: {sample_mask.dtype}")
        print(f"标签值范围: {sample_mask.min().item()} - {sample_mask.max().item()}")
    else:
        mask_np = sample_mask
        print(f"标签形状: {sample_mask.shape}")
        print(f"标签数据类型: {sample_mask.dtype}")
        print(f"标签值范围: {sample_mask.min()} - {sample_mask.max()}")
    
    # 检查标签的完整性
    unique_vals = np.unique(mask_np)
    print(f"标签唯一值: {unique_vals}")
    print(f"预期类别数: {config.NUM_CLASSES}")
    
    # 检查是否有压缩或数据丢失
    if len(unique_vals) > config.NUM_CLASSES:
        print(f"⚠️ 警告: 发现 {len(unique_vals)} 个唯一值，超过预期的 {config.NUM_CLASSES} 个类别")
    
    # 检查标签值是否连续
    expected_vals = set(range(config.NUM_CLASSES))
    actual_vals = set(unique_vals)
    missing_vals = expected_vals - actual_vals
    extra_vals = actual_vals - expected_vals
    
    if missing_vals:
        print(f"⚠️ 警告: 缺失的标签值: {missing_vals}")
    if extra_vals:
        print(f"⚠️ 警告: 额外的标签值: {extra_vals}")
    
    # 检查数据压缩
    print(f"\n数据压缩检查:")
    print(f"图像内存占用: {sample_image.element_size() * sample_image.nelement() / 1024:.2f} KB")
    print(f"标签内存占用: {sample_mask.element_size() * sample_mask.nelement() / 1024:.2f} KB")
    
    # 检查标签分布
    for val in unique_vals:
        count = np.sum(mask_np == val)
        percentage = (count / mask_np.size) * 100
        class_name = list(CLASS_MAPPING.keys())[val] if val < len(CLASS_MAPPING) else f"未知类别{val}"
        print(f"  标签 {val} ({class_name}): {count:,} 像素 ({percentage:.2f}%)")

print("\n" + "="*60)



class UNet2D(nn.Module):
    """2D U-Net architecture for medical image segmentation"""
    def __init__(self, in_channels=1, num_classes=14, base_features=64):
        super(UNet2D, self).__init__()
        self.num_classes = num_classes
        
        # Encoder (Contracting Path)
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        
        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(base_features * 16, base_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_features * 2, base_features)
        
        # Final classification layer
        self.final_conv = nn.Conv2d(base_features, num_classes, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.2)
        
    def _conv_block(self, in_channels, out_channels):
        """Convolutional block with two conv layers"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.dropout(bottleneck)
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final classification
        output = self.final_conv(dec1)
        
        return output

# Initialize 2D U-Net model
print("Initializing 2D U-Net model...")
model = UNet2D(
    in_channels=1,  # Grayscale input
    num_classes=config.NUM_CLASSES,
    base_features=64
)

model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model device: {next(model.parameters()).device}")



class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Convert to probabilities
        inputs = torch.softmax(inputs, dim=1)
        
        # Flatten tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedSegmentationLoss(nn.Module):
    """Combined Dice Loss and Cross Entropy Loss for segmentation"""
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None):
        super(CombinedSegmentationLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, inputs, targets):
        # Cross Entropy Loss
        ce_loss = self.ce_loss(inputs, targets)
        
        # Dice Loss (convert targets to one-hot for dice calculation)
        targets_one_hot = F.one_hot(targets, num_classes=config.NUM_CLASSES).permute(0, 3, 1, 2).float()
        dice_loss = self.dice_loss(inputs, targets_one_hot)
        
        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss

def calculate_dice_score(pred, target, num_classes, smooth=1e-5):
    """Calculate Dice score for each class"""
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    dice_scores = []
    for i in range(num_classes):
        pred_i = (pred == i).float()
        target_i = (target == i).float()
        
        intersection = (pred_i * target_i).sum()
        dice = (2. * intersection + smooth) / (pred_i.sum() + target_i.sum() + smooth)
        dice_scores.append(dice.item())
    
    return dice_scores

def calculate_iou(pred, target, num_classes, smooth=1e-5):
    """Calculate IoU for each class"""
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    iou_scores = []
    for i in range(num_classes):
        pred_i = (pred == i).float()
        target_i = (target == i).float()
        
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())
    
    return iou_scores

# Training setup
criterion = CombinedSegmentationLoss(
    dice_weight=config.DICE_WEIGHT,
    ce_weight=config.BCE_WEIGHT
)
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

print("Training setup complete")
print(f"Using loss function: {type(criterion).__name__}")
print(f"Dice weight: {config.DICE_WEIGHT}, CE weight: {config.BCE_WEIGHT}")



# 训练和验证函数定义
def train_epoch_segmentation(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps):
    """训练一个epoch的分割模型"""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    # 使用tqdm显示训练进度
    pbar = tqdm(train_loader, desc="训练中", leave=False)
    
    for batch_idx, (images, masks, metadata) in enumerate(pbar):
        images = images.to(device, non_blocking=True).half()
        masks = masks.to(device, non_blocking=True)
        
        # 前向传播
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 计算Dice和IoU
            dice_scores = calculate_dice_score(outputs, masks, config.NUM_CLASSES)
            iou_scores = calculate_iou(outputs, masks, config.NUM_CLASSES)
            
            # 平均Dice和IoU
            avg_dice = np.mean(dice_scores)
            avg_iou = np.mean(iou_scores)
        
        # 反向传播（梯度累积）
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        total_dice += avg_dice
        total_iou += avg_iou
        num_batches += 1
        
        # 更新进度条显示
        pbar.set_postfix({
            'Loss': f'{loss.item() * accumulation_steps:.4f}',
            'Dice': f'{avg_dice:.4f}',
            'IoU': f'{avg_iou:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_dice, avg_iou

def validate_epoch_segmentation(model, val_loader, criterion, device):
    """验证一个epoch的分割模型"""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    all_dice_per_class = [[] for _ in range(config.NUM_CLASSES)]
    all_iou_per_class = [[] for _ in range(config.NUM_CLASSES)]
    
    with torch.no_grad():
        # 使用tqdm显示验证进度
        pbar = tqdm(val_loader, desc="验证中", leave=False)
        
        for images, masks, metadata in pbar:
            images = images.to(device, non_blocking=True).half()
            masks = masks.to(device, non_blocking=True)
            
            # 前向传播
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # 计算Dice和IoU
                dice_scores = calculate_dice_score(outputs, masks, config.NUM_CLASSES)
                iou_scores = calculate_iou(outputs, masks, config.NUM_CLASSES)
                
                # 平均Dice和IoU
                avg_dice = np.mean(dice_scores)
                avg_iou = np.mean(iou_scores)
            
            total_loss += loss.item()
            total_dice += avg_dice
            total_iou += avg_iou
            num_batches += 1
            
            # 收集每个类别的分数
            for i in range(config.NUM_CLASSES):
                all_dice_per_class[i].append(dice_scores[i])
                all_iou_per_class[i].append(iou_scores[i])
            
            # 更新进度条显示
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{avg_dice:.4f}',
                'IoU': f'{avg_iou:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    
    # 计算每个类别的平均分数
    dice_per_class = [np.mean(scores) for scores in all_dice_per_class]
    iou_per_class = [np.mean(scores) for scores in all_iou_per_class]
    
    return avg_loss, avg_dice, avg_iou, dice_per_class, iou_per_class

def check_gpu_utilization():
    """检查GPU利用率"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
        return f"GPU内存: {gpu_memory:.2f}GB / {gpu_memory_max:.2f}GB"
    return "GPU不可用"

print("✅ 训练和验证函数已定义")
print("   - train_epoch_segmentation: 训练一个epoch")
print("   - validate_epoch_segmentation: 验证一个epoch")
print("   - check_gpu_utilization: 检查GPU利用率")



# 5-Fold Cross Validation Training Loop
print("=== 开始5折交叉验证训练 ===")
print(f"总折数: {config.NUM_FOLDS}")
print(f"将依次训练所有 {config.NUM_FOLDS} 个fold")

# 存储所有fold的结果
all_fold_results = []

# 循环训练每个fold
for fold_idx in range(config.NUM_FOLDS):
    print(f"\n{'='*60}")
    print(f"开始训练 Fold {fold_idx + 1}/{config.NUM_FOLDS}")
    print(f"{'='*60}")
    
    # 获取当前fold的数据划分
    train_indices, val_indices = cv_splits[fold_idx]
    
    train_fold_df = train_df_filtered.iloc[train_indices]
    val_fold_df = train_df_filtered.iloc[val_indices]
    
    print(f"Fold {fold_idx} 数据划分:")
    print(f"  训练集大小: {len(train_fold_df)}")
    print(f"  验证集大小: {len(val_fold_df)}")
    
    # 检查数据分布
    train_dist = train_fold_df['Aneurysm Present'].value_counts(normalize=True)
    val_dist = val_fold_df['Aneurysm Present'].value_counts(normalize=True)
    print(f"  训练集动脉瘤阳性比例: {train_dist.get(1, 0):.3f}")
    print(f"  验证集动脉瘤阳性比例: {val_dist.get(1, 0):.3f}")
    
    # 检查模态分布
    train_modality = train_fold_df['Modality'].value_counts().to_dict()
    val_modality = val_fold_df['Modality'].value_counts().to_dict()
    print(f"  训练集模态分布: {train_modality}")
    print(f"  验证集模态分布: {val_modality}")
    
    # 创建当前fold的数据集
    print(f"\n创建 Fold {fold_idx} 的数据集...")
    train_dataset = SegmentationDataset(
        train_fold_df, 
        segmentation_data_dict, 
        series_mapping_df=None,  # 不再依赖系列映射
        transform=train_transform,
        is_training=True,
        min_foreground_ratio=0.0
    )
    
    val_dataset = SegmentationDataset(
        val_fold_df,
        segmentation_data_dict,
        series_mapping_df=None,  # 不再依赖系列映射
        transform=val_transform,
        is_training=False,
        min_foreground_ratio=0.0
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR,
        persistent_workers=config.PERSISTENT_WORKERS
    )
    
    print(f"Fold {fold_idx} 数据加载器:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # 初始化模型（每个fold使用新的模型实例）
    print(f"\n初始化 Fold {fold_idx} 的模型...")
    model = UNet2D(
        in_channels=1,  # 灰度输入
        num_classes=config.NUM_CLASSES,
        base_features=64
    )
    model = model.to(device)
    
    # 训练设置
    criterion = CombinedSegmentationLoss(
        dice_weight=config.DICE_WEIGHT,
        ce_weight=config.BCE_WEIGHT
    )
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练循环变量
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_iou_scores = []
    
    print(f"\n开始 Fold {fold_idx} 的训练...")
    print(f"批次大小: {config.BATCH_SIZE}, 工作进程: {config.NUM_WORKERS}")
    print(f"图像大小: {config.IMAGE_SIZE}")
    print(f"类别数: {config.NUM_CLASSES}")
    print(f"CLAHE启用: {config.USE_CLAHE}")
    print(f"强数据增强: {config.USE_STRONG_AUGMENTATION}")
    print(f"真实患者分离: {config.USE_GROUP_CV}")
    
    # 训练循环
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nFold {fold_idx} - Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 50)
        
        # 训练
        train_loss, train_dice, train_iou = train_epoch_segmentation(
            model, train_loader, criterion, optimizer, scaler, device, config.ACCUMULATION_STEPS
        )
        
        # 验证
        val_loss, val_dice, val_iou, val_dice_per_class, val_iou_per_class = validate_epoch_segmentation(
            model, val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step()
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        val_iou_scores.append(val_iou)
        
        print(f"训练损失: {train_loss:.6f}, 训练Dice: {train_dice:.6f}, 训练IoU: {train_iou:.6f}")
        print(f"验证损失: {val_loss:.6f}, 验证Dice: {val_dice:.6f}, 验证IoU: {val_iou:.6f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.8f}")
        
        # 打印前5个类别的Dice分数
        print("各类别Dice分数 (前5个类别):")
        for i in range(min(5, len(val_dice_per_class))):
            class_name = list(CLASS_MAPPING.keys())[i] if i < len(CLASS_MAPPING) else f"类别 {i}"
            print(f"  {class_name}: {val_dice_per_class[i]:.4f}")
        
        # GPU利用率
        gpu_util = check_gpu_utilization()
        
        # 早停和模型保存
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            
            # 保存模型（包含fold信息）
            model_path = os.path.join(config.OUTPUT_DIR, f"{config.MODEL_NAME}_fold{fold_idx}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'fold': fold_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_dice_per_class': val_dice_per_class,
                'val_iou_per_class': val_iou_per_class,
                'config': config,
                'model_config': {
                    'model_type': config.MODEL_TYPE,
                    'num_classes': config.NUM_CLASSES,
                    'use_clahe': config.USE_CLAHE,
                    'use_strong_augmentation': config.USE_STRONG_AUGMENTATION,
                    'use_group_cv': config.USE_GROUP_CV,
                    'dice_weight': config.DICE_WEIGHT,
                    'ce_weight': config.BCE_WEIGHT
                }
            }, model_path)
            
            print(f"新的最佳模型已保存! Dice: {best_dice:.6f}")
        else:
            patience_counter += 1
            print(f"无改善. 耐心计数: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"早停触发于 epoch {epoch + 1}")
                break
        
        # 内存清理
        torch.cuda.empty_cache()
    
    # 记录当前fold的结果
    fold_result = {
        'fold': fold_idx,
        'best_dice': best_dice,
        'best_epoch': best_epoch,
        'final_val_loss': val_loss,
        'final_val_dice': val_dice,
        'final_val_iou': val_iou,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_dice_scores': val_dice_scores,
        'val_iou_scores': val_iou_scores,
        'val_dice_per_class': val_dice_per_class,
        'val_iou_per_class': val_iou_per_class
    }
    all_fold_results.append(fold_result)
    
    print(f"\nFold {fold_idx} 训练完成!")
    print(f"最佳Dice分数: {best_dice:.6f} (Epoch {best_epoch})")
    print(f"最终验证Dice: {val_dice:.6f}")
    print(f"最终验证IoU: {val_iou:.6f}")
    
    # 清理当前fold的模型和数据集
    del model, train_dataset, val_dataset, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*70}")
print("所有5个fold训练完成!")
print(f"{'='*70}")

# 计算和显示所有fold的统计结果
all_best_dices = [result['best_dice'] for result in all_fold_results]
all_final_dices = [result['final_val_dice'] for result in all_fold_results]
all_final_ious = [result['final_val_iou'] for result in all_fold_results]

print(f"\n=== 5折交叉验证结果汇总 ===")
print(f"最佳Dice分数:")
for i, dice in enumerate(all_best_dices):
    print(f"  Fold {i}: {dice:.6f}")

print(f"\n最终验证Dice分数:")
for i, dice in enumerate(all_final_dices):
    print(f"  Fold {i}: {dice:.6f}")

print(f"\n最终验证IoU分数:")
for i, iou in enumerate(all_final_ious):
    print(f"  Fold {i}: {iou:.6f}")

print(f"\n=== 统计摘要 ===")
print(f"最佳Dice分数 - 平均: {np.mean(all_best_dices):.6f}, 标准差: {np.std(all_best_dices):.6f}")
print(f"最终验证Dice - 平均: {np.mean(all_final_dices):.6f}, 标准差: {np.std(all_final_dices):.6f}")
print(f"最终验证IoU - 平均: {np.mean(all_final_ious):.6f}, 标准差: {np.std(all_final_ious):.6f}")

print(f"\n=== 模型文件保存位置 ===")
for i in range(config.NUM_FOLDS):
    model_path = os.path.join(config.OUTPUT_DIR, f"{config.MODEL_NAME}_fold{i}_best.pth")
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024*1024)
        print(f"  Fold {i}: {model_path} ({file_size:.1f} MB)")

print(f"\n所有5个fold的模型已保存，可用于集成推理!")


