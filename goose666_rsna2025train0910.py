# ======================
# 环境配置与库导入
# ======================
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

import timm  # 预训练模型库
import albumentations as A  # 图像增强库
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import pydicom  # DICOM医学图像处理

# 忽略警告信息
warnings.filterwarnings('ignore')

# ======================
# 随机种子设置（保证实验可复现）
# ======================
def set_seed(seed=42):
    """设置所有随机种子保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置cuDNN确定性模式（可能降低性能但保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法

set_seed(42)

# ======================
# 设备配置
# ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {device}")
if torch.cuda.is_available():
    # 打印GPU信息
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA版本: {torch.version.cuda}")
    torch.cuda.empty_cache()  # 清空GPU缓存
else:
    raise RuntimeError("需要GPU支持！本代码必须使用GPU运行")

# ======================
# 配置类（集中管理所有超参数）
# ======================
class Config:
    # 数据路径配置
    DATA_DIR = "/kaggle/input/srna2025pro-data"
    CVT_PNG_DIR = '/kaggle/input/srna2025pro-data/cvt_png'  # 转换后的PNG图像目录
    SERIES_MAPPING_PATH = '/kaggle/input/series-index-mapping/series_index_mapping.csv'  # 序列映射文件
    LOCALIZERS_PATH = '/kaggle/input/train-localizers-with-relative/train_localizers_with_relative.csv'  # 坐标标注文件
    TRAIN_CSV_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"  # 训练数据CSV
    
    # 模型参数（针对8帧处理优化）
    NUM_FRAMES = 8  # 每个样本使用的帧数
    IMAGE_SIZE = 224  # 输入图像尺寸
    NUM_CLASSES = 14  # 分类类别数（13个位置+1个总体）
    BATCH_SIZE = 6  # 批大小（针对8帧处理调整）
    NUM_EPOCHS = 2  # 训练轮数
    LEARNING_RATE = 5e-5  # 初始学习率
    
    # 模型配置开关
    MODEL_NAME_BACKBONE = "tf_efficientnetv2_s.in1k"  # 使用EfficientNetV2-S作为骨干网络
    USE_METADATA = True  # 是否使用患者元数据（年龄、性别）
    USE_WINDOWING = True  # 是否应用DICOM窗宽窗位
    USE_3CHANNEL_INPUT = True  # 是否使用3通道输入（多帧融合）
    USE_IMPROVED_LOSS = True  # 是否使用改进的损失函数
    USE_CLAHE = True  # 是否使用CLAHE对比度增强
    USE_STRONG_AUGMENTATION = True  # 是否使用强数据增强
    
    # GPU优化设置
    NUM_WORKERS = 2  # 数据加载工作线程数
    PIN_MEMORY = True  # 是否使用锁页内存（加速数据传输）
    PREFETCH_FACTOR = 2  # 数据预取因子
    PERSISTENT_WORKERS = True  # 是否保持工作进程
    
    # 训练参数（交叉验证相关）
    NUM_FOLDS = 5  # 交叉验证折数
    FOLD = 0  # 当前使用的折
    ACCUMULATION_STEPS = 5  # 梯度累积步数（模拟更大batch size）
    EARLY_STOPPING_PATIENCE = 3  # 早停耐心值
    USE_GROUP_CV = True  # 是否使用分组交叉验证（患者级别）
    
    # 数据加载优化
    CACHE_SIZE = 100  # 数据缓存大小
    
    # 输出配置
    OUTPUT_DIR = "/kaggle/working"  # 输出目录
    MODEL_NAME = "eightframe_efficientnetv2s"  # 模型名称

# 实例化配置
config = Config()

# 打印配置摘要
print("=== 配置参数摘要 ===")
print(f"模型骨干网络: {config.MODEL_NAME_BACKBONE}")
print(f"每样本帧数: {config.NUM_FRAMES}")
print(f"批大小: {config.BATCH_SIZE}")
print(f"梯度累积步数: {config.ACCUMULATION_STEPS}")
print(f"有效批大小: {config.BATCH_SIZE * config.ACCUMULATION_STEPS}")
print(f"CLAHE增强: {'启用' if config.USE_CLAHE else '禁用'}")
print(f"强数据增强: {'启用' if config.USE_STRONG_AUGMENTATION else '禁用'}")
print(f"分组交叉验证: {'启用' if config.USE_GROUP_CV else '禁用'}")


# ======================
# 数据加载
# ======================
print("\n加载数据...")
train_df = pd.read_csv(config.TRAIN_CSV_PATH)
series_mapping_df = pd.read_csv(config.SERIES_MAPPING_PATH)
localizers_df = pd.read_csv(config.LOCALIZERS_PATH)

print(f"训练数据形状: {train_df.shape}")
print(f"序列映射数据形状: {series_mapping_df.shape}")
print(f"定位数据形状: {localizers_df.shape}")


# ======================
# 目标列定义（14个分类目标）
# ======================
TARGET_COLS = [
    'Left Infraclinoid Internal Carotid Artery',  # 左侧床突下颈内动脉
    'Right Infraclinoid Internal Carotid Artery',  # 右侧床突下颈内动脉
    'Left Supraclinoid Internal Carotid Artery',  # 左侧床突上颈内动脉
    'Right Supraclinoid Internal Carotid Artery',  # 右侧床突上颈内动脉
    'Left Middle Cerebral Artery',  # 左侧大脑中动脉
    'Right Middle Cerebral Artery',  # 右侧大脑中动脉
    'Anterior Communicating Artery',  # 前交通动脉
    'Left Anterior Cerebral Artery',  # 左侧大脑前动脉
    'Right Anterior Cerebral Artery',  # 右侧大脑前动脉
    'Left Posterior Communicating Artery',  # 左侧后交通动脉
    'Right Posterior Communicating Artery',  # 右侧后交通动脉
    'Basilar Tip',  # 基底动脉尖
    'Other Posterior Circulation',  # 其他后循环
    'Aneurysm Present'  # 是否存在动脉瘤
]

print(f"\n目标列数量: {len(TARGET_COLS)}")


# ======================
# 医学图像处理工具函数
# ======================
def get_windowing_params(modality: str) -> Tuple[float, float]:
    """获取不同影像模态的窗宽窗位参数
    Args:
        modality: 影像模态（CT/CTA/MRA/MRI等）
    Returns:
        (window_center, window_width): 窗位和窗宽
    """
    # 预定义各模态的窗宽窗位
    windows = {
        'CT': (40, 80),      # 常规CT窗
        'CTA': (50, 350),    # CT血管造影（宽窗显示血管）
        'MRA': (600, 1200),  # MR血管造影
        'MRI': (40, 80),     # 常规MRI
        'MR': (40, 80)       # MRI别名
    }
    return windows.get(modality, (40, 80))  # 默认返回CT窗参数

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """应用DICOM窗宽窗位处理
    Args:
        img: 原始图像数据
        window_center: 窗位
        window_width: 窗宽
    Returns:
        处理后的8bit图像
    """
    # 计算窗范围
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    # 裁剪像素值到窗范围
    img = np.clip(img, img_min, img_max)
    
    # 归一化到0-1范围
    img = (img - img_min) / (img_max - img_min + 1e-7)  # 添加小量避免除零
    
    # 转换为8bit
    return (img * 255).astype(np.uint8)

def apply_clahe_normalization(img: np.ndarray, modality: str) -> np.ndarray:
    """应用CLAHE对比度限制自适应直方图均衡化
    Args:
        img: 输入图像
        modality: 影像模态（用于调整参数）
    Returns:
        处理后的图像
    """
    if not config.USE_CLAHE:
        return img
        
    # 不同模态使用不同参数
    if modality in ['CTA', 'MRA']:
        # 血管成像：更强的对比度增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = cv2.convertScaleAbs(img_clahe, alpha=1.1, beta=5)
    elif modality in ['MRI', 'MR']:
        # MRI：较温和的增强+gamma校正
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = np.power(img_clahe / 255.0, 0.9) * 255
        img_clahe = img_clahe.astype(np.uint8)
    else:
        # CT：标准CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
    
    return img_clahe

def robust_normalization(volume: np.ndarray) -> np.ndarray:
    """使用百分位数进行鲁棒归一化
    Args:
        volume: 3D体积数据
    Returns:
        归一化后的8bit体积数据
    """
    # 计算1%和99%百分位
    p1, p99 = np.percentile(volume.flatten(), [1, 99])
    volume_norm = np.clip(volume, p1, p99)  # 裁剪到百分位范围
    
    if p99 > p1:
        # 正常归一化
        volume_norm = (volume_norm - p1) / (p99 - p1 + 1e-7)
    else:
        # 异常情况处理
        volume_norm = np.zeros_like(volume_norm)
        
    return (volume_norm * 255).astype(np.uint8)


# # ======================
# # 8帧处理核心函数
# # ======================
# def create_3channel_input_8frame(volume: np.ndarray) -> np.ndarray:
#     """从8帧体积数据创建3通道输入（优化动脉瘤检测）
#     Args:
#         volume: 8帧的3D体积数据
#     Returns:
#         3通道的2D图像（解剖参考+MIP+纹理）
#     """
#     if len(volume) == 0:
#         return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
    
#     # 通道1：中间切片（提供解剖结构参考）
#     middle_slice = volume[len(volume) // 2]
    
#     # 通道2：最大密度投影（突出血管结构）
#     mip = np.max(volume, axis=0)
    
#     # 通道3：标准差投影（捕捉纹理信息）
#     std_proj = np.std(volume, axis=0).astype(np.float32)
    
#     # 对标准差投影进行鲁棒归一化
#     if std_proj.max() > std_proj.min():
#         p1, p99 = np.percentile(std_proj, [5, 95])
#         std_proj = np.clip(std_proj, p1, p99)
#         std_proj = ((std_proj - p1) / (p99 - p1 + 1e-7) * 255).astype(np.uint8)
#     else:
#         std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    
#     # 堆叠为3通道图像
#     return np.stack([middle_slice, mip, std_proj], axis=-1)

# def smart_8_frame_sampling(volume_paths: List[str], series_uid: str = None) -> List[str]:
#     """智能8帧采样策略（均匀覆盖整个体积）
#     Args:
#         volume_paths: 所有帧路径列表
#         series_uid: 序列ID（用于调试）
#     Returns:
#         选中的8个帧路径
#     """
#     n = len(volume_paths)
    
#     # 如果帧数≤8，使用所有帧并重复填充
#     if n <= 8:
#         result = volume_paths[:]
#         while len(result) < 8:
#             result.extend(volume_paths[:8-len(result)])
#         return result[:8]
    
#     # 从体积的10%位置开始采样（避免开头空切片）
#     start_idx = max(0, int(n * 0.1))
    
#     # 计算步长以均匀覆盖整个体积
#     available_frames = n - start_idx
#     step = max(1, available_frames // 8)
    
#     # 生成采样索引
#     indices = []
#     current_idx = start_idx
#     while len(indices) < 8 and current_idx < n:
#         indices.append(current_idx)
#         current_idx += step
    
#     # 不足8帧时从剩余帧补充
#     while len(indices) < 8:
#         remaining = [i for i in range(n) if i not in indices]
#         if remaining:
#             indices.append(remaining[len(indices) % len(remaining)])
#         else:
#             indices.append(indices[-1])  # 复制最后一帧
    
#     return [volume_paths[i] for i in indices[:8]]


# # ======================
# # 患者信息提取（用于分组交叉验证）
# # ======================
# def extract_dicom_patient_info(series_uid: str) -> Tuple[str, str]:
#     """从DICOM元数据提取患者信息
#     Args:
#         series_uid: 序列ID
#     Returns:
#         (study_uid, patient_id): 研究ID和患者ID
#     """
#     try:
#         dicom_dir = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_uid}"
#         if os.path.exists(dicom_dir):
#             # 查找DICOM文件
#             dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
#             if dcm_files:
#                 # 读取第一个DICOM文件的元数据
#                 ds = pydicom.dcmread(
#                     os.path.join(dicom_dir, dcm_files[0]), 
#                     stop_before_pixels=True,  # 不加载像素数据
#                     force=True  # 强制读取可能有问题的文件
#                 )
#                 study_uid = getattr(ds, 'StudyInstanceUID', None)
#                 patient_id = getattr(ds, 'PatientID', None)
#                 return study_uid or f"fallback_{series_uid[:32]}", patient_id
#     except Exception:
#         pass
    
#     # 回退方案：使用序列ID前缀
#     return f"fallback_{series_uid[:32]}", f"fallback_{series_uid[:32]}"

# @functools.lru_cache(maxsize=5000)
# def get_patient_group_cached(series_uid: str) -> str:
#     """带缓存的获取患者分组信息
#     Args:
#         series_uid: 序列ID
#     Returns:
#         患者分组标识（优先使用StudyInstanceUID）
#     """
#     study_uid, patient_id = extract_dicom_patient_info(series_uid)
#     return study_uid if study_uid and not study_uid.startswith('fallback_') else patient_id


# ======================
# 8帧数据集类
# ======================
class EightFrameDataset(Dataset):
    """8帧优化的数据集类"""
    def __init__(self, df, frame_paths_dict, series_mapping_df, num_frames=8, 
                 transform=None, is_training=True):
        """
        Args:
            df: 包含样本信息的DataFrame
            frame_paths_dict: 序列到帧路径的映射字典
            series_mapping_df: 序列元数据DataFrame
            num_frames: 每样本帧数（固定为8）
            transform: 数据增强变换
            is_training: 是否训练模式
        """
        self.df = df.reset_index(drop=True)
        self.frame_paths_dict = frame_paths_dict
        self.series_mapping_df = series_mapping_df
        self.num_frames = num_frames
        self.transform = transform
        self.is_training = is_training
        
        # 简单LRU缓存提高性能
        self._cache = {}
        self._cache_keys = []
        self._max_cache_size = config.CACHE_SIZE
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 先检查缓存
        if idx in self._cache:
            return self._cache[idx]
        
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        
        # 获取标签（14个目标列）
        labels = torch.tensor(row[TARGET_COLS].values.astype(np.float32))
        
        # 提取元数据（年龄、性别）
        metadata = self._extract_metadata(row)
        
        # 加载8帧3通道图像
        image = self._load_8frame_3channel_image(series_uid, row)
        
        result = (image, labels, metadata)
        
        # 更新缓存
        self._update_cache(idx, result)
        
        return result
    
    def _update_cache(self, idx, data):
        """更新LRU缓存"""
        if len(self._cache) >= self._max_cache_size:
            # 移除最旧的条目
            oldest_idx = self._cache_keys.pop(0)
            del self._cache[oldest_idx]
        
        self._cache[idx] = data
        self._cache_keys.append(idx)
    
    def _extract_metadata(self, row) -> torch.Tensor:
        """提取并标准化元数据（年龄、性别）"""
        if not config.USE_METADATA:
            return torch.tensor([0.0, 0.0], dtype=torch.float32)
        
        # 年龄处理（兼容不同格式）
        age = row.get('PatientAge', 50)
        if pd.isna(age):
            age = 50
        elif isinstance(age, str):
            age = int(''.join(filter(str.isdigit, age[:3])) or '50')
        age = min(float(age), 100.0) / 100.0  # 归一化到0-1
        
        # 性别处理（M=1, F=0）
        sex = row.get('PatientSex', 'M')
        sex = 1.0 if sex == 'M' else 0.0
        
        return torch.tensor([age, sex], dtype=torch.float32)
    
    def _load_8frame_3channel_image(self, series_uid: str, row) -> torch.Tensor:
        """加载8帧3通道图像（核心处理逻辑）"""
        paths = self.frame_paths_dict.get(series_uid, [])
        
        try:
            if len(paths) == 0 or paths[0].startswith('dummy_path'):
                # 从DICOM原始数据加载
                volume = self._load_volume_from_dicom_8frame(series_uid, row)
            else:
                # 从预处理PNG加载
                volume = self._load_volume_from_png_8frame(paths)
            
            # 应用鲁棒归一化
            volume = robust_normalization(volume)
            
            # 创建3通道输入
            image = create_3channel_input_8frame(volume)
            
            # 应用数据增强
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image
            
        except Exception as e:
            print(f"加载错误 {series_uid}: {e}")
            # 返回空白图像
            dummy_image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
            if self.transform:
                transformed = self.transform(image=dummy_image)
                return transformed['image']
            return torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    def _load_volume_from_png_8frame(self, paths: List[str]) -> np.ndarray:
        """从PNG加载8帧体积数据"""
        volume = []
        
        # 确保正好8个路径
        if len(paths) != 8:
            paths = smart_8_frame_sampling(paths)
        
        for path in paths:
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE), 
                                   interpolation=cv2.INTER_AREA)
                    volume.append(img)
            except:
                volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
        
        return np.array(volume) if volume else np.zeros((8, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
    
    def _load_volume_from_dicom_8frame(self, series_uid: str, row) -> np.ndarray:
        """从DICOM原始数据加载8帧体积"""
        series_data = self.series_mapping_df[
            self.series_mapping_df['SeriesInstanceUID'] == series_uid
        ].sort_values('relative_index')
        
        if len(series_data) == 0:
            return np.zeros((8, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
        
        volume = []
        modality = row.get('Modality', 'CT')
        
        # 智能采样8帧
        if len(series_data) <= 8:
            sampled_data = series_data
        else:
            all_indices = list(range(len(series_data)))
            sampled_indices = smart_8_frame_sampling([str(i) for i in all_indices])
            sampled_indices = [int(i) for i in sampled_indices]
            sampled_data = series_data.iloc[sampled_indices]
        
        # 逐帧处理
        for _, dicom_row in sampled_data.iterrows():
            try:
                ds = pydicom.dcmread(dicom_row['dicom_filename'])
                img = ds.pixel_array.astype(np.float32)
                
                # 处理多帧/彩色图像
                if img.ndim == 3:
                    if img.shape[-1] == 3:
                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                    else:
                        img = img[:, :, 0]
                
                # 应用DICOM rescale参数
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * ds.RescaleSlope + ds.RescaleIntercept
                
                # 应用窗宽窗位
                if config.USE_WINDOWING:
                    window_center, window_width = get_windowing_params(modality)
                    img = apply_dicom_windowing(img, window_center, window_width)
                else:
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
                
                # 应用CLAHE增强
                img = apply_clahe_normalization(img, modality)
                
                # 高质量resize
                img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE), 
                               interpolation=cv2.INTER_AREA)
                volume.append(img)
                
            except Exception as e:
                volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
                continue
        
        # 确保正好8帧
        while len(volume) < 8:
            if volume:
                volume.append(volume[-1])  # 复制最后一帧
            else:
                volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
        
        return np.array(volume[:8])


# ======================
# 数据准备流程
# ======================
def create_frame_paths_8frame():
    """创建8帧优化的路径映射"""
    frame_paths = {}
    
    print("从series_index_mapping.csv创建8帧优化路径...")
    
    for series_uid in tqdm(train_df['SeriesInstanceUID'].unique(), desc="处理序列"):
        # 获取序列数据
        series_data = series_mapping_df[series_mapping_df['SeriesInstanceUID'] == series_uid]
        
        if len(series_data) == 0:
            frame_paths[series_uid] = []
            continue
            
        # 获取训练数据行
        train_row = train_df[train_df['SeriesInstanceUID'] == series_uid].iloc[0]
        
        # 查找疾病位置对应的PNG路径
        found_paths = []
        for target_col in TARGET_COLS[:-1]:  # 排除"Aneurysm Present"
            if train_row[target_col] == 1:
                location_clean = target_col.replace('/', '_')
                series_dir = os.path.join(config.CVT_PNG_DIR, location_clean, series_uid)
                
                if os.path.exists(series_dir):
                    png_files = sorted(glob.glob(os.path.join(series_dir, "*.png")))
                    if png_files:
                        found_paths = png_files
                        break
        
        # 如果没有找到PNG，使用DICOM路径
        if not found_paths:
            dicom_dir = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_uid}"
            if os.path.exists(dicom_dir):
                num_frames = len(series_data)
                found_paths = [f"dummy_path_{i:04d}.png" for i in range(num_frames)]
        
        # 应用智能8帧采样
        if found_paths:
            found_paths = smart_8_frame_sampling(found_paths, series_uid)
        
        frame_paths[series_uid] = found_paths
    
    return frame_paths

# 创建8帧路径映射
frame_paths_dict = create_frame_paths_8frame()
print(f"已创建{len(frame_paths_dict)}个序列的8帧优化路径")

# 过滤无效序列
valid_series = [uid for uid, paths in frame_paths_dict.items() if len(paths) > 0]
train_df_filtered = train_df[train_df['SeriesInstanceUID'].isin(valid_series)].copy()
print(f"过滤后训练数据形状: {train_df_filtered.shape}")

# 检查动脉瘤分布
aneurysm_dist_filtered = train_df_filtered['Aneurysm Present'].value_counts()
print(f"动脉瘤存在分布: {aneurysm_dist_filtered.to_dict()}")


# ======================
# 稳健的交叉验证拆分
# ======================
def create_robust_cv_split(train_df, n_splits=5):
    """创建患者级别的交叉验证拆分"""
    print("\n创建患者级别的交叉验证拆分...")
    print("从DICOM元数据提取真实患者分组...")
    
    # 提取患者分组信息
    patient_groups = []
    for series_uid in tqdm(train_df['SeriesInstanceUID'], desc="读取DICOM患者信息"):
        patient_group = get_patient_group_cached(series_uid)
        patient_groups.append(patient_group)
    
    # 添加患者分组到DataFrame
    train_df = train_df.copy()
    train_df['patient_id'] = patient_groups
    
    n_groups = train_df['patient_id'].nunique()
    print(f"找到的真实患者组数: {n_groups}")
    
    # 检查是否有足够的分组
    if n_groups < n_splits:
        print(f"患者组数({n_groups})不足{n_splits}-fold CV需求")
        print("回退到StratifiedKFold...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(skf.split(train_df, train_df['Aneurysm Present']))
    
    # 创建分层键（模态+动脉瘤存在）
    train_df['stratify_key'] = (
        train_df['Modality'].astype(str) + '_' + 
        train_df['Aneurysm Present'].astype(str)
    )
    
    print(f"分层键: {train_df['stratify_key'].unique()}")
    
    # 使用GroupKFold确保患者级别分离
    group_kfold = GroupKFold(n_splits=n_splits)
    
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(group_kfold.split(
        train_df, 
        groups=train_df['patient_id']
    )):
        # 验证患者分离
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        # 检查患者重叠（应为0）
        train_patients = set(train_fold['patient_id'])
        val_patients = set(val_fold['patient_id'])
        overlap = train_patients.intersection(val_patients)
        
        # 计算分布
        train_dist = train_fold['Aneurysm Present'].value_counts(normalize=True)
        val_dist = val_fold['Aneurysm Present'].value_counts(normalize=True)
        
        print(f"Fold {fold_idx}:")
        print(f"  训练集: {len(train_fold)}样本 ({len(train_patients)}患者)")
        print(f"  验证集: {len(val_fold)}样本 ({len(val_patients)}患者)")
        print(f"  患者重叠: {len(overlap)} (应为0!)")
        print(f"  动脉瘤存在 - 训练: {train_dist.get(1, 0):.3f}, 验证: {val_dist.get(1, 0):.3f}")
        
        if len(overlap) > 0:
            print(f"  警告: 发现{len(overlap)}个重叠患者!")
        
        splits.append((train_idx, val_idx))
    
    return splits

# 创建交叉验证拆分
cv_splits = create_robust_cv_split(train_df_filtered, config.NUM_FOLDS)
train_indices, val_indices = cv_splits[config.FOLD]

# 获取当前fold的数据
train_fold_df = train_df_filtered.iloc[train_indices]
val_fold_df = train_df_filtered.iloc[val_indices]

print(f"\n稳健CV Fold {config.FOLD}摘要:")
print(f"训练集大小: {len(train_fold_df)}")
print(f"验证集大小: {len(val_fold_df)}")

# 检查分布
print(f"训练集动脉瘤分布: {train_fold_df['Aneurysm Present'].value_counts().to_dict()}")
print(f"验证集动脉瘤分布: {val_fold_df['Aneurysm Present'].value_counts().to_dict()}")

# 检查模态分布
print(f"训练集模态分布: {train_fold_df['Modality'].value_counts().to_dict()}")
print(f"验证集模态分布: {val_fold_df['Modality'].value_counts().to_dict()}")


# ======================
# 数据增强配置
# ======================
if config.USE_STRONG_AUGMENTATION:
    print("\n使用强数据增强提升泛化能力...")
    train_transform = A.Compose([
        # 几何变换（医学图像安全的）
        A.Rotate(limit=15, p=0.7),  # 旋转
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.6),  # 平移缩放旋转
        
        # 高级几何变换（提升鲁棒性）
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),  # 弹性变换
        A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.3),  # 网格畸变
        
        # 图像质量变化（模拟不同扫描仪/协议）
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),  # 亮度对比度
        A.CLAHE(clipLimit=2.0, tileGridSize=(8,8), p=0.5),  # CLAHE
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),  # Gamma校正
        
        # 噪声模拟（扫描仪差异）
        A.GaussNoise(var_limit=(10, 80), p=0.4),  # 高斯噪声
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  # ISO噪声
        A.Blur(blur_limit=3, p=0.2),  # 模糊
        
        # 医学图像特定增强
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),  # 随机遮挡
        
        # 归一化+Tensor转换
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet统计
        ToTensorV2()  # 转为Tensor
    ])
else:
    print("使用标准数据增强...")
    train_transform = A.Compose([
        A.Rotate(limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# 验证集变换（无增强）
val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# ======================
# 创建数据集和数据加载器
# ======================
import time
print("\n创建8帧数据集和加载器...")
train_dataset = EightFrameDataset(
    train_fold_df, 
    frame_paths_dict, 
    series_mapping_df,
    num_frames=config.NUM_FRAMES,
    transform=train_transform,
    is_training=True
)

val_dataset = EightFrameDataset(
    val_fold_df,
    frame_paths_dict,
    series_mapping_df,
    num_frames=config.NUM_FRAMES, 
    transform=val_transform,
    is_training=False
)

# 创建优化的数据加载器
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

print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(val_loader)}")

# 测试数据加载速度
print("\n测试8帧数据加载速度...")
start_time = time.time()
for i, batch in enumerate(train_loader):
    if i >= 5:  # 测试前5个批次
        break
    images, labels, metadata = batch
    print(f"批次 {i+1}: 图像形状: {images.shape}, 设备: {images.device}")

elapsed = time.time() - start_time
print(f"加载5个批次耗时: {elapsed:.2f}秒 (平均{elapsed/5:.2f}秒/批次)")


# ======================
# 模型定义
# ======================
class ImprovedMultiFrameModel(nn.Module):
    """改进的多帧模型（EfficientNetV2骨干+元数据集成）"""
    def __init__(self, num_frames=8, num_classes=14, pretrained=True):
        super(ImprovedMultiFrameModel, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.use_3channel = config.USE_3CHANNEL_INPUT
        self.use_metadata = config.USE_METADATA
        
        # 骨干网络：EfficientNetV2-S
        print(f"\n加载骨干网络: {config.MODEL_NAME_BACKBONE}")
        self.backbone = timm.create_model(
            config.MODEL_NAME_BACKBONE,
            pretrained=pretrained,
            num_classes=0,  # 不使用预训练的分类头
            global_pool='avg'  # 全局平均池化
        )
        
        self.feature_dim = self.backbone.num_features
        print(f"骨干网络特征维度: {self.feature_dim}")
        
        # 元数据处理层
        if self.use_metadata:
            self.meta_fc = nn.Sequential(
                nn.Linear(2, 16),  # 输入：年龄、性别
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 32),
                nn.ReLU()
            )
            classifier_input_dim = self.feature_dim + 32
        else:
            classifier_input_dim = self.feature_dim
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, meta=None):
        # 输入x形状: (batch_size, 3, 224, 224)
        features = self.backbone(x)  # 提取特征
        
        # 元数据集成
        if self.use_metadata and meta is not None:
            meta_features = self.meta_fc(meta)
            features = torch.cat([features, meta_features], dim=1)
        
        # 分类
        output = self.classifier(features)
        return output

# 初始化8帧模型
print("初始化8帧模型...")
model = ImprovedMultiFrameModel(
    num_frames=config.NUM_FRAMES,
    num_classes=config.NUM_CLASSES,
    pretrained=True
)
model = model.to(device)

# 参数统计
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")
print(f"模型设备: {next(model.parameters()).device}")


# ======================
# 损失函数定义
# ======================
class FocalLoss(nn.Module):
    """Focal Loss（解决类别不平衡问题）"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class WeightedMultiLabelLoss(nn.Module):
    """加权多标签损失"""
    def __init__(self, aneurysm_weight=3.0):
        super(WeightedMultiLabelLoss, self).__init__()
        self.weights = torch.ones(config.NUM_CLASSES, device=device)
        self.weights[-1] = aneurysm_weight  # 对"Aneurysm Present"赋予更高权重
        
    def forward(self, outputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        weighted_loss = bce_loss * self.weights
        return weighted_loss.mean()

class ImprovedLoss(nn.Module):
    """改进的复合损失函数（加权BCE+Focal Loss）"""
    def __init__(self, aneurysm_weight=3.0, focal_weight=0.3):
        super(ImprovedLoss, self).__init__()
        self.aneurysm_weight = aneurysm_weight
        self.focal_weight = focal_weight
        
        self.weights = torch.ones(config.NUM_CLASSES, device=device)
        self.weights[-1] = aneurysm_weight
        
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
    def forward(self, outputs, targets):
        # 加权BCE
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        weighted_bce = (bce_loss * self.weights).mean()
        
        # Focal Loss
        focal_loss = self.focal_loss(outputs, targets)
        
        # 组合损失
        return (1 - self.focal_weight) * weighted_bce + self.focal_weight * focal_loss

def get_loss_function():
    """根据配置获取损失函数"""
    if config.USE_IMPROVED_LOSS:
        return ImprovedLoss(aneurysm_weight=3.0, focal_weight=0.3)
    else:
        return WeightedMultiLabelLoss(aneurysm_weight=3.0)


# ======================
# 评估指标计算
# ======================
def calculate_competition_metric(y_true, y_pred):
    """计算比赛指标：加权多标签AUC ROC"""
    individual_aucs = []
    
    # 计算前13个类别的AUC
    for i in range(13):
        try:
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            else:
                auc = 0.5  # 单一类别默认0.5
            individual_aucs.append(auc)
        except:
            individual_aucs.append(0.5)  # 出错时默认0.5
    
    # 计算"Aneurysm Present"的AUC
    try:
        if len(np.unique(y_true[:, 13])) > 1:
            aneurysm_present_auc = roc_auc_score(y_true[:, 13], y_pred[:, 13])
        else:
            aneurysm_present_auc = 0.5
    except:
        aneurysm_present_auc = 0.5
    
    # 最终得分（动脉瘤AUC和平均个体AUC的平均）
    avg_individual = np.mean(individual_aucs)
    final_score = (aneurysm_present_auc + avg_individual) / 2
    
    return final_score, aneurysm_present_auc, avg_individual, individual_aucs


# ======================
# 训练配置
# ======================
# 获取损失函数
criterion = get_loss_function()
# AdamW优化器（带权重衰减）
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
# 余弦退火学习率调度
scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

print("\n训练配置完成")
print(f"使用的损失函数: {type(criterion).__name__}")


# ======================
# 训练和验证函数
# ======================
def train_epoch_optimized(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps):
    """优化的训练epoch（支持梯度累积）"""
    model.train()
    running_loss = 0.0
    
    optimizer.zero_grad()
    
    # 使用tqdm显示进度条
    for batch_idx, (images, targets, metadata) in enumerate(tqdm(train_loader, desc="训练")):
        # 异步传输到GPU
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        
        # 混合精度训练
        with torch.cuda.amp.autocast():
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps  # 梯度累积
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 累积到指定步数后更新
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
    
    # 处理剩余梯度
    if len(train_loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return running_loss / len(train_loader)

def validate_epoch_optimized(model, val_loader, criterion, device):
    """优化的验证epoch"""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, metadata in tqdm(val_loader, desc="验证"):
            # 异步传输到GPU
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
                
            with torch.cuda.amp.autocast():
                logits = model(images, metadata)
                loss = criterion(logits, targets)
            
            # 应用sigmoid获取概率
            outputs = torch.sigmoid(logits)
            
            # 收集结果
            running_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 合并所有批次结果
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    # 计算评估指标
    final_score, aneurysm_auc, avg_individual, individual_aucs = calculate_competition_metric(
        all_targets, all_outputs
    )
    
    return running_loss / len(val_loader), final_score, aneurysm_auc, avg_individual

def check_gpu_utilization():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU内存 - 已分配: {allocated:.2f}GB, 保留: {reserved:.2f}GB, 总量: {max_memory:.2f}GB")
        utilization = (allocated/max_memory)*100
        print(f"GPU利用率: {utilization:.1f}%")
        return utilization
    return 0

print("\n8帧处理的初始GPU状态:")
check_gpu_utilization()


# # ======================
# # 训练主循环
# # ======================
# # 初始化训练跟踪变量
# best_score = 0.0
# best_epoch = 0
# patience_counter = 0
# train_losses = []
# val_losses = []
# val_scores = []

# print("\n开始8帧训练（患者级别分离）...")
# print(f"批大小: {config.BATCH_SIZE}, 工作线程: {config.NUM_WORKERS}")
# print(f"每样本帧数: {config.NUM_FRAMES}")
# print(f"CLAHE增强: {'启用' if config.USE_CLAHE else '禁用'}")
# print(f"强数据增强: {'启用' if config.USE_STRONG_AUGMENTATION else '禁用'}")
# print(f"真实患者分离: {'启用' if config.USE_GROUP_CV else '禁用'}")

# for epoch in range(config.NUM_EPOCHS):
#     print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
#     print("-" * 50)
    
#     # 训练阶段
#     train_loss = train_epoch_optimized(
#         model, train_loader, criterion, optimizer, scaler, device, config.ACCUMULATION_STEPS
#     )
    
#     # 验证阶段
#     val_loss, val_score, aneurysm_auc, avg_individual = validate_epoch_optimized(
#         model, val_loader, criterion, device
#     )
    
#     # 学习率调整
#     scheduler.step()
    
#     # 记录指标
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
#     val_scores.append(val_score)
    
#     print(f"训练损失: {train_loss:.6f}")
#     print(f"验证损失: {val_loss:.6f}")
#     print(f"验证分数: {val_score:.6f}")
#     print(f"动脉瘤AUC: {aneurysm_auc:.6f}")
#     print(f"平均个体AUC: {avg_individual:.6f}")
#     print(f"学习率: {optimizer.param_groups[0]['lr']:.8f}")
    
#     # GPU利用率
#     gpu_util = check_gpu_utilization()
    
#     # 早停和模型保存
#     if val_score > best_score:
#         best_score = val_score
#         best_epoch = epoch + 1
#         patience_counter = 0
        
#         # 保存最佳模型
#         model_path = os.path.join(config.OUTPUT_DIR, f"{config.MODEL_NAME}_best.pth")
#         torch.save({
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'best_score': best_score,
#             'val_loss': val_loss,
#             'aneurysm_auc': aneurysm_auc,
#             'avg_individual_auc': avg_individual,
#             'config': config,
#             'model_config': {
#                 'backbone': config.MODEL_NAME_BACKBONE,
#                 'num_frames': config.NUM_FRAMES,
#                 'use_3channel': config.USE_3CHANNEL_INPUT,
#                 'use_metadata': config.USE_METADATA,
#                 'use_windowing': config.USE_WINDOWING,
#                 'use_improved_loss': config.USE_IMPROVED_LOSS,
#                 'use_clahe': config.USE_CLAHE,
#                 'use_strong_augmentation': config.USE_STRONG_AUGMENTATION,
#                 'use_group_cv': config.USE_GROUP_CV
#             }
#         }, model_path)
        
#         print(f"保存新最佳模型! 分数: {best_score:.6f}")
#     else:
#         patience_counter += 1
#         print(f"无提升. 耐心计数: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
        
#         if patience_counter >= config.EARLY_STOPPING_PATIENCE:
#             print(f"Epoch {epoch + 1}触发早停")
#             break
    
#     # 内存清理
#     torch.cuda.empty_cache()


# ======================
# 训练结果可视化
# ======================
print("\n" + "="*70)
print("8帧训练完成（患者级别分离）")
print("="*70)
print(f"最佳分数: {best_score:.6f} (Epoch {best_epoch})")

# 创建训练曲线图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 损失曲线
axes[0].plot(range(1, len(train_losses)+1), train_losses, 'b-', label='训练损失', linewidth=2)
axes[0].plot(range(1, len(val_losses)+1), val_losses, 'r-', label='验证损失', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('8帧训练: 损失曲线')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 验证分数
axes[1].plot(range(1, len(val_scores)+1), val_scores, 'g-', label='验证分数', linewidth=2)
axes[1].axhline(y=best_score, color='r', linestyle='--', alpha=0.7, 
                label=f'最佳: {best_score:.6f}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('比赛分数')
axes[1].set_title('8帧训练: 比赛分数')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 学习率曲线
lr_values = []
temp_optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
temp_scheduler = CosineAnnealingLR(temp_optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
for _ in range(config.NUM_EPOCHS):
    lr_values.append(temp_optimizer.param_groups[0]['lr'])
    temp_scheduler.step()

axes[2].plot(range(1, len(lr_values)+1), lr_values, 'purple', linewidth=2, label='学习率')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('学习率')
axes[2].set_title('学习率调度')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_yscale('log')  # 对数坐标

plt.tight_layout()
plt.show()


# ======================
# 最终模型摘要
# ======================
model_path = os.path.join(config.OUTPUT_DIR, f"{config.MODEL_NAME}_best.pth")
print(os.path.exists(model_path))
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print("\n" + "="*60)
    print("8帧模型摘要（患者级别分离）")
    print("="*60)
    print(f"最佳Epoch: {checkpoint['epoch']}")
    print(f"最佳分数: {checkpoint['best_score']:.6f}")
    print(f"动脉瘤AUC: {checkpoint['aneurysm_auc']:.6f}")
    print(f"平均个体AUC: {checkpoint['avg_individual_auc']:.6f}")
    print(f"模型大小: {os.path.getsize(model_path) / (1024 * 1024):.1f} MB")
    print(f"- CLAHE对比度增强: {'是' if config.USE_CLAHE else '否'}")
    print(f"- 强数据增强: {'是' if config.USE_STRONG_AUGMENTATION else '否'}")

print("="*60)
print("训练完成（患者级别交叉验证）!")
print("="*60)

# 最终清理
torch.cuda.empty_cache()
gc.collect()

