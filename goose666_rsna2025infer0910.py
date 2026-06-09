class Config:
    NUM_FRAMES = 8
    IMAGE_SIZE = 224
    NUM_CLASSES = 14
    BATCH_SIZE = 6  
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5
    
    MODEL_NAME_BACKBONE = "tf_efficientnetv2_s.in1k"
    USE_METADATA = True
    USE_WINDOWING = True
    USE_3CHANNEL_INPUT = True
    USE_IMPROVED_LOSS = True
    USE_CLAHE = True
    USE_STRONG_AUGMENTATION = True
    
config = Config()



# RSNA颅内动脉瘤检测推理代码
# 导入必要的库
import os
import sys
import gc
import json
import shutil
import warnings
warnings.filterwarnings('ignore')  # 忽略警告
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import functools

# 数据处理相关库
import numpy as np
import polars as pl
import pandas as pd

# 医学影像处理库
import pydicom
import cv2
from pydicom.pixel_data_handlers.util import convert_color_space

# 深度学习相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm  # 预训练模型库

# 图像增强库
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Kaggle竞赛API
import kaggle_evaluation.rsna_inference_server

# 设置计算设备（优先使用GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# 竞赛常量定义
ID_COL = 'SeriesInstanceUID'  # 序列ID列名
LABEL_COLS = [  # 所有标签列
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
    'Aneurysm Present',  # 是否存在动脉瘤
]

class InferenceConfig:
    """推理配置类，包含所有模型和数据处理参数"""
    # 模型架构配置 (必须与训练时完全一致)
    MODEL_NAME_BACKBONE = "tf_efficientnetv2_s.in1k"  # 使用的backbone模型
    NUM_FRAMES = 8                                    # 处理的帧数
    IMAGE_SIZE = 224                                  # 输入图像尺寸
    NUM_CLASSES = 14                                  # 分类类别数

    # 输入处理配置
    USE_3CHANNEL_INPUT = True     # 使用3通道输入(中帧+MIP+标准差投影)
    USE_WINDOWING = True          # 使用DICOM窗宽窗位
    USE_CLAHE = True              # 使用CLAHE增强
    USE_METADATA = True           # 使用元数据(年龄/性别)
    
    # 模型路径配置
    MODEL_PATH = "/kaggle/input/0911epoch20/eightframe_efficientnetv2s_best (6).pth"
    
    # 推理优化配置
    BATCH_SIZE = 1                # 推理批大小(通常为1)
    USE_AMP = True                # 使用混合精度推理
    
    # 调试模式
    DEBUG_MODE = False            # 调试日志开关

# 实例化配置对象
CFG = InferenceConfig()

# 打印配置摘要
print("=== 推理配置摘要 ===")
print(f"模型架构: {CFG.MODEL_NAME_BACKBONE}")
print(f"输入帧数: {CFG.NUM_FRAMES}")
print(f"图像尺寸: {CFG.IMAGE_SIZE}x{CFG.IMAGE_SIZE}")
print(f"窗宽窗位: {'启用' if CFG.USE_WINDOWING else '禁用'}")
print(f"CLAHE增强: {'启用' if CFG.USE_CLAHE else '禁用'}")
print(f"3通道输入: {'是' if CFG.USE_3CHANNEL_INPUT else '否'}")
print(f"元数据使用: {'是' if CFG.USE_METADATA else '否'}")


# DICOM处理工具函数
def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """应用DICOM窗宽窗位调整以增强图像对比度"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def get_windowing_params(modality: str) -> Tuple[float, float]:
    """获取不同模态的推荐窗宽窗位值"""
    windows = {
        'CT': (40, 80),        # CT常用窗宽窗位
        'CTA': (50, 350),      # CT血管造影
        'MRA': (600, 1200),    # MR血管造影
        'MRI': (40, 80),       # 常规MRI
        'MR': (40, 80)         # 常规MR
    }
    return windows.get(modality, (50, 350))  # 默认使用CTA参数

def apply_clahe_normalization(img: np.ndarray, modality: str) -> np.ndarray:
    """应用CLAHE增强，针对不同模态使用不同参数"""
    if not CFG.USE_CLAHE:
        return img
        
    # 血管成像使用更强的对比度增强
    if modality in ['CTA', 'MRA']:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = cv2.convertScaleAbs(img_clahe, alpha=1.1, beta=5)
    # MRI使用更温和的增强
    elif modality in ['MRI', 'MR']:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        img_clahe = np.power(img_clahe / 255.0, 0.9) * 255  # Gamma校正
        img_clahe = img_clahe.astype(np.uint8)
    # CT使用标准CLAHE
    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
    
    return img_clahe

def robust_normalization(volume: np.ndarray) -> np.ndarray:
    """使用百分位数进行鲁棒的归一化"""
    p1, p99 = np.percentile(volume.flatten(), [1, 99])  # 获取1%和99%百分位数
    volume_norm = np.clip(volume, p1, p99)  # 截断到百分位范围内
    
    if p99 > p1:
        volume_norm = (volume_norm - p1) / (p99 - p1 + 1e-7)  # 归一化到[0,1]
    else:
        volume_norm = np.zeros_like(volume_norm)
        
    return (volume_norm * 255).astype(np.uint8)  # 转换为8位图像

def extract_sort_key(path: str) -> Tuple[float, float, str]:
    """从DICOM文件中提取排序键值"""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        instance_number = getattr(ds, 'InstanceNumber', None)
        position = getattr(ds, 'ImagePositionPatient', [None, None, None])
        z = position[2] if position and len(position) == 3 else None

        if instance_number is not None:
            return (int(instance_number), 0, path)
        elif z is not None:
            return (float('inf'), float(z), path)
        else:
            return (float('inf'), float('inf'), path)
    except:
        return (float('inf'), float('inf'), path)

def sort_dicom_paths(dcm_paths: List[str]) -> List[str]:
    """根据医学元数据对DICOM文件路径进行排序"""
    if not dcm_paths:
        return []
    
    # 提取排序键值并排序
    sort_info = [extract_sort_key(path) for path in dcm_paths]
    sort_info.sort()
    return [x[2] for x in sort_info]

def create_3channel_input_8frame(volume: np.ndarray) -> np.ndarray:
    """从8帧数据创建3通道输入图像"""
    if len(volume) == 0:
        return np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)
    
    # 中间切片（最重要的解剖参考）
    middle_slice = volume[len(volume) // 2]
    
    # 最大密度投影（MIP）- 对血管结构优化
    mip = np.max(volume, axis=0)
    
    # 标准差投影用于纹理分析
    std_proj = np.std(volume, axis=0).astype(np.float32)
    
    # 使用鲁棒方法归一化标准差投影
    if std_proj.max() > std_proj.min():
        p1, p99 = np.percentile(std_proj, [5, 95])
        std_proj = np.clip(std_proj, p1, p99)
        std_proj = ((std_proj - p1) / (p99 - p1 + 1e-7) * 255).astype(np.uint8)
    else:
        std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    
    # 堆叠三个通道
    return np.stack([middle_slice, mip, std_proj], axis=-1)

def smart_8_frame_sampling(volume_paths: List[str], series_uid: str = None) -> List[str]:
    """智能8帧采样策略"""
    n = len(volume_paths)
    
    # 如果帧数少于8，使用所有可用帧并重复填充
    if n <= 8:
        result = volume_paths[:]
        while len(result) < 8:
            result.extend(volume_paths[:8-len(result)])
        return result[:8]
    
    # 从体积的10%处开始采样，避免开头空切片
    start_idx = max(0, int(n * 0.1))
    available_frames = n - start_idx
    step = max(1, available_frames // 8)  # 计算步长
    
    # 采样帧索引
    indices = []
    current_idx = start_idx
    while len(indices) < 8 and current_idx < n:
        indices.append(current_idx)
        current_idx += step
    
    # 如果需要更多帧，从剩余帧中填充
    while len(indices) < 8:
        remaining = [i for i in range(n) if i not in indices]
        if remaining:
            indices.append(remaining[len(indices) % len(remaining)])
        else:
            indices.append(indices[-1])  # 复制最后一帧
    
    return [volume_paths[i] for i in indices[:8]]

print("DICOM处理函数准备就绪")


class ImprovedMultiFrameModel(nn.Module):
    """改进的多帧模型，使用EfficientNetV2-S和元数据集成"""
    def __init__(self, num_frames=8, num_classes=14, pretrained=True):
        super(ImprovedMultiFrameModel, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.use_3channel = CFG.USE_3CHANNEL_INPUT
        self.use_metadata = CFG.USE_METADATA
        
        # 骨干网络: EfficientNetV2-S
        print(f"加载骨干网络: {CFG.MODEL_NAME_BACKBONE}")
        self.backbone = timm.create_model(
            CFG.MODEL_NAME_BACKBONE,
            pretrained=pretrained,
            num_classes=0,  # 不包含分类头
            global_pool='avg'  # 全局平均池化
        )
        
        self.feature_dim = self.backbone.num_features
        print(f"骨干网络 {CFG.MODEL_NAME_BACKBONE}: {self.feature_dim} 特征维度")
        
        # 元数据处理
        if self.use_metadata:
            self.meta_fc = nn.Sequential(
                nn.Linear(2, 16),  # 输入年龄和性别两个特征
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
            nn.Linear(256, num_classes)  # 输出14个类别的概率
        )
        
    def forward(self, x, meta=None):
        # 3通道输入处理
        features = self.backbone(x)  # (batch_size, feature_dim)
        
        # 元数据集成
        if self.use_metadata and meta is not None:
            meta_features = self.meta_fc(meta)
            features = torch.cat([features, meta_features], dim=1)
        
        # 分类
        output = self.classifier(features)
        return output

print("模型架构定义完成")


def process_single_dicom(dicom_path: str, modality: str = 'CTA') -> Optional[np.ndarray]:
    """处理单个DICOM文件并返回处理后的图像"""
    try:
        # 读取DICOM文件
        dicom = pydicom.dcmread(dicom_path, force=True)
        
        # 检查像素数据是否存在
        if 'PixelData' not in dicom:
            if CFG.DEBUG_MODE:
                print(f"警告: {dicom_path} 中没有像素数据")
            return None
            
        # 提取像素数组
        img = dicom.pixel_array
        
        # 检查图像是否有效
        if img is None or img.size == 0:
            if CFG.DEBUG_MODE:
                print(f"警告: {dicom_path} 中像素数组为空")
            return None
            
        # 处理光度解释
        interp = getattr(dicom, 'PhotometricInterpretation', 'MONOCHROME2')
        
        # 处理YBR颜色空间转换
        if interp == "YBR_FULL":
            try:
                img = convert_color_space(img, 'YBR_FULL', 'RGB')
            except:
                pass
        
        # 如果是多通道图像，转换为灰度
        if img.ndim == 3:
            if interp in ["RGB", "YBR_FULL"]:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            elif img.shape[2] == 1:
                img = img[:, :, 0]
            else:
                img = img[:, :, 0]  # 取第一个通道
        
        # 确保是2D图像
        if img.ndim != 2:
            return None
            
        # 应用rescale斜率/截距
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            img = img * dicom.RescaleSlope + dicom.RescaleIntercept
        
        # 应用窗宽窗位
        if CFG.USE_WINDOWING:
            window_center, window_width = get_windowing_params(modality)
            img = apply_dicom_windowing(img, window_center, window_width)
        else:
            # 不使用窗宽窗位的归一化
            img = img.astype(np.float32)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        
        # 处理MONOCHROME1（反转灰度）
        if interp == "MONOCHROME1":
            img = 255 - img
            
        # 应用CLAHE增强
        img = apply_clahe_normalization(img, modality)
            
        # 调整大小前验证
        if img.shape[0] == 0 or img.shape[1] == 0:
            return None
            
        # 调整到目标大小
        img = cv2.resize(img, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
        
        return img
        
    except Exception as e:
        if CFG.DEBUG_MODE:
            print(f"处理 {dicom_path} 时出错: {e}")
        return None

def extract_metadata_from_dicom(dicom_path: str) -> Tuple[float, float]:
    """从DICOM元数据中提取年龄和性别"""
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        
        # 年龄处理
        age = getattr(ds, 'PatientAge', '050Y')
        if isinstance(age, str):
            age = int(''.join(filter(str.isdigit, age[:3])) or '50')
        age = min(float(age), 100.0) / 100.0  # 归一化到0-1
        
        # 性别处理
        sex = getattr(ds, 'PatientSex', 'M')
        sex = 1.0 if sex == 'M' else 0.0  # 男性1.0，女性0.0
        
        return age, sex
    except:
        return 0.5, 0.0  # 默认值

def process_dicom_series(series_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """处理DICOM序列并返回多帧张量和元数据"""
    series_path = Path(series_path)
    
    # 查找所有DICOM文件
    dicom_files = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    if not dicom_files:
        if CFG.DEBUG_MODE:
            print(f"警告: {series_path} 中没有找到DICOM文件")
        return create_dummy_tensor(), torch.tensor([0.5, 0.0], dtype=torch.float32)
    
    # 按医学元数据排序文件
    sorted_files = sort_dicom_paths(dicom_files)
    
    # 从第一个文件获取模态和元数据
    try:
        first_dicom = pydicom.dcmread(sorted_files[0], stop_before_pixels=True)
        modality = getattr(first_dicom, 'Modality', 'CTA')
        age, sex = extract_metadata_from_dicom(sorted_files[0])
    except:
        modality = 'CTA'
        age, sex = 0.5, 0.0
    
    # 处理每个DICOM文件
    processed_images = []
    for dicom_path in sorted_files:
        img = process_single_dicom(dicom_path, modality)
        if img is not None:
            processed_images.append(img)
    
    if not processed_images:
        if CFG.DEBUG_MODE:
            print(f"警告: {series_path} 中没有成功处理的图像")
        return create_dummy_tensor(), torch.tensor([age, sex], dtype=torch.float32)
    
    # 采样帧数以匹配目标数量
    sampled_images = smart_8_frame_sampling(processed_images, str(series_path))
    
    # 应用鲁棒归一化
    volume = np.array(sampled_images)
    volume = robust_normalization(volume)
    
    # 创建3通道输入
    image = create_3channel_input_8frame(volume)
    
    # 应用归一化（与训练时匹配）
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    try:
        transformed = transform(image=image)
        image_tensor = transformed['image']
    except:
        # 转换失败时创建虚拟张量
        dummy_img = np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)
        transformed = transform(image=dummy_img)
        image_tensor = transformed['image']
    
    # 创建元数据张量
    metadata_tensor = torch.tensor([age, sex], dtype=torch.float32)
    
    return image_tensor, metadata_tensor

def create_dummy_tensor() -> torch.Tensor:
    """处理失败时创建虚拟张量"""
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    dummy_img = np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)
    transformed = transform(image=dummy_img)
    return transformed['image']

print("DICOM序列处理准备就绪")


# 全局变量
MODEL = None

def load_model() -> nn.Module:
    """加载训练好的模型"""
    print(f"从 {CFG.MODEL_PATH} 加载模型")
    
    if not os.path.exists(CFG.MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {CFG.MODEL_PATH}")
    
    # 初始化模型
    model = ImprovedMultiFrameModel(
        num_frames=CFG.NUM_FRAMES,
        num_classes=CFG.NUM_CLASSES,
        pretrained=False  # 加载训练好的权重
    )
    
    try:
        # 尝试使用weights_only=True加载（最安全）
        checkpoint = torch.load(CFG.MODEL_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint)
        print("成功加载模型权重 (weights_only=True)")
        
    except Exception as e1:
        print(f"weights_only=True加载失败: {e1}")
        try:
            # 回退：加载完整检查点
            checkpoint = torch.load(CFG.MODEL_PATH, map_location='cpu', weights_only=False)
            
            # 加载权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'best_score' in checkpoint:
                    print(f"加载模型，最佳分数: {checkpoint['best_score']:.6f}")
                if 'epoch' in checkpoint:
                    print(f"最佳epoch: {checkpoint['epoch']}")
            else:
                model.load_state_dict(checkpoint)
            print("使用完整检查点加载模型")
            
        except Exception as e2:
            print(f"完整检查点加载失败: {e2}")
            # 最后尝试：仅提取state_dict
            try:
                checkpoint = torch.load(CFG.MODEL_PATH, map_location='cpu', weights_only=False)
                # 仅提取模型权重
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                model.load_state_dict(state_dict)
                print("使用提取的state_dict加载模型")
                
            except Exception as e3:
                raise RuntimeError(f"所有加载方法都失败: {e1}, {e2}, {e3}")
    
    # 移动到设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    return model

def initialize_model():
    """初始化模型并进行预热"""
    global MODEL
    
    if MODEL is None:
        MODEL = load_model()
        
        # 预热模型
        print("预热模型...")
        dummy_input = torch.randn(1, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE).to(device)
        dummy_meta = torch.tensor([[0.5, 0.0]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            with autocast(enabled=CFG.USE_AMP):
                _ = MODEL(dummy_input, dummy_meta)
        
        print("模型已准备好进行推理!")

print("模型加载函数准备就绪")


def create_fallback_predictions() -> np.ndarray:
    """创建保守的备用预测"""
    # 基于训练数据分布的保守预测
    fallback_values = np.array([
        0.05, 0.05, 0.08, 0.08,  # 颈内动脉
        0.12, 0.12,              # 大脑中动脉  
        0.15,                    # 前交通动脉
        0.06, 0.06,              # 大脑前动脉
        0.07, 0.07,              # 后交通动脉
        0.09,                    # 基底动脉尖
        0.11,                    # 其他后循环
        0.43                     # 存在动脉瘤（训练分布）
    ])
    return fallback_values

def predict_series(series_path: str) -> np.ndarray:
    """对单个序列进行预测"""
    global MODEL
    
    # 如果需要，初始化模型
    if MODEL is None:
        initialize_model()
    
    try:
        # 处理DICOM序列
        image_tensor, metadata_tensor = process_dicom_series(series_path)
        
        # 添加批次维度并移动到设备
        image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
        metadata_tensor = metadata_tensor.unsqueeze(0).to(device)  # (1, 2)
        
        # 进行预测
        with torch.no_grad():
            with autocast(enabled=CFG.USE_AMP):
                logits = MODEL(image_tensor, metadata_tensor)
                probabilities = torch.sigmoid(logits)
        
        # 转换为numpy
        predictions = probabilities.cpu().numpy()[0]
        
        # 验证预测结果
        predictions = np.clip(predictions, 0.0, 1.0)
        predictions = np.nan_to_num(predictions, nan=0.1)
        
        return predictions
        
    except Exception as e:
        if CFG.DEBUG_MODE:
            print(f"预测时出错: {e}")
        return create_fallback_predictions()

def _predict_inner(series_path: str) -> pl.DataFrame:
    """内部预测逻辑"""
    # 提取序列ID用于日志记录
    series_id = os.path.basename(series_path)
    
    if CFG.DEBUG_MODE:
        print(f"处理序列: {series_id}")
    
    # 进行预测
    predictions = predict_series(series_path)
    
    # 创建输出数据框（API要求不包含SeriesInstanceUID列）
    predictions_df = pl.DataFrame(
        data=[predictions.tolist()],
        schema=LABEL_COLS,
        orient='row'
    )
    
    if CFG.DEBUG_MODE:
        print(f"预测范围: {predictions.min():.6f} - {predictions.max():.6f}")
        print(f"存在动脉瘤: {predictions[-1]:.6f}")
    
    return predictions_df

print("预测函数准备就绪")

def predict(series_path: str) -> pl.DataFrame:
    """
    Kaggle API的主预测函数。
    推理服务器会为每个测试序列调用此函数。
    """
    try:
        # 调用内部预测逻辑
        return _predict_inner(series_path)
        
    except Exception as e:
        print(f"预测 {os.path.basename(series_path)} 时出错: {e}")
        print("使用备用预测。")
        
        # 返回备用预测
        fallback_preds = create_fallback_predictions()
        predictions_df = pl.DataFrame(
            data=[fallback_preds.tolist()],
            schema=LABEL_COLS,
            orient='row'
        )
        
        return predictions_df
        
    finally:
        # 必要的清理以防止磁盘空间问题
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        # 内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

print("主API函数准备就绪")


# 主执行函数
def main():
    """主执行函数"""
    print("="*70)
    print("RSNA颅内动脉瘤检测 - 推理")
    print("="*70)
    print(f"设备: {device}")
    print(f"模型: ImprovedMultiFrameModel (EfficientNetV2-S)")
    print(f"帧数: {CFG.NUM_FRAMES}")
    print(f"图像尺寸: {CFG.IMAGE_SIZE}")
    print(f"使用窗宽窗位: {CFG.USE_WINDOWING}")
    print(f"使用3通道输入: {CFG.USE_3CHANNEL_INPUT}")
    print(f"使用CLAHE: {CFG.USE_CLAHE}")
    print("-" * 70)
    
    try:
        # 预加载模型
        initialize_model()
        
        # 初始化推理服务器
        print("初始化推理服务器...")
        inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
        
        # 运行推理
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("运行在竞赛模式...")
            inference_server.serve()
        else:
            print("运行在本地网关模式...")
            inference_server.run_local_gateway()
            
            # 如果可用，显示结果
            submission_path = '/kaggle/working/submission.parquet'
            if os.path.exists(submission_path):
                try:
                    submission_df = pl.read_parquet(submission_path)
                    print(f"\n提交预览:")
                    print(f"形状: {submission_df.shape}")
                    print(submission_df.head())
                except Exception as e:
                    print(f"无法读取提交文件: {e}")
        
        print("\n" + "="*70)
        print("推理成功完成!")
        print("="*70)
        
    except Exception as e:
        print(f"严重错误: {e}")
        print("这可能表明模型加载或API配置问题。")
        raise e

# 运行主执行
if __name__ == "__main__":
    main()


# # RSNA颅内动脉瘤检测推理代码
# # 导入必要的库
# import os
# import sys
# import gc
# import json
# import shutil
# import warnings
# warnings.filterwarnings('ignore')  # 忽略警告
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# import functools

# # 数据处理相关库
# import numpy as np
# import polars as pl
# import pandas as pd

# # 医学影像处理库
# import pydicom
# import cv2
# from pydicom.pixel_data_handlers.util import convert_color_space

# # 深度学习相关库
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.cuda.amp import autocast
# import timm  # 预训练模型库

# # 图像增强库
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# # Kaggle竞赛API
# import kaggle_evaluation.rsna_inference_server

# # 设置计算设备（优先使用GPU）
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"使用设备: {device}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
#     print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# # 竞赛常量定义
# ID_COL = 'SeriesInstanceUID'  # 序列ID列名
# LABEL_COLS = [  # 所有标签列
#     'Left Infraclinoid Internal Carotid Artery',
#     'Right Infraclinoid Internal Carotid Artery',
#     'Left Supraclinoid Internal Carotid Artery',
#     'Right Supraclinoid Internal Carotid Artery',
#     'Left Middle Cerebral Artery',
#     'Right Middle Cerebral Artery',
#     'Anterior Communicating Artery',
#     'Left Anterior Cerebral Artery',
#     'Right Anterior Cerebral Artery',
#     'Left Posterior Communicating Artery',
#     'Right Posterior Communicating Artery',
#     'Basilar Tip',
#     'Other Posterior Circulation',
#     'Aneurysm Present',  # 是否存在动脉瘤
# ]

# class InferenceConfig:
#     """推理配置类，包含所有模型和数据处理参数"""
#     # 模型架构配置 (必须与训练时完全一致)
#     MODEL_NAME_BACKBONE = "tf_efficientnetv2_s.in1k"  # 使用的backbone模型
#     NUM_FRAMES = 8                                    # 处理的帧数
#     IMAGE_SIZE = 224                                  # 输入图像尺寸
#     NUM_CLASSES = 14                                  # 分类类别数

#     # 输入处理配置
#     USE_3CHANNEL_INPUT = True     # 使用3通道输入(中帧+MIP+标准差投影)
#     USE_WINDOWING = True          # 使用DICOM窗宽窗位
#     USE_CLAHE = True              # 使用CLAHE增强
#     USE_METADATA = True           # 使用元数据(年龄/性别)
    
#     # TTA配置
#     USE_TTA = True                # 启用测试时增强
#     TTA_FLIPS = ['none', 'horizontal']  # 水平翻转增强
#     TTA_ROTATIONS = [0, 90, 180, 270]  # 旋转增强
#     TTA_BRIGHTNESS = [0.9, 1.0, 1.1]    # 亮度变化
    
#     # 模型路径配置
#     MODEL_PATH = "/kaggle/input/changshi2epoch/eightframe_efficientnetv2s_best (1).pth"
    
#     # 推理优化配置
#     BATCH_SIZE = 1                # 推理批大小(通常为1)
#     USE_AMP = True                # 使用混合精度推理
    
#     # 调试模式
#     DEBUG_MODE = False            # 调试日志开关

# # 实例化配置对象
# CFG = InferenceConfig()

# # 打印配置摘要
# print("=== 推理配置摘要 ===")
# print(f"模型架构: {CFG.MODEL_NAME_BACKBONE}")
# print(f"输入帧数: {CFG.NUM_FRAMES}")
# print(f"图像尺寸: {CFG.IMAGE_SIZE}x{CFG.IMAGE_SIZE}")
# print(f"窗宽窗位: {'启用' if CFG.USE_WINDOWING else '禁用'}")
# print(f"CLAHE增强: {'启用' if CFG.USE_CLAHE else '禁用'}")
# print(f"3通道输入: {'是' if CFG.USE_3CHANNEL_INPUT else '否'}")
# print(f"元数据使用: {'是' if CFG.USE_METADATA else '否'}")
# print(f"TTA增强: {'启用' if CFG.USE_TTA else '禁用'}")

# # DICOM处理工具函数
# def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
#     """应用DICOM窗宽窗位调整以增强图像对比度"""
#     img_min = window_center - window_width // 2
#     img_max = window_center + window_width // 2
#     img = np.clip(img, img_min, img_max)
#     img = (img - img_min) / (img_max - img_min + 1e-7)
#     return (img * 255).astype(np.uint8)

# def get_windowing_params(modality: str) -> Tuple[float, float]:
#     """获取不同模态的推荐窗宽窗位值"""
#     windows = {
#         'CT': (40, 80),        # CT常用窗宽窗位
#         'CTA': (50, 350),      # CT血管造影
#         'MRA': (600, 1200),    # MR血管造影
#         'MRI': (40, 80),       # 常规MRI
#         'MR': (40, 80)         # 常规MR
#     }
#     return windows.get(modality, (50, 350))  # 默认使用CTA参数

# def apply_clahe_normalization(img: np.ndarray, modality: str) -> np.ndarray:
#     """应用CLAHE增强，针对不同模态使用不同参数"""
#     if not CFG.USE_CLAHE:
#         return img
        
#     # 血管成像使用更强的对比度增强
#     if modality in ['CTA', 'MRA']:
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         img_clahe = clahe.apply(img.astype(np.uint8))
#         img_clahe = cv2.convertScaleAbs(img_clahe, alpha=1.1, beta=5)
#     # MRI使用更温和的增强
#     elif modality in ['MRI', 'MR']:
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         img_clahe = clahe.apply(img.astype(np.uint8))
#         img_clahe = np.power(img_clahe / 255.0, 0.9) * 255  # Gamma校正
#         img_clahe = img_clahe.astype(np.uint8)
#     # CT使用标准CLAHE
#     else:
#         clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#         img_clahe = clahe.apply(img.astype(np.uint8))
    
#     return img_clahe

# def robust_normalization(volume: np.ndarray) -> np.ndarray:
#     """使用百分位数进行鲁棒的归一化"""
#     p1, p99 = np.percentile(volume.flatten(), [1, 99])  # 获取1%和99%百分位数
#     volume_norm = np.clip(volume, p1, p99)  # 截断到百分位范围内
    
#     if p99 > p1:
#         volume_norm = (volume_norm - p1) / (p99 - p1 + 1e-7)  # 归一化到[0,1]
#     else:
#         volume_norm = np.zeros_like(volume_norm)
        
#     return (volume_norm * 255).astype(np.uint8)  # 转换为8位图像

# def extract_sort_key(path: str) -> Tuple[float, float, str]:
#     """从DICOM文件中提取排序键值"""
#     try:
#         ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
#         instance_number = getattr(ds, 'InstanceNumber', None)
#         position = getattr(ds, 'ImagePositionPatient', [None, None, None])
#         z = position[2] if position and len(position) == 3 else None

#         if instance_number is not None:
#             return (int(instance_number), 0, path)
#         elif z is not None:
#             return (float('inf'), float(z), path)
#         else:
#             return (float('inf'), float('inf'), path)
#     except:
#         return (float('inf'), float('inf'), path)

# def sort_dicom_paths(dcm_paths: List[str]) -> List[str]:
#     """根据医学元数据对DICOM文件路径进行排序"""
#     if not dcm_paths:
#         return []
    
#     # 提取排序键值并排序
#     sort_info = [extract_sort_key(path) for path in dcm_paths]
#     sort_info.sort()
#     return [x[2] for x in sort_info]

# def create_3channel_input_8frame(volume: np.ndarray) -> np.ndarray:
#     """从8帧数据创建3通道输入图像"""
#     if len(volume) == 0:
#         return np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)
    
#     # 中间切片（最重要的解剖参考）
#     middle_slice = volume[len(volume) // 2]
    
#     # 最大密度投影（MIP）- 对血管结构优化
#     mip = np.max(volume, axis=0)
    
#     # 标准差投影用于纹理分析
#     std_proj = np.std(volume, axis=0).astype(np.float32)
    
#     # 使用鲁棒方法归一化标准差投影
#     if std_proj.max() > std_proj.min():
#         p1, p99 = np.percentile(std_proj, [5, 95])
#         std_proj = np.clip(std_proj, p1, p99)
#         std_proj = ((std_proj - p1) / (p99 - p1 + 1e-7) * 255).astype(np.uint8)
#     else:
#         std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    
#     # 堆叠三个通道
#     return np.stack([middle_slice, mip, std_proj], axis=-1)

# def smart_8_frame_sampling(volume_paths: List[str], series_uid: str = None) -> List[str]:
#     """智能8帧采样策略"""
#     n = len(volume_paths)
    
#     # 如果帧数少于8，使用所有可用帧并重复填充
#     if n <= 8:
#         result = volume_paths[:]
#         while len(result) < 8:
#             result.extend(volume_paths[:8-len(result)])
#         return result[:8]
    
#     # 从体积的10%处开始采样，避免开头空切片
#     start_idx = max(0, int(n * 0.1))
#     available_frames = n - start_idx
#     step = max(1, available_frames // 8)  # 计算步长
    
#     # 采样帧索引
#     indices = []
#     current_idx = start_idx
#     while len(indices) < 8 and current_idx < n:
#         indices.append(current_idx)
#         current_idx += step
    
#     # 如果需要更多帧，从剩余帧中填充
#     while len(indices) < 8:
#         remaining = [i for i in range(n) if i not in indices]
#         if remaining:
#             indices.append(remaining[len(indices) % len(remaining)])
#         else:
#             indices.append(indices[-1])  # 复制最后一帧
    
#     return [volume_paths[i] for i in indices[:8]]

# print("DICOM处理函数准备就绪")

# class ImprovedMultiFrameModel(nn.Module):
#     """改进的多帧模型，使用EfficientNetV2-S和元数据集成"""
#     def __init__(self, num_frames=8, num_classes=14, pretrained=True):
#         super(ImprovedMultiFrameModel, self).__init__()
#         self.num_frames = num_frames
#         self.num_classes = num_classes
#         self.use_3channel = CFG.USE_3CHANNEL_INPUT
#         self.use_metadata = CFG.USE_METADATA
        
#         # 骨干网络: EfficientNetV2-S
#         print(f"加载骨干网络: {CFG.MODEL_NAME_BACKBONE}")
#         self.backbone = timm.create_model(
#             CFG.MODEL_NAME_BACKBONE,
#             pretrained=pretrained,
#             num_classes=0,  # 不包含分类头
#             global_pool='avg'  # 全局平均池化
#         )
        
#         self.feature_dim = self.backbone.num_features
#         print(f"骨干网络 {CFG.MODEL_NAME_BACKBONE}: {self.feature_dim} 特征维度")
        
#         # 元数据处理
#         if self.use_metadata:
#             self.meta_fc = nn.Sequential(
#                 nn.Linear(2, 16),  # 输入年龄和性别两个特征
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(16, 32),
#                 nn.ReLU()
#             )
#             classifier_input_dim = self.feature_dim + 32
#         else:
#             classifier_input_dim = self.feature_dim
        
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(classifier_input_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)  # 输出14个类别的概率
#         )
        
#     def forward(self, x, meta=None):
#         # 3通道输入处理
#         features = self.backbone(x)  # (batch_size, feature_dim)
        
#         # 元数据集成
#         if self.use_metadata and meta is not None:
#             meta_features = self.meta_fc(meta)
#             features = torch.cat([features, meta_features], dim=1)
        
#         # 分类
#         output = self.classifier(features)
#         return output

# print("模型架构定义完成")

# def process_single_dicom(dicom_path: str, modality: str = 'CTA') -> Optional[np.ndarray]:
#     """处理单个DICOM文件并返回处理后的图像"""
#     try:
#         # 读取DICOM文件
#         dicom = pydicom.dcmread(dicom_path, force=True)
        
#         # 检查像素数据是否存在
#         if 'PixelData' not in dicom:
#             if CFG.DEBUG_MODE:
#                 print(f"警告: {dicom_path} 中没有像素数据")
#             return None
            
#         # 提取像素数组
#         img = dicom.pixel_array
        
#         # 检查图像是否有效
#         if img is None or img.size == 0:
#             if CFG.DEBUG_MODE:
#                 print(f"警告: {dicom_path} 中像素数组为空")
#             return None
            
#         # 处理光度解释
#         interp = getattr(dicom, 'PhotometricInterpretation', 'MONOCHROME2')
        
#         # 处理YBR颜色空间转换
#         if interp == "YBR_FULL":
#             try:
#                 img = convert_color_space(img, 'YBR_FULL', 'RGB')
#             except:
#                 pass
        
#         # 如果是多通道图像，转换为灰度
#         if img.ndim == 3:
#             if interp in ["RGB", "YBR_FULL"]:
#                 img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
#             elif img.shape[2] == 1:
#                 img = img[:, :, 0]
#             else:
#                 img = img[:, :, 0]  # 取第一个通道
        
#         # 确保是2D图像
#         if img.ndim != 2:
#             return None
            
#         # 应用rescale斜率/截距
#         if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
#             img = img * dicom.RescaleSlope + dicom.RescaleIntercept
        
#         # 应用窗宽窗位
#         if CFG.USE_WINDOWING:
#             window_center, window_width = get_windowing_params(modality)
#             img = apply_dicom_windowing(img, window_center, window_width)
#         else:
#             # 不使用窗宽窗位的归一化
#             img = img.astype(np.float32)
#             img_min, img_max = img.min(), img.max()
#             if img_max > img_min:
#                 img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
#             else:
#                 img = np.zeros_like(img, dtype=np.uint8)
        
#         # 处理MONOCHROME1（反转灰度）
#         if interp == "MONOCHROME1":
#             img = 255 - img
            
#         # 应用CLAHE增强
#         img = apply_clahe_normalization(img, modality)
            
#         # 调整大小前验证
#         if img.shape[0] == 0 or img.shape[1] == 0:
#             return None
            
#         # 调整到目标大小
#         img = cv2.resize(img, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
        
#         return img
        
#     except Exception as e:
#         if CFG.DEBUG_MODE:
#             print(f"处理 {dicom_path} 时出错: {e}")
#         return None

# def extract_metadata_from_dicom(dicom_path: str) -> Tuple[float, float]:
#     """从DICOM元数据中提取年龄和性别"""
#     try:
#         ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        
#         # 年龄处理
#         age = getattr(ds, 'PatientAge', '050Y')
#         if isinstance(age, str):
#             age = int(''.join(filter(str.isdigit, age[:3])) or '50')
#         age = min(float(age), 100.0) / 100.0  # 归一化到0-1
        
#         # 性别处理
#         sex = getattr(ds, 'PatientSex', 'M')
#         sex = 1.0 if sex == 'M' else 0.0  # 男性1.0，女性0.0
        
#         return age, sex
#     except:
#         return 0.5, 0.0  # 默认值

# def process_dicom_series(series_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
#     """处理DICOM序列并返回多帧张量和元数据"""
#     series_path = Path(series_path)
    
#     # 查找所有DICOM文件
#     dicom_files = []
#     for root, _, files in os.walk(series_path):
#         for file in files:
#             if file.endswith('.dcm'):
#                 dicom_files.append(os.path.join(root, file))
    
#     if not dicom_files:
#         if CFG.DEBUG_MODE:
#             print(f"警告: {series_path} 中没有找到DICOM文件")
#         return create_dummy_tensor(), torch.tensor([0.5, 0.0], dtype=torch.float32)
    
#     # 按医学元数据排序文件
#     sorted_files = sort_dicom_paths(dicom_files)
    
#     # 从第一个文件获取模态和元数据
#     try:
#         first_dicom = pydicom.dcmread(sorted_files[0], stop_before_pixels=True)
#         modality = getattr(first_dicom, 'Modality', 'CTA')
#         age, sex = extract_metadata_from_dicom(sorted_files[0])
#     except:
#         modality = 'CTA'
#         age, sex = 0.5, 0.0
    
#     # 处理每个DICOM文件
#     processed_images = []
#     for dicom_path in sorted_files:
#         img = process_single_dicom(dicom_path, modality)
#         if img is not None:
#             processed_images.append(img)
    
#     if not processed_images:
#         if CFG.DEBUG_MODE:
#             print(f"警告: {series_path} 中没有成功处理的图像")
#         return create_dummy_tensor(), torch.tensor([age, sex], dtype=torch.float32)
    
#     # 采样帧数以匹配目标数量
#     sampled_images = smart_8_frame_sampling(processed_images, str(series_path))
    
#     # 应用鲁棒归一化
#     volume = np.array(sampled_images)
#     volume = robust_normalization(volume)
    
#     # 创建3通道输入
#     image = create_3channel_input_8frame(volume)
    
#     # 应用归一化（与训练时匹配）
#     transform = A.Compose([
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ])
    
#     try:
#         transformed = transform(image=image)
#         image_tensor = transformed['image']
#     except:
#         # 转换失败时创建虚拟张量
#         dummy_img = np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)
#         transformed = transform(image=dummy_img)
#         image_tensor = transformed['image']
    
#     # 创建元数据张量
#     metadata_tensor = torch.tensor([age, sex], dtype=torch.float32)
    
#     return image_tensor, metadata_tensor

# def create_dummy_tensor() -> torch.Tensor:
#     """处理失败时创建虚拟张量"""
#     transform = A.Compose([
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ])
    
#     dummy_img = np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)
#     transformed = transform(image=dummy_img)
#     return transformed['image']

# print("DICOM序列处理准备就绪")

# # 全局变量
# MODEL = None

# def load_model() -> nn.Module:
#     """加载训练好的模型"""
#     print(f"从 {CFG.MODEL_PATH} 加载模型")
    
#     if not os.path.exists(CFG.MODEL_PATH):
#         raise FileNotFoundError(f"找不到模型文件: {CFG.MODEL_PATH}")
    
#     # 初始化模型
#     model = ImprovedMultiFrameModel(
#         num_frames=CFG.NUM_FRAMES,
#         num_classes=CFG.NUM_CLASSES,
#         pretrained=False  # 加载训练好的权重
#     )
    
#     try:
#         # 尝试使用weights_only=True加载（最安全）
#         checkpoint = torch.load(CFG.MODEL_PATH, map_location='cpu', weights_only=True)
#         model.load_state_dict(checkpoint)
#         print("成功加载模型权重 (weights_only=True)")
        
#     except Exception as e1:
#         print(f"weights_only=True加载失败: {e1}")
#         try:
#             # 回退：加载完整检查点
#             checkpoint = torch.load(CFG.MODEL_PATH, map_location='cpu', weights_only=False)
            
#             # 加载权重
#             if 'model_state_dict' in checkpoint:
#                 model.load_state_dict(checkpoint['model_state_dict'])
#                 if 'best_score' in checkpoint:
#                     print(f"加载模型，最佳分数: {checkpoint['best_score']:.6f}")
#                 if 'epoch' in checkpoint:
#                     print(f"最佳epoch: {checkpoint['epoch']}")
#             else:
#                 model.load_state_dict(checkpoint)
#             print("使用完整检查点加载模型")
            
#         except Exception as e2:
#             print(f"完整检查点加载失败: {e2}")
#             # 最后尝试：仅提取state_dict
#             try:
#                 checkpoint = torch.load(CFG.MODEL_PATH, map_location='cpu', weights_only=False)
#                 # 仅提取模型权重
#                 if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#                     state_dict = checkpoint['model_state_dict']
#                 else:
#                     state_dict = checkpoint
                
#                 model.load_state_dict(state_dict)
#                 print("使用提取的state_dict加载模型")
                
#             except Exception as e3:
#                 raise RuntimeError(f"所有加载方法都失败: {e1}, {e2}, {e3}")
    
#     # 移动到设备并设置为评估模式
#     model = model.to(device)
#     model.eval()
    
#     return model

# def initialize_model():
#     """初始化模型并进行预热"""
#     global MODEL
    
#     if MODEL is None:
#         MODEL = load_model()
        
#         # 预热模型
#         print("预热模型...")
#         dummy_input = torch.randn(1, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE).to(device)
#         dummy_meta = torch.tensor([[0.5, 0.0]], dtype=torch.float32).to(device)
        
#         with torch.no_grad():
#             with autocast(enabled=CFG.USE_AMP):
#                 _ = MODEL(dummy_input, dummy_meta)
        
#         print("模型已准备好进行推理!")

# print("模型加载函数准备就绪")

# def create_fallback_predictions() -> np.ndarray:
#     """创建保守的备用预测"""
#     # 基于训练数据分布的保守预测
#     fallback_values = np.array([
#         0.05, 0.05, 0.08, 0.08,  # 颈内动脉
#         0.12, 0.12,              # 大脑中动脉  
#         0.15,                    # 前交通动脉
#         0.06, 0.06,              # 大脑前动脉
#         0.07, 0.07,              # 后交通动脉
#         0.09,                    # 基底动脉尖
#         0.11,                    # 其他后循环
#         0.43                     # 存在动脉瘤（训练分布）
#     ])
#     return fallback_values

# def apply_tta_transforms(image_tensor: torch.Tensor, tta_idx: int) -> torch.Tensor:
#     """应用测试时增强变换"""
#     # 将张量转换回numpy格式进行处理
#     image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
#     image_np = (image_np * 255).astype(np.uint8)
    
#     # 定义TTA变换
#     if tta_idx == 0:  # 原始图像
#         pass
#     elif tta_idx == 1:  # 水平翻转
#         image_np = cv2.flip(image_np, 1)
#     elif tta_idx == 2:  # 旋转90度
#         image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
#     elif tta_idx == 3:  # 旋转180度
#         image_np = cv2.rotate(image_np, cv2.ROTATE_180)
#     elif tta_idx == 4:  # 旋转270度
#         image_np = cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     elif tta_idx == 5:  # 亮度增强
#         image_np = cv2.convertScaleAbs(image_np, alpha=1.1, beta=5)
#     elif tta_idx == 6:  # 亮度减弱
#         image_np = cv2.convertScaleAbs(image_np, alpha=0.9, beta=-5)
    
#     # 重新归一化并转换为张量
#     transform = A.Compose([
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ])
#     transformed = transform(image=image_np)
#     return transformed['image'].to(device)

# def predict_series_with_tta(series_path: str) -> np.ndarray:
#     """使用TTA对单个序列进行预测"""
#     global MODEL
    
#     # 如果需要，初始化模型
#     if MODEL is None:
#         initialize_model()
    
#     try:
#         # 处理DICOM序列
#         image_tensor, metadata_tensor = process_dicom_series(series_path)
        
#         # 准备TTA预测
#         all_predictions = []
        
#         # 定义TTA组合
#         tta_combinations = []
#         if CFG.USE_TTA:
#             # 基础TTA: 原始 + 水平翻转
#             tta_combinations.extend(['none', 'horizontal'])
            
#             # 添加旋转TTA
#             if len(CFG.TTA_ROTATIONS) > 1:
#                 for angle in CFG.TTA_ROTATIONS[1:]:
#                     tta_combinations.append(f'rotate_{angle}')
            
#             # 添加亮度TTA
#             if len(CFG.TTA_BRIGHTNESS) > 1:
#                 for brightness in CFG.TTA_BRIGHTNESS[1:]:
#                     tta_combinations.append(f'brightness_{brightness}')
#         else:
#             tta_combinations = ['none']
        
#         # 对每个TTA变换进行预测
#         for tta_idx, tta_type in enumerate(tta_combinations):
#             # 应用TTA变换
#             if tta_type == 'none':
#                 tta_image = image_tensor.unsqueeze(0).to(device)
#             else:
#                 tta_image = apply_tta_transforms(image_tensor, tta_idx).unsqueeze(0).to(device)
            
#             metadata_tensor_tta = metadata_tensor.unsqueeze(0).to(device)
            
#             # 进行预测
#             with torch.no_grad():
#                 with autocast(enabled=CFG.USE_AMP):
#                     logits = MODEL(tta_image, metadata_tensor_tta)
#                     probabilities = torch.sigmoid(logits)
            
#             # 对TTA结果进行逆变换
#             if tta_type == 'horizontal':
#                 probabilities = probabilities[:, [1, 0, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13]]
#             elif tta_type.startswith('rotate_'):
#                 # 旋转预测可能需要重新排列某些类别
#                 pass  # 这里可以添加特定于旋转的类别重新排列
            
#             all_predictions.append(probabilities.cpu().numpy()[0])
        
#         # 平均所有TTA预测
#         predictions = np.mean(all_predictions, axis=0)
        
#         # 验证预测结果
#         predictions = np.clip(predictions, 0.0, 1.0)
#         predictions = np.nan_to_num(predictions, nan=0.1)
        
#         return predictions
        
#     except Exception as e:
#         if CFG.DEBUG_MODE:
#             print(f"预测时出错: {e}")
#         return create_fallback_predictions()

# def predict_series(series_path: str) -> np.ndarray:
#     """对单个序列进行预测"""
#     if CFG.USE_TTA:
#         return predict_series_with_tta(series_path)
#     else:
#         return predict_series_without_tta(series_path)

# def predict_series_without_tta(series_path: str) -> np.ndarray:
#     """不使用TTA对单个序列进行预测"""
#     global MODEL
    
#     # 如果需要，初始化模型
#     if MODEL is None:
#         initialize_model()
    
#     try:
#         # 处理DICOM序列
#         image_tensor, metadata_tensor = process_dicom_series(series_path)
        
#         # 添加批次维度并移动到设备
#         image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
#         metadata_tensor = metadata_tensor.unsqueeze(0).to(device)  # (1, 2)
        
#         # 进行预测
#         with torch.no_grad():
#             with autocast(enabled=CFG.USE_AMP):
#                 logits = MODEL(image_tensor, metadata_tensor)
#                 probabilities = torch.sigmoid(logits)
        
#         # 转换为numpy
#         predictions = probabilities.cpu().numpy()[0]
        
#         # 验证预测结果
#         predictions = np.clip(predictions, 0.0, 1.0)
#         predictions = np.nan_to_num(predictions, nan=0.1)
        
#         return predictions
        
#     except Exception as e:
#         if CFG.DEBUG_MODE:
#             print(f"预测时出错: {e}")
#         return create_fallback_predictions()

# def _predict_inner(series_path: str) -> pl.DataFrame:
#     """内部预测逻辑"""
#     # 提取序列ID用于日志记录
#     series_id = os.path.basename(series_path)
    
#     if CFG.DEBUG_MODE:
#         print(f"处理序列: {series_id}")
    
#     # 进行预测
#     predictions = predict_series(series_path)
    
#     # 创建输出数据框（API要求不包含SeriesInstanceUID列）
#     predictions_df = pl.DataFrame(
#         data=[predictions.tolist()],
#         schema=LABEL_COLS,
#         orient='row'
#     )
    
#     if CFG.DEBUG_MODE:
#         print(f"预测范围: {predictions.min():.6f} - {predictions.max():.6f}")
#         print(f"存在动脉瘤: {predictions[-1]:.6f}")
    
#     return predictions_df

# print("预测函数准备就绪")

# def predict(series_path: str) -> pl.DataFrame:
#     """
#     Kaggle API的主预测函数。
#     推理服务器会为每个测试序列调用此函数。
#     """
#     try:
#         # 调用内部预测逻辑
#         return _predict_inner(series_path)
        
#     except Exception as e:
#         print(f"预测 {os.path.basename(series_path)} 时出错: {e}")
#         print("使用备用预测。")
        
#         # 返回备用预测
#         fallback_preds = create_fallback_predictions()
#         predictions_df = pl.DataFrame(
#             data=[fallback_preds.tolist()],
#             schema=LABEL_COLS,
#             orient='row'
#         )
        
#         return predictions_df
        
#     finally:
#         # 必要的清理以防止磁盘空间问题
#         shared_dir = '/kaggle/shared'
#         shutil.rmtree(shared_dir, ignore_errors=True)
#         os.makedirs(shared_dir, exist_ok=True)
        
#         # 内存清理
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()

# print("主API函数准备就绪")

# # 主执行函数
# def main():
#     """主执行函数"""
#     print("="*70)
#     print("RSNA颅内动脉瘤检测 - 推理")
#     print("="*70)
#     print(f"设备: {device}")
#     print(f"模型: ImprovedMultiFrameModel (EfficientNetV2-S)")
#     print(f"帧数: {CFG.NUM_FRAMES}")
#     print(f"图像尺寸: {CFG.IMAGE_SIZE}")
#     print(f"使用窗宽窗位: {CFG.USE_WINDOWING}")
#     print(f"使用3通道输入: {CFG.USE_3CHANNEL_INPUT}")
#     print(f"使用CLAHE: {CFG.USE_CLAHE}")
#     print(f"使用TTA: {CFG.USE_TTA}")
#     if CFG.USE_TTA:
#         print(f"TTA翻转: {CFG.TTA_FLIPS}")
#         print(f"TTA旋转: {CFG.TTA_ROTATIONS}")
#         print(f"TTA亮度: {CFG.TTA_BRIGHTNESS}")
#     print("-" * 70)
    
#     try:
#         # 预加载模型
#         initialize_model()
        
#         # 初始化推理服务器
#         print("初始化推理服务器...")
#         inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
        
#         # 运行推理
#         if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#             print("运行在竞赛模式...")
#             inference_server.serve()
#         else:
#             print("运行在本地网关模式...")
#             inference_server.run_local_gateway()
            
#             # 如果可用，显示结果
#             submission_path = '/kaggle/working/submission.parquet'
#             if os.path.exists(submission_path):
#                 try:
#                     submission_df = pl.read_parquet(submission_path)
#                     print(f"\n提交预览:")
#                     print(f"形状: {submission_df.shape}")
#                     print(submission_df.head())
#                 except Exception as e:
#                     print(f"无法读取提交文件: {e}")
        
#         print("\n" + "="*70)
#         print("推理成功完成!")
#         print("="*70)
        
#     except Exception as e:
#         print(f"严重错误: {e}")
#         print("这可能表明模型加载或API配置问题。")
#         raise e

# # 运行主执行
# if __name__ == "__main__":
#     main()

