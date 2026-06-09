import os
import sys
import gc
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import multiprocessing as mp
from functools import partial

# 深度学习框架
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import timm

# 数据增强
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 交叉验证和评估
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ====================================================
# 配置参数 - 优化数据加载
# ====================================================
class CFG:
    # 数据路径
    train_dir = Path('/kaggle/input/rsna-intracranial-aneurysm-detection/series')
    train_csv = Path('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
    
    # 输出路径
    output_dir = Path('/kaggle/working')
    model_dir = output_dir / 'models'
    cache_dir = output_dir / 'cache'
    
    # 图像参数
    img_size = 256
    num_slices = 32
    slice_size = (img_size, img_size)
    in_channels = 32
    
    # 训练参数
    batch_size = 8
    val_batch_size = 16
    chunk_size = 300  # 分批加载大小
    num_workers = min(8, os.cpu_count() - 1)  # 增加工作进程
    epochs = 30
    lr = 1e-4
    weight_decay = 1e-5
    fold = 5
    selected_folds = [0, 1, 2, 3, 4]
    patience = 8
    min_lr = 1e-6
    T_0 = 10
    
    # 模型设置
    model_name = 'tf_efficientnetv2_m.in21k_ft_in1k'
    pretrained = True
    dropout = 0.5
    label_smoothing = 0.05
    
    # 优化设置
    use_amp = True
    gradient_checkpointing = False
    max_cache_size = 12 * 1024**3  # 增加缓存大小到12GB

# 创建目录
for dir_path in [CFG.model_dir, CFG.cache_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ====================================================
# 类别信息
# ====================================================
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
    'Aneurysm Present',
]
NUM_CLASSES = len(LABEL_COLS)

# ====================================================
# 数据加载与预处理 - 增加CTA窗口处理
# ====================================================

def apply_ct_window(img, window_center, window_width):
    """Apply CT windowing to a single image."""
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    img = np.clip(img, min_value, max_value)
    img = (img - min_value) / (max_value - min_value + 1e-5)
    return img

def preprocess_mri(img):
    """Apply MRI-specific preprocessing if needed."""
    # MRI 标准化
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
    return img

def sort_slices_by_position(dicom_files):
    """Sort DICOM files by z-position (ImagePositionPatient[2]) or InstanceNumber."""
    slice_positions = []
    for dcm_file in dicom_files:
        dcm = pydicom.dcmread(dcm_file)
        position = getattr(dcm, 'ImagePositionPatient', None)
        if position is not None and len(position) >= 3:
            z = float(position[2])
        else:
            z = float(getattr(dcm, 'InstanceNumber', 0))
        slice_positions.append((dcm_file, z))
    # Sort by z-position
    sorted_files = [f for f, _ in sorted(slice_positions, key=lambda x: x[1])]
    return sorted_files

def load_dicom_series(series_path, img_size=(128, 128), num_slices=64):
    dicom_files = sorted(list(series_path.glob("*.dcm")))
    slices = []
    # If only one file and it's multi-frame (3D)
    if len(dicom_files) == 1:
        dcm = pydicom.dcmread(dicom_files[0])
        if hasattr(dcm, "NumberOfFrames") and dcm.NumberOfFrames > 1:
            # Multi-frame DICOM (3D)
            frames = dcm.pixel_array.astype(np.float32)  # shape: (num_frames, H, W)
            for img in frames:
                img = cv2.resize(img, img_size)
                modality = getattr(dcm, 'Modality', 'CT')
                if modality == 'CT':
                    img = apply_ct_window(img, window_center=40, window_width=80)
                elif modality == 'MRI':
                    img = preprocess_mri(img)
                else:
                    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
                slices.append(img)
        else:
            # Single 2D image
            img = dcm.pixel_array.astype(np.float32)
            img = cv2.resize(img, img_size)
            modality = getattr(dcm, 'Modality', 'CT')
            if modality == 'CT':
                img = apply_ct_window(img, window_center=40, window_width=80)
            elif modality == 'MRI':
                img = preprocess_mri(img)
            else:
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
            slices.append(img)
    else:
        # Multiple 2D slices
        dicom_files = sort_slices_by_position(dicom_files)
        for dcm_file in dicom_files:
            dcm = pydicom.dcmread(dcm_file)
            img = dcm.pixel_array.astype(np.float32)
            img = cv2.resize(img, img_size)
            modality = getattr(dcm, 'Modality', 'CT')
            if modality == 'CT':
                img = apply_ct_window(img, window_center=40, window_width=80)
            elif modality == 'MRI':
                img = preprocess_mri(img)
            else:
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
            slices.append(img)
    # Pad or crop to num_slices
    if len(slices) < num_slices:
        pad = [np.zeros(img_size, dtype=np.float32)] * (num_slices - len(slices))
        slices = pad + slices
    elif len(slices) > num_slices:
        center = len(slices) // 2
        slices = slices[center - num_slices//2 : center + num_slices//2]
    volume = np.stack(slices, axis=0)  # shape: (num_slices, H, W)
    # Add channel dimension for 3D CNN: (C, D, H, W)
    volume = np.expand_dims(volume, axis=0)  # (1, D, H, W)
    return volume

# 预加载所有 DICOM 卷到内存缓存
def cache_volume_to_disk(series_id, volume, cache_dir):
    cache_path = cache_dir / f"{series_id}.npy"
    np.save(cache_path, volume)

def load_volume_from_disk(series_id, cache_dir):
    cache_path = cache_dir / f"{series_id}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    return None

def preload_dicom_volumes(df, series_dir, img_size=(128,128), num_slices=64, cache_dir=None):
    cache = {}
    for idx, row in df.iterrows():
        series_id = row[ID_COL]
        if cache_dir:
            volume = load_volume_from_disk(series_id, cache_dir)
            if volume is not None:
                cache[series_id] = volume
                continue
        series_path = series_dir / str(series_id)
        volume = load_dicom_series(series_path, img_size, num_slices)
        cache[series_id] = volume
        if cache_dir:
            cache_volume_to_disk(series_id, volume, cache_dir)
    return cache

#多线程加载
def preprocess_worker(args):
    series_id, series_path, img_size, num_slices = args
    volume = load_dicom_series(series_path, img_size, num_slices)
    return series_id, volume

def preload_dicom_volumes_parallel(df, series_dir, img_size=(128,128), num_slices=64, num_workers=4):
    args_list = [
        (row[ID_COL], series_dir / str(row[ID_COL]), img_size, num_slices)
        for _, row in df.iterrows()
    ]
    cache = {}
    with mp.Pool(num_workers) as pool:
        for series_id, volume in pool.imap(preprocess_worker, args_list):
            cache[series_id] = volume
    return cache

# =========================
# 数据增强 Transform 示例
# =========================
class Simple3DTransform:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5, normalize=True):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.normalize = normalize

    def __call__(self, volume):
        # volume shape: (1, D, H, W)
        vol = volume.copy()

        # Random flip along depth axis
        if np.random.rand() < self.flip_prob:
            vol = np.flip(vol, axis=1)  # flip D

        # Random flip along height axis
        if np.random.rand() < self.flip_prob:
            vol = np.flip(vol, axis=2)  # flip H

        # Random flip along width axis
        if np.random.rand() < self.flip_prob:
            vol = np.flip(vol, axis=3)  # flip W

        # Random 90-degree rotation (depth axis stays, rotate H/W)
        if np.random.rand() < self.rotate_prob:
            k = np.random.choice([1, 2, 3])
            vol = np.rot90(vol, k=k, axes=(2, 3))  # rotate H/W

        # Normalize to [0, 1]
        if self.normalize:
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-5)

        return vol.astype(np.float32)

# Usage in RSNADataset
train_transform_3d = Simple3DTransform(flip_prob=0.5, rotate_prob=0.5, normalize=True)

# ====================================================
# 自定义 Dataset
# ====================================================
class RSNADataset(Dataset):
    def __init__(self, df, series_dir, img_size=(128,128), num_slices=64, transforms=None, cache=None):
        self.df = df
        self.series_dir = series_dir
        self.img_size = img_size
        self.num_slices = num_slices
        self.transforms = transforms
        self.cache = cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row[ID_COL]
        label = row[LABEL_COLS].values.astype(np.float32)
        if self.cache and series_id in self.cache:
            volume = self.cache[series_id]
        else:
            series_path = self.series_dir / str(series_id)
            volume = load_dicom_series(series_path, self.img_size, self.num_slices)
        # volume shape: (1, D, H, W)
        if self.transforms:
            volume = self.transforms(volume)  # Apply 3D transform to whole volume
        return torch.tensor(volume, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ====================================================
# 创建 DataLoaders
# ====================================================
# 加载 CSV
train_df = pd.read_csv(CFG.train_csv)
# Example: train_df = train_df[LABEL_COLS + [ID_COL]]

# 创建 dataset 和 dataloader
train_cache = preload_dicom_volumes(train_df, CFG.train_dir, img_size=(128,128), num_slices=64)
train_dataset = RSNADataset(train_df, CFG.train_dir, img_size=(128,128), num_slices=64, transforms=train_transform_3d, cache=train_cache)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)


# ====================================================
# 模型定义（保持不变）
# ====================================================
class AttentionGate(nn.Module):
    """注意力门控模块"""
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        psi = self.relu(g_conv + x_conv)
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class Dense3DBlock(nn.Module):
    """3D卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class RSNA_Model(nn.Module):
    """融合模型架构"""
    def __init__(self, model_name, num_classes, in_chans, pretrained=True):
        super().__init__()
        # 3D特征提取
        self.dense3d = Dense3DBlock(1, 8)
        
        # 主干网络
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans + 8  # 结合原始通道和3D特征
        )
        
        # 注意力机制
        self.attention = AttentionGate(
            in_channels=in_chans,
            gating_channels=in_chans,
            inter_channels=in_chans // 2
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(CFG.dropout),
            nn.Linear(self.backbone.num_features, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入形状: (B, C, H, W)
        batch_size, channels, height, width = x.shape
        
        # 3D特征提取
        x_3d = x.unsqueeze(1)  # 增加通道维度 (B, 1, C, H, W)
        x_3d = self.dense3d(x_3d)  # (B, 8, C, H, W)
        x_3d_agg = torch.mean(x_3d, dim=2)  # 聚合通道维度 (B, 8, H, W)
        
        # 注意力机制
        x_att = self.attention(x, x)  # (B, C, H, W)
        
        # 特征融合
        x_combined = torch.cat([x_att, x_3d_agg], dim=1)  # (B, C+8, H, W)
        
        # 主干网络特征提取
        features = self.backbone.forward_features(x_combined)
        
        # 分类
        output = self.classifier(features)
        return output

# ====================================================
# 损失函数（保持不变）
# ====================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.05):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.long().data, self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# ====================================================
# 训练和验证函数 - 优化数据加载流程
# ====================================================
def train_one_chunk(model, train_df, optimizer, criterion, scheduler, scaler, epoch, chunk_idx, total_chunks):
    """训练一个数据块"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # 创建数据集和加载器 - 优化参数
    dataset = RSNADataset(
        train_df,
        CFG.train_dir,
        img_size=(128,128),
        num_slices=64,
        transforms=train_transform_3d
        # cache=train_cache  # if you have a cache, otherwise omit
    )
    
    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=CFG.prefetch_buffer,  # 预加载
        persistent_workers=True,  # 保持工作进程
        worker_init_fn=lambda _: np.random.seed(torch.initial_seed() % 2**32)
    )
    
    print(f"\nEpoch {epoch+1} - 数据块 {chunk_idx+1}/{total_chunks}")
    start_time = time.time()
    
    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=CFG.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # 记录损失和预测
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        
        # 定期记录预测结果
        if batch_idx % 50 == 0 or batch_idx == len(loader) - 1:
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())
        
        # 打印进度
        if batch_idx % 20 == 0:
            print(f"  批次 {batch_idx+1}/{len(loader)} - 损失: {loss.item():.4f}")
        
        # 清理
        del images, labels, outputs, loss
    
    # 计算指标
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    
    mean_auc = 0.0
    if len(all_preds) > 0 and len(all_labels) > 0:
        auc_scores = []
        for i in range(NUM_CLASSES):
            try:
                if np.sum(all_labels[:, i]) > 0:
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                else:
                    auc = 0.5
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)
        mean_auc = np.mean(auc_scores)
    
    # 输出块训练时间
    chunk_time = time.time() - start_time
    print(f"  数据块训练时间: {chunk_time:.2f}秒 - 平均损失: {epoch_loss:.4f} - AUC: {mean_auc:.4f}")
    
    # 清理
    del loader, dataset, all_preds, all_labels
    gc.collect()
    torch.cuda.empty_cache()
    
    return epoch_loss, mean_auc

def validate(model, val_df, criterion):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # 创建验证数据集和加载器
    dataset = RSNADataset(
        val_df,
        CFG.train_dir,
        img_size=(128,128),
        num_slices=64,
        transforms=train_transform_3d,
        #cache=val_cache  # if you have a cache, otherwise omit
    )
    
    loader = DataLoader(
        dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        prefetch_factor=CFG.prefetch_buffer,
        persistent_workers=True
    )
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=CFG.use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())
            
            del images, labels, outputs, loss
    
    # 计算指标
    val_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    
    mean_auc = 0.0
    if len(all_preds) > 0 and len(all_labels) > 0:
        auc_scores = []
        for i in range(NUM_CLASSES):
            try:
                if np.sum(all_labels[:, i]) > 0:
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                else:
                    auc = 0.5
                auc_scores.append(auc)
            except:
                auc_scores.append(0.5)
        mean_auc = np.mean(auc_scores)
    
    # 输出验证时间
    val_time = time.time() - start_time
    print(f"验证时间: {val_time:.2f}秒 - 损失: {val_loss:.4f} - AUC: {mean_auc:.4f}")
    
    # 清理
    del loader, dataset, all_preds, all_labels
    gc.collect()
    torch.cuda.empty_cache()
    
    return val_loss, mean_auc

# ====================================================
# 主训练函数 - 增加预加载步骤
# ====================================================
def train_fold(fold, train_df, val_df):
    print(f"\n{'='*50}")
    print(f"开始训练 Fold {fold}")
    print(f"训练样本: {len(train_df)}, 验证样本: {len(val_df)}")
    print(f"{'='*50}\n")
    
    # 预加载当前fold的缓存
    print("预加载训练集缓存...")
    train_cache = preload_dicom_volumes(train_df, CFG.train_dir, img_size=(128,128), num_slices=64)
    print("预加载验证集缓存...")
    val_cache = preload_dicom_volumes(val_df, CFG.train_dir, img_size=(128,128), num_slices=64)
    
    # 初始化模型
    model = RSNA_Model(
        model_name=CFG.model_name,
        num_classes=NUM_CLASSES,
        in_chans=CFG.in_channels,
        pretrained=CFG.pretrained
    )
    model.to(device)
    
    # 数据并行（如果有多个GPU）
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # 损失函数和优化器
    if CFG.label_smoothing > 0:
        criterion = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=CFG.label_smoothing)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CFG.T_0,
        eta_min=CFG.min_lr
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.use_amp)
    
    # 训练记录
    best_auc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_auc': [],
        'val_loss': [], 'val_auc': []
    }
    
    # 计算数据块数量
    total_chunks = max(1, len(train_df) // CFG.chunk_size)
    print(f"数据分为 {total_chunks} 块，每块最多 {CFG.chunk_size} 样本")
    
    # 开始训练
    for epoch in range(CFG.epochs):
        start_time = time.time()
        train_loss = 0.0
        train_auc = 0.0
        
        # 按块训练
        for chunk_idx in range(total_chunks):
            # 获取当前块数据
            start = chunk_idx * CFG.chunk_size
            end = min(start + CFG.chunk_size, len(train_df))
            chunk_df = train_df.iloc[start:end].copy()
            
            # 训练当前块
            chunk_loss, chunk_auc = train_one_chunk(
                model, chunk_df, optimizer, criterion, scheduler,
                scaler, epoch, chunk_idx, total_chunks
            )
            
            # 累加
            train_loss += chunk_loss * len(chunk_df)
            train_auc += chunk_auc * len(chunk_df)
            
            # 清理
            del chunk_df
        
        # 计算平均训练指标
        train_loss /= len(train_df)
        train_auc /= len(train_df)
        
        # 验证
        val_loss, val_auc = validate(model, val_df, criterion)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # 打印 epoch 结果
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{CFG.epochs} - 总耗时: {epoch_time:.2f}秒")
        print(f"训练损失: {train_loss:.4f} - 训练AUC: {train_auc:.4f}")
        print(f"验证损失: {val_loss:.4f} - 验证AUC: {val_auc:.4f}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            model_path = CFG.model_dir / f"model_fold{fold}_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"保存最佳模型到 {model_path} (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CFG.patience:
                print(f"早停: {CFG.patience} 个epoch未改善")
                break
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title(f'Fold {fold} 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_auc'], label='训练AUC')
    plt.plot(history['val_auc'], label='验证AUC')
    plt.title(f'Fold {fold} AUC曲线')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(CFG.output_dir / f"fold_{fold}_metrics.png")
    plt.close()
    
    # 清理
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_auc


# ====================================================
# 主执行
# ====================================================
if __name__ == "__main__":
    print("加载训练数据...")
    train_df = pd.read_csv(CFG.train_csv)
    print(f"训练数据形状: {train_df.shape}")

    # 查看标签分布
    print("\n标签分布:")
    print(train_df[LABEL_COLS].sum().sort_values(ascending=False))

    # 创建交叉验证
    skf = StratifiedKFold(n_splits=CFG.fold, shuffle=True, random_state=42)
    train_df['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df['Aneurysm Present'])):
        train_df.loc[val_idx, 'fold'] = fold

    train_df.to_csv(CFG.output_dir / 'train_with_folds.csv', index=False)

    fold_aucs = []
    for fold in CFG.selected_folds:
        print(f"\nFold {fold} 数据预加载中...")
        train_fold_df = train_df[train_df['fold'] != fold].reset_index(drop=True)
        val_fold_df = train_df[train_df['fold'] == fold].reset_index(drop=True)

        # 使用多进程加速预加载
        train_cache = preload_dicom_volumes_parallel(
            train_fold_df, CFG.train_dir, img_size=(128,128), num_slices=64, num_workers=CFG.num_workers
        )
        val_cache = preload_dicom_volumes_parallel(
            val_fold_df, CFG.train_dir, img_size=(128,128), num_slices=64, num_workers=CFG.num_workers
        )

        # 创建 Dataset 和 DataLoader
        train_dataset = RSNADataset(train_fold_df, CFG.train_dir, img_size=(128,128), num_slices=64, transforms=train_transform_3d, cache=train_cache)
        val_dataset = RSNADataset(val_fold_df, CFG.train_dir, img_size=(128,128), num_slices=64, transforms=train_transform_3d, cache=val_cache)
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.val_batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

        # 训练
        fold_auc = train_fold(fold, train_fold_df, val_fold_df)
        fold_aucs.append(fold_auc)

        # 清理缓存为下一个fold做准备
        del train_cache, val_cache, train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("所有Fold训练结果:")
    for i, auc in enumerate(fold_aucs):
        print(f"Fold {i}: {auc:.4f}")
    print(f"平均AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print("="*50)
    
    # 保存总结
    with open(CFG.output_dir / "training_summary.txt", "w") as f:
        f.write("RSNA颅内动脉瘤检测训练总结\n")
        f.write("="*50 + "\n")
        for i, auc in enumerate(fold_aucs):
            f.write(f"Fold {i}: {auc:.4f}\n")
        f.write(f"平均AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}\n")
    
    print("训练完成!")

