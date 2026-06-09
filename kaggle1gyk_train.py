# -*- coding: utf-8 -*-
"""
训练脚本 for BirdCLEF 2025 - EfficientNet B0

基于预计算的多批次梅尔频谱图进行训练。
数据存储在 /kaggle/input/zwy-bird/precomputed_melspec_batch*.npy
参考脚本: [Train]-EfficientNet B0 Pytorch .py
"""
"""
输出什么时候保存？

1.  **最新检查点 (`..._latest.pth`)**:
    *   **何时保存**: 在 **每个 Epoch 结束时** 都会保存或更新一次。
    *   **保存内容**: 包含恢复训练所需的完整信息，包括模型权重、优化器状态、学习率调度器状态、当前 Epoch 数、以及记录的最佳验证 AUC 等。
    *   **目的**: 主要用于**断点续练**。如果训练中断，可以从这个文件恢复。
    *   **文件名**: 由 `cfg.checkpoint_filename` 定义，例如 `efficientnet_b0_fold0_latest.pth`。

2.  **最佳模型 (`..._best.pth`)**:
    *   **何时保存**: 只有当**当前 Epoch 的验证集 AUC (Area Under Curve) 分数 高于 此 Fold 之前所有 Epoch 的最佳 AUC 分数时**，才会保存或更新。
    *   **保存内容**: 默认只保存了模型的权重 (`model.state_dict()` 或 `model.module.state_dict()`)。脚本中也注释掉了保存完整检查点的选项。只保存权重通常用于后续的推理或评估。
    *   **目的**: 保存**验证集上表现最好**的模型，通常用于最终的预测或模型集成。
    *   **文件名**: 例如 `efficientnet_b0_fold0_best.pth`。

这两个文件都会被保存在 `cfg.OUTPUT_DIR` 指定的目录中，默认是 `/kaggle/working/`。

简单来说：

*   **每轮结束**存一个最新的进度 (`_latest.pth`)。
*   **表现有提升时**存一个最好的模型 (`_best.pth`)。

"""
import os
import random
import gc
import time
import glob  # 用于查找匹配的文件
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm.auto import tqdm
import warnings
import cv2  # 确保导入 cv2
import torch.nn.functional as F  # 确保导入 F
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# --- 配置 (CFG) ---
class CFG:
    # 基本设置
    seed = 42
    debug = False  # 是否开启 Debug 模式 (少量数据, 短 epoch)
    apex = False  # 是否使用 Apex 进行混合精度训练
    print_freq = 100  # 打印频率
    num_workers = 2  # DataLoader 使用的进程数

    # 路径设置
    OUTPUT_DIR = "/kaggle/working/"  # 模型和日志输出目录
    train_csv = "/kaggle/input/birdclef-2025/train.csv"  # 训练标签 CSV 文件
    taxonomy_csv = "/kaggle/input/birdclef-2025/taxonomy.csv"  # 物种分类 CSV 文件
    # !!! 修改为你包含所有 batch npy 文件的目录 !!!
    INPUT_SPEC_DIR = "/kaggle/input/zwy-bird/"  # 预计算频谱图 batch 文件所在目录
    # 例如: /kaggle/input/your-merged-dataset-name/

    # 模型设置
    model_name = "efficientnet_b0"  # 使用的模型名称 (来自 timm)
    pretrained = True  # 是否加载 timm 提供的预训练权重
    in_channels = 1  # 输入通道数 (灰度梅尔频谱图为 1)
    num_classes = 182  # 目标类别数 (后面会根据 taxonomy.csv 动态确定)

    # 数据集设置 (因为是预计算好的，所以不需要音频处理参数)
    TARGET_SHAPE = (256, 256)  # 预计算频谱图的目标形状 (需要与预计算时一致)

    # 训练设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 10  # 训练轮数
    batch_size = 128  # 批处理大小
    criterion = "BCEWithLogitsLoss"  # 损失函数 (多标签分类常用)

    # --- 新增：断点续练设置 ---
    resume_training = True  # 是否尝试从检查点恢复训练
    checkpoint_filename = "{model_name}_fold{fold}_latest.pth"  # 检查点文件名格式
    # --- 结束新增 ---

    # 交叉验证设置
    n_fold = 5  # 交叉验证折数
    selected_folds = [0]  # 选择训练的折数 (例如 [0, 1, 2, 3, 4] 训练所有折)

    # 优化器设置
    optimizer = "AdamW"  # 优化器类型
    lr = 1e-3  # 学习率 (AdamW 的推荐值，可以调整)
    weight_decay = 1e-5  # 权重衰减

    # 学习率调度器设置
    scheduler = "CosineAnnealingLR"  # 学习率调度器类型
    min_lr = 1e-6  # 最小学习率 (用于 CosineAnnealingLR)
    T_max = epochs  # CosineAnnealingLR 的周期 (通常设为 epochs)

    # 数据增强设置 (在 Dataset 中实现)
    aug_prob = 0.5  # 应用频谱图增强的概率
    mixup_alpha = 0.0  # Mixup 参数 (0 表示不使用 Mixup)

    def update_debug_settings(self):
        """如果 debug=True, 则减少 epochs 和 folds"""
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]
            self.batch_size = 16  # Debug 时减小 batch size


# --- 工具函数 ---
def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(targets, outputs):
    """计算多标签分类的平均 AUC"""
    num_classes = targets.shape[1]
    aucs = []

    # 计算概率 (Sigmoid)
    probs = 1 / (1 + np.exp(-outputs))

    for i in range(num_classes):
        # 检查该类别是否有正样本
        if (
            np.sum(targets[:, i]) > 0 and np.sum(1 - targets[:, i]) > 0
        ):  # 需要正负样本才能计算 AUC
            try:
                class_auc = roc_auc_score(targets[:, i], probs[:, i])
                aucs.append(class_auc)
            except ValueError as e:
                # print(f"计算类别 {i} 的 AUC 时出错: {e}") # 可以取消注释以调试
                pass  # 如果某个类别无法计算 AUC，则跳过

    return np.mean(aucs) if aucs else 0.0  # 如果没有可计算的 AUC，返回 0


# --- !!! 新增：将索引构建移出类外作为辅助函数 !!! ---
def build_sample_index(batch_files):
    """加载所有 batch 文件的 keys，建立 sample 到 batch 文件路径的映射。"""
    sample_map = {}
    print("正在建立样本索引...")
    for batch_path in tqdm(batch_files, desc="索引 Batch 文件"):
        try:
            # 只加载 keys，避免加载整个 batch 耗费内存
            # 注意：标准 np.load 可能仍会加载整个字典，但只取 keys 还是比全加载好
            batch_keys = list(np.load(batch_path, allow_pickle=True).item().keys())
            for key in batch_keys:
                if key in sample_map:
                    # print(
                    #     f"警告: 样本 '{key}' 在多个 batch 文件中找到。将使用路径: {batch_path}"
                    # )
                    pass
                sample_map[key] = batch_path
        except Exception as e:
            print(f"建立索引时加载 batch 文件 {batch_path} 出错: {e}")
    return sample_map


# --- 数据集类 (BirdCLEFDatasetFromBatches) ---
class BirdCLEFDatasetFromBatches(Dataset):
    """
    从多个预计算的 .npy batch 文件加载梅尔频谱图的数据集类。
    采用按需加载策略优化内存使用。
    接收预构建的索引以避免重复扫描。
    """

    # --- 修改：接收 batch_files 和 sample_index ---
    def __init__(self, df, cfg, mode="train", batch_files=None, sample_index=None):
        """
        Args:
                df (pd.DataFrame): 包含样本信息 (filename, primary_label 等) 的 DataFrame。
                cfg (CFG): 配置对象。
                mode (str): 'train' 或 'valid'/'test'。
                batch_files (list, optional): 预扫描的 batch 文件路径列表。
                sample_index (dict, optional): 预构建的 sample_name 到 batch_path 的映射。
        """
        self.df = df.copy()
        self.cfg = cfg
        self.mode = mode

        # 加载分类信息
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df["primary_label"].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        # --- 优化：使用传入的索引和文件列表 ---
        if batch_files is not None and sample_index is not None:
            print(
                f"使用预构建的索引 (含 {len(sample_index)} 个样本) 和 {len(batch_files)} 个 batch 文件路径。"
            )
            self.batch_files = batch_files
            self.sample_to_batch_info = sample_index
        else:
            # --- Fallback (理论上不应执行，除非直接调用 Dataset 类) ---
            print(
                "警告: 未提供预构建索引或文件列表，将在 Dataset 内部重新扫描和构建..."
            )
            self.batch_files = sorted(
                glob.glob(
                    os.path.join(
                        self.cfg.INPUT_SPEC_DIR, "precomputed_melspec_batch*.npy"
                    )
                )
            )
            if not self.batch_files:
                raise FileNotFoundError(
                    f"在目录下未找到任何 precomputed_melspec_batch*.npy 文件: {self.cfg.INPUT_SPEC_DIR}"
                )
            # 需要一个内部的 build_index 或调用外部函数
            # 为避免混淆，这里直接报错，强制要求从外部传入
            raise ValueError("必须通过构造函数提供 batch_files 和 sample_index！")
            # self.sample_to_batch_info = build_sample_index(self.batch_files) # 或者这样写，但不推荐

        # --- 根据最终使用的索引过滤 DataFrame ---
        self._filter_df_by_index()

        # 检查传入的 df 是否确实有 samplename (过滤后检查)
        if "samplename" not in self.df.columns:
            raise ValueError("错误: DataFrame (过滤后) 必须包含 'samplename' 列！")

        if cfg.debug and mode == "train":  # Debug 模式下减少训练数据量
            sample_size = min(1000, len(self.df))
            self.df = self.df.sample(sample_size, random_state=cfg.seed).reset_index(
                drop=True
            )
            print(f"Debug模式：使用 {len(self.df)} 个训练样本")

        # --- 优化：初始化缓存 ---
        self.current_batch_path = None
        self.current_batch_data = None

    # --- 移除 _build_index 方法 ---
    # def _build_index(self): ... # 不再需要

    def _filter_df_by_index(self, key_column="samplename"):
        # ... (方法内容不变，使用 self.sample_to_batch_info) ...
        original_len = len(self.df)
        if key_column in self.df.columns:
            available_samples = set(self.sample_to_batch_info.keys())
            self.df = self.df[self.df[key_column].isin(available_samples)].reset_index(
                drop=True
            )
            filtered_len = len(self.df)
            if filtered_len < original_len:
                print(
                    f"过滤完成: 从 {original_len} 个样本中移除了 {original_len - filtered_len} 个在预计算数据索引中找不到的样本。剩余 {filtered_len} 个样本。"
                )
            if filtered_len == 0:
                print(
                    f"警告: 过滤后数据集为空！请检查 '{key_column}' 列与 .npy 文件 key 是否匹配。"
                )
        else:
            print(
                f"警告: DataFrame 中缺少用于过滤的列 '{key_column}'。跳过基于索引的过滤。"
            )

    def __len__(self):
        return len(self.df)

    def _load_batch(self, batch_path):
        """加载指定的 batch 文件并更新缓存。"""
        # print(f"缓存未命中，加载 batch: {os.path.basename(batch_path)}") # Debugging line
        try:
            self.current_batch_data = np.load(batch_path, allow_pickle=True).item()
            self.current_batch_path = batch_path
            # gc.collect() # 可以取消注释以更积极地释放内存，但可能影响性能
        except Exception as e:
            print(f"加载 batch 文件 {batch_path} 时出错: {e}")
            # 如果加载失败，清空缓存避免使用错误数据
            self.current_batch_path = None
            self.current_batch_data = None
            return False
        return True

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 假设 'samplename' 列存在且包含正确的 key
        samplename = row["samplename"]

        spec = None
        if samplename in self.sample_to_batch_info:
            target_batch_path = self.sample_to_batch_info[samplename]

            # --- 优化：检查并加载 batch ---
            if target_batch_path != self.current_batch_path:
                if not self._load_batch(target_batch_path):
                    # 加载失败，返回空谱图
                    print(
                        f"错误: 无法加载样本 {samplename} 所在的 batch 文件 {target_batch_path}。"
                    )
                    spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)

            # --- 优化：从缓存中获取数据 ---
            if (
                self.current_batch_data is not None
                and samplename in self.current_batch_data
            ):
                spec = self.current_batch_data[samplename]
            elif spec is None:  # 如果加载成功但key不在里面（理论上不应发生）或加载失败
                print(
                    f"警告: 样本 {samplename} 在其声称的 batch 文件 {target_batch_path} 中未找到！"
                )
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)

        else:
            # 理论上不应该发生，因为我们已经根据索引过滤了 df
            print(f"警告: 样本 {samplename} 在索引中找不到！返回空频谱图。")
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)

        # 确保频谱图是正确的形状和类型
        if spec.shape != self.cfg.TARGET_SHAPE:
            try:
                # print(f"警告: 样本 {samplename} 的频谱图形状 {spec.shape} 与目标形状 {self.cfg.TARGET_SHAPE} 不符。尝试调整大小。")
                spec = cv2.resize(
                    spec, self.cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR
                )
            except Exception as e:
                print(f"错误: 调整样本 {samplename} 大小时出错 ({e})。返回空频谱图。")
                spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)

        spec = torch.tensor(spec.astype(np.float32)).unsqueeze(0)  # 添加通道维度

        # 应用数据增强 (仅训练模式)
        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        # 编码标签 (应用新的权重逻辑)
        target = self.encode_label(
            row["primary_label"], row.get("secondary_labels")
        )  # 使用 .get 处理可能不存在的列

        return {
            "melspec": spec,
            "target": torch.tensor(target, dtype=torch.float32),
            "samplename": samplename,  # 或其他你需要的元数据
        }

    def apply_spec_augmentations(self, spec):
        """对频谱图应用增强 (例如 Time/Frequency Masking)"""
        # ... (可以从参考脚本复制或实现你需要的增强) ...
        # 示例：频率掩码
        if random.random() < 0.5:
            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                height = random.randint(5, 25)  # 掩码高度
                start = random.randint(0, spec.shape[1] - height)  # 频率轴起始位置
                spec[0, start : start + height, :] = 0  # 将该区域置0
        # 示例: 时间掩码
        if random.random() < 0.5:
            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                width = random.randint(5, 25)  # 掩码宽度
                start = random.randint(0, spec.shape[2] - width)  # 时间轴起始位置
                spec[0, :, start : start + width] = 0  # 将该区域置0
        return spec

    def encode_label(self, primary_label, secondary_labels=None):
        """
        将主标签和次要标签编码为目标向量。
        如果存在次要标签，主标签权重0.5，剩余0.5平均分配给次要标签。
        """
        target = np.zeros(self.num_classes, dtype=np.float32)
        valid_secondary = []

        # 解析次要标签
        # 修复 Linter Error: 确保列表格式正确
        if (
            secondary_labels is not None
            and secondary_labels != [""]
            and not pd.isna(secondary_labels)
        ):
            if isinstance(secondary_labels, str):
                try:
                    # 尝试解析字符串形式的列表
                    parsed_labels = eval(secondary_labels)
                    if isinstance(parsed_labels, list):
                        secondary_labels = parsed_labels
                    else:
                        secondary_labels = []  # 解析结果不是列表，视为空
                except:
                    secondary_labels = []  # 解析失败则视为空

            if isinstance(secondary_labels, list):
                # 筛选出有效的次要标签
                valid_secondary = [
                    l for l in secondary_labels if l in self.label_to_idx
                ]

        # 根据是否有有效的次要标签来分配权重
        if primary_label in self.label_to_idx:
            primary_idx = self.label_to_idx[primary_label]
            if valid_secondary:
                # 有次要标签：主标签权重 0.5
                target[primary_idx] = 0.5
                # 剩余 0.5 平均分配给次要标签
                # 确保除数不为零
                if len(valid_secondary) > 0:
                    sec_weight = 0.5 / len(valid_secondary)
                else:
                    sec_weight = 0  # 不应该发生，但作为保险

                for label in valid_secondary:
                    # 注意：如果次要标签和主标签相同，这里会覆盖主标签的0.5权重
                    # 如果不希望覆盖，可以用 max(target[...], sec_weight) 或其他逻辑
                    target[
                        self.label_to_idx[label]
                    ] += sec_weight  # 使用 += 避免覆盖主标签权重（如果次要标签包含主标签）
                # 修正可能存在的重复标签累加问题（例如主标签也在次要里）
                target[primary_idx] = max(
                    target[primary_idx], 0.5
                )  # 确保主标签权重至少为0.5
                # 归一化（可选，但推荐，以防权重总和略超1）
                # current_sum = target.sum()
                # if current_sum > 1e-6: # 避免除以零
                #     target = target / current_sum

            else:
                # 没有次要标签：主标签权重 1.0
                target[primary_idx] = 1.0

        elif valid_secondary:  # 如果主标签无效，但有次要标签
            print(
                f"警告: 主标签 '{primary_label}' 无效，但存在有效的次要标签。仅分配次要标签权重。"
            )
            # 可以选择将所有权重 (1.0) 分配给次要标签
            if len(valid_secondary) > 0:
                sec_weight = 1.0 / len(valid_secondary)
            else:
                sec_weight = 0
            for label in valid_secondary:
                target[self.label_to_idx[label]] = sec_weight

        return target


# --- 模型类 (BirdCLEFModel) ---
class BirdCLEFModel(nn.Module):
    """
    使用 timm 库创建的 EfficientNet B0 模型。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 动态获取类别数
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        self.num_classes = len(taxonomy_df)
        self.cfg.num_classes = self.num_classes  # 更新 CFG 中的 num_classes

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            in_chans=cfg.in_channels,
            drop_rate=0.2,  # 可以调整
            drop_path_rate=0.2,  # 可以调整
        )

        # 获取 backbone 输出特征维度并替换分类器
        if hasattr(self.backbone, "classifier"):
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            # 尝试通用的 get_classifier() 方法
            try:
                backbone_out = self.backbone.get_classifier().in_features
                self.backbone.reset_classifier(0, "")  # 移除分类器
            except AttributeError:
                # 如果以上都不行，可能需要针对特定模型结构进行调整
                # 或者检查 timm 的文档
                raise ValueError(
                    f"无法自动确定模型 {cfg.model_name} 的输出特征维度或移除分类器。"
                )

        self.pooling = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.classifier = nn.Linear(backbone_out, self.num_classes)  # 最终分类层

        # Mixup 相关 (如果启用)
        self.mixup_enabled = hasattr(cfg, "mixup_alpha") and cfg.mixup_alpha > 0
        if self.mixup_enabled:
            self.mixup_alpha = cfg.mixup_alpha

    def forward(self, x, targets=None):
        """模型前向传播"""
        # Mixup 处理 (如果启用且在训练模式)
        if self.training and self.mixup_enabled and targets is not None:
            mixed_x, targets_a, targets_b, lam = self.mixup_data(x, targets)
            x = mixed_x
        else:
            targets_a, targets_b, lam = None, None, None  # 确保定义了这些变量

        # === 修改：显式调用 forward_features ===
        features = self.backbone.forward_features(x)
        # 现在 features 的形状应该是 (B, C, H, W), 例如 (B, 1280, 8, 8)

        # 如果 backbone 输出是字典 (某些 timm 模型会这样，虽然 forward_features 通常不会)
        if isinstance(features, dict):
            features = features["features"]  # 或其他合适的 key

        # 应用池化和展平
        pooled_features = self.pooling(features).flatten(1)  # 形状变为 (B, C)
        logits = self.classifier(
            pooled_features
        )  # 通过分类器得到 logits (C 应该等于 1280)

        # 如果启用了 Mixup，计算混合损失
        if self.training and self.mixup_enabled and targets is not None:
            # 确保 mixup_criterion 定义在类中
            loss = self.mixup_criterion(
                F.binary_cross_entropy_with_logits,  # 使用 nn.functional
                logits,
                targets_a,
                targets_b,
                lam,
            )
            return logits, loss

        # 如果没有 Mixup 或在评估模式，只返回 logits
        return logits

    # --- Mixup 辅助函数 (如果启用 Mixup) ---
    def mixup_data(self, x, targets):
        """应用 Mixup"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        indices = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[indices]
        targets_a, targets_b = targets, targets[indices]
        return mixed_x, targets_a, targets_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """计算 Mixup 损失"""
        return lam * criterion(pred, y_a, reduction="mean") + (1 - lam) * criterion(
            pred, y_b, reduction="mean"
        )


# --- 训练与验证循环 ---
def train_one_epoch(
    model, loader, optimizer, criterion, device, scheduler=None, cfg=None
):
    """训练一个 epoch"""
    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    pbar = tqdm(enumerate(loader), total=len(loader), desc="训练中")
    for step, batch in pbar:
        inputs = batch["melspec"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # 处理 Mixup (如果模型返回 logits 和 loss)
        if hasattr(cfg, "mixup_alpha") and cfg.mixup_alpha > 0:
            logits, loss = model(inputs, targets)
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        # 如果使用 OneCycleLR，则每个 step 更新 scheduler
        if scheduler is not None and cfg.scheduler == "OneCycleLR":
            scheduler.step()

        losses.append(loss.item())
        all_outputs.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

        pbar.set_postfix(
            {
                "loss": np.mean(losses[-10:]) if losses else 0,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    # 计算 epoch 的平均损失和 AUC
    avg_loss = np.mean(losses)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)  # 现在调用补全的函数
    return avg_loss, auc  # 返回平均损失和 AUC


def validate(model, loader, criterion, device):
    """验证模型在一个 epoch 上的表现"""
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc="验证中")
        for step, batch in pbar:
            inputs = batch["melspec"].to(device)
            targets = batch["target"].to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            losses.append(loss.item())
            all_outputs.append(logits.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            pbar.set_postfix({"loss": np.mean(losses[-10:]) if losses else 0})

    # 计算验证集的平均损失和 AUC
    avg_loss = np.mean(losses)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)  # 现在调用补全的函数
    return avg_loss, auc  # 返回平均损失和 AUC


# --- 主训练函数 ---
def run_training(cfg):
    """执行完整的训练流程 (包括交叉验证)"""
    set_seed(cfg.seed)

    # --- 优化：只在开始时扫描文件和构建索引一次 ---
    print("开始扫描 Batch 文件并构建全局索引...")
    batch_files = sorted(
        glob.glob(os.path.join(cfg.INPUT_SPEC_DIR, "precomputed_melspec_batch*.npy"))
    )
    if not batch_files:
        raise FileNotFoundError(f"在 {cfg.INPUT_SPEC_DIR} 未找到任何 batch 文件！")
    print(f"找到 {len(batch_files)} 个 batch 文件。")

    # 构建全局索引
    global_sample_index = build_sample_index(batch_files)
    all_keys = list(global_sample_index.keys())  # 从索引获取 keys

    if not all_keys:
        raise ValueError("未能从任何 batch 文件中加载 keys 或构建索引！")
    print(f"全局索引构建完成，包含 {len(global_sample_index)} 个唯一样本 key。")

    # --- !!! 关键步骤：解析 key 以获取原始文件名 !!! ---
    # (这部分逻辑保持不变，但现在基于 all_keys)
    def parse_key_to_filename_example1(key):
        # ... (你的解析逻辑) ...
        parts = key.split("_chunk_")[0]
        # 检查拆分结果是否符合预期
        if "-" not in parts:
            # print(f"警告: Key '{key}' 不符合 'dirname-basename_chunk_' 格式，跳过解析。")
            return None  # 或引发错误，或返回特殊值
        dirname, basename = parts.split("-", 1)
        return f"{dirname}/{basename}.ogg"

    parsed_data = []
    for key in tqdm(all_keys, desc="解析 Keys"):
        try:
            original_filename = parse_key_to_filename_example1(key)
            if original_filename:  # 确保解析成功
                parsed_data.append(
                    {"samplename": key, "original_filename_parsed": original_filename} 
                )
        except Exception as e:
            print(f"解析 key '{key}' 时出错: {e}")

    if not parsed_data:
        raise ValueError("解析 key 失败，无法创建 key DataFrame。请检查解析逻辑！")

    key_df = pd.DataFrame(parsed_data)
    # ... (加载 original_train_df, 合并 df 的逻辑保持不变) ...
    original_train_df = pd.read_csv(cfg.train_csv)
    df = pd.merge(
        key_df,
        original_train_df,
        left_on="original_filename_parsed",
        right_on="filename",
        how="left",
    )
    # ... (检查 missing_labels 的逻辑保持不变) ...
    missing_labels = df["primary_label"].isnull().sum()
    if missing_labels > 0:
        print(f"警告: 合并后发现 {missing_labels} 行缺少 primary_label。")
        # 可选：移除或填充缺失标签的行
        # df = df.dropna(subset=['primary_label']).reset_index(drop=True)
        # df['primary_label'] = df['primary_label'].fillna('unknown_species') # 或其他填充策略

    # 确保 primary_label 列存在且可用于分层抽样
    if "primary_label" not in df.columns or df["primary_label"].isnull().any():
        raise ValueError(
            "错误: 合并后 DataFrame 缺少 'primary_label' 或存在空值，无法进行分层 K 折拆分。请检查数据或合并逻辑。"
        )

    # --- 现在 df 包含了正确的 'samplename' 和对应的标签 ---

    if cfg.debug:
        cfg.update_debug_settings()
        print("Debug 模式已启用。")

    # 交叉验证设置
    # --- 修正：确保标签列中至少有 n_splits 个不同的类，或者处理无法分层的情况 ---
    if df["primary_label"].nunique() < cfg.n_fold:
        print(
            f"警告: 数据集中不同主标签的数量 ({df['primary_label'].nunique()}) 少于指定的折数 ({cfg.n_fold})。将使用非分层 KFold。"
        )
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
        split_iterator = kf.split(df)  # 非分层拆分
    else:
        skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
        split_iterator = skf.split(df, df["primary_label"])  # 分层拆分

    oof_auc_scores = []

    # --- 修改：使用 split_iterator ---
    for fold, (train_idx, val_idx) in enumerate(split_iterator):
        if fold not in cfg.selected_folds:
            continue

        print(f'\n{"="*30} Fold {fold} {"="*30}')
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # 创建 Dataset 和 DataLoader (传递索引)
        print("创建训练数据集 (使用全局索引)...")
        train_dataset = BirdCLEFDatasetFromBatches(
            train_df,
            cfg,
            mode="train",
            batch_files=batch_files,
            sample_index=global_sample_index,
        )
        print("创建验证数据集 (使用全局索引)...")
        val_dataset = BirdCLEFDatasetFromBatches(
            val_df,
            cfg,
            mode="valid",
            batch_files=batch_files,
            sample_index=global_sample_index,
        )

        # ... (检查数据集是否为空，创建 DataLoader, 初始化模型等逻辑保持不变) ...
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"错误: Fold {fold} 的训练或验证数据集在过滤后为空。跳过此 Fold。")
            continue

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size * 2,
            shuffle=False,  # 验证时 batch 可以大一些
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        # 初始化模型、优化器、损失函数、调度器
        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        scheduler = get_scheduler(optimizer, cfg)  # scheduler 可能为 None

        # --- 修改：添加检查点加载逻辑 ---
        start_epoch = 0
        best_val_auc = 0.0
        checkpoint_path = os.path.join(
            cfg.OUTPUT_DIR,
            cfg.checkpoint_filename.format(model_name=cfg.model_name, fold=fold),
        )

        if cfg.resume_training and os.path.exists(checkpoint_path):
            print(f"发现检查点: {checkpoint_path}，尝试恢复训练...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # 加载 scheduler 状态时需要检查是否存在以及类型是否匹配
                if (
                    scheduler  # 确保 scheduler 被初始化了
                    and "scheduler_state_dict" in checkpoint
                    and checkpoint["scheduler_state_dict"]  # 确保检查点里存了这个状态
                ):
                    try:
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                        print("成功加载 Scheduler 状态")
                    except Exception as scheduler_load_error:
                        print(
                            f"警告：加载 Scheduler 状态时出错 (可能是类型不匹配或 scheduler 配置已更改): {scheduler_load_error}。将使用新的 Scheduler 状态。"
                        )
                        # 可选：根据需要重置 scheduler 或使用新的

                start_epoch = checkpoint["epoch"] + 1
                # 确保从检查点恢复 best_val_auc，即使 scheduler 加载失败
                best_val_auc = checkpoint.get(
                    "best_val_auc", 0.0
                )  # 使用 .get 以兼容旧的检查点
                print(
                    f"恢复成功，将从 Epoch {start_epoch} 开始训练。之前的最佳 AUC: {best_val_auc:.4f}"
                )
                # 清理 checkpoint 占用的内存
                del checkpoint
                gc.collect()
                if cfg.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"加载检查点失败: {e}。将从头开始训练。")
                start_epoch = 0
                best_val_auc = 0.0
        else:
            if cfg.resume_training:
                print(f"未找到检查点 {checkpoint_path} 或检查点无效。")
            print(f"将从 Epoch 0 开始训练。")
            start_epoch = 0
            best_val_auc = 0.0
        # --- 结束修改 ---

        best_epoch = start_epoch - 1  # 初始化 best_epoch，如果从0开始则是-1

        # --- 修改：调整 Epoch 循环范围 ---
        for epoch in range(start_epoch, cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")  # 打印时仍用 epoch+1
            epoch_start_time = time.time()

            # 训练和验证
            train_loss, train_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, cfg.device, scheduler, cfg
            )
            val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)

            # --- 补充：更新学习率 ---
            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_auc)  # 基于验证 AUC 更新
                elif (
                    cfg.scheduler != "OneCycleLR"
                ):  # OneCycleLR 在 train_one_epoch 中更新
                    scheduler.step()

            epoch_time = time.time() - epoch_start_time
            print(f"耗时: {epoch_time:.2f}s")
            print(f"训练损失: {train_loss:.4f}, 训练 AUC: {train_auc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证 AUC: {val_auc:.4f}")

            # --- 修改：保存逻辑 ---
            # 1. 在每个 epoch 结束后都保存最新检查点 (包含所有状态)
            latest_checkpoint_path = os.path.join(
                cfg.OUTPUT_DIR,
                cfg.checkpoint_filename.format(model_name=cfg.model_name, fold=fold),
            )
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc": best_val_auc,  # 保存当前的最佳 AUC，以便恢复时知道历史最佳
                "current_val_auc": val_auc,  # 保存当前的 AUC 便于查看
            }
            # 只有当 scheduler 存在时才保存其状态
            if scheduler is not None:
                save_dict["scheduler_state_dict"] = scheduler.state_dict()

            torch.save(save_dict, latest_checkpoint_path)
            # print(f"已保存最新检查点到: {latest_checkpoint_path}") # 可以取消注释以确认保存

            # 2. 如果当前是最佳模型，额外保存一个 _best.pth (只含模型权重)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1  # 记录最佳 epoch (从 1 开始计数)
                print(
                    f"*** Fold {fold} 找到新的最佳 AUC: {best_val_auc:.4f} at epoch {best_epoch} ***"
                )
                best_model_path = os.path.join(
                    cfg.OUTPUT_DIR, f"{cfg.model_name}_fold{fold}_best.pth"
                )
                # 只保存模型权重，方便后续推理或集成
                torch.save(model.state_dict(), best_model_path)
                print(f"已保存最佳模型权重到: {best_model_path}")

            # 移除旧的 _last.pth 保存逻辑 (如果存在)
            # --- 逻辑已包含在 best_val_auc 更新中 ---

        # --- 结束 Epoch 循环 ---

        print(
            f"\nFold {fold} 训练完成。最佳验证 AUC: {best_val_auc:.4f} (Epoch {best_epoch if best_epoch > 0 else 'N/A'})"
        )
        oof_auc_scores.append(best_val_auc)  # 记录该 fold 的最佳分数

        # --- 补充：清理内存 ---
        print(f"Fold {fold} 结束，清理内存...")
        del (
            model,
            train_dataset,
            val_dataset,
            train_loader,
            val_loader,
            optimizer,
            criterion,
        )
        if scheduler is not None:
            del scheduler
        gc.collect()  # 强制垃圾回收
        if cfg.device == "cuda":
            torch.cuda.empty_cache()  # 清空未使用的 CUDA 缓存

    # --- 结束 Fold 循环 ---

    # --- 补充：打印 OOF AUC 结果 ---
    if oof_auc_scores:  # 确保至少完成了一个 fold
        mean_oof_auc = np.mean(oof_auc_scores)
        print(f'\n{"="*30} 训练结束 {"="*30}')
        print(
            f"所有选定 Folds ({cfg.selected_folds}) 的 OOF AUC 分数: {oof_auc_scores}"
        )
        print(f"平均 OOF AUC: {mean_oof_auc:.4f}")
    else:
        print("\n没有 Fold 被训练或完成，无法计算 OOF AUC。")


# --- Helper Functions (补全实现) ---
def get_optimizer(model, cfg):
    """根据配置返回优化器实例"""
    if cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    # 可以根据需要添加 SGD 等其他优化器选项
    # elif cfg.optimizer == 'SGD':
    #     optimizer = optim.SGD(
    #         model.parameters(),
    #         lr=cfg.lr,
    #         momentum=0.9,
    #         weight_decay=cfg.weight_decay
    #     )
    else:
        raise ValueError(f"不支持的优化器: {cfg.optimizer}")
    return optimizer


def get_scheduler(optimizer, cfg):
    """根据配置返回学习率调度器实例"""
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr
        )
    elif cfg.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # 通常监控验证集 AUC，所以用 max
            factor=0.5,  # 学习率衰减因子
            patience=2,  # 多少个 epoch AUC 没有提升则降低学习率
            min_lr=cfg.min_lr,
            verbose=True,
        )
    # 可以根据需要添加 StepLR 等其他调度器选项
    # elif cfg.scheduler == 'StepLR':
    #     scheduler = lr_scheduler.StepLR(
    #         optimizer,
    #         step_size=cfg.epochs // 3,
    #         gamma=0.5
    #     )
    elif cfg.scheduler == "OneCycleLR":
        # OneCycleLR 需要在每个 step 更新，特殊处理
        # 初始化在主训练循环中进行
        scheduler = None
    elif cfg.scheduler is None:
        scheduler = None
    else:
        raise ValueError(f"不支持的调度器: {cfg.scheduler}")
    return scheduler


def get_criterion(cfg):
    """根据配置返回损失函数实例"""
    if cfg.criterion == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    # 可以根据需要添加 CrossEntropyLoss 等其他损失函数选项
    # elif cfg.criterion == 'CrossEntropyLoss':
    #     criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"不支持的损失函数: {cfg.criterion}")
    return criterion


# --- 主程序入口 (保持不变) ---
if __name__ == "__main__":
    print("初始化配置...")
    cfg = CFG()

    # 创建输出目录
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("开始训练流程...")
    run_training(cfg)
    print("训练流程结束。")





