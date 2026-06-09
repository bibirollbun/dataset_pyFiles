# 这里自己修改
RAW_DATA_PATH = "/kaggle/input/playground-series-s5e11/train.csv"
TEST_DATA_PATH = "/kaggle/input/playground-series-s5e11/test.csv"
SUBMISSION_PATH = "/kaggle/working/best_submission.csv"  # 输出路径

ID_ITEM = "id"
LABEL_ITEM = "loan_paid_back"
LOSS_FUNCTION = "binary_cross_entropy"  # 可选项: mse, l1, binary_cross_entropy
IGNORE_ITEMS = [ID_ITEM, LABEL_ITEM]

NUM_VAL_CYCLES = 20
VAL_PER_STEP = 256
CONFIG = {
    "dim_model": [192],
    "dim_feedforward": [2048],
    "num_layers_per_block": [12],
    "num_interaction_blocks": [1],
    "variance_bins": [1536],
    "batch_size": [1280],
    "continuous_lr": [1e-4],
    "interaction_lr": [5e-5],
    "weight_decay": [1e-2],
    "dropout": [0.9],
}


# 模型的定义
import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim_model: int, dim_feedforward: int, dropout: float = 0.):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, dim_feedforward * 2)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.scale = nn.Parameter(torch.ones(1) * 1e-5)
        self.norm = nn.BatchNorm1d(dim_model)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        for module in [self.linear1, self.linear2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        gate, value = self.linear1(self.norm(x)).chunk(2, dim=-1)
        return x + self.linear2(self.dropout(value * F.silu(gate))) * self.scale


class FeatureInteractionBlock(nn.Module):
    def __init__(self, dim_model: int, dim_feedforward: int, num_layers: int, dropout: float = 0):
        super().__init__()

        # SwiGLU 交互层，学习特征间的高阶交互
        self.interaction_layers = nn.ModuleList(
            SwiGLU(dim_model, dim_feedforward, dropout)
            for _ in range(num_layers)
        )

        # 预测器，将隐藏表示映射到预测值
        self.predictor = nn.Linear(dim_model, 1)

        # 输出缩放因子，控制预测输出的权重
        self.output_scale = nn.Parameter(torch.ones(1) * 1e-5)

        # 初始化权重
        nn.init.xavier_uniform_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 通过交互层学习特征间复杂关系
        for interaction_layer in self.interaction_layers:
            hidden_state = interaction_layer(hidden_state)

        # 预测并调整输出权重
        prediction = self.predictor(hidden_state).squeeze(-1) * self.output_scale
        return prediction, hidden_state


class FeatureInteractionPredictor(nn.Module):
    def __init__(
        self,
        categorical_cardinalities: list[int],  # 离散特征的类别数量 [feature1_classes, feature2_classes, ...]
        num_continuous: int,                   # 连续特征的数量
        dim_model: int,                        # 模型隐藏层维度
        dim_feedforward: int,                  # 前馈网络中间层维度
        num_layers_per_block: int,             # 每个交互块中的层数
        num_interaction_blocks: int,           # 特征交互块的数量
        dropout: float = 0.                    # Dropout 比率
    ):
        super().__init__()
        self.dim_model = dim_model

        # 连续特征处理层
        self.continuous_projection = nn.Linear(num_continuous, 1)

        # 特征嵌入层，将离散特征映射到连续向量空间
        self.feature_embeddings = nn.ModuleList(nn.Embedding(num_categories, dim_model) for num_categories in categorical_cardinalities)

        # 连续特征转换到向量空间
        self.continuous_embedding = nn.Linear(num_continuous, dim_model)

        # 多层特征交互块，相当于 TTS 生成中的 Post-Net
        self.interaction_blocks = nn.ModuleList(FeatureInteractionBlock(dim_model, dim_feedforward, num_layers_per_block, dropout) for _ in range(num_interaction_blocks))

        # 初始化权重
        nn.init.zeros_(self.continuous_projection.weight)
        nn.init.zeros_(self.continuous_projection.bias)
        nn.init.zeros_(self.continuous_embedding.bias)
        for module in [*self.feature_embeddings, self.continuous_embedding]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, categorical_features: torch.LongTensor, continuous_features: torch.Tensor) -> list[torch.Tensor]:
        # 初始化隐藏状态
        hidden_state = self.continuous_embedding(continuous_features)

        # 加权求和所有特征的嵌入表示
        for feature_idx, feature_column in enumerate(categorical_features.T):  # 遍历每个特征列
            hidden_state = hidden_state + self.feature_embeddings[feature_idx](feature_column)

        # 初始预测，连续特征的线性投影
        predictions = [self.continuous_projection(continuous_features).squeeze(-1)]

        # 通过多个交互块逐步细化预测
        for interaction_block in self.interaction_blocks:
            # 残差连接，预测上一个预测与最终值的残差
            prediction, hidden_state = interaction_block(hidden_state)
            predictions.append(predictions[-1] + prediction)

        return predictions



# 数据集和 collate_fn
import json
from typing import Any
import pandas as pd
from torch.utils.data import Dataset


# 定义一个数据集
class FeatureInteractionDataset(Dataset):
    def __init__(self, dataset: list[tuple[int, list[int], list[float], float]]):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx: int) -> tuple[int, list[int], list[float], float]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn(batch: list[tuple[int, list[int], list[float], float]]):
    batch = [
        torch.tensor(item)
        for item in zip(*batch)
    ]
    return [
        item.to(dtype=torch.float32) if item.is_floating_point() else item
        for item in batch
    ]



# 数据预处理
import math
import pandas as pd
import numpy as np
from collections.abc import Callable
from tqdm import tqdm


def prepare_dataset(variance_bins: int):
    # 读取数据
    raw_data = pd.read_csv(RAW_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    
    # 对特征列排序
    sorted_columns = sorted([
        feature_name
        for feature_name in raw_data
        if feature_name not in IGNORE_ITEMS
    ])
    
    # 获离散化映射表和标准化参数
    feature_discretizers = []  # 每个特征的离散化配置
    feature_bin_counts = []  # 每个特征的离散化分箱数量
    feature_standardizers = {}  # 数值特征的标准化参数（均值和标准差）
    for feature_name in sorted_columns:
        feature_values = raw_data[feature_name]
    
        if isinstance(list(feature_values)[0], str):
            # 处理类别型特征，创建类别到索引的映射字典
            category_mapping = {category: idx for idx, category in enumerate(set(feature_values))}
            feature_discretizers.append(category_mapping)
            feature_bin_counts.append(len(category_mapping))
        else:
            # 处理数值型特征，基于百分位数计算离散化边界
            feature_array = feature_values.to_numpy()
    
            # 计算 5% 和 95% 分位数作为离散化边界，避免异常值影响
            lower_bound, upper_bound = np.percentile(feature_array, [5, 95])
            value_range = upper_bound - lower_bound + 1e-8  # 添加小值防止除零
            feature_discretizers.append((lower_bound, value_range))
            feature_bin_counts.append(variance_bins)  # 使用预定义的方差分箱数

            # 为数值特征计算标准化参数（Z-score 标准化）
            feature_standardizers[feature_name] = (feature_array.mean(), feature_array.std() + 1e-8)
    
    # 获取标签处理逆向函数
    if LOSS_FUNCTION == "binary_cross_entropy":
        def label_inverser(x: float):
            if x >= 0:
                x = 1 / (1 + math.exp(-x))
            else:
                # 对于负数，使用另一种形式避免溢出
                exp_x = math.exp(x)
                x = exp_x / (1 + exp_x)
            return x
    else:
        label_array = raw_data[LABEL_ITEM].to_numpy()
        label_mean, label_std = label_array.mean(), label_array.std()
        label_inverser = lambda x, mean=label_mean.item(), std=label_std.item(): x * std + mean
    
    # 转换数据集
    dataset = []
    test_dataset = []
    for source_data, target_dataset in [(raw_data, dataset), (test_data, test_dataset)]:
        ids = source_data[ID_ITEM].tolist()  # 样本 ID
        discretized_features = []  # 离散化特征
        continuous_features = []  # 连续特征
    
        for idx, feature_name in enumerate(sorted_columns):
            discretizer = feature_discretizers[idx]
            feature_values = source_data[feature_name]
            if isinstance(discretizer, tuple):
                lower_bound, value_range = discretizer
                # 将归一化值映射到离散分箱
                discretized_features.append((np.clip((feature_values.to_numpy() - lower_bound) / value_range, 0, 1) * (variance_bins - 1)).round().astype(np.int64))
            else:
                # 将类别映射为索引
                discretized_features.append([discretizer[category] for category in feature_values])
    
            # 对数值特征进行标准化
            if feature_name in feature_standardizers:
                mean, std = feature_standardizers[feature_name]
                continuous_features.append((feature_values.to_numpy() - mean) / std)
    
        # 处理标签
        if LABEL_ITEM in source_data:
            if LOSS_FUNCTION == "binary_cross_entropy":
                labels = source_data[LABEL_ITEM].tolist()
            else:
                # 回归任务，对标签进行标准化
                labels = (source_data[LABEL_ITEM].to_numpy() - label_mean) / label_std
        else:
           # 测试集没有标签，用NaN填充
            labels = [float("nan")] * len(ids)
    
        # 将处理后的数据组合并添加到目标数据集
        # 每个样本包含：ID、离散化特征、标准化连续特征、标签
        target_dataset.extend(zip(ids, zip(*discretized_features), zip(*continuous_features), labels))
    
    # 切割带标签的数据集，分为训练集和验证集
    split_point = int(len(dataset) * 0.9)
    train_dataset = dataset[:split_point]
    val_dataset = dataset[split_point:]
    return train_dataset, val_dataset, test_dataset, label_inverser, feature_bin_counts, len(feature_standardizers)



# 训练函数
import copy
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

# 获取设备
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, train_loader, val_loader, interaction_lr: float, continuous_lr: float, batch_size: int, weight_decay: float, task_id: str, writer: SummaryWriter) -> float:
    # 创建优化器、梯度缩放器
    optimizer = optim.AdamW(
        [
            {"params": model.continuous_embedding.parameters(), "lr": interaction_lr},
            {"params": model.feature_embeddings.parameters(), "lr": interaction_lr},
            {"params": model.interaction_blocks.parameters(), "lr": interaction_lr},
            {"params": model.continuous_projection.parameters(), "lr": continuous_lr},
        ],
         weight_decay=weight_decay
    )
    scaler = GradScaler(device, 1)

    # 准备损失函数
    if LOSS_FUNCTION == "mse":
        criterion = F.mse_loss
    elif LOSS_FUNCTION == "l1":
        criterion = F.l1_loss
    elif LOSS_FUNCTION == "binary_cross_entropy":
        def criterion(inputs: torch.Tensor, labels: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            kwargs["pos_weight"] = (labels <= 0.5).sum() / (labels > 0.5).sum()
            return F.binary_cross_entropy_with_logits(inputs, labels, *args, **kwargs)
    else:
        raise Exception(f"未知损失函数: {LOSS_FUNCTION}")

    # 训练过程
    current_steps = 0
    progress_bar = tqdm(desc=f"Train [{task_id}]", total=VAL_PER_STEP * NUM_VAL_CYCLES)
    best_state, best_score = None, -float("inf")
    while current_steps < NUM_VAL_CYCLES * VAL_PER_STEP:
        for batch in train_loader:
            if current_steps >= VAL_PER_STEP * NUM_VAL_CYCLES:
                break
    
            batch = [x.to(device) for x in batch[1:]]
            with autocast(device, dtype=torch.bfloat16):
                predictions = model(*batch[:-1])
                loss = [criterion(pred, batch[-1]) for pred in predictions]
            optimizer.zero_grad()
            scaler.scale(torch.stack(loss).mean()).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
    
            # 记录损失
            writer.add_scalars("Loss/Train", {task_id: loss[-1].item()}, global_step=current_steps)
    
            # 记录指标
            roc_score = roc_auc_score(batch[-1].cpu(), predictions[-1].detach().cpu())
            if LOSS_FUNCTION == "binary_cross_entropy":
                writer.add_scalars("AUC Score/Train", {task_id: roc_score}, global_step=current_steps)
    
            # 更新计数
            current_steps += 1
    
            # 每隔一定步数进行一次验证
            if current_steps % VAL_PER_STEP == 0:
                # 更新进度条
                if LOSS_FUNCTION == "binary_cross_entropy":
                    progress_bar.set_postfix(loss=loss[-1].item(), roc=roc_score.item())
                else:
                    progress_bar.set_postfix(loss=loss[-1].item())
                progress_bar.update(VAL_PER_STEP)
    
                # 跑验证集
                all_probs = []
                all_labels = []
                losses = []
                model.eval()
                with torch.inference_mode():
                    for batch in val_loader:
                        batch = [x.to(device) for x in batch[1:]]
                        with autocast(device, dtype=torch.float16):
                            prediction = model(*batch[:-1])[-1]
                            loss = criterion(prediction, batch[-1], reduction="none")
                        valid_mask = ~loss.isnan()
                        losses.extend(loss.masked_select(valid_mask).tolist())
                        if LOSS_FUNCTION == "binary_cross_entropy":
                            all_probs.append(F.sigmoid(prediction.masked_select(valid_mask)))
                            all_labels.append(batch[-1].masked_select(valid_mask) > 0.5)
                model.train()
    
                # 记录损失
                avg_loss = sum(losses) / len(losses)
                writer.add_scalars("Loss/Validate", {task_id: avg_loss}, global_step=current_steps)
    
                # 记录指标
                if LOSS_FUNCTION == "binary_cross_entropy":
                    all_probs = torch.cat(all_probs)
                    all_labels = torch.cat(all_labels)
                    score = roc_auc_score(all_labels.cpu(), all_probs.detach().cpu())
                    writer.add_scalars("AUC Score/Validate", {task_id: score}, global_step=current_steps)
                else:
                    score = -avg_loss

                # 记录最佳模型
                if score >= best_score:
                    best_state, best_score = copy.deepcopy(model.state_dict()), score

    model.load_state_dict(best_state)
    progress_bar.close()
    return best_score



# 网格搜索、写结果
import pathlib
from itertools import product
from typing import Any
import torch
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

def list_grid(config: dict[str, list[Any]]):
    return [dict(zip(config, values)) for values in product(*config.values())]

writer = SummaryWriter("/kaggle/working/logdir")
best_model, best_score = None, -float("inf")
for config_grid in list_grid(CONFIG):
    print(config_grid)
    # 数据预处理
    train_dataset, val_dataset, test_dataset, label_inverser, feature_bin_counts, feature_standardizers = prepare_dataset(config_grid["variance_bins"])
    task_id = ",".join(
        f"{name}={value}"
        for name, value in config_grid.items()
        if len(CONFIG[name]) > 1
    )

    # 创建模型
    model = FeatureInteractionPredictor(
        feature_bin_counts,
        feature_standardizers,
        config_grid["dim_model"], config_grid["dim_feedforward"], config_grid["num_layers_per_block"], config_grid["num_interaction_blocks"], config_grid["dropout"]
    ).to(device)

    # 创造训练、验证、测试加载器
    train_loader, val_loader, test_loader = [
        DataLoader(FeatureInteractionDataset(dataset), batch_size=batch, collate_fn=collate_fn)
        for dataset, batch in [
            (train_dataset, config_grid["batch_size"]),
            (val_dataset, config_grid["batch_size"] * 2),
            (test_dataset, config_grid["batch_size"] * 2)
        ]
    ]

    # 训练
    model.train()
    score = train(model, train_loader, val_loader, config_grid["interaction_lr"], config_grid["continuous_lr"], config_grid["batch_size"], config_grid["weight_decay"], task_id, writer)
    print(f"{task_id}: {score=}")

    # 保留最佳模型
    if score >= best_score:
        best_model, best_score = model, score
print(f"最佳分数: {best_score=}")
writer.close()

# 预测结果
model.eval()
pathlib.Path(SUBMISSION_PATH).parent.mkdir(exist_ok=True, parents=True)
with open(SUBMISSION_PATH, "w") as f:
    f.write(f"{ID_ITEM},{LABEL_ITEM}\n")
    with torch.inference_mode():
        for batch in tqdm(test_loader):
            data_indices = batch[0].tolist()
            batch = [x.to(device) for x in batch[1:-1]]
            prediction = model(*batch)[-1]
            for data_idx, output in zip(data_indices, prediction.tolist()):
                f.write(f"{data_idx},{label_inverser(output)}\n")



# 收尾工作，压缩 logdir
!tar -caf /kaggle/working/tensorboard_log.tar.gz -C /kaggle/working/ logdir

