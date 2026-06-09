import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 固定随机种子（保证结果可复现）
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 训练配置（优化参数，避免资源不足）
CONFIG = {
    'batch_size': 8,  # 适当增大批次（根据GPU内存调整）
    'epochs': 5,      # 先减少epoch数快速验证
    'learning_rate': 1e-4,
    'image_size': (128, 128),  
    'num_classes': 14,  # 更新为14个标签
    'train_split': 0.8,
    'max_slices': 8,   # 从16→8（减少每个序列的切片数）
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 14个目标标签列名
TARGET_LABELS = [
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
    'Aneurysm Present'  # 总存在标签（权重13）
]

# 核心数据路径（根据Kaggle目录结构）
BASE_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection'
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
SERIES_PATH = os.path.join(BASE_PATH, 'series')

# 缓存路径（用于存储预处理后的图像，避免重复解析DICOM）
CACHE_PATH = '/kaggle/working/aneurysm_cache'
os.makedirs(CACHE_PATH, exist_ok=True)  # 自动创建缓存文件夹

# 加载并预处理训练数据（多标签版本）
train_df = pd.read_csv(TRAIN_CSV_PATH)

# 按序列ID分组，计算每个序列的14个标签（取最大值：存在则为1）
series_labels_df = train_df.groupby('SeriesInstanceUID')[TARGET_LABELS].max().reset_index()
series_labels_df.rename(columns={'SeriesInstanceUID': 'series_id'}, inplace=True)

# 验证标签分布
print("14个标签的序列级正样本比例：")
label_pos_ratio = (series_labels_df[TARGET_LABELS].sum() / len(series_labels_df)).round(4)
for label, ratio in label_pos_ratio.items():
    print(f"{label}: {ratio:.2%}")

# 划分训练集/验证集（按总存在标签分层）
train_series, val_series = train_test_split(
    series_labels_df['series_id'].values,
    test_size=1 - CONFIG['train_split'],
    random_state=SEED,
    stratify=series_labels_df['Aneurysm Present']
)

# 分离训练/验证集的标签（14维）
train_labels = series_labels_df[series_labels_df['series_id'].isin(train_series)].reset_index(drop=True)
val_labels = series_labels_df[series_labels_df['series_id'].isin(val_series)].reset_index(drop=True)

print(f"\n训练集序列数：{len(train_labels)}，标签维度：{train_labels[TARGET_LABELS].shape}")
print(f"验证集序列数：{len(val_labels)}，标签维度：{val_labels[TARGET_LABELS].shape}")

# DICOM加载和缓存函数
def load_dicom(path):
    """修复DICOM加载：用dcmread替代read_file，添加异常处理"""
    try:
        dicom = pydicom.dcmread(path)
        img = apply_voi_lut(dicom.pixel_array, dicom)

        # 确保单通道（部分DICOM可能多通道，取第一通道）
        if len(img.shape) != 2:
            img = img[:, :, 0]

        # 归一化到[0,255]（避免数值范围异常）
        img = img - img.min()
        if img.max() != 0:  # 避免除以零
            img = img / img.max()
        img = (img * 255).astype(np.uint8)
        return img
    except Exception as e:
        print(f"加载DICOM失败 {path}: {str(e)}")
        return None


def get_cached_series(series_id):
    """缓存机制：第一次加载DICOM并保存为.npy，后续直接加载缓存"""
    # 缓存文件路径（每个序列对应一个缓存文件）
    cache_file = os.path.join(CACHE_PATH, f"{series_id}.npy")

    # 1. 有缓存：直接加载
    if os.path.exists(cache_file):
        return np.load(cache_file)

    # 2. 无缓存：加载DICOM并生成缓存
    series_folder = os.path.join(SERIES_PATH, str(series_id))
    if not os.path.exists(series_folder):
        print(f"序列文件夹不存在: {series_folder}")
        return None

    # 获取DICOM文件（按名称排序，确保切片顺序正确）
    dicom_files = [os.path.join(series_folder, f) for f in os.listdir(series_folder) if f.endswith('.dcm')]
    dicom_files.sort()  # 关键：医学图像切片必须按顺序加载

    # 加载DICOM（只取前CONFIG['max_slices']个）
    images = []
    for i, file in enumerate(dicom_files):
        if i >= CONFIG['max_slices']:  # 超过max_slices直接停止
            break
        img = load_dicom(file)
        if img is not None:
            images.append(img)

    # 保存缓存
    if len(images) > 0:
        np.save(cache_file, np.array(images))
    return np.array(images) if len(images) > 0 else None

# 测试缓存加载效果
sample_series_id = train_series[0]
print(f"\n测试序列 {sample_series_id} 加载...")
sample_images = get_cached_series(sample_series_id)
if sample_images is not None:
    print(f"序列 {sample_series_id} 加载成功，共 {len(sample_images)} 张切片")
    print(f"切片形状: {sample_images[0].shape}")

    # 可视化前3张切片
    plt.figure(figsize=(12, 4))
    for i in range(min(3, len(sample_images))):
        plt.subplot(1, 3, i + 1)
        plt.imshow(sample_images[i], cmap='gray')
        plt.title(f"切片 {i + 1}")
        plt.axis('off')
    plt.show()
else:
    print(f"序列 {sample_series_id} 加载失败")

# 多标签数据集类
class AneurysmMultiLabelDataset(Dataset):
    def __init__(self, series_labels_df, transform=None):
        self.series_labels_df = series_labels_df  # 包含series_id和14个标签
        self.series_ids = series_labels_df['series_id'].values
        self.transform = transform
        self.max_slices = CONFIG['max_slices']
        self.target_labels = TARGET_LABELS  # 14个标签

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        # 1. 加载序列图像（复用缓存机制）
        series_id = self.series_ids[idx]
        images = get_cached_series(series_id)

        # 处理加载失败的情况（返回随机图像+全0标签）
        if images is None or len(images) == 0:
            img_tensor = torch.randn(1, self.max_slices, *CONFIG['image_size'])
            label_tensor = torch.zeros(len(self.target_labels), dtype=torch.float32)
            return img_tensor.float(), label_tensor

        # 2. 调整切片数量（补零/截断）
        if len(images) < self.max_slices:
            pad_width = self.max_slices - len(images)
            images = np.pad(images, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
        else:
            images = images[:self.max_slices]

        # 3. 图像变换
        processed_images = []
        for img in images:
            if self.transform:
                img = self.transform(img)  # [1, H, W]
            processed_images.append(img)

        # 4. 调整维度：[max_slices, 1, H, W] → [1, max_slices, H, W]
        img_tensor = torch.stack(processed_images).permute(1, 0, 2, 3)

        # 5. 获取14维标签
        label_row = self.series_labels_df[self.series_labels_df['series_id'] == series_id]
        label_tensor = torch.tensor(
            label_row[self.target_labels].values[0], 
            dtype=torch.float32
        )

        return img_tensor.float(), label_tensor

# 定义图像变换
transform = transforms.Compose([
    transforms.ToPILImage(),  # 转为PIL图像（便于Resize）
    transforms.Resize(CONFIG['image_size']),  # 统一尺寸
    transforms.ToTensor(),  # 转为Tensor（范围[0,1]）
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
])

# 创建多标签数据集和DataLoader
train_dataset = AneurysmMultiLabelDataset(train_labels, transform=transform)
val_dataset = AneurysmMultiLabelDataset(val_labels, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=0,
    pin_memory=CONFIG['device'] == 'cuda'
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=0,
    pin_memory=CONFIG['device'] == 'cuda'
)

print(f"\n多标签数据集初始化完成：")
print(f"训练集：{len(train_dataset)}个样本，{len(train_loader)}个批次")
print(f"验证集：{len(val_dataset)}个样本，{len(val_loader)}个批次")
print(f"标签维度：{len(TARGET_LABELS)}（13个部位 + 1个总存在）")

# 模型定义
class PatchEmbedding(nn.Module):
    """将单张医学图像切片转为Transformer输入的Patch嵌入"""
    def __init__(self, image_size, patch_size, in_channels=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        # 计算每个切片的Patch数量
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # 用卷积层实现Patch分割+嵌入
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [batch_size, 1, H, W] → 单通道图像
        x = self.proj(x)  # [batch_size, embed_dim, num_patches_H, num_patches_W]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x


class SliceTransformer(nn.Module):
    """处理单张切片的Transformer（提取切片级特征）"""
    def __init__(self, image_size=CONFIG['image_size'], patch_size=32, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 切片特征聚合
        self.feature_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [batch_size, 1, H, W] → 单张切片
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        x = x + self.pos_embed  # 添加位置嵌入

        x = self.transformer(x)  # [batch_size, num_patches, embed_dim]
        x = x.transpose(1, 2)  # [batch_size, embed_dim, num_patches]
        x = self.feature_pool(x).squeeze(-1)  # [batch_size, embed_dim] → 切片级特征
        return x


class SeriesTransformerMultiLabel(nn.Module):
    """适配多标签分类的序列Transformer模型"""
    def __init__(self, max_slices=CONFIG['max_slices'], slice_embed_dim=256, seq_embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        # 1. 切片级特征提取
        self.slice_transformer = SliceTransformer(embed_dim=slice_embed_dim)

        # 2. 序列级特征投影
        self.seq_proj = nn.Linear(slice_embed_dim, seq_embed_dim)
        self.seq_pos_embed = nn.Parameter(torch.randn(1, max_slices, seq_embed_dim))

        # 3. 序列级Transformer
        seq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=seq_embed_dim,
            nhead=num_heads,
            dim_feedforward=seq_embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.seq_transformer = nn.TransformerEncoder(seq_encoder_layer, num_layers=num_layers)

        # 4. 多标签分类头
        self.classifier = nn.Sequential(
            nn.Linear(seq_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, len(TARGET_LABELS)),  # 输出14个概率
            nn.Sigmoid()  # 每个标签独立二分类，用Sigmoid激活
        )

    def forward(self, x):
        # x: [batch_size, 1, max_slices, H, W]
        batch_size, _, max_slices, H, W = x.shape

        # 1. 切片级特征提取
        slice_features = []
        for i in range(max_slices):
            slice_img = x[:, :, i, :, :]  # [batch_size, 1, H, W]
            feat = self.slice_transformer(slice_img)  # [batch_size, slice_embed_dim]
            slice_features.append(feat)

        # 2. 序列级特征聚合
        seq_feat = torch.stack(slice_features, dim=1)  # [batch_size, max_slices, slice_embed_dim]
        seq_feat = self.seq_proj(seq_feat)  # [batch_size, max_slices, seq_embed_dim]
        seq_feat = seq_feat + self.seq_pos_embed
        seq_feat = self.seq_transformer(seq_feat)  # [batch_size, max_slices, seq_embed_dim]
        seq_feat = seq_feat.mean(dim=1)  # [batch_size, seq_embed_dim]

        # 3. 多标签预测
        output = self.classifier(seq_feat)  # [batch_size, 14]
        return output

# 计算模型参数量的函数
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 创建多标签模型
model = SeriesTransformerMultiLabel()
model = model.to(CONFIG['device'])

print(f"\n模型可训练参数量: {count_params(model):,}")
print(f"模型设备: {next(model.parameters()).device}")

# 定义损失函数、优化器和学习率调度器
criterion = nn.BCELoss()  # 多标签二分类交叉熵损失
optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=1e-5  # L2正则化
)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # 目标：最小化验证损失
    patience=2,  # 2个epoch损失不下降则降学习率
    factor=0.5,  # 学习率 *= 0.5
    verbose=True
)

# 训练和评估函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch，返回平均损失"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    loop = tqdm(dataloader, desc=f"训练中", leave=True)
    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        # 更新进度条
        loop.set_postfix({
            'batch_loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/total_samples:.4f}"
        })

    epoch_loss = total_loss / total_samples
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """评估模型，返回平均损失"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"评估中", leave=True)
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            loop.set_postfix({
                'batch_loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/total_samples:.4f}"
            })

    epoch_loss = total_loss / total_samples
    return epoch_loss

# 多标签AUC计算与加权得分函数
def calculate_multilabel_auc(model, dataloader, device, target_labels):
    """计算多标签分类的AUC ROC和加权最终得分"""
    model.eval()
    all_y_true = []  # 存储所有样本的真实标签（shape: [n_samples, 14]）
    all_y_pred = []  # 存储所有样本的预测概率（shape: [n_samples, 14]）

    with torch.no_grad():
        loop = tqdm(dataloader, desc="计算多标签AUC", leave=True)
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向传播获取14维预测概率
            outputs = model(images)  # [batch_size, 14]

            # 收集结果
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(outputs.cpu().numpy())

    # 转为numpy数组
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # 1. 计算每个标签的AUC
    label_aucs = []
    for i, label in enumerate(target_labels):
        y_true = all_y_true[:, i]
        y_pred = all_y_pred[:, i]

        # 处理单类别情况
        if len(np.unique(y_true)) < 2:
            auc = 0.5  # 随机猜测，AUC设为0.5
        else:
            auc = roc_auc_score(y_true, y_pred)
        
        label_aucs.append(auc)
        print(f"{label:<40} AUC: {auc:.4f}")

    # 2. 计算加权最终得分
    present_idx = target_labels.index('Aneurysm Present')
    present_auc = label_aucs[present_idx]
    other_aucs = [auc for i, auc in enumerate(label_aucs) if i != present_idx]

    # 最终得分 = (13*present_auc + sum(other_aucs)) / 26
    weighted_final_score = (13 * present_auc + sum(other_aucs)) / (13 + 13)
    print(f"\n{'='*50}")
    print(f"Aneurysm Present AUC（权重13）: {present_auc:.4f}")
    print(f"其他13个标签平均AUC（权重1）: {np.mean(other_aucs):.4f}")
    print(f"加权最终得分: {weighted_final_score:.4f}")
    print(f"{'='*50}")

    return label_aucs, weighted_final_score, all_y_true, all_y_pred

# AUC可视化函数
def plot_multilabel_auc(label_aucs, target_labels, save_path):
    """绘制多标签AUC条形图"""
    plt.figure(figsize=(16, 8))
    
    # 定义颜色：Aneurysm Present用红色，其他用蓝色
    colors = ['#1f77b4'] * len(target_labels)
    present_idx = target_labels.index('Aneurysm Present')
    colors[present_idx] = '#d62728'  # 红色标记总存在标签

    # 绘制条形图
    bars = plt.barh(
        y=range(len(target_labels)), 
        width=label_aucs, 
        color=colors, 
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    # 添加数值标签
    for i, (bar, auc) in enumerate(zip(bars, label_aucs)):
        plt.text(
            auc + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{auc:.4f}', 
            va='center', 
            fontsize=9
        )

    # 设置坐标轴
    plt.yticks(range(len(target_labels)), target_labels, fontsize=10)
    plt.xlabel('ROC AUC Score', fontsize=12, fontweight='bold')
    plt.title('14 Target Labels ROC AUC Performance\n(Red = Aneurysm Present, Weight=13)', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加水平参考线（AUC=0.5）
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Guess (AUC=0.5)')
    plt.legend(loc='lower right')

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"AUC可视化图已保存至：{save_path}")

# 初始化训练历史
history = {
    'train_loss': [],
    'val_loss': []
}

# 最佳模型保存配置
best_model_path = '/kaggle/working/best_aneurysm_model.pth'
best_val_loss = float('inf')

# 开始训练循环
print(f"\n{'='*50}")
print(f"开始训练（共 {CONFIG['epochs']} 个Epoch）")
print(f"{'='*50}")

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
    print(f"-"*30)

    # 1. 训练一个Epoch
    train_loss = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
    print(f"训练结果 → 损失: {train_loss:.4f}")

    # 2. 验证一个Epoch
    val_loss = evaluate(model, val_loader, criterion, CONFIG['device'])
    print(f"验证结果 → 损失: {val_loss:.4f}")

    # 3. 更新学习率调度器
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    # 4. 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ 保存最佳模型（验证损失: {best_val_loss:.4f}）")

# 训练完成后计算多标签AUC和最终得分
print(f"\n{'='*50}")
print(f"训练完成！加载最佳模型计算AUC...")
print(f"{'='*50}")

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))

# 计算验证集的多标签AUC与最终得分
val_label_aucs, val_final_score, val_y_true, val_y_pred = calculate_multilabel_auc(
    model=model,
    dataloader=val_loader,
    device=CONFIG['device'],
    target_labels=TARGET_LABELS
)

# 绘制并保存AUC结果
auc_plot_path = '/kaggle/working/multilabel_auc_plot.png'
plot_multilabel_auc(
    label_aucs=val_label_aucs,
    target_labels=TARGET_LABELS,
    save_path=auc_plot_path
)

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='训练损失', linewidth=2, marker='o', markersize=4)
plt.plot(history['val_loss'], label='验证损失', linewidth=2, marker='s', markersize=4)
plt.title('训练与验证损失曲线', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('损失值', fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 固定随机种子（保证结果可复现）
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# 训练配置（优化参数，避免资源不足）
CONFIG = {
    'batch_size': 8,  # 适当增大批次（根据GPU内存调整）
    'epochs': 5,      # 先减少epoch数快速验证
    'learning_rate': 1e-4,
    'image_size': (128, 128),  
    'num_classes': 2,
    'train_split': 0.8,
    'max_slices': 8,   # 从16→8（减少每个序列的切片数）
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 核心数据路径（根据Kaggle目录结构）
BASE_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection'
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
SERIES_PATH = os.path.join(BASE_PATH, 'series')

# 新增：缓存路径（用于存储预处理后的图像，避免重复解析DICOM）
CACHE_PATH = '/kaggle/working/aneurysm_cache'
os.makedirs(CACHE_PATH, exist_ok=True)  # 自动创建缓存文件夹


# 加载并预处理训练数据（修改列名相关逻辑）
train_df = pd.read_csv(TRAIN_CSV_PATH)

# 定义真实列名
SERIES_ID_COL = 'SeriesInstanceUID'
ANEURYSM_COL = 'Aneurysm Present'

# 将 Aneurysm Present 列的概率转换为 0/1 标签
train_df['aneurysm_label'] = (train_df[ANEURYSM_COL] > 0.5).astype(int)

# 按序列ID分组，获取每个序列的动脉瘤标签
series_aneurysm = train_df.groupby(SERIES_ID_COL)['aneurysm_label'].max().reset_index()
series_aneurysm.rename(columns={SERIES_ID_COL:'series_id', 'aneurysm_label': 'aneurysm'}, inplace=True)

# 划分训练集/验证集
train_series, val_series = train_test_split(
    series_aneurysm['series_id'].values,
    test_size=1 - CONFIG['train_split'],
    random_state=SEED,
    stratify=series_aneurysm['aneurysm']
)

print(f"训练集序列数: {len(train_series)}")
print(f"验证集序列数: {len(val_series)}")
print(f"训练集正负样本比例: \n{series_aneurysm[series_aneurysm['series_id'].isin(train_series)]['aneurysm'].value_counts(normalize=True)}")
print(f"\n修改后的series_aneurysm前5行：")
print(series_aneurysm[['series_id', 'aneurysm']].head()) 


def load_dicom(path):
    """修复DICOM加载：用dcmread替代read_file，添加异常处理"""
    try:
        dicom = pydicom.dcmread(path)
        img = apply_voi_lut(dicom.pixel_array, dicom)

        # 确保单通道（部分DICOM可能多通道，取第一通道）
        if len(img.shape) != 2:
            img = img[:, :, 0]

        # 归一化到[0,255]（避免数值范围异常）
        img = img - img.min()
        if img.max() != 0:  # 避免除以零
            img = img / img.max()
        img = (img * 255).astype(np.uint8)
        return img
    except Exception as e:
        print(f"加载DICOM失败 {path}: {str(e)}")
        return None


def get_cached_series(series_id):
    """新增缓存机制：第一次加载DICOM并保存为.npy，后续直接加载缓存"""
    # 缓存文件路径（每个序列对应一个缓存文件）
    cache_file = os.path.join(CACHE_PATH, f"{series_id}.npy")

    # 1. 有缓存：直接加载（速度提升10倍以上）
    if os.path.exists(cache_file):
        return np.load(cache_file)

    # 2. 无缓存：加载DICOM并生成缓存
    series_folder = os.path.join(SERIES_PATH, str(series_id))
    if not os.path.exists(series_folder):
        print(f"序列文件夹不存在: {series_folder}")
        return None

    # 获取DICOM文件（按名称排序，确保切片顺序正确）
    dicom_files = [os.path.join(series_folder, f) for f in os.listdir(series_folder) if f.endswith('.dcm')]
    dicom_files.sort()  # 关键：医学图像切片必须按顺序加载

    # 加载DICOM（只取前CONFIG['max_slices']个，减少加载量）
    images = []
    for i, file in enumerate(dicom_files):
        if i >= CONFIG['max_slices']:  # 优化：超过max_slices直接停止，避免冗余加载
            break
        img = load_dicom(file)
        if img is not None:
            images.append(img)

    # 保存缓存（后续训练直接用）
    if len(images) > 0:
        np.save(cache_file, np.array(images))
    return np.array(images) if len(images) > 0 else None

# 测试缓存加载效果
sample_series_id = train_series[0]
print(f"\n测试序列 {sample_series_id} 加载...")
sample_images = get_cached_series(sample_series_id)
if sample_images is not None:
    print(f"序列 {sample_series_id} 加载成功，共 {len(sample_images)} 张切片")
    print(f"切片形状: {sample_images[0].shape}")

    # 可视化前3张切片
    plt.figure(figsize=(12, 4))
    for i in range(min(3, len(sample_images))):
        plt.subplot(1, 3, i + 1)
        plt.imshow(sample_images[i], cmap='gray')
        plt.title(f"切片 {i + 1}")
        plt.axis('off')
    plt.show()
else:
    print(f"序列 {sample_series_id} 加载失败")


class AneurysmDataset(Dataset):
    def __init__(self, series_ids, series_aneurysm, transform=None):
        self.series_ids = series_ids
        self.series_aneurysm = series_aneurysm
        self.transform = transform
        self.max_slices = CONFIG['max_slices']

    def __len__(self):
        return len(self.series_ids)

    def __getitem__(self, idx):
        series_id = self.series_ids[idx]
        images = get_cached_series(series_id)  # 从缓存加载序列

        # 处理加载失败的情况
        if images is None or len(images) == 0:
            # 返回符合预期维度的随机张量：[1, max_slices, H, W]
            return torch.randn(1, self.max_slices, *CONFIG['image_size']), torch.tensor(0, dtype=torch.float32)

        # 步骤1：调整切片数量（补零或截断）
        if len(images) < self.max_slices:
            pad_width = self.max_slices - len(images)
            images = np.pad(images, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
        else:
            images = images[:self.max_slices]

        # 步骤2：应用图像变换（ToTensor()会自动添加通道维度[1, H, W]）
        processed_images = []
        for img in images:
            if self.transform:
                img = self.transform(img)  # 输出：[1, H, W]（单通道）
            processed_images.append(img)

        # 步骤3：堆叠并调整维度顺序（关键修改！解决维度不匹配）
        # 堆叠后维度：[max_slices, 1, H, W] → 调整为 [1, max_slices, H, W]（通道在前，切片数在后）
        images_tensor = torch.stack(processed_images)  # [max_slices, 1, H, W]
        images_tensor = images_tensor.permute(1, 0, 2, 3)  # 交换维度1和0 → [1, max_slices, H, W]

        # 步骤4：获取标签（依赖数据预处理中统一的'series_id'和'aneurysm'列）
        label = self.series_aneurysm.loc[
            self.series_aneurysm['series_id'] == series_id,
            'aneurysm'
        ].values[0]

        # 最终输出：图像[1, max_slices, H, W] + 标签[0/1]（匹配模型输入维度）
        return images_tensor.float(), torch.tensor(label, dtype=torch.float32)

# 定义图像变换
transform = transforms.Compose([
    transforms.ToPILImage(),  # 转为PIL图像（便于Resize）
    transforms.Resize(CONFIG['image_size']),  # 统一尺寸
    transforms.ToTensor(),  # 转为Tensor（范围[0,1]）
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]（适配Transformer）
])

# 创建数据集和DataLoader
train_dataset = AneurysmDataset(train_series, series_aneurysm, transform=transform)
val_dataset = AneurysmDataset(val_series, series_aneurysm, transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=0,
    pin_memory=CONFIG['device'] == 'cuda'
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=0,
    pin_memory=CONFIG['device'] == 'cuda'
)

print(f"\n训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")
print(f"训练集批次数量: {len(train_loader)}")
print(f"验证集批次数量: {len(val_loader)}")


class PatchEmbedding(nn.Module):
    """将单张医学图像切片转为Transformer输入的Patch嵌入"""
    def __init__(self, image_size, patch_size, in_channels=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        # 计算每个切片的Patch数量（如256x256图像，32x32 Patch → 8x8=64个Patch）
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # 用卷积层实现Patch分割+嵌入（替代手动分割）
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [batch_size, 1, H, W] → 单通道图像
        x = self.proj(x)  # [batch_size, embed_dim, num_patches_H, num_patches_W]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim] → 适配Transformer输入
        return x


class SliceTransformer(nn.Module):
    """处理单张切片的Transformer（提取切片级特征）"""
    def __init__(self, image_size=CONFIG['image_size'], patch_size=32, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 位置嵌入（Transformer需要位置信息）
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Transformer编码器（简化结构，避免过拟合）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 切片特征聚合（平均池化，简化计算）
        self.feature_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [batch_size, 1, H, W] → 单张切片
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        x = x + self.pos_embed  # 添加位置嵌入

        x = self.transformer(x)  # [batch_size, num_patches, embed_dim]
        x = x.transpose(1, 2)  # [batch_size, embed_dim, num_patches]
        x = self.feature_pool(x).squeeze(-1)  # [batch_size, embed_dim] → 切片级特征
        return x


class SeriesTransformer(nn.Module):
    """处理整个序列的Transformer（聚合切片特征，判断是否有动脉瘤）"""
    def __init__(self, max_slices=CONFIG['max_slices'], slice_embed_dim=256, seq_embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        # 1. 切片级特征提取
        self.slice_transformer = SliceTransformer(embed_dim=slice_embed_dim)

        # 2. 序列级特征投影（将切片特征转为序列嵌入维度）
        self.seq_proj = nn.Linear(slice_embed_dim, seq_embed_dim)

        # 3. 序列位置嵌入
        self.seq_pos_embed = nn.Parameter(torch.randn(1, max_slices, seq_embed_dim))

        # 4. 序列级Transformer（聚合切片间信息）
        seq_encoder_layer = nn.TransformerEncoderLayer(
            d_model=seq_embed_dim,
            nhead=num_heads,
            dim_feedforward=seq_embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.seq_transformer = nn.TransformerEncoder(seq_encoder_layer, num_layers=num_layers)

        # 5. 最终分类头（二分类：有/无动脉瘤）
        self.classifier = nn.Sequential(
            nn.Linear(seq_embed_dim, 64),  # 降维
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.5),  # 正则化
            nn.Linear(64, 1),  # 输出1个值
            nn.Sigmoid()  # 转为概率（0~1）
        )

    def forward(self, x):
        # x: [batch_size, 1, max_slices, H, W] → 序列数据
        batch_size, _, max_slices, H, W = x.shape

        # 1. 提取每个切片的特征
        slice_features = []
        for i in range(max_slices):
            slice_img = x[:, :, i, :, :]  # [batch_size, 1, H, W] → 单张切片
            feat = self.slice_transformer(slice_img)  # [batch_size, slice_embed_dim]
            slice_features.append(feat)

        # 2. 拼接切片特征，形成序列
        seq_feat = torch.stack(slice_features, dim=1)  # [batch_size, max_slices, slice_embed_dim]
        seq_feat = self.seq_proj(seq_feat)  # [batch_size, max_slices, seq_embed_dim]
        seq_feat = seq_feat + self.seq_pos_embed  # 添加序列位置嵌入

        # 3. 序列级特征聚合
        seq_feat = self.seq_transformer(seq_feat)  # [batch_size, max_slices, seq_embed_dim]
        seq_feat = seq_feat.mean(dim=1)  # [batch_size, seq_embed_dim] → 平均池化聚合序列信息

        # 4. 分类预测
        output = self.classifier(seq_feat).squeeze(-1)  # [batch_size] → 每个样本的概率
        return output


# 创建模型并移至设备
model = SeriesTransformer()
model = model.to(CONFIG['device'])

# 打印模型参数量（避免模型过大导致GPU内存不足）
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"\n模型可训练参数量: {count_params(model):,}")
print(f"模型设备: {next(model.parameters()).device}")


# 定义损失函数、优化器和学习率调度器
criterion = nn.BCELoss()  # 二分类交叉熵损失（适配Sigmoid输出）
optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=1e-5  # L2正则化，减少过拟合
)
# 学习率调度器：验证损失不下降时降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # 目标：最小化验证损失
    patience=2,  # 2个epoch损失不下降则降学习率
    factor=0.5,  # 学习率 *= 0.5
    verbose=True  # 打印学习率变化
)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch，返回平均损失和准确率"""
    model.train()  # 开启训练模式（启用Dropout等）
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # tqdm进度条（显示实时训练状态）
    loop = tqdm(dataloader, desc=f"训练中", leave=True)
    for batch_idx, (images, labels) in enumerate(loop):
        # 数据移至设备（GPU/CPU）
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 1. 前向传播
        outputs = model(images)  # [batch_size] → 预测概率
        loss = criterion(outputs, labels)  # 计算损失

        # 2. 反向传播与参数更新
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        # 3. 计算指标
        total_loss += loss.item() * images.size(0)  # 累计损失（乘以批次大小，避免批次不均衡）
        predictions = (outputs > 0.5).float()  # 概率>0.5视为正样本
        total_correct += (predictions == labels).sum().item()  # 累计正确数
        total_samples += images.size(0)  # 累计样本数

        # 更新进度条显示
        loop.set_postfix({
            'batch_loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/total_samples:.4f}",
            'acc': f"{total_correct/total_samples:.4f}"
        })

    # 计算epoch级指标
    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """评估模型，返回平均损失和准确率（无梯度计算）"""
    model.eval()  # 开启评估模式（关闭Dropout等）
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 禁用梯度计算（加速评估，避免内存占用）
    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"评估中", leave=True)
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 计算指标
            total_loss += loss.item() * images.size(0)
            predictions = (outputs > 0.5).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

            # 更新进度条
            loop.set_postfix({
                'batch_loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/total_samples:.4f}",
                'acc': f"{total_correct/total_samples:.4f}"
            })

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc


# 初始化训练历史（记录损失和准确率）
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# 最佳模型保存配置（保存验证损失最小的模型）
best_model_path = '/kaggle/working/best_aneurysm_model.pth'
best_val_loss = float('inf')  # 初始值设为无穷大

# 开始训练循环
print(f"\n{'='*50}")
print(f"开始训练（共 {CONFIG['epochs']} 个Epoch）")
print(f"{'='*50}")

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
    print(f"-"*30)

    # 1. 训练一个Epoch
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
    print(f"训练结果 → 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")

    # 2. 验证一个Epoch
    val_loss, val_acc = evaluate(model, val_loader, criterion, CONFIG['device'])
    print(f"验证结果 → 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")

    # 3. 更新学习率调度器
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 5. 保存最佳模型（仅当验证损失下降时）
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ 保存最佳模型（验证损失: {best_val_loss:.4f}）")


# 训练完成后可视化训练曲线
print(f"\n{'='*50}")
print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
print(f"最佳模型路径: {best_model_path}")
print(f"{'='*50}")

# 绘制损失曲线和准确率曲线
plt.figure(figsize=(14, 5))

# 1. 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='训练损失', linewidth=2, marker='o', markersize=4)
plt.plot(history['val_loss'], label='验证损失', linewidth=2, marker='s', markersize=4)
plt.title('训练与验证损失曲线', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('损失值', fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# 2. 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='训练准确率', linewidth=2, marker='o', markersize=4)
plt.plot(history['val_acc'], label='验证准确率', linewidth=2, marker='s', markersize=4)
plt.title('训练与验证准确率曲线', fontsize=12)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('准确率', fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# 保存图像（Kaggle中可在"Output"查看）
plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

